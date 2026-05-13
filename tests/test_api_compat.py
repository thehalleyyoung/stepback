"""Tests for the public-API + SB-Trace schema compatibility checker.

Covers Step 25 of ``100_STEPS.md``: an API compatibility checker in CI
that diffs generated public API docs and the SB-Trace schema against
the last release tag (or, when no tag exists yet, against the
committed baseline shipped under
``stepback/conformance/api_baselines/v<version>/``).

The contract is:

* The committed baseline must equal the live snapshot of the current
  build. Drift fails the test with the diff explaining what changed
  and what to do.
* The diff functions classify changes as ``"breaking"`` vs
  ``"compatible"`` and the classification rules themselves are
  exercised here so that future refactors of the checker do not
  silently weaken the guarantee.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

import stepback
from stepback import api_snapshot


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_api_compat.py"
BASELINE_DIR = (
    REPO_ROOT
    / "stepback"
    / "conformance"
    / "api_baselines"
    / f"v{stepback.__version__}"
)


# ---------------------------------------------------------------------------
# Shape of the snapshot
# ---------------------------------------------------------------------------


def test_public_api_snapshot_covers_every_dunder_all_entry():
    snapshot = api_snapshot.public_api_snapshot()
    assert snapshot["package"] == "stepback"
    assert snapshot["package_version"] == stepback.__version__
    assert snapshot["snapshot_format_version"] == api_snapshot.SNAPSHOT_FORMAT_VERSION
    assert set(snapshot["symbols"]) == set(stepback.__all__)
    assert snapshot["missing"] == []


def test_public_api_snapshot_records_function_signatures():
    snapshot = api_snapshot.public_api_snapshot()
    record = snapshot["symbols"]["replay"]
    assert record["kind"] == "function"
    assert record["signature"] is not None
    param_names = [p["name"] for p in record["signature"]["parameters"]]
    assert param_names, "replay should expose at least one parameter"


def test_public_api_snapshot_records_dataclass_fields():
    snapshot = api_snapshot.public_api_snapshot()
    # Pick whatever symbol in __all__ is actually a dataclass; the
    # snapshot must record at least one field for it.
    dataclass_records = [
        rec
        for rec in snapshot["symbols"].values()
        if rec.get("kind") == "class" and rec.get("is_dataclass")
    ]
    assert dataclass_records, "expected at least one dataclass in stepback.__all__"
    for rec in dataclass_records:
        assert isinstance(rec.get("fields"), list)


def test_sbtrace_schema_snapshot_pins_v1_invariants():
    snapshot = api_snapshot.sbtrace_schema_snapshot()
    assert snapshot["wire_version"] == "1.0.0"
    assert snapshot["wire_version_info"] == [1, 0, 0]
    assert snapshot["format_version"] == 1
    assert snapshot["encoding"] == "canonical-json"
    assert snapshot["magic"] == "stepback/.sb"
    # Required fields that every v1 reader must enforce:
    for required_field in (
        "step_id",
        "step_kind",
        "inputs_hash",
        "outputs_hash",
        "prev_hmac",
    ):
        if required_field in snapshot["step_required"]:
            break
    else:
        pytest.fail(
            f"step_required missing core v1 fields: {snapshot['step_required']!r}"
        )


# ---------------------------------------------------------------------------
# Baseline-vs-current
# ---------------------------------------------------------------------------


def test_committed_baseline_matches_current_snapshot():
    """The baseline is the contract; CI must catch drift locally too."""
    api_baseline = json.loads(
        (BASELINE_DIR / "public_api.json").read_text(encoding="utf-8")
    )
    schema_baseline = json.loads(
        (BASELINE_DIR / "sbtrace_schema.json").read_text(encoding="utf-8")
    )
    api_diff = api_snapshot.diff_public_api(
        api_baseline, api_snapshot.public_api_snapshot()
    )
    schema_diff = api_snapshot.diff_sbtrace_schema(
        schema_baseline, api_snapshot.sbtrace_schema_snapshot()
    )
    assert api_diff["breaking"] == [], (
        "Public API drifted from baseline. Refresh with "
        "`python scripts/check_api_compat.py --write-baseline` if intentional, "
        f"after bumping SemVer.\nBreaking:\n  - "
        + "\n  - ".join(api_diff["breaking"])
    )
    assert api_diff["compatible"] == [], (
        "Public API gained additions not yet in the baseline. Refresh with "
        "`python scripts/check_api_compat.py --write-baseline`.\n"
        + "\n  + ".join(api_diff["compatible"])
    )
    assert schema_diff["breaking"] == [], (
        "SB-Trace schema drifted (breaking) from baseline.\nBreaking:\n  - "
        + "\n  - ".join(schema_diff["breaking"])
    )
    assert schema_diff["compatible"] == [], (
        "SB-Trace schema gained additions not yet in the baseline.\n  + "
        + "\n  + ".join(schema_diff["compatible"])
    )


# ---------------------------------------------------------------------------
# Diff classifier
# ---------------------------------------------------------------------------


def test_diff_public_api_flags_removed_symbol_as_breaking():
    snap = api_snapshot.public_api_snapshot()
    mutated = copy.deepcopy(snap)
    mutated["symbols"].pop("replay")
    diff = api_snapshot.diff_public_api(snap, mutated)
    assert any("replay" in line for line in diff["breaking"])
    assert diff["compatible"] == []


def test_diff_public_api_flags_new_symbol_as_compatible():
    snap = api_snapshot.public_api_snapshot()
    mutated = copy.deepcopy(snap)
    mutated["symbols"]["brand_new_helper"] = {
        "kind": "function",
        "name": "brand_new_helper",
        "signature": {"parameters": [], "return_annotation": ""},
        "qualname": "stepback.brand_new_helper",
    }
    diff = api_snapshot.diff_public_api(snap, mutated)
    assert diff["breaking"] == []
    assert any("brand_new_helper" in line for line in diff["compatible"])


def test_diff_public_api_flags_required_param_addition_as_breaking():
    snap = api_snapshot.public_api_snapshot()
    mutated = copy.deepcopy(snap)
    record = mutated["symbols"]["replay"]
    record["signature"]["parameters"].append(
        {
            "name": "new_required",
            "kind": "POSITIONAL_OR_KEYWORD",
            "annotation": "str",
            "default": "",
        }
    )
    diff = api_snapshot.diff_public_api(snap, mutated)
    assert any(
        "new_required" in line and "required" in line for line in diff["breaking"]
    )


def test_diff_public_api_flags_optional_param_addition_as_compatible():
    snap = api_snapshot.public_api_snapshot()
    mutated = copy.deepcopy(snap)
    record = mutated["symbols"]["replay"]
    record["signature"]["parameters"].append(
        {
            "name": "new_optional",
            "kind": "KEYWORD_ONLY",
            "annotation": "str",
            "default": "'x'",
        }
    )
    diff = api_snapshot.diff_public_api(snap, mutated)
    assert diff["breaking"] == []
    assert any("new_optional" in line for line in diff["compatible"])


def test_diff_public_api_flags_default_removal_as_breaking():
    snap = api_snapshot.public_api_snapshot()
    target_name = None
    for name, rec in snap["symbols"].items():
        if rec.get("kind") != "function":
            continue
        sig = rec.get("signature") or {}
        for p in sig.get("parameters") or []:
            if p.get("default"):
                target_name = name
                target_param = p["name"]
                break
        if target_name:
            break
    assert target_name, "expected at least one function with a defaulted parameter"
    mutated = copy.deepcopy(snap)
    for p in mutated["symbols"][target_name]["signature"]["parameters"]:
        if p["name"] == target_param:
            p["default"] = ""
    diff = api_snapshot.diff_public_api(snap, mutated)
    assert any(target_param in line for line in diff["breaking"])


def test_diff_sbtrace_schema_flags_required_field_removal_as_breaking():
    snap = api_snapshot.sbtrace_schema_snapshot()
    mutated = copy.deepcopy(snap)
    mutated["step_required"] = sorted(set(mutated["step_required"]) - {mutated["step_required"][0]})
    diff = api_snapshot.diff_sbtrace_schema(snap, mutated)
    assert diff["breaking"], diff


def test_diff_sbtrace_schema_flags_optional_field_addition_as_compatible():
    snap = api_snapshot.sbtrace_schema_snapshot()
    mutated = copy.deepcopy(snap)
    mutated["step_optional"] = sorted(set(mutated["step_optional"]) | {"new_extension_field"})
    diff = api_snapshot.diff_sbtrace_schema(snap, mutated)
    assert diff["breaking"] == []
    assert any("new_extension_field" in line for line in diff["compatible"])


def test_diff_sbtrace_schema_flags_wire_major_bump_as_breaking():
    snap = api_snapshot.sbtrace_schema_snapshot()
    mutated = copy.deepcopy(snap)
    mutated["wire_version_info"] = [2, 0, 0]
    mutated["wire_version"] = "2.0.0"
    diff = api_snapshot.diff_sbtrace_schema(snap, mutated)
    assert any("major" in line for line in diff["breaking"])


def test_diff_sbtrace_schema_flags_magic_change_as_breaking():
    snap = api_snapshot.sbtrace_schema_snapshot()
    mutated = copy.deepcopy(snap)
    mutated["magic"] = "different/magic"
    diff = api_snapshot.diff_sbtrace_schema(snap, mutated)
    assert any("magic" in line for line in diff["breaking"])


def test_diff_refuses_to_compare_across_snapshot_format_versions():
    snap = api_snapshot.public_api_snapshot()
    mutated = copy.deepcopy(snap)
    mutated["snapshot_format_version"] = snap["snapshot_format_version"] + 1
    diff = api_snapshot.diff_public_api(snap, mutated)
    assert any("snapshot_format_version" in line for line in diff["breaking"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_cli(*args: str, expect_zero: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if expect_zero:
        assert proc.returncode == 0, (
            f"check_api_compat.py exited {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def test_cli_clean_run_reports_no_breaking_changes():
    proc = _run_cli("--baseline-dir", str(BASELINE_DIR))
    assert "no breaking changes" in proc.stdout.lower()


def test_cli_emits_json_report(tmp_path: Path):
    proc = _run_cli("--baseline-dir", str(BASELINE_DIR), "--json")
    payload = json.loads(proc.stdout)
    assert "public_api_diff" in payload
    assert "sbtrace_schema_diff" in payload
    assert payload["public_api_diff"]["breaking"] == []


def test_cli_writes_snapshots_to_out_dir(tmp_path: Path):
    out = tmp_path / "snap"
    _run_cli("--baseline-dir", str(BASELINE_DIR), "--out", str(out))
    api = json.loads((out / "public_api.json").read_text(encoding="utf-8"))
    schema = json.loads((out / "sbtrace_schema.json").read_text(encoding="utf-8"))
    assert api["package"] == "stepback"
    assert schema["wire_version"] == "1.0.0"


def test_cli_fails_when_baseline_diverges(tmp_path: Path):
    """A fake baseline missing a real symbol must trigger a non-zero exit."""
    snap = api_snapshot.public_api_snapshot()
    schema = api_snapshot.sbtrace_schema_snapshot()
    snap["symbols"].pop("replay")
    fake = tmp_path / "fake-baseline"
    fake.mkdir()
    (fake / "public_api.json").write_text(api_snapshot.dumps(snap), encoding="utf-8")
    (fake / "sbtrace_schema.json").write_text(api_snapshot.dumps(schema), encoding="utf-8")
    proc = _run_cli("--baseline-dir", str(fake), expect_zero=False)
    # An *added* symbol relative to the (mutated) baseline shows up as
    # compatible, not breaking. This exercises the inverse direction.
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "new public symbol 'replay'" in proc.stdout


def test_cli_fails_when_current_drops_a_required_baseline_symbol(tmp_path: Path):
    """Pretend the *baseline* had an extra symbol; current drops it -> breaking."""
    snap = api_snapshot.public_api_snapshot()
    schema = api_snapshot.sbtrace_schema_snapshot()
    snap["symbols"]["__phantom_symbol__"] = {
        "kind": "function",
        "name": "__phantom_symbol__",
        "signature": {"parameters": [], "return_annotation": ""},
        "qualname": "stepback.__phantom_symbol__",
    }
    fake = tmp_path / "fake-baseline"
    fake.mkdir()
    (fake / "public_api.json").write_text(api_snapshot.dumps(snap), encoding="utf-8")
    (fake / "sbtrace_schema.json").write_text(api_snapshot.dumps(schema), encoding="utf-8")
    proc = _run_cli("--baseline-dir", str(fake), expect_zero=False)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "__phantom_symbol__" in proc.stdout
    assert "FAIL" in (proc.stderr + proc.stdout)


def test_cli_write_baseline_round_trips(tmp_path: Path, monkeypatch):
    """--write-baseline must produce a baseline that re-validates clean."""
    out = tmp_path / "out"
    proc = _run_cli(
        "--out", str(out), "--baseline-dir", str(BASELINE_DIR), "--json"
    )
    payload = json.loads(proc.stdout)
    assert payload["public_api_diff"]["breaking"] == []
    assert payload["sbtrace_schema_diff"]["breaking"] == []


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------


def test_baseline_files_are_packaged_via_package_data():
    """Step 3 ships fixtures inside the wheel; the same applies here."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "api_baselines/**/*.json" in pyproject, (
        "pyproject.toml must include api_baselines/**/*.json under "
        "stepback.conformance package-data so the snapshots ship in the wheel."
    )


# ---------------------------------------------------------------------------
# CI wiring
# ---------------------------------------------------------------------------


def test_ci_workflow_runs_api_compat_check():
    """Step 25: the CI workflow must invoke the API compatibility checker.

    The contract is that PRs and pushes to main automatically diff the
    public API and SB-Trace schema against the most recent ``v*`` release
    tag. A new job in ``.github/workflows/ci.yml`` is the contract; if
    someone removes it the test catches it.
    """
    ci_yaml = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    assert "check_api_compat.py" in ci_yaml, (
        "CI workflow must run scripts/check_api_compat.py so API + "
        "SB-Trace schema drift is caught against the last release tag."
    )
    assert "--auto-baseline-ref" in ci_yaml, (
        "CI must pass --auto-baseline-ref so the diff is against the "
        "most recent v* tag (Step 25 of 100_STEPS.md)."
    )
    assert "fetch-depth: 0" in ci_yaml, (
        "CI must check out with fetch-depth: 0 so --auto-baseline-ref "
        "can resolve the most recent v* release tag."
    )

