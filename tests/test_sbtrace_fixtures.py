"""Determinism check for the SB-Trace v1 conformance fixture corpus.

The Rust verifier (`stepback-core/crates/sb-verify/tests/fixtures.rs`)
locks itself to the bytes of every committed fixture via SHA-256.
This test covers the Python side: re-running the generator must not
change any fixture or the manifest, otherwise the writer or
canonicalisation drifted and we would silently invalidate every
independent reader's golden hashes.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "stepback-core" / "fixtures" / "v1"
GEN_SCRIPT = REPO_ROOT / "stepback-core" / "scripts" / "gen_fixtures.py"


def _sha256_tree(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _load_gen_module():
    spec = importlib.util.spec_from_file_location("gen_fixtures", GEN_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not GEN_SCRIPT.exists(), reason="fixture generator not present")
def test_fixture_corpus_is_deterministic(tmp_path: Path) -> None:
    if not FIXTURE_ROOT.exists():
        pytest.skip("fixture corpus not generated yet")

    before = _sha256_tree(FIXTURE_ROOT)
    assert before, "fixture corpus is empty"

    # Re-run the generator into the live tree (it is idempotent and
    # only rewrites bytes that change) and assert the on-disk hashes
    # are identical.
    gen = _load_gen_module()
    rc = gen.main()
    assert rc == 0
    after = _sha256_tree(FIXTURE_ROOT)
    assert after == before, (
        "fixture corpus drifted under regeneration — the writer or "
        "canonicalisation changed in a way that would break the Rust "
        "verifier's pinned SHA-256 fingerprints. Diff:\n"
        f"before={before}\nafter={after}"
    )


@pytest.mark.skipif(not GEN_SCRIPT.exists(), reason="fixture generator not present")
def test_manifest_matches_on_disk_bytes() -> None:
    if not FIXTURE_ROOT.exists():
        pytest.skip("fixture corpus not generated yet")
    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_bytes())
    for entry in manifest["good"]:
        data = (FIXTURE_ROOT / "good" / entry["name"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["name"]
        assert len(data) == entry["size_bytes"], entry["name"]
    for entry in manifest["corrupt"]:
        data = (FIXTURE_ROOT / "corrupt" / entry["name"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["name"]
        assert len(data) == entry["size_bytes"], entry["name"]


@pytest.mark.skipif(not GEN_SCRIPT.exists(), reason="fixture generator not present")
def test_good_fixtures_round_trip_through_python_reader() -> None:
    """Sanity check: the Python reader must accept what the writer wrote.

    This protects the corpus from a Python-side regression where the
    writer emits frames the Python reader can't read either —
    catching the bug before the Rust verifier is even consulted.
    """
    if not FIXTURE_ROOT.exists():
        pytest.skip("fixture corpus not generated yet")
    from stepback.trace_reader import iter_steps

    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_bytes())
    hmac_key = bytes.fromhex(manifest["hmac_key_hex"])
    for entry in manifest["good"]:
        path = FIXTURE_ROOT / "good" / entry["name"]
        # iter_steps verifies HMAC + signatures while iterating.
        steps = list(iter_steps(str(path), hmac_key=hmac_key))
        # header_only.sb has 0 step frames (header + tail only); the
        # other fixtures have at least one step.
        if entry["name"] == "header_only.sb":
            assert steps == []
        else:
            assert steps, f"{entry['name']} should yield at least one step"
