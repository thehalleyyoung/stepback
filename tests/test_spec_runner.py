"""Tests for ``stepback spec test`` (Step 47).

These tests exercise the conformance runner against synthetic
"implementation" shims so we never need a real Rust/Go/Node binary on
the developer's machine.
"""
from __future__ import annotations

import io
import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from stepback import cli
from stepback.spec_runner import (
    ConformanceRun,
    FixtureResult,
    _expected_class_alternatives,
    _extract_class,
    default_manifest_path,
    render_text,
    run_conformance,
)


# ----- helpers --------------------------------------------------------- #


def _write_impl_script(tmp_path: Path, body: str) -> Path:
    """Write a minimal Python "implementation" and make it executable."""

    p = tmp_path / "impl.py"
    p.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, hashlib\n"
        + textwrap.dedent(body)
    )
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return p


def _argv_for(script: Path) -> list[str]:
    # Avoid #! reliance on Windows / unusual envs by always invoking
    # the current Python explicitly.
    return [sys.executable, str(script)]


# ----- locate the bundled manifest ------------------------------------ #


def test_default_manifest_path_resolves() -> None:
    p = default_manifest_path()
    assert p is not None, "bundled v1 manifest should be discoverable"
    assert p.is_file()
    data = json.loads(p.read_text())
    assert data["format_version"] == 1
    assert "good" in data and "corrupt" in data


# ----- a "perfect" implementation passes everything ------------------- #


@pytest.fixture
def perfect_impl(tmp_path: Path) -> list[str]:
    """A shim that accepts good fixtures and rejects corrupt ones with
    the precise rejection class the manifest expects."""
    body = """
        # subcommand contract: verify | hash
        cmd = sys.argv[1]
        path = sys.argv[2]
        name = path.rsplit('/', 1)[-1]
        if cmd == 'hash':
            with open(path, 'rb') as fh:
                print(hashlib.sha256(fh.read()).hexdigest())
            sys.exit(0)
        if cmd == 'verify':
            mapping = {
                'truncated_body.sb': ('Parse', 1),
                'flipped_hmac.sb': ('HmacMismatch', 2),
                'flipped_sig.sb': ('SignatureMismatch', 3),
                'broken_chain.sb': ('HmacMismatch', 4),
                'bad_format_version.sb': ('UnsupportedFormatVersion', 5),
                'attested_tampered.pack': ('Parse', 6),
            }
            if name in mapping:
                cls, code = mapping[name]
                print(f"{cls}: simulated", file=sys.stderr)
                sys.exit(code)
            sys.exit(0)
        print(f"unknown subcommand {cmd}", file=sys.stderr)
        sys.exit(127)
    """
    script = _write_impl_script(tmp_path, body)
    return _argv_for(script)


def test_perfect_impl_passes_full_corpus(perfect_impl: list[str]) -> None:
    run = run_conformance(perfect_impl)
    assert run.ok, render_text(run)
    assert run.failed == 0
    assert run.passed == len(run.results)
    # Should have at least the documented good fixtures + 5 corrupt.
    names = {r.name for r in run.results}
    for required in (
        "header_only.sb",
        "multi_step.sb",
        "with_blobs.sb",
        "truncated_body.sb",
        "flipped_hmac.sb",
        "flipped_sig.sb",
        "broken_chain.sb",
        "bad_format_version.sb",
    ):
        assert required in names, f"missing {required!r} from corpus"
    # No warnings expected: rejection classes are the canonical names.
    assert run.warnings == 0, [r.warnings for r in run.results if r.warnings]


# ----- an impl that always succeeds fails on corrupt fixtures --------- #


def test_always_accept_impl_fails(tmp_path: Path) -> None:
    body = """
        # always exit 0
        sys.exit(0)
    """
    impl = _argv_for(_write_impl_script(tmp_path, body))
    run = run_conformance(impl, enable_hash=False)
    assert not run.ok
    # Every corrupt fixture should be a FAIL.
    corrupt_results = [r for r in run.results if r.bucket == "corrupt"]
    assert corrupt_results, "manifest should declare corrupt fixtures"
    assert all(not r.ok for r in corrupt_results)
    # And all good fixtures should still pass.
    good_results = [r for r in run.results if r.bucket == "good"]
    assert all(r.ok for r in good_results)


# ----- an impl that always rejects fails on good fixtures ------------- #


def test_always_reject_impl_fails(tmp_path: Path) -> None:
    body = """
        print("Parse: simulated", file=sys.stderr)
        sys.exit(1)
    """
    impl = _argv_for(_write_impl_script(tmp_path, body))
    run = run_conformance(impl, enable_hash=False)
    assert not run.ok
    good_results = [r for r in run.results if r.bucket == "good"]
    assert all(not r.ok for r in good_results)


# ----- limiting the corpus with --only ------------------------------- #


def test_only_filter(perfect_impl: list[str]) -> None:
    run = run_conformance(perfect_impl, only=["header_only.sb"])
    assert len(run.results) == 1
    assert run.results[0].name == "header_only.sb"
    assert run.ok


# ----- a permissive rejection class is accepted but warned ----------- #


def test_unrecognized_rejection_class_warns(tmp_path: Path) -> None:
    body = """
        cmd = sys.argv[1]
        path = sys.argv[2]
        name = path.rsplit('/', 1)[-1]
        if cmd == 'verify':
            if name in ('header_only.sb', 'multi_step.sb', 'with_blobs.sb'):
                sys.exit(0)
            print("totally-made-up-token: oops", file=sys.stderr)
            sys.exit(7)
        sys.exit(127)
    """
    impl = _argv_for(_write_impl_script(tmp_path, body))
    run = run_conformance(impl, enable_hash=False)
    # Corrupt fixtures should still PASS (rejection happened).
    corrupt = [r for r in run.results if r.bucket == "corrupt"]
    assert all(r.ok for r in corrupt)
    # But each should carry a warning about the missing rejection class.
    assert any(r.warnings for r in corrupt), "expected at least one warning"


# ----- expected_error_kind splitter is sane -------------------------- #


def test_expected_class_alternatives() -> None:
    assert _expected_class_alternatives("Parse") == ["Parse"]
    assert _expected_class_alternatives("BadHexOrHmacMismatch") == [
        "BadHex", "HmacMismatch",
    ]
    assert _expected_class_alternatives(
        "UnsupportedFormatVersionOrHmacMismatch"
    ) == ["UnsupportedFormatVersion", "HmacMismatch"]
    assert _expected_class_alternatives("SignatureMismatch") == [
        "SignatureMismatch"
    ]


def test_extract_class_handles_typical_formats() -> None:
    assert _extract_class("HmacMismatch: oh no") == "HmacMismatch"
    assert _extract_class("error: SignatureMismatch in frame 1") == "SignatureMismatch"
    assert _extract_class("Parse, unexpected eof at byte 42") == "Parse"
    # case-insensitive equality
    assert _extract_class("hmacmismatch: bad") == "HmacMismatch"
    # unknown token
    assert _extract_class("oops: something") is None
    assert _extract_class("") is None


# ----- exit code from CLI mirrors run.ok ---------------------------- #


def test_cli_exit_code_when_perfect_impl(perfect_impl: list[str], capsys, monkeypatch) -> None:
    rc = cli.main(["spec", "test", *perfect_impl])
    assert rc == 0
    out = capsys.readouterr().out
    assert "passed:" in out
    assert "failed: 0" in out


def test_cli_exit_code_when_failing_impl(tmp_path: Path, capsys) -> None:
    impl_script = _write_impl_script(tmp_path, "sys.exit(0)\n")
    rc = cli.main(["spec", "test", sys.executable, str(impl_script)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "failed:" in out
    # at least one FAIL line for a corrupt fixture
    assert "FAIL" in out


def test_cli_json_report(perfect_impl: list[str], capsys) -> None:
    rc = cli.main(["spec", "test", "--json", *perfect_impl])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["summary"]["ok"] is True
    assert payload["format_version"] == 1
    assert payload["summary"]["passed"] == payload["summary"]["total"]
    # results carry the documented shape
    sample = payload["results"][0]
    for key in (
        "name", "bucket", "expected", "expected_error_kind",
        "expected_sha256", "exit_code", "stderr_first_line",
        "observed_error_kind", "observed_sha256", "ok", "warnings", "error",
    ):
        assert key in sample


# ----- missing fixture is surfaced cleanly --------------------------- #


def test_missing_fixture_directory_errors(tmp_path: Path, perfect_impl: list[str]) -> None:
    fake_manifest = tmp_path / "manifest.json"
    fake_manifest.write_text(json.dumps({
        "format_version": 1,
        "canonicalisation_version": "1",
        "good": [{"expected": "ok", "name": "missing.sb", "sha256": "00" * 32, "size_bytes": 0}],
        "corrupt": [],
    }))
    run = run_conformance(perfect_impl, manifest_path=fake_manifest)
    assert not run.ok
    only = run.results[0]
    assert only.error is not None
    assert "missing on disk" in only.error
