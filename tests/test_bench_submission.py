"""Tests for stepback.bench.submission (Step 123).

Covers:
- HardwareManifest: construction, detect(), to_json/from_json round-trip,
  container fields
- CodeProvenance: construction, detect(), to_json/from_json round-trip
- TracePackManifest: construction, from_directory(), to_json/from_json
- AuditDeclaration: construction, today(), to_json/from_json
- SubmissionManifest: construction, create(), to_json/from_json round-trip
- validate_submission: valid manifest, every error path, warnings
- validate_submission_json: parse-from-dict path
- SubmissionValidationResult: to_json/from_json, summary_line
- CLI: stepback bench validate-submission (valid and invalid manifests)
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid

import pytest

from stepback.bench.submission import (
    RULES_VERSION,
    AuditDeclaration,
    CodeProvenance,
    HardwareManifest,
    SubmissionManifest,
    SubmissionValidationResult,
    TracePackManifest,
    ValidationError,
    validate_submission,
    validate_submission_json,
)
from stepback.cli import main as cli_main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_SHA256 = "a" * 64


_SENTINEL = object()


def _valid_manifest(
    *,
    rules_version: str = RULES_VERSION,
    submission_id: str | None = None,
    git_commit: str | None = "abc1234" + "0" * 33,  # 40 chars
    wheel_sha256: str | None = None,
    trace_count: int = 10,
    pack_sha256: str = _FAKE_SHA256,
    verified: bool = True,
    exact_commands=_SENTINEL,
    validator_output: str = "12/12 PASSED",
    grants_source_access: bool = True,
    grants_trace_pack_access: bool = True,
    bench_results=_SENTINEL,
) -> SubmissionManifest:
    if bench_results is _SENTINEL:
        bench_results = [
            {
                "schema_version": "1.0",
                "run_id": str(uuid.uuid4()),
                "timestamp_utc": "2026-05-12T00:00:00Z",
                "corpus_id": "synthetic-10-random_step",
                "trace_count": trace_count,
                "versions": {"stepback_version": "0.9.0"},
                "hardware": {"os": "Linux"},
            }
        ]
    if exact_commands is _SENTINEL:
        exact_commands = ["stepback bench replay-caching --n-steps 10 --n-trials 5"]
    return SubmissionManifest(
        rules_version=rules_version,
        submission_id=submission_id or str(uuid.uuid4()),
        hardware=HardwareManifest(
            os="Linux",
            cpu_count=4,
            cpu_model="FakeCPU",
            ram_gb=16.0,
        ),
        code=CodeProvenance(
            stepback_version="0.9.0",
            python_version="3.11.8",
            python_implementation="CPython",
            git_commit=git_commit,
            wheel_sha256=wheel_sha256,
        ),
        trace_pack=TracePackManifest(
            corpus_id="synthetic-10-random_step",
            trace_count=trace_count,
            pack_sha256=pack_sha256,
            verified=verified,
        ),
        exact_commands=exact_commands,
        validator_output=validator_output,
        bench_results=bench_results,
        audit=AuditDeclaration(
            submitter_name="Alice",
            submitter_email="alice@example.com",
            grants_source_access=grants_source_access,
            grants_trace_pack_access=grants_trace_pack_access,
            submission_date="2026-05-12",
        ),
    )


# ---------------------------------------------------------------------------
# HardwareManifest
# ---------------------------------------------------------------------------


def test_hardware_manifest_basic():
    hw = HardwareManifest(os="Linux", cpu_count=8, cpu_model="FakeCPU", ram_gb=32.0)
    assert hw.os == "Linux"
    assert hw.cpu_count == 8
    assert hw.docker_image_digest is None


def test_hardware_manifest_to_from_json():
    hw = HardwareManifest(
        os="Darwin",
        cpu_count=4,
        cpu_model="Apple M1",
        ram_gb=8.0,
        docker_image_digest="sha256:" + "b" * 64,
        container_runtime="docker 24.0.5",
    )
    d = hw.to_json()
    assert d["os"] == "Darwin"
    assert d["docker_image_digest"] == "sha256:" + "b" * 64
    hw2 = HardwareManifest.from_json(d)
    assert hw2.os == hw.os
    assert hw2.cpu_model == hw.cpu_model
    assert hw2.docker_image_digest == hw.docker_image_digest
    assert hw2.container_runtime == hw.container_runtime


def test_hardware_manifest_detect():
    hw = HardwareManifest.detect()
    assert hw.os  # non-empty
    # cpu_count is int >= 1 on any normal machine
    assert hw.cpu_count is None or hw.cpu_count >= 1


def test_hardware_manifest_from_json_missing_fields():
    hw = HardwareManifest.from_json({})
    assert hw.os == ""
    assert hw.cpu_count is None


# ---------------------------------------------------------------------------
# CodeProvenance
# ---------------------------------------------------------------------------


def test_code_provenance_basic():
    cp = CodeProvenance(
        stepback_version="1.0.0",
        python_version="3.11.8",
        python_implementation="CPython",
        git_commit="a" * 40,
    )
    assert cp.git_commit == "a" * 40
    assert cp.wheel_sha256 is None


def test_code_provenance_to_from_json():
    cp = CodeProvenance(
        stepback_version="1.0.0",
        python_version="3.11.8",
        python_implementation="CPython",
        git_commit="a" * 40,
        git_dirty=False,
        wheel_sha256=_FAKE_SHA256,
        source_uri="https://github.com/example/stepback",
    )
    d = cp.to_json()
    assert d["git_commit"] == "a" * 40
    assert d["wheel_sha256"] == _FAKE_SHA256
    cp2 = CodeProvenance.from_json(d)
    assert cp2.stepback_version == cp.stepback_version
    assert cp2.git_commit == cp.git_commit
    assert cp2.wheel_sha256 == cp.wheel_sha256
    assert cp2.source_uri == cp.source_uri
    assert cp2.git_dirty is False


def test_code_provenance_detect():
    cp = CodeProvenance.detect()
    assert cp.stepback_version  # non-empty
    assert cp.python_version


# ---------------------------------------------------------------------------
# TracePackManifest
# ---------------------------------------------------------------------------


def test_trace_pack_manifest_basic():
    tp = TracePackManifest(
        corpus_id="synthetic-200-random_step",
        trace_count=200,
        pack_sha256=_FAKE_SHA256,
        verified=True,
    )
    assert tp.verified is True
    assert tp.hmac_key_id is None


def test_trace_pack_manifest_to_from_json():
    tp = TracePackManifest(
        corpus_id="swe-bench-verified",
        trace_count=500,
        pack_sha256=_FAKE_SHA256,
        hmac_key_id="key-001",
        verified=True,
    )
    d = tp.to_json()
    tp2 = TracePackManifest.from_json(d)
    assert tp2.corpus_id == tp.corpus_id
    assert tp2.trace_count == tp.trace_count
    assert tp2.pack_sha256 == tp.pack_sha256
    assert tp2.hmac_key_id == "key-001"
    assert tp2.verified is True


def test_trace_pack_manifest_from_directory(tmp_path):
    """from_directory scans .sb files and produces a stable digest."""
    # Create two fake .sb files
    (tmp_path / "trace_001.sb").write_bytes(b"\x00\x01\x02")
    (tmp_path / "trace_002.sb").write_bytes(b"\x03\x04\x05")

    tp = TracePackManifest.from_directory(str(tmp_path), corpus_id="test-corpus")
    assert tp.trace_count == 2
    assert tp.corpus_id == "test-corpus"
    assert len(tp.pack_sha256) == 64  # hex SHA-256
    assert tp.verified is True  # default

    # Digest must be stable (same files → same digest)
    tp2 = TracePackManifest.from_directory(str(tmp_path), corpus_id="test-corpus")
    assert tp2.pack_sha256 == tp.pack_sha256


def test_trace_pack_manifest_from_directory_different_content(tmp_path):
    """Changing a file changes the digest."""
    (tmp_path / "trace_001.sb").write_bytes(b"\x00")
    tp1 = TracePackManifest.from_directory(str(tmp_path), corpus_id="c")
    (tmp_path / "trace_001.sb").write_bytes(b"\xff")
    tp2 = TracePackManifest.from_directory(str(tmp_path), corpus_id="c")
    assert tp1.pack_sha256 != tp2.pack_sha256


def test_trace_pack_manifest_from_directory_empty(tmp_path):
    """Empty directory → trace_count=0."""
    tp = TracePackManifest.from_directory(str(tmp_path), corpus_id="empty")
    assert tp.trace_count == 0
    assert len(tp.pack_sha256) == 64


# ---------------------------------------------------------------------------
# AuditDeclaration
# ---------------------------------------------------------------------------


def test_audit_declaration_basic():
    a = AuditDeclaration(
        submitter_name="Bob",
        grants_source_access=True,
        grants_trace_pack_access=True,
        submission_date="2026-05-12",
    )
    assert a.submitter_email is None


def test_audit_declaration_today():
    a = AuditDeclaration.today(submitter_name="Carol")
    import re
    assert re.match(r"\d{4}-\d{2}-\d{2}", a.submission_date)
    assert a.grants_source_access is True


def test_audit_declaration_to_from_json():
    a = AuditDeclaration(
        submitter_name="Dave",
        submitter_email="dave@x.com",
        submitter_organization="ACME",
        grants_source_access=True,
        grants_trace_pack_access=False,
        submission_date="2026-01-01",
    )
    d = a.to_json()
    a2 = AuditDeclaration.from_json(d)
    assert a2.submitter_name == "Dave"
    assert a2.submitter_email == "dave@x.com"
    assert a2.submitter_organization == "ACME"
    assert a2.grants_source_access is True
    assert a2.grants_trace_pack_access is False
    assert a2.submission_date == "2026-01-01"


# ---------------------------------------------------------------------------
# SubmissionManifest
# ---------------------------------------------------------------------------


def test_submission_manifest_to_from_json():
    m = _valid_manifest()
    d = m.to_json()
    assert d["rules_version"] == RULES_VERSION
    m2 = SubmissionManifest.from_json(d)
    assert m2.submission_id == m.submission_id
    assert m2.rules_version == RULES_VERSION
    assert m2.hardware.os == "Linux"
    assert m2.code.stepback_version == "0.9.0"
    assert m2.trace_pack.corpus_id == "synthetic-10-random_step"
    assert m2.audit.submitter_name == "Alice"


def test_submission_manifest_json_roundtrip_preserves_commands():
    m = _valid_manifest(exact_commands=["cmd1", "cmd2"])
    m2 = SubmissionManifest.from_json(m.to_json())
    assert m2.exact_commands == ["cmd1", "cmd2"]


def test_submission_manifest_create():
    m = SubmissionManifest.create(
        trace_pack=TracePackManifest(
            corpus_id="test",
            trace_count=5,
            pack_sha256=_FAKE_SHA256,
            verified=True,
        ),
        exact_commands=["stepback bench replay-caching"],
        validator_output="12/12 PASSED",
        bench_results=[
            {
                "schema_version": "1.0",
                "run_id": str(uuid.uuid4()),
                "timestamp_utc": "2026-05-12T00:00:00Z",
                "corpus_id": "test",
                "trace_count": 5,
            }
        ],
        audit=AuditDeclaration.today(submitter_name="Eve"),
    )
    assert m.rules_version == RULES_VERSION
    assert uuid.UUID(m.submission_id)  # valid UUID
    assert m.hardware.os  # auto-detected


# ---------------------------------------------------------------------------
# validate_submission — valid path
# ---------------------------------------------------------------------------


def test_validate_valid_manifest():
    result = validate_submission(_valid_manifest())
    assert result.valid is True
    assert result.errors == []


def test_validate_valid_no_warnings_with_email():
    result = validate_submission(_valid_manifest())
    # No email-missing warning because _valid_manifest includes email
    email_warns = [w for w in result.warnings if "email" in w]
    assert email_warns == []


def test_validate_result_to_from_json():
    result = validate_submission(_valid_manifest())
    d = result.to_json()
    r2 = SubmissionValidationResult.from_json(d)
    assert r2.valid == result.valid
    assert r2.rules_version == result.rules_version
    assert r2.submission_id == result.submission_id


def test_validate_summary_line_valid():
    result = validate_submission(_valid_manifest())
    line = result.summary_line()
    assert "VALID" in line
    assert "INVALID" not in line


# ---------------------------------------------------------------------------
# validate_submission — error paths
# ---------------------------------------------------------------------------


def test_validate_wrong_rules_version():
    m = _valid_manifest(rules_version="2.0")
    result = validate_submission(m)
    assert not result.valid
    fields = [e.field for e in result.errors]
    assert "rules_version" in fields


def test_validate_missing_git_commit_and_wheel():
    m = _valid_manifest(git_commit=None, wheel_sha256=None)
    result = validate_submission(m)
    assert not result.valid
    assert any(e.field == "code" for e in result.errors)


def test_validate_unverified_trace_pack():
    m = _valid_manifest(verified=False)
    result = validate_submission(m)
    assert not result.valid
    assert any("verified" in e.field for e in result.errors)


def test_validate_empty_exact_commands():
    m = _valid_manifest(exact_commands=[])
    result = validate_submission(m)
    assert not result.valid
    assert any("exact_commands" in e.field for e in result.errors)


def test_validate_blank_command_entry():
    m = _valid_manifest(exact_commands=["valid cmd", ""])
    result = validate_submission(m)
    assert not result.valid
    assert any("exact_commands[1]" in e.field for e in result.errors)


def test_validate_missing_passed_in_validator_output():
    m = _valid_manifest(validator_output="all done, no summary here")
    result = validate_submission(m)
    assert not result.valid
    assert any("validator_output" in e.field for e in result.errors)


def test_validate_empty_validator_output():
    m = _valid_manifest(validator_output="")
    result = validate_submission(m)
    assert not result.valid
    assert any("validator_output" in e.field for e in result.errors)


def test_validate_grants_source_access_false():
    m = _valid_manifest(grants_source_access=False)
    result = validate_submission(m)
    assert not result.valid
    assert any("grants_source_access" in e.field for e in result.errors)


def test_validate_grants_trace_pack_access_false():
    m = _valid_manifest(grants_trace_pack_access=False)
    result = validate_submission(m)
    assert not result.valid
    assert any("grants_trace_pack_access" in e.field for e in result.errors)


def test_validate_empty_submitter_name():
    m = _valid_manifest()
    m.audit.submitter_name = ""
    result = validate_submission(m)
    assert not result.valid
    assert any("submitter_name" in e.field for e in result.errors)


def test_validate_bad_submission_date():
    m = _valid_manifest()
    m.audit.submission_date = "not-a-date"
    result = validate_submission(m)
    assert not result.valid
    assert any("submission_date" in e.field for e in result.errors)


def test_validate_empty_corpus_id():
    m = _valid_manifest()
    m.trace_pack.corpus_id = ""
    result = validate_submission(m)
    assert not result.valid
    assert any("corpus_id" in e.field for e in result.errors)


def test_validate_short_pack_sha256():
    m = _valid_manifest(pack_sha256="deadbeef")
    result = validate_submission(m)
    assert not result.valid
    assert any("pack_sha256" in e.field for e in result.errors)


def test_validate_zero_trace_count():
    m = _valid_manifest(trace_count=0)
    result = validate_submission(m)
    assert not result.valid
    assert any("trace_count" in e.field for e in result.errors)


def test_validate_empty_bench_results():
    m = _valid_manifest(bench_results=[])
    result = validate_submission(m)
    assert not result.valid
    assert any("bench_results" in e.field for e in result.errors)


def test_validate_bench_result_missing_schema_version():
    bench = [{"run_id": "x", "timestamp_utc": "t", "corpus_id": "c", "trace_count": 1}]
    m = _valid_manifest(bench_results=bench)
    result = validate_submission(m)
    assert not result.valid
    assert any("schema_version" in e.field for e in result.errors)


def test_validate_bench_result_missing_required_key():
    bench = [{"schema_version": "1.0", "corpus_id": "c", "trace_count": 1}]
    m = _valid_manifest(bench_results=bench)
    result = validate_submission(m)
    # run_id and timestamp_utc are both missing
    assert not result.valid


def test_validate_summary_line_invalid():
    m = _valid_manifest(verified=False, grants_source_access=False)
    result = validate_submission(m)
    line = result.summary_line()
    assert "INVALID" in line


# ---------------------------------------------------------------------------
# validate_submission — warning paths
# ---------------------------------------------------------------------------


def test_validate_git_dirty_produces_warning():
    m = _valid_manifest()
    m.code.git_dirty = True
    result = validate_submission(m)
    assert result.valid  # still valid
    assert any("git_dirty" in w for w in result.warnings)


def test_validate_no_email_warning():
    m = _valid_manifest()
    m.audit.submitter_email = None
    result = validate_submission(m)
    assert result.valid
    assert any("email" in w for w in result.warnings)


def test_validate_failed_in_validator_output_is_warning():
    m = _valid_manifest(validator_output="12/12 PASSED but 1 FAILED in optional")
    result = validate_submission(m)
    # Contains PASSED so no error, but FAILED triggers warning
    assert result.valid
    assert any("FAILED" in w for w in result.warnings)


def test_validate_non_uuid_submission_id_warning():
    m = _valid_manifest(submission_id="my-custom-id")
    result = validate_submission(m)
    assert result.valid  # still valid, just a warning
    assert any("UUID4" in w or "uuid" in w.lower() for w in result.warnings)


def test_validate_short_git_commit_warning():
    # 7-char short SHA is fine, other lengths are warned
    m = _valid_manifest(git_commit="abcde12")  # 7 chars → no warn
    result_7 = validate_submission(m)
    m2 = _valid_manifest(git_commit="abcde1234")  # 9 chars → warn
    result_9 = validate_submission(m2)
    assert result_7.valid
    # 9 chars should produce a warning but still be valid
    assert result_9.valid
    assert any("git_commit" in w for w in result_9.warnings)


# ---------------------------------------------------------------------------
# validate_submission_json
# ---------------------------------------------------------------------------


def test_validate_submission_json_valid():
    d = _valid_manifest().to_json()
    result = validate_submission_json(d)
    assert result.valid


def test_validate_submission_json_invalid():
    d = _valid_manifest().to_json()
    d["trace_pack"]["verified"] = False
    result = validate_submission_json(d)
    assert not result.valid


def test_validate_submission_json_missing_keys():
    result = validate_submission_json({})
    assert not result.valid


# ---------------------------------------------------------------------------
# CLI: stepback bench validate-submission
# ---------------------------------------------------------------------------


def test_cli_validate_submission_valid(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(_valid_manifest().to_json()), encoding="utf-8"
    )
    rc = cli_main(["bench", "validate-submission", str(manifest_path)])
    assert rc == 0


def test_cli_validate_submission_invalid_exit_1(tmp_path):
    m = _valid_manifest(verified=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(m.to_json()), encoding="utf-8")
    rc = cli_main(["bench", "validate-submission", str(manifest_path)])
    assert rc == 1


def test_cli_validate_submission_missing_file(tmp_path):
    rc = cli_main(["bench", "validate-submission", str(tmp_path / "nonexistent.json")])
    assert rc == 2


def test_cli_validate_submission_with_out(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    out_path = tmp_path / "result.json"
    manifest_path.write_text(
        json.dumps(_valid_manifest().to_json()), encoding="utf-8"
    )
    rc = cli_main(
        ["bench", "validate-submission", str(manifest_path), "--out", str(out_path)]
    )
    assert rc == 0
    assert out_path.exists()
    d = json.loads(out_path.read_text())
    assert d["valid"] is True
    assert d["rules_version"] == RULES_VERSION
    assert "errors" in d
    assert "warnings" in d
    assert "submission_id" in d
