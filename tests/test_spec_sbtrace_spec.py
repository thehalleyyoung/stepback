"""Tests for ``stepback.spec.SBTraceSpec`` (Step 23).

Covers:

* the built-in ``"1.0.0"`` spec is registered and exposed via
  :meth:`SBTraceSpec.for_wire_version` and :func:`current_spec`;
* JSON and YAML loading via :meth:`SBTraceSpec.load`, including overrides
  of supported capabilities and step kinds;
* ``from_mapping`` round-trips for fully-specified and minimal mappings;
* schema-level frame validators (wrapper, header, step, blob, frame
  dispatcher) emit stable :class:`ConformanceIssue` codes;
* validating a real, freshly recorded ``.sb`` file produces an empty,
  conformant report;
* hand-crafted broken traces (missing header, wrong format_version,
  missing tail, unknown frame type, signature scheme mismatch) all
  surface error-severity issues with the expected codes;
* :meth:`assert_conformant` accepts a single path, a glob pattern, and a
  sequence of paths, and raises :exc:`SBTraceConformanceError` carrying
  the report on its ``.report`` attribute when violations are found;
* mandatory-capability declarations the reader does not understand
  populate ``unsupported_capabilities`` and emit the
  ``capability.unsupported_mandatory`` error.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Iterable, List

import pytest

from stepback import current_spec, record
from stepback.canonical import canonical_json
from stepback.spec import (
    ConformanceIssue,
    ConformanceReport,
    SBTRACE_WIRE_VERSION,
    SBTraceConformanceError,
    SBTraceSpec,
    SBTraceVersionError,
)
from stepback.testing import run_recorded_agent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _record_demo_trace(tmp_path: Path, name: str = "demo.sb") -> Path:
    path = tmp_path / name
    with record(str(path)) as rec:
        run_recorded_agent(rec)
    return path


def _rewrite_frames(path: Path, frames: Iterable[dict]) -> None:
    """Write the given wrapper dicts back to ``path`` as raw .sb frames.

    No HMAC chain or signature is recomputed — the spec validator only
    looks at the schema, so this is enough to exercise the validators
    without touching the crypto code.
    """
    with open(path, "wb") as f:
        for wrapper in frames:
            body = canonical_json(wrapper)
            f.write(struct.pack(">I", len(body)))
            f.write(body)


def _read_wrappers(path: Path) -> List[dict]:
    from stepback.trace_reader import read_frames

    return read_frames(str(path))


def _codes(issues: Iterable[ConformanceIssue]) -> List[str]:
    return [i.code for i in issues]


# ---------------------------------------------------------------------------
# built-in spec
# ---------------------------------------------------------------------------


def test_current_spec_matches_wire_version() -> None:
    spec = current_spec()
    assert isinstance(spec, SBTraceSpec)
    assert spec.wire_version == SBTRACE_WIRE_VERSION
    assert spec.format_version == 1
    assert spec.encoding == "canonical-json"
    assert spec.magic == "stepback/.sb"


def test_for_wire_version_returns_singleton_spec() -> None:
    a = SBTraceSpec.for_wire_version("1.0.0")
    b = SBTraceSpec.for_wire_version("1.0.0")
    assert a is b  # cached/registered, not rebuilt
    assert "llm_call" in a.step_kinds
    assert "tool_call" in a.step_kinds


def test_for_wire_version_rejects_unknown() -> None:
    with pytest.raises(SBTraceVersionError):
        SBTraceSpec.for_wire_version("9.9.9")


def test_for_wire_version_rejects_malformed() -> None:
    with pytest.raises(SBTraceVersionError):
        SBTraceSpec.for_wire_version("not-a-version")


def test_spec_capability_set_includes_core() -> None:
    spec = current_spec()
    assert spec.supports_capability("core")
    assert spec.supports_capability("hmac-sha256-chain")
    assert not spec.supports_capability("post-quantum-receipts")


def test_unsupported_capabilities_returns_only_unknowns() -> None:
    spec = current_spec()
    declared = ["core", "post-quantum-receipts", "blobs", "future-feature"]
    assert spec.unsupported_capabilities(declared) == [
        "post-quantum-receipts",
        "future-feature",
    ]


# ---------------------------------------------------------------------------
# from_mapping / load
# ---------------------------------------------------------------------------


def test_from_mapping_inherits_built_in_defaults() -> None:
    spec = SBTraceSpec.from_mapping({"wire_version": "1.0.0"})
    builtin = SBTraceSpec.for_wire_version("1.0.0")
    assert spec.step_kinds == builtin.step_kinds
    assert spec.header_required == builtin.header_required


def test_from_mapping_can_override_capabilities_and_strict_flags() -> None:
    spec = SBTraceSpec.from_mapping(
        {
            "wire_version": "1.0.0",
            "supported_capabilities": ["core", "post-quantum-receipts"],
            "strict_unknown_step_kinds": True,
        }
    )
    assert spec.supports_capability("post-quantum-receipts")
    assert not spec.supports_capability("blobs")
    assert spec.strict_unknown_step_kinds is True


def test_from_mapping_requires_wire_version() -> None:
    with pytest.raises(SBTraceVersionError):
        SBTraceSpec.from_mapping({})


def test_from_mapping_rejects_non_iterable_collection_field() -> None:
    with pytest.raises(SBTraceVersionError):
        SBTraceSpec.from_mapping(
            {"wire_version": "1.0.0", "step_kinds": 123}
        )


def test_from_mapping_synthesises_minimal_spec_for_unknown_wire() -> None:
    spec = SBTraceSpec.from_mapping(
        {
            "wire_version": "1.99.0",
            "step_kinds": ["llm_call"],
            "supported_capabilities": ["core"],
        }
    )
    assert spec.wire_version == "1.99.0"
    assert spec.step_kinds == frozenset({"llm_call"})


def test_load_json_roundtrip(tmp_path: Path) -> None:
    payload = {
        "wire_version": "1.0.0",
        "supported_capabilities": ["core", "blobs"],
    }
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(payload))
    spec = SBTraceSpec.load(p)
    assert spec.wire_version == "1.0.0"
    assert spec.supports_capability("blobs")
    assert not spec.supports_capability("ed25519-receipts")


def test_load_yaml_roundtrip(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    payload = {
        "wire_version": "1.0.0",
        "supported_capabilities": ["core"],
    }
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(payload))
    spec = SBTraceSpec.load(p)
    assert spec.wire_version == "1.0.0"
    assert spec.supported_capabilities == frozenset({"core"})


def test_load_rejects_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.json"
    p.write_text("")
    with pytest.raises(SBTraceVersionError):
        SBTraceSpec.load(p)


def test_load_rejects_top_level_list(tmp_path: Path) -> None:
    p = tmp_path / "list.json"
    p.write_text("[1, 2, 3]")
    with pytest.raises(SBTraceVersionError):
        SBTraceSpec.load(p)


# ---------------------------------------------------------------------------
# frame-level validators
# ---------------------------------------------------------------------------


def test_validate_wrapper_flags_missing_fields_and_bad_signature_scheme() -> None:
    spec = current_spec()
    issues = spec.validate_wrapper(
        {"body": {}, "prev_hmac": "00", "hmac": "11", "sig": "rsa:abc"}
    )
    assert "wrapper.bad_signature_scheme" in _codes(issues)

    missing = spec.validate_wrapper({"body": {}})
    codes = _codes(missing)
    assert codes.count("wrapper.missing_field") == 3  # prev_hmac, hmac, sig


def test_validate_header_detects_format_version_skew() -> None:
    spec = current_spec()
    issues = spec.validate_header(
        {
            "type": "header",
            "magic": "stepback/.sb",
            "format_version": 2,
            "recorder_version": "0.1.0",
            "canonicalisation_version": "x",
            "public_key": "p",
            "hmac_key_id": "h",
            "price_list_version": "v",
            "wallclock_ns": 1,
        }
    )
    assert "header.format_version_mismatch" in _codes(issues)


def test_validate_header_detects_unknown_field_as_warning() -> None:
    spec = current_spec()
    issues = spec.validate_header(
        {
            "type": "header",
            "magic": "stepback/.sb",
            "format_version": 1,
            "recorder_version": "0.1.0",
            "canonicalisation_version": "x",
            "public_key": "p",
            "hmac_key_id": "h",
            "price_list_version": "v",
            "wallclock_ns": 1,
            "future_field": "tomorrow",
        }
    )
    unknown = [i for i in issues if i.code == "header.unknown_field"]
    assert unknown and unknown[0].severity == "warning"


def test_validate_step_unknown_kind_is_warning_by_default() -> None:
    spec = current_spec()
    issues = spec.validate_step({"step_id": "s1", "step_kind": "future_kind"})
    [issue] = [i for i in issues if i.code == "step.unknown_kind"]
    assert issue.severity == "warning"


def test_validate_step_unknown_kind_is_error_in_strict_mode() -> None:
    spec = SBTraceSpec.from_mapping(
        {"wire_version": "1.0.0", "strict_unknown_step_kinds": True}
    )
    issues = spec.validate_step({"step_id": "s1", "step_kind": "future_kind"})
    [issue] = [i for i in issues if i.code == "step.unknown_kind"]
    assert issue.severity == "error"


def test_validate_step_missing_required_field() -> None:
    spec = current_spec()
    issues = spec.validate_step({"step_kind": "llm_call"})  # no step_id
    assert "step.missing_field" in _codes(issues)


def test_validate_blob_flags_missing_fields() -> None:
    spec = current_spec()
    issues = spec.validate_blob({"type": "blob", "id": "deadbeef"})
    codes = _codes(issues)
    assert codes.count("blob.missing_field") == 2  # encoding, data


def test_validate_frame_routes_by_type() -> None:
    spec = current_spec()
    wrapper = {
        "body": {"type": "tail", "wallclock_ns": 1},
        "prev_hmac": "00",
        "hmac": "11",
        "sig": "ed25519:22",
    }
    assert spec.validate_frame(wrapper) == []

    bad = {
        "body": {"type": "header"},
        "prev_hmac": "00",
        "hmac": "11",
        "sig": "ed25519:22",
    }
    issues = spec.validate_frame(bad)
    assert any(i.code == "header.missing_field" for i in issues)


def test_validate_frame_unknown_kind() -> None:
    spec = current_spec()
    wrapper = {
        "body": {"type": "alien"},
        "prev_hmac": "00",
        "hmac": "11",
        "sig": "ed25519:22",
    }
    issues = spec.validate_frame(wrapper)
    assert "frame.unknown_kind" in _codes(issues)


# ---------------------------------------------------------------------------
# real-trace round-trip
# ---------------------------------------------------------------------------


def test_validate_trace_clean_on_real_recorded_run(tmp_path: Path) -> None:
    p = _record_demo_trace(tmp_path)
    spec = current_spec()
    report = spec.validate_trace(p)
    assert isinstance(report, ConformanceReport)
    assert report.is_conformant, report.summary()
    assert report.errors == []
    assert report.spec_version == "1.0.0"
    assert report.paths == [str(p)]


def test_validate_trace_detects_missing_tail(tmp_path: Path) -> None:
    p = _record_demo_trace(tmp_path)
    frames = _read_wrappers(p)
    _rewrite_frames(p, frames[:-1])  # strip the tail
    report = current_spec().validate_trace(p)
    assert "trace.missing_tail" in _codes(report.issues)
    assert not report.is_conformant


def test_validate_trace_detects_missing_header(tmp_path: Path) -> None:
    p = _record_demo_trace(tmp_path)
    frames = _read_wrappers(p)
    _rewrite_frames(p, frames[1:])  # strip the header
    report = current_spec().validate_trace(p)
    assert "trace.missing_header" in _codes(report.issues)


def test_validate_trace_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.sb"
    p.write_bytes(b"")
    report = current_spec().validate_trace(p)
    assert "trace.empty" in _codes(report.issues)


def test_validate_trace_truncated_file_reports_unreadable(tmp_path: Path) -> None:
    p = _record_demo_trace(tmp_path)
    raw = p.read_bytes()
    p.write_bytes(raw[: max(1, len(raw) // 2)])  # truncate
    report = current_spec().validate_trace(p)
    assert "trace.unreadable" in _codes(report.issues)


def test_validate_trace_format_version_mismatch_against_other_wire(
    tmp_path: Path,
) -> None:
    p = _record_demo_trace(tmp_path)
    spec_v2 = SBTraceSpec.from_mapping(
        {"wire_version": "1.0.0", "format_version": 2}
    )
    report = spec_v2.validate_trace(p)
    assert "trace.format_version_mismatch" in _codes(report.issues)


# ---------------------------------------------------------------------------
# capability negotiation
# ---------------------------------------------------------------------------


def test_unsupported_mandatory_capability_makes_trace_non_conformant(
    tmp_path: Path,
) -> None:
    p = _record_demo_trace(tmp_path)
    frames = _read_wrappers(p)
    cap_frame = {
        "body": {
            "type": "capability",
            "name": "post-quantum-receipts",
            "mandatory": True,
        },
        "prev_hmac": "00",
        "hmac": "11",
        "sig": "ed25519:22",
    }
    # Insert the capability frame between header and the first step.
    new_frames = [frames[0], cap_frame, *frames[1:]]
    _rewrite_frames(p, new_frames)
    report = current_spec().validate_trace(p)
    assert "post-quantum-receipts" in report.unsupported_capabilities
    assert "capability.unsupported_mandatory" in _codes(report.errors)
    assert not report.is_conformant


def test_optional_capability_frame_does_not_invalidate_trace(
    tmp_path: Path,
) -> None:
    p = _record_demo_trace(tmp_path)
    frames = _read_wrappers(p)
    cap_frame = {
        "body": {
            "type": "capability",
            "name": "post-quantum-receipts",
            "mandatory": False,
        },
        "prev_hmac": "00",
        "hmac": "11",
        "sig": "ed25519:22",
    }
    new_frames = [frames[0], cap_frame, *frames[1:]]
    _rewrite_frames(p, new_frames)
    report = current_spec().validate_trace(p)
    assert report.unsupported_capabilities == []
    assert not [
        i for i in report.errors if i.code == "capability.unsupported_mandatory"
    ]


# ---------------------------------------------------------------------------
# assert_conformant
# ---------------------------------------------------------------------------


def test_assert_conformant_accepts_single_path(tmp_path: Path) -> None:
    p = _record_demo_trace(tmp_path)
    report = current_spec().assert_conformant(str(p))
    assert report.is_conformant
    assert report.paths == [str(p)]


def test_assert_conformant_accepts_glob(tmp_path: Path) -> None:
    _record_demo_trace(tmp_path, "a.sb")
    _record_demo_trace(tmp_path, "b.sb")
    report = current_spec().assert_conformant(str(tmp_path / "*.sb"))
    assert report.is_conformant
    assert len(report.paths) == 2


def test_assert_conformant_accepts_sequence_of_paths(tmp_path: Path) -> None:
    a = _record_demo_trace(tmp_path, "a.sb")
    b = _record_demo_trace(tmp_path, "b.sb")
    report = current_spec().assert_conformant([str(a), str(b)])
    assert report.is_conformant
    assert set(report.paths) == {str(a), str(b)}


def test_assert_conformant_glob_with_no_matches_raises(tmp_path: Path) -> None:
    spec = current_spec()
    with pytest.raises(SBTraceConformanceError) as exc_info:
        spec.assert_conformant(str(tmp_path / "no_such_*.sb"))
    assert isinstance(exc_info.value.report, ConformanceReport)


def test_assert_conformant_raises_on_violation_with_report_attached(
    tmp_path: Path,
) -> None:
    p = _record_demo_trace(tmp_path)
    frames = _read_wrappers(p)
    _rewrite_frames(p, frames[:-1])  # strip tail -> error
    spec = current_spec()
    with pytest.raises(SBTraceConformanceError) as exc_info:
        spec.assert_conformant(str(p))
    err = exc_info.value
    assert isinstance(err.report, ConformanceReport)
    assert any(i.code == "trace.missing_tail" for i in err.report.errors)
    # The report's summary should mention the wire version + counts.
    assert "1.0.0" in err.report.summary()


# ---------------------------------------------------------------------------
# public-API surface
# ---------------------------------------------------------------------------


def test_public_api_exports_step23_names() -> None:
    import stepback

    for name in (
        "SBTraceSpec",
        "SBTraceConformanceError",
        "ConformanceIssue",
        "ConformanceReport",
        "current_spec",
    ):
        assert name in stepback.__all__, f"{name} must be in __all__"
        assert getattr(stepback, name, None) is not None
