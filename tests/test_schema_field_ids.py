"""Cross-checks for the machine-readable schema in ``spec/schema/``.

These tests guard the contract that ``spec/schema/v1/*.json`` and the
runtime constants in :mod:`stepback.spec` and
:class:`stepback.step_types.StepKind` are kept in lock-step. Step 45 of
``docs/100_STEPS.md`` introduces the schema files; this test makes the
schema the single source of truth and turns drift into a CI failure.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from stepback.spec import (
    _FRAME_KINDS_V1,
    _HEADER_OPTIONAL_V1,
    _HEADER_REQUIRED_V1,
    _STEP_KINDS_V1,
    _STEP_OPTIONAL_V1,
    _STEP_REQUIRED_V1,
    _SUPPORTED_CAPABILITIES_V1,
    _WRAPPER_REQUIRED,
)
from stepback.step_types import StepKind


REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = REPO_ROOT / "spec" / "schema" / "v1"


def _load(rel: str) -> dict:
    return json.loads((SCHEMA_DIR / rel).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Index sanity
# ---------------------------------------------------------------------------


def test_schema_dir_layout() -> None:
    assert SCHEMA_DIR.is_dir(), f"missing {SCHEMA_DIR}"
    assert (SCHEMA_DIR / "index.json").is_file()
    assert (SCHEMA_DIR / "extension_ranges.json").is_file()
    assert (SCHEMA_DIR / "wrapper.json").is_file()
    for name in ("header", "capability", "step", "blob", "tail"):
        assert (SCHEMA_DIR / "frames" / f"{name}.json").is_file(), name
    for name in StepKind.known_values():
        assert (SCHEMA_DIR / "step_kinds" / f"{name}.json").is_file(), name


def test_index_lists_every_schema_file() -> None:
    index = _load("index.json")
    assert index["wire_version"] == "1.0.0"
    assert index["format_version"] == 1
    assert index["encoding"] == "canonical-json"
    assert index["magic"] == "stepback/.sb"
    # Every referenced file resolves on disk.
    for rel in [index["extension_ranges"], index["wrapper"]]:
        assert (SCHEMA_DIR / rel).is_file(), rel
    for rel in index["frames"].values():
        assert (SCHEMA_DIR / rel).is_file(), rel
    for rel in index["step_kinds"].values():
        assert (SCHEMA_DIR / rel).is_file(), rel
    # Frame kind keys match the runtime constant.
    assert set(index["frames"]) == set(_FRAME_KINDS_V1)
    # Step kind keys match StepKind.
    assert set(index["step_kinds"]) == StepKind.known_values()
    # Registered capability names match the runtime allow-list.
    assert {c["name"] for c in index["registered_capabilities"]} == set(
        _SUPPORTED_CAPABILITIES_V1
    )


# ---------------------------------------------------------------------------
# Extension ranges
# ---------------------------------------------------------------------------


EXPECTED_RANGE_CLASSES = ("core", "registered-extension", "experimental", "reserved")


def _check_ranges(ranges: list[dict], *, where: str) -> None:
    classes = [r["class"] for r in ranges]
    assert classes == list(EXPECTED_RANGE_CLASSES), (
        f"{where}: classes must be {EXPECTED_RANGE_CLASSES}, got {classes}"
    )
    # Contiguous, non-overlapping, covering 1..65535.
    cursor = 1
    for r in ranges:
        assert r["lo"] == cursor, f"{where}: gap at {cursor}, range starts at {r['lo']}"
        assert r["hi"] >= r["lo"], where
        cursor = r["hi"] + 1
    assert cursor == 65536, f"{where}: ranges do not extend to 65535 ({cursor - 1})"


def test_extension_ranges_global() -> None:
    data = _load("extension_ranges.json")
    assert data["wire_version"] == "1.0.0"
    assert data["max_id"] == 65535
    assert any(f["id"] == 0 for f in data["forbidden_ids"])
    _check_ranges(data["ranges"], where="extension_ranges.json")


@pytest.mark.parametrize(
    "rel",
    [
        "wrapper.json",
        "frames/header.json",
        "frames/capability.json",
        "frames/step.json",
        "frames/blob.json",
        "frames/tail.json",
        "step_kinds/llm_call.json",
        "step_kinds/tool_call.json",
        "step_kinds/router.json",
        "step_kinds/policy_check.json",
        "step_kinds/mcp_call.json",
        "step_kinds/parallel_branch_open.json",
        "step_kinds/parallel_branch_join.json",
        "step_kinds/exception.json",
    ],
)
def test_extension_ranges_per_schema(rel: str) -> None:
    schema = _load(rel)
    _check_ranges(schema["extension_ranges"], where=rel)


# ---------------------------------------------------------------------------
# Field-id uniqueness and range membership
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel",
    [
        "wrapper.json",
        "frames/header.json",
        "frames/capability.json",
        "frames/step.json",
        "frames/blob.json",
        "frames/tail.json",
    ],
)
def test_field_ids_are_unique_and_in_core_range(rel: str) -> None:
    schema = _load(rel)
    fields = schema["fields"]
    ids = [f["id"] for f in fields]
    assert len(set(ids)) == len(ids), f"{rel}: duplicate ids {ids}"
    names = [f["name"] for f in fields]
    assert len(set(names)) == len(names), f"{rel}: duplicate names {names}"
    # Every in-spec field must live in the core range [1, 99] (per the
    # README): registered-extension ids belong to a capability frame, not
    # to the spec's base schemas.
    for fid in ids:
        assert 1 <= fid <= 99, f"{rel}: field id {fid} outside core range 1..99"


# ---------------------------------------------------------------------------
# Cross-checks against stepback.spec runtime constants
# ---------------------------------------------------------------------------


def _required_names(schema: dict) -> set[str]:
    return {f["name"] for f in schema["fields"] if f["required"]}


def _optional_names(schema: dict) -> set[str]:
    return {f["name"] for f in schema["fields"] if not f["required"]}


def test_wrapper_required_matches_runtime() -> None:
    schema = _load("wrapper.json")
    assert _required_names(schema) == set(_WRAPPER_REQUIRED)


def test_header_fields_match_runtime() -> None:
    schema = _load("frames/header.json")
    # Drop v2-only fields when comparing to the v1 runtime constant.
    v1_only_required = {
        f["name"]
        for f in schema["fields"]
        if f["required"] and f.get("since", "1.0.0") == "1.0.0"
    }
    v1_only_optional = {
        f["name"]
        for f in schema["fields"]
        if not f["required"] and f.get("since", "1.0.0") == "1.0.0"
    }
    assert v1_only_required == set(_HEADER_REQUIRED_V1)
    assert v1_only_optional == set(_HEADER_OPTIONAL_V1)
    # Type is restricted to "header" via enum.
    type_field = next(f for f in schema["fields"] if f["name"] == "type")
    assert type_field["enum"] == ["header"]
    # v2-only fields are present but never in the v1 runtime sets above.
    v2_field_names = {
        f["name"] for f in schema["fields"] if f.get("since") == "2.0.0"
    }
    assert v2_field_names.isdisjoint(set(_HEADER_REQUIRED_V1))
    assert v2_field_names.isdisjoint(set(_HEADER_OPTIONAL_V1))


def test_step_body_fields_match_runtime() -> None:
    schema = _load("frames/step.json")
    # The step body schema mixes the frame envelope ids (1-9) with the
    # step body ids (>= 10); the runtime constants only describe the
    # body keys, so partition before comparing.
    body_fields = [f for f in schema["fields"] if f["id"] >= 10]
    body_required = {f["name"] for f in body_fields if f["required"]}
    body_optional = {f["name"] for f in body_fields if not f["required"]}
    assert body_required == set(_STEP_REQUIRED_V1)
    assert body_optional == set(_STEP_OPTIONAL_V1)
    # The step_kind enum mirrors the closed StepKind set.
    kind_field = next(f for f in body_fields if f["name"] == "step_kind")
    assert set(kind_field["enum"]) == StepKind.known_values()


def test_step_kinds_match_runtime() -> None:
    index = _load("index.json")
    assert set(index["step_kinds"]) == set(_STEP_KINDS_V1)
    # And every per-step-kind file declares its own step_kind correctly.
    for kind, rel in index["step_kinds"].items():
        schema = _load(rel)
        assert schema["step_kind"] == kind, rel
        # The fixed_value specialisation must agree.
        spec_field = next(
            s for s in schema["specializations"] if s["field_id"] == 11
        )
        assert spec_field["fixed_value"] == kind, rel


def test_frame_kinds_match_runtime() -> None:
    index = _load("index.json")
    assert set(index["frames"]) == set(_FRAME_KINDS_V1)


def test_specialization_field_ids_exist_in_step_schema() -> None:
    """Every per-step-kind specialization references a real step field id."""
    step_schema = _load("frames/step.json")
    valid_ids = {f["id"] for f in step_schema["fields"]}
    index = _load("index.json")
    for kind, rel in index["step_kinds"].items():
        schema = _load(rel)
        for spec in schema["specializations"]:
            assert spec["field_id"] in valid_ids, (
                f"{rel}: specialization references unknown field id "
                f"{spec['field_id']} (step kind {kind})"
            )
