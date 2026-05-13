"""Step 108: JSON export — stable schema and compatibility tests.

This module verifies the ``stepback_native_json_v1`` schema and the
structural-validation helpers that enforce it.  Four categories of test:

1. **Schema file** — the spec/schema/v1/native_json_export.json document
   exists, is valid JSON, is valid Draft-07, and contains the expected
   mandatory keywords.

2. **Positive structural validation** — :func:`validate_native_json_doc`
   accepts well-formed documents and :func:`export_native_json` emits a
   payload that passes validation.

3. **Negative structural validation** — :func:`validate_native_json_doc`
   raises the appropriate exception class for every kind of violation
   (missing required key, wrong type, etc.).

4. **Backward-compatibility** — the frozen golden fixture in
   ``tests/fixtures/importers/native_json_v1_golden.json`` is accepted
   by both the validator and :func:`import_native_json`, ensuring that
   a document produced by a past stepback version can still be round-
   tripped by the current one.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest

# Third-party import only used for Draft-07 meta-schema check; present
# in the dev extras. Skip the Draft-07 test gracefully when not installed.
jsonschema = pytest.importorskip(
    "jsonschema",
    reason="jsonschema not installed; add it to dev extras to run Draft-07 checks",
)

from stepback import (
    ExportError,
    export_native_json,
    import_native_json,
    validate_native_json_doc,
)
from stepback.importers import ImportError as SBImportError
from stepback.recorder import RecorderKey
from stepback.testing import run_recorded_agent
from stepback import record
from stepback.trace_reader import verify_trace


# ------------------------------------------------------------------ helpers


REPO_ROOT = Path(__file__).parent.parent
SCHEMA_PATH = REPO_ROOT / "spec" / "schema" / "v1" / "native_json_export.json"
GOLDEN_PATH = (
    Path(__file__).parent / "fixtures" / "importers" / "native_json_v1_golden.json"
)


def _load_schema() -> Dict[str, Any]:
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _record_fixture(tmp_path) -> tuple:
    sb_path = str(tmp_path / "fixture.sb")
    key = RecorderKey.fresh()
    with record(sb_path, key=key) as rec:
        run_recorded_agent(rec)
    return sb_path, key


def _read_steps(sb_path: str, key: RecorderKey):
    return verify_trace(sb_path, key.hmac_key).steps


# ================================================================== §1 schema file


def test_schema_file_exists():
    assert SCHEMA_PATH.exists(), (
        f"Schema file not found at {SCHEMA_PATH}. "
        "Run step 108 to generate it."
    )


def test_schema_file_is_valid_json():
    schema = _load_schema()
    assert isinstance(schema, dict)


def test_schema_file_draft07_meta_schema():
    """The schema document is a valid JSON Schema Draft-07."""
    schema = _load_schema()
    jsonschema.Draft7Validator.check_schema(schema)


def test_schema_title_and_id():
    schema = _load_schema()
    assert schema.get("title") == "stepback_native_json_v1"
    assert "$schema" in schema
    assert "draft-07" in schema["$schema"]


def test_schema_requires_format_and_steps():
    schema = _load_schema()
    required = schema.get("required", [])
    assert "format" in required
    assert "steps" in required


def test_schema_format_const():
    schema = _load_schema()
    fmt_prop = schema["properties"]["format"]
    assert fmt_prop.get("const") == "stepback_native_json_v1"


def test_schema_steps_items_ref():
    schema = _load_schema()
    steps_prop = schema["properties"]["steps"]
    assert steps_prop["type"] == "array"
    # Items must reference the step definition.
    assert "$ref" in steps_prop.get("items", {})


def test_schema_step_definition_has_required_fields():
    schema = _load_schema()
    step_def = schema["definitions"]["step"]
    assert "step_id" in step_def.get("required", [])
    assert "step_kind" in step_def.get("required", [])


def test_schema_step_kind_is_string_not_enum():
    """step_kind must be ``type: string`` (forward-compatible), not an enum."""
    schema = _load_schema()
    step_def = schema["definitions"]["step"]
    sk = step_def["properties"]["step_kind"]
    assert sk.get("type") == "string", (
        "step_kind must be 'type: string' for forward compatibility; "
        "use x-knownValues for documentation only."
    )
    assert "enum" not in sk, (
        "step_kind must NOT use 'enum' — unknown future step kinds must be tolerated."
    )


def test_schema_additional_properties_allowed():
    """Top-level and step objects must allow extra fields for forward compat."""
    schema = _load_schema()
    assert schema.get("additionalProperties") is not False
    step_def = schema["definitions"]["step"]
    assert step_def.get("additionalProperties") is not False


# ================================================================== §2 positive validation


def test_validate_accepts_minimal_valid_doc():
    doc = {"format": "stepback_native_json_v1", "steps": []}
    validate_native_json_doc(doc)  # must not raise


def test_validate_accepts_doc_with_header():
    doc = {
        "format": "stepback_native_json_v1",
        "header": {"recorder_version": "stepback/0.9.0"},
        "steps": [],
    }
    validate_native_json_doc(doc)


def test_validate_accepts_doc_with_empty_header():
    doc = {"format": "stepback_native_json_v1", "header": {}, "steps": []}
    validate_native_json_doc(doc)


def test_validate_accepts_step_with_known_kind():
    step = {"step_id": "01A", "step_kind": "llm_call"}
    doc = {"format": "stepback_native_json_v1", "steps": [step]}
    validate_native_json_doc(doc)


def test_validate_accepts_step_with_unknown_kind():
    """Unknown step_kinds must be tolerated (forward compatibility)."""
    step = {"step_id": "01A", "step_kind": "future_kind_v99"}
    doc = {"format": "stepback_native_json_v1", "steps": [step]}
    validate_native_json_doc(doc)


def test_validate_accepts_step_with_extra_fields():
    step = {
        "step_id": "01A",
        "step_kind": "tool_call",
        "tool_name": "my_tool",
        "future_extension": {"some": "data"},
    }
    doc = {"format": "stepback_native_json_v1", "steps": [step]}
    validate_native_json_doc(doc)


def test_validate_accepts_real_export(tmp_path):
    """A real export_native_json output passes validate_native_json_doc."""
    sb_path, key = _record_fixture(tmp_path)
    steps = _read_steps(sb_path, key)
    out = str(tmp_path / "trace.json")
    export_native_json(steps, out)
    with open(out, encoding="utf-8") as f:
        doc = json.load(f)
    validate_native_json_doc(doc)  # must not raise


def test_validate_accepts_real_export_via_jsonschema_draft07(tmp_path):
    """The real export output also passes Draft-07 validation."""
    schema = _load_schema()
    validator = jsonschema.Draft7Validator(schema)

    sb_path, key = _record_fixture(tmp_path)
    steps = _read_steps(sb_path, key)
    out = str(tmp_path / "trace.json")
    export_native_json(steps, out)
    with open(out, encoding="utf-8") as f:
        doc = json.load(f)

    errors = list(validator.iter_errors(doc))
    assert not errors, f"Draft-07 validation errors: {errors}"


# ================================================================== §3 negative validation


def test_validate_rejects_non_dict_top_level():
    with pytest.raises(ExportError):
        validate_native_json_doc([1, 2, 3])


def test_validate_rejects_string_top_level():
    with pytest.raises(ExportError):
        validate_native_json_doc("stepback_native_json_v1")  # type: ignore[arg-type]


def test_validate_rejects_missing_format():
    with pytest.raises(ExportError):
        validate_native_json_doc({"steps": []})


def test_validate_rejects_wrong_format_tag():
    with pytest.raises(ExportError):
        validate_native_json_doc({"format": "not_us", "steps": []})


def test_validate_rejects_none_format():
    with pytest.raises(ExportError):
        validate_native_json_doc({"format": None, "steps": []})


def test_validate_rejects_missing_steps():
    with pytest.raises(ExportError):
        validate_native_json_doc({"format": "stepback_native_json_v1"})


def test_validate_rejects_non_list_steps():
    with pytest.raises(ExportError):
        validate_native_json_doc(
            {"format": "stepback_native_json_v1", "steps": {"0": "oops"}}
        )


def test_validate_rejects_non_dict_header():
    with pytest.raises(ExportError):
        validate_native_json_doc(
            {
                "format": "stepback_native_json_v1",
                "header": "bad",
                "steps": [],
            }
        )


def test_validate_rejects_list_header():
    with pytest.raises(ExportError):
        validate_native_json_doc(
            {
                "format": "stepback_native_json_v1",
                "header": ["bad"],
                "steps": [],
            }
        )


def test_validate_rejects_non_dict_step():
    with pytest.raises(ExportError):
        validate_native_json_doc(
            {
                "format": "stepback_native_json_v1",
                "steps": ["not_a_dict"],
            }
        )


def test_validate_rejects_step_missing_step_id():
    step = {"step_kind": "llm_call"}
    with pytest.raises(ExportError):
        validate_native_json_doc(
            {"format": "stepback_native_json_v1", "steps": [step]}
        )


def test_validate_rejects_step_missing_step_kind():
    step = {"step_id": "01A"}
    with pytest.raises(ExportError):
        validate_native_json_doc(
            {"format": "stepback_native_json_v1", "steps": [step]}
        )


def test_validate_rejects_non_string_step_id():
    step = {"step_id": 42, "step_kind": "llm_call"}
    with pytest.raises(ExportError):
        validate_native_json_doc(
            {"format": "stepback_native_json_v1", "steps": [step]}
        )


def test_validate_rejects_non_string_step_kind():
    step = {"step_id": "01A", "step_kind": 99}
    with pytest.raises(ExportError):
        validate_native_json_doc(
            {"format": "stepback_native_json_v1", "steps": [step]}
        )


def test_validate_exc_class_propagates_to_import_error():
    """Using exc_class=ImportError raises ImportError, not ExportError."""
    with pytest.raises(SBImportError):
        validate_native_json_doc(
            {"format": "stepback_native_json_v1", "steps": "bad"},
            exc_class=SBImportError,
        )


# ================================================================== §4 backward compatibility


def test_golden_fixture_file_exists():
    assert GOLDEN_PATH.exists(), (
        f"Golden fixture not found at {GOLDEN_PATH}"
    )


def test_golden_fixture_is_valid_json():
    with GOLDEN_PATH.open(encoding="utf-8") as f:
        doc = json.load(f)
    assert isinstance(doc, dict)


def test_golden_fixture_passes_structural_validation():
    """The frozen golden v1 document passes validate_native_json_doc."""
    with GOLDEN_PATH.open(encoding="utf-8") as f:
        doc = json.load(f)
    validate_native_json_doc(doc)  # must not raise


def test_golden_fixture_passes_draft07_validation():
    """The frozen golden v1 document conforms to the JSON Schema Draft-07."""
    schema = _load_schema()
    validator = jsonschema.Draft7Validator(schema)
    with GOLDEN_PATH.open(encoding="utf-8") as f:
        doc = json.load(f)
    errors = list(validator.iter_errors(doc))
    assert not errors, f"Draft-07 validation errors on golden fixture: {errors}"


def test_golden_fixture_can_be_imported(tmp_path):
    """import_native_json accepts the frozen golden v1 fixture."""
    sb_out = str(tmp_path / "from_golden.sb")
    key = RecorderKey.fresh()
    rep = import_native_json(str(GOLDEN_PATH), sb_out, key=key)
    assert rep.step_count == 2
    assert rep.kind_counts.get("llm_call") == 1
    assert rep.kind_counts.get("tool_call") == 1
    steps = _read_steps(sb_out, key)
    assert len(steps) == 2
    assert steps[0]["step_kind"] == "llm_call"
    assert steps[1]["step_kind"] == "tool_call"


def test_golden_fixture_step_shapes():
    """Each step in the golden fixture has the expected structural shape."""
    with GOLDEN_PATH.open(encoding="utf-8") as f:
        doc = json.load(f)
    steps = doc["steps"]
    assert len(steps) == 2

    llm = steps[0]
    assert llm["step_kind"] == "llm_call"
    assert isinstance(llm["step_id"], str)
    assert llm["parent_step_id"] is None
    assert "llm_request" in llm
    assert "llm_response" in llm
    assert "inputs_hash" in llm
    assert "outputs_hash" in llm

    tool = steps[1]
    assert tool["step_kind"] == "tool_call"
    assert isinstance(tool["step_id"], str)
    assert tool["parent_step_id"] == llm["step_id"]
    assert "tool_name" in tool


def test_golden_fixture_format_tag_stable():
    """The format tag must be exactly 'stepback_native_json_v1' forever."""
    with GOLDEN_PATH.open(encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["format"] == "stepback_native_json_v1"


def test_current_export_produces_v1_compatible_schema(tmp_path):
    """A live export round-trips the golden shape requirements.

    Records a trace with the current exporter and confirms the output
    satisfies the v1 structural invariants that the golden fixture tests
    nail down — ensuring backward compatibility in both directions.
    """
    sb_path, key = _record_fixture(tmp_path)
    steps = _read_steps(sb_path, key)
    out = str(tmp_path / "live.json")
    export_native_json(steps, out)
    with open(out, encoding="utf-8") as f:
        doc = json.load(f)

    # Top-level shape.
    assert doc["format"] == "stepback_native_json_v1"
    assert isinstance(doc["steps"], list)
    assert len(doc["steps"]) > 0

    # Every step has the required fields with the right types.
    for s in doc["steps"]:
        assert isinstance(s, dict)
        assert isinstance(s["step_id"], str) and s["step_id"]
        assert isinstance(s["step_kind"], str) and s["step_kind"]

    # LLM steps carry request and response.
    llm_steps = [s for s in doc["steps"] if s["step_kind"] == "llm_call"]
    assert llm_steps, "Expected at least one llm_call in the fixture agent trace"
    for s in llm_steps:
        assert "llm_request" in s
        assert "llm_response" in s
        assert isinstance(s["llm_request"].get("model"), str)

    # Tool steps carry a name.
    tool_steps = [s for s in doc["steps"] if s["step_kind"] == "tool_call"]
    assert tool_steps, "Expected at least one tool_call in the fixture agent trace"
    for s in tool_steps:
        # Tool steps carry their identifier in either `tool_name` or `name`.
        assert "tool_name" in s or "name" in s
