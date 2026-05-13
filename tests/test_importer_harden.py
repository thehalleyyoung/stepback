"""Hardening tests for the LangSmith and OpenInference importers.

Covers:
  1. Real-fixture loading: import canonical fixture files from
     ``tests/fixtures/importers/`` and assert step counts, kind maps,
     verify+replay, and ``schema_version`` detection.
  2. Schema-version detection unit tests (v1 vs v2 LangSmith;
     dict vs otlp_any_value OpenInference).
  3. Hash-stability tests: importing the same fixture twice (with the
     same RecorderKey and fixed wallclock values) must produce
     identical ``inputs_hash`` and ``outputs_hash`` for every step.
  4. Cross-format hash equality: v1 and v2 LangSmith fixtures contain
     the same semantic runs; imported ``inputs_hash`` / ``outputs_hash``
     must be identical across both.  Likewise for OpenInference dict vs
     OTLP any-value attributes.
"""
from __future__ import annotations

import json
import os
import pathlib
import tempfile

import pytest

from stepback import (
    ImportReport,
    import_langsmith_jsonl,
    import_openinference_spans,
    replay,
)
from stepback.importers import (
    _detect_langsmith_schema_version,
    _detect_oi_attr_format,
    _read_langsmith_input,
)
from stepback.recorder import RecorderKey
from stepback.trace_reader import verify_trace


# ------------------------------------------------------------------ paths

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures" / "importers"

LS_V1 = FIXTURES_DIR / "langsmith_v1.jsonl"
LS_V2 = FIXTURES_DIR / "langsmith_v2.jsonl"
OI_V1 = FIXTURES_DIR / "openinference_v1.json"
OI_OTLP = FIXTURES_DIR / "openinference_otlp.json"


# ------------------------------------------------------------------ helpers

def _tmp(tmp_path, name: str) -> str:
    return os.path.join(str(tmp_path), name)


def _verify_and_replay(path: str, hmac_key: bytes) -> int:
    """Verify the .sb file and assert a no-sub replay is a pure cache hit."""
    info = verify_trace(path, hmac_key)
    assert info is not None
    t = replay(path)
    res = t.replay_forward()
    assert res.real_executions == 0, (
        f"expected pure cache hit, got real_executions={res.real_executions}"
    )
    return len(t.recorded_steps)


def _step_hashes(path: str) -> list[tuple[str, str, str]]:
    """Return (step_id, inputs_hash, outputs_hash) for every step in a .sb file."""
    t = replay(path)
    return [
        (s["step_id"], s["inputs_hash"], s["outputs_hash"])
        for s in t.recorded_steps
    ]


# ------------------------------------------------------------------ fixture files exist

class TestFixtureFilesExist:
    def test_langsmith_v1_exists(self):
        assert LS_V1.exists(), f"missing fixture: {LS_V1}"

    def test_langsmith_v2_exists(self):
        assert LS_V2.exists(), f"missing fixture: {LS_V2}"

    def test_openinference_v1_exists(self):
        assert OI_V1.exists(), f"missing fixture: {OI_V1}"

    def test_openinference_otlp_exists(self):
        assert OI_OTLP.exists(), f"missing fixture: {OI_OTLP}"


# ------------------------------------------------------------------ LangSmith fixture tests

class TestLangSmithFixtures:
    def test_v1_imports_correct_step_count(self, tmp_path):
        key = RecorderKey.fresh()
        report = import_langsmith_jsonl(str(LS_V1), _tmp(tmp_path, "out.sb"), key=key)
        assert isinstance(report, ImportReport)
        assert report.step_count == 5
        assert report.kind_counts["llm_call"] == 2
        assert report.kind_counts["tool_call"] == 2
        assert report.kind_counts["router"] == 1

    def test_v1_schema_version_is_v1(self, tmp_path):
        report = import_langsmith_jsonl(str(LS_V1), _tmp(tmp_path, "out.sb"))
        assert report.schema_version == "v1"

    def test_v1_as_dict_includes_schema_version(self, tmp_path):
        report = import_langsmith_jsonl(str(LS_V1), _tmp(tmp_path, "out.sb"))
        d = report.as_dict()
        assert "schema_version" in d
        assert d["schema_version"] == "v1"

    def test_v1_trace_is_valid_and_replayable(self, tmp_path):
        key = RecorderKey.fresh()
        out = _tmp(tmp_path, "out.sb")
        import_langsmith_jsonl(str(LS_V1), out, key=key)
        n = _verify_and_replay(out, key.hmac_key)
        assert n == 5

    def test_v2_imports_correct_step_count(self, tmp_path):
        key = RecorderKey.fresh()
        report = import_langsmith_jsonl(str(LS_V2), _tmp(tmp_path, "out.sb"), key=key)
        assert report.step_count == 5
        assert report.kind_counts["llm_call"] == 2
        assert report.kind_counts["tool_call"] == 2
        assert report.kind_counts["router"] == 1

    def test_v2_schema_version_is_v2(self, tmp_path):
        report = import_langsmith_jsonl(str(LS_V2), _tmp(tmp_path, "out.sb"))
        assert report.schema_version == "v2"

    def test_v2_trace_is_valid_and_replayable(self, tmp_path):
        key = RecorderKey.fresh()
        out = _tmp(tmp_path, "out.sb")
        import_langsmith_jsonl(str(LS_V2), out, key=key)
        n = _verify_and_replay(out, key.hmac_key)
        assert n == 5

    def test_json_array_input_is_accepted(self, tmp_path):
        """A JSON-array version of the v1 fixture must be parsed identically."""
        with open(str(LS_V1), "r") as f:
            runs = [json.loads(line) for line in f if line.strip()]
        array_path = _tmp(tmp_path, "runs_array.json")
        with open(array_path, "w") as f:
            json.dump(runs, f)
        key = RecorderKey.fresh()
        report = import_langsmith_jsonl(array_path, _tmp(tmp_path, "out.sb"), key=key)
        assert report.step_count == 5

    def test_root_is_written_before_children(self, tmp_path):
        """Parent step must appear before all children in the output trace."""
        out = _tmp(tmp_path, "out.sb")
        import_langsmith_jsonl(str(LS_V1), out)
        t = replay(out)
        seen = set()
        for s in t.recorded_steps:
            seen.add(s["step_id"])
            parent = s.get("parent_step_id")
            if parent is not None:
                assert parent in seen, (
                    f"step {s['step_id']} has parent {parent} not yet seen"
                )


# ------------------------------------------------------------------ OpenInference fixture tests

class TestOpenInferenceFixtures:
    def test_v1_dict_imports_correct_step_count(self, tmp_path):
        key = RecorderKey.fresh()
        report = import_openinference_spans(str(OI_V1), _tmp(tmp_path, "out.sb"), key=key)
        assert isinstance(report, ImportReport)
        assert report.step_count == 3
        assert report.kind_counts.get("llm_call") == 1
        assert report.kind_counts.get("tool_call") == 1
        assert report.kind_counts.get("router") == 1

    def test_v1_schema_version_is_dict(self, tmp_path):
        report = import_openinference_spans(str(OI_V1), _tmp(tmp_path, "out.sb"))
        assert report.schema_version == "dict"

    def test_v1_as_dict_includes_schema_version(self, tmp_path):
        report = import_openinference_spans(str(OI_V1), _tmp(tmp_path, "out.sb"))
        d = report.as_dict()
        assert "schema_version" in d
        assert d["schema_version"] == "dict"

    def test_v1_trace_is_valid_and_replayable(self, tmp_path):
        key = RecorderKey.fresh()
        out = _tmp(tmp_path, "out.sb")
        import_openinference_spans(str(OI_V1), out, key=key)
        n = _verify_and_replay(out, key.hmac_key)
        assert n == 3

    def test_v1_llm_messages_are_decoded(self, tmp_path):
        out = _tmp(tmp_path, "out.sb")
        import_openinference_spans(str(OI_V1), out)
        t = replay(out)
        llm = [s for s in t.recorded_steps if s["step_kind"] == "llm_call"][0]
        msgs = llm["llm_request"]["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "INV-9001" in msgs[1]["content"]

    def test_otlp_imports_correct_step_count(self, tmp_path):
        key = RecorderKey.fresh()
        report = import_openinference_spans(str(OI_OTLP), _tmp(tmp_path, "out.sb"), key=key)
        assert report.step_count == 3
        assert report.kind_counts.get("llm_call") == 1
        assert report.kind_counts.get("tool_call") == 1
        assert report.kind_counts.get("router") == 1

    def test_otlp_schema_version_is_otlp_resource_spans(self, tmp_path):
        report = import_openinference_spans(str(OI_OTLP), _tmp(tmp_path, "out.sb"))
        assert report.schema_version == "otlp_resource_spans"

    def test_otlp_trace_is_valid_and_replayable(self, tmp_path):
        key = RecorderKey.fresh()
        out = _tmp(tmp_path, "out.sb")
        import_openinference_spans(str(OI_OTLP), out, key=key)
        n = _verify_and_replay(out, key.hmac_key)
        assert n == 3

    def test_otlp_llm_messages_are_decoded(self, tmp_path):
        out = _tmp(tmp_path, "out.sb")
        import_openinference_spans(str(OI_OTLP), out)
        t = replay(out)
        llm = [s for s in t.recorded_steps if s["step_kind"] == "llm_call"][0]
        msgs = llm["llm_request"]["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "INV-9001" in msgs[1]["content"]


# ------------------------------------------------------------------ schema-version detection units

class TestSchemaVersionDetectionUnits:
    def test_langsmith_v1_detection_no_v2_fields(self):
        runs = [{"id": "1", "run_type": "llm", "inputs": {}, "outputs": {}}]
        assert _detect_langsmith_schema_version(runs) == "v1"

    def test_langsmith_v2_detection_execution_order(self):
        runs = [{"id": "1", "run_type": "llm", "execution_order": 1}]
        assert _detect_langsmith_schema_version(runs) == "v2"

    def test_langsmith_v2_detection_child_run_ids(self):
        runs = [{"id": "1", "child_run_ids": [], "run_type": "chain"}]
        assert _detect_langsmith_schema_version(runs) == "v2"

    def test_langsmith_v2_detection_feedback_stats(self):
        runs = [{"id": "1", "feedback_stats": {}, "run_type": "chain"}]
        assert _detect_langsmith_schema_version(runs) == "v2"

    def test_langsmith_v2_detection_app_path(self):
        runs = [{"id": "1", "app_path": "/o/org/projects/proj/r/1"}]
        assert _detect_langsmith_schema_version(runs) == "v2"

    def test_langsmith_v2_detection_manifest(self):
        runs = [{"id": "1", "manifest": None}]
        assert _detect_langsmith_schema_version(runs) == "v2"

    def test_langsmith_v2_detection_only_one_run_needs_field(self):
        # Only the last run has an execution_order; still v2.
        runs = [
            {"id": "1", "run_type": "chain"},
            {"id": "2", "run_type": "llm", "execution_order": 2},
        ]
        assert _detect_langsmith_schema_version(runs) == "v2"

    def test_langsmith_empty_list_returns_v1(self):
        assert _detect_langsmith_schema_version([]) == "v1"

    def test_oi_dict_attrs_returns_dict(self):
        spans = [{"span_id": "s1", "attributes": {"openinference.span.kind": "LLM"}}]
        assert _detect_oi_attr_format(spans) == "dict"

    def test_oi_otlp_any_value_attrs(self):
        spans = [
            {
                "span_id": "s1",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "LLM"}}
                ],
            }
        ]
        assert _detect_oi_attr_format(spans) == "otlp_any_value"

    def test_oi_empty_spans_returns_dict(self):
        assert _detect_oi_attr_format([]) == "dict"

    def test_oi_no_attributes_key_returns_dict(self):
        spans = [{"span_id": "s1", "name": "root"}]
        assert _detect_oi_attr_format(spans) == "dict"

    def test_read_langsmith_input_jsonl(self, tmp_path):
        path = _tmp(tmp_path, "runs.jsonl")
        with open(path, "w") as f:
            f.write('{"id": "1", "run_type": "llm"}\n')
            f.write('{"id": "2", "run_type": "tool"}\n')
        runs = _read_langsmith_input(path)
        assert len(runs) == 2
        assert runs[0]["id"] == "1"

    def test_read_langsmith_input_json_array(self, tmp_path):
        path = _tmp(tmp_path, "runs.json")
        with open(path, "w") as f:
            json.dump([{"id": "1", "run_type": "llm"}, {"id": "2"}], f)
        runs = _read_langsmith_input(path)
        assert len(runs) == 2
        assert runs[1]["id"] == "2"

    def test_read_langsmith_input_json_array_with_whitespace(self, tmp_path):
        """Leading whitespace before '[' must still be detected as array."""
        path = _tmp(tmp_path, "runs.json")
        with open(path, "w") as f:
            f.write("  \n  ")
            json.dump([{"id": "1"}], f)
        runs = _read_langsmith_input(path)
        assert len(runs) == 1


# ------------------------------------------------------------------ hash-stability tests

class TestHashStability:
    """Importing the same fixture twice with the same key must produce
    identical ``inputs_hash`` and ``outputs_hash`` for every step."""

    @pytest.mark.parametrize("fixture,importer", [
        (str(LS_V1), import_langsmith_jsonl),
        (str(LS_V2), import_langsmith_jsonl),
        (str(OI_V1), import_openinference_spans),
        (str(OI_OTLP), import_openinference_spans),
    ])
    def test_repeated_import_produces_identical_hashes(self, fixture, importer, tmp_path):
        key = RecorderKey.fresh()
        out1 = _tmp(tmp_path, "out1.sb")
        out2 = _tmp(tmp_path, "out2.sb")
        importer(fixture, out1, key=key)
        importer(fixture, out2, key=key)
        hashes1 = _step_hashes(out1)
        hashes2 = _step_hashes(out2)
        assert hashes1 == hashes2, (
            "importing the same fixture twice yielded different step hashes"
        )

    @pytest.mark.parametrize("fixture,importer", [
        (str(LS_V1), import_langsmith_jsonl),
        (str(LS_V2), import_langsmith_jsonl),
        (str(OI_V1), import_openinference_spans),
        (str(OI_OTLP), import_openinference_spans),
    ])
    def test_wallclock_does_not_affect_content_hashes(self, fixture, importer, tmp_path):
        """wallclock_ns is not part of inputs/outputs so must not appear in hashes."""
        key = RecorderKey.fresh()
        out1 = _tmp(tmp_path, "out1.sb")
        out2 = _tmp(tmp_path, "out2.sb")
        importer(fixture, out1, key=key)
        importer(fixture, out2, key=key)
        hashes1 = {sid: (ih, oh) for sid, ih, oh in _step_hashes(out1)}
        hashes2 = {sid: (ih, oh) for sid, ih, oh in _step_hashes(out2)}
        assert hashes1 == hashes2


# ------------------------------------------------------------------ cross-format hash equality

class TestCrossFormatHashEquality:
    """v1 and v2 LangSmith fixtures encode the same agent run; their
    step hashes must be identical (extra v2 fields must not perturb them).

    Likewise, OpenInference dict-style and OTLP any-value style for the
    same trace must produce identical step hashes.
    """

    def test_langsmith_v1_and_v2_have_same_hashes(self, tmp_path):
        key = RecorderKey.fresh()
        out_v1 = _tmp(tmp_path, "v1.sb")
        out_v2 = _tmp(tmp_path, "v2.sb")
        import_langsmith_jsonl(str(LS_V1), out_v1, key=key)
        import_langsmith_jsonl(str(LS_V2), out_v2, key=key)
        hashes_v1 = _step_hashes(out_v1)
        hashes_v2 = _step_hashes(out_v2)
        assert len(hashes_v1) == len(hashes_v2), (
            f"step count mismatch: v1={len(hashes_v1)}, v2={len(hashes_v2)}"
        )
        for (sid1, ih1, oh1), (sid2, ih2, oh2) in zip(hashes_v1, hashes_v2):
            assert sid1 == sid2, f"step_id mismatch: {sid1!r} vs {sid2!r}"
            assert ih1 == ih2, (
                f"inputs_hash differs for {sid1}: {ih1!r} vs {ih2!r}"
            )
            assert oh1 == oh2, (
                f"outputs_hash differs for {sid1}: {oh1!r} vs {oh2!r}"
            )

    def test_openinference_dict_and_otlp_have_same_hashes(self, tmp_path):
        key = RecorderKey.fresh()
        out_dict = _tmp(tmp_path, "dict.sb")
        out_otlp = _tmp(tmp_path, "otlp.sb")
        import_openinference_spans(str(OI_V1), out_dict, key=key)
        import_openinference_spans(str(OI_OTLP), out_otlp, key=key)
        hashes_dict = _step_hashes(out_dict)
        hashes_otlp = _step_hashes(out_otlp)
        assert len(hashes_dict) == len(hashes_otlp), (
            f"step count mismatch: dict={len(hashes_dict)}, otlp={len(hashes_otlp)}"
        )
        for (sid1, ih1, oh1), (sid2, ih2, oh2) in zip(hashes_dict, hashes_otlp):
            assert sid1 == sid2
            assert ih1 == ih2, (
                f"inputs_hash differs for {sid1}: {ih1!r} vs {ih2!r} "
                f"(dict vs OTLP attribute encoding)"
            )
            assert oh1 == oh2, (
                f"outputs_hash differs for {sid1}: {oh1!r} vs {oh2!r}"
            )
