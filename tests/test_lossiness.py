"""Tests for lossiness reports in importers and exporters (step 112).

Verifies that:
1. LossReport is accessible from the public API.
2. Every importer populates at least one lossiness category.
3. Every foreign-format exporter populates at least one lossiness category.
4. import_native_json / export_native_json are lossless.
5. LossReport.as_dict() round-trips correctly.
6. LossReport.is_lossless() works correctly.
7. ImportReport.as_dict() and ExportReport.as_dict() include 'lossiness'.
8. LangSmith v2 detection adds extra dropped fields.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import List

import pytest

import stepback
from stepback import (
    ImportReport,
    ExportReport,
    LossReport,
    import_openai_chat_log,
    import_langsmith_jsonl,
    import_openinference_spans,
    import_otel_spans,
    import_phoenix_spans,
    import_helicone_log,
    import_langfuse_export,
    import_datadog_apm,
    import_native_json,
    export_openai_chat_log,
    export_langsmith_jsonl,
    export_openinference_spans,
    export_otel_spans,
    export_native_json,
    export_html_view,
    export_cyclonedx_ai,
    record,
)


# ---------------------------------------------------------- helpers


def _tmp(tmp_path: str, name: str) -> str:
    return os.path.join(tmp_path, name)


def _write_json(path: str, obj: object) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    return path


def _openai_log() -> List[dict]:
    return [
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "response": {
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        }
    ]


def _langsmith_v1_runs() -> List[dict]:
    return [
        {"id": "r1", "parent_run_id": None, "run_type": "llm",
         "name": "gpt-4o", "inputs": {"messages": [{"role": "user", "content": "hi"}]},
         "outputs": {"generations": [[{"text": "hello"}]]},
         "extra": {"invocation_params": {"model": "gpt-4o", "temperature": 0.0},
                   "token_usage": {"prompt_tokens": 3, "completion_tokens": 2}}},
    ]


def _langsmith_v2_runs() -> List[dict]:
    runs = _langsmith_v1_runs()
    runs[0]["feedback_stats"] = {"score": 1.0}  # v2-era field
    runs[0]["app_path"] = "/app/trace/123"
    return runs


def _oi_spans() -> List[dict]:
    return [
        {"span_id": "span1", "parent_span_id": None, "name": "llm",
         "attributes": {"openinference.span.kind": "LLM", "llm.model_name": "gpt-4o",
                        "llm.token_count.prompt": 3, "llm.token_count.completion": 2,
                        "llm.token_count.total": 5}},
    ]


def _otel_spans() -> List[dict]:
    return [
        {"span_id": "sp1", "parent_span_id": None, "name": "llm",
         "attributes": [
             {"key": "agent.step.kind", "value": {"stringValue": "LLM_CALL"}},
             {"key": "gen_ai.request.model", "value": {"stringValue": "gpt-4o"}},
             {"key": "gen_ai.usage.input_tokens", "value": {"intValue": 3}},
             {"key": "gen_ai.usage.output_tokens", "value": {"intValue": 2}},
         ]},
    ]


def _phoenix_spans() -> List[dict]:
    return [
        {"span_id": "px1", "parent_span_id": None, "name": "llm",
         "attributes": {"openinference.span.kind": "LLM", "llm.model_name": "gpt-4o",
                        "llm.token_count.prompt": 3, "llm.token_count.completion": 2,
                        "llm.token_count.total": 5}},
    ]


def _helicone_log() -> List[dict]:
    return [
        {"request_id": "hid1",
         "request": {"request_body": {"model": "gpt-4o",
                                      "messages": [{"role": "user", "content": "hi"}]}},
         "response": {"body": {"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                               "usage": {"prompt_tokens": 3, "completion_tokens": 2}}}},
    ]


def _langfuse_traces() -> List[dict]:
    return [
        {"id": "t1", "name": "trace",
         "observations": [
             {"id": "o1", "parentObservationId": None, "type": "GENERATION",
              "name": "gpt-4o", "model": "gpt-4o",
              "input": {"messages": [{"role": "user", "content": "hi"}]},
              "output": {"content": "hello"},
              "usage": {"input": 3, "output": 2, "total": 5}},
         ]},
    ]


def _datadog_spans() -> List[dict]:
    return [
        {"span_id": "dd1", "parent_id": "0", "name": "llm.call", "type": "llm",
         "start": 1700000000000000000,
         "meta": {"ai.model.name": "gpt-4o",
                  "openai.request.messages.0.content": "hi",
                  "openai.request.messages.0.role": "user",
                  "openai.response.completions.0.content": "hello",
                  "openai.response.completions.0.role": "assistant"},
         "metrics": {"openai.response.usage.prompt_tokens": 3,
                     "openai.response.usage.completion_tokens": 2}},
    ]


# ---------------------------------------------------------- LossReport unit tests


class TestLossReport:
    def test_defaults_empty(self):
        lr = LossReport()
        assert lr.absent == []
        assert lr.approximated == []
        assert lr.synthesized == []
        assert lr.dropped == []

    def test_is_lossless_when_empty(self):
        assert LossReport().is_lossless()

    def test_is_not_lossless_when_absent(self):
        lr = LossReport(absent=["a: missing"])
        assert not lr.is_lossless()

    def test_is_not_lossless_when_synthesized(self):
        lr = LossReport(synthesized=["x: invented"])
        assert not lr.is_lossless()

    def test_as_dict_round_trip(self):
        lr = LossReport(
            absent=["a: absent"],
            approximated=["b: approx"],
            synthesized=["c: synth"],
            dropped=["d: dropped"],
        )
        d = lr.as_dict()
        assert d["absent"] == ["a: absent"]
        assert d["approximated"] == ["b: approx"]
        assert d["synthesized"] == ["c: synth"]
        assert d["dropped"] == ["d: dropped"]

    def test_as_dict_returns_copies(self):
        lr = LossReport(absent=["x"])
        d = lr.as_dict()
        d["absent"].append("y")
        assert lr.absent == ["x"]  # original unaffected

    def test_public_api(self):
        assert hasattr(stepback, "LossReport")
        assert stepback.LossReport is LossReport


# ---------------------------------------------------------- ImportReport


class TestImportReportLossiness:
    def test_import_report_has_lossiness_field(self, tmp_path):
        inp = _write_json(str(tmp_path / "in.json"), _openai_log())
        out = str(tmp_path / "out.sb")
        report = import_openai_chat_log(inp, out)
        assert hasattr(report, "lossiness")
        assert isinstance(report.lossiness, LossReport)

    def test_import_report_as_dict_includes_lossiness(self, tmp_path):
        inp = _write_json(str(tmp_path / "in.json"), _openai_log())
        out = str(tmp_path / "out.sb")
        report = import_openai_chat_log(inp, out)
        d = report.as_dict()
        assert "lossiness" in d
        assert isinstance(d["lossiness"], dict)
        for key in ("absent", "approximated", "synthesized", "dropped"):
            assert key in d["lossiness"]

    def test_openai_chat_log_lossiness(self, tmp_path):
        inp = _write_json(str(tmp_path / "in.json"), _openai_log())
        out = str(tmp_path / "out.sb")
        report = import_openai_chat_log(inp, out)
        loss = report.lossiness
        assert len(loss.synthesized) >= 3  # step_id, nondeterminism_hash, parent_step_id
        assert len(loss.approximated) >= 2  # cost_usd, seed
        assert not loss.is_lossless()
        # Check specific synthesized fields are mentioned
        synt_text = " ".join(loss.synthesized)
        assert "step_id" in synt_text
        assert "nondeterminism_hash" in synt_text
        assert "parent_step_id" in synt_text

    def test_langsmith_v1_lossiness(self, tmp_path):
        inp = _write_json(str(tmp_path / "ls.json"), _langsmith_v1_runs())
        out = str(tmp_path / "out.sb")
        report = import_langsmith_jsonl(inp, out)
        loss = report.lossiness
        assert len(loss.synthesized) >= 2
        assert len(loss.approximated) >= 2
        dropped_text = " ".join(loss.dropped)
        assert "tags" in dropped_text
        # v1 should NOT mention feedback_stats
        assert "feedback_stats" not in dropped_text

    def test_langsmith_v2_adds_dropped_fields(self, tmp_path):
        inp = _write_json(str(tmp_path / "ls2.json"), _langsmith_v2_runs())
        out = str(tmp_path / "out.sb")
        report = import_langsmith_jsonl(inp, out)
        assert report.schema_version == "v2"
        dropped_text = " ".join(report.lossiness.dropped)
        assert "feedback_stats" in dropped_text
        assert "app_path" in dropped_text

    def test_openinference_spans_lossiness(self, tmp_path):
        inp = _write_json(str(tmp_path / "oi.json"), _oi_spans())
        out = str(tmp_path / "out.sb")
        report = import_openinference_spans(inp, out)
        loss = report.lossiness
        synt_text = " ".join(loss.synthesized)
        assert "nondeterminism_hash" in synt_text
        approx_text = " ".join(loss.approximated)
        assert "step_kind" in approx_text

    def test_otel_spans_lossiness(self, tmp_path):
        inp = _write_json(str(tmp_path / "otel.json"), _otel_spans())
        out = str(tmp_path / "out.sb")
        report = import_otel_spans(inp, out)
        loss = report.lossiness
        assert len(loss.synthesized) >= 2
        dropped_text = " ".join(loss.dropped)
        assert "resource" in dropped_text

    def test_phoenix_spans_lossiness(self, tmp_path):
        inp = _write_json(str(tmp_path / "px.json"), _phoenix_spans())
        out = str(tmp_path / "out.sb")
        report = import_phoenix_spans(inp, out)
        loss = report.lossiness
        assert len(loss.dropped) >= 4
        dropped_text = " ".join(loss.dropped)
        assert "trace_id" in dropped_text
        assert "cumulative_token_count" in dropped_text

    def test_helicone_log_lossiness(self, tmp_path):
        inp = _write_json(str(tmp_path / "hel.json"), _helicone_log())
        out = str(tmp_path / "out.sb")
        report = import_helicone_log(inp, out)
        loss = report.lossiness
        dropped_text = " ".join(loss.dropped)
        assert "request_id" in dropped_text
        assert "properties" in dropped_text
        assert "feedback" in dropped_text

    def test_langfuse_lossiness(self, tmp_path):
        inp = _write_json(str(tmp_path / "lf.json"), _langfuse_traces())
        out = str(tmp_path / "out.sb")
        report = import_langfuse_export(inp, out)
        loss = report.lossiness
        dropped_text = " ".join(loss.dropped)
        assert "scores" in dropped_text
        assert "userId" in dropped_text or "user" in dropped_text.lower()

    def test_datadog_apm_lossiness(self, tmp_path):
        inp = _write_json(str(tmp_path / "dd.json"), _datadog_spans())
        out = str(tmp_path / "out.sb")
        report = import_datadog_apm(inp, out)
        loss = report.lossiness
        dropped_text = " ".join(loss.dropped)
        assert "service" in dropped_text
        assert "trace_id" in dropped_text

    def test_native_json_lossiness(self, tmp_path):
        """import_native_json is lossless — the lossiness report should be empty."""
        # Record a real trace to native JSON, then re-import it
        from stepback.recorder import RecorderKey
        from stepback.trace_reader import verify_trace
        from stepback.testing import run_recorded_agent
        sb_path = str(tmp_path / "trace.sb")
        key = RecorderKey.fresh()
        with record(sb_path, key=key) as rec:
            run_recorded_agent(rec)
        steps = verify_trace(sb_path, key.hmac_key).steps
        native_json = str(tmp_path / "native.json")
        export_native_json(steps, native_json)

        out = str(tmp_path / "reimported.sb")
        report = import_native_json(native_json, out)
        assert report.lossiness.is_lossless()


# ---------------------------------------------------------- ExportReport


class TestExportReportLossiness:
    def _sample_steps(self, tmp_path) -> List[dict]:
        from stepback.recorder import RecorderKey
        from stepback.trace_reader import verify_trace
        from stepback.testing import run_recorded_agent
        sb_path = str(tmp_path / "trace.sb")
        key = RecorderKey.fresh()
        with record(sb_path, key=key) as rec:
            run_recorded_agent(rec)
        return verify_trace(sb_path, key.hmac_key).steps

    def test_export_report_has_lossiness_field(self, tmp_path):
        steps = self._sample_steps(tmp_path)
        out = str(tmp_path / "out.json")
        report = export_openai_chat_log(steps, out)
        assert hasattr(report, "lossiness")
        assert isinstance(report.lossiness, LossReport)

    def test_export_report_as_dict_includes_lossiness(self, tmp_path):
        steps = self._sample_steps(tmp_path)
        out = str(tmp_path / "out.json")
        report = export_openai_chat_log(steps, out)
        d = report.as_dict()
        assert "lossiness" in d
        for key in ("absent", "approximated", "synthesized", "dropped"):
            assert key in d["lossiness"]

    def test_openai_chat_log_export_lossiness(self, tmp_path):
        steps = self._sample_steps(tmp_path)
        out = str(tmp_path / "out.json")
        report = export_openai_chat_log(steps, out)
        loss = report.lossiness
        dropped_text = " ".join(loss.dropped)
        assert "nondeterminism_hash" in dropped_text
        assert "step_id" in dropped_text
        absent_text = " ".join(loss.absent)
        assert "tool_call" in absent_text

    def test_langsmith_export_lossiness(self, tmp_path):
        steps = self._sample_steps(tmp_path)
        out = str(tmp_path / "out.jsonl")
        report = export_langsmith_jsonl(steps, out)
        loss = report.lossiness
        synt_text = " ".join(loss.synthesized)
        assert "UUID" in synt_text or "id" in synt_text.lower()
        approx_text = " ".join(loss.approximated)
        assert "run_type" in approx_text
        dropped_text = " ".join(loss.dropped)
        assert "nondeterminism_hash" in dropped_text

    def test_openinference_export_lossiness(self, tmp_path):
        steps = self._sample_steps(tmp_path)
        out = str(tmp_path / "out.json")
        report = export_openinference_spans(steps, out)
        loss = report.lossiness
        synt_text = " ".join(loss.synthesized)
        assert "span_id" in synt_text
        dropped_text = " ".join(loss.dropped)
        assert "nondeterminism_hash" in dropped_text

    def test_otel_export_lossiness(self, tmp_path):
        steps = self._sample_steps(tmp_path)
        out = str(tmp_path / "out.json")
        report = export_otel_spans(steps, out)
        loss = report.lossiness
        synt_text = " ".join(loss.synthesized)
        assert "span_id" in synt_text
        dropped_text = " ".join(loss.dropped)
        # nondeterminism_hash and inputs_hash are now emitted as agent.step.* attrs (RFC 0006)
        assert "outputs_hash" in dropped_text

    def test_native_json_export_lossless(self, tmp_path):
        """export_native_json is lossless — no lossiness fields should be set."""
        steps = self._sample_steps(tmp_path)
        out = str(tmp_path / "out.json")
        report = export_native_json(steps, out)
        assert report.lossiness.is_lossless()

    def test_html_export_lossiness(self, tmp_path):
        steps = self._sample_steps(tmp_path)
        out = str(tmp_path / "out.html")
        report = export_html_view(steps, out)
        loss = report.lossiness
        dropped_text = " ".join(loss.dropped)
        assert "nondeterminism_hash" in dropped_text
        absent_text = " ".join(loss.absent)
        assert "round-trip" in absent_text.lower() or "import" in absent_text.lower()

    def test_cyclonedx_export_lossiness(self, tmp_path):
        steps = self._sample_steps(tmp_path)
        out = str(tmp_path / "out.json")
        report = export_cyclonedx_ai(steps, out)
        loss = report.lossiness
        dropped_text = " ".join(loss.dropped)
        assert "execution" in dropped_text.lower() or "sequence" in dropped_text.lower()
        absent_text = " ".join(loss.absent)
        assert "replay" in absent_text.lower()
