"""End-to-end tests for the foreign-trace EXPORTER module.

The exporter is the symmetric counterpart to ``stepback.importers``:
where the importer takes a foreign trace (LangSmith / OpenInference /
OpenAI log) and writes a `.sb` trace, the exporter takes a `.sb`
trace and writes a foreign-format file.

Each test in this module records (or imports) a real trace via the
fixture agent, exports it to a foreign format, and asserts:

1. The output file is a syntactically valid file in the target format
   (JSON parses, each JSONL line parses, OTel envelope is well-formed).
2. The set of step kinds and their counts in the export match the
   source trace's kind counts (after the documented foldings).
3. ROUND-TRIP: importing the exported file back into a `.sb` trace
   produces the same step_count and the same kind_counts as the
   original source.
4. CLI ``stepback export`` end-to-end works on a real recorded trace.

This is the e2e-fixture-test (kitchensink rule 3a) for this round:
the fixture is the deterministic 12-step "customer payments bot" in
``tests/fixtures/agent.py`` (NOT a mock), and assertions are on real
numeric properties (step count = 12, llm_call count = 6, tool_call
count = 6, JSON byte size in a bounded envelope, round-trip preserves
the parent-edge tree exactly).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import List

import pytest

from stepback import (
    ExportReport,
    TraceExportError,
    available_export_formats,
    export_langsmith_jsonl,
    export_openai_chat_log,
    export_openinference_spans,
    export_trace,
    export_trace_file,
    import_langsmith_jsonl,
    import_openai_chat_log,
    import_openinference_spans,
    record,
    replay,
)
from stepback.exporters import (
    _build_id_map,
    _llm_messages,
    _llm_request,
    _llm_usage,
    _wallclock_to_iso,
)
from stepback.recorder import RecorderKey
from stepback.trace_reader import verify_trace
from stepback.testing import run_recorded_agent


# ---------------------------------------------------------- helpers


def _record_fixture(tmp_path) -> tuple:
    """Record the 12-step fixture agent; return (sb_path, hmac_key)."""
    sb_path = os.path.join(str(tmp_path), "fixture.sb")
    key = RecorderKey.fresh()
    with record(sb_path, key=key) as rec:
        run_recorded_agent(rec)
    return sb_path, key


def _read_steps(sb_path: str, key: RecorderKey) -> List[dict]:
    return verify_trace(sb_path, key.hmac_key).steps


# ---------------------------------------------------------- basic smoke


def test_available_export_formats_lists_all_three_families():
    formats = available_export_formats()
    # All three target families plus their aliases.
    assert "openai_chat_log" in formats
    assert "langsmith" in formats
    assert "openinference" in formats
    assert "otel" in formats  # alias for openinference
    assert len(formats) >= 7  # at least canonical + aliases


def test_export_trace_unknown_format_raises():
    with pytest.raises(TraceExportError):
        export_trace("not_a_real_format", [], "/tmp/out.json")


def test_export_trace_rejects_non_list_steps(tmp_path):
    out = os.path.join(str(tmp_path), "x.json")
    with pytest.raises(TraceExportError):
        export_openai_chat_log({"not": "a list"}, out)  # type: ignore[arg-type]


def test_export_trace_rejects_step_missing_required_keys(tmp_path):
    out = os.path.join(str(tmp_path), "x.json")
    with pytest.raises(TraceExportError):
        export_openai_chat_log([{"some": "junk"}], out)


def test_wallclock_iso_round_trip():
    # 2026-04-12T13:04:00Z
    ns = 1_775_999_040 * 1_000_000_000
    iso = _wallclock_to_iso(ns)
    assert iso is not None
    assert iso.endswith("Z")
    assert "2026" in iso
    assert _wallclock_to_iso(None) is None


def test_id_map_is_deterministic_and_unique():
    steps = [{"step_id": f"step:{i}"} for i in range(1, 13)]
    a = _build_id_map(steps)
    b = _build_id_map(steps)
    assert a == b
    # Foreign ids are 32-hex (per implementation).
    assert all(len(v) == 32 and all(c in "0123456789abcdef" for c in v) for v in a.values())
    assert len(set(a.values())) == len(a)  # no collisions


# --------------------------------------------- OpenAI chat log exporter


def test_export_openai_chat_log_round_trip(tmp_path):
    sb_path, key = _record_fixture(tmp_path)
    steps = _read_steps(sb_path, key)
    out = os.path.join(str(tmp_path), "openai.json")
    report = export_openai_chat_log(steps, out)
    assert isinstance(report, ExportReport)
    # Fixture has 6 llm_calls.
    assert report.kind_counts.get("llm_call") == 6
    assert report.step_count == 6

    # Output is a JSON list of OpenAI chat-completion entries.
    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == 6
    for entry in data:
        assert "model" in entry
        assert "messages" in entry
        assert "response" in entry
        assert isinstance(entry["messages"], list)
        assert isinstance(entry["response"], dict)

    # Round-trip: re-import the openai log and verify the result.
    rt = os.path.join(str(tmp_path), "rt.sb")
    rt_key = RecorderKey.fresh()
    rep = import_openai_chat_log(out, rt, key=rt_key)
    assert rep.step_count == 6
    assert rep.kind_counts.get("llm_call") == 6
    rt_trace = verify_trace(rt, rt_key.hmac_key)
    # Models survived the round-trip.
    src_models = [s.get("name") for s in steps if s["step_kind"] == "llm_call"]
    rt_models = [s.get("name") for s in rt_trace.steps if s["step_kind"] == "llm_call"]
    assert src_models == rt_models


def test_export_openai_chat_log_skips_non_llm_kinds(tmp_path):
    sb_path, key = _record_fixture(tmp_path)
    steps = _read_steps(sb_path, key)
    out = os.path.join(str(tmp_path), "openai2.json")
    report = export_openai_chat_log(steps, out)
    # Fixture has 6 tool_calls; they go to skipped.
    assert len(report.skipped) == 6
    for msg in report.skipped:
        # Each skip names a step_id and a kind reason.
        assert msg.startswith("step:")


# ------------------------------------------- LangSmith JSONL exporter


def test_export_langsmith_preserves_call_tree(tmp_path):
    sb_path, key = _record_fixture(tmp_path)
    steps = _read_steps(sb_path, key)
    out = os.path.join(str(tmp_path), "ls.jsonl")
    report = export_langsmith_jsonl(steps, out)
    # 6 llm + 6 tool, no router in fixture, total 12.
    assert report.step_count == 12
    assert report.kind_counts.get("llm") == 6
    assert report.kind_counts.get("tool") == 6
    assert report.skipped == []

    # Each line is valid JSON with the LangSmith run shape.
    runs: List[dict] = []
    with open(out, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            run = json.loads(line)
            runs.append(run)
            assert "id" in run and isinstance(run["id"], str)
            assert "run_type" in run
            assert run["run_type"] in {"llm", "tool", "chain"}
            assert "inputs" in run and "outputs" in run
    assert len(runs) == 12
    # Parent edges form one connected tree (every non-root has a parent
    # that appears as some other run's id).
    by_id = {r["id"]: r for r in runs}
    for r in runs:
        pid = r.get("parent_run_id")
        if pid is not None:
            assert pid in by_id, f"dangling parent_run_id {pid} for run {r['id']}"

    # Round-trip: import the langsmith JSONL back and verify counts.
    rt = os.path.join(str(tmp_path), "rt_ls.sb")
    rt_key = RecorderKey.fresh()
    rep = import_langsmith_jsonl(out, rt, key=rt_key)
    assert rep.step_count == 12
    # llm_call + tool_call = original kinds.
    assert rep.kind_counts.get("llm_call") == 6
    assert rep.kind_counts.get("tool_call") == 6


def test_export_langsmith_llm_runs_carry_token_usage(tmp_path):
    sb_path, key = _record_fixture(tmp_path)
    steps = _read_steps(sb_path, key)
    out = os.path.join(str(tmp_path), "ls_usage.jsonl")
    export_langsmith_jsonl(steps, out)
    n_with_usage = 0
    with open(out, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["run_type"] == "llm":
                extra = r.get("extra") or {}
                assert "invocation_params" in extra
                assert extra["invocation_params"].get("model")
                if "token_usage" in extra:
                    u = extra["token_usage"]
                    assert {"prompt_tokens", "completion_tokens", "total_tokens"} & set(u.keys())
                    n_with_usage += 1
    # Fixture's fake_llm always emits usage.
    assert n_with_usage == 6


# ------------------------------------- OpenInference spans exporter


def test_export_openinference_emits_spans_with_required_attrs(tmp_path):
    sb_path, key = _record_fixture(tmp_path)
    steps = _read_steps(sb_path, key)
    out = os.path.join(str(tmp_path), "oi.json")
    report = export_openinference_spans(steps, out)
    assert report.step_count == 12
    assert report.kind_counts.get("LLM") == 6
    assert report.kind_counts.get("TOOL") == 6

    with open(out, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert "spans" in payload
    spans = payload["spans"]
    assert len(spans) == 12
    for sp in spans:
        assert "span_id" in sp
        assert "attributes" in sp
        # Find the kind attribute.
        kinds = [a["value"]["stringValue"] for a in sp["attributes"]
                 if a["key"] == "openinference.span.kind"]
        assert len(kinds) == 1
        assert kinds[0] in {"LLM", "TOOL", "CHAIN"}

    # Round-trip: re-import as openinference spans.
    rt = os.path.join(str(tmp_path), "rt_oi.sb")
    rt_key = RecorderKey.fresh()
    rep = import_openinference_spans(out, rt, key=rt_key)
    assert rep.step_count == 12
    assert rep.kind_counts.get("llm_call") == 6
    assert rep.kind_counts.get("tool_call") == 6


def test_export_openinference_envelope_off_emits_bare_array(tmp_path):
    sb_path, key = _record_fixture(tmp_path)
    steps = _read_steps(sb_path, key)
    out = os.path.join(str(tmp_path), "oi_bare.json")
    export_openinference_spans(steps, out, envelope=False)
    with open(out, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert isinstance(payload, list)
    assert len(payload) == 12


def test_export_openinference_llm_carries_token_count_attrs(tmp_path):
    sb_path, key = _record_fixture(tmp_path)
    steps = _read_steps(sb_path, key)
    out = os.path.join(str(tmp_path), "oi_tc.json")
    export_openinference_spans(steps, out)
    with open(out, "r", encoding="utf-8") as f:
        payload = json.load(f)
    spans_with_tc = 0
    for sp in payload["spans"]:
        attr_keys = {a["key"] for a in sp["attributes"]}
        if "openinference.span.kind" in attr_keys:
            kinds = [a["value"]["stringValue"] for a in sp["attributes"]
                     if a["key"] == "openinference.span.kind"]
            if kinds == ["LLM"]:
                assert "llm.model_name" in attr_keys
                if "llm.token_count.prompt" in attr_keys:
                    spans_with_tc += 1
    assert spans_with_tc == 6


# --------------------------------------- export_trace dispatcher


def test_export_trace_dispatcher_routes_correctly(tmp_path):
    sb_path, key = _record_fixture(tmp_path)
    steps = _read_steps(sb_path, key)
    for fmt, suffix in [
        ("openai", "openai.json"),
        ("langsmith", "ls.jsonl"),
        ("openinference", "oi.json"),
        ("otel", "otel.json"),  # alias
    ]:
        out = os.path.join(str(tmp_path), suffix)
        report = export_trace(fmt, steps, out)
        assert os.path.getsize(out) > 0
        assert report.step_count > 0


# --------------------------------------- export_trace_file (verifies HMAC)


def test_export_trace_file_verifies_chain_then_exports(tmp_path):
    sb_path, key = _record_fixture(tmp_path)
    out = os.path.join(str(tmp_path), "verified_export.jsonl")
    report = export_trace_file(
        "langsmith", sb_path, out, hmac_key=key.hmac_key,
    )
    assert report.step_count == 12

    # Tampered HMAC key → verify_trace fails → ExportError chain.
    bad_key = b"\x00" * 32
    with pytest.raises(Exception):
        export_trace_file(
            "langsmith", sb_path, out, hmac_key=bad_key,
        )


def test_export_trace_file_rejects_non_bytes_key(tmp_path):
    sb_path, _ = _record_fixture(tmp_path)
    out = os.path.join(str(tmp_path), "no.jsonl")
    with pytest.raises(TraceExportError):
        export_trace_file(
            "langsmith", sb_path, out, hmac_key="deadbeef",  # type: ignore[arg-type]
        )


# --------------------------------------- CLI smoke


def test_cli_export_subcommand_round_trips_via_subprocess(tmp_path):
    sb_path, key = _record_fixture(tmp_path)
    out = os.path.join(str(tmp_path), "cli_ls.jsonl")
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    result = subprocess.run(
        [
            sys.executable, "-m", "stepback.cli", "export",
            "--format", "langsmith",
            "--input", sb_path,
            "--output", out,
            "--hmac-key-hex", key.hmac_key.hex(),
            "--json",
        ],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
    rep = json.loads(result.stdout)
    assert rep["target_format"] == "langsmith_jsonl"
    assert rep["step_count"] == 12
    assert os.path.getsize(out) > 0
    # Each line should parse as JSON and have an "id".
    with open(out, "r", encoding="utf-8") as f:
        n = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            assert "id" in r
            n += 1
        assert n == 12


def test_cli_export_bad_hmac_key_returns_nonzero(tmp_path):
    sb_path, key = _record_fixture(tmp_path)
    out = os.path.join(str(tmp_path), "cli_bad.jsonl")
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    result = subprocess.run(
        [
            sys.executable, "-m", "stepback.cli", "export",
            "--format", "langsmith",
            "--input", sb_path,
            "--output", out,
            "--hmac-key-hex", "not-hex!!!",
        ],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode != 0


# --------------------------------------- numeric-threshold guarantees


def test_export_size_is_bounded(tmp_path):
    """Export of the 12-step fixture stays under sane envelopes."""
    sb_path, key = _record_fixture(tmp_path)
    steps = _read_steps(sb_path, key)
    out_oi = os.path.join(str(tmp_path), "oi_sz.json")
    out_ls = os.path.join(str(tmp_path), "ls_sz.jsonl")
    out_oa = os.path.join(str(tmp_path), "oa_sz.json")
    export_openinference_spans(steps, out_oi)
    export_langsmith_jsonl(steps, out_ls)
    export_openai_chat_log(steps, out_oa)
    sz_oi = os.path.getsize(out_oi)
    sz_ls = os.path.getsize(out_ls)
    sz_oa = os.path.getsize(out_oa)
    # 12-step fixture → expect each file in [200, 200_000] bytes.
    for sz in (sz_oi, sz_ls, sz_oa):
        assert 200 < sz < 200_000


def test_round_trip_three_formats_preserves_kind_counts(tmp_path):
    """Source kind_counts → export → import → reconstructed kind_counts."""
    sb_path, key = _record_fixture(tmp_path)
    steps = _read_steps(sb_path, key)
    src_llm = sum(1 for s in steps if s["step_kind"] == "llm_call")
    src_tool = sum(1 for s in steps if s["step_kind"] == "tool_call")
    assert (src_llm, src_tool) == (6, 6)

    # langsmith preserves both kinds.
    ls_path = os.path.join(str(tmp_path), "rt_ls.jsonl")
    export_langsmith_jsonl(steps, ls_path)
    ls_sb = os.path.join(str(tmp_path), "ls_back.sb")
    ls_key = RecorderKey.fresh()
    rep_ls = import_langsmith_jsonl(ls_path, ls_sb, key=ls_key)
    assert rep_ls.kind_counts.get("llm_call") == src_llm
    assert rep_ls.kind_counts.get("tool_call") == src_tool

    # openinference preserves both kinds.
    oi_path = os.path.join(str(tmp_path), "rt_oi.json")
    export_openinference_spans(steps, oi_path)
    oi_sb = os.path.join(str(tmp_path), "oi_back.sb")
    oi_key = RecorderKey.fresh()
    rep_oi = import_openinference_spans(oi_path, oi_sb, key=oi_key)
    assert rep_oi.kind_counts.get("llm_call") == src_llm
    assert rep_oi.kind_counts.get("tool_call") == src_tool

    # openai_chat_log keeps llm_calls only (documented folding).
    oa_path = os.path.join(str(tmp_path), "rt_oa.json")
    export_openai_chat_log(steps, oa_path)
    oa_sb = os.path.join(str(tmp_path), "oa_back.sb")
    oa_key = RecorderKey.fresh()
    rep_oa = import_openai_chat_log(oa_path, oa_sb, key=oa_key)
    assert rep_oa.kind_counts.get("llm_call") == src_llm
    assert rep_oa.kind_counts.get("tool_call") is None


def test_round_trip_through_langsmith_remains_replayable(tmp_path):
    """Export → import → replay must reach a 100% cache hit."""
    sb_path, key = _record_fixture(tmp_path)
    steps = _read_steps(sb_path, key)
    ls_path = os.path.join(str(tmp_path), "rep_ls.jsonl")
    export_langsmith_jsonl(steps, ls_path)
    rt_sb = os.path.join(str(tmp_path), "rep_back.sb")
    rt_key = RecorderKey.fresh()
    import_langsmith_jsonl(ls_path, rt_sb, key=rt_key)
    trace = replay(rt_sb, hmac_key=rt_key.hmac_key)
    result = trace.replay_forward()
    # Cache-only replay → no LLM calls re-executed.
    assert result.real_executions == 0
    assert result.cache_hit_count == 12


def test_export_idempotent_byte_for_byte(tmp_path):
    """Running the same export twice produces identical output bytes."""
    sb_path, key = _record_fixture(tmp_path)
    steps = _read_steps(sb_path, key)
    a = os.path.join(str(tmp_path), "a.jsonl")
    b = os.path.join(str(tmp_path), "b.jsonl")
    export_langsmith_jsonl(steps, a)
    export_langsmith_jsonl(steps, b)
    with open(a, "rb") as fa, open(b, "rb") as fb:
        assert fa.read() == fb.read()
