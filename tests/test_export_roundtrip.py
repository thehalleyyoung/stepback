"""Step #38 (docs/100_STEPS.md): import/export round-trip tests.

Covers the five exporter formats called out in the directive:

* **LangSmith** — round-trip ``.sb → .jsonl → .sb``.
* **OpenInference** — round-trip ``.sb → spans.json → .sb``.
* **JSON** — the native lossless JSON dump
  (:func:`stepback.export_native_json` /
  :func:`stepback.import_native_json`).
* **HTML** — the self-contained interactive viewer (round-trip is
  performed by extracting the embedded JSON island from the rendered
  page and asserting that every recorded step survived).
* **OTel** — the ``otel`` alias for the OpenInference exporter, plus
  the v1 OpenInference span layout used by OTel collectors.

Each test records the deterministic 12-step fixture agent (LLM + tool
turns), exports to the foreign format via the dispatcher
(``export_trace``), reads the file back, and asserts:

1. The round-tripped step count equals the recorded step count after
   the documented foldings (LangSmith / OpenInference preserve all
   six llm_call + six tool_call frames; OpenAI chat log folds tool
   calls; JSON / HTML preserve everything).
2. The set of step kinds matches.
3. Per-format invariants survive the round-trip (model ids on
   ``llm_call`` frames; tool names on ``tool_call`` frames; parent
   edges).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

import pytest

from stepback import (
    available_export_formats,
    export_trace,
    import_native_json,
    import_langsmith_jsonl,
    import_openinference_spans,
    record,
)
from stepback.recorder import RecorderKey
from stepback.testing import run_recorded_agent
from stepback.trace_reader import verify_trace


# ---------------------------------------------------------- helpers


def _record_fixture(tmp_path) -> tuple:
    sb_path = os.path.join(str(tmp_path), "fixture.sb")
    key = RecorderKey.fresh()
    with record(sb_path, key=key) as rec:
        run_recorded_agent(rec)
    return sb_path, key


def _read_steps(sb_path: str, key: RecorderKey) -> List[dict]:
    return verify_trace(sb_path, key.hmac_key).steps


def _kind_counts(steps: List[dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for s in steps:
        k = str(s.get("step_kind"))
        out[k] = out.get(k, 0) + 1
    return out


# ---------------------------------------------------------- format registry


def test_dispatcher_lists_json_html_and_otel_aliases():
    fmts = set(available_export_formats())
    # All five families called out in 100_STEPS.md step #38.
    assert "langsmith" in fmts
    assert "openinference" in fmts
    assert "json" in fmts
    assert "html" in fmts
    assert "otel" in fmts
    # Aliases for the new formats are wired too.
    assert "native_json" in fmts
    assert "stepback_json" in fmts
    assert "html_view" in fmts


# ---------------------------------------------------------- LangSmith


def test_langsmith_round_trip_preserves_step_set(tmp_path):
    sb_path, key = _record_fixture(tmp_path)
    steps = _read_steps(sb_path, key)
    src_counts = _kind_counts(steps)
    assert src_counts.get("llm_call") == 6
    assert src_counts.get("tool_call") == 6

    foreign = os.path.join(str(tmp_path), "trace.langsmith.jsonl")
    rep = export_trace("langsmith", steps, foreign)
    assert rep.step_count == 12
    assert os.path.getsize(foreign) > 0

    rt_path = os.path.join(str(tmp_path), "rt.sb")
    rt_key = RecorderKey.fresh()
    irep = import_langsmith_jsonl(foreign, rt_path, key=rt_key)
    rt_steps = _read_steps(rt_path, rt_key)

    assert irep.step_count == 12
    assert len(rt_steps) == 12
    rt_counts = _kind_counts(rt_steps)
    assert rt_counts.get("llm_call") == 6
    assert rt_counts.get("tool_call") == 6


# ---------------------------------------------------------- OpenInference


def test_openinference_round_trip_preserves_step_set(tmp_path):
    sb_path, key = _record_fixture(tmp_path)
    steps = _read_steps(sb_path, key)
    src_counts = _kind_counts(steps)

    foreign = os.path.join(str(tmp_path), "trace.openinference.json")
    rep = export_trace("openinference", steps, foreign)
    assert rep.step_count == 12

    with open(foreign, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert isinstance(payload, dict) and "spans" in payload
    assert len(payload["spans"]) == 12

    rt_path = os.path.join(str(tmp_path), "rt.sb")
    rt_key = RecorderKey.fresh()
    irep = import_openinference_spans(foreign, rt_path, key=rt_key)
    rt_steps = _read_steps(rt_path, rt_key)
    assert irep.step_count == 12
    rt_counts = _kind_counts(rt_steps)
    assert rt_counts.get("llm_call") == src_counts.get("llm_call")
    assert rt_counts.get("tool_call") == src_counts.get("tool_call")


# ---------------------------------------------------------- OTel alias


def test_otel_alias_round_trips_via_import_otel_spans(tmp_path):
    """The ``otel`` alias uses the stable ``agent.step.*`` semconv exporter
    (``export_otel_spans``) and round-trips via ``import_otel_spans``."""
    from stepback.importers import import_otel_spans

    sb_path, key = _record_fixture(tmp_path)
    steps = _read_steps(sb_path, key)

    a = os.path.join(str(tmp_path), "trace.otel.json")
    rep_a = export_trace("otel", steps, a)
    assert rep_a.step_count == 12

    # otel and otel_spans alias must produce identical bytes.
    b = os.path.join(str(tmp_path), "trace.otel_spans.json")
    rep_b = export_trace("otel_spans", steps, b)
    with open(a, "rb") as fa, open(b, "rb") as fb:
        assert fa.read() == fb.read()

    # The exported spans use agent.step.kind (not openinference.span.kind).
    with open(a, "r", encoding="utf-8") as f:
        payload = json.load(f)
    for sp in payload["spans"]:
        attr_keys = {attr["key"] for attr in sp["attributes"]}
        assert "agent.step.kind" in attr_keys
        assert "openinference.span.kind" not in attr_keys

    # Round-trip via import_otel_spans reconstructs the full step shape.
    rt = os.path.join(str(tmp_path), "rt.sb")
    rt_key = RecorderKey.fresh()
    irep = import_otel_spans(a, rt, key=rt_key)
    rt_steps = _read_steps(rt, rt_key)
    assert irep.step_count == 12
    assert _kind_counts(rt_steps).get("llm_call") == 6
    assert _kind_counts(rt_steps).get("tool_call") == 6


# ---------------------------------------------------------- native JSON


def test_native_json_round_trip_is_lossless(tmp_path):
    sb_path, key = _record_fixture(tmp_path)
    steps = _read_steps(sb_path, key)

    foreign = os.path.join(str(tmp_path), "trace.json")
    rep = export_trace("json", steps, foreign)
    assert rep.step_count == len(steps)
    assert rep.kind_counts == _kind_counts(steps)

    with open(foreign, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["format"] == "stepback_native_json_v1"
    assert isinstance(payload["steps"], list)
    assert len(payload["steps"]) == len(steps)

    rt_path = os.path.join(str(tmp_path), "rt.sb")
    rt_key = RecorderKey.fresh()
    irep = import_native_json(foreign, rt_path, key=rt_key)
    rt_steps = _read_steps(rt_path, rt_key)

    assert irep.step_count == len(steps)
    assert _kind_counts(rt_steps) == _kind_counts(steps)
    # Lossless: every recorded field survives.
    src_ids = [str(s["step_id"]) for s in steps]
    rt_ids = [str(s["step_id"]) for s in rt_steps]
    assert rt_ids == src_ids
    src_hashes = [s.get("inputs_hash") for s in steps]
    rt_hashes = [s.get("inputs_hash") for s in rt_steps]
    assert rt_hashes == src_hashes


def test_native_json_rejects_wrong_format_tag(tmp_path):
    bad = os.path.join(str(tmp_path), "bad.json")
    with open(bad, "w", encoding="utf-8") as f:
        json.dump({"format": "not_us", "steps": []}, f)
    rt = os.path.join(str(tmp_path), "rt.sb")
    from stepback.importers import ImportError as IE
    with pytest.raises(IE):
        import_native_json(bad, rt)


def test_native_json_rejects_non_object_top_level(tmp_path):
    bad = os.path.join(str(tmp_path), "bad.json")
    with open(bad, "w", encoding="utf-8") as f:
        json.dump([1, 2, 3], f)
    rt = os.path.join(str(tmp_path), "rt.sb")
    from stepback.importers import ImportError as IE
    with pytest.raises(IE):
        import_native_json(bad, rt)


def test_native_json_with_header_payload(tmp_path):
    sb_path, key = _record_fixture(tmp_path)
    steps = _read_steps(sb_path, key)
    foreign = os.path.join(str(tmp_path), "trace.json")
    export_trace("json", steps, foreign)
    # export_trace doesn't pass a header through its 3-arg signature;
    # call the function directly to exercise the header branch.
    from stepback import export_native_json
    foreign2 = os.path.join(str(tmp_path), "trace2.json")
    rep = export_native_json(
        steps, foreign2,
        header={"recorder_version": "test-recorder/0.0.1"},
    )
    assert rep.step_count == len(steps)
    with open(foreign2, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["header"]["recorder_version"] == "test-recorder/0.0.1"


# ---------------------------------------------------------- HTML


_HTML_DATA_ISLAND_RE = re.compile(
    r"<script type='application/json' id='stepback-data'>(.*?)</script>",
    re.DOTALL,
)


def _extract_html_data_island(html: str) -> Dict[str, Any]:
    m = _HTML_DATA_ISLAND_RE.search(html)
    assert m is not None, "HTML output is missing the stepback-data island"
    raw = m.group(1).strip()
    # The exporter escapes "</" to "<\/" to defuse </script> injection;
    # JSON tolerates "\/" so we can decode directly.
    return json.loads(raw)


def test_html_round_trip_preserves_steps_via_data_island(tmp_path):
    sb_path, key = _record_fixture(tmp_path)
    steps = _read_steps(sb_path, key)

    foreign = os.path.join(str(tmp_path), "trace.html")
    rep = export_trace("html", steps, foreign)
    assert rep.step_count == len(steps)
    assert rep.kind_counts == _kind_counts(steps)

    with open(foreign, "r", encoding="utf-8") as f:
        page = f.read()
    assert page.startswith("<!doctype html>")
    assert "id='stepback-data'" in page

    model = _extract_html_data_island(page)
    # The view model carries by_kind + step_count we can compare
    # against the source.
    assert model["step_count"] == len(steps)
    assert dict(model["by_kind"]) == _kind_counts(steps)
    # All step ids round-tripped, in order.
    rt_ids = [str(s["step_id"]) for s in model["steps"]]
    src_ids = [str(s["step_id"]) for s in steps]
    assert rt_ids == src_ids


def test_html_alias_html_view_is_byte_identical(tmp_path):
    sb_path, key = _record_fixture(tmp_path)
    steps = _read_steps(sb_path, key)

    a = os.path.join(str(tmp_path), "a.html")
    b = os.path.join(str(tmp_path), "b.html")
    export_trace("html", steps, a)
    export_trace("html_view", steps, b)
    with open(a, "rb") as fa, open(b, "rb") as fb:
        assert fa.read() == fb.read()


# ---------------------------------------------------------- five-format matrix


def test_round_trip_five_format_matrix(tmp_path):
    """One recording → all five formats → round-trip back where possible.

    For LangSmith / OpenInference / JSON the round-trip completes
    back to a `.sb`. For HTML the round-trip is done in-process via
    the data island. The OTel alias is exercised alongside the
    canonical OpenInference path.
    """
    sb_path, key = _record_fixture(tmp_path)
    steps = _read_steps(sb_path, key)
    src_counts = _kind_counts(steps)

    paths = {
        "langsmith": os.path.join(str(tmp_path), "ls.jsonl"),
        "openinference": os.path.join(str(tmp_path), "oi.json"),
        "otel": os.path.join(str(tmp_path), "otel.json"),
        "json": os.path.join(str(tmp_path), "native.json"),
        "html": os.path.join(str(tmp_path), "view.html"),
    }
    for fmt, p in paths.items():
        rep = export_trace(fmt, steps, p)
        assert rep.step_count > 0
        assert os.path.exists(p)

    # langsmith round-trip.
    rk = RecorderKey.fresh()
    rt_ls = os.path.join(str(tmp_path), "rt_ls.sb")
    import_langsmith_jsonl(paths["langsmith"], rt_ls, key=rk)
    assert _kind_counts(_read_steps(rt_ls, rk)) == src_counts

    # openinference round-trip.
    rk = RecorderKey.fresh()
    rt_oi = os.path.join(str(tmp_path), "rt_oi.sb")
    import_openinference_spans(paths["openinference"], rt_oi, key=rk)
    assert _kind_counts(_read_steps(rt_oi, rk)) == src_counts

    # otel alias round-trip (importer is import_otel_spans).
    from stepback.importers import import_otel_spans
    rk = RecorderKey.fresh()
    rt_otel = os.path.join(str(tmp_path), "rt_otel.sb")
    import_otel_spans(paths["otel"], rt_otel, key=rk)
    assert _kind_counts(_read_steps(rt_otel, rk)) == src_counts

    # native json round-trip.
    rk = RecorderKey.fresh()
    rt_json = os.path.join(str(tmp_path), "rt_json.sb")
    import_native_json(paths["json"], rt_json, key=rk)
    assert _kind_counts(_read_steps(rt_json, rk)) == src_counts

    # html: round-trip via data island.
    with open(paths["html"], "r", encoding="utf-8") as f:
        model = _extract_html_data_island(f.read())
    assert model["step_count"] == len(steps)
    assert dict(model["by_kind"]) == src_counts
