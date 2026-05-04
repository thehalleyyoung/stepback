"""Tests for the self-contained interactive HTML trace viewer.

End-to-end fixture: record the 12-step `customer payments` agent in
``tests/fixtures/agent.py``, render the viewer over the resulting
.sb trace, and assert structural and behavioural properties of the
produced HTML.
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import sys

import pytest

from stepback import (
    TraceViewSummary,
    record,
    RecorderKey,
    render_trace_html,
    replay,
    write_trace_html,
)
from stepback.testing import run_recorded_agent


# ----------------------------------------------------------- helpers


def _record_fixture(tmp_path):
    key = RecorderKey.fresh()
    path = str(tmp_path / "trace.sb")
    with record(path, key=key) as rec:
        run_recorded_agent(rec)
    return path, key


def _extract_data_island(page: str) -> dict:
    m = re.search(
        r"<script type='application/json' id='stepback-data'>(.+?)</script>",
        page,
        re.DOTALL,
    )
    assert m, "data island not found in HTML"
    body = m.group(1)
    # The renderer escapes `</` to `<\/` to prevent script-tag injection
    body = body.replace(r"<\/", "</")
    return json.loads(body)


# ------------------------------------------------------- pure render


def test_render_trace_html_smoke_minimal():
    page = render_trace_html([], header={})
    assert page.startswith("<!doctype html>")
    assert "stepback trace" in page
    assert "<style>" in page and "</style>" in page
    assert "<script type='application/json' id='stepback-data'>" in page
    data = _extract_data_island(page)
    assert data["step_count"] == 0
    assert data["steps"] == []


def test_render_trace_html_self_contained_no_external_refs(tmp_path):
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    page = render_trace_html(t.recorded_steps, t.header, title="incident-2026")

    assert "incident-2026" in page
    # no CDN / external network references
    assert "http://" not in page
    assert "https://" not in page
    assert "src=" not in page
    assert "<link" not in page


# -------------------------------------------------- trace projection


def test_render_includes_every_recorded_step(tmp_path):
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    page = render_trace_html(t.recorded_steps, t.header)
    data = _extract_data_island(page)

    assert data["step_count"] == 12
    assert data["by_kind"]["llm_call"] == 6
    assert data["by_kind"]["tool_call"] == 6

    rendered_ids = {s["step_id"] for s in data["steps"]}
    recorded_ids = {s["step_id"] for s in t.recorded_steps}
    assert rendered_ids == recorded_ids

    for v, s in zip(data["steps"], t.recorded_steps):
        assert v["step_id"] == s["step_id"]
        assert v["kind"] == s["step_kind"]
        # cost surfaces in the view model
        assert isinstance(v["cost_usd"], (int, float))


def test_summaries_capture_user_visible_payment_information(tmp_path):
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    page = render_trace_html(t.recorded_steps, t.header)
    data = _extract_data_island(page)

    # at least one llm_call summary must reflect the conversation
    llm_summaries = [s["summary"] for s in data["steps"] if s["kind"] == "llm_call"]
    assert any("Acme Bolts" in s or "wired" in s or "lookup" in s
               for s in llm_summaries), llm_summaries

    # at least one tool_call summary contains the lookup args
    tool_summaries = [s["summary"] for s in data["steps"] if s["kind"] == "tool_call"]
    assert any("Acme Bolts" in s or "lookup_customer" in s or "amount" in s
               for s in tool_summaries), tool_summaries


def test_total_cost_matches_sum_of_recorded_steps(tmp_path):
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    page = render_trace_html(t.recorded_steps, t.header)
    data = _extract_data_island(page)

    expected = sum(float(s.get("cost_usd") or 0.0) for s in t.recorded_steps)
    assert data["total_cost_usd"] == pytest.approx(expected, abs=1e-9)


def test_kind_filter_chips_appear_for_every_observed_kind(tmp_path):
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    page = render_trace_html(t.recorded_steps, t.header)

    assert "data-kind='llm_call'" in page
    assert "data-kind='tool_call'" in page
    # checkbox, search field, and detail pane all wired up
    assert "id='q'" in page
    assert "id='timeline'" in page
    assert "id='detail'" in page


# ------------------------------------------------------- security


def test_script_tag_in_step_payload_is_neutralised(tmp_path):
    """A user-controlled `</script>` token must not break out of the
    data island."""
    malicious = [
        {
            "step_id": "step:1",
            "step_kind": "tool_call",
            "name": "evil",
            "outputs": {"text": "</script><script>window.pwned=1;</script>"},
            "cost_usd": 0.0,
        }
    ]
    page = render_trace_html(malicious, header={})
    # the literal closing tag must not appear inside the data island
    island_match = re.search(
        r"<script type='application/json' id='stepback-data'>(.+?)</script>",
        page,
        re.DOTALL,
    )
    assert island_match
    body = island_match.group(1)
    assert "</script>" not in body
    # but the data is still recoverable with the trivial unescape
    data = _extract_data_island(page)
    assert data["steps"][0]["raw"]["outputs"]["text"].endswith("</script>")


def test_html_special_chars_in_title_are_escaped():
    page = render_trace_html([], header={}, title="<img src=x onerror=alert(1)>")
    assert "<img src=x onerror=alert(1)>" not in page
    assert html.escape("<img src=x onerror=alert(1)>", quote=True) in page


# ------------------------------------------------------- write + CLI


def test_write_trace_html_creates_file_and_returns_summary(tmp_path):
    path, key = _record_fixture(tmp_path)
    out = tmp_path / "view" / "trace.html"
    summary = write_trace_html(str(path), str(out), hmac_key=key.hmac_key)

    assert isinstance(summary, TraceViewSummary)
    assert summary.step_count == 12
    assert summary.by_kind == {"llm_call": 6, "tool_call": 6}
    assert summary.bytes_written > 1000
    assert out.exists()
    page = out.read_text()
    assert page.startswith("<!doctype html>")
    assert page.rstrip().endswith("</html>")


def test_write_trace_html_creates_missing_parent_dirs(tmp_path):
    path, key = _record_fixture(tmp_path)
    out = tmp_path / "deep" / "nested" / "out.html"
    summary = write_trace_html(str(path), str(out), hmac_key=key.hmac_key)
    assert out.exists()
    assert summary.output_path == str(out)


def test_cli_view_command_writes_html(tmp_path):
    path, key = _record_fixture(tmp_path)
    out = tmp_path / "trace.html"
    rc = subprocess.call(
        [
            sys.executable, "-m", "stepback.cli", "view",
            str(path), "-o", str(out), "--title", "from CLI",
        ]
    )
    assert rc == 0
    assert out.exists()
    page = out.read_text()
    assert "from CLI" in page
    data = _extract_data_island(page)
    assert data["step_count"] == 12


def test_cli_view_command_emits_json_summary(tmp_path):
    path, key = _record_fixture(tmp_path)
    out = tmp_path / "trace.html"
    proc = subprocess.run(
        [
            sys.executable, "-m", "stepback.cli", "view",
            str(path), "-o", str(out), "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    body = json.loads(proc.stdout)
    assert body["step_count"] == 12
    assert body["by_kind"]["llm_call"] == 6
    assert body["bytes_written"] > 0
    assert body["output_path"].endswith("trace.html")
