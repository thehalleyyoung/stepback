"""Tests for the time-travel debugger HTML page (Step 79).

Covers:
* Step forward/back data (step indices present and ordered)
* Cache-hit display (dirty_count / cache_hit_count in data island)
* Canonical input diffs (_flat_diff helper + per-step inputs_diff)
* Causal graph data (parent_step_id links preserved)
* Security: </script> injection neutralised
* write_time_travel_html integration (file written, summary returned)
* CLI subcommand 'stepback debug'
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from typing import Any, Dict, List

import pytest

import stepback
from stepback import (
    RecorderKey,
    TimeTravelSummary,
    record,
    replay,
    render_time_travel_html,
    write_time_travel_html,
)
from stepback.html_view import _flat_diff
from stepback.testing import run_recorded_agent


# ----------------------------------------------------------- helpers


def _record_fixture(tmp_path):
    key = RecorderKey.fresh()
    path = str(tmp_path / "trace.sb")
    with record(path, key=key) as rec:
        run_recorded_agent(rec)
    return path, key


def _extract_tt_data(page: str) -> dict:
    m = re.search(
        r"<script type='application/json' id='tt-data'>(.+?)</script>",
        page,
        re.DOTALL,
    )
    assert m, "tt-data island not found in HTML"
    body = m.group(1).replace(r"<\/", "</")
    return json.loads(body)


# =========================================================== _flat_diff unit tests


def test_flat_diff_identical_returns_empty():
    assert _flat_diff({"a": 1}, {"a": 1}) == []


def test_flat_diff_changed_leaf():
    result = _flat_diff({"a": 1}, {"a": 2})
    assert len(result) == 1
    assert result[0]["kind"] == "changed"
    assert result[0]["path"] == "a"
    assert result[0]["old"] == 1
    assert result[0]["new"] == 2


def test_flat_diff_added_key():
    result = _flat_diff({}, {"x": 42})
    assert len(result) == 1
    assert result[0]["kind"] == "added"
    assert result[0]["new"] == 42


def test_flat_diff_removed_key():
    result = _flat_diff({"x": 99}, {})
    assert len(result) == 1
    assert result[0]["kind"] == "removed"
    assert result[0]["old"] == 99


def test_flat_diff_nested_dict():
    result = _flat_diff({"outer": {"inner": "a"}}, {"outer": {"inner": "b"}})
    assert len(result) == 1
    assert result[0]["path"] == "outer.inner"


def test_flat_diff_list_element_changed():
    result = _flat_diff([1, 2, 3], [1, 9, 3])
    assert len(result) == 1
    assert result[0]["path"] == "[1]"
    assert result[0]["old"] == 2
    assert result[0]["new"] == 9


def test_flat_diff_list_added():
    result = _flat_diff([1], [1, 2])
    assert len(result) == 1
    assert result[0]["kind"] == "added"
    assert result[0]["new"] == 2


def test_flat_diff_non_dict_root():
    result = _flat_diff("hello", "world")
    assert len(result) == 1
    assert result[0]["kind"] == "changed"
    assert result[0]["path"] == "(root)"


# =========================================================== render_time_travel_html


def test_render_time_travel_html_smoke_empty():
    """render_time_travel_html works on an empty replay."""
    from stepback.replay import ReplayResult

    rr = ReplayResult(
        steps=[],
        total_cost_usd=0.0,
        dirty_count=0,
        cache_hit_count=0,
        real_executions=0,
    )
    page = render_time_travel_html(rr, [], header={})
    assert page.startswith("<!doctype html>")
    assert "time-travel" in page.lower()
    data = _extract_tt_data(page)
    assert data["step_count"] == 0
    assert data["dirty_count"] == 0
    assert data["cache_hit_count"] == 0


def test_render_time_travel_html_no_external_refs(tmp_path):
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    result = t.replay_forward()
    page = render_time_travel_html(result, t.recorded_steps, t.header, title="my-debug")

    assert "my-debug" in page
    # No CDN or external link/script tags; the SVG XML namespace URI
    # (http://www.w3.org/2000/svg) lives inside the <script> block and
    # does not cause a network request.
    assert "https://" not in page
    assert "src=" not in page
    assert "<link" not in page
    # If there is an http:// it must be only the SVG XML namespace
    http_occurrences = page.count("http://")
    svg_ns_occurrences = page.count("http://www.w3.org/2000/svg")
    assert http_occurrences == svg_ns_occurrences, (
        "unexpected http:// references that are not the SVG namespace"
    )


def test_render_time_travel_step_count_matches(tmp_path):
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    result = t.replay_forward()
    page = render_time_travel_html(result, t.recorded_steps, t.header)
    data = _extract_tt_data(page)

    assert data["step_count"] == 12
    assert len(data["steps"]) == 12


def test_render_time_travel_step_indices_ordered(tmp_path):
    """Every step has a sequential index 0..N-1."""
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    result = t.replay_forward()
    page = render_time_travel_html(result, t.recorded_steps, t.header)
    data = _extract_tt_data(page)

    indices = [s["index"] for s in data["steps"]]
    assert indices == list(range(len(data["steps"])))


# =========================================================== Cache-hit display


def test_cache_hit_counts_present(tmp_path):
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    result = t.replay_forward()
    page = render_time_travel_html(result, t.recorded_steps, t.header)
    data = _extract_tt_data(page)

    assert data["dirty_count"] >= 0
    assert data["cache_hit_count"] >= 0
    assert data["dirty_count"] + data["cache_hit_count"] == data["step_count"]


def test_cache_hit_badges_in_html(tmp_path):
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    result = t.replay_forward()
    data_model = result  # 12-step clean replay → all cache hits
    page = render_time_travel_html(data_model, t.recorded_steps, t.header)

    assert "dirty" in page
    assert "cache hit" in page


def test_all_steps_are_cache_hits_without_substitution(tmp_path):
    """A plain replay without substitutions should be 100% cache hits."""
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    result = t.replay_forward()
    page = render_time_travel_html(result, t.recorded_steps, t.header)
    data = _extract_tt_data(page)

    assert data["cache_hit_count"] == 12
    assert data["dirty_count"] == 0
    for s in data["steps"]:
        assert s["cache_hit"] is True
        assert s["dirty"] is False


def test_dirty_steps_after_substitution(tmp_path):
    """After substituting step 1's output, at least that step should be dirty."""
    from stepback.substitutions import ToolOutputSubstitution
    from stepback.replay import Executor

    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    first_tool = next(s for s in t.recorded_steps if s.get("step_kind") == "tool_call")
    sub = ToolOutputSubstitution(at_step=first_tool["step_id"], fake_response="patched")
    result = t.substitute(sub).replay_forward(Executor(fallback_recorded=True))

    page = render_time_travel_html(result, t.recorded_steps, t.header)
    data = _extract_tt_data(page)
    assert data["dirty_count"] >= 1


# =========================================================== Canonical input diffs


def test_inputs_diff_absent_for_clean_hit(tmp_path):
    """Cache-hit steps must not have an inputs_diff."""
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    result = t.replay_forward()
    page = render_time_travel_html(result, t.recorded_steps, t.header)
    data = _extract_tt_data(page)

    for s in data["steps"]:
        assert s.get("inputs_diff") is None or s["inputs_diff"] == []


def test_inputs_diff_present_for_dirty_step_with_changed_inputs(tmp_path):
    """A step with substituted inputs must carry a non-empty inputs_diff."""
    from stepback.substitutions import ToolOutputSubstitution
    from stepback.replay import Executor

    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    # Find a tool step and override its outputs to trigger child dirtiness
    tool_step = next(
        s for s in t.recorded_steps if s.get("step_kind") == "tool_call"
    )
    sub = ToolOutputSubstitution(at_step=tool_step["step_id"], fake_response="INJECTED_VALUE")
    result = t.substitute(sub).replay_forward(Executor(fallback_recorded=True))

    page = render_time_travel_html(result, t.recorded_steps, t.header)
    data = _extract_tt_data(page)

    # Find the dirty step that had its inputs change downstream
    dirty_steps_with_diff = [
        s for s in data["steps"]
        if s["dirty"] and s.get("inputs_diff") and len(s["inputs_diff"]) > 0
    ]
    # There must be at least one step with a meaningful diff
    assert len(dirty_steps_with_diff) >= 1, (
        f"expected at least 1 step with inputs_diff after tool substitution; "
        f"dirty steps: {[s['step_id'] for s in data['steps'] if s['dirty']]}"
    )


def test_inputs_diff_structure(tmp_path):
    """Each inputs_diff entry must have path, old, new, kind."""
    from stepback.substitutions import ToolOutputSubstitution
    from stepback.replay import Executor

    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    tool_step = next(
        s for s in t.recorded_steps if s.get("step_kind") == "tool_call"
    )
    sub = ToolOutputSubstitution(at_step=tool_step["step_id"], fake_response="X")
    result = t.substitute(sub).replay_forward(Executor(fallback_recorded=True))
    page = render_time_travel_html(result, t.recorded_steps, t.header)
    data = _extract_tt_data(page)

    for s in data["steps"]:
        if not s.get("inputs_diff"):
            continue
        for entry in s["inputs_diff"]:
            assert "path" in entry
            assert "old" in entry
            assert "new" in entry
            assert "kind" in entry
            assert entry["kind"] in ("changed", "added", "removed")


# =========================================================== Causal graph


def test_causal_graph_parent_links_preserved(tmp_path):
    """parent_step_id must be included for every step that has a parent."""
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    result = t.replay_forward()
    page = render_time_travel_html(result, t.recorded_steps, t.header)
    data = _extract_tt_data(page)

    recorded_parents = {
        s["step_id"]: s.get("parent_step_id")
        for s in t.recorded_steps
    }
    for sv in data["steps"]:
        assert sv.get("parent_step_id") == recorded_parents.get(sv["step_id"]), (
            f"step {sv['step_id']}: parent mismatch"
        )


def test_causal_graph_pane_elements_in_html(tmp_path):
    """The graph pane and navigation buttons must be present in the HTML."""
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    result = t.replay_forward()
    page = render_time_travel_html(result, t.recorded_steps, t.header)

    # Navigation controls
    assert "id='prev-btn'" in page
    assert "id='next-btn'" in page
    assert "id='step-pos'" in page
    # Mode toggle
    assert "id='detail-btn'" in page
    assert "id='graph-btn'" in page
    # Graph pane
    assert "id='graph-pane'" in page
    # Detail pane
    assert "id='detail-pane'" in page
    # Timeline list
    assert "id='timeline'" in page


# =========================================================== Security


def test_script_tag_injection_neutralised():
    """</script> in step data must not break out of the data island."""
    from stepback.replay import ReplayResult, StepView, StepProvenance

    prov = StepProvenance(
        dirty_reason=None,
        cache_source="recorded_trace",
        seed=None,
        model=None,
        provider=None,
        model_version=None,
        provider_version=None,
        policy_blocked=False,
        policy_reason=None,
        executor_version="test",
    )
    sv = StepView(
        step_id="step:1",
        kind="tool_call",
        name="evil",
        parent_step_id=None,
        inputs={"x": "</script><script>alert(1)</script>"},
        outputs={"r": "</script>pwned"},
        cost_usd=0.0,
        dirty=False,
        cache_hit=True,
        recorded_inputs_hash="sha256:aaa",
        current_inputs_hash="sha256:aaa",
        provenance=prov,
    )
    rr = ReplayResult(
        steps=[sv],
        total_cost_usd=0.0,
        dirty_count=0,
        cache_hit_count=1,
        real_executions=0,
    )
    page = render_time_travel_html(rr, [], header={})

    island_m = re.search(
        r"<script type='application/json' id='tt-data'>(.+?)</script>",
        page,
        re.DOTALL,
    )
    assert island_m
    body = island_m.group(1)
    assert "</script>" not in body

    # Data still recoverable
    data = _extract_tt_data(page)
    assert data["steps"][0]["current_inputs"]["x"].endswith("</script>")


def test_html_special_chars_in_title_escaped():
    from stepback.replay import ReplayResult

    rr = ReplayResult(
        steps=[], total_cost_usd=0.0, dirty_count=0,
        cache_hit_count=0, real_executions=0,
    )
    page = render_time_travel_html(rr, [], header={},
                                   title="<img src=x onerror=alert(1)>")
    assert "<img src=x onerror=alert(1)>" not in page


# =========================================================== write_time_travel_html


def test_write_time_travel_html_creates_file(tmp_path):
    path, key = _record_fixture(tmp_path)
    out = tmp_path / "debug" / "trace.html"
    summary = write_time_travel_html(str(path), str(out), hmac_key=key.hmac_key)

    assert isinstance(summary, TimeTravelSummary)
    assert summary.step_count == 12
    assert summary.dirty_count == 0
    assert summary.cache_hit_count == 12
    assert summary.bytes_written > 1000
    assert out.exists()
    page = out.read_text()
    assert page.startswith("<!doctype html>")
    assert page.rstrip().endswith("</html>")


def test_write_time_travel_html_creates_missing_dirs(tmp_path):
    path, key = _record_fixture(tmp_path)
    out = tmp_path / "deep" / "nested" / "debug.html"
    summary = write_time_travel_html(str(path), str(out))
    assert out.exists()
    assert summary.bytes_written > 0


# =========================================================== CLI subcommand


def test_cli_debug_subcommand(tmp_path):
    path, key = _record_fixture(tmp_path)
    out = tmp_path / "debug.html"
    result = subprocess.run(
        [
            sys.executable, "-m", "stepback.cli", "debug",
            str(path), "--output", str(out),
            "--hmac-key-hex", key.hmac_key.hex(),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    assert "wrote" in result.stdout


def test_cli_debug_subcommand_json_output(tmp_path):
    path, key = _record_fixture(tmp_path)
    out = tmp_path / "debug2.html"
    result = subprocess.run(
        [
            sys.executable, "-m", "stepback.cli", "debug",
            str(path), "--output", str(out),
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["step_count"] == 12
    assert "dirty_count" in data
    assert "cache_hit_count" in data
    assert "bytes_written" in data


# =========================================================== Public API


def test_public_api_exports():
    assert hasattr(stepback, "TimeTravelSummary")
    assert hasattr(stepback, "render_time_travel_html")
    assert hasattr(stepback, "write_time_travel_html")
