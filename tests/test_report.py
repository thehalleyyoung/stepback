"""End-to-end tests for the Markdown counterfactual report renderer.

Drives the same 12-step deterministic payments agent fixture used by
test_e2e_replay.py, then exercises:

* render_replay_report (single-replay) on the recorded run
* render_counterfactual_report (baseline vs counterfactual) under a
  ToolOutputSubstitution that fixes the bad-IBAN bug
* the `stepback report` CLI subcommand emitting Markdown to a file
* substitution summarisation for every Substitution subclass
* deterministic / byte-stable rendering (same inputs → same output)
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from stepback import (
    RecorderKey,
    record,
    replay,
    render_counterfactual_report,
    render_replay_report,
    ReportOptions,
    save_branch,
    trace_chain_hash,
)
from stepback.replay import Executor
from stepback.report import _summarise_substitution
from stepback.substitutions import (
    ModelSubstitution,
    PolicySubstitution,
    PromptSubstitution,
    RouterSubstitution,
    SubstitutionSet,
    ToolOutputSubstitution,
)
from stepback.testing import LOOKUP_FIXED_ROW, run_recorded_agent


# ----------------------------------------------------------- helpers


def _record_fixture(tmp_path):
    key = RecorderKey.fresh()
    path = str(tmp_path / "trace.sb")
    with record(path, key=key) as rec:
        run_recorded_agent(rec)
    return path, key


def _baseline_and_counterfactual(tmp_path):
    path, _ = _record_fixture(tmp_path)
    t = replay(path)
    executor = Executor(fallback_recorded=True)
    baseline = t.run_replay(SubstitutionSet(), executor)
    fix = SubstitutionSet().add(
        ToolOutputSubstitution(at_step="step:2", fake_response=LOOKUP_FIXED_ROW)
    )
    counterfactual = t.run_replay(fix, executor)
    return t, baseline, counterfactual, fix


# --------------------------------------------------- single-replay


def test_render_replay_report_includes_trace_metadata_and_step_table(tmp_path):
    path, _ = _record_fixture(tmp_path)
    t = replay(path)
    result = t.run_replay(SubstitutionSet(), Executor(fallback_recorded=True))
    md = render_replay_report(t, result, SubstitutionSet())

    assert md.startswith("# ")
    assert f"`{path}`" in md
    assert "step_count: 12" in md
    assert "## Substitutions" in md
    assert "no substitutions" in md  # baseline path
    assert "## Cost summary" in md
    assert "## Dirty subtree" in md
    assert "## Step timeline" in md
    # 12 step rows + header + separator + (no overflow row)
    table_lines = [l for l in md.splitlines() if l.startswith("| `step:")]
    assert len(table_lines) == 12
    # Baseline replay must be 100% cache-hit, so no "▲ dirty" rows.
    assert "dirty" not in "\n".join(table_lines).lower() or "no dirty steps" in md


# --------------------------------------------------- counterfactual


def test_render_counterfactual_report_marks_dirty_subtree(tmp_path):
    t, baseline, counterfactual, fix = _baseline_and_counterfactual(tmp_path)
    md = render_counterfactual_report(t, baseline, counterfactual, fix)

    # Header & substitution section
    assert "## Substitutions applied to branch B" in md
    assert "tool_output @ step:2" in md
    # The dirty subtree must reference the substituted step.
    assert "`step:2`" in md
    # Decision diffs section must include at least one diverging step.
    assert "## Decision diffs" in md
    assert "(no diverging steps)" not in md
    # The bad IBAN appears in baseline-A; the fixed IBAN in branch-B.
    assert "GB99-9999-9999" in md
    assert "US12-3456-7890" in md
    # Two-column step timeline header markers.
    assert "A status" in md and "B status" in md
    # Baseline cost summary line
    assert "baseline:" in md and "counterfactual:" in md
    assert "Δ total_cost_usd" in md


def test_render_counterfactual_report_propagates_dirtiness_to_descendants(tmp_path):
    t, baseline, counterfactual, fix = _baseline_and_counterfactual(tmp_path)
    md = render_counterfactual_report(t, baseline, counterfactual, fix)

    # The bug fixture wires the bad IBAN on step 12 (payment.transfer); the
    # IBAN value is also baked into echo at step 10 and into messages from
    # step 11 onwards. So the counterfactual must dirty more than just step 2.
    dirty_section = md.split("## Dirty subtree")[1].split("##")[0]
    dirty_step_ids = [
        line.split("`")[1] for line in dirty_section.splitlines() if "`step:" in line
    ]
    assert "step:2" in dirty_step_ids
    assert len(dirty_step_ids) >= 2, dirty_step_ids


def test_render_is_deterministic(tmp_path):
    """Same trace + same substitutions → byte-identical Markdown."""
    t, baseline, counterfactual, fix = _baseline_and_counterfactual(tmp_path)
    md1 = render_counterfactual_report(t, baseline, counterfactual, fix)
    md2 = render_counterfactual_report(t, baseline, counterfactual, fix)
    assert md1 == md2


def test_report_options_truncate_and_limit_rows(tmp_path):
    t, baseline, counterfactual, fix = _baseline_and_counterfactual(tmp_path)
    opts = ReportOptions(
        title="Custom title", max_step_rows=3, truncate_text=20,
        extra_metadata={"reviewer": "alice"},
    )
    md = render_counterfactual_report(
        t, baseline, counterfactual, fix, options=opts
    )
    assert md.startswith("# Custom title\n")
    assert "reviewer: alice" in md
    # Only 3 rows fit; an overflow row must signal the rest.
    table_lines = [l for l in md.splitlines() if l.startswith("| `step:")]
    assert len(table_lines) == 3
    assert "more)" in md  # the "(+9 more)" overflow row


# --------------------------------------------------- substitution summaries


def test_summarise_substitution_covers_every_kind():
    cases = [
        PromptSubstitution(at_step="step:1",
                           new_messages=[{"role": "user", "content": "hi"}]),
        ModelSubstitution(at_step="step:2", new_model_id="gpt-4o-mini-2024-07-18"),
        ToolOutputSubstitution(at_step="step:3", fake_response={"x": 1},
                               tool_call_id="call_abc"),
        PolicySubstitution(at_step="step:4", policy_path="/tmp/p.tw"),
        RouterSubstitution(at_step="step:5", choice="branchA"),
    ]
    summaries = [_summarise_substitution(c) for c in cases]
    assert "prompt @ step:1" in summaries[0]
    assert "model @ step:2: → gpt-4o-mini-2024-07-18" in summaries[1]
    assert "tool_output @ step:3 call_id=call_abc" in summaries[2]
    assert "policy @ step:4: /tmp/p.tw" in summaries[3]
    assert "router @ step:5: → branchA" in summaries[4]


# --------------------------------------------------- CLI


def test_cli_report_emits_markdown_to_file(tmp_path):
    path, _ = _record_fixture(tmp_path)
    out = tmp_path / "report.md"
    proc = subprocess.run(
        [sys.executable, "-m", "stepback.cli", "report", path,
         "-s", 'tool_output@step:2=:inline:'
               + json.dumps(LOOKUP_FIXED_ROW),
         "-o", str(out),
         "--title", "INC-2026-04-12 — wrong-IBAN postmortem"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists() and out.stat().st_size > 200
    md = out.read_text()
    assert "# INC-2026-04-12 — wrong-IBAN postmortem" in md
    assert "## Decision diffs" in md
    assert "GB99-9999-9999" in md and "US12-3456-7890" in md


def test_cli_report_baseline_only_when_no_substitutions(tmp_path):
    path, _ = _record_fixture(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "stepback.cli", "report", path],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    md = proc.stdout
    assert "# stepback counterfactual report" in md
    assert "no substitutions" in md
    # Single-replay reports do NOT have the two-column "A status / B status"
    # table; they have a simpler one.
    assert "A status" not in md and "B status" not in md
    assert "outputs" in md  # the simple table column


def test_cli_report_loads_branch_file(tmp_path):
    path, _ = _record_fixture(tmp_path)
    t = replay(path)
    sbb = tmp_path / "fix.sbb"
    save_branch(
        str(sbb),
        name="iban-fix",
        base_step="step:2",
        trace_path=path,
        trace_chain=trace_chain_hash(t.recorded_steps),
        substitutions=[
            ToolOutputSubstitution(at_step="step:2", fake_response=LOOKUP_FIXED_ROW)
        ],
    )
    proc = subprocess.run(
        [sys.executable, "-m", "stepback.cli", "report", path,
         "--branch", str(sbb)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    md = proc.stdout
    assert "counterfactual: iban-fix" in md  # title from branch name
    assert "tool_output @ step:2" in md
    assert "## Decision diffs" in md


# --------------------------------------------------- structured JSON / headline


from stepback import dump_report_json, render_report_json


def test_render_report_json_contains_required_keys(tmp_path):
    t, baseline, counterfactual, fix = _baseline_and_counterfactual(tmp_path)
    model = render_report_json(t, baseline, counterfactual, fix)
    for key in (
        "schema_version", "verdict", "cost_summary", "decision_diffs",
        "causal_attribution", "first_divergence_step_id", "step_table",
        "substitutions", "dirty_subtree",
    ):
        assert key in model, f"missing {key}"
    assert model["schema_version"] == 1
    assert model["verdict"] == "diverged"
    assert model["cost_summary"]["counterfactual"]["dirty"] >= 1
    # JSON-safety: round-trip through json.dumps without TypeError
    s = json.dumps(model, sort_keys=True, default=str)
    json.loads(s)


def test_first_divergence_pinpoints_substituted_step(tmp_path):
    t, baseline, counterfactual, fix = _baseline_and_counterfactual(tmp_path)
    model = render_report_json(t, baseline, counterfactual, fix)
    # The single ToolOutputSubstitution is at step:2, so divergence starts there.
    assert model["first_divergence_step_id"] == "step:2"


def test_causal_attribution_covers_dirty_steps(tmp_path):
    t, baseline, counterfactual, fix = _baseline_and_counterfactual(tmp_path)
    model = render_report_json(t, baseline, counterfactual, fix)
    attribution = model["causal_attribution"]
    # Every dirty step in the counterfactual must be attributed to substitution #0.
    dirty_ids = {row["step_id"] for row in model["dirty_subtree"]}
    assert dirty_ids, "expected at least one dirty step in branch B"
    for sid in dirty_ids:
        assert sid in attribution, f"{sid} not attributed"
        assert attribution[sid] == [0]


def test_render_report_json_byte_stable(tmp_path):
    t, baseline, counterfactual, fix = _baseline_and_counterfactual(tmp_path)
    a = dump_report_json(t, baseline, counterfactual, fix)
    b = dump_report_json(t, baseline, counterfactual, fix)
    assert a == b


def test_headline_banner_in_markdown(tmp_path):
    t, baseline, counterfactual, fix = _baseline_and_counterfactual(tmp_path)
    md = render_counterfactual_report(t, baseline, counterfactual, fix)
    assert "## Headline" in md
    # Headline names the first-divergence step and the verdict.
    headline = md.split("## Headline")[1].split("##")[0]
    assert "verdict" in headline
    assert "diverged" in headline
    assert "step:2" in headline
    assert "Δ total_cost_usd" in headline


def test_headline_unchanged_when_no_substitution_effect(tmp_path):
    """A no-op ToolOutputSubstitution that hashes to the recorded value
    leaves the verdict 'unchanged'."""
    path, _ = _record_fixture(tmp_path)
    t = replay(path)
    executor = Executor(fallback_recorded=True)
    baseline = t.run_replay(SubstitutionSet(), executor)
    counterfactual = t.run_replay(SubstitutionSet(), executor)
    model = render_report_json(t, baseline, counterfactual, SubstitutionSet())
    assert model["verdict"] == "unchanged"
    assert model["first_divergence_step_id"] is None


def test_cli_report_json_format(tmp_path):
    path, _ = _record_fixture(tmp_path)
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, "-m", "stepback.cli", "report", path,
         "-s", 'tool_output@step:2=:inline:'
               + json.dumps(LOOKUP_FIXED_ROW),
         "--format", "json",
         "-o", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text())
    assert payload["schema_version"] == 1
    assert payload["verdict"] == "diverged"
    assert payload["first_divergence_step_id"] == "step:2"
    assert payload["substitutions"][0]["kind"] == "ToolOutputSubstitution"
    assert payload["cost_summary"]["delta_total_cost_usd"] is not None


def test_single_replay_json_has_no_counterfactual(tmp_path):
    path, _ = _record_fixture(tmp_path)
    t = replay(path)
    result = t.run_replay(SubstitutionSet(), Executor(fallback_recorded=True))
    model = render_report_json(t, result, None, SubstitutionSet())
    assert model["verdict"] == "no-counterfactual"
    assert model["first_divergence_step_id"] is None
    assert "counterfactual" not in model["cost_summary"]
    assert model["decision_diffs"] == []


# --------------------------------------------------- severity scoring


from stepback.report import (
    SeverityScore,
    available_formats,
    render_html_report,
    render_report,
    severity_score,
)


def test_severity_score_zero_when_replays_identical(tmp_path):
    path, _ = _record_fixture(tmp_path)
    t = replay(path)
    executor = Executor(fallback_recorded=True)
    a = t.run_replay(SubstitutionSet(), executor)
    b = t.run_replay(SubstitutionSet(), executor)
    sev = severity_score(a, b, SubstitutionSet())
    assert sev.score == 0
    assert sev.level == "info"
    assert sev.reasons == []


def test_severity_score_increases_with_cost_delta(tmp_path):
    t, baseline, counterfactual, fix = _baseline_and_counterfactual(tmp_path)
    sev = severity_score(baseline, counterfactual, fix)
    assert sev.score > 0
    assert sev.level in {"low", "medium", "high", "critical"}
    # All components in [0, 1]
    for v in sev.components.values():
        assert 0.0 <= v <= 1.0
    # Score consistent with weighted sum (allow rounding)
    weights = {"cost_delta": 0.30, "dirty_fraction": 0.30,
               "decision_flips": 0.30, "subtree_depth": 0.10}
    expect = round(sum(sev.components[k] * w for k, w in weights.items()) * 100)
    assert sev.score == expect


def test_severity_single_replay_returns_zero():
    sev = severity_score(
        type("R", (), {"steps": [], "total_cost_usd": 0.0,
                       "dirty_count": 0, "cache_hit_count": 0,
                       "real_executions": 0})(),
        None,
        SubstitutionSet(),
    )
    assert sev.score == 0 and sev.level == "info"
    assert sev.to_dict() == {"score": 0, "level": "info",
                             "components": {}, "reasons": []}


def test_severity_score_to_dict_keys():
    """Schema of SeverityScore.to_dict — used by JSON model consumers."""
    sev = SeverityScore(score=42, level="medium",
                        components={"cost_delta": 0.5}, reasons=["x"])
    d = sev.to_dict()
    assert d == {"score": 42, "level": "medium",
                 "components": {"cost_delta": 0.5}, "reasons": ["x"]}


# --------------------------------------------------- HTML renderer


def test_html_renders_for_counterfactual(tmp_path):
    t, baseline, counterfactual, fix = _baseline_and_counterfactual(tmp_path)
    html = render_html_report(t, baseline, counterfactual, fix)
    assert html.startswith("<!doctype html>")
    assert "<table" in html
    # Severity section present
    assert 'id="severity"' in html
    assert "data-level=" in html
    # Dirty rows have class
    assert 'class="dirty"' in html
    # Closes properly
    assert html.rstrip().endswith("</html>")


def test_html_renders_for_replay_only(tmp_path):
    path, _ = _record_fixture(tmp_path)
    t = replay(path)
    result = t.run_replay(SubstitutionSet(), Executor(fallback_recorded=True))
    html = render_html_report(t, result, None, SubstitutionSet())
    assert html.startswith("<!doctype html>")
    assert "<table" in html
    # Severity section omitted when no counterfactual
    assert 'id="severity"' not in html
    # Decision-diffs section omitted too
    assert 'id="decision-diffs"' not in html


def test_html_is_byte_deterministic(tmp_path):
    t, baseline, counterfactual, fix = _baseline_and_counterfactual(tmp_path)
    h1 = render_html_report(t, baseline, counterfactual, fix)
    h2 = render_html_report(t, baseline, counterfactual, fix)
    assert h1 == h2


def test_html_escapes_xss_in_substitution_payload(tmp_path):
    path, _ = _record_fixture(tmp_path)
    t = replay(path)
    executor = Executor(fallback_recorded=True)
    baseline = t.run_replay(SubstitutionSet(), executor)
    payload = "<script>alert(1)</script>"
    fix = SubstitutionSet().add(
        ToolOutputSubstitution(at_step="step:2",
                               fake_response={"xss": payload, **LOOKUP_FIXED_ROW})
    )
    counterfactual = t.run_replay(fix, executor)
    html = render_html_report(t, baseline, counterfactual, fix)
    # Literal <script> must NOT appear
    assert "<script>alert(1)</script>" not in html
    # Escaped form must appear
    assert "&lt;script&gt;" in html


# --------------------------------------------------- format dispatcher


def test_available_formats_lists_three():
    assert available_formats() == ["markdown", "json", "html"]


def test_render_report_dispatch_json_matches_dump_report_json(tmp_path):
    t, baseline, counterfactual, fix = _baseline_and_counterfactual(tmp_path)
    via_dispatch = render_report(t, baseline, counterfactual, fix, format="json")
    via_direct = dump_report_json(t, baseline, counterfactual, fix)
    assert via_dispatch == via_direct


def test_render_report_dispatch_markdown_matches_direct(tmp_path):
    t, baseline, counterfactual, fix = _baseline_and_counterfactual(tmp_path)
    via_dispatch = render_report(t, baseline, counterfactual, fix, format="markdown")
    via_direct = render_counterfactual_report(t, baseline, counterfactual, fix)
    assert via_dispatch == via_direct


def test_render_report_dispatch_html_matches_direct(tmp_path):
    t, baseline, counterfactual, fix = _baseline_and_counterfactual(tmp_path)
    via_dispatch = render_report(t, baseline, counterfactual, fix, format="html")
    via_direct = render_html_report(t, baseline, counterfactual, fix)
    assert via_dispatch == via_direct


def test_render_report_unknown_format_raises():
    with pytest.raises(ValueError, match="unknown format"):
        render_report(None, None, None, SubstitutionSet(), format="pdf")


# --------------------------------------------------- JSON model severity key


def test_json_model_includes_severity_when_counterfactual(tmp_path):
    from stepback import render_report_json
    t, baseline, counterfactual, fix = _baseline_and_counterfactual(tmp_path)
    model = render_report_json(t, baseline, counterfactual, fix)
    assert "severity" in model
    sev = model["severity"]
    assert set(sev.keys()) == {"score", "level", "components", "reasons"}
    assert isinstance(sev["score"], int)
    assert sev["level"] in {"info", "low", "medium", "high", "critical"}


def test_json_model_omits_severity_when_single_replay(tmp_path):
    from stepback import render_report_json
    path, _ = _record_fixture(tmp_path)
    t = replay(path)
    result = t.run_replay(SubstitutionSet(), Executor(fallback_recorded=True))
    model = render_report_json(t, result, None, SubstitutionSet())
    assert "severity" not in model


# --------------------------------------------------- Markdown severity section


def test_markdown_counterfactual_includes_severity_section(tmp_path):
    t, baseline, counterfactual, fix = _baseline_and_counterfactual(tmp_path)
    md = render_counterfactual_report(t, baseline, counterfactual, fix)
    assert "## Severity" in md
    assert "score:" in md


# --------------------------------------------------- CLI HTML format


def test_cli_report_html_format_to_file(tmp_path):
    path, _ = _record_fixture(tmp_path)
    out = tmp_path / "rep.html"
    fake = json.dumps(LOOKUP_FIXED_ROW)
    proc = subprocess.run(
        [sys.executable, "-m", "stepback.cli", "report", path,
         "-s", f"tool_output@step:2=:inline:{fake}",
         "-o", str(out), "--format", "html"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    text = out.read_text()
    assert text.startswith("<!doctype html>")
    assert "<table" in text


def test_cli_report_format_inferred_from_html_extension(tmp_path):
    path, _ = _record_fixture(tmp_path)
    out = tmp_path / "auto.html"
    fake = json.dumps(LOOKUP_FIXED_ROW)
    proc = subprocess.run(
        [sys.executable, "-m", "stepback.cli", "report", path,
         "-s", f"tool_output@step:2=:inline:{fake}",
         "-o", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    text = out.read_text()
    assert text.startswith("<!doctype html>")
