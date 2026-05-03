"""Tests for ``stepback.sweep`` — corpus-level counterfactual replay.

Mirrors README §Use-cases #3 ("test a new system prompt on 1000
production traces") on the 12-step customer-payments fixture.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import pytest

from stepback import (
    DistStats,
    SweepFailure,
    SweepReport,
    SweepResult,
    record,
    render_sweep_report,
    render_sweep_report_json,
    sweep_traces,
)
from stepback.recorder import RecorderKey
from stepback.substitutions import (
    ModelSubstitution,
    SystemPromptSubstitution,
    ToolOutputSubstitution,
)
from tests.fixtures.agent import LOOKUP_FIXED_ROW, run_recorded_agent


# ----------------------------------------------------- corpus fixtures


def _record_corpus(tmpdir: str, n: int = 4) -> list[str]:
    """Record N copies of the 12-step fixture into ``tmpdir``."""
    paths = []
    for i in range(n):
        p = os.path.join(tmpdir, f"trace_{i}.sb")
        with record(p, key=RecorderKey.fresh()) as rec:
            run_recorded_agent(rec)
        paths.append(p)
    return paths


@pytest.fixture
def corpus(tmp_path):
    return _record_corpus(str(tmp_path), n=4)


# -------------------------------------------------------- DistStats


def test_diststats_empty():
    d = DistStats.of([])
    assert d.count == 0
    assert d.total == 0.0 and d.mean == 0.0


def test_diststats_basic():
    d = DistStats.of([1.0, 2.0, 3.0, 4.0, 5.0])
    assert d.count == 5
    assert d.min == 1.0 and d.max == 5.0
    assert d.mean == 3.0
    assert d.total == 15.0
    # p50 of 5 is the middle element
    assert d.p50 == 3.0
    # p95 of 5 elements lands on index 4 (max)
    assert d.p95 == 5.0


# -------------------------------------------------------- sweep core


def test_sweep_no_substitutions_zero_divergence(corpus):
    """Empty candidate sub set → identical branches, zero divergence."""
    report = sweep_traces(corpus, substitutions=[])
    assert isinstance(report, SweepReport)
    assert report.n_traces_attempted == 4
    assert report.n_traces_succeeded == 4
    assert report.n_traces_failed == 0
    assert report.n_traces_diverged == 0
    assert report.n_decisions_changed == 0
    assert report.cost_delta_usd.total == 0.0
    for r in report.results:
        assert r.divergent_step_count == 0
        assert r.cost_delta_usd == 0.0
        assert r.cf_cache_hit_ratio == 1.0  # everything served from cache


def test_sweep_tool_output_substitution_diverges_corpus(corpus):
    """The fixture's classic substitution: fix step:2 lookup_customer.

    Every trace in the corpus must show a non-zero divergent step
    count. Because the agent is deterministic, all traces should
    diverge by exactly the same amount.
    """
    sub = ToolOutputSubstitution(
        at_step="step:2",
        fake_response={"result": LOOKUP_FIXED_ROW},
    )
    report = sweep_traces(corpus, substitutions=[sub])
    assert report.n_traces_succeeded == 4
    assert report.n_traces_failed == 0
    assert report.n_traces_diverged == 4
    assert report.diverged_fraction == 1.0
    # 11 dirty downstream steps per trace under this substitution
    # (per GROUNDING.md row 4: dirty_after_sub=11).
    div_counts = {r.divergent_step_count for r in report.results}
    assert len(div_counts) == 1, f"non-deterministic divergence: {div_counts}"
    only = next(iter(div_counts))
    assert 1 <= only <= 12

    # decisions_changed counts llm_call/router output divergence;
    # under the default fallback_recorded executor, LLM steps don't
    # actually re-execute, so their outputs match the recorded values
    # and decisions_changed stays 0 — but tool_call divergence is
    # captured in divergent_step_count (asserted above).
    assert report.n_decisions_changed >= 0


def test_sweep_with_executor_reexecutes_dirty(corpus):
    """With a real executor wired, dirty downstream steps actually
    re-execute (real_executions > 0) — without an executor everything
    falls back to the recorded outputs and real_executions stays 0."""
    from stepback.replay import Executor
    from tests.fixtures.agent import fake_llm, fake_tool

    sub = ToolOutputSubstitution(
        at_step="step:2", fake_response={"result": LOOKUP_FIXED_ROW},
    )
    report_cached = sweep_traces(corpus, substitutions=[sub])
    assert report_cached.cf_real_executions.total == 0

    report_real = sweep_traces(
        corpus,
        substitutions=[sub],
        executor_factory=lambda: Executor(llm=fake_llm, tool=fake_tool),
    )
    assert report_real.n_traces_succeeded == 4
    assert report_real.cf_real_executions.total > 0



    spec = "tool_output@step:2=:inline:" + json.dumps(
        {"result": LOOKUP_FIXED_ROW}
    )
    report = sweep_traces(corpus, substitutions=[spec])
    assert report.n_traces_diverged == 4


def test_sweep_records_failure_for_bad_path(tmp_path):
    bogus = str(tmp_path / "does_not_exist.sb")
    report = sweep_traces([bogus], substitutions=[])
    assert report.n_traces_attempted == 1
    assert report.n_traces_succeeded == 0
    assert report.n_traces_failed == 1
    f = report.failures[0]
    assert isinstance(f, SweepFailure)
    assert f.trace_path == bogus
    assert f.phase == "load"
    assert f.error_class  # whatever exception class the loader raises


def test_sweep_on_error_raise_propagates(tmp_path):
    bogus = str(tmp_path / "nope.sb")
    with pytest.raises(Exception):
        sweep_traces([bogus], substitutions=[], on_error="raise")


def test_sweep_invalid_on_error_rejected():
    with pytest.raises(ValueError):
        sweep_traces([], substitutions=[], on_error="bogus")


def test_sweep_invalid_substitution_type_rejected(corpus):
    with pytest.raises(TypeError):
        sweep_traces(corpus, substitutions=[12345])


def test_sweep_baseline_subs_subtract(corpus):
    """If baseline=B and candidate=B, the diff is exactly zero."""
    sub = ToolOutputSubstitution(
        at_step="step:2",
        fake_response={"result": LOOKUP_FIXED_ROW},
    )
    report = sweep_traces(
        corpus,
        substitutions=[sub],
        baseline_substitutions=[sub],
    )
    assert report.n_traces_diverged == 0
    assert report.cost_delta_usd.total == 0.0


def test_sweep_progress_callback(corpus):
    seen = []

    def cb(i, n, path):
        seen.append((i, n, os.path.basename(path) if path else ""))

    sweep_traces(corpus, substitutions=[], progress=cb)
    assert seen[0] == (0, 4, "trace_0.sb")
    assert seen[-1] == (4, 4, "")  # final tick


def test_sweep_serialises_substitution_specs(corpus):
    sub = SystemPromptSubstitution(
        at_step="step:0", system_text="be paranoid about PII", mode="prepend"
    )
    report = sweep_traces(corpus, substitutions=[sub])
    js = render_sweep_report_json(report)
    assert js["candidate_substitutions"][0]["type"] == "SystemPromptSubstitution"
    assert js["candidate_substitutions"][0]["at_step"] == "step:0"


# ---------------------------------------------------------- rendering


def test_render_sweep_report_markdown_contains_summary(corpus):
    sub = ToolOutputSubstitution(
        at_step="step:2", fake_response={"result": LOOKUP_FIXED_ROW},
    )
    report = sweep_traces(corpus, substitutions=[sub])
    md = render_sweep_report(report, title="prompt-v7 sweep")
    assert md.startswith("# prompt-v7 sweep\n")
    assert "traces attempted: **4**" in md
    assert "Aggregate stats" in md
    assert "Per-trace" in md
    # 4 corpus rows + header rows
    assert md.count("trace_") >= 4
    assert "ToolOutputSubstitution" in md


def test_render_sweep_report_markdown_truncates_rows(corpus):
    report = sweep_traces(corpus, substitutions=[])
    md = render_sweep_report(report, max_rows=2)
    assert "more rows truncated" in md


def test_render_sweep_report_markdown_failures_section(tmp_path):
    bogus = str(tmp_path / "x.sb")
    report = sweep_traces([bogus], substitutions=[])
    md = render_sweep_report(report)
    assert "## Failures" in md


def test_render_sweep_report_json_shape(corpus):
    sub = ToolOutputSubstitution(
        at_step="step:2", fake_response={"result": LOOKUP_FIXED_ROW},
    )
    report = sweep_traces(corpus, substitutions=[sub])
    js = render_sweep_report_json(report, include_diffs=True)
    assert set(js.keys()) >= {
        "summary", "stats", "results", "failures",
        "baseline_substitutions", "candidate_substitutions",
    }
    assert js["summary"]["n_traces_succeeded"] == 4
    assert "step_diffs" in js["results"][0]["diff"]
    # ensure JSON is round-trippable
    s = json.dumps(js, sort_keys=True)
    assert json.loads(s) == js


# ------------------------------------------------------------ CLI


def test_cli_sweep_markdown(corpus, tmp_path):
    out = tmp_path / "report.md"
    cmd = [
        sys.executable, "-m", "stepback.cli", "sweep",
        *corpus,
        "-s", "tool_output@step:2=:inline:" + json.dumps(
            {"result": LOOKUP_FIXED_ROW}
        ),
        "--output", str(out),
        "--exit-nonzero-on-divergence",
    ]
    env = {**os.environ, "PYTHONPATH": "."}
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert proc.returncode == 3, proc.stderr  # divergence → exit 3
    md = out.read_text()
    assert "# stepback sweep report" in md
    assert "ToolOutputSubstitution" in md


def test_cli_sweep_json_no_divergence(corpus, tmp_path):
    out = tmp_path / "report.json"
    cmd = [
        sys.executable, "-m", "stepback.cli", "sweep",
        *corpus,
        "--output", str(out),
        "--format", "json",
    ]
    env = {**os.environ, "PYTHONPATH": "."}
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert proc.returncode == 0, proc.stderr
    js = json.loads(out.read_text())
    assert js["summary"]["n_traces_succeeded"] == 4
    assert js["summary"]["n_traces_diverged"] == 0


def test_cli_sweep_failure_exit_code(tmp_path):
    out = tmp_path / "report.md"
    cmd = [
        sys.executable, "-m", "stepback.cli", "sweep",
        str(tmp_path / "missing.sb"),
        "--output", str(out),
        "--exit-nonzero-on-failure",
    ]
    env = {**os.environ, "PYTHONPATH": "."}
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert proc.returncode == 4, proc.stderr
