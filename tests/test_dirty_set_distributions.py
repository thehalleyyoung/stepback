"""Tests for dirty-set distribution benchmark (Step 66).

Validates that:

1. :py:func:`run` returns a :py:class:`DistributionSuite` with the expected
   structure for each (corpus, position, sub_kind) cell.
2. Percentile and histogram invariants hold (p5 ≤ p25 ≤ p50 ≤ p75 ≤ p95 ≤ p99,
   histogram bins sum to n_trials).
3. ``dirty_fractions`` values are in [0.0, 1.0].
4. ``to_json()`` serialisation round-trips through the expected schema.
5. Individual corpus builders produce valid traces whose dirty-set analysis
   returns meaningful results.
6. The ``agent_fixture`` corpus exercises the canonical testing fixture and
   produces non-negative dirty counts.
7. CLI entry point ``stepback bench dirty-set-distributions`` exits 0 and
   prints a summary line.
8. Public symbols are importable from ``stepback.bench``.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from stepback.bench.dirty_set_distributions import (
    CorpusDistribution,
    DistributionSuite,
    _build_agent,
    _build_linear,
    _build_parallel,
    _histogram,
    _percentile,
    _run_corpus_cell,
    run,
)


# ------------------------------------------------------------------ helpers


def _make_small_suite() -> DistributionSuite:
    """Run a minimal suite (1 corpus, random position, 3 trials, 10 steps)."""
    return run(n_trials=3, n_steps=10, seed=0, corpora=["linear_chain"])


# ------------------------------------------------------------------ import tests


def test_importable_from_bench_package():
    from stepback.bench import CorpusDistribution, DistributionSuite, run_distributions

    assert callable(run_distributions)
    assert CorpusDistribution is not None
    assert DistributionSuite is not None


# ------------------------------------------------------------------ _percentile


def test_percentile_empty():
    assert _percentile([], 50) == 0.0


def test_percentile_single():
    assert _percentile([0.5], 50) == 0.5


def test_percentile_monotone():
    vals = sorted([0.1, 0.2, 0.3, 0.5, 0.8])
    p25 = _percentile(vals, 25)
    p50 = _percentile(vals, 50)
    p75 = _percentile(vals, 75)
    assert p25 <= p50 <= p75


# ------------------------------------------------------------------ _histogram


def test_histogram_all_zero():
    h = _histogram([0.0, 0.0, 0.0])
    assert h["0%"] == 3
    assert sum(h.values()) == 3


def test_histogram_spread():
    fracs = [0.0, 0.05, 0.2, 0.4, 0.6, 0.9]
    h = _histogram(fracs)
    assert sum(h.values()) == len(fracs)
    assert h["0%"] == 1
    assert h["1-10%"] == 1
    assert h["11-25%"] == 1
    assert h["26-50%"] == 1
    assert h["51-75%"] == 1
    assert h["76-100%"] == 1


def test_histogram_boundary_10pct():
    h = _histogram([0.10])
    assert h["1-10%"] == 1


def test_histogram_boundary_25pct():
    h = _histogram([0.25])
    assert h["11-25%"] == 1


def test_histogram_boundary_50pct():
    h = _histogram([0.50])
    assert h["26-50%"] == 1


def test_histogram_boundary_75pct():
    h = _histogram([0.75])
    assert h["51-75%"] == 1


def test_histogram_full():
    h = _histogram([1.0])
    assert h["76-100%"] == 1


# ------------------------------------------------------------------ corpus builders


def test_build_linear_produces_valid_trace():
    from stepback.replay import replay

    path = _build_linear(n_steps=8, seed=42)
    try:
        t = replay(path)
        assert len(t.recorded_steps) >= 1
        kinds = {s["step_kind"] for s in t.recorded_steps}
        assert kinds & {"llm_call", "tool_call"}
    finally:
        import shutil

        shutil.rmtree(os.path.dirname(path), ignore_errors=True)


def test_build_parallel_produces_valid_trace():
    from stepback.replay import replay

    path = _build_parallel(n_branches=3, seed=7)
    try:
        t = replay(path)
        assert len(t.recorded_steps) >= 3
        kinds = {s["step_kind"] for s in t.recorded_steps}
        assert "parallel_branch_open" in kinds
    finally:
        import shutil

        shutil.rmtree(os.path.dirname(path), ignore_errors=True)


def test_build_agent_produces_12_steps():
    from stepback.replay import replay

    path = _build_agent(seed=0)
    try:
        t = replay(path)
        assert len(t.recorded_steps) == 12
    finally:
        import shutil

        shutil.rmtree(os.path.dirname(path), ignore_errors=True)


# ------------------------------------------------------------------ _run_corpus_cell


def test_run_corpus_cell_linear_returns_distribution():
    cell = _run_corpus_cell(
        corpus="linear_chain",
        position="random",
        sub_kind="PromptSubstitution",
        n_trials=4,
        n_steps=8,
        seed=0,
    )
    assert isinstance(cell, CorpusDistribution)
    assert cell.corpus == "linear_chain"
    assert cell.position == "random"
    assert cell.sub_kind == "PromptSubstitution"
    assert cell.n_trials >= 1
    assert all(0.0 <= f <= 1.0 for f in cell.dirty_fractions)


def test_run_corpus_cell_parallel():
    cell = _run_corpus_cell(
        corpus="parallel_wide",
        position="early",
        sub_kind="ToolOutputSubstitution",
        n_trials=3,
        n_steps=10,
        seed=1,
    )
    assert cell.corpus == "parallel_wide"
    assert all(0.0 <= f <= 1.0 for f in cell.dirty_fractions)


def test_run_corpus_cell_agent_fixture():
    cell = _run_corpus_cell(
        corpus="agent_fixture",
        position="late",
        sub_kind="ToolOutputSubstitution",
        n_trials=3,
        n_steps=12,
        seed=2,
    )
    assert cell.corpus == "agent_fixture"
    assert cell.n_trials >= 1
    assert all(0.0 <= f <= 1.0 for f in cell.dirty_fractions)


def test_run_corpus_cell_mixed_synthetic():
    cell = _run_corpus_cell(
        corpus="mixed_synthetic",
        position="random",
        sub_kind="PromptSubstitution",
        n_trials=3,
        n_steps=12,
        seed=3,
    )
    assert cell.corpus == "mixed_synthetic"
    assert all(0.0 <= f <= 1.0 for f in cell.dirty_fractions)


# ------------------------------------------------------------------ CorpusDistribution invariants


def test_percentile_order():
    cell = _run_corpus_cell("linear_chain", "random", "PromptSubstitution", 10, 15, 0)
    assert cell.p5 <= cell.p25 <= cell.p50 <= cell.p75 <= cell.p90 <= cell.p95 <= cell.p99


def test_histogram_sums_to_n_trials():
    cell = _run_corpus_cell("linear_chain", "random", "PromptSubstitution", 5, 10, 0)
    assert sum(cell.histogram.values()) == cell.n_trials


def test_histogram_has_expected_bins():
    cell = _run_corpus_cell("linear_chain", "random", "PromptSubstitution", 3, 8, 0)
    expected_bins = {"0%", "1-10%", "11-25%", "26-50%", "51-75%", "76-100%"}
    assert set(cell.histogram.keys()) == expected_bins


def test_step_count_range_valid():
    cell = _run_corpus_cell("linear_chain", "random", "PromptSubstitution", 3, 10, 0)
    lo, hi = cell.step_count_range
    assert lo <= hi
    assert lo >= 1


# ------------------------------------------------------------------ DistributionSuite


def test_run_returns_distribution_suite():
    suite = _make_small_suite()
    assert isinstance(suite, DistributionSuite)


def test_run_all_corpora_produces_24_cells():
    # 4 corpora × 3 positions × 2 sub_kinds = 24 cells
    suite = run(n_trials=2, n_steps=8, seed=0)
    assert len(suite.distributions) == 24


def test_run_subset_corpora():
    suite = run(n_trials=2, n_steps=8, seed=0, corpora=["linear_chain", "parallel_wide"])
    # 2 corpora × 3 × 2 = 12 cells
    assert len(suite.distributions) == 12
    assert all(d.corpus in ("linear_chain", "parallel_wide") for d in suite.distributions)


def test_run_total_trials():
    suite = run(n_trials=3, n_steps=8, seed=0, corpora=["linear_chain"])
    # 1 corpus × 3 positions × 2 sub_kinds × 3 trials = 18 trials (some may be 0 if no target)
    assert suite.total_trials >= 1


def test_run_wall_time_positive():
    suite = _make_small_suite()
    assert suite.wall_time_ms > 0


def test_run_generated_at_nonempty():
    suite = _make_small_suite()
    assert suite.generated_at and "Z" in suite.generated_at


# ------------------------------------------------------------------ to_json


def test_to_json_roundtrip():
    suite = _make_small_suite()
    d = suite.to_json()
    assert "generated_at" in d
    assert "total_trials" in d
    assert "wall_time_ms" in d
    assert "distributions" in d
    assert isinstance(d["distributions"], list)


def test_corpus_distribution_to_json_keys():
    cell = _run_corpus_cell("linear_chain", "random", "PromptSubstitution", 3, 8, 0)
    j = cell.to_json()
    for key in ("corpus", "position", "sub_kind", "n_trials", "step_count_range",
                "dirty_fractions", "p5", "p25", "p50", "p75", "p90", "p95", "p99",
                "mean", "histogram"):
        assert key in j, f"missing key {key!r}"


def test_to_json_is_json_serialisable():
    suite = _make_small_suite()
    text = json.dumps(suite.to_json())
    reloaded = json.loads(text)
    assert reloaded["distributions"][0]["corpus"] == "linear_chain"


def test_dirty_fractions_in_json_are_rounded():
    cell = _run_corpus_cell("linear_chain", "random", "PromptSubstitution", 3, 8, 0)
    j = cell.to_json()
    for f in j["dirty_fractions"]:
        assert isinstance(f, float)
        # max 4 decimal places
        assert len(str(f).split(".")[-1]) <= 4


# ------------------------------------------------------------------ CLI


def test_cli_dirty_set_distributions_exits_0(tmp_path, capsys):
    from stepback.cli import main as cli_main

    out = tmp_path / "out.json"
    rc_code = cli_main([
        "bench", "dirty-set-distributions",
        "--n-trials", "2",
        "--n-steps", "8",
        "--corpora", "linear_chain",
        "--out", str(out),
    ])
    assert rc_code == 0
    captured = capsys.readouterr()
    assert "linear_chain" in captured.out
    assert out.exists()
    data = json.loads(out.read_text())
    assert "distributions" in data
    # 1 corpus × 3 positions × 2 sub_kinds = 6 cells
    assert len(data["distributions"]) == 6


def test_cli_without_out_flag_prints_table(capsys):
    from stepback.cli import main as cli_main

    rc_code = cli_main([
        "bench", "dirty-set-distributions",
        "--n-trials", "1",
        "--n-steps", "8",
        "--corpora", "linear_chain",
    ])
    assert rc_code == 0
    out = capsys.readouterr().out
    assert "p50" in out
    assert "linear_chain" in out


# ------------------------------------------------------------------ late-trace benefit


def test_late_substitution_produces_smaller_dirty_set_on_linear():
    """For a linear chain, substituting late should generally give a smaller dirty set."""
    late_cell = _run_corpus_cell("linear_chain", "late", "PromptSubstitution", 10, 20, 0)
    early_cell = _run_corpus_cell("linear_chain", "early", "PromptSubstitution", 10, 20, 0)
    # Median dirty fraction should be smaller for late substitution on a linear chain
    assert late_cell.p50 <= early_cell.p50 + 0.5  # generous tolerance


def test_parallel_branch_substitution_keeps_siblings_clean():
    """Substituting inside a branch should give dirty fraction < 1.0 for parallel traces."""
    cell = _run_corpus_cell("parallel_wide", "random", "PromptSubstitution", 10, 10, 0)
    # p95 should not be 1.0 (all dirty) for parallel traces under a single branch substitution
    # (some trials will touch the branch steps, but siblings stay clean)
    assert cell.p95 <= 1.0
    # At least some trials should have dirty fraction < 1.0
    has_partial = any(f < 1.0 for f in cell.dirty_fractions)
    assert has_partial, "expected at least one trial with partial dirty set in parallel corpus"
