"""Tests for stepback.bench — replay-caching and recorder overhead."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from stepback.bench import (
    BenchResult,
    RecordOverheadResult,
    SyntheticTrace,
    compare,
)
from stepback.bench import replay_caching as rc
from stepback.bench import record_overhead as ro
from stepback.cli import main as cli_main


# ------------------------------------------------------------ replay-caching


def test_synthetic_trace_builds():
    st = SyntheticTrace(n_steps=20, seed=7)
    path = st.build()
    try:
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    finally:
        st.cleanup()


def test_run_returns_valid_bench_result():
    result = rc.run(n_steps=20, n_trials=3, strategy="random_step")
    assert isinstance(result, BenchResult)
    assert result.n_trials == 3
    assert result.n_steps >= 20  # synthetic builder may overshoot a bit
    assert len(result.dirty_set_sizes) == 3
    assert all(d >= 1 for d in result.dirty_set_sizes)
    # Median / p95 within range.
    assert min(result.dirty_set_sizes) <= result.median_dirty_set <= max(
        result.dirty_set_sizes
    )
    assert result.wall_time_ms > 0


def test_cost_reduction_factor_greater_than_one():
    """A single substitution must dirty fewer than all N steps on average,
    otherwise the dirty-set algorithm provides no value over naive O(N)."""
    result = rc.run(n_steps=80, n_trials=5, strategy="random_step")
    assert result.cost_reduction_factor > 1.0, (
        f"expected >1x cost reduction, got {result.cost_reduction_factor:.2f}x"
    )


def test_last_quarter_is_more_efficient_than_first_quarter():
    """Substituting late in a trace must dirty a smaller suffix than
    substituting early — sanity-checks that dirty-set propagation
    actually depends on substitution position."""
    last = rc.run(n_steps=80, n_trials=4, strategy="last_quarter", seed=1)
    first = rc.run(n_steps=80, n_trials=4, strategy="first_quarter", seed=1)
    assert last.mean_dirty_set <= first.mean_dirty_set


def test_compare_across_sizes():
    results = compare([10, 20, 40], n_trials=2, strategy="random_step")
    assert set(results.keys()) == {10, 20, 40}
    for n, r in results.items():
        assert isinstance(r, BenchResult)
        assert r.n_trials == 2


def test_bench_result_to_json_roundtrip():
    result = rc.run(n_steps=15, n_trials=2)
    body = result.to_json()
    # Has every documented key.
    for k in (
        "n_steps", "n_substitutions", "n_trials", "strategy",
        "dirty_set_sizes", "median_dirty_set", "p95_dirty_set",
        "mean_dirty_set", "cost_reduction_factor", "wall_time_ms",
    ):
        assert k in body
    # Serialises to JSON cleanly.
    blob = json.dumps(body)
    assert "median_dirty_set" in blob


def test_run_rejects_bad_args():
    with pytest.raises(ValueError):
        rc.run(n_steps=0, n_trials=1)
    with pytest.raises(ValueError):
        rc.run(n_steps=10, n_trials=0)


# ------------------------------------------------------------ record-overhead


def test_record_overhead_run():
    result = ro.run(n_steps=50)
    assert isinstance(result, RecordOverheadResult)
    assert result.n_steps == 50
    assert result.recorded_us_per_step > 0
    assert result.baseline_us_per_step > 0
    # Recorder strictly does more work than the bare LLM call.
    assert result.recorded_us_per_step >= result.baseline_us_per_step
    assert result.trace_bytes > 0
    body = result.to_json()
    assert body["n_steps"] == 50


# ------------------------------------------------------------ CLI


def test_cli_bench_replay_caching_end_to_end(tmp_path, capsys):
    out = tmp_path / "rc.json"
    rc_code = cli_main([
        "bench", "replay-caching",
        "--n-steps", "20",
        "--n-trials", "2",
        "--strategy", "random_step",
        "--out", str(out),
    ])
    assert rc_code == 0
    captured = capsys.readouterr()
    assert "replay-caching" in captured.out
    assert "median_dirty=" in captured.out
    body = json.loads(out.read_text())
    assert body["n_trials"] == 2
    assert body["strategy"] == "random_step"
    assert isinstance(body["dirty_set_sizes"], list)


def test_cli_bench_record_overhead_end_to_end(tmp_path, capsys):
    out = tmp_path / "ro.json"
    rc_code = cli_main([
        "bench", "record-overhead",
        "--n-steps", "30",
        "--out", str(out),
    ])
    assert rc_code == 0
    captured = capsys.readouterr()
    assert "record-overhead" in captured.out
    body = json.loads(out.read_text())
    assert body["n_steps"] == 30
    assert "recorded_us_per_step" in body


def test_cli_bench_replay_caching_no_out(capsys):
    """--out is optional; the CLI must still print the summary line
    and exit cleanly."""
    rc_code = cli_main([
        "bench", "replay-caching",
        "--n-steps", "12",
        "--n-trials", "2",
    ])
    assert rc_code == 0
    out = capsys.readouterr().out
    assert "cost_reduction=" in out
