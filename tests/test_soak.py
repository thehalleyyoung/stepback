"""Tests for :mod:`stepback.bench.soak` — Step 34.

We deliberately run a small fleet (``n_traces=20``) here so unit
tests stay fast (<2s); the scheduled workflow at
``.github/workflows/soak.yml`` runs the real 10,000-trace fleet.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from stepback.bench import SoakResult, run_soak
from stepback.bench.soak import _percentile, _summary, run


def test_summary_handles_empty_and_single():
    s = _summary([])
    assert s["min"] == 0.0
    assert s["mean"] == 0.0
    s1 = _summary([5.0])
    assert s1["min"] == s1["max"] == s1["median"] == s1["mean"] == 5.0
    assert s1["stdev"] == 0.0


def test_percentile_monotone():
    xs = [float(i) for i in range(100)]
    assert _percentile(xs, 0.0) == 0.0
    assert _percentile(xs, 0.5) <= _percentile(xs, 0.95)
    assert _percentile(xs, 0.99) <= _percentile(xs, 1.0) == 99.0


def test_summary_percentiles_sane():
    s = _summary([float(i) for i in range(1, 101)])
    assert s["min"] == 1.0
    assert s["max"] == 100.0
    assert 49.0 <= s["median"] <= 51.0
    assert 94.0 <= s["p95"] <= 96.0
    assert 98.0 <= s["p99"] <= 100.0
    assert s["mean"] == pytest.approx(50.5, rel=1e-9)


def test_run_soak_small_with_substitution():
    res = run_soak(n_traces=20, n_steps=8, seed=7)
    assert isinstance(res, SoakResult)
    assert res.n_traces == 20
    assert res.errors == 0
    assert res.do_substitution is True
    # all aggregates populated
    assert res.total_ms["max"] >= res.total_ms["median"] >= res.total_ms["min"] >= 0.0
    assert res.dirty_count["min"] >= 1.0  # substitution always dirties at least the target
    assert res.real_executions["max"] >= 1.0
    # actual step count is close to target (parallel branches make it inexact)
    assert res.n_steps_actual["min"] >= 1.0
    # rolling determinism digest is a 64-char hex sha256
    assert len(res.digest) == 64 and all(c in "0123456789abcdef" for c in res.digest)
    # JSON round-trip
    blob = json.dumps(res.to_json())
    j = json.loads(blob)
    assert j["n_traces"] == 20
    assert j["digest"] == res.digest
    assert j["traces_per_second"] > 0


def test_run_soak_without_substitution_has_zero_dirty():
    res = run_soak(n_traces=10, n_steps=6, seed=1, do_substitution=False)
    assert res.errors == 0
    assert res.dirty_count["max"] == 0.0
    # No substitution = every step is a cache hit, no real executions.
    assert res.real_executions["max"] == 0.0


def test_run_soak_deterministic_under_same_seed():
    a = run_soak(n_traces=10, n_steps=8, seed=42)
    b = run_soak(n_traces=10, n_steps=8, seed=42)
    assert a.digest == b.digest
    assert a.n_steps_actual["mean"] == b.n_steps_actual["mean"]
    assert a.dirty_count["mean"] == b.dirty_count["mean"]


def test_run_soak_different_seeds_diverge():
    a = run_soak(n_traces=8, n_steps=8, seed=1)
    b = run_soak(n_traces=8, n_steps=8, seed=2)
    assert a.digest != b.digest


def test_run_soak_summary_line_has_required_fields():
    res = run_soak(n_traces=4, n_steps=4, seed=0)
    line = res.summary_line()
    for tok in ("soak", "n_traces=4", "errors=0", "median_total_ms=",
                "p99_total_ms=", "median_dirty=", "p99_dirty=",
                "tps=", "digest="):
        assert tok in line


def test_run_soak_validates_args():
    with pytest.raises(ValueError):
        run_soak(n_traces=0)
    with pytest.raises(ValueError):
        run_soak(n_traces=1, n_steps=0)


def test_run_soak_track_memory_records_peak():
    res = run_soak(n_traces=5, n_steps=4, seed=0, track_memory=True)
    assert res.peak_memory_bytes > 0


def test_aggregator_records_failures_without_aborting(monkeypatch):
    # Patch _one_trace to raise on every other call so we exercise
    # the error-counting branch without requiring an actual recorder
    # crash.
    import stepback.bench.soak as soak_mod

    real = soak_mod._one_trace
    counter = {"n": 0}

    def flaky(*a, **kw):
        counter["n"] += 1
        if counter["n"] % 2 == 0:
            raise RuntimeError("synthetic-flake")
        return real(*a, **kw)

    monkeypatch.setattr(soak_mod, "_one_trace", flaky)
    res = run(n_traces=10, n_steps=4, seed=0)
    assert res.errors == 5
    assert res.error_kinds == {"RuntimeError": 5}
    # Successful 5 still produced metrics.
    assert res.total_ms["mean"] > 0


def test_cli_bench_soak_smoke(tmp_path):
    out = tmp_path / "soak.json"
    proc = subprocess.run(
        [sys.executable, "-m", "stepback.cli", "bench", "soak",
         "--n-traces", "8", "--n-steps", "6", "--seed", "3",
         "--out", str(out)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "soak n_traces=8" in proc.stdout
    blob = json.loads(out.read_text())
    assert blob["n_traces"] == 8
    assert blob["errors"] == 0
    assert "digest" in blob and len(blob["digest"]) == 64


def test_cli_bench_soak_no_substitution(tmp_path):
    out = tmp_path / "soak.json"
    proc = subprocess.run(
        [sys.executable, "-m", "stepback.cli", "bench", "soak",
         "--n-traces", "4", "--n-steps", "4", "--seed", "0",
         "--no-substitution", "--out", str(out)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    blob = json.loads(out.read_text())
    assert blob["do_substitution"] is False
    assert blob["dirty_count"]["max"] == 0.0


def test_scripts_bench_soak_smoke(tmp_path):
    """The legacy ``scripts/bench_soak.py`` wrapper still works."""
    import os
    out = tmp_path / "soak.json"
    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "scripts/bench_soak.py",
         "--n-traces", "4", "--n-steps", "4", "--seed", "0",
         "--out", str(out)],
        capture_output=True, text=True, timeout=60, env=env,
        cwd=repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    assert "soak n_traces=4" in proc.stdout
    blob = json.loads(out.read_text())
    assert blob["n_traces"] == 4
