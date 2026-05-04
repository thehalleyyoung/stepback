"""Recorder-overhead micro-benchmark.

Compares two regimes:

* **baseline**: invoke the deterministic fake LLM directly, with no
  recorder in the loop;
* **recorded**: invoke the same fake LLM through
  :py:meth:`stepback.recorder.Recorder.llm_call`, which canonicalises
  inputs/outputs, computes content hashes, and appends an HMAC-chained
  + Ed25519-signed frame to a ``.sb`` file on disk.
"""
from __future__ import annotations

import os
import statistics
import tempfile
import time
from dataclasses import dataclass
from typing import List

from ..recorder import RecorderKey, record
from .replay_caching import _bench_llm


@dataclass
class RecordOverheadResult:
    n_steps: int
    baseline_us_per_step: float
    recorded_us_per_step: float
    overhead_us_per_step: float
    overhead_pct: float
    baseline_p50_us: float
    baseline_p99_us: float
    recorded_p50_us: float
    recorded_p99_us: float
    trace_bytes: int

    def to_json(self) -> dict:
        return {
            "n_steps": self.n_steps,
            "baseline_us_per_step": self.baseline_us_per_step,
            "recorded_us_per_step": self.recorded_us_per_step,
            "overhead_us_per_step": self.overhead_us_per_step,
            "overhead_pct": self.overhead_pct,
            "baseline_p50_us": self.baseline_p50_us,
            "baseline_p99_us": self.baseline_p99_us,
            "recorded_p50_us": self.recorded_p50_us,
            "recorded_p99_us": self.recorded_p99_us,
            "trace_bytes": self.trace_bytes,
        }

    def summary_line(self) -> str:
        return (
            f"record-overhead n_steps={self.n_steps} "
            f"baseline_us={self.baseline_us_per_step:.1f} "
            f"recorded_us={self.recorded_us_per_step:.1f} "
            f"overhead={self.overhead_pct:+.1f}% "
            f"recorded_p99_us={self.recorded_p99_us:.1f} "
            f"trace_bytes={self.trace_bytes}"
        )


def _percentile(samples: List[float], q: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    idx = min(len(s) - 1, max(0, int(q * len(s))))
    return s[idx]


def run(n_steps: int = 1000) -> RecordOverheadResult:
    """Run the recorder-overhead bench and return a :class:`RecordOverheadResult`."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    convo = [
        {"role": "system", "content": "you are a benchmark agent"},
        {"role": "user", "content": "respond deterministically please."},
    ]

    baseline_us: List[float] = []
    for _ in range(n_steps):
        t0 = time.perf_counter_ns()
        _bench_llm("gpt-4o-2024-11-20", convo)
        baseline_us.append((time.perf_counter_ns() - t0) / 1_000.0)

    recorded_us: List[float] = []
    with tempfile.TemporaryDirectory(prefix="stepback-bench-") as d:
        path = os.path.join(d, "overhead.sb")
        with record(path, key=RecorderKey.fresh()) as rec:
            for _ in range(n_steps):
                t0 = time.perf_counter_ns()
                rec.llm_call("gpt-4o-2024-11-20", convo, executor=_bench_llm)
                recorded_us.append((time.perf_counter_ns() - t0) / 1_000.0)
        trace_bytes = os.path.getsize(path)

    baseline_mean = statistics.mean(baseline_us)
    recorded_mean = statistics.mean(recorded_us)
    overhead = recorded_mean - baseline_mean
    overhead_pct = (overhead / baseline_mean * 100.0) if baseline_mean > 0 else 0.0

    return RecordOverheadResult(
        n_steps=n_steps,
        baseline_us_per_step=baseline_mean,
        recorded_us_per_step=recorded_mean,
        overhead_us_per_step=overhead,
        overhead_pct=overhead_pct,
        baseline_p50_us=_percentile(baseline_us, 0.50),
        baseline_p99_us=_percentile(baseline_us, 0.99),
        recorded_p50_us=_percentile(recorded_us, 0.50),
        recorded_p99_us=_percentile(recorded_us, 0.99),
        trace_bytes=trace_bytes,
    )
