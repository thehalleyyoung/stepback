"""Benchmark per-LLM-call recorder overhead.

Records 1000 llm_call steps with the deterministic fake LLM and
reports per-step wall-clock latency percentiles, in milliseconds.
This grounds the README "Record overhead per LLM call < 5 ms p99"
target on a single laptop.
"""
from __future__ import annotations

import os
import statistics
import tempfile
import time

from stepback import record
from stepback.recorder import RecorderKey
from tests.fixtures.agent import fake_llm

N = 1000


def main() -> None:
    convo = [
        {"role": "system", "content": "you are a payments agent"},
        {"role": "user", "content": "Pay invoice INV-118 to Acme Bolts $50000."},
    ]
    samples_ms: list[float] = []
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bench.sb")
        with record(path, key=RecorderKey.fresh()) as rec:
            for _ in range(N):
                t0 = time.perf_counter_ns()
                rec.llm_call("gpt-4o-2024-11-20", convo, executor=fake_llm)
                samples_ms.append((time.perf_counter_ns() - t0) / 1_000_000.0)
        size = os.path.getsize(path)

    samples_ms.sort()
    p50 = samples_ms[len(samples_ms) // 2]
    p99 = samples_ms[int(len(samples_ms) * 0.99)]
    p999 = samples_ms[min(len(samples_ms) - 1, int(len(samples_ms) * 0.999))]
    mean = statistics.mean(samples_ms)
    print(
        f"n={N} mean_ms={mean:.3f} p50_ms={p50:.3f} "
        f"p99_ms={p99:.3f} p999_ms={p999:.3f} "
        f"trace_bytes={size}"
    )


if __name__ == "__main__":
    main()
