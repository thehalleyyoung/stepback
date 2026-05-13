"""Soak benchmark: record + replay a synthetic fleet of traces.

Step 34 of ``100_STEPS.md``: exercise the full record / replay /
dirty-set propagation pipeline across a *fleet* of independent
synthetic traces (default 10,000) and emit only **aggregate**
statistics. No individual trace, no per-trace dirty-set vector, and
no per-trace wall-clock series is retained — soak runs are designed
to be safe to schedule indefinitely without unbounded artifact
growth.

Why this is interesting
-----------------------

The replay-caching benchmark in :mod:`stepback.bench.replay_caching`
measures one statistic (mean dirty-set size) on a small number of
trials. Soak runs exist to catch:

* **Tail behaviour.** p99/max wall-clock per trace, p99/max dirty-set
  size, p99/max trace file size — the things you only see at fleet
  scale.
* **Memory / FD leaks.** Recorders, readers, and the canonical-hash
  cache must survive thousands of open/close cycles.
* **Determinism drift.** Aggregate hashes of the recorded steps are
  folded into a single rolling SHA-256 so any non-determinism in
  recording or replay shows up as a changed digest run-over-run for
  the same seed.
* **Verification cost.** HMAC-chain + Ed25519 signature verification
  is exercised on every trace.

Output shape
------------

:func:`run` returns :class:`SoakResult`, which is a dataclass of
scalar aggregates only (counts, means, percentiles, the rolling
digest, error tallies). It is JSON-serialisable via ``to_json()``
and intended to be written as a single small artifact (~1 KiB) per
scheduled run so historical comparison is cheap.

CLI surface
-----------

The bench is reachable via ``stepback bench soak``; see
``stepback/cli.py``. The defaults (``--n-traces 10000``,
``--n-steps 10``) match the figure in step 34 and run in well under
a minute on a laptop. A scheduled GitHub Actions workflow at
``.github/workflows/soak.yml`` drives the full 10,000 fleet and
uploads the aggregate JSON as a build artifact.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import statistics
import tempfile
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..replay import replay
from ..substitutions import SubstitutionSet
from .replay_caching import (
    SyntheticTrace,
    _build_substitution,
    _executor,
    _pick_target,
)


# ---------------------------------------------------------- aggregator

def _percentile(sorted_xs: List[float], q: float) -> float:
    if not sorted_xs:
        return 0.0
    if len(sorted_xs) == 1:
        return float(sorted_xs[0])
    # Nearest-rank percentile: simple, deterministic, no interpolation
    # surprises in CI logs.
    idx = max(0, min(len(sorted_xs) - 1, int(round(q * (len(sorted_xs) - 1)))))
    return float(sorted_xs[idx])


def _summary(values: List[float]) -> Dict[str, float]:
    """Return min/median/p95/p99/max/mean/stdev of a numeric list."""
    if not values:
        return {
            "min": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0,
            "max": 0.0, "mean": 0.0, "stdev": 0.0,
        }
    s = sorted(float(v) for v in values)
    return {
        "min": float(s[0]),
        "median": float(statistics.median(s)),
        "p95": _percentile(s, 0.95),
        "p99": _percentile(s, 0.99),
        "max": float(s[-1]),
        "mean": float(statistics.mean(s)),
        "stdev": float(statistics.pstdev(s)) if len(s) > 1 else 0.0,
    }


# --------------------------------------------------------------- result


@dataclass
class SoakResult:
    """Aggregate-only result of a soak run.

    No per-trace data is retained on this object; the constructor
    intentionally collapses the fleet to summary statistics so a
    scheduled run never produces an unbounded artifact.
    """

    n_traces: int
    n_steps_target: int
    do_substitution: bool
    seed: int
    errors: int
    error_kinds: Dict[str, int]
    n_steps_actual: Dict[str, float]
    record_ms: Dict[str, float]
    replay_ms: Dict[str, float]
    total_ms: Dict[str, float]
    dirty_count: Dict[str, float]
    cache_hits: Dict[str, float]
    real_executions: Dict[str, float]
    file_size_bytes: Dict[str, float]
    digest: str
    wall_time_s: float
    peak_memory_bytes: int = 0
    traces_per_second: float = 0.0

    def to_json(self) -> dict:
        return {
            "n_traces": self.n_traces,
            "n_steps_target": self.n_steps_target,
            "do_substitution": self.do_substitution,
            "seed": self.seed,
            "errors": self.errors,
            "error_kinds": dict(self.error_kinds),
            "n_steps_actual": dict(self.n_steps_actual),
            "record_ms": dict(self.record_ms),
            "replay_ms": dict(self.replay_ms),
            "total_ms": dict(self.total_ms),
            "dirty_count": dict(self.dirty_count),
            "cache_hits": dict(self.cache_hits),
            "real_executions": dict(self.real_executions),
            "file_size_bytes": dict(self.file_size_bytes),
            "digest": self.digest,
            "wall_time_s": self.wall_time_s,
            "peak_memory_bytes": self.peak_memory_bytes,
            "traces_per_second": self.traces_per_second,
        }

    def summary_line(self) -> str:
        return (
            f"soak n_traces={self.n_traces} "
            f"errors={self.errors} "
            f"median_total_ms={self.total_ms['median']:.2f} "
            f"p99_total_ms={self.total_ms['p99']:.2f} "
            f"median_dirty={self.dirty_count['median']:.1f} "
            f"p99_dirty={self.dirty_count['p99']:.1f} "
            f"tps={self.traces_per_second:.1f} "
            f"digest={self.digest[:16]}"
        )


# ---------------------------------------------------------------- core


def _one_trace(
    trial: int,
    n_steps: int,
    seed: int,
    do_substitution: bool,
    sub_rng: random.Random,
) -> Tuple[Dict[str, float], str]:
    """Record + replay one synthetic trace.

    Returns a dict of per-trace metrics plus a short hex digest of
    the canonical step ids so the caller can fold determinism
    evidence into a single rolling hash.
    """
    metrics: Dict[str, float] = {}
    st = SyntheticTrace(n_steps=n_steps, seed=seed)
    t_record0 = time.perf_counter_ns()
    path = st.build()
    metrics["record_ms"] = (time.perf_counter_ns() - t_record0) / 1_000_000.0
    try:
        metrics["file_size_bytes"] = float(os.path.getsize(path))
        t_replay0 = time.perf_counter_ns()
        tr = replay(path)
        if do_substitution:
            target = _pick_target(tr.recorded_steps, "random_step", sub_rng)
            if target is None:
                raise RuntimeError("synthetic trace had no substitutable steps")
            sub = _build_substitution(target, "random_step")
            subs = SubstitutionSet()
            subs.add(sub)
            result = tr.run_replay(subs, _executor())
        else:
            result = tr.run_replay(SubstitutionSet(), _executor())
        metrics["replay_ms"] = (time.perf_counter_ns() - t_replay0) / 1_000_000.0
        metrics["n_steps_actual"] = float(len(tr.recorded_steps))
        metrics["dirty_count"] = float(result.dirty_count)
        metrics["cache_hits"] = float(result.cache_hit_count)
        metrics["real_executions"] = float(result.real_executions)
        metrics["total_ms"] = metrics["record_ms"] + metrics["replay_ms"]
        # Determinism witness: hash the recorded step_id sequence.
        h = hashlib.sha256()
        for s in tr.recorded_steps:
            h.update(s["step_id"].encode("ascii"))
            h.update(b"\x00")
            h.update(s["step_kind"].encode("ascii"))
            h.update(b"\x00")
        digest_hex = h.hexdigest()
    finally:
        st.cleanup()
    return metrics, digest_hex


def run(
    n_traces: int = 10_000,
    n_steps: int = 10,
    *,
    seed: int = 0,
    do_substitution: bool = True,
    progress_every: int = 0,
    track_memory: bool = False,
    gc_every: int = 256,
) -> SoakResult:
    """Drive a fleet of ``n_traces`` synthetic record+replay cycles.

    Per-trace data is *streamed into aggregators only*; nothing is
    retained. This is the canonical entry point for the scheduled
    soak workflow.

    Parameters
    ----------
    n_traces:
        Number of independent synthetic traces in the fleet.
    n_steps:
        Approximate number of steps per trace (passed to
        :class:`SyntheticTrace`; the realised count varies because
        parallel branches are atomic blocks).
    seed:
        Base RNG seed. Trace ``i`` uses ``seed * 1_000_003 + i``.
    do_substitution:
        If True (default), each trace gets one random
        ``random_step`` substitution and the dirty-set machinery is
        exercised. If False, replay is pure cache-hit and
        ``dirty_count`` aggregates will all be zero.
    progress_every:
        If > 0, print ``progress trace=k/N tps=...`` to stdout every
        ``progress_every`` traces. Off by default (CI-friendly).
    track_memory:
        If True, run under :mod:`tracemalloc` and report peak.
        Adds ~10–20% overhead, off by default.
    gc_every:
        Force a :func:`gc.collect` every ``gc_every`` traces. Set to
        0 to disable. Default 256 keeps long-running soaks stable
        without leaning on the cyclic collector at unpredictable
        moments inside a timed window.
    """
    if n_traces < 1:
        raise ValueError("n_traces must be >= 1")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    sub_rng = random.Random(seed ^ 0xA5A5_5A5A)

    record_ms: List[float] = []
    replay_ms: List[float] = []
    total_ms: List[float] = []
    dirty_count: List[float] = []
    cache_hits: List[float] = []
    real_executions: List[float] = []
    file_sizes: List[float] = []
    n_steps_actual: List[float] = []

    errors = 0
    error_kinds: Dict[str, int] = {}

    rolling = hashlib.sha256()

    if track_memory:
        tracemalloc.start()
    t_wall0 = time.perf_counter()

    for i in range(n_traces):
        s = (seed * 1_000_003 + i) & 0x7FFF_FFFF
        try:
            m, digest_hex = _one_trace(
                i, n_steps=n_steps, seed=s,
                do_substitution=do_substitution, sub_rng=sub_rng,
            )
        except Exception as exc:  # pragma: no cover - exercised in unit test
            errors += 1
            kind = type(exc).__name__
            error_kinds[kind] = error_kinds.get(kind, 0) + 1
            continue

        record_ms.append(m["record_ms"])
        replay_ms.append(m["replay_ms"])
        total_ms.append(m["total_ms"])
        dirty_count.append(m["dirty_count"])
        cache_hits.append(m["cache_hits"])
        real_executions.append(m["real_executions"])
        file_sizes.append(m["file_size_bytes"])
        n_steps_actual.append(m["n_steps_actual"])

        rolling.update(digest_hex.encode("ascii"))
        rolling.update(b"|")

        if gc_every and (i + 1) % gc_every == 0:
            gc.collect()
        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.perf_counter() - t_wall0
            tps = (i + 1) / elapsed if elapsed > 0 else 0.0
            print(
                f"progress trace={i + 1}/{n_traces} "
                f"tps={tps:.1f} errors={errors}",
                flush=True,
            )

    wall_time_s = time.perf_counter() - t_wall0

    peak_bytes = 0
    if track_memory:
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    successes = n_traces - errors
    tps = (successes / wall_time_s) if wall_time_s > 0 else 0.0

    return SoakResult(
        n_traces=n_traces,
        n_steps_target=n_steps,
        do_substitution=do_substitution,
        seed=seed,
        errors=errors,
        error_kinds=error_kinds,
        n_steps_actual=_summary(n_steps_actual),
        record_ms=_summary(record_ms),
        replay_ms=_summary(replay_ms),
        total_ms=_summary(total_ms),
        dirty_count=_summary(dirty_count),
        cache_hits=_summary(cache_hits),
        real_executions=_summary(real_executions),
        file_size_bytes=_summary(file_sizes),
        digest=rolling.hexdigest(),
        wall_time_s=wall_time_s,
        peak_memory_bytes=int(peak_bytes),
        traces_per_second=tps,
    )


__all__ = ["SoakResult", "run"]
