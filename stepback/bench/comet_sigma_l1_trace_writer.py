"""L1 emission-latency micro-benchmark (Step 6 of ``COMET_SIGMA_1000.md``).

Measures the per-frame latency of the Comet-Σ **L1** ``temporal_basis``
feature emitter wired in :mod:`stepback.comet_sigma.l1_trace_writer` for
``stepback.trace_writer`` + ``.sb`` format v1 frames.

Three regimes are timed against an identical synthetic frame stream:

* **off** — the ``COMET_SIGMA_L1_TEMPORAL`` flag is OFF; every call to
  :func:`stepback.comet_sigma.l1_trace_writer.observe_frame` is the
  single ``is_active()`` early-return. This is the live cost stepback
  pays today on the writer's hot path.
* **on** — the flag is ON, so every base feature is extracted, a
  :class:`~comet_sigma.audit.Receipt` is emitted per feature, the
  per-writer ring buffer is updated and registered observe hooks
  (e.g. the temporal-basis projector) are invoked. This is what users
  pay when they opt-in to L1.
* **on_temporal** — the flag is ON *and* the Step-2 temporal-basis
  projector hook is registered, so each observation also drives the
  ``1s/10s/1m/10m`` window aggregates. This is what users pay when
  the full L1 ``temporal_basis`` projection is enabled.

For every regime we report ``mean / p50 / p95 / p99`` per-frame latency
in microseconds plus the underlying sample count, and we record the
schema version + comet_sigma availability flag in the JSON output so
results are easy to diff across releases.

The benchmark is fully self-contained — it does **not** open a real
``.sb`` file on disk, it constructs frame bodies in-memory and feeds
them straight through ``observe_frame``. That keeps the measurement
focused on L1 *emission* cost (the thing Step 6 asks about) without
contaminating it with file-I/O jitter from
:meth:`stepback.trace_writer.TraceWriter._write_frame`.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Importing the comet_sigma integration is always safe — when the
# upstream package is missing the helpers degrade to no-ops (see
# ``stepback.comet_sigma.__init__``).
from .. import comet_sigma as _cs
from ..comet_sigma import l1_trace_writer as _l1
from ..comet_sigma import l1_trace_writer_temporal as _l1t
from ..canonical import canonical_json


#: Stable schema version for the JSON record we emit. Bump when fields
#: change in a breaking way.
SCHEMA_VERSION: str = "comet_sigma_l1_trace_writer_latency.v1"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class _RegimeStats:
    """Latency statistics for a single timing regime."""

    n_samples: int
    mean_us: float
    p50_us: float
    p95_us: float
    p99_us: float
    min_us: float
    max_us: float

    def to_json(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "mean_us": self.mean_us,
            "p50_us": self.p50_us,
            "p95_us": self.p95_us,
            "p99_us": self.p99_us,
            "min_us": self.min_us,
            "max_us": self.max_us,
        }


@dataclass
class L1TraceWriterLatencyResult:
    """Top-level result of :func:`run`.

    Attributes
    ----------
    n_frames:
        Number of synthetic frames timed per regime.
    warmup_frames:
        Number of warmup frames timed but discarded before the
        measured loop. JIT / CPU-cache warmup; defaults to ``200``.
    comet_sigma_available:
        Whether the upstream ``comet_sigma`` package was importable
        when the bench ran. When ``False`` the on/on_temporal regimes
        will look identical to off (the L1 emitter degrades to no-op).
    schema_version:
        Stable identifier for the JSON shape of this record.
    off / on / on_temporal:
        Per-regime latency stats (see :class:`_RegimeStats`).
    """

    n_frames: int
    warmup_frames: int
    comet_sigma_available: bool
    schema_version: str
    off: _RegimeStats
    on: _RegimeStats
    on_temporal: _RegimeStats
    overhead_on_us: float = field(default=0.0)
    overhead_on_temporal_us: float = field(default=0.0)

    def to_json(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "n_frames": self.n_frames,
            "warmup_frames": self.warmup_frames,
            "comet_sigma_available": self.comet_sigma_available,
            "regimes": {
                "off": self.off.to_json(),
                "on": self.on.to_json(),
                "on_temporal": self.on_temporal.to_json(),
            },
            "overhead_us_per_frame": {
                "on_minus_off": self.overhead_on_us,
                "on_temporal_minus_off": self.overhead_on_temporal_us,
            },
        }

    def summary_line(self) -> str:
        return (
            f"l1-trace-writer-latency n_frames={self.n_frames} "
            f"off_p50={self.off.p50_us:.2f}us "
            f"on_p50={self.on.p50_us:.2f}us "
            f"on_p95={self.on.p95_us:.2f}us "
            f"on_p99={self.on.p99_us:.2f}us "
            f"on_temporal_p99={self.on_temporal.p99_us:.2f}us "
            f"overhead_on={self.overhead_on_us:+.2f}us"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(samples: List[float], q: float) -> float:
    """Nearest-rank percentile of ``samples`` (q in [0,1])."""
    if not samples:
        return 0.0
    s = sorted(samples)
    # nearest-rank index, clamped to the valid range
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def _stats(samples: List[float]) -> _RegimeStats:
    if not samples:
        return _RegimeStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return _RegimeStats(
        n_samples=len(samples),
        mean_us=statistics.fmean(samples),
        p50_us=_percentile(samples, 0.50),
        p95_us=_percentile(samples, 0.95),
        p99_us=_percentile(samples, 0.99),
        min_us=min(samples),
        max_us=max(samples),
    )


def _make_frame_bodies(n: int) -> List[Dict[str, Any]]:
    """Build a deterministic mix of header / step / blob frames.

    The mix loosely matches what a real ``TraceWriter`` produces — one
    header, then a long run of ``step`` frames with the occasional
    ``blob`` and ``tail``. This is enough to exercise every base
    extractor at least once.
    """
    bodies: List[Dict[str, Any]] = []
    bodies.append({
        "type": "header",
        "magic": "stepback/.sb",
        "format_version": 1,
        "wallclock_ns": 0,
    })
    for i in range(n - 1):
        kind = i % 23
        if kind == 7:
            bodies.append({
                "type": "blob",
                "i": i,
                "sha256": ("ab" * 32),
                "len": 4096,
            })
        elif kind == 17:
            bodies.append({
                "type": "tail",
                "wallclock_ns": i * 1_000,
            })
        else:
            bodies.append({
                "type": "step",
                "step": {
                    "id": i,
                    "kind": "llm_call" if i % 2 else "tool_call",
                    "model": "gpt-4o-2024-11-20",
                    "prompt_hash": "deadbeef" * 8,
                    "wallclock_ns": i * 1_000,
                },
            })
    return bodies


def _time_one_regime(
    writer_id: str,
    bodies: List[Dict[str, Any]],
    body_bytes_list: List[bytes],
    *,
    flag_on: bool,
    temporal_hook: bool,
    warmup: int,
) -> List[float]:
    """Time ``observe_frame`` per body and return per-call microseconds."""
    # Configure the flag and hook list in a fully reset state so every
    # regime measures from an identical baseline.
    _l1.reset()
    _l1.OBSERVE_HOOKS.clear()
    if flag_on:
        os.environ[_l1.FLAG_NAME] = "1"
    else:
        os.environ.pop(_l1.FLAG_NAME, None)
    if temporal_hook and _cs.comet_sigma_available():
        # Register the Step-2 temporal-basis projector hook so each
        # observation also drives the 1s/10s/1m/10m window aggregates.
        _l1t.install_hook()

    # Warmup loop (timed but discarded).
    for body, body_bytes in zip(bodies[:warmup], body_bytes_list[:warmup]):
        _l1.observe_frame(writer_id, body, body_bytes)

    # Measured loop.
    samples: List[float] = []
    samples_append = samples.append
    perf = time.perf_counter_ns
    for body, body_bytes in zip(bodies[warmup:], body_bytes_list[warmup:]):
        t0 = perf()
        _l1.observe_frame(writer_id, body, body_bytes)
        samples_append((perf() - t0) / 1_000.0)

    # Restore quiet state for the next regime.
    if temporal_hook and _cs.comet_sigma_available():
        _l1t.uninstall_hook()
        _l1t.reset()
    _l1.OBSERVE_HOOKS.clear()
    os.environ.pop(_l1.FLAG_NAME, None)
    _l1.reset()
    return samples


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(n_frames: int = 5_000, warmup_frames: int = 200) -> L1TraceWriterLatencyResult:
    """Run the L1 emission-latency bench and return the result.

    Parameters
    ----------
    n_frames:
        Total synthetic frames to feed the emitter per regime
        (warmup + measured). Must be ``> warmup_frames``.
    warmup_frames:
        Frames to time and discard before the measured loop. Allows
        the per-writer state to be populated and the JIT / CPU caches
        to warm up so percentiles are not biased by first-call costs.
    """
    if n_frames < 2:
        raise ValueError("n_frames must be >= 2")
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be >= 0")
    if warmup_frames >= n_frames:
        raise ValueError("warmup_frames must be < n_frames")

    bodies = _make_frame_bodies(n_frames)
    # Pre-canonicalise so we don't time the encoder.
    body_bytes_list = [canonical_json(b) for b in bodies]

    # Save & restore any pre-existing flag state so the bench is
    # neighbourly when run inside a larger test session.
    prev_flag = os.environ.get(_l1.FLAG_NAME)
    prev_hooks = list(_l1.OBSERVE_HOOKS)
    try:
        off = _stats(_time_one_regime(
            "bench-off", bodies, body_bytes_list,
            flag_on=False, temporal_hook=False, warmup=warmup_frames,
        ))
        on = _stats(_time_one_regime(
            "bench-on", bodies, body_bytes_list,
            flag_on=True, temporal_hook=False, warmup=warmup_frames,
        ))
        on_temporal = _stats(_time_one_regime(
            "bench-on-temporal", bodies, body_bytes_list,
            flag_on=True, temporal_hook=True, warmup=warmup_frames,
        ))
    finally:
        # Restore flag + hook list exactly as we found them.
        _l1.OBSERVE_HOOKS.clear()
        _l1.OBSERVE_HOOKS.extend(prev_hooks)
        if prev_flag is None:
            os.environ.pop(_l1.FLAG_NAME, None)
        else:
            os.environ[_l1.FLAG_NAME] = prev_flag
        _l1.reset()

    return L1TraceWriterLatencyResult(
        n_frames=n_frames,
        warmup_frames=warmup_frames,
        comet_sigma_available=_cs.comet_sigma_available(),
        schema_version=SCHEMA_VERSION,
        off=off,
        on=on,
        on_temporal=on_temporal,
        overhead_on_us=on.mean_us - off.mean_us,
        overhead_on_temporal_us=on_temporal.mean_us - off.mean_us,
    )


def run_and_save(out_path: str, **kwargs: Any) -> L1TraceWriterLatencyResult:
    """Convenience: run the bench and JSON-dump it to ``out_path``."""
    result = run(**kwargs)
    d = os.path.dirname(os.path.abspath(out_path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result.to_json(), f, indent=2, sort_keys=True)
        f.write("\n")
    return result


__all__ = [
    "SCHEMA_VERSION",
    "L1TraceWriterLatencyResult",
    "run",
    "run_and_save",
]
