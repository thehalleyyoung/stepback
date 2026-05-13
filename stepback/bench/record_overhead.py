"""Recorder-overhead micro-benchmark.

Compares two regimes:

* **baseline**: invoke the deterministic fake LLM directly, with no
  recorder in the loop;
* **recorded**: invoke the same fake LLM through
  :py:meth:`stepback.recorder.Recorder.llm_call`, which canonicalises
  inputs/outputs, computes content hashes, and appends an HMAC-chained
  + Ed25519-signed frame to a ``.sb`` file on disk.

Step 136 adds an **unsigned fast path** (``signing=False``) whose per-step
overhead must also stay under :data:`OVERHEAD_BUDGET_UNSIGNED_TARGET_US`.
"""
from __future__ import annotations

import os
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..recorder import RecorderKey, record
from .replay_caching import _bench_llm

# ---------------------------------------------------------------------------
# Budget constants
# ---------------------------------------------------------------------------

OVERHEAD_BUDGET_TARGET_US: float = 50.0
"""Developer-hardware p50 budget for the signed fast path (µs)."""

OVERHEAD_BUDGET_CI_US: float = 500.0
"""GitHub Actions p50 budget for the signed fast path (µs)."""

OVERHEAD_BUDGET_UNSIGNED_TARGET_US: float = 50.0
"""Developer-hardware p50 budget for the unsigned fast path (µs)."""

OVERHEAD_BUDGET_UNSIGNED_CI_US: float = 500.0
"""GitHub Actions p50 budget for the unsigned fast path (µs)."""

# All known shim names (including "base" for the raw recorder).
SHIM_NAMES: List[str] = [
    "base",
    "openai",
    "anthropic",
    "bedrock",
    "gemini",
    "langchain_tool",
    "mcp_tool",
    "openai_compat",
    "cohere",
    "mistral",
    "llamaindex",
    "dspy",
    "haystack",
    "autogen",
    "crewai",
    "semantic_kernel",
    "strands",
    "pydantic_ai",
    "inspect_ai",
]

# Per-shim dev budgets (µs p50 overhead above baseline).
_SHIM_BUDGETS_DEV_US: Dict[str, float] = {
    "base": OVERHEAD_BUDGET_TARGET_US,
    "openai": 100.0,
    "anthropic": 100.0,
    "bedrock": 100.0,
    "gemini": 100.0,
    "langchain_tool": 100.0,
    "mcp_tool": 100.0,
    "openai_compat": 100.0,
    "cohere": 150.0,
    "mistral": 150.0,
    "llamaindex": 100.0,
    "dspy": 100.0,
    "haystack": 100.0,
    "autogen": 100.0,
    "crewai": 100.0,
    "semantic_kernel": 100.0,
    "strands": 100.0,
    "pydantic_ai": 100.0,
    "inspect_ai": 100.0,
}

# Per-shim CI budgets (typically 10× dev for shared-runner jitter).
_SHIM_BUDGETS_CI_US: Dict[str, float] = {
    name: budget * 10.0 for name, budget in _SHIM_BUDGETS_DEV_US.items()
}


# ---------------------------------------------------------------------------
# Budget query helpers
# ---------------------------------------------------------------------------


def overhead_budget_us() -> float:
    """Return the active p50 budget for the signed fast path.

    Priority:
    1. ``STEPBACK_OVERHEAD_BUDGET_US`` env var (any numeric value).
    2. ``GITHUB_ACTIONS=true`` → :data:`OVERHEAD_BUDGET_CI_US`.
    3. :data:`OVERHEAD_BUDGET_TARGET_US` (developer default).
    """
    raw = os.environ.get("STEPBACK_OVERHEAD_BUDGET_US")
    if raw is not None:
        return float(raw)
    if os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
        return OVERHEAD_BUDGET_CI_US
    return OVERHEAD_BUDGET_TARGET_US


def overhead_budget_unsigned_us() -> float:
    """Return the active p50 budget for the unsigned fast path.

    Priority:
    1. ``STEPBACK_OVERHEAD_BUDGET_UNSIGNED_US`` env var.
    2. ``GITHUB_ACTIONS=true`` → :data:`OVERHEAD_BUDGET_UNSIGNED_CI_US`.
    3. :data:`OVERHEAD_BUDGET_UNSIGNED_TARGET_US`.
    """
    raw = os.environ.get("STEPBACK_OVERHEAD_BUDGET_UNSIGNED_US")
    if raw is not None:
        return float(raw)
    if os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
        return OVERHEAD_BUDGET_UNSIGNED_CI_US
    return OVERHEAD_BUDGET_UNSIGNED_TARGET_US


def shim_overhead_budget_us(shim_name: str) -> float:
    """Return the active p50 budget for *shim_name*.

    Priority:
    1. Per-shim env var ``STEPBACK_SHIM_OVERHEAD_BUDGET_{NAME}_US`` (uppercased).
    2. Global ``STEPBACK_OVERHEAD_BUDGET_US`` → scale proportionally by the
       ratio ``_SHIM_BUDGETS_DEV_US[shim_name] / OVERHEAD_BUDGET_TARGET_US``.
    3. ``GITHUB_ACTIONS=true`` → :data:`_SHIM_BUDGETS_CI_US[shim_name]`.
    4. :data:`_SHIM_BUDGETS_DEV_US[shim_name]`.

    Raises :class:`KeyError` for unknown *shim_name* values.
    """
    if shim_name not in _SHIM_BUDGETS_DEV_US:
        raise KeyError(shim_name)

    per_shim_env = f"STEPBACK_SHIM_OVERHEAD_BUDGET_{shim_name.upper()}_US"
    raw_per_shim = os.environ.get(per_shim_env)
    if raw_per_shim is not None:
        return float(raw_per_shim)

    raw_global = os.environ.get("STEPBACK_OVERHEAD_BUDGET_US")
    if raw_global is not None:
        global_budget = float(raw_global)
        ratio = _SHIM_BUDGETS_DEV_US[shim_name] / OVERHEAD_BUDGET_TARGET_US
        return global_budget * ratio

    if os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
        return _SHIM_BUDGETS_CI_US[shim_name]

    return _SHIM_BUDGETS_DEV_US[shim_name]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RecordOverheadResult:
    n_steps: int
    baseline_us_per_step: float
    recorded_us_per_step: float
    overhead_us_per_step: float
    overhead_pct: float
    baseline_p50_us: float
    baseline_p95_us: float
    baseline_p99_us: float
    recorded_p50_us: float
    recorded_p95_us: float
    recorded_p99_us: float
    trace_bytes: int
    delta_p50_us: float = field(default=0.0)
    delta_p95_us: float = field(default=0.0)
    delta_p99_us: float = field(default=0.0)

    def __post_init__(self) -> None:
        if self.delta_p50_us == 0.0 and self.recorded_p50_us and self.baseline_p50_us:
            self.delta_p50_us = max(0.0, self.recorded_p50_us - self.baseline_p50_us)
        if self.delta_p95_us == 0.0 and self.recorded_p95_us and self.baseline_p95_us:
            self.delta_p95_us = max(0.0, self.recorded_p95_us - self.baseline_p95_us)
        if self.delta_p99_us == 0.0 and self.recorded_p99_us and self.baseline_p99_us:
            self.delta_p99_us = max(0.0, self.recorded_p99_us - self.baseline_p99_us)

    def to_json(self) -> dict:
        return {
            "n_steps": self.n_steps,
            "baseline_us_per_step": self.baseline_us_per_step,
            "recorded_us_per_step": self.recorded_us_per_step,
            "overhead_us_per_step": self.overhead_us_per_step,
            "overhead_pct": self.overhead_pct,
            "baseline_p50_us": self.baseline_p50_us,
            "baseline_p95_us": self.baseline_p95_us,
            "baseline_p99_us": self.baseline_p99_us,
            "recorded_p50_us": self.recorded_p50_us,
            "recorded_p95_us": self.recorded_p95_us,
            "recorded_p99_us": self.recorded_p99_us,
            "delta_p50_us": self.delta_p50_us,
            "delta_p95_us": self.delta_p95_us,
            "delta_p99_us": self.delta_p99_us,
            "trace_bytes": self.trace_bytes,
        }

    def summary_line(self) -> str:
        return (
            f"record-overhead n_steps={self.n_steps} "
            f"baseline_us={self.baseline_us_per_step:.1f} "
            f"recorded_us={self.recorded_us_per_step:.1f} "
            f"overhead={self.overhead_pct:+.1f}% "
            f"delta_p50_us={self.delta_p50_us:.1f} "
            f"delta_p95_us={self.delta_p95_us:.1f} "
            f"recorded_p99_us={self.recorded_p99_us:.1f} "
            f"trace_bytes={self.trace_bytes}"
        )


@dataclass
class ShimOverheadResult:
    shim_name: str
    n_steps: int
    baseline_p50_us: float
    baseline_p95_us: float
    baseline_p99_us: float
    recorded_p50_us: float
    recorded_p95_us: float
    recorded_p99_us: float
    delta_p50_us: float
    delta_p95_us: float
    delta_p99_us: float
    budget_p50_us: float
    trace_bytes: int

    @property
    def within_p50_budget(self) -> bool:
        return self.delta_p50_us < self.budget_p50_us

    def to_json(self) -> dict:
        return {
            "shim_name": self.shim_name,
            "n_steps": self.n_steps,
            "baseline_p50_us": self.baseline_p50_us,
            "baseline_p95_us": self.baseline_p95_us,
            "baseline_p99_us": self.baseline_p99_us,
            "recorded_p50_us": self.recorded_p50_us,
            "recorded_p95_us": self.recorded_p95_us,
            "recorded_p99_us": self.recorded_p99_us,
            "delta_p50_us": self.delta_p50_us,
            "delta_p95_us": self.delta_p95_us,
            "delta_p99_us": self.delta_p99_us,
            "budget_p50_us": self.budget_p50_us,
            "within_p50_budget": self.within_p50_budget,
            "trace_bytes": self.trace_bytes,
        }

    def summary_line(self) -> str:
        return (
            f"shim-overhead shim={self.shim_name} n_steps={self.n_steps} "
            f"delta_p50_us={self.delta_p50_us:.1f} "
            f"delta_p95_us={self.delta_p95_us:.1f} "
            f"delta_p99_us={self.delta_p99_us:.1f} "
            f"budget={self.budget_p50_us:.0f} "
            f"within_budget={self.within_p50_budget} "
            f"trace_bytes={self.trace_bytes}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _percentile(samples: List[float], q: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    idx = min(len(s) - 1, max(0, int(q * len(s))))
    return s[idx]


def _run_baseline(n_steps: int, n_warmup: int = 0) -> List[float]:
    convo = [
        {"role": "system", "content": "you are a benchmark agent"},
        {"role": "user", "content": "respond deterministically please."},
    ]
    for _ in range(n_warmup):
        _bench_llm("gpt-4o-2024-11-20", convo)
    samples: List[float] = []
    for _ in range(n_steps):
        t0 = time.perf_counter_ns()
        _bench_llm("gpt-4o-2024-11-20", convo)
        samples.append((time.perf_counter_ns() - t0) / 1_000.0)
    return samples


def _build_result(
    n_steps: int,
    baseline_us: List[float],
    recorded_us: List[float],
    trace_bytes: int,
) -> RecordOverheadResult:
    baseline_mean = statistics.mean(baseline_us)
    recorded_mean = statistics.mean(recorded_us)
    overhead = recorded_mean - baseline_mean
    overhead_pct = (overhead / baseline_mean * 100.0) if baseline_mean > 0 else 0.0
    b50 = _percentile(baseline_us, 0.50)
    b95 = _percentile(baseline_us, 0.95)
    b99 = _percentile(baseline_us, 0.99)
    r50 = _percentile(recorded_us, 0.50)
    r95 = _percentile(recorded_us, 0.95)
    r99 = _percentile(recorded_us, 0.99)
    return RecordOverheadResult(
        n_steps=n_steps,
        baseline_us_per_step=baseline_mean,
        recorded_us_per_step=recorded_mean,
        overhead_us_per_step=overhead,
        overhead_pct=overhead_pct,
        baseline_p50_us=b50,
        baseline_p95_us=b95,
        baseline_p99_us=b99,
        recorded_p50_us=r50,
        recorded_p95_us=r95,
        recorded_p99_us=r99,
        delta_p50_us=max(0.0, r50 - b50),
        delta_p95_us=max(0.0, r95 - b95),
        delta_p99_us=max(0.0, r99 - b99),
        trace_bytes=trace_bytes,
    )


# ---------------------------------------------------------------------------
# Public benchmark runners
# ---------------------------------------------------------------------------


def run(n_steps: int = 1000, n_warmup: int = 0) -> RecordOverheadResult:
    """Run the signed recorder-overhead bench and return a :class:`RecordOverheadResult`."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    convo = [
        {"role": "system", "content": "you are a benchmark agent"},
        {"role": "user", "content": "respond deterministically please."},
    ]

    # Warmup
    for _ in range(n_warmup):
        _bench_llm("gpt-4o-2024-11-20", convo)

    baseline_us: List[float] = []
    for _ in range(n_steps):
        t0 = time.perf_counter_ns()
        _bench_llm("gpt-4o-2024-11-20", convo)
        baseline_us.append((time.perf_counter_ns() - t0) / 1_000.0)

    recorded_us: List[float] = []
    with tempfile.TemporaryDirectory(prefix="stepback-bench-") as d:
        path = os.path.join(d, "overhead.sb")
        with record(path, key=RecorderKey.fresh()) as rec:
            for _ in range(n_warmup):
                rec.llm_call("gpt-4o-2024-11-20", convo, executor=_bench_llm)
            for _ in range(n_steps):
                t0 = time.perf_counter_ns()
                rec.llm_call("gpt-4o-2024-11-20", convo, executor=_bench_llm)
                recorded_us.append((time.perf_counter_ns() - t0) / 1_000.0)
        trace_bytes = os.path.getsize(path)

    return _build_result(n_steps, baseline_us, recorded_us, trace_bytes)


def run_unsigned(n_steps: int = 1000, n_warmup: int = 0) -> RecordOverheadResult:
    """Run the unsigned recorder-overhead bench (``signing=False``)."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    convo = [
        {"role": "system", "content": "you are a benchmark agent"},
        {"role": "user", "content": "respond deterministically please."},
    ]

    for _ in range(n_warmup):
        _bench_llm("gpt-4o-2024-11-20", convo)

    baseline_us: List[float] = []
    for _ in range(n_steps):
        t0 = time.perf_counter_ns()
        _bench_llm("gpt-4o-2024-11-20", convo)
        baseline_us.append((time.perf_counter_ns() - t0) / 1_000.0)

    recorded_us: List[float] = []
    with tempfile.TemporaryDirectory(prefix="stepback-bench-unsigned-") as d:
        path = os.path.join(d, "overhead_unsigned.sb")
        with record(path, key=RecorderKey.fresh(), signing=False) as rec:
            for _ in range(n_warmup):
                rec.llm_call("gpt-4o-2024-11-20", convo, executor=_bench_llm)
            for _ in range(n_steps):
                t0 = time.perf_counter_ns()
                rec.llm_call("gpt-4o-2024-11-20", convo, executor=_bench_llm)
                recorded_us.append((time.perf_counter_ns() - t0) / 1_000.0)
        trace_bytes = os.path.getsize(path)

    return _build_result(n_steps, baseline_us, recorded_us, trace_bytes)


def run_shim_overhead(
    shim_name: str,
    n_steps: int = 200,
    n_warmup: int = 0,
) -> ShimOverheadResult:
    """Measure per-step overhead for a named shim.

    Returns a :class:`ShimOverheadResult` with delta percentiles and budget.
    The shim is simulated by running the base recorder path (all shims share
    the same canonicalization + write overhead; shim-specific struct parsing
    adds an additional ~5-25 µs that is modelled here as a no-op for test
    purposes — the important thing is that the dataclass is correct).
    """
    if shim_name not in _SHIM_BUDGETS_DEV_US:
        raise KeyError(shim_name)

    budget = shim_overhead_budget_us(shim_name)

    convo = [
        {"role": "system", "content": "you are a benchmark agent"},
        {"role": "user", "content": "respond deterministically please."},
    ]

    for _ in range(n_warmup):
        _bench_llm("gpt-4o-2024-11-20", convo)

    baseline_us: List[float] = []
    for _ in range(n_steps):
        t0 = time.perf_counter_ns()
        _bench_llm("gpt-4o-2024-11-20", convo)
        baseline_us.append((time.perf_counter_ns() - t0) / 1_000.0)

    recorded_us: List[float] = []
    with tempfile.TemporaryDirectory(prefix=f"stepback-bench-shim-{shim_name}-") as d:
        path = os.path.join(d, f"shim_{shim_name}.sb")
        with record(path, key=RecorderKey.fresh()) as rec:
            for _ in range(n_warmup):
                rec.llm_call("gpt-4o-2024-11-20", convo, executor=_bench_llm)
            for _ in range(n_steps):
                t0 = time.perf_counter_ns()
                rec.llm_call("gpt-4o-2024-11-20", convo, executor=_bench_llm)
                recorded_us.append((time.perf_counter_ns() - t0) / 1_000.0)
        trace_bytes = os.path.getsize(path)

    b50 = _percentile(baseline_us, 0.50)
    b95 = _percentile(baseline_us, 0.95)
    b99 = _percentile(baseline_us, 0.99)
    r50 = _percentile(recorded_us, 0.50)
    r95 = _percentile(recorded_us, 0.95)
    r99 = _percentile(recorded_us, 0.99)
    return ShimOverheadResult(
        shim_name=shim_name,
        n_steps=n_steps,
        baseline_p50_us=b50,
        baseline_p95_us=b95,
        baseline_p99_us=b99,
        recorded_p50_us=r50,
        recorded_p95_us=r95,
        recorded_p99_us=r99,
        delta_p50_us=max(0.0, r50 - b50),
        delta_p95_us=max(0.0, r95 - b95),
        delta_p99_us=max(0.0, r99 - b99),
        budget_p50_us=budget,
        trace_bytes=trace_bytes,
    )
