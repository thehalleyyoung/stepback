"""Minimization benchmark for causal trace reduction.

Measures four headline metrics for each supported reduction strategy
(DDMin, LinearShrink, BinaryHalving):

* **final_trace_size** — the number of substitutions in the 1-minimal
  causal subset.  Lower is better; 1 means the algorithm successfully
  isolated the single responsible substitution.
* **predicate_stable** — ``True`` when the predicate still fires on the
  final-replay result *after* minimization.  A stable result means the
  minimal subset is a true causal explanation.
* **llm_calls_spent** — estimated LLM re-executions consumed by the
  oracle.  Computed as ``probes × llm_fraction_of_trace``, where the
  LLM fraction is the fraction of ``llm_call`` steps in the original
  synthetic trace and each probe re-executes one dirty-set slice.
* **speedup_vs_naive** — ratio of naive (LinearShrink) probes to the
  strategy's own probes.  Values > 1 mean the strategy uses fewer
  oracle queries than a sequential drop-one pass.

Comparison to naive ddmin
~~~~~~~~~~~~~~~~~~~~~~~~~
The "naive" baseline is :class:`~stepback.minimize.LinearShrinkStrategy`,
which makes one pass through the substitution list and drops each
candidate if the predicate still fires.  This is O(N) probes in the
worst case and finds 1-minimal subsets for single-blame scenarios.
DDMin (Zeller & Hildebrandt) is O(N log N) in the worst case but
typically far fewer probes due to its coarse-to-fine bisection.
This benchmark measures both and reports the probe-count ratio.

The trace fixture
~~~~~~~~~~~~~~~~~
A deterministic N-step synthetic trace is built using the same
:class:`~stepback.bench.replay_caching.SyntheticTrace` / ``_bench_*``
helpers as the replay-caching benchmark, so the canonical hashes,
parallel-branch frames, and context-rebinding are exactly what
production would produce.

The substitution set contains:

* **One causal substitution** — a :class:`~stepback.substitutions.ToolOutputSubstitution`
  that injects a uniquely-typed sentinel marker into one tool step.
* **N_noisy noisy substitutions** — :class:`~stepback.substitutions.PromptSubstitution`
  objects at other LLM-call steps that perturb prompts without
  introducing the marker.

The predicate returns ``True`` iff the sentinel marker appears in any
step output of the replay, so the minimization algorithm must isolate
exactly the one causal substitution from among the noisy decoys.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..minimize import (
    BinaryHalvingStrategy,
    BudgetExhausted,
    DDMinStrategy,
    LinearShrinkStrategy,
    MinimizationResult,
    MinimizeOptions,
    minimize_substitutions,
)
from ..replay import Executor, ReplayResult, replay
from ..substitutions import (
    PromptSubstitution,
    SubstitutionSet,
    ToolOutputSubstitution,
)
from .replay_caching import (
    SyntheticTrace,
    _bench_llm,
    _bench_router,
    _bench_tool,
)

# Sentinel injected by the causal substitution; the predicate looks for it.
_CAUSAL_MARKER = "stepback-bench-minimization-causal"


# ------------------------------------------------------------ predicate


def _bench_predicate(result: ReplayResult) -> bool:
    """Return True when the causal marker appears in any step's outputs."""
    for sv in result.steps:
        out = getattr(sv, "outputs", None) or {}
        if _CAUSAL_MARKER in repr(out):
            return True
    return False


# ------------------------------------------------------------ helpers


def _executor() -> Executor:
    return Executor(
        llm=_bench_llm,
        tool=_bench_tool,
        router=_bench_router,
    )


def _build_substitution_set(
    recorded_steps: List[dict],
    n_noisy: int,
) -> SubstitutionSet:
    """Build a SubstitutionSet with one causal ToolOutputSubstitution plus
    ``n_noisy`` PromptSubstitution decoys at distinct LLM steps.

    Returns an empty SubstitutionSet if the trace lacks enough steps.
    """
    tool_steps = [
        s for s in recorded_steps if s.get("step_kind") == "tool_call"
    ]
    llm_steps = [
        s for s in recorded_steps if s.get("step_kind") == "llm_call"
    ]
    if not tool_steps:
        return SubstitutionSet(items=[])

    # Pick the first tool step as the causal target.
    causal_step = tool_steps[0]
    causal_sub = ToolOutputSubstitution(
        at_step=causal_step["step_id"],
        fake_response={
            "causal": True,
            "marker": _CAUSAL_MARKER,
            "tool": causal_step.get("inputs", {}).get("name", "unknown"),
        },
    )

    # Add noisy PromptSubstitution decoys at distinct LLM steps (skip step 0
    # so the causal tool step's parent is not disturbed).
    noisy_llm = [s for s in llm_steps if s["step_id"] != causal_step["step_id"]]
    noisy_subs = []
    for i, step in enumerate(noisy_llm[:n_noisy]):
        recorded_msgs = list(step["inputs"].get("messages") or [])
        noisy_msgs = list(recorded_msgs) + [
            {
                "role": "user",
                "content": f"[noisy decoy {i} — bench]",
            }
        ]
        noisy_subs.append(
            PromptSubstitution(at_step=step["step_id"], new_messages=noisy_msgs)
        )

    return SubstitutionSet(items=[causal_sub] + noisy_subs)


# ------------------------------------------------------------ result


@dataclass
class StrategyResult:
    """Benchmark outcome for a single minimization strategy."""

    strategy_name: str
    """Name string from the strategy class (e.g. ``"ddmin"``).
    """
    original_subs: int
    """Number of substitutions in the input set."""
    final_subs: int
    """Number of substitutions in the 1-minimal output."""
    probes: int
    """Number of distinct oracle replays (cache hits excluded)."""
    cache_hits: int
    """Oracle cache hits."""
    llm_calls_spent: float
    """Estimated LLM call re-executions: ``probes × llm_fraction``.
    The LLM fraction is the fraction of ``llm_call`` steps in the
    original trace; each probe replays the dirty-set slice whose
    expected LLM-call count equals ``llm_fraction × dirty_size``.
    """
    predicate_stable: bool
    """``True`` when the predicate still fires on the post-minimize result."""
    isolated_causal: bool
    """``True`` when ``final_subs == 1`` (the single causal sub was found)."""
    wall_time_ms: float

    def to_json(self) -> dict:
        return {
            "strategy_name": self.strategy_name,
            "original_subs": self.original_subs,
            "final_subs": self.final_subs,
            "probes": self.probes,
            "cache_hits": self.cache_hits,
            "llm_calls_spent": self.llm_calls_spent,
            "predicate_stable": self.predicate_stable,
            "isolated_causal": self.isolated_causal,
            "wall_time_ms": self.wall_time_ms,
        }


@dataclass
class MinimizationBenchResult:
    """Aggregate result from a single minimization benchmark run.

    Contains per-strategy :class:`StrategyResult` objects and
    derived comparison metrics.
    """

    n_steps: int
    """Actual number of recorded steps in the synthetic trace."""
    n_noisy: int
    """Number of noisy decoy substitutions."""
    n_trials: int
    """Number of independent traces averaged over."""
    strategy_results: Dict[str, StrategyResult]
    """Per-strategy results keyed by ``strategy_name``."""
    mean_speedup_ddmin_vs_naive: float
    """Mean ratio ``linear_probes / ddmin_probes`` across trials.
    Values > 1.0 mean DDMin used fewer oracle queries than LinearShrink.
    """
    wall_time_ms: float
    """Total wall time for all trials and strategies."""

    def to_json(self) -> dict:
        return {
            "n_steps": self.n_steps,
            "n_noisy": self.n_noisy,
            "n_trials": self.n_trials,
            "strategy_results": {
                k: v.to_json() for k, v in self.strategy_results.items()
            },
            "mean_speedup_ddmin_vs_naive": self.mean_speedup_ddmin_vs_naive,
            "wall_time_ms": self.wall_time_ms,
        }

    def summary_line(self) -> str:
        parts = [
            f"minimization n_steps={self.n_steps} n_noisy={self.n_noisy} "
            f"trials={self.n_trials} speedup_ddmin_vs_naive={self.mean_speedup_ddmin_vs_naive:.2f}x"
        ]
        for name, sr in sorted(self.strategy_results.items()):
            stable_flag = "stable" if sr.predicate_stable else "UNSTABLE"
            parts.append(
                f"  {name}: final_subs={sr.final_subs} probes={sr.probes} "
                f"cache_hits={sr.cache_hits} llm_spent={sr.llm_calls_spent:.1f} "
                f"isolated={sr.isolated_causal} {stable_flag} "
                f"wall_ms={sr.wall_time_ms:.1f}"
            )
        return "\n".join(parts)


# ------------------------------------------------------------ runner


def run(
    n_steps: int,
    n_noisy: int = 5,
    n_trials: int = 5,
    *,
    seed: int = 0,
) -> MinimizationBenchResult:
    """Run the minimization benchmark and return a :class:`MinimizationBenchResult`.

    Parameters
    ----------
    n_steps :
        Target number of steps per synthetic trace.  The trace builder
        may overshoot slightly due to parallel-block rounding.
    n_noisy :
        Number of noisy :class:`~stepback.substitutions.PromptSubstitution`
        decoys to mix with the one causal substitution.  Must be >= 0.
    n_trials :
        Number of independent traces to build and average over.
    seed :
        Base RNG seed; each trial uses ``seed * 1000 + trial``.

    Returns
    -------
    MinimizationBenchResult

    Raises
    ------
    ValueError
        If ``n_steps < 1``, ``n_trials < 1``, or ``n_noisy < 0``.
    RuntimeError
        If the synthetic trace cannot produce enough substitutable steps.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_noisy < 0:
        raise ValueError("n_noisy must be >= 0")

    strategies: List[tuple] = [
        ("ddmin", DDMinStrategy()),
        ("linear_shrink", LinearShrinkStrategy()),
        ("binary_halving", BinaryHalvingStrategy()),
    ]

    # Accumulate per-trial lists keyed by strategy_name.
    acc_final_subs: Dict[str, List[int]] = {n: [] for n, _ in strategies}
    acc_probes: Dict[str, List[int]] = {n: [] for n, _ in strategies}
    acc_cache_hits: Dict[str, List[int]] = {n: [] for n, _ in strategies}
    acc_llm: Dict[str, List[float]] = {n: [] for n, _ in strategies}
    acc_stable: Dict[str, List[bool]] = {n: [] for n, _ in strategies}
    acc_isolated: Dict[str, List[bool]] = {n: [] for n, _ in strategies}
    acc_wall: Dict[str, List[float]] = {n: [] for n, _ in strategies}
    acc_actual_steps: List[int] = []
    ddmin_probe_list: List[int] = []
    linear_probe_list: List[int] = []

    total_t0 = time.perf_counter_ns()

    for trial in range(n_trials):
        st = SyntheticTrace(n_steps=n_steps, seed=seed * 1000 + trial)
        try:
            path = st.build()
            t = replay(path)
            recorded = t.recorded_steps
            acc_actual_steps.append(len(recorded))

            # Compute LLM fraction for this trace.
            llm_count = sum(1 for s in recorded if s.get("step_kind") == "llm_call")
            llm_fraction = llm_count / max(1, len(recorded))

            subs = _build_substitution_set(recorded, n_noisy)
            if len(subs.items) < 2:
                raise RuntimeError(
                    f"trace with {len(recorded)} steps produced fewer than 2 "
                    "substitutable items; increase n_steps"
                )

            for strat_name, strategy in strategies:
                opts = MinimizeOptions(strategy=strategy)
                t_s = time.perf_counter_ns()
                try:
                    result: MinimizationResult = minimize_substitutions(
                        t,
                        subs,
                        _bench_predicate,
                        options=opts,
                        executor=_executor(),
                    )
                except BudgetExhausted as exc:
                    result = exc.partial
                wall_ms = (time.perf_counter_ns() - t_s) / 1_000_000.0

                final_count = len(result.minimal)
                stable = (
                    result.final_result is not None
                    and _bench_predicate(result.final_result)
                )
                isolated = final_count == 1

                # Re-create strategy for next trial (strategies are stateful
                # in the sense that some subclasses may cache internal state).
                acc_final_subs[strat_name].append(final_count)
                acc_probes[strat_name].append(result.probes)
                acc_cache_hits[strat_name].append(result.cache_hits)
                acc_llm[strat_name].append(result.probes * llm_fraction)
                acc_stable[strat_name].append(stable)
                acc_isolated[strat_name].append(isolated)
                acc_wall[strat_name].append(wall_ms)

            # Collect probe counts for speedup calculation.
            ddmin_probe_list.append(acc_probes["ddmin"][-1])
            linear_probe_list.append(acc_probes["linear_shrink"][-1])

            # Re-instantiate strategies for the next trial.
            strategies = [
                ("ddmin", DDMinStrategy()),
                ("linear_shrink", LinearShrinkStrategy()),
                ("binary_halving", BinaryHalvingStrategy()),
            ]
        finally:
            st.cleanup()

    total_wall_ms = (time.perf_counter_ns() - total_t0) / 1_000_000.0
    avg_steps = int(round(statistics.mean(acc_actual_steps))) if acc_actual_steps else n_steps
    original_subs = n_noisy + 1

    # Compute mean speedup DDMin vs. LinearShrink.
    speedup_list = [
        lin / dd if dd > 0 else 1.0
        for lin, dd in zip(linear_probe_list, ddmin_probe_list)
    ]
    mean_speedup = statistics.mean(speedup_list) if speedup_list else 1.0

    # Aggregate per-strategy results.
    strategy_results: Dict[str, StrategyResult] = {}
    for strat_name, _ in [
        ("ddmin", None),
        ("linear_shrink", None),
        ("binary_halving", None),
    ]:
        probes_list = acc_probes[strat_name]
        strategy_results[strat_name] = StrategyResult(
            strategy_name=strat_name,
            original_subs=original_subs,
            final_subs=int(round(statistics.mean(acc_final_subs[strat_name]))),
            probes=int(round(statistics.mean(probes_list))),
            cache_hits=int(round(statistics.mean(acc_cache_hits[strat_name]))),
            llm_calls_spent=statistics.mean(acc_llm[strat_name]),
            predicate_stable=all(acc_stable[strat_name]),
            isolated_causal=all(acc_isolated[strat_name]),
            wall_time_ms=sum(acc_wall[strat_name]),
        )

    return MinimizationBenchResult(
        n_steps=avg_steps,
        n_noisy=n_noisy,
        n_trials=n_trials,
        strategy_results=strategy_results,
        mean_speedup_ddmin_vs_naive=mean_speedup,
        wall_time_ms=total_wall_ms,
    )


def compare(
    suite: List[int],
    *,
    n_noisy: int = 5,
    n_trials: int = 5,
    seed: int = 0,
) -> Dict[int, "MinimizationBenchResult"]:
    """Run :func:`run` across multiple trace sizes."""
    out: Dict[int, MinimizationBenchResult] = {}
    for n in suite:
        out[n] = run(n_steps=n, n_noisy=n_noisy, n_trials=n_trials, seed=seed)
    return out
