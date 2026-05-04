"""Corpus-scale counterfactual sweep — README §Use-cases #3.

The README "Prompt A/B on real traffic" example sketches a loop that
takes a sample of 1000 production traces, branches each at ``step:0``,
applies a candidate substitution (e.g. a new system prompt), replays
forward against the per-step cache, and aggregates the resulting
:py:class:`~stepback.replay.BranchDiff` deltas into "mean cost delta",
"decisions changed", etc.

This module turns that example into a first-class API. Every other
stepback module operates on a single trace; ``sweep`` operates on a
*corpus* of traces and produces a single :py:class:`SweepReport` that:

* per-trace records both branches (baseline `A` vs counterfactual `B`)
  and the diff between them;
* aggregates cost deltas, divergent step counts, dirty-step counts,
  cache-hit ratios, and per-trace failures into distribution stats
  (count / mean / p50 / p95 / min / max / total);
* renders to JSON or Markdown for incident write-ups and CI gating.

Public surface (re-exported from :py:mod:`stepback`):

* :py:class:`SweepResult` — one trace's outcome.
* :py:class:`SweepReport` — the aggregate.
* :py:func:`sweep_traces` — drive the sweep.
* :py:func:`render_sweep_report` — Markdown render.
* :py:func:`render_sweep_report_json` — JSON render.

The sweep is purely cache-driven by default: with no executor, every
step is replayed from the recorded outputs (Executor with
``fallback_recorded=True``) so cost deltas come from *substitutions
themselves* (a model swap changing per-token rates, a policy block
zero-ing a downstream step's cost) rather than from re-execution.
Pass an ``Executor`` to actually re-invoke the LLM/tool on dirty
steps — this is the path the README's "test a new system prompt on
1000 production traces" use-case exercises against a real model.

Failures (corrupt trace, missing step id in a substitution spec,
executor exception) are caught per-trace and recorded as a
:py:class:`SweepFailure` so a single bad trace can never abort a
1000-trace sweep.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .branch_io import diff_replays, parse_substitution_spec
from .replay import BranchDiff, Executor, ReplayResult, Trace, replay
from .substitutions import Substitution, SubstitutionSet


# --------------------------------------------------------------- types


@dataclass
class SweepFailure:
    """A trace that could not be swept; recorded, never raised."""

    trace_path: str
    phase: str  # "load" | "baseline_replay" | "counterfactual_replay" | "diff"
    error_class: str
    message: str


@dataclass
class SweepResult:
    """One trace's contribution to a sweep."""

    trace_path: str
    step_count: int
    base_cost_usd: float
    cf_cost_usd: float
    cost_delta_usd: float
    divergent_step_count: int
    base_dirty_count: int
    cf_dirty_count: int
    base_cache_hits: int
    cf_cache_hits: int
    base_real_executions: int
    cf_real_executions: int
    decisions_changed: int  # divergent llm_call + router steps only
    diff: BranchDiff

    @property
    def divergent_fraction(self) -> float:
        if self.step_count == 0:
            return 0.0
        return self.divergent_step_count / self.step_count

    @property
    def cf_cache_hit_ratio(self) -> float:
        if self.step_count == 0:
            return 0.0
        return self.cf_cache_hits / self.step_count


@dataclass
class DistStats:
    """Min / mean / p50 / p95 / max / total over a numeric series."""

    count: int = 0
    total: float = 0.0
    mean: float = 0.0
    min: float = 0.0
    max: float = 0.0
    p50: float = 0.0
    p95: float = 0.0

    @classmethod
    def of(cls, xs: Sequence[float]) -> "DistStats":
        if not xs:
            return cls()
        s = sorted(float(x) for x in xs)
        n = len(s)
        return cls(
            count=n,
            total=round(sum(s), 8),
            mean=round(sum(s) / n, 8),
            min=round(s[0], 8),
            max=round(s[-1], 8),
            p50=round(s[max(0, n // 2 - (1 if n % 2 == 0 else 0))], 8),
            p95=round(s[min(n - 1, max(0, math.ceil(0.95 * n) - 1))], 8),
        )


@dataclass
class SweepReport:
    """Aggregate over an entire trace corpus."""

    n_traces_attempted: int
    n_traces_succeeded: int
    n_traces_failed: int
    n_traces_diverged: int  # at least one divergent step under the substitution
    n_decisions_changed: int  # llm_call/router steps that diverged across the corpus
    cost_delta_usd: DistStats
    divergent_step_count: DistStats
    divergent_fraction: DistStats
    base_cost_usd: DistStats
    cf_cost_usd: DistStats
    cf_cache_hit_ratio: DistStats
    cf_real_executions: DistStats
    results: List[SweepResult] = field(default_factory=list)
    failures: List[SweepFailure] = field(default_factory=list)
    baseline_substitutions: List[Dict[str, Any]] = field(default_factory=list)
    candidate_substitutions: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def diverged_fraction(self) -> float:
        if self.n_traces_succeeded == 0:
            return 0.0
        return self.n_traces_diverged / self.n_traces_succeeded


# ---------------------------------------------------------------- core


_DECISION_KINDS = {"llm_call", "router"}


def _coerce_subs(subs: Iterable[Any]) -> List[Substitution]:
    out: List[Substitution] = []
    for s in subs:
        if isinstance(s, str):
            out.append(parse_substitution_spec(s))
        elif isinstance(s, Substitution):
            out.append(s)
        elif isinstance(s, SubstitutionSet):
            out.extend(s.items)
        else:
            raise TypeError(
                f"sweep substitution must be Substitution|str spec|SubstitutionSet, "
                f"got {type(s).__name__}"
            )
    return out


def _summarise(
    trace_path: str,
    base: ReplayResult,
    cf: ReplayResult,
    diff: BranchDiff,
) -> SweepResult:
    decision_changes = sum(
        1
        for sd in diff.step_diffs
        if sd.kind in _DECISION_KINDS and sd.output_diff
    )
    step_count = max(len(base.steps), len(cf.steps))
    return SweepResult(
        trace_path=trace_path,
        step_count=step_count,
        base_cost_usd=round(base.total_cost_usd, 8),
        cf_cost_usd=round(cf.total_cost_usd, 8),
        cost_delta_usd=round(cf.total_cost_usd - base.total_cost_usd, 8),
        divergent_step_count=diff.divergent_step_count,
        base_dirty_count=base.dirty_count,
        cf_dirty_count=cf.dirty_count,
        base_cache_hits=base.cache_hit_count,
        cf_cache_hits=cf.cache_hit_count,
        base_real_executions=base.real_executions,
        cf_real_executions=cf.real_executions,
        decisions_changed=decision_changes,
        diff=diff,
    )


def sweep_traces(
    trace_paths: Sequence[str],
    substitutions: Iterable[Any],
    *,
    baseline_substitutions: Iterable[Any] = (),
    base_step: str = "step:1",
    branch_name_a: str = "baseline",
    branch_name_b: str = "candidate",
    executor_factory: Optional[Callable[[], Executor]] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    on_error: str = "record",  # "record" | "raise"
) -> SweepReport:
    """Run the README §Use-case-3 loop over a corpus of traces.

    For every path in ``trace_paths``:

    1. Load the trace with :py:func:`stepback.replay`.
    2. Build a baseline branch (``branch_name_a``) from
       ``baseline_substitutions`` (default: empty — the recorded run).
    3. Build a candidate branch (``branch_name_b``) from
       ``substitutions``.
    4. Replay both branches forward (cached unless an executor is wired).
    5. Diff them with :py:meth:`Trace.compare_branches`.

    A single failing trace does *not* abort the sweep: any exception
    is caught, recorded as a :py:class:`SweepFailure`, and the loop
    continues. Set ``on_error="raise"`` to opt out (useful for tests
    of the substitution spec itself).

    Substitutions may be passed as :py:class:`Substitution` instances,
    :py:class:`SubstitutionSet`, or string specs in the
    :py:func:`stepback.branch_io.parse_substitution_spec` grammar
    (so the CLI can forward ``--substitute`` flags directly).
    """
    if on_error not in ("record", "raise"):
        raise ValueError(f"on_error must be 'record' or 'raise', got {on_error!r}")

    base_subs = _coerce_subs(baseline_substitutions)
    cand_subs = _coerce_subs(substitutions)

    results: List[SweepResult] = []
    failures: List[SweepFailure] = []
    n_diverged = 0
    n_decisions_changed = 0

    paths = list(trace_paths)
    n = len(paths)

    for i, path in enumerate(paths):
        if progress is not None:
            progress(i, n, path)

        trace: Optional[Trace] = None
        phase = "load"
        try:
            trace = replay(path)
            phase = "baseline_replay"
            ba = trace.branch_at(base_step, name=branch_name_a)
            for s in base_subs:
                ba.substitute(s)
            executor_a = executor_factory() if executor_factory else Executor(
                fallback_recorded=True
            )
            ra = ba.replay_forward(executor=executor_a)

            phase = "counterfactual_replay"
            bb = trace.branch_at(base_step, name=branch_name_b)
            for s in cand_subs:
                bb.substitute(s)
            executor_b = executor_factory() if executor_factory else Executor(
                fallback_recorded=True
            )
            rb = bb.replay_forward(executor=executor_b)

            phase = "diff"
            d = trace.compare_branches(ba, bb)

            sr = _summarise(path, ra, rb, d)
            results.append(sr)
            if sr.divergent_step_count > 0:
                n_diverged += 1
            n_decisions_changed += sr.decisions_changed

        except Exception as exc:
            if on_error == "raise":
                raise
            failures.append(
                SweepFailure(
                    trace_path=path,
                    phase=phase,
                    error_class=type(exc).__name__,
                    message=str(exc),
                )
            )
            continue

    if progress is not None:
        progress(n, n, "")

    cost_deltas = [r.cost_delta_usd for r in results]
    div_counts = [float(r.divergent_step_count) for r in results]
    div_fracs = [r.divergent_fraction for r in results]
    base_costs = [r.base_cost_usd for r in results]
    cf_costs = [r.cf_cost_usd for r in results]
    cache_ratios = [r.cf_cache_hit_ratio for r in results]
    real_execs = [float(r.cf_real_executions) for r in results]

    return SweepReport(
        n_traces_attempted=n,
        n_traces_succeeded=len(results),
        n_traces_failed=len(failures),
        n_traces_diverged=n_diverged,
        n_decisions_changed=n_decisions_changed,
        cost_delta_usd=DistStats.of(cost_deltas),
        divergent_step_count=DistStats.of(div_counts),
        divergent_fraction=DistStats.of(div_fracs),
        base_cost_usd=DistStats.of(base_costs),
        cf_cost_usd=DistStats.of(cf_costs),
        cf_cache_hit_ratio=DistStats.of(cache_ratios),
        cf_real_executions=DistStats.of(real_execs),
        results=results,
        failures=failures,
        baseline_substitutions=[_sub_to_dict(s) for s in base_subs],
        candidate_substitutions=[_sub_to_dict(s) for s in cand_subs],
    )


def _sub_to_dict(sub: Substitution) -> Dict[str, Any]:
    from .branch_io import substitution_to_dict
    return substitution_to_dict(sub)


# ----------------------------------------------------------- rendering


def render_sweep_report_json(report: SweepReport, *, include_diffs: bool = False) -> Dict[str, Any]:
    """JSON-serialisable shape for the sweep report."""

    def _ds(d: DistStats) -> Dict[str, Any]:
        return {
            "count": d.count, "total": d.total, "mean": d.mean,
            "min": d.min, "p50": d.p50, "p95": d.p95, "max": d.max,
        }

    def _r(r: SweepResult) -> Dict[str, Any]:
        out = {
            "trace_path": r.trace_path,
            "step_count": r.step_count,
            "base_cost_usd": r.base_cost_usd,
            "cf_cost_usd": r.cf_cost_usd,
            "cost_delta_usd": r.cost_delta_usd,
            "divergent_step_count": r.divergent_step_count,
            "divergent_fraction": round(r.divergent_fraction, 6),
            "decisions_changed": r.decisions_changed,
            "cf_cache_hit_ratio": round(r.cf_cache_hit_ratio, 6),
            "cf_real_executions": r.cf_real_executions,
        }
        if include_diffs:
            out["diff"] = {
                "a": r.diff.a, "b": r.diff.b,
                "divergent_step_count": r.diff.divergent_step_count,
                "total_cost_delta_usd": r.diff.total_cost_delta_usd,
                "step_diffs": [
                    {
                        "step_id": sd.step_id, "kind": sd.kind,
                        "cost_delta_usd": sd.cost_delta_usd,
                        "diverged_from_cache": sd.diverged_from_cache,
                        "output_diff": sd.output_diff,
                    }
                    for sd in r.diff.step_diffs
                ],
            }
        return out

    return {
        "summary": {
            "n_traces_attempted": report.n_traces_attempted,
            "n_traces_succeeded": report.n_traces_succeeded,
            "n_traces_failed": report.n_traces_failed,
            "n_traces_diverged": report.n_traces_diverged,
            "diverged_fraction": round(report.diverged_fraction, 6),
            "n_decisions_changed": report.n_decisions_changed,
        },
        "stats": {
            "cost_delta_usd": _ds(report.cost_delta_usd),
            "divergent_step_count": _ds(report.divergent_step_count),
            "divergent_fraction": _ds(report.divergent_fraction),
            "base_cost_usd": _ds(report.base_cost_usd),
            "cf_cost_usd": _ds(report.cf_cost_usd),
            "cf_cache_hit_ratio": _ds(report.cf_cache_hit_ratio),
            "cf_real_executions": _ds(report.cf_real_executions),
        },
        "baseline_substitutions": report.baseline_substitutions,
        "candidate_substitutions": report.candidate_substitutions,
        "results": [_r(r) for r in report.results],
        "failures": [
            {
                "trace_path": f.trace_path, "phase": f.phase,
                "error_class": f.error_class, "message": f.message,
            }
            for f in report.failures
        ],
    }


def render_sweep_report(
    report: SweepReport,
    *,
    title: Optional[str] = None,
    max_rows: int = 50,
) -> str:
    """Markdown counterfactual sweep report.

    Header, aggregate stats table, per-trace table (truncated to
    ``max_rows``), and a failures section if any.
    """
    title = title or "stepback sweep report"
    lines: List[str] = [f"# {title}", ""]
    s = report
    lines.append(
        f"- traces attempted: **{s.n_traces_attempted}** "
        f"(succeeded: {s.n_traces_succeeded}, failed: {s.n_traces_failed})"
    )
    lines.append(
        f"- traces with at least one divergent step: **{s.n_traces_diverged}** "
        f"({s.diverged_fraction*100:.1f}%)"
    )
    lines.append(f"- decisions changed across corpus: **{s.n_decisions_changed}**")
    lines.append(
        f"- mean Δcost / trace: **${s.cost_delta_usd.mean:+.6f}** "
        f"(p50 ${s.cost_delta_usd.p50:+.6f}, p95 ${s.cost_delta_usd.p95:+.6f}, "
        f"total ${s.cost_delta_usd.total:+.6f})"
    )
    lines.append("")
    lines.append("## Substitutions")
    lines.append("")
    lines.append("Baseline (branch A):")
    if not s.baseline_substitutions:
        lines.append("- _(none — recorded run)_")
    else:
        for d in s.baseline_substitutions:
            lines.append(f"- `{d.get('type', '?')}` @ `{d.get('at_step', '?')}`")
    lines.append("")
    lines.append("Candidate (branch B):")
    if not s.candidate_substitutions:
        lines.append("- _(none)_")
    else:
        for d in s.candidate_substitutions:
            lines.append(f"- `{d.get('type', '?')}` @ `{d.get('at_step', '?')}`")
    lines.append("")
    lines.append("## Aggregate stats")
    lines.append("")
    lines.append("| metric | count | mean | p50 | p95 | min | max | total |")
    lines.append("| - | - | - | - | - | - | - | - |")

    def _row(label: str, d: DistStats, fmt: str = "{:+.6f}") -> str:
        return (
            f"| {label} | {d.count} | {fmt.format(d.mean)} | "
            f"{fmt.format(d.p50)} | {fmt.format(d.p95)} | "
            f"{fmt.format(d.min)} | {fmt.format(d.max)} | "
            f"{fmt.format(d.total)} |"
        )

    lines.append(_row("cost delta USD", s.cost_delta_usd))
    lines.append(_row("base cost USD", s.base_cost_usd, "{:.6f}"))
    lines.append(_row("cf cost USD", s.cf_cost_usd, "{:.6f}"))
    lines.append(_row("divergent step count", s.divergent_step_count, "{:.2f}"))
    lines.append(_row("divergent fraction", s.divergent_fraction, "{:.4f}"))
    lines.append(_row("cf cache-hit ratio", s.cf_cache_hit_ratio, "{:.4f}"))
    lines.append(_row("cf real executions", s.cf_real_executions, "{:.2f}"))
    lines.append("")

    lines.append("## Per-trace")
    lines.append("")
    lines.append(
        "| trace | steps | Δcost USD | divergent | decisions changed | cf cache% |"
    )
    lines.append("| - | - | - | - | - | - |")
    shown = 0
    for r in s.results:
        if shown >= max_rows:
            lines.append(
                f"| _… {len(s.results) - shown} more rows truncated_ |  |  |  |  |  |"
            )
            break
        lines.append(
            f"| `{os.path.basename(r.trace_path)}` | {r.step_count} | "
            f"{r.cost_delta_usd:+.6f} | {r.divergent_step_count} | "
            f"{r.decisions_changed} | {r.cf_cache_hit_ratio*100:.1f}% |"
        )
        shown += 1

    if s.failures:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        lines.append("| trace | phase | error | message |")
        lines.append("| - | - | - | - |")
        for f in s.failures:
            msg = f.message.replace("|", "\\|")[:100]
            lines.append(
                f"| `{os.path.basename(f.trace_path)}` | {f.phase} | "
                f"{f.error_class} | {msg} |"
            )

    lines.append("")
    return "\n".join(lines)


__all__ = [
    "DistStats",
    "SweepFailure",
    "SweepResult",
    "SweepReport",
    "sweep_traces",
    "render_sweep_report",
    "render_sweep_report_json",
]
