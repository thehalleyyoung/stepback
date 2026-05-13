"""Distributed bisect across a worker pool for large trace sets (Step 140).

This module parallelises :func:`~stepback.minimize.minimize_substitutions`
and :func:`~stepback.minimize.multi_objective_minimize` across a *corpus*
of traces using a :class:`concurrent.futures.ThreadPoolExecutor`.  Each
trace's bisect search is an independent unit of work; independent searches
are the natural parallelism axis for large-scale post-incident or regression
sweeps.

**Corpus parallelism**::

    from stepback.distributed_bisect import (
        DistributedBisectItem,
        DistributedBisectOptions,
        distributed_bisect,
    )
    from stepback.substitutions import PromptSubstitution, SubstitutionSet

    items = [
        DistributedBisectItem(
            trace=trace,
            substitutions=SubstitutionSet([PromptSubstitution(...)]),
            predicate=lambda r: r.any_step(lambda s: "error" in str(s.outputs)),
        )
        for trace in my_corpus
    ]
    summary = distributed_bisect(items, DistributedBisectOptions(workers=8))
    for r in summary.successful():
        print(r.item_index, len(r.result.minimal), "subs")

**Multi-objective corpus parallelism**::

    from stepback.distributed_bisect import distributed_multi_objective_bisect

    mo_summary = distributed_multi_objective_bisect(items, options=DistributedBisectOptions(workers=8))
    for r in mo_summary.successful():
        print(r.item_index, r.result.objectives, r.result.pareto_front)

**Thread-safety contract**

* Each worker thread creates its own :class:`~stepback.minimize._OracleCache`
  and runs its own :class:`~stepback.replay.Executor` instance.  Workers do
  not share mutable state.
* If you supply ``executor_factory``, a new :class:`~stepback.replay.Executor`
  is created per trace by calling the factory inside the worker thread.
  This is the recommended pattern when your executor holds per-thread
  resources (HTTP connections, SDK clients, etc.).
* If you supply a single ``executor``, all worker threads share it.  Only
  do this when the executor and all its callbacks are fully thread-safe.
* The ``predicate`` callable in each :class:`DistributedBisectItem` is
  invoked inside the worker thread for that item only; it is never called
  concurrently from multiple threads.
* :class:`~stepback.minimize.MinimizeOptions` ``per_trace_options`` is
  cloned (shallow copy via ``dataclasses.replace``) per item so per-trace
  state cannot leak between workers.

Public surface
--------------
* :class:`DistributedBisectItem` — one unit of work (trace + subs + predicate).
* :class:`DistributedBisectOptions` — concurrency and per-trace strategy config.
* :class:`DistributedBisectResult` — per-trace outcome (success or failure).
* :class:`DistributedBisectSummary` — aggregate over all items.
* :func:`distributed_bisect` — run corpus-parallel bisect.
* :class:`DistributedMultiObjectiveResult` — per-trace multi-objective outcome.
* :class:`DistributedMultiObjectiveSummary` — multi-objective aggregate.
* :func:`distributed_multi_objective_bisect` — run corpus-parallel
  multi-objective bisect.
"""
from __future__ import annotations

import dataclasses
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    Optional,
    Sequence,
)

from .minimize import (
    MinimizationResult,
    MinimizeOptions,
    MultiObjectiveMinimizationResult,
    ParetoEntry,
    multi_objective_minimize,
    minimize_substitutions,
)
from .substitutions import SubstitutionSet

if TYPE_CHECKING:  # pragma: no cover
    from .replay import Executor, ReplayResult, Trace


__all__ = [
    "DistributedBisectItem",
    "DistributedBisectOptions",
    "DistributedBisectResult",
    "DistributedBisectSummary",
    "DistributedMultiObjectiveResult",
    "DistributedMultiObjectiveSummary",
    "distributed_bisect",
    "distributed_multi_objective_bisect",
]


# ================================================================ input


@dataclass
class DistributedBisectItem:
    """One unit of work for a distributed bisect run.

    Attributes
    ----------
    trace :
        The :class:`~stepback.replay.Trace` to bisect.
    substitutions :
        Candidate :class:`~stepback.substitutions.SubstitutionSet` to
        minimise.  The full set must trigger *predicate*; if it does not,
        the corresponding :class:`DistributedBisectResult` will carry a
        ``"PredicateNotTriggered"`` error.
    predicate :
        ``(ReplayResult) -> bool`` — returns ``True`` when the failure is
        reproduced.  Invoked only within the dedicated worker thread for
        this item.
    """

    trace: "Trace"
    substitutions: SubstitutionSet
    predicate: Callable[["ReplayResult"], bool]


# ================================================================ options


@dataclass
class DistributedBisectOptions:
    """Concurrency and strategy options for distributed bisect.

    Attributes
    ----------
    workers :
        Maximum number of worker threads in the pool.  Each thread runs
        one trace's bisect independently.  Default ``4``.
    per_trace_options :
        :class:`~stepback.minimize.MinimizeOptions` applied to every trace.
        Cloned per item before dispatch.  If ``None``, a default
        :class:`~stepback.minimize.MinimizeOptions` is used.
    executor :
        A shared :class:`~stepback.replay.Executor` given to every trace.
        Only safe when the executor and its callbacks are fully thread-safe.
        Mutually exclusive with ``executor_factory``.
    executor_factory :
        ``() -> Executor`` called once per trace *inside its worker thread*
        to produce a per-trace executor.  Use this when your executor holds
        per-thread resources.  Takes precedence over ``executor`` when both
        are set.
    """

    workers: int = 4
    per_trace_options: Optional[MinimizeOptions] = None
    executor: Optional["Executor"] = None
    executor_factory: Optional[Callable[[], "Executor"]] = None

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError(f"workers must be >= 1; got {self.workers!r}")


# ================================================================ results


@dataclass
class DistributedBisectResult:
    """Outcome of one item in a distributed bisect run.

    Attributes
    ----------
    item_index :
        Zero-based index into the input ``items`` sequence.
    result :
        :class:`~stepback.minimize.MinimizationResult` on success;
        ``None`` on failure.
    error_class :
        Class name of the exception raised, or ``None`` on success.
    message :
        Exception message, or ``None`` on success.
    wall_time_s :
        Elapsed time for this item's bisect search (seconds).
    """

    item_index: int
    result: Optional[MinimizationResult]
    error_class: Optional[str]
    message: Optional[str]
    wall_time_s: float


@dataclass
class DistributedBisectSummary:
    """Aggregate outcome of a :func:`distributed_bisect` run.

    Attributes
    ----------
    results :
        Per-item results in input order (sorted by ``item_index``).
    success_count :
        Number of items whose bisect completed without error.
    failure_count :
        Number of items whose bisect raised an exception.
    total_probes :
        Sum of :attr:`~stepback.minimize.MinimizationResult.probes` across
        all successful items.
    total_cache_hits :
        Sum of :attr:`~stepback.minimize.MinimizationResult.cache_hits`
        across all successful items.
    elapsed_wall_time_s :
        Wall-clock time from pool submission to last completion (seconds).
        This is the *corpus-level* wall time accounting for parallelism,
        not the sum of per-item wall times.
    """

    results: List[DistributedBisectResult]
    success_count: int
    failure_count: int
    total_probes: int
    total_cache_hits: int
    elapsed_wall_time_s: float

    def successful(self) -> List[DistributedBisectResult]:
        """Items whose bisect completed without error."""
        return [r for r in self.results if r.error_class is None]

    def failed(self) -> List[DistributedBisectResult]:
        """Items whose bisect raised an exception."""
        return [r for r in self.results if r.error_class is not None]


# ================================================================ multi-objective results


@dataclass
class DistributedMultiObjectiveResult:
    """Outcome of one item in a distributed multi-objective bisect run.

    Attributes
    ----------
    item_index :
        Zero-based index into the input ``items`` sequence.
    result :
        :class:`~stepback.minimize.MultiObjectiveMinimizationResult` on
        success; ``None`` on failure.
    error_class :
        Class name of the exception raised, or ``None`` on success.
    message :
        Exception message, or ``None`` on success.
    wall_time_s :
        Elapsed time for this item's multi-objective bisect search (seconds).
    """

    item_index: int
    result: Optional[MultiObjectiveMinimizationResult]
    error_class: Optional[str]
    message: Optional[str]
    wall_time_s: float


@dataclass
class DistributedMultiObjectiveSummary:
    """Aggregate outcome of a :func:`distributed_multi_objective_bisect` run.

    Attributes
    ----------
    results :
        Per-item results in input order (sorted by ``item_index``).
    success_count :
        Number of items that completed without error.
    failure_count :
        Number of items that raised an exception.
    total_probes :
        Sum of probes across all successful items.
    total_cache_hits :
        Sum of cache hits across all successful items.
    elapsed_wall_time_s :
        Corpus-level wall-clock elapsed time (seconds).
    """

    results: List[DistributedMultiObjectiveResult]
    success_count: int
    failure_count: int
    total_probes: int
    total_cache_hits: int
    elapsed_wall_time_s: float

    def successful(self) -> List[DistributedMultiObjectiveResult]:
        """Items whose multi-objective bisect completed without error."""
        return [r for r in self.results if r.error_class is None]

    def failed(self) -> List[DistributedMultiObjectiveResult]:
        """Items whose multi-objective bisect raised an exception."""
        return [r for r in self.results if r.error_class is not None]

    def pareto_fronts(self) -> List[List[ParetoEntry]]:
        """Pareto front of each successful item, in input order.

        Items with no successful result contribute an empty list.
        """
        fronts: List[List[ParetoEntry]] = []
        per_index = {r.item_index: r for r in self.results}
        for idx in range(len(self.results)):
            r = per_index.get(idx)
            if r is not None and r.result is not None:
                fronts.append(list(r.result.pareto_front))
            else:
                fronts.append([])
        return fronts


# ================================================================ worker helpers


def _make_executor(options: DistributedBisectOptions) -> "Executor":
    """Create (or reuse) an executor for a single worker invocation."""
    from .replay import Executor as _Executor  # avoid circular at module level

    if options.executor_factory is not None:
        return options.executor_factory()
    if options.executor is not None:
        return options.executor
    return _Executor()


def _clone_options(options: DistributedBisectOptions) -> MinimizeOptions:
    """Return a per-trace clone of ``per_trace_options`` (or a fresh default)."""
    base = options.per_trace_options
    if base is None:
        return MinimizeOptions()
    # Shallow clone — strategy objects are stateless and safe to share.
    return dataclasses.replace(base)


def _run_one_bisect(
    item_index: int,
    item: DistributedBisectItem,
    options: DistributedBisectOptions,
) -> DistributedBisectResult:
    """Execute bisect for a single item (called inside a worker thread)."""
    t0 = time.monotonic()
    trace_opts = _clone_options(options)
    executor = _make_executor(options)
    try:
        result = minimize_substitutions(
            item.trace,
            item.substitutions,
            item.predicate,
            options=trace_opts,
            executor=executor,
        )
        return DistributedBisectResult(
            item_index=item_index,
            result=result,
            error_class=None,
            message=None,
            wall_time_s=time.monotonic() - t0,
        )
    except Exception as exc:
        return DistributedBisectResult(
            item_index=item_index,
            result=None,
            error_class=type(exc).__name__,
            message=str(exc),
            wall_time_s=time.monotonic() - t0,
        )


def _run_one_multi_objective(
    item_index: int,
    item: DistributedBisectItem,
    orderings: int,
    rng_seed: int,
    options: DistributedBisectOptions,
) -> DistributedMultiObjectiveResult:
    """Execute multi-objective bisect for a single item."""
    t0 = time.monotonic()
    base_opts = _clone_options(options)
    executor = _make_executor(options)
    try:
        result = multi_objective_minimize(
            item.trace,
            item.substitutions,
            item.predicate,
            orderings=orderings,
            rng_seed=rng_seed,
            options=base_opts,
            executor=executor,
        )
        return DistributedMultiObjectiveResult(
            item_index=item_index,
            result=result,
            error_class=None,
            message=None,
            wall_time_s=time.monotonic() - t0,
        )
    except Exception as exc:
        return DistributedMultiObjectiveResult(
            item_index=item_index,
            result=None,
            error_class=type(exc).__name__,
            message=str(exc),
            wall_time_s=time.monotonic() - t0,
        )


# ================================================================ public API


def distributed_bisect(
    items: Sequence[DistributedBisectItem],
    options: Optional[DistributedBisectOptions] = None,
) -> DistributedBisectSummary:
    """Run :func:`~stepback.minimize.minimize_substitutions` across a corpus.

    Each item is dispatched to a :class:`concurrent.futures.ThreadPoolExecutor`
    worker.  Results are collected in *input order* regardless of completion
    order.  Exceptions from individual items are caught and stored in
    :attr:`DistributedBisectResult.error_class` / ``message`` — a single
    failing trace never aborts the corpus run.

    Parameters
    ----------
    items :
        Sequence of :class:`DistributedBisectItem`, one per trace to bisect.
    options :
        Concurrency and strategy settings.  Defaults to
        ``DistributedBisectOptions()`` (4 workers, default ddmin strategy,
        per-thread ``Executor()``).

    Returns
    -------
    DistributedBisectSummary
        Aggregate result with per-item outcomes sorted by input index.

    Examples
    --------
    ::

        from stepback.distributed_bisect import (
            DistributedBisectItem, DistributedBisectOptions, distributed_bisect,
        )
        from stepback.substitutions import PromptSubstitution, SubstitutionSet

        items = [
            DistributedBisectItem(
                trace=t,
                substitutions=SubstitutionSet([PromptSubstitution(...)]),
                predicate=lambda r: any("fail" in str(s.outputs) for s in r.steps),
            )
            for t in my_traces
        ]
        summary = distributed_bisect(items, DistributedBisectOptions(workers=8))
        for r in summary.successful():
            print(r.item_index, len(r.result.minimal))
    """
    opts = options or DistributedBisectOptions()
    if not items:
        return DistributedBisectSummary(
            results=[],
            success_count=0,
            failure_count=0,
            total_probes=0,
            total_cache_hits=0,
            elapsed_wall_time_s=0.0,
        )

    corpus_t0 = time.monotonic()
    results_by_index: List[Optional[DistributedBisectResult]] = [None] * len(items)

    with ThreadPoolExecutor(max_workers=opts.workers) as pool:
        futures: List[Future[DistributedBisectResult]] = [
            pool.submit(_run_one_bisect, idx, item, opts)
            for idx, item in enumerate(items)
        ]
        for fut in as_completed(futures):
            r = fut.result()  # never raises — _run_one_bisect catches all
            results_by_index[r.item_index] = r

    elapsed = time.monotonic() - corpus_t0
    ordered: List[DistributedBisectResult] = [r for r in results_by_index if r is not None]

    success_count = sum(1 for r in ordered if r.error_class is None)
    failure_count = len(ordered) - success_count
    total_probes = sum(r.result.probes for r in ordered if r.result is not None)
    total_cache_hits = sum(r.result.cache_hits for r in ordered if r.result is not None)

    return DistributedBisectSummary(
        results=ordered,
        success_count=success_count,
        failure_count=failure_count,
        total_probes=total_probes,
        total_cache_hits=total_cache_hits,
        elapsed_wall_time_s=elapsed,
    )


def distributed_multi_objective_bisect(
    items: Sequence[DistributedBisectItem],
    *,
    orderings: int = 4,
    rng_seed: int = 0xC0DD,
    options: Optional[DistributedBisectOptions] = None,
) -> DistributedMultiObjectiveSummary:
    """Run :func:`~stepback.minimize.multi_objective_minimize` across a corpus.

    Parallelises Pareto-guided multi-objective ddmin across a corpus of
    traces.  Each trace's Pareto front is computed independently; the
    returned :class:`DistributedMultiObjectiveSummary` aggregates them.

    Parameters
    ----------
    items :
        Sequence of :class:`DistributedBisectItem`, one per trace.
    orderings :
        Number of random ddmin restarts per trace for Pareto diversity.
        Forwarded to :func:`~stepback.minimize.multi_objective_minimize`.
    rng_seed :
        RNG seed for ordering randomisation.
    options :
        Concurrency and per-trace options.

    Returns
    -------
    DistributedMultiObjectiveSummary
        Aggregate result with per-item outcomes sorted by input index.

    Examples
    --------
    ::

        from stepback.distributed_bisect import (
            DistributedBisectItem, DistributedBisectOptions,
            distributed_multi_objective_bisect,
        )

        summary = distributed_multi_objective_bisect(
            items, orderings=4, options=DistributedBisectOptions(workers=8)
        )
        for r in summary.successful():
            print(r.item_index, r.result.objectives, len(r.result.pareto_front))
    """
    opts = options or DistributedBisectOptions()
    if not items:
        return DistributedMultiObjectiveSummary(
            results=[],
            success_count=0,
            failure_count=0,
            total_probes=0,
            total_cache_hits=0,
            elapsed_wall_time_s=0.0,
        )

    corpus_t0 = time.monotonic()
    results_by_index: List[Optional[DistributedMultiObjectiveResult]] = [None] * len(items)

    with ThreadPoolExecutor(max_workers=opts.workers) as pool:
        futures: List[Future[DistributedMultiObjectiveResult]] = [
            pool.submit(_run_one_multi_objective, idx, item, orderings, rng_seed, opts)
            for idx, item in enumerate(items)
        ]
        for fut in as_completed(futures):
            r = fut.result()
            results_by_index[r.item_index] = r

    elapsed = time.monotonic() - corpus_t0
    ordered: List[DistributedMultiObjectiveResult] = [r for r in results_by_index if r is not None]

    success_count = sum(1 for r in ordered if r.error_class is None)
    failure_count = len(ordered) - success_count
    total_probes = sum(r.result.probes for r in ordered if r.result is not None)
    total_cache_hits = sum(r.result.cache_hits for r in ordered if r.result is not None)

    return DistributedMultiObjectiveSummary(
        results=ordered,
        success_count=success_count,
        failure_count=failure_count,
        total_probes=total_probes,
        total_cache_hits=total_cache_hits,
        elapsed_wall_time_s=elapsed,
    )
