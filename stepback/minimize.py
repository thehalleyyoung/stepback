"""Causal minimisation toolkit for substitution sets.

Given a :class:`~stepback.substitutions.SubstitutionSet` ``S`` whose
replay flips a user-supplied predicate (e.g. "the agent now refuses
the wire", "no step still emits PII", "total cost > $1"), this module
finds *which* substitutions in ``S`` are responsible.

The toolkit ships several **strategies** behind a single
``minimize_substitutions`` entry point:

* :class:`DDMinStrategy` — the classic Zeller & Hildebrandt
  delta-debugging algorithm (*Simplifying and Isolating
  Failure-Inducing Input*, IEEE TSE 2002). Default.
* :class:`LinearShrinkStrategy` — drop one element at a time.
  Cheaper than ddmin when blame is concentrated.
* :class:`BinaryHalvingStrategy` — recursive halving.
* :class:`BruteForceStrategy` — enumerate every non-empty subset
  in increasing size order. Optimal but exponential; capped at
  ``max_n`` items.
* :class:`ShapleyAttributionStrategy` — Shapley-value attribution
  with the predicate as a {0, 1}-valued game. Returns weights per
  substitution in addition to the 1-minimal subset.

Each probe is a :py:meth:`stepback.replay.Trace.run_replay` call;
results are memoised by canonical subset key (sorted item ids plus
the frozen ``excluded`` set), so strategies that revisit the same
subset don't re-pay the replay.

Operational guards via :class:`MinimizeOptions`:

* ``probe_budget`` — abort with :class:`BudgetExhausted` carrying
  the partial result.
* ``time_budget_s`` — wall-clock bound.
* ``progress`` — ``Callable[[probes_so_far, current_size], None]``.
* ``excluded`` — ids of substitutions to treat as forbidden (used
  by :func:`find_all_minimal` to enumerate disjoint witnesses).

**Step-level attribution** via :func:`attribute_steps` answers a
different question: *given a completed replay result, which steps
jointly cause a predicate to be True?*

:func:`attribute_steps` computes Shapley values for each step in
a replay, treating the predicate as a cooperative game over step
visibility. The Shapley value measures each step's average marginal
contribution across all orderings of the coalition game
v(S) = predicate(result masked to steps in S). This is
*observational attribution*: it identifies which steps, when visible,
are sufficient for the predicate to hold.

For predicates that check step-local properties (e.g.
``result.any_step(lambda s: s.error_class == 'Fail')``), observational
attribution equals causal attribution. For predicates that depend on
upstream causation between steps, combine with
:meth:`stepback.replay.Trace.bisect` to trace root causes further back.
"""
from __future__ import annotations

import copy
import itertools
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
)

from .replay import Executor, PartialExecutor, ReplayResult, StepView
from .substitutions import Substitution, SubstitutionSet

if TYPE_CHECKING:  # pragma: no cover
    from .replay import Trace


__all__ = [
    "BinaryHalvingStrategy",
    "BranchGroup",
    "BranchMinimizationResult",
    "BruteForceStrategy",
    "BudgetExhausted",
    "DDMinStrategy",
    "LinearShrinkStrategy",
    "MinimizationResult",
    "MinimizeOptions",
    "MultiObjectiveDDMinStrategy",
    "MultiObjectiveMinimizationResult",
    "ParetoEntry",
    "PredicateNotTriggered",
    "ShapleyAttributionStrategy",
    "StepAttributionResult",
    "Strategy",
    "TraceObjectives",
    "attribute_steps",
    "attribute_substitutions",
    "ddmin_substitutions",
    "extract_objectives",
    "find_all_minimal",
    "identify_branch_groups",
    "minimize_branches",
    "minimize_imported_trace",
    "minimize_substitutions",
    "multi_objective_minimize",
]


# ============================================================ exceptions


class PredicateNotTriggered(ValueError):
    """The full substitution set does not flip the predicate."""

    #: Canonical error code; see :mod:`stepback.errors` for details.
    code: str = "SB303"


class BudgetExhausted(Exception):
    """A configured probe / time budget was exceeded mid-search.

    The best partial result available at the time of the abort is
    attached as :attr:`partial`. Callers should treat the partial
    minimal as a *superset* of the true minimal — search was cut
    short, so further reduction may have been possible.
    """

    #: Canonical error code; see :mod:`stepback.errors` for details.
    code: str = "SB302"

    def __init__(self, message: str, partial: "MinimizationResult") -> None:
        super().__init__(message)
        self.partial = partial


# ============================================================ result


@dataclass
class MinimizationResult:
    """Outcome of a minimisation run.

    Attributes
    ----------
    minimal :
        The reduced subset that still triggers the predicate. For
        ddmin / linear / binary this is 1-minimal: removing any
        single element breaks the predicate. For brute-force this
        is globally minimal (smallest cardinality). For Shapley
        this is the set of items with strictly-positive weight.
    removed :
        Items present in the original input but proved unnecessary.
    probes :
        Number of distinct replays performed (cache hits do not
        count toward this).
    cache_hits :
        Number of times the oracle was asked about a subset whose
        result was already memoised.
    strategy_name :
        Name of the strategy that produced this result.
    weights :
        Optional per-item Shapley weights, keyed by the integer
        ``id(item)`` of each input substitution. Populated only by
        :class:`ShapleyAttributionStrategy`.
    final_result :
        The :class:`ReplayResult` of replaying ``minimal``.
    """

    minimal: List[Substitution]
    removed: List[Substitution] = field(default_factory=list)
    probes: int = 0
    cache_hits: int = 0
    strategy_name: str = ""
    weights: Optional[Dict[int, float]] = None
    final_result: Optional[ReplayResult] = None

    def as_set(self) -> SubstitutionSet:
        return SubstitutionSet(items=list(self.minimal))

    def weight_for(self, item: Substitution) -> float:
        """Look up the Shapley weight for ``item``; 0.0 if unknown."""
        if self.weights is None:
            return 0.0
        return float(self.weights.get(id(item), 0.0))


# ============================================================ options


@dataclass
class MinimizeOptions:
    """Strategy + operational controls for :func:`minimize_substitutions`.

    Attributes
    ----------
    strategy :
        The reduction strategy. Defaults to :class:`DDMinStrategy`.
    probe_budget :
        If set, raise :class:`BudgetExhausted` once this many
        replays have been performed (cache hits do not count).
    time_budget_s :
        If set, raise :class:`BudgetExhausted` once this many
        wall-clock seconds have elapsed since the search started.
    progress :
        Optional callback ``(probes_so_far, current_subset_size)``
        invoked after every actual replay (not on cache hits).
    excluded :
        Frozen set of ``id(item)`` values to treat as forbidden /
        unavailable. Used by :func:`find_all_minimal` to enumerate
        disjoint witnesses.
    skip_unavailable_executors :
        When ``True``, probes that raise
        :class:`~stepback.replay.UnavailableExecutorError` are treated as
        ``False`` (predicate not triggered) rather than propagating the
        exception.  This enables minimisation over traces where only some
        steps can be re-executed — substitutions that would dirty an
        unavailable step are effectively considered "not responsible".
        Intended for use with :class:`~stepback.replay.PartialExecutor`
        and :func:`minimize_imported_trace`.
    """

    strategy: "Strategy" = field(default=None)  # type: ignore[assignment]
    probe_budget: Optional[int] = None
    time_budget_s: Optional[float] = None
    progress: Optional[Callable[[int, int], None]] = None
    excluded: FrozenSet[int] = frozenset()
    skip_unavailable_executors: bool = False

    def __post_init__(self) -> None:
        if self.strategy is None:
            self.strategy = DDMinStrategy()


# ============================================================ oracle cache


class _OracleCache:
    """Wraps the predicate-on-replay closure with memoisation + budgets.

    The cache key is ``(tuple(sorted(id(item) for item in subset)),
    frozenset(excluded))``. Memoising by ``id`` is intentional and
    sound here: the substitution objects are immutable and live for
    the lifetime of the search.
    """

    def __init__(
        self,
        trace: "Trace",
        predicate: Callable[[ReplayResult], bool],
        executor: Executor,
        options: MinimizeOptions,
    ) -> None:
        self._trace = trace
        self._predicate = predicate
        self._executor = executor
        self._options = options
        self._cache: Dict[Tuple[Tuple[int, ...], FrozenSet[int]], bool] = {}
        self._results: Dict[Tuple[Tuple[int, ...], FrozenSet[int]], ReplayResult] = {}
        self.probes: int = 0
        self.cache_hits: int = 0
        self._t0 = time.monotonic()

    # -- key construction ----------------------------------------------
    def _key(self, items: List[Substitution]) -> Tuple[Tuple[int, ...], FrozenSet[int]]:
        return (tuple(sorted(id(x) for x in items)), self._options.excluded)

    # -- budget enforcement --------------------------------------------
    def _check_budget(self) -> None:
        if (
            self._options.probe_budget is not None
            and self.probes >= self._options.probe_budget
        ):
            raise _BudgetSentinel("probe budget exhausted")
        if self._options.time_budget_s is not None:
            if time.monotonic() - self._t0 >= self._options.time_budget_s:
                raise _BudgetSentinel("time budget exhausted")

    # -- public --------------------------------------------------------
    def evaluate(self, items: List[Substitution]) -> bool:
        """Return predicate(replay(items)); memoised.

        When :attr:`MinimizeOptions.skip_unavailable_executors` is ``True``,
        a :class:`~stepback.replay.UnavailableExecutorError` raised during
        replay is caught and treated as ``False`` (predicate not triggered).
        """
        from .replay import UnavailableExecutorError  # local import avoids circular
        key = self._key(items)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self._check_budget()
        try:
            result = self._trace.run_replay(SubstitutionSet(items=list(items)), self._executor)
            verdict = bool(self._predicate(result))
            self._results[key] = result
        except UnavailableExecutorError:
            if self._options.skip_unavailable_executors:
                verdict = False
            else:
                raise
        self._cache[key] = verdict
        self.probes += 1
        if self._options.progress is not None:
            try:
                self._options.progress(self.probes, len(items))
            except Exception:
                # Progress callback bugs must not derail minimisation.
                pass
        return verdict

    def replay_of(self, items: List[Substitution]) -> ReplayResult:
        """Return the (cached if possible) ReplayResult of ``items``."""
        key = self._key(items)
        if key in self._results:
            return self._results[key]
        # Not cached — run a fresh replay (also memoise).
        self.evaluate(items)
        return self._results[key]


class _BudgetSentinel(Exception):
    """Internal: raised by the cache to short-circuit a strategy."""


# ============================================================ strategies


class Strategy(ABC):
    """Abstract base for a minimisation algorithm.

    Implementations get an ``oracle: Callable[[List[Substitution]],
    bool]`` (the cache-wrapped predicate-on-replay) and the original
    item list, and return ``(minimal, removed, weights_or_None)``.
    """

    name: ClassVar[str] = ""

    @abstractmethod
    def run(
        self,
        items: List[Substitution],
        oracle: Callable[[List[Substitution]], bool],
    ) -> Tuple[List[Substitution], List[Substitution], Optional[Dict[int, float]]]:
        ...


class DDMinStrategy(Strategy):
    """Zeller-Hildebrandt 1-minimal delta debugging."""

    name: ClassVar[str] = "ddmin"

    def run(self, items, oracle):
        current = list(items)
        n = 2
        while len(current) >= 2:
            chunk_size = max(1, len(current) // n)
            partitions: List[List[Substitution]] = [
                current[i:i + chunk_size] for i in range(0, len(current), chunk_size)
            ]
            reduced = False
            for part in partitions:
                if oracle(part):
                    current = part
                    n = 2
                    reduced = True
                    break
            if reduced:
                continue
            for part in partitions:
                comp = [x for x in current if x not in part]
                if not comp:
                    continue
                if oracle(comp):
                    current = comp
                    n = max(n - 1, 2)
                    reduced = True
                    break
            if reduced:
                continue
            if n >= len(current):
                break
            n = min(2 * n, len(current))
        removed = [x for x in items if x not in current]
        return list(current), removed, None


class LinearShrinkStrategy(Strategy):
    """Drop one element at a time in input order; n+1 probes worst case."""

    name: ClassVar[str] = "linear"

    def run(self, items, oracle):
        current = list(items)
        i = 0
        while i < len(current):
            candidate = current[:i] + current[i + 1:]
            if candidate and oracle(candidate):
                current = candidate
                # don't advance i: the new current[i] hasn't been tried
            else:
                i += 1
        removed = [x for x in items if x not in current]
        return current, removed, None


class BinaryHalvingStrategy(Strategy):
    """Recursive halving: try each half alone, then both, shrinking each."""

    name: ClassVar[str] = "binary"

    def run(self, items, oracle):
        current = list(items)

        def shrink(xs: List[Substitution]) -> List[Substitution]:
            if len(xs) <= 1:
                return list(xs)
            mid = len(xs) // 2
            left, right = xs[:mid], xs[mid:]
            if oracle(left):
                return shrink(left)
            if oracle(right):
                return shrink(right)
            # Both halves jointly required — shrink each via linear
            # drop within the joint set.
            shrunk_left = self._linear_within(left, right, oracle)
            shrunk_right = self._linear_within(right, shrunk_left, oracle)
            return shrunk_left + shrunk_right

        current = shrink(current)
        removed = [x for x in items if x not in current]
        return current, removed, None

    @staticmethod
    def _linear_within(
        block: List[Substitution],
        other: List[Substitution],
        oracle: Callable[[List[Substitution]], bool],
    ) -> List[Substitution]:
        kept = list(block)
        i = 0
        while i < len(kept):
            candidate = kept[:i] + kept[i + 1:]
            if oracle(candidate + other):
                kept = candidate
            else:
                i += 1
        return kept


class BruteForceStrategy(Strategy):
    """Enumerate non-empty subsets in increasing size; return smallest."""

    name: ClassVar[str] = "brute"

    def __init__(self, max_n: int = 8) -> None:
        self.max_n = int(max_n)

    def run(self, items, oracle):
        if len(items) > self.max_n:
            raise ValueError(
                f"BruteForceStrategy refuses to enumerate {len(items)} items "
                f"(max_n={self.max_n}); 2^n explosion. Raise max_n explicitly "
                f"or pick a different strategy."
            )
        n = len(items)
        for size in range(1, n + 1):
            for combo_idx in itertools.combinations(range(n), size):
                subset = [items[i] for i in combo_idx]
                if oracle(subset):
                    removed = [items[i] for i in range(n) if i not in combo_idx]
                    return subset, removed, None
        # No non-empty subset triggers (shouldn't reach here — full set
        # was already proven to trigger by the orchestrator).
        return list(items), [], None


class ShapleyAttributionStrategy(Strategy):
    """Shapley-value attribution with predicate as {0,1}-game.

    For ``n <= 6`` runs the exact 2^n enumeration. For larger ``n``
    draws ``permutations`` random orderings (default
    ``min(64, 8*n)``) and averages marginal contributions; this is
    Strumbelj-Kononenko / SHAP's permutation estimator.

    The "minimal" returned is the set of items with strictly positive
    weight. Decoy items have weight 0.0; jointly required items split
    weight 1/k each.
    """

    name: ClassVar[str] = "shapley"

    def __init__(
        self,
        permutations: Optional[int] = None,
        rng_seed: int = 0xC0DE,
    ) -> None:
        self.permutations = permutations
        self.rng_seed = int(rng_seed)

    def run(self, items, oracle):
        n = len(items)
        weights: Dict[int, float] = {id(x): 0.0 for x in items}
        if n == 0:
            return [], [], weights

        if n <= 6:
            self._exact(items, oracle, weights)
        else:
            perms = self.permutations if self.permutations is not None else min(64, 8 * n)
            self._sampled(items, oracle, weights, perms)

        eps = 1e-9
        minimal = [x for x in items if weights[id(x)] > eps]
        removed = [x for x in items if weights[id(x)] <= eps]
        return minimal, removed, weights

    @staticmethod
    def _exact(items, oracle, weights):
        n = len(items)
        # Enumerate every coalition; cache value via the oracle.
        from math import comb, factorial
        nfact = factorial(n)
        for k, item in enumerate(items):
            others = [items[j] for j in range(n) if j != k]
            total = 0.0
            for size in range(0, n):  # |S| from 0..n-1
                weight = factorial(size) * factorial(n - size - 1) / nfact
                for combo in itertools.combinations(others, size):
                    s_no = list(combo)
                    s_with = s_no + [item]
                    v_no = 1.0 if (s_no and oracle(s_no)) else 0.0
                    v_with = 1.0 if oracle(s_with) else 0.0
                    total += weight * (v_with - v_no)
            weights[id(item)] = total

    def _sampled(self, items, oracle, weights, permutations):
        n = len(items)
        rng = random.Random(self.rng_seed)
        marginals: Dict[int, List[float]] = {id(x): [] for x in items}
        order = list(range(n))
        for _ in range(permutations):
            rng.shuffle(order)
            prefix: List[Substitution] = []
            v_prev = 0.0  # v(empty) = 0 by definition
            for idx in order:
                item = items[idx]
                next_prefix = prefix + [item]
                v_now = 1.0 if oracle(next_prefix) else 0.0
                marginals[id(item)].append(v_now - v_prev)
                prefix = next_prefix
                v_prev = v_now
        for item_id, vals in marginals.items():
            weights[item_id] = sum(vals) / len(vals) if vals else 0.0


# ============================================================ orchestrator


def minimize_substitutions(
    trace: "Trace",
    substitutions: SubstitutionSet,
    predicate: Callable[[ReplayResult], bool],
    *,
    options: Optional[MinimizeOptions] = None,
    executor: Optional[Executor] = None,
) -> MinimizationResult:
    """Run a strategy-pluggable causal minimisation against ``trace``.

    The function:

    1. Asserts the predicate fires under the *full* set (else raises
       :class:`PredicateNotTriggered`).
    2. Asserts the predicate does NOT fire under the *empty* set —
       otherwise the recorded trace already exhibits the predicate.
    3. Delegates to ``options.strategy.run`` with a memoising oracle.
    4. Runs a final replay over the reduced subset.

    On budget exhaustion mid-search, raises :class:`BudgetExhausted`
    with the best partial result attached.
    """
    options = options or MinimizeOptions()
    executor = executor or Executor()
    items: List[Substitution] = [
        x for x in substitutions.items if id(x) not in options.excluded
    ]

    cache = _OracleCache(trace, predicate, executor, options)

    def oracle(subset: List[Substitution]) -> bool:
        return cache.evaluate(subset)

    # Pre-flight: full set must trigger.
    try:
        if not oracle(items):
            raise PredicateNotTriggered(
                "predicate does not fire under the full substitution set; "
                "minimisation requires a triggering input"
            )
        # Empty must not trigger (else "zero subs are responsible").
        if oracle([]):
            empty_replay = cache.replay_of([])
            return MinimizationResult(
                minimal=[],
                removed=list(items),
                probes=cache.probes,
                cache_hits=cache.cache_hits,
                strategy_name=options.strategy.name,
                weights=None,
                final_result=empty_replay,
            )

        minimal, removed, weights = options.strategy.run(items, oracle)
    except _BudgetSentinel as exc:
        partial = MinimizationResult(
            minimal=list(items),
            removed=[],
            probes=cache.probes,
            cache_hits=cache.cache_hits,
            strategy_name=options.strategy.name,
            weights=None,
            final_result=None,
        )
        raise BudgetExhausted(str(exc), partial) from None

    # Final replay over the reduced subset.
    try:
        final = cache.replay_of(minimal) if minimal else cache.replay_of([])
    except _BudgetSentinel:
        final = None

    return MinimizationResult(
        minimal=list(minimal),
        removed=list(removed),
        probes=cache.probes,
        cache_hits=cache.cache_hits,
        strategy_name=options.strategy.name,
        weights=weights,
        final_result=final,
    )


def find_all_minimal(
    trace: "Trace",
    substitutions: SubstitutionSet,
    predicate: Callable[[ReplayResult], bool],
    *,
    max_witnesses: int = 8,
    options: Optional[MinimizeOptions] = None,
    executor: Optional[Executor] = None,
) -> List[MinimizationResult]:
    """Enumerate up to ``max_witnesses`` disjoint minimal subsets.

    Iteratively runs :func:`minimize_substitutions`, then forbids the
    members of the returned witness via ``options.excluded`` and
    re-runs on the remaining items. Stops when the remaining set no
    longer triggers the predicate or when ``max_witnesses`` is hit.

    The returned witnesses are pairwise disjoint (no item appears in
    more than one). Note this is *not* a guarantee of enumerating all
    distinct minimal subsets — it enumerates "k disjoint witnesses
    found within budget", which is the operationally useful answer.
    """
    base_options = options or MinimizeOptions()
    excluded: FrozenSet[int] = frozenset(base_options.excluded)
    witnesses: List[MinimizationResult] = []

    for _ in range(max_witnesses):
        # Fresh options each iteration — strategy and other knobs are
        # shared, but excluded grows.
        opts = MinimizeOptions(
            strategy=base_options.strategy,
            probe_budget=base_options.probe_budget,
            time_budget_s=base_options.time_budget_s,
            progress=base_options.progress,
            excluded=excluded,
        )
        try:
            r = minimize_substitutions(
                trace, substitutions, predicate, options=opts, executor=executor,
            )
        except PredicateNotTriggered:
            break
        if not r.minimal:
            # Empty witness — recorded trace already triggers; nothing to enumerate.
            if not witnesses:
                witnesses.append(r)
            break
        witnesses.append(r)
        new_excluded = excluded | frozenset(id(x) for x in r.minimal)
        if new_excluded == excluded:
            break
        excluded = new_excluded
    return witnesses


def attribute_substitutions(
    trace: "Trace",
    substitutions: SubstitutionSet,
    predicate: Callable[[ReplayResult], bool],
    *,
    executor: Optional[Executor] = None,
    permutations: Optional[int] = None,
) -> MinimizationResult:
    """Convenience wrapper around :class:`ShapleyAttributionStrategy`.

    Returns a :class:`MinimizationResult` whose ``weights`` field
    contains the Shapley value of each input substitution under the
    predicate-as-game.
    """
    strat = ShapleyAttributionStrategy(permutations=permutations)
    return minimize_substitutions(
        trace, substitutions, predicate,
        options=MinimizeOptions(strategy=strat),
        executor=executor,
    )


# ============================================================ step attribution


@dataclass
class StepAttributionResult:
    """Shapley-value attribution per *step* for a failure predicate.

    Answers: which steps in a replay result jointly cause the predicate
    to be ``True``?

    This is **observational attribution**: the Shapley value of a step
    measures its average marginal contribution across all orderings of
    the coalition game ``v(S) = predicate(result masked to steps in S)``.
    A masked result is a :class:`~stepback.replay.ReplayResult` containing
    only the steps whose ``step_id`` is in the coalition ``S``; aggregate
    fields (``total_cost_usd``, ``dirty_count``, etc.) are recomputed from
    those visible steps.

    For predicates that check step-local properties (e.g.
    ``result.any_step(lambda s: s.error_class == 'Fail')``), observational
    attribution equals causal attribution.  For predicates that depend on
    upstream causation between steps, combine with
    :meth:`stepback.replay.Trace.bisect` to trace root causes further back.

    Attributes
    ----------
    steps :
        All steps present in the original replay result, in recorded order.
    weights :
        Maps ``step_id`` to its Shapley value (a float in ``[-1, 1]``).
        Steps with strictly positive weight contributed to the failure.
        Steps with zero weight are irrelevant under the predicate.
        Negative weights are possible for non-monotone predicates.
    strategy_name :
        Always ``"step_shapley"``; present for consistency with
        :class:`MinimizationResult`.
    probes :
        Number of distinct coalition evaluations performed (after
        memoization).
    """

    steps: List[StepView]
    weights: Dict[str, float]
    probes: int
    strategy_name: str = "step_shapley"

    def weight_for(self, step_id: str) -> float:
        """Return the Shapley weight for ``step_id`` (0.0 if absent)."""
        return self.weights.get(step_id, 0.0)

    @property
    def contributing_steps(self) -> List[StepView]:
        """Steps with strictly positive Shapley weight (``weight > 1e-9``)."""
        eps = 1e-9
        return [s for s in self.steps if self.weights.get(s.step_id, 0.0) > eps]


def _masked_result(result: ReplayResult, step_ids: FrozenSet[str]) -> ReplayResult:
    """Return a copy of *result* restricted to steps whose id is in *step_ids*."""
    visible = [s for s in result.steps if s.step_id in step_ids]
    return ReplayResult(
        steps=visible,
        total_cost_usd=sum(s.cost_usd for s in visible),
        dirty_count=sum(1 for s in visible if s.dirty),
        cache_hit_count=sum(1 for s in visible if s.cache_hit),
        real_executions=sum(1 for s in visible if s.dirty and not s.cache_hit),
        provenance=result.provenance,
    )


def _exact_step_shapley(
    step_ids: List[str],
    v: Callable[[FrozenSet[str]], bool],
    weights: Dict[str, float],
) -> None:
    """Exact Shapley computation for n ≤ 8 steps (2^n coalition evaluations)."""
    from math import factorial

    n = len(step_ids)
    nfact = factorial(n)
    for k, sid in enumerate(step_ids):
        others = [step_ids[j] for j in range(n) if j != k]
        total = 0.0
        for size in range(0, n):
            w = factorial(size) * factorial(n - size - 1) / nfact
            for combo in itertools.combinations(others, size):
                s_no: FrozenSet[str] = frozenset(combo)
                s_with: FrozenSet[str] = s_no | {sid}
                v_no = 1.0 if v(s_no) else 0.0
                v_with = 1.0 if v(s_with) else 0.0
                total += w * (v_with - v_no)
        weights[sid] = total


def _sampled_step_shapley(
    step_ids: List[str],
    v: Callable[[FrozenSet[str]], bool],
    weights: Dict[str, float],
    permutations: int,
    rng_seed: int,
) -> None:
    """Permutation-estimator (Strumbelj-Kononenko / SHAP) for n > 8 steps."""
    n = len(step_ids)
    rng = random.Random(rng_seed)
    order = list(range(n))
    marginals: Dict[str, List[float]] = {sid: [] for sid in step_ids}
    for _ in range(permutations):
        rng.shuffle(order)
        prefix: FrozenSet[str] = frozenset()
        v_prev = 0.0
        for idx in order:
            sid = step_ids[idx]
            next_prefix = prefix | {sid}
            v_now = 1.0 if v(next_prefix) else 0.0
            marginals[sid].append(v_now - v_prev)
            prefix = next_prefix
            v_prev = v_now
    for sid, vals in marginals.items():
        weights[sid] = sum(vals) / len(vals) if vals else 0.0


def attribute_steps(
    trace: "Trace",
    predicate: Callable[[ReplayResult], bool],
    *,
    executor: Optional[Executor] = None,
    permutations: Optional[int] = None,
    rng_seed: int = 0xC0DE,
) -> StepAttributionResult:
    """Shapley-value attribution for steps that jointly cause a failure.

    Runs the trace once to obtain a :class:`~stepback.replay.ReplayResult`,
    then computes the Shapley value of each step under the coalition game
    ``v(S) = predicate(result masked to steps in S)``.

    Parameters
    ----------
    trace :
        A loaded :class:`~stepback.replay.Trace`.
    predicate :
        A callable ``(ReplayResult) -> bool``.  Must return ``True`` for
        the full replay result; raises :class:`PredicateNotTriggered`
        otherwise.
    executor :
        Executor for the replay.  Defaults to a no-op stub
        (all cache hits; suitable for trace-analysis predicates that do not
        need real LLM execution).
    permutations :
        Number of random orderings for the sampled estimator (used when
        the trace has more than 8 steps).  Defaults to
        ``min(64, 8 * n_steps)``.
    rng_seed :
        Seed for the sampled permutation estimator.

    Returns
    -------
    StepAttributionResult
        Contains per-step Shapley weights and convenience accessors.

    Raises
    ------
    PredicateNotTriggered
        If the predicate does not fire on the full replay result.
    ValueError
        If the trace contains duplicate step IDs.

    Notes
    -----
    For predicates that check step-local properties (e.g.
    ``result.any_step(lambda s: s.error_class == 'Fail')``), observational
    attribution equals causal attribution.  For predicates that depend on
    upstream causation between steps, combine with
    :meth:`~stepback.replay.Trace.bisect` to trace root causes further back.

    Coalition evaluation is memoized, so each distinct subset is evaluated
    at most once regardless of which algorithm path is taken.
    """
    executor = executor or Executor()
    result = trace.replay_forward(executor)
    steps = list(result.steps)

    step_ids = [s.step_id for s in steps]
    if len(step_ids) != len(set(step_ids)):
        raise ValueError(
            "trace contains duplicate step IDs; "
            "step attribution requires unique step IDs"
        )

    # Memoized coalition value function.
    _cache: Dict[FrozenSet[str], bool] = {}
    probes = 0

    def v(subset: FrozenSet[str]) -> bool:
        nonlocal probes
        if subset not in _cache:
            probes += 1
            masked = _masked_result(result, subset)
            _cache[subset] = predicate(masked)
        return _cache[subset]

    all_ids: FrozenSet[str] = frozenset(step_ids)

    # Pre-flight: full set must trigger.
    if not v(all_ids):
        raise PredicateNotTriggered(
            "predicate does not fire on the full replay result; "
            "step attribution requires a triggering predicate"
        )

    weights: Dict[str, float] = {sid: 0.0 for sid in step_ids}

    # If the predicate already fires on the empty coalition, no individual
    # step is necessary; return zero weights rather than running attribution.
    if v(frozenset()):
        return StepAttributionResult(
            steps=steps,
            weights=weights,
            probes=probes,
        )

    n = len(steps)
    if n <= 8:
        _exact_step_shapley(step_ids, v, weights)
    else:
        perms = permutations if permutations is not None else min(64, 8 * n)
        _sampled_step_shapley(step_ids, v, weights, perms, rng_seed)

    return StepAttributionResult(
        steps=steps,
        weights=weights,
        probes=probes,
    )


# ============================================================ branch minimization


@dataclass
class BranchGroup:
    """A parallel fan-out / fan-in group within a recorded trace.

    Attributes
    ----------
    open_step_id :
        Step ID of the ``parallel_branch_open`` step.
    join_step_id :
        Step ID of the matching ``parallel_branch_join`` step.
    branch_names :
        Branch names declared in the open step's outputs, one per branch.
    branch_tail_ids :
        The last step ID for each branch (inputs to the join).
    branch_step_ids :
        For each branch (by position), the ordered list of step IDs
        belonging exclusively to that branch.  Does not include the
        open or join steps.
    """

    open_step_id: str
    join_step_id: str
    branch_names: List[str]
    branch_tail_ids: List[str]
    branch_step_ids: List[List[str]]


@dataclass
class BranchMinimizationResult:
    """Outcome of branch-level trace minimization.

    Attributes
    ----------
    minimal_steps :
        The minimized list of recorded step dicts.  Steps belonging to
        dropped branches have been removed; the join step's
        ``branch_tails``, ``branch_tail_hashes``, and
        ``parent_step_ids`` have been updated accordingly.
    groups :
        All :class:`BranchGroup` objects identified in the original trace.
    kept_branch_indices :
        Per-group (keyed by ``open_step_id``): list of 0-based branch
        indices that were retained.
    dropped_branch_indices :
        Per-group (keyed by ``open_step_id``): list of 0-based branch
        indices that were removed.
    probes :
        Number of distinct replays performed (cache hits not counted).
    cache_hits :
        Number of oracle calls that re-used a memoized result.
    """

    minimal_steps: List[dict]
    groups: List[BranchGroup]
    kept_branch_indices: Dict[str, List[int]]
    dropped_branch_indices: Dict[str, List[int]]
    probes: int = 0
    cache_hits: int = 0


def _collect_branch_steps_for_tail(
    steps_by_id: Dict[str, dict],
    open_step_id: str,
    tail_step_id: str,
) -> List[str]:
    """Collect step IDs from ``tail_step_id`` back up to (but not including)
    ``open_step_id``, returned in forward execution order."""
    chain: List[str] = []
    current = tail_step_id
    while current and current != open_step_id:
        chain.append(current)
        step = steps_by_id.get(current)
        if step is None:
            break
        current = step.get("parent_step_id") or ""
    return list(reversed(chain))


def identify_branch_groups(steps: List[dict]) -> List[BranchGroup]:
    """Identify all parallel fan-out / fan-in groups in a recorded trace.

    Parameters
    ----------
    steps :
        Recorded step dicts from ``Trace.recorded_steps``, in topological
        (execution) order.

    Returns
    -------
    List[BranchGroup]
        One entry per ``parallel_branch_open`` / ``parallel_branch_join``
        pair found in ``steps``.  Groups are returned in the order their
        ``parallel_branch_open`` steps appear.
    """
    steps_by_id: Dict[str, dict] = {s["step_id"]: s for s in steps}
    groups: List[BranchGroup] = []

    for step in steps:
        if step["step_kind"] != "parallel_branch_open":
            continue
        open_id = step["step_id"]

        # Find the matching join step (has open_step_id in inputs).
        join_step: Optional[dict] = None
        for s in steps:
            if (
                s["step_kind"] == "parallel_branch_join"
                and s.get("inputs", {}).get("open_step_id") == open_id
            ):
                join_step = s
                break
        if join_step is None:
            continue

        branch_tail_ids: List[str] = list(
            join_step.get("parent_step_ids") or []
        )
        branch_names: List[str] = list(
            step.get("outputs", {}).get("branch_names", [])
        )

        branch_step_ids: List[List[str]] = []
        for tail_id in branch_tail_ids:
            branch_step_ids.append(
                _collect_branch_steps_for_tail(steps_by_id, open_id, tail_id)
            )

        groups.append(
            BranchGroup(
                open_step_id=open_id,
                join_step_id=join_step["step_id"],
                branch_names=branch_names,
                branch_tail_ids=branch_tail_ids,
                branch_step_ids=branch_step_ids,
            )
        )

    return groups


def _build_steps_dropping_branches(
    original_steps: List[dict],
    group: BranchGroup,
    indices_to_drop: List[int],
) -> List[dict]:
    """Return a new recorded-steps list with selected branches removed.

    ``indices_to_drop`` is a list of 0-based branch positions within
    ``group`` (indexing into ``group.branch_tail_ids``) to remove.
    The join step's ``branch_tails``, ``branch_tail_hashes``, and
    ``parent_step_ids`` are updated to reflect only the retained branches
    by locating each tail ID in the *current* join step's lists (which
    may already be shorter from prior calls).

    If *all* branches are dropped the join step's ``branch_tails`` and
    ``parent_step_ids`` become empty; the executor will produce an empty
    merged output when the join is re-run.
    """
    drop_step_ids: Set[str] = set()
    drop_tail_ids: Set[str] = set()
    for idx in indices_to_drop:
        for sid in group.branch_step_ids[idx]:
            drop_step_ids.add(sid)
        drop_tail_ids.add(group.branch_tail_ids[idx])

    new_steps: List[dict] = []
    for rec in original_steps:
        sid = rec["step_id"]
        if sid in drop_step_ids:
            continue
        if sid == group.join_step_id:
            rec = copy.deepcopy(rec)
            current_tails = list(rec["inputs"].get("branch_tails", []))
            current_hashes = list(rec["inputs"].get("branch_tail_hashes", []))
            # Remove each dropped tail by ID, preserving relative order.
            new_tails: List[str] = []
            new_hashes: List[str] = []
            for pos, tail_id in enumerate(current_tails):
                if tail_id not in drop_tail_ids:
                    new_tails.append(tail_id)
                    if pos < len(current_hashes):
                        new_hashes.append(current_hashes[pos])
            rec["inputs"]["branch_tails"] = new_tails
            rec["inputs"]["branch_tail_hashes"] = new_hashes
            parent_ids = list(rec.get("parent_step_ids") or [])
            rec["parent_step_ids"] = [p for p in parent_ids if p not in drop_tail_ids]
        new_steps.append(rec)

    return new_steps


def minimize_branches(
    trace: "Trace",
    predicate: Callable[[ReplayResult], bool],
    *,
    executor: Optional[Executor] = None,
    options: Optional[MinimizeOptions] = None,
) -> BranchMinimizationResult:
    """Drop independent branches from a trace while preserving the predicate.

    For each ``parallel_branch_open`` / ``parallel_branch_join`` group in
    the trace, this function attempts to drop individual branches.  A branch
    is *droppable* when removing its steps and adjusting the join to collect
    one fewer input still results in a replay that satisfies *predicate*.

    The function uses a greedy (linear) strategy: for each group, branches
    are tested for removal one at a time in order.  A branch that is
    successfully dropped remains absent for subsequent probes within the
    same group (accumulated dropping).

    When a branch is dropped the join step becomes dirty — its inputs have
    changed — and will be re-executed by *executor*.  The default
    :class:`~stepback.replay.Executor` produces ``{"branches": outs}`` for
    a ``parallel_branch_join`` with no explicit ``join`` callback, so an
    executor with a ``join`` callback is only needed when the application
    logic requires custom merge semantics.

    Parameters
    ----------
    trace :
        A loaded :class:`~stepback.replay.Trace` with parallel branches.
    predicate :
        ``(ReplayResult) -> bool``.  Checked after each probe replay.
        Must return ``True`` for the *original* trace before any branches
        are removed; otherwise :class:`PredicateNotTriggered` is raised.
    executor :
        Executor for re-running dirty steps.  Defaults to a no-op
        :class:`~stepback.replay.Executor`.  Must be able to handle any
        step kind that becomes dirty as a result of dropped branches
        (including ``parallel_branch_join`` and any downstream steps).
    options :
        Operational controls (``probe_budget``, ``time_budget_s``,
        ``progress``).  ``options.strategy`` is **ignored**; branch
        minimization always uses a greedy linear pass.

    Returns
    -------
    BranchMinimizationResult
        Contains the minimal step list, the groups found, and per-group
        kept / dropped branch indices.

    Raises
    ------
    PredicateNotTriggered
        If *predicate* does not fire on the original trace.
    BudgetExhausted
        If a configured ``probe_budget`` or ``time_budget_s`` is exhausted.
        The partial result (with branches dropped so far) is attached as
        ``exc.partial``.
    """
    from dataclasses import replace as _dc_replace

    executor = executor or Executor()
    opts = options or MinimizeOptions()
    groups = identify_branch_groups(trace.recorded_steps)

    probes = 0
    cache_hits = 0
    _memo: Dict[Tuple[Tuple[int, ...], ...], bool] = {}
    _t0 = time.monotonic()

    def _check_budget() -> None:
        if opts.probe_budget is not None and probes >= opts.probe_budget:
            raise _BudgetSentinel("probe budget exhausted")
        if opts.time_budget_s is not None and time.monotonic() - _t0 >= opts.time_budget_s:
            raise _BudgetSentinel("time budget exhausted")

    def _probe(current_steps: List[dict]) -> bool:
        """Replay current_steps and return predicate verdict; memoized."""
        nonlocal probes, cache_hits
        # Use a hashable key: tuple of step_ids in order.
        key: Tuple[str, ...] = tuple(s["step_id"] for s in current_steps)
        if key in _memo:
            cache_hits += 1
            return _memo[key]
        _check_budget()
        # Build a temporary Trace for replay.
        from .replay import Trace as _Trace
        tmp = _Trace(
            path=trace.path,
            header=trace.header,
            recorded_steps=current_steps,
        )
        result = tmp.run_replay(SubstitutionSet(), executor)
        verdict = bool(predicate(result))
        _memo[key] = verdict
        probes += 1
        if opts.progress is not None:
            try:
                opts.progress(probes, len(current_steps))
            except Exception:
                pass
        return verdict

    # Pre-flight: original trace must satisfy the predicate.
    try:
        if not _probe(trace.recorded_steps):
            raise PredicateNotTriggered(
                "predicate does not fire on the original trace; "
                "branch minimization requires a triggering predicate"
            )
    except _BudgetSentinel as exc:
        partial = BranchMinimizationResult(
            minimal_steps=list(trace.recorded_steps),
            groups=groups,
            kept_branch_indices={g.open_step_id: list(range(len(g.branch_tail_ids))) for g in groups},
            dropped_branch_indices={g.open_step_id: [] for g in groups},
            probes=probes,
            cache_hits=cache_hits,
        )
        raise BudgetExhausted(str(exc), partial) from None

    kept_indices: Dict[str, List[int]] = {
        g.open_step_id: list(range(len(g.branch_tail_ids))) for g in groups
    }
    dropped_indices: Dict[str, List[int]] = {
        g.open_step_id: [] for g in groups
    }

    # Current working set of steps (modified as branches are dropped).
    current_steps: List[dict] = list(trace.recorded_steps)

    try:
        for group in groups:
            if len(group.branch_tail_ids) == 0:
                continue
            # Greedy pass: try dropping each branch in turn.
            i = 0
            while i < len(kept_indices[group.open_step_id]):
                branch_pos = kept_indices[group.open_step_id][i]
                candidate = _build_steps_dropping_branches(
                    current_steps, group, [branch_pos]
                )
                if _probe(candidate):
                    # Branch is droppable: commit the removal.
                    current_steps = candidate
                    kept_indices[group.open_step_id].remove(branch_pos)
                    dropped_indices[group.open_step_id].append(branch_pos)
                    # Don't increment i: the list shrank, so the next item
                    # is now at index i.
                else:
                    i += 1

    except _BudgetSentinel as exc:
        partial = BranchMinimizationResult(
            minimal_steps=current_steps,
            groups=groups,
            kept_branch_indices=kept_indices,
            dropped_branch_indices=dropped_indices,
            probes=probes,
            cache_hits=cache_hits,
        )
        raise BudgetExhausted(str(exc), partial) from None

    return BranchMinimizationResult(
        minimal_steps=current_steps,
        groups=groups,
        kept_branch_indices=kept_indices,
        dropped_branch_indices=dropped_indices,
        probes=probes,
        cache_hits=cache_hits,
    )
def ddmin_substitutions(
    trace: "Trace",
    substitutions: SubstitutionSet,
    predicate: Callable[[ReplayResult], bool],
    *,
    executor: Optional[Executor] = None,
) -> MinimizationResult:
    """Run ddmin on ``substitutions`` (back-compat shim).

    Equivalent to ``minimize_substitutions(..., options=MinimizeOptions(
    strategy=DDMinStrategy()))``. Signature, return type, and exception
    behaviour are unchanged from stepback v0.x.
    """
    return minimize_substitutions(
        trace,
        substitutions,
        predicate,
        options=MinimizeOptions(strategy=DDMinStrategy()),
        executor=executor,
    )


# ============================================================ multi-objective


@dataclass(frozen=True)
class TraceObjectives:
    """Scalar objectives extracted from a :class:`~stepback.replay.ReplayResult`.

    Each field is a non-negative value to be *minimised*.

    Attributes
    ----------
    step_count :
        Total number of replay steps.
    llm_call_count :
        Number of steps whose ``kind`` is ``"llm_call"``.
    total_cost_usd :
        Total LLM cost charged by the replay.
    policy_violation_count :
        Number of steps where ``policy_blocked`` is ``True``.
    latency_s :
        Estimated wall-clock latency in seconds.  Currently always ``0.0``
        because per-step timing is not yet stored in the trace format.

        .. todo::
            Wire to step-level timing when the recorder stores
            ``elapsed_ms`` in each ``step_complete`` event.
    """

    step_count: int
    llm_call_count: int
    total_cost_usd: float
    policy_violation_count: int
    latency_s: float = 0.0

    def dominates(self, other: "TraceObjectives") -> bool:
        """Return ``True`` when *self* Pareto-dominates *other*.

        *self* dominates *other* iff it is at least as good on every
        objective and strictly better on at least one.
        """
        a = self.as_tuple()
        b = other.as_tuple()
        return all(ai <= bi for ai, bi in zip(a, b)) and any(ai < bi for ai, bi in zip(a, b))

    def as_tuple(self) -> Tuple[float, ...]:
        """All objectives as a comparable tuple (lower is better)."""
        return (
            float(self.step_count),
            float(self.llm_call_count),
            self.total_cost_usd,
            float(self.policy_violation_count),
            self.latency_s,
        )


def extract_objectives(result: ReplayResult) -> TraceObjectives:
    """Derive :class:`TraceObjectives` from a completed :class:`ReplayResult`.

    ``policy_blocked`` is read from :attr:`StepView.policy_blocked` when
    present; defaults to ``False`` for step objects that lack the attribute
    (e.g. older in-memory fixtures).
    """
    step_count = len(result.steps)
    llm_call_count = sum(1 for s in result.steps if s.kind == "llm_call")
    total_cost_usd = result.total_cost_usd
    policy_violation_count = sum(
        1 for s in result.steps if getattr(s, "policy_blocked", False)
    )
    return TraceObjectives(
        step_count=step_count,
        llm_call_count=llm_call_count,
        total_cost_usd=total_cost_usd,
        policy_violation_count=policy_violation_count,
    )


@dataclass
class ParetoEntry:
    """One member of a multi-objective Pareto front.

    Attributes
    ----------
    minimal :
        The 1-minimal substitution subset that produced this entry.
    objectives :
        Trace-level objectives extracted from the replay of ``minimal``.
    final_result :
        The :class:`ReplayResult` from replaying ``minimal``; ``None``
        if the replay was not yet materialised (e.g. budget was exhausted
        before the final replay could run).
    """

    minimal: List[Substitution]
    objectives: TraceObjectives
    final_result: Optional[ReplayResult] = None


@dataclass
class MultiObjectiveMinimizationResult:
    """Result of :func:`multi_objective_minimize`.

    The ``minimal`` / ``removed`` / ``probes`` / ``cache_hits`` /
    ``strategy_name`` / ``final_result`` fields match the contract of
    :class:`MinimizationResult` so callers can use both interchangeably.

    Additionally:

    Attributes
    ----------
    objectives :
        Trace-level objectives for the chosen best result.
    pareto_front :
        All non-dominated :class:`ParetoEntry` objects discovered
        across the ``orderings`` ddmin runs.  May contain one entry
        (when all orderings converge to the same 1-minimal subset) or
        several (when different orderings expose distinct cost-quality
        trade-offs).
    """

    minimal: List[Substitution]
    removed: List[Substitution] = field(default_factory=list)
    probes: int = 0
    cache_hits: int = 0
    strategy_name: str = "multi_objective_ddmin"
    weights: Optional[Dict[int, float]] = None
    final_result: Optional[ReplayResult] = None
    objectives: Optional[TraceObjectives] = None
    pareto_front: List[ParetoEntry] = field(default_factory=list)

    def as_set(self) -> SubstitutionSet:
        return SubstitutionSet(items=list(self.minimal))


class MultiObjectiveDDMinStrategy(Strategy):
    """Pareto-guided ddmin: run standard ddmin with several element orderings.

    After obtaining one or more 1-minimal subsets (via random restarts),
    the strategy returns the lexicographically smallest minimal set.
    The Pareto front is computed by the :func:`multi_objective_minimize`
    orchestrator which has access to the replay results.

    Parameters
    ----------
    orderings :
        Number of random element orderings to try in addition to the
        original input order (total runs = ``orderings``).  Extra runs
        find alternative 1-minimal sets whose trace objectives may be
        cheaper even if the substitution count is the same.
    rng_seed :
        Seed for the random ordering generator.
    """

    name: ClassVar[str] = "multi_objective_ddmin"

    def __init__(self, orderings: int = 4, rng_seed: int = 0xC0DD) -> None:
        self.orderings = max(1, int(orderings))
        self.rng_seed = int(rng_seed)

    def run(
        self,
        items: List[Substitution],
        oracle: Callable[[List[Substitution]], bool],
    ) -> Tuple[List[Substitution], List[Substitution], Optional[Dict[int, float]]]:
        rng = random.Random(self.rng_seed)
        base = DDMinStrategy()

        orderings_to_try: List[List[Substitution]] = [list(items)]
        for _ in range(self.orderings - 1):
            shuffled = list(items)
            rng.shuffle(shuffled)
            orderings_to_try.append(shuffled)

        seen: set = set()
        best: Optional[List[Substitution]] = None
        for ordering in orderings_to_try:
            m, _, _ = base.run(ordering, oracle)
            key = frozenset(id(x) for x in m)
            if key in seen:
                continue
            seen.add(key)
            if best is None or len(m) < len(best):
                best = m

        assert best is not None
        removed = [x for x in items if x not in best]
        return list(best), removed, None


def _compute_pareto_front(entries: List[ParetoEntry]) -> List[ParetoEntry]:
    """Return the non-dominated subset of ``entries`` (objectives are minimised)."""
    front: List[ParetoEntry] = []
    for candidate in entries:
        dominated = False
        for other in entries:
            if other is candidate:
                continue
            if other.objectives.dominates(candidate.objectives):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def multi_objective_minimize(
    trace: "Trace",
    substitutions: SubstitutionSet,
    predicate: Callable[[ReplayResult], bool],
    *,
    orderings: int = 4,
    rng_seed: int = 0xC0DD,
    options: Optional[MinimizeOptions] = None,
    executor: Optional[Executor] = None,
) -> MultiObjectiveMinimizationResult:
    """Multi-objective delta-debugging over substitutions **and** trace metrics.

    Extends the standard ddmin with a Pareto-front discovery phase that
    runs ddmin for ``orderings`` different element orderings.  Each run
    produces a 1-minimal triggering subset; the associated
    :class:`TraceObjectives` (step count, LLM calls, cost, policy
    violations) are extracted and the non-dominated entries form the
    returned :attr:`MultiObjectiveMinimizationResult.pareto_front`.

    The primary ``minimal`` in the result is the entry from the Pareto
    front with the *fewest substitutions* (ties broken by
    ``objectives.as_tuple()`` lexicographic order).

    Parameters
    ----------
    trace :
        The recorded :class:`~stepback.replay.Trace` to replay against.
    substitutions :
        The full candidate :class:`~stepback.substitutions.SubstitutionSet`.
    predicate :
        A callable that receives a :class:`~stepback.replay.ReplayResult`
        and returns ``True`` when the failure is reproduced.
    orderings :
        How many ddmin restarts to attempt with different element orderings.
        More restarts find more Pareto-front diversity at the cost of extra
        replays.  Default ``4``.
    rng_seed :
        RNG seed for the random orderings.
    options :
        Operational options (budget, progress callback, excluded items).
        If ``None``, a default :class:`MinimizeOptions` is used.  The
        ``strategy`` field of the supplied options is **ignored** — the
        function always uses :class:`MultiObjectiveDDMinStrategy`.
    executor :
        Replay executor; defaults to ``Executor()``.

    Returns
    -------
    MultiObjectiveMinimizationResult
        The best minimal subset plus the full Pareto front of discovered
        1-minimal subsets and their trace objectives.

    Raises
    ------
    PredicateNotTriggered
        If ``predicate`` does not fire on the full substitution set.
    BudgetExhausted
        If a ``probe_budget`` or ``time_budget_s`` in ``options`` is
        exhausted before the first complete ddmin run finishes.
    """
    base_opts = options or MinimizeOptions()
    mo_opts = MinimizeOptions(
        strategy=MultiObjectiveDDMinStrategy(orderings=orderings, rng_seed=rng_seed),
        probe_budget=base_opts.probe_budget,
        time_budget_s=base_opts.time_budget_s,
        progress=base_opts.progress,
        excluded=base_opts.excluded,
    )
    executor = executor or Executor()
    items: List[Substitution] = [
        x for x in substitutions.items if id(x) not in mo_opts.excluded
    ]

    cache = _OracleCache(trace, predicate, executor, mo_opts)

    def oracle(subset: List[Substitution]) -> bool:
        return cache.evaluate(subset)

    # Pre-flight: full set must trigger.
    try:
        if not oracle(items):
            raise PredicateNotTriggered(
                "predicate does not fire under the full substitution set; "
                "multi-objective minimisation requires a triggering input"
            )
        if oracle([]):
            empty_replay = cache.replay_of([])
            obj = extract_objectives(empty_replay)
            entry = ParetoEntry(minimal=[], objectives=obj, final_result=empty_replay)
            return MultiObjectiveMinimizationResult(
                minimal=[],
                removed=list(items),
                probes=cache.probes,
                cache_hits=cache.cache_hits,
                objectives=obj,
                pareto_front=[entry],
                final_result=empty_replay,
            )
    except _BudgetSentinel as exc:
        partial = MultiObjectiveMinimizationResult(
            minimal=list(items),
            removed=[],
            probes=cache.probes,
            cache_hits=cache.cache_hits,
        )
        raise BudgetExhausted(str(exc), partial) from None  # type: ignore[arg-type]

    # Run ddmin with each ordering; collect distinct 1-minimal subsets.
    rng = random.Random(rng_seed)
    orderings_to_try: List[List[Substitution]] = [list(items)]
    for _ in range(orderings - 1):
        shuffled = list(items)
        rng.shuffle(shuffled)
        orderings_to_try.append(shuffled)

    seen_keys: set = set()
    pareto_entries: List[ParetoEntry] = []

    for ordering in orderings_to_try:
        try:
            m, _, _ = DDMinStrategy().run(ordering, oracle)
        except _BudgetSentinel:
            break
        key = frozenset(id(x) for x in m)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        try:
            replay_result = cache.replay_of(m)
        except _BudgetSentinel:
            break
        obj = extract_objectives(replay_result)
        pareto_entries.append(ParetoEntry(minimal=m, objectives=obj, final_result=replay_result))

    if not pareto_entries:
        # Budget exhausted before any run completed.
        partial = MultiObjectiveMinimizationResult(
            minimal=list(items),
            removed=[],
            probes=cache.probes,
            cache_hits=cache.cache_hits,
        )
        raise BudgetExhausted("budget exhausted before any ddmin run completed", partial)  # type: ignore[arg-type]

    pareto_front = _compute_pareto_front(pareto_entries)

    # Pick best: fewest substitutions, then lexicographically best objectives.
    best = min(pareto_front, key=lambda e: (len(e.minimal),) + e.objectives.as_tuple())
    removed = [x for x in items if x not in best.minimal]

    return MultiObjectiveMinimizationResult(
        minimal=list(best.minimal),
        removed=removed,
        probes=cache.probes,
        cache_hits=cache.cache_hits,
        objectives=best.objectives,
        pareto_front=pareto_front,
        final_result=best.final_result,
    )


def minimize_imported_trace(
    trace: "Trace",
    substitutions: SubstitutionSet,
    predicate: Callable[[ReplayResult], bool],
    *,
    executor: Optional[Executor] = None,
    options: Optional[MinimizeOptions] = None,
) -> MinimizationResult:
    """Minimize substitutions for an imported trace with partial executor support.

    Designed for traces imported from external tools (LangSmith,
    OpenInference, etc.) where only some steps can be re-executed in the
    current environment.

    Behavior
    --------
    * If *executor* is a :class:`~stepback.replay.PartialExecutor`, dirty
      steps for unavailable tools/models raise
      :class:`~stepback.replay.UnavailableExecutorError`.  With
      ``options.skip_unavailable_executors=True`` (the default for this
      function), such probes are treated as ``False`` so minimisation
      continues over substitutions that CAN be replayed.
    * If *executor* is ``None``, a plain :class:`~stepback.replay.Executor`
      with ``fallback_recorded=True`` is used so that dirty steps without a
      registered callback fall back to their recorded outputs.  Predicate
      evaluation is always possible but may be approximate for dirty steps
      that used stale recorded outputs.
    * If *executor* is a plain :class:`~stepback.replay.Executor` with
      ``fallback_recorded=True``, the same approximate-replay mode applies.

    Use :func:`~stepback.replay.audit_executor_requirements` before calling
    this function to understand which steps will use live re-execution vs.
    recorded fallback outputs.

    Parameters
    ----------
    trace :
        The imported :class:`~stepback.replay.Trace` to minimise against.
    substitutions :
        The full :class:`~stepback.substitutions.SubstitutionSet` to reduce.
    predicate :
        A function ``(ReplayResult) -> bool`` that returns ``True`` when the
        failure condition is present.
    executor :
        Optional executor.  Defaults to
        ``Executor(fallback_recorded=True)`` so the function always
        produces a result even when the trace cannot be fully replayed.
    options :
        Strategy and budget options.
        ``skip_unavailable_executors`` defaults to ``True`` when *executor*
        is a :class:`~stepback.replay.PartialExecutor`.

    Returns
    -------
    MinimizationResult
        Same as :func:`minimize_substitutions`.

    Raises
    ------
    PredicateNotTriggered
        If the full substitution set does not trigger the predicate.
    BudgetExhausted
        If a probe/time budget was exceeded.

    Example
    -------
    ::

        from stepback import replay
        from stepback.substitutions import PromptSubstitution, SubstitutionSet
        from stepback.replay import PartialExecutor, audit_executor_requirements
        from stepback.minimize import minimize_imported_trace

        trace = replay("imported_run.sb")
        # Audit what executors each step needs.
        reqs = audit_executor_requirements(trace)
        unavailable = [r for r in reqs if not r.executor_available]

        subs = SubstitutionSet([PromptSubstitution("step-3", "New prompt")])
        result = minimize_imported_trace(
            trace, subs, lambda r: r.any_step(lambda s: "error" in str(s.outputs))
        )
    """
    # Choose a sensible default executor for imported traces.
    if executor is None:
        eff_executor: Executor = Executor(fallback_recorded=True)
    else:
        eff_executor = executor

    # Determine skip_unavailable_executors default.
    base = options or MinimizeOptions()
    auto_skip = isinstance(eff_executor, PartialExecutor)
    eff_options = MinimizeOptions(
        strategy=base.strategy,
        probe_budget=base.probe_budget,
        time_budget_s=base.time_budget_s,
        progress=base.progress,
        excluded=base.excluded,
        skip_unavailable_executors=(
            base.skip_unavailable_executors or auto_skip
        ),
    )

    return minimize_substitutions(
        trace,
        substitutions,
        predicate,
        options=eff_options,
        executor=eff_executor,
    )
