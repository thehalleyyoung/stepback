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
"""
from __future__ import annotations

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
    Tuple,
)

from .replay import Executor, ReplayResult
from .substitutions import Substitution, SubstitutionSet

if TYPE_CHECKING:  # pragma: no cover
    from .replay import Trace


__all__ = [
    "BinaryHalvingStrategy",
    "BruteForceStrategy",
    "BudgetExhausted",
    "DDMinStrategy",
    "LinearShrinkStrategy",
    "MinimizationResult",
    "MinimizeOptions",
    "PredicateNotTriggered",
    "ShapleyAttributionStrategy",
    "Strategy",
    "attribute_substitutions",
    "ddmin_substitutions",
    "find_all_minimal",
    "minimize_substitutions",
]


# ============================================================ exceptions


class PredicateNotTriggered(ValueError):
    """The full substitution set does not flip the predicate."""


class BudgetExhausted(Exception):
    """A configured probe / time budget was exceeded mid-search.

    The best partial result available at the time of the abort is
    attached as :attr:`partial`. Callers should treat the partial
    minimal as a *superset* of the true minimal — search was cut
    short, so further reduction may have been possible.
    """

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
    """

    strategy: "Strategy" = field(default=None)  # type: ignore[assignment]
    probe_budget: Optional[int] = None
    time_budget_s: Optional[float] = None
    progress: Optional[Callable[[int, int], None]] = None
    excluded: FrozenSet[int] = frozenset()

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
        """Return predicate(replay(items)); memoised."""
        key = self._key(items)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self._check_budget()
        result = self._trace.run_replay(SubstitutionSet(items=list(items)), self._executor)
        verdict = bool(self._predicate(result))
        self._cache[key] = verdict
        self._results[key] = result
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


# ============================================================ back-compat


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
