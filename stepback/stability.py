"""Statistical stability metrics for dirty-set replay predicates.

Given a trace and a substitution set, a predicate should fire
*consistently* on every replay — but non-deterministic executors (LLM
sampling, tool side-effects) can produce different results run-to-run.

This module detects *flaky* predicates by running the same replay
multiple times, computing a Wilson-score confidence interval over the
observed fire rate, and classifying the predicate as stable or flaky.
A :class:`FlakyPredicateWarning` is emitted automatically so callers
learn about instability without reading returned values.

Typical usage::

    from stepback.stability import measure_predicate_stability, FlakyClass

    result = measure_predicate_stability(
        trace, substitution_set, predicate=my_pred, executor=my_exec
    )
    print(f"fire_rate={result.fire_rate:.0%}  "
          f"CI=[{result.ci_low:.2f}, {result.ci_high:.2f}]  "
          f"class={result.flaky_class.name}")

    if result.flaky_class == FlakyClass.FLAKY:
        # Predicate is unreliable; minimisation results may be wrong.
        ...
"""
from __future__ import annotations

import enum
import statistics
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

from .replay import Executor, ReplayResult
from .substitutions import SubstitutionSet

if TYPE_CHECKING:  # pragma: no cover
    from .replay import Trace


__all__ = [
    "FlakyClass",
    "FlakyPredicateWarning",
    "StabilityConfig",
    "StabilityResult",
    "measure_predicate_stability",
]


# ---------------------------------------------------------------------------
# Public enum
# ---------------------------------------------------------------------------


class FlakyClass(enum.Enum):
    """Classification of a predicate's firing consistency.

    Attributes
    ----------
    STABLE_TRUE :
        The predicate fired on every run.  The behaviour is consistent
        and the predicate can be trusted for minimisation.
    STABLE_FALSE :
        The predicate never fired.  The behaviour is consistent (but
        may indicate a misconfigured predicate or substitution set).
    FLAKY :
        The predicate fired on some runs and not others.  Results from
        minimisation are unreliable; results should be treated with
        caution and the predicate should be investigated.
    """

    STABLE_TRUE = "stable_true"
    STABLE_FALSE = "stable_false"
    FLAKY = "flaky"


# ---------------------------------------------------------------------------
# Warning
# ---------------------------------------------------------------------------


class FlakyPredicateWarning(UserWarning):
    """Emitted when a predicate is classified as :attr:`FlakyClass.FLAKY`.

    The warning message includes the observed fire rate and the Wilson
    confidence interval so callers can make an informed decision about
    whether to trust downstream minimisation or bisect results.
    """


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class StabilityConfig:
    """Operational settings for :func:`measure_predicate_stability`.

    Attributes
    ----------
    runs :
        Number of independent replay passes to perform.  More runs give
        tighter confidence intervals.  Must be ≥ 1.
    ci_level :
        Confidence level for the Wilson score interval.  Must be in
        the open interval ``(0, 1)``.  Typical values are ``0.90``,
        ``0.95`` (default), or ``0.99``.
    warn :
        When ``True`` (default), emit a :class:`FlakyPredicateWarning`
        whenever the predicate is classified as :attr:`FlakyClass.FLAKY`.
    """

    runs: int = 20
    ci_level: float = 0.95
    warn: bool = True

    def __post_init__(self) -> None:
        if self.runs < 1:
            raise ValueError(f"runs must be >= 1, got {self.runs!r}")
        if not (0.0 < self.ci_level < 1.0):
            raise ValueError(
                f"ci_level must be in (0, 1), got {self.ci_level!r}"
            )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class StabilityResult:
    """Outcome of :func:`measure_predicate_stability`.

    Attributes
    ----------
    runs :
        Total number of replay passes performed.
    fires :
        Number of passes on which the predicate returned ``True``.
    fire_rate :
        ``fires / runs`` — the raw observed firing probability.
    ci_low :
        Lower bound of the Wilson score confidence interval for the
        true firing probability.
    ci_high :
        Upper bound of the Wilson score confidence interval for the
        true firing probability.
    ci_level :
        Confidence level used to compute the interval (e.g. ``0.95``).
    flaky_class :
        Stability classification: :attr:`FlakyClass.STABLE_TRUE` iff the
        predicate fired on *all* runs, :attr:`FlakyClass.STABLE_FALSE` iff
        it fired on *no* runs, and :attr:`FlakyClass.FLAKY` otherwise.
    """

    runs: int
    fires: int
    fire_rate: float
    ci_low: float
    ci_high: float
    ci_level: float
    flaky_class: FlakyClass


# ---------------------------------------------------------------------------
# Wilson score confidence interval (pure stdlib)
# ---------------------------------------------------------------------------


def _wilson_score_interval(k: int, n: int, level: float) -> tuple[float, float]:
    """Compute the Wilson score interval for *k* successes in *n* trials.

    Uses :class:`statistics.NormalDist` (stdlib, Python ≥ 3.8) to obtain the
    z-score so that arbitrary confidence levels are handled correctly.

    Parameters
    ----------
    k :
        Number of successes (0 ≤ k ≤ n).
    n :
        Total number of trials (n ≥ 1).
    level :
        Confidence level, e.g. 0.95 for a 95 % interval.

    Returns
    -------
    (low, high) :
        Lower and upper bounds of the Wilson score interval, clamped to
        ``[0.0, 1.0]``.
    """
    # One-sided quantile: P(Z ≤ z) = (1 + level) / 2.
    z = statistics.NormalDist().inv_cdf((1.0 + level) / 2.0)
    z2 = z * z
    p_hat = k / n
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * ((p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)) ** 0.5)
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return low, high


# ---------------------------------------------------------------------------
# Core measurement function
# ---------------------------------------------------------------------------


def measure_predicate_stability(
    trace: "Trace",
    substitutions: SubstitutionSet,
    predicate: Callable[[ReplayResult], bool],
    *,
    executor: Optional[Executor] = None,
    config: Optional[StabilityConfig] = None,
) -> StabilityResult:
    """Run repeated dirty replays and compute predicate firing statistics.

    The full *substitutions* set is applied on each of ``config.runs``
    independent replay passes.  The predicate is evaluated on each
    :class:`~stepback.replay.ReplayResult`; the fraction of passes where
    it returns ``True`` is the observed *fire rate*.

    A Wilson-score confidence interval is computed over the fire rate at
    the requested confidence level.  The predicate is then classified as:

    * :attr:`FlakyClass.STABLE_TRUE` — fired on every pass.
    * :attr:`FlakyClass.STABLE_FALSE` — never fired.
    * :attr:`FlakyClass.FLAKY` — fired on some passes but not all.

    If the predicate is classified as :attr:`FlakyClass.FLAKY` and
    ``config.warn`` is ``True``, a :class:`FlakyPredicateWarning` is
    emitted (``stacklevel=2``, pointing at the caller).

    .. note::
        Each replay pass uses a *fresh* :class:`~stepback.replay.Executor`
        (same callbacks as *executor* but with ``step_cache=None``) so that
        persistent caching does not mask stochastic behaviour.

    Parameters
    ----------
    trace :
        The :class:`~stepback.replay.Trace` to replay.
    substitutions :
        The candidate :class:`~stepback.substitutions.SubstitutionSet`
        applied identically on every pass.
    predicate :
        A callable that receives a :class:`~stepback.replay.ReplayResult`
        and returns ``True`` when the failure condition is present.
    executor :
        Callbacks for re-executing dirty steps.  If ``None``, a default
        no-op :class:`~stepback.replay.Executor` is used (suitable for
        fully cached or fallback traces).
    config :
        Operational settings.  If ``None``, :class:`StabilityConfig`
        defaults are used (20 runs, 95 % CI, warnings enabled).

    Returns
    -------
    StabilityResult
        Full statistics including fire rate, CI bounds, and classification.

    Raises
    ------
    ValueError
        If *config* contains invalid parameters (see
        :class:`StabilityConfig`).
    """
    if config is None:
        config = StabilityConfig()

    base_executor = executor if executor is not None else Executor()

    # Build a cache-free executor for each run so that step_cache entries
    # from one pass do not shadow stochastic outputs in later passes.
    cacheless_executor = Executor(
        llm=base_executor.llm,
        tool=base_executor.tool,
        router=base_executor.router,
        join=base_executor.join,
        fallback_recorded=base_executor.fallback_recorded,
        step_cache=None,
    )

    fires = 0
    for _ in range(config.runs):
        result = trace.run_replay(substitutions, cacheless_executor)
        if predicate(result):
            fires += 1

    n = config.runs
    fire_rate = fires / n
    ci_low, ci_high = _wilson_score_interval(fires, n, config.ci_level)

    if fires == n:
        flaky_class = FlakyClass.STABLE_TRUE
    elif fires == 0:
        flaky_class = FlakyClass.STABLE_FALSE
    else:
        flaky_class = FlakyClass.FLAKY

    if flaky_class is FlakyClass.FLAKY and config.warn:
        warnings.warn(
            f"Predicate is flaky: fired {fires}/{n} times "
            f"(fire_rate={fire_rate:.1%}, "
            f"{int(config.ci_level * 100)}% CI "
            f"[{ci_low:.3f}, {ci_high:.3f}]). "
            "Minimisation and bisect results may be unreliable.",
            FlakyPredicateWarning,
            stacklevel=2,
        )

    return StabilityResult(
        runs=n,
        fires=fires,
        fire_rate=fire_rate,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=config.ci_level,
        flaky_class=flaky_class,
    )
