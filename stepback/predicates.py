"""Tiny composable predicate helpers for :mod:`stepback.minimize`.

A *predicate* is any ``Callable[[ReplayResult], bool]`` passed to the
minimisation toolkit. These three helpers let you compose predicates
without writing lambdas:

    from stepback.predicates import all_of, any_of, not_

    p = all_of(
        lambda r: r.total_cost_usd > 0.10,
        any_of(
            lambda r: any("GB99" in str(s.outputs) for s in r.steps),
            not_(lambda r: r.steps[-1].outputs.get("status") == "ok"),
        ),
    )

A future round may add a safe predicate DSL string-parser; for now,
this module is intentionally a small set of combinators only.
"""
from __future__ import annotations

from typing import Any, Callable

__all__ = ["all_of", "any_of", "not_"]


def all_of(*predicates: Callable[[Any], bool]) -> Callable[[Any], bool]:
    """Return a predicate that fires iff *all* ``predicates`` fire.

    Short-circuits on the first ``False``. With zero predicates,
    returns a predicate that is always ``True`` (the empty conjunction).
    """
    preds = list(predicates)

    def _combined(value: Any) -> bool:
        for p in preds:
            if not p(value):
                return False
        return True

    return _combined


def any_of(*predicates: Callable[[Any], bool]) -> Callable[[Any], bool]:
    """Return a predicate that fires iff *any* ``predicates`` fire.

    Short-circuits on the first ``True``. With zero predicates,
    returns a predicate that is always ``False`` (the empty disjunction).
    """
    preds = list(predicates)

    def _combined(value: Any) -> bool:
        for p in preds:
            if p(value):
                return True
        return False

    return _combined


def not_(predicate: Callable[[Any], bool]) -> Callable[[Any], bool]:
    """Return the negation of ``predicate``."""

    def _negated(value: Any) -> bool:
        return not predicate(value)

    return _negated
