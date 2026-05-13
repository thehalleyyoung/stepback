"""In-process replay utilities for unit tests and local debugging.

This module is the canonical in-process entry-point for testing and
debugging stepback traces without running a proxy or an LLM provider.
It covers three complementary concerns:

1. **CaptureExecutor** — a drop-in :class:`~stepback.replay.Executor`
   subclass that records every successful executor callback invocation.
   Use it in tests to assert which steps were re-executed and with what
   inputs.

2. **Assertion helpers** — ``assert_*`` functions over
   :class:`~stepback.replay.ReplayResult` that raise :class:`AssertionError`
   with compact, actionable messages.

3. **FallbackExecutor** — a convenience alias for
   ``Executor(fallback_recorded=True)`` for *local debugging* sessions
   where you want to see which steps *would* be dirty without providing
   real executor callbacks.

When to use each path:

* **Unit tests** — use :class:`CaptureExecutor` + assertion helpers.
  Record the trace once, substitute, replay, assert.

* **Local debugging / CLI** — use :class:`FallbackExecutor` (or the
  ``stepback diff`` / ``stepback replay`` commands which wire it
  automatically).  The replayed outputs are the recorded ones so
  no LLM key is required.

* **Production / CI with real providers** — pass a real
  :class:`~stepback.replay.Executor` with ``llm=``, ``tool=``, etc.

* **Remote replay** — see ``stepback-proxy`` (Step 71), which
  submits traces and substitutions over gRPC and streams back replay
  events.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ..replay import Executor, MissingExecutor, ReplayResult, StepView


# ---------------------------------------------------------------- CaptureExecutor


@dataclass
class CapturedCall:
    """One successful executor callback invocation.

    *Successful* means :meth:`~stepback.replay.Executor.execute` returned
    without raising.  Failed calls (e.g. :class:`~stepback.replay.MissingExecutor`)
    are not captured — they propagate as exceptions.

    ``inputs`` and ``output`` are deep-copied at capture time so
    mutations applied by later replay steps do not affect the record.
    """

    kind: str
    inputs: dict
    output: Any
    #: ``branch_outputs`` kwarg as passed to ``execute``, or ``None``.
    branch_outputs: Optional[List[Any]] = None


class CaptureExecutor(Executor):
    """An :class:`~stepback.replay.Executor` that records every successful call.

    Behaves identically to :class:`~stepback.replay.Executor` but captures a
    :class:`CapturedCall` record for each invocation of
    :meth:`~stepback.replay.Executor.execute` that returns without raising.

    Example::

        from stepback.testing.replay import CaptureExecutor

        cap = CaptureExecutor(
            llm=my_llm,
            tool=my_tool,
        )
        result = trace.replay_forward(cap)
        assert len(cap.calls) == result.real_executions
        assert cap.calls[0].kind == "tool_call"

    Notes:

    * ``cap.calls`` contains **only** successful executions, not dirty steps
      whose output was forced via an output-forcing substitution (those
      make a step dirty without invoking the executor).
    * ``len(cap.calls)`` should equal ``result.real_executions`` for a
      correct executor implementation.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls: List[CapturedCall] = []

    def execute(
        self,
        kind: str,
        inputs: dict,
        *,
        branch_outputs: Optional[List[Any]] = None,
    ) -> Any:
        output = super().execute(kind, inputs, branch_outputs=branch_outputs)
        self.calls.append(
            CapturedCall(
                kind=kind,
                inputs=copy.deepcopy(inputs),
                output=copy.deepcopy(output),
                branch_outputs=copy.deepcopy(branch_outputs),
            )
        )
        return output


# ---------------------------------------------------------------- FallbackExecutor


class FallbackExecutor(Executor):
    """An :class:`~stepback.replay.Executor` that reuses recorded outputs for dirty steps.

    Useful for **local debugging**: replay shows *which* steps would be
    dirty under a given substitution without calling any LLM or tool.
    The replayed outputs for dirty steps are the originally recorded ones,
    so the replay is structurally correct even though the re-execution
    results are synthetic.

    Example::

        from stepback.testing.replay import FallbackExecutor

        result = trace.replay_forward(FallbackExecutor())
        # Every dirty step is shown with its recorded output.
        for sv in result.steps:
            if sv.dirty:
                print(f"  would re-run {sv.step_id!r} ({sv.kind})")

    To also capture which steps fell back, inspect ``result.real_executions``
    (will be 0 with pure fallback).
    """

    def __init__(self) -> None:
        super().__init__(fallback_recorded=True)


# ---------------------------------------------------------------- Assertion helpers


def _dirty_summary(result: ReplayResult) -> str:
    dirty = [sv for sv in result.steps if sv.dirty]
    if not dirty:
        return "(none)"
    return ", ".join(f"{sv.step_id}({sv.kind})" for sv in dirty[:5]) + (
        f", …+{len(dirty) - 5}" if len(dirty) > 5 else ""
    )


def _find_step(result: ReplayResult, step_id: str) -> StepView:
    for sv in result.steps:
        if sv.step_id == step_id:
            return sv
    available = [sv.step_id for sv in result.steps]
    raise AssertionError(
        f"step_id {step_id!r} not found in replay result; "
        f"available: {available!r}"
    )


def assert_all_cache_hits(result: ReplayResult) -> None:
    """Assert that every step in *result* was a cache hit (zero dirty steps).

    Raises :class:`AssertionError` with a summary of the dirty steps on
    failure.
    """
    if result.dirty_count != 0:
        raise AssertionError(
            f"Expected all cache hits but got {result.dirty_count} dirty step(s): "
            f"{_dirty_summary(result)}"
        )


def assert_dirty_count(result: ReplayResult, expected: int) -> None:
    """Assert that *result* has exactly *expected* dirty steps.

    Raises :class:`AssertionError` listing the actual dirty steps on failure.
    """
    if result.dirty_count != expected:
        raise AssertionError(
            f"Expected {expected} dirty step(s) but got {result.dirty_count}: "
            f"{_dirty_summary(result)}"
        )


def assert_real_executions(result: ReplayResult, expected: int) -> None:
    """Assert that the executor was invoked *expected* times.

    This counts actual executor callback invocations, not dirty steps.
    Output-forcing substitutions make a step dirty without calling the
    executor and therefore do not contribute to this count.

    Raises :class:`AssertionError` with actual vs. expected on failure.
    """
    if result.real_executions != expected:
        raise AssertionError(
            f"Expected {expected} real executor invocation(s) but got "
            f"{result.real_executions}"
        )


def assert_cache_hit_count(result: ReplayResult, expected: int) -> None:
    """Assert that *result* has exactly *expected* cache hits.

    Raises :class:`AssertionError` with actual vs. expected on failure.
    """
    if result.cache_hit_count != expected:
        raise AssertionError(
            f"Expected {expected} cache hit(s) but got {result.cache_hit_count}"
        )


def assert_step_dirty(result: ReplayResult, step_id: str) -> None:
    """Assert that the step with *step_id* was dirty in *result*.

    Raises :class:`AssertionError` if the step is a cache hit or not found.
    """
    sv = _find_step(result, step_id)
    if not sv.dirty:
        raise AssertionError(
            f"Expected step {step_id!r} ({sv.kind}) to be dirty but it was a cache hit"
        )


def assert_step_clean(result: ReplayResult, step_id: str) -> None:
    """Assert that the step with *step_id* was a cache hit in *result*.

    Raises :class:`AssertionError` if the step is dirty or not found.
    """
    sv = _find_step(result, step_id)
    if sv.dirty:
        raise AssertionError(
            f"Expected step {step_id!r} ({sv.kind}) to be a cache hit but it was dirty"
        )


__all__ = [
    "CapturedCall",
    "CaptureExecutor",
    "FallbackExecutor",
    "assert_all_cache_hits",
    "assert_dirty_count",
    "assert_real_executions",
    "assert_cache_hit_count",
    "assert_step_dirty",
    "assert_step_clean",
]
