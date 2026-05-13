"""Backpressure and sampling controls for the stepback recorder (Step 142).

:class:`RecorderOptions` is the single configuration object that governs
whether recorder failures propagate to the agent (*mandatory* mode) or are
silently absorbed (*fail-open* mode), and what fraction of traces to
actually write (*sample_rate*).  When the step budget (*max_queue_depth*)
is exceeded the *overflow_policy* determines whether the trace is silently
abandoned (``"drop"``) or writing is allowed to continue unrestricted
(``"block"`` — useful when the budget is a soft limit and you'd rather have
a complete trace than a truncated one).

Quick reference
---------------

Fail-open (never crash the agent)::

    from stepback import record, RecorderOptions

    opts = RecorderOptions(mandatory=False)
    with record("./trace.sb", options=opts) as rec:
        rec.llm_call(...)   # any internal recorder error is silently swallowed

Sample 10% of traces (agent behaviour is identical; only 1-in-10 produce a
file)::

    opts = RecorderOptions(sample_rate=0.1)
    with record("./trace.sb", options=opts) as rec:
        rec.llm_call(...)   # executor is ALWAYS called; file may not be written

Limit trace length and abandon recording on overflow::

    opts = RecorderOptions(max_queue_depth=50, overflow_policy="drop",
                           mandatory=False)
    with record("./trace.sb", options=opts) as rec:
        # steps 1-50 are written; step 51+ are silently discarded
        for _ in range(200):
            rec.tool_call(...)

Mandatory — errors propagate exactly as before (default before this
change)::

    opts = RecorderOptions(mandatory=True)  # equivalent to no options at all

Inspecting failure metadata after the block::

    with record("./trace.sb", options=RecorderOptions(mandatory=False)) as rec:
        ...
    print(rec.dropped_steps, rec.recording_errors)
"""
from __future__ import annotations

import random as _random
from dataclasses import dataclass, field
from typing import List, Optional


__all__ = ["RecorderOptions"]


@dataclass
class RecorderOptions:
    """Controls fault isolation, trace sampling, and step budgets.

    Attributes:
        mandatory: When ``True`` all recorder errors propagate to the caller
            (the agent crashes on any recording failure) — this is the
            strictest mode for audit-grade traces.  When ``False`` (the
            default) the recorder *fails open*: internal errors are caught,
            counted in :attr:`~stepback.recorder.Recorder.recording_errors`,
            and the agent continues uninterrupted.

        sample_rate: Fraction of ``record()`` / ``arecord()`` calls that
            actually write a trace.  Must be in ``[0.0, 1.0]``.  At
            ``1.0`` (default) every call is recorded.  At ``0.5`` roughly
            half of calls produce a ``.sb`` file; the other half yield a
            :class:`_NullRecorder` that still *executes* every user
            callable (executor functions, branch closures) but discards
            all output.  Sampling is trace-level: once the decision is
            made at context entry it is stable for the whole trace.

        max_queue_depth: Maximum number of steps to write per trace before
            applying :attr:`overflow_policy`.  ``0`` (the default) means
            no limit.  When positive and the step count reaches this value,
            *overflow_policy* determines what happens.

        overflow_policy: Action to take when :attr:`max_queue_depth` is
            exceeded.  ``"drop"`` (default) abandons the rest of the trace:
            subsequent steps are not written to disk (their executors are
            still called so agent behaviour is unchanged).  ``"block"``
            allows writing to continue without limit — useful when the
            depth limit is a soft budget hint rather than a hard cap.
            ``"drop"`` is rejected when :attr:`mandatory` is ``True``
            (a mandatory trace must be complete).

        rng: :class:`random.Random` instance used for the sampling roll.
            When ``None`` (default) ``random.random()`` is used.  Supply a
            seeded instance for reproducible test behaviour.

    Raises:
        ValueError: If *sample_rate* is outside ``[0.0, 1.0]``, if
            *max_queue_depth* is negative, if *overflow_policy* is not
            ``"drop"`` or ``"block"``, or if *mandatory=True* with
            *overflow_policy="drop"* (contradictory).
    """

    mandatory: bool = False
    sample_rate: float = 1.0
    max_queue_depth: int = 0
    overflow_policy: str = "drop"
    rng: Optional[_random.Random] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.sample_rate <= 1.0):
            raise ValueError(
                f"sample_rate must be in [0.0, 1.0], got {self.sample_rate!r}"
            )
        if self.max_queue_depth < 0:
            raise ValueError(
                f"max_queue_depth must be >= 0, got {self.max_queue_depth!r}"
            )
        if self.overflow_policy not in ("drop", "block"):
            raise ValueError(
                f"overflow_policy must be 'drop' or 'block', "
                f"got {self.overflow_policy!r}"
            )
        if self.mandatory and self.overflow_policy == "drop":
            raise ValueError(
                "mandatory=True with overflow_policy='drop' is contradictory: "
                "a mandatory trace cannot silently lose steps.  "
                "Use mandatory=True with overflow_policy='block', or "
                "mandatory=False with overflow_policy='drop'."
            )

    def _should_sample(self) -> bool:
        """Return True if this trace should be recorded (not sampled out)."""
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False
        roll = self.rng.random() if self.rng is not None else _random.random()
        return roll < self.sample_rate


class _NullTraceWriter:
    """A TraceWriter-compatible stub that silently discards all writes.

    Used by :class:`_NullRecorder` so that the full :class:`Recorder` logic
    (parent tracking, parallel fan-out/join, etc.) works unchanged when a
    trace is sampled out.
    """

    def write_step(self, step: dict) -> None:  # noqa: ARG002
        pass

    def write_capability(
        self,
        name: str,
        *,
        mandatory: bool = False,
        params: Optional[dict] = None,
    ) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def _make_null_recorder() -> "Recorder":  # type: ignore[name-defined]  # noqa: F821
    """Return a :class:`Recorder` backed by a :class:`_NullTraceWriter`.

    The recorder still tracks steps and parent chains internally so that
    ``parallel()`` branch closures, context hashing, and other logic that
    reads ``self.steps`` works correctly.  No data is written to disk.
    """
    # Deferred import to avoid circular dependency.
    from .recorder import Recorder, RecorderKey  # noqa: PLC0415

    null_writer = _NullTraceWriter()
    # A fresh dummy key is fine — it is never used for I/O.
    key = RecorderKey.fresh()
    return Recorder(writer=null_writer, key=key)  # type: ignore[arg-type]
