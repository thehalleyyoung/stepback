"""Tests for distributed replay scheduling (Step 77).

``Trace.replay_forward(distributed=True, workers=N)`` enables distributed
scheduling mode, which uses the same local :class:`~concurrent.futures.ThreadPoolExecutor`
planner as ``workers=N`` (Step 73).  The ``distributed`` flag additionally
auto-selects a worker count when ``workers`` is not specified.

These tests verify:
* ``_effective_workers`` helper logic and the ``cpu_count() is None`` edge case.
* ``distributed=True`` with explicit ``workers`` produces the same result as
  ``workers=N`` alone.
* ``distributed=True`` with no ``workers`` auto-selects an appropriate count.
* ``distributed=True, workers=1`` behaves like sequential replay.
* All four public entrypoints forward ``distributed`` correctly:
  - ``Trace.replay_forward``
  - ``Trace.run_replay``
  - ``Branch.replay_forward``
  - ``ReplayPlan.execute``
* Results are semantically identical to sequential replay on both parallel-
  branch and linear traces.
"""
from __future__ import annotations

import inspect
import os
import unittest.mock

import pytest

from stepback import RecorderKey, record, replay
from stepback.replay import (
    Branch,
    Executor,
    ReplayPlan,
    ReplayResult,
    Trace,
    _DISTRIBUTED_DEFAULT_WORKERS,
    _effective_workers,
)
from stepback.substitutions import ToolOutputSubstitution

from .fixtures.parallel_agent import (
    FACTS,
    fake_llm,
    fake_tool,
    run_parallel_agent,
)
from .fixtures.agent import (
    fake_llm as linear_fake_llm,
    fake_tool as linear_fake_tool,
    run_recorded_agent,
)


# ----------------------------------------------------------------- helpers


def _record_parallel_trace(tmp_path) -> tuple:
    path = os.path.join(str(tmp_path), "parallel.sb")
    key = RecorderKey.fresh()
    with record(path, key=key) as rec:
        run_parallel_agent(rec)
    return path, key


def _record_linear_trace(tmp_path) -> tuple:
    path = os.path.join(str(tmp_path), "linear.sb")
    key = RecorderKey.fresh()
    with record(path, key=key) as rec:
        run_recorded_agent(rec)
    return path, key


def _make_parallel_executor() -> Executor:
    return Executor(llm=fake_llm, tool=fake_tool)


def _make_linear_executor() -> Executor:
    return Executor(llm=linear_fake_llm, tool=linear_fake_tool)


def _dirty_ids(result: ReplayResult) -> set:
    return {sv.step_id for sv in result.steps if sv.dirty}


def _cache_ids(result: ReplayResult) -> set:
    return {sv.step_id for sv in result.steps if sv.cache_hit}


# ================================================================= _effective_workers unit tests


class TestEffectiveWorkers:
    """Unit-test :func:`stepback.replay._effective_workers`."""

    def test_distributed_false_workers_none_returns_none(self):
        assert _effective_workers(False, None) is None

    def test_distributed_false_workers_n_returns_n(self):
        assert _effective_workers(False, 4) == 4

    def test_distributed_false_workers_1_returns_1(self):
        assert _effective_workers(False, 1) == 1

    def test_distributed_true_workers_explicit_returns_workers(self):
        assert _effective_workers(True, 4) == 4

    def test_distributed_true_workers_1_returns_1(self):
        assert _effective_workers(True, 1) == 1

    def test_distributed_true_workers_none_positive(self):
        """Auto-selected count must be a positive integer."""
        result = _effective_workers(True, None)
        assert isinstance(result, int)
        assert result >= 1

    def test_distributed_true_auto_capped_at_default_max(self):
        """Auto-selected count must not exceed _DISTRIBUTED_DEFAULT_WORKERS."""
        with unittest.mock.patch("stepback.replay.os.cpu_count", return_value=1024):
            result = _effective_workers(True, None)
        assert result == _DISTRIBUTED_DEFAULT_WORKERS

    def test_distributed_true_uses_cpu_count(self):
        """Auto-selected count is min(default_max, cpu_count)."""
        with unittest.mock.patch("stepback.replay.os.cpu_count", return_value=2):
            result = _effective_workers(True, None)
        assert result == 2

    def test_distributed_true_cpu_count_none_fallback(self):
        """When os.cpu_count() returns None the fallback is 4."""
        with unittest.mock.patch("stepback.replay.os.cpu_count", return_value=None):
            result = _effective_workers(True, None)
        assert result == min(_DISTRIBUTED_DEFAULT_WORKERS, 4)

    def test_distributed_default_workers_constant_positive(self):
        assert isinstance(_DISTRIBUTED_DEFAULT_WORKERS, int)
        assert _DISTRIBUTED_DEFAULT_WORKERS >= 1


# ================================================================= Trace.replay_forward


class TestTraceReplayForwardDistributed:
    """``Trace.replay_forward(distributed=True, ...)`` tests."""

    def test_no_sub_all_cache_hits(self, tmp_path):
        """distributed=True, no substitutions → every step is a cache hit."""
        path, key = _record_parallel_trace(tmp_path)
        t = replay(path)
        result = t.replay_forward(_make_parallel_executor(), distributed=True)
        assert result.real_executions == 0
        assert result.dirty_count == 0
        assert result.cache_hit_count == len(result.steps)

    def test_distributed_workers_n_matches_workers_n(self, tmp_path):
        """distributed=True, workers=4 gives same dirty/cache split as workers=4 alone."""
        path, key = _record_parallel_trace(tmp_path)
        sub = ToolOutputSubstitution("step:2", {"result": "override"})

        t1 = replay(path)
        t1.substitute(sub)
        r_dist = t1.replay_forward(_make_parallel_executor(), distributed=True, workers=4)

        t2 = replay(path)
        t2.substitute(sub)
        r_work = t2.replay_forward(_make_parallel_executor(), workers=4)

        assert _dirty_ids(r_dist) == _dirty_ids(r_work)
        assert r_dist.real_executions == r_work.real_executions
        assert r_dist.dirty_count == r_work.dirty_count
        assert r_dist.cache_hit_count == r_work.cache_hit_count

    def test_distributed_matches_sequential(self, tmp_path):
        """distributed=True gives same dirty/cache split as sequential replay."""
        path, key = _record_parallel_trace(tmp_path)
        sub = ToolOutputSubstitution("step:2", {"result": "override"})

        t_seq = replay(path)
        t_seq.substitute(sub)
        r_seq = t_seq.replay_forward(_make_parallel_executor())

        t_dist = replay(path)
        t_dist.substitute(sub)
        r_dist = t_dist.replay_forward(_make_parallel_executor(), distributed=True)

        assert _dirty_ids(r_dist) == _dirty_ids(r_seq)
        assert r_dist.dirty_count == r_seq.dirty_count
        assert r_dist.cache_hit_count == r_seq.cache_hit_count

    def test_distributed_workers_1_no_error(self, tmp_path):
        """distributed=True, workers=1 must not error and gives correct result."""
        path, key = _record_parallel_trace(tmp_path)
        t = replay(path)
        result = t.replay_forward(_make_parallel_executor(), distributed=True, workers=1)
        assert result.dirty_count == 0
        assert result.cache_hit_count == len(result.steps)

    def test_distributed_auto_workers_correct_result(self, tmp_path):
        """distributed=True with auto workers produces the same dirty split."""
        path, key = _record_parallel_trace(tmp_path)
        sub = ToolOutputSubstitution("step:2", {"result": "auto-workers-override"})

        t_seq = replay(path)
        t_seq.substitute(sub)
        r_seq = t_seq.replay_forward(_make_parallel_executor())

        t_auto = replay(path)
        t_auto.substitute(sub)
        r_auto = t_auto.replay_forward(_make_parallel_executor(), distributed=True)

        assert r_auto.dirty_count == r_seq.dirty_count
        assert _dirty_ids(r_auto) == _dirty_ids(r_seq)


# ================================================================= Trace.run_replay


class TestTraceRunReplayDistributed:
    """``Trace.run_replay(..., distributed=True)`` forwarding tests."""

    def test_run_replay_distributed_no_sub(self, tmp_path):
        path, key = _record_parallel_trace(tmp_path)
        t = replay(path)
        subs = t.pending_subs
        result = t.run_replay(subs, _make_parallel_executor(), distributed=True)
        assert result.real_executions == 0

    def test_run_replay_distributed_false_matches_workers(self, tmp_path):
        """distributed=False preserves existing workers=N behaviour."""
        path, key = _record_parallel_trace(tmp_path)

        t1 = replay(path)
        r1 = t1.run_replay(t1.pending_subs, _make_parallel_executor(), workers=4)

        t2 = replay(path)
        r2 = t2.run_replay(t2.pending_subs, _make_parallel_executor(), workers=4, distributed=False)

        assert r1.dirty_count == r2.dirty_count
        assert r1.cache_hit_count == r2.cache_hit_count


# ================================================================= Branch.replay_forward


class TestBranchReplayForwardDistributed:
    """``Branch.replay_forward(distributed=True)`` forwarding tests."""

    def test_branch_distributed_no_sub(self, tmp_path):
        path, key = _record_parallel_trace(tmp_path)
        t = replay(path)
        branch = t.branch_at("step:1", "main")
        result = branch.replay_forward(_make_parallel_executor(), distributed=True)
        assert result.dirty_count == 0
        assert result.cache_hit_count == len(result.steps)

    def test_branch_distributed_with_sub_matches_sequential(self, tmp_path):
        """Branch.replay_forward(distributed=True) propagates dirty-set correctly."""
        path, key = _record_parallel_trace(tmp_path)
        sub = ToolOutputSubstitution("step:2", {"result": "branch-override"})

        t1 = replay(path)
        b1 = t1.branch_at("step:1", "seq")
        b1.substitute(sub)
        r_seq = b1.replay_forward(_make_parallel_executor())

        t2 = replay(path)
        b2 = t2.branch_at("step:1", "dist")
        b2.substitute(sub)
        r_dist = b2.replay_forward(_make_parallel_executor(), distributed=True)

        assert _dirty_ids(r_dist) == _dirty_ids(r_seq)


# ================================================================= ReplayPlan.execute


class TestReplayPlanExecuteDistributed:
    """``ReplayPlan.execute(distributed=True)`` forwarding tests."""

    def test_plan_execute_distributed_no_sub(self, tmp_path):
        path, key = _record_parallel_trace(tmp_path)
        t = replay(path)
        plan = t.plan_replay()
        result = plan.execute(_make_parallel_executor(), distributed=True)
        assert result.dirty_count == 0
        assert result.cache_hit_count == len(result.steps)

    def test_plan_execute_distributed_matches_sequential(self, tmp_path):
        path, key = _record_parallel_trace(tmp_path)
        sub = ToolOutputSubstitution("step:2", {"result": "plan-override"})

        t1 = replay(path)
        t1.substitute(sub)
        r_seq = t1.plan_replay().execute(_make_parallel_executor())

        t2 = replay(path)
        t2.substitute(sub)
        r_dist = t2.plan_replay().execute(_make_parallel_executor(), distributed=True)

        assert _dirty_ids(r_dist) == _dirty_ids(r_seq)
        assert r_dist.dirty_count == r_seq.dirty_count

    def test_plan_execute_event_bus_forces_sequential(self, tmp_path):
        """When _event_bus is set, the sequential path is used regardless of distributed."""
        path, key = _record_parallel_trace(tmp_path)
        t = replay(path)
        plan = t.plan_replay()

        events = []

        class _FakeBus:
            def publish(self, event):
                events.append(event)

        # Should not raise even with distributed=True + workers=8
        result = plan.execute(
            _make_parallel_executor(),
            distributed=True,
            workers=8,
            _event_bus=_FakeBus(),
        )
        assert result.dirty_count == 0


# ================================================================= Linear trace


class TestDistributedLinearTrace:
    """distributed=True works correctly on linear (no-branch) traces."""

    def test_linear_trace_no_sub(self, tmp_path):
        path, key = _record_linear_trace(tmp_path)
        t = replay(path)
        result = t.replay_forward(_make_linear_executor(), distributed=True)
        assert result.dirty_count == 0
        assert result.cache_hit_count == len(result.steps)

    def test_linear_trace_with_sub_matches_sequential(self, tmp_path):
        """distributed=True on a linear trace still propagates dirty-set."""
        path, key = _record_linear_trace(tmp_path)
        # Find the first tool step in the linear trace to sub via output-forcing
        t_probe = replay(path)
        tool_step = next(
            s for s in t_probe.recorded_steps if s["step_kind"] == "tool_call"
        )
        sub = ToolOutputSubstitution(tool_step["step_id"], {"result": "override"})

        t_seq = replay(path)
        t_seq.substitute(sub)
        r_seq = t_seq.replay_forward(_make_linear_executor())

        t_dist = replay(path)
        t_dist.substitute(sub)
        r_dist = t_dist.replay_forward(_make_linear_executor(), distributed=True)

        assert r_dist.dirty_count == r_seq.dirty_count
        assert _dirty_ids(r_dist) == _dirty_ids(r_seq)


# ================================================================= API signatures


class TestDistributedAPISignature:
    """Verify the new parameter appears in the correct signatures."""

    def test_trace_replay_forward_accepts_distributed(self):
        sig = inspect.signature(Trace.replay_forward)
        assert "distributed" in sig.parameters
        assert sig.parameters["distributed"].default is False

    def test_trace_run_replay_accepts_distributed(self):
        sig = inspect.signature(Trace.run_replay)
        assert "distributed" in sig.parameters
        assert sig.parameters["distributed"].default is False

    def test_branch_replay_forward_accepts_distributed(self):
        sig = inspect.signature(Branch.replay_forward)
        assert "distributed" in sig.parameters
        assert sig.parameters["distributed"].default is False

    def test_replay_plan_execute_accepts_distributed(self):
        sig = inspect.signature(ReplayPlan.execute)
        assert "distributed" in sig.parameters
        assert sig.parameters["distributed"].default is False

    def test_effective_workers_is_importable(self):
        from stepback.replay import _effective_workers as ew
        assert callable(ew)
