"""Tests for parallel-branch replay scheduling (Step 73).

Independent dirty branches execute in parallel via a ThreadPoolExecutor;
joins wait for all consumed branch outputs before executing.  The result
is semantically identical to sequential replay (:func:`_execute_plan`).

Trace shape used throughout (same 11-step fixture as test_parallel_branches):

    step:1   llm_call            "plan"
    step:2   tool_call           "split_question"   parent=step:1
    step:3   parallel_branch_open "research_fanout"  parent=step:2
    step:4   llm_call             parent=step:3  (branch A history)
    step:5   tool_call            parent=step:4  (branch A lookup_history)
    step:6   llm_call             parent=step:3  (branch B economics)
    step:7   tool_call            parent=step:6  (branch B lookup_econ)
    step:8   llm_call             parent=step:3  (branch C politics)
    step:9   tool_call            parent=step:8  (branch C lookup_polit)
    step:10  parallel_branch_join parent_step_ids=[step:5,step:7,step:9]
    step:11  llm_call            "synthesise"    parent=step:10
"""
from __future__ import annotations

import os
import threading
from typing import List

import pytest

from stepback import RecorderKey, record, replay
from stepback.replay import Executor, MissingExecutor
from stepback.substitutions import PromptSubstitution, ToolOutputSubstitution

from .fixtures.parallel_agent import (
    FACTS,
    fake_llm,
    fake_tool,
    run_parallel_agent,
)


# ----------------------------------------------------------------- fixture


def _record_parallel_trace(tmp_path) -> tuple:
    path = os.path.join(str(tmp_path), "parallel.sb")
    key = RecorderKey.fresh()
    with record(path, key=key) as rec:
        run_parallel_agent(rec)
    return path, key


def _make_executor():
    return Executor(llm=fake_llm, tool=fake_tool)


# ----------------------------------------------------------------- helpers


def _dirty_ids(result) -> set:
    return {sv.step_id for sv in result.steps if sv.dirty}


def _cache_ids(result) -> set:
    return {sv.step_id for sv in result.steps if sv.cache_hit}


# ================================================================= tests


class TestParallelReplayCorrectness:
    """Parallel replay must produce the same result as sequential replay."""

    def test_no_substitution_parallel_equals_sequential(self, tmp_path):
        path, key = _record_parallel_trace(tmp_path)
        t = replay(path)
        ex = _make_executor()
        seq = t.replay_forward(ex)
        t2 = replay(path)
        ex2 = _make_executor()
        par = t2.replay_forward(ex2, workers=4)

        assert par.dirty_count == seq.dirty_count == 0
        assert par.cache_hit_count == seq.cache_hit_count == 11
        assert par.real_executions == seq.real_executions == 0
        assert [sv.step_id for sv in par.steps] == [sv.step_id for sv in seq.steps]
        assert [sv.outputs for sv in par.steps] == [sv.outputs for sv in seq.steps]

    def test_substitution_parallel_equals_sequential(self, tmp_path):
        path, key = _record_parallel_trace(tmp_path)
        # Substitute the branch-B lookup tool output.
        sub = ToolOutputSubstitution(
            at_step="step:7",
            fake_response={"answer": "GDP 99T USD", "confidence": 0.99},
        )
        t_seq = replay(path)
        t_seq.substitute(sub)
        seq = t_seq.replay_forward(_make_executor())

        t_par = replay(path)
        t_par.substitute(sub)
        par = t_par.replay_forward(_make_executor(), workers=4)

        assert par.dirty_count == seq.dirty_count
        assert par.cache_hit_count == seq.cache_hit_count
        assert _dirty_ids(par) == _dirty_ids(seq)
        for sv_par, sv_seq in zip(par.steps, seq.steps):
            assert sv_par.step_id == sv_seq.step_id
            assert sv_par.dirty == sv_seq.dirty
            assert sv_par.cache_hit == sv_seq.cache_hit

    def test_step_order_preserved(self, tmp_path):
        """Steps appear in original topological order, not completion order."""
        path, _ = _record_parallel_trace(tmp_path)
        t = replay(path)
        result = t.replay_forward(_make_executor(), workers=4)
        ids = [sv.step_id for sv in result.steps]
        # Must be in the recorded topological order.
        assert ids == [f"step:{i}" for i in range(1, 12)]

    def test_workers_1_identical_to_none(self, tmp_path):
        """workers=1 produces the same result as workers=None (sequential)."""
        path, _ = _record_parallel_trace(tmp_path)
        sub = ToolOutputSubstitution(
            at_step="step:5",
            fake_response={"answer": "founded 1776"},
        )
        t1 = replay(path)
        t1.substitute(sub)
        r1 = t1.replay_forward(_make_executor(), workers=None)

        t2 = replay(path)
        t2.substitute(sub)
        r2 = t2.replay_forward(_make_executor(), workers=1)

        assert _dirty_ids(r1) == _dirty_ids(r2)
        assert r1.cache_hit_count == r2.cache_hit_count


class TestParallelDirtySetIsolation:
    """Dirty-set isolation guarantees must hold under parallel execution."""

    def test_one_branch_dirty_siblings_clean(self, tmp_path):
        """Substituting branch-B's tail dirties only branch B + join + synth."""
        path, _ = _record_parallel_trace(tmp_path)
        t = replay(path)
        t.substitute(
            ToolOutputSubstitution(
                at_step="step:7",
                fake_response={"answer": "changed", "confidence": 0.0},
            )
        )
        result = t.replay_forward(_make_executor(), workers=4)

        dirty = _dirty_ids(result)
        # step:7 (subst), step:10 (join), step:11 (synth)
        assert "step:7" in dirty
        assert "step:10" in dirty
        assert "step:11" in dirty
        # Sibling branches A and C must be clean.
        for sid in ("step:4", "step:5", "step:8", "step:9"):
            assert sid not in dirty, f"{sid} should not be dirty"

    def test_dirty_count_constant_across_branches(self, tmp_path):
        """Branch isolation: dirty set size is N_substituted_branches + 2."""
        path, _ = _record_parallel_trace(tmp_path)
        for target_sid in ("step:5", "step:7", "step:9"):
            t = replay(path)
            t.substitute(
                ToolOutputSubstitution(
                    at_step=target_sid,
                    fake_response={"answer": "changed"},
                )
            )
            result = t.replay_forward(_make_executor(), workers=4)
            # 1 branch tool (subst) + join + synth = 3
            assert result.dirty_count == 3, (
                f"expected 3 dirty for sub at {target_sid}, got {result.dirty_count}"
            )

    def test_join_waits_for_all_branches(self, tmp_path):
        """Join step is never dirty from a clean substitution (sanity check)."""
        path, _ = _record_parallel_trace(tmp_path)
        t = replay(path)
        result = t.replay_forward(_make_executor(), workers=4)
        # Without any substitution, the join must be a cache hit.
        join_sv = next(sv for sv in result.steps if sv.kind == "parallel_branch_join")
        assert join_sv.cache_hit
        assert not join_sv.dirty


class TestParallelReplayPlan:
    """ReplayPlan.execute should also support workers=N."""

    def test_plan_execute_parallel_equals_sequential(self, tmp_path):
        path, _ = _record_parallel_trace(tmp_path)
        sub = ToolOutputSubstitution(
            at_step="step:9",
            fake_response={"answer": "changed politics"},
        )
        t = replay(path)
        t.substitute(sub)
        plan = t.plan_replay()

        r_seq = plan.execute(_make_executor())
        t2 = replay(path)
        t2.substitute(sub)
        plan2 = t2.plan_replay()
        r_par = plan2.execute(_make_executor(), workers=4)

        assert _dirty_ids(r_seq) == _dirty_ids(r_par)
        assert r_seq.cache_hit_count == r_par.cache_hit_count


class TestParallelReplayLinearTrace:
    """Linear traces (no parallel branches) work correctly with workers=N."""

    def test_linear_trace_parallel_equals_sequential(self, tmp_path):
        from stepback.testing import run_recorded_agent, fake_llm as fl, fake_tool as ft

        path = os.path.join(str(tmp_path), "linear.sb")
        key = RecorderKey.fresh()
        with record(path, key=key) as rec:
            run_recorded_agent(rec)

        t_seq = replay(path)
        t_par = replay(path)
        ex_seq = Executor(llm=fl, tool=ft)
        ex_par = Executor(llm=fl, tool=ft)

        r_seq = t_seq.replay_forward(ex_seq)
        r_par = t_par.replay_forward(ex_par, workers=4)

        assert r_seq.dirty_count == r_par.dirty_count == 0
        assert r_seq.cache_hit_count == r_par.cache_hit_count
        assert [sv.step_id for sv in r_seq.steps] == [sv.step_id for sv in r_par.steps]


class TestExecutorThreadSafety:
    """Executor counters must be accurate under concurrent branch workers."""

    def test_real_calls_accurate_with_parallel_dirty_branches(self, tmp_path):
        """real_calls must equal the total number of executor.execute() calls."""
        path, _ = _record_parallel_trace(tmp_path)
        # Dirty all three branch tails to force 3 real executions for the tool
        # steps, plus the join, plus synthesise = 5 real calls total.
        t = replay(path)
        for sid in ("step:5", "step:7", "step:9"):
            t.substitute(
                ToolOutputSubstitution(
                    at_step=sid,
                    fake_response={"answer": "dirtied"},
                )
            )
        ex = _make_executor()
        result = t.replay_forward(ex, workers=4)

        # real_executions in the result and executor.real_calls must agree.
        assert result.real_executions == ex.real_calls
        # 3 substituted tool steps (output-forcing, no real exec) + join + synth = 2
        # But output-forcing substitutions bypass the executor — join + synth are executed.
        # Join is dirty (branch outputs changed) → real execution.
        # Synth is dirty (parent join changed) → real execution.
        assert result.real_executions == 2

    def test_fallback_uses_accurate_under_parallel(self, tmp_path):
        """fallback_uses must be accurate when branches use fallback_recorded."""
        path, _ = _record_parallel_trace(tmp_path)
        t = replay(path)
        # Dirty all branch LLM steps via prompt substitution so they need re-exec.
        for sid in ("step:4", "step:6", "step:8"):
            t.substitute(
                PromptSubstitution(
                    at_step=sid,
                    new_messages=[{"role": "user", "content": "new prompt"}],
                )
            )
        # Use fallback_recorded so LLM steps fall back to recorded output.
        ex = Executor(tool=fake_tool, fallback_recorded=True)
        result = t.replay_forward(ex, workers=4)

        # 3 LLM steps are dirty and use fallback → fallback_uses == 3.
        assert ex.fallback_uses == 3
        # With fallback, recorded output is reused → no output change → downstream stays clean.
        assert result.real_executions == 0
        assert result.dirty_count == 3

    def test_callbacks_invoked_from_worker_threads(self, tmp_path):
        """Executor callbacks are called from non-main threads in parallel mode."""
        path, _ = _record_parallel_trace(tmp_path)
        call_threads: List[int] = []
        lock = threading.Lock()

        def thread_tracking_llm(model, messages):
            with lock:
                call_threads.append(threading.get_ident())
            return fake_llm(model, messages)

        # Dirty all three branch LLM steps so the executor is called in branches.
        t = replay(path)
        for sid in ("step:4", "step:6", "step:8"):
            t.substitute(
                PromptSubstitution(
                    at_step=sid,
                    new_messages=[{"role": "user", "content": "new"}],
                )
            )
        ex = Executor(llm=thread_tracking_llm, tool=fake_tool)
        t.replay_forward(ex, workers=4)

        main_tid = threading.get_ident()
        # At least some LLM calls should have come from non-main threads.
        assert len(call_threads) >= 3
        assert any(tid != main_tid for tid in call_threads), (
            "Expected at least one LLM callback from a worker thread"
        )


class TestParallelReplayEdgeCases:
    """Edge cases and error propagation."""

    def test_empty_trace_parallel(self, tmp_path):
        from stepback.replay import Trace
        from stepback.substitutions import SubstitutionSet

        t = Trace(path="", header={}, recorded_steps=[])
        result = t.replay_forward(workers=4)
        assert result.dirty_count == 0
        assert result.cache_hit_count == 0
        assert result.steps == []

    def test_missing_executor_raises_in_parallel(self, tmp_path):
        """MissingExecutor propagates from a branch worker."""
        path, _ = _record_parallel_trace(tmp_path)
        t = replay(path)
        # Dirty a branch LLM step; no llm callback → MissingExecutor.
        t.substitute(
            PromptSubstitution(
                at_step="step:4",
                new_messages=[{"role": "user", "content": "oops"}],
            )
        )
        ex = Executor(tool=fake_tool)  # no llm callback
        with pytest.raises(MissingExecutor):
            t.replay_forward(ex, workers=4)

    def test_workers_2_produces_correct_result(self, tmp_path):
        """workers=2 also produces a correct result (not just workers=4)."""
        path, _ = _record_parallel_trace(tmp_path)
        t = replay(path)
        sub = ToolOutputSubstitution(
            at_step="step:7",
            fake_response={"answer": "changed"},
        )
        t.substitute(sub)
        result = t.replay_forward(_make_executor(), workers=2)
        assert result.dirty_count == 3
        assert "step:7" in _dirty_ids(result)
        assert "step:10" in _dirty_ids(result)
        assert "step:11" in _dirty_ids(result)
