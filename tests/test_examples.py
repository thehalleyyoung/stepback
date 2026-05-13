"""Tests for the examples/ gallery.

Each test verifies that:
1. The agent can record a trace to a temp directory.
2. The replay with substitution runs without error.
3. At least one step becomes dirty after substitution.

These tests are fully offline — all agents use deterministic fake LLMs
and scripted tools.
"""
from __future__ import annotations

import os
import sys

import pytest

# Ensure examples/ is importable when running from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# customer_support
# ---------------------------------------------------------------------------

class TestCustomerSupportExample:
    def test_record(self, tmp_path):
        from stepback import record
        from stepback.testing.support_agent import run_support_task

        path = str(tmp_path / "cs.sb")
        with record(path) as rec:
            run_support_task(rec, "order-status-delayed")

        assert os.path.exists(path), "trace file was not created"

    def test_replay_with_substitution(self, tmp_path):
        from stepback import Executor, record, replay
        from stepback.substitutions import ToolOutputSubstitution
        from stepback.testing.support_agent import fake_llm, fake_tool, run_support_task

        path = str(tmp_path / "cs.sb")
        with record(path) as rec:
            run_support_task(rec, "order-status-delayed")

        trace = replay(path)
        shipping_step = next(
            s for s in trace.recorded_steps if s.get("name") == "check_shipping"
        )
        trace.substitute(
            ToolOutputSubstitution(
                at_step=shipping_step["step_id"],
                fake_response={"status": "on_time", "days_overdue": 0},
            )
        )
        result = trace.replay_forward(Executor(llm=fake_llm, tool=fake_tool))
        dirty = [sv for sv in result.steps if sv.dirty]
        assert len(dirty) >= 1, "expected at least one dirty step"

    def test_step_count(self, tmp_path):
        from stepback import record
        from stepback.testing.support_agent import run_support_task

        path = str(tmp_path / "cs.sb")
        with record(path) as rec:
            run_support_task(rec, "order-status-delayed")

        from stepback import replay
        trace = replay(path)
        assert len(trace.recorded_steps) >= 5


# ---------------------------------------------------------------------------
# rag_pipeline
# ---------------------------------------------------------------------------

class TestRagPipelineExample:
    def test_record(self, tmp_path):
        from stepback import record
        from examples.rag_pipeline.agent import run_rag_agent

        path = str(tmp_path / "rag.sb")
        with record(path) as rec:
            run_rag_agent(rec)

        assert os.path.exists(path)

    def test_replay_with_substitution(self, tmp_path):
        from stepback import Executor, record, replay
        from stepback.substitutions import ToolOutputSubstitution
        from examples.rag_pipeline.agent import (
            _ALT_PASSAGES,
            fake_llm,
            fake_tool,
            run_rag_agent,
        )

        path = str(tmp_path / "rag.sb")
        with record(path) as rec:
            run_rag_agent(rec)

        trace = replay(path)
        retrieve_step = next(
            s for s in trace.recorded_steps
            if s.get("name") == "retrieve"
            and s.get("inputs", {}).get("arguments", {}).get("topic") == "history"
        )
        trace.substitute(
            ToolOutputSubstitution(
                at_step=retrieve_step["step_id"],
                fake_response={"passages": _ALT_PASSAGES["history"], "topic": "history"},
            )
        )
        result = trace.replay_forward(Executor(llm=fake_llm, tool=fake_tool))
        dirty = [sv for sv in result.steps if sv.dirty]
        assert len(dirty) >= 1

    def test_retrieve_steps_present(self, tmp_path):
        from stepback import record, replay
        from examples.rag_pipeline.agent import run_rag_agent

        path = str(tmp_path / "rag.sb")
        with record(path) as rec:
            run_rag_agent(rec)

        trace = replay(path)
        retrieve_steps = [s for s in trace.recorded_steps if s.get("name") == "retrieve"]
        assert len(retrieve_steps) == 3  # history, products, leadership


# ---------------------------------------------------------------------------
# tool_using_agent
# ---------------------------------------------------------------------------

class TestToolUsingAgentExample:
    def test_record(self, tmp_path):
        from stepback import record
        from stepback.testing.agent import run_recorded_agent

        path = str(tmp_path / "pay.sb")
        with record(path) as rec:
            run_recorded_agent(rec)

        assert os.path.exists(path)

    def test_bug_is_in_trace(self, tmp_path):
        from stepback import record, replay
        from stepback.testing.agent import LOOKUP_BUG_ROW, run_recorded_agent

        path = str(tmp_path / "pay.sb")
        with record(path) as rec:
            run_recorded_agent(rec)

        trace = replay(path)
        bad_iban = LOOKUP_BUG_ROW["iban"]
        wire = next(
            (s for s in trace.recorded_steps
             if s.get("name") == "payment.transfer"
             and s.get("inputs", {}).get("arguments", {}).get("iban") == bad_iban),
            None,
        )
        assert wire is not None, "expected bad wire step in trace"

    def test_substitution_makes_steps_dirty(self, tmp_path):
        from stepback import Executor, record, replay
        from stepback.substitutions import ToolOutputSubstitution
        from stepback.testing.agent import LOOKUP_FIXED_ROW, fake_llm, fake_tool, run_recorded_agent

        path = str(tmp_path / "pay.sb")
        with record(path) as rec:
            run_recorded_agent(rec)

        trace = replay(path)
        lookup = next(s for s in trace.recorded_steps if s.get("name") == "lookup_customer")
        trace.substitute(
            ToolOutputSubstitution(at_step=lookup["step_id"], fake_response=LOOKUP_FIXED_ROW)
        )
        result = trace.replay_forward(Executor(llm=fake_llm, tool=fake_tool))
        dirty = [sv for sv in result.steps if sv.dirty]
        assert len(dirty) >= 1

    def test_twelve_steps(self, tmp_path):
        from stepback import record, replay
        from stepback.testing.agent import run_recorded_agent

        path = str(tmp_path / "pay.sb")
        with record(path) as rec:
            run_recorded_agent(rec)

        trace = replay(path)
        assert len(trace.recorded_steps) == 12


# ---------------------------------------------------------------------------
# multi_step_planner
# ---------------------------------------------------------------------------

class TestMultiStepPlannerExample:
    def test_record(self, tmp_path):
        from stepback import record
        from examples.multi_step_planner.agent import run_planner_agent

        path = str(tmp_path / "plan.sb")
        with record(path) as rec:
            run_planner_agent(rec)

        assert os.path.exists(path)

    def test_execute_task_steps(self, tmp_path):
        from stepback import record, replay
        from examples.multi_step_planner.agent import run_planner_agent

        path = str(tmp_path / "plan.sb")
        with record(path) as rec:
            run_planner_agent(rec)

        trace = replay(path)
        exec_steps = [s for s in trace.recorded_steps if s.get("name") == "execute_task"]
        assert len(exec_steps) == 3

    def test_substitution_makes_steps_dirty(self, tmp_path):
        from stepback import Executor, record, replay
        from stepback.substitutions import ToolOutputSubstitution
        from examples.multi_step_planner.agent import (
            _FAILED_TEST_RESULT,
            fake_llm,
            fake_tool,
            run_planner_agent,
        )

        path = str(tmp_path / "plan.sb")
        with record(path) as rec:
            run_planner_agent(rec)

        trace = replay(path)
        test_step = next(
            s for s in trace.recorded_steps
            if s.get("name") == "execute_task"
            and s.get("inputs", {}).get("arguments", {}).get("action") == "run_tests"
        )
        trace.substitute(
            ToolOutputSubstitution(at_step=test_step["step_id"], fake_response=_FAILED_TEST_RESULT)
        )
        result = trace.replay_forward(Executor(llm=fake_llm, tool=fake_tool))
        dirty = [sv for sv in result.steps if sv.dirty]
        assert len(dirty) >= 1


# ---------------------------------------------------------------------------
# parallel_branch
# ---------------------------------------------------------------------------

class TestParallelBranchExample:
    def test_record(self, tmp_path):
        from stepback import record
        from stepback.testing.parallel_agent import run_parallel_agent

        path = str(tmp_path / "par.sb")
        with record(path) as rec:
            run_parallel_agent(rec)

        assert os.path.exists(path)

    def test_substitution_makes_steps_dirty(self, tmp_path):
        from stepback import Executor, record, replay
        from stepback.substitutions import ToolOutputSubstitution
        from stepback.testing.parallel_agent import fake_llm, fake_tool, run_parallel_agent

        path = str(tmp_path / "par.sb")
        with record(path) as rec:
            run_parallel_agent(rec)

        trace = replay(path)
        econ = next(s for s in trace.recorded_steps if s.get("name") == "lookup_economics")
        trace.substitute(
            ToolOutputSubstitution(
                at_step=econ["step_id"],
                fake_response={"answer": "GDP 25T USD (revised)", "confidence": 0.85},
            )
        )
        result = trace.replay_forward(Executor(llm=fake_llm, tool=fake_tool))
        dirty = [sv for sv in result.steps if sv.dirty]
        assert len(dirty) >= 1

    def test_lookup_steps_present(self, tmp_path):
        from stepback import record, replay
        from stepback.testing.parallel_agent import run_parallel_agent

        path = str(tmp_path / "par.sb")
        with record(path) as rec:
            run_parallel_agent(rec)

        trace = replay(path)
        lookup_steps = [s for s in trace.recorded_steps if s.get("name", "").startswith("lookup_")]
        assert len(lookup_steps) == 3  # history, economics, politics


# ---------------------------------------------------------------------------
# batch_worker
# ---------------------------------------------------------------------------

class TestBatchWorkerExample:
    def test_record(self, tmp_path):
        from stepback import record
        from examples.batch_worker.agent import run_batch_agent

        path = str(tmp_path / "batch.sb")
        with record(path) as rec:
            run_batch_agent(rec)

        assert os.path.exists(path)

    def test_fifteen_steps(self, tmp_path):
        from stepback import record, replay
        from examples.batch_worker.agent import run_batch_agent

        path = str(tmp_path / "batch.sb")
        with record(path) as rec:
            run_batch_agent(rec)

        trace = replay(path)
        assert len(trace.recorded_steps) == 15  # 3 steps × 5 posts

    def test_substitution_makes_steps_dirty(self, tmp_path):
        from stepback import Executor, record, replay
        from stepback.substitutions import ToolOutputSubstitution
        from examples.batch_worker.agent import fake_llm, fake_tool, run_batch_agent

        path = str(tmp_path / "batch.sb")
        with record(path) as rec:
            run_batch_agent(rec)

        trace = replay(path)
        p2_llm = next(
            s for s in trace.recorded_steps
            if s.get("step_kind") == "llm_call"
            and "p2" in str(s.get("inputs", {}).get("messages", ""))
        )
        override_output = {
            "id": "chatcmpl-override",
            "model": "gpt-4o-2024-11-20",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": "Decision: approve. Ref: override"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        trace.substitute(
            ToolOutputSubstitution(at_step=p2_llm["step_id"], fake_response=override_output)
        )
        result = trace.replay_forward(Executor(llm=fake_llm, tool=fake_tool))
        dirty = [sv for sv in result.steps if sv.dirty]
        # Only p2's downstream steps become dirty; p1,p3,p4,p5 stay cached.
        assert len(dirty) >= 1
        assert len(dirty) < 15  # not all steps are dirty

    def test_only_substituted_post_dirty(self, tmp_path):
        """Verify that other posts' steps stay cached after substituting p2."""
        from stepback import Executor, record, replay
        from stepback.substitutions import ToolOutputSubstitution
        from examples.batch_worker.agent import fake_llm, fake_tool, run_batch_agent

        path = str(tmp_path / "batch.sb")
        with record(path) as rec:
            run_batch_agent(rec)

        trace = replay(path)
        p2_llm = next(
            s for s in trace.recorded_steps
            if s.get("step_kind") == "llm_call"
            and "p2" in str(s.get("inputs", {}).get("messages", ""))
        )
        override_output = {
            "id": "chatcmpl-override",
            "model": "gpt-4o-2024-11-20",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": "Decision: approve. Ref: override"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        trace.substitute(
            ToolOutputSubstitution(at_step=p2_llm["step_id"], fake_response=override_output)
        )
        result = trace.replay_forward(Executor(llm=fake_llm, tool=fake_tool))
        cached = [sv for sv in result.steps if sv.cache_hit]
        # p1, p3, p4, p5 all have 3 steps each = 12 cached minimum.
        assert len(cached) >= 12
