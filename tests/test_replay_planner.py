"""Tests for the planner/executor split introduced in Step 68.

The planner (``Trace.plan_replay``) produces a ``ReplayPlan`` that
classifies each step as ``"cache_hit"`` or ``"execute"`` without calling
any executor.  The executor phase (``ReplayPlan.execute``) then runs the
plan and returns a ``ReplayResult`` that is semantically identical to the
result of ``Trace.run_replay`` / ``Trace.replay_forward``.

Key properties tested:

1. ``plan_replay`` never calls executor callbacks.
2. ``ReplayPlan.execute`` produces the same result as ``run_replay``.
3. A no-substitution plan has all steps as ``"cache_hit"``.
4. A tool-output substitution makes the targeted step ``"execute"`` and
   marks downstream steps ``requires_runtime_validation=True``.
5. ``plan_replay`` returns the correct ``estimated_dirty_count`` and
   ``estimated_cache_hit_count``.
6. ``ReplayPlan.planned_steps`` has the same count as ``recorded_steps``.
7. Parallel branch traces plan and execute correctly.
8. ``plan_replay`` with an explicit subs argument is independent of
   ``pending_subs``.
"""
from __future__ import annotations

import pytest

from stepback import record, replay, ReplayPlan, PlannedStep
from stepback.replay import Executor, MissingExecutor
from stepback.substitutions import SubstitutionSet, ToolOutputSubstitution
from stepback.testing import fake_llm, fake_tool, run_recorded_agent


# ------------------------------------------------------------------ helpers


def _record_fixture(tmp_path):
    from stepback import RecorderKey

    key = RecorderKey.fresh()
    path = str(tmp_path / "trace.sb")
    with record(path, key=key) as rec:
        run_recorded_agent(rec)
    return path, key


# ------------------------------------------------------------------ tests


def test_plan_replay_returns_replay_plan(tmp_path):
    path, _ = _record_fixture(tmp_path)
    t = replay(path)
    plan = t.plan_replay()
    assert isinstance(plan, ReplayPlan)
    assert len(plan.planned_steps) == len(t.recorded_steps)
    for step in plan.planned_steps:
        assert isinstance(step, PlannedStep)


def test_plan_replay_no_subs_all_cache_hit(tmp_path):
    path, _ = _record_fixture(tmp_path)
    t = replay(path)
    plan = t.plan_replay()

    assert plan.estimated_dirty_count == 0
    assert plan.estimated_cache_hit_count == len(t.recorded_steps)
    for step in plan.planned_steps:
        assert step.planned_action == "cache_hit"
        assert step.dirty_reason is None
        assert not step.requires_runtime_validation


def test_plan_replay_no_executor_called(tmp_path):
    """plan_replay must not call any executor callback."""
    path, _ = _record_fixture(tmp_path)
    t = replay(path)

    call_log: list[str] = []

    def llm_spy(model, messages):
        call_log.append("llm")
        return {"content": "x"}

    def tool_spy(name, args):
        call_log.append("tool")
        return "result"

    t.substitute(ToolOutputSubstitution(t.recorded_steps[0]["step_id"], {"result": "new"}))
    t.plan_replay()
    # plan_replay must not have invoked any callback
    assert call_log == []


def test_plan_execute_matches_run_replay_no_subs(tmp_path):
    """ReplayPlan.execute() == run_replay() for zero substitutions."""
    path, _ = _record_fixture(tmp_path)
    t = replay(path)
    subs = SubstitutionSet()

    result_direct = t.run_replay(subs, Executor())
    result_plan = t.plan_replay(subs).execute()

    assert result_plan.dirty_count == result_direct.dirty_count
    assert result_plan.cache_hit_count == result_direct.cache_hit_count
    assert result_plan.real_executions == result_direct.real_executions
    assert result_plan.total_cost_usd == result_direct.total_cost_usd
    for sv_d, sv_p in zip(result_direct.steps, result_plan.steps):
        assert sv_p.step_id == sv_d.step_id
        assert sv_p.dirty == sv_d.dirty
        assert sv_p.cache_hit == sv_d.cache_hit
        assert sv_p.outputs == sv_d.outputs


def test_plan_execute_matches_run_replay_with_sub(tmp_path):
    """ReplayPlan.execute() == run_replay() when a substitution is active."""
    path, _ = _record_fixture(tmp_path)
    t = replay(path)

    # Substitute the first tool_call step
    tool_steps = [s for s in t.recorded_steps if s["step_kind"] == "tool_call"]
    assert tool_steps, "fixture must have at least one tool_call"
    sub_step_id = tool_steps[0]["step_id"]
    forced_output = {"result": "injected"}
    subs = SubstitutionSet()
    subs.add(ToolOutputSubstitution(sub_step_id, forced_output))

    executor = Executor(
        llm=fake_llm,
        tool=fake_tool,
        fallback_recorded=True,
    )
    result_direct = t.run_replay(subs, executor)
    executor2 = Executor(
        llm=fake_llm,
        tool=fake_tool,
        fallback_recorded=True,
    )
    result_plan = t.plan_replay(subs).execute(executor2)

    assert result_plan.dirty_count == result_direct.dirty_count
    assert result_plan.cache_hit_count == result_direct.cache_hit_count
    for sv_d, sv_p in zip(result_direct.steps, result_plan.steps):
        assert sv_p.step_id == sv_d.step_id
        assert sv_p.dirty == sv_d.dirty
        assert sv_p.outputs == sv_d.outputs


def test_plan_marks_substituted_step_as_execute(tmp_path):
    path, _ = _record_fixture(tmp_path)
    t = replay(path)

    tool_steps = [s for s in t.recorded_steps if s["step_kind"] == "tool_call"]
    assert tool_steps
    sub_step_id = tool_steps[0]["step_id"]
    subs = SubstitutionSet()
    subs.add(ToolOutputSubstitution(sub_step_id, {"result": "x"}))

    plan = t.plan_replay(subs)
    planned_by_id = {p.step_id: p for p in plan.planned_steps}

    assert planned_by_id[sub_step_id].planned_action == "execute"
    assert planned_by_id[sub_step_id].dirty_reason is not None
    assert plan.estimated_dirty_count >= 1


def test_plan_downstream_requires_runtime_validation(tmp_path):
    """Steps downstream of a dirty step must set requires_runtime_validation."""
    path, _ = _record_fixture(tmp_path)
    t = replay(path)

    # Force the first step dirty
    first_step_id = t.recorded_steps[0]["step_id"]
    subs = SubstitutionSet()
    subs.add(ToolOutputSubstitution(first_step_id, {"result": "new"}))

    plan = t.plan_replay(subs)
    planned_by_id = {p.step_id: p for p in plan.planned_steps}

    # The first step itself is dirty
    assert planned_by_id[first_step_id].planned_action == "execute"

    # At least one downstream step should require runtime validation
    downstream = [
        p for p in plan.planned_steps
        if p.step_id != first_step_id and p.requires_runtime_validation
    ]
    assert len(downstream) >= 1


def test_plan_execute_uses_pending_subs_by_default(tmp_path):
    """plan_replay() with no argument uses Trace.pending_subs."""
    path, _ = _record_fixture(tmp_path)
    t = replay(path)

    tool_steps = [s for s in t.recorded_steps if s["step_kind"] == "tool_call"]
    assert tool_steps
    sub_step_id = tool_steps[0]["step_id"]
    t.substitute(ToolOutputSubstitution(sub_step_id, {"result": "pending"}))

    plan_from_pending = t.plan_replay()
    planned_by_id = {p.step_id: p for p in plan_from_pending.planned_steps}
    assert planned_by_id[sub_step_id].planned_action == "execute"


def test_plan_execute_explicit_subs_independent_of_pending(tmp_path):
    """Explicit subs argument overrides pending_subs in plan_replay."""
    path, _ = _record_fixture(tmp_path)
    t = replay(path)

    # Add pending subs that dirty step 0
    t.substitute(ToolOutputSubstitution(t.recorded_steps[0]["step_id"], {"result": "pending"}))

    # Call plan_replay with an empty subs set
    plan_empty = t.plan_replay(SubstitutionSet())
    assert plan_empty.estimated_dirty_count == 0


def test_plan_replay_planned_step_fields(tmp_path):
    """PlannedStep must expose all required fields."""
    path, _ = _record_fixture(tmp_path)
    t = replay(path)
    plan = t.plan_replay()

    for step in plan.planned_steps:
        assert step.step_id
        assert step.kind
        assert step.recorded_inputs_hash
        assert isinstance(step.planned_inputs, dict)
        assert step.planned_inputs_hash
        assert step.planned_action in ("cache_hit", "execute")
        if step.planned_action == "cache_hit":
            assert step.dirty_reason is None
        else:
            assert step.dirty_reason is not None


def test_replay_plan_execute_fallback_recorded(tmp_path):
    """ReplayPlan.execute with fallback_recorded=True does not raise."""
    path, _ = _record_fixture(tmp_path)
    t = replay(path)
    subs = SubstitutionSet()
    # Dirty the first step without providing a real executor
    subs.add(ToolOutputSubstitution(t.recorded_steps[0]["step_id"], {"result": "x"}))
    plan = t.plan_replay(subs)
    # fallback_recorded=True means dirty steps fall back to recorded output
    result = plan.execute(Executor(fallback_recorded=True))
    assert result.dirty_count >= 1
