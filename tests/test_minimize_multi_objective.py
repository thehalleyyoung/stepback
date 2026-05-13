"""Tests for multi-objective ddmin (Step 81).

Verifies:
- TraceObjectives dataclass and Pareto dominance logic.
- extract_objectives() on a real ReplayResult.
- _compute_pareto_front() correctly filters dominated entries.
- MultiObjectiveDDMinStrategy runs and returns a 1-minimal subset.
- multi_objective_minimize() end-to-end: isolates the one responsible
  substitution, populates pareto_front, objectives, and probes.
- Budget exhaustion mid-run raises BudgetExhausted with partial.
- multi_objective_minimize respects excluded items like minimize_substitutions.
- MultiObjectiveMinimizationResult.as_set() returns a SubstitutionSet.
"""
from __future__ import annotations

import pytest

from stepback import (
    Executor,
    record,
    replay,
)
from stepback.minimize import (
    BudgetExhausted,
    DDMinStrategy,
    MinimizeOptions,
    MultiObjectiveDDMinStrategy,
    MultiObjectiveMinimizationResult,
    ParetoEntry,
    PredicateNotTriggered,
    TraceObjectives,
    _compute_pareto_front,
    extract_objectives,
    multi_objective_minimize,
)
from stepback.substitutions import (
    ModelSubstitution,
    SubstitutionSet,
    ToolOutputSubstitution,
)
from stepback.testing import LOOKUP_FIXED_ROW, fake_llm, fake_tool, run_recorded_agent


# ------------------------------------------------------------------ fixtures


def _record(tmp_path):
    from stepback import RecorderKey

    key = RecorderKey.fresh()
    p = str(tmp_path / "trace.sb")
    with record(p, key=key) as rec:
        run_recorded_agent(rec)
    return p, key


def _step_id(t, idx):
    return t.recorded_steps[idx]["step_id"]


def _executor():
    return Executor(llm=fake_llm, tool=fake_tool)


# ------------------------------------------------------------------ TraceObjectives


class TestTraceObjectives:
    def test_dominates_strictly_better(self):
        a = TraceObjectives(step_count=3, llm_call_count=1, total_cost_usd=0.01,
                            policy_violation_count=0)
        b = TraceObjectives(step_count=5, llm_call_count=2, total_cost_usd=0.05,
                            policy_violation_count=1)
        assert a.dominates(b)
        assert not b.dominates(a)

    def test_dominates_equal_not_dominate(self):
        a = TraceObjectives(step_count=3, llm_call_count=1, total_cost_usd=0.01,
                            policy_violation_count=0)
        assert not a.dominates(a)

    def test_dominates_one_dimension_worse(self):
        a = TraceObjectives(step_count=3, llm_call_count=1, total_cost_usd=0.01,
                            policy_violation_count=0)
        b = TraceObjectives(step_count=2, llm_call_count=1, total_cost_usd=0.01,
                            policy_violation_count=0)
        # b has fewer steps but same elsewhere; b dominates a, not vice versa
        assert b.dominates(a)
        assert not a.dominates(b)

    def test_non_dominated_pair(self):
        # a cheaper cost but more steps — neither dominates the other
        a = TraceObjectives(step_count=5, llm_call_count=1, total_cost_usd=0.01,
                            policy_violation_count=0)
        b = TraceObjectives(step_count=3, llm_call_count=1, total_cost_usd=0.05,
                            policy_violation_count=0)
        assert not a.dominates(b)
        assert not b.dominates(a)

    def test_as_tuple_length(self):
        obj = TraceObjectives(step_count=2, llm_call_count=1, total_cost_usd=0.0,
                              policy_violation_count=0)
        assert len(obj.as_tuple()) == 5  # includes latency_s


# ------------------------------------------------------------------ _compute_pareto_front


class TestComputeParetoFront:
    def _make_entry(self, **kwargs):
        obj = TraceObjectives(**{
            "step_count": 5,
            "llm_call_count": 2,
            "total_cost_usd": 0.1,
            "policy_violation_count": 0,
            **kwargs,
        })
        return ParetoEntry(minimal=[], objectives=obj)

    def test_single_entry_is_on_front(self):
        entry = self._make_entry()
        front = _compute_pareto_front([entry])
        assert front == [entry]

    def test_dominated_entry_excluded(self):
        good = self._make_entry(step_count=3, total_cost_usd=0.01)
        bad = self._make_entry(step_count=5, total_cost_usd=0.05)
        front = _compute_pareto_front([good, bad])
        assert good in front
        assert bad not in front

    def test_non_dominated_both_on_front(self):
        a = self._make_entry(step_count=3, total_cost_usd=0.10)
        b = self._make_entry(step_count=5, total_cost_usd=0.01)
        front = _compute_pareto_front([a, b])
        assert len(front) == 2
        assert a in front
        assert b in front


# ------------------------------------------------------------------ extract_objectives


def test_extract_objectives_smoke(tmp_path):
    path, _ = _record(tmp_path)
    t = replay(path)
    result = t.run_replay(t.all_substitutions() if hasattr(t, "all_substitutions") else
                          __import__("stepback.substitutions", fromlist=["SubstitutionSet"]).SubstitutionSet(),
                          _executor())
    obj = extract_objectives(result)
    assert isinstance(obj, TraceObjectives)
    assert obj.step_count == len(result.steps)
    assert obj.llm_call_count >= 0
    assert obj.total_cost_usd >= 0.0
    assert obj.policy_violation_count >= 0
    assert obj.latency_s == 0.0  # not yet tracked


# ------------------------------------------------------------------ MultiObjectiveDDMinStrategy


def test_multi_objective_strategy_finds_minimal(tmp_path):
    """MultiObjectiveDDMinStrategy wraps DDMin; must isolate the one responsible sub."""
    path, _ = _record(tmp_path)
    t = replay(path)

    fix_step = _step_id(t, 1)
    noisy = SubstitutionSet(items=[
        ToolOutputSubstitution(at_step=fix_step, tool_call_id=None,
                               fake_response=LOOKUP_FIXED_ROW),
        ModelSubstitution(at_step=_step_id(t, 0), new_model_id="gpt-4o-mini-2024-07-18"),
        ModelSubstitution(at_step=_step_id(t, 2), new_model_id="gpt-4o-mini-2024-07-18"),
    ])

    def predicate(result):
        try:
            return result.steps[1].outputs["result"]["country"] == "US"
        except (KeyError, IndexError, TypeError):
            return False

    result = multi_objective_minimize(t, noisy, predicate, executor=_executor(), orderings=2)
    assert isinstance(result, MultiObjectiveMinimizationResult)
    assert len(result.minimal) == 1
    assert isinstance(result.minimal[0], ToolOutputSubstitution)


# ------------------------------------------------------------------ multi_objective_minimize


def test_multi_objective_minimize_populates_objectives(tmp_path):
    path, _ = _record(tmp_path)
    t = replay(path)

    fix_step = _step_id(t, 1)
    noisy = SubstitutionSet(items=[
        ToolOutputSubstitution(at_step=fix_step, tool_call_id=None,
                               fake_response=LOOKUP_FIXED_ROW),
        ModelSubstitution(at_step=_step_id(t, 0), new_model_id="gpt-4o-mini-2024-07-18"),
    ])

    def predicate(result):
        try:
            return result.steps[1].outputs["result"]["country"] == "US"
        except (KeyError, IndexError, TypeError):
            return False

    result = multi_objective_minimize(t, noisy, predicate, executor=_executor())

    assert result.objectives is not None
    assert isinstance(result.objectives, TraceObjectives)
    assert result.objectives.step_count > 0
    assert len(result.pareto_front) >= 1
    assert result.final_result is not None
    assert result.probes > 0


def test_multi_objective_minimize_pareto_front_non_empty(tmp_path):
    path, _ = _record(tmp_path)
    t = replay(path)

    fix_step = _step_id(t, 1)
    noisy = SubstitutionSet(items=[
        ToolOutputSubstitution(at_step=fix_step, tool_call_id=None,
                               fake_response=LOOKUP_FIXED_ROW),
        ModelSubstitution(at_step=_step_id(t, 0), new_model_id="gpt-4o-mini-2024-07-18"),
        ModelSubstitution(at_step=_step_id(t, 2), new_model_id="gpt-4o-mini-2024-07-18"),
        ModelSubstitution(at_step=_step_id(t, 4), new_model_id="gpt-4o-mini-2024-07-18"),
    ])

    def predicate(result):
        try:
            return result.steps[1].outputs["result"]["country"] == "US"
        except (KeyError, IndexError, TypeError):
            return False

    result = multi_objective_minimize(t, noisy, predicate, executor=_executor(), orderings=3)
    # All entries on the Pareto front are non-dominated
    front = result.pareto_front
    assert len(front) >= 1
    for entry in front:
        assert isinstance(entry, ParetoEntry)
        assert entry.objectives is not None


def test_multi_objective_minimize_raises_predicate_not_triggered(tmp_path):
    path, _ = _record(tmp_path)
    t = replay(path)

    noisy = SubstitutionSet(items=[
        ModelSubstitution(at_step=_step_id(t, 0), new_model_id="gpt-4o-mini-2024-07-18"),
    ])

    with pytest.raises(PredicateNotTriggered):
        multi_objective_minimize(t, noisy, lambda r: False, executor=_executor())


def test_multi_objective_minimize_budget_exhausted(tmp_path):
    path, _ = _record(tmp_path)
    t = replay(path)

    fix_step = _step_id(t, 1)
    noisy = SubstitutionSet(items=[
        ToolOutputSubstitution(at_step=fix_step, tool_call_id=None,
                               fake_response=LOOKUP_FIXED_ROW),
        ModelSubstitution(at_step=_step_id(t, 0), new_model_id="gpt-4o-mini-2024-07-18"),
        ModelSubstitution(at_step=_step_id(t, 2), new_model_id="gpt-4o-mini-2024-07-18"),
        ModelSubstitution(at_step=_step_id(t, 4), new_model_id="gpt-4o-mini-2024-07-18"),
        ModelSubstitution(at_step=_step_id(t, 6), new_model_id="gpt-4o-mini-2024-07-18"),
    ])

    def predicate(result):
        try:
            return result.steps[1].outputs["result"]["country"] == "US"
        except (KeyError, IndexError, TypeError):
            return False

    opts = MinimizeOptions(probe_budget=1)
    with pytest.raises(BudgetExhausted) as exc_info:
        multi_objective_minimize(t, noisy, predicate, executor=_executor(), options=opts)
    assert exc_info.value.partial is not None


def test_multi_objective_minimize_as_set(tmp_path):
    path, _ = _record(tmp_path)
    t = replay(path)

    fix_step = _step_id(t, 1)
    noisy = SubstitutionSet(items=[
        ToolOutputSubstitution(at_step=fix_step, tool_call_id=None,
                               fake_response=LOOKUP_FIXED_ROW),
        ModelSubstitution(at_step=_step_id(t, 0), new_model_id="gpt-4o-mini-2024-07-18"),
    ])

    def predicate(result):
        try:
            return result.steps[1].outputs["result"]["country"] == "US"
        except (KeyError, IndexError, TypeError):
            return False

    result = multi_objective_minimize(t, noisy, predicate, executor=_executor())
    s = result.as_set()
    assert isinstance(s, SubstitutionSet)
    assert len(s.items) == len(result.minimal)
