"""Mutation-contract tests for ``Trace`` and friends.

Pin the documented behavior in ``docs/trace-mutation.md``: which methods
mutate the receiver (``goto``, ``step_back``, ``step_forward``,
``substitute``, ``reset_substitutions``, ``bisect``) and which return
fresh objects without mutation (``branch_at``, ``replay_forward``,
``compare_branches``, ``minimize``, top-level ``sweep_traces``).

These tests exist to catch any future refactor that silently inverts
the mutation semantics — e.g. accidentally making ``step_back`` return
a deep copy of the trace, or making ``replay_forward`` clear
``pending_subs``. Both would be silent breakages of the v0.1 fluent
chaining UX.
"""
from __future__ import annotations

import copy

import pytest

from stepback import RecorderKey, record, replay
from stepback.replay import Branch, Executor, ReplayResult, Trace
from stepback.substitutions import (
    SubstitutionSet,
    ToolOutputSubstitution,
)
from stepback.testing import run_recorded_agent


@pytest.fixture
def fixture_trace(tmp_path) -> Trace:
    key = RecorderKey.fresh()
    path = str(tmp_path / "trace.sb")
    with record(path, key=key) as rec:
        run_recorded_agent(rec)
    return replay(path, hmac_key=key.hmac_key)


# ----------------------------------------------------- navigation


def test_goto_mutates_cursor_and_returns_self(fixture_trace):
    t = fixture_trace
    assert t.cursor == 0
    ret = t.goto("step:5")
    assert ret is t
    assert t.current_step()["step_id"] == "step:5"


def test_step_back_no_arg_decrements_cursor_and_returns_self(fixture_trace):
    t = fixture_trace
    t.goto("step:5")
    cursor_at_5 = t.cursor
    ret = t.step_back()
    assert ret is t
    assert t.cursor == cursor_at_5 - 1


def test_step_back_with_to_jumps_and_returns_self(fixture_trace):
    t = fixture_trace
    ret = t.step_back(to="step:7")
    assert ret is t
    assert t.current_step()["step_id"] == "step:7"


def test_step_back_floors_at_zero(fixture_trace):
    t = fixture_trace
    assert t.cursor == 0
    t.step_back()
    assert t.cursor == 0


def test_step_forward_caps_at_last(fixture_trace):
    t = fixture_trace
    last = len(t.recorded_steps) - 1
    t.goto(t.recorded_steps[last]["step_id"])
    t.step_forward()
    assert t.cursor == last


# ---------------------------------------------------- substitution


def test_substitute_mutates_pending_subs_and_returns_self(fixture_trace):
    t = fixture_trace
    sub = ToolOutputSubstitution("step:2", fake_response={})
    assert len(t.pending_subs.items) == 0
    ret = t.substitute(sub)
    assert ret is t
    assert len(t.pending_subs.items) == 1


def test_reset_substitutions_clears_and_returns_self(fixture_trace):
    t = fixture_trace
    t.substitute(ToolOutputSubstitution("step:2", fake_response={}))
    assert len(t.pending_subs.items) == 1
    ret = t.reset_substitutions()
    assert ret is t
    assert len(t.pending_subs.items) == 0


def test_replay_forward_does_not_mutate_pending_subs_or_cursor(fixture_trace):
    t = fixture_trace
    t.substitute(ToolOutputSubstitution("step:2", fake_response={}))
    t.goto("step:3")
    cursor_before = t.cursor
    subs_before = list(t.pending_subs.items)
    result = t.replay_forward(executor=Executor(fallback_recorded=True))
    assert isinstance(result, ReplayResult)
    assert t.cursor == cursor_before
    assert list(t.pending_subs.items) == subs_before


def test_replay_forward_returns_fresh_result_each_call(fixture_trace):
    t = fixture_trace
    r1 = t.replay_forward(executor=Executor(fallback_recorded=True))
    r2 = t.replay_forward(executor=Executor(fallback_recorded=True))
    assert r1 is not r2
    assert len(r1) == len(r2)


# --------------------------------------------------------- branch


def test_branch_at_returns_fresh_branch_no_trace_mutation(fixture_trace):
    t = fixture_trace
    cursor_before = t.cursor
    subs_before = list(t.pending_subs.items)
    b = t.branch_at("step:3", name="alt")
    assert isinstance(b, Branch)
    assert b.name == "alt"
    assert b.base_step == "step:3"
    assert b._owner is t
    assert t.cursor == cursor_before
    assert list(t.pending_subs.items) == subs_before


def test_two_branches_have_independent_substitution_sets(fixture_trace):
    t = fixture_trace
    a = t.branch_at("step:3", name="a")
    b = t.branch_at("step:3", name="b")
    a.substitute(ToolOutputSubstitution("step:5", fake_response={}))
    assert len(a.substitutions.items) == 1
    assert len(b.substitutions.items) == 0
    assert len(t.pending_subs.items) == 0


def test_branch_substitute_mutates_branch_returns_self(fixture_trace):
    t = fixture_trace
    b = t.branch_at("step:3", name="x")
    ret = b.substitute(ToolOutputSubstitution("step:5", fake_response={}))
    assert ret is b
    assert len(b.substitutions.items) == 1


def test_branch_replay_forward_caches_result_field(fixture_trace):
    t = fixture_trace
    b = t.branch_at("step:3", name="x")
    assert b.result is None
    r = b.replay_forward(executor=Executor(fallback_recorded=True))
    assert b.result is r
    r2 = b.replay_forward(executor=Executor(fallback_recorded=True))
    assert b.result is r2
    assert r2 is not r


# ----------------------------------------------------- bisect


def test_bisect_updates_last_bisect_probes_only(fixture_trace):
    t = fixture_trace
    cursor_before = t.cursor
    subs_before = list(t.pending_subs.items)
    assert t.last_bisect_probes == 0

    first = t.recorded_steps[0]["step_id"]
    last = t.recorded_steps[-1]["step_id"]
    found = t.bisect(first, last, predicate=lambda sv: True)

    assert found is not None
    assert t.last_bisect_probes >= 1
    assert t.cursor == cursor_before
    assert list(t.pending_subs.items) == subs_before


# ---------------------------------------------------- minimize


def test_minimize_does_not_mutate_trace(fixture_trace):
    t = fixture_trace
    cursor_before = t.cursor
    subs_before = list(t.pending_subs.items)
    sub = ToolOutputSubstitution("step:2", fake_response={})
    candidate = SubstitutionSet(items=[sub])
    t.minimize(
        candidate,
        predicate=lambda r: r.dirty_count >= 1,
        executor=Executor(fallback_recorded=True),
    )
    assert t.cursor == cursor_before
    assert list(t.pending_subs.items) == subs_before


# ----------------------------------------------------- sweep


def test_sweep_traces_does_not_mutate_input_paths_list(tmp_path):
    from stepback.sweep import sweep_traces

    key = RecorderKey.fresh()
    path = str(tmp_path / "trace.sb")
    with record(path, key=key) as rec:
        run_recorded_agent(rec)

    paths = [path]
    paths_snapshot = copy.deepcopy(paths)
    report = sweep_traces(
        paths,
        substitutions=[ToolOutputSubstitution("step:2", fake_response={})],
        base_step="step:2",
    )
    assert paths == paths_snapshot
    assert report is not None


# ------------------------------ chained-mutation idiom (smoke)


def test_fluent_chain_returns_same_trace_throughout(fixture_trace):
    t = fixture_trace
    chained = (
        t.step_back(to="step:3")
        .substitute(ToolOutputSubstitution("step:3", fake_response={}))
        .reset_substitutions()
    )
    assert chained is t
    assert t.current_step()["step_id"] == "step:3"
    assert len(t.pending_subs.items) == 0
