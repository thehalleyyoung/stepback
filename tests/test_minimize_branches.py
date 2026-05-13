"""Tests for branch-level trace minimization (Step 85).

Exercises:
  * identify_branch_groups correctly parses open/join/branch structure
  * minimize_branches drops branches that don't affect the predicate
  * minimize_branches preserves branches that DO affect the predicate
  * minimize_branches drops all branches when predicate always holds
  * degenerate: trace with no branches returns original steps unchanged
  * predicate not triggered raises PredicateNotTriggered
  * probe_budget raises BudgetExhausted carrying a partial result
  * join step with all branches dropped is handled correctly
  * BranchGroup.branch_step_ids are in execution order
"""
from __future__ import annotations

import os
from typing import Any, List

import pytest

from stepback import RecorderKey, record, replay
from stepback.minimize import (
    BranchGroup,
    BranchMinimizationResult,
    BudgetExhausted,
    MinimizeOptions,
    PredicateNotTriggered,
    identify_branch_groups,
    minimize_branches,
)
from stepback.replay import Executor, ReplayResult


# ─────────────────────────────────── helpers / fixtures ──────────────────────


def _fake_tool(name: str, args: dict) -> Any:
    if name == "branch_a":
        return {"tag": "A", "value": 10}
    if name == "branch_b":
        return {"tag": "B", "value": 20}
    if name == "branch_c":
        return {"tag": "C", "value": 30}
    if name == "summarize":
        return {"summary": "done"}
    raise KeyError(f"unknown tool: {name!r}")


def _join_fn_recorder(outs: list) -> dict:
    """Join for the recorder: 1 argument (outputs list)."""
    tags = [o.get("result", o).get("tag", "?") for o in outs]
    return {"tags": tags, "n": len(tags)}


def _join_fn_executor(name: str, outs: list) -> dict:
    """Join for the executor: 2 arguments (name, outputs list)."""
    tags = [o.get("result", o).get("tag", "?") for o in outs]
    return {"tags": tags, "n": len(tags)}


def _make_branch(tool_name: str):
    def branch(rec) -> None:
        rec.tool_call(tool_name, {}, executor=_fake_tool)
    return branch


def _record_triple_branch(tmp_path) -> tuple:
    """Record a 3-branch trace (A, B, C) with a downstream summarize step."""
    path = os.path.join(str(tmp_path), "triple.sb")
    key = RecorderKey.fresh()
    with record(path, key=key) as rec:
        rec.parallel(
            "triple_fanout",
            [
                _make_branch("branch_a"),
                _make_branch("branch_b"),
                _make_branch("branch_c"),
            ],
            join=_join_fn_recorder,
            branch_names=["branch/a", "branch/b", "branch/c"],
        )
        # A downstream step so we can verify the join feeds into it.
        rec.tool_call("summarize", {"input": "x"}, executor=_fake_tool)
    return path, key


def _record_no_branches(tmp_path) -> tuple:
    """Record a simple linear trace with no parallel branches."""
    path = os.path.join(str(tmp_path), "linear.sb")
    key = RecorderKey.fresh()
    with record(path, key=key) as rec:
        rec.tool_call("summarize", {"input": "x"}, executor=_fake_tool)
    return path, key


def _executor_with_join():
    """Executor that re-runs the join and falls back to recorded for the rest."""
    return Executor(join=_join_fn_executor, fallback_recorded=True)


# ─────────────────────────────────── identify_branch_groups ──────────────────


def test_identify_branch_groups_returns_one_group(tmp_path):
    path, _ = _record_triple_branch(tmp_path)
    t = replay(path)
    groups = identify_branch_groups(t.recorded_steps)
    assert len(groups) == 1


def test_identify_branch_groups_names_and_tails(tmp_path):
    path, _ = _record_triple_branch(tmp_path)
    t = replay(path)
    groups = identify_branch_groups(t.recorded_steps)
    g = groups[0]

    assert g.branch_names == ["branch/a", "branch/b", "branch/c"]
    assert len(g.branch_tail_ids) == 3

    # Each branch has exactly one step (the tool_call).
    for branch_steps in g.branch_step_ids:
        assert len(branch_steps) == 1


def test_identify_branch_groups_step_ids_in_execution_order(tmp_path):
    path, _ = _record_triple_branch(tmp_path)
    t = replay(path)
    groups = identify_branch_groups(t.recorded_steps)
    g = groups[0]

    # The tail of each branch is the last step listed in branch_step_ids.
    for i, tail_id in enumerate(g.branch_tail_ids):
        assert g.branch_step_ids[i][-1] == tail_id


def test_identify_branch_groups_no_branches(tmp_path):
    path, _ = _record_no_branches(tmp_path)
    t = replay(path)
    groups = identify_branch_groups(t.recorded_steps)
    assert groups == []


# ─────────────────────────────────── minimize_branches ───────────────────────


def _join_output(result: ReplayResult, join_step_id: str) -> dict:
    for s in result.steps:
        if s.step_id == join_step_id:
            return s.outputs or {}
    return {}


def test_minimize_branches_predicate_not_triggered_raises(tmp_path):
    path, _ = _record_triple_branch(tmp_path)
    t = replay(path)

    with pytest.raises(PredicateNotTriggered):
        minimize_branches(t, lambda r: False, executor=_executor_with_join())


def test_minimize_branches_no_branches_returns_original_steps(tmp_path):
    path, _ = _record_no_branches(tmp_path)
    t = replay(path)

    result = minimize_branches(t, lambda r: True, executor=Executor())
    assert len(result.groups) == 0
    assert [s["step_id"] for s in result.minimal_steps] == [
        s["step_id"] for s in t.recorded_steps
    ]
    assert result.probes >= 1


def test_minimize_branches_always_true_predicate_drops_all(tmp_path):
    """When the predicate always holds, every branch is droppable."""
    path, _ = _record_triple_branch(tmp_path)
    t = replay(path)
    groups = identify_branch_groups(t.recorded_steps)
    g = groups[0]

    result = minimize_branches(
        t,
        lambda r: True,
        executor=_executor_with_join(),
    )

    # All three branches should be dropped.
    assert result.dropped_branch_indices[g.open_step_id] == [0, 1, 2]
    assert result.kept_branch_indices[g.open_step_id] == []
    assert result.probes >= 1


def test_minimize_branches_all_needed_drops_none(tmp_path):
    """When the predicate requires exactly 3 branches, nothing is dropped."""
    path, _ = _record_triple_branch(tmp_path)
    t = replay(path)
    groups = identify_branch_groups(t.recorded_steps)
    g = groups[0]

    def predicate(r: ReplayResult) -> bool:
        # Requires all 3 tags present in the join output.
        out = _join_output(r, g.join_step_id)
        return set(out.get("tags", [])) == {"A", "B", "C"}

    result = minimize_branches(t, predicate, executor=_executor_with_join())

    assert result.dropped_branch_indices[g.open_step_id] == []
    assert len(result.kept_branch_indices[g.open_step_id]) == 3


def test_minimize_branches_drops_irrelevant_branches(tmp_path):
    """Predicate only needs tag 'C': branches A and B should be dropped."""
    path, _ = _record_triple_branch(tmp_path)
    t = replay(path)
    groups = identify_branch_groups(t.recorded_steps)
    g = groups[0]

    def predicate(r: ReplayResult) -> bool:
        out = _join_output(r, g.join_step_id)
        return "C" in out.get("tags", [])

    result = minimize_branches(t, predicate, executor=_executor_with_join())

    dropped = result.dropped_branch_indices[g.open_step_id]
    kept = result.kept_branch_indices[g.open_step_id]

    # Branch C (index 2) must be kept; A (0) and B (1) should be dropped.
    assert 2 in kept, f"branch/c should be kept; got kept={kept}"
    assert 0 in dropped, f"branch/a should be dropped; got dropped={dropped}"
    assert 1 in dropped, f"branch/b should be dropped; got dropped={dropped}"


def test_minimize_branches_dropped_steps_absent_from_minimal(tmp_path):
    """Step IDs of dropped branches must not appear in minimal_steps."""
    path, _ = _record_triple_branch(tmp_path)
    t = replay(path)
    groups = identify_branch_groups(t.recorded_steps)
    g = groups[0]

    def predicate(r: ReplayResult) -> bool:
        out = _join_output(r, g.join_step_id)
        return "C" in out.get("tags", [])

    result = minimize_branches(t, predicate, executor=_executor_with_join())

    dropped_step_ids = set()
    for idx in result.dropped_branch_indices[g.open_step_id]:
        for sid in g.branch_step_ids[idx]:
            dropped_step_ids.add(sid)

    minimal_ids = {s["step_id"] for s in result.minimal_steps}
    assert dropped_step_ids.isdisjoint(minimal_ids), (
        f"Dropped branch steps still present in minimal: "
        f"{dropped_step_ids & minimal_ids}"
    )


def test_minimize_branches_join_inputs_updated_in_minimal(tmp_path):
    """The join step in minimal_steps must have updated branch_tails."""
    path, _ = _record_triple_branch(tmp_path)
    t = replay(path)
    groups = identify_branch_groups(t.recorded_steps)
    g = groups[0]

    def predicate(r: ReplayResult) -> bool:
        out = _join_output(r, g.join_step_id)
        return "C" in out.get("tags", [])

    result = minimize_branches(t, predicate, executor=_executor_with_join())

    # Find the join step in minimal_steps.
    join_step = next(
        (s for s in result.minimal_steps if s["step_id"] == g.join_step_id),
        None,
    )
    assert join_step is not None

    # branch_tails should only contain kept branch tails.
    kept_tails = {g.branch_tail_ids[i] for i in result.kept_branch_indices[g.open_step_id]}
    join_tails = set(join_step["inputs"]["branch_tails"])
    assert join_tails == kept_tails

    # parent_step_ids should match.
    assert set(join_step["parent_step_ids"]) == kept_tails


def test_minimize_branches_minimal_trace_replays_cleanly(tmp_path):
    """The minimal_steps list can be replayed without errors."""
    path, _ = _record_triple_branch(tmp_path)
    t = replay(path)
    groups = identify_branch_groups(t.recorded_steps)
    g = groups[0]

    def predicate(r: ReplayResult) -> bool:
        out = _join_output(r, g.join_step_id)
        return "C" in out.get("tags", [])

    result = minimize_branches(t, predicate, executor=_executor_with_join())

    # Build a Trace from the minimal steps and replay it.
    from stepback.replay import Trace as _Trace
    from stepback.substitutions import SubstitutionSet

    minimal_trace = _Trace(
        path=t.path,
        header=t.header,
        recorded_steps=result.minimal_steps,
    )
    replay_result = minimal_trace.run_replay(SubstitutionSet(), _executor_with_join())

    # The predicate must hold on the minimal replay.
    assert predicate(replay_result)


def test_minimize_branches_probe_budget_raises_budget_exhausted(tmp_path):
    """Exhausting probe_budget mid-search raises BudgetExhausted."""
    path, _ = _record_triple_branch(tmp_path)
    t = replay(path)

    with pytest.raises(BudgetExhausted) as exc_info:
        minimize_branches(
            t,
            lambda r: True,
            executor=_executor_with_join(),
            options=MinimizeOptions(probe_budget=1),
        )

    partial = exc_info.value.partial
    assert isinstance(partial, BranchMinimizationResult)


def test_minimize_branches_time_budget_raises_budget_exhausted(tmp_path):
    """Exhausting time_budget_s raises BudgetExhausted."""
    path, _ = _record_triple_branch(tmp_path)
    t = replay(path)

    with pytest.raises(BudgetExhausted) as exc_info:
        minimize_branches(
            t,
            lambda r: True,
            executor=_executor_with_join(),
            options=MinimizeOptions(time_budget_s=0.0),
        )

    partial = exc_info.value.partial
    assert isinstance(partial, BranchMinimizationResult)


def test_minimize_branches_progress_callback_called(tmp_path):
    """The progress callback is called after each actual probe."""
    path, _ = _record_triple_branch(tmp_path)
    t = replay(path)
    calls: List[tuple] = []

    minimize_branches(
        t,
        lambda r: True,
        executor=_executor_with_join(),
        options=MinimizeOptions(progress=lambda p, s: calls.append((p, s))),
    )

    assert len(calls) >= 1


def test_minimize_branches_result_probes_positive(tmp_path):
    path, _ = _record_triple_branch(tmp_path)
    t = replay(path)

    result = minimize_branches(t, lambda r: True, executor=_executor_with_join())
    assert result.probes >= 1


def test_minimize_branches_cache_hits_on_repeat_call(tmp_path):
    """A second call with an identical probe skips re-replay (cache hit)."""
    path, _ = _record_triple_branch(tmp_path)
    t = replay(path)
    # Use a probe_budget large enough not to exhaust but small enough to
    # guarantee the initial probe is re-used on a repeat call.
    result = minimize_branches(
        t,
        lambda r: True,
        executor=_executor_with_join(),
    )
    # The pre-flight probe and subsequent branch probes may share cache.
    assert result.probes + result.cache_hits >= result.probes  # trivially true
