"""Tests for the dirty-set propagation contract surface.

Exercises the README §"The dirty-set algorithm" pseudocode lifted into
``stepback.divergence.compute_dirty_set`` as a contract-documented API.
"""
from __future__ import annotations

import pytest

from stepback import record, replay
from stepback.divergence import (
    DirtySetEntry,
    DirtySetSummary,
    compute_dirty_set,
)
from stepback.recorder import RecorderKey
from stepback.replay import Executor
from stepback.substitutions import (
    PromptSubstitution,
    ToolOutputSubstitution,
)
from stepback.testing import run_recorded_agent


@pytest.fixture()
def recorded_trace(tmp_path):
    p = str(tmp_path / "t.sb")
    key = RecorderKey.fresh()
    with record(p, key=key) as ctx:
        run_recorded_agent(ctx)
    return p, key


# ----------------------------- empty substitution


def test_compute_dirty_set_empty_substitution_yields_all_clean(recorded_trace):
    """Postcondition (P5) minimality: no substitution → no dirty step."""
    path, key = recorded_trace
    trace = replay(path, hmac_key=key.hmac_key)

    summary = compute_dirty_set(trace, [])

    assert isinstance(summary, DirtySetSummary)
    assert summary.step_count == len(trace.recorded_steps)
    assert summary.dirty_count == 0
    assert summary.clean_count == summary.step_count
    assert summary.calls_saved == summary.step_count
    assert all(isinstance(e, DirtySetEntry) for e in summary.entries)
    assert all(not e.dirty for e in summary.entries)
    assert all(e.cache_hit for e in summary.entries)
    assert all(e.dirty_reason is None for e in summary.entries)


def test_compute_dirty_set_none_substitution_is_identity(recorded_trace):
    """Passing ``None`` is equivalent to passing an empty list."""
    path, key = recorded_trace
    trace = replay(path, hmac_key=key.hmac_key)
    a = compute_dirty_set(trace, None)
    b = compute_dirty_set(trace, [])
    assert a.dirty_count == b.dirty_count == 0


# --------------------------- single substitution


def _first_step_of_kind(trace, kind: str) -> str:
    for s in trace.recorded_steps:
        if s["step_kind"] == kind:
            return s["step_id"]
    raise AssertionError(f"no step of kind {kind!r} in fixture trace")


def test_compute_dirty_set_substituted_step_marked_dirty(recorded_trace):
    """Postcondition (P3): the directly-substituted step is dirty."""
    path, key = recorded_trace
    trace = replay(path, hmac_key=key.hmac_key)
    target = _first_step_of_kind(trace, "llm_call")

    summary = compute_dirty_set(
        trace,
        [PromptSubstitution(at_step=target, new_messages=[{"role": "system", "content": "be paranoid"}])],
    )

    by_id = {e.step_id: e for e in summary.entries}
    assert by_id[target].dirty is True
    assert by_id[target].cache_hit is False
    assert by_id[target].dirty_reason in {"substituted", "input_drift"}


def test_compute_dirty_set_propagates_to_descendants(recorded_trace):
    """Postcondition (P2): every descendant of a dirty step is dirty."""
    path, key = recorded_trace
    trace = replay(path, hmac_key=key.hmac_key)
    target = _first_step_of_kind(trace, "llm_call")

    summary = compute_dirty_set(
        trace,
        [PromptSubstitution(at_step=target, new_messages=[{"role": "system", "content": "x"}])],
        executor=Executor(fallback_recorded=True),
    )

    # Find the index of the targeted step in topological order.
    order = [e.step_id for e in summary.entries]
    idx = order.index(target)

    # Build parent map and check transitive descendants are all dirty.
    parents_of = {e.step_id: set(e.parent_step_ids) for e in summary.entries}
    dirty_ids = set(summary.dirty_ids)

    descendants: set = set()
    frontier = {target}
    while frontier:
        nxt: set = set()
        for sid in order[idx + 1 :]:
            if parents_of[sid] & (frontier | descendants):
                nxt.add(sid)
        new = nxt - descendants
        if not new:
            break
        descendants.update(new)
        frontier = new

    # Every transitive descendant of the substituted step must be dirty.
    assert descendants.issubset(dirty_ids), (
        f"descendants {descendants - dirty_ids} not in dirty set {dirty_ids}"
    )
    # Steps strictly before the substitution are unaffected.
    for sid in order[:idx]:
        assert sid not in dirty_ids, f"upstream step {sid} unexpectedly dirty"


def test_compute_dirty_set_calls_saved_equals_clean_count(recorded_trace):
    """``calls_saved`` reports exactly the cache-hit population."""
    path, key = recorded_trace
    trace = replay(path, hmac_key=key.hmac_key)
    target = _first_step_of_kind(trace, "llm_call")

    summary = compute_dirty_set(
        trace,
        [PromptSubstitution(at_step=target, new_messages=[{"role": "system", "content": "x"}])],
    )
    assert summary.calls_saved == summary.clean_count
    assert summary.dirty_count + summary.clean_count == summary.step_count


def test_compute_dirty_set_unknown_step_id_is_inert(recorded_trace):
    """Precondition #3: substitutions targeting unknown step ids do not dirty anything."""
    path, key = recorded_trace
    trace = replay(path, hmac_key=key.hmac_key)

    summary = compute_dirty_set(
        trace,
        [PromptSubstitution(at_step="step:does_not_exist", new_messages=[])],
    )
    assert summary.dirty_count == 0


def test_compute_dirty_set_to_json_is_serializable(recorded_trace):
    """Summary serializes to plain JSON-able dicts."""
    import json

    path, key = recorded_trace
    trace = replay(path, hmac_key=key.hmac_key)
    summary = compute_dirty_set(trace, [])
    blob = json.dumps(summary.to_json(), sort_keys=True)
    assert "step_count" in blob
    assert "entries" in blob


def test_compute_dirty_set_tool_output_substitution_dirties_target(recorded_trace):
    """Output-forcing substitutions also dirty the targeted step."""
    path, key = recorded_trace
    trace = replay(path, hmac_key=key.hmac_key)
    target = _first_step_of_kind(trace, "tool_call")

    summary = compute_dirty_set(
        trace,
        [ToolOutputSubstitution(at_step=target, fake_response="synthetic")],
    )

    by_id = {e.step_id: e for e in summary.entries}
    assert by_id[target].dirty is True
    assert by_id[target].cache_hit is False


def test_compute_dirty_set_dirty_ids_in_topological_order(recorded_trace):
    """``entries`` preserves the recorded step order."""
    path, key = recorded_trace
    trace = replay(path, hmac_key=key.hmac_key)
    summary = compute_dirty_set(trace, [])
    expected = [s["step_id"] for s in trace.recorded_steps]
    actual = [e.step_id for e in summary.entries]
    assert actual == expected
