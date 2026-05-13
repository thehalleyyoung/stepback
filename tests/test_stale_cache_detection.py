"""Tests for Step 61: stale-cache detection via output-change propagation.

A step whose parent was dirty (re-executed) but produced an identical output
must NOT be forced dirty by the replay engine.  The hash-rebinding performed
on the ``context`` field already reflects any real output change; if the hash
is the same, the downstream step's ``current_inputs_hash`` equals its
``recorded_inputs_hash`` and the cache is validly reused.

Scenarios covered:

1. Parent dirty, same output → downstream is a cache hit, ``dirty=False``.
2. Parent dirty, different output → downstream is dirty (via context rebinding).
3. ``output_changed`` field on StepView is False for a cache-hit step.
4. ``output_changed`` field on StepView is True when dirty and output differs.
5. ``output_changed`` is False when step is dirty but produces the same output
   as recorded (i.e. dirty but unchanged output).
6. Grandparent dirty with same output, parent cache-hit → grandchild cache-hit.
7. Parent dirty same output; step lacks context field → downstream cache-hit.
8. ``compute_dirty_set`` agrees with the replay engine on stale-cache semantics.
9. No-substitution baseline: all ``output_changed`` fields are False.
10. Parent dirty different output → output_changed propagates transitively.
"""
from __future__ import annotations

import copy
import tempfile
from typing import Any

import pytest

from stepback import record, replay
from stepback.canonical import hash_obj
from stepback.divergence import compute_dirty_set
from stepback.recorder import RecorderKey
from stepback.replay import Executor, StepView
from stepback.substitutions import SubstitutionSet, ToolOutputSubstitution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_a(name: str, args: dict) -> Any:
    return {"value": "alpha", "count": 1}


def _tool_b(name: str, args: dict) -> Any:
    return {"value": "beta"}


def _tool_c(name: str, args: dict) -> Any:
    return {"value": "gamma"}


def _build_two_step_trace(path: str) -> None:
    """Step A (tool_call) → Step B (tool_call, context from A)."""
    with record(path, key=RecorderKey.fresh()) as rec:
        rec.tool_call("step_a", {}, _tool_a)
        rec.tool_call("step_b", {}, _tool_b)


def _build_three_step_chain(path: str) -> None:
    """A → B → C (each passes context forward)."""
    with record(path, key=RecorderKey.fresh()) as rec:
        rec.tool_call("step_a", {}, _tool_a)
        rec.tool_call("step_b", {}, _tool_b)
        rec.tool_call("step_c", {}, _tool_c)


def _build_no_context_trace(path: str) -> None:
    """Step A → Step B where B opts out of context (context_from_parent=False)."""
    with record(path, key=RecorderKey.fresh()) as rec:
        rec.tool_call("step_a", {}, _tool_a)
        rec.tool_call("step_b", {}, _tool_b, context_from_parent=False)


def _same_output_sub(rec_step: dict) -> ToolOutputSubstitution:
    """Return a substitution that forces step to produce its *recorded* output.

    ToolOutputSubstitution.force_output wraps the fake_response in
    ``{"result": fake_response}``, so we pass the inner ``"result"`` value
    from the recorded output to get an identical final output.
    """
    inner = rec_step["outputs"].get("result", rec_step["outputs"])
    return ToolOutputSubstitution(rec_step["step_id"], inner)


# ---------------------------------------------------------------------------
# Scenario 1 & 5: parent dirty, same output → downstream is a cache hit
# ---------------------------------------------------------------------------

def test_parent_dirty_same_output_downstream_is_cache_hit(tmp_path):
    """Step A is forced with its own recorded output (identical hash).
    Step B must be a cache hit even though its parent is dirty."""
    path = str(tmp_path / "trace.sb")
    _build_two_step_trace(path)
    t = replay(path)

    rec_a = t.recorded_steps[0]
    subs = SubstitutionSet([_same_output_sub(rec_a)])
    result = t.run_replay(subs, Executor())

    step_a: StepView = result.steps[0]
    step_b: StepView = result.steps[1]

    assert step_a.dirty is True, "step A must be dirty (forced by substitution)"
    assert step_a.output_changed is False, "step A output did not change"
    assert step_b.dirty is False, "step B must be a cache hit (parent output unchanged)"
    assert step_b.cache_hit is True
    assert result.cache_hit_count == 1  # only step B


# ---------------------------------------------------------------------------
# Scenario 2 & 4: parent dirty, different output → downstream is dirty
# ---------------------------------------------------------------------------

def test_parent_dirty_different_output_downstream_is_dirty(tmp_path):
    """Step A is forced with a *different* output.  Step B's context is rebound
    to the new hash, so B must be dirty."""
    path = str(tmp_path / "trace.sb")
    _build_two_step_trace(path)
    t = replay(path)

    rec_a = t.recorded_steps[0]
    different_inner = {"value": "CHANGED", "count": 99}  # differs from _tool_a result
    subs = SubstitutionSet([ToolOutputSubstitution(rec_a["step_id"], different_inner)])
    result = t.run_replay(subs, Executor(tool=lambda n, a: _tool_b(n, a)))

    step_a: StepView = result.steps[0]
    step_b: StepView = result.steps[1]

    assert step_a.dirty is True
    assert step_a.output_changed is True, "step A output changed"
    assert step_b.dirty is True, "step B must be dirty (parent output changed)"
    assert step_b.cache_hit is False
    assert result.dirty_count == 2


# ---------------------------------------------------------------------------
# Scenario 3: output_changed is False for cache-hit steps
# ---------------------------------------------------------------------------

def test_output_changed_false_for_cache_hit_step(tmp_path):
    """Cache-hit steps always have output_changed == False."""
    path = str(tmp_path / "trace.sb")
    _build_two_step_trace(path)
    t = replay(path)

    rec_a = t.recorded_steps[0]
    subs = SubstitutionSet([_same_output_sub(rec_a)])
    result = t.run_replay(subs, Executor())

    # step B is a cache hit → output_changed must be False.
    assert result.steps[1].cache_hit is True
    assert result.steps[1].output_changed is False


# ---------------------------------------------------------------------------
# Scenario 4 (positive): output_changed is True when output differs
# ---------------------------------------------------------------------------

def test_output_changed_true_when_dirty_and_content_differs(tmp_path):
    path = str(tmp_path / "trace.sb")
    _build_two_step_trace(path)
    t = replay(path)

    rec_a = t.recorded_steps[0]
    subs = SubstitutionSet([ToolOutputSubstitution(rec_a["step_id"], {"brand": "new"})])
    result = t.run_replay(subs, Executor(tool=lambda n, a: _tool_b(n, a)))

    assert result.steps[0].output_changed is True


# ---------------------------------------------------------------------------
# Scenario 5 (explicit): dirty-but-same-output → output_changed False
# ---------------------------------------------------------------------------

def test_output_changed_false_when_dirty_but_same_content(tmp_path):
    path = str(tmp_path / "trace.sb")
    _build_two_step_trace(path)
    t = replay(path)

    rec_a = t.recorded_steps[0]
    subs = SubstitutionSet([_same_output_sub(rec_a)])
    result = t.run_replay(subs, Executor())

    assert result.steps[0].dirty is True
    assert result.steps[0].output_changed is False


# ---------------------------------------------------------------------------
# Scenario 6: grandparent dirty same output → all downstream cache hits
# ---------------------------------------------------------------------------

def test_grandparent_dirty_same_output_all_downstream_cache_hits(tmp_path):
    """Chain A → B → C.  A is forced with same output.
    B must be a cache hit, C must be a cache hit."""
    path = str(tmp_path / "trace.sb")
    _build_three_step_chain(path)
    t = replay(path)

    rec_a = t.recorded_steps[0]
    subs = SubstitutionSet([_same_output_sub(rec_a)])
    result = t.run_replay(subs, Executor())

    step_a, step_b, step_c = result.steps
    assert step_a.dirty is True
    assert step_a.output_changed is False
    assert step_b.dirty is False, "B must be cache hit"
    assert step_b.output_changed is False
    assert step_c.dirty is False, "C must be cache hit (no upstream output changed)"
    assert result.dirty_count == 1  # only A


# ---------------------------------------------------------------------------
# Scenario 7: step without context field, parent same output → cache hit
# ---------------------------------------------------------------------------

def test_no_context_field_step_cache_hit_when_parent_output_unchanged(tmp_path):
    """Step B opted out of context_from_parent.  When A is dirty with same
    output, B has no context to rebind and its inputs hash is unchanged,
    so B must be a cache hit (no use_parent_dirty leaking through)."""
    path = str(tmp_path / "trace.sb")
    _build_no_context_trace(path)
    t = replay(path)

    rec_a = t.recorded_steps[0]
    subs = SubstitutionSet([_same_output_sub(rec_a)])
    result = t.run_replay(subs, Executor())

    assert result.steps[0].dirty is True
    assert result.steps[0].output_changed is False
    assert result.steps[1].dirty is False
    assert result.steps[1].cache_hit is True


# ---------------------------------------------------------------------------
# Scenario 8: compute_dirty_set consistency
# ---------------------------------------------------------------------------

def test_compute_dirty_set_consistent_with_replay_on_same_output(tmp_path):
    """compute_dirty_set must agree with replay: only A is dirty when A is
    forced with its own output (dirty_count == 1, only A in dirty set)."""
    path = str(tmp_path / "trace.sb")
    _build_two_step_trace(path)
    t = replay(path)

    rec_a = t.recorded_steps[0]
    sub = _same_output_sub(rec_a)
    subs = SubstitutionSet([sub])

    dirty_summary = compute_dirty_set(t, [sub])
    result = t.run_replay(subs, Executor())

    dirty_ids = {e.step_id for e in dirty_summary.entries if e.dirty}
    assert result.dirty_count == 1
    assert dirty_ids == {rec_a["step_id"]}


# ---------------------------------------------------------------------------
# Scenario 9: no-substitution baseline
# ---------------------------------------------------------------------------

def test_no_substitution_all_output_changed_false(tmp_path):
    """Without any substitutions, every step is a cache hit and output_changed
    must be False for all steps."""
    path = str(tmp_path / "trace.sb")
    _build_two_step_trace(path)
    t = replay(path)

    result = t.run_replay(SubstitutionSet(), Executor())

    assert result.dirty_count == 0
    assert result.cache_hit_count == 2
    for sv in result.steps:
        assert sv.output_changed is False, (
            f"{sv.step_id}.output_changed should be False on a cache hit"
        )


# ---------------------------------------------------------------------------
# Scenario 10: dirty+changed output propagates transitively
# ---------------------------------------------------------------------------

def test_changed_output_propagates_transitively(tmp_path):
    """When A's output changes, B becomes dirty (via context rebinding).
    If B's re-execution also produces a different output, C must be dirty too."""
    path = str(tmp_path / "trace.sb")
    _build_three_step_chain(path)
    t = replay(path)

    rec_a = t.recorded_steps[0]
    different_inner = {"value": "DIFFERENT"}
    subs = SubstitutionSet([ToolOutputSubstitution(rec_a["step_id"], different_inner)])

    # Executor that always returns a new, distinct value for dirty steps.
    call_n = [0]
    def varying_tool(name: str, args: dict) -> Any:
        call_n[0] += 1
        return {"value": f"recomputed_{call_n[0]}"}

    result = t.run_replay(subs, Executor(tool=varying_tool))

    assert result.steps[0].output_changed is True
    assert result.steps[1].dirty is True
    assert result.steps[1].output_changed is True
    assert result.steps[2].dirty is True
    assert result.dirty_count == 3
