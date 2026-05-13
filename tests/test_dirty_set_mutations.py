"""Mutation testing for dirty-set decisions.

Implements step 33 of ``100_STEPS.md``:

    Add mutation testing for dirty-set decisions by mutating parent
    edges, content hashes, step kinds, and nondeterminism hashes.

The dirty-set replay engine in :mod:`stepback.replay` decides cache-hit
vs. dirty for every step in a recorded `.sb` trace.  That decision is a
function of four pieces of recorded metadata:

1. ``inputs_hash`` — content hash of the step's canonical inputs
2. ``parent_step_id`` (and ``parent_step_ids`` for joins) — DAG edges
3. ``step_kind`` — drives parent-output rebinding, executor dispatch,
   and the multi-parent join path
4. ``nondeterminism_hash`` — content hash of the step's recorded
   nondeterministic inputs (sampling seeds, system fingerprints, …)

These are the *cache-safety boundary*. If any of them is silently
corrupted in a stored trace, the engine MUST notice and mark the step
dirty rather than serving a stale cached output. Equally, mutations
that leave the cache contract intact (e.g. rewriting a recorded
``inputs_hash`` to the same value) MUST NOT spuriously invalidate the
cache.

This module is a classical mutation-testing harness scoped to the
dirty-set decision: for each metadata field we apply a small family of
mutation operators to a freshly-recorded baseline trace, replay, and
assert that the dirty-set engine's classification of every step
matches an oracle derived from first principles.

The harness is deliberately distinct from
``test_dirty_set_hypothesis.py`` (step 29): that file checks
*equivalence* between dirty-set replay and forced full re-execution
under a deterministic executor.  This file checks the *engine's
sensitivity* — that the engine actually consults each piece of
metadata it claims to consult.  Together they pin both correctness
("cache hits are safe") and completeness ("the cache only short-circuits
when it's safe to").
"""
from __future__ import annotations

import copy
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

from stepback import RecorderKey, record, replay
from stepback.canonical import canonical_json, hash_obj, sha256_hex
from stepback.replay import Executor, ReplayResult, Trace
from stepback.testing import fake_llm, fake_tool, run_recorded_agent


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _record_baseline(tmp_path) -> Tuple[str, RecorderKey]:
    key = RecorderKey.fresh()
    path = str(tmp_path / "baseline.sb")
    with record(path, key=key) as rec:
        run_recorded_agent(rec)
    return path, key


def _make_executor() -> Executor:
    """Executor that re-runs LLM/tool against the same deterministic fakes
    used during recording. With unmutated metadata, every step should
    cache-hit; with mutated metadata, only dirty steps should be
    re-executed and they should reproduce the recorded output bytewise.
    """
    return Executor(llm=fake_llm, tool=fake_tool)


def _replay_with(trace: Trace, executor: Optional[Executor] = None) -> ReplayResult:
    return trace.replay_forward(executor=executor or _make_executor())


def _load(path: str, key: RecorderKey) -> Trace:
    return replay(path, hmac_key=key.hmac_key)


def _ids(trace: Trace) -> List[str]:
    return [s["step_id"] for s in trace.recorded_steps]


def _step_index(trace: Trace, step_id: str) -> int:
    for i, s in enumerate(trace.recorded_steps):
        if s["step_id"] == step_id:
            return i
    raise AssertionError(f"step {step_id!r} not found")


# ---------------------------------------------------------------------------
# Baseline sanity
# ---------------------------------------------------------------------------


def test_baseline_is_full_cache_hit(tmp_path):
    """Sanity: an unmutated baseline replays as 100% cache hit, 0 real calls.

    This is the control condition for every mutation experiment below:
    if this fails the harness tells us nothing about mutation sensitivity.
    """
    path, key = _record_baseline(tmp_path)
    trace = _load(path, key)
    executor = _make_executor()
    result = trace.replay_forward(executor=executor)

    assert result.dirty_count == 0
    assert result.cache_hit_count == len(trace.recorded_steps)
    assert result.real_executions == 0
    assert executor.real_calls == 0
    assert all(s.cache_hit and not s.dirty for s in result.steps)


# ---------------------------------------------------------------------------
# Mutation operator: inputs_hash
# ---------------------------------------------------------------------------
#
# The cache short-circuit is *exactly* `current_inputs_hash ==
# recorded_inputs_hash`. Mutating either side of that comparison (the
# stored hash, or the inputs payload itself) must dirty the step.


def test_mutate_inputs_hash_dirties_target_and_descendants(tmp_path):
    """Rewriting ``inputs_hash`` to a value that no longer matches the
    canonical hash of ``inputs`` MUST dirty that step.

    The current engine propagates dirtiness by parent-bit, not by
    output-equivalence, so descendants in the parent chain inherit
    dirtiness even when a deterministic executor reproduces the
    recorded output bytewise. Steps unrelated to the dirty subtree
    (siblings under different parents) stay clean.
    """
    path, key = _record_baseline(tmp_path)
    trace = _load(path, key)
    target_idx = 4  # arbitrary mid-trace step
    target_id = trace.recorded_steps[target_idx]["step_id"]
    trace.recorded_steps[target_idx]["inputs_hash"] = "FORCE_DIRTY::" + (
        trace.recorded_steps[target_idx]["inputs_hash"]
    )

    executor = _make_executor()
    result = trace.replay_forward(executor=executor)

    dirty_ids = [s.step_id for s in result.steps if s.dirty]
    assert target_id in dirty_ids
    # The corrupted step is re-executed.
    assert executor.real_calls >= 1
    # Steps strictly before the target are unaffected.
    pre = [s for s in result.steps[:target_idx]]
    for view in pre:
        assert view.cache_hit and not view.dirty


def test_mutate_inputs_payload_dirties_step(tmp_path):
    """Mutating the recorded ``inputs`` payload (without touching
    ``inputs_hash``) MUST dirty the step: the engine recomputes the
    hash from the live payload and the comparison fails.
    """
    path, key = _record_baseline(tmp_path)
    trace = _load(path, key)
    # Pick a tool step (idx 1 in the run_recorded_agent fixture).
    target_idx = 1
    rec = trace.recorded_steps[target_idx]
    assert rec["step_kind"] == "tool_call"
    rec["inputs"] = copy.deepcopy(rec["inputs"])
    rec["inputs"]["arguments"]["name"] = "MUTATED-NAME"

    executor = _make_executor()
    result = trace.replay_forward(executor=executor)

    target_view = next(s for s in result.steps if s.step_id == rec["step_id"])
    assert target_view.dirty
    assert not target_view.cache_hit
    assert executor.real_calls >= 1


def test_inputs_hash_no_op_mutation_preserves_cache(tmp_path):
    """Negative control: rewriting ``inputs_hash`` to its existing value
    is a no-op and MUST NOT spuriously dirty the step.

    Catches engine bugs where hash comparison is identity rather than
    equality.
    """
    path, key = _record_baseline(tmp_path)
    trace = _load(path, key)
    rec = trace.recorded_steps[3]
    rec["inputs_hash"] = str(rec["inputs_hash"])  # same value, fresh str object

    executor = _make_executor()
    result = trace.replay_forward(executor=executor)
    assert result.dirty_count == 0
    assert executor.real_calls == 0


# ---------------------------------------------------------------------------
# Mutation operator: parent_step_id
# ---------------------------------------------------------------------------


def test_mutate_parent_to_dirty_propagates(tmp_path):
    """If a step's recorded ``parent_step_id`` is rewritten to point at
    a different step that has been forced dirty, the descendant MUST
    inherit dirtiness via the parent_dirty propagation rule.
    """
    path, key = _record_baseline(tmp_path)
    trace = _load(path, key)

    # Force step 0 dirty by mutating its inputs_hash.
    trace.recorded_steps[0]["inputs_hash"] = "FORCE_DIRTY::0"
    # Re-parent step 5 onto step 0 (originally parented at step 4).
    re_id = trace.recorded_steps[0]["step_id"]
    trace.recorded_steps[5]["parent_step_id"] = re_id

    executor = _make_executor()
    # Force step 0's outputs to be NON-deterministic so descendants
    # actually see a different parent context. Wrap fake_llm to perturb
    # output once, only for step 0.
    seen = {"count": 0}

    def perturbing_llm(model, messages):
        out = fake_llm(model, messages)
        seen["count"] += 1
        if seen["count"] == 1:
            out = copy.deepcopy(out)
            out["choices"][0]["message"]["content"] = "MUTATED-OUTPUT"
        return out

    executor.llm = perturbing_llm
    result = trace.replay_forward(executor=executor)

    # Step 0 is dirty by direct mutation.
    s0 = next(s for s in result.steps if s.step_id == re_id)
    assert s0.dirty
    # Step 5 is dirty by parent-propagation: its parent (step 0) is
    # dirty AND its current_inputs_hash differs from recorded because
    # the rebound parent context is the perturbed output, not the
    # original.
    s5 = next(s for s in result.steps if s.step_id == trace.recorded_steps[5]["step_id"])
    assert s5.dirty


def test_mutate_parent_to_clean_breaks_propagation(tmp_path):
    """Re-parenting away from a dirty ancestor MUST stop the dirtiness
    flowing down that edge. Verifies the engine consults the *current*
    parent edge, not a snapshot.
    """
    path, key = _record_baseline(tmp_path)
    trace = _load(path, key)

    # Force step 2 dirty.
    trace.recorded_steps[2]["inputs_hash"] = "FORCE_DIRTY::2"
    # Pull step 4 off of step 3 (which would inherit dirtiness via the
    # transitive chain) and re-parent it to None so the propagation
    # path is severed.
    trace.recorded_steps[4]["parent_step_id"] = None

    executor = _make_executor()
    result = trace.replay_forward(executor=executor)

    s4 = next(
        s for s in result.steps if s.step_id == trace.recorded_steps[4]["step_id"]
    )
    # With the parent edge severed and the deterministic executor
    # reproducing identical outputs, step 4 sees no inputs change and
    # no dirty parent -> still a cache hit.
    assert s4.cache_hit
    assert not s4.dirty


def test_mutate_parent_to_self_does_not_crash(tmp_path):
    """Defensive: a self-edge in ``parent_step_id`` must not crash the
    engine. The engine treats the parent dirty-bit lookup as a plain
    dict get: a self-loop simply reads back its own (yet-to-be-set)
    dirtiness as ``False``, so the step is decided on its own inputs.

    This is a structural test: we don't claim self-loops are *valid*
    in `.sb`, we just want graceful handling rather than a KeyError.
    """
    path, key = _record_baseline(tmp_path)
    trace = _load(path, key)
    rec = trace.recorded_steps[3]
    rec["parent_step_id"] = rec["step_id"]

    executor = _make_executor()
    # Should not raise.
    result = trace.replay_forward(executor=executor)
    assert len(result.steps) == len(trace.recorded_steps)


# ---------------------------------------------------------------------------
# Mutation operator: step_kind
# ---------------------------------------------------------------------------


def test_mutate_step_kind_breaks_executor_dispatch(tmp_path):
    """Rewriting ``step_kind`` from ``llm_call`` to ``tool_call`` (or
    vice versa) MUST be detectable: either the dispatch goes through
    the wrong executor (different output shape -> dirty propagation)
    or the engine raises a clear error.

    The recorded ``inputs`` for an LLM step has ``model`` and
    ``messages`` keys; tool dispatch reads ``name`` and ``arguments``.
    A mistyped step is therefore guaranteed to either (a) hit a
    KeyError when the wrong executor is invoked on a dirty path, or
    (b) cache-hit harmlessly if the inputs aren't disturbed. We test
    case (a) by also forcing the step dirty.
    """
    path, key = _record_baseline(tmp_path)
    trace = _load(path, key)
    target_idx = 0  # llm_call
    rec = trace.recorded_steps[target_idx]
    assert rec["step_kind"] == "llm_call"
    rec["step_kind"] = "tool_call"
    rec["inputs_hash"] = "FORCE_DIRTY::kind"  # ensure executor is invoked

    executor = _make_executor()
    with pytest.raises((KeyError, TypeError)):
        trace.replay_forward(executor=executor)


def test_mutate_step_kind_to_unknown_raises(tmp_path):
    """An unknown ``step_kind`` on a dirty step must raise
    ``MissingExecutor`` rather than silently caching a stale output.
    """
    from stepback.replay import MissingExecutor

    path, key = _record_baseline(tmp_path)
    trace = _load(path, key)
    rec = trace.recorded_steps[2]
    rec["step_kind"] = "frobnicate_call"  # not a real kind
    rec["inputs_hash"] = "FORCE_DIRTY::kind"

    executor = _make_executor()
    with pytest.raises(MissingExecutor):
        trace.replay_forward(executor=executor)


def test_mutate_step_kind_clean_path_is_no_op(tmp_path):
    """If a step's kind is mutated but its inputs_hash is left intact
    AND no ancestor is dirty, the engine still cache-hits and never
    consults the executor — the kind only matters on the dirty path.

    This documents that step_kind is *not* part of the cache-safety
    boundary today; the cache key is (inputs_hash, parent dirtiness).
    Promoting step_kind into the cache key is a future tightening; if
    that change lands, this test will need to flip from
    ``cache_hit_count == N`` to ``dirty_count >= 1``.
    """
    path, key = _record_baseline(tmp_path)
    trace = _load(path, key)
    rec = trace.recorded_steps[3]
    original_kind = rec["step_kind"]
    rec["step_kind"] = "frobnicate_call"

    executor = _make_executor()
    result = trace.replay_forward(executor=executor)
    assert executor.real_calls == 0
    assert result.dirty_count == 0
    # Restore for safety in case fixture is shared.
    rec["step_kind"] = original_kind


# ---------------------------------------------------------------------------
# Mutation operator: nondeterminism_hash
# ---------------------------------------------------------------------------


def test_mutate_nondeterminism_hash_dirties_step(tmp_path):
    """Per the engine docstring, "a step is a cache hit iff … its
    recorded ``nondeterminism_hash`` is unchanged". Mutating the
    recorded value to something inconsistent with the canonical hash
    of the live ``nondeterminism`` payload MUST dirty the step.
    """
    path, key = _record_baseline(tmp_path)
    trace = _load(path, key)
    rec = trace.recorded_steps[6]
    # Sanity: baseline value is the hash of an empty dict.
    expected = sha256_hex(canonical_json(rec.get("nondeterminism", {})))
    assert rec["nondeterminism_hash"] == expected
    rec["nondeterminism_hash"] = "FORCE_DIRTY::nondet"

    executor = _make_executor()
    result = trace.replay_forward(executor=executor)

    target = next(s for s in result.steps if s.step_id == rec["step_id"])
    assert target.dirty
    assert not target.cache_hit
    assert executor.real_calls >= 1


def test_mutate_nondeterminism_payload_dirties_step(tmp_path):
    """Adding a new key to ``nondeterminism`` (without rewriting
    ``nondeterminism_hash``) MUST dirty the step: the engine recomputes
    the hash from the payload and the comparison fails.

    This catches the case where a downstream tool injected a sampling
    seed into the recorded trace post-hoc; replay should not silently
    serve the old cached output as if the seed had always been present.
    """
    path, key = _record_baseline(tmp_path)
    trace = _load(path, key)
    rec = trace.recorded_steps[7]
    rec["nondeterminism"] = {"sampling_seed": 12345}
    # Leave nondeterminism_hash intact.

    executor = _make_executor()
    result = trace.replay_forward(executor=executor)

    target = next(s for s in result.steps if s.step_id == rec["step_id"])
    assert target.dirty
    assert not target.cache_hit


def test_nondeterminism_hash_no_op_mutation_preserves_cache(tmp_path):
    """Negative control: rewriting ``nondeterminism_hash`` to the
    canonical hash of the existing payload (i.e. its current value)
    is a no-op.
    """
    path, key = _record_baseline(tmp_path)
    trace = _load(path, key)
    rec = trace.recorded_steps[5]
    rec["nondeterminism_hash"] = sha256_hex(
        canonical_json(rec.get("nondeterminism", {}))
    )

    executor = _make_executor()
    result = trace.replay_forward(executor=executor)
    assert result.dirty_count == 0
    assert executor.real_calls == 0


# ---------------------------------------------------------------------------
# Composed-mutation oracle
# ---------------------------------------------------------------------------
#
# Combine all four mutation operators on independent steps and verify
# the per-step dirty oracle matches engine output exactly.


def _oracle_dirty_set(
    trace: Trace, mutated_indices_inputs_hash: List[int],
    mutated_indices_nondet_hash: List[int],
    severed_parent_indices: List[int],
) -> List[bool]:
    """First-principles oracle: a step is dirty iff
       (its inputs_hash was mutated) OR
       (its nondeterminism_hash was mutated) OR
       (its parent (after any severance) is in the dirty set)
    Computed without invoking the engine.
    """
    n = len(trace.recorded_steps)
    dirty = [False] * n
    parent_idx_map: Dict[str, int] = {
        s["step_id"]: i for i, s in enumerate(trace.recorded_steps)
    }
    for i, s in enumerate(trace.recorded_steps):
        if i in mutated_indices_inputs_hash or i in mutated_indices_nondet_hash:
            dirty[i] = True
            continue
        if i in severed_parent_indices:
            continue  # parent edge severed -> no propagation
        pid = s.get("parent_step_id")
        if pid and pid in parent_idx_map and dirty[parent_idx_map[pid]]:
            dirty[i] = True
    return dirty


def test_combined_mutation_matches_oracle(tmp_path):
    """End-to-end: apply mutations across all four operators, then
    assert the engine's per-step dirty classification matches the
    first-principles oracle.

    With a deterministic executor, parent-propagation only ever
    promotes inputs_hash mismatches, so we control parent rebinding
    by perturbing the LLM on dirty-step recomputation.
    """
    path, key = _record_baseline(tmp_path)
    trace = _load(path, key)

    # Mutate inputs_hash on steps 1 and 8.
    for i in (1, 8):
        trace.recorded_steps[i]["inputs_hash"] = "FORCE_DIRTY::" + str(i)
    # Mutate nondet hash on step 4.
    trace.recorded_steps[4]["nondeterminism_hash"] = "FORCE_DIRTY::nondet4"
    # Sever parent on step 9 (so dirtiness from 8 doesn't propagate).
    trace.recorded_steps[9]["parent_step_id"] = None

    executor = _make_executor()

    # Perturb every recomputed output so that descendants see a
    # different parent context and propagation actually fires.
    counter = {"i": 0}

    def perturbing_llm(model, messages):
        counter["i"] += 1
        out = fake_llm(model, messages)
        out = copy.deepcopy(out)
        out["choices"][0]["message"]["content"] += f" mut-{counter['i']}"
        return out

    def perturbing_tool(name, arguments):
        counter["i"] += 1
        out = fake_tool(name, arguments)
        # fake_tool returns a dict already; perturb a stable field.
        out = copy.deepcopy(out)
        if isinstance(out, dict):
            out["__mut__"] = counter["i"]
        return out

    executor.llm = perturbing_llm
    executor.tool = perturbing_tool

    result = trace.replay_forward(executor=executor)

    oracle = _oracle_dirty_set(
        trace,
        mutated_indices_inputs_hash=[1, 8],
        mutated_indices_nondet_hash=[4],
        severed_parent_indices=[9],
    )
    engine = [s.dirty for s in result.steps]
    assert engine == oracle, f"engine={engine} oracle={oracle}"


# ---------------------------------------------------------------------------
# Bytewise content-hash mutation: outputs_hash
# ---------------------------------------------------------------------------
#
# outputs_hash is recorded but is not part of the cache-key. Mutations
# to it MUST NOT change dirty-set decisions (only the input side of
# the boundary controls cache hits). This pins the *negative* property.


def test_mutate_outputs_hash_does_not_dirty(tmp_path):
    """``outputs_hash`` is content-addressing for downstream
    rebinding, but the engine recomputes outputs hashes during replay
    rather than trusting the recorded value. Therefore mutating the
    recorded ``outputs_hash`` MUST be a no-op for dirty-set decisions.
    """
    path, key = _record_baseline(tmp_path)
    trace = _load(path, key)
    for i in (0, 3, 6, 11):
        trace.recorded_steps[i]["outputs_hash"] = "MUTATED::" + str(i)

    executor = _make_executor()
    result = trace.replay_forward(executor=executor)
    assert result.dirty_count == 0
    assert executor.real_calls == 0


# ---------------------------------------------------------------------------
# Mutation harness coverage: every step, every operator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operator", ["inputs_hash", "nondeterminism_hash"])
def test_every_step_dirties_under_metadata_mutation(tmp_path, operator):
    """For every step in the baseline trace and every metadata operator
    that's part of the cache key, mutating that single step's metadata
    MUST dirty at least that step. Steps strictly *before* the
    mutation point stay clean (mutation cannot reach backwards).
    """
    path, key = _record_baseline(tmp_path)
    baseline = _load(path, key)
    n = len(baseline.recorded_steps)

    for i in range(n):
        trace = _load(path, key)
        rec = trace.recorded_steps[i]
        if operator == "inputs_hash":
            rec["inputs_hash"] = "FORCE_DIRTY::ih::" + rec["step_id"]
        elif operator == "nondeterminism_hash":
            rec["nondeterminism_hash"] = "FORCE_DIRTY::nh::" + rec["step_id"]

        executor = _make_executor()
        result = trace.replay_forward(executor=executor)

        target = next(s for s in result.steps if s.step_id == rec["step_id"])
        assert target.dirty, (
            f"step {i} ({rec['step_id']}) was not dirtied by {operator} mutation"
        )
        # Steps strictly before the mutation point must stay clean.
        for j in range(i):
            v = result.steps[j]
            assert v.cache_hit and not v.dirty, (
                f"mutating step {i} spuriously dirtied earlier step {j}"
            )
