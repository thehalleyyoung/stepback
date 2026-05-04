"""Stress tests for the parallel-branch fan-out/fan-in machinery.

Step #35 of ``docs/100_STEPS.md``: record a 1,000-way fan-out and
verify that a substitution inside ONE branch:

  * dirties exactly that branch's tail, the join, and the downstream
    synthesise step (4 dirty steps total),
  * leaves all 999 sibling branches as cache hits,
  * propagates only through the join (not across siblings),
  * preserves the on-disk trace under HMAC + Ed25519 verification,
  * yields O(dirty_set) — not O(N) — re-executions on replay.

These tests pin the *quantitative* contract that fan-out modelling
saves work proportional to the size of the affected branch, not the
total branch count. Any regression that accidentally invalidates
sibling caches will dirty ~1003 steps instead of 4 and trip here
loudly.
"""
from __future__ import annotations

import os
import time

import pytest

from stepback import RecorderKey, record, replay
from stepback.replay import Executor
from stepback.substitutions import ToolOutputSubstitution
from stepback.trace_reader import verify_trace


# Number of fan-out branches. 1,000 is the spec target from Step #35.
# Each branch is exactly one tool_call so the trace is 1 + 1 + 1 + N + 1 + 1
# = N + 5 steps total. With N=1000 that's 1005 recorded steps.
N_BRANCHES = 1000


# --------------------------------------------------------- fixtures


def _fake_llm(model: str, messages: list) -> dict:
    """Cheap deterministic LLM stand-in; hashes inputs into the reply."""
    blob = "|".join(f"{m['role']}={m['content']}" for m in messages)
    text = f"reply:{len(blob)}:{hash(blob) & 0xFFFF:04x}"
    return {
        "id": f"chatcmpl-{len(blob)}",
        "model": model,
        "choices": [
            {"index": 0, "finish_reason": "stop",
             "message": {"role": "assistant", "content": text}}
        ],
        "usage": {"prompt_tokens": len(blob),
                  "completion_tokens": len(text),
                  "total_tokens": len(blob) + len(text)},
    }


def _fake_tool(name: str, args: dict) -> dict:
    """Branch lookup stand-in: returns a stable per-branch payload."""
    if name == "split_question":
        return {"topics": [f"topic_{i}" for i in range(N_BRANCHES)]}
    if name.startswith("lookup_topic_"):
        idx = int(name[len("lookup_topic_"):])
        return {"topic_idx": idx, "value": idx * 7 + 1}
    raise KeyError(f"unknown fake tool: {name!r}")


def _make_branch(idx: int):
    def branch(rec):
        rec.tool_call(
            f"lookup_topic_{idx}",
            {"topic_idx": idx},
            executor=_fake_tool,
        )
    return branch


def _run_wide_fanout(rec) -> dict:
    """Drive an N_BRANCHES-wide fan-out research agent through ``rec``.

    Trace shape::

        step:1   llm_call            "plan"
        step:2   tool_call           "split_question"
        step:3   parallel_branch_open
        step:4..N+3  tool_call (one per branch, all parented to step:3)
        step:N+4 parallel_branch_join (parents = all branch tails)
        step:N+5 llm_call            "synthesise"
    """
    convo = [
        {"role": "system", "content": "You are a wide research agent."},
        {"role": "user", "content": "Summarise everything."},
    ]
    rec.llm_call("gpt-4o-2024-11-20", convo, executor=_fake_llm)
    rec.tool_call("split_question", {"q": convo[-1]["content"]},
                  executor=_fake_tool)
    join_step = rec.parallel(
        "wide_fanout",
        [_make_branch(i) for i in range(N_BRANCHES)],
        join=lambda outs: {"n": len(outs),
                           "sum": sum(o["result"]["value"] for o in outs)},
        branch_names=[f"research/topic_{i}" for i in range(N_BRANCHES)],
    )
    merged_summary = (
        f"n={join_step['outputs']['n']} sum={join_step['outputs']['sum']}"
    )
    return rec.llm_call(
        "gpt-4o-2024-11-20",
        convo + [
            {"role": "assistant", "content": "synthesising"},
            {"role": "user", "content": f"merged: {merged_summary}"},
        ],
        executor=_fake_llm,
    )


def _record_wide_trace(tmp_path) -> tuple:
    path = os.path.join(str(tmp_path), "wide_parallel.sb")
    key = RecorderKey.fresh()
    with record(path, key=key) as rec:
        _run_wide_fanout(rec)
        recorded = list(rec.steps)
    return path, key, recorded


# Module-scoped cache: recording 1,005 signed frames is the slow part.
# Tests share one recorded trace path via this fixture so the suite
# stays under a few seconds of wall time.
@pytest.fixture(scope="module")
def wide_trace(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("wide_parallel")
    return _record_wide_trace(tmp)


# ------------------------------------------------------------ tests


def test_wide_fanout_recorded_shape(wide_trace):
    path, key, steps = wide_trace
    expected_total = N_BRANCHES + 5  # plan + split + open + N + join + synth
    assert len(steps) == expected_total, (len(steps), expected_total)

    kinds = [s["step_kind"] for s in steps]
    assert kinds[0] == "llm_call"
    assert kinds[1] == "tool_call"
    assert kinds[2] == "parallel_branch_open"
    # All N_BRANCHES branch bodies are tool_calls.
    branch_kinds = kinds[3:3 + N_BRANCHES]
    assert set(branch_kinds) == {"tool_call"}, set(branch_kinds)
    assert kinds[3 + N_BRANCHES] == "parallel_branch_join"
    assert kinds[-1] == "llm_call"

    # Open advertises N_BRANCHES distinct branch names.
    open_step = steps[2]
    bnames = open_step["outputs"]["branch_names"]
    assert len(bnames) == N_BRANCHES
    assert len(set(bnames)) == N_BRANCHES

    # Every branch tool_call's parent must be the open step.
    open_id = open_step["step_id"]
    for s in steps[3:3 + N_BRANCHES]:
        assert s["parent_step_id"] == open_id, s

    # Join references all N_BRANCHES branch tails as parent_step_ids.
    join_step = steps[3 + N_BRANCHES]
    assert join_step["parent_step_id"] == open_id
    assert len(join_step["parent_step_ids"]) == N_BRANCHES
    assert len(set(join_step["parent_step_ids"])) == N_BRANCHES

    # Synthesise's parent is the join.
    assert steps[-1]["parent_step_id"] == join_step["step_id"]


def test_wide_fanout_signed_round_trip(wide_trace):
    path, key, _ = wide_trace
    verified = verify_trace(path, key.hmac_key)
    assert len(verified.steps) == N_BRANCHES + 5
    join = next(s for s in verified.steps
                if s["step_kind"] == "parallel_branch_join")
    assert len(join["parent_step_ids"]) == N_BRANCHES
    open_ = next(s for s in verified.steps
                 if s["step_kind"] == "parallel_branch_open")
    assert open_["outputs"]["branch_count"] == N_BRANCHES


def test_wide_fanout_cached_replay_is_zero_llm(wide_trace):
    path, key, _ = wide_trace
    trace = replay(path)
    result = trace.replay_forward(Executor())
    assert result.real_executions == 0
    assert result.dirty_count == 0
    assert result.cache_hit_count == N_BRANCHES + 5


def test_one_branch_substitution_dirties_only_that_branch_plus_tail(wide_trace):
    """The single most important guarantee from Step #35.

    A substitution inside ONE of N_BRANCHES branches must dirty exactly
    that branch's tail + the join + the downstream synthesise. The
    other (N_BRANCHES - 1) sibling branches must remain cache hits.
    """
    path, key, steps = wide_trace
    # Pick a branch in the middle of the pack so any off-by-one mistake
    # in dirty propagation (e.g. "everything after the substitution")
    # is caught.
    target_idx = N_BRANCHES // 2
    target_step_id = steps[3 + target_idx]["step_id"]

    trace = replay(path)
    trace.substitute(ToolOutputSubstitution(
        at_step=target_step_id,
        fake_response={"topic_idx": target_idx, "value": -999},
    ))
    result = trace.replay_forward(Executor(llm=_fake_llm, tool=_fake_tool))

    by_id = {s.step_id: s for s in result.steps}
    join_id = steps[3 + N_BRANCHES]["step_id"]
    synth_id = steps[-1]["step_id"]

    # The substituted branch tail is dirty.
    assert by_id[target_step_id].dirty
    # The join is dirty (multi-parent dirty propagation through it).
    assert by_id[join_id].dirty
    # The synthesise call is dirty (depends on join's hashed output).
    assert by_id[synth_id].dirty

    # All other branch tails must stay cached.
    other_branch_ids = [
        steps[3 + i]["step_id"] for i in range(N_BRANCHES) if i != target_idx
    ]
    dirty_siblings = [b for b in other_branch_ids if by_id[b].dirty]
    assert dirty_siblings == [], (
        f"{len(dirty_siblings)} sibling branches incorrectly dirty: "
        f"{dirty_siblings[:5]}..."
    )
    cached_siblings = [b for b in other_branch_ids if by_id[b].cache_hit]
    assert len(cached_siblings) == N_BRANCHES - 1

    # Open step and upstream (plan/split) must NOT be dirty.
    assert by_id[steps[0]["step_id"]].cache_hit
    assert by_id[steps[1]["step_id"]].cache_hit
    assert by_id[steps[2]["step_id"]].cache_hit

    # Exactly 3 dirty steps: substituted tool + join + synthesise.
    dirty_ids = sorted(s.step_id for s in result.steps if s.dirty)
    assert dirty_ids == sorted([target_step_id, join_id, synth_id]), dirty_ids
    assert sum(1 for s in result.steps if s.dirty) == 3

    # Real-execution accounting: ToolOutputSubstitution satisfies the
    # dirty tool without an executor call, so only the join and the
    # synthesise call are real re-executions.
    assert result.real_executions == 2

    # Cache survival ratio: at least 99.5% of steps must remain cached
    # under a single-branch substitution in a 1000-way fan-out.
    cached = sum(1 for s in result.steps if s.cache_hit)
    assert cached / (N_BRANCHES + 5) >= 0.995


def test_one_branch_substitution_dirty_set_independent_of_branch_count(wide_trace):
    """Numeric guarantee: the dirty-set size from a single-branch
    substitution is independent of N_BRANCHES.

    Catches any regression where dirty propagation degrades to O(N).
    """
    path, key, steps = wide_trace
    trace = replay(path)
    trace.substitute(ToolOutputSubstitution(
        at_step=steps[3]["step_id"],   # first branch tail
        fake_response={"topic_idx": 0, "value": -1},
    ))
    result = trace.replay_forward(Executor(llm=_fake_llm, tool=_fake_tool))
    dirty = sum(1 for s in result.steps if s.dirty)
    # Constant-3 dirty set regardless of N_BRANCHES.
    assert dirty == 3, dirty
    # Serial-equivalent dirty count would be (N_BRANCHES + 5) - 4 + 1 =
    # N_BRANCHES + 1. We must dirty STRICTLY fewer than that, by a
    # huge margin.
    serial_dirty = (N_BRANCHES + 5) - 4 + 1
    assert dirty < serial_dirty / 100, (dirty, serial_dirty)


def test_substitute_last_branch_does_not_dirty_earlier_branches(wide_trace):
    """Substituting the LAST branch tail must not dirty any earlier
    branch — no spurious "dirty everything after step k" behaviour.
    """
    path, key, steps = wide_trace
    last_branch_idx = N_BRANCHES - 1
    last_branch_id = steps[3 + last_branch_idx]["step_id"]

    trace = replay(path)
    trace.substitute(ToolOutputSubstitution(
        at_step=last_branch_id,
        fake_response={"topic_idx": last_branch_idx, "value": 0},
    ))
    result = trace.replay_forward(Executor(llm=_fake_llm, tool=_fake_tool))
    by_id = {s.step_id: s for s in result.steps}

    # Every earlier branch (idx 0..N-2) must be a cache hit.
    earlier = [steps[3 + i]["step_id"] for i in range(N_BRANCHES - 1)]
    assert all(by_id[b].cache_hit for b in earlier)
    assert all(not by_id[b].dirty for b in earlier)
    # Dirty set is still exactly 3.
    assert sum(1 for s in result.steps if s.dirty) == 3


def test_substitute_first_branch_does_not_dirty_later_siblings(wide_trace):
    """Substituting the FIRST branch tail must not dirty any later
    sibling branch via spurious linear "downstream" propagation.
    """
    path, key, steps = wide_trace
    first_branch_id = steps[3]["step_id"]

    trace = replay(path)
    trace.substitute(ToolOutputSubstitution(
        at_step=first_branch_id,
        fake_response={"topic_idx": 0, "value": 0},
    ))
    result = trace.replay_forward(Executor(llm=_fake_llm, tool=_fake_tool))
    by_id = {s.step_id: s for s in result.steps}

    later = [steps[3 + i]["step_id"] for i in range(1, N_BRANCHES)]
    dirty_later = [b for b in later if by_id[b].dirty]
    assert dirty_later == [], (
        f"{len(dirty_later)} later siblings dirty after first-branch "
        f"substitution"
    )
    assert sum(1 for s in result.steps if s.dirty) == 3


def test_wide_fanout_replay_throughput_bound(wide_trace):
    """Cached replay of a 1,005-step trace must complete under a
    generous wall-clock ceiling. Pins replay scaling: linear in N
    with a small per-step constant, no surprise quadratic.
    """
    path, key, _ = wide_trace
    trace = replay(path)
    t0 = time.perf_counter()
    result = trace.replay_forward(Executor())
    elapsed = time.perf_counter() - t0
    assert result.cache_hit_count == N_BRANCHES + 5
    # Hard ceiling: 1,005 cached steps must replay in well under 30s
    # even on slow CI runners. Empirically ~1-3s on a laptop.
    assert elapsed < 30.0, f"cached replay took {elapsed:.2f}s"
    # Throughput floor: at least 30 steps/sec on cached path.
    sps = (N_BRANCHES + 5) / max(elapsed, 1e-6)
    assert sps > 30, f"only {sps:.1f} steps/sec on cached replay"


def test_wide_fanout_per_step_envelope_bounded(wide_trace):
    """Per-step on-disk amortised cost must stay bounded as N grows.

    The join frame embeds N_BRANCHES parent_step_ids and N_BRANCHES
    tail hashes — that's the only frame whose payload scales with N.
    Verify the per-step amortised size still fits a sane envelope.
    """
    path, key, steps = wide_trace
    size = os.path.getsize(path)
    n = len(steps)
    assert n == N_BRANCHES + 5
    # Permissive ceiling: the join frame alone embeds N_BRANCHES
    # step_ids + N_BRANCHES tail hashes, which dominates the trace.
    # Amortised per-step <2 KB is the contract.
    per_step = size / n
    assert per_step < 2000, f"per-step bytes {per_step:.1f} > 2000"
    # Sanity floor: a non-trivial signed trace must exceed 100 KB.
    assert size > 100_000, size


def test_dirty_propagates_through_join_not_across_siblings(wide_trace):
    """Structural invariant: dirty must reach the join via the
    substituted branch's tail, but must not leak across sibling
    branches under any single-branch substitution.

    Probes three independent branch indices to rule out any
    position-dependent bug (off-by-one, modular indexing, etc.).
    """
    path, key, steps = wide_trace
    join_id = steps[3 + N_BRANCHES]["step_id"]
    synth_id = steps[-1]["step_id"]

    for probe_idx in (1, N_BRANCHES // 3, (2 * N_BRANCHES) // 3):
        target_id = steps[3 + probe_idx]["step_id"]
        trace = replay(path)
        trace.substitute(ToolOutputSubstitution(
            at_step=target_id,
            fake_response={"topic_idx": probe_idx, "value": -probe_idx},
        ))
        result = trace.replay_forward(
            Executor(llm=_fake_llm, tool=_fake_tool)
        )
        by_id = {s.step_id: s for s in result.steps}

        # Exactly target + join + synthesise are dirty.
        dirty_ids = sorted(s.step_id for s in result.steps if s.dirty)
        assert dirty_ids == sorted([target_id, join_id, synth_id]), (
            probe_idx, dirty_ids[:10], len(dirty_ids),
        )
        # Sibling sample: spot-check 5 siblings near and far from target.
        for delta in (-2, -1, 1, 2, N_BRANCHES // 4):
            sib_idx = (probe_idx + delta) % N_BRANCHES
            if sib_idx == probe_idx:
                continue
            sib_id = steps[3 + sib_idx]["step_id"]
            assert by_id[sib_id].cache_hit, (probe_idx, sib_idx)
            assert not by_id[sib_id].dirty, (probe_idx, sib_idx)


def test_multi_branch_substitution_dirty_set_linear_in_subs(wide_trace):
    """Substituting K independent branches must dirty exactly K branch
    tails + the join + the synthesise step (K + 2 total).

    This pins the contract that dirty-set growth is *linear in the
    number of substitutions*, not in N_BRANCHES. K is chosen well
    below N_BRANCHES so the cache-hit majority remains overwhelming.
    """
    path, key, steps = wide_trace
    join_id = steps[3 + N_BRANCHES]["step_id"]
    synth_id = steps[-1]["step_id"]

    # Pick K branches spread across the fan-out so positional bugs
    # surface (first, mid, last, plus a couple in between).
    K_indices = [0, 1, N_BRANCHES // 4, N_BRANCHES // 2,
                 (3 * N_BRANCHES) // 4, N_BRANCHES - 2, N_BRANCHES - 1]
    K = len(K_indices)
    assert len({i for i in K_indices}) == K  # distinct
    assert K < N_BRANCHES  # still way fewer than N

    trace = replay(path)
    for idx in K_indices:
        trace.substitute(ToolOutputSubstitution(
            at_step=steps[3 + idx]["step_id"],
            fake_response={"topic_idx": idx, "value": -idx - 1},
        ))
    result = trace.replay_forward(Executor(llm=_fake_llm, tool=_fake_tool))
    by_id = {s.step_id: s for s in result.steps}

    expected_dirty = {steps[3 + idx]["step_id"] for idx in K_indices}
    expected_dirty.add(join_id)
    expected_dirty.add(synth_id)

    actual_dirty = {s.step_id for s in result.steps if s.dirty}
    assert actual_dirty == expected_dirty, (
        len(actual_dirty), len(expected_dirty),
        actual_dirty.symmetric_difference(expected_dirty),
    )
    # Exactly K + 2 dirty: K branch tails + join + synthesise.
    assert len(actual_dirty) == K + 2

    # All untouched siblings must remain cache hits.
    untouched = [steps[3 + i]["step_id"] for i in range(N_BRANCHES)
                 if i not in K_indices]
    assert len(untouched) == N_BRANCHES - K
    assert all(by_id[sid].cache_hit for sid in untouched)
    assert all(not by_id[sid].dirty for sid in untouched)

    # Real-execution accounting: K ToolOutputSubstitutions are satisfied
    # without executor work, so only join + synthesise re-execute.
    assert result.real_executions == 2

    # Cache-survival ratio still ≥ 99% under K-branch substitution.
    cached = sum(1 for s in result.steps if s.cache_hit)
    assert cached >= (N_BRANCHES + 5) - (K + 2)
    assert cached / (N_BRANCHES + 5) > 0.99


def test_synthesise_substitution_dirties_only_synthesise(wide_trace):
    """Substituting the *terminal* synthesise step must dirty exactly
    that one step and leave the entire fan-out — including the join —
    as cache hits.

    Catches regressions where dirtying a leaf accidentally invalidates
    its parents (upstream propagation is forbidden).
    """
    path, key, steps = wide_trace
    synth_id = steps[-1]["step_id"]
    open_id = steps[2]["step_id"]
    join_id = steps[3 + N_BRANCHES]["step_id"]

    trace = replay(path)
    # Use PromptSubstitution rather than ToolOutputSubstitution so we
    # exercise a different substitution kind on the LLM tail.
    from stepback.substitutions import PromptSubstitution
    trace.substitute(PromptSubstitution(
        at_step=synth_id,
        new_messages=[
            {"role": "system", "content": "alt-prompt"},
            {"role": "user", "content": "alt-question"},
        ],
    ))
    result = trace.replay_forward(Executor(llm=_fake_llm, tool=_fake_tool))
    by_id = {s.step_id: s for s in result.steps}

    dirty_ids = [s.step_id for s in result.steps if s.dirty]
    assert dirty_ids == [synth_id], dirty_ids

    # Open, join, all branch tails, plan, split must be cache hits.
    assert by_id[open_id].cache_hit
    assert by_id[join_id].cache_hit
    assert by_id[steps[0]["step_id"]].cache_hit
    assert by_id[steps[1]["step_id"]].cache_hit
    branch_tails = [steps[3 + i]["step_id"] for i in range(N_BRANCHES)]
    assert all(by_id[bid].cache_hit for bid in branch_tails)
    assert all(not by_id[bid].dirty for bid in branch_tails)


def test_split_substitution_does_not_silently_swallow_widespread_dirty(wide_trace):
    """Negative-control sibling of the main contract: substituting an
    *upstream* step that the entire fan-out depends on must dirty the
    full DAG below it (open + every branch tail + join + synthesise).

    Without this, a buggy implementation that simply marked everything
    cache-hit would silently pass all the "siblings stay cached" tests
    while violating soundness. We explicitly verify the dirty-set is
    NOT artificially small when it should be large.
    """
    path, key, steps = wide_trace
    split_id = steps[1]["step_id"]   # tool_call: split_question
    open_id = steps[2]["step_id"]
    join_id = steps[3 + N_BRANCHES]["step_id"]
    synth_id = steps[-1]["step_id"]

    trace = replay(path)
    trace.substitute(ToolOutputSubstitution(
        at_step=split_id,
        fake_response={"topics": [f"topic_{i}" for i in range(N_BRANCHES)]},
    ))
    result = trace.replay_forward(Executor(llm=_fake_llm, tool=_fake_tool))
    by_id = {s.step_id: s for s in result.steps}

    # The substituted split + open + every branch tail + join + synth
    # must all be dirty (the substituted result happens to be value-
    # equal, but the dirty-propagation contract is hash-based, not
    # value-based, so they all flip dirty).
    must_be_dirty = [
        split_id,
        open_id,
        join_id,
        synth_id,
        *(steps[3 + i]["step_id"] for i in range(N_BRANCHES)),
    ]
    dirty_ids = {s.step_id for s in result.steps if s.dirty}
    for sid in must_be_dirty:
        assert sid in dirty_ids, sid

    # Plan (step 0) is upstream of the substituted step — must NOT be dirty.
    assert by_id[steps[0]["step_id"]].cache_hit
    assert not by_id[steps[0]["step_id"]].dirty

    # Total dirty count: 1 (split) + 1 (open) + N (branches) + 1 (join)
    # + 1 (synth) = N + 4.
    assert len(dirty_ids) == N_BRANCHES + 4


def test_repeated_substitute_then_revert_yields_full_cache(wide_trace):
    """A `replay()` of a fresh trace handle, with NO substitutions,
    must always produce zero dirty steps — even after a previous
    substitution exercise on a sibling handle. Pins handle isolation:
    no per-process global cache invalidation can leak across traces.
    """
    path, key, steps = wide_trace
    target = steps[3 + N_BRANCHES // 2]["step_id"]

    # Run a substituted replay on one handle.
    dirty_trace = replay(path)
    dirty_trace.substitute(ToolOutputSubstitution(
        at_step=target,
        fake_response={"topic_idx": N_BRANCHES // 2, "value": 12345},
    ))
    dirty_trace.replay_forward(Executor(llm=_fake_llm, tool=_fake_tool))

    # A *fresh* replay handle with no substitutions must see zero dirty.
    fresh = replay(path)
    fresh_result = fresh.replay_forward(Executor())
    assert fresh_result.dirty_count == 0
    assert fresh_result.real_executions == 0
    assert fresh_result.cache_hit_count == N_BRANCHES + 5
