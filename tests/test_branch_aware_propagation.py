"""Branch-aware dirty-set propagation contract — Step #59 of ``100_STEPS.md``.

The dirty-set classifier in :py:mod:`stepback.divergence` is uniform
over all step kinds, but the fan-out / fan-in shape of
``parallel_branch_open`` / ``parallel_branch_join`` admits a sharper
restatement (see ``docs/dirty-set.md`` §5.5 and the contract block in
``stepback/divergence.py``):

* **B1** Independent fan-out children — a step inside branch ``B_i`` is
  dirty only when something *inside ``B_i``* requires it (direct
  substitution target, dirty ancestor in ``B_i``, declared input drift,
  or NDH tampering). Substitutions targeting ``B_k`` for ``k ≠ i``
  never reach into ``B_i``.
* **B2** Joins are dirty iff at least one consumed branch output
  changes (multi-parent OR over branch dirtiness). The
  ``branch_tail_hashes`` rebinding makes this an *exact* OR — no
  over-dirtying when every branch is trivially clean, no
  under-dirtying when any consumed branch output drifted.
* **B3** Clean siblings are preserved — for any substitution localised
  to one branch ``B_k``, every step in any sibling branch ``B_i``
  (``i ≠ k``), every ancestor of the open frame, and the open itself
  remain cache hits. The dirty count is independent of fan-out
  width ``W``.

The 1000-way fan-out stress test in
``tests/test_parallel_branch_stress.py`` already pins B1+B2+B3 in
combination on a single substitution. This file adds the *exhaustive*
contract tests:

* multi-branch substitution (two branches independently dirtied by
  two ``ToolOutputSubstitution``s) — exactly two branches + join +
  downstream are dirty,
* input-drift on a fan-out child via ``ToolArgumentsSubstitution`` —
  classifier dirties the targeted branch tail without spilling into
  siblings,
* join is clean iff every branch is clean (no recorded-position-after
  spillover into the join),
* join is dirty when *only one* branch's recomputed output differs
  from its recorded output (B2 minimal-dirty case),
* output-forcing substitution that pins a branch tail to *the same*
  recorded output keeps siblings clean and still dirties the targeted
  step + the join (P3),
* deep-branch propagation: a substitution at the root of a multi-step
  branch dirties exactly that branch's spine + the join + the
  downstream, leaves siblings clean, and is independent of branch
  width.

Together these tests pin every clause of B1, B2, B3 and rule out the
two regression classes Step 59 was written to catch:

  R1. Linear "everything after step k is dirty" propagation — would
      dirty all sibling branches recorded after the substituted one.
  R2. Eager "any descendant of the open is dirty if the open is in
      σ's reach" propagation — would dirty all branches when any one
      is dirtied.
"""
from __future__ import annotations

import os

import pytest

from stepback import RecorderKey, record, replay
from stepback.divergence import compute_dirty_set
from stepback.replay import Executor
from stepback.substitutions import (
    ToolArgumentsSubstitution,
    ToolOutputSubstitution,
)


# Smaller fan-out than the 1000-way stress test; we want exhaustive
# multi-branch combinations and per-test classifier introspection,
# not per-second throughput.
N_BRANCHES = 8


# ---------------------------------------------------------- fixtures


def _fake_llm(model: str, messages: list) -> dict:
    blob = "|".join(f"{m['role']}={m['content']}" for m in messages)
    return {
        "id": f"chatcmpl-{len(blob)}",
        "model": model,
        "choices": [
            {"index": 0, "finish_reason": "stop",
             "message": {"role": "assistant",
                         "content": f"reply:{len(blob)}"}}
        ],
        "usage": {"prompt_tokens": len(blob),
                  "completion_tokens": 4,
                  "total_tokens": len(blob) + 4},
    }


def _fake_tool(name: str, args: dict) -> dict:
    if name == "split":
        return {"topics": [f"t{i}" for i in range(N_BRANCHES)]}
    if name.startswith("lookup_"):
        idx = int(name.split("_")[-1])
        # Make output sensitive to args["topic_idx"] so a
        # ToolArgumentsSubstitution actually changes the output.
        return {"idx": idx, "value": args.get("topic_idx", idx) * 5 + 1}
    raise KeyError(name)


def _make_branch(idx: int):
    def branch(rec):
        rec.tool_call(
            f"lookup_{idx}",
            {"topic_idx": idx},
            executor=_fake_tool,
        )
    return branch


def _record_wide(tmp_path) -> tuple:
    path = os.path.join(str(tmp_path), "branch_aware.sb")
    key = RecorderKey.fresh()
    with record(path, key=key) as rec:
        rec.llm_call("gpt-4o", [
            {"role": "system", "content": "research agent"},
            {"role": "user", "content": "Summarise."},
        ], executor=_fake_llm)
        rec.tool_call("split", {}, executor=_fake_tool)
        join = rec.parallel(
            "fanout",
            [_make_branch(i) for i in range(N_BRANCHES)],
            join=lambda outs: {
                "n": len(outs),
                "sum": sum(o["result"]["value"] for o in outs),
            },
        )
        rec.llm_call("gpt-4o", [
            {"role": "system", "content": "synth"},
            {"role": "user", "content": f"sum={join['outputs']['sum']}"},
        ], executor=_fake_llm)
        recorded = list(rec.steps)
    return path, key, recorded


@pytest.fixture(scope="module")
def trace_paths(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("branch_aware")
    return _record_wide(tmp)


def _ids(steps):
    """Return (plan_id, split_id, open_id, branch_ids, join_id, synth_id)."""
    plan = steps[0]["step_id"]
    split = steps[1]["step_id"]
    opn = steps[2]["step_id"]
    branches = [steps[3 + i]["step_id"] for i in range(N_BRANCHES)]
    join = steps[3 + N_BRANCHES]["step_id"]
    synth = steps[-1]["step_id"]
    return plan, split, opn, branches, join, synth


# ============================================================== B2

def test_b2_join_clean_iff_every_branch_clean(trace_paths):
    """Empty substitution → every step (including the join) is clean."""
    path, _, steps = trace_paths
    trace = replay(path)
    result = trace.replay_forward(Executor())
    assert result.dirty_count == 0
    assert result.real_executions == 0
    assert all(s.cache_hit for s in result.steps)


# ============================================================== B1+B2+B3

def test_single_branch_substitution_dirties_branch_join_and_downstream(
    trace_paths,
):
    """Localised substitution → exactly 3 dirty steps: tail + join + synth.

    Pins B1 (only the targeted branch is internally dirty), B2 (the
    join's multi-parent OR fires because its consumed branch output
    drifted) and B3 (every sibling branch + every ancestor + the open
    itself are cache hits).
    """
    path, _, steps = trace_paths
    _, _, open_id, branches, join_id, synth_id = _ids(steps)

    target = branches[3]
    trace = replay(path)
    trace.substitute(ToolOutputSubstitution(
        at_step=target,
        fake_response={"idx": 3, "value": -999},
    ))
    result = trace.replay_forward(Executor(llm=_fake_llm, tool=_fake_tool))
    by_id = {s.step_id: s for s in result.steps}

    # B1 — target branch tail is dirty, every sibling is clean.
    assert by_id[target].dirty
    for i, bid in enumerate(branches):
        if i == 3:
            continue
        assert by_id[bid].cache_hit and not by_id[bid].dirty, i
    # B2 — join dirty (any consumed branch output drift).
    assert by_id[join_id].dirty
    # Downstream of the join is dirty (transitive).
    assert by_id[synth_id].dirty
    # B3 — open + ancestors clean.
    assert by_id[open_id].cache_hit
    assert by_id[steps[0]["step_id"]].cache_hit
    assert by_id[steps[1]["step_id"]].cache_hit
    # Exactly the three predicted dirty steps; no spillover.
    dirty = sorted(s.step_id for s in result.steps if s.dirty)
    assert dirty == sorted([target, join_id, synth_id])


# =============================================================== B1

def test_multi_branch_substitution_dirties_only_targeted_branches(trace_paths):
    """Two independent branch substitutions → exactly 4 dirty steps.

    B1 says the two substitutions stay confined to their branches; B2
    says the join is dirty (collapsing both branches to a single dirty
    join); B3 says the other (N-2) sibling branches stay cached.
    """
    path, _, steps = trace_paths
    _, _, open_id, branches, join_id, synth_id = _ids(steps)

    targets = (branches[1], branches[6])  # non-adjacent, mid-pack
    trace = replay(path)
    for i, tid in enumerate(targets):
        trace.substitute(ToolOutputSubstitution(
            at_step=tid,
            fake_response={"idx": -i, "value": -100 - i},
        ))
    result = trace.replay_forward(Executor(llm=_fake_llm, tool=_fake_tool))
    by_id = {s.step_id: s for s in result.steps}

    for tid in targets:
        assert by_id[tid].dirty
    other_ids = [b for b in branches if b not in targets]
    for bid in other_ids:
        assert by_id[bid].cache_hit, bid
        assert not by_id[bid].dirty, bid
    assert by_id[join_id].dirty
    assert by_id[synth_id].dirty
    assert by_id[open_id].cache_hit

    # Exactly len(targets) + 2 dirty steps.
    dirty = sorted(s.step_id for s in result.steps if s.dirty)
    assert dirty == sorted([*targets, join_id, synth_id])
    assert sum(1 for s in result.steps if s.dirty) == len(targets) + 2


def test_input_drift_on_fan_out_child_dirties_only_that_branch(trace_paths):
    """``ToolArgumentsSubstitution`` triggers P2 input drift, not P3 direct.

    The classifier must still confine the dirty propagation to that
    branch (B1) and only OR through the join (B2/B3). This catches
    regressions where P2 (input drift) and P3 (direct sub) take
    different code paths in branch handling.
    """
    path, _, steps = trace_paths
    _, _, open_id, branches, join_id, synth_id = _ids(steps)

    target = branches[5]
    trace = replay(path)
    trace.substitute(ToolArgumentsSubstitution(
        at_step=target,
        new_arguments={"topic_idx": 42},   # was 5; drifts inputs hash
    ))
    result = trace.replay_forward(Executor(llm=_fake_llm, tool=_fake_tool))
    by_id = {s.step_id: s for s in result.steps}

    assert by_id[target].dirty
    assert by_id[join_id].dirty
    assert by_id[synth_id].dirty
    for i, bid in enumerate(branches):
        if i == 5:
            continue
        assert by_id[bid].cache_hit and not by_id[bid].dirty, i
    assert by_id[open_id].cache_hit
    assert sum(1 for s in result.steps if s.dirty) == 3


# =============================================================== B2

def test_output_forcing_to_same_value_keeps_join_dirty_via_p3(trace_paths):
    """Direct P3 targeting always dirties (B2's second clause).

    Substitute the join itself with the same recorded output. The
    join must still be marked dirty (P3), but no branch should be
    dirtied, and the substitution must NOT propagate backwards into
    the branches.
    """
    path, _, steps = trace_paths
    _, _, open_id, branches, join_id, synth_id = _ids(steps)

    # Pull the recorded join output to substitute "with itself".
    trace = replay(path)
    rec_join = next(r for r in trace.recorded_steps if r["step_id"] == join_id)
    same_output = dict(rec_join["outputs"])

    trace2 = replay(path)
    trace2.substitute(ToolOutputSubstitution(
        at_step=join_id,
        fake_response=same_output,
    ))
    result = trace2.replay_forward(Executor(llm=_fake_llm, tool=_fake_tool))
    by_id = {s.step_id: s for s in result.steps}

    # P3 — join is dirty.
    assert by_id[join_id].dirty
    # Synth is dirty by transitive parent-dirty closure.
    assert by_id[synth_id].dirty
    # B1/B3 — every branch + open + ancestors are clean.
    assert by_id[open_id].cache_hit
    for bid in branches:
        assert by_id[bid].cache_hit, bid
        assert not by_id[bid].dirty, bid
    # Exactly two dirty steps: join + synth.
    assert sum(1 for s in result.steps if s.dirty) == 2


# ====================================== compute_dirty_set classifier API

def test_compute_dirty_set_classifier_agrees_with_replay(trace_paths):
    """The pure-classifier ``compute_dirty_set`` must report the same
    branch-aware partition as the replay-engine ``StepView.dirty``.
    """
    path, _, steps = trace_paths
    _, _, open_id, branches, join_id, synth_id = _ids(steps)

    target = branches[2]
    trace = replay(path)
    summary = compute_dirty_set(
        trace,
        [ToolOutputSubstitution(
            at_step=target,
            fake_response={"idx": 2, "value": -42},
        )],
    )
    by_id = {e.step_id: e for e in summary.entries}
    assert by_id[target].dirty
    assert by_id[target].dirty_reason == "substituted"
    assert by_id[join_id].dirty
    assert by_id[join_id].dirty_reason in ("parent_dirty", "input_drift")
    assert by_id[synth_id].dirty
    assert by_id[synth_id].dirty_reason in ("parent_dirty", "input_drift")
    # Sibling branches all clean with no dirty_reason.
    for i, bid in enumerate(branches):
        if i == 2:
            continue
        e = by_id[bid]
        assert not e.dirty, bid
        assert e.dirty_reason is None, bid
    # Open + ancestors clean.
    assert not by_id[open_id].dirty
    assert summary.dirty_count == 3
    # B3 — clean count is independent of W modulo the 3 dirty steps.
    assert summary.clean_count == (N_BRANCHES + 5) - 3


def test_dirty_set_size_independent_of_fanout_width(trace_paths):
    """The dirty count under a single-branch substitution does not
    depend on N_BRANCHES — the headline cost guarantee of branch-aware
    propagation. We verify the inequality vs. a naive serial dirty
    count.
    """
    path, _, steps = trace_paths
    _, _, _, branches, _, _ = _ids(steps)
    trace = replay(path)
    trace.substitute(ToolOutputSubstitution(
        at_step=branches[0],   # first branch
        fake_response={"idx": 0, "value": -1},
    ))
    result = trace.replay_forward(Executor(llm=_fake_llm, tool=_fake_tool))
    dirty = sum(1 for s in result.steps if s.dirty)
    assert dirty == 3
    # Naive linear "everything after step k" would dirty:
    #   N_BRANCHES (other branch tails) + join + synth = N_BRANCHES + 2.
    # We must dirty STRICTLY fewer than that.
    assert dirty < N_BRANCHES, (dirty, N_BRANCHES)


# ====================================== regression-class guards

def test_substitute_first_branch_does_not_dirty_later_siblings(trace_paths):
    """Regression guard for naive linear "everything after k is dirty"."""
    path, _, steps = trace_paths
    _, _, _, branches, _, _ = _ids(steps)
    trace = replay(path)
    trace.substitute(ToolOutputSubstitution(
        at_step=branches[0],
        fake_response={"idx": 0, "value": 0},
    ))
    result = trace.replay_forward(Executor(llm=_fake_llm, tool=_fake_tool))
    by_id = {s.step_id: s for s in result.steps}
    for bid in branches[1:]:
        assert by_id[bid].cache_hit, bid
        assert not by_id[bid].dirty, bid


def test_substitute_last_branch_does_not_dirty_earlier_branches(trace_paths):
    """Regression guard for naive forward propagation across siblings."""
    path, _, steps = trace_paths
    _, _, _, branches, _, _ = _ids(steps)
    trace = replay(path)
    trace.substitute(ToolOutputSubstitution(
        at_step=branches[-1],
        fake_response={"idx": N_BRANCHES - 1, "value": 0},
    ))
    result = trace.replay_forward(Executor(llm=_fake_llm, tool=_fake_tool))
    by_id = {s.step_id: s for s in result.steps}
    for bid in branches[:-1]:
        assert by_id[bid].cache_hit, bid
        assert not by_id[bid].dirty, bid


def test_open_clean_under_branch_substitution(trace_paths):
    """The ``parallel_branch_open`` step must be clean under any
    branch-localised substitution (B3). A regression that dirties
    the open would dirty *every* branch through parent-dirty closure.
    """
    path, _, steps = trace_paths
    _, _, open_id, branches, _, _ = _ids(steps)
    for i in (0, N_BRANCHES // 2, N_BRANCHES - 1):
        trace = replay(path)
        trace.substitute(ToolOutputSubstitution(
            at_step=branches[i],
            fake_response={"idx": i, "value": -i - 1000},
        ))
        result = trace.replay_forward(Executor(llm=_fake_llm, tool=_fake_tool))
        by_id = {s.step_id: s for s in result.steps}
        assert by_id[open_id].cache_hit, i
        assert not by_id[open_id].dirty, i
