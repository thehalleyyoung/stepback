"""End-to-end parallel-branch fixture test.

Exercises the new `parallel_branch_open` / `parallel_branch_join`
step kinds: records an 11-step research agent that fans out into 3
parallel sub-investigations and joins. Asserts on:

  1. Recorded trace shape (kinds, parent edges, multi-parent join).
  2. Cached replay is zero-LLM, all hits.
  3. A `ToolOutputSubstitution` inside ONE branch dirties only that
     branch's tail + the join + downstream synthesise call —
     sibling branches stay cached. This is the whole point of
     fan-out/fan-in modelling: dirtiness flows through join, not
     across siblings.
  4. The trace round-trips through the signed `.sb` writer/reader and
     verifies under HMAC + Ed25519.
  5. compare_branches between cached vs counterfactual reports the
     join + synthesise as divergent, sibling branches as not.
"""
from __future__ import annotations

import os

from stepback import record, replay, RecorderKey
from stepback.replay import Executor
from stepback.substitutions import ToolOutputSubstitution
from stepback.trace_reader import verify_trace

from .fixtures.parallel_agent import (
    FACTS,
    fake_llm,
    fake_tool,
    run_parallel_agent,
)


# --------------------------------------------------------- helpers


def _record_trace(tmp_path) -> tuple:
    path = os.path.join(str(tmp_path), "parallel.sb")
    key = RecorderKey.fresh()
    with record(path, key=key) as rec:
        run_parallel_agent(rec)
        recorded_steps = list(rec.steps)
    return path, key, recorded_steps


# ----------------------------------------------------------- tests


def test_recorded_trace_shape(tmp_path):
    path, key, steps = _record_trace(tmp_path)

    # Expected 11 steps total.
    assert len(steps) == 11, [s["step_kind"] for s in steps]

    kinds = [s["step_kind"] for s in steps]
    assert kinds == [
        "llm_call",            # 1 plan
        "tool_call",           # 2 split
        "parallel_branch_open",  # 3
        "llm_call",            # 4 branch A llm
        "tool_call",           # 5 branch A tool
        "llm_call",            # 6 branch B llm
        "tool_call",           # 7 branch B tool
        "llm_call",            # 8 branch C llm
        "tool_call",           # 9 branch C tool
        "parallel_branch_join",  # 10
        "llm_call",            # 11 synthesise
    ]

    open_step = steps[2]
    join_step = steps[9]
    assert open_step["parent_step_id"] == "step:2"
    # Each branch's first step must have the open as its parent.
    assert steps[3]["parent_step_id"] == "step:3"
    assert steps[5]["parent_step_id"] == "step:3"
    assert steps[7]["parent_step_id"] == "step:3"
    # Each branch's tail (the lookup_*) parents are the matching llm_call.
    assert steps[4]["parent_step_id"] == "step:4"
    assert steps[6]["parent_step_id"] == "step:6"
    assert steps[8]["parent_step_id"] == "step:8"

    # Join carries multi-parent edges.
    assert join_step["parent_step_id"] == "step:3"
    assert join_step["parent_step_ids"] == ["step:5", "step:7", "step:9"]
    # Synthesise step's parent is the join.
    assert steps[10]["parent_step_id"] == "step:10"

    # Branch names propagated.
    assert open_step["outputs"]["branch_names"] == [
        f"research/{t}" for t in FACTS.keys()
    ]


def test_signed_trace_verifies_round_trip(tmp_path):
    path, key, steps = _record_trace(tmp_path)
    verified = verify_trace(path, key.hmac_key)
    kinds = [s["step_kind"] for s in verified.steps]
    assert "parallel_branch_open" in kinds
    assert "parallel_branch_join" in kinds
    # Multi-parent edge survives the round trip.
    join = next(s for s in verified.steps if s["step_kind"] == "parallel_branch_join")
    assert len(join["parent_step_ids"]) == 3


def test_cached_replay_is_zero_llm(tmp_path):
    path, key, _ = _record_trace(tmp_path)
    trace = replay(path)
    result = trace.replay_forward(Executor())
    assert result.real_executions == 0
    assert result.dirty_count == 0
    assert result.cache_hit_count == 11


def test_substitution_in_one_branch_does_not_dirty_siblings(tmp_path):
    path, key, _ = _record_trace(tmp_path)
    trace = replay(path)
    # Override the economics branch's lookup tool (step:7).
    fake_econ = {"answer": "GDP 25T USD (revised)", "confidence": 0.95}
    trace.substitute(ToolOutputSubstitution(
        at_step="step:7",
        fake_response=fake_econ,
    ))
    result = trace.replay_forward(Executor(llm=fake_llm, tool=fake_tool))

    by_id = {s.step_id: s for s in result.steps}

    # Branch A (history) tail step:5 — sibling — must stay cached.
    assert by_id["step:5"].cache_hit, "history branch tail should be untouched"
    assert not by_id["step:5"].dirty
    # Branch C (politics) tail step:9 — sibling — must stay cached.
    assert by_id["step:9"].cache_hit, "politics branch tail should be untouched"
    assert not by_id["step:9"].dirty
    # Branch A and C inner llm_call steps unchanged too.
    assert by_id["step:4"].cache_hit
    assert by_id["step:8"].cache_hit

    # Branch B's tool step:7 must be dirty (forced by the substitution).
    assert by_id["step:7"].dirty
    assert by_id["step:7"].outputs == {"result": fake_econ}

    # The join step:10 must be dirty (multi-parent dirty propagation).
    assert by_id["step:10"].dirty, "join must dirty when ANY branch tail dirties"
    # Synthesise must also be dirty (depends on join's output via context).
    assert by_id["step:11"].dirty

    # Open step:3 must NOT be dirty — the fan-out structure didn't change.
    assert by_id["step:3"].cache_hit

    # Cost-of-debug accounting: dirty subtree should be SMALL, not the
    # whole 11-step trace. Specifically: only the substituted tool, the
    # join, and the downstream synthesise — sibling branches stay cached.
    dirty_ids = sorted(s.step_id for s in result.steps if s.dirty)
    assert dirty_ids == ["step:10", "step:11", "step:7"], dirty_ids

    # The only REAL LLM re-execution should be the synthesise call.
    # (The join is recomputed by Executor.execute but is not an LLM call;
    # the dirty tool was satisfied by the ToolOutputSubstitution without
    # invoking the tool executor.)
    assert result.real_executions == 2  # join (parallel_branch_join) + synthesise llm


def test_compare_branches_flags_join_and_synthesise_as_divergent(tmp_path):
    path, key, _ = _record_trace(tmp_path)
    trace = replay(path)
    cached = trace.branch_at("step:1", name="cached")
    cached.replay_forward(Executor())

    cf = trace.branch_at("step:1", name="counterfactual")
    cf.substitute(ToolOutputSubstitution(
        at_step="step:7",
        fake_response={"answer": "GDP 25T USD (revised)", "confidence": 0.95},
    ))
    cf.replay_forward(Executor(llm=fake_llm, tool=fake_tool))

    diff = trace.compare_branches(cached, cf)
    by_id = {d.step_id: d for d in diff.step_diffs}
    diverged_ids = {sid for sid, d in by_id.items() if d.output_diff}
    # The substituted tool and the join must show different OUTPUTS.
    assert "step:7" in diverged_ids
    assert "step:10" in diverged_ids
    # Synthesise (step:11) was re-executed (dirty) but its inputs.messages
    # don't textually depend on the join hash — fake_llm hashes the message
    # list and converges on the same output. That's a legitimate cache
    # convergence: the diff flags it as `diverged_from_cache=True`
    # (re-executed under counterfactual) even though `output_diff` is empty.
    assert by_id["step:11"].diverged_from_cache
    # Sibling branches must NOT diverge in output AND must not be marked
    # as having diverged-from-cache.
    assert "step:5" not in diverged_ids
    assert "step:9" not in diverged_ids
    assert not by_id["step:5"].diverged_from_cache
    assert not by_id["step:9"].diverged_from_cache


def test_parallel_branch_with_no_steps_raises(tmp_path):
    path = os.path.join(str(tmp_path), "empty.sb")
    with record(path) as rec:
        rec.llm_call("gpt-4o", [{"role": "user", "content": "hi"}], executor=fake_llm)
        try:
            rec.parallel("bad_fanout", [lambda r: None])
        except RuntimeError as e:
            assert "no steps" in str(e)
        else:  # pragma: no cover
            raise AssertionError("expected RuntimeError for empty branch")
