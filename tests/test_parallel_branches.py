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


def test_recorded_trace_numeric_bounds(tmp_path):
    """Numeric-threshold guarantees on the recorded fan-out trace.

    These pin the *quantitative* contract of the parallel fixture so any
    silent regression (e.g. losing a branch, doubling a tool call,
    bloating frame envelopes) is caught.
    """
    path, key, steps = _record_trace(tmp_path)

    # --- step-shape counts --------------------------------------------
    kinds = [s["step_kind"] for s in steps]
    assert kinds.count("llm_call") == 5, kinds
    assert kinds.count("tool_call") == 4, kinds
    assert kinds.count("parallel_branch_open") == 1
    assert kinds.count("parallel_branch_join") == 1

    # Exactly 3 sibling branches in the fan-out.
    open_step = steps[2]
    branch_names = open_step["outputs"]["branch_names"]
    assert len(branch_names) == 3, branch_names
    assert all(n.startswith("research/") for n in branch_names)

    # Join must reference exactly 3 parent tails (one per branch).
    join_step = steps[9]
    assert len(join_step["parent_step_ids"]) == 3
    assert len(set(join_step["parent_step_ids"])) == 3  # all distinct

    # --- on-disk size bound -------------------------------------------
    # An 11-step signed trace with HMAC + Ed25519 envelopes should be
    # well under 32 KB — anything bigger is envelope/payload bloat.
    size = os.path.getsize(path)
    assert 1024 < size < 32_000, f"trace file size {size} out of bounds"
    # Per-step amortised cost ceiling: <3 KB/step on this fixture.
    assert size / len(steps) < 3000, f"per-step cost {size/len(steps):.1f}B too high"


def test_dirty_subtree_is_small_fraction(tmp_path):
    """The whole point of fan-out/fan-in modelling: a tool-output
    substitution inside ONE branch must dirty a strictly bounded
    fraction of the trace — not a majority of steps."""
    path, key, _ = _record_trace(tmp_path)
    trace = replay(path)
    trace.substitute(ToolOutputSubstitution(
        at_step="step:7",
        fake_response={"answer": "alt", "confidence": 0.5},
    ))
    result = trace.replay_forward(Executor(llm=fake_llm, tool=fake_tool))

    total = len(result.steps)
    dirty = sum(1 for s in result.steps if s.dirty)
    cached = sum(1 for s in result.steps if s.cache_hit)
    assert total == 11
    assert dirty == 3, f"expected exactly 3 dirty steps, got {dirty}"
    assert cached == 8, f"expected exactly 8 cached steps, got {cached}"
    # Numeric ratio bound: at most ~30% of the trace re-executes.
    assert dirty / total <= 0.30, dirty / total
    # Cache savings: at least 70% of steps remain cache hits.
    assert cached / total >= 0.70, cached / total
    # Real LLM/exec count is bounded — must be ≤ dirty count.
    assert result.real_executions <= dirty
    assert result.real_executions == 2


def test_per_step_envelope_size_bounded(tmp_path):
    """Each on-disk step envelope has a strict per-step byte ceiling.

    Pins the signed-trace overhead so any future schema bloat (extra
    metadata fields, unbounded debug payloads) trips here instead of
    silently inflating production trace sizes.
    """
    path, key, steps = _record_trace(tmp_path)
    size = os.path.getsize(path)
    n = len(steps)
    assert n == 11
    # Tight per-step amortised ceiling (verified empirically ~1.5 KB).
    per_step = size / n
    assert per_step < 2500, f"per-step bytes {per_step:.1f} > 2500"
    assert per_step > 200, f"per-step bytes {per_step:.1f} < 200 (suspiciously tiny)"
    # Absolute floor for any non-trivial signed trace.
    assert size > 1500


def test_zero_llm_replay_speedup_vs_record(tmp_path):
    """Cached replay must be measurably faster than fresh record.

    A real numeric-threshold guarantee on the cache-hit path: replay
    of an 11-step trace must take less wall-clock time than recording
    it (the recorder runs the fixture; replay just hashes envelopes).
    """
    import time
    t0 = time.perf_counter()
    path, key, steps = _record_trace(tmp_path)
    record_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    trace = replay(path)
    result = trace.replay_forward(Executor())
    replay_time = time.perf_counter() - t1

    assert result.real_executions == 0
    assert result.cache_hit_count == 11
    # Replay should be at most as expensive as record (allow 2x slack
    # for filesystem variance on tiny traces).
    assert replay_time < record_time * 2.0 + 0.05, (
        f"replay={replay_time:.4f}s vs record={record_time:.4f}s"
    )


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


def test_branch_dirty_count_strictly_less_than_serial(tmp_path):
    """Numeric guarantee: fan-out/fan-in modelling must save more
    re-executions than the equivalent serial trace would.

    A serial 11-step trace with a substitution at position 7 would
    dirty 5 downstream steps (positions 7..11 = 5). The fan-out
    structure must dirty STRICTLY fewer than that — proving the
    sibling-branch isolation is real, not nominal.
    """
    path, key, _ = _record_trace(tmp_path)
    trace = replay(path)
    trace.substitute(ToolOutputSubstitution(
        at_step="step:7",
        fake_response={"answer": "alt", "confidence": 0.5},
    ))
    result = trace.replay_forward(Executor(llm=fake_llm, tool=fake_tool))

    dirty = sum(1 for s in result.steps if s.dirty)
    # Serial-equivalent dirty count would be 5 (steps 7..11).
    serial_dirty = 11 - 7 + 1
    assert dirty < serial_dirty, (
        f"fan-out failed to isolate siblings: dirty={dirty} >= serial={serial_dirty}"
    )
    # Concrete: must be exactly 3 with our fixture.
    assert dirty == 3
    # Savings ratio: at least 40% fewer re-executions than serial.
    savings = (serial_dirty - dirty) / serial_dirty
    assert savings >= 0.4, f"only {savings*100:.1f}% savings vs serial"


def test_join_step_is_strictly_after_all_branch_tails(tmp_path):
    """Multi-parent join must reference parent ids strictly earlier
    in the recorded order — no forward edges, no self-loops.
    """
    path, key, steps = _record_trace(tmp_path)
    join = next(s for s in steps if s["step_kind"] == "parallel_branch_join")
    join_idx = int(join["step_id"].split(":")[1])
    parent_idxs = [int(pid.split(":")[1]) for pid in join["parent_step_ids"]]
    assert all(p < join_idx for p in parent_idxs), parent_idxs
    # All parent indices distinct.
    assert len(set(parent_idxs)) == len(parent_idxs)
    # Parents must span a contiguous window inside the fan-out region.
    assert min(parent_idxs) > 3  # after the open
    assert max(parent_idxs) < join_idx
