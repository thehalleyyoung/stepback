"""Tests for distributed dirty-set computation (Step 64).

Validates that:

1. ``compute_dirty_set_distributed`` returns *identical* results to
   ``compute_dirty_set`` for linear traces, parallel-branch traces, and
   traces with no substitutions.
2. ``partition_dag_regions`` produces correctly structured regions for
   various trace shapes.
3. The ``workers`` parameter does not affect correctness.
4. Branch-tail-hash rebinding across region boundaries is correct: a
   substitution inside one branch should dirty the join and suffix, but
   leave sibling branches as cache hits.
5. New public symbols are exported from ``stepback``.
"""
from __future__ import annotations

import os
import tempfile
from typing import List

import pytest

from stepback import RecorderKey, record, replay
from stepback.divergence import compute_dirty_set
from stepback.distributed_dirty import (
    DagRegion,
    RegionSummary,
    compute_dirty_set_distributed,
    partition_dag_regions,
)
from stepback.substitutions import PromptSubstitution, ToolOutputSubstitution


# ------------------------------------------------------------------ helpers


def _fake_llm(model: str, messages: list) -> dict:
    blob = "|".join(f"{m['role']}={m['content']}" for m in messages)
    text = f"reply:{len(blob)}:{hash(blob) & 0xFFFF:04x}"
    return {
        "id": f"chatcmpl-{len(blob)}",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
        "usage": {
            "prompt_tokens": len(blob),
            "completion_tokens": len(text),
            "total_tokens": len(blob) + len(text),
        },
    }


def _fake_tool(name: str, args: dict) -> dict:
    return {"result": name, "args": args}


def _record_linear(path: str, key: RecorderKey) -> None:
    """Record a simple 12-step linear agent."""
    with record(path, key=key) as rec:
        convo = [{"role": "user", "content": "start"}]
        for i in range(6):
            resp = rec.llm_call(
                "gpt-4o-mini",
                convo + [{"role": "user", "content": f"step {i}"}],
                executor=_fake_llm,
            )
            # resp is a step dict; resp["outputs"] is the LLM response.
            content = resp["outputs"]["choices"][0]["message"]["content"]
            rec.tool_call(
                f"tool_{i}",
                {"idx": i, "context": content},
                executor=_fake_tool,
            )


def _record_parallel(path: str, key: RecorderKey, n_branches: int = 4) -> None:
    """Record a fan-out trace with n_branches parallel branches."""
    with record(path, key=key) as rec:
        rec.tool_call("setup", {"n": n_branches}, executor=_fake_tool)
        rec.parallel(
            "fan_out",
            [
                (lambda i: lambda r: r.tool_call(
                    f"branch_{i}", {"idx": i}, executor=_fake_tool
                ))(j)
                for j in range(n_branches)
            ],
            join=lambda outs: {"n": len(outs), "branches": [o["result"]["result"] for o in outs]},
            branch_names=[f"b{j}" for j in range(n_branches)],
        )
        convo = [{"role": "user", "content": "summarise"}]
        rec.llm_call("gpt-4o-mini", convo, executor=_fake_llm)


# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def linear_trace(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("dd") / "linear.sb")
    key = RecorderKey.fresh()
    _record_linear(p, key)
    return replay(p, hmac_key=key.hmac_key), key


@pytest.fixture(scope="module")
def parallel_trace(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("dd") / "parallel.sb")
    key = RecorderKey.fresh()
    _record_parallel(p, key, n_branches=6)
    return replay(p, hmac_key=key.hmac_key), key


# ================================================================ partition_dag_regions


class TestPartitionDagRegions:

    def test_empty_steps_returns_empty(self):
        assert partition_dag_regions([]) == []

    def test_linear_trace_single_region(self, linear_trace):
        t, _ = linear_trace
        regions = partition_dag_regions(list(t.recorded_steps))
        assert len(regions) == 1
        region = regions[0]
        assert region.region_id == "seq:0"
        assert len(region.steps) == len(t.recorded_steps)
        assert region.external_parent_ids == frozenset()

    def test_parallel_trace_has_branch_regions(self, parallel_trace):
        t, _ = parallel_trace
        regions = partition_dag_regions(list(t.recorded_steps))
        # Expect: seq:0 (setup + open), branch regions, seq:1 (join + suffix)
        region_ids = [r.region_id for r in regions]
        seq_regions = [r for r in regions if r.region_id.startswith("seq:")]
        branch_regions = [r for r in regions if r.region_id.startswith("branch:")]
        assert len(seq_regions) == 2, f"expected 2 seq regions, got: {region_ids}"
        assert len(branch_regions) == 6, f"expected 6 branch regions, got: {region_ids}"

    def test_parallel_trace_prefix_contains_open_step(self, parallel_trace):
        t, _ = parallel_trace
        regions = partition_dag_regions(list(t.recorded_steps))
        prefix = next(r for r in regions if r.region_id == "seq:0")
        kinds = [s["step_kind"] for s in prefix.steps]
        assert "parallel_branch_open" in kinds

    def test_parallel_trace_suffix_contains_join_step(self, parallel_trace):
        t, _ = parallel_trace
        regions = partition_dag_regions(list(t.recorded_steps))
        suffix = next(r for r in regions if r.region_id == "seq:1")
        kinds = [s["step_kind"] for s in suffix.steps]
        assert "parallel_branch_join" in kinds

    def test_branch_regions_have_one_step_each(self, parallel_trace):
        t, _ = parallel_trace
        regions = partition_dag_regions(list(t.recorded_steps))
        branch_regions = [r for r in regions if r.region_id.startswith("branch:")]
        for reg in branch_regions:
            assert len(reg.steps) == 1, (
                f"branch region {reg.region_id!r} should have 1 step, got {len(reg.steps)}"
            )

    def test_branch_regions_external_parents_include_open(self, parallel_trace):
        t, _ = parallel_trace
        regions = partition_dag_regions(list(t.recorded_steps))
        prefix = next(r for r in regions if r.region_id == "seq:0")
        open_step_id = next(
            s["step_id"] for s in prefix.steps if s["step_kind"] == "parallel_branch_open"
        )
        branch_regions = [r for r in regions if r.region_id.startswith("branch:")]
        for reg in branch_regions:
            assert open_step_id in reg.external_parent_ids, (
                f"branch region {reg.region_id!r} should depend on open step"
            )

    def test_suffix_external_parents_include_branch_tails(self, parallel_trace):
        t, _ = parallel_trace
        regions = partition_dag_regions(list(t.recorded_steps))
        branch_regions = [r for r in regions if r.region_id.startswith("branch:")]
        branch_step_ids = {s["step_id"] for reg in branch_regions for s in reg.steps}
        suffix = next(r for r in regions if r.region_id == "seq:1")
        # The join step in the suffix depends on branch tails
        assert branch_step_ids.issubset(suffix.external_parent_ids | branch_step_ids), (
            "suffix external parents should reference branch tail step ids"
        )
        # At minimum the branch tails are in the suffix's external parents
        assert suffix.external_parent_ids & branch_step_ids, (
            "suffix must have at least some branch tail ids as external parents"
        )

    def test_all_steps_covered(self, parallel_trace):
        t, _ = parallel_trace
        steps = list(t.recorded_steps)
        regions = partition_dag_regions(steps)
        covered_ids = {s["step_id"] for reg in regions for s in reg.steps}
        all_ids = {s["step_id"] for s in steps}
        assert covered_ids == all_ids, "all steps must appear in exactly one region"

    def test_no_step_in_multiple_regions(self, parallel_trace):
        t, _ = parallel_trace
        regions = partition_dag_regions(list(t.recorded_steps))
        seen: set = set()
        for reg in regions:
            for s in reg.steps:
                sid = s["step_id"]
                assert sid not in seen, f"step {sid!r} appears in multiple regions"
                seen.add(sid)

    def test_single_step_trace(self, tmp_path):
        p = str(tmp_path / "single.sb")
        key = RecorderKey.fresh()
        with record(p, key=key) as rec:
            rec.tool_call("only", {"x": 1}, executor=_fake_tool)
        t = replay(p, hmac_key=key.hmac_key)
        regions = partition_dag_regions(list(t.recorded_steps))
        assert len(regions) == 1
        assert len(regions[0].steps) == 1


# ================================================================ correctness


class TestDistributedMatchesSequential:
    """compute_dirty_set_distributed must return identical results to compute_dirty_set."""

    def _compare(self, trace, substitutions, workers=4):
        seq = compute_dirty_set(trace, substitutions)
        dist = compute_dirty_set_distributed(trace, substitutions, workers=workers)
        assert dist.step_count == seq.step_count, "step_count mismatch"
        assert dist.dirty_count == seq.dirty_count, "dirty_count mismatch"
        assert dist.clean_count == seq.clean_count, "clean_count mismatch"
        assert dist.calls_saved == seq.calls_saved, "calls_saved mismatch"
        assert len(dist.entries) == len(seq.entries), "entries length mismatch"
        for d_entry, s_entry in zip(dist.entries, seq.entries):
            assert d_entry.step_id == s_entry.step_id, "step_id order mismatch"
            assert d_entry.dirty == s_entry.dirty, (
                f"dirty mismatch at {d_entry.step_id}: dist={d_entry.dirty} seq={s_entry.dirty}"
            )
            assert d_entry.cache_hit == s_entry.cache_hit, "cache_hit mismatch"
            assert d_entry.dirty_reason == s_entry.dirty_reason, "dirty_reason mismatch"
        return seq, dist

    def test_linear_no_substitution(self, linear_trace):
        t, _ = linear_trace
        self._compare(t, None)

    def test_linear_no_substitution_workers1(self, linear_trace):
        t, _ = linear_trace
        self._compare(t, None, workers=1)

    def test_linear_empty_substitutions(self, linear_trace):
        t, _ = linear_trace
        self._compare(t, [])

    def test_linear_prompt_substitution(self, linear_trace):
        t, _ = linear_trace
        # Substitute the first LLM call's prompt
        first_llm = next(s for s in t.recorded_steps if s["step_kind"] == "llm_call")
        sub = PromptSubstitution(
            at_step=first_llm["step_id"],
            new_messages=[{"role": "user", "content": "totally new prompt"}],
        )
        self._compare(t, [sub])

    def test_linear_tool_output_substitution(self, linear_trace):
        t, _ = linear_trace
        first_tool = next(s for s in t.recorded_steps if s["step_kind"] == "tool_call")
        sub = ToolOutputSubstitution(
            at_step=first_tool["step_id"],
            fake_response={"result": "injected", "args": {}},
        )
        self._compare(t, [sub])

    def test_parallel_no_substitution(self, parallel_trace):
        t, _ = parallel_trace
        self._compare(t, None)

    def test_parallel_no_substitution_workers1(self, parallel_trace):
        t, _ = parallel_trace
        self._compare(t, None, workers=1)

    def test_parallel_branch_tool_substitution(self, parallel_trace):
        t, _ = parallel_trace
        # Substitute the first branch tool call
        branch_tools = [
            s for s in t.recorded_steps
            if s["step_kind"] == "tool_call" and s.get("step_id", "").count(":") >= 2
            and s["step_id"] not in {
                s2["step_id"] for s2 in t.recorded_steps
                if s2["step_kind"] in ("parallel_branch_open", "parallel_branch_join", "setup")
            }
        ]
        # More simply: tool calls whose parent is the open step
        open_step_id = next(
            s["step_id"] for s in t.recorded_steps if s["step_kind"] == "parallel_branch_open"
        )
        branch_tools = [
            s for s in t.recorded_steps
            if s.get("parent_step_id") == open_step_id
        ]
        assert branch_tools, "should have branch tool calls"
        target = branch_tools[0]
        sub = ToolOutputSubstitution(
            at_step=target["step_id"],
            fake_response={"result": "injected_branch", "args": {}},
        )
        self._compare(t, [sub])

    def test_parallel_multiple_branch_substitutions(self, parallel_trace):
        t, _ = parallel_trace
        open_step_id = next(
            s["step_id"] for s in t.recorded_steps if s["step_kind"] == "parallel_branch_open"
        )
        branch_tools = [
            s for s in t.recorded_steps if s.get("parent_step_id") == open_step_id
        ]
        subs = [
            ToolOutputSubstitution(
                at_step=branch_tools[i]["step_id"],
                fake_response={"result": f"injected_{i}", "args": {}},
            )
            for i in range(min(3, len(branch_tools)))
        ]
        self._compare(t, subs)

    def test_parallel_workers1_matches_workers4(self, parallel_trace):
        t, _ = parallel_trace
        open_step_id = next(
            s["step_id"] for s in t.recorded_steps if s["step_kind"] == "parallel_branch_open"
        )
        branch_tools = [
            s for s in t.recorded_steps if s.get("parent_step_id") == open_step_id
        ]
        sub = ToolOutputSubstitution(
            at_step=branch_tools[0]["step_id"],
            fake_response={"result": "injected", "args": {}},
        )
        w1 = compute_dirty_set_distributed(t, [sub], workers=1)
        w4 = compute_dirty_set_distributed(t, [sub], workers=4)
        assert w1.dirty_count == w4.dirty_count
        for e1, e4 in zip(w1.entries, w4.entries):
            assert e1.dirty == e4.dirty
            assert e1.step_id == e4.step_id


# ================================================================ branch isolation


class TestBranchIsolation:
    """Verifies the B1/B2/B3 invariants across region boundaries."""

    def test_substituting_one_branch_does_not_dirty_siblings(self, parallel_trace):
        t, _ = parallel_trace
        open_step_id = next(
            s["step_id"] for s in t.recorded_steps if s["step_kind"] == "parallel_branch_open"
        )
        branch_tools = [
            s for s in t.recorded_steps if s.get("parent_step_id") == open_step_id
        ]
        assert len(branch_tools) >= 3, "need at least 3 branches"

        # Substitute the middle branch.
        target = branch_tools[len(branch_tools) // 2]
        sub = ToolOutputSubstitution(
            at_step=target["step_id"],
            fake_response={"result": "injected", "args": {}},
        )
        dist = compute_dirty_set_distributed(t, [sub])
        seq = compute_dirty_set(t, [sub])

        # The substituted step must be dirty.
        dist_entry = next(e for e in dist.entries if e.step_id == target["step_id"])
        assert dist_entry.dirty, "substituted branch step must be dirty"

        # Sibling branches must be clean (exclude the join step from siblings).
        sibling_ids = {
            s["step_id"]
            for s in branch_tools
            if s["step_id"] != target["step_id"]
            and s["step_kind"] != "parallel_branch_join"
        }
        for entry in dist.entries:
            if entry.step_id in sibling_ids:
                assert not entry.dirty, (
                    f"sibling branch {entry.step_id!r} must be a cache hit"
                )

        # Join must be dirty (B2).
        join_step_id = next(
            s["step_id"] for s in t.recorded_steps if s["step_kind"] == "parallel_branch_join"
        )
        join_entry = next(e for e in dist.entries if e.step_id == join_step_id)
        assert join_entry.dirty, "join must be dirty when any branch tail is dirty"

        # Results must match sequential.
        for d_entry, s_entry in zip(dist.entries, seq.entries):
            assert d_entry.dirty == s_entry.dirty, (
                f"step {d_entry.step_id}: dist={d_entry.dirty} seq={s_entry.dirty}"
            )

    def test_dirty_count_independent_of_fan_out_width(self, tmp_path):
        """Substituting one branch among N should dirty only O(1) steps, not O(N)."""
        for n_branches in (3, 6, 12):
            p = str(tmp_path / f"fan_{n_branches}.sb")
            key = RecorderKey.fresh()
            _record_parallel(p, key, n_branches=n_branches)
            t = replay(p, hmac_key=key.hmac_key)

            open_step_id = next(
                s["step_id"] for s in t.recorded_steps
                if s["step_kind"] == "parallel_branch_open"
            )
            branch_tools = [
                s for s in t.recorded_steps if s.get("parent_step_id") == open_step_id
            ]
            sub = ToolOutputSubstitution(
                at_step=branch_tools[0]["step_id"],
                fake_response={"result": "injected", "args": {}},
            )
            dist = compute_dirty_set_distributed(t, [sub])
            # Should dirty: 1 branch tool + join + suffix llm_call = 3 dirty steps.
            # Cache hits: setup + open + (n_branches - 1) sibling branches.
            assert dist.dirty_count == 3, (
                f"n_branches={n_branches}: expected 3 dirty steps, got {dist.dirty_count}"
            )


# ================================================================ edge cases


class TestEdgeCases:

    def test_no_steps(self):
        """compute_dirty_set_distributed on a trace with no steps."""
        # We can't easily build an empty Trace, so test partition directly.
        result = partition_dag_regions([])
        assert result == []

    def test_workers_zero_treated_as_one(self, linear_trace):
        t, _ = linear_trace
        dist = compute_dirty_set_distributed(t, None, workers=0)
        seq = compute_dirty_set(t, None)
        assert dist.dirty_count == seq.dirty_count

    def test_workers_large_value(self, parallel_trace):
        t, _ = parallel_trace
        dist = compute_dirty_set_distributed(t, None, workers=100)
        seq = compute_dirty_set(t, None)
        assert dist.dirty_count == seq.dirty_count

    def test_executor_kwarg_is_accepted(self, linear_trace):
        """executor kwarg must be accepted for API symmetry (not used)."""
        from stepback.replay import Executor

        t, _ = linear_trace
        dist = compute_dirty_set_distributed(
            t, None, executor=Executor(fallback_recorded=True)
        )
        assert dist.step_count == len(t.recorded_steps)

    def test_result_step_count_matches_trace(self, linear_trace):
        t, _ = linear_trace
        dist = compute_dirty_set_distributed(t, None)
        assert dist.step_count == len(t.recorded_steps)
        assert dist.dirty_count + dist.clean_count == dist.step_count

    def test_entries_in_topological_order(self, parallel_trace):
        t, _ = parallel_trace
        dist = compute_dirty_set_distributed(t, None)
        recorded_ids = [s["step_id"] for s in t.recorded_steps]
        result_ids = [e.step_id for e in dist.entries]
        assert result_ids == recorded_ids, "entries must be in recorded topological order"

    def test_p5_empty_substitutions_all_clean(self, parallel_trace):
        t, _ = parallel_trace
        dist = compute_dirty_set_distributed(t, [])
        assert dist.dirty_count == 0
        assert dist.calls_saved == dist.step_count

    def test_dag_region_dataclass_fields(self):
        reg = DagRegion(
            region_id="test",
            steps=[{"step_id": "s1"}],
            external_parent_ids=frozenset({"s0"}),
        )
        assert reg.region_id == "test"
        assert len(reg.steps) == 1
        assert "s0" in reg.external_parent_ids

    def test_region_summary_dataclass_fields(self):
        rs = RegionSummary(
            region_id="r0",
            entries=[],
            outputs_hash_by_id={},
            outputs_by_id={},
            dirty_by_id={},
            output_changed_by_id={},
        )
        assert rs.region_id == "r0"
        assert rs.entries == []


# ================================================================ public exports


class TestPublicExports:

    def test_symbols_exported_from_stepback(self):
        import stepback

        assert hasattr(stepback, "compute_dirty_set_distributed")
        assert hasattr(stepback, "DagRegion")
        assert hasattr(stepback, "RegionSummary")
        assert hasattr(stepback, "partition_dag_regions")

    def test_symbols_in_all(self):
        import stepback

        assert "compute_dirty_set_distributed" in stepback.__all__
        assert "DagRegion" in stepback.__all__
        assert "RegionSummary" in stepback.__all__
        assert "partition_dag_regions" in stepback.__all__

    def test_direct_import_from_distributed_dirty(self):
        from stepback.distributed_dirty import (  # noqa: F401
            DagRegion,
            RegionSummary,
            compute_dirty_set_distributed,
            partition_dag_regions,
        )
