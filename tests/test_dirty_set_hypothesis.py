"""Hypothesis property tests for dirty-set replay correctness.

Implements step 29 of ``100_STEPS.md``:

    Add property tests that dirty-set replay equals full deterministic
    re-execution on generated DAG traces.

The dirty-set replay engine in :mod:`stepback.replay` is the *whole point*
of stepback: it skips re-execution for any step whose inputs hash matches
the recorded inputs hash and whose ancestors are clean. The correctness
contract is:

    Given a deterministic executor, the per-step outputs produced by
    dirty-set replay (cached + selectively re-executed) must equal the
    per-step outputs produced by full re-execution (every step driven
    through the executor) under the same substitutions.

Equivalently: the cache short-circuit must be *sound* — if a step is
declared a cache hit, then re-executing it would have produced the same
output that was recorded.

These tests generate random DAG-shaped traces (linear sequences plus
optional parallel fan-out / fan-in blocks) using Hypothesis, record them
through the real recorder against deterministic fakes, then compare the
outputs of:

    1. ``trace.replay_forward(executor)`` — the dirty-set engine.
    2. ``_force_full_replay(trace, subs, executor)`` — every step is
       forced dirty so the executor runs everywhere.

Per-step outputs, dirty propagation reachability, and the cache-hit
classification must all be consistent.
"""
from __future__ import annotations

import copy
import hashlib
import os
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from stepback import record, replay
from stepback.replay import Executor, ReplayResult, Trace
from stepback.substitutions import (
    ModelSubstitution,
    PromptSubstitution,
    RouterSubstitution,
    SubstitutionSet,
    ToolArgumentsSubstitution,
    ToolOutputSubstitution,
)


# ---------------------------------------------------------------------------
# Deterministic fakes
# ---------------------------------------------------------------------------
#
# These mimic ``stepback.testing.agent`` but are intentionally pure
# (no module-level state) so the same callables can be used for the
# initial recording, the dirty-set replay, and the forced full replay
# without any drift in observable behaviour.


def _digest(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def fake_llm(model: str, messages: List[dict]) -> dict:
    """Deterministic stand-in for an LLM API call.

    Output is a pure function of (model, messages); usage tokens are
    derived from message lengths so cost computations are also stable.
    """
    blob = f"{model}\n" + "\n".join(
        f"{m.get('role', '')}:{m.get('content', '')}" for m in messages
    )
    digest = _digest(blob)
    last = messages[-1].get("content", "") if messages else ""
    text = f"reply-{digest} echo:{last[:32]}"
    prompt_tokens = sum(len(m.get("content", "")) for m in messages)
    completion_tokens = len(text)
    return {
        "id": f"chatcmpl-{digest}",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def fake_tool(name: str, args: dict) -> Any:
    """Deterministic tool: returns a hash-derived payload of (name, args)."""
    blob = name + "\n" + repr(sorted(args.items()))
    digest = _digest(blob)
    return {"tool": name, "fingerprint": digest, "args_seen": dict(args)}


def fake_router(name: str, options: List[str]) -> str:
    if not options:
        return ""
    digest = int(_digest(name + "|" + "|".join(options)), 16)
    return options[digest % len(options)]


def make_executor() -> Executor:
    return Executor(llm=fake_llm, tool=fake_tool, router=fake_router)


# ---------------------------------------------------------------------------
# Trace generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepSpec:
    """One generated step in a synthetic DAG.

    ``kind`` is one of ``"llm"``, ``"tool"``, ``"router"``, ``"parallel"``.
    ``payload`` carries kind-specific data (a stable seed string).
    For ``"parallel"`` ``payload["branches"]`` is a list of inner step
    lists (each inner list is a sequence of non-parallel StepSpecs).
    """

    kind: str
    payload: dict


# Strategies for non-parallel "leaf" steps. Payloads stay small so that
# Hypothesis shrinks toward minimal failing cases quickly.
_text = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=1,
    max_size=8,
)
# Exclude reserved internal markers from generated keys so generated
# user payloads don't collide with the blob-reference convention
# (``{"$blob": "<digest>"}``) or other ``$``/``__``-prefixed sentinels.
_safe_key = _text.filter(lambda s: not s.startswith("$") and not s.startswith("__"))
_models = st.sampled_from(
    ["gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18", "claude-3-5-sonnet"]
)
_tool_names = st.sampled_from(["lookup", "calculate", "fetch", "search"])

_leaf = st.one_of(
    st.builds(
        StepSpec,
        kind=st.just("llm"),
        payload=st.builds(
            lambda model, prompt: {"model": model, "prompt": prompt},
            _models,
            _text,
        ),
    ),
    st.builds(
        StepSpec,
        kind=st.just("tool"),
        payload=st.builds(
            lambda name, k, v: {"name": name, "args": {k: v}},
            _tool_names,
            _safe_key,
            _text,
        ),
    ),
    st.builds(
        StepSpec,
        kind=st.just("router"),
        payload=st.builds(
            lambda name, opts: {"name": name, "options": list(opts)},
            _text,
            st.lists(_text, min_size=2, max_size=4, unique=True),
        ),
    ),
)

_parallel = st.builds(
    StepSpec,
    kind=st.just("parallel"),
    payload=st.builds(
        lambda branches: {"branches": branches, "name": "fanout"},
        st.lists(
            st.lists(_leaf, min_size=1, max_size=3),
            min_size=2,
            max_size=3,
        ),
    ),
)

# Top-level program: sequence of leaves with up to one parallel block.
_program = st.lists(st.one_of(_leaf, _parallel), min_size=1, max_size=6)


def _record_leaf(rec, spec: StepSpec) -> None:
    if spec.kind == "llm":
        rec.llm_call(
            spec.payload["model"],
            [{"role": "user", "content": spec.payload["prompt"]}],
            executor=fake_llm,
        )
    elif spec.kind == "tool":
        rec.tool_call(
            spec.payload["name"], dict(spec.payload["args"]), executor=fake_tool
        )
    elif spec.kind == "router":
        choice = fake_router(spec.payload["name"], spec.payload["options"])
        rec.router(spec.payload["name"], choice, list(spec.payload["options"]))
    else:  # pragma: no cover - defensive
        raise AssertionError(f"unknown leaf kind {spec.kind!r}")


def _record_program(rec, program: List[StepSpec]) -> None:
    for spec in program:
        if spec.kind == "parallel":
            branches = spec.payload["branches"]

            def _make_closure(leaves):
                def _branch(r):
                    for leaf in leaves:
                        _record_leaf(r, leaf)

                return _branch

            rec.parallel(
                spec.payload["name"],
                [_make_closure(b) for b in branches],
            )
        else:
            _record_leaf(rec, spec)


# ---------------------------------------------------------------------------
# Forced full re-execution
# ---------------------------------------------------------------------------


def _force_full_replay(
    trace: Trace, subs: SubstitutionSet, executor: Executor
) -> ReplayResult:
    """Run replay with every step forced dirty.

    The dirty-set engine treats ``current_inputs_hash != recorded_inputs_hash``
    as "dirty -> invoke executor". By corrupting every recorded
    ``inputs_hash`` we force every step through the executor, which —
    for a deterministic executor — must yield the same per-step outputs
    as the dirty-set replay path (subject to the same substitutions).
    """
    saved: List[Tuple[dict, str]] = []
    try:
        for s in trace.recorded_steps:
            saved.append((s, s["inputs_hash"]))
            s["inputs_hash"] = "FORCE_DIRTY::" + s["inputs_hash"]
        return trace.run_replay(subs, executor)
    finally:
        for s, h in saved:
            s["inputs_hash"] = h


# ---------------------------------------------------------------------------
# Substitution generation
# ---------------------------------------------------------------------------


def _build_substitutions(steps: List[dict], picks: List[Tuple[int, int]]) -> SubstitutionSet:
    """Build a SubstitutionSet from ``picks`` = list of (step_idx, choice).

    The choice index selects a kind-appropriate substitution variant.
    Steps that don't admit a substitution variant for the picked choice
    are skipped (no-op) — Hypothesis still gets coverage of the empty
    and partial substitution cases.
    """
    subs = SubstitutionSet()
    for idx, choice in picks:
        if idx >= len(steps):
            continue
        rec = steps[idx]
        sid = rec["step_id"]
        kind = rec["step_kind"]
        if kind == "llm_call":
            if choice % 3 == 0:
                subs.add(
                    ModelSubstitution(at_step=sid, new_model_id="gpt-4o-mini-2024-07-18")
                )
            elif choice % 3 == 1:
                new_msgs = [{"role": "user", "content": f"hyp-sub-{choice}"}]
                subs.add(PromptSubstitution(at_step=sid, new_messages=new_msgs))
            # else: no-op for this step
        elif kind == "tool_call":
            if choice % 3 == 0:
                subs.add(
                    ToolOutputSubstitution(
                        at_step=sid,
                        fake_response={"forced": True, "tag": choice},
                    )
                )
            elif choice % 3 == 1:
                subs.add(
                    ToolArgumentsSubstitution(
                        at_step=sid, new_arguments={"hyp_arg": str(choice)}
                    )
                )
        elif kind == "router":
            opts = rec["inputs"].get("options") or []
            if opts:
                subs.add(
                    RouterSubstitution(at_step=sid, choice=opts[choice % len(opts)])
                )
        # parallel_branch_open / _join: leave alone; no useful sub
    return subs


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


_HYP_SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


def _record_and_load(tmp_path, program: List[StepSpec]) -> Trace:
    path = os.path.join(str(tmp_path), "gen.sb")
    with record(path) as rec:
        _record_program(rec, program)
    return replay(path)


@_HYP_SETTINGS
@given(program=_program)
def test_dirty_set_equals_full_reexecution_no_subs(tmp_path_factory, program):
    """With *no* substitutions, dirty-set replay is all cache hits and
    must agree with forced full re-execution step-for-step.

    This is the tightest sanity check on the cache short-circuit: the
    deterministic executor would produce identical outputs, so cache
    hits must match the executor's would-be outputs.
    """
    tmp = tmp_path_factory.mktemp("nosubs")
    trace = _record_and_load(tmp, program)

    dirty = trace.replay_forward(make_executor())
    full = _force_full_replay(trace, SubstitutionSet(), make_executor())

    # All cache hits, zero real executions in the dirty-set replay path.
    assert dirty.dirty_count == 0
    assert dirty.cache_hit_count == len(dirty.steps)
    assert dirty.real_executions == 0

    # The forced full replay must have invoked the executor for every
    # step except the open/join frames (which are synthesised inside
    # the engine without a user callback).
    user_steps = [
        s for s in trace.recorded_steps
        if s["step_kind"] not in ("parallel_branch_open", "parallel_branch_join")
    ]
    assert full.real_executions >= len(user_steps)

    # Per-step output equality: this is the soundness property.
    assert len(dirty.steps) == len(full.steps)
    for ds, fs in zip(dirty.steps, full.steps):
        assert ds.step_id == fs.step_id
        assert ds.kind == fs.kind
        assert ds.outputs == fs.outputs, (
            f"cache-hit output diverges from re-execution at {ds.step_id} "
            f"({ds.kind}): cached={ds.outputs!r} vs reexec={fs.outputs!r}"
        )
        # Cost must agree to within float tolerance.
        assert abs(ds.cost_usd - fs.cost_usd) < 1e-6


@_HYP_SETTINGS
@given(program=_program, sub_seed=st.lists(st.tuples(st.integers(0, 12), st.integers(0, 5)), max_size=4))
def test_dirty_set_equals_full_reexecution_with_subs(tmp_path_factory, program, sub_seed):
    """With arbitrary substitutions, dirty-set and forced full replay
    must still produce identical per-step outputs.

    This is the headline correctness property of the dirty-set engine:
    selectively re-executing only the steps the engine *thinks* are
    dirty must produce exactly the result of re-executing every step.
    """
    tmp = tmp_path_factory.mktemp("withsubs")
    trace = _record_and_load(tmp, program)

    subs = _build_substitutions(trace.recorded_steps, sub_seed)

    dirty = trace.replay_forward.__self__  # dummy reference; keep linter quiet
    dirty = trace.run_replay(subs, make_executor())
    full = _force_full_replay(trace, subs, make_executor())

    assert len(dirty.steps) == len(full.steps)
    for ds, fs in zip(dirty.steps, full.steps):
        assert ds.step_id == fs.step_id
        assert ds.kind == fs.kind
        assert ds.outputs == fs.outputs, (
            f"dirty-set output diverges from full re-execution at "
            f"{ds.step_id} ({ds.kind}): "
            f"dirty={ds.outputs!r} vs full={fs.outputs!r}"
        )
        assert abs(ds.cost_usd - fs.cost_usd) < 1e-6


@_HYP_SETTINGS
@given(program=_program, sub_seed=st.lists(st.tuples(st.integers(0, 12), st.integers(0, 5)), max_size=4))
def test_dirty_propagation_is_downward_closed(tmp_path_factory, program, sub_seed):
    """Step 61 stale-cache semantics: a clean step may not have a DIRECT
    parent whose output actually changed (``output_changed=True``).

    When a dirty parent's output is unchanged (``output_changed=False``),
    downstream steps whose context rebinding still yields the same hash
    are correctly classified as cache hits — they are not stale.  The
    old stronger invariant ("no clean step has ANY dirty ancestor") was
    too conservative and is intentionally relaxed by Step 61.

    The correct minimal invariant that remains soundness-preserving is:
    for every clean step, none of its DIRECT parents have
    ``output_changed=True``.  If a parent was recomputed but produced
    the same output, the downstream step's inputs are provably unchanged,
    making its cache valid.
    """
    tmp = tmp_path_factory.mktemp("downward")
    trace = _record_and_load(tmp, program)

    subs = _build_substitutions(trace.recorded_steps, sub_seed)
    result = trace.run_replay(subs, make_executor())

    by_id = {s.step_id: s for s in result.steps}
    rec_by_id = {s["step_id"]: s for s in trace.recorded_steps}

    def parents_of(sid: str) -> List[str]:
        rec = rec_by_id[sid]
        ps: List[str] = []
        if rec.get("parent_step_id"):
            ps.append(rec["parent_step_id"])
        ps.extend(rec.get("parent_step_ids") or [])
        return [p for p in ps if p in rec_by_id]

    for sv in result.steps:
        if not sv.dirty:
            # No DIRECT parent may have output_changed=True.
            # If a parent's output hash changed, the context rebinding
            # mechanism would have updated this step's inputs hash, making
            # it dirty (or use_parent_dirty would have fired).  A clean
            # step whose direct parent has output_changed=True is a
            # soundness violation — the step's cache references a stale hash.
            for pid in parents_of(sv.step_id):
                pv = by_id.get(pid)
                if pv is None:
                    continue
                assert not pv.output_changed, (
                    f"clean step {sv.step_id} has direct parent {pid} with "
                    "output_changed=True — stale cache soundness violated"
                )


@_HYP_SETTINGS
@given(program=_program)
def test_idempotent_replay(tmp_path_factory, program):
    """Replaying the same trace twice must yield identical results.

    Replay is documented as non-mutating; a second call with the same
    inputs must therefore produce a bit-identical ReplayResult.
    """
    tmp = tmp_path_factory.mktemp("idem")
    trace = _record_and_load(tmp, program)

    r1 = trace.replay_forward(make_executor())
    r2 = trace.replay_forward(make_executor())

    assert r1.dirty_count == r2.dirty_count
    assert r1.cache_hit_count == r2.cache_hit_count
    assert len(r1.steps) == len(r2.steps)
    for a, b in zip(r1.steps, r2.steps):
        assert a.step_id == b.step_id
        assert a.outputs == b.outputs
        assert a.dirty == b.dirty
        assert a.cache_hit == b.cache_hit
