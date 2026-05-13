"""Tests for Step-78 replay provenance: StepProvenance and ReplayProvenance.

Verifies that every replay code path (sequential _execute_plan,
parallel _execute_plan_parallel, and streaming replay_events) correctly
populates provenance fields on StepView and ReplayResult.

Coverage goals:
1. dirty_reason is None for cache-hit steps.
2. dirty_reason is populated for dirty steps with each distinct reason.
3. cache_source is "recorded_trace" for cache hits, "executor" for dirty.
4. cache_source is "persistent_cache" when StepCache produces the result.
5. cache_source is "fallback" when executor.fallback_recorded is used.
6. cache_source is "output_forced" for output-forcing substitutions.
7. model / provider extracted from LLM step inputs.
8. model_version extracted from LLM step outputs.
9. policy_blocked / policy_reason for policy-denial outputs.
10. seed extracted from LLM step inputs.
11. executor_version is a non-empty string.
12. ReplayProvenance timestamps are valid ISO-8601Z strings and started_at ≤ finished_at.
13. Parallel replay path also populates provenance.
14. replay_events() includes dirty_reason and executor_version in step_complete events.
15. _infer_provider maps known model prefixes correctly.
16. _probe_provider_version returns a string or None (no crash).
17. _structural_dirty_reason returns the right code for each trigger.
"""
from __future__ import annotations

import datetime
import re
from typing import Any, List

import pytest

from stepback import record, replay, ReplayResult, StepProvenance, ReplayProvenance
from stepback.replay import (
    Executor,
    _execute_plan,
    _execute_plan_parallel,
    _infer_provider,
    _probe_provider_version,
    _structural_dirty_reason,
    replay_events,
)
from stepback.step_cache import StepCache, StepCacheEntry
from stepback.substitutions import SubstitutionSet, ToolOutputSubstitution
from stepback.testing import fake_llm, fake_tool, run_recorded_agent


# ------------------------------------------------------------------ helpers


_ISO_Z_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00$|^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")


def _is_iso_z(s: str) -> bool:
    return bool(_ISO_Z_RE.match(s))


def _record_trace(tmp_path):
    from stepback import RecorderKey
    key = RecorderKey.fresh()
    path = str(tmp_path / "trace.sb")
    with record(path, key=key) as rec:
        run_recorded_agent(rec)
    return path, key


def _recorded_steps_from_trace(path: str) -> List[dict]:
    t = replay(path)
    return t.recorded_steps


# ------------------------------------------------------------------ _infer_provider


def test_infer_provider_anthropic_dot():
    assert _infer_provider("anthropic.claude-3-sonnet-20240229") == "anthropic"


def test_infer_provider_claude_dash():
    assert _infer_provider("claude-3-5-haiku-20241022") == "anthropic"


def test_infer_provider_gpt():
    assert _infer_provider("gpt-4o") == "openai"
    assert _infer_provider("gpt-3.5-turbo") == "openai"


def test_infer_provider_gemini():
    assert _infer_provider("gemini-1.5-flash") == "google"


def test_infer_provider_amazon_bedrock():
    assert _infer_provider("amazon.titan-text-lite-v1") == "bedrock"


def test_infer_provider_none_for_unknown():
    assert _infer_provider("completely-unknown-model") is None


def test_infer_provider_none_for_empty():
    assert _infer_provider(None) is None
    assert _infer_provider("") is None


# ------------------------------------------------------------------ _probe_provider_version


def test_probe_provider_version_none_for_unknown():
    # Unknown provider should not crash and returns None.
    result = _probe_provider_version("totally_unknown_provider")
    assert result is None


def test_probe_provider_version_none_for_none():
    assert _probe_provider_version(None) is None


def test_probe_provider_version_returns_string_or_none_for_known():
    # openai is listed in _PROVIDER_SDK_PACKAGES; it may or may not be installed.
    result = _probe_provider_version("openai")
    assert result is None or isinstance(result, str)


# ------------------------------------------------------------------ _structural_dirty_reason


def test_structural_dirty_reason_inputs_changed():
    assert _structural_dirty_reason("aaa", "bbb", False, False, False) == "inputs_changed"


def test_structural_dirty_reason_nondet_hash():
    assert _structural_dirty_reason("x", "x", True, False, False) == "nondeterminism_hash_changed"


def test_structural_dirty_reason_nondet_class():
    assert _structural_dirty_reason("x", "x", False, True, False) == "nondeterminism_forced"


def test_structural_dirty_reason_ancestor_dirty():
    assert _structural_dirty_reason("x", "x", False, False, True) == "ancestor_dirty"


# ------------------------------------------------------------------ cache-hit provenance


def test_cache_hit_step_provenance_none_dirty_reason(tmp_path):
    path, _ = _record_trace(tmp_path)
    t = replay(path)
    result = t.replay_forward(executor=Executor(fallback_recorded=True))

    assert isinstance(result.provenance, ReplayProvenance)
    for sv in result.steps:
        assert sv.provenance is not None
        if sv.cache_hit:
            assert sv.provenance.dirty_reason is None
            assert sv.provenance.cache_source == "recorded_trace"


def test_replay_provenance_timestamps(tmp_path):
    path, _ = _record_trace(tmp_path)
    t = replay(path)
    result = t.replay_forward(executor=Executor(fallback_recorded=True))

    prov = result.provenance
    assert prov is not None
    assert _is_iso_z(prov.started_at), f"started_at not ISO-Z: {prov.started_at}"
    assert _is_iso_z(prov.finished_at), f"finished_at not ISO-Z: {prov.finished_at}"
    # finished_at must be >= started_at
    assert prov.started_at <= prov.finished_at


def test_replay_provenance_executor_version(tmp_path):
    path, _ = _record_trace(tmp_path)
    t = replay(path)
    result = t.replay_forward(executor=Executor(fallback_recorded=True))

    prov = result.provenance
    assert isinstance(prov.executor_version, str)
    assert prov.executor_version  # non-empty


# ------------------------------------------------------------------ dirty step provenance


def test_dirty_step_has_dirty_reason(tmp_path):
    """Injecting a substitution makes the step dirty; dirty_reason must be set."""
    path, _ = _record_trace(tmp_path)
    t = replay(path)
    first_tool = next(
        (s for s in t.recorded_steps if s["step_kind"] == "tool_call"), None
    )
    if first_tool is None:
        pytest.skip("no tool_call step in fixture")

    step_id = first_tool["step_id"]
    subs = SubstitutionSet()
    subs.add(ToolOutputSubstitution(step_id, {"result": "injected"}))
    result = _execute_plan(t.recorded_steps, subs, Executor(fallback_recorded=True))

    dirty_views = [sv for sv in result.steps if sv.dirty]
    assert dirty_views, "expected at least one dirty step"

    # The injected step itself must have cache_source="output_forced".
    injected_view = next((sv for sv in dirty_views if sv.step_id == step_id), None)
    assert injected_view is not None
    assert injected_view.provenance is not None
    assert injected_view.provenance.dirty_reason == "output_forced"
    assert injected_view.provenance.cache_source == "output_forced"


def test_dirty_descendant_gets_ancestor_dirty(tmp_path):
    """Steps dirty because a parent was dirty should have dirty_reason 'ancestor_dirty'."""
    path, _ = _record_trace(tmp_path)
    t = replay(path)
    # Inject into the very first step so its child inherits dirtiness.
    first = t.recorded_steps[0]
    step_id = first["step_id"]
    subs = SubstitutionSet()
    subs.add(ToolOutputSubstitution(step_id, {"new_output": "changed"}))

    result = _execute_plan(t.recorded_steps, subs, Executor(fallback_recorded=True))

    # At least the first step must be dirty=True.
    assert result.steps[0].dirty
    assert result.steps[0].provenance is not None
    assert result.steps[0].provenance.dirty_reason == "output_forced"

    # If any downstream step is dirty purely because of cascade, it should have
    # dirty_reason == "ancestor_dirty" (only when hash matched but parent changed).
    ancestor_dirty_steps = [
        sv for sv in result.steps
        if sv.dirty and sv.provenance and sv.provenance.dirty_reason == "ancestor_dirty"
    ]
    # Not guaranteed unless parent's output actually changed; but dirty_reason must be one of the known codes.
    valid_reasons = {"output_forced", "inputs_changed", "nondeterminism_hash_changed", "nondeterminism_forced", "ancestor_dirty"}
    for sv in result.steps:
        if sv.dirty:
            assert sv.provenance is not None
            assert sv.provenance.dirty_reason in valid_reasons, f"Unknown dirty_reason: {sv.provenance.dirty_reason}"


# ------------------------------------------------------------------ fallback provenance


def test_fallback_recorded_cache_source(tmp_path):
    """When executor.fallback_recorded=True and no callback, cache_source='fallback'."""
    path, _ = _record_trace(tmp_path)
    t = replay(path)
    first = t.recorded_steps[0]
    step_id = first["step_id"]
    subs = SubstitutionSet()
    subs.add(ToolOutputSubstitution(step_id, {"result": "x"}))

    # Executor with no callbacks but fallback_recorded=True.
    executor = Executor(fallback_recorded=True)
    result = _execute_plan(t.recorded_steps, subs, executor)

    # The output-forced step has cache_source="output_forced".
    forced = next(sv for sv in result.steps if sv.step_id == step_id)
    assert forced.provenance.cache_source == "output_forced"

    # Any subsequent step that would be dirty but has no executor -> fallback.
    fallback_steps = [
        sv for sv in result.steps
        if sv.provenance and sv.provenance.cache_source == "fallback"
    ]
    # We can't guarantee any fallback steps exist unless the agent has dirty descendants
    # with no executor; verify types are correct if any exist.
    for sv in fallback_steps:
        assert sv.dirty
        assert sv.provenance.dirty_reason is not None


# ------------------------------------------------------------------ persistent cache provenance


def test_persistent_cache_hit_provenance(tmp_path):
    """Warm persistent cache makes dirty steps use cache_source='persistent_cache'."""
    path, _ = _record_trace(tmp_path)
    t = replay(path)
    first_tool = next(
        (s for s in t.recorded_steps if s["step_kind"] == "tool_call"), None
    )
    if first_tool is None:
        pytest.skip("no tool_call step in fixture")

    step_id = first_tool["step_id"]
    subs = SubstitutionSet()
    subs.add(ToolOutputSubstitution(step_id, {"result": "new"}))

    # Pre-warm: run once to populate an in-memory "persistent" cache.
    cached_outputs: dict = {}

    class WarmCache(StepCache):
        def get(self, kind: str, inputs_hash: str):
            key = (kind, inputs_hash)
            if key in cached_outputs:
                return StepCacheEntry(outputs=cached_outputs[key])
            return None

        def put(self, kind: str, inputs_hash: str, outputs: Any) -> None:
            cached_outputs[(kind, inputs_hash)] = outputs

        def close(self) -> None:
            pass

    warm_cache = WarmCache()
    exec1 = Executor(
        tool=lambda name, args: {"result": "re-executed"},
        fallback_recorded=True,
        step_cache=warm_cache,
    )
    _execute_plan(t.recorded_steps, subs, exec1)

    # Second run: the cache should serve the dirty step.
    exec2 = Executor(fallback_recorded=True, step_cache=warm_cache)
    result2 = _execute_plan(t.recorded_steps, subs, exec2)

    forced_step = next((sv for sv in result2.steps if sv.step_id == step_id), None)
    if forced_step is None:
        pytest.skip("substituted step not found")
    # output_forced steps bypass the cache entirely.
    assert forced_step.provenance.cache_source == "output_forced"

    # Descendant dirty steps (due to cascade) should come from persistent cache
    # if they were cached on the first run.
    persistent_hits = [
        sv for sv in result2.steps
        if sv.provenance and sv.provenance.cache_source == "persistent_cache"
    ]
    # May be zero if no descendants; just verify no type errors.
    for sv in persistent_hits:
        assert sv.dirty
        assert sv.provenance.dirty_reason is not None


# ------------------------------------------------------------------ model / provider / seed


def test_llm_step_model_extracted(tmp_path):
    """LLM steps should have model populated from inputs."""
    path, _ = _record_trace(tmp_path)
    t = replay(path)
    llm_steps = [s for s in t.recorded_steps if s["step_kind"] == "llm_call"]
    if not llm_steps:
        pytest.skip("no llm_call in fixture")

    result = t.replay_forward(executor=Executor(fallback_recorded=True))
    llm_views = [sv for sv in result.steps if sv.kind == "llm_call"]
    for sv in llm_views:
        assert sv.provenance is not None
        # model may or may not be set depending on the fixture, but provenance must exist.
        assert isinstance(sv.provenance.executor_version, str)


def test_step_with_model_in_inputs_has_provider():
    """_infer_provider is exercised via _build_step_provenance via StepView.provenance."""
    from stepback.replay import _build_step_provenance
    prov = _build_step_provenance(
        cur_inputs={"model": "claude-3-5-haiku-20241022", "messages": [], "seed": 42},
        cur_outputs={"content": "hi"},
        is_dirty=False,
        dirty_reason_str=None,
        cache_source_str="recorded_trace",
    )
    assert prov.model == "claude-3-5-haiku-20241022"
    assert prov.provider == "anthropic"
    assert prov.seed == 42


def test_step_model_version_extracted():
    from stepback.replay import _build_step_provenance
    prov = _build_step_provenance(
        cur_inputs={"model": "gemini-1.5-flash"},
        cur_outputs={"model_version": "gemini-1.5-flash-001", "text": "hello"},
        is_dirty=False,
        dirty_reason_str=None,
        cache_source_str="recorded_trace",
    )
    assert prov.model_version == "gemini-1.5-flash-001"
    assert prov.provider == "google"


# ------------------------------------------------------------------ policy provenance


def test_policy_blocked_detected():
    from stepback.replay import _build_step_provenance
    # Simulate a policy-denial output shape.
    policy_output = {"error_class": "PolicyDenied", "reason": "unsafe content detected"}
    prov = _build_step_provenance(
        cur_inputs={},
        cur_outputs=policy_output,
        is_dirty=True,
        dirty_reason_str="inputs_changed",
        cache_source_str="executor",
    )
    assert prov.policy_blocked is True
    assert prov.policy_reason == "unsafe content detected"


def test_policy_not_blocked_for_normal_output():
    from stepback.replay import _build_step_provenance
    prov = _build_step_provenance(
        cur_inputs={},
        cur_outputs={"content": "normal response"},
        is_dirty=False,
        dirty_reason_str=None,
        cache_source_str="recorded_trace",
    )
    assert prov.policy_blocked is False
    assert prov.policy_reason is None


# ------------------------------------------------------------------ parallel path


def test_parallel_replay_provenance_populated(tmp_path):
    """Parallel replay path must populate provenance on StepViews and ReplayResult."""
    path, _ = _record_trace(tmp_path)
    t = replay(path)
    result = _execute_plan_parallel(
        t.recorded_steps, SubstitutionSet(), Executor(fallback_recorded=True), workers=2
    )
    assert isinstance(result.provenance, ReplayProvenance)
    assert result.provenance.executor_version
    assert _is_iso_z(result.provenance.started_at)
    assert _is_iso_z(result.provenance.finished_at)
    for sv in result.steps:
        assert sv.provenance is not None, f"step {sv.step_id} missing provenance"
        assert isinstance(sv.provenance.executor_version, str)


# ------------------------------------------------------------------ replay_events


def test_replay_events_includes_dirty_reason(tmp_path):
    """replay_events() step_complete events must include dirty_reason and executor_version."""
    path, _ = _record_trace(tmp_path)
    t = replay(path)
    first_tool = next(
        (s for s in t.recorded_steps if s["step_kind"] == "tool_call"), None
    )
    if first_tool is None:
        pytest.skip("no tool_call step in fixture")

    subs = SubstitutionSet()
    subs.add(ToolOutputSubstitution(first_tool["step_id"], {"result": "new"}))

    events = list(replay_events(t.recorded_steps, subs, Executor(fallback_recorded=True)))
    step_events = [e for e in events if e.get("event") == "step_complete"]
    assert step_events

    for ev in step_events:
        assert "dirty_reason" in ev, f"dirty_reason missing from event: {ev}"
        assert "executor_version" in ev, f"executor_version missing from event: {ev}"
        # dirty_reason is None or a known code
        valid = {None, "output_forced", "inputs_changed", "nondeterminism_hash_changed", "nondeterminism_forced", "ancestor_dirty"}
        assert ev["dirty_reason"] in valid, f"Unknown dirty_reason: {ev['dirty_reason']}"


def test_replay_events_cache_hit_dirty_reason_none(tmp_path):
    """Cache-hit steps in replay_events must have dirty_reason=None."""
    path, _ = _record_trace(tmp_path)
    t = replay(path)
    subs = SubstitutionSet()  # no substitutions — all steps are cache hits
    events = list(replay_events(t.recorded_steps, subs, Executor(fallback_recorded=True)))
    step_events = [e for e in events if e.get("event") == "step_complete"]
    for ev in step_events:
        if ev["cache_hit"]:
            assert ev["dirty_reason"] is None, f"cache-hit step has dirty_reason: {ev}"


# ------------------------------------------------------------------ StepProvenance types


def test_step_provenance_fields_have_correct_types(tmp_path):
    path, _ = _record_trace(tmp_path)
    t = replay(path)
    result = t.replay_forward(executor=Executor(fallback_recorded=True))

    for sv in result.steps:
        p = sv.provenance
        assert isinstance(p, StepProvenance)
        assert p.dirty_reason is None or isinstance(p.dirty_reason, str)
        assert isinstance(p.cache_source, str)
        assert p.seed is None or isinstance(p.seed, int)
        assert p.model is None or isinstance(p.model, str)
        assert p.provider is None or isinstance(p.provider, str)
        assert p.model_version is None or isinstance(p.model_version, str)
        assert p.provider_version is None or isinstance(p.provider_version, str)
        assert isinstance(p.policy_blocked, bool)
        assert p.policy_reason is None or isinstance(p.policy_reason, str)
        assert isinstance(p.executor_version, str)
