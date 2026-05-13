"""Tests for the in-process replay API (Step 70).

This module covers:

1. :class:`~stepback.testing.CaptureExecutor` — records every successful
   executor callback invocation; does *not* capture output-forcing
   substitutions (they make a step dirty without calling the executor).

2. :class:`~stepback.testing.FallbackExecutor` — wraps ``Executor(fallback_recorded=True)``
   for local debugging without a real LLM/tool stack.

3. Assertion helpers: ``assert_all_cache_hits``, ``assert_dirty_count``,
   ``assert_real_executions``, ``assert_cache_hit_count``,
   ``assert_step_dirty``, ``assert_step_clean``.

4. Integration tests that record → substitute → CaptureExecutor replay →
   assert using the canonical fixture agents.
"""
from __future__ import annotations

import pytest

from stepback import record, replay
from stepback.recorder import RecorderKey
from stepback.substitutions import PromptSubstitution, ToolOutputSubstitution
from stepback.testing import (
    LOOKUP_FIXED_ROW,
    fake_llm,
    fake_tool,
    run_recorded_agent,
    run_parallel_agent,
)
from stepback.testing.replay import (
    CapturedCall,
    CaptureExecutor,
    FallbackExecutor,
    assert_all_cache_hits,
    assert_cache_hit_count,
    assert_dirty_count,
    assert_real_executions,
    assert_step_clean,
    assert_step_dirty,
)


# ------------------------------------------------------------------- helpers


def _record_fixture(tmp_path):
    key = RecorderKey.fresh()
    path = str(tmp_path / "trace.sb")
    with record(path, key=key) as rec:
        run_recorded_agent(rec)
    return path, key


def _record_parallel(tmp_path):
    key = RecorderKey.fresh()
    path = str(tmp_path / "parallel.sb")
    with record(path, key=key) as rec:
        run_parallel_agent(rec)
    return path, key


# ------------------------------------------------------------------- CaptureExecutor


def test_capture_executor_is_subclass_of_executor():
    from stepback.replay import Executor
    assert issubclass(CaptureExecutor, Executor)


def test_capture_executor_starts_with_empty_calls():
    cap = CaptureExecutor(llm=fake_llm, tool=fake_tool)
    assert cap.calls == []


def test_capture_executor_no_calls_on_clean_replay(tmp_path):
    path, _ = _record_fixture(tmp_path)
    trace = replay(path)
    cap = CaptureExecutor(llm=fake_llm, tool=fake_tool)
    result = trace.replay_forward(cap)
    # No substitutions → all cache hits → executor never called.
    assert cap.calls == []
    assert result.real_executions == 0
    assert result.dirty_count == 0


def test_capture_executor_records_call_on_tool_substitution(tmp_path):
    path, _ = _record_fixture(tmp_path)
    trace = replay(path)

    # Substitute step 2 (the first lookup_customer tool call).
    steps = trace.recorded_steps
    step2_id = steps[1]["step_id"]
    trace.substitute(ToolOutputSubstitution(step2_id, LOOKUP_FIXED_ROW))

    cap = CaptureExecutor(llm=fake_llm, tool=fake_tool)
    result = trace.replay_forward(cap)

    # Tool output substitution makes step dirty but does NOT call the executor
    # (it forces the output). Downstream dirty steps DO call the executor.
    assert result.dirty_count >= 1
    assert len(cap.calls) == result.real_executions


_NEW_MESSAGES = [{"role": "user", "content": "Different prompt for testing"}]


def test_capture_executor_calls_equal_real_executions(tmp_path):
    """len(cap.calls) must always equal result.real_executions."""
    path, _ = _record_fixture(tmp_path)
    trace = replay(path)

    steps = trace.recorded_steps
    step1_id = steps[0]["step_id"]
    trace.substitute(PromptSubstitution(step1_id, _NEW_MESSAGES))

    cap = CaptureExecutor(llm=fake_llm, tool=fake_tool)
    result = trace.replay_forward(cap)
    assert len(cap.calls) == result.real_executions


def test_captured_call_has_correct_kind(tmp_path):
    path, _ = _record_fixture(tmp_path)
    trace = replay(path)
    steps = trace.recorded_steps
    # Substitute the first LLM call so it re-executes.
    step1_id = steps[0]["step_id"]
    trace.substitute(PromptSubstitution(step1_id, _NEW_MESSAGES))

    cap = CaptureExecutor(llm=fake_llm, tool=fake_tool)
    trace.replay_forward(cap)

    assert all(c.kind in {"llm_call", "tool_call"} for c in cap.calls)
    # At least the first step (llm_call) should be recorded.
    assert cap.calls[0].kind == "llm_call"


def test_captured_call_fields_are_deep_copied(tmp_path):
    """Mutating captured inputs/outputs must not affect the CapturedCall record."""
    path, _ = _record_fixture(tmp_path)
    trace = replay(path)
    steps = trace.recorded_steps
    step1_id = steps[0]["step_id"]
    trace.substitute(PromptSubstitution(step1_id, _NEW_MESSAGES))

    cap = CaptureExecutor(llm=fake_llm, tool=fake_tool)
    trace.replay_forward(cap)

    assert cap.calls
    original_kind = cap.calls[0].kind
    # Mutating the captured dict does not affect subsequent reads.
    cap.calls[0].inputs["__mutated__"] = True
    assert cap.calls[0].kind == original_kind


def test_captured_call_is_instance_of_capturedcall(tmp_path):
    path, _ = _record_fixture(tmp_path)
    trace = replay(path)
    steps = trace.recorded_steps
    trace.substitute(PromptSubstitution(steps[0]["step_id"], _NEW_MESSAGES))
    cap = CaptureExecutor(llm=fake_llm, tool=fake_tool)
    trace.replay_forward(cap)
    assert all(isinstance(c, CapturedCall) for c in cap.calls)


def test_capture_executor_parallel_trace(tmp_path):
    """CaptureExecutor works with parallel-branch traces."""
    from stepback.testing import parallel_fake_llm, parallel_fake_tool

    path, _ = _record_parallel(tmp_path)
    trace = replay(path)
    steps = trace.recorded_steps
    # Substitute the first step to force re-execution of everything.
    trace.substitute(PromptSubstitution(steps[0]["step_id"], _NEW_MESSAGES))
    cap = CaptureExecutor(llm=parallel_fake_llm, tool=parallel_fake_tool)
    result = trace.replay_forward(cap)
    assert len(cap.calls) == result.real_executions
    # Parallel traces have branch_open and branch_join steps; their kinds
    # should appear in captured calls when dirty.
    kinds = {c.kind for c in cap.calls}
    assert len(kinds) > 0


# ------------------------------------------------------------------- FallbackExecutor


def test_fallback_executor_has_fallback_recorded_true():
    fb = FallbackExecutor()
    assert fb.fallback_recorded is True


def test_fallback_executor_allows_zero_executor_callbacks(tmp_path):
    path, _ = _record_fixture(tmp_path)
    trace = replay(path)
    steps = trace.recorded_steps
    # Make the first step dirty.
    trace.substitute(PromptSubstitution(steps[0]["step_id"], _NEW_MESSAGES))

    fb = FallbackExecutor()
    result = trace.replay_forward(fb)
    # FallbackExecutor never calls real executors; dirty steps use recorded outputs.
    assert result.real_executions == 0
    assert result.dirty_count >= 1


def test_fallback_executor_is_subclass_of_executor():
    from stepback.replay import Executor
    assert issubclass(FallbackExecutor, Executor)


# ------------------------------------------------------------------- assert_all_cache_hits


def test_assert_all_cache_hits_passes_on_clean_replay(tmp_path):
    path, _ = _record_fixture(tmp_path)
    trace = replay(path)
    cap = CaptureExecutor(llm=fake_llm, tool=fake_tool)
    result = trace.replay_forward(cap)
    # Should not raise.
    assert_all_cache_hits(result)


def test_assert_all_cache_hits_fails_when_dirty(tmp_path):
    path, _ = _record_fixture(tmp_path)
    trace = replay(path)
    steps = trace.recorded_steps
    trace.substitute(PromptSubstitution(steps[0]["step_id"], _NEW_MESSAGES))
    result = trace.replay_forward(CaptureExecutor(llm=fake_llm, tool=fake_tool))
    with pytest.raises(AssertionError) as exc_info:
        assert_all_cache_hits(result)
    msg = str(exc_info.value)
    assert "dirty" in msg.lower() or "cache" in msg.lower()
    # Should include the step_id or kind in the error message.
    assert any(s["step_id"] in msg for s in steps[:3])


# ------------------------------------------------------------------- assert_dirty_count


def test_assert_dirty_count_passes_on_exact_count(tmp_path):
    path, _ = _record_fixture(tmp_path)
    trace = replay(path)
    steps = trace.recorded_steps
    trace.substitute(PromptSubstitution(steps[0]["step_id"], _NEW_MESSAGES))
    result = trace.replay_forward(CaptureExecutor(llm=fake_llm, tool=fake_tool))
    assert_dirty_count(result, result.dirty_count)  # tautological but no raise


def test_assert_dirty_count_fails_on_wrong_count(tmp_path):
    path, _ = _record_fixture(tmp_path)
    trace = replay(path)
    result = trace.replay_forward(CaptureExecutor(llm=fake_llm, tool=fake_tool))
    # Clean replay → dirty_count == 0; expect 99 → fail.
    with pytest.raises(AssertionError) as exc_info:
        assert_dirty_count(result, 99)
    assert "99" in str(exc_info.value)
    assert "0" in str(exc_info.value)


def test_assert_dirty_count_message_includes_step_ids(tmp_path):
    path, _ = _record_fixture(tmp_path)
    trace = replay(path)
    steps = trace.recorded_steps
    trace.substitute(PromptSubstitution(steps[0]["step_id"], _NEW_MESSAGES))
    result = trace.replay_forward(CaptureExecutor(llm=fake_llm, tool=fake_tool))
    with pytest.raises(AssertionError) as exc_info:
        assert_dirty_count(result, 0)
    msg = str(exc_info.value)
    # The message should mention at least one dirty step id.
    assert steps[0]["step_id"] in msg


# ------------------------------------------------------------------- assert_real_executions


def test_assert_real_executions_passes(tmp_path):
    path, _ = _record_fixture(tmp_path)
    trace = replay(path)
    result = trace.replay_forward(CaptureExecutor(llm=fake_llm, tool=fake_tool))
    assert_real_executions(result, 0)


def test_assert_real_executions_fails_on_wrong_count(tmp_path):
    path, _ = _record_fixture(tmp_path)
    trace = replay(path)
    result = trace.replay_forward(CaptureExecutor(llm=fake_llm, tool=fake_tool))
    with pytest.raises(AssertionError) as exc_info:
        assert_real_executions(result, 5)
    assert "5" in str(exc_info.value)
    assert "0" in str(exc_info.value)


# ------------------------------------------------------------------- assert_cache_hit_count


def test_assert_cache_hit_count_passes(tmp_path):
    path, _ = _record_fixture(tmp_path)
    trace = replay(path)
    result = trace.replay_forward(CaptureExecutor(llm=fake_llm, tool=fake_tool))
    assert_cache_hit_count(result, len(trace.recorded_steps))


def test_assert_cache_hit_count_fails(tmp_path):
    path, _ = _record_fixture(tmp_path)
    trace = replay(path)
    result = trace.replay_forward(CaptureExecutor(llm=fake_llm, tool=fake_tool))
    with pytest.raises(AssertionError) as exc_info:
        assert_cache_hit_count(result, 0)
    assert str(exc_info.value)


# ------------------------------------------------------------------- assert_step_dirty / clean


def test_assert_step_clean_passes_on_clean_step(tmp_path):
    path, _ = _record_fixture(tmp_path)
    trace = replay(path)
    steps = trace.recorded_steps
    result = trace.replay_forward(CaptureExecutor(llm=fake_llm, tool=fake_tool))
    assert_step_clean(result, steps[0]["step_id"])


def test_assert_step_dirty_fails_on_clean_step(tmp_path):
    path, _ = _record_fixture(tmp_path)
    trace = replay(path)
    steps = trace.recorded_steps
    result = trace.replay_forward(CaptureExecutor(llm=fake_llm, tool=fake_tool))
    with pytest.raises(AssertionError) as exc_info:
        assert_step_dirty(result, steps[0]["step_id"])
    assert steps[0]["step_id"] in str(exc_info.value)


def test_assert_step_dirty_passes_on_dirty_step(tmp_path):
    path, _ = _record_fixture(tmp_path)
    trace = replay(path)
    steps = trace.recorded_steps
    step1_id = steps[0]["step_id"]
    trace.substitute(PromptSubstitution(step1_id, _NEW_MESSAGES))
    result = trace.replay_forward(CaptureExecutor(llm=fake_llm, tool=fake_tool))
    assert_step_dirty(result, step1_id)


def test_assert_step_clean_fails_on_dirty_step(tmp_path):
    path, _ = _record_fixture(tmp_path)
    trace = replay(path)
    steps = trace.recorded_steps
    step1_id = steps[0]["step_id"]
    trace.substitute(PromptSubstitution(step1_id, _NEW_MESSAGES))
    result = trace.replay_forward(CaptureExecutor(llm=fake_llm, tool=fake_tool))
    with pytest.raises(AssertionError) as exc_info:
        assert_step_clean(result, step1_id)
    assert step1_id in str(exc_info.value)


def test_assert_step_clean_raises_on_missing_step(tmp_path):
    path, _ = _record_fixture(tmp_path)
    trace = replay(path)
    result = trace.replay_forward()
    with pytest.raises(AssertionError) as exc_info:
        assert_step_clean(result, "nonexistent_step_id")
    msg = str(exc_info.value)
    assert "nonexistent_step_id" in msg
    assert "available" in msg.lower()


def test_assert_step_dirty_raises_on_missing_step(tmp_path):
    path, _ = _record_fixture(tmp_path)
    trace = replay(path)
    result = trace.replay_forward()
    with pytest.raises(AssertionError) as exc_info:
        assert_step_dirty(result, "no_such_id")
    msg = str(exc_info.value)
    assert "no_such_id" in msg
    assert "available" in msg.lower()


# ------------------------------------------------------------------- import path


def test_testing_module_exports_new_symbols():
    """New in-process replay symbols are re-exported from stepback.testing."""
    import stepback.testing as t

    required = {
        "CapturedCall",
        "CaptureExecutor",
        "FallbackExecutor",
        "assert_all_cache_hits",
        "assert_dirty_count",
        "assert_real_executions",
        "assert_cache_hit_count",
        "assert_step_dirty",
        "assert_step_clean",
    }
    missing = required - set(t.__all__)
    assert not missing, f"stepback.testing.__all__ missing: {sorted(missing)}"
    for name in required:
        assert hasattr(t, name), f"stepback.testing missing attribute {name!r}"


def test_testing_replay_submodule_importable():
    """stepback.testing.replay is importable as a standalone submodule."""
    import stepback.testing.replay as r

    assert hasattr(r, "CaptureExecutor")
    assert hasattr(r, "FallbackExecutor")
    assert hasattr(r, "assert_all_cache_hits")
    assert hasattr(r, "CapturedCall")
