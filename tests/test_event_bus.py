"""Tests for stepback.event_bus — Step 75.

Covers:
* EventKind values and string stability.
* ReplayEvent.to_dict() / from_dict() round-trip.
* NullEventBus: publish is a no-op; consume returns None.
* InMemoryEventBus: publish→consume round-trip; drain(); thread safety.
* KafkaEventBus: raises ImportError when kafka-python is absent.
* compute_replay_idempotency_key: deterministic; changes with trace or subs.
* InMemoryIdempotencyRegistry: claim, complete, fail, concurrent safety.
* execute_with_idempotency: emits events; respects idempotency; re-raises exceptions.

All tests are offline (no network calls).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from stepback.event_bus import (
    EventBus,
    EventKind,
    IdempotencyStatus,
    InMemoryEventBus,
    InMemoryIdempotencyRegistry,
    KafkaEventBus,
    NullEventBus,
    ReplayEvent,
    compute_replay_idempotency_key,
    execute_with_idempotency,
)
from stepback.substitutions import SubstitutionSet


# ---------------------------------------------------------------- helpers


def _sample_steps() -> List[dict]:
    return [
        {
            "step_id": "step:llm_call:1",
            "step_kind": "llm_call",
            "inputs_hash": "sha256:" + "a" * 64,
            "inputs": {"model": "gpt-4", "messages": []},
            "outputs": {"choices": [{"text": "hello"}]},
        },
        {
            "step_id": "step:tool_call:2",
            "step_kind": "tool_call",
            "inputs_hash": "sha256:" + "b" * 64,
            "inputs": {"name": "search", "arguments": {}},
            "outputs": {"result": "ok"},
        },
    ]


def _sample_subs(**kwargs: Any) -> SubstitutionSet:
    return SubstitutionSet()


# --------------------------------------------------------------- EventKind


def test_event_kind_values_are_stable() -> None:
    assert EventKind.REPLAY_JOB_SUBMITTED.value == "replay_job_submitted"
    assert EventKind.REPLAY_STEP_COMPLETE.value == "replay_step_complete"
    assert EventKind.REPLAY_JOB_COMPLETED.value == "replay_job_completed"
    assert EventKind.REPLAY_JOB_FAILED.value == "replay_job_failed"


def test_event_kind_is_str_subclass() -> None:
    # EventKind(str, Enum) so it's usable as a string directly.
    assert EventKind.REPLAY_JOB_SUBMITTED == "replay_job_submitted"


# -------------------------------------------------------------- ReplayEvent


def test_replay_event_to_dict_round_trip() -> None:
    evt = ReplayEvent(
        kind=EventKind.REPLAY_STEP_COMPLETE,
        job_id="job-123",
        step_id="step:llm_call:1",
        step_index=0,
        dirty=True,
        payload={"kind": "llm_call", "cache_hit": False},
        timestamp=1_700_000_000.0,
    )
    d = evt.to_dict()
    assert d["schema_version"] == "1"
    assert d["kind"] == "replay_step_complete"
    assert d["job_id"] == "job-123"
    assert d["step_id"] == "step:llm_call:1"
    assert d["step_index"] == 0
    assert d["dirty"] is True
    assert d["payload"] == {"kind": "llm_call", "cache_hit": False}
    assert d["timestamp"] == 1_700_000_000.0

    # from_dict round-trip
    restored = ReplayEvent.from_dict(d)
    assert restored.kind is EventKind.REPLAY_STEP_COMPLETE
    assert restored.job_id == "job-123"
    assert restored.step_id == "step:llm_call:1"
    assert restored.step_index == 0
    assert restored.dirty is True
    assert restored.timestamp == 1_700_000_000.0


def test_replay_event_defaults() -> None:
    evt = ReplayEvent(kind=EventKind.REPLAY_JOB_SUBMITTED, job_id="x")
    assert evt.step_id is None
    assert evt.step_index is None
    assert evt.dirty is None
    assert evt.payload == {}
    assert evt.timestamp > 0


def test_replay_event_from_dict_forward_compat() -> None:
    # Unknown schema_version is accepted (forward compat).
    d = {
        "schema_version": "99",
        "kind": "replay_job_completed",
        "job_id": "j",
        "step_id": None,
        "step_index": None,
        "dirty": None,
        "payload": {},
        "timestamp": 0.0,
    }
    evt = ReplayEvent.from_dict(d)
    assert evt.kind is EventKind.REPLAY_JOB_COMPLETED


# --------------------------------------------------------------- NullEventBus


def test_null_event_bus_publish_is_noop() -> None:
    bus = NullEventBus()
    # Should not raise.
    bus.publish(ReplayEvent(kind=EventKind.REPLAY_JOB_SUBMITTED, job_id="x"))


def test_null_event_bus_consume_returns_none() -> None:
    bus = NullEventBus()
    assert bus.consume(timeout=0) is None


def test_null_event_bus_close_is_noop() -> None:
    NullEventBus().close()


# --------------------------------------------------------- InMemoryEventBus


def test_inmemory_event_bus_round_trip() -> None:
    bus = InMemoryEventBus()
    evt = ReplayEvent(kind=EventKind.REPLAY_JOB_SUBMITTED, job_id="abc")
    bus.publish(evt)
    received = bus.consume(timeout=0.1)
    assert received is not None
    assert received.kind is EventKind.REPLAY_JOB_SUBMITTED
    assert received.job_id == "abc"


def test_inmemory_event_bus_consume_empty_returns_none() -> None:
    bus = InMemoryEventBus()
    assert bus.consume(timeout=0) is None


def test_inmemory_event_bus_drain_is_nondestructive() -> None:
    bus = InMemoryEventBus()
    for i in range(3):
        bus.publish(ReplayEvent(kind=EventKind.REPLAY_STEP_COMPLETE, job_id=f"j{i}",
                                step_id=f"s{i}", step_index=i, dirty=False))
    drained = bus.drain()
    assert len(drained) == 3
    # drain() does not remove from the queue — consume still works.
    assert bus.consume(timeout=0) is not None


def test_inmemory_event_bus_ordering() -> None:
    bus = InMemoryEventBus()
    kinds = [EventKind.REPLAY_JOB_SUBMITTED, EventKind.REPLAY_STEP_COMPLETE, EventKind.REPLAY_JOB_COMPLETED]
    for k in kinds:
        bus.publish(ReplayEvent(kind=k, job_id="j"))
    received = [bus.consume(timeout=0.1).kind for _ in range(3)]  # type: ignore[union-attr]
    assert received == kinds


def test_inmemory_event_bus_thread_safety() -> None:
    """Multiple publisher threads must not corrupt the events list."""
    bus = InMemoryEventBus()
    n_threads = 8
    n_events_per_thread = 50
    errors: List[Exception] = []

    def publish_many(thread_id: int) -> None:
        try:
            for i in range(n_events_per_thread):
                bus.publish(ReplayEvent(
                    kind=EventKind.REPLAY_STEP_COMPLETE,
                    job_id=f"j{thread_id}",
                    step_index=i,
                    dirty=i % 2 == 0,
                ))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=publish_many, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(bus.events) == n_threads * n_events_per_thread


def test_inmemory_event_bus_close_is_noop() -> None:
    InMemoryEventBus().close()


# ----------------------------------------------------------- KafkaEventBus


def test_kafka_event_bus_raises_import_error_when_unavailable() -> None:
    """If kafka-python is not installed, KafkaEventBus raises ImportError."""
    import sys
    kafka_mod = sys.modules.get("kafka")
    try:
        # Temporarily mask the module.
        sys.modules["kafka"] = None  # type: ignore[assignment]
        with pytest.raises(ImportError, match="kafka-python"):
            KafkaEventBus(topic="test", bootstrap_servers="localhost:9092")
    finally:
        if kafka_mod is not None:
            sys.modules["kafka"] = kafka_mod
        else:
            sys.modules.pop("kafka", None)


# ----------------------------------------------- compute_replay_idempotency_key


def test_idempotency_key_is_deterministic() -> None:
    steps = _sample_steps()
    subs = _sample_subs()
    key1 = compute_replay_idempotency_key(steps, subs)
    key2 = compute_replay_idempotency_key(steps, subs)
    assert key1 == key2
    assert len(key1) == 64  # SHA-256 hex


def test_idempotency_key_changes_with_different_trace() -> None:
    steps_a = _sample_steps()
    steps_b = _sample_steps()
    steps_b[0]["inputs_hash"] = "sha256:" + "c" * 64  # mutate hash
    subs = _sample_subs()
    assert compute_replay_idempotency_key(steps_a, subs) != compute_replay_idempotency_key(steps_b, subs)


def test_idempotency_key_changes_with_different_step_id() -> None:
    steps_a = _sample_steps()
    steps_b = _sample_steps()
    steps_b[0]["step_id"] = "step:llm_call:99"
    subs = _sample_subs()
    assert compute_replay_idempotency_key(steps_a, subs) != compute_replay_idempotency_key(steps_b, subs)


def test_idempotency_key_empty_trace() -> None:
    key = compute_replay_idempotency_key([], _sample_subs())
    assert len(key) == 64


# ------------------------------------------ InMemoryIdempotencyRegistry


def test_registry_claim_new_key_returns_claimed() -> None:
    reg = InMemoryIdempotencyRegistry()
    assert reg.claim("k1") is IdempotencyStatus.CLAIMED


def test_registry_claim_executing_key_returns_already_executing() -> None:
    reg = InMemoryIdempotencyRegistry()
    reg.claim("k1")  # → CLAIMED, now executing
    assert reg.claim("k1") is IdempotencyStatus.ALREADY_EXECUTING


def test_registry_complete_then_claim_returns_already_complete() -> None:
    reg = InMemoryIdempotencyRegistry()
    reg.claim("k1")
    reg.complete("k1")
    assert reg.claim("k1") is IdempotencyStatus.ALREADY_COMPLETE


def test_registry_fail_allows_retry() -> None:
    reg = InMemoryIdempotencyRegistry()
    reg.claim("k1")
    reg.fail("k1", "something went wrong")
    # After fail, key is removed → can be claimed again.
    assert reg.claim("k1") is IdempotencyStatus.CLAIMED


def test_registry_get_status() -> None:
    reg = InMemoryIdempotencyRegistry()
    assert reg.get_status("k1") is None
    reg.claim("k1")
    assert reg.get_status("k1") == "executing"
    reg.complete("k1")
    assert reg.get_status("k1") == "complete"


def test_registry_concurrent_claim() -> None:
    """Only one thread should win the claim for a given key."""
    reg = InMemoryIdempotencyRegistry()
    claimed = []
    lock = threading.Lock()

    def try_claim() -> None:
        status = reg.claim("shared-key")
        if status is IdempotencyStatus.CLAIMED:
            with lock:
                claimed.append(1)

    threads = [threading.Thread(target=try_claim) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(claimed) == 1, "exactly one thread must win the claim"


# ------------------------------------------ execute_with_idempotency


def _make_mock_plan(steps: List[dict], result_dirty: int = 1, result_hits: int = 1) -> Any:
    """Return a mock ReplayPlan whose .execute() returns a canned ReplayResult."""
    from stepback.replay import ReplayResult
    from stepback.substitutions import SubstitutionSet

    plan = MagicMock()
    plan._recorded_steps = steps
    plan._subs = SubstitutionSet()
    plan.execute.return_value = ReplayResult(
        steps=[],
        total_cost_usd=0.0,
        dirty_count=result_dirty,
        cache_hit_count=result_hits,
        real_executions=result_dirty,
    )
    return plan


def test_execute_with_idempotency_emits_job_events() -> None:
    steps = _sample_steps()
    plan = _make_mock_plan(steps)
    bus = InMemoryEventBus()
    reg = InMemoryIdempotencyRegistry()

    result = execute_with_idempotency(plan, registry=reg, event_bus=bus, job_id="test-job")

    assert result is not None
    kinds = [e.kind for e in bus.drain()]
    assert EventKind.REPLAY_JOB_SUBMITTED in kinds
    assert EventKind.REPLAY_JOB_COMPLETED in kinds


def test_execute_with_idempotency_skips_already_complete() -> None:
    steps = _sample_steps()
    plan = _make_mock_plan(steps)
    bus = InMemoryEventBus()
    reg = InMemoryIdempotencyRegistry()

    # First call — should execute.
    r1 = execute_with_idempotency(plan, registry=reg, event_bus=bus)
    assert r1 is not None
    assert plan.execute.call_count == 1

    # Second call — should skip (already complete).
    r2 = execute_with_idempotency(plan, registry=reg, event_bus=bus)
    assert r2 is None
    assert plan.execute.call_count == 1  # not called again


def test_execute_with_idempotency_skips_already_executing() -> None:
    steps = _sample_steps()
    plan = _make_mock_plan(steps)
    reg = InMemoryIdempotencyRegistry()
    key = compute_replay_idempotency_key(steps, plan._subs)
    reg.claim(key)  # pre-claim → simulates another worker

    result = execute_with_idempotency(plan, registry=reg)
    assert result is None
    assert plan.execute.call_count == 0


def test_execute_with_idempotency_marks_failed_on_exception() -> None:
    steps = _sample_steps()
    plan = _make_mock_plan(steps)
    plan.execute.side_effect = RuntimeError("executor failed")
    reg = InMemoryIdempotencyRegistry()
    bus = InMemoryEventBus()

    with pytest.raises(RuntimeError, match="executor failed"):
        execute_with_idempotency(plan, registry=reg, event_bus=bus)

    # Key should be removed (failed → retryable).
    key = compute_replay_idempotency_key(steps, plan._subs)
    assert reg.get_status(key) is None

    # REPLAY_JOB_FAILED event should have been published.
    kinds = [e.kind for e in bus.drain()]
    assert EventKind.REPLAY_JOB_FAILED in kinds


def test_execute_with_idempotency_job_id_in_events() -> None:
    steps = _sample_steps()
    plan = _make_mock_plan(steps)
    bus = InMemoryEventBus()
    reg = InMemoryIdempotencyRegistry()

    execute_with_idempotency(plan, registry=reg, event_bus=bus, job_id="explicit-id")

    for evt in bus.drain():
        assert evt.job_id == "explicit-id"


def test_execute_with_idempotency_null_bus_no_error() -> None:
    """Passing no event_bus defaults to NullEventBus — must not raise."""
    steps = _sample_steps()
    plan = _make_mock_plan(steps)
    reg = InMemoryIdempotencyRegistry()
    result = execute_with_idempotency(plan, registry=reg)
    assert result is not None


# ------------------------------------------- step events from ReplayPlan.execute


def test_replay_plan_execute_emits_step_events(tmp_path) -> None:
    """The modified _execute_plan emits REPLAY_STEP_COMPLETE per step."""
    from stepback.recorder import record, RecorderKey
    from stepback.replay import replay
    from stepback.substitutions import SubstitutionSet
    from stepback.testing import run_recorded_agent

    key = RecorderKey.fresh()
    path = str(tmp_path / "trace.sb")
    with record(path, key=key) as rec:
        run_recorded_agent(rec)

    trace = replay(path, hmac_key=key.hmac_key)
    plan = trace.plan_replay(SubstitutionSet())
    bus = InMemoryEventBus()

    plan.execute(_event_bus=bus, _job_id="trace-job")

    step_events = [e for e in bus.drain() if e.kind is EventKind.REPLAY_STEP_COMPLETE]
    # The trace has at least one step; at least one step event must be emitted.
    assert len(step_events) >= 1
    # All step events must have a step_id.
    for e in step_events:
        assert e.step_id is not None
        assert e.step_index is not None
        assert e.job_id == "trace-job"
