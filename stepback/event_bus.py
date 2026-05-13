"""Kafka-backed event bus for replay jobs and step-complete events (Step 75).

This module provides a pluggable event bus that workers can use to:

* **publish** structured replay lifecycle events (job submitted, step
  complete, job completed, job failed);
* **consume** those events from a topic so distributed workers can
  react to replay progress;
* enforce **idempotency** — if a worker dies after executing a replay
  job but before acknowledging the Kafka message, the redelivered message
  is detected by :class:`IdempotencyRegistry` and skipped rather than
  re-executed.

Architecture overview
---------------------
::

    producer                                consumer
    --------                                --------
    Trace.replay_forward(event_bus=bus)
        │
        ├─ bus.publish(REPLAY_JOB_SUBMITTED)
        │
        ├─ for each step:
        │      bus.publish(REPLAY_STEP_COMPLETE)
        │
        └─ bus.publish(REPLAY_JOB_COMPLETED | REPLAY_JOB_FAILED)

    # On the consuming side:
    while True:
        evt = bus.consume(timeout=1.0)
        if evt is None:
            continue
        handle(evt)

Kafka is optional
-----------------
:class:`KafkaEventBus` wraps ``kafka-python`` with a lazy import.  If the
package is not installed, constructing a ``KafkaEventBus`` raises a clear
:exc:`ImportError`.  The in-process :class:`InMemoryEventBus` works with zero
extra dependencies and is the recommended choice for unit tests and local
development.

Idempotency contract
--------------------
An :class:`IdempotencyRegistry` assigns one of three statuses to each
*idempotency key* (a deterministic hex digest of the trace content hash and
canonicalized substitution set):

``CLAIMED``
    The key was freshly claimed; the caller should proceed with execution.
``ALREADY_EXECUTING``
    Another worker is currently executing this job; the caller should skip
    (acknowledge the message without re-executing).
``ALREADY_COMPLETE``
    The job finished previously; the caller should skip.

:func:`compute_replay_idempotency_key` computes the deterministic key from a
trace's recorded steps list and a :class:`~stepback.substitutions.SubstitutionSet`.

Public surface (re-exported from :mod:`stepback`):

* :class:`EventKind`
* :class:`ReplayEvent`
* :class:`EventBus`
* :class:`NullEventBus`
* :class:`InMemoryEventBus`
* :class:`KafkaEventBus`
* :class:`IdempotencyStatus`
* :class:`IdempotencyRegistry`
* :class:`InMemoryIdempotencyRegistry`
* :func:`compute_replay_idempotency_key`
"""
from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .canonical import canonical_json, sha256_hex


# ------------------------------------------------------------------ EventKind


class EventKind(str, Enum):
    """Lifecycle events emitted during a replay job.

    Values are stable strings suitable for use as Kafka message keys or log
    labels.
    """

    REPLAY_JOB_SUBMITTED = "replay_job_submitted"
    """Emitted when :meth:`ReplayPlan.execute` begins."""

    REPLAY_STEP_COMPLETE = "replay_step_complete"
    """Emitted after each step is evaluated (cache hit or re-executed)."""

    REPLAY_JOB_COMPLETED = "replay_job_completed"
    """Emitted when all steps finish successfully."""

    REPLAY_JOB_FAILED = "replay_job_failed"
    """Emitted when the job raises an unhandled exception."""


# ----------------------------------------------------------------- ReplayEvent


@dataclass
class ReplayEvent:
    """A structured event emitted during a replay job.

    All fields are JSON-serializable so events can be forwarded to Kafka,
    logged to disk, or consumed by downstream analytics.

    Attributes
    ----------
    kind:
        The event kind.  Use :class:`EventKind` values.
    job_id:
        Opaque string identifying the replay job.  Callers may supply a
        meaningful value (e.g. trace id + substitution hash); the planner
        sets this to the :func:`compute_replay_idempotency_key` by default.
    step_id:
        The step id for ``REPLAY_STEP_COMPLETE`` events; ``None`` otherwise.
    step_index:
        Zero-based position of the step in the recorded trace (for ordering
        parallel-completion events).  ``None`` for non-step events.
    dirty:
        Whether the step was dirty (``True``) or a cache hit (``False``).
        ``None`` for non-step events.
    payload:
        Additional key/value metadata.  Reserved keys:
        ``"error"`` (str) for ``REPLAY_JOB_FAILED`` events.
    timestamp:
        Unix timestamp (float) when the event was created.
    """

    kind: EventKind
    job_id: str
    step_id: Optional[str] = None
    step_index: Optional[int] = None
    dirty: Optional[bool] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        The ``"schema_version"`` key is always ``"1"`` and can be used by
        consumers to detect incompatible schema changes.
        """
        return {
            "schema_version": "1",
            "kind": self.kind.value,
            "job_id": self.job_id,
            "step_id": self.step_id,
            "step_index": self.step_index,
            "dirty": self.dirty,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReplayEvent":
        """Deserialize from a dictionary produced by :meth:`to_dict`.

        Unknown ``schema_version`` values are accepted (forward compatibility).
        """
        return cls(
            kind=EventKind(d["kind"]),
            job_id=d["job_id"],
            step_id=d.get("step_id"),
            step_index=d.get("step_index"),
            dirty=d.get("dirty"),
            payload=dict(d.get("payload") or {}),
            timestamp=float(d.get("timestamp", 0.0)),
        )


# ------------------------------------------------------------------- EventBus


class EventBus(ABC):
    """Abstract base class for an event bus.

    Subclasses implement the publish/consume cycle for a specific backend
    (in-memory, Kafka, etc.).  The bus is intentionally **write-biased** for
    replay workers: a ``ReplayPlan`` publishes events; downstream analytics or
    orchestration layers consume them through whatever backend-specific API
    they prefer.

    Thread safety
    -------------
    Implementations must be safe for concurrent calls to :meth:`publish` from
    multiple threads (e.g. parallel branch workers).
    """

    @abstractmethod
    def publish(self, event: ReplayEvent) -> None:
        """Publish *event* to the bus.

        Must not raise for normal operational events.  Backend connectivity
        errors may raise :exc:`RuntimeError`.
        """

    @abstractmethod
    def consume(self, timeout: float = 1.0) -> Optional[ReplayEvent]:
        """Block for up to *timeout* seconds and return the next event.

        Returns ``None`` if no event arrives within the timeout.  Callers
        should loop::

            while True:
                evt = bus.consume(timeout=1.0)
                if evt is None:
                    break  # or continue polling
                handle(evt)
        """

    @abstractmethod
    def close(self) -> None:
        """Release any backend resources (connections, threads, etc.)."""


# ----------------------------------------------------------------- NullEventBus


class NullEventBus(EventBus):
    """An event bus that silently discards all events.

    Use as the default when no external event bus is configured, preserving
    backward compatibility.  :meth:`consume` always returns ``None``.
    """

    def publish(self, event: ReplayEvent) -> None:  # noqa: D401
        """Discard *event* without any side effects."""

    def consume(self, timeout: float = 1.0) -> Optional[ReplayEvent]:
        """Always return ``None``."""
        return None

    def close(self) -> None:
        """No-op."""


# --------------------------------------------------------------- InMemoryEventBus


class InMemoryEventBus(EventBus):
    """An in-process event bus backed by :class:`queue.Queue`.

    All published events are stored in :attr:`events` (a list, for
    inspection in tests) **and** enqueued in an internal ``Queue`` for
    :meth:`consume`.

    Thread safety
    -------------
    :meth:`publish` acquires a lock before appending to :attr:`events` so
    multiple branch workers can publish concurrently without races.
    :meth:`consume` delegates to :class:`queue.Queue` which is already
    thread-safe.

    Example::

        bus = InMemoryEventBus()
        bus.publish(ReplayEvent(kind=EventKind.REPLAY_JOB_SUBMITTED, job_id="x"))
        evt = bus.consume(timeout=0)
        assert evt is not None and evt.kind == EventKind.REPLAY_JOB_SUBMITTED
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[ReplayEvent] = queue.Queue()
        self._lock = threading.Lock()
        self.events: List[ReplayEvent] = []

    def publish(self, event: ReplayEvent) -> None:
        """Append *event* to :attr:`events` and enqueue it."""
        with self._lock:
            self.events.append(event)
        self._queue.put(event)

    def consume(self, timeout: float = 1.0) -> Optional[ReplayEvent]:
        """Return the next event or ``None`` if the timeout expires."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        """No-op for the in-memory bus."""

    def drain(self) -> List[ReplayEvent]:
        """Return all events published so far.

        Unlike :meth:`consume` this is non-destructive and does not dequeue;
        it reads :attr:`events` directly.
        """
        with self._lock:
            return list(self.events)


# ---------------------------------------------------------------- KafkaEventBus


class KafkaEventBus(EventBus):
    """A Kafka-backed event bus using ``kafka-python``.

    Requires the optional ``kafka`` extra::

        pip install stepback[kafka]

    Construction raises :exc:`ImportError` if ``kafka-python`` is not
    installed.

    Parameters
    ----------
    topic:
        The Kafka topic to publish events to and consume events from.
    bootstrap_servers:
        Comma-separated ``host:port`` pairs for the Kafka cluster.
    producer_config:
        Extra keyword arguments forwarded to :class:`kafka.KafkaProducer`.
    consumer_config:
        Extra keyword arguments forwarded to :class:`kafka.KafkaConsumer`.
        Defaults include ``auto_offset_reset="latest"`` and
        ``enable_auto_commit=True``.

    Idempotency and at-least-once delivery
    ---------------------------------------
    ``KafkaEventBus`` is a **publish-only helper** for replay workers.
    Consumption semantics (offset commit, consumer groups) are controlled
    by the caller through *consumer_config*.  For exactly-once worker
    semantics, pair this bus with :class:`InMemoryIdempotencyRegistry` (or a
    durable registry) and wrap replay execution with
    :func:`~stepback.event_bus.execute_with_idempotency`.
    """

    def __init__(
        self,
        topic: str,
        bootstrap_servers: str = "localhost:9092",
        *,
        producer_config: Optional[Dict[str, Any]] = None,
        consumer_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            import kafka  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "KafkaEventBus requires the 'kafka-python' package. "
                "Install it with: pip install stepback[kafka]"
            ) from exc
        import json as _json

        self._topic = topic
        self._kafka = kafka

        producer_kw: Dict[str, Any] = {
            "bootstrap_servers": bootstrap_servers,
            "value_serializer": lambda v: _json.dumps(v).encode("utf-8"),
        }
        if producer_config:
            producer_kw.update(producer_config)
        self._producer = kafka.KafkaProducer(**producer_kw)

        consumer_kw: Dict[str, Any] = {
            "bootstrap_servers": bootstrap_servers,
            "auto_offset_reset": "latest",
            "enable_auto_commit": True,
            "value_deserializer": lambda b: _json.loads(b.decode("utf-8")),
        }
        if consumer_config:
            consumer_kw.update(consumer_config)
        self._consumer = kafka.KafkaConsumer(topic, **consumer_kw)

    def publish(self, event: ReplayEvent) -> None:
        """Serialize *event* to JSON and send it to the configured Kafka topic."""
        self._producer.send(self._topic, value=event.to_dict())

    def consume(self, timeout: float = 1.0) -> Optional[ReplayEvent]:
        """Poll the Kafka consumer for up to *timeout* seconds.

        Returns the next :class:`ReplayEvent` or ``None`` on timeout.
        """
        timeout_ms = int(timeout * 1000)
        records = self._consumer.poll(timeout_ms=timeout_ms)
        for _, msgs in records.items():
            if msgs:
                return ReplayEvent.from_dict(msgs[0].value)
        return None

    def close(self) -> None:
        """Flush the producer and close both producer and consumer."""
        self._producer.flush()
        self._producer.close()
        self._consumer.close()


# ------------------------------------------------------------ IdempotencyStatus


class IdempotencyStatus(str, Enum):
    """Result of :meth:`IdempotencyRegistry.claim`.

    ``CLAIMED``
        The key was freshly claimed; the caller should proceed.
    ``ALREADY_EXECUTING``
        Another worker is executing this key; the caller should skip.
    ``ALREADY_COMPLETE``
        The key completed previously; the caller should skip.
    """

    CLAIMED = "claimed"
    ALREADY_EXECUTING = "already_executing"
    ALREADY_COMPLETE = "already_complete"


# ------------------------------------------------------------ IdempotencyRegistry


class IdempotencyRegistry(ABC):
    """Abstract registry for replay job idempotency.

    Workers call :meth:`claim` before starting execution.  The registry
    atomically transitions ``key → executing`` and returns
    :attr:`~IdempotencyStatus.CLAIMED`.  If the key is already executing or
    complete, the registry returns the appropriate status so the worker can
    skip without re-executing.

    After successful execution :meth:`complete` is called.  On failure
    :meth:`fail` is called so the key can be retried.

    Implementations must be **thread-safe**.  Distributed deployments should
    provide a durable backend (Redis, PostgreSQL, etc.) so the registry
    survives worker crashes.
    """

    @abstractmethod
    def claim(self, key: str) -> IdempotencyStatus:
        """Atomically claim *key* for execution.

        If *key* is new → mark it ``executing``, return
        :attr:`~IdempotencyStatus.CLAIMED`.
        If *key* is already ``executing`` → return
        :attr:`~IdempotencyStatus.ALREADY_EXECUTING`.
        If *key* is already ``complete`` → return
        :attr:`~IdempotencyStatus.ALREADY_COMPLETE`.
        """

    @abstractmethod
    def complete(self, key: str) -> None:
        """Mark *key* as successfully completed."""

    @abstractmethod
    def fail(self, key: str, error: str) -> None:
        """Mark *key* as failed so it can be retried by the next worker."""

    @abstractmethod
    def get_status(self, key: str) -> Optional[str]:
        """Return the current raw status string for *key*, or ``None`` if unknown."""


# ------------------------------------------------------ InMemoryIdempotencyRegistry


class InMemoryIdempotencyRegistry(IdempotencyRegistry):
    """Thread-safe in-memory idempotency registry.

    All state is lost when the process exits.  Suitable for unit tests and
    single-process replay workers.  For multi-process or distributed workers
    use a durable registry backed by Redis, PostgreSQL, etc.

    Example::

        registry = InMemoryIdempotencyRegistry()
        key = compute_replay_idempotency_key(steps, subs)

        status = registry.claim(key)
        if status == IdempotencyStatus.CLAIMED:
            try:
                result = plan.execute(...)
                registry.complete(key)
            except Exception as exc:
                registry.fail(key, str(exc))
                raise
        elif status == IdempotencyStatus.ALREADY_COMPLETE:
            pass  # skip
        else:
            pass  # already executing; skip
    """

    _STATUS_EXECUTING = "executing"
    _STATUS_COMPLETE = "complete"
    # Failed jobs are removed from the registry so they can be retried.

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: Dict[str, str] = {}  # key → status string

    def claim(self, key: str) -> IdempotencyStatus:
        """Atomically claim *key*."""
        with self._lock:
            status = self._store.get(key)
            if status is None:
                self._store[key] = self._STATUS_EXECUTING
                return IdempotencyStatus.CLAIMED
            if status == self._STATUS_COMPLETE:
                return IdempotencyStatus.ALREADY_COMPLETE
            return IdempotencyStatus.ALREADY_EXECUTING

    def complete(self, key: str) -> None:
        """Mark *key* complete."""
        with self._lock:
            self._store[key] = self._STATUS_COMPLETE

    def fail(self, key: str, error: str) -> None:
        """Remove *key* from the registry so the next worker can retry."""
        with self._lock:
            self._store.pop(key, None)

    def get_status(self, key: str) -> Optional[str]:
        """Return the raw status for *key*, or ``None`` if unknown."""
        with self._lock:
            return self._store.get(key)


# ------------------------------------------------- compute_replay_idempotency_key


def compute_replay_idempotency_key(
    recorded_steps: "list[dict]",
    subs: "Any",
) -> str:
    """Compute a deterministic idempotency key for a replay job.

    The key is the SHA-256 hex digest of the canonical JSON of:

    * the list of ``(step_id, inputs_hash)`` pairs from *recorded_steps*
      (captures trace identity and content);
    * the canonical JSON representation of the substitution set (captures the
      exact intervention being replayed).

    Changing either the trace or the substitution set produces a different key,
    guaranteeing that distinct jobs are not incorrectly deduped.

    Parameters
    ----------
    recorded_steps:
        The raw list of step dicts from a :class:`~stepback.replay.Trace`.
    subs:
        A :class:`~stepback.substitutions.SubstitutionSet` (or any object with
        a ``to_list()`` / ``_subs`` attribute that is JSON-serializable).

    Returns
    -------
    str
        A 64-character hex digest.
    """
    step_fingerprints = [
        {"step_id": s["step_id"], "inputs_hash": s["inputs_hash"]}
        for s in recorded_steps
    ]
    # Represent subs as a sorted list of their canonical JSON so the key is
    # stable regardless of insertion order.
    if hasattr(subs, "_subs"):
        raw_subs = subs._subs  # list[Substitution]
        sub_reprs = sorted(repr(s) for s in raw_subs)
    else:
        sub_reprs = []
    payload = {"steps": step_fingerprints, "subs": sub_reprs}
    full = sha256_hex(canonical_json(payload))
    # Strip the "sha256:" prefix to return a clean 64-character hex digest.
    return full.removeprefix("sha256:")


# ---------------------------------------------------- execute_with_idempotency


def execute_with_idempotency(
    plan: "Any",
    *,
    registry: IdempotencyRegistry,
    job_id: Optional[str] = None,
    event_bus: Optional[EventBus] = None,
    executor: Optional["Any"] = None,
    workers: Optional[int] = None,
) -> "Any":
    """Execute *plan* with idempotency and event-bus integration.

    Wraps :meth:`~stepback.replay.ReplayPlan.execute` with:

    1. An idempotency check via *registry* so duplicate Kafka redeliveries are
       skipped without re-executing dirty steps.
    2. Event publication for job lifecycle events to *event_bus* (or the
       :class:`NullEventBus` if unset).

    Parameters
    ----------
    plan:
        A :class:`~stepback.replay.ReplayPlan` returned by
        :meth:`~stepback.replay.Trace.plan_replay`.
    registry:
        The :class:`IdempotencyRegistry` to use for deduplication.
    job_id:
        If provided, used as the ``job_id`` in emitted events; otherwise
        derived from the plan's idempotency key.
    event_bus:
        The :class:`EventBus` to publish lifecycle events to.  Defaults to
        :class:`NullEventBus` (no-op).
    executor:
        Passed through to :meth:`~stepback.replay.ReplayPlan.execute`.
    workers:
        Passed through to :meth:`~stepback.replay.ReplayPlan.execute`.

    Returns
    -------
    ReplayResult or None
        The replay result, or ``None`` if the job was skipped due to
        idempotency (already executing or already complete).

    Raises
    ------
    Exception
        Any exception from :meth:`~stepback.replay.ReplayPlan.execute` is
        re-raised after marking the registry key as failed and publishing a
        ``REPLAY_JOB_FAILED`` event.
    """
    bus = event_bus or NullEventBus()
    key = compute_replay_idempotency_key(
        plan._recorded_steps, plan._subs
    )
    eff_job_id = job_id or key[:16]  # short prefix for human readability

    status = registry.claim(key)
    if status == IdempotencyStatus.ALREADY_COMPLETE:
        return None
    if status == IdempotencyStatus.ALREADY_EXECUTING:
        return None

    # CLAIMED — proceed with execution.
    bus.publish(ReplayEvent(
        kind=EventKind.REPLAY_JOB_SUBMITTED,
        job_id=eff_job_id,
        payload={"idempotency_key": key},
    ))
    try:
        result = plan.execute(
            executor,
            workers=workers,
            _event_bus=bus,
            _job_id=eff_job_id,
        )
        registry.complete(key)
        bus.publish(ReplayEvent(
            kind=EventKind.REPLAY_JOB_COMPLETED,
            job_id=eff_job_id,
            payload={
                "dirty_count": result.dirty_count,
                "cache_hit_count": result.cache_hit_count,
                "real_executions": result.real_executions,
                "total_cost_usd": result.total_cost_usd,
            },
        ))
        return result
    except Exception as exc:
        registry.fail(key, str(exc))
        bus.publish(ReplayEvent(
            kind=EventKind.REPLAY_JOB_FAILED,
            job_id=eff_job_id,
            payload={"error": str(exc)},
        ))
        raise
