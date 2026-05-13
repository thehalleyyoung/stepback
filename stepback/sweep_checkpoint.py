"""Checkpointed replay sweep for fault-tolerant million-point sweeps (Step 76).

:func:`resume_sweep` wraps :func:`~stepback.sweep.sweep_traces` with a
persistent :class:`DiskSweepCheckpoint` so that:

* Completed traces are recorded to disk immediately after processing and
  are **never re-processed** on restart.
* Failed traces are marked and skipped on restart by default (or retried
  when ``retry_failed=True``).
* Multiple workers partition the trace corpus via :class:`WorkerLease`
  claims so each trace is processed by exactly one worker at a time.
  If a worker dies, its leases expire and the traces can be re-claimed.

Architecture
------------
::

    Worker 1                           Worker 2
    --------                           --------
    resume_sweep(paths, ...)           resume_sweep(paths, ...)
      checkpoint.initialise(paths)       checkpoint.initialise(paths)
      for path in pending_paths():       for path in pending_paths():
        lease = try_claim(path)            lease = try_claim(path)
        if lease is None: skip             if lease is None: skip
        … sweep one trace …              … sweep one trace …
        checkpoint.mark_completed        checkpoint.mark_completed
        registry.release(lease)          registry.release(lease)

Disk layout
-----------
Each trace gets one JSON file:
``<checkpoint_dir>/<sha256_of_path[:16]>.json``

The file holds a :class:`CheckpointEntry` with status, per-trace result
or failure, and (if in-progress) the worker id and lease expiry.  Writes
are atomic (write-then-rename) so concurrent workers never see partial
files.

Public surface (re-exported from :mod:`stepback`):

* :class:`CheckpointEntryStatus`
* :class:`CheckpointEntry`
* :class:`SweepCheckpoint`
* :class:`DiskSweepCheckpoint`
* :func:`resume_sweep`
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .worker_lease import (
    InMemoryLeaseRegistry,
    LeaseExpiredError,
    LeaseRegistry,
    WorkerLease,
)


# --------------------------------------------------------------- status


class CheckpointEntryStatus(str, Enum):
    """Processing status of a single trace in a :class:`SweepCheckpoint`."""

    PENDING = "pending"
    """The trace has not yet been claimed by any worker."""

    IN_PROGRESS = "in_progress"
    """A worker holds an active lease for this trace."""

    COMPLETED = "completed"
    """The trace was processed successfully; result is stored."""

    FAILED = "failed"
    """The trace raised an exception during processing."""


# --------------------------------------------------------------- entry


@dataclass
class CheckpointEntry:
    """Per-trace progress record in a :class:`SweepCheckpoint`.

    Attributes
    ----------
    trace_path:
        The original trace file path as passed to :func:`resume_sweep`.
    status:
        Current processing status.
    worker_id:
        Worker that claimed this entry (None if PENDING or COMPLETED).
    lease_id:
        :class:`~stepback.worker_lease.WorkerLease` id (None if not in-progress).
    lease_expires_at:
        Wall-clock time after which the lease is considered expired and
        the trace can be re-claimed.  None when not in-progress.
    started_at:
        ``time.time()`` when the worker claimed this trace.
    completed_at:
        ``time.time()`` when the trace finished (completed or failed).
    result:
        JSON-serialisable dict from :func:`~stepback.sweep.SweepResult`
        (when *status* is ``COMPLETED``).
    failure:
        JSON-serialisable dict from :func:`~stepback.sweep.SweepFailure`
        (when *status* is ``FAILED``).
    """

    trace_path: str
    status: CheckpointEntryStatus = CheckpointEntryStatus.PENDING
    worker_id: Optional[str] = None
    lease_id: Optional[str] = None
    lease_expires_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    failure: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_path": self.trace_path,
            "status": self.status.value,
            "worker_id": self.worker_id,
            "lease_id": self.lease_id,
            "lease_expires_at": self.lease_expires_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "failure": self.failure,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CheckpointEntry":
        return cls(
            trace_path=d["trace_path"],
            status=CheckpointEntryStatus(d["status"]),
            worker_id=d.get("worker_id"),
            lease_id=d.get("lease_id"),
            lease_expires_at=d.get("lease_expires_at"),
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            result=d.get("result"),
            failure=d.get("failure"),
        )

    @property
    def lease_expired(self) -> bool:
        """True if this entry is IN_PROGRESS with an expired lease."""
        return (
            self.status == CheckpointEntryStatus.IN_PROGRESS
            and self.lease_expires_at is not None
            and time.time() > self.lease_expires_at
        )


# --------------------------------------------------------------- ABC


class SweepCheckpoint(ABC):
    """Abstract per-sweep progress store.

    A checkpoint is tied to one sweep invocation (one set of trace paths +
    substitutions).  Multiple workers may call the same checkpoint
    concurrently; implementations must be safe for concurrent readers and
    writers.
    """

    @abstractmethod
    def initialise(self, trace_paths: Sequence[str]) -> None:
        """Register *trace_paths* as PENDING if not already tracked.

        Safe to call from multiple workers; existing entries are not
        overwritten.
        """

    @abstractmethod
    def pending_paths(self) -> List[str]:
        """Return paths that have not yet been completed or failed.

        Includes IN_PROGRESS paths whose leases have expired (they can be
        re-claimed).  Does NOT include paths currently held by an active
        lease.
        """

    @abstractmethod
    def mark_in_progress(
        self,
        trace_path: str,
        lease: WorkerLease,
    ) -> None:
        """Record that *trace_path* is claimed by *lease*."""

    @abstractmethod
    def mark_completed(
        self,
        trace_path: str,
        result: Dict[str, Any],
    ) -> None:
        """Persist a successful sweep result for *trace_path*."""

    @abstractmethod
    def mark_failed(
        self,
        trace_path: str,
        failure: Dict[str, Any],
    ) -> None:
        """Persist a failure record for *trace_path*."""

    @abstractmethod
    def partial_report_data(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return ``(results, failures)`` for already-finished traces.

        *results* is a list of result dicts (COMPLETED entries).
        *failures* is a list of failure dicts (FAILED entries).
        """

    @abstractmethod
    def all_entries(self) -> List[CheckpointEntry]:
        """Return all tracked entries (any status)."""


# --------------------------------------------------------------- disk


def _path_key(trace_path: str) -> str:
    """Stable 16-hex-char filename key for a trace path."""
    return hashlib.sha256(trace_path.encode()).hexdigest()[:16]


class DiskSweepCheckpoint(SweepCheckpoint):
    """File-per-trace checkpoint backed by a local directory.

    Each trace path gets one JSON file:
    ``<root>/<sha256_prefix>.json``

    Writes use atomic rename (write tmp → rename) so concurrent workers
    never read partial files.

    Parameters
    ----------
    root:
        Directory that stores checkpoint files.  Created automatically.
    """

    def __init__(self, root: str) -> None:
        self._root = root
        os.makedirs(root, exist_ok=True)

    def _entry_path(self, trace_path: str) -> str:
        return os.path.join(self._root, _path_key(trace_path) + ".json")

    def _write_entry(self, entry: CheckpointEntry) -> None:
        path = self._entry_path(entry.trace_path)
        fd, tmp = tempfile.mkstemp(dir=self._root, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(entry.to_dict(), fh)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _read_entry(self, trace_path: str) -> Optional[CheckpointEntry]:
        path = self._entry_path(trace_path)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return CheckpointEntry.from_dict(json.load(fh))
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            return None

    def _read_all(self) -> List[CheckpointEntry]:
        entries = []
        try:
            names = os.listdir(self._root)
        except OSError:
            return []
        for name in names:
            if not name.endswith(".json"):
                continue
            path = os.path.join(self._root, name)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    entries.append(CheckpointEntry.from_dict(json.load(fh)))
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                pass
        return entries

    def initialise(self, trace_paths: Sequence[str]) -> None:
        for p in trace_paths:
            if self._read_entry(p) is None:
                self._write_entry(CheckpointEntry(trace_path=p))

    def pending_paths(self) -> List[str]:
        entries = self._read_all()
        result = []
        for e in entries:
            if e.status == CheckpointEntryStatus.PENDING:
                result.append(e.trace_path)
            elif e.status == CheckpointEntryStatus.IN_PROGRESS and e.lease_expired:
                result.append(e.trace_path)
        return result

    def mark_in_progress(self, trace_path: str, lease: WorkerLease) -> None:
        entry = self._read_entry(trace_path) or CheckpointEntry(trace_path=trace_path)
        entry.status = CheckpointEntryStatus.IN_PROGRESS
        entry.worker_id = lease.worker_id
        entry.lease_id = lease.lease_id
        entry.lease_expires_at = lease.expires_at
        entry.started_at = time.time()
        self._write_entry(entry)

    def mark_completed(self, trace_path: str, result: Dict[str, Any]) -> None:
        entry = self._read_entry(trace_path) or CheckpointEntry(trace_path=trace_path)
        entry.status = CheckpointEntryStatus.COMPLETED
        entry.completed_at = time.time()
        entry.result = result
        entry.failure = None
        self._write_entry(entry)

    def mark_failed(self, trace_path: str, failure: Dict[str, Any]) -> None:
        entry = self._read_entry(trace_path) or CheckpointEntry(trace_path=trace_path)
        entry.status = CheckpointEntryStatus.FAILED
        entry.completed_at = time.time()
        entry.failure = failure
        entry.result = None
        self._write_entry(entry)

    def partial_report_data(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        results: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        for e in self._read_all():
            if e.status == CheckpointEntryStatus.COMPLETED and e.result is not None:
                results.append(e.result)
            elif e.status == CheckpointEntryStatus.FAILED and e.failure is not None:
                failures.append(e.failure)
        return results, failures

    def all_entries(self) -> List[CheckpointEntry]:
        return self._read_all()


# --------------------------------------------------------------- resume_sweep


def resume_sweep(
    trace_paths: Sequence[str],
    substitutions: Iterable[Any],
    checkpoint: SweepCheckpoint,
    *,
    baseline_substitutions: Iterable[Any] = (),
    base_step: str = "step:1",
    branch_name_a: str = "baseline",
    branch_name_b: str = "candidate",
    executor_factory: Optional[Callable] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    on_error: str = "record",
    lease_registry: Optional[LeaseRegistry] = None,
    worker_id: Optional[str] = None,
    lease_ttl_seconds: float = 300.0,
    retry_failed: bool = False,
) -> "Any":
    """Run :func:`~stepback.sweep.sweep_traces` with checkpoint recovery.

    Unlike :func:`~stepback.sweep.sweep_traces`, ``resume_sweep``:

    * Reads the *checkpoint* to skip already-completed traces.
    * Claims a :class:`~stepback.worker_lease.WorkerLease` for each trace
      before processing so concurrent workers partition the work without
      overlap.
    * Persists each result or failure immediately to *checkpoint* so a
      worker restart only re-processes traces that were IN_PROGRESS with
      expired leases (i.e. the traces the dead worker had claimed but not
      finished).
    * Merges already-completed results from the checkpoint into the final
      :class:`~stepback.sweep.SweepReport` so the report is always
      complete regardless of how many restarts occurred.

    Parameters
    ----------
    trace_paths:
        Full corpus of trace paths (same as :func:`~stepback.sweep.sweep_traces`).
    substitutions:
        Candidate substitutions for the B branch.
    checkpoint:
        Persistent progress store.  Use :class:`DiskSweepCheckpoint` for
        cross-process / cross-restart recovery.
    baseline_substitutions:
        Substitutions for the A (baseline) branch.  Empty by default.
    base_step:
        Step id at which both branches diverge.
    branch_name_a / branch_name_b:
        Names for the two branches.
    executor_factory:
        Called once per trace to produce an :class:`~stepback.replay.Executor`.
    progress:
        ``progress(i, n, path)`` callback.
    on_error:
        ``"record"`` (default) or ``"raise"``.
    lease_registry:
        Lease backend.  Defaults to a fresh :class:`InMemoryLeaseRegistry`
        when *None* (suitable for single-process sweeps).
    worker_id:
        Human-readable worker identity for audit logs.  Defaults to
        ``"pid-<os.getpid()>"``.
    lease_ttl_seconds:
        TTL for each work-unit lease.  Should be larger than the expected
        per-trace processing time.
    retry_failed:
        When ``True``, previously-failed traces are re-processed.
        Default ``False`` (failed traces are skipped; their failures
        appear in the final report).

    Returns
    -------
    :class:`~stepback.sweep.SweepReport`
        A report over the *full* corpus (completed-from-checkpoint +
        newly processed in this call).
    """
    # Import here to avoid circular dependencies.
    from .sweep import (
        SweepFailure,
        SweepResult,
        SweepReport,
        DistStats,
        _coerce_subs,
        _summarise,
        _sub_to_dict,
        render_sweep_report_json,
    )
    from .replay import Executor, Trace, replay
    from .branch_io import parse_substitution_spec

    if on_error not in ("record", "raise"):
        raise ValueError(f"on_error must be 'record' or 'raise', got {on_error!r}")

    if lease_registry is None:
        lease_registry = InMemoryLeaseRegistry()
    if worker_id is None:
        worker_id = f"pid-{os.getpid()}"

    paths = list(trace_paths)

    # Register all paths in the checkpoint (no-op for already-tracked ones).
    checkpoint.initialise(paths)

    # Determine which paths to process in this call.
    pending = set(checkpoint.pending_paths())
    if retry_failed:
        for e in checkpoint.all_entries():
            if e.status == CheckpointEntryStatus.FAILED:
                pending.add(e.trace_path)

    base_subs = _coerce_subs(baseline_substitutions)
    cand_subs = _coerce_subs(substitutions)

    new_results: List[SweepResult] = []
    new_failures: List[SweepFailure] = []
    n_diverged = 0
    n_decisions_changed = 0

    # Preserve original ordering.
    to_process = [p for p in paths if p in pending]
    n_all = len(paths)

    processed_i = 0
    for i, path in enumerate(paths):
        if path not in pending:
            continue

        if progress is not None:
            progress(i, n_all, path)

        # Try to claim a lease; skip if another worker beat us to it.
        lease = lease_registry.try_claim(
            path, worker_id, ttl_seconds=lease_ttl_seconds
        )
        if lease is None:
            continue

        checkpoint.mark_in_progress(path, lease)

        trace: Optional[Trace] = None
        phase = "load"
        try:
            trace = replay(path)
            phase = "baseline_replay"
            ba = trace.branch_at(base_step, name=branch_name_a)
            for s in base_subs:
                ba.substitute(s)
            executor_a = executor_factory() if executor_factory else Executor(
                fallback_recorded=True
            )
            ra = ba.replay_forward(executor=executor_a)

            phase = "counterfactual_replay"
            bb = trace.branch_at(base_step, name=branch_name_b)
            for s in cand_subs:
                bb.substitute(s)
            executor_b = executor_factory() if executor_factory else Executor(
                fallback_recorded=True
            )
            rb = bb.replay_forward(executor=executor_b)

            phase = "diff"
            d = trace.compare_branches(ba, bb)

            sr = _summarise(path, ra, rb, d)
            new_results.append(sr)
            if sr.divergent_step_count > 0:
                n_diverged += 1
            n_decisions_changed += sr.decisions_changed

            # Persist result.
            checkpoint.mark_completed(path, _sweep_result_to_dict(sr))
            lease_registry.release(lease)

        except Exception as exc:
            if on_error == "raise":
                lease_registry.release(lease)
                raise
            sf = SweepFailure(
                trace_path=path,
                phase=phase,
                error_class=type(exc).__name__,
                message=str(exc),
            )
            new_failures.append(sf)
            checkpoint.mark_failed(path, _sweep_failure_to_dict(sf))
            lease_registry.release(lease)

    if progress is not None:
        progress(n_all, n_all, "")

    # Merge already-completed/failed results from the checkpoint.
    prev_results_raw, prev_failures_raw = checkpoint.partial_report_data()
    prev_results = [_sweep_result_from_dict(r) for r in prev_results_raw]
    prev_failures = [_sweep_failure_from_dict(f) for f in prev_failures_raw]

    # Deduplicate: prefer newly processed results over checkpoint copies.
    new_paths = {r.trace_path for r in new_results} | {f.trace_path for f in new_failures}
    merged_results = new_results + [r for r in prev_results if r.trace_path not in new_paths]
    merged_failures = new_failures + [f for f in prev_failures if f.trace_path not in new_paths]

    # Recompute aggregates from the merged set.
    for r in merged_results:
        if r.trace_path not in {x.trace_path for x in new_results}:
            if r.divergent_step_count > 0:
                n_diverged += 1
            n_decisions_changed += r.decisions_changed

    cost_deltas = [r.cost_delta_usd for r in merged_results]
    div_counts = [float(r.divergent_step_count) for r in merged_results]
    div_fracs = [r.divergent_fraction for r in merged_results]
    base_costs = [r.base_cost_usd for r in merged_results]
    cf_costs = [r.cf_cost_usd for r in merged_results]
    cache_ratios = [r.cf_cache_hit_ratio for r in merged_results]
    real_execs = [float(r.cf_real_executions) for r in merged_results]

    n_succeeded = len(merged_results)
    n_failed = len(merged_failures)
    n_attempted = len(paths)

    return SweepReport(
        n_traces_attempted=n_attempted,
        n_traces_succeeded=n_succeeded,
        n_traces_failed=n_failed,
        n_traces_diverged=n_diverged,
        n_decisions_changed=n_decisions_changed,
        cost_delta_usd=DistStats.of(cost_deltas),
        divergent_step_count=DistStats.of(div_counts),
        divergent_fraction=DistStats.of(div_fracs),
        base_cost_usd=DistStats.of(base_costs),
        cf_cost_usd=DistStats.of(cf_costs),
        cf_cache_hit_ratio=DistStats.of(cache_ratios),
        cf_real_executions=DistStats.of(real_execs),
        results=merged_results,
        failures=merged_failures,
        baseline_substitutions=[_sub_to_dict(s) for s in base_subs],
        candidate_substitutions=[_sub_to_dict(s) for s in cand_subs],
    )


# --------------------------------------------------------------- serialisation helpers


def _sweep_result_to_dict(r: "Any") -> Dict[str, Any]:
    """Minimal serialisation for a SweepResult into the checkpoint."""
    return {
        "_type": "SweepResult",
        "trace_path": r.trace_path,
        "step_count": r.step_count,
        "base_cost_usd": r.base_cost_usd,
        "cf_cost_usd": r.cf_cost_usd,
        "cost_delta_usd": r.cost_delta_usd,
        "divergent_step_count": r.divergent_step_count,
        "base_dirty_count": r.base_dirty_count,
        "cf_dirty_count": r.cf_dirty_count,
        "base_cache_hits": r.base_cache_hits,
        "cf_cache_hits": r.cf_cache_hits,
        "base_real_executions": r.base_real_executions,
        "cf_real_executions": r.cf_real_executions,
        "decisions_changed": r.decisions_changed,
    }


def _sweep_result_from_dict(d: Dict[str, Any]) -> "Any":
    from .sweep import SweepResult
    from .replay import BranchDiff

    # Reconstruct a stub BranchDiff (no step_diffs available from checkpoint).
    stub_diff = BranchDiff(
        a="baseline",
        b="candidate",
        step_diffs=[],
        total_cost_delta_usd=float(d["cost_delta_usd"]),
        divergent_step_count=int(d["divergent_step_count"]),
    )
    return SweepResult(
        trace_path=d["trace_path"],
        step_count=int(d["step_count"]),
        base_cost_usd=float(d["base_cost_usd"]),
        cf_cost_usd=float(d["cf_cost_usd"]),
        cost_delta_usd=float(d["cost_delta_usd"]),
        divergent_step_count=int(d["divergent_step_count"]),
        base_dirty_count=int(d["base_dirty_count"]),
        cf_dirty_count=int(d["cf_dirty_count"]),
        base_cache_hits=int(d["base_cache_hits"]),
        cf_cache_hits=int(d["cf_cache_hits"]),
        base_real_executions=int(d.get("base_real_executions", 0)),
        cf_real_executions=int(d["cf_real_executions"]),
        decisions_changed=int(d.get("decisions_changed", 0)),
        diff=stub_diff,
    )


def _sweep_failure_to_dict(f: "Any") -> Dict[str, Any]:
    return {
        "_type": "SweepFailure",
        "trace_path": f.trace_path,
        "phase": f.phase,
        "error_class": f.error_class,
        "message": f.message,
    }


def _sweep_failure_from_dict(d: Dict[str, Any]) -> "Any":
    from .sweep import SweepFailure
    return SweepFailure(
        trace_path=d["trace_path"],
        phase=d["phase"],
        error_class=d["error_class"],
        message=d["message"],
    )
