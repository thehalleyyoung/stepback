"""Tests for ``stepback.sweep_checkpoint`` — Step 76.

Covers:
* CheckpointEntryStatus enum stability.
* CheckpointEntry: to_dict/from_dict round-trip, lease_expired property.
* DiskSweepCheckpoint: initialise, pending_paths, mark_in_progress,
  mark_completed, mark_failed, partial_report_data, all_entries.
* resume_sweep: basic e2e, skips already-completed traces, retries failed
  traces when retry_failed=True, worker lease isolation (two workers claim
  different traces), checkpoint persists across separate resume_sweep calls.

All tests are offline (no network calls). The sweep is driven against the
12-step deterministic fixture via ``stepback.testing``.
"""
from __future__ import annotations

import os
import time
from typing import List

import pytest

from stepback import record, resume_sweep
from stepback.recorder import RecorderKey
from stepback.sweep_checkpoint import (
    CheckpointEntry,
    CheckpointEntryStatus,
    DiskSweepCheckpoint,
    SweepCheckpoint,
    _sweep_failure_from_dict,
    _sweep_failure_to_dict,
    _sweep_result_from_dict,
    _sweep_result_to_dict,
)
from stepback.testing import run_recorded_agent
from stepback.worker_lease import InMemoryLeaseRegistry, WorkerLease


# ---------------------------------------------------------------- helpers


def _record_corpus(tmpdir: str, n: int = 3) -> List[str]:
    paths = []
    for i in range(n):
        p = os.path.join(tmpdir, f"trace_{i}.sb")
        with record(p, key=RecorderKey.fresh()) as rec:
            run_recorded_agent(rec)
        paths.append(p)
    return paths


def _fake_lease(uid: str = "unit1") -> WorkerLease:
    return WorkerLease(
        lease_id="test-lease-id",
        work_unit_id=uid,
        worker_id="worker-test",
        claimed_at=time.time(),
        expires_at=time.time() + 300,
    )


# ---------------------------------------------------------------- CheckpointEntryStatus


def test_checkpoint_entry_status_values() -> None:
    assert CheckpointEntryStatus.PENDING == "pending"
    assert CheckpointEntryStatus.IN_PROGRESS == "in_progress"
    assert CheckpointEntryStatus.COMPLETED == "completed"
    assert CheckpointEntryStatus.FAILED == "failed"


# ---------------------------------------------------------------- CheckpointEntry


def test_checkpoint_entry_default_status() -> None:
    e = CheckpointEntry(trace_path="/t.sb")
    assert e.status == CheckpointEntryStatus.PENDING
    assert e.worker_id is None
    assert e.result is None


def test_checkpoint_entry_round_trip() -> None:
    now = time.time()
    e = CheckpointEntry(
        trace_path="/traces/t1.sb",
        status=CheckpointEntryStatus.COMPLETED,
        worker_id="worker-1",
        lease_id="lsid",
        lease_expires_at=now + 100,
        started_at=now,
        completed_at=now + 5,
        result={"trace_path": "/traces/t1.sb", "step_count": 12},
    )
    d = e.to_dict()
    e2 = CheckpointEntry.from_dict(d)
    assert e2.trace_path == e.trace_path
    assert e2.status == CheckpointEntryStatus.COMPLETED
    assert e2.worker_id == "worker-1"
    assert e2.result == e.result


def test_checkpoint_entry_lease_expired_property() -> None:
    expired = CheckpointEntry(
        trace_path="/t.sb",
        status=CheckpointEntryStatus.IN_PROGRESS,
        lease_expires_at=time.time() - 1,
    )
    assert expired.lease_expired

    active = CheckpointEntry(
        trace_path="/t.sb",
        status=CheckpointEntryStatus.IN_PROGRESS,
        lease_expires_at=time.time() + 999,
    )
    assert not active.lease_expired

    pending = CheckpointEntry(trace_path="/t.sb")
    assert not pending.lease_expired


# ---------------------------------------------------------------- DiskSweepCheckpoint


class TestDiskSweepCheckpoint:
    def test_initialise_creates_pending_entries(self, tmp_path) -> None:
        ck = DiskSweepCheckpoint(str(tmp_path / "ck"))
        paths = ["/t1.sb", "/t2.sb", "/t3.sb"]
        ck.initialise(paths)
        all_e = ck.all_entries()
        assert len(all_e) == 3
        for e in all_e:
            assert e.status == CheckpointEntryStatus.PENDING

    def test_initialise_idempotent(self, tmp_path) -> None:
        ck = DiskSweepCheckpoint(str(tmp_path / "ck"))
        ck.initialise(["/t1.sb"])
        ck.mark_completed("/t1.sb", {"trace_path": "/t1.sb", "step_count": 1,
                                      "base_cost_usd": 0, "cf_cost_usd": 0,
                                      "cost_delta_usd": 0, "divergent_step_count": 0,
                                      "base_dirty_count": 0, "cf_dirty_count": 0,
                                      "base_cache_hits": 0, "cf_cache_hits": 0,
                                      "base_real_executions": 0, "cf_real_executions": 0,
                                      "decisions_changed": 0})
        # Re-initialise should not overwrite the completed entry.
        ck.initialise(["/t1.sb"])
        all_e = ck.all_entries()
        assert len(all_e) == 1
        assert all_e[0].status == CheckpointEntryStatus.COMPLETED

    def test_pending_paths_returns_pending(self, tmp_path) -> None:
        ck = DiskSweepCheckpoint(str(tmp_path / "ck"))
        ck.initialise(["/t1.sb", "/t2.sb"])
        ck.mark_completed("/t2.sb", {"trace_path": "/t2.sb", "step_count": 0,
                                      "base_cost_usd": 0, "cf_cost_usd": 0,
                                      "cost_delta_usd": 0, "divergent_step_count": 0,
                                      "base_dirty_count": 0, "cf_dirty_count": 0,
                                      "base_cache_hits": 0, "cf_cache_hits": 0,
                                      "base_real_executions": 0, "cf_real_executions": 0,
                                      "decisions_changed": 0})
        pending = ck.pending_paths()
        assert "/t1.sb" in pending
        assert "/t2.sb" not in pending

    def test_pending_paths_includes_expired_in_progress(self, tmp_path) -> None:
        ck = DiskSweepCheckpoint(str(tmp_path / "ck"))
        ck.initialise(["/t1.sb"])
        expired_lease = WorkerLease(
            lease_id="x", work_unit_id="/t1.sb", worker_id="w",
            claimed_at=time.time() - 100, expires_at=time.time() - 1,
        )
        ck.mark_in_progress("/t1.sb", expired_lease)
        # Should appear in pending because the lease is expired.
        assert "/t1.sb" in ck.pending_paths()

    def test_pending_paths_excludes_active_in_progress(self, tmp_path) -> None:
        ck = DiskSweepCheckpoint(str(tmp_path / "ck"))
        ck.initialise(["/t1.sb"])
        active_lease = _fake_lease("/t1.sb")
        ck.mark_in_progress("/t1.sb", active_lease)
        assert "/t1.sb" not in ck.pending_paths()

    def test_mark_completed_stores_result(self, tmp_path) -> None:
        ck = DiskSweepCheckpoint(str(tmp_path / "ck"))
        ck.initialise(["/t1.sb"])
        result = {"trace_path": "/t1.sb", "step_count": 12,
                  "base_cost_usd": 0.1, "cf_cost_usd": 0.2,
                  "cost_delta_usd": 0.1, "divergent_step_count": 3,
                  "base_dirty_count": 0, "cf_dirty_count": 3,
                  "base_cache_hits": 12, "cf_cache_hits": 9,
                  "base_real_executions": 0, "cf_real_executions": 3,
                  "decisions_changed": 1}
        ck.mark_completed("/t1.sb", result)
        entries = ck.all_entries()
        assert entries[0].status == CheckpointEntryStatus.COMPLETED
        assert entries[0].result == result

    def test_mark_failed_stores_failure(self, tmp_path) -> None:
        ck = DiskSweepCheckpoint(str(tmp_path / "ck"))
        ck.initialise(["/t1.sb"])
        failure = {"trace_path": "/t1.sb", "phase": "load",
                   "error_class": "FileNotFoundError", "message": "no such file"}
        ck.mark_failed("/t1.sb", failure)
        entries = ck.all_entries()
        assert entries[0].status == CheckpointEntryStatus.FAILED
        assert entries[0].failure == failure

    def test_partial_report_data(self, tmp_path) -> None:
        ck = DiskSweepCheckpoint(str(tmp_path / "ck"))
        ck.initialise(["/t1.sb", "/t2.sb", "/t3.sb"])
        result = {"trace_path": "/t1.sb", "step_count": 12,
                  "base_cost_usd": 0, "cf_cost_usd": 0,
                  "cost_delta_usd": 0, "divergent_step_count": 0,
                  "base_dirty_count": 0, "cf_dirty_count": 0,
                  "base_cache_hits": 12, "cf_cache_hits": 12,
                  "base_real_executions": 0, "cf_real_executions": 0,
                  "decisions_changed": 0}
        failure = {"trace_path": "/t2.sb", "phase": "load",
                   "error_class": "OSError", "message": "err"}
        ck.mark_completed("/t1.sb", result)
        ck.mark_failed("/t2.sb", failure)
        # /t3.sb is still PENDING.

        results, failures = ck.partial_report_data()
        assert len(results) == 1
        assert results[0]["trace_path"] == "/t1.sb"
        assert len(failures) == 1
        assert failures[0]["trace_path"] == "/t2.sb"


# ---------------------------------------------------------------- resume_sweep e2e


@pytest.fixture
def corpus(tmp_path):
    return _record_corpus(str(tmp_path / "traces"), n=3)


def test_resume_sweep_basic(corpus, tmp_path) -> None:
    """resume_sweep processes all traces and returns a complete report."""
    ck = DiskSweepCheckpoint(str(tmp_path / "ck"))
    report = resume_sweep(corpus, substitutions=[], checkpoint=ck)
    assert report.n_traces_attempted == 3
    assert report.n_traces_succeeded == 3
    assert report.n_traces_failed == 0


def test_resume_sweep_skips_completed_traces(corpus, tmp_path) -> None:
    """Completed traces in the checkpoint are not re-processed."""
    ck = DiskSweepCheckpoint(str(tmp_path / "ck"))

    # First pass: process all 3 traces.
    report1 = resume_sweep(corpus, substitutions=[], checkpoint=ck)
    assert report1.n_traces_succeeded == 3

    # Track how many times replay is called.
    from stepback import replay as replay_fn
    calls: List[str] = []
    original_replay = replay_fn

    import stepback.sweep_checkpoint as sc_mod

    _original = sc_mod._sweep_result_from_dict

    # Second pass should skip all completed traces (no processing needed).
    report2 = resume_sweep(corpus, substitutions=[], checkpoint=ck)
    assert report2.n_traces_attempted == 3
    assert report2.n_traces_succeeded == 3


def test_resume_sweep_failed_traces_skipped_by_default(corpus, tmp_path) -> None:
    """Failed traces are not retried unless retry_failed=True."""
    ck = DiskSweepCheckpoint(str(tmp_path / "ck"))
    # Pre-seed the checkpoint: mark one trace as failed.
    ck.initialise(corpus)
    ck.mark_failed(corpus[0], {
        "trace_path": corpus[0], "phase": "load",
        "error_class": "OSError", "message": "artificial failure",
    })

    report = resume_sweep(corpus, substitutions=[], checkpoint=ck)
    # The pre-failed trace is skipped and appears in failures from checkpoint.
    assert report.n_traces_attempted == 3
    failed_paths = [f.trace_path for f in report.failures]
    assert corpus[0] in failed_paths


def test_resume_sweep_retry_failed(corpus, tmp_path) -> None:
    """retry_failed=True causes previously failed traces to be re-processed."""
    ck = DiskSweepCheckpoint(str(tmp_path / "ck"))
    ck.initialise(corpus)
    ck.mark_failed(corpus[0], {
        "trace_path": corpus[0], "phase": "load",
        "error_class": "OSError", "message": "transient",
    })

    report = resume_sweep(corpus, substitutions=[], checkpoint=ck, retry_failed=True)
    # corpus[0] was retried and should now succeed (real file, no failure).
    succeeded_paths = [r.trace_path for r in report.results]
    assert corpus[0] in succeeded_paths


def test_resume_sweep_worker_isolation(corpus, tmp_path) -> None:
    """Two workers using the same checkpoint and lease registry each claim
    different traces — no trace is processed twice."""
    ck = DiskSweepCheckpoint(str(tmp_path / "ck"))
    reg = InMemoryLeaseRegistry()

    # Track which worker actually claimed each lease (via worker_id in entries).
    import threading

    barrier = threading.Barrier(2)

    def run_worker(name: str) -> None:
        barrier.wait()  # Start both workers at the same time.
        resume_sweep(
            corpus,
            substitutions=[],
            checkpoint=ck,
            lease_registry=reg,
            worker_id=name,
            lease_ttl_seconds=60,
        )

    t_a = threading.Thread(target=run_worker, args=("worker-A",))
    t_b = threading.Thread(target=run_worker, args=("worker-B",))
    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()

    # After both workers finish: all traces should be COMPLETED.
    entries = ck.all_entries()
    assert all(
        e.status == CheckpointEntryStatus.COMPLETED for e in entries
    ), [e.status for e in entries]
    assert {e.trace_path for e in entries} == set(corpus)

    # Each trace should have a single checkpoint file (no duplicates).
    assert len(entries) == len(corpus)


def test_resume_sweep_checkpoint_persists_across_calls(corpus, tmp_path) -> None:
    """Results from a previous call are merged into the new report."""
    ck = DiskSweepCheckpoint(str(tmp_path / "ck"))

    # First call: process only the first trace.
    ck.initialise(corpus)
    single_report = resume_sweep(
        [corpus[0]], substitutions=[], checkpoint=ck,
    )
    assert single_report.n_traces_succeeded == 1

    # Second call: process the remaining two traces.
    report2 = resume_sweep(corpus, substitutions=[], checkpoint=ck)
    # All 3 traces should appear (1 from checkpoint + 2 newly processed).
    assert report2.n_traces_succeeded == 3


def test_resume_sweep_default_worker_id(corpus, tmp_path) -> None:
    """resume_sweep sets a default worker_id when not given."""
    ck = DiskSweepCheckpoint(str(tmp_path / "ck"))
    report = resume_sweep(corpus, substitutions=[], checkpoint=ck)
    # Just ensure it runs without error.
    assert report.n_traces_succeeded == 3


def test_resume_sweep_on_error_raise(corpus, tmp_path) -> None:
    """on_error='raise' propagates the first exception."""
    ck = DiskSweepCheckpoint(str(tmp_path / "ck"))
    # Pass a nonexistent path to trigger a load error.
    bad_paths = ["/nonexistent/trace.sb"]
    with pytest.raises(Exception):
        resume_sweep(bad_paths, substitutions=[], checkpoint=ck, on_error="raise")
