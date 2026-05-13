"""Tests for ``stepback.worker_lease`` — Step 76.

Covers:
* LeaseStatus enum values and string stability.
* WorkerLease property assertions (is_expired, ttl_remaining, to_dict/from_dict).
* InMemoryLeaseRegistry: claim, double-claim rejected, expiry allows re-claim,
  renew extends expiry, release makes unit available, thread-safety.
* DiskLeaseRegistry: claim, double-claim rejected, expiry on disk, renew,
  release, list_expired.
* LeaseExpiredError carries the lease reference.

All tests are offline (no network calls) and use only stdlib + stepback.
"""
from __future__ import annotations

import threading
import time
from typing import List

import pytest

from stepback.worker_lease import (
    DiskLeaseRegistry,
    InMemoryLeaseRegistry,
    LeaseExpiredError,
    LeaseRegistry,
    LeaseStatus,
    WorkerLease,
)


# ---------------------------------------------------------------- LeaseStatus


def test_lease_status_values() -> None:
    assert LeaseStatus.CLAIMED == "claimed"
    assert LeaseStatus.EXPIRED == "expired"
    assert LeaseStatus.RELEASED == "released"


def test_lease_status_is_str_subclass() -> None:
    assert isinstance(LeaseStatus.CLAIMED, str)


# ---------------------------------------------------------------- WorkerLease


def test_worker_lease_not_expired() -> None:
    lse = WorkerLease(
        lease_id="x",
        work_unit_id="u1",
        worker_id="w1",
        claimed_at=time.time(),
        expires_at=time.time() + 9999,
    )
    assert not lse.is_expired
    assert lse.ttl_remaining > 0.0


def test_worker_lease_expired() -> None:
    lse = WorkerLease(
        lease_id="x",
        work_unit_id="u1",
        worker_id="w1",
        claimed_at=time.time() - 10,
        expires_at=time.time() - 1,
    )
    assert lse.is_expired
    assert lse.ttl_remaining == 0.0


def test_worker_lease_to_dict_round_trip() -> None:
    now = time.time()
    lse = WorkerLease(
        lease_id="abc-123",
        work_unit_id="/traces/t1.sb",
        worker_id="host-42",
        claimed_at=now,
        expires_at=now + 300,
    )
    d = lse.to_dict()
    assert d["lease_id"] == "abc-123"
    assert d["work_unit_id"] == "/traces/t1.sb"
    lse2 = WorkerLease.from_dict(d)
    assert lse2.lease_id == lse.lease_id
    assert lse2.work_unit_id == lse.work_unit_id
    assert lse2.worker_id == lse.worker_id
    assert abs(lse2.claimed_at - lse.claimed_at) < 1e-3
    assert abs(lse2.expires_at - lse.expires_at) < 1e-3


# ---------------------------------------------------------------- LeaseExpiredError


def test_lease_expired_error_carries_lease() -> None:
    lse = WorkerLease(
        lease_id="x", work_unit_id="u", worker_id="w",
        claimed_at=time.time() - 10, expires_at=time.time() - 1,
    )
    err = LeaseExpiredError(lse)
    assert err.lease is lse
    assert "u" in str(err)


# ---------------------------------------------------------------- InMemoryLeaseRegistry


class TestInMemoryLeaseRegistry:
    def test_basic_claim(self) -> None:
        reg = InMemoryLeaseRegistry()
        lse = reg.try_claim("unit1", "worker-A", ttl_seconds=60)
        assert lse is not None
        assert lse.work_unit_id == "unit1"
        assert lse.worker_id == "worker-A"
        assert not lse.is_expired

    def test_double_claim_rejected(self) -> None:
        reg = InMemoryLeaseRegistry()
        lse = reg.try_claim("unit1", "worker-A", ttl_seconds=60)
        assert lse is not None
        lse2 = reg.try_claim("unit1", "worker-B", ttl_seconds=60)
        assert lse2 is None

    def test_expired_lease_can_be_reclaimed(self) -> None:
        reg = InMemoryLeaseRegistry()
        lse = reg.try_claim("unit1", "worker-A", ttl_seconds=0.01)
        assert lse is not None
        time.sleep(0.05)
        # The lease is expired; another worker should be able to claim it.
        lse2 = reg.try_claim("unit1", "worker-B", ttl_seconds=60)
        assert lse2 is not None
        assert lse2.worker_id == "worker-B"

    def test_renew_extends_expiry(self) -> None:
        reg = InMemoryLeaseRegistry()
        lse = reg.try_claim("unit1", "worker-A", ttl_seconds=0.5)
        assert lse is not None
        original_expiry = lse.expires_at
        time.sleep(0.05)
        renewed = reg.renew(lse, ttl_seconds=60)
        assert renewed.expires_at > original_expiry + 50  # well beyond original

    def test_renew_expired_lease_raises(self) -> None:
        reg = InMemoryLeaseRegistry()
        lse = reg.try_claim("unit1", "worker-A", ttl_seconds=0.01)
        assert lse is not None
        time.sleep(0.05)
        with pytest.raises(LeaseExpiredError):
            reg.renew(lse, ttl_seconds=60)

    def test_release_allows_reclaim(self) -> None:
        reg = InMemoryLeaseRegistry()
        lse = reg.try_claim("unit1", "worker-A", ttl_seconds=60)
        assert lse is not None
        reg.release(lse)
        lse2 = reg.try_claim("unit1", "worker-B", ttl_seconds=60)
        assert lse2 is not None

    def test_release_idempotent_after_expiry(self) -> None:
        reg = InMemoryLeaseRegistry()
        lse = reg.try_claim("unit1", "worker-A", ttl_seconds=0.01)
        assert lse is not None
        time.sleep(0.05)
        # Should not raise even if lease has expired or been reclaimed.
        reg.release(lse)

    def test_list_expired_returns_expired_units(self) -> None:
        reg = InMemoryLeaseRegistry()
        reg.try_claim("unit1", "worker-A", ttl_seconds=0.01)
        reg.try_claim("unit2", "worker-A", ttl_seconds=9999)
        time.sleep(0.05)
        expired = reg.list_expired()
        assert "unit1" in expired
        assert "unit2" not in expired

    def test_multiple_independent_units(self) -> None:
        reg = InMemoryLeaseRegistry()
        l1 = reg.try_claim("unit1", "worker-A", ttl_seconds=60)
        l2 = reg.try_claim("unit2", "worker-A", ttl_seconds=60)
        assert l1 is not None and l2 is not None
        assert l1.work_unit_id == "unit1"
        assert l2.work_unit_id == "unit2"

    def test_thread_safety_concurrent_claims(self) -> None:
        reg = InMemoryLeaseRegistry()
        results: List = []
        errors: List = []

        def claim(uid: str) -> None:
            try:
                lse = reg.try_claim(uid, "worker", ttl_seconds=60)
                results.append(lse)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=claim, args=("shared_unit",)) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Exactly one thread should have claimed it.
        claimed = [r for r in results if r is not None]
        assert len(claimed) == 1

    def test_unique_lease_ids(self) -> None:
        reg = InMemoryLeaseRegistry()
        l1 = reg.try_claim("unit1", "w", ttl_seconds=60)
        reg.release(l1)
        l2 = reg.try_claim("unit1", "w", ttl_seconds=60)
        assert l1.lease_id != l2.lease_id


# ---------------------------------------------------------------- DiskLeaseRegistry


class TestDiskLeaseRegistry:
    def test_basic_claim(self, tmp_path) -> None:
        reg = DiskLeaseRegistry(str(tmp_path / "leases"))
        lse = reg.try_claim("/traces/t1.sb", "worker-A", ttl_seconds=60)
        assert lse is not None
        assert lse.work_unit_id == "/traces/t1.sb"

    def test_double_claim_rejected(self, tmp_path) -> None:
        reg = DiskLeaseRegistry(str(tmp_path / "leases"))
        lse = reg.try_claim("unit1", "worker-A", ttl_seconds=60)
        assert lse is not None
        lse2 = reg.try_claim("unit1", "worker-B", ttl_seconds=60)
        assert lse2 is None

    def test_expired_lease_can_be_reclaimed(self, tmp_path) -> None:
        reg = DiskLeaseRegistry(str(tmp_path / "leases"))
        lse = reg.try_claim("unit1", "worker-A", ttl_seconds=0.01)
        assert lse is not None
        time.sleep(0.05)
        lse2 = reg.try_claim("unit1", "worker-B", ttl_seconds=60)
        assert lse2 is not None
        assert lse2.worker_id == "worker-B"

    def test_renew_extends_expiry(self, tmp_path) -> None:
        reg = DiskLeaseRegistry(str(tmp_path / "leases"))
        lse = reg.try_claim("unit1", "worker-A", ttl_seconds=0.5)
        assert lse is not None
        original_expiry = lse.expires_at
        time.sleep(0.02)
        renewed = reg.renew(lse, ttl_seconds=60)
        assert renewed.expires_at > original_expiry + 50

    def test_renew_expired_lease_raises(self, tmp_path) -> None:
        reg = DiskLeaseRegistry(str(tmp_path / "leases"))
        lse = reg.try_claim("unit1", "worker-A", ttl_seconds=0.01)
        assert lse is not None
        time.sleep(0.05)
        with pytest.raises(LeaseExpiredError):
            reg.renew(lse, ttl_seconds=60)

    def test_release_allows_reclaim(self, tmp_path) -> None:
        reg = DiskLeaseRegistry(str(tmp_path / "leases"))
        lse = reg.try_claim("unit1", "worker-A", ttl_seconds=60)
        assert lse is not None
        reg.release(lse)
        lse2 = reg.try_claim("unit1", "worker-B", ttl_seconds=60)
        assert lse2 is not None

    def test_list_expired(self, tmp_path) -> None:
        reg = DiskLeaseRegistry(str(tmp_path / "leases"))
        reg.try_claim("unit1", "w", ttl_seconds=0.01)
        reg.try_claim("unit2", "w", ttl_seconds=9999)
        time.sleep(0.05)
        expired = reg.list_expired()
        assert "unit1" in expired
        assert "unit2" not in expired

    def test_lease_file_persists_on_disk(self, tmp_path) -> None:
        lease_dir = str(tmp_path / "leases")
        reg = DiskLeaseRegistry(lease_dir)
        lse = reg.try_claim("unit1", "worker-A", ttl_seconds=60)
        assert lse is not None
        # There should be exactly one .lease file.
        import os
        files = [f for f in os.listdir(lease_dir) if f.endswith(".lease")]
        assert len(files) == 1

    def test_release_removes_lease_file(self, tmp_path) -> None:
        import os
        lease_dir = str(tmp_path / "leases")
        reg = DiskLeaseRegistry(lease_dir)
        lse = reg.try_claim("unit1", "worker-A", ttl_seconds=60)
        assert lse is not None
        reg.release(lse)
        files = [f for f in os.listdir(lease_dir) if f.endswith(".lease")]
        assert len(files) == 0

    def test_path_sanitization(self, tmp_path) -> None:
        reg = DiskLeaseRegistry(str(tmp_path / "leases"))
        # Paths with slashes should not create subdirectories.
        lse = reg.try_claim("/a/b/c.sb", "w", ttl_seconds=60)
        assert lse is not None
        reg.release(lse)
