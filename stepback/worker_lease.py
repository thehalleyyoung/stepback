"""Worker lease protocol for fault-tolerant distributed replay sweeps (Step 76).

This module provides time-limited exclusive claims on units of work
(trace paths, batch ids, etc.) so that multiple workers partitioning a
million-trace sweep neither double-process the same trace nor leave work
orphaned when a worker crashes.

Architecture
------------
::

    Worker A                    Worker B
    --------                    --------
    registry.try_claim("t1")    registry.try_claim("t1")
    → WorkerLease(...)           → None  (already claimed)
    … process trace …
    registry.renew(lease)       (A dies here)
    …
    registry.release(lease)     registry.list_expired() → ["t1"]
                                registry.try_claim("t1") → WorkerLease(...)

Each registry backend guarantees that for any given *work_unit_id* at most
one active lease exists.  When a worker dies without calling
:meth:`LeaseRegistry.release`, the lease eventually shows up in
:meth:`LeaseRegistry.list_expired` and can be re-claimed.

Backends
--------
:class:`InMemoryLeaseRegistry`
    Thread-safe, in-process.  Suitable for unit tests and local multi-thread
    sweeps.

:class:`DiskLeaseRegistry`
    One JSON file per work-unit in a shared directory.  Uses
    ``open(mode='x')`` for atomic exclusive creation on POSIX, so two
    processes racing to claim the same unit will get a clean
    ``FileExistsError`` → ``None`` without any additional locking.
    Renewal overwrites the file; release deletes it.  Suitable for
    multi-process sweeps that share a network filesystem.

Public surface (re-exported from :mod:`stepback`):

* :class:`LeaseStatus`
* :class:`WorkerLease`
* :exc:`LeaseExpiredError`
* :class:`LeaseRegistry`
* :class:`InMemoryLeaseRegistry`
* :class:`DiskLeaseRegistry`
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# --------------------------------------------------------------- status


class LeaseStatus(str, Enum):
    """Status of a worker lease claim."""

    CLAIMED = "claimed"
    """Lease is active; the holder may renew or release it."""

    EXPIRED = "expired"
    """Lease TTL elapsed without renewal; the unit can be re-claimed."""

    RELEASED = "released"
    """Lease was voluntarily released by the holder."""


# --------------------------------------------------------------- data


@dataclass
class WorkerLease:
    """A time-limited exclusive claim on a unit of work.

    Instances are returned by :meth:`LeaseRegistry.try_claim` and updated
    in-place by :meth:`LeaseRegistry.renew`.  The holder must call
    :meth:`LeaseRegistry.renew` periodically (before ``expires_at``) and
    :meth:`LeaseRegistry.release` when done.

    Attributes
    ----------
    lease_id:
        Unique identifier for this particular claim.  Two workers claiming
        the same *work_unit_id* at different times get different lease ids.
    work_unit_id:
        Opaque identifier for the unit of work (e.g. a trace path or batch
        number).
    worker_id:
        Caller-supplied worker identity (host:pid, UUID, etc.).
    claimed_at:
        Wall-clock time (``time.time()``) when the lease was granted.
    expires_at:
        Wall-clock time (``time.time()``) after which the lease is
        considered expired and can be re-claimed.
    """

    lease_id: str
    work_unit_id: str
    worker_id: str
    claimed_at: float
    expires_at: float

    @property
    def is_expired(self) -> bool:
        """True if the current wall-clock time is past *expires_at*."""
        return time.time() > self.expires_at

    @property
    def ttl_remaining(self) -> float:
        """Seconds remaining until expiry (0.0 if already expired)."""
        return max(0.0, self.expires_at - time.time())

    def to_dict(self) -> dict:
        return {
            "lease_id": self.lease_id,
            "work_unit_id": self.work_unit_id,
            "worker_id": self.worker_id,
            "claimed_at": self.claimed_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkerLease":
        return cls(
            lease_id=d["lease_id"],
            work_unit_id=d["work_unit_id"],
            worker_id=d["worker_id"],
            claimed_at=float(d["claimed_at"]),
            expires_at=float(d["expires_at"]),
        )


# --------------------------------------------------------------- errors


class LeaseExpiredError(Exception):
    """Raised when an operation requires an active lease but it has expired.

    Callers that catch this should stop processing the work unit and let it
    be re-claimed by another worker via :meth:`LeaseRegistry.list_expired`.
    """

    def __init__(self, lease: WorkerLease) -> None:
        super().__init__(
            f"Lease {lease.lease_id!r} for unit {lease.work_unit_id!r} "
            f"expired at {lease.expires_at:.3f} (now {time.time():.3f})."
        )
        self.lease = lease


# --------------------------------------------------------------- ABC


class LeaseRegistry(ABC):
    """Abstract lease registry.

    Every backend must guarantee that for any *work_unit_id* at most one
    active (non-expired, non-released) lease exists.
    """

    @abstractmethod
    def try_claim(
        self,
        work_unit_id: str,
        worker_id: str,
        ttl_seconds: float = 300.0,
    ) -> Optional[WorkerLease]:
        """Atomically claim *work_unit_id*.

        Returns a :class:`WorkerLease` if the claim succeeds.  Returns
        ``None`` if another worker currently holds an active lease.

        Parameters
        ----------
        work_unit_id:
            Opaque unit identifier (e.g. trace path or batch id).
        worker_id:
            Caller-supplied worker identity.
        ttl_seconds:
            How long the lease stays valid without renewal.  After this the
            unit appears in :meth:`list_expired`.
        """

    @abstractmethod
    def renew(self, lease: WorkerLease, ttl_seconds: float = 300.0) -> WorkerLease:
        """Extend the lease by *ttl_seconds* from now.

        Returns an updated :class:`WorkerLease` with the new ``expires_at``.
        Raises :exc:`LeaseExpiredError` if the lease has already expired
        (the holder lost the right to renew).
        """

    @abstractmethod
    def release(self, lease: WorkerLease) -> None:
        """Voluntarily release the lease so the unit can be re-claimed.

        Safe to call even if the lease has already expired.
        """

    @abstractmethod
    def list_expired(self) -> List[str]:
        """Return *work_unit_ids* whose leases have expired.

        Callers may re-claim each returned id via :meth:`try_claim`.
        """


# --------------------------------------------------------------- in-memory


class InMemoryLeaseRegistry(LeaseRegistry):
    """Thread-safe in-process lease registry.

    Uses :mod:`threading` internals only; suitable for unit tests and
    multi-thread sweeps within a single process.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # work_unit_id → WorkerLease (only active leases stored)
        self._leases: Dict[str, WorkerLease] = {}

    def try_claim(
        self,
        work_unit_id: str,
        worker_id: str,
        ttl_seconds: float = 300.0,
    ) -> Optional[WorkerLease]:
        with self._lock:
            existing = self._leases.get(work_unit_id)
            if existing is not None and not existing.is_expired:
                return None
            now = time.time()
            lease = WorkerLease(
                lease_id=str(uuid.uuid4()),
                work_unit_id=work_unit_id,
                worker_id=worker_id,
                claimed_at=now,
                expires_at=now + ttl_seconds,
            )
            self._leases[work_unit_id] = lease
            return lease

    def renew(self, lease: WorkerLease, ttl_seconds: float = 300.0) -> WorkerLease:
        with self._lock:
            stored = self._leases.get(lease.work_unit_id)
            if stored is None or stored.lease_id != lease.lease_id:
                raise LeaseExpiredError(lease)
            if stored.is_expired:
                raise LeaseExpiredError(lease)
            new_expires = time.time() + ttl_seconds
            stored.expires_at = new_expires
            lease.expires_at = new_expires
            return lease

    def release(self, lease: WorkerLease) -> None:
        with self._lock:
            stored = self._leases.get(lease.work_unit_id)
            if stored is not None and stored.lease_id == lease.lease_id:
                del self._leases[lease.work_unit_id]

    def list_expired(self) -> List[str]:
        with self._lock:
            return [
                uid
                for uid, lse in self._leases.items()
                if lse.is_expired
            ]


# --------------------------------------------------------------- disk


class DiskLeaseRegistry(LeaseRegistry):
    """File-per-unit lease registry backed by a local (or NFS) directory.

    Each work unit gets one JSON file at ``<root>/<sanitized_id>.lease``.
    ``try_claim`` uses ``open(mode='x')`` (exclusive create) for atomic
    clash detection on POSIX.  If two processes race, exactly one wins
    and the other gets ``None``.

    Renewal overwrites the lease file atomically via a write-then-rename
    pattern.  Release deletes the file.

    Parameters
    ----------
    root:
        Directory to store lease files.  Created automatically if absent.
    """

    def __init__(self, root: str) -> None:
        self._root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, work_unit_id: str) -> str:
        safe = work_unit_id.replace("/", "__").replace("\\", "__")
        return os.path.join(self._root, safe + ".lease")

    def try_claim(
        self,
        work_unit_id: str,
        worker_id: str,
        ttl_seconds: float = 300.0,
    ) -> Optional[WorkerLease]:
        path = self._path(work_unit_id)
        now = time.time()
        lease = WorkerLease(
            lease_id=str(uuid.uuid4()),
            work_unit_id=work_unit_id,
            worker_id=worker_id,
            claimed_at=now,
            expires_at=now + ttl_seconds,
        )
        # Check if an active (non-expired) lease already exists.
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    existing = WorkerLease.from_dict(json.load(fh))
                if not existing.is_expired:
                    return None
                # Expired lease; delete and re-claim below.
                os.remove(path)
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                # Corrupt or deleted; fall through and try to create.
                pass
        # Attempt exclusive creation.
        try:
            with open(path, "x", encoding="utf-8") as fh:
                json.dump(lease.to_dict(), fh)
            return lease
        except FileExistsError:
            # Another worker created it between our check and create.
            return None

    def renew(self, lease: WorkerLease, ttl_seconds: float = 300.0) -> WorkerLease:
        path = self._path(lease.work_unit_id)
        if not os.path.exists(path):
            raise LeaseExpiredError(lease)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                stored = WorkerLease.from_dict(json.load(fh))
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise LeaseExpiredError(lease) from exc
        if stored.lease_id != lease.lease_id:
            raise LeaseExpiredError(lease)
        if stored.is_expired:
            raise LeaseExpiredError(lease)
        lease.expires_at = time.time() + ttl_seconds
        # Atomic overwrite.
        dir_ = os.path.dirname(path)
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".lease.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(lease.to_dict(), fh)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return lease

    def release(self, lease: WorkerLease) -> None:
        path = self._path(lease.work_unit_id)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                stored = WorkerLease.from_dict(json.load(fh))
            if stored.lease_id == lease.lease_id:
                os.remove(path)
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            pass

    def list_expired(self) -> List[str]:
        expired = []
        try:
            entries = os.listdir(self._root)
        except OSError:
            return []
        for name in entries:
            if not name.endswith(".lease"):
                continue
            path = os.path.join(self._root, name)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    lse = WorkerLease.from_dict(json.load(fh))
                if lse.is_expired:
                    expired.append(lse.work_unit_id)
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                pass
        return expired
