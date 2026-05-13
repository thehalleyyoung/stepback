"""Sharded content-addressed step cache (Steps 74 & 138).

Cache entries are keyed by a composite of ``step_kind`` and the hex digest of
``inputs_hash`` (the SHA-256 of the canonical JSON of a step's inputs).  Two
traces whose steps share the same kind+inputs automatically share the same cache
entry — dedup across traces is automatic by construction.

Sharding
--------
The 64-character hex digest is split at position ``shard_width`` (default 2 hex
chars = 1 byte = 8 bits → 256 shards) so directory/prefix listings stay small
even at large scale.

Backends
--------
:class:`DiskStepCache` — stdlib only; stores one JSON file per entry under
    ``<root>/<kind>/<shard>/<hex_digest>.json``.  Writes are atomic
    (write-to-temp then rename) so concurrent branch workers cannot see partial
    files.

:class:`S3StepCache` — requires ``boto3``; stores objects under
    ``<prefix><kind>/<shard>/<hex_digest>.json``.  Raises :exc:`ImportError`
    with a helpful message when ``boto3`` is not installed.

:class:`GCSStepCache` — requires ``google-cloud-storage``; stores blobs under
    ``<prefix><kind>/<shard>/<hex_digest>.json``.  Raises :exc:`ImportError`
    with a helpful message when the SDK is not installed.

:class:`AzureStepCache` — requires ``azure-storage-blob``; stores blobs under
    ``<prefix><kind>/<shard>/<hex_digest>.json``.  Raises :exc:`ImportError`
    with a helpful message when the SDK is not installed.

Determinism contract
--------------------
The step cache is only consulted when a step is dirty due to changed inputs or
a dirty ancestor.  Steps that are forced re-execution by nondeterminism classes
(:func:`~stepback.nondeterminism.forces_dirty`) bypass the cache entirely — a
clock-based or unseeded-RNG step must always call the executor.  Tool-override
substitutions also bypass the cache (the override *is* the output).  Fallback
outputs (``Executor.fallback_recorded``) are never written to the cache.

Cache entry schema
------------------
Entries are stored as JSON with these fields::

    {
        "cache_schema_version": "1",
        "canonicalisation_version": "1",
        "step_kind": "<kind>",
        "inputs_hash": "sha256:<hex>",
        "outputs": <step outputs>,
        "cached_at": <Unix timestamp float>
    }

Entries from an incompatible ``cache_schema_version`` or
``canonicalisation_version`` are silently treated as misses.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

# Canonical version strings so entries can be invalidated on upgrades.
_CACHE_SCHEMA_VERSION = "1"
# Keep in sync with stepback.canonical.CANONICALISATION_VERSION.
_CANONICALISATION_VERSION = "1"


# ------------------------------------------------------------------ helpers


def _hex_digest(inputs_hash: str) -> str:
    """Strip the ``sha256:`` prefix from an inputs_hash and return the hex digest.

    ``hash_obj()`` returns ``'sha256:<64-hex-chars>'``.  We strip the prefix
    before using the string as a path/key component so the colon never appears
    in filenames or object-store keys.

    If the prefix is absent the string is returned unchanged (forward-compat for
    any future hash scheme that does not use ``sha256:``).
    """
    if inputs_hash.startswith("sha256:"):
        return inputs_hash[len("sha256:"):]
    return inputs_hash


def _cache_key(kind: str, inputs_hash: str) -> str:
    """Build a composite cache key from step kind and hex digest.

    Returns ``"<kind>/<hex_digest>"`` — both components are URL/filesystem safe.
    """
    return f"{kind}/{_hex_digest(inputs_hash)}"


# ---------------------------------------------------------------- Entry type


@dataclass
class StepCacheEntry:
    """A single entry in a content-addressed step cache.

    Attributes
    ----------
    step_kind :
        The kind of the recorded step (e.g. ``"llm_call"``, ``"tool_call"``).
    inputs_hash :
        The ``sha256:<hex>`` hash of the canonical-JSON inputs used to produce
        this entry.  Acts as the content-address key.
    outputs :
        The step outputs as returned by the executor.  Must be JSON-serialisable.
    cached_at :
        Unix timestamp (float) when the entry was stored.
    """

    step_kind: str
    inputs_hash: str
    outputs: Any
    cached_at: float

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "cache_schema_version": _CACHE_SCHEMA_VERSION,
            "canonicalisation_version": _CANONICALISATION_VERSION,
            "step_kind": self.step_kind,
            "inputs_hash": self.inputs_hash,
            "outputs": self.outputs,
            "cached_at": self.cached_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StepCacheEntry":
        """Deserialise from a dict; raises :exc:`ValueError` on schema mismatch."""
        if d.get("cache_schema_version") != _CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported cache_schema_version: {d.get('cache_schema_version')!r}"
            )
        if d.get("canonicalisation_version") != _CANONICALISATION_VERSION:
            raise ValueError(
                f"Unsupported canonicalisation_version: {d.get('canonicalisation_version')!r}"
            )
        return cls(
            step_kind=d["step_kind"],
            inputs_hash=d["inputs_hash"],
            outputs=d["outputs"],
            cached_at=float(d["cached_at"]),
        )


# ---------------------------------------------------------------- Abstract base


class StepCache(ABC):
    """Abstract base class for content-addressed step caches.

    All implementations must be **thread-safe**: :meth:`get` and :meth:`put`
    may be called concurrently from parallel branch workers inside
    :func:`~stepback.replay._execute_plan_parallel`.
    """

    @abstractmethod
    def get(self, kind: str, inputs_hash: str) -> Optional[StepCacheEntry]:
        """Return the cached entry for ``(kind, inputs_hash)``, or ``None`` on miss.

        Parameters
        ----------
        kind :
            Step kind string (e.g. ``"llm_call"``).
        inputs_hash :
            ``sha256:<hex>`` string as produced by :func:`~stepback.canonical.hash_obj`.
        """

    @abstractmethod
    def put(self, entry: StepCacheEntry) -> None:
        """Store ``entry`` in the cache.

        Implementations must be idempotent: calling :meth:`put` twice with the
        same ``(step_kind, inputs_hash)`` must not raise and should update the
        entry atomically.
        """

    def close(self) -> None:
        """Release any resources held by the cache (optional; no-op by default)."""


# ----------------------------------------------------------- Disk backend


class DiskStepCache(StepCache):
    """Sharded content-addressed step cache backed by the local filesystem.

    Each entry is stored as a JSON file at::

        <root>/<step_kind>/<shard>/<hex_digest>.json

    where ``<shard>`` is the first ``shard_width`` hex characters of the
    inputs_hash digest.

    Writes are **atomic**: each :meth:`put` writes to a temporary file in the
    same directory as the target, then renames (POSIX ``rename(2)`` is atomic
    within a filesystem).  Concurrent :meth:`get` calls therefore never see
    partial files.

    Parameters
    ----------
    root :
        Root directory for the cache.  Created automatically if it does not
        exist.
    shard_width :
        Number of leading hex characters used as the shard prefix.  Default 2
        gives 256 shards (``00``–``ff``), which keeps listings small for up to
        ~10 million entries per step kind.
    """

    def __init__(self, root: str, shard_width: int = 2) -> None:
        if shard_width < 1:
            raise ValueError(f"shard_width must be ≥ 1, got {shard_width}")
        self.root = os.path.abspath(root)
        self.shard_width = shard_width
        # Per-directory locks ensure only one writer renames into any given shard
        # directory at a time on the same process.  Cross-process safety relies on
        # POSIX rename(2) atomicity.
        self._dir_locks: dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()

    def _lock_for(self, dir_path: str) -> threading.Lock:
        with self._meta_lock:
            if dir_path not in self._dir_locks:
                self._dir_locks[dir_path] = threading.Lock()
            return self._dir_locks[dir_path]

    def _entry_path(self, kind: str, inputs_hash: str) -> str:
        hex_dig = _hex_digest(inputs_hash)
        shard = hex_dig[: self.shard_width]
        return os.path.join(self.root, kind, shard, f"{hex_dig}.json")

    def get(self, kind: str, inputs_hash: str) -> Optional[StepCacheEntry]:
        """Return the cached entry or ``None``."""
        path = self._entry_path(kind, inputs_hash)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            return StepCacheEntry.from_dict(raw)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            # Treat corrupted / incompatible entries as misses.
            return None

    def put(self, entry: StepCacheEntry) -> None:
        """Atomically write ``entry`` to disk."""
        path = self._entry_path(entry.step_kind, entry.inputs_hash)
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        lock = self._lock_for(dir_path)
        payload = json.dumps(entry.to_dict(), ensure_ascii=False)
        with lock:
            # Write to a temp file in the same directory, then rename.
            fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
            try:
                os.write(fd, payload.encode("utf-8"))
                os.close(fd)
                os.replace(tmp_path, path)
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise


# ----------------------------------------------------------- S3 backend


class S3StepCache(StepCache):
    """Sharded content-addressed step cache backed by Amazon S3.

    Objects are stored at::

        <prefix><step_kind>/<shard>/<hex_digest>.json

    Requires ``boto3``.  Raises :exc:`ImportError` at instantiation time if
    ``boto3`` is not installed.

    Parameters
    ----------
    bucket :
        S3 bucket name.
    prefix :
        Object key prefix (e.g. ``"stepback-cache/"``).  Must end with ``"/"``
        if non-empty.
    shard_width :
        Number of leading hex characters used as the shard prefix (default 2).
    boto3_kwargs :
        Extra keyword arguments forwarded to ``boto3.client("s3", **kwargs)``
        (e.g. ``region_name``, ``endpoint_url`` for MinIO / LocalStack).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        shard_width: int = 2,
        **boto3_kwargs: Any,
    ) -> None:
        try:
            import boto3  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "S3StepCache requires 'boto3'.  "
                "Install it with: pip install boto3"
            ) from exc
        self.bucket = bucket
        self.prefix = prefix
        self.shard_width = shard_width
        self._client = boto3.client("s3", **boto3_kwargs)

    def _key(self, kind: str, inputs_hash: str) -> str:
        hex_dig = _hex_digest(inputs_hash)
        shard = hex_dig[: self.shard_width]
        return f"{self.prefix}{kind}/{shard}/{hex_dig}.json"

    def get(self, kind: str, inputs_hash: str) -> Optional[StepCacheEntry]:
        """Return the cached entry or ``None``."""
        try:
            import botocore.exceptions  # type: ignore[import]
        except ImportError:
            return None
        key = self._key(kind, inputs_hash)
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=key)
            raw = json.loads(resp["Body"].read().decode("utf-8"))
            return StepCacheEntry.from_dict(raw)
        except botocore.exceptions.ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def put(self, entry: StepCacheEntry) -> None:
        """Upload ``entry`` to S3."""
        key = self._key(entry.step_kind, entry.inputs_hash)
        payload = json.dumps(entry.to_dict(), ensure_ascii=False).encode("utf-8")
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=payload,
            ContentType="application/json",
        )


# ----------------------------------------------------------- GCS backend


class GCSStepCache(StepCache):
    """Sharded content-addressed step cache backed by Google Cloud Storage.

    Blobs are stored at::

        <prefix><step_kind>/<shard>/<hex_digest>.json

    Requires ``google-cloud-storage``.  Raises :exc:`ImportError` at
    instantiation time if the SDK is not installed.

    Parameters
    ----------
    bucket :
        GCS bucket name.
    prefix :
        Blob name prefix (e.g. ``"stepback-cache/"``).
    shard_width :
        Number of leading hex characters used as the shard prefix (default 2).
    client_kwargs :
        Extra keyword arguments forwarded to ``google.cloud.storage.Client(**kwargs)``.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        shard_width: int = 2,
        **client_kwargs: Any,
    ) -> None:
        try:
            from google.cloud import storage as _gcs  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "GCSStepCache requires 'google-cloud-storage'.  "
                "Install it with: pip install google-cloud-storage"
            ) from exc
        self.bucket_name = bucket
        self.prefix = prefix
        self.shard_width = shard_width
        self._gcs_mod = _gcs
        self._client = _gcs.Client(**client_kwargs)
        self._bucket = self._client.bucket(bucket)

    def _blob_name(self, kind: str, inputs_hash: str) -> str:
        hex_dig = _hex_digest(inputs_hash)
        shard = hex_dig[: self.shard_width]
        return f"{self.prefix}{kind}/{shard}/{hex_dig}.json"

    def get(self, kind: str, inputs_hash: str) -> Optional[StepCacheEntry]:
        """Return the cached entry or ``None``."""
        try:
            from google.cloud.exceptions import NotFound  # type: ignore[import]
        except ImportError:
            return None
        blob = self._bucket.blob(self._blob_name(kind, inputs_hash))
        try:
            data = blob.download_as_bytes()
            raw = json.loads(data.decode("utf-8"))
            return StepCacheEntry.from_dict(raw)
        except NotFound:
            return None
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def put(self, entry: StepCacheEntry) -> None:
        """Upload ``entry`` to GCS."""
        blob = self._bucket.blob(self._blob_name(entry.step_kind, entry.inputs_hash))
        payload = json.dumps(entry.to_dict(), ensure_ascii=False).encode("utf-8")
        blob.upload_from_string(payload, content_type="application/json")


# --------------------------------------------------------- Azure backend


class AzureStepCache(StepCache):
    """Sharded content-addressed step cache backed by Azure Blob Storage.

    Blobs are stored at::

        <prefix><step_kind>/<shard>/<hex_digest>.json

    Requires ``azure-storage-blob``.  Raises :exc:`ImportError` at
    instantiation time if the SDK is not installed.

    Parameters
    ----------
    container :
        Azure Blob Storage container name.
    prefix :
        Blob name prefix (e.g. ``"stepback-cache/"``).
    shard_width :
        Number of leading hex characters used as the shard prefix (default 2).
    connection_string :
        Azure storage connection string.  Passed to
        ``BlobServiceClient.from_connection_string``.
    account_url :
        Alternative to ``connection_string``: storage account URL
        (e.g. ``"https://<account>.blob.core.windows.net"``).
    credential :
        Credential object for ``account_url``-based construction.
    """

    def __init__(
        self,
        container: str,
        prefix: str = "",
        shard_width: int = 2,
        connection_string: Optional[str] = None,
        account_url: Optional[str] = None,
        credential: Any = None,
    ) -> None:
        try:
            from azure.storage.blob import BlobServiceClient  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "AzureStepCache requires 'azure-storage-blob'.  "
                "Install it with: pip install azure-storage-blob"
            ) from exc
        self.container = container
        self.prefix = prefix
        self.shard_width = shard_width
        if connection_string is not None:
            self._service_client = BlobServiceClient.from_connection_string(
                connection_string
            )
        elif account_url is not None:
            self._service_client = BlobServiceClient(
                account_url=account_url, credential=credential
            )
        else:
            raise ValueError(
                "AzureStepCache requires either connection_string or account_url"
            )
        self._container_client = self._service_client.get_container_client(container)

    def _blob_name(self, kind: str, inputs_hash: str) -> str:
        hex_dig = _hex_digest(inputs_hash)
        shard = hex_dig[: self.shard_width]
        return f"{self.prefix}{kind}/{shard}/{hex_dig}.json"

    def get(self, kind: str, inputs_hash: str) -> Optional[StepCacheEntry]:
        """Return the cached entry or ``None``."""
        try:
            from azure.core.exceptions import ResourceNotFoundError  # type: ignore[import]
        except ImportError:
            return None
        blob_client = self._container_client.get_blob_client(
            self._blob_name(kind, inputs_hash)
        )
        try:
            data = blob_client.download_blob().readall()
            raw = json.loads(data.decode("utf-8"))
            return StepCacheEntry.from_dict(raw)
        except ResourceNotFoundError:
            return None
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def put(self, entry: StepCacheEntry) -> None:
        """Upload ``entry`` to Azure Blob Storage."""
        blob_client = self._container_client.get_blob_client(
            self._blob_name(entry.step_kind, entry.inputs_hash)
        )
        payload = json.dumps(entry.to_dict(), ensure_ascii=False).encode("utf-8")
        blob_client.upload_blob(payload, overwrite=True, content_settings=None)


# -------------------------------------------------------- Namespace helpers


import re as _re

_SAFE_NAMESPACE_RE = _re.compile(r"^[A-Za-z0-9._-]{1,256}$")


def _validate_namespace_component(value: str, name: str) -> None:
    """Raise :exc:`ValueError` if *value* is not safe for use in cache key paths.

    Allowed characters: ``[A-Za-z0-9._-]``, length 1–256.  Slashes, dotdot
    sequences, backslashes, and control characters are explicitly rejected to
    prevent path traversal and key collisions.
    """
    if not _SAFE_NAMESPACE_RE.match(value):
        raise ValueError(
            f"Invalid namespace component {name}={value!r}.  "
            "Must match [A-Za-z0-9._-]{1,256} (no slashes, dots-only, "
            "or control characters)."
        )


def _build_namespace_prefix(org_id: str, corpus_id: str) -> str:
    """Return a safe key prefix for the given org / corpus pair.

    Rules
    -----
    * If both *org_id* and *corpus_id* are given → ``"<org>/<corpus>"``
    * If only *org_id* → ``"<org>"``
    * If only *corpus_id* → ``"_/<corpus>"`` (underscore placeholder keeps the
      path depth consistent and avoids ambiguity)
    * If neither → ``""`` (no prefix, same behaviour as the bare backend)
    """
    if org_id and corpus_id:
        return f"{org_id}/{corpus_id}"
    if org_id:
        return org_id
    if corpus_id:
        return f"_/{corpus_id}"
    return ""


# ----------------------------------------------------- NamespacedStepCache


class NamespacedStepCache(StepCache):
    """Wraps any :class:`StepCache` backend and adds org / corpus namespacing.

    Entries are physically stored under a namespace prefix so that different
    organisations and corpora cannot read each other's entries, while identical
    step inputs (same ``inputs_hash``) within the **same** namespace are
    automatically deduplicated across runs.

    Parameters
    ----------
    backend :
        The underlying :class:`StepCache` to delegate reads and writes to.
    org_id :
        Organisation identifier.  Must match ``[A-Za-z0-9._-]{1,256}``; may be
        empty to disable org-level namespacing.
    corpus_id :
        Corpus identifier.  Same character restrictions as *org_id*; may be
        empty.

    Key construction
    ----------------
    The namespace prefix is prepended to the ``kind`` argument so that all
    backends continue to work unchanged::

        namespaced_kind = "<prefix>/<kind>"   # e.g. "acme/ci-corpus/tool_call"

    Returned :class:`StepCacheEntry` objects always carry the **original
    logical** ``step_kind``, not the namespaced one, so the executor and replay
    engine see unmodified kind strings.

    Composition
    -----------
    Wrap a multi-tier cache, then namespace it once::

        cache = NamespacedStepCache(
            MultiTierStepCache(DiskStepCache("/tmp/l1"), S3StepCache("my-bucket")),
            org_id="acme",
            corpus_id="ci-runs",
        )
    """

    def __init__(
        self,
        backend: StepCache,
        org_id: str = "",
        corpus_id: str = "",
    ) -> None:
        if org_id:
            _validate_namespace_component(org_id, "org_id")
        if corpus_id:
            _validate_namespace_component(corpus_id, "corpus_id")
        self._backend = backend
        self.org_id = org_id
        self.corpus_id = corpus_id
        self._prefix = _build_namespace_prefix(org_id, corpus_id)

    def _namespaced_kind(self, kind: str) -> str:
        """Prepend namespace prefix to *kind*.  Returns *kind* unchanged if no prefix."""
        if self._prefix:
            return f"{self._prefix}/{kind}"
        return kind

    def get(self, kind: str, inputs_hash: str) -> Optional[StepCacheEntry]:
        """Return the namespaced cached entry or ``None`` on miss."""
        entry = self._backend.get(self._namespaced_kind(kind), inputs_hash)
        if entry is None:
            return None
        # Restore the original logical step_kind before returning.
        return StepCacheEntry(
            step_kind=kind,
            inputs_hash=entry.inputs_hash,
            outputs=entry.outputs,
            cached_at=entry.cached_at,
        )

    def put(self, entry: StepCacheEntry) -> None:
        """Store *entry* under the namespaced kind."""
        namespaced = StepCacheEntry(
            step_kind=self._namespaced_kind(entry.step_kind),
            inputs_hash=entry.inputs_hash,
            outputs=entry.outputs,
            cached_at=entry.cached_at,
        )
        self._backend.put(namespaced)

    def close(self) -> None:
        """Delegate close to the backend."""
        self._backend.close()


# ---------------------------------------------------- MultiTierStepCache


class MultiTierStepCache(StepCache):
    """Two-tier step cache: fast L1 (e.g. disk) backed by authoritative L2
    (e.g. object storage).

    Read path
    ---------
    1. Check L1.  On hit, return immediately.
    2. On L1 miss, check L2.  On L2 hit, optionally promote the entry to L1
       (best-effort; promotion errors are silently ignored) and return.
    3. If both miss, return ``None``.

    Write path
    ----------
    * L2 is written first and is **authoritative**: an L2 write failure raises
      and propagates to the caller.
    * L1 is written **best-effort** after L2 succeeds; an L1 write error is
      silently suppressed.

    Parameters
    ----------
    l1 :
        Fast local cache (e.g. :class:`DiskStepCache`).
    l2 :
        Slower authoritative cache (e.g. :class:`S3StepCache`).
    promote :
        If ``True`` (default), L2 hits are copied back to L1 on read.
    write_l1 :
        If ``True`` (default), writes go to L1 as well as L2.  Set to
        ``False`` to make L1 read-only (populated only via promotion).
    """

    def __init__(
        self,
        l1: StepCache,
        l2: StepCache,
        promote: bool = True,
        write_l1: bool = True,
    ) -> None:
        self.l1 = l1
        self.l2 = l2
        self.promote = promote
        self.write_l1 = write_l1

    def get(self, kind: str, inputs_hash: str) -> Optional[StepCacheEntry]:
        """Check L1 then L2; promote L2 hits to L1 when ``promote=True``."""
        entry = self.l1.get(kind, inputs_hash)
        if entry is not None:
            return entry
        entry = self.l2.get(kind, inputs_hash)
        if entry is not None and self.promote:
            try:
                self.l1.put(entry)
            except Exception:
                pass  # best-effort; do not let promotion failure lose the hit
        return entry

    def put(self, entry: StepCacheEntry) -> None:
        """Write to L2 (authoritative), then best-effort write to L1."""
        self.l2.put(entry)  # raises on failure — L2 is authoritative
        if self.write_l1:
            try:
                self.l1.put(entry)
            except Exception:
                pass  # best-effort; L2 write already succeeded

    def close(self) -> None:
        """Close both tiers."""
        self.l1.close()
        self.l2.close()

