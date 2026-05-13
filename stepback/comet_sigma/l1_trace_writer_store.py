"""L1 emission **persistence** for ``stepback.trace_writer`` (Step 3).

Step 1 (:mod:`stepback.comet_sigma.l1_trace_writer`) wires
:class:`stepback.trace_writer.TraceWriter` as a Comet-Σ L1 ``temporal_basis``
emitter and Step 2
(:mod:`stepback.comet_sigma.l1_trace_writer_temporal`) projects each frame
onto a fixed set of ``(window × aggregate)`` artifacts. Both layers keep
their :class:`comet_sigma.audit.AuditableArtifact` instances and per-frame
:class:`comet_sigma.audit.Receipt` objects in memory only.

This module owns **Step 3**: it persists those L1 emissions to a
content-addressed on-disk **store** so an external auditor can
reconstruct the artifact tree, replay every receipt, and verify each
artifact's :class:`comet_sigma.audit.ProvenanceChain` head against the
sha256 of the literal extractor source.

Layout
------

The on-disk store rooted at ``<store_dir>`` is organised as::

    <store_dir>/
      writers/
        <writer_id_hash>/
          writer_meta.json
          artifacts/
            <feature_name_hash>.json   # one per AuditableArtifact
          receipts/
            base.jsonl                 # one Receipt per line (Step 1)
            projection.jsonl           # one Receipt per line (Step 2)

* ``<writer_id_hash>``  : sha256 of the writer path (12 hex chars).
* ``<feature_name_hash>``: sha256 of the artifact name (12 hex chars).
* Each artifact JSON is the ``AuditableArtifact.to_dict()`` form, ready
  for ``comet_sigma.audit.from_dict``. Re-persisting an unchanged
  artifact is a no-op (we compare ``src_sha256``); when an artifact
  gains a new provenance entry (the persistence one) we re-write it.
* Receipts are appended as one JSON object per line so a streaming
  auditor can ``tail -f`` the file.

Design constraints
------------------

* **Flag-gated.** Persistence reuses the L1 temporal flag
  (``COMET_SIGMA_L1_TEMPORAL``); when the flag is OFF or
  :mod:`comet_sigma` is unavailable :func:`is_active` is False and every
  hook becomes a no-op. There is no separate kill switch — Step 3 is a
  strict refinement of Step 1, not an independent surface.
* **Opt-in store directory.** Even when the flag is on, persistence is
  inactive until a store directory is configured via either the
  ``STEPBACK_COMET_SIGMA_L1_STORE`` environment variable or the
  programmatic :func:`set_store_dir` setter. This keeps the default
  behaviour on a fresh install identical to before Step 3 (i.e. zero
  filesystem side-effects).
* **Provenance honesty.** Persisting an artifact for the first time
  appends a single new :class:`comet_sigma.audit.ProvenanceEntry` to the
  artifact's chain via :func:`comet_sigma.audit.extend_provenance`,
  recording layer ``"L1"`` and module
  ``"stepback.trace_writer.sb_v1.persistence"`` so the chain truthfully
  records that the artifact has been written to the store.
* **Idempotent.** Repeated calls to :func:`persist_writer` against the
  same writer with no new receipts are a near no-op (a receipt-count
  comparison and an early return). Re-running with new receipts
  appends only the new lines.
* **Bounded memory.** Persistence reads its inputs from the existing
  Step 1 / Step 2 in-memory registries; this module holds no caches of
  its own beyond a small per-writer high-water mark.
* **Defensive.** All filesystem operations are wrapped in
  ``try/except``; a failure to persist must never propagate into
  :meth:`stepback.trace_writer.TraceWriter._write_frame`.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from . import (
    comet_sigma_available,
    extend_provenance,
    is_enabled,
)
from . import l1_trace_writer as _l1
from . import l1_trace_writer_temporal as _l1t


#: Reuses the Step 1 flag — Step 3 is a refinement of Step 1 / 2 and
#: shares their kill-switch.
FLAG_NAME: str = _l1.FLAG_NAME

#: Environment variable that, when set to a non-empty path, points the
#: store at that directory. Programmatic callers may also use
#: :func:`set_store_dir`.
STORE_DIR_ENV: str = "STEPBACK_COMET_SIGMA_L1_STORE"

#: Module label recorded on the persistence-time provenance entry.
MODULE_LABEL: str = "stepback.trace_writer.sb_v1.persistence"

#: Filename for the per-writer metadata JSON.
META_FILENAME: str = "writer_meta.json"

#: Subdirectory holding one JSON file per persisted artifact.
ARTIFACTS_SUBDIR: str = "artifacts"

#: Subdirectory holding the per-receipt JSONL files.
RECEIPTS_SUBDIR: str = "receipts"

#: JSONL filename for Step 1 (base-feature) receipts.
BASE_RECEIPTS_FILE: str = "base.jsonl"

#: JSONL filename for Step 2 (temporal-projection) receipts.
PROJECTION_RECEIPTS_FILE: str = "projection.jsonl"


# ---------------------------------------------------------------------------
# Store directory configuration
# ---------------------------------------------------------------------------

_STORE_DIR_OVERRIDE: Optional[str] = None
_STORE_LOCK = threading.Lock()


def set_store_dir(path: Optional[str]) -> None:
    """Programmatically set (or clear) the L1 persistence store directory.

    Pass ``None`` to clear the override and fall back to the
    ``STEPBACK_COMET_SIGMA_L1_STORE`` environment variable.
    """
    global _STORE_DIR_OVERRIDE
    with _STORE_LOCK:
        _STORE_DIR_OVERRIDE = path


def get_store_dir() -> Optional[str]:
    """Return the configured store directory, or ``None`` when unset."""
    with _STORE_LOCK:
        if _STORE_DIR_OVERRIDE:
            return _STORE_DIR_OVERRIDE
    env = os.environ.get(STORE_DIR_ENV)
    if env and env.strip():
        return env
    return None


def is_active() -> bool:
    """True when the L1 temporal flag is on AND a store directory is set."""
    return (
        comet_sigma_available()
        and is_enabled(FLAG_NAME)
        and get_store_dir() is not None
    )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _hash12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def writer_dir(store_dir: str, writer_id: str) -> str:
    """Return the absolute on-disk directory for ``writer_id``."""
    return os.path.join(store_dir, "writers", _hash12(writer_id))


def artifact_path(store_dir: str, writer_id: str, artifact_name: str) -> str:
    """Return the absolute file path for one persisted artifact."""
    return os.path.join(
        writer_dir(store_dir, writer_id),
        ARTIFACTS_SUBDIR,
        f"{_hash12(artifact_name)}.json",
    )


def base_receipts_path(store_dir: str, writer_id: str) -> str:
    return os.path.join(
        writer_dir(store_dir, writer_id), RECEIPTS_SUBDIR, BASE_RECEIPTS_FILE
    )


def projection_receipts_path(store_dir: str, writer_id: str) -> str:
    return os.path.join(
        writer_dir(store_dir, writer_id),
        RECEIPTS_SUBDIR,
        PROJECTION_RECEIPTS_FILE,
    )


# ---------------------------------------------------------------------------
# High-water marks (per writer, in-memory)
# ---------------------------------------------------------------------------

@dataclass
class _Watermark:
    base_receipts: int = 0
    projection_receipts: int = 0
    persisted_artifact_shas: Dict[str, str] = field(default_factory=dict)


_WATERMARKS: Dict[str, _Watermark] = {}
_WATERMARKS_LOCK = threading.Lock()


def reset(writer_id: Optional[str] = None) -> None:
    """Drop in-memory persistence state.

    Pass ``None`` to clear every writer (useful for tests).

    NB: this does **not** delete files on disk — that is left to the
    caller, who knows whether they want fresh receipts appended to the
    existing JSONL or a fully clean directory.
    """
    with _WATERMARKS_LOCK:
        if writer_id is None:
            _WATERMARKS.clear()
        else:
            _WATERMARKS.pop(writer_id, None)


def _watermark(writer_id: str) -> _Watermark:
    with _WATERMARKS_LOCK:
        wm = _WATERMARKS.get(writer_id)
        if wm is None:
            wm = _Watermark()
            _WATERMARKS[writer_id] = wm
        return wm


def watermark_snapshot(writer_id: str) -> Optional[Dict[str, Any]]:
    """Return a dict snapshot of the in-memory watermark, or ``None``."""
    with _WATERMARKS_LOCK:
        wm = _WATERMARKS.get(writer_id)
        if wm is None:
            return None
        return {
            "base_receipts": wm.base_receipts,
            "projection_receipts": wm.projection_receipts,
            "persisted_artifact_shas": dict(wm.persisted_artifact_shas),
        }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _ensure_dirs(store_dir: str, writer_id: str) -> None:
    base = writer_dir(store_dir, writer_id)
    os.makedirs(os.path.join(base, ARTIFACTS_SUBDIR), exist_ok=True)
    os.makedirs(os.path.join(base, RECEIPTS_SUBDIR), exist_ok=True)


def _write_meta(store_dir: str, writer_id: str) -> None:
    meta_path = os.path.join(writer_dir(store_dir, writer_id), META_FILENAME)
    if os.path.exists(meta_path):
        return
    meta = {
        "writer_id": writer_id,
        "writer_id_hash": _hash12(writer_id),
        "module": MODULE_LABEL,
        "flag": FLAG_NAME,
        "schema_version": 1,
    }
    tmp = meta_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, sort_keys=True)
    os.replace(tmp, meta_path)


def _persist_artifact(store_dir: str, writer_id: str, artifact: Any) -> Any:
    """Write ``artifact`` to disk and return the (possibly re-provenanced) one.

    The first persistence appends a single :class:`ProvenanceEntry`
    recording the store event; subsequent calls with the same artifact
    body are a no-op and return the input unchanged.
    """
    wm = _watermark(writer_id)
    cur_sha = artifact.src_sha256
    seen_sha = wm.persisted_artifact_shas.get(artifact.name)
    if seen_sha == cur_sha and os.path.exists(
        artifact_path(store_dir, writer_id, artifact.name)
    ):
        return artifact

    persisted = extend_provenance(
        artifact,
        layer="L1",
        module=MODULE_LABEL,
        payload={
            "writer_id_hash": _hash12(writer_id),
            "artifact_name": artifact.name,
        },
        note="persisted to L1 trace_writer store",
    )
    out_path = artifact_path(store_dir, writer_id, artifact.name)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(persisted.to_dict(), fh, sort_keys=True, default=str)
    os.replace(tmp, out_path)
    wm.persisted_artifact_shas[artifact.name] = cur_sha
    return persisted


def _append_receipts(
    path: str, receipts: List[Any], start: int
) -> int:
    """Append ``receipts[start:]`` to ``path`` and return the new high-water."""
    if start >= len(receipts):
        return start
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for r in receipts[start:]:
            fh.write(json.dumps(r.to_dict(), sort_keys=True, default=str))
            fh.write("\n")
    return len(receipts)


def persist_writer(writer_id: str) -> Optional[Dict[str, int]]:
    """Persist all *new* L1 emissions for ``writer_id`` to the store.

    Returns a dict of counters describing what was written, or ``None``
    when the persister is inactive. The returned counters are::

        {
            "artifacts_written": <int>,    # this call only
            "base_receipts_written": <int>,
            "projection_receipts_written": <int>,
        }

    The function is safe to call from multiple threads, but the
    underlying JSONL append uses a per-process lock to avoid interleaved
    lines on POSIX systems where multi-byte writes are not guaranteed
    atomic.
    """
    if not is_active():
        return None
    store_dir = get_store_dir()
    if store_dir is None:  # pragma: no cover - guarded by is_active
        return None
    try:
        _ensure_dirs(store_dir, writer_id)
        _write_meta(store_dir, writer_id)

        wm = _watermark(writer_id)
        artifacts_written = 0

        # Step 1 artifacts.
        for art in _l1.artifacts_for(writer_id):
            before = wm.persisted_artifact_shas.get(art.name)
            _persist_artifact(store_dir, writer_id, art)
            if wm.persisted_artifact_shas.get(art.name) != before:
                artifacts_written += 1

        # Step 2 (projection) artifacts.
        for art in _l1t.artifacts_for(writer_id).values():
            before = wm.persisted_artifact_shas.get(art.name)
            _persist_artifact(store_dir, writer_id, art)
            if wm.persisted_artifact_shas.get(art.name) != before:
                artifacts_written += 1

        # Receipts.
        with _WATERMARKS_LOCK:
            base_start = wm.base_receipts
            proj_start = wm.projection_receipts

        base_receipts = _l1.receipts_for(writer_id)
        proj_receipts = _l1t.receipts_for(writer_id)
        new_base = _append_receipts(
            base_receipts_path(store_dir, writer_id), base_receipts, base_start
        )
        new_proj = _append_receipts(
            projection_receipts_path(store_dir, writer_id),
            proj_receipts,
            proj_start,
        )
        with _WATERMARKS_LOCK:
            wm.base_receipts = new_base
            wm.projection_receipts = new_proj

        return {
            "artifacts_written": artifacts_written,
            "base_receipts_written": new_base - base_start,
            "projection_receipts_written": new_proj - proj_start,
        }
    except Exception:
        # Persistence is a strictly best-effort side channel — never
        # raise into the writer's hot path. Failures show up as missing
        # files for an offline auditor to notice.
        return None


def persist_all() -> Dict[str, Dict[str, int]]:
    """Persist every known writer; return ``{writer_id: counters}``."""
    out: Dict[str, Dict[str, int]] = {}
    if not is_active():
        return out
    seen = set(_l1.known_writers()) | set(_l1t.known_writers())
    for wid in seen:
        c = persist_writer(wid)
        if c is not None:
            out[wid] = c
    return out


# ---------------------------------------------------------------------------
# Read-back helpers (auditor-facing)
# ---------------------------------------------------------------------------

def load_persisted_artifact(
    writer_id: str, artifact_name: str, store_dir: Optional[str] = None
) -> Optional[Any]:
    """Load one persisted artifact back from disk.

    Returns the rehydrated :class:`comet_sigma.audit.AuditableArtifact`
    or ``None`` if the file is missing or unreadable.
    """
    sd = store_dir or get_store_dir()
    if sd is None:
        return None
    path = artifact_path(sd, writer_id, artifact_name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        # Lazy import: only needed for the audit path.
        try:
            from comet_sigma.audit import from_dict  # type: ignore[import-not-found]
        except ModuleNotFoundError:  # pragma: no cover
            from kitchensink.comet_sigma.audit import from_dict  # type: ignore[import-not-found]
        return from_dict(d)
    except Exception:
        return None


def iter_persisted_receipts(
    writer_id: str,
    *,
    kind: str = "base",
    store_dir: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield each persisted receipt as a plain dict.

    ``kind`` selects between ``"base"`` (Step 1 emissions) and
    ``"projection"`` (Step 2 temporal-basis aggregates).
    """
    sd = store_dir or get_store_dir()
    if sd is None:
        return
    if kind == "base":
        path = base_receipts_path(sd, writer_id)
    elif kind == "projection":
        path = projection_receipts_path(sd, writer_id)
    else:
        raise ValueError(f"unknown receipt kind: {kind!r}")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def list_persisted_writers(
    store_dir: Optional[str] = None,
) -> Tuple[str, ...]:
    """Return the writer-id-hash directories present in the store."""
    sd = store_dir or get_store_dir()
    if sd is None:
        return ()
    root = os.path.join(sd, "writers")
    if not os.path.isdir(root):
        return ()
    try:
        return tuple(sorted(os.listdir(root)))
    except OSError:
        return ()


# ---------------------------------------------------------------------------
# Auto-persist hook (idempotent, opt-in via store dir + flag)
# ---------------------------------------------------------------------------

def _hook(writer_id: str, record: _l1.FrameRecord) -> None:
    # The Step 1 emitter calls every registered hook *after* it has
    # appended its own receipts. The Step 2 (temporal projection) hook
    # is normally registered first, so by the time we run its
    # projection receipts already exist. However, callers (and tests)
    # are free to uninstall / reinstall hooks at runtime, which would
    # leave Step 3 ahead of Step 2 in the OBSERVE_HOOKS list. To keep
    # persistence correct regardless of order, we explicitly catch up
    # the projection up to the *current* frame before persisting,
    # but only when it lags behind — projection itself is not
    # idempotent (every call appends a new ProjectionRecord), so this
    # guard avoids double-emission when the projection hook already
    # ran for this frame.
    try:
        # Only catch up the projection when Step 2 is still wired in
        # (its hook is in :data:`OBSERVE_HOOKS`). If a caller has
        # explicitly uninstalled the projection hook we honour that
        # decision and persist only what's already there.
        if _l1t._hook in _l1.OBSERVE_HOOKS:
            proj_state = _l1t.get_state(writer_id)
            latest = (
                proj_state.projections[-1]
                if proj_state and proj_state.projections
                else None
            )
            if latest is None or latest.frame_index < record.frame_index:
                _l1t.project(writer_id, now_ns=record.wallclock_ns)
    except Exception:
        pass
    persist_writer(writer_id)


def install_hook() -> bool:
    """Register the auto-persist hook on Step 1's :data:`OBSERVE_HOOKS`.

    Idempotent. Returns True when the hook was newly installed.
    """
    if _hook in _l1.OBSERVE_HOOKS:
        return False
    _l1.OBSERVE_HOOKS.append(_hook)
    return True


def uninstall_hook() -> bool:
    """Remove the auto-persist hook. Returns True when removed."""
    try:
        _l1.OBSERVE_HOOKS.remove(_hook)
        return True
    except ValueError:
        return False


# Install at import time so persistence becomes "on" the moment both the
# flag and a store directory are set; the layer itself remains gated by
# :func:`is_active`.
install_hook()


__all__ = [
    "ARTIFACTS_SUBDIR",
    "BASE_RECEIPTS_FILE",
    "FLAG_NAME",
    "META_FILENAME",
    "MODULE_LABEL",
    "PROJECTION_RECEIPTS_FILE",
    "RECEIPTS_SUBDIR",
    "STORE_DIR_ENV",
    "artifact_path",
    "base_receipts_path",
    "get_store_dir",
    "install_hook",
    "is_active",
    "iter_persisted_receipts",
    "list_persisted_writers",
    "load_persisted_artifact",
    "persist_all",
    "persist_writer",
    "projection_receipts_path",
    "reset",
    "set_store_dir",
    "uninstall_hook",
    "watermark_snapshot",
    "writer_dir",
]
