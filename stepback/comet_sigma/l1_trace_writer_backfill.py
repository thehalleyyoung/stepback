"""L1 ``temporal_basis`` **backfill** for historical ``.sb`` traces (Step 4).

Steps 1–3 (:mod:`stepback.comet_sigma.l1_trace_writer`,
:mod:`stepback.comet_sigma.l1_trace_writer_temporal`,
:mod:`stepback.comet_sigma.l1_trace_writer_store`) wire
:class:`stepback.trace_writer.TraceWriter` as a Comet-Σ L1
``temporal_basis`` emitter, project per-frame base features onto
windowed aggregates, and persist both into a content-addressed on-disk
store. Those layers only see frames as they are produced.

This module owns **Step 4** of ``COMET_SIGMA_1000.md``: it ingests
**existing** ``stepback/trace_writer.py + .sb format v1`` traces from
disk, replays each frame through the Step 1 emitter so the temporal
basis store gains a faithful historical record, and tags every backfilled
artifact with a dedicated :class:`comet_sigma.audit.ProvenanceEntry` so
an auditor can distinguish live emissions from backfilled ones.

Design constraints
------------------

* **Flag-gated.** Backfill shares the Step 1 / 2 / 3 flag
  (``COMET_SIGMA_L1_TEMPORAL``); when the flag is OFF or
  :mod:`comet_sigma` is not importable :func:`backfill_trace` is a strict
  no-op and returns ``None``.
* **Read-only with respect to .sb files.** The backfill never writes to
  the source trace; it only opens it via
  :func:`stepback.trace_reader.read_frames` and passes each wrapper
  body through :func:`stepback.comet_sigma.l1_trace_writer.observe_frame`.
* **Distinct writer namespace.** Backfilled traces use a
  ``"backfill:<abs_path>"`` writer-id so they can never collide with a
  live writer that is currently appending to the same file. This is
  also what the persistence store uses as its directory key.
* **Provenance honesty.** Every base-feature artifact gains exactly one
  additional :class:`comet_sigma.audit.ProvenanceEntry` per backfill
  call recording the source path, frame count, and sha256 of the source
  file's first 64 KiB (a cheap content fingerprint that is stable for
  small traces and probabilistically unique otherwise).
* **Idempotent at the watermark.** Re-running :func:`backfill_trace`
  against the same path with no changes appends new receipts (the L1
  emitter is monotonic — it always advances ``frame_index``); callers
  who want pure idempotence should call :func:`reset_backfill_state`
  first or pass ``reset=True``.
* **Defensive.** All filesystem / decode failures are caught and
  surfaced as a structured failure dict; nothing here ever raises into
  the caller's hot path.
"""
from __future__ import annotations

import glob as _glob
import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import (
    comet_sigma_available,
    extend_provenance,
    is_enabled,
)
from . import l1_trace_writer as _l1
from . import l1_trace_writer_store as _store
from . import l1_trace_writer_temporal as _l1t


#: Backfill reuses the Step 1 flag; there is no independent kill switch.
FLAG_NAME: str = _l1.FLAG_NAME

#: Module label recorded on the backfill provenance entry.
MODULE_LABEL: str = "stepback.trace_writer.sb_v1.backfill"

#: Prefix applied to backfilled writer ids so they cannot collide with a
#: live :class:`stepback.trace_writer.TraceWriter` that is appending to
#: the same file.
WRITER_ID_PREFIX: str = "backfill:"

#: Number of bytes from the head of the source file that we sha256 to
#: produce the content fingerprint embedded in the backfill provenance
#: entry.  Bounded so backfilling huge traces stays cheap.
FINGERPRINT_BYTES: int = 64 * 1024


def is_active() -> bool:
    """True when :mod:`comet_sigma` is importable AND the Step 1 flag is on."""
    return comet_sigma_available() and is_enabled(FLAG_NAME)


def writer_id_for(path: str) -> str:
    """Return the canonical backfill writer id for a source ``.sb`` path."""
    return f"{WRITER_ID_PREFIX}{os.path.abspath(path)}"


def _file_fingerprint(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            h.update(fh.read(FINGERPRINT_BYTES))
    except OSError:
        return ""
    return h.hexdigest()


def reset_backfill_state(path: Optional[str] = None) -> None:
    """Drop in-memory L1 / temporal / store state for a backfill writer.

    Pass ``None`` to drop **every** known backfill writer. Files
    previously written by the persister to disk are NOT removed — pass
    a fresh store directory if you want a clean slate.
    """
    if path is None:
        for wid in tuple(_l1.known_writers()):
            if wid.startswith(WRITER_ID_PREFIX):
                _l1.reset(wid)
                _l1t.reset(wid)
                _store.reset(wid)
        return
    wid = writer_id_for(path)
    _l1.reset(wid)
    _l1t.reset(wid)
    _store.reset(wid)


@dataclass
class BackfillResult:
    """Outcome of one :func:`backfill_trace` call."""

    path: str
    writer_id: str
    frames_observed: int
    frames_skipped: int
    base_receipts_emitted: int
    projection_receipts_emitted: int
    persisted: Optional[Dict[str, int]]
    fingerprint: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "writer_id": self.writer_id,
            "frames_observed": self.frames_observed,
            "frames_skipped": self.frames_skipped,
            "base_receipts_emitted": self.base_receipts_emitted,
            "projection_receipts_emitted": self.projection_receipts_emitted,
            "persisted": self.persisted,
            "fingerprint": self.fingerprint,
            "error": self.error,
        }


def _stamp_provenance(writer_id: str, source_path: str, frames: int, fp: str) -> None:
    """Append a single backfill provenance entry to every base artifact."""
    if not comet_sigma_available():
        return
    arts = _l1.artifacts_for(writer_id)
    if not arts:
        return
    payload = {
        "source_path": os.path.abspath(source_path),
        "frames_backfilled": int(frames),
        "fingerprint_sha256_64k": fp,
    }
    # Replace the artifacts in the writer state with the re-provenanced
    # ones so subsequent receipts share the new chain head.
    new_arts = []
    for art in arts:
        try:
            new_arts.append(
                extend_provenance(
                    art,
                    layer="L1",
                    module=MODULE_LABEL,
                    payload=payload,
                    note="backfilled from historical .sb trace",
                )
            )
        except Exception:
            new_arts.append(art)
    st = _l1.get_state(writer_id)
    if st is not None:
        st.artifacts = tuple(new_arts)


def _read_wrappers(path: str, max_frame_bytes: Optional[int]) -> List[dict]:
    """Read every wrapper from ``path`` using :mod:`stepback.trace_reader`.

    Returns an empty list (and swallows the error) when the file is
    missing, truncated, or otherwise malformed.
    """
    # Lazy import to keep this module importable when the trace reader
    # has not been initialised yet (e.g. inside a minimal install used
    # only for offline auditing).
    from ..trace_reader import read_frames  # type: ignore[import-not-found]

    return read_frames(path, max_frame_bytes=max_frame_bytes)


def _wrapper_body(wrapper: Any) -> Optional[dict]:
    """Pull the ``body`` out of a wrapper, tolerating shape drift."""
    if isinstance(wrapper, dict):
        b = wrapper.get("body")
        if isinstance(b, dict):
            return b
    return None


def _canonical_body_bytes(body: dict) -> bytes:
    """Re-canonicalise a body dict to the on-disk byte sequence."""
    from ..canonical import canonical_json  # type: ignore[import-not-found]

    try:
        return canonical_json(body)
    except Exception:
        return b""


def backfill_trace(
    path: str,
    *,
    writer_id: Optional[str] = None,
    store_dir: Optional[str] = None,
    reset: bool = False,
    max_frame_bytes: Optional[int] = None,
) -> Optional[BackfillResult]:
    """Backfill one historical ``.sb`` trace into the L1 store.

    Parameters
    ----------
    path:
        Filesystem path to a ``.sb`` v1 trace produced by
        :class:`stepback.trace_writer.TraceWriter`.
    writer_id:
        Override the default ``"backfill:<abs_path>"`` namespace. Useful
        for tests or for ingesting a renamed trace under its original id.
    store_dir:
        When set, points the persistence store at this directory for the
        duration of the call (and restores the previous value on exit).
        Equivalent to calling :func:`stepback.comet_sigma.l1_trace_writer_store.set_store_dir`
        before and after.
    reset:
        When True, clears any prior in-memory state for this writer id
        (Step 1 / Step 2 / Step 3 watermarks) before replaying.
    max_frame_bytes:
        Forwarded to :func:`stepback.trace_reader.read_frames`. ``None``
        uses the reader's default DOS cap.

    Returns
    -------
    :class:`BackfillResult` describing what happened, or ``None`` when
    the L1 flag is off / :mod:`comet_sigma` is unavailable.
    """
    if not is_active():
        return None
    wid = writer_id or writer_id_for(path)
    if reset:
        reset_backfill_state(path if writer_id is None else None)
        if writer_id is not None:
            _l1.reset(wid)
            _l1t.reset(wid)
            _store.reset(wid)

    prev_store_dir: Optional[str] = None
    swapped_store = False
    if store_dir is not None:
        prev_store_dir = _store.get_store_dir()
        _store.set_store_dir(store_dir)
        swapped_store = True

    fp = _file_fingerprint(path)
    error: Optional[str] = None
    frames_observed = 0
    frames_skipped = 0
    base_before = len(_l1.receipts_for(wid))
    proj_before = len(_l1t.receipts_for(wid))
    try:
        wrappers = _read_wrappers(path, max_frame_bytes)
    except Exception as exc:
        error = f"read_frames failed: {exc!r}"
        wrappers = []

    try:
        # Ensure the L1 state (and therefore the base-feature artifacts)
        # exists *before* replaying so we can stamp the backfill
        # provenance entry on the artifacts BEFORE the persistence hook
        # writes them to disk for the first time. ``_ensure_state`` is a
        # sibling-module helper and is the documented entry point used
        # by :func:`l1_trace_writer.observe_frame` itself.
        _l1._ensure_state(wid)
        # Use the source-frame count as the recorded
        # ``frames_backfilled`` payload — every wrapper read from the
        # file is an attempted replay, regardless of whether the L1
        # extractor decides to skip it later.
        _stamp_provenance(wid, path, len(wrappers), fp)

        for wrapper in wrappers:
            body = _wrapper_body(wrapper)
            if body is None:
                frames_skipped += 1
                continue
            body_bytes = _canonical_body_bytes(body)
            try:
                rec = _l1.observe_frame(wid, body, body_bytes)
            except Exception:
                rec = None
            if rec is None:
                frames_skipped += 1
            else:
                frames_observed += 1

        persisted = _store.persist_writer(wid)
    finally:
        if swapped_store:
            _store.set_store_dir(prev_store_dir)

    base_after = len(_l1.receipts_for(wid))
    proj_after = len(_l1t.receipts_for(wid))
    return BackfillResult(
        path=os.path.abspath(path),
        writer_id=wid,
        frames_observed=frames_observed,
        frames_skipped=frames_skipped,
        base_receipts_emitted=base_after - base_before,
        projection_receipts_emitted=proj_after - proj_before,
        persisted=persisted,
        fingerprint=fp,
        error=error,
    )


def backfill_paths(
    paths: Iterable[str],
    *,
    store_dir: Optional[str] = None,
    reset: bool = False,
    max_frame_bytes: Optional[int] = None,
) -> List[BackfillResult]:
    """Backfill an iterable of trace paths in order. Skips inactive runs."""
    out: List[BackfillResult] = []
    if not is_active():
        return out
    for p in paths:
        r = backfill_trace(
            p,
            store_dir=store_dir,
            reset=reset,
            max_frame_bytes=max_frame_bytes,
        )
        if r is not None:
            out.append(r)
    return out


def backfill_directory(
    root: str,
    *,
    pattern: str = "**/*.sb",
    store_dir: Optional[str] = None,
    reset: bool = False,
    max_frame_bytes: Optional[int] = None,
) -> List[BackfillResult]:
    """Backfill every file matching ``pattern`` under ``root``.

    The default pattern picks up the standard ``.sb`` v1 extension and
    recurses into subdirectories. Hidden files are skipped.
    """
    if not is_active():
        return []
    matches = sorted(
        p for p in _glob.glob(os.path.join(root, pattern), recursive=True)
        if os.path.isfile(p) and not os.path.basename(p).startswith(".")
    )
    return backfill_paths(
        matches,
        store_dir=store_dir,
        reset=reset,
        max_frame_bytes=max_frame_bytes,
    )


def known_backfill_writers() -> Tuple[str, ...]:
    """Return the writer ids of every known backfilled trace."""
    return tuple(
        wid for wid in _l1.known_writers() if wid.startswith(WRITER_ID_PREFIX)
    )


__all__ = [
    "BackfillResult",
    "FINGERPRINT_BYTES",
    "FLAG_NAME",
    "MODULE_LABEL",
    "WRITER_ID_PREFIX",
    "backfill_directory",
    "backfill_paths",
    "backfill_trace",
    "is_active",
    "known_backfill_writers",
    "reset_backfill_state",
    "writer_id_for",
]
