"""L1 ``temporal_basis`` feature emitter for ``stepback.trace_writer``.

This module wires :class:`stepback.trace_writer.TraceWriter` (the producer
of stepback's ``.sb`` format v1 frames) as a Comet-Σ **L1** feature
emitter. Every frame the writer commits to disk is observed and turned
into a small fixed-shape **base feature** vector — bytes-on-disk, frame
type, inter-frame delta-time, HMAC-chain depth and so on. The emitter
then maintains a running window of those base features and projects
**temporal-basis aggregates** (mean / slope / EWMA / range / std /
last-minus-mean-prev / last-minus-first) over four canonical windows
(``1s / 10s / 1m / 10m`` of frames, see Step 2 of
``COMET_SIGMA_1000.md`` for the temporal-basis projection itself; this
file owns Step 1, the wiring + per-feature ``AuditableArtifact``).

Design constraints
------------------

* **Flag-gated.** The emitter is a strict no-op unless the
  ``COMET_SIGMA_L1_TEMPORAL`` flag from :mod:`comet_sigma.flags` is on.
  When it is off, :func:`observe_frame` is a single dict-lookup + early
  return — no allocations on the hot ``_write_frame`` path.
* **Single source of truth.** Each base feature is declared once as an
  :class:`comet_sigma.audit.AuditableArtifact` with a stable receipt
  schema (``trace_writer_l1_v1``). Every call to
  :func:`observe_frame` produces one :class:`comet_sigma.audit.Receipt`
  per declared feature, all sharing the artifact's
  :class:`comet_sigma.audit.ProvenanceChain` head.
* **Bounded memory.** The per-writer ring buffer is capped at
  :data:`MAX_RECENT_FRAMES`; older frames are evicted FIFO.
* **No upstream dependency at import time.** If ``comet_sigma`` is
  unavailable (see :mod:`stepback.comet_sigma.__init__`), the emitter
  still imports cleanly and behaves as if the flag were off.

The frame schema follows ``spec/sbtrace-v1.md`` and the implementation
of :class:`stepback.trace_writer.TraceWriter._write_frame`. The emitter
is intentionally read-only with respect to the writer: it never mutates
``body`` or the on-disk wrapper.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from . import (
    comet_sigma_available,
    is_enabled,
    new_artifact,
)


#: Hard cap on per-writer ring-buffer size. ``10_000`` frames is enough to
#: comfortably cover a ``10m`` window at common LLM agent step rates while
#: keeping per-writer memory below a few megabytes.
MAX_RECENT_FRAMES: int = 10_000

#: Flag that gates the entire emitter. When ``False`` (the default in
#: :mod:`comet_sigma.flags`) :func:`observe_frame` is a no-op.
FLAG_NAME: str = "COMET_SIGMA_L1_TEMPORAL"

#: Stable receipt schema id used by every base-feature emission.
RECEIPT_SCHEMA_ID: str = "trace_writer_l1_v1"

#: Module label recorded on every provenance entry.
MODULE_LABEL: str = "stepback.trace_writer.sb_v1"


# ---------------------------------------------------------------------------
# Base-feature definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaseFeatureSpec:
    """A single L1 base feature derived from a TraceWriter frame.

    The ``src`` is the literal Python source of the extractor — it is
    sha256'd and embedded in every ``ProvenanceEntry`` so downstream
    auditors can pin which version of the feature produced any receipt.
    """

    name: str
    doc: str
    src: str

    def make_artifact(self):
        """Build an :class:`AuditableArtifact` for this base feature."""
        if not comet_sigma_available():  # pragma: no cover - guarded upstream
            raise RuntimeError("comet_sigma not available")
        return new_artifact(
            kind="feature",
            name=self.name,
            src=self.src,
            doc=self.doc,
            receipt_schema={
                "id": RECEIPT_SCHEMA_ID,
                "fields": {
                    "value": "float",
                    "frame_index": "int",
                    "frame_type": "str",
                },
            },
            layer="L1",
            module=MODULE_LABEL,
            note="trace_writer.sb v1 base feature",
        )


_FRAME_BYTES_SRC = (
    "def frame_bytes(body, body_bytes, ctx):\n"
    "    return float(len(body_bytes))\n"
)
_PREV_DT_NS_SRC = (
    "def prev_dt_ns(body, body_bytes, ctx):\n"
    "    last = ctx.get('last_wallclock_ns')\n"
    "    now = ctx['wallclock_ns']\n"
    "    return float(now - last) if last is not None else 0.0\n"
)
_FRAME_DEPTH_SRC = (
    "def frame_depth(body, body_bytes, ctx):\n"
    "    return float(ctx['frame_index'])\n"
)
_BLOB_FLAG_SRC = (
    "def is_blob_frame(body, body_bytes, ctx):\n"
    "    return 1.0 if body.get('type') == 'blob' else 0.0\n"
)
_STEP_FLAG_SRC = (
    "def is_step_frame(body, body_bytes, ctx):\n"
    "    return 1.0 if body.get('type') == 'step' else 0.0\n"
)
_HEADER_FLAG_SRC = (
    "def is_header_frame(body, body_bytes, ctx):\n"
    "    return 1.0 if body.get('type') == 'header' else 0.0\n"
)
_BODY_KEY_COUNT_SRC = (
    "def body_key_count(body, body_bytes, ctx):\n"
    "    return float(len(body))\n"
)


BASE_FEATURES: Tuple[BaseFeatureSpec, ...] = (
    BaseFeatureSpec(
        name="trace_writer_sb_v1.frame_bytes",
        doc="Number of canonical-JSON bytes in the frame body, before "
            "HMAC/Ed25519 wrapping.",
        src=_FRAME_BYTES_SRC,
    ),
    BaseFeatureSpec(
        name="trace_writer_sb_v1.prev_dt_ns",
        doc="Wall-clock nanoseconds since the previous observed frame on "
            "this writer (0 for the first frame).",
        src=_PREV_DT_NS_SRC,
    ),
    BaseFeatureSpec(
        name="trace_writer_sb_v1.frame_depth",
        doc="Zero-based index of this frame in the writer's HMAC chain.",
        src=_FRAME_DEPTH_SRC,
    ),
    BaseFeatureSpec(
        name="trace_writer_sb_v1.is_blob_frame",
        doc="1.0 iff the frame's body type is the interned-blob frame.",
        src=_BLOB_FLAG_SRC,
    ),
    BaseFeatureSpec(
        name="trace_writer_sb_v1.is_step_frame",
        doc="1.0 iff the frame's body type is 'step' (an LLM-call step).",
        src=_STEP_FLAG_SRC,
    ),
    BaseFeatureSpec(
        name="trace_writer_sb_v1.is_header_frame",
        doc="1.0 iff the frame's body type is the trace 'header' frame.",
        src=_HEADER_FLAG_SRC,
    ),
    BaseFeatureSpec(
        name="trace_writer_sb_v1.body_key_count",
        doc="Number of top-level keys in the frame body dict.",
        src=_BODY_KEY_COUNT_SRC,
    ),
)


# Pure-function extractors mirroring the ``src`` strings above. Keeping
# them inline (rather than ``exec``-ing the ``src``) lets type checkers
# and CPython's optimiser see the full call graph; the ``src`` is solely
# the auditable record of what each extractor computes.
def _x_frame_bytes(body: dict, body_bytes: bytes, ctx: dict) -> float:
    return float(len(body_bytes))


def _x_prev_dt_ns(body: dict, body_bytes: bytes, ctx: dict) -> float:
    last = ctx.get("last_wallclock_ns")
    now = ctx["wallclock_ns"]
    return float(now - last) if last is not None else 0.0


def _x_frame_depth(body: dict, body_bytes: bytes, ctx: dict) -> float:
    return float(ctx["frame_index"])


def _x_is_blob(body: dict, body_bytes: bytes, ctx: dict) -> float:
    return 1.0 if body.get("type") == "blob" else 0.0


def _x_is_step(body: dict, body_bytes: bytes, ctx: dict) -> float:
    return 1.0 if body.get("type") == "step" else 0.0


def _x_is_header(body: dict, body_bytes: bytes, ctx: dict) -> float:
    return 1.0 if body.get("type") == "header" else 0.0


def _x_body_key_count(body: dict, body_bytes: bytes, ctx: dict) -> float:
    return float(len(body))


_EXTRACTORS = (
    _x_frame_bytes,
    _x_prev_dt_ns,
    _x_frame_depth,
    _x_is_blob,
    _x_is_step,
    _x_is_header,
    _x_body_key_count,
)


# ---------------------------------------------------------------------------
# Per-writer state
# ---------------------------------------------------------------------------

@dataclass
class FrameRecord:
    """A single observed frame's base-feature row."""

    wallclock_ns: int
    frame_index: int
    frame_type: str
    values: Dict[str, float]


@dataclass
class WriterState:
    """Per-writer ring buffer + provenance head + cached artifacts."""

    writer_id: str
    artifacts: Tuple[Any, ...] = field(default_factory=tuple)
    receipts: List[Any] = field(default_factory=list)
    frames: Deque[FrameRecord] = field(default_factory=lambda: deque(maxlen=MAX_RECENT_FRAMES))
    last_wallclock_ns: Optional[int] = None
    frame_index: int = 0


_REGISTRY: Dict[str, WriterState] = {}
_REGISTRY_LOCK = threading.Lock()


#: Hooks invoked at the very end of :func:`observe_frame` after the new
#: :class:`FrameRecord` has been appended and per-feature receipts have
#: been emitted. Each hook receives ``(writer_id, record)`` and is
#: expected to swallow its own exceptions so it can never break the
#: writer's hot path. Used by Step 2's temporal-basis projector
#: (:mod:`stepback.comet_sigma.l1_trace_writer_temporal`) to drive
#: time-windowed aggregates off the same observation stream.
OBSERVE_HOOKS: List[Any] = []


def _ensure_state(writer_id: str) -> WriterState:
    with _REGISTRY_LOCK:
        st = _REGISTRY.get(writer_id)
        if st is None:
            st = WriterState(writer_id=writer_id)
            if comet_sigma_available():
                st.artifacts = tuple(spec.make_artifact() for spec in BASE_FEATURES)
            _REGISTRY[writer_id] = st
        return st


def reset(writer_id: Optional[str] = None) -> None:
    """Drop registered state.

    Pass a ``writer_id`` to drop a single writer; pass ``None`` to drop
    every registered writer (useful for tests).
    """
    with _REGISTRY_LOCK:
        if writer_id is None:
            _REGISTRY.clear()
        else:
            _REGISTRY.pop(writer_id, None)


def get_state(writer_id: str) -> Optional[WriterState]:
    """Return the :class:`WriterState` for ``writer_id`` or ``None``."""
    with _REGISTRY_LOCK:
        return _REGISTRY.get(writer_id)


def known_writers() -> Tuple[str, ...]:
    """Return the ids of every writer that has emitted at least one frame."""
    with _REGISTRY_LOCK:
        return tuple(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Hot path
# ---------------------------------------------------------------------------

def is_active() -> bool:
    """True when the emitter is enabled AND comet_sigma is importable."""
    return comet_sigma_available() and is_enabled(FLAG_NAME)


def observe_frame(writer_id: str, body: dict, body_bytes: bytes) -> Optional[FrameRecord]:
    """Record one TraceWriter frame as L1 base features.

    Called from :meth:`stepback.trace_writer.TraceWriter._write_frame`
    after the frame body has been canonical-JSON-encoded but before it
    is committed to disk. Returns the :class:`FrameRecord` that was
    appended, or ``None`` when the emitter is inactive.

    The function is intentionally tolerant of non-dict bodies and
    missing fields; it never raises into the writer's hot path.
    """
    if not is_active():
        return None
    try:
        st = _ensure_state(writer_id)
        wallclock_ns = time.time_ns()
        ctx = {
            "wallclock_ns": wallclock_ns,
            "last_wallclock_ns": st.last_wallclock_ns,
            "frame_index": st.frame_index,
        }
        body = body if isinstance(body, dict) else {}
        body_bytes = body_bytes if isinstance(body_bytes, (bytes, bytearray)) else b""
        frame_type = str(body.get("type", "")) if isinstance(body, dict) else ""
        values: Dict[str, float] = {}
        for spec, fn in zip(BASE_FEATURES, _EXTRACTORS):
            try:
                values[spec.name] = float(fn(body, bytes(body_bytes), ctx))
            except Exception:
                values[spec.name] = 0.0

        record = FrameRecord(
            wallclock_ns=wallclock_ns,
            frame_index=st.frame_index,
            frame_type=frame_type,
            values=values,
        )
        st.frames.append(record)
        st.last_wallclock_ns = wallclock_ns
        st.frame_index += 1

        # Emit one Receipt per base feature — the receipts share the
        # artifact's ProvenanceChain head, so an auditor replaying any
        # one of them can pin the writer + frame index it was derived
        # from.
        for art, spec in zip(st.artifacts, BASE_FEATURES):
            payload = {
                "value": values[spec.name],
                "frame_index": record.frame_index,
                "frame_type": frame_type,
            }
            try:
                receipt = art.emit(RECEIPT_SCHEMA_ID, payload)
                st.receipts.append(receipt)
            except Exception:
                # Defensive: a misconfigured schema must never break the
                # writer. The frame is still appended to ``st.frames``.
                pass
        # Step 2: invoke registered observation hooks (e.g. the
        # temporal-basis projector). Hooks must be defensive — any
        # exception they raise is swallowed here so the writer never
        # observes a side-effect from L1 instrumentation.
        for hook in tuple(OBSERVE_HOOKS):
            try:
                hook(writer_id, record)
            except Exception:
                pass
        return record
    except Exception:
        return None


def latest_feature_vector(writer_id: str) -> Optional[Dict[str, float]]:
    """Return the most recent base-feature vector for ``writer_id``."""
    st = get_state(writer_id)
    if st is None or not st.frames:
        return None
    return dict(st.frames[-1].values)


def feature_history(writer_id: str) -> List[FrameRecord]:
    """Return a list copy of the per-writer ring buffer (oldest first)."""
    st = get_state(writer_id)
    if st is None:
        return []
    return list(st.frames)


def receipts_for(writer_id: str) -> List[Any]:
    """Return a list copy of every receipt emitted for ``writer_id``."""
    st = get_state(writer_id)
    if st is None:
        return []
    return list(st.receipts)


def artifacts_for(writer_id: str) -> Tuple[Any, ...]:
    """Return the (immutable) tuple of base-feature artifacts."""
    st = get_state(writer_id)
    if st is None:
        return ()
    return st.artifacts


__all__ = [
    "BASE_FEATURES",
    "BaseFeatureSpec",
    "FRAME_NAME_PREFIX",
    "FLAG_NAME",
    "FrameRecord",
    "MAX_RECENT_FRAMES",
    "MODULE_LABEL",
    "OBSERVE_HOOKS",
    "RECEIPT_SCHEMA_ID",
    "WriterState",
    "artifacts_for",
    "feature_history",
    "get_state",
    "is_active",
    "known_writers",
    "latest_feature_vector",
    "observe_frame",
    "receipts_for",
    "reset",
]


# Public alias for the prefix used in feature names; helps downstream
# code build per-feature Prometheus metric names without re-deriving it.
FRAME_NAME_PREFIX: str = "trace_writer_sb_v1"
