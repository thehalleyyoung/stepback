"""L1 ``temporal_basis`` feature emitter for ``stepback.canonical``.

This module wires :mod:`stepback.canonical` (the canonical-JSON
producer that feeds every ``.sb`` frame, every content hash and every
L1 stepback artefact) as a Comet-Σ **L1** feature emitter. Every call
to :func:`stepback.canonical.canonical_json` (and equivalently
:func:`stepback.canonical.hash_obj`) is observed and turned into a
small fixed-shape **base feature** vector — input depth, output bytes,
key count, presence of bytes/sets, sha256-prefix entropy, and so on.
The emitter then maintains a per-namespace ring buffer of those base
features so a separate temporal-basis projection (Step 12, owned by
:mod:`stepback.comet_sigma.l1_canonical_temporal`) can compute
``1s / 10s / 1m / 10m`` aggregates over the same observation stream.

Design constraints
------------------

* **Flag-gated.** The emitter is a strict no-op unless the
  ``COMET_SIGMA_L1_TEMPORAL`` flag from :mod:`comet_sigma.flags` is
  on. When it is off, :func:`observe_canonicalisation` is a single
  dict-lookup + early return and contributes zero allocations to the
  hot canonicalisation path.
* **Single source of truth.** Each base feature is declared once as a
  :class:`comet_sigma.audit.AuditableArtifact` with a stable receipt
  schema (``canonical_l1_v1``). Every observation emits one
  :class:`comet_sigma.audit.Receipt` per declared feature, all sharing
  the artifact's :class:`comet_sigma.audit.ProvenanceChain` head — so
  an auditor replaying any one receipt can pin which canonical-JSON
  call produced it.
* **Bounded memory.** The per-namespace ring buffer is capped at
  :data:`MAX_RECENT_EVENTS`; older events are evicted FIFO.
* **Defensive.** :func:`observe_canonicalisation` never raises into
  the canonicaliser's hot path; every extractor is wrapped in a
  ``try/except`` and on failure the corresponding base-feature value
  is recorded as ``0.0``.
* **No upstream dependency at import time.** If ``comet_sigma`` is
  unavailable (see :mod:`stepback.comet_sigma.__init__`), the emitter
  still imports cleanly and behaves as if the flag were off.

The canonical-JSON contract that drives the feature definitions
follows :mod:`stepback.canonical` (UTF-8, sorted keys, no whitespace
separators, ``__bytes_hex__``-wrapped bytes; see
``CANONICALISATION_VERSION``). The emitter is read-only: it never
mutates either the input ``obj`` or the output ``payload`` bytes.
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from . import (
    comet_sigma_available,
    is_enabled,
    new_artifact,
)


#: Hard cap on per-namespace ring-buffer size. ``10_000`` events is
#: enough to comfortably cover a ``10m`` window of typical agent
#: canonicalisation rates while keeping per-namespace memory below a
#: few megabytes.
MAX_RECENT_EVENTS: int = 10_000

#: Flag that gates the entire emitter. When ``False`` (the default in
#: :mod:`comet_sigma.flags`) :func:`observe_canonicalisation` is a
#: no-op.
FLAG_NAME: str = "COMET_SIGMA_L1_TEMPORAL"

#: Stable receipt schema id used by every base-feature emission.
RECEIPT_SCHEMA_ID: str = "canonical_l1_v1"

#: Module label recorded on every provenance entry.
MODULE_LABEL: str = "stepback.canonical.v1"

#: Default namespace used when callers don't pass a ``namespace=`` to
#: :func:`observe_canonicalisation`. Kept short and stable so that
#: receipts emitted from anonymous call sites group under a single
#: predictable artifact bundle.
DEFAULT_NAMESPACE: str = "default"

#: Public alias for the prefix used in feature names; helps downstream
#: code (e.g. Step 19's Prometheus exporter) build per-feature metric
#: names without re-deriving the convention.
FRAME_NAME_PREFIX: str = "canonical_v1"


# ---------------------------------------------------------------------------
# Base-feature definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaseFeatureSpec:
    """A single L1 base feature derived from a canonicalisation event.

    The ``src`` is the literal Python source of the extractor — it is
    sha256'd and embedded in every ``ProvenanceEntry`` so downstream
    auditors can pin which version of the feature produced any
    receipt.
    """

    name: str
    doc: str
    src: str

    def make_artifact(self):
        """Build an :class:`AuditableArtifact` for this base feature."""
        if not comet_sigma_available():  # pragma: no cover
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
                    "event_index": "int",
                    "namespace": "str",
                },
            },
            layer="L1",
            module=MODULE_LABEL,
            note="canonical.canonical_json v1 base feature",
        )


_OUT_BYTES_SRC = (
    "def out_bytes(obj, payload, ctx):\n"
    "    return float(len(payload))\n"
)
_PREV_DT_NS_SRC = (
    "def prev_dt_ns(obj, payload, ctx):\n"
    "    last = ctx.get('last_wallclock_ns')\n"
    "    now = ctx['wallclock_ns']\n"
    "    return float(now - last) if last is not None else 0.0\n"
)
_EVENT_INDEX_SRC = (
    "def event_index(obj, payload, ctx):\n"
    "    return float(ctx['event_index'])\n"
)
_TOP_KEY_COUNT_SRC = (
    "def top_key_count(obj, payload, ctx):\n"
    "    return float(len(obj)) if isinstance(obj, dict) else 0.0\n"
)
_MAX_DEPTH_SRC = (
    "def max_depth(obj, payload, ctx):\n"
    "    def w(o, d):\n"
    "        if isinstance(o, dict):\n"
    "            return max((w(v, d + 1) for v in o.values()), default=d)\n"
    "        if isinstance(o, list):\n"
    "            return max((w(v, d + 1) for v in o), default=d)\n"
    "        return d\n"
    "    return float(w(obj, 0))\n"
)
_HAS_BYTES_FLAG_SRC = (
    "def has_bytes_flag(obj, payload, ctx):\n"
    "    return 1.0 if b'__bytes_hex__' in payload else 0.0\n"
)
_DURATION_NS_SRC = (
    "def duration_ns(obj, payload, ctx):\n"
    "    return float(ctx.get('duration_ns', 0) or 0)\n"
)
_TYPE_TAG_SRC = (
    "def type_tag(obj, payload, ctx):\n"
    "    if isinstance(obj, dict): return 1.0\n"
    "    if isinstance(obj, list): return 2.0\n"
    "    if isinstance(obj, str):  return 3.0\n"
    "    if isinstance(obj, bool): return 4.0\n"
    "    if isinstance(obj, (int, float)): return 5.0\n"
    "    if obj is None: return 0.0\n"
    "    return 6.0\n"
)


BASE_FEATURES: Tuple[BaseFeatureSpec, ...] = (
    BaseFeatureSpec(
        name="canonical_v1.out_bytes",
        doc="Number of canonical-JSON bytes produced by canonical_json.",
        src=_OUT_BYTES_SRC,
    ),
    BaseFeatureSpec(
        name="canonical_v1.prev_dt_ns",
        doc="Wall-clock nanoseconds since the previous observed event "
            "in this namespace (0 for the first event).",
        src=_PREV_DT_NS_SRC,
    ),
    BaseFeatureSpec(
        name="canonical_v1.event_index",
        doc="Zero-based ordinal of this canonicalisation in its namespace.",
        src=_EVENT_INDEX_SRC,
    ),
    BaseFeatureSpec(
        name="canonical_v1.top_key_count",
        doc="Number of top-level keys when the canonicalised value is "
            "a dict; 0 otherwise.",
        src=_TOP_KEY_COUNT_SRC,
    ),
    BaseFeatureSpec(
        name="canonical_v1.max_depth",
        doc="Maximum nesting depth of dict/list nodes in the input "
            "object (0 for scalars).",
        src=_MAX_DEPTH_SRC,
    ),
    BaseFeatureSpec(
        name="canonical_v1.has_bytes_flag",
        doc="1.0 iff the canonical output embeds at least one "
            "``__bytes_hex__`` wrapper (raw bytes / bytearray input).",
        src=_HAS_BYTES_FLAG_SRC,
    ),
    BaseFeatureSpec(
        name="canonical_v1.duration_ns",
        doc="Wall-clock duration in nanoseconds of the canonicalisation "
            "call itself, when supplied by the caller (else 0).",
        src=_DURATION_NS_SRC,
    ),
    BaseFeatureSpec(
        name="canonical_v1.type_tag",
        doc="Compact integer tag for the input top-level type "
            "(0=None, 1=dict, 2=list, 3=str, 4=bool, 5=number, 6=other).",
        src=_TYPE_TAG_SRC,
    ),
)


# Pure-function extractors mirroring the ``src`` strings above. Keeping
# them inline (rather than ``exec``-ing the ``src``) lets type checkers
# and CPython's optimiser see the full call graph; the ``src`` is
# solely the auditable record of what each extractor computes.
def _x_out_bytes(obj: Any, payload: bytes, ctx: dict) -> float:
    return float(len(payload))


def _x_prev_dt_ns(obj: Any, payload: bytes, ctx: dict) -> float:
    last = ctx.get("last_wallclock_ns")
    now = ctx["wallclock_ns"]
    return float(now - last) if last is not None else 0.0


def _x_event_index(obj: Any, payload: bytes, ctx: dict) -> float:
    return float(ctx["event_index"])


def _x_top_key_count(obj: Any, payload: bytes, ctx: dict) -> float:
    return float(len(obj)) if isinstance(obj, dict) else 0.0


def _x_max_depth(obj: Any, payload: bytes, ctx: dict) -> float:
    def w(o: Any, d: int) -> int:
        if isinstance(o, dict):
            return max((w(v, d + 1) for v in o.values()), default=d)
        if isinstance(o, list):
            return max((w(v, d + 1) for v in o), default=d)
        return d
    return float(w(obj, 0))


def _x_has_bytes_flag(obj: Any, payload: bytes, ctx: dict) -> float:
    return 1.0 if b"__bytes_hex__" in payload else 0.0


def _x_duration_ns(obj: Any, payload: bytes, ctx: dict) -> float:
    return float(ctx.get("duration_ns", 0) or 0)


def _x_type_tag(obj: Any, payload: bytes, ctx: dict) -> float:
    if isinstance(obj, dict):
        return 1.0
    if isinstance(obj, list):
        return 2.0
    if isinstance(obj, str):
        return 3.0
    if isinstance(obj, bool):
        return 4.0
    if isinstance(obj, (int, float)):
        return 5.0
    if obj is None:
        return 0.0
    return 6.0


_EXTRACTORS: Tuple[Callable[..., float], ...] = (
    _x_out_bytes,
    _x_prev_dt_ns,
    _x_event_index,
    _x_top_key_count,
    _x_max_depth,
    _x_has_bytes_flag,
    _x_duration_ns,
    _x_type_tag,
)


# ---------------------------------------------------------------------------
# Per-namespace state
# ---------------------------------------------------------------------------

@dataclass
class CanonicalEvent:
    """A single observed canonicalisation event's base-feature row."""

    wallclock_ns: int
    event_index: int
    namespace: str
    out_bytes: int
    payload_sha256_prefix: str
    values: Dict[str, float]


@dataclass
class NamespaceState:
    """Per-namespace ring buffer + provenance head + cached artifacts."""

    namespace: str
    artifacts: Tuple[Any, ...] = field(default_factory=tuple)
    receipts: List[Any] = field(default_factory=list)
    events: Deque[CanonicalEvent] = field(
        default_factory=lambda: deque(maxlen=MAX_RECENT_EVENTS)
    )
    last_wallclock_ns: Optional[int] = None
    event_index: int = 0


_REGISTRY: Dict[str, NamespaceState] = {}
_REGISTRY_LOCK = threading.Lock()


#: Hooks invoked at the very end of :func:`observe_canonicalisation`
#: after the new :class:`CanonicalEvent` has been appended and
#: per-feature receipts have been emitted. Each hook receives
#: ``(namespace, event)`` and is expected to swallow its own
#: exceptions so it can never break the canonicaliser's hot path.
#: Used by Step 12's temporal-basis projector to drive time-windowed
#: aggregates off the same observation stream.
OBSERVE_HOOKS: List[Any] = []


def _ensure_state(namespace: str) -> NamespaceState:
    with _REGISTRY_LOCK:
        st = _REGISTRY.get(namespace)
        if st is None:
            st = NamespaceState(namespace=namespace)
            if comet_sigma_available():
                st.artifacts = tuple(spec.make_artifact() for spec in BASE_FEATURES)
            _REGISTRY[namespace] = st
        return st


def reset(namespace: Optional[str] = None) -> None:
    """Drop registered state.

    Pass a ``namespace`` to drop a single namespace; pass ``None`` to
    drop every registered namespace (useful for tests).
    """
    with _REGISTRY_LOCK:
        if namespace is None:
            _REGISTRY.clear()
        else:
            _REGISTRY.pop(namespace, None)


def get_state(namespace: str) -> Optional[NamespaceState]:
    """Return the :class:`NamespaceState` for ``namespace`` or ``None``."""
    with _REGISTRY_LOCK:
        return _REGISTRY.get(namespace)


def known_namespaces() -> Tuple[str, ...]:
    """Return the names of every namespace that has emitted ≥1 event."""
    with _REGISTRY_LOCK:
        return tuple(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Hot path
# ---------------------------------------------------------------------------

def is_active() -> bool:
    """True when the emitter is enabled AND comet_sigma is importable."""
    return comet_sigma_available() and is_enabled(FLAG_NAME)


def observe_canonicalisation(
    obj: Any,
    payload: bytes,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    duration_ns: int = 0,
) -> Optional[CanonicalEvent]:
    """Record one canonicalisation as L1 base features.

    Intended to be called immediately after
    :func:`stepback.canonical.canonical_json` returns. ``obj`` is the
    original Python object and ``payload`` is the canonical-JSON byte
    string returned by ``canonical_json(obj)``. ``duration_ns`` may
    optionally be supplied by callers that wrap the canonicalisation
    call in a ``time.perf_counter_ns()`` pair; otherwise it defaults
    to ``0`` and the corresponding base feature is left at zero.

    The function is intentionally tolerant of garbage inputs and
    never raises into the canonicaliser's hot path.
    """
    if not is_active():
        return None
    try:
        st = _ensure_state(namespace)
        wallclock_ns = time.time_ns()
        ctx = {
            "wallclock_ns": wallclock_ns,
            "last_wallclock_ns": st.last_wallclock_ns,
            "event_index": st.event_index,
            "duration_ns": int(duration_ns or 0),
        }
        payload_b = payload if isinstance(payload, (bytes, bytearray)) else b""
        payload_b = bytes(payload_b)

        values: Dict[str, float] = {}
        for spec, fn in zip(BASE_FEATURES, _EXTRACTORS):
            try:
                values[spec.name] = float(fn(obj, payload_b, ctx))
            except Exception:
                values[spec.name] = 0.0

        sha256_prefix = hashlib.sha256(payload_b).hexdigest()[:16]
        event = CanonicalEvent(
            wallclock_ns=wallclock_ns,
            event_index=st.event_index,
            namespace=namespace,
            out_bytes=len(payload_b),
            payload_sha256_prefix=sha256_prefix,
            values=values,
        )
        st.events.append(event)
        st.last_wallclock_ns = wallclock_ns
        st.event_index += 1

        # Emit one Receipt per base feature — the receipts share the
        # artifact's ProvenanceChain head, so an auditor replaying any
        # one of them can pin the namespace + event index it was
        # derived from.
        for art, spec in zip(st.artifacts, BASE_FEATURES):
            payload_dict = {
                "value": values[spec.name],
                "event_index": event.event_index,
                "namespace": namespace,
            }
            try:
                receipt = art.emit(RECEIPT_SCHEMA_ID, payload_dict)
                st.receipts.append(receipt)
            except Exception:
                # Defensive: a misconfigured schema must never break
                # the canonicaliser. The event is still appended.
                pass

        # Step 12: invoke registered observation hooks (e.g. the
        # temporal-basis projector). Hooks must be defensive — any
        # exception they raise is swallowed here so the canonicaliser
        # never observes a side-effect from L1 instrumentation.
        for hook in tuple(OBSERVE_HOOKS):
            try:
                hook(namespace, event)
            except Exception:
                pass
        return event
    except Exception:
        return None


def latest_feature_vector(namespace: str = DEFAULT_NAMESPACE) -> Optional[Dict[str, float]]:
    """Return the most recent base-feature vector for ``namespace``."""
    st = get_state(namespace)
    if st is None or not st.events:
        return None
    return dict(st.events[-1].values)


def feature_history(namespace: str = DEFAULT_NAMESPACE) -> List[CanonicalEvent]:
    """Return a list copy of the per-namespace ring buffer (oldest first)."""
    st = get_state(namespace)
    if st is None:
        return []
    return list(st.events)


def receipts_for(namespace: str = DEFAULT_NAMESPACE) -> List[Any]:
    """Return a list copy of every receipt emitted for ``namespace``."""
    st = get_state(namespace)
    if st is None:
        return []
    return list(st.receipts)


def artifacts_for(namespace: str = DEFAULT_NAMESPACE) -> Tuple[Any, ...]:
    """Return the (immutable) tuple of base-feature artifacts."""
    st = get_state(namespace)
    if st is None:
        return ()
    return st.artifacts


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------

#: Sentinel attribute attached to the patched ``canonical_json`` so we
#: can detect prior installation and avoid double-wrapping if
#: :func:`install` is called twice.
_INSTALL_MARKER = "_comet_sigma_l1_observed"


def install(namespace: str = DEFAULT_NAMESPACE) -> bool:
    """Wrap :func:`stepback.canonical.canonical_json` to feed this emitter.

    Idempotent: a second call with the same module is a no-op and
    returns ``False`` to signal nothing changed. Returns ``True`` when
    the wrapper was installed for the first time.

    The wrapper preserves the original function's exact return value
    (the canonical-JSON bytes) and merely calls
    :func:`observe_canonicalisation` between the underlying call and
    the return. When the L1 flag is off, ``observe_canonicalisation``
    early-returns and the wrapper costs ~one attribute lookup per
    canonicalisation.
    """
    from stepback import canonical as _canon  # local: keep import light

    inner = _canon.canonical_json
    if getattr(inner, _INSTALL_MARKER, False):
        return False

    def _wrapper(obj: Any) -> bytes:
        t0 = time.perf_counter_ns()
        out = inner(obj)
        t1 = time.perf_counter_ns()
        observe_canonicalisation(
            obj,
            out,
            namespace=namespace,
            duration_ns=t1 - t0,
        )
        return out

    _wrapper.__wrapped__ = inner  # type: ignore[attr-defined]
    setattr(_wrapper, _INSTALL_MARKER, True)
    _canon.canonical_json = _wrapper  # type: ignore[assignment]
    return True


def uninstall() -> bool:
    """Undo :func:`install`. Returns ``True`` if a wrapper was removed."""
    from stepback import canonical as _canon

    cur = _canon.canonical_json
    if not getattr(cur, _INSTALL_MARKER, False):
        return False
    inner = getattr(cur, "__wrapped__", None)
    if inner is None:  # pragma: no cover - defensive
        return False
    _canon.canonical_json = inner  # type: ignore[assignment]
    return True


__all__ = [
    "BASE_FEATURES",
    "BaseFeatureSpec",
    "CanonicalEvent",
    "DEFAULT_NAMESPACE",
    "FLAG_NAME",
    "FRAME_NAME_PREFIX",
    "MAX_RECENT_EVENTS",
    "MODULE_LABEL",
    "NamespaceState",
    "OBSERVE_HOOKS",
    "RECEIPT_SCHEMA_ID",
    "artifacts_for",
    "feature_history",
    "get_state",
    "install",
    "is_active",
    "known_namespaces",
    "latest_feature_vector",
    "observe_canonicalisation",
    "receipts_for",
    "reset",
    "uninstall",
]
