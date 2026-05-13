"""L1 ``temporal_basis`` projector for ``stepback.trace_writer``.

This module owns **Step 2** of ``COMET_SIGMA_1000.md``: it projects the
per-frame base features emitted by
:mod:`stepback.comet_sigma.l1_trace_writer` (Step 1) onto a fixed set
of **temporal-basis aggregates** computed over four canonical
**wall-clock windows** — ``1s``, ``10s``, ``1m`` and ``10m`` — exactly
the windows called out in the COMET_SIGMA_1000.md text.

Each ``(base_feature × window × aggregate)`` combination is a
first-class :class:`comet_sigma.audit.AuditableArtifact` with its own
provenance entry; every call to :func:`project` emits one
:class:`comet_sigma.audit.Receipt` per artifact recording the value
plus the window's frame count and the per-writer frame index that
triggered the projection. The seven aggregates mirror the upstream
``comet_sigma.l1.temporal_basis`` library:

``mean``, ``slope``, ``ewma`` (α = 0.5), ``range``, ``std``,
``last_minus_first``, ``last_minus_mean_prev``.

Design notes
------------

* **Flag-gated.** The projector shares the
  ``COMET_SIGMA_L1_TEMPORAL`` flag with the Step 1 emitter; if the flag
  is OFF :func:`project` is a no-op and the auto-projection hook never
  runs.
* **Driven off Step 1.** The projector reads its raw inputs from the
  Step 1 ring buffer (:func:`stepback.comet_sigma.l1_trace_writer.feature_history`)
  and only the wall-clock timestamps already recorded there — it never
  re-reads the on-disk trace and adds zero allocations to
  :meth:`stepback.trace_writer.TraceWriter._write_frame`'s body
  encoding path.
* **Auto-projection.** Importing this module installs a single hook in
  :data:`stepback.comet_sigma.l1_trace_writer.OBSERVE_HOOKS`. Every
  observed frame triggers one :func:`project` call so the latest
  ``(window, aggregate)`` values are always one observation behind the
  newest frame. The hook is idempotent — a second import will not add
  it twice.
* **Receipts replay.** Receipts emitted here are valid against their
  declaring artifact (``comet_sigma.audit.replay_receipt`` returns
  True), so an external auditor can pin both the literal aggregate
  source code (``src_sha256``) and the per-frame value.
"""
from __future__ import annotations

import math
import statistics
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import comet_sigma_available, is_enabled, new_artifact
from . import l1_trace_writer as _l1


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Reuses the Step 1 flag — there is no separate kill-switch for the
#: projection layer; either the whole L1 temporal basis is on or it is
#: off.
FLAG_NAME: str = _l1.FLAG_NAME

#: Stable receipt schema id for every projected aggregate.
RECEIPT_SCHEMA_ID: str = "trace_writer_l1_projection_v1"

#: Module label recorded on every provenance entry.
MODULE_LABEL: str = "stepback.trace_writer.sb_v1.temporal_projection"

#: The four canonical wall-clock windows, named exactly as in
#: ``COMET_SIGMA_1000.md`` step 2 ("1s/10s/1m/10m"), expressed in
#: nanoseconds so they can be compared directly against the
#: ``wallclock_ns`` field of :class:`FrameRecord`.
WINDOWS_NS: Tuple[Tuple[str, int], ...] = (
    ("1s", 1_000_000_000),
    ("10s", 10_000_000_000),
    ("1m", 60_000_000_000),
    ("10m", 600_000_000_000),
)


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------

def _mean(x: Sequence[float]) -> float:
    return float(sum(x) / len(x)) if x else 0.0


def _slope(x: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mean_i = (n - 1) / 2.0
    mx = _mean(x)
    num = 0.0
    den = 0.0
    for i, xi in enumerate(x):
        num += (i - mean_i) * (xi - mx)
        den += (i - mean_i) ** 2
    return float(num / den) if den else 0.0


def _ewma(x: Sequence[float], alpha: float = 0.5) -> float:
    if not x:
        return 0.0
    e = float(x[0])
    for v in x[1:]:
        e = alpha * float(v) + (1.0 - alpha) * e
    return e


def _last_minus_first(x: Sequence[float]) -> float:
    return float(x[-1] - x[0]) if len(x) >= 2 else 0.0


def _range(x: Sequence[float]) -> float:
    return float(max(x) - min(x)) if x else 0.0


def _std(x: Sequence[float]) -> float:
    if len(x) < 2:
        return 0.0
    return float(statistics.pstdev(x))


def _last_minus_mean_prev(x: Sequence[float]) -> float:
    if len(x) < 2:
        return 0.0
    prev = x[:-1]
    return float(abs(x[-1] - _mean(prev)))


@dataclass(frozen=True)
class AggregateSpec:
    """A named temporal aggregate over a sliding window of floats.

    The ``src`` is the literal Python source of the aggregate body —
    sha256'd and recorded in the artifact's :class:`ProvenanceEntry`
    so an auditor can pin which aggregate produced any receipt.
    """

    name: str
    fn: Callable[[Sequence[float]], float]
    src: str


_SLOPE_SRC = (
    "def slope(x):\n"
    "    n = len(x)\n"
    "    if n < 2: return 0.0\n"
    "    mi = (n - 1) / 2.0\n"
    "    mx = sum(x) / n\n"
    "    num = sum((i - mi) * (xi - mx) for i, xi in enumerate(x))\n"
    "    den = sum((i - mi) ** 2 for i in range(n))\n"
    "    return float(num / den) if den else 0.0\n"
)
_EWMA_SRC = (
    "def ewma(x, alpha=0.5):\n"
    "    if not x: return 0.0\n"
    "    e = float(x[0])\n"
    "    for v in x[1:]:\n"
    "        e = alpha * float(v) + (1.0 - alpha) * e\n"
    "    return e\n"
)


AGGREGATES: Tuple[AggregateSpec, ...] = (
    AggregateSpec(
        "mean", _mean,
        "def mean(x): return float(sum(x)/len(x)) if x else 0.0\n",
    ),
    AggregateSpec("slope", _slope, _SLOPE_SRC),
    AggregateSpec("ewma", _ewma, _EWMA_SRC),
    AggregateSpec(
        "range", _range,
        "def range_(x): return float(max(x)-min(x)) if x else 0.0\n",
    ),
    AggregateSpec(
        "std", _std,
        "def std(x):\n"
        "    return float(__import__('statistics').pstdev(x)) if len(x) >= 2 else 0.0\n",
    ),
    AggregateSpec(
        "last_minus_first", _last_minus_first,
        "def lmf(x): return float(x[-1]-x[0]) if len(x) >= 2 else 0.0\n",
    ),
    AggregateSpec(
        "last_minus_mean_prev", _last_minus_mean_prev,
        "def lmmp(x):\n"
        "    if len(x) < 2: return 0.0\n"
        "    p = x[:-1]\n"
        "    return float(abs(x[-1] - sum(p)/len(p)))\n",
    ),
)


# ---------------------------------------------------------------------------
# Per-writer state
# ---------------------------------------------------------------------------

@dataclass
class ProjectionRecord:
    """A single ``project()`` invocation's output for one writer."""

    wallclock_ns: int
    frame_index: int
    #: Maps ``"<feature>.<window>.<aggregate>"`` to the float value.
    values: Dict[str, float] = field(default_factory=dict)
    #: Maps ``"<window>"`` to the number of frames inside that window.
    window_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class ProjectionState:
    writer_id: str
    #: Maps ``"<feature>.<window>.<aggregate>"`` to its AuditableArtifact.
    artifacts: Dict[str, Any] = field(default_factory=dict)
    receipts: List[Any] = field(default_factory=list)
    projections: List[ProjectionRecord] = field(default_factory=list)


_REGISTRY: Dict[str, ProjectionState] = {}
_REGISTRY_LOCK = threading.Lock()


def projection_name(feature: str, window: str, aggregate: str) -> str:
    """Stable dotted name for one (feature, window, aggregate) triple."""
    return f"{feature}.{window}.{aggregate}"


def _make_artifact(feature: str, window: str, agg: AggregateSpec) -> Any:
    return new_artifact(
        kind="feature",
        name=projection_name(feature, window, agg.name),
        src=agg.src,
        doc=(
            f"Temporal-basis aggregate '{agg.name}' over a {window} "
            f"sliding wall-clock window of base feature '{feature}' "
            f"emitted by stepback.trace_writer.sb v1."
        ),
        receipt_schema={
            "id": RECEIPT_SCHEMA_ID,
            "fields": {
                "value": "float",
                "window": "str",
                "window_ns": "int",
                "n_frames_in_window": "int",
                "frame_index": "int",
            },
        },
        layer="L1",
        module=MODULE_LABEL,
        note="trace_writer.sb v1 temporal-basis projection",
    )


def _ensure_state(writer_id: str) -> ProjectionState:
    with _REGISTRY_LOCK:
        st = _REGISTRY.get(writer_id)
        if st is None:
            st = ProjectionState(writer_id=writer_id)
            if comet_sigma_available():
                arts: Dict[str, Any] = {}
                for spec in _l1.BASE_FEATURES:
                    for win_name, _ in WINDOWS_NS:
                        for agg in AGGREGATES:
                            arts[projection_name(spec.name, win_name, agg.name)] = (
                                _make_artifact(spec.name, win_name, agg)
                            )
                st.artifacts = arts
            _REGISTRY[writer_id] = st
        return st


def reset(writer_id: Optional[str] = None) -> None:
    """Drop projection state. ``None`` clears every registered writer."""
    with _REGISTRY_LOCK:
        if writer_id is None:
            _REGISTRY.clear()
        else:
            _REGISTRY.pop(writer_id, None)


def get_state(writer_id: str) -> Optional[ProjectionState]:
    """Return the :class:`ProjectionState` for ``writer_id`` or ``None``."""
    with _REGISTRY_LOCK:
        return _REGISTRY.get(writer_id)


def known_writers() -> Tuple[str, ...]:
    """Writers for which at least one projection has been emitted."""
    with _REGISTRY_LOCK:
        return tuple(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def is_active() -> bool:
    """True when both Step 1 wiring AND the L1 temporal flag are on."""
    return comet_sigma_available() and is_enabled(FLAG_NAME)


def _window_slice(
    frames: Sequence[_l1.FrameRecord], now_ns: int, window_ns: int
) -> List[_l1.FrameRecord]:
    """Return frames whose wallclock_ns falls inside ``(now-window, now]``."""
    cutoff = now_ns - window_ns
    # frames are appended in monotonic wallclock order, so we can scan
    # from the tail backwards and stop at the first stale entry.
    out: List[_l1.FrameRecord] = []
    for rec in reversed(frames):
        if rec.wallclock_ns <= cutoff:
            break
        out.append(rec)
    out.reverse()
    return out


def project(writer_id: str, now_ns: Optional[int] = None) -> Optional[ProjectionRecord]:
    """Compute every (feature × window × aggregate) value for ``writer_id``.

    Reads the Step 1 ring buffer for ``writer_id`` and, for each of the
    seven aggregates and each of the four windows, computes one value
    per base feature. Emits one :class:`Receipt` per artifact and
    returns the resulting :class:`ProjectionRecord`. Returns ``None``
    when the projector is inactive or when there are no frames yet.

    ``now_ns`` defaults to the wall-clock timestamp of the writer's
    most recent observed frame so that projections produced inside the
    auto-hook are deterministic with respect to the frames they cover.
    """
    if not is_active():
        return None
    src_state = _l1.get_state(writer_id)
    if src_state is None or not src_state.frames:
        return None
    frames = list(src_state.frames)
    if now_ns is None:
        now_ns = frames[-1].wallclock_ns

    st = _ensure_state(writer_id)

    # Pre-slice once per window for efficiency.
    sliced: Dict[str, List[_l1.FrameRecord]] = {}
    for win_name, win_ns in WINDOWS_NS:
        sliced[win_name] = _window_slice(frames, now_ns, win_ns)

    rec = ProjectionRecord(
        wallclock_ns=now_ns,
        frame_index=frames[-1].frame_index,
        window_counts={w: len(s) for w, s in sliced.items()},
    )

    for spec in _l1.BASE_FEATURES:
        for win_name, win_ns in WINDOWS_NS:
            window_frames = sliced[win_name]
            xs = [fr.values.get(spec.name, 0.0) for fr in window_frames]
            for agg in AGGREGATES:
                try:
                    val = float(agg.fn(xs))
                except Exception:
                    val = 0.0
                if math.isnan(val) or math.isinf(val):
                    val = 0.0
                key = projection_name(spec.name, win_name, agg.name)
                rec.values[key] = val
                art = st.artifacts.get(key)
                if art is None:
                    continue
                payload = {
                    "value": val,
                    "window": win_name,
                    "window_ns": win_ns,
                    "n_frames_in_window": len(window_frames),
                    "frame_index": rec.frame_index,
                }
                try:
                    receipt = art.emit(RECEIPT_SCHEMA_ID, payload)
                    st.receipts.append(receipt)
                except Exception:
                    pass

    st.projections.append(rec)
    return rec


def latest_projection(writer_id: str) -> Optional[ProjectionRecord]:
    """Return the most recent :class:`ProjectionRecord` for ``writer_id``."""
    st = get_state(writer_id)
    if st is None or not st.projections:
        return None
    return st.projections[-1]


def projection_history(writer_id: str) -> List[ProjectionRecord]:
    """Return every :class:`ProjectionRecord` produced for ``writer_id``."""
    st = get_state(writer_id)
    if st is None:
        return []
    return list(st.projections)


def receipts_for(writer_id: str) -> List[Any]:
    """Return every projection :class:`Receipt` emitted for ``writer_id``."""
    st = get_state(writer_id)
    if st is None:
        return []
    return list(st.receipts)


def artifacts_for(writer_id: str) -> Dict[str, Any]:
    """Return the (feature × window × aggregate) → artifact map."""
    st = get_state(writer_id)
    if st is None:
        return {}
    return dict(st.artifacts)


def projection_names() -> Tuple[str, ...]:
    """Every projection key this module would emit, in canonical order."""
    return tuple(
        projection_name(spec.name, win, agg.name)
        for spec in _l1.BASE_FEATURES
        for win, _ in WINDOWS_NS
        for agg in AGGREGATES
    )


# ---------------------------------------------------------------------------
# Auto-projection hook (idempotent)
# ---------------------------------------------------------------------------

def _hook(writer_id: str, record: _l1.FrameRecord) -> None:
    project(writer_id, now_ns=record.wallclock_ns)


def install_hook() -> bool:
    """Register :func:`project` as a per-frame :data:`OBSERVE_HOOKS` hook.

    Idempotent — repeated calls add the hook at most once. Returns
    ``True`` when the hook was newly installed, ``False`` when it was
    already present.
    """
    if _hook in _l1.OBSERVE_HOOKS:
        return False
    _l1.OBSERVE_HOOKS.append(_hook)
    return True


def uninstall_hook() -> bool:
    """Remove the auto-projection hook. Returns ``True`` when removed."""
    try:
        _l1.OBSERVE_HOOKS.remove(_hook)
        return True
    except ValueError:
        return False


# Install the hook at import time so the temporal projector is "on by
# default" once the module is imported (the entire layer remains gated
# by COMET_SIGMA_L1_TEMPORAL via :func:`is_active`).
install_hook()


__all__ = [
    "AGGREGATES",
    "AggregateSpec",
    "FLAG_NAME",
    "MODULE_LABEL",
    "ProjectionRecord",
    "ProjectionState",
    "RECEIPT_SCHEMA_ID",
    "WINDOWS_NS",
    "artifacts_for",
    "get_state",
    "install_hook",
    "is_active",
    "known_writers",
    "latest_projection",
    "project",
    "projection_history",
    "projection_name",
    "projection_names",
    "receipts_for",
    "reset",
    "uninstall_hook",
]
