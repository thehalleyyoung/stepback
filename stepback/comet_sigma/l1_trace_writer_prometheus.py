"""L1 ``temporal_basis`` Prometheus exporter for ``stepback.trace_writer``.

This module owns **Step 9** of ``COMET_SIGMA_1000.md``: every Comet-Σ
**L1** base feature emitted by
:mod:`stepback.comet_sigma.l1_trace_writer` for ``stepback.trace_writer``
+ ``.sb`` format v1 frames is exposed as a Prometheus gauge labelled by
``writer_id``.

The exporter installs a single hook into
:data:`stepback.comet_sigma.l1_trace_writer.OBSERVE_HOOKS` (alongside
the Step 2 temporal projector) which, for each observed frame, calls
``Gauge.labels(writer_id=...).set(value)`` on the gauge associated with
each base feature.  The wiring itself is captured as a
:class:`comet_sigma.audit.AuditableArtifact`; every gauge update emits
one :class:`comet_sigma.audit.Receipt` recording the writer id, frame
index, feature name, and gauge value, so an external auditor can pin a
particular ``/metrics`` line back to a specific frame in a specific
trace.

Design notes
------------

* **Flag-gated.**  The exporter shares the ``COMET_SIGMA_L1_TEMPORAL``
  flag with Steps 1 and 2 (see :mod:`comet_sigma.flags`).  When the
  flag is OFF :func:`is_active` returns ``False`` and the auto-installed
  hook short-circuits to a no-op — no allocations, no gauge updates,
  no receipts.
* **Soft dependency.**  ``prometheus_client`` is an *optional* runtime
  dependency.  When it is not importable the exporter falls back to a
  tiny in-process gauge implementation that supports the same
  ``labels(...).set(...)`` surface plus a deterministic
  :func:`render_text` exposition format.  This keeps the integration
  testable on hosts that do not ship the upstream library.
* **Idempotent install.**  The hook is registered exactly once even
  when the module is imported repeatedly; :func:`uninstall_hook`
  removes it.  Resetting the exporter (:func:`reset`) clears the
  gauge values and the receipt list but leaves the artifact in place
  (the artifact is the *schema*; values are receipts).
* **Single source of truth.**  Gauge names are derived from the
  Step 1 :data:`stepback.comet_sigma.l1_trace_writer.BASE_FEATURES`
  list — there is no parallel table to drift.

Public surface
--------------

* :data:`GAUGES` — mapping of base-feature name → gauge instance.
* :func:`render_text` — produce a Prometheus text-exposition snapshot
  of every L1 trace-writer gauge.  Uses
  ``prometheus_client.generate_latest`` when available and a
  deterministic fallback otherwise.
* :func:`install_hook` / :func:`uninstall_hook` — manage the
  observe-hook registration.
* :func:`reset` — clear gauge values and receipts (used by tests).

The module is intentionally read-only with respect to the trace
writer; it never mutates frame bodies, never blocks the writer's hot
path, and swallows every exception inside the hook so a misbehaving
gauge can never break the writer.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import comet_sigma_available, is_enabled, new_artifact
from . import l1_trace_writer as _l1


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Flag that gates the exporter.  Shared with Steps 1 and 2.
FLAG_NAME: str = _l1.FLAG_NAME

#: Stable receipt schema id for every gauge update.
RECEIPT_SCHEMA_ID: str = "trace_writer_l1_prometheus_v1"

#: Module label recorded on every provenance entry.
MODULE_LABEL: str = "stepback.trace_writer.sb_v1.prometheus"

#: Prefix every gauge name receives — keeps the L1 trace-writer
#: feature space distinct from any other Comet-Σ Prometheus surface.
GAUGE_NAME_PREFIX: str = "comet_sigma_l1_trace_writer_sb_v1"

#: Single label every L1 trace-writer gauge carries.
GAUGE_LABEL: str = "writer_id"


def _gauge_metric_name(feature_name: str) -> str:
    """Return the Prometheus metric name for one base feature.

    The base feature name is ``trace_writer_sb_v1.<short>``; the gauge
    name is ``comet_sigma_l1_trace_writer_sb_v1_<short>``.  Dots are
    replaced with underscores so the result is a valid Prometheus
    metric identifier (``[a-zA-Z_:][a-zA-Z0-9_:]*``).
    """
    short = feature_name.split(".", 1)[-1]
    return f"{GAUGE_NAME_PREFIX}_{short}"


# ---------------------------------------------------------------------------
# Soft dependency on prometheus_client
# ---------------------------------------------------------------------------

try:  # pragma: no cover - exercised by host-dependent paths
    from prometheus_client import (  # type: ignore[import-not-found]
        CollectorRegistry,
        Gauge as _PromGauge,
        generate_latest as _prom_generate_latest,
    )

    _PROM_AVAILABLE = True
except Exception:  # pragma: no cover
    CollectorRegistry = None  # type: ignore[assignment,misc]
    _PromGauge = None  # type: ignore[assignment,misc]
    _prom_generate_latest = None  # type: ignore[assignment,misc]
    _PROM_AVAILABLE = False


def prometheus_client_available() -> bool:
    """Return True iff the upstream ``prometheus_client`` is importable."""
    return _PROM_AVAILABLE


# ---------------------------------------------------------------------------
# Fallback gauge implementation
# ---------------------------------------------------------------------------

class _FallbackLabeledGauge:
    """A tiny labelled-gauge proxy for the ``labels(...).set(...)`` call.

    Only used when :data:`_PROM_AVAILABLE` is False; mirrors the subset
    of the prometheus_client API the exporter actually uses.
    """

    __slots__ = ("_parent", "_label_value")

    def __init__(self, parent: "_FallbackGauge", label_value: str) -> None:
        self._parent = parent
        self._label_value = label_value

    def set(self, value: float) -> None:
        self._parent._set(self._label_value, float(value))

    def get(self) -> float:
        return self._parent._get(self._label_value)


class _FallbackGauge:
    """Minimal stand-in for ``prometheus_client.Gauge``.

    Supports the surface the exporter relies on: ``.labels(writer_id)``
    returning an object with ``.set(value)`` and ``.get()``, plus
    iteration over (label_value, value) pairs for the text exposition
    format.
    """

    def __init__(self, name: str, documentation: str, labelnames: Tuple[str, ...]) -> None:
        if labelnames != (GAUGE_LABEL,):
            raise ValueError(
                f"fallback gauge only supports labelnames=({GAUGE_LABEL!r},), "
                f"got {labelnames!r}"
            )
        self.name = name
        self.documentation = documentation
        self.labelnames = labelnames
        self._values: Dict[str, float] = {}
        self._lock = threading.Lock()

    def labels(self, **kwargs: str) -> _FallbackLabeledGauge:
        if set(kwargs) != {GAUGE_LABEL}:
            raise ValueError(f"unexpected labels: {kwargs!r}")
        return _FallbackLabeledGauge(self, str(kwargs[GAUGE_LABEL]))

    def _set(self, label_value: str, value: float) -> None:
        with self._lock:
            self._values[label_value] = value

    def _get(self, label_value: str) -> float:
        with self._lock:
            return self._values.get(label_value, 0.0)

    def items(self) -> Tuple[Tuple[str, float], ...]:
        with self._lock:
            return tuple(sorted(self._values.items()))

    def clear(self) -> None:
        with self._lock:
            self._values.clear()


# ---------------------------------------------------------------------------
# Gauge registry (one Gauge per base feature)
# ---------------------------------------------------------------------------

# A dedicated CollectorRegistry keeps the L1 trace-writer gauges out of
# the global default registry; that makes the test surface deterministic
# (no leaks between tests) and lets callers compose multiple Comet-Σ
# Prometheus surfaces side-by-side without name collisions.
_REGISTRY_LOCK = threading.Lock()


def _build_registry_and_gauges() -> Tuple[Any, Dict[str, Any]]:
    if _PROM_AVAILABLE:
        registry = CollectorRegistry()  # type: ignore[misc]
    else:
        registry = None
    gauges: Dict[str, Any] = {}
    for spec in _l1.BASE_FEATURES:
        metric_name = _gauge_metric_name(spec.name)
        documentation = (
            f"Comet-Σ L1 trace_writer.sb v1 base feature {spec.name!r}: "
            f"{spec.doc}"
        )
        if _PROM_AVAILABLE:
            gauge = _PromGauge(  # type: ignore[misc]
                metric_name,
                documentation,
                (GAUGE_LABEL,),
                registry=registry,
            )
        else:
            gauge = _FallbackGauge(metric_name, documentation, (GAUGE_LABEL,))
        gauges[spec.name] = gauge
    return registry, gauges


_REGISTRY: Any = None
_GAUGES: Dict[str, Any] = {}


def _ensure_initialised() -> None:
    global _REGISTRY, _GAUGES
    with _REGISTRY_LOCK:
        if not _GAUGES:
            _REGISTRY, _GAUGES = _build_registry_and_gauges()


_ensure_initialised()


def gauges() -> Dict[str, Any]:
    """Return the (live) mapping of base-feature name → gauge instance."""
    _ensure_initialised()
    return dict(_GAUGES)


#: Public alias: ``GAUGES[base_feature_name] -> Gauge``.
GAUGES: Dict[str, Any] = _GAUGES


def gauge_for(feature_name: str) -> Any:
    """Return the gauge associated with ``feature_name``.

    Raises :class:`KeyError` if ``feature_name`` is not a known L1
    trace-writer base feature.
    """
    _ensure_initialised()
    return _GAUGES[feature_name]


def registry() -> Any:
    """Return the underlying ``CollectorRegistry`` (or ``None``).

    ``None`` is returned when the soft ``prometheus_client`` dependency
    is unavailable; in that case use :func:`render_text` for exposition.
    """
    _ensure_initialised()
    return _REGISTRY


# ---------------------------------------------------------------------------
# AuditableArtifact for the wiring
# ---------------------------------------------------------------------------

_HOOK_SRC = (
    "def prometheus_hook(writer_id, record):\n"
    "    for feature_name, value in record.values.items():\n"
    "        gauge = GAUGES[feature_name]\n"
    "        gauge.labels(writer_id=writer_id).set(value)\n"
    "        artifact.emit('trace_writer_l1_prometheus_v1', {\n"
    "            'writer_id': writer_id,\n"
    "            'frame_index': record.frame_index,\n"
    "            'feature': feature_name,\n"
    "            'value': value,\n"
    "        })\n"
)


@dataclass
class _ExporterState:
    """Holds the wiring artifact and the rolling receipt list."""

    artifact: Any = None
    receipts: List[Any] = field(default_factory=list)


_STATE = _ExporterState()


def _ensure_artifact() -> Any:
    if not comet_sigma_available():
        return None
    if _STATE.artifact is None:
        _STATE.artifact = new_artifact(
            kind="feature",
            name="trace_writer_sb_v1.prometheus_export",
            src=_HOOK_SRC,
            doc=(
                "Per-frame Prometheus gauge update for every Comet-Σ L1 "
                "trace_writer.sb v1 base feature. One gauge per "
                f"{len(_l1.BASE_FEATURES)} base features, labelled by writer_id."
            ),
            receipt_schema={
                "id": RECEIPT_SCHEMA_ID,
                "fields": {
                    "writer_id": "str",
                    "frame_index": "int",
                    "feature": "str",
                    "value": "float",
                },
            },
            layer="L1",
            module=MODULE_LABEL,
            note=(
                f"prometheus_client_available={prometheus_client_available()}, "
                f"gauge_prefix={GAUGE_NAME_PREFIX!r}"
            ),
        )
    return _STATE.artifact


def artifact() -> Any:
    """Return the :class:`AuditableArtifact` describing the wiring (or ``None``)."""
    return _ensure_artifact()


def receipts() -> List[Any]:
    """Return a list copy of every gauge-update receipt emitted so far."""
    return list(_STATE.receipts)


# ---------------------------------------------------------------------------
# Hot path
# ---------------------------------------------------------------------------

def is_active() -> bool:
    """True when the exporter is enabled AND comet_sigma is importable."""
    return comet_sigma_available() and is_enabled(FLAG_NAME)


def _hook(writer_id: str, record: _l1.FrameRecord) -> None:
    """Per-frame hook: push values into gauges + emit one Receipt each.

    Defensive: any exception raised in here is swallowed by the
    Step 1 emitter (see ``observe_frame``), but we additionally guard
    each gauge update so one bad feature value cannot starve the others.
    """
    if not is_active():
        return
    art = _ensure_artifact()
    for feature_name, value in record.values.items():
        try:
            gauge = _GAUGES[feature_name]
            gauge.labels(**{GAUGE_LABEL: writer_id}).set(float(value))
        except Exception:
            continue
        if art is None:
            continue
        try:
            receipt = art.emit(
                RECEIPT_SCHEMA_ID,
                {
                    "writer_id": writer_id,
                    "frame_index": record.frame_index,
                    "feature": feature_name,
                    "value": float(value),
                },
            )
            _STATE.receipts.append(receipt)
        except Exception:
            pass


def install_hook() -> bool:
    """Register :func:`_hook` as a per-frame :data:`OBSERVE_HOOKS` hook.

    Idempotent — repeated calls add the hook at most once.  Returns
    ``True`` when the hook was newly installed.
    """
    if _hook in _l1.OBSERVE_HOOKS:
        return False
    _l1.OBSERVE_HOOKS.append(_hook)
    return True


def uninstall_hook() -> bool:
    """Remove the auto-export hook.  Returns ``True`` when removed."""
    try:
        _l1.OBSERVE_HOOKS.remove(_hook)
        return True
    except ValueError:
        return False


def reset() -> None:
    """Clear gauge values + receipts and rebuild the gauge registry.

    Leaves the auto-installed observe hook in place; intended for use
    by tests so each test starts from a known-empty exposition.
    """
    global _REGISTRY, _GAUGES, GAUGES
    with _REGISTRY_LOCK:
        _REGISTRY, fresh = _build_registry_and_gauges()
        _GAUGES.clear()
        _GAUGES.update(fresh)
        GAUGES = _GAUGES
    _STATE.receipts.clear()
    _STATE.artifact = None


# ---------------------------------------------------------------------------
# Text exposition
# ---------------------------------------------------------------------------

def render_text() -> str:
    """Render every L1 trace-writer gauge in Prometheus exposition format.

    When ``prometheus_client`` is importable this delegates to its
    canonical :func:`prometheus_client.generate_latest`.  Otherwise it
    walks the in-process fallback gauges and produces a deterministic
    text-format snapshot (``# HELP`` / ``# TYPE`` / one
    ``metric{writer_id="…"} value`` line per (gauge, label_value)
    pair, sorted by label value within each metric, sorted by metric
    name overall).
    """
    _ensure_initialised()
    if _PROM_AVAILABLE and _REGISTRY is not None:
        raw = _prom_generate_latest(_REGISTRY)  # type: ignore[misc]
        return raw.decode("utf-8")
    parts: List[str] = []
    for feature_name in sorted(_GAUGES.keys()):
        gauge = _GAUGES[feature_name]
        parts.append(f"# HELP {gauge.name} {gauge.documentation}")
        parts.append(f"# TYPE {gauge.name} gauge")
        for label_value, value in gauge.items():
            escaped = (
                label_value.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
            )
            parts.append(f'{gauge.name}{{{GAUGE_LABEL}="{escaped}"}} {value}')
    return "\n".join(parts) + ("\n" if parts else "")


# ---------------------------------------------------------------------------
# Auto-installation
# ---------------------------------------------------------------------------

# Install the hook at import time so the exporter is "on by default"
# once the module is imported (the entire layer remains gated by
# COMET_SIGMA_L1_TEMPORAL via :func:`is_active`).
install_hook()


__all__ = [
    "FLAG_NAME",
    "GAUGE_LABEL",
    "GAUGE_NAME_PREFIX",
    "GAUGES",
    "MODULE_LABEL",
    "RECEIPT_SCHEMA_ID",
    "artifact",
    "gauge_for",
    "gauges",
    "install_hook",
    "is_active",
    "prometheus_client_available",
    "receipts",
    "registry",
    "render_text",
    "reset",
    "uninstall_hook",
]
