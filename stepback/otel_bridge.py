"""Observability bridge: forward stepback steps to an OTel warehouse.

This module provides :class:`OtelBridge`, which reads a ``.sb`` trace and
ships its steps as OpenTelemetry spans to an OTLP/HTTP or OTLP/gRPC endpoint
in real time.  It is the live-forwarding complement to
:func:`stepback.exporters.export_otel_spans` (which writes a static JSON
file).

Use cases
---------
* **Unified observability**: send stepback traces to the same Jaeger /
  Tempo / Honeycomb / Grafana Cloud backend that already collects
  infrastructure spans.
* **Replay dashboards**: re-run a trace and stream the dirty-step diff
  to an OTel collector so operators can see which steps changed.
* **CI/CD integration**: pipe benchmark traces to a time-series store
  for trending.

The bridge is intentionally *write-only* and offline-safe: if the OTLP
endpoint is unreachable, steps are queued and retried up to
*max_retries* times.  A ``DryRunCollector`` is provided for testing
without a live endpoint.

Public API
----------
.. code-block:: python

    from stepback.otel_bridge import OtelBridge, OtlpHttpExporter

    bridge = OtelBridge(
        exporter=OtlpHttpExporter("http://otel-collector:4318"),
        service_name="my-agent",
    )
    bridge.export_trace("run.sb")

.. code-block:: python

    # Dry-run: capture spans without a live endpoint
    from stepback.otel_bridge import OtelBridge, DryRunCollector

    collector = DryRunCollector()
    bridge = OtelBridge(exporter=collector, service_name="test")
    bridge.export_trace("run.sb")
    print(collector.spans)
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .canonical import hash_obj, canonical_json, sha256_hex
from .exporters import export_otel_spans


__all__ = [
    "OtelBridge",
    "OtlpHttpExporter",
    "DryRunCollector",
    "OtelBridgeExportResult",
    "OtelBridgeError",
]


class OtelBridgeError(RuntimeError):
    """Raised when the OTel bridge fails to export a trace."""

    #: Canonical error code; see :mod:`stepback.errors` for details.
    code: str = "SB800"


# ======================================================================
# Result type
# ======================================================================

@dataclass
class OtelBridgeExportResult:
    """Summary of an OtelBridge export operation."""

    span_count: int = 0
    """Number of spans shipped to the exporter."""
    batch_count: int = 0
    """Number of batches sent."""
    retry_count: int = 0
    """Total number of retried sends."""
    failed_spans: int = 0
    """Spans that could not be delivered after all retries."""
    export_duration_ms: float = 0.0
    """Wall-clock time for the entire export, in milliseconds."""

    def __repr__(self) -> str:
        return (
            f"OtelBridgeExportResult(spans={self.span_count}, "
            f"batches={self.batch_count}, failed={self.failed_spans})"
        )


# ======================================================================
# Exporter protocol + built-in implementations
# ======================================================================

class OtlpHttpExporter:
    """Ship spans to an OTLP/HTTP endpoint (``POST /v1/traces``).

    Uses only the Python standard library ``urllib.request`` — no
    ``opentelemetry-sdk`` dependency required.  The payload is the
    simplified OTel JSON envelope produced by
    :func:`~stepback.exporters.export_otel_spans`.

    Args:
        endpoint: Base URL of the OTLP/HTTP receiver, e.g.
            ``"http://localhost:4318"``.  The path ``/v1/traces`` is
            appended automatically.
        headers: Optional extra HTTP headers (e.g. authentication tokens).
        timeout_s: Socket timeout in seconds.  Default 10.
        max_retries: Maximum number of retry attempts on transient
            errors (5xx, connection reset).  Default 3.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        timeout_s: float = 10.0,
        max_retries: int = 3,
    ) -> None:
        self.endpoint = endpoint.rstrip("/") + "/v1/traces"
        self.headers: Dict[str, str] = {"Content-Type": "application/json"}
        if headers:
            self.headers.update(headers)
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def send(self, spans: List[dict]) -> int:
        """Send *spans* to the OTLP endpoint.

        Returns the HTTP status code.  Retries on 5xx and network
        errors up to ``max_retries`` times with exponential back-off.

        Raises:
            :class:`OtelBridgeError`: After exhausting retries.
        """
        import urllib.request
        import urllib.error

        payload = json.dumps({"spans": spans}).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers=self.headers,
            method="POST",
        )

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                time.sleep(0.1 * (2 ** (attempt - 1)))  # 0.1s, 0.2s, 0.4s, …
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    return resp.status
            except urllib.error.HTTPError as exc:
                if exc.code < 500:
                    raise OtelBridgeError(
                        f"OTLP/HTTP {self.endpoint}: HTTP {exc.code}"
                    ) from exc
                last_exc = exc
            except Exception as exc:  # noqa: BLE001
                last_exc = exc

        raise OtelBridgeError(
            f"OTLP/HTTP {self.endpoint}: failed after {self.max_retries} retries: "
            f"{last_exc}"
        ) from last_exc

    def __repr__(self) -> str:
        return f"OtlpHttpExporter(endpoint={self.endpoint!r})"


class DryRunCollector:
    """In-process span collector for testing and local development.

    Spans sent to this collector are appended to :attr:`spans` rather than
    being shipped to a live endpoint.  No network calls are made.

    .. code-block:: python

        collector = DryRunCollector()
        bridge = OtelBridge(exporter=collector, service_name="test")
        bridge.export_trace("run.sb")
        assert len(collector.spans) > 0
    """

    def __init__(self) -> None:
        self.spans: List[dict] = []
        """All spans received so far, in order."""

    def send(self, spans: List[dict]) -> int:
        """Append *spans* to :attr:`spans` and return 200."""
        self.spans.extend(spans)
        return 200

    def clear(self) -> None:
        """Remove all collected spans."""
        self.spans.clear()

    def __repr__(self) -> str:
        return f"DryRunCollector(span_count={len(self.spans)})"


# ======================================================================
# OtelBridge
# ======================================================================

class OtelBridge:
    """Forward stepback ``.sb`` traces to an OTel collector.

    Reads step frames from a ``.sb`` file (via
    :func:`~stepback.trace_reader.iter_steps`), converts them to OTel
    spans (reusing the logic from
    :func:`~stepback.exporters.export_otel_spans`), and ships them to
    the configured exporter in batches.

    Args:
        exporter: An exporter instance — :class:`OtlpHttpExporter` for
            a live endpoint or :class:`DryRunCollector` for testing.
        service_name: ``service.name`` resource attribute attached to
            every span.
        batch_size: Number of spans per send batch.  Default 100.
        max_retries: Maximum retries on transient errors.  Forwarded to
            :class:`OtlpHttpExporter` (ignored by
            :class:`DryRunCollector`).
    """

    def __init__(
        self,
        exporter: Any,
        *,
        service_name: str = "stepback-agent",
        batch_size: int = 100,
    ) -> None:
        if not hasattr(exporter, "send"):
            raise TypeError(
                "OtelBridge: exporter must have a send(spans) method; "
                f"got {type(exporter).__name__}"
            )
        self.exporter = exporter
        self.service_name = service_name
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export_trace(
        self,
        trace_path: str,
        *,
        trace_id: Optional[str] = None,
    ) -> OtelBridgeExportResult:
        """Read *trace_path* and ship its steps as OTel spans.

        Args:
            trace_path: Path to a ``.sb`` trace file.
            trace_id: Optional OTel trace ID (UUID4 hex string).  When
                *None*, a fresh UUID is generated per call so each
                ``export_trace`` invocation produces a distinct OTel
                trace.

        Returns:
            :class:`OtelBridgeExportResult` summary.
        """
        import tempfile, os

        start = time.time()

        # Use replay() without verification to read steps — it decodes
        # compressed frames and materialises blob references.
        from .replay import replay as _replay_trace

        trace = _replay_trace(trace_path)
        steps = list(trace.recorded_steps)

        if not steps:
            return OtelBridgeExportResult(export_duration_ms=(time.time() - start) * 1000)

        # Use a temp file to leverage the existing OTel exporter logic.
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            tmp_path = tmp.name

        try:
            export_otel_spans(
                steps,
                tmp_path,
                envelope=True,
                trace_id=trace_id or str(uuid.uuid4()),
            )
            with open(tmp_path, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        all_spans: List[dict] = doc.get("spans") or (doc if isinstance(doc, list) else [])

        # Attach resource attributes to every span.
        for span in all_spans:
            span.setdefault("resource", {})
            span["resource"].setdefault("attributes", {})
            span["resource"]["attributes"]["service.name"] = self.service_name

        result = OtelBridgeExportResult()

        for batch_start in range(0, max(1, len(all_spans)), self.batch_size):
            batch = all_spans[batch_start: batch_start + self.batch_size]
            if not batch:
                break
            self.exporter.send(batch)
            result.span_count += len(batch)
            result.batch_count += 1

        result.export_duration_ms = (time.time() - start) * 1000
        return result

    def export_steps(
        self,
        steps: List[dict],
        *,
        trace_id: Optional[str] = None,
    ) -> OtelBridgeExportResult:
        """Ship an already-loaded step list as OTel spans.

        This is the in-memory variant of :meth:`export_trace` — no file I/O
        other than a temporary file for the OTel span conversion.

        Args:
            steps: List of step dicts (e.g. from
                :func:`~stepback.trace_reader.iter_steps`).
            trace_id: Optional OTel trace ID.

        Returns:
            :class:`OtelBridgeExportResult` summary.
        """
        import tempfile, os

        if not steps:
            return OtelBridgeExportResult()

        start = time.time()
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            tmp_path = tmp.name

        try:
            export_otel_spans(
                steps,
                tmp_path,
                envelope=True,
                trace_id=trace_id or str(uuid.uuid4()),
            )
            with open(tmp_path, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        all_spans: List[dict] = doc.get("spans") or []

        for span in all_spans:
            span.setdefault("resource", {})
            span["resource"].setdefault("attributes", {})
            span["resource"]["attributes"]["service.name"] = self.service_name

        result = OtelBridgeExportResult()

        for batch_start in range(0, max(1, len(all_spans)), self.batch_size):
            batch = all_spans[batch_start: batch_start + self.batch_size]
            if not batch:
                break
            self.exporter.send(batch)
            result.span_count += len(batch)
            result.batch_count += 1

        result.export_duration_ms = (time.time() - start) * 1000
        return result

    def __repr__(self) -> str:
        return (
            f"OtelBridge(service={self.service_name!r}, "
            f"exporter={self.exporter!r})"
        )
