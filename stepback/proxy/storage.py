"""Shared backing state for the stepback proxy.

A :class:`ProxyState` owns:

* a write directory under which every newly-opened trace lands;
* a registry of currently-open traces keyed by ``trace_id``; and
* a process-wide lock so concurrent ``RecordStep`` calls on different
  traces are safe and concurrent calls on the *same* trace are serialised
  per-trace (``TraceWriter`` is not internally thread-safe).

Both the HTTP server (:mod:`stepback.proxy.server`) and the optional gRPC
server (:mod:`stepback.proxy.grpc_server`) share a single ``ProxyState``
instance per process so the two transports interoperate.

The state never holds raw plaintext credentials longer than necessary: the
HMAC key and Ed25519 signing key are returned to the caller exactly once
(in the ``StartTrace`` reply) and are kept in the live :class:`TraceHandle`
only because the underlying :class:`stepback.trace_writer.TraceWriter` needs
them to keep signing frames. Closing or evicting a handle drops both keys.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ..recorder import RecorderKey
from ..trace_writer import TraceWriter


class ProxyError(Exception):
    """Generic proxy-level error (4xx/5xx mappable)."""


class UnknownTraceError(ProxyError):
    """Raised when a caller references a ``trace_id`` we don't know about."""


def _safe_filename(name: str) -> str:
    """Sanitize a caller-supplied filename. Strips path separators and
    collapses to a basename so a malicious client cannot escape the
    write directory or overwrite arbitrary files.
    """
    base = os.path.basename(name)
    cleaned = "".join(
        c for c in base if c.isalnum() or c in ("-", "_", ".")
    )
    if not cleaned or cleaned in (".", ".."):
        raise ProxyError(f"invalid trace filename: {name!r}")
    if not cleaned.endswith(".sb"):
        cleaned += ".sb"
    return cleaned


@dataclass
class TraceHandle:
    """An open trace tracked by the proxy."""

    trace_id: str
    path: str
    writer: TraceWriter
    key: RecorderKey
    opened_at_ns: int = field(default_factory=time.time_ns)
    step_count: int = 0
    closed: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def info(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "path": self.path,
            "opened_at_ns": self.opened_at_ns,
            "step_count": self.step_count,
            "closed": self.closed,
            "public_key_hex": self.key.signing_key.public_key().public_bytes_raw().hex(),
        }


@dataclass
class ProxyState:
    """Process-wide proxy state. Construct once, share between transports."""

    write_dir: str
    traces: Dict[str, TraceHandle] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    max_open_traces: int = 1024

    def __post_init__(self) -> None:
        os.makedirs(self.write_dir, exist_ok=True)

    # ----------------------------------------------------- StartTrace
    def start_trace(
        self,
        *,
        filename: Optional[str] = None,
        compression: bool = True,
        price_list_version: str = "2026-04-01",
        blob_threshold: Optional[int] = None,
        blob_min_reuse: Optional[int] = None,
    ) -> TraceHandle:
        with self.lock:
            if len(self.traces) >= self.max_open_traces:
                raise ProxyError(
                    f"too many open traces ({len(self.traces)} >= "
                    f"{self.max_open_traces})"
                )
            trace_id = uuid.uuid4().hex
            if filename:
                fname = _safe_filename(filename)
            else:
                fname = f"{trace_id}.sb"
            path = os.path.join(self.write_dir, fname)
            if os.path.exists(path):
                # Refuse to clobber on the proxy side; clients can pre-delete
                # if they actually want to overwrite.
                raise ProxyError(f"trace path already exists: {path}")
            key = RecorderKey.fresh()
            kwargs: dict = {
                "hmac_key": key.hmac_key,
                "signing_key": key.signing_key,
                "price_list_version": price_list_version,
                "compression": compression,
            }
            if blob_threshold is not None:
                kwargs["blob_threshold"] = blob_threshold
            if blob_min_reuse is not None:
                kwargs["blob_min_reuse"] = blob_min_reuse
            writer = TraceWriter.open(path, **kwargs)
            handle = TraceHandle(
                trace_id=trace_id, path=path, writer=writer, key=key
            )
            self.traces[trace_id] = handle
            return handle

    # ----------------------------------------------------- RecordStep
    def record_step(self, trace_id: str, step: dict) -> TraceHandle:
        handle = self._get(trace_id)
        if handle.closed:
            raise ProxyError(f"trace {trace_id} is already closed")
        if not isinstance(step, dict):
            raise ProxyError("step must be a JSON object")
        if "step_id" not in step or "step_kind" not in step:
            raise ProxyError(
                "step missing required fields: step_id, step_kind"
            )
        if "inputs_hash" not in step or "outputs_hash" not in step:
            # The proxy is the *sole* authority for chaining HMACs but
            # not the authority for canonical input/output hashing —
            # callers must compute those (or use a language binding that
            # does it for them) so the hashes match what the producing
            # SDK considered as the request/response.
            raise ProxyError(
                "step missing required content hashes: inputs_hash, outputs_hash"
            )
        with handle.lock:
            handle.writer.write_step(step)
            handle.step_count += 1
        return handle

    # ------------------------------------------------------- EndTrace
    def end_trace(self, trace_id: str) -> TraceHandle:
        handle = self._get(trace_id)
        with handle.lock:
            if not handle.closed:
                handle.writer.close()
                handle.closed = True
        with self.lock:
            self.traces.pop(trace_id, None)
        return handle

    # -------------------------------------------------------- helpers
    def _get(self, trace_id: str) -> TraceHandle:
        with self.lock:
            handle = self.traces.get(trace_id)
        if handle is None:
            raise UnknownTraceError(f"unknown trace_id: {trace_id}")
        return handle

    def get(self, trace_id: str) -> TraceHandle:
        """Public alias of ``_get`` for transports that want to introspect."""
        return self._get(trace_id)

    def list_traces(self) -> Iterable[dict]:
        with self.lock:
            return [h.info() for h in self.traces.values()]

    def close_all(self) -> None:
        """Best-effort flush+close of every open trace. Used on shutdown."""
        with self.lock:
            handles = list(self.traces.values())
            self.traces.clear()
        for h in handles:
            try:
                if not h.closed:
                    h.writer.close()
                    h.closed = True
            except Exception:
                pass
