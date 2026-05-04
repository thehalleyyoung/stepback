"""Optional gRPC transport for the stepback proxy.

The canonical IDL is :file:`stepback/proxy/proto/sbproxy.proto`. The
Python implementation here is gated on the optional ``grpcio`` package
being installed::

    pip install stepback[proxy-grpc]

Importing this module (``from stepback.proxy import grpc_server``) is
always safe; calling :func:`serve_grpc` raises a clear ``ImportError``
if ``grpcio`` is missing.

The gRPC service is implemented "by hand" against
:mod:`grpc.GenericRpcHandler` so we don't need to ship generated stubs
in the wheel — every method is a JSON-encoded unary RPC matching the
HTTP shape, with fields named after the proto.
"""
from __future__ import annotations

import json
import logging
from concurrent import futures
from typing import Any, Optional

from ..trace_reader import TraceVerificationError, verify_trace
from ..trace_writer import FORMAT_VERSION, RECORDER_VERSION
from .server import PROXY_VERSION
from .storage import ProxyError, ProxyState, UnknownTraceError

logger = logging.getLogger("stepback.proxy.grpc")

GRPC_SERVICE = "stepback.proxy.v1.StepbackProxy"


def _require_grpc() -> Any:
    try:
        import grpc  # noqa: WPS433
    except ImportError as e:
        raise ImportError(
            "stepback.proxy.grpc_server requires the optional 'grpcio' "
            "dependency. Install with: pip install stepback[proxy-grpc]"
        ) from e
    return grpc


def _start_trace(state: ProxyState, req: dict) -> dict:
    handle = state.start_trace(
        filename=req.get("filename") or None,
        compression=bool(req.get("compression", True)),
        price_list_version=str(req.get("price_list_version") or "2026-04-01"),
        blob_threshold=req.get("blob_threshold") or None,
        blob_min_reuse=req.get("blob_min_reuse") or None,
    )
    pub = handle.key.signing_key.public_key().public_bytes_raw()
    priv = handle.key.signing_key.private_bytes_raw()
    import hashlib

    return {
        "trace_id": handle.trace_id,
        "path": handle.path,
        "hmac_key_hex": handle.key.hmac_key.hex(),
        "signing_key_hex": priv.hex(),
        "public_key_hex": pub.hex(),
        "hmac_key_id": hashlib.sha256(handle.key.hmac_key).hexdigest()[:16],
        "format_version": FORMAT_VERSION,
        "recorder_version": RECORDER_VERSION,
        "proxy_version": PROXY_VERSION,
    }


def _record_step(state: ProxyState, req: dict) -> dict:
    trace_id = req["trace_id"]
    step_json = req.get("step_json")
    if isinstance(step_json, (bytes, bytearray)):
        step = json.loads(step_json.decode("utf-8"))
    elif isinstance(step_json, str):
        step = json.loads(step_json)
    elif isinstance(req.get("step"), dict):
        step = req["step"]
    else:
        raise ProxyError("RecordStep requires step_json or step")
    handle = state.record_step(trace_id, step)
    return {"trace_id": trace_id, "step_count": handle.step_count}


def _end_trace(state: ProxyState, req: dict) -> dict:
    handle = state.end_trace(req["trace_id"])
    return {
        "trace_id": handle.trace_id,
        "path": handle.path,
        "step_count": handle.step_count,
    }


def _verify_trace(state: ProxyState, req: dict) -> dict:
    hmac_key_hex = req.get("hmac_key_hex")
    if isinstance(req.get("hmac_key"), (bytes, bytearray)):
        hmac_key = bytes(req["hmac_key"])
    elif isinstance(hmac_key_hex, str):
        hmac_key = bytes.fromhex(hmac_key_hex)
    else:
        raise ProxyError("VerifyTrace requires hmac_key (bytes) or hmac_key_hex")
    if req.get("trace_id"):
        handle = state.get(req["trace_id"])
        if not handle.closed:
            raise ProxyError("cannot verify a still-open trace; EndTrace first")
        path = handle.path
    elif req.get("path"):
        path = req["path"]
    else:
        raise ProxyError("VerifyTrace requires path or trace_id")
    try:
        v = verify_trace(path, hmac_key)
    except TraceVerificationError as e:
        return {"ok": False, "error": str(e)}
    header = v.header or {}
    return {
        "ok": True,
        "step_count": len(v.steps),
        "public_key_hex": v.public_key_hex,
        "format_version": header.get("format_version"),
        "recorder_version": header.get("recorder_version"),
        "canonicalisation_version": header.get("canonicalisation_version"),
        "hmac_key_id": header.get("hmac_key_id"),
    }


_DISPATCH = {
    "StartTrace": _start_trace,
    "RecordStep": _record_step,
    "EndTrace": _end_trace,
    "VerifyTrace": _verify_trace,
}


def _make_handler(state: ProxyState):
    grpc = _require_grpc()

    def _serializer(payload: dict) -> bytes:
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    def _deserializer(raw: bytes) -> dict:
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    method_handlers = {}
    for method_name, fn in _DISPATCH.items():
        def _make(fn=fn):
            def _impl(request, context):
                try:
                    return fn(state, request)
                except UnknownTraceError as e:
                    context.abort(grpc.StatusCode.NOT_FOUND, str(e))
                except ProxyError as e:
                    context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
                except Exception as e:  # pragma: no cover - defensive
                    logger.exception("grpc handler error: %s", method_name)
                    context.abort(grpc.StatusCode.INTERNAL, str(e))
            return _impl
        method_handlers[method_name] = grpc.unary_unary_rpc_method_handler(
            _make(), request_deserializer=_deserializer,
            response_serializer=_serializer,
        )
    return grpc.method_handlers_generic_handler(GRPC_SERVICE, method_handlers)


def serve_grpc(
    state: ProxyState,
    *,
    host: str = "127.0.0.1",
    port: int = 4320,
    max_workers: int = 8,
):
    """Start a gRPC server bound to ``state``. Returns the running grpc.Server.

    Caller is responsible for ``server.stop(grace)`` on shutdown.
    """
    grpc = _require_grpc()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    server.add_generic_rpc_handlers((_make_handler(state),))
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    return server
