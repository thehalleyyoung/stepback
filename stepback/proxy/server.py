"""HTTP transport for the stepback proxy.

Wire format
-----------

JSON-over-HTTP. Every request body and every response body is JSON.
``Content-Type: application/json`` is required on requests with bodies.

Endpoints
~~~~~~~~~

``GET  /healthz``
    Returns ``{"ok": true, "proxy_version": ..., "open_traces": N}``.

``GET  /v1/info``
    Returns proxy metadata (version, write directory, schema version).

``GET  /v1/traces``
    Returns ``{"traces": [<TraceInfo>, ...]}`` listing currently-open traces.

``POST /v1/traces`` — **StartTrace**
    Request body (all fields optional)::

        {
          "filename":            "my-trace.sb",
          "compression":         true,
          "price_list_version":  "2026-04-01",
          "blob_threshold":      200,
          "blob_min_reuse":      2
        }

    Response (HTTP 201)::

        {
          "trace_id":           "<hex uuid>",
          "path":               "/abs/path/to/file.sb",
          "hmac_key_hex":       "<64 hex chars>",
          "signing_key_hex":    "<64 hex chars, raw Ed25519 private>",
          "public_key_hex":     "<64 hex chars, raw Ed25519 public>",
          "hmac_key_id":        "<16 hex chars>",
          "format_version":     1,
          "recorder_version":   "0.1.0",
          "proxy_version":      "<proxy semver>"
        }

    The ``hmac_key_hex`` and ``signing_key_hex`` are returned exactly once,
    in plaintext, on the StartTrace response. Clients are responsible for
    stashing them somewhere they can later present to ``stepback verify``.

``POST /v1/traces/{trace_id}/steps`` — **RecordStep**
    Request body::

        {"step": { ...full step dict... }}

    The step dict must already carry ``step_id``, ``step_kind``,
    ``inputs_hash``, and ``outputs_hash`` — the proxy is the authority
    for HMAC-chaining frames but not for canonicalising the LLM/tool I/O
    that the producing SDK saw. Language bindings should reuse the same
    canonical-JSON / SHA-256 routine the Python recorder uses.

    Response::

        {"ok": true, "step_count": N, "trace_id": "..."}

``POST /v1/traces/{trace_id}/end`` — **EndTrace**
    Request body is ignored (may be empty or ``{}``).

    Response::

        {"ok": true, "trace_id": "...", "path": "...", "step_count": N}

    After EndTrace, the trace_id is forgotten by the proxy and further
    RecordStep calls return 404.

``POST /v1/verify`` — **VerifyTrace**
    Request body::

        {"path": "/abs/or/relative/to/proxy/cwd/file.sb",
         "hmac_key_hex": "<64 hex chars>"}

    Or, for already-recorded traces still tracked by *this* proxy::

        {"trace_id": "<hex uuid>", "hmac_key_hex": "<64 hex>"}

    Response on success::

        {"ok": true, "step_count": M,
         "public_key_hex": "...", "format_version": 1,
         "recorder_version": "...", "canonicalisation_version": "...",
         "hmac_key_id": "..."}

    Response on failure (HTTP 400 with body)::

        {"ok": false, "error": "<TraceVerificationError message>"}

``POST /v1/replay`` — **ReplayTrace**
    Submit a closed trace and optional substitutions; receive replay events
    as newline-delimited JSON (NDJSON) streamed in the response body.

    Request body::

        {
          "path":             "/abs/path/to/file.sb",
          "hmac_key_hex":     "<64 hex chars>",
          "substitutions":    [
            {"step_id": "step:3", "kind": "tool_output", "value": {"result": 42}}
          ],
          "fallback_recorded": true
        }

    ``hmac_key_hex`` is optional; if omitted the trace is loaded without
    HMAC verification (useful for inspection).  ``substitutions`` defaults
    to ``[]``.  ``fallback_recorded`` defaults to ``true``.

    Response — one JSON object per line, ``Content-Type:
    application/x-ndjson``::

        {"event": "step_complete", "step_id": "...", "step_kind": "...",
         "dirty": false, "cache_hit": true, "cost_usd": 0.0,
         "current_inputs_hash": "...", "recorded_inputs_hash": "...",
         "output_changed": false}
        ...
        {"event": "replay_done", "step_count": N, "dirty_count": N,
         "cache_hit_count": N, "total_cost_usd": 0.0, "real_executions": 0}

    On error a single line ``{"event": "error", "error": "..."}`` is
    written and the connection closes.

Error model
~~~~~~~~~~~

Errors return non-2xx with body::

    {"error": "<class>", "message": "<human text>"}

Common codes:

* ``400`` — malformed JSON, missing required field, bad hex.
* ``404`` — unknown ``trace_id`` or unknown route.
* ``409`` — trace already closed / clobber refused.
* ``413`` — request body exceeds ``--max-body-bytes``.
* ``500`` — unhandled internal error.
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

from ..trace_reader import verify_trace, TraceVerificationError
from ..trace_writer import FORMAT_VERSION, RECORDER_VERSION
from ..replay import replay as load_trace, replay_events, Executor
from ..substitutions import SubstitutionSet, ToolOutputSubstitution
from .storage import ProxyError, ProxyState, TraceHandle, UnknownTraceError

PROXY_VERSION = "0.1.0"
HTTP_SCHEMA = "stepback-proxy/1"

DEFAULT_MAX_BODY_BYTES = 64 * 1024 * 1024  # 64 MiB hard cap per request

logger = logging.getLogger("stepback.proxy")


def _start_trace_response(handle: TraceHandle) -> dict:
    pub_hex = handle.key.signing_key.public_key().public_bytes_raw().hex()
    priv_bytes = handle.key.signing_key.private_bytes_raw()
    import hashlib

    return {
        "trace_id": handle.trace_id,
        "path": handle.path,
        "hmac_key_hex": handle.key.hmac_key.hex(),
        "signing_key_hex": priv_bytes.hex(),
        "public_key_hex": pub_hex,
        "hmac_key_id": hashlib.sha256(handle.key.hmac_key).hexdigest()[:16],
        "format_version": FORMAT_VERSION,
        "recorder_version": RECORDER_VERSION,
        "proxy_version": PROXY_VERSION,
    }


class _Handler(BaseHTTPRequestHandler):
    server_version = f"stepback-proxy/{PROXY_VERSION}"
    # Bound at server-construction time; see ProxyHTTPServer.__init__.
    state: ProxyState  # type: ignore[assignment]
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES

    # ------------------------------------------------------- helpers
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Route through the module logger so callers can configure verbosity.
        logger.info("%s - - " + format, self.address_string(), *args)

    def _send_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Stepback-Proxy", PROXY_VERSION)
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(self, status: int, error_class: str, message: str) -> None:
        self._send_json(status, {"error": error_class, "message": message})

    def _read_json(self) -> Tuple[Optional[dict], Optional[Tuple[int, str, str]]]:
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            return {}, None
        try:
            length = int(length_header)
        except ValueError:
            return None, (400, "BadRequest", "invalid Content-Length")
        if length < 0:
            return None, (400, "BadRequest", "negative Content-Length")
        if length > self.max_body_bytes:
            return None, (
                413,
                "PayloadTooLarge",
                f"body exceeds {self.max_body_bytes} bytes",
            )
        if length == 0:
            return {}, None
        raw = self.rfile.read(length)
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            return None, (400, "BadRequest", f"invalid JSON body: {e}")
        if not isinstance(obj, dict):
            return None, (400, "BadRequest", "JSON body must be an object")
        return obj, None

    # --------------------------------------------------------- routes
    def do_GET(self) -> None:  # noqa: N802 (stdlib API)
        if self.path == "/healthz":
            self._send_json(
                200,
                {
                    "ok": True,
                    "proxy_version": PROXY_VERSION,
                    "schema": HTTP_SCHEMA,
                    "open_traces": len(self.state.list_traces()),
                },
            )
            return
        if self.path == "/v1/info":
            self._send_json(
                200,
                {
                    "proxy_version": PROXY_VERSION,
                    "schema": HTTP_SCHEMA,
                    "format_version": FORMAT_VERSION,
                    "recorder_version": RECORDER_VERSION,
                    "write_dir": self.state.write_dir,
                    "max_open_traces": self.state.max_open_traces,
                    "max_body_bytes": self.max_body_bytes,
                },
            )
            return
        if self.path == "/v1/traces":
            self._send_json(200, {"traces": list(self.state.list_traces())})
            return
        self._send_error(404, "NotFound", f"unknown route: GET {self.path}")

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._dispatch_post()
        except UnknownTraceError as e:
            self._send_error(404, "UnknownTrace", str(e))
        except ProxyError as e:
            self._send_error(409, "ProxyError", str(e))
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("internal error in POST %s", self.path)
            self._send_error(500, "InternalError", str(e))

    def _dispatch_post(self) -> None:
        path = self.path.rstrip("/")
        if path == "/v1/traces":
            self._post_start_trace()
            return
        if path == "/v1/verify":
            self._post_verify()
            return
        if path == "/v1/replay":
            self._post_replay()
            return
        # /v1/traces/{trace_id}/steps and /v1/traces/{trace_id}/end
        prefix = "/v1/traces/"
        if path.startswith(prefix):
            tail = path[len(prefix):]
            if tail.endswith("/steps"):
                trace_id = tail[: -len("/steps")]
                self._post_record_step(trace_id)
                return
            if tail.endswith("/end"):
                trace_id = tail[: -len("/end")]
                self._post_end_trace(trace_id)
                return
        self._send_error(404, "NotFound", f"unknown route: POST {self.path}")

    # ----------------------------------------------------- StartTrace
    def _post_start_trace(self) -> None:
        body, err = self._read_json()
        if err is not None:
            self._send_error(*err)
            return
        assert body is not None
        try:
            handle = self.state.start_trace(
                filename=body.get("filename"),
                compression=bool(body.get("compression", True)),
                price_list_version=str(
                    body.get("price_list_version", "2026-04-01")
                ),
                blob_threshold=body.get("blob_threshold"),
                blob_min_reuse=body.get("blob_min_reuse"),
            )
        except ProxyError as e:
            self._send_error(409, "ProxyError", str(e))
            return
        self._send_json(201, _start_trace_response(handle))

    # ----------------------------------------------------- RecordStep
    def _post_record_step(self, trace_id: str) -> None:
        body, err = self._read_json()
        if err is not None:
            self._send_error(*err)
            return
        assert body is not None
        step = body.get("step")
        if not isinstance(step, dict):
            self._send_error(
                400, "BadRequest", "request body missing 'step' object"
            )
            return
        handle = self.state.record_step(trace_id, step)
        self._send_json(
            200,
            {"ok": True, "trace_id": trace_id, "step_count": handle.step_count},
        )

    # ------------------------------------------------------- EndTrace
    def _post_end_trace(self, trace_id: str) -> None:
        # Body is optional; we still drain it to keep the connection clean.
        _body, err = self._read_json()
        if err is not None:
            self._send_error(*err)
            return
        handle = self.state.end_trace(trace_id)
        self._send_json(
            200,
            {
                "ok": True,
                "trace_id": handle.trace_id,
                "path": handle.path,
                "step_count": handle.step_count,
            },
        )

    # ---------------------------------------------------- VerifyTrace
    def _post_verify(self) -> None:
        body, err = self._read_json()
        if err is not None:
            self._send_error(*err)
            return
        assert body is not None
        path = body.get("path")
        trace_id = body.get("trace_id")
        if not path and not trace_id:
            self._send_error(
                400, "BadRequest", "must specify 'path' or 'trace_id'"
            )
            return
        if path and trace_id:
            self._send_error(
                400, "BadRequest", "specify exactly one of 'path' or 'trace_id'"
            )
            return
        hmac_key_hex = body.get("hmac_key_hex")
        if not isinstance(hmac_key_hex, str):
            self._send_error(
                400, "BadRequest", "'hmac_key_hex' is required"
            )
            return
        try:
            hmac_key = bytes.fromhex(hmac_key_hex)
        except ValueError as e:
            self._send_error(400, "BadRequest", f"invalid hmac_key_hex: {e}")
            return
        if trace_id:
            try:
                handle = self.state.get(trace_id)
            except UnknownTraceError as e:
                self._send_error(404, "UnknownTrace", str(e))
                return
            if not handle.closed:
                self._send_error(
                    409,
                    "TraceStillOpen",
                    "cannot verify a trace that is still being written; "
                    "call EndTrace first",
                )
                return
            verify_path = handle.path
        else:
            verify_path = path
        try:
            v = verify_trace(verify_path, hmac_key)
        except TraceVerificationError as e:
            self._send_json(400, {"ok": False, "error": str(e)})
            return
        except FileNotFoundError as e:
            self._send_error(404, "NotFound", str(e))
            return
        header = v.header or {}
        self._send_json(
            200,
            {
                "ok": True,
                "step_count": len(v.steps),
                "public_key_hex": v.public_key_hex,
                "format_version": header.get("format_version"),
                "recorder_version": header.get("recorder_version"),
                "canonicalisation_version": header.get(
                    "canonicalisation_version"
                ),
                "hmac_key_id": header.get("hmac_key_id"),
            },
        )

    # ---------------------------------------------------- ReplayTrace
    def _post_replay(self) -> None:
        """Stream replay events as NDJSON for a closed ``.sb`` file."""
        body, err = self._read_json()
        if err is not None:
            self._send_error(*err)
            return
        assert body is not None
        path = body.get("path")
        if not path:
            self._send_error(400, "BadRequest", "'path' is required")
            return
        hmac_key_hex = body.get("hmac_key_hex")
        hmac_key: Optional[bytes] = None
        if hmac_key_hex is not None:
            try:
                hmac_key = bytes.fromhex(hmac_key_hex)
            except ValueError as e:
                self._send_error(400, "BadRequest", f"invalid hmac_key_hex: {e}")
                return
        fallback_recorded = bool(body.get("fallback_recorded", True))

        # Build substitution set from caller-supplied list.
        subs = SubstitutionSet()
        for spec in body.get("substitutions") or []:
            step_id = spec.get("step_id")
            kind = spec.get("kind")
            if not step_id or not kind:
                self._send_error(
                    400, "BadRequest",
                    "each substitution must have 'step_id' and 'kind'"
                )
                return
            if kind == "tool_output":
                value = spec.get("value")
                subs.add(ToolOutputSubstitution(at_step=step_id, fake_response=value))
            else:
                self._send_error(
                    400, "BadRequest",
                    f"unsupported substitution kind: {kind!r}; use 'tool_output'"
                )
                return

        try:
            trace = load_trace(path, hmac_key=hmac_key)
        except TraceVerificationError as e:
            self._send_error(400, "VerificationFailed", str(e))
            return
        except FileNotFoundError as e:
            self._send_error(404, "NotFound", str(e))
            return

        executor = Executor(fallback_recorded=fallback_recorded)

        # Collect all events into NDJSON bytes, then send with Content-Length.
        # Using chunked encoding in BaseHTTPRequestHandler requires explicit
        # chunk framing which complicates client compatibility; buffering is
        # fine because individual events are small and traces are bounded.
        lines = []
        for event in replay_events(trace.recorded_steps, subs, executor):
            lines.append(json.dumps(event, sort_keys=True) + "\n")
        payload = "".join(lines).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Stepback-Proxy", PROXY_VERSION)
        self.end_headers()
        self.wfile.write(payload)


class ProxyHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server bound to a :class:`ProxyState`."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: Tuple[str, int],
        state: ProxyState,
        *,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    ) -> None:
        self.state = state
        # Subclass the handler per server so we can attach state/limits
        # without polluting the module-level handler class.
        cls_name = "_BoundHandler"
        bound_cls = type(
            cls_name,
            (_Handler,),
            {"state": state, "max_body_bytes": max_body_bytes},
        )
        super().__init__(address, bound_cls)


def serve_http(
    state: ProxyState,
    *,
    host: str = "127.0.0.1",
    port: int = 4319,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    ready_event: Optional[threading.Event] = None,
) -> ProxyHTTPServer:
    """Start an :class:`ProxyHTTPServer`. Caller is responsible for shutdown.

    If ``ready_event`` is given, it's set once the socket is bound — useful
    when starting the server on a background thread in tests.
    """
    server = ProxyHTTPServer((host, port), state, max_body_bytes=max_body_bytes)
    if ready_event is not None:
        ready_event.set()
    return server
