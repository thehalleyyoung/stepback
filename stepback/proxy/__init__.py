"""``stepback-proxy`` — HTTP (and optional gRPC) sidecar for recording
``.sb`` traces from non-Python runtimes.

The proxy exposes four core RPCs that mirror the in-process recorder:

* ``StartTrace``  — open a new ``.sb`` file, mint per-trace HMAC + Ed25519
  keys, and return a trace handle.
* ``RecordStep``  — append a step frame to an open trace.
* ``EndTrace``    — flush and close an open trace.
* ``VerifyTrace`` — verify HMAC chain + per-frame Ed25519 signatures of an
  existing ``.sb`` file.

Two transports are provided:

* :mod:`stepback.proxy.server`     — JSON-over-HTTP using the stdlib only.
  This is the always-available transport and is what ``stepback proxy``
  starts by default. The wire schema is documented in
  :mod:`stepback.proxy.server` (see ``HTTP_SCHEMA``).
* :mod:`stepback.proxy.grpc_server` — optional gRPC transport. The proto
  file lives at ``stepback/proxy/proto/sbproxy.proto`` and is the canonical
  IDL. The Python implementation is gated on ``grpcio`` being installed
  (``pip install stepback[proxy-grpc]``) and falls back to an explicit
  error if not.

Both transports operate on the same :class:`stepback.proxy.storage.ProxyState`
backing object, so HTTP and gRPC clients can interoperate against the same
proxy process and same on-disk trace directory.
"""
from __future__ import annotations

from .server import (
    HTTP_SCHEMA,
    PROXY_VERSION,
    ProxyHTTPServer,
    serve_http,
)
from .storage import (
    ProxyError,
    ProxyState,
    TraceHandle,
    UnknownTraceError,
)

__all__ = [
    "HTTP_SCHEMA",
    "PROXY_VERSION",
    "ProxyHTTPServer",
    "ProxyError",
    "ProxyState",
    "TraceHandle",
    "UnknownTraceError",
    "serve_http",
]
