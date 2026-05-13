"""`.sb` reader + verifier.

`read_frames` returns the list of wrappers exactly as written;
`verify_trace` walks the chain and verifies every HMAC link and every
Ed25519 signature, raising `TraceVerificationError` on tamper.

The default verification engine is the pure-Python implementation in
this module. An experimental Rust engine is available via the
``stepback_core`` extension package (built from
``bindings/python/stepback_core``). Opt in by passing ``engine="rust"``
or by setting the ``STEPBACK_VERIFY_ENGINE=rust`` environment variable.
The Rust engine performs the cryptographic chain + signature checks
through ``sb-verify``; semantic frame decoding (steps, blobs, tail)
still happens in Python so callers see the same :class:`Trace_` shape
regardless of engine.
"""
from __future__ import annotations

import base64
import binascii
import gzip
import hashlib
import hmac
import json
import os
import struct
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .canonical import canonical_json
from .merkle import leaf_hash, merkle_root

ZERO_HMAC = b"\x00" * 32
BLOB_REF_KEY = "$blob"

#: Capabilities this build understands by name. Capability frames
#: declared in a trace as ``mandatory=True`` whose ``name`` is not in
#: this set cause :func:`verify_trace` to fail closed (Step 24, see
#: ``spec/sbtrace-v1.md`` §6.2 and §13). ``"core"`` is the implicit
#: minimum capability and need not be declared. Callers may override
#: per-call via the ``supported_capabilities`` argument or globally via
#: the :data:`STEPBACK_SUPPORTED_CAPABILITIES_ENV` environment variable.
DEFAULT_SUPPORTED_CAPABILITIES: frozenset = frozenset(
    {
        "core",
        "blobs",
        "gzip-step-bodies",
        "ed25519-receipts",
        "hmac-sha256-chain",
        "merkle-summary-v1",
    }
)

#: Environment variable name. When set, its value is parsed as a
#: comma-separated list of capability names that *replace* the default
#: allow-list for the duration of the call. The empty string means
#: "only the implicit ``core`` capability is supported". An explicit
#: ``supported_capabilities`` argument to :func:`verify_trace` always
#: takes precedence over the environment variable.
STEPBACK_SUPPORTED_CAPABILITIES_ENV = "STEPBACK_SUPPORTED_CAPABILITIES"

VerifyEngine = Literal["python", "rust", "auto"]
_VERIFY_ENGINE_ENV = "STEPBACK_VERIFY_ENGINE"
_VALID_ENGINES = ("python", "rust", "auto")


#: Maximum size (bytes) of a single on-wire frame body. Length
#: prefixes above this are rejected before any body bytes are read.
#: See ``docs/reader-limits.md`` (Step 50).
MAX_FRAME_BYTES = 64 * 1024 * 1024  # 64 MiB

#: Maximum nesting depth of any JSON value inside a frame wrapper or
#: body. Catches malicious ``{"x":{"x":{"x":...`` ladders before they
#: blow the host runtime's stack.
MAX_NESTING_DEPTH = 256

#: Maximum UTF-8 byte length of any single JSON string value or
#: object key inside a frame wrapper or body. Independent of, and
#: stricter than, ``MAX_FRAME_BYTES``.
MAX_STRING_BYTES = 16 * 1024 * 1024  # 16 MiB


class TraceVerificationError(Exception):
    """Raised when an `.sb` file fails signature or HMAC verification."""


def _enforce_value_limits(value, max_depth: int, max_string_bytes: int) -> None:
    """Walk ``value`` iteratively and reject anything that exceeds the
    documented :data:`MAX_NESTING_DEPTH` or :data:`MAX_STRING_BYTES`
    bounds. Iterative to avoid letting Python's own recursion limit
    become the effective depth cap.
    """
    stack = [(value, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > max_depth:
            raise TraceVerificationError(
                f"nesting depth exceeds MAX_NESTING_DEPTH ({max_depth})"
            )
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str) and len(k.encode("utf-8")) > max_string_bytes:
                    raise TraceVerificationError(
                        f"object key exceeding MAX_STRING_BYTES "
                        f"({max_string_bytes}) bytes"
                    )
                stack.append((v, depth + 1))
        elif isinstance(node, list):
            for v in node:
                stack.append((v, depth + 1))
        elif isinstance(node, str):
            if len(node.encode("utf-8")) > max_string_bytes:
                raise TraceVerificationError(
                    f"string value exceeding MAX_STRING_BYTES "
                    f"({max_string_bytes}) bytes"
                )


def _validate_merkle_summary_body(
    body: dict, leaves: list
) -> tuple[str, int]:
    """Enforce merkle_summary frame shape + recompute the root.

    Returns ``(merkle_root_hex, leaf_count)``. Raises
    :class:`TraceVerificationError` with an explicit message naming the
    failing field, so audit logs remain greppable. See
    ``spec/sbtrace-v1.md`` §6.7.
    """
    from .trace_writer import MERKLE_SCHEME  # local to avoid cycle
    scheme = body.get("scheme")
    if scheme != MERKLE_SCHEME:
        raise TraceVerificationError(
            f"merkle_summary scheme {scheme!r} not supported "
            f"(expected {MERKLE_SCHEME!r})"
        )
    algorithm = body.get("algorithm")
    if algorithm != "sha256":
        raise TraceVerificationError(
            f"merkle_summary algorithm {algorithm!r} not supported "
            f"(expected 'sha256')"
        )
    declared_count = body.get("leaf_count")
    if not isinstance(declared_count, int) or isinstance(declared_count, bool):
        raise TraceVerificationError(
            "merkle_summary 'leaf_count' must be an integer"
        )
    if declared_count != len(leaves):
        raise TraceVerificationError(
            f"merkle_summary leaf_count mismatch: declared "
            f"{declared_count}, observed {len(leaves)} content frames"
        )
    declared_root = body.get("merkle_root")
    if not isinstance(declared_root, str) or len(declared_root) != 64:
        raise TraceVerificationError(
            "merkle_summary 'merkle_root' must be a 64-char hex string"
        )
    try:
        bytes.fromhex(declared_root)
    except ValueError as exc:
        raise TraceVerificationError(
            f"merkle_summary 'merkle_root' is not valid hex: {exc}"
        ) from exc
    recomputed = merkle_root(leaves).hex()
    if recomputed != declared_root:
        raise TraceVerificationError(
            f"merkle_summary root mismatch: declared {declared_root}, "
            f"recomputed {recomputed}"
        )
    return declared_root, declared_count


@dataclass
class Trace_:
    header: dict
    steps: list = field(default_factory=list)
    tail: Optional[dict] = None
    public_key_hex: str = ""
    blobs: dict = field(default_factory=dict)
    capabilities: list = field(default_factory=list)
    merkle_root: Optional[str] = None
    merkle_leaf_count: int = 0


def _resolve_supported_capabilities(
    explicit: Optional[Iterable[str]],
) -> frozenset:
    """Pick the allow-list for ``verify_trace``.

    Precedence: explicit argument > env variable > module default.
    """
    if explicit is not None:
        return frozenset({"core", *(str(n) for n in explicit)})
    env = os.environ.get(STEPBACK_SUPPORTED_CAPABILITIES_ENV)
    if env is None:
        return DEFAULT_SUPPORTED_CAPABILITIES
    names = {n.strip() for n in env.split(",") if n.strip()}
    return frozenset({"core", *names})


def _validate_capability_body(body: dict) -> dict:
    """Enforce capability-frame shape; surface defects as
    :class:`TraceVerificationError` (Step 24).
    """
    name = body.get("name")
    if not isinstance(name, str) or not name:
        raise TraceVerificationError(
            "capability frame is missing required field 'name' "
            "(or it is empty / not a string)"
        )
    mandatory = body.get("mandatory")
    if not isinstance(mandatory, bool):
        raise TraceVerificationError(
            f"capability {name!r} is missing required boolean "
            f"field 'mandatory'"
        )
    params = body.get("params")
    if params is not None and not isinstance(params, dict):
        raise TraceVerificationError(
            f"capability {name!r} has non-object 'params' field"
        )
    out = {"name": name, "mandatory": mandatory}
    if params is not None:
        out["params"] = params
    return out


def _check_capability_negotiation(
    capabilities: list, supported: frozenset
) -> None:
    """Fail closed on any mandatory capability not in ``supported``.

    The error message lists *only* the unknown names, never the known
    ones, so audit log lines reveal the capability gap without leaking
    the full allow-list.
    """
    unknown = [
        c["name"]
        for c in capabilities
        if c.get("mandatory") and c.get("name") not in supported
    ]
    if not unknown:
        return
    if len(unknown) == 1:
        raise TraceVerificationError(
            f"trace declares mandatory capability {unknown[0]!r} which "
            f"this build does not support"
        )
    listed = ", ".join(repr(n) for n in unknown)
    raise TraceVerificationError(
        f"trace declares mandatory capabilities {{{listed}}} which "
        f"this build does not support"
    )


def _has_blob_ref(value) -> bool:
    if isinstance(value, dict):
        if BLOB_REF_KEY in value and len(value) == 1:
            return True
        return any(_has_blob_ref(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_blob_ref(v) for v in value)
    return False


def _materialise(value, blobs: dict):
    if isinstance(value, dict):
        if BLOB_REF_KEY in value and len(value) == 1:
            digest = value[BLOB_REF_KEY]
            if digest not in blobs:
                raise TraceVerificationError(
                    f"blob ref to unknown digest {digest!r}"
                )
            return _materialise(blobs[digest], blobs)
        return {k: _materialise(v, blobs) for k, v in value.items()}
    if isinstance(value, list):
        return [_materialise(v, blobs) for v in value]
    return value


def _decode_gz_step(body: dict) -> dict:
    encoding = body.get("encoding", "json")
    if encoding == "gzip+base64":
        raw = gzip.decompress(base64.b64decode(body["data"].encode("ascii")))
        return json.loads(raw.decode("utf-8"))
    return body["step"]


def _decode_blob(body: dict) -> object:
    digest = body["id"]
    encoding = body.get("encoding", "json")
    raw_str = body["data"]
    if encoding == "gzip+base64":
        raw = gzip.decompress(base64.b64decode(raw_str.encode("ascii")))
    elif encoding == "json":
        raw = raw_str.encode("utf-8")
    else:
        raise TraceVerificationError(f"unknown blob encoding {encoding!r}")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise TraceVerificationError(
            f"blob digest mismatch (declared {digest})"
        )
    return json.loads(raw.decode("utf-8"))


def read_frames(
    path: str,
    *,
    max_frame_bytes: Optional[int] = None,
    max_depth: Optional[int] = None,
    max_string_bytes: Optional[int] = None,
) -> list:
    """Read and decode every length-prefixed frame in ``path``.

    Enforces the Step-50 denial-of-service bounds documented in
    ``docs/reader-limits.md``: every frame body that exceeds
    :data:`MAX_FRAME_BYTES`, contains JSON nested deeper than
    :data:`MAX_NESTING_DEPTH`, or contains a string value or key
    whose UTF-8 encoding exceeds :data:`MAX_STRING_BYTES` is
    rejected with :class:`TraceVerificationError`. The defaults
    are strict; pass higher caps explicitly to opt in.
    """
    cap_frame = MAX_FRAME_BYTES if max_frame_bytes is None else int(max_frame_bytes)
    cap_depth = MAX_NESTING_DEPTH if max_depth is None else int(max_depth)
    cap_str = MAX_STRING_BYTES if max_string_bytes is None else int(max_string_bytes)
    out: list = []
    with open(path, "rb") as f:
        while True:
            ln = f.read(4)
            if not ln:
                break
            if len(ln) < 4:
                raise TraceVerificationError("truncated length prefix")
            (n,) = struct.unpack(">I", ln)
            if n > cap_frame:
                raise TraceVerificationError(
                    f"frame length {n} exceeds MAX_FRAME_BYTES ({cap_frame})"
                )
            payload = f.read(n)
            if len(payload) < n:
                raise TraceVerificationError("truncated frame body")
            try:
                decoded = json.loads(payload.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise TraceVerificationError(
                    f"frame body is not valid UTF-8: {exc}"
                ) from exc
            except json.JSONDecodeError as exc:
                raise TraceVerificationError(
                    f"frame body is not valid JSON: {exc}"
                ) from exc
            _enforce_value_limits(decoded, cap_depth, cap_str)
            out.append(decoded)
    return out


def verify_trace(
    path: str,
    hmac_key: bytes,
    *,
    engine: VerifyEngine = "auto",
    supported_capabilities: Optional[Iterable[str]] = None,
) -> Trace_:
    """Verify ``path`` and return the parsed header/steps/tail.

    Parameters
    ----------
    path:
        Filesystem path to the ``.sb`` file.
    hmac_key:
        HMAC key bytes pinned by the writer (32 bytes for the
        reference recorder; any length the spec allows is accepted).
    engine:
        Which verification engine to use:

        * ``"python"`` (default historical behaviour) — pure-Python
          HMAC chain + Ed25519 signature verification, always
          available.
        * ``"rust"`` — route the cryptographic check through the
          ``sb-verify`` Rust crate via the experimental
          ``stepback_core`` PyO3 binding. Raises :class:`RuntimeError`
          if the binding is not installed. The Python decoder still
          materialises blobs and gzip-compressed step bodies so the
          returned :class:`Trace_` is identical regardless of engine.
        * ``"auto"`` — honour the ``STEPBACK_VERIFY_ENGINE``
          environment variable when set to one of the engine names
          above; otherwise fall back to ``"python"``. Unknown values
          raise :class:`ValueError`.

    supported_capabilities:
        Optional iterable of capability names this caller implements
        in addition to the implicit ``core``. Overrides both the
        module default :data:`DEFAULT_SUPPORTED_CAPABILITIES` and the
        ``STEPBACK_SUPPORTED_CAPABILITIES`` environment variable. A
        trace declaring a ``mandatory`` capability not in the
        resolved set is rejected with :class:`TraceVerificationError`
        (Step 24, see ``spec/sbtrace-v1.md`` §6.2).

    The Rust engine is gated and *experimental*: the wire-format
    contract is identical, but the Python error taxonomy and the
    eventual Rust error taxonomy may diverge in non-OK cases. Pin the
    engine explicitly in production audit pipelines.
    """
    resolved = _resolve_engine(engine)
    allow = _resolve_supported_capabilities(supported_capabilities)
    if resolved == "rust":
        return _verify_trace_rust(path, hmac_key, allow)
    return _verify_trace_python(path, hmac_key, allow)


def _resolve_engine(engine: VerifyEngine) -> str:
    if engine not in _VALID_ENGINES:
        raise ValueError(
            f"engine must be one of {_VALID_ENGINES!r}, got {engine!r}"
        )
    if engine != "auto":
        return engine
    env = os.environ.get(_VERIFY_ENGINE_ENV, "").strip().lower()
    if env in ("", "python"):
        return "python"
    if env == "rust":
        return "rust"
    if env == "auto":
        # ``auto`` in env means: try rust if importable, else python.
        try:
            import stepback_core  # noqa: F401
        except ImportError:
            return "python"
        return "rust"
    raise ValueError(
        f"{_VERIFY_ENGINE_ENV} must be one of "
        f"{_VALID_ENGINES!r}, got {env!r}"
    )


def _verify_trace_rust(path: str, hmac_key: bytes, supported: frozenset) -> Trace_:
    """Rust-backed crypto verification + Python semantic decoding.

    The Rust crate (``sb_verify::verify_bytes``) walks the HMAC chain
    and verifies every Ed25519 signature in one pass. On success we
    re-read the frame bodies in Python *without* re-running the crypto
    checks: the Rust pass already proved the bytes are intact, and the
    decoder only cares about the semantic shape (header, step bodies,
    blob frames, tail).
    """
    try:
        import stepback_core
    except ImportError as exc:  # pragma: no cover - exercised in tests
        raise RuntimeError(
            "engine='rust' requested but the stepback_core extension is "
            "not installed. Build it with `maturin develop --release` "
            "from bindings/python/stepback_core/, or install the "
            "`stepback-core` wheel."
        ) from exc

    try:
        stepback_core.verify_path(path, hmac_key)
    except stepback_core.VerifyError as exc:
        # Surface a TraceVerificationError so existing callers'
        # except-clauses keep working when they flip the engine.
        kind = getattr(exc, "kind", "Unknown")
        idx = getattr(exc, "frame_index", -1)
        raise TraceVerificationError(
            f"Rust verifier rejected trace ({kind} at frame {idx}): {exc}"
        ) from exc

    return _decode_trace_unverified(path, supported)


def _decode_trace_unverified(path: str, supported: frozenset) -> Trace_:
    """Parse an `.sb` we have *already* crypto-verified.

    Mirrors the body of :func:`_verify_trace_python` but skips HMAC
    recomputation and signature verification. Only safe to call when
    a stronger verifier (e.g. ``sb-verify``) has just succeeded on
    the same bytes.
    """
    frames = read_frames(path)
    if not frames:
        raise TraceVerificationError("empty trace")
    header: Optional[dict] = None
    steps: list = []
    tail: Optional[dict] = None
    blobs: dict = {}
    capabilities: list = []
    leaves: list = []
    merkle_root_hex: Optional[str] = None
    merkle_leaf_count: int = 0
    summary_seen = False
    for wrapper in frames:
        if not isinstance(wrapper, dict) or "body" not in wrapper:
            raise TraceVerificationError(
                "frame wrapper must be a JSON object with a 'body' field"
            )
        body = wrapper["body"]
        if not isinstance(body, dict):
            raise TraceVerificationError(
                f"frame body must be a JSON object, got {type(body).__name__}"
            )
        kind = body.get("type")
        if summary_seen and kind != "tail":
            raise TraceVerificationError(
                f"unexpected frame type {kind!r} after merkle_summary "
                f"(only 'tail' may follow)"
            )
        if kind not in ("merkle_summary", "tail"):
            leaves.append(leaf_hash(canonical_json(body)))
        if kind == "header":
            header = body
        elif kind == "step":
            try:
                step = _decode_gz_step(body)
            except (KeyError, ValueError, TypeError, gzip.BadGzipFile,
                    binascii.Error, json.JSONDecodeError,
                    UnicodeDecodeError) as exc:
                raise TraceVerificationError(
                    f"step frame body is malformed: {exc}"
                ) from exc
            if _has_blob_ref(step):
                if not blobs:
                    raise TraceVerificationError(
                        "step frame references a blob but no blob "
                        "frames seen yet"
                    )
                step = _materialise(step, blobs)
            steps.append(step)
        elif kind == "blob":
            try:
                blobs[body["id"]] = _decode_blob(body)
            except (KeyError, ValueError, TypeError, gzip.BadGzipFile,
                    binascii.Error, json.JSONDecodeError,
                    UnicodeDecodeError) as exc:
                raise TraceVerificationError(
                    f"blob frame is malformed: {exc}"
                ) from exc
        elif kind == "tail":
            tail = body
        elif kind == "capability":
            capabilities.append(_validate_capability_body(body))
        elif kind == "merkle_summary":
            if summary_seen:
                raise TraceVerificationError(
                    "more than one merkle_summary frame in trace"
                )
            merkle_root_hex, merkle_leaf_count = _validate_merkle_summary_body(
                body, leaves
            )
            summary_seen = True
    if header is None:
        raise TraceVerificationError("no header frame in trace")
    _check_capability_negotiation(capabilities, supported)
    return Trace_(
        header=header,
        steps=steps,
        tail=tail,
        public_key_hex=header["public_key"],
        blobs=blobs,
        capabilities=capabilities,
        merkle_root=merkle_root_hex,
        merkle_leaf_count=merkle_leaf_count,
    )


def _verify_trace_python(path: str, hmac_key: bytes, supported: frozenset) -> Trace_:
    """Verify ``path`` and return the parsed header/steps/tail."""
    frames = read_frames(path)
    if not frames:
        raise TraceVerificationError("empty trace")
    prev = ZERO_HMAC
    pub: Optional[Ed25519PublicKey] = None
    header: Optional[dict] = None
    steps: list = []
    tail: Optional[dict] = None
    blobs: dict = {}
    capabilities: list = []
    leaves: list = []
    merkle_root_hex: Optional[str] = None
    merkle_leaf_count: int = 0
    summary_seen = False
    for wrapper in frames:
        if not isinstance(wrapper, dict):
            raise TraceVerificationError(
                f"frame wrapper must be a JSON object, got "
                f"{type(wrapper).__name__}"
            )
        for required in ("body", "prev_hmac", "hmac", "sig"):
            if required not in wrapper:
                raise TraceVerificationError(
                    f"frame wrapper missing required field {required!r}"
                )
        body = wrapper["body"]
        if not isinstance(body, dict):
            raise TraceVerificationError(
                f"frame body must be a JSON object, got {type(body).__name__}"
            )
        kind = body.get("type")
        if summary_seen and kind != "tail":
            raise TraceVerificationError(
                f"unexpected frame type {kind!r} after merkle_summary "
                f"(only 'tail' may follow)"
            )
        body_bytes = canonical_json(body)
        h = hmac.new(hmac_key, prev + body_bytes, hashlib.sha256).digest()
        wrapper_hmac = wrapper["hmac"]
        wrapper_prev = wrapper["prev_hmac"]
        if not isinstance(wrapper_hmac, str) or not isinstance(wrapper_prev, str):
            raise TraceVerificationError("hmac/prev_hmac must be hex strings")
        if h.hex() != wrapper_hmac:
            raise TraceVerificationError(
                f"HMAC chain broken at frame type={kind}"
            )
        if wrapper_prev != prev.hex():
            raise TraceVerificationError("prev_hmac mismatch")
        if kind == "header":
            header = body
            pk_hex = body.get("public_key", "")
            if pk_hex:
                try:
                    pub = Ed25519PublicKey.from_public_bytes(
                        bytes.fromhex(pk_hex)
                    )
                except (KeyError, ValueError, TypeError) as exc:
                    raise TraceVerificationError(
                        f"header has invalid public_key: {exc}"
                    ) from exc
            else:
                # Unsigned trace — no public key; sig fields will be "none".
                pub = None  # type: ignore[assignment]
        if pub is None and header is None:
            raise TraceVerificationError("first frame must be a header")
        sig_field = wrapper["sig"]
        if sig_field == "none":
            # Unsigned trace — skip Ed25519 verification.
            pass
        elif not isinstance(sig_field, str) or not sig_field.startswith("ed25519:"):
            raise TraceVerificationError("unknown signature scheme")
        else:
            try:
                sig = bytes.fromhex(sig_field.removeprefix("ed25519:"))
            except ValueError as exc:
                raise TraceVerificationError(
                    f"signature is not valid hex: {exc}"
                ) from exc
            try:
                pub.verify(sig, h)
            except InvalidSignature as exc:
                raise TraceVerificationError("Ed25519 signature invalid") from exc
        prev = h
        if kind not in ("merkle_summary", "tail"):
            leaves.append(leaf_hash(body_bytes))
        if kind == "step":
            try:
                step = _decode_gz_step(body)
            except (KeyError, ValueError, TypeError, gzip.BadGzipFile,
                    binascii.Error, json.JSONDecodeError,
                    UnicodeDecodeError) as exc:
                raise TraceVerificationError(
                    f"step frame body is malformed: {exc}"
                ) from exc
            if _has_blob_ref(step):
                if not blobs:
                    raise TraceVerificationError(
                        "step frame references a blob but no blob frames seen yet"
                    )
                step = _materialise(step, blobs)
            steps.append(step)
        elif kind == "tail":
            tail = body
        elif kind == "blob":
            try:
                blobs[body["id"]] = _decode_blob(body)
            except (KeyError, ValueError, TypeError, gzip.BadGzipFile,
                    binascii.Error, json.JSONDecodeError,
                    UnicodeDecodeError) as exc:
                raise TraceVerificationError(
                    f"blob frame is malformed: {exc}"
                ) from exc
        elif kind == "capability":
            capabilities.append(_validate_capability_body(body))
        elif kind == "merkle_summary":
            if summary_seen:
                raise TraceVerificationError(
                    "more than one merkle_summary frame in trace"
                )
            merkle_root_hex, merkle_leaf_count = _validate_merkle_summary_body(
                body, leaves
            )
            summary_seen = True
    if header is None:
        raise TraceVerificationError("no header frame in trace")
    _check_capability_negotiation(capabilities, supported)
    return Trace_(
        header=header,
        steps=steps,
        tail=tail,
        public_key_hex=header["public_key"],
        blobs=blobs,
        capabilities=capabilities,
        merkle_root=merkle_root_hex,
        merkle_leaf_count=merkle_leaf_count,
    )


def iter_steps(
    path: str,
    hmac_key: bytes,
    *,
    engine: VerifyEngine = "auto",
    supported_capabilities: Optional[Iterable[str]] = None,
) -> Iterable[dict]:
    yield from verify_trace(
        path,
        hmac_key,
        engine=engine,
        supported_capabilities=supported_capabilities,
    ).steps
