"""`.sb` reader + verifier.

`read_frames` returns the list of wrappers exactly as written;
`verify_trace` walks the chain and verifies every HMAC link and every
Ed25519 signature, raising `TraceVerificationError` on tamper.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import hmac
import json
import struct
from dataclasses import dataclass, field
from typing import Iterable, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .canonical import canonical_json

ZERO_HMAC = b"\x00" * 32
BLOB_REF_KEY = "$blob"


class TraceVerificationError(Exception):
    """Raised when an `.sb` file fails signature or HMAC verification."""


@dataclass
class Trace_:
    header: dict
    steps: list = field(default_factory=list)
    tail: Optional[dict] = None
    public_key_hex: str = ""
    blobs: dict = field(default_factory=dict)


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


def read_frames(path: str) -> list:
    out: list = []
    with open(path, "rb") as f:
        while True:
            ln = f.read(4)
            if not ln:
                break
            if len(ln) < 4:
                raise TraceVerificationError("truncated length prefix")
            (n,) = struct.unpack(">I", ln)
            payload = f.read(n)
            if len(payload) < n:
                raise TraceVerificationError("truncated frame body")
            out.append(json.loads(payload.decode("utf-8")))
    return out


def verify_trace(path: str, hmac_key: bytes) -> Trace_:
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
    for wrapper in frames:
        body = wrapper["body"]
        body_bytes = canonical_json(body)
        h = hmac.new(hmac_key, prev + body_bytes, hashlib.sha256).digest()
        if h.hex() != wrapper["hmac"]:
            raise TraceVerificationError(
                f"HMAC chain broken at frame type={body.get('type')}"
            )
        if wrapper["prev_hmac"] != prev.hex():
            raise TraceVerificationError("prev_hmac mismatch")
        if body.get("type") == "header":
            header = body
            pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(body["public_key"]))
        if pub is None:
            raise TraceVerificationError("first frame must be a header")
        sig_field = wrapper["sig"]
        if not sig_field.startswith("ed25519:"):
            raise TraceVerificationError("unknown signature scheme")
        sig = bytes.fromhex(sig_field.removeprefix("ed25519:"))
        try:
            pub.verify(sig, h)
        except InvalidSignature as e:
            raise TraceVerificationError("Ed25519 signature invalid") from e
        prev = h
        if body.get("type") == "step":
            step = _decode_gz_step(body)
            if _has_blob_ref(step):
                if not blobs:
                    raise TraceVerificationError(
                        "step frame references a blob but no blob frames seen yet"
                    )
                step = _materialise(step, blobs)
            steps.append(step)
        elif body.get("type") == "tail":
            tail = body
        elif body.get("type") == "blob":
            blobs[body["id"]] = _decode_blob(body)
    assert header is not None
    return Trace_(
        header=header,
        steps=steps,
        tail=tail,
        public_key_hex=header["public_key"],
        blobs=blobs,
    )


def iter_steps(path: str, hmac_key: bytes) -> Iterable[dict]:
    yield from verify_trace(path, hmac_key).steps
