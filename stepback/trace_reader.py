"""`.sb` reader + verifier.

`read_frames` returns the list of wrappers exactly as written;
`verify_trace` walks the chain and verifies every HMAC link and every
Ed25519 signature, raising `TraceVerificationError` on tamper.
"""
from __future__ import annotations

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


class TraceVerificationError(Exception):
    """Raised when an `.sb` file fails signature or HMAC verification."""


@dataclass
class Trace_:
    header: dict
    steps: list = field(default_factory=list)
    tail: Optional[dict] = None
    public_key_hex: str = ""


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
            steps.append(body["step"])
        elif body.get("type") == "tail":
            tail = body
    assert header is not None
    return Trace_(
        header=header, steps=steps, tail=tail, public_key_hex=header["public_key"]
    )


def iter_steps(path: str, hmac_key: bytes) -> Iterable[dict]:
    yield from verify_trace(path, hmac_key).steps
