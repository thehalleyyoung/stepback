"""Append-only `.sb` writer with chained HMAC + Ed25519 receipts.

Each frame on disk is::

    | 4-byte big-endian length | canonical-JSON wrapper |

where the wrapper is::

    {
      "body": <frame body>,
      "prev_hmac": "<hex>",
      "hmac":      "<hex>",          # HMAC_SHA256(hmac_key, prev_hmac || canonical_json(body))
      "sig":       "ed25519:<hex>",  # Ed25519 signature over `hmac` bytes
    }

Tampering with any frame breaks both the chain (the next frame's
``prev_hmac`` no longer matches) and the per-frame Ed25519 signature.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .canonical import CANONICALISATION_VERSION, canonical_json

ZERO_HMAC = b"\x00" * 32

FORMAT_VERSION = 1
RECORDER_VERSION = "0.1.0"


@dataclass
class TraceWriter:
    path: str
    hmac_key: bytes
    signing_key: Ed25519PrivateKey
    f: object = None
    prev_hmac: bytes = ZERO_HMAC

    @classmethod
    def open(
        cls,
        path: str,
        hmac_key: Optional[bytes] = None,
        signing_key: Optional[Ed25519PrivateKey] = None,
        price_list_version: str = "2026-04-01",
    ) -> "TraceWriter":
        hmac_key = hmac_key or os.urandom(32)
        signing_key = signing_key or Ed25519PrivateKey.generate()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        f = open(path, "wb")
        w = cls(path=path, hmac_key=hmac_key, signing_key=signing_key, f=f)
        public_key_hex = signing_key.public_key().public_bytes_raw().hex()
        w._write_frame(
            {
                "type": "header",
                "magic": "stepback/.sb",
                "format_version": FORMAT_VERSION,
                "recorder_version": RECORDER_VERSION,
                "canonicalisation_version": CANONICALISATION_VERSION,
                "public_key": public_key_hex,
                "hmac_key_id": hashlib.sha256(hmac_key).hexdigest()[:16],
                "price_list_version": price_list_version,
                "wallclock_ns": time.time_ns(),
            }
        )
        return w

    def _write_frame(self, body: dict) -> None:
        body_bytes = canonical_json(body)
        h = hmac.new(self.hmac_key, self.prev_hmac + body_bytes, hashlib.sha256).digest()
        sig = self.signing_key.sign(h)
        wrapper = {
            "body": body,
            "prev_hmac": self.prev_hmac.hex(),
            "hmac": h.hex(),
            "sig": "ed25519:" + sig.hex(),
        }
        wrapper_bytes = canonical_json(wrapper)
        self.f.write(struct.pack(">I", len(wrapper_bytes)))
        self.f.write(wrapper_bytes)
        self.f.flush()
        self.prev_hmac = h

    def write_step(self, step: dict) -> None:
        self._write_frame({"type": "step", "step": step})

    def close(self) -> None:
        if self.f and not self.f.closed:
            self._write_frame({"type": "tail", "wallclock_ns": time.time_ns()})
            self.f.close()
