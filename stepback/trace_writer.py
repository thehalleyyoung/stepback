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

Compression / dedup
-------------------

The writer optionally compresses each step inline (gzip + base64 of
the canonical-JSON form of the step) and additionally interns
sub-trees that repeat across steps as content-addressed blobs.
A blob frame is only emitted when a given sub-tree is referenced at
least ``DEFAULT_BLOB_MIN_REUSE`` times — otherwise the per-blob HMAC
wrapper overhead (~280 bytes) would outweigh the saving.

Together this is the README §"Performance targets" "<30% of raw LLM
payload bytes" implementation: gzip catches local redundancy inside
one step, and the reuse-aware blob table catches cross-step
duplication of recurring objects (chat-message dicts that show up in
every llm_call's ``messages`` array, large ``response_format``
schemas referenced from many calls, etc.).
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import hmac
import json
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

BLOB_REF_KEY = "$blob"
DEFAULT_BLOB_THRESHOLD = 200  # min canonical-JSON bytes to consider interning
DEFAULT_BLOB_MIN_REUSE = 2     # only intern sub-trees referenced >= N times
COMPRESSION_SCHEME = "gzip+dedup-2"


def _walk_candidates(value, threshold: int, counts: dict, raws: dict) -> None:
    """Pre-pass: count canonical sizes of every interior dict/list."""
    if isinstance(value, dict):
        for v in value.values():
            _walk_candidates(v, threshold, counts, raws)
    elif isinstance(value, list):
        for v in value:
            _walk_candidates(v, threshold, counts, raws)
    else:
        return
    raw = canonical_json(value)
    if len(raw) < threshold:
        return
    digest = hashlib.sha256(raw).hexdigest()
    counts[digest] = counts.get(digest, 0) + 1
    raws.setdefault(digest, raw)


def _apply_intern(value, threshold: int, intern_set: set):
    if isinstance(value, dict):
        if BLOB_REF_KEY in value and len(value) == 1:
            return value
        out = {k: _apply_intern(v, threshold, intern_set) for k, v in value.items()}
        node = out
    elif isinstance(value, list):
        node = [_apply_intern(v, threshold, intern_set) for v in value]
    else:
        return value
    raw = canonical_json(node)
    if len(raw) < threshold:
        return node
    digest = hashlib.sha256(raw).hexdigest()
    if digest in intern_set:
        return {BLOB_REF_KEY: digest}
    return node


@dataclass
class TraceWriter:
    path: str
    hmac_key: bytes
    signing_key: Ed25519PrivateKey
    f: object = None
    prev_hmac: bytes = ZERO_HMAC
    compression: bool = True
    blob_threshold: int = DEFAULT_BLOB_THRESHOLD
    blob_min_reuse: int = DEFAULT_BLOB_MIN_REUSE
    pending_steps: list = field(default_factory=list)
    seen_blobs: set = field(default_factory=set)

    @classmethod
    def open(
        cls,
        path: str,
        hmac_key: Optional[bytes] = None,
        signing_key: Optional[Ed25519PrivateKey] = None,
        price_list_version: str = "2026-04-01",
        compression: bool = True,
        blob_threshold: int = DEFAULT_BLOB_THRESHOLD,
        blob_min_reuse: int = DEFAULT_BLOB_MIN_REUSE,
    ) -> "TraceWriter":
        hmac_key = hmac_key or os.urandom(32)
        signing_key = signing_key or Ed25519PrivateKey.generate()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        f = open(path, "wb")
        w = cls(
            path=path,
            hmac_key=hmac_key,
            signing_key=signing_key,
            f=f,
            compression=compression,
            blob_threshold=blob_threshold,
            blob_min_reuse=blob_min_reuse,
        )
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
                "compression": COMPRESSION_SCHEME if compression else "none",
                "blob_threshold": blob_threshold if compression else 0,
                "blob_min_reuse": blob_min_reuse if compression else 0,
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
        if not self.compression:
            self._write_frame({"type": "step", "step": step})
            return
        # Defer step emission until close() so we can globally
        # decide which sub-trees recur often enough to be worth
        # interning as their own blob frames. Snapshot via
        # canonical-JSON round-trip because callers (the recorder)
        # may mutate the step's nested mutable fields after handing
        # the step off (e.g. the ``messages`` list shared across
        # successive llm_call frames in the agent's chat history).
        snapshot = json.loads(canonical_json(step).decode("utf-8"))
        self.pending_steps.append(snapshot)

    def _flush_pending(self) -> None:
        if not self.pending_steps:
            return
        # Pre-pass: count occurrences of every interior dict/list across
        # all pending steps.
        counts: dict = {}
        raws: dict = {}
        for s in self.pending_steps:
            _walk_candidates(s, self.blob_threshold, counts, raws)
        intern_set = {
            d for d, n in counts.items() if n >= self.blob_min_reuse
        }
        # Emit blob frames first so the reader has them in its table
        # before processing the step frames that reference them.
        for digest in sorted(intern_set):
            if digest in self.seen_blobs:
                continue
            self.seen_blobs.add(digest)
            raw = raws[digest]
            gz = gzip.compress(raw, compresslevel=9, mtime=0)
            b64_gz = base64.b64encode(gz).decode("ascii")
            if len(b64_gz) + 24 < len(raw):
                self._write_frame(
                    {
                        "type": "blob",
                        "id": digest,
                        "encoding": "gzip+base64",
                        "data": b64_gz,
                    }
                )
            else:
                self._write_frame(
                    {
                        "type": "blob",
                        "id": digest,
                        "encoding": "json",
                        "data": raw.decode("utf-8"),
                    }
                )
        for s in self.pending_steps:
            interned = _apply_intern(s, self.blob_threshold, intern_set)
            raw = canonical_json(interned)
            gz = gzip.compress(raw, compresslevel=9, mtime=0)
            b64_gz = base64.b64encode(gz).decode("ascii")
            if len(b64_gz) + 24 < len(raw):
                self._write_frame(
                    {
                        "type": "step",
                        "encoding": "gzip+base64",
                        "data": b64_gz,
                    }
                )
            else:
                self._write_frame({"type": "step", "step": interned})
        self.pending_steps = []

    def close(self) -> None:
        if self.f and not self.f.closed:
            self._flush_pending()
            self._write_frame({"type": "tail", "wallclock_ns": time.time_ns()})
            self.f.close()
