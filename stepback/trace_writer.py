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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, BinaryIO, List, Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .canonical import CANONICALISATION_VERSION, canonical_json
from .merkle import leaf_hash, merkle_root

ZERO_HMAC = b"\x00" * 32

FORMAT_VERSION = 1
RECORDER_VERSION = "0.1.0"

BLOB_REF_KEY = "$blob"
DEFAULT_BLOB_THRESHOLD = 200  # min canonical-JSON bytes to consider interning
DEFAULT_BLOB_MIN_REUSE = 2     # only intern sub-trees referenced >= N times
COMPRESSION_SCHEME = "gzip+dedup-2"
DEFAULT_BATCH_SIGN_INTERVAL = 100  # steps per auto-flush in streaming+batch mode

#: v1 Merkle summary scheme identifier. Leaves are
#: ``SHA-256(0x00 || canonical-JSON body bytes)`` of every header,
#: capability, blob, and step frame in on-disk order; the
#: ``merkle_summary`` and ``tail`` frames themselves are NOT leaves.
#: Internal nodes are ``SHA-256(0x01 || left || right)`` per RFC 6962.
MERKLE_SCHEME = "frame-body-sha256-rfc6962"


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


def _apply_intern(value: Any, threshold: int, intern_set: set) -> Any:
    node: Any
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
    f: Optional[BinaryIO] = None
    prev_hmac: bytes = ZERO_HMAC
    compression: bool = True
    blob_threshold: int = DEFAULT_BLOB_THRESHOLD
    blob_min_reuse: int = DEFAULT_BLOB_MIN_REUSE
    pending_steps: list = field(default_factory=list)
    pending_items: list = field(default_factory=list)
    seen_blobs: set = field(default_factory=set)
    emit_merkle_summary: bool = True
    _leaves: list = field(default_factory=list)
    signing: bool = True
    batch_sign: bool = False
    batch_sign_interval: int = DEFAULT_BATCH_SIGN_INTERVAL
    batch_sign_workers: int = 0
    _executor: Optional[ThreadPoolExecutor] = field(default=None, repr=False)
    _streaming_batch: List[dict] = field(default_factory=list)

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
        emit_merkle_summary: bool = True,
        signing: bool = True,
        batch_sign: bool = False,
        batch_sign_interval: int = DEFAULT_BATCH_SIGN_INTERVAL,
        batch_sign_workers: int = 0,
    ) -> "TraceWriter":
        if batch_sign and not signing:
            raise ValueError("batch_sign=True requires signing=True")
        hmac_key = hmac_key or os.urandom(32)
        signing_key = signing_key or Ed25519PrivateKey.generate()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        f = open(path, "wb")
        executor: Optional[ThreadPoolExecutor] = None
        if batch_sign:
            max_w = batch_sign_workers if batch_sign_workers > 0 else None
            executor = ThreadPoolExecutor(max_workers=max_w)
        w = cls(
            path=path,
            hmac_key=hmac_key,
            signing_key=signing_key,
            f=f,
            compression=compression,
            blob_threshold=blob_threshold,
            blob_min_reuse=blob_min_reuse,
            emit_merkle_summary=emit_merkle_summary,
            signing=signing,
            batch_sign=batch_sign,
            batch_sign_interval=batch_sign_interval,
            batch_sign_workers=batch_sign_workers,
            _executor=executor,
        )
        public_key_hex = signing_key.public_key().public_bytes_raw().hex() if signing else ""
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
        # Track every content-frame body in the Merkle leaf list; the
        # summary frame itself and the trailing tail are NOT leaves
        # (see ``spec/sbtrace-v1.md`` §6.7 and ``stepback/merkle.py``).
        if body.get("type") not in ("merkle_summary", "tail"):
            self._leaves.append(leaf_hash(body_bytes))
        h = hmac.new(self.hmac_key, self.prev_hmac + body_bytes, hashlib.sha256).digest()
        if self.signing:
            sig = self.signing_key.sign(h)
            sig_str = "ed25519:" + sig.hex()
        else:
            sig_str = "none"
        wrapper = {
            "body": body,
            "prev_hmac": self.prev_hmac.hex(),
            "hmac": h.hex(),
            "sig": sig_str,
        }
        wrapper_bytes = canonical_json(wrapper)
        assert self.f is not None
        self.f.write(struct.pack(">I", len(wrapper_bytes)))
        self.f.write(wrapper_bytes)
        self.f.flush()
        self.prev_hmac = h

    def _write_frames_batch(self, bodies: List[dict]) -> None:
        """Write multiple frames with parallel Ed25519 signing.

        HMAC chain is computed sequentially (each HMAC depends on the
        previous), but all Ed25519 signatures are computed in parallel
        via the thread pool executor.
        """
        if not bodies:
            return

        # Step 1: compute bodies + HMACs sequentially to maintain chain
        prev_hmacs = []  # prev_hmac for each frame (before its own HMAC)
        body_bytes_list = []
        hmac_list = []
        prev = self.prev_hmac
        for body in bodies:
            bb = canonical_json(body)
            if body.get("type") not in ("merkle_summary", "tail"):
                self._leaves.append(leaf_hash(bb))
            h = hmac.new(self.hmac_key, prev + bb, hashlib.sha256).digest()
            prev_hmacs.append(prev)
            body_bytes_list.append(bb)
            hmac_list.append(h)
            prev = h
        self.prev_hmac = prev

        # Step 2: sign all HMACs in parallel
        assert self.f is not None
        executor = self._executor
        if executor is not None and len(hmac_list) > 1:
            futures = [executor.submit(self.signing_key.sign, h) for h in hmac_list]
            sigs = [f.result() for f in futures]
        else:
            sigs = [self.signing_key.sign(h) for h in hmac_list]

        # Step 3: assemble wrappers and write to disk sequentially
        for ph, bb, h, sig, body in zip(prev_hmacs, body_bytes_list, hmac_list, sigs, bodies):
            wrapper = {
                "body": body,
                "prev_hmac": ph.hex(),
                "hmac": h.hex(),
                "sig": "ed25519:" + sig.hex(),
            }
            wrapper_bytes = canonical_json(wrapper)
            self.f.write(struct.pack(">I", len(wrapper_bytes)))
            self.f.write(wrapper_bytes)
        self.f.flush()

    def write_capability(
        self,
        name: str,
        *,
        mandatory: bool = False,
        params: Optional[dict] = None,
    ) -> None:
        """Emit a capability frame declaring an extension this writer relied on.

        See ``spec/sbtrace-v1.md`` §6.2. ``name`` MUST be a non-empty
        string. ``mandatory=True`` instructs readers to fail closed if
        they do not implement ``name``; ``mandatory=False`` is advisory
        and unknown names MAY be ignored. ``params`` is an optional
        extension-specific JSON object.

        When ``compression=True`` the frame is buffered alongside
        pending step frames so on-disk relative order matches the call
        order of ``write_capability`` and ``write_step``.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("capability name must be a non-empty string")
        if not isinstance(mandatory, bool):
            raise TypeError("capability mandatory must be a bool")
        if params is not None and not isinstance(params, dict):
            raise TypeError("capability params must be a dict or None")
        body: dict = {
            "type": "capability",
            "name": name,
            "mandatory": mandatory,
        }
        if params is not None:
            body["params"] = json.loads(canonical_json(params).decode("utf-8"))
        if self.batch_sign and not self.compression:
            # In streaming batch mode, buffer the capability with the batch
            self._streaming_batch.append(body)
            return
        if not self.compression:
            self._write_frame(body)
            return
        self.pending_items.append(("capability", body))

    def write_step(self, step: dict) -> None:
        if self.batch_sign and not self.compression:
            # Streaming batch mode: buffer steps and flush at interval
            body = {"type": "step", "step": step}
            self._streaming_batch.append(body)
            if (
                self.batch_sign_interval > 0
                and len(self._streaming_batch) >= self.batch_sign_interval
            ):
                self.flush()
            return
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
        self.pending_items.append(("step", snapshot))

    def _flush_pending(self) -> None:
        if not self.pending_steps and not self.pending_items:
            return
        # Pre-pass: count occurrences of every interior dict/list across
        # all pending steps. (Capability params are not interned — they
        # are typically tiny and conceptually part of the negotiation
        # surface, not bulk payload.)
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
        for kind, payload in self.pending_items:
            if kind == "capability":
                self._write_frame(payload)
                continue
            s = payload
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
        self.pending_items = []

    def _flush_pending_batch(self) -> None:
        """Like _flush_pending but uses parallel batch signing for the frame list."""
        if not self.pending_steps and not self.pending_items:
            return
        counts: dict = {}
        raws: dict = {}
        for s in self.pending_steps:
            _walk_candidates(s, self.blob_threshold, counts, raws)
        intern_set = {
            d for d, n in counts.items() if n >= self.blob_min_reuse
        }
        bodies: List[dict] = []
        # Blob frames first
        for digest in sorted(intern_set):
            if digest in self.seen_blobs:
                continue
            self.seen_blobs.add(digest)
            raw = raws[digest]
            gz = gzip.compress(raw, compresslevel=9, mtime=0)
            b64_gz = base64.b64encode(gz).decode("ascii")
            if len(b64_gz) + 24 < len(raw):
                bodies.append({"type": "blob", "id": digest, "encoding": "gzip+base64", "data": b64_gz})
            else:
                bodies.append({"type": "blob", "id": digest, "encoding": "json", "data": raw.decode("utf-8")})
        # Step and capability frames
        for kind, payload in self.pending_items:
            if kind == "capability":
                bodies.append(payload)
                continue
            s = payload
            interned = _apply_intern(s, self.blob_threshold, intern_set)
            raw = canonical_json(interned)
            gz = gzip.compress(raw, compresslevel=9, mtime=0)
            b64_gz = base64.b64encode(gz).decode("ascii")
            if len(b64_gz) + 24 < len(raw):
                bodies.append({"type": "step", "encoding": "gzip+base64", "data": b64_gz})
            else:
                bodies.append({"type": "step", "step": interned})
        self._write_frames_batch(bodies)
        self.pending_steps = []
        self.pending_items = []

    def flush(self) -> None:
        """Flush buffered frames to disk.

        In ``compression=False, batch_sign=True`` (streaming batch) mode,
        writes the current ``_streaming_batch`` to disk using parallel
        Ed25519 signing, then clears the buffer.

        In all other modes this is a no-op (``compression=True`` mode
        defers all writes to ``close()``).
        """
        if not self.batch_sign or self.compression:
            return
        if not self._streaming_batch:
            return
        self._write_frames_batch(self._streaming_batch)
        self._streaming_batch = []

    def close(self) -> None:
        if self.f and not self.f.closed:
            # Flush streaming batch if in batch+no-compression mode
            if self.batch_sign and not self.compression and self._streaming_batch:
                self._write_frames_batch(self._streaming_batch)
                self._streaming_batch = []
            # For compression=True + batch_sign=True, _flush_pending() handles the batch
            if self.compression and self.batch_sign:
                self._flush_pending_batch()
            else:
                self._flush_pending()
            if self.emit_merkle_summary:
                root = merkle_root(self._leaves)
                self._write_frame(
                    {
                        "type": "merkle_summary",
                        "scheme": MERKLE_SCHEME,
                        "algorithm": "sha256",
                        "leaf_count": len(self._leaves),
                        "merkle_root": root.hex(),
                    }
                )
            self._write_frame({"type": "tail", "wallclock_ns": time.time_ns()})
            self.f.close()
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None
