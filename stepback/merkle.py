"""RFC 6962-style Merkle tree for `.sb` trace summary frames.

Used by the trace writer to emit a single ``merkle_summary`` frame
just before ``tail``: a compact, position-independent commitment to
every other content frame's body. The HMAC chain pins frame
*ordering* (every frame depends on its predecessor); the Merkle root
pins frame *content* (every body bytes is committed to in a balanced
tree of fixed depth ⌈log₂ N⌉). Attestation packs can carry the
Merkle root alone and a regulator can later check, against the full
``.sb`` file, that the trace they're auditing is the one the attestor
signed without redoing the entire HMAC walk.

Construction follows RFC 6962 §2.1 ("Certificate Transparency"):

* leaf hash:  ``SHA-256(0x00 || body_bytes)``
* node hash:  ``SHA-256(0x01 || left || right)``
* an unpaired node at any level is *promoted* (carried up unchanged),
  not duplicated — this gives second-preimage resistance against a
  shorter list.
* the root of an empty list is ``SHA-256(b"")`` (the same convention
  RFC 6962 uses for its base case).

The leaf-domain prefix (``0x00``) and node-domain prefix (``0x01``)
are essential: without them, an internal node's pair-of-hashes is
indistinguishable from a leaf whose body happens to be the
concatenation of those two hashes, which would let an attacker
substitute a forged subtree for a real leaf. RFC 6962 calls this the
"second-preimage" defence.

This module deliberately does **not** depend on any other stepback
module — it is pure ``hashlib``. The same primitive lives in the
Rust ``sb-format`` crate (Step 40 of the standardization roadmap)
under the same RFC 6962 conventions, so the Merkle root in a
Python-written trace agrees with the root the Rust verifier would
compute over the same frames byte-for-byte.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, List, Sequence

LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"


def leaf_hash(body_bytes: bytes) -> bytes:
    """RFC 6962 leaf hash: ``SHA-256(0x00 || body_bytes)``."""
    return hashlib.sha256(LEAF_PREFIX + body_bytes).digest()


def node_hash(left: bytes, right: bytes) -> bytes:
    """RFC 6962 internal-node hash: ``SHA-256(0x01 || left || right)``."""
    return hashlib.sha256(NODE_PREFIX + left + right).digest()


def merkle_root(leaves: Sequence[bytes]) -> bytes:
    """Compute the RFC 6962 Merkle root over already-hashed leaves.

    Each ``leaves[i]`` MUST be a 32-byte digest produced by
    :func:`leaf_hash` (callers are responsible for the leaf-domain
    prefix). For the empty list the root is ``SHA-256(b"")``.

    Unpaired nodes at any level are promoted unchanged (RFC 6962),
    not duplicated — this prevents a shorter-list second-preimage.
    """
    if not leaves:
        return hashlib.sha256(b"").digest()
    level: List[bytes] = list(leaves)
    while len(level) > 1:
        nxt: List[bytes] = []
        i = 0
        while i + 1 < len(level):
            nxt.append(node_hash(level[i], level[i + 1]))
            i += 2
        if i < len(level):
            # Unpaired tail: promote unchanged.
            nxt.append(level[i])
        level = nxt
    return level[0]


def merkle_root_from_bodies(body_bytes_iter: Iterable[bytes]) -> bytes:
    """Convenience: hash each body with :func:`leaf_hash` then fold."""
    return merkle_root([leaf_hash(b) for b in body_bytes_iter])


__all__ = [
    "LEAF_PREFIX",
    "NODE_PREFIX",
    "leaf_hash",
    "node_hash",
    "merkle_root",
    "merkle_root_from_bodies",
]
