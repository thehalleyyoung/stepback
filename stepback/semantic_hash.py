"""Encoding-agnostic semantic hashing for SB-Trace v2.

v1 caches and content-addresses by ``sha256(canonical_json(obj))`` —
the cache key is *bytes-derived*. That is acceptable when only one
wire encoding exists, but it locks the format to canonical JSON: the
moment a v2 reader/writer wants to use deterministic CBOR (RFC 8949
§4.2) the cache keys diverge, dirty-set propagation across mixed-
encoding traces breaks, and the soundness theorem in
``docs/dirty-set.md`` no longer applies across encodings.

This module defines :func:`semantic_hash`: a hash that depends only
on the *abstract value tree* and **not** on which wire encoding was
used to transmit it. Canonical-JSON bytes and deterministic-CBOR
bytes that decode to the same abstract tree produce the same
``semantic_hash``. That is the core invariant for SB-Trace v2's
"dual encoding, single semantic hash" claim (see
``spec/sbtrace-v2.md``).

The implementation is a Merkle-style fold over the value tree. Each
leaf and each composite node is tagged with a fixed ASCII type prefix
before being mixed into SHA-256. The prefixes are part of the
spec; changing them is a wire-incompatible change and would require
bumping ``SEMANTIC_HASH_VERSION``.

::

    null    →  H("n:")
    bool    →  H("b:0") | H("b:1")
    int     →  H("i:<decimal>")
    float   →  H("f:<canonical-decimal>")
    str     →  H("s:<utf8 bytes>")
    bytes   →  H("y:<raw bytes>")
    list    →  H("l:" + len(decimal) + ":" + concat(child_hashes_raw))
    map     →  H("m:" + len(decimal) + ":" + concat(
                    key_hash_raw + val_hash_raw
                    for (key_hash, val_hash) in
                        sorted by key_hash bytewise))

The map ordering is *bytewise on the hash of the key*, not on the
key text or its CBOR encoding. That is what lets canonical-JSON
(which sorts by code-point of UTF-8 strings) and deterministic-CBOR
(which sorts by bytewise lexicographic order of the encoded key
bytes) agree on the same map hash: the ordering used here is a
property of the abstract tree, not of either wire encoding.

Float canonicalisation uses ``repr()`` for normal numbers (which is
round-trip exact for IEEE 754 binary64 in CPython >=3.1) plus
explicit tags for the IEEE corner cases NaN, +Inf, -Inf, +0.0, and
-0.0. The negative-zero and NaN cases are pinned because both wire
encoders preserve them (canonical JSON via ``allow_nan=False`` —
NaN is rejected by default — and deterministic CBOR via the
0xf97e00 NaN encoding); ``semantic_hash`` therefore must do the
same to remain a faithful function of the abstract tree.
"""
from __future__ import annotations

import hashlib
import math
import struct
from typing import Any, Iterable

#: Bumped on any change to the tagging or ordering rules above.
#: The ``.sb`` v2 header records this alongside ``format_version=2``
#: and ``canonicalisation_version`` so a reader can refuse a trace
#: whose semantic-hash dialect it does not implement.
SEMANTIC_HASH_VERSION = "1"


_TAG_NULL = b"n:"
_TAG_BOOL_FALSE = b"b:0"
_TAG_BOOL_TRUE = b"b:1"
_TAG_INT = b"i:"
_TAG_FLOAT = b"f:"
_TAG_STR = b"s:"
_TAG_BYTES = b"y:"
_TAG_LIST = b"l:"
_TAG_MAP = b"m:"


def _hash(parts: Iterable[bytes]) -> bytes:
    """SHA-256 over the concatenation of ``parts``, returned as raw 32 bytes."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()


def _canonical_float(f: float) -> bytes:
    """Encoding-agnostic canonical text form of a float.

    ``repr(f)`` is round-trip exact for IEEE 754 binary64 in CPython,
    so two floats with the same bit pattern get the same bytes.
    NaN / Inf / -Inf / -0.0 are pinned explicitly because ``repr``
    of NaN is platform-stable as ``"nan"`` but we want the spec to
    name them.
    """
    if math.isnan(f):
        return b"nan"
    if f == float("inf"):
        return b"+inf"
    if f == float("-inf"):
        return b"-inf"
    if f == 0.0:
        # Distinguish +0.0 and -0.0 by their bit pattern — both
        # canonical JSON (when emitted) and deterministic CBOR
        # preserve the sign of zero, and the abstract tree must
        # follow suit.
        sign_bit = struct.pack(">d", f)[0] & 0x80
        return b"-0" if sign_bit else b"+0"
    return repr(f).encode("ascii")


def _hash_value(obj: Any) -> bytes:
    """Return the raw 32-byte semantic hash of ``obj``."""
    if obj is None:
        return _hash([_TAG_NULL])
    if obj is True:
        return _hash([_TAG_BOOL_TRUE])
    if obj is False:
        return _hash([_TAG_BOOL_FALSE])
    if isinstance(obj, int) and not isinstance(obj, bool):
        return _hash([_TAG_INT, str(obj).encode("ascii")])
    if isinstance(obj, float):
        return _hash([_TAG_FLOAT, _canonical_float(obj)])
    if isinstance(obj, str):
        return _hash([_TAG_STR, obj.encode("utf-8")])
    if isinstance(obj, (bytes, bytearray)):
        return _hash([_TAG_BYTES, bytes(obj)])
    if isinstance(obj, (list, tuple)):
        children = [_hash_value(item) for item in obj]
        return _hash(
            [_TAG_LIST, str(len(children)).encode("ascii"), b":", *children]
        )
    if isinstance(obj, (set, frozenset)):
        # Sets are not part of the SB-Trace abstract data model
        # (neither canonical JSON nor canonical CBOR has a native
        # set type). To keep ``semantic_hash`` well-defined for the
        # same shapes the wire encoders accept, treat them as
        # sorted lists — this matches ``stepback.canonical`` and
        # ``stepback.canonical_cbor``'s reductions.
        try:
            ordered = sorted(obj)
        except TypeError:
            ordered = sorted(obj, key=repr)
        children = [_hash_value(item) for item in ordered]
        return _hash(
            [_TAG_LIST, str(len(children)).encode("ascii"), b":", *children]
        )
    if isinstance(obj, dict):
        items: list[tuple[bytes, bytes]] = []
        seen: set[bytes] = set()
        for k, v in obj.items():
            kh = _hash_value(k)
            if kh in seen:
                raise ValueError(
                    "duplicate map keys after semantic hashing — refusing to "
                    "produce a hash that is not a function of the abstract tree"
                )
            seen.add(kh)
            items.append((kh, _hash_value(v)))
        items.sort(key=lambda pair: pair[0])
        body: list[bytes] = [_TAG_MAP, str(len(items)).encode("ascii"), b":"]
        for kh, vh in items:
            body.append(kh)
            body.append(vh)
        return _hash(body)
    raise TypeError(
        f"Object of type {type(obj).__name__} has no SB-Trace v2 semantic-hash "
        f"representation; allowed types are None, bool, int, float, str, bytes, "
        f"list, tuple, set/frozenset, and dict"
    )


def semantic_hash(obj: Any) -> str:
    """Return the encoding-agnostic SB-Trace v2 semantic hash.

    The result is a string of the form ``"sha256:<lowercase hex>"``,
    matching the existing v1 hash discriminator in
    :func:`stepback.canonical.sha256_hex`. The bytes that go into
    SHA-256, however, are derived from a fold over the abstract
    value tree (see this module's docstring), not from canonical
    JSON or canonical CBOR bytes.

    Two objects produce the same ``semantic_hash`` iff they are
    equal as abstract trees under the SB-Trace v2 data model:

    * ``None``, booleans, ints, floats (with NaN, +0.0, -0.0
      distinguished), text strings, and byte strings compare by
      value.
    * Lists and tuples compare element-wise, in order.
    * Dicts compare as multisets of ``(key, value)`` pairs — key
      ordering is irrelevant, duplicate keys are rejected.
    * Sets and frozensets compare as sorted-list reductions, the
      same shape both wire encoders use.
    """
    return "sha256:" + _hash_value(obj).hex()


__all__ = [
    "SEMANTIC_HASH_VERSION",
    "semantic_hash",
]
