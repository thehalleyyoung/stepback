"""Canonicalisation + content hashing.

The canonicalisation form is intentionally simple for v0.1: UTF-8 JSON
with sorted keys and no whitespace separators. The header pins
``canonicalisation_version`` so future versions (CBOR, message-dict
deduplication, etc.) can be introduced without breaking old traces.

Whatever we feed into ``sha256_hex`` here is *exactly* what the replay
engine and the verifier feed in — that determinism is the entire
point of "LLM-aware caching": same input bytes → same hash → cache hit.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

CANONICALISATION_VERSION = "1"


def canonical_json(obj: Any) -> bytes:
    """Return canonical UTF-8 JSON bytes for ``obj``.

    Sorted keys, no whitespace, ``ensure_ascii=False`` so non-ASCII text
    survives a round trip without backslash-uXXXX expansion.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_default,
    ).encode("utf-8")


def _default(o: Any) -> Any:
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, (set, frozenset)):
        try:
            return sorted(o)
        except TypeError:
            return sorted(o, key=repr)
    if isinstance(o, (bytes, bytearray)):
        return {"__bytes_hex__": bytes(o).hex()}
    raise TypeError(f"Object of type {type(o).__name__} is not canonicalisable")


def sha256_hex(data: bytes) -> str:
    """sha256 of ``data`` in the ``sha256:<hex>`` form used in `.sb`."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def hash_obj(obj: Any) -> str:
    """Convenience: canonicalise and sha256 in one shot."""
    return sha256_hex(canonical_json(obj))
