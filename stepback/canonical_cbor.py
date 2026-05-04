"""Experimental deterministic CBOR encoding (RFC 8949 §4.2).

This module is **experimental** and is **not** part of the v1
``.sb`` trace format. v1 traces are canonical UTF-8 JSON
(``stepback.canonical.canonical_json``); CBOR is a candidate
encoding for ``format_version=2`` (see ``100_STEPS.md`` step 43–44).

The encoder implements RFC 8949 §4.2 "Core Deterministic Encoding":

* Definite-length encoding only (no indefinite items).
* Integers, byte strings, text strings, arrays, and maps use the
  shortest of the 1/2/3/5/9-byte length-prefix forms that fits.
* Map keys are sorted by bytewise lexicographic order of their
  CBOR-encoded form ("bytewise lexicographic ordering of deterministic
  encodings", RFC 8949 §4.2.1).
* Floats use the shortest of half (f16), single (f32), or double (f64)
  that preserves the exact value. NaNs are encoded as 0xf97e00.
  ``-0.0`` is preserved (it is *not* folded to ``+0.0``).
* No tags are emitted by default; ``encode`` rejects types it does not
  understand rather than silently widening.

Type mapping (Python → CBOR):

================  =========================================
Python type       CBOR
================  =========================================
``bool``          simple value 20 / 21 (false / true)
``None``          simple value 22 (null)
``int``           major 0 (unsigned) or major 1 (negative)
``float``         major 7, shortest IEEE 754 round-trip
``str``           major 3 (text string), UTF-8
``bytes``,
``bytearray``     major 2 (byte string)
``list``,
``tuple``         major 4 (array)
``dict``          major 5 (map), keys sorted by encoded bytes
``frozenset``,
``set``           major 4 (array), elements sorted (canonical
                  JSON parity)
================  =========================================

The encoder explicitly rejects:

* ``float('nan')`` is permitted (encoded as canonical ``0xf97e00``)
  but ``allow_nan=False`` (default) rejects it; this matches
  ``canonical_json``'s default policy.
* Integers outside ``[-2**64, 2**64 - 1]`` (CBOR's representable
  range without bignum tags 2/3) raise ``OverflowError``. Bignum
  tags are deliberately out of scope for the experimental encoder.
* Any other type not listed above raises ``TypeError``.

Because this encoder is experimental, none of its outputs feed
into v1 trace hashing today. The reference Python implementation
exists so that:

1. Test fixtures can pin v2-candidate byte sequences before any
   ``.sb`` v2 reader/writer ships.
2. Independent implementations (Rust, TypeScript, Go, JVM, .NET)
   can be cross-checked against the same fixtures.
3. The semantic-hash equivalence claim that motivates ``.sb`` v2
   (RFC 8949 deterministic CBOR and our canonical JSON should
   agree on the *semantic* hash for the data shapes both support)
   has a concrete reference to differentially fuzz against.
"""
from __future__ import annotations

import hashlib
import struct
from typing import Any

CBOR_CANONICALISATION_VERSION = "2-cbor-experimental"

_MT_UINT = 0
_MT_NINT = 1
_MT_BYTES = 2
_MT_TEXT = 3
_MT_ARRAY = 4
_MT_MAP = 5
_MT_SIMPLE = 7

_SIMPLE_FALSE = 20
_SIMPLE_TRUE = 21
_SIMPLE_NULL = 22


def _head(major: int, arg: int) -> bytes:
    """Encode the CBOR head byte(s) for ``major`` with argument ``arg``.

    Uses the *shortest* of the 1, 2, 3, 5, or 9-byte representations
    that fits ``arg``, per RFC 8949 §4.2.1.
    """
    if arg < 0:
        raise ValueError("argument must be non-negative")
    mt = major << 5
    if arg < 24:
        return bytes([mt | arg])
    if arg < 0x100:
        return bytes([mt | 24, arg])
    if arg < 0x10000:
        return bytes([mt | 25]) + arg.to_bytes(2, "big")
    if arg < 0x100000000:
        return bytes([mt | 26]) + arg.to_bytes(4, "big")
    if arg < 0x10000000000000000:
        return bytes([mt | 27]) + arg.to_bytes(8, "big")
    raise OverflowError("CBOR argument exceeds 2**64 - 1; bignum tags are not supported")


def _encode_int(n: int) -> bytes:
    if n >= 0:
        return _head(_MT_UINT, n)
    encoded = -1 - n
    if encoded >= 0x10000000000000000:
        raise OverflowError("CBOR negative integer below -2**64; bignum tags are not supported")
    return _head(_MT_NINT, encoded)


def _encode_float(f: float, *, allow_nan: bool) -> bytes:
    """Encode ``f`` per RFC 8949 §4.2.2: shortest of f16/f32/f64
    that preserves the exact value, with NaN normalised to 0xf97e00.
    """
    if f != f:
        if not allow_nan:
            raise ValueError("NaN is not allowed under allow_nan=False")
        return b"\xf9\x7e\x00"
    if f == float("inf"):
        return b"\xf9\x7c\x00"
    if f == float("-inf"):
        return b"\xf9\xfc\x00"
    f64 = struct.pack(">d", f)
    f32_bytes = struct.pack(">f", f)
    if struct.unpack(">f", f32_bytes)[0] == f and struct.pack(">d", struct.unpack(">f", f32_bytes)[0]) == f64:
        f32_val = struct.unpack(">f", f32_bytes)[0]
        f16 = _try_encode_half(f32_val)
        if f16 is not None:
            return b"\xf9" + f16
        return b"\xfa" + f32_bytes
    return b"\xfb" + f64


def _try_encode_half(f: float) -> bytes | None:
    """Return 2 bytes if ``f`` is exactly representable as IEEE 754
    binary16, else ``None``. Preserves -0.0.
    """
    f32_bytes = struct.pack(">f", f)
    bits = struct.unpack(">I", f32_bytes)[0]
    sign = (bits >> 31) & 0x1
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    if exp == 0xFF:
        return None
    if exp == 0 and mant == 0:
        return struct.pack(">H", sign << 15)
    e = exp - 127
    if -14 <= e <= 15 and (mant & ((1 << 13) - 1)) == 0:
        half_mant = mant >> 13
        half_exp = e + 15
        return struct.pack(">H", (sign << 15) | (half_exp << 10) | half_mant)
    if -24 <= e < -14:
        full_mant = (1 << 23) | mant
        shift = -e - 14 + 13
        if shift >= 25:
            return None
        if (full_mant & ((1 << shift) - 1)) == 0:
            sub_mant = full_mant >> shift
            return struct.pack(">H", (sign << 15) | sub_mant)
    return None


def _encode_simple(value: int) -> bytes:
    if value < 24:
        return bytes([(_MT_SIMPLE << 5) | value])
    if value < 0x100:
        return bytes([(_MT_SIMPLE << 5) | 24, value])
    raise ValueError(f"simple value {value} out of range")


def _encode_value(obj: Any, *, allow_nan: bool) -> bytes:
    if obj is True:
        return _encode_simple(_SIMPLE_TRUE)
    if obj is False:
        return _encode_simple(_SIMPLE_FALSE)
    if obj is None:
        return _encode_simple(_SIMPLE_NULL)
    if isinstance(obj, int) and not isinstance(obj, bool):
        return _encode_int(obj)
    if isinstance(obj, float):
        return _encode_float(obj, allow_nan=allow_nan)
    if isinstance(obj, str):
        data = obj.encode("utf-8", errors="strict")
        return _head(_MT_TEXT, len(data)) + data
    if isinstance(obj, (bytes, bytearray)):
        data = bytes(obj)
        return _head(_MT_BYTES, len(data)) + data
    if isinstance(obj, (list, tuple)):
        parts = [_encode_value(item, allow_nan=allow_nan) for item in obj]
        return _head(_MT_ARRAY, len(parts)) + b"".join(parts)
    if isinstance(obj, (set, frozenset)):
        try:
            ordered = sorted(obj)
        except TypeError:
            ordered = sorted(obj, key=repr)
        parts = [_encode_value(item, allow_nan=allow_nan) for item in ordered]
        return _head(_MT_ARRAY, len(parts)) + b"".join(parts)
    if isinstance(obj, dict):
        encoded_items = []
        for k, v in obj.items():
            if isinstance(k, bool) or not isinstance(k, (str, int, bytes, bytearray)):
                raise TypeError(
                    f"CBOR map keys must be str, int, or bytes; got {type(k).__name__}"
                )
            kb = _encode_value(k, allow_nan=allow_nan)
            vb = _encode_value(v, allow_nan=allow_nan)
            encoded_items.append((kb, vb))
        encoded_items.sort(key=lambda pair: pair[0])
        for i in range(1, len(encoded_items)):
            if encoded_items[i - 1][0] == encoded_items[i][0]:
                raise ValueError("duplicate map keys after canonical encoding")
        return _head(_MT_MAP, len(encoded_items)) + b"".join(k + v for k, v in encoded_items)
    raise TypeError(f"Object of type {type(obj).__name__} is not CBOR-canonicalisable")


def canonical_cbor(obj: Any, *, allow_nan: bool = False) -> bytes:
    """Return RFC 8949 §4.2 deterministic CBOR bytes for ``obj``.

    This encoder is **experimental** and not used by ``.sb`` v1
    trace writing or hashing. See the module docstring for the
    type mapping and the rules enforced.
    """
    return _encode_value(obj, allow_nan=allow_nan)


def sha256_hex_cbor(obj: Any, *, allow_nan: bool = False) -> str:
    """Convenience: canonical-CBOR encode and sha256 in one shot."""
    return "sha256:" + hashlib.sha256(canonical_cbor(obj, allow_nan=allow_nan)).hexdigest()


__all__ = [
    "CBOR_CANONICALISATION_VERSION",
    "canonical_cbor",
    "sha256_hex_cbor",
]
