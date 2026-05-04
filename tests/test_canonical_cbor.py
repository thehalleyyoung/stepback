"""Tests for the experimental RFC 8949 deterministic CBOR encoder.

Implements step 43 of ``100_STEPS.md``:

    Add experimental deterministic CBOR encoding following RFC 8949
    canonical rules; keep canonical JSON as v1.

The vectors below are pinned against RFC 8949 Appendix A
("Examples of Encoded CBOR Data Items") *plus* the §4.2
"Core Deterministic Encoding" rules. They lock in the byte
sequences before any ``.sb`` v2 reader/writer ships.
"""
from __future__ import annotations

import math
import struct

import pytest

from stepback.canonical_cbor import (
    CBOR_CANONICALISATION_VERSION,
    canonical_cbor,
    sha256_hex_cbor,
)


def _hx(b: bytes) -> str:
    return b.hex()


# --- Section: small integers (RFC 8949 Appendix A) ------------------
def test_small_uint_zero():
    assert canonical_cbor(0) == b"\x00"


def test_small_uint_one():
    assert canonical_cbor(1) == b"\x01"


def test_small_uint_ten():
    assert canonical_cbor(10) == b"\x0a"


def test_small_uint_twentythree():
    assert canonical_cbor(23) == b"\x17"


def test_uint_24_uses_one_byte_arg():
    assert canonical_cbor(24) == b"\x18\x18"


def test_uint_25_uses_one_byte_arg():
    assert canonical_cbor(25) == b"\x18\x19"


def test_uint_100_one_byte_arg():
    assert canonical_cbor(100) == b"\x18\x64"


def test_uint_1000_two_byte_arg():
    assert canonical_cbor(1000) == b"\x19\x03\xe8"


def test_uint_1_000_000_four_byte_arg():
    assert canonical_cbor(1_000_000) == b"\x1a\x00\x0f\x42\x40"


def test_uint_1e12_eight_byte_arg():
    # RFC 8949 A: 1_000_000_000_000 → 0x1b000000e8d4a51000
    assert canonical_cbor(1_000_000_000_000) == bytes.fromhex("1b000000e8d4a51000")


def test_uint_max_2_64_minus_1():
    assert canonical_cbor(2**64 - 1) == bytes.fromhex("1bffffffffffffffff")


def test_uint_overflow_2_64_rejected():
    with pytest.raises(OverflowError):
        canonical_cbor(2**64)


def test_negative_one():
    # -1 encodes as major 1, arg 0.
    assert canonical_cbor(-1) == b"\x20"


def test_negative_ten():
    assert canonical_cbor(-10) == b"\x29"


def test_negative_one_hundred():
    assert canonical_cbor(-100) == b"\x38\x63"


def test_negative_one_thousand():
    assert canonical_cbor(-1000) == b"\x39\x03\xe7"


def test_negative_min_neg_2_64():
    assert canonical_cbor(-(2**64)) == bytes.fromhex("3bffffffffffffffff")


def test_negative_overflow_below_min_rejected():
    with pytest.raises(OverflowError):
        canonical_cbor(-(2**64) - 1)


# --- Section: shortest-form length encoding -------------------------
def test_uint_smallest_form_is_used_for_24():
    # 24 must NOT be encoded as 0x1900 18 (two-byte arg) nor 0x1a 0000 0018.
    assert canonical_cbor(24) == b"\x18\x18"


def test_uint_smallest_form_is_used_for_255():
    assert canonical_cbor(255) == b"\x18\xff"


def test_uint_smallest_form_is_used_for_256():
    assert canonical_cbor(256) == b"\x19\x01\x00"


def test_uint_smallest_form_is_used_for_65535():
    assert canonical_cbor(65535) == b"\x19\xff\xff"


def test_uint_smallest_form_is_used_for_65536():
    assert canonical_cbor(65536) == b"\x1a\x00\x01\x00\x00"


# --- Section: simples / null / bool ---------------------------------
def test_false_encodes_to_f4():
    assert canonical_cbor(False) == b"\xf4"


def test_true_encodes_to_f5():
    assert canonical_cbor(True) == b"\xf5"


def test_none_encodes_to_f6():
    assert canonical_cbor(None) == b"\xf6"


# --- Section: floats (RFC 8949 §4.2.2) ------------------------------
def test_float_zero_uses_half():
    # 0.0 is exactly representable as half float 0x0000.
    assert canonical_cbor(0.0) == b"\xf9\x00\x00"


def test_float_neg_zero_preserved_as_half():
    assert canonical_cbor(-0.0) == b"\xf9\x80\x00"


def test_float_one_uses_half():
    assert canonical_cbor(1.0) == b"\xf9\x3c\x00"


def test_float_one_point_five_uses_half():
    assert canonical_cbor(1.5) == b"\xf9\x3e\x00"


def test_float_65504_max_half():
    assert canonical_cbor(65504.0) == b"\xf9\x7b\xff"


def test_float_100000_uses_single():
    # 100000.0 is exact in f32 but not in f16.
    assert canonical_cbor(100000.0) == b"\xfa\x47\xc3\x50\x00"


def test_float_pi_uses_double():
    pi = 3.1415926535897932
    encoded = canonical_cbor(pi)
    assert encoded[0:1] == b"\xfb"
    assert struct.unpack(">d", encoded[1:])[0] == pi


def test_float_third_uses_double():
    # 1/3 is not exactly representable in any IEEE 754 width;
    # the f64 value is the canonical "input value", and shorter
    # widths do not round-trip — must use f64.
    encoded = canonical_cbor(1.0 / 3.0)
    assert encoded[0:1] == b"\xfb"


def test_float_inf_uses_half():
    assert canonical_cbor(float("inf")) == b"\xf9\x7c\x00"


def test_float_neg_inf_uses_half():
    assert canonical_cbor(float("-inf")) == b"\xf9\xfc\x00"


def test_float_nan_rejected_by_default():
    with pytest.raises(ValueError):
        canonical_cbor(float("nan"))


def test_float_nan_canonical_when_allowed():
    assert canonical_cbor(float("nan"), allow_nan=True) == b"\xf9\x7e\x00"


def test_float_subnormal_half_5p9605e_minus8():
    # Smallest positive subnormal half: 2**-24 ≈ 5.9604644775390625e-08 → 0xf90001
    assert canonical_cbor(2.0**-24) == b"\xf9\x00\x01"


# --- Section: byte strings ------------------------------------------
def test_bytes_empty():
    assert canonical_cbor(b"") == b"\x40"


def test_bytes_four():
    assert canonical_cbor(b"\x01\x02\x03\x04") == b"\x44\x01\x02\x03\x04"


def test_bytes_uses_smallest_length():
    long_bytes = b"\x00" * 25
    encoded = canonical_cbor(long_bytes)
    # major 2 (0x40) | additional info 24 (0x18) | length 25 | payload
    assert encoded[0] == 0x58
    assert encoded[1] == 25
    assert encoded[2:] == long_bytes


def test_bytearray_treated_like_bytes():
    assert canonical_cbor(bytearray(b"\x01\x02")) == canonical_cbor(b"\x01\x02")


# --- Section: text strings ------------------------------------------
def test_text_empty():
    assert canonical_cbor("") == b"\x60"


def test_text_a():
    assert canonical_cbor("a") == b"\x61\x61"


def test_text_iuf_per_rfc():
    # RFC 8949 A: "IETF" → 0x6449455446
    assert canonical_cbor("IETF") == b"\x64IETF"


def test_text_quote_backslash():
    # RFC 8949 A: "\"\\" → 0x62225c
    assert canonical_cbor('"\\') == b"\x62\x22\x5c"


def test_text_unicode_lambda():
    # U+00FC ü is 2 bytes in UTF-8: 0xc3bc; per RFC, "\u00fc" → 0x62c3bc
    assert canonical_cbor("\u00fc") == b"\x62\xc3\xbc"


def test_text_unicode_water_kanji():
    # U+6C34 水 → UTF-8 0xe6b0b4: encoded text-string is 0x63e6b0b4
    assert canonical_cbor("\u6c34") == b"\x63\xe6\xb0\xb4"


def test_text_emoji_outside_bmp():
    # U+1F600 grinning face → UTF-8 0xf09f9880 (4 bytes), text-string 0x64f09f9880
    assert canonical_cbor("\U0001f600") == b"\x64\xf0\x9f\x98\x80"


def test_text_lone_surrogate_rejected():
    with pytest.raises(UnicodeEncodeError):
        canonical_cbor("\ud800")


# --- Section: arrays ------------------------------------------------
def test_array_empty():
    assert canonical_cbor([]) == b"\x80"


def test_array_123():
    assert canonical_cbor([1, 2, 3]) == b"\x83\x01\x02\x03"


def test_array_25_smallest_length_form():
    arr = list(range(1, 26))
    encoded = canonical_cbor(arr)
    # major 4 (0x80) | 24 → 0x98; followed by 1-byte length 25; then bodies.
    assert encoded[0] == 0x98
    assert encoded[1] == 25


def test_tuple_treated_as_array():
    assert canonical_cbor((1, 2, 3)) == canonical_cbor([1, 2, 3])


def test_array_nested_per_rfc():
    # RFC 8949 A: [1, [2, 3], [4, 5]] → 0x8301820203820405
    assert canonical_cbor([1, [2, 3], [4, 5]]) == bytes.fromhex("8301820203820405")


# --- Section: maps and key sorting (RFC 8949 §4.2.1) ---------------
def test_map_empty():
    assert canonical_cbor({}) == b"\xa0"


def test_map_single_entry():
    # {1: 2} → 0xa10102
    assert canonical_cbor({1: 2}) == b"\xa1\x01\x02"


def test_map_keys_sorted_bytewise_lex_of_encoding():
    # Per RFC 8949 §4.2.1, sort keys by bytewise lex of *encoded* form.
    # Encoded keys: "a"→0x6161, "b"→0x6162, "aa"→0x626161.
    # Bytewise lex order on encodings: 0x6161 < 0x6162 < 0x626161
    # (because 0x61 < 0x62 at the first byte).
    out = canonical_cbor({"b": 2, "aa": 3, "a": 1})
    assert out == bytes.fromhex("a3" + "6161" + "01" + "6162" + "02" + "626161" + "03")


def test_map_keys_int_vs_text_sort_by_encoded_bytes():
    # Encoded "a"→0x6161; encoded 10 → 0x0a; bytewise: 0x0a < 0x61.
    # So {10: ..., "a": ...} must encode integer-key first.
    out = canonical_cbor({"a": 1, 10: 2})
    assert out.startswith(bytes.fromhex("a2"))
    assert out == bytes.fromhex("a2" + "0a" + "02" + "6161" + "01")


def test_map_short_then_longer_string_keys_sort_correctly():
    # "a"→0x6161, "ab"→0x626162. 0x6161 < 0x6261 (length byte makes "ab" larger).
    out = canonical_cbor({"ab": 2, "a": 1})
    assert out == bytes.fromhex("a2" + "6161" + "01" + "626162" + "02")


def test_map_duplicate_keys_after_encoding_rejected():
    # Python dicts can't naturally hold duplicate keys, but the encoder
    # still defends against constructed inputs (e.g., custom mapping
    # types) that present duplicate post-encoding keys.
    from stepback.canonical_cbor import _encode_value  # type: ignore

    class DupMap(dict):
        def items(self):  # type: ignore[override]
            return [("x", 1), ("x", 2)]

    with pytest.raises(ValueError):
        _encode_value(DupMap(), allow_nan=False)


def test_nested_map_sorting():
    out = canonical_cbor({"a": {"b": 2, "a": 1}})
    inner = bytes.fromhex("a2" + "6161" + "01" + "6162" + "02")
    assert out == b"\xa1" + b"\x61a" + inner


# --- Section: hash convenience and version --------------------------
def test_canonicalisation_version_label():
    assert "cbor" in CBOR_CANONICALISATION_VERSION


def test_sha256_hex_cbor_round_trip():
    obj = {"a": 1, "b": [2, 3]}
    h1 = sha256_hex_cbor(obj)
    assert h1.startswith("sha256:")
    assert len(h1) == len("sha256:") + 64
    # Key-order independence (canonical encoding).
    assert sha256_hex_cbor({"b": [2, 3], "a": 1}) == h1


def test_unknown_type_rejected():
    class Foo:
        pass

    with pytest.raises(TypeError):
        canonical_cbor(Foo())


def test_decimal_rejected_like_canonical_json():
    from decimal import Decimal

    with pytest.raises(TypeError):
        canonical_cbor(Decimal("1.0"))


def test_set_canonicalised_as_sorted_array():
    assert canonical_cbor({3, 1, 2}) == canonical_cbor([1, 2, 3])


def test_v1_unchanged_by_cbor_introduction():
    # Sanity: importing the experimental CBOR module must not perturb the
    # v1 canonical_json contract — we explicitly do not change it.
    from stepback.canonical import canonical_json, hash_obj, CANONICALISATION_VERSION

    assert CANONICALISATION_VERSION == "1"
    assert canonical_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    assert hash_obj({"a": 1}).startswith("sha256:")


def test_cbor_and_json_hashes_differ_for_same_object():
    # The encodings are different by construction; hashes must differ.
    from stepback.canonical import hash_obj

    obj = {"a": 1, "b": "x"}
    assert hash_obj(obj) != sha256_hex_cbor(obj)
