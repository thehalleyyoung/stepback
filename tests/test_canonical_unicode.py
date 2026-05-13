"""Unicode-and-friends canonicalisation tests.

Implements step 28 of ``100_STEPS.md``:

    Add Unicode canonicalization tests for NFC/NFD, surrogate rejection,
    non-ASCII key ordering, decimals, and binary payload markers.

These tests *lock in* the v1 spec for ``canonical.py`` so any change
to its behaviour breaks tests rather than silently invalidating every
existing ``.sb`` trace's content hash.

The v1 contract being asserted here:

* Strings are encoded as UTF-8, ``ensure_ascii=False`` — non-ASCII
  characters appear as their UTF-8 bytes, **not** as ``\\uXXXX`` escapes.
* No Unicode normalisation is performed: NFC and NFD forms of "the same"
  string hash differently. Producers MUST agree on a normalisation form
  out-of-band; the canonicaliser does not silently fold them together.
* Lone surrogates (``U+D800..U+DFFF`` not part of a surrogate pair) are
  rejected with ``UnicodeEncodeError``; they are not legal in well-formed
  Unicode and there is no sound bytes representation in UTF-8.
* Object keys are sorted by Python's ``str`` order — i.e. by Unicode
  code-point. ASCII characters sort before non-ASCII; non-ASCII keys
  sort by raw code-point, **not** by locale or by NFKC-folded form.
* ``decimal.Decimal`` is *not* a supported input type today. It raises
  ``TypeError`` rather than silently widening to ``float`` (which would
  lose precision and break cross-implementation hashes).
* ``bytes`` and ``bytearray`` are encoded as the JSON object
  ``{"__bytes_hex__": "<lowercase-hex>"}``. This tag is the v1 binary
  payload marker; the field name and the hex casing are part of the
  contract.
* Non-finite floats (NaN, +Inf, -Inf) are rejected because they have no
  JSON representation.

If you change any of these behaviours, you MUST bump
``CANONICALISATION_VERSION`` and add a migration story.
"""
from __future__ import annotations

import decimal
import math
import unicodedata

import pytest

from stepback.canonical import (
    CANONICALISATION_VERSION,
    canonical_json,
    hash_obj,
    sha256_hex,
)


# ---------------------------------------------------------------------------
# NFC / NFD: no implicit normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "café",  # 'é' as either single or decomposed codepoint
        "Å",  # ANGSTROM SIGN vs LATIN CAPITAL LETTER A WITH RING ABOVE
        "한",  # Hangul precomposed vs Jamo decomposed
        "ﬁ",  # LATIN SMALL LIGATURE FI vs 'fi'
    ],
)
def test_nfc_and_nfd_hash_differently(raw: str) -> None:
    """The canonicaliser is byte-faithful: NFC ≠ NFD by content hash."""
    nfc = unicodedata.normalize("NFC", raw)
    nfd = unicodedata.normalize("NFD", raw)
    if nfc == nfd:
        pytest.skip(f"{raw!r}: NFC == NFD codepoint-wise; nothing to compare")
    assert canonical_json(nfc) != canonical_json(nfd)
    assert hash_obj(nfc) != hash_obj(nfd)


def test_nfkc_and_nfkd_also_distinct() -> None:
    raw = "ﬁ"  # ligature
    nfc = unicodedata.normalize("NFC", raw)
    nfkc = unicodedata.normalize("NFKC", raw)  # decomposes to "fi"
    assert nfc != nfkc
    assert hash_obj(nfc) != hash_obj(nfkc)


def test_nfc_string_is_utf8_not_ascii_escaped() -> None:
    """Non-ASCII survives as UTF-8 bytes, never ``\\uXXXX`` escapes."""
    out = canonical_json("café")
    assert out == b'"caf\xc3\xa9"'
    assert b"\\u" not in out


def test_normalisation_is_caller_responsibility_in_dict_keys() -> None:
    """Two visually identical keys (NFC vs NFD) are *different* keys."""
    nfc = unicodedata.normalize("NFC", "é")
    nfd = unicodedata.normalize("NFD", "é")
    assert nfc != nfd
    obj = {nfc: 1, nfd: 2}
    out = canonical_json(obj)
    # Both keys appear; we get a 2-entry object.
    assert out.count(b":") == 2
    assert hash_obj(obj) != hash_obj({nfc: 1})


# ---------------------------------------------------------------------------
# Surrogate rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lone",
    [
        "\ud800",  # high surrogate, alone
        "\udfff",  # low surrogate, alone
        "\udcff",  # surrogate-escape range used for invalid UTF-8 round-trip
        "ok\ud800tail",  # embedded mid-string
    ],
)
def test_lone_surrogates_rejected_in_strings(lone: str) -> None:
    with pytest.raises(UnicodeEncodeError):
        canonical_json(lone)


@pytest.mark.parametrize(
    "lone_key",
    [
        "\ud800",
        "\udfff",
    ],
)
def test_lone_surrogates_rejected_in_dict_keys(lone_key: str) -> None:
    with pytest.raises(UnicodeEncodeError):
        canonical_json({lone_key: 1})


def test_lone_surrogates_rejected_in_nested_lists() -> None:
    with pytest.raises(UnicodeEncodeError):
        canonical_json({"items": ["ok", ["nested", "\ud800"]]})


def test_well_formed_supplementary_codepoint_round_trips() -> None:
    """Non-BMP code points (which Python represents as a *single* str char,
    not a surrogate pair) must canonicalise cleanly."""
    smiley = "\U0001f600"  # 😀 U+1F600 GRINNING FACE
    assert len(smiley) == 1
    out = canonical_json(smiley)
    assert out == b'"\xf0\x9f\x98\x80"'
    # Round-trip through hash_obj works.
    assert hash_obj(smiley) == sha256_hex(out)


# ---------------------------------------------------------------------------
# Non-ASCII key ordering
# ---------------------------------------------------------------------------


def test_keys_sort_by_unicode_codepoint_not_locale() -> None:
    """ASCII < Latin-1 supplement < CJK, by raw code point.

    'a' (U+0061) < 'b' (U+0062) < 'z' (U+007A) < 'á' (U+00E1) <
    'é' (U+00E9) < '中' (U+4E2D).

    A locale-aware sort might collate 'á' near 'a'; we don't.
    """
    obj = {"é": 1, "a": 2, "z": 3, "á": 4, "b": 5, "中": 6}
    out = canonical_json(obj)
    # The key bytes must appear in code-point order.
    expected_key_order = ["a", "b", "z", "á", "é", "中"]
    positions = [out.index(f'"{k}"'.encode("utf-8")) for k in expected_key_order]
    assert positions == sorted(positions)


def test_uppercase_sorts_before_lowercase() -> None:
    """ASCII uppercase (U+0041..) sorts before lowercase (U+0061..)."""
    obj = {"b": 1, "A": 2, "a": 3, "B": 4}
    out = canonical_json(obj)
    assert out == b'{"A":2,"B":4,"a":3,"b":1}'


def test_keys_containing_supplementary_planes_sort_correctly() -> None:
    obj = {"\U0001f600": 1, "z": 2, "a": 3}
    out = canonical_json(obj)
    # 'a' < 'z' < non-BMP smiley by code point.
    assert out.index(b'"a"') < out.index(b'"z"') < out.index(b'"\xf0\x9f\x98\x80"')


def test_keys_NFC_vs_NFD_sort_independently() -> None:
    """Two normalisations of "é" sort by raw code point, not by collation."""
    nfc = unicodedata.normalize("NFC", "é")  # "\u00e9"
    nfd = unicodedata.normalize("NFD", "é")  # "e\u0301"
    obj = {nfc: 1, nfd: 2}
    out = canonical_json(obj)
    # NFD ("e\u0301") starts with 'e' (U+0065) so sorts before NFC (U+00E9).
    assert out.index(f'"{nfd}"'.encode("utf-8")) < out.index(
        f'"{nfc}"'.encode("utf-8")
    )


# ---------------------------------------------------------------------------
# Decimals
# ---------------------------------------------------------------------------


def test_decimal_is_rejected() -> None:
    """``decimal.Decimal`` is intentionally not auto-coerced.

    Silently casting Decimal -> float would lose precision and produce
    different hashes on different platforms. Producers must explicitly
    convert to ``str`` (preserves precision) or to ``float`` (commits to
    float semantics).
    """
    with pytest.raises(TypeError, match="Decimal"):
        canonical_json(decimal.Decimal("1.5"))


def test_decimal_string_form_is_stable() -> None:
    """The recommended workaround: serialise Decimal as a string."""
    a = canonical_json(str(decimal.Decimal("3.14159265358979323846")))
    b = canonical_json("3.14159265358979323846")
    assert a == b


def test_decimal_via_float_loses_precision() -> None:
    """Documented loss: float coercion is observable."""
    via_float = canonical_json(0.1 + 0.2)
    via_str = canonical_json("0.3")
    assert via_float != via_str
    # And float(0.1+0.2) is not 0.3:
    assert b"0.30000000000000004" in via_float


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, b"0"),
        (-0, b"0"),
        (1, b"1"),
        (10**30, str(10**30).encode("ascii")),
        (-(10**30), str(-(10**30)).encode("ascii")),
        (1.0, b"1.0"),
        (-1.5, b"-1.5"),
    ],
)
def test_numeric_serialisation_is_stable(value, expected: bytes) -> None:
    assert canonical_json(value) == expected


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        canonical_json(bad)
    # And math.nan in a structure is also rejected.
    with pytest.raises(ValueError):
        canonical_json({"x": bad})


# ---------------------------------------------------------------------------
# Binary payload marker
# ---------------------------------------------------------------------------


def test_bytes_use_bytes_hex_marker() -> None:
    out = canonical_json(b"\x00\x01\xff")
    assert out == b'{"__bytes_hex__":"0001ff"}'


def test_bytearray_uses_same_marker_as_bytes() -> None:
    a = canonical_json(b"abc")
    b = canonical_json(bytearray(b"abc"))
    assert a == b == b'{"__bytes_hex__":"616263"}'


def test_bytes_marker_is_lowercase_hex() -> None:
    out = canonical_json(bytes(range(16)))
    # 0..15 == 000102...0f, all lowercase
    assert b'"__bytes_hex__":"000102030405060708090a0b0c0d0e0f"' in out


def test_empty_bytes_round_trip() -> None:
    assert canonical_json(b"") == b'{"__bytes_hex__":""}'


def test_bytes_inside_structure() -> None:
    obj = {"payload": b"hi", "len": 2}
    out = canonical_json(obj)
    # Sorted keys: "len" before "payload".
    assert out == b'{"len":2,"payload":{"__bytes_hex__":"6869"}}'


def test_bytes_marker_collides_intentionally_with_user_dict() -> None:
    """A user-supplied dict using ``__bytes_hex__`` is indistinguishable from
    a real ``bytes`` payload. This is part of the v1 contract — flagged
    here so a future spec revision (``__sb_bytes__`` marker, or CBOR
    binary frames) is a deliberate breaking change with a version bump.
    """
    user_dict = {"__bytes_hex__": "6869"}
    raw_bytes = b"hi"
    assert canonical_json(user_dict) == canonical_json(raw_bytes)
    assert hash_obj(user_dict) == hash_obj(raw_bytes)


def test_bytes_inside_list_and_nested() -> None:
    obj = [b"a", [b"bc", {"k": b""}]]
    out = canonical_json(obj)
    assert out == (
        b'[{"__bytes_hex__":"61"},'
        b'[{"__bytes_hex__":"6263"},'
        b'{"k":{"__bytes_hex__":""}}]]'
    )


def test_large_bytes_payload_hashes_stably() -> None:
    payload = bytes(range(256)) * 16  # 4 KiB, every byte value present
    h1 = hash_obj(payload)
    h2 = hash_obj(bytearray(payload))
    assert h1 == h2
    # Hex-marker is the entire content.
    assert canonical_json(payload).startswith(b'{"__bytes_hex__":"')


# ---------------------------------------------------------------------------
# Cross-cutting: hash determinism for these tricky inputs
# ---------------------------------------------------------------------------


def test_canonicalisation_version_pin_unchanged() -> None:
    """If you change anything above, you must bump this constant."""
    assert CANONICALISATION_VERSION == "1"


@pytest.mark.parametrize(
    "obj",
    [
        "café",
        unicodedata.normalize("NFD", "café"),
        {"é": "ñ", "a": [1, 2, b"\xff"]},
        b"\x00" * 32,
        {"\U0001f600": True, "z": False},
        ["a", "á", "中", "\U0001f600"],
    ],
)
def test_repeated_canonicalisation_is_deterministic(obj) -> None:
    """Calling canonical_json twice on the same input yields identical bytes."""
    a = canonical_json(obj)
    b = canonical_json(obj)
    assert a == b
    assert hash_obj(obj) == sha256_hex(a)
