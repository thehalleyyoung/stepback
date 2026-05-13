"""Hypothesis property tests for canonical JSON round-tripping.

Implements step 27 of ``100_STEPS.md``:

    Add Hypothesis tests for canonical JSON round-tripping over arbitrary
    nested JSON-like values.

The canonicaliser in :mod:`stepback.canonical` is the cache-safety boundary
for the dirty-set engine: equal *semantic* inputs must hash to the same
bytes, and unequal inputs must hash to different bytes (modulo SHA-256).
These properties are checked here over generated JSON-like values rather
than hand-picked fixtures, so we get coverage of weird-but-legal corners
(non-ASCII keys, deeply nested structures, tuple/list equivalence, set
ordering, byte payloads, decimal-shaped strings, and so on).
"""
from __future__ import annotations

import json
import math
from typing import Any

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from stepback.canonical import (
    CANONICALISATION_VERSION,
    canonical_json,
    hash_obj,
    sha256_hex,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# JSON-native scalars: None, bools, finite ints, finite floats, text.
# We exclude NaN/Infinity because canonical_json sets allow_nan=False on
# purpose; those are covered separately below.
_finite_floats = st.floats(
    allow_nan=False,
    allow_infinity=False,
    # JSON cannot represent +/-0 distinctly anyway.
    allow_subnormal=True,
)

# Keys must be strings for json.dumps. Allow unicode but reject lone
# surrogates because canonical_json uses UTF-8 encoding which rejects them.
_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    max_size=32,
)

_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**63), max_value=2**63 - 1),
    _finite_floats,
    _text,
)


def _json_like(max_leaves: int = 30) -> st.SearchStrategy[Any]:
    """Recursive JSON-like values: scalars, lists, dicts (string keys).

    We deliberately *also* let containers carry tuples and sets so we can
    exercise the canonicaliser's ``_default`` hook in property tests.
    """
    return st.recursive(
        _scalars,
        lambda children: st.one_of(
            st.lists(children, max_size=6),
            st.tuples(children, children),
            st.frozensets(_scalars, max_size=6),
            st.dictionaries(_text, children, max_size=6),
        ),
        max_leaves=max_leaves,
    )


def _pure_json(max_leaves: int = 30) -> st.SearchStrategy[Any]:
    """JSON values without tuples/sets/bytes — suitable for json.loads compare."""
    return st.recursive(
        _scalars,
        lambda children: st.one_of(
            st.lists(children, max_size=6),
            st.dictionaries(_text, children, max_size=6),
        ),
        max_leaves=max_leaves,
    )


# Use a single settings profile for these properties: they are pure-CPU and
# fast, but we cap the deadline to keep CI predictable.
_PROFILE = settings(
    max_examples=200,
    deadline=2000,
    suppress_health_check=[HealthCheck.too_slow],
)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@_PROFILE
@given(_pure_json())
def test_canonical_json_roundtrips_pure_json(obj: Any) -> None:
    """For pure-JSON values, json.loads(canonical_json(x)) == x."""
    blob = canonical_json(obj)
    decoded = json.loads(blob.decode("utf-8"))
    # Floats may compare equal even when bit patterns differ (e.g. -0.0 vs 0.0).
    # The structural compare below handles NaN-free finite floats fine.
    assert _structurally_equal(decoded, obj), (decoded, obj)


@_PROFILE
@given(_json_like())
def test_canonical_json_is_deterministic(obj: Any) -> None:
    """Calling canonical_json twice on the same value yields identical bytes."""
    a = canonical_json(obj)
    b = canonical_json(obj)
    assert a == b


@_PROFILE
@given(_json_like())
def test_canonical_json_is_idempotent_under_reparse(obj: Any) -> None:
    """canonical_json(json.loads(canonical_json(x))) == canonical_json(x).

    This is the *real* round-trip property the cache relies on: once
    something has been canonicalised, decoded as JSON, and re-canonicalised,
    the bytes must not drift. Otherwise a step's input hash could change
    between record and replay even though the semantic content is identical.
    """
    blob = canonical_json(obj)
    reparsed = json.loads(blob.decode("utf-8"))
    assert canonical_json(reparsed) == blob


@_PROFILE
@given(_json_like())
def test_hash_obj_matches_sha256_of_canonical_json(obj: Any) -> None:
    assert hash_obj(obj) == sha256_hex(canonical_json(obj))


@_PROFILE
@given(_pure_json())
def test_dict_key_order_does_not_affect_canonical_form(obj: Any) -> None:
    """Re-inserting dict keys in reverse order must not change the output."""
    shuffled = _reverse_dict_keys(obj)
    assert canonical_json(shuffled) == canonical_json(obj)


@_PROFILE
@given(_json_like())
def test_canonical_json_output_is_valid_utf8(obj: Any) -> None:
    blob = canonical_json(obj)
    # Decoding must succeed and the result must re-encode to the same bytes.
    text = blob.decode("utf-8")
    assert text.encode("utf-8") == blob


@_PROFILE
@given(_pure_json())
def test_canonical_json_keys_are_sorted(obj: Any) -> None:
    """Every dict in the output must have its keys in lexical order."""
    blob = canonical_json(obj)
    decoded = json.loads(blob.decode("utf-8"), object_pairs_hook=list)
    _assert_pairs_sorted(decoded)


@_PROFILE
@given(_pure_json(), _pure_json())
def test_distinct_values_distinct_hashes(a: Any, b: Any) -> None:
    """Different canonical bytes ⇒ different sha256 hashes (collision-free here).

    Equal canonical bytes are allowed because the strategy can produce
    semantically equal values (e.g. ``[1.0]`` and ``[1]``... wait, those
    differ in JSON. Two literally identical values are an obvious case).
    The interesting bit is the *contrapositive*: we never see equal hashes
    for unequal canonical bytes.
    """
    ja = canonical_json(a)
    jb = canonical_json(b)
    if ja != jb:
        assert hash_obj(a) != hash_obj(b)


@_PROFILE
@given(_json_like())
def test_tuple_list_equivalence(obj: Any) -> None:
    """Tuples canonicalise to the same bytes as the equivalent list."""
    listy = _tuples_to_lists(obj)
    assert canonical_json(obj) == canonical_json(listy)


@_PROFILE
@given(st.frozensets(_scalars, max_size=8))
def test_frozenset_canonicalises_to_sorted_list(s: frozenset) -> None:
    blob = canonical_json(s)
    decoded = json.loads(blob.decode("utf-8"))
    assert isinstance(decoded, list)
    # Order must be a total order on the canonicalised representation.
    # We can't rely on Python's < across mixed types, so just assert the
    # multiset equals the input's normalised multiset.
    assert sorted(decoded, key=repr) == sorted(
        [None if x is None else x for x in s], key=repr
    )


@_PROFILE
@given(st.binary(max_size=64))
def test_bytes_payload_marker(payload: bytes) -> None:
    blob = canonical_json(payload)
    decoded = json.loads(blob.decode("utf-8"))
    assert decoded == {"__bytes_hex__": payload.hex()}


def test_nan_and_inf_are_rejected() -> None:
    """``allow_nan=False`` is part of the canonical contract."""
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            canonical_json(bad)
        with pytest.raises(ValueError):
            canonical_json([1, 2, bad])
        with pytest.raises(ValueError):
            canonical_json({"x": bad})


def test_canonicalisation_version_pinned() -> None:
    """The canonicalisation version is part of the wire contract.

    Bumping it is a deliberate, breaking change recorded in the trace
    header; this assertion exists to make accidental bumps loud.
    """
    assert CANONICALISATION_VERSION == "1"


def test_unsupported_type_raises_typeerror() -> None:
    class _Opaque:
        pass

    with pytest.raises(TypeError):
        canonical_json(_Opaque())


# ---------------------------------------------------------------------------
# Additional structural properties (step 27 — comprehensive coverage)
# ---------------------------------------------------------------------------


@_PROFILE
@given(st.lists(_scalars, min_size=2, max_size=8))
def test_list_order_is_significant(items: list) -> None:
    """Reversing a non-palindromic list must produce different canonical bytes.

    Lists are ordered in JSON; the cache must NOT treat ``[1, 2]`` and
    ``[2, 1]`` as the same input. This is the contrapositive of the
    dict-key-order invariance test above.
    """
    reversed_items = list(reversed(items))
    if items != reversed_items:
        assert canonical_json(items) != canonical_json(reversed_items)


@_PROFILE
@given(_pure_json())
def test_empty_containers_canonicalise_consistently(obj: Any) -> None:
    """Wrapping a value in an empty list/dict and unwrapping must round-trip."""
    wrapped = {"v": obj, "empty_list": [], "empty_dict": {}}
    blob = canonical_json(wrapped)
    decoded = json.loads(blob.decode("utf-8"))
    assert decoded["empty_list"] == []
    assert decoded["empty_dict"] == {}
    assert _structurally_equal(decoded["v"], obj)


@_PROFILE
@given(st.integers(min_value=1, max_value=20), _scalars)
def test_deep_nesting_round_trips(depth: int, leaf: Any) -> None:
    """Arbitrary-depth single-key dict nesting must round-trip exactly."""
    obj: Any = leaf
    for i in range(depth):
        obj = {f"k{i}": obj}
    assert canonical_json(json.loads(canonical_json(obj).decode("utf-8"))) == canonical_json(obj)


@_PROFILE
@given(_text, _text, _scalars, _scalars)
def test_dict_key_swap_changes_hash_when_values_differ(
    k1: str, k2: str, v1: Any, v2: Any
) -> None:
    """If keys differ and the (key→value) mapping differs, hashes must differ."""
    assume(k1 != k2)
    assume(v1 != v2)
    a = {k1: v1, k2: v2}
    b = {k1: v2, k2: v1}
    assert canonical_json(a) != canonical_json(b)


@_PROFILE
@given(_text, _text)
def test_string_concatenation_is_unambiguous(a: str, b: str) -> None:
    """``[a, b]`` and ``[a + b]`` must canonicalise distinctly.

    JSON's quoting prevents the ambiguity that hits naive string-concat
    serialisers; this property locks that in across arbitrary unicode.
    """
    if a or b:
        assert canonical_json([a, b]) != canonical_json([a + b])


@_PROFILE
@given(_pure_json())
def test_canonical_json_has_no_whitespace(obj: Any) -> None:
    """The v1 contract is no insignificant whitespace anywhere in the output."""
    blob = canonical_json(obj)
    text = blob.decode("utf-8")
    in_string = False
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            assert ch not in (" ", "\t", "\n", "\r"), text


@_PROFILE
@given(_pure_json())
def test_double_canonical_json_via_bytes_is_idempotent(obj: Any) -> None:
    """canonical_json is a fixed point on already-canonical decoded objects."""
    once = canonical_json(obj)
    twice = canonical_json(json.loads(once.decode("utf-8")))
    thrice = canonical_json(json.loads(twice.decode("utf-8")))
    assert once == twice == thrice


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _structurally_equal(a: Any, b: Any) -> bool:
    """Compare two pure-JSON values structurally.

    Python's ``==`` already does the right thing for these types, but we
    want a single recursive helper that's easy to extend if we ever start
    accepting Decimal etc.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_structurally_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_structurally_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        # No NaN here (filtered by strategy); +/-0 compares equal.
        return a == b
    return a == b


def _reverse_dict_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        items = list(obj.items())
        items.reverse()
        return {k: _reverse_dict_keys(v) for k, v in items}
    if isinstance(obj, list):
        return [_reverse_dict_keys(x) for x in obj]
    return obj


def _tuples_to_lists(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return [_tuples_to_lists(x) for x in obj]
    if isinstance(obj, list):
        return [_tuples_to_lists(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _tuples_to_lists(v) for k, v in obj.items()}
    return obj


def _assert_pairs_sorted(node: Any) -> None:
    if isinstance(node, list):
        # Could be a JSON list of pairs (from object_pairs_hook) or a normal list.
        # Heuristic: if every item is a (str, _) pair, treat as a dict.
        if node and all(
            isinstance(x, list)
            and len(x) == 2
            and isinstance(x[0], str)
            for x in node
        ):
            keys = [x[0] for x in node]
            assert keys == sorted(keys), keys
            for _, v in node:
                _assert_pairs_sorted(v)
        else:
            for x in node:
                _assert_pairs_sorted(x)
