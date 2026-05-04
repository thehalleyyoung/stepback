"""SMT-checked equivalence tests for canonical-JSON (Step 48).

These tests confirm that ``stepback.canonical.canonical_json`` agrees
with the prose-derived reference serialiser in
``spec.canonical.bounded`` on every input in the bounded subset
``B(d=1)``. The equality is mediated by Z3 when available; we still
run the full enumeration with native equality even if Z3 is missing,
so the test never spuriously skips on machines that don't have the
solver installed.

The cross-language arms (Rust, TypeScript) are exercised by
``tests/test_canonical_differential.py`` so that this file stays a
fast pure-Python check.
"""
from __future__ import annotations

import pytest

from spec.canonical.bounded import (
    enumerate_bounded,
    reference_canonical_json,
)
from spec.canonical.smt_equivalence import (
    EquivalenceResult,
    check_equivalence,
    prove_python_vs_reference,
)
from stepback.canonical import canonical_json


def test_default_corpus_is_nontrivial() -> None:
    """Sanity: the bounded enumeration is at least ~1k values, otherwise
    the equivalence check would be vacuous."""
    n = sum(1 for _ in enumerate_bounded())
    assert n >= 1000, f"bounded corpus shrank to {n}; expected >= 1000"


def test_python_canonical_matches_reference_with_z3() -> None:
    """Z3-mediated proof: every bounded value canonicalises identically
    via ``stepback.canonical.canonical_json`` and the reference."""
    result = prove_python_vs_reference()
    assert isinstance(result, EquivalenceResult)
    if not result.ok:
        v, expected, actual = result.diverged  # type: ignore[misc]
        pytest.fail(
            f"Python canonicaliser diverges from reference at value={v!r}\n"
            f"expected={expected!r}\nactual  ={actual!r}"
        )
    assert result.checked >= 1000


def test_python_canonical_matches_reference_without_z3() -> None:
    """Same proof, but force-degraded to native equality so we always
    have a baseline assertion even if z3 fails to import in CI."""
    result = check_equivalence(
        name="python", canonicaliser=canonical_json, use_z3=False
    )
    if not result.ok:
        v, expected, actual = result.diverged  # type: ignore[misc]
        pytest.fail(
            f"Python canonicaliser diverges (no-z3) at value={v!r}\n"
            f"expected={expected!r}\nactual  ={actual!r}"
        )


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, b"null"),
        (True, b"true"),
        (False, b"false"),
        (0, b"0"),
        (-1, b"-1"),
        ("", b'""'),
        ("a", b'"a"'),
        ('"', b'"\\""'),
        ("\\", b'"\\\\"'),
        ("\n", b'"\\n"'),
        ("\x01", b'"\\u0001"'),
        ("é", b'"\xc3\xa9"'),
        ([], b"[]"),
        ({}, b"{}"),
        ({"b": 1, "a": 2}, b'{"a":2,"b":1}'),
        ({"a": [1, 2]}, b'{"a":[1,2]}'),
    ],
)
def test_reference_emits_expected_bytes(value: object, expected: bytes) -> None:
    """Spot-check the reference serialiser's output for the spec-driven
    cases that anchor the bounded subset."""
    got = reference_canonical_json(value)
    assert got == expected, f"reference(<{value!r}>) = {got!r}, expected {expected!r}"


def test_reference_and_python_agree_on_unicode_keys() -> None:
    """Non-ASCII key sort order is one of the trickier conformance cases.
    Confirm the bounded subset exercises it and that both impls agree."""
    samples = [
        {"ä": 1, "a": 2},
        {"ä": 1, "b": 2, '"': 3},
    ]
    for s in samples:
        assert canonical_json(s) == reference_canonical_json(s)
