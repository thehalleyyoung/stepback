"""Bounded JSON subset and reference canonicaliser for Step 48.

This module is the **single source of truth** for the bounded subset
``B(d, w, K, Σ, N)`` used by the SMT equivalence proof and the
cross-language differential runner. See ``spec/canonical/bounded.md`` for
the prose definition.

Two pieces live here:

* ``enumerate_bounded(...)`` — deterministic enumeration of every JSON
  value in the bounded subset.
* ``reference_canonical_json(v)`` — an independent re-derivation of the
  canonical-JSON algorithm. Deliberately written **without importing**
  ``stepback.canonical`` so an equivalence check between the two has
  semantic content. If you change ``stepback/canonical.py`` you should
  *not* automatically change this file in lock-step; the whole point is
  that they are derived independently from the prose spec.
"""
from __future__ import annotations

from typing import Iterable, Iterator, List, Sequence

# ---- Bounded subset parameters ------------------------------------------

#: Default string alphabet. Each character exercises a distinct branch of the
#: canonical-JSON string serialiser:
#:
#: * ``"a"``      — plain ASCII letter
#: * ``" "``      — ASCII space (no escape, but boundary case)
#: * ``"\""``     — must escape as ``\"``
#: * ``"\\"``     — must escape as ``\\``
#: * ``"\n"``     — must escape as ``\n``
#: * ``"\x01"``   — control char, must escape as ``\u0001``
#: * ``"é"``      — non-ASCII, must round-trip as raw UTF-8 ``0xC3 0xA9``
#:
#: Adding characters here grows |B| roughly geometrically; keep small.
DEFAULT_ALPHABET: tuple[str, ...] = ("a", " ", '"', "\\", "\n", "\x01", "é")

#: Default key alphabet. Smaller than the value alphabet for tractability,
#: but deliberately mixes ASCII, an escape-requiring char, and non-ASCII so
#: the key-sort path is exercised under both UTF-8 and UTF-16 code-unit
#: orderings (Python sorts by Unicode code point; TypeScript's default
#: ``Array.prototype.sort`` sorts by UTF-16 code unit — for BMP code points
#: these orderings agree).
DEFAULT_KEY_ALPHABET: tuple[str, ...] = ("a", "b", '"', "ä")

#: Default integer magnitude.
DEFAULT_INT_MAX: int = 2

#: Default container width (max array length / object size).
DEFAULT_WIDTH: int = 2

#: Default recursion depth. Depth 1 means arrays/objects contain only
#: atoms; bumping this to 2 quickly runs into combinatorial explosion at
#: width >= 2 (≈10⁶ values), so the deeper-nesting check is run with a
#: ``cap`` in the SMT module instead of being part of the default corpus.
DEFAULT_DEPTH: int = 1

#: Default max string length.
DEFAULT_STR_LEN: int = 1


def _enumerate_strings(alphabet: Sequence[str], max_len: int) -> Iterator[str]:
    """Yield every string of length 0..max_len over ``alphabet``."""
    yield ""
    if max_len == 0:
        return
    current: list[str] = [""]
    for _ in range(max_len):
        next_strings: list[str] = []
        for s in current:
            for ch in alphabet:
                t = s + ch
                next_strings.append(t)
                yield t
        current = next_strings


def _enumerate_atoms(
    alphabet: Sequence[str],
    max_str_len: int,
    int_max: int,
) -> Iterator[object]:
    """Yield every atomic JSON value (null/bool/int/str) in the subset."""
    yield None
    yield True
    yield False
    for n in range(-int_max, int_max + 1):
        yield n
    for s in _enumerate_strings(alphabet, max_str_len):
        yield s


def _enumerate_arrays(
    inner: Sequence[object], max_width: int
) -> Iterator[list[object]]:
    """Yield every array of length 0..max_width whose elements are drawn from
    ``inner`` (with replacement, order matters)."""
    yield []
    pool: list[list[object]] = [[]]
    for _ in range(max_width):
        nxt: list[list[object]] = []
        for arr in pool:
            for v in inner:
                tail = arr + [v]
                nxt.append(tail)
                yield tail
        pool = nxt


def _enumerate_objects(
    keys: Sequence[str],
    inner: Sequence[object],
    max_width: int,
) -> Iterator[dict[str, object]]:
    """Yield every object of size 0..max_width whose keys are a *subset* of
    ``keys`` (each key used at most once, since JSON-on-the-wire dedups
    duplicate keys per RFC 8259) and whose values are drawn from ``inner``.

    We enumerate over key *combinations* (set semantics) and then over every
    value-tuple, so the caller sees both `{}`, every singleton `{k: v}`, and
    every multi-key combination up to ``max_width``. Different key insertion
    orders are *not* yielded because canonical-JSON sorts keys at every
    depth — different insertion orders produce identical canonical bytes
    and would only inflate the corpus without exercising new code paths.
    """
    yield {}
    seen_subsets: list[tuple[str, ...]] = [()]
    # Build all key subsets up to size ``max_width``.
    for size in range(1, max_width + 1):
        next_subsets: list[tuple[str, ...]] = []
        for prev in seen_subsets:
            tail_start = keys.index(prev[-1]) + 1 if prev else 0
            for i in range(tail_start, len(keys)):
                k = keys[i]
                if k in prev:
                    continue
                subset = prev + (k,)
                next_subsets.append(subset)
        seen_subsets = next_subsets
        for subset in next_subsets:
            yield from _emit_object_assignments(subset, inner)


def _emit_object_assignments(
    keys: Sequence[str], inner: Sequence[object]
) -> Iterator[dict[str, object]]:
    """Yield every assignment of values to ``keys`` from ``inner``."""
    if not keys:
        yield {}
        return
    # Cartesian product without importing itertools so this stays a single
    # file with no extra deps. ``keys`` is small (<=4), ``inner`` modest.
    indices = [0] * len(keys)
    n = len(inner)
    while True:
        yield {k: inner[indices[i]] for i, k in enumerate(keys)}
        # increment
        j = len(keys) - 1
        while j >= 0:
            indices[j] += 1
            if indices[j] < n:
                break
            indices[j] = 0
            j -= 1
        if j < 0:
            return


def enumerate_bounded(
    *,
    depth: int = DEFAULT_DEPTH,
    width: int = DEFAULT_WIDTH,
    str_len: int = DEFAULT_STR_LEN,
    int_max: int = DEFAULT_INT_MAX,
    alphabet: Sequence[str] = DEFAULT_ALPHABET,
    key_alphabet: Sequence[str] = DEFAULT_KEY_ALPHABET,
    cap: int | None = None,
) -> Iterator[object]:
    """Enumerate every JSON value in the bounded subset ``B(d,w,K,Σ,N)``.

    Yields atoms first, then arrays of atoms, then objects of atoms, then
    arrays containing arrays/objects, etc. Yields each value exactly once,
    deterministically. If ``cap`` is given, stops after that many values
    (useful for SMT depths where the cross-product would be too large).
    """
    yielded = 0

    def bump() -> bool:
        nonlocal yielded
        yielded += 1
        return cap is not None and yielded >= cap

    # depth-0 atoms
    pool: list[object] = []
    for v in _enumerate_atoms(alphabet, str_len, int_max):
        pool.append(v)
        yield v
        if bump():
            return

    for _d in range(depth):
        new_layer: list[object] = []
        for arr in _enumerate_arrays(pool, width):
            if not arr:
                continue  # `[]` already yielded? we've not yielded it yet at depth 0
            new_layer.append(arr)
            yield arr
            if bump():
                return
        for obj in _enumerate_objects(key_alphabet, pool, width):
            if not obj:
                continue
            new_layer.append(obj)
            yield obj
            if bump():
                return
        # also yield the empty containers exactly once at the first depth
        if _d == 0:
            yield []
            if bump():
                return
            yield {}
            if bump():
                return
        pool = pool + new_layer


# ---- Reference canonicaliser --------------------------------------------

# This is the *prose-derived* canonicaliser. It is intentionally a fresh
# implementation: changing ``stepback/canonical.py`` MUST NOT trigger a
# mechanical change here — the whole point of having two implementations is
# that an equivalence check has content. The Python ``json`` module is not
# imported.

_ESCAPE_TABLE: dict[int, str] = {
    0x22: '\\"',
    0x5C: "\\\\",
    0x08: "\\b",
    0x0C: "\\f",
    0x0A: "\\n",
    0x0D: "\\r",
    0x09: "\\t",
}


def _escape_str(s: str) -> str:
    out: list[str] = ['"']
    for ch in s:
        cp = ord(ch)
        if cp in _ESCAPE_TABLE:
            out.append(_ESCAPE_TABLE[cp])
        elif cp < 0x20:
            out.append(f"\\u{cp:04x}")
        else:
            out.append(ch)  # raw character; .encode("utf-8") below handles non-ASCII
    out.append('"')
    return "".join(out)


def _emit(value: object, parts: list[str]) -> None:
    if value is None:
        parts.append("null")
        return
    if value is True:
        parts.append("true")
        return
    if value is False:
        parts.append("false")
        return
    if isinstance(value, bool):
        # bool is subclass of int; handled above.
        parts.append("true" if value else "false")
        return
    if isinstance(value, int):
        parts.append(str(value))
        return
    if isinstance(value, str):
        parts.append(_escape_str(value))
        return
    if isinstance(value, list):
        parts.append("[")
        for i, item in enumerate(value):
            if i > 0:
                parts.append(",")
            _emit(item, parts)
        parts.append("]")
        return
    if isinstance(value, dict):
        keys = sorted(value.keys())
        parts.append("{")
        for i, k in enumerate(keys):
            if i > 0:
                parts.append(",")
            parts.append(_escape_str(str(k)))
            parts.append(":")
            _emit(value[k], parts)
        parts.append("}")
        return
    raise TypeError(f"reference canonicaliser does not support {type(value).__name__}")


def reference_canonical_json(value: object) -> bytes:
    """Reference canonical-JSON serialiser, prose-derived.

    Output MUST equal ``stepback.canonical.canonical_json(value)`` for every
    value in the bounded subset. If they ever disagree, ``stepback.canonical``
    is the buggy one (or this reference is — both are equally testable, but
    this one has the simpler prose-only derivation, which makes it the
    reference).
    """
    parts: list[str] = []
    _emit(value, parts)
    return "".join(parts).encode("utf-8")


def iter_default_corpus(cap: int | None = None) -> Iterable[object]:
    """Convenience wrapper for the default-parameter corpus."""
    return enumerate_bounded(cap=cap)


__all__ = [
    "DEFAULT_ALPHABET",
    "DEFAULT_DEPTH",
    "DEFAULT_INT_MAX",
    "DEFAULT_KEY_ALPHABET",
    "DEFAULT_STR_LEN",
    "DEFAULT_WIDTH",
    "enumerate_bounded",
    "iter_default_corpus",
    "reference_canonical_json",
]
