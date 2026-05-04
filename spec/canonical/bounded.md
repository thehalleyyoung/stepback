# Canonical-JSON bounded subset (Step 48)

This document defines the **bounded JSON subset** the SB-Trace canonicalisation
equivalence harness operates on. The harness has two complementary parts:

1. **SMT-checked equivalence.** A symbolic Z3 model encodes the canonical
   serialiser as a function over the bounded grammar `B(d)` defined below and
   asks Z3 to prove (i.e. to fail to satisfy the negation of) byte-equality
   between Python's `stepback.canonical.canonical_json` and an independently
   re-derived reference serialiser. Z3 either returns `unsat` (proof) or
   produces a concrete counterexample input on which the two implementations
   disagree.

2. **Differential testing.** The same bounded grammar is enumerated
   exhaustively up to a configurable depth and width, and every available
   language implementation (Python, Rust, TypeScript, Go, JVM, .NET) emits
   canonical bytes for every member. The harness asserts pairwise
   byte-equality; any divergence is a conformance bug in the emitting
   language.

## Grammar `B(d, w, K, Σ, N)`

```
value(0)  ::= null | true | false | int | str
value(d)  ::= value(0) | array(d) | object(d)        ; d > 0

int       ::= integer in [-N, N]                     ; default N = 4
str       ::= string of length ≤ K over alphabet Σ   ; default K = 2
                                                       Σ = {"a", "z", " ", "\"",
                                                            "\\", "\n", "\u0001",
                                                            "é"}                ; covers
                                                                                 ; ASCII,
                                                                                 ; quote,
                                                                                 ; backslash,
                                                                                 ; whitespace,
                                                                                 ; control
                                                                                 ; (escape
                                                                                 ; path),
                                                                                 ; non-ASCII
                                                                                 ; (multi-byte
                                                                                 ; UTF-8 path)
array(d)  ::= [ value(d-1) , … ]                     ; length 0..w
object(d) ::= { key:value(d-1) , … }                 ; size 0..w,
                                                       keys distinct,
                                                       keys drawn from a small
                                                       fixed set {"a","b","c",
                                                                   "ä","\""}    ; non-ASCII
                                                                                 ; & special
                                                                                 ; chars in
                                                                                 ; keys
```

`d` (depth), `w` (width), `K` (string length), `N` (integer magnitude) are all
small constants chosen to keep the enumerated cardinality tractable while still
exercising every code path:

| dimension | covers |
| --- | --- |
| `int`     | sign handling, zero, positive/negative, multi-digit |
| `str`     | empty, single char, ASCII letter, ASCII space, quote escape, backslash escape, newline escape, U+0001 control escape (`\u0001`), non-ASCII (`é` → `0xC3 0xA9`) |
| keys      | both ASCII and non-ASCII keys; keys containing characters that need escaping; sort order over UTF-8 vs UTF-16 code units |
| arrays    | empty, singleton, multi-element, mixed types, nested |
| objects   | empty, singleton, multi-key, key sort order, nested |

The default cardinality `|B(d=2, w=2, K=2, |Σ|=8, N=4)|` is on the order of
~10⁴ values — enumerable in milliseconds, large enough that any plausible
divergence between two canonicalisers shows up.

## Reference serialiser

The reference serialiser is a re-derivation of the canonical-JSON algorithm
written *only* against this spec, intentionally **not** sharing code with
`stepback.canonical`. It is defined in `spec/canonical/bounded.py` as
`reference_canonical_json`. SMT and differential equivalence proofs target
this reference; if Python's algorithm disagrees with the reference, that is a
bug in `stepback.canonical` (and therefore in every binding that mirrors it).

## What "SMT-checked equivalence" means here

Z3 cannot reason about Python bytecode directly. We make it tractable by:

1. Encoding both algorithms as **pure functions** over a Z3 datatype that
   models `B(d)` with `d=2` (this is the "bounded subset" referred to in Step
   48 of `100_STEPS.md`).
2. Defining a single recursive Z3 function `serialise(v) → Sequence[Bytes]`
   that mirrors the canonical-JSON algorithm — written once, in the SMT
   module, derived independently from the prose above.
3. Proving that for every concrete value `v ∈ B(d=2)` (a finite set with
   distinguished cases), `serialise(v) == python_canonical(v)`. Because `v`
   ranges over a *finite* enumeration, the SMT step is effectively exhaustive
   bounded-model-checking with Z3 as the equality engine.

The same bounded enumeration is used by the differential runner to check Rust,
TypeScript, Go, JVM, and .NET canonicalisers. Languages without a writer-side
canonicaliser (Go, JVM, and .NET ship read-only verifiers as of v0.1) are
exercised by **round-tripping** their reader against bytes produced by
Python/Rust/TypeScript: every member of `B(d)` is emitted, parsed, hashed,
and the hash compared back to Python's hash. A reader that disagrees on bytes
or hash is a conformance bug.

## Reproducing the proof / differential check

```bash
# SMT proof (uses Z3; install via `pip install -e .[dev]`)
python -m spec.canonical.smt_equivalence

# Differential runner (auto-detects available toolchains)
python -m spec.canonical.differential
```

Both are wired into the test suite (`tests/test_canonical_smt_equivalence.py`
and `tests/test_canonical_differential.py`), which skip individual language
arms when the relevant toolchain is unavailable.
