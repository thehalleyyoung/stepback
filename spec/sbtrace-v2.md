# SB-Trace v2 — Dual-Encoding Wire-format Specification (draft)

**Status:** Draft 0 (informational; tracks the v2 design currently
shipping in `stepback >= 0.1` only as a *candidate* — there is no
production v2 reader/writer yet).
**Format version:** `2`
**Canonicalisation version:** `2`
**Semantic-hash version:** `1`
**License:** Apache-2.0.

This document defines SB-Trace v2. v2 is the first version of the
`.sb` wire format to support **two encodings on the wire** — canonical
UTF-8 JSON (the v1 encoding) and deterministic CBOR (RFC 8949 §4.2) —
while preserving a **single encoding-agnostic semantic hash** so that
content-addressed caching, dirty-set propagation, and replay
soundness all continue to hold across mixed-encoding traces.

The intended audience is the same as `spec/sbtrace-v1.md`: anyone
writing an independent reader, writer, verifier, or conformance
harness. v2 is layered on v1; readers that already implement v1 can
add v2 by adding a new wire decoder and the semantic-hash function
in §4.

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**,
**SHOULD NOT**, **MAY**, and **OPTIONAL** are to be interpreted as
described in BCP 14 (RFC 2119, RFC 8174).

---

## 1. Goals

1. **Two wire encodings, one semantic hash.** A v2 frame body MAY be
   encoded as canonical UTF-8 JSON or as deterministic CBOR. Both
   encodings MUST decode to the same abstract value tree (§3) and
   MUST hash to the same value under the v2 semantic-hash function
   (§4). Cache keys, content-addressed step inputs, and dirty-set
   decisions are computed from that semantic hash, never from the
   raw on-disk bytes.

2. **Strict backward compatibility for hashes.** A v1 trace's
   `sha256_hex(canonical_json(obj))` cache key for an object `obj` is
   **not** equal to the v2 `semantic_hash(obj)` cache key in the
   general case — v2 changes how the hash is computed. v1 traces are
   not reinterpreted as v2; readers select the hashing function by
   `format_version` in the header, never by guessing.

3. **Capability-negotiated encoding.** Each frame's wire encoding is
   declared by the trace header and (for mixed traces) by the
   per-frame encoding tag. A v2 reader that does not implement one
   of the two encodings MUST fail closed when it encounters a frame
   in that encoding.

4. **Independent implementability.** The conformance suite
   (`stepback/conformance/`) ships fixtures encoded in both forms
   that hash to the same `semantic_hash`. Any implementation
   passing the suite is, by construction, encoding-portable.

## 2. Non-goals

* No new step kinds, no new receipt fields, no new policy semantics.
  v2 changes only encoding and hashing.
* No streaming change. Frames are still length-prefixed and
  append-only as in v1 §3.
* No bignum CBOR tags (2/3), no decimal fraction tag (4), no other
  CBOR tags. The deterministic-CBOR encoder used here is the
  tag-free subset described in `stepback.canonical_cbor`.

---

## 3. Abstract value model

The v2 abstract data model is the type union:

```
Value ::= Null
        | Bool
        | Int           -- in [-2**63, 2**63 - 1] for portability
        | Float         -- IEEE 754 binary64; NaN and ±0.0 distinguished
        | Str           -- Unicode text, NFC normal form recommended
        | Bytes         -- raw octet string
        | List of Value
        | Map from (Str | Int | Bytes) to Value
```

* Both wire encodings MUST round-trip every shape in this model.
* A `Float` value MUST preserve sign of zero. A canonical-JSON
  emitter MUST emit `-0.0` as `-0` (with a leading minus); a
  CBOR emitter MUST use the canonical 2-byte half encoding
  `0xf9 0x80 0x00` for `-0.0` and `0xf9 0x00 0x00` for `+0.0`.
* `NaN` is forbidden in canonical JSON (`allow_nan=False`); a CBOR
  emitter MUST encode the canonical NaN as `0xf9 0x7e 0x00`. A v2
  trace MAY contain `NaN` only when carried over CBOR; readers that
  only decode JSON MUST reject such a trace with a clear error.
* `Map` keys are values, not strings — CBOR's richer key space is
  preserved. JSON-encoded frames MUST stringify integer and bytes
  keys per §5.2; this stringification is reversed before semantic
  hashing.

## 4. Semantic hash

The v2 semantic hash is defined in `stepback.semantic_hash` and
mirrored here normatively. `H(x)` denotes SHA-256 of byte sequence
`x`. Concatenation is `||`. ASCII tag bytes are written in quotes.

```
sem(null)            = H("n:")
sem(false)           = H("b:0")
sem(true)            = H("b:1")
sem(i: Int)          = H("i:" || ascii(decimal(i)))
sem(f: Float)        = H("f:" || canonical_float(f))
sem(s: Str)          = H("s:" || utf8(s))
sem(y: Bytes)        = H("y:" || y)
sem([v0, ..., vN-1]) = H("l:" || ascii(decimal(N)) || ":"
                            || sem(v0) || ... || sem(vN-1))
sem({k0: v0, ...})   = H("m:" || ascii(decimal(N)) || ":"
                            || sem(k_pi(0)) || sem(v_pi(0))
                            || ... )
   where pi orders keys by sem(k_i) byte-lexicographically.

canonical_float(NaN)  = "nan"
canonical_float(+Inf) = "+inf"
canonical_float(-Inf) = "-inf"
canonical_float(+0.0) = "+0"
canonical_float(-0.0) = "-0"
canonical_float(f)    = ascii(repr_round_trip_binary64(f))
```

A trace's per-step content hash field (`inputs.hash`,
`outputs.hash`, `nondeterminism_hash`, `llm_request.hash`,
`llm_response.hash`) is computed by `"sha256:" + hex(sem(value))`,
not `"sha256:" + hex(sha256(canonical_json(value)))`.

The map-ordering rule **bytewise on `sem(k)`** is the key
encoding-portability hinge:

* Canonical JSON sorts maps by code-point of the UTF-8 string key.
* Deterministic CBOR sorts maps by bytewise lexicographic order of
  the *encoded* CBOR key.
* Neither agrees with the other for string keys whose UTF-8 byte
  ordering differs from their CBOR-encoded byte ordering (which
  prepends a length-prefix head byte).

By ordering on `sem(k)` instead, the two wire encodings can each
sort maps in their own native way **on disk** while still folding
to the same `sem(map)` when hashed.

## 5. On-disk frame layout

Frame layout, length-prefix encoding, HMAC chaining, and Ed25519
signature construction are unchanged from v1 §3 and v1 §7 except
for the items listed below.

### 5.1 Header changes

A v2 header MUST carry:

* `format_version: 2`
* `canonicalisation_version: 2`
* `semantic_hash_version: "1"`
* `wire_encodings: ["canonical-json", "deterministic-cbor"]` —
  ordered list of every encoding that may appear in the trace.
  Readers MUST refuse to start replay if any element is unknown.

### 5.2 Per-frame encoding tag

When `wire_encodings` has more than one element, every non-header
frame MUST carry a top-level wrapper field `enc` whose value is one
of the strings in `wire_encodings`. The `enc` field is itself
canonical-JSON; it sits in the wrapper alongside `prev_hmac`,
`hmac`, and `sig`. The framed body bytes (the bytes that go into
the HMAC and signature) MUST be in the encoding named by `enc`.

When `wire_encodings` is a single-element list, `enc` MAY be
omitted; the encoding is the unique element.

### 5.3 Heterogeneous-key map encoding in JSON

JSON has no native integer or bytes map keys. A v2 JSON-encoded
frame MUST stringify non-string map keys as follows:

* Integer key `n`         → `"!i:<decimal of n>"`
* Bytes key   `b`         → `"!y:<lower-hex of b>"`
* String key  `s` that
  itself begins with `"!"`→ `"!s:<original s>"`

Decoders MUST reverse this stringification before semantic hashing.
A v2 CBOR-encoded frame uses native CBOR map keys and skips the
stringification step. The semantic hash is computed on the decoded,
de-stringified map and is therefore identical for both wire
encodings of the same abstract value.

## 6. Conformance

A v2 implementation passes conformance iff, for every fixture
`spec/fixtures/v2/*.value.json` (the abstract value, expressed as
canonical-JSON for human readability) and every encoding `e` in
`{canonical-json, deterministic-cbor}`:

1. Encoding the value to `e` produces exactly the byte sequence in
   `spec/fixtures/v2/<name>.<e>.bin`.
2. `semantic_hash` of the value equals the hex digest in
   `spec/fixtures/v2/<name>.semhash.txt`.
3. Decoding the byte sequence in step 1 and re-hashing yields the
   same digest as step 2.
4. Mutating any byte of the encoded form (other than to a
   semantically-equivalent re-encoding, which is by construction
   impossible for *deterministic* encodings) changes the digest.

The Python reference implementations are
`stepback.canonical.canonical_json`, `stepback.canonical_cbor.canonical_cbor`,
and `stepback.semantic_hash.semantic_hash`. Independent
implementations MUST produce byte-identical encodings and digest-
identical hashes against the shipped fixtures.

## 7. Migration from v1

* v1 traces remain readable forever; v1 readers MUST NOT attempt to
  reinterpret v1 hashes as v2 hashes. Mixed v1/v2 caches MUST key on
  `(format_version, content_hash)`, never on `content_hash` alone.
* A migration tool (out of scope for this draft) MAY rewrite a v1
  trace into a v2 trace by re-hashing every step under the v2
  semantic-hash function and re-signing every frame. The original
  v1 trace's signature chain is *not* preserved across migration —
  the migration produces a new attestation.

## 8. Open questions

* Whether to add a `text` (msgpack-style) encoding for very
  constrained embedded recorders. This draft does not.
* Whether `semantic_hash_version` should be promoted to its own
  SemVer track separate from `format_version`. Probably yes when
  the first hash-rule clarification ships.
* Whether the JSON-side `"!i:"` / `"!y:"` / `"!s:"` stringification
  should be replaced by a compact tagged-array form
  `[{"$k": ..., "$v": ...}, ...]` for readability. This draft
  picks the prefix encoding for byte-density; a v2.1 capability
  frame could opt into the tagged-array form.

---

*This draft tracks step 44 of `100_STEPS.md`.*
