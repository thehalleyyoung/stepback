# RFC 0002 — Canonical JSON Encoding

| Field | Value |
|---|---|
| RFC number | 0002 |
| Title | Canonical JSON Encoding |
| Status | Draft |
| Supersedes | — |
| Created | 2026-05-12 |
| Authors | stepback maintainers |
| Reference impl | `stepback.canonical` (`stepback >= 0.1`) |

---

## Abstract

This RFC specifies the *canonical JSON* encoding used for all SB-Trace
(RFC 0001) frame bodies, hash inputs, and attestation pack bodies.  Two
encoders producing canonical JSON for the same value MUST emit identical
bytes.  This is the foundation for reproducible BLAKE2b-256 hashes and
Ed25519 signatures across Python, Rust, TypeScript, Go, JVM, and .NET
implementations.

The reference implementation is `stepback.canonical.canonical_json` in
[`stepback/canonical.py`](../../stepback/canonical.py).  The authoritative
prose companion is [`docs/canonicalization.md`](../../docs/canonicalization.md).
This RFC summarises the normative rules in a form suitable for external
review.

---

## 1. Motivation

Hash-based cache invalidation (dirty-set replay, RFC 0003) and
tamper-evident HMAC chains (RFC 0001) both require that any two
implementations compute the same bytes for the same logical value.
Standard JSON allows multiple encodings for the same value (key ordering,
float precision, Unicode escaping), which would break cross-implementation
hash agreement.  Canonical JSON fixes these choices.

---

## 2. Normative rules

### 2.1 Object key ordering

Object keys MUST be sorted **ascending by their Unicode code-point
sequence** (equivalently: by their UTF-8 byte sequence, since UTF-8 is
order-preserving over code points for valid Unicode).

### 2.2 Whitespace

No insignificant whitespace is emitted between tokens.  The item separator
is `,`; the key-value separator is `:`.

### 2.3 Strings

String values MUST be valid Unicode (no unpaired surrogates).  The
encoding MUST be UTF-8 (in the JSON layer, `\uXXXX` escape sequences are
valid but MUST only be used for the mandatory control-character escapes
`\b`, `\f`, `\n`, `\r`, `\t`, and `\\`, `\"`.  Non-ASCII printable
characters MUST be emitted as their literal UTF-8 bytes, not as
`\uXXXX` sequences.

NFC normalisation of string values is NOT performed by the canonical
encoder; callers MUST normalise inputs to NFC before canonicalising if
cross-language stability is required for human-typed text.

### 2.4 Numbers

Integer values MUST be emitted without a decimal point or exponent
(e.g. `42`, not `42.0`).

Floating-point values MUST be finite (no `NaN`, `Infinity`,
`-Infinity`).  The encoder MUST raise `CanonicalError` for non-finite
floats.

Float serialisation uses Python's `repr`-equivalent shortest-round-trip
encoding.  Implementations in other languages MUST use the Grisu3/Ryu or
equivalent shortest-round-trip algorithm to ensure identical byte output.
Specifically: `0.1` → `"0.1"`, `1.0` → `"1.0"`, `1e100` → `"1e+100"`.

### 2.5 Booleans and null

`True` → `true`, `False` → `false`, `None` → `null`.

### 2.6 Arrays

Arrays are emitted in iteration order; no sorting is applied.

### 2.7 Non-JSON Python types (reference implementation only)

The Python reference encoder handles a closed extension list:

| Python type | Canonical representation |
|---|---|
| `tuple` | JSON array (same as `list`) |
| `set` / `frozenset` | JSON array, elements sorted by their canonical JSON representation |
| `bytes` | JSON string, `base64`-encoded without padding |

Any other non-serialisable type MUST raise `TypeError`.

### 2.8 Top-level type

Any JSON value type is permitted at the top level (object, array, string,
number, boolean, null).

---

## 3. Canonical hash function

```
canonical_hash(value) = BLAKE2b-256(canonical_json(value))
```

Encoded as the hex string `"blake2b:<hex>"`.

All `inputs_hash` and `nondeterminism_hash` fields in SB-Trace step frames
are canonical hashes in this format.

---

## 4. Test vectors

| Input (Python repr) | Canonical JSON bytes (hex) |
|---|---|
| `{}` | `7b7d` |
| `{"b": 1, "a": 2}` | `7b2261223a322c2262223a317d` |
| `{"k": "hëllo"}` | `7b226b223a2268c3ab6c6c6f227d` |
| `[1, 2, 3]` | `5b312c322c335d` |
| `True` | `74727565` |
| `None` | `6e756c6c` |

The full test-vector suite is exercised by `tests/test_canonical.py`.

---

## 5. Security considerations

- The canonical encoder MUST reject non-finite floats; allowing them would
  make the hash undefined in languages without `NaN` / `Infinity` support.
- Surrogate Unicode code points MUST be rejected; they are not valid UTF-8
  and implementations disagree on how to encode them.
- The choice of BLAKE2b-256 over SHA-256 is for performance; the security
  level (128-bit collision resistance) is equivalent.

---

## 6. Relationship to other RFCs

- RFC 0001 §5 describes how canonical JSON is used in the HMAC chain.
- RFC 0003 §2 describes how `canonical_hash` drives dirty-set decisions.

---

## Appendix A — Changelog

| Date | Author | Change |
|---|---|---|
| 2026-05-12 | stepback maintainers | Initial draft |
