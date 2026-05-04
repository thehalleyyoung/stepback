# Canonicalization (v1)

This document is the prose companion to `spec/sbtrace-v1.md` §4 ("Canonical
JSON") and the reference implementation in
[`stepback/canonical.py`](../stepback/canonical.py). It explains, axis by
axis, *what* gets canonicalized, *why*, and which inputs are illegal under
v1 so that two independent encoders are guaranteed to produce identical
bytes for any value the recorder is allowed to record.

The pinned name of this format is `canonicalisation_version="1"`. It is
recorded in every trace's header frame; readers MUST refuse to verify
hashes against a body whose canonicalisation version they do not
understand. A v2 candidate (deterministic CBOR, RFC 8949 §4.2) is tracked
in [Step 43 of `100_STEPS.md`](../100_STEPS.md) and is **not** described
here.

> **One-line definition.** `canonical_json(x)` is `json.dumps(x,
> sort_keys=True, separators=(",", ":"), ensure_ascii=False,
> allow_nan=False, default=_default).encode("utf-8")` where `_default`
> handles a closed list of non-JSON Python types (tuples, sets, bytes).
> Anything outside that domain is a programming error and raises
> `TypeError`.

## Why this matters

Every cache hit, every dirty-set decision, every `.sb` HMAC chain step,
and every Ed25519 signature is computed over `canonical_json(body)`.
If two implementations disagree on a single byte for any legal input,
they disagree on every downstream artifact — caches stop hitting, replay
diverges, and conformance tests in `spec/schema/` fail. The whole point
of this document is to nail the encoder down tightly enough that
"identical inputs ⇒ identical bytes" is a checkable property and not a
folklore promise.

---

## 1. Ordering

### 1.1 Object keys

Object (Python `dict`) keys MUST be sorted **ascending by their
UTF-8 byte sequence**. This is what `json.dumps(sort_keys=True)` does:
Python sorts the `str` keys, and since Python 3 strings are sequences of
Unicode code points which are unambiguously encoded as UTF-8 in JSON, the
"code-point ordering" and "UTF-8-byte ordering" are *the same total
order* for valid Unicode (UTF-8 is order-preserving over code points).
Encoders in other languages MUST therefore sort by either the code-point
sequence of the decoded string or by the UTF-8 byte sequence — both
yield identical results for any non-surrogate input.

Keys MUST be strings. Encoders MUST refuse non-string keys (`int`,
`float`, `bool`, `None`, tuples, frozensets, …). The Python reference
encoder relies on `json.dumps`, which raises `TypeError` for non-string
keys when `sort_keys=True`.

There is no concept of "key rewriting" in v1: keys are emitted as-is
(after the JSON string-escape rules in §3 below). In particular:

- Whitespace inside keys is preserved.
- Casing is preserved (`"Model"` and `"model"` are distinct keys).
- Unicode normalization is **not** applied automatically; see §3.4.

### 1.2 Array element order

Arrays preserve insertion order. v1 has no notion of "canonicalize the
elements of a list"; if the producer cares about order-independence, it
must sort the list before handing it to `canonical_json`. Tools, model
messages, and step children are all *ordered* in v1, and reordering
them would change the canonical bytes (and therefore the dirty-set
inputs hash).

### 1.3 Sets and frozensets

Python `set` and `frozenset` are not native JSON types. The v1 encoder
serializes them as a JSON array of their **sorted** elements:

- The encoder first attempts `sorted(o)` (natural Python order).
- If that raises `TypeError` (heterogeneous element types — e.g., a set
  containing both `int` and `str`), it falls back to `sorted(o,
  key=repr)`.

This rule is intentionally narrow. The recorder writes
JSON-shaped data and never produces a set; the path exists only so that
*caller-supplied* inputs (e.g., a user passing a set into a `substitute`
call to be hashed) round-trip deterministically.

### 1.4 Tuples

Python `tuple` serializes as a JSON array, in tuple order. There is no
distinction between `tuple` and `list` in canonical bytes.

---

## 2. Numbers

JSON has one number type. The canonical encoder leans on Python's
`json.dumps`, which means:

### 2.1 Integers

- All Python `int` values, regardless of magnitude, serialize as a JSON
  numeric literal in base 10 with no leading zeros, no decimal point, no
  exponent, and no `+` sign.
- `-0` (Python `int`) does not exist; `0` serializes as `0`.
- `bool` is a subclass of `int` in Python but the JSON encoder writes
  it as `true` / `false` (see §6).

### 2.2 Non-integer floats

- v1 traces SHOULD avoid emitting free-form floats. The recorder writes
  costs in **micro-USD as integers** and timings in **integer
  nanoseconds**. Model parameters (e.g., `temperature: 0.7`) are passed
  through as the recorder received them, so floats can appear inside
  request bodies that originated from user code.
- Where floats *do* appear, the encoder uses Python's shortest
  round-trippable form (the form `json.dumps` emits with default
  settings; this is Python's `repr` for floats and is equivalent to the
  IEEE 754 "shortest such that `float(s) == x`" rule used by
  ECMAScript and Go's `strconv.FormatFloat(x, 'g', -1, 64)`).
- `NaN`, `+Infinity`, `-Infinity` are **rejected**. The encoder is
  configured with `allow_nan=False`. Any producer that wants to record a
  non-finite value MUST first encode it explicitly (e.g., as the string
  `"NaN"` or as a wrapper object whose schema documents the convention).

### 2.3 Decimals and arbitrary-precision numbers

- v1 does **not** support `decimal.Decimal`, `fractions.Fraction`,
  `numpy` numeric types, or any other arbitrary-precision number. The
  default hook in `canonical.py::_default` does not handle these, so
  passing one raises `TypeError`.
- Producers that need exact-decimal arithmetic (e.g., billing
  applications computing per-token costs in fractional cents) MUST
  serialize the value themselves to a string before handing it to the
  recorder. The exact convention is at the schema's discretion.
- A future canonicalisation_version may adopt CBOR tag 4 (decimal
  fraction) and tag 30 (rational); v1 deliberately does not, because
  there is no universally-agreed JSON encoding for decimals (see
  RFC 7159 §6) and we prefer "reject" over "two encoders disagree".

### 2.4 Negative zero

`-0.0` (Python `float`) serializes as `-0.0` and is not equal to `0` or
`0.0` under the canonical bytes comparison. Producers that conflate the
two (e.g., probability post-processors) SHOULD normalize to `+0.0`
before recording, or accept that a downstream substitution that touches
the sign of zero will trigger a dirty-set re-execution.

---

## 3. Strings and Unicode

### 3.1 Encoding

Output is **UTF-8**, no BOM. Python's `json.dumps(...,
ensure_ascii=False)` emits non-ASCII characters as their direct UTF-8
encoding, *not* as `\uXXXX` escapes. This was an explicit choice: the
recorder routinely captures non-ASCII text (LLM outputs, user prompts in
languages other than English) and we did not want canonical output
files to bloat by ~6× per non-ASCII code point, nor for two encoders to
diverge on whether `é` is `\u00e9` or the two-byte UTF-8 sequence
`c3 a9`.

### 3.2 Mandatory escapes

Per RFC 8259 §7, the following bytes inside a JSON string MUST be
escaped:

- `U+0000`–`U+001F` (control characters): `\u0000`–`\u001f`. The
  short-form escapes `\b`, `\t`, `\n`, `\f`, `\r` are emitted by
  Python's `json` module for `\b \t \n \f \r` respectively; v1
  conformant encoders MUST emit the same short forms (this is what
  Python's `json` module does, what Go's `encoding/json` does, and what
  Rust's `serde_json` does, so this is not actually a divergence
  point in practice).
- `"` → `\"`.
- `\` → `\\`.

The forward slash `/` MUST NOT be escaped. Some libraries default to
`\/` for HTML-embedding safety; v1 forbids that to keep bytes
deterministic.

`U+007F` (DEL) MUST NOT be escaped (it is not a control character per
RFC 8259's §7 grammar). Conformant encoders pass it through verbatim as
the byte `0x7F`.

### 3.3 Surrogates

JSON strings model a sequence of *Unicode scalar values*, not a sequence
of UTF-16 code units. v1 forbids both:

- **Lone surrogates** (`U+D800`–`U+DFFF` not in a valid pair). A Python
  `str` cannot contain a lone surrogate without `surrogatepass`-style
  encoding; if one slips through (e.g., via a `bytes`-decoded input that
  used `errors="surrogateescape"`), `canonical_json` raises
  `UnicodeEncodeError`. Conformant encoders MUST reject lone
  surrogates.
- **`\uXXXX\uYYYY` surrogate-pair escapes** in the *output*: forbidden
  because we always emit `ensure_ascii=False`. (A character above
  U+FFFF is emitted as its 4-byte UTF-8 encoding, never as a UTF-16
  surrogate pair.) A reader MAY parse legacy JSON containing surrogate
  pair escapes — the decoded value is what gets canonicalized — but a
  *writer* MUST NOT produce them.

### 3.4 Normalization

v1 does **not** apply Unicode normalization. The string `"caf\u00e9"`
(NFC, 4 code points) and `"cafe\u0301"` (NFD, 5 code points) are
distinct canonical bytes and have distinct hashes. The rationale is
that the recorder captures user/model strings byte-for-byte and any
silent NFC/NFD rewrite would surprise applications doing
forensics on the original prompt.

The Unicode canonicalization tests (Step 28 in `100_STEPS.md`) assert
both that the encoder **does not** normalize and that the encoder
**does** detect lone surrogates as canonicalisation errors.

### 3.5 BOMs and zero-width characters

A leading BOM (`U+FEFF`) inside a string is preserved verbatim. It is
not stripped, normalized, or rejected. Likewise, zero-width joiner /
non-joiner code points (`U+200D`, `U+200C`) are preserved. Producers
that want "what the user typed" canonicalization should strip these
themselves before recording.

---

## 4. Bytes

JSON has no native byte-string type. The v1 encoder represents Python
`bytes` and `bytearray` as the single-key object

    {"__bytes_hex__": "<lowercase hex>"}

with no length prefix and no separators inside the hex string. Hex
digits are `[0-9a-f]`; uppercase is forbidden so two encoders cannot
disagree on case. The wrapper object participates in key sorting like
any other object — i.e., a sibling key `"__zzz__"` would sort *after*
`"__bytes_hex__"`, since `b < z` in UTF-8 byte order.

This shape is intentionally awkward (long, ugly, obviously a wrapper)
because (a) it must round-trip through stock JSON parsers in other
languages, and (b) we want it to be impossible to confuse with a
user-supplied object that happens to have one key called
`__bytes_hex__`. v2 (CBOR) will use CBOR major type 2 (byte strings)
and drop the wrapper.

In `.sb` v1, the recorder does not emit raw bytes inline — large blobs
are written as `blob` frames and referenced by their `blob_id` (an
already-canonical `sha256:<hex>` string), so the `__bytes_hex__`
wrapper appears almost exclusively in caller-supplied substitution
inputs and in test fixtures.

---

## 5. Maps and structured values

### 5.1 Maps with non-string keys

Forbidden, as in §1.1. There is no automatic stringification of integer
or tuple keys; producers MUST do that explicitly with a documented
convention.

### 5.2 Empty values

- `{}` and `[]` serialize as exactly those two-byte sequences.
- A key whose value is `None` serializes as `"<key>":null`. The
  encoder does NOT drop null-valued keys. Producers that want
  "absent ⇔ null" semantics MUST omit the key.

### 5.3 Nested maps

Nesting is not bounded by the canonicalizer; it is bounded by the
reader (see §11 of the spec — implementation limits / DoS bounds). The
canonicalizer recurses freely until it hits Python's recursion limit.
Conformant readers MUST refuse to *parse* a frame deeper than the
documented bound; canonical output that exceeds the bound is the
producer's bug, not the encoder's.

### 5.4 Field schemas

The closed set of fields per frame kind is described in §6 of the spec
and (more formally) in `spec/schema/`. The canonicalizer is
schema-agnostic: it does not know that a `header` frame "should"
contain `recorder_version`. It encodes whatever you hand it. Schema
validation lives in `stepback.spec` and runs separately from
canonicalization.

---

## 6. Booleans and null

- `True` → `true`. `False` → `false`. `None` → `null`.
- These literals are emitted in lowercase exactly. There is no
  alternative spelling.
- The encoder does **not** treat `0` and `False` as equivalent, even
  though Python does for `==`. They serialize as `0` and `false`
  respectively, with distinct canonical bytes.

---

## 7. Timestamps

v1 records all timestamps as **`wallclock_ns`: integer nanoseconds
since the Unix epoch (1970-01-01T00:00:00Z)**. There is no string
timestamp format (`"2025-05-04T03:00:00Z"`, ISO 8601, RFC 3339, …)
anywhere in the canonical schema, and there is no timezone field —
all times are in UTC by construction.

Why integer ns:

- Avoids the float / decimal / leap-second / format-disagreement zoo
  that ISO 8601 introduces. There is exactly one byte sequence for any
  given instant.
- Survives `canonical_json` without going through float at all.
- Matches what most kernels and clock libraries already give us
  (`clock_gettime(CLOCK_REALTIME)` × 1e9, `time.time_ns()` in
  Python ≥ 3.7, `Instant::now()` minus an epoch in Rust).
- Range: `2^63 - 1` ns ≈ 292 years past the epoch, so dates up to ≈
  2262 fit in a signed 64-bit integer. JSON numbers are
  arbitrary-precision conceptually, but readers MAY require that
  `wallclock_ns` fits in `int64` — see §11 of the spec.

Producers that have only a millisecond-resolution clock SHOULD
multiply by 1_000_000 and accept the obvious "trailing six zeros"
caveat. Producers that have a sub-nanosecond clock SHOULD truncate
(not round) to nanoseconds; rounding policies can disagree across
languages, truncation cannot.

Durations (e.g., `latency_ns` inside step bodies) follow the same rule:
non-negative integer nanoseconds. There is no `latency_ms` /
`latency_seconds` / `latency_iso8601` field.

---

## 8. Model identifiers

Model ids are free-form strings as far as canonicalization is
concerned — they go through §3 like any other string. However, two
provider-specific conventions sit *on top of* canonicalization to keep
caches and pricing keys stable:

### 8.1 Bedrock alias collapse

`stepback.shims.canonical_bedrock_model_id` collapses Amazon Bedrock's
provider-prefixed ARNs to the underlying model id used by the price
list. For example:

| Recorded (from SDK) | Canonical (after collapse) |
| --- | --- |
| `anthropic.claude-3-5-sonnet-20241022-v2:0` | `claude-3-5-sonnet-20241022` |
| `anthropic.claude-sonnet-4-20250514-v1:0` | `claude-sonnet-4-20250514` |
| `meta.llama3-...` | (unchanged — Bedrock-native model) |

The full alias table is in
[`stepback/shims.py`](../stepback/shims.py) (`_BEDROCK_PRICING_ALIAS`).
The recorder calls `canonical_bedrock_model_id` *before* writing the
step body, so what lands in `inputs["model"]` is already the canonical
form, and the canonical JSON bytes therefore agree with what the
pricing engine sees.

### 8.2 Gemini alias collapse

Symmetric rule for Google Gemini, in
`canonical_gemini_model_id` / `_GEMINI_PRICING_ALIAS`. Models passed
without a `models/` prefix or with a regional/version suffix are
collapsed to the price-list key.

### 8.3 OpenAI / Anthropic native ids

OpenAI and Anthropic native ids are already in canonical form (e.g.,
`gpt-4o-2024-08-06`, `claude-3-5-sonnet-20241022`); no rewrite is
applied. The recorder records them verbatim.

### 8.4 Snapshots vs. aliases

`gpt-4o` (a moving alias) and `gpt-4o-2024-08-06` (a pinned snapshot)
are *different strings* and *different canonical bytes*. The recorder
does not auto-resolve aliases to their current snapshot; that would
make caches invalidate every time the provider rolled the alias. If
your application wants snapshot-stable hashes, ask the provider for a
snapshot id explicitly and record that.

### 8.5 Substitutions

A `model` substitution (`ModelSubstitution(at_step=..., new_model_id=...)`
in `stepback/substitutions.py`) writes the new model id directly into
the canonical inputs. The dirty-set algorithm then hashes the new
inputs; a model swap is therefore visible at exactly the same
granularity as a tool-result swap.

---

## 9. Unknown fields

v1 takes a deliberately asymmetric stance:

### 9.1 Writers MUST NOT emit unknown fields

A v1 writer SHOULD only emit fields that are in the v1 schema for the
frame kind in question. Adding an extension key like `__my_company__`
to a step body is technically legal canonical JSON — the encoder will
sort it like any other string key — but it pollutes the hash and will
break differential conformance against any other v1 writer that was
*not* told about that extension.

If you want to add fields, do it through the capability-negotiation
mechanism (§13 of the spec) by bumping `format_version` to a v1.x
draft or by carrying the data in an explicit extension frame kind, not
by sneaking keys into existing bodies.

### 9.2 Readers MUST tolerate unknown fields *inside step bodies*

By contrast, a v1 *reader* MUST NOT fail when it encounters an
unrecognized key in a step body. This is what enables forward
compatibility: a v1.1 producer that adds a `cache_breadcrumb` field to
the step body can be read (with that field ignored) by a v1.0 consumer.
Concretely, `stepback.spec.SBTraceSpec` reports `step.unknown_field` as
a *warning*, not an error. Round-trip of the body bytes is preserved,
and the canonical hash continues to verify because the reader
canonicalizes the **whole received body**, including the unknown
field.

### 9.3 Headers and tails are stricter

`header.unknown_field` and the analogous tail-frame check are
*errors* in `stepback.spec`, because those frames are versioned
explicitly and the negotiation envelope is the only safe place to
extend them. Adding a new header field is a wire-format SemVer event
(see Step 22 in `100_STEPS.md`).

### 9.4 Consequences for hashing

Because the canonicalizer is schema-agnostic, an unknown field is
*part of the canonical bytes* and contributes to `inputs_hash`,
`outputs_hash`, and the HMAC chain. Two readers that disagree on
"keep" vs. "strip" of the same unknown field would compute different
hashes and reject the trace. The rule is therefore: **never strip on
read.** A reader either accepts the trace as-is (hashing the bytes the
producer wrote) or rejects it; there is no "accept but rewrite"
mode.

---

## 10. Worked examples

### 10.1 Empty object and empty array

| Input (Python) | Canonical bytes |
| --- | --- |
| `{}` | `{}` |
| `[]` | `[]` |
| `{"a": []}` | `{"a":[]}` |

### 10.2 Key sorting

| Input | Canonical bytes |
| --- | --- |
| `{"b": 1, "a": 2}` | `{"a":2,"b":1}` |
| `{"a10": 1, "a2": 1}` | `{"a10":1,"a2":1}` (lex on UTF-8 bytes; `'1' < '2'`) |
| `{"é": 1, "z": 1}` | `{"z":1,"é":1}` (`'z'` is `0x7A`; `'é'` is `0xC3 0xA9`) |

### 10.3 Booleans, null

| Input | Canonical bytes |
| --- | --- |
| `True` | `true` |
| `False` | `false` |
| `None` | `null` |
| `{"x": None}` | `{"x":null}` |

### 10.4 Numbers

| Input | Canonical bytes |
| --- | --- |
| `0` | `0` |
| `-0` | `0` (Python collapses) |
| `0.0` | `0.0` |
| `-0.0` | `-0.0` |
| `1.5` | `1.5` |
| `1e100` | `1e+100` |
| `float("nan")` | *raises* `ValueError` (allow_nan=False) |
| `decimal.Decimal("1.5")` | *raises* `TypeError` |

### 10.5 Bytes

| Input | Canonical bytes |
| --- | --- |
| `b""` | `{"__bytes_hex__":""}` |
| `b"\xde\xad\xbe\xef"` | `{"__bytes_hex__":"deadbeef"}` |

### 10.6 Sets and tuples

| Input | Canonical bytes |
| --- | --- |
| `(1, 2, 3)` | `[1,2,3]` |
| `{3, 1, 2}` | `[1,2,3]` |
| `frozenset({"b", "a"})` | `["a","b"]` |
| `{1, "a"}` | sorted by `repr`, deterministic but ugly |

---

## 11. Test surface

The canonicalizer is exercised by, among others:

- `tests/test_canonical.py` — basic round-trip and shape rules.
- `tests/test_unicode_canonicalization.py` — §3.4 (no normalization),
  §3.3 (surrogate rejection), §3.2 (escape rules) (Step 28).
- `tests/test_canonical_property.py` — Hypothesis property tests for
  round-trip and order-independence over arbitrary JSON-shaped values
  (Step 27).
- `tests/test_dirty_set_property.py` — uses canonical bytes to assert
  that "dirty-set replay = full replay" for randomly generated traces
  (Step 29).
- Differential tests in `bindings/` and `stepback-core/` — every
  language binding hashes the same fixtures and MUST agree byte-for-byte
  (Step 32).
- `tests/test_canonical_smt_equivalence.py` and
  `tests/test_canonical_differential.py` — Z3-mediated equivalence
  proof of the Python canonicaliser against an independently
  prose-derived reference (`spec/canonical/bounded.py`), plus a
  cross-language differential runner that drives Python, Rust, and
  TypeScript canonicalisers over the same ~1,890-element bounded JSON
  subset and asserts pairwise byte-equality. The bounded subset is
  defined formally in `spec/canonical/bounded.md`. Go, JVM, and .NET
  ship as read-only verifiers in v0.1; the differential report
  documents this rather than silently dropping them, so a future
  writer-side binding will pick the check up automatically (Step 48).

If you change `canonical.py`, update *all* of these and bump
`canonicalisation_version`.

---

## 12. References

- `spec/sbtrace-v1.md` §4 (normative).
- `stepback/canonical.py` (reference implementation).
- RFC 8259 (JSON grammar).
- RFC 8785 (JCS — a different, more elaborate canonical JSON; v1 is
  *not* JCS-compatible because JCS uses ECMAScript-style number
  formatting and a distinct sort order. We considered it; we rejected
  it because pinning to Python's `json.dumps` is what gives us
  byte-identity with the Python reference encoder for free).
- RFC 8949 §4.2 (deterministic CBOR; relevant to v2 only).
  An experimental Python reference encoder lives in
  `stepback/canonical_cbor.py` (`canonical_cbor`, `sha256_hex_cbor`).
  It is **not** wired into v1 `.sb` writing or hashing; it exists
  so that v2 candidate fixtures can be pinned and independent
  implementations (Rust, TypeScript, Go, JVM, .NET) can be cross-
  checked against the same byte sequences before any v2 reader/writer
  ships. See `tests/test_canonical_cbor.py` for the locked-in vectors
  (RFC 8949 Appendix A plus the §4.2.1 map-key sort rule).
