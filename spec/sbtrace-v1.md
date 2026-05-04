# SB-Trace v1 — Wire-format Specification

**Status:** Draft 1 (informational; tracks the v1 implementation shipping in
`stepback >= 0.1`).
**Format version:** `1`
**Canonicalisation version:** `1`
**License:** Apache-2.0.

This document is the normative description of the `.sb` (SB-Trace) v1 byte
layout, frame model, hashing rules, HMAC-chain construction, and Ed25519
signature construction. It also contains a fully worked hex example,
reproducible from the deterministic fixtures shipped under
`stepback-core/fixtures/v1/`.

The intended audience is anyone writing an independent reader, writer,
verifier, or conformance harness. v1 is JSON-framed (see §3); CBOR is a
candidate v2 encoding and is **not** part of v1.

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD
NOT**, **MAY**, and **OPTIONAL** in this document are to be interpreted as
described in BCP 14 (RFC 2119, RFC 8174).

---

## 1. Goals and non-goals

### 1.1 Goals

* **Append-only**, streamable, crash-safe writes — a writer that dies after a
  partial frame leaves a tail that any conforming reader rejects cleanly.
* **Tamper-evident**: every frame carries a per-frame HMAC chained to the
  previous frame and a per-frame Ed25519 signature over that HMAC.
* **Content-addressed**: the canonical encoding of every value is hashed
  with SHA-256 in the form `sha256:<lowercase hex>` so two semantically
  identical inputs collide on cache lookup.
* **Schema-versioned**: the header pins `format_version`,
  `canonicalisation_version`, `recorder_version`, and
  `price_list_version`. Old readers reject unknown mandatory capabilities;
  new readers continue to read v1 forever.
* **Single byte sequence per trace** — no sidecar files required for
  verification. Attestation packs reference traces but are out of scope here.

### 1.2 Non-goals (for v1)

* CBOR, MessagePack, or any binary frame encoding. v1 frames are **canonical
  UTF-8 JSON** (§4).
* Random-access reads. Readers MUST scan from byte 0; an index frame is a
  v2 candidate.
* Encryption at rest. Confidentiality is the caller's job — v1 protects
  integrity and authenticity, not secrecy.
* Multi-writer concurrency to the same file. One trace, one writer.

---

## 2. Terminology

* **Trace.** The full byte sequence written to a single `.sb` file.
* **Frame.** A length-prefixed record (see §3). The unit of read, write,
  HMAC, and signature.
* **Body.** The semantic payload of a frame — a JSON object whose `type`
  field discriminates the frame kind (header / capability / step / blob /
  tail; see §6).
* **Wrapper.** The on-disk JSON object that carries the body alongside its
  receipt fields (`prev_hmac`, `hmac`, `sig`).
* **Receipt.** The `(prev_hmac, hmac, sig)` triple inside the wrapper.
* **Canonical JSON.** The exact byte-level encoding rules in §4. The hash,
  HMAC, and signature inputs all consume canonical-JSON bytes.
* **Step.** One unit of agent execution: an LLM call, tool call, router
  decision, policy check, MCP call, or parallel-branch open/join/exception.
* **Capability.** A named extension declared in a capability frame (§6.2).
  Mandatory capabilities a reader does not understand MUST cause it to
  reject the trace.

---

## 3. Byte layout

A v1 trace is a sequence of one or more frames. Each frame is::

    +------------------+----------------------------------+
    | length (4 bytes) | wrapper bytes (length bytes)     |
    +------------------+----------------------------------+

* **`length`** is a 4-byte big-endian unsigned integer (`uint32be`) giving
  the byte length of the wrapper that immediately follows. `length` MUST be
  > 0. Readers SHOULD enforce an upper bound (the reference reader uses
  16 MiB; see §11).
* **`wrapper bytes`** is exactly `length` bytes of canonical UTF-8 JSON
  (§4) that decode to the wrapper object described in §5. The wrapper is
  not nul-terminated. There is no padding, no separator, and no trailing
  newline between frames.

A reader MUST reach EOF exactly at the end of a wrapper. A trailing
partial length prefix or a length prefix followed by fewer than `length`
bytes MUST be rejected as a truncated trace (`UnexpectedEof`).

There is no overall file header or magic outside the first frame's body
(`type=header`, see §6.1). The magic string `stepback/.sb` lives inside
the header body and MUST be checked by readers.

---

## 4. Canonical JSON (canonicalisation v1)

`canonicalisation_version="1"` is exactly what the reference encoder in
`stepback/canonical.py::canonical_json` emits. A second implementation is
conformant iff it produces the same bytes for the same input value on the
bounded domain below.

### 4.1 Encoding rules

A canonical JSON encoder MUST:

1. Produce **UTF-8** bytes. No BOM. No surrogate escapes for characters
   that have a direct UTF-8 representation; that is, `ensure_ascii=False`
   (Python `json.dumps` semantics). Characters in the range
   `U+0000..U+001F`, `U+0022` (`"`), and `U+005C` (`\`) MUST be escaped
   per RFC 8259 §7. All other code points MUST be emitted verbatim as
   their UTF-8 encoding.
2. **Sort object keys** by their UTF-8 byte sequence, ascending.
3. Use the **separators `,` and `:`** with no surrounding whitespace
   between tokens. There is no whitespace anywhere outside string
   contents.
4. Reject `NaN`, `+Infinity`, `-Infinity` (`allow_nan=false`).
5. Encode integers with no leading zeros and no decimal point. Encode
   non-integer numbers using Python's `repr`-equivalent shortest round-trip
   form (the form `json.dumps` emits with default settings); this is
   permitted because v1 traces SHOULD avoid free-form floats — costs and
   timings are integer ns / micro-USD, and model parameters are passed
   through as the recorder received them. (Decimals and arbitrary-precision
   numbers are a v2 concern.)
6. Encode `true`, `false`, and `null` using the lowercase JSON literals.
7. Encode `bytes` / `bytearray` (where permitted by the schema) as the
   single-key object `{"__bytes_hex__": "<lowercase hex>"}`. This object
   participates in key sorting like any other.
8. Encode Python `tuple` / `list` as JSON arrays. Encode `set` / `frozenset`
   as a JSON array of the **sorted** elements (sort by natural order; if
   that fails, sort by `repr`). The recorder writes lists, so this rule
   only matters for hashing user-supplied inputs to canonical form.
9. Refuse any other Python type (raise `TypeError`). New types MUST be
   added by bumping `canonicalisation_version` and pinning the new value
   in the header.

### 4.2 Hash form

A SHA-256 hash that appears in a body (`inputs_hash`, `outputs_hash`,
`nondeterminism_hash`, blob ids in §6.4, etc.) MUST be the string

    "sha256:" + lowercase_hex(SHA256(canonical_json(value)))

64 hex characters preceded by the literal prefix `sha256:`. Other digest
algorithms are reserved for v2.

---

## 5. Wrapper object

Every frame's wrapper is a JSON object with **exactly** these four keys:

| Key | Type | Description |
| --- | --- | --- |
| `body` | object | The frame body (§6). MUST contain a `type` field. |
| `prev_hmac` | string | 64 lowercase hex chars. The `hmac` field of the previous frame, or `"00" * 32` for the first frame. |
| `hmac` | string | 64 lowercase hex chars. `HMAC_SHA256(K_h, prev_hmac_bytes ‖ canonical_json(body))`. See §7. |
| `sig` | string | `"ed25519:" + lowercase_hex(Ed25519_sign(K_s, hmac_bytes))`. See §8. |

The wrapper MUST contain no other keys. Readers MUST reject wrappers with
unknown keys when verifying a trace; this lets the spec evolve by version
bump rather than by silent extension.

The wrapper itself is canonical-JSON encoded with the §4 rules. Because
key sorting is alphabetical, the on-disk byte order of wrapper keys is
**always** `body`, `hmac`, `prev_hmac`, `sig`.

---

## 6. Frame kinds

The frame kind is determined by `body.type`. Five kinds are defined in v1.

### 6.1 Header (`type="header"`) — REQUIRED, exactly once, first

The first frame in a trace MUST be a header. A reader MUST reject any
trace whose first frame is not a header.

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `type` | string | yes | `"header"` |
| `magic` | string | yes | `"stepback/.sb"` |
| `format_version` | integer | yes | `1` for v1. |
| `recorder_version` | string | yes | SemVer of the recorder that wrote this trace. |
| `canonicalisation_version` | string | yes | `"1"` for v1. |
| `public_key` | string | yes | 64 hex chars: the Ed25519 public key whose private half signed every frame in this trace. |
| `hmac_key_id` | string | yes | First 16 hex chars of `SHA256(hmac_key)`. Identifies the HMAC key without revealing it. |
| `price_list_version` | string | yes | Pins the cost model used for `cost_usd` fields in step frames. |
| `wallclock_ns` | integer | yes | Recorder wall-clock at trace open, ns since Unix epoch. |
| `compression` | string | yes | `"none"` or `"gzip+dedup-2"` (the v1 inline compression scheme; see §6.4 and §10). |
| `blob_threshold` | integer | yes | Minimum canonical-JSON byte size considered for interning (0 if `compression="none"`). |
| `blob_min_reuse` | integer | yes | Minimum reuse count for a sub-tree to be interned (0 if `compression="none"`). |

A reader MUST reject any header with `format_version != 1` (unless it
explicitly claims v2+ support and falls through to a different parser),
`magic != "stepback/.sb"`, or with a `canonicalisation_version` it does
not implement.

### 6.2 Capability (`type="capability"`) — OPTIONAL, zero or more

Capability frames declare named extensions the writer relied on. They MAY
appear after the header and before the tail, in any order, interleaved
with steps, blobs, and other capabilities.

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `type` | string | yes | `"capability"` |
| `name` | string | yes | Non-empty extension identifier (e.g. `"core"`, `"compression.gzip-dedup-2"`). |
| `mandatory` | boolean | yes | If `true`, a reader that does not implement `name` MUST reject the trace. If `false`, the capability is advisory and unknown names MAY be ignored. |
| `params` | object | no | Extension-specific parameters. Opaque to the v1 spec. |

The implicit `core` capability is always supported and need not be
declared. The reference reader's allow-list lives in
`stepback.spec._SUPPORTED_CAPABILITIES_V1` and is overridable per-call.

### 6.3 Step (`type="step"`) — REQUIRED for any non-empty trace, zero or more

A step frame records one unit of agent execution. Two encodings are
permitted:

* **Plain** — `body = {"type":"step", "step": <step object>}`. The
  `step` object is the canonical-JSON form of the recorded step (see
  §6.5).
* **Compressed** — `body = {"type":"step", "encoding":"gzip+base64",
  "data": "<base64>"}`. The base64-decoded bytes, gunzipped, MUST be
  the canonical-JSON encoding of the same step object that the plain form
  would have carried. Compressed step frames MUST only appear when the
  header declared `compression="gzip+dedup-2"`.

A writer that chooses the compressed form MUST do so only when the
compressed wrapper saves space versus the plain form (the reference
writer applies a 24-byte slack to avoid net regressions on small
steps). Readers MUST accept either form regardless of size.

### 6.4 Blob (`type="blob"`) — OPTIONAL, zero or more

Blob frames intern recurring sub-trees so they can be referenced from
many step frames by a single content hash. They appear when the header
declares `compression="gzip+dedup-2"`.

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `type` | string | yes | `"blob"` |
| `id` | string | yes | 64 lowercase hex chars: `SHA256(canonical_json(value))`. **No** `sha256:` prefix in this slot — the prefix is implicit for the blob table. |
| `encoding` | string | yes | `"json"` (the `data` field is the canonical-JSON encoding of the value as a UTF-8 string) or `"gzip+base64"` (base64 of gzipped canonical-JSON). |
| `data` | string | yes | The encoded value. |

A blob frame MUST be emitted **before** the first step frame that
references it. A step references a blob by replacing the sub-tree with
the single-key object `{"$blob": "<id>"}`, where `<id>` matches a blob
frame's `id`. Readers MUST inline blob references back into the
materialised step before exposing it to higher layers, and MUST reject
any `$blob` reference whose id was not previously declared.

The reference writer only interns sub-trees with canonical-JSON size
`>= blob_threshold` (default 200 bytes) that are referenced at least
`blob_min_reuse` (default 2) times; smaller or rarer values are not
interned.

### 6.5 Step body shape

The exact set of fields inside a `step` object is defined by the
recorder (see `stepback/step_types.py::StepKind`). The minimum required
fields a v1-conformant reader MUST recognise are:

| Field | Type | Notes |
| --- | --- | --- |
| `step_id` | string | ULID, monotonic per trace. |
| `step_kind` | string | One of `llm_call`, `tool_call`, `router`, `policy_check`, `mcp_call`, `parallel_branch_open`, `parallel_branch_join`, `exception`. Unknown values MUST be passed through unchanged for forward-compat. |
| `parent_step_id` | string \| null | Edge in the call tree. |
| `inputs` | any | Canonical-JSON encodable. |
| `outputs` | any | Canonical-JSON encodable. |
| `inputs_hash` | string | `sha256:<hex>` over `canonical_json(inputs)`. |
| `outputs_hash` | string | `sha256:<hex>` over `canonical_json(outputs)`. |
| `nondeterminism_hash` | string | `sha256:<hex>` over the recorded non-deterministic inputs. |
| `wallclock_ns` | integer | ns since Unix epoch. |
| `cost_usd` | number | Computed from `price_list_version`. |

LLM-call steps additionally carry `llm_request` and `llm_response` (exact
bytes, model id, sampling params, tool spec). Tool-call steps additionally
carry tool name and arguments. The full per-kind schema with stable field
ids, mandatory/optional flags, and extension ranges is given in
`spec/schema/v1/` (see `spec/schema/README.md`); v1 readers MUST tolerate
unknown fields inside a step body without error.

### 6.6 Tail (`type="tail"`) — REQUIRED, exactly once, last

The last frame in a trace MUST be a tail.

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `type` | string | yes | `"tail"` |
| `wallclock_ns` | integer | yes | Recorder wall-clock at trace close, ns since Unix epoch. |

A reader that reaches EOF without seeing a tail frame MUST treat the
trace as truncated.

### 6.7 Merkle summary (`type="merkle_summary"`) — OPTIONAL, exactly once if present, immediately before tail

When the optional `merkle-summary-v1` capability is in effect, writers
MAY emit a single Merkle summary frame immediately before the `tail`
frame. The summary commits to a single 32-byte SHA-256 root over every
preceding header, capability, blob, and step frame body — the
summary frame itself and the tail frame are NOT leaves.

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `type` | string | yes | `"merkle_summary"` |
| `scheme` | string | yes | Identifier for the leaf-set + tree shape. v1 ships `"frame-body-sha256-rfc6962"`. |
| `algorithm` | string | yes | Hash function. v1 only supports `"sha256"`. |
| `leaf_count` | integer | yes | Number of leaves. Readers MUST reject the trace if this disagrees with the count of preceding non-summary, non-tail frames. |
| `merkle_root` | string | yes | 64-char lower-case hex SHA-256 of the RFC 6962 Merkle root. |

**Tree construction (RFC 6962 §2.1).** Each leaf is
`SHA-256(0x00 || canonical-JSON body bytes)`. Each internal node is
`SHA-256(0x01 || left || right)`. An unpaired node at any level is
*promoted* unchanged (not duplicated). The empty-leaf-list root is
`SHA-256(b"")`. Both leaf and node domain prefixes are mandatory and
provide the standard second-preimage defence: without them, a pair of
node hashes is indistinguishable from a leaf whose body happens to be
their concatenation, so an attacker could substitute a forged subtree
for a real leaf.

**Why both an HMAC chain and a Merkle root.** The HMAC chain pins
frame *order*: every wrapper depends on its predecessor's HMAC, so
deletion, reordering, or insertion is detected at the link adjacent
to the change. The Merkle root pins frame *content* in a single fixed
32-byte commitment with `O(log N)` inclusion-proof depth, which is
the property an attestation pack (§12.4) needs in order to certify a
trace identity to a regulator without shipping the full HMAC walk.

**Reader behaviour.** Readers that recognise `merkle-summary-v1`
MUST recompute the root over the leaves they accumulated and reject
the trace if either `leaf_count` or `merkle_root` disagree. Readers
that do not recognise the frame MUST still HMAC- and signature-verify
it (the wrapper is identical to every other frame) and MAY ignore
its contents, since the frame is non-mandatory.

A summary frame followed by anything other than the `tail` frame MUST
be rejected.

---

## 7. HMAC-chain input

Each frame's `hmac` field is the SHA-256 HMAC of a precise byte sequence
under a 32-byte key shared between writer and verifier (the "HMAC key"
`K_h`). The writer keeps `K_h` private and only publishes
`hmac_key_id = first16hex(SHA256(K_h))` in the header so independent
parties can confirm they hold the same key.

### 7.1 Construction

For frame i (0-indexed):

    M_i = prev_hmac_i_bytes  ||  canonical_json(body_i)
    H_i = HMAC_SHA256(K_h, M_i)

where:

* `prev_hmac_i_bytes` is the 32 raw bytes of the **previous frame's**
  `hmac` (hex-decoded), or 32 zero bytes for the first frame
  (`"00" * 32` in the wrapper).
* `canonical_json(body_i)` is the bytes obtained by §4 over the body
  object exactly as it appears in the wrapper. Compression and blob
  interning happen **before** the HMAC is computed: the bytes hashed are
  the bytes of the (possibly compressed, possibly blob-referenced) body
  the wrapper actually carries on disk.
* `||` denotes byte concatenation. There is **no** length prefix or
  separator inside `M_i`.

The wrapper MUST set `hmac = lowercase_hex(H_i)` and
`prev_hmac = lowercase_hex(prev_hmac_i_bytes)`. Verifiers MUST recompute
`H_i` and reject any mismatch.

### 7.2 Chain semantics

* Inserting a frame anywhere in the file invalidates every subsequent
  `prev_hmac`.
* Deleting a frame likewise breaks the chain at the deletion point.
* Reordering two adjacent frames invalidates both their HMACs.
* Mutating any byte of a body or its receipt fields invalidates that
  frame's HMAC and (because `prev_hmac` propagates) every later frame's
  HMAC as well.
* Tampering with `prev_hmac` alone is detectable because the local
  `hmac` was computed over the genuine prior value.

---

## 8. Signature input

Every frame carries an Ed25519 signature in its `sig` field, computed by
the writer using a single trace-scoped Ed25519 keypair `(K_s, K_p)`. The
public half `K_p` is published in the header as `public_key`.

### 8.1 Construction

    sig_i = Ed25519_Sign(K_s, hmac_i_bytes)

The signed bytes are the **raw 32 bytes** of `hmac_i` (hex-decoded), not
the hex string. The wrapper MUST set
`sig = "ed25519:" + lowercase_hex(sig_i)` (Ed25519 signatures are 64
bytes, so 128 hex chars after the `ed25519:` prefix).

Verifiers MUST:

1. Hex-decode `hmac_i`.
2. Strip the `ed25519:` prefix from `sig` and hex-decode the remainder.
3. Call `Ed25519_Verify(K_p, hmac_i_bytes, sig_bytes)` per RFC 8032.
4. Reject any frame whose verification fails.

### 8.2 Why double-bind

Signing the HMAC instead of the body has two payoffs:

* The signature inherits the chain. A tampered body propagates through
  `H_i` and is detected by either the HMAC check or the signature check
  (whichever the reader tries first). Fuzzers often flip body bytes
  without recomputing the HMAC; both checks then fail independently.
* The signed payload is fixed-size (32 bytes) regardless of body size,
  so signature CPU cost is constant per frame.

---

## 9. Reading and verification algorithm

A conformant verifier MUST implement, in order:

1. Open the file. Maintain `prev_hmac := 0x00 * 32` and a frame counter
   `i := 0`.
2. Read 4 bytes as `length` (big-endian uint32). EOF here is normal **only
   if** the previous frame was a `type="tail"` frame.
3. Read `length` bytes as `wrapper_bytes`. Short read → `UnexpectedEof`.
4. UTF-8 decode and JSON parse `wrapper_bytes`. Reject if the result is
   not an object with **exactly** the keys `body`, `prev_hmac`, `hmac`,
   `sig`.
5. Hex-decode `prev_hmac`, `hmac`, and (after stripping `ed25519:`)
   `sig`. Reject malformed hex or wrong lengths
   (`BadHexOrHmacMismatch` / `SignatureMismatch`).
6. Compare the wrapper's `prev_hmac` bytes to `prev_hmac` from local
   state. Mismatch → `BrokenChainOrHmacMismatch`.
7. Recompute `H = HMAC_SHA256(K_h, prev_hmac_bytes ‖ canonical_json(body))`
   where `canonical_json(body)` re-encodes `body` per §4. Compare to the
   wrapper's `hmac`. Mismatch →
   `BrokenChainOrHmacMismatch`.
   *Implementation note:* a verifier MAY cache the original wrapper
   bytes, locate the body slice, and compare directly without
   re-encoding, but this is only safe if the writer is known to have
   used canonical-JSON for both wrapper and body — which v1 requires.
8. Verify `Ed25519_Verify(K_p, hmac_bytes, sig_bytes)`. Failure →
   `SignatureMismatch`.
9. Dispatch on `body.type`:
   * `i == 0`: MUST be `header`. Validate `magic`, `format_version=1`,
     `canonicalisation_version="1"`, and that the
     `public_key` matches the key being used for signature verification.
     Any unsupported field combination →
     `UnsupportedFormatVersionOrHmacMismatch`.
   * Otherwise: handle `capability`, `step`, `blob`, `tail` per §6.
     Reject mandatory capabilities not in the reader's allow-list.
   * Reject every blob reference (`{"$blob": "..."}`) inside a step body
     that does not match a previously-seen blob frame.
10. Set `prev_hmac := hmac_bytes` and `i += 1`. Loop to step 2.

A reader MUST reject the trace if EOF arrives mid-frame, if the last
frame is not `type="tail"`, or if any check above fails.

---

## 10. Compression and dedup (`gzip+dedup-2`)

When the header declares `compression="gzip+dedup-2"`:

* Each step frame MAY be transmitted in either plain or `gzip+base64`
  form (§6.3). The choice is per-frame and writer-side; readers accept
  either.
* Sub-trees inside step bodies whose canonical-JSON size is at least
  `blob_threshold` and which appear at least `blob_min_reuse` times
  across the *flush window* of pending steps MAY be lifted into blob
  frames (§6.4) and replaced by `{"$blob": "<id>"}` references inside
  the step bodies.
* Compression and blob interning MUST be applied **before** the body is
  canonical-JSON encoded for HMAC. The bytes the HMAC covers and the
  bytes the verifier re-encodes are the same bytes that appear on disk.

This scheme is the reference writer's behaviour. v1 conformance does not
require a writer to implement it — `compression="none"` is fully valid —
but every v1 reader MUST be able to read both modes.

---

## 11. Implementation limits (denial-of-service bounds)

A v1 reader MUST enforce, and document, hard upper bounds on:

* **Frame length.** The reference readers cap `length` at **64 MiB**
  (`MAX_FRAME_BYTES = 64 * 1024 * 1024`). The cap MUST be enforced
  *before* a body buffer of `length` bytes is allocated, so a corrupted
  length prefix can never trigger a multi-gigabyte allocation.
* **JSON nesting depth.** The reference readers cap nesting depth at
  **256** (`MAX_NESTING_DEPTH = 256`). Implementations MUST track depth
  iteratively so the host runtime's stack-overflow behaviour does not
  become the effective cap.
* **Per-string size.** The reference readers cap a single JSON string
  value (or object key) at **16 MiB**
  (`MAX_STRING_BYTES = 16 * 1024 * 1024`). Genuinely larger payloads
  belong in `blob` frames.
* **Frames per trace.** No fixed cap, but readers SHOULD support
  streaming so memory does not grow with frame count.

These bounds are normative. The full DoS contract — including
conformance requirements, the cross-language constant-name table, and
the "tunable upwards but not silently downwards" rule — lives in
`docs/reader-limits.md` (Step 50 of `100_STEPS.md`). A reader that
exposes higher bounds for archival recovery MUST do so behind an
explicit per-call or builder-API opt-in.

---

## 12. Worked hex example

The example below is taken verbatim from
`stepback-core/fixtures/v1/good/header_only.sb`. The fixture is the
shortest possible valid v1 trace: one header frame followed by one tail
frame.

### 12.1 Trace metadata (from `stepback-core/fixtures/v1/manifest.json`)

* `format_version = 1`
* `canonicalisation_version = "1"`
* `price_list_version = "test-2026-01-01"`
* `hmac_key_hex = "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20"` (32 bytes)
* `public_key_hex = "ff57575dc7af8bfc4d0837cc1ce2017b686a88145dc5579a958e3462fe9a908e"`
* Total file size: **1035 bytes**
* SHA-256 of file: `9042bd18f4a784bbb50653869aa33ecc45cce748c500fc5b4b3095cbbac5b800`

### 12.2 Frame 0 — header

**Length prefix.** The first 4 bytes of the file are `00 00 02 9e`
(big-endian) which decodes to **670** — the byte length of the header
wrapper that follows.

**Hex dump of bytes [0, 64):**

```
00000000  00 00 02 9e 7b 22 62 6f 64 79 22 3a 7b 22 62 6c  |....{"body":{"bl|
00000010  6f 62 5f 6d 69 6e 5f 72 65 75 73 65 22 3a 30 2c  |ob_min_reuse":0,|
00000020  22 62 6c 6f 62 5f 74 68 72 65 73 68 6f 6c 64 22  |"blob_threshold"|
00000030  3a 30 2c 22 63 61 6e 6f 6e 69 63 61 6c 69 73 61  |:0,"canonicalisa|
```

**Wrapper bytes [4, 4 + 670):** the canonical-JSON object below. Note
the sorted top-level keys (`body`, `hmac`, `prev_hmac`, `sig`) and the
sorted body keys (`blob_min_reuse`, `blob_threshold`,
`canonicalisation_version`, …, `wallclock_ns`).

```json
{"body":{"blob_min_reuse":0,"blob_threshold":0,"canonicalisation_version":"1","compression":"none","format_version":1,"hmac_key_id":"ae216c2ef5247a37","magic":"stepback/.sb","price_list_version":"test-2026-01-01","public_key":"ff57575dc7af8bfc4d0837cc1ce2017b686a88145dc5579a958e3462fe9a908e","recorder_version":"0.1.0","type":"header","wallclock_ns":1700000000000000000},"hmac":"8f5a4f53b78a598249fed0e3995335644cee53197f696ab4ec305e0884a02321","prev_hmac":"0000000000000000000000000000000000000000000000000000000000000000","sig":"ed25519:30dd8a0ea92b17b287228de16c57299680b1ed995f00fcfd4791ee370de0e496dedbb5bfa633ea297a82625d82ee8d269d1d5f88aec0805a727643ecdece3e0a"}
```

**Receipt fields.**

* `prev_hmac = "00" * 32` (this is frame 0).
* `hmac = 8f5a4f53b78a598249fed0e3995335644cee53197f696ab4ec305e0884a02321`
* `sig = ed25519:30dd8a0ea92b17b287228de16c57299680b1ed995f00fcfd4791ee370de0e496dedbb5bfa633ea297a82625d82ee8d269d1d5f88aec0805a727643ecdece3e0a`

**Reproducing the HMAC.** Let

```
prev_hmac_bytes = 0x00 * 32                              (32 bytes)
body_bytes      = canonical_json(body)                   (363 bytes, the
                                                         substring inside
                                                         "body":<...>)
M_0             = prev_hmac_bytes || body_bytes          (395 bytes)
H_0             = HMAC_SHA256(K_h, M_0)
                = 8f5a4f53b78a598249fed0e3995335644cee53197f696ab4ec305e0884a02321
```

Reproduce in Python:

```python
import hmac, hashlib, json
from stepback.canonical import canonical_json

hmac_key = bytes.fromhex(
    "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20"
)
body = {
    "blob_min_reuse": 0,
    "blob_threshold": 0,
    "canonicalisation_version": "1",
    "compression": "none",
    "format_version": 1,
    "hmac_key_id": "ae216c2ef5247a37",
    "magic": "stepback/.sb",
    "price_list_version": "test-2026-01-01",
    "public_key": "ff57575dc7af8bfc4d0837cc1ce2017b686a88145dc5579a958e3462fe9a908e",
    "recorder_version": "0.1.0",
    "type": "header",
    "wallclock_ns": 1700000000000000000,
}
prev = b"\x00" * 32
h = hmac.new(hmac_key, prev + canonical_json(body), hashlib.sha256).hexdigest()
assert h == "8f5a4f53b78a598249fed0e3995335644cee53197f696ab4ec305e0884a02321"
```

**Reproducing the signature.** Strip the `ed25519:` prefix, hex-decode
the remaining 128 hex characters into 64 signature bytes, and verify
against the 32 raw bytes of `hmac_0` using the public key from §12.1:

```python
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

pk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(
    "ff57575dc7af8bfc4d0837cc1ce2017b686a88145dc5579a958e3462fe9a908e"
))
sig = bytes.fromhex(
    "30dd8a0ea92b17b287228de16c57299680b1ed995f00fcfd4791ee370de0e496"
    "dedbb5bfa633ea297a82625d82ee8d269d1d5f88aec0805a727643ecdece3e0a"
)
hmac_bytes = bytes.fromhex(
    "8f5a4f53b78a598249fed0e3995335644cee53197f696ab4ec305e0884a02321"
)
pk.verify(sig, hmac_bytes)   # raises InvalidSignature on tamper
```

### 12.3 Frame 1 — tail

**Length prefix.** Bytes [674, 678) of the file decode to **357** — the
length of the tail wrapper.

**Wrapper bytes [678, 678 + 357):**

```json
{"body":{"type":"tail","wallclock_ns":1700000000000000001},"hmac":"a01305fddb28480db4a38e6d1b86dc554054801c5f25070918898762a2ad6ca5","prev_hmac":"8f5a4f53b78a598249fed0e3995335644cee53197f696ab4ec305e0884a02321","sig":"ed25519:f4a57730f2d32d0174af90e3aaf72abd4bde02b4f23a70a54fda7e9c7dfb675e06acc639bc2aeb68b085ce725e50f77c24d345d55e353ce05a580c49bd19ad05"}
```

**Receipt fields.**

* `prev_hmac = 8f5a4f53b78a598249fed0e3995335644cee53197f696ab4ec305e0884a02321`
  — exactly equal to frame 0's `hmac`, as the chain rule requires.
* `hmac = a01305fddb28480db4a38e6d1b86dc554054801c5f25070918898762a2ad6ca5`
* `sig  = ed25519:f4a57730f2d32d0174af90e3aaf72abd4bde02b4f23a70a54fda7e9c7dfb675e06acc639bc2aeb68b085ce725e50f77c24d345d55e353ce05a580c49bd19ad05`

**Reproducing the chain link.**

```python
prev = bytes.fromhex(
    "8f5a4f53b78a598249fed0e3995335644cee53197f696ab4ec305e0884a02321"
)
body = {"type": "tail", "wallclock_ns": 1700000000000000001}
h = hmac.new(hmac_key, prev + canonical_json(body), hashlib.sha256).hexdigest()
assert h == "a01305fddb28480db4a38e6d1b86dc554054801c5f25070918898762a2ad6ca5"
```

EOF arrives at byte 1035 = 4 + 670 + 4 + 357 — that is, immediately
after the tail's signature byte. The reader's `length` read at this
position returns 0 bytes; because the previous frame was a tail, this
EOF is accepted.

### 12.4 Negative examples

The same fixture bundle ships five corrupted variants of the multi-step
trace under `stepback-core/fixtures/v1/corrupt/` — each toggles a single
nibble or truncates the file. A conformant reader MUST reject all five:

| Fixture | Corruption | Required rejection class |
| --- | --- | --- |
| `truncated_body.sb` | last frame body chopped in half | `Parse` / `UnexpectedEof` |
| `flipped_hmac.sb` | one nibble of `frame[1].hmac` flipped | `BadHexOrHmacMismatch` |
| `flipped_sig.sb` | one nibble of `frame[1].sig` flipped | `SignatureMismatch` |
| `broken_chain.sb` | `frame[1].prev_hmac` mutated | `BrokenChainOrHmacMismatch` |
| `bad_format_version.sb` | header `format_version` forged to `9` (length-preserving) | `UnsupportedFormatVersionOrHmacMismatch` |

The bytes-flipped cases are length-preserving, so a permissive reader
that re-encodes the body and only checks the HMAC will already reject
them. The `bad_format_version.sb` case additionally exercises the
header-validation path: a reader that ignores `format_version` but still
checks the chain MUST reject because the tampered header bytes invalidate
its HMAC.

### 12.5 Additional good fixtures

Beyond the worked `header_only.sb` example in §12.2, the corpus also
ships:

| Fixture | Purpose |
| --- | --- |
| `good/multi_step.sb` | header + 3 step frames + tail; the source for every corrupt variant in §12.4. |
| `good/with_blobs.sb` | exercises the writer's blob-table path: two consecutive steps share a large payload that is emitted as a separate `blob` frame and referenced by content hash from each step. |
| `good/parallel_branch.sb` | exercises the DAG / fan-out / fan-in path: one `parallel_branch_open`, two sibling `tool_call` branches, one `parallel_branch_join` whose `parent_step_ids` enumerates both branch tails, and a downstream `llm_call` consuming the join's output. Conformant readers MUST reach the tail without rejecting on multi-parent edges. |

All four good fixtures appear in `manifest.json` under the `"good"`
array with their pinned SHA-256s, byte sizes, and minimum frame
counts. New independent readers are expected to verify all of them.

### 12.6 Attestation pack fixtures

`stepback-core/fixtures/v1/` additionally ships a frozen attestation
pack covering the multi-step trace, recorded under
`manifest.json -> attestation_packs`:

| Fixture | Purpose |
| --- | --- |
| `good/attested.pack` | a deterministic, signed attestation pack built by `build_attestation_pack` over `good/multi_step.sb`. The attestor signing key is pinned (see `attestor_public_key_hex` in the manifest), `produced_at` is pinned to a constant timestamp, and the body is signed with that pinned key — so the pack file is bit-stable across runs and machines. Independent attestation verifiers MUST accept it. |
| `corrupt/attested_tampered.pack` | identical bytes to `good/attested.pack` except for a single hex-character flip in the body-level Ed25519 signature. Independent verifiers MUST reject (`AttestationVerificationError`). |

The pack format itself is documented in `stepback/attestation.py`; this
spec covers only the trace wire format.

---

## 13. Versioning and capability negotiation

* The wire format is versioned by the header's `format_version` integer.
  v1 is `1`. A future `format_version=2` MAY introduce a CBOR encoding,
  a Merkle summary frame, or new frame kinds. v1 readers MUST reject
  unknown versions.
* Within v1, additive evolution happens through capability frames (§6.2).
  Mandatory capabilities are the *only* way to make a v1 trace require a
  feature beyond the core. Optional capabilities permit graceful
  degradation.
* Canonicalisation is versioned independently as
  `canonicalisation_version`. A v1 trace MUST set it to `"1"`. Future
  canonicalisation versions (e.g. introducing decimals, or a deterministic
  CBOR encoding inside JSON strings) require a header bump and are not
  permitted in v1.
* The wire-format SemVer policy is documented separately in
  [`docs/api-compat.md`](../docs/api-compat.md).

---

## 14. Security considerations

* **HMAC key custody.** `K_h` is shared between writer and verifier and
  MUST be treated as a secret. Anyone with `K_h` can forge a syntactically
  valid trace whose HMACs verify; the Ed25519 layer is what binds a
  trace to a known signer.
* **Signing key custody.** `K_s` SHOULD be ephemeral per trace (the
  reference writer generates a fresh keypair when none is supplied).
  Long-lived signing keys SHOULD be held in an HSM or a threshold
  signer (Step 56 of `100_STEPS.md`).
* **Replay vs. forgery.** A trace cannot be silently rewritten because
  every frame's HMAC chains to the prior `hmac` and is signed
  individually. An attacker with `K_h` but not `K_s` can re-compute
  HMACs but cannot forge signatures, and vice versa.
* **Truncation attacks.** Because the tail frame is mandatory and
  HMAC-chained, an attacker truncating the file removes evidence of
  later steps but produces a trace any conforming reader rejects (no
  tail). Combined with attestation packs (§ outside this spec) this
  yields cryptographic proof of a complete record.
* **Confidentiality.** v1 does not encrypt step contents. Recorders that
  capture sensitive payloads SHOULD apply `stepback.redact` before
  writing or rely on transport-level protections.
* **Reader DoS.** Without the §11 limits a malicious writer can ship a
  single frame whose `length` claims gigabytes of payload. Implement the
  bounds before exposing a verifier to untrusted input.

---

## 15. Conformance

A v1 implementation is **read-conformant** iff:

1. It accepts every fixture under `stepback-core/fixtures/v1/good/` and
   reaches the tail without error.
2. It rejects every fixture under `stepback-core/fixtures/v1/corrupt/`
   with an error in the class given by `manifest.json`.
3. It enforces the §11 limits (the exact bounds may differ from the
   reference reader but MUST be documented).

A v1 implementation is **write-conformant** iff:

1. The traces it writes are read-conformantly accepted by the reference
   reader (`stepback.trace_reader.verify_trace`).
2. Its canonical-JSON output for the bounded domain in §4 is byte-equal
   to the reference encoder's output.
3. Its HMAC and signature constructions match §7 and §8 — verifiable by
   re-running the worked example in §12 with its emitted bytes.

The full conformance harness is `stepback spec test <implementation>`
(Step 47 of `100_STEPS.md`), backed by the fixture manifest and SHA-256
catalogue in `stepback-core/fixtures/v1/manifest.json`.

---

## 16. References

* RFC 8259 — JSON Data Interchange Format.
* RFC 8174 — Ambiguity of Uppercase vs. Lowercase in RFC 2119 Key Words.
* RFC 8032 — Edwards-Curve Digital Signature Algorithm (EdDSA), §5.1
  (Ed25519).
* FIPS 198-1 — The Keyed-Hash Message Authentication Code (HMAC).
* FIPS 180-4 — Secure Hash Standard (SHS), §6.2 (SHA-256).
* `stepback/canonical.py` — reference canonical-JSON encoder.
* `stepback/trace_writer.py` — reference writer.
* `stepback/trace_reader.py` — reference reader / verifier.
* `stepback-core/fixtures/v1/manifest.json` — fixture catalogue.
* `100_STEPS.md` — surrounding standardisation roadmap; this document is
  Step 40.
