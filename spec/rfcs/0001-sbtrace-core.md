# RFC 0001 — SB-Trace Core Wire Format

| Field | Value |
|---|---|
| RFC number | 0001 |
| Title | SB-Trace Core Wire Format |
| Status | Draft |
| Supersedes | — |
| Superseded by | — |
| Created | 2026-05-12 |
| Authors | stepback maintainers |
| Reference impl | `stepback >= 0.1`, `stepback-core` (Rust) |

---

## Abstract

This RFC defines the *SB-Trace* (`.sb`) v1 binary wire format — the file
layout, frame model, HMAC-chain construction, and Ed25519 signature scheme
used to record, replay, and attest AI-agent execution traces.  It is the
foundation document that RFCs [0002](0002-canonicalization.md),
[0003](0003-dirty-set.md), [0004](0004-attestation-packs.md),
[0005](0005-importer-lossiness.md), and
[0006](0006-otel-agent-step.md) build upon.

The detailed byte-level specification is in
[`spec/sbtrace-v1.md`](../sbtrace-v1.md) and its companion
[`spec/sbtrace-v2.md`](../sbtrace-v2.md).  This RFC document summarises
the rationale, design constraints, and extension points in a form suitable
for external review and upstream standards submission.

---

## 1. Motivation

AI-agent workflows interleave LLM calls, tool invocations, routing
decisions, and policy checks.  Debugging, auditing, and cost-optimising
these workflows requires:

1. **Faithful recording** — every input, output, cost, latency, and
   nondeterminism source captured at runtime without modifying agent
   behaviour.
2. **Counterfactual replay** — "what would have happened if I changed step
   k?" answered cheaply by re-executing only the affected suffix
   (dirty-set replay, RFC 0003).
3. **Tamper-evident audit trails** — an HMAC chain over frame bodies plus
   an Ed25519 end-signature let regulators verify a production trace was
   not modified after recording.
4. **Multi-language interop** — Python, Rust, TypeScript, Go, JVM, and
   .NET all need to read and write the same bytes.
5. **Format evolution** — v1 traces must remain readable by v2+ readers;
   an explicit version field and capability negotiation enable this.

No existing format (OpenTelemetry, LangSmith, OpenInference) satisfies all
five requirements simultaneously.  SB-Trace is purpose-built for them.

---

## 2. File structure

An `.sb` file is a *length-prefixed frame stream*:

```
[4-byte magic "SBTv"] [frame]* [EOF]
```

Each frame is:

```
[4-byte big-endian body length] [body bytes] [32-byte BLAKE2b-256 HMAC]
```

Frames appear in this mandatory order:

| Position | Frame kind | Required? |
|---|---|---|
| First | `header` | Yes |
| 2…N-1 | `step`, `blob`, `capability` | Any order; ≥0 |
| Last | `tail` | Yes |

The `merkle_summary` frame (RFC 0004) is inserted immediately before
the `tail` when producing attestation packs.

Frame bodies are **canonical JSON** (RFC 0002).

---

## 3. Format versioning

The `header` frame carries:

```json
{
  "format_version": 1,
  "canonicalisation_version": 1,
  "dirty_set_version": 1,
  "trace_id": "<uuid4>"
}
```

Readers MUST reject any `format_version` they do not understand.
Readers MUST reject any `canonicalisation_version` they do not understand,
because the HMAC computation depends on it.
`dirty_set_version` is advisory for replay engines; readers MAY ignore it.

Version increments follow SemVer semantics: backward-incompatible changes
bump the major version (e.g. `format_version = 2`).

---

## 4. Step frame

The `step` frame is the central unit of a trace.  Its body is:

```json
{
  "frame_kind": "step",
  "step_id": "<uuid4>",
  "step_kind": "llm_call | tool_call | router | policy_check | mcp_call | parallel_branch_open | parallel_branch_join | exception",
  "parent_step_id": "<uuid4> | null",
  "parent_step_ids": ["<uuid4>", ...],
  "inputs": { ... },
  "outputs": { ... },
  "inputs_hash": "blake2b:<hex>",
  "nondeterminism": { ... },
  "nondeterminism_hash": "blake2b:<hex>",
  "wallclock_ns": 12345,
  "cpu_ns": 9000,
  "cost_usd": 0.00042,
  "receipt": { ... }
}
```

`parent_step_ids` is used for `parallel_branch_join` steps; all other
kinds use `parent_step_id` (singular).  Both fields are always present in
the serialised form; unused fields are `null` / `[]`.

`inputs_hash` and `nondeterminism_hash` are **canonical-JSON BLAKE2b-256**
hashes of `inputs` and `nondeterminism` respectively.  They are computed
by the recorder at record time and re-computed by the replay engine.  A
mismatch triggers dirty-set propagation (RFC 0003).

---

## 5. HMAC chain

```
hmac[0] = HMAC-SHA256(key, canonical_json(header_body))
hmac[i] = HMAC-SHA256(key, canonical_json(frame_body[i]) || hmac[i-1])
```

The key is an HMAC secret held by the recorder.  Verifiers call
`verify_trace(path, hmac_key=...)`.  The chain makes it impossible to
insert, delete, or reorder frames without breaking the chain.

---

## 6. Ed25519 end-signature

The `tail` frame contains:

```json
{
  "frame_kind": "tail",
  "trace_chain_hash": "sha256:<hex>",
  "step_count": 42,
  "signature": "ed25519:<hex>"
}
```

`trace_chain_hash` is `sha256(hmac[0] || hmac[1] || ... || hmac[N-1])`.
`signature` is an Ed25519 signature over `canonical_json(tail_body_without_signature)`.

Verifiers who do not hold the HMAC key can still verify the Ed25519
signature if they trust the recorder's public key.

---

## 7. Capability negotiation

`capability` frames may appear anywhere in the body stream:

```json
{
  "frame_kind": "capability",
  "capability": "merkle_summary | cbor_encoding | ...",
  "mandatory": true
}
```

Readers that do not understand a *mandatory* capability MUST fail-closed
rather than silently misinterpreting the trace.  Optional capabilities
(`"mandatory": false`) may be ignored.

---

## 8. Limits

To prevent resource exhaustion:

| Parameter | Limit |
|---|---|
| Max frame body | 64 MiB |
| Max string length | 1 MiB |
| Max nesting depth | 32 |
| Max `step_count` | 1,000,000 |

Implementations MUST enforce these limits and reject frames that exceed
them with a `FrameSizeError` or equivalent.

---

## 9. Relationship to other RFCs

| RFC | Relationship |
|---|---|
| [0002](0002-canonicalization.md) | Defines the canonical-JSON encoder that frame bodies MUST use |
| [0003](0003-dirty-set.md) | Defines how `inputs_hash` fields drive replay decisions |
| [0004](0004-attestation-packs.md) | Defines the `merkle_summary` frame and the `.pack` bundle |
| [0005](0005-importer-lossiness.md) | Defines how foreign-format importers report field lossiness |
| [0006](0006-otel-agent-step.md) | Maps SB-Trace step kinds to OTel `agent.step.*` attributes |

---

## 10. Security considerations

- The HMAC secret MUST be at least 256 bits (32 random bytes).
- Ed25519 private keys MUST be generated fresh per-recorder and MUST NOT
  be shared across production tenants.
- HMAC and signature algorithms are fixed at v1; negotiating weaker
  algorithms is not permitted.
- Post-quantum experiments are tracked separately (see `stepback.pqsig`)
  and are not part of the v1 normative specification.

---

## 11. Test vectors

The frozen v1 fixtures in `spec/` serve as normative test vectors:

```
spec/fixtures/v1/minimal_trace.sb      — single llm_call step
spec/fixtures/v1/multi_step_trace.sb   — 12-step agent run
spec/fixtures/v1/parallel_trace.sb     — parallel branches
spec/fixtures/v1/corrupt_tail.sb       — tampered tail (MUST fail verify)
```

Implementations MUST pass all conformance tests in `stepback spec test`.

---

## Appendix A — Changelog

| Date | Author | Change |
|---|---|---|
| 2026-05-12 | stepback maintainers | Initial draft |
