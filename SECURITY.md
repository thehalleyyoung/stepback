# Security Policy

## Reporting vulnerabilities

**Please do not file public GitHub issues for suspected security vulnerabilities.**

To report a potential vulnerability, use one of the following paths:

1. **GitHub Security Advisories** — open a private advisory at
   `https://github.com/stepback/stepback/security/advisories/new` (if enabled
   for this repository). This is the preferred path; GitHub notifies the
   maintainers without making the report public.
2. **Email** — send a description to the repository contact listed in
   `CITATION.cff`. Include "SECURITY" in the subject line. Use GPG encryption
   if the report contains sensitive material.

A maintainer will acknowledge receipt within five business days and provide
a resolution timeline. If no response is received within ten business days,
re-send using the alternative contact path.

Stepback follows a **90-day coordinated disclosure window** by default. After
90 days (or sooner if a patch is available) a public advisory will be filed
regardless of patch status.

---

## Threat model

### What the HMAC-SHA256 + Ed25519 scheme proves

Each frame in a `.sb` trace is wrapped as:

```json
{
  "body":      <frame body>,
  "prev_hmac": "<hex>",
  "hmac":      "<hex>",
  "sig":       "ed25519:<hex>"
}
```

where:

- `hmac` = `HMAC-SHA256(hmac_key, prev_hmac_bytes ‖ canonical_json(body))`
- `sig`  = Ed25519 signature over the `hmac` bytes

The scheme provides the following guarantees **when verification passes**:

1. **Per-frame integrity.** The `body` of each frame has not been altered
   after the writer computed its HMAC and signature.

2. **Chain integrity.** Each frame's `prev_hmac` correctly references the
   previous frame. An attacker cannot reorder, insert, or delete individual
   frames without breaking the chain — every subsequent HMAC would fail to
   verify.

3. **Truncation detection (tail-dependent).** If the verifier requires a
   valid `tail` frame (the default), a trace truncated before the tail is
   detected. A trace that ends at a `tail` frame is intact up to that point.
   A verifier that does not require a tail cannot detect truncation.

4. **Capability fail-closed.** A trace declaring a mandatory capability the
   verifier does not understand is rejected with `TraceVerificationError`.
   Unknown optional capabilities are ignored.

5. **Merkle summary consistency.** When an optional `merkle_summary` frame is
   present, the verifier recomputes the Merkle root from all leaf frames and
   rejects the trace if the recorded root does not match.

### What the scheme does NOT prove

| Property | Not guaranteed |
|----------|---------------|
| Content correctness | The recorded tool outputs, LLM responses, costs, and timestamps are what the code says they are. The scheme proves they have not been altered after recording, not that they were recorded honestly. |
| Recorder identity | The Ed25519 public key embedded in the trace header is self-asserted. It is NOT an identity proof unless the verifier compares the embedded key against a trusted key obtained out-of-band (see "Public key pinning" below). |
| Absence of a lying recorder | A compromised recorder can write any `body` it likes and sign it correctly. The scheme provides tamper-evidence for the record; it does not audit the recorder's behavior. |
| Clock accuracy | `wallclock_ns` values come from the recorder's clock. The scheme does not bind them to an external time source. |
| Price accuracy | `cost_usd` values come from a caller-supplied price list. They are recorded as-is and are not independently verified. |
| Absence of PII or secrets | The scheme does not redact, classify, or sanitise frame bodies. |
| Absence of replayed traces | The scheme does not prevent a valid trace from being replayed in a different context. |
| Semantic correctness of replay | `verify_trace` passing does not mean that replaying the trace with a different executor will produce semantically equivalent results. |
| Attestation pack provenance | An attestation pack signature proves the attestor signed a particular verification result; it does not prove that the underlying traces are correct or that the attestor environment was uncompromised. |

---

## Key types and handling

### HMAC key (`hmac_key`)

- **What it is:** A 32-byte secret used to compute the per-frame HMAC chain.
- **Where it is stored:** The `.sb` file contains only the first 16 hex
  characters of `SHA-256(hmac_key)` as the `hmac_key_id` field in the header.
  The key itself is **never written to disk by the reference recorder**.
- **Who needs it:** Any party that calls `verify_trace` must supply the HMAC
  key out-of-band.
- **Security classification:** Treat as a **secret**. An attacker who obtains
  the HMAC key — combined with the absence of external public-key pinning —
  can rewrite a trace, replace the header public key with their own, recompute
  all HMACs, and produce a trace that passes verification. See the next
  section.
- **Rotation:** See [Step 128 of the roadmap](100_STEPS.md) for planned key
  rotation support.

### Ed25519 signing key (`signing_key` / `public_key`)

- **What it is:** An Ed25519 key pair. The private key signs each frame's
  HMAC; the public key is embedded in the trace header as
  `"ed25519:<hex>"`.
- **Where it is stored:** The **private** signing key must never be written
  to a `.sb` file. The **public** key is written to the `public_key` header
  field.
- **Security classification:** Treat the private key as a **secret**. The
  public key is not secret.
- **Rotation:** Key rotation is planned (Step 128). Until then, a compromised
  signing key invalidates all traces that relied on identity proof via that key.

### Public key pinning

The current `verify_trace` API does not accept an expected public key as a
parameter. The verifier reads the public key from the trace header and uses it
to check the per-frame signatures. This means:

> **`verify_trace` passing proves only self-consistency**: the signatures were
> made by the private key corresponding to the embedded public key. It does
> NOT prove the trace was written by a specific, trusted recorder unless the
> caller independently compares `trace.public_key_hex` against a known-good
> value.

**Recommendation for audit pipelines:** After calling `verify_trace`, compare
the returned `Trace_.public_key_hex` against the expected key for the recorder
or data source. Reject traces whose public key does not match.

---

## Verifier guarantees

`verify_trace(path, hmac_key)` returns a `Trace_` object only if **all** of
the following hold:

1. The file at `path` can be read and parsed as a sequence of
   length-prefixed canonical-JSON frames without error.
2. Every frame wrapper is structurally valid: `body`, `prev_hmac`, `hmac`,
   and `sig` fields are present and have the expected types.
3. Every `hmac` field equals
   `HMAC-SHA256(hmac_key, prev_hmac_bytes ‖ canonical_json(body))`.
4. Every `prev_hmac` of frame `n+1` equals the `hmac` of frame `n` (chain
   integrity). The first frame's `prev_hmac` is the all-zeros sentinel.
5. Every `sig` field is a valid Ed25519 signature over the `hmac` bytes,
   using the public key from the trace header.
6. If a `merkle_summary` frame is present, the recomputed Merkle root
   matches the recorded root.
7. The capability allow-list check passes: no mandatory capability in the
   trace is unknown to the verifier.

If any check fails, `TraceVerificationError` is raised and no `Trace_` object
is returned.

`verify_trace` does **not** perform:

- Schema validation of frame body fields (see `SBTraceSpec.validate_trace`
  for schema checks).
- Semantic validation of step kinds, cost fields, or timestamp ordering.
- Proof that the embedded public key belongs to any specific identity.

---

## Denial-of-service bounds

The reader enforces hard limits before processing frame contents:

| Limit | Default value | Notes |
|-------|--------------|-------|
| `MAX_FRAME_BYTES` | 64 MiB | Length-prefix rejection |
| `MAX_NESTING_DEPTH` | 256 | JSON object/array depth |
| `MAX_STRING_BYTES` | 16 MiB | Per string or key |

See `docs/reader-limits.md` for the full table and override instructions.

---

## Attestation packs

An attestation pack bundles the results of `verify_trace` and optional replay
for a corpus of traces. The pack is content-addressed and Ed25519-signed by a
separate **attestor key**.

Trust boundaries:

- A valid pack signature proves the **attestor** signed **this verification
  and replay result** at the time of signing.
- It does NOT independently prove the underlying traces are correct; the
  auditor must trust the attestor and the verifier environment.
- The attestor key is separate from any per-trace signing key. Compromise of
  a trace signing key does not automatically compromise the attestor key, and
  vice-versa.

See `stepback/attestation.py` for the pack format and `stepback verify
--attestation-out` for production use.

---

## Supported versions

| Version | Supported |
|---------|-----------|
| Latest (`main`) | ✅ |
| Tagged releases | ✅ patch releases for the most recent minor |
| Older minors | ❌ (upgrade recommended) |

---

## Out-of-scope items

The following are **not** in scope for this security policy:

- Vulnerabilities in optional dependencies (report upstream).
- Timing-based attacks on the pure-Python crypto path. The reference
  recorder uses `cryptography` (libsodium-backed) for Ed25519 and `hmac`
  (constant-time comparison via `hmac.compare_digest`) for HMAC verification.
- Attacks requiring physical access to the machine running the recorder.
- Social-engineering attacks against maintainers.
