# RFC 0004 — Attestation Packs

| Field | Value |
|---|---|
| RFC number | 0004 |
| Title | Attestation Packs |
| Status | Draft |
| Supersedes | — |
| Created | 2026-05-12 |
| Authors | stepback maintainers |
| Reference impl | `stepback.attestation` (`stepback >= 0.1`) |

---

## Abstract

This RFC specifies the *attestation pack* — a signed, content-addressed
bundle that a regulated organisation hands to an auditor as evidence that
a corpus of AI-agent traces was properly recorded and (optionally) replayed
under a new policy version.  The pack is entirely self-verifying: an
auditor needs only the attestor's Ed25519 public key; they do not need the
original `.sb` files, the Python source, or network access.

The reference implementation is `stepback.attestation` in
[`stepback/attestation.py`](../../stepback/attestation.py).

---

## 1. Motivation

Financial-services, healthcare, and government regulators increasingly
require evidence that AI decisions can be audited and counterfactually
re-examined.  An attestation pack addresses two questions:

1. **Integrity:** "Were these traces recorded faithfully and not tampered
   with after the fact?"
2. **Counterfactual compliance:** "Had the policy in effect at time T₀
   been replaced by the policy now in force, how many decisions would have
   changed, and by how much?"

The pack bundles per-trace verdicts into a single signed artifact so that
the auditor performs one signature check, not one per trace.

---

## 2. On-disk format

The pack is a canonical-JSON (RFC 0002) file with `.pack` extension:

```json
{
  "magic": "stepback/.pack",
  "format_version": 1,
  "produced_at": "<ISO-8601 UTC>",
  "attestor_public_key": "ed25519:<hex>",
  "policy_version_pin": "<string | null>",
  "summary": {
    "trace_count": 12418,
    "verified_ok": 12418,
    "verified_fail": 0,
    "replayed_ok": 12418,
    "divergent_traces": 41,
    "total_cost_delta_usd": 12.47
  },
  "entries": [ <AttestationEntry>, ... ],
  "body_hash": "sha256:<hex>",
  "signature": "ed25519:<hex>"
}
```

`body_hash` is `sha256(canonical_json(pack_without_signature_field))`.
`signature` is an Ed25519 signature over `canonical_json(body_hash_field)`.

### 2.1 AttestationEntry

```json
{
  "trace_id": "<uuid4>",
  "trace_chain_hash": "sha256:<hex>",
  "recorder_key_fingerprint": "sha256:<hex>",
  "verify_ok": true,
  "verify_error": null,
  "replay_ok": true,
  "replay_error": null,
  "dirty_step_count": 3,
  "divergent_step_count": 2,
  "cost_delta_usd": 0.0012,
  "replayed_at": "<ISO-8601 UTC>"
}
```

`trace_chain_hash` is the value from the `.sb` `tail` frame.
`recorder_key_fingerprint` is `sha256` of the recorder's Ed25519 public key bytes.
`verify_ok` / `replay_ok` are booleans; the corresponding `_error` field
is `null` on success or a short error string on failure.

---

## 3. Merkle summary frame

When a `.sb` trace is included in an attestation pack, a `merkle_summary`
frame is appended immediately before the `tail` frame:

```json
{
  "frame_kind": "merkle_summary",
  "merkle_root": "blake2b:<hex>",
  "step_count": 42,
  "leaf_hashes": ["blake2b:<hex>", ...]
}
```

`leaf_hashes` are the per-frame HMAC values from the HMAC chain
(RFC 0001 §5).  `merkle_root` is the root of the binary Merkle tree over
`leaf_hashes` using BLAKE2b-256 as the combination function:

```
merkle(a, b) = blake2b(a || b)
merkle([x]) = x
merkle([]) = blake2b(b"")
```

The Merkle root allows an attestor to include a short proof-of-inclusion
for a single step without revealing the entire trace.

---

## 4. Verification procedure

An auditor verifies an attestation pack as follows:

1. Decode the canonical JSON and check `"magic": "stepback/.pack"` and
   `"format_version": 1`.
2. Compute `sha256(canonical_json(pack_without_signature))` and compare
   against `body_hash`.
3. Verify the Ed25519 `signature` over `canonical_json({"body_hash": ...})`
   using the attestor public key provided out-of-band.
4. Iterate over `entries`; any entry with `verify_ok: false` or
   `replay_ok: false` is a finding.

Steps 2–3 require no `.sb` files; the pack is self-contained.

---

## 5. Key rotation

When the attestor's Ed25519 key is rotated, old packs signed under the
previous key remain valid.  The rotation is documented in a
`key_rotation_event` appended to the pack's `entries` list (separate from
trace entries) with `"entry_kind": "key_rotation"`.  Auditors MUST check
that the new key's cross-signature by the old key is valid before trusting
verdicts signed by the new key.

---

## 6. Post-quantum co-signature (experimental)

When built with the `pqsig` extra, the pack may include an additional
`pq_signature` field produced by ML-DSA-65 (CRYSTALS-Dilithium, FIPS 204).
This is an experimental field; verifiers that do not understand it MUST
ignore it.  It is not part of the v1 normative spec.

---

## 7. Security considerations

- The attestor private key MUST be held in hardware (HSM, YubiKey, cloud
  KMS) for production use.  Software-only key storage is acceptable for
  development.
- The pack does not embed the original traces.  An auditor who wishes to
  independently verify step outputs must separately obtain the `.sb` files.
- `policy_version_pin` is advisory and not cryptographically enforced by
  this format; the replay engine enforces it at replay time.

---

## 8. Relationship to other RFCs

- RFC 0001 defines the `tail.trace_chain_hash` field referenced by entries.
- RFC 0002 defines canonical JSON, used for `body_hash` computation.

---

## Appendix A — Changelog

| Date | Author | Change |
|---|---|---|
| 2026-05-12 | stepback maintainers | Initial draft |
