# Stepback — TLA+ specification of the SB-Trace HMAC chain

This directory contains the TLA+ formal model for the `.sb` HMAC chain
integrity protocol, discharging part of
[Step 145](../../100_STEPS.md) of the OSS-readiness plan.

## Files

| File | Purpose |
|---|---|
| `SBHMACChain.tla` | TLA+ specification — state machine, actions, invariants |
| `SBHMACChain.cfg` | TLC model-checker configuration (finite instantiation) |
| `README.md` | This document |

## What is specified

The `.sb` format appends frames to an append-only log.  Each frame carries:

```
body      -- canonical JSON content (abstract in the model)
prev_hmac -- chaining HMAC from the preceding frame (ZERO_HMAC for frame 1)
hmac      -- HMAC_SHA256(key, prev_hmac ‖ canonical_json(body))
```

The spec models:

- **Writer** — appends authentic frames, maintaining the chain by
  construction.
- **Adversary** — may replace the *body* of any frame on disk but
  **cannot** recompute a valid HMAC without the secret key (ciphertext-only
  threat model; see `A_mac` / `A_coll` in the spec header).

## Properties proved (invariants)

| ID | Name | Statement |
|---|---|---|
| P1 | `IntegrityInvariant` | An untampered on-disk log always validates. |
| P2 | `TamperEvidenceInvariant` | Any body-substitution by the adversary causes log validation to fail. |
| P3 | `WriterCoherenceInvariant` | The writer's own log is always self-consistent. |

## Relationship to the implementation

| TLA+ symbol | Python counterpart | File |
|---|---|---|
| `ZeroHMAC` | `ZERO_HMAC = b"\x00" * 32` | `stepback/trace_writer.py` |
| `AbstractHMAC` | `hmac.new(key, prev ‖ body, sha256).hexdigest()` | `stepback/trace_writer.py` |
| `AppendFrame` | `TraceWriter._write_frame` | `stepback/trace_writer.py` |
| `LogValid` | `TraceReader._verify_chain` | `stepback/reader.py` |
| `TamperBodyOnly` | adversary scenario in `test_attestation.py` | `tests/test_attestation.py` |

## Scope and limitations

1. **No truncation detection.** Suffix truncation leaves the prefix chain
   valid; detection requires the externally-anchored Merkle summary frame
   (`spec/sbtrace-v1.md §4.5`).  That anchor is outside this spec's scope.

2. **No canonicalization proof.** Bodies are modelled as abstract atoms.
   Canonicalization correctness is proved separately (RFC 0002,
   `docs/canonicalization.md`).

3. **Single-key model.** The TLC configuration uses one abstract HMAC key.
   Multi-key rotation semantics (key-id field in the header) are out of scope.

4. **Ed25519 not modelled.** Per-frame Ed25519 signatures are treated as a
   secondary endorsement layer.  The HMAC chain is the primary integrity
   anchor.

## Running TLC

Install the TLA+ Toolbox or the standalone `tlc` command-line checker.

```bash
# From the proofs/tla/ directory:
tlc SBHMACChain -config SBHMACChain.cfg
```

Expected output (all invariants hold):

```
Model checking completed. No error has been found.
  Estimates of the probability that TLC did not check all reachable states
  because two distinct states had the same fingerprint:
  calculated (optimistic):  val = 1.2E-15
```

To increase coverage, edit `SBHMACChain.cfg`:
- `MaxFrames = 5` covers five-frame chains (tractable; ~1.5 M states).
- `Bodies = {b1, b2, b3, b4}` adds a fourth body value.

## Relationship to other proofs

- **Lean 4 soundness** (Step 56): `proofs/lean/` mechanises dirty-set
  soundness, not wire-format integrity.  These two proofs are
  complementary.
- **Paper proof** (Step 55): `docs/dirty-set-soundness.md`.
- **RFC 0001**: `spec/rfcs/0001-sbtrace-core.md` — rationale for the chain
  design; this TLA+ spec is the formal companion to §3.3 "HMAC chain".

## CI

Add `.github/workflows/tla.yml` to run TLC on changes to `proofs/tla/**`:

```yaml
name: TLA+ model check
on:
  push:
    paths: ["proofs/tla/**"]
jobs:
  tlc:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install TLC
        run: |
          wget -q https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar
      - name: Run TLC
        run: |
          java -jar tla2tools.jar -config SBHMACChain.cfg SBHMACChain.tla
        working-directory: proofs/tla
```
