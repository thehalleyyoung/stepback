# SB-Trace Conformance Dashboard

**Last updated:** 2026-05-12  
**Format version tracked:** SB-Trace v1 (`format_version: 1`, `canonicalisation_version: 1`)

This document tracks conformance with the SB-Trace v1 wire format
([`spec/sbtrace-v1.md`](../spec/sbtrace-v1.md)) and the supporting RFCs
across every stepback implementation.

> **Draft status.** This dashboard reflects the current state of the reference
> implementation and bindings.  Multi-implementation production deployment is a
> prerequisite for formal conformance certification.  Until then, all statuses
> other than Python should be treated as best-effort self-assessment.

---

## Status legend

| Symbol | Meaning |
|---|---|
| ✅ | Implemented, tested, evidence linked |
| 🟡 | Implemented; conformance test coverage incomplete |
| ⬜ | Not yet implemented |
| ❓ | Unknown / untested |
| 🚫 | Not applicable |

---

## Implementations

| Implementation | Package / module | Profile |
|---|---|---|
| **Python** | `stepback` (PyPI) — reference implementation | reader + writer + verifier |
| **Rust** | `stepback-core` workspace (`sb-format`, `sb-verify`) | reader + verifier |
| **TypeScript** | `@stepback/core` (`bindings/typescript/`) | reader + verifier |
| **Go** | `github.com/stepback/stepback-go` (`bindings/go/`) | reader + verifier |
| **JVM** | `io.stepback:stepback-jvm` (`bindings/jvm/`) | reader + verifier |
| **.NET** | `Stepback.Sb` (`bindings/dotnet/`) | reader + verifier |
| **WASM** | `@stepback/wasm` (`wasm/`) | verifier (read-only) |
| **Proxy** | `stepback-proxy` (`sb proxy`) | writer (HTTP/gRPC relay) |

---

## §3 — Frame layout (4-byte length prefix + wrapper JSON)

| Feature | Python | Rust | TypeScript | Go | JVM | .NET | WASM | Proxy |
|---|---|---|---|---|---|---|---|---|
| Read 4-byte BE length prefix | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🚫 |
| Write 4-byte BE length prefix | ✅ | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | ✅ |
| Reject partial final frame | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🚫 |
| Reject oversized frame (> 64 MiB) | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🚫 |

**Evidence:** `tests/test_reader_corruption.py`, `tests/test_reader_fuzz.py`;
Rust: `stepback-core/crates/sb-format/tests/`; cross-language:
`tests/test_python_rust_differential.py`

---

## §4 — Canonical JSON

| Feature | Python | Rust | TypeScript | Go | JVM | .NET | WASM | Proxy |
|---|---|---|---|---|---|---|---|---|
| Sorted keys | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🚫 |
| No extra whitespace | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🚫 |
| `\uXXXX` escape normalisation | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🚫 |
| NFC Unicode normalisation | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🚫 |
| Surrogate rejection | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🚫 |
| Frozen conformance fixtures (100 vectors) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🚫 |

**Evidence:** `tests/test_canonical_unicode.py`, `tests/test_canonical_hypothesis.py`,
`tests/test_canonical_differential.py`; fixture corpus:
`stepback-core/fixtures/v1/canonical/`

---

## §5 — HMAC chain (`prev_hmac` / `hmac`)

| Feature | Python | Rust | TypeScript | Go | JVM | .NET | WASM | Proxy |
|---|---|---|---|---|---|---|---|---|
| Write chain with ZERO_HMAC seed | ✅ | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | ✅ |
| Verify chain (sequential) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🚫 |
| Reject chain break at frame N | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🚫 |
| Verify Python-written traces | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🚫 |
| TLA+ formal model (`P1`, `P2`, `P3`) | ✅ | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 |

**Evidence:** `tests/test_attestation.py`, `tests/test_compression.py`;
Rust: `stepback-core/crates/sb-verify/`; cross-language:
`tests/test_python_rust_differential.py`; formal: `proofs/tla/SBHMACChain.tla`

---

## §6 — Frame kinds

### §6.1 Header frame

| Feature | Python | Rust | TypeScript | Go | JVM | .NET | WASM | Proxy |
|---|---|---|---|---|---|---|---|---|
| Write header with all required fields | ✅ | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | ✅ |
| Parse header; validate `format_version` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🚫 |
| Reject unknown `format_version` | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🚫 |
| Parse `public_key` / `hmac_key_id` | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🚫 |

### §6.2 Capability frame

| Feature | Python | Rust | TypeScript | Go | JVM | .NET | WASM | Proxy |
|---|---|---|---|---|---|---|---|---|
| Write mandatory capability frame | ✅ | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 |
| Reject trace with unknown mandatory cap | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🚫 |
| Accept trace with unknown optional cap | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🚫 |

**Evidence:** `tests/test_capability_negotiation.py`

### §6.3 Step frame

| Feature | Python | Rust | TypeScript | Go | JVM | .NET | WASM | Proxy |
|---|---|---|---|---|---|---|---|---|
| Write / read all 8 `step_kind` values | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ✅ |
| `inputs_hash` / `outputs_hash` (BLAKE2b) | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🚫 |
| `nondeterminism_hash` | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🚫 |
| `cost_usd`, `wallclock_ns` | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ✅ |
| Parallel branch open/join step kinds | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🚫 |

**Evidence:** `tests/test_shims.py`, `tests/test_parallel_branches.py`,
`tests/test_step_types.py`

### §6.4 Blob frame (content-addressed dedup)

| Feature | Python | Rust | TypeScript | Go | JVM | .NET | WASM | Proxy |
|---|---|---|---|---|---|---|---|---|
| Write blob frames (reuse threshold) | ✅ | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 |
| Resolve `$blob` refs on read | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🚫 |

**Evidence:** `tests/test_compression.py`

### §6.5 Merkle summary + tail frame

| Feature | Python | Rust | TypeScript | Go | JVM | .NET | WASM | Proxy |
|---|---|---|---|---|---|---|---|---|
| Write merkle summary frame | ✅ | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 |
| Verify merkle root on read | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🚫 |

**Evidence:** `tests/test_attestation.py`

---

## §7 — Ed25519 signatures

| Feature | Python | Rust | TypeScript | Go | JVM | .NET | WASM | Proxy |
|---|---|---|---|---|---|---|---|---|
| Write per-frame Ed25519 signature | ✅ | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | ✅ |
| Verify Ed25519 signature chain | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🚫 |
| Reject invalid signature | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🚫 |
| Unsigned profile (`sig: "none"`) | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🚫 |

**Evidence:** `tests/test_attestation.py`; cross-language:
`tests/test_python_rust_differential.py`

---

## Frozen fixture corpus

The `stepback-core/fixtures/v1/` directory contains deterministic `.sb` files
used as the shared ground truth for cross-language conformance tests.

| Fixture | Purpose | Python | Rust | TS | Go | JVM | .NET | WASM |
|---|---|---|---|---|---|---|---|---|
| `minimal.sb` | 1-step signed trace | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `five_step.sb` | 5-step with all step kinds | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `corrupt_body.sb` | body tampered (P2 check) | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
| `canonical_vectors.jsonl` | 100 canonical-JSON test vectors | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**Evidence:** `tests/test_sbtrace_fixtures.py`; Rust:
`stepback-core/crates/sb-verify/tests/`

---

## RFC conformance

| RFC | Title | Python | Rust | TS | Go | JVM | .NET | WASM | Proxy |
|---|---|---|---|---|---|---|---|---|---|
| 0001 | SB-Trace Core Wire Format | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
| 0002 | Canonicalization | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🚫 |
| 0003 | Dirty-set semantics | ✅ | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 |
| 0004 | Attestation packs | ✅ | 🟡 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🟡 |
| 0005 | Importer lossiness | ✅ | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 |
| 0006 | OTel `agent.step.*` semantics | ✅ | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | ✅ |

---

## Next steps

The following gaps are highest-priority for reaching broad conformance:

1. **Bindings canonicalization edge cases** — NFC normalisation, surrogate
   rejection, and `\uXXXX` handling are 🟡 across all non-Python bindings.
   Add fixture-driven tests for each binding against
   `stepback-core/fixtures/v1/canonical/`.

2. **Step kind coverage in bindings** — all 8 `step_kind` values are only
   fully tested in Python.  Add a `step_kinds.sb` fixture covering every
   kind and run it through each binding reader.

3. **Unsigned profile coverage** — `sig: "none"` unsigned traces are 🟡 in
   all bindings except Rust.

4. **Rust writer** — `sb-format` is read-only; a Rust writer would enable
   round-trip cross-language tests.
