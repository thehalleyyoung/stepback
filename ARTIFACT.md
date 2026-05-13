# stepback Research Artifacts

This document is the canonical index of all research artifacts produced by or
alongside the stepback project. It discharges Step 147 of
[`100_STEPS.md`](100_STEPS.md):

> *"Write `RELATED_WORK.md`, `ARTIFACT.md`, and the 4-5 paper line …"*

Each entry below lists the artifact type, path, maturity/status, reproducibility
command (where applicable), related tests, and redistribution/privacy notes.

---

## Overview

The stepback research program has five paper-grade artifacts spanning systems,
algorithms, benchmarks, and case studies:

| # | Title | Venue target | Status | Doc |
|---|-------|--------------|--------|-----|
| P1 | Dirty-set replay for LLM agents | MLSys / NeurIPS Systems | Implemented & evaluated on synthetic corpora | [`docs/dirty-set-paper.md`](docs/dirty-set-paper.md) |
| P2 | Distributed dirty-set replay at scale | OSDI / ASPLOS | Core implemented; distributed layer prototype | [`docs/distributed-replay-paper.md`](docs/distributed-replay-paper.md) |
| P3 | Stochastic minimization and regression localization | ICSE / ISSTA | Implemented; confidence-interval extension | [`docs/stochastic-minimization-paper.md`](docs/stochastic-minimization-paper.md) |
| P4 | stepback-bench dataset and benchmark suite | NeurIPS Datasets & Benchmarks | Implemented; paper artifact complete | [`docs/neurips-datasets-paper.md`](docs/neurips-datasets-paper.md) |
| P5 | Incident replay and cryptographic audit evidence | FAccT / industry track | Core implemented; case-study corpora synthetic | [`docs/incident-audit-paper.md`](docs/incident-audit-paper.md) |

Supporting formal artifacts:

| Artifact | Path | Status |
|----------|------|--------|
| TLA+ HMAC chain spec | [`proofs/tla/SBHMACChain.tla`](proofs/tla/SBHMACChain.tla) | Complete; verifiable with TLC |
| Lean 4 dirty-set soundness | `proofs/lean/` | Proof sketch (Step 56) |
| SB-Trace v1 spec | [`spec/sbtrace-v1.md`](spec/sbtrace-v1.md) | Normative, frozen for v1 |
| SB-Trace v2 spec (CBOR) | [`spec/sbtrace-v2.md`](spec/sbtrace-v2.md) | Draft / candidate |
| JSON Schema for frame kinds | [`spec/schema/v1/`](spec/schema/v1/) | Stable for v1 |
| Frozen fixture corpus | [`stepback-core/fixtures/v1/`](stepback-core/fixtures/v1/) | Frozen; SHA-256 pinned |
| Author-original `.sb` corpora | [`stepback/bench/corpora/`](stepback/bench/corpora/) | Redistributable; Apache-2.0 |

---

## A. Source-code artifacts

### A.1 Core implementation modules

| Module | Description | Tests |
|--------|-------------|-------|
| [`stepback/divergence.py`](stepback/divergence.py) | Dirty-set propagation engine with contract docs | `tests/test_divergence*.py` |
| [`stepback/replay.py`](stepback/replay.py) | Replay planner and executor; branch-aware scheduling | `tests/test_replay*.py`, `tests/test_trace_mutation_contract.py` |
| [`stepback/minimize.py`](stepback/minimize.py) | DDMin, Shapley, multi-objective, binary halving strategies | `tests/test_minimize*.py`, `tests/test_stochastic_replay.py` |
| [`stepback/canonical.py`](stepback/canonical.py) | Canonical JSON v1 encoder | `tests/test_canonical*.py` |
| [`stepback/trace_writer.py`](stepback/trace_writer.py) | HMAC-chained Ed25519-signed frame writer | `tests/test_attestation.py`, `tests/test_compression.py` |
| [`stepback/trace_reader.py`](stepback/trace_reader.py) | Length-prefixed frame reader with DoS bounds | `tests/test_reader_fuzz.py`, `tests/test_reader_corruption.py` |
| [`stepback/attestation.py`](stepback/attestation.py) | Attestation pack builder and verifier | `tests/test_attestation.py` |
| [`stepback/spec.py`](stepback/spec.py) | SBTraceSpec, conformance validator, wire-version helpers | `tests/test_spec_sbtrace_spec.py`, `tests/test_spec_wire_version.py` |

### A.2 Distributed / scale modules

| Module | Description | Status | Tests |
|--------|-------------|--------|-------|
| [`stepback/distributed_bisect.py`](stepback/distributed_bisect.py) | Corpus-level parallel bisect via thread pool | Implemented | `tests/test_distributed_bisect.py` |
| [`stepback/bench/soak.py`](stepback/bench/soak.py) | 10 000-trace soak benchmark | Implemented | `tests/test_soak.py` |
| `stepback/event_bus.py` | Kafka-backed event bus for replay jobs | Prototype / spec | — |
| `stepback/worker_lease.py` | Worker lease and checkpoint model | Prototype / spec | — |
| `stepback/distributed_cache.py` | Sharded S3/GCS/Azure step cache | Prototype / spec | — |

### A.3 Shims and recorders

All ten provider shims (`openai`, `anthropic`, `bedrock`, `gemini`, `langchain`,
`mcp`, `azure_openai`, `vertex_ai`, `cohere`, `mistral`) plus nine framework
recorders (LlamaIndex, DSPy, Haystack, AutoGen, CrewAI, SemanticKernel, Strands,
PydanticAI, InspectAI) are in [`stepback/shims.py`](stepback/shims.py).

---

## B. Specification artifacts

| Artifact | Path | Version | Redistribution |
|----------|------|---------|----------------|
| SB-Trace v1 wire spec | `spec/sbtrace-v1.md` | v1.0.0 | Apache-2.0 |
| SB-Trace v2 draft | `spec/sbtrace-v2.md` | v2.0 draft | Apache-2.0 |
| Canonical JSON v1 rules | `docs/canonicalization.md` | v1 | Apache-2.0 |
| Dirty-set semantics | `docs/dirty-set.md` | v1 | Apache-2.0 |
| API compatibility reference | `docs/api-compat.md` | v0.1.0 | Apache-2.0 |
| OTel agent.step conformance | `docs/otel-conformance.md` | draft | Apache-2.0 |
| SB-Trace RFC index | `spec/rfcs/` | drafts | Apache-2.0 |

---

## C. Formal proofs and mechanizations

### C.1 TLA+ HMAC chain

**Path:** `proofs/tla/SBHMACChain.tla`

**Verified properties:**
- `IntegrityInvariant` (P1): an untampered log validates.
- `TamperEvidenceInvariant` (P2): a body-modified log fails validation.
- `WriterCoherenceInvariant` (P3): the writer log is self-consistent.

**To verify:**
```bash
tlc -config proofs/tla/SBHMACChain.cfg proofs/tla/SBHMACChain.tla
```
(TLC must be installed separately; see `proofs/tla/README.md`.)

**Tests:** `tests/test_tla_spec.py` (49 tests; structural invariants run without TLC).

### C.2 Lean 4 dirty-set soundness

**Path:** `proofs/lean/`

**Claim:** Under substitution sigma and a collision-free hash assumption, every
non-dirty replayed step is observationally equivalent to full re-execution.

See `docs/dirty-set-soundness.md` for the paper proof; the Lean mechanization
formalizes the same theorem over an immutable step DAG.

---

## D. Benchmark artifacts

### D.1 Author-original corpora (redistributable)

| Corpus | Tasks | Traces | Step range | License |
|--------|-------|--------|------------|---------|
| `support-agent` | 5 | 5 | 5–6 | Apache-2.0 |
| `code-review` | 5 | 5 | 6 | Apache-2.0 |
| `payments-policy` | 5 | 5 | 4–7 | Apache-2.0 |

**Path:** `stepback/bench/corpora/`

**Verify:**
```bash
python3 -m stepback bench soak --n-traces 15 --n-steps 6
```

**Regenerate (idempotent):**
```bash
python3 scripts/generate_author_corpora.py
```

**Tests:** `tests/test_author_corpora.py` (83 tests).

### D.2 External corpus loaders

Six loaders for external benchmarks (SWE-bench-Verified, GAIA, tau-bench,
AgentBench, OSWorld, WebArena) are in `stepback/bench/corpus_loaders.py`.
Each is tested against synthetic fixtures to guard the parsing contract
independently of external corpus downloads.

### D.3 Benchmark result schema

**Path:** `stepback/bench/result_schema.py`  
**Schema version:** `"1.0"`  
**Tests:** `tests/test_bench_schema.py` (32 tests).

**Run benchmark:**
```bash
python3 -m stepback bench replay-caching --n-steps 50 --n-trials 10 --out results.json
```

### D.4 Submission rules and leaderboard

**Path:** `stepback/bench/submission.py`, `stepback/bench/leaderboard.py`  
**Rules doc:** `docs/submission-rules.md`  
**Tests:** `tests/test_bench_submission.py` (52 tests), `tests/test_bench_leaderboard.py`.

---

## E. Paper documents

Each paper document is self-contained, includes an implementation-status
table for every claimed contribution, cites existing tests as evidence,
and contains a "Limitations" section.

| Paper | Path | Target venue | Size |
|-------|------|--------------|------|
| P1: Dirty-set replay | `docs/dirty-set-paper.md` | MLSys / NeurIPS | ~20 KB |
| P2: Distributed replay | `docs/distributed-replay-paper.md` | OSDI / ASPLOS | ~18 KB |
| P3: Stochastic minimization | `docs/stochastic-minimization-paper.md` | ICSE / ISSTA | ~18 KB |
| P4: NeurIPS benchmark | `docs/neurips-datasets-paper.md` | NeurIPS D&B | ~29 KB |
| P5: Incident audit | `docs/incident-audit-paper.md` | FAccT / industry | ~18 KB |

Supporting artifacts:
- `docs/minimization-paper.md` — companion to P3; covers the causal minimization
  toolkit in detail (Step 88).
- `docs/dirty-set-soundness.md` — paper-proof companion to P1.
- `docs/dirty-set-completeness.md` — completeness theorem companion to P1.
- `docs/dirty-set-complexity.md` — asymptotic complexity companion to P1.
- `docs/dirty-set-distributions.md` — empirical distributions companion to P4.

---

## F. Security and compliance artifacts

| Artifact | Path | Notes |
|----------|------|-------|
| Security policy | `SECURITY.md` | Threat model, verifier guarantees, disclosure path |
| Reader DoS bounds | `docs/reader-limits.md` | Frame/string/nesting limits |
| Redaction and ingestion | `stepback/redact.py` | GDPR-friendly redaction + attestation |
| Conformance dashboard | `docs/conformance-dashboard.md` | Cross-language implementation status |

---

## G. Related work

See [`RELATED_WORK.md`](RELATED_WORK.md) for a structured survey of prior work
across five comparison axes: observability, time-travel debugging, build-cache
invalidation, delta debugging, and cryptographic audit logs.

---

## H. How to reproduce key results

```bash
# 1. Install
pipx install stepback          # or: pip install -e ".[dev]" in a venv

# 2. Run full test suite
pytest -q --tb=short

# 3. Replay-caching benchmark
python3 -m stepback bench replay-caching --n-steps 100 --n-trials 20 --out rc.json

# 4. Soak test (small fleet)
python3 -m stepback bench soak --n-traces 100 --n-steps 12 --out soak.json

# 5. Model-swap differential
python3 -m stepback bench model-swap --n-steps 50 --n-trials 10 --out ms.json

# 6. Run conformance suite against the Python implementation
python3 -m stepback spec test -- python3 -m stepback verify

# 7. Validate a submission manifest
python3 -m stepback bench validate-submission path/to/submission.json
```

---

*This index was created as part of Step 147 of the stepback OSS-readiness
plan. Update it when new paper artifacts are added or existing artifacts
reach a new maturity milestone.*
