# Production Case Studies

Six production-shaped case studies demonstrating `stepback` at real-world
operational scale. All case studies are fully offline (no network calls) and
deterministic (seeded RNG), so they run in CI without API keys.

---

## Overview

| Case Study | Scenario | Key APIs |
|---|---|---|
| [High-Volume Recording](#1-high-volume-recording) | Millions-of-runs/day throughput and storage | `record`, `RecorderKey` |
| [Incident Replay](#2-incident-replay) | Root-cause isolation via bisection | `replay`, `bisect`, `ToolOutputSubstitution` |
| [Evidence Packs](#3-evidence-packs) | Regulator/auditor signed attestation packs | `build_attestation_pack`, `write_attestation_pack`, `verify_attestation_pack` |
| [Model Migration](#4-model-migration) | Dirty-set impact of model swap | `ModelSubstitution`, `run_replay` |
| [Parameter Sweep](#5-parameter-sweep) | Grid of substitutions over a trace corpus | `sweep_traces`, `PromptSubstitution`, `ToolOutputSubstitution` |
| [Redacted Publication](#6-redacted-publication) | PII redaction before external sharing | `redact_trace_file`, `sign_redaction_attestation` |

---

## 1. High-Volume Recording

**File:** `stepback/case_studies/high_volume.py`  
**Entry point:** `run_high_volume(n_traces, n_steps, seed)`

### Production problem

An enterprise support bot handles millions of customer queries per day.
Every query must be recorded as a signed, content-addressed `.sb` trace
for audit and replay — but recording must not become a throughput bottleneck.

### What it models

- Records `n_traces` synthetic traces with Ed25519 + HMAC-chain signing enabled.
- Measures raw throughput (`traces_per_second`) and projects to daily capacity.
- Compares compressed vs. uncompressed storage to quantify the trade-off.
- Tracks `recording_errors` (errors in the recorder callback) separately from
  successful writes so operators can set alerting thresholds.

### Key result fields

| Field | What it tells you |
|---|---|
| `projected_runs_per_day` | Whether a single process can sustain the target daily volume |
| `compression_ratio` | Storage saving from zlib frame compression (typically 1.5–3×) |
| `mean_bytes_per_trace` | Baseline storage budget per trace |
| `success_rate` | Whether any recorder errors occurred |

### How to run

```python
from stepback.case_studies.high_volume import run_high_volume

result = run_high_volume(n_traces=100, n_steps=8, seed=42)
print(result.summary_line())
print(result.to_json())
```

---

## 2. Incident Replay

**File:** `stepback/case_studies/incident_replay.py`  
**Entry point:** `run_incident_replay(output_dir)`

### Production problem

An AI compliance agent marked a valid product as "policy blocked" in
production, causing a false rejection.  The on-call engineer needs to find the
root cause without re-running the agent against live APIs.

### What it models

1. Records a 6-step research agent trace where a policy-check tool deliberately
   returns a false positive at step 4.
2. Injects a corrective `ToolOutputSubstitution` (the correct tool response).
3. Calls `trace.bisect()` to find the earliest step whose output changes under
   the corrective substitution.
4. Confirms the identified culprit matches the known fault location.

**Key insight:** The entire bisection is served from the replay cache —
zero live LLM calls are needed, so the post-mortem can run in seconds.

### Key result fields

| Field | What it tells you |
|---|---|
| `culprit_found` | Whether bisection located the correct fault step |
| `bisect_probes` | How many steps were examined (O(log N)) |
| `dirty_after_fix` | How many downstream steps are affected by the fix |

### How to run

```python
from stepback.case_studies.incident_replay import run_incident_replay

result = run_incident_replay()
print(result.summary_line())
```

---

## 3. Evidence Packs

**File:** `stepback/case_studies/evidence_packs.py`  
**Entry point:** `run_evidence_pack(policy_version, output_dir)`

### Production problem

A regulated financial-services AI system must produce legally defensible
evidence for auditor review: tamper-evident records of every lending decision
including the policy version that was in effect and an independent attestor
signature.

### What it models

1. Records a 15-step trace (5 loan applications × 3 steps each).
2. Builds an `AttestationPack` linking the trace to a pinned policy version,
   signed by a dedicated attestor Ed25519 key.
3. Writes the signed pack to a `.sbpack` file.
4. Verifies the pack using `verify_attestation_pack` (as an auditor would).

### Three-layer integrity

| Layer | Mechanism |
|---|---|
| Frame | Per-frame HMAC-SHA256 + Ed25519 signature |
| Trace | HMAC-chain across all frames; Merkle summary |
| Pack | Outer Ed25519 body signature; policy version pin |

### Key result fields

| Field | What it tells you |
|---|---|
| `pack_verified` | Whether the attestation pack passed all integrity checks |
| `attestor_public_key` | Fingerprint to compare against organisation key registry |
| `policy_version_pin` | Policy document version committed into the pack |

### How to run

```python
from stepback.case_studies.evidence_packs import run_evidence_pack

result = run_evidence_pack(policy_version="v2.3.1")
print(result.summary_line())
```

---

## 4. Model Migration

**File:** `stepback/case_studies/model_migration.py`  
**Entry point:** `run_model_migration(n_traces, n_steps, seed)`

### Production problem

An ML team wants to migrate from `gpt-4o-2024-11-20` to the cheaper
`gpt-4o-mini-2024-07-18` but needs to quantify the impact on downstream
decisions before deploying to production.

### What it models

1. Records `n_traces` synthetic traces under GPT-4o (model A).
2. For each trace, applies `ModelSubstitution` on every `llm_call` step to
   swap to GPT-4o-mini (model B).
3. Replays forward with a GPT-4o-mini executor.
4. Measures `dirty_fraction_mean` (fraction of steps that become dirty under
   the swap) and `cost_reduction_factor` (from the stepback pricing catalog).

### Key result fields

| Field | What it tells you |
|---|---|
| `dirty_fraction_mean` | What fraction of steps are affected by the model swap |
| `cost_reduction_factor` | Token-price ratio A/B (> 1 means B is cheaper) |
| `total_cost_a_usd` / `total_cost_b_usd` | Absolute cost comparison |

### How to run

```python
from stepback.case_studies.model_migration import run_model_migration

result = run_model_migration(n_traces=10, n_steps=12, seed=42)
print(result.summary_line())
```

---

## 5. Parameter Sweep

**File:** `stepback/case_studies/parameter_sweep.py`  
**Entry point:** `run_parameter_sweep(n_traces, n_steps, seed)`

### Production problem

A code-review agent team wants to evaluate two counterfactual scenarios before
deploying changes to production:
- Would a stricter security-focused system prompt increase flagging rates?
- Would injecting a critical security finding change downstream decisions?

### What it models

1. Builds a corpus of `n_traces` synthetic code-review agent traces.
2. Defines two substitution candidates: a system-prompt swap and a
   tool-output injection.
3. Runs `sweep_traces()` for each candidate over the corpus.
4. Aggregates `diverged_fraction` (fraction of trace × candidate pairs where
   at least one step changed) and `mean_dirty_count`.

### Key result fields

| Field | What it tells you |
|---|---|
| `diverged_fraction` | Fraction of corpus traces affected by each candidate |
| `dirty_count_per_candidate` | Per-candidate mean dirty step count |
| `total_traces_diverged` | Total (trace, candidate) pairs with divergence |

### How to run

```python
from stepback.case_studies.parameter_sweep import run_parameter_sweep

result = run_parameter_sweep(n_traces=10, n_steps=8, seed=0)
print(result.summary_line())
```

---

## 6. Redacted Publication

**File:** `stepback/case_studies/redacted_publication.py`  
**Entry point:** `run_redacted_publication(output_dir)`

### Production problem

A customer-service AI team wants to share production traces with academic
collaborators, but the traces contain PII (email addresses, phone numbers,
account IDs).  They must redact the PII while preserving the trace's replay
utility and attaching a compliance attestation.

### What it models

1. Records an 8-step trace (4 customers × 2 steps) with synthetic PII in
   both tool outputs and LLM responses.
2. Runs a pre-redaction scan to count PII findings without modifying anything.
3. Applies `STANDARD_POLICY` redaction with the `"hash"` strategy — matched
   PII is replaced by stable `<REDACTED:email:ab12cd34>` tokens so the cache
   structure (same-email → same-token) is preserved for replay.
4. Verifies the redacted trace's HMAC chain under its new recorder key.
5. Signs a `RedactionAttestation` with the compliance team's Ed25519 key.
6. Verifies the attestation and checks that no raw PII remains in the
   redacted file bytes.

### Key result fields

| Field | What it tells you |
|---|---|
| `scan_matches` | How many PII findings were found in the original trace |
| `redaction_matches` | How many matches were actually redacted |
| `redacted_trace_verified` | Whether the redacted trace is self-consistent |
| `attestation_verified` | Whether the compliance signature is valid |
| `no_raw_pii_in_redacted` | Whether all known PII was removed |

### How to run

```python
from stepback.case_studies.redacted_publication import run_redacted_publication

result = run_redacted_publication()
print(result.summary_line())
```

---

## Running all case studies

```python
from stepback.case_studies import (
    run_high_volume,
    run_incident_replay,
    run_evidence_pack,
    run_model_migration,
    run_parameter_sweep,
    run_redacted_publication,
)

for fn, kwargs in [
    (run_high_volume, {"n_traces": 20}),
    (run_incident_replay, {}),
    (run_evidence_pack, {}),
    (run_model_migration, {"n_traces": 5}),
    (run_parameter_sweep, {"n_traces": 5}),
    (run_redacted_publication, {}),
]:
    result = fn(**kwargs)
    print(result.summary_line())
```

## Scale simulation vs. extrapolation

The default parameters are chosen for speed (all six case studies complete
in a few seconds on a laptop).  The `projected_runs_per_day` field in
`HighVolumeResult` and the `cost_reduction_factor` in `ModelMigrationResult`
let you extrapolate from the small-scale run to real production volumes
without having to record millions of traces in a test.
