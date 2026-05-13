# Incident Replay and Cryptographic Audit Evidence for LLM Agent Systems

**Paper artifact — FAccT / industry track (v1)**

This document is the paper-grade artifact for the incident replay and audit
evidence capabilities of stepback. It discharges part of Step 147 of
[`100_STEPS.md`](../100_STEPS.md).

Cross-references:
[`stepback/trace_writer.py`](../stepback/trace_writer.py),
[`stepback/attestation.py`](../stepback/attestation.py),
[`stepback/redact.py`](../stepback/redact.py),
[`stepback/spec.py`](../stepback/spec.py),
[`SECURITY.md`](../SECURITY.md),
[`proofs/tla/SBHMACChain.tla`](../proofs/tla/SBHMACChain.tla),
[`docs/dirty-set-paper.md`](./dirty-set-paper.md).

---

## Abstract

When an LLM agent produces a harmful output, a wrong decision, or a policy
violation, the responsible party must answer: *What did the agent actually do?
Could we reproduce the failure? What was the minimal cause?*

stepback addresses these questions through three mechanisms: (1) a cryptographically
chained trace format (`.sb`) that provides tamper evidence for the recorded steps;
(2) a counterfactual replay engine that lets auditors substitute inputs and observe
the resulting outputs without re-running the full agent; and (3) a redaction and
ingestion pipeline that strips PII from traces while preserving cryptographic chain
continuity under attestation. This paper describes the design of each mechanism,
states what the evidence proves and what it does not prove, and presents case studies
using three synthetic incident corpora.

---

## 1. Motivation

AI agents operating in high-stakes domains — payments authorization, policy
enforcement, medical triage, regulated legal advice — must provide credible audit
trails. Current observability tools (LangSmith, Phoenix, Datadog APM) provide
structured logs, but:

1. **No tamper evidence**: Logs can be modified after the fact. There is no
   cryptographic binding between the recorded outputs and the original run.
2. **No counterfactual capability**: Auditors cannot ask "what would have happened
   if the policy input had been X?" without re-running the entire agent.
3. **No minimization**: The set of inputs that caused a violation may span dozens
   of steps; isolating the minimal cause requires manual inspection or expensive
   full re-execution.
4. **Privacy / redaction gap**: Production traces contain PII that cannot be
   shared with auditors or regulators in raw form.

stepback is designed to close these four gaps.

---

## 2. Cryptographic audit trail

### 2.1 The `.sb` wire format

Each agent step is written as a length-prefixed canonical-JSON frame with:

- A per-frame HMAC-SHA256 chain: frame N's HMAC input includes the previous
  frame's HMAC, so any deletion, insertion, or reordering of frames is detected.
- A per-frame Ed25519 signature over the canonical body bytes and the HMAC.
- A tail frame that closes the chain; truncation is detectable because the
  tail's `prev_hmac` must match the last step frame's `hmac`.

The full wire format is specified in [`spec/sbtrace-v1.md`](../spec/sbtrace-v1.md).

### 2.2 What the evidence proves

Per the [`SECURITY.md`](../SECURITY.md) threat model:

| Claim | Proven? | Mechanism |
|-------|---------|-----------|
| Each frame body was not modified after recording | ✅ Yes | Ed25519 signature over body bytes |
| No frame was inserted, deleted, or reordered | ✅ Yes | HMAC chain; tail `prev_hmac` |
| Truncation after step k is detectable | ✅ Yes | Missing tail; tail `prev_hmac` mismatch |
| Mandatory capabilities are not silently skipped | ✅ Yes | Capability fail-closed check |
| Merkle summary matches HMAC chain | ✅ Yes | Merkle summary frame verification |

### 2.3 What the evidence does NOT prove

Per the [`SECURITY.md`](../SECURITY.md) "What the scheme does NOT prove" table:

| Claim | Not proven | Reason |
|-------|------------|--------|
| The recorded outputs are *correct* or *appropriate* | ❌ | Content correctness is not a cryptographic property. |
| The recorder was honest at record time | ❌ | A lying recorder can write fraudulent frames; signatures prove post-hoc tamper evidence, not recorder honesty. |
| The HMAC key was not exposed | ❌ | If the HMAC key is exposed, an adversary can rewrite the entire chain. See key handling in `SECURITY.md`. |
| The Ed25519 public key belongs to the claimed identity | ❌ | The public key is self-asserted; callers must pin it against a known-good value for identity proof. |
| Clock accuracy | ❌ | `wallclock_ns` is recorded from the system clock; it is not a trusted timestamp. |
| PII is absent | ❌ | Redaction is an application-layer concern handled by `stepback/redact.py`. |

This boundary is formally modelled in
[`proofs/tla/SBHMACChain.tla`](../proofs/tla/SBHMACChain.tla): the TLA+ spec
models the writer, a body-substitution adversary, and proves the three chain
invariants, but explicitly notes that truncation detection and canonicalization
are modelled separately.

### 2.4 Attestation packs

For benchmark submissions and incident packs, `stepback/attestation.py` bundles
multiple `.sb` traces into a signed `.pack` artifact with a Merkle summary.
SLSA and in-toto provenance attestations can be attached for supply-chain
evidence.

---

## 3. Counterfactual incident replay

### 3.1 Substitution model

An auditor can pose counterfactual questions by staging substitutions:

```python
from stepback import record, Trace
from stepback.substitutions import PromptSubstitution, ToolOutputSubstitution

trace = Trace(path="incident_2024_01_15.sb", hmac_key=hmac_key)
# "What if the policy document had not included the exclusion clause?"
trace.substitute(PromptSubstitution(
    target_id="policy_lookup_step",
    replacement={"messages": redacted_policy_messages}
))
result = trace.replay_forward(executor=production_executor_stub)
```

Only the steps affected by the substitution (the dirty set) are re-executed;
all other steps are served from the content-addressed cache. The auditor
can inspect which downstream outputs changed without paying O(N) LLM call cost.

### 3.2 Incident minimization

When the substitution set is large (multiple concurrent policy changes, A/B
model variants, retrieval changes), `minimize` finds the 1-minimal subset:

```python
from stepback import minimize_substitutions, Trace
from stepback.minimize import MinimizationOptions

trace = Trace(path="incident.sb", hmac_key=hmac_key)
result = minimize_substitutions(
    trace,
    substitution_set=all_candidate_substitutions,
    predicate=lambda replay: "violation" in replay.output_text("final_response"),
    options=MinimizationOptions(strategy="ddmin")
)
print(f"Minimal cause: {result.minimal_set}")
```

See [`docs/minimization-paper.md`](./minimization-paper.md) for the full
algorithm description.

---

## 4. Redaction and privacy-preserving ingestion

### 4.1 The ingestion pipeline

`stepback/redact.py` implements:

1. **PII scanning**: pluggable detectors (regex, model-based) scan frame bodies.
2. **Streaming redaction**: PII fields are replaced with deterministic tokens
   (`[REDACTED_EMAIL_1]`, etc.) while preserving the canonical JSON structure.
3. **TOCTOU guard**: the redacted file is re-hashed after writing to detect
   concurrent modification.
4. **Redaction attestation**: an Ed25519-signed `RedactionAttestation` records
   the original and redacted file SHA-256s, the policy fingerprint, and the
   redaction manifest. Verifiers can confirm the redaction was performed by
   an authorized policy without re-reading the original file.

### 4.2 Privacy limitations

| Limitation | Notes |
|------------|-------|
| Hash preservation | Content-addressed step hashes in the `.sb` file may themselves reveal information about the original inputs (pre-image resistance of SHA-256 protects against reversal, but timing / length metadata may leak structure). |
| Embeddings and token-level analysis | LLM request and response embeddings are not recorded in v1 `.sb` files; if they were, redacting the text field would not redact the embeddings. |
| Inference from context | A redacted trace showing step kinds, step counts, and timing may still allow inference about the original agent task. |
| Attestation trust | The `RedactionAttestation` signature proves the stated policy was applied; it does not prevent a malicious policy author from designing a policy that leaks data. |

---

## 5. Case studies (synthetic)

The three author-original corpora in `stepback/bench/corpora/` were designed
to simulate incident scenarios:

### 5.1 Customer support incident (support-agent corpus)

**Scenario**: A support agent incorrectly authorizes a refund for a non-eligible
customer. The auditor stages a `ToolOutputSubstitution` on the
`check_refund_eligibility` step to test the counterfactual: if the eligibility
check had returned "not eligible," would the agent still have authorized the refund?

**Result** (from `tests/test_author_corpora.py`): The dirty set for this
substitution has size 2–3 (eligibility result + authorization step + response
formatting step). The agent would have redirected to the denial path.

### 5.2 Code review policy violation (code-review corpus)

**Scenario**: A code review agent approves a SQL injection vulnerability.
The auditor minimizes the substitution set to find the minimal prompt change
that would have caused rejection.

**Result**: The minimal substitution is a single `PromptSubstitution` on the
security-check step's input. The remaining 5 steps are served from cache.

### 5.3 Payments policy violation (payments-policy corpus)

**Scenario**: A payment is incorrectly authorized for a sanctioned entity.
The auditor replays with and without the `sanctions_check` tool result to
determine whether the policy rule was applied.

**Result**: The sanctions check result flows directly into the authorization
decision; dirtying the sanctions step dirties exactly 2 downstream steps.

**Note**: All three case studies use synthetic deterministic LLMs and scripted
tool responses. Real-world incident analysis requires production traces,
appropriate key custody, privacy review, and possibly legal authorization.

---

## 6. Implementation status

| Feature | Status | Evidence |
|---------|--------|----------|
| HMAC-SHA256 + Ed25519 frame chain | **Implemented** | `tests/test_attestation.py` (7+) |
| Tamper detection (flip any byte) | **Implemented** | `tests/test_reader_corruption.py` |
| Capability fail-closed | **Implemented** | `tests/test_capability_negotiation.py` (14) |
| Merkle summary frame | **Implemented** | `stepback/trace_writer.py` |
| TLA+ formal chain model | **Implemented** | `proofs/tla/SBHMACChain.tla` (49 tests) |
| Counterfactual substitution replay | **Implemented** | `tests/test_dirty_set_*.py` |
| Incident minimization | **Implemented** | `tests/test_minimize*.py`, `docs/minimization-paper.md` |
| Redaction + attestation | **Implemented** | `tests/test_ingestion.py` (29 tests) |
| SLSA / in-toto attestation export | **Implemented** | `stepback/exporters.py` |
| Transparency log integration | **Prototype** | Step 132 |
| Hardware-backed key support (PKCS#11) | **Prototype** | Step 133 |
| Production case study (real traces) | **Future work** | Requires IRB / legal review |

---

## 7. Deployment considerations

| Question | Answer |
|----------|--------|
| Who holds the HMAC key? | The recorder process. The HMAC key must be treated as a secret; if exposed, the entire trace can be rewritten. Only the `hmac_key_id` fingerprint is stored in the trace. |
| Who holds the Ed25519 signing key? | The recorder. For long-lived recorders, consider key rotation (Step 128) and hardware-backed custody (Step 133). |
| What do auditors receive? | The `.sb` trace file and the HMAC key (or just the Ed25519 public key for signature verification without content-address verification). |
| How are PII traces shared with regulators? | Via the redaction + attestation pipeline; the regulator receives a redacted `.sb` file plus a `RedactionAttestation` signed by the privacy officer's key. |
| Can the recorder lie? | Yes; signatures prove post-hoc tamper evidence, not recorder honesty. An honest-recorder assumption must be documented in the audit policy. |

---

## 8. Related work

See [`RELATED_WORK.md`](../RELATED_WORK.md), especially:

- §2.7 (cryptographic audit logs: Certificate Transparency, Sigstore, SLSA)
- §2.5 (HTTP cassette limitations for incident replay)

Additional relevant work:
- **GDPR/CPRA audit requirements**: EU AI Act Art. 12 requires "appropriate levels
  of accuracy, robustness, and cybersecurity" for high-risk AI systems and
  post-market monitoring; a verifiable trace format is a candidate technical
  implementation.
- **NIST AI RMF**: Govern, Map, Measure, Manage framework; the `.sb` trace format
  addresses the Measure quadrant (traceability and explainability).

---

## 9. Limitations

1. **Synthetic corpora only**: No real production incident has been replayed
   using this system. The case studies above use synthetic deterministic agents.
2. **Recorder honesty assumption**: The cryptographic chain proves post-hoc
   tamper evidence, not that the recorder was honest at record time.
3. **No legal admissibility claim**: Whether a `.sb` trace constitutes
   admissible evidence in a legal or regulatory proceeding is a jurisdiction-
   specific legal question outside the scope of this paper.
4. **Redaction does not guarantee anonymization**: Redacting named entities
   does not guarantee that the trace cannot be re-identified from context,
   timing, or structure.
5. **Clock accuracy**: `wallclock_ns` is not a trusted timestamp; it can be
   manipulated by the recording process.

---

## 10. Conclusion

stepback provides a practical foundation for LLM agent audit trails: tamper-evident
frames, counterfactual replay at O(|dirty\_set|) cost, incident minimization, and
privacy-preserving redaction. The cryptographic guarantees are narrow and explicitly
documented (`SECURITY.md`): they prove per-frame tamper evidence and chain integrity,
not content correctness or recorder honesty. Systems that rely on these guarantees for
regulatory compliance or legal proceedings must additionally establish recorder
trustworthiness through deployment controls (key custody, HSM signing, audit logging
of the recorder process itself) that are outside the scope of the `.sb` format.
