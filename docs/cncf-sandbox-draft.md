# SB-Trace — CNCF Sandbox Proposal (Draft)

> **Status: DRAFT — precondition not yet met.**
>
> The CNCF Sandbox acceptance criteria include production deployments and a
> demonstrated community.  This document records the proposal structure so it
> is ready when those conditions are met.  The specific precondition from
> [100_STEPS.md](../100_STEPS.md) §145 is:
>
> > *"once multi-implementation production use exists"*
>
> See the "Preconditions checklist" at the end of this document.

---

## Project name

**SB-Trace**

## CNCF category / SIG

Observability (primary); Runtime (secondary)

## One-line description

An open wire format and multi-language runtime for tamper-evident recording,
counterfactual replay, and auditing of AI-agent execution traces.

---

## 1. About the project

SB-Trace addresses a gap in the cloud-native AI observability landscape: while
OpenTelemetry provides spans and metrics for services, it does not provide the
*replay-capable*, *tamper-evident*, *content-addressed* trace record needed for
AI-agent audit, cost optimisation, and incident replay.

SB-Trace is complementary to OTel: it proposes `agent.step.*` semantic
conventions to the OTel GenAI SIG (RFC 0006) and can export/import OTel spans,
while providing a richer, standalone record for workloads that need it.

### Key properties

| Property | How SB-Trace provides it |
|---|---|
| Tamper-evident | HMAC chain + Ed25519 per-frame signatures |
| Replay-capable | Dirty-set algorithm recomputes only affected steps |
| Content-addressed | BLAKE2b step-input hashes, SHA-256 blob dedup |
| Multi-language | Python, Rust, TypeScript, Go, JVM, .NET, WASM |
| OTel-aligned | `agent.step.*` conventions; import/export round-trip |
| Formally verified | Lean 4 soundness proof; TLA+ HMAC-chain model |

---

## 2. Alignment with CNCF mission

CNCF's mission is to make cloud-native computing universal and sustainable.
SB-Trace advances this by:

- **Portability**: open format with multi-language readers independent of any
  cloud provider or LLM vendor.
- **Observability**: fills the "AI agent execution" gap in the OTel ecosystem.
- **Security**: tamper-evident records with formal proofs support
  compliance-sensitive deployments.
- **Sustainability**: efficient dirty-set replay reduces unnecessary LLM calls,
  cutting cost and energy use during debugging and evaluation.

---

## 3. Sandbox criteria self-assessment

The [CNCF Sandbox criteria](https://github.com/cncf/toc/blob/main/process/sandbox.md)
require:

| Criterion | Status |
|---|---|
| Clear value proposition | ✅ (see §1–2) |
| Basic CI | 🟡 GitHub Actions in place; needs Linux matrix hardening |
| Apache-2.0 or compatible license | ✅ |
| DCO or CLA | ⬜ DCO not yet enforced on PRs |
| Security policy | ✅ `SECURITY.md` present |
| At least 2 maintainers from different companies | ⬜ **Precondition not met** |
| Production use by at least 3 end users | ⬜ **Precondition not met** |
| Alignment with a CNCF SIG | ✅ Observability SIG; OTel GenAI SIG |
| No existing CNCF project overlap | ✅ (OTel spans ≠ replay-capable trace store) |

---

## 4. Governance (proposed)

Follows the CNCF Minimal Viable Governance template with the same TSC
structure described in the Linux Foundation proposal
([`docs/lf-proposal.md`](lf-proposal.md) §6–7).

A CNCF submission would add:

- `CHARTER.md` aligned with CNCF IP policy
- GitHub DCO check on all PRs
- CNCF Code of Conduct adoption
- Annual security audit (CNCF Security TAG)

---

## 5. Relationship to other CNCF projects

| Project | Relationship |
|---|---|
| OpenTelemetry | SB-Trace proposes `agent.step.*` conventions to OTel GenAI SIG; import/export bridge in `stepback/exporters.py` |
| Fluentd / Fluent Bit | SB-Trace traces could be forwarded as structured log events |
| KEDA | Dirty-set replay jobs are natural KEDA-scaled workloads |
| Argo Workflows | Agent replay sweeps map to Argo DAG steps |

---

## 6. Preconditions checklist

The following conditions must all be met before submitting this proposal to
the CNCF TOC:

- [ ] **Multi-implementation production use**: at least 3 independent
      organisations running SB-Trace (any language binding) in production.
- [ ] **Community**: at least 2 maintainers from different organisations
      with sustained contributions over ≥ 6 months.
- [ ] **Governance documents**: `GOVERNANCE.md`, `CONTRIBUTING.md`,
      `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `CHARTER.md`.
- [ ] **DCO** enforced on all PRs.
- [ ] **CI matrix** green on Linux (Ubuntu LTS), macOS, and Windows for
      Python 3.10–3.14.
- [ ] **Signed releases**: PyPI + crates.io + npm + Maven Central +
      NuGet with SLSA provenance ≥ level 2.
- [ ] **OTel SIG feedback**: `agent.step.*` conventions at Experimental
      or higher, or an explicit decision from the OTel GenAI SIG.
- [ ] **Security audit**: informal review by at least one external
      security researcher; findings documented and addressed.
- [ ] **LF membership or sponsorship**: at least one LF member sponsor
      willing to shepherd the CNCF due-diligence process.

---

*Draft prepared by the stepback maintainers, 2026-05-12.*
*No formal CNCF submission has been made.*
