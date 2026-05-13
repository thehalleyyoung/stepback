# SB-Trace — Linux Foundation Proposal (Draft)

> **Status: DRAFT — precondition not yet met.**
>
> Submission to the Linux Foundation requires demonstrated multi-organisation
> production use and a committed governance group.  This document records the
> proposal structure so it is ready to submit when those conditions are met.
> See the "Preconditions checklist" at the end of this document.

---

## Project name

**SB-Trace** (stepback trace format and replay runtime)

## One-line description

An open wire format and multi-language runtime for recording, replaying, and
auditing AI-agent execution traces.

## Proposing organisation

stepback maintainers (open; hosted pending LF acceptance)

## License

Apache-2.0

---

## 1. Problem statement

AI agents — pipelines of LLM calls, tool invocations, routing decisions, and
policy checks — are increasingly deployed in regulated industries where
operators must answer:

- *What exactly happened in run X?*  (audit trail)
- *What would have happened if we changed step k?*  (counterfactual replay)
- *Was the production trace tampered with after recording?*  (tamper evidence)
- *Can a third-party regulator verify the trace independently?*  (multi-language
  interop and open format)

No neutral open standard currently exists.  Vendor-proprietary trace formats
lock audit data to a single SDK and cannot be independently verified.

## 2. Proposed solution

SB-Trace defines:

1. **A wire format** — append-only, HMAC-chained, Ed25519-signed, canonical-JSON
   frames in a `.sb` file.  Tamper-evident by construction.  Formally specified
   in `spec/rfcs/0001-sbtrace-core.md` and `spec/sbtrace-v1.md`.

2. **A dirty-set replay algorithm** — given a substitution on step k, recompute
   only the steps whose inputs transitively depend on k.  Reduces LLM-call cost
   from O(N) to O(|dirty set|) per counterfactual query.  Formally verified
   in Lean 4 (`proofs/lean/`) and TLA+ (`proofs/tla/`).

3. **Multi-language implementations** — reference Python, Rust (`stepback-core`),
   TypeScript, Go, JVM, .NET, WASM, and a proxy — all reading the same bytes
   and passing the same frozen-fixture conformance test corpus.

4. **OTel integration** — `agent.step.*` semantic conventions proposed to the
   OpenTelemetry GenAI SIG (RFC 0006), enabling stepback traces to coexist with
   standard observability infrastructure.

## 3. Why the Linux Foundation?

- **Neutral governance**: a single-company project cannot credibly position
  itself as a cross-industry audit standard.  LF provides the neutral home
  needed for regulators, model providers, and enterprises to adopt the format.

- **Legal clarity**: LF's IP policy and CLA process give enterprises the legal
  certainty needed to embed the format in compliance workflows.

- **Ecosystem reach**: proximity to OTel, CNCF, and other LF projects accelerates
  the OTel `agent.step.*` standardisation track.

## 4. Alignment with Linux Foundation AI & Data (LFAI)

SB-Trace falls naturally within the
[Linux Foundation AI & Data](https://lfaidata.foundation/) umbrella:

- **Auditability and transparency** — core LFAI pillars.
- **Cross-vendor interop** — explicit non-goal of any single vendor's trace SDK.
- **Open governance for AI tooling** — SB-Trace targets the same audience as
  ONNX (model interop) and MLflow (experiment tracking), but for *runtime audit
  trails* rather than model artefacts.

The LFAI Sandbox stage requires: open-source code, a contributing community,
a charter, and a governance document.  See §7 (Governance) and §8 (Checklist).

## 5. Scope

**In scope for v1 (proposed LF contribution):**

- SB-Trace wire format specification (RFCs 0001–0006)
- Python reference implementation (`stepback` package)
- Rust core library (`stepback-core`)
- TypeScript, Go, JVM, .NET, WASM bindings
- HTTP/gRPC proxy (`sb proxy`)
- Frozen conformance fixture corpus
- OTel `agent.step.*` semantic conventions draft

**Out of scope for initial contribution:**

- LLM provider shims (separate package, stepback-shims)
- Dirty-set replay engine (separate package, stepback-replay)
- Commercial services built on top of SB-Trace

## 6. Technical steering committee (proposed)

| Role | Seat | Criteria |
|---|---|---|
| Chair | 1 elected seat | Elected by TSC annually |
| Maintainers | Up to 5 seats | ≥ 10 merged PRs, nominated by Chair |
| End-user representatives | 2 seats | Organisations running SB-Trace in production |
| Vendor representatives | 2 seats | Organisations with an LF member sponsorship |

Decisions by lazy consensus; disputes by simple majority of TSC.

## 7. Governance

The initial governance model follows the
[CNCF Minimal Viable Governance](https://contribute.cncf.io/maintainers/governance/mvg/)
template:

- `GOVERNANCE.md` — TSC composition, voting, election cadence
- `CONTRIBUTING.md` — PR workflow, DCO, coding standards
- `CODE_OF_CONDUCT.md` — Contributor Covenant 2.1
- `SECURITY.md` — vulnerability disclosure (already present)
- `CHARTER.md` — project scope, IP policy, LF relationship

## 8. Preconditions checklist

The following conditions must all be met before submitting this proposal:

- [ ] At least two organisations running SB-Trace in production at > 10 k
      traces/day each (demonstrates real-world demand beyond the originating
      organisation).
- [ ] At least three independent maintainers with commit rights from
      different organisations.
- [ ] `GOVERNANCE.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and
      `CHARTER.md` in the repository root.
- [ ] CI green on all supported platforms (Python 3.10–3.14, Linux/macOS/Windows).
- [ ] Published PyPI releases with signed provenance (SLSA level 2 or above).
- [ ] OTel `agent.step.*` conventions at least at Experimental status upstream.
- [ ] Legal review of the Apache-2.0 contribution agreement.

---

*Draft prepared by the stepback maintainers, 2026-05-12.*
*This document is an internal planning artifact; no formal submission has been made.*
