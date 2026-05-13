# SB-Trace RFCs

This directory contains the Request-for-Comments documents that formally
specify the SB-Trace wire format and associated protocols.  Each RFC is
self-contained and cross-references the others.

| RFC | Title | Status |
|---|---|---|
| [0001](0001-sbtrace-core.md) | SB-Trace Core Wire Format | Draft |
| [0002](0002-canonicalization.md) | Canonical JSON Encoding | Draft |
| [0003](0003-dirty-set.md) | Dirty-Set Replay Semantics | Draft |
| [0004](0004-attestation-packs.md) | Attestation Packs | Draft |
| [0005](0005-importer-lossiness.md) | Importer / Exporter Lossiness Reporting | Draft |
| [0006](0006-otel-agent-step.md) | OTel `agent.step.*` Semantic Conventions | Draft |

## Reading order

New contributors should read the RFCs in numerical order.  RFC 0001 is the
foundation; RFCs 0002–0006 build on it.

## Relationship to existing documentation

These RFCs are the *external-facing* specification documents.  They are
designed for upstream submission to standards bodies (OpenTelemetry,
potential IETF/W3C) and for multi-language implementers who need a
language-agnostic spec.

The *internal* implementation guides are in `docs/`:

- `docs/canonicalization.md` — reference implementation commentary for RFC 0002
- `docs/dirty-set.md` — reference implementation commentary for RFC 0003
- `docs/dirty-set-soundness.md` — paper proof for RFC 0003 §4
- `spec/sbtrace-v1.md` — byte-level layout detail for RFC 0001

## RFC process

1. Propose a new RFC by opening a PR with a new `NNNN-short-title.md`
   file following the existing template.
2. RFCs start in `Status: Draft` and are discussed in the PR.
3. Once two independent implementations pass the conformance tests,
   the RFC is promoted to `Status: Stable`.
4. Breaking changes require a new RFC (not editing an existing Stable
   RFC).
