# Changelog

All notable changes to **stepback** are documented here.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- Lean 4 mechanized soundness proof (`proofs/lean/Stepback/Soundness.lean`),
  discharging Step 56 of the OSS-readiness plan.

---

## [0.1.0] — 2026-05-01

### Added

- Initial public release of `stepback`.
- `Trace`, `RecordedStep`, `TraceHeader`, `Receipt`, and `StepKind` typed
  dataclasses in `stepback/__init__.py`.
- Dirty-set replay engine (`stepback/replay.py`, `stepback/divergence.py`).
- Canonical JSON encoder with RFC 8785-style key ordering (`stepback/canonical.py`).
- SB-Trace v1 wire format reader/writer (`stepback/importers.py`,
  `stepback/exporters.py`).
- Provider shims for OpenAI, Anthropic, Bedrock, Gemini, Azure OpenAI,
  Cohere, Mistral, Together, Fireworks, Groq, Cerebras, NVIDIA NIM, vLLM,
  TGI, llama.cpp, and Ollama.
- Streaming and async recorder support.
- `stepback.testing` module with public deterministic fixture agents.
- `stepback.spec.SBTraceSpec` versioned spec loader and validator.
- `stepback-proxy` HTTP/gRPC replay sidecar.
- WASM build for in-browser trace verification.
- PyO3 Python bindings to the Rust `sb-format` core library.
- TypeScript, Go, JVM, and .NET bindings.
- Merkle summary frame for end-of-trace attestation.
- Distributed dirty-set computation over a worker pool.
- ClickHouse dirty-set summary tables.
- Minimization engine: multi-objective ddmin, incremental bisect, Shapley
  attribution, branch minimization, and HTML reports.
- Time-travel web debugger for step-level replay inspection.
- Full conformance test suite and `stepback spec test` CLI command.
- Soak, fuzz, and property-based tests.

### Deprecation policy

No public API was deprecated in this release.  Future deprecations follow
the policy in [`docs/deprecation.md`](docs/deprecation.md): every removal
requires at least one minor release with a live `DeprecationWarning` naming
the replacement and the planned removal release.

---

[Unreleased]: https://github.com/stepback-io/stepback/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/stepback-io/stepback/releases/tag/v0.1.0
