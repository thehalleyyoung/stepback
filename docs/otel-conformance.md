# OTel `agent.step.*` Conformance Status

This document tracks conformance with [RFC 0006](../spec/rfcs/0006-otel-agent-step.md)
(`agent.step.*` semantic conventions) across every stepback implementation.
Status was last updated 2026-05-12.

The upstream proposal is being submitted to the OpenTelemetry GenAI SIG for
review. Until the conventions reach at least *Experimental* status in the
upstream repo, all implementations should treat these attributes as **draft**
and subject to change.

---

## Status legend

| Symbol | Meaning |
|---|---|
| ✅ | Implemented and tested |
| 🟡 | Implemented; no `agent.step.*` conformance tests yet |
| ⬜ | Not implemented / not applicable |
| ❓ | Unknown / untested |

---

## Required attributes (RFC 0006 §2.1)

Every span produced by a conformant exporter MUST carry:

| Attribute | Description |
|---|---|
| `agent.step.id` | UUID4 step identifier |
| `agent.step.kind` | One of `LLM_CALL`, `TOOL_CALL`, `ROUTER`, `POLICY_CHECK`, `MCP_CALL`, `PARALLEL_BRANCH_OPEN`, `PARALLEL_BRANCH_JOIN`, `EXCEPTION` |
| `agent.step.name` | Human-readable step name |

## Optional attributes (RFC 0006 §2.2)

| Attribute | Description |
|---|---|
| `agent.step.parent_id` | Immediate parent step UUID4 |
| `agent.step.parent_ids` | Parent-ID array (PARALLEL_BRANCH_JOIN only) |
| `agent.step.trace_id` | SB-Trace run UUID4 |
| `agent.step.inputs_hash` | `blake2b:<hex>` canonical inputs hash |
| `agent.step.nondeterminism_hash` | `blake2b:<hex>` nondeterminism hash |

---

## Python (reference implementation)

**Package**: `stepback` (PyPI)
**Exporter**: `stepback.exporters.export_otel_spans`
**Importer**: `stepback.importers.import_otel_spans`

| Feature | Export | Import |
|---|---|---|
| `agent.step.id` | ✅ | ✅ (preserved as source `step_id` via `agent.step.id` attr) |
| `agent.step.kind` (all 8 RFC values) | ✅ | ✅ |
| `agent.step.name` | ✅ | ✅ |
| `agent.step.parent_id` | ✅ | ✅ (via `parent_span_id` + attribute) |
| `agent.step.parent_ids` (join) | ✅ | 🟡 (array parsed; full join reconstruction pending) |
| `agent.step.trace_id` | ✅ (opt-in `trace_id=` arg) | ⬜ (no SB-Trace run-id field) |
| `agent.step.inputs_hash` | ✅ | ✅ (stored as `source_agent_step_inputs_hash` extra) |
| `agent.step.nondeterminism_hash` | ✅ | ✅ (stored as `source_agent_step_nondeterminism_hash` extra) |
| `gen_ai.*` stable attrs (LLM_CALL) | ✅ | ✅ |
| `gen_ai.tool.*` attrs (TOOL_CALL) | ✅ | ✅ |
| `agent.step.policy.*` (POLICY_CHECK) | ✅ | ✅ |
| `agent.step.mcp.method` (MCP_CALL) | ✅ | ✅ |
| `agent.step.router.decision` (ROUTER) | ✅ | ✅ |
| Message span events (input) | ✅ system/user/assistant/tool | ✅ |
| Message span events (output `gen_ai.choice`) | ✅ | ✅ |
| Round-trip fidelity (kind, model, tool name, parent edges) | ✅ (tested) | ✅ (tested) |

**Conformance test files**: `tests/test_exporters.py` (`test_export_otel_spans_*`),
`tests/test_export_roundtrip.py`, `tests/test_importers.py`

---

## Rust (`stepback-core`)

**Crate**: `sb-format` in `stepback-core/`

The Rust crate implements `SbTraceReader` and `SbTraceVerifier` for the binary
`.sb` format but does **not** yet include an OTel exporter or importer.

| Feature | Status |
|---|---|
| Read `.sb` trace, verify HMAC chain | ✅ |
| `agent.step.*` OTel export | ⬜ (not implemented) |
| `agent.step.*` OTel import | ⬜ (not implemented) |

**Path to conformance**: Implement `export_otel_spans` in Rust mirroring RFC 0006
and add conformance tests against the same fixture set used by the Python impl.

---

## TypeScript

**Package**: `stepback` (npm, `bindings/typescript/`)

The TypeScript bindings provide a reader and verifier for `.sb` files.  An
OTel exporter stub exists but does not yet emit `agent.step.*` attributes.

| Feature | Status |
|---|---|
| Read `.sb` trace | ✅ |
| `gen_ai.*` stable attrs export | 🟡 (partial, no conformance tests) |
| `agent.step.*` RFC 0006 export | ⬜ (not implemented) |
| `agent.step.*` RFC 0006 import | ⬜ (not implemented) |

**Path to conformance**: Align `exportOtelSpans()` with the Python reference
exporter and add Jest/Vitest tests using the frozen v1 fixtures.

---

## Go (`stepback-go`)

**Module**: `github.com/stepback/stepback-go` (`bindings/go/`)

The Go module provides `Reader`, `Frame`, and `Verify` APIs for `.sb` files.
No OTel exporter/importer is present.

| Feature | Status |
|---|---|
| Read `.sb` trace | ✅ |
| `agent.step.*` RFC 0006 export | ⬜ (not implemented) |
| `agent.step.*` RFC 0006 import | ⬜ (not implemented) |

**Path to conformance**: Add `ExportOtelSpans(steps []Frame, w io.Writer) error`
using the OTLP protobuf or JSON wire format, guided by RFC 0006 §2.

---

## JVM (Java / Kotlin)

**Artifact**: `io.stepback:stepback-jvm` (`bindings/jvm/`)

The JVM bindings provide a `SbTraceReader` and basic verification.  No OTel
export is implemented.

| Feature | Status |
|---|---|
| Read `.sb` trace | ✅ |
| `agent.step.*` RFC 0006 export | ⬜ (not implemented) |
| `agent.step.*` RFC 0006 import | ⬜ (not implemented) |

**Path to conformance**: Use the OTel Java SDK `OpenTelemetry` API to emit
spans from `SbTraceReader`, add JUnit 5 tests against frozen fixtures.

---

## .NET (C#)

**Package**: `Stepback` (NuGet, `bindings/dotnet/`)

The .NET bindings provide `SbTraceReader` and `SbVerifier`.  No OTel export
is implemented.

| Feature | Status |
|---|---|
| Read `.sb` trace | ✅ |
| `agent.step.*` RFC 0006 export | ⬜ (not implemented) |
| `agent.step.*` RFC 0006 import | ⬜ (not implemented) |

**Path to conformance**: Use the `System.Diagnostics.Activity` / OTel .NET SDK
to emit spans, add NUnit/xUnit tests against frozen fixtures.

---

## Proxy (`stepback-proxy`)

**Image**: `ghcr.io/stepback/stepback-proxy` (`docker/`)

The proxy accepts HTTP and gRPC requests and replays traces.  It re-uses the
Python `export_otel_spans` / `import_otel_spans` functions via the proxy's
Python runtime when the `--otel-export` flag is passed.

| Feature | Status |
|---|---|
| OTel export via Python backend | 🟡 (delegates to Python; no proxy-specific conformance tests) |
| OTLP gRPC push (exporter to collector) | ⬜ (not yet wired) |

**Path to conformance**: Wire an OTLP gRPC exporter to push spans directly
from the proxy to an OTel collector, add a docker-compose smoke test.

---

## WASM

**Package**: `@stepback/wasm` (`wasm/`)

The WASM build verifies and summarizes traces in-browser.  No OTel export is
implemented.

| Feature | Status |
|---|---|
| Verify `.sb` trace | ✅ |
| `agent.step.*` RFC 0006 export | ⬜ (not implemented) |

**Path to conformance**: Expose a `exportOtelSpans(bytes: Uint8Array): string`
function that returns OTLP JSON, add Playwright/Puppeteer smoke tests.

---

## Summary table

| Implementation | Export | Import | Round-trip |
|---|---|---|---|
| Python (reference) | ✅ | ✅ | ✅ tested |
| Rust | ⬜ | ⬜ | ⬜ |
| TypeScript | ⬜ | ⬜ | ⬜ |
| Go | ⬜ | ⬜ | ⬜ |
| JVM | ⬜ | ⬜ | ⬜ |
| .NET | ⬜ | ⬜ | ⬜ |
| Proxy | 🟡 (via Python) | 🟡 (via Python) | 🟡 |
| WASM | ⬜ | ⬜ | ⬜ |

---

## Upstream submission status

A PR proposing `agent.step.*` as *Experimental* attributes has been prepared
for submission to `opentelemetry/semantic-conventions`.  The draft document
(RFC 0006) is at `spec/rfcs/0006-otel-agent-step.md`.

Once the upstream PR is opened the table below will be updated:

| Milestone | Status |
|---|---|
| RFC 0006 published internally | ✅ 2026-05-12 |
| PR opened against `opentelemetry/semantic-conventions` | ⬜ pending |
| GenAI SIG initial review | ⬜ pending |
| *Experimental* status granted | ⬜ pending |
| Second independent implementation (Rust or TypeScript) | ⬜ pending |
| *Stable* status granted | ⬜ pending |
