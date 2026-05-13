# RFC 0006 — OTel `agent.step.*` Semantic Conventions

| Field | Value |
|---|---|
| RFC number | 0006 |
| Title | OpenTelemetry `agent.step.*` Semantic Conventions |
| Status | Draft (upstream proposal pending) |
| Supersedes | — |
| Created | 2026-05-12 |
| Authors | stepback maintainers |
| Reference impl | `stepback.exporters.export_otel_spans` (`stepback >= 0.1`) |
| OTel SIG | opentelemetry/semantic-conventions (GenAI SIG) |

---

## Abstract

This RFC proposes the `agent.step.*` attribute namespace for OpenTelemetry
semantic conventions, extending the stable `gen_ai.*` attributes to cover
multi-step AI-agent workflows.  A *step* is one unit of agent work: an LLM
call, a tool invocation, a routing decision, a policy check, or an MCP
call.  The attributes defined here enable correlated tracing across the
entire agent lifecycle and are designed to round-trip through SB-Trace
(RFC 0001) without information loss.

---

## 1. Motivation

The stable OTel Gen AI semantic conventions (as of v1.26.0) cover
individual LLM and embedding API calls (`gen_ai.operation.name`,
`gen_ai.request.model`, `gen_ai.usage.input_tokens`, etc.).  They do not
cover:

1. **Multi-step agents** — a single agent run generates dozens of
   inter-dependent LLM calls, tool calls, and routing decisions.  Without
   parent-step linkage, the causal graph is invisible.
2. **Non-LLM steps** — tool calls, routers, policy checks, and MCP calls
   are first-class citizens in modern agent frameworks but have no OTel
   representation.
3. **Replay metadata** — `inputs_hash` and `nondeterminism_hash` are
   needed for counterfactual replay; no existing convention captures them.

The `agent.step.*` namespace fills these gaps while remaining additive to
the existing `gen_ai.*` attributes.

---

## 2. Attribute definitions

### 2.1 Required on every agent span

| Attribute | Type | Description |
|---|---|---|
| `agent.step.kind` | string | Step kind: `LLM_CALL`, `TOOL_CALL`, `ROUTER`, `POLICY_CHECK`, `MCP_CALL`, `PARALLEL_BRANCH_OPEN`, `PARALLEL_BRANCH_JOIN`, `EXCEPTION` |
| `agent.step.name` | string | Human-readable name for the step (e.g. `"llm_call 1"`, `"search"`) |
| `agent.step.id` | string | UUID4 identifier for this step; equal to `step_id` in SB-Trace |

### 2.2 Optional agent-workflow attributes

| Attribute | Type | Description |
|---|---|---|
| `agent.step.parent_id` | string | `step_id` of the immediate parent step (omit for root steps) |
| `agent.step.parent_ids` | string[] | `step_id` list for `PARALLEL_BRANCH_JOIN` steps (multiple parents) |
| `agent.step.trace_id` | string | UUID4 of the containing SB-Trace; correlates all spans in one agent run |
| `agent.step.inputs_hash` | string | `blake2b:<hex>` of the canonical inputs (RFC 0002); enables cache-hit detection |
| `agent.step.nondeterminism_hash` | string | `blake2b:<hex>` of the nondeterminism record |

### 2.3 LLM_CALL steps

Use the stable `gen_ai.*` attributes alongside `agent.step.*`:

| Attribute | Type | Description |
|---|---|---|
| `gen_ai.operation.name` | string | `"chat"` |
| `gen_ai.request.model` | string | Model identifier (e.g. `"gpt-4o"`, `"claude-3-5-sonnet-20241022"`) |
| `gen_ai.request.temperature` | double | Sampling temperature (if set) |
| `gen_ai.request.seed` | int | Deterministic seed (if set) |
| `gen_ai.usage.input_tokens` | int | Prompt token count |
| `gen_ai.usage.output_tokens` | int | Completion token count |

Input messages are emitted as span events (`gen_ai.system.message`,
`gen_ai.user.message`, `gen_ai.assistant.message`, `gen_ai.tool.message`)
with `gen_ai.event.content` set to the canonical JSON of the message dict.
Output messages are emitted as `gen_ai.choice` events.

### 2.4 TOOL_CALL steps

| Attribute | Type | Description |
|---|---|---|
| `gen_ai.operation.name` | string | `"execute_tool"` |
| `gen_ai.tool.name` | string | Tool / function name |
| `gen_ai.tool.call.id` | string | Tool call ID if provided by the LLM |
| `gen_ai.tool.call.arguments` | string | JSON string of tool arguments |
| `gen_ai.tool.output` | string | JSON string of tool result |

### 2.5 ROUTER / POLICY_CHECK / MCP_CALL / PARALLEL_BRANCH_* steps

| Attribute | Type | Description |
|---|---|---|
| `gen_ai.operation.name` | string | `"execute_agent"` |
| `agent.step.router.decision` | string | (ROUTER only) The routing decision taken |
| `agent.step.policy.name` | string | (POLICY_CHECK only) Policy identifier |
| `agent.step.policy.result` | string | (POLICY_CHECK only) `"allow"` / `"deny"` / `"flag"` |
| `agent.step.mcp.method` | string | (MCP_CALL only) MCP method name |

---

## 3. Span structure

Each agent step maps to exactly one OTel span.  The span hierarchy mirrors
the SB-Trace parent-edge relation:

```
Trace span (root)
  └─ agent.step.kind=ROUTER span          ← router step
       ├─ agent.step.kind=LLM_CALL span   ← first llm call
       ├─ agent.step.kind=TOOL_CALL span  ← tool invocation
       └─ agent.step.kind=LLM_CALL span   ← second llm call
```

The root span for the entire agent run MAY be emitted with
`agent.step.kind = "AGENT_RUN"` (future extension) or as a plain
instrumentation scope span.

---

## 4. Round-trip with SB-Trace

The `export_otel_spans` / `import_otel_spans` pair in the stepback
reference implementation guarantees:

```
.sb → export_otel_spans → OTel JSON → import_otel_spans → .sb'
```

where `.sb'` has equal `step_count`, `kind_counts`, parent edges,
`gen_ai.request.model` values, and tool names.  The lossiness on the
import side is:

- `nondeterminism_hash`: synthesised as `canonical_hash({})` (no OTel
  equivalent).
- `cost_usd`: absent if not set in the original export (OTel has no
  standardised cost attribute yet).

---

## 5. Upstream submission plan

1. Open a PR against `opentelemetry/semantic-conventions` with the
   `agent.step.*` namespace definitions.
2. Request review from the GenAI SIG (weekly call) and the Agents
   working group.
3. Target *Experimental* status initially; promote to *Stable* once two
   independent implementations (stepback Python + one other SDK) pass the
   conformance tests in `stepback spec test`.
4. Keep `stepback.exporters.export_otel_spans` aligned with review
   feedback and publish the conformance status for Python, Rust,
   TypeScript, Go, JVM, .NET, proxy, and WASM in `docs/otel-conformance.md`.

---

## 6. Relationship to other RFCs

- RFC 0001 defines the SB-Trace step fields that map to these attributes.
- RFC 0002 defines `canonical_hash`, used for `agent.step.inputs_hash`.
- RFC 0005 §A (per-format lossiness) covers the OTel round-trip loss.

---

## 7. Relationship to existing OTel Gen AI conventions

The `agent.step.*` attributes are *additive*: they MUST be used alongside
(not instead of) the stable `gen_ai.*` attributes on the same span.
Collectors that do not understand `agent.step.*` will still see standard
`gen_ai.*` attributes and can process them normally.

---

## Appendix A — Mapping table (SB-Trace ↔ OTel)

| SB-Trace field | OTel attribute | Notes |
|---|---|---|
| `step_id` | `agent.step.id` | |
| `step_kind` | `agent.step.kind` | Uppercased; see §2.1 |
| `parent_step_id` | `agent.step.parent_id` | Omit if null |
| `parent_step_ids` | `agent.step.parent_ids` | PARALLEL_BRANCH_JOIN only |
| `inputs.model` | `gen_ai.request.model` | LLM_CALL only |
| `inputs.temperature` | `gen_ai.request.temperature` | LLM_CALL only |
| `inputs.seed` | `gen_ai.request.seed` | LLM_CALL only |
| `outputs.usage.prompt_tokens` | `gen_ai.usage.input_tokens` | LLM_CALL only |
| `outputs.usage.completion_tokens` | `gen_ai.usage.output_tokens` | LLM_CALL only |
| `inputs_hash` | `agent.step.inputs_hash` | |
| `nondeterminism_hash` | `agent.step.nondeterminism_hash` | |
| messages (input) | span events `gen_ai.{role}.message` | Content in `gen_ai.event.content` |
| messages (output) | span events `gen_ai.choice` | Content in `gen_ai.event.content` |

---

## Appendix B — Changelog

| Date | Author | Change |
|---|---|---|
| 2026-05-12 | stepback maintainers | Initial draft |
