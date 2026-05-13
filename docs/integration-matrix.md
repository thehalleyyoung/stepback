# stepback Public Integration Matrix

This document is the **authoritative reference** for every supported source
system, framework, output format, and exporter in the stepback ecosystem.
Entries are labelled **Stable** / **Experimental** / **Preview** following
the [install guide criteria](INSTALL.md#stability-labels).

---

## Contents

1. [LLM provider shims](#llm-provider-shims)
2. [Agent framework recorders](#agent-framework-recorders)
3. [MCP integration](#mcp-integration)
4. [Importers — foreign trace formats](#importers)
5. [Exporters — output formats](#exporters)
6. [Observability bridges](#observability-bridges)
7. [Provenance & supply-chain](#provenance)
8. [Ecosystem integrations](#ecosystem-integrations)
9. [Bindings matrix](#bindings)
10. [Compatibility matrix — writer × reader](#compat-matrix)

---

## LLM provider shims <a id="llm-provider-shims"></a>

| Provider | Wrap function | Async | Streaming | Status | Notes |
|---|---|---|---|---|---|
| **OpenAI Chat** | `wrap_openai` | `wrap_openai_async` | ✓ | Stable | `gpt-4o`, `o1`, `o3` |
| **OpenAI Responses** | `wrap_openai_responses` | — | ✓ | Stable | New Responses API |
| **Azure OpenAI** | `wrap_azure_openai` | — | ✓ | Stable | Deployment-name model IDs |
| **Anthropic** | `wrap_anthropic` | `wrap_anthropic_async` | ✓ | Stable | `claude-3-*` |
| **AWS Bedrock** | `wrap_bedrock` | — | — | Stable | Converse API |
| **Google Gemini** | `wrap_gemini` | — | — | Stable | `gemini-1.5-*` |
| **Vertex AI** | `wrap_vertex_model` | — | — | Stable | `google-vertex:…` IDs |
| **Cohere** | `wrap_cohere` | — | — | Stable | `command-r-*` |
| **Mistral** | `wrap_mistral` | — | — | Stable | `mistral-large-*` |
| **Groq** | `wrap_groq` | — | — | Stable | OpenAI-compat wrapper |
| **Together AI** | `wrap_together` | — | — | Stable | OpenAI-compat wrapper |
| **Fireworks** | `wrap_fireworks` | — | — | Stable | OpenAI-compat wrapper |
| **Cerebras** | `wrap_cerebras` | — | — | Stable | OpenAI-compat wrapper |
| **NVIDIA NIM** | `wrap_nvidia_nim` | — | — | Stable | OpenAI-compat wrapper |
| **vLLM** | `wrap_vllm` | — | — | Stable | OpenAI-compat wrapper |
| **TGI** | `wrap_tgi` | — | — | Stable | OpenAI-compat wrapper |
| **llama.cpp** | `wrap_llamacpp` | — | — | Stable | OpenAI-compat wrapper |
| **Ollama** | `wrap_ollama` | — | — | Stable | OpenAI-compat wrapper |

All provider shims:
- Record `llm_call` steps with canonicalized `llm_request` / `llm_response`.
- Normalize token counts and cost into the `cost_usd` field (using the
  built-in price list at `stepback/pricing.py`).
- Duck-type the SDK — no hard dependency on any provider SDK.

---

## Agent framework recorders <a id="agent-framework-recorders"></a>

| Framework | Entry point | Transport | Status | Notes |
|---|---|---|---|---|
| **LangChain / LangGraph** | `StepbackCallbackHandler` | Callback | Stable | Hooks run/chain/tool events |
| **LlamaIndex** | `StepbackLlamaIndexHandler` | Event handler | Stable | Query, synthesis, retrieval |
| **DSPy** | `StepbackDspyCallback` | Callback | Stable | Module calls + LLM calls |
| **Haystack** | `StepbackHaystackTracer` | `Tracer` protocol | Stable | Component span mapping |
| **AutoGen** | `StepbackAutoGenLogger` | Agent event logger | Stable | `AgentEvent` to `tool_call` |
| **CrewAI** | `StepbackCrewAIObserver` | Observer pattern | Stable | Task/crew event recording |
| **Semantic Kernel** | `StepbackSKFilter` | Function filter | Stable | Prompt/function execution |
| **Strands** | `StepbackStrandsHook` | Hook API | Experimental | Early access |
| **Pydantic AI** | `StepbackPydanticHook` | Hook API | Experimental | Early access |

---

## MCP integration <a id="mcp-integration"></a>

| Mode | Entry point | Description | Status |
|---|---|---|---|
| **Client-side session wrap** | `wrap_mcp_session` | Wrap an existing MCP `ClientSession` to record `tools/call` invocations as `tool_call` steps. | Stable |
| **Replay executor** | `mcp_tool_executor` | Replay-side executor that routes dirty-step re-executions to the correct MCP server by qualified name. | Stable |
| **Recorder proxy** | `MCPRecorderProxy` (`stepback.integrations.mcp_proxy`) | Drop-in MCP session proxy that sits transparently between an MCP client and a real upstream server, recording all tool calls with full replay semantics. Supports `record_list_tools=True` to capture session initialization. | Stable |
| **Event log import** | `import_mcp_log` (`stepback.integrations.mcp_proxy`) | Convert an MCP server-side JSON event log (`{"events": [...]}`) into a `.sb` trace for offline replay, bisect, and substitution. | Stable |

### MCP proxy quick-start

```python
from stepback import record
from stepback.integrations.mcp_proxy import MCPRecorderProxy

real_session = MyMCPClient.connect("http://localhost:8090")

with record("mcp_run.sb") as rec:
    session = MCPRecorderProxy(real_session, rec, server_name="filesystem")
    result = session.call_tool("read_file", {"path": "/tmp/hello.txt"})
    # → recorded as tool_call step "filesystem:read_file"
```

---

## Importers — foreign trace formats <a id="importers"></a>

All importers produce a `.sb` trace that can be replayed, branched,
substituted, attested, and diffed exactly like a natively-recorded trace.

| Source format | Function | `import_trace` alias | Status | Notes |
|---|---|---|---|---|
| OpenAI chat log | `import_openai_chat_log` | `openai_chat_log` | Stable | JSON array of chat calls |
| LangSmith JSONL | `import_langsmith_jsonl` | `langsmith` | Stable | Run export |
| OpenInference spans | `import_openinference_spans` | `openinference` | Stable | OTel-based spans |
| OTel spans | `import_otel_spans` | `otel` | Stable | Stable Gen AI + `agent.step.*` |
| Arize Phoenix | `import_phoenix_spans` | `phoenix` | Stable | Phoenix-envelope JSON |
| Helicone log | `import_helicone_log` | `helicone` | Stable | LLM call log |
| Langfuse export | `import_langfuse_export` | `langfuse` | Stable | GENERATION observations |
| Datadog APM | `import_datadog_apm` | `datadog` | Stable | `dd-trace` AI spans |
| CycloneDX-AI SBOM | `import_cyclonedx_ai` | `cyclonedx` | Stable | BOM → step per component |
| MCP event log | `import_mcp_log` | — | Stable | `{"events": [...]}` JSON |
| stepback native JSON | `import_native_json` | `json` | Stable | Lossless round-trip |

---

## Exporters — output formats <a id="exporters"></a>

| Output format | Function | `export_trace` alias | Status | Notes |
|---|---|---|---|---|
| OTel spans (JSON) | `export_otel_spans` | `otel` | Stable | Round-trip with `import_otel_spans` |
| LangSmith JSONL | `export_langsmith` | `langsmith` | Stable | LangSmith run format |
| CycloneDX-AI SBOM | `export_cyclonedx_ai` | `cyclonedx_ai` | Stable | JSON, spec v1.6 |
| HTML interactive viewer | `export_html_view` | `html` | Stable | Self-contained single-file |
| stepback native JSON | `export_native_json` | `native_json` | Stable | Lossless; round-trips with `import_native_json` |
| SLSA / in-toto DSSE | `sign_provenance` | — | Stable | Via `stepback.provenance` |
| CycloneDX-AI + SLSA | `export_cyclonedx_ai` + `sign_provenance` | — | Stable | Combine both |
| SLSA in-toto examples | `scripts/slsa_example.py` | — | Stable | End-to-end runnable example |
| in-toto link | `pack_provenance` | — | Stable | Attestation pack provenance |

---

## Observability bridges <a id="observability-bridges"></a>

| Bridge | Class / function | Protocol | Status | Notes |
|---|---|---|---|---|
| **OTel/OTLP HTTP** | `OtelBridge` + `OtlpHttpExporter` | HTTP POST `/v1/traces` | Stable | No SDK dependency; stdlib-only |
| **Dry-run / test** | `OtelBridge` + `DryRunCollector` | In-process | Stable | Useful for unit tests and CI |
| **Static OTel JSON** | `export_otel_spans` | File | Stable | For batch pipelines |
| **ClickHouse** | `stepback.analytics.ClickHouseExporter` | TCP | Experimental | Requires `clickhouse-driver` |

### OTel bridge quick-start

```python
from stepback.otel_bridge import OtelBridge, OtlpHttpExporter

bridge = OtelBridge(
    exporter=OtlpHttpExporter("http://otel-collector:4318"),
    service_name="my-agent",
)
result = bridge.export_trace("run.sb")
print(f"Shipped {result.span_count} spans in {result.batch_count} batches")
```

The bridge uses :func:`stepback.exporters.export_otel_spans` internally and
ships spans in configurable batches (default 100).  See
:class:`stepback.otel_bridge.OtlpHttpExporter` for retry and timeout options.

---

## Provenance & supply-chain <a id="provenance"></a>

| Artifact type | Function | Output | Status |
|---|---|---|---|
| Trace provenance | `trace_provenance` + `sign_provenance` | DSSE JSON | Stable |
| Benchmark submission | `benchmark_provenance` + `sign_provenance` | DSSE JSON | Stable |
| Attestation pack | `pack_provenance` + `sign_provenance` | DSSE JSON | Stable |
| Verify signature | `verify_provenance_signature` | Exception on fail | Stable |
| CycloneDX-AI SBOM | `export_cyclonedx_ai` | JSON (BOM v1.6) | Stable |
| SLSA + CycloneDX | combine above two | — | Stable |
| Key rotation | `rotate_attestation_key` | `.pack` + new header | Stable |
| Threshold signing | `ThresholdSigner` | M-of-N DSSE | Stable |
| Post-quantum (ML-DSA) | `PQSigner` | DSSE + PQ sig | Experimental |
| Hardware-backed keys | `PKCS11Provider`, `YubiHSMProvider`, cloud KMS | — | Experimental |

---

## Ecosystem integrations <a id="ecosystem-integrations"></a>

| System | Module | Key classes / functions | Status |
|---|---|---|---|
| **toolwarden** | `stepback.integrations.toolwarden` | `WardenRecorderShim`, `import_toolwarden_audit` | Stable |
| **flowwarden** | `stepback.integrations.flowwarden` | `import_flowwarden_runlog`, `FlowWardenRecorderAttachment` | Stable |
| **ragdoctor** | `stepback.integrations.ragdoctor` | `RagDoctorShim`, `record_rag_query`, `import_ragdoctor_trace` | Stable |
| **MCP (proxy)** | `stepback.integrations.mcp_proxy` | `MCPRecorderProxy`, `import_mcp_log` | Stable |

---

## Bindings matrix <a id="bindings"></a>

| Language | Package | Read | Verify | Write | Status |
|---|---|---|---|---|---|
| Python | `stepback` (this repo) | ✓ | ✓ | ✓ | Stable |
| Python (Rust-backed) | `stepback-core` (PyO3) | ✓ | ✓ | — | Experimental |
| Rust | `stepback-core` (crates) | ✓ | ✓ | — | Experimental |
| TypeScript / JS (ESM) | `@stepback/core` (WASM) | ✓ | ✓ | — | Experimental |
| Go | `stepback-go` | ✓ | ✓ | — | Preview |
| Java / Kotlin | `stepback-jvm` (Gradle) | ✓ | ✓ | — | Preview |
| .NET | `Stepback.Sb` | ✓ | ✓ | — | Preview |
| Browser (WASM) | `stepback_wasm.js` | ✓ | ✓ | — | Experimental |

See [INSTALL.md](INSTALL.md) for per-language install paths.

---

## Compatibility matrix — writer × reader <a id="compat-matrix"></a>

A `.sb` trace written by one implementation can be read and verified by any
conformant implementation.  The table below shows which reader/writer
combinations have been exercised by the test suite (Step 32:
`tests/test_python_rust_differential.py`).

| Writer | Python reader | Rust reader | JS reader |
|---|---|---|---|
| **Python** | ✓ tested | ✓ tested (corpus) | ✓ tested (corpus) |
| **Rust** | ✓ tested (corpus) | ✓ tested | — |

The conformance corpus lives at `stepback-core/fixtures/v1/` and is
catalogued by `manifest.json` (Step 41).  Any conformant implementation must:

1. **Accept** all fixtures under `fixtures/v1/good/`.
2. **Reject** all fixtures under `fixtures/v1/corrupt/` with the documented
   error kind from `manifest.json`.

Run `stepback spec test` to verify a reader against the corpus.

---

*This document is generated from the audit snapshot.  To add a new
integration, follow the [CONTRIBUTING.md](../CONTRIBUTING.md) guide,
add entries to the relevant section above, and update
`tests/test_integration_matrix.py`.*
