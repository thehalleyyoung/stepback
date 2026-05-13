# Related Work

This document surveys prior systems, tools, and research most relevant to the
stepback design and positions the contribution along five comparison axes.

Step 147 of [`100_STEPS.md`](100_STEPS.md) requested this document.

---

## 1. Comparison axes

| Axis | stepback contribution |
|------|-----------------------|
| [Observability vs. replay](#21-llm-observability-tools) | Causal replay of recorded agent runs, not just structured logging. |
| [Trace capture vs. time-travel](#22-time-travel-debuggers-and-replay-systems) | Content-addressed, signed `.sb` traces with semantic-level step caching. |
| [Build-cache invalidation vs. dirty-set replay](#23-content-addressed-caching-and-build-systems) | Dirty-set propagation over agent step DAGs; soundness proven under collision-free hash assumption. |
| [Delta debugging vs. causal substitution minimization](#24-delta-debugging-and-fault-localization) | Minimization extends ddmin to stochastic, expensive, non-freely-droppable steps. |
| [Audit logs vs. verifiable incident evidence](#25-cryptographic-audit-logs-and-provenance) | HMAC-chained, Ed25519-signed frames; formal tamper-evidence proof in TLA+. |

A differentiation table comparing stepback to each closely related system appears
at the end of each subsection.

---

## 2. Related systems

### 2.1 LLM observability tools

Modern agent observability tools offer structured logging, trace visualization, and
limited single-call editing:

| System | Description | Key difference from stepback |
|--------|-------------|-------------------------------|
| [LangSmith](https://smith.langchain.com/) | Trace visualization and simple run editing for LangChain agents. | No step-level dirty-set propagation; editing one step re-runs the full agent. |
| [Arize Phoenix](https://phoenix.arize.com/) | OpenInference-based span capture with embeddings analysis. | Observability focus; no signed traces, no counterfactual replay. |
| [LangFuse](https://langfuse.com/) | Open-source LLM tracing and scoring. | Score post-hoc; no forward-propagating cache reuse after substitution. |
| [Helicone](https://helicone.ai/) | LLM proxy with cost tracking, rate limiting, and caching. | Request-level caching; no step-DAG semantics or causal invalidation. |
| [PromptLayer](https://promptlayer.com/) | Prompt version tracking and analytics. | Prompt versioning; no multi-step trace replay. |
| [Datadog APM](https://www.datadoghq.com/product/apm/) | Distributed tracing for production services. | Distributed trace visualization; no semantic step caching or minimization. |
| [Weights & Biases Weave](https://wandb.ai/site/weave) | Experiment tracking and trace capture for ML runs. | Experiment tracking; no cryptographic chaining or counterfactual replay. |

**Differentiation:** All observability tools answer "what happened." stepback additionally
answers "what would have happened if step k had a different output," by computing exactly
which downstream steps must be re-executed (the dirty set) and serving the rest from a
content-addressed step cache.

### 2.2 Time-travel debuggers and replay systems

Time-travel debugging records execution history and replays it deterministically:

| System | Description | Key difference from stepback |
|--------|-------------|-------------------------------|
| [rr (Mozilla)](https://rr-project.org/) | Hardware-assisted record-replay at the kernel syscall level (Linux). | Binary-level determinism for C/C++ processes; no semantic notion of "agent step" or LLM call caching. |
| [Pernosco](https://pernos.co/) | rr-based time-travel debugging with structured queries and data flow. | C/C++ binary debugging; not applicable to Python agent steps or LLM API calls. |
| [UDB (Undo)](https://undo.io/) | Enterprise reverse debugging for C/C++/Python. | General-purpose reverse execution; does not model content-addressed step caches. |
| [Microsoft TTD](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/time-travel-debugging-overview) | Time Travel Debugging for Windows processes. | Windows process replay; no support for non-deterministic LLM calls. |
| [Chronon (Twitter)](https://engineering.twitter.com/en/blog/2020/12/chronon-mlplatform) | Feature computation with time-travel consistency guarantees. | ML feature stores; not agent trace replay. |

**Differentiation:** rr/Pernosco record at the operating-system level and replay
deterministically because the OS environment is deterministic. LLM APIs are natively
non-deterministic; stepback side-steps this by recording outputs at the agent-step
boundary, making each recorded step a cached pure function of its canonical input hash.

### 2.3 Content-addressed caching and build systems

Build systems that recompute only the changed transitive closure of a dependency graph
are the closest structural analogue:

| System | Description | Key difference from stepback |
|--------|-------------|-------------------------------|
| [Bazel](https://bazel.build/) | Hermetic build system with content-addressed action cache. | Build actions are deterministic; LLM steps are stochastic. Dirty-set propagation is the same forward-invalidation pattern but stepback applies it to non-deterministic outputs. |
| [Buck2 (Meta)](https://buck2.build/) | Distributed build system with content-addressed remote execution. | Same forward-invalidation analogy as Bazel. |
| [Nix](https://nixos.org/) | Functional package manager; every derivation is a pure function. | Package builds, not agent runs. |
| [ccache](https://ccache.dev/) | Compilation cache keyed on preprocessed source. | Single-artifact caching; no DAG-level dirty-set propagation. |
| [Pants](https://www.pantsbuild.org/) | Python/Java/Go build system with remote caching. | Build graph invalidation, not agent step invalidation. |
| [Turborepo](https://turbo.build/) | Monorepo build cache for JS/TS. | Frontend build pipeline caching. |

**Differentiation:** Build systems assume actions are deterministic pure functions; their
cache keys are exact cryptographic hashes of inputs. stepback adds two extensions: (1) a
*nondeterminism hash* captures sampled values that are inputs to later steps, enabling
cache reuse when the sampled value is unchanged; (2) the dirty-set propagation rule
handles parallel branches by dirtying joins only when consumed branch outputs change,
preserving clean siblings — a pattern without a direct analogue in single-rooted build
DAGs.

### 2.4 Delta debugging and fault localization

Delta debugging and fault localization find the minimal cause of a test failure:

| System | Description | Key difference from stepback |
|--------|-------------|-------------------------------|
| [ddmin (Zeller 1999)](https://www.st.cs.uni-saarland.de/dd/) | Binary search over test inputs to find the 1-minimal failing subset. | Original ddmin assumes cheap, freely-droppable, deterministic test cases. Agent steps are expensive (LLM calls), stochastic, and have causal dependencies that make arbitrary dropping incorrect. |
| [HDD (Misherghi & Su 2006)](https://dl.acm.org/doi/10.1145/1134285.1134307) | Hierarchical delta debugging preserving tree-shaped inputs. | Tree-structured input minimization (AST, XML); not applicable to agent step DAGs. |
| [Picire](https://github.com/renatahodovan/picire) | Configurable delta debugging with parallel execution. | General-purpose ddmin; no agent step semantics. |
| [CReduce](https://embed.cs.utah.edu/creduce/) | Credit-based delta debugging for C programs. | C source minimization; not applicable to agent runs. |
| [Spectrum-Based Fault Localization (SBFL)](https://dl.acm.org/doi/10.1145/1375581.1375606) | Ranks statements by correlation with passing/failing tests (Ochiai, Tarantula). | Source-code coverage matrices; not applicable to recorded LLM traces. |
| [Statistical Debugging (Liblit et al.)](https://dl.acm.org/doi/10.1145/1005 823.1005852) | Predicate-based statistical fault localization. | Requires many execution samples; not applicable when each replay is expensive. |

**Differentiation:** `stepback/minimize.py` adapts ddmin to the agent-trace domain by:
(1) staging substitutions rather than dropping them, so expensive executor calls are
reused through the dirty-set cache; (2) tracking monotonicity violations from stochastic
predicates and applying confidence-interval guards; (3) supporting Shapley attribution
for joint-cause analysis beyond 1-minimal witnesses. See
[`docs/minimization-paper.md`](docs/minimization-paper.md).

### 2.5 HTTP/network cassette replay

Cassette-based replay records HTTP requests and replays them from disk:

| System | Description | Key difference from stepback |
|--------|-------------|-------------------------------|
| [VCR (Ruby)](https://github.com/vcr/vcr) / [pytest-recording](https://github.com/kiwicom/pytest-recording) | Record HTTP interactions; replay them deterministically in tests. | Request/response byte cassettes; no multi-step agent DAG, no dirty-set propagation, no minimization. |
| [Betamax](https://github.com/betamaxpy/betamax) | Python HTTP cassette library for requests. | Same as VCR. |
| [Polly.JS](https://netflix.github.io/pollyjs/) | Record, replay, and stub HTTP traffic. | Same as VCR. |
| [OpenAI Evals](https://github.com/openai/evals) | Evaluation framework for LLM outputs. | Evaluates response quality; no trace replay, caching, or minimization. |

**Differentiation:** Cassette tools replay at the HTTP byte level; they cannot replace
one tool result mid-trace, observe what the LLM would generate in response, and cache
the unaffected downstream suffix. stepback operates at the semantic agent-step level.

### 2.6 Agent evaluation and testing frameworks

| System | Description | Key difference from stepback |
|--------|-------------|-------------------------------|
| [PromptFoo](https://www.promptfoo.dev/) | LLM prompt evaluation and red-teaming. | Prompt evaluation harness; no multi-step trace replay or dirty-set caching. |
| [LangChain Unit Tests](https://python.langchain.com/docs/guides/testing/) | Mock LLM class for unit testing. | Mock-based; does not record real runs or derive cache reuse from content hashes. |
| [AgentEval (Microsoft)](https://github.com/microsoft/autogen) | Multi-criteria LLM agent evaluation in AutoGen. | Metric evaluation; no causal minimization or counterfactual replay. |
| [Inspect-AI (UK AISI)](https://ukgovernment.github.io/inspect_ai/) | Structured evaluation framework for AI safety research. | Task-level evaluation scaffolding; no step-level trace caching. |
| [TruLens](https://github.com/truera/trulens) | Feedback functions and tracing for LLM apps. | Feedback evaluation; no causal dirty-set propagation. |

### 2.7 Cryptographic audit logs and provenance

| System | Description | Key difference from stepback |
|--------|-------------|-------------------------------|
| [Certificate Transparency (RFC 6962)](https://www.rfc-editor.org/rfc/rfc6962) | Append-only Merkle log of issued X.509 certificates. | Focuses on public-key infrastructure; not applicable to agent trace integrity. |
| [Sigstore / Rekor](https://sigstore.dev/) | Transparency log for software artifact signing. | Artifact-signing transparency; the `.sb` attestation model draws inspiration from Rekor's append-only log but embeds chain within the trace file. |
| [SLSA](https://slsa.dev/) | Supply-chain security framework for build provenance. | Build provenance; stepback adds `SLSA attestation` export for replay evidence packs. |
| [in-toto](https://in-toto.io/) | Framework for supply-chain metadata and link constraints. | Supply-chain link metadata; stepback generates in-toto link metadata for trace steps. |
| [CycloneDX-AI](https://cyclonedx.org/capabilities/ai-ml/) | AI model/data BOM specification. | Model inventory; stepback exports CycloneDX-AI for model references in traces. |
| [OpenTelemetry](https://opentelemetry.io/) | Observability framework: traces, metrics, logs. | General observability; stepback aligns with OTel `gen_ai.*` semantic conventions and proposes `agent.step.*` extensions. |

**Differentiation:** The `.sb` HMAC-SHA256 + Ed25519 chain proves per-frame tamper
evidence and truncation detection; a formal TLA+ model of the chain is in
[`proofs/tla/SBHMACChain.tla`](proofs/tla/SBHMACChain.tla). The Merkle summary frame
enables compact attestation packs. What the scheme does *not* prove is documented in
[`SECURITY.md`](SECURITY.md): content correctness, recorder identity, lying recorder,
or semantic validity of the recorded steps.

### 2.8 Formal verification of distributed protocols

| System / work | Description | Relationship to stepback |
|---------------|-------------|--------------------------|
| [TLA+ (Lamport)](https://lamport.azurewebsites.net/tla/tla.html) | Temporal Logic of Actions for distributed protocol verification. | `proofs/tla/SBHMACChain.tla` models the HMAC chain writer, adversary, and three invariants. |
| [Lean 4 / Mathlib](https://leanprover.github.io/) | Interactive theorem prover for formal mathematics. | `proofs/lean/` mechanizes the dirty-set soundness theorem (Step 56). |
| [Coq](https://coq.inria.fr/) | Proof assistant for type-theoretic proofs. | Referenced in Step 56 proposal; Lean was chosen for the mechanization. |
| [Ivy](https://microsoft.github.io/ivy/) | Specification language and verifier for distributed protocols. | Not used; TLA+ covers the HMAC chain model. |

### 2.9 Agent frameworks and recorders

| Framework | Recorder support | Notes |
|-----------|-----------------|-------|
| [LangChain](https://python.langchain.com/) | `StepbackCallbackHandler` | `on_llm_start/end`, `on_tool_start/end`, `on_chain_start/end` hooks. |
| [LangGraph](https://github.com/langchain-ai/langgraph) | `StepbackCallbackHandler` | Node-level hooks via `on_node_start/end`. |
| [LlamaIndex](https://www.llamaindex.ai/) | `LlamaIndexCallbackHandler` | CBEventType-based. |
| [DSPy](https://dspy.ai/) | `DSPyCallbackHandler` | `on_lm_start/end` signature events. |
| [Haystack](https://haystack.deepset.ai/) | `HaystackTracer` | Context-manager `trace()` with span nesting. |
| [AutoGen (Microsoft)](https://github.com/microsoft/autogen) | `AutoGenEventHandler` | `on_llm_call/result`, IOStream interception. |
| [CrewAI](https://github.com/crewAIInc/crewAI) | `CrewAIStepRecorder` | `step_callback` integration. |
| [Semantic Kernel (Microsoft)](https://github.com/microsoft/semantic-kernel) | `SemanticKernelFilter` | `on_function_invocation` filter. |
| [Pydantic AI](https://ai.pydantic.dev/) | `PydanticAIInstrument` | `on_model_request/response` hooks. |
| [Inspect-AI](https://ukgovernment.github.io/inspect_ai/) | `InspectAIRecorder` | `on_model_call`, `on_tool_call` hooks. |
| [MCP](https://modelcontextprotocol.io/) | `wrap_mcp_session` | Tool-call level recording. |

---

## 3. Summary: where stepback sits

stepback occupies a gap between observability (what happened) and deterministic
time-travel debugging (replay at byte level):

```
                     ┌──────────────────────────────────────────┐
                     │         Semantic Replay Layer            │
                     │  (stepback: dirty-set, substitutions,    │
                     │   minimization, counterfactual sweeps)   │
                     └──────────────────────────────────────────┘
            ▲ stepback                              ▼ below stepback
 ┌─────────────────────┐                ┌───────────────────────────┐
 │  Observability:     │                │  Time-travel debugging:   │
 │  LangSmith, Phoenix,│                │  rr, Pernosco, UDB        │
 │  LangFuse, Datadog  │                │  (deterministic binaries) │
 └─────────────────────┘                └───────────────────────────┘
```

The core claim is that content-addressed step caching plus forward-propagating
dirty-set computation lets you ask "what if?" on a real recorded agent run in
O(|dirty_set|) LLM calls instead of O(N), without losing cryptographic evidence
of the original run.

---

## 4. References

Selected primary references for the related systems above. All URLs were valid at
the time of writing; external pages may change.

- Zeller, A. (1999). "Yesterday, my program worked. Today, it does not. Why?"
  *FSE 1999*. The original ddmin paper.
- Misherghi, G., & Su, Z. (2006). "HDD: Hierarchical delta debugging."
  *ICSE 2006*.
- Liblit, B., Naik, M., Zheng, A. X., Aiken, A., & Jordan, M. I. (2005).
  "Scalable statistical bug isolation." *PLDI 2005*.
- Jones, J. A., & Harrold, M. J. (2005). "Empirical evaluation of the
  Tarantula automatic fault-localization technique." *ASE 2005*.
- Lamport, L. (1994). "The temporal logic of actions." *ACM TOPLAS*.
- Bernstein, D. J., & Lange, T. (2017). "Montgomery curves and the Montgomery
  ladder." *Topics in Computational Number Theory*.
- RFC 6962: Certificate Transparency.
- SLSA specification: https://slsa.dev/spec/
- OpenTelemetry Semantic Conventions for GenAI:
  https://opentelemetry.io/docs/specs/semconv/gen-ai/
