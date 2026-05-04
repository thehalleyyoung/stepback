# stepback

**The open reversible debugger and counterfactual evaluation substrate for LLM agents.** stepback records a real agent run as a content-addressed, signed `.sb` trace, then lets you replay it deterministically, branch at any step, substitute prompts / tool outputs / policies / models, and re-execute *only the steps affected by the substitution*. Every other step is served from a per-step content-addressed cache.

This is not meant to be just a Python library. The goal is the runtime layer the field adopts: the rr / Pernosco / DDT for the agent era. The `.sb` trace is the standardization point: an open SB-Trace specification on a public RFC track, independent implementations, conformance fixtures, and a versioning policy strict enough for production incident records and research benchmarks.

The core technical claim remains: **counterfactual debugging of an N-step agent trace costs O(dirty_set) LLM calls instead of O(N)**. `stepback/divergence.py` implements dirty-set propagation, `stepback/replay.py` implements replay, `stepback/minimize.py` implements trace minimization, `stepback/sweep.py` amortizes parameter sweeps, and `trace_writer.py` / `attestation.py` provide signed trace evidence.

License: **Apache-2.0.** The target home is neutral infrastructure: SB-Trace in the OpenTelemetry standards conversation, a Linux Foundation project when governance is ready, and a CNCF sandbox application once the proxy/runtime has production users.

---

## Why this exists

Existing tools fall into two camps:

- **Observability**: LangSmith, LangFuse, Arize Phoenix, Helicone, PromptLayer, Datadog APM. They tell you *what happened*, often with a single-call edit-and-rerun panel. They do not chain a substitution forward through the rest of the agent run, bisect a regression, or cache unaffected downstream steps by semantic content hash.
- **HTTP cassettes**: pytest-vcr, pytest-recording. They replay bytes, not agent semantics. They cannot replace one tool result mid-trace, let the LLM react, and then distinguish the affected suffix from the reusable suffix.

stepback is the missing systems layer. The unit of execution is an agent step: LLM call, tool call, router decision, policy check, MCP call, or parallel branch. The trick is to turn recorded LLM calls into cached pure functions of canonicalized inputs.

If agents are going to make code changes, payments, safety decisions, and regulated workflow decisions, "what happened" is too weak. We need reversible traces, counterfactual replay, model-swap differential testing, incident minimization, provenance, policy audit, and cryptographic evidence that the trace was not rewritten after the fact.

---

## What's novel (and publishable)

### 1. SB-Trace: an open `.sb` trace specification

The `.sb` format is the headline claim. v1 is an append-only stream of length-prefixed canonical JSON frames, HMAC-chained and signed per frame. The header pins `format_version`, `recorder_version`, `canonicalisation_version`, `price_list_version`, signer public key, and HMAC key id. A deterministic CBOR encoding is a candidate for `format_version=2`, not a claim about v1.

The public track should look IETF-style: RFCs for byte layout, semantic model, canonicalization, security, versioning, and OpenTelemetry semantic conventions for `agent.step`. Multiple independent implementations should pass the same conformance suite.

### 2. Multi-language recorders and replay engines

The Python implementation is the seed. The intended architecture is:

- `stepback-core`: Rust trace writer/reader, canonicalizer, dirty-set engine, replay planner, verifier, and sharded cache client.
- Native bindings for Python, TypeScript/Node, Go, JVM, and .NET.
- WASM replay and verification for in-browser inspection.
- `stepback-proxy` / `sb`: HTTP and gRPC sidecar so Go, Rust, JVM, Node, and legacy stacks can record without a Python dependency.

### 3. LLM-aware step caching with dirty-set propagation

Given a trace and a substitution at step k, `divergence.py` computes downstream steps whose canonical inputs would hash differently and therefore must be re-executed. Every other step is replayed from cache. The formal artifact is a Coq/Lean mechanization of the soundness theorem: under substitution sigma, every non-dirty replayed step is observationally equivalent to full re-execution.

The systems artifact is a distributed dirty-set engine over a worker pool: sharded step cache on object storage, Kafka-backed event bus, ClickHouse trace warehouse, and distributed bisect across parallel replay workers.

### 4. Delta-debugging for stochastic traces

`minimize.py` adapts ddmin to traces where steps are stochastic, expensive, and not freely droppable. The ambitious version adds multi-objective minimization, predicate stability metrics, repeated dirty-step re-execution when needed, and incremental bisect across multiple regressions.

### 5. Trace-level differential testing and counterfactual sweeps

`trace_diff.py` and `sweep.py` compare model swaps, prompt edits, policy changes, retrieval changes, and parameter grids. The target is not a three-model demo; it is what-if exploration over grids of size one million plus, with statistical tests for whether a difference is real or stochastic noise.

### 6. Cryptographic replay evidence

Ed25519 is the current signature path. The roadmap adds post-quantum signatures with ML-DSA / SLH-DSA, threshold signing for long-lived recorders, witness cosigning for public benchmark traces, transparency-log integration, and CycloneDX-AI / SLSA / in-toto interop.

### 7. A benchmark consortium

The benchmark program should look more like MLPerf than a one-off chart: submission rules, hosted leaderboard, scheduled frontier-model re-evaluations, signed trace packs, hardware manifests, and public validators. Corpora should cover SWE-bench-Verified, GAIA, tau-bench, AgentBench, OSWorld, WebArena, and at least three author-original redistributable trace corpora.

Metrics: replay-caching cost reduction, dirty-set distribution on anonymized production traces, minimization stability under stochasticity, model-swap differential fidelity, recorder overhead, trace-reader throughput, and storage compression.

### 8. Paper outputs

The natural paper sequence is four or five artifacts: an MLSys / NeurIPS systems paper on dirty-set replay, an OSDI / ASPLOS systems paper on distributed replay, an ICSE / ISSTA paper on stochastic minimization and regression localization, a NeurIPS-Datasets benchmark paper, and a FAccT / industry case-study paper on incident debugging and auditability.

---

## Install

Today, use pipx or a virtualenv. Distro and Homebrew Pythons enforce PEP 668 and may reject a bare install into the system environment.

```bash
pipx install stepback
# or
python -m venv .venv && source .venv/bin/activate
pip install stepback
```

Source install:

```bash
git clone https://github.com/stepback-dev/stepback && cd stepback
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Future installs should look like normal ecosystem packages:

```bash
cargo install stepback-cli
npm install @stepback/recorder
pip install stepback[shims,bench]
go get github.com/stepback-dev/stepback-go
sb proxy --listen :4319 --write ./traces
```

Python is first. The target is multi-language. Recorder coverage starts with OpenAI, Anthropic, Bedrock, and Gemini, then expands to Vertex, Azure OpenAI, Cohere, Mistral, Together, Fireworks, Groq, Cerebras, NVIDIA NIM, vLLM, TGI, llama.cpp, and Ollama. Framework recorders should cover LangChain, LangGraph, LlamaIndex, DSPy, Haystack, AutoGen, CrewAI, Semantic Kernel, MCP, Strands, Pydantic-AI, and Inspect-AI.

---

## 60-second tour

```python
from stepback import record, replay

with record("./traces/incident-2026-04-12.sb"):
    agent.invoke({"input": "Pay invoice INV-118 to vendor 'Acme Bolts'."})

trace = replay("./traces/incident-2026-04-12.sb")

counterfactual = (trace
    .step_back(to="step:tool_call:lookup_customer")
    .substitute(prompt="System: be paranoid about PII")
    .replay_forward())

counterfactual.diff(trace).open_in_browser()
bad_step = trace.bisect(predicate=lambda s: s.cost_usd > 0.50)

sweep = trace.sweep(model=["gpt-4o", "claude-4.5", "gemini-2.5-pro"])
sweep.report().write("./sweep.html")
```

Target distributed use:

```python
from stepback import replay
from stepback.spec import SBTraceSpec

SBTraceSpec.load("sbtrace-v1.0.rfc.yaml").assert_conformant("./traces/*.sb")

trace = replay("s3://agent-traces/prod/2026/04/12/run.sb")
(trace.surgery()
    .graft(from_trace="s3://bench/golden/customer_lookup.sb",
           source="step:tool_call:lookup_customer",
           target="step:tool_call:lookup_customer")
    .sweep(model=["gpt-4o", "claude-4.5", "mistral-large"],
           temperature=[0, 0.2, 0.7],
           policy=["pci-strict", "pci-baseline"])
    .run(distributed=True, max_points=1_000_000))
```

CLI:

```text
stepback record -- python my_agent.py             record a run
stepback replay <trace.sb>                        deterministic replay
stepback inspect <trace.sb>                       step-by-step viewer
stepback bisect <trace.sb> --predicate cost_gt:0.5
stepback minimize <trace.sb> --predicate failed   delta-debug to minimal failing trace
stepback diff <a.sb> <b.sb>                       structural trace diff
stepback sweep <trace.sb> --model gpt-4o,claude-4.5
stepback bench replay-caching                     reproduce the seed benchmark
stepback report <trace.sb>                        HTML report
sb proxy --listen :4319 --write ./traces          sidecar recorder for non-Python stacks
```

The web UI roadmap adds a time-travel debugger, causal-graph viewer, multi-step simultaneous substitutions, automatic regression localization, trace-level differential testing, and trace surgery APIs that graft a subtrace from one run into another.

---

## The `.sb` trace format

Append-only, content-addressed, signed. In v1, each frame is a length-prefixed canonical JSON object preceded by a per-frame HMAC chained to the previous frame's HMAC. Tampering, reordering, and truncation are detectable. The header pins recorder version, canonicalization version, format version, price-list version, signer public key, and HMAC key id.

Each step records:

| Field | Purpose |
| --- | --- |
| `step_id` | ULID, monotonic per trace |
| `step_kind` | `llm_call`, `tool_call`, `router`, `policy_check`, `mcp_call`, `parallel_branch_open`, `parallel_branch_join`, `exception` |
| `parent_step_id` | edge in the call tree or DAG |
| `inputs` / `outputs` | canonical JSON plus content hash |
| `nondeterminism_hash` | hash of non-deterministic inputs consumed by the step |
| `llm_request` / `llm_response` | exact bytes, model id, messages, sampling params, tool spec, response |
| `policy_decision` | enforcement or audit decision when flowwarden/toolwarden is present |
| `wallclock_ns`, `cpu_ns`, `cost_usd` | timing and cost from pinned price-list version |
| `receipt` | Ed25519 signature chained via `prev_hmac` |

Canonicalization (`canonical.py`) is the cache-safety boundary. The roadmap requires SMT-checked canonicalizer equivalence across implementations, differential fuzzing across recorders, and conformance fixtures every implementation must pass before claiming `.sb` support.

Versioning is strict: old readers reject unknown mandatory fields; new readers read v1 forever; v2 features are negotiated via capability frames rather than guessed from optional blobs.

---

## The dirty-set algorithm (`divergence.py`)

```text
Given:
  trace T = [s0, s1, ..., sN]
  substitution sigma at step sk

Compute dirty-set D:
  D := {sk}
  for i in k..N:
    inputs_prime_i := apply sigma and propagate D to recompute si inputs
    if hash(inputs_prime_i) != hash(inputs_i):
      D := D union {si}
      mark si outputs as to-be-recomputed
    else:
      reuse cached outputs(si)

Replay cost: number of dirty steps that require real LLM/tool execution.
Correctness target: assuming canonical input hashing is sound, replay is
observationally equivalent to full re-execution under sigma on every
non-dirty step.
```

The production engine generalizes this from a list to a DAG with parallel branches, joins, policy decisions, tool calls, and imported spans. The distributed version shards dirty-set computation across workers, stores content-addressed outputs on object storage, indexes traces in ClickHouse, and schedules independent branch replay in parallel.

Formal artifacts are part of scope: Coq/Lean proof for dirty-set soundness, TLA+ spec of the append-only HMAC chain, and oracle tests comparing dirty-set replay against full re-execution on generated traces.

---

## Trace-minimization for stochastic traces (`minimize.py`)

Classical ddmin assumes deterministic programs and freely droppable input chunks. LLM traces are neither. stepback's minimizer:

1. Treats removal as a substitution.
2. Uses dirty-set propagation to compute the affected suffix.
3. Re-executes only dirty steps against real executors.
4. Evaluates the failure predicate on the resulting trace.
5. Continues ddmin over the remaining steps.

The ambitious version supports multi-objective minimization, stochastic confidence intervals, predicate DSL extensions, incremental bisect across multiple regressions, and automatic regression localization over a fleet of traces.

---

## Repository layout

```text
stepback/
├── stepback/
│   ├── recorder.py        # capture from shimmed clients
│   ├── shims.py           # OpenAI / Anthropic / Bedrock / Gemini shims today
│   ├── trace_writer.py    # .sb v1 writer, canonical JSON frames
│   ├── trace_reader.py    # .sb v1 reader / verifier
│   ├── canonical.py       # canonical input hashing
│   ├── replay.py          # deterministic replay engine
│   ├── divergence.py      # dirty-set propagation
│   ├── substitutions.py   # typed substitution builder
│   ├── minimize.py        # delta-debugging for traces
│   ├── sweep.py           # parameter sweeps amortized via cache
│   ├── trace_diff.py      # structural diff between two traces
│   ├── predicates.py      # predicate DSL for bisect / minimize
│   ├── attestation.py     # Ed25519 signing + attestation packs
│   ├── policy_audit.py    # replay-time policy checks
│   ├── pricing.py         # cost computation from pinned price lists
│   ├── importers.py       # LangSmith / OpenInference today; Phoenix/OTel roadmap
│   ├── exporters.py       # export to OTel, JSON, HTML
│   ├── html_view.py / report.py
│   └── cli.py
├── scripts/
│   ├── bench_record_overhead.py
│   └── bench_replay_caching.py
├── tests/                 # 605 collected in the audit snapshot
├── GROUNDING.md
└── README.md
```

Target layout adds `stepback-core/`, `bindings/python/`, `bindings/typescript/`, `bindings/go/`, `bindings/jvm/`, `bindings/dotnet/`, `wasm/`, `proxy/`, `spec/`, `bench/`, and `ui/`.

---

## Reproducing the bench

Current seed benchmark:

```bash
git clone https://github.com/<org>/stepback && cd stepback
python -m venv .venv && source .venv/bin/activate
pip install -e ".[bench]"
PYTHONPATH=. python scripts/bench_replay_caching.py
```

Target benchmark CLI:

```bash
stepback bench replay-caching --suite swe-bench-verified --n 50
stepback bench dirty-set --suite production-anon-2026q2 --substitutions 10000
stepback bench model-swap --suite gaia --from gpt-4o --to claude-4.5
stepback bench overhead --recorder openai --p50-budget-us 50
```

The consortium benchmark covers replay-caching cost reduction, dirty-set distribution on anonymized production traces, minimization stability under stochasticity, model-swap differential fidelity, recorder overhead, reader throughput, and storage compression. Submissions include trace packs, attestation receipts, hardware manifests, recorder versions, model versions, price-list versions, and a reproducibility script.

---

## What this is not

- **Not an observability dashboard.** stepback ingests LangSmith / LangFuse / Phoenix / Helicone / Datadog traces, but its native surface is counterfactual replay.
- **Not a single-call prompt playground.** Substitutions chain forward through every affected step.
- **Not an HTTP cassette layer.** Canonicalization works at LLM-call and agent-step semantics, not raw bytes.
- **Not a policy enforcer.** toolwarden and flowwarden can enforce. stepback records decisions, replays them, audits them, and packages evidence.
- **Not a closed hosted product.** Hosted viewers and leaderboards may exist, but `.sb`, the conformance suite, and the core replay machinery stay open.

---

## Project status

Substantial but early. The audit snapshot found about 25,448 lines of Python across `stepback/` and `tests/`, 605 tests passing on Python 3.14, `.sb` writing with canonical JSON frames, HMAC chaining, Ed25519 signatures, attestation packs, four provider shims, importers, exporters, dirty-set propagation, replay, minimization, parameter sweeps, policy audit, and parallel-branch tests.

Also true: no CI yet, no public benchmark consortium, no multi-language core, no formal proof, no independent implementation, and the current runnable benchmark is a script rather than the full CLI benchmark suite. That is the roadmap.

---

## Contributing

PRs welcome. The areas where outside contribution moves the needle most:

- independent `.sb` readers/writers and conformance fixtures,
- provider shims and framework recorders,
- importers from LangSmith, Phoenix, Helicone, Langfuse, OpenTelemetry/OpenInference, and Datadog APM,
- benchmark corpora and leaderboard submissions,
- formal artifacts: Coq/Lean, TLA+, SMT canonicalizer checks,
- production reports of dirty-set distributions on anonymized traces,
- integrations with ragdoctor, flowwarden, toolwarden, CycloneDX-AI, SLSA, and in-toto.

`pytest` must stay green. Public APIs in `stepback/__init__.py` are covered by semver from v0.1 onward; SB-Trace gets its own wire-format semver and compatibility policy.

---

## License

Apache-2.0. See `LICENSE`.

## Citation

If you use stepback, SB-Trace, or the benchmark suite in academic work:

```bibtex
@software{stepback,
  title  = {stepback: reversible debugging and counterfactual evaluation for LLM agents via SB-Trace and dirty-set replay},
  year   = {2026},
  url    = {https://github.com/<org>/stepback}
}
```
