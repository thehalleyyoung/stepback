# stepback

**Time-travel debugger for AI agents.** Record LLM-agent runs as signed `.sb`
traces, replay counterfactuals with dirty-set propagation, and bisect
regressions — re-executing only the steps whose inputs actually changed.

```bash
pip install git+https://github.com/thehalleyyoung/stepback.git
```

```python
from stepback import record, replay, Executor, RecorderKey
from stepback.substitutions import ToolOutputSubstitution

llm = lambda m, msgs: {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}}
tool = lambda n, a: {"country": "GB"}
key = RecorderKey.fresh()
with record("agent.sb", key=key) as r:
    r.llm_call("gpt-4o-mini", [{"role": "user", "content": "help c1"}], executor=llm)
    r.tool_call("fetch", {"id": "c1"}, executor=tool)
trace = replay("agent.sb", hmac_key=key.hmac_key)
print(trace.replay_forward().real_executions)   # → 0  (all cache hits)
trace.substitute(ToolOutputSubstitution(at_step="step:2", fake_response={"country": "US"}))
changed = trace.replay_forward(Executor(llm=llm, tool=tool))
culprit = trace.bisect("step:1", "step:2", predicate=lambda s: "US" in str(s.outputs))
```

> After one tool substitution, `changed.real_executions` is 1 — only the
> downstream LLM step re-ran; unaffected steps are served from recorded
> outputs with zero executor calls.

---

## Motivation

Agent debugging usually starts from an execution trace: the prompt, tool inputs, tool outputs, routing decisions, model responses, costs, and failures that occurred in one real run. Observability systems are good at showing that history, but a debugger also needs to ask counterfactual questions: what if one tool result had been different, a system prompt changed, a router picked another branch, or a policy file had blocked a later action?

Re-running the whole agent is expensive and often changes unrelated steps. HTTP cassette tools avoid live calls, but they replay raw requests rather than agent-step semantics. stepback's model is to treat each recorded step as a content-addressed computation over canonical inputs. If a substitution changes only part of the trace, the replay engine can propagate that change through the dependency graph and re-execute only the affected dirty set.

That makes local debugging, incident write-ups, policy audits, model-swap checks, and corpus sweeps cheaper than full re-execution while still exposing where the counterfactual run diverged from the recorded run. The current implementation is code-first: the Python package is the reference implementation, and the repository includes tests and conformance fixtures for the trace format.

## Technical ideas

### `.sb` trace format

The v1 `.sb` format is implemented by `stepback/trace_writer.py` and `stepback/trace_reader.py`. On disk it is a sequence of frames:

```text
| 4-byte big-endian length | canonical-JSON wrapper |
```

Each wrapper contains a frame body plus `prev_hmac`, `hmac`, and `sig` fields. The HMAC is `HMAC-SHA256(hmac_key, prev_hmac || canonical_json(body))`; the signature is Ed25519 over the frame HMAC. The first frame is a header with `magic`, `format_version`, `recorder_version`, `canonicalisation_version`, `price_list_version`, `public_key`, and `hmac_key_id`. Step frames store `step_id`, `step_kind`, parent links, canonical input/output hashes, optional LLM request/response data, timing, cost, and nondeterminism metadata. Tail, blob, capability, and Merkle summary frames are supported.

`stepback/spec.py` defines the package-independent SB-Trace wire version (`1.0.0`, canonical JSON) and a schema/conformance API. `spec/sbtrace-v1.md` documents the v1 byte layout and fixtures live under `stepback-core/fixtures/v1/`.

### Canonicalization and hashes

`stepback/canonical.py` is the cache-safety boundary. It emits deterministic UTF-8 JSON with sorted keys and no extra whitespace, rejects unsupported values, and computes `sha256:<hex>` hashes. Tests cover round-trips, Unicode, bytes, non-finite floats, nesting/size limits, and property-based cases. `stepback/canonical_cbor.py` and `stepback/semantic_hash.py` contain experimental v2 work, but v1 traces written by the Python package are canonical JSON.

### Recording

`stepback/recorder.py` provides the public `record()` context manager and `Recorder` primitives:

- `llm_call(model, messages, executor=...)`
- `tool_call(name, arguments, executor=...)`
- `router(name, choice, options)`
- `exception(error_class, message)`
- `parallel(name, branches, join=...)`

The recorder computes input/output hashes, records parent-step dependencies through `context` hashes, records branch fan-out/fan-in metadata, computes costs via `stepback/pricing.py`, and writes signed frames through `TraceWriter`. `arecord()` and `get_current_recorder()` use `contextvars` for async code. `stepback/autorecord.py` and `stepback/shims.py` add optional SDK/framework adapters.

### Provider and framework shims

`stepback/shims.py` contains duck-typed wrappers and replay executors for OpenAI, Anthropic, Bedrock, Gemini, Azure OpenAI, LangChain tools, and MCP sessions. The shim layer normalizes provider-native responses into a shared chat-completion shape before hashing. The repo also contains tests and compatibility cassettes for provider response shapes.

### Dirty-set propagation

`stepback/divergence.py` implements `compute_dirty_set(trace, substitutions)`, a pure classifier that walks recorded steps in topological order and marks a step dirty when:

- a substitution targets it directly,
- its recomputed canonical input hash differs,
- a parent it depends on is dirty,
- recorded nondeterminism metadata requires re-execution.

The algorithm understands single-parent `context` dependencies and multi-parent `parallel_branch_join` dependencies via `branch_tail_hashes`. `stepback/distributed_dirty.py` partitions branch regions and computes the same summary over a worker pool. `docs/dirty-set.md`, `docs/dirty-set-soundness.md`, `docs/dirty-set-completeness.md`, and `proofs/lean/Stepback/Soundness.lean` describe and mechanize the core soundness argument for the modeled DAG.

### Replay engine

`stepback/replay.py` is the executable replay engine. `replay(path)` returns a `Trace`; `Trace.replay_forward()`, `Trace.run_replay()`, `Trace.step_back()`, `Trace.branch_at()`, `Trace.compare_branches()`, and `Trace.bisect()` are part of the public API. Clean steps are served from recorded outputs. Dirty LLM/tool/router steps call an `Executor`, unless a substitution supplies the output or `Executor(fallback_recorded=True)` is used for offline analysis.

Replay results include per-step dirty/cache state, cost summaries, executor-call counts, and provenance fields. The engine has sequential execution, branch-parallel execution via `workers=N`, a `distributed=True` convenience mode backed by the same local planner, and an event stream (`replay_events`) used by the proxy replay endpoint.

### Step cache

`stepback/step_cache.py` implements a content-addressed cache keyed by `(step_kind, inputs_hash)`. `DiskStepCache` is the local implementation and stores JSON entries under sharded directories. `S3StepCache`, `GCSStepCache`, and `AzureStepCache` are dependency-gated object-store implementations. Replay consults the cache for dirty steps whose new inputs match a cached output and avoids executor calls on hits. Nondeterminism-forced steps, tool-output substitutions, and fallback-recorded outputs bypass or avoid polluting the cache.

### Substitutions, diffs, minimization, and sweeps

`stepback/substitutions.py` defines typed substitutions for prompts, messages, tool outputs, fields, JSON patches, models, sampling, routers, policies, and exceptions. `stepback/branch_io.py` persists substitution sets as `.sbb` branch files. `stepback/trace_diff.py` and `stepback/branch_io.py` compare replays and traces.

`stepback/minimize.py` implements replay-backed minimization over substitution sets. It includes ddmin, linear shrink, binary halving, brute force, Shapley attribution, multi-objective minimization, branch-aware minimization, and imported-trace handling when some executors are unavailable. `stepback/minimize_report.py` renders self-contained HTML minimization reports.

`stepback/sweep.py` applies substitutions across a corpus of `.sb` traces and aggregates cost deltas, dirty counts, cache-hit ratios, divergent-step counts, and failures. `stepback/sweep_checkpoint.py` adds disk checkpoints and leases for resumable sweeps.

### Attestation, verification, reports, and import/export

`stepback/attestation.py` builds and verifies signed `.pack` attestation files over one or more traces. `stepback/verify_policy.py`, `stepback/redact.py`, and `stepback/policy_audit.py` add strict verification, redaction scans/rewrites, and counterfactual policy reports.

`stepback/html_view.py`, `stepback/report.py`, and `stepback/minimize_report.py` render self-contained HTML/Markdown/JSON reports. `stepback/importers.py` imports OpenAI chat logs, LangSmith JSONL, OpenInference/OTel spans, Phoenix, Helicone, Langfuse, Datadog APM, native JSON, and CycloneDX-AI. `stepback/exporters.py` exports to OpenAI chat log, LangSmith JSONL, OpenInference/OTel spans, native JSON, HTML, and CycloneDX-AI, with lossiness reporting.

### Other components in the repo

The Python package is the most complete implementation. The repository also contains:

- `stepback/proxy/`: HTTP and optional gRPC sidecar for start/record/end/verify and replay events.
- `stepback/bench/`: synthetic replay-caching, dirty-set-distribution, minimization, model-swap, storage-compression, microbenchmark, soak, submission, leaderboard, and frontier-reevaluation utilities.
- `stepback-core/`: Rust crates for v1 format parsing, canonicalization, verification, a partial dirty-set kernel, replay-planner traits, and a WASM verifier/summarizer.
- `bindings/`: Python/PyO3, TypeScript, Go, JVM, and .NET read/verify bindings with fixture tests.
- `wasm/`: a demo page and build instructions for the Rust WASM verifier/summarizer; generated bundles are gitignored.
- `proofs/`: Lean and TLA+ artifacts for the modeled dirty-set and HMAC-chain properties.
- `docker/`: Dockerfiles and proxy smoke-test support.

## How to use

### Install from this checkout

Use a virtual environment rather than installing into the system Python:

```bash
git clone https://github.com/your-org/stepback
cd stepback
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras in `pyproject.toml` are `shims`, `bench`, `proxy-grpc`, and `quickstart`:

```bash
pip install -e ".[dev,shims,bench,proxy-grpc,quickstart]"
```

The installed console script is `stepback` (`stepback.cli:main`).

For extended Python examples used by the test suite, see `stepback/testing/` and `tests/test_e2e_replay.py`.

### CLI examples

Record a Python script with ambient autorecording:

```bash
stepback record --output trace.sb -- python my_agent.py
```

Inspect, replay, and save a counterfactual branch:

```bash
stepback inspect trace.sb
stepback replay trace.sb --json
stepback replay trace.sb \
  --substitute 'tool_output@step:2=:inline:{"customer_id":"c1","country":"US"}' \
  --branch-out fixed.sbb \
  --json
stepback diff trace.sb --a-branch fixed.sbb
```

Verify a trace when you have the HMAC key:

```bash
stepback verify trace.sb --hmac-key-hex "$STEPBACK_HMAC_KEY_HEX"
```

Render local reports and viewers:

```bash
stepback view trace.sb --output trace.html
stepback debug trace.sb --output debug.html --sub 'model@step:1=gpt-4o-mini-2024-07-18'
stepback report trace.sb --substitute 'router@step:3=alternate' --output report.md
stepback trace-diff before.sb after.sb --format markdown
```

Run analysis commands:

```bash
stepback bisect trace.sb --good step:1 --bad step:12 --predicate 'step.cost_usd > 0.50'
stepback minimize trace.sb \
  --substitute 'model@step:1=gpt-4o-mini-2024-07-18' \
  --predicate 'result.total_cost_usd > 0.01'
stepback sweep traces/*.sb --substitute 'policy@step:7=policies/strict.json' --format json
stepback divergence trace.sb --hmac-key-hex "$STEPBACK_HMAC_KEY_HEX" --executor-recorded
```

Import/export and redaction:

```bash
stepback import --format langsmith_jsonl --input runs.jsonl --output imported.sb
stepback export --format native_json --input trace.sb --output trace.json --hmac-key-hex "$STEPBACK_HMAC_KEY_HEX"
stepback redact-scan trace.sb --hmac-key-hex "$STEPBACK_HMAC_KEY_HEX" --json
stepback redact trace.sb --output redacted.sb --hmac-key-hex "$STEPBACK_HMAC_KEY_HEX" --policy standard
```

Attestation and conformance:

```bash
stepback attest trace.sb --hmac-key-hex "$STEPBACK_HMAC_KEY_HEX" --out trace.pack
stepback verify-pack trace.pack
stepback spec test -- ./path/to/verifier
```

Benchmarks and health checks:

```bash
stepback bench replay-caching --n-steps 200 --n-trials 10 --out replay-caching.json
stepback bench dirty-set-distributions --out dirty-set-distributions.json
stepback bench record-overhead --n-steps 1000
stepback doctor
stepback diagnose --all
```

Run the HTTP proxy sidecar:

```bash
stepback proxy --listen 127.0.0.1:4319 --write ./traces
# add --grpc-listen 127.0.0.1:4320 when the proxy-grpc extra is installed
```

### Tests

The repository uses pytest:

```bash
python3 -m pytest
```

The test tree covers record/replay, trace verification and corruption rejection, dirty-set propagation, parallel branches, substitutions, minimization, sweeps, shims, import/export, proxy endpoints, benchmarks, bindings, and spec/conformance invariants.

## Status / scope

stepback is alpha software. The Python recorder/replay/minimization stack is the reference implementation and has broad in-repo test coverage. The Rust, WASM, and language-binding directories contain real read/verify and conformance work, but they are not complete replacements for the Python recorder and replay engine. Hosted services, public benchmark leaderboards, upstream standards adoption, package-manager releases, production governance, and third-party operational guarantees are outside the current implemented scope unless their code is present in this repository.

## License

Apache-2.0. See `LICENSE`.
