# stepback

**A time-travel debugger for AI agents.** stepback records a production
agent run as an append-only, content-addressed, signed trace, then lets
you replay it deterministically, branch the execution at any step,
substitute prompts / tool outputs / policies / models, and re-execute
only the steps actually affected by the substitution against a real
LLM — every other step is served from the per-step cache.

Where `rr` and Pernosco give native-code engineers a reversible
debugger over CPU instructions, stepback gives agent engineers a
reversible debugger over LLM calls, tool calls, router decisions, and
parallel branches. The unit of stepping is *one agent step*; the
core trick is LLM-aware caching of step outputs so counterfactual
debugging of a 200-step trace costs ~5 LLM calls, not 200.

It is **not** an observability dashboard (LangSmith / LangFuse / Arize
Phoenix / Helicone) — those let you look at what happened. It is
**not** a single-call prompt playground (PromptLayer, the LangSmith
"edit & rerun" panel) — those re-execute one call in isolation. It is
**not** an HTTP cassette layer (pytest-vcr, pytest-recording) — those
replay bytes, not semantics. It is **not** a policy enforcer
(toolwarden, Microsoft Agent Governance Toolkit) — those decide what is
*allowed*; stepback reconstructs what *happened* and what *would have
happened*.

---

## Status

Pre-v0. Active design + scaffolding. v0.1 target is the Python SDK
covering the OpenAI + Anthropic clients and the LangChain tool
registry, plus the `.sb` trace format, the local replay engine with
substitution + dirty-set propagation, and the `stepback inspect` /
`stepback bisect` CLIs.

---

## Why nothing else fills this gap

Every tool below was checked against the requirement: *record a real
production agent trace, branch it at an arbitrary step, substitute
something, and replay only the affected sub-trace against a real LLM
with everything else served from cache.* No existing tool does this.

| Tool | What it does | Why it doesn't fill the gap |
| --- | --- | --- |
| **rr** (rr-debugger/rr) | Deterministic record-and-replay debugger for native code: records every syscall + non-deterministic CPU event so you can step a C/C++/Rust binary backwards under gdb. | Replays machine instructions, not agent steps. Has no concept of an LLM call, no notion of "the same prompt with one tool output changed", and no caching of semantic step outputs. |
| **Pernosco** | Hosted reverse-debugger built on rr; adds dataflow queries over a recorded native execution. | Same scope as rr — native code. Doesn't ingest LLM/tool traces and has no substitution model for prompts or tool responses. |
| **LangSmith** (LangChain) | Hosted trace viewer with a single-step "edit prompt and rerun this one call" panel. | Reruns one LLM call in isolation. Does not chain the substitution forward through the rest of the agent run, does not bisect, does not branch the trace at an arbitrary step, does not cache unaffected downstream steps. |
| **LangFuse** | Open-source observability + per-call "replay this prompt" against a chosen model. | Same single-step-rerun limitation. No dirty-set propagation, no multi-step counterfactual, no bisect. |
| **Arize Phoenix** | OSS observability with prompt-playground-style counterfactual edits at the single-call level. | Single-call scope; no notion of replaying the *agent* (router decisions, tool calls, parallel branches) under a substitution. |
| **Helicone** | Proxy-based logging and cost analytics for LLM calls. | Logging only. No replay engine, no substitution API, no cache of semantic step outputs. |
| **PromptLayer** | Capture + variant rerun of individual prompts; prompt registry. | Prompt-level only. No tool-output substitution, no policy substitution, no multi-step replay. |
| **TruLens** (truera/trulens) | Instrumentation framework + evaluative tracing (feedback functions over recorded runs). | An evaluator, not a debugger. Cannot alter a step and replay forward. |
| **airportyh/time-traveling-debugger** | Generic Python step-level time-travel debugger; records local-variable state per line. | Not LLM-aware: no canonicalisation of prompt/tool I/O, no per-step content-addressed cache, no notion of "this LLM call's output is reusable iff its inputs hash unchanged", no tool-output substitution. |
| **pytest-recording** / **pytest-vcr** | HTTP-cassette replay for tests; matches requests by URL/headers/body and replays the recorded response. | Bytes-level HTTP replay. Does not understand LLM-call semantics (model id, sampling, tool spec), cannot substitute *one* tool call's output mid-trace and let the LLM react, cannot bisect. |
| **Microsoft Agent Governance Toolkit** (Apr 2026) | Runtime policy + governance plane for agents: identity, allow/deny, audit trail. | Enforcement, not debugging. Records that a call happened; does not let you re-run the agent under a counterfactual. |
| **Sentry / Datadog APM** | Production observability for application code: spans, error grouping, profiling. | Span-level, not step-level; no LLM semantics, no replay, no substitution. |

> There is no time-travel debugger for AI agents that lets you record a
> production trace, branch it at any step, substitute prompts / tools /
> policies, replay only the changed sub-trace against the LLM (cached
> otherwise), and bisect to find the step that introduced a regression.
> stepback is that tool.

---

## What "specific" looks like

### 1. The three-line install + record / replay (Python)

```python
from stepback import record, replay

with record("./traces/incident-2026-04-12.sb") as rec:
    agent.invoke({"input": "Pay invoice INV-118 to vendor 'Acme Bolts'."})

# later, in a debugger session:
trace = replay("./traces/incident-2026-04-12.sb")
trace.step_back(to="step:tool_call:lookup_customer")
trace.substitute(prompt="System: be paranoid about PII").replay_forward()
trace.bisect(predicate=lambda step: step.cost_usd > 0.50)
```

`record()` is a context manager that wraps the agent's LLM client(s)
and tool registry for the duration of the block. `replay()` returns a
`Trace` object with reversible step navigation (`step_back`,
`step_forward`, `goto(step_id)`), a typed `substitute()` builder, and
the `bisect()` / `branch_at()` / `compare_branches()` operations
described below.

### 2. The trace file format (`.sb`)

`.sb` is an append-only, content-addressed file. Each frame is a
length-prefixed CBOR object preceded by a per-frame HMAC chained to
the previous frame's HMAC (so any tampering or truncation is
detectable, like toolwarden's audit ledger). The header pins the
recorder version, the canonicalisation version, and the public key
fingerprint of the signer.

Each step records:

| Field | Purpose |
| --- | --- |
| `step_id` | ULID, monotonic per trace |
| `step_kind` | `llm_call` \| `tool_call` \| `router` \| `parallel_branch_open` \| `parallel_branch_join` \| `exception` |
| `parent_step_id` | edge in the call tree (null for the root) |
| `inputs` | canonicalised CBOR + content hash (`sha256`) |
| `outputs` | canonicalised CBOR + content hash |
| `nondeterminism_hash` | hash of every non-deterministic input the step actually consumed (system clock reads, RNG draws, env vars, network responses) — divergence detector |
| `llm_request` | for `llm_call`: exact bytes — model id, full message list, temperature, top_p, seed, tools spec, response_format |
| `llm_response` | for `llm_call`: full response incl. streaming chunks, finish_reason, tool_calls, usage |
| `wallclock_ns`, `cpu_ns` | timing |
| `cost_usd` | computed from usage × price-list version pinned in header |
| `receipt` | Ed25519 signature over the frame, chained via `prev_hmac` |

A single `llm_call` step (truncated):

```json
{
  "step_id": "01HXYZK4...",
  "step_kind": "llm_call",
  "parent_step_id": "01HXYZK3...",
  "inputs_hash":  "sha256:7b1c...",
  "outputs_hash": "sha256:e904...",
  "nondeterminism_hash": "sha256:0000...",
  "llm_request": {
    "model": "gpt-4o-2024-11-20",
    "temperature": 0.0,
    "seed": 42,
    "messages": [
      {"role": "system", "content": "You are a payments agent..."},
      {"role": "user",   "content": "Pay invoice INV-118..."}
    ],
    "tools": [{"type": "function", "function": {"name": "lookup_customer", "...": "..."}}]
  },
  "llm_response": {
    "finish_reason": "tool_calls",
    "tool_calls": [{"id": "call_a1", "name": "lookup_customer",
                    "arguments": "{\"name\": \"Acme Bolts\"}"}],
    "usage": {"prompt_tokens": 412, "completion_tokens": 38}
  },
  "wallclock_ns": 1734012345678901234,
  "cpu_ns": 1240000,
  "cost_usd": 0.00219,
  "prev_hmac": "...",
  "sig": "ed25519:..."
}
```

### 3. The substitution API

`Substitution` is a typed object. Only these forms exist; arbitrary
monkey-patching is rejected at replay time so every counterfactual is
reproducible from the trace + the substitution list alone.

```python
from stepback.substitutions import (
    PromptSubstitution, ToolOutputSubstitution,
    PolicySubstitution, ModelSubstitution, RouterSubstitution,
)

trace = replay("./traces/incident-2026-04-12.sb")

trace.substitute(
    PromptSubstitution(at_step="step:7",
                       new_messages=[{"role": "system",
                                      "content": "Refuse if PII present."}]),
    ToolOutputSubstitution(at_step="step:12",
                           tool_call_id="call_a1",
                           fake_response={"customer_id": None, "error": "not_found"}),
    ModelSubstitution(at_step="step:7", new_model_id="gpt-4o-mini-2024-07-18"),
).replay_forward()
```

Each replay records the exact substitution list it ran under, so two
engineers comparing counterfactual branches always know which inputs
differ.

### 4. The replay semantics (LLM-aware caching)

This is the core insight that makes counterfactual debugging
affordable. Every step in the recorded trace already has its
`inputs_hash`, `outputs_hash`, and `nondeterminism_hash`. On replay:

1. Walk the call tree in topological order.
2. For each step, recompute `inputs_hash` from current (post-
   substitution) inputs.
3. If the new `inputs_hash` matches the recorded one **and** no
   ancestor step is dirty **and** `nondeterminism_hash` is unchanged,
   serve the recorded `outputs` from the content-addressed cache —
   *no real LLM or tool call*.
4. Otherwise, mark the step **dirty**, execute it for real (LLM or
   tool), and propagate dirtiness to every descendant.

```
recorded:    [s1]──[s2]──[s3]──[s4]──[s5]──[s6]──[s7]──...──[s200]
                                  ▲
                                  └── PromptSubstitution at s4

replay:      [s1]  [s2]  [s3]  [s4*] [s5*] [s6*] [s7*] ... [s200*]
              cache cache cache  RUN   RUN   RUN   RUN     (most cached
                                                            if subtree
                                                            converges)
```

Substituting one step in a 200-step trace typically re-executes ~5
steps (the dirty subtree until the agent's behaviour converges back
onto a cached path), instead of all 200.

### 5. Bisection

`trace.bisect(predicate)` does git-bisect-style binary search through
the linearised step sequence to find the earliest step at which
`predicate(step)` first becomes true. The classic use: a 3-hour agent
run made a bad payment at step 174 — find the earlier step that
caused the bad reasoning.

```python
trace = replay("./traces/incident-2026-04-12.sb")

bad_step = trace.bisect(
    good="step:1",
    bad="step:174",
    predicate=lambda s: s.kind == "llm_call"
                        and "wire to IBAN GB99" in s.outputs.get("text", ""),
)
print(bad_step.step_id, bad_step.parent_step_id)
# → step:88   (the lookup_customer call that returned the wrong row)
```

Each bisect probe is a cached replay (no LLM calls if the substitution
set is empty), so bisecting a 200-step trace is ~8 probes × ~0
LLM calls.

### 6. Branching

Branches are first-class, named, and comparable.

```python
trace = replay("./traces/incident-2026-04-12.sb")

main = trace.branch_at("step:7", name="main")
paranoid = (trace.branch_at("step:7", name="counterfactual-paranoid")
                 .substitute(PolicySubstitution(at_step="step:7",
                                                policy_path="./pii-strict.tw"))
                 .replay_forward())

diff = trace.compare_branches(main, paranoid)
for d in diff.step_diffs:
    print(d.step_id, d.kind, d.output_diff, d.cost_delta_usd)
```

`compare_branches` produces a step-by-step diff: output diffs,
decision changes, cost deltas, and which steps diverged from cache.
This is the substrate for A/B testing prompts, policies, and models on
real production traces.

### 7. The CLI

```
stepback record  --output trace.sb -- python my_agent.py
stepback inspect trace.sb                          # interactive TUI: timeline + step viewer
stepback bisect  trace.sb --good "step:1" --bad "step:174" \
                          --predicate "cost > 0.50"
stepback replay  trace.sb --substitute prompt@step:7=new_prompt.txt
stepback diff    trace.sb branch:main branch:counterfactual-paranoid
stepback verify  trace.sb --policy-changed-since "2026-04-01"
```

`stepback verify` re-executes a trace under a newer policy version
(via toolwarden) and emits a structured report of every decision that
would now differ — the regulator-replay use-case.

---

## Architecture

```
                 ┌──────────────────────────────────────┐
 agent.invoke ──▶│  Recorder shim (in-process)          │
                 │   - wraps LLM client (OpenAI / etc.) │
                 │   - wraps tool registry (LC / MCP)   │
                 │   - canonicalises inputs + outputs   │
                 │   - hashes; signs; HMAC-chains       │
                 └──────────────────────────────────────┘
                                  │ frames
                                  ▼
                 ┌──────────────────────────────────────┐
                 │  .sb trace (append-only, signed)     │
                 │   header │ step │ step │ ... │ tail  │
                 └──────────────────────────────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
 ┌──────────────────────────┐          ┌──────────────────────────┐
 │  Replay Engine           │          │  Verifier                │
 │   - reads .sb            │          │   - sig + HMAC chain     │
 │   - reconstructs tree    │          │   - canonicalisation     │
 │   - applies Substitutions│          │     version match        │
 │   - dirty-set propagator │          └──────────────────────────┘
 │   - serves cache hits    │
 │   - executes dirty steps │
 │     against real LLM     │
 └──────────────────────────┘
              │
              ▼
 ┌──────────────────────────┐
 │  Surfaces                │
 │   - CLI (record/inspect/ │
 │     bisect/replay/diff)  │
 │   - TUI (textual)        │
 │   - Web UI (team tier)   │
 └──────────────────────────┘
```

Recorder writes append-only frames + signed receipts. Replay engine
walks the call tree, applies typed substitutions, and executes only
dirty steps against the real LLM, caching the rest by content hash.
Surfaces (CLI, TUI, web) read the same `.sb` format.

---

## Repository layout (planned)

```
stepback/
├── stepback/
│   ├── recorder/
│   │   ├── llm_clients.py    # shims for OpenAI / Anthropic / Bedrock / vLLM
│   │   ├── tool_registry.py  # shims for LangChain / CrewAI / AutoGen / MCP
│   │   ├── trace_writer.py   # append-only .sb writer with HMAC chain
│   │   └── canonicalizer.py  # input/output canonicalisation for hashing
│   ├── replay/
│   │   ├── engine.py         # walk call-tree, apply substitutions, schedule
│   │   ├── cache.py          # content-addressed step-output cache
│   │   ├── nondet.py         # divergence detector
│   │   └── bisect.py
│   ├── substitutions/
│   ├── trace/
│   │   ├── reader.py
│   │   ├── canonical.py
│   │   └── verify.py         # signature + chain verification
│   ├── cli/
│   └── tui/                  # textual-based interactive timeline
├── ts/                       # TypeScript SDK
├── server/                   # optional team-tier hosted replay engine
├── examples/
│   ├── 01-bisect-bad-payment.md
│   ├── 02-counterfactual-policy-change.md
│   ├── 03-prompt-AB-on-prod-traffic.md
│   ├── 04-debug-tool-cascade-failure.md
│   └── 05-regulator-replay-under-new-policy.md
├── tests/
└── docs/
```

---

## Performance targets

| Budget | Target |
| --- | --- |
| Record overhead per LLM call | **< 5 ms p99** (canonicalise + hash + sign + HMAC + append) |
| Trace size on disk | **< 30%** of raw LLM payload bytes (CBOR + gzip + per-message dedup against header dictionary) |
| Replay of an unchanged 200-step trace from cache | **< 2 s** end-to-end on a laptop, zero LLM calls |
| Bisect of a 200-step trace, no substitutions | **≤ 10 probes**, ≤ 5 s wall, zero LLM calls |
| Substitution of one step in a 200-step trace | **typically ≤ 10** real LLM re-executions (dirty-set size) |

---

## Milestones

| Milestone | What works end-to-end |
| --- | --- |
| **m0 — scaffolding** | repo layout, CI, `.sb` file format spec, HMAC + Ed25519 signature chain, `stepback verify` |
| **m1 — first recorder** | OpenAI client recorder + replay-from-cache + `stepback inspect` (read-only timeline) |
| **m2 — tool tree** | LangChain tool recorder; full call-tree reconstruction across LLM + tool + router steps |
| **m3 — substitutions** | typed Substitution API + dirty-set propagation + LLM-aware re-execution + content-addressed cache |
| **m4 — bisect** | `stepback bisect` + the five worked examples in `examples/` |
| **m5 — branches + diff + TUI** | `branch_at`, `compare_branches`, `stepback diff`, textual TUI |
| **m6 — Anthropic + Bedrock + MCP + TS SDK** | parity recorders for second + third LLM providers; MCP server-side recorder; TypeScript SDK |
| **m7 — hosted team tier** | multi-tenant trace storage, RBAC, replay scheduler, web UI |
| **m8 — regulator-replay tier** | replay every past trace under any new policy version (via toolwarden); produces a signed attestation pack |

v1.0 = m0 through m6.

---

## Synergy with the wider stack

stepback's `.sb` signature + HMAC chain is the same shape as
kitchensink's `AuditableArtifact` substrate; the plan is to depend on
the same `auditable-controller-core` package toolwarden uses, so all
three projects share one verified append-only-log implementation.

stepback and toolwarden compose: toolwarden writes per-call
**decision** receipts (was this allowed, under which policy, with
which approver), stepback writes per-call **execution** receipts (what
were the exact bytes in and out, what did the LLM actually return).
Together they answer the regulator's full question: *what did the
agent decide, why was it permitted, and what would have happened if
our current policy had been in force?* — by replaying every recorded
`.sb` under the latest toolwarden policy version and diffing the
decisions.

---

## Use-cases

### 1. Production incident — "agent paid $50k to the wrong vendor"

```
$ stepback inspect ./traces/2026-04-12T13-04Z-finbot.sb
  → 312 steps, root: agent.invoke({"input": "Pay invoice INV-118..."})
  → step:174 tool_call payment.transfer  amount_usd=50000  recipient="Acme Bolts Ltd UK"
$ stepback bisect ./traces/2026-04-12T13-04Z-finbot.sb \
    --good step:1 --bad step:174 \
    --predicate 'step.kind=="tool_call" and step.name=="lookup_customer" \
                 and "Acme Bolts Ltd UK" in step.outputs.get("name","")'
  → first-bad: step:88   lookup_customer({"name":"Acme Bolts"})
                          → returned UK entity instead of US entity
                          (fuzzy match across two customer rows)
```

Root cause located in 8 cached probes, 0 LLM calls.

### 2. Counterfactual policy — "if our current PII policy had been in force last quarter"

```python
from stepback import replay
from stepback.substitutions import PolicySubstitution

blocked = 0
for path in glob("./traces/2026-Q1/*.sb"):
    t = replay(path)
    t.substitute(PolicySubstitution(at_step="step:0",
                                    policy_path="./policies/pii-2026-04.tw"))
    if t.replay_forward().any_step(lambda s: s.kind == "exception"
                                            and s.error_class == "PolicyDenied"):
        blocked += 1
print(f"{blocked}/12000 runs would have been blocked")
```

Cache hit rate for this kind of audit is typically > 99% — the
substitution only flips outcomes for steps that touched PII tools.

### 3. Prompt A/B on real traffic — "test a new system prompt on 1000 production traces"

```python
from stepback import replay
from stepback.substitutions import PromptSubstitution

new_sys = open("./prompts/v7.txt").read()
deltas = []
for path in sample_traces(n=1000):
    t = replay(path)
    b_main = t.branch_at("step:0", name="main")
    b_v7   = (t.branch_at("step:0", name="v7")
                .substitute(PromptSubstitution(at_step="step:0",
                                               new_messages=[{"role":"system",
                                                              "content":new_sys}]))
                .replay_forward())
    deltas.append(t.compare_branches(b_main, b_v7))

print("mean cost delta:", mean(d.total_cost_delta_usd for d in deltas))
print("decisions changed:", sum(d.divergent_step_count for d in deltas))
```

Underlying tool calls are not re-executed (cached) unless the prompt
change actually causes the agent to ask different questions.

### 4. Tool cascade debugging — "47 tool calls, find the one that returned the bad row"

```
$ stepback bisect ./traces/support-7711.sb \
    --good step:1 --bad step:end \
    --predicate 'step.kind=="llm_call" and "refund issued" in str(step.outputs)'
  → first-bad: step:118  tool_call db.query
                          → returned ticket_id=7710 instead of 7711
                          (off-by-one in the query template)
```

### 5. Regulator replay — "Big-4 auditor wants deterministic re-execution of every Q1 2026 agent decision"

```
$ stepback verify ./traces/2026-Q1/*.sb \
    --policy-version-pin 2026-04-15 \
    --attestation-out ./attestations/Q1-2026.pack
  → 12,418 traces verified  (signature + HMAC chain ok)
  → 12,418 traces re-executed under policy 2026-04-15
  → 41 traces produced different decisions
  → attestation pack: 184 MB, signed Ed25519 fingerprint d4:5a:...
```

The attestation pack is itself an append-only signed artifact and is
what gets handed to the auditor.

---

## License

TBD. SDK + CLI + TUI will be Apache-2.0; hosted team tier and
regulator-replay tier are commercial.

---

## Status of this README

This README is a *specification*, not a record of what is built. Every
section describes the intended v0.1–v1.0 surface. Actual implementation
status will be tracked in `MILESTONES.md` once code starts landing.
