# stepback Examples

Six self-contained examples that demonstrate stepback's record → replay →
substitute workflow with agents of increasing complexity.  Every example is
**fully offline**: it uses a deterministic fake LLM and scripted tool
stand-ins so no API keys are required.

| Directory | Agent pattern | Steps |
|---|---|---|
| `customer_support/` | Linear multi-tool support bot | 5–6 |
| `rag_pipeline/` | Retrieve → re-rank → synthesise | 6 |
| `tool_using_agent/` | Payments bot with bad tool output | 12 |
| `multi_step_planner/` | Plan → decompose → execute → verify | 8 |
| `parallel_branch/` | Fan-out / fan-in research agent | 11 |
| `batch_worker/` | Sequential batch over a job queue | variable |

## Quick start

```bash
# Re-record all reference traces:
make -C examples

# Re-record a single example:
make -C examples customer_support
```

Each subdirectory also has its own `README.md` with a deeper explanation
and copy-pasteable replay / substitution snippets.

## How it works

1. **Record** — an agent is driven through `stepback.record(path)`.  Every
   `rec.llm_call(...)` and `rec.tool_call(...)` call is appended as a
   signed, HMAC-chained frame to a `.sb` file.

2. **Replay** — `stepback.replay(path)` loads the trace.  `replay_forward()`
   re-executes every step with its original executor; cached steps are
   returned immediately without hitting the LLM.

3. **Substitute** — `trace.substitute(step_id, ToolOutputSubstitution(...))`
   pins a step to a counterfactual output.  `replay_forward()` then only
   re-executes the *dirty* downstream steps.

4. **Bisect** — `trace.bisect(predicate)` binary-searches for the first
   step that causes the predicate to flip from passing to failing.

See the individual `README.md` files and each `agent.py` for worked
examples of all four operations.
