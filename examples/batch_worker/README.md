# Batch worker (content moderation)

A content-moderation pipeline that processes a queue of five user posts.
Each post goes through three steps: LLM decision, rule-based classification,
and persistence — 15 steps total.

## Agent flow (15 steps, 3 per post × 5 posts)

```
[post p1]  llm_call → classify_post → record_decision
[post p2]  llm_call → classify_post → record_decision
[post p3]  llm_call → classify_post → record_decision
[post p4]  llm_call → classify_post → record_decision
[post p5]  llm_call → classify_post → record_decision
```

## Re-record

```bash
make -C examples batch_worker
```

## Override one post's decision

```python
from stepback import replay
from stepback.substitutions import ToolOutputSubstitution
from examples.batch_worker.agent import fake_llm, fake_tool

trace = replay("examples/batch_worker/trace.sb")

# Find the LLM step that handled post p2.
p2_llm = next(
    s for s in trace.recorded_steps
    if s.get("step_kind") == "llm_call" and "p2" in str(s.get("inputs", ""))
)

# Override: flip spam rejection to approval.
trace.substitute(
    p2_llm["step_id"],
    ToolOutputSubstitution(output={
        "id": "override", "model": "gpt-4o-2024-11-20",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant",
                                 "content": "Decision: approve. Ref: override"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }),
)

result = trace.replay_forward(
    llm_executors={"gpt-4o-2024-11-20": fake_llm},
    tool_executors={"classify_post": fake_tool, "record_decision": fake_tool},
)
dirty = [s for s in result.steps if s.get("dirty")]
print(f"dirty steps: {len(dirty)}")  # only the 2 downstream steps for p2 are dirty
```

## What to learn here

- In a loop/batch trace, substituting one item's LLM output makes only that
  item's downstream steps dirty — the other four posts stay fully cached.
- This is the key cost-saving property of stepback's dirty-set propagation.
