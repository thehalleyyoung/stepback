# Customer-support agent

A deterministic support bot that handles a delayed-order enquiry.

## Agent flow (6 steps)

```
step 1  llm_call   "decide to look up the order"
step 2  tool_call  lookup_order        → {status: shipped, overdue: 3 days}
step 3  tool_call  check_shipping      → {status: delayed, days_overdue: 3}
step 4  llm_call   "decide to escalate"
step 5  tool_call  escalate_to_shipping → {ticket_id: ESC-…, status: escalated}
step 6  llm_call   "draft reply to customer"
```

## Re-record

```bash
make -C examples customer_support
# or
python examples/customer_support/record.py
```

## Replay with substitution

```python
from stepback import replay
from stepback.substitutions import ToolOutputSubstitution
from stepback.testing.support_agent import fake_tool, fake_llm

trace = replay("examples/customer_support/trace.sb")
shipping_step = next(s for s in trace.recorded_steps
                     if s.get("name") == "check_shipping")

# Counterfactual: order arrived on time — no escalation needed.
trace.substitute(
    shipping_step["step_id"],
    ToolOutputSubstitution(output={"status": "on_time", "days_overdue": 0}),
)

result = trace.replay_forward(
    llm_executors={"gpt-4o-2024-11-20": fake_llm},
    tool_executors={"check_shipping": fake_tool,
                    "lookup_order": fake_tool,
                    "escalate_to_shipping": fake_tool},
)
dirty = [s for s in result.steps if s.get("dirty")]
print(f"dirty steps: {len(dirty)}")  # escalate + final LLM become dirty
```

## What to learn here

- `ToolOutputSubstitution` lets you pin any tool's output to a fixed value.
- Only the steps that **depend** on the changed output are re-executed (the
  dirty set); the unchanged early steps are served from the cache.
