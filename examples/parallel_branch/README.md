# Parallel-branch research agent

A fan-out / fan-in research agent that launches three concurrent
sub-investigations (history, economics, politics) and synthesises the results.

## Agent flow (11 steps)

```
step 1   llm_call               plan
step 2   tool_call              split_question
step 3   parallel_branch_open   research_fanout
step 4   llm_call               branch A: history
step 5   tool_call              lookup_history
step 6   llm_call               branch B: economics
step 7   tool_call              lookup_economics
step 8   llm_call               branch C: politics
step 9   tool_call              lookup_politics
step 10  parallel_branch_join   merge branches A+B+C
step 11  llm_call               synthesise
```

## Re-record

```bash
make -C examples parallel_branch
```

## Substitute one branch

```python
from stepback import replay
from stepback.substitutions import ToolOutputSubstitution
from stepback.testing.parallel_agent import fake_llm, fake_tool

trace = replay("examples/parallel_branch/trace.sb")
econ_step = next(s for s in trace.recorded_steps
                 if s.get("name") == "lookup_economics")

trace.substitute(
    econ_step["step_id"],
    ToolOutputSubstitution(output={"answer": "GDP 25T USD (revised)", "confidence": 0.85}),
)

result = trace.replay_forward(
    llm_executors={"gpt-4o-2024-11-20": fake_llm},
    tool_executors={"split_question": fake_tool,
                    "lookup_history": fake_tool,
                    "lookup_economics": fake_tool,
                    "lookup_politics": fake_tool},
)
dirty = [s for s in result.steps if s.get("dirty")]
print(f"dirty: {len(dirty)}")  # join + synthesise become dirty; other branches stay cached
```

## What to learn here

- Substituting one branch's tool output propagates through the join into the
  synthesise step but leaves the other two branches completely cached.
- The join step itself is re-executed because its inputs changed.
