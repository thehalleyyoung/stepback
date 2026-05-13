# Tool-using payments agent

The canonical stepback fixture: a 12-step customer payments bot that wires
$50,000 to the wrong vendor because `lookup_customer` returns the UK row
("Acme Bolts Ltd UK") instead of the US row ("Acme Bolts Inc").

## Agent flow (12 steps)

```
step 1   llm_call   plan
step 2   tool_call  lookup_customer → {iban: GB99-9999-9999}  ← BUG
step 3   llm_call   verify
step 4   tool_call  echo(amount)
step 5   llm_call   continue
step 6   tool_call  echo(country)
step 7   llm_call   think
step 8   tool_call  echo(ready)
step 9   llm_call   think again
step 10  tool_call  echo(iban preview)
step 11  llm_call   emit wire instruction
step 12  tool_call  payment.transfer(iban=GB99-…)  ← bad outcome
```

## Re-record

```bash
make -C examples tool_using_agent
```

## Find and fix the bug

```python
from stepback import replay
from stepback.substitutions import ToolOutputSubstitution
from stepback.testing.agent import LOOKUP_FIXED_ROW, fake_llm, fake_tool

trace = replay("examples/tool_using_agent/trace.sb")
lookup = next(s for s in trace.recorded_steps if s.get("name") == "lookup_customer")

trace.substitute(lookup["step_id"], ToolOutputSubstitution(output=LOOKUP_FIXED_ROW))

result = trace.replay_forward(
    llm_executors={"gpt-4o-2024-11-20": fake_llm},
    tool_executors={"lookup_customer": fake_tool,
                    "payment.transfer": fake_tool,
                    "echo": fake_tool},
)
wire = next(s for s in result.steps if s.get("name") == "payment.transfer")
print(wire.get("inputs", {}).get("iban"))  # → US12-3456-7890
```

## What to learn here

- `ToolOutputSubstitution` replaces a single tool result; all downstream
  steps that read that result become dirty and are re-executed.
- The `echo` steps that don't depend on the lookup output stay cached.
