# Multi-step planner

An agent that decomposes a high-level goal ("set up a dev environment") into
three sub-tasks, executes each one, and verifies the outcome.

## Agent flow (11 steps)

```
step 1   llm_call   decompose goal
step 2   tool_call  create_plan
step 3   llm_call   action for task-1 (install deps)
step 4   tool_call  execute_task(install_dependencies)
step 5   llm_call   action for task-2 (run tests)
step 6   tool_call  execute_task(run_tests)
step 7   llm_call   action for task-3 (deploy staging)
step 8   tool_call  execute_task(deploy_staging)
step 9   llm_call   summarise results
step 10  tool_call  verify_plan
step 11  llm_call   final report
```

## Re-record

```bash
make -C examples multi_step_planner
```

## Substitute a task failure

```python
from stepback import replay
from stepback.substitutions import ToolOutputSubstitution
from examples.multi_step_planner.agent import (
    fake_llm, fake_tool, _FAILED_TEST_RESULT
)

trace = replay("examples/multi_step_planner/trace.sb")
test_step = next(
    s for s in trace.recorded_steps
    if s.get("name") == "execute_task"
    and s.get("inputs", {}).get("action") == "run_tests"
)
trace.substitute(test_step["step_id"],
                 ToolOutputSubstitution(output=_FAILED_TEST_RESULT))

result = trace.replay_forward(
    llm_executors={"gpt-4o-2024-11-20": fake_llm},
    tool_executors={"create_plan": fake_tool, "execute_task": fake_tool,
                    "verify_plan": fake_tool},
)
verify = next(s for s in result.steps if s.get("name") == "verify_plan")
print(verify["outputs"]["result"]["achieved"])  # → False
```

## What to learn here

- Substituting a mid-plan execution result propagates dirtiness to every
  subsequent step (later tasks, summarise, verify, final report).
- `step_back` lets you inspect the agent state after any individual step
  without re-running the entire trace from scratch.
