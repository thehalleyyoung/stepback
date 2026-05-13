"""Multi-step planning agent.

Simulates an agent that decomposes a high-level goal into sub-tasks, executes
each sub-task, and verifies the outcome:

  1. LLM decomposes the goal into a list of sub-tasks.
  2. ``create_plan`` tool persists the plan and returns a plan id.
  3. For each sub-task (three tasks):
       a. LLM generates the action for that sub-task.
       b. ``execute_task`` tool runs the action and returns a result.
  4. LLM summarises all results.
  5. ``verify_plan`` tool checks whether the goal was achieved.
  6. LLM produces the final report.

Agent pattern: **plan → execute × N → verify**.

Demonstrates:
- Recording a planner trace with repeated execution cycles.
- Substituting one sub-task execution result and observing that subsequent
  sub-tasks and the verification are marked dirty.
- Navigating backward with ``step_back`` to inspect mid-plan state.

Run directly::

    python examples/multi_step_planner/agent.py

Or via Make::

    make -C examples multi_step_planner
"""
from __future__ import annotations

import hashlib
import os
import sys
from typing import Any, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from stepback import Executor, record, replay
from stepback.substitutions import ToolOutputSubstitution

TRACE_PATH = os.path.join(os.path.dirname(__file__), "trace.sb")

_GOAL = "Set up a new development environment: install deps, run tests, deploy staging."

_SUBTASKS = [
    {"id": "task-1", "action": "install_dependencies",
     "description": "Install all project dependencies from requirements.txt"},
    {"id": "task-2", "action": "run_tests",
     "description": "Execute the full test suite and ensure all tests pass"},
    {"id": "task-3", "action": "deploy_staging",
     "description": "Deploy the current build to the staging environment"},
]

# Scripted results for each task action.
_TASK_RESULTS = {
    "install_dependencies": {
        "status": "success", "packages_installed": 42, "warnings": 0,
    },
    "run_tests": {
        "status": "success", "tests_run": 317, "failed": 0, "skipped": 3,
    },
    "deploy_staging": {
        "status": "success", "url": "https://staging.example.com", "build_id": "b-9821",
    },
}

# Counterfactual: tests fail.
_FAILED_TEST_RESULT = {
    "status": "failed", "tests_run": 317, "failed": 7, "skipped": 3,
    "failure_summary": "7 test failures in auth module",
}


def _digest(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Fake LLM
# ---------------------------------------------------------------------------

def fake_llm(model: str, messages: List[dict]) -> dict:
    """Deterministic LLM stand-in."""
    blob = "\n".join(f"{m['role']}:{m.get('content','')}" for m in messages)
    digest = _digest(blob)
    last = messages[-1].get("content", "") if messages else ""

    if "decompose" in last.lower() or "sub-task" in last.lower():
        text = f"Sub-tasks: install deps, run tests, deploy staging. ({digest})"
    elif "summarise" in last.lower() or "summary" in last.lower():
        text = f"All sub-tasks completed successfully. ({digest})"
    elif "report" in last.lower() or "final" in last.lower():
        text = f"Goal achieved: environment ready. ({digest})"
    elif "action" in last.lower() or "execute" in last.lower():
        text = f"Executing next sub-task. ({digest})"
    else:
        text = f"reply-{digest}"

    return {
        "id": f"chatcmpl-{digest}",
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": text}}],
        "usage": {
            "prompt_tokens": sum(len(m.get("content", "")) for m in messages),
            "completion_tokens": len(text),
            "total_tokens": sum(len(m.get("content", "")) for m in messages) + len(text),
        },
    }


# ---------------------------------------------------------------------------
# Fake tools
# ---------------------------------------------------------------------------

def fake_tool(name: str, args: dict) -> Any:
    """Tool stand-ins for the multi-step planner."""
    if name == "create_plan":
        return {
            "plan_id": f"plan-{_digest(str(args))}",
            "subtasks": args.get("subtasks", []),
            "status": "created",
        }

    if name == "execute_task":
        action = args.get("action", "")
        return _TASK_RESULTS.get(action, {"status": "unknown", "action": action})

    if name == "verify_plan":
        results = args.get("results", [])
        all_ok = all(r.get("status") == "success" for r in results)
        return {
            "plan_id": args.get("plan_id", ""),
            "achieved": all_ok,
            "summary": "All tasks completed." if all_ok else "Some tasks failed.",
            "failed_tasks": [r["action"] for r in results if r.get("status") != "success"],
        }

    raise KeyError(f"unknown planner tool: {name!r}")


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

def run_planner_agent(rec: Any) -> None:
    """Drive an 8-step planner agent through *rec*."""
    model = "gpt-4o-2024-11-20"
    convo = [
        {"role": "system", "content": "You are a planning agent."},
        {"role": "user", "content": f"Goal: {_GOAL}"},
    ]

    # 1) LLM decomposes goal.
    rec.llm_call(
        model,
        convo + [{"role": "user", "content": "decompose this goal into sub-tasks"}],
        executor=fake_llm,
    )

    # 2) Create plan tool.
    plan = rec.tool_call("create_plan",
                         {"goal": _GOAL, "subtasks": _SUBTASKS},
                         executor=fake_tool)
    plan_id = plan["outputs"]["result"]["plan_id"]

    # 3–8) Execute each sub-task (llm_call + execute_task × 3).
    task_results = []
    for task in _SUBTASKS:
        rec.llm_call(
            model,
            convo + [{"role": "assistant", "content": f"executing {task['action']}"}],
            executor=fake_llm,
        )
        result = rec.tool_call("execute_task",
                               {"action": task["action"],
                                "description": task["description"]},
                               executor=fake_tool)
        task_results.append({
            "action": task["action"],
            **result["outputs"]["result"],
        })

    # 9) LLM summarises.
    rec.llm_call(
        model,
        convo + [{"role": "assistant",
                  "content": f"results: {task_results}"},
                 {"role": "user", "content": "summarise the results"}],
        executor=fake_llm,
    )

    # 10) Verify plan.
    rec.tool_call("verify_plan",
                  {"plan_id": plan_id, "results": task_results},
                  executor=fake_tool)

    # 11) Final report.
    rec.llm_call(
        model,
        convo + [{"role": "assistant", "content": "verification done"},
                 {"role": "user", "content": "produce final report"}],
        executor=fake_llm,
    )


# ---------------------------------------------------------------------------
# Record & replay demo
# ---------------------------------------------------------------------------

def record_trace(path: str = TRACE_PATH) -> None:
    with record(path) as rec:
        run_planner_agent(rec)
    print(f"  recorded → {path}")


def replay_demo(path: str = TRACE_PATH) -> None:
    """Substitute 'run_tests' result with a failure and observe dirty set."""
    trace = replay(path)
    steps = trace.recorded_steps

    test_step = next(
        s for s in steps
        if s.get("name") == "execute_task"
        and s.get("inputs", {}).get("arguments", {}).get("action") == "run_tests"
    )

    trace.substitute(
        ToolOutputSubstitution(
            at_step=test_step["step_id"],
            fake_response=_FAILED_TEST_RESULT,
        )
    )

    result = trace.replay_forward(
        Executor(llm=fake_llm, tool=fake_tool)
    )
    dirty = [sv for sv in result.steps if sv.dirty]
    print(f"  dirty steps after test failure substitution: {len(dirty)}")

    verify_step = next(
        (sv for sv in result.steps if sv.name == "verify_plan"), None
    )
    if verify_step:
        achieved = verify_step.outputs.get("result", {}).get("achieved") if isinstance(verify_step.outputs, dict) else None
        print(f"  plan achieved: {achieved}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== multi_step_planner example ===")
    print("Recording…")
    record_trace()
    print("Replaying with substitution…")
    replay_demo()
    print("Done.")
