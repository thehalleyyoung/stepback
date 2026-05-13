"""Parallel-branch research agent.

Re-uses ``stepback.testing.parallel_agent`` — a fan-out / fan-in research
agent that launches three concurrent sub-investigations (history, economics,
politics) and merges the results.

Demonstrates:
- Recording a parallel-branch trace.
- Substituting one branch's tool output and observing that the synthesise
  step downstream of the join becomes dirty.
- Using ``branch_at`` to explore a counterfactual without mutating the main
  trace.

Run directly::

    python examples/parallel_branch/agent.py

Or via Make::

    make -C examples parallel_branch
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from stepback import Executor, record, replay
from stepback.substitutions import ToolOutputSubstitution
from stepback.testing.parallel_agent import (
    FACTS,
    fake_llm,
    fake_tool,
    run_parallel_agent,
)

TRACE_PATH = os.path.join(os.path.dirname(__file__), "trace.sb")

# Alternative economics fact for substitution demo.
_ALT_ECONOMICS = {"answer": "GDP 25T USD (revised)", "confidence": 0.85}


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

def record_trace(path: str = TRACE_PATH) -> None:
    with record(path) as rec:
        run_parallel_agent(rec)
    print(f"  recorded → {path}")


# ---------------------------------------------------------------------------
# Substitute demo
# ---------------------------------------------------------------------------

def replay_demo(path: str = TRACE_PATH) -> None:
    """Substitute the economics lookup and see the synthesise step go dirty."""
    trace = replay(path)
    steps = trace.recorded_steps

    econ_step = next(
        s for s in steps if s.get("name") == "lookup_economics"
    )

    trace.substitute(
        ToolOutputSubstitution(
            at_step=econ_step["step_id"],
            fake_response=_ALT_ECONOMICS,
        )
    )

    result = trace.replay_forward(
        Executor(llm=fake_llm, tool=fake_tool)
    )
    dirty = [sv for sv in result.steps if sv.dirty]
    print(f"  dirty steps after economics substitution: {len(dirty)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== parallel_branch example ===")
    print("Recording…")
    record_trace()
    print("Replaying with substitution…")
    replay_demo()
    print("Done.")
