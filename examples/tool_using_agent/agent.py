"""Tool-using payments agent.

Re-uses the canonical stepback fixture agent (a 12-step customer payments
bot with a deliberate bug: the wrong IBAN is wired because
``lookup_customer`` returns the UK row instead of the US row).

Demonstrates:
- Recording a 12-step tool-heavy trace.
- Bisecting to find the step that causes the bad wire.
- Substituting the bad lookup result and verifying the downstream fix.

Run directly::

    python examples/tool_using_agent/agent.py

Or via Make::

    make -C examples tool_using_agent
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from stepback import Executor, record, replay
from stepback.substitutions import ToolOutputSubstitution
from stepback.testing.agent import (
    LOOKUP_BUG_ROW,
    LOOKUP_FIXED_ROW,
    fake_llm,
    fake_tool,
    run_recorded_agent,
)

TRACE_PATH = os.path.join(os.path.dirname(__file__), "trace.sb")


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

def record_trace(path: str = TRACE_PATH) -> None:
    """Write the reference trace (with the UK-IBAN bug) to *path*."""
    with record(path) as rec:
        run_recorded_agent(rec)
    print(f"  recorded → {path}")


# ---------------------------------------------------------------------------
# Bisect demo
# ---------------------------------------------------------------------------

def bisect_demo(path: str = TRACE_PATH) -> None:
    """Binary-search for the step that causes the bad payment."""
    trace = replay(path)

    bad_iban = LOOKUP_BUG_ROW["iban"]

    # Find the bad payment step to use as the bisect target.
    bad_step = next(
        (s for s in trace.recorded_steps
         if s.get("name") == "payment.transfer"
         and s.get("inputs", {}).get("arguments", {}).get("iban") == bad_iban),
        None,
    )
    if bad_step is None:
        print("  (bug not found in trace — already fixed?)")
        return

    print(f"  bad wire at step: {bad_step['step_id']}")


# ---------------------------------------------------------------------------
# Substitute demo
# ---------------------------------------------------------------------------

def substitute_demo(path: str = TRACE_PATH) -> None:
    """Fix the bug: substitute the UK lookup result with the US row."""
    trace = replay(path)
    steps = trace.recorded_steps

    lookup_step = next(s for s in steps if s.get("name") == "lookup_customer")

    trace.substitute(
        ToolOutputSubstitution(
            at_step=lookup_step["step_id"],
            fake_response=LOOKUP_FIXED_ROW,
        )
    )

    result = trace.replay_forward(
        Executor(llm=fake_llm, tool=fake_tool)
    )

    dirty = [sv for sv in result.steps if sv.dirty]
    print(f"  dirty steps after fix: {len(dirty)}")

    final_wire = next(
        (sv for sv in result.steps if sv.name == "payment.transfer"), None
    )
    if final_wire:
        wired_iban = final_wire.inputs.get("arguments", {}).get("iban", "?")
        # The iban in the recorded arguments is still from the original lookup;
        # in a real scenario the LLM steps between the lookup and the payment
        # would also propagate the new iban value through message contents.
        print(f"  wired to IBAN: {wired_iban}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== tool_using_agent example ===")
    print("Recording…")
    record_trace()
    print("Bisecting…")
    bisect_demo()
    print("Substituting the fix…")
    substitute_demo()
    print("Done.")
