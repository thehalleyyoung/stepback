"""Customer-support agent example.

Demonstrates:
- Recording a linear multi-tool agent.
- Replaying with a ToolOutputSubstitution to change the order lookup result.
- Inspecting which steps become dirty after the substitution.

Run this file directly to re-record the reference trace::

    python examples/customer_support/agent.py

Or use the Makefile::

    make -C examples customer_support
"""
from __future__ import annotations

import os
import sys

# Allow running from the repo root without a formal install.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from stepback import Executor, record, replay
from stepback.substitutions import ToolOutputSubstitution
from stepback.testing.support_agent import fake_llm, fake_tool, run_support_task

TRACE_PATH = os.path.join(os.path.dirname(__file__), "trace.sb")


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

def record_trace(path: str = TRACE_PATH) -> None:
    """Record the 'order-status-delayed' support task to *path*."""
    with record(path) as rec:
        run_support_task(rec, "order-status-delayed")
    print(f"  recorded → {path}")


# ---------------------------------------------------------------------------
# Replay & substitute
# ---------------------------------------------------------------------------

def replay_demo(path: str = TRACE_PATH) -> None:
    """Show a substitution: pretend the order arrived on time."""
    trace = replay(path)
    steps = trace.recorded_steps

    # Find the check_shipping tool step.
    shipping_step = next(
        s for s in steps if s.get("name") == "check_shipping"
    )

    # Counterfactual: shipping is on time.
    on_time_output = {"status": "on_time", "days_overdue": 0, "carrier": "FedEx",
                      "tracking": "TRK-on-time"}

    trace.substitute(
        ToolOutputSubstitution(
            at_step=shipping_step["step_id"],
            fake_response=on_time_output,
        )
    )

    result = trace.replay_forward(
        Executor(llm=fake_llm, tool=fake_tool)
    )

    dirty = [sv for sv in result.steps if sv.dirty]
    print(f"  dirty steps after substitution: {len(dirty)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== customer_support example ===")
    print("Recording…")
    record_trace()
    print("Replaying with substitution…")
    replay_demo()
    print("Done.")
