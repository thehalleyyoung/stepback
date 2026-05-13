"""Record the tool_using_agent reference trace."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from stepback import record
from stepback.testing.agent import run_recorded_agent

TRACE_PATH = os.path.join(os.path.dirname(__file__), "trace.sb")


def main() -> None:
    with record(TRACE_PATH) as rec:
        run_recorded_agent(rec)
    print(f"tool_using_agent: trace written to {TRACE_PATH}")


if __name__ == "__main__":
    main()
