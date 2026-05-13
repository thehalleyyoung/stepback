"""Record the parallel_branch reference trace."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from stepback import record
from stepback.testing.parallel_agent import run_parallel_agent

TRACE_PATH = os.path.join(os.path.dirname(__file__), "trace.sb")


def main() -> None:
    with record(TRACE_PATH) as rec:
        run_parallel_agent(rec)
    print(f"parallel_branch: trace written to {TRACE_PATH}")


if __name__ == "__main__":
    main()
