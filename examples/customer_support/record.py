"""Record the customer_support reference trace.

Run from repo root::

    python examples/customer_support/record.py

Or via Make::

    make -C examples customer_support
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from stepback import record
from stepback.testing.support_agent import run_support_task

TRACE_PATH = os.path.join(os.path.dirname(__file__), "trace.sb")


def main() -> None:
    with record(TRACE_PATH) as rec:
        run_support_task(rec, "order-status-delayed")
    print(f"customer_support: trace written to {TRACE_PATH}")


if __name__ == "__main__":
    main()
