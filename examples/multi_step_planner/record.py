"""Record the multi_step_planner reference trace."""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", ".."))  # repo root

from stepback import record  # noqa: E402

# Import agent module directly to avoid relying on the examples package.
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("msp_agent", os.path.join(_HERE, "agent.py"))
_mod = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

TRACE_PATH = _mod.TRACE_PATH
run_planner_agent = _mod.run_planner_agent


def main() -> None:
    with record(TRACE_PATH) as rec:
        run_planner_agent(rec)
    print(f"multi_step_planner: trace written to {TRACE_PATH}")


if __name__ == "__main__":
    main()
