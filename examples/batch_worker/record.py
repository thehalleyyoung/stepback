"""Record the batch_worker reference trace."""
from __future__ import annotations

import importlib.util as _ilu
import os
import sys

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", ".."))  # repo root

from stepback import record  # noqa: E402

_spec = _ilu.spec_from_file_location("bw_agent", os.path.join(_HERE, "agent.py"))
_mod = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

TRACE_PATH = _mod.TRACE_PATH
run_batch_agent = _mod.run_batch_agent


def main() -> None:
    with record(TRACE_PATH) as rec:
        run_batch_agent(rec)
    print(f"batch_worker: trace written to {TRACE_PATH}")


if __name__ == "__main__":
    main()
