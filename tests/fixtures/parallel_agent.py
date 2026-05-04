"""Back-compat shim: the canonical fixture lives at
``stepback.testing.parallel_agent``.

New code should import from :mod:`stepback.testing` instead. See
Step 20 of ``100_STEPS.md``.
"""
from __future__ import annotations

from stepback.testing.parallel_agent import (  # noqa: F401
    FACTS,
    fake_llm,
    fake_tool,
    run_parallel_agent,
)

__all__ = [
    "FACTS",
    "fake_llm",
    "fake_tool",
    "run_parallel_agent",
]
