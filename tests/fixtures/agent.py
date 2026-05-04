"""Back-compat shim: the canonical fixture lives at ``stepback.testing.agent``.

This module re-exports the public fixture symbols so that any pre-existing
import of ``tests.fixtures.agent`` continues to work for one deprecation
window. New code should import from :mod:`stepback.testing` instead::

    # old
    from stepback.testing import run_recorded_agent

    # new
    from stepback.testing import run_recorded_agent

See Step 20 of ``100_STEPS.md``.
"""
from __future__ import annotations

from stepback.testing.agent import (  # noqa: F401
    CUSTOMER_DB,
    LOOKUP_BUG_ROW,
    LOOKUP_FIXED_ROW,
    fake_llm,
    fake_tool,
    run_recorded_agent,
)

__all__ = [
    "CUSTOMER_DB",
    "LOOKUP_BUG_ROW",
    "LOOKUP_FIXED_ROW",
    "fake_llm",
    "fake_tool",
    "run_recorded_agent",
]
