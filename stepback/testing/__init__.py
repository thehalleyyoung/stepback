"""Public deterministic fixture agents for stepback users and tests.

This package promotes the previously-private ``tests.fixtures.*`` agents
to a stable, semver-covered location under ``stepback.testing`` so that
downstream users (and stepback's own tests) no longer have to reach
into the test tree to get a hands-on, reproducible toy agent for
record/replay/substitute experiments.

Two fixture agents are provided:

* :func:`stepback.testing.run_recorded_agent` — a 12-step deterministic
  "customer payments bot" (6 ``llm_call`` + 6 ``tool_call`` steps,
  alternating) where the recorded run produces a buggy wire transfer
  because ``lookup_customer`` returns the wrong row. This is the
  canonical fixture used throughout the docs and benchmark scripts.

* :func:`stepback.testing.run_parallel_agent` — an 11-step research
  agent demonstrating ``parallel_branch_open`` / ``parallel_branch_join``
  with three sibling fan-out branches and a downstream synthesise step
  that depends on the merged join output (so a substitution inside one
  branch propagates through the join into the synthesise call).

Both agents are deterministic, network-free, and use small in-process
fakes for the LLM and tool executors. They are designed to be the
default fixture for any user trying ``stepback.record`` / ``replay``
without having to bring an LLM provider key or a real tool stack.

Example::

    from stepback import record, replay
    from stepback.testing import run_recorded_agent

    with record("./trace.sb") as rec:
        run_recorded_agent(rec)

    trace = replay("./trace.sb")

In-process replay utilities are exposed under :mod:`stepback.testing.replay`
and re-exported here:

* :class:`stepback.testing.CaptureExecutor` — records every successful executor
  callback invocation for test assertions.
* :class:`stepback.testing.FallbackExecutor` — ``Executor(fallback_recorded=True)``
  alias for local debugging without a real LLM/tool stack.
* Assertion helpers: :func:`assert_all_cache_hits`, :func:`assert_dirty_count`,
  :func:`assert_real_executions`, :func:`assert_cache_hit_count`,
  :func:`assert_step_dirty`, :func:`assert_step_clean`.

The exposed API is part of the v0.1 semver contract; new fixtures may
be added but existing ones will not be renamed or removed without a
deprecation cycle.
"""
from __future__ import annotations

from .agent import (
    CUSTOMER_DB,
    LOOKUP_BUG_ROW,
    LOOKUP_FIXED_ROW,
    fake_llm,
    fake_tool,
    run_recorded_agent,
)
from .parallel_agent import (
    FACTS,
    fake_llm as parallel_fake_llm,
    fake_tool as parallel_fake_tool,
    run_parallel_agent,
)
from .replay import (
    CapturedCall,
    CaptureExecutor,
    FallbackExecutor,
    assert_all_cache_hits,
    assert_cache_hit_count,
    assert_dirty_count,
    assert_real_executions,
    assert_step_clean,
    assert_step_dirty,
)
from .support_agent import SUPPORT_TASKS, run_support_task
from .code_review_agent import CODE_REVIEW_TASKS, run_code_review_task
from .payments_policy_agent import PAYMENTS_TASKS, run_payments_task

__all__ = [
    "CUSTOMER_DB",
    "LOOKUP_BUG_ROW",
    "LOOKUP_FIXED_ROW",
    "fake_llm",
    "fake_tool",
    "run_recorded_agent",
    "FACTS",
    "parallel_fake_llm",
    "parallel_fake_tool",
    "run_parallel_agent",
    # In-process replay utilities (Step 70)
    "CapturedCall",
    "CaptureExecutor",
    "FallbackExecutor",
    "assert_all_cache_hits",
    "assert_cache_hit_count",
    "assert_dirty_count",
    "assert_real_executions",
    "assert_step_clean",
    "assert_step_dirty",
    # Author corpus agents (Step 116)
    "SUPPORT_TASKS",
    "run_support_task",
    "CODE_REVIEW_TASKS",
    "run_code_review_task",
    "PAYMENTS_TASKS",
    "run_payments_task",
]
