"""Tests for the public ``stepback.testing`` fixture-agent module (Step 20).

These tests exercise the promoted-out-of-tests fixture agents through
their stable public import path so that downstream users have a
maintained contract: ``from stepback.testing import run_recorded_agent``.

The same fixtures are exercised heavily through their previous private
location elsewhere in the suite; this module specifically guards the
*public* surface so a future move/rename of the fixture would be
flagged loudly here.
"""
from __future__ import annotations

import os
import tempfile

import pytest

import stepback
from stepback import record, replay
from stepback.recorder import RecorderKey


def test_public_testing_module_exposes_recorded_agent() -> None:
    """``stepback.testing`` must expose the canonical 12-step fixture."""
    import stepback.testing as t

    expected = {
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
    }
    assert expected.issubset(set(t.__all__)), (
        f"stepback.testing.__all__ missing required names: "
        f"{sorted(expected - set(t.__all__))}"
    )
    for name in expected:
        assert hasattr(t, name), f"stepback.testing missing public symbol {name!r}"


def test_public_testing_module_is_listed_in_stepback_all() -> None:
    """``testing`` must be a documented submodule of the ``stepback`` package."""
    assert "testing" in stepback.__all__
    assert hasattr(stepback, "testing")
    assert stepback.testing.__doc__ and len(stepback.testing.__doc__.strip()) > 20


def test_run_recorded_agent_via_public_path_records_12_steps(tmp_path) -> None:
    """End-to-end smoke: the public fixture records exactly 12 verifiable steps."""
    from stepback.testing import run_recorded_agent
    from stepback.trace_reader import verify_trace

    p = tmp_path / "public.sb"
    key = RecorderKey.fresh()
    with record(str(p), key=key) as rec:
        run_recorded_agent(rec)

    verified = verify_trace(str(p), key.hmac_key)
    assert len(verified.steps) == 12
    kinds = [s["step_kind"] for s in verified.steps]
    assert kinds == ["llm_call", "tool_call"] * 6


def test_run_parallel_agent_via_public_path(tmp_path) -> None:
    """End-to-end smoke: the public parallel fixture records 11 steps with a join."""
    from stepback.testing.parallel_agent import run_parallel_agent
    from stepback.trace_reader import verify_trace

    p = tmp_path / "public_parallel.sb"
    key = RecorderKey.fresh()
    with record(str(p), key=key) as rec:
        run_parallel_agent(rec)

    verified = verify_trace(str(p), key.hmac_key)
    assert len(verified.steps) == 11
    kinds = [s["step_kind"] for s in verified.steps]
    assert "parallel_branch_open" in kinds
    assert "parallel_branch_join" in kinds


def test_back_compat_shim_re_exports_from_public_module() -> None:
    """``tests.fixtures.agent`` still re-exports the same callables (one-window shim).

    This guards the deprecation window promised in the docstring of
    ``tests/fixtures/agent.py``: the old import path must still produce
    the *same* callable objects as the new public path.
    """
    from stepback.testing import (
        CUSTOMER_DB as new_db,
        run_recorded_agent as new_runner,
        fake_llm as new_llm,
        fake_tool as new_tool,
    )
    from tests.fixtures.agent import (
        CUSTOMER_DB as old_db,
        run_recorded_agent as old_runner,
        fake_llm as old_llm,
        fake_tool as old_tool,
    )

    assert old_db is new_db
    assert old_runner is new_runner
    assert old_llm is new_llm
    assert old_tool is new_tool


def test_public_path_does_not_depend_on_tests_package() -> None:
    """``stepback.testing`` must be importable without the ``tests`` package on path.

    This is the whole point of Step 20: a user installing ``stepback``
    from a wheel does not have ``tests/`` available, so the public
    fixture must work standalone.
    """
    import importlib
    import sys

    # Drop any cached `tests` module so a fresh import of stepback.testing
    # cannot accidentally satisfy itself by reaching back into the test tree.
    for mod_name in [m for m in list(sys.modules) if m == "tests" or m.startswith("tests.")]:
        sys.modules.pop(mod_name, None)

    mod = importlib.import_module("stepback.testing")
    # Re-import its submodules and confirm they expose the runner without
    # needing the test tree.
    agent = importlib.import_module("stepback.testing.agent")
    parallel = importlib.import_module("stepback.testing.parallel_agent")
    assert callable(mod.run_recorded_agent)
    assert callable(agent.run_recorded_agent)
    assert callable(parallel.run_parallel_agent)
