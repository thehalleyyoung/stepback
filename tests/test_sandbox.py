"""Tests for stepback/sandbox.py (Step 72 — sandboxed replay modes).

All tests are offline; no network calls are made.
"""
from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import stepback
from stepback.replay import Executor, MissingExecutor
from stepback.sandbox import (
    SandboxConfig,
    SandboxError,
    SandboxMode,
    SandboxResourceError,
    SandboxTimeoutError,
    SandboxUnavailableError,
    SandboxViolationError,
    SandboxedExecutor,
    create_sandbox,
)


# ----------------------------------------------------------------- helpers


def _simple_tool(name: str, arguments: dict) -> Any:
    """Module-level tool function (picklable on all platforms)."""
    if name == "echo":
        return arguments.get("value", "")
    if name == "add":
        return arguments["a"] + arguments["b"]
    if name == "sleep":
        import time
        time.sleep(arguments.get("seconds", 0))
        return "done"
    raise ValueError(f"Unknown tool: {name}")


def _simple_llm(model: str, messages: list) -> dict:
    return {"choices": [{"message": {"content": "ok"}}], "usage": {}}


def _make_base(tool=_simple_tool, llm=_simple_llm):
    return Executor(llm=llm, tool=tool)


# ================================================================= SandboxMode


class TestSandboxMode:
    def test_enum_has_four_members(self):
        assert len(SandboxMode) == 4

    def test_string_values(self):
        assert SandboxMode.NONE == "none"
        assert SandboxMode.SUBPROCESS == "subprocess"
        assert SandboxMode.GVISOR == "gvisor"
        assert SandboxMode.FIRECRACKER == "firecracker"

    def test_is_str_subclass(self):
        assert isinstance(SandboxMode.NONE, str)


# ================================================================= SandboxConfig


class TestSandboxConfig:
    def test_defaults(self):
        cfg = SandboxConfig()
        assert cfg.mode == SandboxMode.NONE
        assert cfg.timeout_sec == 30.0
        assert cfg.memory_limit_mb == 512
        assert cfg.cpu_limit_sec == 60
        assert cfg.network_access is False
        assert cfg.gvisor_runsc_path == "runsc"
        assert cfg.firecracker_binary_path == "firecracker"
        assert cfg.firecracker_kernel_path is None
        assert cfg.firecracker_rootfs_path is None
        assert cfg.tool_runner_argv is None

    def test_custom_values(self):
        cfg = SandboxConfig(
            mode=SandboxMode.SUBPROCESS,
            timeout_sec=5.0,
            memory_limit_mb=128,
            cpu_limit_sec=10,
            network_access=True,
        )
        assert cfg.mode == SandboxMode.SUBPROCESS
        assert cfg.timeout_sec == 5.0
        assert cfg.memory_limit_mb == 128
        assert cfg.cpu_limit_sec == 10
        assert cfg.network_access is True


# ================================================================= Exceptions


class TestSandboxExceptions:
    def test_hierarchy(self):
        assert issubclass(SandboxTimeoutError, SandboxError)
        assert issubclass(SandboxResourceError, SandboxError)
        assert issubclass(SandboxUnavailableError, SandboxError)
        assert issubclass(SandboxViolationError, SandboxError)
        assert issubclass(SandboxError, RuntimeError)

    def test_instantiable(self):
        for cls in [
            SandboxError,
            SandboxTimeoutError,
            SandboxResourceError,
            SandboxUnavailableError,
            SandboxViolationError,
        ]:
            e = cls("test message")
            assert "test message" in str(e)


# ================================================================= SandboxedExecutor


class TestSandboxedExecutorInterface:
    def test_is_executor_subclass(self):
        se = SandboxedExecutor(_make_base())
        assert isinstance(se, Executor)

    def test_real_calls_starts_at_zero(self):
        se = SandboxedExecutor(_make_base())
        assert se.real_calls == 0

    def test_base_real_calls_not_incremented_for_non_tool(self):
        base = _make_base()
        se = SandboxedExecutor(base, SandboxConfig(mode=SandboxMode.NONE))
        se.execute("llm_call", {"model": "x", "messages": []})
        assert base.real_calls == 0  # base.execute() was NOT called
        assert se.real_calls == 1

    def test_fallback_recorded_propagated(self):
        base = Executor(fallback_recorded=True)
        se = SandboxedExecutor(base)
        assert se.fallback_recorded is True


# ================================================================= NONE mode


class TestNoneMode:
    def setup_method(self):
        self.cfg = SandboxConfig(mode=SandboxMode.NONE)
        self.base = _make_base()
        self.se = SandboxedExecutor(self.base, self.cfg)

    def test_tool_call_delegates_directly(self):
        result = self.se.execute("tool_call", {"name": "echo", "arguments": {"value": "hello"}})
        assert result == {"result": "hello"}

    def test_tool_call_add(self):
        result = self.se.execute("tool_call", {"name": "add", "arguments": {"a": 3, "b": 4}})
        assert result == {"result": 7}

    def test_llm_call_delegates(self):
        result = self.se.execute("llm_call", {"model": "gpt-4", "messages": []})
        assert "choices" in result

    def test_router_delegates(self):
        base = Executor(router=lambda name, opts: opts[0])
        se = SandboxedExecutor(base, self.cfg)
        result = se.execute("router", {"name": "r", "options": ["a", "b"]})
        assert result == {"choice": "a"}

    def test_parallel_branch_open(self):
        result = self.se.execute(
            "parallel_branch_open",
            {"branch_names": ["x", "y"], "branch_count": 2},
        )
        assert result == {"branch_names": ["x", "y"], "branch_count": 2}

    def test_parallel_branch_join_no_callback(self):
        result = self.se.execute(
            "parallel_branch_join",
            {},
            branch_outputs=["a", "b"],
        )
        assert result == {"branches": ["a", "b"]}

    def test_exception_step(self):
        result = self.se.execute("exception", {})
        assert result["error_class"] == "Replayed"

    def test_unknown_kind_raises(self):
        with pytest.raises(MissingExecutor):
            self.se.execute("nonexistent_kind", {})

    def test_tool_missing_raises(self):
        base = Executor(llm=_simple_llm, tool=None)
        se = SandboxedExecutor(base, self.cfg)
        with pytest.raises(MissingExecutor):
            se.execute("tool_call", {"name": "echo", "arguments": {}})

    def test_real_calls_incremented(self):
        self.se.execute("tool_call", {"name": "echo", "arguments": {"value": "x"}})
        self.se.execute("tool_call", {"name": "echo", "arguments": {"value": "y"}})
        assert self.se.real_calls == 2


# ================================================================= SUBPROCESS mode


class TestSubprocessMode:
    def setup_method(self):
        self.cfg = SandboxConfig(
            mode=SandboxMode.SUBPROCESS,
            timeout_sec=15.0,
            memory_limit_mb=256,
            cpu_limit_sec=10,
        )
        self.base = _make_base()
        self.se = SandboxedExecutor(self.base, self.cfg)

    def test_echo_tool(self):
        result = self.se.execute(
            "tool_call", {"name": "echo", "arguments": {"value": "subprocess-test"}}
        )
        assert result == {"result": "subprocess-test"}

    def test_add_tool(self):
        result = self.se.execute(
            "tool_call", {"name": "add", "arguments": {"a": 10, "b": 20}}
        )
        assert result == {"result": 30}

    def test_real_calls_incremented(self):
        self.se.execute("tool_call", {"name": "echo", "arguments": {"value": "x"}})
        assert self.se.real_calls == 1

    def test_non_tool_bypasses_sandbox(self):
        """LLM calls bypass the subprocess and execute directly."""
        result = self.se.execute("llm_call", {"model": "gpt-4", "messages": []})
        assert "choices" in result

    def test_timeout_raises_sandbox_timeout_error(self):
        cfg = SandboxConfig(
            mode=SandboxMode.SUBPROCESS,
            timeout_sec=0.1,  # 100ms — will fire before sleep(5)
        )
        base = _make_base(tool=_simple_tool)
        se = SandboxedExecutor(base, cfg)
        with pytest.raises(SandboxTimeoutError):
            se.execute("tool_call", {"name": "sleep", "arguments": {"seconds": 5}})

    def test_tool_exception_propagates(self):
        """Exceptions from the tool function surface in the parent."""
        def bad_tool(name, args):
            raise ValueError("intentional tool failure")

        base = _make_base(tool=bad_tool)
        se = SandboxedExecutor(base, self.cfg)
        with pytest.raises(Exception):
            se.execute("tool_call", {"name": "bad", "arguments": {}})


# ================================================================= GVISOR mode


class TestGVisorMode:
    """gVisor tests operate offline by monkeypatching shutil.which."""

    def test_runsc_not_found_raises_unavailable(self):
        cfg = SandboxConfig(
            mode=SandboxMode.GVISOR,
            tool_runner_argv=["python3", "-m", "mytool"],
        )
        se = SandboxedExecutor(_make_base(), cfg)
        with patch("stepback.sandbox.shutil.which", return_value=None):
            with pytest.raises(SandboxUnavailableError) as exc_info:
                se.execute("tool_call", {"name": "echo", "arguments": {}})
        assert "runsc" in str(exc_info.value)

    def test_missing_tool_runner_argv_raises_unavailable(self):
        cfg = SandboxConfig(
            mode=SandboxMode.GVISOR,
            tool_runner_argv=None,  # missing
        )
        se = SandboxedExecutor(_make_base(), cfg)
        with patch("stepback.sandbox.shutil.which", return_value="/usr/bin/runsc"):
            with pytest.raises(SandboxUnavailableError) as exc_info:
                se.execute("tool_call", {"name": "echo", "arguments": {}})
        assert "tool_runner_argv" in str(exc_info.value)

    def test_gvisor_runsc_path_in_error_message(self):
        cfg = SandboxConfig(
            mode=SandboxMode.GVISOR,
            gvisor_runsc_path="/custom/runsc",
            tool_runner_argv=["mytool"],
        )
        se = SandboxedExecutor(_make_base(), cfg)
        with patch("stepback.sandbox.shutil.which", return_value=None):
            with pytest.raises(SandboxUnavailableError) as exc_info:
                se.execute("tool_call", {"name": "echo", "arguments": {}})
        assert "/custom/runsc" in str(exc_info.value)

    def test_gvisor_config_preserved(self):
        cfg = SandboxConfig(
            mode=SandboxMode.GVISOR,
            gvisor_runsc_path="/usr/local/bin/runsc",
            tool_runner_argv=["my_runner"],
        )
        se = SandboxedExecutor(_make_base(), cfg)
        assert se._config.gvisor_runsc_path == "/usr/local/bin/runsc"
        assert se._config.tool_runner_argv == ["my_runner"]


# ================================================================= FIRECRACKER mode


class TestFirecrackerMode:
    """Firecracker tests operate offline by monkeypatching shutil.which and os.path.exists."""

    def test_firecracker_not_found_raises_unavailable(self):
        cfg = SandboxConfig(
            mode=SandboxMode.FIRECRACKER,
            firecracker_binary_path="firecracker",
            firecracker_kernel_path="/vmlinux",
            firecracker_rootfs_path="/rootfs.ext4",
            tool_runner_argv=["mytool"],
        )
        se = SandboxedExecutor(_make_base(), cfg)
        with patch("stepback.sandbox.shutil.which", return_value=None):
            with pytest.raises(SandboxUnavailableError) as exc_info:
                se.execute("tool_call", {"name": "echo", "arguments": {}})
        assert "firecracker" in str(exc_info.value).lower()

    def test_missing_kernel_path_raises_unavailable(self):
        cfg = SandboxConfig(
            mode=SandboxMode.FIRECRACKER,
            firecracker_kernel_path=None,  # missing
            firecracker_rootfs_path="/rootfs.ext4",
            tool_runner_argv=["mytool"],
        )
        se = SandboxedExecutor(_make_base(), cfg)
        with patch("stepback.sandbox.shutil.which", return_value="/usr/bin/firecracker"), \
             patch("stepback.sandbox.os.path.exists", return_value=True):
            with pytest.raises(SandboxUnavailableError) as exc_info:
                se.execute("tool_call", {"name": "echo", "arguments": {}})
        assert "kernel" in str(exc_info.value).lower()

    def test_missing_rootfs_path_raises_unavailable(self):
        cfg = SandboxConfig(
            mode=SandboxMode.FIRECRACKER,
            firecracker_kernel_path="/vmlinux",
            firecracker_rootfs_path=None,  # missing
            tool_runner_argv=["mytool"],
        )
        se = SandboxedExecutor(_make_base(), cfg)
        with patch("stepback.sandbox.shutil.which", return_value="/usr/bin/firecracker"), \
             patch("stepback.sandbox.os.path.exists", return_value=True):
            with pytest.raises(SandboxUnavailableError) as exc_info:
                se.execute("tool_call", {"name": "echo", "arguments": {}})
        assert "rootfs" in str(exc_info.value).lower()

    def test_missing_tool_runner_argv_raises_unavailable(self):
        cfg = SandboxConfig(
            mode=SandboxMode.FIRECRACKER,
            firecracker_kernel_path="/vmlinux",
            firecracker_rootfs_path="/rootfs.ext4",
            tool_runner_argv=None,  # missing
        )
        se = SandboxedExecutor(_make_base(), cfg)
        with patch("stepback.sandbox.shutil.which", return_value="/usr/bin/firecracker"), \
             patch("stepback.sandbox.os.path.exists", return_value=True):
            with pytest.raises(SandboxUnavailableError) as exc_info:
                se.execute("tool_call", {"name": "echo", "arguments": {}})
        assert "tool_runner_argv" in str(exc_info.value)


# ================================================================= create_sandbox factory


class TestCreateSandbox:
    def test_returns_sandboxed_executor(self):
        cfg = SandboxConfig(mode=SandboxMode.NONE)
        base = _make_base()
        se = create_sandbox(cfg, base)
        assert isinstance(se, SandboxedExecutor)

    def test_config_preserved(self):
        cfg = SandboxConfig(mode=SandboxMode.SUBPROCESS, timeout_sec=5.0)
        se = create_sandbox(cfg, _make_base())
        assert se._config.timeout_sec == 5.0

    def test_functional_with_none_mode(self):
        cfg = SandboxConfig(mode=SandboxMode.NONE)
        se = create_sandbox(cfg, _make_base())
        result = se.execute("tool_call", {"name": "echo", "arguments": {"value": "factory"}})
        assert result == {"result": "factory"}


# ================================================================= Public API


class TestPublicApi:
    def test_sandbox_symbols_importable_from_stepback(self):
        for name in [
            "SandboxMode",
            "SandboxConfig",
            "SandboxError",
            "SandboxTimeoutError",
            "SandboxResourceError",
            "SandboxUnavailableError",
            "SandboxViolationError",
            "SandboxedExecutor",
            "create_sandbox",
        ]:
            assert hasattr(stepback, name), f"stepback.{name} not exported"

    def test_sandbox_in_all(self):
        for name in [
            "SandboxMode",
            "SandboxConfig",
            "SandboxError",
            "SandboxTimeoutError",
            "SandboxResourceError",
            "SandboxUnavailableError",
            "SandboxViolationError",
            "SandboxedExecutor",
            "create_sandbox",
        ]:
            assert name in stepback.__all__, f"{name} missing from stepback.__all__"

    def test_sandboxed_executor_is_executor_subclass_via_public(self):
        assert issubclass(stepback.SandboxedExecutor, stepback.Executor)


# ================================================================= Integration


class TestIntegrationWithReplay:
    """Integration: replay_forward through a SandboxedExecutor."""

    def test_replay_with_none_sandbox(self):
        """NONE mode sandbox gives same results as a bare Executor."""
        from stepback import record, replay
        from stepback.recorder import RecorderKey
        from stepback.testing import run_recorded_agent
        import tempfile, pathlib

        key = RecorderKey.fresh()
        with tempfile.TemporaryDirectory() as td:
            trace_path = pathlib.Path(td) / "trace.sb"
            with record(str(trace_path), key=key) as rec:
                run_recorded_agent(rec)
            trace = replay(str(trace_path))

        base = Executor(fallback_recorded=True)
        se = SandboxedExecutor(base, SandboxConfig(mode=SandboxMode.NONE))
        result = trace.replay_forward(se)
        assert result.dirty_count == 0
        assert result.cache_hit_count > 0

    def test_replay_with_subprocess_sandbox(self):
        """SUBPROCESS mode gives correct results for clean (cache-hit) replay."""
        from stepback import record, replay
        from stepback.recorder import RecorderKey
        from stepback.testing import run_recorded_agent
        import tempfile, pathlib

        key = RecorderKey.fresh()
        with tempfile.TemporaryDirectory() as td:
            trace_path = pathlib.Path(td) / "trace.sb"
            with record(str(trace_path), key=key) as rec:
                run_recorded_agent(rec)
            trace = replay(str(trace_path))

        # Use fallback_recorded so no real tool calls are needed for dirty steps.
        base = Executor(fallback_recorded=True)
        se = SandboxedExecutor(
            base,
            SandboxConfig(mode=SandboxMode.SUBPROCESS, timeout_sec=30.0),
        )
        result = trace.replay_forward(se)
        # All steps should be cache hits since no substitutions were staged.
        assert result.dirty_count == 0
        assert result.cache_hit_count == len(result)
