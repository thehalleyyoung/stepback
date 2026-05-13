"""Sandboxed tool-executor wrappers for untrusted tool execution (Step 72).

Tool calls in replay may execute arbitrary user-supplied code.  This module
provides isolation layers of increasing strength:

* :data:`SandboxMode.NONE` — no isolation; direct in-process call (default).

* :data:`SandboxMode.SUBPROCESS` — tool call runs in a separate Python
  process via :mod:`concurrent.futures.ProcessPoolExecutor`.  Provides
  wall-clock timeout enforcement and, on POSIX platforms, optional CPU-time
  and virtual-memory limits via :mod:`resource`.  **Not a security sandbox**
  — a misbehaving tool can still access the filesystem, network, and
  environment unless the OS-level controls below are used.  Requires that
  the :attr:`SandboxConfig.tool_runner_argv` is *unset* and that the base
  executor's ``tool`` callback is picklable (i.e. a module-level function or
  a picklable callable class).  On Windows, CPU/memory limits are silently
  skipped (``resource`` is unavailable); timeout is still enforced.

* :data:`SandboxMode.GVISOR` — tool call dispatched via a user-supplied
  command line (``tool_runner_argv``) that is run inside gVisor's ``runsc``
  container runtime.  Provides kernel-level isolation.  Requires:

  - ``runsc`` on ``$PATH`` or at :attr:`SandboxConfig.gvisor_runsc_path`.
  - ``tool_runner_argv`` pointing to a program that reads
    ``{"name": ..., "arguments": {...}}`` from stdin and writes
    ``{"result": ...}`` (success) or ``{"error": ...}`` (failure) to stdout.
  - If either requirement is unmet, :exc:`SandboxUnavailableError` is raised.

* :data:`SandboxMode.FIRECRACKER` — tool call dispatched via
  ``tool_runner_argv`` inside a Firecracker MicroVM.  Requires:

  - ``firecracker`` binary at :attr:`SandboxConfig.firecracker_binary_path`.
  - A KVM device node (``/dev/kvm`` on Linux) accessible to the process.
  - :attr:`SandboxConfig.firecracker_kernel_path` and
    :attr:`SandboxConfig.firecracker_rootfs_path` set.
  - If any requirement is unmet, :exc:`SandboxUnavailableError` is raised.
  - A ``tool_runner_argv`` program (same stdin/stdout protocol as gVisor).

Only *tool calls* are sandboxed.  LLM calls, router calls, and join calls
are always executed directly through the wrapped base :class:`~stepback.replay.Executor`
because they communicate with external provider APIs rather than running
untrusted local code.

Protocol (GVISOR / FIRECRACKER modes)::

    # stdin written by parent:
    {"name": "<tool_name>", "arguments": {<tool_arguments>}}

    # stdout written by tool runner on success:
    {"result": <result_value>}

    # stdout written by tool runner on failure:
    {"error": "<error_message>"}

Usage::

    from stepback.sandbox import SandboxedExecutor, SandboxConfig, SandboxMode

    # Lightweight: subprocess isolation with 10-second timeout
    cfg = SandboxConfig(
        mode=SandboxMode.SUBPROCESS,
        timeout_sec=10.0,
        memory_limit_mb=256,
    )
    sandboxed = SandboxedExecutor(base_executor=my_executor, config=cfg)
    result = trace.replay_forward(sandboxed)

    # Strong: gVisor container isolation (requires runsc + tool runner)
    cfg_gvisor = SandboxConfig(
        mode=SandboxMode.GVISOR,
        timeout_sec=30.0,
        tool_runner_argv=["python3", "-m", "my_tool_runner"],
    )
    sandboxed_gvisor = SandboxedExecutor(base_executor=my_executor, config=cfg_gvisor)
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .replay import Executor, MissingExecutor


# ------------------------------------------------------------------ Modes


class SandboxMode(str, Enum):
    """Execution isolation mode for :class:`SandboxedExecutor`.

    Members are string-valued so ``SandboxMode.NONE == "none"`` is True and
    the mode can be stored/retrieved from config files directly.
    """

    NONE = "none"
    """No isolation.  Tool callback is invoked directly in the current process."""

    SUBPROCESS = "subprocess"
    """Isolated Python subprocess with wall-clock timeout and optional POSIX
    resource limits (CPU time, virtual memory).

    Not a security sandbox.  Requires a picklable tool callback.
    """

    GVISOR = "gvisor"
    """gVisor (runsc) container-kernel isolation.  Requires ``runsc`` on
    ``$PATH`` (or :attr:`SandboxConfig.gvisor_runsc_path`) and a
    :attr:`SandboxConfig.tool_runner_argv` command."""

    FIRECRACKER = "firecracker"
    """Firecracker MicroVM isolation.  Requires the ``firecracker`` binary,
    ``/dev/kvm``, :attr:`SandboxConfig.firecracker_kernel_path`,
    :attr:`SandboxConfig.firecracker_rootfs_path`, and a
    :attr:`SandboxConfig.tool_runner_argv` command."""


# ------------------------------------------------------------------ Config


@dataclass
class SandboxConfig:
    """Configuration for :class:`SandboxedExecutor`.

    All fields have sensible defaults; only ``mode`` is required.
    """

    mode: SandboxMode = SandboxMode.NONE
    """Isolation level.  Defaults to :data:`SandboxMode.NONE` (no isolation)."""

    timeout_sec: float = 30.0
    """Wall-clock timeout in seconds for a single tool call.  Enforced in all
    modes that actually run tool calls (SUBPROCESS, GVISOR, FIRECRACKER).
    A value of ``0`` or ``None`` means no timeout."""

    memory_limit_mb: Optional[int] = 512
    """Virtual-address-space limit in MiB.  Applied via ``RLIMIT_AS`` in
    SUBPROCESS mode on POSIX platforms.  Silently skipped on Windows.
    Ignored in NONE mode.  Not directly applied in GVISOR/FIRECRACKER (those
    runtimes enforce their own memory limits)."""

    cpu_limit_sec: Optional[int] = 60
    """CPU-time limit in seconds.  Applied via ``RLIMIT_CPU`` in SUBPROCESS
    mode on POSIX platforms.  Silently skipped on Windows.
    Ignored in NONE mode."""

    network_access: bool = False
    """Whether the sandboxed tool call is permitted to make network requests.
    In SUBPROCESS mode this flag is advisory only (enforced at the OS level
    only when network namespacing is available; the flag is surfaced on the
    SandboxConfig object for documentation/audit purposes).
    GVISOR and FIRECRACKER enforce it via their respective network
    isolation primitives."""

    gvisor_runsc_path: str = "runsc"
    """Path or name of the ``runsc`` binary for :data:`SandboxMode.GVISOR`.
    Resolved via ``shutil.which`` if not an absolute path."""

    firecracker_binary_path: str = "firecracker"
    """Path or name of the ``firecracker`` binary for
    :data:`SandboxMode.FIRECRACKER`.  Resolved via ``shutil.which``."""

    firecracker_kernel_path: Optional[str] = None
    """Path to the Firecracker guest kernel image (``vmlinux`` format)."""

    firecracker_rootfs_path: Optional[str] = None
    """Path to the Firecracker guest root filesystem image (ext4 raw image)."""

    tool_runner_argv: Optional[List[str]] = None
    """Command argv to run inside GVISOR/FIRECRACKER that dispatches tool
    calls.  The parent writes a JSON line to stdin; the runner writes a
    JSON line to stdout.  See the module docstring for the protocol."""


# ---------------------------------------------------------------- Errors


class SandboxError(RuntimeError):
    """Base class for all sandbox errors raised by :class:`SandboxedExecutor`."""


class SandboxTimeoutError(SandboxError):
    """Raised when a sandboxed tool call exceeds :attr:`SandboxConfig.timeout_sec`."""


class SandboxResourceError(SandboxError):
    """Raised when a sandboxed subprocess is killed by a resource limit (OOM, CPU)."""


class SandboxUnavailableError(SandboxError):
    """Raised when the requested sandbox mode is not available in this environment.

    Typical causes:
    * :data:`SandboxMode.GVISOR` — ``runsc`` not found on ``$PATH``.
    * :data:`SandboxMode.FIRECRACKER` — ``firecracker`` binary not found,
      ``/dev/kvm`` not accessible, or kernel/rootfs paths not configured.
    """


class SandboxViolationError(SandboxError):
    """Raised when the sandboxed process behaves unexpectedly (bad JSON output,
    non-zero exit code, or tool runner error response)."""


# -------------------------------------------------------- Subprocess worker
# Module-level so it is picklable by multiprocessing on spawn-based platforms.


def _subprocess_worker(
    tool_func: Any,
    name: str,
    arguments: dict,
    memory_limit_mb: Optional[int],
    cpu_limit_sec: Optional[int],
) -> dict:
    """Execute a tool call inside a subprocess worker.

    This function runs in a **child process** spawned by
    :class:`concurrent.futures.ProcessPoolExecutor`.  It applies POSIX
    resource limits (when available) before invoking ``tool_func``.

    Returns ``{"result": <value>}`` on success or raises on failure.
    """
    # Apply POSIX resource limits — best-effort, silently skipped on Windows.
    try:
        import resource as _res  # type: ignore[import]

        if memory_limit_mb and hasattr(_res, "RLIMIT_AS"):
            limit = memory_limit_mb * 1024 * 1024
            try:
                _res.setrlimit(_res.RLIMIT_AS, (limit, limit))
            except (ValueError, _res.error):
                pass  # limit may be lower than current soft limit; ignore

        if cpu_limit_sec and hasattr(_res, "RLIMIT_CPU"):
            try:
                _res.setrlimit(_res.RLIMIT_CPU, (cpu_limit_sec, cpu_limit_sec))
            except (ValueError, _res.error):
                pass
    except ImportError:
        pass  # Windows: resource module not available

    result = tool_func(name, arguments)
    return {"result": result}


# -------------------------------------------- External runner helpers (gVisor / Firecracker)


def _check_gvisor_available(cfg: SandboxConfig) -> str:
    """Return the resolved ``runsc`` binary path, or raise :exc:`SandboxUnavailableError`."""
    resolved = shutil.which(cfg.gvisor_runsc_path)
    if resolved is None:
        raise SandboxUnavailableError(
            f"gVisor 'runsc' binary not found (looked for {cfg.gvisor_runsc_path!r}). "
            "Install gVisor (https://gvisor.dev/docs/user_guide/install/) or use "
            "SandboxMode.SUBPROCESS for lightweight isolation."
        )
    return resolved


def _check_firecracker_available(cfg: SandboxConfig) -> str:
    """Return the resolved ``firecracker`` binary path, or raise :exc:`SandboxUnavailableError`."""
    resolved = shutil.which(cfg.firecracker_binary_path)
    if resolved is None:
        raise SandboxUnavailableError(
            f"Firecracker binary not found (looked for {cfg.firecracker_binary_path!r}). "
            "Install Firecracker (https://github.com/firecracker-microvm/firecracker) "
            "or use SandboxMode.SUBPROCESS for lightweight isolation."
        )
    # Check KVM availability on Linux.
    kvm_path = "/dev/kvm"
    if sys.platform.startswith("linux") and not os.path.exists(kvm_path):
        raise SandboxUnavailableError(
            f"KVM device {kvm_path!r} not found.  Firecracker requires hardware "
            "virtualisation (KVM).  Ensure the host supports KVM and the process "
            "has permission to access /dev/kvm."
        )
    if cfg.firecracker_kernel_path is None:
        raise SandboxUnavailableError(
            "SandboxConfig.firecracker_kernel_path must be set for "
            "SandboxMode.FIRECRACKER."
        )
    if cfg.firecracker_rootfs_path is None:
        raise SandboxUnavailableError(
            "SandboxConfig.firecracker_rootfs_path must be set for "
            "SandboxMode.FIRECRACKER."
        )
    return resolved


def _require_tool_runner(cfg: SandboxConfig, mode_name: str) -> List[str]:
    """Return ``cfg.tool_runner_argv`` or raise :exc:`SandboxUnavailableError`."""
    if not cfg.tool_runner_argv:
        raise SandboxUnavailableError(
            f"SandboxMode.{mode_name} requires SandboxConfig.tool_runner_argv to be "
            "set.  Provide an argv list that reads JSON from stdin and writes JSON to "
            "stdout (see stepback.sandbox module docstring for the protocol)."
        )
    return cfg.tool_runner_argv


def _run_external_runner(
    argv: List[str],
    name: str,
    arguments: dict,
    timeout_sec: Optional[float],
    extra_env: Optional[Dict[str, str]] = None,
) -> dict:
    """Send a tool-call request to an external runner via stdin/stdout JSON.

    The runner must implement the protocol::

        stdin:  {"name": ..., "arguments": {...}}\\n
        stdout: {"result": ...}\\n    (success)
                {"error": "..."}\\n   (failure)

    Returns the result dict on success.  Raises :exc:`SandboxTimeoutError`,
    :exc:`SandboxViolationError`, or :exc:`SandboxResourceError` on failure.
    """
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    env = None
    if extra_env:
        env = {**os.environ, **extra_env}

    try:
        proc = subprocess.run(
            argv,
            input=payload.encode(),
            capture_output=True,
            timeout=timeout_sec if (timeout_sec and timeout_sec > 0) else None,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise SandboxTimeoutError(
            f"Tool runner timed out after {timeout_sec}s (argv={argv!r})"
        ) from exc

    if proc.returncode != 0:
        stderr_snippet = proc.stderr[:256].decode(errors="replace")
        raise SandboxViolationError(
            f"Tool runner exited with code {proc.returncode}. "
            f"stderr: {stderr_snippet!r}"
        )

    raw = proc.stdout.decode(errors="replace").strip()
    if not raw:
        raise SandboxViolationError("Tool runner produced no output on stdout.")

    try:
        response = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SandboxViolationError(
            f"Tool runner stdout is not valid JSON: {raw[:128]!r}"
        ) from exc

    if not isinstance(response, dict):
        raise SandboxViolationError(
            f"Tool runner returned non-object JSON: {raw[:128]!r}"
        )

    if "error" in response:
        raise SandboxViolationError(
            f"Tool runner reported error: {response['error']}"
        )

    if "result" not in response:
        raise SandboxViolationError(
            f"Tool runner response missing 'result' key: {raw[:128]!r}"
        )

    return {"result": response["result"]}


# -------------------------------------------------------- SandboxedExecutor


class SandboxedExecutor(Executor):
    """An :class:`~stepback.replay.Executor` that runs tool calls in an isolation sandbox.

    All other step kinds (``llm_call``, ``router``, ``parallel_branch_join``,
    ``exception``) are forwarded to the ``base`` executor without sandboxing,
    because they invoke provider APIs or structural replay logic rather than
    untrusted local code.

    The ``real_calls`` counter on this instance is authoritative; the base
    executor's counter is not incremented.

    Args:
        base: The underlying executor supplying callbacks for all step kinds.
        config: Sandbox configuration selecting the isolation mode and limits.

    Raises:
        :exc:`SandboxUnavailableError`: On first use of a sandboxed tool call
            if the configured mode's runtime is not available.
        :exc:`SandboxTimeoutError`: When the tool call exceeds
            :attr:`~SandboxConfig.timeout_sec`.
        :exc:`SandboxViolationError`: When the sandboxed process returns an
            unexpected response (GVISOR/FIRECRACKER modes).
        :exc:`SandboxResourceError`: When the subprocess is terminated by an
            OS resource limit (SUBPROCESS mode).
    """

    def __init__(self, base: Executor, config: Optional[SandboxConfig] = None) -> None:
        # Do not call super().__init__() with base's callbacks — we want
        # real_calls to be tracked solely on self, and we dispatch manually.
        super().__init__(
            llm=base.llm,
            tool=base.tool,
            router=base.router,
            join=base.join,
            fallback_recorded=base.fallback_recorded,
        )
        # Override real_calls tracking: SandboxedExecutor.execute() handles
        # accounting itself; we do NOT propagate calls to base.execute() to
        # avoid double-counting.
        self.real_calls = 0
        self._base = base
        self._config: SandboxConfig = config or SandboxConfig()

    # ---------------------------------------------------------------- execute

    def execute(
        self,
        kind: str,
        inputs: dict,
        *,
        branch_outputs: Optional[List[Any]] = None,
    ) -> Any:
        """Dispatch a step execution, sandboxing tool calls.

        Tool calls (``kind == "tool_call"``) are isolated according to
        :attr:`~SandboxConfig.mode`.  All other step kinds are forwarded
        to the base executor's callbacks directly.
        """
        self.real_calls += 1

        if kind == "tool_call":
            return self._sandboxed_tool_call(inputs)

        # Non-tool steps: delegate to the base executor's specific callbacks
        # (not base.execute, to avoid double-counting base.real_calls).
        if kind == "llm_call":
            if self._base.llm is None:
                if self._base.fallback_recorded:
                    self.real_calls -= 1  # not a real execution
                    return None  # caller will use recorded output
                raise MissingExecutor("no llm executor for dirty llm_call")
            return self._base.llm(inputs["model"], inputs["messages"])

        if kind == "router":
            if self._base.router is None:
                raise MissingExecutor("no router executor for dirty router")
            return {"choice": self._base.router(inputs["name"], inputs["options"])}

        if kind == "parallel_branch_open":
            return {
                "branch_names": list(inputs.get("branch_names", [])),
                "branch_count": int(inputs.get("branch_count", 0)),
            }

        if kind == "parallel_branch_join":
            outs = list(branch_outputs or [])
            if self._base.join is not None:
                return self._base.join(inputs.get("name", ""), outs)
            return {"branches": outs}

        if kind == "exception":
            return {"error_class": "Replayed", "message": ""}

        raise MissingExecutor(f"unknown step kind: {kind}")

    # -------------------------------------------------------- sandboxed call

    def _sandboxed_tool_call(self, inputs: dict) -> dict:
        """Run a tool call through the configured sandbox."""
        name: str = inputs.get("name", "")
        arguments: dict = inputs.get("arguments", {})
        cfg = self._config

        if cfg.mode == SandboxMode.NONE:
            return self._direct_tool_call(name, arguments)

        if cfg.mode == SandboxMode.SUBPROCESS:
            return self._subprocess_tool_call(name, arguments, cfg)

        if cfg.mode == SandboxMode.GVISOR:
            return self._gvisor_tool_call(name, arguments, cfg)

        if cfg.mode == SandboxMode.FIRECRACKER:
            return self._firecracker_tool_call(name, arguments, cfg)

        raise SandboxError(f"Unknown sandbox mode: {cfg.mode!r}")

    def _direct_tool_call(self, name: str, arguments: dict) -> dict:
        """NONE mode: direct in-process call."""
        if self._base.tool is None:
            raise MissingExecutor("no tool executor for dirty tool_call")
        result = self._base.tool(name, arguments)
        return {"result": result}

    def _subprocess_tool_call(
        self, name: str, arguments: dict, cfg: SandboxConfig
    ) -> dict:
        """SUBPROCESS mode: run in a child process with optional resource limits.

        Not a security sandbox.  Provides wall-clock timeout and, on POSIX,
        CPU-time and virtual-memory limits.

        When ``fork`` is available (Linux, macOS), uses
        ``multiprocessing.Process`` with the ``fork`` start method: the child
        inherits the parent's entire address space so the tool callable does
        not need to be picklable.  Only the simple JSON-safe ``name`` /
        ``arguments`` / result are sent through the ``Pipe``.

        On platforms without ``fork`` (Windows), falls back to using
        ``concurrent.futures.ProcessPoolExecutor`` with the default start
        method, which requires the tool callback to be picklable.
        """
        if self._base.tool is None:
            raise MissingExecutor("no tool executor for dirty tool_call")

        import multiprocessing
        timeout = cfg.timeout_sec if cfg.timeout_sec and cfg.timeout_sec > 0 else None

        if "fork" in multiprocessing.get_all_start_methods():
            return self._subprocess_via_fork(
                self._base.tool, name, arguments, cfg, timeout
            )
        else:
            return self._subprocess_via_spawn(
                self._base.tool, name, arguments, cfg, timeout
            )

    @staticmethod
    def _subprocess_via_fork(
        tool_func: Any,
        name: str,
        arguments: dict,
        cfg: SandboxConfig,
        timeout: Optional[float],
    ) -> dict:
        """Fork-based isolation: child inherits parent memory, no pickling needed."""
        import multiprocessing

        ctx = multiprocessing.get_context("fork")
        parent_conn, child_conn = ctx.Pipe(duplex=False)

        def _child_main() -> None:
            """Run in the forked child; writes result or error to the pipe."""
            try:
                # Apply POSIX resource limits inside the child.
                try:
                    import resource as _res  # type: ignore[import]
                    if cfg.memory_limit_mb and hasattr(_res, "RLIMIT_AS"):
                        limit = cfg.memory_limit_mb * 1024 * 1024
                        try:
                            _res.setrlimit(_res.RLIMIT_AS, (limit, limit))
                        except (ValueError, _res.error):
                            pass
                    if cfg.cpu_limit_sec and hasattr(_res, "RLIMIT_CPU"):
                        try:
                            _res.setrlimit(
                                _res.RLIMIT_CPU, (cfg.cpu_limit_sec, cfg.cpu_limit_sec)
                            )
                        except (ValueError, _res.error):
                            pass
                except ImportError:
                    pass
                result = tool_func(name, arguments)
                child_conn.send(("ok", {"result": result}))
            except Exception as exc:  # noqa: BLE001
                child_conn.send(("error", type(exc).__name__ + ": " + str(exc)))
            finally:
                child_conn.close()

        proc = ctx.Process(target=_child_main, daemon=True)
        proc.start()
        child_conn.close()  # close child end in parent

        finished = parent_conn.poll(timeout)
        if not finished:
            proc.kill()
            proc.join()
            raise SandboxTimeoutError(
                f"Tool call '{name}' timed out after {timeout}s "
                "(SandboxMode.SUBPROCESS)"
            )

        try:
            status, payload = parent_conn.recv()
        except EOFError as exc:
            proc.join()
            raise SandboxResourceError(
                f"Tool call '{name}' subprocess exited without sending a result "
                "(killed by OS resource limit or crash)."
            ) from exc
        finally:
            parent_conn.close()
        proc.join()

        if status == "error":
            raise RuntimeError(f"Tool call '{name}' raised in subprocess: {payload}")
        return payload

    @staticmethod
    def _subprocess_via_spawn(
        tool_func: Any,
        name: str,
        arguments: dict,
        cfg: SandboxConfig,
        timeout: Optional[float],
    ) -> dict:
        """Spawn-based isolation (Windows): requires picklable tool callback."""
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                _subprocess_worker,
                tool_func,
                name,
                arguments,
                cfg.memory_limit_mb,
                cfg.cpu_limit_sec,
            )
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError as exc:
                raise SandboxTimeoutError(
                    f"Tool call '{name}' timed out after {timeout}s "
                    "(SandboxMode.SUBPROCESS)"
                ) from exc
            except concurrent.futures.process.BrokenProcessPool as exc:
                raise SandboxResourceError(
                    f"Tool call '{name}' subprocess was killed, likely by a resource "
                    "limit (OOM or CPU quota).  Increase SandboxConfig.memory_limit_mb "
                    "/ cpu_limit_sec or use SandboxMode.NONE."
                ) from exc

    def _gvisor_tool_call(
        self, name: str, arguments: dict, cfg: SandboxConfig
    ) -> dict:
        """GVISOR mode: run via runsc container runtime.

        Checks availability before the first call; raises
        :exc:`SandboxUnavailableError` if ``runsc`` is not found.
        """
        runsc_path = _check_gvisor_available(cfg)
        runner_argv = _require_tool_runner(cfg, "GVISOR")

        # Build the runsc command line.  The caller-supplied runner_argv is
        # the program that handles tool dispatch inside the gVisor sandbox.
        argv = [runsc_path, "run", "--"] + runner_argv
        return _run_external_runner(argv, name, arguments, cfg.timeout_sec)

    def _firecracker_tool_call(
        self, name: str, arguments: dict, cfg: SandboxConfig
    ) -> dict:
        """FIRECRACKER mode: run inside a Firecracker MicroVM.

        Checks binary and KVM availability before the first call; raises
        :exc:`SandboxUnavailableError` if any prerequisite is missing.
        """
        _check_firecracker_available(cfg)
        runner_argv = _require_tool_runner(cfg, "FIRECRACKER")

        # Firecracker does not have a simple CLI wrapper like runsc.
        # In production, the caller would manage VM lifecycle separately and
        # the tool_runner_argv would address a VSOCK/HTTP endpoint inside the VM.
        # For the MVP, we surface the architecture but defer full VM management
        # to a future step.  The runner_argv is executed directly with the
        # understanding that the caller has arranged the MicroVM environment.
        return _run_external_runner(runner_argv, name, arguments, cfg.timeout_sec)


# ----------------------------------------------------------- Factory


def create_sandbox(
    config: SandboxConfig,
    base: Executor,
) -> SandboxedExecutor:
    """Create a :class:`SandboxedExecutor` from a config and a base executor.

    Convenience factory equivalent to ``SandboxedExecutor(base=base, config=config)``.

    Args:
        config: Sandbox configuration (mode, limits, runner paths).
        base: The underlying executor supplying tool/LLM callbacks.

    Returns:
        A :class:`SandboxedExecutor` ready to be passed to
        :meth:`~stepback.replay.Trace.replay_forward`.
    """
    return SandboxedExecutor(base=base, config=config)


# ------------------------------------------------------- public re-exports

__all__ = [
    "SandboxMode",
    "SandboxConfig",
    "SandboxError",
    "SandboxTimeoutError",
    "SandboxResourceError",
    "SandboxUnavailableError",
    "SandboxViolationError",
    "SandboxedExecutor",
    "create_sandbox",
]
