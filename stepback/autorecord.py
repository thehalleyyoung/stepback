"""Process-global ambient recorder for ``stepback record -- python script.py``.

The README §7 advertises::

    stepback record  --output trace.sb -- python my_agent.py

so a user can record an *unmodified* agent script — they shouldn't have to
rewrite the script around an explicit ``with record(...) as rec:`` block. To
support that, this module exposes:

* :func:`enable(path, key=None)` — context manager that opens a `.sb` writer,
  installs a process-global :class:`stepback.recorder.Recorder`, and (best
  effort) monkey-patches any already-imported LLM/tool clients
  (``openai``, ``anthropic``, ``langchain_core``) so subsequent calls are
  recorded transparently.
* :func:`aenable(path, key=None)` — async variant of :func:`enable` for use
  inside ``async with`` blocks; patches ``AsyncOpenAI`` and ``AsyncAnthropic``
  in addition to the sync clients.
* :func:`current_recorder()` — returns the active ambient recorder, or
  raises :class:`RuntimeError` outside an :func:`enable` block. Agent scripts
  that want to be recordable but don't use one of the autopatched clients
  can call this to record explicitly.
* :func:`active()` — non-raising boolean form, useful for guards in agent
  code that should still run when ``stepback record`` is *not* active.

The autopatch is intentionally narrow:

* ``openai.OpenAI(...)`` constructor returns a :class:`WrappedOpenAI`.
* ``openai.AsyncOpenAI(...)`` constructor returns an :class:`AsyncWrappedOpenAI`
  when inside :func:`aenable`; under :func:`enable` it logs a warning instead.
* ``anthropic.Anthropic(...)`` constructor returns a :class:`WrappedAnthropic`.
* ``anthropic.AsyncAnthropic(...)`` constructor returns an async-wrapped client
  when inside :func:`aenable`.
* Any module attribute named ``langchain_tool`` / ``langchain_tools`` /
  ``tool_registry`` on the user's ``__main__`` module is wrapped if it
  duck-types as a LangChain ``BaseTool`` (has ``.name`` + ``.invoke``).

Patches are reverted on context-manager exit so a Python REPL session can
``enable()`` and ``disable()`` repeatedly without leaking state.
"""
from __future__ import annotations

import contextlib
import os
import sys
import threading
import warnings
from typing import Any, Callable, List, Optional, Tuple

from .recorder import Recorder, RecorderKey, _RECORDER_VAR, _PARENT_STEP_VAR, record as _record


_LOCK = threading.RLock()
_CURRENT: Optional[Recorder] = None
_PATCHES: List[Tuple[Any, str, Any]] = []  # (module, attr, original)
# Tokens returned by ContextVar.set() so we can reset cleanly on exit.
_REC_TOKEN: Any = None
_PARENT_TOKEN: Any = None


def current_recorder() -> Recorder:
    """Return the ambient recorder; raise if no :func:`enable` or :func:`aenable` is active.

    Checks the task-local :data:`~stepback.recorder._RECORDER_VAR` ContextVar
    first (set by :func:`~stepback.recorder.record`,
    :func:`~stepback.recorder.arecord`, and :func:`aenable`), then falls back
    to the process-global ``_CURRENT`` set by :func:`enable`.  The ContextVar
    takes precedence so nested async recordings can override the ambient one.
    """
    # Prefer the task-local ContextVar (works in both sync and async).
    cv = _RECORDER_VAR.get()
    if cv is not None:
        return cv
    with _LOCK:
        if _CURRENT is None:
            raise RuntimeError(
                "stepback.autorecord.current_recorder() called outside an "
                "active enable() / `stepback record` block. Wrap the script "
                "in `with stepback.autorecord.enable(path):` or run it via "
                "`stepback record --output PATH -- python script.py`."
            )
        return _CURRENT


def active() -> bool:
    """True iff an ambient recorder is installed in this process."""
    if _RECORDER_VAR.get() is not None:
        return True
    with _LOCK:
        return _CURRENT is not None


def _set_current(rec: Optional[Recorder]) -> None:
    global _CURRENT, _REC_TOKEN, _PARENT_TOKEN
    with _LOCK:
        _CURRENT = rec
        if rec is not None:
            _REC_TOKEN = _RECORDER_VAR.set(rec)
            _PARENT_TOKEN = _PARENT_STEP_VAR.set(None)
        else:
            if _REC_TOKEN is not None:
                _RECORDER_VAR.reset(_REC_TOKEN)
                _REC_TOKEN = None
            if _PARENT_TOKEN is not None:
                _PARENT_STEP_VAR.reset(_PARENT_TOKEN)
                _PARENT_TOKEN = None


# --------------------------------------------------------------------- patches

def _patch(mod: Any, attr: str, new: Any) -> None:
    original = getattr(mod, attr)
    _PATCHES.append((mod, attr, original))
    setattr(mod, attr, new)


def _revert_patches() -> None:
    while _PATCHES:
        mod, attr, original = _PATCHES.pop()
        try:
            setattr(mod, attr, original)
        except Exception:
            pass


def _patch_openai(rec: Recorder, *, include_async: bool = False) -> None:
    """Make ``openai.OpenAI(...)`` return a :class:`WrappedOpenAI`.

    When *include_async* is true (set by :func:`aenable`), also wraps
    ``openai.AsyncOpenAI(...)`` with :func:`~stepback.shims.wrap_openai_async`.
    """
    mod = sys.modules.get("openai")
    if mod is None:
        return
    from .shims import wrap_openai, wrap_openai_async

    if hasattr(mod, "OpenAI"):
        original_cls = mod.OpenAI

        def _factory(*args: Any, **kwargs: Any) -> Any:
            client = original_cls(*args, **kwargs)
            return wrap_openai(client, rec)

        _patch(mod, "OpenAI", _factory)

    if hasattr(mod, "AsyncOpenAI"):
        async_cls = mod.AsyncOpenAI
        if include_async:
            def _async_factory(*args: Any, **kwargs: Any) -> Any:
                client = async_cls(*args, **kwargs)
                return wrap_openai_async(client, rec)

            _patch(mod, "AsyncOpenAI", _async_factory)
        else:
            def _async_warn_factory(*args: Any, **kwargs: Any) -> Any:
                warnings.warn(
                    "stepback autorecord: AsyncOpenAI calls are not recorded "
                    "under enable(); use aenable() to record async OpenAI calls.",
                    stacklevel=2,
                )
                return async_cls(*args, **kwargs)

            _patch(mod, "AsyncOpenAI", _async_warn_factory)


def _patch_anthropic(rec: Recorder, *, include_async: bool = False) -> None:
    mod = sys.modules.get("anthropic")
    if mod is None:
        return
    from .shims import wrap_anthropic, wrap_anthropic_async

    if hasattr(mod, "Anthropic"):
        original_cls = mod.Anthropic

        def _factory(*args: Any, **kwargs: Any) -> Any:
            client = original_cls(*args, **kwargs)
            return wrap_anthropic(client, rec)

        _patch(mod, "Anthropic", _factory)

    if hasattr(mod, "AsyncAnthropic") and include_async:
        async_cls = mod.AsyncAnthropic

        def _async_anthropic_factory(*args: Any, **kwargs: Any) -> Any:
            client = async_cls(*args, **kwargs)
            return wrap_anthropic_async(client, rec)

        _patch(mod, "AsyncAnthropic", _async_anthropic_factory)


def _patch_langchain(rec: Recorder) -> None:
    """If ``langchain_core.tools.BaseTool`` is loaded, wrap its ``invoke``."""
    mod = sys.modules.get("langchain_core.tools")
    if mod is None or not hasattr(mod, "BaseTool"):
        return
    BaseTool = mod.BaseTool
    original_invoke = BaseTool.invoke

    def _patched_invoke(self: Any, arguments: Any, *args: Any, **kwargs: Any) -> Any:
        # Record one tool_call step per invocation.
        return rec.tool_call(
            getattr(self, "name", type(self).__name__),
            arguments,
            executor=lambda name, args_: original_invoke(self, args_),
        )["outputs"]

    _patch(BaseTool, "invoke", _patched_invoke)


def _install_all(rec: Recorder, *, include_async: bool = False) -> None:
    _patch_openai(rec, include_async=include_async)
    _patch_anthropic(rec, include_async=include_async)
    _patch_langchain(rec)


# ---------------------------------------------------------------- public API


@contextlib.contextmanager
def enable(
    path: str,
    *,
    key: Optional[RecorderKey] = None,
    autopatch: bool = True,
):
    """Install a process-global ambient recorder writing to ``path``.

    Yields the :class:`stepback.recorder.Recorder` so callers can also
    record explicitly. Reverts all monkey-patches and clears the global on
    exit. Re-entrant ``enable()`` calls raise :class:`RuntimeError` — there
    can only be one ambient recorder per process.
    """
    with _LOCK:
        if _CURRENT is not None:
            raise RuntimeError(
                "stepback.autorecord.enable() is already active in this "
                "process; only one ambient recorder is supported. Call "
                "disable() first or use the explicit `record(path)` "
                "context manager for nested traces."
            )

    cm = _record(path, key=key)
    rec = cm.__enter__()
    try:
        _set_current(rec)
        if autopatch:
            _install_all(rec, include_async=False)
        # Set env var so child code can detect (subprocesses won't see the
        # in-process patches but can still find the path and re-enable).
        os.environ["STEPBACK_RECORD_PATH"] = path
        yield rec
    finally:
        os.environ.pop("STEPBACK_RECORD_PATH", None)
        _revert_patches()
        _set_current(None)
        cm.__exit__(None, None, None)


def disable() -> None:
    """Tear down whatever :func:`enable` installed (idempotent)."""
    if not active():
        return
    _revert_patches()
    _set_current(None)
    os.environ.pop("STEPBACK_RECORD_PATH", None)


@contextlib.asynccontextmanager
async def aenable(
    path: str,
    *,
    key: Optional[RecorderKey] = None,
    autopatch: bool = True,
):
    """Async variant of :func:`enable` for async agent scripts.

    Sets the ambient recorder via the task-local
    :data:`~stepback.recorder._RECORDER_VAR` ContextVar so it propagates
    through ``await`` boundaries and into child tasks created by
    ``asyncio.create_task`` or ``asyncio.TaskGroup``.

    When *autopatch* is true (the default), patches both sync and async
    provider constructors:

    * ``openai.OpenAI(...)`` → :func:`~stepback.shims.wrap_openai`
    * ``openai.AsyncOpenAI(...)`` → :func:`~stepback.shims.wrap_openai_async`
    * ``anthropic.Anthropic(...)`` → :func:`~stepback.shims.wrap_anthropic`
    * ``anthropic.AsyncAnthropic(...)`` → :func:`~stepback.shims.wrap_anthropic_async`

    Usage::

        async def main():
            async with aenable("./trace.sb") as rec:
                client = openai.AsyncOpenAI()  # auto-patched
                resp = await client.chat.completions.create(...)

    Raises :class:`RuntimeError` if :func:`enable` or :func:`aenable` is
    already active.
    """
    with _LOCK:
        if _CURRENT is not None:
            raise RuntimeError(
                "stepback.autorecord.aenable() is already active in this "
                "process; only one ambient recorder is supported."
            )

    cm = _record(path, key=key)
    rec = cm.__enter__()
    try:
        _set_current(rec)
        if autopatch:
            _install_all(rec, include_async=True)
        os.environ["STEPBACK_RECORD_PATH"] = path
        yield rec
    finally:
        os.environ.pop("STEPBACK_RECORD_PATH", None)
        _revert_patches()
        _set_current(None)
        cm.__exit__(None, None, None)


__all__ = [
    "enable",
    "aenable",
    "disable",
    "active",
    "current_recorder",
]
