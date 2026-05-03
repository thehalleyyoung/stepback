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
* :func:`current_recorder()` — returns the active ambient recorder, or
  raises :class:`RuntimeError` outside an :func:`enable` block. Agent scripts
  that want to be recordable but don't use one of the autopatched clients
  can call this to record explicitly.
* :func:`active()` — non-raising boolean form, useful for guards in agent
  code that should still run when ``stepback record`` is *not* active.

The autopatch is intentionally narrow:

* ``openai.OpenAI(...)`` constructor returns a :class:`WrappedOpenAI`.
* ``openai.AsyncOpenAI(...)`` is left alone (async support is m6 — see
  README §Milestones); the autorecorder logs a one-line warning instead of
  silently dropping calls.
* ``anthropic.Anthropic(...)`` constructor returns a :class:`WrappedAnthropic`.
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

from .recorder import Recorder, RecorderKey, record as _record


_LOCK = threading.RLock()
_CURRENT: Optional[Recorder] = None
_PATCHES: List[Tuple[Any, str, Any]] = []  # (module, attr, original)


def current_recorder() -> Recorder:
    """Return the ambient recorder; raise if no :func:`enable` is active."""
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
    with _LOCK:
        return _CURRENT is not None


def _set_current(rec: Optional[Recorder]) -> None:
    global _CURRENT
    with _LOCK:
        _CURRENT = rec


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


def _patch_openai(rec: Recorder) -> None:
    """Make ``openai.OpenAI(...)`` return a :class:`WrappedOpenAI`."""
    mod = sys.modules.get("openai")
    if mod is None:
        return
    from .shims import wrap_openai

    if hasattr(mod, "OpenAI"):
        original_cls = mod.OpenAI

        def _factory(*args: Any, **kwargs: Any) -> Any:
            client = original_cls(*args, **kwargs)
            return wrap_openai(client, rec)

        _patch(mod, "OpenAI", _factory)

    if hasattr(mod, "AsyncOpenAI"):
        async_cls = mod.AsyncOpenAI

        def _async_factory(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                "stepback autorecord: AsyncOpenAI is not yet recorded "
                "(planned for m6). Calls will pass through unwrapped.",
                stacklevel=2,
            )
            return async_cls(*args, **kwargs)

        _patch(mod, "AsyncOpenAI", _async_factory)


def _patch_anthropic(rec: Recorder) -> None:
    mod = sys.modules.get("anthropic")
    if mod is None:
        return
    from .shims import wrap_anthropic

    if hasattr(mod, "Anthropic"):
        original_cls = mod.Anthropic

        def _factory(*args: Any, **kwargs: Any) -> Any:
            client = original_cls(*args, **kwargs)
            return wrap_anthropic(client, rec)

        _patch(mod, "Anthropic", _factory)


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


def _install_all(rec: Recorder) -> None:
    _patch_openai(rec)
    _patch_anthropic(rec)
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
            _install_all(rec)
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


__all__ = [
    "enable",
    "disable",
    "active",
    "current_recorder",
]
