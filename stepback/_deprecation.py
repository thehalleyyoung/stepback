"""Internal helpers that implement the project deprecation policy.

The policy itself is documented in ``docs/deprecation.md`` and summarised
in ``CHANGELOG.md``. The short version is:

* Every removal must be preceded by **at least one minor release** in
  which the old API still works but emits a :class:`DeprecationWarning`.
* The warning message must name the **replacement API** (or state that
  no replacement exists) and the **release in which removal will
  occur**.
* Each deprecation is announced in ``CHANGELOG.md`` under a
  ``Deprecated`` heading at deprecation time and again under
  ``Removed`` at removal time.

The helpers in this module provide a single, consistent way to honour
that policy from inside ``stepback`` and from downstream code that
extends our public surface.

Public callable surface (re-exported from :mod:`stepback`):

``warn_deprecated(name, *, replacement, since, removal)``
    Emit a ``DeprecationWarning`` for an arbitrary call site. Useful at
    the top of a deprecated function body when the function cannot be
    wrapped (for example, an ``__init__`` signature change).

``deprecated(*, replacement, since, removal)``
    Decorator for functions, classes, or methods. The wrapped object
    keeps its original behaviour and signature; calling it emits a
    single, well-formed :class:`DeprecationWarning`.

``deprecated_alias(target, *, name, since, removal, module=None)``
    Build a thin shim that calls ``target`` after warning. The shim
    is suitable for assignment into a module's globals or class body
    so that ``module.OldName`` keeps working for one minor release.

The module is intentionally dependency-free; do not import anything
from ``stepback`` here so it can be used during package import.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, Optional, TypeVar

__all__ = [
    "DeprecationPolicyError",
    "deprecated",
    "deprecated_alias",
    "format_deprecation_message",
    "warn_deprecated",
]


F = TypeVar("F", bound=Callable[..., Any])


class DeprecationPolicyError(ValueError):
    """Raised when a deprecation declaration violates the policy.

    The policy requires *both* a ``since`` version (when the API became
    deprecated) and a ``removal`` version (the first release in which
    the API will be deleted). Skipping either makes the user-facing
    warning useless, so it is treated as a programmer error rather
    than something to silently paper over.
    """


def _validate(name: str, since: str, removal: str) -> None:
    if not isinstance(name, str) or not name:
        raise DeprecationPolicyError("deprecated API must have a non-empty name")
    if not isinstance(since, str) or not since:
        raise DeprecationPolicyError(
            f"deprecation of {name!r} is missing 'since' (the version "
            "that introduced the DeprecationWarning)"
        )
    if not isinstance(removal, str) or not removal:
        raise DeprecationPolicyError(
            f"deprecation of {name!r} is missing 'removal' (the first "
            "version in which the API will be deleted)"
        )
    if removal == since:
        raise DeprecationPolicyError(
            f"deprecation of {name!r}: 'removal' must be a later "
            "release than 'since'; the policy requires at least one "
            "minor release of overlap"
        )


def format_deprecation_message(
    name: str,
    *,
    replacement: Optional[str],
    since: str,
    removal: str,
) -> str:
    """Return the canonical, policy-compliant deprecation message.

    The exact wording is intentionally stable: downstream linters and
    release-note generators grep for the ``since=`` and ``removal=``
    tokens to build the changelog entries.
    """
    _validate(name, since, removal)
    if replacement:
        repl = f"use {replacement} instead"
    else:
        repl = "no direct replacement is planned"
    return (
        f"{name} is deprecated (since={since}); it will be removed "
        f"in stepback {removal}. {repl}. See docs/deprecation.md "
        "for the full deprecation policy."
    )


def warn_deprecated(
    name: str,
    *,
    replacement: Optional[str],
    since: str,
    removal: str,
    stacklevel: int = 2,
) -> None:
    """Emit a single :class:`DeprecationWarning` for ``name``.

    ``stacklevel`` defaults to ``2`` so the warning points at the
    caller's caller — which, when used at the top of a deprecated
    function body, is the user's own code.
    """
    msg = format_deprecation_message(
        name, replacement=replacement, since=since, removal=removal
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=stacklevel)


def deprecated(
    *,
    replacement: Optional[str],
    since: str,
    removal: str,
    name: Optional[str] = None,
) -> Callable[[F], F]:
    """Decorator that marks a function, method, or class as deprecated.

    The wrapped object keeps its behaviour and ``__wrapped__`` is set
    so :func:`inspect.signature` and :func:`functools.wraps`-aware
    tooling continue to work. Each *call* (not each import) emits one
    :class:`DeprecationWarning`. The default warning name is the
    target's ``__qualname__``; pass ``name`` to override it for
    re-exported aliases.
    """

    def decorate(target: F) -> F:
        target_name = name or getattr(target, "__qualname__", None) or repr(target)
        # Validate eagerly so policy errors surface at import time
        # rather than at first call.
        _validate(target_name, since, removal)

        if isinstance(target, type):
            orig_init = target.__init__

            @functools.wraps(orig_init)
            def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
                warn_deprecated(
                    target_name,
                    replacement=replacement,
                    since=since,
                    removal=removal,
                    stacklevel=3,
                )
                orig_init(self, *args, **kwargs)

            target.__init__ = new_init  # type: ignore[method-assign]
            _append_deprecation_doc(target, target_name, replacement, since, removal)
            return target  # type: ignore[return-value]

        @functools.wraps(target)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warn_deprecated(
                target_name,
                replacement=replacement,
                since=since,
                removal=removal,
                stacklevel=3,
            )
            return target(*args, **kwargs)

        _append_deprecation_doc(wrapper, target_name, replacement, since, removal)
        return wrapper  # type: ignore[return-value]

    return decorate


def deprecated_alias(
    target: Callable[..., Any],
    *,
    name: str,
    since: str,
    removal: str,
    replacement: Optional[str] = None,
) -> Callable[..., Any]:
    """Return a callable shim that warns and then forwards to ``target``.

    Use when a function or class has been *renamed*: keep the old name
    importable for one minor release by binding it to
    ``deprecated_alias(NewName, name="OldName", since=..., removal=...)``.

    If ``replacement`` is omitted, it defaults to the qualified name
    of ``target`` — which is almost always what you want.
    """
    repl = replacement or getattr(target, "__qualname__", None) or repr(target)
    _validate(name, since, removal)

    if isinstance(target, type):

        class _Alias(target):  # type: ignore[misc, valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                warn_deprecated(
                    name,
                    replacement=repl,
                    since=since,
                    removal=removal,
                    stacklevel=3,
                )
                super().__init__(*args, **kwargs)

        _Alias.__name__ = name
        _Alias.__qualname__ = name
        _append_deprecation_doc(_Alias, name, repl, since, removal)
        return _Alias

    @functools.wraps(target)
    def shim(*args: Any, **kwargs: Any) -> Any:
        warn_deprecated(
            name,
            replacement=repl,
            since=since,
            removal=removal,
            stacklevel=3,
        )
        return target(*args, **kwargs)

    shim.__name__ = name
    shim.__qualname__ = name
    _append_deprecation_doc(shim, name, repl, since, removal)
    return shim


def _append_deprecation_doc(
    obj: Any,
    name: str,
    replacement: Optional[str],
    since: str,
    removal: str,
) -> None:
    note = (
        f"\n\n.. deprecated:: {since}\n"
        f"   ``{name}`` will be removed in stepback {removal}. "
        + (f"Use ``{replacement}`` instead." if replacement else "No replacement is planned.")
    )
    existing = getattr(obj, "__doc__", None) or ""
    try:
        obj.__doc__ = existing + note
    except (AttributeError, TypeError):
        # Some built-in or slotted types refuse __doc__ assignment;
        # the warning itself is the load-bearing contract, not the
        # docstring annotation.
        pass
