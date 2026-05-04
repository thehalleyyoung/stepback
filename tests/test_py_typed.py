"""Step 10 — py.typed marker + end-to-end typed public surface.

These tests pin the contract that downstream users get mypy / pyright
coverage out of the box when they ``pip install stepback`` and write
``from stepback import record, replay``.

We intentionally do **not** spawn mypy here — that would couple CI to a
mypy version. Instead we:

1. Assert the ``py.typed`` PEP 561 marker ships inside the installed
   package directory.
2. Walk every name in :data:`stepback.__all__` and assert it is a
   real importable attribute (catches typos in the re-export block).
3. For every callable / class re-exported, assert it carries
   :func:`inspect.signature`-resolvable annotations on every parameter
   *and* a return annotation — i.e. no ``def f(x):`` slipping through.

The third check is the load-bearing one: it's what keeps the public
surface honest about being typed end-to-end.
"""
from __future__ import annotations

import inspect
import pathlib

import pytest

import stepback


def test_py_typed_marker_ships_in_package() -> None:
    pkg_root = pathlib.Path(stepback.__file__).resolve().parent
    marker = pkg_root / "py.typed"
    assert marker.is_file(), (
        f"PEP 561 marker missing: {marker}. Downstream type checkers "
        f"will skip stepback without it."
    )


def test_all_is_defined_and_nonempty() -> None:
    assert hasattr(stepback, "__all__"), "stepback.__init__ must define __all__"
    assert isinstance(stepback.__all__, list)
    assert len(stepback.__all__) > 50, "expected the full public surface"


def test_every_public_name_is_importable() -> None:
    missing = [n for n in stepback.__all__ if not hasattr(stepback, n)]
    assert not missing, f"stepback.__all__ references missing names: {missing}"


# Symbols that are re-exported as *modules* (not callables) — exempt
# from the per-parameter annotation check.
_MODULE_NAMES = {"autopilot", "predicates", "autorecord"}

# A small allow-list of names where adding a strict signature is not
# meaningful (constants, sentinels, str enums). Empty by default — we
# want this list to stay empty.
_NON_CALLABLE_ALLOWED = {
    "DIVERGENCE_SEVERITY",
    "DIVERGENCE_SEVERITY_WEIGHT",
    "DIVERGENCE_VOLATILE_KEYS",
}


def _public_callables():
    for name in stepback.__all__:
        if name in _MODULE_NAMES or name in _NON_CALLABLE_ALLOWED:
            continue
        obj = getattr(stepback, name)
        if inspect.ismodule(obj):
            continue
        if isinstance(obj, (str, int, float, bool, dict, list, tuple, set, frozenset)):
            continue
        yield name, obj


@pytest.mark.parametrize("name,obj", list(_public_callables()))
def test_public_callable_has_full_annotations(name: str, obj: object) -> None:
    """Every public callable / class must have annotations on every
    parameter and a return annotation.

    Dataclasses synthesise ``__init__`` from field annotations, so they
    are typed-by-construction and pass automatically.
    """
    if inspect.isclass(obj):
        # For classes, look at __init__ if user-defined; otherwise
        # accept (e.g. Exception subclasses inherit BaseException.__init__).
        init = obj.__init__
        if init is object.__init__ or init is BaseException.__init__:
            return
        try:
            sig = inspect.signature(init)
        except (TypeError, ValueError):
            return
    else:
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            return

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            # *args / **kwargs are intentionally allowed without annot.
            continue
        assert param.annotation is not inspect.Parameter.empty, (
            f"{name}({pname}): missing parameter annotation. "
            f"Public surface must be typed end-to-end (Step 10)."
        )

    # Functions must declare a return annotation; dataclasses /
    # exception subclasses don't need one on __init__.
    if not inspect.isclass(obj):
        assert sig.return_annotation is not inspect.Signature.empty, (
            f"{name}: missing return annotation. "
            f"Public surface must be typed end-to-end (Step 10)."
        )
