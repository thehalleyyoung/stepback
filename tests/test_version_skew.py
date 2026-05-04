"""Guard against version skew between ``stepback.__version__`` and the
package metadata advertised by ``importlib.metadata``.

100_STEPS.md step 2 makes ``stepback.__version__`` the canonical source of
truth and requires ``pyproject.toml`` to derive its ``version`` field from
that attribute (via ``[tool.setuptools.dynamic]``). This test fails loudly
if the two ever drift.

The test is also a sanity check for the dynamic-version wiring: if
setuptools ever loses the ``attr = "stepback.__version__"`` rule, an
installed copy of the package would carry a stale literal and this test
would catch it.

When the package is *not* installed (e.g. the source tree was added to
``sys.path`` directly without ``pip install -e .``), ``importlib.metadata``
raises ``PackageNotFoundError``. In that mode we still validate that
``__version__`` itself is a non-empty PEP 440-shaped string and that the
``pyproject.toml`` declaration is wired to read it dynamically — those are
the static guarantees the step asks for.
"""

from __future__ import annotations

import re
import tomllib
from importlib import metadata
from pathlib import Path

import pytest

import stepback


PEP440_LOOSE = re.compile(
    r"^\d+(?:\.\d+){0,3}"
    r"(?:[abc]|rc)?\d*"
    r"(?:\.post\d+)?"
    r"(?:\.dev\d+)?"
    r"(?:\+[a-zA-Z0-9.]+)?$"
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_dunder_version_is_a_pep440_string() -> None:
    assert isinstance(stepback.__version__, str)
    assert stepback.__version__, "stepback.__version__ must not be empty"
    assert PEP440_LOOSE.match(stepback.__version__), (
        f"stepback.__version__={stepback.__version__!r} is not PEP 440-shaped"
    )


def test_pyproject_derives_version_from_dunder() -> None:
    """Static guarantee: ``pyproject.toml`` must mark ``version`` as
    dynamic and source it from ``stepback.__version__``. If a future
    edit hard-codes a literal version in ``[project]``, this test will
    surface the regression long before a release goes out."""
    pyproject_path = _project_root() / "pyproject.toml"
    if not pyproject_path.exists():
        pytest.skip("pyproject.toml not in the working tree")
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    project = data.get("project", {})
    dynamic = project.get("dynamic", [])
    assert "version" in dynamic, (
        "[project].dynamic must include 'version' so the package "
        "metadata is derived from stepback.__version__"
    )
    assert "version" not in project, (
        "[project].version must NOT be a literal: it has to come from "
        "stepback.__version__ to prevent skew (see 100_STEPS.md step 2)."
    )

    dyn = data.get("tool", {}).get("setuptools", {}).get("dynamic", {})
    version_rule = dyn.get("version")
    assert isinstance(version_rule, dict), (
        "[tool.setuptools.dynamic].version must be a table that points "
        "at stepback.__version__"
    )
    assert version_rule.get("attr") == "stepback.__version__", (
        f"[tool.setuptools.dynamic].version.attr must be "
        f"'stepback.__version__', got {version_rule.get('attr')!r}"
    )


def test_installed_metadata_matches_dunder_version() -> None:
    """If ``stepback`` is installed (sdist, wheel, or editable), the
    distribution metadata must match ``stepback.__version__`` exactly.
    Skips cleanly when running against an uninstalled source tree (e.g.
    ``PYTHONPATH=.`` in CI before ``pip install -e .``)."""
    try:
        installed = metadata.version("stepback")
    except metadata.PackageNotFoundError:
        pytest.skip(
            "stepback is not installed in this environment; "
            "skip the runtime parity check (the static check above "
            "still guards the pyproject wiring)"
        )
    assert installed == stepback.__version__, (
        f"version skew: importlib.metadata reports {installed!r} but "
        f"stepback.__version__ is {stepback.__version__!r}. "
        "Did pyproject.toml hard-code a literal version, or did an "
        "editable install go stale? Reinstall with `pip install -e .`."
    )
