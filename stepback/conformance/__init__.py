"""Frozen ``.sb`` conformance fixture corpus.

This subpackage holds frozen v1 ``.sb`` traces that every independent
implementation of the SB-Trace format (Python, Rust, TypeScript, Go, JVM,
.NET, WASM, the ``sb proxy``) must be able to read, verify, canonicalize,
and reject in the documented corrupt variants.

Fixtures land here under :func:`fixtures_dir` and are enumerated by
:func:`iter_fixtures`. Step 41 of ``100_STEPS.md`` is responsible for
populating the corpus; this module guarantees that whatever is dropped in
``stepback/conformance/fixtures/`` ships in both the wheel and the sdist.
"""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path
from typing import Iterator

__all__ = ["fixtures_dir", "iter_fixtures"]


def fixtures_dir() -> Path:
    """Return the on-disk path to the bundled fixture directory.

    Works both for editable installs (returns the source tree path) and
    installed wheels (returns a path inside the unpacked resource).
    """

    resource = files(__name__).joinpath("fixtures")
    with as_file(resource) as path:
        return Path(path)


def iter_fixtures(suffix: str = ".sb") -> Iterator[Path]:
    """Yield every bundled conformance fixture matching ``suffix``."""

    root = fixtures_dir()
    if not root.is_dir():
        return
    for path in sorted(root.rglob(f"*{suffix}")):
        if path.is_file():
            yield path
