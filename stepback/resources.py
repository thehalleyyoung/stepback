"""Public accessors for resources bundled inside the installed wheel.

The repository root ships ``LICENSE``, ``CITATION.cff``, and ``README.md``
for humans browsing the source. ``scripts/sync_packaged_resources.py``
mirrors them into :mod:`stepback._resources` so they are also available
at runtime — for ``stepback --version`` style banners, license-display
helpers, and downstream packagers that need to ingest the citation
metadata without depending on the source layout.
"""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path
from typing import Final

__all__ = [
    "PACKAGED_FILES",
    "package_file",
    "read_text",
    "license_text",
    "citation_text",
    "readme_text",
]

PACKAGED_FILES: Final[tuple[str, ...]] = ("LICENSE", "CITATION.cff", "README.md")


def package_file(name: str) -> Path:
    """Return an on-disk path to a bundled resource file.

    Parameters
    ----------
    name:
        File name relative to ``stepback/_resources/``. Must be one of
        :data:`PACKAGED_FILES`.
    """

    if name not in PACKAGED_FILES:
        raise FileNotFoundError(
            f"{name!r} is not a packaged resource; expected one of {PACKAGED_FILES}"
        )
    resource = files("stepback._resources").joinpath(name)
    with as_file(resource) as path:
        return Path(path)


def read_text(name: str, encoding: str = "utf-8") -> str:
    """Return the text contents of a bundled resource file."""

    return package_file(name).read_text(encoding=encoding)


def license_text() -> str:
    """Return the bundled Apache-2.0 LICENSE text."""

    return read_text("LICENSE")


def citation_text() -> str:
    """Return the bundled ``CITATION.cff`` contents."""

    return read_text("CITATION.cff")


def readme_text() -> str:
    """Return the bundled ``README.md`` contents."""

    return read_text("README.md")
