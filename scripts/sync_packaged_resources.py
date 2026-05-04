#!/usr/bin/env python3
"""Mirror repo-root packaging files into the installable ``stepback`` package.

The canonical copies of ``LICENSE``, ``CITATION.cff``, and ``README.md``
live at the repository root. So that downstream wheels can serve them via
``importlib.resources`` (see :mod:`stepback.resources`), this script
copies them byte-for-byte into ``stepback/_resources/``. Run it any time
the source-of-truth file changes; the matching test in
``tests/test_packaged_resources.py`` will fail loudly if the mirrored
copy drifts.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESOURCES_DIR = REPO_ROOT / "stepback" / "_resources"
PACKAGED_FILES: tuple[str, ...] = ("LICENSE", "CITATION.cff", "README.md")


def main() -> int:
    if not RESOURCES_DIR.is_dir():
        RESOURCES_DIR.mkdir(parents=True)
    drift = 0
    for name in PACKAGED_FILES:
        src = REPO_ROOT / name
        dst = RESOURCES_DIR / name
        if not src.is_file():
            print(f"missing source: {src}", file=sys.stderr)
            return 2
        if dst.is_file() and dst.read_bytes() == src.read_bytes():
            continue
        shutil.copyfile(src, dst)
        print(f"synced {name}")
        drift += 1
    if drift == 0:
        print("packaged resources already in sync")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
