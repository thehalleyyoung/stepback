#!/usr/bin/env python3
"""Enforce per-module coverage floors declared in ``pyproject.toml``.

Step 26 of ``100_STEPS.md`` requires that the canonicalization, trace read/
write, attestation, divergence, and replay modules carry an enforced coverage
floor. The floors live in ``[tool.coverage.floors]`` of ``pyproject.toml`` and
are checked here against the JSON report emitted by
``coverage json -o coverage.json``.

Usage::

    coverage run -m pytest
    coverage json -o coverage.json
    python scripts/check_coverage_floors.py [--coverage-json coverage.json]
                                            [--pyproject pyproject.toml]

Exit code is 0 iff every floored file meets its floor (and the file appears in
the report). A non-zero exit prints a table of failing modules suitable for
copy-paste into a PR review.

The script is dependency-free aside from the standard library: it uses
``tomllib`` on Python 3.11+ and falls back to ``tomli`` on 3.10. It is safe to
call from CI before ``coverage`` itself is installed in the verifier image,
provided the JSON report has already been produced.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

try:  # Python 3.11+
    import tomllib as _toml  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - 3.10 fallback
    import tomli as _toml  # type: ignore[import-not-found,no-redef]


def load_floors(pyproject_path: Path) -> dict[str, float]:
    """Return a mapping of normalized file path -> floor percent (0-100)."""
    with pyproject_path.open("rb") as fh:
        data: Mapping[str, Any] = _toml.load(fh)
    coverage_section = data.get("tool", {}).get("coverage", {})
    floors_raw = coverage_section.get("floors", {})
    if not isinstance(floors_raw, dict):
        print(
            f"error: [tool.coverage.floors] in {pyproject_path} must be a table",
            file=sys.stderr,
        )
        raise SystemExit(2)
    floors: dict[str, float] = {}
    for path, threshold in floors_raw.items():
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            print(
                f"error: floor for {path!r} must be a number, got {threshold!r}",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if not 0 <= float(threshold) <= 100:
            print(
                f"error: floor for {path!r} must be in [0, 100], got {threshold!r}",
                file=sys.stderr,
            )
            raise SystemExit(2)
        floors[_normalize(path)] = float(threshold)
    return floors


def _normalize(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def load_file_coverages(coverage_json: Path) -> dict[str, float]:
    """Return mapping of normalized file path -> percent_covered (0-100)."""
    with coverage_json.open("r", encoding="utf-8") as fh:
        report = json.load(fh)
    files = report.get("files", {})
    out: dict[str, float] = {}
    for path, info in files.items():
        summary = info.get("summary", {})
        pct = summary.get("percent_covered")
        if pct is None:
            continue
        out[_normalize(path)] = float(pct)
    return out


def evaluate(
    floors: Mapping[str, float], coverages: Mapping[str, float]
) -> tuple[list[tuple[str, float, float]], list[str]]:
    """Return (failing, missing) where failing is (path, actual, floor)."""
    failing: list[tuple[str, float, float]] = []
    missing: list[str] = []
    for path, floor in floors.items():
        if path not in coverages:
            missing.append(path)
            continue
        actual = coverages[path]
        if actual + 1e-9 < floor:
            failing.append((path, actual, floor))
    return failing, missing


def _format_table(rows: list[tuple[str, float, float]]) -> str:
    if not rows:
        return ""
    width = max(len(p) for p, _, _ in rows)
    lines = [f"{'file'.ljust(width)}  {'actual':>8}  {'floor':>8}"]
    lines.append("-" * (width + 2 + 8 + 2 + 8))
    for path, actual, floor in rows:
        lines.append(f"{path.ljust(width)}  {actual:>7.2f}%  {floor:>7.2f}%")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage-json",
        default="coverage.json",
        type=Path,
        help="Path to a coverage.py JSON report (default: coverage.json).",
    )
    parser.add_argument(
        "--pyproject",
        default="pyproject.toml",
        type=Path,
        help="Path to pyproject.toml carrying [tool.coverage.floors].",
    )
    args = parser.parse_args(argv)

    if not args.pyproject.is_file():
        print(f"error: {args.pyproject} not found", file=sys.stderr)
        return 2
    if not args.coverage_json.is_file():
        print(
            f"error: {args.coverage_json} not found. "
            f"Run `coverage json -o {args.coverage_json}` first.",
            file=sys.stderr,
        )
        return 2

    floors = load_floors(args.pyproject)
    if not floors:
        print(
            "error: [tool.coverage.floors] is empty; nothing to enforce",
            file=sys.stderr,
        )
        return 2
    coverages = load_file_coverages(args.coverage_json)
    failing, missing = evaluate(floors, coverages)

    if missing:
        print(
            "error: floor declared for files not present in coverage report:",
            file=sys.stderr,
        )
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 2

    if failing:
        print("Coverage floors NOT met:", file=sys.stderr)
        print(_format_table(failing), file=sys.stderr)
        return 1

    print(
        f"Coverage floors met for {len(floors)} module(s). "
        f"Margins:"
    )
    for path, floor in sorted(floors.items()):
        actual = coverages[path]
        margin = actual - floor
        print(f"  {path}: {actual:.2f}% (floor {floor:.2f}%, +{margin:.2f})")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
