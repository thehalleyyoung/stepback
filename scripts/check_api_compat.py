#!/usr/bin/env python3
"""API + SB-Trace schema compatibility checker.

Generates the current public-API and SB-Trace schema snapshots and
diffs them against a baseline. The baseline source is, in priority
order:

1. ``--baseline-dir DIR`` if explicitly given.
2. ``--baseline-ref REF`` (a git ref). Snapshots are produced from a
   worktree of that ref by re-importing the package from there.
3. The most recent ``v*`` git tag (``--auto-baseline-ref``, the CI
   default).
4. The committed baselines under
   ``stepback/conformance/api_baselines/v<version>/``, where
   ``<version>`` is the current ``stepback.__version__``. This is the
   fallback when no release tag exists yet.

A non-zero exit status means at least one *breaking* change was
detected. Compatible additions (new symbols, new optional fields, etc.)
are printed but do not fail the check.

Usage::

    python scripts/check_api_compat.py                   # auto baseline
    python scripts/check_api_compat.py --baseline-dir X  # explicit dir
    python scripts/check_api_compat.py --baseline-ref v0.1.0
    python scripts/check_api_compat.py --write-baseline  # refresh files

Designed to run inside CI without network access. The auto-baseline-ref
mode requires that the repository was checked out with
``fetch-depth: 0`` (or with at least the relevant tags fetched).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _import_snapshot_module():
    from stepback import api_snapshot  # noqa: WPS433 - intentional late import

    return api_snapshot


def _git(args: list[str], cwd: Optional[Path] = None) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(cwd or REPO_ROOT),
        stderr=subprocess.STDOUT,
        text=True,
    ).strip()


def _baseline_dir_for_current_version() -> Path:
    import stepback

    version = stepback.__version__
    return (
        REPO_ROOT
        / "stepback"
        / "conformance"
        / "api_baselines"
        / f"v{version}"
    )


def _read_baseline_dir(path: Path) -> Dict[str, Mapping[str, Any]]:
    api_path = path / "public_api.json"
    schema_path = path / "sbtrace_schema.json"
    if not api_path.exists() or not schema_path.exists():
        raise FileNotFoundError(
            f"baseline directory missing public_api.json or sbtrace_schema.json: {path}"
        )
    return {
        "public_api": json.loads(api_path.read_text(encoding="utf-8")),
        "sbtrace_schema": json.loads(schema_path.read_text(encoding="utf-8")),
    }


def _snapshots_from_ref(ref: str) -> Dict[str, Mapping[str, Any]]:
    """Check out ``ref`` into a temp worktree and dump snapshots from it."""
    with tempfile.TemporaryDirectory(prefix="stepback-baseline-") as td:
        worktree = Path(td) / "worktree"
        try:
            _git(["worktree", "add", "--detach", str(worktree), ref])
        except subprocess.CalledProcessError as exc:  # pragma: no cover
            raise RuntimeError(
                f"git worktree add failed for ref {ref!r}: {exc.output}"
            ) from exc
        try:
            cmd = [
                sys.executable,
                "-c",
                (
                    "import json, sys;"
                    "sys.path.insert(0, '.');"
                    "from stepback import api_snapshot;"
                    "out = {"
                    "  'public_api': api_snapshot.public_api_snapshot(),"
                    "  'sbtrace_schema': api_snapshot.sbtrace_schema_snapshot(),"
                    "};"
                    "sys.stdout.write(json.dumps(out))"
                ),
            ]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(worktree) + os.pathsep + env.get(
                "PYTHONPATH", ""
            )
            try:
                raw = subprocess.check_output(
                    cmd, cwd=str(worktree), env=env, text=True
                )
                return json.loads(raw)
            except subprocess.CalledProcessError:
                # Older releases predate api_snapshot; fall back to baselines
                # checked into THAT ref if present, else just an empty
                # snapshot so the comparison reports everything as "new".
                base_in_ref = (
                    worktree
                    / "stepback"
                    / "conformance"
                    / "api_baselines"
                )
                if base_in_ref.exists():
                    versions = sorted(
                        p for p in base_in_ref.iterdir() if p.is_dir()
                    )
                    if versions:
                        return _read_baseline_dir(versions[-1])
                snapshot_module = _import_snapshot_module()
                return {
                    "public_api": {
                        "snapshot_format_version": snapshot_module.SNAPSHOT_FORMAT_VERSION,
                        "package": "stepback",
                        "package_version": ref,
                        "symbols": {},
                        "missing": [],
                    },
                    "sbtrace_schema": {
                        "snapshot_format_version": snapshot_module.SNAPSHOT_FORMAT_VERSION,
                    },
                }
        finally:
            try:
                _git(["worktree", "remove", "--force", str(worktree)])
            except Exception:  # pragma: no cover - best-effort cleanup
                shutil.rmtree(worktree, ignore_errors=True)


def _latest_release_tag() -> Optional[str]:
    try:
        tags = _git(["tag", "--list", "v*", "--sort=-v:refname"]).splitlines()
    except subprocess.CalledProcessError:
        return None
    for t in tags:
        t = t.strip()
        if t:
            return t
    return None


def _resolve_baseline(
    args: argparse.Namespace,
) -> tuple[str, Dict[str, Mapping[str, Any]]]:
    if args.baseline_dir:
        path = Path(args.baseline_dir)
        return f"directory {path}", _read_baseline_dir(path)
    if args.baseline_ref:
        return f"git ref {args.baseline_ref}", _snapshots_from_ref(
            args.baseline_ref
        )
    if args.auto_baseline_ref:
        tag = _latest_release_tag()
        if tag:
            return f"git tag {tag}", _snapshots_from_ref(tag)
    fallback = _baseline_dir_for_current_version()
    if fallback.exists():
        return f"committed baseline {fallback.name}", _read_baseline_dir(fallback)
    raise SystemExit(
        "no baseline available: pass --baseline-dir / --baseline-ref or "
        "commit a baseline under stepback/conformance/api_baselines/."
    )


def _render(diff: Mapping[str, Any], header: str) -> bool:
    breaking = list(diff.get("breaking") or [])
    compatible = list(diff.get("compatible") or [])
    print(f"\n=== {header} ===")
    if not breaking and not compatible:
        print("  (no changes)")
        return False
    if compatible:
        print(f"  Compatible additions ({len(compatible)}):")
        for line in compatible:
            print(f"    + {line}")
    if breaking:
        print(f"  Breaking changes ({len(breaking)}):")
        for line in breaking:
            print(f"    ! {line}")
    return bool(breaking)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir",
        help="Directory containing public_api.json and sbtrace_schema.json.",
    )
    parser.add_argument(
        "--baseline-ref",
        help="Git ref to use as the baseline (e.g. v0.1.0 or main).",
    )
    parser.add_argument(
        "--auto-baseline-ref",
        action="store_true",
        help=(
            "Fall back to the most recent v* tag when --baseline-dir / "
            "--baseline-ref are not given. CI default."
        ),
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help=(
            "Write the *current* snapshots into the committed baseline "
            "directory for stepback.__version__ and exit 0."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Write the current snapshots into the given directory.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON report on stdout instead of text.",
    )
    args = parser.parse_args(argv)

    snapshot_module = _import_snapshot_module()
    current = {
        "public_api": snapshot_module.public_api_snapshot(),
        "sbtrace_schema": snapshot_module.sbtrace_schema_snapshot(),
    }

    if args.write_baseline:
        target = _baseline_dir_for_current_version()
        target.mkdir(parents=True, exist_ok=True)
        (target / "public_api.json").write_text(
            snapshot_module.dumps(current["public_api"]), encoding="utf-8"
        )
        (target / "sbtrace_schema.json").write_text(
            snapshot_module.dumps(current["sbtrace_schema"]), encoding="utf-8"
        )
        print(f"wrote baseline snapshots to {target}")
        return 0

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "public_api.json").write_text(
            snapshot_module.dumps(current["public_api"]), encoding="utf-8"
        )
        (args.out / "sbtrace_schema.json").write_text(
            snapshot_module.dumps(current["sbtrace_schema"]), encoding="utf-8"
        )

    label, baseline = _resolve_baseline(args)
    api_diff = snapshot_module.diff_public_api(
        baseline["public_api"], current["public_api"]
    )
    schema_diff = snapshot_module.diff_sbtrace_schema(
        baseline["sbtrace_schema"], current["sbtrace_schema"]
    )

    if args.json:
        print(
            json.dumps(
                {
                    "baseline": label,
                    "public_api_diff": api_diff,
                    "sbtrace_schema_diff": schema_diff,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"baseline: {label}")
        print(f"current package_version: {current['public_api'].get('package_version')}")
        print(
            f"current sbtrace wire_version: "
            f"{current['sbtrace_schema'].get('wire_version')}"
        )

    api_breaking = bool(api_diff.get("breaking"))
    schema_breaking = bool(schema_diff.get("breaking"))

    if not args.json:
        api_breaking = _render(api_diff, "Public API diff") or api_breaking
        schema_breaking = (
            _render(schema_diff, "SB-Trace schema diff") or schema_breaking
        )

    if api_breaking or schema_breaking:
        if not args.json:
            print(
                "\nFAIL: breaking changes detected. Bump the relevant SemVer "
                "track (Python package or SB-Trace wire) and refresh baselines "
                "with --write-baseline.",
                file=sys.stderr,
            )
        else:
            print(
                "FAIL: breaking changes detected.",
                file=sys.stderr,
            )
        return 1
    if not args.json:
        print("\nOK: no breaking changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
