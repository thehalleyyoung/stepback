"""``stepback comet-sigma-l1-trace-writer-features`` — print the live L1
feature vector for ``stepback.trace_writer`` + ``.sb`` format v1.

This module owns **Step 8** of ``COMET_SIGMA_1000.md``: a CLI subcommand
that exposes the Comet-Σ **L1** ``temporal_basis`` feature vector
emitted by :mod:`stepback.comet_sigma.l1_trace_writer` for any local
``.sb`` file.

The command reads frames out of the trace with the existing
:func:`stepback.trace_reader.read_frames` (no HMAC key required — frame
bodies are unverified, just like the temporal-basis emitter itself,
which observes ``_write_frame`` *before* signing). Each frame body is
re-canonicalised so ``body_bytes`` matches what the writer would have
fed to :func:`stepback.comet_sigma.l1_trace_writer.observe_frame`. The
``COMET_SIGMA_L1_TEMPORAL`` flag is enabled for the duration of the
command and restored on exit so running this CLI never leaks flag state
into the surrounding process.

Output formats
--------------

* **text** (default) — pretty-prints the latest base feature vector,
  one ``feature\tvalue`` line per :data:`BASE_FEATURES` entry. With
  ``--include-temporal`` also prints the most-recent
  ``(feature, window, aggregate)`` projection values, again one per
  line, and the per-window frame counts.
* **json** — emits a single deterministic JSON object on stdout with
  ``writer_id``, ``frames_observed``, ``latest`` (the base feature
  vector), and (when ``--include-temporal`` is set) ``temporal`` (the
  projection values + window counts + frame_index/wallclock_ns).

Exit codes
----------

* ``0`` — features were successfully computed and printed.
* ``2`` — the trace path is missing or unreadable / malformed.
* ``3`` — the upstream ``comet_sigma`` package is not importable on
  this interpreter (so the L1 emitter is permanently a no-op).
* ``4`` — the trace contained zero observable frames.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from . import comet_sigma_available
from . import l1_trace_writer as _l1
from . import l1_trace_writer_temporal as _l1t


#: Stable schema id for the JSON output of this CLI. Bump when the
#: shape of the JSON object below changes in a breaking way.
SCHEMA_VERSION: str = "comet_sigma_l1_trace_writer_features.v1"


def _writer_id_for(trace_path: str, override: Optional[str]) -> str:
    """Return a stable writer-id for the registry entry."""
    if override:
        return override
    return f"cli:{os.path.basename(os.path.abspath(trace_path))}"


def _feed_frames(
    trace_path: str,
    writer_id: str,
    *,
    max_frames: Optional[int],
    include_temporal: bool,
) -> int:
    """Read every frame body from ``trace_path`` and observe it.

    Returns the number of frames successfully observed (i.e. the number
    of times :func:`stepback.comet_sigma.l1_trace_writer.observe_frame`
    returned a non-None record).
    """
    # Local import keeps the CLI startup cost low and avoids a hard
    # dependency on trace_reader at module-import time.
    from ..trace_reader import read_frames
    from ..canonical import canonical_json

    bodies = read_frames(trace_path)
    if max_frames is not None:
        bodies = bodies[:max_frames]

    if include_temporal:
        _l1t.install_hook()

    n_observed = 0
    for body in bodies:
        body_bytes = canonical_json(body) if isinstance(body, dict) else b""
        rec = _l1.observe_frame(writer_id, body if isinstance(body, dict) else {}, body_bytes)
        if rec is not None:
            n_observed += 1
    return n_observed


def _result_json(
    *,
    writer_id: str,
    trace_path: str,
    frames_observed: int,
    include_temporal: bool,
) -> Dict[str, Any]:
    """Build the deterministic JSON object printed when ``--json`` is set."""
    latest = _l1.latest_feature_vector(writer_id) or {}
    out: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "trace_path": os.path.abspath(trace_path),
        "writer_id": writer_id,
        "frames_observed": frames_observed,
        "latest": {k: float(v) for k, v in sorted(latest.items())},
    }
    if include_temporal:
        proj = _l1t.latest_projection(writer_id)
        if proj is not None:
            out["temporal"] = {
                "wallclock_ns": int(proj.wallclock_ns),
                "frame_index": int(proj.frame_index),
                "window_counts": {w: int(n) for w, n in sorted(proj.window_counts.items())},
                "values": {k: float(v) for k, v in sorted(proj.values.items())},
            }
        else:
            out["temporal"] = None
    return out


def _render_text(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# stepback L1 trace_writer.sb v1 features")
    lines.append(f"trace_path\t{result['trace_path']}")
    lines.append(f"writer_id\t{result['writer_id']}")
    lines.append(f"frames_observed\t{result['frames_observed']}")
    lines.append("")
    lines.append("# latest base feature vector")
    for name, val in result["latest"].items():
        lines.append(f"{name}\t{val:.6g}")
    if "temporal" in result:
        proj = result["temporal"]
        lines.append("")
        lines.append("# latest temporal-basis projection")
        if proj is None:
            lines.append("(no projection — flag off or no frames observed)")
        else:
            lines.append(f"wallclock_ns\t{proj['wallclock_ns']}")
            lines.append(f"frame_index\t{proj['frame_index']}")
            for w, n in proj["window_counts"].items():
                lines.append(f"window_count[{w}]\t{n}")
            for name, val in proj["values"].items():
                lines.append(f"{name}\t{val:.6g}")
    return "\n".join(lines) + "\n"


def cmd(args: argparse.Namespace) -> int:
    """Implement ``stepback comet-sigma-l1-trace-writer-features``.

    See module docstring for argument semantics and exit codes. Always
    restores the ``COMET_SIGMA_L1_TEMPORAL`` env var and the per-writer
    L1 + temporal registries to whatever state they were in before the
    command ran.
    """
    if not comet_sigma_available():
        print(
            "comet_sigma not importable on this interpreter; install "
            "the kitchensink package to enable L1 features.",
            file=sys.stderr,
        )
        return 3

    trace_path = args.trace
    if not os.path.isfile(trace_path):
        print(f"trace not found: {trace_path}", file=sys.stderr)
        return 2

    writer_id = _writer_id_for(trace_path, args.writer_id)
    include_temporal = bool(args.include_temporal)
    max_frames = args.max_frames if args.max_frames and args.max_frames > 0 else None

    # Save & restore flag + registry state so the CLI is neighbourly.
    prev_flag = os.environ.get(_l1.FLAG_NAME)
    prev_hooks = list(_l1.OBSERVE_HOOKS)
    try:
        os.environ[_l1.FLAG_NAME] = "1"
        # Clean slate per CLI invocation so repeated runs are deterministic.
        _l1.reset(writer_id)
        _l1t.reset(writer_id)

        try:
            n_observed = _feed_frames(
                trace_path,
                writer_id,
                max_frames=max_frames,
                include_temporal=include_temporal,
            )
        except Exception as e:  # malformed trace, IO error, etc.
            print(f"failed to read frames from {trace_path}: {e}", file=sys.stderr)
            return 2

        if n_observed == 0:
            print(
                "no frames observed — trace is empty or every frame was "
                "rejected by the L1 emitter",
                file=sys.stderr,
            )
            return 4

        result = _result_json(
            writer_id=writer_id,
            trace_path=trace_path,
            frames_observed=n_observed,
            include_temporal=include_temporal,
        )

        if args.json:
            json.dump(result, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        else:
            sys.stdout.write(_render_text(result))
        return 0
    finally:
        # Always restore the flag + hook list. Drop *only* the writer
        # state we created so we don't accidentally clobber any other
        # in-process registrations.
        _l1.reset(writer_id)
        _l1t.reset(writer_id)
        # Restore hooks: install_hook() is idempotent but uninstall is
        # the cleanest way to undo a fresh install.
        if include_temporal and len(_l1.OBSERVE_HOOKS) != len(prev_hooks):
            _l1t.uninstall_hook()
        _l1.OBSERVE_HOOKS[:] = prev_hooks
        if prev_flag is None:
            os.environ.pop(_l1.FLAG_NAME, None)
        else:
            os.environ[_l1.FLAG_NAME] = prev_flag


def add_subparser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the subcommand on the top-level ``stepback`` parser."""
    p = sub.add_parser(
        "comet-sigma-l1-trace-writer-features",
        help=(
            "print the live L1 temporal_basis feature vector for "
            "stepback.trace_writer + .sb format v1"
        ),
        description=(
            "Comet-Σ Step 8 — read every frame from a .sb trace, feed "
            "it through the stepback.comet_sigma.l1_trace_writer "
            "emitter, and print the most-recent base feature vector. "
            "The COMET_SIGMA_L1_TEMPORAL flag is enabled for the "
            "duration of the command and restored on exit. With "
            "--include-temporal the Step-2 temporal-basis projection "
            "(1s/10s/1m/10m windows × 7 aggregates) is also printed."
        ),
    )
    p.add_argument("trace", help="input .sb trace path")
    p.add_argument(
        "--writer-id", default=None,
        help=(
            "explicit writer-id for the L1 registry entry "
            "(default: 'cli:<basename(trace)>')"
        ),
    )
    p.add_argument(
        "--include-temporal", action="store_true",
        help="also print the Step-2 temporal-basis projection",
    )
    p.add_argument(
        "--max-frames", type=int, default=0,
        help="only feed the first N frames (0 means all, the default)",
    )
    p.add_argument(
        "--json", action="store_true",
        help="print the structured JSON record on stdout instead of text",
    )
    p.set_defaults(func=cmd)
    return p


__all__ = [
    "SCHEMA_VERSION",
    "add_subparser",
    "cmd",
]
