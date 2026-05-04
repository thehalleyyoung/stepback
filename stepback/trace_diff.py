"""Cross-trace diff: structural comparison of two recorded `.sb` traces.

This is the *regression analysis* primitive — orthogonal to
:py:meth:`stepback.replay.Trace.compare_branches`, which compares two
**replays of the same recorded trace** under different substitutions.
This module instead aligns the steps of **two distinct recorded
traces** (e.g. one from yesterday and one from today, or one from
prod-A and one from prod-B) and reports where the agent's behaviour
diverged.

Typical use case::

    from stepback.trace_diff import diff_traces, render_trace_diff
    d = diff_traces("traces/yesterday.sb", "traces/today.sb")
    print(d.divergence_step, d.shared_prefix_len, d.total_cost_delta_usd)
    print(render_trace_diff(d, format="markdown"))

CLI::

    stepback trace-diff yesterday.sb today.sb --format markdown -o report.md

Algorithm
---------

The two traces are walked in recorded (linear) order and paired by
index. For each pair we classify the relationship:

* ``identical`` — same kind, same ``inputs_hash``, same ``outputs_hash``.
* ``outputs_differ`` — same kind, same ``inputs_hash``, different
  ``outputs_hash``. Indicates non-determinism, an environment change,
  or a model-version drift the recorder didn't pin.
* ``inputs_differ`` — same kind, different ``inputs_hash``. The agent
  reached the same structural point but got there with different data
  (e.g. a different prompt template was deployed).
* ``kind_differ`` — different ``step_kind``. The control-flow
  diverged: trace B took a fundamentally different path at this step.
* ``a_only`` / ``b_only`` — one trace is longer than the other; the
  trailing steps have no counterpart.

The **divergence point** is the index of the first non-identical pair;
the **shared prefix length** is that same number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from .replay import replay


# ----------------------------------------------------------- statuses

STATUS_IDENTICAL = "identical"
STATUS_OUTPUTS_DIFFER = "outputs_differ"
STATUS_INPUTS_DIFFER = "inputs_differ"
STATUS_KIND_DIFFER = "kind_differ"
STATUS_A_ONLY = "a_only"
STATUS_B_ONLY = "b_only"

DIVERGENT_STATUSES = frozenset(
    {STATUS_OUTPUTS_DIFFER, STATUS_INPUTS_DIFFER, STATUS_KIND_DIFFER}
)


# ----------------------------------------------------- dataclasses


@dataclass
class StepPair:
    """One aligned (or unaligned) pair of steps across the two traces."""

    index: int
    status: str
    a_step_id: Optional[str]
    b_step_id: Optional[str]
    kind: Optional[str]
    name: Optional[str] = None
    input_diff: dict = field(default_factory=dict)
    output_diff: dict = field(default_factory=dict)
    cost_a_usd: float = 0.0
    cost_b_usd: float = 0.0
    cost_delta_usd: float = 0.0
    model_change: Optional[Tuple[str, str]] = None  # (a_model, b_model)

    @property
    def is_divergent(self) -> bool:
        return self.status in DIVERGENT_STATUSES

    @property
    def is_unaligned(self) -> bool:
        return self.status in {STATUS_A_ONLY, STATUS_B_ONLY}


@dataclass
class CrossTraceDiff:
    """Structural diff between two recorded `.sb` traces.

    Attributes
    ----------
    a_path, b_path : the source paths (informational only).
    a_header, b_header : the recorded ``header`` dicts of each trace.
    step_pairs : aligned pair list, one entry per index up to
        ``max(len(a), len(b))``.
    shared_prefix_len : number of leading ``identical`` pairs.
    divergence_step : ``a_step_id`` of the first non-identical pair,
        or ``None`` if the traces are identical.
    aligned_count : number of pairs where both sides have a step
        (irrespective of status).
    unaligned_count : pairs where one side is missing.
    a_only_count, b_only_count : breakdown of ``unaligned_count``.
    total_cost_delta_usd : ``sum(b.cost) - sum(a.cost)`` over all pairs.
    model_changes : list of ``(step_id, old_model, new_model)`` for
        every ``llm_call`` whose model id changed across the pair.
    """

    a_path: str
    b_path: str
    a_header: dict
    b_header: dict
    step_pairs: List[StepPair]
    shared_prefix_len: int
    divergence_step: Optional[str]
    aligned_count: int
    unaligned_count: int
    a_only_count: int
    b_only_count: int
    total_cost_delta_usd: float
    model_changes: List[Tuple[str, str, str]]

    # ----------- convenience views

    @property
    def divergent_count(self) -> int:
        return sum(1 for p in self.step_pairs if p.is_divergent)

    @property
    def identical_count(self) -> int:
        return sum(1 for p in self.step_pairs if p.status == STATUS_IDENTICAL)

    @property
    def is_identical(self) -> bool:
        """True iff the two traces produced the same step sequence with
        the same hashes everywhere."""
        return self.divergence_step is None and self.unaligned_count == 0

    def summary(self) -> dict:
        """Return a compact JSON-friendly summary (no per-step detail)."""
        return {
            "a_path": self.a_path,
            "b_path": self.b_path,
            "a_step_count": sum(1 for p in self.step_pairs if p.a_step_id),
            "b_step_count": sum(1 for p in self.step_pairs if p.b_step_id),
            "shared_prefix_len": self.shared_prefix_len,
            "divergence_step": self.divergence_step,
            "is_identical": self.is_identical,
            "aligned_count": self.aligned_count,
            "unaligned_count": self.unaligned_count,
            "a_only_count": self.a_only_count,
            "b_only_count": self.b_only_count,
            "identical_count": self.identical_count,
            "divergent_count": self.divergent_count,
            "total_cost_delta_usd": round(self.total_cost_delta_usd, 8),
            "model_changes": [
                {"step_id": sid, "from": a, "to": b}
                for (sid, a, b) in self.model_changes
            ],
        }

    def to_json_dict(self, *, max_pairs: Optional[int] = None) -> dict:
        """Full JSON-friendly representation including per-step detail."""
        pairs = self.step_pairs
        if max_pairs is not None:
            pairs = pairs[:max_pairs]
        return {
            **self.summary(),
            "step_pairs": [
                {
                    "index": p.index,
                    "status": p.status,
                    "a_step_id": p.a_step_id,
                    "b_step_id": p.b_step_id,
                    "kind": p.kind,
                    "name": p.name,
                    "input_diff": p.input_diff,
                    "output_diff": p.output_diff,
                    "cost_a_usd": round(p.cost_a_usd, 8),
                    "cost_b_usd": round(p.cost_b_usd, 8),
                    "cost_delta_usd": round(p.cost_delta_usd, 8),
                    "model_change": list(p.model_change) if p.model_change else None,
                }
                for p in pairs
            ],
        }


# --------------------------------------------------------- helpers


def _shallow_dict_diff(a: Any, b: Any) -> dict:
    """Return ``{key: {"a": ..., "b": ...}}`` for keys that differ.

    Falls back to ``{"_a": a, "_b": b}`` when either side is not a
    dict and they differ.
    """
    if a == b:
        return {}
    if not isinstance(a, dict) or not isinstance(b, dict):
        return {"_a": a, "_b": b}
    out: dict = {}
    for k in sorted(set(a) | set(b)):
        av = a.get(k, _MISSING)
        bv = b.get(k, _MISSING)
        if av != bv:
            out[k] = {
                "a": None if av is _MISSING else av,
                "b": None if bv is _MISSING else bv,
            }
    return out


class _Sentinel:
    pass


_MISSING = _Sentinel()


def _step_name(step: dict) -> Optional[str]:
    """Best-effort human label for a step (tool name, llm model, ...)."""
    if not step:
        return None
    inputs = step.get("inputs") or {}
    if step.get("step_kind") == "tool_call":
        return inputs.get("name")
    if step.get("step_kind") == "llm_call":
        return inputs.get("model")
    if step.get("step_kind") == "router":
        return inputs.get("name")
    return None


# ----------------------------------------------------------- main API


def diff_traces(
    a_path: str,
    b_path: str,
    *,
    hmac_key_a: Optional[bytes] = None,
    hmac_key_b: Optional[bytes] = None,
) -> CrossTraceDiff:
    """Compute a structural diff between two recorded `.sb` traces.

    Both traces are loaded with :func:`stepback.replay.replay`, which
    succeeds without an HMAC key (verification is opt-in). Pass
    ``hmac_key_a`` / ``hmac_key_b`` to verify chain integrity before
    diffing.
    """
    ta = replay(a_path, hmac_key=hmac_key_a)
    tb = replay(b_path, hmac_key=hmac_key_b)

    a_steps = ta.recorded_steps
    b_steps = tb.recorded_steps

    pairs: List[StepPair] = []
    n = max(len(a_steps), len(b_steps))
    aligned = 0
    a_only = 0
    b_only = 0
    cost_delta = 0.0
    model_changes: List[Tuple[str, str, str]] = []
    shared_prefix_len = 0
    divergence_step: Optional[str] = None
    saw_divergence = False

    for i in range(n):
        sa = a_steps[i] if i < len(a_steps) else None
        sb = b_steps[i] if i < len(b_steps) else None

        if sa is not None and sb is None:
            b_only_pair_status = STATUS_A_ONLY  # the *a* side has a step that b lacks
            ca = float(sa.get("cost_usd") or 0.0)
            pair = StepPair(
                index=i,
                status=b_only_pair_status,
                a_step_id=sa.get("step_id"),
                b_step_id=None,
                kind=sa.get("step_kind"),
                name=_step_name(sa),
                cost_a_usd=ca,
                cost_b_usd=0.0,
                cost_delta_usd=-ca,
            )
            cost_delta += -ca
            a_only += 1
            if not saw_divergence:
                divergence_step = sa.get("step_id")
                saw_divergence = True
            pairs.append(pair)
            continue

        if sa is None and sb is not None:
            cb = float(sb.get("cost_usd") or 0.0)
            pair = StepPair(
                index=i,
                status=STATUS_B_ONLY,
                a_step_id=None,
                b_step_id=sb.get("step_id"),
                kind=sb.get("step_kind"),
                name=_step_name(sb),
                cost_a_usd=0.0,
                cost_b_usd=cb,
                cost_delta_usd=cb,
            )
            cost_delta += cb
            b_only += 1
            if not saw_divergence:
                # No a-side step_id; record the b-side as the divergence.
                divergence_step = sb.get("step_id")
                saw_divergence = True
            pairs.append(pair)
            continue

        # Both present.
        assert sa is not None and sb is not None
        aligned += 1
        ka = sa.get("step_kind")
        kb = sb.get("step_kind")
        ca = float(sa.get("cost_usd") or 0.0)
        cb = float(sb.get("cost_usd") or 0.0)
        pair_cost_delta = cb - ca
        cost_delta += pair_cost_delta

        in_diff: dict = {}
        out_diff: dict = {}
        model_change: Optional[Tuple[str, str]] = None

        if ka != kb:
            status = STATUS_KIND_DIFFER
            in_diff = {"step_kind": {"a": ka, "b": kb}}
        elif sa.get("inputs_hash") == sb.get("inputs_hash"):
            if sa.get("outputs_hash") == sb.get("outputs_hash"):
                status = STATUS_IDENTICAL
            else:
                status = STATUS_OUTPUTS_DIFFER
                out_diff = _shallow_dict_diff(sa.get("outputs"), sb.get("outputs"))
        else:
            status = STATUS_INPUTS_DIFFER
            in_diff = _shallow_dict_diff(sa.get("inputs"), sb.get("inputs"))
            if sa.get("outputs_hash") != sb.get("outputs_hash"):
                out_diff = _shallow_dict_diff(sa.get("outputs"), sb.get("outputs"))

        # Detect model change for llm_call steps.
        if ka == "llm_call" == kb:
            am = (sa.get("inputs") or {}).get("model")
            bm = (sb.get("inputs") or {}).get("model")
            if am and bm and am != bm:
                model_change = (am, bm)
                model_changes.append((str(sa.get("step_id") or ""), am, bm))

        if status == STATUS_IDENTICAL and not saw_divergence:
            shared_prefix_len += 1
        elif status != STATUS_IDENTICAL and not saw_divergence:
            divergence_step = sa.get("step_id")
            saw_divergence = True

        pairs.append(
            StepPair(
                index=i,
                status=status,
                a_step_id=sa.get("step_id"),
                b_step_id=sb.get("step_id"),
                kind=ka if ka == kb else f"{ka}|{kb}",
                name=_step_name(sa) or _step_name(sb),
                input_diff=in_diff,
                output_diff=out_diff,
                cost_a_usd=ca,
                cost_b_usd=cb,
                cost_delta_usd=round(pair_cost_delta, 10),
                model_change=model_change,
            )
        )

    return CrossTraceDiff(
        a_path=a_path,
        b_path=b_path,
        a_header=ta.header,
        b_header=tb.header,
        step_pairs=pairs,
        shared_prefix_len=shared_prefix_len,
        divergence_step=divergence_step,
        aligned_count=aligned,
        unaligned_count=a_only + b_only,
        a_only_count=a_only,
        b_only_count=b_only,
        total_cost_delta_usd=round(cost_delta, 10),
        model_changes=model_changes,
    )


# ---------------------------------------------------------- rendering


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)] + "…"


def render_trace_diff(
    diff: CrossTraceDiff,
    *,
    format: str = "markdown",
    max_rows: int = 200,
    truncate: int = 80,
) -> str:
    """Render a cross-trace diff as ``markdown`` or ``json``."""
    fmt = format.lower()
    if fmt in ("md", "markdown"):
        return _render_markdown(diff, max_rows=max_rows, truncate=truncate)
    if fmt == "json":
        import json

        return json.dumps(diff.to_json_dict(max_pairs=max_rows), indent=2, sort_keys=True)
    raise ValueError(f"unknown format: {format!r} (expected 'markdown' or 'json')")


def _render_markdown(diff: CrossTraceDiff, *, max_rows: int, truncate: int) -> str:
    lines: List[str] = []
    lines.append("# stepback cross-trace diff")
    lines.append("")
    lines.append(f"- **A:** `{diff.a_path}`")
    lines.append(f"- **B:** `{diff.b_path}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- aligned pairs: **{diff.aligned_count}**")
    lines.append(f"- identical pairs: **{diff.identical_count}**")
    lines.append(f"- divergent pairs: **{diff.divergent_count}**")
    lines.append(
        f"- unaligned (a-only / b-only): **{diff.a_only_count}** / **{diff.b_only_count}**"
    )
    lines.append(f"- shared prefix length: **{diff.shared_prefix_len}**")
    lines.append(
        f"- divergence point: "
        f"**{diff.divergence_step or '— (traces are structurally identical)'}**"
    )
    lines.append(f"- total cost delta (B − A): **${diff.total_cost_delta_usd:+.6f}**")
    if diff.model_changes:
        lines.append("")
        lines.append("### Model changes")
        for sid, a, b in diff.model_changes:
            lines.append(f"- `{sid}`: `{a}` → `{b}`")
    lines.append("")
    lines.append("## Step pairs")
    lines.append("")
    lines.append("| # | a_step | b_step | kind | name | status | Δ$ |")
    lines.append("|---|--------|--------|------|------|--------|----|")
    shown = 0
    for p in diff.step_pairs:
        if shown >= max_rows:
            lines.append(
                f"| … | … | … | … | … | (truncated; {len(diff.step_pairs) - shown} more) | |"
            )
            break
        lines.append(
            f"| {p.index} | {p.a_step_id or '—'} | {p.b_step_id or '—'} | "
            f"{p.kind or '—'} | {_truncate(str(p.name or '—'), truncate)} | "
            f"{p.status} | {p.cost_delta_usd:+.6f} |"
        )
        shown += 1
    # Add a focused diff section for the first few divergent pairs.
    div_pairs = [p for p in diff.step_pairs if p.is_divergent or p.is_unaligned]
    if div_pairs:
        lines.append("")
        lines.append("## Divergent steps (detail)")
        for p in div_pairs[:10]:
            lines.append("")
            lines.append(
                f"### #{p.index} `{p.a_step_id or '—'}` ↔ `{p.b_step_id or '—'}` "
                f"({p.status})"
            )
            if p.model_change:
                lines.append(
                    f"- model change: `{p.model_change[0]}` → `{p.model_change[1]}`"
                )
            if p.input_diff:
                lines.append("- input diff:")
                for k, v in list(p.input_diff.items())[:8]:
                    lines.append(
                        f"  - `{k}`: `{_truncate(repr(v), truncate)}`"
                    )
            if p.output_diff:
                lines.append("- output diff:")
                for k, v in list(p.output_diff.items())[:8]:
                    lines.append(
                        f"  - `{k}`: `{_truncate(repr(v), truncate)}`"
                    )
        if len(div_pairs) > 10:
            lines.append("")
            lines.append(f"_… {len(div_pairs) - 10} more divergent pairs_")
    return "\n".join(lines) + "\n"
