"""Human-readable counterfactual reports.

The companion to :func:`stepback.branch_io.diff_replays`. ``diff_replays``
emits machine-readable JSON; this module emits the Markdown report a
human (incident-write-up author, regulator, code reviewer) actually
reads.

A report is built from one or two replays of the same trace plus the
substitution list that produced the counterfactual. The output is
deterministic and content-addressable so two engineers reading the
same trace + substitutions get byte-identical reports — important
both for review-by-PR and for the `stepback verify` regulator-replay
use-case.

Public surface::

    from stepback.report import (
        render_counterfactual_report,
        render_replay_report,
        ReportOptions,
    )

The CLI subcommand ``stepback report TRACE [--substitute SPEC ...] \
[--branch FILE] [--baseline-branch FILE] [-o OUT.md]`` is wired in
:mod:`stepback.cli`.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .canonical import hash_obj
from .replay import ReplayResult, StepView, Trace
from .substitutions import (
    ModelSubstitution,
    PolicySubstitution,
    PromptSubstitution,
    RouterSubstitution,
    Substitution,
    SubstitutionSet,
    ToolOutputSubstitution,
)


SCHEMA_VERSION = 1


# ----------------------------------------------------------- options


@dataclass
class ReportOptions:
    """Tunables for report rendering. All optional."""

    title: str = "stepback counterfactual report"
    show_step_table: bool = True
    show_substitution_section: bool = True
    show_dirty_subtree: bool = True
    show_decision_diffs: bool = True
    show_cost_summary: bool = True
    show_executive_summary: bool = True
    show_causal_attribution: bool = True
    max_step_rows: int = 200
    truncate_text: int = 120
    include_step_inputs: bool = False  # opt-in: can be PII-sensitive
    extra_metadata: dict = field(default_factory=dict)


# --------------------------------------------------- substitution rendering


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)] + "…"


def _summarise_substitution(sub: Substitution, *, truncate: int = 120) -> str:
    """Single-line human-readable summary of one substitution."""
    if isinstance(sub, PromptSubstitution):
        nmsg = len(sub.new_messages)
        first = sub.new_messages[0]["content"] if sub.new_messages else ""
        return (
            f"prompt @ {sub.at_step}: {nmsg} message(s); "
            f"first={_truncate(first, truncate)!r}"
        )
    if isinstance(sub, ModelSubstitution):
        return f"model @ {sub.at_step}: → {sub.new_model_id}"
    if isinstance(sub, ToolOutputSubstitution):
        body = json.dumps(sub.fake_response, sort_keys=True)
        cid = f" call_id={sub.tool_call_id}" if sub.tool_call_id else ""
        return f"tool_output @ {sub.at_step}{cid}: {_truncate(body, truncate)}"
    if isinstance(sub, PolicySubstitution):
        return f"policy @ {sub.at_step}: {sub.policy_path}"
    if isinstance(sub, RouterSubstitution):
        return f"router @ {sub.at_step}: → {sub.choice}"
    return f"{type(sub).__name__} @ {getattr(sub, 'at_step', '?')}"


def _render_substitutions(subs: SubstitutionSet) -> str:
    if not subs.items:
        return "_(no substitutions — baseline replay)_\n"
    lines = []
    for s in subs.items:
        lines.append(f"- {_summarise_substitution(s)}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------- dirty-subtree


def _dirty_subtree(steps: Sequence[StepView]) -> List[StepView]:
    """Return every dirty step, in trace order.

    Because the replay engine already propagates dirtiness through
    parent edges, we just filter on ``dirty``. This matches the
    "dirty subtree" notion in the README.
    """
    return [s for s in steps if s.dirty]


def _render_dirty_subtree(steps: Sequence[StepView]) -> str:
    dirty = _dirty_subtree(steps)
    if not dirty:
        return "_(no dirty steps — trace fully cache-hit)_\n"
    rows = []
    for s in dirty:
        rows.append(
            f"- `{s.step_id}` {s.kind} "
            f"name={(s.name or '')!r} cost=${s.cost_usd:.5f}"
        )
    return "\n".join(rows) + "\n"


# ------------------------------------------------------ step table


def _short_hash(h: Optional[str]) -> str:
    if not h:
        return "·"
    h = h.split(":", 1)[-1]
    return h[:8]


def _step_marker(s: StepView) -> str:
    if s.cache_hit:
        return "✓ cached"
    if s.dirty:
        return "▲ dirty "
    return "  ?     "


def _render_step_table(
    a_steps: Sequence[StepView],
    b_steps: Optional[Sequence[StepView]],
    *,
    max_rows: int,
) -> str:
    """Two-column step table: baseline vs counterfactual.

    If ``b_steps`` is None, only one column is rendered.
    """
    out = io.StringIO()
    if b_steps is None:
        out.write("| step | kind | name | status | cost_usd | outputs |\n")
        out.write("|------|------|------|--------|----------|---------|\n")
        for s in a_steps[:max_rows]:
            out.write(
                f"| `{s.step_id}` | {s.kind} | {(s.name or '')[:24]} | "
                f"{_step_marker(s).strip()} | {s.cost_usd:.5f} | "
                f"`{_short_hash(hash_obj(s.outputs))}` |\n"
            )
        if len(a_steps) > max_rows:
            out.write(f"| … | … | … | … | … | … (+{len(a_steps) - max_rows} more) |\n")
        return out.getvalue()

    a_by = {s.step_id: s for s in a_steps}
    b_by = {s.step_id: s for s in b_steps}
    ids = sorted(set(a_by) | set(b_by), key=lambda x: int(x.split(":")[-1]))
    out.write(
        "| step | kind | A status | A cost | B status | B cost | "
        "Δ cost | diverged |\n"
    )
    out.write(
        "|------|------|----------|--------|----------|--------|--------|----------|\n"
    )
    for sid in ids[:max_rows]:
        sa = a_by.get(sid)
        sb = b_by.get(sid)
        kind = (sa or sb).kind
        ah = hash_obj(sa.outputs) if sa else None
        bh = hash_obj(sb.outputs) if sb else None
        diverged = ah != bh
        ca = sa.cost_usd if sa else 0.0
        cb = sb.cost_usd if sb else 0.0
        out.write(
            f"| `{sid}` | {kind} | "
            f"{_step_marker(sa).strip() if sa else '—'} | "
            f"{ca:.5f} | "
            f"{_step_marker(sb).strip() if sb else '—'} | "
            f"{cb:.5f} | {cb - ca:+.5f} | {'✗' if diverged else ' '} |\n"
        )
    extra = len(ids) - max_rows
    if extra > 0:
        out.write(f"| … | … | … | … | … | … | … | (+{extra} more) |\n")
    return out.getvalue()


# ------------------------------------------------- decision diffs


def _decision_diff_rows(
    a_steps: Sequence[StepView], b_steps: Sequence[StepView]
) -> List[dict]:
    a_by = {s.step_id: s for s in a_steps}
    b_by = {s.step_id: s for s in b_steps}
    rows: List[dict] = []
    for sid in sorted(set(a_by) | set(b_by), key=lambda x: int(x.split(":")[-1])):
        sa = a_by.get(sid)
        sb = b_by.get(sid)
        ao = sa.outputs if sa else None
        bo = sb.outputs if sb else None
        if ao == bo:
            continue
        rows.append(
            {
                "step_id": sid,
                "kind": (sa or sb).kind,
                "a": ao,
                "b": bo,
                "cost_delta_usd": round(
                    (sb.cost_usd if sb else 0.0) - (sa.cost_usd if sa else 0.0), 8
                ),
            }
        )
    return rows


def _short_output(o: Any, *, truncate: int) -> str:
    if isinstance(o, dict) and "choices" in o:
        # OpenAI-shape: surface the assistant text, much more useful in a
        # human-readable report than the full JSON envelope.
        try:
            txt = o["choices"][0]["message"]["content"]
            return f"text: {_truncate(txt, truncate)!r}"
        except (KeyError, IndexError, TypeError):
            pass
    if isinstance(o, dict) and "result" in o:
        return f"result: {_truncate(json.dumps(o['result'], sort_keys=True), truncate)}"
    return _truncate(json.dumps(o, sort_keys=True, default=str), truncate)


def _render_decision_diffs(
    a_steps: Sequence[StepView],
    b_steps: Sequence[StepView],
    *,
    truncate: int,
) -> str:
    rows = _decision_diff_rows(a_steps, b_steps)
    if not rows:
        return "_(no diverging steps)_\n"
    out = io.StringIO()
    for r in rows:
        out.write(f"### `{r['step_id']}`  ({r['kind']})\n\n")
        out.write(f"- **A (baseline):** {_short_output(r['a'], truncate=truncate)}\n")
        out.write(f"- **B (counterfactual):** {_short_output(r['b'], truncate=truncate)}\n")
        out.write(f"- **Δ cost:** ${r['cost_delta_usd']:+.5f}\n\n")
    return out.getvalue()


# --------------------------------------------------- causal attribution


def _step_id_int(sid: str) -> int:
    try:
        return int(sid.split(":")[-1])
    except (ValueError, AttributeError):
        return -1


def _attribute_dirty_steps(
    steps: Sequence[StepView], subs: SubstitutionSet
) -> Dict[str, List[int]]:
    """Map each dirty step_id → indices of substitutions that explain it.

    A substitution at step S explains a dirty step T iff T is reachable
    from S by walking parent→child edges in the trace topology.
    """
    if not subs.items:
        return {}
    by_id: Dict[str, StepView] = {s.step_id: s for s in steps}
    children: Dict[str, List[str]] = {sid: [] for sid in by_id}
    for s in steps:
        if s.parent_step_id and s.parent_step_id in children:
            children[s.parent_step_id].append(s.step_id)

    attribution: Dict[str, List[int]] = {}
    for idx, sub in enumerate(subs.items):
        root = getattr(sub, "at_step", None)
        if root is None or root not in by_id:
            continue
        # BFS from root, collect every dirty step in the cone.
        seen = {root}
        stack = [root]
        while stack:
            sid = stack.pop()
            sv = by_id.get(sid)
            if sv is not None and sv.dirty:
                attribution.setdefault(sid, []).append(idx)
            for c in children.get(sid, []):
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
    # Keep substitution-index lists sorted+unique for byte-stable output.
    return {k: sorted(set(v)) for k, v in attribution.items()}


def _first_divergence(
    a_steps: Optional[Sequence[StepView]],
    b_steps: Optional[Sequence[StepView]],
) -> Optional[str]:
    """Step id where the two replays first disagree (in trace order)."""
    if a_steps is None or b_steps is None:
        return None
    a_by = {s.step_id: s for s in a_steps}
    b_by = {s.step_id: s for s in b_steps}
    ids = sorted(set(a_by) | set(b_by), key=_step_id_int)
    for sid in ids:
        sa = a_by.get(sid)
        sb = b_by.get(sid)
        ah = hash_obj(sa.outputs) if sa else None
        bh = hash_obj(sb.outputs) if sb else None
        if ah != bh:
            return sid
    return None


def _render_causal_attribution(
    attribution: Dict[str, List[int]], subs: SubstitutionSet
) -> str:
    if not attribution:
        return "_(no dirty steps attributable to substitutions)_\n"
    out = io.StringIO()
    # Group by substitution index for readability.
    by_sub: Dict[int, List[str]] = {}
    for sid, idxs in attribution.items():
        for i in idxs:
            by_sub.setdefault(i, []).append(sid)
    for idx in sorted(by_sub):
        sub = subs.items[idx]
        summary = _summarise_substitution(sub)
        sids = sorted(by_sub[idx], key=_step_id_int)
        joined = ", ".join(f"`{s}`" for s in sids)
        out.write(f"- **#{idx}** {summary}\n")
        out.write(f"  - causes: {joined}\n")
    return out.getvalue()


def _render_headline(
    baseline: ReplayResult,
    counterfactual: Optional[ReplayResult],
    first_div: Optional[str],
    subs: SubstitutionSet,
    attribution: Dict[str, List[int]],
) -> str:
    out = io.StringIO()
    if counterfactual is None:
        out.write(f"- **verdict:** baseline-only (no counterfactual)\n")
        out.write(
            f"- **baseline cost:** ${baseline.total_cost_usd:.5f} "
            f"(dirty={baseline.dirty_count}, real_executions={baseline.real_executions})\n"
        )
        return out.getvalue()
    delta = counterfactual.total_cost_usd - baseline.total_cost_usd
    if first_div is None:
        verdict = "unchanged"
    else:
        verdict = "diverged"
    out.write(f"- **verdict:** {verdict}\n")
    out.write(
        f"- **first divergence:** "
        f"{('`' + first_div + '`') if first_div else '_(none)_'}\n"
    )
    if first_div and first_div in attribution:
        idxs = ", ".join(f"#{i}" for i in attribution[first_div])
        out.write(f"  - attributed to substitution(s): {idxs}\n")
    out.write(
        f"- **Δ total_cost_usd:** ${delta:+.5f} "
        f"(A=${baseline.total_cost_usd:.5f}, B=${counterfactual.total_cost_usd:.5f})\n"
    )
    out.write(
        f"- **branch B:** dirty={counterfactual.dirty_count}, "
        f"real_executions={counterfactual.real_executions}\n"
    )
    out.write(f"- **substitutions applied:** {len(subs.items)}\n")
    return out.getvalue()


# --------------------------------------------------- structured model


def _round8(x: float) -> float:
    return round(float(x), 8)


def _step_view_to_row(s: StepView) -> dict:
    return {
        "step_id": s.step_id,
        "kind": s.kind,
        "name": s.name,
        "cache_hit": bool(s.cache_hit),
        "dirty": bool(s.dirty),
        "cost_usd": _round8(s.cost_usd),
        "outputs_hash": hash_obj(s.outputs),
    }


def _substitution_to_dict(idx: int, sub: Substitution) -> dict:
    return {
        "index": idx,
        "kind": type(sub).__name__,
        "at_step": getattr(sub, "at_step", None),
        "summary": _summarise_substitution(sub),
    }


def _build_report_model(
    trace: Trace,
    baseline: ReplayResult,
    counterfactual: Optional[ReplayResult],
    subs: SubstitutionSet,
    options: "ReportOptions",
) -> dict:
    """Build the JSON-safe intermediate model both renderers consume."""
    target = counterfactual if counterfactual is not None else baseline
    attribution = _attribute_dirty_steps(target.steps, subs)
    first_div = _first_divergence(
        baseline.steps if counterfactual is not None else None,
        counterfactual.steps if counterfactual is not None else None,
    )

    # Cost summary
    cost = {
        "baseline": {
            "total_cost_usd": _round8(baseline.total_cost_usd),
            "cache_hits": baseline.cache_hit_count,
            "dirty": baseline.dirty_count,
            "real_executions": baseline.real_executions,
        }
    }
    if counterfactual is not None:
        cost["counterfactual"] = {
            "total_cost_usd": _round8(counterfactual.total_cost_usd),
            "cache_hits": counterfactual.cache_hit_count,
            "dirty": counterfactual.dirty_count,
            "real_executions": counterfactual.real_executions,
        }
        cost["delta_total_cost_usd"] = _round8(
            counterfactual.total_cost_usd - baseline.total_cost_usd
        )

    if counterfactual is None:
        verdict = "no-counterfactual"
    elif first_div is None:
        verdict = "unchanged"
    else:
        verdict = "diverged"

    # Step table — two-column when counterfactual exists, single otherwise.
    if counterfactual is not None:
        a_by = {s.step_id: s for s in baseline.steps}
        b_by = {s.step_id: s for s in counterfactual.steps}
        ids = sorted(set(a_by) | set(b_by), key=_step_id_int)
        step_table = []
        for sid in ids:
            sa = a_by.get(sid)
            sb = b_by.get(sid)
            ah = hash_obj(sa.outputs) if sa else None
            bh = hash_obj(sb.outputs) if sb else None
            row = {
                "step_id": sid,
                "kind": (sa or sb).kind,
                "a": _step_view_to_row(sa) if sa else None,
                "b": _step_view_to_row(sb) if sb else None,
                "diverged": ah != bh,
                "cost_delta_usd": _round8(
                    (sb.cost_usd if sb else 0.0) - (sa.cost_usd if sa else 0.0)
                ),
            }
            step_table.append(row)
        decision_diffs = []
        for r in _decision_diff_rows(baseline.steps, counterfactual.steps):
            decision_diffs.append(
                {
                    "step_id": r["step_id"],
                    "kind": r["kind"],
                    "a": r["a"],
                    "b": r["b"],
                    "cost_delta_usd": _round8(r["cost_delta_usd"]),
                }
            )
    else:
        step_table = [_step_view_to_row(s) for s in baseline.steps]
        decision_diffs = []

    dirty_subtree = [
        {
            "step_id": s.step_id,
            "kind": s.kind,
            "name": s.name,
            "cost_usd": _round8(s.cost_usd),
        }
        for s in _dirty_subtree(target.steps)
    ]

    model = {
        "schema_version": SCHEMA_VERSION,
        "title": options.title,
        "trace_path": str(trace.path),
        "recorder_version": trace.header.get("recorder_version"),
        "canonicalisation_version": trace.header.get("canonicalisation_version"),
        "step_count": len(trace.recorded_steps),
        "extra_metadata": dict(options.extra_metadata or {}),
        "substitutions": [
            _substitution_to_dict(i, s) for i, s in enumerate(subs.items)
        ],
        "cost_summary": cost,
        "dirty_subtree": dirty_subtree,
        "decision_diffs": decision_diffs,
        "step_table": step_table,
        "first_divergence_step_id": first_div,
        "causal_attribution": attribution,
        "verdict": verdict,
    }
    return model


def render_report_json(
    trace: Trace,
    baseline: ReplayResult,
    counterfactual: Optional[ReplayResult],
    subs: SubstitutionSet,
    *,
    options: Optional["ReportOptions"] = None,
) -> dict:
    """Build the structured (JSON-safe) report dict.

    ``counterfactual`` may be ``None`` for a single-replay report.
    """
    options = options or ReportOptions()
    return _build_report_model(trace, baseline, counterfactual, subs, options)


def dump_report_json(
    trace: Trace,
    baseline: ReplayResult,
    counterfactual: Optional[ReplayResult],
    subs: SubstitutionSet,
    *,
    options: Optional["ReportOptions"] = None,
) -> str:
    """JSON string form of :func:`render_report_json`. Byte-stable: keys
    sorted, ``indent=2``, costs rounded to 8dp.
    """
    model = render_report_json(
        trace, baseline, counterfactual, subs, options=options
    )
    return json.dumps(model, sort_keys=True, indent=2, default=str)


# --------------------------------------------------- top-level renders


def _render_cost_summary(
    a: ReplayResult, b: Optional[ReplayResult]
) -> str:
    out = io.StringIO()
    out.write(f"- baseline: total_cost_usd=${a.total_cost_usd:.5f}, ")
    out.write(f"cache_hits={a.cache_hit_count}, dirty={a.dirty_count}, ")
    out.write(f"real_executions={a.real_executions}\n")
    if b is not None:
        out.write(f"- counterfactual: total_cost_usd=${b.total_cost_usd:.5f}, ")
        out.write(f"cache_hits={b.cache_hit_count}, dirty={b.dirty_count}, ")
        out.write(f"real_executions={b.real_executions}\n")
        delta = b.total_cost_usd - a.total_cost_usd
        out.write(f"- **Δ total_cost_usd:** ${delta:+.5f}\n")
    return out.getvalue()


def render_replay_report(
    trace: Trace,
    result: ReplayResult,
    subs: SubstitutionSet,
    *,
    options: Optional[ReportOptions] = None,
) -> str:
    """Single-replay report (no baseline comparison)."""
    options = options or ReportOptions()
    out = io.StringIO()
    out.write(f"# {options.title}\n\n")
    out.write(f"- trace: `{trace.path}`\n")
    out.write(f"- recorder_version: `{trace.header.get('recorder_version')}`\n")
    out.write(
        f"- canonicalisation_version: "
        f"`{trace.header.get('canonicalisation_version')}`\n"
    )
    out.write(f"- step_count: {len(trace.recorded_steps)}\n")
    for k, v in (options.extra_metadata or {}).items():
        out.write(f"- {k}: {v}\n")
    out.write("\n")

    if getattr(options, "show_executive_summary", True):
        out.write("## Headline\n\n")
        out.write(
            _render_headline(result, None, None, subs, {})
        )
        out.write("\n")

    if options.show_substitution_section:
        out.write("## Substitutions\n\n")
        out.write(_render_substitutions(subs))
        out.write("\n")

    if options.show_cost_summary:
        out.write("## Cost summary\n\n")
        out.write(_render_cost_summary(result, None))
        out.write("\n")

    if options.show_dirty_subtree:
        out.write("## Dirty subtree\n\n")
        out.write(_render_dirty_subtree(result.steps))
        out.write("\n")

    if options.show_step_table:
        out.write("## Step timeline\n\n")
        out.write(_render_step_table(result.steps, None, max_rows=options.max_step_rows))
        out.write("\n")

    return out.getvalue()


def render_counterfactual_report(
    trace: Trace,
    baseline: ReplayResult,
    counterfactual: ReplayResult,
    subs: SubstitutionSet,
    *,
    options: Optional[ReportOptions] = None,
) -> str:
    """Two-column counterfactual report (baseline vs counterfactual)."""
    options = options or ReportOptions()
    attribution = _attribute_dirty_steps(counterfactual.steps, subs)
    first_div = _first_divergence(baseline.steps, counterfactual.steps)
    out = io.StringIO()
    out.write(f"# {options.title}\n\n")
    out.write(f"- trace: `{trace.path}`\n")
    out.write(f"- recorder_version: `{trace.header.get('recorder_version')}`\n")
    out.write(f"- step_count: {len(trace.recorded_steps)}\n")
    for k, v in (options.extra_metadata or {}).items():
        out.write(f"- {k}: {v}\n")
    out.write("\n")

    if options.show_executive_summary:
        out.write("## Headline\n\n")
        out.write(
            _render_headline(baseline, counterfactual, first_div, subs, attribution)
        )
        out.write("\n")

    if options.show_substitution_section:
        out.write("## Substitutions applied to branch B\n\n")
        out.write(_render_substitutions(subs))
        out.write("\n")

    if options.show_cost_summary:
        out.write("## Cost summary\n\n")
        out.write(_render_cost_summary(baseline, counterfactual))
        out.write("\n")

    if options.show_dirty_subtree:
        out.write("## Dirty subtree (branch B)\n\n")
        out.write(_render_dirty_subtree(counterfactual.steps))
        out.write("\n")

    if options.show_decision_diffs:
        out.write("## Decision diffs\n\n")
        out.write(
            _render_decision_diffs(
                baseline.steps, counterfactual.steps, truncate=options.truncate_text
            )
        )
        out.write("\n")

    if options.show_causal_attribution:
        out.write("## Causal attribution\n\n")
        out.write(_render_causal_attribution(attribution, subs))
        out.write("\n")

    if options.show_step_table:
        out.write("## Step timeline\n\n")
        out.write(
            _render_step_table(
                baseline.steps, counterfactual.steps, max_rows=options.max_step_rows
            )
        )
        out.write("\n")

    return out.getvalue()


__all__ = [
    "ReportOptions",
    "render_counterfactual_report",
    "render_replay_report",
    "render_report_json",
    "dump_report_json",
    "SCHEMA_VERSION",
]
