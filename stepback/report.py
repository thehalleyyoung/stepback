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

import html as _html
import io
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    show_severity: bool = True
    html_inline_css: bool = True
    html_collapsed_step_table: bool = True


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
    if counterfactual is not None:
        model["severity"] = severity_score(baseline, counterfactual, subs).to_dict()
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

    if getattr(options, "show_severity", True):
        sev = severity_score(baseline, counterfactual, subs)
        out.write("## Severity\n\n")
        out.write(f"- score: **{sev.score}/100** ({sev.level})\n")
        for k, _w in _SEVERITY_WEIGHTS:
            out.write(f"- {k}: {sev.components.get(k, 0.0):.3f}\n")
        if sev.reasons:
            out.write(f"\nReasons: {', '.join(sev.reasons)}\n")
        else:
            out.write("\nReasons: _(none)_\n")
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


# ------------------------------------------------ severity scoring


@dataclass(frozen=True)
class SeverityScore:
    """Deterministic severity rubric for a counterfactual replay.

    Components are each in [0, 1]; the score is
    ``round(sum(component * weight) * 100)``. Weights:

    * cost_delta      — 0.30
    * dirty_fraction  — 0.30
    * decision_flips  — 0.30
    * subtree_depth   — 0.10

    Bands: <10 info, <25 low, <50 medium, <75 high, >=75 critical.
    """

    score: int
    level: str
    components: Dict[str, float]
    reasons: List[str]

    def to_dict(self) -> dict:
        return {
            "score": int(self.score),
            "level": self.level,
            "components": {k: round(float(v), 6) for k, v in self.components.items()},
            "reasons": list(self.reasons),
        }


_SEVERITY_WEIGHTS: Tuple[Tuple[str, float], ...] = (
    ("cost_delta", 0.30),
    ("dirty_fraction", 0.30),
    ("decision_flips", 0.30),
    ("subtree_depth", 0.10),
)


def _llm_decision_signature(s: StepView) -> Tuple[Optional[str], Optional[str]]:
    """Return (finish_reason, first_tool_call_name) or (None, None).

    Same OpenAI-shape access path used by ``_short_output``; swallows
    KeyError/IndexError/TypeError on non-conforming outputs.
    """
    o = s.outputs
    if not isinstance(o, dict):
        return (None, None)
    finish: Optional[str] = None
    tool_name: Optional[str] = None
    try:
        finish = o["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError):
        finish = None
    try:
        msg = o["choices"][0]["message"]
        tcs = msg.get("tool_calls") or []
        if tcs:
            tc0 = tcs[0]
            if isinstance(tc0, dict):
                if "function" in tc0 and isinstance(tc0["function"], dict):
                    tool_name = tc0["function"].get("name")
                else:
                    tool_name = tc0.get("name")
    except (KeyError, IndexError, TypeError):
        tool_name = None
    return (finish, tool_name)


def _depth_of(step_id: str, parent_by: Dict[str, Optional[str]]) -> int:
    d = 0
    cur: Optional[str] = step_id
    seen = set()
    while cur is not None and cur in parent_by and cur not in seen:
        seen.add(cur)
        cur = parent_by.get(cur)
        if cur is None:
            break
        d += 1
        if d > 10_000:  # cycle / pathological guard
            break
    return d


def severity_score(
    baseline: ReplayResult,
    counterfactual: Optional[ReplayResult],
    subs: SubstitutionSet,
) -> SeverityScore:
    """Compute the deterministic severity of a counterfactual replay.

    For ``counterfactual is None`` returns a degenerate score (0, info)
    so callers don't need to branch.
    """
    if counterfactual is None:
        return SeverityScore(score=0, level="info", components={}, reasons=[])

    a_total = float(baseline.total_cost_usd)
    b_total = float(counterfactual.total_cost_usd)
    cost_delta_abs = abs(b_total - a_total)
    if cost_delta_abs == 0.0:
        cost_delta = 0.0
    else:
        cost_delta = min(1.0, cost_delta_abs / max(a_total, 0.01))

    n_b = len(counterfactual.steps)
    dirty_fraction = (counterfactual.dirty_count / n_b) if n_b > 0 else 0.0
    dirty_fraction = max(0.0, min(1.0, dirty_fraction))

    a_by = {s.step_id: s for s in baseline.steps}
    b_by = {s.step_id: s for s in counterfactual.steps}
    llm_total = 0
    flips = 0
    for sid, sb in b_by.items():
        if sb.kind != "llm_call":
            continue
        llm_total += 1
        sa = a_by.get(sid)
        if sa is None:
            flips += 1
            continue
        if _llm_decision_signature(sa) != _llm_decision_signature(sb):
            flips += 1
    decision_flips = (flips / llm_total) if llm_total > 0 else 0.0
    decision_flips = max(0.0, min(1.0, decision_flips))

    parent_by = {s.step_id: s.parent_step_id for s in counterfactual.steps}
    total_depth = max((_depth_of(sid, parent_by) for sid in parent_by), default=0)
    dirty_steps = [s for s in counterfactual.steps if s.dirty]
    if dirty_steps and total_depth > 0:
        dd = max(_depth_of(s.step_id, parent_by) for s in dirty_steps)
        subtree_depth = max(0.0, min(1.0, dd / total_depth))
    else:
        subtree_depth = 0.0

    components = {
        "cost_delta": cost_delta,
        "dirty_fraction": dirty_fraction,
        "decision_flips": decision_flips,
        "subtree_depth": subtree_depth,
    }
    weighted = sum(components[k] * w for k, w in _SEVERITY_WEIGHTS)
    score = int(round(weighted * 100))
    score = max(0, min(100, score))

    if score < 10:
        level = "info"
    elif score < 25:
        level = "low"
    elif score < 50:
        level = "medium"
    elif score < 75:
        level = "high"
    else:
        level = "critical"

    reasons: List[str] = []
    if cost_delta > 0:
        reasons.append(f"cost {b_total - a_total:+.6f}")
    if dirty_fraction > 0:
        reasons.append(f"dirty {counterfactual.dirty_count}/{n_b} steps")
    if decision_flips > 0:
        reasons.append(f"{flips} decision flip(s) of {llm_total} llm step(s)")
    if subtree_depth > 0:
        reasons.append(
            f"dirty subtree depth {int(round(subtree_depth * total_depth))}/{total_depth}"
        )

    return SeverityScore(
        score=score, level=level, components=components, reasons=reasons
    )


# --------------------------------------------------- HTML renderer


_INLINE_CSS = (
    "body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;"
    "margin:2rem;max-width:64rem;color:#222}"
    "h1{font-size:1.6rem}h2{font-size:1.15rem;border-bottom:1px solid #ddd;"
    "padding-bottom:.2rem;margin-top:1.5rem}"
    "table{border-collapse:collapse;width:100%;font-size:.9rem}"
    "th,td{padding:.25rem .5rem;border-bottom:1px solid #eee;text-align:left;"
    "vertical-align:top}"
    "th{background:#f4f4f4;font-weight:600}"
    "tr.dirty td{background:#ffe9e9}tr.cached td{background:#e9f7e9}"
    "pre{background:#f6f6f6;padding:.5rem;overflow-x:auto;font-size:.85rem}"
    "code{background:#f6f6f6;padding:0 .25rem;border-radius:3px}"
    "dl{display:grid;grid-template-columns:max-content auto;gap:.1rem .75rem}"
    "dt{font-weight:600;color:#555}"
    ".badge{display:inline-block;padding:.2rem .55rem;border-radius:.4rem;"
    "font-weight:600;color:#fff}"
    ".badge-info{background:#888}.badge-low{background:#3a7}"
    ".badge-medium{background:#d80}.badge-high{background:#c33}"
    ".badge-critical{background:#600}"
    ".delta-pos{color:#a30;font-weight:600}.delta-neg{color:#063;font-weight:600}"
)


def _h(s: Any) -> str:
    """Escape arbitrary value for HTML body / attribute use."""
    return _html.escape("" if s is None else str(s), quote=True)


def _html_substitutions(model: dict) -> str:
    out = io.StringIO()
    subs = model.get("substitutions") or []
    if not subs:
        out.write("<p><em>(no substitutions)</em></p>")
        return out.getvalue()
    out.write("<ol>")
    for s in subs:
        out.write(
            f"<li><code>{_h(s.get('kind'))}</code> at "
            f"<code>{_h(s.get('at_step'))}</code> — {_h(s.get('summary'))}</li>"
        )
    out.write("</ol>")
    return out.getvalue()


def _html_cost(model: dict) -> str:
    cost = model.get("cost_summary") or {}
    a = cost.get("baseline") or {}
    b = cost.get("counterfactual")
    a_total = float(a.get("total_cost_usd", 0.0))
    out = io.StringIO()
    out.write("<dl>")
    out.write(
        "<dt>baseline total_cost_usd</dt><dd>$"
        + _h(format(a_total, ".5f"))
        + "</dd>"
    )
    out.write(f"<dt>baseline cache_hits</dt><dd>{_h(a.get('cache_hits'))}</dd>")
    out.write(f"<dt>baseline dirty</dt><dd>{_h(a.get('dirty'))}</dd>")
    out.write(f"<dt>baseline real_executions</dt><dd>{_h(a.get('real_executions'))}</dd>")
    if b is not None:
        b_total = float(b.get("total_cost_usd", 0.0))
        out.write(
            "<dt>counterfactual total_cost_usd</dt><dd>$"
            + _h(format(b_total, ".5f"))
            + "</dd>"
        )
        out.write(f"<dt>counterfactual cache_hits</dt><dd>{_h(b.get('cache_hits'))}</dd>")
        out.write(f"<dt>counterfactual dirty</dt><dd>{_h(b.get('dirty'))}</dd>")
        out.write(f"<dt>counterfactual real_executions</dt><dd>{_h(b.get('real_executions'))}</dd>")
        delta = float(cost.get("delta_total_cost_usd", 0.0))
        cls = "delta-pos" if delta > 0 else ("delta-neg" if delta < 0 else "")
        out.write(
            '<dt>Δ total_cost_usd</dt><dd class="'
            + cls
            + '">$'
            + _h(format(delta, "+.5f"))
            + "</dd>"
        )
    out.write("</dl>")
    return out.getvalue()


def _html_dirty(model: dict) -> str:
    dirty = model.get("dirty_subtree") or []
    if not dirty:
        return "<p><em>(no dirty steps — trace fully cache-hit)</em></p>"
    out = io.StringIO()
    out.write("<ul>")
    for d in dirty:
        cost_str = format(float(d.get("cost_usd", 0.0)), ".5f")
        out.write(
            "<li><code>"
            + _h(d.get("step_id"))
            + "</code> "
            + _h(d.get("kind"))
            + " name="
            + _h(d.get("name"))
            + " cost=$"
            + _h(cost_str)
            + "</li>"
        )
    out.write("</ul>")
    return out.getvalue()


def _html_decision_diffs(model: dict) -> str:
    diffs = model.get("decision_diffs") or []
    if not diffs:
        return "<p><em>(no diverging steps)</em></p>"
    out = io.StringIO()
    for r in diffs:
        out.write(
            "<h3><code>"
            + _h(r.get("step_id"))
            + "</code> ("
            + _h(r.get("kind"))
            + ")</h3>"
        )
        a_json = json.dumps(r.get("a"), sort_keys=True, default=str)
        b_json = json.dumps(r.get("b"), sort_keys=True, default=str)
        out.write(
            "<pre><strong>A:</strong> "
            + _h(a_json)
            + "\n<strong>B:</strong> "
            + _h(b_json)
            + "</pre>"
        )
        delta_str = format(float(r.get("cost_delta_usd", 0.0)), "+.5f")
        out.write("<p>Δ cost: $" + _h(delta_str) + "</p>")
    return out.getvalue()


def _html_step_table(model: dict, max_rows: int) -> str:
    rows = model.get("step_table") or []
    out = io.StringIO()
    out.write(
        "<table><thead><tr>"
        "<th>step_id</th><th>kind</th><th>name</th>"
        "<th>dirty</th><th>cache</th><th>cost_usd</th>"
        "</tr></thead><tbody>"
    )
    for r in rows[:max_rows]:
        if "a" in r and "b" in r:
            view = r.get("b") or r.get("a") or {}
        else:
            view = r
        dirty = bool(view.get("dirty"))
        cache = bool(view.get("cache_hit"))
        cls = "dirty" if dirty else ("cached" if cache else "")
        sid = view.get("step_id") or r.get("step_id")
        kind = view.get("kind") or r.get("kind")
        cost_str = format(float(view.get("cost_usd", 0.0)), ".8f")
        out.write(
            '<tr class="' + cls + '" data-step-id="' + _h(sid) + '">'
            + "<td><code>" + _h(sid) + "</code></td>"
            + "<td>" + _h(kind) + "</td>"
            + "<td>" + _h(view.get("name")) + "</td>"
            + "<td>" + ("yes" if dirty else "no") + "</td>"
            + "<td>" + ("yes" if cache else "no") + "</td>"
            + "<td>$" + _h(cost_str) + "</td>"
            + "</tr>"
        )
    if len(rows) > max_rows:
        more = len(rows) - max_rows
        out.write(
            '<tr><td colspan="6"><em>… '
            + str(more)
            + " more rows truncated …</em></td></tr>"
        )
    out.write("</tbody></table>")
    return out.getvalue()


def render_html_report(
    trace: Trace,
    baseline: ReplayResult,
    counterfactual: Optional[ReplayResult],
    subs: SubstitutionSet,
    *,
    options: Optional[ReportOptions] = None,
) -> str:
    """Self-contained HTML report. Inline CSS, no JS, no remote assets.

    Byte-deterministic for the same inputs. All user-controlled text
    (substitution payloads, headline, metadata) passes through
    ``html.escape(..., quote=True)``.
    """
    options = options or ReportOptions()
    model = _build_report_model(trace, baseline, counterfactual, subs, options)

    sev = severity_score(baseline, counterfactual, subs)

    out = io.StringIO()
    out.write("<!doctype html>\n")
    out.write("<html lang=\"en\"><head>\n")
    out.write("<meta charset=\"utf-8\">\n")
    out.write(f"<title>{_h(options.title)}</title>\n")
    if options.html_inline_css:
        out.write(f"<style>{_INLINE_CSS}</style>\n")
    out.write("</head><body>\n")
    out.write(f"<h1>{_h(options.title)}</h1>\n")

    out.write("<section id=\"meta\"><dl>")
    out.write(f"<dt>trace</dt><dd><code>{_h(model.get('trace_path'))}</code></dd>")
    out.write(
        f"<dt>recorder_version</dt><dd><code>{_h(model.get('recorder_version'))}</code></dd>"
    )
    out.write(
        f"<dt>canonicalisation_version</dt><dd><code>{_h(model.get('canonicalisation_version'))}</code></dd>"
    )
    out.write(f"<dt>step_count</dt><dd>{_h(model.get('step_count'))}</dd>")
    for k, v in (model.get("extra_metadata") or {}).items():
        out.write(f"<dt>{_h(k)}</dt><dd>{_h(v)}</dd>")
    out.write("</dl></section>\n")

    if options.show_severity and counterfactual is not None:
        out.write(
            f'<section id="severity" data-level="{_h(sev.level)}">'
            f"<h2>Severity</h2>"
            f'<p><span class="badge badge-{_h(sev.level)}">{sev.score}/100 — {_h(sev.level)}</span></p>'
        )
        if sev.reasons:
            out.write("<ul>")
            for r in sev.reasons:
                out.write(f"<li>{_h(r)}</li>")
            out.write("</ul>")
        else:
            out.write("<p><em>(no contributing axes)</em></p>")
        out.write("</section>\n")

    attribution = model.get("causal_attribution") or {}
    headline = _render_headline(
        baseline, counterfactual, model.get("first_divergence_step_id"), subs, attribution
    )
    out.write(f'<section id="headline"><h2>Headline</h2><pre>{_h(headline)}</pre></section>\n')

    if options.show_substitution_section:
        out.write('<section id="substitutions"><h2>Substitutions</h2>')
        out.write(_html_substitutions(model))
        out.write("</section>\n")

    if options.show_cost_summary:
        out.write('<section id="cost"><h2>Cost summary</h2>')
        out.write(_html_cost(model))
        out.write("</section>\n")

    if options.show_dirty_subtree:
        out.write('<section id="dirty"><h2>Dirty subtree</h2>')
        out.write(_html_dirty(model))
        out.write("</section>\n")

    if options.show_decision_diffs and counterfactual is not None:
        out.write('<section id="decision-diffs"><h2>Decision diffs</h2>')
        out.write(_html_decision_diffs(model))
        out.write("</section>\n")

    if options.show_step_table:
        out.write('<section id="step-table"><h2>Step timeline</h2>')
        out.write(_html_step_table(model, options.max_step_rows))
        out.write("</section>\n")

    out.write("</body></html>\n")
    return out.getvalue()


# --------------------------------------------------- format dispatcher


_FORMATS: Tuple[str, ...] = ("markdown", "json", "html")


def available_formats() -> List[str]:
    """Return the list of format ids accepted by :func:`render_report`."""
    return list(_FORMATS)


def render_report(
    trace: Trace,
    baseline: ReplayResult,
    counterfactual: Optional[ReplayResult],
    subs: SubstitutionSet,
    *,
    format: str = "markdown",
    options: Optional[ReportOptions] = None,
) -> str:
    """Dispatch to the markdown / json / html renderer for ``format``.

    See :func:`available_formats` for the supported ids. ``"md"`` is
    accepted as an alias for ``"markdown"``.
    """
    options = options or ReportOptions()
    fmt = format.lower()
    if fmt in ("markdown", "md"):
        if counterfactual is None:
            return render_replay_report(trace, baseline, subs, options=options)
        return render_counterfactual_report(
            trace, baseline, counterfactual, subs, options=options
        )
    if fmt == "json":
        return dump_report_json(
            trace, baseline, counterfactual, subs, options=options
        )
    if fmt == "html":
        return render_html_report(
            trace, baseline, counterfactual, subs, options=options
        )
    raise ValueError(
        f"unknown format: {format!r}; available: {available_formats()}"
    )


__all__ = [
    "ReportOptions",
    "SeverityScore",
    "available_formats",
    "dump_report_json",
    "render_counterfactual_report",
    "render_html_report",
    "render_replay_report",
    "render_report",
    "render_report_json",
    "severity_score",
    "SCHEMA_VERSION",
]
