"""HTML minimization reports for :class:`MinimizationResult` objects.

Produces self-contained, offline HTML reports that show:

* **Summary** — strategy, substitution counts before and after, probe
  and cache-hit statistics.
* **Before / after graph** — CSS-only horizontal bar chart comparing
  original substitution count to the minimal subset.
* **Minimal substitutions** — table of the substitutions that still
  trigger the predicate, with optional Shapley attribution weights.
* **Removed substitutions** — table of items proved unnecessary.
* **Probe statistics** — breakdown of oracle calls, cache hits, and
  cache hit rate.
* **Attribution / confidence summary** — per-substitution Shapley
  weights when provided by a :class:`~stepback.minimize.ShapleyAttributionStrategy`.
* **Final replay result** — step-level table when the minimisation
  recorded a final replay.

For multi-objective results (:class:`~stepback.minimize.MultiObjectiveMinimizationResult`)
an additional **Pareto front** section is rendered showing every
non-dominated (substitution-count, objective-value) pair.

Usage::

    from stepback import render_html_minimize_report, MinimizeReportOptions
    from stepback import record, replay, minimize_substitutions

    with record(...) as rec:
        result = run_agent()

    trace = Trace.load(rec.path, rec.key)
    baseline = trace.replay_forward(executor)
    sub_set = SubstitutionSet(...)
    min_result = minimize_substitutions(trace, sub_set, predicate, executor)

    html = render_html_minimize_report(min_result)
    Path("report.html").write_text(html)

The output is byte-deterministic for the same inputs.
"""
from __future__ import annotations

import html as _html
import io
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .minimize import (
    MinimizationResult,
    MultiObjectiveMinimizationResult,
    TraceObjectives,
)
from .replay import ReplayResult, StepView
from .substitutions import Substitution


__all__ = [
    "MinimizeReportOptions",
    "render_html_minimize_report",
]

# ------------------------------------------------------------------ CSS

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
    ".badge-removed{background:#888}.badge-minimal{background:#3a7}"
    ".bar-wrap{background:#e8e8e8;border-radius:3px;height:1.2rem;width:100%;"
    "max-width:20rem;display:inline-block}"
    ".bar-fill{height:100%;border-radius:3px;background:#3a7;display:block}"
    ".bar-fill.removed{background:#c33}"
    ".weight-high{color:#a30;font-weight:600}"
    ".weight-low{color:#555}"
)


# ------------------------------------------------------------------ helpers

def _h(s: Any) -> str:
    """Escape arbitrary value for safe insertion into HTML."""
    return _html.escape("" if s is None else str(s), quote=True)


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + "…"


def _sub_kind(sub: Substitution) -> str:
    return type(sub).__name__


def _sub_at(sub: Substitution) -> str:
    return getattr(sub, "at_step", "?")


def _sub_summary(sub: Substitution, truncate: int = 80) -> str:
    """One-line human-readable description of a substitution."""
    kind = _sub_kind(sub)
    at = _sub_at(sub)
    if kind == "PromptSubstitution":
        msgs = getattr(sub, "new_messages", None) or []
        return f"{kind} @ {at}: {len(msgs)} message(s)"
    if kind == "ModelSubstitution":
        return f"{kind} @ {at}: → {getattr(sub, 'new_model_id', '?')}"
    if kind == "ToolOutputSubstitution":
        output = getattr(sub, "fake_response", None) or getattr(sub, "output", None)
        body = _truncate(str(output or ""), truncate)
        return f"{kind} @ {at}: {body}"
    if kind == "SystemPromptSubstitution":
        return f"{kind} @ {at}: mode={getattr(sub, 'mode', '?')}"
    if kind == "PolicySubstitution":
        return f"{kind} @ {at}: {getattr(sub, 'policy_path', '?')}"
    if kind == "RouterSubstitution":
        return f"{kind} @ {at}: → {getattr(sub, 'choice', '?')}"
    return f"{kind} @ {at}"


def _bar(filled: int, total: int, extra_class: str = "") -> str:
    """Return a CSS bar element. ``filled``/``total`` gives the fraction."""
    if total == 0:
        pct = 0
    else:
        pct = max(0, min(100, int(100 * filled / total)))
    cls = ("bar-fill " + extra_class).strip()
    return (
        f'<div class="bar-wrap">'
        f'<span class="{_h(cls)}" style="width:{pct}%"></span>'
        f"</div>"
    )


# ------------------------------------------------------------------ options


@dataclass
class MinimizeReportOptions:
    """Tunables for :func:`render_html_minimize_report`.

    All options are optional; the defaults produce a full report.

    Attributes
    ----------
    title :
        Document ``<title>`` and ``<h1>`` heading.
    show_before_after :
        Render the before / after bar-chart section.
    show_minimal_substitutions :
        Render the table of substitutions that still trigger the predicate.
    show_removed_substitutions :
        Render the table of proved-unnecessary substitutions.
    show_probe_stats :
        Render the oracle-call / cache-hit statistics section.
    show_attribution :
        Render the Shapley attribution / confidence-summary section (no-op
        when no weights are attached to the result).
    show_final_result :
        Render the final-replay step table when
        :attr:`MinimizationResult.final_result` is set.
    show_pareto_front :
        Render the Pareto-front section for
        :class:`~stepback.minimize.MultiObjectiveMinimizationResult`.
    html_inline_css :
        Embed the built-in CSS in the ``<head>`` so the output is
        fully self-contained.
    max_step_rows :
        Maximum rows to include in the final-replay step table.
    truncate_text :
        Maximum characters for inline text values (substitution
        summaries, step names, etc.) before truncation.
    extra_metadata :
        Additional ``key → value`` pairs inserted into the summary
        ``<dl>`` block.  Keys and values are HTML-escaped; keys are
        iterated in sorted order for byte-determinism.
    """

    title: str = "stepback minimization report"
    show_before_after: bool = True
    show_minimal_substitutions: bool = True
    show_removed_substitutions: bool = True
    show_probe_stats: bool = True
    show_attribution: bool = True
    show_final_result: bool = True
    show_pareto_front: bool = True
    html_inline_css: bool = True
    max_step_rows: int = 200
    truncate_text: int = 120
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ sections


def _section_summary(result: MinimizationResult, opts: MinimizeReportOptions) -> str:
    out = io.StringIO()
    orig_n = len(result.minimal) + len(result.removed)
    minimal_n = len(result.minimal)
    removed_n = len(result.removed)
    out.write('<section id="summary"><h2>Summary</h2><dl>')
    out.write(f"<dt>strategy</dt><dd><code>{_h(result.strategy_name)}</code></dd>")
    out.write(f"<dt>original substitutions</dt><dd>{_h(orig_n)}</dd>")
    out.write(f"<dt>minimal subset</dt><dd>{_h(minimal_n)}</dd>")
    out.write(f"<dt>removed (proved unnecessary)</dt><dd>{_h(removed_n)}</dd>")
    out.write(f"<dt>oracle probes</dt><dd>{_h(result.probes)}</dd>")
    out.write(f"<dt>cache hits</dt><dd>{_h(result.cache_hits)}</dd>")
    for k in sorted(opts.extra_metadata):
        out.write(f"<dt>{_h(k)}</dt><dd>{_h(opts.extra_metadata[k])}</dd>")
    out.write("</dl></section>\n")
    return out.getvalue()


def _section_before_after(result: MinimizationResult) -> str:
    orig_n = len(result.minimal) + len(result.removed)
    minimal_n = len(result.minimal)
    removed_n = len(result.removed)
    out = io.StringIO()
    out.write('<section id="before-after"><h2>Before / after</h2>')
    out.write('<table><thead><tr><th>Metric</th><th>Before</th><th>After</th>'
              '<th>Removed</th></tr></thead><tbody>')
    # substitutions row
    out.write(
        "<tr>"
        "<td>Substitutions</td>"
        f"<td>{_h(orig_n)} {_bar(orig_n, orig_n)}</td>"
        f"<td>{_h(minimal_n)} {_bar(minimal_n, orig_n)}</td>"
        f"<td>{_h(removed_n)} {_bar(removed_n, orig_n, 'removed')}</td>"
        "</tr>"
    )
    # if final_result is available show step counts
    final = result.final_result
    if final is not None:
        n_steps = len(final.steps)
        out.write(
            "<tr>"
            "<td>Replay steps (minimal)</td>"
            f"<td colspan=\"3\">{_h(n_steps)}</td>"
            "</tr>"
        )
        out.write(
            "<tr>"
            "<td>Real executions (minimal replay)</td>"
            f"<td colspan=\"3\">{_h(final.real_executions)}</td>"
            "</tr>"
        )
        cost_str = format(final.total_cost_usd, ".5f")
        out.write(
            "<tr>"
            "<td>Total cost (minimal replay)</td>"
            f"<td colspan=\"3\">${_h(cost_str)}</td>"
            "</tr>"
        )
    out.write("</tbody></table></section>\n")
    return out.getvalue()


def _section_subs_table(
    subs: Sequence[Substitution],
    section_id: str,
    heading: str,
    weights: Optional[Dict[int, float]],
    opts: MinimizeReportOptions,
) -> str:
    out = io.StringIO()
    out.write(f'<section id="{_h(section_id)}"><h2>{_h(heading)}</h2>')
    if not subs:
        out.write("<p><em>(none)</em></p></section>\n")
        return out.getvalue()
    has_weights = weights is not None and any(id(s) in weights for s in subs)
    out.write("<table><thead><tr>"
              "<th>#</th><th>Kind</th><th>At step</th><th>Summary</th>")
    if has_weights:
        out.write("<th>Shapley weight</th>")
    out.write("</tr></thead><tbody>")
    for idx, sub in enumerate(subs, start=1):
        summary = _truncate(_sub_summary(sub, opts.truncate_text), opts.truncate_text)
        out.write(
            f"<tr>"
            f"<td>{_h(idx)}</td>"
            f"<td><code>{_h(_sub_kind(sub))}</code></td>"
            f"<td><code>{_h(_sub_at(sub))}</code></td>"
            f"<td>{_h(summary)}</td>"
        )
        if has_weights:
            w = (weights or {}).get(id(sub), 0.0)
            w_str = format(w, ".4f")
            cls = "weight-high" if abs(w) > 0.1 else "weight-low"
            out.write(f'<td class="{_h(cls)}">{_h(w_str)}</td>')
        out.write("</tr>")
    out.write("</tbody></table></section>\n")
    return out.getvalue()


def _section_probe_stats(result: MinimizationResult) -> str:
    out = io.StringIO()
    out.write('<section id="probe-stats"><h2>Probe statistics</h2><dl>')
    total_calls = result.probes + result.cache_hits
    hit_rate_pct = (
        format(100.0 * result.cache_hits / total_calls, ".1f") + "%"
        if total_calls > 0
        else "n/a"
    )
    out.write(f"<dt>oracle calls (actual replays)</dt><dd>{_h(result.probes)}</dd>")
    out.write(f"<dt>cache hits (memoised)</dt><dd>{_h(result.cache_hits)}</dd>")
    out.write(f"<dt>total subset evaluations</dt><dd>{_h(total_calls)}</dd>")
    out.write(f"<dt>cache hit rate</dt><dd>{_h(hit_rate_pct)}</dd>")
    out.write(f"<dt>strategy</dt><dd><code>{_h(result.strategy_name)}</code></dd>")
    out.write("</dl></section>\n")
    return out.getvalue()


def _section_attribution(result: MinimizationResult) -> str:
    """Shapley weight table for the *minimal* set."""
    weights = result.weights
    if not weights:
        return (
            '<section id="attribution"><h2>Attribution / confidence summary</h2>'
            "<p><em>No Shapley weights attached to this result. "
            "Run minimisation with "
            "<code>ShapleyAttributionStrategy</code> to enable this section.</em></p>"
            "</section>\n"
        )
    out = io.StringIO()
    out.write('<section id="attribution">'
              "<h2>Attribution / confidence summary</h2>"
              "<p>Shapley weights measure each substitution's average marginal "
              "contribution to the predicate across all orderings of the minimal "
              "subset. Higher weight → stronger causal role.</p>")
    # Build rows sorted by descending weight for readability (deterministic tie-break by sub index)
    indexed = [(id(s), s, weights.get(id(s), 0.0)) for s in result.minimal]
    indexed.sort(key=lambda t: (-t[2], result.minimal.index(t[1])))
    out.write("<table><thead><tr>"
              "<th>Rank</th><th>Kind</th><th>At step</th>"
              "<th>Shapley weight</th></tr></thead><tbody>")
    for rank, (_, sub, w) in enumerate(indexed, start=1):
        w_str = format(w, ".4f")
        cls = "weight-high" if abs(w) > 0.1 else "weight-low"
        out.write(
            f"<tr>"
            f"<td>{_h(rank)}</td>"
            f"<td><code>{_h(_sub_kind(sub))}</code></td>"
            f"<td><code>{_h(_sub_at(sub))}</code></td>"
            f'<td class="{_h(cls)}">{_h(w_str)}</td>'
            f"</tr>"
        )
    out.write("</tbody></table></section>\n")
    return out.getvalue()


def _section_final_result(
    final: ReplayResult,
    max_rows: int,
) -> str:
    out = io.StringIO()
    out.write('<section id="final-result"><h2>Final replay (minimal subset)</h2>')
    steps = final.steps[:max_rows]
    truncated = len(final.steps) > max_rows
    out.write(
        f"<p>Steps: {_h(len(final.steps))} | "
        f"Dirty: {_h(final.dirty_count)} | "
        f"Cache hits: {_h(final.cache_hit_count)} | "
        f"Real executions: {_h(final.real_executions)} | "
        f"Total cost: ${_h(format(final.total_cost_usd, '.5f'))}</p>"
    )
    if truncated:
        out.write(
            f"<p><em>(Showing first {_h(max_rows)} of {_h(len(final.steps))} steps.)</em></p>"
        )
    out.write("<table><thead><tr>"
              "<th>Step ID</th><th>Kind</th><th>Name</th>"
              "<th>Dirty</th><th>Cost $</th></tr></thead><tbody>")
    for sv in steps:
        dirty_flag = "dirty" if getattr(sv, "dirty", False) else "cached"
        cls = "dirty" if dirty_flag == "dirty" else "cached"
        cost_str = format(getattr(sv, "cost_usd", 0.0), ".5f")
        name = _truncate(str(sv.name or ""), 40)
        out.write(
            f'<tr class="{cls}">'
            f"<td><code>{_h(sv.step_id)}</code></td>"
            f"<td>{_h(sv.kind)}</td>"
            f"<td>{_h(name)}</td>"
            f'<td><span class="badge badge-{cls}">{_h(dirty_flag)}</span></td>'
            f"<td>${_h(cost_str)}</td>"
            f"</tr>"
        )
    out.write("</tbody></table></section>\n")
    return out.getvalue()


def _section_pareto_front(result: MultiObjectiveMinimizationResult) -> str:
    """Pareto-front section; only rendered for multi-objective results."""
    front = result.pareto_front or []
    out = io.StringIO()
    out.write('<section id="pareto-front"><h2>Pareto front</h2>')
    if not front:
        out.write("<p><em>(No Pareto entries recorded.)</em></p></section>\n")
        return out.getvalue()
    out.write(
        "<p>Each row is a non-dominated (minimal-subset-size, objective) "
        "trade-off discovered during the multi-objective search.</p>"
    )
    # Collect all objective fields present across entries
    obj_keys: List[str] = []
    for entry in front:
        obj = getattr(entry, "objectives", None)
        if obj is not None:
            for k in _objectives_keys(obj):
                if k not in obj_keys:
                    obj_keys.append(k)
    out.write("<table><thead><tr><th>#</th><th>Subset size</th>")
    for k in obj_keys:
        out.write(f"<th>{_h(k)}</th>")
    out.write("</tr></thead><tbody>")
    for idx, entry in enumerate(front, start=1):
        subset_size = len(getattr(entry, "minimal", None) or [])
        out.write(f"<tr><td>{_h(idx)}</td><td>{_h(subset_size)}</td>")
        obj = getattr(entry, "objectives", None)
        for k in obj_keys:
            v = _objectives_get(obj, k)
            out.write(f"<td>{_h(v)}</td>")
        out.write("</tr>")
    out.write("</tbody></table></section>\n")
    return out.getvalue()


# ---- helpers for TraceObjectives (duck-typed to avoid circular imports) ----

def _objectives_keys(obj: Any) -> List[str]:
    """Return attribute names of *obj* that look like objective values."""
    known = [
        "step_count", "llm_call_count", "total_cost_usd",
        "policy_violation_count", "latency_s",
    ]
    return [k for k in known if hasattr(obj, k)]


def _objectives_get(obj: Any, key: str) -> str:
    if obj is None:
        return ""
    v = getattr(obj, key, None)
    if v is None:
        return ""
    if isinstance(v, float):
        return format(v, ".4f")
    return str(v)


# ------------------------------------------------------------------ public API


def render_html_minimize_report(
    result: "MinimizationResult",
    *,
    options: Optional[MinimizeReportOptions] = None,
) -> str:
    """Render a self-contained, offline HTML minimization report.

    Parameters
    ----------
    result :
        A :class:`~stepback.minimize.MinimizationResult` (or the
        compatible :class:`~stepback.minimize.MultiObjectiveMinimizationResult`)
        returned by :func:`~stepback.minimize_substitutions` or one of the
        strategy wrappers.
    options :
        Optional :class:`MinimizeReportOptions`.  Defaults are used when
        ``None``.

    Returns
    -------
    str
        A byte-deterministic HTML string.  All user-supplied text
        (substitution payloads, step names, metadata values) is passed
        through :func:`html.escape` before insertion.
    """
    opts = options or MinimizeReportOptions()
    is_multi = isinstance(result, MultiObjectiveMinimizationResult)

    out = io.StringIO()
    out.write("<!doctype html>\n")
    out.write('<html lang="en"><head>\n')
    out.write('<meta charset="utf-8">\n')
    out.write(f"<title>{_h(opts.title)}</title>\n")
    if opts.html_inline_css:
        out.write(f"<style>{_INLINE_CSS}</style>\n")
    out.write("</head><body>\n")
    out.write(f"<h1>{_h(opts.title)}</h1>\n")

    # Summary
    out.write(_section_summary(result, opts))

    # Before / after graph
    if opts.show_before_after:
        out.write(_section_before_after(result))

    # Minimal substitutions
    if opts.show_minimal_substitutions:
        out.write(
            _section_subs_table(
                result.minimal,
                "minimal-subs",
                "Minimal substitutions",
                result.weights,
                opts,
            )
        )

    # Removed substitutions
    if opts.show_removed_substitutions:
        out.write(
            _section_subs_table(
                result.removed,
                "removed-subs",
                "Removed substitutions (proved unnecessary)",
                None,
                opts,
            )
        )

    # Probe statistics
    if opts.show_probe_stats:
        out.write(_section_probe_stats(result))

    # Attribution / confidence summary
    if opts.show_attribution:
        out.write(_section_attribution(result))

    # Final replay
    if opts.show_final_result and result.final_result is not None:
        out.write(_section_final_result(result.final_result, opts.max_step_rows))

    # Pareto front (multi-objective only)
    if is_multi and opts.show_pareto_front:
        out.write(_section_pareto_front(result))  # type: ignore[arg-type]

    out.write("</body></html>\n")
    return out.getvalue()
