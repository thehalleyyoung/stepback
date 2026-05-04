"""Self-contained, single-file interactive HTML trace viewer.

`stepback view TRACE -o trace.html` produces an offline HTML file that
renders the recorded steps as a navigable timeline: filter by step
kind, search by text, expand a step to see its full LLM
request/response or tool args/result, and follow parent/child links
through the call tree.

The output is a single ``.html`` file with **no external CDN
dependencies** — all CSS and JavaScript are inlined and the trace
data is embedded as a JSON blob in a ``<script type="application/json">``
tag — so it can be emailed, attached to a Jira ticket, or opened
straight from the file system without a server.

This complements :func:`stepback.report.render_html_report` (which
renders a *counterfactual replay report*) by giving an interactive
read-only browser over the *recorded* trace itself.
"""
from __future__ import annotations

import html as _html
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .replay import replay


# ----------------------------------------------------------- helpers


_KIND_BADGES = {
    "llm_call": "llm",
    "tool_call": "tool",
    "router": "rt",
    "parallel_branch_open": "fork",
    "parallel_branch_join": "join",
    "exception": "exc",
}


def _short_hash(h: Optional[str]) -> str:
    if not h:
        return ""
    if h.startswith("sha256:"):
        h = h[len("sha256:"):]
    return h[:10]


def _safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _step_summary_text(step: dict) -> str:
    kind = step.get("step_kind")
    if kind == "llm_call":
        msgs = (
            _safe_get(step, "inputs", "messages")
            or _safe_get(step, "llm_request", "messages")
            or _safe_get(step, "inputs", "llm_request", "messages")
            or []
        )
        if msgs:
            last = msgs[-1]
            content = last.get("content")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            content = str(content or "")
            return f"{last.get('role', '?')}: {content[:160]}"
        return (
            _safe_get(step, "inputs", "model")
            or _safe_get(step, "llm_request", "model")
            or ""
        )
    if kind == "tool_call":
        args = (
            _safe_get(step, "inputs", "arguments")
            or _safe_get(step, "inputs", "tool_args")
            or _safe_get(step, "inputs", "args")
            or step.get("tool_args")
            or {}
        )
        name = step.get("name") or _safe_get(step, "inputs", "name") or ""
        try:
            arg_str = json.dumps(args)
        except Exception:
            arg_str = str(args)
        return f"{name}({arg_str[:140]})" if name else arg_str[:160]
    if kind == "exception":
        return str(step.get("error_class") or step.get("error") or "")[:160]
    return ""


def _step_to_view(step: dict) -> Dict[str, Any]:
    """Project a recorded step into the viewer's display schema."""
    kind = step.get("step_kind", "")
    out: Dict[str, Any] = {
        "step_id": step.get("step_id"),
        "kind": kind,
        "kind_badge": _KIND_BADGES.get(kind, kind[:4] if kind else "?"),
        "name": step.get("name") or "",
        "parent_step_id": step.get("parent_step_id"),
        "cost_usd": float(step.get("cost_usd") or 0.0),
        "wallclock_ns": step.get("wallclock_ns"),
        "cpu_ns": step.get("cpu_ns"),
        "inputs_hash": _short_hash(step.get("inputs_hash")),
        "outputs_hash": _short_hash(step.get("outputs_hash")),
        "nondeterminism_hash": _short_hash(step.get("nondeterminism_hash")),
        "summary": _step_summary_text(step),
        "raw": step,
    }
    return out


def _build_view_model(steps: Sequence[dict], header: dict) -> Dict[str, Any]:
    views = [_step_to_view(s) for s in steps]
    total_cost = sum(v["cost_usd"] for v in views)
    by_kind: Dict[str, int] = {}
    for v in views:
        by_kind[v["kind"]] = by_kind.get(v["kind"], 0) + 1
    return {
        "header": {
            "magic": header.get("magic"),
            "format_version": header.get("format_version"),
            "canonicalisation_version": header.get("canonicalisation_version"),
            "recorder_version": header.get("recorder_version"),
            "price_list_version": header.get("price_list_version"),
            "public_key": header.get("public_key"),
        },
        "step_count": len(views),
        "total_cost_usd": total_cost,
        "by_kind": by_kind,
        "steps": views,
    }


# ------------------------------------------------------- HTML assets


_INLINE_CSS = """
:root {
  --bg: #0f1419; --panel: #161c24; --panel2: #1d242e; --fg: #d8dee4;
  --muted: #8a96a3; --accent: #6ab0ff; --warn: #f4a261;
  --err: #ef6b6b; --ok: #6cc070; --border: #283341;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--bg); color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
  "Helvetica Neue", Arial, sans-serif; font-size: 13px; height: 100%; }
header { padding: 10px 16px; background: var(--panel); border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
header h1 { margin: 0; font-size: 15px; font-weight: 600; }
header .meta { color: var(--muted); font-size: 12px; }
header input[type=search] { background: var(--panel2); border: 1px solid var(--border);
  color: var(--fg); padding: 4px 8px; border-radius: 4px; min-width: 220px; }
header label { color: var(--muted); font-size: 12px; cursor: pointer; user-select: none; }
header label input { vertical-align: middle; margin-right: 4px; }
main { display: flex; height: calc(100% - 50px); }
#timeline { width: 46%; min-width: 360px; max-width: 720px; overflow-y: auto;
  border-right: 1px solid var(--border); }
#detail { flex: 1; overflow-y: auto; padding: 16px; }
.step { padding: 8px 12px; border-bottom: 1px solid var(--border); cursor: pointer;
  display: flex; align-items: flex-start; gap: 8px; }
.step:hover { background: var(--panel); }
.step.selected { background: var(--panel2); border-left: 3px solid var(--accent);
  padding-left: 9px; }
.step .badge { display: inline-block; padding: 1px 6px; border-radius: 3px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 10px;
  background: var(--panel2); color: var(--fg); border: 1px solid var(--border);
  min-width: 36px; text-align: center; }
.step .badge.k-llm_call { background: #1e3a5f; color: #cfe1ff; border-color: #2a4d7a; }
.step .badge.k-tool_call { background: #4a3a1c; color: #ffe2b3; border-color: #6a5527; }
.step .badge.k-router { background: #2e1c4a; color: #d6bdff; border-color: #432a6e; }
.step .badge.k-exception { background: #4a1e1e; color: #ffc3c3; border-color: #7a2a2a; }
.step .meta { flex: 1; min-width: 0; }
.step .id { font-family: ui-monospace, monospace; font-size: 11px; color: var(--muted); }
.step .title { font-weight: 500; }
.step .summary { color: var(--muted); font-size: 11px; margin-top: 2px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.step .cost { color: var(--ok); font-family: ui-monospace, monospace; font-size: 10px; }
.step.hidden { display: none; }
#detail h2 { margin-top: 0; font-size: 16px; }
#detail .row { margin: 4px 0; }
#detail .row .k { color: var(--muted); display: inline-block; min-width: 140px; }
#detail .row .v { font-family: ui-monospace, monospace; font-size: 12px; }
#detail pre { background: var(--panel); border: 1px solid var(--border);
  padding: 8px; border-radius: 4px; overflow: auto; max-height: 380px;
  white-space: pre-wrap; word-break: break-word; font-size: 12px; }
#detail .section { margin-top: 14px; }
#detail .section h3 { font-size: 13px; margin: 8px 0 4px; color: var(--accent); }
.linkbtn { background: none; border: 1px solid var(--border); color: var(--accent);
  padding: 2px 8px; border-radius: 3px; cursor: pointer; font-size: 11px;
  font-family: ui-monospace, monospace; }
.linkbtn:hover { background: var(--panel); }
.empty { color: var(--muted); padding: 20px; text-align: center; }
"""


_INLINE_JS = r"""
(function(){
  var raw = document.getElementById('stepback-data').textContent;
  var data = JSON.parse(raw);
  window.__sbData = data;
  var listEl = document.getElementById('timeline');
  var detailEl = document.getElementById('detail');
  var searchEl = document.getElementById('q');
  var kindBoxes = Array.prototype.slice.call(
      document.querySelectorAll('input[data-kind]'));
  var idIndex = {};

  function el(tag, attrs, children){
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs){
      if (k === 'class') e.className = attrs[k];
      else if (k === 'text') e.textContent = attrs[k];
      else e.setAttribute(k, attrs[k]);
    }
    if (children) children.forEach(function(c){
      if (c == null) return;
      if (typeof c === 'string') e.appendChild(document.createTextNode(c));
      else e.appendChild(c);
    });
    return e;
  }

  function renderList(){
    listEl.innerHTML = '';
    data.steps.forEach(function(s, i){
      var row = el('div', {class: 'step', 'data-id': s.step_id, 'data-kind': s.kind,
                           'data-search': ((s.step_id||'') + ' ' + (s.name||'') + ' ' +
                                           (s.kind||'') + ' ' + (s.summary||'')).toLowerCase()});
      row.appendChild(el('span', {class: 'badge k-' + s.kind, text: s.kind_badge}));
      var meta = el('div', {class: 'meta'});
      meta.appendChild(el('div', {class: 'title', text: (s.name || s.kind) + '  '},
        [el('span', {class: 'id', text: '#' + (i+1) + ' ' + s.step_id})]));
      meta.appendChild(el('div', {class: 'summary', text: s.summary || ''}));
      meta.appendChild(el('div', {class: 'cost',
        text: '$' + (s.cost_usd||0).toFixed(5) +
              '   in=' + (s.inputs_hash||'-') +
              '   out=' + (s.outputs_hash||'-')}));
      row.appendChild(meta);
      row.addEventListener('click', function(){ selectStep(s.step_id); });
      idIndex[s.step_id] = row;
      listEl.appendChild(row);
    });
  }

  function applyFilters(){
    var q = (searchEl.value || '').toLowerCase().trim();
    var allowed = {};
    kindBoxes.forEach(function(b){ if (b.checked) allowed[b.getAttribute('data-kind')] = true; });
    var visible = 0;
    Array.prototype.slice.call(listEl.children).forEach(function(row){
      var k = row.getAttribute('data-kind');
      var s = row.getAttribute('data-search');
      var hit = (allowed[k] !== undefined) && (q === '' || s.indexOf(q) >= 0);
      if (hit) { row.classList.remove('hidden'); visible++; }
      else { row.classList.add('hidden'); }
    });
    document.getElementById('count').textContent =
        visible + ' / ' + data.steps.length + ' steps';
  }

  function selectStep(id){
    Array.prototype.slice.call(listEl.querySelectorAll('.step.selected'))
        .forEach(function(r){ r.classList.remove('selected'); });
    var row = idIndex[id];
    if (row){ row.classList.add('selected'); row.scrollIntoView({block: 'nearest'}); }
    var step = null;
    for (var i=0;i<data.steps.length;i++){ if (data.steps[i].step_id === id){ step = data.steps[i]; break; } }
    if (!step){ detailEl.innerHTML = '<div class="empty">step not found</div>'; return; }
    renderDetail(step);
  }

  function fmt(v){ try { return JSON.stringify(v, null, 2); } catch(e){ return String(v); } }
  function row(k, v){
    return el('div', {class: 'row'},
      [el('span', {class: 'k', text: k}), el('span', {class: 'v', text: v == null ? '—' : String(v)})]);
  }

  function renderDetail(step){
    detailEl.innerHTML = '';
    detailEl.appendChild(el('h2', {text: (step.name || step.kind) + '  '},
      [el('span', {class: 'id', text: step.step_id})]));
    detailEl.appendChild(row('kind', step.kind));
    detailEl.appendChild(row('cost (USD)', '$' + (step.cost_usd||0).toFixed(6)));
    detailEl.appendChild(row('wallclock (ns)', step.wallclock_ns));
    detailEl.appendChild(row('cpu (ns)', step.cpu_ns));
    detailEl.appendChild(row('inputs hash',  step.inputs_hash));
    detailEl.appendChild(row('outputs hash', step.outputs_hash));
    detailEl.appendChild(row('nondet hash',  step.nondeterminism_hash));
    if (step.parent_step_id){
      var pbtn = el('button', {class: 'linkbtn', text: '→ ' + step.parent_step_id});
      pbtn.addEventListener('click', function(){ selectStep(step.parent_step_id); });
      var prow = el('div', {class: 'row'},
        [el('span', {class: 'k', text: 'parent'}), pbtn]);
      detailEl.appendChild(prow);
    }
    var children = data.steps.filter(function(s){ return s.parent_step_id === step.step_id; });
    if (children.length){
      var crow = el('div', {class: 'row'}, [el('span', {class: 'k', text: 'children'})]);
      children.forEach(function(c){
        var b = el('button', {class: 'linkbtn', text: c.step_id + ' (' + c.kind + ')'});
        b.addEventListener('click', function(){ selectStep(c.step_id); });
        crow.appendChild(b);
        crow.appendChild(document.createTextNode(' '));
      });
      detailEl.appendChild(crow);
    }
    var sec = el('div', {class: 'section'},
      [el('h3', {text: 'raw step (canonicalised)'}),
       el('pre', {text: fmt(step.raw)})]);
    detailEl.appendChild(sec);
  }

  renderList();
  applyFilters();
  searchEl.addEventListener('input', applyFilters);
  kindBoxes.forEach(function(b){ b.addEventListener('change', applyFilters); });
  if (data.steps.length) selectStep(data.steps[0].step_id);
})();
"""


# --------------------------------------------------------- rendering


def render_trace_html(
    steps: Sequence[dict],
    header: Optional[Dict[str, Any]] = None,
    *,
    title: str = "stepback trace",
) -> str:
    """Render a self-contained interactive HTML viewer for ``steps``.

    The returned string is a complete ``<!doctype html>`` document with
    inlined CSS, JavaScript, and a JSON data island. No network
    requests are made when the page loads.
    """
    header = header or {}
    model = _build_view_model(list(steps), header)
    data_json = json.dumps(model, separators=(",", ":"), default=str)
    # Defend against `</script>` injection from any user-controlled text
    # (e.g. an LLM that returned that literal token in its response).
    data_json = data_json.replace("</", "<\\/")

    by_kind = model["by_kind"]
    kind_chips = " ".join(
        f"<label><input type='checkbox' data-kind='{_html.escape(k)}' "
        f"checked>{_html.escape(k)} ({n})</label>"
        for k, n in sorted(by_kind.items())
    )

    parts: List[str] = []
    parts.append("<!doctype html>")
    parts.append("<html lang='en'><head>")
    parts.append("<meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append(f"<title>{_html.escape(title)}</title>")
    parts.append(f"<style>{_INLINE_CSS}</style>")
    parts.append("</head><body>")
    parts.append("<header>")
    parts.append(f"<h1>{_html.escape(title)}</h1>")
    parts.append(
        "<div class='meta'>"
        f"steps: {model['step_count']} &middot; "
        f"total cost: ${model['total_cost_usd']:.5f} &middot; "
        f"recorder: {_html.escape(str(model['header'].get('recorder_version') or 'n/a'))}"
        "</div>"
    )
    parts.append("<input type='search' id='q' placeholder='search steps...'>")
    parts.append(kind_chips)
    parts.append("<div class='meta' id='count'></div>")
    parts.append("</header>")
    parts.append("<main>")
    parts.append("<div id='timeline'></div>")
    parts.append("<div id='detail'><div class='empty'>select a step</div></div>")
    parts.append("</main>")
    parts.append(
        "<script type='application/json' id='stepback-data'>"
        + data_json
        + "</script>"
    )
    parts.append(f"<script>{_INLINE_JS}</script>")
    parts.append("</body></html>")
    return "\n".join(parts)


@dataclass
class TraceViewSummary:
    """Side-channel summary returned by :func:`write_trace_html`."""

    output_path: str
    step_count: int
    total_cost_usd: float
    by_kind: Dict[str, int] = field(default_factory=dict)
    bytes_written: int = 0


def write_trace_html(
    trace_path: str,
    output_path: str,
    *,
    hmac_key: Optional[bytes] = None,
    title: Optional[str] = None,
) -> TraceViewSummary:
    """Read ``trace_path``, render the interactive viewer, write to ``output_path``.

    If the parent directory of ``output_path`` does not exist it is
    created. ``hmac_key`` is forwarded to :func:`stepback.replay.replay`
    when provided.
    """
    t = replay(trace_path, hmac_key=hmac_key) if hmac_key is not None else replay(trace_path)
    page = render_trace_html(
        t.recorded_steps,
        t.header,
        title=title or f"stepback trace: {os.path.basename(trace_path)}",
    )
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        n = f.write(page)
    by_kind: Dict[str, int] = {}
    for s in t.recorded_steps:
        k = s.get("step_kind", "")
        by_kind[k] = by_kind.get(k, 0) + 1
    total_cost = sum(float(s.get("cost_usd") or 0.0) for s in t.recorded_steps)
    return TraceViewSummary(
        output_path=output_path,
        step_count=len(t.recorded_steps),
        total_cost_usd=total_cost,
        by_kind=by_kind,
        bytes_written=n,
    )


__all__ = [
    "TraceViewSummary",
    "render_trace_html",
    "write_trace_html",
]
