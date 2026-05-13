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
    "TimeTravelSummary",
    "render_time_travel_html",
    "write_time_travel_html",
    "FullReportSummary",
    "render_full_html_report",
    "write_full_html_report",
]


# =========================================================== Time-Travel Debugger
# Step 79: web time-travel debugger with step forward/back, cache-hit display,
# canonical input diffs, and causal graph view.
# ==========================================================


def _flat_diff(
    a: Any,
    b: Any,
    path: str = "",
    out: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Compute a flat list of JSON path differences between ``a`` and ``b``.

    Each entry is a dict with keys ``path``, ``old``, ``new``, and
    ``kind`` (``"changed"``, ``"added"``, or ``"removed"``).  Only
    leaf-level changes are emitted; unchanged subtrees are omitted.
    """
    if out is None:
        out = []
    if a == b:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        all_keys = set(a) | set(b)
        for k in sorted(all_keys):
            child_path = f"{path}.{k}" if path else k
            if k not in a:
                out.append({"path": child_path, "old": None, "new": b[k], "kind": "added"})
            elif k not in b:
                out.append({"path": child_path, "old": a[k], "new": None, "kind": "removed"})
            else:
                _flat_diff(a[k], b[k], child_path, out)
    elif isinstance(a, list) and isinstance(b, list):
        max_len = max(len(a), len(b))
        for i in range(max_len):
            child_path = f"{path}[{i}]"
            if i >= len(a):
                out.append({"path": child_path, "old": None, "new": b[i], "kind": "added"})
            elif i >= len(b):
                out.append({"path": child_path, "old": a[i], "new": None, "kind": "removed"})
            else:
                _flat_diff(a[i], b[i], child_path, out)
    else:
        out.append({"path": path or "(root)", "old": a, "new": b, "kind": "changed"})
    return out


def _build_time_travel_model(
    replay_result: Any,  # ReplayResult
    recorded_steps: Sequence[dict],
    header: dict,
) -> Dict[str, Any]:
    """Build the JSON data island for the time-travel debugger.

    ``replay_result`` is a :class:`~stepback.replay.ReplayResult` from a
    ``Trace.replay_forward()`` call (with or without substitutions).
    ``recorded_steps`` is ``Trace.recorded_steps`` — the original raw dicts.
    """
    # Build a lookup from step_id → recorded step for diff computation.
    recorded_by_id: Dict[str, dict] = {
        s["step_id"]: s for s in recorded_steps if "step_id" in s
    }

    steps: List[Dict[str, Any]] = []
    for sv in replay_result.steps:
        rec = recorded_by_id.get(sv.step_id, {})

        # Recorded inputs come from the original trace step dict.
        rec_inputs = rec.get("inputs") or {}
        cur_inputs = sv.inputs or {}

        # Compute canonical input diff (only for dirty steps with changed inputs).
        inputs_diff: Optional[List[Dict[str, Any]]] = None
        if sv.dirty and sv.recorded_inputs_hash != sv.current_inputs_hash:
            diff = _flat_diff(rec_inputs, cur_inputs)
            if diff:
                inputs_diff = diff

        dirty_reason: Optional[str] = None
        cache_source: str = "recorded_trace"
        if sv.provenance is not None:
            dirty_reason = sv.provenance.dirty_reason
            cache_source = sv.provenance.cache_source

        kind = sv.kind or ""
        steps.append({
            "step_id": sv.step_id,
            "index": len(steps),
            "kind": kind,
            "kind_badge": _KIND_BADGES.get(kind, kind[:4] if kind else "?"),
            "name": sv.name or "",
            "parent_step_id": sv.parent_step_id,
            "dirty": sv.dirty,
            "cache_hit": sv.cache_hit,
            "dirty_reason": dirty_reason,
            "cache_source": cache_source,
            "recorded_inputs_hash": sv.recorded_inputs_hash,
            "current_inputs_hash": sv.current_inputs_hash,
            "inputs_diff": inputs_diff,
            "recorded_inputs": rec_inputs,
            "current_inputs": cur_inputs,
            "outputs": sv.outputs,
            "cost_usd": float(sv.cost_usd or 0.0),
            "summary": _step_summary_text(rec or {"step_kind": kind, "inputs": cur_inputs}),
            "wallclock_ns": rec.get("wallclock_ns"),
        })

    total_cost = sum(s["cost_usd"] for s in steps)
    dirty_count = sum(1 for s in steps if s["dirty"])
    cache_hit_count = sum(1 for s in steps if s["cache_hit"])
    by_kind: Dict[str, int] = {}
    for s in steps:
        by_kind[s["kind"]] = by_kind.get(s["kind"], 0) + 1

    return {
        "header": {
            "magic": header.get("magic"),
            "format_version": header.get("format_version"),
            "recorder_version": header.get("recorder_version"),
        },
        "step_count": len(steps),
        "dirty_count": dirty_count,
        "cache_hit_count": cache_hit_count,
        "total_cost_usd": total_cost,
        "by_kind": by_kind,
        "steps": steps,
    }


# ----------------------------- CSS and JS assets for the time-travel viewer

_TT_CSS = """
:root {
  --bg: #0f1419; --panel: #161c24; --panel2: #1d242e; --fg: #d8dee4;
  --muted: #8a96a3; --accent: #6ab0ff; --warn: #f4a261;
  --err: #ef6b6b; --ok: #6cc070; --border: #283341;
  --dirty: #f4a261; --hit: #6cc070; --neutral: #8a96a3;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--bg); color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
  "Helvetica Neue", Arial, sans-serif; font-size: 13px; height: 100%; }

/* ---- header ---- */
header { padding: 8px 14px; background: var(--panel); border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
header h1 { margin: 0; font-size: 14px; font-weight: 600; }
.meta-bar { color: var(--muted); font-size: 12px; }
.nav-group { display: flex; align-items: center; gap: 6px; }
.nav-btn { background: var(--panel2); border: 1px solid var(--border); color: var(--fg);
  padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; }
.nav-btn:hover { background: var(--border); }
.nav-btn:disabled { opacity: 0.4; cursor: default; }
#step-pos { font-family: ui-monospace, monospace; font-size: 12px; min-width: 80px;
  text-align: center; color: var(--accent); }
.badge-pill { display: inline-block; padding: 2px 8px; border-radius: 10px;
  font-size: 11px; font-weight: 600; }
.badge-dirty { background: #4a2e0a; color: var(--dirty); }
.badge-hit   { background: #0d3317; color: var(--hit); }
.mode-btn { background: var(--panel2); border: 1px solid var(--border); color: var(--muted);
  padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; }
.mode-btn.active { color: var(--accent); border-color: var(--accent); }
header input[type=search] { background: var(--panel2); border: 1px solid var(--border);
  color: var(--fg); padding: 4px 8px; border-radius: 4px; min-width: 180px; }

/* ---- layout ---- */
main { display: flex; height: calc(100% - 46px); }
#timeline { width: 280px; min-width: 200px; max-width: 340px; overflow-y: auto;
  border-right: 1px solid var(--border); flex-shrink: 0; }
#right-pane { flex: 1; overflow: hidden; display: flex; flex-direction: column; }

/* ---- step list ---- */
.step { padding: 6px 10px; border-bottom: 1px solid var(--border); cursor: pointer;
  display: flex; align-items: flex-start; gap: 6px; }
.step:hover { background: var(--panel); }
.step.selected { background: var(--panel2); border-left: 3px solid var(--accent);
  padding-left: 7px; }
.step.is-dirty { border-left: 3px solid var(--dirty); padding-left: 7px; }
.step.is-dirty.selected { border-left: 3px solid var(--accent); }
.step .badge { display: inline-block; padding: 1px 5px; border-radius: 3px;
  font-family: ui-monospace, monospace; font-size: 10px;
  background: var(--panel2); color: var(--fg); border: 1px solid var(--border);
  min-width: 32px; text-align: center; }
.step .badge.k-llm_call { background: #1e3a5f; color: #cfe1ff; border-color: #2a4d7a; }
.step .badge.k-tool_call { background: #4a3a1c; color: #ffe2b3; border-color: #6a5527; }
.step .badge.k-router { background: #2e1c4a; color: #d6bdff; border-color: #432a6e; }
.step .badge.k-exception { background: #4a1e1e; color: #ffc3c3; border-color: #7a2a2a; }
.step .smeta { flex: 1; min-width: 0; }
.step .stitle { font-size: 11px; font-weight: 500;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.step .ssummary { font-size: 10px; color: var(--muted); margin-top: 1px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.step .status-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
  margin-top: 4px; }
.dot-hit   { background: var(--hit); }
.dot-dirty { background: var(--dirty); }
.dot-none  { background: var(--neutral); opacity: 0.4; }
.step.hidden { display: none; }

/* ---- detail pane ---- */
#detail-pane { flex: 1; overflow-y: auto; padding: 14px 18px;
  display: none; }
#detail-pane.active { display: block; }
#detail-pane h2 { margin: 0 0 10px; font-size: 15px; }
.drow { margin: 3px 0; }
.drow .dk { color: var(--muted); display: inline-block; min-width: 160px; font-size: 12px; }
.drow .dv { font-family: ui-monospace, monospace; font-size: 12px; }
.drow .dv.dirty { color: var(--dirty); }
.drow .dv.hit { color: var(--hit); }
.section { margin-top: 14px; }
.section h3 { font-size: 12px; margin: 6px 0 4px; color: var(--accent); text-transform: uppercase;
  letter-spacing: 0.05em; }
pre.code { background: var(--panel); border: 1px solid var(--border);
  padding: 8px; border-radius: 4px; overflow: auto; max-height: 280px;
  white-space: pre-wrap; word-break: break-word; font-size: 11px; margin: 0; }
.linkbtn { background: none; border: 1px solid var(--border); color: var(--accent);
  padding: 2px 7px; border-radius: 3px; cursor: pointer; font-size: 11px;
  font-family: ui-monospace, monospace; }
.linkbtn:hover { background: var(--panel); }
.empty { color: var(--muted); padding: 20px; text-align: center; }

/* ---- diff pane ---- */
.diff-table { width: 100%; border-collapse: collapse; font-size: 11px;
  font-family: ui-monospace, monospace; }
.diff-table th { color: var(--muted); text-align: left; padding: 2px 6px;
  border-bottom: 1px solid var(--border); font-weight: normal; }
.diff-table td { padding: 2px 6px; vertical-align: top; }
.diff-table tr.diff-changed td.old { color: var(--err); }
.diff-table tr.diff-changed td.new { color: var(--ok); }
.diff-table tr.diff-added td.new   { color: var(--ok); }
.diff-table tr.diff-removed td.old { color: var(--err); }
.diff-table td.path { color: var(--muted); }
.diff-none { color: var(--muted); font-size: 11px; padding: 8px 0; }

/* ---- graph pane ---- */
#graph-pane { flex: 1; overflow: auto; padding: 14px;
  display: none; }
#graph-pane.active { display: block; }
#graph-svg { display: block; }
.gnode rect { rx: 5; fill: var(--panel2); stroke: var(--border); stroke-width: 1; cursor: pointer; }
.gnode rect.dirty   { stroke: var(--dirty); }
.gnode rect.hit     { stroke: var(--hit); }
.gnode rect.selected { stroke: var(--accent); stroke-width: 2; }
.gnode text { font-size: 10px; fill: var(--fg); pointer-events: none;
  font-family: ui-monospace, monospace; }
.gedge { stroke: var(--border); stroke-width: 1; fill: none; }
.gedge.dirty-edge { stroke: var(--dirty); opacity: 0.5; }
"""

_TT_JS = r"""
(function(){
  var raw = document.getElementById('tt-data').textContent;
  var data = JSON.parse(raw);
  window.__sbTTData = data;

  var currentIdx = 0;
  var prevBtn   = document.getElementById('prev-btn');
  var nextBtn   = document.getElementById('next-btn');
  var stepPos   = document.getElementById('step-pos');
  var timelineEl = document.getElementById('timeline');
  var detailPane = document.getElementById('detail-pane');
  var graphPane  = document.getElementById('graph-pane');
  var detailBtn  = document.getElementById('detail-btn');
  var graphBtn   = document.getElementById('graph-btn');
  var searchEl   = document.getElementById('q');
  var idIndex    = {};
  var currentMode = 'detail';

  /* ---------- helpers ---------- */
  function el(tag, attrs, children){
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs){
      if (k === 'class') e.className = attrs[k];
      else if (k === 'text') e.textContent = attrs[k];
      else if (k === 'html') e.innerHTML = attrs[k];
      else e.setAttribute(k, attrs[k]);
    }
    if (children) children.forEach(function(c){
      if (c == null) return;
      if (typeof c === 'string') e.appendChild(document.createTextNode(c));
      else e.appendChild(c);
    });
    return e;
  }

  function fmt(v){
    try { return JSON.stringify(v, null, 2); } catch(e){ return String(v); }
  }

  function drow(k, vEl){
    var d = el('div', {class: 'drow'});
    d.appendChild(el('span', {class: 'dk', text: k}));
    if (typeof vEl === 'string') d.appendChild(el('span', {class: 'dv', text: vEl}));
    else d.appendChild(vEl);
    return d;
  }

  /* ---------- step list ---------- */
  function renderList(){
    timelineEl.innerHTML = '';
    data.steps.forEach(function(s, i){
      var dot = s.dirty ? 'dot-dirty' : (s.cache_hit ? 'dot-hit' : 'dot-none');
      var rowCls = 'step' + (s.dirty ? ' is-dirty' : '');
      var row = el('div', {
        class: rowCls,
        'data-id': s.step_id,
        'data-idx': String(i),
        'data-kind': s.kind,
        'data-search': ((s.step_id||'') + ' ' + (s.name||'') + ' ' +
                        (s.kind||'') + ' ' + (s.summary||'')).toLowerCase()
      });
      row.appendChild(el('div', {class: 'status-dot ' + dot}));
      row.appendChild(el('span', {class: 'badge k-' + s.kind, text: s.kind_badge}));
      var meta = el('div', {class: 'smeta'});
      meta.appendChild(el('div', {class: 'stitle', text: (s.name || s.kind) || '?'}));
      meta.appendChild(el('div', {class: 'ssummary', text: s.summary || ('$' + (s.cost_usd||0).toFixed(5))}));
      row.appendChild(meta);
      row.addEventListener('click', function(){ goToIndex(i); });
      idIndex[s.step_id] = {row: row, idx: i};
      timelineEl.appendChild(row);
    });
  }

  function applyFilter(){
    var q = (searchEl.value || '').toLowerCase().trim();
    Array.prototype.slice.call(timelineEl.children).forEach(function(row){
      var s = row.getAttribute('data-search') || '';
      var hit = (q === '' || s.indexOf(q) >= 0);
      if (hit) row.classList.remove('hidden');
      else row.classList.add('hidden');
    });
  }

  /* ---------- navigation ---------- */
  function goToIndex(idx){
    if (idx < 0 || idx >= data.steps.length) return;
    currentIdx = idx;
    var step = data.steps[idx];

    // update list selection
    Array.prototype.slice.call(timelineEl.querySelectorAll('.step.selected'))
        .forEach(function(r){ r.classList.remove('selected'); });
    var entry = idIndex[step.step_id];
    if (entry){ entry.row.classList.add('selected'); entry.row.scrollIntoView({block: 'nearest'}); }

    // update nav controls
    prevBtn.disabled = (idx === 0);
    nextBtn.disabled = (idx === data.steps.length - 1);
    stepPos.textContent = 'Step ' + (idx + 1) + ' / ' + data.steps.length;

    if (currentMode === 'detail') renderDetail(step);
    if (currentMode === 'graph') highlightNode(step.step_id);
  }

  prevBtn.addEventListener('click', function(){ goToIndex(currentIdx - 1); });
  nextBtn.addEventListener('click', function(){ goToIndex(currentIdx + 1); });
  document.addEventListener('keydown', function(e){
    if (e.target.tagName === 'INPUT') return;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp')   { e.preventDefault(); goToIndex(currentIdx - 1); }
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { e.preventDefault(); goToIndex(currentIdx + 1); }
  });

  /* ---------- mode toggle ---------- */
  function setMode(m){
    currentMode = m;
    detailBtn.classList.toggle('active', m === 'detail');
    graphBtn.classList.toggle('active',  m === 'graph');
    detailPane.classList.toggle('active', m === 'detail');
    graphPane.classList.toggle('active',  m === 'graph');
    if (m === 'graph' && !graphPane.dataset.built) buildGraph();
    if (m === 'detail') renderDetail(data.steps[currentIdx]);
  }

  detailBtn.addEventListener('click', function(){ setMode('detail'); });
  graphBtn.addEventListener('click',  function(){ setMode('graph');  });

  /* ---------- detail pane ---------- */
  function renderDetail(step){
    detailPane.innerHTML = '';

    var statusTxt = step.dirty
      ? (step.dirty_reason ? 'dirty — ' + step.dirty_reason : 'dirty')
      : (step.cache_hit ? 'cache hit' : 'clean');
    var statusCls = step.dirty ? 'dv dirty' : (step.cache_hit ? 'dv hit' : 'dv');

    detailPane.appendChild(el('h2', {text: (step.name || step.kind) + '  '},
      [el('span', {class: 'dv', style: 'font-size:12px;color:var(--muted)',
                  text: '#' + (step.index + 1) + ' ' + step.step_id})]));

    detailPane.appendChild(drow('kind', step.kind));

    var sv = el('span', {class: statusCls, text: statusTxt});
    detailPane.appendChild(drow('status', sv));

    if (step.dirty_reason)
      detailPane.appendChild(drow('dirty reason', step.dirty_reason));
    if (step.cache_source)
      detailPane.appendChild(drow('cache source', step.cache_source));

    detailPane.appendChild(drow('cost (USD)', '$' + (step.cost_usd||0).toFixed(6)));
    if (step.wallclock_ns != null)
      detailPane.appendChild(drow('wallclock (ns)', String(step.wallclock_ns)));

    // hashes
    detailPane.appendChild(drow('recorded inputs hash', step.recorded_inputs_hash || '—'));
    if (step.dirty && step.current_inputs_hash !== step.recorded_inputs_hash){
      var hv = el('span', {class: 'dv dirty', text: step.current_inputs_hash || '—'});
      detailPane.appendChild(drow('current inputs hash', hv));
    }

    // parent link
    if (step.parent_step_id){
      var pbtn = el('button', {class: 'linkbtn', text: '→ ' + step.parent_step_id});
      pbtn.addEventListener('click', function(){
        var e2 = idIndex[step.parent_step_id];
        if (e2) goToIndex(e2.idx);
      });
      detailPane.appendChild(drow('parent', pbtn));
    }
    // children links
    var children = data.steps.filter(function(s){ return s.parent_step_id === step.step_id; });
    if (children.length){
      var crow = el('div', {class: 'drow'}, [el('span', {class: 'dk', text: 'children'})]);
      children.forEach(function(c){
        var b = el('button', {class: 'linkbtn', text: c.step_id + ' (' + c.kind + ')'});
        (function(c2){ b.addEventListener('click', function(){
          var e3 = idIndex[c2.step_id]; if (e3) goToIndex(e3.idx);
        }); })(c);
        crow.appendChild(b);
        crow.appendChild(document.createTextNode(' '));
      });
      detailPane.appendChild(crow);
    }

    // inputs diff section (only for dirty steps with changed inputs)
    if (step.dirty && step.inputs_diff && step.inputs_diff.length > 0){
      var diffSec = el('div', {class: 'section'});
      diffSec.appendChild(el('h3', {text: 'Canonical inputs diff (recorded → current)'}));
      diffSec.appendChild(renderDiff(step.inputs_diff));
      detailPane.appendChild(diffSec);
    } else if (step.dirty && step.recorded_inputs_hash === step.current_inputs_hash){
      var diffSec2 = el('div', {class: 'section'});
      diffSec2.appendChild(el('h3', {text: 'Canonical inputs diff'}));
      diffSec2.appendChild(el('div', {class: 'diff-none',
        text: 'Inputs hash unchanged — step is dirty due to ' + (step.dirty_reason || 'ancestor')}));
      detailPane.appendChild(diffSec2);
    }

    // outputs section
    var outSec = el('div', {class: 'section'});
    outSec.appendChild(el('h3', {text: 'Outputs'}));
    outSec.appendChild(el('pre', {class: 'code', text: fmt(step.outputs)}));
    detailPane.appendChild(outSec);

    // current inputs section
    var inSec = el('div', {class: 'section'});
    inSec.appendChild(el('h3', {text: 'Current inputs'}));
    inSec.appendChild(el('pre', {class: 'code', text: fmt(step.current_inputs)}));
    detailPane.appendChild(inSec);

    // recorded inputs section (collapsed if same as current)
    if (step.dirty && step.inputs_diff && step.inputs_diff.length > 0){
      var recSec = el('div', {class: 'section'});
      recSec.appendChild(el('h3', {text: 'Recorded inputs'}));
      recSec.appendChild(el('pre', {class: 'code', text: fmt(step.recorded_inputs)}));
      detailPane.appendChild(recSec);
    }
  }

  function renderDiff(diffs){
    var tbl = el('table', {class: 'diff-table'});
    tbl.appendChild(el('tr', {}, [
      el('th', {text: 'path'}),
      el('th', {text: 'recorded (old)'}),
      el('th', {text: 'current (new)'}),
    ]));
    diffs.forEach(function(d){
      var tr = el('tr', {class: 'diff-' + d.kind});
      tr.appendChild(el('td', {class: 'path', text: d.path}));
      tr.appendChild(el('td', {class: 'old',
        text: d.kind === 'added' ? '' : (typeof d.old === 'object' ? JSON.stringify(d.old) : String(d.old === null ? '∅' : d.old))}));
      tr.appendChild(el('td', {class: 'new',
        text: d.kind === 'removed' ? '' : (typeof d.new === 'object' ? JSON.stringify(d.new) : String(d.new === null ? '∅' : d.new))}));
      tbl.appendChild(tr);
    });
    return tbl;
  }

  /* ---------- causal graph pane ---------- */
  var NS = 'http://www.w3.org/2000/svg';
  var NODE_W = 160, NODE_H = 28, H_GAP = 40, V_GAP = 20;

  function buildGraph(){
    graphPane.dataset.built = '1';
    graphPane.innerHTML = '';

    // Assign columns by topological order; rows within each column
    var idToStep = {};
    data.steps.forEach(function(s){ idToStep[s.step_id] = s; });

    // Simple rank assignment: rank(step) = max(rank(parent)) + 1
    var rank = {};
    data.steps.forEach(function(s){ rank[s.step_id] = 0; });
    data.steps.forEach(function(s){
      if (s.parent_step_id && rank[s.parent_step_id] !== undefined){
        if (rank[s.parent_step_id] + 1 > rank[s.step_id])
          rank[s.step_id] = rank[s.parent_step_id] + 1;
      }
    });

    var maxRank = 0;
    for (var id in rank) if (rank[id] > maxRank) maxRank = rank[id];

    // Group steps by rank
    var byRank = {};
    for (var i = 0; i <= maxRank; i++) byRank[i] = [];
    data.steps.forEach(function(s){ byRank[rank[s.step_id]].push(s); });

    var maxInCol = 1;
    for (var r = 0; r <= maxRank; r++)
      if (byRank[r].length > maxInCol) maxInCol = byRank[r].length;

    var svgW = (maxRank + 1) * (NODE_W + H_GAP) + H_GAP;
    var svgH = maxInCol * (NODE_H + V_GAP) + V_GAP;

    var svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('id', 'graph-svg');
    svg.setAttribute('width', String(svgW));
    svg.setAttribute('height', String(svgH));

    // defs for arrowhead
    var defs = document.createElementNS(NS, 'defs');
    var marker = document.createElementNS(NS, 'marker');
    marker.setAttribute('id', 'arrow');
    marker.setAttribute('markerWidth', '6');
    marker.setAttribute('markerHeight', '6');
    marker.setAttribute('refX', '6');
    marker.setAttribute('refY', '3');
    marker.setAttribute('orient', 'auto');
    var mp = document.createElementNS(NS, 'path');
    mp.setAttribute('d', 'M0,0 L0,6 L6,3 z');
    mp.setAttribute('fill', '#283341');
    marker.appendChild(mp);
    defs.appendChild(marker);
    svg.appendChild(defs);

    // Compute node positions
    var pos = {};
    for (var r2 = 0; r2 <= maxRank; r2++){
      var col = byRank[r2];
      var colX = H_GAP + r2 * (NODE_W + H_GAP);
      col.forEach(function(s, ri){
        var colY = V_GAP + ri * (NODE_H + V_GAP);
        pos[s.step_id] = {x: colX, y: colY};
      });
    }

    // Draw edges first (so they appear behind nodes)
    data.steps.forEach(function(s){
      if (!s.parent_step_id || !pos[s.parent_step_id] || !pos[s.step_id]) return;
      var p = pos[s.parent_step_id];
      var c = pos[s.step_id];
      var x1 = p.x + NODE_W, y1 = p.y + NODE_H/2;
      var x2 = c.x, y2 = c.y + NODE_H/2;
      var path = document.createElementNS(NS, 'path');
      var cx1 = x1 + (x2 - x1) * 0.5, cx2 = x2 - (x2 - x1) * 0.5;
      path.setAttribute('d', 'M' + x1 + ',' + y1 + ' C' + cx1 + ',' + y1 + ' ' + cx2 + ',' + y2 + ' ' + x2 + ',' + y2);
      path.setAttribute('class', 'gedge' + (s.dirty ? ' dirty-edge' : ''));
      path.setAttribute('marker-end', 'url(#arrow)');
      svg.appendChild(path);
    });

    // Draw nodes
    data.steps.forEach(function(s){
      var p = pos[s.step_id];
      if (!p) return;
      var g = document.createElementNS(NS, 'g');
      g.setAttribute('class', 'gnode');
      g.setAttribute('data-id', s.step_id);

      var statusCls = s.dirty ? 'dirty' : (s.cache_hit ? 'hit' : '');
      var rect = document.createElementNS(NS, 'rect');
      rect.setAttribute('x', String(p.x));
      rect.setAttribute('y', String(p.y));
      rect.setAttribute('width', String(NODE_W));
      rect.setAttribute('height', String(NODE_H));
      rect.setAttribute('rx', '4');
      rect.setAttribute('class', statusCls);
      g.appendChild(rect);

      var label = (s.name || s.kind || '?').substring(0, 20);
      var txt = document.createElementNS(NS, 'text');
      txt.setAttribute('x', String(p.x + 6));
      txt.setAttribute('y', String(p.y + 18));
      txt.textContent = label;
      g.appendChild(txt);

      g.addEventListener('click', function(){
        var e2 = idIndex[s.step_id]; if (e2) goToIndex(e2.idx);
      });
      svg.appendChild(g);
    });

    graphPane.appendChild(svg);
  }

  function highlightNode(stepId){
    if (!graphPane.dataset.built) return;
    var allNodes = graphPane.querySelectorAll('.gnode rect');
    Array.prototype.slice.call(allNodes).forEach(function(r){
      r.classList.remove('selected');
    });
    var g = graphPane.querySelector('.gnode[data-id="' + stepId + '"] rect');
    if (g) g.classList.add('selected');
  }

  /* ---------- search ---------- */
  searchEl.addEventListener('input', applyFilter);

  /* ---------- init ---------- */
  renderList();
  applyFilter();
  setMode('detail');
  if (data.steps.length) goToIndex(0);
})();
"""


def render_time_travel_html(
    replay_result: Any,  # ReplayResult
    recorded_steps: Sequence[dict],
    header: Optional[Dict[str, Any]] = None,
    *,
    title: str = "stepback time-travel debugger",
) -> str:
    """Render a self-contained interactive time-travel debugger for a replay.

    The returned page embeds:

    * **Step forward/back navigation** — Prev/Next buttons and arrow-key
      shortcuts to walk the replay step by step.
    * **Cache-hit display** — colour-coded status dots (green = cache hit,
      orange = dirty/re-executed) and a header summary badge.
    * **Canonical input diffs** — for dirty steps whose inputs changed, a
      table showing every leaf-level difference between the recorded and
      post-substitution inputs.
    * **Causal graph view** — an SVG DAG of parent/child relationships;
      selecting a node in the graph syncs the step list and detail pane.

    All assets are inlined; the page requires no network access.

    Parameters
    ----------
    replay_result:
        A :class:`~stepback.replay.ReplayResult` produced by
        ``Trace.replay_forward()`` (or the module-level ``replay()``
        helper).
    recorded_steps:
        The raw step dicts from the original trace
        (``Trace.recorded_steps``).  Used to extract recorded inputs for
        the diff view.
    header:
        Optional trace header dict (from ``Trace.header``).
    title:
        Title shown in the browser tab and page header.
    """
    header = header or {}
    model = _build_time_travel_model(replay_result, recorded_steps, header)
    data_json = json.dumps(model, separators=(",", ":"), default=str)
    data_json = data_json.replace("</", "<\\/")

    dirty_count = model["dirty_count"]
    cache_hit_count = model["cache_hit_count"]
    step_count = model["step_count"]

    parts: List[str] = []
    parts.append("<!doctype html>")
    parts.append("<html lang='en'><head>")
    parts.append("<meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append(f"<title>{_html.escape(title)}</title>")
    parts.append(f"<style>{_TT_CSS}</style>")
    parts.append("</head><body>")
    parts.append("<header>")
    parts.append(f"<h1>{_html.escape(title)}</h1>")
    parts.append(
        f"<div class='meta-bar'>{step_count} steps &middot; "
        f"total cost: ${model['total_cost_usd']:.5f}</div>"
    )
    parts.append(
        "<div class='nav-group'>"
        "<button class='nav-btn' id='prev-btn' disabled>&#8592; Prev</button>"
        "<span id='step-pos'>Step — / —</span>"
        "<button class='nav-btn' id='next-btn' disabled>Next &#8594;</button>"
        "</div>"
    )
    parts.append(
        f"<span class='badge-pill badge-dirty'>{dirty_count} dirty</span>"
        f"<span class='badge-pill badge-hit'>{cache_hit_count} cache hits</span>"
    )
    parts.append(
        "<button class='mode-btn active' id='detail-btn'>&#9776; Detail</button>"
        "<button class='mode-btn' id='graph-btn'>&#9674; Graph</button>"
    )
    parts.append("<input type='search' id='q' placeholder='search steps...'>")
    parts.append("</header>")
    parts.append("<main>")
    parts.append("<div id='timeline'></div>")
    parts.append("<div id='right-pane'>")
    parts.append("<div id='detail-pane' class='active'><div class='empty'>select a step</div></div>")
    parts.append("<div id='graph-pane'></div>")
    parts.append("</div>")
    parts.append("</main>")
    parts.append(
        "<script type='application/json' id='tt-data'>"
        + data_json
        + "</script>"
    )
    parts.append(f"<script>{_TT_JS}</script>")
    parts.append("</body></html>")
    return "\n".join(parts)


@dataclass
class TimeTravelSummary:
    """Side-channel summary returned by :func:`write_time_travel_html`."""

    output_path: str
    step_count: int
    dirty_count: int
    cache_hit_count: int
    total_cost_usd: float
    by_kind: Dict[str, int] = field(default_factory=dict)
    bytes_written: int = 0


def write_time_travel_html(
    trace_path: str,
    output_path: str,
    *,
    hmac_key: Optional[bytes] = None,
    substitutions: Optional[Any] = None,
    title: Optional[str] = None,
) -> TimeTravelSummary:
    """Read ``trace_path``, replay it (with optional substitutions), and write a
    time-travel debugger HTML file to ``output_path``.

    Parameters
    ----------
    trace_path:
        Path to a recorded ``.sb`` trace file.
    output_path:
        Destination HTML file path.  Parent directories are created if
        they do not exist.
    hmac_key:
        Optional HMAC key bytes used when loading the trace.
    substitutions:
        Optional list of :class:`~stepback.substitutions.Substitution`
        objects to apply before replaying.
    title:
        Override the page title; defaults to the trace filename.
    """
    t = replay(trace_path, hmac_key=hmac_key) if hmac_key is not None else replay(trace_path)
    subs_list = list(substitutions) if substitutions is not None else []
    result = t.substitute(*subs_list).replay_forward()

    page = render_time_travel_html(
        result,
        t.recorded_steps,
        t.header,
        title=title or f"stepback debugger: {os.path.basename(trace_path)}",
    )
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        n = f.write(page)

    model = _build_time_travel_model(result, t.recorded_steps, t.header)
    return TimeTravelSummary(
        output_path=output_path,
        step_count=model["step_count"],
        dirty_count=model["dirty_count"],
        cache_hit_count=model["cache_hit_count"],
        total_cost_usd=model["total_cost_usd"],
        by_kind=model["by_kind"],
        bytes_written=n,
    )


# =========================================================== Full HTML Report (Step 109)
# Self-contained HTML export with causal graph, diff panes, minimization
# report, and attestation summary.
# ===========================================================


def _attest_get(entry: Any, key: str, default: Any = None) -> Any:
    """Get a value from either a dict or an object with attributes."""
    if isinstance(entry, dict):
        return entry.get(key, default)
    return getattr(entry, key, default)


# CSS additions for the full report
_FR_CSS = _TT_CSS + """
/* ---- Full-report tab bar ---- */
.fr-tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border);
  background: var(--panel); }
.fr-tab { padding: 8px 18px; cursor: pointer; font-size: 13px; border: none;
  background: none; color: var(--muted); border-bottom: 2px solid transparent; }
.fr-tab.active { color: var(--accent); border-bottom-color: var(--accent); }
.fr-tab:hover { color: var(--fg); }
.fr-section { display: none; padding: 18px; overflow-y: auto;
  height: calc(100% - 42px); }
.fr-section.active { display: block; }

/* ---- Attestation section ---- */
.fr-attest-block { max-width: 680px; }
.fr-row { display: flex; gap: 8px; margin: 5px 0; font-size: 12px; }
.fr-label { color: var(--muted); min-width: 220px; flex-shrink: 0; }
.fr-val { font-family: ui-monospace, monospace; word-break: break-all; }
.fr-badge { display: inline-block; padding: 2px 10px; border-radius: 10px;
  font-size: 11px; font-weight: 600; }
.fr-badge-ok   { background: #0d3317; color: var(--ok); }
.fr-badge-fail { background: #4a1e1e; color: var(--err); }
.fr-badge-skip { background: #2a2a2a; color: var(--muted); }
.fr-id-list { font-family: ui-monospace, monospace; font-size: 11px;
  margin: 4px 0 0 228px; color: var(--muted); }
.fr-more { color: var(--warn); font-style: italic; }

/* ---- Minimization section ---- */
.fr-min-block { max-width: 720px; }
.fr-min-table { width: 100%; border-collapse: collapse; font-size: 11px;
  font-family: ui-monospace, monospace; margin-top: 8px; }
.fr-min-table th { color: var(--muted); text-align: left; padding: 3px 8px;
  border-bottom: 1px solid var(--border); font-weight: normal; }
.fr-min-table td { padding: 3px 8px; vertical-align: top;
  border-bottom: 1px solid var(--border); }
.fr-min-table td.keep { color: var(--ok); }
.fr-min-table td.removed { color: var(--err); }
.fr-min-table td.weight { color: var(--accent); }
.fr-empty { color: var(--muted); font-style: italic; }
"""


def _render_attestation_section(entry: Any) -> str:
    """Render the attestation summary HTML for a single attestation entry.

    ``entry`` may be a dict or any object with matching attributes.
    Returns an HTML fragment (not a full document).
    """
    g = lambda k, d=None: _attest_get(entry, k, d)  # noqa: E731

    verify_status = g("verify_status", "")
    replay_status = g("replay_status", "")

    def _badge(status: str) -> str:
        cls = (
            "fr-badge-ok" if status == "ok"
            else "fr-badge-fail" if status == "fail"
            else "fr-badge-skip"
        )
        label = _html.escape(str(status)) if status else "—"
        return f"<span class='fr-badge {cls}'>{label}</span>"

    def _row(label: str, val: str, val_cls: str = "") -> str:
        cls_attr = f" class='{val_cls}'" if val_cls else ""
        return (
            f"<div class='fr-row'>"
            f"<span class='fr-label'>{_html.escape(label)}</span>"
            f"<span class='fr-val'{cls_attr}>{val}</span>"
            f"</div>"
        )

    parts: List[str] = []
    parts.append("<div class='fr-attest-block'>")
    parts.append("<h2>Attestation Summary</h2>")

    # Status badges
    parts.append(
        f"<div class='fr-row'>"
        f"<span class='fr-label'>Chain verification</span>"
        f"{_badge(verify_status)}"
        f"</div>"
    )
    verify_error = g("verify_error")
    if verify_error:
        parts.append(_row("Verification error", _html.escape(str(verify_error))))

    parts.append(
        f"<div class='fr-row'>"
        f"<span class='fr-label'>Replay status</span>"
        f"{_badge(replay_status)}"
        f"</div>"
    )
    replay_error = g("replay_error")
    if replay_error:
        parts.append(_row("Replay error", _html.escape(str(replay_error))))

    # Core metadata
    pub_key = g("recorder_public_key") or ""
    parts.append(_row("Recorder public key", _html.escape(str(pub_key))))
    parts.append(_row("Recorder version", _html.escape(str(g("recorder_version") or "—"))))
    parts.append(_row("Canonicalisation version", _html.escape(str(g("canonicalisation_version") or "—"))))
    parts.append(_row("Trace path", _html.escape(str(g("trace_path") or "—"))))
    chain_hash = g("trace_chain_hash") or "—"
    parts.append(_row("Trace chain hash", _html.escape(str(chain_hash))))

    merkle_root = g("merkle_root")
    if merkle_root:
        parts.append(_row("Merkle root", _html.escape(str(merkle_root))))
    else:
        parts.append(_row("Merkle root", "<span class='fr-empty'>—</span>"))

    # Step counts and costs
    parts.append(_row("Step count", _html.escape(str(g("step_count") or "—"))))
    parts.append(_row("Dirty steps", _html.escape(str(g("dirty_step_count") or 0))))
    parts.append(_row("Cache hits", _html.escape(str(g("cache_hit_count") or 0))))
    parts.append(_row("Real executions", _html.escape(str(g("real_executions") or 0))))
    parts.append(_row("Divergent steps", _html.escape(str(g("divergent_step_count") or 0))))

    cost_rec = g("total_cost_recorded_usd")
    if cost_rec is not None:
        parts.append(_row("Cost (recorded)", f"${float(cost_rec):.6f}"))
    cost_rep = g("total_cost_replayed_usd")
    if cost_rep is not None:
        parts.append(_row("Cost (replayed)", f"${float(cost_rep):.6f}"))
    cost_delta = g("total_cost_delta_usd")
    if cost_delta is not None:
        parts.append(_row("Cost delta", f"${float(cost_delta):.6f}"))

    # Divergent step IDs
    divergent_ids = g("divergent_step_ids") or []
    try:
        divergent_ids = list(divergent_ids)
    except TypeError:
        divergent_ids = []

    if divergent_ids:
        parts.append(
            "<div class='fr-row'>"
            "<span class='fr-label'>Divergent step IDs</span>"
            f"<span class='fr-val'>{len(divergent_ids)} step(s)</span>"
            "</div>"
        )
        visible = divergent_ids[:50]
        hidden = len(divergent_ids) - 50
        id_items = "  ".join(_html.escape(str(sid)) for sid in visible)
        parts.append(f"<div class='fr-id-list'>{id_items}")
        if hidden > 0:
            parts.append(f"  <span class='fr-more'>and {hidden} more</span>")
        parts.append("</div>")

    parts.append("</div>")
    return "\n".join(parts)


def _render_minimization_section(min_result: Any) -> str:
    """Render the minimization report HTML for a minimization result.

    ``min_result`` should have attributes: ``minimal``, ``removed``,
    ``probes``, ``cache_hits``, ``strategy_name``, and optionally ``weights``.
    Returns an HTML fragment.
    """
    def _gattr(obj: Any, attr: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    strategy = _gattr(min_result, "strategy_name", "")
    minimal = list(_gattr(min_result, "minimal") or [])
    removed = list(_gattr(min_result, "removed") or [])
    probes = _gattr(min_result, "probes", 0)
    cache_hits = _gattr(min_result, "cache_hits", 0)
    weights: Optional[Dict[int, float]] = _gattr(min_result, "weights")

    total = len(minimal) + len(removed)

    parts: List[str] = []
    parts.append("<div class='fr-min-block'>")
    parts.append("<h2>Minimization Report</h2>")

    parts.append(
        "<div class='fr-row'>"
        f"<span class='fr-label'>Strategy</span>"
        f"<span class='fr-val'>{_html.escape(str(strategy))}</span>"
        "</div>"
    )
    parts.append(
        "<div class='fr-row'>"
        f"<span class='fr-label'>Input substitutions</span>"
        f"<span class='fr-val'>{total}</span>"
        "</div>"
    )
    parts.append(
        "<div class='fr-row'>"
        f"<span class='fr-label'>Minimal substitutions</span>"
        f"<span class='fr-val'>{len(minimal)}</span>"
        "</div>"
    )
    parts.append(
        "<div class='fr-row'>"
        f"<span class='fr-label'>Removed substitutions</span>"
        f"<span class='fr-val'>{len(removed)}</span>"
        "</div>"
    )
    parts.append(
        "<div class='fr-row'>"
        f"<span class='fr-label'>Oracle probes</span>"
        f"<span class='fr-val'>{_html.escape(str(probes))}</span>"
        "</div>"
    )
    parts.append(
        "<div class='fr-row'>"
        f"<span class='fr-label'>Cache hits</span>"
        f"<span class='fr-val'>{_html.escape(str(cache_hits))}</span>"
        "</div>"
    )

    # Build a table of the minimal set (truncate at 200 with "and N more")
    all_items = list(minimal) + list(removed)
    if all_items:
        has_weights = weights is not None and len(weights) > 0

        header_cols = ["#", "Kind", "Step ID", "Status"]
        if has_weights:
            header_cols.append("Shapley weight")

        parts.append("<table class='fr-min-table'><thead><tr>")
        for col in header_cols:
            parts.append(f"<th>{_html.escape(col)}</th>")
        parts.append("</tr></thead><tbody>")

        minimal_set = set(id(s) for s in minimal)
        display_items = all_items[:200]
        hidden_count = len(all_items) - 200

        for i, sub in enumerate(display_items):
            is_kept = id(sub) in minimal_set
            status_cls = "keep" if is_kept else "removed"
            status_label = "keep" if is_kept else "removed"

            kind = type(sub).__name__
            step_id = str(getattr(sub, "step_id", "?"))
            kind_esc = _html.escape(kind)
            step_esc = _html.escape(step_id)

            parts.append(
                f"<tr>"
                f"<td>{i + 1}</td>"
                f"<td>{kind_esc}</td>"
                f"<td>{step_esc}</td>"
                f"<td class='{status_cls}'>{status_label}</td>"
            )
            if has_weights:
                w = weights.get(id(sub))  # type: ignore[union-attr]
                w_str = f"{w:.4f}" if w is not None else "—"
                parts.append(f"<td class='weight'>{w_str}</td>")
            parts.append("</tr>")

        if hidden_count > 0:
            span = len(header_cols)
            parts.append(
                f"<tr><td colspan='{span}' class='fr-more'>"
                f"and {hidden_count} more"
                f"</td></tr>"
            )

        parts.append("</tbody></table>")
    else:
        parts.append("<div class='fr-empty'>— no substitutions —</div>")

    parts.append("</div>")
    return "\n".join(parts)


@dataclass
class FullReportSummary:
    """Side-channel summary returned by :func:`write_full_html_report`."""

    output_path: str
    step_count: int
    total_cost_usd: float
    has_replay: bool = False
    has_minimization: bool = False
    has_attestation: bool = False
    bytes_written: int = 0
    by_kind: Dict[str, int] = field(default_factory=dict)


def render_full_html_report(
    steps: Sequence[dict],
    header: Optional[Dict[str, Any]] = None,
    *,
    replay_result: Any = None,
    min_result: Any = None,
    attest_entry: Any = None,
    title: str = "stepback full report",
) -> str:
    """Render a self-contained full HTML report page.

    Combines the step viewer (trace tab), optional time-travel diff panes
    (when *replay_result* is provided), optional minimization report, and
    optional attestation summary into a single tabbed, offline-capable HTML
    document.

    Parameters
    ----------
    steps:
        Recorded step dicts (``Trace.recorded_steps``).
    header:
        Optional trace header dict (``Trace.header``).
    replay_result:
        Optional :class:`~stepback.replay.ReplayResult`.  When given the
        Trace tab gains time-travel navigation (Prev/Next, diff panes,
        causal graph) and dirty/cache-hit stats.
    min_result:
        Optional minimization result object (from
        :func:`~stepback.minimize.minimize`).  When given a
        *Minimization* tab is added.
    attest_entry:
        Optional attestation entry dict or object.  When given an
        *Attestation* tab is added.
    title:
        Page title (HTML-escaped).
    """
    header = header or {}
    title_safe = _html.escape(title, quote=True)

    # ---- build step viewer data ----
    basic_model = _build_view_model(list(steps), header)
    step_count = basic_model["step_count"]
    total_cost = basic_model["total_cost_usd"]
    by_kind = basic_model["by_kind"]

    # ---- build time-travel model if replay result provided ----
    tt_json: Optional[str] = None
    dirty_count = 0
    cache_hit_count = 0
    if replay_result is not None:
        tt_model = _build_time_travel_model(replay_result, list(steps), header)
        raw_json = json.dumps(tt_model, separators=(",", ":"), default=str)
        tt_json = raw_json.replace("</", "<\\/")
        dirty_count = tt_model["dirty_count"]
        cache_hit_count = tt_model["cache_hit_count"]

    # ---- basic data island for pure trace view ----
    basic_json = json.dumps(basic_model, separators=(",", ":"), default=str)
    basic_json = basic_json.replace("</", "<\\/")

    # ---- tabs definition ----
    tabs = [("fr-trace", "Trace")]
    if replay_result is not None:
        pass  # replay info is embedded within the Trace tab
    if min_result is not None:
        tabs.append(("fr-min", "Minimization"))
    if attest_entry is not None:
        tabs.append(("fr-attest", "Attestation"))

    # ---- render ----
    parts: List[str] = []
    parts.append("<!doctype html>")
    parts.append("<html lang='en'><head>")
    parts.append("<meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append(f"<title>{title_safe}</title>")
    parts.append(f"<style>{_FR_CSS}</style>")
    parts.append("</head><body style='height:100%;display:flex;flex-direction:column;'>")

    # header
    parts.append(
        "<header>"
        f"<h1>{title_safe}</h1>"
        f"<div class='meta-bar'>{step_count} steps &middot; "
        f"total cost: ${total_cost:.5f}</div>"
    )
    if replay_result is not None:
        parts.append(
            "<div class='nav-group'>"
            "<button class='nav-btn' id='prev-btn' disabled>&#8592; Prev</button>"
            "<span id='step-pos'>Step — / —</span>"
            "<button class='nav-btn' id='next-btn' disabled>Next &#8594;</button>"
            "</div>"
            f"<span class='badge-pill badge-dirty'>{dirty_count} dirty</span>"
            f"<span class='badge-pill badge-hit'>{cache_hit_count} cache hits</span>"
            "<button class='mode-btn active' id='detail-btn'>&#9776; Detail</button>"
            "<button class='mode-btn' id='graph-btn'>&#9674; Graph</button>"
        )
    parts.append("</header>")

    # tab bar
    parts.append("<div class='fr-tabs'>")
    for tab_id, tab_label in tabs:
        first = tab_id == tabs[0][0]
        active = " active" if first else ""
        parts.append(
            f"<button class='fr-tab{active}' data-tab='{tab_id}'>"
            f"{_html.escape(tab_label)}"
            "</button>"
        )
    parts.append("</div>")

    # --- Trace section ---
    parts.append("<div class='fr-section active' id='fr-trace'>")
    if replay_result is not None:
        # Use time-travel viewer inside the trace tab
        parts.append("<main style='height:100%;'>")
        parts.append("<div id='timeline'></div>")
        parts.append("<div id='right-pane'>")
        parts.append("<div id='detail-pane' class='active'><div class='empty'>select a step</div></div>")
        parts.append("<div id='graph-pane'></div>")
        parts.append("</div>")
        parts.append("</main>")
    else:
        # Basic step viewer
        by_kind_items = by_kind
        kind_chips = " ".join(
            f"<label><input type='checkbox' data-kind='{_html.escape(k)}' "
            f"checked>{_html.escape(k)} ({n})</label>"
            for k, n in sorted(by_kind_items.items())
        )
        parts.append(
            "<div style='padding:8px 14px;border-bottom:1px solid var(--border);'>"
            "<input type='search' id='q' placeholder='search steps...' "
            "style='background:var(--panel2);border:1px solid var(--border);"
            "color:var(--fg);padding:4px 8px;border-radius:4px;min-width:200px;'> "
            + kind_chips
            + "<span class='meta-bar' id='count'></span>"
            "</div>"
        )
        parts.append("<div style='display:flex;height:calc(100% - 50px);'>")
        parts.append("<div id='timeline' style='width:46%;min-width:320px;max-width:680px;overflow-y:auto;border-right:1px solid var(--border);'></div>")
        parts.append("<div id='detail' style='flex:1;overflow-y:auto;padding:16px;'><div class='empty'>select a step</div></div>")
        parts.append("</div>")
    parts.append("</div>")  # end fr-trace

    # --- Minimization section ---
    if min_result is not None:
        parts.append("<div class='fr-section' id='fr-min'>")
        parts.append(_render_minimization_section(min_result))
        parts.append("</div>")

    # --- Attestation section ---
    if attest_entry is not None:
        parts.append("<div class='fr-section' id='fr-attest'>")
        parts.append(_render_attestation_section(attest_entry))
        parts.append("</div>")

    # data islands (no external src= attributes)
    if tt_json is not None:
        parts.append(
            "<script type='application/json' id='tt-data'>"
            + tt_json
            + "</script>"
        )
    else:
        parts.append(
            "<script type='application/json' id='stepback-data'>"
            + basic_json
            + "</script>"
        )
    # always embed the basic data for the trace tab JS
    parts.append(
        "<script type='application/json' id='tt-data-basic'>"
        + basic_json
        + "</script>"
    )

    # inline JS
    if replay_result is not None:
        parts.append(f"<script>{_TT_JS}</script>")
    else:
        parts.append(f"<script>{_INLINE_JS}</script>")

    # tab switching JS
    parts.append(r"""<script>
(function(){
  var tabs = document.querySelectorAll('.fr-tab');
  var sections = document.querySelectorAll('.fr-section');
  tabs.forEach(function(btn){
    btn.addEventListener('click', function(){
      var target = btn.getAttribute('data-tab');
      tabs.forEach(function(b){ b.classList.remove('active'); });
      sections.forEach(function(s){ s.classList.remove('active'); });
      btn.classList.add('active');
      var sec = document.getElementById(target);
      if (sec) sec.classList.add('active');
    });
  });
})();
</script>""")

    parts.append("</body></html>")
    return "\n".join(parts)


def write_full_html_report(
    trace_path: str,
    output_path: str,
    *,
    hmac_key: Optional[bytes] = None,
    attest_entry: Any = None,
    min_result: Any = None,
    title: Optional[str] = None,
) -> FullReportSummary:
    """Read ``trace_path``, replay it, and write a full HTML report to ``output_path``.

    Always performs a ``replay_forward()`` so the Trace tab includes
    time-travel navigation, diff panes, and cache-hit stats.  Pass
    *attest_entry* and/or *min_result* to enable those optional tabs.

    Parameters
    ----------
    trace_path:
        Path to a recorded ``.sb`` trace file.
    output_path:
        Destination HTML file path.  Parent directories are created if
        they do not exist.
    hmac_key:
        Optional HMAC key bytes forwarded to :func:`~stepback.replay.replay`.
    attest_entry:
        Optional attestation entry dict or object to include.
    min_result:
        Optional minimization result object to include.
    title:
        Override the page title; defaults to the trace filename.
    """
    t = replay(trace_path, hmac_key=hmac_key) if hmac_key is not None else replay(trace_path)
    result = t.replay_forward()

    page = render_full_html_report(
        t.recorded_steps,
        t.header,
        replay_result=result,
        attest_entry=attest_entry,
        min_result=min_result,
        title=title or f"stepback full report: {os.path.basename(trace_path)}",
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

    return FullReportSummary(
        output_path=output_path,
        step_count=len(t.recorded_steps),
        total_cost_usd=total_cost,
        has_replay=True,
        has_minimization=min_result is not None,
        has_attestation=attest_entry is not None,
        bytes_written=n,
        by_kind=by_kind,
    )
