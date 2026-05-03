"""Foreign-trace importers.

stepback's value proposition (README §"Why nothing else fills this gap")
is that it lets you replay-with-substitutions an *already-recorded*
production agent run. But the team using LangSmith / OpenInference /
Helicone today has thousands of historical traces in those tools'
formats, and re-instrumenting just to replay them is a non-starter.

This module bridges that gap: it converts foreign trace formats into
native ``.sb`` files that the rest of stepback (replay engine,
substitutions, branches, bisect, attestation) can consume unchanged.

Supported source formats (v0.1):

* :func:`import_openai_chat_log` — JSON array of OpenAI chat
  completion calls. The simplest format: each entry has
  ``{"model": ..., "messages": [...], "response": {...}}``.
* :func:`import_langsmith_jsonl` — LangSmith run export. JSONL where
  each line is a run dict with ``id``, ``parent_run_id``, ``run_type``
  (``llm`` / ``tool`` / ``chain`` / ``retriever``), ``inputs``,
  ``outputs``, ``name``, optional ``extra.invocation_params`` and
  ``extra.token_usage``.
* :func:`import_openinference_spans` — OpenInference / OpenTelemetry
  JSON spans. Each span has ``span_id``, ``parent_span_id``,
  ``attributes`` (where ``openinference.span.kind`` selects the
  step kind and standard attributes carry messages / tool args).

All importers return a :class:`ImportReport` with counts of steps
emitted per kind plus any spans that were skipped (e.g. unrecognised
``run_type``). The output ``.sb`` file is a normal stepback trace:
header, signed step frames, tail. It can be replayed, branched,
substituted, attested, and verified exactly like a recorded one.

Determinism + caching: the importer fabricates a stable
``nondeterminism_hash`` of zero so a cache-only replay of an imported
trace is a pure cache hit (no LLM calls), enabling counterfactual
substitutions on imported production traces with the same dirty-set
propagation guarantees as native traces.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .canonical import canonical_json, hash_obj, sha256_hex
from .pricing import compute_cost
from .recorder import RecorderKey
from .trace_writer import TraceWriter


__all__ = [
    "ImportError",
    "ImportReport",
    "import_openai_chat_log",
    "import_langsmith_jsonl",
    "import_openinference_spans",
    "import_trace",
]


class ImportError(ValueError):
    """Raised when a foreign trace cannot be parsed into stepback steps."""


@dataclass
class ImportReport:
    """Summary of an import operation."""

    output_path: str
    source_format: str
    step_count: int = 0
    kind_counts: Dict[str, int] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)
    total_cost_usd: float = 0.0

    def as_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "source_format": self.source_format,
            "step_count": self.step_count,
            "kind_counts": dict(self.kind_counts),
            "skipped": list(self.skipped),
            "total_cost_usd": self.total_cost_usd,
        }


# --------------------------------------------------------------- helpers


_ZERO_NONDET = sha256_hex(canonical_json({}))


def _emit_step(
    writer: TraceWriter,
    *,
    step_id: str,
    step_kind: str,
    name: Optional[str],
    parent_step_id: Optional[str],
    inputs: dict,
    outputs: dict,
    cost_usd: float = 0.0,
    extras: Optional[dict] = None,
    wallclock_ns: Optional[int] = None,
) -> dict:
    """Write a single step frame; return the canonicalised step dict."""
    step: Dict[str, Any] = {
        "step_id": step_id,
        "step_kind": step_kind,
        "name": name,
        "parent_step_id": parent_step_id,
        "inputs": inputs,
        "outputs": outputs,
        "inputs_hash": hash_obj(inputs),
        "outputs_hash": hash_obj(outputs),
        "nondeterminism_hash": _ZERO_NONDET,
        "wallclock_ns": wallclock_ns if wallclock_ns is not None else time.time_ns(),
        "cost_usd": float(cost_usd),
    }
    if extras:
        for k, v in extras.items():
            step.setdefault(k, v)
    writer.write_step(step)
    return step


def _open_writer(
    output_path: str,
    *,
    key: Optional[RecorderKey],
    compression: bool,
) -> Tuple[TraceWriter, RecorderKey]:
    if key is None:
        key = RecorderKey.fresh()
    w = TraceWriter.open(
        output_path,
        hmac_key=key.hmac_key,
        signing_key=key.signing_key,
        compression=compression,
    )
    return w, key


def _bump_kind(report: ImportReport, kind: str) -> None:
    report.kind_counts[kind] = report.kind_counts.get(kind, 0) + 1
    report.step_count += 1


# ---------------------------------------------------------- OpenAI log


def import_openai_chat_log(
    input_path: str,
    output_path: str,
    *,
    key: Optional[RecorderKey] = None,
    compression: bool = True,
) -> ImportReport:
    """Import a JSON array of OpenAI chat-completion calls into a ``.sb`` file.

    The input file must be a JSON array. Each element is an object
    of the form::

        {
            "model": "gpt-4o-2024-11-20",
            "messages": [{"role": "system", "content": "..."}, ...],
            "response": {"choices": [...], "usage": {...}},
            "temperature": 0.0,           # optional, default 0.0
            "seed": 42,                    # optional, default 42
            "tools": [...],                # optional
            "response_format": {...},      # optional
            "wallclock_ns": 1234567890     # optional
        }

    Each entry becomes one ``llm_call`` step, parented to the
    previous one, so a 12-call log produces a flat 12-step trace
    that is fully replayable + substitutable.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ImportError(
            f"openai_chat_log: top-level JSON must be a list, got {type(data).__name__}"
        )

    report = ImportReport(output_path=output_path, source_format="openai_chat_log")
    writer, _ = _open_writer(output_path, key=key, compression=compression)
    try:
        parent: Optional[str] = None
        parent_outputs_hash: Optional[str] = None
        for i, entry in enumerate(data, start=1):
            if not isinstance(entry, dict):
                report.skipped.append(f"entry[{i}]: not an object")
                continue
            if "model" not in entry or "messages" not in entry:
                report.skipped.append(f"entry[{i}]: missing model/messages")
                continue
            sid = f"step:{i}"
            model = str(entry["model"])
            messages = list(entry["messages"])
            request = {
                "model": model,
                "temperature": float(entry.get("temperature", 0.0)),
                "seed": entry.get("seed", 42),
                "messages": messages,
                "tools": entry.get("tools"),
                "response_format": entry.get("response_format"),
            }
            response = entry.get("response") or {"choices": [], "usage": {}}
            usage = response.get("usage", {}) if isinstance(response, dict) else {}
            cost = compute_cost(model, usage)
            inputs = {"kind": "llm_call", **request}
            if parent is not None:
                inputs["context"] = parent_outputs_hash
            _emit_step(
                writer,
                step_id=sid,
                step_kind="llm_call",
                name=model,
                parent_step_id=parent,
                inputs=inputs,
                outputs=response,
                cost_usd=cost,
                extras={"llm_request": request, "llm_response": response},
                wallclock_ns=entry.get("wallclock_ns"),
            )
            _bump_kind(report, "llm_call")
            report.total_cost_usd += cost
            parent = sid
            parent_outputs_hash = hash_obj(response)
    finally:
        writer.close()
    return report


# ------------------------------------------------------- LangSmith


_LS_KIND_MAP = {
    "llm": "llm_call",
    "chat_model": "llm_call",
    "chatmodel": "llm_call",
    "tool": "tool_call",
    "retriever": "tool_call",
    "chain": "router",
    "agent": "router",
    "prompt": "router",
    "parser": "router",
}


def _topological_order(
    nodes: List[dict],
    id_key: str,
    parent_key: str,
) -> List[dict]:
    """Order ``nodes`` so each parent appears before its children.

    Nodes with parents not present in the input are treated as roots.
    Cycles raise :class:`ImportError`.
    """
    by_id = {str(n[id_key]): n for n in nodes if id_key in n}
    if len(by_id) != len(nodes):
        # silently drop entries without an id
        nodes = [n for n in nodes if id_key in n]

    children: Dict[str, List[str]] = {nid: [] for nid in by_id}
    indeg: Dict[str, int] = {nid: 0 for nid in by_id}
    for n in nodes:
        nid = str(n[id_key])
        pid = n.get(parent_key)
        if pid is None:
            continue
        pid = str(pid)
        if pid in by_id:
            children[pid].append(nid)
            indeg[nid] += 1

    # stable order: roots first, by start_time then id
    def _key(nid: str) -> Tuple:
        n = by_id[nid]
        return (n.get("start_time") or n.get("startTimeUnixNano") or "", nid)

    order: List[str] = []
    ready: List[str] = sorted([nid for nid, d in indeg.items() if d == 0], key=_key)
    visited = 0
    while ready:
        # pop deterministically
        ready.sort(key=_key)
        nid = ready.pop(0)
        order.append(nid)
        visited += 1
        for c in sorted(children.get(nid, []), key=_key):
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)

    if visited != len(by_id):
        raise ImportError(
            f"cycle in foreign trace: visited {visited} of {len(by_id)} nodes"
        )
    return [by_id[nid] for nid in order]


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ImportError(f"{path}:{lineno}: invalid JSON: {e}") from e


def import_langsmith_jsonl(
    input_path: str,
    output_path: str,
    *,
    key: Optional[RecorderKey] = None,
    compression: bool = True,
) -> ImportReport:
    """Import a LangSmith JSONL run export into a ``.sb`` file.

    Each line is a run dict. Recognised fields:

    * ``id`` — run id (string / UUID)
    * ``parent_run_id`` — parent run id, or null for a root
    * ``run_type`` — ``"llm"`` / ``"chat_model"`` / ``"tool"`` /
      ``"retriever"`` / ``"chain"`` / ``"agent"`` / ``"prompt"`` /
      ``"parser"``
    * ``name`` — human-readable name (model id for llm, tool name
      for tool)
    * ``inputs`` — call inputs (e.g. ``{"messages": [...]}`` for llm,
      ``{"input": "..."}`` for tool)
    * ``outputs`` — call outputs (e.g. ``{"generations": [...]}`` or
      ``{"output": "..."}``)
    * ``extra.invocation_params`` — for llm: ``model``, ``temperature``,
      etc.
    * ``extra.token_usage`` — for llm: ``{"prompt_tokens": ...,
      "completion_tokens": ...}``
    * ``start_time`` — ISO8601 or epoch-ns; used for ordering siblings

    Run type → step kind mapping:

    =================  ====================
    LangSmith run_type  stepback step_kind
    =================  ====================
    llm, chat_model    llm_call
    tool, retriever    tool_call
    chain, agent,      router
      prompt, parser
    =================  ====================
    """
    runs = list(_iter_jsonl(input_path))
    if not runs:
        raise ImportError(f"{input_path}: empty LangSmith export")

    ordered = _topological_order(runs, id_key="id", parent_key="parent_run_id")

    # Map foreign uuid → stepback step_id (assigned in topo order).
    id_map: Dict[str, str] = {}
    out_hash_by_sid: Dict[str, str] = {}

    report = ImportReport(output_path=output_path, source_format="langsmith_jsonl")
    writer, _ = _open_writer(output_path, key=key, compression=compression)
    try:
        for i, run in enumerate(ordered, start=1):
            run_id = str(run["id"])
            sid = f"step:{i}"
            id_map[run_id] = sid

            run_type = str(run.get("run_type") or "").lower()
            kind = _LS_KIND_MAP.get(run_type)
            if kind is None:
                report.skipped.append(f"{run_id}: unknown run_type {run_type!r}")
                # still emit as a router-style passthrough so the call
                # tree stays connected
                kind = "router"

            parent_run = run.get("parent_run_id")
            parent_sid = id_map.get(str(parent_run)) if parent_run else None

            name = run.get("name")
            ls_inputs = run.get("inputs") or {}
            ls_outputs = run.get("outputs") or {}

            wallclock = _parse_wallclock(run.get("start_time"))

            if kind == "llm_call":
                extra = run.get("extra") or {}
                inv = extra.get("invocation_params") or {}
                model = inv.get("model") or inv.get("model_name") or name or "unknown"
                messages = (
                    ls_inputs.get("messages")
                    or ls_inputs.get("prompts")
                    or ls_inputs.get("input")
                    or []
                )
                if isinstance(messages, str):
                    messages = [{"role": "user", "content": messages}]
                request = {
                    "model": str(model),
                    "temperature": float(inv.get("temperature", 0.0)),
                    "seed": inv.get("seed", 42),
                    "messages": list(messages),
                    "tools": inv.get("tools"),
                    "response_format": inv.get("response_format"),
                }
                usage = extra.get("token_usage") or {}
                cost = compute_cost(str(model), usage)
                response = dict(ls_outputs) if isinstance(ls_outputs, dict) else {"output": ls_outputs}
                if usage and "usage" not in response:
                    response = {**response, "usage": usage}
                inputs = {"kind": "llm_call", **request}
                if parent_sid is not None:
                    inputs["context"] = out_hash_by_sid[parent_sid]
                step = _emit_step(
                    writer,
                    step_id=sid,
                    step_kind="llm_call",
                    name=str(model),
                    parent_step_id=parent_sid,
                    inputs=inputs,
                    outputs=response,
                    cost_usd=cost,
                    extras={"llm_request": request, "llm_response": response},
                    wallclock_ns=wallclock,
                )
                _bump_kind(report, "llm_call")
                report.total_cost_usd += cost
            elif kind == "tool_call":
                tool_name = str(name or run_type)
                args = ls_inputs if isinstance(ls_inputs, dict) else {"input": ls_inputs}
                result = ls_outputs
                inputs = {"kind": "tool_call", "name": tool_name, "arguments": args}
                if parent_sid is not None:
                    inputs["context"] = out_hash_by_sid[parent_sid]
                outputs = {"result": result}
                step = _emit_step(
                    writer,
                    step_id=sid,
                    step_kind="tool_call",
                    name=tool_name,
                    parent_step_id=parent_sid,
                    inputs=inputs,
                    outputs=outputs,
                    wallclock_ns=wallclock,
                )
                _bump_kind(report, "tool_call")
            else:  # router/passthrough
                rname = str(name or run_type or "chain")
                inputs = {"kind": "router", "name": rname, "options": []}
                if parent_sid is not None:
                    inputs["context"] = out_hash_by_sid[parent_sid]
                outputs = {"choice": rname, "outputs": ls_outputs}
                step = _emit_step(
                    writer,
                    step_id=sid,
                    step_kind="router",
                    name=rname,
                    parent_step_id=parent_sid,
                    inputs=inputs,
                    outputs=outputs,
                    wallclock_ns=wallclock,
                )
                _bump_kind(report, "router")

            out_hash_by_sid[sid] = step["outputs_hash"]
    finally:
        writer.close()
    return report


# ----------------------------------------------------- OpenInference


# OpenInference span-kind attribute → stepback kind. The OpenInference
# spec puts the kind at the attribute key ``openinference.span.kind``;
# values are upper-case enums (LLM, TOOL, RETRIEVER, CHAIN, AGENT,
# EMBEDDING, RERANKER, GUARDRAIL, EVALUATOR, UNKNOWN).
_OI_KIND_MAP = {
    "LLM": "llm_call",
    "TOOL": "tool_call",
    "RETRIEVER": "tool_call",
    "EMBEDDING": "tool_call",
    "RERANKER": "tool_call",
    "CHAIN": "router",
    "AGENT": "router",
    "GUARDRAIL": "router",
    "EVALUATOR": "router",
}


def _flatten_attrs(attrs: Any) -> Dict[str, Any]:
    """Normalise OTel attributes (list of {key, value} or plain dict)."""
    if isinstance(attrs, dict):
        return dict(attrs)
    out: Dict[str, Any] = {}
    if isinstance(attrs, list):
        for kv in attrs:
            if not isinstance(kv, dict):
                continue
            k = kv.get("key")
            v = kv.get("value")
            if isinstance(v, dict):
                # OTel any-value envelope: {"stringValue": "..."}, etc.
                for vk in ("stringValue", "intValue", "doubleValue", "boolValue"):
                    if vk in v:
                        v = v[vk]
                        break
                else:
                    if "arrayValue" in v:
                        v = v["arrayValue"].get("values", [])
            if k:
                out[str(k)] = v
    return out


def _oi_collect_messages(attrs: Dict[str, Any], prefix: str) -> List[dict]:
    """Reconstruct ``llm.input_messages`` / ``llm.output_messages`` arrays.

    OpenInference encodes messages as flat keys
    ``llm.input_messages.0.message.role`` etc. when emitted via OTel.
    """
    msgs: Dict[int, dict] = {}
    pdot = prefix + "."
    for k, v in attrs.items():
        if not k.startswith(pdot):
            continue
        rest = k[len(pdot):]
        parts = rest.split(".")
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        msg = msgs.setdefault(idx, {})
        # collapse "message.role" → "role", "message.content" → "content"
        suffix = ".".join(parts[1:])
        if suffix.startswith("message."):
            suffix = suffix[len("message."):]
        if suffix in ("role", "content", "name"):
            msg[suffix] = v
    return [msgs[i] for i in sorted(msgs.keys())]


def import_openinference_spans(
    input_path: str,
    output_path: str,
    *,
    key: Optional[RecorderKey] = None,
    compression: bool = True,
) -> ImportReport:
    """Import OpenInference / OTel JSON spans into a ``.sb`` file.

    Accepts either:

    * a JSON array of span dicts, or
    * an OTLP-ish ``{"spans": [...]}`` envelope.

    Each span must have ``span_id`` (or ``spanId``) and may have
    ``parent_span_id`` (or ``parentSpanId``). Span kind comes from
    ``attributes["openinference.span.kind"]``. Standard OpenInference
    attributes are decoded:

    * ``llm.model_name`` → step name + request.model
    * ``llm.input_messages.<i>.message.{role,content}`` → request.messages
    * ``llm.output_messages.<i>.message.content`` → response message
    * ``llm.token_count.{prompt,completion,total}`` → usage
    * ``tool.name`` / ``tool.parameters`` / ``output.value`` → tool I/O
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        spans = data.get("spans") or data.get("resourceSpans") or []
        # collapse OTLP resourceSpans → flat spans list
        if spans and isinstance(spans[0], dict) and "scopeSpans" in spans[0]:
            flat: List[dict] = []
            for rs in spans:
                for ss in rs.get("scopeSpans", []):
                    flat.extend(ss.get("spans", []))
            spans = flat
    else:
        spans = data
    if not isinstance(spans, list):
        raise ImportError(
            f"{input_path}: expected a list of spans, got {type(spans).__name__}"
        )
    if not spans:
        raise ImportError(f"{input_path}: no spans in input")

    # Normalise key names: spanId↔span_id, parentSpanId↔parent_span_id.
    normalised: List[dict] = []
    for s in spans:
        if not isinstance(s, dict):
            continue
        sid = s.get("span_id") or s.get("spanId")
        if not sid:
            continue
        pid = s.get("parent_span_id") or s.get("parentSpanId")
        normalised.append({
            **s,
            "span_id": str(sid),
            "parent_span_id": str(pid) if pid else None,
            "_attrs": _flatten_attrs(s.get("attributes")),
        })

    if not normalised:
        raise ImportError(f"{input_path}: no spans had a span_id")

    ordered = _topological_order(normalised, id_key="span_id", parent_key="parent_span_id")

    id_map: Dict[str, str] = {}
    out_hash_by_sid: Dict[str, str] = {}
    report = ImportReport(output_path=output_path, source_format="openinference_spans")
    writer, _ = _open_writer(output_path, key=key, compression=compression)
    try:
        for i, span in enumerate(ordered, start=1):
            sid = f"step:{i}"
            id_map[span["span_id"]] = sid
            attrs: Dict[str, Any] = span["_attrs"]
            kind_attr = (
                attrs.get("openinference.span.kind")
                or attrs.get("openinference.kind")
                or "UNKNOWN"
            )
            kind = _OI_KIND_MAP.get(str(kind_attr).upper())
            if kind is None:
                report.skipped.append(f"{span['span_id']}: kind={kind_attr!r}")
                kind = "router"

            parent_sid = id_map.get(span["parent_span_id"]) if span["parent_span_id"] else None
            name = span.get("name")
            wallclock = _parse_wallclock(
                span.get("start_time")
                or span.get("startTimeUnixNano")
                or span.get("start_time_unix_nano")
            )

            if kind == "llm_call":
                model = (
                    attrs.get("llm.model_name")
                    or attrs.get("llm.model")
                    or name
                    or "unknown"
                )
                messages = _oi_collect_messages(attrs, "llm.input_messages")
                if not messages:
                    raw = attrs.get("input.value")
                    if isinstance(raw, str):
                        messages = [{"role": "user", "content": raw}]
                    elif isinstance(raw, list):
                        messages = list(raw)
                request = {
                    "model": str(model),
                    "temperature": float(attrs.get("llm.invocation_parameters.temperature", 0.0)),
                    "seed": attrs.get("llm.invocation_parameters.seed", 42),
                    "messages": messages,
                    "tools": attrs.get("llm.tools"),
                    "response_format": None,
                }
                out_messages = _oi_collect_messages(attrs, "llm.output_messages")
                pt = int(attrs.get("llm.token_count.prompt", 0) or 0)
                ct = int(attrs.get("llm.token_count.completion", 0) or 0)
                tt = int(attrs.get("llm.token_count.total", pt + ct) or 0)
                usage = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}
                cost = compute_cost(str(model), usage)
                response = {
                    "choices": [
                        {
                            "index": idx,
                            "finish_reason": "stop",
                            "message": m,
                        }
                        for idx, m in enumerate(out_messages)
                    ],
                    "usage": usage,
                    "model": str(model),
                }
                inputs = {"kind": "llm_call", **request}
                if parent_sid is not None:
                    inputs["context"] = out_hash_by_sid[parent_sid]
                step = _emit_step(
                    writer,
                    step_id=sid,
                    step_kind="llm_call",
                    name=str(model),
                    parent_step_id=parent_sid,
                    inputs=inputs,
                    outputs=response,
                    cost_usd=cost,
                    extras={"llm_request": request, "llm_response": response},
                    wallclock_ns=wallclock,
                )
                _bump_kind(report, "llm_call")
                report.total_cost_usd += cost
            elif kind == "tool_call":
                tool_name = str(attrs.get("tool.name") or name or "tool")
                args_raw = attrs.get("tool.parameters") or attrs.get("input.value") or {}
                if isinstance(args_raw, str):
                    try:
                        args = json.loads(args_raw)
                        if not isinstance(args, dict):
                            args = {"input": args}
                    except json.JSONDecodeError:
                        args = {"input": args_raw}
                else:
                    args = args_raw if isinstance(args_raw, dict) else {"input": args_raw}
                out_raw = attrs.get("output.value")
                if isinstance(out_raw, str):
                    try:
                        result = json.loads(out_raw)
                    except json.JSONDecodeError:
                        result = out_raw
                else:
                    result = out_raw if out_raw is not None else {}
                inputs = {"kind": "tool_call", "name": tool_name, "arguments": args}
                if parent_sid is not None:
                    inputs["context"] = out_hash_by_sid[parent_sid]
                outputs = {"result": result}
                step = _emit_step(
                    writer,
                    step_id=sid,
                    step_kind="tool_call",
                    name=tool_name,
                    parent_step_id=parent_sid,
                    inputs=inputs,
                    outputs=outputs,
                    wallclock_ns=wallclock,
                )
                _bump_kind(report, "tool_call")
            else:
                rname = str(name or kind_attr or "chain")
                inputs = {"kind": "router", "name": rname, "options": []}
                if parent_sid is not None:
                    inputs["context"] = out_hash_by_sid[parent_sid]
                out_raw = attrs.get("output.value")
                outputs = {"choice": rname, "outputs": out_raw}
                step = _emit_step(
                    writer,
                    step_id=sid,
                    step_kind="router",
                    name=rname,
                    parent_step_id=parent_sid,
                    inputs=inputs,
                    outputs=outputs,
                    wallclock_ns=wallclock,
                )
                _bump_kind(report, "router")

            out_hash_by_sid[sid] = step["outputs_hash"]
    finally:
        writer.close()
    return report


def _parse_wallclock(value: Any) -> Optional[int]:
    """Best-effort parse of a foreign timestamp into wall-clock nanoseconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = int(value)
        # Heuristic: if it looks like seconds since epoch, scale to ns.
        if v < 10**12:
            return v * 1_000_000_000
        if v < 10**15:
            return v * 1_000_000  # ms
        if v < 10**18:
            return v * 1_000  # us
        return v
    if isinstance(value, str):
        # Best-effort ISO8601 → ns. Use fromisoformat (3.11+ accepts Z).
        try:
            from datetime import datetime
            s = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            return int(dt.timestamp() * 1_000_000_000)
        except Exception:
            return None
    return None


# --------------------------------------------------------- dispatcher


_FORMAT_DISPATCH = {
    "openai_chat_log": import_openai_chat_log,
    "openai": import_openai_chat_log,
    "langsmith": import_langsmith_jsonl,
    "langsmith_jsonl": import_langsmith_jsonl,
    "openinference": import_openinference_spans,
    "openinference_spans": import_openinference_spans,
    "otel": import_openinference_spans,
}


def import_trace(
    source_format: str,
    input_path: str,
    output_path: str,
    *,
    key: Optional[RecorderKey] = None,
    compression: bool = True,
) -> ImportReport:
    """Dispatch to the importer for ``source_format``.

    ``source_format`` ∈ ``{"openai_chat_log", "langsmith", "openinference"}``
    (with aliases). Raises :class:`ImportError` for unknown formats.
    """
    fn = _FORMAT_DISPATCH.get(source_format.lower())
    if fn is None:
        raise ImportError(
            f"unknown source_format {source_format!r}; "
            f"known: {sorted(set(_FORMAT_DISPATCH))}"
        )
    return fn(input_path, output_path, key=key, compression=compression)
