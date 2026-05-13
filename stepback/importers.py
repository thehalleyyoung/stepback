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
    "LossReport",
    "TraceImportError",
    "import_openai_chat_log",
    "import_langsmith_jsonl",
    "import_openinference_spans",
    "import_otel_spans",
    "import_phoenix_spans",
    "import_helicone_log",
    "import_langfuse_export",
    "import_datadog_apm",
    "import_native_json",
    "import_trace",
    "_detect_langsmith_schema_version",
    "_detect_oi_attr_format",
    "_read_langsmith_input",
]

NATIVE_JSON_FORMAT_TAG = "stepback_native_json_v1"


class ImportError(ValueError):
    """Raised when a foreign trace cannot be parsed into stepback steps."""


TraceImportError = ImportError


@dataclass
class LossReport:
    """Documents information fidelity lost when converting between formats.

    Four categories of loss:

    * ``absent`` — fields that cannot be represented in the target format
      (round-trip is impossible even with approximation).
    * ``approximated`` — fields that were estimated from available data
      (e.g. cost_usd computed from token counts; seed defaulted to 42).
    * ``synthesized`` — fields that were fabricated with no source
      information (e.g. step_id from a sequential counter).
    * ``dropped`` — fields present in the source that were silently
      discarded (e.g. LangSmith tags, feedback_stats).

    Each entry in a list is a human-readable string in the form
    ``"field_name: reason"`` or ``"category: description"``.
    """

    absent: List[str] = field(default_factory=list)
    approximated: List[str] = field(default_factory=list)
    synthesized: List[str] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)

    def is_lossless(self) -> bool:
        """Return True iff all four loss lists are empty."""
        return not (self.absent or self.approximated or self.synthesized or self.dropped)

    def as_dict(self) -> dict:
        return {
            "absent": list(self.absent),
            "approximated": list(self.approximated),
            "synthesized": list(self.synthesized),
            "dropped": list(self.dropped),
        }


@dataclass
class ImportReport:
    """Summary of an import operation."""

    output_path: str
    source_format: str
    step_count: int = 0
    kind_counts: Dict[str, int] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)
    total_cost_usd: float = 0.0
    lossiness: LossReport = field(default_factory=LossReport)
    schema_version: Optional[str] = None

    def as_dict(self) -> dict:
        d: Dict[str, Any] = {
            "output_path": self.output_path,
            "source_format": self.source_format,
            "step_count": self.step_count,
            "kind_counts": dict(self.kind_counts),
            "skipped": list(self.skipped),
            "total_cost_usd": self.total_cost_usd,
            "lossiness": self.lossiness.as_dict(),
            "schema_version": self.schema_version,
        }
        return d


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
    report.lossiness.synthesized.extend([
        "step_id: fabricated from sequential index (no run-id in format)",
        "nondeterminism_hash: set to zero-hash (not recorded by OpenAI)",
        "parent_step_id: linearised chain (no call-tree in chat log format)",
    ])
    report.lossiness.approximated.extend([
        "cost_usd: estimated from token counts via pricing table",
        "seed: defaulted to 42 when not present in entry",
    ])
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


# V2 LangSmith runs include fields like feedback_stats and app_path.
_LS_V2_FIELDS = frozenset([
    "feedback_stats", "app_path", "manifest_id", "total_tokens",
    "execution_order", "child_run_ids", "manifest", "session_id",
    "reference_example_id",
])
_LS_V2_DROPPED = ["feedback_stats", "app_path", "manifest_id"]


def _detect_langsmith_schema_version(runs: List[dict]) -> str:
    """Return ``"v2"`` if any run contains a known v2-era field; else ``"v1"``."""
    for run in runs:
        if isinstance(run, dict) and _LS_V2_FIELDS.intersection(run.keys()):
            return "v2"
    return "v1"


def _detect_oi_attr_format(spans: List[dict]) -> str:
    """Return ``"otlp_any_value"`` if attributes use OTel AnyValue envelopes, else ``"dict"``."""
    for span in spans:
        attrs = span.get("attributes")
        if isinstance(attrs, list) and attrs:
            first = attrs[0]
            if isinstance(first, dict) and "key" in first and "value" in first:
                return "otlp_any_value"
    return "dict"


def _read_langsmith_input(path: str) -> List[dict]:
    """Read a LangSmith export: JSONL or JSON array."""
    # Try JSON array first, then JSONL
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read(1)
    if raw == "[":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ImportError(f"{path}: expected a JSON array")
        return data
    return list(_iter_jsonl(path))


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
    runs = _read_langsmith_input(input_path)
    if not runs:
        raise ImportError(f"{input_path}: empty LangSmith export")

    ordered = _topological_order(runs, id_key="id", parent_key="parent_run_id")

    # Map foreign uuid → stepback step_id (assigned in topo order).
    id_map: Dict[str, str] = {}
    out_hash_by_sid: Dict[str, str] = {}

    schema_version = _detect_langsmith_schema_version(runs)
    report = ImportReport(output_path=output_path, source_format="langsmith_jsonl",
                          schema_version=schema_version)
    report.lossiness.synthesized.extend([
        "step_id: fabricated from sequential index (LangSmith UUIDs are not preserved)",
        "nondeterminism_hash: set to zero-hash (not recorded by LangSmith)",
    ])
    report.lossiness.approximated.extend([
        "step_kind: inferred from run_type field (may differ from actual call pattern)",
        "cost_usd: estimated from token counts via pricing table",
    ])
    report.lossiness.dropped.extend([
        "tags: LangSmith run tags are not mapped to any stepback field",
        "metadata: run-level metadata key/value pairs are discarded",
        "feedback: user feedback attached to runs is not representable",
    ])
    if schema_version == "v2":
        report.lossiness.dropped.extend([
            "feedback_stats: aggregate feedback statistics (v2 field) are discarded",
            "app_path: application path URL (v2 field) is discarded",
        ])
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
    _oi_schema_version = "dict"
    if isinstance(data, dict):
        if "resourceSpans" in data:
            _oi_schema_version = "otlp_resource_spans"
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
        _oi_schema_version = _detect_oi_attr_format(spans) if isinstance(spans, list) else "dict"
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
    report = ImportReport(output_path=output_path, source_format="openinference_spans",
                          schema_version=_oi_schema_version)
    report.lossiness.synthesized.extend([
        "nondeterminism_hash: set to zero-hash (not recorded by OpenInference)",
        "step_id: fabricated from sequential index",
    ])
    report.lossiness.approximated.extend([
        "step_kind: approximated from openinference.span.kind attribute",
        "cost_usd: estimated from token counts via pricing table",
    ])
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


# --------------------------------------------------------- OTel spans


# Stable OTel semantic conventions for agent steps (RFC 0006 / OTEP-0228).
# gen_ai.* conventions (OTEP-0217) overlap with these for LLM calls.
_OTEL_STEP_KIND_MAP = {
    "LLM_CALL": "llm_call",
    "TOOL_CALL": "tool_call",
    "RETRIEVAL": "tool_call",
    "EMBEDDING": "tool_call",
    "ROUTER": "router",
    "AGENT": "router",
    "CHAIN": "router",
}


def import_otel_spans(
    input_path: str,
    output_path: str,
    *,
    key: Optional[RecorderKey] = None,
    compression: bool = True,
) -> ImportReport:
    """Import OpenTelemetry spans using stable semantic conventions.

    Accepts either:

    * a JSON array of span dicts, or
    * an OTLP-ish ``{"spans": [...]}`` or ``{"resourceSpans": [...]}`` envelope.

    Step kind is determined by the ``agent.step.kind`` attribute (preferred)
    or ``gen_ai.system`` + span name heuristics. Uses ``gen_ai.request.model``,
    ``gen_ai.usage.input_tokens`` / ``gen_ai.usage.output_tokens`` for LLM
    calls.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        spans = data.get("spans") or data.get("resourceSpans") or []
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
    report = ImportReport(output_path=output_path, source_format="otel_spans")
    report.lossiness.synthesized.extend([
        "nondeterminism_hash: set to zero-hash (not recorded by OTel)",
        "step_id: fabricated from sequential index",
    ])
    report.lossiness.dropped.extend([
        "resource.*: OTel resource attributes (service.name, etc.) are discarded",
        "instrumentation_scope: library name/version is discarded",
        "trace_id: OTel trace_id is not mapped to any stepback field",
        "span context flags: trace-flags and trace-state are discarded",
    ])
    writer, _ = _open_writer(output_path, key=key, compression=compression)
    try:
        for i, span in enumerate(ordered, start=1):
            sid = f"step:{i}"
            id_map[span["span_id"]] = sid
            attrs: Dict[str, Any] = span["_attrs"]
            # Prefer agent.step.kind; fall back to gen_ai.system heuristic.
            step_kind_attr = str(
                attrs.get("agent.step.kind")
                or attrs.get("openinference.span.kind")
                or "UNKNOWN"
            ).upper()
            kind = _OTEL_STEP_KIND_MAP.get(step_kind_attr)
            if kind is None:
                # Detect from gen_ai.* attributes
                if attrs.get("gen_ai.request.model") or attrs.get("gen_ai.system"):
                    kind = "llm_call"
                else:
                    kind = "router"

            parent_sid = id_map.get(span["parent_span_id"]) if span["parent_span_id"] else None
            name = span.get("name")
            wallclock = _parse_wallclock(
                span.get("start_time")
                or attrs.get("start_time_unix_nano")
                or span.get("startTimeUnixNano")
            )

            if kind == "llm_call":
                model = str(
                    attrs.get("gen_ai.request.model")
                    or attrs.get("llm.model_name")
                    or name or "unknown"
                )
                messages = _oi_collect_messages(attrs, "llm.input_messages")
                if not messages:
                    raw = attrs.get("input.value") or attrs.get("gen_ai.prompt")
                    if isinstance(raw, str):
                        messages = [{"role": "user", "content": raw}]
                request = {
                    "model": model,
                    "temperature": float(attrs.get("gen_ai.request.temperature", 0.0)),
                    "seed": attrs.get("gen_ai.request.seed", 42),
                    "messages": messages,
                    "tools": None,
                    "response_format": None,
                }
                in_tok = int(attrs.get("gen_ai.usage.input_tokens", 0) or
                             attrs.get("llm.token_count.prompt", 0) or 0)
                out_tok = int(attrs.get("gen_ai.usage.output_tokens", 0) or
                              attrs.get("llm.token_count.completion", 0) or 0)
                usage = {"prompt_tokens": in_tok, "completion_tokens": out_tok,
                         "total_tokens": in_tok + out_tok}
                cost = compute_cost(model, usage)
                out_messages = _oi_collect_messages(attrs, "llm.output_messages")
                response = {
                    "choices": [{"index": i, "finish_reason": "stop", "message": m}
                                for i, m in enumerate(out_messages)],
                    "usage": usage,
                    "model": model,
                }
                inputs = {"kind": "llm_call", **request}
                if parent_sid:
                    inputs["context"] = out_hash_by_sid[parent_sid]
                step = _emit_step(
                    writer, step_id=sid, step_kind="llm_call", name=model,
                    parent_step_id=parent_sid, inputs=inputs, outputs=response,
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
                            args = {"input": args_raw}
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
                if parent_sid:
                    inputs["context"] = out_hash_by_sid[parent_sid]
                outputs = {"result": result}
                step = _emit_step(
                    writer, step_id=sid, step_kind="tool_call", name=tool_name,
                    parent_step_id=parent_sid, inputs=inputs, outputs=outputs,
                    wallclock_ns=wallclock,
                )
                _bump_kind(report, "tool_call")
            else:
                rname = str(name or step_kind_attr or "span")
                inputs = {"kind": "router", "name": rname, "options": []}
                if parent_sid:
                    inputs["context"] = out_hash_by_sid[parent_sid]
                step = _emit_step(
                    writer, step_id=sid, step_kind="router", name=rname,
                    parent_step_id=parent_sid,
                    inputs=inputs, outputs={"choice": rname, "outputs": None},
                    wallclock_ns=wallclock,
                )
                _bump_kind(report, "router")
            out_hash_by_sid[sid] = step["outputs_hash"]
    finally:
        writer.close()
    return report


# --------------------------------------------------------- Phoenix spans


def import_phoenix_spans(
    input_path: str,
    output_path: str,
    *,
    key: Optional[RecorderKey] = None,
    compression: bool = True,
) -> ImportReport:
    """Import Arize Phoenix spans (OpenInference-compatible) into a ``.sb`` file.

    Phoenix exports OpenInference-formatted spans but also includes
    Phoenix-specific fields (``trace_id``, ``cumulative_token_count.*``,
    ``phoenix.span_id``, ``context.span_id``, ``context.trace_id``) that are
    dropped during import. The core span data is mapped identically to
    :func:`import_openinference_spans`.
    """
    # Phoenix spans are structurally identical to OpenInference spans.
    # We delegate to the OI importer and then overlay the lossiness.
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        spans = data.get("spans") or data.get("data") or []
    else:
        spans = data
    if not isinstance(spans, list):
        raise ImportError(
            f"{input_path}: expected a list of Phoenix spans, got {type(spans).__name__}"
        )
    if not spans:
        raise ImportError(f"{input_path}: no spans in input")

    normalised: List[dict] = []
    for s in spans:
        if not isinstance(s, dict):
            continue
        sid = s.get("span_id") or s.get("context", {}).get("span_id")
        if not sid:
            continue
        pid = (s.get("parent_span_id")
               or s.get("parent_id")
               or s.get("context", {}).get("parent_span_id"))
        normalised.append({
            **s,
            "span_id": str(sid),
            "parent_span_id": str(pid) if pid else None,
            "_attrs": _flatten_attrs(s.get("attributes")),
        })
    if not normalised:
        raise ImportError(f"{input_path}: no Phoenix spans had a span_id")

    ordered = _topological_order(normalised, id_key="span_id", parent_key="parent_span_id")

    id_map: Dict[str, str] = {}
    out_hash_by_sid: Dict[str, str] = {}
    report = ImportReport(output_path=output_path, source_format="phoenix_spans")
    report.lossiness.synthesized.extend([
        "nondeterminism_hash: set to zero-hash",
        "step_id: fabricated from sequential index",
    ])
    report.lossiness.dropped.extend([
        "trace_id: Phoenix trace_id is not mapped to any stepback field",
        "cumulative_token_count.*: Phoenix cumulative token counts are discarded",
        "context.span_id / context.trace_id: Phoenix context envelope fields are discarded",
        "phoenix.* attributes: Phoenix-specific metadata attributes are discarded",
    ])

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
                kind = "router"

            parent_sid = id_map.get(span["parent_span_id"]) if span["parent_span_id"] else None
            name = span.get("name")
            wallclock = _parse_wallclock(
                span.get("start_time") or span.get("startTimeUnixNano")
            )

            if kind == "llm_call":
                model = str(
                    attrs.get("llm.model_name") or attrs.get("llm.model") or name or "unknown"
                )
                messages = _oi_collect_messages(attrs, "llm.input_messages")
                if not messages:
                    raw = attrs.get("input.value")
                    if isinstance(raw, str):
                        messages = [{"role": "user", "content": raw}]
                request = {
                    "model": model,
                    "temperature": float(attrs.get("llm.invocation_parameters.temperature", 0.0)),
                    "seed": attrs.get("llm.invocation_parameters.seed", 42),
                    "messages": messages,
                    "tools": attrs.get("llm.tools"),
                    "response_format": None,
                }
                out_messages = _oi_collect_messages(attrs, "llm.output_messages")
                pt = int(attrs.get("llm.token_count.prompt", 0) or 0)
                ct = int(attrs.get("llm.token_count.completion", 0) or 0)
                usage = {"prompt_tokens": pt, "completion_tokens": ct,
                         "total_tokens": pt + ct}
                cost = compute_cost(model, usage)
                response = {
                    "choices": [{"index": j, "finish_reason": "stop", "message": m}
                                for j, m in enumerate(out_messages)],
                    "usage": usage, "model": model,
                }
                inputs = {"kind": "llm_call", **request}
                if parent_sid:
                    inputs["context"] = out_hash_by_sid[parent_sid]
                step = _emit_step(
                    writer, step_id=sid, step_kind="llm_call", name=model,
                    parent_step_id=parent_sid, inputs=inputs, outputs=response,
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
                            args = {"input": args_raw}
                    except json.JSONDecodeError:
                        args = {"input": args_raw}
                else:
                    args = args_raw if isinstance(args_raw, dict) else {"input": args_raw}
                out_raw = attrs.get("output.value")
                result = json.loads(out_raw) if isinstance(out_raw, str) else (out_raw or {})
                inputs = {"kind": "tool_call", "name": tool_name, "arguments": args}
                if parent_sid:
                    inputs["context"] = out_hash_by_sid[parent_sid]
                step = _emit_step(
                    writer, step_id=sid, step_kind="tool_call", name=tool_name,
                    parent_step_id=parent_sid, inputs=inputs, outputs={"result": result},
                    wallclock_ns=wallclock,
                )
                _bump_kind(report, "tool_call")
            else:
                rname = str(name or kind_attr or "chain")
                inputs = {"kind": "router", "name": rname, "options": []}
                if parent_sid:
                    inputs["context"] = out_hash_by_sid[parent_sid]
                step = _emit_step(
                    writer, step_id=sid, step_kind="router", name=rname,
                    parent_step_id=parent_sid,
                    inputs=inputs, outputs={"choice": rname, "outputs": None},
                    wallclock_ns=wallclock,
                )
                _bump_kind(report, "router")
            out_hash_by_sid[sid] = step["outputs_hash"]
    finally:
        writer.close()
    return report


# --------------------------------------------------------- Helicone log


def import_helicone_log(
    input_path: str,
    output_path: str,
    *,
    key: Optional[RecorderKey] = None,
    compression: bool = True,
) -> ImportReport:
    """Import a Helicone request log (JSON array) into a ``.sb`` file.

    Helicone stores each LLM call as a flat JSON object::

        {
            "request_id": "...",
            "request": {"request_body": {"model": "...", "messages": [...]}},
            "response": {"body": {"choices": [...], "usage": {...}}},
            "properties": {...},   # custom key/value tags  -- dropped
            "feedback": {...},     # user feedback          -- dropped
        }

    Each entry becomes one ``llm_call`` step.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data") or [data]
    if not isinstance(data, list):
        raise ImportError(
            f"{input_path}: expected a JSON array, got {type(data).__name__}"
        )
    if not data:
        raise ImportError(f"{input_path}: empty Helicone log")

    report = ImportReport(output_path=output_path, source_format="helicone_log")
    report.lossiness.synthesized.extend([
        "step_id: fabricated from sequential index",
        "nondeterminism_hash: set to zero-hash",
        "parent_step_id: linearised chain (Helicone logs are flat per-request)",
    ])
    report.lossiness.approximated.extend([
        "cost_usd: estimated from token counts",
    ])
    report.lossiness.dropped.extend([
        "request_id: Helicone request UUID is not mapped to any stepback field",
        "properties: custom Helicone tag key/value pairs are discarded",
        "feedback: user feedback scores attached to requests are discarded",
        "user: Helicone user identifier is discarded",
        "provider: provider metadata is discarded",
    ])

    writer, _ = _open_writer(output_path, key=key, compression=compression)
    try:
        parent: Optional[str] = None
        parent_outputs_hash: Optional[str] = None
        for i, entry in enumerate(data, start=1):
            if not isinstance(entry, dict):
                report.skipped.append(f"entry[{i}]: not an object")
                continue
            req_body = (entry.get("request") or {}).get("request_body") or {}
            resp_body = (entry.get("response") or {}).get("body") or {}
            model = str(req_body.get("model") or entry.get("model") or "unknown")
            messages = list(req_body.get("messages") or [])
            usage = resp_body.get("usage") or {}
            cost = compute_cost(model, usage)
            request = {
                "model": model,
                "temperature": float(req_body.get("temperature", 0.0)),
                "seed": req_body.get("seed", 42),
                "messages": messages,
                "tools": req_body.get("tools"),
                "response_format": req_body.get("response_format"),
            }
            sid = f"step:{i}"
            inputs = {"kind": "llm_call", **request}
            if parent:
                inputs["context"] = parent_outputs_hash
            step = _emit_step(
                writer, step_id=sid, step_kind="llm_call", name=model,
                parent_step_id=parent, inputs=inputs, outputs=resp_body,
                cost_usd=cost,
                extras={"llm_request": request, "llm_response": resp_body},
                wallclock_ns=_parse_wallclock(entry.get("created_at")),
            )
            _bump_kind(report, "llm_call")
            report.total_cost_usd += cost
            parent = sid
            parent_outputs_hash = hash_obj(resp_body)
    finally:
        writer.close()
    return report


# --------------------------------------------------------- Langfuse export


def import_langfuse_export(
    input_path: str,
    output_path: str,
    *,
    key: Optional[RecorderKey] = None,
    compression: bool = True,
) -> ImportReport:
    """Import a Langfuse trace export (JSON array) into a ``.sb`` file.

    Langfuse exports traces as a JSON array where each trace has an
    ``observations`` array. Each observation has a ``type`` field:
    ``"GENERATION"`` → ``llm_call``, ``"SPAN"`` → ``router``,
    ``"EVENT"`` → skipped.

    Dropped fields: ``scores``, ``userId``, ``sessionId``, ``tags``,
    observation-level ``metadata``.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data") or data.get("traces") or [data]
    if not isinstance(data, list):
        raise ImportError(
            f"{input_path}: expected a JSON array, got {type(data).__name__}"
        )
    if not data:
        raise ImportError(f"{input_path}: empty Langfuse export")

    report = ImportReport(output_path=output_path, source_format="langfuse_export")
    report.lossiness.synthesized.extend([
        "step_id: fabricated from sequential index",
        "nondeterminism_hash: set to zero-hash",
    ])
    report.lossiness.approximated.extend([
        "step_kind: inferred from observation type field",
        "cost_usd: estimated from token counts",
    ])
    report.lossiness.dropped.extend([
        "scores: Langfuse evaluation scores are not representable in stepback",
        "userId: user identifier attached to traces is discarded",
        "sessionId: session identifier is discarded",
        "tags: trace-level and observation-level tags are discarded",
        "metadata: arbitrary trace/observation metadata key/value pairs are discarded",
    ])

    # Collect all observations across all traces, preserving parentObservationId
    all_obs: List[dict] = []
    for trace in data:
        if not isinstance(trace, dict):
            continue
        for obs in (trace.get("observations") or []):
            if isinstance(obs, dict):
                all_obs.append(obs)

    if not all_obs:
        raise ImportError(f"{input_path}: no observations found in Langfuse export")

    ordered = _topological_order(
        all_obs, id_key="id", parent_key="parentObservationId"
    )

    id_map: Dict[str, str] = {}
    out_hash_by_sid: Dict[str, str] = {}
    writer, _ = _open_writer(output_path, key=key, compression=compression)
    try:
        for i, obs in enumerate(ordered, start=1):
            obs_id = str(obs["id"])
            sid = f"step:{i}"
            id_map[obs_id] = sid

            obs_type = str(obs.get("type") or "SPAN").upper()
            parent_obs = obs.get("parentObservationId")
            parent_sid = id_map.get(str(parent_obs)) if parent_obs else None
            name = obs.get("name")
            wallclock = _parse_wallclock(obs.get("startTime") or obs.get("start_time"))

            if obs_type == "GENERATION":
                model = str(obs.get("model") or name or "unknown")
                inp = obs.get("input") or {}
                messages = (inp.get("messages") if isinstance(inp, dict)
                            else [{"role": "user", "content": str(inp)}] if isinstance(inp, str)
                            else [])
                usage_raw = obs.get("usage") or {}
                usage = {
                    "prompt_tokens": int(usage_raw.get("input", 0) or 0),
                    "completion_tokens": int(usage_raw.get("output", 0) or 0),
                    "total_tokens": int(usage_raw.get("total", 0) or 0),
                }
                cost = compute_cost(model, usage)
                out = obs.get("output") or {}
                if isinstance(out, str):
                    response = {"choices": [{"index": 0, "finish_reason": "stop",
                                             "message": {"role": "assistant", "content": out}}],
                                "usage": usage, "model": model}
                elif isinstance(out, dict):
                    response = {**out, "usage": usage, "model": model}
                else:
                    response = {"output": out, "usage": usage, "model": model}
                request = {"model": model, "temperature": 0.0, "seed": 42,
                           "messages": messages, "tools": None, "response_format": None}
                inputs = {"kind": "llm_call", **request}
                if parent_sid:
                    inputs["context"] = out_hash_by_sid[parent_sid]
                step = _emit_step(
                    writer, step_id=sid, step_kind="llm_call", name=model,
                    parent_step_id=parent_sid, inputs=inputs, outputs=response,
                    cost_usd=cost,
                    extras={"llm_request": request, "llm_response": response},
                    wallclock_ns=wallclock,
                )
                _bump_kind(report, "llm_call")
                report.total_cost_usd += cost
            elif obs_type == "EVENT":
                report.skipped.append(f"{obs_id}: EVENT observations are skipped")
                # Still assign an id so children can link to it.
                # Use a router passthrough.
                rname = str(name or "event")
                inputs = {"kind": "router", "name": rname, "options": []}
                if parent_sid:
                    inputs["context"] = out_hash_by_sid.get(parent_sid, "")
                step = _emit_step(
                    writer, step_id=sid, step_kind="router", name=rname,
                    parent_step_id=parent_sid,
                    inputs=inputs, outputs={"choice": rname, "outputs": None},
                    wallclock_ns=wallclock,
                )
                _bump_kind(report, "router")
            else:  # SPAN or unknown → router
                rname = str(name or obs_type or "span")
                inputs = {"kind": "router", "name": rname, "options": []}
                if parent_sid:
                    inputs["context"] = out_hash_by_sid.get(parent_sid, "")
                step = _emit_step(
                    writer, step_id=sid, step_kind="router", name=rname,
                    parent_step_id=parent_sid,
                    inputs=inputs, outputs={"choice": rname, "outputs": None},
                    wallclock_ns=wallclock,
                )
                _bump_kind(report, "router")
            out_hash_by_sid[sid] = step["outputs_hash"]
    finally:
        writer.close()
    return report


# --------------------------------------------------------- Datadog APM


def import_datadog_apm(
    input_path: str,
    output_path: str,
    *,
    key: Optional[RecorderKey] = None,
    compression: bool = True,
) -> ImportReport:
    """Import Datadog APM spans (JSON array) into a ``.sb`` file.

    Datadog spans have ``span_id``, ``parent_id`` (``"0"`` for root), ``name``,
    ``type``, ``meta`` (string tags) and ``metrics`` (numeric tags). LLM spans
    are identified by ``type == "llm"`` or the ``ai.model.name`` meta tag.

    Dropped fields: ``service``, ``resource``, ``trace_id``, ``error``,
    ``start`` (nanosecond epoch), ``duration``, Datadog-internal tags.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Datadog trace API wraps spans in {"traces": [[span, ...], ...]}
        traces = data.get("traces") or []
        if traces and isinstance(traces[0], list):
            spans: List[dict] = []
            for t in traces:
                spans.extend(t)
        else:
            spans = traces or [data]
    else:
        spans = data
    if not isinstance(spans, list):
        raise ImportError(f"{input_path}: expected a list of spans")
    if not spans:
        raise ImportError(f"{input_path}: empty Datadog APM export")

    # Normalise: Datadog uses string span_id / parent_id
    normalised: List[dict] = []
    for s in spans:
        if not isinstance(s, dict):
            continue
        sid = str(s.get("span_id") or s.get("spanID") or "")
        if not sid:
            continue
        pid_raw = str(s.get("parent_id") or s.get("parentID") or "0")
        pid = None if pid_raw in ("0", "00000000000000000000") else pid_raw
        normalised.append({**s, "span_id": sid, "parent_span_id": pid})
    if not normalised:
        raise ImportError(f"{input_path}: no Datadog spans had a span_id")

    ordered = _topological_order(normalised, id_key="span_id", parent_key="parent_span_id")

    id_map: Dict[str, str] = {}
    out_hash_by_sid: Dict[str, str] = {}
    report = ImportReport(output_path=output_path, source_format="datadog_apm")
    report.lossiness.synthesized.extend([
        "step_id: fabricated from sequential index",
        "nondeterminism_hash: set to zero-hash",
    ])
    report.lossiness.approximated.extend([
        "step_kind: inferred from span type and meta tags",
        "cost_usd: estimated from token counts",
    ])
    report.lossiness.dropped.extend([
        "service: Datadog service name is not mapped to any stepback field",
        "trace_id: Datadog trace_id is not mapped to any stepback field",
        "resource: Datadog resource name (URL/query) is discarded",
        "error: Datadog error flag and error.* meta tags are discarded",
        "start/duration: raw timing nanoseconds are discarded (wallclock_ns used)",
    ])

    writer, _ = _open_writer(output_path, key=key, compression=compression)
    try:
        for i, span in enumerate(ordered, start=1):
            sid = f"step:{i}"
            id_map[span["span_id"]] = sid
            meta = span.get("meta") or {}
            metrics = span.get("metrics") or {}
            parent_sid = id_map.get(span["parent_span_id"]) if span["parent_span_id"] else None
            name = span.get("name") or span.get("resource") or "span"
            wallclock = _parse_wallclock(span.get("start"))

            # Detect LLM call
            is_llm = (
                str(span.get("type") or "").lower() == "llm"
                or "ai.model.name" in meta
                or "openai.request.model" in meta
            )

            if is_llm:
                model = str(
                    meta.get("ai.model.name") or meta.get("openai.request.model") or name
                )
                # Reconstruct messages from openai.request.messages.N.{role,content}
                messages: List[dict] = []
                idx = 0
                while True:
                    role = meta.get(f"openai.request.messages.{idx}.role")
                    content = meta.get(f"openai.request.messages.{idx}.content")
                    if role is None and content is None:
                        break
                    messages.append({"role": role or "user", "content": content or ""})
                    idx += 1
                in_tok = int(metrics.get("openai.response.usage.prompt_tokens", 0) or 0)
                out_tok = int(metrics.get("openai.response.usage.completion_tokens", 0) or 0)
                usage = {"prompt_tokens": in_tok, "completion_tokens": out_tok,
                         "total_tokens": in_tok + out_tok}
                cost = compute_cost(model, usage)
                # Reconstruct response messages
                resp_messages: List[dict] = []
                ridx = 0
                while True:
                    role = meta.get(f"openai.response.completions.{ridx}.role")
                    content = meta.get(f"openai.response.completions.{ridx}.content")
                    if role is None and content is None:
                        break
                    resp_messages.append({"role": role or "assistant",
                                          "content": content or ""})
                    ridx += 1
                request = {"model": model, "temperature": 0.0, "seed": 42,
                           "messages": messages, "tools": None, "response_format": None}
                response = {
                    "choices": [{"index": j, "finish_reason": "stop", "message": m}
                                for j, m in enumerate(resp_messages)],
                    "usage": usage, "model": model,
                }
                inputs = {"kind": "llm_call", **request}
                if parent_sid:
                    inputs["context"] = out_hash_by_sid[parent_sid]
                step = _emit_step(
                    writer, step_id=sid, step_kind="llm_call", name=model,
                    parent_step_id=parent_sid, inputs=inputs, outputs=response,
                    cost_usd=cost,
                    extras={"llm_request": request, "llm_response": response},
                    wallclock_ns=wallclock,
                )
                _bump_kind(report, "llm_call")
                report.total_cost_usd += cost
            elif str(span.get("type") or "").lower() in ("tool", "retrieval"):
                tool_name = str(meta.get("tool.name") or name)
                args = {"input": meta.get("input.value") or ""}
                result = meta.get("output.value") or {}
                inputs = {"kind": "tool_call", "name": tool_name, "arguments": args}
                if parent_sid:
                    inputs["context"] = out_hash_by_sid[parent_sid]
                step = _emit_step(
                    writer, step_id=sid, step_kind="tool_call", name=tool_name,
                    parent_step_id=parent_sid, inputs=inputs, outputs={"result": result},
                    wallclock_ns=wallclock,
                )
                _bump_kind(report, "tool_call")
            else:
                rname = str(name)
                inputs = {"kind": "router", "name": rname, "options": []}
                if parent_sid:
                    inputs["context"] = out_hash_by_sid.get(parent_sid, "")
                step = _emit_step(
                    writer, step_id=sid, step_kind="router", name=rname,
                    parent_step_id=parent_sid,
                    inputs=inputs, outputs={"choice": rname, "outputs": None},
                    wallclock_ns=wallclock,
                )
                _bump_kind(report, "router")
            out_hash_by_sid[sid] = step["outputs_hash"]
    finally:
        writer.close()
    return report


# --------------------------------------------------------- dispatcher


_FORMAT_DISPATCH = {
    "openai_chat_log": import_openai_chat_log,
    "openai": import_openai_chat_log,
    "langsmith": import_langsmith_jsonl,
    "langsmith_jsonl": import_langsmith_jsonl,
    "openinference": import_openinference_spans,
    "openinference_spans": import_openinference_spans,
    "otel": import_otel_spans,
    "otel_spans": import_otel_spans,
    "phoenix": import_phoenix_spans,
    "phoenix_spans": import_phoenix_spans,
    "helicone": import_helicone_log,
    "helicone_log": import_helicone_log,
    "langfuse": import_langfuse_export,
    "langfuse_export": import_langfuse_export,
    "datadog": import_datadog_apm,
    "datadog_apm": import_datadog_apm,
}


# ---------------------------------------------------------- native JSON


def import_native_json(
    input_path: str,
    output_path: str,
    *,
    key: Optional[RecorderKey] = None,
    compression: bool = True,
) -> ImportReport:
    """Import a stepback native JSON dump emitted by
    :func:`stepback.export_native_json` back into a ``.sb`` file.

    The expected payload has shape::

        {"format": "stepback_native_json_v1",
         "header": {...optional...},
         "steps":  [<step>, ...]}

    Each step dict is written verbatim as a step frame, preserving
    every field from the original recording. The receipt chain (HMAC
    + Ed25519) is re-minted with the supplied :class:`RecorderKey`.
    """
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ImportError(
            f"could not read native JSON {input_path!r}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ImportError(
            f"native JSON top-level must be an object, "
            f"got {type(payload).__name__}"
        )
    fmt = payload.get("format")
    if fmt != NATIVE_JSON_FORMAT_TAG:
        raise ImportError(
            f"native JSON format tag mismatch: "
            f"expected {NATIVE_JSON_FORMAT_TAG!r}, got {fmt!r}"
        )
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        raise ImportError(
            f"native JSON 'steps' must be a list, "
            f"got {type(raw_steps).__name__}"
        )

    writer, key = _open_writer(output_path, key=key, compression=compression)
    report = ImportReport(output_path=output_path, source_format="native_json")
    try:
        for i, step in enumerate(raw_steps):
            if not isinstance(step, dict):
                raise ImportError(
                    f"steps[{i}] is {type(step).__name__}, expected dict"
                )
            if "step_id" not in step or "step_kind" not in step:
                raise ImportError(
                    f"steps[{i}] missing step_id/step_kind"
                )
            writer.write_step(step)
            _bump_kind(report, str(step.get("step_kind") or "unknown"))
            cost = step.get("cost_usd")
            if isinstance(cost, (int, float)):
                report.total_cost_usd += float(cost)
    finally:
        writer.close()
    return report


_FORMAT_DISPATCH["json"] = import_native_json
_FORMAT_DISPATCH["native_json"] = import_native_json
_FORMAT_DISPATCH["stepback_json"] = import_native_json


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


# --------------------------------------------------------- CycloneDX-AI importer

def import_cyclonedx_ai(
    input_path: str,
    output_path: str,
    *,
    key: Optional[RecorderKey] = None,
    compression: bool = True,
) -> ImportReport:
    """Import a CycloneDX-AI BOM JSON file into a ``.sb`` trace.

    Each BOM component is mapped to a stepback step:

    * ``machine-learning-model`` → ``llm_call``
    * ``library`` → ``tool_call``
    * ``data`` → ``router``
    * anything else → ``tool_call``

    Accepts either a full CycloneDX BOM object (``bomFormat == "CycloneDX"``
    plus a ``components`` array) or a bare JSON array of component objects.

    Raises :class:`ImportError` for missing files, invalid JSON, or documents
    that are not CycloneDX BOMs.
    """
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ImportError(
            f"could not read CycloneDX-AI file {input_path!r}: {exc}"
        ) from exc

    schema_version: Optional[str] = None

    if isinstance(raw, list):
        components = raw
    elif isinstance(raw, dict):
        if raw.get("bomFormat") != "CycloneDX":
            raise ImportError(
                "document is not a CycloneDX BOM "
                f"(bomFormat={raw.get('bomFormat')!r})"
            )
        components = raw.get("components", [])
        schema_version = raw.get("specVersion")
    else:
        raise ImportError(
            f"CycloneDX-AI input must be a JSON object or array, "
            f"got {type(raw).__name__}"
        )

    _COMPONENT_KIND: Dict[str, str] = {
        "machine-learning-model": "llm_call",
        "library": "tool_call",
        "data": "router",
    }

    writer, _key = _open_writer(output_path, key=key, compression=compression)
    report = ImportReport(
        output_path=output_path,
        source_format="cyclonedx_ai",
        schema_version=schema_version,
    )
    try:
        for i, comp in enumerate(components):
            if not isinstance(comp, dict):
                continue
            comp_type = str(comp.get("type", "library"))
            step_kind = _COMPONENT_KIND.get(comp_type, "tool_call")
            name = comp.get("name") or f"component-{i}"
            version = comp.get("version")
            inputs: Dict[str, Any] = {"component_name": name}
            if version:
                inputs["version"] = version
            bom_ref = comp.get("bom-ref")
            if bom_ref:
                inputs["bom_ref"] = bom_ref
            model_card = comp.get("modelCard")
            if model_card is not None:
                inputs["modelCard"] = model_card
            _emit_step(
                writer,
                step_id=f"cydx-{i:04d}",
                step_kind=step_kind,
                name=name,
                parent_step_id=None,
                inputs=inputs,
                outputs={"type": comp_type, "name": name},
            )
            _bump_kind(report, step_kind)
    finally:
        writer.close()
    return report


__all__ = list(__all__) + ["import_cyclonedx_ai"]
_FORMAT_DISPATCH["cyclonedx_ai"] = import_cyclonedx_ai
_FORMAT_DISPATCH["cyclonedx"] = import_cyclonedx_ai
