"""Foreign-trace exporters.

The symmetric counterpart to :mod:`stepback.importers`. Where the
importer module bridges *foreign → stepback* (so that LangSmith /
OpenInference / OpenAI-log users can replay-with-substitutions an
already-recorded production trace), the exporter module bridges
*stepback → foreign* (so a team that has adopted stepback for its
record/replay/branch superpowers can keep visualising those traces in
the dashboards they already pay for).

Supported target formats (v0.1):

* :func:`export_openai_chat_log` — Emits a JSON array of OpenAI
  chat-completion calls. The simplest format: each ``llm_call`` step
  becomes one ``{"model": ..., "messages": [...], "response": {...}}``
  entry. ``tool_call`` steps are folded into the *next* ``llm_call``'s
  message list as ``{"role": "tool", ...}`` messages so the round-trip
  via the importer reconstructs an equivalent flat call sequence.

* :func:`export_langsmith_jsonl` — LangSmith run export. JSONL where
  each line is a run dict with ``id``, ``parent_run_id``, ``run_type``
  (``llm`` / ``tool`` / ``chain``), ``inputs``, ``outputs``, ``name``,
  ``extra.invocation_params``, ``extra.token_usage``, and
  ``start_time``. The full call tree (parent edges) is preserved.

* :func:`export_openinference_spans` — OpenInference / OpenTelemetry
  JSON spans. Each step becomes one span with ``span_id``,
  ``parent_span_id``, and the standard OpenInference attribute set
  (``openinference.span.kind``, ``llm.model_name``,
  ``llm.input_messages.<i>.message.{role,content}``,
  ``llm.token_count.{prompt,completion,total}``, ``tool.name``,
  ``tool.parameters``, ``output.value``).

All exporters return an :class:`ExportReport` with counts of frames
emitted per kind plus any steps that were skipped (e.g. a step kind
that has no representation in the target format — currently
``parallel_branch_open`` / ``parallel_branch_join`` are folded into
parent edges and counted as ``skipped`` for langsmith/openai).

Round-trip property (verified by :mod:`tests.test_exporters`)::

    .sb  ──export──▶  langsmith.jsonl  ──import──▶  .sb'
                                                     │
                                                     ▼
            same step_count, same kind_counts, same parent edges,
            same llm_request.model values, same tool_call names.

The exporters never re-execute LLMs and never need network access:
they read the steps verbatim out of the trace, transform them into
the target schema, and write them out.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .canonical import canonical_json
from .importers import LossReport
from .recorder import RecorderKey
from .trace_reader import verify_trace


__all__ = [
    "ExportError",
    "TraceExportError",
    "ExportReport",
    "validate_native_json_doc",
    "export_openai_chat_log",
    "export_langsmith_jsonl",
    "export_openinference_spans",
    "export_otel_spans",
    "export_cyclonedx_ai",
    "export_native_json",
    "export_html_view",
    "export_trace",
    "export_trace_file",
    "available_export_formats",
]

NATIVE_JSON_FORMAT_TAG = "stepback_native_json_v1"


class ExportError(ValueError):
    """Raised when a stepback trace cannot be rendered into the target format."""

    #: Canonical error code; see :mod:`stepback.errors` for details.
    code: str = "SB403"


TraceExportError = ExportError


@dataclass
class ExportReport:
    """Summary of an export operation."""

    output_path: str
    target_format: str
    step_count: int = 0
    kind_counts: Dict[str, int] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)
    lossiness: LossReport = field(default_factory=LossReport)

    def as_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "target_format": self.target_format,
            "step_count": self.step_count,
            "kind_counts": dict(self.kind_counts),
            "skipped": list(self.skipped),
            "lossiness": self.lossiness.as_dict(),
        }


# --------------------------------------------------------------- helpers


def _bump(report: ExportReport, kind: str) -> None:
    report.kind_counts[kind] = report.kind_counts.get(kind, 0) + 1
    report.step_count += 1


def _normalise_steps(steps: Sequence[dict]) -> List[dict]:
    """Defensive copy + light validation of the input step list."""
    if not isinstance(steps, (list, tuple)):
        raise ExportError(
            f"steps must be a list/tuple, got {type(steps).__name__}"
        )
    out: List[dict] = []
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            raise ExportError(f"steps[{i}] is {type(s).__name__}, expected dict")
        if "step_id" not in s or "step_kind" not in s:
            raise ExportError(
                f"steps[{i}] missing step_id/step_kind: keys={list(s)}"
            )
        out.append(s)
    return out


def _wallclock_to_iso(ns: Optional[int]) -> Optional[str]:
    if ns is None:
        return None
    try:
        sec = int(ns) / 1_000_000_000
        return datetime.fromtimestamp(sec, tz=timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _llm_request(step: dict) -> Dict[str, Any]:
    """Best-effort recovery of the original llm_request dict."""
    if isinstance(step.get("llm_request"), dict):
        return dict(step["llm_request"])
    inp = step.get("inputs") or {}
    if not isinstance(inp, dict):
        return {}
    out: Dict[str, Any] = {}
    for k in ("model", "temperature", "seed", "messages", "tools", "response_format"):
        if k in inp:
            out[k] = inp[k]
    return out


def _llm_response(step: dict) -> Dict[str, Any]:
    if isinstance(step.get("llm_response"), dict):
        return dict(step["llm_response"])
    out = step.get("outputs")
    return dict(out) if isinstance(out, dict) else {}


def _tool_args(step: dict) -> Dict[str, Any]:
    inp = step.get("inputs") or {}
    if isinstance(inp, dict):
        a = inp.get("arguments")
        if isinstance(a, dict):
            return dict(a)
        if a is not None:
            return {"input": a}
    return {}


def _tool_result(step: dict) -> Any:
    out = step.get("outputs") or {}
    if isinstance(out, dict) and "result" in out:
        return out["result"]
    return out


def _llm_messages(step: dict) -> List[dict]:
    req = _llm_request(step)
    msgs = req.get("messages")
    if isinstance(msgs, list):
        return [m for m in msgs if isinstance(m, dict)]
    return []


def _llm_usage(step: dict) -> Dict[str, int]:
    resp = _llm_response(step)
    if isinstance(resp, dict):
        u = resp.get("usage")
        if isinstance(u, dict):
            return {k: int(v) for k, v in u.items() if isinstance(v, (int, float))}
    return {}


def _llm_output_messages(step: dict) -> List[dict]:
    resp = _llm_response(step)
    out: List[dict] = []
    if isinstance(resp, dict):
        choices = resp.get("choices")
        if isinstance(choices, list):
            for c in choices:
                if isinstance(c, dict):
                    msg = c.get("message")
                    if isinstance(msg, dict):
                        out.append(msg)
    return out


def _build_id_map(steps: Sequence[dict]) -> Dict[str, str]:
    """Map stepback step_id → stable foreign id (sha256-derived, raw hex)."""
    import hashlib
    out: Dict[str, str] = {}
    for s in steps:
        sid = str(s["step_id"])
        digest = hashlib.sha256(canonical_json({"step_id": sid})).hexdigest()
        out[sid] = digest[:32]
    return out


# ---------------------------------------------------------- OpenAI log


def export_openai_chat_log(
    steps: Sequence[dict],
    output_path: str,
    *,
    include_tool_calls: bool = True,
) -> ExportReport:
    """Export a stepback step list to a JSON array of OpenAI chat calls.

    Only ``llm_call`` steps produce one entry each. ``tool_call``
    steps are skipped (their results are typically already encoded in
    the next ``llm_call``'s message list as a ``role: tool`` message
    when produced by openai's tool-use loop). ``router`` and
    ``parallel_branch_*`` steps are skipped and recorded under
    ``skipped`` in the report.

    The output JSON is a list and each entry has the exact shape that
    :func:`stepback.importers.import_openai_chat_log` accepts, so the
    round-trip ``.sb → json → .sb`` reconstructs the same llm_call
    sequence.
    """
    steps = _normalise_steps(steps)
    report = ExportReport(output_path=output_path, target_format="openai_chat_log")
    report.lossiness.dropped.extend([
        "nondeterminism_hash: not representable in OpenAI chat log format",
        "step_id: step IDs are not emitted in OpenAI chat log format",
        "inputs_hash / outputs_hash: not representable in OpenAI chat log format",
    ])
    report.lossiness.absent.extend([
        "tool_call steps: tool call steps are folded into adjacent llm_call messages; "
        "round-trip will not preserve them as separate steps",
        "router steps: router/chain steps have no representation in OpenAI chat log format",
    ])
    out: List[dict] = []
    for s in steps:
        kind = s.get("step_kind")
        if kind == "llm_call":
            req = _llm_request(s)
            entry = {
                "model": str(req.get("model", "unknown")),
                "messages": list(req.get("messages") or []),
                "temperature": float(req.get("temperature", 0.0) or 0.0),
                "seed": req.get("seed", 42),
                "response": _llm_response(s),
            }
            if req.get("tools") is not None:
                entry["tools"] = req["tools"]
            if req.get("response_format") is not None:
                entry["response_format"] = req["response_format"]
            wc = s.get("wallclock_ns")
            if wc is not None:
                entry["wallclock_ns"] = int(wc)
            out.append(entry)
            _bump(report, "llm_call")
        elif kind == "tool_call" and include_tool_calls:
            # Encode as a tool-message-only entry — the importer skips
            # entries without "model"/"messages", so we don't emit one.
            report.skipped.append(
                f"{s.get('step_id')}: tool_call (folded into next llm_call upstream)"
            )
        else:
            report.skipped.append(
                f"{s.get('step_id')}: kind={kind!r} not representable in openai_chat_log"
            )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=False)
        f.write("\n")
    return report


# ------------------------------------------------------------ LangSmith


_KIND_TO_LS = {
    "llm_call": "llm",
    "tool_call": "tool",
    "router": "chain",
    "parallel_branch_open": "chain",
    "parallel_branch_join": "chain",
    "exception": "chain",
}


def export_langsmith_jsonl(
    steps: Sequence[dict],
    output_path: str,
) -> ExportReport:
    """Export a stepback step list to LangSmith JSONL run format.

    The full parent-child tree is preserved: each step gets a
    deterministic foreign UUID derived from its ``step_id`` (so two
    exports of the same trace are byte-identical), and
    ``parent_run_id`` points at the foreign id of the parent step.

    Each line is one JSON object with the LangSmith run fields
    accepted by :func:`stepback.importers.import_langsmith_jsonl`:
    ``id``, ``parent_run_id``, ``run_type``, ``name``, ``inputs``,
    ``outputs``, ``extra.invocation_params``, ``extra.token_usage``,
    ``start_time``.
    """
    steps = _normalise_steps(steps)
    id_map = _build_id_map(steps)
    report = ExportReport(output_path=output_path, target_format="langsmith_jsonl")
    report.lossiness.synthesized.extend([
        "UUID run id: stepback step_ids are converted to generated UUIDs",
    ])
    report.lossiness.approximated.extend([
        "run_type: approximated from step_kind (may differ from original LangSmith run_type)",
    ])
    report.lossiness.dropped.extend([
        "nondeterminism_hash: not representable in LangSmith format",
        "inputs_hash / outputs_hash: not emitted in LangSmith format",
        "cost_usd: not emitted; token usage is emitted instead",
    ])
    with open(output_path, "w", encoding="utf-8") as f:
        for s in steps:
            sid = str(s["step_id"])
            kind = s.get("step_kind")
            ls_kind = _KIND_TO_LS.get(str(kind))
            if ls_kind is None:
                report.skipped.append(f"{sid}: kind={kind!r} not representable")
                continue
            parent_sid = s.get("parent_step_id")
            run: Dict[str, Any] = {
                "id": id_map[sid],
                "parent_run_id": id_map.get(str(parent_sid)) if parent_sid else None,
                "run_type": ls_kind,
                "name": s.get("name") or kind,
            }
            wc_iso = _wallclock_to_iso(s.get("wallclock_ns"))
            if wc_iso:
                run["start_time"] = wc_iso
            if kind == "llm_call":
                req = _llm_request(s)
                resp = _llm_response(s)
                run["inputs"] = {"messages": list(req.get("messages") or [])}
                run["outputs"] = resp
                inv: Dict[str, Any] = {
                    "model": str(req.get("model", "unknown")),
                    "temperature": float(req.get("temperature", 0.0) or 0.0),
                    "seed": req.get("seed", 42),
                }
                if req.get("tools") is not None:
                    inv["tools"] = req["tools"]
                if req.get("response_format") is not None:
                    inv["response_format"] = req["response_format"]
                extra: Dict[str, Any] = {"invocation_params": inv}
                usage = _llm_usage(s)
                if usage:
                    extra["token_usage"] = usage
                run["extra"] = extra
                run["name"] = inv["model"]
            elif kind == "tool_call":
                run["inputs"] = _tool_args(s)
                run["outputs"] = {"output": _tool_result(s)}
            else:  # router / parallel_branch_* / exception
                inp = s.get("inputs") or {}
                out = s.get("outputs") or {}
                run["inputs"] = dict(inp) if isinstance(inp, dict) else {"input": inp}
                run["outputs"] = dict(out) if isinstance(out, dict) else {"output": out}
            cost = s.get("cost_usd")
            if cost:
                run["extra"] = run.get("extra") or {}
                run["extra"]["cost_usd"] = float(cost)
            f.write(json.dumps(run, sort_keys=True))
            f.write("\n")
            _bump(report, ls_kind)
    return report


# --------------------------------------------------------- OpenInference


_KIND_TO_OI = {
    "llm_call": "LLM",
    "tool_call": "TOOL",
    "router": "CHAIN",
    "parallel_branch_open": "CHAIN",
    "parallel_branch_join": "CHAIN",
    "exception": "CHAIN",
}


def export_openinference_spans(
    steps: Sequence[dict],
    output_path: str,
    *,
    envelope: bool = True,
) -> ExportReport:
    """Export a stepback step list to OpenInference / OTel JSON spans.

    Output is a JSON object ``{"spans": [ ...span objects... ]}`` when
    ``envelope=True`` (the default, matching what most OpenInference
    consumers expect), or a bare ``[ ...span objects... ]`` array
    otherwise.

    Each span has:

    * ``span_id`` — deterministic 32-hex derived from the stepback id
    * ``parent_span_id`` — the parent step's foreign id (omitted for
      the root)
    * ``name`` — step name (model id for llm, tool name for tool)
    * ``start_time_unix_nano`` — wall-clock from the step
    * ``attributes`` — list of ``{"key", "value"}`` OTel KVs:

      For llm_call: ``openinference.span.kind = "LLM"``,
      ``llm.model_name``, ``llm.input_messages.<i>.message.{role,content}``,
      ``llm.output_messages.<i>.message.{role,content}``,
      ``llm.token_count.{prompt,completion,total}``,
      ``llm.invocation_parameters.temperature``,
      ``llm.invocation_parameters.seed``.

      For tool_call: ``openinference.span.kind = "TOOL"``,
      ``tool.name``, ``tool.parameters`` (JSON-encoded string),
      ``output.value`` (JSON-encoded string).

      For router/passthrough: ``openinference.span.kind = "CHAIN"``,
      ``input.value``, ``output.value``.
    """
    steps = _normalise_steps(steps)
    id_map = _build_id_map(steps)
    report = ExportReport(output_path=output_path, target_format="openinference_spans")
    report.lossiness.synthesized.extend([
        "span_id: fabricated from step_id (OpenInference uses string span IDs)",
    ])
    report.lossiness.dropped.extend([
        "nondeterminism_hash: not representable as an OpenInference span attribute",
        "cost_usd: not emitted in OpenInference format",
    ])
    spans: List[dict] = []
    for s in steps:
        sid = str(s["step_id"])
        kind = s.get("step_kind")
        oi_kind = _KIND_TO_OI.get(str(kind))
        if oi_kind is None:
            report.skipped.append(f"{sid}: kind={kind!r} not representable")
            continue
        attrs: List[dict] = [
            {"key": "openinference.span.kind", "value": {"stringValue": oi_kind}},
        ]
        if kind == "llm_call":
            req = _llm_request(s)
            model = str(req.get("model", "unknown"))
            attrs.append({"key": "llm.model_name", "value": {"stringValue": model}})
            attrs.append({
                "key": "llm.invocation_parameters.temperature",
                "value": {"doubleValue": float(req.get("temperature", 0.0) or 0.0)},
            })
            attrs.append({
                "key": "llm.invocation_parameters.seed",
                "value": {"intValue": int(req.get("seed", 42) or 0)},
            })
            for i, m in enumerate(_llm_messages(s)):
                role = m.get("role", "")
                content = m.get("content", "")
                if not isinstance(content, str):
                    content = json.dumps(content, sort_keys=True)
                attrs.append({
                    "key": f"llm.input_messages.{i}.message.role",
                    "value": {"stringValue": str(role)},
                })
                attrs.append({
                    "key": f"llm.input_messages.{i}.message.content",
                    "value": {"stringValue": content},
                })
            for i, m in enumerate(_llm_output_messages(s)):
                role = m.get("role", "assistant")
                content = m.get("content", "")
                if not isinstance(content, str):
                    content = json.dumps(content, sort_keys=True)
                attrs.append({
                    "key": f"llm.output_messages.{i}.message.role",
                    "value": {"stringValue": str(role)},
                })
                attrs.append({
                    "key": f"llm.output_messages.{i}.message.content",
                    "value": {"stringValue": content},
                })
            usage = _llm_usage(s)
            for k_name, attr_name in (
                ("prompt_tokens", "llm.token_count.prompt"),
                ("completion_tokens", "llm.token_count.completion"),
                ("total_tokens", "llm.token_count.total"),
            ):
                if k_name in usage:
                    attrs.append({
                        "key": attr_name,
                        "value": {"intValue": int(usage[k_name])},
                    })
            name = model
        elif kind == "tool_call":
            tool_name = str(s.get("name") or "tool")
            attrs.append({"key": "tool.name", "value": {"stringValue": tool_name}})
            attrs.append({
                "key": "tool.parameters",
                "value": {"stringValue": json.dumps(_tool_args(s), sort_keys=True)},
            })
            attrs.append({
                "key": "output.value",
                "value": {"stringValue": json.dumps(_tool_result(s), sort_keys=True)},
            })
            name = tool_name
        else:
            inp = s.get("inputs") or {}
            out = s.get("outputs") or {}
            attrs.append({
                "key": "input.value",
                "value": {"stringValue": json.dumps(inp, sort_keys=True)},
            })
            attrs.append({
                "key": "output.value",
                "value": {"stringValue": json.dumps(out, sort_keys=True)},
            })
            name = str(s.get("name") or kind)

        span: Dict[str, Any] = {
            "span_id": id_map[sid],
            "name": name,
            "attributes": attrs,
        }
        parent_sid = s.get("parent_step_id")
        if parent_sid:
            span["parent_span_id"] = id_map.get(str(parent_sid))
        if s.get("wallclock_ns") is not None:
            span["start_time_unix_nano"] = int(s["wallclock_ns"])
        spans.append(span)
        _bump(report, oi_kind)

    payload: Any = {"spans": spans} if envelope else spans
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")
    return report


# --------------------------------------------------------- native JSON


def validate_native_json_doc(
    doc: Any,
    *,
    exc_class: type = ExportError,
) -> None:
    """Validate a stepback native-JSON document against the v1 schema.

    Performs a fast structural check (format tag, required fields, types)
    without loading the full JSON Schema. Raises ``exc_class`` (default:
    :class:`ExportError`) on the first validation failure.

    For full Draft-07 JSON Schema validation use :mod:`jsonschema` directly
    with the schema at ``spec/schema/v1/native_json_export.json``.
    """
    if not isinstance(doc, dict):
        raise exc_class(
            f"validate_native_json_doc: top-level must be a JSON object, "
            f"got {type(doc).__name__}"
        )
    fmt = doc.get("format")
    if fmt != NATIVE_JSON_FORMAT_TAG:
        raise exc_class(
            f"validate_native_json_doc: 'format' must be "
            f"{NATIVE_JSON_FORMAT_TAG!r}, got {fmt!r}"
        )
    if "steps" not in doc:
        raise exc_class(
            "validate_native_json_doc: required key 'steps' is missing"
        )
    steps = doc["steps"]
    if not isinstance(steps, list):
        raise exc_class(
            f"validate_native_json_doc: 'steps' must be an array, "
            f"got {type(steps).__name__}"
        )
    header = doc.get("header")
    if header is not None and not isinstance(header, dict):
        raise exc_class(
            f"validate_native_json_doc: 'header' must be an object or absent, "
            f"got {type(header).__name__}"
        )
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise exc_class(
                f"validate_native_json_doc: steps[{i}] must be a JSON object, "
                f"got {type(step).__name__}"
            )
        sid = step.get("step_id")
        if sid is None or not isinstance(sid, str):
            raise exc_class(
                f"validate_native_json_doc: steps[{i}] missing or non-string 'step_id'"
            )
        sk = step.get("step_kind")
        if sk is None or not isinstance(sk, str):
            raise exc_class(
                f"validate_native_json_doc: steps[{i}] missing or non-string 'step_kind'"
            )


def export_native_json(
    steps: Sequence[dict],
    output_path: str,
    *,
    header: Optional[Dict[str, Any]] = None,
) -> ExportReport:
    """Lossless JSON dump of recorded steps.

    Produces a single JSON object with shape::

        {
          "format": "stepback_native_json_v1",
          "header": {...},          # optional, recorder metadata
          "steps":  [<step>, ...]   # verbatim canonicalised step dicts
        }

    Every recorded field on each step (``step_id``, ``step_kind``,
    ``inputs``, ``outputs``, ``inputs_hash``, ``outputs_hash``,
    ``nondeterminism_hash``, ``parent_step_id``, ``llm_request``,
    ``llm_response``, ``cost_usd``, ``wallclock_ns``, ...) is preserved
    bit-for-bit, so the round-trip via
    :func:`stepback.import_native_json` reconstructs an
    observationally identical ``.sb`` trace (modulo the freshly-minted
    recorder key + receipt chain).
    """
    norm = _normalise_steps(steps)
    report = ExportReport(output_path=output_path, target_format="native_json")
    payload: Dict[str, Any] = {"format": NATIVE_JSON_FORMAT_TAG}
    if header is not None:
        if not isinstance(header, dict):
            raise ExportError(
                f"header must be a dict, got {type(header).__name__}"
            )
        payload["header"] = dict(header)
    payload["steps"] = list(norm)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False, default=str)
        f.write("\n")
    for s in norm:
        _bump(report, str(s.get("step_kind") or "unknown"))
    return report


# --------------------------------------------------------- HTML view


def export_html_view(
    steps: Sequence[dict],
    output_path: str,
    *,
    header: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
) -> ExportReport:
    """Render the self-contained interactive HTML viewer for ``steps``.

    Produces the same single-file ``<!doctype html>`` page that
    :func:`stepback.html_view.render_trace_html` would, including the
    embedded ``<script type='application/json' id='stepback-data'>``
    data island. The data island is the round-trip surface: parsing
    the embedded JSON reconstructs the recorded step set (id, kind,
    parent edge, hashes, cost) without re-running any LLM call.
    """
    from .html_view import render_trace_html  # local import to avoid cycle

    norm = _normalise_steps(steps)
    report = ExportReport(output_path=output_path, target_format="html")
    report.lossiness.dropped.extend([
        "nondeterminism_hash: not rendered in the HTML view",
        "inputs_hash / outputs_hash: not rendered in the HTML view",
    ])
    report.lossiness.absent.extend([
        "round-trip import: HTML is a display-only format; "
        "re-import is not supported",
    ])
    page = render_trace_html(
        norm,
        header or {},
        title=title or "stepback trace",
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(page)
    for s in norm:
        _bump(report, str(s.get("step_kind") or "unknown"))
    return report


# --------------------------------------------------------- OTel export


def export_otel_spans(
    steps: Sequence[dict],
    output_path: str,
    *,
    envelope: bool = False,
    trace_id: Optional[str] = None,
) -> ExportReport:
    """Export stepback steps as OpenTelemetry spans (JSON array).

    Uses stable ``agent.step.*`` attributes (RFC 0006) for stepback-
    specific fields, and standard ``gen_ai.*`` attributes (OTEP-0217)
    for LLM call fields.

    Lossiness: ``outputs_hash`` is dropped (nondeterminism_hash and
    inputs_hash are preserved as ``agent.step.*`` attributes).
    ``span_id`` values are fabricated from step_id.
    """
    from .canonical import sha256_hex, canonical_json  # local to avoid cycle at import

    norm = _normalise_steps(steps)
    id_map = _build_id_map(norm)
    report = ExportReport(output_path=output_path, target_format="otel_spans")
    report.lossiness.synthesized.extend([
        "span_id: fabricated from step_id hash (no stable OTel span ID in stepback)",
    ])
    report.lossiness.dropped.extend([
        "outputs_hash: not emitted as OTel attribute (nondeterminism_hash and "
        "inputs_hash are preserved as agent.step.* attrs, but outputs_hash is dropped)",
    ])

    spans: List[dict] = []
    for s in norm:
        sid = str(s["step_id"])
        kind = s.get("step_kind") or "unknown"
        parent_sid = s.get("parent_step_id")
        span_id = sha256_hex(canonical_json(sid))[:16]
        parent_span_id = (
            sha256_hex(canonical_json(str(id_map.get(parent_sid, parent_sid))))[:16]
            if parent_sid else None
        )

        attrs: List[dict] = [
            {"key": "agent.step.kind", "value": {"stringValue": kind}},
            {"key": "agent.step.id", "value": {"stringValue": sid}},
            {"key": "agent.step.inputs_hash",
             "value": {"stringValue": str(s.get("inputs_hash", ""))}},
            {"key": "agent.step.nondeterminism_hash",
             "value": {"stringValue": str(s.get("nondeterminism_hash", ""))}},
        ]
        if kind == "llm_call":
            req = s.get("llm_request") or s.get("inputs") or {}
            resp = s.get("llm_response") or s.get("outputs") or {}
            if isinstance(req, dict):
                model = req.get("model", "")
                attrs.append({"key": "gen_ai.request.model",
                               "value": {"stringValue": str(model)}})
                attrs.append({"key": "gen_ai.system",
                               "value": {"stringValue": "openai"}})
                for i, msg in enumerate(req.get("messages") or []):
                    if isinstance(msg, dict):
                        attrs.append({"key": f"llm.input_messages.{i}.message.role",
                                       "value": {"stringValue": str(msg.get("role", ""))}})
                        attrs.append({"key": f"llm.input_messages.{i}.message.content",
                                       "value": {"stringValue": str(msg.get("content", ""))}})
            if isinstance(resp, dict):
                usage = resp.get("usage") or {}
                if isinstance(usage, dict):
                    attrs.append({"key": "gen_ai.usage.input_tokens",
                                   "value": {"intValue": int(usage.get("prompt_tokens", 0))}})
                    attrs.append({"key": "gen_ai.usage.output_tokens",
                                   "value": {"intValue": int(usage.get("completion_tokens", 0))}})
                for i, ch in enumerate(resp.get("choices") or []):
                    if isinstance(ch, dict):
                        msg = ch.get("message") or {}
                        if isinstance(msg, dict):
                            attrs.append(
                                {"key": f"llm.output_messages.{i}.message.content",
                                 "value": {"stringValue": str(msg.get("content", ""))}})
        elif kind == "tool_call":
            inp = s.get("inputs") or {}
            if isinstance(inp, dict):
                attrs.append({"key": "tool.name",
                               "value": {"stringValue": str(inp.get("name", ""))}})
                args = inp.get("arguments")
                if args is not None:
                    attrs.append({"key": "tool.parameters",
                                   "value": {"stringValue": json.dumps(args)}})
            out = s.get("outputs") or {}
            if isinstance(out, dict) and "result" in out:
                attrs.append({"key": "output.value",
                               "value": {"stringValue": json.dumps(out["result"])}})

        span: Dict[str, Any] = {
            "span_id": span_id,
            "name": str(s.get("name") or kind),
            "start_time": _wallclock_to_iso(s.get("wallclock_ns")),
            "attributes": attrs,
        }
        if parent_span_id:
            span["parent_span_id"] = parent_span_id

        spans.append(span)
        _bump(report, kind)

    with open(output_path, "w", encoding="utf-8") as f:
        if envelope:
            doc: Dict[str, Any] = {"spans": spans}
            if trace_id:
                doc["trace_id"] = trace_id
            json.dump(doc, f, indent=2, default=str)
        else:
            json.dump(spans, f, indent=2, default=str)
        f.write("\n")
    return report


# --------------------------------------------------------- CycloneDX-AI export

_PROMPT_TRUNCATE = 2048  # chars – keep prompts readable but not enormous


def _cdx_prompt_properties(messages: List[dict]) -> List[dict]:
    """Return CycloneDX properties for the prompt messages of an LLM call."""
    props: List[dict] = []
    for i, msg in enumerate(messages):
        role = str(msg.get("role", ""))
        content = msg.get("content") or ""
        if isinstance(content, list):
            # multi-part content – join text parts
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        content = str(content)[:_PROMPT_TRUNCATE]
        props.append({"name": f"prompt:messages[{i}].role", "value": role})
        if content:
            props.append({"name": f"prompt:messages[{i}].content", "value": content})
    return props


def _cdx_tool_service(tool_name: str, tool_schema: Optional[dict] = None) -> dict:
    """Return a CycloneDX service component representing a callable tool."""
    svc: Dict[str, Any] = {
        "type": "service",
        "bom-ref": f"tool:{tool_name}",
        "name": tool_name,
        "properties": [{"name": "stepback:step_kind", "value": "tool_call"}],
    }
    if tool_schema:
        description = tool_schema.get("description") or tool_schema.get("function", {}).get("description", "")
        if description:
            svc["description"] = str(description)[:512]
        params = tool_schema.get("parameters") or tool_schema.get("function", {}).get("parameters")
        if params:
            svc["properties"].append(
                {"name": "tool:parameters_schema", "value": json.dumps(params, separators=(",", ":"))}
            )
    return svc


def _cdx_policy_properties(step: dict) -> List[dict]:
    """Extract policy / safety related properties from a step."""
    props: List[dict] = []
    nd = step.get("nondeterminism") or {}
    sources = nd.get("sources") if isinstance(nd, dict) else []
    for src in (sources or []):
        if not isinstance(src, dict):
            continue
        cls = str(src.get("class", ""))
        if cls in ("policy", "safety", "content_filter", "guardrail"):
            props.append({"name": f"policy:{cls}", "value": json.dumps(src, separators=(",", ":"))})
    # explicit policy_decisions field if present
    pd = step.get("policy_decisions")
    if pd:
        props.append({"name": "policy:decisions", "value": json.dumps(pd, separators=(",", ":"))})
    # finish_reason from LLM response (e.g. "content_filter")
    for choice in (_llm_response(step).get("choices") or []):
        reason = choice.get("finish_reason") if isinstance(choice, dict) else None
        if reason and reason not in ("stop", "length"):
            props.append({"name": "policy:finish_reason", "value": str(reason)})
    return props


def export_cyclonedx_ai(
    steps: Sequence[dict],
    output_path: str,
) -> ExportReport:
    """Export stepback steps as a CycloneDX-AI ML BOM JSON document.

    Produces an ``application/vnd.cyclonedx+json`` BOM (spec version 1.6)
    with the following mappings:

    * **Models** – each unique LLM model name becomes a
      ``machine-learning-model`` component with a ``modelCard`` carrying
      architecture, token-usage aggregates, temperature, and seed.
    * **Prompts** – system and user messages are embedded as
      ``prompt:messages[i].role / .content`` properties on the model
      component that consumed them.
    * **Tools** – each distinct tool name used in the trace becomes a
      ``service`` component; its description and parameter schema are
      included when available from the LLM request ``tools`` array.
    * **Datasets** – steps that carry a ``dataset_id`` field produce a
      ``data`` component referencing the dataset.
    * **Policy decisions** – safety/guardrail nondeterminism sources and
      explicit ``policy_decisions`` fields are surfaced as
      ``policy:*`` properties on the step component.

    Fields with no CycloneDX equivalent (execution order, wallclock
    timing, HMAC hashes, per-step cost) are recorded in the
    :attr:`ExportReport.lossiness` report.
    """
    import uuid as _uuid

    norm = _normalise_steps(steps)
    report = ExportReport(output_path=output_path, target_format="cyclonedx_ai")
    report.lossiness.dropped.extend([
        "execution sequence: CycloneDX-AI BOM lists components; "
        "the execution order of steps is not representable",
        "timing / wallclock_ns: per-step timing is discarded",
        "nondeterminism_hash / inputs_hash / outputs_hash: hash fields are discarded",
        "cost_usd: per-step cost is discarded",
    ])
    report.lossiness.absent.extend([
        "replay: CycloneDX-AI format has no mechanism to represent a replayable "
        "execution trace; import is not supported",
    ])

    # ---- first pass: collect unique tools and their schemas -----------------
    # Tools from LLM request ``tools`` arrays take precedence over bare names.
    tool_schemas: Dict[str, Optional[dict]] = {}
    for s in norm:
        req = _llm_request(s)
        for t in (req.get("tools") or []):
            if not isinstance(t, dict):
                continue
            fn = t.get("function") or {}
            tname = fn.get("name") or t.get("name", "")
            if tname and tname not in tool_schemas:
                tool_schemas[tname] = t
        # also collect from bare tool_call steps
        inp = s.get("inputs") or {}
        if isinstance(inp, dict) and s.get("step_kind") == "tool_call":
            tname = str(inp.get("name", ""))
            if tname and tname not in tool_schemas:
                tool_schemas[tname] = None  # schema not available here

    # service components for tools
    services: List[dict] = [
        _cdx_tool_service(tn, schema) for tn, schema in tool_schemas.items()
    ]

    # ---- second pass: build per-step components and dataset components -------
    components: List[dict] = []
    dataset_refs: Dict[str, dict] = {}
    dependencies: List[dict] = []

    for s in norm:
        kind = str(s.get("step_kind") or "unknown")
        name = str(s.get("name") or s.get("step_id") or "step")
        inp = s.get("inputs") or {}
        step_ref = str(s.get("step_id"))

        comp: Dict[str, Any] = {
            "type": "machine-learning-model",
            "bom-ref": step_ref,
            "name": name,
            "properties": [
                {"name": "stepback:step_kind", "value": kind},
            ],
        }

        if kind == "llm_call" and isinstance(inp, dict):
            model = inp.get("model") or s.get("name") or ""
            model_str = str(model) if model else ""
            comp["properties"].append(
                {"name": "gen_ai:model", "value": model_str}
            )
            # temperature / seed
            temperature = inp.get("temperature")
            if temperature is not None:
                comp["properties"].append(
                    {"name": "gen_ai:temperature", "value": str(temperature)}
                )
            seed = inp.get("seed")
            if seed is not None:
                comp["properties"].append(
                    {"name": "gen_ai:seed", "value": str(seed)}
                )
            # token usage from response
            usage = _llm_usage(s)
            for tok_key, tok_val in usage.items():
                comp["properties"].append(
                    {"name": f"gen_ai:token_usage.{tok_key}", "value": str(tok_val)}
                )
            # modelCard
            card: Dict[str, Any] = {}
            if model_str:
                card["modelParameters"] = {"modelArchitecture": model_str}
            if card:
                comp["modelCard"] = card

            # prompts
            messages = _llm_messages(s)
            comp["properties"].extend(_cdx_prompt_properties(messages))

        elif kind == "tool_call" and isinstance(inp, dict):
            tool_name = str(inp.get("name", ""))
            comp["properties"].append(
                {"name": "tool:name", "value": tool_name}
            )
            # tool arguments as property
            args = _tool_args(s)
            if args:
                comp["properties"].append(
                    {"name": "tool:arguments", "value": json.dumps(args, separators=(",", ":"))}
                )
            # link this step to its service component
            if tool_name in tool_schemas:
                dependencies.append({
                    "ref": step_ref,
                    "dependsOn": [f"tool:{tool_name}"],
                })

        elif kind in ("policy_check", "guardrail"):
            pass  # handled below via _cdx_policy_properties

        # policy decisions (applicable to any step kind)
        comp["properties"].extend(_cdx_policy_properties(s))

        # dataset reference
        dataset_id = s.get("dataset_id")
        if dataset_id:
            ds_ref = f"dataset:{dataset_id}"
            if ds_ref not in dataset_refs:
                dataset_refs[ds_ref] = {
                    "type": "data",
                    "bom-ref": ds_ref,
                    "name": str(dataset_id),
                    "properties": [
                        {"name": "stepback:component_kind", "value": "dataset"}
                    ],
                }
            dependencies.append({"ref": step_ref, "dependsOn": [ds_ref]})

        components.append(comp)
        _bump(report, kind)

    # merge dataset components into components list
    components.extend(dataset_refs.values())

    bom: Dict[str, Any] = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": f"urn:uuid:{_uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tools": [{"name": "stepback", "version": "0.1.0"}],
        },
        "components": components,
    }
    if services:
        bom["services"] = services
    if dependencies:
        bom["dependencies"] = dependencies

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bom, f, indent=2, default=str)
        f.write("\n")
    return report


# --------------------------------------------------------- dispatcher


_FORMAT_DISPATCH: Dict[str, Callable[..., Any]] = {
    "openai_chat_log": export_openai_chat_log,
    "openai": export_openai_chat_log,
    "langsmith": export_langsmith_jsonl,
    "langsmith_jsonl": export_langsmith_jsonl,
    "openinference": export_openinference_spans,
    "openinference_spans": export_openinference_spans,
    "otel": export_otel_spans,
    "otel_spans": export_otel_spans,
    "cyclonedx_ai": export_cyclonedx_ai,
    "cyclonedx": export_cyclonedx_ai,
    "json": export_native_json,
    "native_json": export_native_json,
    "stepback_json": export_native_json,
    "html": export_html_view,
    "html_view": export_html_view,
}


def available_export_formats() -> List[str]:
    """Sorted list of accepted ``target_format`` strings (incl. aliases)."""
    return sorted(_FORMAT_DISPATCH.keys())


def export_trace(
    target_format: str,
    steps: Sequence[dict],
    output_path: str,
) -> ExportReport:
    """Dispatch to the exporter for ``target_format``.

    ``target_format`` ∈ ``{"openai_chat_log", "langsmith", "openinference"}``
    (with aliases). Raises :class:`ExportError` for unknown formats.
    """
    fn = _FORMAT_DISPATCH.get(target_format.lower())
    if fn is None:
        raise ExportError(
            f"unknown target_format {target_format!r}; "
            f"known: {available_export_formats()}"
        )
    return fn(steps, output_path)


def export_trace_file(
    target_format: str,
    sb_path: str,
    output_path: str,
    *,
    hmac_key: bytes,
) -> ExportReport:
    """Convenience wrapper: verify a `.sb` file and export its steps.

    Reads ``sb_path`` with :func:`stepback.trace_reader.verify_trace`
    (so HMAC + Ed25519 chain are checked end-to-end), then dispatches
    to :func:`export_trace` with the verified step list.
    """
    if not isinstance(hmac_key, (bytes, bytearray)):
        raise ExportError(
            f"hmac_key must be bytes, got {type(hmac_key).__name__}"
        )
    trace = verify_trace(sb_path, bytes(hmac_key))
    return export_trace(target_format, trace.steps, output_path)
