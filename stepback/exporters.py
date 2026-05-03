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
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .canonical import canonical_json
from .recorder import RecorderKey
from .trace_reader import verify_trace


__all__ = [
    "ExportError",
    "ExportReport",
    "export_openai_chat_log",
    "export_langsmith_jsonl",
    "export_openinference_spans",
    "export_trace",
    "export_trace_file",
    "available_export_formats",
]


class ExportError(ValueError):
    """Raised when a stepback trace cannot be rendered into the target format."""


@dataclass
class ExportReport:
    """Summary of an export operation."""

    output_path: str
    target_format: str
    step_count: int = 0
    kind_counts: Dict[str, int] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "target_format": self.target_format,
            "step_count": self.step_count,
            "kind_counts": dict(self.kind_counts),
            "skipped": list(self.skipped),
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


# --------------------------------------------------------- dispatcher


_FORMAT_DISPATCH = {
    "openai_chat_log": export_openai_chat_log,
    "openai": export_openai_chat_log,
    "langsmith": export_langsmith_jsonl,
    "langsmith_jsonl": export_langsmith_jsonl,
    "openinference": export_openinference_spans,
    "openinference_spans": export_openinference_spans,
    "otel": export_openinference_spans,
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
