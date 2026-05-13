"""stepback ↔ ragdoctor integration.

Records ragdoctor diagnostic RAG runs as ``.sb`` traces, enabling stepback's
replay, substitution, and bisect machinery to be applied to RAG pipeline
quality regressions.

Two entry points:

* :class:`RagDoctorShim` — wrap a ``RagPipeline`` (and optionally a
  ``Doctor``) so that every ``.query()`` and ``.diagnose()`` call is
  auto-recorded into a stepback ``Recorder``.

* :func:`import_ragdoctor_trace` — convert a ragdoctor ``Trace`` object
  (from ``ragdoctor.tracing``) into a ``.sb`` file.  ``retrieve`` /
  ``rerank`` spans become ``tool_call`` steps; ``generate`` / ``answer``
  spans become ``llm_call`` steps; other spans become ``router`` steps.

Replay-time utility
-------------------
Once a RAG diagnostic run is captured as a ``.sb`` trace:

* **Substitution** — swap the retrieval query or prompt and use the
  dirty-set engine to identify which downstream answer steps need
  re-execution.
* **Bisect** — identify which specific retrieval / generation step caused
  a quality regression compared to a baseline trace.
* **Sweep** — run the same query against a corpus of traces to find
  systematic retrieval failures.
* **Minimize** — reduce a failing trace to the minimal set of steps that
  reproduce a ``Doctor`` finding.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from ..canonical import hash_obj, sha256_hex, canonical_json
from ..importers import ImportReport, _emit_step, _open_writer, RecorderKey
from ..recorder import Recorder


__all__ = [
    "RagDoctorShim",
    "import_ragdoctor_trace",
    "record_rag_query",
]

_ZERO_NONDET = sha256_hex(canonical_json({}))

# Span names used by ragdoctor's Tracer that map to LLM-style steps.
_LLM_SPAN_NAMES = frozenset({"generate", "answer", "llm", "generation", "rerank_llm"})
# Span names that map to tool-call steps (retrieval + reranking).
_TOOL_SPAN_NAMES = frozenset({"retrieve", "retrieval", "rerank", "reranker", "embed", "expand"})


# ======================================================================
# RagDoctorShim
# ======================================================================


class RagDoctorShim:
    """Wrap a ragdoctor ``RagPipeline`` to auto-record into a stepback ``Recorder``.

    Usage::

        from ragdoctor import RagPipeline, Doctor
        from stepback import record
        from stepback.integrations.ragdoctor import RagDoctorShim

        pipe = RagPipeline.from_paths(["docs/"])
        with record("rag_trace.sb") as rec:
            shim = RagDoctorShim(pipe, rec)
            hits = shim.query("what is dirty-set propagation?", k=5)
            report = shim.diagnose(["what is dirty-set propagation?"])

    Args:
        pipeline: A ``ragdoctor.RagPipeline`` instance (or any duck-typed
            object with a ``.query(q, k)`` method returning a list of
            hit-like objects with ``.chunk.text``, ``.score``, and
            ``.chunk.doc_id``).
        recorder: An active :class:`~stepback.recorder.Recorder`.
        doctor: Optional ``ragdoctor.Doctor`` instance.  When ``None``
            and ``diagnose()`` is called, one is instantiated on the fly
            using ``Doctor(pipeline)``.
    """

    def __init__(
        self,
        pipeline: Any,
        recorder: Recorder,
        *,
        doctor: Optional[Any] = None,
    ) -> None:
        self._pipeline = pipeline
        self._recorder = recorder
        self._doctor = doctor

    def query(self, q: str, *, k: int = 5) -> List[Any]:
        """Query the pipeline and record a ``tool_call`` (retrieval) step.

        Args:
            q: Query string.
            k: Number of hits to retrieve.

        Returns:
            The raw list of ``Hit`` objects returned by the pipeline.
        """
        hits = self._pipeline.query(q, k=k)

        def _retrieval_exec(_n: str, _a: Dict) -> Dict[str, Any]:
            return {
                "n_hits": len(hits),
                "hits": _normalize_hits(hits),
            }

        self._recorder.tool_call(
            "rag_retrieval",
            {"query": q, "k": k},
            executor=_retrieval_exec,
        )
        return hits

    def answer(self, q: str, *, k: int = 5) -> Any:
        """Query + answer and record both retrieval and generation steps.

        Only available when the pipeline exposes an ``.answer()`` method.
        Falls back to :meth:`query` (recording retrieval only) when the
        pipeline has no answerer.

        Args:
            q: Query string.
            k: Number of context hits.

        Returns:
            The raw answer returned by the pipeline.
        """
        if not hasattr(self._pipeline, "answer"):
            return self.query(q, k=k)

        hits = self._pipeline.query(q, k=k)

        def _retr_exec(_n: str, _a: Dict) -> Dict[str, Any]:
            return {"n_hits": len(hits), "hits": _normalize_hits(hits)}

        self._recorder.tool_call(
            "rag_retrieval",
            {"query": q, "k": k},
            executor=_retr_exec,
        )

        answer_obj = self._pipeline.answer(q, k=k)

        def _gen_exec(model: str, messages: List[Dict]) -> Dict[str, Any]:
            return {
                "choices": [{"message": {"content": _safe_text(answer_obj)}, "finish_reason": "stop"}],
                "usage": {},
            }

        self._recorder.llm_call(
            "ragdoctor_answerer",
            [{"role": "user", "content": q}],
            executor=_gen_exec,
            temperature=0.0,
            seed=None,
        )
        return answer_obj

    def diagnose(
        self,
        queries: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run ``Doctor.diagnose()`` and record a ``tool_call`` (diagnostic) step.

        Args:
            queries: Optional list of query strings to pass to ``diagnose()``.
            **kwargs: Additional keyword arguments forwarded to ``diagnose()``.

        Returns:
            The raw :class:`ragdoctor.doctor.DiagnosticReport` object.
        """
        doctor = self._doctor
        if doctor is None:
            try:
                from ragdoctor.doctor import Doctor
            except ImportError:
                # Duck-typing fallback: look for a Doctor class on the pipeline.
                Doctor = getattr(self._pipeline, "__class__", None)
            doctor = Doctor(self._pipeline)

        report = doctor.diagnose(queries, **kwargs)

        def _diagnose_exec(_n: str, _a: Dict) -> Dict[str, Any]:
            return _normalize_diagnostic_report(report)

        self._recorder.tool_call(
            "rag_diagnose",
            {"queries": list(queries or [])},
            executor=_diagnose_exec,
        )
        return report


# ======================================================================
# record_rag_query  (functional form)
# ======================================================================


def record_rag_query(
    pipeline: Any,
    query: str,
    recorder: Recorder,
    *,
    k: int = 5,
) -> List[Any]:
    """Record a single RAG query (retrieve) as a ``tool_call`` step.

    Convenience function equivalent to ``RagDoctorShim(pipeline, recorder).query(query, k=k)``.

    Args:
        pipeline: A ``ragdoctor.RagPipeline`` or duck-typed equivalent.
        query: The query string.
        recorder: An active :class:`~stepback.recorder.Recorder`.
        k: Number of hits to retrieve.

    Returns:
        The raw list of ``Hit`` objects.
    """
    return RagDoctorShim(pipeline, recorder).query(query, k=k)


# ======================================================================
# import_ragdoctor_trace
# ======================================================================


def import_ragdoctor_trace(
    trace: Any,
    output_path: str,
    *,
    key: Optional[RecorderKey] = None,
    compression: bool = False,
) -> ImportReport:
    """Convert a ragdoctor ``Trace`` object into a ``.sb`` trace file.

    Span-to-step mapping:

    * ``retrieve`` / ``retrieval`` / ``rerank`` / ``embed`` / ``expand`` spans
      → ``tool_call`` steps.
    * ``generate`` / ``answer`` / ``llm`` / ``generation`` spans
      → ``llm_call`` steps.
    * All other spans (e.g. ``pipeline``, ``query``) → ``router`` steps.

    Parent/child span relationships are preserved via stepback's
    ``parent_step_id`` linkage.

    Args:
        trace: A ``ragdoctor.tracing.Trace`` object (or any duck-typed
            object with a ``trace_id`` string and a ``spans`` list of
            objects with ``name``, ``span_id``, ``parent_id``,
            ``start_ts``, ``end_ts``, ``status``, ``error``, and ``attrs``).
        output_path: Destination ``.sb`` file path.
        key: Optional :class:`~stepback.importers.RecorderKey`.
        compression: Write compressed frames (default ``False``).

    Returns:
        An :class:`~stepback.importers.ImportReport`.
    """
    trace_id = str(getattr(trace, "trace_id", "unknown"))
    spans: List[Any] = list(getattr(trace, "spans", []))

    writer, key = _open_writer(output_path, key=key, compression=compression)
    report = ImportReport(output_path=output_path, source_format="ragdoctor_trace")

    # Map span_id → step_id for parent chain.
    span_id_to_step_id: Dict[str, str] = {}
    step_counter = 0

    try:
        for span in spans:
            step_counter += 1
            sid = f"step:{step_counter}"

            span_name = str(getattr(span, "name", "span"))
            span_id = str(getattr(span, "span_id", sid))
            parent_id = getattr(span, "parent_id", None)
            start_ts = float(getattr(span, "start_ts", 0.0))
            end_ts = float(getattr(span, "end_ts", 0.0))
            status = str(getattr(span, "status", "ok"))
            error = getattr(span, "error", None)
            attrs = dict(getattr(span, "attrs", {}) or {})

            parent_step_id: Optional[str] = None
            if parent_id:
                parent_step_id = span_id_to_step_id.get(str(parent_id))

            span_id_to_step_id[span_id] = sid

            wallclock_ns = int(start_ts * 1_000_000_000) if start_ts else time.time_ns()
            duration_ms = max(0.0, (end_ts - start_ts) * 1000.0) if end_ts else 0.0

            lower_name = span_name.lower()
            if any(lower_name.startswith(tok) or lower_name == tok for tok in _TOOL_SPAN_NAMES):
                step_kind = "tool_call"
            elif any(lower_name.startswith(tok) or lower_name == tok for tok in _LLM_SPAN_NAMES):
                step_kind = "llm_call"
            else:
                step_kind = "router"

            # Build inputs / outputs from span attrs + status.
            inputs: Dict[str, Any] = {
                "kind": step_kind,
                "name": span_name,
                "rd_trace_id": trace_id,
                "rd_span_id": span_id,
                "rd_duration_ms": duration_ms,
            }
            inputs.update({f"rd_{k}": _safe_json(v) for k, v in attrs.items()})

            outputs: Dict[str, Any] = {
                "status": status,
            }
            if error:
                outputs["error"] = str(error)

            if step_kind == "tool_call":
                inputs["arguments"] = {}
                outputs["result"] = outputs.pop("status")
                outputs["rd_ok"] = status == "ok"
            elif step_kind == "llm_call":
                inputs["model"] = span_name
                inputs["messages"] = []
                inputs["temperature"] = 0.0
                inputs["seed"] = None
                outputs = {
                    "choices": [{"message": {"content": str(attrs.get("response", ""))}, "finish_reason": "stop"}],
                    "usage": {},
                    "rd_status": status,
                }
                if error:
                    outputs["rd_error"] = str(error)
            else:
                inputs["options"] = [status]
                outputs["choice"] = status

            _emit_step(
                writer,
                step_id=sid,
                step_kind=step_kind,
                name=span_name,
                parent_step_id=parent_step_id,
                inputs=inputs,
                outputs=outputs,
                wallclock_ns=wallclock_ns,
            )
            report.kind_counts[step_kind] = report.kind_counts.get(step_kind, 0) + 1
            report.step_count += 1

    finally:
        writer.close()

    return report


# ======================================================================
# Internal helpers
# ======================================================================


def _normalize_hits(hits: List[Any]) -> List[Dict[str, Any]]:
    """Convert a list of ragdoctor ``Hit`` objects to JSON-safe dicts."""
    out = []
    for h in hits:
        try:
            chunk = getattr(h, "chunk", None)
            out.append({
                "score": float(getattr(h, "score", 0.0)),
                "text_preview": str(getattr(chunk, "text", ""))[:200] if chunk else "",
                "doc_id": str(getattr(chunk, "doc_id", "")) if chunk else "",
            })
        except Exception:
            out.append({"score": 0.0, "text_preview": str(h)[:200], "doc_id": ""})
    return out


def _normalize_diagnostic_report(report: Any) -> Dict[str, Any]:
    """Convert a ragdoctor ``DiagnosticReport`` to a JSON-safe dict."""
    findings = []
    try:
        for f in (getattr(report, "findings", None) or []):
            findings.append({
                "code": str(getattr(f, "code", "")),
                "severity": str(getattr(f, "severity", "info")),
                "message": str(getattr(f, "message", "")),
            })
    except Exception:
        pass

    metrics = {}
    try:
        raw = getattr(report, "metrics", {}) or {}
        for k, v in raw.items():
            try:
                metrics[str(k)] = float(v) if not isinstance(v, (str, bool, type(None))) else v
            except (TypeError, ValueError):
                metrics[str(k)] = str(v)
    except Exception:
        pass

    errors = sum(1 for f in findings if f.get("severity") == "error")
    warnings = sum(1 for f in findings if f.get("severity") == "warn")
    return {
        "n_findings": len(findings),
        "n_errors": errors,
        "n_warnings": warnings,
        "findings": findings,
        "metrics": metrics,
    }


def _safe_text(obj: Any) -> str:
    """Return a string representation of an answer object."""
    if isinstance(obj, str):
        return obj
    if hasattr(obj, "text"):
        return str(obj.text)
    if hasattr(obj, "answer"):
        return str(obj.answer)
    return str(obj)


def _safe_json(obj: Any) -> Any:
    """Recursively coerce to JSON-serializable primitives."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    # numpy scalars etc.
    try:
        return float(obj)
    except (TypeError, ValueError):
        pass
    return str(obj)
