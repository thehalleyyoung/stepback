"""Replay-divergence detector — README §"Repository layout" `replay/nondet.py`.

stepback's whole value proposition rests on **deterministic replay**:
"same canonical inputs → same hash → cache hit → no LLM re-execution".
But real LLM endpoints (OpenAI, Anthropic, Bedrock, vLLM, …) are not
fully deterministic. ``temperature=0`` reduces variance but doesn't
eliminate it; provider-side speculative decoding, model fleet
heterogeneity, and silent model upgrades all break the bytewise
guarantee that the per-step cache relies on.

This module gives users a principled answer to the question

    "If I re-run my agent right now against the same provider, how
    much of my recorded trace would still be reproducible — and where
    are the actual semantic divergences vs. cosmetic noise?"

It does that by re-executing every step in a recorded `.sb` trace
against a user-supplied :py:class:`~stepback.replay.Executor` and
comparing the new outputs to the recorded outputs through a
**structural classifier** (not a heuristic): two outputs are equal up
to a controlled set of *volatile fields* (provider response IDs,
``created`` / ``request_id`` timestamps, fingerprint strings) before
the structural comparison runs.

Classifications, in increasing severity:

* :py:data:`IDENTICAL` — bytewise equal after canonical JSON.
* :py:data:`EQUIVALENT` — equal after stripping volatile fields
  (``id``, ``created``, ``request_id``, ``response_id``,
  ``system_fingerprint``, ``timestamp`` …) and zeroing token-usage
  counts.  This is the "noise floor" — replay is still cache-safe in
  spirit; only provider bookkeeping differs.
* :py:data:`MINOR` — same top-level structure and same final
  assistant message text, but other fields differ
  (e.g. usage counts, finish reason).
* :py:data:`SEMANTIC` — same top-level structure but the final
  assistant message text or the tool result body changed.  This is
  the case stepback users care about: the model produced a different
  answer.
* :py:data:`STRUCTURAL` — different keys / types at the top level,
  or one side is non-dict.  The output shape itself changed —
  schema drift, model upgrade with new fields, or a tool returning
  a different envelope.

The detector is **not GOFAI**: each rule is a structural fact about
JSON shape or a hard-coded volatile-key list; nothing is doing
fuzzy text matching or hand-tuned scoring.  For the *quality* of an
LLM output (is the new answer worse?), use the agent CLI through
the existing :py:mod:`stepback.report` critic surface, not this
module.

Public surface (re-exported from :py:mod:`stepback`):

* :py:data:`IDENTICAL`, :py:data:`EQUIVALENT`, :py:data:`MINOR`,
  :py:data:`SEMANTIC`, :py:data:`STRUCTURAL` — class constants.
* :py:data:`SEVERITY` — ordered list (lowest → highest).
* :py:data:`SEVERITY_WEIGHT` — int weights for scoring.
* :py:data:`VOLATILE_KEYS` — provider-noise key set.
* :py:func:`compare_outputs` — classify a single (recorded, replayed)
  pair into a :py:class:`Divergence`.
* :py:func:`detect_divergences` — re-execute a whole trace and
  return a :py:class:`DivergenceReport`.
* :py:class:`Divergence`, :py:class:`DivergenceReport` —
  serialisable result types with ``to_json`` / Markdown rendering.

Usage::

    from stepback.divergence import detect_divergences
    from stepback.replay import Executor

    report = detect_divergences(
        "trace.sb",
        hmac_key=key.hmac_key,
        executor=Executor(llm=my_real_llm, tool=my_real_tool),
    )
    print(report.render_markdown())
    if report.severity_score > 5:
        sys.exit(2)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .canonical import canonical_json, hash_obj
from .replay import Executor, MissingExecutor
from .trace_reader import verify_trace

# ----------------------------------------------------------------- consts

IDENTICAL = "identical"
EQUIVALENT = "equivalent"
MINOR = "minor"
SEMANTIC = "semantic"
STRUCTURAL = "structural"

SEVERITY: Tuple[str, ...] = (IDENTICAL, EQUIVALENT, MINOR, SEMANTIC, STRUCTURAL)
SEVERITY_WEIGHT: Dict[str, int] = {
    IDENTICAL: 0,
    EQUIVALENT: 1,
    MINOR: 2,
    SEMANTIC: 4,
    STRUCTURAL: 5,
}

# Provider-side bookkeeping that is allowed to vary between runs without
# being treated as a real divergence. Keys are matched case-insensitively
# at any depth in the JSON tree.
VOLATILE_KEYS: Tuple[str, ...] = (
    "id",
    "created",
    "created_at",
    "request_id",
    "response_id",
    "x_request_id",
    "system_fingerprint",
    "fingerprint",
    "timestamp",
    "trace_id",
    "span_id",
)

# Token counts get zeroed (not removed) before structural compare so a
# different completion length doesn't cascade through every nested
# `usage` block.
USAGE_KEYS: Tuple[str, ...] = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_tokens",
    "input_tokens",
    "output_tokens",
)


# --------------------------------------------------------------- helpers


def _strip_volatile(obj: Any) -> Any:
    """Recursively drop volatile bookkeeping keys; zero token counters.

    Returns a NEW structure; the input is not mutated.
    """
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in VOLATILE_KEYS:
                continue
            if kl in USAGE_KEYS and isinstance(v, (int, float)):
                out[k] = 0
                continue
            out[k] = _strip_volatile(v)
        return out
    if isinstance(obj, list):
        return [_strip_volatile(x) for x in obj]
    return obj


def _assistant_text(outputs: Any) -> Optional[str]:
    """Extract the final assistant message text from an OpenAI/Anthropic-shaped output."""
    if not isinstance(outputs, dict):
        return None
    # OpenAI chat/completion shape.
    choices = outputs.get("choices")
    if isinstance(choices, list) and choices:
        last = choices[-1]
        if isinstance(last, dict):
            msg = last.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(last.get("text"), str):
                return last["text"]
    # Anthropic messages shape.
    content = outputs.get("content")
    if isinstance(content, list) and content:
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        if parts:
            return "".join(parts)
    if isinstance(content, str):
        return content
    # Tool-call envelope shape.
    if "result" in outputs:
        r = outputs["result"]
        if isinstance(r, str):
            return r
    return None


def _structural_skeleton(obj: Any, _depth: int = 0) -> Any:
    """A type/shape-only skeleton: dict→sorted key list, list→length, leaf→type name."""
    if _depth > 6:
        return "..."
    if isinstance(obj, dict):
        return {k: _structural_skeleton(obj[k], _depth + 1) for k in sorted(obj)}
    if isinstance(obj, list):
        if not obj:
            return []
        # Use the first element's skeleton as a representative.
        return [_structural_skeleton(obj[0], _depth + 1)]
    return type(obj).__name__


# -------------------------------------------------------------- dataclass


@dataclass
class Divergence:
    """One step's divergence classification."""

    step_id: str
    kind: str
    name: Optional[str]
    classification: str
    recorded_hash: str
    replayed_hash: str
    summary: str
    error: Optional[str] = None  # set if executor raised on this step

    @property
    def severity(self) -> int:
        return SEVERITY_WEIGHT.get(self.classification, 0)

    @property
    def is_divergent(self) -> bool:
        return self.classification != IDENTICAL

    def to_json(self) -> dict:
        d = {
            "step_id": self.step_id,
            "kind": self.kind,
            "name": self.name,
            "classification": self.classification,
            "severity": self.severity,
            "recorded_hash": self.recorded_hash,
            "replayed_hash": self.replayed_hash,
            "summary": self.summary,
        }
        if self.error is not None:
            d["error"] = self.error
        return d


@dataclass
class DivergenceReport:
    """Aggregate divergence over a whole trace re-execution."""

    trace_path: str
    step_count: int
    divergences: List[Divergence] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)  # step_ids without executor support

    @property
    def class_counts(self) -> Dict[str, int]:
        out = {c: 0 for c in SEVERITY}
        for d in self.divergences:
            out[d.classification] = out.get(d.classification, 0) + 1
        return out

    @property
    def divergent_count(self) -> int:
        return sum(1 for d in self.divergences if d.is_divergent)

    @property
    def severity_score(self) -> int:
        """Sum of per-step severity weights — 0 means perfect replay."""
        return sum(d.severity for d in self.divergences)

    @property
    def reproducibility_pct(self) -> float:
        """Percent of compared steps classified as IDENTICAL or EQUIVALENT."""
        if not self.divergences:
            return 100.0
        good = sum(1 for d in self.divergences if d.classification in (IDENTICAL, EQUIVALENT))
        return round(100.0 * good / len(self.divergences), 2)

    def to_json(self) -> dict:
        return {
            "trace_path": self.trace_path,
            "step_count": self.step_count,
            "compared_count": len(self.divergences),
            "divergent_count": self.divergent_count,
            "severity_score": self.severity_score,
            "reproducibility_pct": self.reproducibility_pct,
            "class_counts": self.class_counts,
            "skipped_step_ids": list(self.skipped),
            "divergences": [d.to_json() for d in self.divergences],
        }

    def render_markdown(self, max_rows: int = 50) -> str:
        lines: List[str] = []
        lines.append("# stepback divergence report")
        lines.append("")
        lines.append(f"trace: `{self.trace_path}`")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("| - | - |")
        lines.append(f"| step_count | {self.step_count} |")
        lines.append(f"| compared | {len(self.divergences)} |")
        lines.append(f"| divergent | {self.divergent_count} |")
        lines.append(f"| severity_score | {self.severity_score} |")
        lines.append(f"| reproducibility_pct | {self.reproducibility_pct} |")
        for c in SEVERITY:
            lines.append(f"| {c} | {self.class_counts.get(c, 0)} |")
        lines.append("")
        if self.divergences:
            lines.append("## per-step")
            lines.append("")
            lines.append("| step | kind | name | class | severity | summary |")
            lines.append("| - | - | - | - | - | - |")
            for d in self.divergences[:max_rows]:
                summary = d.summary.replace("|", "\\|")
                lines.append(
                    f"| {d.step_id} | {d.kind} | {d.name or ''} | {d.classification} "
                    f"| {d.severity} | {summary} |"
                )
            if len(self.divergences) > max_rows:
                lines.append("")
                lines.append(f"_... {len(self.divergences) - max_rows} more rows truncated_")
        if self.skipped:
            lines.append("")
            lines.append(f"_Skipped {len(self.skipped)} step(s) (no executor): "
                         f"{', '.join(self.skipped[:10])}"
                         f"{'…' if len(self.skipped) > 10 else ''}_")
        return "\n".join(lines) + "\n"


# ------------------------------------------------------------ classifier


def compare_outputs(recorded: Any, replayed: Any) -> Tuple[str, str]:
    """Classify ``(recorded, replayed)`` into one of the SEVERITY classes.

    Returns ``(classification, summary)`` where ``summary`` is a
    short human-readable explanation suitable for a report row.
    """
    if hash_obj(recorded) == hash_obj(replayed):
        return IDENTICAL, "bytewise-equal"

    sr = _strip_volatile(recorded)
    sp = _strip_volatile(replayed)
    if hash_obj(sr) == hash_obj(sp):
        return EQUIVALENT, "equal after stripping volatile fields"

    # From here on we know there's a real content difference.
    skel_r = _structural_skeleton(sr)
    skel_p = _structural_skeleton(sp)
    if hash_obj(skel_r) != hash_obj(skel_p):
        return STRUCTURAL, _structural_summary(sr, sp)

    text_r = _assistant_text(recorded)
    text_p = _assistant_text(replayed)
    if text_r is not None and text_p is not None and text_r == text_p:
        return MINOR, "same assistant text; metadata differs"
    if text_r is None and text_p is None:
        # No extractable assistant text — fall back to "semantic" since
        # structures match but content differs.
        return SEMANTIC, "no extractable assistant text; bodies differ"
    return SEMANTIC, _semantic_summary(text_r, text_p)


def _structural_summary(rec: Any, rep: Any) -> str:
    if not isinstance(rec, dict) or not isinstance(rep, dict):
        return f"top-level types differ: {type(rec).__name__} vs {type(rep).__name__}"
    rk = set(rec.keys())
    pk = set(rep.keys())
    only_r = sorted(rk - pk)
    only_p = sorted(pk - rk)
    parts = []
    if only_r:
        parts.append(f"recorded-only keys: {only_r[:5]}")
    if only_p:
        parts.append(f"replayed-only keys: {only_p[:5]}")
    if not parts:
        parts.append("nested type/shape mismatch")
    return "; ".join(parts)


def _semantic_summary(text_r: Optional[str], text_p: Optional[str]) -> str:
    def _trim(s: Optional[str]) -> str:
        if s is None:
            return "<none>"
        s = s.strip()
        return s if len(s) <= 60 else s[:57] + "..."
    return f"recorded={_trim(text_r)!r} → replayed={_trim(text_p)!r}"


# --------------------------------------------------------- whole-trace API


def detect_divergences(
    trace_path: str,
    hmac_key: bytes,
    executor: Executor,
    *,
    step_kinds: Iterable[str] = ("llm_call", "tool_call", "router"),
    max_steps: Optional[int] = None,
) -> DivergenceReport:
    """Re-execute every supported step in ``trace_path`` and classify divergences.

    Steps whose kind is not in ``step_kinds``, or for which the
    executor has no callback, are recorded in ``report.skipped``.

    The executor is called with the **recorded inputs**, so this is a
    pure replay-determinism probe — it does not propagate
    counterfactuals (use the substitution / replay-forward API for
    that).  This is exactly the question "if I re-ran the same
    prompts today, would I get the same outputs?".
    """
    trace = verify_trace(trace_path, hmac_key)
    steps = trace.steps
    if max_steps is not None:
        steps = steps[:max_steps]

    divergences: List[Divergence] = []
    skipped: List[str] = []
    for rec in steps:
        sid = rec["step_id"]
        kind = rec.get("step_kind") or rec.get("kind")
        name = rec.get("name")
        if kind not in step_kinds:
            skipped.append(sid)
            continue
        # Skip kinds the executor can't handle without aborting the sweep.
        if kind == "llm_call" and executor.llm is None:
            skipped.append(sid)
            continue
        if kind == "tool_call" and executor.tool is None:
            skipped.append(sid)
            continue
        if kind == "router" and executor.router is None:
            skipped.append(sid)
            continue
        recorded_outputs = rec.get("outputs")
        try:
            replayed = executor.execute(kind, rec.get("inputs", {}))
        except Exception as exc:  # executor failure is a real divergence signal
            divergences.append(Divergence(
                step_id=sid,
                kind=kind,
                name=name,
                classification=STRUCTURAL,
                recorded_hash=hash_obj(recorded_outputs),
                replayed_hash="<error>",
                summary=f"executor raised {type(exc).__name__}: {exc}",
                error=f"{type(exc).__name__}: {exc}",
            ))
            continue
        cls, summary = compare_outputs(recorded_outputs, replayed)
        divergences.append(Divergence(
            step_id=sid,
            kind=kind,
            name=name,
            classification=cls,
            recorded_hash=hash_obj(recorded_outputs),
            replayed_hash=hash_obj(replayed),
            summary=summary,
        ))
    return DivergenceReport(
        trace_path=trace_path,
        step_count=len(trace.steps),
        divergences=divergences,
        skipped=skipped,
    )


__all__ = [
    "IDENTICAL",
    "EQUIVALENT",
    "MINOR",
    "SEMANTIC",
    "STRUCTURAL",
    "SEVERITY",
    "SEVERITY_WEIGHT",
    "VOLATILE_KEYS",
    "USAGE_KEYS",
    "Divergence",
    "DivergenceReport",
    "compare_outputs",
    "detect_divergences",
]
