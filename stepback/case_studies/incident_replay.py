"""Case study: incident replay and root-cause isolation.

This case study models a common production post-mortem workflow: an AI agent
produced an unexpected output in production, and the on-call engineer needs
to isolate the exact step that caused the deviation without re-running the
full agent against live APIs.

Scenario
--------
A 6-step research agent (system prompt → web search → summarise → policy
check → final answer → audit log) was recorded as usual.  In one trace the
policy-check tool returned a **false positive** ("policy_blocked=True") for
a routine request.  The on-call engineer:

1. Loads the production trace.
2. Injects a ``ToolOutputSubstitution`` at the policy-check step to simulate
   the *corrected* tool response.
3. Calls ``trace.bisect()`` to find the earliest step whose output differs
   between the original and the corrected replay.
4. Confirms the culprit step matches the injected substitution point.

This pattern is useful because:

* **Zero LLM calls** — the bisection is served entirely from the replay
  cache; no live API keys are needed.
* **Deterministic** — the synthetic trace uses a hash-based fake LLM so the
  "incident" is reproducible from the seed.
* **Auditable** — the substitution and its effect are recorded in the result
  for the post-mortem report.

Example::

    from stepback.case_studies.incident_replay import run_incident_replay
    result = run_incident_replay()
    print(result.summary_line())
    # → "incident_replay: culprit=step:4 (policy_check) bisect_probes=2 dirty_after_fix=3"
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..recorder import RecorderKey, record
from ..replay import Executor, replay
from ..substitutions import SubstitutionSet, ToolOutputSubstitution


# ---------------------------------------------------------------------------
# Deterministic fakes
# ---------------------------------------------------------------------------

_POLICY_FAULT_STEP = 4  # 1-indexed step number where the bad tool output is

def _llm(model: str, messages: list) -> dict:
    blob = model + "|" + "\n".join(
        f"{m.get('role', '')}:{m.get('content', '')}" for m in messages
    )
    digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
    text = f"answer-{digest}"
    return {
        "id": f"chatcmpl-{digest}",
        "model": model,
        "choices": [
            {"index": 0, "finish_reason": "stop",
             "message": {"role": "assistant", "content": text}},
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": len(text), "total_tokens": 10 + len(text)},
    }


def _tool(name: str, args: dict) -> Any:
    """Scripted tool — the policy check deliberately returns a false positive."""
    if name == "web_search":
        return {"results": [{"title": "Result A", "snippet": "Relevant content"}]}
    if name == "policy_check":
        # In the "buggy" recording the tool always returns blocked
        return {"policy_blocked": True, "reason": "false-positive-rule-42"}
    if name == "audit_log":
        return {"logged": True, "log_id": f"log-{hashlib.sha256(repr(args).encode()).hexdigest()[:8]}"}
    return {"result": "ok"}


def _tool_fixed(name: str, args: dict) -> Any:
    """Corrected tool — policy check returns the right answer."""
    if name == "policy_check":
        return {"policy_blocked": False, "reason": None}
    return _tool(name, args)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class IncidentReplayResult:
    """Root-cause isolation result.

    Attributes
    ----------
    trace_path : str
        Path to the recorded production trace.
    n_steps : int
        Total number of steps in the trace.
    injected_at_step : str
        The step id at which the corrective substitution was injected.
    expected_culprit_step : str
        The step id we expect bisection to identify (same as ``injected_at_step``
        since the fault is localised).
    culprit_step_id : str or None
        The step id returned by ``bisect()``; ``None`` if bisection found no
        culprit (i.e., the predicate was never triggered).
    culprit_found : bool
        ``True`` iff ``culprit_step_id == expected_culprit_step``.
    bisect_probes : int
        Number of steps examined during binary search.
    dirty_after_fix : int
        Number of steps that became dirty after the corrective substitution.
    original_policy_output : dict
        The raw tool output recorded in the original (buggy) trace.
    fixed_policy_output : dict
        The substituted (corrected) tool output used in the replay.
    """

    trace_path: str
    n_steps: int
    injected_at_step: str
    expected_culprit_step: str
    culprit_step_id: Optional[str]
    culprit_found: bool
    bisect_probes: int
    dirty_after_fix: int
    original_policy_output: Dict[str, Any]
    fixed_policy_output: Dict[str, Any]

    def summary_line(self) -> str:
        """One-line human-readable summary."""
        return (
            f"incident_replay: culprit={self.culprit_step_id} "
            f"(expected={self.expected_culprit_step}) "
            f"found={self.culprit_found} "
            f"bisect_probes={self.bisect_probes} "
            f"dirty_after_fix={self.dirty_after_fix}"
        )

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(
            {
                "trace_path": self.trace_path,
                "n_steps": self.n_steps,
                "injected_at_step": self.injected_at_step,
                "expected_culprit_step": self.expected_culprit_step,
                "culprit_step_id": self.culprit_step_id,
                "culprit_found": self.culprit_found,
                "bisect_probes": self.bisect_probes,
                "dirty_after_fix": self.dirty_after_fix,
                "original_policy_output": self.original_policy_output,
                "fixed_policy_output": self.fixed_policy_output,
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Trace builder
# ---------------------------------------------------------------------------

def _build_incident_trace(path: str, key: RecorderKey) -> List[str]:
    """Record a 6-step trace with a buggy policy check at step 4.

    Returns the list of step ids in order.
    """
    step_ids: List[str] = []
    convo: List[dict] = [
        {"role": "system", "content": "research-agent v1.0"},
        {"role": "user", "content": "Is widget-XJ23 compliant with policy v3?"},
    ]

    with record(path, key=key) as rec:
        # Step 1: initial LLM call
        s = rec.llm_call("gpt-4o-2024-11-20", convo, executor=_llm)
        step_ids.append(s["step_id"])
        convo.append({"role": "assistant", "content": s["outputs"].get("choices", [{}])[0].get("message", {}).get("content", "")})

        # Step 2: web search
        s = rec.tool_call("web_search", {"query": "widget-XJ23 compliance policy v3"}, executor=_tool)
        step_ids.append(s["step_id"])

        # Step 3: summarise results
        convo.append({"role": "user", "content": "Summarise the search results."})
        s = rec.llm_call("gpt-4o-2024-11-20", convo, executor=_llm)
        step_ids.append(s["step_id"])
        convo.append({"role": "assistant", "content": s["outputs"].get("choices", [{}])[0].get("message", {}).get("content", "")})

        # Step 4: policy check (the buggy one)
        s = rec.tool_call(
            "policy_check",
            {"item_id": "widget-XJ23", "policy_version": "v3"},
            executor=_tool,
        )
        step_ids.append(s["step_id"])

        # Step 5: final answer
        convo.append({"role": "user", "content": "Based on the policy check, what is the compliance verdict?"})
        s = rec.llm_call("gpt-4o-2024-11-20", convo, executor=_llm)
        step_ids.append(s["step_id"])
        convo.append({"role": "assistant", "content": s["outputs"].get("choices", [{}])[0].get("message", {}).get("content", "")})

        # Step 6: audit log
        s = rec.tool_call(
            "audit_log",
            {"decision": "blocked", "item_id": "widget-XJ23"},
            executor=_tool,
        )
        step_ids.append(s["step_id"])

    return step_ids


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_incident_replay(output_dir: Optional[str] = None) -> IncidentReplayResult:
    """Demonstrate root-cause isolation on a synthetic incident trace.

    Records a 6-step trace where the policy-check tool returns a false
    positive, then replays it with a corrective substitution and uses
    ``bisect()`` to identify the offending step.

    Parameters
    ----------
    output_dir : str, optional
        Directory for the ``.sb`` file.  A ``TemporaryDirectory`` is used
        when ``None``; it is cleaned up before the function returns.
    """
    cleanup = output_dir is None
    tmpdir = tempfile.TemporaryDirectory(prefix="sb-incident-") if cleanup else None
    base_dir = tmpdir.name if tmpdir else output_dir

    key = RecorderKey.fresh()
    trace_path = os.path.join(base_dir, "incident.sb")

    try:
        step_ids = _build_incident_trace(trace_path, key)

        # The policy-check step is step_ids[3] (0-indexed; the 4th step).
        policy_step_id = step_ids[3]
        injected_step_id = policy_step_id

        # Fixed output for the substitution
        fixed_output = {"policy_blocked": False, "reason": None}
        original_output = {"policy_blocked": True, "reason": "false-positive-rule-42"}

        # Load the trace and inject the corrective substitution
        trace = replay(trace_path)
        sub = ToolOutputSubstitution(at_step=policy_step_id, fake_response=fixed_output)
        trace.substitute(sub)

        # Replay with the corrective executor (for dirty steps after the substitution)
        executor = Executor(llm=_llm, tool=_tool_fixed, fallback_recorded=True)
        result = trace.run_replay(trace.pending_subs, executor)
        dirty_count = result.dirty_count

        # Bisect: find the earliest step whose output changed under the substitution.
        # We define "output changed" by checking the dirty flag.
        bisect_trace = replay(trace_path)
        bisect_trace.substitute(
            ToolOutputSubstitution(at_step=policy_step_id, fake_response=fixed_output)
        )
        culprit_sv = bisect_trace.bisect(
            good=step_ids[0],
            bad=step_ids[-1],
            predicate=lambda sv: sv.dirty,
            executor=Executor(llm=_llm, tool=_tool_fixed, fallback_recorded=True),
        )
        bisect_probes = bisect_trace._last_bisect_probes  # type: ignore[attr-defined]
        culprit_step_id = culprit_sv.step_id if culprit_sv is not None else None

        return IncidentReplayResult(
            trace_path=trace_path if not cleanup else "<temp>",
            n_steps=len(step_ids),
            injected_at_step=injected_step_id,
            expected_culprit_step=policy_step_id,
            culprit_step_id=culprit_step_id,
            culprit_found=(culprit_step_id == policy_step_id),
            bisect_probes=bisect_probes,
            dirty_after_fix=dirty_count,
            original_policy_output=original_output,
            fixed_policy_output=fixed_output,
        )
    finally:
        if tmpdir:
            tmpdir.cleanup()
