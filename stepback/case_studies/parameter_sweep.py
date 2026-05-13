"""Case study: large parameter sweeps over a trace corpus.

This case study models the workflow of running systematic counterfactual
experiments over a corpus of recorded production traces — for example, an
ML team evaluating the effect of multiple different system prompt variants on
agent decisions before deploying a new prompt to production.

Scenario
--------
A code-review agent has been running in production for a week, generating 10
recorded traces per day.  The team wants to know:

* Does replacing the system prompt with a "strict" security-focused variant
  increase the fraction of traces that flag a security issue?
* Does replacing a tool output with a different data source change downstream
  decisions?

Rather than re-running the agent against live APIs (expensive, slow, and
non-deterministic), the team uses :func:`~stepback.sweep.sweep_traces` over
the recorded corpus to replay each trace under each substitution candidate
and measure **divergence** — how many steps differ between the baseline and
the counterfactual.

Production context
------------------
In a real deployment, the corpus would contain thousands of traces.  Here
we use a small synthetic corpus (5 traces, 8 steps each) for speed.  The
result's ``diverged_fraction`` and ``mean_dirty_count`` metrics generalise
directly from the small corpus to large ones: a ``diverged_fraction`` of 0.6
means 60% of production traces would show at least one changed decision under
the candidate substitution.

Example::

    from stepback.case_studies.parameter_sweep import run_parameter_sweep
    result = run_parameter_sweep(n_traces=5, n_steps=8)
    print(result.summary_line())
    # → "parameter_sweep: 5 traces × 2 candidates | diverged=3/10 mean_dirty=2.1"
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..recorder import RecorderKey, record
from ..replay import Executor, replay
from ..substitutions import (
    ModelSubstitution,
    PromptSubstitution,
    SubstitutionSet,
    ToolOutputSubstitution,
)
from ..sweep import SweepReport, sweep_traces


# ---------------------------------------------------------------------------
# Deterministic fakes
# ---------------------------------------------------------------------------

def _llm(model: str, messages: list) -> dict:
    blob = model + "|" + "\n".join(
        f"{m.get('role', '')}:{m.get('content', '')}" for m in messages
    )
    digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
    text = f"review-{digest}"
    return {
        "id": f"chatcmpl-{digest}",
        "model": model,
        "choices": [
            {"index": 0, "finish_reason": "stop",
             "message": {"role": "assistant", "content": text}},
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": len(text), "total_tokens": 20 + len(text)},
    }


def _tool(name: str, args: dict) -> Any:
    blob = name + "|" + repr(sorted(args.items()))
    digest = hashlib.sha256(blob.encode()).hexdigest()[:12]
    if name == "static_analysis":
        # Deterministic analysis output keyed on args
        return {"findings": [{"severity": "low", "id": digest[:4]}]}
    return {"result": digest}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ParameterSweepResult:
    """Aggregate parameter-sweep result.

    Attributes
    ----------
    n_traces : int
        Number of traces in the corpus.
    n_steps_per_trace : int
        Target steps per trace.
    n_candidates : int
        Number of substitution candidates swept.
    candidate_labels : list of str
        Human-readable names for each candidate.
    reports : list of SweepReport
        One :class:`~stepback.sweep.SweepReport` per candidate.
    total_traces_diverged : int
        Total count of (trace, candidate) pairs where at least one step
        diverged.
    diverged_fraction : float
        ``total_traces_diverged / (n_traces × n_candidates)``.
    mean_dirty_count : float
        Mean dirty step count per trace across all candidates and traces.
    dirty_count_per_candidate : dict
        ``{label: mean_dirty}`` broken down by candidate.
    """

    n_traces: int
    n_steps_per_trace: int
    n_candidates: int
    candidate_labels: List[str]
    reports: List[SweepReport] = field(default_factory=list)
    total_traces_diverged: int = 0
    diverged_fraction: float = 0.0
    mean_dirty_count: float = 0.0
    dirty_count_per_candidate: Dict[str, float] = field(default_factory=dict)

    def summary_line(self) -> str:
        """One-line human-readable summary."""
        return (
            f"parameter_sweep: {self.n_traces} traces × {self.n_candidates} candidates | "
            f"diverged={self.total_traces_diverged}/{self.n_traces * self.n_candidates} "
            f"mean_dirty={self.mean_dirty_count:.1f}"
        )

    def to_json(self) -> str:
        """Serialise to a JSON string (without the full SweepReport objects)."""
        return json.dumps(
            {
                "n_traces": self.n_traces,
                "n_steps_per_trace": self.n_steps_per_trace,
                "n_candidates": self.n_candidates,
                "candidate_labels": self.candidate_labels,
                "total_traces_diverged": self.total_traces_diverged,
                "diverged_fraction": self.diverged_fraction,
                "mean_dirty_count": self.mean_dirty_count,
                "dirty_count_per_candidate": self.dirty_count_per_candidate,
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Trace builder
# ---------------------------------------------------------------------------

def _build_corpus(base_dir: str, n_traces: int, n_steps: int, seed: int) -> List[str]:
    """Build a corpus of synthetic traces; return list of paths."""
    paths = []
    rng = random.Random(seed)
    for i in range(n_traces):
        path = os.path.join(base_dir, f"corpus-{i:04d}.sb")
        key = RecorderKey.fresh()
        convo = [
            {"role": "system", "content": "code-review-agent v1"},
            {"role": "user", "content": f"Review PR #{rng.randint(100, 999)}"},
        ]
        with record(path, key=key) as rec:
            steps_written = 0
            while steps_written < n_steps:
                if steps_written < n_steps - 1 and rng.random() < 0.4:
                    s = rec.tool_call(
                        "static_analysis",
                        {"file": f"src/module-{rng.randint(0, 9)}.py"},
                        executor=_tool,
                    )
                    steps_written += 1
                if steps_written < n_steps:
                    s = rec.llm_call("gpt-4o-2024-11-20", convo, executor=_llm)
                    convo.append({"role": "assistant", "content": s["outputs"].get("choices", [{}])[0].get("message", {}).get("content", "")})
                    steps_written += 1
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Substitution candidate definitions
# ---------------------------------------------------------------------------

def _make_candidates(trace_paths: List[str]) -> List[tuple]:
    """Build two substitution candidate configs: prompt swap and tool-output swap.

    Returns a list of ``(label, substitutions_list)`` pairs.
    """
    candidates = []

    # Candidate 1: system-prompt swap to a "strict" security-focused variant
    # We substitute the first llm_call step in every trace.
    strict_system = [
        {"role": "system", "content": "code-review-agent v1 [STRICT-SECURITY-MODE]"},
        {"role": "user", "content": "Review this PR with maximum security scrutiny."},
    ]
    prompt_subs = []
    for path in trace_paths:
        t = replay(path)
        first_llm = next(
            (s for s in t.recorded_steps if s.get("step_kind") == "llm_call"),
            None,
        )
        if first_llm:
            prompt_subs.append(
                PromptSubstitution(
                    at_step=first_llm["step_id"],
                    new_messages=strict_system,
                )
            )
    candidates.append(("strict_system_prompt", prompt_subs))

    # Candidate 2: tool-output swap — inject a high-severity finding
    tool_subs = []
    for path in trace_paths:
        t = replay(path)
        first_tool = next(
            (s for s in t.recorded_steps if s.get("step_kind") == "tool_call"),
            None,
        )
        if first_tool:
            tool_subs.append(
                ToolOutputSubstitution(
                    at_step=first_tool["step_id"],
                    fake_response={"findings": [{"severity": "critical", "id": "INJECTED-001"}]},
                )
            )
    candidates.append(("critical_finding_injection", tool_subs))

    return candidates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_parameter_sweep(
    n_traces: int = 5,
    n_steps: int = 8,
    seed: int = 0,
    output_dir: Optional[str] = None,
) -> ParameterSweepResult:
    """Sweep two substitution candidates over a synthetic trace corpus.

    Parameters
    ----------
    n_traces : int
        Number of traces in the corpus.
    n_steps : int
        Target steps per trace.
    seed : int
        RNG seed.
    output_dir : str, optional
        Directory for files.  Cleaned up when ``None``.

    Raises
    ------
    ValueError
        If ``n_traces < 1`` or ``n_steps < 2``.
    """
    if n_traces < 1:
        raise ValueError(f"n_traces must be >= 1, got {n_traces}")
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2, got {n_steps}")

    cleanup = output_dir is None
    tmpdir = tempfile.TemporaryDirectory(prefix="sb-sweep-") if cleanup else None
    base_dir = tmpdir.name if tmpdir else output_dir

    try:
        trace_paths = _build_corpus(base_dir, n_traces, n_steps, seed)
        candidates = _make_candidates(trace_paths)

        executor_factory = lambda: Executor(llm=_llm, tool=_tool, fallback_recorded=True)

        reports: List[SweepReport] = []
        labels: List[str] = []
        total_diverged = 0
        all_dirty: List[float] = []
        dirty_per_cand: Dict[str, float] = {}

        for label, subs in candidates:
            labels.append(label)
            report = sweep_traces(
                trace_paths,
                subs,
                executor_factory=executor_factory,
                on_error="record",
            )
            reports.append(report)
            total_diverged += report.n_traces_diverged
            cand_dirty = [r.base_dirty_count + r.cf_dirty_count for r in report.results]
            cand_mean = statistics.mean(cand_dirty) if cand_dirty else 0.0
            dirty_per_cand[label] = cand_mean
            all_dirty.extend(cand_dirty)

        mean_dirty = statistics.mean(all_dirty) if all_dirty else 0.0
        total_pairs = n_traces * len(candidates)

        return ParameterSweepResult(
            n_traces=n_traces,
            n_steps_per_trace=n_steps,
            n_candidates=len(candidates),
            candidate_labels=labels,
            reports=reports,
            total_traces_diverged=total_diverged,
            diverged_fraction=total_diverged / total_pairs if total_pairs > 0 else 0.0,
            mean_dirty_count=mean_dirty,
            dirty_count_per_candidate=dirty_per_cand,
        )
    finally:
        if tmpdir:
            tmpdir.cleanup()
