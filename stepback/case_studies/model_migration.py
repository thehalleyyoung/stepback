"""Case study: LLM model migration with dirty-set analysis.

This case study models the operational workflow for safely migrating an AI
agent from one LLM model to a cheaper or faster successor — for example,
replacing ``gpt-4o-2024-11-20`` with ``gpt-4o-mini-2024-07-18`` in a
production pipeline.

Scenario
--------
A 10-step research agent has been running in production with GPT-4o.  The
ML team wants to evaluate GPT-4o-mini as a cost-reduction candidate without
re-running the full agent against live APIs.  They:

1. Load a representative production trace recorded under GPT-4o.
2. Apply a :class:`~stepback.substitutions.ModelSubstitution` on every
   ``llm_call`` step to swap to GPT-4o-mini.
3. Replay forward with a GPT-4o-mini executor.
4. Measure the **dirty-set fraction** (all LLM steps become dirty after a
   model swap) and the **cost reduction factor** (token rates differ between
   models).
5. Optionally sweep across multiple traces (a corpus) to compute aggregate
   statistics.

Interpreting the results
------------------------
* ``dirty_count`` should equal the number of ``llm_call`` steps in the trace:
  all LLM steps are dirty because their model input changed.
* ``cost_reduction_factor`` captures the *average* cost ratio between models,
  computed from the step replay pricing information.  A factor > 1 means the
  new model is cheaper per call.
* ``replay_fidelity_rate`` is always 1.0 here because the two fake LLMs
  produce different outputs — this metric only makes sense with real SDKs.

Example::

    from stepback.case_studies.model_migration import run_model_migration
    result = run_model_migration(n_traces=5, n_steps=10)
    print(result.summary_line())
    # → "model_migration: 5 traces a=gpt-4o-2024-11-20 b=gpt-4o-mini-2024-07-18 | dirty_frac=0.50 cost_red=3.0x"
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import tempfile
from dataclasses import dataclass
from typing import List, Optional

from ..recorder import RecorderKey, record
from ..replay import Executor, replay
from ..substitutions import ModelSubstitution, SubstitutionSet


# ---------------------------------------------------------------------------
# Known catalog model IDs (exist in stepback.pricing.PRICE_LIST)
# ---------------------------------------------------------------------------

_MODEL_A = "gpt-4o-2024-11-20"
_MODEL_B = "gpt-4o-mini-2024-07-18"


# ---------------------------------------------------------------------------
# Deterministic fake executors for both models
# ---------------------------------------------------------------------------

def _make_llm(model_tag: str):
    """Return a fake LLM executor that produces different outputs per model_tag."""
    def _llm(model: str, messages: list) -> dict:
        blob = model_tag + "|" + model + "|" + "\n".join(
            f"{m.get('role', '')}:{m.get('content', '')}" for m in messages
        )
        digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
        text = f"{model_tag}-reply-{digest}"
        return {
            "id": f"chatcmpl-{digest}",
            "model": model,
            "choices": [
                {"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": text}},
            ],
            "usage": {
                "prompt_tokens": sum(len(m.get("content", "")) for m in messages),
                "completion_tokens": len(text),
                "total_tokens": sum(len(m.get("content", "")) for m in messages) + len(text),
            },
        }
    return _llm


_llm_a = _make_llm("model-a")
_llm_b = _make_llm("model-b")


def _tool(name: str, args: dict) -> dict:
    blob = name + "|" + repr(sorted(args.items()))
    digest = hashlib.sha256(blob.encode()).hexdigest()[:12]
    return {"tool": name, "result": digest}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelMigrationResult:
    """Aggregate migration analysis result.

    Attributes
    ----------
    n_traces : int
        Number of traces analysed.
    n_steps_per_trace : int
        Steps per trace.
    model_a : str
        Source model ID (the one being replaced).
    model_b : str
        Target model ID (the migration candidate).
    llm_steps_per_trace : int
        Number of ``llm_call`` steps per trace (all become dirty on swap).
    dirty_count_mean : float
        Mean dirty step count across traces.
    dirty_fraction_mean : float
        Mean dirty fraction (``dirty_count / n_steps``) across traces.
    cost_reduction_factor : float
        Ratio of model-A to model-B cost per token (from the stepback pricing
        catalog).  A factor > 1 means model B is cheaper.  ``None`` if neither
        model is in the catalog.
    total_cost_a_usd : float
        Aggregate recorded cost for model-A runs.
    total_cost_b_usd : float
        Aggregate replayed cost for model-B runs.
    """

    n_traces: int
    n_steps_per_trace: int
    model_a: str
    model_b: str
    llm_steps_per_trace: int
    dirty_count_mean: float
    dirty_fraction_mean: float
    cost_reduction_factor: Optional[float]
    total_cost_a_usd: float
    total_cost_b_usd: float

    def summary_line(self) -> str:
        """One-line human-readable summary."""
        cr = f"{self.cost_reduction_factor:.1f}x" if self.cost_reduction_factor is not None else "N/A"
        return (
            f"model_migration: {self.n_traces} traces "
            f"a={self.model_a} b={self.model_b} | "
            f"dirty_frac={self.dirty_fraction_mean:.2f} "
            f"cost_red={cr}"
        )

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(
            {
                "n_traces": self.n_traces,
                "n_steps_per_trace": self.n_steps_per_trace,
                "model_a": self.model_a,
                "model_b": self.model_b,
                "llm_steps_per_trace": self.llm_steps_per_trace,
                "dirty_count_mean": self.dirty_count_mean,
                "dirty_fraction_mean": self.dirty_fraction_mean,
                "cost_reduction_factor": self.cost_reduction_factor,
                "total_cost_a_usd": self.total_cost_a_usd,
                "total_cost_b_usd": self.total_cost_b_usd,
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Trace builder
# ---------------------------------------------------------------------------

def _build_trace(path: str, n_steps: int, seed: int) -> RecorderKey:
    """Record a mixed LLM+tool trace under model A; return the recorder key."""
    rng = random.Random(seed)
    key = RecorderKey.fresh()
    convo: List[dict] = [
        {"role": "system", "content": "migration-test-agent v1"},
        {"role": "user", "content": f"task-{rng.randint(0, 2**32)}"},
    ]
    with record(path, key=key) as rec:
        steps_written = 0
        while steps_written < n_steps:
            if steps_written < n_steps - 1 and rng.random() < 0.3:
                # Tool step (will NOT become dirty on model swap)
                rec.tool_call(
                    "fetch_data",
                    {"key": f"k-{rng.randint(0, 999)}"},
                    executor=_tool,
                )
                steps_written += 1
            if steps_written < n_steps:
                s = rec.llm_call(_MODEL_A, convo, executor=_llm_a)
                convo.append({"role": "assistant", "content": s["outputs"].get("choices", [{}])[0].get("message", {}).get("content", "")})
                steps_written += 1
    return key


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_model_migration(
    n_traces: int = 5,
    n_steps: int = 10,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> ModelMigrationResult:
    """Analyse the impact of a model swap across multiple traces.

    Parameters
    ----------
    n_traces : int
        Number of traces to analyse.
    n_steps : int
        Target steps per trace.
    seed : int
        RNG seed for trace content.
    output_dir : str, optional
        Directory for ``.sb`` files.  Cleaned up when ``None``.

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
    tmpdir = tempfile.TemporaryDirectory(prefix="sb-migration-") if cleanup else None
    base_dir = tmpdir.name if tmpdir else output_dir

    # Pricing catalog lookup (best-effort; may not contain fake model IDs)
    try:
        from ..pricing import PRICE_LIST
        rate_a = PRICE_LIST.get(_MODEL_A)
        rate_b = PRICE_LIST.get(_MODEL_B)
        cost_reduction_factor: Optional[float] = (
            (rate_a[0] / rate_b[0]) if rate_a and rate_b and rate_b[0] > 0 else None
        )
    except Exception:  # noqa: BLE001
        cost_reduction_factor = None

    dirty_counts: List[float] = []
    dirty_fracs: List[float] = []
    total_cost_a = 0.0
    total_cost_b = 0.0
    llm_step_counts: List[int] = []

    try:
        for i in range(n_traces):
            path = os.path.join(base_dir, f"trace-{i:04d}.sb")
            trace_key = _build_trace(path, n_steps, seed=seed + i)

            trace = replay(path)
            n_trace_steps = len(trace.recorded_steps)

            # Collect recorded (model-A) cost
            total_cost_a += sum(
                s.get("cost_usd", 0.0) for s in trace.recorded_steps
            )

            # Count LLM steps in this trace
            n_llm = sum(
                1 for s in trace.recorded_steps
                if s.get("step_kind") == "llm_call"
            )
            llm_step_counts.append(n_llm)

            # Apply ModelSubstitution on every llm_call step
            subs = SubstitutionSet()
            for step in trace.recorded_steps:
                if step.get("step_kind") == "llm_call":
                    subs.add(ModelSubstitution(at_step=step["step_id"], new_model_id=_MODEL_B))

            executor = Executor(llm=_llm_b, tool=_tool, fallback_recorded=True)
            result = trace.run_replay(subs, executor)
            dirty_counts.append(result.dirty_count)
            dirty_fracs.append(result.dirty_count / n_trace_steps if n_trace_steps > 0 else 0.0)
            total_cost_b += result.total_cost_usd
    finally:
        if tmpdir:
            tmpdir.cleanup()

    return ModelMigrationResult(
        n_traces=n_traces,
        n_steps_per_trace=n_steps,
        model_a=_MODEL_A,
        model_b=_MODEL_B,
        llm_steps_per_trace=int(statistics.mean(llm_step_counts)) if llm_step_counts else 0,
        dirty_count_mean=statistics.mean(dirty_counts) if dirty_counts else 0.0,
        dirty_fraction_mean=statistics.mean(dirty_fracs) if dirty_fracs else 0.0,
        cost_reduction_factor=cost_reduction_factor,
        total_cost_a_usd=total_cost_a,
        total_cost_b_usd=total_cost_b,
    )
