"""Empirical dirty-set distributions over multiple synthetic fixture corpora.

Step 66 of ``100_STEPS.md``: publish empirical dirty-set distributions over
synthetic fixtures, public benchmark corpora, and anonymized production
traces.

This module extends the replay-caching benchmark (which reports only a
single *mean* dirty-set size per configuration) with **full distributions**
— percentile tables and normalized histograms of ``dirty_count / step_count``
— across several representative trace corpora:

Corpora
-------
``linear_chain``
    Purely sequential ``llm_call`` / ``tool_call`` chains.  This is the
    **worst case** for dirty-set growth: every step has a
    ``context_from_parent`` edge, so a substitution at step *k* makes the
    entire suffix dirty.

``parallel_wide``
    A single fan-out block with *W* branches (W = 3..10), each containing
    two steps, followed by a join.  Substituting inside one branch leaves
    all other branches and the join clean, demonstrating **B1 / B3**
    clean-sibling preservation.

``mixed_synthetic``
    The general :class:`~stepback.bench.replay_caching.SyntheticTrace`
    builder (mix of linear and occasional parallel blocks).  This is the
    canonical reference corpus for the existing benchmark tables in
    ``bench-results/README.md``.

``agent_fixture``
    The deterministic 12-step customer-payments agent from
    :mod:`stepback.testing` (:func:`~stepback.testing.run_recorded_agent`).
    Represents a realistic recorded LLM + tool-call trace used in end-to-end
    tests.

Substitution coverage
---------------------
Each corpus is exercised with three substitution **positions**:
``random`` (uniform-random), ``early`` (first 25 % of steps), and ``late``
(final 25 % of steps); and two substitution **types**: ``PromptSubstitution``
(for ``llm_call`` targets) and ``ToolOutputSubstitution`` (for ``tool_call``
targets).  The position and type are chosen so that a substitution is always
applicable; if no matching step exists for the requested type, the position
selector falls back to the other type.

Output
------
:func:`run` returns :class:`DistributionSuite` — JSON-serialisable via
:meth:`~DistributionSuite.to_json` — containing a list of
:class:`CorpusDistribution` records.  Each record reports:

* raw ``dirty_fractions`` (dirty / total steps) for every trial,
* percentile table (p5 / p25 / p50 / p75 / p90 / p95 / p99),
* normalized histogram (six bins: 0 %, 1–10 %, 11–25 %, 26–50 %,
  51–75 %, 76–100 %).

The CLI exposes ``stepback bench dirty-set-distributions``.
"""
from __future__ import annotations

import os
import random
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

from ..recorder import RecorderKey, record
from ..replay import replay
from ..substitutions import (
    PromptSubstitution,
    SubstitutionSet,
    ToolOutputSubstitution,
)
from ..divergence import compute_dirty_set

Corpus = Literal["linear_chain", "parallel_wide", "mixed_synthetic", "agent_fixture"]
Position = Literal["random", "early", "late"]
SubKind = Literal["PromptSubstitution", "ToolOutputSubstitution"]

# ------------------------------------------------------------------ helpers


def _bench_llm(model: str, messages: List[dict]) -> dict:
    import hashlib
    blob = model + "|" + "\n".join(
        f"{m.get('role', '')}:{m.get('content', '')}" for m in messages
    )
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    text = f"reply-{digest}"
    return {
        "id": f"chatcmpl-{digest}",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
        "usage": {
            "prompt_tokens": sum(len(str(m)) for m in messages),
            "completion_tokens": len(text),
            "total_tokens": sum(len(str(m)) for m in messages) + len(text),
        },
    }


def _bench_tool(name: str, args: dict) -> dict:
    import hashlib
    blob = name + "|" + str(sorted(args.items()))
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
    return {"result": f"tool-{name}-{digest}", "args": args}


# ------------------------------------------------------------------ trace builders


def _build_linear(n_steps: int, seed: int) -> str:
    """Build a purely sequential llm_call / tool_call chain trace."""
    rng = random.Random(seed)
    tmpdir = tempfile.mkdtemp(prefix="stepback-dist-")
    path = os.path.join(tmpdir, "linear.sb")
    with record(path, key=RecorderKey.fresh()) as rec:
        convo = [
            {"role": "system", "content": "linear bench"},
            {"role": "user", "content": f"start seed={seed}"},
        ]
        rec.llm_call("gpt-4o-2024-11-20", convo, executor=_bench_llm)
        for i in range(1, n_steps):
            if rng.random() < 0.5:
                rec.llm_call(
                    "gpt-4o-2024-11-20",
                    [{"role": "user", "content": f"step-{i}"}],
                    executor=_bench_llm,
                )
            else:
                rec.tool_call(
                    "lookup",
                    {"q": f"item-{i}", "seed": seed},
                    executor=_bench_tool,
                )
    return path


def _build_parallel(n_branches: int, seed: int) -> str:
    """Build a single fan-out / fan-in trace with *n_branches* branches."""
    rng = random.Random(seed)
    tmpdir = tempfile.mkdtemp(prefix="stepback-dist-")
    path = os.path.join(tmpdir, "parallel.sb")

    def make_branch(bi: int):
        def _branch(rec_inner) -> None:
            rec_inner.llm_call(
                "gpt-4o-mini-2024-07-18",
                [{"role": "user", "content": f"branch{bi} research seed={seed}"}],
                executor=_bench_llm,
            )
            rec_inner.tool_call(
                "fetch",
                {"url": f"http://example.com/b{bi}", "seed": seed},
                executor=_bench_tool,
            )

        return _branch

    with record(path, key=RecorderKey.fresh()) as rec:
        rec.llm_call(
            "gpt-4o-2024-11-20",
            [{"role": "user", "content": f"dispatch seed={seed}"}],
            executor=_bench_llm,
        )
        branches = [make_branch(i) for i in range(n_branches)]
        rec.parallel(
            f"fanout.seed{seed}",
            branches,
            branch_names=[f"b{i}" for i in range(n_branches)],
        )
        rec.llm_call(
            "gpt-4o-2024-11-20",
            [{"role": "user", "content": f"summarise seed={seed}"}],
            executor=_bench_llm,
        )
    return path


def _build_mixed(n_steps: int, seed: int) -> str:
    """Build a mixed linear+parallel trace using the existing SyntheticTrace."""
    import shutil
    from .replay_caching import SyntheticTrace

    st = SyntheticTrace(n_steps=n_steps, seed=seed)
    st.build()
    assert st.path is not None
    # Copy to a self-managed tmpdir so the SyntheticTrace (and its
    # TemporaryDirectory) can be garbage-collected without taking the
    # trace file with it.
    dst_dir = tempfile.mkdtemp(prefix="stepback-dist-")
    dst = os.path.join(dst_dir, "mixed.sb")
    shutil.copy2(st.path, dst)
    st.cleanup()
    return dst


def _build_agent(seed: int) -> str:
    """Build the canonical 12-step customer-payments agent fixture trace."""
    from stepback.testing import run_recorded_agent
    from stepback.testing.agent import fake_llm, fake_tool

    tmpdir = tempfile.mkdtemp(prefix="stepback-dist-")
    path = os.path.join(tmpdir, "agent.sb")
    with record(path, key=RecorderKey.fresh()) as rec:
        run_recorded_agent(rec)
    return path


# ------------------------------------------------------------------ substitution helpers


def _pick_target(
    steps: List[dict],
    position: Position,
    sub_kind: SubKind,
    rng: random.Random,
) -> Optional[dict]:
    """Pick a substitutable step from *steps* matching *position* and *sub_kind*."""
    want_kind = "llm_call" if sub_kind == "PromptSubstitution" else "tool_call"
    candidates = [s for s in steps if s.get("step_kind") == want_kind]
    if not candidates:
        candidates = [s for s in steps if s.get("step_kind") in ("llm_call", "tool_call")]
    if not candidates:
        return None

    n = len(candidates)
    if position == "early":
        pool = candidates[: max(1, n // 4)]
    elif position == "late":
        pool = candidates[max(0, 3 * n // 4) :]
    else:  # random
        pool = candidates
    return rng.choice(pool)


def _build_substitution_for(step: dict) -> Optional[SubstitutionSet]:
    sid = step["step_id"]
    kind = step.get("step_kind")
    subs = SubstitutionSet()
    if kind == "llm_call":
        recorded = list(step["inputs"].get("messages") or [])
        new_msgs = list(recorded) + [
            {"role": "user", "content": f"[dist-bench counterfactual @ {sid}]"}
        ]
        subs.add(PromptSubstitution(at_step=sid, new_messages=new_msgs))
    elif kind == "tool_call":
        subs.add(
            ToolOutputSubstitution(
                at_step=sid,
                fake_response={"perturbed": True, "step": sid},
            )
        )
    else:
        return None
    return subs


# ------------------------------------------------------------------ dataclasses


def _percentile(sorted_vals: List[float], p: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    idx = min(n - 1, int(p / 100.0 * n))
    return sorted_vals[idx]


def _histogram(fractions: List[float]) -> Dict[str, int]:
    bins: Dict[str, int] = {
        "0%": 0,
        "1-10%": 0,
        "11-25%": 0,
        "26-50%": 0,
        "51-75%": 0,
        "76-100%": 0,
    }
    for f in fractions:
        if f == 0.0:
            bins["0%"] += 1
        elif f <= 0.10:
            bins["1-10%"] += 1
        elif f <= 0.25:
            bins["11-25%"] += 1
        elif f <= 0.50:
            bins["26-50%"] += 1
        elif f <= 0.75:
            bins["51-75%"] += 1
        else:
            bins["76-100%"] += 1
    return bins


@dataclass
class CorpusDistribution:
    """Distribution of dirty-set fractions for one (corpus, position, sub_kind) cell."""

    corpus: str
    position: str
    sub_kind: str
    n_trials: int
    step_count_range: Tuple[int, int]
    dirty_fractions: List[float]

    # Percentile table
    p5: float = 0.0
    p25: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    mean: float = 0.0

    # Histogram (fraction of steps dirty, binned)
    histogram: Dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "corpus": self.corpus,
            "position": self.position,
            "sub_kind": self.sub_kind,
            "n_trials": self.n_trials,
            "step_count_range": list(self.step_count_range),
            "dirty_fractions": [round(f, 4) for f in self.dirty_fractions],
            "p5": round(self.p5, 4),
            "p25": round(self.p25, 4),
            "p50": round(self.p50, 4),
            "p75": round(self.p75, 4),
            "p90": round(self.p90, 4),
            "p95": round(self.p95, 4),
            "p99": round(self.p99, 4),
            "mean": round(self.mean, 4),
            "histogram": self.histogram,
        }


@dataclass
class DistributionSuite:
    """Collection of per-corpus dirty-set distributions.

    JSON-serialisable via :meth:`to_json`.
    """

    generated_at: str
    total_trials: int
    wall_time_ms: float
    distributions: List[CorpusDistribution] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "total_trials": self.total_trials,
            "wall_time_ms": round(self.wall_time_ms, 1),
            "distributions": [d.to_json() for d in self.distributions],
        }


# ------------------------------------------------------------------ runner


def _run_corpus_cell(
    corpus: Corpus,
    position: Position,
    sub_kind: SubKind,
    n_trials: int,
    n_steps: int,
    seed: int,
) -> CorpusDistribution:
    """Run *n_trials* dirty-set analyses for one (corpus, position, sub_kind) cell."""
    rng = random.Random(seed)
    fractions: List[float] = []
    step_counts: List[int] = []

    for trial in range(n_trials):
        trial_seed = seed * 10_000 + trial
        path: Optional[str] = None
        try:
            if corpus == "linear_chain":
                path = _build_linear(n_steps, trial_seed)
            elif corpus == "parallel_wide":
                n_branches = rng.randint(3, 10)
                path = _build_parallel(n_branches, trial_seed)
            elif corpus == "mixed_synthetic":
                path = _build_mixed(n_steps, trial_seed)
            else:  # agent_fixture
                path = _build_agent(trial_seed)

            t = replay(path)
            steps = t.recorded_steps
            target = _pick_target(steps, position, sub_kind, rng)
            if target is None:
                continue

            subs = _build_substitution_for(target)
            if subs is None:
                continue

            summary = compute_dirty_set(t, list(subs.items))
            n = summary.step_count
            step_counts.append(n)
            fractions.append(summary.dirty_count / n if n > 0 else 0.0)
        finally:
            if path:
                import shutil
                parent = os.path.dirname(path)
                if parent and os.path.isdir(parent):
                    shutil.rmtree(parent, ignore_errors=True)

    sorted_f = sorted(fractions)
    return CorpusDistribution(
        corpus=corpus,
        position=position,
        sub_kind=sub_kind,
        n_trials=len(fractions),
        step_count_range=(
            (min(step_counts), max(step_counts)) if step_counts else (0, 0)
        ),
        dirty_fractions=fractions,
        p5=_percentile(sorted_f, 5),
        p25=_percentile(sorted_f, 25),
        p50=_percentile(sorted_f, 50),
        p75=_percentile(sorted_f, 75),
        p90=_percentile(sorted_f, 90),
        p95=_percentile(sorted_f, 95),
        p99=_percentile(sorted_f, 99),
        mean=float(statistics.mean(fractions)) if fractions else 0.0,
        histogram=_histogram(fractions),
    )


def run(
    n_trials: int = 20,
    n_steps: int = 50,
    *,
    seed: int = 0,
    corpora: Optional[List[str]] = None,
) -> DistributionSuite:
    """Compute dirty-set distributions over multiple synthetic fixture corpora.

    Parameters
    ----------
    n_trials:
        Number of independent traces per (corpus, position, sub_kind) cell.
        Default 20.
    n_steps:
        Target trace length for length-parameterised corpora (``linear_chain``,
        ``mixed_synthetic``).  ``parallel_wide`` derives its length from the
        random branch-width; ``agent_fixture`` has a fixed 12-step shape.
        Default 50.
    seed:
        Base RNG seed.  Reproducible across runs for the same ``n_trials``,
        ``n_steps``, and ``seed``.
    corpora:
        Subset of corpora to run; defaults to all four.

    Returns
    -------
    DistributionSuite
        JSON-serialisable collection of per-corpus distributions.
    """
    import datetime

    all_corpora: List[Corpus] = [
        "linear_chain",
        "parallel_wide",
        "mixed_synthetic",
        "agent_fixture",
    ]
    if corpora is not None:
        all_corpora = [c for c in all_corpora if c in corpora]  # type: ignore[assignment]

    positions: List[Position] = ["random", "early", "late"]
    sub_kinds: List[SubKind] = ["PromptSubstitution", "ToolOutputSubstitution"]

    t0 = time.perf_counter_ns()
    dists: List[CorpusDistribution] = []
    total = 0

    for corpus in all_corpora:
        for position in positions:
            for sub_kind in sub_kinds:
                cell_seed = seed + hash((corpus, position, sub_kind)) % (2**20)
                cell = _run_corpus_cell(
                    corpus=corpus,  # type: ignore[arg-type]
                    position=position,
                    sub_kind=sub_kind,
                    n_trials=n_trials,
                    n_steps=n_steps,
                    seed=cell_seed,
                )
                dists.append(cell)
                total += cell.n_trials

    wall_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
    return DistributionSuite(
        generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        total_trials=total,
        wall_time_ms=wall_ms,
        distributions=dists,
    )
