"""Replay-caching / dirty-set-propagation benchmark.

Builds a deterministic synthetic agent trace using the *real*
:py:class:`stepback.recorder.Recorder` (so the canonical hashes,
context-from-parent rebinding, and parallel-branch frames are exactly
what the production replay engine sees), then applies a single
substitution per the requested strategy and measures the dirty-set
size produced by :py:meth:`stepback.replay.Trace.run_replay`.

The headline metric is ``cost_reduction_factor = n_steps /
mean(dirty_set_sizes)``, expressing how many real LLM calls are
avoided by per-step content-addressed caching with dirty-set
propagation, vs. naive O(N) re-execution.
"""
from __future__ import annotations

import hashlib
import os
import random
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from ..recorder import RecorderKey, record
from ..replay import Executor, replay
from ..substitutions import (
    PromptSubstitution,
    Substitution,
    SubstitutionSet,
    ToolOutputSubstitution,
)

Strategy = Literal[
    "random_step",
    "first_quarter",
    "last_quarter",
    "prompt_only",
    "tool_only",
]


# ------------------------------------------------------------ fakes
#
# A tiny self-contained deterministic LLM and tool. Outputs are pure
# functions of inputs so the bench is bytewise-reproducible across
# runs and the recorded outputs round-trip exactly through replay.

def _bench_llm(model: str, messages: List[dict]) -> dict:
    blob = model + "|" + "\n".join(
        f"{m.get('role','')}:{m.get('content','')}" for m in messages
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
            "prompt_tokens": sum(len(m.get("content", "")) for m in messages),
            "completion_tokens": len(text),
            "total_tokens": sum(len(m.get("content", "")) for m in messages)
            + len(text),
        },
    }


def _bench_tool(name: str, args: dict) -> dict:
    blob = name + "|" + repr(sorted(args.items()))
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
    return {"tool": name, "digest": digest, "args": args}


def _bench_router(name: str, options: List[str]) -> str:
    return options[0] if options else ""


# ------------------------------------------------------------ trace


@dataclass
class SyntheticTrace:
    """Builder for a deterministic N-step synthetic agent trace.

    The shape mixes sequential ``llm_call``/``tool_call`` steps with
    occasional fan-out / fan-in parallel blocks, so the dirty-set
    distribution under a random single-step substitution exercises
    both the linear ``parent_step_id`` chain and the multi-parent
    ``parallel_branch_join`` rebinding.
    """

    n_steps: int
    seed: int = 0
    path: Optional[str] = None
    _tmpdir: Optional[tempfile.TemporaryDirectory] = field(
        default=None, repr=False
    )

    def build(self) -> str:
        """Materialize the trace under a fresh temp directory and return
        the .sb path."""
        self._tmpdir = tempfile.TemporaryDirectory(prefix="stepback-bench-")
        path = os.path.join(self._tmpdir.name, "synthetic.sb")
        rng = random.Random(self.seed)
        emitted = 0

        def need() -> int:
            return self.n_steps - emitted

        with record(path, key=RecorderKey.fresh()) as rec:
            convo = [
                {"role": "system", "content": "synthetic bench agent"},
                {"role": "user", "content": f"task seed={self.seed}"},
            ]
            rec.llm_call("gpt-4o-2024-11-20", convo, executor=_bench_llm)
            emitted += 1

            block = 0
            while need() > 0:
                block += 1
                if need() >= 6 and rng.random() < 0.4:
                    branches_remaining = min(3, max(2, need() // 3))

                    def make_branch(bi: int):
                        def _branch(rec_inner) -> None:
                            rec_inner.llm_call(
                                "gpt-4o-mini-2024-07-18",
                                [
                                    {
                                        "role": "user",
                                        "content": f"block{block} branch{bi} step1",
                                    }
                                ],
                                executor=_bench_llm,
                            )
                            rec_inner.tool_call(
                                "lookup",
                                {"q": f"block{block}/branch{bi}"},
                                executor=_bench_tool,
                            )

                        return _branch

                    branches = [make_branch(i) for i in range(branches_remaining)]
                    rec.parallel(
                        f"block{block}.fanout",
                        branches,
                        branch_names=[
                            f"block{block}.b{i}" for i in range(branches_remaining)
                        ],
                    )
                    emitted += 1 + 2 * branches_remaining + 1
                else:
                    pick = rng.random()
                    if pick < 0.5 and need() >= 1:
                        rec.llm_call(
                            "gpt-4o-2024-11-20",
                            [
                                {
                                    "role": "user",
                                    "content": f"block{block}.llm",
                                }
                            ],
                            executor=_bench_llm,
                        )
                        emitted += 1
                    elif pick < 0.85 and need() >= 1:
                        rec.tool_call(
                            "lookup",
                            {"q": f"block{block}.tool"},
                            executor=_bench_tool,
                        )
                        emitted += 1
                    else:
                        rec.router(
                            f"router.b{block}",
                            choice="A",
                            options=["A", "B", "C"],
                        )
                        emitted += 1

            self.path = path
            return path

    def cleanup(self) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None


# ------------------------------------------------------------ result


@dataclass
class BenchResult:
    n_steps: int
    n_substitutions: int
    n_trials: int
    strategy: str
    dirty_set_sizes: List[int]
    median_dirty_set: float
    p95_dirty_set: float
    mean_dirty_set: float
    cost_reduction_factor: float
    wall_time_ms: float

    def to_json(self) -> dict:
        return {
            "n_steps": self.n_steps,
            "n_substitutions": self.n_substitutions,
            "n_trials": self.n_trials,
            "strategy": self.strategy,
            "dirty_set_sizes": list(self.dirty_set_sizes),
            "median_dirty_set": self.median_dirty_set,
            "p95_dirty_set": self.p95_dirty_set,
            "mean_dirty_set": self.mean_dirty_set,
            "cost_reduction_factor": self.cost_reduction_factor,
            "wall_time_ms": self.wall_time_ms,
        }

    def summary_line(self) -> str:
        return (
            f"replay-caching n_steps={self.n_steps} "
            f"strategy={self.strategy} trials={self.n_trials} "
            f"median_dirty={self.median_dirty_set:.1f} "
            f"p95_dirty={self.p95_dirty_set:.1f} "
            f"cost_reduction={self.cost_reduction_factor:.2f}x "
            f"wall_ms={self.wall_time_ms:.1f}"
        )


# ------------------------------------------------------------ pickers


def _pick_target(
    rec_steps: List[dict],
    strategy: Strategy,
    rng: random.Random,
) -> Optional[dict]:
    n = len(rec_steps)
    if n == 0:
        return None

    def _candidates(pool):
        return [s for s in pool if s["step_kind"] in ("llm_call", "tool_call")]

    if strategy == "random_step":
        c = _candidates(rec_steps)
    elif strategy == "first_quarter":
        c = _candidates(rec_steps[: max(1, n // 4)])
    elif strategy == "last_quarter":
        c = _candidates(rec_steps[max(0, 3 * n // 4):])
    elif strategy == "prompt_only":
        c = [s for s in rec_steps if s["step_kind"] == "llm_call"]
    elif strategy == "tool_only":
        c = [s for s in rec_steps if s["step_kind"] == "tool_call"]
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")

    if not c:
        c = _candidates(rec_steps)
    if not c:
        return None
    return rng.choice(c)


def _build_substitution(step: dict, strategy: Strategy) -> Substitution:
    """Build a typed Substitution that *will* mark this step dirty."""
    sid = step["step_id"]
    kind = step["step_kind"]
    if kind == "llm_call":
        recorded_messages = list(step["inputs"].get("messages") or [])
        new_messages = list(recorded_messages) + [
            {
                "role": "user",
                "content": f"[counterfactual perturbation @ {sid}]",
            }
        ]
        return PromptSubstitution(at_step=sid, new_messages=new_messages)
    if kind == "tool_call":
        return ToolOutputSubstitution(
            at_step=sid,
            fake_response={
                "perturbed": True,
                "step_id": sid,
                "marker": "stepback-bench-counterfactual",
            },
        )
    raise ValueError(f"no substitution defined for step kind {kind!r}")


# ------------------------------------------------------------ runner


def _executor() -> Executor:
    return Executor(
        llm=_bench_llm,
        tool=_bench_tool,
        router=_bench_router,
    )


def run(
    n_steps: int,
    n_trials: int = 10,
    strategy: Strategy = "random_step",
    *,
    seed: int = 0,
) -> BenchResult:
    """Run the replay-caching benchmark and return a :class:`BenchResult`."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    rng = random.Random(seed)
    dirty_sizes: List[int] = []
    actual_steps: List[int] = []
    t0 = time.perf_counter_ns()

    for trial in range(n_trials):
        st = SyntheticTrace(n_steps=n_steps, seed=seed * 1000 + trial)
        path = st.build()
        try:
            t = replay(path)
            target = _pick_target(t.recorded_steps, strategy, rng)
            if target is None:
                raise RuntimeError(
                    "synthetic trace produced zero substitutable steps"
                )
            sub = _build_substitution(target, strategy)
            subs = SubstitutionSet()
            subs.add(sub)
            result = t.run_replay(subs, _executor())
            dirty_sizes.append(result.dirty_count)
            actual_steps.append(len(t.recorded_steps))
        finally:
            st.cleanup()

    wall_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
    sorted_sizes = sorted(dirty_sizes)
    n = len(sorted_sizes)
    median = float(statistics.median(sorted_sizes))
    p95_idx = min(n - 1, int(0.95 * n))
    p95 = float(sorted_sizes[p95_idx])
    mean_dirty = float(statistics.mean(sorted_sizes))
    avg_actual = statistics.mean(actual_steps) if actual_steps else float(n_steps)
    cost_reduction = avg_actual / mean_dirty if mean_dirty > 0 else float("inf")

    return BenchResult(
        n_steps=int(round(avg_actual)),
        n_substitutions=1,
        n_trials=n_trials,
        strategy=str(strategy),
        dirty_set_sizes=dirty_sizes,
        median_dirty_set=median,
        p95_dirty_set=p95,
        mean_dirty_set=mean_dirty,
        cost_reduction_factor=cost_reduction,
        wall_time_ms=wall_ms,
    )


def compare(
    suite: List[int],
    *,
    n_trials: int = 10,
    strategy: Strategy = "random_step",
    seed: int = 0,
) -> Dict[int, BenchResult]:
    """Run :func:`run` across multiple trace sizes."""
    out: Dict[int, BenchResult] = {}
    for n in suite:
        out[n] = run(n_steps=n, n_trials=n_trials, strategy=strategy, seed=seed)
    return out
