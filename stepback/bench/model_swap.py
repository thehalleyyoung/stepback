"""Model-swap differential benchmark.

Measures two separable properties when swapping one LLM model for another:

**Fidelity against full re-execution**
    After applying :class:`~stepback.substitutions.ModelSubstitution` on every
    ``llm_call`` step and replaying with a model-B executor, the replay engine's
    dirty-set output must be byte-identical to a *fresh* model-B recording of the
    same trace topology.  This confirms the soundness guarantee holds under a full
    model swap: no step is incorrectly served from a stale model-A cache entry.

**Statistically grounded difference detection**
    A paired t-test and Cohen's d effect size measure whether model B's output
    lengths are significantly different from model A's across the same prompts.
    The outputs are variable-length deterministic strings whose length depends on
    the model name, so within-group variance is positive and the tests are
    meaningful.

Headline metrics
~~~~~~~~~~~~~~~~
* ``replay_fidelity_rate`` — fraction of model-B replay outputs matching the
  corresponding fresh model-B outputs (expected 1.0; any deviation signals a
  replay soundness bug).
* ``model_agreement_rate`` — fraction of LLM steps where model A and model B
  produce identical output (expected 0.0 for clearly distinct models; serves as
  a sanity check that the two fake LLMs are genuinely different).
* ``t_stat``, ``p_value`` — paired Welch t-test on per-step output-length
  differences (B minus A).  ``p_value < 0.05`` confirms the two models produce
  statistically distinguishable output lengths.
* ``effect_size`` — Cohen's d (paired), measuring the magnitude of the
  output-length shift independent of sample size.
* ``output_length_ratio`` — ``mean_length_b / mean_length_a``, with
  ``ci_95_lower`` and ``ci_95_upper`` from a simple percentile bootstrap.

Workflow (per trial)
~~~~~~~~~~~~~~~~~~~~
1. Build a deterministic synthetic N-step trace with ``_model_a_llm``; collect
   the text output for every ``llm_call`` step.
2. Build an identical synthetic N-step trace with ``_model_b_llm`` (same seed,
   same topology, same messages); collect the text output for every
   ``llm_call`` step.
3. Replay the model-A ``.sb`` trace with
   :class:`~stepback.substitutions.ModelSubstitution` on every ``llm_call``
   step (forcing all LLM steps dirty) and the ``_model_b_llm`` executor;
   collect the replay output for every ``llm_call`` step.
4. Assert (Phase 3 vs Phase 2): replay outputs must match fresh model-B outputs.
5. Record output lengths and compute statistics across all trials.
"""
from __future__ import annotations

import hashlib
import math
import os
import random
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..recorder import RecorderKey, record
from ..replay import Executor, replay
from ..substitutions import ModelSubstitution, SubstitutionSet

# ------------------------------------------------------------------- models

_MODEL_A = "bench-model-a-v1"
_MODEL_B = "bench-model-b-v1"


def _output_text_a(model: str, messages: list) -> str:
    """Deterministic variable-length output for model A.

    Length = ``10 + (msg_nibble + 1)`` chars, so 11..26 chars total.  The
    suffix length is determined by the *messages-only* hash so that model A
    and model B have the same suffix-length nibble for the same prompt,
    guaranteeing model B is *always* longer (see :func:`_output_text_b`).
    """
    content_blob = model + "|" + "\n".join(
        f"{m.get('role', '')}:{m.get('content', '')}" for m in messages
    )
    content_digest = hashlib.sha256(content_blob.encode()).hexdigest()[:8]
    msg_blob = "\n".join(
        f"{m.get('role', '')}:{m.get('content', '')}" for m in messages
    )
    nibble = int(hashlib.sha256(msg_blob.encode()).hexdigest()[0], 16)
    suffix_len = nibble + 1  # 1..16
    return "A:" + content_digest + ("!" * suffix_len)


def _output_text_b(model: str, messages: list) -> str:
    """Deterministic variable-length output for model B.

    Length = ``10 + 2 * (msg_nibble + 1)`` chars, so 12..42 chars — always
    strictly longer than the model-A output for the *same prompt* because both
    models use the same message-derived nibble for the suffix but model B
    doubles it.
    """
    content_blob = model + "|" + "\n".join(
        f"{m.get('role', '')}:{m.get('content', '')}" for m in messages
    )
    content_digest = hashlib.sha256(content_blob.encode()).hexdigest()[:8]
    msg_blob = "\n".join(
        f"{m.get('role', '')}:{m.get('content', '')}" for m in messages
    )
    nibble = int(hashlib.sha256(msg_blob.encode()).hexdigest()[0], 16)
    suffix_len = 2 * (nibble + 1)  # 2..32
    return "B:" + content_digest + ("!" * suffix_len)


def _model_a_llm(model: str, messages: list) -> dict:
    text = _output_text_a(model, messages)
    digest = hashlib.sha256((model + repr(messages)).encode()).hexdigest()[:16]
    return {
        "id": f"chatcmpl-a-{digest}",
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


def _model_b_llm(model: str, messages: list) -> dict:
    text = _output_text_b(model, messages)
    digest = hashlib.sha256((model + repr(messages)).encode()).hexdigest()[:16]
    return {
        "id": f"chatcmpl-b-{digest}",
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
    digest = hashlib.sha256(blob.encode()).hexdigest()[:12]
    return {"tool": name, "digest": digest, "args": args}


def _bench_router(name: str, options: list) -> str:
    return options[0] if options else ""


# ---------------------------------------------------------------- trace builder

def _build_trace(
    n_steps: int,
    seed: int,
    llm_fn,
    model_name: str,
    tmpdir: str,
    filename: str = "trace.sb",
) -> str:
    """Record a synthetic N-step trace using *llm_fn* and *model_name*.

    Returns the path to the written ``.sb`` file.
    """
    path = os.path.join(tmpdir, filename)
    rng = random.Random(seed)
    emitted = 0

    def need() -> int:
        return n_steps - emitted

    with record(path, key=RecorderKey.fresh()) as rec:
        convo = [
            {"role": "system", "content": "model-swap bench agent"},
            {"role": "user", "content": f"task seed={seed}"},
        ]
        rec.llm_call(model_name, convo, executor=llm_fn)
        emitted += 1

        block = 0
        while need() > 0:
            block += 1
            pick = rng.random()
            if pick < 0.5 and need() >= 1:
                rec.llm_call(
                    model_name,
                    [{"role": "user", "content": f"block{block}.llm"}],
                    executor=llm_fn,
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
    return path


# ---------------------------------------------------------------- statistics

def _betacf(a: float, b: float, x: float) -> float:
    """Lentz continued-fraction expansion of the incomplete beta function."""
    MAX_ITER = 200
    EPS = 3e-14
    FPMIN = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function :math:`I_x(a, b)`.

    Uses the symmetry relation for numerical stability:
    :math:`I_x(a,b) = 1 - I_{1-x}(b, a)` when
    :math:`x > (a+1)/(a+b+2)`.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betai(b, a, 1.0 - x)
    # log(B(a, b)) = lgamma(a) + lgamma(b) - lgamma(a+b)
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a
    return front * _betacf(a, b, x)


def _t_two_tailed_p(t: float, df: float) -> float:
    """Two-tailed p-value for Student's t-distribution with *df* degrees of freedom.

    Uses :func:`_betai` via the identity
    :math:`p = I_{df/(df+t^2)}(df/2, 1/2)`.
    """
    if df <= 0:
        return 1.0
    x = df / (df + t * t)
    return _betai(df / 2.0, 0.5, x)


def _paired_t_test(
    diffs: List[float],
) -> Tuple[float, float, float]:
    """One-sample (paired) t-test on *diffs*.

    Returns ``(t_stat, df, p_value)`` for :math:`H_0: \\mu_d = 0`.
    Returns ``(0.0, 0.0, 1.0)`` when fewer than two samples are available or
    when the sample standard deviation is zero.
    """
    n = len(diffs)
    if n < 2:
        return 0.0, 0.0, 1.0
    d_bar = statistics.mean(diffs)
    s_d = statistics.stdev(diffs)
    if s_d == 0.0:
        return float("inf") if d_bar != 0 else 0.0, float(n - 1), 0.0 if d_bar != 0 else 1.0
    t = d_bar / (s_d / math.sqrt(n))
    df = float(n - 1)
    p = _t_two_tailed_p(t, df)
    return t, df, p


def _cohens_d_paired(diffs: List[float]) -> float:
    """Cohen's d for a paired design: :math:`\\bar{d} / s_d`.

    Returns 0.0 when fewer than 2 samples or when :math:`s_d = 0`.
    """
    n = len(diffs)
    if n < 2:
        return 0.0
    d_bar = statistics.mean(diffs)
    s_d = statistics.stdev(diffs)
    if s_d == 0.0:
        return 0.0
    return d_bar / s_d


def _bootstrap_ratio_ci(
    lengths_a: List[float],
    lengths_b: List[float],
    *,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI for ``mean(b) / mean(a)``.

    Returns ``(lower, upper)``.  Falls back to ``(1.0, 1.0)`` when either list
    is empty or ``mean(a) == 0``.
    """
    n = min(len(lengths_a), len(lengths_b))
    if n == 0 or statistics.mean(lengths_a) == 0:
        return 1.0, 1.0
    rng = random.Random(seed)
    ratios = []
    for _ in range(n_boot):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        ma = statistics.mean(lengths_a[i] for i in idx)
        mb = statistics.mean(lengths_b[i] for i in idx)
        if ma > 0:
            ratios.append(mb / ma)
    if not ratios:
        return 1.0, 1.0
    ratios.sort()
    lo = int(math.floor(alpha / 2 * len(ratios)))
    hi = min(len(ratios) - 1, int(math.ceil((1 - alpha / 2) * len(ratios))))
    return ratios[lo], ratios[hi]


# ---------------------------------------------------------------- LLM output extraction

def _extract_llm_text(outputs: object) -> Optional[str]:
    """Return the assistant text from an OpenAI-shaped response dict, or None."""
    if not isinstance(outputs, dict):
        return None
    choices = outputs.get("choices")
    if not choices or not isinstance(choices, list):
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    msg = first.get("message", {})
    if isinstance(msg, dict):
        return msg.get("content")
    return None


# ---------------------------------------------------------------- result

@dataclass
class ModelSwapResult:
    """Aggregate result from the model-swap differential benchmark.

    The three headline metrics map directly to the two guarantees in the
    step description:

    *Fidelity against full re-execution* (correctness of dirty-set replay)
        * ``replay_fidelity_rate`` — 1.0 means every ``llm_call`` step in the
          model-B replay matched the corresponding fresh model-B output.

    *Statistically grounded difference detection*
        * ``t_stat`` / ``p_value`` — paired t-test on per-step output-length
          differences.
        * ``effect_size`` — Cohen's d (paired), magnitude of the difference.
        * ``output_length_ratio`` with ``ci_95_lower`` / ``ci_95_upper``.
    """

    model_a_name: str
    model_b_name: str
    n_trials: int
    n_steps: int
    """Mean actual step count per trial."""
    n_llm_steps: int
    """Total paired LLM-step observations across all trials."""

    # Cross-model agreement / fidelity
    model_agreement_rate: float
    """Fraction of LLM steps where model A and model B produce identical
    output text.  Expected ≈ 0.0 for clearly distinct models."""
    replay_fidelity_rate: float
    """Fraction of model-B replay outputs that match the corresponding
    fresh model-B output.  Expected = 1.0 (soundness guarantee)."""

    # Output length statistics
    mean_length_a: float
    mean_length_b: float
    output_length_ratio: float
    """``mean_length_b / mean_length_a``."""
    ci_95_lower: float
    ci_95_upper: float
    """95% bootstrap CI for ``output_length_ratio``."""

    # Statistical test (paired, B − A differences)
    t_stat: float
    p_value: float
    effect_size: float
    """Cohen's d for paired output-length differences."""

    # Dirty-set counts from the model-swap replay
    dirty_count_mean: float
    """Mean number of dirty steps per trial."""
    cache_hit_count_mean: float
    """Mean number of cache-hit steps per trial (non-LLM steps whose
    inputs were unaffected by the model swap)."""

    wall_time_ms: float

    def to_json(self) -> dict:
        return {
            "model_a_name": self.model_a_name,
            "model_b_name": self.model_b_name,
            "n_trials": self.n_trials,
            "n_steps": self.n_steps,
            "n_llm_steps": self.n_llm_steps,
            "model_agreement_rate": self.model_agreement_rate,
            "replay_fidelity_rate": self.replay_fidelity_rate,
            "mean_length_a": self.mean_length_a,
            "mean_length_b": self.mean_length_b,
            "output_length_ratio": self.output_length_ratio,
            "ci_95_lower": self.ci_95_lower,
            "ci_95_upper": self.ci_95_upper,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "dirty_count_mean": self.dirty_count_mean,
            "cache_hit_count_mean": self.cache_hit_count_mean,
            "wall_time_ms": self.wall_time_ms,
        }

    def summary_line(self) -> str:
        sig = "sig" if (self.p_value < 0.05 and self.p_value >= 0) else "ns"
        return (
            f"model-swap n_steps={self.n_steps} trials={self.n_trials} "
            f"llm_steps={self.n_llm_steps} "
            f"replay_fidelity={self.replay_fidelity_rate:.3f} "
            f"agreement={self.model_agreement_rate:.3f} "
            f"ratio={self.output_length_ratio:.3f}"
            f"[{self.ci_95_lower:.3f},{self.ci_95_upper:.3f}] "
            f"t={self.t_stat:.2f} p={self.p_value:.4f}({sig}) "
            f"d={self.effect_size:.2f} "
            f"dirty_mean={self.dirty_count_mean:.1f} "
            f"cache_hit_mean={self.cache_hit_count_mean:.1f} "
            f"wall_ms={self.wall_time_ms:.1f}"
        )


# ---------------------------------------------------------------- runner

def run(
    n_steps: int,
    n_trials: int = 5,
    *,
    seed: int = 0,
    model_a_name: str = _MODEL_A,
    model_b_name: str = _MODEL_B,
) -> ModelSwapResult:
    """Run the model-swap differential benchmark and return a :class:`ModelSwapResult`.

    Per-trial workflow
    ~~~~~~~~~~~~~~~~~~
    1. Build a deterministic synthetic trace with model A.
    2. Build an identical synthetic trace with model B.
    3. Replay the model-A ``.sb`` trace with
       :class:`~stepback.substitutions.ModelSubstitution` on every
       ``llm_call`` step plus the model-B executor.
    4. Collect paired LLM outputs (A, B-fresh, B-replay) and accumulate
       statistics.

    Parameters
    ----------
    n_steps :
        Target number of steps per synthetic trace (may be slightly exceeded
        by the trace builder).
    n_trials :
        Number of independent trials; statistics are aggregated across all
        trials.
    seed :
        Base RNG seed; trial *i* uses ``seed * 1000 + i``.
    model_a_name, model_b_name :
        Model-ID strings embedded in the trace (and used as hash inputs for
        the deterministic fake LLMs).  Changing these produces different
        output-length distributions.

    Raises
    ------
    ValueError
        If ``n_steps < 1`` or ``n_trials < 1``.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    total_t0 = time.perf_counter_ns()

    all_lengths_a: List[float] = []
    all_lengths_b: List[float] = []
    all_diffs: List[float] = []
    agree_count = 0
    total_llm = 0
    fidelity_match = 0
    fidelity_total = 0
    actual_steps: List[int] = []
    dirty_counts: List[int] = []
    cache_hit_counts: List[int] = []

    for trial in range(n_trials):
        trial_seed = seed * 1000 + trial
        tmpdir_obj = tempfile.TemporaryDirectory(prefix="stepback-bench-ms-")
        try:
            tmpdir = tmpdir_obj.name

            # Phase 1: model A trace
            path_a = _build_trace(
                n_steps, trial_seed,
                _model_a_llm, model_a_name,
                tmpdir, "trace_a.sb",
            )
            t_a = replay(path_a)
            recorded_a = t_a.recorded_steps
            actual_steps.append(len(recorded_a))

            # Gather model-A LLM outputs in step order.
            llm_steps_a = [
                (s["step_id"], s["outputs"])
                for s in recorded_a
                if s.get("step_kind") == "llm_call"
            ]

            # Phase 2: fresh model B trace (same topology, same seed)
            path_b = _build_trace(
                n_steps, trial_seed,
                _model_b_llm, model_b_name,
                tmpdir, "trace_b.sb",
            )
            t_b = replay(path_b)
            recorded_b = t_b.recorded_steps
            llm_steps_b = [
                (s["step_id"], s["outputs"])
                for s in recorded_b
                if s.get("step_kind") == "llm_call"
            ]

            # Phase 3: replay model-A trace with ModelSubstitution on every
            # llm_call step, using the model-B executor.
            subs = SubstitutionSet()
            for step in recorded_a:
                if step.get("step_kind") == "llm_call":
                    subs.add(
                        ModelSubstitution(
                            at_step=step["step_id"],
                            new_model_id=model_b_name,
                        )
                    )

            executor_b = Executor(
                llm=_model_b_llm,
                tool=_bench_tool,
                router=_bench_router,
            )
            replay_result = t_a.run_replay(subs, executor_b)
            dirty_counts.append(replay_result.dirty_count)
            cache_hit_counts.append(replay_result.cache_hit_count)

            # Collect replay outputs for llm_call steps.
            replay_llm_outputs: Dict[str, object] = {
                sv.step_id: sv.outputs
                for sv in replay_result.steps
                if sv.kind == "llm_call"
            }

            # Pair A vs B-fresh outputs by ordered LLM-step index.
            n_pairs = min(len(llm_steps_a), len(llm_steps_b))
            for i in range(n_pairs):
                sid_a, out_a = llm_steps_a[i]
                _sid_b, out_b_fresh = llm_steps_b[i]
                out_b_replay = replay_llm_outputs.get(sid_a)

                text_a = _extract_llm_text(out_a)
                text_b_fresh = _extract_llm_text(out_b_fresh)
                text_b_replay = _extract_llm_text(out_b_replay)

                # Agreement (A vs B-fresh)
                if text_a is not None and text_b_fresh is not None:
                    if text_a == text_b_fresh:
                        agree_count += 1
                    total_llm += 1
                    all_lengths_a.append(float(len(text_a)))
                    all_lengths_b.append(float(len(text_b_fresh)))
                    all_diffs.append(float(len(text_b_fresh)) - float(len(text_a)))

                # Fidelity (B-replay vs B-fresh)
                if text_b_fresh is not None and text_b_replay is not None:
                    fidelity_total += 1
                    if text_b_fresh == text_b_replay:
                        fidelity_match += 1

        finally:
            tmpdir_obj.cleanup()

    total_wall_ms = (time.perf_counter_ns() - total_t0) / 1_000_000.0

    # Aggregate statistics.
    mean_len_a = statistics.mean(all_lengths_a) if all_lengths_a else 0.0
    mean_len_b = statistics.mean(all_lengths_b) if all_lengths_b else 0.0
    ratio = mean_len_b / mean_len_a if mean_len_a > 0 else 1.0
    ci_lo, ci_hi = _bootstrap_ratio_ci(all_lengths_a, all_lengths_b, seed=seed)

    t_stat, _df, p_value = _paired_t_test(all_diffs)
    effect_size = _cohens_d_paired(all_diffs)

    model_agreement_rate = agree_count / total_llm if total_llm > 0 else 0.0
    replay_fidelity_rate = fidelity_match / fidelity_total if fidelity_total > 0 else 0.0
    mean_actual_steps = statistics.mean(actual_steps) if actual_steps else float(n_steps)
    dirty_mean = statistics.mean(dirty_counts) if dirty_counts else 0.0
    cache_hit_mean = statistics.mean(cache_hit_counts) if cache_hit_counts else 0.0

    return ModelSwapResult(
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        n_trials=n_trials,
        n_steps=int(round(mean_actual_steps)),
        n_llm_steps=total_llm,
        model_agreement_rate=model_agreement_rate,
        replay_fidelity_rate=replay_fidelity_rate,
        mean_length_a=mean_len_a,
        mean_length_b=mean_len_b,
        output_length_ratio=ratio,
        ci_95_lower=ci_lo,
        ci_95_upper=ci_hi,
        t_stat=t_stat,
        p_value=p_value,
        effect_size=effect_size,
        dirty_count_mean=dirty_mean,
        cache_hit_count_mean=cache_hit_mean,
        wall_time_ms=total_wall_ms,
    )


def compare(
    suite: List[int],
    *,
    n_trials: int = 5,
    seed: int = 0,
) -> Dict[int, ModelSwapResult]:
    """Run :func:`run` across multiple trace sizes and return a dict."""
    out: Dict[int, ModelSwapResult] = {}
    for n in suite:
        out[n] = run(n_steps=n, n_trials=n_trials, seed=seed)
    return out
