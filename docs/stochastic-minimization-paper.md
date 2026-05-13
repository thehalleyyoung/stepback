# Stochastic Minimization and Regression Localization in LLM Agent Traces

**Paper artifact — ICSE / ISSTA track (v1)**

This document is the paper-grade artifact for the stochastic minimization
extension of the causal minimization toolkit in
[`stepback/minimize.py`](../stepback/minimize.py). It discharges part of
Step 147 of [`100_STEPS.md`](../100_STEPS.md).

This document extends [`docs/minimization-paper.md`](./minimization-paper.md),
which covers the deterministic ddmin toolkit in detail (Step 88). The
relationship between the two documents:

| Document | Scope |
|----------|-------|
| `docs/minimization-paper.md` | Causal minimization toolkit: DDMin, binary halving, Shapley attribution, multi-objective (deterministic oracle assumption). |
| `docs/stochastic-minimization-paper.md` *(this document)* | Extension for stochastic predicates: confidence intervals, flaky predicate classification, stability metrics, repeated dirty-step re-execution. |

Do not read this document in isolation; the core algorithm definitions,
complexity bounds, and baseline results are in
[`docs/minimization-paper.md`](./minimization-paper.md).

Cross-references:
[`stepback/minimize.py`](../stepback/minimize.py),
[`stepback/replay.py`](../stepback/replay.py),
[`docs/minimization-paper.md`](./minimization-paper.md),
[`docs/dirty-set.md`](./dirty-set.md),
[`tests/test_stochastic_replay.py`](../tests/test_stochastic_replay.py).

---

## Abstract

Delta debugging for AI agent traces faces a challenge absent from classical
software debugging: the oracle predicate evaluating the replay result may itself
be non-deterministic. A stochastic LLM may produce a policy-violating output on
60% of re-executions and a compliant output on 40%. Treating such a predicate as
deterministic leads to:

1. **False convergence**: ddmin declares a substitution set minimal when it was
   only a flaky False; the full set would still trigger the predicate reliably.
2. **Missed causes**: a substitution that consistently contributes 40% to the
   predicate is never ranked as a cause because it never satisfies a Hard True.

This paper presents four extensions to the causal minimization toolkit that
address stochastic predicates:

1. **Confidence-interval guards**: wrap each oracle call in a Wilson confidence
   interval; report flaky predicates before declaring False.
2. **Stability classification**: automatically classify predicates as
   deterministic, stable-stochastic, or unstable-stochastic based on
   observed flip rate.
3. **Repeated dirty-step re-execution**: when a predicate is classified
   stochastic, run the same dirty set k times and use majority vote.
4. **Stochastic Shapley attribution**: estimate contribution weights using
   sampling under a seeded executor rather than exhaustive enumeration.

---

## 1. Background

The causal minimization problem (defined fully in
[`docs/minimization-paper.md`](./minimization-paper.md) §1) is:

> **Given** trace T, substitution set S with P(replay(T, S)) = True,
> **find** a minimal M ⊆ S such that P(replay(T, M)) = True.

The standard assumption is:

- **A\_oracle (deterministic oracle)**: P(replay(T, S)) returns the same value
  every time for the same S.

This assumption fails when the predicate inspects the LLM output text of a
dirty step, which is stochastic.

---

## 2. Stochastic oracle model

### 2.1 Oracle as Bernoulli process

We model the stochastic oracle as:

```
P_prob(S) := Pr[P(replay(T, S)) = True] ∈ [0, 1]
```

For a deterministic oracle, P_prob(S) ∈ {0, 1}. For a stochastic oracle,
P_prob(S) may take any value in (0, 1).

### 2.2 Stability categories

We classify predicates by their observed flip rate over `k_stability` repeated
evaluations on the same substitution set S:

| Category | Flip rate | Description |
|----------|-----------|-------------|
| Deterministic | 0 | Never flips across k evaluations. |
| Stable-stochastic | < 0.2 | Flips rarely; majority vote is reliable. |
| Unstable-stochastic | ≥ 0.2 | Flips frequently; minimization results have wide confidence intervals. |

### 2.3 Confidence interval guards

Before declaring P(replay(T, S')) = False during minimization, we check:

```
n_true = count of True in k_guard evaluations
ci_lower = Wilson_lower(n_true, k_guard, alpha=0.05)
```

If `ci_lower > threshold` (e.g., 0.3), we flag the predicate as "unstable
stochastic" and report it to the caller rather than treating it as a Hard False.

The Wilson confidence interval is:

```
p̂ = n_true / k
z = 1.96  # 95% CI
Wilson_lower = (p̂ + z²/(2k) - z√(p̂(1-p̂)/k + z²/(4k²))) / (1 + z²/k)
```

---

## 3. Stochastic extensions to the minimization toolkit

### 3.1 Repeated dirty-step re-execution

When `stochastic_mode=True` and the predicate is stable-stochastic:

1. After computing the dirty set D for S', replay T with S' `k_repeats` times,
   using a fresh executor RNG seed each time.
2. Report `True` if majority-vote over k_repeats runs is True; report `False`
   otherwise.
3. Append `flaky_count` (number of times the result differed from majority) to
   the minimization trace.

Cost: each call to the oracle now costs `O(k_repeats × |D|)` LLM calls instead
of `O(|D|)`. For stable-stochastic predicates with flip rate 0.1 and k_repeats=5,
the error rate under majority vote falls from 10% to ~0.8%.

### 3.2 Stochastic predicate stability metrics

`stepback/minimize.py` exposes `StochasticPredicateStats` (proposed addition):

```python
@dataclass
class StochasticPredicateStats:
    evaluations: int           # total oracle calls made
    flips: int                 # times result differed from previous evaluation
    flip_rate: float           # flips / max(evaluations - 1, 1)
    category: str              # "deterministic" | "stable" | "unstable"
    wilson_lower: float        # Wilson CI lower bound for final True rate
    wilson_upper: float        # Wilson CI upper bound for final True rate
```

### 3.3 Stochastic Shapley attribution

The deterministic Shapley implementation (`ShapleyAttributionStrategy`)
evaluates each coalition exactly once, which is valid under A\_oracle. For
stochastic oracles, we instead:

1. Sample `n_samples` random coalitions from the 2ⁿ powerset (stratified
   by coalition size).
2. For each coalition, replay with a fixed seed pool and record the mean
   outcome over `k_samples` evaluations.
3. Compute expected marginal contribution for each substitution using the
   sampled coalitions.

The sampling estimator is unbiased; variance decreases as O(1/√n\_samples).
For |S| ≤ 20 and n\_samples = 5 000, the 95% CI on each weight is typically
< 0.05.

---

## 4. Predicate stability vs. replay correctness

A critical separation (tested in `tests/test_stochastic_replay.py`):

**Replay correctness** (sound dirty-set caching) is *invariant* under executor
noise. The dirty-set set, cache-hit count, real-executions count, and upstream
cached-step outputs are all pure functions of the recorded trace and staged
substitutions. They do not depend on whether the executor is seeded or unseeded.

**Predicate stability** is a *user-side* concern. A predicate that inspects
LLM output text will flip when the LLM output changes, even if the dirty-set
computation is correct. The replay engine cannot make content predicates stable;
it can only make *structural* predicates (dirty count, step kind, cost bounds)
stable.

This separation is pinned by two test categories in `tests/test_stochastic_replay.py`:

1. **Correctness invariants**: same dirty-set, same cache-hit count, same
   upstream outputs regardless of executor seed.
2. **Stability demonstrations**: structural predicates are stable across seeds;
   content predicates flip; the confidence interval width shrinks under more
   re-executions.

---

## 5. Implementation status

| Feature | Status | Evidence |
|---------|--------|----------|
| DDMin, binary halving, Shapley (deterministic) | **Implemented** | `docs/minimization-paper.md`, `tests/test_minimize*.py` |
| Replay-correctness invariance under executor noise | **Implemented** | `tests/test_stochastic_replay.py` (15 tests) |
| Stochastic predicate classification framework | **Prototype** | Demonstrated in `tests/test_stochastic_replay.py`; `StochasticPredicateStats` not yet a public API. |
| Wilson confidence-interval guards | **Prototype** | Math in test file; not yet integrated into `MinimizationOptions`. |
| Repeated dirty-step re-execution (k_repeats) | **Prototype** | Demonstrated in tests; not yet a `MinimizationOptions` flag. |
| Stochastic Shapley estimation | **Spec-only** | Algorithm described here; deterministic version is implemented. |

---

## 6. Evaluation

### 6.1 Correctness under noise

For 15 parametrized test cases across 5 seeds × 3 substitution types × 1 corpus:

- Dirty-set size is identical under all seeds: 100% stable.
- Cache-hit count is identical under all seeds: 100% stable.
- Upstream cached-step outputs are byte-identical under all seeds: 100% stable.
- Only dirty-step output content differs: 100% of dirty-step content varies
  when the executor is unseeded.

See `tests/test_stochastic_replay.py::test_dirty_set_identical_under_all_seeds`.

### 6.2 Predicate stability

For a structural predicate ("dirty count == 1"):

- True for all 10 seeds tested. Stability: 100%.

For a content predicate ("output contains token X"):

- True for ~60% of seeds (seeded distribution). Stability: 60%.
- Wilson 95% CI: [0.31, 0.83] at k=10, narrows to [0.51, 0.69] at k=100.

See `tests/test_stochastic_replay.py::test_content_predicate_wilson_ci_shrinks`.

---

## 7. Limitations

1. **k\_repeats cost**: Stability verification multiplies oracle cost by
   k\_repeats. For expensive LLM steps ($1/call), k=5 raises the cost
   from $5 to $25 per oracle invocation.
2. **Monotonicity failure**: ddmin's 1-minimality guarantee requires a
   monotone oracle. Stochastic predicates are not monotone; the guarantees
   from `docs/minimization-paper.md` do not transfer without the majority-vote
   wrapper.
3. **Provider non-determinism**: Some LLM providers do not guarantee
   temperature-0 determinism across API versions. Replaying with `seed=42`
   may still produce different outputs after a provider model update.
4. **Stochastic Shapley is not yet public API**: The sampling estimator is
   described here but not yet wired into `ShapleyAttributionStrategy`.

---

## 8. Related work

- Zeller (1999) ddmin: assumes deterministic oracle; see §1.
- Klees et al. (2018) "Evaluating Fuzz Testing": discusses predicate
  instability in fuzz campaigns; motivates statistical evaluation.
- [Wilson score interval (Wilson 1927)](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval):
  the CI formula used for oracle stability guards.
- See [`RELATED_WORK.md`](../RELATED_WORK.md) §2.4 for the full
  delta-debugging comparison table.

---

## 9. Conclusion

Stochastic LLM predicates introduce flakiness that makes classical ddmin
unreliable: it may terminate early on a flaky False or miss low-probability
causes. The extensions described here — confidence-interval guards, stability
classification, repeated re-execution, and stochastic Shapley — address each
failure mode at the cost of additional oracle calls. The replay-correctness
invariant (the dirty set and cached outputs are unaffected by executor noise)
is the foundation that makes these extensions composable: stability verification
is purely a predicate-evaluation policy that rides on top of a sound replay engine.
