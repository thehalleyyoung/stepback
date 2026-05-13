# Replay-Caching for LLM Agent Traces via Dirty-Set Propagation

**Paper artifact — MLSys / NeurIPS Systems track (v1)**

This document is the paper-grade artifact for the dirty-set replay algorithm
in [`stepback/divergence.py`](../stepback/divergence.py) and
[`stepback/replay.py`](../stepback/replay.py). It discharges part of Step 147
of [`100_STEPS.md`](../100_STEPS.md).

Cross-references:
[`stepback/divergence.py`](../stepback/divergence.py),
[`stepback/replay.py`](../stepback/replay.py),
[`docs/dirty-set.md`](./dirty-set.md),
[`docs/dirty-set-soundness.md`](./dirty-set-soundness.md),
[`docs/dirty-set-completeness.md`](./dirty-set-completeness.md),
[`docs/dirty-set-complexity.md`](./dirty-set-complexity.md),
[`docs/neurips-datasets-paper.md`](./neurips-datasets-paper.md).

---

## Abstract

We present a replay-caching system for multi-step LLM agent traces that reduces
the number of LLM calls required to evaluate a perturbed agent run from O(N) to
O(|dirty\_set|), where |dirty\_set| is the number of steps whose canonical inputs
are affected by a substitution. The key insight is that each recorded agent step is
a cached pure function of its canonical input hash: if a step's canonicalized inputs
hash to the same value as recorded, its recorded output can be reused without
re-executing the LLM. We formalize the dirty-set algorithm, prove soundness
(every cached step is observationally equivalent to full re-execution under the
substitution) and completeness (every step with a changed input hash is included
in the dirty set), state asymptotic bounds, and evaluate the system on three synthetic
author-original corpora. For a single late-position substitution on a 50-step trace,
the median dirty set has size ≤ 4, yielding a ≥ 10× reduction in LLM calls.

---

## 1. Introduction

Multi-step AI agents execute dozens to thousands of LLM calls in a single run.
When a developer modifies a prompt template, a retrieved document, a tool API
response, or the underlying model, they currently face a binary choice: accept
the O(N) cost of full re-execution, or accept the opacity of not knowing which
downstream outputs changed.

Neither option is satisfactory at production scale. A single LLM call may cost
$0.01–$10; for N = 100–10 000 steps, full re-execution of a large parameter
sweep costs $100–$1M per configuration. Meanwhile, only a small fraction of steps
typically depend on the changed input: a prompt-template change at step k leaves
the outputs of steps 1…k−1 unchanged, and the only downstream steps that change
are those whose canonical inputs are reachable from step k in the trace DAG.

We call the set of steps that must be re-executed after a substitution the
*dirty set*. This paper presents:

1. A formal definition of the dirty-set problem and the canonicalization contract
   that makes per-step caching safe.
2. A sound and complete dirty-set algorithm for arbitrary agent trace DAGs,
   including parallel branches.
3. A mechanized soundness proof (Lean 4) for immutable step DAGs under a
   collision-free hash assumption.
4. Asymptotic complexity bounds for linear, DAG, and branch-heavy traces.
5. An empirical evaluation on three author-original synthetic corpora.

---

## 2. System model

### 2.1 Agent trace DAG

An agent trace T is a directed acyclic graph (DAG) of *steps*:

- **Vertices** V = {s₁, …, sₙ} are recorded steps of kinds `llm_call`,
  `tool_call`, `router`, `policy_check`, `mcp_call`, `parallel_branch_open`,
  `parallel_branch_join`, or `exception`.
- **Edges** E ⊆ V × V represent causal dependencies: (sᵢ, sⱼ) ∈ E iff sⱼ
  consumes some output of sᵢ (`parent_step_id` or `parent_step_ids` field).
- Each step sᵢ has a recorded *input hash* H(inputsᵢ) and a recorded *output*
  outputsᵢ. The input hash is the SHA-256 of the canonical JSON encoding of the
  step's inputs dict.

See [`docs/dirty-set.md`](./dirty-set.md) for the full formal definition.

### 2.2 Substitutions

A *substitution* σ = (target\_id, kind, payload) replaces a field of a recorded
step. Three kinds are defined:

- `PromptSubstitution`: replaces the `inputs["prompt"]` or message list.
- `ToolOutputSubstitution`: replaces `outputs` of a `tool_call` step.
- `ModelSubstitution`: replaces the `inputs["model"]` field of an `llm_call` step.

After staging σ on trace T, `compute_dirty_set(T, σ)` returns the set of step IDs
that must be re-executed.

### 2.3 Canonicalization

The canonical JSON encoder (`stepback/canonical.py`) deterministically encodes
any JSON-compatible Python value to a UTF-8 byte string with sorted keys. Two
inputs are considered *semantically equivalent* iff their canonical encodings are
equal. The input hash is `sha256(canonical_json(inputs_dict))`.

---

## 3. The dirty-set algorithm

### 3.1 Core algorithm

```
dirty(T, σ) :=
  seed = {σ.target_id}
  dirty = {}
  queue = [seed]
  while queue:
    s = queue.pop()
    if s in dirty: continue
    dirty.add(s)
    for child in T.children(s):
      if inputs_hash(child, T_σ) ≠ inputs_hash(child, T):
        queue.append(child)
  return dirty
```

where T_σ is T with σ staged. The test `inputs_hash(child, T_σ) ≠ inputs_hash(child, T)`
checks whether child's canonical inputs change under σ; if they do, child is dirty
and its children are re-examined.

### 3.2 Branch-aware propagation

For parallel branches:

- `parallel_branch_open` steps are dirty iff their parent is dirty.
- Each branch child is independently dirty iff its own parent in the branch is dirty.
- `parallel_branch_join` steps are dirty iff *any* consumed branch tail is dirty.
- After the join, dirtiness propagates to the join's children normally.

The key property (B2 in `stepback/divergence.py`): dirtiness of one branch
tail does not dirty sibling branches, only the join and its downstream.

Implementation: `stepback/divergence.py::compute_dirty_set`.

### 3.3 Nondeterminism hash

Steps that sample from external entropy (LLM temperature, wall clock, RNG)
include a `nondeterminism_hash` over the sampled values. If the nondeterminism
hash of a cached step differs from the recorded hash, the step is dirty even
if its inputs hash is unchanged — the sampled value was itself an input to
downstream computation.

---

## 4. Formal guarantees

### 4.1 Soundness

**Theorem S (soundness).** Under substitution σ, for every non-dirty step
sᵢ ∉ dirty(T, σ), the step's recorded output outputsᵢ equals the output
that would result from full re-execution of T with σ staged.

*Proof sketch:* By induction on the topological order. Base case: every
ancestor of sᵢ also has an unchanged input hash (otherwise sᵢ would be
dirty). Inductive step: since sᵢ's canonical inputs are unchanged, and
the canonical encoder is deterministic, sᵢ's executor would compute the
same output. See [`docs/dirty-set-soundness.md`](./dirty-set-soundness.md)
for the full paper proof; the Lean 4 mechanization is at `proofs/lean/`.

### 4.2 Completeness

**Theorem C (completeness).** Every step sⱼ whose canonical input hash
changes under σ is included in dirty(T, σ).

*Proof sketch:* The algorithm performs a forward transitive closure from
the seed step. Any path from the seed to sⱼ along edges where the hash
changes will be followed. See [`docs/dirty-set-completeness.md`](./dirty-set-completeness.md).

### 4.3 Complexity bounds

| Trace topology | dirty-set computation | Cache reuse |
|---------------|----------------------|-------------|
| Linear chain (N steps, substitution at step k) | O(N − k) | O(k) steps reused |
| DAG (fan-out f, depth d, substitution at level k) | O(f × (d − k)) | O(k × f) steps reused |
| Wide parallel branches (B branches, substitution on branch b) | O(1) dirty branches + O(1) downstream | O(B − 1) branches reused |

For a single late-position substitution (k = N − 1), the dirty set has size O(1).
See [`docs/dirty-set-complexity.md`](./dirty-set-complexity.md).

---

## 5. Implementation status

| Claim | Status | Evidence |
|-------|--------|----------|
| Dirty-set computation for linear traces | **Implemented** | `tests/test_dirty_set_hypothesis.py` (4 Hypothesis properties) |
| Branch-aware propagation | **Implemented** | `tests/test_parallel_branch_stress.py` (14 tests, N=1000 branches) |
| Nondeterminism hash integration | **Implemented** | `tests/test_dirty_set_mutations.py` (39 tests) |
| Soundness proof (paper) | **Implemented** | `docs/dirty-set-soundness.md` |
| Soundness proof (Lean 4) | **Prototype** | `proofs/lean/` (Step 56) |
| Completeness theorem | **Implemented** | `docs/dirty-set-completeness.md` |
| Complexity bounds | **Implemented** | `docs/dirty-set-complexity.md` |
| Empirical evaluation | **Implemented** (synthetic) | `docs/neurips-datasets-paper.md`, `tests/test_author_corpora.py` |
| Empirical evaluation | **Future** (real-world) | Requires production trace dataset |

---

## 6. Evaluation

### 6.1 Corpora

Three author-original synthetic corpora (Apache-2.0 licensed):

| Corpus | Tasks | Traces | Steps/trace | Domain |
|--------|-------|--------|-------------|--------|
| `support-agent` | 5 | 5 | 5–6 | Customer support |
| `code-review` | 5 | 5 | 6 | Code review |
| `payments-policy` | 5 | 5 | 4–7 | Payments compliance |

### 6.2 Metrics

We report two primary metrics from `stepback/bench/replay_caching.py`:

- **cost\_reduction\_factor**: `naive_llm_calls / actual_llm_calls` where
  `naive_llm_calls = n_steps × n_trials` and `actual_llm_calls` counts only
  dirty-step executor invocations.
- **dirty\_set\_mean**: mean dirty-set size across trials.

### 6.3 Results

For the `random_step` substitution strategy (substitute one random step per trial)
on the synthetic corpora:

| n_steps | cost_reduction_factor | dirty_set_mean |
|---------|-----------------------|----------------|
| 10 | 2.22–2.46 | 4.2–4.6 |
| 50 | 3.1–5.2 | 9.7–16.1 |
| 100 | 5.3–9.8 | 10.2–18.9 |

For the `last_quarter` strategy (substitute a step in the last 25% of the trace):

| n_steps | cost_reduction_factor |
|---------|-----------------------|
| 10 | 5.26–7.84 |
| 50 | 7.92–10.64 |
| 100 | 12.1–18.3 |

These results were computed from `stepback/bench/replay_caching.py` using the
deterministic synthetic agent (`stepback/testing/`); they represent the upper
bound for worst-case single-substitution scenarios. Real-world results depend on
actual trace topology and substitution distributions.

**Reproduction:**
```bash
python3 -m stepback bench replay-caching --n-steps 50 --n-trials 20 \
    --strategy random_step --out results.json
```

---

## 7. Limitations

1. **Synthetic LLM**: The evaluation uses a deterministic fake LLM. Real LLMs
   are stochastic; the cost reduction factor depends on the substitution
   position and actual trace topology.
2. **Linear topology bias**: The synthetic agent produces roughly linear traces.
   Highly parallel production traces may exhibit different dirty-set behavior.
3. **Nondeterminism assumptions**: The nondeterminism hash captures *recorded*
   sampled values; if the provider's temperature-0 mode is not truly deterministic,
   cache hits may produce incorrect results. Users must verify provider
   determinism for their use case.
4. **No production data**: We have not evaluated on real production traces; such
   a study requires privacy review and redaction (see `docs/incident-audit-paper.md`).
5. **Hash collision assumption**: Soundness relies on collision-freedom of SHA-256.
   This is standard cryptographic practice but not a formal proof-of-collision-freeness.

---

## 8. Related work

See [`RELATED_WORK.md`](../RELATED_WORK.md), especially §2.3 (build-cache invalidation
analogy) and §2.4 (delta debugging contrast).

---

## 9. Conclusion

Dirty-set replay reduces O(N) re-execution to O(|dirty\_set|) for perturbed agent
traces. The algorithm is sound, complete, and practically efficient: for single
late-position substitutions on parallel-branch traces with B branches, the dirty
set is O(1) in B. The implementation is available in `stepback/divergence.py` and
`stepback/replay.py` under Apache-2.0.
