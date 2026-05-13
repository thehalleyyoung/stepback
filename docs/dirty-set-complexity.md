# Asymptotic complexity bounds for the dirty-set algorithm (v1)

This document is the formal companion to §7 ("Asymptotic complexity") of
[`docs/dirty-set.md`](./dirty-set.md) and to the *Complexity* block of
the contract in
[`stepback/divergence.py`](../stepback/divergence.py). It states and
proves the worst-case time and memory bounds of the v1 dirty-set
classifier and replay engine on three trace shapes — *linear*, *general
DAG*, and *branch-heavy* — and pins those bounds to specific lines in
[`stepback/replay.py::Trace.run_replay`](../stepback/replay.py).

Step 58 of [`100_STEPS.md`](../100_STEPS.md) tracks this analysis.
This document is part of `dirty_set_version="1"` and is subject to
the same versioning policy as [`docs/dirty-set.md`](./dirty-set.md) §8.

> **One-line summary.** On every trace shape v1 admits, the classifier
> runs in `Θ(N + E + Σ_s |s.inputs|)` time and `Θ(N + Σ_s |s.outputs|)`
> peak heap, performs at most `D_real ≤ |D|` real executor calls, and
> emits at most `|D| − D_real` cache-pin writes. There is no shape on
> which the algorithm is asymptotically worse than full re-execution.

## 0. Notation

We reuse the symbols of [`docs/dirty-set.md`](./dirty-set.md) §1–§5.

| Symbol | Meaning |
|---|---|
| `T` | A v1 `.sb` trace as parsed by `read_trace`. |
| `G(T) = (V, E)` | The recorded parent-edge DAG of `T`. |
| `N` | `|V|` — number of recorded `step` frames. |
| `E` | `|E|` — number of recorded parent edges. For single-parent steps, the parent contributes one edge; for `parallel_branch_join` steps, every entry in `parent_step_ids` contributes one edge. |
| `w(s)` | Branch fan-in of step `s`: `len(s.parent_step_ids)` for joins, `1` for single-parent steps, `0` for roots. By construction `Σ_s w(s) = E`. |
| `W` | `max_s w(s)` — the maximum join width occurring in `T`. `W ≤ N − 1`. |
| `B` | Maximum branch fan-out: `max_p |{s : p ∈ parents(s)}|`. `B ≤ N − 1`. |
| `I_s` | `|J(s.inputs)|` — the canonical-JSON byte length of step `s`'s recorded inputs. |
| `O_s` | `|J(s.outputs)|` — the canonical-JSON byte length of step `s`'s recorded outputs. |
| `I = Σ_s I_s` | Total recorded-input bytes across the trace. |
| `O = Σ_s O_s` | Total recorded-output bytes across the trace. |
| `σ` | The substitution set; `m = |σ|`. |
| `D = D(T, σ)` | The dirty set; `|D| ≤ N`. |
| `D_real` | Subset of `D` that is *not* pinned by an output-forcing substitution. `D_real ≤ |D|`. |
| `H_step` | Cost of one BLAKE2b-256 hash of canonical-JSON bytes — `Θ(byte length)` of the hashed object. |
| `J_step` | Cost of canonicalizing one JSON object — `Θ(size + size · log size)` worst case (sort by key); see [`docs/canonicalization.md`](./canonicalization.md). |
| `X` | Cost of one executor call. Treated as a black box; bounded externally and by A2 of the contract block. |

We write `f = Θ(g)` for tight asymptotic bounds, `f = O(g)` for
upper bounds, and `f = Ω(g)` for lower bounds, all with the standard
hidden constants.

We assume A1–A3 of [`docs/dirty-set.md`](./dirty-set.md) §6, R1–R4 of
the recorder contract, and a reader that has already validated the
trace via `verify_trace` (so cycles, duplicate `step_id`s, and
out-of-order parent edges are excluded as preconditions).

## 1. The reference loop

The reference implementation is the topological walk in
[`stepback/replay.py::Trace.run_replay`](../stepback/replay.py)
(lines 300–408 in HEAD). Per iteration, for each `s ∈ V` in recorded
order, the engine performs:

1. **Input rebinding.** Rewrite `s.inputs["context"]` (if present) and
   `s.inputs["branch_tail_hashes"]` (if present) in place against
   `outputs_hash_by_id` for `s`'s declared parents. This touches at
   most `1 + w(s)` map entries — single-parent + branch-tail count.
2. **Substitution application.** If any `σ_i` targets `s`, mutate
   `cur_inputs` (input-mutating substitutions) or stash a pinned
   output (output-forcing substitutions). Each substitution is matched
   against `s.step_id` in `O(1)` via the precomputed
   `sub_set._by_target` index.
3. **Hash and classify.** Compute `current_hash(s, out)` over the
   rebound inputs — one canonicalize-and-hash pass of cost
   `Θ(I_s + I_s · log I_s)` worst case (sorting keys), and one
   BLAKE2b-256 pass of cost `Θ(I_s)`.
4. **Reuse or recompute.** If clean, copy `s.outputs` into
   `outputs_by_id[s.step_id]` and store `H(s.outputs)` in
   `outputs_hash_by_id[s.step_id]`. If dirty and not output-forced,
   call `executor.execute(s.kind, cur_inputs)` for cost `X`. Either
   way, hash the resulting outputs once for downstream rebinding —
   `Θ(O_s)`.

Steps 1–4 are executed exactly once per `s ∈ V`. There is no nested
loop over the trace. Substitution lookup is `O(1)` per step. Parent
lookup is `O(w(s))` per step.

## 2. Master theorem for v1 dirty-set classification

**Theorem 2.1 (Time complexity, classification).** *Under R1–R4 and
assuming `verify_trace` has accepted `T`, computing `D(T, σ)` takes*

```
T_classify(T, σ) = Θ(N + E + I + O + m)
```

*time, where the hidden constant is dominated by canonical-JSON sort
and BLAKE2b-256 hashing.*

**Proof.**

- *Lower bound `Ω(N + E + I + O)`.* The classifier must touch every
  recorded step at least once to decide its dirty/clean status (else
  it has no information about that step), so `Ω(N)` is necessary. Every
  parent edge participates in a hash-rebind (`"context"` or a
  `branch_tail_hashes` slot); skipping an edge can change a join's
  `current_hash` and thus its dirty classification, so `Ω(E)` is
  necessary in the adversarial case. Every recorded input must be
  re-canonicalized to compute `current_hash`, so `Ω(I)`. Every recorded
  output that is consumed by some descendant must be hashed once for
  rebinding (the `outputs_hash_by_id[s.step_id] = hash_obj(cur_outputs)`
  line at `replay.py:408`), so `Ω(O)`. The bound is `Ω(m)` because every
  substitution must be at least matched against `target_id`.

- *Upper bound `O(N + E + I + O + m)`.* By inspection of §1, each step
  performs `O(1)` map operations, `O(w(s))` parent-edge work, one
  canonicalize-and-hash of cost `O(I_s)`, and one output-hash of cost
  `O(O_s)`. Summing over all `s ∈ V`:

  ```
  Σ_s [O(1) + O(w(s)) + O(I_s) + O(O_s)]
    = O(N) + O(Σ_s w(s)) + O(Σ_s I_s) + O(Σ_s O_s)
    = O(N + E + I + O).
  ```

  Substitution registration is `O(m)` for the precomputed index
  (`SubstitutionSet.add` is `O(1)`, applied `m` times). No other
  per-step cost has a hidden factor of `N` or `E`. ∎

**Corollary 2.2 (Time complexity, classification + replay).** *The
total time including dirty re-execution is*

```
T_total(T, σ) = Θ(N + E + I + O + m + D_real · X)
```

*where `X` is the per-call executor cost.* This is tight because the
`D_real` real executor calls are each `Θ(X)` by definition and the
classifier work is the additive `Θ(N + E + I + O + m)` of Theorem 2.1.

**Theorem 2.3 (Memory complexity).** *Peak heap usage of the
classifier is*

```
M(T, σ) = Θ(N + O + S_max + m)
```

*where `S_max = max_s I_s + max_s O_s` is the largest single-step
input+output payload and `m` is the substitution count. Equivalently
`Θ(N + O + max_s (I_s + O_s) + m)`.*

**Proof.** The two persistent maps `outputs_by_id` and
`outputs_hash_by_id` hold `N` keys each; the value map holds outputs
totalling `O` bytes (alias-shared with the recorded trace, but counted
distinctly in the worst case where `Trace` was streamed and outputs
have been copied for substitution safety); the hash map holds `N`
fixed-32-byte digests, `Θ(N)`. The dirty-set bookkeeping adds two
`Dict[str, bool|str]` of size `N`. The substitution index holds `m`
entries. Per-iteration scratch (`cur_inputs`, `cur_outputs`,
`current_hash` bytes) is bounded by the largest single step
`S_max`, never accumulating across steps because the `for rec in
self.recorded_steps:` loop releases its locals on each turn. The total
is `Θ(N + O + S_max + m)`. ∎

In practice `O = O(N · O_avg)` and `S_max ≪ O`, so this collapses to
the headline bound in [`docs/dirty-set.md`](./dirty-set.md) §7:
`O(N)` memory for the bookkeeping plus the `Trace` itself.

## 3. Linear traces

A *linear* trace has `E = N − 1`, every non-root step has exactly one
parent, and the parent of `s_k` is `s_{k−1}`. This is the shape of
the 12-step `tests/fixtures/agent.py` fixture and of every
single-thread agent that does not fan out.

### 3.1 Time

Substituting `E = N − 1` into Theorem 2.1:

```
T_classify_linear = Θ(N + I + O + m).
```

The `O(E)` term is absorbed into `O(N)` because `E < N`. If inputs
and outputs are bounded (`I_s, O_s ≤ const`), this further collapses
to `Θ(N + m)`.

### 3.2 Dirty-set size

Dirty-set propagation is monotone in topological order. On a linear
trace, the dirty set is an upward-closed prefix of the suffix starting
at the earliest dirty step:

> **Lemma 3.1.** *On a linear trace `T`, `D(T, σ) = {s_k, s_{k+1}, …,
> s_N}` where `k` is the smallest index for which classify reports
> DIRTY (or `D = ∅` if no such `k` exists), modulo isolated `ndh_tamper`
> dirties.*

**Proof.** By the parent-dirty closure (`classify` line `if any
dirty(parents): DIRTY`) and the fact that on a linear trace `s_{j+1}`
has exactly one parent `s_j`, dirtiness propagates strictly forward
once kindled. The only exception is `ndh_tamper`, which can fire at
any isolated step independent of parents; the *minimal* `k` may be
overridden upward only if every earlier step is `ndh_tamper`-clean.
∎

**Corollary 3.2.** *Under a single input-mutating substitution at the
root of a linear trace, `|D| = N` and `D_real = N`.* This is exactly
the `dirty_after_sub=11` shape of the 12-step bench fixture (one
forced/substituted root + 11 propagated descendants), reconciled in
[`docs/dirty-set.md`](./dirty-set.md) §9.

### 3.3 Memory

Linear traces have `S_max = max_s (I_s + O_s) = O(I_avg + O_avg)`.
Theorem 2.3 collapses to `Θ(N + O + m)`, dominated by the recorded
`outputs_by_id` map.

### 3.4 Tightness

The lower bound `Ω(N + I + O)` holds on any linear trace because the
classifier must read every recorded input and output at least once
(see Theorem 2.1 lower-bound argument). Therefore the linear-shape
bound is tight up to constants.

## 4. General DAG traces

A *general DAG* trace permits arbitrary in-degree subject to
acyclicity and topological ordering of recorded frames. This shape
covers `parallel_branch_open` / `parallel_branch_join` patterns,
mid-trace router fan-outs that re-converge, and tool-graph composition
patterns where a single retrieval is consumed by multiple downstream
LLM calls.

### 4.1 Time

Theorem 2.1 already states the general DAG bound directly:

```
T_classify_dag = Θ(N + E + I + O + m).
```

The `Θ(E)` term is non-negligible: a DAG can have `E = Θ(N²)` (every
step a join over all earlier steps), in which case the classifier
work is `Θ(N²)` even before considering input/output bytes. v1
recorders may emit such DAGs only if they declare the parent edges
explicitly via `parent_step_ids`; if they do, the cost is paid
honestly.

### 4.2 Dirty-set size

DAG topology gives the algorithm room to *preserve* clean siblings
that were not reachable from a substitution target.

> **Lemma 4.1 (Reachability bound).** *Let `R(σ) = ⋃_i Reach(σ_i.target_id)`
> denote the set of forward-reachable descendants in `G(T)` of the
> substituted targets. Then*
>
> ```
> {targets of σ} ∪ R(σ) ⊆ D(T, σ) ⊆ V.
> ```

The lower-bound containment is from P3 (substituted) and the
parent-dirty closure of P2; the upper bound is trivial. If the
substitution targets a step in only one of `k` parallel branches,
the other `k − 1` branches have intersection-empty reachability with
`R(σ)` and remain entirely clean. This is precisely where the
"counterfactual costs `O(|D|)` instead of `O(N)`" headline pays off
on real workloads: the median dirty set on branch-rich DAGs is `≪ N`.

### 4.3 Memory

`Θ(N + O + S_max + m)` from Theorem 2.3 is unchanged. Note that on
DAGs the `outputs_hash_by_id` map cannot be discarded after a step's
last child is processed without an additional reference-count pass;
v1 conservatively keeps it for the full traversal. Step 60 of
[`100_STEPS.md`](../100_STEPS.md) (partial recompute) is the future
work that would tighten this to `Θ(width(G) · O_avg + m)`.

### 4.4 Tightness

The `Ω(N + E)` lower bound is also tight on general DAGs: an
adversary can construct a trace whose classification depends on
exactly `E` rebind operations (set every join's `branch_tail_hashes`
to be derived from a different parent), forcing the algorithm to
visit every edge at least once.

## 5. Branch-heavy traces

A *branch-heavy* trace is a DAG where the maximum join width `W` is
non-trivially large or the maximum fan-out `B` dominates: i.e., it
makes sense to track per-step parameters separately from `N` and `E`.
Real examples: a retrieval-augmented agent that fans out 50 parallel
retriever queries and joins them, or an evaluation harness that
spawns 1,000 sandbox replicas before a single aggregation step.

### 5.1 Time

Theorem 2.1 still bounds the worst case, but it pays to expose the
join-width factor explicitly:

```
T_classify_branch = Θ(N + E + I + O + m)
                  = Θ(N + Σ_s w(s) + Σ_s I_s + Σ_s O_s + m).
```

The classifier visits each `s` once, hashes each branch tail of `s`
once (cost `Θ(w(s) + I_s)` when rebinding `branch_tail_hashes` against
`outputs_hash_by_id`; see `replay.py:325–335`), and never revisits a
prior parent's hash beyond the single `outputs_hash_by_id` lookup. On
a single join of width `W`, the per-step work is `Θ(W + I_s + O_s)`,
not `Θ(W²)` or `Θ(W · N)`.

### 5.2 Dirty-set size

Branch-heavy traces have the most favourable dirty-set distribution
because parallel children are independently dirty-able. Step 59 of
[`100_STEPS.md`](../100_STEPS.md) implements this: dirty fan-out
children independently, dirty joins iff *any* consumed branch output
changed, preserve clean siblings.

> **Lemma 5.1 (Fan-out independence).** *Let `s_open` be a
> `parallel_branch_open` step with children `c_1, …, c_B` in disjoint
> sub-DAGs, and let `c_join` be the corresponding
> `parallel_branch_join`. Then for any substitution `σ` whose targets
> all lie within sub-DAG `c_i`,*
>
> ```
> D(T, σ) ∩ (sub-DAG of c_j) = ∅      for every j ≠ i,
> ```
>
> *and `c_join ∈ D(T, σ)` iff the `c_i`-sub-DAG produces a different
> output hash than the recorded one.*

**Proof.** Sub-DAGs of distinct fan-out children are vertex-disjoint
by R4 (DAG topology) and by the recorder's R2 obligation that
cross-branch dependencies be declared via `branch_tails`. The dirty
walk on the `j ≠ i` sub-DAG sees no dirty parent and no input drift,
so every step there classifies CLEAN. The join `c_join` declares
`branch_tails = [c_1, …, c_B]`; rebinding the `c_i`-slot hash and
recomputing `current_hash(c_join, out)` is what triggers the join's
own classification — DIRTY iff that one hash changed. ∎

### 5.3 Memory

`Θ(N + O + S_max + m)`. Note `S_max` may grow with `W` if a join's
`inputs` payload concatenates `W` branch summaries verbatim, but
`S_max` does not multiply: the loop releases per-iteration scratch.

### 5.4 Worked numbers

For the planned `bench replay-caching` corpus
(see [`100_STEPS.md`](../100_STEPS.md) §"Benchmark") — a synthetic
agent with `N = 200` linear-then-branch steps, `B = 4` fan-out
factor, `W = 4` join width, `I_s, O_s ≤ 4 KiB`, single-step
prompt substitution at a non-root branch interior:

| Quantity | Linear-only | This DAG |
|---|---|---|
| `N` | 200 | 200 |
| `E` | 199 | ≈ 250 |
| Hash work | `Θ(200 · 4 KiB)` | `Θ(250 · 4 KiB)` |
| `D` (substitution at sub-DAG of `c_2`) | 200 (entire suffix) | ≈ 50 (sub-DAG only + join + suffix) |
| `D_real` | 200 | ≈ 50 |
| Cost reduction vs. full recompute | 0× | ≈ 4× |

The exact distribution is the deliverable of Step 66
(*Publish empirical dirty-set distributions over synthetic fixtures,
public corpora, anonymized production traces*). The algorithmic
ceiling is set by this section: 4× on a `B = 4` corpus is consistent
with `|D|/N ≈ 1/B` on the dirty branch.

## 6. Substitution-set scaling

The substitution set is matched once per step in `O(1)` against an
index built in `O(m)` time. There is no `N · m` cross product:

```
T_substitution = Θ(m + N).
```

Output-forcing substitutions further reduce `D_real`: each pin is one
fewer executor call. The minimum is

```
D_real ≥ |D ∖ outputs_forced(σ)|
```

with equality when every dirty step that is not an output-forcing
target requires a real call. There is no v1 mechanism for "implicit
recompute deferral" — every dirty `D_real` step is a real `X`-cost
call.

## 7. Streaming readers and very large traces

Traces too large to fit in memory are handled by the streaming reader
(see Step 50 of [`100_STEPS.md`](../100_STEPS.md) and
[`spec/sbtrace-v1.md`](../spec/sbtrace-v1.md) §reader-limits).
Streaming changes only the `Trace` parse cost, not the classifier
bound: classification still requires touching every step once,
because `current_hash(s, out)` depends on prior outputs, and a
streaming DAG classifier still keeps `outputs_hash_by_id` in memory
for all live ancestors.

> **Lemma 7.1 (Streaming memory floor).** *A streaming v1 dirty-set
> classifier requires `Ω(width(G) · max_p O_p + S_max)` memory, where
> `width(G)` is the antichain width of `G(T)`.* This is the trivial
> reachability bound on the live ancestor set; v1's `Θ(N + O + S_max +
> m)` figure is the relaxed in-memory implementation. The width-aware
> bound is future work tracked in Step 60.

## 8. Putting it together

The headline performance claim of stepback —
*"counterfactual debugging of an N-step agent trace costs `O(|D|)`
LLM calls instead of `O(N)`"* — decomposes into two algorithmic
guarantees that this document proves:

1. **Classifier overhead is sublinear in executor cost.** The
   classifier work is `Θ(N + E + I + O + m)` *bytes-of-bookkeeping*,
   plus only `D_real` `Θ(X)`-cost real executor calls. For real LLM
   workloads, `X ∼ 100 ms–10 s` per call dominates `N + E + I + O`
   by 4–6 orders of magnitude. Replay overhead is, in practice,
   negligible.
2. **Dirty-set propagation is structurally minimal.** Lemmas 3.1, 4.1,
   5.1 show that the dirty set is exactly `targets(σ) ∪
   forward-reach(σ) ∪ ndh_tamper`. There is no over-dirtying clause
   in the classifier (P5 of [`docs/dirty-set.md`](./dirty-set.md) §6).
   On branch-rich DAGs the median dirty set is therefore `O(|D|)`,
   not `O(N)`.

These two facts jointly establish that, on every trace shape v1
admits, replaying with `compute_dirty_set` is asymptotically optimal
up to the recorder-declared dependency model A1.

## 9. Cross-references

* [`stepback/divergence.py`](../stepback/divergence.py)
  *§Complexity* — three-line summary that this document expands.
* [`stepback/replay.py::Trace.run_replay`](../stepback/replay.py)
  lines 300–408 — the loop whose per-iteration cost is the basis of
  Theorem 2.1.
* [`docs/dirty-set.md`](./dirty-set.md) §7 — lay-summary table of
  these bounds.
* [`docs/dirty-set-soundness.md`](./dirty-set-soundness.md) — proof
  of P1 (correctness of cache reuse) for the same loop.
* [`docs/dirty-set-completeness.md`](./dirty-set-completeness.md) —
  proof of P2 (completeness of dirtying).
* [`docs/canonicalization.md`](./canonicalization.md) — the cost
  model for `J_step` (canonicalize) and `H_step` (hash).
* [`100_STEPS.md`](../100_STEPS.md) Steps 58–61 — the roadmap of
  follow-on work that builds on these bounds (branch-aware
  propagation, partial recompute, stale-cache detection,
  width-aware streaming memory).
