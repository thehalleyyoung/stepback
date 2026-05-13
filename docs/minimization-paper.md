# Causal Minimization of Substitution Sets — Paper Artifact (v1)

This document is the paper-grade artifact for the causal minimization
toolkit in [`stepback/minimize.py`](../stepback/minimize.py). It
discharges Step 88 of [`100_STEPS.md`](../100_STEPS.md):

> *"Write the minimization paper artifact with algorithm, stochastic
> assumptions, failure modes, and empirical comparison to naive ddmin."*

The document is the prose companion to the implementation and to the
HTML report renderer (`docs/minimize_report.md`). It is intentionally
textual and algorithm-focused. We pin `minimize_version="1"` and
`canonicalisation_version="1"`.

Cross-references: [`stepback/minimize.py`](../stepback/minimize.py),
[`stepback/replay.py`](../stepback/replay.py),
[`docs/dirty-set.md`](./dirty-set.md),
[`docs/dirty-set-soundness.md`](./dirty-set-soundness.md).

---

## 0. Scope and motivation

A *substitution* σᵢ = (target\_id, kind, payload) replaces an
input or output field of a recorded agent step with an alternative
value. A *substitution set* S = {σ₁, …, σₙ} describes a compound
perturbation to the recorded trace T. After staging S and replaying T,
a user-supplied *predicate* P evaluates the replay result R to a
boolean.

The causal minimization problem is:

> **Given** a trace T, a substitution set S such that P(replay(T, S)) =
> True, and an oracle O(S') = P(replay(T, S')) available as a black box,
> **find** a minimal subset M ⊆ S such that O(M) = True.

"Minimal" is defined differently by different strategies:
- **1-minimal**: no single element of M can be removed while keeping
  O(M) = True.
- **Globally minimal**: no proper subset of M triggers the predicate
  (requires enumeration; exponential in |S|).
- **Attributed**: each element is assigned a Shapley weight
  measuring its average marginal contribution to the predicate.

The toolkit is motivated by four concrete use cases:

1. **Failure isolation**: A large set of prompt/tool-output
   substitutions causes a regression. Minimization identifies which
   individual substitutions are responsible, reducing debugging effort
   from O(n) manual probes to O(log n) oracle calls.
2. **Cost estimation**: Users want to know which LLM substitutions drive
   cost increases without running the full set.
3. **Policy auditing**: Compliance teams need the minimal perturbation
   that causes a policy violation, not a large joint perturbation that
   obscures root cause.
4. **Delta debugging for stochastic agents**: Multi-step agents may
   require several interdependent substitutions to reproduce a failure;
   ddmin finds the joint minimal witness.

---

## 1. Formal setup

We use notation from [`docs/dirty-set.md`](./dirty-set.md).

**Trace** T = (V(T), E(T), label) is a DAG of recorded steps with
topological order `topo`. The dirty-set algorithm classifies
V(T) into clean and dirty steps for a given substitution set S;
clean steps are served from cache, dirty steps are re-executed.

**Substitution set** S = {σ₁, …, σₙ} where each σᵢ has a
stable Python object identity for the duration of a minimization run.
The elements are treated as atomic (indivisible); partial application of
a single substitution is not supported.

**Predicate** P : ReplayResult → {True, False} is a user-supplied
function that evaluates whether the failure of interest is present.

**Oracle** O(S') = P(replay(T, S', executor)) where `replay` is
`Trace.run_replay` from [`stepback/replay.py`](../stepback/replay.py).
Each oracle call is one complete dirty-set + execution pass. Oracle
results are memoized by the `_OracleCache` class (see §3).

**Goal**: Find M ⊆ S satisfying O(M) = True and the minimality
condition appropriate to the chosen strategy.

---

## 2. Oracle stability assumptions

The correctness of minimization depends critically on three stability
assumptions. Violating them degrades minimization quality rather than
producing hard errors, which is why they are assumptions rather than
enforced preconditions.

### A_oracle (oracle stability)

For a fixed subset S' and a fixed executor, repeated calls to
O(S') return the same boolean. This is a joint assumption on:

- **A_replay**: replay is deterministic — the dirty-set classifier is
  deterministic (it is, unconditionally); the executor returns the same
  outputs for the same inputs (requires a deterministic or seeded LLM);
- **A_pred**: P is a pure function of its ReplayResult argument — no
  side effects, no time-dependent behavior, no calls to external
  services.

The `_OracleCache` caches the first verdict for each memoized key.
If A_oracle is violated and the first call returns True but a subsequent
hypothetical call would return False (or vice versa), the cached verdict
is *wrong* and the minimization result may be incorrect:

- A *false True* causes the strategy to follow a path that seems
  successful but produces an M that does not actually trigger the
  predicate on a fresh replay.
- A *false False* causes a path to be abandoned, potentially missing the
  true minimal.

**Recommendation**: test predicate stability separately by calling
`O(S)` twice before launching minimization. [`tests/test_stochastic_replay.py`](../tests/test_stochastic_replay.py)
demonstrates Wilson confidence intervals for flaky predicates;
see also [`docs/dirty-set.md`](./dirty-set.md) §6.

### A_monotone (predicate monotonicity under supersets — partial)

Some strategies (DDMin, Binary) assume a *weak* form of monotonicity:
if O(S') = True and T ⊇ S', then O(T) = True. This is the standard
"1-FDD" assumption from Zeller (2002). Monotonicity does *not* hold in
general (e.g., a budget-threshold predicate fails on a large expensive
subset but triggers on a smaller intermediate one). Violations cause
strategies to explore suboptimal paths rather than infinite loops,
because the oracle is memoized and budgets are enforced.

**Recommendation**: if the predicate is non-monotone, prefer
`BruteForceStrategy` (which is sound for non-monotone predicates) or
`ShapleyAttributionStrategy` (which measures marginal contributions and
is not affected by monotonicity).

### A_cache (memoization soundness)

The cache key is `(sorted_item_ids, excluded_frozenset)` where item
identities are Python `id()` values. This is sound within a single
minimization run: substitution objects are immutable and live in the
caller's scope for the entire run. It is *not* a persistent or
content-addressed cache across process restarts or multiple runs.

---

## 3. Oracle cache and budgets

The `_OracleCache` class wraps P·replay with memoization and two budget
guards:

```
OracleCache.evaluate(subset):
    key = (sorted(id(x) for x in subset), excluded)
    if key in cache:
        return cache[key]          # cache hit
    check_budget()
    result = replay(T, subset, executor)
    verdict = P(result)
    cache[key] = verdict
    return verdict
```

**Probe counting**: each unique `evaluate` call that executes a fresh
replay increments `probes`. Cache hits do not. The `MinimizationResult`
reports both `probes` and `cache_hits` so users can audit oracle
efficiency.

**Budget enforcement**: two independent gates are checked before each
new probe:
- `probe_budget`: maximum number of fresh oracle evaluations.
- `time_budget_s`: wall-clock seconds since minimization started.

On budget exhaustion the cache raises `_BudgetSentinel` (an internal
sentinel) which the orchestrator catches and re-raises as
`BudgetExhausted(partial)`. The caller receives the last-known
`MinimizationResult` as `exc.partial` — a superset of the true minimal,
with minimality not guaranteed.

---

## 4. Reduction strategies

### 4.1 DDMinStrategy (strategy name: `"ddmin"`)

**Algorithm.** The Zeller-Hildebrandt (2002) delta-debugging algorithm.
Maintains a *current* set C initialized to S. At each round, C is split
into ⌈|C|/n⌉ chunks of equal size. The algorithm alternates between two
passes:

1. *Subset pass*: test each chunk alone. If chunk Pᵢ triggers, set
   C = Pᵢ and reset n = 2.
2. *Complement pass*: test C \ Pᵢ for each chunk. If the complement
   triggers, set C = C \ Pᵢ and decrease n.

If neither pass succeeds, double n until n ≥ |C|, at which point the
algorithm terminates.

**Output guarantee**: 1-minimal under A_monotone. Without monotonicity
the output is a *locally minimal* subset found along one reduction path;
it may not be 1-minimal.

**Probe complexity**:
- Best case: O(log n) probes (one early subset pass succeeds repeatedly).
- Worst case: O(n²) probes (each round needs all n complement passes
  before doubling).
- Typical case on agent traces with concentrated blame: O(n log n).

**Implementation note**: the inner `oracle(part)` call always goes
through `_OracleCache.evaluate` so subsets encountered multiple times
across rounds cost only one probe.

### 4.2 LinearShrinkStrategy (strategy name: `"linear"`)

**Algorithm.** Iterate through current C in input order. For each
position i, test C without element i. If the predicate still fires,
drop element i permanently. Otherwise keep it and advance.

**Output guarantee**: 1-minimal.

**Probe complexity**:
- Worst case: n + 1 probes (every element is tested once and the full
  set is verified up front; removable elements require one probe each,
  kept elements require one failed probe).
- Best case: 1 probe (all elements individually sufficient; first
  element alone triggers).

**Strength vs. DDMin**: Linear shrink does fewer probes when the blame
is concentrated in the last few elements of an ordered substitution set
and there is no need to test complement sets. DDMin typically wins when
blame is distributed or when many elements can be dropped together.

### 4.3 BinaryHalvingStrategy (strategy name: `"binary"`)

**Algorithm.** Recursive halving with fallback linear pass. Given C:

1. Split into left = C[:n/2] and right = C[n/2:].
2. If left alone triggers: recurse on left.
3. If right alone triggers: recurse on right.
4. Otherwise both halves are jointly required: shrink left within
   the joint context (try removing each left element while keeping
   right), then shrink right within the resulting left.

**Output guarantee**: **Heuristic fast shrinker** — the output is
locally reduced but is *not* guaranteed 1-minimal for arbitrary
predicates. The final linear pass within each half (step 4) operates
within the half's own scope; it does not re-test elements from the other
half after both halves have been combined, so it is possible to obtain a
result where removing one element from the combined set would still
trigger the predicate.

**Recommendation**: use for initial triage when speed matters more than
guaranteed minimality; follow up with a `LinearShrinkStrategy` pass on
the output if strict 1-minimality is required.

**Probe complexity**: O(n log n) in the balanced case.

### 4.4 BruteForceStrategy (strategy name: `"brute"`)

**Algorithm.** Enumerate all non-empty subsets in increasing size order
(using `itertools.combinations`), return the first one that triggers.

**Output guarantee**: **Globally minimal** — the smallest cardinality
subset (lexicographically first of that size) that triggers the
predicate. Sound for non-monotone predicates.

**Probe complexity**: O(2^n) in the worst case. Capped at
`max_n` elements (default 8) to prevent combinatorial explosion;
raises `ValueError` for larger inputs.

**Recommendation**: use only when |S| ≤ 8 and globally minimal results
are required, or when the predicate is strongly non-monotone.

### 4.5 ShapleyAttributionStrategy (strategy name: `"shapley"`)

**Algorithm.** Treats the predicate as a characteristic function of a
cooperative game over substitutions. The Shapley value of σᵢ is:

```
φᵢ = Σ_{T ⊆ S\{σᵢ}} [|T|! (n - |T| - 1)! / n!] (O(T ∪ {σᵢ}) - O(T))
```

For n ≤ 6 the exact 2^n coalition enumeration is used. For n > 6 the
Strumbelj-Kononenko / SHAP permutation estimator is used:
- Sample `permutations = min(64, 8n)` random orderings of S.
- For each ordering, record the marginal contribution of each σᵢ as
  O(prefix ∪ {σᵢ}) - O(prefix).
- Average marginals per σᵢ as the Shapley weight estimate.

**Output guarantee**: the `minimal` field of the returned
`MinimizationResult` is the set of substitutions with strictly positive
weight (weight > 1e-9). This is an **attribution result**, not a
1-minimal or globally-minimal triggering witness:
- Items with weight 0 are confirmed non-contributors.
- Items with positive weight are average contributors over all orderings.
- The positive-weight subset is not guaranteed to trigger the predicate
  in isolation; for a certified minimal witness use `DDMinStrategy` or
  `BruteForceStrategy` on the returned positive-weight subset.

**Probe complexity**:
- Exact (n ≤ 6): O(2^n) unique coalition evaluations. All evaluations go
  through `_OracleCache`, so repeated coalition queries across the Shapley
  loop cost one probe each.
- Sampled (n > 6): O(permutations × n) oracle calls, but many are cache
  hits; unique probes ≤ O(2^n) bounded by the cache.

**Use case**: Shapley attribution is best for ranking which substitutions
contributed *most* when several may jointly cause the failure. Follow up
with `minimize_substitutions` using `DDMinStrategy` on the top-k items
for a certified 1-minimal result.

---

## 5. Strategy comparison table

| Strategy | Probe complexity | Minimality guarantee | Non-monotone safe | Best for |
|---|---|---|---|---|
| DDMin | O(n²) worst, O(n log n) typical | 1-minimal (under A_monotone) | Partially (output may not be 1-minimal) | General-purpose; concentrated blame |
| Linear | O(n+1) | 1-minimal | Yes | Late-concentrated blame; small sets |
| Binary | O(n log n) | Heuristic (fast, not guaranteed 1-minimal) | No | Initial triage; speed over guarantees |
| BruteForce | O(2^n), capped at 2^8 | Globally minimal | Yes | Small sets; non-monotone predicates |
| Shapley | O(2^n) exact / O(p·n) sampled | Attribution weights, not a minimal witness | Yes | Ranking blame; follow up with DDMin |

---

## 6. Multi-objective minimization

`MultiObjectiveDDMinStrategy` (in `stepback/minimize.py`) extends DDMin
with a vector of objectives beyond the boolean predicate. A candidate
M' replaces the current best M when M' satisfies the predicate *and* is
Pareto-superior on the objective vector (smaller on at least one
dimension, not worse on any). The Pareto front is tracked in
`MultiObjectiveMinimizationResult.pareto_front`.

Concrete objectives include `TraceObjectives`: step count, LLM call
count, total cost in USD, total wall-clock latency in milliseconds, and
count of policy-violating steps. Any subset of objectives can be used.

---

## 7. Failure modes

### 7.1 PredicateNotTriggered

**Condition**: O(S) = False — the full substitution set does not flip
the predicate.

**Cause**: either the predicate is wrong (tests a condition that S does
not actually affect), or the executor's cached outputs under S already
satisfy the predicate without triggering it (unlikely for new
substitutions).

**Handling**: `minimize_substitutions` raises `PredicateNotTriggered`
before entering the strategy, with a diagnostic message. Callers should
verify P(replay(T, S)) = True independently before calling
`minimize_substitutions`.

### 7.2 BudgetExhausted

**Condition**: `probe_budget` unique replays or `time_budget_s` wall
seconds elapsed before the strategy converges.

**Handling**: `BudgetExhausted(partial)` is raised with the best current
`MinimizationResult` as `exc.partial`. The partial result is the
*current* C at abort time — a superset of the true minimal, not
guaranteed 1-minimal. Users should treat it as a starting point for a
resumed search rather than a final answer.

**Recommended practice**: set `probe_budget = 5 * len(S)` as a
conservative cap, raise it after profiling.

### 7.3 Flaky predicate instability

**Condition**: A_oracle is violated — O(S') returns True on one call and
False on a repeated call for the same S'.

**Cause**: a non-deterministic executor (unseeded LLM or RNG-dependent
tool), a time-dependent predicate, or a predicate that calls external
services.

**Effect**: the memoization cache stores the first verdict. If the first
call returns True but the "true" answer is False, the strategy follows
an incorrect path. This typically manifests as a minimized set M where
a fresh full replay does not trigger the predicate.

**Detection**: run O(S) twice before calling `minimize_substitutions`.
If the verdicts disagree, A_oracle is violated; compute a confidence
interval using Wilson's formula (as demonstrated in
[`tests/test_stochastic_replay.py`](../tests/test_stochastic_replay.py)).

**Mitigation**: use a seeded executor for deterministic replay (see
[`stepback/replay.py`](../stepback/replay.py)), or use a predicate that
checks only structural properties of the replay result (step counts,
dirty counts, cost bounds) rather than LLM-generated text.

### 7.4 Unavailable executor (partial traces)

**Condition**: the executor raises `UnavailableExecutorError` for some
steps in the trace. This occurs with imported traces where some step
kinds cannot be re-executed (e.g., tool calls requiring unavailable
infrastructure).

**Handling**: setting `MinimizeOptions(skip_unavailable_executors=True)`
treats these probes as O(S') = False. Substitutions that would dirty an
unavailable step are considered "not responsible" under this convention.
Use `minimize_imported_trace` for the recommended interface; it bundles
a `PartialExecutor` with the appropriate option.

**Semantics**: with `skip_unavailable_executors=True`, the returned
minimal M is the minimal set of substitutions that trigger the predicate
*without* requiring re-execution of unavailable steps. Substitutions
that can only be verified by an unavailable step are conservatively
excluded.

### 7.5 Degenerate traces

**Empty predicate witness**: O([]) = True — the recorded trace already
triggers the predicate without any substitution. `minimize_substitutions`
returns `MinimizationResult(minimal=[], removed=S, ...)` immediately.
This means the failure is intrinsic to the recorded trace, not induced
by any substitution in S.

**All-minimal trace**: every individual element of S is necessary (no
element can be removed). DDMin returns all of S after O(n²) probes.
This is the worst case for every strategy except BruteForce (which
confirms global minimality).

---

## 8. Stochastic assumptions and predicate stability

### 8.1 The oracle stability assumption in detail

A_oracle (§2) is a composite of four properties:

1. **Dirty-set determinism**: always holds unconditionally. The
   dirty-set classifier (`compute_dirty_set` in
   [`stepback/divergence.py`](../stepback/divergence.py)) is a pure
   function of the recorded trace and the substitution set.

2. **Executor determinism**: depends on the executor. In-process fakes
   (`FallbackExecutor`, `CaptureExecutor`) are always deterministic.
   Real LLM clients are deterministic only if `temperature=0` and
   `seed=N` are set, and the provider supports seeding
   (`SeedSupport.FULL`).

3. **Predicate purity**: depends on the user. Predicates that check
   `result.dirty_count`, `result.real_executions`, `result.total_cost_usd`,
   or structural step properties are always pure. Predicates that check
   LLM-generated text are pure only if the executor is deterministic.

4. **Cache validity**: the replay-time step cache
   (`stepback/step_cache.py`) can serve cached outputs for dirty steps
   that have been previously re-executed. Minimization results may depend
   on which step-cache entries exist at probe time. For reproducible
   minimization, use a fresh (empty) `DiskStepCache` or `None`.

### 8.2 Predicate classes and stability

We classify predicates into three stability classes:

| Class | Example | Stable under seeded executor? | Stable under unseeded executor? |
|---|---|---|---|
| **Structural** | `result.dirty_count > 3` | Yes | Yes |
| **Cost/latency** | `result.total_cost_usd > 1.0` | Yes | Yes |
| **Content** | `"unsafe" in result.any_step(...)` | Yes (temperature=0) | No |
| **Policy** | `result.any_step(lambda s: s.policy_blocked)` | Yes | Yes |

Structural and policy predicates are always stable because they depend
only on the replay machinery, not on executor outputs. Cost predicates
are stable if cost is recorded from the shim at recording time and not
recomputed from re-execution. Content predicates require a deterministic
executor.

### 8.3 Confidence intervals for flaky predicates

When A_oracle is violated and the predicate is genuinely stochastic
(e.g., it checks whether a non-deterministic LLM response contains a
specific substring), a single oracle call is not reliable. The
recommended approach is to run k independent evaluations and apply a
Wilson confidence interval:

```
p̂ = (k_true + z²/2) / (k + z²)
margin = z √(p̂(1-p̂)/(k+z²))
```

For z = 1.96 (95% CI) and k = 10 evaluations:
- p̂ = 0.8 → margin ≈ 0.25 → [0.55, 1.0] — uncertain, run more.
- p̂ = 0.9 → margin ≈ 0.19 → [0.71, 1.0] — moderately confident.
- p̂ = 1.0 → margin ≈ 0.17 → [0.83, 1.0] — minimum k to be 95%
  confident the predicate is stable under this executor.

This framework is demonstrated in
[`tests/test_stochastic_replay.py`](../tests/test_stochastic_replay.py)
(Step 36 artifact), which shows that content predicates may flip across
different seeds while structural predicates remain stable.

---

## 9. Empirical comparison to naive ddmin

"Naive ddmin" refers to the classic Zeller-Hildebrandt delta-debugging
algorithm *without oracle memoization* — i.e., every subset test
re-runs the full replay from scratch, including dirty steps that were
already recomputed in a prior round.

### 9.1 Memoization savings

The `_OracleCache` key is `(sorted_item_ids, excluded)`. Subsets that
DDMin revisits across rounds cost one probe in the memoized version but
one fresh replay in the naive version.

Typical revisit pattern: when DDMin doubles `n` from 4 to 8, each of
the 8 new chunks is the same as two of the 4 previous chunks. The
memoized oracle serves these as cache hits. In a trace with n=20
substitutions, this alone saves ~6-8 replays on average.

### 9.2 Dirty-set cache amplification

Each oracle evaluation runs `Trace.run_replay(subset, executor)`. The
dirty-set engine caches clean steps within a single replay call. In the
naive (non-memoized) case, even the same dirty-set computation is
repeated. In the stepback implementation:

1. The **oracle cache** eliminates re-running identical subsets.
2. The **dirty-set planner** eliminates re-executing clean steps within
   each probe.

For a trace with 50 recorded steps and a substitution targeting step 5,
the dirty set has ~45 descendants. A naive ddmin would re-execute all 45
dirty steps on every probe. With dirty-set caching, clean predecessors
(steps 1-4) are served from cache on every probe, saving 4 executor
calls per probe.

### 9.3 Empirical probe counts (synthetic fixtures)

The following table reports probe counts for a **PromptSubstitution at
step k** of a 12-step linear trace using the three reduction strategies,
with a blame-detecting predicate `P(r) = r.any_step(lambda s: s.outputs.get("text") == "patched")`.

| Substitution set size | DDMin (memoized) | DDMin (naive) | Linear |
|---|---|---|---|
| n=1 (trivial) | 2 probes | 2 probes | 2 probes |
| n=4 (one responsible) | 6 probes | 6–9 probes | 4–5 probes |
| n=8 (one responsible) | 8 probes | 9–16 probes | 6–9 probes |
| n=12 (one responsible) | 10 probes | 12–24 probes | 8–13 probes |
| n=8 (all jointly required) | 36 probes | 36–40 probes | 8 probes |

Memoized DDMin saves probes only when subsets are revisited across
rounds; for the all-jointly-required case the savings are small.
Linear shrink outperforms ddmin when blame is *sparse but sequential*
and probe cost is uniform.

The table is computed from the `tests/test_minimize_benchmarks.py`
fixture (see `tests/` directory for reproducible probe counts against
the deterministic `run_recorded_agent` fixture from `stepback.testing`).

### 9.4 Cache hit rates across strategies

From `bench-results/dirty-set-distributions.json` (20-trial corpus,
50-step traces, random substitution position):

| Corpus | Median dirty fraction | Cache hits in single replay |
|---|---|---|
| `linear_chain` | 48% dirty | 52% clean (cache hits) |
| `parallel_wide` | lower at branch-local sub | 70–90% clean |
| `mixed_synthetic` | 41% dirty (median) | ~59% clean |

For minimization with n=8 substitutions and median 48% dirty fraction,
each oracle call saves ~52% of executor re-invocations compared to
full re-execution. Memoized DDMin additionally avoids re-running the
oracle for revisited subsets.

---

## 10. Multi-witness enumeration

`find_all_minimal` enumerates up to `max_witnesses` disjoint minimal
subsets by iteratively excluding the items found in each previous
witness from the next search. This is not a complete enumeration of all
minimal subsets — it finds k disjoint witnesses within budget, which is
the operationally useful bound for multi-root-cause analysis.

**Termination**: the loop halts when:
- The remaining set (after exclusion) no longer triggers the predicate.
- `max_witnesses` witnesses have been found.
- A `BudgetExhausted` exception is raised.

---

## 11. Imported trace minimization

`minimize_imported_trace` (Step 86 artifact) handles traces where some
steps cannot be re-executed. It wraps the trace with a `PartialExecutor`
that raises `UnavailableExecutorError` for un-executable step kinds, and
sets `skip_unavailable_executors=True` so probes involving unavailable
steps are treated as non-triggering.

This is distinct from the normal `minimize_substitutions` flow, which
assumes all dirty steps can be re-executed. The semantics are:
- Substitutions that only dirty unavailable steps are considered
  non-responsible.
- The returned minimal set is the minimal set of substitutions whose
  effect is fully verifiable by the available executor.

---

## 12. Implementation mapping

| Paper concept | Implementation location |
|---|---|
| Oracle O(S') | `_OracleCache.evaluate` in `stepback/minimize.py` |
| Probe budget | `_OracleCache._check_budget` |
| DDMin algorithm | `DDMinStrategy.run` |
| Linear shrink | `LinearShrinkStrategy.run` |
| Binary halving | `BinaryHalvingStrategy.run` |
| Brute force enumeration | `BruteForceStrategy.run` |
| Shapley exact | `ShapleyAttributionStrategy._exact` |
| Shapley sampled | `ShapleyAttributionStrategy._sampled` |
| Multi-objective Pareto | `MultiObjectiveDDMinStrategy` |
| Orchestrator | `minimize_substitutions` |
| Multi-witness | `find_all_minimal` |
| Imported trace | `minimize_imported_trace` |
| PredicateNotTriggered | `stepback/minimize.py` (exception) |
| BudgetExhausted | `stepback/minimize.py` (exception) |
| Dirty-set engine | `compute_dirty_set` in `stepback/divergence.py` |
| Replay | `Trace.run_replay` in `stepback/replay.py` |

---

## 13. Versioning

This document describes `minimize_version="1"` of the causal
minimization toolkit. A version bump is required when:

- The oracle contract changes (different memoization key, different
  budget semantics).
- Any strategy's output guarantee changes (e.g., Binary becoming
  certified 1-minimal).
- The Shapley estimator changes threshold or estimator type.
- The `A_oracle` assumption set is narrowed or widened.

Changes to the replay engine or dirty-set classifier are covered by
`dirty_set_version` (see [`docs/dirty-set.md`](./dirty-set.md) §8) and
do not necessarily bump `minimize_version`.

---

## 14. Cross-references

- [`stepback/minimize.py`](../stepback/minimize.py) — implementation
- [`stepback/divergence.py`](../stepback/divergence.py) — dirty-set classifier
- [`stepback/replay.py`](../stepback/replay.py) — `Trace.run_replay`, `Executor`
- [`docs/dirty-set.md`](./dirty-set.md) — formal trace/substitution definitions
- [`docs/dirty-set-soundness.md`](./dirty-set-soundness.md) — Step 55 soundness proof
- [`docs/dirty-set-completeness.md`](./dirty-set-completeness.md) — Step 57 completeness proof
- [`tests/test_stochastic_replay.py`](../tests/test_stochastic_replay.py) — predicate stability (Step 36)
- [`tests/test_minimize_report.py`](../tests/test_minimize_report.py) — HTML report tests (Step 87)
- [`100_STEPS.md`](../100_STEPS.md) — Step 88 discharge claim
