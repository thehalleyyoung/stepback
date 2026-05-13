# The dirty-set algorithm (v1)

This document is the prose companion to the contract block at the top of
the §"Dirty-set propagation" section in
[`stepback/divergence.py`](../stepback/divergence.py) and the reference
implementation in
[`stepback/replay.py::Trace.run_replay`](../stepback/replay.py). It defines
the formal objects the algorithm operates over (trace DAG, canonical input
function, substitution sigma, dirty set, cache reuse) and states the
observational-equivalence theorem that connects them.

The pinned name of this algorithm is `dirty_set_version="1"`. It is bound
to `canonicalisation_version="1"` (see
[`docs/canonicalization.md`](./canonicalization.md)) and to the v1 wire
format described in [`spec/sbtrace-v1.md`](../spec/sbtrace-v1.md). Any
change to the canonicalizer or to the set of declared dependencies that a
recorder may emit is a breaking change to the dirty-set version and MUST
bump it.

> **One-line definition.** Under substitution `σ`, the dirty set
> `D(T, σ) ⊆ V(T)` is the smallest set containing every step directly
> targeted by `σ`, every step whose recomputed canonical inputs hash
> differently from the recorded inputs hash, every transitive descendant
> of any such step in the recorded DAG, and every step whose recorded
> `nondeterminism_hash` is internally inconsistent. Steps in
> `V(T) ∖ D(T, σ)` are *clean* and their recorded outputs are reused
> verbatim.

## Why this matters

The whole performance claim of stepback —
*"counterfactual debugging of an N-step agent trace costs `O(|D|)` LLM
calls instead of `O(N)`"* — is exactly the claim that the dirty-set
algorithm computes a sound *and* small set. If `D` is unsound (some step
in the clean set should have been dirty), stale cached outputs corrupt
the replay and every downstream artifact. If `D` is non-minimal (some
step is dirty for no reason), every cache miss is paid in real LLM cost
or in `MissingExecutor` errors. The two failure modes are not symmetric:

| Failure mode | Symptom | Detection |
|---|---|---|
| Under-dirtying (P5 violation, but worse) | Replay reuses stale outputs; downstream answers reference a now-impossible past. | Property tests `test_dirty_set_equals_full_recompute` (Step 29) and the differential-fuzzing harness (Step 49). |
| Over-dirtying (cost regression, P5 violation) | Cache miss rate goes up; `bench replay-caching` regresses; `MissingExecutor` raised when no live executor is wired. | Soak benchmark `bench dirty-set` (Step 65) and the `dirty_after_sub=11` reconciliation tracker (Step 67). |

Under-dirtying is a *correctness* bug. Over-dirtying is a *cost* bug.
Both are bugs.

## 1. Trace DAG

Let `T` be a recorded `.sb` trace, parsed by
[`stepback/trace_reader.py::read_trace`](../stepback/trace_reader.py). `T`
defines a finite directed acyclic graph `G(T) = (V(T), E(T))`:

* `V(T)` is the set of recorded `step` frames in trace order. Each
  vertex `s ∈ V(T)` carries the eight fields enumerated in README
  §"The `.sb` trace format": `step_id`, `step_kind`, `parent_step_id`
  (or `parent_step_ids` for joins), `inputs`, `outputs`, `inputs_hash`,
  `nondeterminism`, `nondeterminism_hash`, plus accounting fields
  (`wallclock_ns`, `cpu_ns`, `cost_usd`) and the per-frame `receipt`.
* `E(T)` is the *recorded parent-edge relation*: there is an edge
  `(p, s) ∈ E(T)` iff `p ∈ {s.parent_step_id} ∪ s.parent_step_ids`.

`G(T)` is required to be acyclic and topologically ordered by recorded
position: every parent appears strictly earlier than its child in the
frame stream. Cycles are rejected by `verify_trace`. The empty trace is
the trivial DAG `(∅, ∅)`.

The `step_kind` of `s` partitions vertices into seven classes:

```
llm_call | tool_call | router | policy_check | mcp_call
parallel_branch_open | parallel_branch_join | exception
```

The dirty-set algorithm is *kind-agnostic* with two narrowly-scoped
exceptions: `parallel_branch_join` consumes `branch_tail_hashes` (see
§5) and exception steps participate in `RaiseSubstitution` semantics
(specified in §5.6).

## 2. Canonical input function

Let `J = canonical_json` be the canonicalizer documented in
[`docs/canonicalization.md`](./canonicalization.md). Let `H = hash_obj`
be its companion `BLAKE2b-256(J(·))` hash.

For a step `s`, the *recorded* canonical inputs hash is `s.inputs_hash`.
The recorder contract is

```
s.inputs_hash == H(s.inputs)        for every recorded step s.
```

The *current* canonical inputs of `s` under a partial assignment
`out: V(T) ⇀ JSON` of currently-known parent outputs is the function

```
inputs(s, out) =
    let I = deepcopy(s.inputs)
    if "context" in I and parent_step_id(s) ∈ dom(out):
        I["context"] = H(out[parent_step_id(s)])
    if "branch_tail_hashes" in I and "branch_tails" in I:
        for i, t in enumerate(I["branch_tails"]):
            if t ∈ dom(out):
                I["branch_tail_hashes"][i] = H(out[t])
    return I
```

This is the *conservative dependency model* (assumption A1 of the
contract block in `divergence.py`): the only declared cross-step
dependencies a step may have are `"context"` (single-parent) and
`"branch_tails"` / `"branch_tail_hashes"` (multi-parent, joins only).
Any other channel — for example, a tool call that secretly reads a
prior tool's stdout via a side path — is **not** modelled and is the
recorder's responsibility to surface (see §6, "Recorder obligations").

Define `current_hash(s, out) = H(inputs(s, out))`.

## 3. Substitution σ

A substitution is a typed object from
[`stepback/substitutions.py`](../stepback/substitutions.py). For the
purposes of this document a substitution `σ_i` is a triple

```
σ_i = (target_id, kind, payload)
   target_id ∈ {step_id : s ∈ V(T)} ∪ {⊥}
   kind ∈ {prompt, tool_input, tool_output, router_decision,
           output_force, raise, ...}
   payload : kind-specific JSON
```

A substitution targeting an `target_id ∉ V(T)` is *inert*: it cannot
dirty any step. (See P3 below.)

A *substitution set* `σ = {σ_1, ..., σ_m}` is applied to a step `s`
during the dirty-set walk in two ways:

1. **Input-mutating substitutions** (`prompt`, `tool_input`,
   `router_decision`, …) replace fields in `inputs(s, out)` *before*
   `current_hash(s, out)` is computed. Hash drift dirties `s`.
2. **Output-forcing substitutions** (`tool_output`, `output_force`,
   `raise`) pin `s.outputs` to the payload, mark `s` dirty regardless
   of whether the inputs hash changed (P3), and mark every descendant
   of `s` reachable in `G(T)` dirty by transitive closure (P2). See
   §5.6 for the special case where the payload is an error sentinel
   (`RaiseSubstitution`).

The empty substitution `σ = ∅` is the *deterministic-replay* case.
Under `σ = ∅` and an untampered, well-formed trace, the postconditions
collapse to `D(T, ∅) = ∅` and every step is a cache hit.

## 4. The dirty set `D(T, σ)`

Walk the vertices of `G(T)` in topological order. Maintain a partial
output map `out`. The classification of `s` is:

```
function classify(s, σ, out):
    # P3: direct substitution targeting always dirties.
    if any σ_i with σ_i.target_id == s.step_id:
        return DIRTY (reason="substituted")

    # P4: tampered nondeterminism hashes always dirty.
    if s.nondeterminism_hash != H(s.nondeterminism):
        return DIRTY (reason="ndh_tamper")

    # P2 (transitive): any dirty parent dirties s.
    for p in parents(s):
        if classify(p, σ, out) is DIRTY:
            return DIRTY (reason="parent_dirty")

    # P2 (input drift): rebound canonical inputs hash differently.
    if current_hash(s, out) != s.inputs_hash:
        return DIRTY (reason="input_drift")

    return CLEAN
```

The dirty set is `D(T, σ) = { s ∈ V(T) : classify(s, σ, out) = DIRTY }`
where `out` is the running output map populated by either reusing
`s.outputs` (clean steps) or recomputing via the executor (dirty steps,
see §5).

The four `dirty_reason` tags correspond exactly to the four reasons P3,
P4, P2-via-parent, and P2-via-input-drift. They are surfaced verbatim
on `DirtySetEntry.dirty_reason` in
[`stepback/divergence.py`](../stepback/divergence.py).

`D` is well-defined: parents are always classified before children
(topological walk), and there is no other source of recursion.

## 5. Cache reuse and replay semantics

Reuse is a property of the *replay engine*
([`stepback/replay.py::Trace.run_replay`](../stepback/replay.py)), not
of the dirty-set classifier. Given `D(T, σ)`, replay produces a new
output map `out_σ : V(T) → JSON` step-by-step:

```
function replay(T, σ, executor):
    out_σ := {}
    for s in topo_order(V(T)):
        if s ∉ D(T, σ):
            # P1 (cache reuse).
            out_σ[s] := s.outputs
            cache_hit[s] := True
        else if any σ_i is output-forcing on s:
            out_σ[s] := σ_i.payload
            cache_hit[s] := False
        else:
            # Real re-execution.
            out_σ[s] := executor.execute(s.kind, inputs(s, out_σ))
            cache_hit[s] := False
    return out_σ
```

Two practical consequences:

1. The number of *real executor calls* is at most
   `|D(T, σ)| − |{σ_i : σ_i is output-forcing}|`. Output-forcing
   substitutions dirty their target but pay zero LLM cost.
2. `executor.execute(s.kind, ·)` is called with the *current* canonical
   inputs `inputs(s, out_σ)`, not with the recorded `s.inputs`. For
   dirty steps whose parents are also dirty, this is the only place the
   substitution actually flows forward.

When `executor.fallback_recorded` is true and no callback is registered
for `s.kind`, `executor.execute` returns `s.outputs` instead of raising
`MissingExecutor`. This makes `compute_dirty_set` usable as a *pure
classifier* without wiring live LLM/tool callbacks; the classification
itself is independent of the executor.

## 5.5. Branch-aware propagation

The classifier in §4 is uniform over all step kinds, but the
fan-out/fan-in shape produced by `parallel_branch_open` /
`parallel_branch_join` admits a sharper restatement of the contract.
This subsection pins the three properties that distinguish *branch-aware*
propagation from a naive linear "everything after step `k` is dirty"
walk. They are not new rules — they are theorems implied by the
classifier and the recorder obligations §6 (R1–R4) — but they are
called out because regressions in branch handling have an outsized
cost: a 1,000-way fan-out with naive propagation dirties ~1,003 steps
under a single-branch substitution instead of 3.

Let `b ∈ V(T)` be a `parallel_branch_open` step with branch tails
`tails(b) = {t_1, …, t_W}` (one per branch) and let
`j ∈ V(T)` be the matching `parallel_branch_join` whose
`parent_step_ids` is exactly `tails(b)`. Each tail `t_i` has a
*branch* `B_i ⊆ V(T)`, the set of vertices reachable from `b` in
`G(T)` whose unique path back to `b` goes through `t_i`. Branches
are pairwise disjoint by R4 (DAG topology, no cross-branch parent
edges declared).

**Property B1 (independent fan-out children).** *For every branch
`B_i` and every `s ∈ B_i`, `s ∈ D(T, σ)` only if either*

* *some `s' ∈ {b} ∪ {ancestors of s within B_i}` is dirty (parent-dirty
  closure within the branch), or*
* *some `σ_k ∈ σ` directly targets `s` (P3), or*
* *`current_hash(s, out) ≠ s.inputs_hash` (P2 input drift, possible
  when the recorder declared a `"context"` dependency on a parent
  inside `B_i` that has been recomputed), or*
* *`s.nondeterminism_hash` is tampered (P4).*

In particular, for `i ≠ k`, *no step in `B_i` is dirtied by
substitutions whose `target_id ∈ B_k`.* The recorder's R2 obligation
forbids declaring a `"context"` dependency from `B_i` into `B_k`
(branches are independent by construction), and the classifier's
parent-dirty clause walks edges in `G(T)`, not chronological
recorded-position ordering. Sibling branches therefore stay clean.

**Property B2 (join dirty iff any consumed branch output changes).**
*Let `j` be a `parallel_branch_join` with parents `tails(b)`.
Then `j ∈ D(T, σ)` iff at least one of:*

* *some `t_i ∈ tails(b)` is in `D(T, σ)` (parent-dirty closure on the
  multi-parent edge set, P2), or*
* *some `σ_k ∈ σ` directly targets `j` (P3), or*
* *`j.nondeterminism_hash` is tampered (P4).*

The branch-tail-hash rebinding rule of §2 gives the iff direction:
`current_hash(j, out_σ) = H(inputs(j, out_σ))` includes one
`H(out_σ[t_i])` per branch tail. If every tail is clean,
`out_σ[t_i] == t_i.outputs` for every `i` (by P1), so
`current_hash(j, out_σ) == j.inputs_hash` and the input-drift clause
does not fire. If some tail `t_i` is dirty, then either `out_σ[t_i]`
was recomputed by the executor or pinned by an output-forcing
substitution; in both cases `H(out_σ[t_i])` is the
*post-substitution* hash, which differs from the recorded
`branch_tail_hashes[i]` whenever the post-substitution output differs
from `t_i.outputs`. Joins are therefore *exact* multi-parent OR
gates over branch dirtiness — no over-dirtying when every branch is
trivially clean, no under-dirtying when any consumed branch output
actually drifted.

**Property B3 (clean siblings are preserved).** *For any branch
substitution localised to one branch — formally, for any
`σ ⊆ {σ_k : target_id(σ_k) ∈ B_k}` — the set of clean steps
contains every `s ∈ B_i` for `i ≠ k`, every ancestor of `b` in
`G(T)`, and `b` itself.*

This is a consequence of B1 (sibling branches contain no
declared dependency on `B_k`) plus the topo-walk classifier (no
classifier rule looks at later-recorded steps, so a substitution at
position `p_k` cannot dirty any step at position `p < p_k` in the
recorded stream). Combined with B2, the dirty set under a
single-branch substitution `σ_k` localised to `B_k` is exactly

```
D(T, σ_k) = (D(T, σ_k) ∩ B_k) ∪ {j} ∪ descendants_after_join(j)
```

with `|D(T, σ_k) ∩ B_k| ≤ |B_k|` and the count *independent of W*
(the fan-out width). This is the formal statement of the
"O(dirty branch) not O(N)" headline that motivates parallel
recording in stepback.

**Engineering implications.** All three properties hold by
construction in [`stepback/replay.py::Trace.run_replay`](../stepback/replay.py)
(see lines 320–347 for the join multi-parent rebinding and
multi-parent dirty OR), and are pinned by stress tests in
[`tests/test_parallel_branch_stress.py`](../tests/test_parallel_branch_stress.py)
and [`tests/test_branch_aware_propagation.py`](../tests/test_branch_aware_propagation.py).
A regression that, e.g., dirties siblings by recorded-position
ordering (the naive linear walk) would trip B1/B3 immediately on
the 1,000-way fan-out fixture (3 dirty → ~1,003 dirty).

## 5.6. `RaiseSubstitution` semantics

`RaiseSubstitution` is an output-forcing substitution that makes a
step appear to have raised an exception. It is a *replay* concept,
orthogonal to the recorder's `exception` step kind. The specification
below is normative for `dirty_set_version="1"`.

### Forced-output format

When `RaiseSubstitution(at_step=id, exception_type=T, message=M)` is
applied, the targeted step's output is replaced with the error
sentinel dict:

```json
{"__error__": {"type": T, "message": M}}
```

The sentinel is a regular JSON dict. Its canonical-JSON serialization
is computed by the canonicalizer and participates in hash-based
downstream rebinding exactly like any other step output. Cost is
reported as zero for the targeted step (no LLM or tool is called).

### Classification of the targeted step

Under P3 of the dirty-set contract, the targeted step is **always
dirty** with `dirty_reason="substituted"`, regardless of its recorded
`step_kind`. The `exception` step kind is a recorder concept (the
original run raised); `RaiseSubstitution` is a replay concept (the
counterfactual run raises). A `RaiseSubstitution` may target any
step kind, including `llm_call`, `tool_call`, `router`, or
`exception`.

### Downstream cache invalidation

Downstream steps become dirty iff the forced error output hash
differs from the recorded output hash of the substituted step. This
condition holds whenever the error sentinel differs from the recorded
outputs, which is guaranteed unless the step's recorded output
happens to equal `{"__error__": {"type": T, "message": M}}` exactly
(an event that can only occur if the original recorder emitted that
exact error sentinel, or under an adversarial trace).

Propagation follows the standard dependency channels:

1. **Single-parent `"context"` steps.** Any downstream step `d` that
   declared a `"context"` dependency on the substituted step `s` will
   have its rebound `"context"` value computed as
   `H(error_sentinel)`. If `H(error_sentinel) ≠ s.inputs_hash`
   stored in `d.inputs["context"]`, the canonical inputs hash of `d`
   changes and `d` is dirty with `dirty_reason="input_drift"`. This
   propagates transitively through every `"context"` chain rooted at
   `s`.

2. **Multi-parent `"branch_tail_hashes"` (join steps).** If the
   substituted step is a branch tail consumed by a
   `parallel_branch_join`, the join's rebound
   `branch_tail_hashes[i]` becomes `H(error_sentinel)`. If this
   differs from the recorded tail hash, the join's canonical inputs
   hash changes → `dirty_reason="input_drift"` on the join. All
   descendants of the join are then dirtied by parent-dirty closure
   (P2) or further input-drift.

3. **Sibling branches are not affected.** Under R4 (branches are
   disjoint), a `RaiseSubstitution` inside branch `B_k` cannot dirty
   any step in a sibling branch `B_i (i ≠ k)` — the error hash
   reaches only steps that declared a dependency on the substituted
   step's output.

### The `exception` step kind

A trace may contain steps with `step_kind="exception"` produced by a
recorder when the original run raised. These steps are classified
under the same dirty-set rules as any other step kind. Applying a
`RaiseSubstitution` to an `exception` step replaces its recorded
error payload with the new sentinel; the downstream propagation rules
above apply unchanged.

### Recorder obligation

If a downstream step depends on the output of a step that may raise
in the counterfactual replay, it **must** declare that dependency via
`"context"` (or `"branch_tail_hashes"` for joins). The
`RaiseSubstitution` propagation rule is not a special-case exception
to this obligation — it is the general output-forcing propagation
rule applied to the error sentinel as the forced output. A downstream
step that does not declare its dependency on the substituted step will
not see input-drift and will be classified clean (potential
under-dirtying: a recorder bug per R2).

### Pinned tests

The semantics in this section are pinned by
[`tests/test_raise_substitution_semantics.py`](../tests/test_raise_substitution_semantics.py).
Key invariants tested:

* Substituted step: `dirty=True`, `dirty_reason="substituted"`,
  `outputs == {"__error__": {"type": T, "message": M}}`, `cost_usd == 0.0`.
* Downstream context-chain steps: `dirty=True`,
  `dirty_reason` in `{"input_drift", "parent_dirty"}`.
* A `RaiseSubstitution` targeting a step inside one parallel branch
  does not dirty sibling branches.
* `compute_dirty_set` classification agrees with `replay_forward`
  dirty/clean assignment.

## 6. Observational equivalence

Define the *observable* of a step to be its canonical-JSON output:
`obs(s, out) = J(out[s])`. Define the observable of a trace under a
replay `out` to be the tuple `obs(T, out) = (obs(s_1, out), …,
obs(s_N, out))` in topological order.

**Theorem (soundness, P1).** *Under assumptions A1–A3 of the contract
block, for every clean step `s ∈ V(T) ∖ D(T, σ)`,*

```
obs(s, out_σ) == obs(s, out_full)
```

*where `out_full` is the output map produced by re-executing every
step from scratch under `σ` with the same executor.*

The full paper-grade proof is given in
[`docs/dirty-set-soundness.md`](./dirty-set-soundness.md) (Step 55 of
[`100_STEPS.md`](../100_STEPS.md)); a Lean/Coq mechanization is
tracked as Step 56.

**Proof sketch.** Induction over topological order, exactly as
recorded in `divergence.py` lines 574–590. Base case: roots have no
parents, no `"context"` rebinding, and `current_hash(s, ∅) ==
H(s.inputs) == s.inputs_hash` iff no input-mutating substitution
targets `s`. Inductive step: assume the theorem holds for every
ancestor of `s`. Then `out_σ[p] == out_full[p]` for every parent `p`
(IH on clean parents; equality holds trivially on dirty parents
because `out_σ` *is* the live re-execution map for them). Therefore
`inputs(s, out_σ) == inputs(s, out_full)`. By A3 (collision-free
hashing) and the classifier guard `current_hash(s, out_σ) ==
s.inputs_hash`, the canonical JSON of `inputs(s, out_σ)` equals
`s.inputs`. By A2 (executor purity), re-execution under those inputs
would produce `s.outputs`. Therefore reusing `s.outputs` as
`out_σ[s]` is observationally equivalent to recomputation. ∎

**Theorem (completeness, P2).** *For every step `s ∈ V(T)` such that
`current_hash(s, out_σ) ≠ s.inputs_hash`, `s ∈ D(T, σ)`.*

This is immediate from the `input_drift` clause of `classify`. The
statement is separated from soundness because they constrain the
classifier in opposite directions: soundness forbids cleaning a
dirty-by-reality step; completeness forbids the trivial "always
return CLEAN" implementation. The full paper-grade proof —
including the four-clause exhaustiveness lemma, the parent-dirty
closure lemma, and an assumption-tightness audit showing that
completeness needs only A3 and R1/R3/R4 (not A1, A2, or R2) — is
given in [`docs/dirty-set-completeness.md`](./dirty-set-completeness.md)
(Step 57 of [`100_STEPS.md`](../100_STEPS.md)).

**Theorem (minimality, P5).** *Under the conservative dependency model
(A1), no step is dirtied unless one of (P2)–(P4) requires it.*

This is the contract that prevents over-dirtying. The classifier has
exactly four DIRTY clauses; each corresponds to one of the
required-dirty conditions. There is no fifth "be safe" clause. If a
recorder declares a step's dependency on a parent's output through
neither `"context"` nor `"branch_tail_hashes"`, A1 fails and P5 may
under-dirty: that is a recorder bug, not an algorithm bug. (Steps 60
and 61 of `100_STEPS.md` track tightening A1 with structured-input
partial recompute and stale-cache detection beyond direct parent-edge
hashing.)

### Recorder obligations

The four clauses of the contract above bind the recorder, not just the
replay engine:

* **R1.** `s.inputs_hash == H(s.inputs)` for every recorded step.
* **R2.** Every cross-step dependency a step has on a parent's output
  is declared either as `"context"` (single-parent) or as
  `"branch_tails"`/`"branch_tail_hashes"` (multi-parent). Ambient
  dependencies — environment variables, wall-clock reads, RNG draws —
  are recorded under `nondeterminism` and folded into
  `nondeterminism_hash`. (See Step 63 for the per-class taxonomy.)
* **R3.** `s.nondeterminism_hash == H(s.nondeterminism)`.
* **R4.** Parent edges form a DAG; topological position in
  `recorded_steps` matches that DAG.

Violations of R1, R3, or R4 are detected by `verify_trace` and the
trace is rejected before the dirty-set classifier runs. Violations of
R2 are *not* detectable by replay alone — they manifest as
under-dirtying and silently corrupt counterfactuals. The differential
fuzzer (Step 49) and the property tests in `tests/test_divergence.py`
exist specifically to bound R2 in practice.

## 7. Asymptotic complexity

Let `N = |V(T)|`, `E = |E(T)|`, `D = |D(T, σ)|`, and `D_real ≤ D` be
the number of dirty steps that require a real executor call (i.e.,
not pinned by an output-forcing substitution). Let `I = Σ_s |J(s.inputs)|`
and `O = Σ_s |J(s.outputs)|` be the total canonical-JSON byte length
of recorded inputs and outputs respectively, and `m = |σ|` the
substitution count.

| Trace shape | Classifier time | Executor calls | Memory |
|---|---|---|---|
| Linear (`E = N − 1`) | `Θ(N + I + O + m)` | `O(D_real)` | `Θ(N + O + m)` |
| General DAG | `Θ(N + E + I + O + m)` | `O(D_real)` | `Θ(N + O + S_max + m)` |
| Branch-heavy (max join width `W`) | `Θ(N + E + I + O + m)` (each join touches `w(s)` parent-output hashes; `Σ_s w(s) = E`) | `O(D_real)` | `Θ(N + O + S_max + m)` |

Here `S_max = max_s (I_s + O_s)` is the largest single-step input+output
payload. `O(N)` memory is dominated by the `outputs_by_id` and
`outputs_hash_by_id` maps maintained during the topological walk; the
dirty-set bookkeeping itself is two `Dict[str, ·]` of size `N`. The
total wall-clock cost including dirty re-execution is `T_total =
T_classify + Θ(D_real · X)` for per-call executor cost `X`.

These bounds are tight: the lower bound `Ω(N + E + I + O)` follows
from the requirement to read every recorded step, edge, input, and
descendant-consumed output at least once. The full proofs, the
linear/DAG/branch-heavy lemmas (parent-dirty closure on linear traces,
reachability bound on DAGs, fan-out independence on branch-heavy
DAGs), and the streaming-memory floor are in
[`docs/dirty-set-complexity.md`](./dirty-set-complexity.md) (Step 58
of [`100_STEPS.md`](../100_STEPS.md)). Step 60 (partial recompute)
and Step 61 (stale-cache detection) are tracked there as future work
that would tighten the memory bound to `Θ(width(G) · O_avg + m)`.

## 8. Versioning

This document describes `dirty_set_version="1"`. The version is
**implicit** in v1 traces — it is currently bound to
`canonicalisation_version="1"` and is not yet a separate header field.
Any of the following changes MUST bump it:

* a new `dirty_reason` tag,
* a new declared-dependency channel beyond `"context"` and
  `"branch_tail_hashes"`,
* relaxation or tightening of A1 (the conservative dependency model),
* any change to `RaiseSubstitution` propagation semantics
  (now specified in §5.6),
* any change to the `nondeterminism_hash` consistency check (Step 63).

Old replay engines MUST refuse to classify traces whose
`canonicalisation_version` they do not understand. New replay engines
MUST honour `dirty_set_version="1"` semantics on v1 traces forever; v2
features are negotiated through capability frames per
[`spec/sbtrace-v1.md`](../spec/sbtrace-v1.md) §6.

## 9. Worked example and reconciliation (Step 67)

### 9.1 The 12-step linear fixture

`stepback.testing.run_recorded_agent` records a 12-step
customer-payments pipeline (6 alternating `llm_call` + `tool_call`
steps in a single linear chain — no parallel branches). Under
`ToolOutputSubstitution(at_step="step:2")` (targeting the
`lookup_customer` step):

```python
from stepback.divergence import compute_dirty_set
from stepback.substitutions import ToolOutputSubstitution

ds = compute_dirty_set(trace, [ToolOutputSubstitution(
    at_step="step:2",
    fake_response={"result": LOOKUP_FIXED_ROW},
)])
assert ds.dirty_count == 11  # steps 2–12, all dirty
```

`compute_dirty_set` (the conservative static classifier) marks 11 of
12 steps dirty.  This is the expected answer:

* step:2 is directly substituted → P3.
* Every subsequent step carries a `"context"` field equal to the
  **hash** of its parent's output.  Because step:2's forced output
  differs from the recorded output, step:2's output hash changes.
  step:3's `"context"` is rebound to that new hash, so
  `hash(step:3.inputs) ≠ recorded_inputs_hash` → P2 dirtying.
* By P1 (parent-dirty closure), the dirty flag propagates through
  steps 4–12 regardless of whether re-executed outputs happen to
  match recorded outputs.

### 9.2 Why this is the worst case, not the typical case

The 12-step fixture has the most pessimistic shape for the
dirty-set headline:

| Property | 12-step fixture | Parallel branch fixture |
|----------|----------------|------------------------|
| Structure | Single linear chain | Fan-out / fan-in DAG |
| Sibling isolation | None (single branch) | Full (B1–B3) |
| Dirty steps under step:2 sub | **11 / 12 (92 %)** | **3 / 11 (27 %)** |
| Dirty fraction (static) | O(N − k), k = 1 | O(1) per clean sibling |

From `bench-results/dirty-set-distributions.json` (20 trials each,
50 steps per corpus):

| Corpus | Position | sub_kind | Median dirty fraction |
|--------|----------|----------|-----------------------|
| `linear_chain` | random | PromptSubstitution | 0.48 |
| `linear_chain` | late | ToolOutputSubstitution | 0.18 |
| `parallel_wide` | random | ToolOutputSubstitution | 0.17 |
| `parallel_wide` | late | ToolOutputSubstitution | 0.21 |
| `agent_fixture` (12 steps) | late | ToolOutputSubstitution | 0.25 |

Small dirty sets arise from **parallel branches** (B1–B3 sibling
isolation) or **late-position substitutions** (short downstream
suffix), not from universal properties of the algorithm.

### 9.3 Runtime vs. static classifier

`compute_dirty_set` is a *conservative static analysis* (parent-dirty
closure, P1).  The runtime replay engine (`run_replay`, Step 61)
applies *stale-cache detection*: after re-executing a dirty step it
checks whether the output hash *actually changed*.  If the output is
identical to the recorded value (e.g. a deterministic fake executor
produces the same result despite the changed context hash), downstream
steps are cache hits even though the static classifier marked them
dirty.

On the 12-step fixture with `fake_llm` from `stepback.testing`:

```
compute_dirty_set  →  dirty_count = 11  (conservative upper bound)
run_replay(fake)   →  dirty_count =  2  (step:2 forced + step:3
                                          re-executed but output
                                          unchanged → steps 4–12
                                          stale-cache hits)
```

With a real, non-deterministic LLM the re-executed output will
typically differ, so the runtime dirty count will match or exceed the
static estimate.  **The static classifier is always a sound upper
bound; the runtime may find fewer dirty steps when outputs are stable
under re-execution.**

### 9.4 The "median dirty-set of size 3" claim

An earlier version of the README cited a median dirty-set of 3 for a
single-branch substitution on the 11-step parallel-branch fixture
(`stepback.testing.run_parallel_agent`).  This is correct for that
fixture: the substitution targets one branch, dirties that branch plus
the join and the downstream synthesise step — exactly 3 steps — while
the two sibling branches remain clean (B1–B3).  The claim does not
generalise to linear chains, where dirty-set size is O(N − k).

The current README makes no specific numeric claim about dirty-set
size; it states only the asymptotic bound O(dirty\_set) vs. O(N).
The benchmark data in `bench-results/` provides the empirically
measured distributions for synthetic corpora.

## 10. Cross-references

* [`stepback/divergence.py`](../stepback/divergence.py) §"Dirty-set
  propagation" — contract block (P1–P5, A1–A3) and
  `compute_dirty_set` reference implementation.
* [`stepback/replay.py::Trace.run_replay`](../stepback/replay.py) —
  topological walk, `"context"` and `"branch_tail_hashes"` rebinding,
  cache-hit accounting.
* [`stepback/substitutions.py::RaiseSubstitution`](../stepback/substitutions.py) —
  implementation of the forced-error output-forcing substitution
  specified in §5.6.
* [`tests/test_raise_substitution_semantics.py`](../tests/test_raise_substitution_semantics.py) —
  normative tests pinning the §5.6 semantics.
* [`spec/sbtrace-v1.md`](../spec/sbtrace-v1.md) — wire format and
  recorded-field schema referenced throughout this document.
* [`docs/canonicalization.md`](./canonicalization.md) — `J` and `H`
  definitions referenced in §2 and §6.
* [`100_STEPS.md`](../100_STEPS.md) §"Dirty-set algorithm"
  (Steps 53–67) — the roadmap of items that depend on this document
  being stable.
