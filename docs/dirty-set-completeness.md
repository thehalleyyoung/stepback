# Dirty-set replay completeness — paper proof (v1)

This document is the paper-grade companion to the **completeness**
clause of the dirty-set classifier. It is the sibling of
[`docs/dirty-set-soundness.md`](./dirty-set-soundness.md) (which
discharges Step 55 of [`100_STEPS.md`](../100_STEPS.md)) and discharges
Step 57:

> *"State completeness separately: every step whose recomputed inputs
> hash differently is included in the dirty set."*

Soundness and completeness are intentionally treated as separate
artifacts. They constrain the classifier in **opposite directions**:

| Property | Forbids | Failure mode if violated |
|---|---|---|
| **Soundness (P1)** | cleaning a step whose recorded outputs would diverge from re-execution | *under-dirtying* — stale cached outputs corrupt the replay |
| **Completeness (P2)** | dirtying *fewer* steps than the four declared reasons require, in particular missing a step with input drift | a *trivial* implementation that always returns CLEAN; defeats the cache-correctness guarantee at its semantic boundary |
| **Minimality (P5)** | dirtying *more* steps than the four declared reasons require | a *trivial* implementation that always returns DIRTY; collapses to full re-execution and defeats the performance claim |

Soundness is a *correctness* constraint. Completeness is a constraint
on the classifier's *coverage* of dirty-by-reality conditions and is
the half of the contract that closes the loop with soundness.
Minimality is the dual constraint and bounds *over-dirtying*. All
three are stated in [`docs/dirty-set.md`](./dirty-set.md) §6 as
theorem statements; this document gives a full paper proof of P2.

Notation tracks [`docs/dirty-set.md`](./dirty-set.md) and
[`docs/dirty-set-soundness.md`](./dirty-set-soundness.md) verbatim.
We pin `canonicalisation_version="1"` and `dirty_set_version="1"`.

## 0. Scope

We prove the **completeness theorem** (P2) of the dirty-set classifier
as implemented by `compute_dirty_set` in
[`stepback/divergence.py`](../stepback/divergence.py):

> Under the recorder obligations **R1–R4** and assumption **A3**
> (canonical-encoder injectivity / collision-freeness on the input
> distribution), every step `s ∈ V(T)` whose post-substitution canonical
> inputs hash to a value different from `s.inputs_hash` is in the dirty
> set `D(T, σ)`. By the same classifier, every transitive descendant of
> a directly-targeted, input-drifted, parent-dirty, or
> nondeterminism-tampered step is also in `D(T, σ)`.

Explicitly out of scope of this document:

* **Soundness (P1)** — discharged by
  [`docs/dirty-set-soundness.md`](./dirty-set-soundness.md) (Step 55)
  and mechanized in Lean 4 under
  [`proofs/lean/Stepback/Soundness.lean`](../proofs/lean/Stepback/Soundness.lean)
  (Step 56).
* **Minimality (P5)** — stated in
  [`docs/dirty-set.md`](./dirty-set.md) §6 as a *contract on the
  classifier shape* and not a deductive consequence of A1–A3 alone.
  A minimality counterexample is a recorder bug, not an algorithm bug.
* **`RaiseSubstitution` semantics** — Step 62.
* **Stochastic / non-pure executors** — A2 is *not* needed for
  completeness because completeness does not invoke the executor; it
  only inspects the recorded `inputs_hash` and the post-substitution
  canonical inputs. Step 36 covers the stochastic regime separately.

## 1. Objects and notation

We reuse the formal objects of
[`docs/dirty-set-soundness.md`](./dirty-set-soundness.md) §1 and
[`docs/dirty-set.md`](./dirty-set.md) §§1–4. To make this document
self-contained for an auditor reading it standalone:

* `J : JSON → bytes` — canonical encoder of
  [`stepback/canonical.py`](../stepback/canonical.py); see
  [`docs/canonicalization.md`](./canonicalization.md). Total,
  deterministic, and injective on the recorded JSON domain (A3 below).
* `H : JSON → {0,1}^256` — `BLAKE2b-256 ∘ J`. Injective on the same
  domain under A3.
* `T = (V(T), E(T), label)` — recorded trace; `label(s)` returns the
  eight recorded fields of [`spec/sbtrace-v1.md`](../spec/sbtrace-v1.md).
* `topo : V(T) → ℕ` — recorded topological order (R4).
* `parents(s) ⊆ V(T)` — multi-set of `s`'s recorded parents.
* `σ` — substitution set partitioned into input-mutating `M_in` and
  output-forcing `M_out`; see
  [`docs/dirty-set-soundness.md`](./dirty-set-soundness.md) §1.
* `inputs(s, out)` — rebinding function of
  [`docs/dirty-set.md`](./dirty-set.md) §2: deep-copies `s.inputs`
  and rebinds `"context"` and `"branch_tail_hashes"` from `out`.
* `current_hash(s, out) := H(inputs(s, out))`.
* `out_σ : V(T) → JSON` — dirty-set replay map of
  [`docs/dirty-set-soundness.md`](./dirty-set-soundness.md) §3
  Definition 2.
* `D(T, σ) ⊆ V(T)` — dirty set computed by `classify` of
  [`docs/dirty-set.md`](./dirty-set.md) §4. `C(T, σ) := V(T) ∖ D(T, σ)`.
* `dirty_reason : D(T, σ) → {substituted, ndh_tamper, parent_dirty,
  input_drift}` — the surfaced tag on `DirtySetEntry.dirty_reason`.

## 2. Assumptions

Completeness has a strictly weaker assumption profile than soundness:
A1 and A2 are **not** required.

**A3 (Canonical encoder injectivity).** `J` is injective on the
recorded JSON domain. Equivalently, `J(a) == J(b) ⇒ a ≡ b` for any
two recorded or recomputable JSON values, where `≡` is JSON equality
up to dictionary ordering. Since `H` factors through `J`,
`H(a) == H(b) ⇒ a ≡ b` under the standard cryptographic assumption
that BLAKE2b-256 is collision-resistant on `J(V_recorded)`.

**Recorder obligations (R1, R3, R4)** — restated from
[`docs/dirty-set.md`](./dirty-set.md) §6:

* **R1.** `s.inputs_hash == H(s.inputs)` for every `s ∈ V(T)`.
* **R3.** `s.nondeterminism_hash == H(s.nondeterminism)` for every
  `s ∈ V(T)`.
* **R4.** `(V(T), E(T))` is a DAG and `topo` respects every edge.

R2 (declared-channel-only cross-step dependencies) is **not used by
completeness**; A1 / R2 only constrain the *upper bound* of P5
(minimality) and are load-bearing in the soundness proof of
[`docs/dirty-set-soundness.md`](./dirty-set-soundness.md). Dropping
R2 cannot break completeness because completeness is monotone in the
classifier: any extra DIRTY clause (e.g. an unstated extra channel)
can only enlarge `D`, which trivially preserves the
"`s` whose recomputed inputs hash differs ⇒ `s ∈ D`" implication.

**A2 (executor purity)** is **not used by completeness**; the
classifier never invokes the executor. Both `inputs(s, out_σ)` and the
hash check in `classify` depend only on the recorded fields and on
`out_σ` at strictly earlier vertices, which are themselves either
recorded outputs (clean steps) or executor outputs (dirty steps) but
fed into the classifier via `out_σ` only as opaque JSON values.

## 3. Definitions

**Definition 1 (Drift step).** A step `s ∈ V(T)` is a **drift step**
under `σ` if
```
current_hash(s, out_σ) ≠ s.inputs_hash.
```
Equivalently (by R1 and A3), `inputs(s, out_σ) ≢ s.inputs` canonically.

We let `Drift(T, σ) := { s ∈ V(T) : current_hash(s, out_σ) ≠ s.inputs_hash }`.

**Definition 2 (Targeted step).** A step `s` is **targeted** by `σ` if
some `σ_i ∈ σ` satisfies `σ_i.target_id == s.step_id`. Let
`Targeted(σ) := { s ∈ V(T) : ∃ σ_i ∈ σ. σ_i.target_id == s.step_id }`.

**Definition 3 (Tampered step).** A step `s` is **tampered** if its
recorded `nondeterminism_hash` is internally inconsistent:
`s.nondeterminism_hash ≠ H(s.nondeterminism)`. Let
`Tamper(T) := { s ∈ V(T) : s.nondeterminism_hash ≠ H(s.nondeterminism) }`.
By R3 and `verify_trace`, `Tamper(T) = ∅` for any trace that has
passed verification; the classifier nevertheless rechecks this clause
in defence-in-depth (clause P4 of [`docs/dirty-set.md`](./dirty-set.md) §4).

**Definition 4 (Parent-dirty step).** A step `s` is **parent-dirty**
under `σ` if some `p ∈ parents(s)` is in `D(T, σ)`. Let
`ParentDirty(T, σ) := { s ∈ V(T) : ∃ p ∈ parents(s). p ∈ D(T, σ) }`.

**Definition 5 (Reachable from a dirty seed).** A step `s` is
**reachable from a dirty seed** under `σ` if there exists a directed
path `s_0, s_1, …, s_k = s` in `(V(T), E(T))` with `s_0 ∈ Targeted(σ) ∪ Tamper(T) ∪ Drift(T, σ)`
and `s_{i+1} ∈ children(s_i)` for `0 ≤ i < k`. Let `Reach(T, σ)` be
the set of all such `s` (including the seeds themselves; `k = 0` is
permitted).

## 4. Theorem and main lemmas

**Theorem 2 (P2, completeness of dirty-set replay).** Let `T` be a
trace satisfying R1, R3, R4, and assume A3. Then

```
Drift(T, σ) ⊆ D(T, σ),
Targeted(σ) ⊆ D(T, σ),
Tamper(T) ⊆ D(T, σ),
ParentDirty(T, σ) ⊆ D(T, σ),
```

and the four inclusions are jointly tight:

```
D(T, σ) = Targeted(σ) ∪ Tamper(T) ∪ Drift(T, σ) ∪ ParentDirty(T, σ).
```

Equivalently, `D(T, σ) = Reach(T, σ)` — the dirty set is exactly the
set of steps reachable from any of the three dirty *seed* sources
(targeted, tampered, drift) by walking forward along recorded parent
edges.

**Corollary 2 (Headline form, Step 57).** *Every step `s ∈ V(T)` such
that `current_hash(s, out_σ) ≠ s.inputs_hash` is in `D(T, σ)`.*

This is the headline statement of Step 57 verbatim and is the first
clause of Theorem 2.

**Corollary 3 (Descendant closure, P2 transitive).** *Every transitive
descendant of any step in `Targeted(σ) ∪ Tamper(T) ∪ Drift(T, σ)` is
in `D(T, σ)`.*

This is the second statement of P2 in `divergence.py` line 530–531.

We prove Theorem 2 via two lemmas: a **structural** lemma (the four
DIRTY clauses of `classify` are exhaustive of `D`) and an **inductive**
lemma (the parent-dirty clause closes `D` under descendants).

**Lemma 2 (classifier exhaustiveness).** For every `s ∈ D(T, σ)`,
`dirty_reason(s) ∈ {substituted, ndh_tamper, parent_dirty,
input_drift}`, and there is no fifth possible value.

*Proof.* Direct inspection of `classify` in
[`docs/dirty-set.md`](./dirty-set.md) §4: the function is a chain of
four guarded `return DIRTY` statements followed by a single
`return CLEAN` fallback. Each `return DIRTY` carries a literal reason
tag, and the four tags exhaust the cases by construction. The
implementation in [`stepback/divergence.py`](../stepback/divergence.py)
populates `DirtySetEntry.dirty_reason` from a closed set of four
string literals (`"substituted" | "input_drift" | "parent_dirty" |
"ndh_tamper"`, documented inline at the field declaration; treated
as a closed sum throughout `compute_dirty_set`). ∎

**Lemma 3 (parent-dirty closure).** Suppose the topological walk of
[`docs/dirty-set.md`](./dirty-set.md) §4 has classified every
`s' ∈ V(T)` with `topo(s') < topo(s)`. If any `p ∈ parents(s)`
satisfies `p ∈ D(T, σ)`, then `s ∈ D(T, σ)` with `dirty_reason(s) ∈
{substituted, parent_dirty}` (the former dominates if `s` is also
targeted; the latter is the relevant tag otherwise).

*Proof.* By R4, `topo(p) < topo(s)` for every `p ∈ parents(s)`, so
`p`'s classification is available when `s` is classified. The third
guard of `classify`,
```
for p in parents(s):
    if classify(p, σ, out) is DIRTY:
        return DIRTY (reason="parent_dirty")
```
returns DIRTY when any parent is in `D`. The first guard
(`reason="substituted"`) precedes the parent-dirty guard and dominates
when `s` is also targeted. ∎

## 5. Proof of Theorem 2

We prove the four inclusions and the equality.

**(a) `Targeted(σ) ⊆ D(T, σ)`.** Let `s ∈ Targeted(σ)`. By
Definition 2 there exists `σ_i ∈ σ` with `σ_i.target_id == s.step_id`.
The first guard of `classify`,
```
if any σ_i with σ_i.target_id == s.step_id:
    return DIRTY (reason="substituted")
```
fires unconditionally. Hence `s ∈ D(T, σ)` with `dirty_reason(s) =
substituted`. ∎

**(b) `Tamper(T) ⊆ D(T, σ)`.** Let `s ∈ Tamper(T)`. By Definition 3,
`s.nondeterminism_hash ≠ H(s.nondeterminism)`. The second guard,
```
if s.nondeterminism_hash != H(s.nondeterminism):
    return DIRTY (reason="ndh_tamper")
```
fires (after the targeting guard, which may already have fired). In
either case `s ∈ D(T, σ)`. (Under R3 + `verify_trace`, `Tamper(T) = ∅`,
so this clause is vacuously satisfied for any verified trace; it
remains an honest implication and is the *only* part of completeness
that is "trivially" true for verified inputs.) ∎

**(c) `ParentDirty(T, σ) ⊆ D(T, σ)`.** Immediate from Lemma 3. ∎

**(d) `Drift(T, σ) ⊆ D(T, σ)`.** Let `s ∈ Drift(T, σ)`. We argue by
strong induction on `topo(s)` that `s ∈ D(T, σ)`.

Strengthen the induction hypothesis to:

**(IH).** For every `s' ∈ V(T)` with `topo(s') ≤ topo(s)`, the
classifier has correctly resolved `s'` against `D(T, σ)` per Lemma 2.

This holds at `topo(s) = 0` vacuously (no strictly earlier vertices)
and persists by Lemma 3.

There are now two sub-cases for `s ∈ Drift(T, σ)`.

* **Sub-case (d.i): some `p ∈ parents(s)` is in `D(T, σ)`.** By
  Lemma 3, `s ∈ D(T, σ)` with `dirty_reason(s) ∈ {substituted,
  parent_dirty}`. ∎ for this sub-case.

* **Sub-case (d.ii): every `p ∈ parents(s)` is in `C(T, σ)`.** Then
  the parent-dirty guard does not fire on `s`. The targeting and
  tampering guards may or may not fire; if either does, `s ∈ D(T, σ)`
  by (a) or (b) and we are done. Otherwise, `classify` reaches the
  fourth guard:
  ```
  if current_hash(s, out) != s.inputs_hash:
      return DIRTY (reason="input_drift")
  ```
  By hypothesis `s ∈ Drift(T, σ)`, hence
  `current_hash(s, out_σ) ≠ s.inputs_hash`. We must show that the
  *value of `current_hash(s, out)` consumed by this guard at
  classification time* equals `current_hash(s, out_σ)`.

  The classifier consumes `out` as the partial output map populated
  by the topological walk: by [`docs/dirty-set.md`](./dirty-set.md) §4,
  for every `p ∈ parents(s)`, `out[p] = s.outputs` (cache reuse) iff
  `p ∈ C(T, σ)`. By the sub-case assumption every parent is clean,
  hence `out[p] = p.outputs` for every parent. The dirty-set replay
  map `out_σ` of [`docs/dirty-set-soundness.md`](./dirty-set-soundness.md)
  Definition 2 likewise sets `out_σ[p] = p.outputs` for every clean
  parent. Therefore `out` and `out_σ` agree pointwise on
  `parents(s)`, and since `inputs(s, ·)` reads `out` only at
  `parents(s)` (this is a property of the rebinding function — it
  does not require A1, only the syntactic shape of `inputs`),
  `current_hash(s, out) = current_hash(s, out_σ)`. The fourth guard
  fires, and `s ∈ D(T, σ)` with `dirty_reason(s) = input_drift`. ∎

This completes the inductive case. By strong induction, every
`s ∈ Drift(T, σ)` is in `D(T, σ)`. ∎ for (d).

**Equality.** The four inclusions give
`Targeted(σ) ∪ Tamper(T) ∪ Drift(T, σ) ∪ ParentDirty(T, σ) ⊆ D(T, σ)`.
The reverse inclusion is exactly Lemma 2: every `s ∈ D(T, σ)` has a
`dirty_reason` in the four-element literal type, and each value of the
literal corresponds to membership in exactly one of the four sets.
Hence the equality. ∎

**`D(T, σ) = Reach(T, σ)`.** Forward inclusion: by induction on
`topo(s)`. If `s ∈ D(T, σ)` and `dirty_reason(s) ∈ {substituted,
ndh_tamper, input_drift}`, then `s` is its own seed and `s ∈ Reach(T,
σ)` with `k = 0`. If `dirty_reason(s) = parent_dirty`, then some
`p ∈ parents(s)` has `p ∈ D(T, σ)` with `topo(p) < topo(s)`; by IH
`p ∈ Reach(T, σ)`, and `Reach` is closed under children by definition,
so `s ∈ Reach(T, σ)`.

Reverse inclusion: by induction on path length `k` of Definition 5.
If `k = 0`, `s` is itself a seed and so is in `D(T, σ)` by (a), (b),
or (d). If `k > 0`, `s_{k-1} ∈ D(T, σ)` by IH, and `s = s_k` is a
child of `s_{k-1}`; by (c) `s ∈ D(T, σ)`. ∎

This proves Corollaries 2 and 3 simultaneously: Corollary 2 is
sub-case (d) read off as a standalone implication; Corollary 3 is the
`D(T, σ) = Reach(T, σ)` equality applied to the descendants of any
seed.

## 6. Where each assumption is used

A reader auditing the proof for assumption-fragility should note:

* **R1** is used implicitly in (d.ii) to interpret `s.inputs_hash` as
  `H(s.inputs)` — without R1, "drift" is defined against a free
  parameter and the implication
  `current_hash(s, out_σ) ≠ s.inputs_hash ⇒ s ∈ D(T, σ)` is still
  *literally* true (it is the contrapositive of the fourth guard of
  `classify`) but its *meaning* — that the post-σ inputs differ from
  the recorded inputs — is lost. R1 is what makes Corollary 2 a
  cache-correctness statement and not a tautology over an arbitrary
  hash field.
* **R3** is used in (b) to guarantee that `Tamper(T) = ∅` for any
  trace that has passed `verify_trace`; the classifier nevertheless
  rechecks this clause in defence-in-depth.
* **R4** underwrites the topological walk of `classify` and the
  strong induction in (d). Without R4, `parents(s)` may not be fully
  classified when `s` is reached, and Lemma 3 becomes ill-defined.
* **A3** is used in Lemma 2's auxiliary claim that `dirty_reason ∈
  {substituted, ndh_tamper, parent_dirty, input_drift}` is *meaningful*
  — i.e. that the `input_drift` clause means what its name says.
  Without A3, two distinct JSON values could share an `inputs_hash`
  and the fourth guard would correctly mark a step DIRTY under a
  different labelling than its semantic name suggests. A3 is therefore
  used identically by soundness (where it lifts hash-equality to
  JSON-equality) and by completeness (where it lifts hash-inequality
  to JSON-inequality).

**A1, A2, R2 are not used.** This is a load-bearing observation: the
completeness theorem of P2 holds *even if the recorder declares cross-
step dependencies through channels other than `"context"` and
`"branch_tail_hashes"`*. A1/R2 violations only manifest as soundness
failures (under-dirtying via the `input_drift` clause failing to
detect the drift, because the rebinding function does not see the
hidden channel). They cannot turn a step that *would* drift under the
declared rebinding into a non-dirty step — they can only *mask* drift
that the recorder failed to expose. The asymmetry is part of the
classifier's design: completeness is the half that survives recorder
sloppiness; soundness is the half that does not.

## 7. Counterexamples to motivate each assumption

These constructions are not run in CI; they are *intended*
counterexamples that justify why each assumption is necessary. They
should be implementable as tests by the property/fuzz harnesses of
Steps 29 and 49.

* **R1 violation:** A trace with `s.inputs == {"q": "x"}` and
  `s.inputs_hash == H({"q": "y"})` is rejected by `verify_trace`
  before classification. Without `verify_trace`, the fourth guard of
  `classify` would compare `current_hash(s, out_σ)` against a hash of
  *some other* value; the resulting DIRTY/CLEAN call would no longer
  reflect input drift. The cache-correctness implication of
  Corollary 2 evaporates. (This is exactly why `verify_trace` is
  mandatory for the completeness statement to be cache-meaningful.)
* **R3 violation:** A trace with
  `s.nondeterminism_hash ≠ H(s.nondeterminism)` is rejected by
  `verify_trace`; if it sneaks past, clause P4 of the classifier
  catches it and `s ∈ Tamper(T) ⊆ D(T, σ)` by (b). Completeness
  *cannot fail* on tampered traces — it can only become vacuous.
* **R4 violation:** A trace with a cycle `s → s' → s` cannot be
  topologically walked; `classify` is undefined and the proof of
  Lemma 3 is ill-formed. This is why `verify_trace` rejects cycles.
* **A3 violation (synthetic hash collision):** With probability
  `2^{-256}` a recomputed `current_hash(s, out_σ)` collides with an
  unrelated `s.inputs_hash`, the fourth guard fails to fire, and
  `s ∉ D(T, σ)` even though `inputs(s, out_σ) ≢ s.inputs` canonically.
  This is the cryptographic risk that all hash-based replay caches
  inherit and is treated as negligible.
* **(Why A1 is not listed.)** A1 / R2 violations cannot break
  completeness; they break soundness. The classifier's input-drift
  guard is *robust* to undeclared cross-step dependencies because the
  guard does not need to see the dependency to fire — it only needs
  the canonical inputs to differ. (It can fail to *fire when it
  should*, but that is a soundness gap, not a completeness gap.)

## 8. Relationship to the implementation

Each statement of Theorem 2 corresponds to a specific code line:

| Proof statement | Implementation site |
|---|---|
| `Targeted(σ) ⊆ D(T, σ)` (a) | `compute_dirty_set` in [`stepback/divergence.py`](../stepback/divergence.py), the `reason="substituted"` branch. |
| `Tamper(T) ⊆ D(T, σ)` (b) | `compute_dirty_set`, the `reason="ndh_tamper"` branch. |
| `ParentDirty(T, σ) ⊆ D(T, σ)` (c) | `compute_dirty_set`, the `reason="parent_dirty"` branch. |
| `Drift(T, σ) ⊆ D(T, σ)` (d) | `compute_dirty_set`, the `reason="input_drift"` branch. |
| Lemma 2 exhaustiveness | `DirtySetEntry.dirty_reason` field in `stepback/divergence.py` line ~615, declared as `Optional[str]` with the inline comment `"substituted" | "input_drift" | "parent_dirty" | "ndh_tamper" | None`; `compute_dirty_set` only ever assigns from this closed set. |
| Lemma 3 parent-dirty closure | `compute_dirty_set` walks `recorded_steps` in topological order; the parent-dirty guard reads from a `dirty_ids` set populated for strictly earlier vertices. |
| Topological walk (R4) | `compute_dirty_set`, the outer `for rec in trace.recorded_steps` loop. |
| Equality `D = Targeted ∪ Tamper ∪ Drift ∪ ParentDirty` | `Literal[…]` totality of `dirty_reason` plus the closed disjunction in `compute_dirty_set`. |

The mapping is *line-level* so any future refactor of `classify` is
forced to preserve or explicitly bump `dirty_set_version`
([`docs/dirty-set.md`](./dirty-set.md) §8).

## 9. From paper proof to mechanization

The Lean 4 development under [`proofs/lean/`](../proofs/lean/)
currently mechanizes only soundness (Step 56). A future mechanization
of completeness is straightforward given the same framework:

* `Drift T σ : Set Step` — the set defined by
  `current_hash s out_σ ≠ s.inputs_hash`.
* `Targeted σ : Set Step`, `Tamper T : Set Step`,
  `ParentDirty T σ : Set Step` — direct transcriptions of
  Definitions 2–4.
* `Reach T σ : Set Step` — inductive type with two constructors
  (`seed` and `child`), or a fixed-point definition over the parent
  relation.
* **Theorem 2** discharges as a `case`-split on `dirty_reason` in
  the implementation of `classify`, plus strong induction on
  `topo(s)` for the input-drift case (mirroring §5(d) verbatim).

Because completeness does not need A1, A2, or R2, the mechanization
has a strictly smaller assumption surface than `Stepback.soundness`.
In particular, the `executor_pure` axiom of Soundness.lean is *not*
needed and the Lean development can drop it for the completeness
side.

The trust base is identical: Lean 4 kernel only, no `axiom`, no
`sorry`, no `partial`. We expect to land this as `Stepback.completeness`
in the same `proofs/lean/Stepback/` namespace as part of a future
mechanization step (out of scope of Step 57, which is a paper proof).

## 10. Cross-references

* [`docs/dirty-set.md`](./dirty-set.md) — prose definitions and
  high-level theorem statement (P2 in §6).
* [`docs/dirty-set-soundness.md`](./dirty-set-soundness.md) — the
  sibling document for soundness (P1, Step 55) with the parent-
  agreement lemma and assumption-tightness audit.
* [`stepback/divergence.py`](../stepback/divergence.py) lines
  470–596 — contract block (P1–P5, A1–A3) and `compute_dirty_set`
  reference implementation. Postcondition (P2) on lines 528–531
  is the one-line statement that this document discharges.
* [`stepback/replay.py`](../stepback/replay.py) — `Trace.run_replay`
  reference implementation; cache reuse is a property of replay, not
  of the classifier this document proves complete.
* [`docs/canonicalization.md`](./canonicalization.md) — `J` and `H`.
* [`spec/sbtrace-v1.md`](../spec/sbtrace-v1.md) — wire format and the
  recorded fields the proof inspects.
* [`proofs/lean/`](../proofs/lean/) — Lean 4 mechanization of P1;
  a future P2 mechanization would land in the same workspace.
* [`100_STEPS.md`](../100_STEPS.md) Steps 53–67 — the full
  algorithm workstream this proof is part of. Step 55 is discharged
  by [`docs/dirty-set-soundness.md`](./dirty-set-soundness.md);
  Step 56 by [`proofs/lean/Stepback/Soundness.lean`](../proofs/lean/Stepback/Soundness.lean);
  Step 57 by this document.
