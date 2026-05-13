# Dirty-set replay soundness — paper proof (v1)

This document is the paper-grade companion to the proof *sketches* in
[`stepback/divergence.py`](../stepback/divergence.py) (lines 574–590)
and [`docs/dirty-set.md`](./dirty-set.md) §6. It discharges Step 55 of
[`100_STEPS.md`](../100_STEPS.md) — *"Prove soundness on paper: every
non-dirty replayed step has the same observable output as full
re-execution under sigma."*

It is intentionally textual / mathematical and is the input artefact
for Step 56 (Lean/Coq mechanization), which has now landed under
[`proofs/lean/`](../proofs/lean/) — see `Stepback.soundness` in
[`proofs/lean/Stepback/Soundness.lean`](../proofs/lean/Stepback/Soundness.lean)
for the kernel-checked Lean 4 statement, and
[`proofs/lean/README.md`](../proofs/lean/README.md) for the
symbol-by-symbol cross-reference into this document. Notation follows
[`docs/dirty-set.md`](./dirty-set.md) verbatim. We pin
`canonicalisation_version="1"` and `dirty_set_version="1"`.

## 0. Scope

We prove the **soundness theorem** (P1) of the dirty-set classifier as
implemented by `compute_dirty_set` in
[`stepback/divergence.py`](../stepback/divergence.py) and consumed by
`Trace.run_replay` in
[`stepback/replay.py`](../stepback/replay.py):

> Under assumptions **A1** (conservative dependency model), **A2**
> (executor purity), and **A3** (canonical-JSON / BLAKE2b-256
> collision-freeness on the input distribution), and the recorder
> obligations **R1–R4** of [`docs/dirty-set.md`](./dirty-set.md) §6,
> for every clean step `s ∈ V(T) ∖ D(T, σ)`, reusing the recorded
> output `s.outputs` as `out_σ[s]` is observationally equivalent to
> re-executing `s` under `σ` from scratch.

Explicitly out of scope of this document:

* **Completeness** (P2 in [`docs/dirty-set.md`](./dirty-set.md) §6) —
  separately discharged by
  [`docs/dirty-set-completeness.md`](./dirty-set-completeness.md)
  (Step 57). That document has a strictly smaller assumption profile
  (no A1, no A2, no R2) and is the dual of this document.
* **Minimality** (P5) — stated in [`docs/dirty-set.md`](./dirty-set.md)
  §6 as a *contract on the classifier shape* and not a deductive
  consequence of A1–A3 alone.
* **`RaiseSubstitution` semantics** — see Step 62; v1 forbids
  substitutions of `kind == "raise"` from being treated as clean
  through descendants.
* **Stochastic / non-pure executors** — A2 is assumed, *not* proved.
  The interface `Executor.execute(kind, inputs)` is treated as a
  total function. Step 36 covers stochastic relaxations.
* **Cryptographic adversaries** — A3 is assumed, *not* proved. We do
  not analyse a forging adversary against BLAKE2b-256.

## 1. Objects and notation

We instantiate the formal objects of [`docs/dirty-set.md`](./dirty-set.md)
§§1–4:

* `J : JSON → bytes` is the canonical encoder of
  [`stepback/canonical.py`](../stepback/canonical.py) /
  [`docs/canonicalization.md`](./canonicalization.md). `J` is a
  *function* — total, deterministic, and injective on the JSON values
  produced by recorders (assumption A3 below).
* `H : JSON → {0,1}^256` is `BLAKE2b-256 ∘ J`. Under A3, `H` is
  injective on the same domain as `J`.
* `T = (V(T), E(T), label)` is a recorded trace where `label(s)`
  returns the eight recorded fields of [`spec/sbtrace-v1.md`](../spec/sbtrace-v1.md):
  `step_id, step_kind, parent_step_id (or parent_step_ids),
   inputs, outputs, inputs_hash, nondeterminism, nondeterminism_hash`.
* `topo : V(T) → ℕ` is the recorded topological order. By R4, every
  edge `(p, s) ∈ E(T)` satisfies `topo(p) < topo(s)`.
* `parents(s) ⊆ V(T)` is the multi-set of `s`'s recorded parents.
  `|parents(s)| ∈ {0, 1, w}` where `w ≥ 2` only if
  `s.step_kind == "parallel_branch_join"`.
* `σ` is a substitution set; `σ_i = (target_id, kind, payload)`.
  Substitution-kind families are partitioned into
  *input-mutating* `M_in = {prompt, tool_input, router_decision, …}`
  and *output-forcing* `M_out = {tool_output, output_force, raise}`.
  These two sets are disjoint by construction in
  [`stepback/substitutions.py`](../stepback/substitutions.py).
* `inputs(s, out)` is the rebinding function defined in
  [`docs/dirty-set.md`](./dirty-set.md) §2: it deep-copies `s.inputs`
  and (i) rebinds `"context"` to `H(out[parent_step_id(s)])` when the
  parent's output is known, and (ii) rebinds each
  `"branch_tail_hashes"[i]` to `H(out[branch_tails[i]])` likewise.
  These are the **only two** declared cross-step dependency channels
  in v1 (A1). `current_hash(s, out) := H(inputs(s, out))`.
* `out : V(T) ⇀ JSON` is a partial output assignment.
* `D(T, σ) ⊆ V(T)` is the dirty set computed by `classify` in
  [`docs/dirty-set.md`](./dirty-set.md) §4. `C(T, σ) := V(T) ∖ D(T, σ)`
  is the *clean set*.
* `obs(s, out) := J(out[s])` is the *observable* of `s` under `out`.

We freely identify a JSON value with its canonical-JSON byte sequence
when the context disambiguates; equality `==` between JSON values
denotes *canonical-byte equality* (i.e., `J(a) == J(b)`).

## 2. Assumptions

**A1 (Conservative dependency model).** For every step `s ∈ V(T)`
and every parent-edge `(p, s) ∈ E(T)`, the recorder declares the
dependency of `s` on `p` either as a single-parent `"context"` field
in `s.inputs` or, when `s.step_kind == "parallel_branch_join"`, as an
entry of `s.inputs["branch_tail_hashes"]` aligned with
`s.inputs["branch_tails"]`. No other channel of cross-step
dependency is permitted in v1. (Any violation is a *recorder bug*
addressed by R2; see §10.)

**A2 (Executor purity).** For every dirty step `s ∈ D(T, σ)`,
`Executor.execute(s.step_kind, ·) : JSON → JSON` is a *deterministic
total function* for the duration of one `Trace.run_replay` call. This
includes a fixed seed for any stochastic provider, fixed
`temperature`, fixed model id, and a frozen tool implementation.
Stochastic executors are out of scope of P1; see Step 36.

**A3 (Canonical encoder injectivity).** `J` is injective on the
domain of recorded JSON values. Equivalently, for any two recorded or
re-executable JSON values `a`, `b`, `J(a) == J(b) ⇒ a ≡ b` (where
`≡` is JSON value equality up to dictionary ordering, i.e. the
equivalence the canonicalizer collapses). Consequently, since `H`
factors through `J`, `H(a) == H(b) ⇒ a ≡ b` *under the standard
cryptographic assumption that BLAKE2b-256 is collision-resistant on
the input distribution `J(V_recorded)`*. In the proof we therefore
treat `H(a) == H(b)` as `a ≡ b`.

**Recorder obligations (R1–R4)** are restated from
[`docs/dirty-set.md`](./dirty-set.md) §6 and are pre-conditions for
the trace `T` itself, enforced by `verify_trace`:

* **R1.** `s.inputs_hash == H(s.inputs)` for every `s ∈ V(T)`.
* **R2.** Every cross-step dependency of `s` on a parent's output is
  declared as `"context"` or `"branch_tail_hashes"`. (This is the
  recorder side of A1.)
* **R3.** `s.nondeterminism_hash == H(s.nondeterminism)` for every
  `s ∈ V(T)`.
* **R4.** `(V(T), E(T))` is a DAG and `topo` respects every edge.

## 3. Definitions

**Definition 1 (Full re-execution under `σ`).** The *full re-execution*
output map `out_full : V(T) → JSON` is the unique map produced by
walking `V(T)` in `topo` order and, for every `s`:

* if some `σ_i ∈ σ ∩ M_out` targets `s`, set `out_full[s] := σ_i.payload`;
* otherwise, set
  `out_full[s] := Executor.execute(s.step_kind, inputs_full(s))`,

where `inputs_full(s)` is `inputs(s, out_full)` with every
input-mutating substitution `σ_j ∈ σ ∩ M_in` targeting `s` applied to
the corresponding field of `s.inputs` *before* the rebinding of
`"context"` and `"branch_tail_hashes"`. Equivalently, `out_full` is
the result of `Trace.run_replay(σ, executor, dirty_set := V(T))`,
i.e. as if every step were dirty and recomputed from scratch.

`out_full` exists and is unique by induction over `topo` order: at
each step the inputs depend only on outputs of strictly earlier
vertices (R4) and either the executor (A2) or an output-forcing
substitution determines the output unambiguously.

**Definition 2 (Dirty-set replay under `σ`).** The *dirty-set replay*
output map `out_σ : V(T) → JSON` is the unique map produced by the
algorithm of [`docs/dirty-set.md`](./dirty-set.md) §5, namely: for
every `s` in `topo` order,

* if `s ∈ C(T, σ)`, set `out_σ[s] := s.outputs` (cache reuse);
* if `s ∈ D(T, σ)` and some `σ_i ∈ σ ∩ M_out` targets `s`, set
  `out_σ[s] := σ_i.payload`;
* if `s ∈ D(T, σ)` and no output-forcing substitution targets `s`,
  set `out_σ[s] := Executor.execute(s.step_kind, inputs_σ(s))`,

where `inputs_σ(s)` is the same construction as `inputs_full(s)` but
with `out_σ` in place of `out_full` for the parent-output rebindings.

**Definition 3 (Observational equivalence at `s`).**
`s ⊨ obs_σ ≡ obs_full` iff `J(out_σ[s]) == J(out_full[s])`.

## 4. Theorem and main lemma

**Theorem 1 (P1, soundness of dirty-set replay).** Let `T` be a trace
satisfying R1–R4, `σ` any substitution set, and `Executor` an
executor satisfying A2. Assume A1 and A3. Then for every clean step
`s ∈ C(T, σ)`,

```
J(out_σ[s]) == J(out_full[s]).
```

In particular, for *every* `s ∈ V(T)` (whether clean or dirty), the
parent-output values consumed by descendants are pairwise canonically
equal between the two replay maps.

**Corollary 1 (P1 globally).** `obs(T, out_σ) == obs(T, out_full)` —
i.e. the two replays produce canonically identical trace observables.

We prove Theorem 1 by strong induction on `topo(s)`. The key
inductive lemma is:

**Lemma 1 (parent agreement).** Let `s ∈ V(T)`. Suppose for every
`p ∈ parents(s)`, `J(out_σ[p]) == J(out_full[p])`. Then

```
J(inputs_σ(s)) == J(inputs_full(s)).
```

*Proof of Lemma 1.* By A1, `inputs(·, out)` only inspects `out` at
indices `p ∈ parents(s)` and only via `H(out[p])`. By the parent
agreement hypothesis and A3,

```
H(out_σ[p]) == H(J(out_σ[p])) == H(J(out_full[p])) == H(out_full[p])
```

for every parent `p`. Hence the two rebinding operations
(`"context"` and `"branch_tail_hashes"[i]`) substitute byte-identical
values into byte-identical input scaffolds. Input-mutating
substitutions `σ_j ∈ σ ∩ M_in` targeting `s` are applied identically
in both `inputs_σ` and `inputs_full` (they read only from `σ.payload`,
not from `out`). Therefore `inputs_σ(s)` and `inputs_full(s)` are JSON
values with canonically identical encodings, and
`J(inputs_σ(s)) == J(inputs_full(s))`. ∎

## 5. Proof of Theorem 1

Let `s ∈ V(T)`. We strengthen the induction hypothesis to cover *all*
steps (not just clean ones), to make the parent-agreement hypothesis
of Lemma 1 available unconditionally:

**(IH).** For every `s' ∈ V(T)` with `topo(s') < topo(s)`,
`J(out_σ[s']) == J(out_full[s'])`.

Base case (`topo(s) = 0`, no parents): `s` is a root.

* If `s ∈ D(T, σ)` and an output-forcing `σ_i` targets `s`, then both
  `out_σ[s] = σ_i.payload = out_full[s]`. ✓
* If `s ∈ D(T, σ)` and no output-forcing substitution targets `s`,
  then both maps re-execute `s`. By Lemma 1 (vacuously, no parents)
  and any input-mutating `σ_j` applied identically,
  `inputs_σ(s) ≡ inputs_full(s)` canonically. By A2,
  `out_σ[s] = Executor.execute(s.step_kind, inputs_σ(s)) =
   Executor.execute(s.step_kind, inputs_full(s)) = out_full[s]`. ✓
* If `s ∈ C(T, σ)`, the classifier of [`docs/dirty-set.md`](./dirty-set.md)
  §4 guarantees:
  (a) no `σ_i` targets `s` (P3 not triggered);
  (b) `s.nondeterminism_hash == H(s.nondeterminism)` (P4 not triggered;
      enforced by R3 and rechecked by `classify`);
  (c) every parent of `s` is clean (P2 transitive not triggered) —
      vacuously true for a root;
  (d) `current_hash(s, ∅) == s.inputs_hash` (P2 input-drift not
      triggered).

  Then `cache reuse` sets `out_σ[s] := s.outputs`, while `out_full`
  re-executes `s`. We must show `s.outputs ≡ out_full[s]` canonically.

  By (d) and A3, `H(inputs(s, ∅)) == s.inputs_hash`. By R1,
  `s.inputs_hash == H(s.inputs)`. Therefore
  `H(inputs(s, ∅)) == H(s.inputs)`, hence by A3
  `J(inputs(s, ∅)) == J(s.inputs)` and the JSON values are
  canonically equal. Because no input-mutating substitution targets
  `s` (by (a)), `inputs_full(s) = inputs(s, ∅)` likewise, and so
  `inputs_full(s) ≡ s.inputs` canonically. By A2,

  ```
  out_full[s] = Executor.execute(s.step_kind, inputs_full(s))
              = Executor.execute(s.step_kind, s.inputs).
  ```

  But `Executor.execute(s.step_kind, s.inputs)` is *exactly* the
  output the recorder observed at record time — that is the meaning
  of `s.outputs` under R1 + recorder contract. Hence
  `out_full[s] ≡ s.outputs ≡ out_σ[s]` canonically. ✓

Inductive step (`topo(s) > 0`): assume (IH).

* **Case `s ∈ D(T, σ)`, output-forced:** identical to the root case;
  both maps assign the same `σ_i.payload`. ✓
* **Case `s ∈ D(T, σ)`, re-executed:** by (IH), the parent-agreement
  hypothesis of Lemma 1 holds; therefore
  `J(inputs_σ(s)) == J(inputs_full(s))`, and by A2 the two executor
  invocations return canonically equal outputs. ✓
* **Case `s ∈ C(T, σ)`:** the classifier guarantees (a)–(d) of the
  base case verbatim. We must establish that
  `J(inputs(s, out_σ)) == J(s.inputs)` so that the same equation as
  the root case yields `s.outputs ≡ out_full[s]`.

  By (d), `current_hash(s, out_σ) == s.inputs_hash`. By A3,
  `J(inputs(s, out_σ)) == J(s.inputs)`. (This is the *only* place A3
  is used to convert a hash equality back to a JSON equality, and it
  is the cryptographic load-bearing step of the proof.) By R1,
  `J(s.inputs) == J(s.inputs)` trivially.

  Now we must align `inputs(s, out_σ)` with `inputs_full(s)`. By
  Lemma 1 with `out := out_σ` and `out := out_full`,
  `J(inputs_σ(s)) == J(inputs_full(s))`; and because no
  input-mutating substitution targets `s` (clause (a)),
  `inputs_σ(s) = inputs(s, out_σ)`. Composing,

  ```
  J(inputs_full(s)) == J(inputs_σ(s)) == J(inputs(s, out_σ))
                                       == J(s.inputs).
  ```

  By A2, `out_full[s] = Executor.execute(s.step_kind, inputs_full(s)) =
  Executor.execute(s.step_kind, s.inputs) = s.outputs = out_σ[s]`. ✓

In all three cases, `J(out_σ[s]) == J(out_full[s])`, completing the
induction step. By strong induction over `topo`, the theorem holds
for every `s ∈ V(T)`. ∎

*Proof of Corollary 1.* Apply Theorem 1 component-wise:
`obs(T, out_σ) = (J(out_σ[s_i]))_i = (J(out_full[s_i]))_i =
obs(T, out_full)`. ∎

## 6. Where each assumption is used

A reader auditing the proof for assumption-fragility should note:

* **R1** is used exactly once (clean-step case (d) ⇒ JSON equality of
  `inputs(s, ∅)` with `s.inputs`). Without R1, `s.inputs_hash` is a
  free parameter and the classifier's hash check carries no
  information about `s.inputs`.
* **R2 / A1** is used in Lemma 1: the rebinding function inspects
  parent outputs *only* through the two declared channels, so parent
  agreement at the JSON level lifts to input agreement at the JSON
  level. If a recorder secretly threads a parent output through some
  other field, A1 fails, the classifier may declare `s` clean even
  though its true inputs changed, and **soundness fails** — this is
  the under-dirtying failure mode of
  [`docs/dirty-set.md`](./dirty-set.md) §1.
* **R3 + classifier clause P4** are used to reject tampered
  `nondeterminism_hash` values. The proof above does *not* prove that
  any cache reuse is safe in the presence of recorder tampering; it
  proves that cache reuse is safe *given that the trace passed
  `verify_trace`*, which is exactly the contract.
* **R4** underwrites the well-foundedness of strong induction on
  `topo`.
* **A2** is used twice (root and inductive step) to commute
  re-execution with input-equality. Without A2 (e.g. a stochastic
  executor), `Executor.execute(s.step_kind, s.inputs)` may return
  values not equal to `s.outputs`, and the cache reuse becomes a
  *probabilistic* equivalence rather than an observational one. Step
  36 / §5 of `docs/dirty-set.md` covers this regime.
* **A3** is used in two places: (i) Lemma 1 lifts hash equality of
  parent outputs to JSON equality, and (ii) the clean-step
  `current_hash(s, out_σ) == s.inputs_hash` deduction. A3 is the
  cryptographic axiom of the proof; *all* hash-based replay caches
  inherit this assumption.

If any one of A1–A3 / R1–R4 is dropped, exactly one named clause of
the proof breaks, and which one is enumerated above. This makes the
proof *assumption-tight*: weakening any assumption produces a
specific, named counterexample, not a vague "robustness gap".

## 7. Counterexamples to motivate each assumption

These constructions are not run in CI; they are *intended*
counterexamples that justify why each assumption is necessary. They
should be implementable as tests by the property/fuzz harnesses of
Steps 29 and 49.

* **A1 violation (under-dirtying):** Step `s` records
  `s.inputs == {"q": "what's the weather?"}` but its tool
  implementation reads `os.environ["LOCATION"]`. A substitution that
  changes a parent's output (which had previously written
  `LOCATION`) produces no `"context"` rebinding, no input drift,
  classifier returns CLEAN, cache reuse serves the recorded weather
  for the wrong city. This is precisely the failure mode that the
  recorder must surface via the `nondeterminism` field (R3) or by
  declaring an explicit `"context"` link.
* **R1 violation:** A trace with `s.inputs == {"q": "x"}` and
  `s.inputs_hash == H({"q": "y"})` is rejected by `verify_trace`
  before classification, so no soundness claim is asserted. (This is
  exactly why `verify_trace` is mandatory.)
* **A2 violation (stochastic executor):** A clean step `s` is reused
  from cache; meanwhile a sibling re-execution path through
  `out_full` re-samples the LLM and gets a different completion.
  `obs_σ(s) ≠ obs_full(s)` by chance even though the recorder logged
  fully and the classifier was right *given a deterministic
  re-executor*. Step 36 reframes the theorem as a statistical
  equivalence under a noise model.
* **A3 violation (synthetic hash collision):** With probability
  `2^{-256}` a recorded `inputs_hash` collides with the hash of a
  different JSON value, the classifier wrongly accepts cache reuse.
  This is the cryptographic risk every content-addressed cache
  inherits and is treated as negligible.

## 8. Relationship to the implementation

Each statement of Theorem 1 corresponds to a specific code line:

| Proof statement | Implementation site |
|---|---|
| Topological walk of §1 | `Trace.run_replay`, [`stepback/replay.py`](../stepback/replay.py) lines ~280–430. |
| Rebinding of `"context"` (Lemma 1) | `replay.py` line 318. |
| Rebinding of `"branch_tail_hashes"` (Lemma 1) | `replay.py` lines 326–329. |
| `current_inputs_hash == recorded_inputs_hash` (clean clause (d)) | `replay.py` lines 341, 365. |
| Classifier P3 (substituted) | `compute_dirty_set` in [`stepback/divergence.py`](../stepback/divergence.py). |
| Classifier P4 (`ndh_consistent`) | `divergence.py`, `nondeterminism_hash` recheck. |
| Clean-step cache reuse | `replay.py`, the branch where `dirty == False`. |
| `verify_trace` enforcing R1, R3, R4 | [`stepback/trace_reader.py`](../stepback/trace_reader.py). |

The mapping is *line-level* so that any future refactor is forced to
preserve or explicitly bump the dirty-set version
([`docs/dirty-set.md`](./dirty-set.md) §8).

## 9. From paper proof to mechanization (Step 56)

The intended mechanization (Step 56) is a Lean 4 development with
the following correspondences:

* JSON values: a `inductive` datatype `Json` with constructors `null`,
  `bool`, `num`, `str`, `arr`, `obj`. Object keys are `String`.
* Canonical encoder `J`: a `def canonical : Json → ByteArray`
  emitting RFC 8785-style sorted-key UTF-8. Injectivity is a theorem
  about `Json` (proved structurally), not a cryptographic axiom.
* Hash function `H`: an *opaque* `axiom h_inj : Function.Injective H`.
  This is the only cryptographic axiom; it isolates the place where
  the mechanized proof depends on BLAKE2b-256 collision-resistance.
* Trace `T`: a structure with `steps : List Step` and a
  `wellFormed : Prop` predicate capturing R1–R4.
* Substitution `σ`: a `List Subst` with a partition function
  `inputMutating` / `outputForcing`.
* `inputs`, `current_hash`, `classify`, `replay`: Lean functions
  matching §1 of [`docs/dirty-set.md`](./dirty-set.md) one-to-one.
* `out_full`, `out_σ`: defined by structural recursion on the
  topologically sorted step list.
* Lemma 1 and Theorem 1: proved by strong induction on the position
  in the sorted step list. Each step of §5 above is a Lean
  `case`-split.

The point of writing this paper proof first is that the mechanization
becomes a transcription, not a discovery. The risky / cryptographic
content lives in two named axioms (`h_inj`, `executor_pure`); the
combinatorial content is fully discharged here.

## 10. Cross-references

* [`docs/dirty-set.md`](./dirty-set.md) — prose definitions and
  high-level theorem statement.
* [`stepback/divergence.py`](../stepback/divergence.py) lines 471–590
  — contract block (P1–P5, A1–A3) and proof sketch.
* [`stepback/replay.py`](../stepback/replay.py) — `Trace.run_replay`
  reference implementation.
* [`docs/canonicalization.md`](./canonicalization.md) — `J` and `H`.
* [`spec/sbtrace-v1.md`](../spec/sbtrace-v1.md) — wire format and the
  recorded fields the proof inspects.
* [`100_STEPS.md`](../100_STEPS.md) Steps 53–67 — the full algorithm
  workstream this proof is part of. Step 55 is discharged by this
  document; Step 56 (mechanization) is discharged by
  [`proofs/lean/Stepback/Soundness.lean`](../proofs/lean/Stepback/Soundness.lean);
  Step 57 (completeness) is discharged by
  [`docs/dirty-set-completeness.md`](./dirty-set-completeness.md).
