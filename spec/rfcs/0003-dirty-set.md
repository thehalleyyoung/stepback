# RFC 0003 — Dirty-Set Semantics

| Field | Value |
|---|---|
| RFC number | 0003 |
| Title | Dirty-Set Replay Semantics |
| Status | Draft |
| Supersedes | — |
| Created | 2026-05-12 |
| Authors | stepback maintainers |
| Reference impl | `stepback.divergence.compute_dirty_set`, `stepback.replay.Trace.run_replay` |

---

## Abstract

This RFC specifies the *dirty-set algorithm* — the core of stepback's
counterfactual replay capability.  Given a recorded SB-Trace `T` and a
*substitution* `σ` that changes one or more step inputs, the dirty-set
`D(T, σ)` is the minimal set of steps that must be re-executed to produce
a sound replay.  Steps not in `D(T, σ)` are *clean*: their recorded
outputs are reused verbatim.

The authoritative prose proof is in
[`docs/dirty-set.md`](../../docs/dirty-set.md) and
[`docs/dirty-set-soundness.md`](../../docs/dirty-set-soundness.md).
The Lean 4 mechanisation is in [`proofs/lean/`](../../proofs/lean/).
This RFC summarises the algorithm contract in a form suitable for
external review and multi-language implementation.

---

## 1. Motivation

Counterfactual questions — "what would the agent have said if I used a
different system prompt?" or "would adding this guard have caught the
failure?" — require re-executing the agent.  Re-executing all `N` steps
is expensive.  The dirty-set algorithm proves that only the `|D|` steps
downstream of the change need to run; clean steps produce the same output
under the new inputs, so their recorded outputs are valid cache entries.

The algorithm is *sound* (no clean step should have been dirty) and
*minimal-by-construction* (no dirty step can be removed without violating
soundness under the stated assumptions).

---

## 2. Formal objects

### 2.1 Trace DAG

Let `T` be a parsed SB-Trace.  `T` defines a finite directed acyclic graph
`G(T) = (V(T), E(T))` where:

- `V(T)` is the set of `step` frames in trace order.
- `E(T)` is the recorded parent-edge relation: `(p, s) ∈ E(T)` iff
  `p = s.parent_step_id` or `p ∈ s.parent_step_ids`.

`G(T)` is required to be acyclic and topologically ordered.

### 2.2 Canonical input function

For a step `s ∈ V(T)`:

```
canonical_inputs(s) = canonical_hash(s.inputs)   # RFC 0002
```

The canonical input hash is stored in `s.inputs_hash` by the recorder at
record time.

### 2.3 Substitution

A substitution `σ` is a finite map from `step_id → new_inputs`.  It is
produced by the caller (e.g. `Trace.run_replay(substitutions=[...])`) and
represents the counterfactual change being explored.

---

## 3. Dirty-set algorithm (v1)

```
D(T, σ) = smallest S ⊆ V(T) such that:

  (D1) ∀ s ∈ dom(σ): s ∈ S
       (Every directly substituted step is dirty.)

  (D2) ∀ s ∈ V(T): canonical_hash(σ(s).inputs) ≠ s.inputs_hash ⟹ s ∈ S
       (A step is dirty if its recomputed inputs hash differs from the
        recorded hash — i.e. a clean ancestor produced different outputs
        and those outputs flow into this step.)

  (D3) ∀ (p, s) ∈ E(T): p ∈ S ⟹ s ∈ S
       (Transitive closure over the recorded DAG: a step is dirty if any
        of its parents is dirty.)

  (D4) ∀ s ∈ V(T): s.nondeterminism_hash ≠ canonical_hash(s.nondeterminism) ⟹ s ∈ S
       (A step whose recorded nondeterminism_hash is internally
        inconsistent is dirty regardless of σ.)
```

Steps in `V(T) ∖ D(T, σ)` are *clean*.  Their recorded outputs are reused
verbatim by the replay engine.

### 3.1 `parallel_branch_join` extension

For `parallel_branch_join` steps, `inputs` includes a `branch_tail_hashes`
map `{ branch_id → hash }`.  Rule (D2) applies normally; the hash of the
full `inputs` dict — including `branch_tail_hashes` — is compared against
`s.inputs_hash`.

### 3.2 `RaiseSubstitution` extension

When `σ` includes a substitution of kind `RaiseSubstitution`, the
exception step is marked dirty (D1) and all steps with `parent_step_id`
pointing to the exception step are also dirty (D3).  Steps in branches
that never observed the exception are not affected.

---

## 4. Soundness theorem (P1)

**Theorem.** Under assumptions:

- **A1** (conservative dependency model): Every dependency between steps
  is captured in `E(T)`.
- **A2** (executor purity): A step's outputs are a deterministic function
  of its recorded inputs (for steps marked `nondeterminism_class = "none"`).
- **A3** (collision-freeness): BLAKE2b-256 is collision-resistant on the
  input distribution.

For every clean step `s ∉ D(T, σ)`, the replay output `s.outputs` equals
the output that full re-execution under `σ` would produce.

*Proof sketch:* By induction on topological order.  The base case (steps
with no parents, or no parent in `D`) holds by A2+A3.  The inductive step
follows from D3 (parents are clean ⟹ inputs unchanged ⟹ outputs unchanged
by A2).  The full proof is in `docs/dirty-set-soundness.md`; the
mechanised version is in `proofs/lean/Stepback/Soundness.lean`.

---

## 5. Complexity

| Trace topology | `|D|` worst case | Work |
|---|---|---|
| Linear chain of `N` steps | `N` | `O(N)` |
| DAG with `N` vertices, `E` edges | `N` | `O(N + E)` |
| `K` independent parallel branches, each length `L` | `L` (one branch dirty) | `O(L + K)` |

Full complexity analysis is in `docs/dirty-set-complexity.md`.

---

## 6. Version pinning

`dirty_set_version = "1"` is bound to `canonicalisation_version = "1"`
(RFC 0002).  Any change to the canonical encoder or to the declared
dependency model is a breaking change and MUST increment both versions.

---

## 7. Relationship to other RFCs

- RFC 0002 defines `canonical_hash`, used in rules D2 and D4.
- RFC 0001 defines the `step` frame fields referenced by this algorithm.

---

## Appendix A — Changelog

| Date | Author | Change |
|---|---|---|
| 2026-05-12 | stepback maintainers | Initial draft |
