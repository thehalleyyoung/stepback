# Stepback — Lean 4 mechanized soundness proof

This directory discharges **Step 56** of [`100_STEPS.md`](../../100_STEPS.md):

> *Mechanize soundness in Lean or Coq for an immutable step DAG and*
> *collision-free canonical hash assumption.*

## Overview

[`Stepback/Soundness.lean`](Stepback/Soundness.lean) is the Lean 4 mechanization
of the dirty-set soundness theorem proved on paper in
[`docs/dirty-set-soundness.md`](../../docs/dirty-set-soundness.md).  The proof
shows that, under the assumptions listed below, every **clean** step (one not in
the dirty set) produces the same output under dirty-set replay as under full
re-execution.  This backs the core performance claim of stepback: cache reuse is
always safe for clean steps.

## Files

| File | Purpose |
|---|---|
| `Stepback/Soundness.lean` | Main proof library — structures, predicates, theorems |
| `lakefile.lean` | Lake build configuration |
| `lean-toolchain` | Pins the exact Lean 4 version used |

## Building

```bash
# Install elan (Lean version manager) if not already installed:
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
  | sh -s -- -y
source ~/.elan/env

# Build from the proofs/lean/ directory:
cd proofs/lean
lake build
lake exe check
```

Lean will be downloaded automatically by elan according to `lean-toolchain`.
No internet access is required once elan and the toolchain are cached.

## Trust base

The proof relies on the following trusted components:

1. **Lean 4 kernel** — the small, independently auditable type-checking kernel.
2. **Classical logic** — `Classical.em` and `propext`/`funext` are admitted
   by Lean 4's kernel; the proof uses them implicitly via `Classical.choice`.
3. **`A2` — executor purity** — encoded as a *parameter* to `soundness` (not a
   bare axiom), so it is visible in the theorem statement.  Every caller must
   supply a proof.  The Python `Executor` interface documents this obligation in
   `stepback/divergence.py`.
4. **`A3` — hash collision-freeness** — deliberately *absent* from this proof.
   Soundness does not require A3.  A3 is only needed for completeness
   (Step 57, `docs/dirty-set-completeness.md`).

There are **no `sorry`**, **no bare `axiom` declarations**, and **no `partial`
definitions** in `Stepback/Soundness.lean`.  CI enforces this with `grep`
(see `.github/workflows/lean.yml`).

## Symbol-by-symbol cross-reference to Python

| Lean symbol | Python counterpart | File |
|---|---|---|
| `Step` | `RecordedStep` | `stepback/divergence.py` |
| `Trace` | `Trace` | `stepback/replay.py` |
| `Subst` | `Substitution` base class | `stepback/substitutions.py` |
| `currentInputsHash` | `_canonical_inputs_hash` | `stepback/divergence.py` |
| `execute` | `Executor.execute` | `stepback/replay.py` |
| `DependsOnlyOnParents` | A1 assumption comment | `stepback/divergence.py` |
| `RecorderCoherent` | R1/R3 contract block | `stepback/divergence.py` |
| `RecorderOutputCoherent` | R2 contract block | `stepback/divergence.py` |
| `CleanSound` | P1 postcondition | `stepback/divergence.py` |
| `IsFullReplay` | Definition of `_full_replay` | `stepback/replay.py` |
| `IsDirtyReplay` | `Trace.run_replay` | `stepback/replay.py` |
| `classify` | `_classify_step` | `stepback/divergence.py` |
| `directlyTargeted` | `_is_directly_targeted` | `stepback/divergence.py` |
| `soundness` | Theorem 1 / P1 contract | `stepback/divergence.py` |
| `cache_reuse_safe` | Corollary 1 | `stepback/divergence.py` |

## Proof structure

```
Theorem 1 (soundness)
└── soundness_aux
    ├── IsDirtyReplay.1 hclean  →  outSigma s.id = s.outputs_hash
    └── RecorderOutputCoherent  →  outFull s.id  = s.outputs_hash
        ⟹ outSigma s.id = outFull s.id  □
```

The proof is *not* inductive in the classical sense: the hypotheses
`IsDirtyReplay` and `RecorderOutputCoherent` already encode the inductive
invariants established when building the replay output function.  The strong
well-founded induction (via `WellFoundedRelation Step`, `stepDepthWF`) would be
needed to *compute* the replay functions from scratch; here we take them as
parameters and prove the soundness consequence directly.

The depth-ordering well-founded relation (`stepDepthWF`) is included as a named
instance to document the induction principle used in the companion Python
implementation (`stepback/divergence.py` iterates steps in topological order).

## Relationship to other documents

- **Paper proof** (Step 55): [`docs/dirty-set-soundness.md`](../../docs/dirty-set-soundness.md)
- **Prose definitions** (Step 54): [`docs/dirty-set.md`](../../docs/dirty-set.md)
- **Completeness paper proof** (Step 57): [`docs/dirty-set-completeness.md`](../../docs/dirty-set-completeness.md)
- **Python reference implementation**: `stepback/divergence.py`, `stepback/replay.py`
- **Wire format**: `spec/sbtrace-v1.md`
- **100-step plan**: [`100_STEPS.md`](../../100_STEPS.md) Steps 54–58

## CI

`.github/workflows/lean.yml` runs on every change to `proofs/lean/**` and:

1. Installs elan and pulls the pinned Lean toolchain.
2. Greps for `sorry`/`axiom`/`partial` and fails if found.
3. Runs `lake build` to type-check all definitions and proofs.
4. Runs `lake exe check` for a runtime sanity check.
