# Proposer 2 — Causal *attribution*: not just "which", but "how much"

## Target module
`stepback/minimize.py`.

## Theme
Today the user gets a binary answer per substitution: in the minimal
set or not. That's necessary for a 1-minimal witness but throws away
information. A debugger user actually wants to see the **causal weight**
of every substitution — Shapley-style attribution over the substitution
set, with the predicate as the value function. Then the 1-minimal answer
falls out as "the items with non-zero Shapley value".

## What to add

1. `attribute_substitutions(trace, subs, predicate, *, executor=None,
   permutations: int | None = None) -> AttributionResult`.
   - Value function `v(S) = 1.0 if predicate(replay(S)) else 0.0`.
   - Exact Shapley for `|subs| <= 6` (enumerate all 2^n subsets).
   - Permutation sampling Shapley for larger sets — `permutations`
     defaults to `min(64, 4 * n)`.
2. `AttributionResult`:
   - `weights: dict[item_id, float]`  — Shapley value per substitution.
   - `minimal: list[Substitution]` — items with weight > 0 (this *is*
     a minimal sufficient subset under the binary value function).
   - `probes: int`, `cached_probes: int`, `mode: "exact" | "sampled"`.
3. **Cause vs. counter-cause split.** Some substitutions can *suppress*
   the predicate when added to others. Report negative Shapley values
   (cause) vs positive (cause). Surface `inhibitors: list[...]` — items
   whose Shapley weight is negative.
4. Keep ddmin as a *fast* path: a thin `ddmin_substitutions` that calls
   `attribute_substitutions(..., permutations=1)` and returns the
   first 1-minimal it finds.
5. Probe-result memoisation by canonical-subset key (sorted item IDs)
   shared across attribution + ddmin.

## CLI
- `stepback attribute TRACE --substitute ... --predicate ...` →
  emits a table:
    ```
    sub-id           weight   role
    tool@step-7      +1.000   cause
    model@step-3      0.000   noise
    prompt@step-9    -0.250   inhibitor
    ```
- `--shapley-mode {exact,sampled}` and `--permutations N`.

## Tests
- `test_attribution_assigns_full_weight_to_lone_cause`: 1 cause + 5
  decoys → cause has weight ≈ 1.0, decoys ≈ 0.0.
- `test_attribution_splits_weight_for_joint_cause`: 2 substitutions
  jointly required → each gets weight ≈ 0.5.
- `test_attribution_detects_inhibitor`: design a triplet where one
  substitution actively suppresses the predicate; assert negative
  weight.
- `test_attribution_sampled_within_tolerance_of_exact`: run both modes
  on n=5; assert `max(|w_exact - w_sampled|) < 0.15`.

## Why this framing
Attribution > 1-minimal. A 1-minimal answer is a single witness;
Shapley gives you the *whole picture* of who-is-causing-what, which
is what an agent debugger user wants when staring at a bug
reproduction. Inhibitor detection is a genuinely new debugging
capability — it surfaces "this prompt edit was actually masking the
bug, not causing it".

## Risk
Shapley with binary predicates can give degenerate weights when the
predicate is non-monotone. Document the assumption that for n≥7 the
result is sampled and approximate.

## LOC estimate
~350 LOC of new logic, ~200 LOC of tests, ~60 LOC CLI.
