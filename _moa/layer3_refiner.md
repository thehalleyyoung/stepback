# Layer 3 Refiner — final pinned design + concrete file plan

## What's locked in from Layer 2

L2 picked **Proposer 1's Strategy ABC** as the backbone, integrated
**Proposer 2's Shapley attribution** as a fifth strategy, and adopted
**Proposer 3's `MinimizeOptions` + `find_all_minimal` + budget
controls + 3-helper predicates namespace**. Inhibitor detection,
predicate DSL, and brute-force-on-large-n are deferred.

This refiner pins the implementation details and tightens scope to
fit one round + tests.

## Concrete file changes

### 1. `stepback/minimize.py` — extended (~440 LOC up from 168)

New public surface (matches L2 §"Final API surface"):

- `Strategy(ABC)` with `name: ClassVar[str]` and
  `run(items, oracle) -> tuple[list[Sub], list[Sub], dict|None]`.
  The third tuple element is `weights or None`.
- `DDMinStrategy` — verbatim port of the existing ddmin loop, lifted
  out of `ddmin_substitutions` into the strategy class.
- `LinearShrinkStrategy` — drop one at a time, accept if oracle still
  fires.
- `BinaryHalvingStrategy` — recursive halving with both-halves
  fallback (L1 §2 binary).
- `BruteForceStrategy(max_n: int = 8)` — enumerate non-empty subsets
  in increasing size order; raise `ValueError` if `len(items) > max_n`.
- `ShapleyAttributionStrategy(permutations: int|None = None,
  rng_seed: int = 0xC0DE)` — exact (`n<=6`) or sampled.
- `MinimizeOptions` dataclass — strategy, probe_budget, time_budget_s,
  progress, excluded.
- `MinimizationResult` extended with `strategy_name`, `cache_hits`,
  `weights: dict[int,float] | None` keyed by `id(item)` *but also
  exposed as `weight_for(item) -> float`* — id-keyed dict is brittle
  across processes; provide a method.
- `BudgetExhausted(Exception)` carrying `.partial: MinimizationResult`.
- `_OracleCache` — wraps the predicate-on-replay closure;
  canonicalises a subset by `tuple(sorted(map(id, items)))` *plus*
  the frozen `excluded` set; counts hits and probes; honours
  `probe_budget` (raises `BudgetExhausted` mid-flight) and
  `time_budget_s` (checks wall clock before each new probe).
- `minimize_substitutions(trace, subs, predicate, *, options=None,
  executor=None) -> MinimizationResult` — orchestrates: trigger-on-full
  check, empty-already-triggers check, delegates to
  `options.strategy.run`, then runs a final replay.
- `find_all_minimal(trace, subs, predicate, *, max_witnesses=8,
  options=None, executor=None) -> list[MinimizationResult]` — loop
  using `excluded` to block prior witnesses; returns up to k
  disjoint-on-at-least-one-element witnesses.
- `attribute_substitutions(...)` — convenience wrapping
  `ShapleyAttributionStrategy`; returns `MinimizationResult` with
  populated weights.
- `ddmin_substitutions(trace, subs, predicate, *, executor=None)` —
  back-compat shim. **Signature, return type, exception type
  unchanged. All four existing tests must still pass.**
- Module-level `__all__` updated.

### 2. `stepback/predicates.py` — NEW (~70 LOC)

Tiny module with `all_of`, `any_of`, `not_` combinators only. Each
takes 1+ `Callable[[ReplayResult], bool]` and returns one. Doctests
plus three unit tests. Re-exported from `stepback/__init__.py`.

### 3. `stepback/replay.py` — NO CHANGES

`Trace.minimize` already forwards to `ddmin_substitutions`. Add a
**new** `Trace.minimize_with(strategy, ...)` thin method that
forwards to `minimize_substitutions` so users can pick a strategy
without leaving the Trace API. **Wait** — L2 forbids replay.py
changes. Move that method onto `Trace` via a tiny helper exported
from minimize.py and have `Trace.minimize_with` be added in a
`Trace.minimize_with = ...` assignment at the bottom of
`minimize.py`. (Same monkey-patch pattern as `Trace.minimize`,
which is already done that way — confirmed in `__init__.py`.)

  Actually re-checking: `Trace.minimize` is currently *not* set in
  `replay.py`. Let me verify in the implementation phase. If it IS
  set in `replay.py`, the monkey-patch goes in `__init__.py`
  alongside the existing one. If `Trace.minimize` is currently a
  method on `Trace`, then it's a real `replay.py` change and we'll
  add `minimize_with` as an alias function instead — no monkey-patch
  needed.

### 4. `stepback/cli.py` — extended (~80 new LOC)

In the `minimize` subcommand:
- Add `--strategy {ddmin,linear,binary,brute,shapley}` (default ddmin).
- Add `--probe-budget INT`.
- Add `--all-witnesses` + `--max-witnesses K` (default 4).
- JSON payload gains `strategy`, `cache_hits`, optional `weights`
  (when shapley), optional `witnesses` (when `--all-witnesses`).
- Existing flags / exit codes unchanged.

### 5. `stepback/__init__.py` — re-exports

Add: `Strategy`, `DDMinStrategy`, `LinearShrinkStrategy`,
`BinaryHalvingStrategy`, `BruteForceStrategy`,
`ShapleyAttributionStrategy`, `MinimizeOptions`, `BudgetExhausted`,
`find_all_minimal`, `attribute_substitutions`,
`minimize_substitutions`, plus `predicates` module.

### 6. `tests/test_minimize.py` — extended (~250 new LOC)

Existing 5 tests stay green. New tests:
- `test_strategy_pluggable_returns_same_minimal[ddmin|linear|binary|brute|shapley]` — parametrised; all 5 strategies on the
  6-noisy fixture must return the same 1-element minimal.
- `test_brute_force_finds_global_optimum` — pin a small fixture
  where ddmin happens to find a 1-element minimal; brute force on
  same input also returns size-1 minimal. (Joint-cause fixture is
  hard to construct with current substitutions — defer to a future
  round.)
- `test_oracle_cache_hits_recorded` — same minimisation across
  ddmin and binary shows `cache_hits >= 1`.
- `test_minimize_options_probe_budget_raises` — set `probe_budget=2`
  on the standard fixture; assert `BudgetExhausted`, partial.probes
  <= 2.
- `test_minimize_options_progress_callback` — collects events;
  asserts `len(events) == result.probes`.
- `test_find_all_minimal_returns_at_least_one` — on the standard
  6-noisy fixture, returns >= 1 witness; first witness equals what
  ddmin would return.
- `test_shapley_assigns_full_weight_to_lone_cause` — exact mode (n=4),
  the one cause has weight ≈ 1.0 (within 1e-9), decoys ≈ 0.0.
- `test_shapley_sampled_within_tolerance_of_exact` — n=5; both modes;
  tolerance 0.2.
- `test_attribute_substitutions_convenience_wrapper` — returns a
  `MinimizationResult` with non-None `weights`.

### 7. `tests/test_predicates.py` — NEW (~50 LOC)

- `test_all_of_short_circuits_false`
- `test_any_of_short_circuits_true`
- `test_not_inverts`
- `test_combinators_compose` — `all_of(p1, any_of(p2, not_(p3)))`.

### 8. `tests/test_cli_minimize_strategy.py` — NEW (~100 LOC)

- `test_cli_minimize_strategy_brute` — `--strategy brute` produces
  same JSON shape, with `strategy="brute"` field.
- `test_cli_minimize_strategy_shapley_emits_weights` — JSON payload
  has `weights` array; cause weight ≈ 1.0.
- `test_cli_minimize_all_witnesses_emits_list` — payload has
  `witnesses: [...]` of length >= 1.

## Implementation order
1. Write `predicates.py` + its tests.
2. Refactor `minimize.py`: introduce `Strategy`, port ddmin into
   `DDMinStrategy`, keep `ddmin_substitutions` as shim. Run existing
   5 minimise tests — they must stay green.
3. Add `LinearShrink`, `BinaryHalving`, `BruteForce`,
   `ShapleyAttribution`. Add `_OracleCache` + budget enforcement.
   Add `find_all_minimal` + `attribute_substitutions`.
4. CLI plumbing.
5. New tests.
6. Full pytest -x -q.

## Citations (per-section)
- §1 Strategy ABC + 4 base strategies + oracle memoisation + brute
  cap: **Proposer 1** §1–3, §"Risk".
- §1 ShapleyAttributionStrategy + exact/sampled split: **Proposer 2**
  §1, §"Risk".
- §1 MinimizeOptions, BudgetExhausted, progress callback,
  `find_all_minimal`, `excluded`: **Proposer 3** §2, §1.
- §2 predicates module (3 helpers only, no DSL): **Proposer 3** §3
  (trimmed per L2).
- §4 CLI flags: **Proposer 1** §"CLI" + **Proposer 3** §"CLI".
- §"Skipped" inhibitor weights, predicate DSL, large-n brute:
  **Layer 2** decisions.

## Acceptance gate
- All 359 existing tests stay green.
- New strategy/predicates/CLI tests pass.
- `ddmin_substitutions(...)` signature & exception unchanged.
- `pytest -x -q` exits 0.
