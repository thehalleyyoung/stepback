# Proposer 1 — Algorithmic breadth: a strategy-pluggable minimizer

## Target module
`stepback/minimize.py` (currently 168 LOC: one algorithm — Zeller-Hildebrandt
ddmin — and one result dataclass).

## Theme
The module has exactly one algorithm. The literature on delta-debugging /
causal isolation has many useful flavours, each with a different
probe-budget vs. minimality trade-off. Expose a **strategy** abstraction
and ship four implementations behind one entry point.

## What to add

1. `Strategy` ABC with one method:
   `run(items, oracle) -> (minimal_items, removed_items)` where
   `oracle(subset) -> bool` is the cached predicate-on-replay.
2. Concrete strategies:
   - `DDMinStrategy` — wrap the existing Zeller code path (default).
   - `LinearShrinkStrategy` — drop one element at a time, accept if
     predicate still fires. Simple, n+1 probes worst case, gives a
     1-minimal answer when items are independent.
   - `BinaryHalvingStrategy` — split in half, recurse into the half(s)
     that still trigger; if neither half alone triggers, keep both
     halves and shrink each. Often beats ddmin when blame is
     concentrated.
   - `BruteForceStrategy(max_n=8)` — enumerate every non-empty subset
     up to size `max_n`, return the smallest one that triggers.
     Optimal but exponential; useful as a ground-truth oracle in
     tests and for tiny inputs.
3. Memoise oracle calls by hashing `frozenset(id(item))` — every
   strategy benefits, ddmin re-tests overlapping subsets often.
4. `ddmin_substitutions(...)` keeps its current signature for back-compat
   and forwards to `minimize_substitutions(..., strategy=DDMinStrategy())`.
5. New `MinimizationResult` fields: `strategy_name: str`,
   `cache_hits: int`.

## CLI
Add `--strategy {ddmin,linear,binary,brute}` to `stepback minimize`
(default `ddmin`). Echo the chosen strategy into the JSON payload.

## Tests
- `test_strategy_pluggable_returns_same_minimal`: each of the four
  strategies returns a 1-element minimal on the existing payments
  fixture (the lookup substitution is the only cause).
- `test_brute_force_finds_global_optimum_2_of_5`: craft a fixture
  where two substitutions are *jointly* required (neither alone
  triggers). DDMin/linear find a 2-set; brute force confirms it is
  globally minimal at 2.
- `test_oracle_cache_hits_recorded`: assert `cache_hits >= 1` when a
  strategy revisits the same subset.

## Why this framing
The cleanest extension: keep ddmin as default, add diversity via
strategy injection. No changes to `replay.py` or `Trace.minimize`'s
signature. Algorithmic diversity is the most useful thing for a user
who hits a pathological input where ddmin oscillates.

## Risk
Brute force is exponential — must guard with `max_n`. If users pass
50 substitutions with `--strategy brute`, refuse with a clear error.

## LOC estimate
~250 LOC of new code in `minimize.py`, ~150 LOC of new tests, ~30
LOC CLI.
