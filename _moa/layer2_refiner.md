# Layer 2 Refiner — pick a backbone, integrate the rest

## Decision

**Backbone: Proposer 1's Strategy abstraction** (algorithmic breadth
with a clean ABC).
- Why: it's the only proposer whose change is locally contained in
  `minimize.py` without touching `replay.py` or invading the CLI
  with unsafe `eval` features. The Strategy hook is the natural
  extension point that the other two proposers' ideas slot into.

## Integrated from Proposer 2 (attribution)

Add `ShapleyAttributionStrategy` as a *fourth* concrete strategy
(P1 had brute/linear/binary/ddmin). The Shapley path naturally
extends `Strategy.run` to also populate `weights: dict`. So:
- Strategies return a richer `StrategyResult` carrying the optional
  `weights` field; ddmin/linear/binary leave it `None`, Shapley
  fills it.
- Keep the `attribute_substitutions(...)` convenience function from P2
  but implement it as `minimize_substitutions(strategy=ShapleyStrategy())`.
- Skip P2's "negative inhibitor weight" first cut — it's a v2 feature
  and complicates the binary value function. Shapley returns
  non-negative weights only in this round; revisit when we have a
  real fixture exhibiting inhibition.
- Keep P2's exact-vs-sampled split inside the Shapley strategy:
  `mode="exact"` for `n <= 6`, permutation sampling above.

## Integrated from Proposer 3 (operational)

- **Adopt** the `MinimizeOptions` dataclass with `probe_budget`,
  `time_budget_s`, `progress` callback, and the `BudgetExhausted`
  exception. These are pure quality-of-life and small.
- **Adopt** `find_all_minimal` — it's just a loop on top of
  `minimize_substitutions` and gives genuinely new value (alternative
  root causes).
- **Defer** the predicate DSL. It's a parser-shaped sub-feature with
  its own surface area; squeezing it into this round bloats scope and
  risks the test budget. We DO add a minimal `predicates` namespace
  with three composable helpers (`all_of`, `any_of`, `not_`) and
  document the DSL as future work. CLI still accepts `--predicate
  '<python expr>'` as today (already in tree).
- **Keep** `progress` callback — costs ~5 LOC, big debugging win.

## Skipped from all proposers (with reasons)

- **P1 brute force on n>8** — implement, but cap at `max_n=8` as
  Proposer 1 said. Bigger would tempt foot-guns.
- **P2 inhibitor detection** — see above.
- **P3 predicate DSL** — see above. Three helper combinators only.
- **P3 multi-witness "block by forbidden=True"** — the cleanest
  blocking is to pass `excluded: set[Substitution]` to
  `minimize_substitutions` and have the strategy skip those items
  rather than mark them on the substitution objects themselves
  (substitutions are immutable dataclasses).

## Final API surface (what Layer 3 will pin)

```python
# minimize.py public symbols
class Strategy(ABC): ...
class DDMinStrategy(Strategy): ...
class LinearShrinkStrategy(Strategy): ...
class BinaryHalvingStrategy(Strategy): ...
class BruteForceStrategy(Strategy): ...
class ShapleyAttributionStrategy(Strategy): ...

@dataclass
class MinimizeOptions:
    strategy: Strategy = DDMinStrategy()
    probe_budget: int | None = None
    time_budget_s: float | None = None
    progress: Callable[[int, int], None] | None = None
    excluded: frozenset[int] = frozenset()  # item-id

@dataclass
class MinimizationResult:
    minimal: list[Substitution]
    removed: list[Substitution]
    probes: int
    cache_hits: int
    strategy_name: str
    weights: dict[int, float] | None    # populated by Shapley
    final_result: ReplayResult | None

class BudgetExhausted(Exception):
    partial: MinimizationResult

def minimize_substitutions(trace, subs, predicate, *, options=None, executor=None) -> MinimizationResult: ...
def find_all_minimal(trace, subs, predicate, *, max_witnesses=8, options=None, executor=None) -> list[MinimizationResult]: ...
def attribute_substitutions(trace, subs, predicate, *, executor=None, permutations=None) -> MinimizationResult: ...
def ddmin_substitutions(trace, subs, predicate, *, executor=None) -> MinimizationResult: ...   # back-compat shim

# minimize/predicates.py mini-namespace (or sub-attr)
def all_of(*preds): ...
def any_of(*preds): ...
def not_(p): ...
```

## CLI surface

- Existing `stepback minimize TRACE --substitute … --predicate …` keeps
  working unchanged.
- New flags:
  - `--strategy {ddmin,linear,binary,brute,shapley}` (default ddmin)
  - `--probe-budget N`
  - `--all-witnesses` + `--max-witnesses K`
- JSON payload extended with `strategy`, `cache_hits`, `weights`
  (when present), `witnesses` (when `--all-witnesses`).

## Citations
- Proposer 1: Strategy ABC, four base strategies, oracle memoisation,
  brute-force ground-truth oracle, CLI `--strategy`. (Proposer 1 §1–5.)
- Proposer 2: ShapleyAttributionStrategy + `attribute_substitutions`
  convenience, exact vs. sampled mode, weights field.
  (Proposer 2 §1–2 and §"Why this framing".)
- Proposer 3: `MinimizeOptions`, `BudgetExhausted`, progress callback,
  `find_all_minimal`, `excluded` set, three predicate combinators
  only. (Proposer 3 §1–3.)

## Risk register
- Shapley sampling variance — mitigate by deterministic seed.
- Strategy refactor risks back-compat — keep `ddmin_substitutions`
  signature byte-identical and forward to new path.
- Multi-witness loop with `excluded` — must include the excluded set
  in the cache key.
