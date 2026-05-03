# Proposer 3 — ergonomics: aggregation, formatting, budgets

## Framing

Two siblings call `compute_cost` (`recorder.py:120`,
`replay.py:384`) and write the result back as a float field
`cost_usd` on each step record. Then `report.py` (720 lines!)
re-aggregates those floats: per-step, per-branch, per-model,
delta between branches, percent-of-budget. None of that
aggregation lives in `pricing.py`. The result is:

* `report.py` and `cli.py` independently call `sum(s["cost_usd"]
  for s in steps)` and independently format dollars, leading to
  inconsistent rounding (some places 4 decimals, some places 6).
* No notion of a *budget* — you can't ask `did this branch blow
  past 25¢?` without bespoke code.
* No formatting helpers — every site that prints a cost uses a
  different `f"${x:.4f}"` template.

This proposer keeps `pricing.py` numerically simple but adds the
**operations** report.py and cli.py have been doing by hand.

## Public surface (additive)

```python
PRICE_LIST: dict[str, tuple[float, float]]      # unchanged
def compute_cost(model, usage) -> float: ...    # unchanged

# NEW

def format_usd(amount: float, *, precision: int = 4,
               unit: str = "$") -> str:
    """Stable currency formatting used everywhere."""

def aggregate_costs(steps: Iterable[dict]) -> CostSummary:
    """Walk an iterable of step records and return a structured
    summary: total_usd, per_model, per_step_kind, n_steps."""

@dataclass(frozen=True)
class CostSummary:
    total_usd:       float
    n_steps:         int
    per_model:       dict[str, float]   # model -> usd
    per_step_kind:   dict[str, float]   # llm_call/tool_call/router -> usd
    most_expensive:  list[tuple[str, float]]  # top-5 (step_id, usd)

@dataclass(frozen=True)
class BudgetCheck:
    budget_usd:      float
    spent_usd:       float
    remaining_usd:   float
    over_budget:     bool
    fraction_used:   float

def check_budget(steps: Iterable[dict],
                 budget_usd: float) -> BudgetCheck: ...

def diff_costs(steps_a: Iterable[dict],
               steps_b: Iterable[dict]) -> dict[str, float]:
    """Per-model delta b - a; key '__total__' carries the overall
    delta. Used by report.py to compare a branch against the
    baseline trace."""
```

## Why these specifically

* `format_usd` — `f"${x:.4f}"` appears 7 times in `report.py` and
  `cli.py`. Centralising kills inconsistency and lets us swap to
  `${x:.2¢}` later if we want.
* `aggregate_costs` — the report's "summary table" is exactly this
  structure. Today it's recomputed inline.
* `BudgetCheck` — for branch experiments ("what if we swapped the
  big router LLM for a small one — does the trace still fit in
  $0.10?"), the user wants a single boolean answer.
* `diff_costs` — the README brags about "cost deltas across
  branches"; this is the function that makes that one line of
  code instead of twenty.

## Tests (`tests/test_pricing.py`)

1. `test_format_usd_default_precision` — `format_usd(0.001234) ==
   "$0.0012"`.
2. `test_format_usd_negative` — handles negative deltas:
   `"-$0.0012"`.
3. `test_aggregate_costs_per_model` — three steps, two models, the
   sum across `per_model.values()` equals `total_usd`.
4. `test_aggregate_costs_per_step_kind` — llm_call vs tool_call
   buckets.
5. `test_aggregate_costs_most_expensive_top5` — returns at most 5
   entries, sorted descending.
6. `test_check_budget_under` — `over_budget=False`,
   `fraction_used < 1`.
7. `test_check_budget_over` — `over_budget=True`,
   `remaining_usd < 0`.
8. `test_diff_costs_total_and_per_model` — branch vs baseline.
9. `test_diff_costs_handles_disjoint_models` — model only in branch.

## Refactor opportunities (light, optional)

* `report.py` and `cli.py` can be patched to call the new helpers
  in a follow-up round. This proposer does NOT touch those files
  — purely additive in `pricing.py` so backward compat is bulletproof.

## What this proposer is NOT trying to fix

* Catalog completeness or aliasing (Proposer 1's domain).
* Tier accuracy — cached / reasoning / image tokens (Proposer 2's
  domain).

The bet here: **shape of the API matters more than the catalog
size**. A two-row catalog with great aggregation/diff utilities is
more useful for the debugger UX than a fifty-row catalog with
nothing to slice it by.
