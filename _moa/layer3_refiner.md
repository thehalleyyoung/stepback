# Layer 3 Refiner — final design + concrete file plan

## What's locked in from Layer 2

The L2 refiner picked **Proposer 2's CostBreakdown/TokenRates**
backbone, layered **P1**'s catalog/aliases/deprecation/strict mode,
and added **P3**'s aggregation/format/diff/budget helpers. Sign of
`diff_costs` pinned to `b - a`. Defensive clamps on cached and
reasoning subtraction. `set_strict` returns a context manager.

L3 keeps all of that and deepens along three axes the prior
layers under-specified.

## L3 deepenings (~30% more depth, per advisory)

### A. `to_dict` / JSON-serialisable dataclasses

`CostBreakdown`, `CostSummary`, `BudgetCheck`, and `TokenRates`
all get a `to_dict()` method returning a flat str→primitive
mapping. Reason: `recorder.py:120` writes step records that are
serialised by `trace_writer.py`. If a follow-up round wants to
persist breakdown alongside the float `cost_usd`, it needs a
JSON-safe dict today. P2 mentioned `to_dict` for `CostBreakdown`
only — extend to all four.

### B. Snapshot freshness warning

`SNAPSHOT_DATE = "2026-04-01"`. Add `def
snapshot_age_days(today=None) -> int` and a one-shot warning the
first time `compute_cost` runs in a process where the snapshot is
older than 180 days. Tracker: `_stale_warned: bool`. P1 mentioned
the snapshot but never used it. This makes the staleness *visible*
to the user instead of silently rotting.

### C. Step-record schema convergence

`aggregate_costs` and `diff_costs` accept iterables of dict
records. Pin the keys we read so future rounds can rely on them:

```
{
    "step_id":   str,
    "step_kind": str,    # "llm_call" | "tool_call" | "router" | ...
    "model":     str,    # may be empty for non-llm
    "usage":     dict,   # may be empty
    "cost_usd":  float,  # optional; recomputed if missing
}
```

Anything not matching is silently treated as a no-cost step (so
report.py can pass *every* step through aggregate_costs without
filtering first). Document this contract at the top of the module.

### D. `cli` ergonomics

Add a `__main__` shim: `python -m stepback.pricing` prints the
catalog as a Markdown table to stdout. Useful for users sanity-
checking what their installed snapshot covers without grepping
the source. Adds 15 lines, zero risk.

## Final file plan

### `stepback/pricing.py` — full rewrite

Sections, in order:

1. Module docstring (snapshot date, contract for step records,
   pointer to `_moa/layer3_refiner.md`).
2. `SNAPSHOT_DATE`, `CURRENCY` constants.
3. `MissingPriceError` exception.
4. `TokenRates`, `CostBreakdown`, `CostSummary`, `BudgetCheck`
   dataclasses with `to_dict`.
5. Inline `RATE_TABLE` literal (the catalog from L2).
6. `ALIASES` literal.
7. Derived `PRICE_LIST = {m: (r.input_per_1k, r.output_per_1k)
   for m, r in RATE_TABLE.items()}`.
8. Internal helpers: `_resolve`, `_warn_deprecated_once`,
   `_warn_stale_once`, `_extract_buckets`.
9. Public functions: `resolve_model`, `is_known`, `set_strict`,
   `compute_cost_breakdown`, `compute_cost`, `format_usd`,
   `aggregate_costs`, `check_budget`, `diff_costs`,
   `snapshot_age_days`.
10. `if __name__ == "__main__":` markdown table dump.

### `tests/test_pricing.py` — new

All 20 tests from L2 plus three new for the L3 deepenings:

21. `test_breakdown_to_dict_round_trips_via_json` — `json.dumps(
    breakdown.to_dict())` parses back, all values are floats/strs.
22. `test_snapshot_age_days_uses_today_param` — passing a future
    date returns a positive integer matching the day delta.
23. `test_aggregate_costs_skips_records_without_step_kind`
    — `[{}, {"cost_usd": 1.0}]` → total_usd is 1.0 since the
    second has no step_kind we recognise but cost_usd is read.
    (Refines C.)

### Deprecated models

P1 specified `gpt-4-0613`, `gpt-3.5-turbo-0613`, `claude-2.1`.
Mark them with `replaced_by` and verify the warning fires exactly
once.

## Verification plan

* Run `pytest -x -q` — all 82 existing tests must still pass.
  Risk surface:
  - `test_shims.py:169` reads `pricing.PRICE_LIST` — still exposed
    as a `dict[str, tuple[float, float]]`.
  - `recorder.py:120` and `replay.py:384` import `compute_cost` and
    expect a `float` — preserved.
* Run new `test_pricing.py` — all 23 tests must pass.
* Run `python -m stepback.pricing` — must print a non-empty
  Markdown table without crashing.

## Citation map (which idea from where)

| Element                              | Source     |
|--------------------------------------|------------|
| `TokenRates`, `CostBreakdown`        | Proposer 2 |
| Multi-tier algorithm (cached/reason) | Proposer 2 |
| Provider-shape normalisation         | Proposer 2 |
| Catalog expansion (~17 models)       | Proposer 1 |
| Aliases + `resolve_model` + `is_known`| Proposer 1 |
| Deprecation warnings                 | Proposer 1 |
| `MissingPriceError` + `set_strict`   | Proposer 1 |
| `SNAPSHOT_DATE`                      | Proposer 1 |
| `format_usd`, `aggregate_costs`      | Proposer 3 |
| `CostSummary`, `BudgetCheck`         | Proposer 3 |
| `check_budget`, `diff_costs`         | Proposer 3 |
| Defensive clamps, `with set_strict`  | Layer 2    |
| `b - a` sign convention              | Layer 2    |
| `to_dict` on every dataclass         | Layer 3    |
| Snapshot freshness warning           | Layer 3    |
| Step-record schema contract          | Layer 3    |
| `__main__` markdown dump             | Layer 3    |

## Non-goals (explicit, this round)

* Touching `recorder.py`, `replay.py`, `report.py`, or `cli.py`.
  They keep importing `compute_cost` and reading `PRICE_LIST`
  exactly as today — pure additive change.
* Adding a JSON data file. L1 wanted this; L2 rejected it; L3
  agrees — keep the catalog inline as a Python literal for
  diff-friendliness.
* Currency conversion (FX). Out of scope; documented as future.
