# Layer 2 Refiner — pick Proposer 2's structure, integrate P1 + P3

## Choice

Adopt **Proposer 2's `CostBreakdown` + `TokenRates`** as the
backbone. Reasoning: the README's headline claim — "cost deltas
across branches are meaningful" — is *false* under the current
two-bucket pricing for any reasoning-heavy or vision-heavy step,
and false numbers can't be fixed by better helpers (P3) or a
bigger catalog (P1). Correct numerics first.

Then layer in:

* From **Proposer 1**: catalog expansion (~15 models),
  alias-resolution, deprecation warnings, `SNAPSHOT_DATE`,
  `MissingPriceError` + `set_strict()`. Reject P1's split into
  `_pricing_data.json` — added file/IO surface area for no real
  win because the catalog still has to be hand-edited per release;
  keep the data inline as a Python literal (it stays diff-friendly
  in code review).
* From **Proposer 3**: `format_usd`, `CostSummary`,
  `aggregate_costs`, `BudgetCheck`, `check_budget`, `diff_costs`.
  These are pure, additive, and unblock the next round's
  `report.py` cleanup. Reject the suggestion to refactor
  `report.py` / `cli.py` *this round* — out of scope per
  Constitution rule 2 (one coherent thematic improvement).

## Combined public surface

```python
# constants
SNAPSHOT_DATE: str
CURRENCY: str = "USD"
PRICE_LIST: dict[str, tuple[float, float]]    # back-compat view
RATE_TABLE: dict[str, TokenRates]              # full rates
ALIASES: dict[str, str]                        # alias -> canonical

# dataclasses
class TokenRates: ...        # P2
class CostBreakdown: ...     # P2
class CostSummary: ...       # P3
class BudgetCheck: ...       # P3

# errors
class MissingPriceError(KeyError): ...

# functions
def compute_cost(model, usage) -> float
def compute_cost_breakdown(model, usage) -> CostBreakdown
def resolve_model(name) -> str | None
def is_known(name) -> bool
def set_strict(on: bool) -> None
def format_usd(amount, *, precision=4, unit="$") -> str
def aggregate_costs(steps) -> CostSummary
def check_budget(steps, budget_usd) -> BudgetCheck
def diff_costs(steps_a, steps_b) -> dict[str, float]
```

## Algorithm refinements over P2

P2's pseudocode silently assumed `prompt_total >= cached`. Real
provider responses occasionally violate that (rounding glitches,
double-counting). Defensive clamp: `non_cached = max(0,
prompt_total - cached)` and same for `visible_out = max(0,
completion - reasoning)`. Otherwise we'd report negative dollars.

P2's `compute_cost_breakdown` should also accept the legacy
flat-dict shape (`{prompt_tokens, completion_tokens}` only) and
treat all the optional fields as 0 — this is what
`tests/test_shims.py:169` will pass in via the FakeOpenAI usage
shape.

## Algorithm refinements over P1

P1 wanted `set_strict` as a module-level toggle. That makes
test isolation painful. Refine to: `set_strict(on)` returns a
context-manager-friendly object (also callable as a setter) so
tests can write `with set_strict(True): ...`. Implementation:
`set_strict` returns a small `_StrictContext` whose `__enter__` /
`__exit__` save and restore the previous value.

P1's deprecation warning: emit at most once per process per model.
Use a module-level `_warned: set[str]`.

## Algorithm refinements over P3

`aggregate_costs(steps)` should tolerate steps without a
`cost_usd` field by recomputing from `model` + `usage` if those
are present — recovers cost when reading a trace that pre-dates
the field. Falls back to 0.0 only when neither is available.

`diff_costs(a, b)` returns numbers as **`b - a`** consistently;
`__total__` is the overall delta. P3 didn't pin the sign — pin
it now.

## Catalog (final list for this round)

OpenAI: `gpt-4o-2024-11-20`, `gpt-4o-2024-08-06`,
`gpt-4o-mini-2024-07-18`, `gpt-4.1-2025-04-14`,
`gpt-4.1-mini-2025-04-14`, `gpt-4.1-nano-2025-04-14`,
`o1-2024-12-17`, `o1-mini-2024-09-12`, `o3-mini-2025-01-31`.

Anthropic: `claude-3-5-sonnet-20241022`,
`claude-3-5-haiku-20241022`, `claude-3-7-sonnet-20250219`,
`claude-sonnet-4-20250514`, `claude-opus-4-20250514`,
`claude-haiku-4-20250514`.

Google: `gemini-2.5-pro-2025-03-25`, `gemini-2.5-flash-2025-04-09`.

Test stub: `fake-llm`.

Deprecated rows (priced + warned): `gpt-4-0613`, `gpt-3.5-turbo-0613`,
`claude-2.1`.

Aliases: `gpt-4o -> gpt-4o-2024-11-20`,
`gpt-4o-mini -> gpt-4o-mini-2024-07-18`,
`gpt-4.1 -> gpt-4.1-2025-04-14`,
`gpt-4.1-mini -> gpt-4.1-mini-2025-04-14`,
`o1 -> o1-2024-12-17`,
`o3-mini -> o3-mini-2025-01-31`,
`claude-3.5-sonnet -> claude-3-5-sonnet-20241022`,
`claude-3.5-haiku -> claude-3-5-haiku-20241022`,
`claude-3.7-sonnet -> claude-3-7-sonnet-20250219`,
`claude-sonnet-4 -> claude-sonnet-4-20250514`,
`claude-opus-4 -> claude-opus-4-20250514`,
`claude-haiku-4 -> claude-haiku-4-20250514`,
`gemini-2.5-pro -> gemini-2.5-pro-2025-03-25`,
`gemini-2.5-flash -> gemini-2.5-flash-2025-04-09`.

## Tests (deduped from P1+P2+P3)

`tests/test_pricing.py`:

1-7: from P2 (signature, breakdown sums, cached, reasoning,
     cache_creation, unknown-model zero, recorder/replay path).
8-13: from P1 (alias round-trip, unknown strict raises, deprecated
     warns once, snapshot date format, PRICE_LIST back-compat,
     resolve_model None on unknown).
14-18: from P3 (format_usd default + negative, aggregate_costs
     per_model + per_step_kind + most_expensive top-5, check_budget
     under/over, diff_costs total + disjoint).
19: `test_set_strict_is_context_manager` — refinement above.
20: `test_aggregate_costs_recomputes_from_usage_when_cost_missing`
     — refinement above.

## Citation map

* P1: catalog scope, aliases, snapshot date, deprecation warnings,
  `MissingPriceError`, `is_known`, `resolve_model`.
* P2: `TokenRates`, `CostBreakdown`, multi-tier algorithm, OpenAI
  /Anthropic usage normalization, derived `PRICE_LIST` for back-compat.
* P3: `format_usd`, `CostSummary`, `aggregate_costs`,
  `BudgetCheck`, `check_budget`, `diff_costs`.

## Out of scope (saved for L3 to deepen)

* Currency conversion (multi-currency BudgetCheck).
* JSON serialisation of CostBreakdown into trace headers.
* Light surface integration with `recorder.py` to also persist
  the breakdown alongside the float on each step.
* Snapshot freshness check / staleness warning.
