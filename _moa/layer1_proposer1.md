# Proposer 1 — data-driven catalog with aliasing & deprecation

## Framing

`stepback/pricing.py` is a 25-line dict with five rows. For an
agent-time-travel debugger that brags in the README about "cost
deltas across branches are meaningful", a five-model frozen catalog
hard-coded inline is a liability:

* New models get added every couple of weeks; today's catalog is
  already missing `gpt-4.1`, `gpt-4.1-mini`, `o1-2024-12-17`,
  `o3-mini-2025-01-31`, `claude-sonnet-4-20250514`,
  `claude-opus-4-20250514`, `gemini-2.5-pro-2025-03-25`.
* Real provider names get versioned: a trace recorded against
  `claude-3-5-sonnet-20241022` should still resolve cost when the
  user replays it under the alias `claude-3.5-sonnet`.
* Deprecated models silently fall through `PRICE_LIST.get(model,
  (0.0, 0.0))` and report **zero cost**, which is exactly the kind
  of silent-zero that makes branch-cost comparison useless.

This proposer treats pricing as **data**, not code.

## Architecture

```
stepback/
  pricing.py                 # public API, thin
  _pricing_data.py           # AUTO-GENERATED catalog dict (kept in
                             # repo for hermetic tests)
  _pricing_data.json         # source of truth, snapshot date pinned
```

* `_pricing_data.json` carries `{snapshot_date, currency, models:
  [...]}` where each model row is
  `{id, in_per_1k, out_per_1k, aliases: [...], deprecated: bool,
   replaced_by: str|null, family: str}`.
* `pricing.py` loads JSON at import, builds a flattened dict
  `_RESOLVED` keyed by both canonical id and every alias.
* `compute_cost(model, usage) -> float` keeps its signature.
* New: `resolve_model(name) -> str | None` returns the canonical
  id (or None if unknown).
* New: `is_known(name) -> bool`.
* New: `MissingPriceError` raised when `compute_cost` is called in
  *strict mode* against an unknown model. Default behavior stays
  permissive (returns 0.0) so existing callers don't break, but a
  module-level `set_strict(True)` flag makes silent zeros an
  exception — useful in tests and CI.
* New: warnings via `warnings.warn(DeprecationWarning, ...)` first
  time a deprecated model is priced in a process; suggests
  `replaced_by`.

## Public surface (final)

```python
PRICE_LIST: dict[str, tuple[float, float]]   # backwards-compatible
SNAPSHOT_DATE: str                           # e.g. "2026-04-01"
CURRENCY: str                                # "USD"

def compute_cost(model: str, usage: dict) -> float: ...
def resolve_model(name: str) -> str | None: ...
def is_known(name: str) -> bool: ...
def set_strict(on: bool) -> None: ...

class MissingPriceError(KeyError): ...
```

`PRICE_LIST` stays exposed because `tests/test_shims.py` and the
README docstring reference it ("pricing.PRICE_LIST has both
models").

## Catalog scope (this round)

Add at minimum:

* `gpt-4o-2024-11-20`, `gpt-4o-2024-08-06`, `gpt-4o-mini-2024-07-18`
* `gpt-4.1-2025-04-14`, `gpt-4.1-mini-2025-04-14`,
  `gpt-4.1-nano-2025-04-14`
* `o1-2024-12-17`, `o1-mini-2024-09-12`, `o3-mini-2025-01-31`
* `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`
* `claude-3-7-sonnet-20250219`
* `claude-sonnet-4-20250514`, `claude-opus-4-20250514`
* `claude-haiku-4-20250514`
* `gemini-2.5-pro-2025-03-25`, `gemini-2.5-flash-2025-04-09`
* keep `fake-llm` for tests
* aliases: `claude-3.5-sonnet -> claude-3-5-sonnet-20241022`,
  `claude-sonnet-4 -> claude-sonnet-4-20250514`,
  `gpt-4o -> gpt-4o-2024-11-20`, `gpt-4.1 -> gpt-4.1-2025-04-14`,
  `o1 -> o1-2024-12-17`.
* deprecated rows: `gpt-4-0613`, `claude-2.1`, both with
  `replaced_by` pointing at the most current model in the family.

## Tests

`tests/test_pricing.py`:

1. `test_known_models_round_trip` — every canonical id resolves to
   itself.
2. `test_aliases_resolve` — every alias resolves to a canonical
   id and `compute_cost` over alias == compute_cost over canonical.
3. `test_unknown_model_returns_zero_in_default_mode`.
4. `test_unknown_model_raises_in_strict_mode`.
5. `test_deprecated_model_emits_warning_once`.
6. `test_snapshot_date_format` — `re.match(r"\d{4}-\d{2}-\d{2}",
   SNAPSHOT_DATE)`.
7. `test_price_list_backward_compat` — `("gpt-4o-2024-11-20" in
   PRICE_LIST and "fake-llm" in PRICE_LIST)`.

## Risks

* `set_strict(True)` could leak across tests. Mitigation: tests use
  a `monkeypatch` context that restores after each test.
* JSON I/O at import: must read from a path computed via
  `importlib.resources` so it works in installed wheels too.

## Out of scope (saved for L2/L3)

* Cached-input / reasoning / image-token pricing tiers.
* Aggregation helpers (sum across steps).
* Currency conversion.
