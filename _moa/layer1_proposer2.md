# Proposer 2 — structured `CostBreakdown` with multi-tier token pricing

## Framing

The single biggest factual error in the current `pricing.py` is its
two-bucket model: `(in_per_1k, out_per_1k)` × `(prompt_tokens,
completion_tokens)`. Every major provider in 2026 charges along
**at least four** axes:

* **Cached input tokens** — OpenAI prompt caching, Anthropic prompt
  caching, Gemini context caching. Cached input is typically priced
  at 10–25% of the full input rate.
* **Reasoning tokens** — `o1`, `o3-mini`, Claude extended-thinking
  charge for hidden reasoning at the *output* rate, but they appear
  in `usage` as `reasoning_tokens` and are NOT in `completion_tokens`
  on every provider's return shape.
* **Image / audio input tokens** — gpt-4o vision, Claude vision,
  Gemini multimodal report image and audio token counts separately.
* **Cache-write tokens** (Anthropic only) — written tokens cost
  ~25% MORE than normal input the first time and become cheap on
  subsequent reads.

A cost number that flattens all of those into `pt * in_p + ct * out_p`
under-reports a vision-heavy or reasoning-heavy step by 5–50×. For
an agent debugger whose marketing pitch is "cost deltas across
branches are meaningful", that's the central correctness bug.

## Architecture

Replace the float return with a `CostBreakdown` dataclass; keep the
`compute_cost(model, usage) -> float` signature for backward compat
by returning `breakdown.total_usd`.

```python
@dataclass(frozen=True)
class TokenRates:
    input_per_1k:        float
    cached_input_per_1k: float | None  # None => fall back to input
    output_per_1k:       float
    reasoning_per_1k:    float | None  # None => fall back to output
    image_input_per_1k:  float | None
    audio_input_per_1k:  float | None
    cache_write_per_1k:  float | None  # Anthropic
    snapshot_date:       str

@dataclass(frozen=True)
class CostBreakdown:
    model:               str
    input_usd:           float
    cached_input_usd:    float
    output_usd:          float
    reasoning_usd:       float
    image_input_usd:     float
    audio_input_usd:     float
    cache_write_usd:     float
    total_usd:           float

    def to_dict(self) -> dict[str, float]: ...
```

## Public surface

```python
PRICE_LIST: dict[str, tuple[float, float]]   # backward compat:
                                              # only (in, out) shown
RATE_TABLE: dict[str, TokenRates]            # full tiers

def compute_cost(model: str, usage: dict) -> float:
    return compute_cost_breakdown(model, usage).total_usd

def compute_cost_breakdown(model: str, usage: dict) -> CostBreakdown:
    ...
```

`compute_cost_breakdown` reads optional usage keys:
`prompt_tokens`, `completion_tokens`, `cached_tokens`,
`prompt_tokens_details.cached_tokens` (OpenAI),
`completion_tokens_details.reasoning_tokens` (OpenAI),
`cache_creation_input_tokens` (Anthropic),
`cache_read_input_tokens` (Anthropic), `image_tokens`,
`audio_tokens`. Tokens that are accounted for under a sub-bucket
are subtracted from the parent bucket so we never double-charge.

Algorithm pseudocode for OpenAI shape:

```
prompt_total   = usage["prompt_tokens"]
cached         = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
non_cached     = prompt_total - cached
completion     = usage["completion_tokens"]
reasoning      = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
visible_out    = completion - reasoning
input_usd      = non_cached * rates.input_per_1k / 1000
cached_usd     = cached     * (rates.cached_input_per_1k or rates.input_per_1k) / 1000
output_usd     = visible_out * rates.output_per_1k / 1000
reasoning_usd  = reasoning  * (rates.reasoning_per_1k or rates.output_per_1k) / 1000
total_usd      = sum of the above
```

Anthropic's usage shape is normalised in the same function:
`cache_creation_input_tokens` -> `cache_write_usd`,
`cache_read_input_tokens` -> `cached_input_usd`.

## Catalog scope

`RATE_TABLE` carries TokenRates for at least:

* `gpt-4o-2024-11-20`           (cached at 50%, no reasoning)
* `gpt-4o-mini-2024-07-18`
* `gpt-4.1-2025-04-14`           (cached at 25%)
* `o1-2024-12-17`                (reasoning_per_1k = output_per_1k)
* `o3-mini-2025-01-31`
* `claude-3-5-sonnet-20241022`   (cache_write at +25%, cache_read at 10%)
* `claude-3-5-haiku-20241022`
* `claude-sonnet-4-20250514`
* `gemini-2.5-pro-2025-03-25`
* `fake-llm`                     (kept for tests)

`PRICE_LIST` is derived: `{m: (r.input_per_1k, r.output_per_1k) for
m, r in RATE_TABLE.items()}`. So existing `PRICE_LIST` consumers
(`tests/test_shims.py:169`) keep working with no edit.

## Tests (`tests/test_pricing.py`)

1. `test_compute_cost_signature_unchanged` — round-trips a basic
   usage dict to a positive float; result equals the legacy
   formula `pt*in + ct*out` when no cache/reasoning fields.
2. `test_breakdown_attributes_total` — sum of sub-tier USD numbers
   equals `total_usd` to 1e-9.
3. `test_cached_tokens_priced_at_cached_rate` — a usage with
   80% cached input on `gpt-4.1-2025-04-14` is ~3-4× cheaper than
   the same prompt with no cache.
4. `test_reasoning_tokens_charged_at_output_rate` — `o1` step
   with `completion_tokens=100` and
   `completion_tokens_details.reasoning_tokens=80` charges
   reasoning at the output rate.
5. `test_anthropic_cache_creation_charged_more` — a Claude usage
   with `cache_creation_input_tokens` is more expensive than the
   same step with `cache_read_input_tokens`.
6. `test_unknown_model_zero_breakdown` — every field is 0.0.
7. `test_compute_cost_recorder_replay_path` — sanity check the
   `recorder.compute_cost(model, usage)` call site stays a float.

## Why structured beats float

Inside `report.py`, branch-cost diffing today shows "branch B
costs $0.018 more". With a breakdown it can show "branch B costs
$0.018 more, of which $0.014 is reasoning tokens" — that is the
debugging insight `report.py` exists to surface.

## Out of scope

Catalog file format / aliasing (handled differently — see Proposer 1).
Aggregation utilities across many steps (Proposer 3).
