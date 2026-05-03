"""Per-model price catalog + cost analytics for stepback traces.

Prices are USD per 1k tokens, taken from published lists as of
``SNAPSHOT_DATE``. Replay computes cost the same way as record so
cost deltas across branches are meaningful.

Public surface (see ``_moa/layer3_refiner.md`` for design):

* Constants: ``SNAPSHOT_DATE``, ``CURRENCY``, ``PRICE_LIST``,
  ``RATE_TABLE``, ``ALIASES``.
* Dataclasses: ``TokenRates``, ``CostBreakdown``, ``CostSummary``,
  ``BudgetCheck``.  All carry ``to_dict()``.
* Errors: ``MissingPriceError``.
* Functions: ``compute_cost``, ``compute_cost_breakdown``,
  ``resolve_model``, ``is_known``, ``set_strict``, ``format_usd``,
  ``aggregate_costs``, ``check_budget``, ``diff_costs``,
  ``snapshot_age_days``.

Step-record contract (consumed by ``aggregate_costs`` /
``diff_costs``)::

    {
        "step_id":   str,
        "step_kind": str,    # "llm_call" | "tool_call" | "router" | ...
        "model":     str,    # may be empty for non-llm
        "usage":     dict,   # may be empty
        "cost_usd":  float,  # optional; recomputed if missing
    }

Records that don't match are silently treated as no-cost steps.
"""
from __future__ import annotations

import datetime as _dt
import warnings
from dataclasses import asdict, dataclass, field
from typing import Iterable, Mapping

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SNAPSHOT_DATE: str = "2026-04-01"
CURRENCY: str = "USD"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MissingPriceError(KeyError):
    """Raised by ``compute_cost`` in strict mode when the model is unknown."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenRates:
    input_per_1k: float
    output_per_1k: float
    cached_input_per_1k: float | None = None
    reasoning_per_1k: float | None = None
    image_input_per_1k: float | None = None
    audio_input_per_1k: float | None = None
    cache_write_per_1k: float | None = None
    deprecated: bool = False
    replaced_by: str | None = None
    family: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CostBreakdown:
    model: str
    input_usd: float = 0.0
    cached_input_usd: float = 0.0
    output_usd: float = 0.0
    reasoning_usd: float = 0.0
    image_input_usd: float = 0.0
    audio_input_usd: float = 0.0
    cache_write_usd: float = 0.0
    total_usd: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CostSummary:
    total_usd: float
    n_steps: int
    per_model: dict[str, float] = field(default_factory=dict)
    per_step_kind: dict[str, float] = field(default_factory=dict)
    most_expensive: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_usd": self.total_usd,
            "n_steps": self.n_steps,
            "per_model": dict(self.per_model),
            "per_step_kind": dict(self.per_step_kind),
            "most_expensive": [list(p) for p in self.most_expensive],
        }


@dataclass(frozen=True)
class BudgetCheck:
    budget_usd: float
    spent_usd: float
    remaining_usd: float
    over_budget: bool
    fraction_used: float

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

# All numbers are published-list, USD per 1k tokens, snapshot 2026-04-01.
RATE_TABLE: dict[str, TokenRates] = {
    # ---------------- OpenAI ----------------
    "gpt-4o-2024-11-20": TokenRates(
        input_per_1k=0.0025, output_per_1k=0.01,
        cached_input_per_1k=0.00125, family="gpt-4o",
    ),
    "gpt-4o-2024-08-06": TokenRates(
        input_per_1k=0.0025, output_per_1k=0.01,
        cached_input_per_1k=0.00125, family="gpt-4o",
    ),
    "gpt-4o-mini-2024-07-18": TokenRates(
        input_per_1k=0.00015, output_per_1k=0.0006,
        cached_input_per_1k=0.000075, family="gpt-4o-mini",
    ),
    "gpt-4.1-2025-04-14": TokenRates(
        input_per_1k=0.002, output_per_1k=0.008,
        cached_input_per_1k=0.0005, family="gpt-4.1",
    ),
    "gpt-4.1-mini-2025-04-14": TokenRates(
        input_per_1k=0.0004, output_per_1k=0.0016,
        cached_input_per_1k=0.0001, family="gpt-4.1-mini",
    ),
    "gpt-4.1-nano-2025-04-14": TokenRates(
        input_per_1k=0.0001, output_per_1k=0.0004,
        cached_input_per_1k=0.000025, family="gpt-4.1-nano",
    ),
    "o1-2024-12-17": TokenRates(
        input_per_1k=0.015, output_per_1k=0.06,
        cached_input_per_1k=0.0075, reasoning_per_1k=0.06, family="o1",
    ),
    "o1-mini-2024-09-12": TokenRates(
        input_per_1k=0.003, output_per_1k=0.012,
        cached_input_per_1k=0.0015, reasoning_per_1k=0.012, family="o1-mini",
    ),
    "o3-mini-2025-01-31": TokenRates(
        input_per_1k=0.0011, output_per_1k=0.0044,
        cached_input_per_1k=0.00055, reasoning_per_1k=0.0044, family="o3-mini",
    ),
    # ---------------- Anthropic ----------------
    "claude-3-5-sonnet-20241022": TokenRates(
        input_per_1k=0.003, output_per_1k=0.015,
        cached_input_per_1k=0.0003, cache_write_per_1k=0.00375,
        family="claude-3.5-sonnet",
    ),
    "claude-3-5-haiku-20241022": TokenRates(
        input_per_1k=0.00025, output_per_1k=0.00125,
        cached_input_per_1k=0.000025, cache_write_per_1k=0.0003125,
        family="claude-3.5-haiku",
    ),
    "claude-3-7-sonnet-20250219": TokenRates(
        input_per_1k=0.003, output_per_1k=0.015,
        cached_input_per_1k=0.0003, cache_write_per_1k=0.00375,
        family="claude-3.7-sonnet",
    ),
    "claude-sonnet-4-20250514": TokenRates(
        input_per_1k=0.003, output_per_1k=0.015,
        cached_input_per_1k=0.0003, cache_write_per_1k=0.00375,
        family="claude-sonnet-4",
    ),
    "claude-opus-4-20250514": TokenRates(
        input_per_1k=0.015, output_per_1k=0.075,
        cached_input_per_1k=0.0015, cache_write_per_1k=0.01875,
        family="claude-opus-4",
    ),
    "claude-haiku-4-20250514": TokenRates(
        input_per_1k=0.0008, output_per_1k=0.004,
        cached_input_per_1k=0.00008, cache_write_per_1k=0.001,
        family="claude-haiku-4",
    ),
    # ---------------- Google ----------------
    "gemini-2.5-pro-2025-03-25": TokenRates(
        input_per_1k=0.00125, output_per_1k=0.005,
        cached_input_per_1k=0.0003125, family="gemini-2.5-pro",
    ),
    "gemini-2.5-flash-2025-04-09": TokenRates(
        input_per_1k=0.000075, output_per_1k=0.0003,
        cached_input_per_1k=0.0000188, family="gemini-2.5-flash",
    ),
    # ---------------- AWS Bedrock (native models) ----------------
    # Re-hosted Anthropic Claude on Bedrock prices identically to
    # native Anthropic; the Bedrock shim canonicalises modelId so
    # those calls hit the existing claude-* rows. The rates below
    # cover Bedrock-native foundation models that aren't otherwise in
    # the catalog. Snapshot 2026-04-01, on-demand US-East-1 pricing.
    "meta.llama3-1-70b-instruct-v1:0": TokenRates(
        input_per_1k=0.00099, output_per_1k=0.00099,
        family="llama3.1-70b-bedrock",
    ),
    "meta.llama3-1-8b-instruct-v1:0": TokenRates(
        input_per_1k=0.00022, output_per_1k=0.00022,
        family="llama3.1-8b-bedrock",
    ),
    "meta.llama3-1-405b-instruct-v1:0": TokenRates(
        input_per_1k=0.00532, output_per_1k=0.016,
        family="llama3.1-405b-bedrock",
    ),
    "mistral.mistral-large-2407-v1:0": TokenRates(
        input_per_1k=0.002, output_per_1k=0.006,
        family="mistral-large-bedrock",
    ),
    "cohere.command-r-plus-v1:0": TokenRates(
        input_per_1k=0.003, output_per_1k=0.015,
        family="command-r-plus-bedrock",
    ),
    "amazon.nova-pro-v1:0": TokenRates(
        input_per_1k=0.0008, output_per_1k=0.0032,
        cached_input_per_1k=0.0002, family="nova-pro",
    ),
    "amazon.nova-lite-v1:0": TokenRates(
        input_per_1k=0.00006, output_per_1k=0.00024,
        cached_input_per_1k=0.000015, family="nova-lite",
    ),
    "amazon.nova-micro-v1:0": TokenRates(
        input_per_1k=0.000035, output_per_1k=0.00014,
        cached_input_per_1k=0.00000875, family="nova-micro",
    ),
    # ---------------- Test stub ----------------
    "fake-llm": TokenRates(
        input_per_1k=0.0001, output_per_1k=0.0001, family="fake",
    ),
    # ---------------- Deprecated ----------------
    "gpt-4-0613": TokenRates(
        input_per_1k=0.03, output_per_1k=0.06,
        deprecated=True, replaced_by="gpt-4.1-2025-04-14",
        family="gpt-4-legacy",
    ),
    "gpt-3.5-turbo-0613": TokenRates(
        input_per_1k=0.0015, output_per_1k=0.002,
        deprecated=True, replaced_by="gpt-4o-mini-2024-07-18",
        family="gpt-3.5-legacy",
    ),
    "claude-2.1": TokenRates(
        input_per_1k=0.008, output_per_1k=0.024,
        deprecated=True, replaced_by="claude-sonnet-4-20250514",
        family="claude-legacy",
    ),
}

ALIASES: dict[str, str] = {
    "gpt-4o": "gpt-4o-2024-11-20",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
    "o1": "o1-2024-12-17",
    "o1-mini": "o1-mini-2024-09-12",
    "o3-mini": "o3-mini-2025-01-31",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3.5-haiku": "claude-3-5-haiku-20241022",
    "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "claude-opus-4": "claude-opus-4-20250514",
    "claude-haiku-4": "claude-haiku-4-20250514",
    "gemini-2.5-pro": "gemini-2.5-pro-2025-03-25",
    "gemini-2.5-flash": "gemini-2.5-flash-2025-04-09",
    # Bedrock aliases — short forms so users don't need the full ARN-ish id.
    "llama3.1-70b": "meta.llama3-1-70b-instruct-v1:0",
    "llama3.1-8b": "meta.llama3-1-8b-instruct-v1:0",
    "llama3.1-405b": "meta.llama3-1-405b-instruct-v1:0",
    "mistral-large": "mistral.mistral-large-2407-v1:0",
    "command-r-plus": "cohere.command-r-plus-v1:0",
    "nova-pro": "amazon.nova-pro-v1:0",
    "nova-lite": "amazon.nova-lite-v1:0",
    "nova-micro": "amazon.nova-micro-v1:0",
}

# Backward-compat view consumed by tests/test_shims.py and the trace
# header.  Two-tuple (input_per_1k, output_per_1k) per model id.
PRICE_LIST: dict[str, tuple[float, float]] = {
    m: (r.input_per_1k, r.output_per_1k) for m, r in RATE_TABLE.items()
}


# ---------------------------------------------------------------------------
# Strict mode (context-manager friendly)
# ---------------------------------------------------------------------------


_strict: bool = False
_warned_deprecated: set[str] = set()
_stale_warned: bool = False


class _StrictContext:
    def __init__(self, previous: bool):
        self._previous = previous

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        global _strict
        _strict = self._previous
        return False


def set_strict(on: bool) -> _StrictContext:
    """Toggle strict mode.

    Usable as a setter (``set_strict(True)``) OR as a context manager
    (``with set_strict(True): ...``) which restores the previous value
    on exit.  The context-manager form is preferred in tests.
    """
    global _strict
    previous = _strict
    _strict = bool(on)
    return _StrictContext(previous)


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def resolve_model(name: str) -> str | None:
    if not name:
        return None
    if name in RATE_TABLE:
        return name
    if name in ALIASES:
        return ALIASES[name]
    return None


def is_known(name: str) -> bool:
    return resolve_model(name) is not None


def _warn_deprecated_once(canonical: str, rates: TokenRates) -> None:
    if not rates.deprecated or canonical in _warned_deprecated:
        return
    _warned_deprecated.add(canonical)
    msg = f"pricing: model {canonical!r} is deprecated"
    if rates.replaced_by:
        msg += f"; consider {rates.replaced_by!r}"
    warnings.warn(msg, DeprecationWarning, stacklevel=3)


def snapshot_age_days(today: _dt.date | None = None) -> int:
    if today is None:
        today = _dt.date.today()
    snap = _dt.date.fromisoformat(SNAPSHOT_DATE)
    return (today - snap).days


def _warn_stale_once() -> None:
    global _stale_warned
    if _stale_warned:
        return
    age = snapshot_age_days()
    if age > 180:
        _stale_warned = True
        warnings.warn(
            f"pricing: snapshot {SNAPSHOT_DATE} is {age} days old; "
            "numbers may have drifted from current published rates",
            UserWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------


def _extract_buckets(usage: Mapping) -> dict[str, int]:
    """Normalise OpenAI- and Anthropic-shaped usage dicts into named
    integer token counts.  Defensive: clamps negatives to 0."""
    if not usage:
        return {}
    pt = int(usage.get("prompt_tokens", 0) or 0)
    ct = int(usage.get("completion_tokens", 0) or 0)

    # OpenAI nested details
    pt_details = usage.get("prompt_tokens_details") or {}
    ct_details = usage.get("completion_tokens_details") or {}
    cached = int(pt_details.get("cached_tokens", 0) or 0)
    reasoning = int(ct_details.get("reasoning_tokens", 0) or 0)

    # Top-level alternates (some shims flatten)
    cached = cached or int(usage.get("cached_tokens", 0) or 0)
    reasoning = reasoning or int(usage.get("reasoning_tokens", 0) or 0)

    image_in = int(usage.get("image_tokens", 0) or 0)
    audio_in = int(usage.get("audio_tokens", 0) or 0)

    # Anthropic
    cache_write = int(usage.get("cache_creation_input_tokens", 0) or 0)
    cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
    if cache_read and not cached:
        cached = cache_read

    non_cached = max(0, pt - cached - cache_write)
    visible_out = max(0, ct - reasoning)

    return {
        "non_cached_input": non_cached,
        "cached_input": cached,
        "cache_write": cache_write,
        "visible_output": visible_out,
        "reasoning": reasoning,
        "image_input": image_in,
        "audio_input": audio_in,
    }


def compute_cost_breakdown(model: str, usage: Mapping) -> CostBreakdown:
    _warn_stale_once()
    canonical = resolve_model(model)
    if canonical is None:
        if _strict and model:
            raise MissingPriceError(f"no price entry for model {model!r}")
        return CostBreakdown(model=model)

    rates = RATE_TABLE[canonical]
    _warn_deprecated_once(canonical, rates)

    buckets = _extract_buckets(usage)
    if not buckets:
        return CostBreakdown(model=canonical)

    in_p = rates.input_per_1k
    cached_p = rates.cached_input_per_1k if rates.cached_input_per_1k is not None else in_p
    out_p = rates.output_per_1k
    reason_p = rates.reasoning_per_1k if rates.reasoning_per_1k is not None else out_p
    img_p = rates.image_input_per_1k if rates.image_input_per_1k is not None else in_p
    audio_p = rates.audio_input_per_1k if rates.audio_input_per_1k is not None else in_p
    write_p = rates.cache_write_per_1k if rates.cache_write_per_1k is not None else in_p

    input_usd = buckets["non_cached_input"] * in_p / 1000.0
    cached_usd = buckets["cached_input"] * cached_p / 1000.0
    output_usd = buckets["visible_output"] * out_p / 1000.0
    reasoning_usd = buckets["reasoning"] * reason_p / 1000.0
    image_usd = buckets["image_input"] * img_p / 1000.0
    audio_usd = buckets["audio_input"] * audio_p / 1000.0
    write_usd = buckets["cache_write"] * write_p / 1000.0

    total = (
        input_usd + cached_usd + output_usd + reasoning_usd
        + image_usd + audio_usd + write_usd
    )

    return CostBreakdown(
        model=canonical,
        input_usd=round(input_usd, 10),
        cached_input_usd=round(cached_usd, 10),
        output_usd=round(output_usd, 10),
        reasoning_usd=round(reasoning_usd, 10),
        image_input_usd=round(image_usd, 10),
        audio_input_usd=round(audio_usd, 10),
        cache_write_usd=round(write_usd, 10),
        total_usd=round(total, 10),
    )


def compute_cost(model: str, usage: Mapping) -> float:
    """Backward-compatible: returns the total USD cost as a float."""
    return round(compute_cost_breakdown(model, usage or {}).total_usd, 8)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_usd(amount: float, *, precision: int = 4, unit: str = "$") -> str:
    """Stable currency formatting; negatives render as ``-$0.0012``."""
    if amount < 0:
        return f"-{unit}{abs(amount):.{precision}f}"
    return f"{unit}{amount:.{precision}f}"


# ---------------------------------------------------------------------------
# Aggregation / budget / diff
# ---------------------------------------------------------------------------


def _step_cost(rec: Mapping) -> float:
    if "cost_usd" in rec:
        try:
            return float(rec["cost_usd"])
        except (TypeError, ValueError):
            pass
    model = rec.get("model") or ""
    usage = rec.get("usage") or {}
    if model or usage:
        return compute_cost(model, usage)
    return 0.0


def aggregate_costs(steps: Iterable[Mapping]) -> CostSummary:
    per_model: dict[str, float] = {}
    per_kind: dict[str, float] = {}
    by_step: list[tuple[str, float]] = []
    n = 0
    total = 0.0
    for rec in steps:
        if not isinstance(rec, Mapping):
            continue
        cost = _step_cost(rec)
        n += 1
        total += cost
        model = rec.get("model") or "(none)"
        per_model[model] = per_model.get(model, 0.0) + cost
        kind = rec.get("step_kind")
        if kind:
            per_kind[kind] = per_kind.get(kind, 0.0) + cost
        sid = str(rec.get("step_id") or f"#{n}")
        by_step.append((sid, cost))
    by_step.sort(key=lambda p: p[1], reverse=True)
    return CostSummary(
        total_usd=round(total, 10),
        n_steps=n,
        per_model={k: round(v, 10) for k, v in per_model.items()},
        per_step_kind={k: round(v, 10) for k, v in per_kind.items()},
        most_expensive=by_step[:5],
    )


def check_budget(steps: Iterable[Mapping], budget_usd: float) -> BudgetCheck:
    summary = aggregate_costs(steps)
    spent = summary.total_usd
    remaining = budget_usd - spent
    fraction = (spent / budget_usd) if budget_usd > 0 else float("inf")
    return BudgetCheck(
        budget_usd=float(budget_usd),
        spent_usd=spent,
        remaining_usd=round(remaining, 10),
        over_budget=spent > budget_usd,
        fraction_used=fraction,
    )


def diff_costs(
    steps_a: Iterable[Mapping], steps_b: Iterable[Mapping]
) -> dict[str, float]:
    """Per-model delta ``b - a``.  ``__total__`` is the overall delta."""
    a = aggregate_costs(steps_a)
    b = aggregate_costs(steps_b)
    keys = set(a.per_model) | set(b.per_model)
    out: dict[str, float] = {}
    for k in keys:
        out[k] = round(b.per_model.get(k, 0.0) - a.per_model.get(k, 0.0), 10)
    out["__total__"] = round(b.total_usd - a.total_usd, 10)
    return out


# ---------------------------------------------------------------------------
# CLI: ``python -m stepback.pricing``
# ---------------------------------------------------------------------------


def _markdown_table() -> str:
    lines = [
        f"# stepback price catalog (snapshot {SNAPSHOT_DATE}, {CURRENCY})",
        "",
        "| model | family | in /1k | out /1k | cached /1k | reasoning /1k | deprecated |",
        "|---|---|---:|---:|---:|---:|:---:|",
    ]
    for mid, r in RATE_TABLE.items():
        cached = "—" if r.cached_input_per_1k is None else f"{r.cached_input_per_1k}"
        reason = "—" if r.reasoning_per_1k is None else f"{r.reasoning_per_1k}"
        dep = "yes" if r.deprecated else ""
        lines.append(
            f"| `{mid}` | {r.family} | {r.input_per_1k} | {r.output_per_1k} "
            f"| {cached} | {reason} | {dep} |"
        )
    lines.append("")
    lines.append(f"_{len(ALIASES)} aliases registered._")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(_markdown_table())
