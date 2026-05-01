"""Per-model price list, pinned in the trace header.

Prices are USD per 1k tokens, taken from published lists as of
2026-04-01. Replay computes cost the same way so cost deltas across
branches are meaningful.
"""
from __future__ import annotations

PRICE_LIST = {
    # model                       (in_per_1k,  out_per_1k)
    "gpt-4o-2024-11-20":          (0.0025,     0.01),
    "gpt-4o-mini-2024-07-18":     (0.00015,    0.0006),
    "claude-3-5-sonnet-20241022": (0.003,      0.015),
    "claude-3-5-haiku-20241022":  (0.00025,    0.00125),
    "fake-llm":                   (0.0001,     0.0001),
}


def compute_cost(model: str, usage: dict) -> float:
    if not usage:
        return 0.0
    in_p, out_p = PRICE_LIST.get(model, (0.0, 0.0))
    pt = float(usage.get("prompt_tokens", 0))
    ct = float(usage.get("completion_tokens", 0))
    return round((pt * in_p + ct * out_p) / 1000.0, 8)
