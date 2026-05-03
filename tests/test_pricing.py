"""Tests for ``stepback.pricing`` — applies _moa/layer3_refiner.md."""
from __future__ import annotations

import datetime as dt
import json
import re
import warnings

import pytest

from stepback import pricing
from stepback.pricing import (
    ALIASES,
    PRICE_LIST,
    RATE_TABLE,
    SNAPSHOT_DATE,
    BudgetCheck,
    CostBreakdown,
    CostSummary,
    MissingPriceError,
    TokenRates,
    aggregate_costs,
    check_budget,
    compute_cost,
    compute_cost_breakdown,
    diff_costs,
    format_usd,
    is_known,
    resolve_model,
    set_strict,
    snapshot_age_days,
)


# ---------------------------------------------------------------------------
# 1-7  Proposer 2: structured breakdown / multi-tier
# ---------------------------------------------------------------------------


def test_compute_cost_signature_unchanged():
    cost = compute_cost(
        "gpt-4o-2024-11-20",
        {"prompt_tokens": 1000, "completion_tokens": 500},
    )
    expected = (1000 * 0.0025 + 500 * 0.01) / 1000.0
    assert isinstance(cost, float)
    assert cost == pytest.approx(expected, rel=1e-9)


def test_breakdown_attributes_total():
    b = compute_cost_breakdown(
        "claude-3-5-sonnet-20241022",
        {
            "prompt_tokens": 2000,
            "completion_tokens": 800,
            "cache_creation_input_tokens": 500,
            "cache_read_input_tokens": 300,
        },
    )
    parts = (
        b.input_usd + b.cached_input_usd + b.output_usd + b.reasoning_usd
        + b.image_input_usd + b.audio_input_usd + b.cache_write_usd
    )
    assert b.total_usd == pytest.approx(parts, abs=1e-9)


def test_cached_tokens_priced_at_cached_rate():
    full = compute_cost(
        "gpt-4.1-2025-04-14",
        {"prompt_tokens": 10000, "completion_tokens": 0},
    )
    cached = compute_cost(
        "gpt-4.1-2025-04-14",
        {
            "prompt_tokens": 10000,
            "completion_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 8000},
        },
    )
    # gpt-4.1 cached rate is 0.0005 vs full 0.002 — 4x cheaper for cached.
    assert cached < full
    assert cached < full * 0.6


def test_reasoning_tokens_charged_at_output_rate():
    b = compute_cost_breakdown(
        "o1-2024-12-17",
        {
            "prompt_tokens": 0,
            "completion_tokens": 100,
            "completion_tokens_details": {"reasoning_tokens": 80},
        },
    )
    # 80 reasoning at $0.06/1k = $0.0048; 20 visible output at $0.06/1k = $0.0012
    assert b.reasoning_usd == pytest.approx(0.0048, abs=1e-9)
    assert b.output_usd == pytest.approx(0.0012, abs=1e-9)


def test_anthropic_cache_creation_charged_more():
    write = compute_cost(
        "claude-3-5-sonnet-20241022",
        {
            "prompt_tokens": 1000,
            "completion_tokens": 0,
            "cache_creation_input_tokens": 1000,
        },
    )
    read = compute_cost(
        "claude-3-5-sonnet-20241022",
        {
            "prompt_tokens": 1000,
            "completion_tokens": 0,
            "cache_read_input_tokens": 1000,
        },
    )
    assert write > read


def test_unknown_model_zero_breakdown():
    b = compute_cost_breakdown("not-a-real-model", {"prompt_tokens": 100})
    assert b.total_usd == 0.0
    assert b.input_usd == 0.0
    assert b.output_usd == 0.0


def test_recorder_replay_path_returns_float():
    # The signature recorder.py:120 and replay.py:384 rely on.
    out = compute_cost("fake-llm", {"prompt_tokens": 5, "completion_tokens": 5})
    assert isinstance(out, float)
    assert out >= 0.0


# ---------------------------------------------------------------------------
# 8-13  Proposer 1: catalog / aliases / strict / deprecation
# ---------------------------------------------------------------------------


def test_aliases_round_trip_to_canonical():
    for alias, canonical in ALIASES.items():
        assert resolve_model(alias) == canonical
        assert canonical in RATE_TABLE
        # Cost via alias equals cost via canonical id.
        u = {"prompt_tokens": 100, "completion_tokens": 50}
        assert compute_cost(alias, u) == compute_cost(canonical, u)


def test_unknown_model_raises_in_strict_mode():
    with set_strict(True):
        with pytest.raises(MissingPriceError):
            compute_cost("does-not-exist-2099", {"prompt_tokens": 1})
    # Strict was restored on exit.
    assert compute_cost("does-not-exist-2099", {"prompt_tokens": 1}) == 0.0


def test_set_strict_restores_previous_value():
    with set_strict(True):
        with set_strict(False):
            assert compute_cost("does-not-exist-2099", {"prompt_tokens": 1}) == 0.0
        # back to True after inner context exits
        with pytest.raises(MissingPriceError):
            compute_cost("does-not-exist-2099", {"prompt_tokens": 1})


def test_deprecated_model_warns_once():
    pricing._warned_deprecated.discard("gpt-4-0613")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_cost("gpt-4-0613", {"prompt_tokens": 100, "completion_tokens": 0})
        compute_cost("gpt-4-0613", {"prompt_tokens": 100, "completion_tokens": 0})
    deps = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deps) == 1
    assert "gpt-4.1-2025-04-14" in str(deps[0].message)


def test_snapshot_date_format():
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", SNAPSHOT_DATE)


def test_price_list_backward_compat():
    assert "gpt-4o-2024-11-20" in PRICE_LIST
    assert "fake-llm" in PRICE_LIST
    in_p, out_p = PRICE_LIST["gpt-4o-2024-11-20"]
    assert in_p > 0 and out_p > 0
    # Tuple of two floats — the old shape tests/test_shims.py expects.
    assert len(PRICE_LIST["fake-llm"]) == 2


def test_resolve_model_returns_none_for_unknown():
    assert resolve_model("not-a-real-thing") is None
    assert resolve_model("") is None
    assert is_known("gpt-4o") is True
    assert is_known("totally-fake") is False


# ---------------------------------------------------------------------------
# 14-18  Proposer 3: format / aggregate / budget / diff
# ---------------------------------------------------------------------------


def test_format_usd_default_precision():
    assert format_usd(0.001234) == "$0.0012"
    assert format_usd(0) == "$0.0000"


def test_format_usd_negative():
    assert format_usd(-0.001234) == "-$0.0012"


def _step(sid, kind, model, cost):
    return {"step_id": sid, "step_kind": kind, "model": model, "cost_usd": cost}


def test_aggregate_costs_per_model_and_kind():
    steps = [
        _step("s1", "llm_call", "gpt-4o-2024-11-20", 0.01),
        _step("s2", "llm_call", "gpt-4o-2024-11-20", 0.02),
        _step("s3", "tool_call", "fake-llm", 0.001),
    ]
    s = aggregate_costs(steps)
    assert s.n_steps == 3
    assert s.total_usd == pytest.approx(0.031, abs=1e-9)
    assert s.per_model["gpt-4o-2024-11-20"] == pytest.approx(0.03, abs=1e-9)
    assert s.per_model["fake-llm"] == pytest.approx(0.001, abs=1e-9)
    assert s.per_step_kind["llm_call"] == pytest.approx(0.03, abs=1e-9)
    assert s.per_step_kind["tool_call"] == pytest.approx(0.001, abs=1e-9)


def test_aggregate_costs_most_expensive_top5():
    steps = [_step(f"s{i}", "llm_call", "fake-llm", float(i)) for i in range(10)]
    s = aggregate_costs(steps)
    assert len(s.most_expensive) == 5
    # Top of list is the largest cost.
    assert s.most_expensive[0][1] == 9.0
    assert s.most_expensive[-1][1] == 5.0


def test_check_budget_under_and_over():
    steps = [_step("s1", "llm_call", "fake-llm", 0.05)]
    under = check_budget(steps, 1.0)
    assert isinstance(under, BudgetCheck)
    assert not under.over_budget
    assert under.fraction_used == pytest.approx(0.05, abs=1e-9)
    over = check_budget(steps, 0.01)
    assert over.over_budget
    assert over.remaining_usd < 0


def test_diff_costs_total_and_per_model_signed_b_minus_a():
    a = [_step("s1", "llm_call", "gpt-4o-2024-11-20", 0.01)]
    b = [_step("s1", "llm_call", "gpt-4o-2024-11-20", 0.03),
         _step("s2", "llm_call", "fake-llm", 0.001)]
    d = diff_costs(a, b)
    assert d["__total__"] == pytest.approx(0.021, abs=1e-9)
    assert d["gpt-4o-2024-11-20"] == pytest.approx(0.02, abs=1e-9)
    assert d["fake-llm"] == pytest.approx(0.001, abs=1e-9)


def test_diff_costs_handles_disjoint_models():
    a = [_step("s1", "llm_call", "gpt-4o-2024-11-20", 0.05)]
    b = [_step("s2", "llm_call", "claude-3-5-sonnet-20241022", 0.02)]
    d = diff_costs(a, b)
    assert d["gpt-4o-2024-11-20"] == pytest.approx(-0.05, abs=1e-9)
    assert d["claude-3-5-sonnet-20241022"] == pytest.approx(0.02, abs=1e-9)


# ---------------------------------------------------------------------------
# 19-20  Layer 2 refinements
# ---------------------------------------------------------------------------


def test_set_strict_is_context_manager():
    pre = pricing._strict
    with set_strict(True):
        assert pricing._strict is True
    assert pricing._strict == pre


def test_aggregate_costs_recomputes_from_usage_when_cost_missing():
    rec = {
        "step_id": "s1",
        "step_kind": "llm_call",
        "model": "fake-llm",
        "usage": {"prompt_tokens": 100, "completion_tokens": 100},
    }
    s = aggregate_costs([rec])
    expected = compute_cost("fake-llm", rec["usage"])
    assert s.total_usd == pytest.approx(expected, abs=1e-9)
    assert s.total_usd > 0


# ---------------------------------------------------------------------------
# 21-23  Layer 3 deepenings
# ---------------------------------------------------------------------------


def test_breakdown_to_dict_round_trips_via_json():
    b = compute_cost_breakdown(
        "gpt-4o-2024-11-20",
        {"prompt_tokens": 100, "completion_tokens": 50},
    )
    blob = json.dumps(b.to_dict())
    parsed = json.loads(blob)
    assert parsed["model"] == "gpt-4o-2024-11-20"
    assert parsed["total_usd"] == pytest.approx(b.total_usd, abs=1e-12)
    # Other dataclasses are serialisable too.
    assert json.dumps(TokenRates(0.001, 0.002).to_dict())
    assert json.dumps(BudgetCheck(1.0, 0.5, 0.5, False, 0.5).to_dict())
    s = aggregate_costs([{"cost_usd": 1.0, "step_kind": "llm_call", "model": "x",
                           "step_id": "s1"}])
    assert json.dumps(s.to_dict())


def test_snapshot_age_days_uses_today_param():
    snap = dt.date.fromisoformat(SNAPSHOT_DATE)
    future = snap + dt.timedelta(days=42)
    assert snapshot_age_days(today=future) == 42


def test_aggregate_costs_skips_records_without_recognisable_fields():
    steps = [{}, {"cost_usd": 1.0}, "not-a-mapping", {"step_kind": "x"}]
    s = aggregate_costs(steps)
    # 3 mapping records counted; total = 1.0 (only one carries cost_usd,
    # the others have no model/usage/cost_usd).
    assert s.n_steps == 3
    assert s.total_usd == pytest.approx(1.0, abs=1e-9)
