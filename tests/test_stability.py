"""Tests for stepback.stability — statistical predicate stability metrics.

Uses the same deterministic 12-step payments fixture as test_minimize.py
so that tests are offline and reproducible.
"""
from __future__ import annotations

import warnings

import pytest

import stepback
from stepback import (
    Executor,
    replay,
    record,
    RecorderKey,
)
from stepback.stability import (
    FlakyClass,
    FlakyPredicateWarning,
    StabilityConfig,
    StabilityResult,
    _wilson_score_interval,
    measure_predicate_stability,
)
from stepback.substitutions import (
    SubstitutionSet,
    ToolOutputSubstitution,
    ModelSubstitution,
)
from stepback.testing import LOOKUP_FIXED_ROW, fake_llm, fake_tool, run_recorded_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(tmp_path):
    key = RecorderKey.fresh()
    p = str(tmp_path / "trace.sb")
    with record(p, key=key) as rec:
        run_recorded_agent(rec)
    return p, key


def _step_id(t, idx):
    return t.recorded_steps[idx]["step_id"]


def _executor():
    return Executor(llm=fake_llm, tool=fake_tool)


# Predicate that always fires (lookup step returns US country when substitution active).
def _country_us_predicate(result):
    try:
        return result.steps[1].outputs["result"]["country"] == "US"
    except (KeyError, IndexError, TypeError):
        return False


# Predicate that never fires under any substitution used in these tests.
def _never_predicate(result):
    return False


# ---------------------------------------------------------------------------
# Wilson score CI unit tests (pure math, no I/O)
# ---------------------------------------------------------------------------


class TestWilsonScoreInterval:
    def test_all_success(self):
        low, high = _wilson_score_interval(10, 10, 0.95)
        # All successes: CI should be (something, 1.0).
        assert 0.69 < low <= 1.0
        assert high == 1.0

    def test_all_failure(self):
        low, high = _wilson_score_interval(0, 10, 0.95)
        assert low == 0.0
        assert high < 0.31

    def test_half_success(self):
        low, high = _wilson_score_interval(5, 10, 0.95)
        # 50% fire rate, CI should straddle 0.5.
        assert low < 0.5
        assert high > 0.5

    def test_single_run_success(self):
        low, high = _wilson_score_interval(1, 1, 0.95)
        assert low >= 0.0
        assert high == 1.0

    def test_bounds_clamped_to_unit_interval(self):
        low, high = _wilson_score_interval(0, 1, 0.95)
        assert 0.0 <= low <= 1.0
        assert 0.0 <= high <= 1.0

    def test_90_percent_ci_narrower_than_99(self):
        low90, high90 = _wilson_score_interval(5, 20, 0.90)
        low99, high99 = _wilson_score_interval(5, 20, 0.99)
        assert (high90 - low90) < (high99 - low99)


# ---------------------------------------------------------------------------
# StabilityConfig validation
# ---------------------------------------------------------------------------


class TestStabilityConfig:
    def test_default_values(self):
        cfg = StabilityConfig()
        assert cfg.runs == 20
        assert cfg.ci_level == 0.95
        assert cfg.warn is True

    def test_runs_zero_raises(self):
        with pytest.raises(ValueError, match="runs"):
            StabilityConfig(runs=0)

    def test_runs_negative_raises(self):
        with pytest.raises(ValueError, match="runs"):
            StabilityConfig(runs=-1)

    def test_ci_level_zero_raises(self):
        with pytest.raises(ValueError, match="ci_level"):
            StabilityConfig(ci_level=0.0)

    def test_ci_level_one_raises(self):
        with pytest.raises(ValueError, match="ci_level"):
            StabilityConfig(ci_level=1.0)

    def test_custom_valid_config(self):
        cfg = StabilityConfig(runs=5, ci_level=0.90, warn=False)
        assert cfg.runs == 5
        assert cfg.ci_level == 0.90
        assert cfg.warn is False


# ---------------------------------------------------------------------------
# measure_predicate_stability — STABLE_TRUE
# ---------------------------------------------------------------------------


class TestMeasureStableTrue:
    """Deterministic predicate that fires on every run → STABLE_TRUE."""

    def test_stable_true_classification(self, tmp_path):
        path, _ = _record(tmp_path)
        t = replay(path)
        fix_step = _step_id(t, 1)
        subs = SubstitutionSet(
            items=[
                ToolOutputSubstitution(
                    at_step=fix_step,
                    tool_call_id=None,
                    fake_response=LOOKUP_FIXED_ROW,
                )
            ]
        )
        result = measure_predicate_stability(
            t, subs, _country_us_predicate,
            executor=_executor(),
            config=StabilityConfig(runs=5, warn=False),
        )
        assert result.flaky_class == FlakyClass.STABLE_TRUE
        assert result.fires == result.runs == 5
        assert result.fire_rate == 1.0

    def test_stable_true_no_warning(self, tmp_path):
        path, _ = _record(tmp_path)
        t = replay(path)
        fix_step = _step_id(t, 1)
        subs = SubstitutionSet(
            items=[
                ToolOutputSubstitution(
                    at_step=fix_step,
                    tool_call_id=None,
                    fake_response=LOOKUP_FIXED_ROW,
                )
            ]
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            measure_predicate_stability(
                t, subs, _country_us_predicate,
                executor=_executor(),
                config=StabilityConfig(runs=3, warn=True),
            )
        flaky_warns = [x for x in w if issubclass(x.category, FlakyPredicateWarning)]
        assert flaky_warns == [], "STABLE_TRUE must not emit FlakyPredicateWarning"

    def test_stable_true_ci_bounds(self, tmp_path):
        path, _ = _record(tmp_path)
        t = replay(path)
        fix_step = _step_id(t, 1)
        subs = SubstitutionSet(
            items=[
                ToolOutputSubstitution(
                    at_step=fix_step,
                    tool_call_id=None,
                    fake_response=LOOKUP_FIXED_ROW,
                )
            ]
        )
        result = measure_predicate_stability(
            t, subs, _country_us_predicate,
            executor=_executor(),
            config=StabilityConfig(runs=10, warn=False),
        )
        assert 0.0 <= result.ci_low <= result.ci_high <= 1.0
        assert result.ci_level == 0.95


# ---------------------------------------------------------------------------
# measure_predicate_stability — STABLE_FALSE
# ---------------------------------------------------------------------------


class TestMeasureStableFalse:
    """Never-firing predicate → STABLE_FALSE."""

    def test_stable_false_classification(self, tmp_path):
        path, _ = _record(tmp_path)
        t = replay(path)
        subs = SubstitutionSet(items=[])
        result = measure_predicate_stability(
            t, subs, _never_predicate,
            executor=_executor(),
            config=StabilityConfig(runs=5, warn=False),
        )
        assert result.flaky_class == FlakyClass.STABLE_FALSE
        assert result.fires == 0
        assert result.fire_rate == 0.0

    def test_stable_false_no_warning(self, tmp_path):
        path, _ = _record(tmp_path)
        t = replay(path)
        subs = SubstitutionSet(items=[])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            measure_predicate_stability(
                t, subs, _never_predicate,
                executor=_executor(),
                config=StabilityConfig(runs=3, warn=True),
            )
        flaky_warns = [x for x in w if issubclass(x.category, FlakyPredicateWarning)]
        assert flaky_warns == [], "STABLE_FALSE must not emit FlakyPredicateWarning"


# ---------------------------------------------------------------------------
# measure_predicate_stability — FLAKY
# ---------------------------------------------------------------------------


class TestMeasureFlaky:
    """Predicate with a counter-based flip pattern → FLAKY."""

    def test_flaky_classification(self, tmp_path):
        path, _ = _record(tmp_path)
        t = replay(path)
        fix_step = _step_id(t, 1)
        subs = SubstitutionSet(
            items=[
                ToolOutputSubstitution(
                    at_step=fix_step,
                    tool_call_id=None,
                    fake_response=LOOKUP_FIXED_ROW,
                )
            ]
        )
        # Predicate fires on even-numbered calls only (alternating).
        call_count = [0]

        def alternating_predicate(result):
            call_count[0] += 1
            return (call_count[0] % 2) == 0

        result = measure_predicate_stability(
            t, subs, alternating_predicate,
            executor=_executor(),
            config=StabilityConfig(runs=6, warn=False),
        )
        assert result.flaky_class == FlakyClass.FLAKY
        assert result.fires == 3  # fires on calls 2, 4, 6
        assert result.fire_rate == pytest.approx(0.5, abs=1e-9)

    def test_flaky_emits_warning(self, tmp_path):
        path, _ = _record(tmp_path)
        t = replay(path)
        fix_step = _step_id(t, 1)
        subs = SubstitutionSet(
            items=[
                ToolOutputSubstitution(
                    at_step=fix_step,
                    tool_call_id=None,
                    fake_response=LOOKUP_FIXED_ROW,
                )
            ]
        )
        call_count = [0]

        def alternating_predicate(result):
            call_count[0] += 1
            return (call_count[0] % 2) == 0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            measure_predicate_stability(
                t, subs, alternating_predicate,
                executor=_executor(),
                config=StabilityConfig(runs=4, warn=True),
            )
        flaky_warns = [x for x in w if issubclass(x.category, FlakyPredicateWarning)]
        assert len(flaky_warns) == 1
        msg = str(flaky_warns[0].message)
        assert "flaky" in msg.lower() or "fire" in msg.lower()

    def test_flaky_no_warning_when_warn_false(self, tmp_path):
        path, _ = _record(tmp_path)
        t = replay(path)
        fix_step = _step_id(t, 1)
        subs = SubstitutionSet(
            items=[
                ToolOutputSubstitution(
                    at_step=fix_step,
                    tool_call_id=None,
                    fake_response=LOOKUP_FIXED_ROW,
                )
            ]
        )
        call_count = [0]

        def alternating_predicate(result):
            call_count[0] += 1
            return (call_count[0] % 2) == 0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            measure_predicate_stability(
                t, subs, alternating_predicate,
                executor=_executor(),
                config=StabilityConfig(runs=4, warn=False),
            )
        flaky_warns = [x for x in w if issubclass(x.category, FlakyPredicateWarning)]
        assert flaky_warns == []

    def test_flaky_ci_bounds_valid(self, tmp_path):
        path, _ = _record(tmp_path)
        t = replay(path)
        fix_step = _step_id(t, 1)
        subs = SubstitutionSet(
            items=[
                ToolOutputSubstitution(
                    at_step=fix_step,
                    tool_call_id=None,
                    fake_response=LOOKUP_FIXED_ROW,
                )
            ]
        )
        call_count = [0]

        def alternating_predicate(result):
            call_count[0] += 1
            return (call_count[0] % 2) == 0

        result = measure_predicate_stability(
            t, subs, alternating_predicate,
            executor=_executor(),
            config=StabilityConfig(runs=10, ci_level=0.90, warn=False),
        )
        # CI must be a valid interval within [0, 1].
        assert 0.0 <= result.ci_low < result.ci_high <= 1.0
        assert result.ci_level == 0.90
        # For 50% fire rate the CI should straddle 0.5.
        assert result.ci_low < 0.5 < result.ci_high


# ---------------------------------------------------------------------------
# StabilityResult fields
# ---------------------------------------------------------------------------


class TestStabilityResultFields:
    def test_result_has_expected_fields(self, tmp_path):
        path, _ = _record(tmp_path)
        t = replay(path)
        subs = SubstitutionSet(items=[])
        result = measure_predicate_stability(
            t, subs, _never_predicate,
            executor=_executor(),
            config=StabilityConfig(runs=3, warn=False),
        )
        assert isinstance(result, StabilityResult)
        assert result.runs == 3
        assert isinstance(result.fires, int)
        assert isinstance(result.fire_rate, float)
        assert isinstance(result.ci_low, float)
        assert isinstance(result.ci_high, float)
        assert isinstance(result.ci_level, float)
        assert isinstance(result.flaky_class, FlakyClass)

    def test_single_run_stable_true(self, tmp_path):
        path, _ = _record(tmp_path)
        t = replay(path)
        fix_step = _step_id(t, 1)
        subs = SubstitutionSet(
            items=[
                ToolOutputSubstitution(
                    at_step=fix_step,
                    tool_call_id=None,
                    fake_response=LOOKUP_FIXED_ROW,
                )
            ]
        )
        result = measure_predicate_stability(
            t, subs, _country_us_predicate,
            executor=_executor(),
            config=StabilityConfig(runs=1, warn=False),
        )
        assert result.flaky_class == FlakyClass.STABLE_TRUE
        assert result.fires == 1
        assert result.runs == 1
        assert result.fire_rate == 1.0

    def test_single_run_stable_false(self, tmp_path):
        path, _ = _record(tmp_path)
        t = replay(path)
        subs = SubstitutionSet(items=[])
        result = measure_predicate_stability(
            t, subs, _never_predicate,
            executor=_executor(),
            config=StabilityConfig(runs=1, warn=False),
        )
        assert result.flaky_class == FlakyClass.STABLE_FALSE
        assert result.fires == 0


# ---------------------------------------------------------------------------
# Public API accessibility
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_all_symbols_accessible_from_stepback(self):
        assert hasattr(stepback, "FlakyClass")
        assert hasattr(stepback, "FlakyPredicateWarning")
        assert hasattr(stepback, "StabilityConfig")
        assert hasattr(stepback, "StabilityResult")
        assert hasattr(stepback, "measure_predicate_stability")

    def test_flaky_class_in_all(self):
        assert "FlakyClass" in stepback.__all__
        assert "FlakyPredicateWarning" in stepback.__all__
        assert "StabilityConfig" in stepback.__all__
        assert "StabilityResult" in stepback.__all__
        assert "measure_predicate_stability" in stepback.__all__

    def test_flaky_warning_is_user_warning_subclass(self):
        assert issubclass(FlakyPredicateWarning, UserWarning)

    def test_flaky_class_enum_members(self):
        assert FlakyClass.STABLE_TRUE is not None
        assert FlakyClass.STABLE_FALSE is not None
        assert FlakyClass.FLAKY is not None
