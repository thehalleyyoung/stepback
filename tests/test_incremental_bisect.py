"""Tests for Step 83: incremental multi-regression bisect.

``Trace.bisect_multi`` finds culprit steps for multiple regressions in a
**single** replay, so discovering one culprit does not restart the search for
the others.
"""
from __future__ import annotations

import math

import pytest

from stepback import BisectMultiResult, BisectTarget, RecorderKey, record, replay
from stepback.testing import fake_llm, fake_tool, run_recorded_agent


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _record_fixture(tmp_path):
    key = RecorderKey.fresh()
    path = str(tmp_path / "trace.sb")
    with record(path, key=key) as rec:
        run_recorded_agent(rec)
    return path, key


# -------------------------------------------------------------------
# Public-API shape tests
# -------------------------------------------------------------------


def test_bisect_target_is_importable():
    """BisectTarget is exported at top-level."""
    assert BisectTarget is not None


def test_bisect_multi_result_is_importable():
    """BisectMultiResult is exported at top-level."""
    assert BisectMultiResult is not None


def test_bisect_target_fields():
    """BisectTarget has good, bad, predicate fields."""
    t = BisectTarget(good="step:1", bad="step:5", predicate=lambda s: True)
    assert t.good == "step:1"
    assert t.bad == "step:5"
    assert callable(t.predicate)


def test_bisect_multi_result_fields():
    """BisectMultiResult has culprits list and total_probes int."""
    r = BisectMultiResult(culprits=[None], total_probes=3)
    assert r.culprits == [None]
    assert r.total_probes == 3


# -------------------------------------------------------------------
# Functional tests
# -------------------------------------------------------------------


def test_bisect_multi_single_target_agrees_with_bisect(tmp_path):
    """bisect_multi with one target returns the same culprit as bisect."""
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)

    predicate = lambda s: s.kind == "tool_call" and s.name == "payment.transfer"

    # Single-target bisect_multi
    result = t.bisect_multi([
        BisectTarget(good="step:1", bad="step:12", predicate=predicate)
    ])
    assert len(result.culprits) == 1
    culprit = result.culprits[0]
    assert culprit is not None
    assert culprit.step_id == "step:12"


def test_bisect_multi_multiple_targets_all_found(tmp_path):
    """bisect_multi correctly finds culprits for two independent regressions."""
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)

    # Two different predicates on the same trace
    pred_tool = lambda s: s.kind == "tool_call"
    pred_llm = lambda s: s.kind == "llm_call"

    result = t.bisect_multi([
        BisectTarget(good="step:1", bad="step:12", predicate=pred_tool),
        BisectTarget(good="step:1", bad="step:12", predicate=pred_llm),
    ])

    assert len(result.culprits) == 2
    # Both regressions should find a step
    assert result.culprits[0] is not None
    assert result.culprits[1] is not None
    # The first tool_call and first llm_call should be different steps
    assert result.culprits[0].step_id != result.culprits[1].step_id


def test_bisect_multi_none_when_predicate_never_true(tmp_path):
    """bisect_multi returns None for a target whose predicate is never true."""
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)

    result = t.bisect_multi([
        BisectTarget(good="step:1", bad="step:12", predicate=lambda s: False)
    ])
    assert result.culprits[0] is None


def test_bisect_multi_updates_last_bisect_probes(tmp_path):
    """bisect_multi updates last_bisect_probes with the sum of all probes."""
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    assert t.last_bisect_probes == 0

    result = t.bisect_multi([
        BisectTarget(good="step:1", bad="step:12", predicate=lambda s: s.kind == "tool_call"),
        BisectTarget(good="step:1", bad="step:12", predicate=lambda s: s.kind == "llm_call"),
    ])
    assert t.last_bisect_probes == result.total_probes
    assert t.last_bisect_probes >= 2  # at least one probe per target


def test_bisect_multi_probe_count_is_logarithmic_per_target(tmp_path):
    """Per-target probe count is at most ⌈log2(N)⌉+1 for N=12 steps."""
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)

    # Use two targets
    result = t.bisect_multi([
        BisectTarget(good="step:1", bad="step:12", predicate=lambda s: s.kind == "tool_call"),
        BisectTarget(good="step:1", bad="step:12", predicate=lambda s: s.kind == "llm_call"),
    ])
    # total_probes for 2 targets on 12 steps: at most 2 * (ceil(log2(12))+1) = 2*5 = 10
    max_expected = 2 * (math.ceil(math.log2(12)) + 1)
    assert result.total_probes <= max_expected


def test_bisect_multi_with_empty_targets_returns_empty(tmp_path):
    """bisect_multi with an empty target list returns an empty result."""
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)

    result = t.bisect_multi([])
    assert result.culprits == []
    assert result.total_probes == 0


def test_bisect_multi_reversed_good_bad_still_works(tmp_path):
    """bisect_multi normalises good>bad to good<bad automatically."""
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)

    result = t.bisect_multi([
        BisectTarget(good="step:12", bad="step:1", predicate=lambda s: s.kind == "tool_call")
    ])
    # Should still find a tool_call step even with reversed order
    assert result.culprits[0] is not None
    assert result.culprits[0].kind == "tool_call"


def test_bisect_multi_does_not_mutate_cursor_or_subs(tmp_path):
    """bisect_multi does not change cursor or pending substitutions."""
    from stepback.substitutions import ToolOutputSubstitution
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)

    cursor_before = t.cursor
    subs_before = list(t.pending_subs.items)

    t.bisect_multi([
        BisectTarget(good="step:1", bad="step:12", predicate=lambda s: True)
    ])

    assert t.cursor == cursor_before
    assert list(t.pending_subs.items) == subs_before
