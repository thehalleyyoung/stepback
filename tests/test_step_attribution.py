"""Tests for Shapley-value step attribution (Step 84).

Exercises :func:`stepback.minimize.attribute_steps` and the
:meth:`stepback.replay.Trace.attribute_steps` convenience method.
"""
from __future__ import annotations

import pytest

from stepback import (
    Executor,
    PredicateNotTriggered,
    StepAttributionResult,
    attribute_steps,
    record,
    replay,
)
from stepback.testing import fake_llm, fake_tool, run_recorded_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(tmp_path):
    from stepback import RecorderKey

    key = RecorderKey.fresh()
    p = str(tmp_path / "trace.sb")
    with record(p, key=key) as rec:
        run_recorded_agent(rec)
    return p, key


def _executor():
    return Executor(llm=fake_llm, tool=fake_tool)


def _step_id(t, idx: int) -> str:
    return t.recorded_steps[idx]["step_id"]


# ---------------------------------------------------------------------------
# Basic contract tests
# ---------------------------------------------------------------------------


def test_attribute_steps_returns_step_attribution_result(tmp_path):
    """attribute_steps returns a StepAttributionResult with expected shape."""
    path, _ = _record(tmp_path)
    t = replay(path)

    # Predicate: always True (every step present in a full result).
    # But to test the zero-weights case we need a predicate that fires
    # on the empty coalition too — skip that here; use a step-specific one.
    step1_id = _step_id(t, 1)

    def predicate(result):
        return any(s.step_id == step1_id for s in result.steps)

    out = attribute_steps(t, predicate, executor=_executor())

    assert isinstance(out, StepAttributionResult)
    assert out.strategy_name == "step_shapley"
    assert len(out.steps) == len(t.recorded_steps)
    assert set(out.weights.keys()) == {s.step_id for s in out.steps}
    assert out.probes >= 1


def test_attribute_steps_lone_cause_gets_full_weight(tmp_path):
    """When a single step's presence solely triggers the predicate, it gets weight ~1.0."""
    path, _ = _record(tmp_path)
    t = replay(path)

    # The predicate fires only when step index 1 is visible.
    target_id = _step_id(t, 1)

    def predicate(result):
        return any(s.step_id == target_id for s in result.steps)

    out = attribute_steps(t, predicate, executor=_executor())

    # The target step should have weight close to 1.0.
    target_weight = out.weight_for(target_id)
    assert target_weight > 0.9, f"expected weight ~1.0, got {target_weight}"

    # All other steps should have weight ~0.0.
    for s in out.steps:
        if s.step_id != target_id:
            assert out.weight_for(s.step_id) < 0.1, (
                f"step {s.step_id} expected ~0.0, got {out.weight_for(s.step_id)}"
            )


def test_attribute_steps_joint_cause_splits_weight(tmp_path):
    """Two steps jointly required: each gets Shapley weight ~0.5."""
    path, _ = _record(tmp_path)
    t = replay(path)

    id_a = _step_id(t, 0)
    id_b = _step_id(t, 1)

    # Predicate: fires only when BOTH steps are present.
    def predicate(result):
        ids = {s.step_id for s in result.steps}
        return id_a in ids and id_b in ids

    out = attribute_steps(t, predicate, executor=_executor())

    w_a = out.weight_for(id_a)
    w_b = out.weight_for(id_b)

    # Both should have weight ~0.5; all others ~0.0.
    assert abs(w_a - 0.5) < 0.05, f"w_a={w_a}"
    assert abs(w_b - 0.5) < 0.05, f"w_b={w_b}"
    for s in out.steps:
        if s.step_id not in (id_a, id_b):
            assert out.weight_for(s.step_id) < 0.05


def test_attribute_steps_raises_predicate_not_triggered(tmp_path):
    """attribute_steps raises PredicateNotTriggered when predicate never fires."""
    path, _ = _record(tmp_path)
    t = replay(path)

    def never_fires(result):
        return False

    with pytest.raises(PredicateNotTriggered):
        attribute_steps(t, never_fires, executor=_executor())


def test_attribute_steps_empty_coalition_returns_zero_weights(tmp_path):
    """If predicate fires even on an empty result, all weights are 0.0."""
    path, _ = _record(tmp_path)
    t = replay(path)

    # Predicate that fires regardless of which steps are visible.
    def always_fires(result):
        return True

    out = attribute_steps(t, always_fires, executor=_executor())

    for sid, w in out.weights.items():
        assert w == 0.0, f"step {sid} should have weight 0.0, got {w}"


def test_attribute_steps_contributing_steps_property(tmp_path):
    """contributing_steps returns only steps with weight > 1e-9."""
    path, _ = _record(tmp_path)
    t = replay(path)

    target_id = _step_id(t, 2)

    def predicate(result):
        return any(s.step_id == target_id for s in result.steps)

    out = attribute_steps(t, predicate, executor=_executor())

    cs = out.contributing_steps
    assert all(out.weight_for(s.step_id) > 1e-9 for s in cs)
    assert any(s.step_id == target_id for s in cs)


def test_attribute_steps_trace_convenience_method(tmp_path):
    """Trace.attribute_steps delegates to attribute_steps correctly."""
    path, _ = _record(tmp_path)
    t = replay(path)

    target_id = _step_id(t, 3)

    def predicate(result):
        return any(s.step_id == target_id for s in result.steps)

    out = t.attribute_steps(predicate, executor=_executor())

    assert isinstance(out, StepAttributionResult)
    assert out.weight_for(target_id) > 0.9


def test_attribute_steps_probes_count_is_reasonable(tmp_path):
    """Probes for a lone-cause predicate should be <= 2^n (exact mode, n<=8)."""
    path, _ = _record(tmp_path)
    t = replay(path)

    target_id = _step_id(t, 0)

    def predicate(result):
        return any(s.step_id == target_id for s in result.steps)

    out = attribute_steps(t, predicate, executor=_executor())

    n = len(out.steps)
    if n <= 8:
        # Memoized exact mode: at most 2^n distinct coalitions evaluated.
        assert out.probes <= 2**n + 1
    else:
        # Sampled mode.
        assert out.probes > 0


def test_attribute_steps_sampled_mode_consistent(tmp_path):
    """Sampled mode (forced via permutations) still identifies the lone cause."""
    path, _ = _record(tmp_path)
    t = replay(path)

    target_id = _step_id(t, 1)

    def predicate(result):
        return any(s.step_id == target_id for s in result.steps)

    # Force sampled mode by passing a small permutations value.
    out = attribute_steps(
        t, predicate, executor=_executor(), permutations=128, rng_seed=42
    )

    # Even with sampled estimation, the lone-cause step should have the
    # highest weight among all steps.
    max_weight_id = max(out.weights, key=lambda sid: out.weights[sid])
    assert max_weight_id == target_id, (
        f"sampled mode: expected target={target_id} to dominate, "
        f"got {max_weight_id}. Weights: {out.weights}"
    )


def test_attribute_steps_weight_for_unknown_id_returns_zero(tmp_path):
    """weight_for returns 0.0 for a step_id not in the result."""
    path, _ = _record(tmp_path)
    t = replay(path)

    target_id = _step_id(t, 0)

    def predicate(result):
        return any(s.step_id == target_id for s in result.steps)

    out = attribute_steps(t, predicate, executor=_executor())
    assert out.weight_for("nonexistent:0") == 0.0


def test_attribute_steps_public_exports():
    """StepAttributionResult and attribute_steps are importable from stepback."""
    import stepback

    assert hasattr(stepback, "StepAttributionResult")
    assert hasattr(stepback, "attribute_steps")
    assert "StepAttributionResult" in stepback.__all__
    assert "attribute_steps" in stepback.__all__
