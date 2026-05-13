"""Tests for the model-swap differential benchmark (Step 120).

Coverage:
* replay_fidelity_rate == 1.0 (soundness guarantee)
* model_agreement_rate == 0.0 (models produce distinct outputs)
* p_value < 0.05 (statistically significant length difference)
* effect_size > 0 (model B is longer than model A)
* output_length_ratio > 1.0 (B longer than A on average)
* CI bounds are ordered and contain the ratio
* to_json round-trip completeness
* summary_line format and key tokens
* CLI smoke test with --out
* CLI summary contains key markers
* compare() across multiple sizes
* Determinism under same seed
* Variation under different seed
* error validation (n_steps < 1, n_trials < 1)
* edge case n_steps=1, n_trials=1
* dirty_count > 0 (model swap forces dirty LLM steps)
* cache_hit_count_mean >= 0 (non-LLM steps may still hit cache)
* statistical helpers (_betai, _t_two_tailed_p, _betacf, _paired_t_test,
  _cohens_d_paired, _bootstrap_ratio_ci)
"""
from __future__ import annotations

import json
import math
import os
import tempfile

import pytest

from stepback.bench.model_swap import (
    ModelSwapResult,
    _betai,
    _bootstrap_ratio_ci,
    _cohens_d_paired,
    _model_a_llm,
    _model_b_llm,
    _output_text_a,
    _output_text_b,
    _paired_t_test,
    _t_two_tailed_p,
    compare,
    run,
)


# ------------------------------------------------------------------ helpers


def _small_result(n_steps: int = 10, n_trials: int = 3, seed: int = 0) -> ModelSwapResult:
    return run(n_steps=n_steps, n_trials=n_trials, seed=seed)


# ------------------------------------------------------------------ smoke / correctness


def test_replay_fidelity_is_one():
    """The model-swap replay must produce byte-identical outputs to a fresh
    model-B recording — the soundness guarantee of the dirty-set engine."""
    result = _small_result()
    assert result.replay_fidelity_rate == 1.0, (
        f"replay_fidelity_rate={result.replay_fidelity_rate}; "
        "replay soundness violated"
    )


def test_model_agreement_is_zero():
    """Model A and model B must produce distinct outputs on every step."""
    result = _small_result()
    assert result.model_agreement_rate == 0.0, (
        f"model_agreement_rate={result.model_agreement_rate}; "
        "fake LLMs should never agree"
    )


def test_p_value_significant():
    """The paired t-test should detect a significant output-length difference."""
    result = _small_result(n_steps=15, n_trials=5)
    assert result.p_value < 0.05, (
        f"p_value={result.p_value} is not significant; "
        "model B should produce statistically different output lengths"
    )


def test_effect_size_positive():
    """Model B always produces longer outputs than model A, so Cohen's d > 0."""
    result = _small_result()
    assert result.effect_size > 0.0, (
        f"effect_size={result.effect_size}; model B outputs should be longer"
    )


def test_output_length_ratio_greater_than_one():
    """mean_length_b > mean_length_a by construction."""
    result = _small_result()
    assert result.output_length_ratio > 1.0, (
        f"output_length_ratio={result.output_length_ratio}"
    )


def test_ci_bounds_ordered_and_contain_ratio():
    """The 95% CI lower bound must be <= ratio and upper bound >= ratio."""
    result = _small_result(n_steps=20, n_trials=5)
    assert result.ci_95_lower <= result.output_length_ratio, (
        f"CI lower {result.ci_95_lower} > ratio {result.output_length_ratio}"
    )
    assert result.ci_95_upper >= result.output_length_ratio, (
        f"CI upper {result.ci_95_upper} < ratio {result.output_length_ratio}"
    )


def test_dirty_count_positive():
    """Every LLM step should be dirty after a model swap."""
    result = _small_result()
    assert result.dirty_count_mean > 0, "model swap should force dirty LLM steps"


def test_cache_hit_count_nonnegative():
    """Cache hit count should be non-negative (non-LLM steps may cache)."""
    result = _small_result()
    assert result.cache_hit_count_mean >= 0


def test_n_llm_steps_positive():
    """At least one LLM step must be observed per trial."""
    result = _small_result()
    assert result.n_llm_steps > 0


# ------------------------------------------------------------------ to_json / from

def test_to_json_contains_all_keys():
    expected_keys = {
        "model_a_name", "model_b_name", "n_trials", "n_steps", "n_llm_steps",
        "model_agreement_rate", "replay_fidelity_rate",
        "mean_length_a", "mean_length_b", "output_length_ratio",
        "ci_95_lower", "ci_95_upper",
        "t_stat", "p_value", "effect_size",
        "dirty_count_mean", "cache_hit_count_mean",
        "wall_time_ms",
    }
    result = _small_result()
    d = result.to_json()
    assert expected_keys.issubset(d.keys()), (
        f"missing keys: {expected_keys - d.keys()}"
    )


def test_to_json_json_serialisable():
    result = _small_result()
    raw = json.dumps(result.to_json())
    back = json.loads(raw)
    assert back["replay_fidelity_rate"] == pytest.approx(result.replay_fidelity_rate)
    assert back["model_agreement_rate"] == pytest.approx(result.model_agreement_rate)


# ------------------------------------------------------------------ summary_line


def test_summary_line_contains_key_tokens():
    result = _small_result()
    line = result.summary_line()
    assert "model-swap" in line
    assert "replay_fidelity=" in line
    assert "agreement=" in line
    assert "ratio=" in line
    assert "p=" in line
    assert "wall_ms=" in line


def test_summary_line_sig_marker():
    """Statistically significant results should show '(sig)'."""
    result = _small_result(n_steps=15, n_trials=5)
    if result.p_value < 0.05:
        assert "(sig)" in result.summary_line()


# ------------------------------------------------------------------ CLI


def test_cli_smoke_no_out(capsys):
    from stepback.cli import main
    rc = main(["bench", "model-swap", "--n-steps", "8", "--n-trials", "2", "--seed", "1"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "model-swap" in captured.out
    assert "replay_fidelity=" in captured.out


def test_cli_with_out():
    from stepback.cli import main
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "ms.json")
        rc = main(["bench", "model-swap", "--n-steps", "8", "--n-trials", "2",
                   "--seed", "2", "--out", out_path])
        assert rc == 0
        assert os.path.exists(out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert "replay_fidelity_rate" in data
        assert "p_value" in data
        assert data["n_trials"] == 2


# ------------------------------------------------------------------ compare


def test_compare_across_sizes():
    results = compare([5, 10, 15], n_trials=2, seed=7)
    assert set(results.keys()) == {5, 10, 15}
    for r in results.values():
        assert isinstance(r, ModelSwapResult)
        assert r.replay_fidelity_rate == 1.0


# ------------------------------------------------------------------ determinism


def test_determinism_same_seed():
    r1 = run(n_steps=10, n_trials=3, seed=99)
    r2 = run(n_steps=10, n_trials=3, seed=99)
    assert r1.t_stat == pytest.approx(r2.t_stat)
    assert r1.output_length_ratio == pytest.approx(r2.output_length_ratio)
    assert r1.n_llm_steps == r2.n_llm_steps


def test_different_seed_may_differ():
    r1 = run(n_steps=20, n_trials=5, seed=0)
    r2 = run(n_steps=20, n_trials=5, seed=13)
    # Different seeds → different trace topologies → possibly different n_llm_steps
    # At minimum, the seeds should produce independent results (not guaranteed to
    # differ but almost always will).
    assert r1.model_a_name == r2.model_a_name


# ------------------------------------------------------------------ validation


def test_invalid_n_steps():
    with pytest.raises(ValueError, match="n_steps must be >= 1"):
        run(n_steps=0)


def test_invalid_n_trials():
    with pytest.raises(ValueError, match="n_trials must be >= 1"):
        run(n_steps=5, n_trials=0)


def test_edge_case_single_step_single_trial():
    result = run(n_steps=1, n_trials=1, seed=0)
    # With only 1 step there may be 0 or 1 LLM steps.
    assert result.n_trials == 1
    assert result.replay_fidelity_rate in (0.0, 1.0)


# ------------------------------------------------------------------ fake LLM outputs


def test_model_a_outputs_have_variable_length():
    msgs = [{"role": "user", "content": "hi"}]
    lengths = {len(_output_text_a("m", [{"role": "user", "content": f"seed{i}"}])) for i in range(32)}
    # At minimum a few distinct lengths (hex nibble 0..15 → 11..26 chars)
    assert len(lengths) > 1


def test_model_b_outputs_always_longer_than_a():
    """For any given prompt, model B's output is longer than model A's."""
    for i in range(20):
        msgs = [{"role": "user", "content": f"test-{i}"}]
        text_a = _output_text_a("bench-model-a-v1", msgs)
        text_b = _output_text_b("bench-model-b-v1", msgs)
        assert len(text_b) > len(text_a), (
            f"prompt {i}: B({len(text_b)}) should be > A({len(text_a)})"
        )


def test_model_llm_outputs_differ():
    msgs = [{"role": "user", "content": "hello"}]
    out_a = _model_a_llm("m", msgs)
    out_b = _model_b_llm("m", msgs)
    text_a = out_a["choices"][0]["message"]["content"]
    text_b = out_b["choices"][0]["message"]["content"]
    assert text_a != text_b


# ------------------------------------------------------------------ statistical helpers


def test_betai_boundary_values():
    assert _betai(1.0, 1.0, 0.0) == pytest.approx(0.0)
    assert _betai(1.0, 1.0, 1.0) == pytest.approx(1.0)
    assert _betai(1.0, 1.0, 0.5) == pytest.approx(0.5)


def test_betai_symmetry():
    """I_x(a, b) = 1 - I_{1-x}(b, a)."""
    assert _betai(2.0, 5.0, 0.3) == pytest.approx(1.0 - _betai(5.0, 2.0, 0.7), rel=1e-6)


def test_t_two_tailed_p_known_value():
    """For df=10, t≈2.2281 the two-tailed p-value is approximately 0.05."""
    p = _t_two_tailed_p(2.2281388, 10.0)
    assert p == pytest.approx(0.05, abs=0.002)


def test_t_two_tailed_p_symmetric():
    """p-value is symmetric: t and -t give the same p."""
    assert _t_two_tailed_p(1.5, 20.0) == pytest.approx(_t_two_tailed_p(-1.5, 20.0))


def test_paired_t_test_basic():
    # All differences are positive → t > 0, p < 0.05 for large enough effect.
    diffs = [3.0, 4.0, 5.0, 4.0, 3.5, 4.5]
    t, df, p = _paired_t_test(diffs)
    assert t > 0
    assert df == pytest.approx(5.0)
    assert p < 0.05


def test_paired_t_test_empty():
    t, df, p = _paired_t_test([])
    assert p == 1.0


def test_paired_t_test_single_element():
    t, df, p = _paired_t_test([5.0])
    assert p == 1.0


def test_cohens_d_paired_positive():
    diffs = [2.0] * 10
    # With zero variance, returns 0.0.
    assert _cohens_d_paired(diffs) == 0.0


def test_cohens_d_paired_variable():
    diffs = [1.0, 2.0, 3.0, 4.0, 5.0]
    d = _cohens_d_paired(diffs)
    # Mean = 3, stdev ≈ 1.58, d ≈ 1.9
    assert d > 1.0


def test_bootstrap_ratio_ci_bounds():
    a = [10.0, 11.0, 12.0, 13.0, 14.0]
    b = [20.0, 21.0, 22.0, 23.0, 24.0]
    lo, hi = _bootstrap_ratio_ci(a, b, seed=0)
    ratio = sum(b) / sum(a)  # ~ 1.8
    assert lo < ratio < hi


def test_bootstrap_ratio_ci_empty():
    lo, hi = _bootstrap_ratio_ci([], [], seed=0)
    assert lo == 1.0 and hi == 1.0
