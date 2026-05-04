"""End-to-end tests for the ddmin substitution-minimisation primitive.

Drives the same 12-step deterministic payments fixture used by
``test_e2e_replay``, builds a noisy ``SubstitutionSet`` of 6
substitutions in which only ONE actually flips the predicate, and
asserts that ``Trace.minimize`` correctly shrinks it to a 1-minimal
single-element set.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from stepback import (
    Executor,
    MinimizationResult,
    PredicateNotTriggered,
    RecorderKey,
    ddmin_substitutions,
    record,
    replay,
)
from stepback.substitutions import (
    ModelSubstitution,
    PromptSubstitution,
    SubstitutionSet,
    ToolOutputSubstitution,
)
from stepback.testing import LOOKUP_FIXED_ROW, fake_llm, fake_tool, run_recorded_agent


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


# --------------------------------------------------------- core ddmin


def test_ddmin_isolates_single_responsible_substitution(tmp_path):
    """Only one of six subs is the cause; ddmin must isolate it."""
    path, _ = _record(tmp_path)
    t = replay(path)

    # Step 1 (zero-indexed) is the lookup_customer tool call. Pinning it
    # to the US row removes the GB99 IBAN from the rest of the trace.
    fix_step = _step_id(t, 1)
    payment_step = _step_id(t, 11)

    noisy = SubstitutionSet(
        items=[
            # The one that matters:
            ToolOutputSubstitution(
                at_step=fix_step,
                tool_call_id=None,
                fake_response=LOOKUP_FIXED_ROW,
            ),
            # Five harmless decoys (each at_step exists, but does not
            # remove "GB99" from the recorded trace's downstream steps).
            ModelSubstitution(at_step=_step_id(t, 0), new_model_id="gpt-4o-mini-2024-07-18"),
            ModelSubstitution(at_step=_step_id(t, 2), new_model_id="gpt-4o-mini-2024-07-18"),
            ModelSubstitution(at_step=_step_id(t, 4), new_model_id="gpt-4o-mini-2024-07-18"),
            ModelSubstitution(at_step=_step_id(t, 6), new_model_id="gpt-4o-mini-2024-07-18"),
            PromptSubstitution(
                at_step=_step_id(t, 8),
                new_messages=[{"role": "system", "content": "noop decoy"}],
            ),
        ]
    )

    # Predicate: the lookup step now resolves to a US customer (with
    # the fix-substitution active) instead of the recorded UK one.
    def predicate(result):
        try:
            return result.steps[1].outputs["result"]["country"] == "US"
        except (KeyError, IndexError, TypeError):
            return False

    out = t.minimize(noisy, predicate, executor=_executor())

    assert isinstance(out, MinimizationResult)
    # 1-minimal => exactly one substitution remains, and it's the
    # ToolOutputSubstitution at the lookup step.
    assert len(out.minimal) == 1, [type(x).__name__ for x in out.minimal]
    survivor = out.minimal[0]
    assert isinstance(survivor, ToolOutputSubstitution)
    assert survivor.at_step == fix_step
    # The other 5 must be in `removed`.
    assert len(out.removed) == 5
    # The reported probe count must be modest (ddmin on n=6 is tiny).
    assert 0 < out.probes <= 30, out.probes
    # Final replay confirms predicate still holds under the minimal set.
    assert out.final_result is not None
    assert predicate(out.final_result)


def test_ddmin_raises_when_full_set_does_not_trigger(tmp_path):
    """Minimising a non-triggering set is meaningless and must error."""
    path, _ = _record(tmp_path)
    t = replay(path)

    # An entirely no-op set (decoys only).
    noisy = SubstitutionSet(
        items=[
            ModelSubstitution(at_step=_step_id(t, 0), new_model_id="gpt-4o-mini-2024-07-18"),
            ModelSubstitution(at_step=_step_id(t, 2), new_model_id="gpt-4o-mini-2024-07-18"),
        ]
    )
    # Predicate that the empty / decoy-only set will not satisfy:
    # the recorded lookup row is the UK row.
    def predicate(result):
        try:
            return result.steps[1].outputs["result"]["country"] == "US"
        except (KeyError, IndexError, TypeError):
            return False

    with pytest.raises(PredicateNotTriggered):
        ddmin_substitutions(t, noisy, predicate, executor=_executor())


def test_ddmin_returns_empty_when_recorded_trace_already_triggers(tmp_path):
    """If the empty subset already triggers, the answer is the empty set."""
    path, _ = _record(tmp_path)
    t = replay(path)

    noisy = SubstitutionSet(
        items=[
            ToolOutputSubstitution(
                at_step=_step_id(t, 1),
                tool_call_id=None,
                fake_response=LOOKUP_FIXED_ROW,
            ),
            ModelSubstitution(
                at_step=_step_id(t, 0),
                new_model_id="gpt-4o-mini-2024-07-18",
            ),
        ]
    )
    # Trivially-true predicate: the recorded trace has >=1 step.
    out = t.minimize(noisy, predicate=lambda r: len(r.steps) >= 1, executor=_executor())
    assert out.minimal == []
    assert len(out.removed) == 2


def test_ddmin_probe_count_is_bounded_by_input_size(tmp_path):
    """ddmin is theoretically O(n^2) probes; assert a sane upper bound."""
    path, _ = _record(tmp_path)
    t = replay(path)

    fix_step = _step_id(t, 1)
    items = [
        ToolOutputSubstitution(
            at_step=fix_step, tool_call_id=None, fake_response=LOOKUP_FIXED_ROW,
        )
    ]
    # Pad with 9 decoys.
    for i in [0, 2, 3, 4, 5, 6, 7, 8, 9]:
        items.append(
            ModelSubstitution(at_step=_step_id(t, i), new_model_id="gpt-4o-mini-2024-07-18")
        )
    noisy = SubstitutionSet(items=items)

    def predicate(result):
        try:
            return result.steps[1].outputs["result"]["country"] == "US"
        except (KeyError, IndexError, TypeError):
            return False

    out = t.minimize(noisy, predicate, executor=_executor())
    assert len(out.minimal) == 1
    # n=10 => ddmin upper bound is well under 100 probes.
    assert out.probes < 100, out.probes


# ----------------------------------------------------------- CLI


def test_cli_minimize_emits_json_summary(tmp_path):
    """`stepback minimize` emits a usable JSON report."""
    path, _ = _record(tmp_path)
    t = replay(path)
    fix_step = _step_id(t, 1)
    decoy_step = _step_id(t, 0)

    cmd = [
        sys.executable, "-m", "stepback.cli", "minimize", path,
        "--substitute", f"tool_output@{fix_step}=:inline:" + json.dumps(LOOKUP_FIXED_ROW),
        "--substitute", f"model@{decoy_step}=gpt-4o-mini-2024-07-18",
        "--predicate",
            "'acme-us' in str(result.steps[1].outputs)",
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr
    payload = json.loads(cp.stdout)
    assert payload["minimal_count"] == 1
    assert payload["removed_count"] == 1
    assert payload["minimal"][0]["kind"] == "ToolOutputSubstitution"
    assert payload["minimal"][0]["at_step"] == fix_step
    assert payload["probes"] >= 1


def test_cli_minimize_errors_on_non_triggering_set(tmp_path):
    """If the full set doesn't trigger, CLI exits non-zero with a message."""
    path, _ = _record(tmp_path)
    t = replay(path)
    decoy_step = _step_id(t, 0)
    cmd = [
        sys.executable, "-m", "stepback.cli", "minimize", path,
        "--substitute", f"model@{decoy_step}=gpt-4o-mini-2024-07-18",
        "--predicate",
            "'acme-us' in str(result.steps[1].outputs)",
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    assert cp.returncode == 4
    assert "predicate does not fire" in cp.stderr


# ================================================================
# Strategy-pluggable, budget, multi-witness, and Shapley tests
# (Layer-3 MoA round expansion of minimize.py)
# ================================================================
from stepback import (
    BinaryHalvingStrategy,
    BruteForceStrategy,
    BudgetExhausted,
    DDMinStrategy,
    LinearShrinkStrategy,
    MinimizeOptions,
    ShapleyAttributionStrategy,
    attribute_substitutions,
    find_all_minimal,
    minimize_substitutions,
)


def _build_noisy_set(t):
    """Standard 6-noisy-subs fixture used by several new tests."""
    fix_step = _step_id(t, 1)
    return SubstitutionSet(items=[
        ToolOutputSubstitution(
            at_step=fix_step, tool_call_id=None, fake_response=LOOKUP_FIXED_ROW,
        ),
        ModelSubstitution(at_step=_step_id(t, 0), new_model_id="gpt-4o-mini-2024-07-18"),
        ModelSubstitution(at_step=_step_id(t, 2), new_model_id="gpt-4o-mini-2024-07-18"),
        ModelSubstitution(at_step=_step_id(t, 4), new_model_id="gpt-4o-mini-2024-07-18"),
        ModelSubstitution(at_step=_step_id(t, 6), new_model_id="gpt-4o-mini-2024-07-18"),
        PromptSubstitution(
            at_step=_step_id(t, 8),
            new_messages=[{"role": "system", "content": "noop decoy"}],
        ),
    ]), fix_step


def _country_us_predicate(result):
    try:
        return result.steps[1].outputs["result"]["country"] == "US"
    except (KeyError, IndexError, TypeError):
        return False


@pytest.mark.parametrize(
    "strategy_cls",
    [DDMinStrategy, LinearShrinkStrategy, BinaryHalvingStrategy, BruteForceStrategy],
)
def test_strategy_pluggable_returns_same_minimal(tmp_path, strategy_cls):
    """ddmin / linear / binary / brute all isolate the lone causal sub."""
    path, _ = _record(tmp_path)
    t = replay(path)
    noisy, fix_step = _build_noisy_set(t)
    out = minimize_substitutions(
        t, noisy, _country_us_predicate,
        options=MinimizeOptions(strategy=strategy_cls()),
        executor=_executor(),
    )
    assert out.strategy_name == strategy_cls.name
    assert len(out.minimal) == 1, [type(x).__name__ for x in out.minimal]
    survivor = out.minimal[0]
    assert isinstance(survivor, ToolOutputSubstitution)
    assert survivor.at_step == fix_step
    assert len(out.removed) == 5
    assert out.final_result is not None
    assert _country_us_predicate(out.final_result)


def test_brute_force_refuses_oversized_input(tmp_path):
    path, _ = _record(tmp_path)
    t = replay(path)
    # 10 items > default max_n=8.
    items = []
    fix_step = _step_id(t, 1)
    items.append(ToolOutputSubstitution(
        at_step=fix_step, tool_call_id=None, fake_response=LOOKUP_FIXED_ROW,
    ))
    for i in [0, 2, 3, 4, 5, 6, 7, 8, 9]:
        items.append(ModelSubstitution(at_step=_step_id(t, i), new_model_id="gpt-4o-mini-2024-07-18"))
    noisy = SubstitutionSet(items=items)
    with pytest.raises(ValueError, match="refuses"):
        minimize_substitutions(
            t, noisy, _country_us_predicate,
            options=MinimizeOptions(strategy=BruteForceStrategy()),
            executor=_executor(),
        )


def test_oracle_cache_hits_recorded(tmp_path):
    """The oracle cache de-duplicates repeated subset queries."""
    from stepback.minimize import _OracleCache

    path, _ = _record(tmp_path)
    t = replay(path)
    noisy, _ = _build_noisy_set(t)
    items = list(noisy.items)

    cache = _OracleCache(t, _country_us_predicate, _executor(), MinimizeOptions())
    # Evaluate the same subset twice: second call must be a cache hit.
    cache.evaluate(items)
    cache.evaluate(items)
    cache.evaluate(items[:3])
    cache.evaluate(items[:3])
    assert cache.probes == 2, cache.probes
    assert cache.cache_hits == 2, cache.cache_hits

    # And an end-to-end run reports cache_hits as a non-negative int.
    out = minimize_substitutions(
        t, noisy, _country_us_predicate,
        options=MinimizeOptions(strategy=BinaryHalvingStrategy()),
        executor=_executor(),
    )
    assert out.cache_hits >= 0
    assert isinstance(out.cache_hits, int)


def test_minimize_options_probe_budget_raises_with_partial(tmp_path):
    path, _ = _record(tmp_path)
    t = replay(path)
    noisy, _ = _build_noisy_set(t)
    with pytest.raises(BudgetExhausted) as excinfo:
        minimize_substitutions(
            t, noisy, _country_us_predicate,
            options=MinimizeOptions(probe_budget=1),
            executor=_executor(),
        )
    partial = excinfo.value.partial
    assert partial is not None
    assert partial.probes <= 1
    assert partial.strategy_name == "ddmin"


def test_minimize_options_progress_callback(tmp_path):
    path, _ = _record(tmp_path)
    t = replay(path)
    noisy, _ = _build_noisy_set(t)
    events = []

    def progress(probes_so_far, current_size):
        events.append((probes_so_far, current_size))

    out = minimize_substitutions(
        t, noisy, _country_us_predicate,
        options=MinimizeOptions(progress=progress),
        executor=_executor(),
    )
    assert len(events) == out.probes
    # The probe counter must monotonically increase.
    assert [e[0] for e in events] == sorted(e[0] for e in events)


def test_find_all_minimal_returns_at_least_one_witness(tmp_path):
    path, _ = _record(tmp_path)
    t = replay(path)
    noisy, fix_step = _build_noisy_set(t)
    witnesses = find_all_minimal(
        t, noisy, _country_us_predicate,
        max_witnesses=4,
        executor=_executor(),
    )
    assert len(witnesses) >= 1
    # First witness matches what plain ddmin would produce.
    first = witnesses[0]
    assert len(first.minimal) == 1
    assert isinstance(first.minimal[0], ToolOutputSubstitution)
    assert first.minimal[0].at_step == fix_step
    # All witnesses are pairwise disjoint by item identity.
    seen_ids: set = set()
    for w in witnesses:
        ids = {id(x) for x in w.minimal}
        assert ids.isdisjoint(seen_ids), "witnesses must share no items"
        seen_ids |= ids


def test_shapley_assigns_full_weight_to_lone_cause(tmp_path):
    """Shapley exact mode: lone cause gets weight ~1.0; decoys ~0.0."""
    path, _ = _record(tmp_path)
    t = replay(path)
    fix_step = _step_id(t, 1)
    cause = ToolOutputSubstitution(
        at_step=fix_step, tool_call_id=None, fake_response=LOOKUP_FIXED_ROW,
    )
    decoy_a = ModelSubstitution(at_step=_step_id(t, 0), new_model_id="gpt-4o-mini-2024-07-18")
    decoy_b = ModelSubstitution(at_step=_step_id(t, 2), new_model_id="gpt-4o-mini-2024-07-18")
    decoy_c = PromptSubstitution(
        at_step=_step_id(t, 8),
        new_messages=[{"role": "system", "content": "noop decoy"}],
    )
    noisy = SubstitutionSet(items=[cause, decoy_a, decoy_b, decoy_c])

    out = attribute_substitutions(t, noisy, _country_us_predicate, executor=_executor())
    assert out.weights is not None
    assert abs(out.weight_for(cause) - 1.0) < 1e-9, out.weight_for(cause)
    for d in (decoy_a, decoy_b, decoy_c):
        assert abs(out.weight_for(d)) < 1e-9, (type(d).__name__, out.weight_for(d))
    # The minimal subset is exactly the items with positive weight.
    assert len(out.minimal) == 1
    assert out.minimal[0] is cause


def test_shapley_sampled_within_tolerance_of_exact(tmp_path):
    """Shapley sampled (n>6) is within tolerance of exact weights for the lone-cause case."""
    path, _ = _record(tmp_path)
    t = replay(path)
    fix_step = _step_id(t, 1)
    cause = ToolOutputSubstitution(
        at_step=fix_step, tool_call_id=None, fake_response=LOOKUP_FIXED_ROW,
    )
    # Build 7 items so we land in the sampled branch.
    items = [cause]
    for i in [0, 2, 3, 4, 5, 6]:
        items.append(ModelSubstitution(at_step=_step_id(t, i), new_model_id="gpt-4o-mini-2024-07-18"))
    noisy = SubstitutionSet(items=items)

    sampled = minimize_substitutions(
        t, noisy, _country_us_predicate,
        options=MinimizeOptions(strategy=ShapleyAttributionStrategy(permutations=64)),
        executor=_executor(),
    )
    assert sampled.weights is not None
    # The lone cause should still attract essentially all the weight.
    assert sampled.weight_for(cause) > 0.7, sampled.weight_for(cause)
    # Decoys should each have small weight.
    for d in items[1:]:
        assert abs(sampled.weight_for(d)) < 0.2, sampled.weight_for(d)


def test_attribute_substitutions_convenience_wrapper(tmp_path):
    path, _ = _record(tmp_path)
    t = replay(path)
    noisy, _ = _build_noisy_set(t)
    out = attribute_substitutions(t, noisy, _country_us_predicate, executor=_executor())
    assert isinstance(out, MinimizationResult)
    assert out.weights is not None
    assert out.strategy_name == "shapley"


def test_back_compat_ddmin_substitutions_unchanged(tmp_path):
    """The legacy entry point still works with its original signature."""
    path, _ = _record(tmp_path)
    t = replay(path)
    noisy, fix_step = _build_noisy_set(t)
    out = ddmin_substitutions(t, noisy, _country_us_predicate, executor=_executor())
    assert isinstance(out, MinimizationResult)
    assert len(out.minimal) == 1
    assert out.minimal[0].at_step == fix_step
    # New fields present on the result, defaulted sensibly.
    assert out.strategy_name == "ddmin"
    assert out.cache_hits >= 0
    assert out.weights is None


# --------------------------------- new CLI flag tests


def test_cli_minimize_strategy_brute(tmp_path):
    path, _ = _record(tmp_path)
    t = replay(path)
    fix_step = _step_id(t, 1)
    decoy_step = _step_id(t, 0)
    cmd = [
        sys.executable, "-m", "stepback.cli", "minimize", path,
        "--substitute", f"tool_output@{fix_step}=:inline:" + json.dumps(LOOKUP_FIXED_ROW),
        "--substitute", f"model@{decoy_step}=gpt-4o-mini-2024-07-18",
        "--predicate", "'acme-us' in str(result.steps[1].outputs)",
        "--strategy", "brute",
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr
    payload = json.loads(cp.stdout)
    assert payload["strategy"] == "brute"
    assert payload["minimal_count"] == 1
    assert "cache_hits" in payload


def test_cli_minimize_strategy_shapley_emits_weights(tmp_path):
    path, _ = _record(tmp_path)
    t = replay(path)
    fix_step = _step_id(t, 1)
    decoy_step = _step_id(t, 0)
    cmd = [
        sys.executable, "-m", "stepback.cli", "minimize", path,
        "--substitute", f"tool_output@{fix_step}=:inline:" + json.dumps(LOOKUP_FIXED_ROW),
        "--substitute", f"model@{decoy_step}=gpt-4o-mini-2024-07-18",
        "--predicate", "'acme-us' in str(result.steps[1].outputs)",
        "--strategy", "shapley",
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr
    payload = json.loads(cp.stdout)
    assert payload["strategy"] == "shapley"
    assert "weights" in payload
    # The cause weight should be ~1.0; decoy ~0.0.
    weights = {(w["kind"], w["at_step"]): w["weight"] for w in payload["weights"]}
    cause_key = ("ToolOutputSubstitution", fix_step)
    decoy_key = ("ModelSubstitution", decoy_step)
    assert weights[cause_key] > 0.9, weights
    assert abs(weights[decoy_key]) < 0.1, weights


def test_cli_minimize_all_witnesses(tmp_path):
    path, _ = _record(tmp_path)
    t = replay(path)
    fix_step = _step_id(t, 1)
    decoy_step = _step_id(t, 0)
    cmd = [
        sys.executable, "-m", "stepback.cli", "minimize", path,
        "--substitute", f"tool_output@{fix_step}=:inline:" + json.dumps(LOOKUP_FIXED_ROW),
        "--substitute", f"model@{decoy_step}=gpt-4o-mini-2024-07-18",
        "--predicate", "'acme-us' in str(result.steps[1].outputs)",
        "--all-witnesses", "--max-witnesses", "3",
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr
    payload = json.loads(cp.stdout)
    assert "witnesses" in payload
    assert payload["witness_count"] >= 1
    assert payload["witnesses"][0]["minimal_count"] == 1


def test_cli_minimize_probe_budget_exit_code_5(tmp_path):
    path, _ = _record(tmp_path)
    t = replay(path)
    fix_step = _step_id(t, 1)
    decoy_step = _step_id(t, 0)
    cmd = [
        sys.executable, "-m", "stepback.cli", "minimize", path,
        "--substitute", f"tool_output@{fix_step}=:inline:" + json.dumps(LOOKUP_FIXED_ROW),
        "--substitute", f"model@{decoy_step}=gpt-4o-mini-2024-07-18",
        "--predicate", "'acme-us' in str(result.steps[1].outputs)",
        "--probe-budget", "1",
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    assert cp.returncode == 5, (cp.returncode, cp.stderr, cp.stdout)
    payload = json.loads(cp.stdout)
    assert payload.get("budget_exhausted") is True
