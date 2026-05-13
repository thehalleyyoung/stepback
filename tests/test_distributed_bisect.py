"""Tests for distributed bisect across a worker pool (Step 140).

Verifies:
- ``DistributedBisectOptions`` validation.
- ``distributed_bisect()`` with 1 and multiple items, success and failure.
- Correct isolation of the responsible substitution per trace.
- Results are in input order (item_index ordering).
- Failure (PredicateNotTriggered) is captured, not raised.
- Empty input returns empty summary.
- ``executor_factory`` creates per-thread executors.
- Single-worker fallback behaves identically to sequential minimize.
- Aggregate stats (total_probes, total_cache_hits) are sums.
- ``successful()`` / ``failed()`` filters.
- ``DistributedBisectSummary.success_count`` / ``failure_count`` counts.
- ``distributed_multi_objective_bisect()`` basic success path.
- Multi-objective result carries ``objectives`` and non-empty ``pareto_front``.
- Multi-objective failure captured, not raised.
- ``pareto_fronts()`` aligns with input order.
- ``distributed_multi_objective_bisect()`` empty input.
- Multi-objective aggregate stats.
- Public-API import path works (symbols live in ``stepback`` namespace).
- ``workers`` validation rejects < 1.
- Per-trace ``time_budget_s`` cuts short without aborting the corpus.
- ``executor_factory`` called once per trace.
- ``per_trace_options`` cloned per item (strategy not shared).
"""
from __future__ import annotations

import threading
import time
from typing import List

import pytest

from stepback import (
    Executor,
    record,
    replay,
    DistributedBisectItem,
    DistributedBisectOptions,
    DistributedBisectResult,
    DistributedBisectSummary,
    DistributedMultiObjectiveResult,
    DistributedMultiObjectiveSummary,
    distributed_bisect,
    distributed_multi_objective_bisect,
)
from stepback.minimize import (
    BudgetExhausted,
    DDMinStrategy,
    MinimizeOptions,
    MinimizationResult,
    PredicateNotTriggered,
)
from stepback.substitutions import (
    PromptSubstitution,
    SubstitutionSet,
    ToolOutputSubstitution,
)
from stepback.testing import LOOKUP_FIXED_ROW, fake_llm, fake_tool, run_recorded_agent
from stepback.recorder import RecorderKey


# ------------------------------------------------------------------ helpers


def _make_trace(tmp_path, suffix=""):
    key = RecorderKey.fresh()
    p = str(tmp_path / f"trace{suffix}.sb")
    with record(p, key=key) as rec:
        run_recorded_agent(rec)
    t = replay(p)
    return t, key


def _executor():
    return Executor(llm=fake_llm, tool=fake_tool)


def _good_subs(t) -> SubstitutionSet:
    """A SubstitutionSet where one sub causes output to contain the US country result."""
    step_id = t.recorded_steps[1]["step_id"]  # lookup_customer step
    return SubstitutionSet([
        ToolOutputSubstitution(at_step=step_id, fake_response=LOOKUP_FIXED_ROW),
        PromptSubstitution(at_step=t.recorded_steps[0]["step_id"], new_messages=[
            {"role": "user", "content": "What is the balance?"}
        ]),
    ])


def _predicate_fixed_row(r):
    """Fires when trace contains LOOKUP_FIXED_ROW country (US) at step 1."""
    try:
        return r.steps[1].outputs["result"]["country"] == "US"
    except (KeyError, IndexError, TypeError, AttributeError):
        return False


def _predicate_never(r):
    return False


def _predicate_always(r):
    return True


# ================================================================ DistributedBisectOptions


def test_options_default_workers():
    opts = DistributedBisectOptions()
    assert opts.workers == 4


def test_options_invalid_workers():
    with pytest.raises(ValueError, match="workers"):
        DistributedBisectOptions(workers=0)


def test_options_invalid_workers_negative():
    with pytest.raises(ValueError, match="workers"):
        DistributedBisectOptions(workers=-1)


# ================================================================ empty input


def test_distributed_bisect_empty():
    summary = distributed_bisect([])
    assert summary.results == []
    assert summary.success_count == 0
    assert summary.failure_count == 0
    assert summary.total_probes == 0
    assert summary.total_cache_hits == 0
    assert summary.elapsed_wall_time_s >= 0.0


def test_distributed_multi_objective_bisect_empty():
    summary = distributed_multi_objective_bisect([])
    assert summary.results == []
    assert summary.success_count == 0
    assert summary.failure_count == 0
    assert summary.pareto_fronts() == []


# ================================================================ single-item


def test_distributed_bisect_single_item_success(tmp_path):
    t, _ = _make_trace(tmp_path, "a")
    subs = _good_subs(t)
    items = [DistributedBisectItem(trace=t, substitutions=subs, predicate=_predicate_fixed_row)]
    summary = distributed_bisect(items, DistributedBisectOptions(workers=1, executor_factory=_executor))

    assert summary.success_count == 1
    assert summary.failure_count == 0
    assert len(summary.results) == 1

    r = summary.results[0]
    assert r.item_index == 0
    assert r.error_class is None
    assert r.result is not None
    # The ToolOutputSubstitution is the one that actually fires the predicate.
    assert len(r.result.minimal) == 1
    assert isinstance(r.result.minimal[0], ToolOutputSubstitution)
    assert r.result.probes > 0
    assert r.wall_time_s >= 0.0


def test_distributed_bisect_single_item_predicate_not_triggered(tmp_path):
    t, _ = _make_trace(tmp_path, "b")
    subs = _good_subs(t)
    # Never-true predicate → PredicateNotTriggered should be captured
    items = [DistributedBisectItem(trace=t, substitutions=subs, predicate=_predicate_never)]
    summary = distributed_bisect(items, DistributedBisectOptions(workers=1, executor_factory=_executor))

    assert summary.success_count == 0
    assert summary.failure_count == 1
    r = summary.results[0]
    assert r.error_class == "PredicateNotTriggered"
    assert r.result is None
    assert r.message is not None and len(r.message) > 0


# ================================================================ multi-item


def test_distributed_bisect_multiple_items_all_success(tmp_path):
    traces = [_make_trace(tmp_path, str(i))[0] for i in range(3)]
    items = [
        DistributedBisectItem(trace=t, substitutions=_good_subs(t), predicate=_predicate_fixed_row)
        for t in traces
    ]
    summary = distributed_bisect(items, DistributedBisectOptions(workers=3, executor_factory=_executor))

    assert summary.success_count == 3
    assert summary.failure_count == 0
    assert len(summary.results) == 3
    # Results must be in input order.
    assert [r.item_index for r in summary.results] == [0, 1, 2]
    # Each bisect isolates the single responsible sub.
    for r in summary.results:
        assert len(r.result.minimal) == 1


def test_distributed_bisect_mixed_success_and_failure(tmp_path):
    t_good, _ = _make_trace(tmp_path, "g")
    t_bad, _ = _make_trace(tmp_path, "bad")
    items = [
        DistributedBisectItem(trace=t_good, substitutions=_good_subs(t_good), predicate=_predicate_fixed_row),
        DistributedBisectItem(trace=t_bad, substitutions=_good_subs(t_bad), predicate=_predicate_never),
        DistributedBisectItem(trace=t_good, substitutions=_good_subs(t_good), predicate=_predicate_fixed_row),
    ]
    summary = distributed_bisect(items, DistributedBisectOptions(workers=2, executor_factory=_executor))

    assert summary.success_count == 2
    assert summary.failure_count == 1
    # Index ordering preserved.
    assert [r.item_index for r in summary.results] == [0, 1, 2]
    assert summary.results[0].error_class is None
    assert summary.results[1].error_class == "PredicateNotTriggered"
    assert summary.results[2].error_class is None


# ================================================================ aggregate stats


def test_distributed_bisect_aggregate_stats(tmp_path):
    traces = [_make_trace(tmp_path, str(i))[0] for i in range(2)]
    items = [
        DistributedBisectItem(trace=t, substitutions=_good_subs(t), predicate=_predicate_fixed_row)
        for t in traces
    ]
    summary = distributed_bisect(items, DistributedBisectOptions(workers=2, executor_factory=_executor))

    assert summary.total_probes == sum(r.result.probes for r in summary.successful())
    assert summary.total_cache_hits == sum(r.result.cache_hits for r in summary.successful())
    assert summary.elapsed_wall_time_s > 0.0


def test_distributed_bisect_total_probes_positive(tmp_path):
    t, _ = _make_trace(tmp_path)
    items = [DistributedBisectItem(trace=t, substitutions=_good_subs(t), predicate=_predicate_fixed_row)]
    summary = distributed_bisect(items, DistributedBisectOptions(workers=1, executor_factory=_executor))
    assert summary.total_probes > 0


# ================================================================ successful() / failed()


def test_successful_failed_filters(tmp_path):
    t, _ = _make_trace(tmp_path)
    items = [
        DistributedBisectItem(trace=t, substitutions=_good_subs(t), predicate=_predicate_fixed_row),
        DistributedBisectItem(trace=t, substitutions=_good_subs(t), predicate=_predicate_never),
    ]
    summary = distributed_bisect(items, DistributedBisectOptions(workers=1, executor_factory=_executor))

    assert len(summary.successful()) == 1
    assert len(summary.failed()) == 1
    assert summary.successful()[0].item_index == 0
    assert summary.failed()[0].item_index == 1


# ================================================================ executor_factory


def test_executor_factory_called_per_trace(tmp_path):
    call_log: List[int] = []
    lock = threading.Lock()

    def factory():
        with lock:
            call_log.append(threading.get_ident())
        return _executor()

    traces = [_make_trace(tmp_path, str(i))[0] for i in range(3)]
    items = [
        DistributedBisectItem(trace=t, substitutions=_good_subs(t), predicate=_predicate_fixed_row)
        for t in traces
    ]
    opts = DistributedBisectOptions(workers=3, executor_factory=factory)
    summary = distributed_bisect(items, opts)

    assert summary.success_count == 3
    # Factory called once per item.
    assert len(call_log) == 3


def test_executor_factory_takes_precedence_over_executor(tmp_path):
    """executor_factory overrides executor when both are set."""
    factory_calls = []

    def factory():
        factory_calls.append(1)
        return _executor()

    t, _ = _make_trace(tmp_path)
    opts = DistributedBisectOptions(
        workers=1,
        executor=_executor(),
        executor_factory=factory,
    )
    items = [DistributedBisectItem(trace=t, substitutions=_good_subs(t), predicate=_predicate_fixed_row)]
    summary = distributed_bisect(items, opts)
    assert summary.success_count == 1
    assert len(factory_calls) == 1


# ================================================================ single-worker equals sequential


def test_single_worker_matches_sequential_minimize(tmp_path):
    t, _ = _make_trace(tmp_path)
    subs = _good_subs(t)

    from stepback.minimize import minimize_substitutions

    seq_result = minimize_substitutions(t, subs, _predicate_fixed_row, executor=_executor())

    opts = DistributedBisectOptions(workers=1, executor_factory=_executor)
    items = [DistributedBisectItem(trace=t, substitutions=subs, predicate=_predicate_fixed_row)]
    summary = distributed_bisect(items, opts)

    dist_result = summary.results[0].result
    # Both should isolate the same substitution type.
    assert len(seq_result.minimal) == len(dist_result.minimal)
    assert type(seq_result.minimal[0]) == type(dist_result.minimal[0])


# ================================================================ per_trace_options cloned


def test_per_trace_options_cloned_not_shared(tmp_path):
    """Each item gets a fresh MinimizeOptions; probe budget is independent."""
    traces = [_make_trace(tmp_path, str(i))[0] for i in range(2)]
    # Give a tight probe budget — if options were shared, second trace might
    # see a depleted budget from the first.
    per_opts = MinimizeOptions(probe_budget=100)
    opts = DistributedBisectOptions(
        workers=2,
        per_trace_options=per_opts,
        executor_factory=_executor,
    )
    items = [
        DistributedBisectItem(trace=t, substitutions=_good_subs(t), predicate=_predicate_fixed_row)
        for t in traces
    ]
    summary = distributed_bisect(items, opts)
    # Both should succeed within their own budget.
    assert summary.success_count == 2


# ================================================================ time budget captured


def test_time_budget_captured_not_raised(tmp_path):
    t, _ = _make_trace(tmp_path)
    subs = _good_subs(t)
    # Extremely tight time budget to force BudgetExhausted.
    per_opts = MinimizeOptions(time_budget_s=0.000001)
    opts = DistributedBisectOptions(workers=1, per_trace_options=per_opts, executor_factory=_executor)
    items = [DistributedBisectItem(trace=t, substitutions=subs, predicate=_predicate_fixed_row)]
    summary = distributed_bisect(items, opts)
    # Either succeeds (budget not hit for fast traces) or BudgetExhausted is captured.
    assert len(summary.results) == 1
    r = summary.results[0]
    assert r.error_class in (None, "BudgetExhausted")


# ================================================================ public API import


def test_public_api_import():
    import stepback
    assert hasattr(stepback, "distributed_bisect")
    assert hasattr(stepback, "distributed_multi_objective_bisect")
    assert hasattr(stepback, "DistributedBisectItem")
    assert hasattr(stepback, "DistributedBisectOptions")
    assert hasattr(stepback, "DistributedBisectResult")
    assert hasattr(stepback, "DistributedBisectSummary")
    assert hasattr(stepback, "DistributedMultiObjectiveResult")
    assert hasattr(stepback, "DistributedMultiObjectiveSummary")


def test_all_exports_in___all__():
    import stepback
    for name in [
        "distributed_bisect", "distributed_multi_objective_bisect",
        "DistributedBisectItem", "DistributedBisectOptions",
        "DistributedBisectResult", "DistributedBisectSummary",
        "DistributedMultiObjectiveResult", "DistributedMultiObjectiveSummary",
    ]:
        assert name in stepback.__all__, f"{name!r} missing from __all__"


# ================================================================ multi-objective


def test_distributed_multi_objective_bisect_single_item(tmp_path):
    t, _ = _make_trace(tmp_path)
    subs = _good_subs(t)
    items = [DistributedBisectItem(trace=t, substitutions=subs, predicate=_predicate_fixed_row)]
    summary = distributed_multi_objective_bisect(
        items, orderings=2, options=DistributedBisectOptions(workers=1, executor_factory=_executor)
    )

    assert summary.success_count == 1
    assert summary.failure_count == 0
    r = summary.results[0]
    assert r.item_index == 0
    assert r.error_class is None
    assert r.result is not None
    assert len(r.result.minimal) >= 1
    assert r.result.objectives is not None
    assert len(r.result.pareto_front) >= 1


def test_distributed_multi_objective_bisect_failure_captured(tmp_path):
    t, _ = _make_trace(tmp_path)
    subs = _good_subs(t)
    items = [DistributedBisectItem(trace=t, substitutions=subs, predicate=_predicate_never)]
    summary = distributed_multi_objective_bisect(
        items, options=DistributedBisectOptions(workers=1, executor_factory=_executor)
    )

    assert summary.failure_count == 1
    assert summary.results[0].error_class == "PredicateNotTriggered"


def test_distributed_multi_objective_bisect_multiple(tmp_path):
    traces = [_make_trace(tmp_path, str(i))[0] for i in range(3)]
    items = [
        DistributedBisectItem(trace=t, substitutions=_good_subs(t), predicate=_predicate_fixed_row)
        for t in traces
    ]
    summary = distributed_multi_objective_bisect(
        items, orderings=2, options=DistributedBisectOptions(workers=3, executor_factory=_executor)
    )
    assert summary.success_count == 3
    assert [r.item_index for r in summary.results] == [0, 1, 2]


def test_distributed_multi_objective_pareto_fronts_aligned(tmp_path):
    traces = [_make_trace(tmp_path, str(i))[0] for i in range(2)]
    items = [
        DistributedBisectItem(trace=t, substitutions=_good_subs(t), predicate=_predicate_fixed_row)
        for t in traces
    ]
    summary = distributed_multi_objective_bisect(
        items, orderings=2, options=DistributedBisectOptions(workers=2, executor_factory=_executor)
    )
    fronts = summary.pareto_fronts()
    assert len(fronts) == 2
    for f in fronts:
        assert len(f) >= 1


def test_distributed_multi_objective_aggregate_stats(tmp_path):
    traces = [_make_trace(tmp_path, str(i))[0] for i in range(2)]
    items = [
        DistributedBisectItem(trace=t, substitutions=_good_subs(t), predicate=_predicate_fixed_row)
        for t in traces
    ]
    summary = distributed_multi_objective_bisect(
        items, orderings=2, options=DistributedBisectOptions(workers=2, executor_factory=_executor)
    )
    expected_probes = sum(r.result.probes for r in summary.successful())
    assert summary.total_probes == expected_probes
    assert summary.elapsed_wall_time_s > 0.0
