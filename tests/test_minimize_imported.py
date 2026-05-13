"""Tests for Step 86: minimization for imported traces with partial executors.

Covers:
1. UnavailableExecutorError exception hierarchy and attributes.
2. PartialExecutor.execute() raises UnavailableExecutorError for unavailable steps.
3. PartialExecutor.is_available() logic for tools and models.
4. replay with PartialExecutor + fallback_recorded=True falls back for unavailable steps.
5. replay with PartialExecutor + fallback_recorded=False propagates UnavailableExecutorError.
6. audit_executor_requirements returns correct requirements and availability flags.
7. MinimizeOptions.skip_unavailable_executors treats UnavailableExecutorError as False.
8. minimize_imported_trace with default executor (fallback) produces results.
9. minimize_imported_trace with PartialExecutor + skip_unavailable_executors.
10. Public API exports.
"""
from __future__ import annotations

import pytest

from stepback import (
    Executor,
    MissingExecutor,
    MinimizeOptions,
    MinimizationResult,
    PartialExecutor,
    PredicateNotTriggered,
    StepExecutorRequirement,
    UnavailableExecutorError,
    audit_executor_requirements,
    minimize_imported_trace,
    record,
    replay,
)
from stepback.recorder import RecorderKey
from stepback.substitutions import (
    PromptSubstitution,
    SubstitutionSet,
    ToolOutputSubstitution,
)
from stepback.testing import (
    LOOKUP_FIXED_ROW,
    fake_llm,
    fake_tool,
    run_recorded_agent,
)


# ------------------------------------------------------------------- helpers

# The canonical fixture uses model "gpt-4o-2024-11-20" and tools
# "lookup_customer", "payment.transfer", "echo".
_FIXTURE_MODEL = "gpt-4o-2024-11-20"
_FIXTURE_TOOLS = {"lookup_customer", "payment.transfer", "echo"}


def _record_fixture(tmp_path):
    """Record the canonical 12-step fixture agent."""
    key = RecorderKey.fresh()
    path = str(tmp_path / "run.sb")
    with record(path, key=key) as rec:
        run_recorded_agent(rec)
    return path, key


def _executor():
    return Executor(llm=fake_llm, tool=fake_tool)


# ============================================================ UnavailableExecutorError

class TestUnavailableExecutorError:
    def test_is_subclass_of_missing_executor(self):
        assert issubclass(UnavailableExecutorError, MissingExecutor)

    def test_is_subclass_of_runtime_error(self):
        assert issubclass(UnavailableExecutorError, RuntimeError)

    def test_attributes_default_none(self):
        e = UnavailableExecutorError("msg")
        assert e.step_id is None
        assert e.kind is None
        assert e.name is None
        assert e.executor_type is None

    def test_attributes_populated(self):
        e = UnavailableExecutorError(
            "tool not available",
            step_id="s-42",
            kind="tool_call",
            name="lookup_db",
            executor_type="tool",
        )
        assert e.step_id == "s-42"
        assert e.kind == "tool_call"
        assert e.name == "lookup_db"
        assert e.executor_type == "tool"
        assert "tool not available" in str(e)

    def test_catchable_as_missing_executor(self):
        """UnavailableExecutorError must be catchable as MissingExecutor."""
        with pytest.raises(MissingExecutor):
            raise UnavailableExecutorError("unavailable")


# ============================================================ StepExecutorRequirement

class TestStepExecutorRequirement:
    def test_dataclass_fields(self):
        req = StepExecutorRequirement(
            step_id="step:1",
            kind="llm_call",
            name=_FIXTURE_MODEL,
            executor_type="llm",
            executor_available=True,
        )
        assert req.step_id == "step:1"
        assert req.kind == "llm_call"
        assert req.name == _FIXTURE_MODEL
        assert req.executor_type == "llm"
        assert req.executor_available is True

    def test_unavailable_flag(self):
        req = StepExecutorRequirement(
            step_id="step:2",
            kind="tool_call",
            name="lookup_customer",
            executor_type="tool",
            executor_available=False,
        )
        assert req.executor_available is False


# ============================================================ PartialExecutor

class TestPartialExecutor:
    """Tests for PartialExecutor.is_available() and execute()."""

    def test_is_available_all_tools_when_none(self):
        ex = PartialExecutor(tool=lambda n, a: "r")
        assert ex.is_available("tool_call", {"name": "search"}) is True
        assert ex.is_available("tool_call", {"name": "anything"}) is True

    def test_is_available_restricts_tools(self):
        ex = PartialExecutor(tool=lambda n, a: "r", available_tools={"search"})
        assert ex.is_available("tool_call", {"name": "search"}) is True
        assert ex.is_available("tool_call", {"name": "lookup_customer"}) is False

    def test_is_available_empty_tools_set(self):
        ex = PartialExecutor(tool=lambda n, a: "r", available_tools=set())
        assert ex.is_available("tool_call", {"name": "search"}) is False

    def test_is_available_all_models_when_none(self):
        ex = PartialExecutor(llm=lambda m, msgs: {})
        assert ex.is_available("llm_call", {"model": _FIXTURE_MODEL}) is True
        assert ex.is_available("llm_call", {"model": "claude-3"}) is True

    def test_is_available_restricts_models(self):
        ex = PartialExecutor(llm=lambda m, msgs: {}, available_models={_FIXTURE_MODEL})
        assert ex.is_available("llm_call", {"model": _FIXTURE_MODEL}) is True
        assert ex.is_available("llm_call", {"model": "claude-3"}) is False

    def test_is_available_no_llm_callback(self):
        ex = PartialExecutor(available_models={_FIXTURE_MODEL})
        assert ex.is_available("llm_call", {"model": _FIXTURE_MODEL}) is False

    def test_is_available_builtin_kinds(self):
        ex = PartialExecutor(available_tools=set(), available_models=set())
        assert ex.is_available("parallel_branch_open", {}) is True
        assert ex.is_available("exception", {}) is True
        assert ex.is_available("parallel_branch_join", {}) is True

    def test_execute_raises_unavailable_for_restricted_tool(self):
        ex = PartialExecutor(
            tool=lambda n, a: "ok",
            available_tools={"search"},
        )
        with pytest.raises(UnavailableExecutorError) as exc_info:
            ex.execute("tool_call", {"name": "lookup_customer", "arguments": {}})
        err = exc_info.value
        assert err.kind == "tool_call"
        assert err.name == "lookup_customer"
        assert err.executor_type == "tool"

    def test_execute_raises_unavailable_for_restricted_model(self):
        ex = PartialExecutor(
            llm=lambda m, msgs: {},
            available_models={_FIXTURE_MODEL},
        )
        with pytest.raises(UnavailableExecutorError) as exc_info:
            ex.execute("llm_call", {"model": "claude-3", "messages": []})
        err = exc_info.value
        assert err.kind == "llm_call"
        assert err.name == "claude-3"
        assert err.executor_type == "llm"

    def test_execute_succeeds_for_available_tool(self):
        ex = PartialExecutor(
            tool=fake_tool,
            available_tools={"echo"},
        )
        result = ex.execute("tool_call", {"name": "echo", "arguments": {"text": "hi"}})
        assert result == {"result": {"echo": "hi"}}

    def test_execute_succeeds_for_available_model(self):
        ex = PartialExecutor(
            llm=fake_llm,
            available_models={_FIXTURE_MODEL},
        )
        msgs = [{"role": "user", "content": "hello"}]
        result = ex.execute("llm_call", {"model": _FIXTURE_MODEL, "messages": msgs})
        assert "choices" in result

    def test_partial_executor_is_subclass_of_executor(self):
        assert issubclass(PartialExecutor, Executor)


# ============================================================ replay with PartialExecutor

class TestReplayWithPartialExecutor:
    """Integration: replay engine correctly handles UnavailableExecutorError."""

    def test_fallback_recorded_true_uses_recorded_output_for_unavailable(self, tmp_path):
        """With fallback_recorded=True, unavailable dirty steps use recorded output."""
        path, key = _record_fixture(tmp_path)
        trace = replay(path)

        # Substitute step:1 (llm_call) → makes downstream steps dirty.
        # Make tools unavailable so step:2 (tool_call) falls back to recorded.
        ex = PartialExecutor(
            llm=fake_llm,
            available_tools=set(),        # no tools available
            available_models={_FIXTURE_MODEL},
            fallback_recorded=True,
        )
        trace.substitute(PromptSubstitution(
            "step:1", [{"role": "user", "content": "different prompt"}]
        ))
        result = trace.replay_forward(ex)

        # step:1 should be re-executed (LLM, available).
        s1 = result.find(lambda s: s.step_id == "step:1")
        assert s1 is not None
        assert s1.dirty

        # step:2 is a tool_call (lookup_customer) → tool unavailable → fallback.
        s2 = result.find(lambda s: s.step_id == "step:2")
        assert s2 is not None
        assert s2.dirty
        assert s2.provenance.cache_source == "fallback"

    def test_fallback_recorded_false_raises_for_unavailable(self, tmp_path):
        """Without fallback, UnavailableExecutorError propagates from the replay."""
        path, key = _record_fixture(tmp_path)
        trace = replay(path)

        ex = PartialExecutor(
            llm=fake_llm,
            available_tools=set(),        # no tools
            available_models={_FIXTURE_MODEL},
            fallback_recorded=False,
        )
        trace.substitute(PromptSubstitution(
            "step:1", [{"role": "user", "content": "different"}]
        ))
        with pytest.raises(UnavailableExecutorError):
            trace.replay_forward(ex)

    def test_all_available_replays_normally(self, tmp_path):
        """When all steps are available, PartialExecutor behaves like Executor."""
        path, key = _record_fixture(tmp_path)
        trace = replay(path)

        ex = PartialExecutor(
            llm=fake_llm,
            tool=fake_tool,
            available_models={_FIXTURE_MODEL},
            available_tools=_FIXTURE_TOOLS,
        )
        trace.substitute(PromptSubstitution(
            "step:1", [{"role": "user", "content": "different"}]
        ))
        result = trace.replay_forward(ex)
        assert result.real_executions > 0
        # No fallback should have been used for any dirty step.
        fallback_steps = [
            s for s in result.steps
            if s.dirty and s.provenance and s.provenance.cache_source == "fallback"
        ]
        assert fallback_steps == []


# ============================================================ audit_executor_requirements

class TestAuditExecutorRequirements:
    def test_returns_requirements_for_all_executor_kinds(self, tmp_path):
        path, key = _record_fixture(tmp_path)
        trace = replay(path)
        reqs = audit_executor_requirements(trace)
        kinds = {r.kind for r in reqs}
        # The 12-step fixture has llm_call and tool_call steps.
        assert "llm_call" in kinds
        assert "tool_call" in kinds

    def test_none_executor_marks_all_unavailable(self, tmp_path):
        path, key = _record_fixture(tmp_path)
        trace = replay(path)
        reqs = audit_executor_requirements(trace, executor=None)
        assert all(not r.executor_available for r in reqs)

    def test_plain_executor_no_callbacks_marks_unavailable(self, tmp_path):
        path, key = _record_fixture(tmp_path)
        trace = replay(path)
        ex = Executor()  # no callbacks
        reqs = audit_executor_requirements(trace, executor=ex)
        assert all(not r.executor_available for r in reqs)

    def test_plain_executor_with_callbacks_marks_available(self, tmp_path):
        path, key = _record_fixture(tmp_path)
        trace = replay(path)
        ex = Executor(llm=fake_llm, tool=fake_tool)
        reqs = audit_executor_requirements(trace, executor=ex)
        assert all(r.executor_available for r in reqs)

    def test_partial_executor_selective_availability(self, tmp_path):
        path, key = _record_fixture(tmp_path)
        trace = replay(path)
        ex = PartialExecutor(
            llm=fake_llm,
            tool=fake_tool,
            available_tools=set(),           # no tools available
            available_models={_FIXTURE_MODEL},
        )
        reqs = audit_executor_requirements(trace, executor=ex)
        llm_reqs = [r for r in reqs if r.kind == "llm_call"]
        tool_reqs = [r for r in reqs if r.kind == "tool_call"]
        assert all(r.executor_available for r in llm_reqs)
        assert all(not r.executor_available for r in tool_reqs)

    def test_executor_type_field_populated(self, tmp_path):
        path, key = _record_fixture(tmp_path)
        trace = replay(path)
        reqs = audit_executor_requirements(trace)
        for r in reqs:
            if r.kind == "llm_call":
                assert r.executor_type == "llm"
            elif r.kind == "tool_call":
                assert r.executor_type == "tool"
            elif r.kind == "router":
                assert r.executor_type == "router"

    def test_name_field_populated(self, tmp_path):
        path, key = _record_fixture(tmp_path)
        trace = replay(path)
        reqs = audit_executor_requirements(trace)
        llm_req = next(r for r in reqs if r.kind == "llm_call")
        assert llm_req.name == _FIXTURE_MODEL
        tool_req = next(r for r in reqs if r.kind == "tool_call")
        assert tool_req.name in _FIXTURE_TOOLS

    def test_builtin_steps_not_included(self, tmp_path):
        """parallel_branch_open, exception steps should not appear in audit."""
        path, key = _record_fixture(tmp_path)
        trace = replay(path)
        reqs = audit_executor_requirements(trace)
        kinds = {r.kind for r in reqs}
        assert "parallel_branch_open" not in kinds
        assert "exception" not in kinds
        assert "parallel_branch_join" not in kinds

    def test_step_ids_present(self, tmp_path):
        """The fixture steps step:1 and step:2 must appear in the audit."""
        path, key = _record_fixture(tmp_path)
        trace = replay(path)
        reqs = audit_executor_requirements(trace)
        ids = {r.step_id for r in reqs}
        assert "step:1" in ids
        assert "step:2" in ids


# ============================================================ MinimizeOptions.skip_unavailable_executors

class TestSkipUnavailableExecutors:
    def test_field_default_false(self):
        opts = MinimizeOptions()
        assert opts.skip_unavailable_executors is False

    def test_field_settable(self):
        opts = MinimizeOptions(skip_unavailable_executors=True)
        assert opts.skip_unavailable_executors is True

    def test_skip_treats_unavailable_as_false(self, tmp_path):
        """When skip_unavailable_executors=True, UnavailableExecutorError → False."""
        path, key = _record_fixture(tmp_path)
        trace = replay(path)

        # All-unavailable executor: every dirty step raises UnavailableExecutorError.
        ex = PartialExecutor(
            llm=fake_llm,
            tool=fake_tool,
            available_models=set(),  # no models available
            available_tools=set(),   # no tools available
        )
        # With all steps unavailable, the predicate can never fire with
        # skip_unavailable_executors=True — every probe returns False.
        # So minimize_imported_trace raises PredicateNotTriggered.
        subs = SubstitutionSet([
            PromptSubstitution("step:1", [{"role": "user", "content": "changed"}])
        ])
        opts = MinimizeOptions(skip_unavailable_executors=True)
        with pytest.raises(PredicateNotTriggered):
            minimize_imported_trace(trace, subs, lambda r: True, executor=ex, options=opts)

    def test_no_skip_propagates_unavailable(self, tmp_path):
        """Without skip, UnavailableExecutorError propagates from the oracle.

        This uses minimize_substitutions directly (not minimize_imported_trace,
        which automatically enables skip for PartialExecutor).
        """
        from stepback import minimize_substitutions
        path, key = _record_fixture(tmp_path)
        trace = replay(path)

        ex = PartialExecutor(
            llm=fake_llm,
            available_models=set(),  # no models → UnavailableExecutorError on step:1
        )
        subs = SubstitutionSet([
            PromptSubstitution("step:1", [{"role": "user", "content": "changed"}])
        ])
        opts = MinimizeOptions(skip_unavailable_executors=False)
        with pytest.raises(UnavailableExecutorError):
            minimize_substitutions(trace, subs, lambda r: True, executor=ex, options=opts)


# ============================================================ minimize_imported_trace

class TestMinimizeImportedTrace:
    def test_default_executor_uses_fallback(self, tmp_path):
        """Default executor (fallback_recorded=True) always produces a result."""
        path, key = _record_fixture(tmp_path)
        trace = replay(path)
        step1_id = trace.recorded_steps[0]["step_id"]

        subs = SubstitutionSet([
            PromptSubstitution(step1_id, [{"role": "user", "content": "changed"}])
        ])
        result = minimize_imported_trace(
            trace,
            subs,
            lambda r: r.any_step(lambda s: s.step_id == step1_id and s.dirty),
        )
        assert result is not None
        assert len(result.minimal) == 1

    def test_with_partial_executor_auto_skip(self, tmp_path):
        """PartialExecutor triggers auto skip_unavailable_executors.

        Tool steps that become dirty (unavailable) are skipped by the oracle
        (treated as False), so minimisation still finds the causal LLM sub.
        We use fallback_recorded=True so the replay can complete end-to-end
        even when tool steps are unavailable.
        """
        path, key = _record_fixture(tmp_path)
        trace = replay(path)
        step1_id = trace.recorded_steps[0]["step_id"]

        ex = PartialExecutor(
            llm=fake_llm,
            available_models={_FIXTURE_MODEL},
            available_tools=set(),  # tools unavailable → fallback to recorded
            fallback_recorded=True,
        )
        subs = SubstitutionSet([
            PromptSubstitution(step1_id, [{"role": "user", "content": "different"}])
        ])
        result = minimize_imported_trace(
            trace,
            subs,
            lambda r: r.any_step(lambda s: s.step_id == step1_id and s.dirty),
            executor=ex,
        )
        assert len(result.minimal) == 1
        assert result.final_result is not None

    def test_explicit_options_respected(self, tmp_path):
        """Explicit MinimizeOptions are merged, not overridden."""
        path, key = _record_fixture(tmp_path)
        trace = replay(path)
        step1_id = trace.recorded_steps[0]["step_id"]

        called = []

        def progress(probes, size):
            called.append(probes)

        subs = SubstitutionSet([
            PromptSubstitution(step1_id, [{"role": "user", "content": "p"}])
        ])
        minimize_imported_trace(
            trace,
            subs,
            lambda r: r.any_step(lambda s: s.step_id == step1_id and s.dirty),
            options=MinimizeOptions(progress=progress),
        )
        assert len(called) > 0

    def test_returns_minimization_result(self, tmp_path):
        path, key = _record_fixture(tmp_path)
        trace = replay(path)
        step1_id = trace.recorded_steps[0]["step_id"]
        subs = SubstitutionSet([
            PromptSubstitution(step1_id, [{"role": "user", "content": "x"}])
        ])
        result = minimize_imported_trace(
            trace,
            subs,
            lambda r: r.any_step(lambda s: s.step_id == step1_id and s.dirty),
        )
        assert isinstance(result, MinimizationResult)

    def test_predicate_not_triggered_raises(self, tmp_path):
        path, key = _record_fixture(tmp_path)
        trace = replay(path)
        step1_id = trace.recorded_steps[0]["step_id"]
        subs = SubstitutionSet([
            PromptSubstitution(step1_id, [{"role": "user", "content": "x"}])
        ])
        with pytest.raises(PredicateNotTriggered):
            minimize_imported_trace(trace, subs, lambda r: False)

    def test_full_fixture_fallback_minimizes(self, tmp_path):
        """Full 12-step fixture: causal sub is isolated from a decoy."""
        path, key = _record_fixture(tmp_path)
        trace = replay(path)

        steps = trace.recorded_steps
        llm_steps = [s for s in steps if s["step_kind"] == "llm_call"]
        assert len(llm_steps) >= 2

        causal_id = llm_steps[0]["step_id"]
        decoy_id = llm_steps[-1]["step_id"]

        subs = SubstitutionSet([
            PromptSubstitution(causal_id, [{"role": "user", "content": "INJECTED_CAUSAL"}]),
            PromptSubstitution(decoy_id, [{"role": "user", "content": "decoy"}]),
        ])
        result = minimize_imported_trace(
            trace,
            subs,
            lambda r: r.any_step(lambda s: s.step_id == causal_id and s.dirty),
        )
        assert len(result.minimal) == 1
        assert result.minimal[0].at_step == causal_id

    def test_partial_executor_multiple_subs_isolated(self, tmp_path):
        """PartialExecutor + multiple subs: only LLM sub is causal."""
        path, key = _record_fixture(tmp_path)
        trace = replay(path)

        steps = trace.recorded_steps
        llm_steps = [s for s in steps if s["step_kind"] == "llm_call"]
        tool_steps = [s for s in steps if s["step_kind"] == "tool_call"]
        assert len(llm_steps) >= 1 and len(tool_steps) >= 1

        llm_id = llm_steps[0]["step_id"]
        tool_id = tool_steps[0]["step_id"]

        ex = PartialExecutor(
            llm=fake_llm,
            available_models={_FIXTURE_MODEL},
            available_tools=set(),   # tools unavailable → skip in minimization
        )
        subs = SubstitutionSet([
            PromptSubstitution(llm_id, [{"role": "user", "content": "CHANGED"}]),
            ToolOutputSubstitution(tool_id, {"result": {"country": "US", "iban": "US11"}})
        ])
        result = minimize_imported_trace(
            trace,
            subs,
            lambda r: r.any_step(lambda s: s.step_id == llm_id and s.dirty),
            executor=ex,
        )
        # The LLM sub should be in minimal; the tool sub may be excluded (unavailable).
        minimal_ids = {s.at_step for s in result.minimal}
        assert llm_id in minimal_ids


# ============================================================ Public API

class TestPublicAPI:
    def test_all_symbols_importable(self):
        import stepback
        for sym in [
            "MissingExecutor",
            "UnavailableExecutorError",
            "StepExecutorRequirement",
            "PartialExecutor",
            "audit_executor_requirements",
            "minimize_imported_trace",
        ]:
            assert hasattr(stepback, sym), f"stepback.{sym} not in public API"

    def test_in_all(self):
        import stepback
        for sym in [
            "MissingExecutor",
            "UnavailableExecutorError",
            "StepExecutorRequirement",
            "PartialExecutor",
            "audit_executor_requirements",
            "minimize_imported_trace",
        ]:
            assert sym in stepback.__all__, f"{sym!r} not in __all__"
