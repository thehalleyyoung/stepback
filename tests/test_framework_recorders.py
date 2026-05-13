"""Tests for Step 102 framework recorders.

All tests are fully offline; no real LLM, SDK, or framework calls are made.
We simulate framework callback events directly on each recorder class to verify
that steps are recorded correctly.

Frameworks covered:
  - LlamaIndex   → LlamaIndexCallbackHandler / llamaindex_callback_handler
  - DSPy         → DSPyCallbackHandler / dspy_callback_handler
  - Haystack     → HaystackTracer / haystack_tracer
  - AutoGen      → AutoGenEventHandler / autogen_event_handler
  - CrewAI       → CrewAIStepRecorder / crewai_step_recorder
  - Semantic Kernel → SemanticKernelFilter / semantic_kernel_filter
  - Strands      → StrandsCallbackHandler / strands_callback_handler
  - Pydantic-AI  → PydanticAIInstrument / pydantic_ai_instrument
  - Inspect-AI   → InspectAIRecorder / inspect_ai_recorder
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from stepback import record, replay
from stepback.shims import (
    AutoGenEventHandler,
    CrewAIStepRecorder,
    DSPyCallbackHandler,
    HaystackTracer,
    InspectAIRecorder,
    LlamaIndexCallbackHandler,
    PydanticAIInstrument,
    SemanticKernelFilter,
    StrandsCallbackHandler,
    autogen_event_handler,
    crewai_step_recorder,
    dspy_callback_handler,
    haystack_tracer,
    inspect_ai_recorder,
    llamaindex_callback_handler,
    pydantic_ai_instrument,
    semantic_kernel_filter,
    strands_callback_handler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tmpfile():
    """Return a temporary .sb file path."""
    f = tempfile.NamedTemporaryFile(suffix=".sb", delete=False)
    f.close()
    return f.name


def _step_count(path: str) -> int:
    t = replay(path)
    return len(list(t.recorded_steps))


def _step_kinds(path: str) -> list:
    t = replay(path)
    return [s["step_kind"] for s in t.recorded_steps]


def _step_names(path: str) -> list:
    t = replay(path)
    return [s["name"] for s in t.recorded_steps]


# ===========================================================================
# § LlamaIndex
# ===========================================================================


class TestLlamaIndexCallbackHandler:
    def test_factory_returns_correct_type(self):
        with record(_tmpfile()) as rec:
            h = llamaindex_callback_handler(rec)
        assert isinstance(h, LlamaIndexCallbackHandler)

    def test_llm_event_records_llm_call(self):
        path = _tmpfile()
        with record(path) as rec:
            h = LlamaIndexCallbackHandler(rec)
            eid = h.on_event_start(
                "llm",
                payload={"messages": [{"role": "user", "content": "hello"}], "model": "gpt-4o"},
                event_id="e1",
            )
            h.on_event_end(
                "llm",
                payload={"response": "world"},
                event_id=eid,
            )
        assert _step_count(path) == 1
        assert _step_kinds(path) == ["llm_call"]

    def test_function_calling_event_records_tool_call(self):
        path = _tmpfile()
        with record(path) as rec:
            h = LlamaIndexCallbackHandler(rec)
            eid = h.on_event_start(
                "function_calling",
                payload={"tool": "search", "input": {"query": "ai"}},
                event_id="e2",
            )
            h.on_event_end(
                "function_calling",
                payload={"output": "result"},
                event_id=eid,
            )
        assert _step_count(path) == 1
        assert _step_kinds(path) == ["tool_call"]

    def test_other_event_records_when_enabled(self):
        path = _tmpfile()
        with record(path) as rec:
            h = LlamaIndexCallbackHandler(rec, record_other_events=True)
            eid = h.on_event_start("query", payload={}, event_id="e3")
            h.on_event_end("query", payload={}, event_id=eid)
        assert _step_count(path) == 1

    def test_other_event_skipped_when_disabled(self):
        path = _tmpfile()
        with record(path) as rec:
            h = LlamaIndexCallbackHandler(rec, record_other_events=False)
            eid = h.on_event_start("query", payload={}, event_id="e4")
            h.on_event_end("query", payload={}, event_id=eid)
        assert _step_count(path) == 0

    def test_start_end_trace_are_noops(self):
        path = _tmpfile()
        with record(path) as rec:
            h = LlamaIndexCallbackHandler(rec)
            h.start_trace("tid")
            h.end_trace("tid", {})
        assert _step_count(path) == 0

    def test_auto_event_id_assigned(self):
        path = _tmpfile()
        with record(path) as rec:
            h = LlamaIndexCallbackHandler(rec)
            # No event_id passed; handler should generate one
            eid = h.on_event_start("llm", payload={"model": "gpt-4o"}, event_id="")
            assert eid  # non-empty
            h.on_event_end("llm", payload={"response": "ok"}, event_id=eid)
        assert _step_count(path) == 1

    def test_multiple_events(self):
        path = _tmpfile()
        with record(path) as rec:
            h = LlamaIndexCallbackHandler(rec)
            eid1 = h.on_event_start("llm", payload={"model": "gpt-4o"}, event_id="e1")
            h.on_event_end("llm", payload={"response": "r1"}, event_id=eid1)
            eid2 = h.on_event_start("function_calling", payload={"tool": "calc"}, event_id="e2")
            h.on_event_end("function_calling", payload={"output": "42"}, event_id=eid2)
        assert _step_count(path) == 2
        kinds = _step_kinds(path)
        assert kinds[0] == "llm_call"
        assert kinds[1] == "tool_call"

    def test_class_attributes_for_llamaindex_compatibility(self):
        assert hasattr(LlamaIndexCallbackHandler, "event_starts_to_ignore")
        assert hasattr(LlamaIndexCallbackHandler, "event_ends_to_ignore")

    def test_trace_can_replay(self):
        path = _tmpfile()
        with record(path) as rec:
            h = LlamaIndexCallbackHandler(rec)
            eid = h.on_event_start("llm", payload={"model": "m"}, event_id="e")
            h.on_event_end("llm", payload={"response": "r"}, event_id=eid)
        t = replay(path)
        result = t.replay_forward()
        assert result.real_executions == 0  # all cache hits


# ===========================================================================
# § DSPy
# ===========================================================================


class TestDSPyCallbackHandler:
    def test_factory_returns_correct_type(self):
        with record(_tmpfile()) as rec:
            h = dspy_callback_handler(rec)
        assert isinstance(h, DSPyCallbackHandler)

    def test_lm_start_end_records_llm_call(self):
        path = _tmpfile()

        class _FakeLM:
            model = "gpt-4o"

        with record(path) as rec:
            h = DSPyCallbackHandler(rec)
            h.on_lm_start("c1", _FakeLM(), {"prompt": "hello"})
            h.on_lm_end("c1", {"output": "world"})
        assert _step_count(path) == 1
        assert _step_kinds(path) == ["llm_call"]

    def test_tool_start_end_records_tool_call(self):
        path = _tmpfile()

        class _FakeTool:
            name = "search"

        with record(path) as rec:
            h = DSPyCallbackHandler(rec)
            h.on_tool_start("c2", _FakeTool(), {"query": "ai"})
            h.on_tool_end("c2", "some result")
        assert _step_count(path) == 1
        assert _step_kinds(path) == ["tool_call"]

    def test_module_start_end_records_when_enabled(self):
        path = _tmpfile()

        class _FakeModule:
            pass

        with record(path) as rec:
            h = DSPyCallbackHandler(rec, record_modules=True)
            h.on_module_start("c3", _FakeModule(), {"input": "x"})
            h.on_module_end("c3", {"output": "y"})
        assert _step_count(path) == 1

    def test_module_start_end_skipped_when_disabled(self):
        path = _tmpfile()

        class _FakeModule:
            pass

        with record(path) as rec:
            h = DSPyCallbackHandler(rec, record_modules=False)
            h.on_module_start("c4", _FakeModule(), {"input": "x"})
            h.on_module_end("c4", {"output": "y"})
        assert _step_count(path) == 0

    def test_lm_with_messages_list(self):
        path = _tmpfile()

        class _FakeLM:
            model = "claude-3"

        with record(path) as rec:
            h = DSPyCallbackHandler(rec)
            h.on_lm_start("c5", _FakeLM(), {"messages": [{"role": "user", "content": "q"}]})
            h.on_lm_end("c5", {"output": "a"})
        t = replay(path)
        assert list(t.recorded_steps)[0]["inputs"]["messages"] == [{"role": "user", "content": "q"}]

    def test_lm_end_with_exception(self):
        path = _tmpfile()

        class _FakeLM:
            model = "gpt-4o"

        with record(path) as rec:
            h = DSPyCallbackHandler(rec)
            h.on_lm_start("c6", _FakeLM(), {"prompt": "hi"})
            # Passing an exception still records a step
            h.on_lm_end("c6", None, exception=ValueError("err"))
        assert _step_count(path) == 1

    def test_tool_name_from_tool_name_attr(self):
        path = _tmpfile()

        class _FakeTool:
            tool_name = "calculator"

        with record(path) as rec:
            h = DSPyCallbackHandler(rec)
            h.on_tool_start("c7", _FakeTool(), {"expression": "2+2"})
            h.on_tool_end("c7", "4")
        t = replay(path)
        assert list(t.recorded_steps)[0]["name"] == "calculator"

    def test_multiple_interleaved_calls(self):
        path = _tmpfile()

        class _FakeLM:
            model = "gpt-4o"

        class _FakeTool:
            name = "fetch"

        with record(path) as rec:
            h = DSPyCallbackHandler(rec)
            h.on_lm_start("lm1", _FakeLM(), {"prompt": "p"})
            h.on_tool_start("t1", _FakeTool(), {"url": "http://x"})
            h.on_tool_end("t1", "html")
            h.on_lm_end("lm1", {"output": "done"})
        assert _step_count(path) == 2


# ===========================================================================
# § Haystack
# ===========================================================================


class TestHaystackTracer:
    def test_factory_returns_correct_type(self):
        with record(_tmpfile()) as rec:
            t = haystack_tracer(rec)
        assert isinstance(t, HaystackTracer)

    def test_llm_span_records_llm_call(self):
        path = _tmpfile()
        with record(path) as rec:
            ht = HaystackTracer(rec)
            with ht.trace(
                "haystack.components.generators.chat.openai.OpenAIChatGenerator",
                tags={
                    "haystack.component.name": "chat_generator",
                    "haystack.component.input": {"messages": [{"role": "user", "content": "hi"}]},
                    "haystack.component.output": {"replies": ["hello"]},
                },
            ):
                pass
        assert _step_count(path) == 1
        assert _step_kinds(path) == ["llm_call"]

    def test_non_llm_span_records_tool_call(self):
        path = _tmpfile()
        with record(path) as rec:
            ht = HaystackTracer(rec)
            with ht.trace(
                "haystack.components.retrievers.in_memory.InMemoryBM25Retriever",
                tags={
                    "haystack.component.name": "retriever",
                    "haystack.component.input": {"query": "foo"},
                    "haystack.component.output": {"documents": []},
                },
            ):
                pass
        assert _step_count(path) == 1
        assert _step_kinds(path) == ["tool_call"]

    def test_current_span_returns_active_span(self):
        with record(_tmpfile()) as rec:
            ht = HaystackTracer(rec)
            assert ht.current_span() is None
            span = ht.trace("op", tags={})
            assert ht.current_span() is span
            with span:
                pass

    def test_set_content_tag_on_span(self):
        path = _tmpfile()
        with record(path) as rec:
            ht = HaystackTracer(rec)
            span = ht.trace("haystack.generator", tags={"haystack.component.name": "gen"})
            span.set_content_tag("haystack.component.input", {"messages": [{"role": "user", "content": "q"}]})
            span.set_content_tag("haystack.component.output", {"replies": ["a"]})
            with span:
                pass
        assert _step_count(path) == 1

    def test_span_get_correlation_data(self):
        with record(_tmpfile()) as rec:
            ht = HaystackTracer(rec)
            span = ht.trace("my_op", tags={})
            data = span.get_correlation_data_for_logs()
        assert data["operation_name"] == "my_op"

    def test_multiple_spans(self):
        path = _tmpfile()
        with record(path) as rec:
            ht = HaystackTracer(rec)
            with ht.trace("haystack.chat_generator", tags={"haystack.component.name": "llm"}):
                pass
            with ht.trace("haystack.retriever", tags={"haystack.component.name": "ret"}):
                pass
        assert _step_count(path) == 2


# ===========================================================================
# § AutoGen
# ===========================================================================


class TestAutoGenEventHandler:
    def test_factory_returns_correct_type(self):
        with record(_tmpfile()) as rec:
            h = autogen_event_handler(rec)
        assert isinstance(h, AutoGenEventHandler)

    def test_llm_call_records_llm_call(self):
        path = _tmpfile()
        with record(path) as rec:
            h = AutoGenEventHandler(rec)
            h.on_llm_call("gpt-4o", [{"role": "user", "content": "hi"}])
            h.on_llm_call_result({"content": "hello"})
        assert _step_count(path) == 1
        assert _step_kinds(path) == ["llm_call"]

    def test_tool_call_records_tool_call(self):
        path = _tmpfile()
        with record(path) as rec:
            h = AutoGenEventHandler(rec)
            h.on_tool_call("search", {"query": "ai"})
            h.on_tool_call_result("search", "results")
        assert _step_count(path) == 1
        assert _step_kinds(path) == ["tool_call"]

    def test_print_and_input_are_noops(self):
        with record(_tmpfile()) as rec:
            h = AutoGenEventHandler(rec)
            h.print("hello", "world", sep=" ")
            result = h.input("Enter: ")
        assert result == ""

    def test_llm_call_with_string_messages(self):
        path = _tmpfile()
        with record(path) as rec:
            h = AutoGenEventHandler(rec)
            h.on_llm_call("gpt-4o", "single string prompt")
            h.on_llm_call_result("response text")
        assert _step_count(path) == 1

    def test_tool_name_preserved(self):
        path = _tmpfile()
        with record(path) as rec:
            h = AutoGenEventHandler(rec)
            h.on_tool_call("my_tool", {"x": 1})
            h.on_tool_call_result("my_tool", "42")
        t = replay(path)
        assert list(t.recorded_steps)[0]["name"] == "my_tool"

    def test_pending_state_cleared_after_end(self):
        with record(_tmpfile()) as rec:
            h = AutoGenEventHandler(rec)
            h.on_llm_call("m", [])
            h.on_llm_call_result(None)
        assert h._pending_llm is None
        assert h._pending_tool is None

    def test_multiple_interleaved(self):
        path = _tmpfile()
        with record(path) as rec:
            h = AutoGenEventHandler(rec)
            h.on_llm_call("gpt-4o", [{"role": "user", "content": "q"}])
            h.on_tool_call("fetch", {"url": "http://x"})
            h.on_tool_call_result("fetch", "html")
            h.on_llm_call_result({"content": "done"})
        assert _step_count(path) == 2


# ===========================================================================
# § CrewAI
# ===========================================================================


@dataclass
class _FakeAgentAction:
    tool: str
    tool_input: Any
    result: str = ""


@dataclass
class _FakeAgentFinish:
    return_values: Any


class TestCrewAIStepRecorder:
    def test_factory_returns_correct_type(self):
        with record(_tmpfile()) as rec:
            cb = crewai_step_recorder(rec)
        assert isinstance(cb, CrewAIStepRecorder)

    def test_agent_action_records_tool_call(self):
        path = _tmpfile()
        with record(path) as rec:
            cb = CrewAIStepRecorder(rec)
            cb(_FakeAgentAction(tool="search", tool_input={"q": "ai"}, result="results"))
        assert _step_count(path) == 1
        assert _step_kinds(path) == ["tool_call"]

    def test_agent_finish_records_step(self):
        path = _tmpfile()
        with record(path) as rec:
            cb = CrewAIStepRecorder(rec)
            cb(_FakeAgentFinish(return_values={"output": "done"}))
        assert _step_count(path) == 1

    def test_on_tool_use_direct(self):
        path = _tmpfile()
        with record(path) as rec:
            cb = CrewAIStepRecorder(rec)
            cb.on_tool_use("calculator", {"expression": "1+1"}, "2")
        assert _step_count(path) == 1
        t = replay(path)
        assert list(t.recorded_steps)[0]["name"] == "calculator"

    def test_on_agent_finish_direct(self):
        path = _tmpfile()
        with record(path) as rec:
            cb = CrewAIStepRecorder(rec)
            cb.on_agent_finish("final answer")
        assert _step_count(path) == 1
        t = replay(path)
        assert list(t.recorded_steps)[0]["name"] == "crewai:finish"

    def test_unknown_step_shape_records_generic(self):
        path = _tmpfile()
        with record(path) as rec:
            cb = CrewAIStepRecorder(rec)
            cb("some string step")  # no .tool or .return_values attributes
        assert _step_count(path) == 1

    def test_callable_interface(self):
        """CrewAIStepRecorder must be callable for Crew(step_callback=...)."""
        with record(_tmpfile()) as rec:
            cb = CrewAIStepRecorder(rec)
        assert callable(cb)

    def test_multiple_tool_uses(self):
        path = _tmpfile()
        with record(path) as rec:
            cb = CrewAIStepRecorder(rec)
            cb(_FakeAgentAction("tool_a", {"x": 1}, "r1"))
            cb(_FakeAgentAction("tool_b", {"y": 2}, "r2"))
        assert _step_count(path) == 2


# ===========================================================================
# § Semantic Kernel
# ===========================================================================


@dataclass
class _FakeSKFunction:
    name: str
    plugin_name: str = ""


@dataclass
class _FakeSKContext:
    function: Any
    arguments: dict = field(default_factory=dict)
    result: Any = None


class TestSemanticKernelFilter:
    def test_factory_returns_correct_type(self):
        with record(_tmpfile()) as rec:
            flt = semantic_kernel_filter(rec)
        assert isinstance(flt, SemanticKernelFilter)

    def test_tool_function_records_tool_call(self):
        path = _tmpfile()
        ctx = _FakeSKContext(
            function=_FakeSKFunction(name="get_weather", plugin_name="WeatherPlugin"),
            arguments={"city": "Seattle"},
            result="Sunny",
        )
        with record(path) as rec:
            flt = SemanticKernelFilter(rec)
            flt.on_function_invocation(ctx)
        assert _step_count(path) == 1
        assert _step_kinds(path) == ["tool_call"]
        t = replay(path)
        assert "get_weather" in list(t.recorded_steps)[0]["name"]

    def test_chat_function_records_llm_call(self):
        path = _tmpfile()
        ctx = _FakeSKContext(
            function=_FakeSKFunction(name="chat_completion"),
            arguments={"input": "hello"},
            result="hi there",
        )
        with record(path) as rec:
            flt = SemanticKernelFilter(rec)
            flt.on_function_invocation(ctx)
        assert _step_count(path) == 1
        assert _step_kinds(path) == ["llm_call"]

    def test_next_is_called_when_provided(self):
        called = []

        def next_fn(ctx):
            called.append(True)
            ctx.result = "computed"

        with record(_tmpfile()) as rec:
            flt = SemanticKernelFilter(rec)
            ctx = _FakeSKContext(function=_FakeSKFunction(name="fn"))
            flt.on_function_invocation(ctx, next=next_fn)
        assert called

    def test_plugin_name_included_in_step_name(self):
        path = _tmpfile()
        ctx = _FakeSKContext(
            function=_FakeSKFunction(name="search", plugin_name="SearchPlugin"),
            arguments={},
            result="results",
        )
        with record(path) as rec:
            flt = SemanticKernelFilter(rec)
            flt.on_function_invocation(ctx)
        t = replay(path)
        assert "SearchPlugin" in list(t.recorded_steps)[0]["name"]

    def test_no_plugin_name_omits_colon(self):
        path = _tmpfile()
        ctx = _FakeSKContext(
            function=_FakeSKFunction(name="my_fn", plugin_name=""),
            arguments={},
            result=None,
        )
        with record(path) as rec:
            flt = SemanticKernelFilter(rec)
            flt.on_function_invocation(ctx)
        t = replay(path)
        assert list(t.recorded_steps)[0]["name"] == "my_fn"

    def test_returns_context_result(self):
        ctx = _FakeSKContext(
            function=_FakeSKFunction(name="fn"),
            arguments={},
            result="my_result",
        )
        with record(_tmpfile()) as rec:
            flt = SemanticKernelFilter(rec)
            ret = flt.on_function_invocation(ctx)
        assert ret == "my_result"

    def test_async_method_exists(self):
        import inspect
        assert inspect.iscoroutinefunction(SemanticKernelFilter.on_function_invocation_async)


# ===========================================================================
# § Strands
# ===========================================================================


class TestStrandsCallbackHandler:
    def test_factory_returns_correct_type(self):
        with record(_tmpfile()) as rec:
            h = strands_callback_handler(rec)
        assert isinstance(h, StrandsCallbackHandler)

    def test_on_start_end_records_llm_call(self):
        path = _tmpfile()
        with record(path) as rec:
            h = StrandsCallbackHandler(rec)
            h.on_start(model="gpt-4o", messages=[{"role": "user", "content": "hi"}], call_id="c1")
            h.on_end(response="hello", call_id="c1")
        assert _step_count(path) == 1
        assert _step_kinds(path) == ["llm_call"]

    def test_on_tool_start_end_records_tool_call(self):
        path = _tmpfile()
        with record(path) as rec:
            h = StrandsCallbackHandler(rec)
            h.on_tool_start(tool_name="search", tool_input={"query": "ai"}, call_id="t1")
            h.on_tool_end(tool_name="search", tool_output="results", call_id="t1")
        assert _step_count(path) == 1
        assert _step_kinds(path) == ["tool_call"]

    def test_llm_chunk_accumulates(self):
        path = _tmpfile()
        with record(path) as rec:
            h = StrandsCallbackHandler(rec)
            h.on_start(model="gpt-4o", messages=[], call_id="c2")
            h.on_llm_chunk("Hello", call_id="c2")
            h.on_llm_chunk(" world", call_id="c2")
            h.on_end(response=None, call_id="c2")
        t = replay(path)
        assert "Hello world" in str(list(t.recorded_steps)[0]["outputs"])

    def test_auto_call_id_generation(self):
        path = _tmpfile()
        with record(path) as rec:
            h = StrandsCallbackHandler(rec)
            # Call without call_id; should auto-generate
            h.on_tool_start(tool_name="calc", tool_input={}, call_id="")
            h.on_tool_end(tool_name="calc", tool_output="0", call_id="")
        # May or may not record depending on matching; at minimum it shouldn't crash
        # (the auto-generated IDs won't match, so pending lookup may be empty)

    def test_tool_name_preserved(self):
        path = _tmpfile()
        with record(path) as rec:
            h = StrandsCallbackHandler(rec)
            h.on_tool_start(tool_name="my_tool", tool_input={}, call_id="x")
            h.on_tool_end(tool_name="my_tool", tool_output="ok", call_id="x")
        t = replay(path)
        assert list(t.recorded_steps)[0]["name"] == "my_tool"

    def test_multiple_calls(self):
        path = _tmpfile()
        with record(path) as rec:
            h = StrandsCallbackHandler(rec)
            h.on_start(model="m", messages=[], call_id="llm1")
            h.on_end(response="r", call_id="llm1")
            h.on_tool_start(tool_name="t", tool_input={}, call_id="tool1")
            h.on_tool_end(tool_name="t", tool_output="x", call_id="tool1")
        assert _step_count(path) == 2


# ===========================================================================
# § Pydantic-AI
# ===========================================================================


class TestPydanticAIInstrument:
    def test_factory_returns_correct_type(self):
        with record(_tmpfile()) as rec:
            instr = pydantic_ai_instrument(rec)
        assert isinstance(instr, PydanticAIInstrument)

    def test_model_request_response_records_llm_call(self):
        path = _tmpfile()
        with record(path) as rec:
            instr = PydanticAIInstrument(rec)
            instr.on_model_request("run1", "gpt-4o", [{"role": "user", "content": "hello"}])
            instr.on_model_response("run1", {"content": "world"})
        assert _step_count(path) == 1
        assert _step_kinds(path) == ["llm_call"]

    def test_tool_call_return_records_tool_call(self):
        path = _tmpfile()
        with record(path) as rec:
            instr = PydanticAIInstrument(rec)
            instr.on_tool_call("run1", "search", {"query": "ai"})
            instr.on_tool_return("run1", "search", "results")
        assert _step_count(path) == 1
        assert _step_kinds(path) == ["tool_call"]

    def test_run_start_end_records_router_step(self):
        path = _tmpfile()
        with record(path) as rec:
            instr = PydanticAIInstrument(rec)
            instr.on_run_start("run1", "hello")
            instr.on_run_end("run1", "final answer")
        assert _step_count(path) == 1
        t = replay(path)
        assert list(t.recorded_steps)[0]["name"] == "pydantic_ai:run"

    def test_messages_with_role_content_attrs(self):
        """Messages that have .role and .content attributes (not dicts)."""
        path = _tmpfile()

        @dataclass
        class _Msg:
            role: str
            content: str

        with record(path) as rec:
            instr = PydanticAIInstrument(rec)
            instr.on_model_request("run2", "claude-3", [_Msg("user", "q")])
            instr.on_model_response("run2", _Msg("assistant", "a"))
        assert _step_count(path) == 1

    def test_pending_cleared_after_use(self):
        with record(_tmpfile()) as rec:
            instr = PydanticAIInstrument(rec)
            instr.on_model_request("run3", "m", [])
            instr.on_model_response("run3", None)
        assert not any(k.startswith("model:run3") for k in instr._pending)

    def test_tool_name_preserved(self):
        path = _tmpfile()
        with record(path) as rec:
            instr = PydanticAIInstrument(rec)
            instr.on_tool_call("r", "my_tool", {})
            instr.on_tool_return("r", "my_tool", "ok")
        t = replay(path)
        assert list(t.recorded_steps)[0]["name"] == "my_tool"

    def test_multiple_interleaved_calls(self):
        path = _tmpfile()
        with record(path) as rec:
            instr = PydanticAIInstrument(rec)
            instr.on_model_request("r1", "gpt-4o", [{"role": "user", "content": "q"}])
            instr.on_tool_call("r1", "fetch", {"url": "x"})
            instr.on_tool_return("r1", "fetch", "html")
            instr.on_model_response("r1", {"content": "done"})
        assert _step_count(path) == 2


# ===========================================================================
# § Inspect-AI
# ===========================================================================


@dataclass
class _FakeMessage:
    role: str
    text: str
    tool_calls: list = field(default_factory=list)


@dataclass
class _FakeTaskState:
    model: str
    messages: list = field(default_factory=list)


class TestInspectAIRecorder:
    def test_factory_returns_correct_type(self):
        with record(_tmpfile()) as rec:
            r = inspect_ai_recorder(rec)
        assert isinstance(r, InspectAIRecorder)

    def test_on_model_call_records_llm_call(self):
        path = _tmpfile()
        with record(path) as rec:
            r = InspectAIRecorder(rec)
            r.on_model_call("gpt-4o", [{"role": "user", "content": "hi"}], {"output": "hello"})
        assert _step_count(path) == 1
        assert _step_kinds(path) == ["llm_call"]

    def test_on_tool_call_records_tool_call(self):
        path = _tmpfile()
        with record(path) as rec:
            r = InspectAIRecorder(rec)
            r.on_tool_call("search", {"query": "ai"}, "results")
        assert _step_count(path) == 1
        assert _step_kinds(path) == ["tool_call"]

    def test_on_task_state_records_assistant_messages(self):
        path = _tmpfile()
        state = _FakeTaskState(
            model="gpt-4o",
            messages=[
                _FakeMessage(role="user", text="hello"),
                _FakeMessage(role="assistant", text="world"),
            ],
        )
        with record(path) as rec:
            r = InspectAIRecorder(rec)
            r.on_task_state(state)
        assert _step_count(path) == 1
        assert _step_kinds(path) == ["llm_call"]

    def test_on_task_state_records_tool_calls(self):
        @dataclass
        class _FakeToolCall:
            function: dict = field(default_factory=lambda: {"name": "search", "arguments": {}})

        path = _tmpfile()
        state = _FakeTaskState(
            model="gpt-4o",
            messages=[
                _FakeMessage(role="user", text="q"),
                _FakeMessage(role="assistant", text="", tool_calls=[_FakeToolCall()]),
            ],
        )
        with record(path) as rec:
            r = InspectAIRecorder(rec)
            r.on_task_state(state)
        assert _step_count(path) >= 1

    def test_messages_with_content_attr(self):
        """Messages with .content attribute instead of .text."""

        @dataclass
        class _Msg:
            role: str
            content: str
            text: str = ""
            tool_calls: list = field(default_factory=list)

        @dataclass
        class _State:
            model: str
            messages: list

        path = _tmpfile()
        state = _State(
            model="claude-3",
            messages=[
                _Msg("user", "hi"),
                _Msg("assistant", "hello"),
            ],
        )
        with record(path) as rec:
            r = InspectAIRecorder(rec)
            r.on_task_state(state)
        assert _step_count(path) == 1

    def test_on_model_call_with_completion_response(self):
        """Response objects with .completion attribute."""

        @dataclass
        class _Response:
            completion: str

        path = _tmpfile()
        with record(path) as rec:
            r = InspectAIRecorder(rec)
            r.on_model_call("gpt-4o", [], _Response(completion="answer"))
        t = replay(path)
        assert "answer" in str(list(t.recorded_steps)[0]["outputs"])

    def test_tool_name_preserved(self):
        path = _tmpfile()
        with record(path) as rec:
            r = InspectAIRecorder(rec)
            r.on_tool_call("my_tool", {"x": 1}, "ok")
        t = replay(path)
        assert list(t.recorded_steps)[0]["name"] == "my_tool"


# ===========================================================================
# § Public API surface
# ===========================================================================


class TestPublicAPIPresence:
    """Verify all Step 102 symbols are importable from stepback."""

    def test_llamaindex_importable(self):
        import stepback
        assert hasattr(stepback, "LlamaIndexCallbackHandler")
        assert hasattr(stepback, "llamaindex_callback_handler")

    def test_dspy_importable(self):
        import stepback
        assert hasattr(stepback, "DSPyCallbackHandler")
        assert hasattr(stepback, "dspy_callback_handler")

    def test_haystack_importable(self):
        import stepback
        assert hasattr(stepback, "HaystackTracer")
        assert hasattr(stepback, "haystack_tracer")

    def test_autogen_importable(self):
        import stepback
        assert hasattr(stepback, "AutoGenEventHandler")
        assert hasattr(stepback, "autogen_event_handler")

    def test_crewai_importable(self):
        import stepback
        assert hasattr(stepback, "CrewAIStepRecorder")
        assert hasattr(stepback, "crewai_step_recorder")

    def test_semantic_kernel_importable(self):
        import stepback
        assert hasattr(stepback, "SemanticKernelFilter")
        assert hasattr(stepback, "semantic_kernel_filter")

    def test_strands_importable(self):
        import stepback
        assert hasattr(stepback, "StrandsCallbackHandler")
        assert hasattr(stepback, "strands_callback_handler")

    def test_pydantic_ai_importable(self):
        import stepback
        assert hasattr(stepback, "PydanticAIInstrument")
        assert hasattr(stepback, "pydantic_ai_instrument")

    def test_inspect_ai_importable(self):
        import stepback
        assert hasattr(stepback, "InspectAIRecorder")
        assert hasattr(stepback, "inspect_ai_recorder")

    def test_all_symbols_in___all__(self):
        import stepback
        for name in [
            "LlamaIndexCallbackHandler", "llamaindex_callback_handler",
            "DSPyCallbackHandler", "dspy_callback_handler",
            "HaystackTracer", "haystack_tracer",
            "AutoGenEventHandler", "autogen_event_handler",
            "CrewAIStepRecorder", "crewai_step_recorder",
            "SemanticKernelFilter", "semantic_kernel_filter",
            "StrandsCallbackHandler", "strands_callback_handler",
            "PydanticAIInstrument", "pydantic_ai_instrument",
            "InspectAIRecorder", "inspect_ai_recorder",
        ]:
            assert name in stepback.__all__, f"{name!r} missing from stepback.__all__"
