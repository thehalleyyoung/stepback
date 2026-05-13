"""Tests for StepbackCallbackHandler — the LangChain / LangGraph callback recorder.

All tests are fully offline; no real LLM or framework calls are made.
We simulate LangChain callback events directly on the handler to test
that steps are recorded correctly, parent/child relationships are
preserved, and run_ids appear in metadata.
"""
from __future__ import annotations

import uuid
from typing import Any, List, Optional
from dataclasses import dataclass, field

import pytest

from stepback import record
from stepback.shims import (
    StepbackCallbackHandler,
    langchain_callback_handler,
)
from stepback import record, replay


# ---------------------------------------------------------------------------
# Minimal LangChain-shaped fakes (duck-typed, no real langchain required)
# ---------------------------------------------------------------------------


def _run_id() -> str:
    return str(uuid.uuid4())


@dataclass
class _FakeMessage:
    """Minimal BaseMessage-shaped object."""
    type: str
    content: str
    tool_calls: list = field(default_factory=list)
    tool_call_id: Optional[str] = None


@dataclass
class _FakeChatGeneration:
    """Minimal ChatGeneration-shaped object."""
    message: Any
    text: str = ""
    generation_info: dict = field(default_factory=dict)


@dataclass
class _FakeLLMResult:
    """Minimal LLMResult-shaped object."""
    generations: list  # List[List[Generation]]
    llm_output: Optional[dict] = None


# ---------------------------------------------------------------------------
# Helper: fire a complete LLM call through the handler
# ---------------------------------------------------------------------------


def _fire_llm_call(
    handler: StepbackCallbackHandler,
    *,
    model: str = "gpt-4o-mini",
    prompt: str = "Hello",
    response_text: str = "Hi there",
    run_id: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    usage: Optional[dict] = None,
) -> str:
    """Simulate on_chat_model_start → on_llm_end; returns run_id."""
    rid = run_id or _run_id()
    messages = [[_FakeMessage(type="human", content=prompt)]]
    serialized = {"kwargs": {"model_name": model}, "name": model}
    handler.on_chat_model_start(
        serialized, messages, run_id=rid, parent_run_id=parent_run_id
    )
    llm_output = {"model_name": model, "token_usage": usage or {
        "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15
    }}
    result = _FakeLLMResult(
        generations=[[_FakeChatGeneration(
            message=_FakeMessage(type="ai", content=response_text),
            generation_info={"finish_reason": "stop"},
        )]],
        llm_output=llm_output,
    )
    handler.on_llm_end(result, run_id=rid, parent_run_id=parent_run_id)
    return rid


def _fire_tool_call(
    handler: StepbackCallbackHandler,
    *,
    tool_name: str = "search",
    input_str: str = "query",
    output: str = "result",
    run_id: Optional[str] = None,
    parent_run_id: Optional[str] = None,
) -> str:
    """Simulate on_tool_start → on_tool_end; returns run_id."""
    rid = run_id or _run_id()
    serialized = {"name": tool_name}
    handler.on_tool_start(
        serialized, input_str, run_id=rid, parent_run_id=parent_run_id
    )
    handler.on_tool_end(output, run_id=rid, parent_run_id=parent_run_id)
    return rid


def _fire_chain(
    handler: StepbackCallbackHandler,
    *,
    chain_name: str = "MyChain",
    inputs: Optional[dict] = None,
    outputs: Optional[dict] = None,
    run_id: Optional[str] = None,
    parent_run_id: Optional[str] = None,
) -> str:
    """Simulate on_chain_start → on_chain_end; returns run_id."""
    rid = run_id or _run_id()
    serialized = {"name": chain_name}
    handler.on_chain_start(
        serialized,
        inputs or {"question": "What?"},
        run_id=rid,
        parent_run_id=parent_run_id,
    )
    handler.on_chain_end(
        outputs or {"answer": "42"},
        run_id=rid,
        parent_run_id=parent_run_id,
    )
    return rid


# ===========================================================================
# Tests
# ===========================================================================


class TestBasicConstruction:
    def test_factory_returns_handler(self, tmp_path):
        with record(str(tmp_path / "t.sb")) as rec:
            handler = langchain_callback_handler(rec)
        assert isinstance(handler, StepbackCallbackHandler)

    def test_direct_construction(self, tmp_path):
        with record(str(tmp_path / "t.sb")) as rec:
            handler = StepbackCallbackHandler(rec)
        assert handler._record_chains is True

    def test_record_chains_false(self, tmp_path):
        with record(str(tmp_path / "t.sb")) as rec:
            handler = StepbackCallbackHandler(rec, record_chains=False)
        assert handler._record_chains is False


class TestLLMCall:
    def test_chat_model_start_end_records_llm_step(self, tmp_path):
        path = str(tmp_path / "t.sb")
        with record(path) as rec:
            handler = StepbackCallbackHandler(rec)
            _fire_llm_call(handler, model="gpt-4o-mini", prompt="Hi", response_text="Hello!")

        steps = [s for s in replay(path).recorded_steps]
        llm_steps = [s for s in steps if s["step_kind"] == "llm_call"]
        assert len(llm_steps) == 1
        step = llm_steps[0]
        assert step["name"] == "gpt-4o-mini"
        assert step["inputs"]["model"] == "gpt-4o-mini"

    def test_llm_step_has_run_id_metadata(self, tmp_path):
        path = str(tmp_path / "t.sb")
        run_id = _run_id()
        with record(path) as rec:
            handler = StepbackCallbackHandler(rec)
            _fire_llm_call(handler, run_id=run_id)

        steps = [s for s in replay(path).recorded_steps if s["step_kind"] == "llm_call"]
        assert steps[0]["metadata"]["langchain_run_id"] == run_id

    def test_llm_step_messages_normalized(self, tmp_path):
        path = str(tmp_path / "t.sb")
        with record(path) as rec:
            handler = StepbackCallbackHandler(rec)
            _fire_llm_call(handler, prompt="Tell me a joke")

        steps = [s for s in replay(path).recorded_steps if s["step_kind"] == "llm_call"]
        msgs = steps[0]["inputs"]["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Tell me a joke"

    def test_llm_step_response_in_outputs(self, tmp_path):
        path = str(tmp_path / "t.sb")
        with record(path) as rec:
            handler = StepbackCallbackHandler(rec)
            _fire_llm_call(handler, response_text="Why did the chicken...")

        steps = [s for s in replay(path).recorded_steps if s["step_kind"] == "llm_call"]
        outputs = steps[0]["outputs"]
        assert outputs["choices"][0]["message"]["role"] == "assistant"
        assert "chicken" in outputs["choices"][0]["message"]["content"]

    def test_llm_step_usage_extracted(self, tmp_path):
        path = str(tmp_path / "t.sb")
        with record(path) as rec:
            handler = StepbackCallbackHandler(rec)
            _fire_llm_call(handler, usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30})

        steps = [s for s in replay(path).recorded_steps if s["step_kind"] == "llm_call"]
        usage = steps[0]["outputs"]["usage"]
        assert usage["prompt_tokens"] == 20
        assert usage["completion_tokens"] == 10

    def test_on_llm_start_plain_prompts(self, tmp_path):
        """on_llm_start (non-chat) should also record an llm_call step."""
        path = str(tmp_path / "t.sb")
        run_id = _run_id()
        with record(path) as rec:
            handler = StepbackCallbackHandler(rec)
            serialized = {"kwargs": {"model_name": "text-davinci-003"}}
            handler.on_llm_start(
                serialized, ["Translate to French: Hello"], run_id=run_id
            )
            result = _FakeLLMResult(
                generations=[[_FakeChatGeneration(
                    message=_FakeMessage(type="ai", content="Bonjour"),
                    generation_info={"finish_reason": "stop"},
                )]],
                llm_output={"model_name": "text-davinci-003", "token_usage": {}},
            )
            handler.on_llm_end(result, run_id=run_id)

        steps = [s for s in replay(path).recorded_steps if s["step_kind"] == "llm_call"]
        assert len(steps) == 1
        assert steps[0]["name"] == "text-davinci-003"

    def test_llm_error_cleans_up_pending(self, tmp_path):
        """on_llm_error should clean up pending state without recording a step."""
        path = str(tmp_path / "t.sb")
        run_id = _run_id()
        with record(path) as rec:
            handler = StepbackCallbackHandler(rec)
            serialized = {"kwargs": {"model_name": "gpt-4o"}}
            handler.on_chat_model_start(
                serialized,
                [[_FakeMessage(type="human", content="Hi")]],
                run_id=run_id,
            )
            assert run_id in handler._pending
            handler.on_llm_error(RuntimeError("timeout"), run_id=run_id)
            assert run_id not in handler._pending

        steps = [s for s in replay(path).recorded_steps if s["step_kind"] == "llm_call"]
        assert len(steps) == 0

    def test_multiple_llm_calls_recorded(self, tmp_path):
        path = str(tmp_path / "t.sb")
        with record(path) as rec:
            handler = StepbackCallbackHandler(rec)
            _fire_llm_call(handler, model="gpt-4o-mini")
            _fire_llm_call(handler, model="gpt-4o")
            _fire_llm_call(handler, model="claude-3-5-sonnet-20241022")

        steps = [s for s in replay(path).recorded_steps if s["step_kind"] == "llm_call"]
        assert len(steps) == 3
        model_names = [s["name"] for s in steps]
        assert "gpt-4o-mini" in model_names
        assert "claude-3-5-sonnet-20241022" in model_names


class TestToolCall:
    def test_tool_start_end_records_tool_step(self, tmp_path):
        path = str(tmp_path / "t.sb")
        with record(path) as rec:
            handler = StepbackCallbackHandler(rec)
            _fire_tool_call(handler, tool_name="calculator", input_str="2+2", output="4")

        steps = [s for s in replay(path).recorded_steps if s["step_kind"] == "tool_call"]
        assert len(steps) == 1
        step = steps[0]
        assert step["name"] == "calculator"
        assert step["outputs"]["result"] == "4"

    def test_tool_step_has_run_id_metadata(self, tmp_path):
        path = str(tmp_path / "t.sb")
        run_id = _run_id()
        with record(path) as rec:
            handler = StepbackCallbackHandler(rec)
            _fire_tool_call(handler, run_id=run_id)

        steps = [s for s in replay(path).recorded_steps if s["step_kind"] == "tool_call"]
        assert steps[0]["metadata"]["langchain_run_id"] == run_id

    def test_tool_dict_input_preserved(self, tmp_path):
        """Dict inputs should be preserved as-is in arguments."""
        path = str(tmp_path / "t.sb")
        with record(path) as rec:
            handler = StepbackCallbackHandler(rec)
            handler.on_tool_start(
                {"name": "search"},
                {"query": "latest news", "max_results": 5},
                run_id=_run_id(),
            )
            handler.on_tool_end("News here", run_id=list(handler._pending.keys())[-1])

        steps = [s for s in replay(path).recorded_steps if s["step_kind"] == "tool_call"]
        assert steps[0]["inputs"]["arguments"]["query"] == "latest news"

    def test_tool_error_cleans_up(self, tmp_path):
        path = str(tmp_path / "t.sb")
        run_id = _run_id()
        with record(path) as rec:
            handler = StepbackCallbackHandler(rec)
            handler.on_tool_start({"name": "flaky_tool"}, "arg", run_id=run_id)
            assert run_id in handler._pending
            handler.on_tool_error(RuntimeError("timeout"), run_id=run_id)
            assert run_id not in handler._pending

        steps = [s for s in replay(path).recorded_steps if s["step_kind"] == "tool_call"]
        assert len(steps) == 0


class TestChainRecording:
    def test_chain_records_router_step(self, tmp_path):
        path = str(tmp_path / "t.sb")
        with record(path) as rec:
            handler = StepbackCallbackHandler(rec)
            _fire_chain(handler, chain_name="SummaryChain", outputs={"summary": "Short text."})

        steps = [s for s in replay(path).recorded_steps if s["step_kind"] == "router"]
        assert len(steps) == 1
        assert steps[0]["name"] == "SummaryChain"

    def test_chain_disabled_records_nothing(self, tmp_path):
        path = str(tmp_path / "t.sb")
        with record(path) as rec:
            handler = StepbackCallbackHandler(rec, record_chains=False)
            _fire_chain(handler)

        steps = [s for s in replay(path).recorded_steps if s["step_kind"] == "router"]
        assert len(steps) == 0

    def test_chain_error_cleans_up(self, tmp_path):
        path = str(tmp_path / "t.sb")
        run_id = _run_id()
        with record(path) as rec:
            handler = StepbackCallbackHandler(rec)
            handler.on_chain_start(
                {"name": "BadChain"},
                {"x": 1},
                run_id=run_id,
            )
            assert run_id in handler._pending
            handler.on_chain_error(RuntimeError("oops"), run_id=run_id)
            assert run_id not in handler._pending


class TestParentChildRelationship:
    def test_nested_llm_has_chain_as_parent(self, tmp_path):
        """Chain → LLM call: llm step's parent_step_id should equal the chain's step_id."""
        path = str(tmp_path / "t.sb")
        chain_run_id = _run_id()
        llm_run_id = _run_id()

        with record(path) as rec:
            handler = StepbackCallbackHandler(rec)
            # Start chain
            handler.on_chain_start(
                {"name": "QAChain"},
                {"question": "What is 2+2?"},
                run_id=chain_run_id,
            )
            # LLM call nested inside chain
            _fire_llm_call(
                handler,
                run_id=llm_run_id,
                parent_run_id=chain_run_id,
            )
            # End chain
            handler.on_chain_end(
                {"answer": "4"},
                run_id=chain_run_id,
            )

        all_steps = [s for s in replay(path).recorded_steps]
        chain_steps = [s for s in all_steps if s["step_kind"] == "router"]
        llm_steps = [s for s in all_steps if s["step_kind"] == "llm_call"]

        assert len(chain_steps) == 1
        assert len(llm_steps) == 1
        chain_step_id = chain_steps[0]["step_id"]
        assert llm_steps[0]["parent_step_id"] == chain_step_id

    def test_nested_tool_has_chain_as_parent(self, tmp_path):
        """Chain → tool call: tool step's parent_step_id should equal the chain's step_id."""
        path = str(tmp_path / "t.sb")
        chain_run_id = _run_id()
        tool_run_id = _run_id()

        with record(path) as rec:
            handler = StepbackCallbackHandler(rec)
            handler.on_chain_start(
                {"name": "AgentChain"},
                {"input": "Search for cats"},
                run_id=chain_run_id,
            )
            _fire_tool_call(
                handler,
                tool_name="web_search",
                run_id=tool_run_id,
                parent_run_id=chain_run_id,
            )
            handler.on_chain_end(
                {"output": "cats are cute"},
                run_id=chain_run_id,
            )

        all_steps = [s for s in replay(path).recorded_steps]
        chain_steps = [s for s in all_steps if s["step_kind"] == "router"]
        tool_steps = [s for s in all_steps if s["step_kind"] == "tool_call"]

        chain_step_id = chain_steps[0]["step_id"]
        assert tool_steps[0]["parent_step_id"] == chain_step_id

    def test_top_level_llm_has_no_parent(self, tmp_path):
        path = str(tmp_path / "t.sb")
        with record(path) as rec:
            handler = StepbackCallbackHandler(rec)
            _fire_llm_call(handler)

        steps = [s for s in replay(path).recorded_steps if s["step_kind"] == "llm_call"]
        assert steps[0]["parent_step_id"] is None

    def test_run_id_preserved_in_step_id_map(self, tmp_path):
        """After an llm_end, the run_id should be in _run_to_step."""
        path = str(tmp_path / "t.sb")
        run_id = _run_id()
        with record(path) as rec:
            handler = StepbackCallbackHandler(rec)
            _fire_llm_call(handler, run_id=run_id)

        assert run_id in handler._run_to_step
        assert isinstance(handler._run_to_step[run_id], str)


class TestLangGraphNodes:
    def test_on_node_start_end_delegates_to_chain(self, tmp_path):
        """LangGraph on_node_start/end should produce router steps."""
        path = str(tmp_path / "t.sb")
        node_run_id = _run_id()

        with record(path) as rec:
            handler = StepbackCallbackHandler(rec)
            handler.on_node_start(
                {"name": "call_model"},
                {"messages": ["hello"]},
                run_id=node_run_id,
            )
            handler.on_node_end(
                {"messages": ["hi there"]},
                run_id=node_run_id,
            )

        steps = [s for s in replay(path).recorded_steps if s["step_kind"] == "router"]
        assert len(steps) == 1
        assert steps[0]["name"] == "call_model"

    def test_langgraph_multi_node_trace(self, tmp_path):
        """Simulate a 2-node LangGraph: call_model → use_tool."""
        path = str(tmp_path / "t.sb")
        node1_id = _run_id()
        node2_id = _run_id()
        llm_id = _run_id()
        tool_id = _run_id()

        with record(path) as rec:
            handler = StepbackCallbackHandler(rec)
            # node 1: call_model
            handler.on_node_start(
                {"name": "call_model"},
                {"messages": [{"role": "user", "content": "run tools"}]},
                run_id=node1_id,
            )
            _fire_llm_call(handler, run_id=llm_id, parent_run_id=node1_id)
            handler.on_node_end(
                {"messages": ["tool_calls_pending"]},
                run_id=node1_id,
            )
            # node 2: use_tool
            handler.on_node_start(
                {"name": "use_tool"},
                {"tool_name": "search"},
                run_id=node2_id,
                parent_run_id=node1_id,
            )
            _fire_tool_call(handler, tool_name="search", run_id=tool_id, parent_run_id=node2_id)
            handler.on_node_end(
                {"results": ["answer"]},
                run_id=node2_id,
            )

        all_steps = [s for s in replay(path).recorded_steps]
        router_steps = [s for s in all_steps if s["step_kind"] == "router"]
        llm_steps = [s for s in all_steps if s["step_kind"] == "llm_call"]
        tool_steps = [s for s in all_steps if s["step_kind"] == "tool_call"]

        assert len(router_steps) == 2
        assert len(llm_steps) == 1
        assert len(tool_steps) == 1
        # Verify parent chain
        node1_step_id = next(s["step_id"] for s in router_steps if s["name"] == "call_model")
        assert llm_steps[0]["parent_step_id"] == node1_step_id


class TestMessageNormalization:
    def test_human_message_normalized(self):
        msg = _FakeMessage(type="human", content="Hello")
        from stepback.shims import _normalize_lc_message
        result = _normalize_lc_message(msg)
        assert result == {"role": "user", "content": "Hello"}

    def test_ai_message_normalized(self):
        from stepback.shims import _normalize_lc_message
        msg = _FakeMessage(type="ai", content="Hi")
        result = _normalize_lc_message(msg)
        assert result == {"role": "assistant", "content": "Hi"}

    def test_system_message_normalized(self):
        from stepback.shims import _normalize_lc_message
        msg = _FakeMessage(type="system", content="You are helpful")
        result = _normalize_lc_message(msg)
        assert result == {"role": "system", "content": "You are helpful"}

    def test_dict_message_passthrough(self):
        from stepback.shims import _normalize_lc_message
        d = {"role": "user", "content": "hi"}
        result = _normalize_lc_message(d)
        assert result["role"] == "user"
        assert result["content"] == "hi"

    def test_list_content_flattened(self):
        from stepback.shims import _normalize_lc_message

        class MsgWithListContent:
            type = "human"
            content = [{"type": "text", "text": "Hello"}, {"type": "text", "text": " world"}]
            tool_calls = []
            tool_call_id = None

        result = _normalize_lc_message(MsgWithListContent())
        assert result["content"] == "Hello world"

    def test_tool_call_id_preserved(self):
        from stepback.shims import _normalize_lc_message
        msg = _FakeMessage(type="tool", content="result", tool_call_id="call_abc")
        result = _normalize_lc_message(msg)
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_abc"


class TestLLMResultNormalization:
    def test_chat_generation_extracted(self):
        from stepback.shims import _normalize_lc_llm_result
        gen = _FakeChatGeneration(
            message=_FakeMessage(type="ai", content="The answer is 42"),
            generation_info={"finish_reason": "stop"},
        )
        result = _FakeLLMResult(
            generations=[[gen]],
            llm_output={"model_name": "gpt-4o", "token_usage": {
                "prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8
            }},
        )
        canonical = _normalize_lc_llm_result(result, "gpt-4o")
        assert canonical["model"] == "gpt-4o"
        assert canonical["choices"][0]["message"]["content"] == "The answer is 42"
        assert canonical["usage"]["total_tokens"] == 8

    def test_empty_result_defaults(self):
        from stepback.shims import _normalize_lc_llm_result
        result = _FakeLLMResult(generations=[], llm_output=None)
        canonical = _normalize_lc_llm_result(result, "unknown")
        assert canonical["choices"][0]["message"]["role"] == "assistant"
        assert canonical["choices"][0]["message"]["content"] == ""

    def test_total_tokens_computed_when_zero(self):
        from stepback.shims import _normalize_lc_llm_result
        gen = _FakeChatGeneration(
            message=_FakeMessage(type="ai", content="hi"),
            generation_info={},
        )
        result = _FakeLLMResult(
            generations=[[gen]],
            llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 0}},
        )
        canonical = _normalize_lc_llm_result(result, "m")
        assert canonical["usage"]["total_tokens"] == 15


class TestNoOpCallbacks:
    """Ensure no-op callbacks don't raise and don't record spurious steps."""

    def test_on_llm_new_token_noop(self, tmp_path):
        path = str(tmp_path / "t.sb")
        with record(path) as rec:
            handler = StepbackCallbackHandler(rec)
            handler.on_llm_new_token("token", run_id=_run_id())

        steps = [s for s in replay(path).recorded_steps]
        assert len(steps) == 0

    def test_on_agent_action_noop(self, tmp_path):
        path = str(tmp_path / "t.sb")
        with record(path) as rec:
            handler = StepbackCallbackHandler(rec)
            handler.on_agent_action(object(), run_id=_run_id())

        steps = [s for s in replay(path).recorded_steps]
        assert len(steps) == 0

    def test_on_retriever_noop(self, tmp_path):
        path = str(tmp_path / "t.sb")
        with record(path) as rec:
            handler = StepbackCallbackHandler(rec)
            handler.on_retriever_start({"name": "chroma"}, "query", run_id=_run_id())
            handler.on_retriever_end([], run_id=_run_id())

        steps = [s for s in replay(path).recorded_steps]
        assert len(steps) == 0

    def test_on_text_noop(self, tmp_path):
        path = str(tmp_path / "t.sb")
        with record(path) as rec:
            handler = StepbackCallbackHandler(rec)
            handler.on_text("some text", run_id=_run_id())

        steps = [s for s in replay(path).recorded_steps]
        assert len(steps) == 0


class TestClassAttributes:
    """LangChain checks class attributes to configure callback dispatch."""

    def test_raise_error_false(self):
        assert StepbackCallbackHandler.raise_error is False

    def test_ignore_flags_false(self):
        assert StepbackCallbackHandler.ignore_llm is False
        assert StepbackCallbackHandler.ignore_chain is False
        assert StepbackCallbackHandler.ignore_agent is False

    def test_public_api_exported(self):
        import stepback
        assert hasattr(stepback, "StepbackCallbackHandler")
        assert hasattr(stepback, "langchain_callback_handler")
