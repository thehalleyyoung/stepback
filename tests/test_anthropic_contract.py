"""Anthropic contract tests — Step 91 in ``100_STEPS.md``.

Covers:
- Messages: ``wrap_anthropic`` recording basic chat, system prompt, multi-turn
  conversations, cache replay, and cost accounting.
- Tool use: ``tool_use`` content blocks canonicalised to OpenAI ``tool_calls``
  shape; ``finish_reason`` mapped to ``"tool_calls"``.
- Streaming: ``wrap_anthropic`` with ``stream=True``, SSE-event accumulation
  via ``_accumulate_anthropic_streaming_chunks``, and streaming tool calls.
- Thinking blocks: thinking-block events are silently dropped from the
  canonical content list; text blocks still appear correctly.
- Async clients: ``wrap_anthropic_async`` recording, replay cache hit,
  ``AnthropicShimContract.async_request`` hook.
- ABC hooks: ``AnthropicShimContract.stream_request`` and ``async_request``
  are implemented (not ``NotImplementedError``); other providers still
  raise ``NotImplementedError`` for those hooks.
- ``AnthropicShimContract.canonical_request``, ``canonical_response``,
  ``make_executor``, and ``version_probe``.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional

import pytest

from stepback import record, replay
from stepback.replay import Executor
from stepback.shims import (
    AnthropicMessage,
    AnthropicShimContract,
    AsyncWrappedAnthropic,
    WrappedAnthropic,
    _accumulate_anthropic_streaming_chunks,
    anthropic_executor,
    wrap_anthropic,
    wrap_anthropic_async,
)
from stepback.substitutions import SubstitutionSet


# =====================================================================
# Fake client helpers
# =====================================================================


@dataclass
class _FakeAnthropicMessages:
    """Fake ``anthropic.resources.Messages`` that returns pre-set responses."""

    responses: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    _idx: int = field(default=0, init=False, repr=False)

    def create(self, *, model: str, messages: list,
               system: Optional[str] = None,
               max_tokens: int = 1024,
               **kwargs: Any) -> Any:
        self.calls.append({
            "model": model, "messages": messages,
            "system": system, "max_tokens": max_tokens, "kwargs": kwargs,
        })
        resp = self.responses[self._idx % len(self.responses)]
        self._idx += 1
        if callable(getattr(resp, "__iter__", None)) and not isinstance(resp, dict):
            # streaming — return the iterable directly
            return resp
        return resp


@dataclass
class _FakeAnthropic:
    messages: _FakeAnthropicMessages

    @classmethod
    def with_response(cls, raw: dict) -> "_FakeAnthropic":
        return cls(messages=_FakeAnthropicMessages(responses=[raw]))

    @classmethod
    def with_stream(cls, events: list) -> "_FakeAnthropic":
        """Returns events as an iterator on the next create() call."""
        return cls(messages=_FakeAnthropicMessages(responses=[iter(events)]))

    @classmethod
    def with_multi(cls, responses: list) -> "_FakeAnthropic":
        return cls(messages=_FakeAnthropicMessages(responses=responses))


# Standard fake response dicts (non-streaming).
_SIMPLE_RESPONSE = {
    "id": "msg_001",
    "model": "claude-3-5-haiku-20241022",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello!"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}

_TOOL_USE_RESPONSE = {
    "id": "msg_002",
    "model": "claude-3-5-haiku-20241022",
    "role": "assistant",
    "content": [
        {
            "type": "tool_use",
            "id": "toolu_abc123",
            "name": "get_weather",
            "input": {"city": "Paris"},
        }
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 20, "output_tokens": 8},
}

_MIXED_RESPONSE = {
    "id": "msg_003",
    "model": "claude-3-5-haiku-20241022",
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Let me check that."},
        {
            "type": "tool_use",
            "id": "toolu_def456",
            "name": "search",
            "input": {"query": "stepback"},
        },
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 25, "output_tokens": 12},
}


def _replay_cache_only(trace_path: str) -> None:
    """Assert replay hits 100% cache (no executor calls)."""
    t = replay(trace_path)
    exec_ = Executor()
    result = t.run_replay(subs=SubstitutionSet(), executor=exec_)
    assert all(not s.dirty for s in result), "replay produced dirty steps"
    assert exec_.real_calls == 0, (
        f"expected 0 executor calls on cache replay, got {exec_.real_calls}"
    )


# =====================================================================
# § 1  _accumulate_anthropic_streaming_chunks — unit tests
# =====================================================================


def _make_events_text(msg_id: str = "msg_s1",
                      model: str = "claude-3-5-haiku-20241022",
                      text: str = "Hi there.",
                      input_tokens: int = 10,
                      output_tokens: int = 3) -> list:
    """Build a minimal text streaming event sequence."""
    return [
        {"type": "message_start", "message": {
            "id": msg_id, "model": model,
            "usage": {"input_tokens": input_tokens},
        }},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta",
         "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": output_tokens}},
        {"type": "message_stop"},
    ]


def _make_events_tool_use(msg_id: str = "msg_st",
                           model: str = "claude-3-5-haiku-20241022") -> list:
    """Build a tool_use streaming event sequence."""
    return [
        {"type": "message_start", "message": {
            "id": msg_id, "model": model,
            "usage": {"input_tokens": 15},
        }},
        {"type": "content_block_start", "index": 0,
         "content_block": {
             "type": "tool_use",
             "id": "toolu_stream_01",
             "name": "get_weather",
             "input": {},
         }},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": '{"city":'}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": '"Paris"}'}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta",
         "delta": {"stop_reason": "tool_use"},
         "usage": {"output_tokens": 8}},
        {"type": "message_stop"},
    ]


def _make_events_thinking_plus_text() -> list:
    """Build a stream with a thinking block followed by a text block."""
    return [
        {"type": "message_start", "message": {
            "id": "msg_think",
            "model": "claude-3-7-sonnet-20250219",
            "usage": {"input_tokens": 20},
        }},
        # Thinking block (should be excluded from final content).
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "thinking_delta", "thinking": "I need to think..."}},
        {"type": "content_block_stop", "index": 0},
        # Text block.
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "text_delta", "text": "The answer is 42."}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta",
         "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 10}},
        {"type": "message_stop"},
    ]


def test_accumulate_simple_text():
    events = _make_events_text(text="Hi there.")
    result = _accumulate_anthropic_streaming_chunks(iter(events))
    assert result["id"] == "msg_s1"
    assert result["model"] == "claude-3-5-haiku-20241022"
    assert result["role"] == "assistant"
    assert len(result["content"]) == 1
    assert result["content"][0]["type"] == "text"
    assert result["content"][0]["text"] == "Hi there."
    assert result["stop_reason"] == "end_turn"
    assert result["usage"]["input_tokens"] == 10
    assert result["usage"]["output_tokens"] == 3


def test_accumulate_multiple_text_deltas():
    events = [
        {"type": "message_start", "message": {
            "id": "m1", "model": "claude-3-5-haiku-20241022",
            "usage": {"input_tokens": 5},
        }},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "Hello"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": ", world"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "!"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta",
         "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 4}},
        {"type": "message_stop"},
    ]
    result = _accumulate_anthropic_streaming_chunks(iter(events))
    assert result["content"][0]["text"] == "Hello, world!"


def test_accumulate_tool_use():
    events = _make_events_tool_use()
    result = _accumulate_anthropic_streaming_chunks(iter(events))
    assert result["stop_reason"] == "tool_use"
    assert len(result["content"]) == 1
    block = result["content"][0]
    assert block["type"] == "tool_use"
    assert block["id"] == "toolu_stream_01"
    assert block["name"] == "get_weather"
    assert block["input"] == {"city": "Paris"}
    assert result["usage"]["input_tokens"] == 15
    assert result["usage"]["output_tokens"] == 8


def test_accumulate_thinking_blocks_excluded():
    """Thinking blocks do not appear in final content; text block is present."""
    events = _make_events_thinking_plus_text()
    result = _accumulate_anthropic_streaming_chunks(iter(events))
    assert len(result["content"]) == 1
    assert result["content"][0]["type"] == "text"
    assert result["content"][0]["text"] == "The answer is 42."


def test_accumulate_object_based_events():
    """Events may be objects with attributes rather than dicts."""
    class _Event:
        def __init__(self, **kw: Any) -> None:
            self.__dict__.update(kw)

    events = [
        _Event(type="message_start", message=_Event(
            id="obj_1", model="claude-3-5-haiku-20241022",
            usage=_Event(input_tokens=8),
        )),
        _Event(type="content_block_start", index=0,
               content_block=_Event(type="text", text="")),
        _Event(type="content_block_delta", index=0,
               delta=_Event(type="text_delta", text="Object-based!")),
        _Event(type="content_block_stop", index=0),
        _Event(type="message_delta",
               delta=_Event(stop_reason="end_turn"),
               usage=_Event(output_tokens=3)),
        _Event(type="message_stop"),
    ]
    result = _accumulate_anthropic_streaming_chunks(iter(events))
    assert result["id"] == "obj_1"
    assert result["content"][0]["text"] == "Object-based!"
    assert result["stop_reason"] == "end_turn"
    assert result["usage"]["input_tokens"] == 8
    assert result["usage"]["output_tokens"] == 3


def test_accumulate_empty_stream():
    """Empty stream yields a safe empty result."""
    result = _accumulate_anthropic_streaming_chunks(iter([]))
    assert result["content"] == []
    assert result["stop_reason"] is None


# =====================================================================
# § 2  wrap_anthropic basic messages
# =====================================================================


def test_wrap_anthropic_records_one_step(tmp_path):
    trace_path = str(tmp_path / "anth_basic.sb")
    fake = _FakeAnthropic.with_response(dict(_SIMPLE_RESPONSE))
    with record(trace_path) as rec:
        client = wrap_anthropic(fake, rec)
        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "ping"}],
        )
    assert isinstance(resp, AnthropicMessage)
    assert resp.stop_reason == "end_turn"
    t = replay(trace_path)
    assert len(t.recorded_steps) == 1
    assert t.recorded_steps[0]["step_kind"] == "llm_call"


def test_wrap_anthropic_system_prompt_unified(tmp_path):
    """System prompt appears as ``{"role": "system"}`` in recorded messages."""
    trace_path = str(tmp_path / "anth_sys.sb")
    fake = _FakeAnthropic.with_response(dict(_SIMPLE_RESPONSE))
    with record(trace_path) as rec:
        client = wrap_anthropic(fake, rec)
        client.messages.create(
            model="claude-3-5-haiku-20241022",
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "hello"}],
        )
    t = replay(trace_path)
    step = t.recorded_steps[0]
    msgs = step["inputs"]["messages"]
    assert msgs[0]["role"] == "system"
    assert "helpful assistant" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"


def test_wrap_anthropic_response_content(tmp_path):
    """The returned AnthropicMessage mirrors the response content."""
    trace_path = str(tmp_path / "anth_content.sb")
    fake = _FakeAnthropic.with_response(dict(_SIMPLE_RESPONSE))
    with record(trace_path) as rec:
        client = wrap_anthropic(fake, rec)
        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "say hello"}],
        )
    assert resp.content[0].text == "Hello!"
    assert resp.id == "msg_001"
    assert resp.model == "claude-3-5-haiku-20241022"


def test_wrap_anthropic_canonical_shape_stored(tmp_path):
    """The llm_response in the trace is in OpenAI canonical shape."""
    trace_path = str(tmp_path / "anth_shape.sb")
    fake = _FakeAnthropic.with_response(dict(_SIMPLE_RESPONSE))
    with record(trace_path) as rec:
        client = wrap_anthropic(fake, rec)
        client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "hi"}],
        )
    t = replay(trace_path)
    step = t.recorded_steps[0]
    llm_resp = step["llm_response"]
    assert "choices" in llm_resp
    assert "usage" in llm_resp
    assert llm_resp["choices"][0]["finish_reason"] == "stop"
    assert llm_resp["usage"]["prompt_tokens"] == 10
    assert llm_resp["usage"]["completion_tokens"] == 5
    # Original Anthropic payload preserved.
    assert llm_resp["_anthropic"]["stop_reason"] == "end_turn"


def test_wrap_anthropic_cost_accounting(tmp_path):
    """Cost is > 0 for known models."""
    trace_path = str(tmp_path / "anth_cost.sb")
    fake = _FakeAnthropic.with_response(dict(_SIMPLE_RESPONSE))
    with record(trace_path) as rec:
        client = wrap_anthropic(fake, rec)
        client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "hi"}],
        )
    t = replay(trace_path)
    assert t.recorded_steps[0]["cost_usd"] > 0


def test_wrap_anthropic_replay_cache_hit(tmp_path):
    """Second replay serves from cache without executor calls."""
    trace_path = str(tmp_path / "anth_cache.sb")
    fake = _FakeAnthropic.with_response(dict(_SIMPLE_RESPONSE))
    with record(trace_path) as rec:
        client = wrap_anthropic(fake, rec)
        client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "hi"}],
        )
    _replay_cache_only(trace_path)


def test_wrap_anthropic_multi_turn(tmp_path):
    """Multi-turn conversation records each turn as a separate step."""
    trace_path = str(tmp_path / "anth_multi.sb")
    fake = _FakeAnthropic.with_multi([
        dict(_SIMPLE_RESPONSE),
        {**_SIMPLE_RESPONSE, "id": "msg_002_turn2",
         "content": [{"type": "text", "text": "Turn 2 reply."}]},
    ])
    with record(trace_path) as rec:
        client = wrap_anthropic(fake, rec)
        r1 = client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "turn 1"}],
        )
        r2 = client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[
                {"role": "user", "content": "turn 1"},
                {"role": "assistant", "content": r1.content[0].text},
                {"role": "user", "content": "turn 2"},
            ],
        )
    t = replay(trace_path)
    assert len(t.recorded_steps) == 2
    assert t.recorded_steps[1]["inputs"]["messages"][-1]["content"] == "turn 2"
    assert r2.content[0].text == "Turn 2 reply."


def test_wrap_anthropic_rejects_non_anthropic_client(tmp_path):
    with record(str(tmp_path / "x.sb")) as rec:
        with pytest.raises(TypeError, match="lacks .messages"):
            wrap_anthropic(object(), rec)


# =====================================================================
# § 3  Tool use
# =====================================================================


def test_wrap_anthropic_tool_use_finish_reason(tmp_path):
    """tool_use stop_reason maps to finish_reason 'tool_calls'."""
    trace_path = str(tmp_path / "anth_tool.sb")
    fake = _FakeAnthropic.with_response(dict(_TOOL_USE_RESPONSE))
    with record(trace_path) as rec:
        client = wrap_anthropic(fake, rec)
        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "What's the weather?"}],
        )
    assert resp.stop_reason == "tool_use"
    t = replay(trace_path)
    llm_resp = t.recorded_steps[0]["llm_response"]
    assert llm_resp["choices"][0]["finish_reason"] == "tool_calls"


def test_wrap_anthropic_tool_use_block_canonicalised(tmp_path):
    """tool_use block is present in the AnthropicMessage returned."""
    trace_path = str(tmp_path / "anth_tool2.sb")
    fake = _FakeAnthropic.with_response(dict(_TOOL_USE_RESPONSE))
    with record(trace_path) as rec:
        client = wrap_anthropic(fake, rec)
        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "weather?"}],
        )
    assert len(resp.content) == 1
    block = resp.content[0]
    assert block.type == "tool_use"
    assert block.id == "toolu_abc123"
    assert block.name == "get_weather"
    assert block.input == {"city": "Paris"}


def test_wrap_anthropic_tool_use_openai_tool_calls(tmp_path):
    """tool_use blocks project to OpenAI tool_calls in llm_response."""
    trace_path = str(tmp_path / "anth_tc.sb")
    fake = _FakeAnthropic.with_response(dict(_TOOL_USE_RESPONSE))
    with record(trace_path) as rec:
        client = wrap_anthropic(fake, rec)
        client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "weather?"}],
        )
    t = replay(trace_path)
    tc = t.recorded_steps[0]["llm_response"]["choices"][0]["message"]["tool_calls"]
    assert tc is not None and len(tc) == 1
    assert tc[0]["id"] == "toolu_abc123"
    assert tc[0]["function"]["name"] == "get_weather"
    assert tc[0]["function"]["arguments"] == {"city": "Paris"}


def test_wrap_anthropic_mixed_text_and_tool(tmp_path):
    """Responses with both text and tool_use blocks record correctly."""
    trace_path = str(tmp_path / "anth_mixed.sb")
    fake = _FakeAnthropic.with_response(dict(_MIXED_RESPONSE))
    with record(trace_path) as rec:
        client = wrap_anthropic(fake, rec)
        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "search something"}],
        )
    assert len(resp.content) == 2
    assert resp.content[0].type == "text"
    assert resp.content[1].type == "tool_use"
    _replay_cache_only(trace_path)


# =====================================================================
# § 4  Streaming
# =====================================================================


def test_wrap_anthropic_stream_records_one_step(tmp_path):
    trace_path = str(tmp_path / "anth_stream.sb")
    events = _make_events_text(text="Hello streaming!")
    fake = _FakeAnthropic.with_stream(events)
    with record(trace_path) as rec:
        client = wrap_anthropic(fake, rec)
        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    # Returns StreamedLLMResponse — Step 97
    from stepback.streaming import StreamedLLMResponse
    assert isinstance(resp, StreamedLLMResponse)
    # native holds the AnthropicMessage for backward compat
    assert isinstance(resp.native, AnthropicMessage)
    assert resp.native.content[0].text == "Hello streaming!"
    t = replay(trace_path)
    assert len(t.recorded_steps) == 1
    assert t.recorded_steps[0]["step_kind"] == "llm_call"


def test_wrap_anthropic_stream_assembled_response(tmp_path):
    trace_path = str(tmp_path / "anth_stream2.sb")
    events = _make_events_text(text="Streaming text.", input_tokens=12, output_tokens=4)
    fake = _FakeAnthropic.with_stream(events)
    with record(trace_path) as rec:
        client = wrap_anthropic(fake, rec)
        client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    t = replay(trace_path)
    step = t.recorded_steps[0]
    assert step["llm_response"]["choices"][0]["message"]["content"] == "Streaming text."
    assert step["llm_response"]["choices"][0]["finish_reason"] == "stop"
    assert step["llm_response"]["usage"]["prompt_tokens"] == 12
    assert step["llm_response"]["usage"]["completion_tokens"] == 4


def test_wrap_anthropic_stream_replay_cache(tmp_path):
    trace_path = str(tmp_path / "anth_stream_cache.sb")
    fake = _FakeAnthropic.with_stream(_make_events_text(text="cached."))
    with record(trace_path) as rec:
        client = wrap_anthropic(fake, rec)
        client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    _replay_cache_only(trace_path)


def test_wrap_anthropic_stream_tool_use(tmp_path):
    """Streaming tool_use blocks are assembled and recorded correctly."""
    trace_path = str(tmp_path / "anth_stream_tool.sb")
    fake = _FakeAnthropic.with_stream(_make_events_tool_use())
    with record(trace_path) as rec:
        client = wrap_anthropic(fake, rec)
        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "weather?"}],
            stream=True,
        )
    assert resp.native.content[0].type == "tool_use"
    assert resp.native.content[0].name == "get_weather"
    assert resp.native.content[0].input == {"city": "Paris"}
    t = replay(trace_path)
    tc = t.recorded_steps[0]["llm_response"]["choices"][0]["message"]["tool_calls"]
    assert tc is not None
    assert tc[0]["function"]["name"] == "get_weather"
    _replay_cache_only(trace_path)


def test_wrap_anthropic_stream_thinking_blocks_excluded(tmp_path):
    """Thinking blocks in stream are excluded; text block reaches the caller."""
    trace_path = str(tmp_path / "anth_think.sb")
    fake = _FakeAnthropic.with_stream(_make_events_thinking_plus_text())
    with record(trace_path) as rec:
        client = wrap_anthropic(fake, rec)
        resp = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            messages=[{"role": "user", "content": "42?"}],
            stream=True,
        )
    assert len(resp.native.content) == 1
    assert resp.native.content[0].type == "text"
    assert resp.native.content[0].text == "The answer is 42."


# =====================================================================
# § 5  AnthropicShimContract hooks
# =====================================================================


def test_anthropic_shim_contract_stream_request():
    """stream_request is implemented (not NotImplementedError)."""
    contract = AnthropicShimContract()

    class _FakeMsgs:
        def create(self, *, model, messages, system=None, max_tokens=1024, **kw):
            return iter(_make_events_text(text="contract-stream"))

    class _FakeClient:
        def __init__(self):
            self.messages = _FakeMsgs()

    messages = [{"role": "user", "content": "hi"}]
    result = contract.stream_request(
        _FakeClient(), messages, model="claude-3-5-haiku-20241022"
    )
    assert "choices" in result
    assert result["choices"][0]["message"]["content"] == "contract-stream"


def test_anthropic_shim_contract_async_request():
    """async_request is implemented (not NotImplementedError)."""
    contract = AnthropicShimContract()

    class _FakeAsyncMsgs:
        async def create(self, *, model, messages, system=None, max_tokens=1024, **kw):
            return dict(_SIMPLE_RESPONSE)

    class _FakeAsyncClient:
        def __init__(self):
            self.messages = _FakeAsyncMsgs()

    messages = [{"role": "user", "content": "async hi"}]
    result = asyncio.run(
        contract.async_request(
            _FakeAsyncClient(), messages, model="claude-3-5-haiku-20241022"
        )
    )
    assert "choices" in result
    assert result["choices"][0]["finish_reason"] == "stop"


def test_anthropic_shim_contract_canonical_request_no_system():
    contract = AnthropicShimContract()
    msgs = [{"role": "user", "content": "hi"}]
    result = contract.canonical_request(messages=msgs)
    assert result == msgs


def test_anthropic_shim_contract_canonical_request_with_system():
    contract = AnthropicShimContract()
    msgs = [{"role": "user", "content": "hi"}]
    result = contract.canonical_request(messages=msgs, system="You are helpful.")
    assert result[0] == {"role": "system", "content": "You are helpful."}
    assert result[1] == msgs[0]


def test_anthropic_shim_contract_canonical_response():
    contract = AnthropicShimContract()
    result = contract.canonical_response(dict(_SIMPLE_RESPONSE))
    assert "choices" in result
    assert result["choices"][0]["message"]["content"] == "Hello!"
    assert result["usage"]["prompt_tokens"] == 10


def test_anthropic_shim_contract_make_executor():
    """make_executor returns a callable."""
    contract = AnthropicShimContract()
    fake = _FakeAnthropic.with_response(dict(_SIMPLE_RESPONSE))
    executor = contract.make_executor(fake)
    assert callable(executor)


def test_anthropic_shim_contract_version_probe_missing_sdk():
    """version_probe returns None when anthropic is not installed."""
    import sys
    contract = AnthropicShimContract()
    # Temporarily hide the anthropic module to test the missing-SDK path.
    original = sys.modules.get("anthropic", None)
    sys.modules["anthropic"] = None  # type: ignore[assignment]
    try:
        result = contract.version_probe(None)
        assert result is None
    finally:
        if original is None:
            del sys.modules["anthropic"]
        else:
            sys.modules["anthropic"] = original


def test_other_providers_stream_raises():
    """Non-OpenAI / non-Anthropic contracts still raise NotImplementedError."""
    from stepback.shims import BedrockShimContract, GeminiShimContract
    for cls in (BedrockShimContract, GeminiShimContract):
        with pytest.raises(NotImplementedError):
            cls().stream_request(None, [])


def test_other_providers_async_raises():
    """Non-OpenAI / non-Anthropic contracts still raise NotImplementedError."""
    from stepback.shims import BedrockShimContract, GeminiShimContract
    for cls in (BedrockShimContract, GeminiShimContract):
        with pytest.raises(NotImplementedError):
            asyncio.run(cls().async_request(None, []))


# =====================================================================
# § 6  Async clients
# =====================================================================


@dataclass
class _FakeAsyncAnthropicMessages:
    raw_response: dict
    calls: list = field(default_factory=list)

    async def create(self, *, model: str, messages: list,
                     system: Optional[str] = None,
                     max_tokens: int = 1024,
                     **kwargs: Any) -> dict:
        self.calls.append({
            "model": model, "messages": messages,
            "system": system, "kwargs": kwargs,
        })
        return dict(self.raw_response)


@dataclass
class _FakeAsyncAnthropic:
    messages: _FakeAsyncAnthropicMessages

    @classmethod
    def with_response(cls, raw: dict) -> "_FakeAsyncAnthropic":
        return cls(messages=_FakeAsyncAnthropicMessages(raw_response=raw))


def test_wrap_anthropic_async_returns_correct_type(tmp_path):
    trace_path = str(tmp_path / "anth_async.sb")
    fake = _FakeAsyncAnthropic.with_response(dict(_SIMPLE_RESPONSE))

    async def _run():
        with record(trace_path) as rec:
            client = wrap_anthropic_async(fake, rec)
            assert isinstance(client, AsyncWrappedAnthropic)
            return await client.messages.create(
                model="claude-3-5-haiku-20241022",
                messages=[{"role": "user", "content": "async ping"}],
            )

    resp = asyncio.run(_run())
    assert isinstance(resp, AnthropicMessage)
    assert resp.content[0].text == "Hello!"


def test_wrap_anthropic_async_records_step(tmp_path):
    trace_path = str(tmp_path / "anth_async_rec.sb")
    fake = _FakeAsyncAnthropic.with_response(dict(_SIMPLE_RESPONSE))

    async def _run():
        with record(trace_path) as rec:
            client = wrap_anthropic_async(fake, rec)
            await client.messages.create(
                model="claude-3-5-haiku-20241022",
                messages=[{"role": "user", "content": "async hi"}],
            )

    asyncio.run(_run())
    t = replay(trace_path)
    assert len(t.recorded_steps) == 1
    assert t.recorded_steps[0]["step_kind"] == "llm_call"


def test_wrap_anthropic_async_system_prompt(tmp_path):
    trace_path = str(tmp_path / "anth_async_sys.sb")
    fake = _FakeAsyncAnthropic.with_response(dict(_SIMPLE_RESPONSE))

    async def _run():
        with record(trace_path) as rec:
            client = wrap_anthropic_async(fake, rec)
            await client.messages.create(
                model="claude-3-5-haiku-20241022",
                system="Be concise.",
                messages=[{"role": "user", "content": "hi"}],
            )

    asyncio.run(_run())
    t = replay(trace_path)
    step = t.recorded_steps[0]
    assert step["inputs"]["messages"][0]["role"] == "system"
    assert "concise" in step["inputs"]["messages"][0]["content"]


def test_wrap_anthropic_async_replay_cache_hit(tmp_path):
    trace_path = str(tmp_path / "anth_async_cache.sb")
    fake = _FakeAsyncAnthropic.with_response(dict(_SIMPLE_RESPONSE))

    async def _run():
        with record(trace_path) as rec:
            client = wrap_anthropic_async(fake, rec)
            await client.messages.create(
                model="claude-3-5-haiku-20241022",
                messages=[{"role": "user", "content": "cache me"}],
            )

    asyncio.run(_run())
    _replay_cache_only(trace_path)


def test_wrap_anthropic_async_rejects_non_anthropic_client(tmp_path):
    with record(str(tmp_path / "x.sb")) as rec:
        with pytest.raises(TypeError, match="lacks .messages"):
            wrap_anthropic_async(object(), rec)


def test_wrap_anthropic_async_canonical_shape(tmp_path):
    """Async recording stores the OpenAI canonical shape in the trace."""
    trace_path = str(tmp_path / "anth_async_shape.sb")
    fake = _FakeAsyncAnthropic.with_response(dict(_SIMPLE_RESPONSE))

    async def _run():
        with record(trace_path) as rec:
            client = wrap_anthropic_async(fake, rec)
            await client.messages.create(
                model="claude-3-5-haiku-20241022",
                messages=[{"role": "user", "content": "hi"}],
            )

    asyncio.run(_run())
    t = replay(trace_path)
    llm_resp = t.recorded_steps[0]["llm_response"]
    assert "choices" in llm_resp
    assert llm_resp["choices"][0]["finish_reason"] == "stop"
    assert llm_resp["_anthropic"]["stop_reason"] == "end_turn"


# =====================================================================
# § 7  anthropic_executor replay
# =====================================================================


def test_anthropic_executor_strips_system_from_messages(tmp_path):
    """anthropic_executor separates system message from messages list."""
    trace_path = str(tmp_path / "anth_exec.sb")
    with record(trace_path) as rec:
        fake = _FakeAnthropic.with_response(dict(_SIMPLE_RESPONSE))
        client = wrap_anthropic(fake, rec)
        client.messages.create(
            model="claude-3-5-haiku-20241022",
            system="My system.",
            messages=[{"role": "user", "content": "hello"}],
        )

    t = replay(trace_path)
    fake2 = _FakeAnthropic.with_response(dict(_SIMPLE_RESPONSE))
    from stepback.substitutions import PromptSubstitution
    exec_ = Executor(llm=anthropic_executor(fake2))
    sub = PromptSubstitution(
        at_step="step:1",
        new_messages=[
            {"role": "system", "content": "My system."},
            {"role": "user", "content": "hello-changed"},
        ],
    )
    t.run_replay(subs=SubstitutionSet([sub]), executor=exec_)
    call = fake2.messages.calls[0]
    # System should be passed separately, not in messages list.
    assert call["system"] == "My system."
    assert all(m["role"] != "system" for m in call["messages"])
    assert call["messages"][0]["content"] == "hello-changed"
