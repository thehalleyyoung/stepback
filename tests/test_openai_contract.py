"""OpenAI contract tests — Step 90 in ``100_STEPS.md``.

Covers:
- Streaming: ``_accumulate_streaming_chunks`` unit tests, ``wrap_openai``
  with ``stream=True``, cache replay after streaming, parallel streaming
  tool calls, ``OpenAIShimContract.stream_request`` hook.
- Async clients: ``wrap_openai_async`` recording, replay cache hit,
  ``OpenAIShimContract.async_request`` hook.
- Responses API: ``wrap_openai_responses`` recording + replay,
  ``_canonicalise_openai_responses_output`` coercion, tool-call output,
  ``OpenAIResponsesOutput.output_text``.
- OpenAI-compatible endpoints: ``wrap_openai`` accepts any client whose
  ``chat.completions.create`` returns the OpenAI wire format (Together,
  Fireworks, vLLM, Ollama, etc.).
- ABC hooks: ``OpenAIShimContract.stream_request`` and ``async_request``
  are implemented (not ``NotImplementedError``); other providers still
  raise ``NotImplementedError`` for those hooks.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional

import pytest

from stepback import record, replay
from stepback.replay import Executor
from stepback.shims import (
    AsyncWrappedOpenAI,
    OpenAIResponsesOutput,
    OpenAIShimContract,
    _accumulate_streaming_chunks,
    _canonicalise_openai_responses_output,
    wrap_openai,
    wrap_openai_async,
    wrap_openai_responses,
    AnthropicShimContract,
    BedrockShimContract,
    GeminiShimContract,
)
from stepback.substitutions import SubstitutionSet


# =====================================================================
# Fake client helpers
# =====================================================================


@dataclass
class _FakeCompletions:
    responses: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    _idx: int = field(default=0, init=False, repr=False)

    def create(self, *, model: str, messages: list, **kwargs: Any):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        resp = self.responses[self._idx % len(self.responses)]
        self._idx += 1
        # If stream=True, return the resp as-is (caller expects iterable).
        return resp


@dataclass
class _FakeChat:
    completions: _FakeCompletions


@dataclass
class _FakeOAIClient:
    chat: _FakeChat

    @classmethod
    def with_response(cls, raw: dict) -> "_FakeOAIClient":
        return cls(chat=_FakeChat(completions=_FakeCompletions(responses=[raw])))

    @classmethod
    def with_chunks(cls, chunks: list) -> "_FakeOAIClient":
        return cls(chat=_FakeChat(completions=_FakeCompletions(responses=[iter(chunks)])))

    @classmethod
    def with_multi(cls, responses: list) -> "_FakeOAIClient":
        """Returns each response in order (streaming or non-streaming)."""
        return cls(chat=_FakeChat(completions=_FakeCompletions(responses=responses)))


@dataclass
class _FakeAsyncCompletions:
    raw_response: dict
    calls: list = field(default_factory=list)

    async def create(self, *, model: str, messages: list, **kwargs: Any) -> dict:
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        return dict(self.raw_response)


@dataclass
class _FakeAsyncChat:
    completions: _FakeAsyncCompletions


@dataclass
class _FakeAsyncOAIClient:
    chat: _FakeAsyncChat

    @classmethod
    def with_response(cls, raw: dict) -> "_FakeAsyncOAIClient":
        return cls(chat=_FakeAsyncChat(completions=_FakeAsyncCompletions(raw_response=raw)))


@dataclass
class _FakeResponsesAPI:
    raw_response: dict
    calls: list = field(default_factory=list)

    def create(self, *, model: str, input: Any, **kwargs: Any):  # noqa: A002
        self.calls.append({"model": model, "input": input, "kwargs": kwargs})
        return dict(self.raw_response)


@dataclass
class _FakeOAIWithResponses:
    """Fake client that has both chat.completions and responses."""
    chat: _FakeChat
    responses: _FakeResponsesAPI

    @classmethod
    def build(cls, chat_raw: dict, responses_raw: dict) -> "_FakeOAIWithResponses":
        return cls(
            chat=_FakeChat(completions=_FakeCompletions(responses=[chat_raw])),
            responses=_FakeResponsesAPI(raw_response=responses_raw),
        )


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
# § 1  _accumulate_streaming_chunks — unit tests
# =====================================================================


def test_accumulate_text_chunks():
    chunks = [
        {"id": "c1", "model": "gpt-4o", "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}
        ]},
        {"id": "c1", "model": "gpt-4o", "choices": [
            {"index": 0, "delta": {"content": "lo"}, "finish_reason": None}
        ]},
        {"id": "c1", "model": "gpt-4o", "choices": [
            {"index": 0, "delta": {}, "finish_reason": "stop"}
        ]},
    ]
    result = _accumulate_streaming_chunks(iter(chunks))
    assert result["id"] == "c1"
    assert result["model"] == "gpt-4o"
    choices = result["choices"]
    assert len(choices) == 1
    msg = choices[0]["message"]
    assert msg["content"] == "Hello"
    assert msg["role"] == "assistant"
    assert choices[0]["finish_reason"] == "stop"
    assert msg["tool_calls"] is None


def test_accumulate_tool_call_chunks():
    """Parallel tool call deltas are accumulated by (choice_index, tc_index)."""
    chunks = [
        {"id": "s1", "model": "gpt-4o", "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": None,
                "tool_calls": [{"index": 0, "id": "call_abc", "type": "function",
                                 "function": {"name": "get_weather", "arguments": ""}}]},
             "finish_reason": None}
        ]},
        {"id": "s1", "model": "gpt-4o", "choices": [
            {"index": 0, "delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": '{"city"'}}]},
             "finish_reason": None}
        ]},
        {"id": "s1", "model": "gpt-4o", "choices": [
            {"index": 0, "delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": ':"Paris"}'}}]},
             "finish_reason": None}
        ]},
        {"id": "s1", "model": "gpt-4o", "choices": [
            {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
        ]},
    ]
    result = _accumulate_streaming_chunks(iter(chunks))
    msg = result["choices"][0]["message"]
    assert msg["content"] is None
    tc = msg["tool_calls"]
    assert tc is not None and len(tc) == 1
    assert tc[0]["id"] == "call_abc"
    assert tc[0]["function"]["name"] == "get_weather"
    assert tc[0]["function"]["arguments"] == '{"city":"Paris"}'
    assert result["choices"][0]["finish_reason"] == "tool_calls"


def test_accumulate_parallel_tool_calls():
    """Two parallel tool calls with interleaved deltas."""
    chunks = [
        {"id": "p1", "model": "gpt-4o", "choices": [
            {"index": 0, "delta": {"role": "assistant",
                "tool_calls": [
                    {"index": 0, "id": "call_A", "type": "function",
                     "function": {"name": "fn_a", "arguments": ""}},
                    {"index": 1, "id": "call_B", "type": "function",
                     "function": {"name": "fn_b", "arguments": ""}},
                ]}, "finish_reason": None}
        ]},
        {"id": "p1", "model": "gpt-4o", "choices": [
            {"index": 0, "delta": {
                "tool_calls": [
                    {"index": 0, "function": {"arguments": '{"a":1}'}},
                    {"index": 1, "function": {"arguments": '{"b":2}'}},
                ]}, "finish_reason": None}
        ]},
        {"id": "p1", "model": "gpt-4o", "choices": [
            {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
        ]},
    ]
    result = _accumulate_streaming_chunks(iter(chunks))
    tc = result["choices"][0]["message"]["tool_calls"]
    assert tc is not None and len(tc) == 2
    assert tc[0]["function"]["arguments"] == '{"a":1}'
    assert tc[1]["function"]["arguments"] == '{"b":2}'


def test_accumulate_usage_in_last_chunk():
    """Usage may arrive in a trailing chunk with choices=[]."""
    chunks = [
        {"id": "u1", "model": "gpt-4o", "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": "Hi"},
             "finish_reason": "stop"}
        ]},
        {"id": "u1", "model": "gpt-4o", "choices": [],
         "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}},
    ]
    result = _accumulate_streaming_chunks(iter(chunks))
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 2
    assert result["usage"]["total_tokens"] == 12


def test_accumulate_duck_typed_chunk():
    """Chunks that expose .model_dump() are coerced correctly."""
    from dataclasses import dataclass as dc

    @dc
    class _Chunk:
        data: dict
        def model_dump(self):
            return dict(self.data)

    chunks = [
        _Chunk({"id": "d1", "model": "gpt-4o", "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": "ok"},
             "finish_reason": "stop"}
        ]}),
    ]
    result = _accumulate_streaming_chunks(iter(chunks))
    assert result["choices"][0]["message"]["content"] == "ok"


# =====================================================================
# § 2  wrap_openai with stream=True
# =====================================================================

_STREAMING_CHUNKS = [
    {"id": "sc1", "model": "gpt-4o-mini-2024-07-18", "choices": [
        {"index": 0, "delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}
    ]},
    {"id": "sc1", "model": "gpt-4o-mini-2024-07-18", "choices": [
        {"index": 0, "delta": {"content": "."}, "finish_reason": None}
    ]},
    {"id": "sc1", "model": "gpt-4o-mini-2024-07-18", "choices": [
        {"index": 0, "delta": {}, "finish_reason": "stop"}
    ]},
    {"id": "sc1", "model": "gpt-4o-mini-2024-07-18", "choices": [],
     "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
]


def test_wrap_openai_stream_records_one_step(tmp_path):
    fake = _FakeOAIClient.with_chunks(list(_STREAMING_CHUNKS))
    trace_path = str(tmp_path / "stream.sb")
    with record(trace_path) as rec:
        client = wrap_openai(fake, rec, default_model="gpt-4o-mini-2024-07-18")
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    # Returns StreamedLLMResponse (iterable of chunks) — Step 97
    from stepback.streaming import StreamedLLMResponse
    assert isinstance(resp, StreamedLLMResponse)
    # native holds the assembled OpenAIChatCompletion for backward compat
    assert resp.native.choices[0].message.content == "Hi."
    # assembled_response is the canonical dict
    assert resp.assembled_response["choices"][0]["message"]["content"] == "Hi."
    t = replay(trace_path)
    assert len(t.recorded_steps) == 1
    assert t.recorded_steps[0]["step_kind"] == "llm_call"


def test_wrap_openai_stream_assembled_response_correct(tmp_path):
    fake = _FakeOAIClient.with_chunks(list(_STREAMING_CHUNKS))
    trace_path = str(tmp_path / "stream2.sb")
    with record(trace_path) as rec:
        client = wrap_openai(fake, rec, default_model="gpt-4o-mini-2024-07-18")
        client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    t = replay(trace_path)
    llm_resp = t.recorded_steps[0]["llm_response"]
    assert llm_resp["choices"][0]["message"]["content"] == "Hi."
    assert llm_resp["choices"][0]["finish_reason"] == "stop"
    assert llm_resp["usage"]["prompt_tokens"] == 5


def test_wrap_openai_stream_replay_serves_from_cache(tmp_path):
    fake = _FakeOAIClient.with_chunks(list(_STREAMING_CHUNKS))
    trace_path = str(tmp_path / "stream_cache.sb")
    with record(trace_path) as rec:
        client = wrap_openai(fake, rec, default_model="gpt-4o-mini-2024-07-18")
        client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    _replay_cache_only(trace_path)


def test_wrap_openai_stream_tool_calls(tmp_path):
    """Streaming with tool call deltas assembles correctly and records."""
    stream_chunks = [
        {"id": "tc1", "model": "gpt-4o", "choices": [
            {"index": 0, "delta": {"role": "assistant",
                "tool_calls": [{"index": 0, "id": "call_xyz", "type": "function",
                                 "function": {"name": "get_weather", "arguments": ""}}]},
             "finish_reason": None}
        ]},
        {"id": "tc1", "model": "gpt-4o", "choices": [
            {"index": 0, "delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": '{"city":"Paris"}'}}]},
             "finish_reason": None}
        ]},
        {"id": "tc1", "model": "gpt-4o", "choices": [
            {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
        ]},
        {"id": "tc1", "model": "gpt-4o", "choices": [],
         "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20}},
    ]
    fake = _FakeOAIClient.with_chunks(stream_chunks)
    trace_path = str(tmp_path / "stream_tool.sb")
    with record(trace_path) as rec:
        client = wrap_openai(fake, rec, default_model="gpt-4o")
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "weather in paris?"}],
            stream=True,
        )
    # native holds the assembled OpenAIChatCompletion
    assert resp.native.choices[0].message.tool_calls[0]["function"]["name"] == "get_weather"
    t = replay(trace_path)
    llm_resp = t.recorded_steps[0]["llm_response"]
    tc = llm_resp["choices"][0]["message"]["tool_calls"]
    assert tc[0]["function"]["arguments"] == '{"city":"Paris"}'
    _replay_cache_only(trace_path)


# =====================================================================
# § 3  OpenAIShimContract.stream_request hook
# =====================================================================


def test_openai_shim_contract_stream_request():
    """stream_request is implemented (not NotImplementedError)."""
    contract = OpenAIShimContract()
    chunks = [
        {"id": "h1", "model": "gpt-4o", "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": "hi"},
             "finish_reason": "stop"}
        ]},
    ]

    class _FakeCompletionsDirect:
        def create(self, *, model, messages, **kwargs):
            return iter(chunks)

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletionsDirect()

    class _FakeClientDirect:
        def __init__(self):
            self.chat = _FakeChat()

    result = contract.stream_request(
        _FakeClientDirect(),
        [{"role": "user", "content": "hello"}],
        model="gpt-4o",
    )
    assert isinstance(result, dict)
    assert result["choices"][0]["message"]["content"] == "hi"


def test_other_providers_still_raise_not_implemented_for_stream_request():
    """Bedrock / Gemini stream_request still raises NotImplementedError.

    Anthropic now implements stream_request so it is excluded from this check.
    """
    for contract in [BedrockShimContract(), GeminiShimContract()]:
        with pytest.raises(NotImplementedError):
            contract.stream_request(object(), [], model="m")


# =====================================================================
# § 4  Async client — wrap_openai_async
# =====================================================================

_ASYNC_RAW = {
    "id": "chatcmpl-async001",
    "object": "chat.completion",
    "created": 1726000000,
    "model": "gpt-4o-mini-2024-07-18",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "Async hi.", "tool_calls": None}}],
    "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
}


def test_wrap_openai_async_returns_async_wrapped(tmp_path):
    fake = _FakeAsyncOAIClient.with_response(_ASYNC_RAW)
    trace_path = str(tmp_path / "async.sb")

    async def _run():
        with record(trace_path) as rec:
            client = wrap_openai_async(fake, rec, default_model="gpt-4o-mini-2024-07-18")
            return await client.chat.completions.create(
                messages=[{"role": "user", "content": "hi async"}]
            )

    resp = asyncio.run(_run())
    assert isinstance(resp.choices[0].message.content, str)
    assert "Async hi." in resp.choices[0].message.content


def test_wrap_openai_async_records_one_step(tmp_path):
    fake = _FakeAsyncOAIClient.with_response(_ASYNC_RAW)
    trace_path = str(tmp_path / "async_step.sb")

    async def _run():
        with record(trace_path) as rec:
            client = wrap_openai_async(fake, rec, default_model="gpt-4o-mini-2024-07-18")
            await client.chat.completions.create(
                messages=[{"role": "user", "content": "hi async"}]
            )

    asyncio.run(_run())
    t = replay(trace_path)
    assert len(t.recorded_steps) == 1
    assert t.recorded_steps[0]["step_kind"] == "llm_call"
    assert t.recorded_steps[0]["llm_response"]["id"] == "chatcmpl-async001"


def test_wrap_openai_async_replay_serves_from_cache(tmp_path):
    fake = _FakeAsyncOAIClient.with_response(_ASYNC_RAW)
    trace_path = str(tmp_path / "async_cache.sb")

    async def _run():
        with record(trace_path) as rec:
            client = wrap_openai_async(fake, rec, default_model="gpt-4o-mini-2024-07-18")
            await client.chat.completions.create(
                messages=[{"role": "user", "content": "hi async"}]
            )

    asyncio.run(_run())
    _replay_cache_only(trace_path)


def test_wrap_openai_async_is_AsyncWrappedOpenAI(tmp_path):
    fake = _FakeAsyncOAIClient.with_response(_ASYNC_RAW)
    with record(str(tmp_path / "x.sb")) as rec:
        client = wrap_openai_async(fake, rec, default_model="gpt-4o-mini-2024-07-18")
    assert isinstance(client, AsyncWrappedOpenAI)


def test_wrap_openai_async_missing_completions_raises():
    with pytest.raises(TypeError, match="lacks .chat.completions"):
        wrap_openai_async(object(), object(), default_model="m")


# =====================================================================
# § 5  OpenAIShimContract.async_request hook
# =====================================================================


def test_openai_shim_contract_async_request_is_coroutine():
    """async_request is implemented and is actually async."""
    import inspect
    contract = OpenAIShimContract()
    # Should be an async method (coroutine function)
    assert inspect.iscoroutinefunction(contract.async_request)


def test_openai_shim_contract_async_request_runs():
    """async_request returns a canonical dict."""
    raw = dict(_ASYNC_RAW)

    class _FakeAsyncCompletionsDirect:
        async def create(self, *, model, messages, **kwargs):
            return dict(raw)

    class _FakeAsyncChatDirect:
        def __init__(self):
            self.completions = _FakeAsyncCompletionsDirect()

    class _FakeAsyncClientDirect:
        def __init__(self):
            self.chat = _FakeAsyncChatDirect()

    contract = OpenAIShimContract()
    result = asyncio.run(
        contract.async_request(
            _FakeAsyncClientDirect(),
            [{"role": "user", "content": "hello"}],
            model="gpt-4o-mini-2024-07-18",
        )
    )
    assert isinstance(result, dict)
    assert result["choices"][0]["message"]["content"] == "Async hi."


def test_other_providers_still_raise_not_implemented_for_async_request():
    """Bedrock / Gemini async_request still raises NotImplementedError.

    Anthropic now implements async_request so it is excluded from this check.
    """
    async def _call(contract):
        await contract.async_request(object(), [], model="m")

    for contract in [BedrockShimContract(), GeminiShimContract()]:
        with pytest.raises(NotImplementedError):
            asyncio.run(_call(contract))


# =====================================================================
# § 6  OpenAI Responses API — _canonicalise_openai_responses_output
# =====================================================================

_RESPONSES_TEXT_RAW = {
    "id": "resp_text001",
    "object": "response",
    "model": "gpt-4o-mini-2024-07-18",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "id": "msg_001",
            "role": "assistant",
            "content": [{"type": "text", "text": "Responses hi."}],
        }
    ],
    "usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
}

_RESPONSES_TOOL_RAW = {
    "id": "resp_tool001",
    "object": "response",
    "model": "gpt-4o-2024-11-20",
    "status": "completed",
    "output": [
        {
            "type": "function_call",
            "id": "call_resp1",
            "name": "get_weather",
            "arguments": '{"city":"London"}',
        }
    ],
    "usage": {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
}


def test_canonicalise_responses_output_text():
    canonical = _canonicalise_openai_responses_output(_RESPONSES_TEXT_RAW)
    assert canonical["id"] == "resp_text001"
    assert canonical["model"] == "gpt-4o-mini-2024-07-18"
    ch0 = canonical["choices"][0]
    assert ch0["finish_reason"] == "stop"
    assert ch0["message"]["content"] == "Responses hi."
    # _strip_none removes None values — no tool_calls key is equivalent to None
    assert ch0["message"].get("tool_calls") is None
    assert canonical["usage"]["prompt_tokens"] == 10
    assert canonical["usage"]["completion_tokens"] == 3


def test_canonicalise_responses_output_tool_call():
    canonical = _canonicalise_openai_responses_output(_RESPONSES_TOOL_RAW)
    ch0 = canonical["choices"][0]
    assert ch0["finish_reason"] == "tool_calls"
    tc = ch0["message"]["tool_calls"]
    assert tc is not None and len(tc) == 1
    assert tc[0]["function"]["name"] == "get_weather"
    assert tc[0]["function"]["arguments"] == '{"city":"London"}'


def test_canonicalise_responses_output_model_dump():
    """Accepts objects with .model_dump()."""
    from dataclasses import dataclass as dc

    @dc
    class _Resp:
        data: dict
        def model_dump(self):
            return dict(self.data)

    canonical = _canonicalise_openai_responses_output(_Resp(data=_RESPONSES_TEXT_RAW))
    assert canonical["choices"][0]["message"]["content"] == "Responses hi."


def test_openai_responses_output_output_text():
    out = OpenAIResponsesOutput.from_dict(_RESPONSES_TEXT_RAW)
    assert out.output_text == "Responses hi."
    assert out.id == "resp_text001"


def test_openai_responses_output_tool_call_text_is_none():
    out = OpenAIResponsesOutput.from_dict(_RESPONSES_TOOL_RAW)
    # No message output → output_text is None
    assert out.output_text is None


# =====================================================================
# § 7  wrap_openai_responses — recording + replay
# =====================================================================


def test_wrap_openai_responses_records_one_step(tmp_path):
    fake = _FakeOAIWithResponses.build(
        chat_raw={},  # not used
        responses_raw=_RESPONSES_TEXT_RAW,
    )
    trace_path = str(tmp_path / "resp.sb")
    with record(trace_path) as rec:
        client = wrap_openai_responses(fake, rec, default_model="gpt-4o-mini-2024-07-18")
        resp = client.responses.create(
            input=[{"role": "user", "content": "hi responses"}]
        )
    assert isinstance(resp, OpenAIResponsesOutput)
    assert resp.output_text == "Responses hi."
    t = replay(trace_path)
    assert len(t.recorded_steps) == 1
    assert t.recorded_steps[0]["step_kind"] == "llm_call"


def test_wrap_openai_responses_replay_from_cache(tmp_path):
    fake = _FakeOAIWithResponses.build(
        chat_raw={},
        responses_raw=_RESPONSES_TEXT_RAW,
    )
    trace_path = str(tmp_path / "resp_cache.sb")
    with record(trace_path) as rec:
        client = wrap_openai_responses(fake, rec, default_model="gpt-4o-mini-2024-07-18")
        client.responses.create(
            input=[{"role": "user", "content": "hi responses"}]
        )
    _replay_cache_only(trace_path)


def test_wrap_openai_responses_tool_call(tmp_path):
    fake = _FakeOAIWithResponses.build(chat_raw={}, responses_raw=_RESPONSES_TOOL_RAW)
    trace_path = str(tmp_path / "resp_tool.sb")
    with record(trace_path) as rec:
        client = wrap_openai_responses(fake, rec, default_model="gpt-4o-2024-11-20")
        resp = client.responses.create(
            input=[{"role": "user", "content": "weather in london"}]
        )
    # output should contain the function_call item
    assert any(item.get("type") == "function_call" for item in resp.output)
    t = replay(trace_path)
    llm_resp = t.recorded_steps[0]["llm_response"]
    tc = llm_resp["choices"][0]["message"]["tool_calls"]
    assert tc[0]["function"]["name"] == "get_weather"
    _replay_cache_only(trace_path)


def test_wrap_openai_responses_string_input(tmp_path):
    """String input is normalised to a single user message."""
    fake = _FakeOAIWithResponses.build(chat_raw={}, responses_raw=_RESPONSES_TEXT_RAW)
    trace_path = str(tmp_path / "resp_str.sb")
    with record(trace_path) as rec:
        client = wrap_openai_responses(fake, rec, default_model="gpt-4o-mini-2024-07-18")
        client.responses.create(input="hello from string")
    t = replay(trace_path)
    # The messages recorded should include the user message
    step = t.recorded_steps[0]
    messages = step["inputs"]["messages"]
    assert any(m.get("role") == "user" for m in messages)


def test_wrap_openai_responses_missing_create_raises():
    with pytest.raises(TypeError, match="lacks .responses.create"):
        wrap_openai_responses(object(), object(), default_model="m")


# =====================================================================
# § 8  OpenAI-compatible endpoints
# =====================================================================


def test_wrap_openai_compatible_endpoint_records(tmp_path):
    """wrap_openai accepts any client with chat.completions.create using the
    OpenAI wire format (Together, Fireworks, vLLM, Ollama, etc.)."""
    # Simulate a Together-AI-shaped response (same wire format as OpenAI)
    together_raw = {
        "id": "together-8675309",
        "object": "chat.completion",
        "created": 1726001000,
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "Together hi.",
                                 "tool_calls": None}}],
        "usage": {"prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8},
    }
    fake = _FakeOAIClient.with_response(together_raw)
    trace_path = str(tmp_path / "compat.sb")
    with record(trace_path) as rec:
        # Use a custom model name typical of Together endpoints
        client = wrap_openai(fake, rec, default_model="meta-llama/Llama-3-8b-chat-hf")
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "hi from compatible"}]
        )
    assert resp.choices[0].message.content == "Together hi."
    t = replay(trace_path)
    assert len(t.recorded_steps) == 1
    assert t.recorded_steps[0]["llm_response"]["id"] == "together-8675309"
    _replay_cache_only(trace_path)


def test_wrap_openai_compatible_endpoint_custom_contract(tmp_path):
    """wrap_openai accepts a custom ShimContract for compatible providers."""
    from stepback.shims import ShimContract, register_shim_contract, shim_contract_for
    import inspect

    # A custom contract that adds a provider-specific sidecar (like Anthropic does)
    class TogetherContract(ShimContract):
        provider_name = "together_test_step90"

        def canonical_request(self, **kwargs):
            return list(kwargs.get("messages") or [])

        def canonical_response(self, native):
            from stepback.shims import _canonicalise_openai_response
            return _canonicalise_openai_response(native)

        def make_executor(self, client):
            from stepback.shims import openai_executor
            return openai_executor(client)

    contract = TogetherContract()
    raw = {
        "id": "tog-001", "object": "chat.completion", "created": 1726001000,
        "model": "together/llama3", "choices": [
            {"index": 0, "finish_reason": "stop",
             "message": {"role": "assistant", "content": "Custom hi.", "tool_calls": None}}
        ], "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }
    fake = _FakeOAIClient.with_response(raw)
    trace_path = str(tmp_path / "custom_contract.sb")
    with record(trace_path) as rec:
        client = wrap_openai(fake, rec, default_model="together/llama3", contract=contract)
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "hi custom"}]
        )
    assert resp.choices[0].message.content == "Custom hi."
    _replay_cache_only(trace_path)


def test_wrap_openai_no_stream_still_works_after_streaming_code_added(tmp_path):
    """Non-streaming path is unaffected by the streaming changes."""
    raw = {
        "id": "ns001", "object": "chat.completion", "created": 1726001000,
        "model": "gpt-4o-mini-2024-07-18", "choices": [
            {"index": 0, "finish_reason": "stop",
             "message": {"role": "assistant", "content": "Non-stream hi.", "tool_calls": None}}
        ], "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }
    fake = _FakeOAIClient.with_response(raw)
    trace_path = str(tmp_path / "ns.sb")
    with record(trace_path) as rec:
        client = wrap_openai(fake, rec, default_model="gpt-4o-mini-2024-07-18")
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}]
        )
    assert resp.choices[0].message.content == "Non-stream hi."
    _replay_cache_only(trace_path)


# =====================================================================
# § 9  Public API exports
# =====================================================================


def test_new_symbols_in_shims_all():
    """New symbols are listed in stepback.shims.__all__."""
    import stepback.shims as s
    for sym in ("wrap_openai_async", "wrap_openai_responses",
                "AsyncWrappedOpenAI", "OpenAIResponsesOutput",
                "_accumulate_streaming_chunks"):
        assert sym in s.__all__, f"{sym!r} missing from stepback.shims.__all__"


def test_new_symbols_importable_from_stepback():
    """New symbols are importable from the top-level package."""
    import stepback as sb
    for sym in ("wrap_openai_async", "wrap_openai_responses",
                "AsyncWrappedOpenAI", "OpenAIResponsesOutput",
                "_accumulate_streaming_chunks"):
        assert hasattr(sb, sym), f"{sym!r} not importable from stepback"
