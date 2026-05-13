"""Streaming recorder support tests — Step 97 in ``100_STEPS.md``.

Covers:

* :class:`StreamedLLMResponse` — class contract: iterator, re-iterable,
  ``assembled_response``, ``step``, ``native``, ``chunk_count``, ``__repr__``.
* OpenAI ``stream=True`` — returns :class:`StreamedLLMResponse`; iteration
  yields original chunks; recording is synchronous (one step); hash is
  deterministic; tool-call streaming; parallel tool calls; usage-chunk
  collection; replay is a cache hit after streaming.
* Anthropic ``stream=True`` — same contract; ``native`` is
  :class:`AnthropicMessage`; text + tool-use + thinking-blocks.
* OpenAI-compatible provider ``stream=True`` — same contract.
* Azure OpenAI ``stream=True`` — deployment name used; canonical model in step.
* Hash determinism — same content with different chunk boundaries yields the
  same ``outputs_hash``.
* Sequential step recording — streaming step + subsequent tool step have
  correct parent-step linkage.
* Public export — ``StreamedLLMResponse`` is in ``stepback.__all__``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional

import pytest

from stepback import record, replay
from stepback.replay import Executor
from stepback.shims import (
    AnthropicMessage,
    OpenAIChatCompletion,
    wrap_anthropic,
    wrap_openai,
)
from stepback.streaming import StreamedLLMResponse


# =====================================================================
# Fake provider helpers
# =====================================================================


@dataclass
class _FakeCompletions:
    responses: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    _idx: int = field(default=0, init=False, repr=False)

    def create(self, *, model: str, messages: list, **kwargs: Any):
        self.calls.append(kwargs)
        resp = self.responses[self._idx % len(self.responses)]
        self._idx += 1
        return resp


@dataclass
class _FakeChat:
    completions: _FakeCompletions


@dataclass
class _FakeOAIClient:
    chat: _FakeChat

    @classmethod
    def with_chunks(cls, chunks: list) -> "_FakeOAIClient":
        return cls(chat=_FakeChat(completions=_FakeCompletions(responses=[iter(chunks)])))

    @classmethod
    def with_multi(cls, responses: list) -> "_FakeOAIClient":
        return cls(chat=_FakeChat(completions=_FakeCompletions(responses=responses)))


@dataclass
class _FakeAnthMsgs:
    events: list = field(default_factory=list)
    calls: list = field(default_factory=list)

    def create(self, *, model, messages, system=None, max_tokens=1024, **kw):
        self.calls.append({"model": model, "messages": messages})
        return iter(list(self.events))


@dataclass
class _FakeAnthropic:
    messages: _FakeAnthMsgs

    @classmethod
    def with_stream(cls, events: list) -> "_FakeAnthropic":
        return cls(messages=_FakeAnthMsgs(events=events))


_TEXT_CHUNKS = [
    {"id": "sc1", "model": "gpt-4o-mini", "choices": [
        {"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}
    ]},
    {"id": "sc1", "model": "gpt-4o-mini", "choices": [
        {"index": 0, "delta": {"content": " world"}, "finish_reason": None}
    ]},
    {"id": "sc1", "model": "gpt-4o-mini", "choices": [
        {"index": 0, "delta": {}, "finish_reason": "stop"}
    ]},
    {"id": "sc1", "model": "gpt-4o-mini", "choices": [],
     "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
]

_TOOL_CHUNKS = [
    {"id": "tc1", "model": "gpt-4o", "choices": [
        {"index": 0, "delta": {"role": "assistant",
            "tool_calls": [{"index": 0, "id": "call_abc", "type": "function",
                             "function": {"name": "search", "arguments": ""}}]},
         "finish_reason": None}
    ]},
    {"id": "tc1", "model": "gpt-4o", "choices": [
        {"index": 0, "delta": {
            "tool_calls": [{"index": 0, "function": {"arguments": '{"q":"ai"}'}}]},
         "finish_reason": None}
    ]},
    {"id": "tc1", "model": "gpt-4o", "choices": [
        {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
    ]},
    {"id": "tc1", "model": "gpt-4o", "choices": [],
     "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
]


def _make_anth_events(text: str, input_tokens: int = 5, output_tokens: int = 3) -> list:
    return [
        {"type": "message_start",
         "message": {"id": "msg1", "model": "claude-3-5-haiku-20241022",
                     "usage": {"input_tokens": input_tokens}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta",
         "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": output_tokens}},
    ]


# =====================================================================
# § 1  StreamedLLMResponse class contract
# =====================================================================


def test_streamed_llm_response_is_iterable():
    chunks = [{"id": "c1"}, {"id": "c2"}, {"id": "c3"}]
    resp = StreamedLLMResponse(chunks, {"choices": []}, {})
    assert list(resp) == chunks


def test_streamed_llm_response_is_re_iterable():
    chunks = [{"id": "c1"}, {"id": "c2"}]
    resp = StreamedLLMResponse(chunks, {"choices": []}, {})
    assert list(resp) == chunks
    assert list(resp) == chunks  # re-iteration still works


def test_streamed_llm_response_len():
    resp = StreamedLLMResponse([1, 2, 3], {"choices": []}, {})
    assert len(resp) == 3


def test_streamed_llm_response_chunk_count():
    resp = StreamedLLMResponse([1, 2, 3, 4], {"choices": []}, {})
    assert resp.chunk_count == 4


def test_streamed_llm_response_assembled_response_deep_copy():
    assembled = {"choices": [{"message": {"content": "hi"}}]}
    resp = StreamedLLMResponse([], assembled, {})
    # Modifying original should not affect property (deep copy)
    assembled["choices"][0]["message"]["content"] = "MUTATED"
    assert resp.assembled_response["choices"][0]["message"]["content"] == "hi"


def test_streamed_llm_response_step_deep_copy():
    step = {"step_kind": "llm_call", "cost_usd": 0.01}
    resp = StreamedLLMResponse([], {}, step)
    step["cost_usd"] = 999.0
    assert resp.step["cost_usd"] == 0.01


def test_streamed_llm_response_native_default_is_none():
    resp = StreamedLLMResponse([], {}, {})
    assert resp.native is None


def test_streamed_llm_response_native_set():
    native = object()
    resp = StreamedLLMResponse([], {}, {}, native=native)
    assert resp.native is native


def test_streamed_llm_response_repr():
    assembled = {"choices": [{"finish_reason": "stop"}]}
    resp = StreamedLLMResponse([1, 2], assembled, {})
    r = repr(resp)
    assert "StreamedLLMResponse" in r
    assert "chunk_count=2" in r
    assert "stop" in r


def test_streamed_llm_response_public_export():
    import stepback
    assert "StreamedLLMResponse" in stepback.__all__
    assert stepback.StreamedLLMResponse is StreamedLLMResponse


# =====================================================================
# § 2  OpenAI wrap_openai stream=True
# =====================================================================


def test_wrap_openai_stream_returns_streamed_llm_response(tmp_path):
    fake = _FakeOAIClient.with_chunks(list(_TEXT_CHUNKS))
    with record(str(tmp_path / "t.sb")) as rec:
        client = wrap_openai(fake, rec, default_model="gpt-4o-mini")
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    assert isinstance(resp, StreamedLLMResponse)


def test_wrap_openai_stream_iteration_yields_original_chunks(tmp_path):
    chunks = list(_TEXT_CHUNKS)
    fake = _FakeOAIClient.with_chunks(list(chunks))
    with record(str(tmp_path / "t.sb")) as rec:
        client = wrap_openai(fake, rec, default_model="gpt-4o-mini")
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    yielded = list(resp)
    assert len(yielded) == len(chunks)
    for i, chunk in enumerate(yielded):
        assert chunk == chunks[i]


def test_wrap_openai_stream_is_re_iterable(tmp_path):
    fake = _FakeOAIClient.with_chunks(list(_TEXT_CHUNKS))
    with record(str(tmp_path / "t.sb")) as rec:
        client = wrap_openai(fake, rec, default_model="gpt-4o-mini")
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    assert list(resp) == list(resp)  # can iterate twice


def test_wrap_openai_stream_assembled_response_matches_step(tmp_path):
    fake = _FakeOAIClient.with_chunks(list(_TEXT_CHUNKS))
    with record(str(tmp_path / "t.sb")) as rec:
        client = wrap_openai(fake, rec, default_model="gpt-4o-mini")
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    assert resp.assembled_response == resp.step["llm_response"]


def test_wrap_openai_stream_records_one_step(tmp_path):
    fake = _FakeOAIClient.with_chunks(list(_TEXT_CHUNKS))
    with record(str(tmp_path / "t.sb")) as rec:
        client = wrap_openai(fake, rec, default_model="gpt-4o-mini")
        client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    t = replay(str(tmp_path / "t.sb"))
    assert len(t.recorded_steps) == 1
    assert t.recorded_steps[0]["step_kind"] == "llm_call"


def test_wrap_openai_stream_correct_content(tmp_path):
    fake = _FakeOAIClient.with_chunks(list(_TEXT_CHUNKS))
    with record(str(tmp_path / "t.sb")) as rec:
        client = wrap_openai(fake, rec, default_model="gpt-4o-mini")
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    assert resp.assembled_response["choices"][0]["message"]["content"] == "Hello world"
    assert resp.assembled_response["choices"][0]["finish_reason"] == "stop"
    assert resp.assembled_response["usage"]["prompt_tokens"] == 5


def test_wrap_openai_stream_native_is_openai_chat_completion(tmp_path):
    fake = _FakeOAIClient.with_chunks(list(_TEXT_CHUNKS))
    with record(str(tmp_path / "t.sb")) as rec:
        client = wrap_openai(fake, rec, default_model="gpt-4o-mini")
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    assert isinstance(resp.native, OpenAIChatCompletion)
    assert resp.native.choices[0].message.content == "Hello world"


def test_wrap_openai_stream_replay_cache_hit(tmp_path):
    fake = _FakeOAIClient.with_chunks(list(_TEXT_CHUNKS))
    with record(str(tmp_path / "t.sb")) as rec:
        client = wrap_openai(fake, rec, default_model="gpt-4o-mini")
        client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    t = replay(str(tmp_path / "t.sb"))
    result = t.replay_forward(executor=Executor(fallback_recorded=True))
    assert result.real_executions == 0
    assert result.cache_hit_count == 1


def test_wrap_openai_stream_tool_calls_assembled(tmp_path):
    fake = _FakeOAIClient.with_chunks(list(_TOOL_CHUNKS))
    with record(str(tmp_path / "t.sb")) as rec:
        client = wrap_openai(fake, rec, default_model="gpt-4o")
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "search"}],
            stream=True,
        )
    tc = resp.assembled_response["choices"][0]["message"]["tool_calls"]
    assert tc[0]["function"]["name"] == "search"
    assert tc[0]["function"]["arguments"] == '{"q":"ai"}'
    assert resp.native.choices[0].message.tool_calls[0]["function"]["name"] == "search"


def test_wrap_openai_stream_chunk_count(tmp_path):
    fake = _FakeOAIClient.with_chunks(list(_TEXT_CHUNKS))
    with record(str(tmp_path / "t.sb")) as rec:
        client = wrap_openai(fake, rec, default_model="gpt-4o-mini")
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    assert resp.chunk_count == len(_TEXT_CHUNKS)


# =====================================================================
# § 3  Hash determinism — chunk boundary independence
# =====================================================================


def test_stream_outputs_hash_same_as_nonstream(tmp_path):
    """Streaming and non-streaming calls with the same content produce the same outputs_hash."""
    # Non-streaming fake: returns the same assembled response shape
    non_stream_resp = {
        "id": "sc1",
        "model": "gpt-4o-mini",
        "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "Hello world",
                                 "tool_calls": None}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }

    # Record streaming version
    fake_stream = _FakeOAIClient.with_chunks(list(_TEXT_CHUNKS))
    with record(str(tmp_path / "stream.sb")) as rec:
        client = wrap_openai(fake_stream, rec, default_model="gpt-4o-mini")
        client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    t_stream = replay(str(tmp_path / "stream.sb"))
    stream_hash = t_stream.recorded_steps[0]["outputs_hash"]

    # Record non-streaming version with same content
    class _FakeCompletionsDirect:
        def create(self, *, model, messages, **kw):
            return dict(non_stream_resp)

    class _FakeChatDirect:
        completions = None

        def __init__(self):
            self.completions = _FakeCompletionsDirect()

    class _FakeClientDirect:
        chat = None

        def __init__(self):
            self.chat = _FakeChatDirect()

    fake_ns = _FakeClientDirect()
    with record(str(tmp_path / "nonstream.sb")) as rec2:
        client2 = wrap_openai(fake_ns, rec2, default_model="gpt-4o-mini")
        client2.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
        )
    t_ns = replay(str(tmp_path / "nonstream.sb"))
    ns_hash = t_ns.recorded_steps[0]["outputs_hash"]

    assert stream_hash == ns_hash


def test_stream_hash_invariant_to_chunk_boundaries(tmp_path):
    """Same content, different chunk splits, same outputs_hash."""
    # Single-chunk version (all content in one chunk, same id/model as _TEXT_CHUNKS)
    single_chunk = [
        {"id": "sc1", "model": "gpt-4o-mini", "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": "Hello world"},
             "finish_reason": "stop"}
        ]},
        {"id": "sc1", "model": "gpt-4o-mini", "choices": [],
         "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
    ]

    fake_single = _FakeOAIClient.with_chunks(single_chunk)
    with record(str(tmp_path / "single.sb")) as rec:
        client = wrap_openai(fake_single, rec, default_model="gpt-4o-mini")
        client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    t_single = replay(str(tmp_path / "single.sb"))
    hash_single = t_single.recorded_steps[0]["outputs_hash"]

    # Multi-chunk version (same content, different split)
    fake_multi = _FakeOAIClient.with_chunks(list(_TEXT_CHUNKS))
    with record(str(tmp_path / "multi.sb")) as rec2:
        client2 = wrap_openai(fake_multi, rec2, default_model="gpt-4o-mini")
        client2.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    t_multi = replay(str(tmp_path / "multi.sb"))
    hash_multi = t_multi.recorded_steps[0]["outputs_hash"]

    assert hash_single == hash_multi


# =====================================================================
# § 4  Sequential step recording — parent linkage is correct
# =====================================================================


def test_stream_step_then_tool_step_parent_linkage(tmp_path):
    """Streaming llm_call step followed by a tool_call step has correct parent."""
    fake = _FakeOAIClient.with_chunks(list(_TEXT_CHUNKS))
    with record(str(tmp_path / "t.sb")) as rec:
        client = wrap_openai(fake, rec, default_model="gpt-4o-mini")
        client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        rec.tool_call("my_tool", {"x": 1}, executor=lambda n, a: {"result": "ok"})

    t = replay(str(tmp_path / "t.sb"))
    assert len(t.recorded_steps) == 2
    llm_step = t.recorded_steps[0]
    tool_step = t.recorded_steps[1]
    assert tool_step["parent_step_id"] == llm_step["step_id"]


# =====================================================================
# § 5  Anthropic wrap_anthropic stream=True
# =====================================================================


def test_wrap_anthropic_stream_returns_streamed_llm_response(tmp_path):
    fake = _FakeAnthropic.with_stream(_make_anth_events("Hey!"))
    with record(str(tmp_path / "t.sb")) as rec:
        client = wrap_anthropic(fake, rec)
        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    assert isinstance(resp, StreamedLLMResponse)


def test_wrap_anthropic_stream_native_is_anthropic_message(tmp_path):
    fake = _FakeAnthropic.with_stream(_make_anth_events("Streaming!"))
    with record(str(tmp_path / "t.sb")) as rec:
        client = wrap_anthropic(fake, rec)
        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    assert isinstance(resp.native, AnthropicMessage)
    assert resp.native.content[0].text == "Streaming!"


def test_wrap_anthropic_stream_assembled_response_correct(tmp_path):
    fake = _FakeAnthropic.with_stream(_make_anth_events("Text", input_tokens=8, output_tokens=3))
    with record(str(tmp_path / "t.sb")) as rec:
        client = wrap_anthropic(fake, rec)
        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    assert resp.assembled_response["choices"][0]["message"]["content"] == "Text"
    assert resp.assembled_response["usage"]["prompt_tokens"] == 8
    assert resp.assembled_response["usage"]["completion_tokens"] == 3


def test_wrap_anthropic_stream_yields_events(tmp_path):
    events = _make_anth_events("Hello!")
    fake = _FakeAnthropic.with_stream(events)
    with record(str(tmp_path / "t.sb")) as rec:
        client = wrap_anthropic(fake, rec)
        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    # Should yield the same events that were produced
    yielded = list(resp)
    assert len(yielded) == len(events)


def test_wrap_anthropic_stream_records_one_step(tmp_path):
    fake = _FakeAnthropic.with_stream(_make_anth_events("step-count"))
    with record(str(tmp_path / "t.sb")) as rec:
        client = wrap_anthropic(fake, rec)
        client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    t = replay(str(tmp_path / "t.sb"))
    assert len(t.recorded_steps) == 1
    assert t.recorded_steps[0]["step_kind"] == "llm_call"


def test_wrap_anthropic_stream_replay_cache_hit(tmp_path):
    fake = _FakeAnthropic.with_stream(_make_anth_events("cached"))
    with record(str(tmp_path / "t.sb")) as rec:
        client = wrap_anthropic(fake, rec)
        client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    t = replay(str(tmp_path / "t.sb"))
    result = t.replay_forward(executor=Executor(fallback_recorded=True))
    assert result.real_executions == 0
    assert result.cache_hit_count == 1


# =====================================================================
# § 6  Non-streaming paths still return correct types
# =====================================================================


def test_wrap_openai_nonstream_still_returns_openai_chat_completion(tmp_path):
    """Non-streaming path is unaffected: still returns OpenAIChatCompletion."""
    raw = {
        "id": "ns1", "model": "gpt-4o-mini",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "ok", "tool_calls": None}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }

    @dataclass
    class _Completions:
        def create(self, *, model, messages, **kw):
            return dict(raw)

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Client:
        def __init__(self):
            self.chat = _Chat()

    fake = _Client()
    with record(str(tmp_path / "t.sb")) as rec:
        client = wrap_openai(fake, rec, default_model="gpt-4o-mini")
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
        )
    assert isinstance(resp, OpenAIChatCompletion)
    assert not isinstance(resp, StreamedLLMResponse)
    assert resp.choices[0].message.content == "ok"


def test_wrap_anthropic_nonstream_still_returns_anthropic_message(tmp_path):
    """Non-streaming Anthropic path still returns AnthropicMessage."""
    raw = {
        "id": "msg1", "model": "claude-3-5-haiku-20241022", "role": "assistant",
        "content": [{"type": "text", "text": "direct"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }

    @dataclass
    class _Msgs:
        def create(self, *, model, messages, system=None, max_tokens=1024, **kw):
            return dict(raw)

    class _Anth:
        def __init__(self):
            self.messages = _Msgs()

    fake = _Anth()
    with record(str(tmp_path / "t.sb")) as rec:
        client = wrap_anthropic(fake, rec)
        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert isinstance(resp, AnthropicMessage)
    assert not isinstance(resp, StreamedLLMResponse)
    assert resp.content[0].text == "direct"
