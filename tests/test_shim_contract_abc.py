"""Tests for the ShimContract ABC and its four built-in concrete implementations.

Step #89 in ``100_STEPS.md``: Refactor provider shims behind a
``ShimContract`` ABC: canonical request, canonical response, executor,
streaming hooks, async hooks, version probe.

These tests verify:
1. Each concrete contract instantiates without error.
2. ``shim_contract_for`` returns the right type.
3. ``canonical_request`` round-trips correctly for each provider.
4. ``canonical_response`` results match the existing low-level helpers.
5. ``make_executor`` returns a callable.
6. ``stream_request`` raises ``NotImplementedError`` (not yet implemented).
7. ``async_request`` raises ``NotImplementedError`` (not yet implemented).
8. ``version_probe`` returns a str or None (never raises).
9. Custom contracts can be registered and retrieved.
10. Duplicate registration raises without ``overwrite=True``.
11. ``wrap_*`` factories accept an explicit ``contract`` kwarg.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, List, Optional

import pytest

from stepback.shims import (
    AnthropicShimContract,
    BedrockShimContract,
    GeminiShimContract,
    OpenAIShimContract,
    ShimContract,
    _anthropic_to_openai_shape,
    _bedrock_messages_to_unified,
    _bedrock_to_openai_shape,
    _canonicalise_openai_response,
    _coerce_anthropic_response,
    _coerce_bedrock_response,
    _coerce_gemini_response,
    _gemini_contents_to_unified,
    _gemini_to_openai_shape,
    register_shim_contract,
    shim_contract_for,
    wrap_anthropic,
    wrap_bedrock,
    wrap_gemini,
    wrap_openai,
)

# =====================================================================
# Provider name mapping
# =====================================================================

_BUILTIN_CONTRACTS = [
    ("openai", OpenAIShimContract),
    ("anthropic", AnthropicShimContract),
    ("bedrock", BedrockShimContract),
    ("gemini", GeminiShimContract),
]


# =====================================================================
# 1. Instantiation
# =====================================================================


@pytest.mark.parametrize("provider,cls", _BUILTIN_CONTRACTS)
def test_builtin_contract_instantiates(provider, cls):
    obj = cls()
    assert isinstance(obj, ShimContract)
    assert obj.provider_name == provider


# =====================================================================
# 2. Registry lookup
# =====================================================================


@pytest.mark.parametrize("provider,cls", _BUILTIN_CONTRACTS)
def test_shim_contract_for_returns_correct_type(provider, cls):
    contract = shim_contract_for(provider)
    assert isinstance(contract, cls)


def test_shim_contract_for_unknown_raises():
    with pytest.raises(KeyError, match="no_such_provider_xyz"):
        shim_contract_for("no_such_provider_xyz")


# =====================================================================
# 3. canonical_request — matches existing low-level helpers
# =====================================================================


def test_openai_canonical_request_passthrough():
    messages = [{"role": "user", "content": "hi"}]
    result = OpenAIShimContract().canonical_request(messages=messages)
    assert result == messages


def test_openai_canonical_request_empty():
    assert OpenAIShimContract().canonical_request() == []
    assert OpenAIShimContract().canonical_request(messages=[]) == []


def test_anthropic_canonical_request_no_system():
    messages = [{"role": "user", "content": "hello"}]
    result = AnthropicShimContract().canonical_request(messages=messages)
    assert result == messages


def test_anthropic_canonical_request_with_system():
    messages = [{"role": "user", "content": "hello"}]
    result = AnthropicShimContract().canonical_request(
        messages=messages, system="be terse"
    )
    assert result[0] == {"role": "system", "content": "be terse"}
    assert result[1:] == messages


def test_bedrock_canonical_request_matches_helper():
    messages = [
        {"role": "user", "content": [{"text": "hi"}]},
    ]
    system = [{"text": "be helpful"}]
    contract_result = BedrockShimContract().canonical_request(
        messages=messages, system=system
    )
    helper_result = _bedrock_messages_to_unified(messages, system)
    assert contract_result == helper_result


def test_gemini_canonical_request_matches_helper():
    contents = [{"role": "user", "parts": [{"text": "hello"}]}]
    system_instruction = "be terse"
    contract_result = GeminiShimContract().canonical_request(
        contents=contents, system_instruction=system_instruction
    )
    helper_result = _gemini_contents_to_unified(contents, system_instruction)
    assert contract_result == helper_result


def test_gemini_canonical_request_no_system():
    contents = "just a string"
    result = GeminiShimContract().canonical_request(contents=contents)
    assert result == [{"role": "user", "content": "just a string"}]


# =====================================================================
# 4. canonical_response — matches existing low-level helpers
# =====================================================================

_OPENAI_RAW = {
    "id": "chatcmpl-001",
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Hello!"},
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}

_ANTHROPIC_RAW = {
    "id": "msg_001",
    "model": "claude-3-5-haiku-20241022",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hi there"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 4},
}

_BEDROCK_RAW = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [{"text": "Bedrock reply"}],
        }
    },
    "stopReason": "end_turn",
    "usage": {"inputTokens": 8, "outputTokens": 3, "totalTokens": 11},
    "ResponseMetadata": {"RequestId": "req-abc"},
    "_modelId": "anthropic.claude-3-5-haiku-20241022-v1:0",
}

_GEMINI_RAW = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "Gemini reply"}]},
            "finish_reason": "STOP",
        }
    ],
    "usage_metadata": {
        "prompt_token_count": 6,
        "candidates_token_count": 2,
        "total_token_count": 8,
    },
    "model_version": "gemini-2.5-flash",
}


def test_openai_canonical_response_matches_helper():
    assert OpenAIShimContract().canonical_response(_OPENAI_RAW) == \
        _canonicalise_openai_response(_OPENAI_RAW)


def test_anthropic_canonical_response_matches_helper():
    assert AnthropicShimContract().canonical_response(_ANTHROPIC_RAW) == \
        _anthropic_to_openai_shape(_coerce_anthropic_response(_ANTHROPIC_RAW))


def test_bedrock_canonical_response_matches_helper():
    assert BedrockShimContract().canonical_response(_BEDROCK_RAW) == \
        _bedrock_to_openai_shape(_coerce_bedrock_response(_BEDROCK_RAW))


def test_gemini_canonical_response_matches_helper():
    assert GeminiShimContract().canonical_response(_GEMINI_RAW) == \
        _gemini_to_openai_shape(_coerce_gemini_response(_GEMINI_RAW))


def test_openai_canonical_response_has_required_keys():
    result = OpenAIShimContract().canonical_response(_OPENAI_RAW)
    assert "choices" in result
    assert "usage" in result
    assert result["choices"][0]["message"]["content"] == "Hello!"


def test_anthropic_canonical_response_openai_shape():
    result = AnthropicShimContract().canonical_response(_ANTHROPIC_RAW)
    assert result["choices"][0]["message"]["content"] == "Hi there"
    assert result["usage"]["prompt_tokens"] == 10


def test_bedrock_canonical_response_openai_shape():
    result = BedrockShimContract().canonical_response(_BEDROCK_RAW)
    assert result["choices"][0]["message"]["content"] == "Bedrock reply"
    assert result["usage"]["total_tokens"] == 11


def test_gemini_canonical_response_openai_shape():
    result = GeminiShimContract().canonical_response(_GEMINI_RAW)
    assert result["choices"][0]["message"]["content"] == "Gemini reply"
    assert result["usage"]["total_tokens"] == 8


# =====================================================================
# 5. make_executor returns a callable
# =====================================================================


@dataclass
class _FakeOAIClient:
    """Minimal duck-typed openai.OpenAI shape."""
    class chat:
        class completions:
            @staticmethod
            def create(*, model, messages, **kw):
                return {
                    "id": "x", "model": model,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }


@dataclass
class _FakeAnthropicClient:
    class messages:
        @staticmethod
        def create(*, model, messages, system=None, max_tokens=1024, **kw):
            return {
                "id": "m", "model": model, "role": "assistant",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1},
            }


@dataclass
class _FakeBedrockClient:
    def converse(self, *, modelId, messages, **kw):
        return {
            "output": {"message": {"role": "assistant",
                                   "content": [{"text": "bedrock"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        }


@dataclass
class _FakeGeminiModels:
    def generate_content(self, *, model, contents, config=None, **kw):
        return {
            "candidates": [{"content": {"role": "model", "parts": [{"text": "gem"}]},
                            "finish_reason": "STOP"}],
            "usage_metadata": {"prompt_token_count": 1, "candidates_token_count": 1,
                               "total_token_count": 2},
            "model_version": model,
        }


@dataclass
class _FakeGeminiClient:
    models: _FakeGeminiModels = None  # type: ignore[assignment]
    def __post_init__(self):
        if self.models is None:
            self.models = _FakeGeminiModels()


def test_openai_make_executor_returns_callable():
    fake = _FakeOAIClient()
    ex = OpenAIShimContract().make_executor(fake)
    assert callable(ex)
    result = ex("gpt-4o-mini", [{"role": "user", "content": "hi"}])
    assert "choices" in result


def test_anthropic_make_executor_returns_callable():
    fake = _FakeAnthropicClient()
    ex = AnthropicShimContract().make_executor(fake)
    assert callable(ex)
    result = ex("claude-3-5-haiku-20241022", [{"role": "user", "content": "hi"}])
    assert "choices" in result


def test_bedrock_make_executor_returns_callable():
    fake = _FakeBedrockClient()
    ex = BedrockShimContract().make_executor(fake)
    assert callable(ex)
    result = ex("anthropic.claude-3-5-haiku-20241022-v1:0",
                [{"role": "user", "content": "hi"}])
    assert "choices" in result


def test_gemini_make_executor_returns_callable():
    fake = _FakeGeminiClient()
    ex = GeminiShimContract().make_executor(fake)
    assert callable(ex)
    result = ex("gemini-2.5-flash", [{"role": "user", "content": "hi"}])
    assert "choices" in result


# =====================================================================
# 6. stream_request raises NotImplementedError for Bedrock / Gemini;
#    OpenAI and Anthropic implement it.
# =====================================================================

_NON_OPENAI_CONTRACTS = [
    ("bedrock", BedrockShimContract),
    ("gemini", GeminiShimContract),
]


@pytest.mark.parametrize("provider,cls", _NON_OPENAI_CONTRACTS)
def test_stream_request_not_implemented(provider, cls):
    contract = cls()
    with pytest.raises(NotImplementedError, match="streaming"):
        contract.stream_request(object(), [])


def test_openai_stream_request_is_implemented():
    """OpenAI stream_request is live — it calls chat.completions.create(stream=True)."""
    chunks = [
        {"id": "abc", "model": "gpt-4o", "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": "hi"},
             "finish_reason": "stop"}
        ]},
    ]

    class _Completions:
        def create(self, *, model, messages, **kwargs):
            assert kwargs.get("stream") is True
            return iter(chunks)

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Client:
        def __init__(self):
            self.chat = _Chat()

    result = OpenAIShimContract().stream_request(_Client(), [], model="gpt-4o")
    assert isinstance(result, dict)
    assert result["choices"][0]["message"]["content"] == "hi"


# =====================================================================
# 7. async_request raises NotImplementedError for Bedrock / Gemini;
#    OpenAI and Anthropic implement it as a coroutine.
# =====================================================================


@pytest.mark.parametrize("provider,cls", _NON_OPENAI_CONTRACTS)
def test_async_request_not_implemented(provider, cls):
    contract = cls()
    with pytest.raises(NotImplementedError, match="async"):
        asyncio.run(contract.async_request(object(), []))


def test_openai_async_request_is_implemented():
    """OpenAI async_request is live — awaits chat.completions.create."""
    raw = {
        "id": "abc-async", "object": "chat.completion", "created": 1726000000,
        "model": "gpt-4o", "choices": [
            {"index": 0, "finish_reason": "stop",
             "message": {"role": "assistant", "content": "async ok", "tool_calls": None}}
        ], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }

    class _AsyncCompletions:
        async def create(self, *, model, messages, **kwargs):
            return dict(raw)

    class _AsyncChat:
        def __init__(self):
            self.completions = _AsyncCompletions()

    class _AsyncClient:
        def __init__(self):
            self.chat = _AsyncChat()

    result = asyncio.run(
        OpenAIShimContract().async_request(_AsyncClient(), [], model="gpt-4o")
    )
    assert isinstance(result, dict)
    assert result["choices"][0]["message"]["content"] == "async ok"


# =====================================================================
# 8. version_probe returns str or None (never raises)
# =====================================================================


@pytest.mark.parametrize("provider,cls", _BUILTIN_CONTRACTS)
def test_version_probe_returns_str_or_none(provider, cls):
    result = cls().version_probe(object())
    assert result is None or isinstance(result, str)


# =====================================================================
# 9. Custom contract registration and retrieval
# =====================================================================


def test_register_and_retrieve_custom_contract():
    class _TestContract(ShimContract):
        provider_name = "_test_unique_xyz_123"

        def canonical_request(self, **kwargs):
            return []

        def canonical_response(self, native):
            return {}

        def make_executor(self, client):
            return lambda m, msgs: {}

    contract = _TestContract()
    register_shim_contract(contract)
    retrieved = shim_contract_for("_test_unique_xyz_123")
    assert retrieved is contract
    # Clean up — re-register with overwrite=True to avoid polluting other tests
    register_shim_contract(_TestContract(), overwrite=True)


# =====================================================================
# 10. Duplicate registration error
# =====================================================================


def test_duplicate_registration_raises():
    # All four built-ins are already registered; trying again should fail.
    with pytest.raises(KeyError, match="already registered"):
        register_shim_contract(OpenAIShimContract())


def test_duplicate_registration_succeeds_with_overwrite():
    # Should not raise.
    register_shim_contract(OpenAIShimContract(), overwrite=True)
    # Registry still works after overwrite.
    assert isinstance(shim_contract_for("openai"), OpenAIShimContract)


def test_register_missing_provider_name_raises():
    class _Bad(ShimContract):
        # No provider_name set
        def canonical_request(self, **kwargs): return []
        def canonical_response(self, native): return {}
        def make_executor(self, client): return lambda m, msgs: {}

    with pytest.raises(TypeError, match="provider_name"):
        register_shim_contract(_Bad())


# =====================================================================
# 11. wrap_* factories accept explicit contract kwarg
# =====================================================================


class _CustomOpenAIContract(OpenAIShimContract):
    """Spy contract that records whether canonical_response was called."""
    provider_name = "openai"

    def __init__(self):
        super().__init__()
        self.response_calls: List[Any] = []

    def canonical_response(self, native):
        self.response_calls.append(native)
        return super().canonical_response(native)


def test_wrap_openai_uses_custom_contract():
    from stepback import record

    spy = _CustomOpenAIContract()

    class _FakeOAICompletions:
        def create(self, *, model, messages, **kw):
            return {
                "id": "cmp-1", "model": model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "hi"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    class _FakeChat:
        completions = _FakeOAICompletions()

    class _FakeClient:
        chat = _FakeChat()

    import io
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        path = f.name
    try:
        with record(path) as rec:
            client = wrap_openai(_FakeClient(), rec, default_model="gpt-4o-mini",
                                 contract=spy)
            client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
        assert len(spy.response_calls) == 1, "spy.canonical_response should have been called once"
    finally:
        os.unlink(path)


class _CustomAnthropicContract(AnthropicShimContract):
    provider_name = "anthropic"

    def __init__(self):
        super().__init__()
        self.request_calls: List[Any] = []

    def canonical_request(self, **kwargs):
        self.request_calls.append(kwargs)
        return super().canonical_request(**kwargs)


def test_wrap_anthropic_uses_custom_contract():
    from stepback import record
    import tempfile, os

    spy = _CustomAnthropicContract()

    class _FakeMessages:
        def create(self, *, model, messages, system=None, max_tokens=1024, **kw):
            return {
                "id": "msg-1", "model": model, "role": "assistant",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

    class _FakeAnth:
        messages = _FakeMessages()

    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        path = f.name
    try:
        with record(path) as rec:
            client = wrap_anthropic(_FakeAnth(), rec, contract=spy)
            client.messages.create(
                model="claude-3-5-haiku-20241022",
                messages=[{"role": "user", "content": "hello"}],
                system="be terse",
            )
        assert len(spy.request_calls) == 1
        assert spy.request_calls[0].get("system") == "be terse"
    finally:
        os.unlink(path)
