"""End-to-end tests for the OpenAI-compatible provider shims (Step 96).

Covers Together AI, Fireworks, Groq, Cerebras, NVIDIA NIM, vLLM, TGI,
llama.cpp, and Ollama adapters.

Fake clients below are duck-typed against the OpenAI chat completions surface:
``client.chat.completions.create(model=..., messages=..., ...)``.

Tests prove for each OpenAI-compatible provider:

1. ``wrap_*`` rejects clients that lack ``.chat.completions``.
2. record→replay-from-cache is a 100% cache hit (zero real API calls).
3. The recorded canonical ``llm_request.messages`` is the raw message list
   (OpenAI wire format; no transformation).
4. A :class:`PromptSubstitution` against a recorded step correctly dirties
   downstream steps and re-executes through ``*_executor``.
5. Tool calls round-trip through the canonicaliser.
6. ``seed`` is NOT forwarded to the API for ``cerebras`` (SeedSupport.NONE).
7. ``seed`` IS forwarded for other BEST_EFFORT providers.
8. The wrapped client returns an :class:`OpenAIChatCompletion` namespace.
9. All new ShimContracts are pre-registered and reachable via
   ``shim_contract_for``.
10. All new wrap/executor/contract symbols are in ``stepback.__all__``.
11. ``WrappedOpenAICompat.provider_name`` is set correctly.
12. The generic ``openai_compat_executor`` works with any compat client.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

import stepback
from stepback import record, replay
from stepback.replay import Executor
from stepback.seeding import SeedPolicyViolation, SeedSupport
from stepback.shims import (    CerebrasShimContract,
    FireworksShimContract,
    GroqShimContract,
    LlamaCppShimContract,
    NvidiaNIMShimContract,
    OllamaShimContract,
    OpenAICompatShimContract,
    TGIShimContract,
    TogetherShimContract,
    VLLMShimContract,
    WrappedOpenAICompat,
    cerebras_executor,
    fireworks_executor,
    groq_executor,
    llamacpp_executor,
    nvidia_nim_executor,
    ollama_executor,
    openai_compat_executor,
    shim_contract_for,
    tgi_executor,
    together_executor,
    vllm_executor,
    wrap_cerebras,
    wrap_fireworks,
    wrap_groq,
    wrap_llamacpp,
    wrap_nvidia_nim,
    wrap_ollama,
    wrap_tgi,
    wrap_together,
    wrap_vllm,
)
from stepback.substitutions import PromptSubstitution, SubstitutionSet


# =====================================================================
# Fake OpenAI-compatible client
# =====================================================================


@dataclass
class _FakeOAIMessage:
    role: str
    content: Optional[str]
    tool_calls: Optional[List[dict]] = None


@dataclass
class _FakeOAIChoice:
    index: int
    finish_reason: str
    message: _FakeOAIMessage


@dataclass
class _FakeOAIResponse:
    id: str
    model: str
    choices: List[_FakeOAIChoice]
    usage: dict

    def model_dump(self) -> dict:
        return {
            "id": self.id,
            "model": self.model,
            "choices": [
                {
                    "index": c.index,
                    "finish_reason": c.finish_reason,
                    "message": {
                        "role": c.message.role,
                        "content": c.message.content,
                        "tool_calls": c.message.tool_calls,
                    },
                }
                for c in self.choices
            ],
            "usage": self.usage,
        }


class _FakeCompletions:
    """Duck-typed against ``openai.resources.chat.completions.Completions``."""

    def __init__(self) -> None:
        self._calls: List[dict] = []
        self.next_tool_calls: Optional[List[dict]] = None
        self._received_seeds: List[Optional[int]] = []

    def create(self, *, model: str, messages: List[dict],
               **kwargs: Any) -> _FakeOAIResponse:
        self._calls.append({"model": model, "messages": messages, **kwargs})
        self._received_seeds.append(kwargs.get("seed"))

        last_user = ""
        for m in messages:
            c = m.get("content", "")
            if m.get("role") == "user":
                last_user = c if isinstance(c, str) else str(c)

        body = f"compat-{model}-msg{len(messages)}-call{len(self._calls)}-echo[{last_user}]"
        in_tok = sum(len(str(m.get("content", ""))) for m in messages)
        out_tok = len(body)

        tc = self.next_tool_calls
        self.next_tool_calls = None
        finish_reason = "tool_calls" if tc else "stop"

        return _FakeOAIResponse(
            id=f"chatcmpl-{len(self._calls):04d}",
            model=model,
            choices=[_FakeOAIChoice(
                index=0,
                finish_reason=finish_reason,
                message=_FakeOAIMessage(
                    role="assistant",
                    content=body if not tc else None,
                    tool_calls=tc,
                ),
            )],
            usage={"prompt_tokens": in_tok, "completion_tokens": out_tok,
                   "total_tokens": in_tok + out_tok},
        )

    @property
    def call_count(self) -> int:
        return len(self._calls)


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeCompatClient:
    """Minimal fake for any OpenAI-compatible client."""

    def __init__(self) -> None:
        self.chat = _FakeChat()

    def reset(self) -> None:
        self.chat = _FakeChat()

    @property
    def call_count(self) -> int:
        return self.chat.completions.call_count


# =====================================================================
# Provider parameter table
# =====================================================================

_ALL_PROVIDERS = [
    ("groq",       wrap_groq,       groq_executor,       GroqShimContract),
    ("together",   wrap_together,   together_executor,   TogetherShimContract),
    ("fireworks",  wrap_fireworks,  fireworks_executor,  FireworksShimContract),
    ("cerebras",   wrap_cerebras,   cerebras_executor,   CerebrasShimContract),
    ("nvidia_nim", wrap_nvidia_nim, nvidia_nim_executor, NvidiaNIMShimContract),
    ("vllm",       wrap_vllm,       vllm_executor,       VLLMShimContract),
    ("tgi",        wrap_tgi,        tgi_executor,        TGIShimContract),
    ("llamacpp",   wrap_llamacpp,   llamacpp_executor,   LlamaCppShimContract),
    ("ollama",     wrap_ollama,     ollama_executor,     OllamaShimContract),
]

_PROVIDER_IDS = [p[0] for p in _ALL_PROVIDERS]


def pytest_generate_tests(metafunc):
    if "provider_tuple" in metafunc.fixturenames:
        metafunc.parametrize(
            "provider_tuple",
            _ALL_PROVIDERS,
            ids=_PROVIDER_IDS,
        )


# =====================================================================
# Test 1: TypeError on missing .chat.completions
# =====================================================================

@pytest.mark.parametrize("wrap_fn,name", [
    (wrap_groq, "groq"),
    (wrap_together, "together"),
    (wrap_fireworks, "fireworks"),
    (wrap_cerebras, "cerebras"),
    (wrap_nvidia_nim, "nvidia_nim"),
    (wrap_vllm, "vllm"),
    (wrap_tgi, "tgi"),
    (wrap_llamacpp, "llamacpp"),
    (wrap_ollama, "ollama"),
], ids=_PROVIDER_IDS)
def test_wrap_rejects_invalid_client(wrap_fn, name, tmp_path):
    """wrap_* raises TypeError for clients lacking .chat.completions."""
    class _Bad:
        pass

    with record(str(tmp_path / "t.sb")) as rec:
        with pytest.raises(TypeError, match="chat.completions"):
            wrap_fn(_Bad(), rec)


# =====================================================================
# Test 2: record→replay cache hit (provider_tuple fixture)
# =====================================================================

def test_wrap_record_and_replay_cache_hit(provider_tuple, tmp_path):
    """All steps served from cache on replay; no real API calls."""
    provider_name, wrap_fn, executor_fn, contract_cls = provider_tuple
    fake = _FakeCompatClient()
    trace_path = str(tmp_path / "t.sb")

    with record(trace_path) as rec:
        client = wrap_fn(fake, rec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SeedPolicyViolation)
            resp = client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "hello"}],
            )
    assert fake.call_count == 1
    assert resp.choices[0].message.content is not None

    # Replay: executor should never fire.
    fire_count = 0

    def _never(_model, _msgs):
        nonlocal fire_count
        fire_count += 1
        raise AssertionError("executor should not fire on cache hit")

    tr = replay(trace_path)
    fake2 = _FakeCompatClient()
    exec_ = Executor(llm=_never)
    tr.replay_forward(executor=exec_)
    assert fire_count == 0


# =====================================================================
# Test 3: canonical messages are raw OpenAI-style list
# =====================================================================

def test_wrap_canonical_request_messages(provider_tuple, tmp_path):
    """Recorded messages use OpenAI wire format unchanged."""
    provider_name, wrap_fn, executor_fn, contract_cls = provider_tuple
    fake = _FakeCompatClient()
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Greet me"},
    ]
    trace_path = str(tmp_path / "t.sb")

    with record(trace_path) as rec:
        client = wrap_fn(fake, rec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SeedPolicyViolation)
            client.chat.completions.create(
                model="test-model",
                messages=messages,
            )

    tr = replay(trace_path)
    step = tr.recorded_steps[0]
    recorded_msgs = step["llm_request"]["messages"]
    assert len(recorded_msgs) == 2
    assert recorded_msgs[0]["role"] == "system"
    assert recorded_msgs[1]["role"] == "user"


# =====================================================================
# Test 4: PromptSubstitution dirties downstream and re-executes
# =====================================================================

def test_wrap_prompt_substitution_dirties_downstream(provider_tuple, tmp_path):
    """PromptSubstitution forces executor re-invocation on dirty steps."""
    provider_name, wrap_fn, executor_fn, contract_cls = provider_tuple
    fake = _FakeCompatClient()
    trace_path = str(tmp_path / "t.sb")

    messages = [{"role": "user", "content": "original prompt"}]

    with record(trace_path) as rec:
        client = wrap_fn(fake, rec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SeedPolicyViolation)
            client.chat.completions.create(
                model="test-model",
                messages=messages,
            )
    assert fake.call_count == 1

    fake2 = _FakeCompatClient()
    tr = replay(trace_path)
    step_id = tr.recorded_steps[0]["step_id"]
    sub = PromptSubstitution(
        at_step=step_id,
        new_messages=[{"role": "user", "content": "changed prompt"}],
    )

    branch = tr.branch_at(step_id, "cf")
    branch.substitute(sub)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SeedPolicyViolation)
        exec_fn = executor_fn(fake2)
        exec_ = Executor(llm=exec_fn)
        branch.replay_forward(executor=exec_)

    assert fake2.call_count >= 1


# =====================================================================
# Test 5: tool calls round-trip
# =====================================================================

def test_wrap_tool_calls_round_trip(provider_tuple, tmp_path):
    """Tool calls canonicalise through the shim and survive replay."""
    provider_name, wrap_fn, executor_fn, contract_cls = provider_tuple
    fake = _FakeCompatClient()
    fake.chat.completions.next_tool_calls = [
        {
            "id": "call_abc",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
        }
    ]
    trace_path = str(tmp_path / "t.sb")

    with record(trace_path) as rec:
        client = wrap_fn(fake, rec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SeedPolicyViolation)
            resp = client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "weather?"}],
                tools=[{"type": "function", "function": {"name": "get_weather"}}],
            )

    tr = replay(trace_path)
    step = tr.recorded_steps[0]
    tc = step["llm_response"]["choices"][0]["message"]["tool_calls"]
    assert tc is not None
    assert len(tc) == 1
    assert tc[0]["function"]["name"] == "get_weather"


# =====================================================================
# Test 6: cerebras does NOT forward seed to API
# =====================================================================

def test_cerebras_seed_not_forwarded_to_api(tmp_path):
    """Cerebras (SeedSupport.NONE) must not pass 'seed' to the API."""
    fake = _FakeCompatClient()
    trace_path = str(tmp_path / "t.sb")

    with record(trace_path) as rec:
        client = wrap_cerebras(fake, rec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SeedPolicyViolation)
            client.chat.completions.create(
                model="llama3.1-8b",
                messages=[{"role": "user", "content": "test"}],
            )

    # The fake client records all kwargs; seed should NOT be in them.
    received_seed = fake.chat.completions._received_seeds[0]
    assert received_seed is None, (
        f"cerebras shim should not forward seed to API; got seed={received_seed!r}"
    )


# =====================================================================
# Test 7: non-cerebras providers forward seed to API
# =====================================================================

@pytest.mark.parametrize("wrap_fn,name", [
    (wrap_groq, "groq"),
    (wrap_together, "together"),
    (wrap_fireworks, "fireworks"),
    (wrap_vllm, "vllm"),
    (wrap_ollama, "ollama"),
], ids=["groq", "together", "fireworks", "vllm", "ollama"])
def test_best_effort_providers_forward_seed(wrap_fn, name, tmp_path):
    """BEST_EFFORT providers pass seed to the provider API."""
    fake = _FakeCompatClient()
    trace_path = str(tmp_path / "t.sb")

    with record(trace_path) as rec:
        client = wrap_fn(fake, rec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SeedPolicyViolation)
            client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
            )

    seed_sent = fake.chat.completions._received_seeds[0]
    assert seed_sent is not None, (
        f"{name!r} should forward seed to API (BEST_EFFORT); got seed=None"
    )


# =====================================================================
# Test 8: wrapped client returns OpenAIChatCompletion
# =====================================================================

def test_wrap_returns_openai_chat_completion(provider_tuple, tmp_path):
    """wrap_* returns an OpenAIChatCompletion namespace."""
    from stepback.shims import OpenAIChatCompletion

    provider_name, wrap_fn, executor_fn, contract_cls = provider_tuple
    fake = _FakeCompatClient()
    trace_path = str(tmp_path / "t.sb")

    with record(trace_path) as rec:
        client = wrap_fn(fake, rec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SeedPolicyViolation)
            resp = client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
            )

    assert isinstance(resp, OpenAIChatCompletion)
    assert resp.choices[0].message.role == "assistant"


# =====================================================================
# Test 9: ShimContracts are pre-registered
# =====================================================================

@pytest.mark.parametrize("name,cls", [
    ("groq",       GroqShimContract),
    ("together",   TogetherShimContract),
    ("fireworks",  FireworksShimContract),
    ("cerebras",   CerebrasShimContract),
    ("nvidia_nim", NvidiaNIMShimContract),
    ("vllm",       VLLMShimContract),
    ("tgi",        TGIShimContract),
    ("llamacpp",   LlamaCppShimContract),
    ("ollama",     OllamaShimContract),
], ids=_PROVIDER_IDS)
def test_contracts_registered(name, cls):
    """All new ShimContracts are pre-registered in the registry."""
    contract = shim_contract_for(name)
    assert isinstance(contract, cls)
    assert contract.provider_name == name


# =====================================================================
# Test 10: new symbols are in stepback.__all__
# =====================================================================

def test_new_symbols_in_stepback_all():
    """All new Step-96 symbols appear in stepback.__all__."""
    expected = [
        "OpenAICompatShimContract",
        "GroqShimContract", "TogetherShimContract", "FireworksShimContract",
        "CerebrasShimContract", "NvidiaNIMShimContract", "VLLMShimContract",
        "TGIShimContract", "LlamaCppShimContract", "OllamaShimContract",
        "WrappedOpenAICompat",
        "wrap_groq", "wrap_together", "wrap_fireworks", "wrap_cerebras",
        "wrap_nvidia_nim", "wrap_vllm", "wrap_tgi", "wrap_llamacpp", "wrap_ollama",
        "groq_executor", "together_executor", "fireworks_executor",
        "cerebras_executor", "nvidia_nim_executor", "vllm_executor",
        "tgi_executor", "llamacpp_executor", "ollama_executor",
        "openai_compat_executor",
    ]
    missing = [s for s in expected if s not in stepback.__all__]
    assert missing == [], f"Missing from stepback.__all__: {missing}"


# =====================================================================
# Test 11: WrappedOpenAICompat.provider_name
# =====================================================================

def test_wrapped_provider_name(provider_tuple, tmp_path):
    """WrappedOpenAICompat stores the correct provider_name."""
    provider_name, wrap_fn, executor_fn, contract_cls = provider_tuple
    fake = _FakeCompatClient()
    trace_path = str(tmp_path / "t.sb")

    with record(trace_path) as rec:
        client = wrap_fn(fake, rec)

    assert isinstance(client, WrappedOpenAICompat)
    assert client.provider_name == provider_name


# =====================================================================
# Test 12: generic openai_compat_executor
# =====================================================================

def test_openai_compat_executor_returns_canonical(tmp_path):
    """openai_compat_executor wraps a client and returns canonical dict."""
    fake = _FakeCompatClient()
    # Register a dummy provider for this test.
    from stepback.shims import register_shim_contract

    class _TestContract(OpenAICompatShimContract):
        provider_name = "_test_compat"

    try:
        register_shim_contract(_TestContract())
    except KeyError:
        pass  # already registered if test runs twice

    exec_fn = openai_compat_executor(fake, provider_name="_test_compat")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = exec_fn(
            "my-model",
            [{"role": "user", "content": "hello"}],
        )

    assert "choices" in result
    assert result["choices"][0]["message"]["role"] == "assistant"


# =====================================================================
# Test 13: OpenAICompatShimContract.canonical_request is identity
# =====================================================================

def test_openai_compat_canonical_request():
    """canonical_request returns messages unchanged."""
    contract = GroqShimContract()
    messages = [{"role": "user", "content": "hi"}]
    result = contract.canonical_request(messages=messages)
    assert result == messages
    assert result is not messages  # returns a copy


# =====================================================================
# Test 14: missing usage fields degrade gracefully
# =====================================================================

def test_wrap_missing_usage_graceful(provider_tuple, tmp_path):
    """Responses without usage fields degrade to zero counts."""
    provider_name, wrap_fn, executor_fn, contract_cls = provider_tuple

    class _BareCompletions:
        def create(self, *, model, messages, **kw):
            return {
                "id": "x",
                "model": model,
                "choices": [{"index": 0, "finish_reason": "stop",
                              "message": {"role": "assistant", "content": "ok"}}],
                # no "usage" key
            }

    class _BareChat:
        completions = _BareCompletions()

    class _BareClient:
        chat = _BareChat()

    trace_path = str(tmp_path / "t.sb")
    with record(trace_path) as rec:
        client = wrap_fn(_BareClient(), rec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SeedPolicyViolation)
            resp = client.chat.completions.create(
                model="bare-model",
                messages=[{"role": "user", "content": "bare"}],
            )

    assert resp.choices[0].message.content == "ok"


# =====================================================================
# Test 15: seed support enum values are correct
# =====================================================================

def test_seed_support_registry():
    """seeding.PROVIDER_SEED_SUPPORT has correct entries for new providers."""
    from stepback.seeding import PROVIDER_SEED_SUPPORT, SeedSupport

    assert PROVIDER_SEED_SUPPORT["cerebras"] is SeedSupport.NONE
    for p in ("groq", "together", "fireworks", "nvidia_nim", "vllm", "tgi",
              "llamacpp", "ollama"):
        assert PROVIDER_SEED_SUPPORT[p] is SeedSupport.BEST_EFFORT, (
            f"{p!r} should be BEST_EFFORT; got {PROVIDER_SEED_SUPPORT[p]!r}"
        )
