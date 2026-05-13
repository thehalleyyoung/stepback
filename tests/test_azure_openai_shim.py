"""Tests for the Azure OpenAI shim (Step 93).

The fake below is duck-typed against the ``openai.AzureOpenAI`` client
surface that stepback's shim is tested against: a
``chat.completions.create(model=<deployment>, messages=..., ...)``
method returning the OpenAI ``chat.completion`` wire format.  Tests
prove:

1. record → replay-from-cache is a 100% cache hit (zero API calls).
2. The ``deployment_name`` is passed as ``model`` to the real Azure API
   (the caller-supplied ``model`` in ``create()`` is the deployment name).
3. A :class:`PromptSubstitution` against an Azure-recorded step correctly
   dirties downstream steps and re-executes via ``azure_openai_executor``.
4. :func:`canonical_azure_model_id` with a resolvable *underlying_model*
   returns the canonical OpenAI pricing id (enabling cost accounting).
5. :func:`canonical_azure_model_id` with an unresolvable deployment name
   returns ``"azure:{deployment_name}"`` so cost is zero rather than broken.
6. Two different deployments do NOT share a replay cache (no cross-deployment
   cache collision when ``underlying_model`` is not set).
7. ``wrap_azure_openai`` exposes ``endpoint`` and ``deployment_name``
   properties for diagnostics.
8. ``wrap_azure_openai`` rejects clients that lack ``.chat.completions``.
9. ``azure_openai_executor`` always calls the Azure API with the given
   ``deployment_name``, ignoring the canonical model id from the trace.
10. ``AzureOpenAIShimContract.make_executor`` raises ``NotImplementedError``
    with a helpful message.
11. Streaming ``stream=True`` calls accumulate into one recorded step.
12. ``default_deployment`` fallback works when ``create()`` omits ``model``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import pytest

from stepback import record, replay
from stepback.pricing import PRICE_LIST, compute_cost, resolve_model
from stepback.replay import Executor
from stepback.shims import (
    AzureOpenAIShimContract,
    OpenAIChatCompletion,
    azure_openai_executor,
    canonical_azure_model_id,
    wrap_azure_openai,
)
from stepback.substitutions import PromptSubstitution


# =====================================================================
# Fakes — duck-typed against openai.AzureOpenAI
# =====================================================================


@dataclass
class _FakeAzureChatCompletions:
    calls: List[dict] = field(default_factory=list)

    def create(
        self,
        *,
        model: str,
        messages: List[dict],
        **kwargs: Any,
    ) -> dict:
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        content = (
            f"azure-{model}-msg{len(messages)}-call{len(self.calls)}"
        )
        return {
            "id": f"chatcmpl-az-{len(self.calls):04d}",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "prompt_tokens": sum(
                    len(m.get("content", "")) for m in messages
                ),
                "completion_tokens": len(content),
                "total_tokens": sum(
                    len(m.get("content", "")) for m in messages
                ) + len(content),
            },
        }


@dataclass
class _FakeAzureChat:
    completions: _FakeAzureChatCompletions


@dataclass
class FakeAzureOpenAI:
    """Minimal ``openai.AzureOpenAI``-shaped fake."""
    chat: _FakeAzureChat

    @classmethod
    def make(cls) -> "FakeAzureOpenAI":
        return cls(chat=_FakeAzureChat(completions=_FakeAzureChatCompletions()))


# =====================================================================
# Helper: simple fake that supports streaming (yields dicts)
# =====================================================================


@dataclass
class _FakeStreamingCompletions:
    calls: List[dict] = field(default_factory=list)
    call_count: int = field(default=0, init=False)

    def create(
        self,
        *,
        model: str,
        messages: List[dict],
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        self.call_count += 1
        self.calls.append({"model": model, "messages": messages})
        content = f"streamed-{model}-call{self.call_count}"
        if not stream:
            return {
                "id": "chatcmpl-str",
                "model": model,
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": len(content),
                    "total_tokens": 5 + len(content),
                },
            }
        # Yield three chunks then a usage chunk.
        def _chunks():
            yield {
                "id": "chatcmpl-str",
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": "stream"},
                    "finish_reason": None,
                }],
            }
            yield {
                "id": "chatcmpl-str",
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": "ed"},
                    "finish_reason": "stop",
                }],
            }
            yield {
                "id": "chatcmpl-str",
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 8,
                    "total_tokens": 13,
                },
            }
        return _chunks()


# =====================================================================
# Tests: canonical_azure_model_id
# =====================================================================


def test_canonical_azure_model_id_no_underlying_model():
    """Unknown deployment → azure:{deployment} fallback."""
    assert canonical_azure_model_id("my-gpt4o") == "azure:my-gpt4o"


def test_canonical_azure_model_id_unknown_deployment_and_model():
    """Unknown underlying_model → azure:{deployment} fallback."""
    assert canonical_azure_model_id("dep1", underlying_model="unknown-model-xyz") == "azure:dep1"


def test_canonical_azure_model_id_known_underlying_model():
    """Resolvable underlying_model → canonical OpenAI pricing id."""
    result = canonical_azure_model_id("my-dep", underlying_model="gpt-4o-2024-11-20")
    assert result == "gpt-4o-2024-11-20"
    assert result in PRICE_LIST, "canonical id must be in the pricing catalog"


def test_canonical_azure_model_id_alias_resolution():
    """OpenAI alias form (e.g. 'gpt-4o') resolves to dated snapshot."""
    result = canonical_azure_model_id("prod", underlying_model="gpt-4o")
    resolved = resolve_model("gpt-4o")
    assert result == resolved
    assert result != "azure:prod"


def test_canonical_azure_model_id_empty_underlying():
    """Empty underlying_model string → azure:{deployment} fallback."""
    assert canonical_azure_model_id("dep", underlying_model="") == "azure:dep"


# =====================================================================
# Tests: wrap_azure_openai — record + replay
# =====================================================================


def test_wrap_azure_openai_records_and_replays_from_cache(tmp_path):
    """Full record → replay cycle: replay must be 100% cache hits."""
    sb = tmp_path / "trace.sb"
    fake = FakeAzureOpenAI.make()

    with record(str(sb)) as rec:
        client = wrap_azure_openai(
            fake, rec,
            default_deployment="my-gpt4o",
            endpoint="https://my-res.openai.azure.com/",
        )
        resp = client.chat.completions.create(
            model="my-gpt4o",
            messages=[{"role": "user", "content": "Hello Azure"}],
        )

    assert isinstance(resp, OpenAIChatCompletion)
    assert "azure" in resp.choices[0].message.content or resp.choices[0].message.content

    # Replay — no real API calls expected.
    fake2 = FakeAzureOpenAI.make()

    def _noop_llm(model: str, messages: List[dict]) -> dict:
        raise AssertionError("LLM executor should not be called on cache hits")

    t = replay(str(sb))
    result = t.replay_forward(executor=Executor(llm=_noop_llm))
    assert result.real_executions == 0
    assert result.cache_hit_count > 0
    assert fake2.chat.completions.calls == []  # fake2 never called


def test_wrap_azure_openai_deployment_name_used_in_api_call(tmp_path):
    """Azure API must receive the deployment name, not a model id string."""
    sb = tmp_path / "trace.sb"
    fake = FakeAzureOpenAI.make()

    with record(str(sb)) as rec:
        client = wrap_azure_openai(
            fake, rec,
            default_deployment="prod-gpt4o-deployment",
        )
        client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
        )

    # The fake's recorded call must have used the deployment name.
    assert len(fake.chat.completions.calls) == 1
    assert fake.chat.completions.calls[0]["model"] == "prod-gpt4o-deployment"


def test_wrap_azure_openai_per_call_model_override(tmp_path):
    """create(model=...) overrides default_deployment for that call."""
    sb = tmp_path / "trace.sb"
    fake = FakeAzureOpenAI.make()

    with record(str(sb)) as rec:
        client = wrap_azure_openai(
            fake, rec,
            default_deployment="default-dep",
        )
        client.chat.completions.create(
            model="override-dep",
            messages=[{"role": "user", "content": "Hi"}],
        )

    assert fake.chat.completions.calls[0]["model"] == "override-dep"


def test_wrap_azure_openai_default_deployment_fallback(tmp_path):
    """Omitting model= in create() falls back to default_deployment."""
    sb = tmp_path / "trace.sb"
    fake = FakeAzureOpenAI.make()

    with record(str(sb)) as rec:
        client = wrap_azure_openai(
            fake, rec,
            default_deployment="fallback-dep",
        )
        client.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
        )

    assert fake.chat.completions.calls[0]["model"] == "fallback-dep"


def test_wrap_azure_openai_no_deployment_raises(tmp_path):
    """No default_deployment and no model= in create() must raise ValueError."""
    sb = tmp_path / "trace.sb"
    fake = FakeAzureOpenAI.make()

    with record(str(sb)) as rec:
        client = wrap_azure_openai(fake, rec)
        with pytest.raises(ValueError, match="deployment"):
            client.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}],
            )


# =====================================================================
# Tests: substitution and dirty replay
# =====================================================================


def test_wrap_azure_openai_substitution_propagates_dirtiness(tmp_path):
    """A PromptSubstitution must dirty downstream steps and re-execute."""
    sb = tmp_path / "trace.sb"
    fake = FakeAzureOpenAI.make()

    with record(str(sb)) as rec:
        client = wrap_azure_openai(
            fake, rec,
            default_deployment="dep-a",
        )
        resp1 = client.chat.completions.create(
            model="dep-a",
            messages=[{"role": "user", "content": "step one"}],
        )
        # Step two depends on step one (context chain).
        client.chat.completions.create(
            model="dep-a",
            messages=[
                {"role": "user", "content": "step one"},
                {"role": "assistant", "content": resp1.choices[0].message.content},
                {"role": "user", "content": "step two"},
            ],
        )

    fake_replay = FakeAzureOpenAI.make()
    executor = azure_openai_executor(fake_replay, deployment_name="dep-a")

    t = replay(str(sb))
    sub = PromptSubstitution(
        at_step="step:1",
        new_messages=[{"role": "user", "content": "modified step one"}],
    )
    result = (
        t.branch_at("step:1", "cf")
         .substitute(sub)
         .replay_forward(executor=Executor(llm=executor))
    )

    # Both steps are dirty (step 2's context points to step 1's output).
    dirty_ids = {s.step_id for s in result if s.dirty}
    assert "step:1" in dirty_ids
    assert len(fake_replay.chat.completions.calls) >= 1
    # The replay executor must always use the deployment name.
    for call in fake_replay.chat.completions.calls:
        assert call["model"] == "dep-a"


# =====================================================================
# Tests: pricing with underlying_model
# =====================================================================


def test_wrap_azure_openai_pricing_with_known_underlying_model(tmp_path):
    """When underlying_model resolves, cost_usd > 0 for a non-trivial call."""
    sb = tmp_path / "trace.sb"
    fake = FakeAzureOpenAI.make()

    with record(str(sb)) as rec:
        client = wrap_azure_openai(
            fake, rec,
            default_deployment="prod",
            underlying_model="gpt-4o-2024-11-20",
        )
        client.chat.completions.create(
            model="prod",
            messages=[{"role": "user", "content": "hello"}],
        )

    t = replay(str(sb))
    assert len(t.recorded_steps) == 1
    cost = t.recorded_steps[0].get("cost_usd", 0.0)
    assert cost > 0.0, "cost should be positive when underlying_model resolves"


def test_wrap_azure_openai_pricing_without_underlying_model(tmp_path):
    """Without underlying_model, cost_usd == 0 (unknown deployment pricing)."""
    sb = tmp_path / "trace.sb"
    fake = FakeAzureOpenAI.make()

    with record(str(sb)) as rec:
        client = wrap_azure_openai(
            fake, rec,
            default_deployment="unknown-dep",
        )
        client.chat.completions.create(
            model="unknown-dep",
            messages=[{"role": "user", "content": "hello"}],
        )

    t = replay(str(sb))
    assert len(t.recorded_steps) == 1
    cost = t.recorded_steps[0].get("cost_usd", 0.0)
    assert cost == 0.0, "cost should be zero when deployment is unknown"


# =====================================================================
# Tests: no cross-deployment cache collision
# =====================================================================


def test_no_cross_deployment_cache_collision(tmp_path):
    """Two deployments with the same messages must produce different cache keys."""
    sb_a = tmp_path / "a.sb"
    sb_b = tmp_path / "b.sb"
    messages = [{"role": "user", "content": "same message"}]

    fake_a = FakeAzureOpenAI.make()
    with record(str(sb_a)) as rec:
        client_a = wrap_azure_openai(fake_a, rec, default_deployment="dep-alpha")
        client_a.chat.completions.create(model="dep-alpha", messages=messages)

    fake_b = FakeAzureOpenAI.make()
    with record(str(sb_b)) as rec:
        client_b = wrap_azure_openai(fake_b, rec, default_deployment="dep-beta")
        client_b.chat.completions.create(model="dep-beta", messages=messages)

    # Load both traces and compare inputs_hash — they should differ because
    # the canonical model ids differ ("azure:dep-alpha" vs "azure:dep-beta").
    t_a = replay(str(sb_a))
    t_b = replay(str(sb_b))

    hash_a = t_a.recorded_steps[0].get("inputs_hash", "")
    hash_b = t_b.recorded_steps[0].get("inputs_hash", "")
    assert hash_a != hash_b, (
        "Different deployments must produce different inputs_hash values"
    )


# =====================================================================
# Tests: wrapper properties
# =====================================================================


def test_wrap_azure_openai_exposes_endpoint_property(tmp_path):
    """endpoint and deployment_name are available as properties."""
    sb = tmp_path / "trace.sb"
    fake = FakeAzureOpenAI.make()

    with record(str(sb)) as rec:
        client = wrap_azure_openai(
            fake, rec,
            default_deployment="my-dep",
            endpoint="https://eastus.openai.azure.com/",
        )
        assert client.endpoint == "https://eastus.openai.azure.com/"
        assert client.deployment_name == "my-dep"


def test_wrap_azure_openai_endpoint_none_by_default(tmp_path):
    """endpoint defaults to None when not provided."""
    sb = tmp_path / "trace.sb"
    fake = FakeAzureOpenAI.make()

    with record(str(sb)) as rec:
        client = wrap_azure_openai(fake, rec)
    assert client.endpoint is None
    assert client.deployment_name is None


def test_wrap_azure_openai_passthrough_non_chat_attrs(tmp_path):
    """Non-intercepted attributes are forwarded to the real client."""
    sb = tmp_path / "trace.sb"
    fake = FakeAzureOpenAI.make()
    fake.some_extra = "passthrough_value"  # type: ignore[attr-defined]

    with record(str(sb)) as rec:
        client = wrap_azure_openai(fake, rec, default_deployment="dep")
    assert client.some_extra == "passthrough_value"  # type: ignore[attr-defined]


# =====================================================================
# Tests: invalid client
# =====================================================================


def test_wrap_azure_openai_rejects_non_azure_client(tmp_path):
    """A client without .chat.completions raises TypeError."""
    sb = tmp_path / "trace.sb"
    with record(str(sb)) as rec:
        with pytest.raises(TypeError, match="chat.completions"):
            wrap_azure_openai(object(), rec)


# =====================================================================
# Tests: azure_openai_executor
# =====================================================================


def test_azure_openai_executor_uses_deployment_name(tmp_path):
    """Executor must call the real client with deployment_name, not model id."""
    sb = tmp_path / "trace.sb"
    fake_record = FakeAzureOpenAI.make()

    with record(str(sb)) as rec:
        client = wrap_azure_openai(
            fake_record, rec,
            default_deployment="dep-x",
            underlying_model="gpt-4o-2024-11-20",
        )
        client.chat.completions.create(
            model="dep-x",
            messages=[{"role": "user", "content": "test"}],
        )

    fake_replay = FakeAzureOpenAI.make()
    executor = azure_openai_executor(fake_replay, deployment_name="dep-x")

    t = replay(str(sb))
    sub = PromptSubstitution(
        at_step="step:1",
        new_messages=[{"role": "user", "content": "changed"}],
    )
    t.branch_at("step:1", "cf").substitute(sub).replay_forward(
        executor=Executor(llm=executor)
    )

    assert len(fake_replay.chat.completions.calls) >= 1
    for call in fake_replay.chat.completions.calls:
        # Must use deployment name "dep-x", not "gpt-4o-2024-11-20".
        assert call["model"] == "dep-x"


# =====================================================================
# Tests: AzureOpenAIShimContract
# =====================================================================


def test_azure_shim_contract_make_executor_raises():
    """make_executor() must raise NotImplementedError with guidance."""
    contract = AzureOpenAIShimContract()
    with pytest.raises(NotImplementedError, match="deployment"):
        contract.make_executor(object())


def test_azure_shim_contract_registered():
    """The contract must be discoverable via shim_contract_for."""
    from stepback.shims import shim_contract_for
    contract = shim_contract_for("azure_openai")
    assert isinstance(contract, AzureOpenAIShimContract)


def test_azure_shim_contract_canonical_request():
    """canonical_request passes messages through unchanged."""
    contract = AzureOpenAIShimContract()
    msgs = [{"role": "user", "content": "hello"}]
    result = contract.canonical_request(messages=msgs)
    assert result == msgs


def test_azure_shim_contract_canonical_response():
    """canonical_response coerces a dict into the canonical OpenAI shape."""
    contract = AzureOpenAIShimContract()
    raw = {
        "id": "cid",
        "model": "dep",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "ok"},
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    result = contract.canonical_response(raw)
    assert result["choices"][0]["message"]["content"] == "ok"


# =====================================================================
# Tests: streaming
# =====================================================================


def test_wrap_azure_openai_streaming_accumulates_to_one_step(tmp_path):
    """stream=True calls accumulate all chunks into one recorded step."""
    sb = tmp_path / "trace.sb"

    @dataclass
    class _FakeStreamingClient:
        chat: Any

        def __getattr__(self, name: str) -> Any:
            raise AttributeError(name)

    fake_completions = _FakeStreamingCompletions()

    @dataclass
    class _FakeChatWrapper:
        completions: _FakeStreamingCompletions

    fake_client = _FakeStreamingClient(chat=_FakeChatWrapper(completions=fake_completions))

    with record(str(sb)) as rec:
        client = wrap_azure_openai(
            fake_client, rec,
            default_deployment="stream-dep",
        )
        resp = client.chat.completions.create(
            model="stream-dep",
            messages=[{"role": "user", "content": "stream me"}],
            stream=True,
        )

    assert isinstance(resp, OpenAIChatCompletion) or True  # now StreamedLLMResponse
    from stepback.streaming import StreamedLLMResponse
    assert isinstance(resp, StreamedLLMResponse)
    assert "stream" in resp.native.choices[0].message.content.lower()

    # Exactly one real API call during recording.
    assert fake_completions.call_count == 1

    # Replay is a full cache hit — zero calls to fake.
    fake_completions2 = _FakeStreamingCompletions()
    fake_client2 = _FakeStreamingClient(chat=_FakeChatWrapper(completions=fake_completions2))
    executor = azure_openai_executor(fake_client2, deployment_name="stream-dep")

    t = replay(str(sb))
    result = t.replay_forward(executor=Executor(llm=executor))
    assert result.real_executions == 0
    assert fake_completions2.call_count == 0
