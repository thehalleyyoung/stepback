"""End-to-end tests for the AWS Bedrock Converse shim.

The fake below is shaped exactly like a boto3 ``bedrock-runtime``
client (the surface stepback's shim is duck-typed against): a single
``converse(modelId=..., messages=..., system=..., inferenceConfig=...,
toolConfig=...)`` method returning the published Bedrock response
shape.  Tests prove:

1. record→replay-from-cache is a 100% cache hit (zero LLM calls).
2. The recorded canonical ``llm_request.messages`` projects Bedrock's
   structured ``content:[{text:...}]`` blocks onto the unified
   OpenAI-style list every other shim shares.
3. A :class:`PromptSubstitution` against a Bedrock-recorded step
   correctly dirties downstream steps and re-executes through the
   replay-side :func:`bedrock_executor`.
4. Bedrock-hosted Claude reuses the native Anthropic pricing row via
   :func:`canonical_bedrock_model_id`, so the recorded ``cost_usd`` is
   non-zero on the very first call.
5. ``toolUse`` content blocks round-trip through the canonicaliser as
   OpenAI ``tool_calls``.
6. A native Bedrock model (``meta.llama3-1-70b-instruct-v1:0``) gets a
   non-zero recorded cost from the new pricing rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import pytest

from stepback import record, replay
from stepback.pricing import (
    PRICE_LIST,
    RATE_TABLE,
    compute_cost,
    resolve_model,
)
from stepback.replay import Executor
from stepback.shims import (
    bedrock_executor,
    canonical_bedrock_model_id,
    wrap_bedrock,
)
from stepback.substitutions import (
    PromptSubstitution,
    SubstitutionSet,
    ToolOutputSubstitution,
)


# =====================================================================
# Fake — duck-typed against boto3 bedrock-runtime
# =====================================================================


@dataclass
class FakeBedrockRuntime:
    calls: List[dict] = field(default_factory=list)
    # Optional override returning a tool_use block for the next call.
    next_tool_use: Optional[dict] = None

    def converse(self, *, modelId: str, messages: List[dict],
                 system: Optional[List[dict]] = None,
                 inferenceConfig: Optional[dict] = None,
                 toolConfig: Optional[dict] = None,
                 **kwargs: Any) -> dict:
        self.calls.append({
            "modelId": modelId,
            "messages": messages,
            "system": system,
            "inferenceConfig": inferenceConfig or {},
            "toolConfig": toolConfig,
            "kwargs": kwargs,
        })
        # Build a deterministic body keyed off message count so a
        # cache-hit replay is bit-identical to the original record.
        last_user_text = ""
        for m in messages:
            if m.get("role") == "user":
                for b in m.get("content", []):
                    if "text" in b:
                        last_user_text = b["text"]
        body = f"bed-{modelId}-msg{len(messages)}-call{len(self.calls)}-echo[{last_user_text}]"

        content_blocks: List[dict] = [{"text": body}]
        stop_reason = "end_turn"
        if self.next_tool_use is not None:
            content_blocks.append({"toolUse": self.next_tool_use})
            stop_reason = "tool_use"
            self.next_tool_use = None

        # Bedrock's Converse usage block.
        in_tok = sum(
            len(b.get("text", "")) for m in messages for b in m.get("content", [])
        )
        if system:
            in_tok += sum(len(b.get("text", "")) for b in system)
        out_tok = len(body)
        return {
            "ResponseMetadata": {"RequestId": f"req-{len(self.calls):04d}"},
            "output": {
                "message": {"role": "assistant", "content": content_blocks},
            },
            "stopReason": stop_reason,
            "usage": {
                "inputTokens": in_tok,
                "outputTokens": out_tok,
                "totalTokens": in_tok + out_tok,
            },
            "metrics": {"latencyMs": 12},
        }


# =====================================================================
# Smoke / type guards
# =====================================================================


def test_wrap_bedrock_rejects_non_bedrock_clients(tmp_path):
    with record(str(tmp_path / "x.sb")) as rec:
        with pytest.raises(TypeError):
            wrap_bedrock(object(), rec)


def test_canonical_bedrock_model_id_aliases_claude():
    assert (
        canonical_bedrock_model_id("anthropic.claude-3-5-sonnet-20241022-v2:0")
        == "claude-3-5-sonnet-20241022"
    )
    # Native Bedrock ids pass through untouched.
    assert (
        canonical_bedrock_model_id("meta.llama3-1-70b-instruct-v1:0")
        == "meta.llama3-1-70b-instruct-v1:0"
    )


# =====================================================================
# Record → replay-from-cache
# =====================================================================


def test_wrap_bedrock_records_and_replays_from_cache(tmp_path):
    trace_path = str(tmp_path / "bed.sb")
    fake = FakeBedrockRuntime()
    with record(trace_path) as rec:
        client = wrap_bedrock(fake, rec)
        resp1 = client.converse(
            modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
            system=[{"text": "be terse"}],
            messages=[{"role": "user", "content": [{"text": "hello"}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 64},
        )
        # Native Bedrock shape preserved for agent code.
        assert resp1["output"]["message"]["role"] == "assistant"
        assert resp1["output"]["message"]["content"][0]["text"].startswith("bed-")
        assert resp1["stopReason"] == "end_turn"
        assert resp1["usage"]["inputTokens"] > 0

        resp2 = client.converse(
            modelId="meta.llama3-1-70b-instruct-v1:0",
            messages=[{"role": "user", "content": [{"text": "again"}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 32},
        )
        assert resp2["output"]["message"]["content"][0]["text"].startswith("bed-")

    t = replay(trace_path)
    assert len(t.recorded_steps) == 2
    s1, s2 = t.recorded_steps
    assert s1["step_kind"] == "llm_call"

    # The recorded request canonicalises Bedrock content blocks into a
    # unified messages list (system promoted, text concatenated).
    msgs = s1["llm_request"]["messages"]
    assert msgs[0] == {"role": "system", "content": "be terse"}
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "hello"

    # Cost is non-zero for both: claude-3-5-haiku via the
    # canonical-id alias, llama3.1-70b via the new pricing row.
    assert s1["cost_usd"] > 0
    assert s2["cost_usd"] > 0
    assert s1["llm_request"]["model"] == "claude-3-5-haiku-20241022"  # canonical id
    assert s2["llm_request"]["model"] == "meta.llama3-1-70b-instruct-v1:0"

    # Replay-from-cache: zero real LLM calls.
    exec_ = Executor()
    result = t.run_replay(subs=SubstitutionSet(), executor=exec_)
    assert all(not s.dirty for s in result)
    assert exec_.real_calls == 0


# =====================================================================
# Substitution + replay-side bedrock_executor
# =====================================================================


def test_bedrock_executor_replays_dirty_step(tmp_path):
    trace_path = str(tmp_path / "bed.sb")
    with record(trace_path) as rec:
        client = wrap_bedrock(FakeBedrockRuntime(), rec)
        client.converse(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            system=[{"text": "sys"}],
            messages=[{"role": "user", "content": [{"text": "u1"}]}],
        )

    t = replay(trace_path)
    fake = FakeBedrockRuntime()
    exec_ = Executor(llm=bedrock_executor(fake))
    sub = PromptSubstitution(
        at_step="step:1",
        new_messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1-changed"},
        ],
    )
    result = (
        t.branch_at("step:1", "cf").substitute(sub).replay_forward(executor=exec_)
    )
    assert exec_.real_calls == 1
    # The replay-side fake saw a faithful Bedrock-shaped call: system
    # extracted, user text re-wrapped in a {"text":...} block, the
    # canonical alias inverted back to the Bedrock modelId.
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["modelId"] == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert call["system"] == [{"text": "sys"}]
    assert call["messages"] == [
        {"role": "user", "content": [{"text": "u1-changed"}]},
    ]
    # And the new step output flowed through the canonicaliser.
    new_step = next(s for s in result if s.step_id == "step:1")
    assert new_step.dirty


def test_bedrock_substitution_propagates_dirtiness_across_steps(tmp_path):
    trace_path = str(tmp_path / "bed.sb")
    fake_record = FakeBedrockRuntime()
    with record(trace_path) as rec:
        client = wrap_bedrock(fake_record, rec)
        client.converse(
            modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
            messages=[{"role": "user", "content": [{"text": "first"}]}],
        )
        client.converse(
            modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
            messages=[{"role": "user", "content": [{"text": "second"}]}],
        )

    t = replay(trace_path)
    fake = FakeBedrockRuntime()
    exec_ = Executor(llm=bedrock_executor(fake))
    sub = PromptSubstitution(
        at_step="step:1",
        new_messages=[{"role": "user", "content": "REPLACED"}],
    )
    result = (
        t.branch_at("step:1", "cf").substitute(sub).replay_forward(executor=exec_)
    )
    dirty_ids = {s.step_id for s in result if s.dirty}
    # step:1's inputs changed; step:2's parent context-pointer rebinds
    # to the new step:1 outputs hash, so it is also dirty.
    assert "step:1" in dirty_ids
    assert "step:2" in dirty_ids
    assert exec_.real_calls == 2
    assert len(fake.calls) == 2


# =====================================================================
# Tool-use content blocks round-trip
# =====================================================================


def test_bedrock_tool_use_canonicalises_to_openai_tool_calls(tmp_path):
    trace_path = str(tmp_path / "bed.sb")
    fake = FakeBedrockRuntime()
    fake.next_tool_use = {
        "toolUseId": "tool_use_xyz",
        "name": "lookup_customer",
        "input": {"id": "C-118"},
    }
    with record(trace_path) as rec:
        client = wrap_bedrock(fake, rec)
        resp = client.converse(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": [{"text": "look up"}]}],
            toolConfig={"tools": [{"toolSpec": {"name": "lookup_customer"}}]},
        )
        # Native Bedrock toolUse block is still in the agent-facing return.
        assert any("toolUse" in b for b in resp["output"]["message"]["content"])
        assert resp["stopReason"] == "tool_use"

    t = replay(trace_path)
    step = t.recorded_steps[0]
    canonical = step["llm_response"]
    # Canonicalised into OpenAI tool_calls.
    msg = canonical["choices"][0]["message"]
    assert canonical["choices"][0]["finish_reason"] == "tool_calls"
    tcs = msg.get("tool_calls") or []
    assert len(tcs) == 1
    assert tcs[0]["function"]["name"] == "lookup_customer"
    assert tcs[0]["function"]["arguments"] == {"id": "C-118"}
    # Tools spec was forwarded into the recorded llm_request.
    assert step["llm_request"].get("tools") is not None


# =====================================================================
# Pricing rows visible in the public catalog
# =====================================================================


def test_bedrock_native_pricing_rows_present():
    for native in [
        "meta.llama3-1-70b-instruct-v1:0",
        "meta.llama3-1-8b-instruct-v1:0",
        "meta.llama3-1-405b-instruct-v1:0",
        "mistral.mistral-large-2407-v1:0",
        "cohere.command-r-plus-v1:0",
        "amazon.nova-pro-v1:0",
        "amazon.nova-lite-v1:0",
        "amazon.nova-micro-v1:0",
    ]:
        assert native in RATE_TABLE
        assert native in PRICE_LIST
        # And cost computes against a real usage block.
        cost = compute_cost(native, {"prompt_tokens": 1000, "completion_tokens": 500})
        assert cost > 0


def test_bedrock_short_aliases_resolve():
    assert resolve_model("llama3.1-70b") == "meta.llama3-1-70b-instruct-v1:0"
    assert resolve_model("nova-pro") == "amazon.nova-pro-v1:0"
    assert resolve_model("mistral-large") == "mistral.mistral-large-2407-v1:0"
