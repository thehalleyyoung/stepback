"""Tests for Step 69: deterministic seeding policy for LLM / tool executors.

Covers:
* :class:`~stepback.seeding.SeedSupport` enum values.
* :class:`~stepback.seeding.SeedWarnLevel` enum values.
* :class:`~stepback.seeding.PROVIDER_SEED_SUPPORT` registry.
* :class:`~stepback.seeding.SeedPolicy` — default_seed, effective_seed,
  seed_support, check() at SILENT / WARN / ERROR levels.
* :class:`~stepback.seeding.SeedPolicyViolation` emitted as UserWarning.
* :class:`~stepback.seeding.SeedPolicyError` raised at ERROR level.
* :func:`~stepback.seeding.get_seed_policy` / :func:`~stepback.seeding.set_seed_policy`
  module-level singleton management.
* Integration with the OpenAI, Anthropic, and Bedrock shims: warnings
  emitted for Anthropic/Bedrock, no warnings for OpenAI.
* OpenAI shim actually passes the effective seed to the underlying API call.
* Anthropic shim does NOT pass seed to the underlying API call.
* Bedrock shim does NOT pass seed to the underlying API call.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional

import pytest

import stepback
from stepback.seeding import (
    DEFAULT_SEED_POLICY,
    PROVIDER_SEED_SUPPORT,
    SeedPolicy,
    SeedPolicyError,
    SeedPolicyViolation,
    SeedSupport,
    SeedWarnLevel,
    get_seed_policy,
    set_seed_policy,
)
from stepback.shims import wrap_anthropic, wrap_bedrock, wrap_openai


# ---------------------------------------------------------------------------
# SeedSupport enum
# ---------------------------------------------------------------------------


def test_seed_support_values() -> None:
    assert SeedSupport.FULL == "full"
    assert SeedSupport.BEST_EFFORT == "best_effort"
    assert SeedSupport.NONE == "none"


def test_seed_support_is_str_subclass() -> None:
    for member in SeedSupport:
        assert isinstance(member, str)


# ---------------------------------------------------------------------------
# SeedWarnLevel enum
# ---------------------------------------------------------------------------


def test_seed_warn_level_values() -> None:
    assert SeedWarnLevel.SILENT == "silent"
    assert SeedWarnLevel.WARN == "warn"
    assert SeedWarnLevel.ERROR == "error"


# ---------------------------------------------------------------------------
# PROVIDER_SEED_SUPPORT registry
# ---------------------------------------------------------------------------


def test_registry_has_known_providers() -> None:
    assert "openai" in PROVIDER_SEED_SUPPORT
    assert "anthropic" in PROVIDER_SEED_SUPPORT
    assert "bedrock" in PROVIDER_SEED_SUPPORT
    assert "gemini" in PROVIDER_SEED_SUPPORT


def test_registry_openai_is_full() -> None:
    assert PROVIDER_SEED_SUPPORT["openai"] is SeedSupport.FULL


def test_registry_anthropic_is_none() -> None:
    assert PROVIDER_SEED_SUPPORT["anthropic"] is SeedSupport.NONE


def test_registry_bedrock_is_none() -> None:
    assert PROVIDER_SEED_SUPPORT["bedrock"] is SeedSupport.NONE


def test_registry_gemini_is_best_effort() -> None:
    assert PROVIDER_SEED_SUPPORT["gemini"] is SeedSupport.BEST_EFFORT


# ---------------------------------------------------------------------------
# SeedPolicy.effective_seed
# ---------------------------------------------------------------------------


def test_effective_seed_uses_caller_if_provided() -> None:
    policy = SeedPolicy(default_seed=42)
    assert policy.effective_seed(7) == 7


def test_effective_seed_falls_back_to_default() -> None:
    policy = SeedPolicy(default_seed=99)
    assert policy.effective_seed(None) == 99


def test_effective_seed_none_default() -> None:
    policy = SeedPolicy(default_seed=None)
    assert policy.effective_seed(None) is None


# ---------------------------------------------------------------------------
# SeedPolicy.seed_support
# ---------------------------------------------------------------------------


def test_seed_support_lookup_known() -> None:
    policy = SeedPolicy()
    assert policy.seed_support("openai") is SeedSupport.FULL
    assert policy.seed_support("anthropic") is SeedSupport.NONE


def test_seed_support_case_insensitive() -> None:
    policy = SeedPolicy()
    assert policy.seed_support("OpenAI") is SeedSupport.FULL
    assert policy.seed_support("ANTHROPIC") is SeedSupport.NONE


def test_seed_support_unknown_provider_defaults_none() -> None:
    policy = SeedPolicy()
    assert policy.seed_support("unknown_provider_xyz") is SeedSupport.NONE


def test_seed_support_override() -> None:
    policy = SeedPolicy(provider_overrides={"anthropic": SeedSupport.BEST_EFFORT})
    assert policy.seed_support("anthropic") is SeedSupport.BEST_EFFORT


# ---------------------------------------------------------------------------
# SeedPolicy.check — SILENT level
# ---------------------------------------------------------------------------


def test_check_silent_no_warning_for_unsupported_provider() -> None:
    policy = SeedPolicy(default_seed=42, warn_level=SeedWarnLevel.SILENT)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = policy.check("anthropic", seed=None, temperature=0.7)
    assert result == 42  # effective_seed = default


def test_check_silent_returns_effective_seed() -> None:
    policy = SeedPolicy(default_seed=10, warn_level=SeedWarnLevel.SILENT)
    assert policy.check("anthropic", seed=5) == 5
    assert policy.check("anthropic", seed=None) == 10


# ---------------------------------------------------------------------------
# SeedPolicy.check — WARN level
# ---------------------------------------------------------------------------


def test_check_warn_emits_warning_for_anthropic() -> None:
    policy = SeedPolicy(default_seed=42, warn_level=SeedWarnLevel.WARN)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = policy.check("anthropic", seed=None, temperature=0.0)
    assert result == 42
    assert any(issubclass(w.category, SeedPolicyViolation) for w in caught), (
        f"Expected SeedPolicyViolation warning, got: {[w.category for w in caught]}"
    )


def test_check_warn_message_contains_provider() -> None:
    policy = SeedPolicy(default_seed=42, warn_level=SeedWarnLevel.WARN)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        policy.check("bedrock", seed=None, temperature=0.0, model="claude-3")
    messages = [str(w.message) for w in caught if issubclass(w.category, SeedPolicyViolation)]
    assert messages, "Expected at least one SeedPolicyViolation"
    assert "bedrock" in messages[0]


def test_check_warn_no_warning_for_openai() -> None:
    policy = SeedPolicy(default_seed=42, warn_level=SeedWarnLevel.WARN)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        policy.check("openai", seed=None, temperature=0.7)
    violation_warnings = [w for w in caught if issubclass(w.category, SeedPolicyViolation)]
    assert not violation_warnings, (
        f"Unexpected SeedPolicyViolation for openai: {violation_warnings}"
    )


def test_check_warn_no_warning_when_seed_is_none_and_default_is_none() -> None:
    """When both caller seed and default seed are None, no seed → no violation."""
    policy = SeedPolicy(default_seed=None, warn_level=SeedWarnLevel.WARN)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        policy.check("anthropic", seed=None, temperature=0.7)
    violation_warnings = [w for w in caught if issubclass(w.category, SeedPolicyViolation)]
    assert not violation_warnings


# ---------------------------------------------------------------------------
# SeedPolicy.check — ERROR level
# ---------------------------------------------------------------------------


def test_check_error_raises_for_unsupported_provider() -> None:
    policy = SeedPolicy(default_seed=42, warn_level=SeedWarnLevel.ERROR)
    with pytest.raises(SeedPolicyError, match="anthropic"):
        policy.check("anthropic", seed=None, temperature=0.0)


def test_check_error_not_raised_for_supported_provider() -> None:
    policy = SeedPolicy(default_seed=42, warn_level=SeedWarnLevel.ERROR)
    result = policy.check("openai", seed=None, temperature=0.7)
    assert result == 42


def test_check_error_not_raised_when_no_seed() -> None:
    """No seed configured means nothing is being violated."""
    policy = SeedPolicy(default_seed=None, warn_level=SeedWarnLevel.ERROR)
    result = policy.check("anthropic", seed=None, temperature=0.7)
    assert result is None


# ---------------------------------------------------------------------------
# get_seed_policy / set_seed_policy
# ---------------------------------------------------------------------------


def test_get_seed_policy_returns_default() -> None:
    # Reset to canonical default first.
    original = get_seed_policy()
    try:
        assert isinstance(get_seed_policy(), SeedPolicy)
    finally:
        set_seed_policy(original)


def test_set_seed_policy_replaces_default() -> None:
    original = get_seed_policy()
    new_policy = SeedPolicy(default_seed=7, warn_level=SeedWarnLevel.SILENT)
    try:
        set_seed_policy(new_policy)
        assert get_seed_policy() is new_policy
    finally:
        set_seed_policy(original)


def test_DEFAULT_SEED_POLICY_updated_by_set() -> None:
    original = get_seed_policy()
    new_policy = SeedPolicy(default_seed=99)
    try:
        set_seed_policy(new_policy)
        import stepback.seeding as seeding_mod
        assert seeding_mod.DEFAULT_SEED_POLICY is new_policy
    finally:
        set_seed_policy(original)


# ---------------------------------------------------------------------------
# Integration with shims — helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeOAICompletions:
    """Minimal duck-typed OpenAI completions fake that records kwargs."""
    calls: list

    def create(self, *, model: str, messages: list, **kwargs: Any) -> dict:
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        text = f"resp-{len(self.calls)}"
        return {
            "id": "cmp-001",
            "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }


@dataclass
class _FakeOAIChat:
    completions: _FakeOAICompletions


@dataclass
class _FakeOAI:
    chat: _FakeOAIChat

    @classmethod
    def make(cls) -> "_FakeOAI":
        return cls(chat=_FakeOAIChat(completions=_FakeOAICompletions(calls=[])))


@dataclass
class _FakeAnthropicMessages:
    calls: list

    def create(self, *, model: str, messages: list, system: Any = None,
               max_tokens: int = 1024, **kwargs: Any) -> dict:
        self.calls.append({"model": model, "messages": messages,
                           "system": system, "kwargs": kwargs})
        text = f"anth-{len(self.calls)}"
        return {
            "id": "msg-001",
            "model": model,
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }


@dataclass
class _FakeAnthropic:
    messages: _FakeAnthropicMessages

    @classmethod
    def make(cls) -> "_FakeAnthropic":
        return cls(messages=_FakeAnthropicMessages(calls=[]))


@dataclass
class _FakeBedrock:
    """Fake AWS Bedrock client with .converse()"""
    calls: list

    def converse(self, *, modelId: str, messages: list,
                 inferenceConfig: Optional[dict] = None, **kwargs: Any) -> dict:
        self.calls.append({
            "modelId": modelId, "messages": messages,
            "inferenceConfig": inferenceConfig, "kwargs": kwargs
        })
        return {
            "output": {"message": {"role": "assistant",
                                   "content": [{"text": "bedrock-resp"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8},
        }

    @classmethod
    def make(cls) -> "_FakeBedrock":
        return cls(calls=[])


# ---------------------------------------------------------------------------
# Integration: OpenAI shim passes seed to the API
# ---------------------------------------------------------------------------


def test_openai_shim_passes_effective_seed_to_api(tmp_path: Any) -> None:
    """OpenAI supports seed; the shim should forward it to the underlying API."""
    trace_path = str(tmp_path / "trace.sb")
    fake = _FakeOAI.make()
    policy = SeedPolicy(default_seed=77, warn_level=SeedWarnLevel.SILENT)

    with stepback.record(trace_path) as rec:
        client = wrap_openai(fake, rec, default_model="gpt-4o", seed_policy=policy)
        client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])

    assert fake.chat.completions.calls, "Expected at least one API call"
    call = fake.chat.completions.calls[0]
    assert call["kwargs"].get("seed") == 77, (
        f"Expected seed=77 forwarded to OpenAI API, got: {call['kwargs']}"
    )


def test_openai_shim_caller_seed_overrides_default(tmp_path: Any) -> None:
    trace_path = str(tmp_path / "trace.sb")
    fake = _FakeOAI.make()
    policy = SeedPolicy(default_seed=42, warn_level=SeedWarnLevel.SILENT)

    with stepback.record(trace_path) as rec:
        client = wrap_openai(fake, rec, default_model="gpt-4o", seed_policy=policy)
        client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}], seed=5
        )

    call = fake.chat.completions.calls[0]
    assert call["kwargs"].get("seed") == 5


# ---------------------------------------------------------------------------
# Integration: Anthropic shim does NOT pass seed to the API
# ---------------------------------------------------------------------------


def test_anthropic_shim_does_not_pass_seed_to_api(tmp_path: Any) -> None:
    """Anthropic doesn't support seed; the shim must strip it before the call."""
    trace_path = str(tmp_path / "trace.sb")
    fake = _FakeAnthropic.make()
    policy = SeedPolicy(default_seed=42, warn_level=SeedWarnLevel.SILENT)

    with stepback.record(trace_path) as rec:
        client = wrap_anthropic(fake, rec, seed_policy=policy)
        client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "hi"}],
        )

    call = fake.messages.calls[0]
    assert "seed" not in call["kwargs"], (
        f"Seed must not be forwarded to Anthropic API; got kwargs={call['kwargs']}"
    )


def test_anthropic_shim_emits_warning_by_default(tmp_path: Any) -> None:
    """Default policy warns when seed is configured for Anthropic."""
    trace_path = str(tmp_path / "trace.sb")
    fake = _FakeAnthropic.make()
    # Default policy has default_seed=42 and warn_level=WARN.
    policy = SeedPolicy(default_seed=42, warn_level=SeedWarnLevel.WARN)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with stepback.record(trace_path) as rec:
            client = wrap_anthropic(fake, rec, seed_policy=policy)
            client.messages.create(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": "hi"}],
            )

    violation_warnings = [w for w in caught if issubclass(w.category, SeedPolicyViolation)]
    assert violation_warnings, "Expected SeedPolicyViolation warning for Anthropic"


def test_anthropic_shim_silent_no_warning(tmp_path: Any) -> None:
    """SILENT policy never emits warnings."""
    trace_path = str(tmp_path / "trace.sb")
    fake = _FakeAnthropic.make()
    policy = SeedPolicy(default_seed=42, warn_level=SeedWarnLevel.SILENT)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with stepback.record(trace_path) as rec:
            client = wrap_anthropic(fake, rec, seed_policy=policy)
            client.messages.create(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": "hi"}],
            )

    violation_warnings = [w for w in caught if issubclass(w.category, SeedPolicyViolation)]
    assert not violation_warnings, (
        f"Expected no SeedPolicyViolation with SILENT policy, got: {violation_warnings}"
    )


# ---------------------------------------------------------------------------
# Integration: Bedrock shim does NOT pass seed to inferenceConfig
# ---------------------------------------------------------------------------


def test_bedrock_shim_does_not_pass_seed_in_inferenceconfig(tmp_path: Any) -> None:
    """Bedrock doesn't support seed in inferenceConfig; shim must strip it."""
    trace_path = str(tmp_path / "trace.sb")
    fake = _FakeBedrock.make()
    policy = SeedPolicy(default_seed=42, warn_level=SeedWarnLevel.SILENT)

    with stepback.record(trace_path) as rec:
        client = wrap_bedrock(fake, rec, seed_policy=policy)
        client.converse(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 256},
        )

    call = fake.calls[0]
    cfg = call.get("inferenceConfig") or {}
    assert "seed" not in cfg, (
        f"seed must not be forwarded in Bedrock inferenceConfig; got {cfg}"
    )


def test_bedrock_shim_emits_warning_by_default(tmp_path: Any) -> None:
    """Default policy warns when seed is configured for Bedrock."""
    trace_path = str(tmp_path / "trace.sb")
    fake = _FakeBedrock.make()
    policy = SeedPolicy(default_seed=42, warn_level=SeedWarnLevel.WARN)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with stepback.record(trace_path) as rec:
            client = wrap_bedrock(fake, rec, seed_policy=policy)
            client.converse(
                modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
                messages=[{"role": "user", "content": [{"text": "hi"}]}],
            )

    violation_warnings = [w for w in caught if issubclass(w.category, SeedPolicyViolation)]
    assert violation_warnings, "Expected SeedPolicyViolation warning for Bedrock"


# ---------------------------------------------------------------------------
# Integration: seed recorded in trace metadata
# ---------------------------------------------------------------------------


def test_seed_recorded_in_trace(tmp_path: Any) -> None:
    """The effective seed must appear in the recorded trace step."""
    trace_path = str(tmp_path / "trace.sb")
    fake = _FakeOAI.make()
    policy = SeedPolicy(default_seed=55, warn_level=SeedWarnLevel.SILENT)

    with stepback.record(trace_path) as rec:
        client = wrap_openai(fake, rec, default_model="gpt-4o", seed_policy=policy)
        client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])

    # Read the trace back and verify the seed is stored.
    trace = stepback.replay(trace_path)
    llm_steps = [s for s in trace.recorded_steps if s.get("step_kind") == "llm_call"]
    assert llm_steps, "Expected at least one llm_call step in trace"
    recorded_seed = llm_steps[0]["inputs"].get("seed")
    assert recorded_seed == 55, f"Expected seed=55 in trace, got {recorded_seed}"


# ---------------------------------------------------------------------------
# Public API: seeding symbols importable from top-level
# ---------------------------------------------------------------------------


def test_seeding_symbols_in_public_api() -> None:
    for name in (
        "SeedSupport", "SeedWarnLevel", "SeedPolicyViolation", "SeedPolicyError",
        "SeedPolicy", "PROVIDER_SEED_SUPPORT", "DEFAULT_SEED_POLICY",
        "get_seed_policy", "set_seed_policy",
    ):
        assert hasattr(stepback, name), f"stepback.{name} missing from public API"


def test_seeding_symbols_have_docstrings() -> None:
    import inspect
    for name in (
        "SeedSupport", "SeedWarnLevel", "SeedPolicyViolation", "SeedPolicyError",
        "SeedPolicy", "get_seed_policy", "set_seed_policy",
    ):
        obj = getattr(stepback, name)
        doc = inspect.getdoc(obj)
        assert doc and len(doc.strip()) >= 3, f"stepback.{name} missing docstring"
