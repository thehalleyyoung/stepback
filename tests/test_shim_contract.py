"""SDK contract tests using recorded cassettes for every provider shim.

Step #37 in ``100_STEPS.md``: duck-typed fakes alone don't catch the case where
an upstream SDK changes its response shape. These tests freeze the *contract*
between each shim in :mod:`stepback.shims` and the SDK shape it accepts. Each
cassette under ``tests/fixtures/sdk_cassettes/<provider>/*.json`` carries a
sample raw SDK response plus a ``contract`` block describing the canonical
``llm_response`` (or recorded tool output) the shim must produce.

For every cassette we exercise three independent code paths:

1. The recording path — ``wrap_*`` against an in-process fake whose only job
   is to hand back the cassette's ``raw_response`` byte-for-byte. We assert
   the recorded ``llm_response`` (or ``outputs``) matches the cassette's
   ``contract`` block, then replay the resulting ``.sb`` and confirm every
   step is served from cache (executor is never reinvoked).
2. The pure coercion path — feed the cassette's ``raw_response`` directly to
   ``_<provider>_to_openai_shape`` (after ``_coerce_<provider>_response``)
   and assert the same canonical contract holds. This locks down the
   coercion functions independently of the recorder plumbing.
3. The duck-typed object path — wrap the dict in a stub object exposing
   ``.model_dump()`` (pydantic-like) or attribute access (dataclass-like)
   and verify the coercion path normalizes it identically. This is the
   surface the *real* SDK's response objects hit.

Adding a new cassette is one JSON file; the test runner discovers it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import pytest

from stepback import record, replay
from stepback.replay import Executor
from stepback.shims import (
    _anthropic_to_openai_shape,
    _bedrock_to_openai_shape,
    _canonicalise_openai_response,
    _coerce_anthropic_response,
    _coerce_bedrock_response,
    _coerce_gemini_response,
    _gemini_to_openai_shape,
    wrap_anthropic,
    wrap_bedrock,
    wrap_gemini,
    wrap_langchain_tool,
    wrap_mcp_session,
    wrap_openai,
)
from stepback.substitutions import SubstitutionSet


CASSETTE_ROOT = Path(__file__).parent / "fixtures" / "sdk_cassettes"


# =====================================================================
# Cassette discovery
# =====================================================================


def _load_cassettes(provider: str) -> List[tuple[str, dict]]:
    folder = CASSETTE_ROOT / provider
    out: List[tuple[str, dict]] = []
    for p in sorted(folder.glob("*.json")):
        with p.open() as fh:
            out.append((p.stem, json.load(fh)))
    return out


def _discover_all() -> Dict[str, List[tuple[str, dict]]]:
    return {
        provider: _load_cassettes(provider)
        for provider in ("openai", "anthropic", "bedrock", "gemini",
                         "langchain", "mcp")
    }


ALL_CASSETTES = _discover_all()


# =====================================================================
# Generic contract-asserting helpers
# =====================================================================


def _assert_openai_shape(llm_response: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    """Assert ``llm_response`` (already in OpenAI chat-completion shape)
    satisfies the cassette's ``contract.openai_shape`` block."""
    assert llm_response.get("id") == spec["id"], \
        f"id mismatch: got {llm_response.get('id')!r}, want {spec['id']!r}"
    assert llm_response.get("model") == spec["model"], \
        f"model mismatch: got {llm_response.get('model')!r}, want {spec['model']!r}"

    choices = llm_response.get("choices") or []
    assert len(choices) == 1, f"expected exactly 1 choice, got {len(choices)}"
    ch0 = choices[0]
    assert ch0.get("finish_reason") == spec["finish_reason"], \
        f"finish_reason mismatch: got {ch0.get('finish_reason')!r}, want {spec['finish_reason']!r}"

    msg = ch0.get("message") or {}
    # _strip_none scrubs explicit nulls; treat absent as None.
    got_content = msg.get("content")
    assert got_content == spec["content"], \
        f"content mismatch: got {got_content!r}, want {spec['content']!r}"

    got_tool_calls = msg.get("tool_calls")
    want_tool_calls = spec.get("tool_calls")
    assert got_tool_calls == want_tool_calls, \
        f"tool_calls mismatch: got {got_tool_calls!r}, want {want_tool_calls!r}"

    usage = llm_response.get("usage") or {}
    want_usage = spec["usage"]
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert usage.get(k) == want_usage[k], \
            f"usage.{k} mismatch: got {usage.get(k)!r}, want {want_usage[k]!r}"

    native_key = spec.get("preserves_native_under")
    if native_key:
        assert native_key in llm_response, \
            f"native payload missing: expected key {native_key!r} in llm_response"
        # Native payload must be non-empty (the provider gave us something).
        assert llm_response[native_key], \
            f"native payload {native_key!r} is empty"
    else:
        # OpenAI cassettes shouldn't gain a spurious provider sidecar.
        for forbidden in ("_anthropic", "_bedrock", "_gemini"):
            assert forbidden not in llm_response, \
                f"unexpected sidecar {forbidden!r} on OpenAI-native response"


# =====================================================================
# Duck-typed SDK-object stand-ins for the coercion-path tests
# =====================================================================


@dataclass
class _PydanticLike:
    """Mimics a pydantic model: exposes ``.model_dump()``."""
    payload: dict

    def model_dump(self) -> dict:
        return dict(self.payload)


@dataclass
class _ToDictLike:
    """Mimics a boto3-style response: exposes ``.to_dict()``."""
    payload: dict

    def to_dict(self) -> dict:
        return dict(self.payload)


# =====================================================================
# Fakes shaped exactly like real SDK clients
# =====================================================================


@dataclass
class _OAICompletionsFake:
    raw_response: dict
    calls: list = field(default_factory=list)

    def create(self, *, model: str, messages: list, **kwargs: Any) -> dict:
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        return dict(self.raw_response)


@dataclass
class _OAIChatFake:
    completions: _OAICompletionsFake


@dataclass
class _OAIClientFake:
    chat: _OAIChatFake

    @classmethod
    def from_response(cls, raw: dict) -> "_OAIClientFake":
        return cls(chat=_OAIChatFake(completions=_OAICompletionsFake(raw_response=raw)))


@dataclass
class _AnthMessagesFake:
    raw_response: dict
    calls: list = field(default_factory=list)

    def create(self, *, model: str, messages: list, **kwargs: Any) -> dict:
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        return dict(self.raw_response)


@dataclass
class _AnthClientFake:
    messages: _AnthMessagesFake

    @classmethod
    def from_response(cls, raw: dict) -> "_AnthClientFake":
        return cls(messages=_AnthMessagesFake(raw_response=raw))


@dataclass
class _BedrockClientFake:
    raw_response: dict
    calls: list = field(default_factory=list)

    def converse(self, **kwargs: Any) -> dict:
        self.calls.append(kwargs)
        return dict(self.raw_response)

    @classmethod
    def from_response(cls, raw: dict) -> "_BedrockClientFake":
        return cls(raw_response=raw)


@dataclass
class _GeminiModelsFake:
    raw_response: dict
    calls: list = field(default_factory=list)

    def generate_content(self, *, model: str, contents: Any,
                         config: Any = None, **kwargs: Any):
        self.calls.append({"model": model, "contents": contents,
                           "config": config, "kwargs": kwargs})
        return dict(self.raw_response)


@dataclass
class _GeminiClientFake:
    models: _GeminiModelsFake

    @classmethod
    def from_response(cls, raw: dict) -> "_GeminiClientFake":
        return cls(models=_GeminiModelsFake(raw_response=raw))


@dataclass
class _LangchainToolFake:
    name: str
    description: str
    return_value: Any

    def invoke(self, args: Any) -> Any:
        return self.return_value

    def run(self, args: Any) -> Any:
        return self.return_value


@dataclass
class _MCPSessionFake:
    return_value: Any
    calls: list = field(default_factory=list)

    def call_tool(self, name: str, arguments: Optional[dict] = None) -> Any:
        self.calls.append({"name": name, "arguments": dict(arguments or {})})
        return self.return_value


# =====================================================================
# Per-provider drivers: (record-and-assert) + (pure-coerce) + (duck-typed)
# =====================================================================


def _last_step(rec_inst) -> dict:
    steps = rec_inst._writer.frames if hasattr(rec_inst, "_writer") else None
    if steps:
        return steps[-1]
    raise AssertionError("recorder exposed no frames")


def _replay_must_cache(trace_path: str) -> None:
    t = replay(trace_path)
    exec_ = Executor()
    result = t.run_replay(subs=SubstitutionSet(), executor=exec_)
    assert all(not s.dirty for s in result), "replay produced dirty steps"
    assert exec_.real_calls == 0, \
        f"replay should serve from cache, executor saw {exec_.real_calls} calls"


# --- OpenAI ----------------------------------------------------------


@pytest.mark.parametrize("name,cassette",
                         ALL_CASSETTES["openai"],
                         ids=[n for n, _ in ALL_CASSETTES["openai"]])
def test_openai_cassette_recording_contract(tmp_path, name, cassette):
    spec = cassette["contract"]["openai_shape"]
    req = cassette["request"]
    raw = cassette["raw_response"]

    trace_path = str(tmp_path / f"{name}.sb")
    fake = _OAIClientFake.from_response(raw)
    with record(trace_path) as rec:
        client = wrap_openai(fake, rec, default_model=req["model"])
        resp = client.chat.completions.create(
            messages=req["messages"], **req.get("kwargs", {}),
        )
        # The wrapper returns OpenAIChatCompletion
        assert resp.id == spec["id"]
        assert resp.model == spec["model"]

    # Inspect the recorded frame via replay
    t = replay(trace_path)
    assert len(t.recorded_steps) == 1
    llm_resp = t.recorded_steps[0]["llm_response"]
    _assert_openai_shape(llm_resp, spec)
    _replay_must_cache(trace_path)


@pytest.mark.parametrize("name,cassette",
                         ALL_CASSETTES["openai"],
                         ids=[n for n, _ in ALL_CASSETTES["openai"]])
def test_openai_cassette_pure_coercion(name, cassette):
    spec = cassette["contract"]["openai_shape"]
    raw = cassette["raw_response"]
    canonical = _canonicalise_openai_response(raw)
    _assert_openai_shape(canonical, spec)


@pytest.mark.parametrize("name,cassette",
                         ALL_CASSETTES["openai"],
                         ids=[n for n, _ in ALL_CASSETTES["openai"]])
def test_openai_cassette_pydantic_object_path(name, cassette):
    spec = cassette["contract"]["openai_shape"]
    raw = cassette["raw_response"]
    canonical = _canonicalise_openai_response(_PydanticLike(payload=raw))
    _assert_openai_shape(canonical, spec)


# --- Anthropic -------------------------------------------------------


@pytest.mark.parametrize("name,cassette",
                         ALL_CASSETTES["anthropic"],
                         ids=[n for n, _ in ALL_CASSETTES["anthropic"]])
def test_anthropic_cassette_recording_contract(tmp_path, name, cassette):
    spec = cassette["contract"]["openai_shape"]
    req = cassette["request"]
    raw = cassette["raw_response"]

    trace_path = str(tmp_path / f"{name}.sb")
    fake = _AnthClientFake.from_response(raw)
    with record(trace_path) as rec:
        client = wrap_anthropic(fake, rec)
        client.messages.create(
            model=req["model"], messages=req["messages"], **req.get("kwargs", {}),
        )

    t = replay(trace_path)
    assert len(t.recorded_steps) == 1
    llm_resp = t.recorded_steps[0]["llm_response"]
    _assert_openai_shape(llm_resp, spec)
    _replay_must_cache(trace_path)


@pytest.mark.parametrize("name,cassette",
                         ALL_CASSETTES["anthropic"],
                         ids=[n for n, _ in ALL_CASSETTES["anthropic"]])
def test_anthropic_cassette_pure_coercion(name, cassette):
    spec = cassette["contract"]["openai_shape"]
    raw = cassette["raw_response"]
    native = _coerce_anthropic_response(raw)
    canonical = _anthropic_to_openai_shape(native)
    _assert_openai_shape(canonical, spec)


@pytest.mark.parametrize("name,cassette",
                         ALL_CASSETTES["anthropic"],
                         ids=[n for n, _ in ALL_CASSETTES["anthropic"]])
def test_anthropic_cassette_pydantic_object_path(name, cassette):
    spec = cassette["contract"]["openai_shape"]
    raw = cassette["raw_response"]
    native = _coerce_anthropic_response(_PydanticLike(payload=raw))
    canonical = _anthropic_to_openai_shape(native)
    _assert_openai_shape(canonical, spec)


# --- Bedrock ---------------------------------------------------------


@pytest.mark.parametrize("name,cassette",
                         ALL_CASSETTES["bedrock"],
                         ids=[n for n, _ in ALL_CASSETTES["bedrock"]])
def test_bedrock_cassette_recording_contract(tmp_path, name, cassette):
    spec = cassette["contract"]["openai_shape"]
    req = cassette["request"]
    raw = cassette["raw_response"]

    trace_path = str(tmp_path / f"{name}.sb")
    fake = _BedrockClientFake.from_response(raw)
    with record(trace_path) as rec:
        client = wrap_bedrock(fake, rec)
        client.converse(modelId=req["model"],
                        messages=[{"role": m["role"],
                                    "content": [{"text": m["content"]}]}
                                   for m in req["messages"]])

    t = replay(trace_path)
    assert len(t.recorded_steps) == 1
    llm_resp = t.recorded_steps[0]["llm_response"]
    _assert_openai_shape(llm_resp, spec)
    _replay_must_cache(trace_path)


@pytest.mark.parametrize("name,cassette",
                         ALL_CASSETTES["bedrock"],
                         ids=[n for n, _ in ALL_CASSETTES["bedrock"]])
def test_bedrock_cassette_pure_coercion(name, cassette):
    spec = cassette["contract"]["openai_shape"]
    raw = cassette["raw_response"]
    native = _coerce_bedrock_response(raw)
    native.setdefault("_modelId", cassette["request"]["model"])
    canonical = _bedrock_to_openai_shape(native)
    _assert_openai_shape(canonical, spec)


@pytest.mark.parametrize("name,cassette",
                         ALL_CASSETTES["bedrock"],
                         ids=[n for n, _ in ALL_CASSETTES["bedrock"]])
def test_bedrock_cassette_to_dict_object_path(name, cassette):
    spec = cassette["contract"]["openai_shape"]
    raw = cassette["raw_response"]
    native = _coerce_bedrock_response(_ToDictLike(payload=raw))
    native.setdefault("_modelId", cassette["request"]["model"])
    canonical = _bedrock_to_openai_shape(native)
    _assert_openai_shape(canonical, spec)


# --- Gemini ----------------------------------------------------------


@pytest.mark.parametrize("name,cassette",
                         ALL_CASSETTES["gemini"],
                         ids=[n for n, _ in ALL_CASSETTES["gemini"]])
def test_gemini_cassette_recording_contract(tmp_path, name, cassette):
    spec = cassette["contract"]["openai_shape"]
    req = cassette["request"]
    raw = cassette["raw_response"]

    trace_path = str(tmp_path / f"{name}.sb")
    fake = _GeminiClientFake.from_response(raw)
    with record(trace_path) as rec:
        client = wrap_gemini(fake, rec, default_model=req["model"])
        # Pass `contents` as a plain string user prompt (the simplest
        # surface accepted by the real SDK).
        user_text = "".join(m["content"] for m in req["messages"]
                            if m["role"] == "user")
        client.models.generate_content(contents=user_text)

    t = replay(trace_path)
    assert len(t.recorded_steps) == 1
    llm_resp = t.recorded_steps[0]["llm_response"]
    _assert_openai_shape(llm_resp, spec)
    _replay_must_cache(trace_path)


@pytest.mark.parametrize("name,cassette",
                         ALL_CASSETTES["gemini"],
                         ids=[n for n, _ in ALL_CASSETTES["gemini"]])
def test_gemini_cassette_pure_coercion(name, cassette):
    spec = cassette["contract"]["openai_shape"]
    raw = cassette["raw_response"]
    native = _coerce_gemini_response(raw)
    canonical = _gemini_to_openai_shape(native)
    _assert_openai_shape(canonical, spec)


@pytest.mark.parametrize("name,cassette",
                         ALL_CASSETTES["gemini"],
                         ids=[n for n, _ in ALL_CASSETTES["gemini"]])
def test_gemini_cassette_pydantic_object_path(name, cassette):
    spec = cassette["contract"]["openai_shape"]
    raw = cassette["raw_response"]
    native = _coerce_gemini_response(_PydanticLike(payload=raw))
    canonical = _gemini_to_openai_shape(native)
    _assert_openai_shape(canonical, spec)


# --- LangChain -------------------------------------------------------


@pytest.mark.parametrize("name,cassette",
                         ALL_CASSETTES["langchain"],
                         ids=[n for n, _ in ALL_CASSETTES["langchain"]])
def test_langchain_cassette_recording_contract(tmp_path, name, cassette):
    spec = cassette["contract"]["tool_outputs"]
    req = cassette["request"]
    raw = cassette["raw_response"]

    trace_path = str(tmp_path / f"{name}.sb")
    fake = _LangchainToolFake(name=req["tool_name"],
                              description=req["description"],
                              return_value=raw)
    with record(trace_path) as rec:
        wrapped = wrap_langchain_tool(fake, rec)
        if req.get("use_run"):
            result = wrapped.run(req["arguments"])
        else:
            result = wrapped.invoke(req["arguments"])
        assert result == raw

    t = replay(trace_path)
    assert len(t.recorded_steps) == 1
    step = t.recorded_steps[0]
    assert step["step_kind"] == "tool_call"
    assert step["inputs"]["name"] == spec["name_recorded_as"]
    assert step["inputs"]["arguments"] == spec["arguments_recorded_as"]
    assert step["outputs"]["result"] == spec["result"]
    _replay_must_cache(trace_path)


# --- MCP -------------------------------------------------------------


@pytest.mark.parametrize("name,cassette",
                         ALL_CASSETTES["mcp"],
                         ids=[n for n, _ in ALL_CASSETTES["mcp"]])
def test_mcp_cassette_recording_contract(tmp_path, name, cassette):
    spec = cassette["contract"]["tool_outputs"]
    req = cassette["request"]
    raw = cassette["raw_response"]

    trace_path = str(tmp_path / f"{name}.sb")
    fake = _MCPSessionFake(return_value=raw)
    with record(trace_path) as rec:
        sess = wrap_mcp_session(fake, rec, server_name=req["server_name"])
        result = sess.call_tool(req["tool_name"], req["arguments"])
        assert result == raw

    t = replay(trace_path)
    assert len(t.recorded_steps) == 1
    step = t.recorded_steps[0]
    assert step["step_kind"] == "tool_call"
    assert step["inputs"]["name"] == spec["name_recorded_as"]
    assert step["inputs"]["arguments"] == spec["arguments_recorded_as"]
    assert step["outputs"]["result"] == spec["result"]
    _replay_must_cache(trace_path)


# =====================================================================
# Coverage / discovery sanity check
# =====================================================================


def test_every_provider_has_cassettes():
    """Each provider shim must ship at least one contract cassette so a
    drop-in agent author can see the SDK shape we lock in."""
    minimums = {
        "openai": 4, "anthropic": 4, "bedrock": 4, "gemini": 4,
        "langchain": 3, "mcp": 3,
    }
    for provider, minimum in minimums.items():
        cassettes = ALL_CASSETTES[provider]
        assert len(cassettes) >= minimum, (
            f"{provider}: only {len(cassettes)} cassettes, expected >= {minimum}"
        )
        for name, c in cassettes:
            assert c["provider"] == provider, \
                f"{provider}/{name}: provider field mismatch"
            assert "raw_response" in c, f"{provider}/{name}: missing raw_response"
            assert "contract" in c, f"{provider}/{name}: missing contract"
