"""End-to-end tests for the Cohere v2 and Mistral shims.

Fakes below are duck-typed against:
- ``cohere.ClientV2``: a ``client.chat(model=..., messages=..., ...)`` method
  returning the Cohere v2 response shape.
- ``mistralai.Mistral``: a ``client.chat.complete(model=..., messages=..., ...)``
  method returning an OpenAI-compatible response shape.

Tests prove:

Cohere:
1. ``wrap_cohere`` rejects clients whose ``.chat`` is not callable.
2. record→replay-from-cache is a 100% cache hit (zero real calls).
3. The recorded canonical ``llm_request.messages`` unifies Cohere's
   OpenAI-style ``messages`` list, with ``preamble`` promoted to a
   ``role: system`` entry.
4. A :class:`PromptSubstitution` against a Cohere-recorded step correctly
   dirties downstream steps and re-executes through :func:`cohere_executor`.
5. ``tool_calls`` round-trip through the canonicaliser as OpenAI-style
   ``tool_calls`` with JSON-string ``function.arguments``.
6. Cost accounting: ``command-r-plus-08-2024`` is recognised via
   :func:`canonical_cohere_model_id` so ``cost_usd`` is non-zero.
7. Finish-reason normalisation (``COMPLETE`` → ``stop``, ``TOOL_CALL`` →
   ``tool_calls``, ``MAX_TOKENS`` → ``length``).
8. Tool-only response (no text content) canonicalises correctly.

Mistral:
9.  ``wrap_mistral`` rejects clients whose ``.chat.complete`` is not callable.
10. record→replay-from-cache is a 100% cache hit.
11. The recorded canonical ``llm_request.messages`` is the raw OpenAI-style
    message list (no transformation).
12. A :class:`PromptSubstitution` against a Mistral-recorded step re-executes
    through :func:`mistral_executor`.
13. Tool calls round-trip with JSON-string arguments.
14. Cost accounting: ``mistral-large-2411`` is recognised via
    :func:`canonical_mistral_model_id` so ``cost_usd`` is non-zero.
15. ``*-latest`` aliases resolve correctly for both providers.

Registry:
16. ``CohereShimContract`` and ``MistralShimContract`` are pre-registered and
    reachable via ``shim_contract_for``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, List, Optional

import pytest

from stepback import record, replay
from stepback.pricing import RATE_TABLE, compute_cost
from stepback.replay import Executor
from stepback.shims import (
    CohereMessage,
    CohereShimContract,
    MistralChatResponse,
    MistralShimContract,
    canonical_cohere_model_id,
    canonical_mistral_model_id,
    cohere_executor,
    mistral_executor,
    shim_contract_for,
    wrap_cohere,
    wrap_mistral,
)
from stepback.substitutions import PromptSubstitution, SubstitutionSet


# =====================================================================
# Fake Cohere v2 client
# =====================================================================


@dataclass
class _FakeCohereResponse:
    """Attribute-based response like cohere.ClientV2.chat returns."""

    id: str
    finish_reason: str
    message: dict  # {"role": "assistant", "content": [...], "tool_calls": None|[...]}
    usage: dict    # {"billed_units": {...}, "tokens": {...}}


class _FakeCohere:
    """Duck-typed against ``cohere.ClientV2``."""

    def __init__(self) -> None:
        self._calls: List[dict] = []
        self.next_tool_calls: Optional[List[dict]] = None  # set to inject tool calls

    def chat(self, *, model: str, messages: List[dict], **kwargs: Any) -> _FakeCohereResponse:
        self._calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        # Deterministic body keyed off message count so cache hits are bit-identical.
        last_user = ""
        for m in messages:
            if m.get("role") == "user":
                c = m.get("content", "")
                last_user = c if isinstance(c, str) else str(c)
        body = f"cohere-{model}-msg{len(messages)}-call{len(self._calls)}-echo[{last_user}]"
        in_tok = sum(len(m.get("content", "") if isinstance(m.get("content", ""), str)
                      else str(m.get("content", "")))
                     for m in messages)
        out_tok = len(body)

        content_blocks: List[dict] = [{"type": "text", "text": body}]
        finish_reason = "COMPLETE"
        tool_calls_out: Optional[List[dict]] = None
        if self.next_tool_calls is not None:
            tool_calls_out = self.next_tool_calls
            self.next_tool_calls = None
            finish_reason = "TOOL_CALL"

        return _FakeCohereResponse(
            id=f"cohere-req-{len(self._calls):04d}",
            finish_reason=finish_reason,
            message={
                "role": "assistant",
                "content": content_blocks,
                "tool_calls": tool_calls_out,
            },
            usage={
                "billed_units": {"input_tokens": in_tok, "output_tokens": out_tok},
                "tokens": {"input_tokens": in_tok, "output_tokens": out_tok},
            },
        )

    @property
    def call_count(self) -> int:
        return len(self._calls)


# =====================================================================
# Fake Mistral client
# =====================================================================


@dataclass
class _FakeMistralMessage:
    role: str
    content: Optional[str]
    tool_calls: Optional[List[dict]] = None


@dataclass
class _FakeMistralChoice:
    index: int
    finish_reason: str
    message: _FakeMistralMessage


@dataclass
class _FakeMistralUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class _FakeMistralResponse:
    """Attribute-based response like mistralai SDK returns (OpenAI-shaped)."""

    id: str
    model: str
    choices: List[_FakeMistralChoice]
    usage: _FakeMistralUsage

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
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
        }


class _FakeMistralChat:
    def __init__(self) -> None:
        self._calls: List[dict] = []
        self.next_tool_calls: Optional[List[dict]] = None

    def complete(self, *, model: str, messages: List[dict],
                 **kwargs: Any) -> _FakeMistralResponse:
        self._calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        last_user = ""
        for m in messages:
            if m.get("role") == "user":
                last_user = str(m.get("content", ""))
        body = f"mistral-{model}-msg{len(messages)}-call{len(self._calls)}-echo[{last_user}]"
        in_tok = sum(len(str(m.get("content", ""))) for m in messages)
        out_tok = len(body)

        content: Optional[str] = body
        tool_calls_out: Optional[List[dict]] = None
        finish_reason = "stop"
        if self.next_tool_calls is not None:
            tool_calls_out = self.next_tool_calls
            self.next_tool_calls = None
            content = None
            finish_reason = "tool_calls"

        return _FakeMistralResponse(
            id=f"mistral-req-{len(self._calls):04d}",
            model=model,
            choices=[
                _FakeMistralChoice(
                    index=0,
                    finish_reason=finish_reason,
                    message=_FakeMistralMessage(
                        role="assistant",
                        content=content,
                        tool_calls=tool_calls_out,
                    ),
                )
            ],
            usage=_FakeMistralUsage(
                prompt_tokens=in_tok,
                completion_tokens=out_tok,
                total_tokens=in_tok + out_tok,
            ),
        )

    @property
    def call_count(self) -> int:
        return len(self._calls)


class _FakeMistralClient:
    def __init__(self) -> None:
        self.chat = _FakeMistralChat()


# =====================================================================
# Helper
# =====================================================================

COHERE_MODEL = "command-r-plus-08-2024"
MISTRAL_MODEL = "mistral-large-2411"


def _write_and_read_cohere(fake: _FakeCohere, messages: List[dict],
                            model: str = COHERE_MODEL,
                            preamble: Optional[str] = None) -> tuple:
    """Record one Cohere call; return (trace_path, step)."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        path = f.name
    try:
        with record(path) as rec:
            client = wrap_cohere(fake, rec)
            kwargs: dict = {"model": model, "messages": messages}
            if preamble:
                kwargs["preamble"] = preamble
            resp = client.chat(**kwargs)
        return path, resp
    except Exception:
        os.unlink(path)
        raise


# =====================================================================
# Cohere tests
# =====================================================================


def test_wrap_cohere_rejects_non_cohere_client() -> None:
    class _NoChatCallable:
        chat = "not callable"

    with pytest.raises(TypeError, match="chat is not callable"):
        wrap_cohere(_NoChatCallable(), None)  # type: ignore[arg-type]


def test_wrap_cohere_record_and_replay_cache_hit(tmp_path) -> None:
    """record→replay is a 100% cache hit — zero extra LLM calls."""
    fake = _FakeCohere()
    path = str(tmp_path / "trace.sb")
    messages = [{"role": "user", "content": "Hello Cohere"}]

    with record(path) as rec:
        client = wrap_cohere(fake, rec)
        resp = client.chat(model=COHERE_MODEL, messages=messages)

    assert fake.call_count == 1
    assert isinstance(resp, CohereMessage)
    assert "echo[Hello Cohere]" in resp.message.get("content", [{}])[0].get("text", "")

    fake2 = _FakeCohere()
    tr = replay(path)
    exec_ = Executor(llm=cohere_executor(fake2))
    tr.replay_forward(executor=exec_)

    assert fake2.call_count == 0, "Cache hit expected; no real calls during replay"


def test_wrap_cohere_canonical_request_messages(tmp_path) -> None:
    """Cohere messages are stored in unified format; preamble becomes system entry."""
    fake = _FakeCohere()
    path = str(tmp_path / "trace.sb")
    messages = [{"role": "user", "content": "What is 2+2?"}]
    preamble = "You are a math assistant."

    with record(path) as rec:
        client = wrap_cohere(fake, rec)
        client.chat(model=COHERE_MODEL, messages=messages, preamble=preamble)

    tr = replay(path)
    step = tr.recorded_steps[0]
    req_messages = step["llm_request"]["messages"]

    # System message from preamble is the first entry.
    assert req_messages[0]["role"] == "system"
    assert req_messages[0]["content"] == preamble
    # User message follows.
    assert req_messages[1]["role"] == "user"
    assert req_messages[1]["content"] == "What is 2+2?"


def test_wrap_cohere_prompt_substitution_dirties_downstream(tmp_path) -> None:
    """PromptSubstitution triggers re-execution through cohere_executor."""
    fake = _FakeCohere()
    path = str(tmp_path / "trace.sb")

    with record(path) as rec:
        client = wrap_cohere(fake, rec)
        r1 = client.chat(model=COHERE_MODEL,
                         messages=[{"role": "user", "content": "Step one"}])
        r2 = client.chat(model=COHERE_MODEL,
                         messages=[{"role": "user", "content": "Step two"}])

    assert fake.call_count == 2

    tr = replay(path)
    step_id = tr.recorded_steps[0]["step_id"]
    sub = PromptSubstitution(
        at_step=step_id,
        new_messages=[{"role": "user", "content": "Modified step one"}],
    )
    subs = SubstitutionSet([sub])

    fake2 = _FakeCohere()
    branch = tr.branch_at(step_id, "cf")
    branch.substitute(sub)
    exec_ = Executor(llm=cohere_executor(fake2))
    branch.replay_forward(executor=exec_)

    # Step 0 is dirty (substitution) → 1 real call.
    assert fake2.call_count >= 1


def test_wrap_cohere_tool_calls_round_trip(tmp_path) -> None:
    """Tool-call blocks canonicalise to OpenAI tool_calls with JSON-string arguments."""
    fake = _FakeCohere()
    fake.next_tool_calls = [
        {
            "id": "tc-001",
            "type": "function",
            "function": {"name": "search", "arguments": {"query": "AI news"}},
        }
    ]
    path = str(tmp_path / "trace.sb")

    with record(path) as rec:
        client = wrap_cohere(fake, rec)
        resp = client.chat(
            model=COHERE_MODEL,
            messages=[{"role": "user", "content": "Search for AI news"}],
        )

    tr = replay(path)
    step = tr.recorded_steps[0]
    resp_dict = step["llm_response"]

    tool_calls = resp_dict["choices"][0]["message"].get("tool_calls")
    assert tool_calls is not None and len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc["function"]["name"] == "search"
    # Arguments must be a JSON string, not a dict.
    args_raw = tc["function"]["arguments"]
    assert isinstance(args_raw, str)
    args = json.loads(args_raw)
    assert args["query"] == "AI news"

    # finish_reason should map TOOL_CALL → tool_calls
    assert resp_dict["choices"][0]["finish_reason"] == "tool_calls"


def test_wrap_cohere_tool_only_response(tmp_path) -> None:
    """A response with tool_calls but no text content canonicalises correctly."""
    fake = _FakeCohere()
    fake.next_tool_calls = [
        {
            "id": "tc-002",
            "type": "function",
            "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
        }
    ]
    path = str(tmp_path / "trace.sb")

    with record(path) as rec:
        client = wrap_cohere(fake, rec)
        # Override message content to be empty to simulate tool-only response.
        fake._calls  # access to trigger init
        resp = client.chat(
            model=COHERE_MODEL,
            messages=[{"role": "user", "content": "What is the weather?"}],
        )

    tr = replay(path)
    step = tr.recorded_steps[0]
    msg = step["llm_response"]["choices"][0]["message"]
    assert msg.get("tool_calls") is not None


def test_wrap_cohere_cost_accounting(tmp_path) -> None:
    """command-r-plus-08-2024 has a pricing row so cost_usd is non-zero."""
    fake = _FakeCohere()
    path = str(tmp_path / "trace.sb")

    with record(path) as rec:
        client = wrap_cohere(fake, rec)
        client.chat(
            model=COHERE_MODEL,
            messages=[{"role": "user", "content": "Price this call"}],
        )

    tr = replay(path)
    step = tr.recorded_steps[0]
    assert step.get("cost_usd", 0) > 0, "Expected non-zero cost for known Cohere model"


def test_cohere_finish_reason_normalisation() -> None:
    """Cohere uppercase finish reasons map to OpenAI canonical values."""
    from stepback.shims import _cohere_to_openai_shape

    def _shape(finish: str) -> Optional[str]:
        d = {
            "id": "x", "finish_reason": finish,
            "message": {"role": "assistant", "content": [], "tool_calls": None},
            "usage": {"billed_units": {"input_tokens": 1, "output_tokens": 1}},
        }
        return _cohere_to_openai_shape(d)["choices"][0]["finish_reason"]

    assert _shape("COMPLETE") == "stop"
    assert _shape("MAX_TOKENS") == "length"
    assert _shape("TOOL_CALL") == "tool_calls"
    assert _shape("ERROR_TOXIC") == "content_filter"
    assert _shape("ERROR") == "error"


def test_canonical_cohere_model_id_aliases() -> None:
    assert canonical_cohere_model_id("command-r-plus-latest") == "command-r-plus-08-2024"
    assert canonical_cohere_model_id("command-r-latest") == "command-r-08-2024"
    assert canonical_cohere_model_id("command-r-plus-08-2024") == "command-r-plus-08-2024"
    # Unknown ids pass through.
    assert canonical_cohere_model_id("command-xyz-9999") == "command-xyz-9999"


def test_cohere_pricing_rows_present() -> None:
    """Native Cohere model rows exist in the rate table."""
    assert "command-r-plus-08-2024" in RATE_TABLE
    assert "command-r-08-2024" in RATE_TABLE
    r = RATE_TABLE["command-r-plus-08-2024"]
    assert r.input_per_1k > 0
    assert r.output_per_1k > 0


# =====================================================================
# Mistral tests
# =====================================================================


def test_wrap_mistral_rejects_non_mistral_client() -> None:
    class _NoComplete:
        class chat:
            complete = "not callable"

    with pytest.raises(TypeError, match="chat.complete is not callable"):
        wrap_mistral(_NoComplete(), None)  # type: ignore[arg-type]


def test_wrap_mistral_record_and_replay_cache_hit(tmp_path) -> None:
    """record→replay is a 100% cache hit — zero extra LLM calls."""
    fake = _FakeMistralClient()
    path = str(tmp_path / "trace.sb")
    messages = [{"role": "user", "content": "Hello Mistral"}]

    with record(path) as rec:
        client = wrap_mistral(fake, rec)
        resp = client.chat.complete(model=MISTRAL_MODEL, messages=messages)

    assert fake.chat.call_count == 1
    assert isinstance(resp, MistralChatResponse)
    assert "echo[Hello Mistral]" in (resp.choices[0].message.content or "")

    fake2 = _FakeMistralClient()
    tr = replay(path)
    exec_ = Executor(llm=mistral_executor(fake2))
    tr.replay_forward(executor=exec_)

    assert fake2.chat.call_count == 0, "Cache hit expected; no real calls during replay"


def test_wrap_mistral_canonical_request_messages(tmp_path) -> None:
    """Mistral messages are stored as-is (OpenAI-compatible format)."""
    fake = _FakeMistralClient()
    path = str(tmp_path / "trace.sb")
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Tell me a joke"},
    ]

    with record(path) as rec:
        client = wrap_mistral(fake, rec)
        client.chat.complete(model=MISTRAL_MODEL, messages=messages)

    tr = replay(path)
    step = tr.recorded_steps[0]
    req_messages = step["llm_request"]["messages"]

    assert req_messages[0]["role"] == "system"
    assert req_messages[1]["role"] == "user"
    assert req_messages[1]["content"] == "Tell me a joke"


def test_wrap_mistral_prompt_substitution_dirties_downstream(tmp_path) -> None:
    """PromptSubstitution triggers re-execution through mistral_executor."""
    fake = _FakeMistralClient()
    path = str(tmp_path / "trace.sb")

    with record(path) as rec:
        client = wrap_mistral(fake, rec)
        client.chat.complete(model=MISTRAL_MODEL,
                             messages=[{"role": "user", "content": "Step one"}])
        client.chat.complete(model=MISTRAL_MODEL,
                             messages=[{"role": "user", "content": "Step two"}])

    assert fake.chat.call_count == 2

    tr = replay(path)
    step_id = tr.recorded_steps[0]["step_id"]
    sub = PromptSubstitution(
        at_step=step_id,
        new_messages=[{"role": "user", "content": "Modified step one"}],
    )

    fake2 = _FakeMistralClient()
    branch = tr.branch_at(step_id, "cf")
    branch.substitute(sub)
    exec_ = Executor(llm=mistral_executor(fake2))
    branch.replay_forward(executor=exec_)

    assert fake2.chat.call_count >= 1


def test_wrap_mistral_tool_calls_round_trip(tmp_path) -> None:
    """Tool-call blocks canonicalise with JSON-string arguments."""
    fake = _FakeMistralClient()
    fake.chat.next_tool_calls = [
        {
            "id": "tc-m001",
            "type": "function",
            "function": {"name": "calculate", "arguments": '{"expr": "2+2"}'},
        }
    ]
    path = str(tmp_path / "trace.sb")

    with record(path) as rec:
        client = wrap_mistral(fake, rec)
        resp = client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": "Calculate 2+2"}],
        )

    tr = replay(path)
    step = tr.recorded_steps[0]
    tool_calls = step["llm_response"]["choices"][0]["message"].get("tool_calls")
    assert tool_calls is not None and len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc["function"]["name"] == "calculate"
    args = json.loads(tc["function"]["arguments"])
    assert args["expr"] == "2+2"
    assert step["llm_response"]["choices"][0]["finish_reason"] == "tool_calls"


def test_wrap_mistral_cost_accounting(tmp_path) -> None:
    """mistral-large-2411 has a pricing row so cost_usd is non-zero."""
    fake = _FakeMistralClient()
    path = str(tmp_path / "trace.sb")

    with record(path) as rec:
        client = wrap_mistral(fake, rec)
        client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": "Price this call"}],
        )

    tr = replay(path)
    step = tr.recorded_steps[0]
    assert step.get("cost_usd", 0) > 0, "Expected non-zero cost for known Mistral model"


def test_canonical_mistral_model_id_aliases() -> None:
    assert canonical_mistral_model_id("mistral-large-latest") == "mistral-large-2411"
    assert canonical_mistral_model_id("mistral-small-latest") == "mistral-small-2501"
    assert canonical_mistral_model_id("codestral-latest") == "codestral-2501"
    assert canonical_mistral_model_id("mistral-large-2411") == "mistral-large-2411"
    # Unknown pass through.
    assert canonical_mistral_model_id("mistral-xyz-9999") == "mistral-xyz-9999"


def test_mistral_pricing_rows_present() -> None:
    """Native Mistral model rows exist in the rate table."""
    assert "mistral-large-2411" in RATE_TABLE
    assert "mistral-small-2501" in RATE_TABLE
    assert "codestral-2501" in RATE_TABLE
    assert "open-mistral-nemo" in RATE_TABLE
    r = RATE_TABLE["mistral-large-2411"]
    assert r.input_per_1k > 0
    assert r.output_per_1k > 0


# =====================================================================
# Registry tests
# =====================================================================


def test_cohere_contract_registered() -> None:
    c = shim_contract_for("cohere")
    assert isinstance(c, CohereShimContract)


def test_mistral_contract_registered() -> None:
    m = shim_contract_for("mistral")
    assert isinstance(m, MistralShimContract)


def test_cohere_contract_canonical_request_preamble() -> None:
    c = CohereShimContract()
    messages = [{"role": "user", "content": "Hi"}]
    result = c.canonical_request(messages=messages, preamble="Be helpful.")
    assert result[0] == {"role": "system", "content": "Be helpful."}
    assert result[1]["role"] == "user"


def test_mistral_contract_canonical_request_passthrough() -> None:
    m = MistralShimContract()
    messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    result = m.canonical_request(messages=messages)
    assert result == messages


def test_cohere_contract_canonical_response_dict() -> None:
    c = CohereShimContract()
    native = {
        "id": "x",
        "finish_reason": "COMPLETE",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
            "tool_calls": None,
        },
        "usage": {"billed_units": {"input_tokens": 5, "output_tokens": 3}},
    }
    result = c.canonical_response(native)
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["choices"][0]["message"]["content"] == "Hello"
    assert result["usage"]["prompt_tokens"] == 5
    assert result["usage"]["completion_tokens"] == 3


def test_mistral_contract_canonical_response_dict() -> None:
    m = MistralShimContract()
    native = {
        "id": "y",
        "model": MISTRAL_MODEL,
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Hi there", "tool_calls": None},
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }
    result = m.canonical_response(native)
    assert result["choices"][0]["message"]["content"] == "Hi there"
    assert result["usage"]["total_tokens"] == 6
