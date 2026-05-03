"""End-to-end tests for the LLM-client / tool-registry shims.

We exercise the *real* record → replay pipeline end-to-end against
duck-typed fakes shaped exactly like the OpenAI, Anthropic, LangChain,
and MCP SDKs. The point is to prove agent code calling the wrapped
clients produces a valid `.sb` trace, that the trace replays from
cache 100%, and that a typed substitution still propagates dirtiness
through steps recorded via the shims.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional

import pytest

from stepback import record, replay
from stepback.replay import Executor
from stepback.shims import (
    AnthropicMessage,
    OpenAIChatCompletion,
    anthropic_executor,
    langchain_tool_executor,
    mcp_tool_executor,
    openai_executor,
    wrap_anthropic,
    wrap_langchain_tool,
    wrap_langchain_tools,
    wrap_mcp_session,
    wrap_openai,
)
from stepback.substitutions import (
    PromptSubstitution,
    SubstitutionSet,
    ToolOutputSubstitution,
)


# =====================================================================
# Fakes — duck-typed against the real SDK shapes
# =====================================================================


@dataclass
class _OAIChatCompletionsFake:
    calls: List[dict]

    def create(self, *, model: str, messages: List[dict], **kwargs: Any) -> dict:
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        text = f"oai-{model}-msg{len(messages)}-call{len(self.calls)}"
        return {
            "id": f"chatcmpl-{len(self.calls):04d}",
            "model": model,
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }],
            "usage": {
                "prompt_tokens": sum(len(m["content"]) for m in messages),
                "completion_tokens": len(text),
                "total_tokens": sum(len(m["content"]) for m in messages) + len(text),
            },
        }


@dataclass
class _OAIChatFake:
    completions: _OAIChatCompletionsFake


@dataclass
class FakeOpenAI:
    chat: _OAIChatFake
    embeddings: Any = None  # passthrough surface

    @classmethod
    def make(cls) -> "FakeOpenAI":
        return cls(chat=_OAIChatFake(completions=_OAIChatCompletionsFake(calls=[])))


@dataclass
class _AnthMessagesFake:
    calls: List[dict]

    def create(self, *, model: str, messages: List[dict],
               system: Optional[str] = None, max_tokens: int = 1024,
               **kwargs: Any) -> dict:
        self.calls.append({
            "model": model, "messages": messages, "system": system,
            "max_tokens": max_tokens, "kwargs": kwargs,
        })
        body = f"anth-{model}-msg{len(messages)}-call{len(self.calls)}"
        return {
            "id": f"msg_{len(self.calls):04d}",
            "model": model,
            "role": "assistant",
            "content": [{"type": "text", "text": body}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": sum(len(m["content"]) for m in messages),
                "output_tokens": len(body),
            },
        }


@dataclass
class FakeAnthropic:
    messages: _AnthMessagesFake

    @classmethod
    def make(cls) -> "FakeAnthropic":
        return cls(messages=_AnthMessagesFake(calls=[]))


@dataclass
class FakeLangchainTool:
    name: str
    description: str
    handler: Any  # callable(args_dict) -> result

    def invoke(self, args: Any) -> Any:
        return self.handler(args if isinstance(args, dict) else {"input": args})


@dataclass
class FakeMCPSession:
    server: str
    handlers: dict

    def call_tool(self, name: str, arguments: Optional[dict] = None) -> Any:
        if name not in self.handlers:
            raise KeyError(name)
        return self.handlers[name](arguments or {})

    # Surface carried by passthrough
    def list_tools(self) -> List[str]:
        return sorted(self.handlers.keys())


# =====================================================================
# OpenAI shim
# =====================================================================


def test_wrap_openai_records_and_replays_from_cache(tmp_path):
    trace_path = str(tmp_path / "openai.sb")
    with record(trace_path) as rec:
        client = wrap_openai(FakeOpenAI.make(), rec,
                             default_model="gpt-4o-mini-2024-07-18")
        resp1 = client.chat.completions.create(
            messages=[{"role": "user", "content": "hello"}],
        )
        assert isinstance(resp1, OpenAIChatCompletion)
        assert resp1.choices[0].message.content.startswith("oai-")
        assert resp1.usage["prompt_tokens"] > 0

        resp2 = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": "again"}],
            temperature=0.2,
        )
        assert resp2.model == "gpt-4o-2024-11-20"

    t = replay(trace_path)
    assert len(t.recorded_steps) == 2
    assert all(s["step_kind"] == "llm_call" for s in t.recorded_steps)
    # Cost is real — pricing.PRICE_LIST has both models.
    total_cost = sum(s.get("cost_usd", 0) for s in t.recorded_steps)
    assert total_cost > 0

    exec_ = Executor()
    result = t.run_replay(subs=SubstitutionSet(), executor=exec_)
    assert all(not s.dirty for s in result)
    assert exec_.real_calls == 0


def test_wrap_openai_requires_default_model_or_explicit_model(tmp_path):
    with record(str(tmp_path / "x.sb")) as rec:
        client = wrap_openai(FakeOpenAI.make(), rec)
        with pytest.raises(ValueError):
            client.chat.completions.create(messages=[{"role": "user", "content": "?"}])


def test_wrap_openai_passthrough_for_other_attrs(tmp_path):
    fake = FakeOpenAI.make()
    fake.embeddings = "passthrough-marker"
    with record(str(tmp_path / "x.sb")) as rec:
        client = wrap_openai(fake, rec, default_model="gpt-4o-mini-2024-07-18")
        assert client.embeddings == "passthrough-marker"


def test_wrap_openai_rejects_non_openai_clients(tmp_path):
    with record(str(tmp_path / "x.sb")) as rec:
        with pytest.raises(TypeError):
            wrap_openai(object(), rec)


def test_wrap_openai_substitution_propagates_dirtiness(tmp_path):
    trace_path = str(tmp_path / "openai.sb")
    with record(trace_path) as rec:
        client = wrap_openai(FakeOpenAI.make(), rec,
                             default_model="gpt-4o-mini-2024-07-18")
        client.chat.completions.create(
            messages=[{"role": "user", "content": "first"}],
        )
        client.chat.completions.create(
            messages=[{"role": "user", "content": "second"}],
        )

    t = replay(trace_path)
    new_messages = [{"role": "user", "content": "REPLACED"}]
    sub = PromptSubstitution(at_step="step:1", new_messages=new_messages)

    # Use a replay-side OpenAI executor (against a fresh fake) so the
    # dirty step actually re-executes through the same code path.
    fake = FakeOpenAI.make()
    exec_ = Executor(llm=openai_executor(fake))
    result = (
        t.branch_at("step:1", "cf")
         .substitute(sub)
         .replay_forward(executor=exec_)
    )
    # step:1 is dirty (its inputs changed); step:2's "context" pointer
    # is rebound to the new outputs hash so it is also dirty.
    dirty_ids = {s.step_id for s in result if s.dirty}
    assert "step:1" in dirty_ids
    assert "step:2" in dirty_ids
    assert exec_.real_calls == 2
    assert len(fake.chat.completions.calls) == 2


# =====================================================================
# Anthropic shim
# =====================================================================


def test_wrap_anthropic_records_and_canonicalises(tmp_path):
    trace_path = str(tmp_path / "anth.sb")
    with record(trace_path) as rec:
        client = wrap_anthropic(FakeAnthropic.make(), rec)
        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            system="You are terse.",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=64,
        )
        assert isinstance(resp, AnthropicMessage)
        assert resp.content[0].type == "text"
        assert resp.stop_reason == "end_turn"
        # Cost is real — Anthropic prices are pinned for this model.
        assert resp.usage["input_tokens"] > 0

    t = replay(trace_path)
    assert len(t.recorded_steps) == 1
    step = t.recorded_steps[0]
    # Recorded `llm_request.messages` includes the system message
    # promoted into the OpenAI-style list.
    msgs = step["llm_request"]["messages"]
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == "You are terse."
    # And the response was projected into OpenAI shape with native
    # payload preserved.
    resp_dict = step["llm_response"]
    assert resp_dict["choices"][0]["finish_reason"] == "stop"
    assert resp_dict["usage"]["prompt_tokens"] > 0
    assert resp_dict["_anthropic"]["stop_reason"] == "end_turn"
    # Cost > 0 because claude-3-5-haiku is in the price list.
    assert step["cost_usd"] > 0


def test_anthropic_executor_replays_dirty_step(tmp_path):
    trace_path = str(tmp_path / "anth.sb")
    with record(trace_path) as rec:
        client = wrap_anthropic(FakeAnthropic.make(), rec)
        client.messages.create(
            model="claude-3-5-haiku-20241022",
            system="sys",
            messages=[{"role": "user", "content": "u1"}],
        )

    t = replay(trace_path)
    fake = FakeAnthropic.make()
    exec_ = Executor(llm=anthropic_executor(fake))
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
    assert len(fake.messages.calls) == 1
    # The replay-side fake saw the user message without the system
    # message (which was extracted into the `system=` kwarg).
    call = fake.messages.calls[0]
    assert call["system"] == "sys"
    assert call["messages"] == [{"role": "user", "content": "u1-changed"}]


def test_wrap_anthropic_rejects_non_anthropic_clients(tmp_path):
    with record(str(tmp_path / "x.sb")) as rec:
        with pytest.raises(TypeError):
            wrap_anthropic(object(), rec)


# =====================================================================
# LangChain tools shim
# =====================================================================


def test_wrap_langchain_tools_records_and_replays(tmp_path):
    trace_path = str(tmp_path / "lc.sb")
    invocations: List[tuple] = []

    def lookup_handler(args):
        invocations.append(("lookup", args))
        return {"customer_id": "acme-uk", "country": "UK"}

    def transfer_handler(args):
        invocations.append(("transfer", args))
        return {"status": "ok", "iban": args["iban"]}

    real_tools = [
        FakeLangchainTool(name="lookup_customer",
                          description="lookup", handler=lookup_handler),
        FakeLangchainTool(name="payment.transfer",
                          description="transfer", handler=transfer_handler),
    ]

    with record(trace_path) as rec:
        wrapped = wrap_langchain_tools(real_tools, rec)
        assert [t.name for t in wrapped] == ["lookup_customer", "payment.transfer"]
        result = wrapped[0].invoke({"name": "Acme Bolts"})
        assert result["customer_id"] == "acme-uk"
        wrapped[1].invoke({"iban": "GB99-9999-9999", "amount": 100})

    t = replay(trace_path)
    assert [s["step_kind"] for s in t.recorded_steps] == ["tool_call", "tool_call"]
    assert [s["name"] for s in t.recorded_steps] == ["lookup_customer", "payment.transfer"]

    # Replay from cache (zero real tool calls).
    invocations.clear()
    exec_ = Executor(tool=langchain_tool_executor(real_tools))
    result = t.replay_forward(executor=exec_)
    assert exec_.real_calls == 0
    assert invocations == []
    assert all(not s.dirty for s in result)


def test_lc_tool_substitution_fixes_the_bug(tmp_path):
    """Reproduce the README's bisect-and-fix flow via the LC shim."""
    trace_path = str(tmp_path / "lc.sb")

    def lookup(args):
        return {"customer_id": "acme-uk", "country": "UK", "iban": "GB99"}

    def transfer(args):
        return {"status": "ok", "wire_to_iban": args["iban"]}

    real_tools = [
        FakeLangchainTool(name="lookup_customer", description="", handler=lookup),
        FakeLangchainTool(name="payment.transfer", description="", handler=transfer),
    ]

    with record(trace_path) as rec:
        wrapped = wrap_langchain_tools(real_tools, rec)
        row = wrapped[0].invoke({"name": "Acme Bolts"})
        wrapped[1].invoke({"iban": row["iban"], "amount": 50000})

    t = replay(trace_path)
    fixed = ToolOutputSubstitution(
        at_step="step:1",
        fake_response={"result": {"customer_id": "acme-us", "country": "US",
                                   "iban": "US12-3456-7890"}},
    )
    exec_ = Executor(tool=langchain_tool_executor(real_tools))
    result = (
        t.branch_at("step:1", "fixed").substitute(fixed).replay_forward(executor=exec_)
    )
    dirty = {s.step_id for s in result if s.dirty}
    # step:1's output was forced; step:2's "context" depends on it.
    assert dirty == {"step:1", "step:2"}


def test_wrap_langchain_tool_rejects_non_tool_objects(tmp_path):
    with record(str(tmp_path / "x.sb")) as rec:
        with pytest.raises(TypeError):
            wrap_langchain_tool(object(), rec)


# =====================================================================
# MCP session shim
# =====================================================================


def test_wrap_mcp_session_records_with_server_namespace(tmp_path):
    trace_path = str(tmp_path / "mcp.sb")
    invocations: List[tuple] = []

    def search(args):
        invocations.append(("search", args))
        return {"hits": [args.get("q", "")]}

    session = FakeMCPSession(server="search-srv", handlers={"search": search})

    with record(trace_path) as rec:
        wrapped = wrap_mcp_session(session, rec, server_name="search-srv")
        result = wrapped.call_tool("search", {"q": "stepback"})
        assert result == {"hits": ["stepback"]}
        # passthrough still works
        assert wrapped.list_tools() == ["search"]

    t = replay(trace_path)
    assert len(t.recorded_steps) == 1
    assert t.recorded_steps[0]["name"] == "search-srv:search"

    # Replay through mcp_tool_executor against the same server.
    invocations.clear()
    exec_ = Executor(tool=mcp_tool_executor({"search-srv": session}))
    result = t.replay_forward(executor=exec_)
    assert exec_.real_calls == 0
    assert invocations == []
    assert all(not s.dirty for s in result)


def test_mcp_executor_dispatches_dirty_call_to_right_server(tmp_path):
    trace_path = str(tmp_path / "mcp.sb")
    s1_calls, s2_calls = [], []
    s1 = FakeMCPSession(server="srv1", handlers={
        "ping": lambda a: s1_calls.append(a) or {"pong": "from-1"}})
    s2 = FakeMCPSession(server="srv2", handlers={
        "ping": lambda a: s2_calls.append(a) or {"pong": "from-2"}})

    with record(trace_path) as rec:
        w1 = wrap_mcp_session(s1, rec, server_name="srv1")
        w2 = wrap_mcp_session(s2, rec, server_name="srv2")
        w1.call_tool("ping", {"x": 1})
        w2.call_tool("ping", {"x": 2})

    t = replay(trace_path)
    sub = ToolOutputSubstitution(
        at_step="step:1",
        fake_response={"result": {"pong": "FORCED"}},
    )
    s1_calls.clear(); s2_calls.clear()
    exec_ = Executor(tool=mcp_tool_executor({"srv1": s1, "srv2": s2}))
    result = (
        t.branch_at("step:1", "cf").substitute(sub).replay_forward(executor=exec_)
    )
    # step:1 forced (no real call), step:2's context changed → dirty (real call).
    dirty = {s.step_id for s in result if s.dirty}
    assert dirty == {"step:1", "step:2"}
    assert s1_calls == []  # step:1 was forced not invoked on replay
    assert len(s2_calls) == 1  # step:2 re-executed against srv2 on replay


def test_wrap_mcp_session_rejects_non_session(tmp_path):
    with record(str(tmp_path / "x.sb")) as rec:
        with pytest.raises(TypeError):
            wrap_mcp_session(object(), rec)


# =====================================================================
# Multi-provider trace (the README's promise)
# =====================================================================


def test_mixed_provider_trace_replays_from_cache(tmp_path):
    """A single trace mixes OpenAI llm_call, Anthropic llm_call, LangChain
    tool_call, and MCP tool_call frames — and still replays 100% from
    the content-addressed cache."""
    trace_path = str(tmp_path / "mixed.sb")

    oai = FakeOpenAI.make()
    anth = FakeAnthropic.make()
    lookup_tool = FakeLangchainTool(
        name="lookup", description="",
        handler=lambda a: {"id": "x"},
    )
    mcp_session = FakeMCPSession(server="srv", handlers={
        "fetch": lambda a: {"data": [1, 2, 3]},
    })

    with record(trace_path) as rec:
        oai_w = wrap_openai(oai, rec, default_model="gpt-4o-mini-2024-07-18")
        anth_w = wrap_anthropic(anth, rec)
        lc_w = wrap_langchain_tool(lookup_tool, rec)
        mcp_w = wrap_mcp_session(mcp_session, rec, server_name="srv")

        oai_w.chat.completions.create(messages=[{"role": "user", "content": "u"}])
        anth_w.messages.create(model="claude-3-5-haiku-20241022",
                               messages=[{"role": "user", "content": "u"}])
        lc_w.invoke({"q": "y"})
        mcp_w.call_tool("fetch", {"id": 7})

    t = replay(trace_path)
    kinds = [s["step_kind"] for s in t.recorded_steps]
    assert kinds == ["llm_call", "llm_call", "tool_call", "tool_call"]

    exec_ = Executor()
    result = t.replay_forward(executor=exec_)
    assert exec_.real_calls == 0
    assert all(not s.dirty for s in result)
