"""End-to-end tests for the Google Gemini (google-genai SDK) shim.

The fake below is shaped exactly like a ``google.genai.Client``: a
single ``client.models.generate_content(model=..., contents=...,
config=...)`` method returning the published Gemini response shape.
Tests prove:

1. ``wrap_gemini`` rejects non-Gemini-shaped clients.
2. record→replay-from-cache is a 100% cache hit (zero LLM calls).
3. The recorded canonical ``llm_request.messages`` projects Gemini's
   ``contents=[{role, parts:[{text:...}]}]`` onto the unified
   OpenAI-style list every other shim shares (system_instruction is
   promoted to a ``role: system`` entry; ``model`` role is renamed
   to ``assistant``).
4. A :class:`PromptSubstitution` against a Gemini-recorded step
   correctly dirties downstream steps and re-executes through the
   replay-side :func:`gemini_executor`.
5. ``function_call`` parts round-trip through the canonicaliser as
   OpenAI ``tool_calls``.
6. Pricing: ``gemini-2.5-flash`` is recognised via
   :func:`canonical_gemini_model_id` so cost is non-zero on first
   call; ``models/gemini-2.5-flash`` and the ``-latest`` alias also
   resolve to the same dated row.
7. The Vertex AI ``GenerativeModel`` adapter records via the same
   canonical pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional

import pytest

from stepback import record, replay
from stepback.replay import Executor
from stepback.shims import (
    GeminiResponse,
    canonical_gemini_model_id,
    gemini_executor,
    wrap_gemini,
    wrap_vertex_model,
)
from stepback.substitutions import PromptSubstitution, SubstitutionSet


# =====================================================================
# Fake — duck-typed against google.genai.Client
# =====================================================================


@dataclass
class _FakeGenaiModels:
    calls: List[dict] = field(default_factory=list)
    next_function_call: Optional[dict] = None

    def generate_content(self, *, model: str, contents: Any,
                         config: Any = None, **kwargs: Any) -> dict:
        # Capture exactly what the SDK would have received.
        self.calls.append({
            "model": model, "contents": contents, "config": config,
            "kwargs": kwargs,
        })
        # Concatenate user text from all input contents to drive the
        # body so a cache-hit replay is bit-identical.
        last_user_text_parts: List[str] = []
        if isinstance(contents, str):
            last_user_text_parts.append(contents)
        else:
            for c in contents or []:
                if isinstance(c, Mapping):
                    for p in c.get("parts", []) or []:
                        if isinstance(p, Mapping) and "text" in p:
                            last_user_text_parts.append(p["text"])
        body = (
            f"gem-{model}-msg{len(contents) if not isinstance(contents, str) else 1}"
            f"-call{len(self.calls)}-echo[{''.join(last_user_text_parts)}]"
        )

        parts: List[dict] = [{"text": body}]
        finish = "STOP"
        if self.next_function_call is not None:
            parts.append({"function_call": dict(self.next_function_call)})
            finish = "MALFORMED_FUNCTION_CALL"
            self.next_function_call = None

        # Token counts driven by content lengths so substitutions
        # produce different usages on dirty re-execution.
        in_tok = sum(len(t) for t in last_user_text_parts)
        sys_text = ""
        if isinstance(config, Mapping):
            si = config.get("system_instruction")
            if isinstance(si, str):
                sys_text = si
            elif isinstance(si, Mapping):
                for p in si.get("parts", []) or []:
                    if isinstance(p, Mapping) and "text" in p:
                        sys_text += p["text"]
        in_tok += len(sys_text)
        out_tok = len(body)
        return {
            "candidates": [{
                "content": {"role": "model", "parts": parts},
                "finish_reason": finish,
            }],
            "usage_metadata": {
                "prompt_token_count": in_tok,
                "candidates_token_count": out_tok,
                "total_token_count": in_tok + out_tok,
            },
            "model_version": model,
            "response_id": f"resp-{len(self.calls):04d}",
        }


@dataclass
class _FakeGenaiClient:
    models: _FakeGenaiModels = field(default_factory=_FakeGenaiModels)


# =====================================================================
# Smoke / type guards
# =====================================================================


def test_wrap_gemini_rejects_non_gemini_clients(tmp_path):
    with record(str(tmp_path / "x.sb")) as rec:
        with pytest.raises(TypeError):
            wrap_gemini(object(), rec)


def test_canonical_gemini_model_id_aliases():
    # ``models/`` and ``-latest`` aliases resolve to the dated row.
    assert (
        canonical_gemini_model_id("models/gemini-2.5-flash")
        == "gemini-2.5-flash-2025-04-09"
    )
    assert (
        canonical_gemini_model_id("gemini-2.5-pro-latest")
        == "gemini-2.5-pro-2025-03-25"
    )
    assert (
        canonical_gemini_model_id("publishers/google/models/gemini-2.5-pro")
        == "gemini-2.5-pro-2025-03-25"
    )
    # Already-canonical pass through.
    assert (
        canonical_gemini_model_id("gemini-2.5-flash-2025-04-09")
        == "gemini-2.5-flash-2025-04-09"
    )
    # Unknown ids pass through.
    assert canonical_gemini_model_id("gemini-foo") == "gemini-foo"


def test_wrap_gemini_requires_explicit_or_default_model(tmp_path):
    with record(str(tmp_path / "x.sb")) as rec:
        client = wrap_gemini(_FakeGenaiClient(), rec)
        with pytest.raises(ValueError):
            client.models.generate_content(contents="hi")


# =====================================================================
# Record → replay-from-cache
# =====================================================================


def test_wrap_gemini_records_and_replays_from_cache(tmp_path):
    trace_path = str(tmp_path / "gem.sb")
    fake = _FakeGenaiClient()
    with record(trace_path) as rec:
        client = wrap_gemini(fake, rec, default_model="gemini-2.5-flash")
        resp1 = client.models.generate_content(
            contents=[{"role": "user", "parts": [{"text": "hello"}]}],
            config={
                "system_instruction": "be terse",
                "temperature": 0.0,
                "max_output_tokens": 64,
            },
        )
        # SDK-shaped namespace returned to caller.
        assert isinstance(resp1, GeminiResponse)
        assert resp1.text and resp1.text.startswith("gem-")
        assert resp1.candidates[0].finish_reason == "STOP"
        assert resp1.usage_metadata.prompt_token_count > 0
        assert resp1.usage_metadata.candidates_token_count > 0
        assert resp1.model_version == "gemini-2.5-flash"
        # Dict-style access exposes the canonical OpenAI-shaped payload.
        assert resp1["choices"][0]["message"]["content"].startswith("gem-")

        resp2 = client.models.generate_content(
            model="gemini-2.5-pro",
            contents="again",
        )
        assert resp2.text and resp2.text.startswith("gem-")

    t = replay(trace_path)
    assert len(t.recorded_steps) == 2
    s1, s2 = t.recorded_steps
    assert s1["step_kind"] == "llm_call"

    # Canonical messages: system promoted, model→assistant, parts→content.
    msgs = s1["llm_request"]["messages"]
    assert msgs[0] == {"role": "system", "content": "be terse"}
    assert msgs[1] == {"role": "user", "content": "hello"}

    # Cost non-zero for both — both are Gemini 2.5 priced rows.
    assert s1["llm_request"]["model"] == "gemini-2.5-flash"  # canonical id (alias not needed; literal already canonical for SDK alias)
    assert s2["llm_request"]["model"] == "gemini-2.5-pro"
    # Note: the bare ``gemini-2.5-flash`` SDK alias resolves through
    # the resolver in ``stepback/pricing.py``; we just need cost > 0.
    assert s1["cost_usd"] > 0
    assert s2["cost_usd"] > 0

    # Replay-from-cache: zero real LLM calls.
    exec_ = Executor()
    result = t.run_replay(subs=SubstitutionSet(), executor=exec_)
    assert all(not s.dirty for s in result)
    assert exec_.real_calls == 0


def test_wrap_gemini_models_path_alias_resolves_pricing(tmp_path):
    trace_path = str(tmp_path / "gem_alias.sb")
    fake = _FakeGenaiClient()
    with record(trace_path) as rec:
        client = wrap_gemini(fake, rec, default_model="models/gemini-2.5-flash")
        client.models.generate_content(contents="hi")
    t = replay(trace_path)
    s = t.recorded_steps[0]
    # The ``models/`` alias should canonicalise onto the dated row.
    assert s["llm_request"]["model"] == "gemini-2.5-flash-2025-04-09"
    assert s["cost_usd"] > 0


def test_wrap_gemini_function_call_round_trips_as_tool_calls(tmp_path):
    trace_path = str(tmp_path / "gem_fc.sb")
    fake = _FakeGenaiClient()
    fake.models.next_function_call = {"name": "lookup_customer",
                                      "args": {"name": "Acme"}}
    with record(trace_path) as rec:
        client = wrap_gemini(fake, rec, default_model="gemini-2.5-flash")
        resp = client.models.generate_content(contents="who is Acme?")
        # Native ``function_calls`` exposed on the response namespace.
        assert resp.function_calls and resp.function_calls[0]["name"] == "lookup_customer"
        assert resp.function_calls[0]["args"] == {"name": "Acme"}

    t = replay(trace_path)
    s = t.recorded_steps[0]
    # Canonical OpenAI-shaped ``tool_calls`` exposed to substitutions.
    msg = s["llm_response"]["choices"][0]["message"]
    assert msg["tool_calls"]
    tc = msg["tool_calls"][0]
    assert tc["function"]["name"] == "lookup_customer"
    assert tc["function"]["arguments"] == {"name": "Acme"}
    # Finish reason canonicalised.
    assert s["llm_response"]["choices"][0]["finish_reason"] == "tool_calls"


# =====================================================================
# Substitution + replay-side gemini_executor
# =====================================================================


def test_gemini_executor_replays_dirty_step(tmp_path):
    trace_path = str(tmp_path / "gem_sub.sb")
    with record(trace_path) as rec:
        client = wrap_gemini(_FakeGenaiClient(), rec,
                             default_model="gemini-2.5-flash")
        client.models.generate_content(
            contents=[{"role": "user", "parts": [{"text": "first"}]}],
            config={"system_instruction": "sys-A"},
        )
        client.models.generate_content(
            contents=[{"role": "user", "parts": [{"text": "second"}]}],
            config={"system_instruction": "sys-A"},
        )

    t = replay(trace_path)
    sid = t.recorded_steps[0]["step_id"]
    subs = SubstitutionSet([
        PromptSubstitution(
            at_step=sid,
            new_messages=[
                {"role": "system", "content": "sys-B-rewritten"},
                {"role": "user", "content": "completely different"},
            ],
        ),
    ])
    # Use a fresh fake on the replay side so we can observe the
    # re-executed call.
    replay_fake = _FakeGenaiClient()
    exec_ = Executor(llm=gemini_executor(replay_fake))
    result = t.run_replay(subs=subs, executor=exec_)
    # The substituted step must be dirty and re-executed.
    assert result[0].dirty
    assert exec_.real_calls >= 1
    # The replay-side fake received Gemini-shaped contents
    # (``[{role, parts:[{text:...}]}]``), not the unified messages.
    # The replay-side fake received Gemini-shaped contents
    # (``[{role, parts:[{text:...}]}]``), not the unified messages.
    # ``calls[0]`` is the re-execution of the substituted step;
    # later calls are downstream dirty propagation.
    first = replay_fake.models.calls[0]
    assert isinstance(first["contents"], list)
    assert first["contents"][0]["role"] == "user"
    assert first["contents"][0]["parts"][0]["text"] == "completely different"
    # System instruction was lifted back into the config.
    assert first["config"]["system_instruction"]["parts"][0]["text"] == "sys-B-rewritten"


# =====================================================================
# Vertex AI adapter
# =====================================================================


@dataclass
class _FakeVertexModel:
    """Duck-typed against ``vertexai.generative_models.GenerativeModel``."""
    model_name: str = "gemini-2.5-flash"
    calls: List[dict] = field(default_factory=list)

    def generate_content(self, contents: Any, *,
                         generation_config: Any = None,
                         system_instruction: Any = None,
                         tools: Any = None,
                         **kwargs: Any) -> dict:
        self.calls.append({
            "contents": contents,
            "generation_config": generation_config,
            "system_instruction": system_instruction,
            "tools": tools, "kwargs": kwargs,
        })
        text = ""
        if isinstance(contents, str):
            text = contents
        else:
            for c in contents or []:
                if isinstance(c, Mapping):
                    for p in c.get("parts", []) or []:
                        if isinstance(p, Mapping) and "text" in p:
                            text += p["text"]
        body = f"vertex-{self.model_name}-echo[{text}]"
        return {
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": body}]},
                "finish_reason": "STOP",
            }],
            "usage_metadata": {
                "prompt_token_count": max(1, len(text)),
                "candidates_token_count": len(body),
                "total_token_count": max(1, len(text)) + len(body),
            },
            "model_version": self.model_name,
        }


def test_wrap_vertex_model_records_via_canonical_pipeline(tmp_path):
    trace_path = str(tmp_path / "vertex.sb")
    vmodel = _FakeVertexModel(model_name="gemini-2.5-flash")
    with record(trace_path) as rec:
        proxy = wrap_vertex_model(vmodel, rec)
        resp = proxy.generate_content(
            contents=[{"role": "user", "parts": [{"text": "vx-hi"}]}],
            config={"system_instruction": "vx-sys", "temperature": 0.0,
                    "max_output_tokens": 32},
        )
        assert isinstance(resp, GeminiResponse)
        assert resp.text and resp.text.startswith("vertex-")

    # Vertex received system_instruction lifted out of config and
    # passed as a top-level kwarg (its native surface).
    assert vmodel.calls[0]["system_instruction"] is not None
    assert vmodel.calls[0]["generation_config"] == {
        "temperature": 0.0, "max_output_tokens": 32,
    }

    t = replay(trace_path)
    s = t.recorded_steps[0]
    assert s["llm_request"]["model"] == "gemini-2.5-flash"
    assert s["llm_request"]["messages"][0] == {"role": "system",
                                                "content": "vx-sys"}
    assert s["cost_usd"] > 0
