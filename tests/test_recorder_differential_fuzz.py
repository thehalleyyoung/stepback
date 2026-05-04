"""Differential fuzzing across recorders.

Implements step #49 of ``100_STEPS.md``:

    Add differential fuzzing across recorders that generate semantically
    equivalent requests for different providers.

Each provider shim in :mod:`stepback.shims` accepts a provider-specific
request shape (OpenAI ``messages=[{role, content}]``, Anthropic
``messages=[...]`` plus ``system=...``, Bedrock ``messages=[{role,
content:[{text}]}]`` plus ``system=[{text}]``, Gemini
``contents=[{role, parts:[{text}]}]`` plus ``config.system_instruction``)
and translates it onto a single canonical surface:

  * the recorder writes a unified OpenAI-shape ``inputs.messages`` list,
    used for canonical input hashing and substitution,
  * the recorder writes a unified OpenAI-shape ``llm_response`` (with the
    raw provider payload preserved under a ``_anthropic`` / ``_bedrock``
    / ``_gemini`` sidecar).

If two semantically-equivalent calls land on different canonical inputs
or different canonical responses, the dirty-set engine cannot share
caches across providers, model-swap differential testing breaks, and
substitutions written against one provider don't replay against
another.

This file generates a large family of semantically-equivalent unified
requests (system + user/assistant turns, sampling params, optional tool
spec) and a matching family of unified responses (text, tool calls,
finish reason, token usage). For each test point we synthesise the
provider-specific request and provider-specific raw response for all
four shipped LLM provider shims (OpenAI, Anthropic, Bedrock, Gemini),
drive every shim through ``stepback.record(...)``, and assert:

  1. ``inputs.messages`` is byte-identical across all four shims modulo
     ``_bedrock_blocks`` / ``_gemini_parts`` markers that round-trip
     non-text blocks (we strip those for the comparison; their absence
     is verified independently).
  2. The canonical input hash (i.e. the dirty-set cache key) is
     identical across providers when ``model``, ``temperature``, and
     ``seed`` are pinned to the same canonical values.
  3. The OpenAI-shape ``llm_response`` agrees on
     ``choices[0].message.content``, ``choices[0].finish_reason``,
     ``usage.{prompt_tokens,completion_tokens,total_tokens}``, and on
     the *function/arguments* of ``choices[0].message.tool_calls``
     (modulo the tool-call ``id``, which every provider names
     differently and which the OpenAI shape does not normalize).
  4. Recorded steps round-trip through ``replay`` with zero LLM calls
     (every provider lands the same canonical entry in the step cache).

The combination "fuzz the unified request + four parallel recorders"
gives us actual differential coverage: a regression in any single
shim's canonicalisation surfaces as a per-provider divergence.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, note, settings, strategies as st

from stepback import record, replay
from stepback.canonical import hash_obj
from stepback.replay import Executor
from stepback.shims import (
    wrap_anthropic,
    wrap_bedrock,
    wrap_gemini,
    wrap_openai,
)
from stepback.substitutions import SubstitutionSet


# =====================================================================
# Provider fakes (smallest possible duck-typed clients)
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


@dataclass
class _BedrockClientFake:
    raw_response: dict
    calls: list = field(default_factory=list)

    def converse(self, **kwargs: Any) -> dict:
        self.calls.append(kwargs)
        return dict(self.raw_response)


@dataclass
class _GeminiModelsFake:
    raw_response: dict
    calls: list = field(default_factory=list)

    def generate_content(self, *, model: str, contents: Any,
                         config: Any = None, **kwargs: Any) -> dict:
        self.calls.append({"model": model, "contents": contents,
                           "config": config, "kwargs": kwargs})
        return dict(self.raw_response)


@dataclass
class _GeminiClientFake:
    models: _GeminiModelsFake


# =====================================================================
# Unified semantic request + response specs
# =====================================================================


@dataclass(frozen=True)
class _UnifiedTurn:
    role: str  # "system" | "user" | "assistant"
    text: str


@dataclass(frozen=True)
class _UnifiedToolCall:
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class _UnifiedRequest:
    system: Optional[str]
    turns: Tuple[_UnifiedTurn, ...]    # user/assistant turns only (system handled separately)
    temperature: float
    seed: int
    max_tokens: int
    tool_spec: Optional[Tuple[str, Tuple[str, ...]]]  # (tool_name, (param_name1,...)) or None


@dataclass(frozen=True)
class _UnifiedResponse:
    text: Optional[str]
    tool_calls: Tuple[_UnifiedToolCall, ...]
    finish_reason: str  # "stop" | "length" | "tool_calls" | "content_filter"
    usage_in: int
    usage_out: int


# =====================================================================
# Strategies
# =====================================================================


_safe_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    min_size=0, max_size=24,
).map(str.strip)

_role_alt = st.sampled_from(["user", "assistant"])


@st.composite
def _unified_request(draw: st.DrawFn) -> _UnifiedRequest:
    has_system = draw(st.booleans())
    system = draw(_safe_text) if has_system else None
    if system == "":
        system = None
    n_turns = draw(st.integers(min_value=1, max_value=4))
    turns: List[_UnifiedTurn] = []
    role = "user"
    for _ in range(n_turns):
        text = draw(_safe_text)
        turns.append(_UnifiedTurn(role=role, text=text))
        role = "assistant" if role == "user" else "user"
    # Last turn must be a user turn for every provider to accept the
    # request (Anthropic in particular). Rebuild if we ended on assistant.
    if turns[-1].role != "user":
        turns.append(_UnifiedTurn(role="user", text=draw(_safe_text)))
    temperature = draw(st.sampled_from([0.0, 0.2, 0.7, 1.0]))
    seed = draw(st.sampled_from([0, 1, 42, 1234]))
    max_tokens = draw(st.sampled_from([16, 64, 256, 1024]))
    has_tools = draw(st.booleans())
    if has_tools:
        params = tuple(sorted(draw(st.lists(
            st.sampled_from(["query", "id", "limit", "verbose"]),
            min_size=1, max_size=3, unique=True,
        ))))
        tool_spec: Optional[Tuple[str, Tuple[str, ...]]] = ("lookup", params)
    else:
        tool_spec = None
    return _UnifiedRequest(
        system=system, turns=tuple(turns), temperature=temperature,
        seed=seed, max_tokens=max_tokens, tool_spec=tool_spec,
    )


@st.composite
def _unified_response(draw: st.DrawFn, req: _UnifiedRequest) -> _UnifiedResponse:
    """Build a response shape consistent with the request's tool spec.

    Currently used only by the regression cases (the property test
    inlines a deterministic ``random.Random``-driven variant for tighter
    control over the request/response correspondence). Kept here so
    future fuzz layers — e.g. structured tool-arguments — can reuse it.
    """
    if req.tool_spec is not None and draw(st.booleans()):
        # tool-call response
        params = req.tool_spec[1]
        args: Dict[str, Any] = {}
        for p in params:
            args[p] = draw(st.one_of(
                st.integers(min_value=0, max_value=99),
                st.text(alphabet=st.characters(blacklist_categories=("Cs",),
                                               blacklist_characters="\x00\""),
                        min_size=0, max_size=8),
            ))
        return _UnifiedResponse(
            text=None,
            tool_calls=(_UnifiedToolCall(name=req.tool_spec[0], arguments=args),),
            finish_reason="tool_calls",
            usage_in=draw(st.integers(min_value=0, max_value=200)),
            usage_out=draw(st.integers(min_value=0, max_value=200)),
        )
    text = draw(_safe_text)
    finish = draw(st.sampled_from(["stop", "length", "content_filter"]))
    return _UnifiedResponse(
        text=text or "",
        tool_calls=(),
        finish_reason=finish,
        usage_in=draw(st.integers(min_value=0, max_value=200)),
        usage_out=draw(st.integers(min_value=0, max_value=200)),
    )


# =====================================================================
# Provider-specific lowering of the unified request
# =====================================================================


def _to_openai_messages(req: _UnifiedRequest) -> List[dict]:
    msgs: List[dict] = []
    if req.system:
        msgs.append({"role": "system", "content": req.system})
    for t in req.turns:
        msgs.append({"role": t.role, "content": t.text})
    return msgs


def _to_anthropic(req: _UnifiedRequest) -> Tuple[List[dict], Optional[str]]:
    return [{"role": t.role, "content": t.text} for t in req.turns], req.system


def _to_bedrock(req: _UnifiedRequest) -> Tuple[List[dict], Optional[List[dict]]]:
    msgs = [{"role": t.role, "content": [{"text": t.text}]} for t in req.turns]
    system = [{"text": req.system}] if req.system else None
    return msgs, system


_GEMINI_ROLE_TO = {"user": "user", "assistant": "model"}


def _to_gemini(req: _UnifiedRequest) -> Tuple[List[dict], Optional[dict]]:
    contents = [
        {"role": _GEMINI_ROLE_TO[t.role], "parts": [{"text": t.text}]}
        for t in req.turns
    ]
    sys_inst = ({"parts": [{"text": req.system}]}) if req.system else None
    return contents, sys_inst


# =====================================================================
# Provider-specific lowering of the unified response (raw SDK shape)
# =====================================================================


def _openai_raw(resp: _UnifiedResponse, model: str) -> dict:
    finish_map = {
        "stop": "stop",
        "length": "length",
        "tool_calls": "tool_calls",
        "content_filter": "content_filter",
    }
    msg: Dict[str, Any] = {"role": "assistant", "content": resp.text}
    if resp.tool_calls:
        msg["tool_calls"] = [
            {
                "id": f"call_{i:04d}",
                "type": "function",
                "function": {"name": tc.name,
                             "arguments": dict(tc.arguments)},
            }
            for i, tc in enumerate(resp.tool_calls)
        ]
    return {
        "id": "chatcmpl_diff_fuzz",
        "model": model,
        "choices": [{
            "index": 0,
            "finish_reason": finish_map[resp.finish_reason],
            "message": msg,
        }],
        "usage": {
            "prompt_tokens": resp.usage_in,
            "completion_tokens": resp.usage_out,
            "total_tokens": resp.usage_in + resp.usage_out,
        },
    }


def _anthropic_raw(resp: _UnifiedResponse, model: str) -> dict:
    finish_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",  # Anthropic has no content_filter
    }
    blocks: List[dict] = []
    if resp.text is not None and resp.text != "":
        blocks.append({"type": "text", "text": resp.text})
    for i, tc in enumerate(resp.tool_calls):
        blocks.append({
            "type": "tool_use",
            "id": f"toolu_{i:04d}",
            "name": tc.name,
            "input": dict(tc.arguments),
        })
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return {
        "id": "msg_diff_fuzz",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": finish_map[resp.finish_reason],
        "stop_sequence": None,
        "usage": {
            "input_tokens": resp.usage_in,
            "output_tokens": resp.usage_out,
        },
    }


def _bedrock_raw(resp: _UnifiedResponse, model: str) -> dict:
    finish_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "guardrail_intervened",
    }
    blocks: List[dict] = []
    if resp.text is not None and resp.text != "":
        blocks.append({"text": resp.text})
    for i, tc in enumerate(resp.tool_calls):
        blocks.append({
            "toolUse": {
                "toolUseId": f"bedtu_{i:04d}",
                "name": tc.name,
                "input": dict(tc.arguments),
            },
        })
    if not blocks:
        blocks.append({"text": ""})
    return {
        "ResponseMetadata": {"RequestId": "req-diff-fuzz"},
        "_modelId": model,
        "output": {"message": {"role": "assistant", "content": blocks}},
        "stopReason": finish_map[resp.finish_reason],
        "usage": {
            "inputTokens": resp.usage_in,
            "outputTokens": resp.usage_out,
            "totalTokens": resp.usage_in + resp.usage_out,
        },
    }


def _gemini_raw(resp: _UnifiedResponse, model: str) -> dict:
    finish_map = {
        "stop": "STOP",
        "length": "MAX_TOKENS",
        "tool_calls": "STOP",
        "content_filter": "SAFETY",
    }
    parts: List[dict] = []
    if resp.text is not None and resp.text != "":
        parts.append({"text": resp.text})
    for tc in resp.tool_calls:
        parts.append({
            "function_call": {"name": tc.name, "args": dict(tc.arguments)},
        })
    if not parts:
        parts.append({"text": ""})
    return {
        "response_id": "resp-diff-fuzz",
        "model_version": model,
        "candidates": [{
            "content": {"role": "model", "parts": parts},
            "finish_reason": finish_map[resp.finish_reason],
            "index": 0,
        }],
        "usage_metadata": {
            "prompt_token_count": resp.usage_in,
            "candidates_token_count": resp.usage_out,
            "total_token_count": resp.usage_in + resp.usage_out,
        },
    }


# =====================================================================
# Comparison helpers
# =====================================================================


def _strip_provider_markers(messages: List[dict]) -> List[dict]:
    """Drop ``_bedrock_blocks`` and ``_gemini_parts`` so we can compare
    text-only payloads. Pure-text fuzz inputs should never populate
    these fields; we still guard against accidental leakage by
    stripping them and asserting they were empty."""
    out = []
    for m in messages:
        clean = {k: v for k, v in m.items()
                 if k not in ("_bedrock_blocks", "_gemini_parts")}
        out.append(clean)
    return out


def _ensure_no_structured_block_leak(messages: List[dict]) -> None:
    for m in messages:
        if "_bedrock_blocks" in m:
            raise AssertionError(
                f"text-only fuzz request leaked _bedrock_blocks into "
                f"recorded inputs: {m['_bedrock_blocks']!r}"
            )
        if "_gemini_parts" in m:
            raise AssertionError(
                f"text-only fuzz request leaked _gemini_parts into "
                f"recorded inputs: {m['_gemini_parts']!r}"
            )


def _normalize_response_for_diff(resp: dict) -> dict:
    """Strip provider-native sidecars and tool-call ids so we can
    compare across providers."""
    out = {k: v for k, v in resp.items()
           if k not in ("_anthropic", "_bedrock", "_gemini",
                        "id", "model")}
    choices = []
    for ch in out.get("choices") or []:
        new_ch = dict(ch)
        msg = dict(new_ch.get("message") or {})
        tcs = msg.get("tool_calls")
        if tcs:
            msg["tool_calls"] = [
                {
                    "type": tc.get("type"),
                    "function": {
                        "name": (tc.get("function") or {}).get("name"),
                        "arguments": (tc.get("function") or {}).get("arguments"),
                    },
                }
                for tc in tcs
            ]
        new_ch["message"] = msg
        choices.append(new_ch)
    out["choices"] = choices
    return out


# =====================================================================
# Per-provider drivers
# =====================================================================


_OPENAI_MODEL = "gpt-4o-mini-2024-07-18"
_ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"
_BEDROCK_MODEL = "anthropic.claude-3-5-haiku-20241022-v1:0"
_GEMINI_MODEL = "models/gemini-2.5-flash"


def _drive_openai(req: _UnifiedRequest, raw: dict, *, tmpdir: str) -> dict:
    fake = _OAIClientFake(chat=_OAIChatFake(
        completions=_OAICompletionsFake(raw_response=raw),
    ))
    path = os.path.join(tmpdir, "openai.sb")
    captured: List[dict] = []
    with record(path) as rec:
        client = wrap_openai(fake, rec, default_model=_OPENAI_MODEL)
        kw: Dict[str, Any] = {
            "messages": _to_openai_messages(req),
            "temperature": req.temperature,
            "seed": req.seed,
            "max_tokens": req.max_tokens,
        }
        if req.tool_spec is not None:
            kw["tools"] = [{
                "type": "function",
                "function": {"name": req.tool_spec[0],
                             "parameters": {"type": "object",
                                            "properties": {
                                                p: {"type": "string"}
                                                for p in req.tool_spec[1]
                                            }}},
            }]
        client.chat.completions.create(**kw)
        captured.extend(rec.steps)
    return {"path": path, "fake": fake, "steps": captured}


def _drive_anthropic(req: _UnifiedRequest, raw: dict, *, tmpdir: str) -> dict:
    fake = _AnthClientFake(messages=_AnthMessagesFake(raw_response=raw))
    msgs, system = _to_anthropic(req)
    path = os.path.join(tmpdir, "anthropic.sb")
    captured: List[dict] = []
    with record(path) as rec:
        client = wrap_anthropic(fake, rec)
        kw: Dict[str, Any] = {
            "model": _ANTHROPIC_MODEL,
            "messages": msgs,
            "temperature": req.temperature,
            "seed": req.seed,
            "max_tokens": req.max_tokens,
        }
        if system is not None:
            kw["system"] = system
        if req.tool_spec is not None:
            kw["tools"] = [{
                "name": req.tool_spec[0],
                "input_schema": {"type": "object",
                                 "properties": {
                                     p: {"type": "string"}
                                     for p in req.tool_spec[1]
                                 }},
            }]
        client.messages.create(**kw)
        captured.extend(rec.steps)
    return {"path": path, "fake": fake, "steps": captured}


def _drive_bedrock(req: _UnifiedRequest, raw: dict, *, tmpdir: str) -> dict:
    fake = _BedrockClientFake(raw_response=raw)
    msgs, system = _to_bedrock(req)
    path = os.path.join(tmpdir, "bedrock.sb")
    captured: List[dict] = []
    with record(path) as rec:
        client = wrap_bedrock(fake, rec)
        kw: Dict[str, Any] = {
            "modelId": _BEDROCK_MODEL,
            "messages": msgs,
            "inferenceConfig": {"temperature": req.temperature,
                                "maxTokens": req.max_tokens,
                                "seed": req.seed},
        }
        if system is not None:
            kw["system"] = system
        if req.tool_spec is not None:
            kw["toolConfig"] = {"tools": [{
                "toolSpec": {
                    "name": req.tool_spec[0],
                    "inputSchema": {"json": {
                        "type": "object",
                        "properties": {
                            p: {"type": "string"} for p in req.tool_spec[1]
                        },
                    }},
                },
            }]}
        client.converse(**kw)
        captured.extend(rec.steps)
    return {"path": path, "fake": fake, "steps": captured}


def _drive_gemini(req: _UnifiedRequest, raw: dict, *, tmpdir: str) -> dict:
    fake = _GeminiClientFake(models=_GeminiModelsFake(raw_response=raw))
    contents, sys_inst = _to_gemini(req)
    cfg: Dict[str, Any] = {
        "temperature": req.temperature,
        "seed": req.seed,
        "max_output_tokens": req.max_tokens,
    }
    if sys_inst is not None:
        cfg["system_instruction"] = sys_inst
    if req.tool_spec is not None:
        cfg["tools"] = [{
            "function_declarations": [{
                "name": req.tool_spec[0],
                "parameters": {"type": "object",
                               "properties": {
                                   p: {"type": "string"}
                                   for p in req.tool_spec[1]
                               }},
            }],
        }]
    path = os.path.join(tmpdir, "gemini.sb")
    captured: List[dict] = []
    with record(path) as rec:
        client = wrap_gemini(fake, rec, default_model=_GEMINI_MODEL)
        client.models.generate_content(contents=contents, config=cfg)
        captured.extend(rec.steps)
    return {"path": path, "fake": fake, "steps": captured}


# =====================================================================
# Recorded-step extraction
# =====================================================================


def _read_llm_step(drv: dict) -> dict:
    """Return the single ``llm_call`` step recorded by a driver."""
    steps = [s for s in drv["steps"] if s.get("step_kind") == "llm_call"]
    assert len(steps) == 1, (
        f"expected 1 llm_call step in {drv['path']}, got {len(steps)}"
    )
    return steps[0]


# =====================================================================
# The differential property
# =====================================================================


def _assert_consistent_recordings(steps: Dict[str, dict]) -> None:
    """All four recordings must agree modulo provider-specific knobs."""
    # 1. inputs.messages — strip provider markers and assert equality.
    msg_lists: Dict[str, List[dict]] = {}
    for prov, step in steps.items():
        msgs = step["inputs"]["messages"]
        _ensure_no_structured_block_leak(msgs)
        msg_lists[prov] = _strip_provider_markers(msgs)
    ref_prov = "openai"
    ref = msg_lists[ref_prov]
    for prov, msgs in msg_lists.items():
        if msgs != ref:
            raise AssertionError(
                f"inputs.messages diverge between {ref_prov} and {prov}:\n"
                f"  {ref_prov}: {ref!r}\n"
                f"  {prov}:    {msgs!r}"
            )

    # 2. canonical input hash — only the bits the dirty-set engine uses.
    #    The recorder's hash includes ``model``, which differs across
    #    providers (canonical_*_model_id maps Bedrock+Gemini onto their
    #    pricing key but OpenAI / Anthropic stay literal); ``tools`` is
    #    recorded as the user-provided provider-native tool spec (every
    #    provider has a different schema for tool declarations) and is
    #    deliberately *not* normalised by the shim. We hash the
    #    intersection slice — kind, temperature, seed, messages,
    #    response_format, and any data-dependency context — and check
    #    that's identical across providers.
    hashes: Dict[str, str] = {}
    for prov, step in steps.items():
        slice_ = {k: v for k, v in step["inputs"].items()
                  if k not in ("model", "tools")}
        hashes[prov] = hash_obj(slice_)
    ref_h = hashes[ref_prov]
    for prov, h in hashes.items():
        if h != ref_h:
            raise AssertionError(
                f"canonical input hash (model-stripped) diverges between "
                f"{ref_prov} ({ref_h}) and {prov} ({h}); inputs:\n"
                f"  {ref_prov}: {steps[ref_prov]['inputs']!r}\n"
                f"  {prov}:    {steps[prov]['inputs']!r}"
            )

    # 3. llm_response — content / finish_reason / usage / tool fn.
    norm: Dict[str, dict] = {}
    for prov, step in steps.items():
        norm[prov] = _normalize_response_for_diff(step["llm_response"])
    ref_resp = norm[ref_prov]
    ref_msg = ref_resp["choices"][0]["message"]
    ref_fr = ref_resp["choices"][0]["finish_reason"]
    ref_usage = ref_resp["usage"]
    for prov, r in norm.items():
        msg = r["choices"][0]["message"]
        fr = r["choices"][0]["finish_reason"]
        usage = r["usage"]
        # finish_reason: every provider but Anthropic has a content_filter
        # mapping; Anthropic emits "stop" for content_filter inputs. Only
        # require the OpenAI/Bedrock/Gemini providers to converge on the
        # mapped value; allow Anthropic to emit "stop" for a content_filter
        # request.
        if fr != ref_fr:
            # Cross-provider finish_reason caveats:
            #   * Anthropic has no content_filter signal — it emits
            #     "stop" for safety stops, so accept that mapping.
            #   * Gemini has no dedicated tool_calls finish_reason —
            #     it returns STOP and signals the call via a
            #     function_call part. Accept "stop" when the
            #     normalised message has tool_calls.
            tolerated = False
            if prov == "anthropic" and ref_fr == "content_filter" and fr == "stop":
                tolerated = True
            if (prov == "gemini" and ref_fr == "tool_calls" and fr == "stop"
                    and msg.get("tool_calls")):
                tolerated = True
            if not tolerated:
                raise AssertionError(
                    f"finish_reason diverges between {ref_prov}={ref_fr!r} "
                    f"and {prov}={fr!r}"
                )
        if (msg.get("content") or "") != (ref_msg.get("content") or ""):
            raise AssertionError(
                f"choices[0].message.content diverges between "
                f"{ref_prov}={ref_msg.get('content')!r} and "
                f"{prov}={msg.get('content')!r}"
            )
        if msg.get("tool_calls") != ref_msg.get("tool_calls"):
            raise AssertionError(
                f"tool_calls diverge between {ref_prov} and {prov}:\n"
                f"  {ref_prov}: {ref_msg.get('tool_calls')!r}\n"
                f"  {prov}:    {msg.get('tool_calls')!r}"
            )
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if usage.get(k) != ref_usage.get(k):
                raise AssertionError(
                    f"usage.{k} diverges between {ref_prov}={ref_usage.get(k)!r} "
                    f"and {prov}={usage.get(k)!r}"
                )

    # 4. Each recorded trace must replay with zero LLM calls (every
    #    step is served from the cache).
    for prov, step in steps.items():
        path = step.get("__path")
        if not path:
            continue
        t = replay(path)
        result = t.run_replay(subs=SubstitutionSet(), executor=Executor())
        if any(s.dirty for s in result):
            dirty = [s for s in result if s.dirty]
            raise AssertionError(
                f"{prov} trace replayed dirty steps: {dirty!r}"
            )


# =====================================================================
# The actual property test
# =====================================================================


@settings(
    deadline=None,
    max_examples=120,
    suppress_health_check=[HealthCheck.too_slow,
                           HealthCheck.function_scoped_fixture],
)
@given(req=_unified_request(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_recorders_agree_on_canonical_inputs_and_outputs(req, seed):
    # Keep response generation deterministic per (request, seed) so
    # shrinkage points at the unified request, not at a re-rolled
    # response. Hypothesis can't easily compose a strategy that *takes*
    # the drawn request, so we do the second-stage roll manually.
    import random
    rng = random.Random(seed)

    use_tool = req.tool_spec is not None and rng.random() < 0.5
    if use_tool:
        params = req.tool_spec[1]
        args = {p: rng.choice(["a", "bb", "ccc", "0", "1"]) for p in params}
        resp = _UnifiedResponse(
            text=None,
            tool_calls=(_UnifiedToolCall(name=req.tool_spec[0], arguments=args),),
            finish_reason="tool_calls",
            usage_in=rng.randint(0, 50),
            usage_out=rng.randint(0, 50),
        )
    else:
        finish = rng.choice(["stop", "length", "content_filter"])
        text_pool = ["", "ok", "hello", "done.", "answer: 42"]
        resp = _UnifiedResponse(
            text=rng.choice(text_pool),
            tool_calls=(),
            finish_reason=finish,
            usage_in=rng.randint(0, 50),
            usage_out=rng.randint(0, 50),
        )

    note(f"req={req!r}")
    note(f"resp={resp!r}")

    with tempfile.TemporaryDirectory() as tmp:
        rec_steps: Dict[str, dict] = {}

        oai = _drive_openai(req, _openai_raw(resp, _OPENAI_MODEL), tmpdir=tmp)
        ant = _drive_anthropic(req, _anthropic_raw(resp, _ANTHROPIC_MODEL),
                               tmpdir=tmp)
        bed = _drive_bedrock(req, _bedrock_raw(resp, _BEDROCK_MODEL),
                             tmpdir=tmp)
        gem = _drive_gemini(req, _gemini_raw(resp, _GEMINI_MODEL), tmpdir=tmp)

        for prov, drv in (("openai", oai), ("anthropic", ant),
                          ("bedrock", bed), ("gemini", gem)):
            step = _read_llm_step(drv)
            step["__path"] = drv["path"]
            rec_steps[prov] = step

        # Sanity: every fake was actually invoked.
        assert len(oai["fake"].chat.completions.calls) == 1
        assert len(ant["fake"].messages.calls) == 1
        assert len(bed["fake"].calls) == 1
        assert len(gem["fake"].models.calls) == 1

        _assert_consistent_recordings(rec_steps)


# =====================================================================
# Concrete regression cases (no Hypothesis): hand-picked corners
# =====================================================================


_REGRESSION_CASES: List[Tuple[str, _UnifiedRequest, _UnifiedResponse]] = [
    (
        "simple_text",
        _UnifiedRequest(
            system="Be terse.",
            turns=(_UnifiedTurn("user", "say hi"),),
            temperature=0.0, seed=42, max_tokens=64, tool_spec=None,
        ),
        _UnifiedResponse(text="hi", tool_calls=(),
                         finish_reason="stop", usage_in=5, usage_out=1),
    ),
    (
        "no_system_multiturn",
        _UnifiedRequest(
            system=None,
            turns=(_UnifiedTurn("user", "what is 2+2"),
                   _UnifiedTurn("assistant", "4"),
                   _UnifiedTurn("user", "and 3+3?")),
            temperature=0.7, seed=1234, max_tokens=128, tool_spec=None,
        ),
        _UnifiedResponse(text="6", tool_calls=(),
                         finish_reason="stop", usage_in=12, usage_out=1),
    ),
    (
        "max_tokens_truncation",
        _UnifiedRequest(
            system=None,
            turns=(_UnifiedTurn("user", "recite digits forever"),),
            temperature=0.0, seed=42, max_tokens=4, tool_spec=None,
        ),
        _UnifiedResponse(text="0 1 2 3", tool_calls=(),
                         finish_reason="length", usage_in=7, usage_out=4),
    ),
    (
        "tool_call_response",
        _UnifiedRequest(
            system="tool-augmented",
            turns=(_UnifiedTurn("user", "look up 99"),),
            temperature=0.0, seed=42, max_tokens=256,
            tool_spec=("lookup_customer", ("id",)),
        ),
        _UnifiedResponse(
            text=None,
            tool_calls=(_UnifiedToolCall("lookup_customer", {"id": "99"}),),
            finish_reason="tool_calls", usage_in=14, usage_out=9,
        ),
    ),
    (
        "empty_text_assistant",
        _UnifiedRequest(
            system=None,
            turns=(_UnifiedTurn("user", "ack only"),),
            temperature=0.0, seed=0, max_tokens=16, tool_spec=None,
        ),
        _UnifiedResponse(text="", tool_calls=(),
                         finish_reason="stop", usage_in=2, usage_out=0),
    ),
    (
        "unicode_payload",
        _UnifiedRequest(
            system="日本語で",
            turns=(_UnifiedTurn("user", "こんにちは — émoji 🌟"),),
            temperature=0.2, seed=1, max_tokens=64, tool_spec=None,
        ),
        _UnifiedResponse(text="やぁ 🌟", tool_calls=(),
                         finish_reason="stop", usage_in=8, usage_out=4),
    ),
]


@pytest.mark.parametrize("name,req,resp",
                         _REGRESSION_CASES,
                         ids=[c[0] for c in _REGRESSION_CASES])
def test_recorders_agree_on_hand_picked_corners(name, req, resp, tmp_path):
    tmp = str(tmp_path)
    rec_steps: Dict[str, dict] = {}
    oai = _drive_openai(req, _openai_raw(resp, _OPENAI_MODEL), tmpdir=tmp)
    ant = _drive_anthropic(req, _anthropic_raw(resp, _ANTHROPIC_MODEL),
                           tmpdir=tmp)
    bed = _drive_bedrock(req, _bedrock_raw(resp, _BEDROCK_MODEL), tmpdir=tmp)
    gem = _drive_gemini(req, _gemini_raw(resp, _GEMINI_MODEL), tmpdir=tmp)

    for prov, drv in (("openai", oai), ("anthropic", ant),
                      ("bedrock", bed), ("gemini", gem)):
        step = _read_llm_step(drv)
        step["__path"] = drv["path"]
        rec_steps[prov] = step

    _assert_consistent_recordings(rec_steps)


# =====================================================================
# Negative test: a divergent canonicaliser is detected
# =====================================================================


def test_negative_divergence_is_detected(tmp_path):
    """Sanity check: if we deliberately give one provider a different
    user message, the differential assertion must fire. Guards against
    a silently-passing equality check."""
    tmp = str(tmp_path)
    req = _UnifiedRequest(
        system=None,
        turns=(_UnifiedTurn("user", "same"),),
        temperature=0.0, seed=42, max_tokens=64, tool_spec=None,
    )
    req_alt = _UnifiedRequest(
        system=None,
        turns=(_UnifiedTurn("user", "different"),),
        temperature=0.0, seed=42, max_tokens=64, tool_spec=None,
    )
    resp = _UnifiedResponse(text="ok", tool_calls=(),
                            finish_reason="stop", usage_in=1, usage_out=1)

    oai = _drive_openai(req, _openai_raw(resp, _OPENAI_MODEL), tmpdir=tmp)
    ant = _drive_anthropic(req_alt,
                           _anthropic_raw(resp, _ANTHROPIC_MODEL), tmpdir=tmp)
    bed = _drive_bedrock(req, _bedrock_raw(resp, _BEDROCK_MODEL), tmpdir=tmp)
    gem = _drive_gemini(req, _gemini_raw(resp, _GEMINI_MODEL), tmpdir=tmp)

    rec_steps: Dict[str, dict] = {}
    for prov, drv in (("openai", oai), ("anthropic", ant),
                      ("bedrock", bed), ("gemini", gem)):
        step = _read_llm_step(drv)
        step["__path"] = drv["path"]
        rec_steps[prov] = step

    with pytest.raises(AssertionError, match="inputs.messages diverge"):
        _assert_consistent_recordings(rec_steps)
