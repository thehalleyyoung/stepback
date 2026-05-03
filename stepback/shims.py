"""LLM-client and tool-registry shims.

The README's v0.1 target is ``the Python SDK covering the OpenAI +
Anthropic clients and the LangChain tool registry``. The base
:class:`stepback.recorder.Recorder` exposes explicit
``llm_call(model, messages, executor=...)`` /
``tool_call(name, arguments, executor=...)`` primitives so the
substitution + replay semantics can be tested in isolation. This
module sits on top and provides the *adapter* layer agents actually
use in production: drop-in wrappers around the real client objects so
existing agent code records itself just by swapping one constructor.

Because we don't take a hard dependency on ``openai`` / ``anthropic``
/ ``langchain`` / ``mcp`` (they're optional, and the shims are
duck-typed against their public surface), every ``wrap_*`` helper
works against any object whose method shape matches the reference
SDKs as of the dates pinned in ``stepback/pricing.py``.

Public surface
--------------

* :func:`wrap_openai(client, recorder, *, model=None)` — wraps an
  ``openai.OpenAI`` (or ``AsyncOpenAI``-shaped) client. Calls to
  ``client.chat.completions.create(model=..., messages=..., ...)`` are
  recorded as ``llm_call`` steps. The wrapped client returns a
  :class:`OpenAIChatCompletion` namespace mirroring the SDK fields
  agents read (``id``, ``model``, ``choices``, ``usage``,
  ``choices[i].message.content`` / ``.tool_calls``,
  ``choices[i].finish_reason``).

* :func:`wrap_anthropic(client, recorder)` — wraps an ``Anthropic``
  client. Calls to ``client.messages.create(model=..., system=...,
  messages=..., max_tokens=..., ...)`` are recorded as ``llm_call``
  steps. The Anthropic response shape is *canonicalised* into the
  OpenAI ``{"choices": [...], "usage": {...}}`` shape so cost
  accounting and replay semantics are uniform across providers; the
  caller still gets an :class:`AnthropicMessage` namespace mirroring
  the SDK's native ``content[]`` / ``stop_reason`` fields.

* :func:`wrap_langchain_tool(tool, recorder)` /
  :func:`wrap_langchain_tools(tools, recorder)` — wraps a tool whose
  surface is ``.name`` (str) + ``.invoke(arguments)`` (the LangChain
  ``BaseTool`` protocol). Each ``.invoke`` call records a ``tool_call``
  step. Returns a list with the same iteration order so existing
  ``[t.name for t in tools]`` / ``ToolNode([t1, t2])`` registrations
  keep working.

* :func:`wrap_mcp_session(session, recorder)` — wraps an MCP
  ``ClientSession``. Calls to ``session.call_tool(name, arguments)``
  are recorded as ``tool_call`` steps under the prefix
  ``"mcp:"`` so tools from different MCP servers can be told apart on
  the timeline.

Replay parity
-------------

When you replay a trace recorded through these shims you can wire the
replay engine's :class:`stepback.replay.Executor` to a real client
again — :func:`openai_executor`, :func:`anthropic_executor`,
:func:`langchain_tool_executor`, :func:`mcp_tool_executor` are
matching adapters that take the real client/registry and return a
callable shaped the way :class:`Executor` expects. The same
canonicalisation runs on both sides so a cache hit is bit-identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from .recorder import Recorder


# =====================================================================
# OpenAI shim
# =====================================================================


@dataclass
class _OAIMessage:
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[dict]] = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "_OAIMessage":
        return cls(
            role=str(d.get("role", "assistant")),
            content=d.get("content"),
            tool_calls=list(d["tool_calls"]) if d.get("tool_calls") else None,
        )


@dataclass
class _OAIChoice:
    index: int
    finish_reason: Optional[str]
    message: _OAIMessage


@dataclass
class OpenAIChatCompletion:
    """Lightweight namespace mirroring ``openai.types.chat.ChatCompletion``.

    Agents reading ``resp.choices[0].message.content`` / ``.tool_calls``
    / ``resp.usage.prompt_tokens`` keep working unmodified.
    """

    id: str
    model: str
    choices: List[_OAIChoice]
    usage: dict
    raw: dict = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "OpenAIChatCompletion":
        choices = [
            _OAIChoice(
                index=int(c.get("index", i)),
                finish_reason=c.get("finish_reason"),
                message=_OAIMessage.from_dict(c.get("message", {})),
            )
            for i, c in enumerate(d.get("choices", []))
        ]
        return cls(
            id=str(d.get("id", "")),
            model=str(d.get("model", "")),
            choices=choices,
            usage=dict(d.get("usage", {})),
            raw=dict(d),
        )


def _canonicalise_openai_response(resp: Any) -> dict:
    """Reduce any OpenAI-shaped chat-completion response to a plain dict.

    Accepts the SDK's ``ChatCompletion`` pydantic model, our own
    :class:`OpenAIChatCompletion`, or already-plain dicts. The output
    is always JSON-safe and has the keys the replay engine expects:
    ``id``, ``model``, ``choices`` (each with ``index``,
    ``finish_reason``, ``message``), and ``usage``.
    """
    if isinstance(resp, OpenAIChatCompletion):
        return _canonicalise_openai_response(resp.raw or {
            "id": resp.id,
            "model": resp.model,
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
                for c in resp.choices
            ],
            "usage": dict(resp.usage),
        })
    if isinstance(resp, Mapping):
        return _strip_none(dict(resp))
    if hasattr(resp, "model_dump"):
        return _strip_none(resp.model_dump())  # type: ignore[no-any-return]
    if hasattr(resp, "to_dict"):
        return _strip_none(resp.to_dict())  # type: ignore[no-any-return]
    raise TypeError(
        f"unsupported OpenAI response type: {type(resp).__name__}; "
        "expected ChatCompletion / dict / mapping"
    )


def _strip_none(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _strip_none(v) for k, v in d.items() if v is not None}
    if isinstance(d, list):
        return [_strip_none(x) for x in d]
    return d


class _OAIChatCompletionsProxy:
    def __init__(self, real: Any, recorder: Recorder, *, default_model: Optional[str]) -> None:
        self._real = real
        self._rec = recorder
        self._default_model = default_model

    def create(self, *, messages: List[dict], model: Optional[str] = None,
               **kwargs: Any) -> OpenAIChatCompletion:
        chosen_model = model or self._default_model
        if chosen_model is None:
            raise ValueError(
                "wrap_openai: no model given and no default_model set on the wrapper"
            )

        def executor(_model: str, _messages: List[dict]) -> dict:
            resp = self._real.create(model=_model, messages=_messages, **kwargs)
            return _canonicalise_openai_response(resp)

        step = self._rec.llm_call(
            model=chosen_model,
            messages=list(messages),
            executor=executor,
            temperature=float(kwargs.get("temperature", 0.0)),
            seed=kwargs.get("seed", 42),
            tools=kwargs.get("tools"),
            response_format=kwargs.get("response_format"),
        )
        return OpenAIChatCompletion.from_dict(step["llm_response"])


class _OAIChatProxy:
    def __init__(self, real: Any, recorder: Recorder, *, default_model: Optional[str]) -> None:
        self.completions = _OAIChatCompletionsProxy(
            real.completions, recorder, default_model=default_model,
        )


@dataclass
class WrappedOpenAI:
    """Drop-in replacement for ``openai.OpenAI(...)``.

    Only ``client.chat.completions.create(...)`` is intercepted; every
    other attribute is passed through to the real client so unrelated
    surfaces (``client.embeddings``, ``client.files``, ...) keep
    working untouched.
    """

    chat: _OAIChatProxy
    _real: Any
    _rec: Recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def wrap_openai(client: Any, recorder: Recorder, *,
                default_model: Optional[str] = None) -> WrappedOpenAI:
    """Wrap a real OpenAI client so chat completions are recorded.

    Example::

        from openai import OpenAI
        from stepback import record
        from stepback.shims import wrap_openai

        with record("./trace.sb") as rec:
            client = wrap_openai(OpenAI(), rec, default_model="gpt-4o-mini-2024-07-18")
            client.chat.completions.create(messages=[...])
    """
    if not hasattr(client, "chat") or not hasattr(client.chat, "completions"):
        raise TypeError(
            "wrap_openai: client lacks .chat.completions; "
            "expected an openai.OpenAI / AsyncOpenAI-shaped object"
        )
    chat_proxy = _OAIChatProxy(client.chat, recorder, default_model=default_model)
    return WrappedOpenAI(chat=chat_proxy, _real=client, _rec=recorder)


# =====================================================================
# Anthropic shim
# =====================================================================


@dataclass
class _AnthropicContentBlock:
    type: str
    text: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[dict] = None


@dataclass
class AnthropicMessage:
    """Mirror of ``anthropic.types.Message`` enough for agent code."""

    id: str
    model: str
    role: str
    content: List[_AnthropicContentBlock]
    stop_reason: Optional[str]
    usage: dict
    raw: dict = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AnthropicMessage":
        blocks = [
            _AnthropicContentBlock(
                type=str(b.get("type", "text")),
                text=b.get("text"),
                id=b.get("id"),
                name=b.get("name"),
                input=b.get("input"),
            )
            for b in d.get("content", [])
        ]
        return cls(
            id=str(d.get("id", "")),
            model=str(d.get("model", "")),
            role=str(d.get("role", "assistant")),
            content=blocks,
            stop_reason=d.get("stop_reason"),
            usage=dict(d.get("usage", {})),
            raw=dict(d),
        )


def _anthropic_to_openai_shape(d: Mapping[str, Any]) -> dict:
    """Project an Anthropic message into the OpenAI chat-completion shape.

    Concatenates text blocks into ``message.content``; turns
    ``tool_use`` blocks into ``message.tool_calls``; maps
    ``stop_reason`` to ``finish_reason``; renames token-usage fields.
    """
    content_text = "".join(
        b.get("text", "") for b in d.get("content", []) if b.get("type") == "text"
    ) or None
    tool_calls = [
        {
            "id": b.get("id", ""),
            "type": "function",
            "function": {
                "name": b.get("name", ""),
                "arguments": b.get("input", {}),
            },
        }
        for b in d.get("content", []) if b.get("type") == "tool_use"
    ] or None

    stop_map = {
        "end_turn": "stop", "stop_sequence": "stop",
        "max_tokens": "length", "tool_use": "tool_calls",
    }
    finish_reason = stop_map.get(str(d.get("stop_reason", "")), d.get("stop_reason"))

    usage = d.get("usage", {}) or {}
    canonical_usage = {
        "prompt_tokens": int(usage.get("input_tokens", 0)),
        "completion_tokens": int(usage.get("output_tokens", 0)),
        "total_tokens": int(usage.get("input_tokens", 0))
                        + int(usage.get("output_tokens", 0)),
    }
    return _strip_none({
        "id": d.get("id"),
        "model": d.get("model"),
        "choices": [{
            "index": 0,
            "finish_reason": finish_reason,
            "message": {
                "role": d.get("role", "assistant"),
                "content": content_text,
                "tool_calls": tool_calls,
            },
        }],
        "usage": canonical_usage,
        "_anthropic": dict(d),
    })


def _coerce_anthropic_response(resp: Any) -> dict:
    if isinstance(resp, AnthropicMessage):
        return resp.raw or {
            "id": resp.id, "model": resp.model, "role": resp.role,
            "content": [vars(b) for b in resp.content],
            "stop_reason": resp.stop_reason, "usage": dict(resp.usage),
        }
    if isinstance(resp, Mapping):
        return dict(resp)
    if hasattr(resp, "model_dump"):
        return resp.model_dump()  # type: ignore[no-any-return]
    if hasattr(resp, "to_dict"):
        return resp.to_dict()  # type: ignore[no-any-return]
    raise TypeError(
        f"unsupported Anthropic response type: {type(resp).__name__}"
    )


class _AnthropicMessagesProxy:
    def __init__(self, real: Any, recorder: Recorder) -> None:
        self._real = real
        self._rec = recorder

    def create(self, *, model: str, messages: List[dict],
               system: Optional[str] = None,
               max_tokens: int = 1024,
               **kwargs: Any) -> AnthropicMessage:
        # Build the OpenAI-style message list we record on the inputs
        # side, so substitutions written against either provider land
        # in the same field.
        unified_messages: List[dict] = []
        if system:
            unified_messages.append({"role": "system", "content": system})
        unified_messages.extend(messages)

        def executor(_model: str, _messages: List[dict]) -> dict:
            anth_messages = [m for m in _messages if m.get("role") != "system"]
            anth_system = next(
                (m["content"] for m in _messages if m.get("role") == "system"), system
            )
            kwargs_clean = {k: v for k, v in kwargs.items() if k != "system"}
            resp = self._real.create(
                model=_model,
                messages=anth_messages,
                system=anth_system,
                max_tokens=max_tokens,
                **kwargs_clean,
            )
            native = _coerce_anthropic_response(resp)
            return _anthropic_to_openai_shape(native)

        step = self._rec.llm_call(
            model=model,
            messages=unified_messages,
            executor=executor,
            temperature=float(kwargs.get("temperature", 0.0)),
            seed=kwargs.get("seed", 42),
            tools=kwargs.get("tools"),
            response_format=None,
        )
        native = step["llm_response"].get("_anthropic", {})
        return AnthropicMessage.from_dict(native or {
            "id": step["llm_response"].get("id", ""),
            "model": step["llm_response"].get("model", model),
            "role": "assistant",
            "content": [{"type": "text",
                         "text": (step["llm_response"]["choices"][0]
                                  ["message"].get("content") or "")}],
            "stop_reason": step["llm_response"]["choices"][0].get("finish_reason"),
            "usage": {
                "input_tokens": step["llm_response"]["usage"].get("prompt_tokens", 0),
                "output_tokens": step["llm_response"]["usage"].get("completion_tokens", 0),
            },
        })


@dataclass
class WrappedAnthropic:
    messages: _AnthropicMessagesProxy
    _real: Any
    _rec: Recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def wrap_anthropic(client: Any, recorder: Recorder) -> WrappedAnthropic:
    """Wrap a real Anthropic client so ``messages.create`` is recorded.

    The recorded ``llm_response`` is the OpenAI-canonical shape so
    cost / cache semantics are uniform across providers; the original
    Anthropic payload is preserved under ``llm_response._anthropic``.
    """
    if not hasattr(client, "messages"):
        raise TypeError(
            "wrap_anthropic: client lacks .messages; "
            "expected an anthropic.Anthropic-shaped object"
        )
    return WrappedAnthropic(
        messages=_AnthropicMessagesProxy(client.messages, recorder),
        _real=client, _rec=recorder,
    )


# =====================================================================
# LangChain tool registry shim
# =====================================================================


@dataclass
class WrappedLangchainTool:
    """Mirror of ``langchain_core.tools.BaseTool`` enough for agent code.

    The agent calls ``tool.invoke(args)`` (or ``tool.run(args)`` on the
    legacy surface); both forward through the recorder.
    """

    name: str
    description: str
    _real: Any
    _rec: Recorder

    def invoke(self, arguments: Any, **kwargs: Any) -> Any:
        args_dict = arguments if isinstance(arguments, dict) else {"input": arguments}

        def executor(_name: str, _args: dict) -> Any:
            target = self._real.invoke if hasattr(self._real, "invoke") else self._real.run
            payload = _args if isinstance(arguments, dict) else _args.get("input")
            return target(payload, **kwargs)

        step = self._rec.tool_call(self.name, args_dict, executor=executor)
        return step["outputs"]["result"]

    # Legacy surface
    def run(self, arguments: Any, **kwargs: Any) -> Any:
        return self.invoke(arguments, **kwargs)


def wrap_langchain_tool(tool: Any, recorder: Recorder) -> WrappedLangchainTool:
    if not hasattr(tool, "name"):
        raise TypeError(
            "wrap_langchain_tool: object lacks .name; "
            "expected a LangChain BaseTool-shaped object"
        )
    if not hasattr(tool, "invoke") and not hasattr(tool, "run"):
        raise TypeError(
            "wrap_langchain_tool: object lacks .invoke / .run; "
            "expected a LangChain BaseTool-shaped object"
        )
    return WrappedLangchainTool(
        name=str(tool.name),
        description=str(getattr(tool, "description", "")),
        _real=tool, _rec=recorder,
    )


def wrap_langchain_tools(
    tools: Iterable[Any], recorder: Recorder,
) -> List[WrappedLangchainTool]:
    return [wrap_langchain_tool(t, recorder) for t in tools]


# =====================================================================
# MCP session shim
# =====================================================================


@dataclass
class WrappedMCPSession:
    """Wrap an MCP ``ClientSession`` so ``call_tool`` records.

    Tool names are namespaced with an ``mcp:`` prefix on the timeline
    so cross-server traces stay legible. Other session methods
    (``list_tools``, ``initialize``, ...) pass through to the real
    session unmodified.
    """

    _real: Any
    _rec: Recorder
    _server_name: str = "mcp"

    def call_tool(self, name: str, arguments: Optional[dict] = None) -> Any:
        args = dict(arguments or {})
        qualified = f"{self._server_name}:{name}"

        def executor(_qname: str, _args: dict) -> Any:
            return self._real.call_tool(name, _args)

        step = self._rec.tool_call(qualified, args, executor=executor)
        return step["outputs"]["result"]

    def __getattr__(self, item: str) -> Any:
        return getattr(self._real, item)


def wrap_mcp_session(session: Any, recorder: Recorder, *,
                     server_name: str = "mcp") -> WrappedMCPSession:
    if not hasattr(session, "call_tool"):
        raise TypeError(
            "wrap_mcp_session: object lacks .call_tool; "
            "expected an MCP ClientSession-shaped object"
        )
    return WrappedMCPSession(_real=session, _rec=recorder, _server_name=server_name)


# =====================================================================
# Replay-side executors
# =====================================================================


# =====================================================================
# AWS Bedrock shim (Converse API)
# =====================================================================
#
# AWS Bedrock's modern surface is the ``Converse`` API exposed by a
# ``bedrock-runtime`` client. The shape (as of the 2025-Q1 boto3
# release pinned in ``stepback/pricing.py``) is:
#
#     resp = client.converse(
#         modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
#         messages=[{"role": "user",
#                    "content": [{"text": "hi"}]}],
#         system=[{"text": "be terse"}],
#         inferenceConfig={"temperature": 0.0, "maxTokens": 1024},
#         toolConfig={...},                # optional
#     )
#     # → {"output": {"message": {"role": "assistant",
#     #                            "content": [{"text": "..."},
#     #                                        {"toolUse": {...}}]}},
#     #    "stopReason": "end_turn",
#     #    "usage": {"inputTokens": ..., "outputTokens": ...,
#     #              "totalTokens": ...},
#     #    "metrics": {"latencyMs": ...}}
#
# We canonicalise the Bedrock response into the same OpenAI chat-
# completion shape every other shim emits, so the substitution / cache
# / cost-accounting paths are uniform across providers. The native
# Bedrock payload is preserved under ``llm_response._bedrock`` for
# replay-side reconstruction.


def _bedrock_messages_to_unified(
    messages: Sequence[Mapping[str, Any]],
    system: Optional[Sequence[Mapping[str, Any]]],
) -> List[dict]:
    """Project Bedrock's ``[{role, content:[{text|toolUse|toolResult}]}]``
    onto the OpenAI ``[{role, content}]`` list used for hashing /
    substitution. Concatenates ``text`` blocks; preserves structured
    blocks under ``_bedrock_blocks`` so a substitution writer can still
    round-trip them."""
    unified: List[dict] = []
    if system:
        sys_text = "".join(b.get("text", "") for b in system if "text" in b)
        if sys_text:
            unified.append({"role": "system", "content": sys_text})
    for m in messages:
        role = m.get("role", "user")
        blocks = m.get("content") or []
        if isinstance(blocks, str):
            unified.append({"role": role, "content": blocks})
            continue
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, Mapping) and "text" in b)
        entry: dict = {"role": role, "content": text}
        # Preserve non-text blocks so a faithful round-trip is possible
        # (tool results, images, document attachments).
        non_text = [b for b in blocks if isinstance(b, Mapping) and "text" not in b]
        if non_text:
            entry["_bedrock_blocks"] = list(non_text)
        unified.append(entry)
    return unified


def _unified_to_bedrock_messages(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[List[dict], Optional[List[dict]]]:
    """Inverse of :func:`_bedrock_messages_to_unified` for replay."""
    system: Optional[List[dict]] = None
    bed_messages: List[dict] = []
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            sys_text = m.get("content") or ""
            system = [{"text": sys_text}] if sys_text else []
            continue
        content_blocks: List[dict] = []
        text = m.get("content") or ""
        if text:
            content_blocks.append({"text": text})
        for extra in m.get("_bedrock_blocks", []) or []:
            content_blocks.append(dict(extra))
        bed_messages.append({"role": role, "content": content_blocks})
    return bed_messages, system


def _bedrock_to_openai_shape(d: Mapping[str, Any]) -> dict:
    """Project a Bedrock Converse response into the OpenAI chat-
    completion shape. Preserves the native payload under
    ``_bedrock`` so :func:`bedrock_executor` can rehydrate it on
    cached replay."""
    out_msg = (d.get("output") or {}).get("message") or {}
    blocks = out_msg.get("content") or []
    content_text = "".join(
        b.get("text", "") for b in blocks if isinstance(b, Mapping) and "text" in b
    ) or None
    tool_calls = []
    for b in blocks:
        if not isinstance(b, Mapping):
            continue
        if "toolUse" in b:
            tu = b["toolUse"]
            tool_calls.append({
                "id": tu.get("toolUseId", ""),
                "type": "function",
                "function": {
                    "name": tu.get("name", ""),
                    "arguments": tu.get("input", {}),
                },
            })
    tool_calls = tool_calls or None

    stop_map = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "content_filtered": "content_filter",
        "guardrail_intervened": "content_filter",
    }
    stop_reason = str(d.get("stopReason", ""))
    finish_reason = stop_map.get(stop_reason, stop_reason or None)

    usage = d.get("usage", {}) or {}
    canonical_usage = {
        "prompt_tokens": int(usage.get("inputTokens", 0)),
        "completion_tokens": int(usage.get("outputTokens", 0)),
        "total_tokens": int(
            usage.get("totalTokens",
                      int(usage.get("inputTokens", 0)) + int(usage.get("outputTokens", 0)))
        ),
    }
    return _strip_none({
        "id": d.get("ResponseMetadata", {}).get("RequestId") or d.get("id"),
        "model": d.get("_modelId") or d.get("modelId"),
        "choices": [{
            "index": 0,
            "finish_reason": finish_reason,
            "message": {
                "role": out_msg.get("role", "assistant"),
                "content": content_text,
                "tool_calls": tool_calls,
            },
        }],
        "usage": canonical_usage,
        "_bedrock": dict(d),
    })


def _coerce_bedrock_response(resp: Any) -> dict:
    if isinstance(resp, Mapping):
        return dict(resp)
    if hasattr(resp, "model_dump"):
        return resp.model_dump()  # type: ignore[no-any-return]
    if hasattr(resp, "to_dict"):
        return resp.to_dict()  # type: ignore[no-any-return]
    raise TypeError(
        f"unsupported Bedrock response type: {type(resp).__name__}"
    )


# Map a Bedrock modelId to a stepback-canonical pricing key. Bedrock
# uses provider-prefixed ids (``anthropic.claude-3-5-sonnet-...``)
# whereas ``stepback/pricing.py`` keys claudes by their native ids
# (``claude-3-5-sonnet-20241022``). The canonicalisation lets a
# Bedrock-hosted Claude reuse the same pricing row as a native one.
_BEDROCK_PRICING_ALIAS: Dict[str, str] = {
    "anthropic.claude-3-5-sonnet-20241022-v2:0": "claude-3-5-sonnet-20241022",
    "anthropic.claude-3-5-haiku-20241022-v1:0": "claude-3-5-haiku-20241022",
    "anthropic.claude-3-7-sonnet-20250219-v1:0": "claude-3-7-sonnet-20250219",
    "anthropic.claude-sonnet-4-20250514-v1:0": "claude-sonnet-4-20250514",
    "anthropic.claude-opus-4-20250514-v1:0": "claude-opus-4-20250514",
    "anthropic.claude-haiku-4-20250514-v1:0": "claude-haiku-4-20250514",
}


def canonical_bedrock_model_id(model_id: str) -> str:
    """Return the stepback canonical pricing key for a Bedrock modelId.

    Falls back to the input string when the model is not a re-hosted
    third-party model (e.g., Bedrock-native ``meta.llama3-...``).
    """
    return _BEDROCK_PRICING_ALIAS.get(model_id, model_id)


class _BedrockConverseProxy:
    def __init__(self, real: Any, recorder: Recorder) -> None:
        self._real = real
        self._rec = recorder

    def __call__(self, *,
                 modelId: str,
                 messages: List[dict],
                 system: Optional[List[dict]] = None,
                 inferenceConfig: Optional[dict] = None,
                 toolConfig: Optional[dict] = None,
                 **kwargs: Any) -> dict:
        unified_messages = _bedrock_messages_to_unified(messages, system)
        cfg = dict(inferenceConfig or {})
        temperature = float(cfg.get("temperature", 0.0))
        max_tokens = int(cfg.get("maxTokens", 1024))
        seed = cfg.get("seed", 42)
        canonical_model = canonical_bedrock_model_id(modelId)

        def executor(_model: str, _messages: List[dict]) -> dict:
            bed_messages, bed_system = _unified_to_bedrock_messages(_messages)
            kwargs_clean = {k: v for k, v in kwargs.items()
                            if k not in ("system", "inferenceConfig", "toolConfig")}
            call_kwargs = {
                "modelId": modelId,
                "messages": bed_messages,
                "inferenceConfig": cfg or {"temperature": temperature, "maxTokens": max_tokens},
            }
            if bed_system is not None:
                call_kwargs["system"] = bed_system
            if toolConfig is not None:
                call_kwargs["toolConfig"] = toolConfig
            call_kwargs.update(kwargs_clean)
            resp = self._real.converse(**call_kwargs)
            native = _coerce_bedrock_response(resp)
            native.setdefault("_modelId", modelId)
            return _bedrock_to_openai_shape(native)

        step = self._rec.llm_call(
            model=canonical_model,
            messages=unified_messages,
            executor=executor,
            temperature=temperature,
            seed=seed,
            tools=(toolConfig.get("tools") if toolConfig else None),
            response_format=None,
        )
        # Return the native Bedrock payload so existing agent code that
        # reads ``resp["output"]["message"]["content"][0]["text"]``
        # keeps working.
        native = step["llm_response"].get("_bedrock")
        if native:
            return native
        # Synthesise a minimal native shape from the canonical fields.
        usage = step["llm_response"].get("usage", {})
        return {
            "output": {
                "message": {
                    "role": step["llm_response"]["choices"][0]["message"].get("role", "assistant"),
                    "content": [{"text": step["llm_response"]["choices"][0]["message"].get("content") or ""}],
                },
            },
            "stopReason": step["llm_response"]["choices"][0].get("finish_reason") or "end_turn",
            "usage": {
                "inputTokens": int(usage.get("prompt_tokens", 0)),
                "outputTokens": int(usage.get("completion_tokens", 0)),
                "totalTokens": int(usage.get("total_tokens", 0)),
            },
        }


@dataclass
class WrappedBedrock:
    converse: _BedrockConverseProxy
    _real: Any
    _rec: Recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def wrap_bedrock(client: Any, recorder: Recorder) -> WrappedBedrock:
    """Wrap a real ``bedrock-runtime`` client so ``converse`` is recorded.

    The wrapped object exposes ``client.converse(modelId=..., messages=...,
    system=..., inferenceConfig=..., toolConfig=...)`` and returns the
    native Bedrock response dict (so existing agent code keeps working).
    Internally each call is canonicalised into the OpenAI chat-completion
    shape used by every other provider shim, so substitutions, cache
    semantics, and cost accounting are uniform.

    Raises :class:`TypeError` if the client lacks ``.converse`` (i.e.
    isn't a Bedrock-runtime-shaped object).
    """
    if not hasattr(client, "converse"):
        raise TypeError(
            "wrap_bedrock: client lacks .converse; "
            "expected a boto3 bedrock-runtime-shaped object"
        )
    return WrappedBedrock(
        converse=_BedrockConverseProxy(client, recorder),
        _real=client, _rec=recorder,
    )


def openai_executor(client: Any) -> Callable[[str, List[dict]], dict]:
    """Adapter so a real OpenAI client can serve dirty replay steps.

    Returns a callable shaped the way :class:`stepback.replay.Executor`
    expects (``llm(model, messages) -> dict``). The response is
    canonicalised before being handed back to the engine so the step's
    outputs hash stays identical to the recorded one when the
    sub-tree converges.
    """
    def _llm(model: str, messages: List[dict]) -> dict:
        resp = client.chat.completions.create(model=model, messages=messages)
        return _canonicalise_openai_response(resp)

    return _llm


def anthropic_executor(client: Any) -> Callable[[str, List[dict]], dict]:
    """Adapter wrapping a real Anthropic client for replay."""
    def _llm(model: str, messages: List[dict]) -> dict:
        anth_messages = [m for m in messages if m.get("role") != "system"]
        system = next((m["content"] for m in messages if m.get("role") == "system"), None)
        resp = client.messages.create(
            model=model, messages=anth_messages, system=system, max_tokens=1024,
        )
        return _anthropic_to_openai_shape(_coerce_anthropic_response(resp))

    return _llm


def langchain_tool_executor(
    tools: Sequence[Any],
) -> Callable[[str, dict], Any]:
    """Build a tool dispatcher across a registry of LangChain tools."""
    by_name: Dict[str, Any] = {getattr(t, "name", ""): t for t in tools}

    def _tool(name: str, arguments: dict) -> Any:
        if name not in by_name:
            raise KeyError(f"langchain_tool_executor: no such tool {name!r}")
        target = by_name[name]
        op = target.invoke if hasattr(target, "invoke") else target.run
        return op(arguments)

    return _tool


def bedrock_executor(client: Any) -> Callable[[str, List[dict]], dict]:
    """Adapter wrapping a real Bedrock-runtime client for replay.

    Accepts the canonical ``model`` (which may be the stepback alias —
    e.g. ``claude-3-5-sonnet-20241022`` — for a Bedrock-hosted Claude)
    and translates it back into a Bedrock ``modelId`` before calling
    ``client.converse``. Falls back to the input string when no inverse
    alias is registered (e.g. native ``meta.llama3-...``).
    """
    inverse: Dict[str, str] = {v: k for k, v in _BEDROCK_PRICING_ALIAS.items()}

    def _llm(model: str, messages: List[dict]) -> dict:
        bed_messages, bed_system = _unified_to_bedrock_messages(messages)
        bedrock_id = inverse.get(model, model)
        kwargs: dict = {
            "modelId": bedrock_id,
            "messages": bed_messages,
            "inferenceConfig": {"temperature": 0.0, "maxTokens": 1024},
        }
        if bed_system is not None:
            kwargs["system"] = bed_system
        resp = client.converse(**kwargs)
        native = _coerce_bedrock_response(resp)
        native.setdefault("_modelId", bedrock_id)
        return _bedrock_to_openai_shape(native)

    return _llm


def mcp_tool_executor(
    sessions: Mapping[str, Any],
) -> Callable[[str, dict], Any]:
    """Dispatch ``mcp:server:tool`` qualified names to the right session."""
    def _tool(qualified_name: str, arguments: dict) -> Any:
        parts = qualified_name.split(":", 2)
        if len(parts) == 3 and parts[0] == "mcp":
            server, tool = parts[1], parts[2]
        elif len(parts) == 2:
            server, tool = parts[0], parts[1]
        else:
            raise ValueError(
                f"mcp_tool_executor: cannot parse qualified name {qualified_name!r}; "
                "expected 'mcp:<server>:<tool>' or '<server>:<tool>'"
            )
        if server not in sessions:
            raise KeyError(
                f"mcp_tool_executor: no session registered for server {server!r}"
            )
        return sessions[server].call_tool(tool, arguments)

    return _tool


__all__ = [
    "WrappedOpenAI",
    "WrappedAnthropic",
    "WrappedBedrock",
    "WrappedLangchainTool",
    "WrappedMCPSession",
    "OpenAIChatCompletion",
    "AnthropicMessage",
    "wrap_openai",
    "wrap_anthropic",
    "wrap_bedrock",
    "wrap_langchain_tool",
    "wrap_langchain_tools",
    "wrap_mcp_session",
    "openai_executor",
    "anthropic_executor",
    "bedrock_executor",
    "langchain_tool_executor",
    "mcp_tool_executor",
    "canonical_bedrock_model_id",
]
