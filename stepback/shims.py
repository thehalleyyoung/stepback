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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence

from .recorder import Recorder
from .seeding import SeedPolicy, SeedSupport, get_seed_policy
from .streaming import StreamedLLMResponse


# =====================================================================
# ShimContract — abstract base class for all provider shims
# =====================================================================


class ShimContract(ABC):
    """Abstract base class defining the contract every LLM provider shim must satisfy.

    Each concrete subclass represents one provider (OpenAI, Anthropic, Bedrock,
    Gemini, …).  Implementing this ABC is all that is needed to add a new provider
    because the generic recording and replay machinery in :mod:`stepback.shims`
    can drive any conforming contract.

    **Mandatory abstract methods** — every subclass *must* implement:

    * :meth:`canonical_request` — convert the provider-native request payload
      (messages, system prompt, tool declarations, …) into the unified
      ``[{"role": …, "content": …}, …]`` list stored in the trace.  All
      provider-specific keys should be passed as keyword arguments so the
      signature is forward-compatible.

    * :meth:`canonical_response` — convert a provider-native response object
      (or its already-coerced dict form) into the OpenAI
      ``chat.completions`` shape:
      ``{"id": …, "model": …, "choices": […], "usage": {…}}``.
      This shape is what the recorder hashes, caches, and hands to the
      substitution engine.

    * :meth:`make_executor` — given a live provider client, return a replay-side
      callable ``(model: str, messages: List[dict]) -> dict`` that calls the real
      API and returns a canonical response dict.  The caller (replay engine) only
      uses this when a step is dirty (cache miss); on cache hits it never fires.

    **Optional hooks** (default implementations raise ``NotImplementedError`` or
    return ``None``; subclasses override to add capability):

    * :meth:`stream_request` — streaming variant.  Override when the provider
      offers a streaming surface and you want recording to capture incremental
      chunks as a single step.  Returns an iterator/generator of chunks.

    * :meth:`async_request` — async variant.  Override when the provider's async
      client surface should be recorded.  The coroutine should return or yield
      the same types as the sync surfaces.

    * :meth:`version_probe` — inspect the installed provider SDK (e.g. via
      ``import openai; return openai.__version__``) and return the version string
      so callers can emit compatibility warnings.  Returns ``None`` when the SDK
      is not installed or the version cannot be determined.

    **Registration:**

    Built-in contracts are pre-registered in :data:`_CONTRACT_REGISTRY` under
    their :attr:`provider_name`.  Third-party contracts can be added with
    :func:`register_shim_contract`; an error is raised on duplicate names unless
    ``overwrite=True`` is passed.

    Example — adding a custom provider::

        from stepback.shims import ShimContract, register_shim_contract

        class MyProviderContract(ShimContract):
            provider_name = "myprovider"

            def canonical_request(self, **kwargs):
                ...

            def canonical_response(self, native):
                ...

            def make_executor(self, client):
                ...

        register_shim_contract(MyProviderContract())
    """

    #: Unique provider identifier.  Used as the registry key.  Subclasses
    #: *must* set this to a non-empty lowercase string, e.g. ``"openai"``.
    provider_name: ClassVar[str]

    @abstractmethod
    def canonical_request(self, **kwargs: Any) -> List[dict]:
        """Convert provider-native request fields to a unified message list.

        All provider-specific parameters are passed as keyword arguments so
        the signature is uniform across providers even though each uses a
        different request shape.

        Typical kwargs per provider:

        * OpenAI: ``messages``
        * Anthropic: ``messages``, ``system``
        * Bedrock: ``messages``, ``system``
        * Gemini: ``contents``, ``system_instruction``

        Returns a list of ``{"role": …, "content": …}`` dicts in OpenAI style.
        System prompts, if present, are prepended as ``{"role": "system", …}``.
        """

    @abstractmethod
    def canonical_response(self, native: Any) -> dict:
        """Convert a provider-native response to canonical OpenAI chat-completion shape.

        *native* may be:

        * the raw SDK response object (pydantic model, dataclass, …),
        * an already-coerced plain ``dict``, or
        * the provider-specific wrapper returned by ``wrap_*`` helpers
          (e.g. :class:`AnthropicMessage`, :class:`GeminiResponse`).

        The returned dict must contain at minimum:

        .. code-block:: python

            {
                "id": str,        # provider request/response id
                "model": str,     # canonical model id
                "choices": [
                    {
                        "index": int,
                        "finish_reason": str | None,
                        "message": {
                            "role": str,
                            "content": str | None,
                            "tool_calls": list | None,
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": int,
                    "completion_tokens": int,
                    "total_tokens": int,
                },
            }
        """

    @abstractmethod
    def make_executor(self, client: Any) -> Callable[[str, List[dict]], dict]:
        """Return a replay-side executor ``(model, messages) -> canonical_dict``.

        The executor is used only when a step is dirty (cache miss) during
        replay.  It must call the real provider API and return a dict in the
        same canonical shape as :meth:`canonical_response`.
        """

    def stream_request(
        self, client: Any, messages: List[dict], **kwargs: Any
    ) -> Any:
        """Streaming hook.  Override to support streaming recording.

        Default implementation raises :class:`NotImplementedError`.  When
        streaming is supported, implementations should yield canonical chunk
        dicts and, after the final chunk, flush a complete canonical response
        that the recorder can hash and cache.
        """
        raise NotImplementedError(
            f"{type(self).__name__} (provider={getattr(self, 'provider_name', '?')!r})"
            " does not implement streaming recording; override stream_request() to add it"
        )

    async def async_request(
        self, client: Any, messages: List[dict], **kwargs: Any
    ) -> Any:
        """Async hook.  Override to support async recording.

        Default implementation raises :class:`NotImplementedError`.  When
        async is supported, implementations should call the provider's async
        client surface and return a canonical response dict.
        """
        raise NotImplementedError(
            f"{type(self).__name__} (provider={getattr(self, 'provider_name', '?')!r})"
            " does not implement async recording; override async_request() to add it"
        )

    def version_probe(self, client: Any) -> Optional[str]:
        """Return the installed provider SDK version string, or ``None``.

        The base implementation always returns ``None``.  Subclasses override
        to try importing the SDK and reading ``__version__`` (catching all
        import/attribute errors and returning ``None`` on failure).

        This method must *never* raise; it is intended to be called as a
        best-effort diagnostic.
        """
        return None


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


def _chunk_to_dict(chunk: Any) -> dict:
    """Coerce an OpenAI streaming chunk to a plain dict."""
    if isinstance(chunk, Mapping):
        return dict(chunk)
    if hasattr(chunk, "model_dump"):
        return chunk.model_dump()  # type: ignore[no-any-return]
    if hasattr(chunk, "to_dict"):
        return chunk.to_dict()  # type: ignore[no-any-return]
    raise TypeError(
        f"unsupported streaming chunk type: {type(chunk).__name__}; "
        "expected ChatCompletionChunk / dict / mapping"
    )


def _accumulate_streaming_chunks(chunks: Any) -> dict:
    """Consume an OpenAI-compatible streaming response and assemble a canonical
    ``chat.completion``-shaped dict.

    Each chunk has ``id``, ``model``, and ``choices[].delta`` (with ``role``,
    ``content``, and ``tool_calls``). The final chunk carries a non-null
    ``finish_reason``.  Some streams emit a trailing usage chunk (when
    ``stream_options={"include_usage": True}``) with ``choices=[]`` and a
    ``usage`` key.

    Tool-call deltas are accumulated by ``(choice_index, tool_call_index)`` to
    handle parallel tool calls correctly: ``function.name`` arrives in the
    first delta for an index; ``function.arguments`` is concatenated over all
    deltas for that index.
    """
    msg_id: Optional[str] = None
    msg_model: Optional[str] = None
    role = "assistant"
    content_parts: List[str] = []
    # key: (choice_index, tc_index) → {id, type, name, args_parts}
    tc_bufs: Dict[tuple, dict] = {}
    finish_reason: Optional[str] = None
    usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for raw_chunk in chunks:
        chunk = _chunk_to_dict(raw_chunk)
        if msg_id is None:
            msg_id = chunk.get("id")
        if msg_model is None:
            msg_model = chunk.get("model")
        # Usage may arrive in the final synthetic chunk with choices=[].
        if chunk.get("usage"):
            u = chunk["usage"]
            usage = {
                "prompt_tokens": int(u.get("prompt_tokens", 0)),
                "completion_tokens": int(u.get("completion_tokens", 0)),
                "total_tokens": int(u.get("total_tokens", 0)),
            }
        for choice in chunk.get("choices") or []:
            ci = int(choice.get("index", 0))
            delta = choice.get("delta") or {}
            if delta.get("role"):
                role = delta["role"]
            if delta.get("content"):
                content_parts.append(delta["content"])
            for tc_delta in delta.get("tool_calls") or []:
                ti = int(tc_delta.get("index", 0))
                key = (ci, ti)
                if key not in tc_bufs:
                    tc_bufs[key] = {
                        "id": tc_delta.get("id", ""),
                        "type": tc_delta.get("type", "function"),
                        "name": "",
                        "args_parts": [],
                    }
                else:
                    # id and type may arrive only on the first delta
                    if tc_delta.get("id"):
                        tc_bufs[key]["id"] = tc_delta["id"]
                    if tc_delta.get("type"):
                        tc_bufs[key]["type"] = tc_delta["type"]
                fn = tc_delta.get("function") or {}
                if fn.get("name"):
                    tc_bufs[key]["name"] += fn["name"]
                if fn.get("arguments") is not None:
                    tc_bufs[key]["args_parts"].append(fn["arguments"])
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    content = "".join(content_parts) or None
    tool_calls: Optional[List[dict]] = None
    if tc_bufs:
        tool_calls = [
            {
                "id": buf["id"],
                "type": buf["type"],
                "function": {
                    "name": buf["name"],
                    "arguments": "".join(buf["args_parts"]),
                },
            }
            for (_, _ti), buf in sorted(tc_bufs.items())
        ]
    return {
        "id": msg_id,
        "model": msg_model,
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "finish_reason": finish_reason,
            "message": {
                "role": role,
                "content": content,
                "tool_calls": tool_calls,
            },
        }],
        "usage": usage,
    }


class _OAIChatCompletionsProxy:
    def __init__(self, real: Any, recorder: Recorder, *, default_model: Optional[str],
                 seed_policy: Optional[SeedPolicy] = None,
                 contract: Optional["ShimContract"] = None) -> None:
        self._real = real
        self._rec = recorder
        self._default_model = default_model
        self._seed_policy = seed_policy or get_seed_policy()
        self._contract = contract

    def _get_contract(self) -> "ShimContract":
        if self._contract is not None:
            return self._contract
        return shim_contract_for("openai")

    def create(self, *, messages: List[dict], model: Optional[str] = None,
               **kwargs: Any) -> OpenAIChatCompletion:
        chosen_model = model or self._default_model
        if chosen_model is None:
            raise ValueError(
                "wrap_openai: no model given and no default_model set on the wrapper"
            )
        temperature = float(kwargs.get("temperature", 0.0))
        caller_seed = kwargs.get("seed")
        effective_seed = self._seed_policy.check(
            "openai", caller_seed, temperature, model=chosen_model
        )
        # Merge effective seed back into kwargs so the real API receives it.
        api_kwargs = dict(kwargs)
        if effective_seed is not None:
            api_kwargs["seed"] = effective_seed

        contract = self._get_contract()
        is_stream = bool(api_kwargs.get("stream"))

        if is_stream:
            # Streaming: eagerly consume all chunks, buffer them so the caller
            # can replay them for progressive display, then record as one step.
            _raw_chunks: List[Any] = []

            def executor(_model: str, _messages: List[dict]) -> dict:
                stream = self._real.create(model=_model, messages=_messages, **api_kwargs)
                for chunk in stream:
                    _raw_chunks.append(chunk)
                assembled = _accumulate_streaming_chunks(iter(_raw_chunks))
                return contract.canonical_response(assembled)
        else:
            def executor(_model: str, _messages: List[dict]) -> dict:  # type: ignore[no-redef]
                resp = self._real.create(model=_model, messages=_messages, **api_kwargs)
                return contract.canonical_response(resp)

        step = self._rec.llm_call(
            model=chosen_model,
            messages=contract.canonical_request(messages=list(messages)),
            executor=executor,
            temperature=temperature,
            seed=effective_seed,
            tools=kwargs.get("tools"),
            response_format=kwargs.get("response_format"),
        )
        if is_stream:
            native = OpenAIChatCompletion.from_dict(step["llm_response"])
            return StreamedLLMResponse(  # type: ignore[return-value]
                _raw_chunks, step["llm_response"], step, native=native
            )
        return OpenAIChatCompletion.from_dict(step["llm_response"])


class _OAIChatProxy:
    def __init__(self, real: Any, recorder: Recorder, *, default_model: Optional[str],
                 seed_policy: Optional[SeedPolicy] = None,
                 contract: Optional["ShimContract"] = None) -> None:
        self.completions = _OAIChatCompletionsProxy(
            real.completions, recorder, default_model=default_model,
            seed_policy=seed_policy, contract=contract,
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
                default_model: Optional[str] = None,
                seed_policy: Optional[SeedPolicy] = None,
                contract: Optional["ShimContract"] = None) -> WrappedOpenAI:
    """Wrap a real OpenAI client so chat completions are recorded.

    The *seed_policy* controls the default seed applied to every call and
    the warning emitted when seed support is missing.  Defaults to the
    module-level :data:`~stepback.seeding.DEFAULT_SEED_POLICY`.

    The optional *contract* parameter accepts a :class:`ShimContract` instance
    to override the built-in :class:`OpenAIShimContract`.  Useful for testing
    or for providers that share the OpenAI wire format.

    **Streaming:** When ``stream=True`` is passed to
    ``client.chat.completions.create``, the shim transparently accumulates
    all chunks into a final ``chat.completion``-shaped response before
    recording. The caller receives an :class:`OpenAIChatCompletion` (not a
    generator). Replay always returns the final assembled response, which
    is the correct contract for a replay-caching system.

    **OpenAI-compatible endpoints:** Any client whose ``chat.completions.create``
    returns the OpenAI ``chat.completion`` wire format (e.g. Together AI,
    Fireworks, local vLLM, Ollama) is compatible; pass it directly to
    ``wrap_openai``.

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
    chat_proxy = _OAIChatProxy(client.chat, recorder, default_model=default_model,
                               seed_policy=seed_policy, contract=contract)
    return WrappedOpenAI(chat=chat_proxy, _real=client, _rec=recorder)


# =====================================================================
# Async OpenAI client wrapper
# =====================================================================


class _AsyncOAIChatCompletionsProxy:
    """Async counterpart to :class:`_OAIChatCompletionsProxy`.

    ``create(...)`` is an *async* method: it eagerly awaits the real
    ``AsyncOpenAI`` client call (no lazy executor), then records the
    obtained response synchronously through the recorder.  On replay,
    the synchronous engine serves from cache as normal.
    """

    def __init__(self, real: Any, recorder: Recorder, *, default_model: Optional[str],
                 seed_policy: Optional[SeedPolicy] = None,
                 contract: Optional["ShimContract"] = None) -> None:
        self._real = real
        self._rec = recorder
        self._default_model = default_model
        self._seed_policy = seed_policy or get_seed_policy()
        self._contract = contract

    def _get_contract(self) -> "ShimContract":
        if self._contract is not None:
            return self._contract
        return shim_contract_for("openai")

    async def create(self, *, messages: List[dict], model: Optional[str] = None,
                     **kwargs: Any) -> OpenAIChatCompletion:
        """Await the real async API call, then record the response."""
        chosen_model = model or self._default_model
        if chosen_model is None:
            raise ValueError(
                "wrap_openai_async: no model given and no default_model set on the wrapper"
            )
        temperature = float(kwargs.get("temperature", 0.0))
        caller_seed = kwargs.get("seed")
        effective_seed = self._seed_policy.check(
            "openai", caller_seed, temperature, model=chosen_model
        )
        api_kwargs = dict(kwargs)
        if effective_seed is not None:
            api_kwargs["seed"] = effective_seed

        contract = self._get_contract()

        # Eagerly await the real API — record it synchronously afterward.
        resp = await self._real.create(model=chosen_model, messages=messages, **api_kwargs)
        canonical = contract.canonical_response(resp)

        # Provide a sync executor that returns the pre-obtained response so
        # the recorder can use it directly (cache miss path on first record,
        # ignored on replay cache hits).
        def _sync_executor(_m: str, _msgs: List[dict]) -> dict:
            return canonical

        step = self._rec.llm_call(
            model=chosen_model,
            messages=contract.canonical_request(messages=list(messages)),
            executor=_sync_executor,
            temperature=temperature,
            seed=effective_seed,
            tools=kwargs.get("tools"),
            response_format=kwargs.get("response_format"),
        )
        return OpenAIChatCompletion.from_dict(step["llm_response"])


class _AsyncOAIChatProxy:
    def __init__(self, real: Any, recorder: Recorder, *, default_model: Optional[str],
                 seed_policy: Optional[SeedPolicy] = None,
                 contract: Optional["ShimContract"] = None) -> None:
        self.completions = _AsyncOAIChatCompletionsProxy(
            real.completions, recorder, default_model=default_model,
            seed_policy=seed_policy, contract=contract,
        )


@dataclass
class AsyncWrappedOpenAI:
    """Async drop-in replacement for ``openai.AsyncOpenAI(...)``.

    ``client.chat.completions.create(...)`` is an ``async def`` that
    awaits the real async client, then records the response.  All other
    attributes are passed through to the real client.
    """

    chat: _AsyncOAIChatProxy
    _real: Any
    _rec: Recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def wrap_openai_async(client: Any, recorder: Recorder, *,
                      default_model: Optional[str] = None,
                      seed_policy: Optional[SeedPolicy] = None,
                      contract: Optional["ShimContract"] = None) -> AsyncWrappedOpenAI:
    """Wrap a real ``openai.AsyncOpenAI`` client so async chat completions
    are recorded.

    Usage::

        from openai import AsyncOpenAI
        from stepback import record
        from stepback.shims import wrap_openai_async

        async def my_agent():
            with record("./trace.sb") as rec:
                client = wrap_openai_async(AsyncOpenAI(), rec,
                                           default_model="gpt-4o-mini-2024-07-18")
                resp = await client.chat.completions.create(messages=[...])

    Replay is always synchronous (the replay engine is not async-aware);
    async recording simply collapses the awaited response into the trace.

    Raises :class:`TypeError` if *client* lacks ``.chat.completions``.
    """
    if not hasattr(client, "chat") or not hasattr(client.chat, "completions"):
        raise TypeError(
            "wrap_openai_async: client lacks .chat.completions; "
            "expected an openai.AsyncOpenAI-shaped object"
        )
    chat_proxy = _AsyncOAIChatProxy(client.chat, recorder, default_model=default_model,
                                    seed_policy=seed_policy, contract=contract)
    return AsyncWrappedOpenAI(chat=chat_proxy, _real=client, _rec=recorder)


# =====================================================================
# OpenAI Responses API shim  (client.responses.create)
# =====================================================================
#
# The OpenAI Responses API (introduced 2025) is at /v1/responses and
# differs from Chat Completions:
#
#   resp = client.responses.create(
#       model="gpt-4o",
#       input=[{"role": "user", "content": "hi"}],
#       tools=[...],   # optional
#   )
#   # → {
#   #     "id": "resp_...",
#   #     "object": "response",
#   #     "model": "gpt-4o-mini-2024-07-18",
#   #     "output": [
#   #       {"type": "message", "id": "msg_...", "role": "assistant",
#   #        "content": [{"type": "text", "text": "Hi."}]},
#   #     ],
#   #     "usage": {"input_tokens": N, "output_tokens": N, "total_tokens": N}
#   #   }
#
# Tool-call output items:
#   {"type": "function_call", "id": "call_...", "name": "fn", "arguments": "{}"}
#
# We project the Responses API output onto the canonical OpenAI
# chat-completion shape so all replay / substitution / cost machinery
# remains uniform across both API surfaces.


def _canonicalise_openai_responses_output(resp: Any) -> dict:
    """Project an OpenAI Responses API response to the canonical
    ``chat.completion`` shape.

    Accepts the SDK response object, an already-plain dict, or any
    object with ``.model_dump()`` / ``.to_dict()``.

    The ``output`` array may contain:

    * ``{"type": "message", "role": ..., "content": [{"type": "text", "text": ...}]}``
    * ``{"type": "function_call", "id": ..., "name": ..., "arguments": ...}``
    * ``{"type": "reasoning", ...}`` (ignored, no canonical equivalent)
    """
    if hasattr(resp, "model_dump"):
        d = resp.model_dump()
    elif hasattr(resp, "to_dict"):
        d = resp.to_dict()
    elif isinstance(resp, Mapping):
        d = dict(resp)
    else:
        raise TypeError(
            f"unsupported Responses API response type: {type(resp).__name__}"
        )

    output_items = d.get("output") or []
    content_parts: List[str] = []
    tool_calls: List[dict] = []
    finish_reason: Optional[str] = None

    for item in output_items:
        if not isinstance(item, Mapping):
            try:
                item = dict(vars(item))
            except Exception:
                continue
        item_type = item.get("type", "")
        if item_type == "message":
            # role is usually "assistant"
            for block in item.get("content") or []:
                if not isinstance(block, Mapping):
                    continue
                if block.get("type") == "text":
                    content_parts.append(block.get("text") or "")
            finish_reason = finish_reason or "stop"
        elif item_type == "function_call":
            tool_calls.append({
                "id": item.get("id", ""),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}"),
                },
            })
            finish_reason = "tool_calls"

    # Status → finish_reason overrides
    status = d.get("status", "")
    if status == "incomplete":
        finish_reason = "length"
    elif status == "failed":
        finish_reason = "error"

    content_text = "".join(content_parts) or None
    tool_calls_out = tool_calls or None

    # usage: Responses API uses input_tokens/output_tokens
    usage_d = d.get("usage") or {}
    canonical_usage = {
        "prompt_tokens": int(usage_d.get("input_tokens", usage_d.get("prompt_tokens", 0))),
        "completion_tokens": int(usage_d.get("output_tokens", usage_d.get("completion_tokens", 0))),
        "total_tokens": int(usage_d.get("total_tokens", 0)),
    }
    if not canonical_usage["total_tokens"]:
        canonical_usage["total_tokens"] = (
            canonical_usage["prompt_tokens"] + canonical_usage["completion_tokens"]
        )

    return _strip_none({
        "id": d.get("id"),
        "model": d.get("model"),
        "choices": [{
            "index": 0,
            "finish_reason": finish_reason,
            "message": {
                "role": "assistant",
                "content": content_text,
                "tool_calls": tool_calls_out,
            },
        }],
        "usage": canonical_usage,
        "_openai_responses": dict(d),
    })


@dataclass
class OpenAIResponsesOutput:
    """Lightweight namespace mirroring the OpenAI Responses API output.

    Agents reading ``resp.output_text`` / ``resp.output[i].content[j].text``
    keep working.  The wrapped client returns this object.
    """

    id: str
    model: str
    output: List[dict]
    usage: dict
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def output_text(self) -> Optional[str]:
        """Concatenated text from all message output items."""
        parts: List[str] = []
        for item in self.output:
            if not isinstance(item, Mapping):
                continue
            if item.get("type") == "message":
                for block in item.get("content") or []:
                    if isinstance(block, Mapping) and block.get("type") == "text":
                        parts.append(block.get("text") or "")
        return "".join(parts) or None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "OpenAIResponsesOutput":
        return cls(
            id=str(d.get("id", "")),
            model=str(d.get("model", "")),
            output=list(d.get("output") or []),
            usage=dict(d.get("usage") or {}),
            raw=dict(d),
        )


class _OAIResponsesProxy:
    def __init__(self, real: Any, recorder: Recorder, *,
                 default_model: Optional[str],
                 seed_policy: Optional[SeedPolicy] = None) -> None:
        self._real = real
        self._rec = recorder
        self._default_model = default_model
        self._seed_policy = seed_policy or get_seed_policy()

    def create(self, *, input: Any, model: Optional[str] = None,  # noqa: A002
               **kwargs: Any) -> OpenAIResponsesOutput:
        """Record an OpenAI Responses API call as a ``llm_call`` step."""
        chosen_model = model or self._default_model
        if chosen_model is None:
            raise ValueError(
                "wrap_openai_responses: no model given and no default_model set"
            )
        # Normalise ``input`` to a list of messages in OpenAI style.
        if isinstance(input, str):
            messages: List[dict] = [{"role": "user", "content": input}]
        elif isinstance(input, list):
            messages = list(input)
        else:
            messages = [{"role": "user", "content": str(input)}]

        temperature = float(kwargs.get("temperature", 0.0))
        caller_seed = kwargs.get("seed")
        effective_seed = self._seed_policy.check(
            "openai", caller_seed, temperature, model=chosen_model
        )
        api_kwargs = dict(kwargs)
        if effective_seed is not None:
            api_kwargs["seed"] = effective_seed

        def executor(_model: str, _msgs: List[dict]) -> dict:
            resp = self._real.create(model=_model, input=_msgs, **api_kwargs)
            return _canonicalise_openai_responses_output(resp)

        step = self._rec.llm_call(
            model=chosen_model,
            messages=messages,
            executor=executor,
            temperature=temperature,
            seed=effective_seed,
            tools=kwargs.get("tools"),
            response_format=kwargs.get("text", {}).get("format") if isinstance(kwargs.get("text"), Mapping) else None,
        )
        # Return the native Responses API shape.
        native = step["llm_response"].get("_openai_responses")
        if native:
            return OpenAIResponsesOutput.from_dict(native)
        # Synthesise minimal native shape from canonical fields.
        canonical = step["llm_response"]
        ch0 = (canonical.get("choices") or [{}])[0]
        msg = ch0.get("message") or {}
        output_items: List[dict] = []
        if msg.get("content"):
            output_items.append({
                "type": "message",
                "id": f"msg_{canonical.get('id', '')}",
                "role": "assistant",
                "content": [{"type": "text", "text": msg["content"]}],
            })
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            output_items.append({
                "type": "function_call",
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments", "{}"),
            })
        usage = canonical.get("usage") or {}
        synth: dict = {
            "id": canonical.get("id", ""),
            "model": canonical.get("model", chosen_model),
            "output": output_items,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }
        return OpenAIResponsesOutput.from_dict(synth)


def wrap_openai_responses(client: Any, recorder: Recorder, *,
                          default_model: Optional[str] = None,
                          seed_policy: Optional[SeedPolicy] = None) -> "_WrappedOpenAIResponses":
    """Wrap an OpenAI client so ``responses.create(...)`` calls are recorded.

    The OpenAI Responses API (``/v1/responses``) uses a different request
    shape from Chat Completions: ``input`` (list of messages or a string)
    instead of ``messages``, and a different output format.  This wrapper
    records each call as a ``llm_call`` step using the same canonical
    ``chat.completion`` shape, so substitutions and replay work identically
    to the Chat Completions path.

    Usage::

        from stepback.shims import wrap_openai_responses
        with record("./trace.sb") as rec:
            client = wrap_openai_responses(openai.OpenAI(), rec,
                                           default_model="gpt-4o")
            resp = client.responses.create(
                input=[{"role": "user", "content": "hi"}]
            )
            print(resp.output_text)

    Raises :class:`TypeError` if *client* lacks ``.responses.create``.
    """
    if not hasattr(client, "responses") or not hasattr(client.responses, "create"):
        raise TypeError(
            "wrap_openai_responses: client lacks .responses.create; "
            "expected an openai.OpenAI-shaped object (SDK ≥ 1.66)"
        )
    proxy = _OAIResponsesProxy(client.responses, recorder,
                               default_model=default_model,
                               seed_policy=seed_policy)
    return _WrappedOpenAIResponses(responses=proxy, _real=client, _rec=recorder)


@dataclass
class _WrappedOpenAIResponses:
    """Thin wrapper exposing only ``client.responses.create(...)`` for recording."""

    responses: _OAIResponsesProxy
    _real: Any
    _rec: Recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


# =====================================================================
# Azure OpenAI shim
# =====================================================================
# Azure OpenAI uses the same Chat Completions wire format as OpenAI but
# addresses models by *deployment name* (a user-defined string) rather than a
# versioned model id such as ``"gpt-4o-2024-11-20"``.  The deployment name
# is passed as the ``model`` parameter to ``chat.completions.create`` against
# an ``openai.AzureOpenAI``-shaped client.
#
# Key design decisions
# --------------------
# 1. The step's canonical ``model`` field is ``"azure:{deployment_name}"``
#    (or ``canonical_azure_model_id(…, underlying_model=…)`` when an
#    underlying model is declared).  Using the raw deployment name as part of
#    the key prevents cross-deployment cache collisions even when two
#    deployments serve the same underlying model.
# 2. The regional *endpoint* URL is stored as display metadata on the wrapper
#    object; it is NOT hashed into the step so that migrating a trace across
#    regions does not invalidate the cache.
# 3. For replay the executor must receive the *deployment name*, not the
#    canonical model id, because that is what Azure's API requires.
#    ``azure_openai_executor(client, deployment_name=…)`` accepts it
#    explicitly.


def canonical_azure_model_id(
    deployment_name: str,
    *,
    underlying_model: Optional[str] = None,
) -> str:
    """Return the stepback canonical model id for an Azure OpenAI deployment.

    Azure deployment names are user-defined (e.g. ``"prod-gpt4o"``) and do
    not appear in the global pricing catalog.  When *underlying_model* is a
    recognisable OpenAI model id (e.g. ``"gpt-4o-2024-11-20"``), that
    canonical id is returned so cost accounting works correctly.  Otherwise
    ``"azure:{deployment_name}"`` is returned — the step is still recorded,
    but with zero-dollar pricing.

    Note: when *underlying_model* resolves to a canonical OpenAI id the
    returned string does **not** encode the deployment name.  Two deployments
    backed by the same underlying model therefore share a replay cache by
    default, which is the desired behaviour (identical inputs to the same
    model produce the same output regardless of deployment name).  To force
    strict deployment-level cache isolation, omit *underlying_model* so the
    deployment-scoped ``"azure:{deployment_name}"`` form is used instead.
    """
    from .pricing import resolve_model
    if underlying_model:
        resolved = resolve_model(underlying_model)
        if resolved:
            return resolved
    return f"azure:{deployment_name}"


class _AzureOAIChatCompletionsProxy:
    """Proxy for ``AzureOpenAI().chat.completions`` that records every call.

    The ``model`` parameter to :meth:`create` is the **deployment name**, not
    an OpenAI model id — this matches the Azure OpenAI SDK convention.
    ``default_deployment`` on the proxy serves as the fallback when ``model``
    is omitted from a ``create()`` call.
    """

    def __init__(
        self,
        real: Any,
        recorder: Recorder,
        *,
        default_deployment: Optional[str] = None,
        underlying_model: Optional[str] = None,
        endpoint: Optional[str] = None,
        seed_policy: Optional[SeedPolicy] = None,
        contract: Optional["ShimContract"] = None,
    ) -> None:
        self._real = real
        self._rec = recorder
        self._default_deployment = default_deployment
        self._underlying_model = underlying_model
        self._endpoint = endpoint
        self._seed_policy = seed_policy or get_seed_policy()
        self._contract = contract

    def _get_contract(self) -> "ShimContract":
        if self._contract is not None:
            return self._contract
        return shim_contract_for("azure_openai")

    def create(
        self,
        *,
        messages: List[dict],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> OpenAIChatCompletion:
        """Record a chat-completion call to an Azure OpenAI deployment.

        *model* is the Azure *deployment name* (following the Azure SDK
        convention).  Falls back to ``default_deployment`` when omitted.
        If *model* differs from ``default_deployment`` the step is still
        recorded, but with ``"azure:{model}"`` as the canonical id (no
        pricing) because the underlying model for the override deployment
        is unknown.
        """
        effective_deployment = model or self._default_deployment
        if effective_deployment is None:
            raise ValueError(
                "wrap_azure_openai: no deployment name given and no "
                "default_deployment set on the wrapper"
            )
        # Use underlying_model only when the caller is using the deployment
        # that was declared at wrap time; for ad-hoc overrides we cannot
        # assume the same underlying model applies.
        underlying = (
            self._underlying_model
            if effective_deployment == self._default_deployment
            else None
        )
        canonical_model = canonical_azure_model_id(
            effective_deployment, underlying_model=underlying
        )
        temperature = float(kwargs.get("temperature", 0.0))
        caller_seed = kwargs.get("seed")
        effective_seed = self._seed_policy.check(
            "azure_openai", caller_seed, temperature, model=effective_deployment
        )
        api_kwargs = dict(kwargs)
        if effective_seed is not None:
            api_kwargs["seed"] = effective_seed

        contract = self._get_contract()
        deployment = effective_deployment
        is_stream = bool(api_kwargs.get("stream"))

        if is_stream:
            _raw_chunks: List[Any] = []

            def executor(_m: str, _msgs: List[dict]) -> dict:
                stream = self._real.create(
                    model=deployment, messages=_msgs, **api_kwargs
                )
                for chunk in stream:
                    _raw_chunks.append(chunk)
                assembled = _accumulate_streaming_chunks(iter(_raw_chunks))
                return contract.canonical_response(assembled)
        else:
            def executor(_m: str, _msgs: List[dict]) -> dict:  # type: ignore[no-redef]
                resp = self._real.create(
                    model=deployment, messages=_msgs, **api_kwargs
                )
                return contract.canonical_response(resp)

        step = self._rec.llm_call(
            model=canonical_model,
            messages=contract.canonical_request(messages=list(messages)),
            executor=executor,
            temperature=temperature,
            seed=effective_seed,
            tools=kwargs.get("tools"),
            response_format=kwargs.get("response_format"),
        )
        if is_stream:
            native = OpenAIChatCompletion.from_dict(step["llm_response"])
            return StreamedLLMResponse(  # type: ignore[return-value]
                _raw_chunks, step["llm_response"], step, native=native
            )
        return OpenAIChatCompletion.from_dict(step["llm_response"])


@dataclass
class WrappedAzureOpenAI:
    """Drop-in replacement for ``openai.AzureOpenAI(...)``.

    Only ``client.chat.completions.create(model=<deployment>, ...)`` is
    intercepted; every other attribute is passed through to the real client.
    ``endpoint`` and ``deployment_name`` are exposed as read-only properties
    for diagnostics and tooling.
    """

    chat: "_AzureOAIChatProxy"
    _real: Any
    _rec: Recorder
    _endpoint: Optional[str]
    _deployment_name: Optional[str]

    @property
    def endpoint(self) -> Optional[str]:
        """The Azure endpoint URL this wrapper targets (display metadata only)."""
        return self._endpoint

    @property
    def deployment_name(self) -> Optional[str]:
        """The default deployment name used when ``create(model=…)`` is omitted."""
        return self._deployment_name

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


@dataclass
class _AzureOAIChatProxy:
    completions: _AzureOAIChatCompletionsProxy


def wrap_azure_openai(
    client: Any,
    recorder: Recorder,
    *,
    default_deployment: Optional[str] = None,
    underlying_model: Optional[str] = None,
    endpoint: Optional[str] = None,
    seed_policy: Optional[SeedPolicy] = None,
    contract: Optional["ShimContract"] = None,
) -> WrappedAzureOpenAI:
    """Wrap a real ``openai.AzureOpenAI`` client so chat completions are recorded.

    Azure OpenAI addresses models by *deployment name* rather than a versioned
    model id.  Pass the deployment name as ``model=`` in each
    ``chat.completions.create(...)`` call (matching the Azure SDK convention),
    or supply ``default_deployment`` here as a fallback.

    Parameters
    ----------
    client:
        An ``openai.AzureOpenAI``-shaped object — any client whose
        ``chat.completions.create(model=<deployment>, messages=…)`` returns
        the OpenAI ``chat.completion`` wire format.
    recorder:
        The active :class:`~stepback.recorder.Recorder` context.
    default_deployment:
        Default Azure deployment name used when ``create()`` is called
        without an explicit ``model=``.
    underlying_model:
        Optional canonical model id for the *default* deployment
        (e.g. ``"gpt-4o-2024-11-20"``).  When supplied and resolvable,
        cost accounting uses this id instead of returning zero.  Only
        applied when the caller uses the default deployment; per-call
        deployment overrides do not inherit this mapping.
    endpoint:
        The Azure endpoint URL (e.g.
        ``"https://my-resource.openai.azure.com/"``).  Stored as metadata
        on the returned wrapper; **not** hashed into the step so that
        regional migrations do not invalidate the cache.
    seed_policy:
        Controls seed application and warnings.  Defaults to the
        module-level :data:`~stepback.seeding.DEFAULT_SEED_POLICY`.
    contract:
        Optional :class:`ShimContract` override.  Defaults to the
        registered ``"azure_openai"`` contract which uses the same
        canonicalisers as the OpenAI shim.

    Returns
    -------
    :class:`WrappedAzureOpenAI`
        A proxy client whose ``.chat.completions.create(...)`` is
        instrumented.  All other attributes are forwarded to the real
        client.

    Raises
    ------
    TypeError
        If *client* lacks a ``.chat.completions`` interface.

    Example
    -------
    ::

        from openai import AzureOpenAI
        from stepback import record
        from stepback.shims import wrap_azure_openai

        with record("./trace.sb") as rec:
            client = wrap_azure_openai(
                AzureOpenAI(
                    azure_endpoint="https://my-res.openai.azure.com/",
                    api_version="2024-02-01",
                ),
                rec,
                default_deployment="my-gpt4o-deployment",
                underlying_model="gpt-4o-2024-11-20",
                endpoint="https://my-res.openai.azure.com/",
            )
            resp = client.chat.completions.create(
                model="my-gpt4o-deployment",
                messages=[{"role": "user", "content": "Hello"}],
            )
            print(resp.choices[0].message.content)
    """
    if not hasattr(client, "chat") or not hasattr(client.chat, "completions"):
        raise TypeError(
            "wrap_azure_openai: client lacks .chat.completions; "
            "expected an openai.AzureOpenAI-shaped object"
        )
    completions_proxy = _AzureOAIChatCompletionsProxy(
        client.chat.completions,
        recorder,
        default_deployment=default_deployment,
        underlying_model=underlying_model,
        endpoint=endpoint,
        seed_policy=seed_policy,
        contract=contract,
    )
    chat_proxy = _AzureOAIChatProxy(completions=completions_proxy)
    return WrappedAzureOpenAI(
        chat=chat_proxy,
        _real=client,
        _rec=recorder,
        _endpoint=endpoint,
        _deployment_name=default_deployment,
    )


def azure_openai_executor(
    client: Any,
    *,
    deployment_name: str,
) -> Callable[[str, List[dict]], dict]:
    """Adapter so a real ``openai.AzureOpenAI`` client can serve dirty replay steps.

    Returns a callable shaped the way :class:`stepback.replay.Executor` expects
    (``llm(model, messages) -> dict``).  The *deployment_name* must match the
    one used during recording so the same Azure deployment is called on replay.
    The *model* argument received from the replay engine (the canonical model id
    stored in the trace) is **ignored**; only ``deployment_name`` is used in the
    actual API call.

    Parameters
    ----------
    client:
        An ``openai.AzureOpenAI``-shaped client targeting the same resource
        and API version as the one used during recording.
    deployment_name:
        The Azure deployment name to pass as ``model=`` in
        ``chat.completions.create``.

    Example
    -------
    ::

        from openai import AzureOpenAI
        from stepback.shims import azure_openai_executor
        from stepback.replay import Executor

        client = AzureOpenAI(azure_endpoint=…, api_version=…)
        executor = Executor(llm=azure_openai_executor(client, deployment_name="my-gpt4o"))
        trace.replay_forward(executor)
    """
    def _llm(_model: str, messages: List[dict]) -> dict:
        resp = client.chat.completions.create(model=deployment_name, messages=messages)
        return _canonicalise_openai_response(resp)

    return _llm


class AzureOpenAIShimContract(ShimContract):
    """:class:`ShimContract` for ``openai.AzureOpenAI``-shaped clients.

    Azure OpenAI uses the same Chat Completions wire format as OpenAI, so
    the canonicalisers are identical.  The only difference is that ``model``
    in the request refers to a deployment name rather than a versioned model id;
    that distinction is handled at the :func:`wrap_azure_openai` / executor
    layer, not here.
    """

    provider_name: ClassVar[str] = "azure_openai"

    def canonical_request(self, *, messages: List[dict], **kwargs: Any) -> List[dict]:
        """Pass messages through unchanged (same format as OpenAI)."""
        return list(messages)

    def canonical_response(self, response: Any) -> dict:
        """Coerce an Azure OpenAI response to the canonical OpenAI shape."""
        return _canonicalise_openai_response(response)

    def make_executor(self, client: Any) -> Callable:
        """Not supported: Azure executors require an explicit *deployment_name*.

        Use :func:`azure_openai_executor(client, deployment_name=…)` directly.
        """
        raise NotImplementedError(
            "AzureOpenAIShimContract.make_executor() cannot construct an executor "
            "without a deployment name.  Use "
            "azure_openai_executor(client, deployment_name='…') instead."
        )


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


def _get_event_attr(event: Any, attr: str, default: Any = None) -> Any:
    """Retrieve *attr* from an event object or dict."""
    if isinstance(event, Mapping):
        return event.get(attr, default)
    return getattr(event, attr, default)


def _accumulate_anthropic_streaming_chunks(chunks: Any) -> dict:
    """Consume an Anthropic-compatible streaming response and assemble a
    canonical Anthropic message dict.

    Handles the Anthropic SSE event stream as returned by
    ``client.messages.create(stream=True)``.  Each event has a ``type``
    attribute (or key) and additional fields depending on the type:

    * ``message_start`` — ``message`` field carries ``id``, ``model``,
      ``usage.input_tokens``.
    * ``content_block_start`` — ``index`` + ``content_block`` carrying
      ``type`` (``"text"``, ``"tool_use"``, ``"thinking"``), ``id``, ``name``.
    * ``content_block_delta`` — ``index`` + ``delta`` carrying ``type``
      (``"text_delta"``, ``"input_json_delta"``, ``"thinking_delta"``) and
      the incremental payload.
    * ``message_delta`` — ``delta`` carrying ``stop_reason``; ``usage``
      carrying ``output_tokens``.

    Thinking blocks (``type="thinking"``) are accumulated but excluded from
    the final ``content`` list because they are internal reasoning and callers
    read only ``text`` / ``tool_use`` blocks via :class:`AnthropicMessage`.

    Returns a plain dict in the same shape as a non-streaming Anthropic
    ``messages.create`` response so the same
    :func:`_anthropic_to_openai_shape` canonicalisation applies.
    """
    msg_id: Optional[str] = None
    msg_model: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: Optional[str] = None

    # Per-block accumulators: index → {"type", "text"/"thinking"/"tool_use", ...}
    blocks: Dict[int, dict] = {}

    for event in chunks:
        etype = _get_event_attr(event, "type", "")

        if etype == "message_start":
            msg = _get_event_attr(event, "message") or {}
            if isinstance(msg, Mapping):
                msg_id = msg.get("id", msg_id)
                msg_model = msg.get("model", msg_model)
                usage = msg.get("usage") or {}
                input_tokens = int(usage.get("input_tokens", input_tokens))
            else:
                msg_id = getattr(msg, "id", msg_id)
                msg_model = getattr(msg, "model", msg_model)
                usage = getattr(msg, "usage", None) or {}
                if isinstance(usage, Mapping):
                    input_tokens = int(usage.get("input_tokens", input_tokens))
                else:
                    input_tokens = int(getattr(usage, "input_tokens", input_tokens))

        elif etype == "content_block_start":
            idx = int(_get_event_attr(event, "index", 0))
            cb = _get_event_attr(event, "content_block") or {}
            if isinstance(cb, Mapping):
                btype = cb.get("type", "text")
                blocks[idx] = {
                    "type": btype,
                    "id": cb.get("id"),
                    "name": cb.get("name"),
                    "text_parts": [],
                    "thinking_parts": [],
                    "json_parts": [],
                }
            else:
                btype = getattr(cb, "type", "text")
                blocks[idx] = {
                    "type": btype,
                    "id": getattr(cb, "id", None),
                    "name": getattr(cb, "name", None),
                    "text_parts": [],
                    "thinking_parts": [],
                    "json_parts": [],
                }

        elif etype == "content_block_delta":
            idx = int(_get_event_attr(event, "index", 0))
            delta = _get_event_attr(event, "delta") or {}
            if idx not in blocks:
                blocks[idx] = {
                    "type": "text",
                    "id": None, "name": None,
                    "text_parts": [], "thinking_parts": [], "json_parts": [],
                }
            if isinstance(delta, Mapping):
                dtype = delta.get("type", "")
                if dtype == "text_delta":
                    blocks[idx]["text_parts"].append(delta.get("text", ""))
                elif dtype == "thinking_delta":
                    blocks[idx]["thinking_parts"].append(delta.get("thinking", ""))
                elif dtype == "input_json_delta":
                    blocks[idx]["json_parts"].append(delta.get("partial_json", ""))
            else:
                dtype = getattr(delta, "type", "")
                if dtype == "text_delta":
                    blocks[idx]["text_parts"].append(getattr(delta, "text", ""))
                elif dtype == "thinking_delta":
                    blocks[idx]["thinking_parts"].append(getattr(delta, "thinking", ""))
                elif dtype == "input_json_delta":
                    blocks[idx]["json_parts"].append(getattr(delta, "partial_json", ""))

        elif etype == "message_delta":
            delta = _get_event_attr(event, "delta") or {}
            if isinstance(delta, Mapping):
                stop_reason = delta.get("stop_reason", stop_reason)
            else:
                stop_reason = getattr(delta, "stop_reason", stop_reason)
            usage = _get_event_attr(event, "usage") or {}
            if isinstance(usage, Mapping):
                output_tokens = int(usage.get("output_tokens", output_tokens))
            else:
                output_tokens = int(getattr(usage, "output_tokens", output_tokens))

    # Assemble content blocks in index order, skipping thinking blocks.
    content: List[dict] = []
    for idx in sorted(blocks.keys()):
        b = blocks[idx]
        btype = b["type"]
        if btype == "thinking":
            # Thinking blocks are internal; exclude from the canonical content list.
            continue
        if btype == "text":
            content.append({
                "type": "text",
                "text": "".join(b["text_parts"]),
            })
        elif btype == "tool_use":
            import json as _json
            raw_json = "".join(b["json_parts"])
            try:
                parsed_input: Any = _json.loads(raw_json) if raw_json else {}
            except Exception:
                parsed_input = {}
            block: dict = {"type": "tool_use"}
            if b["id"] is not None:
                block["id"] = b["id"]
            if b["name"] is not None:
                block["name"] = b["name"]
            block["input"] = parsed_input
            content.append(block)

    return {
        "id": msg_id or "",
        "model": msg_model or "",
        "role": "assistant",
        "content": content,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


class _AnthropicMessagesProxy:
    def __init__(self, real: Any, recorder: Recorder,
                 seed_policy: Optional[SeedPolicy] = None,
                 contract: Optional["ShimContract"] = None) -> None:
        self._real = real
        self._rec = recorder
        self._seed_policy = seed_policy or get_seed_policy()
        self._contract = contract

    def _get_contract(self) -> "ShimContract":
        if self._contract is not None:
            return self._contract
        return shim_contract_for("anthropic")

    def create(self, *, model: str, messages: List[dict],
               system: Optional[str] = None,
               max_tokens: int = 1024,
               **kwargs: Any) -> AnthropicMessage:
        # Build the OpenAI-style message list we record on the inputs
        # side, so substitutions written against either provider land
        # in the same field.
        contract = self._get_contract()
        unified_messages = contract.canonical_request(messages=messages, system=system)
        temperature = float(kwargs.get("temperature", 0.0))
        caller_seed = kwargs.get("seed")
        # Anthropic does not support seed; check emits a warning if needed.
        effective_seed = self._seed_policy.check(
            "anthropic", caller_seed, temperature, model=model
        )
        # Strip seed from the kwargs sent to the Anthropic API (unsupported).
        api_kwargs = {k: v for k, v in kwargs.items() if k not in ("seed", "system")}
        is_stream = bool(api_kwargs.get("stream"))

        if is_stream:
            _raw_events: List[Any] = []

            def executor(_model: str, _messages: List[dict]) -> dict:
                anth_messages = [m for m in _messages if m.get("role") != "system"]
                anth_system = next(
                    (m["content"] for m in _messages if m.get("role") == "system"), system
                )
                stream = self._real.create(
                    model=_model,
                    messages=anth_messages,
                    system=anth_system,
                    max_tokens=max_tokens,
                    **api_kwargs,
                )
                for event in stream:
                    _raw_events.append(event)
                raw_dict = _accumulate_anthropic_streaming_chunks(iter(_raw_events))
                return _anthropic_to_openai_shape(raw_dict)
        else:
            def executor(_model: str, _messages: List[dict]) -> dict:  # type: ignore[no-redef]
                anth_messages = [m for m in _messages if m.get("role") != "system"]
                anth_system = next(
                    (m["content"] for m in _messages if m.get("role") == "system"), system
                )
                resp = self._real.create(
                    model=_model,
                    messages=anth_messages,
                    system=anth_system,
                    max_tokens=max_tokens,
                    **api_kwargs,
                )
                return contract.canonical_response(resp)

        step = self._rec.llm_call(
            model=model,
            messages=unified_messages,
            executor=executor,
            temperature=temperature,
            seed=effective_seed,
            tools=kwargs.get("tools"),
            response_format=None,
        )
        native_raw = step["llm_response"].get("_anthropic", {})
        native_msg = AnthropicMessage.from_dict(native_raw or {
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
        if is_stream:
            return StreamedLLMResponse(  # type: ignore[return-value]
                _raw_events, step["llm_response"], step, native=native_msg
            )
        return native_msg


@dataclass
class WrappedAnthropic:
    messages: _AnthropicMessagesProxy
    _real: Any
    _rec: Recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def wrap_anthropic(client: Any, recorder: Recorder,
                   seed_policy: Optional[SeedPolicy] = None,
                   contract: Optional["ShimContract"] = None) -> WrappedAnthropic:
    """Wrap a real Anthropic client so ``messages.create`` is recorded.

    The recorded ``llm_response`` is the OpenAI-canonical shape so
    cost / cache semantics are uniform across providers; the original
    Anthropic payload is preserved under ``llm_response._anthropic``.

    The *seed_policy* controls the warning emitted because Anthropic does
    not support a ``seed`` parameter.  Defaults to
    :data:`~stepback.seeding.DEFAULT_SEED_POLICY`.

    The optional *contract* parameter accepts a :class:`ShimContract` instance
    to override the built-in :class:`AnthropicShimContract`.
    """
    if not hasattr(client, "messages"):
        raise TypeError(
            "wrap_anthropic: client lacks .messages; "
            "expected an anthropic.Anthropic-shaped object"
        )
    return WrappedAnthropic(
        messages=_AnthropicMessagesProxy(client.messages, recorder,
                                         seed_policy=seed_policy, contract=contract),
        _real=client, _rec=recorder,
    )


# =====================================================================
# Async Anthropic client wrapper
# =====================================================================


class _AsyncAnthropicMessagesProxy:
    """Async counterpart to :class:`_AnthropicMessagesProxy`.

    ``create(...)`` is an *async* method: it eagerly awaits the real async
    client call and records the response synchronously through the recorder.
    On replay, the synchronous engine serves from cache as normal.
    """

    def __init__(self, real: Any, recorder: Recorder,
                 seed_policy: Optional[SeedPolicy] = None,
                 contract: Optional["ShimContract"] = None) -> None:
        self._real = real
        self._rec = recorder
        self._seed_policy = seed_policy or get_seed_policy()
        self._contract = contract

    def _get_contract(self) -> "ShimContract":
        if self._contract is not None:
            return self._contract
        return shim_contract_for("anthropic")

    async def create(self, *, model: str, messages: List[dict],
                     system: Optional[str] = None,
                     max_tokens: int = 1024,
                     **kwargs: Any) -> AnthropicMessage:
        """Await the real async Anthropic API call, then record the response."""
        contract = self._get_contract()
        unified_messages = contract.canonical_request(messages=messages, system=system)
        temperature = float(kwargs.get("temperature", 0.0))
        caller_seed = kwargs.get("seed")
        effective_seed = self._seed_policy.check(
            "anthropic", caller_seed, temperature, model=model
        )
        api_kwargs = {k: v for k, v in kwargs.items() if k not in ("seed", "system")}

        anth_messages = [m for m in messages if m.get("role") != "system"]
        anth_system = system

        resp = await self._real.create(
            model=model,
            messages=anth_messages,
            system=anth_system,
            max_tokens=max_tokens,
            **api_kwargs,
        )
        canonical = contract.canonical_response(resp)

        def _sync_executor(_m: str, _msgs: List[dict]) -> dict:
            return canonical

        step = self._rec.llm_call(
            model=model,
            messages=unified_messages,
            executor=_sync_executor,
            temperature=temperature,
            seed=effective_seed,
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
class AsyncWrappedAnthropic:
    """Async drop-in replacement for ``anthropic.AsyncAnthropic(...)``.

    ``client.messages.create(...)`` is an ``async def`` that awaits the real
    async client, then records the response.  All other attributes are passed
    through to the real client.
    """

    messages: _AsyncAnthropicMessagesProxy
    _real: Any
    _rec: Recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def wrap_anthropic_async(client: Any, recorder: Recorder,
                         seed_policy: Optional[SeedPolicy] = None,
                         contract: Optional["ShimContract"] = None) -> AsyncWrappedAnthropic:
    """Wrap a real ``anthropic.AsyncAnthropic`` client so async ``messages.create``
    calls are recorded.

    The recorded ``llm_response`` is the OpenAI-canonical shape so cost / cache
    semantics are uniform; the original Anthropic payload is preserved under
    ``llm_response._anthropic``.

    Raises :class:`TypeError` if *client* lacks ``.messages``.
    """
    if not hasattr(client, "messages"):
        raise TypeError(
            "wrap_anthropic_async: client lacks .messages; "
            "expected an anthropic.AsyncAnthropic-shaped object"
        )
    return AsyncWrappedAnthropic(
        messages=_AsyncAnthropicMessagesProxy(client.messages, recorder,
                                              seed_policy=seed_policy, contract=contract),
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
    """Wrap a single LangChain ``BaseTool``-shaped object so its invocations record.

    The returned :class:`WrappedLangchainTool` exposes the same ``invoke`` /
    ``run`` surface as the underlying tool but every call appends a typed
    ``tool_call`` frame to the recorder's ``.sb`` trace.
    """
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
    """Wrap an iterable of LangChain tools, returning a list of recording wrappers."""
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
    """Wrap an MCP ``ClientSession`` so its ``call_tool`` invocations record.

    Tool names are namespaced on the timeline as ``{server_name}:{tool}`` to
    keep cross-server traces legible. All other session methods pass through
    unmodified.
    """
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
    tool_calls: list[dict[str, Any]] = []
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
    tool_calls_out: Optional[list[dict[str, Any]]] = tool_calls or None

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
                "tool_calls": tool_calls_out,
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
    def __init__(self, real: Any, recorder: Recorder,
                 seed_policy: Optional[SeedPolicy] = None,
                 contract: Optional["ShimContract"] = None) -> None:
        self._real = real
        self._rec = recorder
        self._seed_policy = seed_policy or get_seed_policy()
        self._contract = contract

    def _get_contract(self) -> "ShimContract":
        if self._contract is not None:
            return self._contract
        return shim_contract_for("bedrock")

    def __call__(self, *,
                 modelId: str,
                 messages: List[dict],
                 system: Optional[List[dict]] = None,
                 inferenceConfig: Optional[dict] = None,
                 toolConfig: Optional[dict] = None,
                 **kwargs: Any) -> dict:
        contract = self._get_contract()
        unified_messages = contract.canonical_request(messages=messages, system=system)
        cfg = dict(inferenceConfig or {})
        temperature = float(cfg.get("temperature", 0.0))
        max_tokens = int(cfg.get("maxTokens", 1024))
        caller_seed = cfg.get("seed")
        canonical_model = canonical_bedrock_model_id(modelId)
        # Bedrock's inferenceConfig does not expose a seed field for most
        # models; the policy check emits a warning when a seed is configured.
        effective_seed = self._seed_policy.check(
            "bedrock", caller_seed, temperature, model=modelId
        )
        # Build the config sent to Bedrock without the seed key (unsupported).
        bedrock_cfg = {k: v for k, v in cfg.items() if k != "seed"}
        if not bedrock_cfg:
            bedrock_cfg = {"temperature": temperature, "maxTokens": max_tokens}

        def executor(_model: str, _messages: List[dict]) -> dict:
            bed_messages, bed_system = _unified_to_bedrock_messages(_messages)
            kwargs_clean = {k: v for k, v in kwargs.items()
                            if k not in ("system", "inferenceConfig", "toolConfig")}
            call_kwargs: dict = {
                "modelId": modelId,
                "messages": bed_messages,
                "inferenceConfig": bedrock_cfg,
            }
            if bed_system is not None:
                call_kwargs["system"] = bed_system
            if toolConfig is not None:
                call_kwargs["toolConfig"] = toolConfig
            call_kwargs.update(kwargs_clean)
            resp = self._real.converse(**call_kwargs)
            native = _coerce_bedrock_response(resp)
            native.setdefault("_modelId", modelId)
            return contract.canonical_response(native)

        step = self._rec.llm_call(
            model=canonical_model,
            messages=unified_messages,
            executor=executor,
            temperature=temperature,
            seed=effective_seed,
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


def wrap_bedrock(client: Any, recorder: Recorder,
                 seed_policy: Optional[SeedPolicy] = None,
                 contract: Optional["ShimContract"] = None) -> WrappedBedrock:
    """Wrap a real ``bedrock-runtime`` client so ``converse`` is recorded.

    The wrapped object exposes ``client.converse(modelId=..., messages=...,
    system=..., inferenceConfig=..., toolConfig=...)`` and returns the
    native Bedrock response dict (so existing agent code keeps working).
    Internally each call is canonicalised into the OpenAI chat-completion
    shape used by every other provider shim, so substitutions, cache
    semantics, and cost accounting are uniform.

    The *seed_policy* controls the warning emitted because Bedrock's
    ``inferenceConfig`` does not expose a ``seed`` field.  Defaults to
    :data:`~stepback.seeding.DEFAULT_SEED_POLICY`.

    The optional *contract* parameter accepts a :class:`ShimContract` instance
    to override the built-in :class:`BedrockShimContract`.

    Raises :class:`TypeError` if the client lacks ``.converse`` (i.e.
    isn't a Bedrock-runtime-shaped object).
    """
    if not hasattr(client, "converse"):
        raise TypeError(
            "wrap_bedrock: client lacks .converse; "
            "expected a boto3 bedrock-runtime-shaped object"
        )
    return WrappedBedrock(
        converse=_BedrockConverseProxy(client, recorder, seed_policy=seed_policy,
                                       contract=contract),
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


# =====================================================================
# Google Gemini shim (google-genai SDK)
# =====================================================================
#
# The 2025-Q1 ``google-genai`` SDK exposes Gemini through:
#
#     from google import genai
#     client = genai.Client(api_key=...)
#     resp = client.models.generate_content(
#         model="gemini-2.5-flash",
#         contents=[
#             {"role": "user",
#              "parts": [{"text": "hello"}]}
#         ],
#         config={
#             "system_instruction": "be terse",
#             "temperature": 0.0,
#             "max_output_tokens": 1024,
#             "tools": [...],
#             "seed": 42,
#         },
#     )
#     # → resp.text, resp.candidates[*].content.parts[*],
#     #   resp.candidates[*].finish_reason,
#     #   resp.usage_metadata.{prompt_token_count,
#     #                       candidates_token_count,
#     #                       total_token_count}
#
# Vertex AI's ``vertexai.generative_models.GenerativeModel`` exposes
# the same shape under ``model.generate_content(contents, generation_
# config=..., tools=..., system_instruction=...)``; the duck-typed
# helper :func:`wrap_vertex_model` below records that surface too,
# delegating to the same canonicaliser.
#
# We project Gemini's ``contents=[{role, parts:[{text|function_call|
# function_response}]}]`` onto the unified OpenAI-style
# ``[{role, content}]`` list so substitutions, the content-addressed
# cache, and ``stepback/pricing.py`` cost accounting are uniform
# across providers. The native Gemini payload is preserved under
# ``llm_response._gemini`` for replay-side reconstruction.


_GEMINI_ROLE_FROM = {"user": "user", "model": "assistant", "assistant": "assistant"}
_GEMINI_ROLE_TO = {"user": "user", "assistant": "model", "system": "user"}

# ── Safety-settings canonicalization ────────────────────────────────────────
#
# Gemini/Vertex accept safety settings as a list of {harm_category, threshold}
# pairs using either string literals, enum objects, or SDK-typed values. We
# normalise everything to a canonical string so the inputs hash is stable
# regardless of whether the caller used the SDK enum or the raw string.

_GEMINI_HARM_CATEGORY_MAP: Dict[str, str] = {
    "HARM_CATEGORY_UNSPECIFIED": "HARM_CATEGORY_UNSPECIFIED",
    "HARM_CATEGORY_DEROGATORY": "HARM_CATEGORY_DEROGATORY",
    "HARM_CATEGORY_TOXICITY": "HARM_CATEGORY_TOXICITY",
    "HARM_CATEGORY_VIOLENCE": "HARM_CATEGORY_VIOLENCE",
    "HARM_CATEGORY_SEXUAL": "HARM_CATEGORY_SEXUAL",
    "HARM_CATEGORY_MEDICAL": "HARM_CATEGORY_MEDICAL",
    "HARM_CATEGORY_DANGEROUS": "HARM_CATEGORY_DANGEROUS",
    "HARM_CATEGORY_HARASSMENT": "HARM_CATEGORY_HARASSMENT",
    "HARASSMENT": "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH": "HARM_CATEGORY_HATE_SPEECH",
    "HATE_SPEECH": "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "SEXUALLY_EXPLICIT": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "DANGEROUS_CONTENT": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY": "HARM_CATEGORY_CIVIC_INTEGRITY",
    "CIVIC_INTEGRITY": "HARM_CATEGORY_CIVIC_INTEGRITY",
}

_GEMINI_HARM_THRESHOLD_MAP: Dict[str, str] = {
    "HARM_BLOCK_THRESHOLD_UNSPECIFIED": "HARM_BLOCK_THRESHOLD_UNSPECIFIED",
    "BLOCK_LOW_AND_ABOVE": "BLOCK_LOW_AND_ABOVE",
    "BLOCK_MEDIUM_AND_ABOVE": "BLOCK_MEDIUM_AND_ABOVE",
    "BLOCK_ONLY_HIGH": "BLOCK_ONLY_HIGH",
    "BLOCK_NONE": "BLOCK_NONE",
    "OFF": "BLOCK_NONE",
}


def _coerce_enum_str(val: Any) -> str:
    """Extract a plain string from an enum-like value.

    Tries ``.value`` first (SDK enums store the canonical string there),
    then ``.name``, then falls back to ``str()``.
    """
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    v = getattr(val, "value", None)
    if isinstance(v, str):
        return v
    n = getattr(val, "name", None)
    if isinstance(n, str):
        return n
    return str(val)


def _canonicalize_gemini_safety_settings(
    safety_settings: Any,
) -> Optional[List[dict]]:
    """Normalise Gemini/Vertex safety settings to a canonical list.

    Accepts any of the forms the SDK and REST surface allow:

    * ``[{"harm_category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}]``
    * ``[SafetySetting(harm_category=HarmCategory.HARASSMENT, threshold=...)]``
    * A single entry (not a list) is wrapped automatically.

    Returns a list of ``{"category": <canonical>, "threshold": <canonical>}``
    dicts in their **original order** (order may affect which filters the API
    applies first; we preserve it to be safe). Returns ``None`` when the input
    is ``None`` or empty.
    """
    if safety_settings is None:
        return None
    if not isinstance(safety_settings, (list, tuple)):
        safety_settings = [safety_settings]
    if not safety_settings:
        return None

    normalised: List[dict] = []
    for s in safety_settings:
        if isinstance(s, Mapping):
            cat_raw = _coerce_enum_str(
                s.get("harm_category") or s.get("category") or ""
            ).upper()
            thr_raw = _coerce_enum_str(
                s.get("threshold") or ""
            ).upper()
            extra = {k: v for k, v in s.items()
                     if k not in ("harm_category", "category", "threshold")}
        else:
            cat_attr = getattr(s, "harm_category", None) or getattr(s, "category", None)
            thr_attr = getattr(s, "threshold", None)
            cat_raw = _coerce_enum_str(cat_attr).upper()
            thr_raw = _coerce_enum_str(thr_attr).upper()
            extra = {}
            for attr in ("method",):
                v = getattr(s, attr, None)
                if v is not None:
                    extra[attr] = _coerce_enum_str(v)

        entry: dict = {
            "category": _GEMINI_HARM_CATEGORY_MAP.get(cat_raw, cat_raw),
            "threshold": _GEMINI_HARM_THRESHOLD_MAP.get(thr_raw, thr_raw),
        }
        entry.update(extra)
        normalised.append(entry)

    return normalised or None


# ── Tool-declarations canonicalization ──────────────────────────────────────
#
# Gemini tools can be SDK-typed objects (``Tool``, ``FunctionDeclaration``),
# Pydantic models, plain dicts, or a mix. We recursively coerce everything
# to plain JSON-compatible dicts so the canonical JSON hash is stable.
# List order is **preserved** (declaration order may influence model output).


def _deep_to_dict(obj: Any) -> Any:
    """Recursively coerce an SDK/Pydantic object tree to plain JSON-compatible
    Python structures (dicts and lists of primitives)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Mapping):
        return {k: _deep_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep_to_dict(v) for v in obj]
    # Pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return _deep_to_dict(obj.model_dump())
        except Exception:
            pass
    # Pydantic v1 / other .dict()
    if hasattr(obj, "dict") and callable(obj.dict):
        try:
            return _deep_to_dict(obj.dict())
        except Exception:
            pass
    # to_dict / as_dict helpers
    for meth in ("to_dict", "as_dict"):
        if hasattr(obj, meth) and callable(getattr(obj, meth)):
            try:
                return _deep_to_dict(getattr(obj, meth)())
            except Exception:
                pass
    # __dict__ fallback (dataclasses, plain objects)
    if hasattr(obj, "__dict__"):
        return _deep_to_dict(vars(obj))
    # Last resort: str
    return str(obj)


def _canonicalize_gemini_tool_declaration(decl: Any) -> dict:
    """Coerce a single ``FunctionDeclaration``-like object to a canonical dict."""
    if not isinstance(decl, Mapping):
        # Use getattr so both instance-level and class-level attributes are found.
        name = str(getattr(decl, "name", None) or "")
        desc = str(getattr(decl, "description", None) or "")
        params = getattr(decl, "parameters", None)
        out: dict = {"name": name}
        if desc:
            out["description"] = desc
        if params is not None:
            out["parameters"] = _deep_to_dict(params)
        return out
    # Mapping path: convert values recursively.
    name = str(decl.get("name") or "")
    out = {"name": name}
    if decl.get("description"):
        out["description"] = str(decl["description"])
    if decl.get("parameters") is not None:
        out["parameters"] = _deep_to_dict(decl["parameters"])
    return out


def _canonicalize_gemini_tools(tools: Any) -> Optional[List[dict]]:
    """Normalise Gemini tool declarations to a canonical list of dicts.

    Accepts:

    * ``[{"function_declarations": [{"name": ..., ...}]}]``  — REST/dict form
    * ``[Tool(function_declarations=[FunctionDeclaration(...)])]`` — SDK form
    * A mix of SDK objects and dicts

    Returns a list of ``{"function_declarations": [...]}`` dicts with each
    declaration coerced to a plain dict.  List ordering is **preserved**.
    Returns ``None`` when *tools* is ``None`` or empty.
    """
    if tools is None:
        return None
    if not isinstance(tools, (list, tuple)):
        tools = [tools]
    if not tools:
        return None

    canonical_tools: List[dict] = []
    for tool in tools:
        if isinstance(tool, Mapping):
            # Standard Tool shape: {"function_declarations": [...], ...}
            fds_raw = tool.get("function_declarations") or tool.get("function_declaration")
            if fds_raw is not None:
                fds = fds_raw if isinstance(fds_raw, list) else [fds_raw]
                canonical_tools.append({
                    "function_declarations": [
                        _canonicalize_gemini_tool_declaration(d) for d in fds
                    ],
                })
            elif "name" in tool:
                # Bare FunctionDeclaration dict used directly as a Tool.
                canonical_tools.append({
                    "function_declarations": [_canonicalize_gemini_tool_declaration(tool)],
                })
            else:
                canonical_tools.append(_deep_to_dict(tool))
        else:
            # SDK-typed object: use getattr to access class-level or instance attrs.
            fds_raw = (
                getattr(tool, "function_declarations", None)
                or getattr(tool, "function_declaration", None)
            )
            if fds_raw is not None:
                fds = fds_raw if isinstance(fds_raw, (list, tuple)) else [fds_raw]
                canonical_tools.append({
                    "function_declarations": [
                        _canonicalize_gemini_tool_declaration(d) for d in fds
                    ],
                })
            elif getattr(tool, "name", None) is not None:
                # Bare FunctionDeclaration object used as a Tool.
                canonical_tools.append({
                    "function_declarations": [_canonicalize_gemini_tool_declaration(tool)],
                })
            else:
                raw = _deep_to_dict(tool)
                canonical_tools.append(raw if isinstance(raw, Mapping) else {"raw": str(raw)})

    return canonical_tools or None


def _gemini_contents_to_unified(
    contents: Any,
    system_instruction: Optional[Any],
) -> List[dict]:
    """Project Gemini ``contents=[{role, parts:[...]}]`` onto the
    unified OpenAI-style ``[{role, content}]`` list. Concatenates
    ``text`` parts; preserves structured parts (``function_call``,
    ``function_response``, inline data, file data) under
    ``_gemini_parts`` so a substitution writer can still round-trip
    them on replay."""
    unified: List[dict] = []
    if system_instruction is not None:
        sys_text = _gemini_extract_system_text(system_instruction)
        if sys_text:
            unified.append({"role": "system", "content": sys_text})

    if contents is None:
        return unified
    # Allow a bare string (the SDK accepts it) and bare list-of-strings.
    if isinstance(contents, str):
        return unified + [{"role": "user", "content": contents}]
    if not isinstance(contents, (list, tuple)):
        contents = [contents]

    for c in contents:
        if isinstance(c, str):
            unified.append({"role": "user", "content": c})
            continue
        if not isinstance(c, Mapping):
            # Duck-typed object: pull .role / .parts off it.
            role = getattr(c, "role", "user") or "user"
            parts = getattr(c, "parts", []) or []
        else:
            role = c.get("role", "user") or "user"
            parts = c.get("parts") or []
        unified_role = _GEMINI_ROLE_FROM.get(role, role)
        if isinstance(parts, str):
            unified.append({"role": unified_role, "content": parts})
            continue
        text_chunks: List[str] = []
        non_text: List[dict] = []
        for p in parts:
            if isinstance(p, str):
                text_chunks.append(p)
                continue
            if isinstance(p, Mapping):
                if "text" in p and isinstance(p["text"], str):
                    text_chunks.append(p["text"])
                else:
                    non_text.append(dict(p))
            else:
                # Duck-typed Part object.
                t = getattr(p, "text", None)
                if isinstance(t, str) and t:
                    text_chunks.append(t)
                    continue
                fc = getattr(p, "function_call", None)
                if fc is not None:
                    name = getattr(fc, "name", None) or (fc.get("name") if isinstance(fc, Mapping) else None)
                    args = getattr(fc, "args", None) or (fc.get("args") if isinstance(fc, Mapping) else None)
                    non_text.append({"function_call": {"name": name, "args": args or {}}})
                    continue
                fr = getattr(p, "function_response", None)
                if fr is not None:
                    name = getattr(fr, "name", None) or (fr.get("name") if isinstance(fr, Mapping) else None)
                    resp_ = getattr(fr, "response", None) or (fr.get("response") if isinstance(fr, Mapping) else None)
                    non_text.append({"function_response": {"name": name, "response": resp_ or {}}})
                    continue
        entry: dict = {"role": unified_role, "content": "".join(text_chunks)}
        if non_text:
            entry["_gemini_parts"] = non_text
        unified.append(entry)
    return unified


def _gemini_extract_system_text(system_instruction: Any) -> str:
    """Coerce the variety of shapes the SDK accepts for
    ``system_instruction`` (str, dict with ``parts``, list of parts,
    duck-typed Content) into a flat string."""
    if system_instruction is None:
        return ""
    if isinstance(system_instruction, str):
        return system_instruction
    if isinstance(system_instruction, Mapping):
        parts = system_instruction.get("parts") or []
    else:
        parts = getattr(system_instruction, "parts", None) or []
        if not parts and isinstance(system_instruction, (list, tuple)):
            parts = system_instruction
    chunks: List[str] = []
    for p in parts or []:
        if isinstance(p, str):
            chunks.append(p)
        elif isinstance(p, Mapping) and isinstance(p.get("text"), str):
            chunks.append(p["text"])
        else:
            t = getattr(p, "text", None)
            if isinstance(t, str):
                chunks.append(t)
    return "".join(chunks)


def _unified_to_gemini_contents(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[List[dict], Optional[dict]]:
    """Inverse of :func:`_gemini_contents_to_unified` for replay."""
    system: Optional[dict] = None
    contents: List[dict] = []
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            sys_text = m.get("content") or ""
            system = {"parts": [{"text": sys_text}]} if sys_text else None
            continue
        gem_role = _GEMINI_ROLE_TO.get(role, role)
        parts: List[dict] = []
        text = m.get("content") or ""
        if text:
            parts.append({"text": text})
        for extra in m.get("_gemini_parts", []) or []:
            parts.append(dict(extra))
        if not parts:
            # Gemini rejects empty parts lists; emit a single empty
            # text part so the round-trip survives.
            parts.append({"text": ""})
        contents.append({"role": gem_role, "parts": parts})
    return contents, system


_GEMINI_FINISH_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "MALFORMED_FUNCTION_CALL": "tool_calls",
    "TOOL_CODE": "tool_calls",
    "OTHER": "stop",
}


# Map a Gemini SDK alias to the canonical pricing key. The SDK lets
# users say ``gemini-2.5-flash`` and routes to whatever the latest
# dated snapshot is; we record the alias as the canonical key so the
# ``RESOLVER`` table in ``stepback/pricing.py`` can reuse the row.
_GEMINI_PRICING_ALIAS: Dict[str, str] = {
    "gemini-2.5-pro-latest": "gemini-2.5-pro-2025-03-25",
    "gemini-2.5-flash-latest": "gemini-2.5-flash-2025-04-09",
    "models/gemini-2.5-pro": "gemini-2.5-pro-2025-03-25",
    "models/gemini-2.5-flash": "gemini-2.5-flash-2025-04-09",
    "publishers/google/models/gemini-2.5-pro": "gemini-2.5-pro-2025-03-25",
    "publishers/google/models/gemini-2.5-flash": "gemini-2.5-flash-2025-04-09",
}


def canonical_gemini_model_id(model_id: str) -> str:
    """Return the stepback canonical pricing key for a Gemini model id.

    Strips the ``models/`` and Vertex
    ``publishers/google/models/`` prefixes and collapses ``-latest``
    suffixes onto the dated snapshot listed in
    :data:`stepback.pricing.PRICE_LIST`. Falls back to the input string
    when no alias is registered (so unknown models still record, just
    with a 0 price).
    """
    return _GEMINI_PRICING_ALIAS.get(model_id, model_id)


def _gemini_to_openai_shape(d: Mapping[str, Any]) -> dict:
    """Project a Gemini ``generate_content`` response into the OpenAI
    chat-completion shape every other shim emits. Preserves the native
    payload under ``_gemini`` so :func:`gemini_executor` can rehydrate
    it on cached replay."""
    candidates = d.get("candidates") or []
    cand0: dict = candidates[0] if candidates else {}
    content = cand0.get("content") or {}
    parts = content.get("parts") or []

    text_chunks: List[str] = []
    tool_calls: List[dict] = []
    for p in parts:
        if not isinstance(p, Mapping):
            continue
        if "text" in p and isinstance(p["text"], str):
            text_chunks.append(p["text"])
        elif "function_call" in p:
            fc = p["function_call"] or {}
            tool_calls.append({
                "id": fc.get("id") or f"gemcall_{len(tool_calls):04d}",
                "type": "function",
                "function": {
                    "name": fc.get("name", ""),
                    "arguments": fc.get("args") or {},
                },
            })

    content_text = "".join(text_chunks) or None
    finish_raw = str(cand0.get("finish_reason") or "")
    finish_reason = _GEMINI_FINISH_MAP.get(finish_raw, finish_raw or None)

    usage = d.get("usage_metadata") or {}
    canonical_usage: dict[str, Any] = {
        "prompt_tokens": int(usage.get("prompt_token_count", 0)),
        "completion_tokens": int(usage.get("candidates_token_count", 0)),
        "total_tokens": int(
            usage.get("total_token_count",
                      int(usage.get("prompt_token_count", 0))
                      + int(usage.get("candidates_token_count", 0)))
        ),
    }
    cached = usage.get("cached_content_token_count")
    if cached:
        canonical_usage["prompt_tokens_details"] = {"cached_tokens": int(cached)}

    return _strip_none({
        "id": d.get("response_id") or d.get("id"),
        "model": d.get("model_version") or d.get("_modelId"),
        "choices": [{
            "index": 0,
            "finish_reason": finish_reason,
            "message": {
                "role": "assistant",
                "content": content_text,
                "tool_calls": tool_calls or None,
            },
        }],
        "usage": canonical_usage,
        "_gemini": dict(d),
    })


def _coerce_gemini_response(resp: Any) -> dict:
    if isinstance(resp, Mapping):
        return dict(resp)
    if hasattr(resp, "model_dump"):
        try:
            return resp.model_dump()  # type: ignore[no-any-return]
        except Exception:
            pass
    if hasattr(resp, "to_dict"):
        try:
            return resp.to_dict()  # type: ignore[no-any-return]
        except Exception:
            pass
    # Duck-typed dataclass-style object: walk known attributes.
    out: dict = {}
    cand_attr = getattr(resp, "candidates", None)
    if cand_attr is not None:
        cands: List[dict] = []
        for c in cand_attr:
            content = getattr(c, "content", None)
            content_d: dict = {}
            if isinstance(content, Mapping):
                content_d = dict(content)
            elif content is not None:
                role = getattr(content, "role", "model")
                parts_attr = getattr(content, "parts", []) or []
                parts_d: List[dict] = []
                for p in parts_attr:
                    if isinstance(p, Mapping):
                        parts_d.append(dict(p))
                        continue
                    pt = getattr(p, "text", None)
                    if isinstance(pt, str):
                        parts_d.append({"text": pt})
                        continue
                    fc = getattr(p, "function_call", None)
                    if fc is not None:
                        parts_d.append({"function_call": {
                            "name": getattr(fc, "name", "") or (fc.get("name") if isinstance(fc, Mapping) else ""),
                            "args": getattr(fc, "args", {}) or (fc.get("args") if isinstance(fc, Mapping) else {}),
                        }})
                content_d = {"role": role, "parts": parts_d}
            cands.append({
                "content": content_d,
                "finish_reason": getattr(c, "finish_reason", None),
            })
        out["candidates"] = cands
    um = getattr(resp, "usage_metadata", None)
    if um is not None:
        if isinstance(um, Mapping):
            out["usage_metadata"] = dict(um)
        else:
            out["usage_metadata"] = {
                "prompt_token_count": getattr(um, "prompt_token_count", 0),
                "candidates_token_count": getattr(um, "candidates_token_count", 0),
                "total_token_count": getattr(um, "total_token_count", 0),
            }
    mv = getattr(resp, "model_version", None)
    if mv is not None:
        out["model_version"] = mv
    if not out:
        raise TypeError(
            f"unsupported Gemini response type: {type(resp).__name__}"
        )
    return out


@dataclass
class GeminiCandidate:
    content: dict
    finish_reason: Optional[str]
    index: int = 0


@dataclass
class GeminiUsageMetadata:
    prompt_token_count: int
    candidates_token_count: int
    total_token_count: int


@dataclass
class GeminiResponse:
    """SDK-shaped namespace returned by the wrapped ``generate_content``.

    Mirrors the fields agents read off the real
    ``google.genai`` response: ``.text``, ``.candidates``,
    ``.usage_metadata``, plus dict-style access to the canonical
    OpenAI-shaped payload for stepback's own tooling."""

    text: Optional[str]
    candidates: List[GeminiCandidate]
    usage_metadata: GeminiUsageMetadata
    model_version: Optional[str]
    _native: dict
    _canonical: dict

    def __getitem__(self, k: str) -> Any:
        return self._canonical[k]

    def get(self, k: str, default: Any = None) -> Any:
        return self._canonical.get(k, default)

    @property
    def function_calls(self) -> List[dict]:
        out = []
        for c in self.candidates:
            for p in (c.content or {}).get("parts", []) or []:
                if isinstance(p, Mapping) and "function_call" in p:
                    out.append(p["function_call"])
        return out

    @classmethod
    def from_native_and_canonical(cls, native: dict, canonical: dict) -> "GeminiResponse":
        cands_native = native.get("candidates") or []
        cands: List[GeminiCandidate] = []
        for i, c in enumerate(cands_native):
            cands.append(GeminiCandidate(
                content=dict(c.get("content") or {}),
                finish_reason=c.get("finish_reason"),
                index=i,
            ))
        usage_n = native.get("usage_metadata") or {}
        usage = GeminiUsageMetadata(
            prompt_token_count=int(usage_n.get("prompt_token_count", 0)),
            candidates_token_count=int(usage_n.get("candidates_token_count", 0)),
            total_token_count=int(usage_n.get("total_token_count", 0)),
        )
        text = canonical["choices"][0]["message"].get("content") if canonical.get("choices") else None
        return cls(
            text=text,
            candidates=cands,
            usage_metadata=usage,
            model_version=native.get("model_version"),
            _native=native,
            _canonical=canonical,
        )


def _gemini_config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


class _GeminiModelsProxy:
    def __init__(self, real: Any, recorder: Recorder, *,
                 default_model: Optional[str],
                 seed_policy: Optional[SeedPolicy] = None,
                 contract: Optional["ShimContract"] = None) -> None:
        self._real = real
        self._rec = recorder
        self._default_model = default_model
        self._seed_policy = seed_policy or get_seed_policy()
        self._contract = contract

    def _get_contract(self) -> "ShimContract":
        if self._contract is not None:
            return self._contract
        return shim_contract_for("gemini")

    def generate_content(self, *,
                         contents: Any,
                         model: Optional[str] = None,
                         config: Any = None,
                         **kwargs: Any) -> GeminiResponse:
        chosen_model = model or self._default_model
        if chosen_model is None:
            raise ValueError(
                "wrap_gemini: no model given and no default_model set on the wrapper"
            )
        contract = self._get_contract()
        system_instruction = _gemini_config_get(config, "system_instruction")
        unified_messages = contract.canonical_request(
            contents=contents, system_instruction=system_instruction
        )
        temperature = float(_gemini_config_get(config, "temperature", 0.0) or 0.0)
        caller_seed = _gemini_config_get(config, "seed")
        tools = _gemini_config_get(config, "tools")
        safety_settings_raw = _gemini_config_get(config, "safety_settings")
        response_format = None
        rmime = _gemini_config_get(config, "response_mime_type")
        rschema = _gemini_config_get(config, "response_schema")
        if rmime or rschema:
            response_format = {
                "mime_type": rmime, "schema": rschema,
            }
        canonical_model = canonical_gemini_model_id(chosen_model)
        # Canonicalize tools and safety settings before hashing.
        canonical_tools = _canonicalize_gemini_tools(tools)
        canonical_safety = _canonicalize_gemini_safety_settings(safety_settings_raw)
        # Gemini offers best-effort seed support; check for warnings.
        effective_seed = self._seed_policy.check(
            "gemini", caller_seed if isinstance(caller_seed, int) else None,
            temperature, model=chosen_model
        )

        def executor(_model: str, _messages: List[dict]) -> dict:
            gem_contents, gem_system = _unified_to_gemini_contents(_messages)
            call_kwargs = dict(kwargs)
            cfg = config
            if gem_system is not None:
                if isinstance(cfg, Mapping):
                    cfg = dict(cfg)
                    cfg["system_instruction"] = gem_system
                elif cfg is None:
                    cfg = {"system_instruction": gem_system}
                else:
                    try:
                        setattr(cfg, "system_instruction", gem_system)
                    except Exception:
                        cfg = {"system_instruction": gem_system}
            resp = self._real.generate_content(
                model=chosen_model, contents=gem_contents, config=cfg, **call_kwargs
            )
            native = _coerce_gemini_response(resp)
            native.setdefault("model_version", chosen_model)
            return contract.canonical_response(native)

        step = self._rec.llm_call(
            model=canonical_model,
            messages=unified_messages,
            executor=executor,
            temperature=temperature,
            seed=effective_seed,
            tools=canonical_tools,
            response_format=response_format,
            safety_settings=canonical_safety,
        )
        native = step["llm_response"].get("_gemini")
        if not native:
            # Synthesise a minimal native shape from the canonical fields.
            usage = step["llm_response"].get("usage", {}) or {}
            ch0 = (step["llm_response"].get("choices") or [{}])[0]
            msg = ch0.get("message") or {}
            parts: List[dict] = []
            if msg.get("content"):
                parts.append({"text": msg["content"]})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                parts.append({"function_call": {
                    "name": fn.get("name", ""),
                    "args": fn.get("arguments") or {},
                }})
            native = {
                "candidates": [{
                    "content": {"role": "model", "parts": parts or [{"text": ""}]},
                    "finish_reason": (ch0.get("finish_reason") or "STOP").upper(),
                }],
                "usage_metadata": {
                    "prompt_token_count": int(usage.get("prompt_tokens", 0)),
                    "candidates_token_count": int(usage.get("completion_tokens", 0)),
                    "total_token_count": int(usage.get("total_tokens", 0)),
                },
                "model_version": chosen_model,
            }
        return GeminiResponse.from_native_and_canonical(native, step["llm_response"])

    def embed_content(self, *,
                      model: Optional[str] = None,
                      contents: Any,
                      config: Any = None,
                      **kwargs: Any) -> Any:
        """Record a ``models.embed_content`` call as a ``tool_call`` step.

        The embedding vector is large and not useful for substitution, so it
        is stored in the tool call's ``result`` field as-is.  The canonical
        ``arguments`` include the model id, the text contents, and any
        task-type / output-dimensionality settings from *config* so that
        different embedding configurations produce distinct cache keys.

        Returns the raw response dict (or SDK object) from the underlying
        ``embed_content`` call, unchanged.

        .. note::
            The standalone :func:`gemini_executor` does not handle embedding
            replay; dirty replay of embedding steps requires a custom executor.
        """
        chosen_model = model or self._default_model
        if chosen_model is None:
            raise ValueError(
                "wrap_gemini: no model given for embed_content and no default_model set"
            )
        canonical_model = canonical_gemini_model_id(chosen_model)

        # Extract embedding-specific config for canonical argument hashing.
        task_type = _gemini_config_get(config, "task_type")
        output_dim = _gemini_config_get(config, "output_dimensionality")
        title = _gemini_config_get(config, "title")

        # Normalise contents to a canonical form for hashing.
        if isinstance(contents, str):
            canonical_contents: Any = contents
        elif isinstance(contents, (list, tuple)):
            canonical_contents = [
                item if isinstance(item, str) else _deep_to_dict(item)
                for item in contents
            ]
        else:
            canonical_contents = _deep_to_dict(contents)

        arguments: dict = {"model": canonical_model, "contents": canonical_contents}
        if task_type is not None:
            arguments["task_type"] = _coerce_enum_str(task_type)
        if output_dim is not None:
            arguments["output_dimensionality"] = int(output_dim)
        if title is not None:
            arguments["title"] = str(title)

        def _executor(_name: str, _args: dict) -> Any:
            resp = self._real.embed_content(
                model=chosen_model, contents=contents, config=config, **kwargs
            )
            # Coerce SDK object to a plain dict for storage.
            if isinstance(resp, Mapping):
                return dict(resp)
            if hasattr(resp, "model_dump"):
                try:
                    return resp.model_dump()
                except Exception:
                    pass
            if hasattr(resp, "to_dict"):
                try:
                    return resp.to_dict()
                except Exception:
                    pass
            return resp

        step = self._rec.tool_call(
            name=f"embed_content:{canonical_model}",
            arguments=arguments,
            executor=_executor,
        )
        return step["outputs"]["result"]


@dataclass
class WrappedGemini:
    """Drop-in wrapper for ``google.genai.Client`` recording every
    ``client.models.generate_content`` call.

    Other surfaces (``client.files``, ``client.caches``, ...) are
    passed through to the real client untouched."""

    models: _GeminiModelsProxy
    _real: Any
    _rec: Recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def wrap_gemini(client: Any, recorder: Recorder, *,
                default_model: Optional[str] = None,
                seed_policy: Optional[SeedPolicy] = None,
                contract: Optional["ShimContract"] = None) -> WrappedGemini:
    """Wrap a real ``google.genai.Client`` so ``models.generate_content``
    is recorded.

    The *seed_policy* controls the default seed applied to every call.
    Gemini offers best-effort seed support; a :class:`SeedPolicyViolation`
    warning is emitted when ``temperature > 0`` and the policy level is
    :attr:`SeedWarnLevel.WARN`.  Defaults to
    :data:`~stepback.seeding.DEFAULT_SEED_POLICY`.

    The optional *contract* parameter accepts a :class:`ShimContract` instance
    to override the built-in :class:`GeminiShimContract`.

    Example::

        from google import genai
        from stepback import record
        from stepback.shims import wrap_gemini

        with record("./trace.sb") as rec:
            client = wrap_gemini(genai.Client(), rec,
                                 default_model="gemini-2.5-flash")
            resp = client.models.generate_content(
                contents="hi",
                config={"temperature": 0.0,
                        "system_instruction": "be terse"},
            )

    Raises :class:`TypeError` if the client lacks ``models.generate_content``
    (i.e. isn't a google-genai-shaped object).
    """
    models = getattr(client, "models", None)
    if models is None or not hasattr(models, "generate_content"):
        raise TypeError(
            "wrap_gemini: client lacks .models.generate_content; "
            "expected a google.genai.Client-shaped object"
        )
    return WrappedGemini(
        models=_GeminiModelsProxy(models, recorder, default_model=default_model,
                                  seed_policy=seed_policy, contract=contract),
        _real=client, _rec=recorder,
    )


def wrap_vertex_model(model: Any, recorder: Recorder, *,
                      model_name: Optional[str] = None,
                      seed_policy: Optional[SeedPolicy] = None) -> "_GeminiModelsProxy":
    """Wrap a Vertex AI ``GenerativeModel`` instance. Vertex's
    ``model.generate_content(contents, generation_config=...,
    system_instruction=..., tools=...)`` is duck-typed onto the
    google-genai surface so the same canonicaliser handles it.

    The *seed_policy* controls the default seed and warnings for best-effort
    Vertex AI seed support.  Defaults to
    :data:`~stepback.seeding.DEFAULT_SEED_POLICY`.

    Returns a proxy whose ``generate_content`` records the call.
    """
    inferred_name = model_name or getattr(model, "_model_name", None) or getattr(model, "model_name", None)

    class _VertexAdapter:
        def generate_content(self, *, model: Optional[str] = None,
                             contents: Any, config: Any = None,
                             **kwargs: Any) -> Any:
            gen_kwargs: dict = dict(kwargs)
            if config is not None:
                if isinstance(config, Mapping):
                    cfg = dict(config)
                    sys_inst = cfg.pop("system_instruction", None)
                    tools = cfg.pop("tools", None)
                    safety = cfg.pop("safety_settings", None)
                    if sys_inst is not None:
                        gen_kwargs["system_instruction"] = sys_inst
                    if tools is not None:
                        gen_kwargs["tools"] = tools
                    if safety is not None:
                        gen_kwargs["safety_settings"] = safety
                    gen_kwargs["generation_config"] = cfg
                else:
                    gen_kwargs["generation_config"] = config
            return model_obj.generate_content(contents, **gen_kwargs)

    model_obj = model
    return _GeminiModelsProxy(_VertexAdapter(), recorder, default_model=inferred_name,
                              seed_policy=seed_policy)


def gemini_executor(client: Any) -> Callable[[str, List[dict]], dict]:
    """Adapter wrapping a real ``google.genai`` client for replay.

    Translates the canonical pricing-id back into a Gemini SDK model
    name (e.g. ``gemini-2.5-flash-2025-04-09`` → kept as-is, while
    ``models/gemini-2.5-flash`` is resolved via
    :func:`canonical_gemini_model_id` on record). Returns a callable
    shaped the way :class:`stepback.replay.Executor` expects.
    """
    inverse: Dict[str, str] = {v: k for k, v in _GEMINI_PRICING_ALIAS.items()}

    def _llm(model: str, messages: List[dict]) -> dict:
        gem_contents, gem_system = _unified_to_gemini_contents(messages)
        gem_id = inverse.get(model, model)
        config: dict = {"temperature": 0.0, "max_output_tokens": 1024}
        if gem_system is not None:
            config["system_instruction"] = gem_system
        resp = client.models.generate_content(
            model=gem_id, contents=gem_contents, config=config,
        )
        native = _coerce_gemini_response(resp)
        native.setdefault("model_version", gem_id)
        return _gemini_to_openai_shape(native)

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


# =====================================================================
# Concrete ShimContract implementations (one per built-in provider)
# =====================================================================


class OpenAIShimContract(ShimContract):
    """ShimContract for the OpenAI ``chat.completions`` surface."""

    provider_name: ClassVar[str] = "openai"

    def canonical_request(self, **kwargs: Any) -> List[dict]:
        """OpenAI messages are already in unified format; return them as-is."""
        return list(kwargs.get("messages") or [])

    def canonical_response(self, native: Any) -> dict:
        """Delegate to :func:`_canonicalise_openai_response`."""
        return _canonicalise_openai_response(native)

    def make_executor(self, client: Any) -> Callable[[str, List[dict]], dict]:
        """Return an :func:`openai_executor` for *client*."""
        return openai_executor(client)

    def stream_request(self, client: Any, messages: List[dict], **kwargs: Any) -> dict:
        """Call ``client.chat.completions.create(stream=True)`` and accumulate chunks.

        Returns a canonical ``chat.completion``-shaped dict assembled from the
        full stream — individual chunks are not exposed since stepback records
        the *final* response as a single trace step.
        """
        stream_kwargs = dict(kwargs)
        stream_kwargs["stream"] = True
        stream = client.chat.completions.create(
            model=kwargs.get("model"),
            messages=messages,
            **{k: v for k, v in stream_kwargs.items() if k != "model"},
        )
        assembled = _accumulate_streaming_chunks(stream)
        return self.canonical_response(assembled)

    async def async_request(self, client: Any, messages: List[dict], **kwargs: Any) -> dict:
        """Await ``client.chat.completions.create(...)`` and return a canonical dict."""
        resp = await client.chat.completions.create(
            model=kwargs.get("model"),
            messages=messages,
            **{k: v for k, v in kwargs.items() if k != "model"},
        )
        return self.canonical_response(resp)

    def version_probe(self, client: Any) -> Optional[str]:
        try:
            import openai  # type: ignore[import-not-found]
            return str(openai.__version__)
        except Exception:
            return None


class AnthropicShimContract(ShimContract):
    """ShimContract for the Anthropic ``messages.create`` surface."""

    provider_name: ClassVar[str] = "anthropic"

    def canonical_request(self, **kwargs: Any) -> List[dict]:
        """Prepend system prompt, then append user/assistant messages."""
        messages = list(kwargs.get("messages") or [])
        system = kwargs.get("system")
        unified: List[dict] = []
        if system:
            unified.append({"role": "system", "content": system})
        unified.extend(messages)
        return unified

    def canonical_response(self, native: Any) -> dict:
        """Delegate to :func:`_anthropic_to_openai_shape` via coercion."""
        return _anthropic_to_openai_shape(_coerce_anthropic_response(native))

    def make_executor(self, client: Any) -> Callable[[str, List[dict]], dict]:
        """Return an :func:`anthropic_executor` for *client*."""
        return anthropic_executor(client)

    def stream_request(self, client: Any, messages: List[dict], **kwargs: Any) -> dict:
        """Call ``client.messages.create(stream=True)`` and accumulate events.

        Returns a canonical ``chat.completion``-shaped dict assembled from the
        full stream — individual events are not exposed since stepback records
        the *final* response as a single trace step.
        """
        anth_messages = [m for m in messages if m.get("role") != "system"]
        anth_system = next((m["content"] for m in messages if m.get("role") == "system"), None)
        model = kwargs.get("model")
        max_tokens = int(kwargs.get("max_tokens", 1024))
        extra = {k: v for k, v in kwargs.items()
                 if k not in ("model", "messages", "system", "max_tokens")}
        stream = client.messages.create(
            model=model,
            messages=anth_messages,
            system=anth_system,
            max_tokens=max_tokens,
            stream=True,
            **extra,
        )
        raw_dict = _accumulate_anthropic_streaming_chunks(stream)
        return self.canonical_response(raw_dict)

    async def async_request(self, client: Any, messages: List[dict], **kwargs: Any) -> dict:
        """Await ``client.messages.create(...)`` and return a canonical dict."""
        anth_messages = [m for m in messages if m.get("role") != "system"]
        anth_system = next((m["content"] for m in messages if m.get("role") == "system"), None)
        model = kwargs.get("model")
        max_tokens = int(kwargs.get("max_tokens", 1024))
        extra = {k: v for k, v in kwargs.items()
                 if k not in ("model", "messages", "system", "max_tokens")}
        resp = await client.messages.create(
            model=model,
            messages=anth_messages,
            system=anth_system,
            max_tokens=max_tokens,
            **extra,
        )
        return self.canonical_response(resp)

    def version_probe(self, client: Any) -> Optional[str]:
        try:
            import anthropic  # type: ignore[import-not-found]
            return str(anthropic.__version__)
        except Exception:
            return None


class BedrockShimContract(ShimContract):
    """ShimContract for the AWS Bedrock ``converse`` surface."""

    provider_name: ClassVar[str] = "bedrock"

    def canonical_request(self, **kwargs: Any) -> List[dict]:
        """Delegate to :func:`_bedrock_messages_to_unified`."""
        return _bedrock_messages_to_unified(
            kwargs.get("messages") or [], kwargs.get("system")
        )

    def canonical_response(self, native: Any) -> dict:
        """Delegate to :func:`_bedrock_to_openai_shape` via coercion."""
        return _bedrock_to_openai_shape(_coerce_bedrock_response(native))

    def make_executor(self, client: Any) -> Callable[[str, List[dict]], dict]:
        """Return a :func:`bedrock_executor` for *client*."""
        return bedrock_executor(client)

    def version_probe(self, client: Any) -> Optional[str]:
        try:
            import boto3  # type: ignore[import-not-found]
            return str(boto3.__version__)
        except Exception:
            return None


class GeminiShimContract(ShimContract):
    """ShimContract for the Google Gemini ``models.generate_content`` surface."""

    provider_name: ClassVar[str] = "gemini"

    def canonical_request(self, **kwargs: Any) -> List[dict]:
        """Delegate to :func:`_gemini_contents_to_unified`."""
        return _gemini_contents_to_unified(
            kwargs.get("contents"), kwargs.get("system_instruction")
        )

    def canonical_response(self, native: Any) -> dict:
        """Delegate to :func:`_gemini_to_openai_shape` via coercion.

        *native* may be the raw SDK response object or an already-coerced
        dict; :func:`_coerce_gemini_response` handles both.
        """
        return _gemini_to_openai_shape(_coerce_gemini_response(native))

    def make_executor(self, client: Any) -> Callable[[str, List[dict]], dict]:
        """Return a :func:`gemini_executor` for *client*."""
        return gemini_executor(client)

    def version_probe(self, client: Any) -> Optional[str]:
        try:
            from google import genai  # type: ignore[import-not-found]
            return str(getattr(genai, "__version__", None))
        except Exception:
            return None


# =====================================================================
# Cohere shim  (cohere Python SDK v2 — client.chat(...))
# =====================================================================


import json as _json


@dataclass
class CohereMessage:
    """Lightweight namespace mirroring the Cohere v2 ``NonStreamedChatResponse``.

    Agents reading ``resp.message.content[0].text`` /
    ``resp.message.tool_calls`` / ``resp.finish_reason`` keep working
    unmodified after wrapping.
    """

    id: str
    finish_reason: Optional[str]
    message: dict  # {"role": str, "content": [...], "tool_calls": [...] | None}
    usage: dict    # {"billed_units": {...}, "tokens": {...}}
    raw: dict = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CohereMessage":
        msg = d.get("message") or {}
        # Ensure content / tool_calls are plain lists
        if not isinstance(msg, dict):
            msg = {}
        return cls(
            id=str(d.get("id", "")),
            finish_reason=d.get("finish_reason"),
            message=dict(msg),
            usage=dict(d.get("usage") or {}),
            raw=dict(d),
        )


# Normalise Cohere finish reasons → OpenAI finish reasons.
_COHERE_FINISH_REASON_MAP: Dict[str, str] = {
    "COMPLETE": "stop",
    "MAX_TOKENS": "length",
    "TOOL_CALL": "tool_calls",
    "ERROR": "error",
    "ERROR_LIMIT": "error",
    "ERROR_TOXIC": "content_filter",
}


def _coerce_cohere_response(resp: Any) -> dict:
    """Coerce a Cohere SDK response object to a plain dict.

    Handles SDK typed objects (attribute-based), mappings, and objects
    with ``model_dump`` / ``to_dict``.
    """
    if isinstance(resp, CohereMessage):
        return resp.raw or {
            "id": resp.id,
            "finish_reason": resp.finish_reason,
            "message": resp.message,
            "usage": resp.usage,
        }
    if isinstance(resp, Mapping):
        return dict(resp)
    if hasattr(resp, "model_dump"):
        return resp.model_dump()  # type: ignore[no-any-return]
    if hasattr(resp, "to_dict"):
        return resp.to_dict()  # type: ignore[no-any-return]
    # Cohere SDK objects expose fields as attributes; extract them manually.
    try:
        message_obj = getattr(resp, "message", None)
        msg_dict: dict = {}
        if message_obj is not None:
            if isinstance(message_obj, Mapping):
                msg_dict = dict(message_obj)
            else:
                # Extract role, content, tool_calls from the message object.
                role = getattr(message_obj, "role", "assistant")
                content_obj = getattr(message_obj, "content", None) or []
                if not isinstance(content_obj, list):
                    content_obj = [content_obj]
                content_list: List[dict] = []
                for blk in content_obj:
                    if isinstance(blk, Mapping):
                        content_list.append(dict(blk))
                    elif hasattr(blk, "text"):
                        content_list.append({"type": "text", "text": str(blk.text)})
                    else:
                        content_list.append({"type": "text", "text": str(blk)})
                tc_obj = getattr(message_obj, "tool_calls", None)
                tool_calls_list: Optional[List[dict]] = None
                if tc_obj:
                    tool_calls_list = []
                    for tc in tc_obj:
                        if isinstance(tc, Mapping):
                            tool_calls_list.append(dict(tc))
                        else:
                            fn = getattr(tc, "function", None) or {}
                            fn_name = (
                                fn.get("name") if isinstance(fn, Mapping)
                                else getattr(fn, "name", "")
                            ) or ""
                            fn_args = (
                                fn.get("arguments") if isinstance(fn, Mapping)
                                else getattr(fn, "arguments", {})
                            ) or {}
                            tool_calls_list.append({
                                "id": str(getattr(tc, "id", "") or ""),
                                "type": "function",
                                "function": {"name": fn_name, "arguments": fn_args},
                            })
                msg_dict = {
                    "role": str(role),
                    "content": content_list,
                    "tool_calls": tool_calls_list,
                }

        usage_obj = getattr(resp, "usage", None)
        usage_dict: dict = {}
        if usage_obj is not None:
            if isinstance(usage_obj, Mapping):
                usage_dict = dict(usage_obj)
            else:
                bu = getattr(usage_obj, "billed_units", None) or {}
                tok = getattr(usage_obj, "tokens", None) or {}
                usage_dict = {
                    "billed_units": (
                        dict(bu) if isinstance(bu, Mapping)
                        else {
                            "input_tokens": int(getattr(bu, "input_tokens", 0) or 0),
                            "output_tokens": int(getattr(bu, "output_tokens", 0) or 0),
                        }
                    ),
                    "tokens": (
                        dict(tok) if isinstance(tok, Mapping)
                        else {
                            "input_tokens": int(getattr(tok, "input_tokens", 0) or 0),
                            "output_tokens": int(getattr(tok, "output_tokens", 0) or 0),
                        }
                    ),
                }

        return {
            "id": str(getattr(resp, "id", "") or ""),
            "finish_reason": getattr(resp, "finish_reason", None),
            "message": msg_dict,
            "usage": usage_dict,
        }
    except Exception as exc:
        raise TypeError(
            f"unsupported Cohere response type: {type(resp).__name__}; "
            "expected CohereMessage / dict / mapping or Cohere SDK response"
        ) from exc


def _cohere_to_openai_shape(d: Mapping[str, Any]) -> dict:
    """Project a Cohere chat response dict into the OpenAI chat-completion shape.

    * ``message.content`` (list of blocks) → ``message.content`` (str)
    * ``message.tool_calls`` → OpenAI-style ``tool_calls`` list with JSON-string arguments
    * ``finish_reason`` (Cohere uppercase enum) → OpenAI lowercase string
    * ``usage.billed_units`` → ``usage.{prompt,completion,total}_tokens``
    """
    msg = d.get("message") or {}
    if not isinstance(msg, Mapping):
        msg = {}

    # Extract text content by concatenating text blocks.
    content_blocks = msg.get("content") or []
    if isinstance(content_blocks, str):
        content_text: Optional[str] = content_blocks or None
    else:
        parts = [
            blk.get("text", "") if isinstance(blk, Mapping) else str(blk)
            for blk in content_blocks
            if (isinstance(blk, Mapping) and blk.get("type") in (None, "text", "TEXT"))
            or not isinstance(blk, Mapping)
        ]
        content_text = "".join(parts) or None

    # Extract tool calls.
    tc_raw = msg.get("tool_calls")
    tool_calls: Optional[List[dict]] = None
    if tc_raw:
        tool_calls = []
        for tc in tc_raw:
            if not isinstance(tc, Mapping):
                continue
            fn = tc.get("function") or {}
            if not isinstance(fn, Mapping):
                fn = {}
            fn_args = fn.get("arguments", {})
            # Canonical form: JSON string for arguments.
            if isinstance(fn_args, dict):
                fn_args = _json.dumps(fn_args, sort_keys=True, ensure_ascii=False)
            tool_calls.append({
                "id": str(tc.get("id", "")),
                "type": "function",
                "function": {
                    "name": str(fn.get("name", "")),
                    "arguments": fn_args,
                },
            })
        if not tool_calls:
            tool_calls = None

    raw_finish = d.get("finish_reason") or ""
    if isinstance(raw_finish, str):
        finish_reason = _COHERE_FINISH_REASON_MAP.get(raw_finish.upper(), raw_finish.lower() or None)
    else:
        finish_reason = None

    # Use billed_units for cost accounting; fall back to tokens dict.
    usage_raw = d.get("usage") or {}
    if not isinstance(usage_raw, Mapping):
        usage_raw = {}
    bu = usage_raw.get("billed_units") or usage_raw.get("tokens") or {}
    if not isinstance(bu, Mapping):
        bu = {}
    prompt_tokens = int(bu.get("input_tokens", 0) or 0)
    completion_tokens = int(bu.get("output_tokens", 0) or 0)

    return _strip_none({
        "id": d.get("id"),
        "model": d.get("model"),
        "choices": [{
            "index": 0,
            "finish_reason": finish_reason,
            "message": {
                "role": str(msg.get("role", "assistant")),
                "content": content_text,
                "tool_calls": tool_calls,
            },
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "_cohere": dict(d),
    })


def canonical_cohere_model_id(model_id: str) -> str:
    """Return the stepback canonical pricing key for a Cohere model id.

    Passes through dated ids unchanged; resolves ``command-r-plus-latest``
    and similar ``-latest`` aliases to their concrete dated rows.
    """
    return _COHERE_MODEL_ALIASES.get(model_id, model_id)


#: Cohere model aliases → dated pricing row.
_COHERE_MODEL_ALIASES: Dict[str, str] = {
    "command-r-plus-latest": "command-r-plus-08-2024",
    "command-r-latest": "command-r-08-2024",
    "command-r-plus": "command-r-plus-08-2024",
    "command-r": "command-r-08-2024",
    "command": "command-r-08-2024",
    "command-a-03-2025": "command-a-03-2025",
}


class _CohereProxy:
    """Proxy for ``cohere.ClientV2.chat(...)``."""

    def __init__(self, real: Any, recorder: Recorder,
                 seed_policy: Optional[SeedPolicy] = None,
                 contract: Optional["ShimContract"] = None) -> None:
        self._real = real
        self._rec = recorder
        self._seed_policy = seed_policy or get_seed_policy()
        self._contract = contract

    def _get_contract(self) -> "ShimContract":
        if self._contract is not None:
            return self._contract
        return shim_contract_for("cohere")

    def __call__(self, *, model: str, messages: List[dict], **kwargs: Any) -> CohereMessage:
        contract = self._get_contract()
        # Pass preamble (and any other kwarg) to canonical_request so the
        # CohereShimContract can promote it to a leading system message.
        unified = contract.canonical_request(messages=messages, **kwargs)
        temperature = float(kwargs.get("temperature", 0.0))
        # Cohere v2 does not support a reproducibility seed; note in policy.
        effective_seed = self._seed_policy.check("cohere", None, temperature, model=model)
        canonical_model = canonical_cohere_model_id(model)

        def executor(_model: str, _messages: List[dict]) -> dict:
            # Reconstruct native Cohere messages from unified format.
            cohere_messages = [
                m for m in _messages if m.get("role") != "system"
            ]
            cohere_system = next(
                (m.get("content", "") for m in _messages if m.get("role") == "system"),
                None,
            )
            call_kwargs = dict(kwargs)
            if cohere_system:
                call_kwargs.setdefault("preamble", cohere_system)
            resp = self._real(model=model, messages=cohere_messages, **call_kwargs)
            coerced = _coerce_cohere_response(resp)
            coerced.setdefault("model", model)
            return contract.canonical_response(coerced)

        step = self._rec.llm_call(
            model=canonical_model,
            messages=unified,
            executor=executor,
            temperature=temperature,
            seed=effective_seed,
            tools=kwargs.get("tools"),
            response_format=None,
        )
        native = step["llm_response"].get("_cohere")
        if native:
            return CohereMessage.from_dict(native)
        # Synthesise minimal native shape from canonical fields.
        ch = step["llm_response"]["choices"][0]
        usage = step["llm_response"].get("usage", {})
        synth: dict = {
            "id": step["llm_response"].get("id", ""),
            "finish_reason": ch.get("finish_reason"),
            "message": ch.get("message", {}),
            "usage": {
                "billed_units": {
                    "input_tokens": int(usage.get("prompt_tokens", 0)),
                    "output_tokens": int(usage.get("completion_tokens", 0)),
                },
                "tokens": {
                    "input_tokens": int(usage.get("prompt_tokens", 0)),
                    "output_tokens": int(usage.get("completion_tokens", 0)),
                },
            },
        }
        return CohereMessage.from_dict(synth)


@dataclass
class WrappedCohere:
    """Drop-in replacement for ``cohere.ClientV2``.

    Only ``client.chat(...)`` is intercepted; every other attribute is
    passed through to the real client.
    """

    chat: _CohereProxy
    _real: Any
    _rec: Recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def wrap_cohere(client: Any, recorder: Recorder,
                seed_policy: Optional[SeedPolicy] = None,
                contract: Optional["ShimContract"] = None) -> WrappedCohere:
    """Wrap a real Cohere v2 client so ``chat(...)`` calls are recorded.

    The wrapped object exposes ``client.chat(model=..., messages=..., ...)``
    and returns a :class:`CohereMessage` namespace. Internally each call is
    canonicalised into the OpenAI chat-completion shape shared by every other
    provider shim.

    *seed_policy* defaults to the module-level
    :data:`~stepback.seeding.DEFAULT_SEED_POLICY`.  Cohere v2 does not expose
    a seed parameter, so a :class:`~stepback.seeding.SeedPolicyViolation`
    warning is emitted unless the policy is silenced.

    The optional *contract* parameter accepts a :class:`ShimContract` instance
    to override the built-in :class:`CohereShimContract`.

    Raises :class:`TypeError` if the client's ``.chat`` attribute is not
    callable (i.e. isn't a Cohere v2-shaped client).

    Example::

        import cohere
        from stepback import record
        from stepback.shims import wrap_cohere

        co = cohere.ClientV2(api_key="...")
        with record("./trace.sb") as rec:
            client = wrap_cohere(co, rec)
            resp = client.chat(
                model="command-r-plus-08-2024",
                messages=[{"role": "user", "content": "Hello"}],
            )
            print(resp.message.content[0].text)
    """
    if not callable(getattr(client, "chat", None)):
        raise TypeError(
            "wrap_cohere: client.chat is not callable; "
            "expected a cohere.ClientV2-shaped object"
        )
    proxy = _CohereProxy(client.chat, recorder, seed_policy=seed_policy, contract=contract)
    return WrappedCohere(chat=proxy, _real=client, _rec=recorder)


def cohere_executor(client: Any) -> Callable[[str, List[dict]], dict]:
    """Adapter so a real Cohere v2 client can serve dirty replay steps.

    Returns a callable shaped ``(model, messages) -> canonical_dict``.
    The *messages* list uses the unified OpenAI-style format; this function
    converts to Cohere native format before calling the real API.
    """
    def _llm(model: str, messages: List[dict]) -> dict:
        cohere_messages = [m for m in messages if m.get("role") != "system"]
        cohere_system = next(
            (m.get("content", "") for m in messages if m.get("role") == "system"),
            None,
        )
        call_kwargs: dict = {}
        if cohere_system:
            call_kwargs["preamble"] = cohere_system
        resp = client.chat(model=model, messages=cohere_messages, **call_kwargs)
        coerced = _coerce_cohere_response(resp)
        coerced.setdefault("model", model)
        return _cohere_to_openai_shape(coerced)

    return _llm


class CohereShimContract(ShimContract):
    """ShimContract for the Cohere v2 ``chat(...)`` surface."""

    provider_name: ClassVar[str] = "cohere"

    def canonical_request(self, **kwargs: Any) -> List[dict]:
        """Convert Cohere request to unified message list.

        Cohere v2 uses the same role/content message schema as OpenAI for
        the ``messages`` list.  A system prompt may arrive as a ``preamble``
        kwarg or as a ``{"role": "system", ...}`` message; both are normalised
        to a leading system message in the unified list.
        """
        messages = list(kwargs.get("messages") or [])
        preamble = kwargs.get("preamble")
        unified: List[dict] = []
        # Honour an explicit preamble kwarg as the system message.
        if preamble:
            unified.append({"role": "system", "content": str(preamble)})
        # Pass through messages; system-role entries already present are kept.
        unified.extend(messages)
        return unified

    def canonical_response(self, native: Any) -> dict:
        """Delegate to :func:`_cohere_to_openai_shape` via coercion."""
        return _cohere_to_openai_shape(_coerce_cohere_response(native))

    def make_executor(self, client: Any) -> Callable[[str, List[dict]], dict]:
        """Return a :func:`cohere_executor` for *client*."""
        return cohere_executor(client)

    def version_probe(self, client: Any) -> Optional[str]:
        try:
            import cohere  # type: ignore[import-not-found]
            return str(getattr(cohere, "__version__", None))
        except Exception:
            return None


# =====================================================================
# Mistral shim  (mistralai Python SDK — client.chat.complete(...))
# =====================================================================


@dataclass
class MistralChatResponse:
    """Lightweight namespace mirroring the Mistral ``ChatCompletion`` response.

    The Mistral SDK uses an OpenAI-compatible response shape so agent code
    reading ``resp.choices[0].message.content`` keeps working.
    """

    id: str
    model: str
    choices: List[_OAIChoice]
    usage: dict
    raw: dict = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MistralChatResponse":
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


def _coerce_mistral_response(resp: Any) -> dict:
    """Coerce a Mistral SDK response to a plain dict.

    The Mistral SDK response is OpenAI-shaped so this is thin.
    """
    if isinstance(resp, MistralChatResponse):
        return resp.raw or {
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
        }
    if isinstance(resp, Mapping):
        return _strip_none(dict(resp))
    if hasattr(resp, "model_dump"):
        return resp.model_dump()  # type: ignore[no-any-return]
    if hasattr(resp, "to_dict"):
        return resp.to_dict()  # type: ignore[no-any-return]
    raise TypeError(
        f"unsupported Mistral response type: {type(resp).__name__}; "
        "expected ChatCompletion / dict / mapping or Mistral SDK response"
    )


def _mistral_to_openai_shape(d: Mapping[str, Any]) -> dict:
    """Project a Mistral response dict into the canonical OpenAI shape.

    The Mistral SDK already uses the OpenAI wire format, so this is
    essentially an identity mapping with a ``_mistral`` provenance tag
    and tool-call argument normalisation to JSON strings.
    """
    choices_raw = d.get("choices") or []
    choices_out: List[dict] = []
    for i, ch in enumerate(choices_raw):
        if not isinstance(ch, Mapping):
            continue
        msg = ch.get("message") or {}
        if not isinstance(msg, Mapping):
            msg = {}
        tc_raw = msg.get("tool_calls")
        tool_calls: Optional[List[dict]] = None
        if tc_raw:
            tool_calls = []
            for tc in tc_raw:
                if not isinstance(tc, Mapping):
                    continue
                fn = tc.get("function") or {}
                if not isinstance(fn, Mapping):
                    fn = {}
                fn_args = fn.get("arguments", "")
                # Normalise dict arguments to JSON string.
                if isinstance(fn_args, dict):
                    fn_args = _json.dumps(fn_args, sort_keys=True, ensure_ascii=False)
                tool_calls.append({
                    "id": str(tc.get("id", "")),
                    "type": str(tc.get("type", "function")),
                    "function": {
                        "name": str(fn.get("name", "")),
                        "arguments": fn_args,
                    },
                })
            if not tool_calls:
                tool_calls = None
        choices_out.append({
            "index": int(ch.get("index", i)),
            "finish_reason": ch.get("finish_reason"),
            "message": {
                "role": str(msg.get("role", "assistant")),
                "content": msg.get("content"),
                "tool_calls": tool_calls,
            },
        })

    usage_raw = d.get("usage") or {}
    if not isinstance(usage_raw, Mapping):
        usage_raw = {}
    canonical_usage = {
        "prompt_tokens": int(usage_raw.get("prompt_tokens", 0)),
        "completion_tokens": int(usage_raw.get("completion_tokens", 0)),
        "total_tokens": int(usage_raw.get("total_tokens", 0)),
    }

    return _strip_none({
        "id": d.get("id"),
        "model": d.get("model"),
        "choices": choices_out,
        "usage": canonical_usage,
        "_mistral": dict(d),
    })


def canonical_mistral_model_id(model_id: str) -> str:
    """Return the stepback canonical pricing key for a Mistral model id.

    Resolves ``*-latest`` aliases to dated model rows; passes unknown ids through.
    """
    return _MISTRAL_MODEL_ALIASES.get(model_id, model_id)


#: Mistral model aliases → dated pricing row.
_MISTRAL_MODEL_ALIASES: Dict[str, str] = {
    "mistral-large-latest": "mistral-large-2411",
    "mistral-small-latest": "mistral-small-2501",
    "codestral-latest": "codestral-2501",
    "open-mistral-nemo-latest": "open-mistral-nemo",
    "mistral-large": "mistral-large-2411",
    "mistral-small": "mistral-small-2501",
    "codestral": "codestral-2501",
}


class _MistralChatCompletionsProxy:
    """Proxy for ``mistral.Mistral().chat.complete(...)``."""

    def __init__(self, real: Any, recorder: Recorder, *,
                 seed_policy: Optional[SeedPolicy] = None,
                 contract: Optional["ShimContract"] = None) -> None:
        self._real = real
        self._rec = recorder
        self._seed_policy = seed_policy or get_seed_policy()
        self._contract = contract

    def _get_contract(self) -> "ShimContract":
        if self._contract is not None:
            return self._contract
        return shim_contract_for("mistral")

    def complete(self, *, model: str, messages: List[dict],
                 **kwargs: Any) -> MistralChatResponse:
        contract = self._get_contract()
        unified = contract.canonical_request(messages=list(messages))
        temperature = float(kwargs.get("temperature", 0.0))
        caller_seed = kwargs.get("random_seed") or kwargs.get("seed")
        # Mistral uses ``random_seed``; map to the unified seed policy.
        effective_seed = self._seed_policy.check(
            "mistral", caller_seed, temperature, model=model
        )
        api_kwargs = dict(kwargs)
        if effective_seed is not None:
            api_kwargs.setdefault("random_seed", effective_seed)
        canonical_model = canonical_mistral_model_id(model)

        def executor(_model: str, _messages: List[dict]) -> dict:
            resp = self._real.complete(model=model, messages=_messages, **api_kwargs)
            coerced = _coerce_mistral_response(resp)
            coerced.setdefault("model", model)
            return contract.canonical_response(coerced)

        step = self._rec.llm_call(
            model=canonical_model,
            messages=unified,
            executor=executor,
            temperature=temperature,
            seed=effective_seed,
            tools=kwargs.get("tools"),
            response_format=kwargs.get("response_format"),
        )
        native = step["llm_response"].get("_mistral")
        if native:
            return MistralChatResponse.from_dict(native)
        return MistralChatResponse.from_dict(step["llm_response"])


class _MistralChatProxy:
    def __init__(self, real: Any, recorder: Recorder, *,
                 seed_policy: Optional[SeedPolicy] = None,
                 contract: Optional["ShimContract"] = None) -> None:
        self.complete = _MistralChatCompletionsProxy(
            real, recorder, seed_policy=seed_policy, contract=contract
        ).complete


@dataclass
class WrappedMistral:
    """Drop-in replacement for ``mistralai.Mistral``.

    Only ``client.chat.complete(...)`` is intercepted; every other attribute
    is passed through to the real client.
    """

    chat: _MistralChatProxy
    _real: Any
    _rec: Recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def wrap_mistral(client: Any, recorder: Recorder,
                 seed_policy: Optional[SeedPolicy] = None,
                 contract: Optional["ShimContract"] = None) -> WrappedMistral:
    """Wrap a real Mistral client so ``chat.complete(...)`` calls are recorded.

    The wrapped object exposes ``client.chat.complete(model=..., messages=..., ...)``
    and returns a :class:`MistralChatResponse` namespace mirroring the OpenAI
    shape.  Internally each call is canonicalised into the unified
    chat-completion shape shared by every other provider shim.

    *seed_policy* defaults to :data:`~stepback.seeding.DEFAULT_SEED_POLICY`.
    Mistral uses ``random_seed`` rather than ``seed``; the shim maps the
    effective seed to both the trace header and the ``random_seed`` API kwarg.

    The optional *contract* parameter accepts a :class:`ShimContract` instance
    to override the built-in :class:`MistralShimContract`.

    Raises :class:`TypeError` if the client lacks a ``.chat.complete``
    callable (i.e. isn't a Mistral SDK-shaped object).

    Example::

        from mistralai import Mistral
        from stepback import record
        from stepback.shims import wrap_mistral

        with record("./trace.sb") as rec:
            client = wrap_mistral(Mistral(api_key="..."), rec)
            resp = client.chat.complete(
                model="mistral-large-2411",
                messages=[{"role": "user", "content": "Hello"}],
            )
            print(resp.choices[0].message.content)
    """
    chat_obj = getattr(client, "chat", None)
    if chat_obj is None or not callable(getattr(chat_obj, "complete", None)):
        raise TypeError(
            "wrap_mistral: client.chat.complete is not callable; "
            "expected a mistralai.Mistral-shaped object"
        )
    proxy = _MistralChatProxy(chat_obj, recorder, seed_policy=seed_policy, contract=contract)
    return WrappedMistral(chat=proxy, _real=client, _rec=recorder)


def mistral_executor(client: Any) -> Callable[[str, List[dict]], dict]:
    """Adapter so a real Mistral client can serve dirty replay steps.

    Returns a callable shaped ``(model, messages) -> canonical_dict``.
    """
    def _llm(model: str, messages: List[dict]) -> dict:
        resp = client.chat.complete(model=model, messages=messages)
        coerced = _coerce_mistral_response(resp)
        coerced.setdefault("model", model)
        return _mistral_to_openai_shape(coerced)

    return _llm


class MistralShimContract(ShimContract):
    """ShimContract for the Mistral ``chat.complete(...)`` surface."""

    provider_name: ClassVar[str] = "mistral"

    def canonical_request(self, **kwargs: Any) -> List[dict]:
        """Mistral uses the OpenAI message format; return messages as-is."""
        return list(kwargs.get("messages") or [])

    def canonical_response(self, native: Any) -> dict:
        """Delegate to :func:`_mistral_to_openai_shape` via coercion."""
        return _mistral_to_openai_shape(_coerce_mistral_response(native))

    def make_executor(self, client: Any) -> Callable[[str, List[dict]], dict]:
        """Return a :func:`mistral_executor` for *client*."""
        return mistral_executor(client)

    def version_probe(self, client: Any) -> Optional[str]:
        try:
            import mistralai  # type: ignore[import-not-found]
            return str(getattr(mistralai, "__version__", None))
        except Exception:
            return None


# =====================================================================
# OpenAI-compatible provider shims (Step 96)
# =====================================================================
# Together AI, Fireworks, Groq, Cerebras, NVIDIA NIM, vLLM, TGI,
# llama.cpp, and Ollama all expose an OpenAI-compatible
# ``client.chat.completions.create(...)`` surface.  A single generic
# proxy handles the wire format; provider-specific subcontracts set
# the correct ``provider_name`` so seed policies, cost accounting, and
# trace annotations use the right provider identifier.
#
# Note: these wrappers accept *any* OpenAI SDK-shaped client configured
# for the provider's endpoint (e.g. ``openai.OpenAI(base_url="http://localhost:11434/v1")``
# for Ollama).  Canonical model IDs are stored as-is; users who want
# cost accounting should extend :data:`stepback.pricing.RATE_TABLE`.





class _OpenAICompatChatCompletionsProxy:
    """Generic proxy for any OpenAI-wire-compatible ``chat.completions.create``.

    Identical to :class:`_OAIChatCompletionsProxy` but the *provider_name*
    parameter drives seed-support lookup and trace annotations.
    """

    def __init__(
        self,
        real: Any,
        recorder: Recorder,
        *,
        provider_name: str,
        default_model: Optional[str] = None,
        seed_policy: Optional[SeedPolicy] = None,
        contract: Optional["ShimContract"] = None,
    ) -> None:
        self._real = real
        self._rec = recorder
        self._provider_name = provider_name
        self._default_model = default_model
        self._seed_policy = seed_policy or get_seed_policy()
        self._contract = contract

    def _get_contract(self) -> "ShimContract":
        if self._contract is not None:
            return self._contract
        return shim_contract_for(self._provider_name)

    def create(
        self,
        *,
        messages: List[dict],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> OpenAIChatCompletion:
        chosen_model = model or self._default_model
        if chosen_model is None:
            raise ValueError(
                f"wrap_{self._provider_name}: no model given and no default_model set"
            )
        temperature = float(kwargs.get("temperature", 0.0))
        caller_seed = kwargs.get("seed")
        effective_seed = self._seed_policy.check(
            self._provider_name, caller_seed, temperature, model=chosen_model
        )
        api_kwargs = dict(kwargs)
        # Only pass seed to the provider API when it is supported; NONE
        # providers reject unknown parameters or simply ignore them.
        support = self._seed_policy.seed_support(self._provider_name)
        if effective_seed is not None and support is not SeedSupport.NONE:
            api_kwargs["seed"] = effective_seed
        elif "seed" in api_kwargs and support is SeedSupport.NONE:
            del api_kwargs["seed"]

        contract = self._get_contract()
        is_stream = bool(api_kwargs.get("stream"))

        if is_stream:
            _raw_chunks: List[Any] = []

            def executor(_model: str, _messages: List[dict]) -> dict:
                stream = self._real.create(model=_model, messages=_messages, **api_kwargs)
                for chunk in stream:
                    _raw_chunks.append(chunk)
                assembled = _accumulate_streaming_chunks(iter(_raw_chunks))
                return contract.canonical_response(assembled)
        else:
            def executor(_model: str, _messages: List[dict]) -> dict:  # type: ignore[no-redef]
                resp = self._real.create(model=_model, messages=_messages, **api_kwargs)
                return contract.canonical_response(resp)

        step = self._rec.llm_call(
            model=chosen_model,
            messages=contract.canonical_request(messages=list(messages)),
            executor=executor,
            temperature=temperature,
            seed=effective_seed,
            tools=kwargs.get("tools"),
            response_format=kwargs.get("response_format"),
        )
        if is_stream:
            native = OpenAIChatCompletion.from_dict(step["llm_response"])
            return StreamedLLMResponse(  # type: ignore[return-value]
                _raw_chunks, step["llm_response"], step, native=native
            )
        return OpenAIChatCompletion.from_dict(step["llm_response"])


class _OpenAICompatChatProxy:
    def __init__(
        self,
        real: Any,
        recorder: Recorder,
        *,
        provider_name: str,
        default_model: Optional[str] = None,
        seed_policy: Optional[SeedPolicy] = None,
        contract: Optional["ShimContract"] = None,
    ) -> None:
        self.completions = _OpenAICompatChatCompletionsProxy(
            real.completions,
            recorder,
            provider_name=provider_name,
            default_model=default_model,
            seed_policy=seed_policy,
            contract=contract,
        )


@dataclass
class WrappedOpenAICompat:
    """Drop-in replacement for any OpenAI-wire-compatible client.

    ``client.chat.completions.create(...)`` is intercepted and recorded.
    All other attributes are passed through to the real client.

    The :attr:`provider_name` slot identifies which provider was wrapped
    (e.g. ``"groq"``, ``"together"``, ``"ollama"``).
    """

    chat: _OpenAICompatChatProxy
    _real: Any
    _rec: Recorder
    provider_name: str

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def _wrap_openai_compat(
    client: Any,
    recorder: Recorder,
    *,
    provider_name: str,
    default_model: Optional[str] = None,
    seed_policy: Optional[SeedPolicy] = None,
    contract: Optional["ShimContract"] = None,
) -> WrappedOpenAICompat:
    """Internal helper: wrap an OpenAI-compatible client for *provider_name*."""
    if not hasattr(client, "chat") or not hasattr(getattr(client, "chat", None), "completions"):
        raise TypeError(
            f"wrap_{provider_name}: client lacks .chat.completions; "
            "expected an openai.OpenAI-shaped client configured for "
            f"{provider_name!r}"
        )
    chat_proxy = _OpenAICompatChatProxy(
        client.chat,
        recorder,
        provider_name=provider_name,
        default_model=default_model,
        seed_policy=seed_policy,
        contract=contract,
    )
    return WrappedOpenAICompat(
        chat=chat_proxy,
        _real=client,
        _rec=recorder,
        provider_name=provider_name,
    )


def openai_compat_executor(
    client: Any, *, provider_name: str = "openai_compat"
) -> Callable[[str, List[dict]], dict]:
    """Generic replay executor for any OpenAI-compatible client.

    Returns a callable shaped ``(model, messages) -> canonical_dict``.
    """
    def _llm(model: str, messages: List[dict]) -> dict:
        resp = client.chat.completions.create(model=model, messages=messages)
        contract = shim_contract_for(provider_name)
        return contract.canonical_response(resp)

    return _llm


# ------------------------------------------------------------------ #
# OpenAICompatShimContract — base class shared by all OAI-compat      #
# providers.  Subclasses only need to set provider_name.              #
# ------------------------------------------------------------------ #


class OpenAICompatShimContract(ShimContract):
    """Base :class:`ShimContract` for OpenAI-wire-compatible providers.

    Subclasses must set :attr:`provider_name`.  The request/response
    canonicalisation delegates to the same helpers used by
    :class:`OpenAIShimContract` because the wire format is identical.
    """

    provider_name: ClassVar[str]  # must be set by subclass

    def canonical_request(self, **kwargs: Any) -> List[dict]:
        """Return the ``messages`` list unchanged (OpenAI wire format)."""
        return list(kwargs.get("messages") or [])

    def canonical_response(self, native: Any) -> dict:
        """Delegate to :func:`_canonicalise_openai_response`."""
        return _canonicalise_openai_response(native)

    def make_executor(self, client: Any) -> Callable[[str, List[dict]], dict]:
        """Return a replay executor for *client*."""
        return openai_compat_executor(client, provider_name=self.provider_name)

    def version_probe(self, client: Any) -> Optional[str]:
        """Try to return the installed ``openai`` SDK version (used as compat client)."""
        try:
            import openai  # type: ignore[import-not-found]
            return str(getattr(openai, "__version__", None))
        except Exception:
            return None


# ------------------------------------------------------------------ #
# Per-provider ShimContracts                                           #
# ------------------------------------------------------------------ #


class GroqShimContract(OpenAICompatShimContract):
    """ShimContract for Groq (``groq`` SDK or openai client with Groq base_url)."""
    provider_name: ClassVar[str] = "groq"


class TogetherShimContract(OpenAICompatShimContract):
    """ShimContract for Together AI (``together`` SDK or openai-compat client)."""
    provider_name: ClassVar[str] = "together"


class FireworksShimContract(OpenAICompatShimContract):
    """ShimContract for Fireworks AI (``fireworks-ai`` SDK or openai-compat client)."""
    provider_name: ClassVar[str] = "fireworks"


class CerebrasShimContract(OpenAICompatShimContract):
    """ShimContract for Cerebras (openai-compat client; seed not supported)."""
    provider_name: ClassVar[str] = "cerebras"


class NvidiaNIMShimContract(OpenAICompatShimContract):
    """ShimContract for NVIDIA NIM (openai-compat client with NIM base_url)."""
    provider_name: ClassVar[str] = "nvidia_nim"


class VLLMShimContract(OpenAICompatShimContract):
    """ShimContract for vLLM (self-hosted OpenAI-compatible server)."""
    provider_name: ClassVar[str] = "vllm"


class TGIShimContract(OpenAICompatShimContract):
    """ShimContract for HuggingFace Text Generation Inference (OpenAI-compat mode)."""
    provider_name: ClassVar[str] = "tgi"


class LlamaCppShimContract(OpenAICompatShimContract):
    """ShimContract for llama.cpp server (OpenAI-compatible REST server)."""
    provider_name: ClassVar[str] = "llamacpp"


class OllamaShimContract(OpenAICompatShimContract):
    """ShimContract for Ollama (OpenAI-compatible ``/v1`` endpoint).

    Note: pass an ``openai.OpenAI(base_url="http://localhost:11434/v1")``
    client, not the native ``ollama`` Python SDK.
    """
    provider_name: ClassVar[str] = "ollama"


# ------------------------------------------------------------------ #
# Named wrap_* functions                                              #
# ------------------------------------------------------------------ #


def wrap_groq(
    client: Any,
    recorder: Recorder,
    *,
    default_model: Optional[str] = None,
    seed_policy: Optional[SeedPolicy] = None,
    contract: Optional["ShimContract"] = None,
) -> WrappedOpenAICompat:
    """Wrap a Groq client so ``chat.completions.create(...)`` calls are recorded.

    *client* may be either an official ``groq.Groq`` SDK client or an
    ``openai.OpenAI(base_url="https://api.groq.com/openai/v1")`` client.
    Both expose the OpenAI-wire format.

    Example::

        from groq import Groq
        from stepback import record
        from stepback.shims import wrap_groq

        with record("./trace.sb") as rec:
            client = wrap_groq(Groq(api_key="..."), rec)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "Hello"}],
            )
            print(resp.choices[0].message.content)
    """
    return _wrap_openai_compat(
        client, recorder, provider_name="groq",
        default_model=default_model, seed_policy=seed_policy, contract=contract,
    )


def groq_executor(client: Any) -> Callable[[str, List[dict]], dict]:
    """Adapter so a real Groq client can serve dirty replay steps.

    Returns a callable shaped ``(model, messages) -> canonical_dict``.
    """
    return openai_compat_executor(client, provider_name="groq")


def wrap_together(
    client: Any,
    recorder: Recorder,
    *,
    default_model: Optional[str] = None,
    seed_policy: Optional[SeedPolicy] = None,
    contract: Optional["ShimContract"] = None,
) -> WrappedOpenAICompat:
    """Wrap a Together AI client so ``chat.completions.create(...)`` calls are recorded.

    *client* may be either an official ``together.Together`` SDK client or an
    ``openai.OpenAI(base_url="https://api.together.xyz/v1")`` client.

    Example::

        from together import Together
        from stepback import record
        from stepback.shims import wrap_together

        with record("./trace.sb") as rec:
            client = wrap_together(Together(api_key="..."), rec)
            resp = client.chat.completions.create(
                model="meta-llama/Llama-3-8b-chat-hf",
                messages=[{"role": "user", "content": "Hello"}],
            )
    """
    return _wrap_openai_compat(
        client, recorder, provider_name="together",
        default_model=default_model, seed_policy=seed_policy, contract=contract,
    )


def together_executor(client: Any) -> Callable[[str, List[dict]], dict]:
    """Adapter so a real Together AI client can serve dirty replay steps."""
    return openai_compat_executor(client, provider_name="together")


def wrap_fireworks(
    client: Any,
    recorder: Recorder,
    *,
    default_model: Optional[str] = None,
    seed_policy: Optional[SeedPolicy] = None,
    contract: Optional["ShimContract"] = None,
) -> WrappedOpenAICompat:
    """Wrap a Fireworks AI client so ``chat.completions.create(...)`` calls are recorded.

    *client* may be either an official ``fireworks.client.Fireworks`` SDK client or an
    ``openai.OpenAI(base_url="https://api.fireworks.ai/inference/v1")`` client.

    Example::

        from fireworks.client import Fireworks
        from stepback import record
        from stepback.shims import wrap_fireworks

        with record("./trace.sb") as rec:
            client = wrap_fireworks(Fireworks(api_key="..."), rec)
            resp = client.chat.completions.create(
                model="accounts/fireworks/models/llama-v3-8b-instruct",
                messages=[{"role": "user", "content": "Hello"}],
            )
    """
    return _wrap_openai_compat(
        client, recorder, provider_name="fireworks",
        default_model=default_model, seed_policy=seed_policy, contract=contract,
    )


def fireworks_executor(client: Any) -> Callable[[str, List[dict]], dict]:
    """Adapter so a real Fireworks AI client can serve dirty replay steps."""
    return openai_compat_executor(client, provider_name="fireworks")


def wrap_cerebras(
    client: Any,
    recorder: Recorder,
    *,
    default_model: Optional[str] = None,
    seed_policy: Optional[SeedPolicy] = None,
    contract: Optional["ShimContract"] = None,
) -> WrappedOpenAICompat:
    """Wrap a Cerebras client so ``chat.completions.create(...)`` calls are recorded.

    Cerebras does not support a ``seed`` parameter; the shim will record the
    seed in trace metadata but will not pass it to the API.

    Example::

        from cerebras.cloud.sdk import Cerebras
        from stepback import record
        from stepback.shims import wrap_cerebras

        with record("./trace.sb") as rec:
            client = wrap_cerebras(Cerebras(api_key="..."), rec)
            resp = client.chat.completions.create(
                model="llama3.1-8b",
                messages=[{"role": "user", "content": "Hello"}],
            )
    """
    return _wrap_openai_compat(
        client, recorder, provider_name="cerebras",
        default_model=default_model, seed_policy=seed_policy, contract=contract,
    )


def cerebras_executor(client: Any) -> Callable[[str, List[dict]], dict]:
    """Adapter so a real Cerebras client can serve dirty replay steps."""
    return openai_compat_executor(client, provider_name="cerebras")


def wrap_nvidia_nim(
    client: Any,
    recorder: Recorder,
    *,
    default_model: Optional[str] = None,
    seed_policy: Optional[SeedPolicy] = None,
    contract: Optional["ShimContract"] = None,
) -> WrappedOpenAICompat:
    """Wrap a NVIDIA NIM client so ``chat.completions.create(...)`` calls are recorded.

    Pass an ``openai.OpenAI(base_url="https://integrate.api.nvidia.com/v1")``
    client or any other OpenAI-compatible client pointing at a NIM endpoint.

    Example::

        from openai import OpenAI
        from stepback import record
        from stepback.shims import wrap_nvidia_nim

        with record("./trace.sb") as rec:
            client = wrap_nvidia_nim(
                OpenAI(api_key="nvapi-...", base_url="https://integrate.api.nvidia.com/v1"),
                rec,
            )
            resp = client.chat.completions.create(
                model="meta/llama-3.1-8b-instruct",
                messages=[{"role": "user", "content": "Hello"}],
            )
    """
    return _wrap_openai_compat(
        client, recorder, provider_name="nvidia_nim",
        default_model=default_model, seed_policy=seed_policy, contract=contract,
    )


def nvidia_nim_executor(client: Any) -> Callable[[str, List[dict]], dict]:
    """Adapter so a real NVIDIA NIM client can serve dirty replay steps."""
    return openai_compat_executor(client, provider_name="nvidia_nim")


def wrap_vllm(
    client: Any,
    recorder: Recorder,
    *,
    default_model: Optional[str] = None,
    seed_policy: Optional[SeedPolicy] = None,
    contract: Optional["ShimContract"] = None,
) -> WrappedOpenAICompat:
    """Wrap a vLLM OpenAI-compatible client so ``chat.completions.create(...)`` calls are recorded.

    vLLM exposes an OpenAI-compatible server at ``http://localhost:8000/v1`` by default.
    Pass an ``openai.OpenAI(base_url="http://localhost:8000/v1")`` client.

    Example::

        from openai import OpenAI
        from stepback import record
        from stepback.shims import wrap_vllm

        with record("./trace.sb") as rec:
            client = wrap_vllm(
                OpenAI(api_key="token", base_url="http://localhost:8000/v1"),
                rec,
            )
            resp = client.chat.completions.create(
                model="meta-llama/Llama-3-8b-Instruct",
                messages=[{"role": "user", "content": "Hello"}],
            )
    """
    return _wrap_openai_compat(
        client, recorder, provider_name="vllm",
        default_model=default_model, seed_policy=seed_policy, contract=contract,
    )


def vllm_executor(client: Any) -> Callable[[str, List[dict]], dict]:
    """Adapter so a real vLLM client can serve dirty replay steps."""
    return openai_compat_executor(client, provider_name="vllm")


def wrap_tgi(
    client: Any,
    recorder: Recorder,
    *,
    default_model: Optional[str] = None,
    seed_policy: Optional[SeedPolicy] = None,
    contract: Optional["ShimContract"] = None,
) -> WrappedOpenAICompat:
    """Wrap a HuggingFace TGI client so ``chat.completions.create(...)`` calls are recorded.

    TGI exposes an OpenAI-compatible endpoint at ``http://localhost:8080/v1`` by default.
    Pass an ``openai.OpenAI(base_url="http://localhost:8080/v1")`` client.

    Example::

        from openai import OpenAI
        from stepback import record
        from stepback.shims import wrap_tgi

        with record("./trace.sb") as rec:
            client = wrap_tgi(
                OpenAI(api_key="token", base_url="http://localhost:8080/v1"),
                rec,
            )
    """
    return _wrap_openai_compat(
        client, recorder, provider_name="tgi",
        default_model=default_model, seed_policy=seed_policy, contract=contract,
    )


def tgi_executor(client: Any) -> Callable[[str, List[dict]], dict]:
    """Adapter so a real TGI client can serve dirty replay steps."""
    return openai_compat_executor(client, provider_name="tgi")


def wrap_llamacpp(
    client: Any,
    recorder: Recorder,
    *,
    default_model: Optional[str] = None,
    seed_policy: Optional[SeedPolicy] = None,
    contract: Optional["ShimContract"] = None,
) -> WrappedOpenAICompat:
    """Wrap a llama.cpp server client so ``chat.completions.create(...)`` calls are recorded.

    The llama.cpp server exposes an OpenAI-compatible endpoint at
    ``http://localhost:8080/v1`` by default.  Pass an
    ``openai.OpenAI(base_url="http://localhost:8080/v1")`` client.

    Example::

        from openai import OpenAI
        from stepback import record
        from stepback.shims import wrap_llamacpp

        with record("./trace.sb") as rec:
            client = wrap_llamacpp(
                OpenAI(api_key="none", base_url="http://localhost:8080/v1"),
                rec,
            )
    """
    return _wrap_openai_compat(
        client, recorder, provider_name="llamacpp",
        default_model=default_model, seed_policy=seed_policy, contract=contract,
    )


def llamacpp_executor(client: Any) -> Callable[[str, List[dict]], dict]:
    """Adapter so a real llama.cpp server client can serve dirty replay steps."""
    return openai_compat_executor(client, provider_name="llamacpp")


def wrap_ollama(
    client: Any,
    recorder: Recorder,
    *,
    default_model: Optional[str] = None,
    seed_policy: Optional[SeedPolicy] = None,
    contract: Optional["ShimContract"] = None,
) -> WrappedOpenAICompat:
    """Wrap an Ollama client so ``chat.completions.create(...)`` calls are recorded.

    **Important:** this shim wraps the *OpenAI-compatible* Ollama endpoint,
    not the native ``ollama`` Python SDK.  Pass an
    ``openai.OpenAI(base_url="http://localhost:11434/v1")`` client.

    Example::

        from openai import OpenAI
        from stepback import record
        from stepback.shims import wrap_ollama

        with record("./trace.sb") as rec:
            client = wrap_ollama(
                OpenAI(api_key="ollama", base_url="http://localhost:11434/v1"),
                rec,
            )
            resp = client.chat.completions.create(
                model="llama3.2",
                messages=[{"role": "user", "content": "Hello"}],
            )
    """
    return _wrap_openai_compat(
        client, recorder, provider_name="ollama",
        default_model=default_model, seed_policy=seed_policy, contract=contract,
    )


def ollama_executor(client: Any) -> Callable[[str, List[dict]], dict]:
    """Adapter so a real Ollama client can serve dirty replay steps."""
    return openai_compat_executor(client, provider_name="ollama")


# =====================================================================
# ShimContract registry
# =====================================================================

#: Module-level registry mapping ``provider_name`` → :class:`ShimContract`.
#: The four built-in providers are pre-registered below; third-party
#: providers call :func:`register_shim_contract` to extend it.
_CONTRACT_REGISTRY: Dict[str, ShimContract] = {}


def register_shim_contract(
    contract: ShimContract, *, overwrite: bool = False
) -> None:
    """Register a :class:`ShimContract` under its :attr:`~ShimContract.provider_name`.

    Parameters
    ----------
    contract:
        The contract instance to register.
    overwrite:
        When ``False`` (default) registering a name already in the registry
        raises :class:`KeyError`.  Pass ``overwrite=True`` to replace an
        existing entry explicitly.

    Raises
    ------
    TypeError
        If *contract* does not have a non-empty ``provider_name`` str.
    KeyError
        If *overwrite* is ``False`` and the name is already registered.
    """
    name = getattr(contract, "provider_name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(
            "ShimContract must declare a non-empty str class attribute "
            f"'provider_name'; got {name!r}"
        )
    if name in _CONTRACT_REGISTRY and not overwrite:
        raise KeyError(
            f"A ShimContract for {name!r} is already registered; "
            "pass overwrite=True to replace it"
        )
    _CONTRACT_REGISTRY[name] = contract


def shim_contract_for(provider: str) -> ShimContract:
    """Return the registered :class:`ShimContract` for *provider*.

    Raises
    ------
    KeyError
        If no contract has been registered for *provider*.
    """
    try:
        return _CONTRACT_REGISTRY[provider]
    except KeyError:
        raise KeyError(
            f"No ShimContract registered for provider {provider!r}. "
            "Call register_shim_contract() to add one, or import a "
            "built-in contract (openai, anthropic, bedrock, gemini)."
        ) from None


# Register built-in contracts.
register_shim_contract(OpenAIShimContract())
register_shim_contract(AnthropicShimContract())
register_shim_contract(BedrockShimContract())
register_shim_contract(GeminiShimContract())
register_shim_contract(AzureOpenAIShimContract())
register_shim_contract(CohereShimContract())
register_shim_contract(MistralShimContract())
# OpenAI-compatible provider contracts (Step 96).
register_shim_contract(GroqShimContract())
register_shim_contract(TogetherShimContract())
register_shim_contract(FireworksShimContract())
register_shim_contract(CerebrasShimContract())
register_shim_contract(NvidiaNIMShimContract())
register_shim_contract(VLLMShimContract())
register_shim_contract(TGIShimContract())
register_shim_contract(LlamaCppShimContract())
register_shim_contract(OllamaShimContract())

# Public aliases for Gemini canonicalization helpers (Step 94).
canonicalize_gemini_safety_settings = _canonicalize_gemini_safety_settings
canonicalize_gemini_tools = _canonicalize_gemini_tools


# =====================================================================
# LangChain / LangGraph callback recorder (Step 101)
# =====================================================================

def _normalize_lc_message(msg: Any) -> dict:
    """Normalize a LangChain-shaped message object or dict to a canonical dict.

    Handles:
    - Objects with ``.type`` and ``.content`` attributes (BaseMessage duck type)
    - Plain dicts with ``role``/``content``
    - Content that is a list of content blocks (flattened to str)
    """
    if isinstance(msg, dict):
        return dict(msg)
    # Duck-type: use .type for role mapping
    msg_type = getattr(msg, "type", None) or "unknown"
    role_map = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
        "function": "tool",
    }
    role = role_map.get(msg_type, msg_type)
    content = getattr(msg, "content", "")
    # Flatten list-of-blocks content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        content = "".join(parts)
    result: dict = {"role": role, "content": content}
    # Preserve tool_calls if present
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = tool_calls
    # Preserve tool_call_id for tool/function messages
    tool_call_id = getattr(msg, "tool_call_id", None)
    if tool_call_id is not None:
        result["tool_call_id"] = tool_call_id
    return result


def _normalize_lc_llm_result(result: Any, model: str) -> dict:
    """Normalize a LangChain LLMResult to a canonical OpenAI-shaped response dict."""
    llm_output = getattr(result, "llm_output", None) or {}
    token_usage = llm_output.get("token_usage", {}) if isinstance(llm_output, dict) else {}
    prompt_tokens = token_usage.get("prompt_tokens", 0)
    completion_tokens = token_usage.get("completion_tokens", 0)
    total_tokens = token_usage.get("total_tokens", 0)
    if total_tokens == 0 and (prompt_tokens or completion_tokens):
        total_tokens = prompt_tokens + completion_tokens

    generations = getattr(result, "generations", []) or []
    # Take first generation from first batch
    first_gen = None
    if generations and generations[0]:
        first_gen = generations[0][0]

    if first_gen is not None:
        raw_msg = getattr(first_gen, "message", None)
        if raw_msg is not None:
            choice_msg = _normalize_lc_message(raw_msg)
        else:
            # Plain text generation
            text = getattr(first_gen, "text", "") or ""
            choice_msg = {"role": "assistant", "content": text}
        finish_reason = (getattr(first_gen, "generation_info", None) or {}).get(
            "finish_reason", "stop"
        )
    else:
        choice_msg = {"role": "assistant", "content": ""}
        finish_reason = "stop"

    return {
        "model": model,
        "choices": [
            {
                "message": choice_msg,
                "finish_reason": finish_reason,
                "index": 0,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


class StepbackCallbackHandler:
    """LangChain / LangGraph callback handler that records steps into a stepback trace.

    Usage::

        with record("trace.sb") as rec:
            handler = StepbackCallbackHandler(rec)
            llm.invoke(messages, config={"callbacks": [handler]})

    Parameters
    ----------
    recorder:
        An active :class:`~stepback.recorder.Recorder` opened with
        :func:`~stepback.record`.
    record_chains:
        When ``True`` (default) chain-start/end events are recorded as
        ``router`` steps.  Set to ``False`` to record only LLM and tool
        calls.
    """

    # LangChain reads these class attributes to configure callback dispatch.
    raise_error: bool = False
    ignore_llm: bool = False
    ignore_chain: bool = False
    ignore_agent: bool = False

    def __init__(self, recorder: Any, *, record_chains: bool = True) -> None:
        self._rec = recorder
        self._record_chains = record_chains
        # run_id → {"kind", "name", "inputs", "start_time"} — in-flight events
        self._pending: Dict[str, dict] = {}
        # run_id → step_id (for parent resolution after completion)
        self._run_to_step: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_parent(self, parent_run_id: Optional[str]) -> Optional[str]:
        """Return the recorded step_id for a parent run_id, or None."""
        if parent_run_id is None:
            return None
        return self._run_to_step.get(parent_run_id)

    def _commit_step(self, step: dict) -> None:
        """Append a complete step dict to the recorder."""
        from .canonical import hash_obj, canonical_json, sha256_hex
        import time as _time
        step.setdefault("wallclock_ns", _time.time_ns())
        step["inputs_hash"] = hash_obj(step["inputs"])
        step["outputs_hash"] = hash_obj(step["outputs"])
        step.setdefault(
            "nondeterminism_hash",
            sha256_hex(canonical_json(step.get("nondeterminism", {}))),
        )
        step.setdefault("cost_usd", 0.0)
        self._rec.writer.write_step(step)
        self._rec.steps.append(step)
        # Register step_id for parent resolution
        sid = step["step_id"]
        self._run_to_step[step.get("_lc_run_id", sid)] = sid

    def _new_step_id(self) -> str:
        return self._rec._new_id()

    # ------------------------------------------------------------------
    # LLM callbacks
    # ------------------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict,
        messages: List[Any],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        model = (
            (serialized.get("kwargs") or {}).get("model_name")
            or serialized.get("name")
            or "unknown"
        )
        # Normalize message list-of-lists to flat list
        flat_msgs: List[dict] = []
        for batch in messages:
            if isinstance(batch, list):
                for m in batch:
                    flat_msgs.append(_normalize_lc_message(m))
            else:
                flat_msgs.append(_normalize_lc_message(batch))
        self._pending[run_id] = {
            "kind": "llm_call",
            "model": model,
            "messages": flat_msgs,
            "parent_run_id": parent_run_id,
        }

    def on_llm_start(
        self,
        serialized: dict,
        prompts: List[str],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        model = (
            (serialized.get("kwargs") or {}).get("model_name")
            or serialized.get("name")
            or "unknown"
        )
        # Convert plain text prompts to user messages
        flat_msgs = [{"role": "user", "content": p} for p in prompts]
        self._pending[run_id] = {
            "kind": "llm_call",
            "model": model,
            "messages": flat_msgs,
            "parent_run_id": parent_run_id,
        }

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        pending = self._pending.pop(run_id, None)
        if pending is None:
            return
        model = pending["model"]
        canonical_response = _normalize_lc_llm_result(response, model)
        sid = self._new_step_id()
        parent_step_id = self._resolve_parent(pending.get("parent_run_id"))
        step = {
            "step_id": sid,
            "step_kind": "llm_call",
            "name": model,
            "parent_step_id": parent_step_id,
            "inputs": {
                "kind": "llm_call",
                "model": model,
                "messages": pending["messages"],
            },
            "outputs": canonical_response,
            "metadata": {"langchain_run_id": run_id},
            "_lc_run_id": run_id,
        }
        self._commit_step(step)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        self._pending.pop(run_id, None)

    # ------------------------------------------------------------------
    # Tool callbacks
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict,
        input_str: Any,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "unknown_tool")
        # Normalize input: if dict, keep as-is; else wrap in {"input": ...}
        if isinstance(input_str, dict):
            arguments = input_str
        else:
            arguments = {"input": input_str}
        self._pending[run_id] = {
            "kind": "tool_call",
            "name": tool_name,
            "arguments": arguments,
            "parent_run_id": parent_run_id,
        }

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        pending = self._pending.pop(run_id, None)
        if pending is None:
            return
        tool_name = pending["name"]
        sid = self._new_step_id()
        parent_step_id = self._resolve_parent(pending.get("parent_run_id"))
        step = {
            "step_id": sid,
            "step_kind": "tool_call",
            "name": tool_name,
            "parent_step_id": parent_step_id,
            "inputs": {
                "kind": "tool_call",
                "name": tool_name,
                "arguments": pending["arguments"],
            },
            "outputs": {"result": output},
            "metadata": {"langchain_run_id": run_id},
            "_lc_run_id": run_id,
        }
        self._commit_step(step)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        self._pending.pop(run_id, None)

    # ------------------------------------------------------------------
    # Chain callbacks (also used by LangGraph nodes)
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict,
        inputs: Any,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if not self._record_chains:
            return
        chain_name = serialized.get("name", "unknown_chain")
        # Pre-allocate a step_id and register it immediately so nested
        # LLM/tool callbacks can resolve this chain as their parent even
        # before on_chain_end fires.
        sid = self._new_step_id()
        self._run_to_step[run_id] = sid
        self._pending[run_id] = {
            "kind": "router",
            "name": chain_name,
            "inputs": inputs if isinstance(inputs, dict) else {"input": inputs},
            "parent_run_id": parent_run_id,
            "step_id": sid,
        }

    def on_chain_end(
        self,
        outputs: Any,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if not self._record_chains:
            return
        pending = self._pending.pop(run_id, None)
        if pending is None:
            return
        chain_name = pending["name"]
        # Use the pre-allocated step_id from on_chain_start
        sid = pending["step_id"]
        parent_step_id = self._resolve_parent(pending.get("parent_run_id"))
        chain_outputs = outputs if isinstance(outputs, dict) else {"output": outputs}
        step = {
            "step_id": sid,
            "step_kind": "router",
            "name": chain_name,
            "parent_step_id": parent_step_id,
            "inputs": {
                "kind": "router",
                "name": chain_name,
                "options": list(chain_outputs.keys()),
                **pending["inputs"],
            },
            "outputs": chain_outputs,
            "metadata": {"langchain_run_id": run_id},
            "_lc_run_id": run_id,
        }
        self._commit_step(step)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        self._pending.pop(run_id, None)

    # ------------------------------------------------------------------
    # LangGraph node callbacks (delegate to chain handler)
    # ------------------------------------------------------------------

    def on_node_start(
        self,
        serialized: dict,
        inputs: Any,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.on_chain_start(
            serialized, inputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )

    def on_node_end(
        self,
        outputs: Any,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.on_chain_end(
            outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )

    # ------------------------------------------------------------------
    # No-op callbacks (satisfy LangChain's BaseCallbackHandler interface)
    # ------------------------------------------------------------------

    def on_llm_new_token(self, token: str, *, run_id: str, **kwargs: Any) -> None:
        pass

    def on_agent_action(self, action: Any, *, run_id: str, **kwargs: Any) -> None:
        pass

    def on_retriever_start(
        self, serialized: dict, query: str, *, run_id: str, **kwargs: Any
    ) -> None:
        pass

    def on_retriever_end(
        self, documents: Any, *, run_id: str, **kwargs: Any
    ) -> None:
        pass

    def on_text(self, text: str, *, run_id: str, **kwargs: Any) -> None:
        pass


def langchain_callback_handler(
    recorder: Any, *, record_chains: bool = True
) -> StepbackCallbackHandler:
    """Create a :class:`StepbackCallbackHandler` bound to *recorder*.

    Parameters
    ----------
    recorder:
        An active :class:`~stepback.recorder.Recorder`.
    record_chains:
        When ``True`` (default) LangChain chain events are recorded as
        ``router`` steps.  Set to ``False`` to suppress chain steps.

    Returns
    -------
    StepbackCallbackHandler
        A handler ready to be passed to LangChain's ``callbacks=`` parameter.
    """
    return StepbackCallbackHandler(recorder, record_chains=record_chains)


# =====================================================================
# LlamaIndex callback handler (Step 102)
# =====================================================================

class LlamaIndexCallbackHandler:
    """LlamaIndex ``BaseCallbackHandler``-compatible recorder.

    Tracks ``llm`` and ``function_calling`` events as ``llm_call`` /
    ``tool_call`` steps; optionally records all other event types as
    ``tool_call`` steps.

    Parameters
    ----------
    recorder:
        Active :class:`~stepback.recorder.Recorder`.
    record_other_events:
        When ``True`` unknown events are recorded as generic ``tool_call``
        steps.  Default is ``False``.
    """

    # LlamaIndex checks these class attributes for dispatch filtering.
    event_starts_to_ignore: list = []
    event_ends_to_ignore: list = []

    def __init__(self, recorder: Any, *, record_other_events: bool = False) -> None:
        self._rec = recorder
        self._record_other_events = record_other_events
        self._pending: Dict[str, dict] = {}

    def on_event_start(
        self,
        event_type: str,
        payload: Optional[dict] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> str:
        import uuid as _uuid
        eid = event_id or str(_uuid.uuid4())
        self._pending[eid] = {
            "event_type": event_type,
            "payload": payload or {},
        }
        return eid

    def on_event_end(
        self,
        event_type: str,
        payload: Optional[dict] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        pending = self._pending.pop(event_id, None)
        if pending is None:
            return
        start_payload = pending["payload"]
        end_payload = payload or {}

        if event_type == "llm":
            model = start_payload.get("model", "unknown")
            messages = start_payload.get("messages", [])
            step = self._build_step(
                step_kind="llm_call",
                name=model,
                inputs={"kind": "llm_call", "model": model, "messages": messages},
                outputs={"response": end_payload.get("response", "")},
            )
        elif event_type == "function_calling":
            tool_name = start_payload.get("tool", "unknown_tool")
            tool_input = start_payload.get("input", {})
            step = self._build_step(
                step_kind="tool_call",
                name=tool_name,
                inputs={"kind": "tool_call", "name": tool_name, "arguments": tool_input},
                outputs={"result": end_payload.get("output", "")},
            )
        elif self._record_other_events:
            name = event_type
            step = self._build_step(
                step_kind="tool_call",
                name=name,
                inputs={"kind": "tool_call", "name": name, "arguments": start_payload},
                outputs=end_payload or {},
            )
        else:
            return

        self._commit_step(step)

    def _build_step(
        self, *, step_kind: str, name: str, inputs: dict, outputs: dict
    ) -> dict:
        return {
            "step_id": self._rec._new_id(),
            "step_kind": step_kind,
            "name": name,
            "parent_step_id": None,
            "inputs": inputs,
            "outputs": outputs,
        }

    def _commit_step(self, step: dict) -> None:
        from .canonical import hash_obj, canonical_json, sha256_hex
        import time as _time
        step.setdefault("wallclock_ns", _time.time_ns())
        step["inputs_hash"] = hash_obj(step["inputs"])
        step["outputs_hash"] = hash_obj(step["outputs"])
        step.setdefault(
            "nondeterminism_hash",
            sha256_hex(canonical_json(step.get("nondeterminism", {}))),
        )
        step.setdefault("cost_usd", 0.0)
        self._rec.writer.write_step(step)
        self._rec.steps.append(step)

    def start_trace(self, trace_id: str = "", **kwargs: Any) -> None:
        """No-op: satisfies LlamaIndex tracer interface."""

    def end_trace(self, trace_id: str = "", trace_map: Optional[dict] = None, **kwargs: Any) -> None:
        """No-op: satisfies LlamaIndex tracer interface."""


def llamaindex_callback_handler(
    recorder: Any, *, record_other_events: bool = False
) -> LlamaIndexCallbackHandler:
    """Create a :class:`LlamaIndexCallbackHandler` bound to *recorder*."""
    return LlamaIndexCallbackHandler(recorder, record_other_events=record_other_events)


# =====================================================================
# DSPy callback handler (Step 102)
# =====================================================================

class DSPyCallbackHandler:
    """DSPy callback handler that records LM calls, tool calls, and module
    invocations as stepback trace steps.

    Parameters
    ----------
    recorder:
        Active :class:`~stepback.recorder.Recorder`.
    record_modules:
        When ``True`` module-level start/end events are recorded as
        ``router`` steps.  Default is ``True``.
    """

    def __init__(self, recorder: Any, *, record_modules: bool = True) -> None:
        self._rec = recorder
        self._record_modules = record_modules
        self._pending: Dict[str, dict] = {}

    def _commit_step(self, step: dict) -> None:
        from .canonical import hash_obj, canonical_json, sha256_hex
        import time as _time
        step.setdefault("wallclock_ns", _time.time_ns())
        step["inputs_hash"] = hash_obj(step["inputs"])
        step["outputs_hash"] = hash_obj(step["outputs"])
        step.setdefault(
            "nondeterminism_hash",
            sha256_hex(canonical_json(step.get("nondeterminism", {}))),
        )
        step.setdefault("cost_usd", 0.0)
        self._rec.writer.write_step(step)
        self._rec.steps.append(step)

    def on_lm_start(self, call_id: str, lm: Any, inputs: dict) -> None:
        model = getattr(lm, "model", None) or getattr(lm, "model_name", "unknown")
        self._pending[call_id] = {
            "kind": "llm_call",
            "model": model,
            "inputs": inputs,
        }

    def on_lm_end(
        self,
        call_id: str,
        outputs: Any,
        *,
        exception: Optional[BaseException] = None,
    ) -> None:
        pending = self._pending.pop(call_id, None)
        if pending is None:
            return
        model = pending["model"]
        start_inputs = pending["inputs"]
        messages = start_inputs.get("messages", [{"role": "user", "content": start_inputs.get("prompt", "")}])
        out = outputs if isinstance(outputs, dict) else {"output": outputs}
        step = {
            "step_id": self._rec._new_id(),
            "step_kind": "llm_call",
            "name": model,
            "parent_step_id": None,
            "inputs": {"kind": "llm_call", "model": model, "messages": messages},
            "outputs": out or {},
        }
        self._commit_step(step)

    def on_tool_start(self, call_id: str, tool: Any, inputs: dict) -> None:
        name = (
            getattr(tool, "tool_name", None)
            or getattr(tool, "name", None)
            or "unknown_tool"
        )
        self._pending[call_id] = {
            "kind": "tool_call",
            "name": name,
            "inputs": inputs,
        }

    def on_tool_end(self, call_id: str, output: Any) -> None:
        pending = self._pending.pop(call_id, None)
        if pending is None:
            return
        name = pending["name"]
        step = {
            "step_id": self._rec._new_id(),
            "step_kind": "tool_call",
            "name": name,
            "parent_step_id": None,
            "inputs": {"kind": "tool_call", "name": name, "arguments": pending["inputs"]},
            "outputs": {"result": output},
        }
        self._commit_step(step)

    def on_module_start(self, call_id: str, module: Any, inputs: dict) -> None:
        if not self._record_modules:
            return
        name = getattr(module, "__class__", module).__name__ if not isinstance(module, str) else module
        self._pending[call_id] = {
            "kind": "router",
            "name": name,
            "inputs": inputs,
        }

    def on_module_end(self, call_id: str, outputs: dict) -> None:
        if not self._record_modules:
            self._pending.pop(call_id, None)
            return
        pending = self._pending.pop(call_id, None)
        if pending is None:
            return
        name = pending["name"]
        out = outputs if isinstance(outputs, dict) else {"output": outputs}
        step = {
            "step_id": self._rec._new_id(),
            "step_kind": "router",
            "name": name,
            "parent_step_id": None,
            "inputs": {"kind": "router", "name": name, "options": list(out.keys()), **pending["inputs"]},
            "outputs": out,
        }
        self._commit_step(step)


def dspy_callback_handler(recorder: Any, *, record_modules: bool = True) -> DSPyCallbackHandler:
    """Create a :class:`DSPyCallbackHandler` bound to *recorder*."""
    return DSPyCallbackHandler(recorder, record_modules=record_modules)


# =====================================================================
# Haystack tracer (Step 102)
# =====================================================================

class _HaystackSpan:
    """Context-manager span produced by :class:`HaystackTracer`."""

    def __init__(self, tracer: "HaystackTracer", operation_name: str, tags: dict) -> None:
        self._tracer = tracer
        self._operation_name = operation_name
        self._tags: dict = dict(tags)

    def set_content_tag(self, key: str, value: Any) -> None:
        self._tags[key] = value

    def get_correlation_data_for_logs(self) -> dict:
        return {"operation_name": self._operation_name}

    def __enter__(self) -> "_HaystackSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        self._tracer._flush_span(self)
        if self._tracer._current_span is self:
            self._tracer._current_span = None


_LLM_COMPONENT_SUBSTRINGS = (
    "generator", "chatgenerator", "completiongenerator", "openai", "anthropic",
    "huggingface", "cohere", "mistral", "gemini", "azure"
)


class HaystackTracer:
    """Haystack ``Tracer``-compatible recorder.

    Parameters
    ----------
    recorder:
        Active :class:`~stepback.recorder.Recorder`.
    """

    def __init__(self, recorder: Any) -> None:
        self._rec = recorder
        self._current_span: Optional[_HaystackSpan] = None

    def trace(self, operation_name: str, tags: Optional[dict] = None) -> _HaystackSpan:
        span = _HaystackSpan(self, operation_name, tags or {})
        self._current_span = span
        return span

    def current_span(self) -> Optional[_HaystackSpan]:
        return self._current_span

    def _flush_span(self, span: _HaystackSpan) -> None:
        tags = span._tags
        comp_name: str = tags.get("haystack.component.name", span._operation_name)
        comp_input: dict = tags.get("haystack.component.input", {})
        comp_output: dict = tags.get("haystack.component.output", {})

        # Determine step kind from operation/component name
        op_lower = span._operation_name.lower()
        is_llm = any(s in op_lower for s in _LLM_COMPONENT_SUBSTRINGS)
        step_kind = "llm_call" if is_llm else "tool_call"

        if step_kind == "llm_call":
            messages = comp_input.get("messages", [])
            model = tags.get("haystack.component.model", comp_name)
            step = {
                "step_id": self._rec._new_id(),
                "step_kind": "llm_call",
                "name": model,
                "parent_step_id": None,
                "inputs": {"kind": "llm_call", "model": model, "messages": messages},
                "outputs": comp_output or {},
            }
        else:
            step = {
                "step_id": self._rec._new_id(),
                "step_kind": "tool_call",
                "name": comp_name,
                "parent_step_id": None,
                "inputs": {"kind": "tool_call", "name": comp_name, "arguments": comp_input},
                "outputs": comp_output or {},
            }

        from .canonical import hash_obj, canonical_json, sha256_hex
        import time as _time
        step.setdefault("wallclock_ns", _time.time_ns())
        step["inputs_hash"] = hash_obj(step["inputs"])
        step["outputs_hash"] = hash_obj(step["outputs"])
        step.setdefault(
            "nondeterminism_hash",
            sha256_hex(canonical_json(step.get("nondeterminism", {}))),
        )
        step.setdefault("cost_usd", 0.0)
        self._rec.writer.write_step(step)
        self._rec.steps.append(step)


def haystack_tracer(recorder: Any) -> HaystackTracer:
    """Create a :class:`HaystackTracer` bound to *recorder*."""
    return HaystackTracer(recorder)


# =====================================================================
# AutoGen event handler (Step 102)
# =====================================================================

class AutoGenEventHandler:
    """AutoGen-compatible event handler that records LLM and tool calls.

    Parameters
    ----------
    recorder:
        Active :class:`~stepback.recorder.Recorder`.
    """

    def __init__(self, recorder: Any) -> None:
        self._rec = recorder
        self._pending_llm: Optional[dict] = None
        self._pending_tool: Optional[dict] = None

    def _commit_step(self, step: dict) -> None:
        from .canonical import hash_obj, canonical_json, sha256_hex
        import time as _time
        step.setdefault("wallclock_ns", _time.time_ns())
        step["inputs_hash"] = hash_obj(step["inputs"])
        step["outputs_hash"] = hash_obj(step["outputs"])
        step.setdefault(
            "nondeterminism_hash",
            sha256_hex(canonical_json(step.get("nondeterminism", {}))),
        )
        step.setdefault("cost_usd", 0.0)
        self._rec.writer.write_step(step)
        self._rec.steps.append(step)

    def on_llm_call(self, model: str, messages: Any) -> None:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        self._pending_llm = {"model": model, "messages": messages}

    def on_llm_call_result(self, response: Any) -> None:
        pending = self._pending_llm
        self._pending_llm = None
        if pending is None:
            return
        model = pending["model"]
        out = response if isinstance(response, dict) else {"content": str(response) if response is not None else ""}
        step = {
            "step_id": self._rec._new_id(),
            "step_kind": "llm_call",
            "name": model,
            "parent_step_id": None,
            "inputs": {"kind": "llm_call", "model": model, "messages": pending["messages"]},
            "outputs": out,
        }
        self._commit_step(step)

    def on_tool_call(self, tool_name: str, tool_input: Any) -> None:
        args = tool_input if isinstance(tool_input, dict) else {"input": tool_input}
        self._pending_tool = {"name": tool_name, "arguments": args}

    def on_tool_call_result(self, tool_name: str, result: Any) -> None:
        pending = self._pending_tool
        self._pending_tool = None
        if pending is None:
            return
        name = pending["name"]
        step = {
            "step_id": self._rec._new_id(),
            "step_kind": "tool_call",
            "name": name,
            "parent_step_id": None,
            "inputs": {"kind": "tool_call", "name": name, "arguments": pending["arguments"]},
            "outputs": {"result": result},
        }
        self._commit_step(step)

    def print(self, *args: Any, sep: str = " ", **kwargs: Any) -> None:
        """No-op (satisfies AutoGen's print interface)."""

    def input(self, prompt: str = "", **kwargs: Any) -> str:
        """No-op (satisfies AutoGen's input interface)."""
        return ""


def autogen_event_handler(recorder: Any) -> AutoGenEventHandler:
    """Create an :class:`AutoGenEventHandler` bound to *recorder*."""
    return AutoGenEventHandler(recorder)


# =====================================================================
# CrewAI step recorder (Step 102)
# =====================================================================

class CrewAIStepRecorder:
    """CrewAI ``step_callback``-compatible recorder.

    Usage::

        from stepback import record
        from stepback.shims import CrewAIStepRecorder

        with record("trace.sb") as rec:
            cb = CrewAIStepRecorder(rec)
            crew = Crew(..., step_callback=cb)

    Parameters
    ----------
    recorder:
        Active :class:`~stepback.recorder.Recorder`.
    """

    def __init__(self, recorder: Any) -> None:
        self._rec = recorder

    def _commit_step(self, step: dict) -> None:
        from .canonical import hash_obj, canonical_json, sha256_hex
        import time as _time
        step.setdefault("wallclock_ns", _time.time_ns())
        step["inputs_hash"] = hash_obj(step["inputs"])
        step["outputs_hash"] = hash_obj(step["outputs"])
        step.setdefault(
            "nondeterminism_hash",
            sha256_hex(canonical_json(step.get("nondeterminism", {}))),
        )
        step.setdefault("cost_usd", 0.0)
        self._rec.writer.write_step(step)
        self._rec.steps.append(step)

    def __call__(self, step_output: Any) -> None:
        """Process a CrewAI step output (AgentAction or AgentFinish shape)."""
        # AgentAction shape: has .tool and .tool_input attributes
        tool = getattr(step_output, "tool", None)
        if tool is not None:
            tool_input = getattr(step_output, "tool_input", {})
            args = tool_input if isinstance(tool_input, dict) else {"input": tool_input}
            result = getattr(step_output, "result", "")
            self.on_tool_use(tool, args, result)
            return
        # AgentFinish shape: has .return_values attribute
        return_values = getattr(step_output, "return_values", None)
        if return_values is not None:
            out = return_values if isinstance(return_values, dict) else {"output": return_values}
            self.on_agent_finish(out)
            return
        # Generic fallback
        step = {
            "step_id": self._rec._new_id(),
            "step_kind": "tool_call",
            "name": "crewai:step",
            "parent_step_id": None,
            "inputs": {"kind": "tool_call", "name": "crewai:step", "arguments": {"step": str(step_output)}},
            "outputs": {"result": str(step_output)},
        }
        self._commit_step(step)

    def on_tool_use(self, tool_name: str, arguments: Any, result: Any) -> None:
        args = arguments if isinstance(arguments, dict) else {"input": arguments}
        step = {
            "step_id": self._rec._new_id(),
            "step_kind": "tool_call",
            "name": tool_name,
            "parent_step_id": None,
            "inputs": {"kind": "tool_call", "name": tool_name, "arguments": args},
            "outputs": {"result": result},
        }
        self._commit_step(step)

    def on_agent_finish(self, output: Any) -> None:
        out = output if isinstance(output, dict) else {"output": str(output)}
        step = {
            "step_id": self._rec._new_id(),
            "step_kind": "tool_call",
            "name": "crewai:finish",
            "parent_step_id": None,
            "inputs": {"kind": "tool_call", "name": "crewai:finish", "arguments": {}},
            "outputs": out,
        }
        self._commit_step(step)


def crewai_step_recorder(recorder: Any) -> CrewAIStepRecorder:
    """Create a :class:`CrewAIStepRecorder` bound to *recorder*."""
    return CrewAIStepRecorder(recorder)


# =====================================================================
# Semantic Kernel filter (Step 102)
# =====================================================================

_SK_LLM_FUNCTION_NAMES = frozenset({
    "chat_completion", "text_completion", "get_chat_message_contents",
    "get_text_contents", "invoke", "chat",
})


class SemanticKernelFilter:
    """Semantic Kernel function-invocation filter that records steps.

    Usage::

        kernel.add_filter("function_invocation", SemanticKernelFilter(rec))

    Parameters
    ----------
    recorder:
        Active :class:`~stepback.recorder.Recorder`.
    """

    def __init__(self, recorder: Any) -> None:
        self._rec = recorder

    def _commit_step(self, step: dict) -> None:
        from .canonical import hash_obj, canonical_json, sha256_hex
        import time as _time
        step.setdefault("wallclock_ns", _time.time_ns())
        step["inputs_hash"] = hash_obj(step["inputs"])
        step["outputs_hash"] = hash_obj(step["outputs"])
        step.setdefault(
            "nondeterminism_hash",
            sha256_hex(canonical_json(step.get("nondeterminism", {}))),
        )
        step.setdefault("cost_usd", 0.0)
        self._rec.writer.write_step(step)
        self._rec.steps.append(step)

    def on_function_invocation(
        self,
        context: Any,
        *,
        next: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Record a Semantic Kernel function invocation."""
        func = getattr(context, "function", None)
        func_name: str = getattr(func, "name", "unknown") if func else "unknown"
        plugin_name: str = getattr(func, "plugin_name", "") if func else ""
        arguments = getattr(context, "arguments", {}) or {}
        result = getattr(context, "result", None)

        # Call next middleware if provided
        if next is not None:
            next(context)
            result = getattr(context, "result", result)

        # Build step name
        if plugin_name:
            step_name = f"{plugin_name}:{func_name}"
        else:
            step_name = func_name

        # Classify as LLM or tool call
        is_llm = func_name.lower() in _SK_LLM_FUNCTION_NAMES or "chat" in func_name.lower() or "completion" in func_name.lower()
        step_kind = "llm_call" if is_llm else "tool_call"

        if step_kind == "llm_call":
            inputs = {"kind": "llm_call", "model": step_name, "messages": [], **arguments}
        else:
            inputs = {"kind": "tool_call", "name": step_name, "arguments": arguments}

        step = {
            "step_id": self._rec._new_id(),
            "step_kind": step_kind,
            "name": step_name,
            "parent_step_id": None,
            "inputs": inputs,
            "outputs": {"result": result},
        }
        self._commit_step(step)
        return result

    async def on_function_invocation_async(
        self,
        context: Any,
        *,
        next: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Async variant of :meth:`on_function_invocation`."""
        return self.on_function_invocation(context, next=next, **kwargs)


def semantic_kernel_filter(recorder: Any) -> SemanticKernelFilter:
    """Create a :class:`SemanticKernelFilter` bound to *recorder*."""
    return SemanticKernelFilter(recorder)


# =====================================================================
# Strands callback handler (Step 102)
# =====================================================================

class StrandsCallbackHandler:
    """Strands-compatible callback handler for recording LLM and tool calls.

    Parameters
    ----------
    recorder:
        Active :class:`~stepback.recorder.Recorder`.
    """

    def __init__(self, recorder: Any) -> None:
        self._rec = recorder
        self._pending: Dict[str, dict] = {}

    def _commit_step(self, step: dict) -> None:
        from .canonical import hash_obj, canonical_json, sha256_hex
        import time as _time
        step.setdefault("wallclock_ns", _time.time_ns())
        step["inputs_hash"] = hash_obj(step["inputs"])
        step["outputs_hash"] = hash_obj(step["outputs"])
        step.setdefault(
            "nondeterminism_hash",
            sha256_hex(canonical_json(step.get("nondeterminism", {}))),
        )
        step.setdefault("cost_usd", 0.0)
        self._rec.writer.write_step(step)
        self._rec.steps.append(step)

    def on_start(
        self,
        *,
        model: str,
        messages: List[Any],
        call_id: str,
        **kwargs: Any,
    ) -> None:
        import uuid as _uuid
        cid = call_id or str(_uuid.uuid4())
        self._pending[cid] = {
            "kind": "llm_call",
            "model": model,
            "messages": messages,
            "chunks": [],
        }

    def on_llm_chunk(self, chunk: str, *, call_id: str, **kwargs: Any) -> None:
        pending = self._pending.get(call_id)
        if pending is not None:
            pending["chunks"].append(chunk)

    def on_end(
        self,
        *,
        response: Any,
        call_id: str,
        **kwargs: Any,
    ) -> None:
        pending = self._pending.pop(call_id, None)
        if pending is None:
            return
        model = pending["model"]
        # Prefer streamed chunks over final response
        chunks = pending.get("chunks", [])
        if chunks:
            content = "".join(chunks)
        elif isinstance(response, str):
            content = response
        elif isinstance(response, dict):
            content = response.get("content", str(response))
        else:
            content = str(response) if response is not None else ""

        step = {
            "step_id": self._rec._new_id(),
            "step_kind": "llm_call",
            "name": model,
            "parent_step_id": None,
            "inputs": {"kind": "llm_call", "model": model, "messages": pending["messages"]},
            "outputs": {"content": content},
        }
        self._commit_step(step)

    def on_tool_start(
        self,
        *,
        tool_name: str,
        tool_input: Any,
        call_id: str,
        **kwargs: Any,
    ) -> None:
        import uuid as _uuid
        cid = call_id or str(_uuid.uuid4())
        args = tool_input if isinstance(tool_input, dict) else {"input": tool_input}
        self._pending[cid] = {"kind": "tool_call", "name": tool_name, "arguments": args}

    def on_tool_end(
        self,
        *,
        tool_name: str,
        tool_output: Any,
        call_id: str,
        **kwargs: Any,
    ) -> None:
        pending = self._pending.pop(call_id, None)
        if pending is None:
            return
        name = pending["name"]
        step = {
            "step_id": self._rec._new_id(),
            "step_kind": "tool_call",
            "name": name,
            "parent_step_id": None,
            "inputs": {"kind": "tool_call", "name": name, "arguments": pending["arguments"]},
            "outputs": {"result": tool_output},
        }
        self._commit_step(step)


def strands_callback_handler(recorder: Any) -> StrandsCallbackHandler:
    """Create a :class:`StrandsCallbackHandler` bound to *recorder*."""
    return StrandsCallbackHandler(recorder)


# =====================================================================
# Pydantic-AI instrument (Step 102)
# =====================================================================

class PydanticAIInstrument:
    """Pydantic-AI instrumentation that records model and tool calls.

    Parameters
    ----------
    recorder:
        Active :class:`~stepback.recorder.Recorder`.
    """

    def __init__(self, recorder: Any) -> None:
        self._rec = recorder
        self._pending: Dict[str, dict] = {}

    def _commit_step(self, step: dict) -> None:
        from .canonical import hash_obj, canonical_json, sha256_hex
        import time as _time
        step.setdefault("wallclock_ns", _time.time_ns())
        step["inputs_hash"] = hash_obj(step["inputs"])
        step["outputs_hash"] = hash_obj(step["outputs"])
        step.setdefault(
            "nondeterminism_hash",
            sha256_hex(canonical_json(step.get("nondeterminism", {}))),
        )
        step.setdefault("cost_usd", 0.0)
        self._rec.writer.write_step(step)
        self._rec.steps.append(step)

    def _normalize_msg(self, msg: Any) -> dict:
        if isinstance(msg, dict):
            return msg
        return {
            "role": getattr(msg, "role", "user"),
            "content": getattr(msg, "content", None) or getattr(msg, "text", ""),
        }

    def on_run_start(self, run_id: str, input_prompt: Any) -> None:
        self._pending[f"run:{run_id}"] = {
            "kind": "router",
            "input": input_prompt,
        }

    def on_run_end(self, run_id: str, output: Any) -> None:
        pending = self._pending.pop(f"run:{run_id}", None)
        if pending is None:
            return
        out = output if isinstance(output, dict) else {"output": str(output) if output is not None else ""}
        step = {
            "step_id": self._rec._new_id(),
            "step_kind": "router",
            "name": "pydantic_ai:run",
            "parent_step_id": None,
            "inputs": {"kind": "router", "name": "pydantic_ai:run", "options": list(out.keys())},
            "outputs": out,
        }
        self._commit_step(step)

    def on_model_request(self, run_id: str, model: str, messages: List[Any]) -> None:
        norm_msgs = [self._normalize_msg(m) for m in messages]
        self._pending[f"model:{run_id}"] = {
            "model": model,
            "messages": norm_msgs,
        }

    def on_model_response(self, run_id: str, response: Any) -> None:
        pending = self._pending.pop(f"model:{run_id}", None)
        if pending is None:
            return
        model = pending["model"]
        if response is None:
            out: dict = {"content": ""}
        elif isinstance(response, dict):
            out = response
        else:
            out = {
                "role": getattr(response, "role", "assistant"),
                "content": getattr(response, "content", None) or getattr(response, "text", str(response)),
            }
        step = {
            "step_id": self._rec._new_id(),
            "step_kind": "llm_call",
            "name": model,
            "parent_step_id": None,
            "inputs": {"kind": "llm_call", "model": model, "messages": pending["messages"]},
            "outputs": out,
        }
        self._commit_step(step)

    def on_tool_call(self, run_id: str, tool_name: str, arguments: Any) -> None:
        args = arguments if isinstance(arguments, dict) else {"input": arguments}
        self._pending[f"tool:{run_id}:{tool_name}"] = {
            "name": tool_name,
            "arguments": args,
        }

    def on_tool_return(self, run_id: str, tool_name: str, result: Any) -> None:
        pending = self._pending.pop(f"tool:{run_id}:{tool_name}", None)
        if pending is None:
            return
        step = {
            "step_id": self._rec._new_id(),
            "step_kind": "tool_call",
            "name": tool_name,
            "parent_step_id": None,
            "inputs": {"kind": "tool_call", "name": tool_name, "arguments": pending["arguments"]},
            "outputs": {"result": result},
        }
        self._commit_step(step)


def pydantic_ai_instrument(recorder: Any) -> PydanticAIInstrument:
    """Create a :class:`PydanticAIInstrument` bound to *recorder*."""
    return PydanticAIInstrument(recorder)


# =====================================================================
# Inspect-AI recorder (Step 102)
# =====================================================================

class InspectAIRecorder:
    """Inspect-AI compatible recorder.

    Parameters
    ----------
    recorder:
        Active :class:`~stepback.recorder.Recorder`.
    """

    def __init__(self, recorder: Any) -> None:
        self._rec = recorder

    def _commit_step(self, step: dict) -> None:
        from .canonical import hash_obj, canonical_json, sha256_hex
        import time as _time
        step.setdefault("wallclock_ns", _time.time_ns())
        step["inputs_hash"] = hash_obj(step["inputs"])
        step["outputs_hash"] = hash_obj(step["outputs"])
        step.setdefault(
            "nondeterminism_hash",
            sha256_hex(canonical_json(step.get("nondeterminism", {}))),
        )
        step.setdefault("cost_usd", 0.0)
        self._rec.writer.write_step(step)
        self._rec.steps.append(step)

    def _normalize_response(self, response: Any) -> dict:
        """Normalize various response shapes to a dict."""
        if response is None:
            return {"content": ""}
        if isinstance(response, dict):
            return response
        # Objects with .completion attribute (Inspect-AI ModelOutput)
        completion = getattr(response, "completion", None)
        if completion is not None:
            return {"content": completion}
        return {"content": str(response)}

    def on_model_call(
        self,
        model: str,
        messages: List[Any],
        response: Any,
    ) -> None:
        """Record a completed model call."""
        norm_msgs = []
        for m in messages:
            if isinstance(m, dict):
                norm_msgs.append(m)
            else:
                content = getattr(m, "content", None) or getattr(m, "text", "")
                norm_msgs.append({
                    "role": getattr(m, "role", "user"),
                    "content": content,
                })
        step = {
            "step_id": self._rec._new_id(),
            "step_kind": "llm_call",
            "name": model,
            "parent_step_id": None,
            "inputs": {"kind": "llm_call", "model": model, "messages": norm_msgs},
            "outputs": self._normalize_response(response),
        }
        self._commit_step(step)

    def on_tool_call(self, tool_name: str, arguments: Any, result: Any) -> None:
        """Record a completed tool call."""
        args = arguments if isinstance(arguments, dict) else {"input": arguments}
        step = {
            "step_id": self._rec._new_id(),
            "step_kind": "tool_call",
            "name": tool_name,
            "parent_step_id": None,
            "inputs": {"kind": "tool_call", "name": tool_name, "arguments": args},
            "outputs": {"result": result},
        }
        self._commit_step(step)

    def on_task_state(self, state: Any) -> None:
        """Record steps from an Inspect-AI TaskState.

        Walks the messages list looking for assistant turns (LLM outputs)
        and tool call results.
        """
        model = getattr(state, "model", "unknown")
        messages: list = getattr(state, "messages", []) or []
        for msg in messages:
            role = getattr(msg, "role", "")
            if role != "assistant":
                continue
            tool_calls = getattr(msg, "tool_calls", []) or []
            content = getattr(msg, "content", None) or getattr(msg, "text", "")
            if tool_calls:
                for tc in tool_calls:
                    func = getattr(tc, "function", {}) if not isinstance(tc, dict) else tc
                    fn_name = func.get("name", "unknown") if isinstance(func, dict) else getattr(func, "name", "unknown")
                    fn_args = func.get("arguments", {}) if isinstance(func, dict) else getattr(func, "arguments", {})
                    self.on_tool_call(fn_name, fn_args, content)
            else:
                self.on_model_call(model, [], {"content": content})


def inspect_ai_recorder(recorder: Any) -> InspectAIRecorder:
    """Create an :class:`InspectAIRecorder` bound to *recorder*."""
    return InspectAIRecorder(recorder)


__all__ = [
    # ABC + registry
    "ShimContract",
    "OpenAIShimContract",
    "AnthropicShimContract",
    "BedrockShimContract",
    "GeminiShimContract",
    "AzureOpenAIShimContract",
    "CohereShimContract",
    "MistralShimContract",
    # OpenAI-compatible provider contracts (Step 96)
    "OpenAICompatShimContract",
    "GroqShimContract",
    "TogetherShimContract",
    "FireworksShimContract",
    "CerebrasShimContract",
    "NvidiaNIMShimContract",
    "VLLMShimContract",
    "TGIShimContract",
    "LlamaCppShimContract",
    "OllamaShimContract",
    "register_shim_contract",
    "shim_contract_for",
    # Wrapped clients
    "WrappedOpenAI",
    "AsyncWrappedOpenAI",
    "WrappedAnthropic",
    "AsyncWrappedAnthropic",
    "WrappedBedrock",
    "WrappedGemini",
    "WrappedAzureOpenAI",
    "WrappedCohere",
    "WrappedMistral",
    "WrappedOpenAICompat",
    "WrappedLangchainTool",
    "WrappedMCPSession",
    # Response namespaces
    "OpenAIChatCompletion",
    "OpenAIResponsesOutput",
    "AnthropicMessage",
    "GeminiResponse",
    "GeminiCandidate",
    "GeminiUsageMetadata",
    "CohereMessage",
    "MistralChatResponse",
    # Factory functions
    "wrap_openai",
    "wrap_openai_async",
    "wrap_openai_responses",
    "wrap_anthropic",
    "wrap_anthropic_async",
    "wrap_azure_openai",
    "wrap_bedrock",
    "wrap_gemini",
    "wrap_vertex_model",
    "wrap_cohere",
    "wrap_mistral",
    # OpenAI-compatible provider wrap functions (Step 96)
    "wrap_groq",
    "wrap_together",
    "wrap_fireworks",
    "wrap_cerebras",
    "wrap_nvidia_nim",
    "wrap_vllm",
    "wrap_tgi",
    "wrap_llamacpp",
    "wrap_ollama",
    "wrap_langchain_tool",
    "wrap_langchain_tools",
    "wrap_mcp_session",
    # Replay executors
    "openai_executor",
    "anthropic_executor",
    "azure_openai_executor",
    "bedrock_executor",
    "gemini_executor",
    "cohere_executor",
    "mistral_executor",
    # OpenAI-compatible provider executors (Step 96)
    "openai_compat_executor",
    "groq_executor",
    "together_executor",
    "fireworks_executor",
    "cerebras_executor",
    "nvidia_nim_executor",
    "vllm_executor",
    "tgi_executor",
    "llamacpp_executor",
    "ollama_executor",
    "langchain_tool_executor",
    "mcp_tool_executor",
    # Model-id helpers
    "canonical_azure_model_id",
    "canonical_bedrock_model_id",
    "canonical_gemini_model_id",
    "canonical_cohere_model_id",
    "canonical_mistral_model_id",
    # Safety settings + tool declarations canonicalization (Step 94)
    "canonicalize_gemini_safety_settings",
    "canonicalize_gemini_tools",
    # Streaming helpers
    "_accumulate_streaming_chunks",
    "_accumulate_anthropic_streaming_chunks",
    # Streaming recorder support (Step 97)
    "StreamedLLMResponse",
    # LangChain / LangGraph callback recorder (Step 101)
    "StepbackCallbackHandler",
    "langchain_callback_handler",
    "_normalize_lc_message",
    "_normalize_lc_llm_result",
    # Framework recorders (Step 102)
    "LlamaIndexCallbackHandler",
    "llamaindex_callback_handler",
    "DSPyCallbackHandler",
    "dspy_callback_handler",
    "HaystackTracer",
    "haystack_tracer",
    "AutoGenEventHandler",
    "autogen_event_handler",
    "CrewAIStepRecorder",
    "crewai_step_recorder",
    "SemanticKernelFilter",
    "semantic_kernel_filter",
    "StrandsCallbackHandler",
    "strands_callback_handler",
    "PydanticAIInstrument",
    "pydantic_ai_instrument",
    "InspectAIRecorder",
    "inspect_ai_recorder",
]
