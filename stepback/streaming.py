"""Streaming recorder support (Step 97).

:class:`StreamedLLMResponse` is returned when ``stream=True`` is passed to a
wrapped provider client (``wrap_openai``, ``wrap_anthropic``, OpenAI-compatible
wrappers, Azure OpenAI).  It:

1. Is a re-iterable iterator that yields the **original raw chunks/events**
   emitted by the provider API.
2. Stores the **assembled canonical final response** (same shape as a
   non-streaming call).
3. Provides access to the **recorded llm_call step** dict.
4. Holds the **provider-specific response object** (``OpenAIChatCompletion``,
   ``AnthropicMessage``, …) for backward-compatible attribute access.

Recording model
---------------
When ``stream=True``, the shim **eagerly consumes** all chunks from the
provider before returning.  The chunks are buffered so the caller can replay
them for progressive display.  The llm_call step is recorded *synchronously*
before this object is returned, which ensures:

- Correct ``parent_step_id`` linkage (the parent is fixed at call time).
- The trace is always consistent even if the caller uses early-exit or raises
  an exception during iteration.

Hash determinism
----------------
``outputs_hash`` is computed over the *assembled* canonical response dict,
not over the raw chunks.  Because :func:`~stepback.shims._accumulate_streaming_chunks`
is deterministic — same content in any chunk boundary partition always produces
the same assembled dict — the same streaming API call always yields the same
``outputs_hash`` and cache key.
"""
from __future__ import annotations

import copy
from typing import Any, Generic, Iterator, List, Optional, TypeVar

__all__ = ["StreamedLLMResponse"]

T = TypeVar("T")


class StreamedLLMResponse(Generic[T]):
    """Recorded streaming LLM response.

    Returned when ``stream=True`` is passed to a wrapped provider client.
    Yields the original streaming chunks / SSE events when iterated;
    re-iteration is supported because all chunks are buffered.

    Example — OpenAI-compatible::

        with record("trace.sb") as rec:
            client = wrap_openai(openai_client, rec)
            stream = client.chat.completions.create(
                messages=[{"role": "user", "content": "hello"}],
                model="gpt-4o",
                stream=True,
            )
            # Progressive display:
            for chunk in stream:
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                print(delta.get("content", ""), end="", flush=True)

            # Access the assembled final response:
            print(stream.assembled_response["choices"][0]["message"]["content"])

            # Backward-compatible: provider-specific response object.
            oai_resp = stream.native  # OpenAIChatCompletion

    Example — Anthropic::

        stream = client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[...],
            stream=True,
        )
        for event in stream:
            print(event)  # raw SSE event dict

        msg = stream.native  # AnthropicMessage
        print(msg.content[0].text)

    Attributes
    ----------
    assembled_response
        The assembled canonical final response dict (OpenAI chat-completion
        shape).  This is the value recorded as ``llm_response`` in the trace.
    step
        A deep copy of the recorded ``llm_call`` step dict.
    native
        The provider-specific response object (``OpenAIChatCompletion``,
        ``AnthropicMessage``, …).  ``None`` if not provided.
    chunk_count
        Number of raw chunks / events in the buffered stream.
    """

    def __init__(
        self,
        raw_chunks: List[T],
        assembled_response: dict,
        step: dict,
        native: Any = None,
    ) -> None:
        self._raw_chunks: List[T] = list(raw_chunks)
        self._assembled_response: dict = copy.deepcopy(assembled_response)
        self._step: dict = copy.deepcopy(step)
        self._native: Any = native

    # ------------------------------------------------------------------
    # Iterator protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[T]:
        """Yield the original streaming chunks / events (re-iterable)."""
        return iter(self._raw_chunks)

    def __len__(self) -> int:
        return len(self._raw_chunks)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def assembled_response(self) -> dict:
        """Assembled canonical final response (OpenAI chat-completion shape).

        Identical to ``step["llm_response"]``.  The ``outputs_hash`` in the
        trace is computed over this dict, making it independent of chunk
        boundaries.
        """
        return self._assembled_response

    @property
    def step(self) -> dict:
        """Deep copy of the recorded ``llm_call`` step dict."""
        return self._step

    @property
    def native(self) -> Any:
        """Provider-specific response object for backward compatibility.

        * OpenAI / OpenAI-compatible / Azure OpenAI: :class:`OpenAIChatCompletion`
        * Anthropic: :class:`AnthropicMessage`
        * ``None`` when not provided.
        """
        return self._native

    @property
    def chunk_count(self) -> int:
        """Number of raw chunks / events buffered from the provider stream."""
        return len(self._raw_chunks)

    def __repr__(self) -> str:
        return (
            f"StreamedLLMResponse("
            f"chunk_count={self.chunk_count}, "
            f"finish_reason={self._assembled_response.get('choices', [{}])[0].get('finish_reason')!r}"
            f")"
        )
