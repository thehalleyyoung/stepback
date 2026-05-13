"""Cassette-based compatibility tests for the Bedrock and Gemini shims.

Each test loads a recorded response from ``tests/cassettes/`` and
drives it through the shim's internal canonicalization pipeline (the
same path a real SDK response takes).  This hardens the shims against
real SDK response shapes — including edge-case content blocks like
``toolUse``, ``guardrail_intervened``, ``max_tokens``, ``SAFETY``,
``function_call``, and ``cached_content_token_count`` — without making
any network calls.

The ``COMPAT_MATRIX.json`` file lists the SDK versions whose shapes
each cassette represents.  The test at the bottom of this file
verifies that the matrix is self-consistent with the files on disk.

Note on internal helpers
------------------------
The tests call ``_bedrock_to_openai_shape`` and
``_gemini_to_openai_shape`` directly because those are the functions
the shims use on every response path; testing them with cassette data
is the most direct proof that real SDK shapes survive canonicalization.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional

import pytest

from stepback.shims import (
    GeminiResponse,
    _bedrock_to_openai_shape,
    _coerce_bedrock_response,
    _coerce_gemini_response,
    _gemini_to_openai_shape,
    canonical_bedrock_model_id,
    canonical_gemini_model_id,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CASSETTES = pathlib.Path(__file__).parent / "cassettes"


def _load(name: str) -> dict:
    return json.loads((_CASSETTES / name).read_text())


# ---------------------------------------------------------------------------
# Bedrock cassette tests
# ---------------------------------------------------------------------------


class TestBedrockCassettesTextCompletion:
    """bedrock_converse_text_v1.json — simple text response."""

    def test_coerce_preserves_all_top_level_keys(self) -> None:
        raw = _load("bedrock_converse_text_v1.json")
        coerced = _coerce_bedrock_response(raw)
        for key in ("ResponseMetadata", "output", "stopReason", "usage", "metrics"):
            assert key in coerced, f"missing key: {key}"

    def test_bedrock_to_openai_shape_text(self) -> None:
        raw = _load("bedrock_converse_text_v1.json")
        coerced = _coerce_bedrock_response(raw)
        # Inject the modelId so the canonical shape carries it.
        coerced["_modelId"] = raw["_meta"]["model_id"]
        out = _bedrock_to_openai_shape(coerced)

        assert out["choices"][0]["finish_reason"] == "stop"
        msg = out["choices"][0]["message"]
        assert msg["role"] == "assistant"
        assert "Paris" in (msg["content"] or "")
        assert out["usage"]["prompt_tokens"] == 28
        assert out["usage"]["completion_tokens"] == 9
        assert out["usage"]["total_tokens"] == 37
        # Native payload preserved for rehydration.
        assert "_bedrock" in out

    def test_model_id_alias_resolves(self) -> None:
        raw = _load("bedrock_converse_text_v1.json")
        canonical = canonical_bedrock_model_id(raw["_meta"]["model_id"])
        assert canonical == "claude-3-5-haiku-20241022"


class TestBedrockCassettesToolUse:
    """bedrock_converse_tool_use_v1.json — toolUse content block."""

    def test_tool_use_canonicalises_to_openai_tool_calls(self) -> None:
        raw = _load("bedrock_converse_tool_use_v1.json")
        coerced = _coerce_bedrock_response(raw)
        coerced["_modelId"] = raw["_meta"]["model_id"]
        out = _bedrock_to_openai_shape(coerced)

        assert out["choices"][0]["finish_reason"] == "tool_calls"
        msg = out["choices"][0]["message"]
        # Text part preserved alongside tool call.
        assert "Paris" in (msg.get("content") or "")
        tool_calls = msg["tool_calls"]
        assert len(tool_calls) == 1
        tc = tool_calls[0]
        assert tc["id"] == "tooluse_abc123xyz"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == {"location": "Paris"}

    def test_model_id_alias_resolves(self) -> None:
        raw = _load("bedrock_converse_tool_use_v1.json")
        canonical = canonical_bedrock_model_id(raw["_meta"]["model_id"])
        assert canonical == "claude-3-5-sonnet-20241022"


class TestBedrockCassettesNativeLlama:
    """bedrock_converse_native_llama_v1.json — native Bedrock model."""

    def test_native_model_id_passes_through(self) -> None:
        raw = _load("bedrock_converse_native_llama_v1.json")
        model_id = raw["_meta"]["model_id"]
        assert canonical_bedrock_model_id(model_id) == model_id

    def test_text_content_extracted(self) -> None:
        raw = _load("bedrock_converse_native_llama_v1.json")
        coerced = _coerce_bedrock_response(raw)
        coerced["_modelId"] = raw["_meta"]["model_id"]
        out = _bedrock_to_openai_shape(coerced)

        assert out["choices"][0]["finish_reason"] == "stop"
        assert "gravity" in (out["choices"][0]["message"]["content"] or "").lower()
        assert out["usage"]["prompt_tokens"] == 19


class TestBedrockCassettesMaxTokens:
    """bedrock_converse_max_tokens_v1.json — stopReason=max_tokens."""

    def test_max_tokens_maps_to_length(self) -> None:
        raw = _load("bedrock_converse_max_tokens_v1.json")
        coerced = _coerce_bedrock_response(raw)
        out = _bedrock_to_openai_shape(coerced)
        assert out["choices"][0]["finish_reason"] == "length"

    def test_partial_content_preserved(self) -> None:
        raw = _load("bedrock_converse_max_tokens_v1.json")
        coerced = _coerce_bedrock_response(raw)
        out = _bedrock_to_openai_shape(coerced)
        assert out["choices"][0]["message"]["content"] == "Once upon a"


class TestBedrockCassettesGuardrail:
    """bedrock_converse_guardrail_v1.json — stopReason=guardrail_intervened."""

    def test_guardrail_maps_to_content_filter(self) -> None:
        raw = _load("bedrock_converse_guardrail_v1.json")
        coerced = _coerce_bedrock_response(raw)
        out = _bedrock_to_openai_shape(coerced)
        assert out["choices"][0]["finish_reason"] == "content_filter"

    def test_partial_text_still_returned(self) -> None:
        raw = _load("bedrock_converse_guardrail_v1.json")
        coerced = _coerce_bedrock_response(raw)
        out = _bedrock_to_openai_shape(coerced)
        assert "Sorry" in (out["choices"][0]["message"]["content"] or "")


# ---------------------------------------------------------------------------
# Bedrock: duck-typed object (model_dump / attribute) coercion paths
# ---------------------------------------------------------------------------


class TestBedrockCoerceDuckTyped:
    """_coerce_bedrock_response handles pydantic-style model_dump objects."""

    def test_model_dump_path(self) -> None:
        raw = _load("bedrock_converse_text_v1.json")

        class _FakeModel:
            def model_dump(self) -> dict:
                return dict(raw)

        coerced = _coerce_bedrock_response(_FakeModel())
        assert coerced["stopReason"] == "end_turn"

    def test_to_dict_path(self) -> None:
        raw = _load("bedrock_converse_native_llama_v1.json")

        class _FakeSDKObj:
            def to_dict(self) -> dict:
                return dict(raw)

        coerced = _coerce_bedrock_response(_FakeSDKObj())
        assert "output" in coerced

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError, match="unsupported Bedrock response type"):
            _coerce_bedrock_response(42)


# ---------------------------------------------------------------------------
# Gemini cassette tests
# ---------------------------------------------------------------------------


class TestGeminiCassettesTextCompletion:
    """gemini_generate_content_text_v1.json — simple text response."""

    def test_coerce_preserves_top_level_keys(self) -> None:
        raw = _load("gemini_generate_content_text_v1.json")
        coerced = _coerce_gemini_response(raw)
        for key in ("candidates", "usage_metadata", "model_version"):
            assert key in coerced, f"missing key: {key}"

    def test_gemini_to_openai_shape_text(self) -> None:
        raw = _load("gemini_generate_content_text_v1.json")
        coerced = _coerce_gemini_response(raw)
        out = _gemini_to_openai_shape(coerced)

        assert out["choices"][0]["finish_reason"] == "stop"
        msg = out["choices"][0]["message"]
        assert msg["role"] == "assistant"
        assert "Paris" in (msg["content"] or "")
        assert out["usage"]["prompt_tokens"] == 22
        assert out["usage"]["completion_tokens"] == 8
        assert out["usage"]["total_tokens"] == 30
        assert "_gemini" in out

    def test_model_id_alias_resolves(self) -> None:
        raw = _load("gemini_generate_content_text_v1.json")
        model = raw["model_version"]
        assert canonical_gemini_model_id(model) == model  # already canonical


class TestGeminiCassettesFunctionCall:
    """gemini_generate_content_function_call_v1.json — function_call part."""

    def test_function_call_canonicalises_to_tool_calls(self) -> None:
        raw = _load("gemini_generate_content_function_call_v1.json")
        coerced = _coerce_gemini_response(raw)
        out = _gemini_to_openai_shape(coerced)

        # MALFORMED_FUNCTION_CALL maps to "tool_calls" (see _GEMINI_FINISH_MAP).
        assert out["choices"][0]["finish_reason"] == "tool_calls"
        msg = out["choices"][0]["message"]
        # Text part preserved.
        assert "Paris" in (msg.get("content") or "")
        tool_calls = msg["tool_calls"]
        assert len(tool_calls) == 1
        tc = tool_calls[0]
        assert tc["id"] == "call_abc123xyz"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == {"location": "Paris"}


class TestGeminiCassettesCachedTokens:
    """gemini_generate_content_cached_tokens_v1.json — cached_content_token_count."""

    def test_cached_tokens_mapped_to_prompt_tokens_details(self) -> None:
        raw = _load("gemini_generate_content_cached_tokens_v1.json")
        coerced = _coerce_gemini_response(raw)
        out = _gemini_to_openai_shape(coerced)

        usage = out["usage"]
        assert usage["prompt_tokens"] == 5120
        assert usage["completion_tokens"] == 19
        # Cached tokens exposed in prompt_tokens_details.
        assert usage.get("prompt_tokens_details", {}).get("cached_tokens") == 5000

    def test_model_id_alias_resolves(self) -> None:
        raw = _load("gemini_generate_content_cached_tokens_v1.json")
        model = raw["model_version"]
        assert canonical_gemini_model_id(model) == model


class TestGeminiCassettesSafetyBlocked:
    """gemini_generate_content_safety_blocked_v1.json — SAFETY finish_reason."""

    def test_safety_finish_reason_maps_to_content_filter(self) -> None:
        raw = _load("gemini_generate_content_safety_blocked_v1.json")
        coerced = _coerce_gemini_response(raw)
        out = _gemini_to_openai_shape(coerced)
        assert out["choices"][0]["finish_reason"] == "content_filter"

    def test_empty_parts_produces_none_content(self) -> None:
        raw = _load("gemini_generate_content_safety_blocked_v1.json")
        coerced = _coerce_gemini_response(raw)
        out = _gemini_to_openai_shape(coerced)
        # No text parts → content is None; _strip_none removes the key entirely.
        assert out["choices"][0]["message"].get("content") is None

    def test_zero_completion_tokens(self) -> None:
        raw = _load("gemini_generate_content_safety_blocked_v1.json")
        coerced = _coerce_gemini_response(raw)
        out = _gemini_to_openai_shape(coerced)
        assert out["usage"]["completion_tokens"] == 0


class TestGeminiCassettesVertexShape:
    """gemini_generate_content_vertex_v1.json — Vertex AI extra metadata fields."""

    def test_vertex_extra_fields_do_not_break_canonicalization(self) -> None:
        raw = _load("gemini_generate_content_vertex_v1.json")
        coerced = _coerce_gemini_response(raw)
        out = _gemini_to_openai_shape(coerced)

        assert out["choices"][0]["finish_reason"] == "stop"
        assert "artificial intelligence" in (
            out["choices"][0]["message"]["content"] or ""
        ).lower()

    def test_vertex_citation_metadata_ignored_gracefully(self) -> None:
        """citation_metadata on candidate must not raise."""
        raw = _load("gemini_generate_content_vertex_v1.json")
        # Confirm the cassette has citation_metadata.
        assert "citation_metadata" in raw["candidates"][0]
        coerced = _coerce_gemini_response(raw)
        out = _gemini_to_openai_shape(coerced)
        # Canonicalization must not surface citation_metadata as a
        # top-level key (it should be buried inside _gemini only).
        assert "citation_metadata" not in out


# ---------------------------------------------------------------------------
# Gemini: duck-typed object (model_dump / attribute) coercion paths
# ---------------------------------------------------------------------------


@dataclass
class _FakePart:
    text: str


@dataclass
class _FakeContent:
    role: str
    parts: List[_FakePart]


@dataclass
class _FakeCandidate:
    content: _FakeContent
    finish_reason: str
    index: int = 0


@dataclass
class _FakeUsageMeta:
    prompt_token_count: int
    candidates_token_count: int
    total_token_count: int


@dataclass
class _FakeSDKResponse:
    """Mimics a pydantic / dataclass SDK response object (duck-typed, no model_dump)."""
    candidates: List[_FakeCandidate]
    usage_metadata: _FakeUsageMeta
    model_version: str


class TestGeminiCoerceDuckTyped:
    """_coerce_gemini_response handles attribute-based SDK objects."""

    def test_dataclass_response_coerces_correctly(self) -> None:
        obj = _FakeSDKResponse(
            candidates=[
                _FakeCandidate(
                    content=_FakeContent(
                        role="model",
                        parts=[_FakePart(text="The answer is 42.")],
                    ),
                    finish_reason="STOP",
                )
            ],
            usage_metadata=_FakeUsageMeta(
                prompt_token_count=7,
                candidates_token_count=5,
                total_token_count=12,
            ),
            model_version="gemini-2.5-flash-2025-04-09",
        )
        coerced = _coerce_gemini_response(obj)
        out = _gemini_to_openai_shape(coerced)
        assert out["choices"][0]["finish_reason"] == "stop"
        assert "42" in (out["choices"][0]["message"]["content"] or "")
        assert out["usage"]["prompt_tokens"] == 7

    def test_model_dump_path(self) -> None:
        raw = _load("gemini_generate_content_text_v1.json")

        class _FakePydantic:
            def model_dump(self) -> dict:
                return dict(raw)

        coerced = _coerce_gemini_response(_FakePydantic())
        assert "candidates" in coerced

    def test_to_dict_path(self) -> None:
        raw = _load("gemini_generate_content_text_v1.json")

        class _FakeSDK:
            def to_dict(self) -> dict:
                return dict(raw)

        coerced = _coerce_gemini_response(_FakeSDK())
        assert "candidates" in coerced


# ---------------------------------------------------------------------------
# GeminiResponse namespace
# ---------------------------------------------------------------------------


class TestGeminiResponseNamespace:
    """GeminiResponse.from_native_and_canonical correctly populates all fields."""

    def _make(self, cassette: str) -> GeminiResponse:
        raw = _load(cassette)
        coerced = _coerce_gemini_response(raw)
        canonical = _gemini_to_openai_shape(coerced)
        return GeminiResponse.from_native_and_canonical(coerced, canonical)

    def test_text_response_fields(self) -> None:
        resp = self._make("gemini_generate_content_text_v1.json")
        assert "Paris" in (resp.text or "")
        assert len(resp.candidates) == 1
        assert resp.candidates[0].finish_reason == "STOP"
        assert resp.usage_metadata.prompt_token_count == 22
        assert resp.usage_metadata.candidates_token_count == 8
        assert resp.model_version == "gemini-2.5-flash-2025-04-09"
        # Dict-style access returns canonical shape.
        assert resp["choices"][0]["finish_reason"] == "stop"

    def test_function_call_response_fields(self) -> None:
        resp = self._make("gemini_generate_content_function_call_v1.json")
        fcs = resp.function_calls
        assert len(fcs) == 1
        assert fcs[0]["name"] == "get_weather"
        assert fcs[0]["args"] == {"location": "Paris"}

    def test_safety_blocked_text_is_none(self) -> None:
        resp = self._make("gemini_generate_content_safety_blocked_v1.json")
        assert resp.text is None
        assert resp.usage_metadata.candidates_token_count == 0

    def test_vertex_response_no_error(self) -> None:
        resp = self._make("gemini_generate_content_vertex_v1.json")
        assert resp.text is not None
        assert resp.model_version == "gemini-2.5-pro-2025-03-25"


# ---------------------------------------------------------------------------
# Compatibility matrix consistency check
# ---------------------------------------------------------------------------


def test_compat_matrix_is_consistent() -> None:
    """Every cassette listed in COMPAT_MATRIX.json must exist on disk, and
    every cassette file in the directory must be listed in the matrix."""
    matrix = _load("COMPAT_MATRIX.json")

    listed: set[str] = set()
    for provider in ("bedrock", "gemini"):
        for fname in matrix[provider]["cassettes"]:
            listed.add(fname)
            path = _CASSETTES / fname
            assert path.exists(), (
                f"COMPAT_MATRIX references {fname} but the file does not exist"
            )

    on_disk: set[str] = {
        p.name
        for p in _CASSETTES.iterdir()
        if p.suffix == ".json" and p.name != "COMPAT_MATRIX.json"
    }
    missing_from_matrix = on_disk - listed
    assert not missing_from_matrix, (
        f"These cassette files are not listed in COMPAT_MATRIX.json: "
        f"{sorted(missing_from_matrix)}"
    )
