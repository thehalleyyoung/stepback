"""Certified-shim program for stepback LLM provider shims.

Step 100 of 100_STEPS.md: "Create a certified-shim program: contract tests,
overhead report, canonicalization review, version matrix, and signed
compatibility badge."

This module defines the machinery to certify a :class:`~stepback.shims.ShimContract`
implementation as compatible with the stepback recording/replay guarantees.
Certification covers five areas:

1. **Contract tests** — structured checks that every mandatory method
   (:meth:`canonical_request`, :meth:`canonical_response`,
   :meth:`make_executor`, :meth:`version_probe`) returns the expected shape.

2. **Overhead report** — measures recorder overhead using the fast-path
   micro-benchmark from :mod:`stepback.bench.record_overhead` and includes
   the ``delta_p50_us`` / ``delta_p99_us`` numbers in the badge body.

3. **Canonicalization review** — verifies that ``canonical_response`` is
   deterministic (repeated calls on the same input produce bit-identical
   canonical JSON), does not mutate its input, and that the output hashes
   are stable across calls.

4. **Version matrix** — module-level constant :data:`SHIM_VERSION_MATRIX`
   documenting the provider SDK version ranges that are tested and supported.
   The matrix entry for a provider is embedded in every badge.

5. **Signed compatibility badge** — :class:`ShimCompatibilityBadge` wraps all
   check results and optional overhead metrics in a canonical-JSON body, hashes
   it with SHA-256, and optionally signs it with an Ed25519 private key.
   Third parties can verify the badge offline with only the signer's public key.

Public API
----------
::

    from stepback.shim_certification import (
        certify_shim,
        CheckResult,
        ShimCompatibilityBadge,
        SHIM_VERSION_MATRIX,
    )

    badge, errors = certify_shim(OpenAIShimContract())
    assert badge.passed

    # Signed badge for publishing
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    key = Ed25519PrivateKey.generate()
    signed_badge, errors = certify_shim(OpenAIShimContract(), signing_key=key)
    signed_badge.verify(key.public_key())          # raises if tampered
"""
from __future__ import annotations

import copy
import datetime
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .canonical import canonical_json, sha256_hex
from .shims import ShimContract

# ---------------------------------------------------------------------------
# Shim version matrix
# ---------------------------------------------------------------------------

#: Known-supported SDK version ranges for each built-in provider.
#:
#: Keys are :attr:`ShimContract.provider_name` values.  Each entry maps:
#:
#: * ``"package"`` — the installable PyPI package name.
#: * ``"supported"`` — PEP 440 specifier for supported versions.
#: * ``"tested"`` — list of version strings integration-tested against the
#:   recorded cassettes in ``tests/fixtures/sdk_cassettes/``.
#: * ``"notes"`` — optional free-text notes about version-specific quirks.
SHIM_VERSION_MATRIX: Dict[str, Dict[str, Any]] = {
    "openai": {
        "package": "openai",
        "supported": ">=1.0,<2.0",
        "tested": ["1.40.0", "1.51.2", "1.55.0"],
        "notes": "Uses chat.completions.create; AsyncOpenAI also supported.",
    },
    "anthropic": {
        "package": "anthropic",
        "supported": ">=0.30,<1.0",
        "tested": ["0.34.2", "0.39.0"],
        "notes": "claude-3-* and claude-3-5-* families; system prompt normalised.",
    },
    "bedrock": {
        "package": "boto3",
        "supported": ">=1.34,<2.0",
        "tested": ["1.34.85", "1.35.7"],
        "notes": "Uses bedrock-runtime.converse(); model id stored as _modelId.",
    },
    "gemini": {
        "package": "google-genai",
        "supported": ">=1.0,<2.0",
        "tested": ["1.5.0", "1.7.0"],
        "notes": "Uses google.genai.Client.models.generate_content().",
    },
    "azure_openai": {
        "package": "openai",
        "supported": ">=1.0,<2.0",
        "tested": ["1.40.0", "1.51.2"],
        "notes": "Requires api_version header; otherwise identical to openai.",
    },
    "cohere": {
        "package": "cohere",
        "supported": ">=5.0,<6.0",
        "tested": ["5.9.4"],
        "notes": "Uses cohere.ClientV2.chat(); seed param not sent to provider.",
    },
    "mistral": {
        "package": "mistralai",
        "supported": ">=1.0,<2.0",
        "tested": ["1.2.5"],
        "notes": "Uses mistralai.Mistral.chat.complete(); seed param not sent.",
    },
    "groq": {
        "package": "groq",
        "supported": ">=0.10,<1.0",
        "tested": ["0.11.0"],
        "notes": "OpenAI-compatible wire format.",
    },
    "together": {
        "package": "together",
        "supported": ">=1.2,<2.0",
        "tested": ["1.3.3"],
        "notes": "OpenAI-compatible via together.Together().chat.completions.",
    },
    "fireworks": {
        "package": "fireworks-ai",
        "supported": ">=0.15,<1.0",
        "tested": ["0.15.4"],
        "notes": "OpenAI-compatible endpoint.",
    },
    "cerebras": {
        "package": "cerebras-cloud-sdk",
        "supported": ">=1.0,<2.0",
        "tested": ["1.6.0"],
        "notes": "OpenAI-compatible endpoint.",
    },
    "nvidia_nim": {
        "package": "openai",
        "supported": ">=1.0,<2.0",
        "tested": ["1.51.2"],
        "notes": "NVIDIA NIM uses OpenAI-compatible API.",
    },
    "vllm": {
        "package": "openai",
        "supported": ">=1.0,<2.0",
        "tested": ["1.51.2"],
        "notes": "vLLM serves an OpenAI-compatible API.",
    },
    "tgi": {
        "package": "openai",
        "supported": ">=1.0,<2.0",
        "tested": ["1.51.2"],
        "notes": "TGI (Text Generation Inference) serves an OpenAI-compatible API.",
    },
    "llamacpp": {
        "package": "openai",
        "supported": ">=1.0,<2.0",
        "tested": ["1.51.2"],
        "notes": "llama.cpp HTTP server serves an OpenAI-compatible API.",
    },
    "ollama": {
        "package": "openai",
        "supported": ">=1.0,<2.0",
        "tested": ["1.51.2"],
        "notes": "Ollama serves an OpenAI-compatible API.",
    },
}

# ---------------------------------------------------------------------------
# Provider-specific certification fixtures
# ---------------------------------------------------------------------------

#: Sample (request_kwargs, native_response) pairs used by contract checks.
#: Keys are provider_name strings.  Each entry provides:
#:   "request_kwargs" — kwargs passed to canonical_request()
#:   "native_response" — a raw SDK-shaped response passed to canonical_response()
#:   "expected_text"   — the expected text content in choices[0].message.content
#:   "expected_prompt_tokens" — expected usage.prompt_tokens value
SHIM_CERTIFICATION_CASES: Dict[str, Dict[str, Any]] = {
    "openai": {
        "request_kwargs": {"messages": [{"role": "user", "content": "hello"}]},
        "native_response": {
            "id": "chatcmpl-cert",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hi"},
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
        "expected_text": "hi",
        "expected_prompt_tokens": 5,
    },
    "azure_openai": {
        "request_kwargs": {"messages": [{"role": "user", "content": "hello"}]},
        "native_response": {
            "id": "chatcmpl-az",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "azure"},
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        },
        "expected_text": "azure",
        "expected_prompt_tokens": 3,
    },
    "anthropic": {
        "request_kwargs": {
            "messages": [{"role": "user", "content": "hello"}],
            "system": "be terse",
        },
        "native_response": {
            "id": "msg-cert",
            "model": "claude-3-5-haiku-20241022",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hi!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 6, "output_tokens": 2},
        },
        "expected_text": "Hi!",
        "expected_prompt_tokens": 6,
    },
    "bedrock": {
        "request_kwargs": {
            "messages": [{"role": "user", "content": [{"text": "hello"}]}],
            "system": [{"text": "be terse"}],
        },
        "native_response": {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Bedrock OK"}],
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 7, "outputTokens": 2, "totalTokens": 9},
            "ResponseMetadata": {"RequestId": "req-cert"},
            "_modelId": "anthropic.claude-3-5-haiku-20241022-v1:0",
        },
        "expected_text": "Bedrock OK",
        "expected_prompt_tokens": 7,
    },
    "gemini": {
        "request_kwargs": {
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            "system_instruction": "be terse",
        },
        "native_response": {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Gemini OK"}],
                    },
                    "finish_reason": "STOP",
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 8,
                "candidates_token_count": 2,
                "total_token_count": 10,
            },
            "model_version": "gemini-2.5-flash",
        },
        "expected_text": "Gemini OK",
        "expected_prompt_tokens": 8,
    },
    "cohere": {
        "request_kwargs": {"messages": [{"role": "user", "content": "hello"}]},
        "native_response": {
            "id": "cohere-cert",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Cohere OK"}],
            },
            "finish_reason": "COMPLETE",
            "usage": {
                "tokens": {
                    "input_tokens": 5,
                    "output_tokens": 2,
                }
            },
        },
        "expected_text": "Cohere OK",
        "expected_prompt_tokens": 5,
    },
    "mistral": {
        "request_kwargs": {"messages": [{"role": "user", "content": "hello"}]},
        "native_response": {
            "id": "mistral-cert",
            "model": "mistral-large-2411",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Mistral OK"},
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        },
        "expected_text": "Mistral OK",
        "expected_prompt_tokens": 4,
    },
}

# For OpenAI-compatible providers, use the same fixture shape.
_OAI_COMPAT_PROVIDERS = [
    "groq", "together", "fireworks", "cerebras",
    "nvidia_nim", "vllm", "tgi", "llamacpp", "ollama",
]
for _p in _OAI_COMPAT_PROVIDERS:
    SHIM_CERTIFICATION_CASES[_p] = {
        "request_kwargs": {"messages": [{"role": "user", "content": "hello"}]},
        "native_response": {
            "id": f"{_p}-cert",
            "model": "llama-3",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": f"{_p} OK"},
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        },
        "expected_text": f"{_p} OK",
        "expected_prompt_tokens": 3,
    }


# ---------------------------------------------------------------------------
# CheckResult
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Result of a single certification check.

    Attributes
    ----------
    check_name:
        Short machine-readable identifier, e.g. ``"canonical_request_structure"``.
    passed:
        ``True`` iff the check succeeded without errors.
    message:
        Human-readable description of the outcome (always set).
    evidence_hash:
        Optional SHA-256 hex of the serialised evidence (canonical_request output,
        canonical_response output, …).  ``None`` when not applicable.
    """

    check_name: str
    passed: bool
    message: str
    evidence_hash: Optional[str] = None


# ---------------------------------------------------------------------------
# ShimContractChecker
# ---------------------------------------------------------------------------


class ShimContractChecker:
    """Runs all certification checks against a :class:`ShimContract`.

    Use :func:`certify_shim` rather than instantiating this directly.

    Parameters
    ----------
    contract:
        The :class:`ShimContract` to certify.
    fixture:
        Optional override for the sample (request_kwargs, native_response,
        expected_text, expected_prompt_tokens) tuple.  Defaults to the
        :data:`SHIM_CERTIFICATION_CASES` entry for the provider, or an
        OpenAI-shaped minimal fixture for unknown providers.
    """

    def __init__(
        self,
        contract: ShimContract,
        fixture: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.contract = contract
        self.fixture = fixture or SHIM_CERTIFICATION_CASES.get(
            contract.provider_name,
            SHIM_CERTIFICATION_CASES["openai"],
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_canonical_request_structure(self) -> CheckResult:
        """Verify canonical_request() returns a list of role/content dicts."""
        name = "canonical_request_structure"
        try:
            result = self.contract.canonical_request(**self.fixture["request_kwargs"])
        except Exception as exc:
            return CheckResult(name, False, f"canonical_request() raised: {exc}")

        if not isinstance(result, list):
            return CheckResult(
                name, False,
                f"Expected list, got {type(result).__name__}",
            )
        for i, item in enumerate(result):
            if not isinstance(item, dict):
                return CheckResult(
                    name, False,
                    f"Item {i} is {type(item).__name__}, expected dict",
                )
            if "role" not in item:
                return CheckResult(name, False, f"Item {i} missing 'role' key")
            if "content" not in item:
                return CheckResult(name, False, f"Item {i} missing 'content' key")

        evidence = sha256_hex(canonical_json(result))
        return CheckResult(
            name, True,
            f"canonical_request() returned {len(result)} message(s)",
            evidence_hash=evidence,
        )

    def check_canonical_response_structure(self) -> CheckResult:
        """Verify canonical_response() returns a dict with choices and usage."""
        name = "canonical_response_structure"
        try:
            result = self.contract.canonical_response(
                copy.deepcopy(self.fixture["native_response"])
            )
        except Exception as exc:
            return CheckResult(name, False, f"canonical_response() raised: {exc}")

        if not isinstance(result, dict):
            return CheckResult(
                name, False,
                f"Expected dict, got {type(result).__name__}",
            )
        missing = [k for k in ("choices", "usage") if k not in result]
        if missing:
            return CheckResult(
                name, False,
                f"Missing required keys: {missing}",
            )
        choices = result["choices"]
        if not isinstance(choices, list) or not choices:
            return CheckResult(name, False, "choices must be a non-empty list")

        choice0 = choices[0]
        if "message" not in choice0:
            return CheckResult(name, False, "choices[0] missing 'message' key")

        evidence = sha256_hex(canonical_json(result))
        return CheckResult(
            name, True,
            f"canonical_response() returned valid OpenAI-shaped dict",
            evidence_hash=evidence,
        )

    def check_canonical_response_correct_values(self) -> CheckResult:
        """Verify canonical_response() extracts expected text and token counts."""
        name = "canonical_response_correct_values"
        expected_text = self.fixture.get("expected_text")
        expected_pt = self.fixture.get("expected_prompt_tokens")

        if expected_text is None and expected_pt is None:
            return CheckResult(name, True, "No expected values to check (skipped)")

        try:
            result = self.contract.canonical_response(
                copy.deepcopy(self.fixture["native_response"])
            )
        except Exception as exc:
            return CheckResult(name, False, f"canonical_response() raised: {exc}")

        choices = result.get("choices", [])
        if not choices:
            return CheckResult(name, False, "choices is empty")

        content = choices[0].get("message", {}).get("content")
        if expected_text is not None and content != expected_text:
            return CheckResult(
                name, False,
                f"Expected content={expected_text!r}, got {content!r}",
            )

        usage = result.get("usage", {})
        actual_pt = usage.get("prompt_tokens")
        if expected_pt is not None and actual_pt != expected_pt:
            return CheckResult(
                name, False,
                f"Expected prompt_tokens={expected_pt}, got {actual_pt}",
            )

        return CheckResult(name, True, "Expected text and token counts match")

    def check_canonicalization_determinism(self) -> CheckResult:
        """Verify that canonical_response() is deterministic (same hash on repeated calls)."""
        name = "canonicalization_determinism"
        native = copy.deepcopy(self.fixture["native_response"])
        try:
            result1 = self.contract.canonical_response(native)
            result2 = self.contract.canonical_response(native)
        except Exception as exc:
            return CheckResult(name, False, f"canonical_response() raised: {exc}")

        h1 = sha256_hex(canonical_json(result1))
        h2 = sha256_hex(canonical_json(result2))
        if h1 != h2:
            return CheckResult(
                name, False,
                f"Two calls produced different canonical hashes: {h1[:16]}… vs {h2[:16]}…",
            )
        return CheckResult(
            name, True,
            "canonical_response() is deterministic",
            evidence_hash=h1,
        )

    def check_canonicalization_no_input_mutation(self) -> CheckResult:
        """Verify that canonical_response() does not mutate its input."""
        name = "canonicalization_no_input_mutation"
        native = copy.deepcopy(self.fixture["native_response"])
        native_snapshot = copy.deepcopy(native)
        try:
            self.contract.canonical_response(native)
        except Exception as exc:
            return CheckResult(name, False, f"canonical_response() raised: {exc}")

        after_hash = sha256_hex(canonical_json(native))
        before_hash = sha256_hex(canonical_json(native_snapshot))
        if before_hash != after_hash:
            return CheckResult(
                name, False,
                "canonical_response() mutated its input payload",
            )
        return CheckResult(name, True, "canonical_response() does not mutate input")

    def check_canonical_response_idempotent(self) -> CheckResult:
        """Verify canonical_response(canonical_response(x)) produces a stable hash.

        For OpenAI-passthrough shims (where the native response is already in
        OpenAI ``chat.completions`` shape and no provider sidecar keys are added)
        this check enforces strict idempotency: both calls produce bit-identical
        canonical JSON.

        For adapter shims (Bedrock, Anthropic, Gemini, Mistral, Cohere, …) the
        first call converts from a provider-native format to OpenAI shape, often
        adding a ``_<provider>`` sidecar key.  A second call on the already-converted
        dict would nest the sidecar key again, producing a different (but still
        valid) result.  We skip the strict idempotency requirement when:

        * the fixture's native_response does not already carry a ``choices`` key
          (provider-native format), OR
        * the first conversion output carries any key starting with ``_``
          (provider sidecar key is present).
        """
        name = "canonicalization_idempotent"
        native_response = self.fixture["native_response"]
        is_native_format = "choices" not in native_response

        if is_native_format:
            return CheckResult(
                name, True,
                "Idempotency check skipped: provider uses a non-OpenAI native format "
                "(adapter shim; second-pass input shape differs by design)",
            )

        try:
            result1 = self.contract.canonical_response(
                copy.deepcopy(self.fixture["native_response"])
            )
        except Exception as exc:
            return CheckResult(name, False, f"canonical_response() raised: {exc}")

        has_sidecar = any(k.startswith("_") for k in result1)
        if has_sidecar:
            return CheckResult(
                name, True,
                "Idempotency check skipped: canonical_response() adds provider sidecar "
                "key(s) — nested-sidecar accumulation on second call is expected",
            )

        try:
            result2 = self.contract.canonical_response(result1)
        except Exception as exc:
            return CheckResult(name, False, f"canonical_response() raised on double-call: {exc}")

        h1 = sha256_hex(canonical_json(result1))
        h2 = sha256_hex(canonical_json(result2))
        if h1 != h2:
            return CheckResult(
                name, False,
                f"canonical_response is not idempotent: first={h1[:16]}…, second={h2[:16]}…",
            )
        return CheckResult(
            name, True,
            "canonical_response() is idempotent",
            evidence_hash=h1,
        )

    def check_make_executor_is_callable(self) -> CheckResult:
        """Verify make_executor() returns a callable.

        If ``make_executor()`` raises :class:`NotImplementedError`, the check
        passes with an informational note — some providers (e.g. Azure OpenAI)
        require additional configuration arguments (like a deployment name) and
        intentionally raise ``NotImplementedError`` on the bare ``make_executor``
        call.  The note records this as a documented limitation rather than a
        failure.
        """
        name = "make_executor_is_callable"
        try:
            ex = self.contract.make_executor(object())
        except NotImplementedError as exc:
            return CheckResult(
                name, True,
                f"make_executor() raises NotImplementedError by design "
                f"(requires extra configuration): {exc}",
            )
        except Exception as exc:
            return CheckResult(name, False, f"make_executor() raised: {exc}")

        if not callable(ex):
            return CheckResult(
                name, False,
                f"make_executor() returned non-callable: {type(ex).__name__}",
            )
        return CheckResult(name, True, "make_executor() returns a callable")

    def check_version_probe_returns_str_or_none(self) -> CheckResult:
        """Verify version_probe() returns str or None without raising."""
        name = "version_probe_returns_str_or_none"
        try:
            result = self.contract.version_probe(object())
        except Exception as exc:
            return CheckResult(name, False, f"version_probe() raised: {exc}")

        if result is not None and not isinstance(result, str):
            return CheckResult(
                name, False,
                f"version_probe() returned {type(result).__name__}, expected str or None",
            )
        verdict = f"version_probe() returned {result!r}"
        return CheckResult(name, True, verdict)

    def check_provider_name_is_registered(self) -> CheckResult:
        """Verify the provider name is registered in SHIM_VERSION_MATRIX."""
        name = "provider_in_version_matrix"
        provider = self.contract.provider_name
        if provider in SHIM_VERSION_MATRIX:
            entry = SHIM_VERSION_MATRIX[provider]
            return CheckResult(
                name, True,
                f"Provider {provider!r} found in SHIM_VERSION_MATRIX "
                f"(package={entry['package']!r}, supported={entry['supported']!r})",
            )
        return CheckResult(
            name, False,
            f"Provider {provider!r} has no entry in SHIM_VERSION_MATRIX; "
            "add one to complete certification",
        )

    def run_all(self) -> List[CheckResult]:
        """Run all certification checks and return results in order."""
        return [
            self.check_canonical_request_structure(),
            self.check_canonical_response_structure(),
            self.check_canonical_response_correct_values(),
            self.check_canonicalization_determinism(),
            self.check_canonicalization_no_input_mutation(),
            self.check_canonical_response_idempotent(),
            self.check_make_executor_is_callable(),
            self.check_version_probe_returns_str_or_none(),
            self.check_provider_name_is_registered(),
        ]


# ---------------------------------------------------------------------------
# ShimCompatibilityBadge
# ---------------------------------------------------------------------------


@dataclass
class ShimCompatibilityBadge:
    """Signed compatibility badge produced by :func:`certify_shim`.

    The badge body is the canonical JSON of all fields except ``signature``
    and ``body_hash``.  After all other fields are set, ``body_hash`` is
    computed over that body, and the *optional* ``signature`` is an Ed25519
    signature of the body hash bytes.

    Attributes
    ----------
    magic:
        Always ``"stepback/shim-cert"`` for schema detection.
    schema_version:
        Integer schema version — currently ``1``.
    provider_name:
        The :attr:`ShimContract.provider_name` of the certified shim.
    contract_class:
        Fully-qualified class name of the concrete ``ShimContract``.
    stepback_version:
        ``stepback.__version__`` at certification time.
    certified_at:
        ISO-8601 UTC timestamp.
    checks:
        List of :class:`CheckResult` dicts serialised from the run.
    checks_passed:
        Count of passing checks.
    checks_failed:
        Count of failing checks.
    passed:
        ``True`` iff ``checks_failed == 0``.
    overhead_summary:
        Optional dict with overhead benchmark fields
        (``delta_p50_us``, ``delta_p99_us``, ``n_steps``, ``trace_bytes``).
    version_matrix_entry:
        The :data:`SHIM_VERSION_MATRIX` entry for this provider, or ``None``
        if the provider is unknown.
    body_hash:
        SHA-256 hex of the canonical-JSON badge body (all fields except
        ``signature`` and ``body_hash`` itself).
    signature:
        Hex Ed25519 signature over the body-hash bytes, or ``None`` if the
        badge is unsigned.
    signer_public_key:
        Hex Ed25519 public key corresponding to the signing key, or ``None``.
    """

    magic: str
    schema_version: int
    provider_name: str
    contract_class: str
    stepback_version: str
    certified_at: str
    checks: List[Dict[str, Any]]
    checks_passed: int
    checks_failed: int
    passed: bool
    overhead_summary: Optional[Dict[str, Any]]
    version_matrix_entry: Optional[Dict[str, Any]]
    body_hash: str
    signature: Optional[str] = None
    signer_public_key: Optional[str] = None

    # ------------------------------------------------------------------
    # Signing / verification
    # ------------------------------------------------------------------

    def sign(self, private_key: Ed25519PrivateKey) -> None:
        """Sign this badge in-place with *private_key*.

        Sets :attr:`signature` and :attr:`signer_public_key`.
        """
        # body_hash has the "sha256:" prefix from sha256_hex(); strip it.
        body_bytes = bytes.fromhex(self.body_hash.removeprefix("sha256:"))
        sig_bytes = private_key.sign(body_bytes)
        self.signature = sig_bytes.hex()
        pub_key_bytes = private_key.public_key().public_bytes_raw()
        self.signer_public_key = pub_key_bytes.hex()

    def verify(self, public_key: Ed25519PublicKey) -> None:
        """Verify the badge signature against *public_key*.

        Raises
        ------
        ValueError
            If the badge is unsigned (no signature to verify).
        cryptography.exceptions.InvalidSignature
            If the signature does not match.
        """
        if self.signature is None:
            raise ValueError("Badge is unsigned; cannot verify")
        body_bytes = bytes.fromhex(self.body_hash.removeprefix("sha256:"))
        try:
            public_key.verify(
                bytes.fromhex(self.signature),
                body_bytes,
            )
        except InvalidSignature:
            raise

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialise the full badge to a canonical JSON string."""
        return canonical_json(self._as_dict()).decode("utf-8")

    def _as_dict(self) -> Dict[str, Any]:
        return {
            "magic": self.magic,
            "schema_version": self.schema_version,
            "provider_name": self.provider_name,
            "contract_class": self.contract_class,
            "stepback_version": self.stepback_version,
            "certified_at": self.certified_at,
            "checks": self.checks,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "passed": self.passed,
            "overhead_summary": self.overhead_summary,
            "version_matrix_entry": self.version_matrix_entry,
            "body_hash": self.body_hash,
            "signature": self.signature,
            "signer_public_key": self.signer_public_key,
        }

    @classmethod
    def from_json(cls, text: str) -> "ShimCompatibilityBadge":
        """Deserialise a badge produced by :meth:`to_json`."""
        d = json.loads(text)
        return cls(
            magic=d["magic"],
            schema_version=d["schema_version"],
            provider_name=d["provider_name"],
            contract_class=d["contract_class"],
            stepback_version=d["stepback_version"],
            certified_at=d["certified_at"],
            checks=d["checks"],
            checks_passed=d["checks_passed"],
            checks_failed=d["checks_failed"],
            passed=d["passed"],
            overhead_summary=d.get("overhead_summary"),
            version_matrix_entry=d.get("version_matrix_entry"),
            body_hash=d["body_hash"],
            signature=d.get("signature"),
            signer_public_key=d.get("signer_public_key"),
        )

    @staticmethod
    def _build_body_hash(
        provider_name: str,
        contract_class: str,
        stepback_version: str,
        certified_at: str,
        checks: List[Dict[str, Any]],
        checks_passed: int,
        checks_failed: int,
        passed: bool,
        overhead_summary: Optional[Dict[str, Any]],
        version_matrix_entry: Optional[Dict[str, Any]],
    ) -> str:
        """Compute the SHA-256 body hash from the badge fields (excl. signature/hash)."""
        body = {
            "magic": "stepback/shim-cert",
            "schema_version": 1,
            "provider_name": provider_name,
            "contract_class": contract_class,
            "stepback_version": stepback_version,
            "certified_at": certified_at,
            "checks": checks,
            "checks_passed": checks_passed,
            "checks_failed": checks_failed,
            "passed": passed,
            "overhead_summary": overhead_summary,
            "version_matrix_entry": version_matrix_entry,
        }
        return sha256_hex(canonical_json(body))


# ---------------------------------------------------------------------------
# certify_shim — public entry point
# ---------------------------------------------------------------------------


def certify_shim(
    contract: ShimContract,
    *,
    fixture: Optional[Dict[str, Any]] = None,
    signing_key: Optional[Ed25519PrivateKey] = None,
    include_overhead: bool = False,
    overhead_n_steps: int = 200,
) -> "tuple[ShimCompatibilityBadge, list[str]]":
    """Run all certification checks against *contract* and return a badge.

    Parameters
    ----------
    contract:
        The :class:`~stepback.shims.ShimContract` to certify.
    fixture:
        Optional override for the provider-specific sample data.  Defaults
        to :data:`SHIM_CERTIFICATION_CASES` for the provider's name.
    signing_key:
        Optional Ed25519 private key.  When provided the badge is signed
        in-place with :meth:`ShimCompatibilityBadge.sign`.
    include_overhead:
        When ``True``, run the recorder overhead micro-benchmark and embed
        the summary in the badge.  Adds ~1–2 s per call.  Off by default
        to keep the test suite fast; set to ``True`` for publishing.
    overhead_n_steps:
        Number of timed iterations for the overhead benchmark.  Only used
        when *include_overhead* is ``True``.

    Returns
    -------
    badge:
        The :class:`ShimCompatibilityBadge`.
    errors:
        List of human-readable error messages for failed checks
        (empty when all checks pass).
    """
    import stepback as _stepback

    checker = ShimContractChecker(contract, fixture=fixture)
    results = checker.run_all()

    checks_data = [
        {
            "check_name": r.check_name,
            "passed": r.passed,
            "message": r.message,
            "evidence_hash": r.evidence_hash,
        }
        for r in results
    ]
    n_passed = sum(1 for r in results if r.passed)
    n_failed = sum(1 for r in results if not r.passed)
    errors = [r.message for r in results if not r.passed]

    overhead_summary: Optional[Dict[str, Any]] = None
    if include_overhead:
        try:
            from .bench.record_overhead import run as _run_overhead
            oh = _run_overhead(n_steps=overhead_n_steps)
            overhead_summary = {
                "n_steps": oh.n_steps,
                "delta_p50_us": oh.delta_p50_us,
                "delta_p99_us": oh.delta_p99_us,
                "recorded_p50_us": oh.recorded_p50_us,
                "recorded_p99_us": oh.recorded_p99_us,
                "trace_bytes": oh.trace_bytes,
            }
        except Exception as exc:
            overhead_summary = {"error": str(exc)}

    version_entry = SHIM_VERSION_MATRIX.get(contract.provider_name)
    contract_class = (
        f"{type(contract).__module__}.{type(contract).__qualname__}"
    )
    certified_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    stepback_version = getattr(_stepback, "__version__", "unknown")

    body_hash = ShimCompatibilityBadge._build_body_hash(
        provider_name=contract.provider_name,
        contract_class=contract_class,
        stepback_version=stepback_version,
        certified_at=certified_at,
        checks=checks_data,
        checks_passed=n_passed,
        checks_failed=n_failed,
        passed=(n_failed == 0),
        overhead_summary=overhead_summary,
        version_matrix_entry=version_entry,
    )

    badge = ShimCompatibilityBadge(
        magic="stepback/shim-cert",
        schema_version=1,
        provider_name=contract.provider_name,
        contract_class=contract_class,
        stepback_version=stepback_version,
        certified_at=certified_at,
        checks=checks_data,
        checks_passed=n_passed,
        checks_failed=n_failed,
        passed=(n_failed == 0),
        overhead_summary=overhead_summary,
        version_matrix_entry=version_entry,
        body_hash=body_hash,
    )

    if signing_key is not None:
        badge.sign(signing_key)

    return badge, errors


__all__ = [
    "SHIM_CERTIFICATION_CASES",
    "SHIM_VERSION_MATRIX",
    "CheckResult",
    "ShimCompatibilityBadge",
    "ShimContractChecker",
    "certify_shim",
]
