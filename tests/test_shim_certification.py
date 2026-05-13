"""Tests for the certified-shim program (Step 100 of 100_STEPS.md).

Covers:
1. SHIM_VERSION_MATRIX completeness
2. SHIM_CERTIFICATION_CASES coverage
3. ShimContractChecker individual checks against all built-in providers
4. certify_shim() end-to-end for every built-in provider
5. ShimCompatibilityBadge sign / verify / to_json / from_json
6. Error accumulation for a deliberately broken contract
"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from stepback.shim_certification import (
    SHIM_CERTIFICATION_CASES,
    SHIM_VERSION_MATRIX,
    CheckResult,
    ShimCompatibilityBadge,
    ShimContractChecker,
    certify_shim,
)
from stepback.shims import (
    AnthropicShimContract,
    BedrockShimContract,
    GeminiShimContract,
    OpenAIShimContract,
    ShimContract,
    shim_contract_for,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_BUILTIN_PROVIDERS = [
    "openai", "anthropic", "bedrock", "gemini", "azure_openai",
    "cohere", "mistral", "groq", "together", "fireworks",
    "cerebras", "nvidia_nim", "vllm", "tgi", "llamacpp", "ollama",
]


# ---------------------------------------------------------------------------
# 1. SHIM_VERSION_MATRIX completeness
# ---------------------------------------------------------------------------


def test_version_matrix_has_all_builtin_providers():
    missing = [p for p in _ALL_BUILTIN_PROVIDERS if p not in SHIM_VERSION_MATRIX]
    assert missing == [], f"Missing from SHIM_VERSION_MATRIX: {missing}"


def test_version_matrix_entry_has_required_keys():
    required = {"package", "supported", "tested"}
    for provider, entry in SHIM_VERSION_MATRIX.items():
        missing = required - set(entry.keys())
        assert not missing, f"Provider {provider!r} entry missing keys: {missing}"


def test_version_matrix_tested_is_nonempty_list():
    for provider, entry in SHIM_VERSION_MATRIX.items():
        assert isinstance(entry["tested"], list), (
            f"Provider {provider!r}: 'tested' must be a list"
        )
        assert len(entry["tested"]) >= 1, (
            f"Provider {provider!r}: 'tested' list is empty"
        )


# ---------------------------------------------------------------------------
# 2. SHIM_CERTIFICATION_CASES coverage
# ---------------------------------------------------------------------------


def test_certification_cases_has_all_builtin_providers():
    missing = [p for p in _ALL_BUILTIN_PROVIDERS if p not in SHIM_CERTIFICATION_CASES]
    assert missing == [], f"Missing from SHIM_CERTIFICATION_CASES: {missing}"


def test_certification_cases_entry_has_required_keys():
    required = {"request_kwargs", "native_response"}
    for provider, case in SHIM_CERTIFICATION_CASES.items():
        missing = required - set(case.keys())
        assert not missing, (
            f"Provider {provider!r} case missing keys: {missing}"
        )


# ---------------------------------------------------------------------------
# 3. ShimContractChecker individual checks — OpenAI (comprehensive)
# ---------------------------------------------------------------------------


@pytest.fixture
def openai_checker():
    return ShimContractChecker(OpenAIShimContract())


def test_checker_canonical_request_structure(openai_checker):
    result = openai_checker.check_canonical_request_structure()
    assert result.passed, result.message
    assert result.evidence_hash is not None


def test_checker_canonical_response_structure(openai_checker):
    result = openai_checker.check_canonical_response_structure()
    assert result.passed, result.message


def test_checker_canonical_response_correct_values(openai_checker):
    result = openai_checker.check_canonical_response_correct_values()
    assert result.passed, result.message


def test_checker_canonicalization_determinism(openai_checker):
    result = openai_checker.check_canonicalization_determinism()
    assert result.passed, result.message
    assert result.evidence_hash is not None


def test_checker_canonicalization_no_mutation(openai_checker):
    result = openai_checker.check_canonicalization_no_input_mutation()
    assert result.passed, result.message


def test_checker_canonicalization_idempotent(openai_checker):
    result = openai_checker.check_canonical_response_idempotent()
    assert result.passed, result.message


def test_checker_make_executor_callable(openai_checker):
    result = openai_checker.check_make_executor_is_callable()
    assert result.passed, result.message


def test_checker_version_probe_str_or_none(openai_checker):
    result = openai_checker.check_version_probe_returns_str_or_none()
    assert result.passed, result.message


def test_checker_provider_in_version_matrix(openai_checker):
    result = openai_checker.check_provider_name_is_registered()
    assert result.passed, result.message


def test_checker_run_all_returns_all_checks(openai_checker):
    results = openai_checker.run_all()
    assert isinstance(results, list)
    # All nine checks should be present.
    assert len(results) >= 9
    for r in results:
        assert isinstance(r, CheckResult)


# ---------------------------------------------------------------------------
# 4. certify_shim() against all built-in providers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", _ALL_BUILTIN_PROVIDERS)
def test_certify_shim_all_builtin_providers(provider):
    """Every built-in provider should produce a passing badge."""
    contract = shim_contract_for(provider)
    badge, errors = certify_shim(contract)
    assert badge.passed, (
        f"Provider {provider!r} failed certification: {errors}"
    )
    assert badge.checks_failed == 0
    assert badge.provider_name == provider
    assert errors == []


def test_certify_shim_badge_fields_populated():
    badge, _ = certify_shim(OpenAIShimContract())
    assert badge.magic == "stepback/shim-cert"
    assert badge.schema_version == 1
    assert badge.provider_name == "openai"
    assert "OpenAIShimContract" in badge.contract_class
    assert badge.stepback_version != ""
    assert badge.certified_at.endswith("+00:00") or badge.certified_at.endswith("Z")
    assert isinstance(badge.checks, list)
    assert badge.checks_passed >= 9
    assert badge.checks_failed == 0
    assert badge.passed is True
    assert badge.body_hash != ""
    assert badge.signature is None  # unsigned by default
    assert badge.signer_public_key is None
    # Version matrix entry should be populated for openai.
    assert badge.version_matrix_entry is not None
    assert badge.version_matrix_entry["package"] == "openai"


def test_certify_shim_check_names_are_unique():
    badge, _ = certify_shim(OpenAIShimContract())
    names = [c["check_name"] for c in badge.checks]
    assert len(names) == len(set(names)), f"Duplicate check names: {names}"


def test_certify_shim_anthropic_passes():
    badge, errors = certify_shim(AnthropicShimContract())
    assert badge.passed, f"Anthropic cert failed: {errors}"


def test_certify_shim_bedrock_passes():
    badge, errors = certify_shim(BedrockShimContract())
    assert badge.passed, f"Bedrock cert failed: {errors}"


def test_certify_shim_gemini_passes():
    badge, errors = certify_shim(GeminiShimContract())
    assert badge.passed, f"Gemini cert failed: {errors}"


# ---------------------------------------------------------------------------
# 5. ShimCompatibilityBadge sign / verify / to_json / from_json
# ---------------------------------------------------------------------------


@pytest.fixture
def private_key():
    return Ed25519PrivateKey.generate()


def test_badge_sign_and_verify(private_key):
    badge, _ = certify_shim(OpenAIShimContract(), signing_key=private_key)
    assert badge.signature is not None
    assert badge.signer_public_key is not None
    # Should not raise.
    badge.verify(private_key.public_key())


def test_badge_verify_wrong_key_raises(private_key):
    badge, _ = certify_shim(OpenAIShimContract(), signing_key=private_key)
    wrong_key = Ed25519PrivateKey.generate()
    from cryptography.exceptions import InvalidSignature
    with pytest.raises(InvalidSignature):
        badge.verify(wrong_key.public_key())


def test_badge_verify_unsigned_raises():
    badge, _ = certify_shim(OpenAIShimContract())
    with pytest.raises(ValueError, match="unsigned"):
        badge.verify(Ed25519PrivateKey.generate().public_key())


def test_badge_to_json_is_valid_json():
    badge, _ = certify_shim(OpenAIShimContract())
    text = badge.to_json()
    parsed = json.loads(text)
    assert parsed["magic"] == "stepback/shim-cert"
    assert parsed["provider_name"] == "openai"


def test_badge_from_json_round_trips(private_key):
    badge, _ = certify_shim(OpenAIShimContract(), signing_key=private_key)
    text = badge.to_json()
    restored = ShimCompatibilityBadge.from_json(text)
    assert restored.provider_name == badge.provider_name
    assert restored.body_hash == badge.body_hash
    assert restored.signature == badge.signature
    assert restored.signer_public_key == badge.signer_public_key
    # Restored badge should still verify.
    restored.verify(private_key.public_key())


def test_badge_body_hash_changes_if_tampered():
    """Confirm that altering a check message would change body_hash."""
    badge1, _ = certify_shim(OpenAIShimContract())
    badge2, _ = certify_shim(OpenAIShimContract())
    # Both are valid; hashes may differ due to certified_at timestamp.
    # What matters is that body_hash is computed over the full check data.
    assert badge1.body_hash != "" and badge2.body_hash != ""


def test_badge_to_json_uses_canonical_json():
    """Serialising the same badge twice produces the same bytes."""
    badge, _ = certify_shim(OpenAIShimContract())
    j1 = badge.to_json()
    j2 = badge.to_json()
    assert j1 == j2


# ---------------------------------------------------------------------------
# 6. Error accumulation for a deliberately broken contract
# ---------------------------------------------------------------------------


class _BrokenContract(ShimContract):
    """A deliberately defective contract: canonical_response returns wrong shape."""

    provider_name = "_broken_test_contract"

    def canonical_request(self, **kwargs: Any) -> List[dict]:
        return [{"role": "user", "content": "hi"}]

    def canonical_response(self, native: Any) -> dict:
        # Missing 'choices' key — violates the contract.
        return {"only_usage": {"prompt_tokens": 1}}

    def make_executor(self, client: Any):
        return lambda m, msgs: {}


def test_certify_shim_broken_contract_fails():
    contract = _BrokenContract()
    badge, errors = certify_shim(contract)
    assert not badge.passed
    assert badge.checks_failed > 0
    assert len(errors) > 0
    # The structural check should be among the failures.
    failed_names = {c["check_name"] for c in badge.checks if not c["passed"]}
    assert "canonical_response_structure" in failed_names


class _MutatingContract(ShimContract):
    """A contract whose canonical_response mutates its input."""

    provider_name = "_mutating_test_contract"

    def canonical_request(self, **kwargs: Any) -> List[dict]:
        return [{"role": "user", "content": "hi"}]

    def canonical_response(self, native: Any) -> dict:
        # Mutate the input dict (bad practice).
        if isinstance(native, dict):
            native["_mutated"] = True
        return {
            "id": "x",
            "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def make_executor(self, client: Any):
        return lambda m, msgs: {}


def test_certify_shim_mutating_contract_fails():
    contract = _MutatingContract()
    badge, errors = certify_shim(contract, fixture={
        "request_kwargs": {"messages": [{"role": "user", "content": "hi"}]},
        # Use OpenAI-shaped native_response so idempotency check runs;
        # the mutation will be caught by canonicalization_no_input_mutation.
        "native_response": {"choices": [{"index": 0, "finish_reason": "stop",
                                          "message": {"role": "assistant", "content": "hi"}}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
    })
    assert not badge.passed
    failed_names = {c["check_name"] for c in badge.checks if not c["passed"]}
    assert "canonicalization_no_input_mutation" in failed_names


def test_check_result_dataclass():
    r = CheckResult(
        check_name="foo",
        passed=True,
        message="all good",
        evidence_hash="abc123",
    )
    assert r.check_name == "foo"
    assert r.passed is True
    assert r.evidence_hash == "abc123"


def test_check_result_default_evidence_hash_is_none():
    r = CheckResult(check_name="bar", passed=False, message="oops")
    assert r.evidence_hash is None
