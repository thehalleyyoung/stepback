"""Tests for production-trace ingestion: attestation, ingestion rules.

Covers:
- policy_fingerprint is stable and sensitive to policy changes
- RedactionAttestation round-trip: sign → verify
- verify_redaction_attestation rejects tampered body
- verify_redaction_attestation rejects tampered signature
- verify_redaction_attestation rejects wrong expected_public_key
- verify_redaction_attestation rejects wrong magic / format_version
- IngestionRules rejects require_attestation=True with no key
- ingest_trace_file full pipeline: scan + redact + attest
- ingest_trace_file block_if_any_findings=True blocks when PII found
- ingest_trace_file max_findings_before_block threshold respected
- ingest_trace_file with require_attestation=False skips signing
- attestation body_hash excludes both body_hash and signature fields
- policy fingerprint changes when strategy changes
- policy fingerprint changes when salt changes
- policy fingerprint changes when detector order changes
- IngestionResult fields are populated correctly
- PROTECTED_KEYS risk is documented via KeyContextDetector workaround
"""
from __future__ import annotations

import json
import os

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from stepback import record
from stepback.recorder import RecorderKey
from stepback.redact import (
    AttestationVerificationError,
    Detector,
    EMAIL_RE,
    IngestionResult,
    IngestionRules,
    KeyContextDetector,
    PrivacyReviewRequired,
    RedactionPolicy,
    STANDARD_POLICY,
    STRICT_POLICY,
    RedactionAttestation,
    _file_sha256,
    _policy_fingerprint,
    ingest_trace_file,
    sign_redaction_attestation,
    verify_redaction_attestation,
)
from stepback.testing import run_recorded_agent


# ---------------------------------------------------------------- helpers


def _make_trace(tmp_path, suffix="in.sb"):
    """Write a small agent trace and return (path, key)."""
    in_path = str(tmp_path / suffix)
    key = RecorderKey.fresh()
    with record(in_path, key=key) as rec:
        run_recorded_agent(rec)
    return in_path, key


def _fresh_signing_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


# ---------------------------------------------------------------- policy fingerprint


def test_policy_fingerprint_is_stable_across_calls():
    fp1 = _policy_fingerprint(STANDARD_POLICY)
    fp2 = _policy_fingerprint(STANDARD_POLICY)
    assert fp1 == fp2
    assert fp1.startswith("sha256:")


def test_policy_fingerprint_differs_by_name():
    p1 = RedactionPolicy(name="alpha", detectors=list(STANDARD_POLICY.detectors))
    p2 = RedactionPolicy(name="beta", detectors=list(STANDARD_POLICY.detectors))
    assert _policy_fingerprint(p1) != _policy_fingerprint(p2)


def test_policy_fingerprint_differs_by_detector_order():
    dets = list(STANDARD_POLICY.detectors)
    if len(dets) < 2:
        pytest.skip("need at least 2 detectors")
    p1 = RedactionPolicy(name="x", detectors=dets)
    p2 = RedactionPolicy(name="x", detectors=[dets[1], dets[0]] + dets[2:])
    assert _policy_fingerprint(p1) != _policy_fingerprint(p2)


def test_policy_fingerprint_differs_by_strategy():
    d_hash = Detector(name="email", pattern=EMAIL_RE, strategy="hash")
    d_mask = Detector(name="email", pattern=EMAIL_RE, strategy="mask")
    p1 = RedactionPolicy(name="x", detectors=[d_hash])
    p2 = RedactionPolicy(name="x", detectors=[d_mask])
    assert _policy_fingerprint(p1) != _policy_fingerprint(p2)


def test_policy_fingerprint_differs_by_salt():
    import os
    p1 = RedactionPolicy(name="x", detectors=[], salt=os.urandom(16))
    p2 = RedactionPolicy(name="x", detectors=[], salt=os.urandom(16))
    # Different salts → different salt_id → different fingerprint.
    assert _policy_fingerprint(p1) != _policy_fingerprint(p2)


def test_policy_fingerprint_includes_allowlist():
    p1 = RedactionPolicy(name="x", detectors=[], allowlist=frozenset({"alice@example.com"}))
    p2 = RedactionPolicy(name="x", detectors=[])
    assert _policy_fingerprint(p1) != _policy_fingerprint(p2)


def test_policy_fingerprint_callable_detector(tmp_path):
    """Callable patterns should not raise and produce a stable fingerprint."""
    kcd = KeyContextDetector()
    p = RedactionPolicy(name="x", detectors=[kcd])
    fp1 = _policy_fingerprint(p)
    fp2 = _policy_fingerprint(p)
    assert fp1 == fp2
    assert fp1.startswith("sha256:")


# ---------------------------------------------------------------- attestation sign / verify


def test_sign_and_verify_attestation(tmp_path):
    in_path, in_key = _make_trace(tmp_path)
    out_path = str(tmp_path / "out.sb")
    signing_key = _fresh_signing_key()

    rules = IngestionRules(
        policy=STANDARD_POLICY,
        require_attestation=True,
        attestation_key=signing_key,
    )
    result = ingest_trace_file(
        in_path, out_path, in_hmac_key=in_key.hmac_key, rules=rules
    )
    att = result.attestation
    assert att is not None
    assert att.magic == "stepback/redaction-attestation"
    assert att.format_version == 1
    assert att.original_trace_hash.startswith("sha256:")
    assert att.redacted_trace_hash.startswith("sha256:")
    assert att.original_trace_hash != att.redacted_trace_hash
    assert att.policy_fingerprint.startswith("sha256:")
    assert att.body_hash.startswith("sha256:")
    assert att.signature.startswith("ed25519:")

    # Verification should pass without raising.
    data = verify_redaction_attestation(att)
    assert data["magic"] == "stepback/redaction-attestation"


def test_verify_with_expected_public_key(tmp_path):
    in_path, in_key = _make_trace(tmp_path)
    signing_key = _fresh_signing_key()
    pub_hex = signing_key.public_key().public_bytes_raw().hex()

    rules = IngestionRules(
        policy=STANDARD_POLICY,
        require_attestation=True,
        attestation_key=signing_key,
    )
    result = ingest_trace_file(
        in_path, str(tmp_path / "out.sb"), in_hmac_key=in_key.hmac_key, rules=rules
    )
    att = result.attestation
    verify_redaction_attestation(att, expected_public_key=f"ed25519:{pub_hex}")


def test_verify_rejects_wrong_expected_public_key(tmp_path):
    in_path, in_key = _make_trace(tmp_path)
    signing_key = _fresh_signing_key()
    rules = IngestionRules(
        policy=STANDARD_POLICY,
        require_attestation=True,
        attestation_key=signing_key,
    )
    result = ingest_trace_file(
        in_path, str(tmp_path / "out.sb"), in_hmac_key=in_key.hmac_key, rules=rules
    )
    att = result.attestation
    other_pub = Ed25519PrivateKey.generate().public_key().public_bytes_raw().hex()
    with pytest.raises(AttestationVerificationError, match="mismatch"):
        verify_redaction_attestation(att, expected_public_key=f"ed25519:{other_pub}")


def test_verify_rejects_tampered_body(tmp_path):
    in_path, in_key = _make_trace(tmp_path)
    signing_key = _fresh_signing_key()
    rules = IngestionRules(
        policy=STANDARD_POLICY,
        require_attestation=True,
        attestation_key=signing_key,
    )
    result = ingest_trace_file(
        in_path, str(tmp_path / "out.sb"), in_hmac_key=in_key.hmac_key, rules=rules
    )
    att = result.attestation
    # Tamper with one field after signing.
    d = att.to_dict()
    d["n_redactions"] = d["n_redactions"] + 9999
    with pytest.raises(AttestationVerificationError, match="body_hash"):
        verify_redaction_attestation(d)


def test_verify_rejects_tampered_signature(tmp_path):
    in_path, in_key = _make_trace(tmp_path)
    signing_key = _fresh_signing_key()
    rules = IngestionRules(
        policy=STANDARD_POLICY,
        require_attestation=True,
        attestation_key=signing_key,
    )
    result = ingest_trace_file(
        in_path, str(tmp_path / "out.sb"), in_hmac_key=in_key.hmac_key, rules=rules
    )
    att = result.attestation
    d = att.to_dict()
    # Flip one byte of the signature.
    sig_bytes = bytes.fromhex(d["signature"].removeprefix("ed25519:"))
    flipped = bytes([sig_bytes[0] ^ 0xFF]) + sig_bytes[1:]
    d["signature"] = "ed25519:" + flipped.hex()
    with pytest.raises(AttestationVerificationError):
        verify_redaction_attestation(d)


def test_verify_rejects_wrong_magic():
    d = {
        "magic": "not-a-redaction-attestation",
        "format_version": 1,
        "body_hash": "sha256:abc",
        "signature": "ed25519:abc",
        "attestor_public_key": "ed25519:abc",
        "original_trace_hash": "sha256:abc",
        "redacted_trace_hash": "sha256:abc",
        "policy_name": "standard",
        "policy_fingerprint": "sha256:abc",
        "n_steps": 0,
        "n_redactions": 0,
        "per_detector": {},
        "redacted_at": "2026-01-01T00:00:00Z",
    }
    with pytest.raises(AttestationVerificationError, match="magic"):
        verify_redaction_attestation(d)


def test_verify_rejects_unknown_format_version():
    d = {
        "magic": "stepback/redaction-attestation",
        "format_version": 99,
        "body_hash": "sha256:abc",
        "signature": "ed25519:abc",
    }
    with pytest.raises(AttestationVerificationError, match="format_version"):
        verify_redaction_attestation(d)


def test_body_hash_excludes_both_hash_and_signature_fields(tmp_path):
    """body_hash must NOT include body_hash or signature in its preimage."""
    in_path, in_key = _make_trace(tmp_path)
    signing_key = _fresh_signing_key()
    rules = IngestionRules(
        policy=STANDARD_POLICY,
        require_attestation=True,
        attestation_key=signing_key,
    )
    result = ingest_trace_file(
        in_path, str(tmp_path / "out.sb"), in_hmac_key=in_key.hmac_key, rules=rules
    )
    att = result.attestation
    from stepback.canonical import canonical_json, sha256_hex

    body = {
        k: v
        for k, v in att.to_dict().items()
        if k not in ("body_hash", "signature")
    }
    # sha256_hex already returns "sha256:<hex>", so compare directly.
    recomputed = sha256_hex(canonical_json(body))
    assert att.body_hash == recomputed


# ---------------------------------------------------------------- IngestionRules validation


def test_ingestion_rules_require_attestation_without_key_raises():
    with pytest.raises(ValueError, match="attestation_key"):
        IngestionRules(policy=STANDARD_POLICY, require_attestation=True)


def test_ingestion_rules_no_attestation_no_key_ok():
    # Should not raise.
    rules = IngestionRules(policy=STANDARD_POLICY, require_attestation=False)
    assert rules.attestation_key is None


# ---------------------------------------------------------------- ingest_trace_file pipeline


def test_ingest_full_pipeline_returns_populated_result(tmp_path):
    in_path, in_key = _make_trace(tmp_path)
    signing_key = _fresh_signing_key()
    rules = IngestionRules(
        policy=STANDARD_POLICY,
        require_attestation=True,
        attestation_key=signing_key,
    )
    out_path = str(tmp_path / "out.sb")
    result = ingest_trace_file(in_path, out_path, in_hmac_key=in_key.hmac_key, rules=rules)

    assert isinstance(result, IngestionResult)
    assert result.redacted_path == out_path
    assert os.path.exists(out_path)
    assert result.original_trace_hash.startswith("sha256:")
    assert result.original_trace_hash == _file_sha256(in_path)
    assert result.scan_report is not None
    assert result.manifest is not None
    assert result.manifest.n_steps >= 1
    assert result.attestation is not None
    # Attestation original hash must match the file hash we computed.
    assert result.attestation.original_trace_hash == result.original_trace_hash


def test_ingest_without_attestation(tmp_path):
    in_path, in_key = _make_trace(tmp_path)
    rules = IngestionRules(policy=STANDARD_POLICY, require_attestation=False)
    out_path = str(tmp_path / "out.sb")
    result = ingest_trace_file(in_path, out_path, in_hmac_key=in_key.hmac_key, rules=rules)

    assert result.attestation is None
    assert os.path.exists(out_path)
    assert result.manifest.n_steps >= 1


def test_ingest_block_if_any_findings(tmp_path):
    """block_if_any_findings=True raises when scan finds any PII."""
    in_path, in_key = _make_trace(tmp_path)
    # The fixture agent writes IBAN GB99-9999-9999; STANDARD_POLICY will find it.
    rules = IngestionRules(
        policy=STANDARD_POLICY,
        block_if_any_findings=True,
        require_attestation=False,
    )
    with pytest.raises(PrivacyReviewRequired) as exc_info:
        ingest_trace_file(
            in_path, str(tmp_path / "out.sb"), in_hmac_key=in_key.hmac_key, rules=rules
        )
    assert exc_info.value.scan_report.n_findings > 0
    # Output file must NOT have been written on a block.
    assert not os.path.exists(str(tmp_path / "out.sb"))


def test_ingest_no_pii_passes_block_if_any_findings(tmp_path):
    """block_if_any_findings=True passes when trace has no PII findings."""
    in_path = str(tmp_path / "clean.sb")
    key = RecorderKey.fresh()
    with record(in_path, key=key) as rec:
        rec.tool_call(
            name="pure_math",
            arguments={"a": 1, "b": 2},
            executor=lambda n, args: {"result": args["a"] + args["b"]},
        )
    # A policy that matches nothing (empty detectors).
    clean_policy = RedactionPolicy(name="empty", detectors=[])
    rules = IngestionRules(
        policy=clean_policy,
        block_if_any_findings=True,
        require_attestation=False,
    )
    out_path = str(tmp_path / "out.sb")
    result = ingest_trace_file(in_path, out_path, in_hmac_key=key.hmac_key, rules=rules)
    assert result.scan_report.n_findings == 0
    assert os.path.exists(out_path)


def test_ingest_max_findings_threshold(tmp_path):
    """max_findings_before_block=0 blocks even on 1 finding."""
    in_path, in_key = _make_trace(tmp_path)
    rules = IngestionRules(
        policy=STANDARD_POLICY,
        max_findings_before_block=0,
        require_attestation=False,
    )
    with pytest.raises(PrivacyReviewRequired, match="exceed"):
        ingest_trace_file(
            in_path, str(tmp_path / "out.sb"), in_hmac_key=in_key.hmac_key, rules=rules
        )


def test_ingest_max_findings_threshold_not_exceeded(tmp_path):
    """max_findings_before_block=1000 passes when findings are well under threshold."""
    in_path, in_key = _make_trace(tmp_path)
    signing_key = _fresh_signing_key()
    rules = IngestionRules(
        policy=STANDARD_POLICY,
        max_findings_before_block=1000,
        require_attestation=True,
        attestation_key=signing_key,
    )
    out_path = str(tmp_path / "out.sb")
    result = ingest_trace_file(in_path, out_path, in_hmac_key=in_key.hmac_key, rules=rules)
    assert os.path.exists(out_path)
    assert result.attestation is not None


def test_ingest_redacted_file_is_verifiable(tmp_path):
    """The redacted .sb produced by ingest_trace_file passes verify_trace."""
    in_path, in_key = _make_trace(tmp_path)
    out_key = RecorderKey.fresh()
    rules = IngestionRules(policy=STANDARD_POLICY, require_attestation=False)
    out_path = str(tmp_path / "out.sb")
    ingest_trace_file(
        in_path, out_path,
        in_hmac_key=in_key.hmac_key,
        rules=rules,
        out_key=out_key,
    )
    from stepback.trace_reader import verify_trace
    parsed = verify_trace(out_path, out_key.hmac_key)
    assert len(parsed.steps) >= 1


def test_ingest_attestation_policy_name_and_fingerprint(tmp_path):
    in_path, in_key = _make_trace(tmp_path)
    signing_key = _fresh_signing_key()
    rules = IngestionRules(
        policy=STANDARD_POLICY,
        require_attestation=True,
        attestation_key=signing_key,
    )
    result = ingest_trace_file(
        in_path, str(tmp_path / "out.sb"), in_hmac_key=in_key.hmac_key, rules=rules
    )
    att = result.attestation
    assert att.policy_name == STANDARD_POLICY.name
    expected_fp = _policy_fingerprint(STANDARD_POLICY)
    assert att.policy_fingerprint == expected_fp


def test_ingest_attestation_step_and_redaction_counts(tmp_path):
    in_path, in_key = _make_trace(tmp_path)
    signing_key = _fresh_signing_key()
    rules = IngestionRules(
        policy=STANDARD_POLICY,
        require_attestation=True,
        attestation_key=signing_key,
    )
    result = ingest_trace_file(
        in_path, str(tmp_path / "out.sb"), in_hmac_key=in_key.hmac_key, rules=rules
    )
    att = result.attestation
    manifest = result.manifest
    assert att.n_steps == manifest.n_steps
    assert att.n_redactions == manifest.n_redactions
    assert dict(att.per_detector) == dict(manifest.per_detector)


def test_privacy_review_required_carries_scan_report(tmp_path):
    in_path, in_key = _make_trace(tmp_path)
    rules = IngestionRules(
        policy=STANDARD_POLICY,
        block_if_any_findings=True,
        require_attestation=False,
    )
    with pytest.raises(PrivacyReviewRequired) as exc_info:
        ingest_trace_file(
            in_path, str(tmp_path / "out.sb"), in_hmac_key=in_key.hmac_key, rules=rules
        )
    err = exc_info.value
    assert err.scan_report is not None
    assert err.scan_report.policy_name == STANDARD_POLICY.name


def test_sign_redaction_attestation_direct(tmp_path):
    """sign_redaction_attestation can be called without ingest_trace_file."""
    in_path, in_key = _make_trace(tmp_path)
    from stepback.redact import redact_trace_file_streaming, RedactionManifest
    out_path = str(tmp_path / "out.sb")
    out_key = RecorderKey.fresh()
    manifest = redact_trace_file_streaming(
        in_path, out_path, in_hmac_key=in_key.hmac_key, policy=STANDARD_POLICY, out_key=out_key
    )
    signing_key = _fresh_signing_key()
    att = sign_redaction_attestation(in_path, out_path, manifest, STANDARD_POLICY, signing_key)
    assert att.n_steps == manifest.n_steps
    # Verify the signed attestation.
    verify_redaction_attestation(att)


def test_key_context_detector_reaches_value_under_sensitive_key(tmp_path):
    """BEARER_RE finds API-key-like tokens in dict values not protected by PROTECTED_KEYS."""
    # "api_key" is not in PROTECTED_KEYS so its value is scanned normally.
    from stepback.redact import BEARER_RE, scan_value
    bearer_det = Detector(name="bearer", pattern=BEARER_RE, strategy="mask")
    policy = RedactionPolicy.fresh("bearer_test", [bearer_det])
    # A token matching BEARER_RE pattern (sk- + 20+ alnum chars)
    value = {"api_key": "sk-verylongtoken12345678901"}
    report = scan_value(value, policy)
    assert report.n_findings >= 1, (
        f"Expected BEARER_RE to find 'sk-verylongtoken...' in dict value; "
        f"findings={report.n_findings}"
    )
