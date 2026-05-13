"""Tests for stepback.bench.witness_cosigning (Step 131).

Covers:
- sign_trace_pack_commitment: round-trip with verify_witness_commitments
- verify: tampered trace_pack_sha256 fails
- verify: tampered corpus_id fails
- verify: wrong Ed25519 key fails
- verify: duplicate identity is counted only once
- verify: unknown witness skipped when trusted_witnesses provided
- verify: trusted witness counted
- verify: untrusted-only witnesses fail threshold with trusted_witnesses set
- verify: evaluation_timestamp before committed_at raises WitnessCosigningError
- verify: evaluation_timestamp after committed_at passes
- verify: min_witnesses not met raises WitnessCosigningError
- verify: malformed sha256 raises WitnessCosigningError
- verify: normalise sha256 with and without "sha256:" prefix
- sign: default committed_at is filled in automatically
- WitnessCommitment: to_dict / from_dict round-trip

Leaderboard integration:
- build_leaderboard: submission with valid cosignature → is_development=False
- build_leaderboard: submission without cosignatures → is_development=True
- build_leaderboard: submission with invalid cosignature → is_development=True
- build_leaderboard: submission with valid cosig but git_dirty=True → is_development=True
- LeaderboardEntry.to_json: has_witness_cosignatures and witness_count present
"""
from __future__ import annotations

import datetime
import json
import uuid

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from stepback.bench.witness_cosigning import (
    CAPABILITY_WITNESS_COSIGNING,
    WitnessCommitment,
    WitnessCosigningError,
    _normalise_sha256,
    sign_trace_pack_commitment,
    verify_witness_commitments,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

_PACK_SHA256_HEX = "a" * 64
_CORPUS_ID = "synthetic-200"
_WITNESS_IDENTITY = "test-witness-alpha"


def _new_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


def _committed_at(offset_seconds: int = -60) -> str:
    """Return a UTC ISO 8601 timestamp offset_seconds from now."""
    ts = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(
        seconds=offset_seconds
    )
    return ts.isoformat()


# ---------------------------------------------------------------------------
# Unit tests for WitnessCommitment serialisation
# ---------------------------------------------------------------------------


def test_witness_commitment_round_trip() -> None:
    key = _new_key()
    commitment = sign_trace_pack_commitment(
        _PACK_SHA256_HEX,
        _CORPUS_ID,
        _WITNESS_IDENTITY,
        key,
        committed_at=_committed_at(-120),
    )
    d = commitment.to_dict()
    restored = WitnessCommitment.from_dict(d)
    assert restored.committed_at == commitment.committed_at
    assert restored.witness_identity == commitment.witness_identity
    assert restored.witness_public_key == commitment.witness_public_key
    assert restored.signature == commitment.signature


# ---------------------------------------------------------------------------
# Unit tests for _normalise_sha256
# ---------------------------------------------------------------------------


def test_normalise_raw_hex() -> None:
    assert _normalise_sha256("a" * 64) == "sha256:" + "a" * 64


def test_normalise_prefixed() -> None:
    assert _normalise_sha256("sha256:" + "b" * 64) == "sha256:" + "b" * 64


def test_normalise_uppercase_lowercased() -> None:
    assert _normalise_sha256("A" * 64) == "sha256:" + "a" * 64


def test_normalise_invalid_raises() -> None:
    with pytest.raises(WitnessCosigningError):
        _normalise_sha256("tooshort")


# ---------------------------------------------------------------------------
# Round-trip: sign then verify
# ---------------------------------------------------------------------------


def test_sign_verify_round_trip() -> None:
    key = _new_key()
    commitment = sign_trace_pack_commitment(
        _PACK_SHA256_HEX,
        _CORPUS_ID,
        _WITNESS_IDENTITY,
        key,
        committed_at=_committed_at(-60),
    )
    count = verify_witness_commitments(
        _PACK_SHA256_HEX,
        _CORPUS_ID,
        [commitment],
        min_witnesses=1,
    )
    assert count == 1


def test_sign_default_committed_at() -> None:
    """sign_trace_pack_commitment fills committed_at if not provided."""
    key = _new_key()
    commitment = sign_trace_pack_commitment(
        _PACK_SHA256_HEX,
        _CORPUS_ID,
        _WITNESS_IDENTITY,
        key,
    )
    assert commitment.committed_at != ""
    # Must be a plausible ISO 8601 string.
    datetime.datetime.fromisoformat(commitment.committed_at.replace("Z", "+00:00"))


def test_sign_with_sha256_prefix() -> None:
    """sign_trace_pack_commitment accepts 'sha256:' prefixed hash."""
    key = _new_key()
    commitment = sign_trace_pack_commitment(
        "sha256:" + _PACK_SHA256_HEX,
        _CORPUS_ID,
        _WITNESS_IDENTITY,
        key,
        committed_at=_committed_at(-60),
    )
    # Verification with the raw hex form should succeed.
    count = verify_witness_commitments(
        _PACK_SHA256_HEX,
        _CORPUS_ID,
        [commitment],
        min_witnesses=1,
    )
    assert count == 1


# ---------------------------------------------------------------------------
# Tamper detection
# ---------------------------------------------------------------------------


def test_tampered_pack_sha256_fails() -> None:
    key = _new_key()
    commitment = sign_trace_pack_commitment(
        _PACK_SHA256_HEX,
        _CORPUS_ID,
        _WITNESS_IDENTITY,
        key,
        committed_at=_committed_at(-60),
    )
    different_hash = "b" * 64
    # Payload contains original hash, so signature is invalid for different_hash.
    with pytest.raises(WitnessCosigningError, match="threshold not met"):
        verify_witness_commitments(
            different_hash,
            _CORPUS_ID,
            [commitment],
            min_witnesses=1,
        )


def test_tampered_corpus_id_fails() -> None:
    key = _new_key()
    commitment = sign_trace_pack_commitment(
        _PACK_SHA256_HEX,
        _CORPUS_ID,
        _WITNESS_IDENTITY,
        key,
        committed_at=_committed_at(-60),
    )
    with pytest.raises(WitnessCosigningError, match="threshold not met"):
        verify_witness_commitments(
            _PACK_SHA256_HEX,
            "wrong-corpus",
            [commitment],
            min_witnesses=1,
        )


def test_wrong_key_fails() -> None:
    """A commitment signed with one key is invalid under a different key."""
    key_signer = _new_key()
    key_other = _new_key()
    commitment = sign_trace_pack_commitment(
        _PACK_SHA256_HEX,
        _CORPUS_ID,
        _WITNESS_IDENTITY,
        key_signer,
        committed_at=_committed_at(-60),
    )
    # Swap witness_public_key for the other key's pubkey.
    tampered = WitnessCommitment(
        committed_at=commitment.committed_at,
        witness_identity=commitment.witness_identity,
        witness_public_key="ed25519:" + key_other.public_key().public_bytes_raw().hex(),
        signature=commitment.signature,
    )
    with pytest.raises(WitnessCosigningError, match="threshold not met"):
        verify_witness_commitments(
            _PACK_SHA256_HEX,
            _CORPUS_ID,
            [tampered],
            min_witnesses=1,
        )


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_duplicate_identity_counted_once() -> None:
    key = _new_key()
    commitment = sign_trace_pack_commitment(
        _PACK_SHA256_HEX,
        _CORPUS_ID,
        _WITNESS_IDENTITY,
        key,
        committed_at=_committed_at(-60),
    )
    count = verify_witness_commitments(
        _PACK_SHA256_HEX,
        _CORPUS_ID,
        [commitment, commitment],  # duplicate
        min_witnesses=1,
    )
    assert count == 1


def test_two_witnesses_counted_separately() -> None:
    key_a = _new_key()
    key_b = _new_key()
    c_a = sign_trace_pack_commitment(
        _PACK_SHA256_HEX, _CORPUS_ID, "witness-a", key_a, committed_at=_committed_at(-60)
    )
    c_b = sign_trace_pack_commitment(
        _PACK_SHA256_HEX, _CORPUS_ID, "witness-b", key_b, committed_at=_committed_at(-60)
    )
    count = verify_witness_commitments(
        _PACK_SHA256_HEX,
        _CORPUS_ID,
        [c_a, c_b],
        min_witnesses=2,
    )
    assert count == 2


# ---------------------------------------------------------------------------
# Trusted-witness registry
# ---------------------------------------------------------------------------


def test_trusted_witness_accepted() -> None:
    key = _new_key()
    pub_hex = "ed25519:" + key.public_key().public_bytes_raw().hex()
    commitment = sign_trace_pack_commitment(
        _PACK_SHA256_HEX,
        _CORPUS_ID,
        _WITNESS_IDENTITY,
        key,
        committed_at=_committed_at(-60),
    )
    count = verify_witness_commitments(
        _PACK_SHA256_HEX,
        _CORPUS_ID,
        [commitment],
        trusted_witnesses=[(_WITNESS_IDENTITY, pub_hex)],
        min_witnesses=1,
    )
    assert count == 1


def test_untrusted_witness_skipped() -> None:
    """A valid signature from an unknown witness is ignored when registry is set."""
    key = _new_key()
    commitment = sign_trace_pack_commitment(
        _PACK_SHA256_HEX,
        _CORPUS_ID,
        _WITNESS_IDENTITY,
        key,
        committed_at=_committed_at(-60),
    )
    trusted_key = _new_key()
    trusted_pub_hex = "ed25519:" + trusted_key.public_key().public_bytes_raw().hex()
    with pytest.raises(WitnessCosigningError, match="threshold not met"):
        verify_witness_commitments(
            _PACK_SHA256_HEX,
            _CORPUS_ID,
            [commitment],
            trusted_witnesses=[("different-witness", trusted_pub_hex)],
            min_witnesses=1,
        )


def test_trusted_registry_invalid_pk_raises() -> None:
    with pytest.raises(WitnessCosigningError, match="must start with 'ed25519:'"):
        verify_witness_commitments(
            _PACK_SHA256_HEX,
            _CORPUS_ID,
            [],
            trusted_witnesses=[("w1", "not-a-key")],
            min_witnesses=0,
        )


# ---------------------------------------------------------------------------
# Evaluation-timestamp ordering
# ---------------------------------------------------------------------------


def test_committed_before_evaluation_passes() -> None:
    key = _new_key()
    committed_at = "2026-05-01T10:00:00+00:00"
    eval_at = "2026-05-01T12:00:00+00:00"  # after committed_at → OK
    commitment = sign_trace_pack_commitment(
        _PACK_SHA256_HEX, _CORPUS_ID, _WITNESS_IDENTITY, key, committed_at=committed_at
    )
    count = verify_witness_commitments(
        _PACK_SHA256_HEX,
        _CORPUS_ID,
        [commitment],
        min_witnesses=1,
        evaluation_timestamp=eval_at,
    )
    assert count == 1


def test_committed_after_evaluation_raises() -> None:
    key = _new_key()
    committed_at = "2026-05-01T14:00:00+00:00"  # AFTER evaluation
    eval_at = "2026-05-01T12:00:00+00:00"
    commitment = sign_trace_pack_commitment(
        _PACK_SHA256_HEX, _CORPUS_ID, _WITNESS_IDENTITY, key, committed_at=committed_at
    )
    with pytest.raises(WitnessCosigningError, match="after the earliest evaluation"):
        verify_witness_commitments(
            _PACK_SHA256_HEX,
            _CORPUS_ID,
            [commitment],
            min_witnesses=1,
            evaluation_timestamp=eval_at,
        )


def test_no_evaluation_timestamp_skips_ordering() -> None:
    key = _new_key()
    # Future committed_at is accepted when no evaluation_timestamp given.
    future = (
        datetime.datetime.now(tz=datetime.timezone.utc)
        + datetime.timedelta(days=365)
    ).isoformat()
    commitment = sign_trace_pack_commitment(
        _PACK_SHA256_HEX, _CORPUS_ID, _WITNESS_IDENTITY, key, committed_at=future
    )
    count = verify_witness_commitments(
        _PACK_SHA256_HEX,
        _CORPUS_ID,
        [commitment],
        min_witnesses=1,
        evaluation_timestamp=None,
    )
    assert count == 1


# ---------------------------------------------------------------------------
# Threshold enforcement
# ---------------------------------------------------------------------------


def test_min_witnesses_not_met_raises() -> None:
    key = _new_key()
    commitment = sign_trace_pack_commitment(
        _PACK_SHA256_HEX, _CORPUS_ID, _WITNESS_IDENTITY, key, committed_at=_committed_at(-60)
    )
    with pytest.raises(WitnessCosigningError, match="threshold not met"):
        verify_witness_commitments(
            _PACK_SHA256_HEX,
            _CORPUS_ID,
            [commitment],
            min_witnesses=2,
        )


def test_zero_min_witnesses_always_passes() -> None:
    count = verify_witness_commitments(
        _PACK_SHA256_HEX,
        _CORPUS_ID,
        [],
        min_witnesses=0,
    )
    assert count == 0


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------


def test_malformed_signature_hex_raises() -> None:
    commitment = WitnessCommitment(
        committed_at=_committed_at(-60),
        witness_identity=_WITNESS_IDENTITY,
        witness_public_key="ed25519:" + "a" * 64,
        signature="ed25519:not-valid-hex",
    )
    with pytest.raises(WitnessCosigningError):
        verify_witness_commitments(
            _PACK_SHA256_HEX,
            _CORPUS_ID,
            [commitment],
            min_witnesses=1,
        )


def test_bad_signature_scheme_raises() -> None:
    commitment = WitnessCommitment(
        committed_at=_committed_at(-60),
        witness_identity=_WITNESS_IDENTITY,
        witness_public_key="ed25519:" + "a" * 64,
        signature="rsa:" + "a" * 128,
    )
    with pytest.raises(WitnessCosigningError, match="unsupported signature scheme"):
        verify_witness_commitments(
            _PACK_SHA256_HEX,
            _CORPUS_ID,
            [commitment],
            min_witnesses=1,
        )


def test_bad_public_key_scheme_raises() -> None:
    commitment = WitnessCommitment(
        committed_at=_committed_at(-60),
        witness_identity=_WITNESS_IDENTITY,
        witness_public_key="rsa:not-valid",
        signature="ed25519:" + "a" * 128,
    )
    with pytest.raises(WitnessCosigningError, match="unsupported public_key scheme"):
        verify_witness_commitments(
            _PACK_SHA256_HEX,
            _CORPUS_ID,
            [commitment],
            min_witnesses=1,
        )


# ---------------------------------------------------------------------------
# Capability constant
# ---------------------------------------------------------------------------


def test_capability_constant() -> None:
    assert CAPABILITY_WITNESS_COSIGNING == "witness-cosigning"


# ===========================================================================
# Leaderboard integration tests
# ===========================================================================

import pytest

from stepback.bench.leaderboard import (
    LeaderboardEntry,
    build_leaderboard,
    generate_leaderboard_html,
    generate_leaderboard_json,
)

# Reuse the helper from test_bench_leaderboard where possible.
_FAKE_SHA256 = "a" * 64
_FAKE_SHA256_B = "b" * 64


def _make_bench_result(
    corpus_id: str = "test-corpus",
    timestamp_utc: str = "2026-05-01T12:00:00Z",
) -> dict:
    return {
        "schema_version": "1.0",
        "run_id": str(uuid.uuid4()),
        "timestamp_utc": timestamp_utc,
        "corpus_id": corpus_id,
        "trace_count": 10,
        "versions": {"stepback_version": "1.0.0"},
        "hardware": {"os": "Linux", "cpu_count": 4},
    }


def _valid_submission_raw(
    *,
    pack_sha256: str = _FAKE_SHA256,
    corpus_id: str = "test-corpus",
    git_dirty: bool = False,
    git_commit: str = "abc1234" + "0" * 33,
    witness_cosignatures: list | None = None,
    bench_timestamp: str = "2026-05-01T12:00:00Z",
) -> dict:
    d: dict = {
        "rules_version": "1.0",
        "submission_id": str(uuid.uuid4()),
        "hardware": {"os": "Linux", "cpu_count": 4},
        "code": {
            "stepback_version": "1.0.0",
            "git_commit": git_commit,
            "git_dirty": git_dirty,
        },
        "trace_pack": {
            "corpus_id": corpus_id,
            "trace_count": 10,
            "pack_sha256": pack_sha256,
            "verified": True,
        },
        "exact_commands": ["stepback bench replay-caching"],
        "validator_output": "12/12 PASSED",
        "bench_results": [_make_bench_result(corpus_id=corpus_id, timestamp_utc=bench_timestamp)],
        "audit": {
            "submitter_name": "Alice",
            "submitter_email": "alice@example.com",
            "submitter_organization": "Acme",
            "grants_source_access": True,
            "grants_trace_pack_access": True,
            "submission_date": "2026-05-01",
        },
    }
    if witness_cosignatures is not None:
        d["witness_cosignatures"] = witness_cosignatures
    return d


def _make_commitment(
    pack_sha256: str = _FAKE_SHA256,
    corpus_id: str = "test-corpus",
    committed_at: str = "2026-05-01T10:00:00+00:00",
) -> dict:
    key = Ed25519PrivateKey.generate()
    commitment = sign_trace_pack_commitment(
        pack_sha256, corpus_id, "trusted-witness", key, committed_at=committed_at
    )
    return commitment.to_dict()


# ---------------------------------------------------------------------------
# Without cosignatures → development
# ---------------------------------------------------------------------------


def test_leaderboard_no_cosignatures_is_development() -> None:
    raw = _valid_submission_raw(witness_cosignatures=None)
    lb = build_leaderboard([("test", raw)])
    assert len(lb.accepted) == 1
    entry = lb.accepted[0]
    assert entry.is_development is True
    assert entry.has_witness_cosignatures is False
    assert entry.witness_count == 0


def test_leaderboard_empty_cosignatures_is_development() -> None:
    raw = _valid_submission_raw(witness_cosignatures=[])
    lb = build_leaderboard([("test", raw)])
    assert lb.accepted[0].is_development is True


# ---------------------------------------------------------------------------
# With valid cosignature → public
# ---------------------------------------------------------------------------


def test_leaderboard_valid_cosignature_is_public() -> None:
    cosig = _make_commitment(
        pack_sha256=_FAKE_SHA256,
        corpus_id="test-corpus",
        committed_at="2026-05-01T10:00:00+00:00",
    )
    raw = _valid_submission_raw(
        witness_cosignatures=[cosig],
        bench_timestamp="2026-05-01T12:00:00Z",
    )
    lb = build_leaderboard([("test", raw)])
    assert len(lb.accepted) == 1
    entry = lb.accepted[0]
    assert entry.is_development is False
    assert entry.has_witness_cosignatures is True
    assert entry.witness_count == 1


def test_leaderboard_cosignature_after_eval_is_development() -> None:
    """Commitment timestamp after evaluation timestamp → cosig invalid."""
    cosig = _make_commitment(
        pack_sha256=_FAKE_SHA256,
        corpus_id="test-corpus",
        committed_at="2026-05-01T14:00:00+00:00",  # AFTER bench run
    )
    raw = _valid_submission_raw(
        witness_cosignatures=[cosig],
        bench_timestamp="2026-05-01T12:00:00Z",
    )
    lb = build_leaderboard([("test", raw)])
    entry = lb.accepted[0]
    assert entry.is_development is True
    assert entry.has_witness_cosignatures is False


# ---------------------------------------------------------------------------
# Git dirty + valid cosig → still development
# ---------------------------------------------------------------------------


def test_leaderboard_git_dirty_with_cosig_is_development() -> None:
    cosig = _make_commitment()
    raw = _valid_submission_raw(
        git_dirty=True,
        witness_cosignatures=[cosig],
    )
    lb = build_leaderboard([("test", raw)])
    assert lb.accepted[0].is_development is True


# ---------------------------------------------------------------------------
# Invalid cosig (wrong signature) → development
# ---------------------------------------------------------------------------


def test_leaderboard_invalid_cosig_is_development() -> None:
    key = Ed25519PrivateKey.generate()
    # Commitment signed for different corpus_id.
    cosig = sign_trace_pack_commitment(
        _FAKE_SHA256,
        "wrong-corpus-id",  # mismatch
        "witness-one",
        key,
        committed_at="2026-05-01T10:00:00+00:00",
    ).to_dict()
    raw = _valid_submission_raw(
        witness_cosignatures=[cosig],
        bench_timestamp="2026-05-01T12:00:00Z",
    )
    lb = build_leaderboard([("test", raw)])
    entry = lb.accepted[0]
    assert entry.is_development is True
    assert entry.has_witness_cosignatures is False


# ---------------------------------------------------------------------------
# to_json includes cosigning fields
# ---------------------------------------------------------------------------


def test_entry_to_json_includes_cosigning_fields() -> None:
    cosig = _make_commitment()
    raw = _valid_submission_raw(witness_cosignatures=[cosig])
    lb = build_leaderboard([("test", raw)])
    j = lb.accepted[0].to_json()
    assert "has_witness_cosignatures" in j
    assert "witness_count" in j


def test_generate_leaderboard_json_round_trip() -> None:
    cosig = _make_commitment()
    raw = _valid_submission_raw(witness_cosignatures=[cosig])
    lb = build_leaderboard([("test", raw)])
    j = generate_leaderboard_json(lb)
    # Must be JSON serialisable.
    _ = json.dumps(j)


# ---------------------------------------------------------------------------
# HTML contains witness column
# ---------------------------------------------------------------------------


def test_generate_leaderboard_html_contains_witnesses_column() -> None:
    cosig = _make_commitment()
    raw = _valid_submission_raw(witness_cosignatures=[cosig])
    lb = build_leaderboard([("test", raw)])
    html = generate_leaderboard_html(lb)
    assert "Witnesses" in html


def test_generate_leaderboard_html_escapes_identity() -> None:
    """Malicious witness identity in submission does not escape HTML context."""
    key = Ed25519PrivateKey.generate()
    evil_identity = "<script>alert(1)</script>"
    # Build a commitment and inject a tampered identity into the raw dict.
    commitment = sign_trace_pack_commitment(
        _FAKE_SHA256,
        "test-corpus",
        "legit-witness",
        key,
        committed_at="2026-05-01T10:00:00+00:00",
    )
    # Tampering the identity field after signing → invalid sig → dev entry
    cosig_dict = commitment.to_dict()
    cosig_dict["witness_identity"] = evil_identity
    raw = _valid_submission_raw(
        witness_cosignatures=[cosig_dict],
        bench_timestamp="2026-05-01T12:00:00Z",
    )
    lb = build_leaderboard([("test", raw)])
    html = generate_leaderboard_html(lb)
    # The raw script tag should not appear unescaped.
    assert "<script>alert(1)</script>" not in html
