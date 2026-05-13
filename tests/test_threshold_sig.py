"""Tests for M-of-N threshold signing (step 130).

Covers:
* Unit tests of threshold_sig module (sign/verify, domain separation)
* Integration with attestation packs (build → write → verify round-trip)
* Security: downgrade attack, unknown witnesses, duplicate witnesses,
  tampered signatures, cross-pack replay prevention
* Error paths: missing envelope, insufficient witnesses, bad hex
"""
from __future__ import annotations

import json
import os

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from stepback import RecorderKey, record
from stepback.attestation import (
    AttestationVerificationError,
    build_attestation_pack,
    verify_attestation_pack,
    write_attestation_pack,
)
from stepback.threshold_sig import (
    CAPABILITY_THRESHOLD_SIGNING,
    ThresholdPolicy,
    ThresholdSignatureError,
    WitnessSignature,
    WitnessSpec,
    collect_witness_signatures,
    make_threshold_policy,
    sign_as_witness,
    threshold_witness_payload,
    verify_threshold_signatures,
)
from stepback.testing import run_recorded_agent


# ── helpers ─────────────────────────────────────────────────────────────────


def _fresh_keys(n: int) -> dict[str, Ed25519PrivateKey]:
    """Generate n fresh Ed25519 keys keyed by identity strings."""
    return {f"witness:{i}": Ed25519PrivateKey.generate() for i in range(n)}


def _record_trace(tmp_path, name: str):
    key = RecorderKey.fresh()
    path = str(tmp_path / f"{name}.sb")
    with record(path, key=key) as rec:
        run_recorded_agent(rec)
    return path, key


# ── WitnessSpec / ThresholdPolicy construction ──────────────────────────────


def test_threshold_policy_rejects_m_zero():
    with pytest.raises(ValueError, match="m_required must be"):
        ThresholdPolicy(m_required=0, witnesses=[])


def test_threshold_policy_rejects_m_greater_than_n():
    w = WitnessSpec("a", "ed25519:" + "aa" * 32)
    with pytest.raises(ValueError, match="m_required"):
        ThresholdPolicy(m_required=2, witnesses=[w])


def test_threshold_policy_rejects_duplicate_identities():
    key = "ed25519:" + "aa" * 32
    key2 = "ed25519:" + "bb" * 32
    w1 = WitnessSpec("dup", key)
    w2 = WitnessSpec("dup", key2)
    with pytest.raises(ValueError, match="unique identities"):
        ThresholdPolicy(m_required=1, witnesses=[w1, w2])


def test_threshold_policy_rejects_duplicate_public_keys():
    key = "ed25519:" + "aa" * 32
    w1 = WitnessSpec("a", key)
    w2 = WitnessSpec("b", key)
    with pytest.raises(ValueError, match="unique public keys"):
        ThresholdPolicy(m_required=1, witnesses=[w1, w2])


def test_threshold_policy_round_trip_dict():
    keys = _fresh_keys(3)
    policy = make_threshold_policy(keys, m_required=2)
    d = policy.to_dict()
    policy2 = ThresholdPolicy.from_dict(d)
    assert policy2.m_required == 2
    assert len(policy2.witnesses) == 3
    assert {w.identity for w in policy2.witnesses} == set(keys)


# ── domain-separated payload ─────────────────────────────────────────────────


def test_threshold_witness_payload_is_deterministic():
    body_hash = "sha256:" + "ab" * 32
    p1 = threshold_witness_payload(body_hash, "secops", "ed25519:" + "cc" * 32)
    p2 = threshold_witness_payload(body_hash, "secops", "ed25519:" + "cc" * 32)
    assert p1 == p2


def test_threshold_witness_payload_differs_by_identity():
    body_hash = "sha256:" + "ab" * 32
    pub = "ed25519:" + "cc" * 32
    p1 = threshold_witness_payload(body_hash, "secops", pub)
    p2 = threshold_witness_payload(body_hash, "legal", pub)
    assert p1 != p2


def test_threshold_witness_payload_differs_by_body_hash():
    pub = "ed25519:" + "cc" * 32
    p1 = threshold_witness_payload("sha256:" + "aa" * 32, "w", pub)
    p2 = threshold_witness_payload("sha256:" + "bb" * 32, "w", pub)
    assert p1 != p2


def test_threshold_witness_payload_not_raw_body_hash():
    """The domain-separated payload must differ from the plain body_hash bytes."""
    body_hash = "sha256:" + "ab" * 32
    pub = "ed25519:" + "cc" * 32
    payload = threshold_witness_payload(body_hash, "w", pub)
    assert payload != body_hash.encode("ascii")


# ── sign_as_witness + verify_threshold_signatures ───────────────────────────


def test_2of3_threshold_sign_and_verify():
    keys = _fresh_keys(3)
    policy = make_threshold_policy(keys, m_required=2)
    body_hash = "sha256:" + "de" * 32

    # Sign with all 3.
    sigs = collect_witness_signatures(body_hash, policy, keys)
    valid = verify_threshold_signatures(body_hash, policy, sigs)
    assert valid == 3


def test_exactly_m_signatures_sufficient():
    keys = _fresh_keys(3)
    policy = make_threshold_policy(keys, m_required=2)
    body_hash = "sha256:" + "de" * 32

    # Sign with only first 2 witnesses.
    partial_keys = dict(list(keys.items())[:2])
    sigs = collect_witness_signatures(body_hash, policy, partial_keys)
    assert len(sigs) == 2
    valid = verify_threshold_signatures(body_hash, policy, sigs)
    assert valid == 2


def test_fewer_than_m_raises():
    keys = _fresh_keys(3)
    policy = make_threshold_policy(keys, m_required=2)
    body_hash = "sha256:" + "de" * 32

    # Manually sign with only 1 witness (bypassing collect_witness_signatures
    # which would itself raise — we want to test the verifier path).
    identity = list(keys)[0]
    spec = policy.witness_by_identity(identity)
    one_sig = sign_as_witness(body_hash, spec, keys[identity])
    with pytest.raises(ThresholdSignatureError, match="threshold not met"):
        verify_threshold_signatures(body_hash, policy, [one_sig])


def test_collect_raises_if_too_few_authorized_keys():
    keys = _fresh_keys(3)
    policy = make_threshold_policy(keys, m_required=3)
    body_hash = "sha256:" + "de" * 32

    # Only pass 2 keys — must fail before even attempting signatures.
    two_keys = dict(list(keys.items())[:2])
    with pytest.raises(ThresholdSignatureError, match="m_required"):
        collect_witness_signatures(body_hash, policy, two_keys)


def test_tampered_signature_does_not_count():
    keys = _fresh_keys(3)
    policy = make_threshold_policy(keys, m_required=2)
    body_hash = "sha256:" + "de" * 32

    sigs = collect_witness_signatures(body_hash, policy, keys)
    # Corrupt the first signature by flipping a byte.
    raw = bytes.fromhex(sigs[0].signature.removeprefix("ed25519:"))
    corrupted = bytes([raw[0] ^ 0xFF]) + raw[1:]
    sigs[0] = WitnessSignature(
        identity=sigs[0].identity,
        signature="ed25519:" + corrupted.hex(),
    )
    # 2 remaining signatures still satisfy 2-of-3.
    valid = verify_threshold_signatures(body_hash, policy, sigs)
    assert valid == 2


def test_all_signatures_tampered_fails():
    keys = _fresh_keys(2)
    policy = make_threshold_policy(keys, m_required=2)
    body_hash = "sha256:" + "de" * 32

    sigs = collect_witness_signatures(body_hash, policy, keys)
    # Corrupt all signatures.
    bad_sigs = []
    for ws in sigs:
        raw = bytes.fromhex(ws.signature.removeprefix("ed25519:"))
        corrupted = bytes([raw[0] ^ 0xFF]) + raw[1:]
        bad_sigs.append(WitnessSignature(ws.identity, "ed25519:" + corrupted.hex()))
    with pytest.raises(ThresholdSignatureError, match="threshold not met"):
        verify_threshold_signatures(body_hash, policy, bad_sigs)


def test_unknown_witnesses_are_ignored():
    keys = _fresh_keys(2)
    policy = make_threshold_policy(keys, m_required=1)
    body_hash = "sha256:" + "de" * 32

    # Create a signature from a non-authorized key.
    unknown_key = Ed25519PrivateKey.generate()
    unknown_spec = WitnessSpec(
        identity="unknown",
        public_key="ed25519:" + unknown_key.public_key().public_bytes_raw().hex(),
    )
    unknown_sig = sign_as_witness(body_hash, unknown_spec, unknown_key)

    # Only the unknown sig — should not count.
    with pytest.raises(ThresholdSignatureError, match="threshold not met"):
        verify_threshold_signatures(body_hash, policy, [unknown_sig])


def test_duplicate_witness_counted_once():
    keys = _fresh_keys(3)
    policy = make_threshold_policy(keys, m_required=2)
    body_hash = "sha256:" + "de" * 32

    sigs = collect_witness_signatures(body_hash, policy, keys)
    # Duplicate the first signature.
    doubled = [sigs[0], sigs[0], sigs[1]]
    valid = verify_threshold_signatures(body_hash, policy, doubled)
    # Still 2: sigs[0] counted once, sigs[1] counted once.
    assert valid == 2


def test_cross_pack_replay_fails():
    """A witness signature from pack A must not verify against pack B's body_hash."""
    keys = _fresh_keys(2)
    policy = make_threshold_policy(keys, m_required=2)
    body_hash_a = "sha256:" + "aa" * 32
    body_hash_b = "sha256:" + "bb" * 32

    sigs_a = collect_witness_signatures(body_hash_a, policy, keys)
    with pytest.raises(ThresholdSignatureError, match="threshold not met"):
        verify_threshold_signatures(body_hash_b, policy, sigs_a)


# ── capability constant ──────────────────────────────────────────────────────


def test_capability_constant():
    assert CAPABILITY_THRESHOLD_SIGNING == "threshold-signing"


# ── attestation pack integration ─────────────────────────────────────────────


def test_threshold_pack_write_and_verify_round_trip(tmp_path):
    p, k = _record_trace(tmp_path, "trace")
    attestor = Ed25519PrivateKey.generate()

    witness_keys = _fresh_keys(3)
    policy = make_threshold_policy(witness_keys, m_required=2)

    pack = build_attestation_pack(
        [(p, k.hmac_key)],
        attestor_signing_key=attestor,
        threshold_policy=policy,
    )
    assert pack.threshold_policy is not None
    assert "threshold_policy" in pack.body_dict()

    out = str(tmp_path / "thresh.pack")
    body_hash = write_attestation_pack(
        pack, out,
        signing_key=attestor,
        threshold_signing_keys=witness_keys,
    )
    assert body_hash.startswith("sha256:")

    data = verify_attestation_pack(out)
    assert "threshold_policy" in data
    assert data["threshold_policy"]["m_required"] == 2
    assert "threshold_signatures" in data
    assert len(data["threshold_signatures"]) == 3


def test_threshold_pack_with_minimum_m_witnesses(tmp_path):
    """Exactly m=2 out of n=3 keys still passes verification."""
    p, k = _record_trace(tmp_path, "trace")
    attestor = Ed25519PrivateKey.generate()

    witness_keys = _fresh_keys(3)
    policy = make_threshold_policy(witness_keys, m_required=2)

    pack = build_attestation_pack(
        [(p, k.hmac_key)],
        attestor_signing_key=attestor,
        threshold_policy=policy,
    )
    out = str(tmp_path / "thresh_partial.pack")
    # Supply only 2 of 3 keys.
    partial_keys = dict(list(witness_keys.items())[:2])
    write_attestation_pack(
        pack, out,
        signing_key=attestor,
        threshold_signing_keys=partial_keys,
    )
    # Should verify with 2 valid sigs.
    data = verify_attestation_pack(out)
    assert len(data["threshold_signatures"]) == 2


def test_threshold_pack_missing_envelope_fails_verification(tmp_path):
    """Stripping threshold_signatures from a policy-declaring pack must fail."""
    p, k = _record_trace(tmp_path, "trace")
    attestor = Ed25519PrivateKey.generate()

    witness_keys = _fresh_keys(2)
    policy = make_threshold_policy(witness_keys, m_required=2)

    pack = build_attestation_pack(
        [(p, k.hmac_key)],
        attestor_signing_key=attestor,
        threshold_policy=policy,
    )
    out = str(tmp_path / "stripped.pack")
    write_attestation_pack(
        pack, out,
        signing_key=attestor,
        threshold_signing_keys=witness_keys,
    )
    # Strip the envelope field.
    with open(out) as f:
        raw = json.load(f)
    del raw["threshold_signatures"]
    with open(out, "w") as f:
        json.dump(raw, f, sort_keys=True, indent=2)

    with pytest.raises(AttestationVerificationError, match="threshold_signatures"):
        verify_attestation_pack(out)


def test_threshold_pack_tampered_witness_signature_fails(tmp_path):
    p, k = _record_trace(tmp_path, "trace")
    attestor = Ed25519PrivateKey.generate()

    witness_keys = _fresh_keys(2)
    policy = make_threshold_policy(witness_keys, m_required=2)

    pack = build_attestation_pack(
        [(p, k.hmac_key)],
        attestor_signing_key=attestor,
        threshold_policy=policy,
    )
    out = str(tmp_path / "tampered.pack")
    write_attestation_pack(
        pack, out,
        signing_key=attestor,
        threshold_signing_keys=witness_keys,
    )
    with open(out) as f:
        raw = json.load(f)
    # Corrupt every witness signature.
    for entry in raw["threshold_signatures"]:
        sig_bytes = bytes.fromhex(entry["signature"].removeprefix("ed25519:"))
        corrupted = bytes([sig_bytes[0] ^ 0xFF]) + sig_bytes[1:]
        entry["signature"] = "ed25519:" + corrupted.hex()
    with open(out, "w") as f:
        json.dump(raw, f, sort_keys=True, indent=2)

    with pytest.raises(AttestationVerificationError, match="threshold"):
        verify_attestation_pack(out)


def test_verify_threshold_false_skips_threshold_check(tmp_path):
    """verify_threshold=False bypasses threshold verification (escape hatch)."""
    p, k = _record_trace(tmp_path, "trace")
    attestor = Ed25519PrivateKey.generate()

    witness_keys = _fresh_keys(2)
    policy = make_threshold_policy(witness_keys, m_required=2)

    pack = build_attestation_pack(
        [(p, k.hmac_key)],
        attestor_signing_key=attestor,
        threshold_policy=policy,
    )
    out = str(tmp_path / "skip_thresh.pack")
    write_attestation_pack(
        pack, out,
        signing_key=attestor,
        threshold_signing_keys=witness_keys,
    )
    with open(out) as f:
        raw = json.load(f)
    # Corrupt all sigs — would normally fail.
    for entry in raw["threshold_signatures"]:
        entry["signature"] = "ed25519:" + "ff" * 64
    with open(out, "w") as f:
        json.dump(raw, f, sort_keys=True, indent=2)

    # With verify_threshold=False, verification succeeds despite bad sigs.
    data = verify_attestation_pack(out, verify_threshold=False)
    assert "threshold_policy" in data


def test_threshold_policy_missing_keys_raises_on_write(tmp_path):
    """write_attestation_pack raises ValueError if policy set but no keys given."""
    p, k = _record_trace(tmp_path, "trace")
    attestor = Ed25519PrivateKey.generate()

    witness_keys = _fresh_keys(2)
    policy = make_threshold_policy(witness_keys, m_required=1)

    pack = build_attestation_pack(
        [(p, k.hmac_key)],
        attestor_signing_key=attestor,
        threshold_policy=policy,
    )
    with pytest.raises(ValueError, match="threshold_signing_keys"):
        write_attestation_pack(pack, str(tmp_path / "x.pack"), signing_key=attestor)


def test_threshold_keys_without_policy_raises_on_write(tmp_path):
    """write_attestation_pack raises ValueError if keys given but no policy set."""
    p, k = _record_trace(tmp_path, "trace")
    attestor = Ed25519PrivateKey.generate()

    pack = build_attestation_pack(
        [(p, k.hmac_key)],
        attestor_signing_key=attestor,
        # No threshold_policy.
    )
    witness_keys = _fresh_keys(2)
    with pytest.raises(ValueError, match="threshold_policy"):
        write_attestation_pack(
            pack, str(tmp_path / "x.pack"),
            signing_key=attestor,
            threshold_signing_keys=witness_keys,
        )


def test_threshold_body_hash_includes_policy(tmp_path):
    """A pack with threshold_policy has a different body_hash than one without."""
    p, k = _record_trace(tmp_path, "trace")
    attestor = Ed25519PrivateKey.generate()

    pack_no_thresh = build_attestation_pack(
        [(p, k.hmac_key)], attestor_signing_key=attestor
    )
    witness_keys = _fresh_keys(2)
    policy = make_threshold_policy(witness_keys, m_required=2)
    pack_with_thresh = build_attestation_pack(
        [(p, k.hmac_key)],
        attestor_signing_key=attestor,
        threshold_policy=policy,
    )
    # body_hash computed from body_dict() — use canonical_json to simulate.
    from stepback.canonical import canonical_json, sha256_hex
    h1 = sha256_hex(canonical_json(pack_no_thresh.body_dict()))
    h2 = sha256_hex(canonical_json(pack_with_thresh.body_dict()))
    assert h1 != h2


def test_threshold_pack_normal_pack_has_no_threshold_field(tmp_path):
    """Packs without a threshold policy must not have a threshold_signatures field."""
    p, k = _record_trace(tmp_path, "trace")
    attestor = Ed25519PrivateKey.generate()
    pack = build_attestation_pack([(p, k.hmac_key)], attestor_signing_key=attestor)
    out = str(tmp_path / "normal.pack")
    write_attestation_pack(pack, out, signing_key=attestor)

    with open(out) as f:
        raw = json.load(f)
    assert "threshold_policy" not in raw
    assert "threshold_signatures" not in raw

    # Normal verification still works.
    verify_attestation_pack(out)


def test_threshold_1of1_round_trip(tmp_path):
    """Edge case: 1-of-1 is valid and should work like a standard co-signature."""
    p, k = _record_trace(tmp_path, "trace")
    attestor = Ed25519PrivateKey.generate()

    witness_keys = {"sole-witness": Ed25519PrivateKey.generate()}
    policy = make_threshold_policy(witness_keys, m_required=1)

    pack = build_attestation_pack(
        [(p, k.hmac_key)],
        attestor_signing_key=attestor,
        threshold_policy=policy,
    )
    out = str(tmp_path / "1of1.pack")
    write_attestation_pack(
        pack, out, signing_key=attestor, threshold_signing_keys=witness_keys
    )
    data = verify_attestation_pack(out)
    assert data["threshold_policy"]["m_required"] == 1


def test_threshold_make_threshold_policy_correct_keys(tmp_path):
    """make_threshold_policy extracts correct public key from private key."""
    priv = Ed25519PrivateKey.generate()
    policy = make_threshold_policy({"w": priv}, m_required=1)
    expected_pub = "ed25519:" + priv.public_key().public_bytes_raw().hex()
    assert policy.witnesses[0].public_key == expected_pub
