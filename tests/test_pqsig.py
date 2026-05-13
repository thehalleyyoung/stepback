"""Tests for the post-quantum signature experiment module (step 129).

All tests are offline-only.  Tests that require a specific PQ algorithm are
skipped when the system OpenSSL does not support that algorithm, so CI on
older OpenSSL builds is not broken.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from stepback import RecorderKey, record
from stepback.attestation import (
    AttestationVerificationError,
    build_attestation_pack,
    read_attestation_pack,
    verify_attestation_pack,
    write_attestation_pack,
)
from stepback.pqsig import (
    CAPABILITY_PQ_ATTESTATION,
    PQ_ALGORITHMS,
    PQKeyPair,
    PQSignatureInvalid,
    PQUnavailableError,
    generate_keypair,
    is_algorithm_available,
    pq_attestation_payload,
    public_key_fingerprint,
    sign_bytes,
    verify_bytes,
)
from stepback.testing import run_recorded_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip_if_unavailable(algorithm: str) -> None:
    if not is_algorithm_available(algorithm):
        pytest.skip(f"{algorithm} not available in this OpenSSL build")


def _record_trace(tmp_path: Path, name: str) -> tuple[str, RecorderKey]:
    key = RecorderKey.fresh()
    path = str(tmp_path / f"{name}.sb")
    with record(path, key=key) as rec:
        run_recorded_agent(rec)
    return path, key


# ---------------------------------------------------------------------------
# pqsig module — standalone tests
# ---------------------------------------------------------------------------

class TestAlgorithmAvailability:
    def test_known_unavailable_returns_false(self) -> None:
        assert is_algorithm_available("not-a-real-algorithm") is False

    def test_ml_dsa_65_probe(self) -> None:
        result = is_algorithm_available("ml-dsa-65")
        assert isinstance(result, bool)

    def test_slh_dsa_probe(self) -> None:
        result = is_algorithm_available("slh-dsa-shake-256f")
        assert isinstance(result, bool)

    def test_case_insensitive(self) -> None:
        # Algorithm names are normalised to lowercase.
        result_lower = is_algorithm_available("ml-dsa-65")
        result_upper = is_algorithm_available("ML-DSA-65")
        assert result_lower == result_upper

    def test_pq_algorithms_dict_keys_are_lowercase(self) -> None:
        for key in PQ_ALGORITHMS:
            assert key == key.lower(), f"PQ_ALGORITHMS key {key!r} must be lowercase"


class TestUnknownAlgorithmRaises:
    """These always run regardless of OpenSSL version."""

    def test_generate_unknown_raises(self) -> None:
        with pytest.raises(PQUnavailableError, match="unknown PQ algorithm"):
            generate_keypair("not-real-1234")

    def test_is_available_unknown_returns_false(self) -> None:
        assert is_algorithm_available("xyzdsa-99") is False


class TestMLDSA65:
    @pytest.fixture(autouse=True)
    def _skip(self) -> None:
        _skip_if_unavailable("ml-dsa-65")

    def test_generate_keypair_returns_pem(self) -> None:
        kp = generate_keypair("ml-dsa-65")
        assert kp.algorithm == "ml-dsa-65"
        assert b"PRIVATE KEY" in kp.private_key_pem
        assert b"PUBLIC KEY" in kp.public_key_pem

    def test_sign_verify_roundtrip(self) -> None:
        kp = generate_keypair("ml-dsa-65")
        msg = b"hello post-quantum world"
        sig = sign_bytes(msg, kp)
        assert isinstance(sig, bytes)
        assert len(sig) > 0
        # Should not raise.
        verify_bytes(msg, sig, kp.public_key_pem)

    def test_tampered_message_fails(self) -> None:
        kp = generate_keypair("ml-dsa-65")
        msg = b"original"
        sig = sign_bytes(msg, kp)
        with pytest.raises(PQSignatureInvalid):
            verify_bytes(b"tampered", sig, kp.public_key_pem)

    def test_tampered_signature_fails(self) -> None:
        kp = generate_keypair("ml-dsa-65")
        msg = b"original"
        sig = sign_bytes(msg, kp)
        bad_sig = bytes([sig[0] ^ 0xFF]) + sig[1:]
        with pytest.raises(PQSignatureInvalid):
            verify_bytes(msg, bad_sig, kp.public_key_pem)

    def test_wrong_key_fails(self) -> None:
        kp1 = generate_keypair("ml-dsa-65")
        kp2 = generate_keypair("ml-dsa-65")
        msg = b"message"
        sig = sign_bytes(msg, kp1)
        with pytest.raises(PQSignatureInvalid):
            verify_bytes(msg, sig, kp2.public_key_pem)

    def test_fingerprint_format(self) -> None:
        kp = generate_keypair("ml-dsa-65")
        fp = public_key_fingerprint(kp)
        assert fp.startswith("ml-dsa-65:")
        alg, hexpart = fp.split(":", 1)
        assert len(hexpart) == 64, "fingerprint should be sha256 hex (64 chars)"

    def test_fingerprints_differ_for_different_keys(self) -> None:
        kp1 = generate_keypair("ml-dsa-65")
        kp2 = generate_keypair("ml-dsa-65")
        assert public_key_fingerprint(kp1) != public_key_fingerprint(kp2)

    def test_capability_constant(self) -> None:
        assert CAPABILITY_PQ_ATTESTATION == "pqsig"


class TestMLDSA44:
    @pytest.fixture(autouse=True)
    def _skip(self) -> None:
        _skip_if_unavailable("ml-dsa-44")

    def test_sign_verify_roundtrip(self) -> None:
        kp = generate_keypair("ml-dsa-44")
        msg = b"ml-dsa-44 test"
        sig = sign_bytes(msg, kp)
        verify_bytes(msg, sig, kp.public_key_pem)  # must not raise

    def test_fingerprint_prefix(self) -> None:
        kp = generate_keypair("ml-dsa-44")
        fp = public_key_fingerprint(kp)
        assert fp.startswith("ml-dsa-44:")


class TestMLDSA87:
    @pytest.fixture(autouse=True)
    def _skip(self) -> None:
        _skip_if_unavailable("ml-dsa-87")

    def test_sign_verify_roundtrip(self) -> None:
        kp = generate_keypair("ml-dsa-87")
        msg = b"ml-dsa-87 test"
        sig = sign_bytes(msg, kp)
        verify_bytes(msg, sig, kp.public_key_pem)


class TestSLHDSA:
    @pytest.fixture(autouse=True)
    def _skip(self) -> None:
        _skip_if_unavailable("slh-dsa-shake-256f")

    def test_sign_verify_roundtrip(self) -> None:
        kp = generate_keypair("slh-dsa-shake-256f")
        msg = b"sphincs+ test message"
        sig = sign_bytes(msg, kp)
        verify_bytes(msg, sig, kp.public_key_pem)

    def test_fingerprint_prefix(self) -> None:
        kp = generate_keypair("slh-dsa-shake-256f")
        fp = public_key_fingerprint(kp)
        assert fp.startswith("slh-dsa-shake-256f:")


class TestPQAttestationPayload:
    def test_deterministic(self) -> None:
        p1 = pq_attestation_payload("ml-dsa-65", "sha256:abc123")
        p2 = pq_attestation_payload("ml-dsa-65", "sha256:abc123")
        assert p1 == p2

    def test_different_for_different_alg(self) -> None:
        p1 = pq_attestation_payload("ml-dsa-65", "sha256:abc123")
        p2 = pq_attestation_payload("ml-dsa-87", "sha256:abc123")
        assert p1 != p2

    def test_different_for_different_body_hash(self) -> None:
        p1 = pq_attestation_payload("ml-dsa-65", "sha256:abc")
        p2 = pq_attestation_payload("ml-dsa-65", "sha256:xyz")
        assert p1 != p2

    def test_contains_type_discriminator(self) -> None:
        payload = json.loads(pq_attestation_payload("ml-dsa-65", "sha256:x"))
        assert payload["type"] == "stepback.attestation.pqsig.v1"
        assert payload["algorithm"] == "ml-dsa-65"
        assert payload["body_hash"] == "sha256:x"


# ---------------------------------------------------------------------------
# Attestation pack integration
# ---------------------------------------------------------------------------

@pytest.fixture
def trace_path(tmp_path: Path) -> tuple[str, RecorderKey]:
    return _record_trace(tmp_path, "pq_test")


class TestAttestationPackPQCoSignature:
    @pytest.fixture(autouse=True)
    def _skip(self) -> None:
        _skip_if_unavailable("ml-dsa-65")

    def _build_and_write(
        self,
        trace: tuple[str, RecorderKey],
        tmp_path: Path,
        *,
        pq_keypair: PQKeyPair,
        attestor_key: Ed25519PrivateKey,
    ) -> tuple[str, str]:
        """Build, write, and return (pack_path, body_hash)."""
        trace_path_str, rec_key = trace
        pack = build_attestation_pack(
            [(trace_path_str, rec_key.hmac_key)],
            attestor_signing_key=attestor_key,
            pq_keypair=pq_keypair,
        )
        pack_path = str(tmp_path / "pq_pack.pack")
        body_hash = write_attestation_pack(
            pack, pack_path,
            signing_key=attestor_key,
            pq_signing_key=pq_keypair,
        )
        return pack_path, body_hash

    def test_pq_cosigned_pack_verifies(
        self,
        trace_path: tuple[str, RecorderKey],
        tmp_path: Path,
    ) -> None:
        kp = generate_keypair("ml-dsa-65")
        attestor = Ed25519PrivateKey.generate()
        pack_path, _ = self._build_and_write(trace_path, tmp_path, pq_keypair=kp, attestor_key=attestor)
        data = verify_attestation_pack(pack_path)
        assert "pq_attestor_public_key" in data
        assert data["pq_attestor_public_key"].startswith("ml-dsa-65:")
        assert "pq_signature" in data

    def test_pq_public_key_is_body_bound(
        self,
        trace_path: tuple[str, RecorderKey],
        tmp_path: Path,
    ) -> None:
        """pq_attestor_public_key is inside the body → tampering it breaks body_hash."""
        kp = generate_keypair("ml-dsa-65")
        attestor = Ed25519PrivateKey.generate()
        pack_path, _ = self._build_and_write(trace_path, tmp_path, pq_keypair=kp, attestor_key=attestor)

        with open(pack_path) as f:
            data = json.load(f)
        # Replace the embedded PQ public key.
        kp2 = generate_keypair("ml-dsa-65")
        data["pq_attestor_public_key"] = public_key_fingerprint(kp2)
        with open(pack_path, "w") as f:
            json.dump(data, f, sort_keys=True)
        with pytest.raises(AttestationVerificationError, match="body_hash mismatch"):
            verify_attestation_pack(pack_path)

    def test_missing_pq_signature_fails(
        self,
        trace_path: tuple[str, RecorderKey],
        tmp_path: Path,
    ) -> None:
        """Strip pq_signature → verifier must reject (fail closed)."""
        kp = generate_keypair("ml-dsa-65")
        attestor = Ed25519PrivateKey.generate()
        pack_path, _ = self._build_and_write(trace_path, tmp_path, pq_keypair=kp, attestor_key=attestor)

        with open(pack_path) as f:
            data = json.load(f)
        del data["pq_signature"]
        with open(pack_path, "w") as f:
            json.dump(data, f, sort_keys=True)
        with pytest.raises(AttestationVerificationError, match="pq_signature.*missing"):
            verify_attestation_pack(pack_path)

    def test_tampered_pq_signature_fails(
        self,
        trace_path: tuple[str, RecorderKey],
        tmp_path: Path,
    ) -> None:
        kp = generate_keypair("ml-dsa-65")
        attestor = Ed25519PrivateKey.generate()
        pack_path, _ = self._build_and_write(trace_path, tmp_path, pq_keypair=kp, attestor_key=attestor)

        with open(pack_path) as f:
            data = json.load(f)
        sig_hex = data["pq_signature"]["signature"]
        bad_hex = format(int(sig_hex[:8], 16) ^ 0xDEADBEEF, "08x") + sig_hex[8:]
        data["pq_signature"]["signature"] = bad_hex
        with open(pack_path, "w") as f:
            json.dump(data, f, sort_keys=True)
        with pytest.raises(AttestationVerificationError, match="PQ co-signature invalid"):
            verify_attestation_pack(pack_path)

    def test_replaced_pq_algorithm_fails(
        self,
        trace_path: tuple[str, RecorderKey],
        tmp_path: Path,
    ) -> None:
        """Change pq_signature.algorithm → algorithm mismatch error."""
        kp = generate_keypair("ml-dsa-65")
        attestor = Ed25519PrivateKey.generate()
        pack_path, _ = self._build_and_write(trace_path, tmp_path, pq_keypair=kp, attestor_key=attestor)

        with open(pack_path) as f:
            data = json.load(f)
        data["pq_signature"]["algorithm"] = "ml-dsa-87"
        with open(pack_path, "w") as f:
            json.dump(data, f, sort_keys=True)
        with pytest.raises(AttestationVerificationError, match="algorithm"):
            verify_attestation_pack(pack_path)

    def test_verify_pq_false_skips_pq_check(
        self,
        trace_path: tuple[str, RecorderKey],
        tmp_path: Path,
    ) -> None:
        """verify_pq=False allows skipping PQ verification."""
        kp = generate_keypair("ml-dsa-65")
        attestor = Ed25519PrivateKey.generate()
        pack_path, _ = self._build_and_write(trace_path, tmp_path, pq_keypair=kp, attestor_key=attestor)

        with open(pack_path) as f:
            data = json.load(f)
        # Corrupt the PQ signature.
        orig_sig = data["pq_signature"]["signature"]
        data["pq_signature"]["signature"] = "00" * (len(orig_sig) // 2)
        with open(pack_path, "w") as f:
            json.dump(data, f, sort_keys=True)
        # With verify_pq=False, the corrupted PQ sig should be ignored.
        result = verify_attestation_pack(pack_path, verify_pq=False)
        assert result is not None

    def test_no_pq_pack_works_unchanged(
        self,
        trace_path: tuple[str, RecorderKey],
        tmp_path: Path,
    ) -> None:
        """Packs without PQ co-signature still verify normally."""
        trace_path_str, rec_key = trace_path
        attestor = Ed25519PrivateKey.generate()
        pack = build_attestation_pack(
            [(trace_path_str, rec_key.hmac_key)],
            attestor_signing_key=attestor,
        )
        pack_path = str(tmp_path / "no_pq_pack.pack")
        write_attestation_pack(pack, pack_path, signing_key=attestor)
        data = verify_attestation_pack(pack_path)
        assert "pq_attestor_public_key" not in data or data.get("pq_attestor_public_key") is None

    def test_pq_key_mismatch_raises_at_write(
        self,
        trace_path: tuple[str, RecorderKey],
        tmp_path: Path,
    ) -> None:
        """Supplying a different PQ key than the one in the pack body raises ValueError."""
        kp1 = generate_keypair("ml-dsa-65")
        kp2 = generate_keypair("ml-dsa-65")
        attestor = Ed25519PrivateKey.generate()
        trace_path_str, rec_key = trace_path
        pack = build_attestation_pack(
            [(trace_path_str, rec_key.hmac_key)],
            attestor_signing_key=attestor,
            pq_keypair=kp1,
        )
        pack_path = str(tmp_path / "mismatch_pack.pack")
        with pytest.raises(ValueError, match="pq_signing_key fingerprint"):
            write_attestation_pack(pack, pack_path, signing_key=attestor, pq_signing_key=kp2)

    def test_pq_key_without_body_field_raises_at_write(
        self,
        trace_path: tuple[str, RecorderKey],
        tmp_path: Path,
    ) -> None:
        """Supplying pq_signing_key without corresponding body field raises ValueError."""
        kp = generate_keypair("ml-dsa-65")
        attestor = Ed25519PrivateKey.generate()
        trace_path_str, rec_key = trace_path
        # Build WITHOUT pq_keypair — so pq_attestor_public_key is not in body.
        pack = build_attestation_pack(
            [(trace_path_str, rec_key.hmac_key)],
            attestor_signing_key=attestor,
        )
        pack_path = str(tmp_path / "nofield_pack.pack")
        with pytest.raises(ValueError, match="pq_attestor_public_key is not set"):
            write_attestation_pack(pack, pack_path, signing_key=attestor, pq_signing_key=kp)

    def test_body_field_without_pq_key_raises_at_write(
        self,
        trace_path: tuple[str, RecorderKey],
        tmp_path: Path,
    ) -> None:
        """pq_attestor_public_key in body but no pq_signing_key raises ValueError."""
        kp = generate_keypair("ml-dsa-65")
        attestor = Ed25519PrivateKey.generate()
        trace_path_str, rec_key = trace_path
        pack = build_attestation_pack(
            [(trace_path_str, rec_key.hmac_key)],
            attestor_signing_key=attestor,
            pq_keypair=kp,
        )
        pack_path = str(tmp_path / "nopqkey_pack.pack")
        with pytest.raises(ValueError, match="no pq_signing_key was provided"):
            write_attestation_pack(pack, pack_path, signing_key=attestor)


class TestAttestationPackPQWithSLH:
    """Same integration tests but with SLH-DSA-SHAKE-256f."""

    @pytest.fixture(autouse=True)
    def _skip(self) -> None:
        _skip_if_unavailable("slh-dsa-shake-256f")

    def test_slh_dsa_pack_verifies(
        self,
        trace_path: tuple[str, RecorderKey],
        tmp_path: Path,
    ) -> None:
        kp = generate_keypair("slh-dsa-shake-256f")
        attestor = Ed25519PrivateKey.generate()
        trace_path_str, rec_key = trace_path
        pack = build_attestation_pack(
            [(trace_path_str, rec_key.hmac_key)],
            attestor_signing_key=attestor,
            pq_keypair=kp,
        )
        pack_path = str(tmp_path / "slh_pack.pack")
        write_attestation_pack(pack, pack_path, signing_key=attestor, pq_signing_key=kp)
        data = verify_attestation_pack(pack_path)
        assert data["pq_attestor_public_key"].startswith("slh-dsa-shake-256f:")
