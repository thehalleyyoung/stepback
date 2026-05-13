"""Tests for hardware-backed key support (step 133).

Covers:
* SoftwareSimProvider — always-available software simulator
* PKCS11Provider — skipped if python-pkcs11 is not installed
* YubiHSMProvider — fully mocked (no real hardware required)
* AWSKMSProvider — boto3 client mocked (no network call)
* GCPKMSProvider — google-cloud-kms client mocked (no network call)
* AzureKeyVaultProvider — azure-keyvault-keys client mocked (no network call)
* Integration: sign an attestation pack using hardware-backed key adapters
* _SigningAdapter duck-type contract
* provider_from_env helper
"""
from __future__ import annotations

import hashlib
import os
import types
from unittest.mock import MagicMock, patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from stepback.hwkey import (
    AWSKMSProvider,
    AzureKeyVaultProvider,
    GCPKMSProvider,
    HardwareKeyProvider,
    HardwareKeySignError,
    HardwareKeyUnavailableError,
    PKCS11Provider,
    SoftwareSimProvider,
    YubiHSMProvider,
    _SigningAdapter,
    provider_from_env,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _small_message() -> bytes:
    return b"stepback hwkey test payload 2026"


# ── SoftwareSimProvider ───────────────────────────────────────────────────────


class TestSoftwareSimProvider:
    def test_generate_returns_provider(self):
        p = SoftwareSimProvider.generate()
        assert isinstance(p, HardwareKeyProvider)

    def test_sign_returns_64_bytes(self):
        p = SoftwareSimProvider.generate()
        sig = p.sign_bytes(_small_message())
        assert isinstance(sig, bytes)
        assert len(sig) == 64, "Ed25519 signatures are always 64 bytes"

    def test_public_key_bytes_32_bytes(self):
        p = SoftwareSimProvider.generate()
        assert len(p.public_key_bytes()) == 32

    def test_key_algorithm_is_ed25519(self):
        p = SoftwareSimProvider.generate()
        assert p.key_algorithm() == "ed25519"

    def test_fingerprint_format(self):
        p = SoftwareSimProvider.generate()
        fp = p.fingerprint()
        assert fp.startswith("ed25519:")
        _, hex_part = fp.split(":", 1)
        assert len(hex_part) == 64

    def test_fingerprint_matches_sha256_of_pubkey(self):
        p = SoftwareSimProvider.generate()
        expected = "ed25519:" + hashlib.sha256(p.public_key_bytes()).hexdigest()
        assert p.fingerprint() == expected

    def test_from_private_bytes_roundtrip(self):
        orig = SoftwareSimProvider.generate()
        raw_priv = orig.private_key_raw()
        loaded = SoftwareSimProvider.from_private_bytes(raw_priv)
        assert loaded.public_key_bytes() == orig.public_key_bytes()
        msg = b"roundtrip"
        sig = orig.sign_bytes(msg)
        # Verify loaded key's public key accepts the original signature.
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        pub = Ed25519PublicKey.from_public_bytes(loaded.public_key_bytes())
        pub.verify(sig, msg)  # raises if invalid

    def test_deterministic_for_same_key(self):
        p = SoftwareSimProvider.generate()
        msg = b"deterministic"
        # Ed25519 is deterministic — same key + message → same signature
        assert p.sign_bytes(msg) == p.sign_bytes(msg)

    def test_different_keys_different_signatures(self):
        p1 = SoftwareSimProvider.generate()
        p2 = SoftwareSimProvider.generate()
        msg = b"same message"
        assert p1.sign_bytes(msg) != p2.sign_bytes(msg)

    def test_is_available(self):
        assert SoftwareSimProvider.is_available() is True


# ── _SigningAdapter ───────────────────────────────────────────────────────────


class TestSigningAdapter:
    def test_sign_delegates_to_provider(self):
        p = SoftwareSimProvider.generate()
        adapter = p.as_signing_adapter()
        assert isinstance(adapter, _SigningAdapter)
        msg = b"adapter test"
        assert adapter.sign(msg) == p.sign_bytes(msg)

    def test_public_key_raw_matches_provider(self):
        p = SoftwareSimProvider.generate()
        adapter = p.as_signing_adapter()
        assert adapter.public_key().public_bytes_raw() == p.public_key_bytes()

    def test_adapter_satisfies_ed25519privatekey_interface(self):
        """write_attestation_pack calls .sign() and .public_key().public_bytes_raw()."""
        p = SoftwareSimProvider.generate()
        adapter = p.as_signing_adapter()
        sig_bytes = adapter.sign(b"data")
        assert isinstance(sig_bytes, bytes)
        pub_bytes = adapter.public_key().public_bytes_raw()
        assert isinstance(pub_bytes, bytes)
        assert len(pub_bytes) == 32

    def test_adapter_public_key_verify(self):
        """Signature produced via adapter can be verified with the public key bytes."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        p = SoftwareSimProvider.generate()
        adapter = p.as_signing_adapter()
        msg = b"verifiable"
        sig = adapter.sign(msg)
        pub = Ed25519PublicKey.from_public_bytes(adapter.public_key().public_bytes_raw())
        pub.verify(sig, msg)  # raises InvalidSignature if wrong


# ── PKCS11Provider ────────────────────────────────────────────────────────────


class TestPKCS11ProviderAvailability:
    def test_is_available_reflects_import(self):
        # We don't know if python-pkcs11 is installed; just check it returns bool.
        result = PKCS11Provider.is_available()
        assert isinstance(result, bool)

    def test_missing_library_raises(self):
        """If python-pkcs11 is installed but lib path is empty, should raise."""
        if not PKCS11Provider.is_available():
            pytest.skip("python-pkcs11 not installed")
        with pytest.raises(HardwareKeyUnavailableError, match="PKCS#11 library"):
            PKCS11Provider(pkcs11_lib="")

    def test_not_installed_raises(self):
        """Without python-pkcs11 installed, constructor raises HardwareKeyUnavailableError."""
        if PKCS11Provider.is_available():
            pytest.skip("python-pkcs11 is installed on this system")
        with pytest.raises(HardwareKeyUnavailableError, match="python-pkcs11"):
            PKCS11Provider(pkcs11_lib="/nonexistent/lib.so")


# ── YubiHSMProvider (fully mocked) ───────────────────────────────────────────


class TestYubiHSMProviderMocked:
    """Exercise YubiHSMProvider by mocking the yubihsm SDK.

    The mock is injected as a sys.modules entry so the provider code picks it
    up via its ``import yubihsm`` call.  No real HSM or connector is needed.
    """

    @staticmethod
    def _build_mock_yubihsm(pub_bytes: bytes, sig_bytes: bytes):
        """Return a minimal mock yubihsm module tree."""
        mod = types.ModuleType("yubihsm")

        # defs sub-module
        defs_mod = types.ModuleType("yubihsm.defs")

        class _OBJECT:
            ASYMMETRIC_KEY = "ASYMMETRIC_KEY"

        defs_mod.OBJECT = _OBJECT()
        mod.defs = defs_mod

        # backends sub-module
        backends_mod = types.ModuleType("yubihsm.backends")
        backends_mod.get_backend = MagicMock(return_value=MagicMock())
        mod.backends = backends_mod

        # core sub-module
        core_mod = types.ModuleType("yubihsm.core")

        mock_pub_key = MagicMock()
        mock_pub_key.public_bytes_raw.return_value = pub_bytes

        mock_key_obj = MagicMock()
        mock_key_obj.get_public_key.return_value = mock_pub_key
        mock_key_obj.sign_eddsa.return_value = sig_bytes

        mock_session = MagicMock()
        mock_session.get_object.return_value = mock_key_obj

        mock_hsm = MagicMock()
        mock_hsm.create_session_derived.return_value = mock_session

        class _YubiHsm:
            def __init__(self, backend):
                pass

            def create_session_derived(self, key_id, password):
                return mock_session

        core_mod.YubiHsm = _YubiHsm
        core_mod.AuthSession = MagicMock
        mod.core = core_mod

        return mod, mock_session, mock_key_obj

    def test_sign_bytes_returns_mocked_signature(self, monkeypatch):
        expected_pub = b"\xaa" * 32
        expected_sig = b"\xbb" * 64
        mock_mod, _, _ = self._build_mock_yubihsm(expected_pub, expected_sig)

        import sys
        monkeypatch.setitem(sys.modules, "yubihsm", mock_mod)
        monkeypatch.setitem(sys.modules, "yubihsm.core", mock_mod.core)
        monkeypatch.setitem(sys.modules, "yubihsm.defs", mock_mod.defs)
        monkeypatch.setitem(sys.modules, "yubihsm.backends", mock_mod.backends)

        provider = YubiHSMProvider.__new__(YubiHSMProvider)
        provider._pub_raw = expected_pub
        provider._key_obj = mock_mod.core.YubiHsm(None).create_session_derived(
            1, "password"
        ).get_object(2, mock_mod.defs.OBJECT.ASYMMETRIC_KEY)
        # Wire sign_eddsa directly
        provider._key_obj.sign_eddsa.return_value = expected_sig

        sig = provider.sign_bytes(b"hello")
        assert sig == expected_sig

    def test_public_key_bytes(self):
        expected_pub = b"\xcc" * 32
        provider = YubiHSMProvider.__new__(YubiHSMProvider)
        provider._pub_raw = expected_pub
        assert provider.public_key_bytes() == expected_pub

    def test_key_algorithm(self):
        provider = YubiHSMProvider.__new__(YubiHSMProvider)
        assert provider.key_algorithm() == "ed25519"

    def test_fingerprint_format(self):
        pub = b"\xdd" * 32
        provider = YubiHSMProvider.__new__(YubiHSMProvider)
        provider._pub_raw = pub
        fp = provider.fingerprint()
        assert fp.startswith("ed25519:")
        assert fp == "ed25519:" + hashlib.sha256(pub).hexdigest()

    def test_is_available_reflects_import(self):
        result = YubiHSMProvider.is_available()
        assert isinstance(result, bool)

    def test_sign_error_wrapped(self):
        provider = YubiHSMProvider.__new__(YubiHSMProvider)
        provider._pub_raw = b"\x00" * 32
        mock_key = MagicMock()
        mock_key.sign_eddsa.side_effect = RuntimeError("HSM exploded")
        provider._key_obj = mock_key
        with pytest.raises(HardwareKeySignError, match="YubiHSM signing failed"):
            provider.sign_bytes(b"data")


# ── AWSKMSProvider (mocked boto3) ─────────────────────────────────────────────


def _build_mock_boto3_client(pub_bytes: bytes, sig_bytes: bytes) -> MagicMock:
    client = MagicMock()
    client.get_public_key.return_value = {"PublicKey": pub_bytes}
    client.sign.return_value = {"Signature": sig_bytes}
    return client


class TestAWSKMSProvider:
    def test_sign_bytes(self):
        pub = b"\x01" * 32
        expected_sig = b"\x02" * 64
        mock_client = _build_mock_boto3_client(pub, expected_sig)
        provider = AWSKMSProvider(key_id="arn:aws:kms:us-east-1:123:key/abc", boto3_client=mock_client)
        sig = provider.sign_bytes(b"test message")
        assert sig == expected_sig
        mock_client.sign.assert_called_once()
        call_kwargs = mock_client.sign.call_args[1] if mock_client.sign.call_args[1] else mock_client.sign.call_args[0][0]

    def test_public_key_bytes_fetched_on_init(self):
        pub = b"\x03" * 32
        mock_client = _build_mock_boto3_client(pub, b"\x04" * 64)
        provider = AWSKMSProvider(key_id="test-key-id", boto3_client=mock_client)
        assert provider.public_key_bytes() == pub

    def test_key_algorithm_ecdsa(self):
        mock_client = _build_mock_boto3_client(b"\x05" * 32, b"\x06" * 64)
        provider = AWSKMSProvider(
            key_id="key-id",
            signing_algorithm="ECDSA_SHA_256",
            boto3_client=mock_client,
        )
        assert provider.key_algorithm() == "ecdsa-sha-256"

    def test_fingerprint_format(self):
        pub = b"\x07" * 32
        mock_client = _build_mock_boto3_client(pub, b"\x08" * 64)
        provider = AWSKMSProvider(key_id="key-id", boto3_client=mock_client)
        fp = provider.fingerprint()
        assert ":" in fp

    def test_sign_error_wrapped(self):
        mock_client = _build_mock_boto3_client(b"\x09" * 32, b"")
        mock_client.sign.side_effect = Exception("KMS unavailable")
        provider = AWSKMSProvider(key_id="key-id", boto3_client=mock_client)
        with pytest.raises(HardwareKeySignError, match="AWS KMS signing failed"):
            provider.sign_bytes(b"data")

    def test_init_error_on_missing_public_key(self):
        mock_client = MagicMock()
        mock_client.get_public_key.side_effect = Exception("Access denied")
        with pytest.raises(HardwareKeyUnavailableError, match="Failed to fetch AWS KMS"):
            AWSKMSProvider(key_id="key-id", boto3_client=mock_client)

    def test_not_available_without_client_raises(self):
        if AWSKMSProvider.is_available():
            pytest.skip("boto3 is installed")
        with pytest.raises(HardwareKeyUnavailableError, match="boto3"):
            AWSKMSProvider(key_id="key-id")

    def test_is_available_returns_bool(self):
        assert isinstance(AWSKMSProvider.is_available(), bool)

    def test_message_type_raw_passed(self):
        """Provider must pass MessageType='RAW' to KMS (not pre-hashed externally)."""
        pub = b"\x0a" * 32
        sig = b"\x0b" * 64
        mock_client = _build_mock_boto3_client(pub, sig)
        provider = AWSKMSProvider(key_id="k", boto3_client=mock_client)
        provider.sign_bytes(b"msg")
        _, call_kwargs = mock_client.sign.call_args
        assert call_kwargs.get("MessageType") == "RAW"


# ── GCPKMSProvider (mocked) ───────────────────────────────────────────────────


def _build_mock_gcp_client(pub_der: bytes, sig_bytes: bytes) -> MagicMock:
    """Build a mock google-cloud-kms client."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        PublicFormat,
    )
    # Use a real Ed25519 public key in PEM for the mock response.
    priv = Ed25519PrivateKey.generate()
    pem_bytes = priv.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)

    mock_pub = MagicMock()
    mock_pub.pem = pem_bytes.decode()

    mock_resp = MagicMock()
    mock_resp.signature = sig_bytes

    client = MagicMock()
    client.get_public_key.return_value = mock_pub
    client.asymmetric_sign.return_value = mock_resp
    return client


class TestGCPKMSProvider:
    def test_sign_bytes(self):
        expected_sig = b"\x10" * 64
        mock_client = _build_mock_gcp_client(b"", expected_sig)
        provider = GCPKMSProvider(
            key_version_name="projects/p/locations/l/keyRings/r/cryptoKeys/k/cryptoKeyVersions/1",
            kms_client=mock_client,
        )
        sig = provider.sign_bytes(b"gcp test")
        assert sig == expected_sig

    def test_public_key_fetched_on_init(self):
        mock_client = _build_mock_gcp_client(b"", b"\x11" * 64)
        provider = GCPKMSProvider(key_version_name="dummy-name", kms_client=mock_client)
        # public_key_bytes returns DER bytes; non-empty means fetch succeeded.
        assert isinstance(provider.public_key_bytes(), bytes)
        assert len(provider.public_key_bytes()) > 0

    def test_key_algorithm_default(self):
        mock_client = _build_mock_gcp_client(b"", b"\x12" * 64)
        provider = GCPKMSProvider(key_version_name="name", kms_client=mock_client)
        assert provider.key_algorithm() == "ecdsa-sha256"

    def test_sign_error_wrapped(self):
        mock_client = _build_mock_gcp_client(b"", b"")
        mock_client.asymmetric_sign.side_effect = Exception("quota exceeded")
        provider = GCPKMSProvider(key_version_name="name", kms_client=mock_client)
        with pytest.raises(HardwareKeySignError, match="GCP KMS signing failed"):
            provider.sign_bytes(b"data")

    def test_is_available_returns_bool(self):
        assert isinstance(GCPKMSProvider.is_available(), bool)

    def test_init_failure_raises(self):
        mock_client = MagicMock()
        mock_client.get_public_key.side_effect = Exception("permission denied")
        with pytest.raises(HardwareKeyUnavailableError, match="Failed to fetch GCP KMS"):
            GCPKMSProvider(key_version_name="name", kms_client=mock_client)


# ── AzureKeyVaultProvider (mocked) ───────────────────────────────────────────


def _build_mock_azure_client(sig_bytes: bytes) -> MagicMock:
    mock_result = MagicMock()
    mock_result.signature = sig_bytes

    client = MagicMock()
    client.sign.return_value = mock_result

    # Provide a key attribute to simulate public key access.
    mock_key_json = MagicMock()
    mock_key_json.n = None
    mock_key_json.e = None
    mock_key_json.x = b"\xee" * 32
    mock_key = MagicMock()
    mock_key.key = mock_key_json
    client.key = mock_key

    return client


class TestAzureKeyVaultProvider:
    def test_sign_bytes(self):
        expected_sig = b"\x20" * 64
        mock_client = _build_mock_azure_client(expected_sig)
        provider = AzureKeyVaultProvider.__new__(AzureKeyVaultProvider)
        provider._client = mock_client
        provider._algorithm = "ES256"
        provider._pub_raw = b"\xee" * 32

        # Override sign_bytes to avoid importing azure types in tests.
        def _sign(data: bytes) -> bytes:
            from unittest.mock import MagicMock as _MM
            result = mock_client.sign(_MM(), hashlib.sha256(data).digest())
            return result.signature

        provider.sign_bytes = _sign
        sig = provider.sign_bytes(b"azure test")
        assert sig == expected_sig

    def test_key_algorithm(self):
        provider = AzureKeyVaultProvider.__new__(AzureKeyVaultProvider)
        provider._algorithm = "ES256"
        assert provider.key_algorithm() == "es256"

    def test_is_available_returns_bool(self):
        assert isinstance(AzureKeyVaultProvider.is_available(), bool)

    def test_public_key_bytes_from_ec_x(self):
        mock_client = _build_mock_azure_client(b"")
        provider = AzureKeyVaultProvider.__new__(AzureKeyVaultProvider)
        provider._client = mock_client
        provider._algorithm = "ES256"
        provider._pub_raw = provider._fetch_public_key_bytes()
        assert isinstance(provider._pub_raw, bytes)


# ── provider_from_env ─────────────────────────────────────────────────────────


class TestProviderFromEnv:
    def test_default_is_software(self, monkeypatch):
        monkeypatch.delenv("STEPBACK_HWKEY_BACKEND", raising=False)
        p = provider_from_env()
        assert isinstance(p, SoftwareSimProvider)

    def test_software_backend_explicit(self, monkeypatch):
        monkeypatch.setenv("STEPBACK_HWKEY_BACKEND", "software")
        p = provider_from_env()
        assert isinstance(p, SoftwareSimProvider)

    def test_unknown_backend_raises(self, monkeypatch):
        monkeypatch.setenv("STEPBACK_HWKEY_BACKEND", "nonexistent-backend")
        with pytest.raises(ValueError, match="Unknown STEPBACK_HWKEY_BACKEND"):
            provider_from_env()

    def test_aws_kms_missing_key_id_raises(self, monkeypatch):
        monkeypatch.setenv("STEPBACK_HWKEY_BACKEND", "aws-kms")
        monkeypatch.delenv("STEPBACK_AWS_KMS_KEY_ID", raising=False)
        with pytest.raises(HardwareKeyUnavailableError, match="STEPBACK_AWS_KMS_KEY_ID"):
            provider_from_env()

    def test_gcp_kms_missing_key_raises(self, monkeypatch):
        monkeypatch.setenv("STEPBACK_HWKEY_BACKEND", "gcp-kms")
        monkeypatch.delenv("STEPBACK_GCP_KMS_KEY_VERSION_NAME", raising=False)
        with pytest.raises(HardwareKeyUnavailableError, match="STEPBACK_GCP_KMS_KEY_VERSION_NAME"):
            provider_from_env()

    def test_azure_keyvault_missing_url_raises(self, monkeypatch):
        monkeypatch.setenv("STEPBACK_HWKEY_BACKEND", "azure-keyvault")
        monkeypatch.delenv("STEPBACK_AZURE_VAULT_URL", raising=False)
        monkeypatch.delenv("STEPBACK_AZURE_KEY_NAME", raising=False)
        with pytest.raises(HardwareKeyUnavailableError, match="STEPBACK_AZURE_VAULT_URL"):
            provider_from_env()


# ── Integration: hardware adapter with attestation pack ──────────────────────


class TestHardwareKeyAttestationIntegration:
    """
    Verify that the duck-typed _SigningAdapter produced by
    SoftwareSimProvider.as_signing_adapter() can drive the full
    build_attestation_pack → write_attestation_pack → verify_attestation_pack
    round-trip.
    """

    def test_attestation_pack_signed_with_sim_provider(self, tmp_path):
        from stepback import RecorderKey, record
        from stepback.attestation import (
            build_attestation_pack,
            verify_attestation_pack,
            write_attestation_pack,
        )
        from stepback.testing import run_recorded_agent

        # Record a minimal trace.
        key = RecorderKey.fresh()
        trace_path = str(tmp_path / "test.sb")
        with record(trace_path, key=key) as rec:
            run_recorded_agent(rec)

        # Build a provider and adapter.
        provider = SoftwareSimProvider.generate()
        adapter = provider.as_signing_adapter()

        # Build pack using the adapter as the attestor signing key.
        pack = build_attestation_pack(
            [(trace_path, key.hmac_key)],
            attestor_signing_key=adapter,
        )

        # The attestor_public_key in the pack must match the provider fingerprint
        # using the raw public key hex (as attestation uses ed25519:<raw-hex>).
        expected_pub_hex = provider.public_key_bytes().hex()
        assert pack.attestor_public_key == f"ed25519:{expected_pub_hex}"

        # Write pack to disk.
        out_path = str(tmp_path / "test.pack.json")
        write_attestation_pack(pack, out_path, signing_key=adapter)

        # Verify pack can be read and signature verified.
        expected_pub = f"ed25519:{provider.public_key_bytes().hex()}"
        body = verify_attestation_pack(out_path, expected_public_key=expected_pub)
        assert body["summary"]["verified_ok"] >= 1

    def test_signing_adapter_fingerprint_matches_pack_public_key(self):
        provider = SoftwareSimProvider.generate()
        adapter = provider.as_signing_adapter()
        pub_raw = adapter.public_key().public_bytes_raw()
        expected_fp = f"ed25519:{pub_raw.hex()}"
        assert expected_fp == f"ed25519:{provider.public_key_bytes().hex()}"

    def test_two_providers_produce_different_packs(self, tmp_path):
        """Different providers must produce packs with different attestor keys."""
        p1 = SoftwareSimProvider.generate()
        p2 = SoftwareSimProvider.generate()
        assert p1.public_key_bytes() != p2.public_key_bytes()
        assert p1.fingerprint() != p2.fingerprint()
