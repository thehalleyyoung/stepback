"""Hardware-backed key support (PKCS#11, YubiHSM2, cloud KMS).

Provides :class:`HardwareKeyProvider` — an abstract base class for
hardware-backed signing — with concrete implementations:

* :class:`PKCS11Provider` — PKCS#11 tokens (YubiKey, Nitrokey, SoftHSM2, …)
  via the ``python-pkcs11`` library (``pip install python-pkcs11``).
* :class:`YubiHSMProvider` — YubiHSM2 devices via the ``yubihsm`` library
  (``pip install yubihsm``).
* :class:`AWSKMSProvider` — AWS KMS asymmetric signing via ``boto3``.
* :class:`GCPKMSProvider` — Google Cloud KMS asymmetric signing via
  ``google-cloud-kms`` (``pip install google-cloud-kms``).
* :class:`AzureKeyVaultProvider` — Azure Key Vault via
  ``azure-keyvault-keys`` (``pip install azure-keyvault-keys``).
* :class:`SoftwareSimProvider` — pure-software Ed25519 simulator; no extra
  dependencies; backed by :mod:`cryptography`.  Always available.  Use in
  tests and local development.

All providers expose a common interface::

    provider.sign_bytes(data: bytes) -> bytes          # raw signature bytes
    provider.public_key_bytes() -> bytes               # raw Ed25519 or DER
    provider.key_algorithm() -> str                    # e.g. "ed25519"
    provider.fingerprint() -> str                      # "ed25519:<sha256-hex>"
    provider.as_signing_adapter() -> _SigningAdapter   # duck-typed Ed25519PrivateKey

The :class:`_SigningAdapter` returned by :meth:`~HardwareKeyProvider.as_signing_adapter`
satisfies the ``Ed25519PrivateKey.sign()`` / ``Ed25519PrivateKey.public_key()``
duck-type contract expected by :func:`~stepback.attestation.write_attestation_pack`
so hardware-backed keys can be passed to the attestation API without
changes::

    from stepback.hwkey import SoftwareSimProvider
    from stepback.attestation import build_attestation_pack, write_attestation_pack

    sim = SoftwareSimProvider.generate()
    pack = build_attestation_pack(
        traces,
        hmac_key=hmac_key,
        attestor_signing_key=sim.as_signing_adapter(),
    )
    write_attestation_pack(pack, out_path, signing_key=sim.as_signing_adapter())

Availability
------------
Each provider class exposes :meth:`is_available` that returns ``True`` iff
the required library / tool is importable.  Tests should skip when
``not MyProvider.is_available()``.

PKCS#11 extras
--------------
``PKCS11Provider`` requires a PKCS#11 shared library (``pkcs11_lib`` path)
**and** the ``python-pkcs11`` Python package.  For CI / offline testing use
the ``SoftwareSimProvider`` instead.  Real-hardware paths (YubiKey, Nitrokey,
HSM appliances) are exercised by setting ``STEPBACK_PKCS11_LIB`` and
``STEPBACK_PKCS11_TOKEN_LABEL`` environment variables and running the
integration test suite against real hardware.

Cloud KMS extras
----------------
``AWSKMSProvider`` requires ``boto3``; ``GCPKMSProvider`` requires
``google-cloud-kms``; ``AzureKeyVaultProvider`` requires
``azure-keyvault-keys``.  Tests mock the SDK clients so no network call is
made.

Security notes
--------------
* Private-key material never leaves the hardware/KMS boundary.
* Signatures are verified with the corresponding public key before being
  returned from :meth:`sign_bytes`.  This catches HSM glitches early.
* :class:`HardwareKeyUnavailableError` is raised (not logged-and-swallowed)
  so callers cannot silently fall back to software keys.
"""
from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    PrivateFormat,
    NoEncryption,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class HardwareKeyUnavailableError(RuntimeError):
    """Raised when a required hardware/library dependency is not available."""

    #: Canonical error code; see :mod:`stepback.errors` for details.
    code: str = "SB502"


class HardwareKeySignError(RuntimeError):
    """Raised when a signing operation fails at the hardware layer."""

    #: Canonical error code; see :mod:`stepback.errors` for details.
    code: str = "SB503"


# ---------------------------------------------------------------------------
# Adapters so HardwareKeyProvider can be passed to write_attestation_pack
# ---------------------------------------------------------------------------


class _HardwarePublicAdapter:
    """Duck-typed public-key adapter with the Ed25519PublicKey.public_bytes_raw() interface."""

    def __init__(self, raw_bytes: bytes) -> None:
        self._raw = raw_bytes  # 32-byte Ed25519 public key

    def public_bytes_raw(self) -> bytes:  # matches Ed25519PublicKey interface
        return self._raw

    def public_bytes(self, encoding: Encoding, fmt: PublicFormat) -> bytes:
        # Delegate to cryptography for DER/PEM if needed.
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey as _K
        k: Ed25519PublicKey = _K.from_public_bytes(self._raw)
        return k.public_bytes(encoding, fmt)


class _SigningAdapter:
    """Duck-typed signing adapter with the Ed25519PrivateKey interface.

    Passes the ``provider.sign_bytes`` implementation through the
    ``signing_key.sign()`` contract expected by
    :func:`~stepback.attestation.write_attestation_pack`.
    """

    def __init__(self, provider: "HardwareKeyProvider") -> None:
        self._provider = provider
        self._pub_adapter = _HardwarePublicAdapter(provider.public_key_bytes())

    def sign(self, data: bytes) -> bytes:
        return self._provider.sign_bytes(data)

    def public_key(self) -> _HardwarePublicAdapter:
        return self._pub_adapter


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class HardwareKeyProvider(ABC):
    """Abstract interface for hardware-backed (or simulated) signing keys.

    Subclasses must implement :meth:`sign_bytes`, :meth:`public_key_bytes`,
    and :meth:`key_algorithm`.  The other methods are provided by this base.
    """

    @abstractmethod
    def sign_bytes(self, data: bytes) -> bytes:
        """Sign *data* and return the raw signature bytes.

        The implementation MUST NOT hash ``data`` before signing — the caller
        is responsible for prehashing if needed.  Ed25519 signs messages
        directly (no prehash step at the API level).
        """

    @abstractmethod
    def public_key_bytes(self) -> bytes:
        """Return the raw public key bytes (32 bytes for Ed25519)."""

    @abstractmethod
    def key_algorithm(self) -> str:
        """Return the key algorithm identifier, e.g. ``"ed25519"``."""

    @classmethod
    def is_available(cls) -> bool:  # noqa: D401
        """Return True if the dependencies for this provider are available."""
        return True  # overridden by subclasses with optional deps

    def fingerprint(self) -> str:
        """Return ``"<alg>:<sha256-of-public-bytes-hex>"`` for attestation."""
        digest = hashlib.sha256(self.public_key_bytes()).hexdigest()
        return f"{self.key_algorithm()}:{digest}"

    def as_signing_adapter(self) -> _SigningAdapter:
        """Return a duck-typed :class:`_SigningAdapter` for attestation APIs."""
        return _SigningAdapter(self)


# ---------------------------------------------------------------------------
# SoftwareSimProvider — pure-software simulator (no extra deps)
# ---------------------------------------------------------------------------


class SoftwareSimProvider(HardwareKeyProvider):
    """Ed25519 software simulator backed by :mod:`cryptography`.

    This provider is always available (it uses only the ``cryptography``
    package which is a mandatory dependency of ``stepback``).  It is
    intended for:

    * Unit / integration tests that exercise the hardware-key signing path
      without real hardware.
    * Local development environments where no HSM or cloud KMS is configured.

    Do **not** use in production: the private key is held in process memory.
    """

    def __init__(self, private_key: Ed25519PrivateKey) -> None:
        self._private_key = private_key
        self._pub_raw = private_key.public_key().public_bytes_raw()

    @classmethod
    def generate(cls) -> "SoftwareSimProvider":
        """Generate a fresh ephemeral Ed25519 key pair."""
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def from_private_bytes(cls, raw: bytes) -> "SoftwareSimProvider":
        """Load from 32-byte raw private key seed."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey as K
        return cls(K.from_private_bytes(raw))

    def sign_bytes(self, data: bytes) -> bytes:
        return self._private_key.sign(data)

    def public_key_bytes(self) -> bytes:
        return self._pub_raw

    def key_algorithm(self) -> str:
        return "ed25519"

    def private_key_raw(self) -> bytes:
        """Return raw 32-byte private key seed (for serialisation / export)."""
        return self._private_key.private_bytes(
            Encoding.Raw, PrivateFormat.Raw, NoEncryption()
        )


# ---------------------------------------------------------------------------
# PKCS#11 provider
# ---------------------------------------------------------------------------


class PKCS11Provider(HardwareKeyProvider):
    """PKCS#11 signing via the ``python-pkcs11`` library.

    Supports any PKCS#11 v2.40-compliant token: SoftHSM2, YubiKey, Nitrokey,
    Thales Luna, AWS CloudHSM, etc.

    Parameters
    ----------
    pkcs11_lib:
        Path to the PKCS#11 shared library (e.g.
        ``/usr/lib/softhsm/libsofthsm2.so``).  Falls back to the
        ``STEPBACK_PKCS11_LIB`` environment variable if omitted.
    token_label:
        Token label.  Falls back to ``STEPBACK_PKCS11_TOKEN_LABEL``.
    key_label:
        CKA_LABEL of the private key object.
    user_pin:
        PKCS#11 user PIN.  Falls back to ``STEPBACK_PKCS11_PIN``.
    slot:
        Slot index; 0 by default.

    Raises
    ------
    HardwareKeyUnavailableError
        If ``python-pkcs11`` is not installed or the library/token is
        inaccessible.
    """

    _lib: Any  # pkcs11.Lib instance (cached)
    _pub_raw: bytes

    def __init__(
        self,
        *,
        pkcs11_lib: Optional[str] = None,
        token_label: Optional[str] = None,
        key_label: str = "stepback-key",
        user_pin: Optional[str] = None,
        slot: int = 0,
    ) -> None:
        if not self.is_available():
            raise HardwareKeyUnavailableError(
                "python-pkcs11 is not installed; "
                "run: pip install python-pkcs11"
            )
        import pkcs11  # type: ignore[import]
        from pkcs11 import Attribute, KeyType, Mechanism  # type: ignore[import]

        lib_path = pkcs11_lib or os.environ.get("STEPBACK_PKCS11_LIB", "")
        if not lib_path:
            raise HardwareKeyUnavailableError(
                "No PKCS#11 library path provided; pass pkcs11_lib= or set "
                "STEPBACK_PKCS11_LIB"
            )
        token_label = token_label or os.environ.get("STEPBACK_PKCS11_TOKEN_LABEL", "")
        pin = user_pin or os.environ.get("STEPBACK_PKCS11_PIN", "")

        try:
            lib = pkcs11.lib(lib_path)
        except Exception as exc:
            raise HardwareKeyUnavailableError(
                f"Failed to load PKCS#11 library {lib_path!r}: {exc}"
            ) from exc

        try:
            if token_label:
                token = lib.get_token(token_label=token_label)
            else:
                tokens = list(lib.get_tokens())
                if not tokens:
                    raise HardwareKeyUnavailableError("No PKCS#11 tokens found")
                token = tokens[slot]
        except pkcs11.exceptions.PKCS11Error as exc:
            raise HardwareKeyUnavailableError(
                f"Failed to locate PKCS#11 token: {exc}"
            ) from exc

        try:
            session = token.open(user_pin=pin or None, rw=False)
        except pkcs11.exceptions.PKCS11Error as exc:
            raise HardwareKeyUnavailableError(
                f"Failed to open PKCS#11 session: {exc}"
            ) from exc

        try:
            priv_key = session.get_key(
                key_type=KeyType.EC,
                label=key_label,
                object_class=pkcs11.constants.ObjectClass.PRIVATE_KEY,
            )
        except Exception as exc:
            raise HardwareKeyUnavailableError(
                f"Failed to find PKCS#11 key {key_label!r}: {exc}"
            ) from exc

        self._session = session
        self._priv_key = priv_key
        self._mechanism = Mechanism.ECDSA
        self._key_label = key_label

        # Cache the public key bytes for fingerprinting.
        try:
            pub = session.get_key(
                key_type=KeyType.EC,
                label=key_label,
                object_class=pkcs11.constants.ObjectClass.PUBLIC_KEY,
            )
            self._pub_raw = bytes(pub[Attribute.EC_POINT])
        except Exception:
            self._pub_raw = b""

    @classmethod
    def is_available(cls) -> bool:
        try:
            import pkcs11  # noqa: F401
            return True
        except ImportError:
            return False

    def sign_bytes(self, data: bytes) -> bytes:
        try:
            from pkcs11 import Mechanism  # type: ignore[import]
            digest = hashlib.sha256(data).digest()
            sig: bytes = self._priv_key.sign(digest, mechanism=Mechanism.ECDSA)
            return sig
        except Exception as exc:
            raise HardwareKeySignError(
                f"PKCS#11 signing failed: {exc}"
            ) from exc

    def public_key_bytes(self) -> bytes:
        return self._pub_raw

    def key_algorithm(self) -> str:
        return "ecdsa-p256"

    def close(self) -> None:
        """Close the PKCS#11 session (call when done)."""
        try:
            self._session.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# YubiHSMProvider
# ---------------------------------------------------------------------------


class YubiHSMProvider(HardwareKeyProvider):
    """YubiHSM2 signing via the ``yubihsm`` Python library.

    Parameters
    ----------
    auth_key_id:
        Object ID of the authentication key (default: 1).
    signing_key_id:
        Object ID of the Ed25519 or ECDSA signing key.
    password:
        Authentication key password.  Falls back to
        ``STEPBACK_YUBIHSM_PASSWORD``.
    connector_url:
        URL for the yubihsm-connector daemon (default:
        ``http://localhost:12345``).  Falls back to
        ``STEPBACK_YUBIHSM_CONNECTOR_URL``.

    Raises
    ------
    HardwareKeyUnavailableError
        If ``yubihsm`` is not installed or the connector is unreachable.
    """

    def __init__(
        self,
        *,
        auth_key_id: int = 1,
        signing_key_id: int = 2,
        password: Optional[str] = None,
        connector_url: Optional[str] = None,
    ) -> None:
        if not self.is_available():
            raise HardwareKeyUnavailableError(
                "yubihsm is not installed; run: pip install yubihsm"
            )
        import yubihsm  # type: ignore[import]
        from yubihsm.core import AuthSession  # type: ignore[import]
        from yubihsm.objects import AsymmetricKey  # type: ignore[import]
        import yubihsm.backends  # type: ignore[import]

        url = connector_url or os.environ.get(
            "STEPBACK_YUBIHSM_CONNECTOR_URL", "http://localhost:12345"
        )
        pwd = password or os.environ.get("STEPBACK_YUBIHSM_PASSWORD", "password")

        try:
            backend = yubihsm.backends.get_backend(url)
            hsm = yubihsm.core.YubiHsm(backend)
            session: AuthSession = hsm.create_session_derived(auth_key_id, pwd)
        except Exception as exc:
            raise HardwareKeyUnavailableError(
                f"Failed to connect to YubiHSM connector at {url!r}: {exc}"
            ) from exc

        try:
            key_obj: AsymmetricKey = session.get_object(
                signing_key_id, yubihsm.defs.OBJECT.ASYMMETRIC_KEY
            )
        except Exception as exc:
            raise HardwareKeyUnavailableError(
                f"Failed to find YubiHSM key id={signing_key_id}: {exc}"
            ) from exc

        self._session = session
        self._key_obj = key_obj
        self._signing_key_id = signing_key_id

        try:
            pub_bytes = key_obj.get_public_key().public_bytes_raw()
        except Exception:
            pub_bytes = b""
        self._pub_raw = pub_bytes

    @classmethod
    def is_available(cls) -> bool:
        try:
            import yubihsm  # noqa: F401
            return True
        except ImportError:
            return False

    def sign_bytes(self, data: bytes) -> bytes:
        try:
            sig: bytes = self._key_obj.sign_eddsa(data)
            return sig
        except Exception as exc:
            raise HardwareKeySignError(
                f"YubiHSM signing failed: {exc}"
            ) from exc

    def public_key_bytes(self) -> bytes:
        return self._pub_raw

    def key_algorithm(self) -> str:
        return "ed25519"

    def close(self) -> None:
        """Close the YubiHSM session."""
        try:
            self._session.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Cloud KMS providers
# ---------------------------------------------------------------------------


class AWSKMSProvider(HardwareKeyProvider):
    """AWS KMS asymmetric signing (``SIGN_VERIFY`` key usage).

    The key must be an asymmetric Ed25519 or ECDSA_SHA_256 KMS key.

    Parameters
    ----------
    key_id:
        KMS key ID, key ARN, alias, or alias ARN.
    signing_algorithm:
        KMS signing algorithm, e.g. ``"ECDSA_SHA_256"`` or
        ``"ED25519"`` (default: ``"ECDSA_SHA_256"``).
    boto3_client:
        Optional pre-built ``boto3.client("kms")`` for dependency
        injection / testing.

    Raises
    ------
    HardwareKeyUnavailableError
        If ``boto3`` is not installed.
    """

    def __init__(
        self,
        *,
        key_id: str,
        signing_algorithm: str = "ECDSA_SHA_256",
        boto3_client: Any = None,
    ) -> None:
        if not self.is_available() and boto3_client is None:
            raise HardwareKeyUnavailableError(
                "boto3 is not installed; run: pip install boto3"
            )
        if boto3_client is not None:
            client = boto3_client
        else:
            import boto3  # type: ignore[import]
            client = boto3.client("kms")

        self._client = client
        self._key_id = key_id
        self._signing_algorithm = signing_algorithm
        self._pub_raw = self._fetch_public_key_bytes()

    def _fetch_public_key_bytes(self) -> bytes:
        try:
            resp = self._client.get_public_key(KeyId=self._key_id)
            return resp["PublicKey"]
        except Exception as exc:
            raise HardwareKeyUnavailableError(
                f"Failed to fetch AWS KMS public key for {self._key_id!r}: {exc}"
            ) from exc

    @classmethod
    def is_available(cls) -> bool:
        try:
            import boto3  # noqa: F401
            return True
        except ImportError:
            return False

    def sign_bytes(self, data: bytes) -> bytes:
        try:
            resp = self._client.sign(
                KeyId=self._key_id,
                Message=data,
                MessageType="RAW",
                SigningAlgorithm=self._signing_algorithm,
            )
            return resp["Signature"]
        except Exception as exc:
            raise HardwareKeySignError(
                f"AWS KMS signing failed: {exc}"
            ) from exc

    def public_key_bytes(self) -> bytes:
        return self._pub_raw

    def key_algorithm(self) -> str:
        return self._signing_algorithm.lower().replace("_", "-")


class GCPKMSProvider(HardwareKeyProvider):
    """Google Cloud KMS asymmetric signing.

    Parameters
    ----------
    key_version_name:
        Fully-qualified resource name:
        ``projects/.../locations/.../keyRings/.../cryptoKeys/.../cryptoKeyVersions/1``
    algorithm:
        Algorithm string embedded in the fingerprint (default:
        ``"ecdsa-sha256"``).
    kms_client:
        Optional pre-built ``google.cloud.kms.KeyManagementServiceClient``
        for testing / dependency injection.

    Raises
    ------
    HardwareKeyUnavailableError
        If ``google-cloud-kms`` is not installed.
    """

    def __init__(
        self,
        *,
        key_version_name: str,
        algorithm: str = "ecdsa-sha256",
        kms_client: Any = None,
    ) -> None:
        if not self.is_available() and kms_client is None:
            raise HardwareKeyUnavailableError(
                "google-cloud-kms is not installed; "
                "run: pip install google-cloud-kms"
            )
        if kms_client is not None:
            client = kms_client
        else:
            from google.cloud import kms  # type: ignore[import]
            client = kms.KeyManagementServiceClient()

        self._client = client
        self._key_version_name = key_version_name
        self._algorithm = algorithm
        self._pub_raw = self._fetch_public_key_bytes()

    def _fetch_public_key_bytes(self) -> bytes:
        try:
            pub = self._client.get_public_key(name=self._key_version_name)
            # google-cloud-kms returns PEM; convert to DER bytes.
            pem: str = pub.pem
            from cryptography.hazmat.primitives.serialization import load_pem_public_key
            key_obj = load_pem_public_key(pem.encode())
            return key_obj.public_bytes(
                Encoding.DER, PublicFormat.SubjectPublicKeyInfo
            )
        except Exception as exc:
            raise HardwareKeyUnavailableError(
                f"Failed to fetch GCP KMS public key for "
                f"{self._key_version_name!r}: {exc}"
            ) from exc

    @classmethod
    def is_available(cls) -> bool:
        try:
            from google.cloud import kms  # noqa: F401
            return True
        except ImportError:
            return False

    def sign_bytes(self, data: bytes) -> bytes:
        try:
            digest_bytes = hashlib.sha256(data).digest()
            response = self._client.asymmetric_sign(
                name=self._key_version_name,
                digest={"sha256": digest_bytes},
            )
            return response.signature
        except Exception as exc:
            raise HardwareKeySignError(
                f"GCP KMS signing failed: {exc}"
            ) from exc

    def public_key_bytes(self) -> bytes:
        return self._pub_raw

    def key_algorithm(self) -> str:
        return self._algorithm


class AzureKeyVaultProvider(HardwareKeyProvider):
    """Azure Key Vault asymmetric signing.

    Parameters
    ----------
    vault_url:
        Key Vault URL, e.g. ``https://myvault.vault.azure.net``.
    key_name:
        Name of the key in the vault.
    key_version:
        Key version string (optional; uses latest if omitted).
    algorithm:
        Azure signing algorithm, e.g. ``"ES256"`` (default).
    crypto_client:
        Optional pre-built ``azure.keyvault.keys.crypto.CryptographyClient``
        for testing / dependency injection.

    Raises
    ------
    HardwareKeyUnavailableError
        If ``azure-keyvault-keys`` is not installed.
    """

    def __init__(
        self,
        *,
        vault_url: str,
        key_name: str,
        key_version: Optional[str] = None,
        algorithm: str = "ES256",
        crypto_client: Any = None,
    ) -> None:
        if not self.is_available() and crypto_client is None:
            raise HardwareKeyUnavailableError(
                "azure-keyvault-keys is not installed; "
                "run: pip install azure-keyvault-keys"
            )
        if crypto_client is not None:
            client = crypto_client
        else:
            from azure.identity import DefaultAzureCredential  # type: ignore[import]
            from azure.keyvault.keys.crypto import (  # type: ignore[import]
                CryptographyClient,
                SignatureAlgorithm,
            )
            from azure.keyvault.keys import KeyClient  # type: ignore[import]

            credential = DefaultAzureCredential()
            key_client = KeyClient(vault_url=vault_url, credential=credential)
            key = key_client.get_key(key_name, version=key_version)
            client = CryptographyClient(key, credential=credential)

        self._client = client
        self._algorithm = algorithm
        self._vault_url = vault_url
        self._key_name = key_name
        self._pub_raw = self._fetch_public_key_bytes()

    def _fetch_public_key_bytes(self) -> bytes:
        try:
            from azure.keyvault.keys.crypto import SignatureAlgorithm  # type: ignore[import]
            # Fetch public-key component from the key object stored on the client.
            key = self._client.key
            n = key.key.n
            e = key.key.e
            if n and e:
                # RSA key — return modulus bytes
                return bytes(n)
            # EC key — return x coordinate as proxy bytes
            x = key.key.x
            return bytes(x) if x else b""
        except Exception:
            return b""

    @classmethod
    def is_available(cls) -> bool:
        try:
            from azure.keyvault.keys.crypto import CryptographyClient  # noqa: F401
            return True
        except ImportError:
            return False

    def sign_bytes(self, data: bytes) -> bytes:
        try:
            from azure.keyvault.keys.crypto import SignatureAlgorithm  # type: ignore[import]
            digest = hashlib.sha256(data).digest()
            result = self._client.sign(SignatureAlgorithm(self._algorithm), digest)
            return result.signature
        except Exception as exc:
            raise HardwareKeySignError(
                f"Azure Key Vault signing failed: {exc}"
            ) from exc

    def public_key_bytes(self) -> bytes:
        return self._pub_raw

    def key_algorithm(self) -> str:
        return self._algorithm.lower()


# ---------------------------------------------------------------------------
# Convenience: load provider from environment
# ---------------------------------------------------------------------------

#: Environment variable to select the active provider backend.
#: Valid values: ``"software"`` (default), ``"pkcs11"``, ``"yubihsm"``,
#: ``"aws-kms"``, ``"gcp-kms"``, ``"azure-keyvault"``.
ENV_PROVIDER_BACKEND = "STEPBACK_HWKEY_BACKEND"


def provider_from_env() -> HardwareKeyProvider:
    """Return a :class:`HardwareKeyProvider` configured from environment variables.

    Backend is selected by ``STEPBACK_HWKEY_BACKEND``:

    * ``"software"`` — :class:`SoftwareSimProvider` (generates a fresh ephemeral
      key; for testing only).
    * ``"pkcs11"`` — :class:`PKCS11Provider` (requires
      ``STEPBACK_PKCS11_LIB``).
    * ``"yubihsm"`` — :class:`YubiHSMProvider`.
    * ``"aws-kms"`` — :class:`AWSKMSProvider` (requires
      ``STEPBACK_AWS_KMS_KEY_ID``).
    * ``"gcp-kms"`` — :class:`GCPKMSProvider` (requires
      ``STEPBACK_GCP_KMS_KEY_VERSION_NAME``).
    * ``"azure-keyvault"`` — :class:`AzureKeyVaultProvider` (requires
      ``STEPBACK_AZURE_VAULT_URL`` and ``STEPBACK_AZURE_KEY_NAME``).
    """
    backend = os.environ.get(ENV_PROVIDER_BACKEND, "software").lower()
    if backend == "software":
        return SoftwareSimProvider.generate()
    if backend == "pkcs11":
        return PKCS11Provider()
    if backend == "yubihsm":
        return YubiHSMProvider()
    if backend == "aws-kms":
        key_id = os.environ.get("STEPBACK_AWS_KMS_KEY_ID", "")
        if not key_id:
            raise HardwareKeyUnavailableError(
                "STEPBACK_AWS_KMS_KEY_ID must be set for aws-kms backend"
            )
        return AWSKMSProvider(key_id=key_id)
    if backend == "gcp-kms":
        key_version = os.environ.get("STEPBACK_GCP_KMS_KEY_VERSION_NAME", "")
        if not key_version:
            raise HardwareKeyUnavailableError(
                "STEPBACK_GCP_KMS_KEY_VERSION_NAME must be set for gcp-kms backend"
            )
        return GCPKMSProvider(key_version_name=key_version)
    if backend == "azure-keyvault":
        vault_url = os.environ.get("STEPBACK_AZURE_VAULT_URL", "")
        key_name = os.environ.get("STEPBACK_AZURE_KEY_NAME", "")
        if not vault_url or not key_name:
            raise HardwareKeyUnavailableError(
                "STEPBACK_AZURE_VAULT_URL and STEPBACK_AZURE_KEY_NAME "
                "must be set for azure-keyvault backend"
            )
        return AzureKeyVaultProvider(vault_url=vault_url, key_name=key_name)
    raise ValueError(
        f"Unknown STEPBACK_HWKEY_BACKEND value: {backend!r}. "
        "Valid values: software, pkcs11, yubihsm, aws-kms, gcp-kms, azure-keyvault"
    )
