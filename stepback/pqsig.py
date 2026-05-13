"""Post-quantum signature experiments (ML-DSA and SLH-DSA).

This module provides experimental post-quantum digital signature support
using ML-DSA (CRYSTALS-Dilithium, FIPS 204) and SLH-DSA (SPHINCS+, FIPS 205)
algorithms.  It is an *experiment*: the API, on-disk format, and algorithm
choices may change between minor versions.  Do not use for production key
management without a stability commitment.

Signatures are computed via the system OpenSSL (≥ 3.3 required for ML-DSA;
≥ 3.6 for SLH-DSA SHAKE variants).  If the algorithm is unavailable, every
function raises :class:`PQUnavailableError` so callers can decide whether to
skip or abort.

On-disk key format is standard PEM as produced by OpenSSL.  Signatures are
raw bytes.  Public-key fingerprints use the format ``<alg-id>:<sha256hex>``
where ``<alg-id>`` is the lower-case algorithm id (e.g. ``ml-dsa-65``) and
``<sha256hex>`` is the hex-encoded SHA-256 of the DER public key bytes (64
hex characters, no prefix).

Attestation-pack integration
-----------------------------
When an :class:`~stepback.attestation.AttestationPack` is PQ-co-signed the
pack body gains a ``pq_attestor_public_key`` field (included in the
``body_hash``) and the pack envelope gains a ``pq_signature`` field (excluded
from the hash, just like the Ed25519 ``signature`` field).  The PQ signature
covers a domain-separated canonical payload::

    canonical_json({
        "type": "stepback.attestation.pqsig.v1",
        "algorithm": "<alg-id>",
        "body_hash": "sha256:<hex>"
    })

Capability-frame constant
-------------------------
The string :data:`CAPABILITY_PQ_ATTESTATION` (``"pqsig"``) is the name to use
in capability frames written to ``.sb`` traces when the recorder also
attaches a PQ co-signature at attestation time.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

from .canonical import canonical_json

#: Capability name that announces PQ co-signing in trace and pack metadata.
CAPABILITY_PQ_ATTESTATION: str = "pqsig"

#: Canonical type discriminator used inside the PQ attestation payload.
_PQ_SIG_TYPE: str = "stepback.attestation.pqsig.v1"

#: Supported algorithms: maps lower-case stepback alg-id to OpenSSL alg name.
PQ_ALGORITHMS: dict[str, str] = {
    "ml-dsa-44": "ML-DSA-44",
    "ml-dsa-65": "ML-DSA-65",
    "ml-dsa-87": "ML-DSA-87",
    "slh-dsa-shake-256f": "SLH-DSA-SHAKE-256f",
}


class PQUnavailableError(Exception):
    """Raised when the requested PQ algorithm is not available.

    This can happen because the system OpenSSL predates the algorithm's
    introduction, or because the algorithm name is unknown.
    """

    #: Canonical error code; see :mod:`stepback.errors` for details.
    code: str = "SB501"


@dataclass
class PQKeyPair:
    """A post-quantum key pair (private + public PEM, plus algorithm id).

    The ``algorithm`` field uses the lower-case stepback alg-id, e.g.
    ``"ml-dsa-65"``.  ``private_key_pem`` and ``public_key_pem`` are
    standard PEM bytes as produced by OpenSSL.
    """

    algorithm: str
    private_key_pem: bytes
    public_key_pem: bytes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _openssl_bin() -> str:
    """Return the path to the system ``openssl`` binary."""
    path = shutil.which("openssl")
    if path is None:
        raise PQUnavailableError(
            "openssl binary not found; post-quantum signatures require "
            "OpenSSL ≥ 3.3 on PATH"
        )
    return path


def _run(args: list[str], *, stdin: Optional[bytes] = None) -> bytes:
    """Run an openssl command and return stdout bytes.

    Raises :class:`PQUnavailableError` when the error text indicates the
    algorithm is unknown.  Raises :class:`RuntimeError` for other failures.
    """
    result = subprocess.run(
        args,
        input=stdin,
        capture_output=True,
        timeout=30,
        shell=False,
    )
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace").strip()
        # OpenSSL reports unknown algorithms in several ways.
        if any(
            phrase in err.lower()
            for phrase in ("unknown algorithm", "no such algorithm", "unsupported algorithm",
                           "algorithm not found", "invalid algorithm")
        ):
            raise PQUnavailableError(
                f"PQ algorithm unavailable in this OpenSSL build: {err}"
            )
        raise RuntimeError(f"openssl failed ({result.returncode}): {err}")
    return result.stdout


def _alg_to_openssl(algorithm: str) -> str:
    """Resolve a stepback alg-id to an OpenSSL algorithm name."""
    alg_id = algorithm.lower()
    openssl_name = PQ_ALGORITHMS.get(alg_id)
    if openssl_name is None:
        raise PQUnavailableError(
            f"unknown PQ algorithm {algorithm!r}; supported: "
            + ", ".join(PQ_ALGORITHMS)
        )
    return openssl_name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_algorithm_available(algorithm: str) -> bool:
    """Return ``True`` iff ``algorithm`` is available in the system OpenSSL.

    Uses a fast ``openssl list`` probe rather than generating a full key.
    Returns ``False`` for any unknown algorithm name or if ``openssl`` is
    not on PATH.
    """
    try:
        openssl = _openssl_bin()
    except PQUnavailableError:
        return False
    try:
        alg_name = _alg_to_openssl(algorithm)
    except PQUnavailableError:
        return False
    result = subprocess.run(
        [openssl, "list", "-signature-algorithms"],
        capture_output=True,
        timeout=10,
        shell=False,
    )
    if result.returncode != 0:
        return False
    output = result.stdout.decode("utf-8", errors="replace")
    return alg_name in output


def generate_keypair(algorithm: str) -> PQKeyPair:
    """Generate a new PQ key pair for ``algorithm``.

    ``algorithm`` must be one of :data:`PQ_ALGORITHMS` (e.g. ``"ml-dsa-65"``).
    Raises :class:`PQUnavailableError` if the algorithm is not available.

    Private key material is written to a temporary file with mode 0o600 and
    deleted immediately after reading.
    """
    openssl = _openssl_bin()
    alg_id = algorithm.lower()
    openssl_name = _alg_to_openssl(alg_id)

    fd, privkey_path = tempfile.mkstemp(suffix=".pem", prefix="sb_pq_priv_")
    try:
        os.close(fd)
        os.chmod(privkey_path, 0o600)
        _run([openssl, "genpkey", "-algorithm", openssl_name, "-out", privkey_path])
        with open(privkey_path, "rb") as f:
            private_key_pem = f.read()
        # Extract the public key from the private key.
        public_key_pem = _run(
            [openssl, "pkey", "-in", privkey_path, "-pubout"]
        )
    finally:
        if os.path.exists(privkey_path):
            os.unlink(privkey_path)

    return PQKeyPair(
        algorithm=alg_id,
        private_key_pem=private_key_pem,
        public_key_pem=public_key_pem,
    )


def sign_bytes(message: bytes, keypair: PQKeyPair) -> bytes:
    """Return the raw PQ signature over ``message`` using ``keypair``.

    Both the private-key PEM and the message are passed via temporary files
    (not via argv or environment variables) so that key material does not
    appear in process listings.  Files are created with mode 0o600 and
    deleted on return.
    """
    openssl = _openssl_bin()
    privkey_fd, privkey_path = tempfile.mkstemp(suffix=".pem", prefix="sb_pq_priv_")
    msg_fd, msg_path = tempfile.mkstemp(suffix=".bin", prefix="sb_pq_msg_")
    sig_fd, sig_path = tempfile.mkstemp(suffix=".sig", prefix="sb_pq_sig_")
    try:
        os.chmod(privkey_path, 0o600)
        os.write(privkey_fd, keypair.private_key_pem)
        os.close(privkey_fd)
        privkey_fd = -1
        os.write(msg_fd, message)
        os.close(msg_fd)
        msg_fd = -1
        os.close(sig_fd)
        sig_fd = -1
        _run([
            openssl, "pkeyutl", "-sign",
            "-inkey", privkey_path,
            "-rawin",
            "-in", msg_path,
            "-out", sig_path,
        ])
        with open(sig_path, "rb") as f:
            return f.read()
    finally:
        for fd in (privkey_fd, msg_fd, sig_fd):
            if fd != -1:
                try:
                    os.close(fd)
                except OSError:
                    pass
        for p in (privkey_path, msg_path, sig_path):
            if os.path.exists(p):
                os.unlink(p)


def verify_bytes(
    message: bytes,
    signature: bytes,
    public_key_pem: bytes,
) -> None:
    """Verify a PQ signature; raise :class:`PQSignatureInvalid` on failure.

    ``public_key_pem`` is the PEM public key as returned by
    :func:`generate_keypair` or :func:`export_public_key_pem`.
    Raises :class:`PQUnavailableError` if OpenSSL lacks the algorithm.
    Raises :class:`PQSignatureInvalid` if the signature does not verify.
    """
    openssl = _openssl_bin()
    pub_fd, pub_path = tempfile.mkstemp(suffix=".pem", prefix="sb_pq_pub_")
    msg_fd, msg_path = tempfile.mkstemp(suffix=".bin", prefix="sb_pq_msg_")
    sig_fd, sig_path = tempfile.mkstemp(suffix=".sig", prefix="sb_pq_sig_")
    try:
        os.write(pub_fd, public_key_pem)
        os.close(pub_fd)
        pub_fd = -1
        os.write(msg_fd, message)
        os.close(msg_fd)
        msg_fd = -1
        os.write(sig_fd, signature)
        os.close(sig_fd)
        sig_fd = -1
        result = subprocess.run(
            [
                openssl, "pkeyutl", "-verify",
                "-pubin", "-inkey", pub_path,
                "-rawin",
                "-in", msg_path,
                "-sigfile", sig_path,
            ],
            capture_output=True,
            timeout=30,
            shell=False,
        )
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace").strip()
            if any(
                phrase in err.lower()
                for phrase in ("unknown algorithm", "no such algorithm",
                               "unsupported algorithm", "algorithm not found",
                               "invalid algorithm")
            ):
                raise PQUnavailableError(
                    f"PQ algorithm unavailable during verification: {err}"
                )
            raise PQSignatureInvalid(
                f"PQ signature verification failed: "
                + (result.stdout.decode("utf-8", errors="replace").strip() or err)
            )
    finally:
        for fd in (pub_fd, msg_fd, sig_fd):
            if fd != -1:
                try:
                    os.close(fd)
                except OSError:
                    pass
        for p in (pub_path, msg_path, sig_path):
            if os.path.exists(p):
                os.unlink(p)


class PQSignatureInvalid(Exception):
    """Raised when a PQ signature verification fails."""


def public_key_fingerprint(keypair: PQKeyPair) -> str:
    """Return the canonical public-key fingerprint string.

    Format: ``<alg-id>:<sha256hex>``

    The DER encoding is obtained from the PEM public key; the SHA-256 of the
    DER bytes is hex-encoded (64 characters, no prefix) to form the fingerprint
    value.
    """
    der = _pem_public_key_to_der(keypair.public_key_pem)
    sha256_digest = hashlib.sha256(der).hexdigest()
    return f"{keypair.algorithm}:{sha256_digest}"


def public_key_pem_from_fingerprint_is_unsupported() -> None:  # noqa: D401
    """Fingerprints are one-way; you cannot recover the PEM from one."""
    raise NotImplementedError("PQ fingerprints are SHA-256 digests; not reversible")


def _pem_public_key_to_der(public_key_pem: bytes) -> bytes:
    """Convert PEM public key to DER bytes via openssl."""
    openssl = _openssl_bin()
    pub_fd, pub_path = tempfile.mkstemp(suffix=".pem", prefix="sb_pq_pub_")
    try:
        os.write(pub_fd, public_key_pem)
        os.close(pub_fd)
        pub_fd = -1
        return _run([
            openssl, "pkey", "-pubin", "-in", pub_path, "-pubout", "-outform", "DER"
        ])
    finally:
        if pub_fd != -1:
            try:
                os.close(pub_fd)
            except OSError:
                pass
        if os.path.exists(pub_path):
            os.unlink(pub_path)


# ---------------------------------------------------------------------------
# Attestation pack integration helpers
# ---------------------------------------------------------------------------

def pq_attestation_payload(algorithm: str, body_hash: str) -> bytes:
    """Canonical payload that the PQ key signs in an attestation pack.

    The payload is domain-separated and binds the algorithm name so that
    a PQ signature over one algorithm cannot be confused with one over
    another.  It is deterministic given ``algorithm`` and ``body_hash``.
    """
    return canonical_json({
        "type": _PQ_SIG_TYPE,
        "algorithm": algorithm,
        "body_hash": body_hash,
    })


def pem_public_key_to_der(public_key_pem: bytes) -> bytes:
    """Convert a PEM public key to DER bytes via openssl."""
    return _pem_public_key_to_der(public_key_pem)


__all__ = [
    "CAPABILITY_PQ_ATTESTATION",
    "PQ_ALGORITHMS",
    "PQUnavailableError",
    "PQSignatureInvalid",
    "PQKeyPair",
    "is_algorithm_available",
    "generate_keypair",
    "sign_bytes",
    "verify_bytes",
    "public_key_fingerprint",
    "pem_public_key_to_der",
    "pq_attestation_payload",
]
