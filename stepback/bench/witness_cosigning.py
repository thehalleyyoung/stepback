"""Witness cosigning for public benchmark trace packs (Step 131).

Witness cosigning lets independent third parties attest that a benchmark
trace pack existed **before** evaluation took place, preventing submitters
from cherry-picking or tuning a trace pack after seeing evaluation results.

Protocol
--------
A witness produces a :class:`WitnessCommitment` by signing the canonical JSON
encoding of a domain-separated payload::

    {
        "type": "stepback.benchmark.witness.v1",
        "trace_pack_sha256": "sha256:<64-char lowercase hex>",
        "corpus_id": "<corpus identifier>",
        "committed_at": "<ISO 8601 UTC, e.g. 2026-05-12T09:00:00Z>",
        "witness_identity": "<stable string identifier>",
        "witness_public_key": "ed25519:<64-char lowercase hex>"
    }

The binding of ``trace_pack_sha256`` to ``committed_at`` inside the signed
payload means the witness asserts both *what* they saw and *when* they saw
it.  The ``committed_at`` field is an *assertion* by the witness — it does
not by itself prevent a colluding witness from backdating.  For stronger
guarantees, integrate with a transparency log (see Step 132).

Threat model
~~~~~~~~~~~~
* Prevents **unilateral** submitter backdating: the submitter cannot forge a
  cosignature without the witness private key.
* Does **not** prevent witness collusion or backdating by a complicit
  witness.  A trusted witness registry (see :func:`verify_witness_commitments`
  ``trusted_witnesses`` parameter) raises the bar by requiring the attesting
  key to be known to the leaderboard operator.

Trusted-witness registry
~~~~~~~~~~~~~~~~~~~~~~~~
``verify_witness_commitments`` accepts an optional ``trusted_witnesses``
list of ``(identity, public_key)`` pairs.  When supplied, only commitments
from registered witnesses are counted.  Unknown witnesses are skipped but
do not cause an error.  When ``trusted_witnesses`` is ``None`` the function
counts any cryptographically-valid commitment (useful for testing; not
recommended for public leaderboards).

On-disk shape
~~~~~~~~~~~~~
Each commitment is serialised as a compact JSON object and embedded in the
submission manifest under ``"witness_cosignatures"``::

    {
        "committed_at": "2026-05-12T09:00:00Z",
        "witness_identity": "stepback-foundation",
        "witness_public_key": "ed25519:<hex>",
        "signature": "ed25519:<hex>"
    }

``trace_pack_sha256`` and ``corpus_id`` are *not* repeated in the stored
object — they are taken from the surrounding ``SubmissionManifest`` for
deduplication.
"""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from ..canonical import canonical_json

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Capability name to record in capability frames when a pack is cosigned.
CAPABILITY_WITNESS_COSIGNING: str = "witness-cosigning"

#: Type discriminator used inside the domain-separated payload.
_WITNESS_SIG_TYPE: str = "stepback.benchmark.witness.v1"

#: Regex for a raw 64-char lowercase hex SHA-256 (without "sha256:" prefix).
_SHA256_RAW_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WitnessCosigningError(Exception):
    """Raised when witness commitment verification fails."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class WitnessCommitment:
    """A single witness's commitment that a trace pack existed at *committed_at*.

    Fields
    ------
    committed_at:
        UTC ISO 8601 string representing the witness's assertion of when they
        saw the trace pack.
    witness_identity:
        Stable, human-readable string identifying the witness (e.g.
        ``"stepback-foundation"`` or ``"github-actions-ci"``).
    witness_public_key:
        ``"ed25519:<64-char lowercase hex>"`` public key of the witness.
    signature:
        ``"ed25519:<128-char lowercase hex>"`` Ed25519 signature over the
        domain-separated payload produced by
        :func:`_witness_payload`.
    """

    committed_at: str
    witness_identity: str
    witness_public_key: str  # "ed25519:<hex>"
    signature: str  # "ed25519:<hex>"

    def to_dict(self) -> Dict[str, str]:
        return {
            "committed_at": self.committed_at,
            "witness_identity": self.witness_identity,
            "witness_public_key": self.witness_public_key,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WitnessCommitment":
        return cls(
            committed_at=str(d.get("committed_at", "")),
            witness_identity=str(d.get("witness_identity", "")),
            witness_public_key=str(d.get("witness_public_key", "")),
            signature=str(d.get("signature", "")),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_sha256(raw_or_prefixed: str) -> str:
    """Return ``"sha256:<64-char lowercase hex>"`` regardless of input form.

    Accepts:
    * ``"sha256:<hex>"`` (already prefixed)
    * ``"<64-char lowercase hex>"`` (raw hex)

    Raises :class:`WitnessCosigningError` for any other form.
    """
    s = raw_or_prefixed.lower()
    if s.startswith("sha256:"):
        hex_part = s[len("sha256:"):]
    else:
        hex_part = s
    if not _SHA256_RAW_RE.match(hex_part):
        raise WitnessCosigningError(
            f"invalid trace_pack_sha256 {raw_or_prefixed!r}: expected "
            "64-char lowercase hex (with or without 'sha256:' prefix)"
        )
    return "sha256:" + hex_part


def _witness_payload(
    trace_pack_sha256_normalised: str,
    corpus_id: str,
    committed_at: str,
    witness_identity: str,
    witness_public_key: str,
) -> bytes:
    """Return the canonical JSON bytes that a witness signs."""
    return canonical_json({
        "type": _WITNESS_SIG_TYPE,
        "trace_pack_sha256": trace_pack_sha256_normalised,
        "corpus_id": corpus_id,
        "committed_at": committed_at,
        "witness_identity": witness_identity,
        "witness_public_key": witness_public_key,
    })


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sign_trace_pack_commitment(
    trace_pack_sha256: str,
    corpus_id: str,
    witness_identity: str,
    private_key: Ed25519PrivateKey,
    *,
    committed_at: Optional[str] = None,
) -> WitnessCommitment:
    """Create and sign a witness commitment for *trace_pack_sha256*.

    Parameters
    ----------
    trace_pack_sha256:
        SHA-256 of the trace pack (raw hex or ``"sha256:<hex>"``).
    corpus_id:
        Corpus identifier (must match the submission's ``corpus_id``).
    witness_identity:
        Stable string identity of the signing witness.
    private_key:
        Ed25519 private key of the witness.
    committed_at:
        UTC ISO 8601 string for the commitment timestamp.  Defaults to
        *now* (``datetime.now(tz=timezone.utc).isoformat()``).

    Returns
    -------
    WitnessCommitment
        Signed commitment ready to embed in a submission manifest.
    """
    normalised = _normalise_sha256(trace_pack_sha256)
    if committed_at is None:
        committed_at = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()

    pub_bytes = private_key.public_key().public_bytes_raw()
    witness_public_key = "ed25519:" + pub_bytes.hex()

    payload = _witness_payload(
        normalised, corpus_id, committed_at, witness_identity, witness_public_key
    )
    sig_bytes = private_key.sign(payload)
    return WitnessCommitment(
        committed_at=committed_at,
        witness_identity=witness_identity,
        witness_public_key=witness_public_key,
        signature="ed25519:" + sig_bytes.hex(),
    )


def verify_witness_commitments(
    trace_pack_sha256: str,
    corpus_id: str,
    commitments: Sequence[WitnessCommitment],
    *,
    trusted_witnesses: Optional[List[Tuple[str, str]]] = None,
    min_witnesses: int = 1,
    evaluation_timestamp: Optional[str] = None,
) -> int:
    """Verify witness commitments and return the count of valid ones.

    Parameters
    ----------
    trace_pack_sha256:
        SHA-256 of the trace pack (raw hex or ``"sha256:<hex>"``).
    corpus_id:
        Corpus identifier; must match each commitment's signed payload.
    commitments:
        Sequence of :class:`WitnessCommitment` objects to verify.
    trusted_witnesses:
        Optional list of ``(identity, "ed25519:<hex>")`` pairs representing
        the operator's trusted-witness registry.  Only commitments from
        registered witnesses are counted.  If ``None``, any
        cryptographically-valid commitment is counted.
    min_witnesses:
        Minimum number of valid commitments required.  Raises
        :class:`WitnessCosigningError` if fewer are found.
    evaluation_timestamp:
        Optional UTC ISO 8601 string of the earliest benchmark evaluation
        time.  When supplied, commitments with ``committed_at`` *after*
        ``evaluation_timestamp`` are rejected (they cannot prove pre-existence
        before evaluation).

    Returns
    -------
    int
        Number of valid, non-duplicate, in-trust commitments.

    Raises
    ------
    WitnessCosigningError
        * Malformed key or signature hex.
        * ``committed_at`` after ``evaluation_timestamp``.
        * Fewer than ``min_witnesses`` valid commitments.
    """
    normalised = _normalise_sha256(trace_pack_sha256)

    # Build trusted registry as a set of (identity, pk_hex) if provided.
    trusted_set: Optional[set] = None
    if trusted_witnesses is not None:
        trusted_set = set()
        for identity, pk in trusted_witnesses:
            norm_pk = pk.lower()
            if not norm_pk.startswith("ed25519:"):
                raise WitnessCosigningError(
                    f"trusted witness {identity!r}: public key must start with "
                    f"'ed25519:', got {pk!r}"
                )
            trusted_set.add((identity, norm_pk))

    seen_identities: set = set()
    valid_count = 0

    for commitment in commitments:
        identity = commitment.witness_identity
        pub_key_field = commitment.witness_public_key.lower()

        # Registry check: skip unknown witnesses (do not error).
        if trusted_set is not None:
            if (identity, pub_key_field) not in trusted_set:
                continue

        # Deduplication by identity (first occurrence wins).
        if identity in seen_identities:
            continue
        seen_identities.add(identity)

        # Timestamp ordering: commitment must predate evaluation.
        if evaluation_timestamp is not None:
            try:
                _committed = _parse_iso(commitment.committed_at)
                _eval = _parse_iso(evaluation_timestamp)
                if _committed > _eval:
                    raise WitnessCosigningError(
                        f"witness {identity!r}: committed_at "
                        f"{commitment.committed_at!r} is after the earliest "
                        f"evaluation timestamp {evaluation_timestamp!r}"
                    )
            except WitnessCosigningError:
                raise
            except (ValueError, TypeError) as exc:
                raise WitnessCosigningError(
                    f"witness {identity!r}: could not parse timestamps: {exc}"
                ) from exc

        # Cryptographic verification.
        try:
            if not pub_key_field.startswith("ed25519:"):
                raise WitnessCosigningError(
                    f"witness {identity!r}: unsupported public_key scheme "
                    f"{commitment.witness_public_key!r}"
                )
            pub_bytes = bytes.fromhex(pub_key_field.removeprefix("ed25519:"))
            pub: Ed25519PublicKey = Ed25519PublicKey.from_public_bytes(pub_bytes)

            sig_field = commitment.signature.lower()
            if not sig_field.startswith("ed25519:"):
                raise WitnessCosigningError(
                    f"witness {identity!r}: unsupported signature scheme "
                    f"{commitment.signature!r}"
                )
            sig_bytes = bytes.fromhex(sig_field.removeprefix("ed25519:"))

            payload = _witness_payload(
                normalised,
                corpus_id,
                commitment.committed_at,
                identity,
                commitment.witness_public_key,
            )
            pub.verify(sig_bytes, payload)
            valid_count += 1

        except WitnessCosigningError:
            raise
        except (ValueError, KeyError) as exc:
            raise WitnessCosigningError(
                f"witness {identity!r}: malformed key or signature hex: {exc}"
            ) from exc
        except InvalidSignature:
            # Signature bytes are well-formed but cryptographically invalid.
            # Do not count; continue to other witnesses.
            pass

    if valid_count < min_witnesses:
        raise WitnessCosigningError(
            f"witness cosigning threshold not met: required {min_witnesses} "
            f"valid commitment(s), got {valid_count}"
        )
    return valid_count


def _parse_iso(ts: str) -> _dt.datetime:
    """Parse an ISO 8601 UTC string to an aware datetime.

    Handles both ``Z`` suffix and ``+00:00`` offset.
    """
    # Python 3.10 fromisoformat does not handle 'Z'; normalise it.
    ts_norm = ts.replace("Z", "+00:00")
    dt = _dt.datetime.fromisoformat(ts_norm)
    if dt.tzinfo is None:
        # Treat naive as UTC.
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt
