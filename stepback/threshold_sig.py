"""M-of-N threshold signing for long-lived production recorders.

Step 130 — incident-grade traces that require multiple independent
organizational witnesses to co-sign an attestation pack.

Design
------
Threshold policy is split across two on-disk regions, mirroring the
PQ co-signature pattern:

* **Body** (included in ``body_hash``, protected by the attestor's
  Ed25519 signature)::

      "threshold_policy": {
          "m_required": 2,
          "witnesses": [
              {"identity": "secops-primary", "public_key": "ed25519:<hex>"},
              {"identity": "legal-archive",  "public_key": "ed25519:<hex>"},
              {"identity": "prod-recorder",  "public_key": "ed25519:<hex>"}
          ]
      }

* **Envelope** (excluded from ``body_hash``, appended after signing)::

      "threshold_signatures": [
          {"identity": "secops-primary", "signature": "ed25519:<hex>"},
          {"identity": "legal-archive",  "signature": "ed25519:<hex>"}
      ]

Because the policy is in the body, the authorized-witness set and
the threshold ``M`` are signed by the attestor — stripping the
envelope or substituting fake witnesses is detected.

Domain-separated payload
------------------------
Each witness signs::

    canonical_json({
        "type": "stepback.attestation.threshold.v1",
        "body_hash": "sha256:...",
        "witness_identity": "<identity>",
        "witness_public_key": "ed25519:<hex>"
    })

This prevents cross-protocol replay (distinct from the primary
Ed25519 signature which covers only ``body_hash`` directly) and
binds each signature to a specific authorized identity and key.

Capability constant
-------------------
:data:`CAPABILITY_THRESHOLD_SIGNING` (``"threshold-signing"``) is
the name to record in capability frames when a trace pack uses
threshold signing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .canonical import canonical_json

#: Capability name for threshold-signed packs.
CAPABILITY_THRESHOLD_SIGNING: str = "threshold-signing"

#: Type discriminator embedded in each witness payload.
_THRESHOLD_SIG_TYPE: str = "stepback.attestation.threshold.v1"


class ThresholdSignatureError(Exception):
    """Raised when threshold signature verification fails."""

    #: Canonical error code; see :mod:`stepback.errors` for details.
    code: str = "SB504"


@dataclass
class WitnessSpec:
    """One authorized witness: stable identity + Ed25519 public key.

    ``identity`` must be unique within a :class:`ThresholdPolicy`; it is
    also embedded in each domain-separated witness payload.
    ``public_key`` is ``"ed25519:<hex>"`` (32 raw bytes, lower-case hex).
    """

    identity: str
    public_key: str  # "ed25519:<hex>"

    def to_dict(self) -> Dict[str, str]:
        return {"identity": self.identity, "public_key": self.public_key}

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "WitnessSpec":
        return cls(identity=d["identity"], public_key=d["public_key"])


@dataclass
class ThresholdPolicy:
    """Policy embedded in the attestation pack body (included in body hash).

    ``m_required``: minimum number of valid witness signatures needed.
    ``witnesses``: ordered list of authorized witnesses (must be unique
    by identity; duplicates are rejected at construction time).
    """

    m_required: int
    witnesses: List[WitnessSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.m_required < 1:
            raise ValueError(f"m_required must be >= 1, got {self.m_required}")
        identities = [w.identity for w in self.witnesses]
        if len(identities) != len(set(identities)):
            raise ValueError("ThresholdPolicy witnesses must have unique identities")
        pub_keys = [w.public_key for w in self.witnesses]
        if len(pub_keys) != len(set(pub_keys)):
            raise ValueError("ThresholdPolicy witnesses must have unique public keys")
        if self.m_required > len(self.witnesses):
            raise ValueError(
                f"m_required ({self.m_required}) > number of witnesses "
                f"({len(self.witnesses)})"
            )

    def to_dict(self) -> Dict:
        return {
            "m_required": self.m_required,
            "witnesses": [w.to_dict() for w in self.witnesses],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ThresholdPolicy":
        return cls(
            m_required=int(d["m_required"]),
            witnesses=[WitnessSpec.from_dict(w) for w in d.get("witnesses", [])],
        )

    def witness_by_identity(self, identity: str) -> Optional[WitnessSpec]:
        for w in self.witnesses:
            if w.identity == identity:
                return w
        return None


@dataclass
class WitnessSignature:
    """One witness's signature — goes in the pack envelope (not in body hash)."""

    identity: str
    signature: str  # "ed25519:<hex>"

    def to_dict(self) -> Dict[str, str]:
        return {"identity": self.identity, "signature": self.signature}

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "WitnessSignature":
        return cls(identity=d["identity"], signature=d["signature"])


def threshold_witness_payload(
    body_hash: str,
    witness_identity: str,
    witness_public_key: str,
) -> bytes:
    """Return the canonical bytes that a witness signs.

    Domain-separated to avoid cross-protocol collisions with the primary
    attestor Ed25519 signature (which signs only the bare ``body_hash``
    ASCII string).
    """
    return canonical_json({
        "type": _THRESHOLD_SIG_TYPE,
        "body_hash": body_hash,
        "witness_identity": witness_identity,
        "witness_public_key": witness_public_key,
    })


def sign_as_witness(
    body_hash: str,
    witness_spec: WitnessSpec,
    private_key: Ed25519PrivateKey,
) -> WitnessSignature:
    """Sign ``body_hash`` as the specified witness; return the signature object.

    The ``private_key`` must correspond to ``witness_spec.public_key``.
    No cross-check is performed here (the caller is responsible for
    key management); mismatches will be caught at verification.
    """
    payload = threshold_witness_payload(
        body_hash, witness_spec.identity, witness_spec.public_key
    )
    sig_bytes = private_key.sign(payload)
    return WitnessSignature(
        identity=witness_spec.identity,
        signature="ed25519:" + sig_bytes.hex(),
    )


def verify_threshold_signatures(
    body_hash: str,
    policy: ThresholdPolicy,
    signatures: Sequence[WitnessSignature],
    *,
    require_exact_m: bool = False,
) -> int:
    """Verify witness signatures against the policy; return the valid count.

    Algorithm:

    1. Iterate ``signatures`` in order; skip any whose identity is not
       in the policy (unknown witnesses are ignored, not counted).
    2. Deduplicate by identity — if the same identity appears twice,
       only the first occurrence is attempted.
    3. For each remaining entry, verify the Ed25519 signature against
       the domain-separated payload.
    4. Count valid signatures.
    5. If ``count < policy.m_required``, raise :class:`ThresholdSignatureError`.

    Returns the number of valid witness signatures on success.

    Raises :class:`ThresholdSignatureError` on:
    * malformed public-key or signature hex
    * fewer than ``m_required`` valid signatures
    """
    seen_identities: set[str] = set()
    valid_count = 0

    for ws in signatures:
        if ws.identity in seen_identities:
            # Duplicate — do not double-count.
            continue
        spec = policy.witness_by_identity(ws.identity)
        if spec is None:
            # Unknown witness — not in the authorized set; skip.
            continue
        seen_identities.add(ws.identity)

        # Decode and verify.
        try:
            pub_field = spec.public_key
            if not pub_field.startswith("ed25519:"):
                raise ThresholdSignatureError(
                    f"witness {ws.identity!r}: unsupported public_key scheme "
                    f"{pub_field!r}"
                )
            pub_bytes = bytes.fromhex(pub_field.removeprefix("ed25519:"))
            pub: Ed25519PublicKey = Ed25519PublicKey.from_public_bytes(pub_bytes)

            sig_field = ws.signature
            if not sig_field.startswith("ed25519:"):
                raise ThresholdSignatureError(
                    f"witness {ws.identity!r}: unsupported signature scheme "
                    f"{sig_field!r}"
                )
            sig_bytes = bytes.fromhex(sig_field.removeprefix("ed25519:"))

            payload = threshold_witness_payload(
                body_hash, ws.identity, spec.public_key
            )
            pub.verify(sig_bytes, payload)
            valid_count += 1
        except (ValueError, KeyError) as exc:
            # Hex decode failures or missing fields — treat as invalid.
            raise ThresholdSignatureError(
                f"witness {ws.identity!r}: malformed key or signature: {exc}"
            ) from exc
        except InvalidSignature:
            # Signature bytes are well-formed but cryptographically invalid.
            # Continue counting other witnesses rather than aborting: M-of-N
            # allows some witnesses to have invalid/corrupt entries.
            pass

    if valid_count < policy.m_required:
        raise ThresholdSignatureError(
            f"threshold not met: required {policy.m_required} valid witness "
            f"signatures, got {valid_count}"
        )
    return valid_count


def make_threshold_policy(
    witness_keys: Dict[str, Ed25519PrivateKey],
    m_required: int,
) -> ThresholdPolicy:
    """Convenience helper: build a :class:`ThresholdPolicy` from a dict of
    ``identity -> private_key`` pairs.

    The public key is extracted from each private key.  The returned policy
    is ready to embed in the pack body before signing.
    """
    witnesses = [
        WitnessSpec(
            identity=identity,
            public_key="ed25519:" + key.public_key().public_bytes_raw().hex(),
        )
        for identity, key in witness_keys.items()
    ]
    return ThresholdPolicy(m_required=m_required, witnesses=witnesses)


def collect_witness_signatures(
    body_hash: str,
    policy: ThresholdPolicy,
    signing_keys: Dict[str, Ed25519PrivateKey],
) -> List[WitnessSignature]:
    """Sign ``body_hash`` with every key in ``signing_keys`` that is
    authorized in ``policy``; return the list of :class:`WitnessSignature`.

    Keys whose identity is not in the policy are silently skipped.
    Raises :class:`ThresholdSignatureError` if fewer than ``policy.m_required``
    keys are authorized (i.e. the caller cannot produce a valid threshold
    set).
    """
    sigs: List[WitnessSignature] = []
    for identity, priv_key in signing_keys.items():
        spec = policy.witness_by_identity(identity)
        if spec is None:
            continue
        sigs.append(sign_as_witness(body_hash, spec, priv_key))
    if len(sigs) < policy.m_required:
        raise ThresholdSignatureError(
            f"only {len(sigs)} authorized signing keys provided, "
            f"but m_required={policy.m_required}"
        )
    return sigs
