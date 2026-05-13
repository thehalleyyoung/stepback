"""Regulator-replay attestation packs.

An attestation pack is the artifact `stepback verify --attestation-out`
hands to a regulator (README use-case 5). It bundles, for each `.sb`
trace in a corpus:

* The trace's identity (``trace_chain_hash`` plus the recorder's
  Ed25519 public key fingerprint).
* The result of HMAC-chain + signature verification.
* The result of replaying the trace under an optional substitution
  list (typically a ``PolicySubstitution`` pinning a newer policy
  version), including the dirty-step count, the divergent-step count
  vs. the recorded run, and the cost delta.

The pack itself is content-addressed (sha256 of its canonical JSON
body) and Ed25519-signed by an attestor key. The auditor verifies
the pack signature and then trusts every per-trace verdict inside.

On-disk shape (canonical JSON, no embedded binary)::

    {
      "magic": "stepback/.pack",
      "format_version": 1,
      "produced_at": "2026-04-15T13:04:00Z",
      "attestor_public_key": "ed25519:<hex>",
      "policy_version_pin": "2026-04-15",   # optional
      "summary": {
        "trace_count": 12418,
        "verified_ok": 12418,
        "verified_fail": 0,
        "replayed_ok": 12418,
        "divergent_traces": 41,
        "total_cost_delta_usd": 12.47
      },
      "entries": [ {AttestationEntry...}, ... ],
      "body_hash": "sha256:<hex>",
      "signature": "ed25519:<hex>"
    }

Verification is *strictly* offline: the auditor needs only the
attestor public key — no `.sb` traces, no Python source, no policy
file.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .branch_io import substitution_from_dict, substitution_to_dict, trace_chain_hash
from .canonical import canonical_json, sha256_hex
from .replay import Executor, Trace, replay
from .substitutions import Substitution, SubstitutionSet
from .trace_reader import TraceVerificationError, verify_trace

PACK_MAGIC = "stepback/.pack"
PACK_FORMAT_VERSION = 1


class AttestationVerificationError(Exception):
    """Raised when an attestation pack fails signature or hash verification."""

    #: Canonical error code; see :mod:`stepback.errors` for details.
    code: str = "SB500"


@dataclass
class AttestationEntry:
    """One trace's verdict inside an attestation pack."""

    trace_path: str
    trace_chain_hash: str
    recorder_public_key: str
    recorder_version: Optional[str]
    canonicalisation_version: Optional[str]
    step_count: int
    verify_status: str  # "ok" | "fail"
    verify_error: Optional[str] = None
    replay_status: str = "skipped"  # "ok" | "skipped" | "error"
    replay_error: Optional[str] = None
    dirty_step_count: int = 0
    cache_hit_count: int = 0
    real_executions: int = 0
    divergent_step_count: int = 0
    total_cost_recorded_usd: float = 0.0
    total_cost_replayed_usd: float = 0.0
    total_cost_delta_usd: float = 0.0
    divergent_step_ids: List[str] = field(default_factory=list)
    substitutions: List[Dict[str, Any]] = field(default_factory=list)
    merkle_root: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _utc_now_iso() -> str:
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _public_key_fingerprint(pub_hex: str) -> str:
    return f"ed25519:{pub_hex}"


def _coerce_subs(
    subs: Optional[Iterable[Substitution] | SubstitutionSet],
) -> SubstitutionSet:
    if subs is None:
        return SubstitutionSet()
    if isinstance(subs, SubstitutionSet):
        return subs
    out = SubstitutionSet()
    for s in subs:
        out.add(s)
    return out


def _attest_one(
    trace_path: str,
    *,
    hmac_key: bytes,
    substitutions: SubstitutionSet,
    executor: Optional[Executor],
) -> AttestationEntry:
    # 1) HMAC + signature verification.
    recorder_pub = ""
    recorder_version = None
    canon_version = None
    step_count = 0
    merkle_root_hex: Optional[str] = None
    try:
        v = verify_trace(trace_path, hmac_key)
        recorder_pub = v.public_key_hex
        recorder_version = v.header.get("recorder_version")
        canon_version = v.header.get("canonicalisation_version")
        step_count = len(v.steps)
        merkle_root_hex = v.merkle_root
    except TraceVerificationError as e:
        return AttestationEntry(
            trace_path=trace_path,
            trace_chain_hash="",
            recorder_public_key="",
            recorder_version=None,
            canonicalisation_version=None,
            step_count=0,
            verify_status="fail",
            verify_error=str(e),
            replay_status="skipped",
            substitutions=[substitution_to_dict(s) for s in substitutions.items],
        )

    # 2) Replay (recorded baseline + counterfactual under substitutions).
    try:
        t: Trace = replay(trace_path, hmac_key=hmac_key)
        chain = trace_chain_hash(t.recorded_steps)
        # Default to a recorded-fallback executor: at audit time the
        # auditor doesn't have the LLM wired up, but every step
        # downstream of a substitution is still semantically dirty.
        # Fallback lets the engine surface the divergence without
        # crashing on a real LLM call.
        ex = executor or Executor(fallback_recorded=True)
        baseline = t.run_replay(SubstitutionSet(), ex)
        counter = (
            t.run_replay(substitutions, executor or Executor(fallback_recorded=True))
            if substitutions
            else baseline
        )
        divergent_ids: List[str] = []
        for a, b in zip(baseline.steps, counter.steps):
            # Compare the live current_inputs_hash + outputs_hash via
            # the StepView surface (recorded_inputs_hash vs current).
            if a.current_inputs_hash != b.current_inputs_hash or b.dirty:
                divergent_ids.append(b.step_id)
        delta = round(counter.total_cost_usd - baseline.total_cost_usd, 8)
        return AttestationEntry(
            trace_path=trace_path,
            trace_chain_hash=chain,
            recorder_public_key=_public_key_fingerprint(recorder_pub),
            recorder_version=recorder_version,
            canonicalisation_version=canon_version,
            step_count=step_count,
            verify_status="ok",
            replay_status="ok",
            dirty_step_count=counter.dirty_count,
            cache_hit_count=counter.cache_hit_count,
            real_executions=counter.real_executions,
            divergent_step_count=len(divergent_ids),
            total_cost_recorded_usd=round(baseline.total_cost_usd, 8),
            total_cost_replayed_usd=round(counter.total_cost_usd, 8),
            total_cost_delta_usd=delta,
            divergent_step_ids=divergent_ids,
            substitutions=[substitution_to_dict(s) for s in substitutions.items],
            merkle_root=merkle_root_hex,
        )
    except Exception as e:  # pragma: no cover - replay errors are rare
        return AttestationEntry(
            trace_path=trace_path,
            trace_chain_hash="",
            recorder_public_key=_public_key_fingerprint(recorder_pub),
            recorder_version=recorder_version,
            canonicalisation_version=canon_version,
            step_count=step_count,
            verify_status="ok",
            replay_status="error",
            replay_error=f"{type(e).__name__}: {e}",
            substitutions=[substitution_to_dict(s) for s in substitutions.items],
            merkle_root=merkle_root_hex,
        )


@dataclass
class AttestationPack:
    """In-memory representation of an attestation pack body."""

    produced_at: str
    attestor_public_key: str
    entries: List[AttestationEntry]
    summary: Dict[str, Any]
    policy_version_pin: Optional[str] = None
    pack_format_version: int = PACK_FORMAT_VERSION
    builder_version: str = "stepback/0.1"
    threshold_policy: Optional[Any] = None  # ThresholdPolicy | None

    def body_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "magic": PACK_MAGIC,
            "format_version": self.pack_format_version,
            "builder_version": self.builder_version,
            "produced_at": self.produced_at,
            "attestor_public_key": self.attestor_public_key,
            "policy_version_pin": self.policy_version_pin,
            "summary": self.summary,
            "entries": [e.to_dict() for e in self.entries],
        }
        if self.threshold_policy is not None:
            d["threshold_policy"] = self.threshold_policy.to_dict()
        return d


def _summarise(entries: List[AttestationEntry]) -> Dict[str, Any]:
    return {
        "trace_count": len(entries),
        "verified_ok": sum(1 for e in entries if e.verify_status == "ok"),
        "verified_fail": sum(1 for e in entries if e.verify_status == "fail"),
        "replayed_ok": sum(1 for e in entries if e.replay_status == "ok"),
        "replayed_error": sum(1 for e in entries if e.replay_status == "error"),
        "divergent_traces": sum(1 for e in entries if e.divergent_step_count > 0),
        "total_cost_delta_usd": round(
            sum(e.total_cost_delta_usd for e in entries), 8
        ),
        "total_dirty_steps": sum(e.dirty_step_count for e in entries),
        "merkle_summarised_traces": sum(
            1 for e in entries if e.merkle_root is not None
        ),
    }


def build_attestation_pack(
    traces: Iterable[str | tuple[str, bytes]],
    *,
    hmac_key: Optional[bytes] = None,
    substitutions: Optional[Iterable[Substitution] | SubstitutionSet] = None,
    policy_version_pin: Optional[str] = None,
    attestor_signing_key: Optional[Ed25519PrivateKey] = None,
    executor: Optional[Executor] = None,
    threshold_policy: Optional[Any] = None,
) -> AttestationPack:
    """Build an attestation pack over ``traces``.

    Each entry in ``traces`` is either:
      * a string path (uses the shared ``hmac_key``), or
      * a ``(path, per_trace_hmac_key)`` tuple (different traces may
        be signed under different keys, which is the production case).

    ``substitutions`` is applied to every trace (typically a single
    ``PolicySubstitution`` pinning a newer policy version).

    If ``attestor_signing_key`` is omitted, a fresh Ed25519 key is
    generated; the public key is recorded in the pack and the private
    key is *not* returned (lost on garbage collection — caller must
    pass one in for any persistent attestation).
    """
    sub_set = _coerce_subs(substitutions)
    entries: List[AttestationEntry] = []
    for spec in traces:
        if isinstance(spec, tuple):
            path, key = spec
        else:
            path = spec
            if hmac_key is None:
                raise ValueError(
                    "hmac_key is required when traces are bare path strings"
                )
            key = hmac_key
        entries.append(
            _attest_one(
                path,
                hmac_key=key,
                substitutions=sub_set,
                executor=executor,
            )
        )
    summary = _summarise(entries)
    if attestor_signing_key is None:
        attestor_signing_key = Ed25519PrivateKey.generate()
    pub = attestor_signing_key.public_key().public_bytes_raw().hex()
    return AttestationPack(
        produced_at=_utc_now_iso(),
        attestor_public_key=_public_key_fingerprint(pub),
        entries=entries,
        summary=summary,
        policy_version_pin=policy_version_pin,
        threshold_policy=threshold_policy,
    )


def write_attestation_pack(
    pack: AttestationPack,
    path: str,
    *,
    signing_key: Ed25519PrivateKey,
    threshold_signing_keys: Optional[Dict[str, Any]] = None,
) -> str:
    """Serialise + sign + write ``pack`` to ``path``. Returns body sha256.

    The on-disk file is canonical JSON with two extra fields appended
    by the writer: ``body_hash`` (sha256 of the canonical body) and
    ``signature`` (Ed25519 over the body hash bytes). The attestor
    public key inside the body MUST match ``signing_key`` — passing
    a mismatched key raises ``ValueError``.

    If ``threshold_signing_keys`` is provided and the pack carries a
    ``threshold_policy``, the witnesses sign the body hash and their
    signatures are appended as an unsigned envelope field
    ``threshold_signatures``.
    """
    pub = signing_key.public_key().public_bytes_raw().hex()
    expected = _public_key_fingerprint(pub)
    if pack.attestor_public_key != expected:
        raise ValueError(
            "attestor_public_key in pack does not match signing_key; "
            "rebuild the pack with this signing key"
        )
    body = pack.body_dict()
    body_bytes = canonical_json(body)
    body_hash = sha256_hex(body_bytes)
    sig = signing_key.sign(body_hash.encode("ascii"))
    out = dict(body)
    out["body_hash"] = body_hash
    out["signature"] = "ed25519:" + sig.hex()
    if threshold_signing_keys is not None and pack.threshold_policy is not None:
        from .threshold_sig import collect_witness_signatures
        witness_sigs = collect_witness_signatures(
            body_hash, pack.threshold_policy, threshold_signing_keys
        )
        out["threshold_signatures"] = [
            {"identity": ws.identity, "signature": ws.signature}
            for ws in witness_sigs
        ]
    elif threshold_signing_keys is not None and pack.threshold_policy is None:
        raise ValueError(
            "threshold_signing_keys provided but pack has no threshold_policy; "
            "pass threshold_policy= to build_attestation_pack"
        )
    elif threshold_signing_keys is None and pack.threshold_policy is not None:
        raise ValueError(
            "pack has a threshold_policy but no threshold_signing_keys provided; "
            "pass threshold_signing_keys= to write_attestation_pack"
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, sort_keys=True, indent=2, ensure_ascii=False)
        f.write("\n")
    return body_hash


def read_attestation_pack(path: str) -> Dict[str, Any]:
    """Parse a pack file from disk; returns the raw dict (unverified)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("magic") != PACK_MAGIC:
        raise AttestationVerificationError(
            f"not a stepback attestation pack: magic={data.get('magic')!r}"
        )
    if data.get("format_version") != PACK_FORMAT_VERSION:
        raise AttestationVerificationError(
            f"unsupported pack format_version={data.get('format_version')!r}"
        )
    return data


def verify_attestation_pack(
    path: str,
    *,
    expected_public_key: Optional[str] = None,
    verify_threshold: bool = True,
) -> Dict[str, Any]:
    """Verify the attestation pack at ``path`` and return its parsed body.

    Checks:
      * magic + format_version
      * body_hash matches sha256 of the canonical body (everything
        except ``body_hash`` and unsigned envelope fields)
      * signature verifies under the attestor public key embedded
        in the body
      * if the body contains a ``threshold_policy`` and
        ``verify_threshold`` is True, verifies the ``threshold_signatures``
        envelope field meets the policy's ``m_required`` threshold
      * if ``expected_public_key`` is given, it matches the embedded
        attestor key (so an auditor can pin the attestor)

    Raises :class:`AttestationVerificationError` on any mismatch.
    """
    data = read_attestation_pack(path)
    sig_field = data.get("signature", "")
    body_hash = data.get("body_hash", "")
    if not sig_field.startswith("ed25519:") or not body_hash.startswith("sha256:"):
        raise AttestationVerificationError("missing or malformed signature/body_hash")
    # Unsigned envelope fields are excluded from body-hash computation.
    # ``transparency_log_proof`` and ``threshold_signatures`` are appended
    # after signing so they must also be excluded here.
    _UNSIGNED_FIELDS = {"signature", "body_hash", "transparency_log_proof", "threshold_signatures"}
    body = {k: v for k, v in data.items() if k not in _UNSIGNED_FIELDS}
    recomputed = sha256_hex(canonical_json(body))
    if recomputed != body_hash:
        raise AttestationVerificationError(
            f"body_hash mismatch: pack={body_hash} recomputed={recomputed}"
        )
    pub_field = body.get("attestor_public_key", "")
    if not pub_field.startswith("ed25519:"):
        raise AttestationVerificationError(
            f"unsupported attestor_public_key scheme: {pub_field!r}"
        )
    if expected_public_key is not None and expected_public_key != pub_field:
        raise AttestationVerificationError(
            f"attestor public key mismatch: expected={expected_public_key} "
            f"got={pub_field}"
        )
    pub_hex = pub_field.removeprefix("ed25519:")
    pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex))
    sig = bytes.fromhex(sig_field.removeprefix("ed25519:"))
    try:
        pub.verify(sig, body_hash.encode("ascii"))
    except InvalidSignature as e:
        raise AttestationVerificationError("Ed25519 signature invalid") from e

    # Threshold-signature verification
    threshold_policy_dict = body.get("threshold_policy")
    if threshold_policy_dict is not None and verify_threshold:
        from .threshold_sig import (
            ThresholdPolicy,
            ThresholdSignatureError,
            WitnessSignature,
            verify_threshold_signatures,
        )
        policy = ThresholdPolicy.from_dict(threshold_policy_dict)
        raw_sigs = data.get("threshold_signatures")
        if not raw_sigs:
            raise AttestationVerificationError(
                "pack has threshold_policy but threshold_signatures envelope "
                "field is missing or empty"
            )
        witness_sigs = [WitnessSignature.from_dict(s) for s in raw_sigs]
        try:
            verify_threshold_signatures(body_hash, policy, witness_sigs)
        except ThresholdSignatureError as e:
            raise AttestationVerificationError(
                f"threshold signature verification failed: {e}"
            ) from e

    return data


def parse_substitution_specs_from_pack(entry: Dict[str, Any]) -> List[Substitution]:
    """Re-hydrate the substitutions an entry was built with."""
    return [substitution_from_dict(s) for s in entry.get("substitutions", [])]


__all__ = [
    "PACK_MAGIC",
    "PACK_FORMAT_VERSION",
    "AttestationEntry",
    "AttestationPack",
    "AttestationVerificationError",
    "build_attestation_pack",
    "write_attestation_pack",
    "read_attestation_pack",
    "verify_attestation_pack",
    "parse_substitution_specs_from_pack",
]
