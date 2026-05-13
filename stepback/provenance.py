"""SLSA and in-toto provenance attestations for traces, benchmark
submissions, and incident replay packs.

This module implements the
`in-toto Attestation Framework v1 <https://github.com/in-toto/attestation/blob/main/spec/v1/statement.md>`_
envelope and the
`SLSA Provenance v1 <https://slsa.dev/provenance/v1>`_ predicate so that
every stepback artifact can be bound to a machine-verifiable supply-chain
statement.

Supported artifact types
------------------------
* **Trace** – a single ``.sb`` recording produced by :func:`stepback.record`.
* **Benchmark submission** – a JSON result document produced by
  ``stepback bench``.
* **Incident replay pack** – an attestation pack (``stepback attest``) that
  bundles per-trace verification and counterfactual replay verdicts.

Wire format
-----------
Statements follow the `DSSE (Dead Simple Signing Envelope)
<https://github.com/secure-systems-lab/dsse/blob/master/protocol.md>`_
protocol:

.. code-block:: json

    {
      "payloadType": "application/vnd.in-toto+json",
      "payload": "<base64(statement_bytes)>",
      "signatures": [{"keyid": "<ed25519:<hex pubkey>>", "sig": "<base64(sig)>"}]
    }

The signed bytes are the DSSE Pre-Authentication Encoding (PAE) of
``payloadType`` and ``payload_bytes`` (not the envelope JSON itself):

.. code-block:: text

    PAE = b"DSSEv1 " + LEN(payloadType_bytes) + b" " + payloadType_bytes
              + b" " + LEN(payload_bytes) + b" " + payload_bytes

The ``payload_bytes`` are the **deterministic canonical JSON** (sorted keys,
no whitespace) of the in-toto ``Statement`` dict.  Verification re-derives
this PAE before checking the Ed25519 signature.

Typical use
-----------
.. code-block:: python

    from stepback.provenance import trace_provenance, sign_provenance
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    stmt = trace_provenance(
        "run-001.sb",
        sha256_digest="abc123...",
        builder_id="https://ci.example.com/builder/v1",
    )
    envelope = sign_provenance(stmt, key)
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

# ------------------------------------------------------------------ constants

INTOTO_STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
SLSA_PREDICATE_TYPE = "https://slsa.dev/provenance/v1"
DSSE_PAYLOAD_TYPE = "application/vnd.in-toto+json"

BUILD_TYPE_TRACE = "https://stepback.ai/provenance/trace/v1"
BUILD_TYPE_BENCHMARK = "https://stepback.ai/provenance/benchmark/v1"
BUILD_TYPE_PACK = "https://stepback.ai/provenance/pack/v1"

BUILDER_ID = "https://github.com/stepback-dev/stepback"

MEDIA_TYPE_TRACE = "application/vnd.stepback.trace"
MEDIA_TYPE_BENCHMARK = "application/json"
MEDIA_TYPE_PACK = "application/vnd.stepback.attestation-pack+json"

# ------------------------------------------------------------------ errors


class ProvenanceVerificationError(Exception):
    """Raised when DSSE envelope verification fails."""

    #: Canonical error code; see :mod:`stepback.errors` for details.
    code: str = "SB602"


# ------------------------------------------------------------------ helpers


def _canonical_json_bytes(obj: Any) -> bytes:
    """Deterministic, compact JSON serialization (sorted keys, ASCII-safe)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _dsse_pae(payload_type: str, payload_bytes: bytes) -> bytes:
    """DSSE Pre-Authentication Encoding.

    Follows the DSSE spec::

        PAE = "DSSEv1 " + LEN(payloadType_bytes) + " " + payloadType_bytes
                + " " + LEN(payload_bytes) + " " + payload_bytes

    All lengths are decimal ASCII byte-counts.
    """
    pt = payload_type.encode("utf-8")
    return (
        b"DSSEv1 "
        + str(len(pt)).encode("ascii") + b" "
        + pt + b" "
        + str(len(payload_bytes)).encode("ascii") + b" "
        + payload_bytes
    )


def _sha256_file(path: str) -> str:
    """Return ``sha256:<hex>`` of the raw bytes of *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    """Return ``sha256:<hex>`` of *data*."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------ statement


def _make_statement(
    subjects: List[Dict[str, Any]],
    predicate: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble a raw in-toto Statement v1 dict (unsigned)."""
    return {
        "_type": INTOTO_STATEMENT_TYPE,
        "subject": subjects,
        "predicateType": SLSA_PREDICATE_TYPE,
        "predicate": predicate,
    }


def _slsa_predicate(
    build_type: str,
    external_parameters: Dict[str, Any],
    internal_parameters: Dict[str, Any],
    resolved_dependencies: List[Dict[str, Any]],
    builder_id: str,
    invocation_id: str,
    started_on: Optional[str],
    finished_on: Optional[str],
    byproducts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Construct a SLSA Provenance v1 predicate dict."""
    predicate: Dict[str, Any] = {
        "buildDefinition": {
            "buildType": build_type,
            "externalParameters": external_parameters,
            "internalParameters": internal_parameters,
            "resolvedDependencies": resolved_dependencies,
        },
        "runDetails": {
            "builder": {"id": builder_id},
            "metadata": {
                "invocationId": invocation_id,
                "startedOn": started_on,
                "finishedOn": finished_on,
            },
            "byproducts": byproducts or [],
        },
    }
    return predicate


def _resource_descriptor(
    name: str,
    sha256_digest: str,
    *,
    uri: Optional[str] = None,
    media_type: Optional[str] = None,
    annotations: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a SLSA ResourceDescriptor dict."""
    rd: Dict[str, Any] = {
        "name": name,
        "digest": {"sha256": sha256_digest.removeprefix("sha256:")},
    }
    if uri:
        rd["uri"] = uri
    if media_type:
        rd["mediaType"] = media_type
    if annotations:
        rd["annotations"] = annotations
    return rd


# ------------------------------------------------------------------ factories


def trace_provenance(
    artifact_name: str,
    sha256_digest: str,
    *,
    trace_chain_hash: Optional[str] = None,
    merkle_root: Optional[str] = None,
    recorder_public_key: Optional[str] = None,
    recorder_version: Optional[str] = None,
    canonicalisation_version: Optional[str] = None,
    step_count: Optional[int] = None,
    started_on: Optional[str] = None,
    finished_on: Optional[str] = None,
    builder_id: str = BUILDER_ID,
) -> Dict[str, Any]:
    """Return an in-toto Statement for a single ``.sb`` trace.

    Parameters
    ----------
    artifact_name:
        Logical name for the trace artifact (e.g. ``"run-001.sb"``).
        May be a bare filename or a relative path.
    sha256_digest:
        ``"sha256:<hex>"`` of the *raw bytes* of the ``.sb`` file.  Callers
        may produce this with :func:`sha256_of_file`.
    trace_chain_hash:
        The HMAC chain root hash produced by :func:`stepback.branch_io.trace_chain_hash`;
        stored as a byproduct annotation.
    merkle_root:
        The Merkle frame summary root hash; stored as a byproduct annotation.
    recorder_public_key:
        ``"ed25519:<hex>"`` fingerprint of the recorder key.
    recorder_version, canonicalisation_version:
        Version strings recorded in the trace header.
    step_count:
        Number of steps in the trace.
    started_on, finished_on:
        ISO-8601 UTC timestamps for the build metadata.
    builder_id:
        SLSA builder URI; defaults to the stepback GitHub repo.
    """
    subjects = [
        _resource_descriptor(
            artifact_name,
            sha256_digest,
            media_type=MEDIA_TYPE_TRACE,
        )
    ]
    external_params: Dict[str, Any] = {"source": artifact_name}
    internal_params: Dict[str, Any] = {}
    if recorder_version is not None:
        internal_params["recorder_version"] = recorder_version
    if canonicalisation_version is not None:
        internal_params["canonicalisation_version"] = canonicalisation_version

    byproducts: List[Dict[str, Any]] = []
    if trace_chain_hash:
        byproducts.append({"name": "trace_chain_hash", "value": trace_chain_hash})
    if merkle_root:
        byproducts.append({"name": "merkle_root", "value": merkle_root})
    if recorder_public_key:
        byproducts.append({"name": "recorder_public_key", "value": recorder_public_key})
    if step_count is not None:
        byproducts.append({"name": "step_count", "value": step_count})

    invocation_id = trace_chain_hash or sha256_digest
    predicate = _slsa_predicate(
        BUILD_TYPE_TRACE,
        external_params,
        internal_params,
        resolved_dependencies=[],
        builder_id=builder_id,
        invocation_id=invocation_id,
        started_on=started_on,
        finished_on=finished_on,
        byproducts=byproducts,
    )
    return _make_statement(subjects, predicate)


def benchmark_provenance(
    artifact_name: str,
    content_bytes: bytes,
    *,
    corpus_id: Optional[str] = None,
    trace_count: Optional[int] = None,
    stepback_version: Optional[str] = None,
    started_on: Optional[str] = None,
    finished_on: Optional[str] = None,
    builder_id: str = BUILDER_ID,
) -> Dict[str, Any]:
    """Return an in-toto Statement for a benchmark submission JSON.

    The subject digest is the SHA-256 of *content_bytes* (the exact bytes
    of the result document), so verifiers can confirm the provenance
    matches the submitted file byte-for-byte.

    Parameters
    ----------
    artifact_name:
        Logical name for the result file (e.g. ``"bench-replay-caching.json"``).
    content_bytes:
        Raw bytes of the benchmark JSON document; the subject digest is
        computed from these bytes.
    corpus_id:
        Identifier for the benchmark corpus (e.g. ``"swe-bench-verified"``).
    trace_count:
        Number of traces evaluated.
    stepback_version:
        Version string of the stepback package used.
    started_on, finished_on:
        ISO-8601 UTC timestamps.
    builder_id:
        SLSA builder URI.
    """
    sha256_digest = _sha256_bytes(content_bytes)
    subjects = [
        _resource_descriptor(
            artifact_name,
            sha256_digest,
            media_type=MEDIA_TYPE_BENCHMARK,
        )
    ]
    external_params: Dict[str, Any] = {"source": artifact_name}
    if corpus_id:
        external_params["corpus_id"] = corpus_id
    internal_params: Dict[str, Any] = {}
    if stepback_version is not None:
        internal_params["stepback_version"] = stepback_version

    byproducts: List[Dict[str, Any]] = []
    if trace_count is not None:
        byproducts.append({"name": "trace_count", "value": trace_count})

    predicate = _slsa_predicate(
        BUILD_TYPE_BENCHMARK,
        external_params,
        internal_params,
        resolved_dependencies=[],
        builder_id=builder_id,
        invocation_id=sha256_digest,
        started_on=started_on,
        finished_on=finished_on,
        byproducts=byproducts,
    )
    return _make_statement(subjects, predicate)


def pack_provenance(
    artifact_name: str,
    sha256_digest: str,
    *,
    trace_descriptors: Optional[List[Dict[str, Any]]] = None,
    policy_version_pin: Optional[str] = None,
    attestor_public_key: Optional[str] = None,
    trace_count: Optional[int] = None,
    verified_ok: Optional[int] = None,
    divergent_traces: Optional[int] = None,
    started_on: Optional[str] = None,
    finished_on: Optional[str] = None,
    builder_id: str = BUILDER_ID,
) -> Dict[str, Any]:
    """Return an in-toto Statement for an incident replay attestation pack.

    The pack itself is the SLSA *subject*.  Each constituent trace is listed
    as a ``resolvedDependency`` so verifiers can confirm the pack covers the
    expected corpus.

    Parameters
    ----------
    artifact_name:
        Logical name for the ``.pack`` file (e.g. ``"Q1-2026.pack"``).
    sha256_digest:
        ``"sha256:<hex>"`` of the raw bytes of the ``.pack`` file.
    trace_descriptors:
        Optional list of :func:`_resource_descriptor`-shaped dicts for each
        trace that contributed to the pack.  Each entry should have at least
        ``name`` and ``digest``.
    policy_version_pin:
        The policy version label embedded in the pack (if any).
    attestor_public_key:
        ``"ed25519:<hex>"`` of the attestor key that signed the pack.
    trace_count, verified_ok, divergent_traces:
        Summary statistics from the pack's ``summary`` block.
    started_on, finished_on:
        ISO-8601 UTC timestamps.
    builder_id:
        SLSA builder URI.
    """
    subjects = [
        _resource_descriptor(
            artifact_name,
            sha256_digest,
            media_type=MEDIA_TYPE_PACK,
        )
    ]
    external_params: Dict[str, Any] = {"source": artifact_name}
    if policy_version_pin:
        external_params["policy_version_pin"] = policy_version_pin

    internal_params: Dict[str, Any] = {}
    if attestor_public_key:
        internal_params["attestor_public_key"] = attestor_public_key

    resolved_deps: List[Dict[str, Any]] = list(trace_descriptors or [])

    byproducts: List[Dict[str, Any]] = []
    if trace_count is not None:
        byproducts.append({"name": "trace_count", "value": trace_count})
    if verified_ok is not None:
        byproducts.append({"name": "verified_ok", "value": verified_ok})
    if divergent_traces is not None:
        byproducts.append({"name": "divergent_traces", "value": divergent_traces})

    predicate = _slsa_predicate(
        BUILD_TYPE_PACK,
        external_params,
        internal_params,
        resolved_dependencies=resolved_deps,
        builder_id=builder_id,
        invocation_id=sha256_digest,
        started_on=started_on,
        finished_on=finished_on,
        byproducts=byproducts,
    )
    return _make_statement(subjects, predicate)


# ------------------------------------------------------------------ DSSE


def sign_provenance(
    statement: Dict[str, Any],
    signing_key: Ed25519PrivateKey,
    *,
    keyid: Optional[str] = None,
) -> Dict[str, Any]:
    """Sign *statement* and return a DSSE envelope dict.

    The statement is serialized to deterministic canonical JSON (sorted keys,
    no whitespace).  The Ed25519 signature covers the DSSE PAE of
    ``payloadType`` and ``payload_bytes``.

    The returned dict has shape::

        {
          "payloadType": "application/vnd.in-toto+json",
          "payload": "<base64(statement_bytes)>",
          "signatures": [{"keyid": "ed25519:<pubhex>", "sig": "<base64(sig)>"}]
        }

    Parameters
    ----------
    statement:
        A dict produced by :func:`trace_provenance`, :func:`benchmark_provenance`,
        or :func:`pack_provenance`.
    signing_key:
        Ed25519 private key.
    keyid:
        Override the ``keyid`` field in the signature object.  Defaults to
        ``"ed25519:<public_key_hex>"``.  This field is informational only;
        :func:`verify_provenance_signature` requires the expected public key
        to be passed explicitly.
    """
    payload_bytes = _canonical_json_bytes(statement)
    pae = _dsse_pae(DSSE_PAYLOAD_TYPE, payload_bytes)
    raw_sig = signing_key.sign(pae)
    pub_hex = signing_key.public_key().public_bytes_raw().hex()
    sig_b64 = base64.b64encode(raw_sig).decode("ascii")
    payload_b64 = base64.b64encode(payload_bytes).decode("ascii")
    effective_keyid = keyid or f"ed25519:{pub_hex}"
    return {
        "payloadType": DSSE_PAYLOAD_TYPE,
        "payload": payload_b64,
        "signatures": [{"keyid": effective_keyid, "sig": sig_b64}],
    }


def verify_provenance_signature(
    envelope: Dict[str, Any],
    expected_public_key: Ed25519PublicKey,
) -> Dict[str, Any]:
    """Verify the DSSE *envelope* and return the parsed statement dict.

    Parameters
    ----------
    envelope:
        Dict produced by :func:`sign_provenance` or loaded from a ``.intoto.jsonl``
        file.
    expected_public_key:
        The Ed25519 public key the caller trusts.  The ``keyid`` field is
        treated as a hint only; the actual cryptographic check is performed
        against *expected_public_key*.

    Returns
    -------
    The parsed in-toto statement dict (the decoded ``payload`` field).

    Raises
    ------
    :class:`ProvenanceVerificationError`
        If the payload type is wrong, no signatures are present, or every
        signature fails to verify.
    """
    pt = envelope.get("payloadType", "")
    if pt != DSSE_PAYLOAD_TYPE:
        raise ProvenanceVerificationError(
            f"unexpected payloadType: {pt!r}; expected {DSSE_PAYLOAD_TYPE!r}"
        )
    payload_b64 = envelope.get("payload", "")
    try:
        payload_bytes = base64.b64decode(payload_b64)
    except Exception as exc:
        raise ProvenanceVerificationError(
            f"payload is not valid base64: {exc}"
        ) from exc

    pae = _dsse_pae(DSSE_PAYLOAD_TYPE, payload_bytes)
    sigs = envelope.get("signatures", [])
    if not sigs:
        raise ProvenanceVerificationError("envelope has no signatures")

    last_exc: Optional[Exception] = None
    for sig_entry in sigs:
        sig_b64 = sig_entry.get("sig", "")
        try:
            raw_sig = base64.b64decode(sig_b64)
        except Exception as exc:
            last_exc = exc
            continue
        try:
            expected_public_key.verify(raw_sig, pae)
            # At least one signature verified — return parsed payload.
            return json.loads(payload_bytes)
        except InvalidSignature as exc:
            last_exc = exc

    raise ProvenanceVerificationError(
        f"no valid signature found for the expected public key: {last_exc}"
    )


def sha256_of_file(path: str) -> str:
    """Return ``sha256:<hex>`` of the raw bytes of the file at *path*."""
    return _sha256_file(path)


def sha256_of_bytes(data: bytes) -> str:
    """Return ``sha256:<hex>`` of *data*."""
    return _sha256_bytes(data)


# ------------------------------------------------------------------ __all__

__all__ = [
    # constants
    "INTOTO_STATEMENT_TYPE",
    "SLSA_PREDICATE_TYPE",
    "DSSE_PAYLOAD_TYPE",
    "BUILD_TYPE_TRACE",
    "BUILD_TYPE_BENCHMARK",
    "BUILD_TYPE_PACK",
    "BUILDER_ID",
    "MEDIA_TYPE_TRACE",
    "MEDIA_TYPE_BENCHMARK",
    "MEDIA_TYPE_PACK",
    # errors
    "ProvenanceVerificationError",
    # factories
    "trace_provenance",
    "benchmark_provenance",
    "pack_provenance",
    # DSSE
    "sign_provenance",
    "verify_provenance_signature",
    # utilities
    "sha256_of_file",
    "sha256_of_bytes",
]
