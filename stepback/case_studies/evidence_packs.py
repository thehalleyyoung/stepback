"""Case study: regulator/auditor evidence packs.

This case study models the workflow for building legally defensible evidence
packs that a regulated AI system must produce for auditor review or regulatory
submission — for example, a financial-services firm demonstrating that its
AI-assisted lending decisions comply with explainability and fairness
requirements.

Scenario
--------
An AI agent evaluated five loan applications in a single trace.  The
compliance team needs to:

1. Record the trace with HMAC + Ed25519 chain signing.
2. Build an attestation pack linking the trace to the agent version and policy
   that was in effect at decision time.
3. Sign the pack with a dedicated attestor key.
4. Write the pack to a ``.sbpack`` file.
5. Verify the pack (as an auditor would) — checking the body signature,
   Merkle summary, and per-entry integrity.

The result carries the attestation pack path, verification status, step
count, and pack summary for inclusion in a compliance report.

Why attestation packs
---------------------
Raw ``.sb`` files contain per-frame HMAC and Ed25519 signatures that allow
any single step to be verified in isolation.  The attestation pack adds an
**outer signature** by an independent attestor (typically the compliance team,
not the agent runtime), a **Merkle summary** over all entries so the pack's
total-count and step-hashes can be checked without reading every frame, and
an optional **policy version pin** that binds the pack to the exact policy
document that was in effect.  This three-layer structure (frame → trace →
pack) is what regulators typically require for a chain-of-custody evidence
exhibit.

Example::

    from stepback.case_studies.evidence_packs import run_evidence_pack
    result = run_evidence_pack()
    print(result.summary_line())
    # → "evidence_pack: 5 steps | pack_verified=True | entries=1 signed=True"
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ..attestation import (
    AttestationPack,
    build_attestation_pack,
    verify_attestation_pack,
    write_attestation_pack,
)
from ..recorder import RecorderKey, record


# ---------------------------------------------------------------------------
# Deterministic fake LLM and tool
# ---------------------------------------------------------------------------

def _llm(model: str, messages: list) -> dict:
    blob = model + "|" + "\n".join(
        f"{m.get('role', '')}:{m.get('content', '')}" for m in messages
    )
    digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
    text = f"decision-{digest}"
    return {
        "id": f"chatcmpl-{digest}",
        "model": model,
        "choices": [
            {"index": 0, "finish_reason": "stop",
             "message": {"role": "assistant", "content": text}},
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": len(text), "total_tokens": 15 + len(text)},
    }


def _tool(name: str, args: dict) -> Any:
    blob = name + "|" + repr(sorted(args.items()))
    digest = hashlib.sha256(blob.encode()).hexdigest()[:12]
    if name == "score_application":
        # Deterministic credit score from application id
        app_id = args.get("application_id", "")
        score = 600 + (int(hashlib.sha256(app_id.encode()).hexdigest()[:4], 16) % 250)
        return {"score": score, "band": "prime" if score >= 700 else "subprime"}
    if name == "policy_check":
        return {"approved": True, "policy_version": "v2.3.1", "reasons": []}
    return {"tool": name, "result": digest}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvidencePackResult:
    """Result of building and verifying an auditor evidence pack.

    Attributes
    ----------
    trace_path : str
        Path to the recorded agent trace (``.sb`` file).
    pack_path : str
        Path to the attestation pack (``.sbpack`` file).
    n_steps : int
        Number of steps in the recorded trace.
    n_entries : int
        Number of trace entries in the attestation pack (always 1 in this
        case study — a single trace per pack is the simplest structure).
    policy_version_pin : str
        The policy document version pinned into the pack body.
    pack_verified : bool
        Whether :func:`~stepback.attestation.verify_attestation_pack`
        succeeded (confirms body signature + Merkle root integrity).
    pack_signature_valid : bool
        Whether the Ed25519 body signature verified correctly (same as
        ``pack_verified`` when no other verification steps fail).
    attestor_public_key : str
        Hex fingerprint of the attestor Ed25519 public key embedded in the
        pack.  An auditor would compare this against the organization's
        published key registry.
    produced_at : str
        ISO 8601 UTC timestamp embedded in the pack when it was built.
    """

    trace_path: str
    pack_path: str
    n_steps: int
    n_entries: int
    policy_version_pin: str
    pack_verified: bool
    pack_signature_valid: bool
    attestor_public_key: str
    produced_at: str

    def summary_line(self) -> str:
        """One-line human-readable summary."""
        return (
            f"evidence_pack: {self.n_steps} steps | "
            f"pack_verified={self.pack_verified} | "
            f"entries={self.n_entries} signed={self.pack_signature_valid} | "
            f"policy={self.policy_version_pin}"
        )

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(
            {
                "trace_path": self.trace_path,
                "pack_path": self.pack_path,
                "n_steps": self.n_steps,
                "n_entries": self.n_entries,
                "policy_version_pin": self.policy_version_pin,
                "pack_verified": self.pack_verified,
                "pack_signature_valid": self.pack_signature_valid,
                "attestor_public_key": self.attestor_public_key,
                "produced_at": self.produced_at,
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Trace builder
# ---------------------------------------------------------------------------

def _build_lending_trace(path: str, key: RecorderKey) -> int:
    """Record a 5-application lending decision trace; return step count."""
    applications = [
        {"application_id": f"app-{i:04d}", "amount": 10_000 + i * 5_000}
        for i in range(1, 6)
    ]
    n_steps = 0
    with record(path, key=key) as rec:
        for app in applications:
            convo = [
                {"role": "system", "content": "Lending decision agent v1.2"},
                {
                    "role": "user",
                    "content": (
                        f"Evaluate loan application {app['application_id']} "
                        f"for amount ${app['amount']:,}."
                    ),
                },
            ]
            rec.tool_call("score_application", app, executor=_tool)
            n_steps += 1
            rec.tool_call("policy_check", {"application_id": app["application_id"]}, executor=_tool)
            n_steps += 1
            rec.llm_call("gpt-4o-2024-11-20", convo, executor=_llm)
            n_steps += 1
    return n_steps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_evidence_pack(
    policy_version: str = "v2.3.1",
    output_dir: Optional[str] = None,
) -> EvidencePackResult:
    """Build and verify a signed attestation pack for a lending-decision trace.

    Records a 5-application lending evaluation trace (15 steps), then builds,
    signs, writes, and verifies an attestation pack suitable for regulatory
    submission.

    Parameters
    ----------
    policy_version : str
        Policy document version to pin into the attestation pack.
    output_dir : str, optional
        Directory for output files.  A ``TemporaryDirectory`` is used when
        ``None``; it is cleaned up before the function returns.
    """
    cleanup = output_dir is None
    tmpdir = tempfile.TemporaryDirectory(prefix="sb-evidence-") if cleanup else None
    base_dir = tmpdir.name if tmpdir else output_dir

    trace_key = RecorderKey.fresh()
    trace_path = os.path.join(base_dir, "lending_decisions.sb")
    pack_path = os.path.join(base_dir, "lending_decisions.sbpack")

    # Generate a dedicated attestor key (in production this comes from a
    # hardware security module or secrets manager, not generate()).
    attestor_key: Ed25519PrivateKey = Ed25519PrivateKey.generate()

    try:
        # 1. Record the trace
        n_steps = _build_lending_trace(trace_path, trace_key)

        # 2. Build the attestation pack (uses the trace's HMAC key for
        #    per-entry integrity; the attestor key signs the outer body).
        pack: AttestationPack = build_attestation_pack(
            [(trace_path, trace_key.hmac_key)],
            attestor_signing_key=attestor_key,
            policy_version_pin=policy_version,
        )

        # 3. Write the signed pack to disk.
        write_attestation_pack(pack, pack_path, signing_key=attestor_key)

        # 4. Verify (as an auditor would).
        verified_pack = verify_attestation_pack(pack_path)
        pack_verified = verified_pack is not None

        return EvidencePackResult(
            trace_path=trace_path if not cleanup else "<temp>",
            pack_path=pack_path if not cleanup else "<temp>",
            n_steps=n_steps,
            n_entries=len(pack.entries),
            policy_version_pin=policy_version,
            pack_verified=pack_verified,
            pack_signature_valid=pack_verified,  # verify_attestation_pack checks sig
            attestor_public_key=pack.attestor_public_key,
            produced_at=pack.produced_at,
        )
    finally:
        if tmpdir:
            tmpdir.cleanup()
