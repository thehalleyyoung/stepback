"""Case study: redacted trace publication.

This case study models the workflow for safely publishing AI agent traces
externally — for example, sharing production traces with an academic
collaborator, regulatory body, or open-source benchmark.

Raw ``.sb`` trace files routinely contain:

* **PII** — customer emails, phone numbers, names, account IDs.
* **Credentials** — bearer tokens, API keys, internal service URLs.
* **Business-sensitive data** — pricing rules, internal tool outputs, model
  configurations.

Before any external publication, the compliance team must:

1. Scan the trace for sensitive data (without modifying it yet).
2. Apply a :class:`~stepback.redact.RedactionPolicy` that hashes or masks
   matches while preserving the trace's structural replay properties.
3. Verify the redacted trace — it must pass HMAC-chain and signature
   verification under its new recorder key.
4. Attach a :class:`~stepback.redact.RedactionAttestation` signed by the
   compliance team to prove the redaction was performed correctly and that
   the original was not altered before redaction.

Preservation of replay utility
--------------------------------
The ``"hash"`` replacement strategy is key: matched PII is replaced with
``<REDACTED:email:ab12cd34>`` tokens derived from
``HMAC-SHA256(policy.salt, "email" || matched_text)``.  Because the same
input always maps to the same token, the cache structure of the trace is
preserved: two LLM calls that received the same customer email will still
have the same hashed token in the redacted trace, so the replay engine can
still distinguish "same-input" from "different-input" steps.

Example::

    from stepback.case_studies.redacted_publication import run_redacted_publication
    result = run_redacted_publication()
    print(result.summary_line())
    # → "redacted_publication: 8 steps | matches=4 attestation_verified=True | no_raw_pii=True"
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ..recorder import RecorderKey, record
from ..redact import (
    STANDARD_POLICY,
    RedactionManifest,
    RedactionPolicy,
    ingest_trace_file,
    redact_trace_file,
    scan_trace_file,
    sign_redaction_attestation,
    verify_redaction_attestation,
)
from ..trace_reader import verify_trace


# ---------------------------------------------------------------------------
# Deterministic fake executors with synthetic PII in outputs
# ---------------------------------------------------------------------------

# These inputs/outputs are synthetic and contain realistic-looking but
# entirely fake PII that the STANDARD_POLICY detectors will flag.

_SYNTHETIC_CUSTOMERS = [
    {
        "email": "alice.smith@example.com",
        "phone": "+1-555-867-5309",
        "account_id": "acct-00123",
    },
    {
        "email": "bob.jones@company.org",
        "phone": "+44 20 7946 0958",
        "account_id": "acct-00456",
    },
    {
        "email": "carol.white@testcorp.net",
        "phone": "+1 (800) 555-0100",
        "account_id": "acct-00789",
    },
    {
        "email": "dave.black@demo.io",
        "phone": "+49 30 12345678",
        "account_id": "acct-01011",
    },
]


def _llm(model: str, messages: list) -> dict:
    """Fake LLM that echoes PII from the latest user message into its reply."""
    user_content = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_content = m.get("content", "")
            break
    digest = hashlib.sha256((model + user_content).encode()).hexdigest()[:12]
    # Embed a fake customer reference in the reply so redaction fires on outputs
    cust = _SYNTHETIC_CUSTOMERS[len(digest) % len(_SYNTHETIC_CUSTOMERS)]
    text = f"Processed request for {cust['email']} (acc {cust['account_id']}). ref={digest}"
    return {
        "id": f"chatcmpl-{digest}",
        "model": model,
        "choices": [
            {"index": 0, "finish_reason": "stop",
             "message": {"role": "assistant", "content": text}},
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": len(text), "total_tokens": 20 + len(text)},
    }


def _tool(name: str, args: dict) -> Any:
    """Fake tool that returns a synthetic customer record with PII."""
    i = hash(name + repr(sorted(args.items()))) % len(_SYNTHETIC_CUSTOMERS)
    cust = _SYNTHETIC_CUSTOMERS[i]
    if name == "lookup_customer":
        return {
            "customer_id": cust["account_id"],
            "email": cust["email"],
            "phone": cust["phone"],
            "status": "active",
        }
    return {"result": "ok"}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RedactedPublicationResult:
    """Result of the PII-redaction pipeline.

    Attributes
    ----------
    original_path : str
        Path to the original (unredacted) trace.
    redacted_path : str
        Path to the redacted trace (``<temp>`` if in a temp directory).
    n_steps : int
        Number of steps in the trace.
    scan_matches : int
        Total number of PII matches found by the pre-redaction scan.
    redaction_matches : int
        Total matches recorded in the :class:`~stepback.redact.RedactionManifest`.
    redacted_trace_verified : bool
        Whether the redacted trace passes HMAC-chain + signature verification.
    attestation_verified : bool
        Whether the compliance attestation signature verified correctly.
    no_raw_pii_in_redacted : bool
        Whether none of the known synthetic PII strings appear verbatim in the
        redacted trace file bytes.
    known_emails_redacted : int
        How many of the known synthetic email addresses were replaced.
    """

    original_path: str
    redacted_path: str
    n_steps: int
    scan_matches: int
    redaction_matches: int
    redacted_trace_verified: bool
    attestation_verified: bool
    no_raw_pii_in_redacted: bool
    known_emails_redacted: int

    def summary_line(self) -> str:
        """One-line human-readable summary."""
        return (
            f"redacted_publication: {self.n_steps} steps | "
            f"scan_matches={self.scan_matches} redacted={self.redaction_matches} | "
            f"verified={self.redacted_trace_verified} "
            f"attest={self.attestation_verified} "
            f"no_raw_pii={self.no_raw_pii_in_redacted}"
        )

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(
            {
                "original_path": self.original_path,
                "redacted_path": self.redacted_path,
                "n_steps": self.n_steps,
                "scan_matches": self.scan_matches,
                "redaction_matches": self.redaction_matches,
                "redacted_trace_verified": self.redacted_trace_verified,
                "attestation_verified": self.attestation_verified,
                "no_raw_pii_in_redacted": self.no_raw_pii_in_redacted,
                "known_emails_redacted": self.known_emails_redacted,
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Trace builder
# ---------------------------------------------------------------------------

def _build_pii_trace(path: str, key: RecorderKey) -> int:
    """Record an 8-step trace containing synthetic PII; return step count."""
    customers = _SYNTHETIC_CUSTOMERS
    n_steps = 0
    with record(path, key=key) as rec:
        for cust in customers:
            convo = [
                {"role": "system", "content": "customer-service-agent v1"},
                {
                    "role": "user",
                    "content": (
                        f"Help customer {cust['email']} "
                        f"(phone: {cust['phone']}) with account {cust['account_id']}."
                    ),
                },
            ]
            # Tool call: returns PII in outputs
            rec.tool_call(
                "lookup_customer",
                {"account_id": cust["account_id"]},
                executor=_tool,
            )
            n_steps += 1
            # LLM call: embeds PII in the reply text
            rec.llm_call("gpt-4o-2024-11-20", convo, executor=_llm)
            n_steps += 1
    return n_steps


# ---------------------------------------------------------------------------
# PII check helper
# ---------------------------------------------------------------------------

def _check_no_raw_pii(path: str) -> tuple:
    """Return (no_raw_pii: bool, emails_still_present: int).

    Reads the raw file bytes and searches for each known synthetic email.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    found = 0
    for cust in _SYNTHETIC_CUSTOMERS:
        if cust["email"].encode() in raw:
            found += 1
    return (found == 0), found


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_redacted_publication(
    output_dir: Optional[str] = None,
) -> RedactedPublicationResult:
    """Run the full PII-redaction pipeline on a synthetic trace.

    Records an 8-step customer-service trace that contains synthetic email
    addresses and phone numbers, then scans, redacts, verifies, and attests
    the redacted output.

    Parameters
    ----------
    output_dir : str, optional
        Directory for output files.  A ``TemporaryDirectory`` is used when
        ``None``; it is cleaned up before the function returns.
    """
    cleanup = output_dir is None
    tmpdir = tempfile.TemporaryDirectory(prefix="sb-redact-") if cleanup else None
    base_dir = tmpdir.name if tmpdir else output_dir

    original_key = RecorderKey.fresh()
    original_path = os.path.join(base_dir, "original.sb")
    redacted_path = os.path.join(base_dir, "redacted.sb")

    # Compliance team's attestor key (production: from HSM/secrets manager)
    attestor_key: Ed25519PrivateKey = Ed25519PrivateKey.generate()

    try:
        # 1. Record the original trace
        n_steps = _build_pii_trace(original_path, original_key)

        # 2. Scan without modifying
        scan_report = scan_trace_file(
            original_path,
            in_hmac_key=original_key.hmac_key,
            policy=STANDARD_POLICY,
        )
        scan_matches = scan_report.n_findings

        # 3. Redact — pass an explicit out_key so we can verify later
        out_key = RecorderKey.fresh()
        manifest: RedactionManifest = redact_trace_file(
            original_path,
            redacted_path,
            in_hmac_key=original_key.hmac_key,
            policy=STANDARD_POLICY,
            out_key=out_key,
        )
        redaction_matches = manifest.n_redactions

        # 4. Verify the redacted trace's HMAC chain + Ed25519 signatures
        try:
            verify_trace(redacted_path, out_key.hmac_key)
            redacted_verified = True
        except Exception:  # noqa: BLE001
            redacted_verified = False

        # 5. Sign a compliance attestation
        attestation = sign_redaction_attestation(
            original_path,
            redacted_path,
            manifest,
            STANDARD_POLICY,
            attestor_key,
        )

        # 6. Verify the attestation (as an auditor would)
        attestor_pub_str = "ed25519:" + attestor_key.public_key().public_bytes_raw().hex()
        try:
            verify_redaction_attestation(
                attestation,
                expected_public_key=attestor_pub_str,
            )
            attestation_verified = True
        except Exception:  # noqa: BLE001
            attestation_verified = False

        # 7. Check that no raw PII appears in the redacted file
        no_raw_pii, emails_still_present = _check_no_raw_pii(redacted_path)
        known_emails_redacted = len(_SYNTHETIC_CUSTOMERS) - emails_still_present

        return RedactedPublicationResult(
            original_path=original_path if not cleanup else "<temp>",
            redacted_path=redacted_path if not cleanup else "<temp>",
            n_steps=n_steps,
            scan_matches=scan_matches,
            redaction_matches=redaction_matches,
            redacted_trace_verified=redacted_verified,
            attestation_verified=attestation_verified,
            no_raw_pii_in_redacted=no_raw_pii,
            known_emails_redacted=known_emails_redacted,
        )
    finally:
        if tmpdir:
            tmpdir.cleanup()
