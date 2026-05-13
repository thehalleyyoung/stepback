"""Documentation invariant tests for SECURITY.md (Step 127).

These tests pin the presence and content of the SECURITY.md threat model so
that future edits cannot silently remove required security documentation.
"""
from __future__ import annotations

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).parent.parent
SECURITY_MD = REPO_ROOT / "SECURITY.md"


def _text() -> str:
    return SECURITY_MD.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# File presence
# ---------------------------------------------------------------------------


def test_security_md_exists():
    assert SECURITY_MD.exists(), "SECURITY.md must exist in the repo root"


def test_security_md_non_empty():
    assert len(_text()) > 500, "SECURITY.md appears too short to be meaningful"


# ---------------------------------------------------------------------------
# Required top-level sections
# ---------------------------------------------------------------------------


def test_has_reporting_section():
    text = _text()
    assert re.search(r"(?i)#.*report", text), (
        "SECURITY.md must contain a 'Reporting vulnerabilities' section"
    )


def test_has_threat_model_section():
    text = _text()
    assert re.search(r"(?i)#.*threat model", text), (
        "SECURITY.md must contain a 'Threat model' section"
    )


def test_has_what_proves_subsection():
    text = _text()
    assert re.search(r"(?i)#.*what.*proves?", text), (
        "SECURITY.md must have a 'What the scheme proves' subsection"
    )


def test_has_what_does_not_prove_subsection():
    text = _text()
    assert re.search(r"(?i)#.*does not prove|not.*guarantee", text, re.IGNORECASE), (
        "SECURITY.md must have a 'What the scheme does NOT prove' subsection"
    )


def test_has_key_handling_section():
    text = _text()
    assert re.search(r"(?i)#.*key.*handling|handling.*key", text), (
        "SECURITY.md must contain a 'Key types and handling' section"
    )


def test_has_verifier_guarantees_section():
    text = _text()
    assert re.search(r"(?i)#.*verifier.*guarantee|guarantee.*verifier", text), (
        "SECURITY.md must contain a 'Verifier guarantees' section"
    )


def test_has_disclosure_path():
    text = _text()
    # Must mention a reporting path (security advisories, email, etc.)
    assert (
        "security advisories" in text.lower()
        or "security advisory" in text.lower()
        or "github.com" in text.lower()
        or "email" in text.lower()
    ), "SECURITY.md must document a vulnerability disclosure path"


# ---------------------------------------------------------------------------
# Cryptographic primitives referenced
# ---------------------------------------------------------------------------


def test_mentions_hmac():
    assert "HMAC" in _text(), "SECURITY.md must mention HMAC"


def test_mentions_ed25519():
    assert "Ed25519" in _text(), "SECURITY.md must mention Ed25519"


def test_mentions_trace_verification_error():
    assert "TraceVerificationError" in _text(), (
        "SECURITY.md must reference TraceVerificationError for auditors"
    )


# ---------------------------------------------------------------------------
# Key-handling accuracy (guards against misstatements caught in Step 127 review)
# ---------------------------------------------------------------------------


def test_hmac_key_not_stored_plaintext():
    """HMAC key itself must NOT be described as stored in the trace."""
    text = _text()
    # The file must say the key is NOT embedded / NOT written to disk
    assert re.search(
        r"(?i)(never written|not.{0,30}written|key itself|only.{0,60}hmac_key_id"
        r"|key.*not.*stored|not.*store.*key|out.{0,10}of.{0,10}band)",
        text,
    ), (
        "SECURITY.md must document that the HMAC key is not stored in the .sb file; "
        "only the hmac_key_id fingerprint is stored"
    )


def test_mentions_hmac_key_id():
    assert "hmac_key_id" in _text(), (
        "SECURITY.md must mention hmac_key_id to clarify what is stored in the header"
    )


def test_embedded_public_key_self_asserted():
    """SECURITY.md must warn that the embedded public key is self-asserted."""
    text = _text()
    assert re.search(
        r"(?i)(self.?assert|not.*identity|out.{0,10}of.{0,10}band|pin.*public|"
        r"public.*key.*pin|pinning|compare.*public|trusted.*key)",
        text,
    ), (
        "SECURITY.md must warn that the embedded Ed25519 public key is self-asserted "
        "and does not prove recorder identity unless externally pinned"
    )


# ---------------------------------------------------------------------------
# Attestation pack trust boundary
# ---------------------------------------------------------------------------


def test_mentions_attestation_pack():
    text = _text()
    assert re.search(r"(?i)attestation pack", text), (
        "SECURITY.md must describe the attestation pack trust boundary"
    )


def test_attestation_trust_boundary():
    """Attestation packs should be described as only as trustworthy as the attestor."""
    text = _text()
    assert re.search(
        r"(?i)(attestor|trust.*pack|pack.*trust|trust.*attestation|attestation.*trust)",
        text,
    ), (
        "SECURITY.md must describe the trust relationship for attestation packs"
    )


# ---------------------------------------------------------------------------
# Capability negotiation
# ---------------------------------------------------------------------------


def test_mentions_capability_fail_closed():
    text = _text()
    assert re.search(
        r"(?i)(fail.{0,10}closed|mandatory.*capabilit|capabilit.*mandatory)",
        text,
    ), (
        "SECURITY.md must mention that unknown mandatory capabilities fail closed"
    )


# ---------------------------------------------------------------------------
# DoS bounds
# ---------------------------------------------------------------------------


def test_mentions_dos_bounds():
    text = _text()
    assert re.search(
        r"(?i)(denial.of.service|DoS|MAX_FRAME_BYTES|reader.*limit|reader-limits)",
        text,
    ), "SECURITY.md must reference denial-of-service / reader limits"


# ---------------------------------------------------------------------------
# Supported versions table
# ---------------------------------------------------------------------------


def test_has_supported_versions_section():
    text = _text()
    assert re.search(r"(?i)#.*supported.*version", text), (
        "SECURITY.md must include a 'Supported versions' section"
    )


# ---------------------------------------------------------------------------
# What verify_trace does NOT check
# ---------------------------------------------------------------------------


def test_documents_verify_trace_does_not_guarantee_identity():
    """Document that verify_trace does not check recorder identity."""
    text = _text()
    assert re.search(
        r"(?i)(does not.*identity|identity.*not.*prove|not.*prove.*identity"
        r"|without.*pinning|pinning.*required|NOT.*prove.*trace.*written"
        r"|NOT.*an.*identity.*proof|self.?assert)",
        text,
    ), (
        "SECURITY.md must state that verify_trace does not prove recorder identity"
    )


def test_documents_hmac_key_rewrite_risk():
    """The risk that a known HMAC key allows trace rewriting must be mentioned."""
    text = _text()
    assert re.search(
        r"(?i)(rewrite|rewrit|replace.*header|recomput.*hmac|hmac.*recomput"
        r"|attacker.*hmac|hmac.*attacker|secret|treat.*secret)",
        text,
    ), (
        "SECURITY.md must document that the HMAC key must be treated as a secret "
        "because its exposure (combined with lack of public-key pinning) allows "
        "trace rewriting"
    )
