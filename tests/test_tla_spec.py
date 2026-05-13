"""Tests for the TLA+ formal specification of the SB-Trace HMAC chain.

These tests verify that:
  1. The TLA+ spec file exists and is structurally complete.
  2. The TLC configuration references the same invariant names defined in the spec.
  3. The conformance dashboard and standards proposal documents exist and are
     structurally sound.
  4. (Optional) When TLC is available, the spec model-checks without violations.

All tests are offline; no network calls are made.
"""
from __future__ import annotations

import os
import re
import subprocess
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
TLA_DIR = REPO_ROOT / "proofs" / "tla"
TLA_FILE = TLA_DIR / "SBHMACChain.tla"
CFG_FILE = TLA_DIR / "SBHMACChain.cfg"
README_FILE = TLA_DIR / "README.md"

DOCS_DIR = REPO_ROOT / "docs"
CONFORMANCE_DASHBOARD = DOCS_DIR / "conformance-dashboard.md"
LF_PROPOSAL = DOCS_DIR / "lf-proposal.md"
CNCF_DRAFT = DOCS_DIR / "cncf-sandbox-draft.md"


# ---------------------------------------------------------------------------
# TLA+ spec structural tests
# ---------------------------------------------------------------------------

class TestTLASpecExists:
    def test_tla_file_exists(self):
        assert TLA_FILE.exists(), f"TLA+ spec not found: {TLA_FILE}"

    def test_cfg_file_exists(self):
        assert CFG_FILE.exists(), f"TLC config not found: {CFG_FILE}"

    def test_readme_exists(self):
        assert README_FILE.exists(), f"TLA+ README not found: {README_FILE}"


class TestTLASpecStructure:
    """The TLA+ spec must have the expected structural elements."""

    @pytest.fixture(autouse=True)
    def tla_content(self):
        self._text = TLA_FILE.read_text(encoding="utf-8")

    def test_module_declaration(self):
        assert "MODULE SBHMACChain" in self._text

    def test_extends_integers_sequences(self):
        assert "EXTENDS" in self._text
        assert "Integers" in self._text
        assert "Sequences" in self._text

    def test_constants_declared(self):
        assert "CONSTANTS" in self._text
        assert "MaxFrames" in self._text
        assert "Bodies" in self._text
        assert "HMACKey" in self._text

    def test_variables_declared(self):
        assert "VARIABLES" in self._text
        assert "writer_log" in self._text
        assert "disk_log" in self._text
        assert "tampered" in self._text

    def test_init_and_next_defined(self):
        assert "Init ==" in self._text
        assert "Next ==" in self._text

    def test_spec_formula_defined(self):
        assert "Spec ==" in self._text

    def test_zero_hmac_defined(self):
        assert "ZeroHMAC" in self._text

    def test_abstract_hmac_defined(self):
        assert "AbstractHMAC" in self._text

    def test_log_valid_defined(self):
        assert "LogValid" in self._text

    def test_integrity_invariant_defined(self):
        assert "IntegrityInvariant ==" in self._text

    def test_tamper_evidence_invariant_defined(self):
        assert "TamperEvidenceInvariant ==" in self._text

    def test_writer_coherence_invariant_defined(self):
        assert "WriterCoherenceInvariant ==" in self._text

    def test_append_frame_action_defined(self):
        assert "AppendFrame" in self._text

    def test_tamper_body_action_defined(self):
        assert "TamperBodyOnly" in self._text

    def test_no_placeholder_sorry(self):
        """Ensure no placeholder markers are present (analogue of Lean 'sorry')."""
        for marker in ("ASSUME FALSE", "TODO", "FIXME", "XXX", "TBD"):
            assert marker not in self._text, (
                f"Placeholder marker {marker!r} found in TLA+ spec"
            )

    def test_module_terminates_with_equals(self):
        """TLA+ modules must end with a line of '=' characters."""
        lines = self._text.splitlines()
        terminal_lines = [l for l in lines if re.match(r"^={4,}\s*$", l)]
        assert len(terminal_lines) >= 1, (
            "TLA+ module must end with a '===...===' delimiter"
        )


class TestTLAConfig:
    """The TLC configuration must reference the same invariants as the spec."""

    @pytest.fixture(autouse=True)
    def cfg_content(self):
        self._cfg = CFG_FILE.read_text(encoding="utf-8")
        self._tla = TLA_FILE.read_text(encoding="utf-8")

    def _cfg_invariants(self) -> list[str]:
        return re.findall(r"^INVARIANT\s+(\w+)", self._cfg, re.MULTILINE)

    def test_specification_references_spec(self):
        """The CFG SPECIFICATION keyword must name a formula defined in the .tla file."""
        m = re.search(r"^SPECIFICATION\s+(\w+)", self._cfg, re.MULTILINE)
        assert m is not None, "CFG must have a SPECIFICATION line"
        spec_name = m.group(1)
        assert spec_name + " ==" in self._tla, (
            f"CFG SPECIFICATION '{spec_name}' not defined in {TLA_FILE.name}"
        )

    def test_invariants_match_definitions(self):
        """Every INVARIANT in the CFG must have a matching '==' definition in the spec."""
        invariants = self._cfg_invariants()
        assert len(invariants) >= 2, "Expected at least 2 invariants in CFG"
        for inv in invariants:
            assert inv + " ==" in self._tla, (
                f"CFG INVARIANT '{inv}' has no '==' definition in {TLA_FILE.name}"
            )

    def test_cfg_has_max_frames(self):
        assert "MaxFrames" in self._cfg

    def test_cfg_has_bodies(self):
        assert "Bodies" in self._cfg

    def test_cfg_has_hmac_key(self):
        assert "HMACKey" in self._cfg


class TestTLAReadme:
    """The TLA+ README must document the key properties and trust assumptions."""

    @pytest.fixture(autouse=True)
    def readme_content(self):
        self._text = README_FILE.read_text(encoding="utf-8")

    def test_has_properties_section(self):
        assert "Properties" in self._text or "Invariants" in self._text or "proved" in self._text

    def test_references_integrity_invariant(self):
        assert "IntegrityInvariant" in self._text

    def test_references_tamper_evidence(self):
        assert "TamperEvidenceInvariant" in self._text

    def test_references_zero_hmac(self):
        assert "ZeroHMAC" in self._text or "ZERO_HMAC" in self._text

    def test_references_trace_writer(self):
        assert "trace_writer" in self._text or "TraceWriter" in self._text

    def test_scope_limitations_mentioned(self):
        """README must mention truncation or scope limitations."""
        text_lower = self._text.lower()
        assert "truncat" in text_lower or "scope" in text_lower or "limitation" in text_lower

    def test_tlc_run_instructions(self):
        assert "tlc" in self._text.lower()


# ---------------------------------------------------------------------------
# Conformance dashboard tests
# ---------------------------------------------------------------------------

class TestConformanceDashboard:
    @pytest.fixture(autouse=True)
    def content(self):
        assert CONFORMANCE_DASHBOARD.exists(), (
            f"Conformance dashboard missing: {CONFORMANCE_DASHBOARD}"
        )
        self._text = CONFORMANCE_DASHBOARD.read_text(encoding="utf-8")

    def test_has_status_legend(self):
        assert "legend" in self._text.lower() or "Symbol" in self._text

    def test_mentions_python(self):
        assert "Python" in self._text

    def test_mentions_rust(self):
        assert "Rust" in self._text

    def test_mentions_hmac_chain(self):
        assert "HMAC" in self._text

    def test_references_tla_spec(self):
        """Dashboard should reference the TLA+ formal model."""
        assert "tla" in self._text.lower() or "TLA" in self._text

    def test_has_last_updated(self):
        assert "Last updated" in self._text or "last updated" in self._text

    def test_implementations_table(self):
        """Must have a table listing the implementations."""
        assert "Python" in self._text
        assert "TypeScript" in self._text
        assert "Go" in self._text

    def test_evidence_links(self):
        """Must link to test files as evidence."""
        assert "test_" in self._text


# ---------------------------------------------------------------------------
# Standards proposal tests
# ---------------------------------------------------------------------------

class TestLFProposal:
    @pytest.fixture(autouse=True)
    def content(self):
        assert LF_PROPOSAL.exists(), f"LF proposal missing: {LF_PROPOSAL}"
        self._text = LF_PROPOSAL.read_text(encoding="utf-8")

    def test_has_draft_marker(self):
        """Must be clearly marked as a draft."""
        assert "DRAFT" in self._text or "draft" in self._text.lower()

    def test_has_preconditions_checklist(self):
        assert "Precondition" in self._text or "precondition" in self._text.lower()

    def test_mentions_apache_license(self):
        assert "Apache" in self._text

    def test_mentions_linux_foundation(self):
        assert "Linux Foundation" in self._text

    def test_mentions_governance(self):
        assert "Governance" in self._text or "governance" in self._text


class TestCNCFDraft:
    @pytest.fixture(autouse=True)
    def content(self):
        assert CNCF_DRAFT.exists(), f"CNCF sandbox draft missing: {CNCF_DRAFT}"
        self._text = CNCF_DRAFT.read_text(encoding="utf-8")

    def test_has_draft_marker(self):
        assert "DRAFT" in self._text or "draft" in self._text.lower()

    def test_has_preconditions_checklist(self):
        assert "Precondition" in self._text or "precondition" in self._text.lower()

    def test_mentions_production_use_precondition(self):
        """Must explicitly note the multi-implementation production use gate."""
        text_lower = self._text.lower()
        assert "production" in text_lower and ("precondition" in text_lower or "condition" in text_lower)

    def test_mentions_otel_alignment(self):
        assert "OTel" in self._text or "OpenTelemetry" in self._text

    def test_mentions_cncf(self):
        assert "CNCF" in self._text


# ---------------------------------------------------------------------------
# Optional: TLC model-checker integration (skipped unless TLC is available)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    shutil.which("tlc") is None and not os.path.exists("tla2tools.jar"),
    reason="TLC model checker not installed (set STEPBACK_RUN_TLC=1 and provide tlc or tla2tools.jar)",
)
def test_tlc_no_violations(tmp_path):
    """Run TLC and assert no invariant violations."""
    import shutil as _shutil

    # Determine how to invoke TLC
    if shutil.which("tlc"):
        cmd = ["tlc", str(TLA_FILE), "-config", str(CFG_FILE)]
    else:
        cmd = ["java", "-jar", "tla2tools.jar", str(TLA_FILE), "-config", str(CFG_FILE)]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(TLA_DIR),
        timeout=300,
    )
    output = result.stdout + result.stderr
    assert "Error" not in output or "No error has been found" in output, (
        f"TLC reported an error:\n{output}"
    )
    assert result.returncode == 0, (
        f"TLC exited with code {result.returncode}:\n{output}"
    )
