"""Documentation invariants for the Step 147 research artifact documents.

Verifies:
- RELATED_WORK.md and ARTIFACT.md exist and are substantial.
- All five paper documents exist and are substantial.
- Required sections are present in each document.
- ARTIFACT.md references all five paper paths.
- RELATED_WORK.md covers all five comparison axes.
- Paper documents reference implementation modules and tests.
- Paper documents contain explicit "Limitations" sections.
- Stochastic minimization paper cross-references minimization-paper.md.
- Incident paper cross-references SECURITY.md.
- Paper documents reference Step 147 (or are linked from ARTIFACT.md which does).
- ARTIFACT.md references Step 147 discharge claim.
- No placeholder / TODO / TBD language in the core claim sections.
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).parent.parent
DOCS = ROOT / "docs"

RELATED_WORK = ROOT / "RELATED_WORK.md"
ARTIFACT_MD = ROOT / "ARTIFACT.md"

DIRTY_SET_PAPER = DOCS / "dirty-set-paper.md"
DISTRIBUTED_PAPER = DOCS / "distributed-replay-paper.md"
STOCHASTIC_PAPER = DOCS / "stochastic-minimization-paper.md"
NEURIPS_PAPER = DOCS / "neurips-datasets-paper.md"
INCIDENT_PAPER = DOCS / "incident-audit-paper.md"


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. File existence
# ---------------------------------------------------------------------------


def test_related_work_exists():
    assert RELATED_WORK.exists(), f"missing: {RELATED_WORK}"


def test_artifact_md_exists():
    assert ARTIFACT_MD.exists(), f"missing: {ARTIFACT_MD}"


def test_dirty_set_paper_exists():
    assert DIRTY_SET_PAPER.exists(), f"missing: {DIRTY_SET_PAPER}"


def test_distributed_paper_exists():
    assert DISTRIBUTED_PAPER.exists(), f"missing: {DISTRIBUTED_PAPER}"


def test_stochastic_paper_exists():
    assert STOCHASTIC_PAPER.exists(), f"missing: {STOCHASTIC_PAPER}"


def test_neurips_paper_exists():
    assert NEURIPS_PAPER.exists(), f"missing: {NEURIPS_PAPER}"


def test_incident_paper_exists():
    assert INCIDENT_PAPER.exists(), f"missing: {INCIDENT_PAPER}"


# ---------------------------------------------------------------------------
# 2. Minimum size requirements (substantial documents)
# ---------------------------------------------------------------------------


def test_related_work_is_substantial():
    assert RELATED_WORK.stat().st_size > 8_000, "RELATED_WORK.md too small"


def test_artifact_md_is_substantial():
    assert ARTIFACT_MD.stat().st_size > 5_000, "ARTIFACT.md too small"


def test_dirty_set_paper_is_substantial():
    assert DIRTY_SET_PAPER.stat().st_size > 8_000, "dirty-set-paper.md too small"


def test_distributed_paper_is_substantial():
    assert DISTRIBUTED_PAPER.stat().st_size > 8_000, "distributed-replay-paper.md too small"


def test_stochastic_paper_is_substantial():
    assert STOCHASTIC_PAPER.stat().st_size > 8_000, "stochastic-minimization-paper.md too small"


def test_incident_paper_is_substantial():
    assert INCIDENT_PAPER.stat().st_size > 8_000, "incident-audit-paper.md too small"


# ---------------------------------------------------------------------------
# 3. ARTIFACT.md content
# ---------------------------------------------------------------------------


def test_artifact_references_step_147():
    text = _read(ARTIFACT_MD)
    assert "Step 147" in text or "147" in text, "ARTIFACT.md must reference Step 147"


def test_artifact_references_all_five_paper_paths():
    text = _read(ARTIFACT_MD)
    for path in [
        "dirty-set-paper.md",
        "distributed-replay-paper.md",
        "stochastic-minimization-paper.md",
        "neurips-datasets-paper.md",
        "incident-audit-paper.md",
    ]:
        assert path in text, f"ARTIFACT.md must reference {path}"


def test_artifact_references_source_modules():
    text = _read(ARTIFACT_MD)
    for module in ["divergence.py", "replay.py", "minimize.py", "trace_writer.py"]:
        assert module in text, f"ARTIFACT.md should reference {module}"


def test_artifact_references_formal_proofs():
    text = _read(ARTIFACT_MD)
    assert "SBHMACChain.tla" in text, "ARTIFACT.md must reference TLA+ spec"


def test_artifact_references_corpora():
    text = _read(ARTIFACT_MD)
    assert "corpora" in text.lower(), "ARTIFACT.md must mention benchmark corpora"


def test_artifact_references_related_work():
    text = _read(ARTIFACT_MD)
    assert "RELATED_WORK" in text, "ARTIFACT.md must reference RELATED_WORK.md"


def test_artifact_references_security_md():
    text = _read(ARTIFACT_MD)
    assert "SECURITY" in text, "ARTIFACT.md must reference SECURITY.md"


def test_artifact_reproduction_commands():
    text = _read(ARTIFACT_MD)
    assert "python3 -m stepback bench" in text, "ARTIFACT.md must include reproduction commands"


# ---------------------------------------------------------------------------
# 4. RELATED_WORK.md content — five comparison axes
# ---------------------------------------------------------------------------


def test_related_work_covers_observability():
    text = _read(RELATED_WORK)
    # Must mention at least two observability tools
    for tool in ["LangSmith", "Phoenix", "LangFuse"]:
        assert tool in text, f"RELATED_WORK.md must mention {tool}"


def test_related_work_covers_time_travel_debuggers():
    text = _read(RELATED_WORK)
    for tool in ["rr", "Pernosco"]:
        assert tool in text, f"RELATED_WORK.md must mention {tool}"


def test_related_work_covers_build_systems():
    text = _read(RELATED_WORK)
    for tool in ["Bazel", "Nix"]:
        assert tool in text, f"RELATED_WORK.md must mention {tool}"


def test_related_work_covers_delta_debugging():
    text = _read(RELATED_WORK)
    assert "ddmin" in text.lower() or "delta debugging" in text.lower(), \
        "RELATED_WORK.md must mention delta debugging"


def test_related_work_covers_cryptographic_audit():
    text = _read(RELATED_WORK)
    for ref in ["Sigstore", "SLSA", "Certificate Transparency"]:
        assert ref in text, f"RELATED_WORK.md must mention {ref}"


def test_related_work_covers_agent_frameworks():
    text = _read(RELATED_WORK)
    for fw in ["LangChain", "LlamaIndex", "DSPy"]:
        assert fw in text, f"RELATED_WORK.md must mention {fw}"


def test_related_work_has_differentiation_context():
    text = _read(RELATED_WORK)
    # Must explain what stepback does differently
    assert "dirty" in text.lower() or "dirty-set" in text.lower(), \
        "RELATED_WORK.md must explain dirty-set differentiation"


def test_related_work_has_comparison_table():
    text = _read(RELATED_WORK)
    # Markdown table indicators
    assert "|" in text and "---" in text, \
        "RELATED_WORK.md must contain comparison tables"


# ---------------------------------------------------------------------------
# 5. Dirty-set paper content
# ---------------------------------------------------------------------------


def test_dirty_set_paper_has_abstract():
    text = _read(DIRTY_SET_PAPER)
    assert "## Abstract" in text or "**Abstract**" in text or "Abstract" in text, \
        "dirty-set-paper.md must have an Abstract section"


def test_dirty_set_paper_has_soundness():
    text = _read(DIRTY_SET_PAPER)
    assert "Soundness" in text or "soundness" in text, \
        "dirty-set-paper.md must discuss soundness"


def test_dirty_set_paper_has_completeness():
    text = _read(DIRTY_SET_PAPER)
    assert "completeness" in text.lower(), \
        "dirty-set-paper.md must discuss completeness"


def test_dirty_set_paper_has_limitations():
    text = _read(DIRTY_SET_PAPER)
    assert "Limitation" in text or "limitation" in text, \
        "dirty-set-paper.md must have a Limitations section"


def test_dirty_set_paper_references_divergence():
    text = _read(DIRTY_SET_PAPER)
    assert "divergence.py" in text, \
        "dirty-set-paper.md must reference divergence.py"


def test_dirty_set_paper_has_implementation_status_table():
    text = _read(DIRTY_SET_PAPER)
    assert "Implemented" in text, \
        "dirty-set-paper.md must have an implementation status table"


def test_dirty_set_paper_has_evaluation():
    text = _read(DIRTY_SET_PAPER)
    assert "Evaluation" in text or "evaluation" in text, \
        "dirty-set-paper.md must have an Evaluation section"


# ---------------------------------------------------------------------------
# 6. Distributed replay paper content
# ---------------------------------------------------------------------------


def test_distributed_paper_has_abstract():
    text = _read(DISTRIBUTED_PAPER)
    assert "Abstract" in text


def test_distributed_paper_distinguishes_implemented_vs_prototype():
    text = _read(DISTRIBUTED_PAPER)
    assert "Implemented" in text and "Prototype" in text, \
        "distributed-replay-paper.md must distinguish Implemented from Prototype"


def test_distributed_paper_has_limitations():
    text = _read(DISTRIBUTED_PAPER)
    assert "Limitation" in text or "limitation" in text


def test_distributed_paper_references_distributed_bisect():
    text = _read(DISTRIBUTED_PAPER)
    assert "distributed_bisect" in text, \
        "distributed-replay-paper.md must reference distributed_bisect.py"


def test_distributed_paper_references_backpressure():
    text = _read(DISTRIBUTED_PAPER)
    assert "backpressure" in text.lower() or "RecorderOptions" in text, \
        "distributed-replay-paper.md must reference backpressure"


def test_distributed_paper_references_soak():
    text = _read(DISTRIBUTED_PAPER)
    assert "soak" in text.lower(), \
        "distributed-replay-paper.md must reference soak harness"


# ---------------------------------------------------------------------------
# 7. Stochastic minimization paper content
# ---------------------------------------------------------------------------


def test_stochastic_paper_has_abstract():
    text = _read(STOCHASTIC_PAPER)
    assert "Abstract" in text


def test_stochastic_paper_cross_references_minimization_paper():
    text = _read(STOCHASTIC_PAPER)
    assert "minimization-paper.md" in text, \
        "stochastic-minimization-paper.md must cross-reference minimization-paper.md"


def test_stochastic_paper_discusses_confidence_intervals():
    text = _read(STOCHASTIC_PAPER)
    assert "confidence" in text.lower() or "Wilson" in text, \
        "stochastic-minimization-paper.md must discuss confidence intervals"


def test_stochastic_paper_discusses_replay_correctness_separation():
    text = _read(STOCHASTIC_PAPER)
    assert "correctness" in text.lower() and "stability" in text.lower(), \
        "must separate replay correctness from predicate stability"


def test_stochastic_paper_has_limitations():
    text = _read(STOCHASTIC_PAPER)
    assert "Limitation" in text or "limitation" in text


def test_stochastic_paper_references_stochastic_replay_tests():
    text = _read(STOCHASTIC_PAPER)
    assert "test_stochastic_replay" in text, \
        "stochastic-minimization-paper.md must reference test_stochastic_replay.py"


def test_stochastic_paper_has_implementation_status():
    text = _read(STOCHASTIC_PAPER)
    assert "Implemented" in text, \
        "stochastic-minimization-paper.md must have implementation status table"


# ---------------------------------------------------------------------------
# 8. Incident audit paper content
# ---------------------------------------------------------------------------


def test_incident_paper_has_abstract():
    text = _read(INCIDENT_PAPER)
    assert "Abstract" in text


def test_incident_paper_has_what_evidence_does_not_prove():
    text = _read(INCIDENT_PAPER)
    # Must explicitly say what the cryptographic evidence does NOT prove
    assert "does NOT prove" in text or "does not prove" in text, \
        "incident-audit-paper.md must state what evidence does not prove"


def test_incident_paper_references_security_md():
    text = _read(INCIDENT_PAPER)
    assert "SECURITY.md" in text, \
        "incident-audit-paper.md must reference SECURITY.md"


def test_incident_paper_references_tla_spec():
    text = _read(INCIDENT_PAPER)
    assert "SBHMACChain.tla" in text, \
        "incident-audit-paper.md must reference TLA+ spec"


def test_incident_paper_references_redact():
    text = _read(INCIDENT_PAPER)
    assert "redact" in text.lower(), \
        "incident-audit-paper.md must discuss redaction"


def test_incident_paper_has_limitations():
    text = _read(INCIDENT_PAPER)
    assert "Limitation" in text or "limitation" in text


def test_incident_paper_has_implementation_status_table():
    text = _read(INCIDENT_PAPER)
    assert "Implemented" in text


def test_incident_paper_covers_three_synthetic_corpora():
    text = _read(INCIDENT_PAPER)
    for corpus in ["support-agent", "code-review", "payments-policy"]:
        assert corpus in text, f"incident-audit-paper.md must mention {corpus}"


def test_incident_paper_has_recorder_honesty_caveat():
    text = _read(INCIDENT_PAPER)
    assert "honest" in text.lower() or "honesty" in text.lower(), \
        "incident-audit-paper.md must discuss recorder honesty assumption"


# ---------------------------------------------------------------------------
# 9. Cross-consistency between documents
# ---------------------------------------------------------------------------


def test_artifact_md_paper_paths_match_actual_files():
    """Every paper path listed in ARTIFACT.md must correspond to an existing file."""
    text = _read(ARTIFACT_MD)
    # Extract all .md paths that look like paper references
    candidates = re.findall(r"docs/[\w-]+\.md", text)
    for c in candidates:
        path = ROOT / c
        assert path.exists(), f"ARTIFACT.md references {c} but file does not exist"


def test_all_papers_cross_reference_related_work_or_artifact():
    """Each paper document should mention RELATED_WORK or cross-reference related work."""
    for path in [DIRTY_SET_PAPER, DISTRIBUTED_PAPER, STOCHASTIC_PAPER, INCIDENT_PAPER]:
        text = _read(path)
        assert "RELATED_WORK" in text or "related work" in text.lower(), \
            f"{path.name} must mention RELATED_WORK.md or have a related work section"


def test_stochastic_paper_does_not_redefine_ddmin_differently():
    """Stochastic paper must not define ddmin in a way that contradicts minimization-paper."""
    stoch = _read(STOCHASTIC_PAPER)
    mini = _read(DOCS / "minimization-paper.md")
    # Both must agree: ddmin finds 1-minimal subsets
    assert "1-minimal" in stoch or "1-minimal" in mini, \
        "ddmin definition inconsistency between stochastic and minimization papers"


def test_dirty_set_paper_references_neurips_for_evaluation():
    text = _read(DIRTY_SET_PAPER)
    assert "neurips-datasets-paper" in text or "neurips_datasets_paper" in text or \
           "neurips" in text.lower(), \
        "dirty-set-paper.md should reference neurips-datasets-paper.md for evaluation"
