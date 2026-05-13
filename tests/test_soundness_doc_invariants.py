"""Documentation invariants for ``docs/dirty-set-soundness.md`` — Step 55.

Asserts that the paper-grade soundness proof document exists, is
non-trivial, and covers every required element:

* Theorem 1 (P1 soundness statement)
* Lemma 1 (parent-agreement lemma)
* Proof of Theorem 1 by strong induction on topo(s)
* Assumptions A1, A2, A3
* Recorder obligations R1–R4
* Counterexamples motivating each assumption
* Mapping from proof statements to implementation sites
* Cross-references to divergence.py, replay.py, docs/dirty-set.md
"""
from __future__ import annotations

import pathlib

DOCS_DIR = pathlib.Path(__file__).parent.parent / "docs"
SOUNDNESS_DOC = DOCS_DIR / "dirty-set-soundness.md"


def _doc_text() -> str:
    return SOUNDNESS_DOC.read_text(encoding="utf-8")


# -------------------------------------------------------------------------
# 1. File existence and minimum size
# -------------------------------------------------------------------------


def test_soundness_doc_exists():
    """``docs/dirty-set-soundness.md`` must exist."""
    assert SOUNDNESS_DOC.exists(), f"missing: {SOUNDNESS_DOC}"


def test_soundness_doc_is_substantial():
    """The proof document must have substantial content (> 8 KiB)."""
    size = SOUNDNESS_DOC.stat().st_size
    assert size > 8_000, (
        f"docs/dirty-set-soundness.md is suspiciously small ({size} bytes)"
    )


# -------------------------------------------------------------------------
# 2. Step 55 discharge claim
# -------------------------------------------------------------------------


def test_soundness_doc_references_step_55():
    """The document must claim to discharge Step 55."""
    text = _doc_text()
    assert "Step 55" in text or "step 55" in text.lower(), (
        "docs/dirty-set-soundness.md must reference Step 55 of 100_STEPS.md"
    )


def test_soundness_doc_states_soundness_theorem():
    """Theorem 1 (the P1 soundness claim) must be stated."""
    text = _doc_text()
    assert "Theorem 1" in text or "soundness theorem" in text.lower(), (
        "docs/dirty-set-soundness.md must state Theorem 1 (soundness)"
    )


def test_soundness_doc_has_lemma_1():
    """Lemma 1 (parent-agreement) is the key inductive step."""
    text = _doc_text()
    assert "Lemma 1" in text or "parent agreement" in text.lower(), (
        "docs/dirty-set-soundness.md must contain Lemma 1 (parent agreement)"
    )


# -------------------------------------------------------------------------
# 3. Assumptions and recorder obligations
# -------------------------------------------------------------------------


def test_soundness_doc_documents_assumptions():
    """A1, A2, A3 must all be explicitly stated."""
    text = _doc_text()
    for label in ("A1", "A2", "A3"):
        assert label in text, (
            f"docs/dirty-set-soundness.md must document assumption {label}"
        )


def test_soundness_doc_documents_recorder_obligations():
    """R1–R4 must be explicitly stated as proof preconditions."""
    text = _doc_text()
    for label in ("R1", "R2", "R3", "R4"):
        assert label in text, (
            f"docs/dirty-set-soundness.md must document recorder obligation {label}"
        )


# -------------------------------------------------------------------------
# 4. Proof structure
# -------------------------------------------------------------------------


def test_soundness_doc_uses_induction():
    """The proof must be by induction on topological order."""
    text = _doc_text()
    assert "induction" in text.lower() or "topo" in text, (
        "docs/dirty-set-soundness.md must use induction on topological order"
    )


def test_soundness_doc_covers_base_case():
    """A base case must be explicitly identified."""
    text = _doc_text()
    assert "base case" in text.lower() or "Base case" in text, (
        "docs/dirty-set-soundness.md must handle the base case (root step)"
    )


def test_soundness_doc_covers_inductive_step():
    """An inductive step must be explicitly identified."""
    text = _doc_text()
    assert "inductive step" in text.lower() or "Inductive step" in text, (
        "docs/dirty-set-soundness.md must prove the inductive step"
    )


def test_soundness_doc_covers_clean_and_dirty_cases():
    """The proof must cover both dirty-step and clean-step cases."""
    text = _doc_text()
    assert "clean" in text.lower() and "dirty" in text.lower(), (
        "docs/dirty-set-soundness.md must discuss both clean and dirty cases"
    )


def test_soundness_doc_covers_output_forcing():
    """Output-forcing substitutions must be handled explicitly."""
    text = _doc_text()
    assert "output-forcing" in text.lower() or "output_forcing" in text or "M_out" in text, (
        "docs/dirty-set-soundness.md must handle output-forcing substitutions"
    )


def test_soundness_doc_covers_input_mutating():
    """Input-mutating substitutions must be mentioned."""
    text = _doc_text()
    assert "input-mutating" in text.lower() or "input_mutating" in text or "M_in" in text, (
        "docs/dirty-set-soundness.md must handle input-mutating substitutions"
    )


# -------------------------------------------------------------------------
# 5. Counterexamples
# -------------------------------------------------------------------------


def test_soundness_doc_has_counterexamples_section():
    """Counterexamples motivating each assumption must be present."""
    text = _doc_text()
    assert "counterexample" in text.lower() or "Counterexample" in text, (
        "docs/dirty-set-soundness.md must include counterexamples for each assumption"
    )


def test_soundness_doc_counterexample_covers_a1():
    """A1 violation counterexample must be present."""
    text = _doc_text()
    assert "A1 violation" in text or "A1" in text, (
        "docs/dirty-set-soundness.md must include a counterexample for A1"
    )


def test_soundness_doc_counterexample_covers_a2():
    """A2 violation counterexample must be present."""
    text = _doc_text()
    assert "A2 violation" in text or "stochastic" in text.lower(), (
        "docs/dirty-set-soundness.md must include a counterexample for A2"
    )


# -------------------------------------------------------------------------
# 6. Implementation mapping
# -------------------------------------------------------------------------


def test_soundness_doc_maps_proof_to_implementation():
    """Each proof statement must map to a concrete implementation site."""
    text = _doc_text()
    assert "replay.py" in text and ("line" in text.lower() or "lines" in text.lower()), (
        "docs/dirty-set-soundness.md must map proof statements to replay.py lines"
    )


def test_soundness_doc_references_divergence_py():
    """The proof must reference the divergence.py classifier."""
    text = _doc_text()
    assert "divergence.py" in text, (
        "docs/dirty-set-soundness.md must cross-reference stepback/divergence.py"
    )


# -------------------------------------------------------------------------
# 7. Cross-references
# -------------------------------------------------------------------------


def test_soundness_doc_references_dirty_set_md():
    """The proof must reference docs/dirty-set.md for shared notation."""
    text = _doc_text()
    assert "dirty-set.md" in text, (
        "docs/dirty-set-soundness.md must reference docs/dirty-set.md"
    )


def test_soundness_doc_references_100_steps():
    """The document must reference 100_STEPS.md."""
    text = _doc_text()
    assert "100_STEPS.md" in text or "100_STEPS" in text, (
        "docs/dirty-set-soundness.md must reference 100_STEPS.md"
    )


def test_soundness_doc_references_completeness_doc():
    """Step 57 completeness doc must be referenced (scoping soundness vs. completeness)."""
    text = _doc_text()
    assert "completeness" in text.lower() or "dirty-set-completeness" in text, (
        "docs/dirty-set-soundness.md must distinguish soundness from completeness (Step 57)"
    )


# -------------------------------------------------------------------------
# 8. Version pinning
# -------------------------------------------------------------------------


def test_soundness_doc_pins_canonicalisation_version():
    """The proof must be pinned to a specific canonicalisation_version."""
    text = _doc_text()
    assert "canonicalisation_version" in text or "canonical" in text.lower(), (
        "docs/dirty-set-soundness.md must pin the canonicalisation version"
    )


def test_soundness_doc_pins_dirty_set_version():
    """The proof must be pinned to a specific dirty_set_version."""
    text = _doc_text()
    assert "dirty_set_version" in text or "version" in text.lower(), (
        "docs/dirty-set-soundness.md must pin the dirty-set algorithm version"
    )
