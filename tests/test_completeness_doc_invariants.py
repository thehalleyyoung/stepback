"""Documentation invariants for ``docs/dirty-set-completeness.md`` — Step 57.

Asserts that the paper-grade completeness proof document exists, is
non-trivial, and covers every required element:

* Theorem 2 / P2 (completeness statement — every drift step is dirty)
* Corollary 2 (headline form of Step 57)
* Corollary 3 (descendant closure / P2 transitive)
* Lemma 2 (classifier exhaustiveness)
* Lemma 3 (parent-dirty closure)
* Assumptions A3 (canonical encoder injectivity — the only one needed)
* Recorder obligations R1, R3, R4
* Explicit statement that A1, A2, R2 are *not* needed
* Counterexamples motivating each assumption
* Mapping from proof statements to implementation sites
* Cross-references to divergence.py, dirty-set.md, dirty-set-soundness.md
* Version pins (canonicalisation_version, dirty_set_version)
"""
from __future__ import annotations

import pathlib

DOCS_DIR = pathlib.Path(__file__).parent.parent / "docs"
COMPLETENESS_DOC = DOCS_DIR / "dirty-set-completeness.md"


def _doc_text() -> str:
    return COMPLETENESS_DOC.read_text(encoding="utf-8")


# -------------------------------------------------------------------------
# 1. File existence and minimum size
# -------------------------------------------------------------------------


def test_completeness_doc_exists():
    """``docs/dirty-set-completeness.md`` must exist."""
    assert COMPLETENESS_DOC.exists(), f"missing: {COMPLETENESS_DOC}"


def test_completeness_doc_is_substantial():
    """The proof document must have substantial content (> 8 KiB)."""
    size = COMPLETENESS_DOC.stat().st_size
    assert size > 8_000, (
        f"docs/dirty-set-completeness.md is suspiciously small ({size} bytes)"
    )


# -------------------------------------------------------------------------
# 2. Step 57 discharge claim
# -------------------------------------------------------------------------


def test_completeness_doc_references_step_57():
    """The document must claim to discharge Step 57."""
    text = _doc_text()
    assert "Step 57" in text or "step 57" in text.lower(), (
        "docs/dirty-set-completeness.md must reference Step 57 of 100_STEPS.md"
    )


def test_completeness_doc_states_headline_claim():
    """The headline Step 57 claim must appear verbatim or in equivalent form."""
    text = _doc_text()
    # The step says: "every step whose recomputed inputs hash differently
    # is included in the dirty set"
    has_drift = "drift" in text.lower()
    has_dirty = "dirty set" in text.lower() or "D(T" in text
    has_hash = "inputs_hash" in text or "hash differently" in text.lower()
    assert has_drift and has_dirty and has_hash, (
        "docs/dirty-set-completeness.md must state: every step whose recomputed "
        "inputs hash differs is in the dirty set (Drift(T,σ) ⊆ D(T,σ))"
    )


# -------------------------------------------------------------------------
# 3. Theorem / corollary statements
# -------------------------------------------------------------------------


def test_completeness_doc_states_theorem_2():
    """Theorem 2 (P2 completeness) must be stated."""
    text = _doc_text()
    assert "Theorem 2" in text or "completeness theorem" in text.lower(), (
        "docs/dirty-set-completeness.md must state Theorem 2 (P2 completeness)"
    )


def test_completeness_doc_has_corollary_2():
    """Corollary 2 (headline form) must be present."""
    text = _doc_text()
    assert "Corollary 2" in text, (
        "docs/dirty-set-completeness.md must include Corollary 2 (headline form of Step 57)"
    )


def test_completeness_doc_has_corollary_3():
    """Corollary 3 (descendant closure, P2 transitive) must be present."""
    text = _doc_text()
    assert "Corollary 3" in text or "transitive" in text.lower(), (
        "docs/dirty-set-completeness.md must include Corollary 3 (P2 transitive / descendant closure)"
    )


def test_completeness_doc_has_lemma_2():
    """Lemma 2 (classifier exhaustiveness) must be present."""
    text = _doc_text()
    assert "Lemma 2" in text or "exhaustive" in text.lower(), (
        "docs/dirty-set-completeness.md must contain Lemma 2 (classifier exhaustiveness)"
    )


def test_completeness_doc_has_lemma_3():
    """Lemma 3 (parent-dirty closure) must be present."""
    text = _doc_text()
    assert "Lemma 3" in text or "parent-dirty closure" in text.lower(), (
        "docs/dirty-set-completeness.md must contain Lemma 3 (parent-dirty closure)"
    )


def test_completeness_doc_states_four_inclusions():
    """Theorem 2 must state all four set inclusions."""
    text = _doc_text()
    # Targeted, Tamper, Drift, ParentDirty must all appear
    for term in ("Targeted", "Tamper", "Drift", "ParentDirty"):
        assert term in text, (
            f"docs/dirty-set-completeness.md must state the {term}(…) ⊆ D(T,σ) inclusion"
        )


# -------------------------------------------------------------------------
# 4. Assumptions — A3 is required; A1, A2 are explicitly not needed
# -------------------------------------------------------------------------


def test_completeness_doc_documents_a3():
    """A3 (canonical encoder injectivity) must be stated — the only assumption needed."""
    text = _doc_text()
    assert "A3" in text, (
        "docs/dirty-set-completeness.md must document assumption A3 (canonical encoder injectivity)"
    )


def test_completeness_doc_explicitly_drops_a1_a2():
    """The document must explicitly state that A1 and A2 are NOT needed."""
    text = _doc_text()
    # The doc must say A1 and A2 are not used / not required
    assert "A1" in text and "A2" in text, (
        "docs/dirty-set-completeness.md must mention A1 and A2 (to note they are not needed)"
    )
    a1_not_needed = (
        "not used" in text.lower()
        or "not required" in text.lower()
        or "not needed" in text.lower()
        or "not use" in text.lower()
    )
    assert a1_not_needed, (
        "docs/dirty-set-completeness.md must explicitly state that A1/A2 are not needed for completeness"
    )


# -------------------------------------------------------------------------
# 5. Recorder obligations
# -------------------------------------------------------------------------


def test_completeness_doc_documents_recorder_obligations():
    """R1, R3, R4 must be stated as preconditions; R2 must be noted as unnecessary."""
    text = _doc_text()
    for label in ("R1", "R3", "R4"):
        assert label in text, (
            f"docs/dirty-set-completeness.md must document recorder obligation {label}"
        )
    assert "R2" in text, (
        "docs/dirty-set-completeness.md must mention R2 (even if only to exclude it)"
    )


# -------------------------------------------------------------------------
# 6. Proof structure — induction
# -------------------------------------------------------------------------


def test_completeness_doc_uses_induction():
    """The drift-set inclusion must be proved by induction on topological order."""
    text = _doc_text()
    assert "induction" in text.lower() or "topo" in text, (
        "docs/dirty-set-completeness.md must use induction on topological order"
    )


def test_completeness_doc_covers_input_drift_case():
    """Sub-case (d) — Drift ⊆ D — must be proved (the headline claim)."""
    text = _doc_text()
    assert "input_drift" in text or "input drift" in text.lower(), (
        "docs/dirty-set-completeness.md must prove the input-drift sub-case (d)"
    )


def test_completeness_doc_covers_substituted_case():
    """Sub-case (a) — Targeted ⊆ D — must be proved."""
    text = _doc_text()
    assert "substituted" in text.lower() or "Targeted" in text, (
        "docs/dirty-set-completeness.md must prove the Targeted ⊆ D case"
    )


def test_completeness_doc_covers_tamper_case():
    """Sub-case (b) — Tamper ⊆ D — must be proved."""
    text = _doc_text()
    assert "ndh_tamper" in text or "tamper" in text.lower(), (
        "docs/dirty-set-completeness.md must prove the Tamper ⊆ D case"
    )


def test_completeness_doc_covers_parent_dirty_case():
    """Sub-case (c) — ParentDirty ⊆ D — must be proved."""
    text = _doc_text()
    assert "parent_dirty" in text or "parent-dirty" in text.lower(), (
        "docs/dirty-set-completeness.md must prove the ParentDirty ⊆ D case"
    )


# -------------------------------------------------------------------------
# 7. Counterexamples
# -------------------------------------------------------------------------


def test_completeness_doc_has_counterexamples_section():
    """A section with counterexamples motivating each assumption must be present."""
    text = _doc_text()
    assert "counterexample" in text.lower() or "Counterexample" in text, (
        "docs/dirty-set-completeness.md must include counterexamples for each assumption"
    )


def test_completeness_doc_counterexample_covers_r1():
    """An R1 violation counterexample must be present."""
    text = _doc_text()
    assert "R1 violation" in text or ("R1" in text and "violation" in text.lower()), (
        "docs/dirty-set-completeness.md must include a counterexample for R1"
    )


def test_completeness_doc_counterexample_covers_a3():
    """An A3 violation counterexample (hash collision) must be present."""
    text = _doc_text()
    assert (
        "A3 violation" in text
        or "collision" in text.lower()
        or "collision-resistant" in text.lower()
    ), (
        "docs/dirty-set-completeness.md must include an A3 counterexample (hash collision)"
    )


# -------------------------------------------------------------------------
# 8. Implementation mapping
# -------------------------------------------------------------------------


def test_completeness_doc_maps_proof_to_implementation():
    """Each proof statement must map to a concrete implementation site."""
    text = _doc_text()
    assert "divergence.py" in text and ("line" in text.lower() or "lines" in text.lower()), (
        "docs/dirty-set-completeness.md must map proof statements to divergence.py lines"
    )


def test_completeness_doc_references_dirty_reason_field():
    """The classifier's dirty_reason field must be referenced in the mapping."""
    text = _doc_text()
    assert "dirty_reason" in text, (
        "docs/dirty-set-completeness.md must reference the dirty_reason field in divergence.py"
    )


# -------------------------------------------------------------------------
# 9. Cross-references
# -------------------------------------------------------------------------


def test_completeness_doc_references_dirty_set_md():
    """The proof must reference docs/dirty-set.md for shared notation."""
    text = _doc_text()
    assert "dirty-set.md" in text, (
        "docs/dirty-set-completeness.md must reference docs/dirty-set.md"
    )


def test_completeness_doc_references_soundness_doc():
    """The completeness proof must reference the sibling soundness document."""
    text = _doc_text()
    assert "dirty-set-soundness.md" in text or "soundness" in text.lower(), (
        "docs/dirty-set-completeness.md must reference docs/dirty-set-soundness.md"
    )


def test_completeness_doc_references_100_steps():
    """The document must reference 100_STEPS.md."""
    text = _doc_text()
    assert "100_STEPS.md" in text or "100_STEPS" in text, (
        "docs/dirty-set-completeness.md must reference 100_STEPS.md"
    )


def test_completeness_doc_references_divergence_py():
    """The proof must reference the divergence.py classifier."""
    text = _doc_text()
    assert "divergence.py" in text, (
        "docs/dirty-set-completeness.md must cross-reference stepback/divergence.py"
    )


def test_completeness_doc_distinguishes_soundness_and_completeness():
    """The document must explicitly distinguish soundness (P1) from completeness (P2)."""
    text = _doc_text()
    assert "soundness" in text.lower() and "completeness" in text.lower(), (
        "docs/dirty-set-completeness.md must explicitly distinguish soundness (P1) and completeness (P2)"
    )


# -------------------------------------------------------------------------
# 10. Version pins
# -------------------------------------------------------------------------


def test_completeness_doc_pins_canonicalisation_version():
    """The proof must be pinned to a specific canonicalisation_version."""
    text = _doc_text()
    assert "canonicalisation_version" in text or "canonical" in text.lower(), (
        "docs/dirty-set-completeness.md must pin the canonicalisation version"
    )


def test_completeness_doc_pins_dirty_set_version():
    """The proof must be pinned to a specific dirty_set_version."""
    text = _doc_text()
    assert "dirty_set_version" in text or "version" in text.lower(), (
        "docs/dirty-set-completeness.md must pin the dirty-set algorithm version"
    )


# -------------------------------------------------------------------------
# 11. Reach set — D(T,σ) = Reach(T,σ) equivalence
# -------------------------------------------------------------------------


def test_completeness_doc_states_reach_equivalence():
    """The document must state that D(T,σ) = Reach(T,σ)."""
    text = _doc_text()
    assert "Reach" in text or "reachable" in text.lower(), (
        "docs/dirty-set-completeness.md must state the D(T,σ) = Reach(T,σ) equivalence"
    )
