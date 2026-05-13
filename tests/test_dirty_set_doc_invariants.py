"""Documentation invariants for ``docs/dirty-set.md`` — Step 54.

Asserts that the canonical reference document for the dirty-set algorithm
exists and covers every topic required by Step 54 of ``100_STEPS.md``:

* Trace DAG definition
* Canonical input function
* Substitution σ
* Dirty set D(T, σ)
* Cache reuse
* Observational equivalence

These tests are intentionally lightweight: they do not parse markdown
AST, they just grep the raw text for required headings and key terms.
The goal is to make future accidental deletions or renames of the doc
trip a test before anyone notices.
"""
from __future__ import annotations

import os
import pathlib

DOCS_DIR = pathlib.Path(__file__).parent.parent / "docs"
DIRTY_SET_DOC = DOCS_DIR / "dirty-set.md"


def _doc_text() -> str:
    return DIRTY_SET_DOC.read_text(encoding="utf-8")


# -------------------------------------------------------------------------
# 1. File existence and minimum size
# -------------------------------------------------------------------------


def test_dirty_set_doc_exists():
    """``docs/dirty-set.md`` must exist."""
    assert DIRTY_SET_DOC.exists(), f"missing: {DIRTY_SET_DOC}"


def test_dirty_set_doc_is_not_empty():
    """The doc must have substantial content (> 5 KiB)."""
    assert DIRTY_SET_DOC.stat().st_size > 5_000, (
        f"docs/dirty-set.md is suspiciously small "
        f"({DIRTY_SET_DOC.stat().st_size} bytes)"
    )


# -------------------------------------------------------------------------
# 2. Required section headings
# -------------------------------------------------------------------------


def test_dirty_set_doc_has_trace_dag_section():
    """§1 must define the Trace DAG."""
    text = _doc_text()
    assert "Trace DAG" in text or "trace DAG" in text, (
        "docs/dirty-set.md must contain a 'Trace DAG' section"
    )


def test_dirty_set_doc_has_canonical_input_function_section():
    """§2 must define the canonical input function."""
    text = _doc_text()
    assert "Canonical input function" in text or "canonical input" in text, (
        "docs/dirty-set.md must define the canonical input function"
    )


def test_dirty_set_doc_has_substitution_section():
    """§3 must define the substitution σ."""
    text = _doc_text()
    assert "Substitution" in text and ("sigma" in text or "σ" in text), (
        "docs/dirty-set.md must define substitution σ"
    )


def test_dirty_set_doc_has_dirty_set_section():
    """§4 must define the dirty set D(T, σ)."""
    text = _doc_text()
    assert "dirty set" in text.lower(), (
        "docs/dirty-set.md must define the dirty set"
    )
    # Check for the formal set notation or algorithmic pseudocode.
    assert "D(T" in text or "dirty_count" in text or "classify" in text, (
        "docs/dirty-set.md must include the dirty-set definition or classify function"
    )


def test_dirty_set_doc_has_cache_reuse_section():
    """§5 must cover cache reuse semantics."""
    text = _doc_text()
    assert "cache" in text.lower() and "reuse" in text.lower(), (
        "docs/dirty-set.md must cover cache reuse"
    )


def test_dirty_set_doc_has_observational_equivalence_section():
    """§6 must state the observational-equivalence theorem."""
    text = _doc_text()
    assert "observational" in text.lower() and "equivalence" in text.lower(), (
        "docs/dirty-set.md must state the observational-equivalence theorem"
    )


# -------------------------------------------------------------------------
# 3. Required formal objects and postconditions
# -------------------------------------------------------------------------


def test_dirty_set_doc_defines_postconditions():
    """P1–P4 postconditions must be documented."""
    text = _doc_text()
    for label in ("P1", "P2", "P3", "P4"):
        assert label in text, (
            f"docs/dirty-set.md must document postcondition {label}"
        )


def test_dirty_set_doc_defines_assumptions():
    """A1–A3 assumptions must be documented."""
    text = _doc_text()
    for label in ("A1", "A2", "A3"):
        assert label in text, (
            f"docs/dirty-set.md must document assumption {label}"
        )


def test_dirty_set_doc_defines_branch_invariants():
    """B1–B3 branch-aware propagation invariants must be documented."""
    text = _doc_text()
    for label in ("B1", "B2", "B3"):
        assert label in text, (
            f"docs/dirty-set.md must document branch invariant {label}"
        )


# -------------------------------------------------------------------------
# 4. Cross-references to implementation
# -------------------------------------------------------------------------


def test_dirty_set_doc_references_divergence_py():
    """The doc must cross-reference the implementation in divergence.py."""
    text = _doc_text()
    assert "divergence.py" in text, (
        "docs/dirty-set.md must reference stepback/divergence.py"
    )


def test_dirty_set_doc_references_replay_py():
    """The doc must cross-reference the replay engine in replay.py."""
    text = _doc_text()
    assert "replay.py" in text, (
        "docs/dirty-set.md must reference stepback/replay.py"
    )


def test_dirty_set_doc_references_canonicalization_doc():
    """The doc must cross-reference docs/canonicalization.md."""
    text = _doc_text()
    assert "canonicalization.md" in text, (
        "docs/dirty-set.md must link to docs/canonicalization.md"
    )


# -------------------------------------------------------------------------
# 5. Key algorithmic terms
# -------------------------------------------------------------------------


def test_dirty_set_doc_mentions_topological_order():
    """Topological ordering is a precondition; it must be mentioned."""
    text = _doc_text()
    assert "topological" in text.lower(), (
        "docs/dirty-set.md must mention topological order"
    )


def test_dirty_set_doc_mentions_inputs_hash():
    """``inputs_hash`` is the central cache key; it must be named."""
    text = _doc_text()
    assert "inputs_hash" in text, (
        "docs/dirty-set.md must mention 'inputs_hash'"
    )


def test_dirty_set_doc_mentions_nondeterminism_hash():
    """``nondeterminism_hash`` is part of the dirty classification."""
    text = _doc_text()
    assert "nondeterminism_hash" in text, (
        "docs/dirty-set.md must mention 'nondeterminism_hash'"
    )


def test_dirty_set_doc_mentions_parent_step_id():
    """``parent_step_id`` is the single-parent edge field."""
    text = _doc_text()
    assert "parent_step_id" in text, (
        "docs/dirty-set.md must mention 'parent_step_id'"
    )


def test_dirty_set_doc_mentions_soundness_theorem():
    """The soundness theorem must be stated explicitly."""
    text = _doc_text()
    assert "soundness" in text.lower() or "Theorem" in text, (
        "docs/dirty-set.md must state the soundness theorem"
    )


def test_dirty_set_doc_mentions_recorder_obligations():
    """Recorder obligations R1–R4 must be documented."""
    text = _doc_text()
    for label in ("R1", "R2", "R3", "R4"):
        assert label in text, (
            f"docs/dirty-set.md must document recorder obligation {label}"
        )


# -------------------------------------------------------------------------
# 6. Version pinning
# -------------------------------------------------------------------------


def test_dirty_set_doc_mentions_version():
    """The dirty-set version must be pinned in the doc."""
    text = _doc_text()
    assert "dirty_set_version" in text or "version" in text.lower(), (
        "docs/dirty-set.md must mention the algorithm version"
    )
