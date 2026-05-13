"""Contract-documentation invariants for ``stepback/divergence.py`` — Step 53.

Asserts that the README pseudocode has been lifted into ``stepback/divergence.py``
as contract documentation covering:

* The dirty-set algorithm pseudocode (from README §"The dirty-set algorithm")
* Postconditions P1–P6 (labelled in contract tests)
* Assumptions A1–A4
* Branch-aware propagation invariants B1–B3
* Preconditions and postconditions in ``compute_dirty_set`` docstring

These tests are intentionally lightweight: they do not parse Python AST,
they just grep the raw source for required labels and key terms.  The goal
is to make future accidental deletions or renames trip a test before anyone
notices.
"""
from __future__ import annotations

import inspect
import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent
DIVERGENCE_PY = REPO_ROOT / "stepback" / "divergence.py"


def _module_text() -> str:
    return DIVERGENCE_PY.read_text(encoding="utf-8")


# -------------------------------------------------------------------------
# 1. File existence
# -------------------------------------------------------------------------


def test_divergence_py_exists():
    """``stepback/divergence.py`` must exist."""
    assert DIVERGENCE_PY.exists(), f"missing: {DIVERGENCE_PY}"


# -------------------------------------------------------------------------
# 2. README pseudocode presence
# -------------------------------------------------------------------------


def test_divergence_has_readme_pseudocode_label():
    """Module docstring must reference the README dirty-set algorithm section."""
    text = _module_text()
    assert "README" in text and "dirty-set algorithm" in text.lower(), (
        "divergence.py module docstring must reference the README dirty-set algorithm"
    )


def test_divergence_has_pseudocode_given_block():
    """Module docstring must include the 'Given:' pseudocode preamble."""
    text = _module_text()
    assert "Given:" in text, (
        "divergence.py must contain the pseudocode 'Given:' block from README"
    )


def test_divergence_has_pseudocode_trace_definition():
    """The pseudocode must define trace T as a sequence of steps."""
    text = _module_text()
    assert "trace T" in text or "T = [s0" in text, (
        "divergence.py must define trace T in the pseudocode"
    )


def test_divergence_has_pseudocode_dirty_set_D():
    """The pseudocode must define dirty set D."""
    text = _module_text()
    assert "D := {}" in text or "Compute dirty-set D" in text, (
        "divergence.py must define dirty set D in the pseudocode"
    )


def test_divergence_has_pseudocode_cache_hit_clause():
    """The pseudocode must include the cache-hit reuse branch."""
    text = _module_text()
    assert "reuse cached" in text, (
        "divergence.py must document the cache-hit reuse clause in pseudocode"
    )


def test_divergence_has_replay_cost_comment():
    """The pseudocode must state the replay cost claim."""
    text = _module_text()
    assert "Replay cost" in text, (
        "divergence.py must document the replay cost in the pseudocode block"
    )


def test_divergence_has_correctness_target():
    """The pseudocode must include the correctness target / soundness claim."""
    text = _module_text()
    assert "Correctness target" in text or "observationally equivalent" in text, (
        "divergence.py must state the correctness/soundness target"
    )


# -------------------------------------------------------------------------
# 3. Postconditions P1–P6
# -------------------------------------------------------------------------


def test_divergence_documents_all_postconditions():
    """All six postconditions (P1–P6) must be documented in the module."""
    text = _module_text()
    for label in ("P1", "P2", "P3", "P4", "P5", "P6"):
        assert label in text, (
            f"divergence.py must document postcondition {label}"
        )


def test_divergence_p1_parent_dirty_closure():
    """P1 (parent-dirty closure) must be named and explained."""
    text = _module_text()
    assert "P1" in text and ("parent" in text.lower() or "closure" in text.lower()), (
        "divergence.py P1 must describe parent-dirty closure"
    )


def test_divergence_p2_input_drift_inclusion():
    """P2 (input-drift inclusion) must be named and explained."""
    text = _module_text()
    assert "P2" in text and ("drift" in text.lower() or "input" in text.lower()), (
        "divergence.py P2 must describe input-drift inclusion"
    )


def test_divergence_p3_direct_substitution():
    """P3 (direct-substitution inclusion) must be named and explained."""
    text = _module_text()
    assert "P3" in text and "substitut" in text.lower(), (
        "divergence.py P3 must describe direct-substitution inclusion"
    )


def test_divergence_p4_clean_step_soundness():
    """P4 (clean-step soundness) must be named and explained."""
    text = _module_text()
    assert "P4" in text and "clean" in text.lower(), (
        "divergence.py P4 must describe clean-step soundness"
    )


def test_divergence_p5_minimality_empty_sigma():
    """P5 (minimality under empty sigma) must be named and explained."""
    text = _module_text()
    assert "P5" in text and ("empty" in text.lower() or "minimality" in text.lower()), (
        "divergence.py P5 must describe minimality under empty sigma"
    )


def test_divergence_p6_monotonicity():
    """P6 (monotonicity) must be named and explained."""
    text = _module_text()
    assert "P6" in text and "monoton" in text.lower(), (
        "divergence.py P6 must describe monotonicity"
    )


# -------------------------------------------------------------------------
# 4. Assumptions A1–A4
# -------------------------------------------------------------------------


def test_divergence_documents_all_assumptions():
    """All four assumptions (A1–A4) must be documented."""
    text = _module_text()
    for label in ("A1", "A2", "A3", "A4"):
        assert label in text, (
            f"divergence.py must document assumption {label}"
        )


def test_divergence_a1_collision_free_hashing():
    """A1 (collision-free hashing) must be stated."""
    text = _module_text()
    assert "A1" in text and ("collision" in text.lower() or "hash" in text.lower()), (
        "divergence.py A1 must state the collision-free hashing assumption"
    )


def test_divergence_a2_dependency_structure():
    """A2 (dependency structure via parent_step_id) must be stated."""
    text = _module_text()
    assert "A2" in text and "parent_step_id" in text, (
        "divergence.py A2 must reference parent_step_id dependency structure"
    )


def test_divergence_a3_inputs_hash_integrity():
    """A3 (recorded inputs_hash integrity) must be stated."""
    text = _module_text()
    assert "A3" in text and "inputs_hash" in text, (
        "divergence.py A3 must reference the recorded inputs_hash assumption"
    )


def test_divergence_a4_topological_order():
    """A4 (topological order) must be stated."""
    text = _module_text()
    assert "A4" in text and "topological" in text.lower(), (
        "divergence.py A4 must state the topological-order assumption"
    )


# -------------------------------------------------------------------------
# 5. Branch-aware propagation invariants B1–B3
# -------------------------------------------------------------------------


def test_divergence_documents_all_branch_invariants():
    """Branch-aware invariants B1–B3 must be documented."""
    text = _module_text()
    for label in ("B1", "B2", "B3"):
        assert label in text, (
            f"divergence.py must document branch invariant {label}"
        )


def test_divergence_b1_independent_fanout():
    """B1 (independent fan-out) must describe branch isolation."""
    text = _module_text()
    assert "B1" in text and ("fan-out" in text.lower() or "fanout" in text.lower() or "branch" in text.lower()), (
        "divergence.py B1 must describe independent fan-out branch isolation"
    )


def test_divergence_b2_join_dirtying():
    """B2 (join dirtying) must describe join-step dirty semantics."""
    text = _module_text()
    assert "B2" in text and "join" in text.lower(), (
        "divergence.py B2 must describe join-step dirty semantics"
    )


def test_divergence_b3_clean_sibling_preservation():
    """B3 (clean-sibling preservation) must be stated."""
    text = _module_text()
    assert "B3" in text and ("sibling" in text.lower() or "clean" in text.lower()), (
        "divergence.py B3 must describe clean-sibling preservation"
    )


# -------------------------------------------------------------------------
# 6. compute_dirty_set function-level docs
# -------------------------------------------------------------------------


def test_compute_dirty_set_docstring_has_preconditions():
    """``compute_dirty_set`` must have a Preconditions section in its docstring."""
    from stepback.divergence import compute_dirty_set
    doc = inspect.getdoc(compute_dirty_set) or ""
    assert "Precondition" in doc or "precondition" in doc, (
        "compute_dirty_set docstring must contain a Preconditions section"
    )


def test_compute_dirty_set_docstring_has_postconditions():
    """``compute_dirty_set`` must have a Postconditions section in its docstring."""
    from stepback.divergence import compute_dirty_set
    doc = inspect.getdoc(compute_dirty_set) or ""
    assert "Postcondition" in doc or "postcondition" in doc, (
        "compute_dirty_set docstring must contain a Postconditions section"
    )


def test_compute_dirty_set_docstring_references_p1_through_p6():
    """``compute_dirty_set`` docstring must reference P1 through P6."""
    from stepback.divergence import compute_dirty_set
    doc = inspect.getdoc(compute_dirty_set) or ""
    for label in ("P1", "P2", "P3", "P4", "P5", "P6"):
        assert label in doc, (
            f"compute_dirty_set docstring must reference postcondition {label}"
        )


def test_compute_dirty_set_docstring_mentions_topological_order():
    """Precondition A4 (topological order) must appear in the function docstring."""
    from stepback.divergence import compute_dirty_set
    doc = inspect.getdoc(compute_dirty_set) or ""
    assert "topological" in doc.lower() or "A4" in doc, (
        "compute_dirty_set docstring must state the topological-order precondition"
    )


def test_compute_dirty_set_docstring_mentions_inputs_hash():
    """Precondition A3 (inputs_hash integrity) must appear in the function docstring."""
    from stepback.divergence import compute_dirty_set
    doc = inspect.getdoc(compute_dirty_set) or ""
    assert "inputs_hash" in doc, (
        "compute_dirty_set docstring must reference inputs_hash (assumption A3)"
    )
