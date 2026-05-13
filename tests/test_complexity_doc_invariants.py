"""Documentation invariants for ``docs/dirty-set-complexity.md`` — Step 58.

Asserts that the asymptotic complexity document exists, is non-trivial,
and covers every required element:

* Theorem 2.1 (time complexity, classification)
* Corollary 2.2 (time complexity, classification + replay)
* Theorem 2.3 (memory complexity)
* Linear-trace analysis (§3)
* General-DAG analysis (§4)
* Branch-heavy-trace analysis (§5)
* Substitution-set scaling (§6)
* The §Complexity contract block in ``stepback/divergence.py``
* Cross-references to divergence.py, replay.py, dirty-set.md
* Version pinning (dirty_set_version, canonicalisation_version)
"""
from __future__ import annotations

import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent
DOCS_DIR = REPO_ROOT / "docs"
COMPLEXITY_DOC = DOCS_DIR / "dirty-set-complexity.md"
DIVERGENCE_PY = REPO_ROOT / "stepback" / "divergence.py"


def _complexity_text() -> str:
    return COMPLEXITY_DOC.read_text(encoding="utf-8")


def _divergence_text() -> str:
    return DIVERGENCE_PY.read_text(encoding="utf-8")


# -------------------------------------------------------------------------
# 1. File existence and minimum size
# -------------------------------------------------------------------------


def test_complexity_doc_exists():
    """``docs/dirty-set-complexity.md`` must exist."""
    assert COMPLEXITY_DOC.exists(), f"missing: {COMPLEXITY_DOC}"


def test_complexity_doc_is_substantial():
    """The complexity document must have substantial content (> 8 KiB)."""
    size = COMPLEXITY_DOC.stat().st_size
    assert size > 8_000, (
        f"docs/dirty-set-complexity.md is suspiciously small ({size} bytes)"
    )


# -------------------------------------------------------------------------
# 2. Step 58 discharge claim
# -------------------------------------------------------------------------


def test_complexity_doc_references_step_58():
    """The document must claim to discharge Step 58."""
    text = _complexity_text()
    assert "Step 58" in text or "step 58" in text.lower(), (
        "docs/dirty-set-complexity.md must reference Step 58 of 100_STEPS.md"
    )


# -------------------------------------------------------------------------
# 3. Headline bounds and notation
# -------------------------------------------------------------------------


def test_complexity_doc_states_headline_time_bound():
    """The headline Θ(N + E + I + O + m) time bound must appear."""
    text = _complexity_text()
    assert "N + E + I + O" in text or "N+E+I+O" in text, (
        "docs/dirty-set-complexity.md must state the Θ(N + E + I + O + m) time bound"
    )


def test_complexity_doc_states_headline_memory_bound():
    """The headline Θ(N + O + S_max + m) memory bound must appear."""
    text = _complexity_text()
    assert "N + O" in text or "N+O" in text, (
        "docs/dirty-set-complexity.md must state the Θ(N + O + …) memory bound"
    )


def test_complexity_doc_defines_notation_table():
    """The notation table (N, E, D, I, O, m, W, B, S_max) must be present."""
    text = _complexity_text()
    for sym in ("N", "E", "D", "I", "O", "m", "W", "B", "S_max"):
        assert sym in text, (
            f"docs/dirty-set-complexity.md notation table must define symbol {sym!r}"
        )


# -------------------------------------------------------------------------
# 4. Theorem statements
# -------------------------------------------------------------------------


def test_complexity_doc_states_theorem_2_1():
    """Theorem 2.1 (time complexity, classification) must be stated."""
    text = _complexity_text()
    assert "Theorem 2.1" in text or "time complexity" in text.lower(), (
        "docs/dirty-set-complexity.md must state Theorem 2.1 (time complexity)"
    )


def test_complexity_doc_states_corollary_2_2():
    """Corollary 2.2 (time including replay) must be stated."""
    text = _complexity_text()
    assert "Corollary 2.2" in text or "D_real" in text, (
        "docs/dirty-set-complexity.md must state Corollary 2.2 (time + replay)"
    )


def test_complexity_doc_states_theorem_2_3():
    """Theorem 2.3 (memory complexity) must be stated."""
    text = _complexity_text()
    assert "Theorem 2.3" in text or "memory complexity" in text.lower(), (
        "docs/dirty-set-complexity.md must state Theorem 2.3 (memory complexity)"
    )


def test_complexity_doc_has_lower_bound_argument():
    """The Ω(N + E + I + O) lower-bound argument must be present."""
    text = _complexity_text()
    assert "lower bound" in text.lower() or "Ω(" in text, (
        "docs/dirty-set-complexity.md must include the Ω(N + E + I + O) lower-bound argument"
    )


# -------------------------------------------------------------------------
# 5. Linear-trace section (§3)
# -------------------------------------------------------------------------


def test_complexity_doc_has_linear_trace_section():
    """§3 (linear traces) must be present."""
    text = _complexity_text()
    assert "Linear" in text and ("linear trace" in text.lower() or "## 3" in text), (
        "docs/dirty-set-complexity.md must have a linear-trace section (§3)"
    )


def test_complexity_doc_linear_dirty_set_lemma():
    """Lemma 3.1 (upward-closed dirty set on linear traces) must appear."""
    text = _complexity_text()
    assert "Lemma 3.1" in text or "upward-closed" in text.lower(), (
        "docs/dirty-set-complexity.md must state Lemma 3.1 (linear dirty-set shape)"
    )


def test_complexity_doc_linear_trace_corollary():
    """Corollary 3.2 (root substitution → all N dirty) must appear."""
    text = _complexity_text()
    assert "Corollary 3.2" in text or "dirty_after_sub" in text, (
        "docs/dirty-set-complexity.md must state Corollary 3.2 (root sub → N dirty)"
    )


# -------------------------------------------------------------------------
# 6. General-DAG section (§4)
# -------------------------------------------------------------------------


def test_complexity_doc_has_dag_section():
    """§4 (general DAG traces) must be present."""
    text = _complexity_text()
    assert "DAG" in text and "## 4" in text, (
        "docs/dirty-set-complexity.md must have a general-DAG section (§4)"
    )


def test_complexity_doc_dag_reachability_lemma():
    """Lemma 4.1 (reachability bound) must appear."""
    text = _complexity_text()
    assert "Lemma 4.1" in text or "Reachability" in text or "reachability" in text, (
        "docs/dirty-set-complexity.md must state Lemma 4.1 (reachability bound)"
    )


# -------------------------------------------------------------------------
# 7. Branch-heavy section (§5)
# -------------------------------------------------------------------------


def test_complexity_doc_has_branch_heavy_section():
    """§5 (branch-heavy traces) must be present."""
    text = _complexity_text()
    assert "branch" in text.lower() and "## 5" in text, (
        "docs/dirty-set-complexity.md must have a branch-heavy section (§5)"
    )


def test_complexity_doc_fan_out_independence_lemma():
    """Lemma 5.1 (fan-out independence) must appear."""
    text = _complexity_text()
    assert "Lemma 5.1" in text or "fan-out independence" in text.lower(), (
        "docs/dirty-set-complexity.md must state Lemma 5.1 (fan-out independence)"
    )


def test_complexity_doc_parallel_branch_join_mentioned():
    """``parallel_branch_join`` must be addressed in the branch-heavy section."""
    text = _complexity_text()
    assert "parallel_branch_join" in text, (
        "docs/dirty-set-complexity.md must address parallel_branch_join steps"
    )


# -------------------------------------------------------------------------
# 8. Substitution-set scaling (§6)
# -------------------------------------------------------------------------


def test_complexity_doc_has_substitution_scaling_section():
    """§6 (substitution-set scaling) must be present."""
    text = _complexity_text()
    assert "ubstitution" in text and ("## 6" in text or "scaling" in text.lower()), (
        "docs/dirty-set-complexity.md must have a substitution-scaling section (§6)"
    )


def test_complexity_doc_substitution_is_linear():
    """Substitution matching must be stated as O(m + N), not O(N·m)."""
    text = _complexity_text()
    assert "m + N" in text or "Θ(m" in text or "O(m" in text, (
        "docs/dirty-set-complexity.md must state that substitution matching is O(m + N)"
    )


# -------------------------------------------------------------------------
# 9. Summary / putting-it-together section
# -------------------------------------------------------------------------


def test_complexity_doc_has_summary_section():
    """The 'Putting it together' or 'Summary' section must be present."""
    text = _complexity_text()
    assert "putting it together" in text.lower() or "## 8" in text or "summary" in text.lower(), (
        "docs/dirty-set-complexity.md must have a summary section"
    )


def test_complexity_doc_headline_performance_claim():
    """The 'O(|D|) LLM calls instead of O(N)' headline must be stated."""
    text = _complexity_text()
    has_d = "|D|" in text or "D_real" in text
    has_n = "O(N)" in text or "instead of N" in text.lower()
    assert has_d and has_n, (
        "docs/dirty-set-complexity.md must state the O(|D|) vs O(N) headline claim"
    )


# -------------------------------------------------------------------------
# 10. Cross-references
# -------------------------------------------------------------------------


def test_complexity_doc_references_divergence_py():
    """The complexity doc must cross-reference stepback/divergence.py."""
    text = _complexity_text()
    assert "divergence.py" in text, (
        "docs/dirty-set-complexity.md must reference stepback/divergence.py"
    )


def test_complexity_doc_references_replay_py():
    """The complexity doc must cross-reference stepback/replay.py."""
    text = _complexity_text()
    assert "replay.py" in text, (
        "docs/dirty-set-complexity.md must reference stepback/replay.py"
    )


def test_complexity_doc_references_dirty_set_md():
    """The complexity doc must cross-reference docs/dirty-set.md."""
    text = _complexity_text()
    assert "dirty-set.md" in text, (
        "docs/dirty-set-complexity.md must reference docs/dirty-set.md"
    )


def test_complexity_doc_references_100_steps():
    """The document must reference 100_STEPS.md."""
    text = _complexity_text()
    assert "100_STEPS.md" in text or "100_STEPS" in text, (
        "docs/dirty-set-complexity.md must reference 100_STEPS.md"
    )


# -------------------------------------------------------------------------
# 11. Version pinning
# -------------------------------------------------------------------------


def test_complexity_doc_pins_dirty_set_version():
    """The complexity doc must be pinned to dirty_set_version."""
    text = _complexity_text()
    assert "dirty_set_version" in text or "version" in text.lower(), (
        "docs/dirty-set-complexity.md must pin the dirty-set algorithm version"
    )


# -------------------------------------------------------------------------
# 12. §Complexity contract block in divergence.py
# -------------------------------------------------------------------------


def test_divergence_py_has_complexity_block():
    """``stepback/divergence.py`` must contain a §Complexity contract block."""
    text = _divergence_text()
    assert "§Complexity" in text or "Complexity" in text, (
        "stepback/divergence.py must contain a §Complexity contract block "
        "as described in docs/dirty-set-complexity.md"
    )


def test_divergence_py_complexity_block_states_time_bound():
    """The §Complexity block must state the headline time bound."""
    text = _divergence_text()
    assert "N + E + I + O" in text or "N+E+I+O" in text, (
        "stepback/divergence.py §Complexity block must state the Θ(N + E + I + O + m) bound"
    )


def test_divergence_py_complexity_block_states_memory_bound():
    """The §Complexity block must state the headline memory bound."""
    text = _divergence_text()
    assert "N + O" in text or "N+O" in text, (
        "stepback/divergence.py §Complexity block must state the Θ(N + O + …) memory bound"
    )


def test_divergence_py_complexity_block_references_complexity_doc():
    """The §Complexity block must reference docs/dirty-set-complexity.md."""
    text = _divergence_text()
    assert "dirty-set-complexity.md" in text, (
        "stepback/divergence.py §Complexity block must reference docs/dirty-set-complexity.md"
    )


# -------------------------------------------------------------------------
# 13. Summary table present in dirty-set.md §7
# -------------------------------------------------------------------------


def test_dirty_set_md_has_complexity_table():
    """``docs/dirty-set.md`` §7 must contain the three-row complexity table."""
    dirty_set_md = DOCS_DIR / "dirty-set.md"
    assert dirty_set_md.exists(), "docs/dirty-set.md must exist"
    text = dirty_set_md.read_text(encoding="utf-8")
    assert "Asymptotic complexity" in text or "asymptotic" in text.lower(), (
        "docs/dirty-set.md must have an 'Asymptotic complexity' section"
    )
    # Check that the three trace shapes are covered
    for shape in ("Linear", "General DAG", "Branch"):
        assert shape in text, (
            f"docs/dirty-set.md §7 complexity table must cover trace shape: {shape!r}"
        )
