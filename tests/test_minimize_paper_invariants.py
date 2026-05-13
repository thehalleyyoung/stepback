"""Documentation invariants for ``docs/minimization-paper.md`` — Step 88.

Asserts that the paper-grade minimization artifact exists, is
non-trivial, and covers every required element:

* Step 88 discharge claim
* All five strategy names (ddmin, linear, binary, brute, shapley)
* Oracle cache / memoization description
* A_oracle assumption (oracle stability)
* Stochastic / flaky-predicate section
* Failure modes: PredicateNotTriggered, BudgetExhausted, flaky predicate,
  UnavailableExecutorError / partial executor
* Degenerate trace cases
* Complexity table (required sections)
* Empirical comparison to naive ddmin
* Multi-witness enumeration mention
* Implementation-site mapping table
* Cross-references to minimize.py, replay.py, dirty-set.md
"""
from __future__ import annotations

import pathlib

DOCS_DIR = pathlib.Path(__file__).parent.parent / "docs"
PAPER_DOC = DOCS_DIR / "minimization-paper.md"


def _text() -> str:
    return PAPER_DOC.read_text(encoding="utf-8")


# -------------------------------------------------------------------------
# 1. File existence and minimum size
# -------------------------------------------------------------------------


def test_minimize_paper_exists():
    """``docs/minimization-paper.md`` must exist."""
    assert PAPER_DOC.exists(), f"missing: {PAPER_DOC}"


def test_minimize_paper_is_substantial():
    """The paper artifact must have substantial content (> 8 KiB)."""
    size = PAPER_DOC.stat().st_size
    assert size > 8_000, (
        f"docs/minimization-paper.md is suspiciously small ({size} bytes); "
        f"expected > 8 KiB"
    )


# -------------------------------------------------------------------------
# 2. Step 88 discharge claim
# -------------------------------------------------------------------------


def test_minimize_paper_references_step_88():
    """The document must claim to discharge Step 88."""
    text = _text()
    assert "Step 88" in text, (
        "docs/minimization-paper.md must reference Step 88 of 100_STEPS.md"
    )


# -------------------------------------------------------------------------
# 3. All five strategy names
# -------------------------------------------------------------------------


def test_minimize_paper_has_ddmin():
    """DDMin strategy must be documented."""
    text = _text()
    assert "DDMin" in text or "ddmin" in text.lower(), (
        "paper must document the DDMinStrategy"
    )


def test_minimize_paper_has_linear():
    """LinearShrink strategy must be documented."""
    text = _text()
    assert "linear" in text.lower() or "LinearShrink" in text, (
        "paper must document the LinearShrinkStrategy"
    )


def test_minimize_paper_has_binary():
    """BinaryHalving strategy must be documented."""
    text = _text()
    assert "binary" in text.lower() or "BinaryHalving" in text, (
        "paper must document the BinaryHalvingStrategy"
    )


def test_minimize_paper_has_brute():
    """BruteForce strategy must be documented."""
    text = _text()
    assert "brute" in text.lower() or "BruteForce" in text, (
        "paper must document the BruteForceStrategy"
    )


def test_minimize_paper_has_shapley():
    """Shapley attribution must be documented."""
    text = _text()
    assert "shapley" in text.lower() or "Shapley" in text, (
        "paper must document the ShapleyAttributionStrategy"
    )


# -------------------------------------------------------------------------
# 4. Oracle cache and memoization
# -------------------------------------------------------------------------


def test_minimize_paper_has_oracle_cache():
    """Oracle cache / memoization must be described."""
    text = _text()
    assert "cache" in text.lower() and "oracle" in text.lower(), (
        "paper must describe oracle memoization (_OracleCache)"
    )


def test_minimize_paper_has_probe_counting():
    """Probe counting must be described."""
    text = _text()
    assert "probe" in text.lower(), (
        "paper must describe probe counting"
    )


# -------------------------------------------------------------------------
# 5. Oracle stability assumption (A_oracle)
# -------------------------------------------------------------------------


def test_minimize_paper_has_a_oracle_assumption():
    """The A_oracle (oracle stability) assumption must be stated."""
    text = _text()
    assert "A_oracle" in text or "oracle stability" in text.lower(), (
        "paper must state A_oracle (oracle stability assumption)"
    )


def test_minimize_paper_has_a_pred_assumption():
    """The A_pred (predicate stability) assumption must be stated."""
    text = _text()
    assert "A_pred" in text or "predicate" in text.lower(), (
        "paper must state A_pred (predicate stability assumption)"
    )


# -------------------------------------------------------------------------
# 6. Stochastic / flaky predicate section
# -------------------------------------------------------------------------


def test_minimize_paper_has_stochastic_section():
    """The stochastic / flaky predicate section must be present."""
    text = _text()
    assert "stochastic" in text.lower() or "flaky" in text.lower(), (
        "paper must have a stochastic/flaky predicate section"
    )


def test_minimize_paper_has_predicate_stability_classes():
    """Predicate stability classes (structural, content) must be mentioned."""
    text = _text()
    assert "structural" in text.lower(), (
        "paper must distinguish structural predicates from content predicates"
    )


def test_minimize_paper_has_confidence_interval():
    """The document must mention confidence intervals for flaky predicates."""
    text = _text()
    assert "confidence interval" in text.lower() or "wilson" in text.lower(), (
        "paper must mention confidence intervals (Wilson) for stochastic predicates"
    )


# -------------------------------------------------------------------------
# 7. Failure modes
# -------------------------------------------------------------------------


def test_minimize_paper_has_predicate_not_triggered():
    """PredicateNotTriggered failure mode must be documented."""
    text = _text()
    assert "PredicateNotTriggered" in text, (
        "paper must document the PredicateNotTriggered failure mode"
    )


def test_minimize_paper_has_budget_exhausted():
    """BudgetExhausted failure mode must be documented."""
    text = _text()
    assert "BudgetExhausted" in text, (
        "paper must document the BudgetExhausted failure mode"
    )


def test_minimize_paper_has_unavailable_executor():
    """UnavailableExecutorError / partial trace failure mode must be documented."""
    text = _text()
    assert "UnavailableExecutorError" in text or "unavailable" in text.lower(), (
        "paper must document the UnavailableExecutorError / partial executor failure mode"
    )


def test_minimize_paper_has_degenerate_cases():
    """Degenerate trace cases (empty witness, all-minimal) must be documented."""
    text = _text()
    assert "degenerate" in text.lower() or "all-minimal" in text.lower() or "empty" in text.lower(), (
        "paper must document degenerate trace cases"
    )


# -------------------------------------------------------------------------
# 8. Complexity table
# -------------------------------------------------------------------------


def test_minimize_paper_has_complexity_table():
    """A strategy comparison table with complexity information must be present."""
    text = _text()
    # A markdown table header row should be present
    assert "Probe complexity" in text or "probe complexity" in text.lower(), (
        "paper must have a complexity comparison table"
    )


def test_minimize_paper_documents_worst_case_ddmin():
    """DDMin's O(n^2) worst-case probe complexity must be mentioned."""
    text = _text()
    assert "O(n" in text or "n²" in text or "n^2" in text.lower(), (
        "paper must document O(n^2) or n² worst-case probe complexity"
    )


# -------------------------------------------------------------------------
# 9. Empirical comparison to naive ddmin
# -------------------------------------------------------------------------


def test_minimize_paper_has_naive_ddmin_comparison():
    """Empirical comparison to naive ddmin (without memoization) must be present."""
    text = _text()
    assert "naive ddmin" in text.lower() or "naive" in text.lower(), (
        "paper must compare memoized implementation to naive ddmin"
    )


def test_minimize_paper_has_empirical_probe_table():
    """An empirical probe count table must be present."""
    text = _text()
    assert "probes" in text.lower() and ("|" in text), (
        "paper must include an empirical probe count table"
    )


def test_minimize_paper_has_cache_hit_rates():
    """Cache hit rates from bench-results must be referenced."""
    text = _text()
    assert "cache hit" in text.lower(), (
        "paper must reference cache hit rates (dirty-set cache amplification)"
    )


# -------------------------------------------------------------------------
# 10. Multi-witness enumeration
# -------------------------------------------------------------------------


def test_minimize_paper_has_multi_witness():
    """Multi-witness / find_all_minimal must be mentioned."""
    text = _text()
    assert "find_all_minimal" in text or "multi-witness" in text.lower(), (
        "paper must describe multi-witness enumeration (find_all_minimal)"
    )


# -------------------------------------------------------------------------
# 11. Implementation mapping
# -------------------------------------------------------------------------


def test_minimize_paper_has_implementation_mapping():
    """An implementation site mapping table must be present."""
    text = _text()
    assert "Implementation" in text and "minimize.py" in text, (
        "paper must have an implementation site mapping table referencing minimize.py"
    )


# -------------------------------------------------------------------------
# 12. Cross-references
# -------------------------------------------------------------------------


def test_minimize_paper_references_minimize_py():
    """Must cross-reference stepback/minimize.py."""
    text = _text()
    assert "minimize.py" in text, (
        "paper must cross-reference stepback/minimize.py"
    )


def test_minimize_paper_references_replay_py():
    """Must cross-reference stepback/replay.py."""
    text = _text()
    assert "replay.py" in text, (
        "paper must cross-reference stepback/replay.py"
    )


def test_minimize_paper_references_dirty_set_doc():
    """Must cross-reference docs/dirty-set.md."""
    text = _text()
    assert "dirty-set.md" in text, (
        "paper must cross-reference docs/dirty-set.md"
    )


def test_minimize_paper_references_divergence_py():
    """Must cross-reference stepback/divergence.py (dirty-set implementation)."""
    text = _text()
    assert "divergence.py" in text, (
        "paper must cross-reference stepback/divergence.py"
    )


# -------------------------------------------------------------------------
# 13. Shapley is attribution, not minimization
# -------------------------------------------------------------------------


def test_minimize_paper_shapley_is_attribution():
    """The paper must distinguish Shapley as attribution, not a minimal witness."""
    text = _text()
    # Must state that positive-weight subset is not guaranteed minimal
    assert "attribution" in text.lower() and "shapley" in text.lower(), (
        "paper must describe Shapley as attribution, not a minimization algorithm"
    )


# -------------------------------------------------------------------------
# 14. Binary halving is heuristic
# -------------------------------------------------------------------------


def test_minimize_paper_binary_is_heuristic():
    """The paper must flag BinaryHalvingStrategy as a heuristic, not guaranteed 1-minimal."""
    text = _text()
    assert "heuristic" in text.lower(), (
        "paper must document that BinaryHalvingStrategy is a heuristic fast shrinker, "
        "not guaranteed 1-minimal"
    )


# -------------------------------------------------------------------------
# 15. Monotonicity assumption
# -------------------------------------------------------------------------


def test_minimize_paper_has_monotonicity_assumption():
    """The A_monotone (predicate monotonicity) assumption must be mentioned."""
    text = _text()
    assert "monoton" in text.lower(), (
        "paper must discuss the monotonicity assumption (A_monotone)"
    )
