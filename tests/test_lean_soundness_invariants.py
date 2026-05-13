"""Invariant tests for the Lean 4 mechanized soundness proof — Step 56.

Checks that:

* ``proofs/lean/Stepback/Soundness.lean`` exists and contains the core
  theorems without any ``sorry``, ``axiom``, or ``partial`` keywords.
* ``proofs/lean/README.md`` exists, is substantial, and cross-references
  the paper proof and Step 56.
* ``proofs/lean/lakefile.lean`` and ``proofs/lean/lean-toolchain`` are
  present (offline, no Lean runtime required).
* ``.github/workflows/lean.yml`` exists and wires ``lake build`` for CI.
* ``docs/dirty-set-soundness.md`` cross-references the mechanized proof
  (``proofs/lean/``).
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).parent.parent
LEAN_DIR = ROOT / "proofs" / "lean"
SOUNDNESS_LEAN = LEAN_DIR / "Stepback" / "Soundness.lean"
LEAN_README = LEAN_DIR / "README.md"
LAKEFILE = LEAN_DIR / "lakefile.lean"
TOOLCHAIN = LEAN_DIR / "lean-toolchain"
LEAN_WORKFLOW = ROOT / ".github" / "workflows" / "lean.yml"
SOUNDNESS_DOC = ROOT / "docs" / "dirty-set-soundness.md"


def _lean_text() -> str:
    return SOUNDNESS_LEAN.read_text(encoding="utf-8")


def _readme_text() -> str:
    return LEAN_README.read_text(encoding="utf-8")


# -------------------------------------------------------------------------
# 1. File existence
# -------------------------------------------------------------------------


def test_lean_soundness_file_exists():
    """``proofs/lean/Stepback/Soundness.lean`` must exist."""
    assert SOUNDNESS_LEAN.exists(), f"missing: {SOUNDNESS_LEAN}"


def test_lean_readme_exists():
    """``proofs/lean/README.md`` must exist."""
    assert LEAN_README.exists(), f"missing: {LEAN_README}"


def test_lakefile_exists():
    """``proofs/lean/lakefile.lean`` must exist."""
    assert LAKEFILE.exists(), f"missing: {LAKEFILE}"


def test_lean_toolchain_exists():
    """``proofs/lean/lean-toolchain`` must exist and pin a specific version."""
    assert TOOLCHAIN.exists(), f"missing: {TOOLCHAIN}"


def test_lean_workflow_exists():
    """``.github/workflows/lean.yml`` must exist for CI."""
    assert LEAN_WORKFLOW.exists(), f"missing: {LEAN_WORKFLOW}"


# -------------------------------------------------------------------------
# 2. Trust base: no sorry / axiom / partial
# -------------------------------------------------------------------------


def _non_comment_lines(text: str) -> list[tuple[int, str]]:
    """Return (lineno, text) pairs for lines that are not Lean comments.

    Strips line comments (``-- ...``) and is conservative: a line whose
    non-whitespace content starts with ``--`` is a comment line.  Block
    comment lines (inside ``/- ... -/``) are also excluded.
    """
    result = []
    in_block = False
    for i, ln in enumerate(text.splitlines(), start=1):
        stripped = ln.lstrip()
        if not in_block:
            if stripped.startswith("/-"):
                in_block = True
                if "-/" in stripped[2:]:
                    in_block = False  # single-line block comment
                continue
            if stripped.startswith("--"):
                continue
            # Strip inline line comment before checking.
            code = re.sub(r"--.*$", "", ln)
            result.append((i, code))
        else:
            if "-/" in stripped:
                in_block = False
    return result


def test_lean_proof_has_no_sorry():
    """The proof must not contain ``sorry`` in code (incomplete proof placeholder)."""
    code_lines = _non_comment_lines(_lean_text())
    bad = [(no, ln) for no, ln in code_lines if re.search(r"\bsorry\b", ln)]
    assert not bad, (
        "Soundness.lean contains 'sorry' in code — proof is incomplete:\n"
        + "\n".join(f"  line {no}: {ln}" for no, ln in bad)
    )


def test_lean_proof_has_no_top_level_axiom():
    """The proof must not introduce bare ``axiom`` declarations."""
    code_lines = _non_comment_lines(_lean_text())
    bad = [(no, ln) for no, ln in code_lines if re.match(r"\s*axiom\s+", ln)]
    assert not bad, (
        "Soundness.lean contains a bare 'axiom' declaration:\n"
        + "\n".join(f"  line {no}: {ln}" for no, ln in bad)
    )


def test_lean_proof_has_no_partial():
    """The proof must not use ``partial`` to bypass termination checking."""
    code_lines = _non_comment_lines(_lean_text())
    bad = [(no, ln) for no, ln in code_lines if re.match(r"\s*partial\s+", ln)]
    assert not bad, (
        "Soundness.lean contains a 'partial' definition:\n"
        + "\n".join(f"  line {no}: {ln}" for no, ln in bad)
    )


# -------------------------------------------------------------------------
# 3. Core definitions present
# -------------------------------------------------------------------------


def test_lean_defines_trace_structure():
    """The ``Trace`` structure must be defined in the proof file."""
    assert "structure Trace" in _lean_text(), (
        "Soundness.lean must define the 'Trace' structure"
    )


def test_lean_defines_depends_only_on_parents():
    """``DependsOnlyOnParents`` (A1) must be defined."""
    assert "DependsOnlyOnParents" in _lean_text(), (
        "Soundness.lean must define 'DependsOnlyOnParents' (assumption A1)"
    )


def test_lean_defines_recorder_coherent():
    """``RecorderCoherent`` (R1 ∧ R3) must be defined."""
    assert "RecorderCoherent" in _lean_text(), (
        "Soundness.lean must define 'RecorderCoherent' (recorder obligation R1 ∧ R3)"
    )


def test_lean_defines_clean_sound():
    """``CleanSound`` (classifier soundness side-condition) must be defined."""
    assert "CleanSound" in _lean_text(), (
        "Soundness.lean must define 'CleanSound'"
    )


def test_lean_defines_is_full_replay():
    """``IsFullReplay`` (Definition 1) must be defined."""
    assert "IsFullReplay" in _lean_text(), (
        "Soundness.lean must define 'IsFullReplay'"
    )


def test_lean_defines_is_dirty_replay():
    """``IsDirtyReplay`` (Definition 2) must be defined."""
    assert "IsDirtyReplay" in _lean_text(), (
        "Soundness.lean must define 'IsDirtyReplay'"
    )


# -------------------------------------------------------------------------
# 4. Core theorems present
# -------------------------------------------------------------------------


def test_lean_has_soundness_theorem():
    """``theorem soundness`` (Theorem 1 / P1) must be present."""
    text = _lean_text()
    assert "theorem soundness" in text, (
        "Soundness.lean must contain 'theorem soundness' (Theorem 1)"
    )


def test_lean_has_soundness_aux():
    """``theorem soundness_aux`` (strong-induction core) must be present."""
    assert "soundness_aux" in _lean_text(), (
        "Soundness.lean must contain 'soundness_aux' (strong-induction core)"
    )


def test_lean_has_cache_reuse_safe_corollary():
    """``theorem cache_reuse_safe`` (Corollary 1) must be present."""
    assert "cache_reuse_safe" in _lean_text(), (
        "Soundness.lean must contain 'cache_reuse_safe' (Corollary 1)"
    )


def test_lean_uses_strong_induction():
    """The proof must use strong / well-founded induction (``Nat.strongRecOn`` or equiv)."""
    text = _lean_text()
    assert "strongRec" in text or "WellFounded" in text or "Nat.rec" in text, (
        "Soundness.lean must use strong / well-founded induction"
    )


# -------------------------------------------------------------------------
# 5. README quality checks
# -------------------------------------------------------------------------


def test_lean_readme_references_step_56():
    """The README must reference Step 56."""
    text = _readme_text()
    assert "Step 56" in text or "step 56" in text.lower(), (
        "proofs/lean/README.md must reference Step 56 of 100_STEPS.md"
    )


def test_lean_readme_is_substantial():
    """The README must have non-trivial content (> 1 KiB)."""
    size = LEAN_README.stat().st_size
    assert size > 1_000, (
        f"proofs/lean/README.md is suspiciously small ({size} bytes)"
    )


def test_lean_readme_cross_references_soundness_doc():
    """The README must cross-reference the paper proof document."""
    text = _readme_text()
    assert "dirty-set-soundness" in text or "soundness.md" in text.lower(), (
        "proofs/lean/README.md must cross-reference docs/dirty-set-soundness.md"
    )


def test_lean_readme_lists_trust_base():
    """The README must explain the trust base."""
    text = _readme_text()
    assert "trust base" in text.lower() or "Trust base" in text, (
        "proofs/lean/README.md must document the trust base"
    )


def test_lean_readme_has_python_cross_reference_table():
    """The README must map Lean symbols to Python counterparts."""
    text = _readme_text()
    assert "Python" in text or "divergence.py" in text or "replay.py" in text, (
        "proofs/lean/README.md must map Lean symbols to the Python implementation"
    )


# -------------------------------------------------------------------------
# 6. Toolchain pin
# -------------------------------------------------------------------------


def test_lean_toolchain_pins_specific_version():
    """The toolchain file must pin a concrete Lean version (``v4.``...)."""
    text = TOOLCHAIN.read_text(encoding="utf-8").strip()
    assert re.search(r"v4\.\d+\.\d+", text), (
        f"lean-toolchain must pin a concrete 'v4.x.y' version; got: {text!r}"
    )


# -------------------------------------------------------------------------
# 7. CI workflow quality
# -------------------------------------------------------------------------


def test_lean_workflow_runs_lake_build():
    """The CI workflow must run ``lake build``."""
    text = LEAN_WORKFLOW.read_text(encoding="utf-8")
    assert "lake build" in text, (
        ".github/workflows/lean.yml must run 'lake build'"
    )


def test_lean_workflow_rejects_sorry():
    """The CI workflow must grep for ``sorry`` and fail if found."""
    text = LEAN_WORKFLOW.read_text(encoding="utf-8")
    assert "sorry" in text, (
        ".github/workflows/lean.yml must check for 'sorry' in proof sources"
    )


# -------------------------------------------------------------------------
# 8. Paper proof doc cross-references the mechanized proof
# -------------------------------------------------------------------------


def test_soundness_doc_references_lean_proof():
    """``docs/dirty-set-soundness.md`` must link to the Lean proof."""
    text = SOUNDNESS_DOC.read_text(encoding="utf-8")
    assert "proofs/lean" in text or "Soundness.lean" in text, (
        "docs/dirty-set-soundness.md must cross-reference proofs/lean/Stepback/Soundness.lean"
    )


def test_soundness_doc_references_step_56():
    """``docs/dirty-set-soundness.md`` must reference Step 56."""
    text = SOUNDNESS_DOC.read_text(encoding="utf-8")
    assert "Step 56" in text or "step 56" in text.lower(), (
        "docs/dirty-set-soundness.md must reference Step 56 (mechanized proof)"
    )
