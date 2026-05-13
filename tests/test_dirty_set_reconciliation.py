"""Tests pinning the Step 67 dirty-set reconciliation conclusions.

Step 67 of ``100_STEPS.md`` reconciles the 12-step linear fixture result
(``dirty_after_sub=11``) with any README claim about "small dirty sets".
The resolution is:

1. ``compute_dirty_set`` (the conservative static classifier) correctly
   marks 11 of 12 steps dirty for the 12-step linear fixture under
   ``ToolOutputSubstitution(at_step="step:2")`` — the expected O(N−k)
   worst case for a linear chain.

2. Small dirty sets arise from **parallel branches** (B1–B3 sibling
   isolation) or **late substitutions**, not from universal properties of
   the algorithm.

3. The runtime replay engine (``run_replay``, Step 61 stale-cache
   detection) may show fewer dirty steps than the static classifier when
   re-executed outputs happen to match recorded outputs.

4. The current ``README.md`` makes no specific numeric dirty-set claim;
   it states only the asymptotic bound O(dirty_set) vs. O(N).

See ``docs/dirty-set.md §9`` for the full narrative.
"""
from __future__ import annotations

import os
import re
import tempfile

import pytest

from stepback import record, replay
from stepback.divergence import compute_dirty_set
from stepback.recorder import RecorderKey
from stepback.replay import Executor
from stepback.substitutions import PromptSubstitution, SubstitutionSet, ToolOutputSubstitution
from stepback.testing import (
    LOOKUP_BUG_ROW,
    LOOKUP_FIXED_ROW,
    fake_llm,
    fake_tool,
    run_recorded_agent,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ helpers


def _record_12_step_fixture():
    """Return a (path, tmp_dir) pair for a freshly recorded 12-step trace."""
    tmp = tempfile.mkdtemp(prefix="stepback-test-recon-")
    path = os.path.join(tmp, "trace.sb")
    with record(path, key=RecorderKey.fresh()) as rec:
        run_recorded_agent(rec)
    return path, tmp


# ------------------------------------------------------------------ static classifier tests


class TestLinearFixtureStaticDirtyCount:
    """compute_dirty_set gives 11 dirty steps on the linear 12-step fixture."""

    def test_linear_fixture_static_dirty_count_is_11(self, tmp_path):
        """The conservative classifier marks 11 of 12 steps dirty for an
        early ToolOutputSubstitution — the O(N−k) worst case for a linear
        chain (Step 67)."""
        path = str(tmp_path / "trace.sb")
        with record(path, key=RecorderKey.fresh()) as rec:
            run_recorded_agent(rec)

        t = replay(path)
        sub = ToolOutputSubstitution(
            at_step="step:2",
            fake_response={"result": LOOKUP_FIXED_ROW},
        )
        ds = compute_dirty_set(t, [sub])
        assert ds.step_count == 12
        assert ds.dirty_count == 11, (
            f"Expected 11 dirty steps (steps 2–12) for linear early substitution; "
            f"got dirty_count={ds.dirty_count}, dirty_ids={sorted(ds.dirty_ids)}"
        )

    def test_substituted_step_is_dirty(self, tmp_path):
        """The directly-substituted step (step:2) must be in the dirty set (P3)."""
        path = str(tmp_path / "trace.sb")
        with record(path, key=RecorderKey.fresh()) as rec:
            run_recorded_agent(rec)

        t = replay(path)
        sub = ToolOutputSubstitution(at_step="step:2", fake_response={"result": LOOKUP_FIXED_ROW})
        ds = compute_dirty_set(t, [sub])
        assert "step:2" in ds.dirty_ids

    def test_all_descendants_are_dirty(self, tmp_path):
        """P1 (parent-dirty closure): every step downstream of step:2 is dirty."""
        path = str(tmp_path / "trace.sb")
        with record(path, key=RecorderKey.fresh()) as rec:
            run_recorded_agent(rec)

        t = replay(path)
        sub = ToolOutputSubstitution(at_step="step:2", fake_response={"result": LOOKUP_FIXED_ROW})
        ds = compute_dirty_set(t, [sub])
        expected_dirty = {f"step:{i}" for i in range(2, 13)}
        assert set(ds.dirty_ids) == expected_dirty, (
            f"Expected steps 2–12 dirty; got {sorted(ds.dirty_ids)}"
        )

    def test_step1_is_not_dirty(self, tmp_path):
        """step:1 comes before the substitution; it must be clean."""
        path = str(tmp_path / "trace.sb")
        with record(path, key=RecorderKey.fresh()) as rec:
            run_recorded_agent(rec)

        t = replay(path)
        sub = ToolOutputSubstitution(at_step="step:2", fake_response={"result": LOOKUP_FIXED_ROW})
        ds = compute_dirty_set(t, [sub])
        assert "step:1" not in ds.dirty_ids

    def test_empty_substitution_gives_zero_dirty(self, tmp_path):
        """P5 (minimality under empty sigma): no dirty steps with no substitutions."""
        path = str(tmp_path / "trace.sb")
        with record(path, key=RecorderKey.fresh()) as rec:
            run_recorded_agent(rec)

        t = replay(path)
        ds = compute_dirty_set(t, [])
        assert ds.dirty_count == 0

    def test_early_vs_late_substitution_dirty_counts(self, tmp_path):
        """Late substitutions produce smaller dirty sets than early ones (O(N−k) bound)."""
        path = str(tmp_path / "trace.sb")
        with record(path, key=RecorderKey.fresh()) as rec:
            run_recorded_agent(rec)

        t = replay(path)
        early_sub = ToolOutputSubstitution(
            at_step="step:2", fake_response={"result": LOOKUP_FIXED_ROW}
        )
        late_sub = PromptSubstitution(
            at_step="step:11",
            new_messages=[{"role": "user", "content": "final step substitution"}],
        )
        ds_early = compute_dirty_set(t, [early_sub])
        ds_late = compute_dirty_set(t, [late_sub])
        assert ds_late.dirty_count < ds_early.dirty_count, (
            f"Late sub should dirty fewer steps than early sub: "
            f"late={ds_late.dirty_count} vs early={ds_early.dirty_count}"
        )


# ------------------------------------------------------------------ runtime vs. static


class TestRuntimeVsStaticClassifier:
    """run_replay (Step 61 stale-cache detection) may show fewer dirty steps than
    compute_dirty_set when re-executed outputs happen to match recorded outputs."""

    def test_static_classifier_is_upper_bound_of_runtime_dirty(self, tmp_path):
        """The static classifier's dirty count is always >= the runtime dirty count."""
        path = str(tmp_path / "trace.sb")
        with record(path, key=RecorderKey.fresh()) as rec:
            run_recorded_agent(rec)

        t = replay(path)
        sub = ToolOutputSubstitution(at_step="step:2", fake_response={"result": LOOKUP_FIXED_ROW})
        static_ds = compute_dirty_set(t, [sub])

        subs = SubstitutionSet()
        subs.add(ToolOutputSubstitution(at_step="step:2", fake_response={"result": LOOKUP_FIXED_ROW}))
        ex = Executor(llm=fake_llm, tool=fake_tool)
        t2 = replay(path)
        runtime_result = t2.run_replay(subs, ex)

        assert runtime_result.dirty_count <= static_ds.dirty_count, (
            f"Static classifier ({static_ds.dirty_count}) must be an upper bound "
            f"on runtime dirty ({runtime_result.dirty_count})"
        )

    def test_static_classifier_gives_11_not_runtime_dirty(self, tmp_path):
        """Regression guard: static dirty count is 11 regardless of runtime behaviour."""
        path = str(tmp_path / "trace.sb")
        with record(path, key=RecorderKey.fresh()) as rec:
            run_recorded_agent(rec)

        t = replay(path)
        sub = ToolOutputSubstitution(at_step="step:2", fake_response={"result": LOOKUP_FIXED_ROW})
        ds = compute_dirty_set(t, [sub])
        assert ds.dirty_count == 11


# ------------------------------------------------------------------ README claim audit


class TestReadmeDoesNotOverclaim:
    """The current README.md contains no stale numeric dirty-set claims."""

    def _readme_text(self) -> str:
        path = os.path.join(_REPO_ROOT, "README.md")
        with open(path, encoding="utf-8") as f:
            return f.read()

    def test_readme_has_no_median_dirty_set_of_3_claim(self):
        """The README must not claim 'median dirty-set of 3' as a universal property."""
        text = self._readme_text()
        # Pattern: "median dirty-set of <digit>" or "median dirty set of <digit>"
        pattern = r"median dirty.set of [0-9]"
        found = re.search(pattern, text, re.IGNORECASE)
        assert found is None, (
            f"README contains a stale 'median dirty-set of <N>' claim at "
            f"position {found.start()}: {found.group()!r}.  "
            f"See docs/dirty-set.md §9 for the correct reconciliation."
        )

    def test_readme_has_no_60x_cost_reduction_claim(self):
        """The README must not claim '>60× cost reduction' without context."""
        text = self._readme_text()
        pattern = r">60[×x]|>60 ?times"
        found = re.search(pattern, text, re.IGNORECASE)
        assert found is None, (
            f"README contains a stale '>60× cost reduction' claim: {found.group()!r}."
        )

    def test_readme_states_asymptotic_bound_not_specific_number(self):
        """The README's dirty-set claim should be expressed as O(dirty_set) vs O(N)."""
        text = self._readme_text()
        assert "O(dirty_set)" in text or "O(dirty" in text, (
            "README should express the dirty-set advantage as an asymptotic bound."
        )


# ------------------------------------------------------------------ doc invariants


class TestDirtySetDocReconciliation:
    """docs/dirty-set.md §9 contains the Step 67 reconciliation."""

    def _doc_text(self) -> str:
        path = os.path.join(_REPO_ROOT, "docs", "dirty-set.md")
        with open(path, encoding="utf-8") as f:
            return f.read()

    def test_section_9_exists(self):
        text = self._doc_text()
        assert "## 9." in text, "docs/dirty-set.md must have a §9 section"

    def test_reconciliation_mentions_worst_case(self):
        text = self._doc_text()
        assert "worst case" in text.lower(), (
            "§9 should explain that the 12-step linear fixture is the worst case"
        )

    def test_reconciliation_mentions_parallel_branches(self):
        text = self._doc_text()
        assert "parallel" in text.lower(), (
            "§9 should contrast with parallel-branch traces"
        )

    def test_reconciliation_mentions_o_n_minus_k(self):
        text = self._doc_text()
        assert "O(N" in text or "O(N−k)" in text, (
            "§9 should state the O(N−k) bound for linear chains"
        )

    def test_reconciliation_mentions_stale_cache(self):
        text = self._doc_text()
        assert "stale-cache" in text or "stale cache" in text.lower(), (
            "§9 should explain the stale-cache detection (Step 61) difference "
            "between static and runtime dirty counts"
        )

    def test_reconciliation_explains_parallel_3_steps(self):
        text = self._doc_text()
        assert "3 steps" in text or "3 / 11" in text or "3 of 11" in text, (
            "§9 should mention that the parallel-branch fixture dirties 3 steps"
        )

    def test_no_forward_reference_to_step_67(self):
        text = self._doc_text()
        assert "Step 67 of `100_STEPS.md` tracks reconciling" not in text, (
            "docs/dirty-set.md §9 should contain the actual reconciliation, "
            "not a forward reference to Step 67"
        )


# ------------------------------------------------------------------ bench-results docs


class TestBenchResultsDocsAreAccurate:
    """bench-results/README.md and docs/dirty-set-distributions.md must not
    incorrectly attribute the '>60× cost reduction' claim to the current README."""

    def _bench_readme(self) -> str:
        path = os.path.join(_REPO_ROOT, "bench-results", "README.md")
        with open(path, encoding="utf-8") as f:
            return f.read()

    def _distributions_doc(self) -> str:
        path = os.path.join(_REPO_ROOT, "docs", "dirty-set-distributions.md")
        with open(path, encoding="utf-8") as f:
            return f.read()

    def test_bench_readme_does_not_say_readme_advertises_60x(self):
        text = self._bench_readme()
        bad = "README.md` advertises a **>60"
        assert bad not in text, (
            "bench-results/README.md incorrectly attributes '>60× cost reduction' "
            "to the current README.md; the current README makes no such claim."
        )

    def test_distributions_doc_does_not_say_readme_advertises_60x(self):
        text = self._distributions_doc()
        bad = "README advertises a **>60"
        assert bad not in text, (
            "docs/dirty-set-distributions.md incorrectly attributes '>60× cost "
            "reduction' to the current README.md; the README does not say that."
        )
