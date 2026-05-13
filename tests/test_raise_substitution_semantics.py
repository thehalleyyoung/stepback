"""Normative tests pinning the RaiseSubstitution semantics specified in
docs/dirty-set.md §5.6.

These tests verify:
  1. The substituted step is always dirty (P3), has the error sentinel as
     its output, and incurs zero cost.
  2. The directly downstream step (with "context" dependency on the
     substituted step) is classified dirty via input_drift by compute_dirty_set.
  3. compute_dirty_set is a sound over-approximation of replay_forward:
     replay dirty ⊆ classifier dirty (not equality, because the classifier
     uses sentinel hashes for input-drift steps while replay uses real
     re-execution outputs for rebinding).
  4. A RaiseSubstitution inside one parallel branch does NOT dirty sibling
     branches — only the targeted branch, the join, and post-join steps.
  5. The "no executor call" contract: the substituted step is output-forcing
     and never triggers a real LLM or tool callback.
"""
from __future__ import annotations

import os

from stepback import RecorderKey, record, replay
from stepback.divergence import compute_dirty_set
from stepback.replay import Executor
from stepback.substitutions import RaiseSubstitution
from stepback.testing import fake_llm, fake_tool, run_recorded_agent
from stepback.testing import parallel_fake_llm, parallel_fake_tool, run_parallel_agent


# ──────────────────────────────────── helpers ───────────────────────────────

def _record_linear(tmp_path: str) -> str:
    path = os.path.join(tmp_path, "linear.sb")
    key = RecorderKey.fresh()
    with record(path, key=key) as rec:
        run_recorded_agent(rec)
    return path


def _record_parallel(tmp_path: str) -> str:
    path = os.path.join(tmp_path, "parallel.sb")
    key = RecorderKey.fresh()
    with record(path, key=key) as rec:
        run_parallel_agent(rec)
    return path


def _linear_executor() -> Executor:
    return Executor(llm=fake_llm, tool=fake_tool)


def _parallel_executor() -> Executor:
    return Executor(llm=parallel_fake_llm, tool=parallel_fake_tool)


# ──────────────────────── 1. Targeted step classification ───────────────────

class TestTargetedStep:
    """The substituted step is always dirty, outputs the error sentinel, costs 0."""

    def test_targeted_step_is_dirty(self, tmp_path):
        t = replay(_record_linear(str(tmp_path)))
        t.substitute(RaiseSubstitution(at_step="step:2", exception_type="TimeoutError"))
        r = t.replay_forward(executor=_linear_executor())
        s2 = next(s for s in r.steps if s.step_id == "step:2")
        assert s2.dirty is True
        assert s2.cache_hit is False

    def test_targeted_step_has_error_sentinel_output(self, tmp_path):
        t = replay(_record_linear(str(tmp_path)))
        t.substitute(
            RaiseSubstitution(at_step="step:2", exception_type="ValueError", message="bad input")
        )
        r = t.replay_forward(executor=_linear_executor())
        s2 = next(s for s in r.steps if s.step_id == "step:2")
        assert s2.outputs == {"__error__": {"type": "ValueError", "message": "bad input"}}

    def test_targeted_step_zero_cost(self, tmp_path):
        t = replay(_record_linear(str(tmp_path)))
        t.substitute(RaiseSubstitution(at_step="step:2", exception_type="APIError"))
        r = t.replay_forward(executor=_linear_executor())
        s2 = next(s for s in r.steps if s.step_id == "step:2")
        assert s2.cost_usd == 0.0

    def test_dirty_reason_is_substituted_in_classifier(self, tmp_path):
        """compute_dirty_set must report dirty_reason='substituted' for the target."""
        t = replay(_record_linear(str(tmp_path)))
        subs = [RaiseSubstitution(at_step="step:2", exception_type="TimeoutError")]
        summary = compute_dirty_set(t, subs)
        entry = next(e for e in summary.entries if e.step_id == "step:2")
        assert entry.dirty is True
        assert entry.dirty_reason == "substituted"

    def test_targets_any_step_kind(self, tmp_path):
        """RaiseSubstitution may target any step_kind, including llm_call (step:1)."""
        t = replay(_record_linear(str(tmp_path)))
        t.substitute(RaiseSubstitution(at_step="step:1", exception_type="BudgetExceeded"))
        r = t.replay_forward(executor=_linear_executor())
        s1 = next(s for s in r.steps if s.step_id == "step:1")
        assert s1.dirty is True
        assert s1.outputs == {"__error__": {"type": "BudgetExceeded", "message": ""}}


# ──────────────────────── 2. Downstream cache invalidation ──────────────────

class TestDownstreamInvalidation:
    """The directly downstream step (with "context" dependency on the substituted
    step) has its cache entry invalidated because its canonical inputs hash
    changes: its "context" field is rebound to H(error_sentinel) which differs
    from H(recorded_output)."""

    def test_direct_downstream_step_is_dirty_in_classifier(self, tmp_path):
        """compute_dirty_set must classify the direct downstream step as dirty
        with dirty_reason='input_drift' when step:2 gets a RaiseSubstitution.
        step:3 has context = H(step:2.outputs), which changes to
        H({"__error__": ...}) → hash drift → dirty."""
        t = replay(_record_linear(str(tmp_path)))
        subs = [RaiseSubstitution(at_step="step:2", exception_type="TimeoutError")]
        summary = compute_dirty_set(t, subs)
        s3 = next(e for e in summary.entries if e.step_id == "step:3")
        assert s3.dirty is True
        assert s3.dirty_reason == "input_drift", (
            f"step:3 should be dirty via input_drift (context hash changed), "
            f"got {s3.dirty_reason!r}"
        )

    def test_direct_downstream_step_is_dirty_in_replay(self, tmp_path):
        """replay_forward must also mark step:3 dirty after RaiseSubstitution at step:2.
        The forced error sentinel's hash differs from step:2's recorded output hash,
        so step:3's "context" field is rebound to a different value."""
        t = replay(_record_linear(str(tmp_path)))
        t.substitute(RaiseSubstitution(at_step="step:2", exception_type="TimeoutError"))
        r = t.replay_forward(executor=_linear_executor())
        s3 = next(s for s in r.steps if s.step_id == "step:3")
        assert s3.dirty is True, (
            "step:3 must be dirty: its 'context' = H(step:2.outputs) changes to "
            "H({'__error__': ...}), causing input-hash drift"
        )

    def test_steps_before_substitution_are_clean(self, tmp_path):
        """Steps recorded before the substituted step must remain cache hits."""
        t = replay(_record_linear(str(tmp_path)))
        t.substitute(RaiseSubstitution(at_step="step:6", exception_type="NetworkError"))
        r = t.replay_forward(executor=_linear_executor())
        for step_n in range(1, 6):
            s = next(s for s in r.steps if s.step_id == f"step:{step_n}")
            assert s.cache_hit is True, (
                f"step:{step_n} should be clean — it was recorded before the substitution"
            )

    def test_no_real_executor_call_for_substituted_step(self, tmp_path):
        """The substituted step is output-forcing: the real executor is not called."""
        t = replay(_record_linear(str(tmp_path)))
        call_log = []

        def counting_tool(name, args):
            call_log.append(("tool", name))
            return fake_tool(name, args)

        t.substitute(RaiseSubstitution(at_step="step:2", exception_type="TimeoutError"))
        executor = Executor(llm=fake_llm, tool=counting_tool)
        t.replay_forward(executor=executor)
        # step:2 is a tool_call to "lookup_customer"; it must NOT appear in the log.
        tool_names_called = [name for _, name in call_log]
        assert "lookup_customer" not in tool_names_called, (
            "RaiseSubstitution must not invoke the real tool for the substituted step"
        )

    def test_classifier_marks_all_downstream_steps_dirty(self, tmp_path):
        """compute_dirty_set uses sentinel output hashes for input-drift steps, so
        all N-k steps after the substitution point are classified dirty (conservative
        over-approximation). This is P1 parent-dirty closure on the sentinel chain.

        Note: replay_forward may classify fewer steps dirty because it uses actual
        re-execution outputs for rebinding; if a re-executed step produces the same
        output as recorded, its downstream steps remain clean. This is expected and
        correct — compute_dirty_set is a sound over-approximation."""
        t = replay(_record_linear(str(tmp_path)))
        subs = [RaiseSubstitution(at_step="step:2", exception_type="TimeoutError")]
        summary = compute_dirty_set(t, subs)
        dirty_ids = {e.step_id for e in summary.entries if e.dirty}
        # All steps from step:2 onwards should be dirty in the classifier.
        for n in range(2, 13):
            assert f"step:{n}" in dirty_ids, (
                f"step:{n} should be dirty in compute_dirty_set "
                f"(sentinel hash chain propagation)"
            )


# ──────────────────────── 3. Classifier is sound over-approximation ─────────

class TestClassifierSoundness:
    """replay_forward dirty ⊆ compute_dirty_set dirty.

    The classifier marks at least as many steps dirty as replay_forward.
    It is not required to be exact (it can be more conservative).
    But if replay says clean, classifier must also say clean (soundness)."""

    def test_replay_dirty_subset_of_classifier_dirty(self, tmp_path):
        path = _record_linear(str(tmp_path))
        t_classify = replay(path)
        t_replay = replay(path)
        sub = RaiseSubstitution(at_step="step:2", exception_type="TimeoutError")
        summary = compute_dirty_set(t_classify, [sub])
        t_replay.substitute(sub)
        r = t_replay.replay_forward(executor=_linear_executor())

        classified_dirty = {e.step_id for e in summary.entries if e.dirty}
        replayed_dirty = {s.step_id for s in r.steps if s.dirty}

        # Every step that replay marks dirty must also be marked dirty by classifier.
        unexpected_clean = replayed_dirty - classified_dirty
        assert not unexpected_clean, (
            f"Soundness violation: replay marked these dirty but classifier said clean: "
            f"{unexpected_clean}. The classifier must be a sound over-approximation."
        )

    def test_classifier_clean_implies_replay_clean(self, tmp_path):
        """If compute_dirty_set says a step is clean, replay_forward must not dirty it."""
        path = _record_linear(str(tmp_path))
        t_classify = replay(path)
        t_replay = replay(path)
        sub = RaiseSubstitution(at_step="step:6", exception_type="NetworkError")
        summary = compute_dirty_set(t_classify, [sub])
        t_replay.substitute(sub)
        r = t_replay.replay_forward(executor=_linear_executor())

        classified_clean = {e.step_id for e in summary.entries if not e.dirty}
        replayed_dirty = {s.step_id for s in r.steps if s.dirty}

        spuriously_dirty = classified_clean & replayed_dirty
        assert not spuriously_dirty, (
            f"Soundness violation: classifier said clean but replay dirtied: "
            f"{spuriously_dirty}"
        )


# ──────────────────────── 4. Parallel-branch isolation ──────────────────────

class TestParallelBranchIsolation:
    """A RaiseSubstitution inside one branch must not dirty sibling branches."""

    def test_raise_in_branch_a_does_not_dirty_sibling_branches(self, tmp_path):
        """RaiseSubstitution at step:5 (branch A tail: tool_call inside branch A)
        must dirty step:5, the join (step:10), and post-join (step:11), but
        must NOT dirty branch B or C steps (step:6–9)."""
        t = replay(_record_parallel(str(tmp_path)))
        t.substitute(RaiseSubstitution(at_step="step:5", exception_type="ToolFailed"))
        r = t.replay_forward(executor=_parallel_executor())

        steps_by_id = {s.step_id: s for s in r.steps}

        # Branch A tail must be dirty.
        assert steps_by_id["step:5"].dirty is True

        # Join and post-join must be dirty (hash of branch A tail changed).
        assert steps_by_id["step:10"].dirty is True
        assert steps_by_id["step:11"].dirty is True

        # Sibling branch B steps (step:6, step:7) and C steps (step:8, step:9)
        # must be clean — they have no declared dependency on step:5.
        for sibling in ("step:6", "step:7", "step:8", "step:9"):
            assert steps_by_id[sibling].cache_hit is True, (
                f"{sibling} should be a cache hit (sibling branch, unaffected)"
            )

    def test_raise_in_branch_dirty_set_count_via_classifier(self, tmp_path):
        """compute_dirty_set: exactly step:5 (substituted) + step:10 (join) +
        step:11 (post-join) are dirty under a single-branch RaiseSubstitution."""
        t = replay(_record_parallel(str(tmp_path)))
        subs = [RaiseSubstitution(at_step="step:5", exception_type="ToolFailed")]
        summary = compute_dirty_set(t, subs)
        dirty_ids = {e.step_id for e in summary.entries if e.dirty}
        assert dirty_ids == {"step:5", "step:10", "step:11"}, (
            f"Unexpected dirty set under single-branch RaiseSubstitution: {dirty_ids}"
        )


# ──────────────────────── 5. Error sentinel format variants ─────────────────

class TestErrorSentinelFormat:
    """The error sentinel format is canonical JSON and its hash participates in
    downstream rebinding exactly like any other output hash."""

    def test_different_exception_types_produce_different_outputs(self, tmp_path):
        """Two different RaiseSubstitutions with distinct exception_type produce
        distinct outputs (and therefore distinct output hashes)."""
        path = _record_linear(str(tmp_path))
        t1 = replay(path)
        t2 = replay(path)
        t1.substitute(RaiseSubstitution(at_step="step:2", exception_type="ErrorA"))
        t2.substitute(RaiseSubstitution(at_step="step:2", exception_type="ErrorB"))
        r1 = t1.replay_forward(executor=_linear_executor())
        r2 = t2.replay_forward(executor=_linear_executor())
        s2_a = next(s for s in r1.steps if s.step_id == "step:2")
        s2_b = next(s for s in r2.steps if s.step_id == "step:2")
        assert s2_a.outputs != s2_b.outputs

    def test_empty_message_is_valid(self, tmp_path):
        """RaiseSubstitution with empty message is valid; message key is present."""
        t = replay(_record_linear(str(tmp_path)))
        t.substitute(RaiseSubstitution(at_step="step:2", exception_type="SomeError"))
        r = t.replay_forward(executor=_linear_executor())
        s2 = next(s for s in r.steps if s.step_id == "step:2")
        assert s2.outputs == {"__error__": {"type": "SomeError", "message": ""}}

    def test_exception_step_kind_can_be_targeted(self, tmp_path):
        """RaiseSubstitution can target an 'exception' step kind recorded by the
        recorder. The forced output uses the __error__ sentinel format regardless
        of the step's recorded output format."""
        path = os.path.join(str(tmp_path), "exc.sb")
        key = RecorderKey.fresh()
        with record(path, key=key) as rec:
            rec.llm_call(
                "gpt-4o-2024-11-20",
                [{"role": "user", "content": "hi"}],
                executor=fake_llm,
            )
            # Record an exception step (original run raised).
            rec.exception("OriginalError", "original message")
            rec.llm_call(
                "gpt-4o-2024-11-20",
                [{"role": "user", "content": "follow-up"}],
                executor=fake_llm,
            )

        t = replay(path)
        # Applying a RaiseSubstitution to the exception step forces a new sentinel.
        t.substitute(
            RaiseSubstitution(at_step="step:2", exception_type="NewError", message="new msg")
        )
        r = t.replay_forward(executor=_linear_executor())
        s2 = next(s for s in r.steps if s.step_id == "step:2")
        assert s2.dirty is True
        assert s2.outputs == {"__error__": {"type": "NewError", "message": "new msg"}}
        assert s2.cost_usd == 0.0
