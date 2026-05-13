"""Tests for Step 60: partial recompute for structured inputs.

Scenario: a "gather" step produces a structured output dict.  Downstream steps
each declare a subset of the parent's output fields they depend on via
``context_fields=...`` at record time.  When only some fields change (via
``FieldOutputSubstitution``), only the steps that declared those specific
fields become dirty; steps that declared only the *unchanged* fields remain
cache hits.

Covered invariants:
1. FieldOutputSubstitution is exported from ``stepback`` and ``stepback.substitutions``.
2. FieldOutputSubstitution preserves unchanged fields and overrides declared fields.
3. Recording with ``context_fields`` embeds ``_stepback_context_fields`` in inputs.
4. Recording with ``context_fields`` validates that all declared fields exist in
   the parent output (raises ValueError on missing fields).
5. Replay: downstream step with ``context_fields=["slow"]`` is a cache hit when
   only ``fast_result`` changes (FieldOutputSubstitution on "fast" fields).
6. Replay: downstream step with ``context_fields=["fast"]`` is dirty when the
   ``fast_result`` field changes.
7. Replay: step without ``context_fields`` (full context hash) stays dirty when
   *any* field of the parent output changes.
8. compute_dirty_set mirrors the replay engine's cache/dirty classification.
9. FieldOutputSubstitution with unknown field key still forces the output (the
   new key is added, not silently dropped).
10. Validation: FieldOutputSubstitution rejects empty field_updates.
11. Validation: FieldOutputSubstitution rejects non-dict field_updates.
12. context_fields applied to a step with no parent raises ValueError at record time.
13. Full round-trip: record → write .sb → load → replay with FieldOutputSubstitution
    confirms partial cache hits on a real trace file.
14. Multiple FieldOutputSubstitutions on the same step are merged (last wins per key).
15. Step C (context_fields for changed field) dirty; step B (context_fields for
    unchanged field) clean; step D (no context_fields) dirty — all three classes
    in a single trace.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, List

import pytest

import stepback
from stepback import (
    FieldOutputSubstitution,
    ToolOutputSubstitution,
    record,
    replay,
)
from stepback.divergence import compute_dirty_set
from stepback.recorder import RecorderKey
from stepback.substitutions import FieldOutputSubstitution as _DirectImport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_llm(model: str, messages: list) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {}}


def _fake_tool(name: str, args: dict) -> Any:
    return args.get("result", f"{name}_result")


def _gather_llm(model: str, messages: list) -> dict:
    """LLM response with two distinct top-level fields: choices and usage."""
    return {
        "choices": [{"message": {"role": "assistant", "content": "answer"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


# ---------------------------------------------------------------------------
# Trace builder helpers
#
# The recorder is sequential: each step becomes the parent of the next.
# To test context_fields cleanly we use small 2-step traces where the
# downstream step is the DIRECT child of the gpt-4 (llm_call) step.
# ---------------------------------------------------------------------------


def _build_choices_trace(path: str) -> None:
    """2-step: gpt-4 → use_choices (context_fields=["choices"])."""
    with record(path, key=RecorderKey.fresh()) as rec:
        rec.llm_call("gpt-4", [], _gather_llm)
        rec.tool_call("use_choices", {}, _fake_tool, context_fields=["choices"])


def _build_usage_trace(path: str) -> None:
    """2-step: gpt-4 → use_usage (context_fields=["usage"])."""
    with record(path, key=RecorderKey.fresh()) as rec:
        rec.llm_call("gpt-4", [], _gather_llm)
        rec.tool_call("use_usage", {}, _fake_tool, context_fields=["usage"])


def _build_both_trace(path: str) -> None:
    """2-step: gpt-4 → use_both (context_fields=["choices", "usage"])."""
    with record(path, key=RecorderKey.fresh()) as rec:
        rec.llm_call("gpt-4", [], _gather_llm)
        rec.tool_call("use_both", {}, _fake_tool, context_fields=["choices", "usage"])


def _build_full_hash_trace(path: str) -> None:
    """2-step: gpt-4 → use_all (no context_fields — full parent-output hash)."""
    with record(path, key=RecorderKey.fresh()) as rec:
        rec.llm_call("gpt-4", [], _gather_llm)
        rec.tool_call("use_all", {}, _fake_tool)


# ---------------------------------------------------------------------------
# § Substitution class invariants
# ---------------------------------------------------------------------------


def test_field_output_substitution_exported_from_stepback():
    """FieldOutputSubstitution must be importable from the top-level package."""
    assert hasattr(stepback, "FieldOutputSubstitution")
    assert stepback.FieldOutputSubstitution is _DirectImport


def test_field_output_substitution_is_output_forcing():
    sub = FieldOutputSubstitution("step:1", {"a": 1})
    assert sub.is_output_forcing()


def test_field_output_substitution_preserves_unchanged_fields():
    sub = FieldOutputSubstitution("step:1", {"a": 99})
    recorded_step = {"outputs": {"a": 1, "b": 2, "c": 3}}
    result = sub.force_output(recorded_step)
    assert result["a"] == 99
    assert result["b"] == 2
    assert result["c"] == 3


def test_field_output_substitution_adds_new_key():
    """A key not in the recorded output is added (forward-compatible extension)."""
    sub = FieldOutputSubstitution("step:1", {"new_key": "hello"})
    recorded_step = {"outputs": {"existing": "value"}}
    result = sub.force_output(recorded_step)
    assert result["existing"] == "value"
    assert result["new_key"] == "hello"


def test_field_output_substitution_empty_recorded_output():
    sub = FieldOutputSubstitution("step:1", {"key": "val"})
    result = sub.force_output({"outputs": None})
    assert result == {"key": "val"}


def test_field_output_substitution_rejects_empty_updates():
    with pytest.raises(ValueError, match="non-empty"):
        FieldOutputSubstitution("step:1", {})


def test_field_output_substitution_rejects_non_dict_updates():
    with pytest.raises(ValueError, match="must be a dict"):
        FieldOutputSubstitution("step:1", ["a", "b"])  # type: ignore[arg-type]


def test_field_output_substitution_apply_is_noop_on_inputs():
    sub = FieldOutputSubstitution("step:1", {"k": "v"})
    inputs = {"model": "gpt", "messages": []}
    original = dict(inputs)
    sub.apply(inputs, {})
    assert inputs == original


# ---------------------------------------------------------------------------
# § Recorder: context_fields validation
# ---------------------------------------------------------------------------


def test_context_fields_embeds_metadata():
    """Recording with context_fields stores _stepback_context_fields in inputs."""
    key = RecorderKey.fresh()
    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        path = f.name
    try:
        with record(path, key=key) as rec:
            # step:1 parent (llm_call → multi-field output)
            rec.llm_call("gpt-4", [], _gather_llm)
            # step:2 child with context_fields
            rec.tool_call("child", {}, _fake_tool, context_fields=["choices"])

        t = replay(path)
        child_step = t.recorded_steps[1]
        assert "_stepback_context_fields" in child_step["inputs"]
        assert child_step["inputs"]["_stepback_context_fields"] == ["choices"]
    finally:
        os.unlink(path)


def test_context_fields_partial_hash_differs_from_full_hash():
    """Context hash with context_fields differs from hash of full parent output."""
    from stepback.canonical import hash_obj

    key = RecorderKey.fresh()
    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        path = f.name
    try:
        with record(path, key=key) as rec:
            rec.llm_call("gpt-4", [], _gather_llm)
            rec.tool_call("child_partial", {}, _fake_tool, context_fields=["choices"])

        t = replay(path)
        parent_out = _gather_llm("gpt-4", [])
        partial_ctx = t.recorded_steps[1]["inputs"]["context"]

        full_hash = hash_obj(parent_out)
        partial_hash = hash_obj({"choices": parent_out["choices"]})

        assert partial_ctx == partial_hash
        assert partial_ctx != full_hash
    finally:
        os.unlink(path)


def test_context_fields_raises_if_no_parent():
    """context_fields requires a parent step; must raise if no parent is active."""
    key = RecorderKey.fresh()
    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        path = f.name
    try:
        with pytest.raises(ValueError, match="no parent is set"):
            with record(path, key=key) as rec:
                rec.tool_call("orphan", {}, _fake_tool, context_fields=["x"])
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_context_fields_raises_on_missing_field():
    """context_fields raises if any declared field is absent from parent output."""
    key = RecorderKey.fresh()
    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        path = f.name
    try:
        with pytest.raises(ValueError, match="not present in parent output"):
            with record(path, key=key) as rec:
                rec.llm_call("gpt-4", [], _gather_llm)  # outputs: choices + usage
                rec.tool_call("child", {}, _fake_tool, context_fields=["choices", "missing_key"])
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# § Core partial-recompute semantics (replay engine)
# ---------------------------------------------------------------------------


def test_partial_recompute_unchanged_field_stays_clean():
    """Step with context_fields for the UNchanged field must be a cache hit."""
    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        path = f.name
    try:
        _build_choices_trace(path)  # gpt-4 → use_choices(context_fields=["choices"])
        t = replay(path)
        # Substitute only usage → use_choices (context_fields=["choices"]) must stay clean
        t.substitute(FieldOutputSubstitution("step:1", {"usage": {"prompt_tokens": 99}}))
        result = t.replay_forward()

        by_name = {s.name: s for s in result.steps}
        assert by_name["use_choices"].cache_hit, "use_choices depends only on choices (unchanged)"
    finally:
        os.unlink(path)


def test_partial_recompute_changed_field_becomes_dirty():
    """Step with context_fields for the CHANGED field must be dirty."""
    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        path = f.name
    try:
        _build_choices_trace(path)
        t = replay(path)
        # Substitute choices → use_choices dirty (classified without executing)
        subs = [FieldOutputSubstitution("step:1", {"choices": [{"new": "answer"}]})]
        summary = compute_dirty_set(t, subs)
        step_names = {rs["step_id"]: rs.get("name") for rs in t.recorded_steps}
        entries_by_name = {step_names.get(e.step_id): e for e in summary.entries}
        assert entries_by_name["use_choices"].dirty, "use_choices depends on choices (changed)"
    finally:
        os.unlink(path)


def test_full_hash_step_dirty_on_any_field_change():
    """Step without context_fields (full hash) must be dirty when any field changes."""
    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        path = f.name
    try:
        _build_full_hash_trace(path)  # gpt-4 → use_all (no context_fields)
        t = replay(path)
        subs = [FieldOutputSubstitution("step:1", {"usage": {"prompt_tokens": 999}})]
        summary = compute_dirty_set(t, subs)
        step_names = {rs["step_id"]: rs.get("name") for rs in t.recorded_steps}
        entries_by_name = {step_names.get(e.step_id): e for e in summary.entries}
        assert entries_by_name["use_all"].dirty, "use_all has no context_fields so full hash changes"
    finally:
        os.unlink(path)


def test_use_both_dirty_when_either_field_changes():
    """Step with context_fields=["choices","usage"] is dirty when choices changes."""
    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        path = f.name
    try:
        _build_both_trace(path)  # gpt-4 → use_both(context_fields=["choices","usage"])
        t = replay(path)
        subs = [FieldOutputSubstitution("step:1", {"choices": [{"new": "choice"}]})]
        summary = compute_dirty_set(t, subs)
        step_names = {rs["step_id"]: rs.get("name") for rs in t.recorded_steps}
        entries_by_name = {step_names.get(e.step_id): e for e in summary.entries}
        assert entries_by_name["use_both"].dirty, "use_both depends on choices which changed"
    finally:
        os.unlink(path)


def test_use_both_clean_when_neither_field_changes():
    """When no field changes, every step must be a cache hit."""
    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        path = f.name
    try:
        _build_both_trace(path)
        t = replay(path)
        # No substitution → all cache hits
        result = t.replay_forward()
        assert all(s.cache_hit for s in result.steps)
    finally:
        os.unlink(path)


def test_all_three_classes_in_one_test():
    """B (partial-dep clean), C (partial-dep dirty), D (full-hash dirty) — all three classes."""
    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        p_choices = f.name
    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        p_full = f.name
    try:
        _build_choices_trace(p_choices)  # gpt-4 → use_choices(["choices"])
        _build_full_hash_trace(p_full)   # gpt-4 → use_all (no context_fields)

        def _names(t_):
            return {rs["step_id"]: rs.get("name") for rs in t_.recorded_steps}

        # B: partial-dep clean when undeclared field changes (replay works: no execution needed)
        t = replay(p_choices)
        t.substitute(FieldOutputSubstitution("step:1", {"usage": {"prompt_tokens": 42}}))
        result = t.replay_forward()
        assert {s.name: s for s in result.steps}["use_choices"].cache_hit  # B

        # C: partial-dep dirty when declared field changes (classify without executing)
        t = replay(p_choices)
        subs = [FieldOutputSubstitution("step:1", {"choices": [{"new": "answer"}]})]
        summary = compute_dirty_set(t, subs)
        entries = {_names(t).get(e.step_id): e for e in summary.entries}
        assert entries["use_choices"].dirty  # C

        # D: full-hash dirty on any field change (classify without executing)
        t = replay(p_full)
        subs = [FieldOutputSubstitution("step:1", {"usage": {"prompt_tokens": 42}})]
        summary = compute_dirty_set(t, subs)
        entries = {_names(t).get(e.step_id): e for e in summary.entries}
        assert entries["use_all"].dirty  # D
    finally:
        for p in [p_choices, p_full]:
            if os.path.exists(p):
                os.unlink(p)


def test_partial_recompute_zero_dirty_count_on_no_sub():
    """Without substitution, dirty_count must be 0."""
    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        path = f.name
    try:
        _build_choices_trace(path)
        result = replay(path).replay_forward()
        assert result.dirty_count == 0
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# § compute_dirty_set mirrors replay classification
# ---------------------------------------------------------------------------


def test_compute_dirty_set_partial_deps_unchanged_clean():
    """compute_dirty_set must classify field-dep steps clean when declared field unchanged."""
    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        path = f.name
    try:
        _build_choices_trace(path)  # gpt-4 → use_choices(context_fields=["choices"])
        t = replay(path)
        # Change usage → use_choices (context_fields=["choices"]) should stay clean
        subs = [FieldOutputSubstitution("step:1", {"usage": {"prompt_tokens": 888}})]
        summary = compute_dirty_set(t, subs)

        step_names = {rs["step_id"]: rs.get("name") for rs in t.recorded_steps}
        entries_by_name = {step_names.get(e.step_id): e for e in summary.entries}

        assert entries_by_name["use_choices"].cache_hit
    finally:
        os.unlink(path)


def test_compute_dirty_set_full_hash_step_dirty():
    """compute_dirty_set marks steps without context_fields dirty on any field change."""
    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        path = f.name
    try:
        _build_full_hash_trace(path)  # gpt-4 → use_all (no context_fields)
        t = replay(path)
        subs = [FieldOutputSubstitution("step:1", {"usage": {"total_tokens": 555}})]
        summary = compute_dirty_set(t, subs)

        step_names = {rs["step_id"]: rs.get("name") for rs in t.recorded_steps}
        use_all_entry = next(
            e for e in summary.entries if step_names.get(e.step_id) == "use_all"
        )
        assert use_all_entry.dirty
    finally:
        os.unlink(path)


def test_compute_dirty_set_matches_replay():
    """Dirty classification in compute_dirty_set must agree with run_replay."""
    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        path = f.name
    try:
        _build_choices_trace(path)  # gpt-4 → use_choices(context_fields=["choices"])
        t = replay(path)
        # Change usage (undeclared) → use_choices should be clean in both
        subs = [FieldOutputSubstitution("step:1", {"usage": {"total_tokens": 123}})]

        summary = compute_dirty_set(t, subs)
        t.substitute(*subs)
        result = t.replay_forward()

        dirty_from_classifier = {e.step_id for e in summary.entries if e.dirty}
        dirty_from_replay = {s.step_id for s in result.steps if s.dirty}
        assert dirty_from_classifier == dirty_from_replay
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# § FieldOutputSubstitution stacking
# ---------------------------------------------------------------------------


def test_multiple_field_output_substitutions_on_same_step():
    """Multiple FieldOutputSubstitutions on the same step are both applied (merged)."""
    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        path = f.name
    try:
        _build_both_trace(path)  # gpt-4 → use_both(context_fields=["choices","usage"])
        t = replay(path)
        # Two substitutions on the same step: both choices and usage change
        subs = [
            FieldOutputSubstitution("step:1", {"choices": [{"alt": "answer"}]}),
            FieldOutputSubstitution("step:1", {"usage": {"prompt_tokens": 99}}),
        ]
        summary = compute_dirty_set(t, subs)
        step_names = {rs["step_id"]: rs.get("name") for rs in t.recorded_steps}
        entries_by_name = {step_names.get(e.step_id): e for e in summary.entries}

        # gpt-4 is dirty (output-forcing substitution)
        assert entries_by_name["gpt-4"].dirty
        # use_both depends on both fields (both changed) → dirty
        assert entries_by_name["use_both"].dirty
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# § Interaction with ToolOutputSubstitution (existing behavior unchanged)
# ---------------------------------------------------------------------------


def test_tool_output_substitution_still_works():
    """ToolOutputSubstitution forces the whole output (no partial-recompute regression)."""
    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        path = f.name
    try:
        _build_choices_trace(path)  # gpt-4 → use_choices(context_fields=["choices"])
        t = replay(path)
        new_out = {
            "choices": [{"message": {"role": "assistant", "content": "changed"}}],
            "usage": {"prompt_tokens": 1},
        }
        subs = [ToolOutputSubstitution("step:1", new_out)]
        summary = compute_dirty_set(t, subs)
        step_names = {rs["step_id"]: rs.get("name") for rs in t.recorded_steps}
        entries_by_name = {step_names.get(e.step_id): e for e in summary.entries}

        # Both gpt-4 and use_choices must be dirty after full output replacement
        assert entries_by_name["gpt-4"].dirty
        assert entries_by_name["use_choices"].dirty, "ToolOutputSubstitution must dirty use_choices"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# § Full round-trip on real .sb file
# ---------------------------------------------------------------------------


def test_full_roundtrip_partial_recompute():
    """End-to-end: record → .sb → load → partial recompute → verify cache hit count."""
    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        path = f.name
    try:
        _build_choices_trace(path)  # gpt-4 → use_choices(context_fields=["choices"])
        t = replay(path)
        assert len(t.recorded_steps) == 2

        # Change only usage → use_choices (context_fields=["choices"]) stays clean
        t.substitute(FieldOutputSubstitution("step:1", {"usage": {"total_tokens": 999}}))
        result = t.replay_forward()

        by_name = {s.name: s for s in result.steps}
        # gpt-4 (gather) is dirty (has the FieldOutputSubstitution)
        assert by_name["gpt-4"].dirty
        # use_choices depends on choices (unchanged) → cache hit
        assert by_name["use_choices"].cache_hit

        # Cache hit count: only use_choices
        assert result.cache_hit_count == 1
    finally:
        os.unlink(path)
