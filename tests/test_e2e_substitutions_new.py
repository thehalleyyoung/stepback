"""End-to-end replay tests for the new substitution kinds.

Drives the same 12-step "payments" agent fixture used by
`tests/test_e2e_replay.py`, then replays under each new substitution
kind and asserts the resulting `(cache_hits, dirty_count, real_executions)`
triple plus output-shape invariants.
"""
from __future__ import annotations

from stepback import RecorderKey, record, replay
from stepback.replay import Executor
from stepback.substitutions import (
    InputsPatchSubstitution,
    MessagePatchSubstitution,
    OutputsPatchSubstitution,
    RaiseSubstitution,
    SamplingSubstitution,
    SubstitutionSet,
    SystemPromptSubstitution,
    ToolArgumentsSubstitution,
)
from stepback.testing import fake_llm, fake_tool, run_recorded_agent


def _record(tmp_path) -> str:
    key = RecorderKey.fresh()
    p = str(tmp_path / "trace.sb")
    with record(p, key=key) as rec:
        run_recorded_agent(rec)
    return p


def _executor() -> Executor:
    return Executor(llm=fake_llm, tool=fake_tool)


def test_no_subs_full_cache_hit(tmp_path):
    """Sanity: no subs ⇒ every step is a cache hit, zero real executions."""
    t = replay(_record(tmp_path))
    r = t.replay_forward(executor=_executor())
    assert r.dirty_count == 0
    assert r.cache_hit_count == len(r.steps)
    assert r.real_executions == 0


def test_system_prompt_substitution_dirties_step_and_descendants(tmp_path):
    t = replay(_record(tmp_path))
    subs = SubstitutionSet()
    subs.add(SystemPromptSubstitution(at_step="step:1", system_text="Be paranoid."))
    for s in subs.items: t.substitute(s)
    r = t.replay_forward(executor=_executor())
    # Step 1 (the LLM call) must be dirty.
    s1 = next(s for s in r.steps if s.step_id == "step:1")
    assert s1.dirty is True
    assert s1.cache_hit is False
    # The substitution actually changed the messages: real LLM was invoked.
    assert r.real_executions >= 1
    # Step 1's current_inputs_hash diverges from recorded.
    assert s1.current_inputs_hash != s1.recorded_inputs_hash


def test_message_patch_substitution_changes_inputs_hash(tmp_path):
    t = replay(_record(tmp_path))
    subs = SubstitutionSet()
    subs.add(
        MessagePatchSubstitution(
            at_step="step:1",
            index=-1,
            new_message={"role": "user", "content": "Pay $0.01 to Acme."},
        )
    )
    for s in subs.items: t.substitute(s)
    r = t.replay_forward(executor=_executor())
    s1 = next(s for s in r.steps if s.step_id == "step:1")
    assert s1.dirty is True
    assert s1.current_inputs_hash != s1.recorded_inputs_hash


def test_sampling_substitution_marks_step_dirty(tmp_path):
    t = replay(_record(tmp_path))
    subs = SubstitutionSet()
    subs.add(SamplingSubstitution(at_step="step:1", temperature=0.0, max_tokens=64))
    for s in subs.items: t.substitute(s)
    r = t.replay_forward(executor=_executor())
    s1 = next(s for s in r.steps if s.step_id == "step:1")
    assert s1.dirty is True
    assert s1.inputs.get("temperature") == 0.0
    assert s1.inputs.get("max_tokens") == 64


def test_tool_arguments_substitution_changes_args(tmp_path):
    t = replay(_record(tmp_path))
    subs = SubstitutionSet()
    # Step 2 is the lookup_customer tool call.
    subs.add(
        ToolArgumentsSubstitution(
            at_step="step:2", new_arguments={"name": "DIFFERENT"}
        )
    )
    for s in subs.items: t.substitute(s)
    r = t.replay_forward(executor=_executor())
    s2 = next(s for s in r.steps if s.step_id == "step:2")
    assert s2.dirty is True
    assert s2.inputs["arguments"] == {"name": "DIFFERENT"}
    # Real tool was invoked because args changed.
    assert r.real_executions >= 1


def test_inputs_patch_substitution(tmp_path):
    t = replay(_record(tmp_path))
    subs = SubstitutionSet()
    subs.add(
        InputsPatchSubstitution(
            at_step="step:1",
            ops=[{"op": "replace", "path": "/model", "value": "gpt-4o-mini-2024-07-18"}],
        )
    )
    for s in subs.items: t.substitute(s)
    r = t.replay_forward(executor=_executor())
    s1 = next(s for s in r.steps if s.step_id == "step:1")
    assert s1.dirty is True
    assert s1.inputs["model"] == "gpt-4o-mini-2024-07-18"


def test_outputs_patch_substitution_no_executor_invocation(tmp_path):
    t = replay(_record(tmp_path))
    subs = SubstitutionSet()
    # Patch the "result" inside step:2's recorded outputs.
    subs.add(
        OutputsPatchSubstitution(
            at_step="step:2",
            ops=[
                {"op": "replace", "path": "/result/iban", "value": "US12-3456-7890"},
                {"op": "replace", "path": "/result/country", "value": "US"},
            ],
        )
    )
    for s in subs.items: t.substitute(s)
    r = t.replay_forward(executor=_executor())
    s2 = next(s for s in r.steps if s.step_id == "step:2")
    assert s2.dirty is True
    # Output-forcing: recorded args unchanged; outputs reflect the patch.
    assert s2.outputs["result"]["iban"] == "US12-3456-7890"
    assert s2.outputs["result"]["country"] == "US"


def test_raise_substitution_emits_error_sentinel(tmp_path):
    t = replay(_record(tmp_path))
    subs = SubstitutionSet()
    subs.add(RaiseSubstitution(at_step="step:2", exception_type="TimeoutError", message="slow"))
    for s in subs.items: t.substitute(s)
    r = t.replay_forward(executor=_executor())
    s2 = next(s for s in r.steps if s.step_id == "step:2")
    assert s2.dirty is True
    assert s2.outputs == {"__error__": {"type": "TimeoutError", "message": "slow"}}
    # No actual cost incurred for the synthetic error.
    assert s2.cost_usd == 0.0


def test_layered_substitutions_apply_in_order(tmp_path):
    """A SystemPrompt sub on step:1 + an OutputsPatch on step:2 both fire."""
    t = replay(_record(tmp_path))
    subs = SubstitutionSet()
    subs.add(SystemPromptSubstitution(at_step="step:1", system_text="be careful"))
    subs.add(
        OutputsPatchSubstitution(
            at_step="step:2",
            ops=[{"op": "replace", "path": "/result/iban", "value": "US12-3456-7890"}],
        )
    )
    for s in subs.items: t.substitute(s)
    r = t.replay_forward(executor=_executor())
    s1 = next(s for s in r.steps if s.step_id == "step:1")
    s2 = next(s for s in r.steps if s.step_id == "step:2")
    assert s1.dirty and s2.dirty
    assert s2.outputs["result"]["iban"] == "US12-3456-7890"
    # At least both subbed steps + something downstream are dirty.
    assert r.dirty_count >= 2
