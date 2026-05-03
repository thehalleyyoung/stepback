"""Unit tests for the new substitution kinds added in the MoA round."""
from __future__ import annotations

import pytest

from stepback.branch_io import substitution_from_dict, substitution_to_dict
from stepback.substitutions import (
    InputsPatchSubstitution,
    MessagePatchSubstitution,
    OutputsPatchSubstitution,
    RaiseSubstitution,
    SamplingSubstitution,
    SystemPromptSubstitution,
    ToolArgumentsSubstitution,
    ToolOutputSubstitution,
)


# --------------- SystemPromptSubstitution ----------------


def test_system_replace_inserts_when_missing():
    sub = SystemPromptSubstitution(at_step="step:1", system_text="be nice")
    inputs = {"messages": [{"role": "user", "content": "hi"}]}
    sub.apply(inputs, {})
    assert inputs["messages"][0] == {"role": "system", "content": "be nice"}
    assert inputs["messages"][1] == {"role": "user", "content": "hi"}


def test_system_replace_overwrites_existing():
    sub = SystemPromptSubstitution(at_step="step:1", system_text="be terse")
    inputs = {"messages": [{"role": "system", "content": "be verbose"}]}
    sub.apply(inputs, {})
    assert inputs["messages"][0]["content"] == "be terse"


def test_system_prepend_concatenates():
    sub = SystemPromptSubstitution(at_step="step:1", system_text="GUARD:", mode="prepend")
    inputs = {"messages": [{"role": "system", "content": "you are"}]}
    sub.apply(inputs, {})
    assert inputs["messages"][0]["content"] == "GUARD:\n\nyou are"


def test_system_append_concatenates():
    sub = SystemPromptSubstitution(at_step="step:1", system_text="cite", mode="append")
    inputs = {"messages": [{"role": "system", "content": "you are"}]}
    sub.apply(inputs, {})
    assert inputs["messages"][0]["content"] == "you are\n\ncite"


def test_system_no_messages_key():
    sub = SystemPromptSubstitution(at_step="step:1", system_text="hi")
    inputs: dict = {}
    sub.apply(inputs, {})
    assert inputs["messages"] == [{"role": "system", "content": "hi"}]


def test_system_invalid_mode_rejected():
    with pytest.raises(ValueError):
        SystemPromptSubstitution(at_step="step:1", system_text="x", mode="bogus")


# --------------- MessagePatchSubstitution ----------------


def test_message_patch_replaces_at_index():
    sub = MessagePatchSubstitution(
        at_step="step:1", index=1, new_message={"role": "user", "content": "new"}
    )
    inputs = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "old"},
        ]
    }
    sub.apply(inputs, {})
    assert inputs["messages"][1] == {"role": "user", "content": "new"}


def test_message_patch_negative_index():
    sub = MessagePatchSubstitution(
        at_step="step:1", index=-1, new_message={"role": "user", "content": "last"}
    )
    inputs = {"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]}
    sub.apply(inputs, {})
    assert inputs["messages"][1]["content"] == "last"


def test_message_patch_at_len_appends():
    sub = MessagePatchSubstitution(
        at_step="step:1", index=2, new_message={"role": "assistant", "content": "x"}
    )
    inputs = {"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]}
    sub.apply(inputs, {})
    assert len(inputs["messages"]) == 3
    assert inputs["messages"][2]["content"] == "x"


def test_message_patch_out_of_range_raises():
    sub = MessagePatchSubstitution(
        at_step="step:1", index=99, new_message={"role": "user", "content": "x"}
    )
    with pytest.raises(IndexError):
        sub.apply({"messages": [{"role": "user", "content": "a"}]}, {})


def test_message_patch_requires_role_and_content():
    with pytest.raises(ValueError):
        MessagePatchSubstitution(at_step="step:1", index=0, new_message={"role": "user"})


# --------------- SamplingSubstitution ----------------


def test_sampling_validation_temperature():
    SamplingSubstitution(at_step="step:1", temperature=0.0)
    SamplingSubstitution(at_step="step:1", temperature=2.0)
    with pytest.raises(ValueError):
        SamplingSubstitution(at_step="step:1", temperature=-0.1)
    with pytest.raises(ValueError):
        SamplingSubstitution(at_step="step:1", temperature=2.1)


def test_sampling_validation_top_p():
    with pytest.raises(ValueError):
        SamplingSubstitution(at_step="step:1", top_p=1.5)


def test_sampling_validation_max_tokens():
    with pytest.raises(ValueError):
        SamplingSubstitution(at_step="step:1", max_tokens=0)
    with pytest.raises(ValueError):
        SamplingSubstitution(at_step="step:1", max_tokens=-1)


def test_sampling_only_writes_set_fields():
    sub = SamplingSubstitution(at_step="step:1", temperature=0.0, max_tokens=128)
    inputs = {"top_p": 0.9}
    sub.apply(inputs, {})
    assert inputs["temperature"] == 0.0
    assert inputs["max_tokens"] == 128
    assert inputs["top_p"] == 0.9  # untouched


def test_sampling_seed_int_required():
    SamplingSubstitution(at_step="step:1", seed=42)
    SamplingSubstitution(at_step="step:1", seed=-7)
    with pytest.raises(ValueError):
        SamplingSubstitution(at_step="step:1", seed="42")  # type: ignore[arg-type]


# --------------- ToolArgumentsSubstitution ----------------


def test_tool_arguments_replaces():
    sub = ToolArgumentsSubstitution(at_step="step:1", new_arguments={"vendor": "Acme"})
    inputs = {"name": "lookup", "arguments": {"vendor": "Bolts"}}
    sub.apply(inputs, {})
    assert inputs["arguments"] == {"vendor": "Acme"}


# --------------- InputsPatchSubstitution ----------------


def test_inputs_patch_replaces_field():
    sub = InputsPatchSubstitution(
        at_step="step:1",
        ops=[{"op": "replace", "path": "/model", "value": "gpt-4o-mini"}],
    )
    inputs = {"model": "gpt-4o", "messages": []}
    sub.apply(inputs, {})
    assert inputs["model"] == "gpt-4o-mini"


def test_inputs_patch_test_failure_propagates():
    from stepback.jsonpatch import PatchTestFailed

    sub = InputsPatchSubstitution(
        at_step="step:1",
        ops=[{"op": "test", "path": "/model", "value": "x"}],
    )
    with pytest.raises(PatchTestFailed):
        sub.apply({"model": "y"}, {})


# --------------- OutputsPatchSubstitution ----------------


def test_outputs_patch_force_output():
    sub = OutputsPatchSubstitution(
        at_step="step:1",
        ops=[{"op": "replace", "path": "/result", "value": 42}],
    )
    assert sub.is_output_forcing()
    rec = {"outputs": {"result": 0, "extra": "x"}}
    out = sub.force_output(rec)
    assert out == {"result": 42, "extra": "x"}
    # Recorded outputs not mutated
    assert rec["outputs"]["result"] == 0


# --------------- RaiseSubstitution ----------------


def test_raise_force_output():
    sub = RaiseSubstitution(at_step="step:1", exception_type="TimeoutError", message="slow")
    assert sub.is_output_forcing()
    out = sub.force_output({"outputs": {}})
    assert out == {"__error__": {"type": "TimeoutError", "message": "slow"}}


def test_raise_validation():
    with pytest.raises(ValueError):
        RaiseSubstitution(at_step="step:1", exception_type="")


# --------------- output-forcing predicate ----------------


def test_output_forcing_predicate_default_false():
    sub = SamplingSubstitution(at_step="step:1", temperature=0.0)
    assert sub.is_output_forcing() is False


def test_output_forcing_predicate_true_for_force_subs():
    assert ToolOutputSubstitution(at_step="step:1", fake_response={}).is_output_forcing()
    assert OutputsPatchSubstitution(at_step="step:1", ops=[]).is_output_forcing()
    assert RaiseSubstitution(at_step="step:1", exception_type="X").is_output_forcing()


# --------------- round-trip serialisation ----------------


@pytest.mark.parametrize(
    "sub",
    [
        SystemPromptSubstitution(at_step="step:1", system_text="hi"),
        SystemPromptSubstitution(at_step="step:1", system_text="hi", mode="prepend"),
        MessagePatchSubstitution(
            at_step="step:1", index=0, new_message={"role": "user", "content": "x"}
        ),
        SamplingSubstitution(at_step="step:1", temperature=0.0, max_tokens=128, seed=42),
        ToolArgumentsSubstitution(at_step="step:1", new_arguments={"vendor": "Acme"}),
        InputsPatchSubstitution(
            at_step="step:1",
            ops=[{"op": "replace", "path": "/model", "value": "x"}],
        ),
        OutputsPatchSubstitution(
            at_step="step:1",
            ops=[{"op": "replace", "path": "/result", "value": 1}],
        ),
        RaiseSubstitution(at_step="step:1", exception_type="TimeoutError", message="m"),
    ],
)
def test_round_trip_through_branch_io(sub):
    d = substitution_to_dict(sub)
    assert d["type"] == type(sub).__name__
    sub2 = substitution_from_dict(d)
    assert type(sub2) is type(sub)
    assert sub2 == sub
