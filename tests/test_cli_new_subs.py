"""Tests for new substitution spec verbs in `parse_substitution_spec`."""
from __future__ import annotations

import json

import pytest

from stepback.branch_io import parse_substitution_spec
from stepback.substitutions import (
    InputsPatchSubstitution,
    MessagePatchSubstitution,
    OutputsPatchSubstitution,
    RaiseSubstitution,
    SamplingSubstitution,
    SystemPromptSubstitution,
    ToolArgumentsSubstitution,
)


def test_system_inline_string():
    sub = parse_substitution_spec('system@step:3=:inline:"Be brief."')
    assert isinstance(sub, SystemPromptSubstitution)
    assert sub.at_step == "step:3"
    assert sub.system_text == "Be brief."
    assert sub.mode == "replace"


def test_system_prepend_and_append():
    sp = parse_substitution_spec('system_prepend@step:3=:inline:"GUARD"')
    sa = parse_substitution_spec('system_append@step:3=:inline:"CITE"')
    assert isinstance(sp, SystemPromptSubstitution) and sp.mode == "prepend"
    assert isinstance(sa, SystemPromptSubstitution) and sa.mode == "append"


def test_message_inline():
    spec = 'message@step:3=:idx=2,inline:{"role":"user","content":"hi"}'
    sub = parse_substitution_spec(spec)
    assert isinstance(sub, MessagePatchSubstitution)
    assert sub.index == 2
    assert sub.new_message == {"role": "user", "content": "hi"}


def test_message_path(tmp_path):
    p = tmp_path / "msg.json"
    p.write_text(json.dumps({"role": "assistant", "content": "ok"}))
    sub = parse_substitution_spec(f"message@step:5=:idx=0,path={p}")
    assert isinstance(sub, MessagePatchSubstitution)
    assert sub.index == 0
    assert sub.new_message["role"] == "assistant"


def test_message_bad_body():
    with pytest.raises(ValueError):
        parse_substitution_spec("message@step:1=bogus")


def test_sampling_kv():
    sub = parse_substitution_spec(
        "sampling@step:1=:kv:temperature=0.0,max_tokens=128,seed=42"
    )
    assert isinstance(sub, SamplingSubstitution)
    assert sub.temperature == 0.0
    assert sub.max_tokens == 128
    assert sub.seed == 42
    assert sub.top_p is None


def test_sampling_kv_null():
    sub = parse_substitution_spec("sampling@step:1=:kv:seed=null,temperature=0.5")
    assert sub.seed is None
    assert sub.temperature == 0.5


def test_sampling_inline_json():
    sub = parse_substitution_spec(
        'sampling@step:1=:inline:{"temperature":0.7,"top_p":0.9}'
    )
    assert isinstance(sub, SamplingSubstitution)
    assert sub.temperature == 0.7
    assert sub.top_p == 0.9


def test_tool_args_inline():
    sub = parse_substitution_spec(
        'tool_args@step:2=:inline:{"vendor":"Acme"}'
    )
    assert isinstance(sub, ToolArgumentsSubstitution)
    assert sub.new_arguments == {"vendor": "Acme"}


def test_inputs_patch_inline():
    sub = parse_substitution_spec(
        'inputs_patch@step:1=:inline:[{"op":"replace","path":"/model","value":"x"}]'
    )
    assert isinstance(sub, InputsPatchSubstitution)
    assert sub.ops[0]["path"] == "/model"


def test_outputs_patch_inline():
    sub = parse_substitution_spec(
        'outputs_patch@step:1=:inline:[{"op":"replace","path":"/result","value":1}]'
    )
    assert isinstance(sub, OutputsPatchSubstitution)
    assert sub.ops[0]["value"] == 1


def test_raise_with_message():
    sub = parse_substitution_spec("raise@step:3=TimeoutError:request timed out")
    assert isinstance(sub, RaiseSubstitution)
    assert sub.exception_type == "TimeoutError"
    assert sub.message == "request timed out"


def test_raise_without_message():
    sub = parse_substitution_spec("raise@step:3=ValueError")
    assert isinstance(sub, RaiseSubstitution)
    assert sub.exception_type == "ValueError"
    assert sub.message == ""


def test_unknown_kind_lists_new_verbs():
    with pytest.raises(ValueError) as e:
        parse_substitution_spec("frobnicate@step:1=foo")
    assert "sampling" in str(e.value) or "system" in str(e.value)


def test_existing_verbs_still_work():
    sub = parse_substitution_spec("model@step:1=gpt-4o-mini")
    assert sub.kind() == "ModelSubstitution"
