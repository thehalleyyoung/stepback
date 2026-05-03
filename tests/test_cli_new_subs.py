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


# --------------- numeric-threshold metrics ----------------


def test_sampling_kv_field_count_threshold():
    """Only fields named in the spec should be set; others remain None."""
    sub = parse_substitution_spec(
        "sampling@step:1=:kv:temperature=0.0,max_tokens=128,seed=42"
    )
    set_fields = sum(
        1
        for v in (sub.temperature, sub.max_tokens, sub.seed, sub.top_p)
        if v is not None
    )
    assert set_fields == 3
    assert sub.top_p is None


def test_inputs_patch_op_count_matches_spec():
    sub = parse_substitution_spec(
        'inputs_patch@step:1=:inline:[{"op":"replace","path":"/a","value":1},'
        '{"op":"add","path":"/b","value":2},'
        '{"op":"remove","path":"/c"}]'
    )
    assert len(sub.ops) == 3
    assert sum(1 for op in sub.ops if op["op"] == "replace") == 1
    assert sum(1 for op in sub.ops if op["op"] == "add") == 1
    assert sum(1 for op in sub.ops if op["op"] == "remove") == 1


def test_message_path_roundtrip_byte_size(tmp_path):
    payload = {"role": "assistant", "content": "ok" * 32}
    p = tmp_path / "msg.json"
    raw = json.dumps(payload)
    p.write_text(raw)
    sub = parse_substitution_spec(f"message@step:5=:idx=0,path={p}")
    # Content length is preserved exactly
    assert len(sub.new_message["content"]) == len(payload["content"])
    assert len(sub.new_message["content"]) == 64


# --------------- additional strict checks ----------------


def test_system_inline_text_byte_exact():
    sub = parse_substitution_spec('system@step:3=:inline:"Be brief."')
    assert isinstance(sub, SystemPromptSubstitution)
    assert len(sub.system_text) == 9
    assert sub.system_text == "Be brief."
    assert sub.mode == "replace"
    assert sub.at_step == "step:3"


def test_sampling_inline_field_count_threshold():
    sub = parse_substitution_spec(
        'sampling@step:1=:inline:{"temperature":0.7,"top_p":0.9}'
    )
    set_fields = sum(
        1
        for v in (sub.temperature, sub.max_tokens, sub.seed, sub.top_p)
        if v is not None
    )
    assert set_fields == 2
    assert sub.max_tokens is None
    assert sub.seed is None
    # Numeric values exact
    assert sub.temperature - sub.top_p == pytest.approx(-0.2)


def test_outputs_patch_op_count_and_kinds():
    sub = parse_substitution_spec(
        'outputs_patch@step:1=:inline:'
        '[{"op":"replace","path":"/r","value":1},'
        '{"op":"add","path":"/x","value":2}]'
    )
    assert isinstance(sub, OutputsPatchSubstitution)
    assert len(sub.ops) == 2
    kinds = {op["op"] for op in sub.ops}
    assert kinds == {"replace", "add"}
    assert len(kinds) == 2


def test_raise_message_byte_exact():
    sub = parse_substitution_spec("raise@step:3=TimeoutError:request timed out")
    assert isinstance(sub, RaiseSubstitution)
    assert len(sub.exception_type) == 12
    assert len(sub.message) == 17
    assert sub.message == "request timed out"


def test_message_inline_strict_keys():
    spec = 'message@step:3=:idx=2,inline:{"role":"user","content":"hi"}'
    sub = parse_substitution_spec(spec)
    assert isinstance(sub, MessagePatchSubstitution)
    # Strict shape: exactly the keys we provided
    assert set(sub.new_message.keys()) == {"role", "content"}
    assert len(sub.new_message) == 2
    assert sub.index == 2


def test_existing_model_verb_kind_string_exact():
    sub = parse_substitution_spec("model@step:1=gpt-4o-mini")
    name = sub.kind()
    assert name == "ModelSubstitution"
    assert len(name) == 17


def test_bulk_parse_all_new_verbs_succeeds():
    """All seven new verbs must parse to the correct dataclass — exact count."""
    specs = [
        ('system@step:1=:inline:"x"', SystemPromptSubstitution),
        ('system_prepend@step:1=:inline:"x"', SystemPromptSubstitution),
        ('system_append@step:1=:inline:"x"', SystemPromptSubstitution),
        ('message@step:1=:idx=0,inline:{"role":"user","content":"x"}',
         MessagePatchSubstitution),
        ('sampling@step:1=:kv:temperature=0.1', SamplingSubstitution),
        ('tool_args@step:1=:inline:{"k":"v"}', ToolArgumentsSubstitution),
        ('inputs_patch@step:1=:inline:[{"op":"add","path":"/x","value":1}]',
         InputsPatchSubstitution),
        ('outputs_patch@step:1=:inline:[{"op":"add","path":"/x","value":1}]',
         OutputsPatchSubstitution),
        ('raise@step:1=ValueError:boom', RaiseSubstitution),
    ]
    parsed = [(parse_substitution_spec(s), cls) for s, cls in specs]
    correct = sum(1 for sub, cls in parsed if isinstance(sub, cls))
    assert correct == 9
    assert correct == len(specs)
    # All parsed objects are non-None
    assert sum(1 for sub, _ in parsed if sub is None) == 0


def test_inputs_patch_large_op_list_exact_count():
    n = 25
    ops_json = ",".join(
        f'{{"op":"add","path":"/k{i}","value":{i}}}' for i in range(n)
    )
    spec = f'inputs_patch@step:1=:inline:[{ops_json}]'
    sub = parse_substitution_spec(spec)
    assert isinstance(sub, InputsPatchSubstitution)
    assert len(sub.ops) == n
    assert len(sub.ops) == 25
    assert sum(1 for op in sub.ops if op["op"] == "add") == 25
    # Path indices preserved in order
    assert [op["path"] for op in sub.ops[:3]] == ["/k0", "/k1", "/k2"]


def test_bulk_parse_kind_string_lengths_exact():
    """Each parsed sub's kind() string must equal its class.__name__ exactly."""
    pairs = [
        ('system@step:1=:inline:"x"', "SystemPromptSubstitution"),
        ('message@step:1=:idx=0,inline:{"role":"user","content":"x"}',
         "MessagePatchSubstitution"),
        ('sampling@step:1=:kv:temperature=0.1', "SamplingSubstitution"),
        ('tool_args@step:1=:inline:{"k":"v"}', "ToolArgumentsSubstitution"),
        ('inputs_patch@step:1=:inline:[{"op":"add","path":"/x","value":1}]',
         "InputsPatchSubstitution"),
        ('outputs_patch@step:1=:inline:[{"op":"add","path":"/x","value":1}]',
         "OutputsPatchSubstitution"),
        ('raise@step:1=ValueError:boom', "RaiseSubstitution"),
    ]
    matches = sum(
        1 for spec, name in pairs if parse_substitution_spec(spec).kind() == name
    )
    assert matches == 7
    assert matches == len(pairs)


def test_sampling_kv_all_four_fields_set():
    sub = parse_substitution_spec(
        "sampling@step:1=:kv:temperature=0.25,max_tokens=512,seed=7,top_p=0.95"
    )
    assert isinstance(sub, SamplingSubstitution)
    set_fields = sum(
        1
        for v in (sub.temperature, sub.max_tokens, sub.seed, sub.top_p)
        if v is not None
    )
    assert set_fields == 4
    assert sub.temperature == 0.25
    assert sub.max_tokens == 512
    assert sub.seed == 7
    assert sub.top_p == 0.95
    # Numeric ratio bound
    assert sub.top_p / sub.temperature == pytest.approx(3.8)


def test_raise_message_contains_no_extra_whitespace():
    sub = parse_substitution_spec("raise@step:3=TimeoutError:request timed out")
    assert isinstance(sub, RaiseSubstitution)
    # Message must not be padded with leading/trailing whitespace
    assert sub.message == sub.message.strip()
    assert sub.message.count(" ") == 2
    assert sub.exception_type.isalpha()


def test_inputs_patch_path_indices_strictly_increasing():
    n = 16
    ops_json = ",".join(
        f'{{"op":"add","path":"/k{i}","value":{i}}}' for i in range(n)
    )
    spec = f'inputs_patch@step:1=:inline:[{ops_json}]'
    sub = parse_substitution_spec(spec)
    assert isinstance(sub, InputsPatchSubstitution)
    assert len(sub.ops) == 16
    indices = [int(op["path"].lstrip("/k")) for op in sub.ops]
    assert indices == list(range(16))
    # Strictly increasing -- exact diff invariant
    diffs = [b - a for a, b in zip(indices, indices[1:])]
    assert diffs == [1] * 15
    assert sum(diffs) == 15


def test_all_substitution_classes_have_distinct_kinds():
    """Each parsed sub returns a distinct kind() — set size == count."""
    specs = [
        'system@step:1=:inline:"x"',
        'message@step:1=:idx=0,inline:{"role":"user","content":"x"}',
        'sampling@step:1=:kv:temperature=0.1',
        'tool_args@step:1=:inline:{"k":"v"}',
        'inputs_patch@step:1=:inline:[{"op":"add","path":"/x","value":1}]',
        'outputs_patch@step:1=:inline:[{"op":"add","path":"/x","value":1}]',
        'raise@step:1=ValueError:boom',
    ]
    kinds = [parse_substitution_spec(s).kind() for s in specs]
    assert len(set(kinds)) == 7
    assert len(kinds) == 7
    # Every kind name ends with "Substitution"
    assert sum(1 for k in kinds if k.endswith("Substitution")) == 7


def test_sampling_kv_numeric_field_ranges_strict():
    sub = parse_substitution_spec(
        "sampling@step:1=:kv:temperature=0.0,max_tokens=2048,seed=123,top_p=1.0"
    )
    assert isinstance(sub, SamplingSubstitution)
    assert 0.0 <= sub.temperature <= 2.0
    assert 1 <= sub.max_tokens <= 32_768
    assert sub.max_tokens == 2048
    assert sub.seed == 123
    assert 0.0 < sub.top_p <= 1.0
    # Exact zero must round-trip as float, not int
    assert isinstance(sub.temperature, float)
    assert sub.temperature == 0.0


def test_inputs_patch_huge_op_list_count_and_byte_size(tmp_path):
    """200-op patch parses with exact count and bounded JSON byte size."""
    n = 200
    ops_json = ",".join(
        f'{{"op":"replace","path":"/k{i}","value":{i}}}' for i in range(n)
    )
    spec = f'inputs_patch@step:1=:inline:[{ops_json}]'
    sub = parse_substitution_spec(spec)
    assert isinstance(sub, InputsPatchSubstitution)
    assert len(sub.ops) == n
    # Every op preserved its index in path
    paths = [op["path"] for op in sub.ops]
    assert paths == [f"/k{i}" for i in range(n)]
    # Re-serialized payload size lies within tight bounds
    serialized = json.dumps(sub.ops, separators=(",", ":"))
    assert 6_000 <= len(serialized) <= 9_000


def test_message_path_unicode_byte_count(tmp_path):
    """Unicode content survives path-load with exact codepoint count."""
    text = "héllo wörld " * 8  # 12 codepoints * 8 = 96
    p = tmp_path / "msg.json"
    p.write_text(json.dumps({"role": "user", "content": text}), encoding="utf-8")
    sub = parse_substitution_spec(f"message@step:1=:idx=0,path={p}")
    assert sub.new_message["content"] == text
    assert len(sub.new_message["content"]) == 96
    assert sub.new_message["content"].count("ö") == 8
