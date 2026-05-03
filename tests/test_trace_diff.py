"""Tests for cross-trace structural diff."""
from __future__ import annotations

import json
import subprocess
import sys
from typing import List

import pytest

from stepback import diff_traces, record, render_trace_diff
from stepback.recorder import RecorderKey
from stepback.trace_diff import (
    DIVERGENT_STATUSES,
    STATUS_A_ONLY,
    STATUS_B_ONLY,
    STATUS_IDENTICAL,
    STATUS_INPUTS_DIFFER,
    STATUS_KIND_DIFFER,
    STATUS_OUTPUTS_DIFFER,
    CrossTraceDiff,
)
from tests.fixtures.agent import fake_llm, fake_tool


# ----------------------------------------------------------- helpers


def _record_simple(path: str, *, prompt_suffix: str = "", model: str = "gpt-4o-2024-11-20",
                   extra_tool_call: bool = False, swap_kind_at_step3: bool = False) -> str:
    """Record a small deterministic 4-or-5-step trace.

    Parameters give us the levers we need to construct controlled
    same/different traces for the assertions below.
    """
    key = RecorderKey.fresh()
    convo = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": f"Look up vendor {prompt_suffix}".strip()},
    ]
    with record(path, key=key) as rec:
        rec.llm_call(model, convo, executor=fake_llm)
        rec.tool_call("lookup_customer", {"name": "Acme Bolts"}, executor=fake_tool)
        if swap_kind_at_step3:
            # Diverge the third step's *kind*: tool_call instead of llm_call.
            rec.tool_call("echo", {"text": "different kind here"}, executor=fake_tool)
        else:
            rec.llm_call(model, convo + [{"role": "assistant", "content": "ok"}],
                         executor=fake_llm)
        rec.tool_call("echo", {"text": "done"}, executor=fake_tool)
        if extra_tool_call:
            rec.tool_call("echo", {"text": "extra"}, executor=fake_tool)
    return path


# --------------------------------------------------- identical paths


def test_identical_traces_have_no_divergence(tmp_path):
    a = _record_simple(str(tmp_path / "a.sb"))
    b = _record_simple(str(tmp_path / "b.sb"))
    d = diff_traces(a, b)

    assert isinstance(d, CrossTraceDiff)
    assert d.is_identical
    assert d.divergence_step is None
    assert d.shared_prefix_len == len(d.step_pairs) == 4
    assert d.divergent_count == 0
    assert d.unaligned_count == 0
    assert d.identical_count == 4
    assert d.aligned_count == 4
    assert d.total_cost_delta_usd == 0.0
    assert d.model_changes == []
    for p in d.step_pairs:
        assert p.status == STATUS_IDENTICAL
        assert p.input_diff == {}
        assert p.output_diff == {}
        assert p.cost_delta_usd == 0.0


# --------------------------------------------------- input divergence


def test_inputs_differ_when_user_prompt_changes(tmp_path):
    a = _record_simple(str(tmp_path / "a.sb"))
    b = _record_simple(str(tmp_path / "b.sb"), prompt_suffix="Acme Bolts please")
    d = diff_traces(a, b)

    assert not d.is_identical
    # Step 1 is the LLM call whose user-message changed → inputs differ.
    assert d.divergence_step == "step:1"
    assert d.shared_prefix_len == 0
    pair0 = d.step_pairs[0]
    assert pair0.status == STATUS_INPUTS_DIFFER
    assert pair0.kind == "llm_call"
    # The downstream LLM step (step 3) also has the prompt embedded → its
    # inputs differ too.
    statuses = [p.status for p in d.step_pairs]
    assert statuses.count(STATUS_INPUTS_DIFFER) >= 2
    # All four steps will diverge: the recorder binds each step's
    # parent-output hash into its inputs, so a change at step 1 ripples
    # forward through tool_call 2, llm_call 3, and tool_call 4.
    assert all(p.status == STATUS_INPUTS_DIFFER for p in d.step_pairs)


# --------------------------------------------------- model change


def test_model_change_is_detected_and_classified(tmp_path):
    a = _record_simple(str(tmp_path / "a.sb"), model="gpt-4o-2024-11-20")
    b = _record_simple(str(tmp_path / "b.sb"), model="gpt-4o-mini-2024-07-18")
    d = diff_traces(a, b)

    # Model is part of inputs, so the LLM steps have inputs_differ status.
    assert any(p.status == STATUS_INPUTS_DIFFER for p in d.step_pairs)
    # Both LLM steps changed model → 2 model_changes.
    assert len(d.model_changes) == 2
    sids = [m[0] for m in d.model_changes]
    assert sids == ["step:1", "step:3"]
    for sid, am, bm in d.model_changes:
        assert am == "gpt-4o-2024-11-20"
        assert bm == "gpt-4o-mini-2024-07-18"
    # And the StepPair carries the per-pair model_change tuple.
    llm_pairs = [p for p in d.step_pairs if p.kind == "llm_call"]
    assert all(p.model_change == ("gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18")
               for p in llm_pairs)
    # Total cost delta must be a real number (mini is cheaper, so usually
    # negative — but we don't pin the sign because pricing tables evolve).
    assert isinstance(d.total_cost_delta_usd, float)


# --------------------------------------------------- kind divergence


def test_kind_differ_status_when_step_kind_diverges(tmp_path):
    a = _record_simple(str(tmp_path / "a.sb"))
    b = _record_simple(str(tmp_path / "b.sb"), swap_kind_at_step3=True)
    d = diff_traces(a, b)

    p2 = d.step_pairs[2]
    assert p2.status == STATUS_KIND_DIFFER
    assert p2.kind == "llm_call|tool_call"
    assert p2.input_diff == {"step_kind": {"a": "llm_call", "b": "tool_call"}}
    assert d.divergence_step == "step:3"
    assert d.shared_prefix_len == 2


# --------------------------------------------------- length divergence


def test_b_only_steps_when_b_is_longer(tmp_path):
    a = _record_simple(str(tmp_path / "a.sb"))
    b = _record_simple(str(tmp_path / "b.sb"), extra_tool_call=True)
    d = diff_traces(a, b)

    assert d.aligned_count == 4
    assert d.unaligned_count == 1
    assert d.b_only_count == 1
    assert d.a_only_count == 0
    last = d.step_pairs[-1]
    assert last.status == STATUS_B_ONLY
    assert last.a_step_id is None
    assert last.b_step_id == "step:5"
    # Identical for the first 4 → divergence point is the b-only step.
    assert d.shared_prefix_len == 4
    assert d.divergence_step == "step:5"


def test_a_only_steps_when_a_is_longer(tmp_path):
    a = _record_simple(str(tmp_path / "a.sb"), extra_tool_call=True)
    b = _record_simple(str(tmp_path / "b.sb"))
    d = diff_traces(a, b)

    assert d.aligned_count == 4
    assert d.a_only_count == 1
    assert d.b_only_count == 0
    last = d.step_pairs[-1]
    assert last.status == STATUS_A_ONLY
    assert last.a_step_id == "step:5"
    assert last.b_step_id is None
    # Cost delta for an a-only pair is -ca (we lost that cost going from A→B).
    assert last.cost_delta_usd <= 0.0


# --------------------------------------------------- summary / json


def test_summary_is_json_serialisable_and_compact(tmp_path):
    a = _record_simple(str(tmp_path / "a.sb"))
    b = _record_simple(str(tmp_path / "b.sb"), prompt_suffix="changed")
    d = diff_traces(a, b)

    s = d.summary()
    # Must round-trip through JSON without surprises.
    encoded = json.dumps(s, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["divergence_step"] == "step:1"
    assert decoded["is_identical"] is False
    assert "step_pairs" not in decoded
    assert decoded["aligned_count"] == 4
    assert decoded["a_step_count"] == 4
    assert decoded["b_step_count"] == 4

    # Full to_json_dict includes per-step pairs and is also serialisable.
    full = d.to_json_dict()
    json.dumps(full, sort_keys=True)
    assert len(full["step_pairs"]) == len(d.step_pairs)
    assert full["step_pairs"][0]["status"] == STATUS_INPUTS_DIFFER

    # max_pairs caps the output.
    capped = d.to_json_dict(max_pairs=2)
    assert len(capped["step_pairs"]) == 2


def test_to_json_dict_max_pairs_zero(tmp_path):
    a = _record_simple(str(tmp_path / "a.sb"))
    b = _record_simple(str(tmp_path / "b.sb"))
    d = diff_traces(a, b)
    assert d.to_json_dict(max_pairs=0)["step_pairs"] == []


# --------------------------------------------------- rendering


def test_render_markdown_contains_divergence_section(tmp_path):
    a = _record_simple(str(tmp_path / "a.sb"))
    b = _record_simple(str(tmp_path / "b.sb"), prompt_suffix="changed", extra_tool_call=True)
    md = render_trace_diff(diff_traces(a, b), format="markdown")
    assert md.startswith("# stepback cross-trace diff")
    assert "## Summary" in md
    assert "## Step pairs" in md
    assert "| # |" in md  # table header
    assert "## Divergent steps (detail)" in md
    # Divergence point + b-only step both surfaced.
    assert "step:1" in md
    assert "step:5" in md


def test_render_json_is_parsable(tmp_path):
    a = _record_simple(str(tmp_path / "a.sb"))
    b = _record_simple(str(tmp_path / "b.sb"), prompt_suffix="x")
    out = render_trace_diff(diff_traces(a, b), format="json")
    parsed = json.loads(out)
    assert parsed["divergence_step"] == "step:1"
    assert isinstance(parsed["step_pairs"], list)


def test_render_unknown_format_raises(tmp_path):
    a = _record_simple(str(tmp_path / "a.sb"))
    b = _record_simple(str(tmp_path / "b.sb"))
    with pytest.raises(ValueError, match="unknown format"):
        render_trace_diff(diff_traces(a, b), format="xml")


def test_markdown_truncates_when_max_rows_exceeded(tmp_path):
    a = _record_simple(str(tmp_path / "a.sb"))
    b = _record_simple(str(tmp_path / "b.sb"))
    md = render_trace_diff(diff_traces(a, b), format="markdown", max_rows=2)
    assert "(truncated;" in md


# --------------------------------------------------- hmac verification path


def test_diff_traces_with_hmac_keys_verifies_chain(tmp_path):
    """Passing hmac keys should run full chain verification before diffing."""
    key_a = RecorderKey.fresh()
    key_b = RecorderKey.fresh()
    a_path = str(tmp_path / "a.sb")
    b_path = str(tmp_path / "b.sb")
    convo = [{"role": "user", "content": "hi"}]
    with record(a_path, key=key_a) as rec:
        rec.llm_call("gpt-4o-2024-11-20", convo, executor=fake_llm)
        rec.tool_call("echo", {"text": "x"}, executor=fake_tool)
    with record(b_path, key=key_b) as rec:
        rec.llm_call("gpt-4o-2024-11-20", convo, executor=fake_llm)
        rec.tool_call("echo", {"text": "x"}, executor=fake_tool)

    d = diff_traces(a_path, b_path,
                    hmac_key_a=key_a.hmac_key, hmac_key_b=key_b.hmac_key)
    assert d.is_identical


def test_diff_traces_with_wrong_hmac_key_raises(tmp_path):
    """A bogus HMAC key must surface verification failure."""
    from stepback.trace_reader import TraceVerificationError
    key = RecorderKey.fresh()
    a_path = str(tmp_path / "a.sb")
    b_path = str(tmp_path / "b.sb")
    convo = [{"role": "user", "content": "hi"}]
    with record(a_path, key=key) as rec:
        rec.llm_call("gpt-4o-2024-11-20", convo, executor=fake_llm)
    with record(b_path, key=key) as rec:
        rec.llm_call("gpt-4o-2024-11-20", convo, executor=fake_llm)

    with pytest.raises(TraceVerificationError):
        diff_traces(a_path, b_path, hmac_key_a=b"\x00" * 32)


# --------------------------------------------------- divergent statuses set


def test_divergent_statuses_set_is_complete():
    expected = {STATUS_INPUTS_DIFFER, STATUS_OUTPUTS_DIFFER, STATUS_KIND_DIFFER}
    assert DIVERGENT_STATUSES == expected


# --------------------------------------------------- CLI smoke tests


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "stepback.cli", *args],
        capture_output=True, text=True, check=False,
    )


def test_cli_trace_diff_emits_markdown(tmp_path):
    a = _record_simple(str(tmp_path / "a.sb"))
    b = _record_simple(str(tmp_path / "b.sb"), prompt_suffix="x")
    out = tmp_path / "diff.md"
    r = _run_cli("trace-diff", a, b, "-o", str(out), "--format", "markdown")
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert "# stepback cross-trace diff" in text
    assert "step:1" in text


def test_cli_trace_diff_summary_only(tmp_path):
    a = _record_simple(str(tmp_path / "a.sb"))
    b = _record_simple(str(tmp_path / "b.sb"))
    r = _run_cli("trace-diff", a, b, "--summary-only")
    assert r.returncode == 0, r.stderr
    parsed = json.loads(r.stdout)
    assert parsed["is_identical"] is True
    assert "step_pairs" not in parsed


def test_cli_trace_diff_format_inferred_from_output_extension(tmp_path):
    a = _record_simple(str(tmp_path / "a.sb"))
    b = _record_simple(str(tmp_path / "b.sb"))
    out = tmp_path / "diff.json"
    r = _run_cli("trace-diff", a, b, "-o", str(out))
    assert r.returncode == 0, r.stderr
    parsed = json.loads(out.read_text())
    assert "step_pairs" in parsed


def test_cli_trace_diff_exit_nonzero_on_divergence(tmp_path):
    a = _record_simple(str(tmp_path / "a.sb"))
    b = _record_simple(str(tmp_path / "b.sb"), prompt_suffix="x")
    r = _run_cli(
        "trace-diff", a, b, "--summary-only", "--exit-nonzero-on-divergence"
    )
    assert r.returncode == 3, r.stderr

    # Identical traces under the same flag must exit 0.
    a2 = _record_simple(str(tmp_path / "a2.sb"))
    b2 = _record_simple(str(tmp_path / "b2.sb"))
    r2 = _run_cli(
        "trace-diff", a2, b2, "--summary-only", "--exit-nonzero-on-divergence"
    )
    assert r2.returncode == 0, r2.stderr


def test_cli_trace_diff_with_hmac_keys(tmp_path):
    key_a = RecorderKey.fresh()
    key_b = RecorderKey.fresh()
    a = str(tmp_path / "a.sb")
    b = str(tmp_path / "b.sb")
    convo = [{"role": "user", "content": "hi"}]
    with record(a, key=key_a) as rec:
        rec.llm_call("gpt-4o-2024-11-20", convo, executor=fake_llm)
    with record(b, key=key_b) as rec:
        rec.llm_call("gpt-4o-2024-11-20", convo, executor=fake_llm)

    r = _run_cli(
        "trace-diff", a, b,
        "--a-hmac-key-hex", key_a.hmac_key.hex(),
        "--b-hmac-key-hex", key_b.hmac_key.hex(),
        "--summary-only",
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["is_identical"] is True
