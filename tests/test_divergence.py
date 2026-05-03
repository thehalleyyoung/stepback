"""Tests for stepback.divergence — replay non-determinism detector."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import pytest

from stepback import record
from stepback.divergence import (
    EQUIVALENT,
    IDENTICAL,
    MINOR,
    SEMANTIC,
    SEVERITY,
    SEVERITY_WEIGHT,
    STRUCTURAL,
    VOLATILE_KEYS,
    Divergence,
    DivergenceReport,
    compare_outputs,
    detect_divergences,
)
from stepback.recorder import RecorderKey
from stepback.replay import Executor

from tests.fixtures.agent import (
    fake_llm,
    fake_tool,
    run_recorded_agent,
)


# --------------------------------------------------------------- unit


def test_compare_identical():
    a = {"choices": [{"message": {"role": "assistant", "content": "hi"}}],
         "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}
    cls, _ = compare_outputs(a, dict(a))
    assert cls == IDENTICAL


def test_compare_equivalent_when_only_volatile_differs():
    a = {"id": "chatcmpl-AAA", "created": 1, "system_fingerprint": "fp_x",
         "choices": [{"message": {"role": "assistant", "content": "hello"}}],
         "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}
    b = {"id": "chatcmpl-BBB", "created": 999, "system_fingerprint": "fp_y",
         "choices": [{"message": {"role": "assistant", "content": "hello"}}],
         "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}
    cls, summary = compare_outputs(a, b)
    assert cls == EQUIVALENT
    assert "volatile" in summary


def test_compare_minor_same_assistant_text_diff_metadata():
    a = {"choices": [{"message": {"role": "assistant", "content": "answer"},
                      "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}
    b = {"choices": [{"message": {"role": "assistant", "content": "answer"},
                      "finish_reason": "length"}],
         "usage": {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14}}
    cls, _ = compare_outputs(a, b)
    assert cls == MINOR


def test_compare_semantic_different_assistant_text():
    a = {"choices": [{"message": {"role": "assistant", "content": "yes"}}]}
    b = {"choices": [{"message": {"role": "assistant", "content": "no"}}]}
    cls, summary = compare_outputs(a, b)
    assert cls == SEMANTIC
    assert "yes" in summary and "no" in summary


def test_compare_structural_keys_differ():
    a = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    b = {"completion": "hi"}
    cls, summary = compare_outputs(a, b)
    assert cls == STRUCTURAL
    assert "key" in summary or "type" in summary or "shape" in summary


def test_compare_structural_top_type_mismatch():
    cls, summary = compare_outputs({"x": 1}, [1, 2, 3])
    assert cls == STRUCTURAL
    assert "type" in summary


def test_compare_anthropic_shape_equivalent_after_volatile_strip():
    a = {"id": "msg_01", "type": "message",
         "content": [{"type": "text", "text": "Hello"}]}
    b = {"id": "msg_02", "type": "message",
         "content": [{"type": "text", "text": "Hello"}]}
    cls, _ = compare_outputs(a, b)
    assert cls == EQUIVALENT


def test_compare_anthropic_shape_semantic_when_text_differs():
    a = {"id": "msg_01", "type": "message",
         "content": [{"type": "text", "text": "Hello"}]}
    b = {"id": "msg_02", "type": "message",
         "content": [{"type": "text", "text": "Hi"}]}
    cls, _ = compare_outputs(a, b)
    assert cls == SEMANTIC


def test_severity_ordering_strict():
    assert SEVERITY == (IDENTICAL, EQUIVALENT, MINOR, SEMANTIC, STRUCTURAL)
    weights = [SEVERITY_WEIGHT[c] for c in SEVERITY]
    assert weights == sorted(weights)
    assert weights[0] == 0
    assert weights[-1] >= 4


def test_volatile_keys_include_provider_bookkeeping():
    for k in ("id", "created", "request_id", "system_fingerprint"):
        assert k in VOLATILE_KEYS


# --------------------------------------------------------- end-to-end


@pytest.fixture()
def recorded_trace(tmp_path):
    p = str(tmp_path / "t.sb")
    key = RecorderKey.fresh()
    with record(p, key=key) as ctx:
        run_recorded_agent(ctx)
    return p, key


def test_detect_divergences_self_replay_is_perfect(recorded_trace):
    """Running the SAME deterministic executor over the SAME trace yields zero divergence."""
    path, key = recorded_trace

    def _llm(model, messages):
        return fake_llm(model, messages)

    def _tool(name, args):
        return fake_tool(name, args)  # Executor wraps in {"result": ...}

    executor = Executor(llm=_llm, tool=_tool)
    report = detect_divergences(path, hmac_key=key.hmac_key, executor=executor)
    assert report.step_count == 12
    assert len(report.divergences) == 12
    assert report.divergent_count == 0
    assert report.severity_score == 0
    assert report.reproducibility_pct == 100.0
    assert all(d.classification == IDENTICAL for d in report.divergences)


def test_detect_divergences_executor_recorded_cli_path(recorded_trace):
    path, key = recorded_trace
    out = subprocess.check_output(
        [sys.executable, "-m", "stepback.cli", "divergence", path,
         "--hmac-key-hex", key.hmac_key.hex(),
         "--executor-recorded", "--format", "json"],
        text=True,
    )
    data = json.loads(out)
    assert data["step_count"] == 12
    assert data["compared_count"] == 12
    assert data["divergent_count"] == 0
    assert data["severity_score"] == 0
    assert data["reproducibility_pct"] == 100.0


def test_detect_divergences_volatile_only_drift_is_equivalent(recorded_trace):
    """Perturbing ONLY volatile fields is classified as EQUIVALENT, not divergent."""
    path, key = recorded_trace
    counter = {"n": 0}

    def _llm(model, messages):
        out = fake_llm(model, messages)
        counter["n"] += 1
        # Mutate volatile fields only.
        out["id"] = f"chatcmpl-perturbed-{counter['n']}"
        out["created"] = 1700000000 + counter["n"]
        out["system_fingerprint"] = f"fp_{counter['n']}"
        return out

    def _tool(name, args):
        return fake_tool(name, args)

    report = detect_divergences(
        path, hmac_key=key.hmac_key,
        executor=Executor(llm=_llm, tool=_tool),
    )
    counts = report.class_counts
    # All 6 LLM steps should classify as EQUIVALENT.
    assert counts[EQUIVALENT] == 6
    # Tool steps were unchanged → IDENTICAL.
    assert counts[IDENTICAL] == 6
    assert counts[SEMANTIC] == 0 and counts[STRUCTURAL] == 0
    assert report.divergent_count == 6
    assert report.reproducibility_pct == 100.0  # all in IDENTICAL ∪ EQUIVALENT


def test_detect_divergences_semantic_drift_on_one_step(recorded_trace):
    path, key = recorded_trace

    def _llm(model, messages):
        out = fake_llm(model, messages)
        # Catastrophic: corrupt the assistant text on every llm call.
        out["choices"][0]["message"]["content"] = "TOTALLY DIFFERENT"
        return out

    def _tool(name, args):
        return fake_tool(name, args)

    report = detect_divergences(
        path, hmac_key=key.hmac_key,
        executor=Executor(llm=_llm, tool=_tool),
    )
    counts = report.class_counts
    assert counts[SEMANTIC] == 6  # all 6 LLM steps drift semantically
    assert counts[IDENTICAL] == 6  # tools still IDENTICAL
    assert report.severity_score == 6 * SEVERITY_WEIGHT[SEMANTIC]
    assert report.reproducibility_pct == 50.0


def test_detect_divergences_structural_drift_on_tool(recorded_trace):
    path, key = recorded_trace

    def _tool(name, args):
        # Wrong envelope: drop the {"result": ...} wrapper Executor adds,
        # so the comparator sees a raw dict where it expected {"result":...}.
        return "totally-the-wrong-shape"

    report = detect_divergences(
        path, hmac_key=key.hmac_key,
        executor=Executor(llm=fake_llm, tool=_tool),
        step_kinds=("tool_call",),
    )
    counts = report.class_counts
    assert counts[STRUCTURAL] == 6
    assert report.severity_score == 6 * SEVERITY_WEIGHT[STRUCTURAL]


def test_detect_divergences_executor_exception_recorded_as_structural(recorded_trace):
    path, key = recorded_trace

    def _llm(model, messages):
        raise RuntimeError("provider 503")

    report = detect_divergences(
        path, hmac_key=key.hmac_key,
        executor=Executor(llm=_llm, tool=fake_tool),
    )
    err_divs = [d for d in report.divergences if d.error]
    assert len(err_divs) == 6
    assert all(d.classification == STRUCTURAL for d in err_divs)
    assert all("RuntimeError" in (d.error or "") for d in err_divs)


def test_detect_divergences_skips_kinds_without_executor(recorded_trace):
    path, key = recorded_trace
    # Only an LLM executor — every tool_call step should be skipped.
    report = detect_divergences(
        path, hmac_key=key.hmac_key,
        executor=Executor(llm=fake_llm),
    )
    assert len(report.divergences) == 6  # only LLMs compared
    assert len(report.skipped) == 6      # tools skipped


def test_detect_divergences_max_steps_limits_scope(recorded_trace):
    path, key = recorded_trace
    report = detect_divergences(
        path, hmac_key=key.hmac_key,
        executor=Executor(llm=fake_llm, tool=fake_tool),
        max_steps=3,
    )
    assert report.step_count == 12
    assert len(report.divergences) + len(report.skipped) <= 3


# --------------------------------------------------------- report shapes


def test_report_to_json_has_required_keys(recorded_trace):
    path, key = recorded_trace
    report = detect_divergences(
        path, hmac_key=key.hmac_key,
        executor=Executor(llm=fake_llm, tool=fake_tool),
    )
    j = report.to_json()
    for k in ("trace_path", "step_count", "compared_count", "divergent_count",
              "severity_score", "reproducibility_pct", "class_counts",
              "skipped_step_ids", "divergences"):
        assert k in j
    assert j["step_count"] == 12
    assert isinstance(j["divergences"], list)
    # each per-step row has the schema we promised
    for d in j["divergences"]:
        for k in ("step_id", "kind", "classification", "severity",
                  "recorded_hash", "replayed_hash", "summary"):
            assert k in d


def test_report_render_markdown_contains_metric_rows(recorded_trace):
    path, key = recorded_trace
    report = detect_divergences(
        path, hmac_key=key.hmac_key,
        executor=Executor(llm=fake_llm, tool=fake_tool),
    )
    md = report.render_markdown()
    assert "# stepback divergence report" in md
    assert "severity_score" in md
    assert "reproducibility_pct" in md
    for c in SEVERITY:
        assert c in md


# ----------------------------------------------------------- numeric


def test_severity_score_is_sum_of_per_step_weights():
    divs = [
        Divergence("step:1", "llm_call", None, IDENTICAL, "sha256:a", "sha256:a", "x"),
        Divergence("step:2", "llm_call", None, EQUIVALENT, "sha256:b", "sha256:c", "x"),
        Divergence("step:3", "tool_call", None, MINOR, "sha256:d", "sha256:e", "x"),
        Divergence("step:4", "tool_call", None, SEMANTIC, "sha256:f", "sha256:g", "x"),
        Divergence("step:5", "tool_call", None, STRUCTURAL, "sha256:h", "sha256:i", "x"),
    ]
    rep = DivergenceReport(trace_path="t.sb", step_count=5, divergences=divs)
    expected = (SEVERITY_WEIGHT[IDENTICAL] + SEVERITY_WEIGHT[EQUIVALENT]
                + SEVERITY_WEIGHT[MINOR] + SEVERITY_WEIGHT[SEMANTIC]
                + SEVERITY_WEIGHT[STRUCTURAL])
    assert rep.severity_score == expected
    assert rep.divergent_count == 4  # everything except the IDENTICAL row
    # 2 of 5 (IDENTICAL + EQUIVALENT) are "good"
    assert rep.reproducibility_pct == 40.0


def test_reproducibility_pct_perfect_on_empty():
    rep = DivergenceReport(trace_path="t.sb", step_count=0, divergences=[])
    assert rep.reproducibility_pct == 100.0
    assert rep.severity_score == 0


def test_class_counts_zero_initialised_for_all_classes():
    rep = DivergenceReport(trace_path="t.sb", step_count=0, divergences=[])
    for c in SEVERITY:
        assert rep.class_counts[c] == 0


# --------------------------------------------------------- public API


def test_module_is_re_exported_from_top_level():
    import stepback as sb
    assert sb.detect_divergences is detect_divergences
    assert sb.compare_outputs is compare_outputs
    assert sb.DivergenceReport is DivergenceReport
    assert sb.Divergence is Divergence
    assert sb.DIVERGENCE_SEVERITY == SEVERITY
