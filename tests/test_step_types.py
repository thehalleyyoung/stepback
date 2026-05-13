"""Tests for the typed step/header/receipt views (Step 18)."""
from __future__ import annotations

import os

import pytest

import stepback
from stepback.recorder import record
from stepback.step_types import (
    Receipt,
    RecordedStep,
    StepKind,
    TraceHeader,
)
from stepback.trace_reader import read_frames, verify_trace


# ---------------------------------------------------------------------------
# StepKind
# ---------------------------------------------------------------------------


def test_stepkind_values_match_recorder_emitted_strings() -> None:
    assert StepKind.LLM_CALL == "llm_call"
    assert StepKind.TOOL_CALL == "tool_call"
    assert StepKind.ROUTER == "router"
    assert StepKind.PARALLEL_BRANCH_OPEN == "parallel_branch_open"
    assert StepKind.PARALLEL_BRANCH_JOIN == "parallel_branch_join"
    assert StepKind.EXCEPTION == "exception"


def test_stepkind_coerce_round_trips_known_values() -> None:
    assert StepKind.coerce("llm_call") is StepKind.LLM_CALL
    # idempotent on enum members
    assert StepKind.coerce(StepKind.TOOL_CALL) is StepKind.TOOL_CALL


def test_stepkind_coerce_passes_unknown_through_unchanged() -> None:
    # Forward-compat: a future recorder may emit kinds we don't know yet.
    assert StepKind.coerce("future_kind") == "future_kind"


def test_stepkind_known_values_is_a_frozenset_of_strings() -> None:
    kv = StepKind.known_values()
    assert isinstance(kv, frozenset)
    assert "llm_call" in kv and "tool_call" in kv
    # Snapshot: every member must appear
    assert kv == frozenset(m.value for m in StepKind)


# ---------------------------------------------------------------------------
# RecordedStep
# ---------------------------------------------------------------------------


def _sample_step() -> dict:
    return {
        "step_id": "step:1",
        "step_kind": "llm_call",
        "name": "gpt-4o",
        "parent_step_id": None,
        "inputs": {"kind": "llm_call", "model": "gpt-4o"},
        "outputs": {"choices": [{"message": {"content": "ok"}}]},
        "inputs_hash": "a" * 64,
        "outputs_hash": "b" * 64,
        "nondeterminism_hash": "c" * 64,
        "wallclock_ns": 12345,
        "cost_usd": 0.0125,
        "llm_request": {"model": "gpt-4o"},
        "llm_response": {"choices": []},
    }


def test_recorded_step_typed_accessors_read_through_to_dict() -> None:
    d = _sample_step()
    s = RecordedStep.from_dict(d)
    assert s.step_id == "step:1"
    assert s.step_kind is StepKind.LLM_CALL
    assert s.name == "gpt-4o"
    assert s.parent_step_id is None
    assert s.inputs == {"kind": "llm_call", "model": "gpt-4o"}
    assert s.outputs["choices"][0]["message"]["content"] == "ok"
    assert s.inputs_hash == "a" * 64
    assert s.outputs_hash == "b" * 64
    assert s.nondeterminism_hash == "c" * 64
    assert s.wallclock_ns == 12345
    assert s.cost_usd == pytest.approx(0.0125)
    assert s.llm_request == {"model": "gpt-4o"}
    assert s.llm_response == {"choices": []}


def test_recorded_step_setters_write_through_to_dict() -> None:
    d = _sample_step()
    s = RecordedStep.from_dict(d)
    s.cost_usd = 0.99
    s.name = "claude-4.5"
    s.step_kind = StepKind.TOOL_CALL
    assert d["cost_usd"] == 0.99
    assert d["name"] == "claude-4.5"
    # enum is normalised to its raw string value on disk
    assert d["step_kind"] == "tool_call"


def test_recorded_step_dict_view_is_back_compat() -> None:
    """Existing dict-style access must keep working."""
    d = _sample_step()
    s = RecordedStep.from_dict(d)

    # __getitem__, __setitem__, get, in, len, iter, .keys/.values/.items
    assert s["step_id"] == "step:1"
    s["custom_field"] = 7
    assert d["custom_field"] == 7
    assert s.get("missing", "fallback") == "fallback"
    assert "step_id" in s
    assert len(s) == len(d)
    assert set(iter(s)) == set(d.keys())
    assert dict(s) == d
    assert "step_id" in s.keys()
    assert ("step_id", "step:1") in s.items()

    # setdefault flows through
    assert s.setdefault("inputs", {"already": "set"}) == d["inputs"]
    assert s.setdefault("brand_new", []) == []
    assert d["brand_new"] == []

    # update() & __delitem__
    s.update({"another": 1})
    assert d["another"] == 1
    del s["another"]
    assert "another" not in d


def test_recorded_step_to_dict_preserves_identity() -> None:
    d = _sample_step()
    s = RecordedStep.from_dict(d)
    assert s.to_dict() is d


def test_recorded_step_constructor_rejects_non_dict() -> None:
    with pytest.raises(TypeError):
        RecordedStep(["not", "a", "dict"])  # type: ignore[arg-type]


def test_recorded_step_unknown_step_kind_passes_through() -> None:
    d = dict(_sample_step(), step_kind="future_only")
    s = RecordedStep.from_dict(d)
    # Not coerced into the enum; raw string preserved
    assert s.step_kind == "future_only"
    assert not isinstance(s.step_kind, StepKind)


def test_recorded_step_equality_and_repr() -> None:
    a = RecordedStep.from_dict(_sample_step())
    b = RecordedStep.from_dict(_sample_step())
    assert a == b
    assert a == _sample_step()  # equal to a plain dict with the same contents
    r = repr(a)
    assert r.startswith("RecordedStep(")
    assert "step_id='step:1'" in r


def test_recorded_step_is_unhashable_like_dict() -> None:
    s = RecordedStep.from_dict(_sample_step())
    with pytest.raises(TypeError):
        hash(s)


def test_recorded_step_default_construction_is_empty() -> None:
    s = RecordedStep()
    assert len(s) == 0
    assert s.cost_usd == 0.0  # default for missing cost
    assert s.parent_step_id is None


def test_recorded_step_wraps_real_recorder_output(tmp_path) -> None:
    p = tmp_path / "t.sb"
    with record(str(p)) as rec:
        rec.llm_call(
            "gpt-4o",
            [{"role": "user", "content": "hi"}],
            executor=lambda m, ms: {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )
        rec.tool_call("noop", {"x": 1}, executor=lambda n, a: {"ok": True})
    # The recorder's in-memory step list is dicts; wrap them in views.
    assert len(rec.steps) == 2
    s0 = RecordedStep.from_dict(rec.steps[0])
    s1 = RecordedStep.from_dict(rec.steps[1])
    assert s0.step_kind is StepKind.LLM_CALL
    assert s1.step_kind is StepKind.TOOL_CALL
    assert s0.parent_step_id is None
    assert s1.parent_step_id == s0.step_id
    assert s0.inputs_hash and s0.outputs_hash and s0.nondeterminism_hash


# ---------------------------------------------------------------------------
# TraceHeader
# ---------------------------------------------------------------------------


def test_trace_header_typed_accessors(tmp_path) -> None:
    p = tmp_path / "t.sb"
    with record(str(p)) as rec:
        rec.llm_call(
            "gpt-4o", [], executor=lambda m, ms: {"choices": [], "usage": {}}
        )
    frames = read_frames(str(p))
    raw_header = frames[0]["body"]
    h = TraceHeader.from_dict(raw_header)
    assert h.type == "header"
    assert h.magic == "stepback/.sb"
    assert h.format_version == 1
    assert h.recorder_version
    assert h.canonicalisation_version
    assert len(h.public_key) == 64  # 32 bytes hex-encoded
    assert h.hmac_key_id and len(h.hmac_key_id) == 16
    assert h.price_list_version
    assert h.wallclock_ns > 0
    assert h.compression in ("none", "gzip+dedup-2")
    assert h.blob_threshold >= 0
    assert h.blob_min_reuse >= 0


def test_trace_header_is_back_compat_dict_view() -> None:
    raw = {
        "type": "header",
        "magic": "stepback/.sb",
        "format_version": 1,
        "recorder_version": "0.1.0",
        "canonicalisation_version": "v1",
        "public_key": "00" * 32,
        "hmac_key_id": "deadbeefcafef00d",
        "price_list_version": "2026-04-01",
        "wallclock_ns": 1,
        "compression": "gzip+dedup-2",
        "blob_threshold": 200,
        "blob_min_reuse": 2,
    }
    h = TraceHeader.from_dict(raw)
    assert h["public_key"] == raw["public_key"]
    assert h.public_key == raw["public_key"]
    assert h.to_dict() is raw
    assert dict(h) == raw


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def test_receipt_typed_accessors(tmp_path) -> None:
    p = tmp_path / "t.sb"
    with record(str(p)) as rec:
        rec.llm_call(
            "gpt-4o", [], executor=lambda m, ms: {"choices": [], "usage": {}}
        )
    frames = read_frames(str(p))
    r0 = Receipt.from_dict(frames[0])
    assert r0.frame_kind == "header"
    assert r0.prev_hmac == "0" * 64
    assert r0.prev_hmac_bytes == b"\x00" * 32
    assert len(r0.hmac) == 64
    assert r0.hmac_bytes == bytes.fromhex(r0.hmac)
    assert r0.signature_scheme == "ed25519"
    assert r0.signature_hex
    assert r0.signature_bytes == bytes.fromhex(r0.signature_hex)
    # Body is the underlying dict; back-compat
    assert r0["body"] is r0.body

    # The next frame's prev_hmac chains to the header's hmac.
    r1 = Receipt.from_dict(frames[1])
    assert r1.prev_hmac == r0.hmac


def test_receipt_back_compat_dict_view() -> None:
    raw = {
        "body": {"type": "tail", "wallclock_ns": 1},
        "prev_hmac": "ab" * 32,
        "hmac": "cd" * 32,
        "sig": "ed25519:" + ("ef" * 64),
    }
    r = Receipt.from_dict(raw)
    assert r["sig"] == raw["sig"]
    assert r.signature_scheme == "ed25519"
    assert r.signature_hex == "ef" * 64
    assert r.frame_kind == "tail"
    assert r.to_dict() is raw
    assert dict(r) == raw


def test_receipt_setters_write_through() -> None:
    raw = {
        "body": {"type": "step"},
        "prev_hmac": "00" * 32,
        "hmac": "11" * 32,
        "sig": "ed25519:" + ("22" * 64),
    }
    r = Receipt.from_dict(raw)
    r.prev_hmac = "33" * 32
    r.hmac = "44" * 32
    r.sig = "ed25519:" + ("55" * 64)
    r.body = {"type": "header"}
    assert raw["prev_hmac"] == "33" * 32
    assert raw["hmac"] == "44" * 32
    assert raw["sig"] == "ed25519:" + ("55" * 64)
    assert raw["body"] == {"type": "header"}


# ---------------------------------------------------------------------------
# Public re-export
# ---------------------------------------------------------------------------


def test_step_types_are_publicly_exported() -> None:
    assert stepback.RecordedStep is RecordedStep
    assert stepback.TraceHeader is TraceHeader
    assert stepback.Receipt is Receipt
    assert stepback.StepKind is StepKind
    for name in ("RecordedStep", "TraceHeader", "Receipt", "StepKind"):
        assert name in stepback.__all__
