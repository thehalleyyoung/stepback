"""Tests for Step 63: nondeterminism class taxonomy and dirty-forcing semantics.

Covers:
* :class:`~stepback.nondeterminism.NondeterminismClass` enum values and string equality.
* Helper constructor shapes (``clock_nondeterminism``, ``rng_nondeterminism``,
  ``env_nondeterminism``, ``network_nondeterminism``,
  ``model_sampling_nondeterminism``).
* ``combine_nondeterminism`` multi-source format.
* ``forces_dirty`` semantics for every class and edge case.
* Integration with :py:func:`~stepback.divergence.compute_dirty_set` — a step
  whose class forces dirty must appear in the dirty set even when inputs and
  nondeterminism hashes are unchanged.
* Integration with the replay engine — same step must not be served as a
  cache hit.
* Auto-recording of model_sampling nondeterminism in ``Recorder.llm_call``.
"""
from __future__ import annotations

from pathlib import Path

import stepback
from stepback import (
    record,
    replay,
    NondeterminismClass,
    nondeterminism_forces_dirty,
    clock_nondeterminism,
    rng_nondeterminism,
    env_nondeterminism,
    network_nondeterminism,
    model_sampling_nondeterminism,
    combine_nondeterminism,
)
from stepback.canonical import canonical_json, sha256_hex
from stepback.divergence import compute_dirty_set
from stepback.nondeterminism import forces_dirty, _source_forces_dirty
from stepback.replay import Executor
from stepback.testing import fake_llm


# ---------------------------------------------------------------------------
# NondeterminismClass enum
# ---------------------------------------------------------------------------


def test_enum_values_and_string_equality() -> None:
    assert NondeterminismClass.CLOCK == "clock"
    assert NondeterminismClass.RNG == "rng"
    assert NondeterminismClass.ENV == "env"
    assert NondeterminismClass.NETWORK == "network"
    assert NondeterminismClass.MODEL_SAMPLING == "model_sampling"


def test_enum_is_str_subclass() -> None:
    for member in NondeterminismClass:
        assert isinstance(member, str), f"{member} must be a str subclass"


def test_all_five_classes_present() -> None:
    names = {m.value for m in NondeterminismClass}
    assert names == {"clock", "rng", "env", "network", "model_sampling"}


# ---------------------------------------------------------------------------
# Helper constructor shapes
# ---------------------------------------------------------------------------


def test_clock_nondeterminism_shape() -> None:
    d = clock_nondeterminism(observed_ns=1_715_000_000_000_000_000)
    assert d["class"] == "clock"
    assert d["observed_ns"] == 1_715_000_000_000_000_000
    assert d["controlled"] is False


def test_clock_nondeterminism_controlled() -> None:
    d = clock_nondeterminism(observed_ns=0, controlled=True)
    assert d["controlled"] is True


def test_rng_nondeterminism_shape_no_seed() -> None:
    d = rng_nondeterminism()
    assert d["class"] == "rng"
    assert d["seed"] is None
    assert d["algorithm"] == "default"


def test_rng_nondeterminism_with_seed() -> None:
    d = rng_nondeterminism(seed=42, algorithm="numpy.default_rng")
    assert d["seed"] == 42
    assert d["algorithm"] == "numpy.default_rng"


def test_env_nondeterminism_shape() -> None:
    d = env_nondeterminism({"HOME": "/home/user"})
    assert d["class"] == "env"
    assert d["observed"] == {"HOME": "/home/user"}
    assert d["controlled"] is False


def test_env_nondeterminism_controlled() -> None:
    d = env_nondeterminism({}, controlled=True)
    assert d["controlled"] is True


def test_network_nondeterminism_shape() -> None:
    d = network_nondeterminism("https://api.example.com")
    assert d["class"] == "network"
    assert d["endpoint"] == "https://api.example.com"
    assert d["controlled"] is False


def test_network_nondeterminism_controlled() -> None:
    d = network_nondeterminism("https://api.example.com", controlled=True)
    assert d["controlled"] is True


def test_model_sampling_shape_no_seed() -> None:
    d = model_sampling_nondeterminism(temperature=0.7)
    assert d["class"] == "model_sampling"
    assert d["temperature"] == 0.7
    assert d["seed"] is None


def test_model_sampling_with_seed() -> None:
    d = model_sampling_nondeterminism(temperature=0.7, seed=42)
    assert d["seed"] == 42


def test_model_sampling_zero_temperature() -> None:
    d = model_sampling_nondeterminism(temperature=0.0)
    assert d["temperature"] == 0.0


def test_combine_nondeterminism_shape() -> None:
    d = combine_nondeterminism(
        clock_nondeterminism(12345),
        model_sampling_nondeterminism(0.7, seed=None),
    )
    assert "sources" in d
    assert len(d["sources"]) == 2
    assert d["sources"][0]["class"] == "clock"
    assert d["sources"][1]["class"] == "model_sampling"


def test_combine_single_source() -> None:
    d = combine_nondeterminism(rng_nondeterminism(seed=7))
    assert len(d["sources"]) == 1


# ---------------------------------------------------------------------------
# forces_dirty semantics — single-source payloads
# ---------------------------------------------------------------------------


def test_empty_payload_not_dirty() -> None:
    assert forces_dirty({}) is False


def test_none_payload_not_dirty() -> None:
    assert forces_dirty(None) is False  # type: ignore[arg-type]


def test_unknown_format_not_dirty() -> None:
    # Legacy payload without "class" or "sources" → backward compat → clean
    assert forces_dirty({"sampling_seed": 42}) is False


# Clock
def test_clock_live_forces_dirty() -> None:
    assert forces_dirty(clock_nondeterminism(observed_ns=1)) is True


def test_clock_controlled_not_dirty() -> None:
    assert forces_dirty(clock_nondeterminism(observed_ns=1, controlled=True)) is False


# RNG
def test_rng_no_seed_forces_dirty() -> None:
    assert forces_dirty(rng_nondeterminism(seed=None)) is True


def test_rng_with_seed_not_dirty() -> None:
    assert forces_dirty(rng_nondeterminism(seed=0)) is False


def test_rng_seed_zero_not_dirty() -> None:
    # seed=0 is a valid fixed seed, not "no seed"
    assert forces_dirty(rng_nondeterminism(seed=0)) is False


# Env
def test_env_uncontrolled_forces_dirty() -> None:
    assert forces_dirty(env_nondeterminism({"HOME": "/home/user"})) is True


def test_env_controlled_not_dirty() -> None:
    assert forces_dirty(env_nondeterminism({}, controlled=True)) is False


# Network
def test_network_uncontrolled_forces_dirty() -> None:
    assert forces_dirty(network_nondeterminism("https://api.example.com")) is True


def test_network_controlled_not_dirty() -> None:
    assert forces_dirty(
        network_nondeterminism("https://api.example.com", controlled=True)
    ) is False


# Model sampling
def test_model_sampling_temp_nonzero_no_seed_forces_dirty() -> None:
    assert forces_dirty(model_sampling_nondeterminism(0.7, seed=None)) is True


def test_model_sampling_temp_nonzero_with_seed_not_dirty() -> None:
    assert forces_dirty(model_sampling_nondeterminism(0.7, seed=42)) is False


def test_model_sampling_temp_zero_no_seed_not_dirty() -> None:
    assert forces_dirty(model_sampling_nondeterminism(0.0, seed=None)) is False


def test_model_sampling_temp_zero_with_seed_not_dirty() -> None:
    assert forces_dirty(model_sampling_nondeterminism(0.0, seed=42)) is False


def test_model_sampling_temp_positive_very_small_forces_dirty() -> None:
    # temperature=1e-10 is still > 0
    assert forces_dirty(model_sampling_nondeterminism(1e-10, seed=None)) is True


# Unknown class
def test_unknown_class_forces_dirty() -> None:
    assert forces_dirty({"class": "quantum_superposition"}) is True


# ---------------------------------------------------------------------------
# forces_dirty semantics — multi-source (combine_nondeterminism)
# ---------------------------------------------------------------------------


def test_multi_source_dirty_if_any_dirty() -> None:
    # clock (dirty) + rng with seed (not dirty) → dirty
    d = combine_nondeterminism(
        clock_nondeterminism(1),
        rng_nondeterminism(seed=42),
    )
    assert forces_dirty(d) is True


def test_multi_source_clean_if_all_clean() -> None:
    d = combine_nondeterminism(
        clock_nondeterminism(1, controlled=True),
        rng_nondeterminism(seed=42),
        model_sampling_nondeterminism(0.0, seed=None),
    )
    assert forces_dirty(d) is False


def test_multi_source_empty_sources_not_dirty() -> None:
    assert forces_dirty({"sources": []}) is False


def test_multi_source_single_dirty_source_forces_dirty() -> None:
    d = combine_nondeterminism(model_sampling_nondeterminism(0.5, seed=None))
    assert forces_dirty(d) is True


# Malformed inputs
def test_malformed_payload_not_dict_forces_dirty() -> None:
    assert forces_dirty("not a dict") is True  # type: ignore[arg-type]


def test_malformed_sources_not_list_forces_dirty() -> None:
    assert forces_dirty({"sources": "bad"}) is True


def test_malformed_source_in_list_forces_dirty() -> None:
    # a malformed source (not a dict) is conservatively dirty
    assert forces_dirty({"sources": [None]}) is True


# ---------------------------------------------------------------------------
# Public alias
# ---------------------------------------------------------------------------


def test_public_alias_matches_internal() -> None:
    """stepback.nondeterminism_forces_dirty must be the same callable as forces_dirty."""
    assert nondeterminism_forces_dirty is forces_dirty


# ---------------------------------------------------------------------------
# Helper: record a 1-step trace with given nondeterminism set before write
# ---------------------------------------------------------------------------


def _make_trace_file(tmp_path: Path, nondet: dict) -> Path:
    """Record a 1-step trace where the step has the given nondeterminism payload.

    The nondeterminism is set before ``_record()`` is called so the hash is
    computed correctly and the file is written with consistent data.
    """
    sb_path = tmp_path / "trace.sb"
    with record(str(sb_path)) as rec:
        sid = rec._new_id()
        step = {
            "step_id": sid,
            "step_kind": "llm_call",
            "name": "test-model",
            "parent_step_id": None,
            "inputs": {
                "kind": "llm_call",
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
            "outputs": {
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "usage": {},
            },
            "nondeterminism": nondet,
        }
        rec._record(step)
    return sb_path


# ---------------------------------------------------------------------------
# Integration: compute_dirty_set respects nondeterminism class
# ---------------------------------------------------------------------------


def test_compute_dirty_set_clock_forces_dirty(tmp_path: Path) -> None:
    """A step with uncontrolled clock nondeterminism must appear in dirty set."""
    sb_path = _make_trace_file(tmp_path, clock_nondeterminism(12345))
    trace = replay(str(sb_path))
    summary = compute_dirty_set(trace, [])
    step1_id = trace.recorded_steps[0]["step_id"]
    entry = next(e for e in summary.entries if e.step_id == step1_id)
    assert entry.dirty
    assert entry.dirty_reason == "nondeterminism"


def test_compute_dirty_set_clock_controlled_is_clean(tmp_path: Path) -> None:
    """A controlled-clock step must be a cache hit."""
    sb_path = _make_trace_file(tmp_path, clock_nondeterminism(12345, controlled=True))
    trace = replay(str(sb_path))
    summary = compute_dirty_set(trace, [])
    step1_id = trace.recorded_steps[0]["step_id"]
    entry = next(e for e in summary.entries if e.step_id == step1_id)
    assert not entry.dirty


def test_compute_dirty_set_rng_no_seed_forces_dirty(tmp_path: Path) -> None:
    sb_path = _make_trace_file(tmp_path, rng_nondeterminism(seed=None))
    trace = replay(str(sb_path))
    summary = compute_dirty_set(trace, [])
    entry = next(e for e in summary.entries)
    assert entry.dirty
    assert entry.dirty_reason == "nondeterminism"


def test_compute_dirty_set_rng_with_seed_is_clean(tmp_path: Path) -> None:
    sb_path = _make_trace_file(tmp_path, rng_nondeterminism(seed=7))
    trace = replay(str(sb_path))
    summary = compute_dirty_set(trace, [])
    entry = next(e for e in summary.entries)
    assert not entry.dirty


def test_compute_dirty_set_network_uncontrolled_forces_dirty(tmp_path: Path) -> None:
    sb_path = _make_trace_file(tmp_path, network_nondeterminism("https://api.example.com"))
    trace = replay(str(sb_path))
    summary = compute_dirty_set(trace, [])
    entry = next(e for e in summary.entries)
    assert entry.dirty
    assert entry.dirty_reason == "nondeterminism"


def test_compute_dirty_set_model_sampling_temp_high_no_seed(tmp_path: Path) -> None:
    sb_path = _make_trace_file(tmp_path, model_sampling_nondeterminism(1.0, seed=None))
    trace = replay(str(sb_path))
    summary = compute_dirty_set(trace, [])
    entry = next(e for e in summary.entries)
    assert entry.dirty
    assert entry.dirty_reason == "nondeterminism"


def test_compute_dirty_set_model_sampling_seeded_is_clean(tmp_path: Path) -> None:
    sb_path = _make_trace_file(tmp_path, model_sampling_nondeterminism(0.7, seed=42))
    trace = replay(str(sb_path))
    summary = compute_dirty_set(trace, [])
    entry = next(e for e in summary.entries)
    assert not entry.dirty


def test_compute_dirty_set_multi_source_forces_dirty(tmp_path: Path) -> None:
    nondet = combine_nondeterminism(
        clock_nondeterminism(1),               # dirty
        model_sampling_nondeterminism(0.0, seed=42),  # clean
    )
    sb_path = _make_trace_file(tmp_path, nondet)
    trace = replay(str(sb_path))
    summary = compute_dirty_set(trace, [])
    entry = next(e for e in summary.entries)
    assert entry.dirty
    assert entry.dirty_reason == "nondeterminism"


def test_compute_dirty_set_empty_nondet_no_effect(tmp_path: Path) -> None:
    """An empty nondeterminism payload must not force dirty."""
    sb_path = _make_trace_file(tmp_path, {})
    trace = replay(str(sb_path))
    summary = compute_dirty_set(trace, [])
    entry = next(e for e in summary.entries)
    # Inputs hash matches (first step, no parent) → clean
    assert not entry.dirty


# ---------------------------------------------------------------------------
# Integration: replay engine respects nondeterminism class
# ---------------------------------------------------------------------------


def test_replay_clock_dirty_step_is_reexecuted(tmp_path: Path) -> None:
    """Replay must re-execute a step whose nondeterminism class forces dirty."""
    sb_path = _make_trace_file(tmp_path, clock_nondeterminism(12345, controlled=False))
    trace = replay(str(sb_path))

    call_count = [0]

    def counting_llm(model: str, messages: list) -> dict:
        call_count[0] += 1
        return {"choices": [{"message": {"role": "assistant", "content": "new"}}], "usage": {}}

    result = trace.replay_forward(Executor(llm=counting_llm))
    assert result.real_executions >= 1
    assert call_count[0] >= 1


def test_replay_controlled_clock_is_cache_hit(tmp_path: Path) -> None:
    """A controlled-clock step must be a cache hit on replay."""
    sb_path = _make_trace_file(tmp_path, clock_nondeterminism(12345, controlled=True))
    trace = replay(str(sb_path))
    result = trace.replay_forward(Executor(fallback_recorded=True))
    assert result.cache_hit_count >= 1
    assert result.real_executions == 0


def test_replay_model_sampling_seeded_is_cache_hit(tmp_path: Path) -> None:
    """temperature=0.7/seed=42 step must be a cache hit."""
    nondet = combine_nondeterminism(model_sampling_nondeterminism(0.7, seed=42))
    sb_path = _make_trace_file(tmp_path, nondet)
    trace = replay(str(sb_path))
    result = trace.replay_forward(Executor(fallback_recorded=True))
    assert result.cache_hit_count >= 1
    assert result.real_executions == 0


def test_replay_model_sampling_unseeded_forces_reexecution(tmp_path: Path) -> None:
    """temperature=0.5/seed=None step must be re-executed."""
    nondet = combine_nondeterminism(model_sampling_nondeterminism(0.5, seed=None))
    sb_path = _make_trace_file(tmp_path, nondet)
    trace = replay(str(sb_path))

    call_count = [0]

    def llm(model: str, messages: list) -> dict:
        call_count[0] += 1
        return {"choices": [{"message": {"role": "assistant", "content": "new"}}], "usage": {}}

    result = trace.replay_forward(Executor(llm=llm))
    assert result.real_executions >= 1
    assert call_count[0] >= 1


# ---------------------------------------------------------------------------
# Auto-recording of model_sampling in Recorder.llm_call
# ---------------------------------------------------------------------------


def test_recorder_llm_call_auto_records_model_sampling(tmp_path: Path) -> None:
    """llm_call must auto-populate nondeterminism with model_sampling source."""
    sb_path = tmp_path / "trace.sb"
    with record(str(sb_path)) as rec:
        step = rec.llm_call(
            "test-model",
            [{"role": "user", "content": "hello"}],
            fake_llm,
            temperature=0.7,
            seed=None,
        )

    nondet = step.get("nondeterminism", {})
    assert nondet, "nondeterminism field must be populated"
    sources = nondet.get("sources")
    if sources:
        classes = [s.get("class") for s in sources]
        assert "model_sampling" in classes
        ms = next(s for s in sources if s.get("class") == "model_sampling")
    else:
        assert nondet.get("class") == "model_sampling"
        ms = nondet
    assert ms["temperature"] == 0.7
    assert ms["seed"] is None


def test_recorder_llm_call_seeded_model_sampling(tmp_path: Path) -> None:
    """llm_call with seed=42 must record model_sampling with seed=42."""
    sb_path = tmp_path / "trace.sb"
    with record(str(sb_path)) as rec:
        step = rec.llm_call(
            "test-model",
            [{"role": "user", "content": "hello"}],
            fake_llm,
            temperature=0.0,
            seed=42,
        )

    nondet = step.get("nondeterminism", {})
    sources = nondet.get("sources", [])
    ms_sources = [s for s in sources if s.get("class") == "model_sampling"]
    assert ms_sources, "model_sampling source must be present"
    assert ms_sources[0]["seed"] == 42


def test_recorder_llm_call_nondeterminism_hash_matches(tmp_path: Path) -> None:
    """nondeterminism_hash in the recorded step must match the nondeterminism payload."""
    sb_path = tmp_path / "trace.sb"
    with record(str(sb_path)) as rec:
        step = rec.llm_call(
            "test-model",
            [{"role": "user", "content": "hello"}],
            fake_llm,
            temperature=0.5,
            seed=None,
        )

    expected_hash = sha256_hex(canonical_json(step["nondeterminism"]))
    assert step["nondeterminism_hash"] == expected_hash


def test_recorder_default_temp_seed_not_dirty(tmp_path: Path) -> None:
    """llm_call with temperature=0.0/seed=42 must not force dirty."""
    sb_path = tmp_path / "trace.sb"
    with record(str(sb_path)) as rec:
        step = rec.llm_call(
            "test-model",
            [{"role": "user", "content": "hello"}],
            fake_llm,
            temperature=0.0,
            seed=42,
        )

    nondet = step.get("nondeterminism", {})
    assert not forces_dirty(nondet), (
        f"temperature=0/seed=42 must not force dirty; nondet={nondet}"
    )


def test_recorder_high_temp_no_seed_forces_dirty(tmp_path: Path) -> None:
    """llm_call with temperature=0.7/seed=None must produce dirty-forcing payload."""
    sb_path = tmp_path / "trace.sb"
    with record(str(sb_path)) as rec:
        step = rec.llm_call(
            "test-model",
            [{"role": "user", "content": "hello"}],
            fake_llm,
            temperature=0.7,
            seed=None,
        )

    nondet = step.get("nondeterminism", {})
    assert forces_dirty(nondet), (
        f"temperature=0.7/seed=None must force dirty; nondet={nondet}"
    )
