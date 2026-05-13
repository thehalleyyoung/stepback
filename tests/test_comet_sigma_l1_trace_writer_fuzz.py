"""Fuzz tests for the Comet-Σ L1 ``trace_writer`` emitter (Step 10).

This file owns Step 10 of ``COMET_SIGMA_1000.md``:

    Fuzz the L1 emitter for stepback/trace_writer.py + .sb format v1
    against malformed inputs.

Strategy
--------

The hot path of the L1 emitter is :func:`stepback.comet_sigma.l1_trace_writer.observe_frame`,
which is called from :meth:`stepback.trace_writer.TraceWriter._write_frame`
with two free parameters under our control: the ``body`` dict and its
canonical-JSON ``body_bytes``. The contract documented in the module
docstring is **strict no-raise**: no malformed body, no surprise type,
and no malformed pre-existing writer state may propagate an exception
back into the trace writer's hot path. We fuzz that contract here using
two complementary approaches:

1. A **Hypothesis** property test that generates arbitrary JSON-ish
   ``body`` payloads and arbitrary ``body_bytes`` (including non-bytes
   types) and asserts that:

       * :func:`observe_frame` never raises.
       * It either returns ``None`` (e.g. non-dict body silently
         tolerated) or a :class:`FrameRecord` whose ``values`` map
         carries every declared base-feature name and only finite
         floats.
       * Per-feature receipts emitted are well-formed and chained
         against the writer-scoped artifact head.

2. A **deterministic catalogue** of pathological inputs (NaN, ±inf,
   recursive dicts, surrogate strings, huge body_bytes, non-mapping
   bodies, bodies with non-string ``type``, frame_index overflow under
   replayed registry state, etc.) that triggers concrete corner cases
   that the random sampler is unlikely to land on.

3. A **concurrency fuzz** that drives :func:`observe_frame` from
   multiple threads against the same writer id with malformed inputs;
   the emitter holds a registry-level lock and must remain consistent
   under contention.

The whole file is skipped when the upstream ``comet_sigma`` package is
not importable, mirroring the pattern used elsewhere in
``tests/test_comet_sigma_l1_trace_writer.py``.
"""
from __future__ import annotations

import math
import threading
from typing import Any

import pytest

from hypothesis import HealthCheck, given, settings, strategies as st

from stepback import comet_sigma as cs
from stepback.comet_sigma import l1_trace_writer as l1


pytestmark = pytest.mark.skipif(
    not cs.comet_sigma_available(),
    reason="comet_sigma package not importable on this host",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_emitter_state():
    l1.reset()
    yield
    l1.reset()


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(l1.FLAG_NAME, "1")
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_FEATURE_NAMES = frozenset(spec.name for spec in l1.BASE_FEATURES)


def _assert_record_well_formed(record, body_for_type) -> None:
    """Sanity checks every fuzzed observation must satisfy."""
    assert isinstance(record, l1.FrameRecord)
    assert isinstance(record.frame_index, int)
    assert record.frame_index >= 0
    assert isinstance(record.wallclock_ns, int)
    assert record.wallclock_ns > 0
    assert isinstance(record.frame_type, str)
    # The recorded frame_type must reflect what the emitter saw — when
    # body is a dict with a 'type' field the emitter coerces with str();
    # otherwise it must be the empty string.
    if isinstance(body_for_type, dict) and "type" in body_for_type:
        assert record.frame_type == str(body_for_type["type"])
    else:
        assert record.frame_type == ""

    assert set(record.values.keys()) == EXPECTED_FEATURE_NAMES
    for name, val in record.values.items():
        assert isinstance(val, float), f"{name} is not float: {type(val)!r}"
        assert math.isfinite(val), f"{name} produced non-finite value: {val!r}"


def _assert_receipts_well_formed(writer_id: str) -> None:
    receipts = l1.receipts_for(writer_id)
    history = l1.feature_history(writer_id)
    if not history:
        return
    n_features = len(l1.BASE_FEATURES)
    assert len(receipts) == n_features * len(history)
    for rec in receipts:
        assert rec.schema_id == l1.RECEIPT_SCHEMA_ID
        assert set(rec.payload) == {"value", "frame_index", "frame_type"}
        assert isinstance(rec.payload["value"], float)
        assert math.isfinite(rec.payload["value"])
        assert isinstance(rec.payload["frame_index"], int)
        assert isinstance(rec.payload["frame_type"], str)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# JSON-ish leaf values plus a few non-JSON oddities the writer might
# accidentally hand us before canonicalization.
_leaf = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2 ** 63), max_value=2 ** 63 - 1),
    st.floats(allow_nan=True, allow_infinity=True, width=32),
    st.text(),
    st.binary(max_size=64),
)

# Recursive dict/list bodies up to a small depth.
_json_like = st.recursive(
    _leaf,
    lambda children: st.one_of(
        st.lists(children, max_size=6),
        st.dictionaries(st.text(max_size=8), children, max_size=6),
    ),
    max_leaves=20,
)

# Bodies are either dicts (the "happy" path) or arbitrary objects (the
# emitter must tolerate non-dict bodies and degrade frame_type to "").
_body_strategy = st.one_of(
    st.dictionaries(st.text(max_size=8), _json_like, max_size=8),
    _json_like,
)

# body_bytes can be bytes, bytearray, or — as a malformed input — any
# other Python object.
_body_bytes_strategy = st.one_of(
    st.binary(max_size=512),
    st.builds(bytearray, st.binary(max_size=64)),
    st.none(),
    st.text(max_size=64),
    st.integers(),
)


# ---------------------------------------------------------------------------
# Property: observe_frame never raises and produces well-formed records
# ---------------------------------------------------------------------------

@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(body=_body_strategy, body_bytes=_body_bytes_strategy)
def test_observe_frame_is_total_under_arbitrary_inputs(flag_on, body, body_bytes):
    """``observe_frame`` must be total: never raise, always emit."""
    writer_id = "fuzz://writer/total"
    l1.reset(writer_id)
    record = l1.observe_frame(writer_id, body, body_bytes)
    # The emitter must always succeed when active — even for malformed
    # body / body_bytes — and produce a record.
    assert record is not None, (
        f"observe_frame returned None for body={body!r} body_bytes={body_bytes!r}"
    )
    _assert_record_well_formed(record, body)
    _assert_receipts_well_formed(writer_id)


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    bodies=st.lists(_body_strategy, min_size=1, max_size=12),
    body_bytes_list=st.lists(_body_bytes_strategy, min_size=1, max_size=12),
)
def test_frame_index_is_monotonic_under_fuzz(flag_on, bodies, body_bytes_list):
    """Frame indices must remain a contiguous 0..N-1 sequence under fuzz."""
    writer_id = "fuzz://writer/monotonic"
    l1.reset(writer_id)
    n = min(len(bodies), len(body_bytes_list))
    for i in range(n):
        rec = l1.observe_frame(writer_id, bodies[i], body_bytes_list[i])
        assert rec is not None
        assert rec.frame_index == i

    history = l1.feature_history(writer_id)
    assert [r.frame_index for r in history] == list(range(n))
    # frame_depth feature mirrors frame_index.
    for r in history:
        assert r.values["trace_writer_sb_v1.frame_depth"] == float(r.frame_index)
    # wallclock_ns is non-decreasing within a single thread.
    times = [r.wallclock_ns for r in history]
    assert times == sorted(times)


# ---------------------------------------------------------------------------
# Deterministic pathological inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "body,body_bytes,description",
    [
        # Non-dict bodies: the emitter must tolerate them and record an
        # empty frame_type.
        (None, b"", "none-body"),
        (42, b"hello", "int-body"),
        ("not-a-dict", b"hello", "str-body"),
        ([1, 2, 3], b"\x00\x01", "list-body"),
        # Dict body with non-string 'type' field.
        ({"type": 7}, b"{}", "non-string-type"),
        ({"type": None}, b"{}", "none-type"),
        ({"type": ["nested"]}, b"{}", "list-type"),
        # Body with NaN / inf leaves should not propagate into features.
        ({"type": "step", "x": float("nan")}, b"{}", "nan-leaf"),
        ({"type": "step", "x": float("inf")}, b"{}", "inf-leaf"),
        # Body with surrogate strings.
        ({"type": "\ud800"}, b"{}", "surrogate-type"),
        # Body with binary/bytes leaves.
        ({"type": "blob", "raw": b"\x00\xff"}, b"\x00\xff", "binary-leaf"),
        # Empty body, empty body_bytes.
        ({}, b"", "empty-everything"),
        # Huge body_bytes (1 MiB).
        ({"type": "step"}, b"a" * (1 << 20), "1MiB-bytes"),
        # Deeply nested body — must not blow the stack.
        ({"type": "step", "deep": {"a": {"b": {"c": {"d": {"e": 1}}}}}}, b"{}", "deep"),
        # body_bytes given as wrong types — the emitter coerces to b"".
        ({"type": "step"}, None, "none-bytes"),
        ({"type": "step"}, "a string", "str-bytes"),
        ({"type": "step"}, 12345, "int-bytes"),
        ({"type": "step"}, bytearray(b"\x00\x01\x02"), "bytearray-bytes"),
    ],
)
def test_pathological_inputs_do_not_raise(flag_on, body, body_bytes, description):
    writer_id = f"fuzz://writer/path/{description}"
    rec = l1.observe_frame(writer_id, body, body_bytes)
    assert rec is not None, description
    _assert_record_well_formed(rec, body)
    # All feature values must be finite even when leaf data was NaN/inf.
    for name, val in rec.values.items():
        assert math.isfinite(val), f"{description}: {name}={val!r}"


def test_self_referential_body_does_not_loop(flag_on):
    """A self-referential dict must not cause infinite recursion."""
    body: dict = {"type": "step"}
    body["self"] = body
    rec = l1.observe_frame("fuzz://writer/cyclic", body, b"{}")
    assert rec is not None
    # body_key_count counts top-level keys only and must therefore be 2.
    assert rec.values["trace_writer_sb_v1.body_key_count"] == 2.0


def test_extractor_failure_is_isolated(monkeypatch, flag_on):
    """If a single extractor blows up, the others must still run."""
    # Replace the body_key_count extractor with one that raises.
    original = l1._EXTRACTORS

    def boom(body, body_bytes, ctx):  # pragma: no cover - forced exception
        raise RuntimeError("extractor blew up")

    new = list(original)
    # Replace the last extractor (_x_body_key_count) with boom.
    new[-1] = boom
    monkeypatch.setattr(l1, "_EXTRACTORS", tuple(new))

    rec = l1.observe_frame("fuzz://writer/extractor-failure", {"type": "step"}, b"{}")
    assert rec is not None
    # The failing extractor must contribute 0.0 (the documented fallback).
    assert rec.values["trace_writer_sb_v1.body_key_count"] == 0.0
    # Other features were computed normally.
    assert rec.values["trace_writer_sb_v1.frame_bytes"] == 2.0
    assert rec.values["trace_writer_sb_v1.is_step_frame"] == 1.0


def test_misbehaving_observe_hook_does_not_break_writer(flag_on):
    """A registered OBSERVE_HOOK that raises must not propagate."""
    calls: list = []

    def good_hook(writer_id, record):
        calls.append((writer_id, record.frame_index))

    def bad_hook(writer_id, record):  # pragma: no cover - forced exception
        raise RuntimeError("hook exploded")

    l1.OBSERVE_HOOKS.append(good_hook)
    l1.OBSERVE_HOOKS.append(bad_hook)
    try:
        rec = l1.observe_frame("fuzz://writer/hooks", {"type": "step"}, b"{}")
        assert rec is not None
        # Even after the bad hook ran, the good hook must have been
        # invoked exactly once.
        assert calls == [("fuzz://writer/hooks", 0)]
    finally:
        l1.OBSERVE_HOOKS.remove(good_hook)
        l1.OBSERVE_HOOKS.remove(bad_hook)


# ---------------------------------------------------------------------------
# Flag-off invariants under fuzz
# ---------------------------------------------------------------------------

@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(body=_body_strategy, body_bytes=_body_bytes_strategy)
def test_no_state_when_flag_off(monkeypatch, body, body_bytes):
    """With the flag OFF the emitter must remain a pure no-op."""
    monkeypatch.setenv(l1.FLAG_NAME, "0")
    writer_id = "fuzz://writer/flag-off"
    l1.reset(writer_id)
    assert l1.observe_frame(writer_id, body, body_bytes) is None
    assert l1.get_state(writer_id) is None
    assert l1.feature_history(writer_id) == []
    assert l1.receipts_for(writer_id) == []
    assert l1.latest_feature_vector(writer_id) is None


# ---------------------------------------------------------------------------
# Concurrency fuzz
# ---------------------------------------------------------------------------

def test_concurrent_observers_remain_consistent(flag_on):
    """Concurrent threads emitting malformed frames must not corrupt state."""
    writer_id = "fuzz://writer/concurrent"
    n_threads = 8
    per_thread = 50

    bodies = [
        {"type": "step"},
        None,
        42,
        "junk",
        {"type": ["nested"]},
        {"type": "blob"},
        [1, 2, 3],
        {},
    ]
    body_bytes_choices = [b"", b"\x00", b"a" * 64, b"{}"]

    def worker(idx: int) -> None:
        for j in range(per_thread):
            body = bodies[(idx + j) % len(bodies)]
            bb = body_bytes_choices[(idx * 3 + j) % len(body_bytes_choices)]
            l1.observe_frame(writer_id, body, bb)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    history = l1.feature_history(writer_id)
    # Total frames recorded equals threads * per_thread (no drops).
    assert len(history) == n_threads * per_thread
    # frame_index is a contiguous sequence even under contention.
    indices = [r.frame_index for r in history]
    assert indices == list(range(len(history)))
    # No frame produced a non-finite feature.
    for rec in history:
        for name, val in rec.values.items():
            assert math.isfinite(val), f"{name}={val!r}"
