"""Tests for ``stepback.comet_sigma.l1_trace_writer`` (Step 1).

These tests cover the wiring of :class:`stepback.trace_writer.TraceWriter`
as a Comet-Σ L1 ``temporal_basis`` feature emitter.
"""
from __future__ import annotations

import os
import pathlib
import pytest

from stepback import comet_sigma as cs
from stepback.comet_sigma import l1_trace_writer as l1
from stepback.trace_writer import TraceWriter


# Skip the whole file when the upstream comet_sigma package isn't on the
# import path — keeping the test file collectable so users without
# kitchensink installed still get an explicit "skipped" line rather than
# a missing-module ImportError.
pytestmark = pytest.mark.skipif(
    not cs.comet_sigma_available(),
    reason="comet_sigma package not importable on this host",
)


@pytest.fixture(autouse=True)
def _clean_emitter_state():
    l1.reset()
    yield
    l1.reset()


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(l1.FLAG_NAME, "1")
    return True


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.setenv(l1.FLAG_NAME, "0")
    return False


def _write_a_few_frames(tmp_path: pathlib.Path) -> str:
    path = str(tmp_path / "trace.sb")
    w = TraceWriter.open(path, signing=False)
    for i in range(3):
        w.write_step({"type": "step", "i": i, "payload": "x" * 16})
    w.close()
    return path


def test_default_flag_is_off_and_emitter_is_inactive():
    # When the flag has not been set, the upstream comet_sigma default
    # is OFF (see comet_sigma/flags.py); is_active() must agree.
    os.environ.pop(l1.FLAG_NAME, None)
    assert cs.is_enabled(l1.FLAG_NAME) is False
    assert l1.is_active() is False


def test_writer_is_a_noop_when_flag_off(tmp_path, flag_off):
    path = _write_a_few_frames(tmp_path)
    # No state should have been registered for this writer.
    assert l1.get_state(path) is None
    assert l1.feature_history(path) == []
    assert l1.receipts_for(path) == []
    assert l1.latest_feature_vector(path) is None


def test_writer_emits_features_when_flag_on(tmp_path, flag_on):
    path = _write_a_few_frames(tmp_path)
    history = l1.feature_history(path)
    # 1 header + 3 step frames + 1 merkle_summary + 1 tail = 6 frames
    assert len(history) >= 5
    types = [r.frame_type for r in history]
    assert types[0] == "header"
    assert "step" in types
    assert "tail" in types

    # Every record carries the full base-feature vector.
    expected_names = {spec.name for spec in l1.BASE_FEATURES}
    for rec in history:
        assert set(rec.values.keys()) == expected_names
        # is_step / is_blob / is_header are 0/1 indicators.
        assert rec.values["trace_writer_sb_v1.is_header_frame"] in (0.0, 1.0)
        assert rec.values["trace_writer_sb_v1.is_step_frame"] in (0.0, 1.0)
        assert rec.values["trace_writer_sb_v1.is_blob_frame"] in (0.0, 1.0)
        # frame_bytes is positive for every non-empty frame body.
        assert rec.values["trace_writer_sb_v1.frame_bytes"] > 0.0

    # frame_depth is monotonic and starts at 0.
    depths = [r.values["trace_writer_sb_v1.frame_depth"] for r in history]
    assert depths == sorted(depths)
    assert depths[0] == 0.0

    # prev_dt_ns is 0 for the very first frame.
    assert history[0].values["trace_writer_sb_v1.prev_dt_ns"] == 0.0


def test_artifacts_have_provenance_and_schema(tmp_path, flag_on):
    path = _write_a_few_frames(tmp_path)
    arts = l1.artifacts_for(path)
    assert len(arts) == len(l1.BASE_FEATURES)
    for art, spec in zip(arts, l1.BASE_FEATURES):
        assert art.kind == "feature"
        assert art.name == spec.name
        assert art.receipt_schema["id"] == l1.RECEIPT_SCHEMA_ID
        head = art.provenance.head()
        assert head is not None
        assert head.layer == "L1"
        assert head.module == l1.MODULE_LABEL
        # src_sha256 must be the sha256 of the spec's literal source.
        import hashlib
        assert head.src_sha256 == hashlib.sha256(spec.src.encode()).hexdigest()


def test_receipts_match_artifacts_and_payload_shape(tmp_path, flag_on):
    path = _write_a_few_frames(tmp_path)
    receipts = l1.receipts_for(path)
    assert receipts, "expected receipts to be emitted"
    n_features = len(l1.BASE_FEATURES)
    history = l1.feature_history(path)
    assert len(receipts) == n_features * len(history)

    for rec in receipts:
        assert rec.schema_id == l1.RECEIPT_SCHEMA_ID
        assert set(rec.payload) == {"value", "frame_index", "frame_type"}
        assert isinstance(rec.payload["value"], float)
        assert isinstance(rec.payload["frame_index"], int)
        assert isinstance(rec.payload["frame_type"], str)
        # Receipts can be replayed against their declaring artifact.
        # We look up the artifact by name.
        from stepback.comet_sigma import (  # local import to avoid module load order issues
            l1_trace_writer as _l1,
        )
        arts = {a.name: a for a in _l1.artifacts_for(path)}
        try:
            from comet_sigma.audit import replay_receipt as _replay  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            from kitchensink.comet_sigma.audit import replay_receipt as _replay  # type: ignore[import-not-found]
        assert _replay(arts[rec.artifact_name], rec) is True


def test_latest_feature_vector_round_trip(tmp_path, flag_on):
    path = _write_a_few_frames(tmp_path)
    latest = l1.latest_feature_vector(path)
    assert latest is not None
    assert latest == l1.feature_history(path)[-1].values


def test_ring_buffer_is_bounded(monkeypatch, tmp_path, flag_on):
    monkeypatch.setattr(l1, "MAX_RECENT_FRAMES", 4)
    # Re-open a writer; ring buffer is sized at WriterState construction.
    path = str(tmp_path / "small.sb")
    w = TraceWriter.open(path, signing=False)
    # Force the WriterState to be (re-)created with the patched cap by
    # explicitly priming it via observe_frame (the writer's header frame
    # already created one with the OLD cap, so reset first).
    l1.reset(path)
    for i in range(20):
        w.write_step({"type": "step", "i": i})
    w.close()
    history = l1.feature_history(path)
    # The ring buffer must never exceed the configured cap.
    assert len(history) <= 4


def test_known_writers_listed(tmp_path, flag_on):
    p1 = _write_a_few_frames(tmp_path)
    p2 = str(tmp_path / "second.sb")
    w = TraceWriter.open(p2, signing=False)
    w.write_step({"type": "step", "i": 0})
    w.close()
    known = set(l1.known_writers())
    assert {p1, p2} <= known


def test_emitter_does_not_corrupt_trace(tmp_path, flag_on):
    """The emitter must be read-only with respect to the on-disk trace."""
    from stepback.trace_reader import read_frames

    path = _write_a_few_frames(tmp_path)
    # Reading the trace back must succeed exactly as in the no-emitter
    # case — the L1 hook must never mutate body bytes.
    frames = list(read_frames(path))
    assert frames, "trace must contain frames"
    assert any(f["body"].get("type") == "step" for f in frames)


def test_observe_frame_tolerates_garbage_inputs(flag_on):
    # Non-dict body and non-bytes payload must NOT raise; emitter must
    # return None or a record with zeroed-out feature values.
    rec = l1.observe_frame("garbage", "not a dict", "not bytes")  # type: ignore[arg-type]
    assert rec is None or set(rec.values) == {s.name for s in l1.BASE_FEATURES}


def test_flags_module_lists_l1_temporal():
    flags = cs.all_flags()
    assert l1.FLAG_NAME in flags
