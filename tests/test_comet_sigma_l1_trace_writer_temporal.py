"""Tests for ``stepback.comet_sigma.l1_trace_writer_temporal`` (Step 2).

Covers the temporal-basis projection (1s/10s/1m/10m wall-clock windows)
over the per-frame base features emitted by the Step 1 wiring of
:class:`stepback.trace_writer.TraceWriter`.
"""
from __future__ import annotations

import pathlib

import pytest

from stepback import comet_sigma as cs
from stepback.comet_sigma import l1_trace_writer as l1
from stepback.comet_sigma import l1_trace_writer_temporal as l1t
from stepback.trace_writer import TraceWriter


pytestmark = pytest.mark.skipif(
    not cs.comet_sigma_available(),
    reason="comet_sigma package not importable on this host",
)


@pytest.fixture(autouse=True)
def _clean_state():
    l1.reset()
    l1t.reset()
    # Ensure the auto-projection hook is installed exactly once.
    l1t.install_hook()
    yield
    l1.reset()
    l1t.reset()


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(l1.FLAG_NAME, "1")
    return True


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.setenv(l1.FLAG_NAME, "0")
    return False


def _write_a_few_frames(tmp_path: pathlib.Path, n: int = 4) -> str:
    path = str(tmp_path / "trace.sb")
    w = TraceWriter.open(path, signing=False)
    for i in range(n):
        w.write_step({"type": "step", "i": i, "payload": "x" * 16})
    w.close()
    return path


# --------------------------------------------------------------------------
# Flag handling
# --------------------------------------------------------------------------

def test_default_is_inactive_when_flag_off(tmp_path, flag_off):
    path = _write_a_few_frames(tmp_path)
    assert l1t.is_active() is False
    assert l1t.get_state(path) is None
    assert l1t.latest_projection(path) is None
    assert l1t.projection_history(path) == []
    assert l1t.receipts_for(path) == []


def test_window_set_matches_spec():
    names = [name for name, _ in l1t.WINDOWS_NS]
    assert names == ["1s", "10s", "1m", "10m"]
    # Windows must be strictly increasing in nanoseconds.
    nss = [ns for _, ns in l1t.WINDOWS_NS]
    assert nss == sorted(nss)
    assert nss == [1_000_000_000, 10_000_000_000, 60_000_000_000, 600_000_000_000]


# --------------------------------------------------------------------------
# Auto-projection hook
# --------------------------------------------------------------------------

def test_auto_hook_runs_per_frame(tmp_path, flag_on):
    path = _write_a_few_frames(tmp_path, n=3)
    hist = l1t.projection_history(path)
    # one projection per observed frame (header + 3 step + merkle + tail)
    assert len(hist) == len(l1.feature_history(path))
    assert hist == sorted(hist, key=lambda r: r.frame_index)


def test_install_hook_is_idempotent():
    n_before = sum(1 for h in l1.OBSERVE_HOOKS if h is l1t._hook)
    assert n_before == 1
    assert l1t.install_hook() is False
    n_after = sum(1 for h in l1.OBSERVE_HOOKS if h is l1t._hook)
    assert n_after == 1


def test_uninstall_hook_then_reinstall(tmp_path, flag_on):
    assert l1t.uninstall_hook() is True
    # Now writing frames should NOT auto-project.
    path = _write_a_few_frames(tmp_path, n=2)
    assert l1t.get_state(path) is None
    # Re-install for cleanliness.
    assert l1t.install_hook() is True


# --------------------------------------------------------------------------
# Projection contents
# --------------------------------------------------------------------------

def test_projection_record_covers_every_feature_window_aggregate(tmp_path, flag_on):
    path = _write_a_few_frames(tmp_path, n=3)
    rec = l1t.latest_projection(path)
    assert rec is not None

    expected_keys = set(l1t.projection_names())
    assert set(rec.values.keys()) == expected_keys

    n_features = len(l1.BASE_FEATURES)
    n_windows = len(l1t.WINDOWS_NS)
    n_aggs = len(l1t.AGGREGATES)
    assert len(expected_keys) == n_features * n_windows * n_aggs

    # window_counts populated for every named window.
    assert set(rec.window_counts.keys()) == {w for w, _ in l1t.WINDOWS_NS}
    # The 10m window must contain everything inside the 1s window.
    assert rec.window_counts["10m"] >= rec.window_counts["1s"]
    # Largest window must include every observed frame in this short test.
    assert rec.window_counts["10m"] == len(l1.feature_history(path))


def test_projection_values_are_finite_floats(tmp_path, flag_on):
    import math
    path = _write_a_few_frames(tmp_path, n=4)
    rec = l1t.latest_projection(path)
    assert rec is not None
    for k, v in rec.values.items():
        assert isinstance(v, float)
        assert math.isfinite(v), f"{k} = {v}"


def test_mean_aggregate_matches_python(tmp_path, flag_on):
    path = _write_a_few_frames(tmp_path, n=3)
    history = l1.feature_history(path)
    rec = l1t.latest_projection(path)
    assert rec is not None
    feature = "trace_writer_sb_v1.frame_bytes"
    # 10m window covers everything in this short test.
    expected = sum(r.values[feature] for r in history) / len(history)
    key = l1t.projection_name(feature, "10m", "mean")
    assert rec.values[key] == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_range_aggregate_matches_python(tmp_path, flag_on):
    path = _write_a_few_frames(tmp_path, n=4)
    history = l1.feature_history(path)
    rec = l1t.latest_projection(path)
    assert rec is not None
    feature = "trace_writer_sb_v1.frame_bytes"
    xs = [r.values[feature] for r in history]
    expected = max(xs) - min(xs)
    key = l1t.projection_name(feature, "10m", "range")
    assert rec.values[key] == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_frame_depth_slope_is_nonnegative(tmp_path, flag_on):
    # frame_depth is monotonic non-decreasing, so its slope across any
    # window must be >= 0.
    path = _write_a_few_frames(tmp_path, n=5)
    rec = l1t.latest_projection(path)
    assert rec is not None
    for win, _ in l1t.WINDOWS_NS:
        key = l1t.projection_name("trace_writer_sb_v1.frame_depth", win, "slope")
        assert rec.values[key] >= 0.0


# --------------------------------------------------------------------------
# Provenance + receipts
# --------------------------------------------------------------------------

def test_artifacts_have_provenance_and_distinct_src_per_aggregate(tmp_path, flag_on):
    path = _write_a_few_frames(tmp_path, n=2)
    arts = l1t.artifacts_for(path)
    expected_keys = set(l1t.projection_names())
    assert set(arts.keys()) == expected_keys

    # src_sha256 must group by aggregate (same aggregate => same src).
    import hashlib
    by_agg: dict = {}
    for spec in l1.BASE_FEATURES:
        for win, _ in l1t.WINDOWS_NS:
            for agg in l1t.AGGREGATES:
                k = l1t.projection_name(spec.name, win, agg.name)
                head = arts[k].provenance.head()
                assert head is not None
                assert head.layer == "L1"
                assert head.module == l1t.MODULE_LABEL
                expected_sha = hashlib.sha256(agg.src.encode()).hexdigest()
                assert head.src_sha256 == expected_sha
                by_agg.setdefault(agg.name, set()).add(head.src_sha256)
    for agg_name, shas in by_agg.items():
        assert len(shas) == 1, f"aggregate {agg_name} has multiple src hashes"


def test_receipts_replay(tmp_path, flag_on):
    path = _write_a_few_frames(tmp_path, n=2)
    receipts = l1t.receipts_for(path)
    arts = l1t.artifacts_for(path)
    assert receipts
    try:
        from comet_sigma.audit import replay_receipt
    except ModuleNotFoundError:
        from kitchensink.comet_sigma.audit import replay_receipt  # type: ignore[import-not-found]
    # Sample-replay a handful per aggregate to keep the test fast.
    sampled = receipts[:: max(1, len(receipts) // 50)]
    for r in sampled:
        assert r.schema_id == l1t.RECEIPT_SCHEMA_ID
        assert set(r.payload) == {
            "value", "window", "window_ns", "n_frames_in_window", "frame_index",
        }
        assert r.payload["window"] in {w for w, _ in l1t.WINDOWS_NS}
        assert isinstance(r.payload["n_frames_in_window"], int)
        assert r.payload["n_frames_in_window"] >= 1
        art = arts[r.artifact_name]
        assert replay_receipt(art, r) is True


def test_receipt_count_matches_projection_count(tmp_path, flag_on):
    path = _write_a_few_frames(tmp_path, n=3)
    n_artifacts = len(l1t.artifacts_for(path))
    n_projections = len(l1t.projection_history(path))
    assert n_projections > 0
    assert len(l1t.receipts_for(path)) == n_artifacts * n_projections


# --------------------------------------------------------------------------
# Window slicing semantics
# --------------------------------------------------------------------------

def test_window_slice_filters_by_wallclock(flag_on):
    # Hand-construct frame records spanning > 1s but < 10s.
    base_ns = 1_000_000_000_000  # an arbitrary anchor
    frames = [
        l1.FrameRecord(wallclock_ns=base_ns + i * 500_000_000,  # 0.5s apart
                       frame_index=i,
                       frame_type="step",
                       values={s.name: float(i) for s in l1.BASE_FEATURES})
        for i in range(8)
    ]
    now = frames[-1].wallclock_ns
    sliced_1s = l1t._window_slice(frames, now, 1_000_000_000)
    sliced_10s = l1t._window_slice(frames, now, 10_000_000_000)
    # The 1s window contains *strictly newer than* now-1s == frames at
    # offsets 3.5s, where frame_index=7 sits at offset 3.5s. So 1s slice
    # captures the last 2 frames (indices 6,7) which sit within (now-1s, now].
    assert all(f.wallclock_ns > now - 1_000_000_000 for f in sliced_1s)
    assert sliced_10s == frames  # all within 10s


def test_emitter_does_not_corrupt_trace(tmp_path, flag_on):
    from stepback.trace_reader import read_frames
    path = _write_a_few_frames(tmp_path, n=3)
    frames = list(read_frames(path))
    assert frames
    assert any(f["body"].get("type") == "step" for f in frames)
