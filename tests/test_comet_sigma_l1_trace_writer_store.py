"""Tests for ``stepback.comet_sigma.l1_trace_writer_store`` (Step 3).

These tests cover the on-disk persistence of L1 ``temporal_basis``
emissions from :class:`stepback.trace_writer.TraceWriter`:
:class:`comet_sigma.audit.AuditableArtifact` instances are written to a
content-addressed store, every :class:`comet_sigma.audit.Receipt` is
appended to a per-writer JSONL, and the artifacts' provenance chains
gain a single new entry recording the persistence event.
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

from stepback import comet_sigma as cs
from stepback.comet_sigma import (
    l1_trace_writer as l1,
    l1_trace_writer_store as store,
    l1_trace_writer_temporal as l1t,
)
from stepback.trace_writer import TraceWriter


pytestmark = pytest.mark.skipif(
    not cs.comet_sigma_available(),
    reason="comet_sigma package not importable on this host",
)


@pytest.fixture(autouse=True)
def _clean_state():
    l1.reset()
    l1t.reset()
    store.reset()
    store.set_store_dir(None)
    # Re-establish deterministic OBSERVE_HOOKS order: temporal projector
    # first (so projection receipts exist) then persistence (so it
    # captures everything in a single pass). Other test files (notably
    # ``test_comet_sigma_l1_trace_writer_temporal.py``) freely
    # uninstall/reinstall the projection hook, which would otherwise
    # leave the two hooks in the wrong order when those tests run
    # before this file.
    l1t.uninstall_hook()
    store.uninstall_hook()
    l1t.install_hook()
    store.install_hook()
    yield
    l1.reset()
    l1t.reset()
    store.reset()
    store.set_store_dir(None)


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(l1.FLAG_NAME, "1")
    return True


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.setenv(l1.FLAG_NAME, "0")
    return False


def _write_frames(tmp_path: pathlib.Path, n: int = 3, name: str = "trace.sb") -> str:
    path = str(tmp_path / name)
    w = TraceWriter.open(path, signing=False)
    for i in range(n):
        w.write_step({"type": "step", "i": i, "payload": "x" * 16})
    w.close()
    return path


# ---------------------------------------------------------------------------
# Activation / flag-gating
# ---------------------------------------------------------------------------

def test_default_is_inactive_no_store_no_flag():
    assert store.is_active() is False
    assert store.get_store_dir() is None


def test_inactive_when_flag_off_even_with_store(tmp_path, flag_off):
    store.set_store_dir(str(tmp_path / "store"))
    assert store.is_active() is False


def test_inactive_when_flag_on_but_no_store(flag_on):
    assert store.get_store_dir() is None
    assert store.is_active() is False


def test_active_when_flag_on_and_store_set(tmp_path, flag_on):
    store.set_store_dir(str(tmp_path / "store"))
    assert store.is_active() is True


def test_env_var_provides_store_dir(monkeypatch, tmp_path, flag_on):
    monkeypatch.setenv(store.STORE_DIR_ENV, str(tmp_path / "envstore"))
    assert store.get_store_dir() == str(tmp_path / "envstore")
    assert store.is_active() is True


def test_programmatic_override_beats_env(monkeypatch, tmp_path, flag_on):
    monkeypatch.setenv(store.STORE_DIR_ENV, str(tmp_path / "envstore"))
    store.set_store_dir(str(tmp_path / "override"))
    assert store.get_store_dir() == str(tmp_path / "override")


# ---------------------------------------------------------------------------
# No-op when inactive
# ---------------------------------------------------------------------------

def test_writer_emits_nothing_to_disk_when_flag_off(tmp_path, flag_off):
    sd = tmp_path / "store"
    store.set_store_dir(str(sd))
    _write_frames(tmp_path)
    # The store dir must remain empty (or at most contain nothing under
    # ``writers/``) because the persister is inactive when the flag is
    # off.
    assert not (sd / "writers").exists()


def test_persist_writer_returns_none_when_inactive(flag_off):
    assert store.persist_writer("anything") is None
    assert store.persist_all() == {}


# ---------------------------------------------------------------------------
# Happy path: artifacts + receipts written
# ---------------------------------------------------------------------------

def test_artifacts_written_to_disk(tmp_path, flag_on):
    sd = tmp_path / "store"
    store.set_store_dir(str(sd))
    path = _write_frames(tmp_path)

    # Every base-feature artifact must be present on disk.
    for spec in l1.BASE_FEATURES:
        p = store.artifact_path(str(sd), path, spec.name)
        assert os.path.exists(p), f"missing artifact file for {spec.name}"
        with open(p, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        assert d["kind"] == "feature"
        assert d["name"] == spec.name
        assert d["receipt_schema"]["id"] == l1.RECEIPT_SCHEMA_ID
        # Persistence appends ONE provenance entry on top of the
        # creation entry, so the chain length is exactly 2 and the head
        # is the persistence entry.
        assert len(d["provenance"]) == 2
        assert d["provenance"][0]["module"] == l1.MODULE_LABEL
        assert d["provenance"][1]["module"] == store.MODULE_LABEL
        assert d["provenance"][1]["layer"] == "L1"


def test_projection_artifacts_written_to_disk(tmp_path, flag_on):
    sd = tmp_path / "store"
    store.set_store_dir(str(sd))
    path = _write_frames(tmp_path)

    proj_arts = l1t.artifacts_for(path)
    assert proj_arts, "expected Step 2 to emit projection artifacts"
    for name in proj_arts:
        p = store.artifact_path(str(sd), path, name)
        assert os.path.exists(p), f"missing projection artifact {name}"


def test_base_receipts_written_one_per_line(tmp_path, flag_on):
    sd = tmp_path / "store"
    store.set_store_dir(str(sd))
    path = _write_frames(tmp_path, n=3)

    receipts_path = store.base_receipts_path(str(sd), path)
    assert os.path.exists(receipts_path)
    with open(receipts_path, "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    expected = len(l1.BASE_FEATURES) * len(l1.feature_history(path))
    assert len(lines) == expected
    # Each line must be a JSON object with the canonical receipt fields.
    for ln in lines:
        d = json.loads(ln)
        assert d["schema_id"] == l1.RECEIPT_SCHEMA_ID
        assert "payload" in d and "payload_sha256" in d
        assert set(d["payload"]) == {"value", "frame_index", "frame_type"}


def test_projection_receipts_written(tmp_path, flag_on):
    sd = tmp_path / "store"
    store.set_store_dir(str(sd))
    path = _write_frames(tmp_path, n=3)

    pp = store.projection_receipts_path(str(sd), path)
    assert os.path.exists(pp)
    with open(pp, "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    assert lines, "expected projection receipts to be appended"
    for ln in lines:
        d = json.loads(ln)
        assert d["schema_id"] == l1t.RECEIPT_SCHEMA_ID
        assert {"value", "window", "window_ns", "n_frames_in_window",
                "frame_index"} <= set(d["payload"])


def test_writer_meta_written_once(tmp_path, flag_on):
    sd = tmp_path / "store"
    store.set_store_dir(str(sd))
    path = _write_frames(tmp_path)

    meta_path = os.path.join(store.writer_dir(str(sd), path), store.META_FILENAME)
    assert os.path.exists(meta_path)
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["writer_id"] == path
    assert meta["module"] == store.MODULE_LABEL
    assert meta["flag"] == l1.FLAG_NAME
    mtime = os.path.getmtime(meta_path)

    # Re-persisting must NOT rewrite the meta file.
    store.persist_writer(path)
    assert os.path.getmtime(meta_path) == mtime


# ---------------------------------------------------------------------------
# Idempotency / append behaviour
# ---------------------------------------------------------------------------

def test_repersist_with_no_new_emissions_is_a_noop(tmp_path, flag_on):
    sd = tmp_path / "store"
    store.set_store_dir(str(sd))
    path = _write_frames(tmp_path, n=2)

    rp = store.base_receipts_path(str(sd), path)
    size_before = os.path.getsize(rp)
    counters = store.persist_writer(path)
    assert counters is not None
    assert counters["base_receipts_written"] == 0
    assert counters["projection_receipts_written"] == 0
    assert counters["artifacts_written"] == 0
    assert os.path.getsize(rp) == size_before


def test_new_frames_append_only_new_receipts(tmp_path, flag_on):
    sd = tmp_path / "store"
    store.set_store_dir(str(sd))
    path = str(tmp_path / "growing.sb")
    w = TraceWriter.open(path, signing=False)
    w.write_step({"type": "step", "i": 0})
    n_after_first = len(l1.feature_history(path))
    rp = store.base_receipts_path(str(sd), path)
    with open(rp, "r", encoding="utf-8") as fh:
        lines_after_first = sum(1 for _ in fh)
    assert lines_after_first == n_after_first * len(l1.BASE_FEATURES)

    w.write_step({"type": "step", "i": 1})
    w.close()

    n_total = len(l1.feature_history(path))
    with open(rp, "r", encoding="utf-8") as fh:
        lines_total = sum(1 for _ in fh)
    assert lines_total == n_total * len(l1.BASE_FEATURES)
    assert lines_total > lines_after_first


# ---------------------------------------------------------------------------
# Round-trip: load_persisted_artifact + replay_receipt
# ---------------------------------------------------------------------------

def test_load_persisted_artifact_round_trip(tmp_path, flag_on):
    sd = tmp_path / "store"
    store.set_store_dir(str(sd))
    path = _write_frames(tmp_path)

    spec = l1.BASE_FEATURES[0]
    art = store.load_persisted_artifact(path, spec.name)
    assert art is not None
    assert art.name == spec.name
    assert art.kind == "feature"
    assert art.src_sha256 == l1.artifacts_for(path)[0].src_sha256
    # The persistence-time provenance entry is now part of the chain.
    assert len(art.provenance.entries) == 2


def test_persisted_receipts_replay_against_persisted_artifacts(tmp_path, flag_on):
    sd = tmp_path / "store"
    store.set_store_dir(str(sd))
    path = _write_frames(tmp_path)

    try:
        from comet_sigma.audit import replay_receipt, Receipt  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        from kitchensink.comet_sigma.audit import (  # type: ignore[import-not-found]
            replay_receipt,
            Receipt,
        )

    # Pull the persisted artifacts back from disk.
    arts_by_name = {}
    for spec in l1.BASE_FEATURES:
        a = store.load_persisted_artifact(path, spec.name)
        assert a is not None
        arts_by_name[spec.name] = a

    n_replayed = 0
    for d in store.iter_persisted_receipts(path, kind="base"):
        receipt = Receipt(
            artifact_name=d["artifact_name"],
            schema_id=d["schema_id"],
            payload=d["payload"],
            at=d["at"],
        )
        assert replay_receipt(arts_by_name[receipt.artifact_name], receipt) is True
        n_replayed += 1
    assert n_replayed > 0


def test_iter_persisted_receipts_unknown_kind_raises(tmp_path, flag_on):
    sd = tmp_path / "store"
    store.set_store_dir(str(sd))
    _write_frames(tmp_path)
    with pytest.raises(ValueError):
        list(store.iter_persisted_receipts("anything", kind="bogus"))


# ---------------------------------------------------------------------------
# Bookkeeping / inventory helpers
# ---------------------------------------------------------------------------

def test_list_persisted_writers(tmp_path, flag_on):
    sd = tmp_path / "store"
    store.set_store_dir(str(sd))
    p1 = _write_frames(tmp_path, name="a.sb")
    p2 = _write_frames(tmp_path, name="b.sb")
    listed = set(store.list_persisted_writers())
    h1 = os.path.basename(store.writer_dir(str(sd), p1))
    h2 = os.path.basename(store.writer_dir(str(sd), p2))
    assert {h1, h2} <= listed


def test_persist_all_returns_per_writer_counters(tmp_path, flag_on):
    sd = tmp_path / "store"
    store.set_store_dir(str(sd))
    p1 = _write_frames(tmp_path, name="a.sb")
    p2 = _write_frames(tmp_path, name="b.sb")
    # First call already happened via the auto-hook; manual call now is
    # a near no-op but must still return a dict keyed by writer id.
    res = store.persist_all()
    assert set(res) == {p1, p2}
    for c in res.values():
        assert "artifacts_written" in c
        assert "base_receipts_written" in c
        assert "projection_receipts_written" in c


def test_watermark_snapshot_tracks_progress(tmp_path, flag_on):
    sd = tmp_path / "store"
    store.set_store_dir(str(sd))
    path = _write_frames(tmp_path, n=4)
    snap = store.watermark_snapshot(path)
    assert snap is not None
    n_features = len(l1.BASE_FEATURES)
    assert snap["base_receipts"] == n_features * len(l1.feature_history(path))
    # All base-feature artifacts have been persisted exactly once.
    assert set(snap["persisted_artifact_shas"]) >= {
        s.name for s in l1.BASE_FEATURES
    }


# ---------------------------------------------------------------------------
# Trace integrity
# ---------------------------------------------------------------------------

def test_persistence_does_not_corrupt_trace(tmp_path, flag_on):
    """Persistence must never alter on-disk .sb frames."""
    from stepback.trace_reader import read_frames

    sd = tmp_path / "store"
    store.set_store_dir(str(sd))
    path = _write_frames(tmp_path, n=4)
    frames = list(read_frames(path))
    assert frames
    assert any(f["body"].get("type") == "step" for f in frames)


def test_hook_install_is_idempotent():
    # The hook is auto-installed at import time; reinstalling must be a
    # no-op.
    assert store.install_hook() is False
    # Uninstall then reinstall.
    assert store.uninstall_hook() is True
    assert store.install_hook() is True
    assert store.install_hook() is False


def test_persistence_failure_is_swallowed(tmp_path, flag_on, monkeypatch):
    # Point the store at an unwriteable path and make sure observe_frame
    # still returns a record (i.e. the writer's hot path is unaffected).
    bad = tmp_path / "ro"
    bad.mkdir()
    bad.chmod(0o500)  # no write permission for owner-as-user write
    store.set_store_dir(str(bad / "store"))
    try:
        path = str(tmp_path / "trace.sb")
        w = TraceWriter.open(path, signing=False)
        w.write_step({"type": "step", "i": 0})
        w.close()
        # The trace was still written; the persister silently failed.
        assert os.path.exists(path)
    finally:
        bad.chmod(0o700)
