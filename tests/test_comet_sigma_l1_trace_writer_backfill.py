"""Tests for ``stepback.comet_sigma.l1_trace_writer_backfill`` (Step 4).

Backfill replays historical ``.sb`` v1 traces through the L1 emitter
(:mod:`stepback.comet_sigma.l1_trace_writer`), the temporal projector
(:mod:`stepback.comet_sigma.l1_trace_writer_temporal`) and the
persistence store (:mod:`stepback.comet_sigma.l1_trace_writer_store`),
producing the same content-addressed artifact tree a live writer
would.
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

from stepback import comet_sigma as cs
from stepback.comet_sigma import (
    l1_trace_writer as l1,
    l1_trace_writer_backfill as backfill,
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


def _write_trace(tmp_path: pathlib.Path, n: int = 5, name: str = "hist.sb") -> str:
    """Write a .sb trace to disk **with the L1 flag off** so backfill is the
    sole source of receipts/artifacts in the tests below.
    """
    # Make absolutely sure live emission is OFF for the trace creation
    # phase — the per-test fixtures only flip the env var inside their
    # body, but we want a clean separation.
    prior = os.environ.pop(l1.FLAG_NAME, None)
    try:
        path = str(tmp_path / name)
        w = TraceWriter.open(path, signing=False)
        for i in range(n):
            w.write_step({"type": "step", "i": i, "payload": "p" * 8})
        w.close()
        return path
    finally:
        if prior is not None:
            os.environ[l1.FLAG_NAME] = prior


# ---------------------------------------------------------------------------
# Activation / flag-gating
# ---------------------------------------------------------------------------

def test_inactive_when_flag_off_returns_none(tmp_path, flag_off):
    p = _write_trace(tmp_path)
    assert backfill.is_active() is False
    assert backfill.backfill_trace(p) is None
    assert backfill.backfill_paths([p]) == []
    assert backfill.backfill_directory(str(tmp_path)) == []


def test_active_when_flag_on(flag_on):
    assert backfill.is_active() is True


def test_writer_id_for_uses_abspath_and_prefix(tmp_path):
    p = str(tmp_path / "x.sb")
    wid = backfill.writer_id_for(p)
    assert wid.startswith(backfill.WRITER_ID_PREFIX)
    assert wid.endswith(os.path.abspath(p))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_backfill_replays_every_frame(tmp_path, flag_on):
    sd = tmp_path / "store"
    p = _write_trace(tmp_path, n=4)

    res = backfill.backfill_trace(p, store_dir=str(sd))
    assert res is not None
    assert res.error is None
    # The trace contains: 1 header + 4 step + 1 trailer/tail-ish frames.
    # Be tolerant about the exact count (writer adds header + footer
    # frames) but we MUST observe at least the step frames.
    assert res.frames_observed >= 4
    assert res.fingerprint  # non-empty hex string
    # Per-feature receipt count == frame_count for each base feature.
    assert res.base_receipts_emitted == res.frames_observed * len(l1.BASE_FEATURES)
    # Projection receipts also accumulated.
    assert res.projection_receipts_emitted > 0


def test_backfill_persists_artifacts_to_store(tmp_path, flag_on):
    sd = tmp_path / "store"
    p = _write_trace(tmp_path, n=3)
    # Read the wrapper count directly so we know what
    # ``frames_backfilled`` will be (writer adds header / trailer
    # frames around the user-supplied step frames).
    from stepback.trace_reader import read_frames as _rf
    expected_frames = len(_rf(p))

    res = backfill.backfill_trace(p, store_dir=str(sd))
    assert res is not None and res.persisted is not None

    wid = res.writer_id
    # Every base-feature artifact has been written to the store.
    for spec in l1.BASE_FEATURES:
        ap = store.artifact_path(str(sd), wid, spec.name)
        assert os.path.exists(ap), f"missing persisted artifact for {spec.name}"
        with open(ap, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        # Provenance chain: creation (L1) → backfill (L1) → persistence (L1).
        modules = [e["module"] for e in d["provenance"]]
        assert l1.MODULE_LABEL in modules
        assert backfill.MODULE_LABEL in modules
        assert store.MODULE_LABEL in modules
        # Backfill entry must record the source path + fingerprint via
        # ``payload_sha256`` — payloads are not retained verbatim, so we
        # recompute the hash and compare.
        bf_entry = next(e for e in d["provenance"] if e["module"] == backfill.MODULE_LABEL)
        import hashlib as _h
        import json as _j
        expected_payload = {
            "source_path": os.path.abspath(p),
            "frames_backfilled": expected_frames,
            "fingerprint_sha256_64k": res.fingerprint,
        }
        body = _j.dumps(expected_payload, sort_keys=True, default=str)
        expected_sha = _h.sha256(body.encode()).hexdigest()
        assert bf_entry["payload_sha256"] == expected_sha


def test_backfill_writes_receipts_jsonl(tmp_path, flag_on):
    sd = tmp_path / "store"
    p = _write_trace(tmp_path, n=3)
    res = backfill.backfill_trace(p, store_dir=str(sd))
    assert res is not None
    rp = store.base_receipts_path(str(sd), res.writer_id)
    assert os.path.exists(rp)
    with open(rp, "r", encoding="utf-8") as fh:
        lines = [l for l in fh if l.strip()]
    assert len(lines) == res.base_receipts_emitted
    # Each line is a valid JSON receipt.
    for line in lines:
        obj = json.loads(line)
        assert "schema_id" in obj or "schema" in obj or "payload" in obj


# ---------------------------------------------------------------------------
# Idempotence / namespace separation
# ---------------------------------------------------------------------------

def test_backfill_writer_id_does_not_collide_with_live(tmp_path, flag_on):
    sd = tmp_path / "store"
    store.set_store_dir(str(sd))
    p = _write_trace(tmp_path, n=2)

    # Live writer appends to a different path under the live writer-id.
    live_path = str(tmp_path / "live.sb")
    w = TraceWriter.open(live_path, signing=False)
    w.write_step({"type": "step", "i": 0})
    w.close()

    res = backfill.backfill_trace(p)
    assert res is not None
    assert res.writer_id != live_path
    assert res.writer_id.startswith(backfill.WRITER_ID_PREFIX)
    # Two separate writer dirs on disk.
    persisted = store.list_persisted_writers(str(sd))
    assert len(persisted) >= 2


def test_reset_drops_only_backfill_writers(tmp_path, flag_on):
    p = _write_trace(tmp_path, n=2)
    backfill.backfill_trace(p, store_dir=str(tmp_path / "s"))
    assert backfill.known_backfill_writers()
    backfill.reset_backfill_state()
    assert backfill.known_backfill_writers() == ()


def test_backfill_with_reset_starts_fresh(tmp_path, flag_on):
    sd = tmp_path / "store"
    p = _write_trace(tmp_path, n=3)
    r1 = backfill.backfill_trace(p, store_dir=str(sd))
    r2 = backfill.backfill_trace(p, store_dir=str(sd), reset=True)
    assert r1 is not None and r2 is not None
    # After reset, observed frame count matches a single replay.
    assert r2.frames_observed == r1.frames_observed


# ---------------------------------------------------------------------------
# Directory walk
# ---------------------------------------------------------------------------

def test_backfill_directory_picks_up_every_sb(tmp_path, flag_on):
    sd = tmp_path / "store"
    paths = [
        _write_trace(tmp_path, n=2, name="a.sb"),
        _write_trace(tmp_path, n=2, name="b.sb"),
    ]
    sub = tmp_path / "sub"
    sub.mkdir()
    paths.append(_write_trace(sub, n=2, name="c.sb"))

    out = backfill.backfill_directory(str(tmp_path), store_dir=str(sd))
    assert {r.path for r in out} == {os.path.abspath(p) for p in paths}
    for r in out:
        assert r.error is None
        assert r.frames_observed >= 2


# ---------------------------------------------------------------------------
# Error tolerance
# ---------------------------------------------------------------------------

def test_backfill_missing_file_records_error(tmp_path, flag_on):
    res = backfill.backfill_trace(str(tmp_path / "nope.sb"))
    assert res is not None
    assert res.error is not None
    assert res.frames_observed == 0


def test_backfill_truncated_file_does_not_raise(tmp_path, flag_on):
    p = _write_trace(tmp_path, n=3)
    # Corrupt by truncating in the middle of a frame.
    with open(p, "rb") as fh:
        data = fh.read()
    with open(p, "wb") as fh:
        fh.write(data[: len(data) // 2])
    res = backfill.backfill_trace(p)
    assert res is not None
    # Either a clean error, or partial frames — neither should raise.
    assert res.error is not None or res.frames_observed >= 0


# ---------------------------------------------------------------------------
# BackfillResult.to_dict shape
# ---------------------------------------------------------------------------

def test_backfill_result_to_dict_has_expected_keys(tmp_path, flag_on):
    p = _write_trace(tmp_path, n=2)
    res = backfill.backfill_trace(p, store_dir=str(tmp_path / "s"))
    assert res is not None
    d = res.to_dict()
    for k in (
        "path",
        "writer_id",
        "frames_observed",
        "frames_skipped",
        "base_receipts_emitted",
        "projection_receipts_emitted",
        "persisted",
        "fingerprint",
        "error",
    ):
        assert k in d
