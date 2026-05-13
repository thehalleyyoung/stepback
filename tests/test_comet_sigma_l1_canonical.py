"""Tests for ``stepback.comet_sigma.l1_canonical`` (Step 11).

Cover the wiring of :mod:`stepback.canonical` as a Comet-Σ L1
``temporal_basis`` feature emitter: flag-gating, base-feature shape,
provenance, receipt round-trip, ring-buffer bounds, and that the
``install``/``uninstall`` wrappers around ``canonical_json`` are
transparent (same return bytes) and non-destructive.
"""
from __future__ import annotations

import hashlib
import os

import pytest

from stepback import comet_sigma as cs
from stepback import canonical as _canon
from stepback.comet_sigma import l1_canonical as l1


pytestmark = pytest.mark.skipif(
    not cs.comet_sigma_available(),
    reason="comet_sigma package not importable on this host",
)


@pytest.fixture(autouse=True)
def _clean_emitter_state():
    # Make sure no test leaks installation state or registry entries.
    l1.uninstall()
    l1.reset()
    yield
    l1.uninstall()
    l1.reset()


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(l1.FLAG_NAME, "1")
    return True


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.setenv(l1.FLAG_NAME, "0")
    return False


# ---------------------------------------------------------------------------
# Flag-gating
# ---------------------------------------------------------------------------

def test_default_flag_off_emitter_inactive():
    os.environ.pop(l1.FLAG_NAME, None)
    assert cs.is_enabled(l1.FLAG_NAME) is False
    assert l1.is_active() is False


def test_observe_is_noop_when_flag_off(flag_off):
    out = _canon.canonical_json({"a": 1})
    assert l1.observe_canonicalisation({"a": 1}, out) is None
    assert l1.feature_history() == []
    assert l1.receipts_for() == []
    assert l1.latest_feature_vector() is None
    assert l1.get_state(l1.DEFAULT_NAMESPACE) is None


# ---------------------------------------------------------------------------
# Base-feature shape and values
# ---------------------------------------------------------------------------

def test_observe_emits_features_when_flag_on(flag_on):
    obj = {"k1": [1, 2, {"deep": {"x": "y"}}], "k2": True}
    out = _canon.canonical_json(obj)
    ev = l1.observe_canonicalisation(obj, out, namespace="ns")
    assert ev is not None
    expected_names = {s.name for s in l1.BASE_FEATURES}
    assert set(ev.values.keys()) == expected_names

    assert ev.values["canonical_v1.out_bytes"] == float(len(out))
    assert ev.values["canonical_v1.event_index"] == 0.0
    assert ev.values["canonical_v1.prev_dt_ns"] == 0.0
    assert ev.values["canonical_v1.top_key_count"] == 2.0
    # max_depth: dict(0) -> list(1) -> dict(2) -> dict(3) -> str(4)
    assert ev.values["canonical_v1.max_depth"] == 4.0
    assert ev.values["canonical_v1.has_bytes_flag"] == 0.0
    assert ev.values["canonical_v1.type_tag"] == 1.0


def test_bytes_input_sets_has_bytes_flag(flag_on):
    obj = {"raw": b"\x00\x01\x02"}
    out = _canon.canonical_json(obj)
    ev = l1.observe_canonicalisation(obj, out)
    assert ev is not None
    assert ev.values["canonical_v1.has_bytes_flag"] == 1.0


def test_event_index_monotonic_and_prev_dt(flag_on):
    for i in range(5):
        out = _canon.canonical_json({"i": i})
        l1.observe_canonicalisation({"i": i}, out, namespace="seq")
    hist = l1.feature_history("seq")
    assert [int(e.values["canonical_v1.event_index"]) for e in hist] == [0, 1, 2, 3, 4]
    # First prev_dt_ns is 0; subsequent must be ≥ 0.
    assert hist[0].values["canonical_v1.prev_dt_ns"] == 0.0
    assert all(e.values["canonical_v1.prev_dt_ns"] >= 0.0 for e in hist[1:])


# ---------------------------------------------------------------------------
# Artifacts, provenance, receipts
# ---------------------------------------------------------------------------

def test_artifacts_have_provenance_and_schema(flag_on):
    out = _canon.canonical_json({"x": 1})
    l1.observe_canonicalisation({"x": 1}, out, namespace="ns")
    arts = l1.artifacts_for("ns")
    assert len(arts) == len(l1.BASE_FEATURES)
    for art, spec in zip(arts, l1.BASE_FEATURES):
        assert art.kind == "feature"
        assert art.name == spec.name
        assert art.receipt_schema["id"] == l1.RECEIPT_SCHEMA_ID
        head = art.provenance.head()
        assert head is not None
        assert head.layer == "L1"
        assert head.module == l1.MODULE_LABEL
        assert head.src_sha256 == hashlib.sha256(spec.src.encode()).hexdigest()


def test_receipts_round_trip(flag_on):
    out = _canon.canonical_json({"x": 1})
    l1.observe_canonicalisation({"x": 1}, out, namespace="rt")
    receipts = l1.receipts_for("rt")
    assert len(receipts) == len(l1.BASE_FEATURES)
    arts = {a.name: a for a in l1.artifacts_for("rt")}
    try:
        from comet_sigma.audit import replay_receipt
    except ModuleNotFoundError:  # pragma: no cover
        from kitchensink.comet_sigma.audit import replay_receipt
    for r in receipts:
        assert r.schema_id == l1.RECEIPT_SCHEMA_ID
        assert set(r.payload) == {"value", "event_index", "namespace"}
        assert r.payload["namespace"] == "rt"
        assert isinstance(r.payload["value"], float)
        assert isinstance(r.payload["event_index"], int)
        assert replay_receipt(arts[r.artifact_name], r) is True


# ---------------------------------------------------------------------------
# Ring-buffer + namespace bookkeeping
# ---------------------------------------------------------------------------

def test_ring_buffer_bounded(monkeypatch, flag_on):
    monkeypatch.setattr(l1, "MAX_RECENT_EVENTS", 4)
    l1.reset("cap")
    for i in range(20):
        out = _canon.canonical_json({"i": i})
        l1.observe_canonicalisation({"i": i}, out, namespace="cap")
    hist = l1.feature_history("cap")
    assert len(hist) <= 4


def test_known_namespaces(flag_on):
    out = _canon.canonical_json({"a": 1})
    l1.observe_canonicalisation({"a": 1}, out, namespace="A")
    l1.observe_canonicalisation({"a": 2}, out, namespace="B")
    assert {"A", "B"} <= set(l1.known_namespaces())


def test_observe_tolerates_garbage(flag_on):
    # Non-bytes payload must not raise; ev returned with zeroed
    # extractor values where applicable.
    ev = l1.observe_canonicalisation({"a": 1}, "not bytes")  # type: ignore[arg-type]
    assert ev is None or ev.values["canonical_v1.out_bytes"] == 0.0


# ---------------------------------------------------------------------------
# install()/uninstall() wrapper
# ---------------------------------------------------------------------------

def test_install_is_transparent_to_canonical_json(flag_on):
    obj = {"hello": "world", "n": 3}
    expected = _canon.canonical_json(obj)
    assert l1.install(namespace="wrapped") is True
    try:
        got = _canon.canonical_json(obj)
        assert got == expected
        # Wrapper should have produced one event.
        hist = l1.feature_history("wrapped")
        assert len(hist) == 1
        assert hist[0].values["canonical_v1.out_bytes"] == float(len(expected))
        # duration_ns is >= 0; we can't assert >0 reliably on fast machines.
        assert hist[0].values["canonical_v1.duration_ns"] >= 0.0
    finally:
        assert l1.uninstall() is True
    # Second uninstall is a no-op.
    assert l1.uninstall() is False


def test_install_idempotent(flag_on):
    assert l1.install() is True
    try:
        assert l1.install() is False
    finally:
        l1.uninstall()


def test_install_with_flag_off_is_zero_overhead(flag_off):
    obj = {"x": 1}
    expected = _canon.canonical_json(obj)
    l1.install(namespace="off")
    try:
        got = _canon.canonical_json(obj)
        assert got == expected
        # No state when flag is off.
        assert l1.get_state("off") is None
    finally:
        l1.uninstall()


# ---------------------------------------------------------------------------
# Flag registry
# ---------------------------------------------------------------------------

def test_flags_module_lists_l1_temporal():
    flags = cs.all_flags()
    assert l1.FLAG_NAME in flags
