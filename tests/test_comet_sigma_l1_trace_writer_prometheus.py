"""Tests for ``stepback.comet_sigma.l1_trace_writer_prometheus`` (Step 9).

These tests cover the per-base-feature Prometheus gauge exporter that
wires :mod:`stepback.comet_sigma.l1_trace_writer` (Step 1) to a set of
``prometheus_client.Gauge`` (or in-process fallback) instances.
"""
from __future__ import annotations

import os
import pytest

from stepback import comet_sigma as cs
from stepback.comet_sigma import l1_trace_writer as l1
from stepback.comet_sigma import l1_trace_writer_prometheus as pe


pytestmark = pytest.mark.skipif(
    not cs.comet_sigma_available(),
    reason="comet_sigma package not importable on this host",
)


@pytest.fixture(autouse=True)
def _clean_state():
    l1.reset()
    pe.reset()
    pe.install_hook()
    yield
    l1.reset()
    pe.reset()


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(l1.FLAG_NAME, "1")
    return True


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.setenv(l1.FLAG_NAME, "0")
    return False


# ---------------------------------------------------------------------------
# Static surface
# ---------------------------------------------------------------------------

def test_one_gauge_per_base_feature():
    assert set(pe.gauges().keys()) == {spec.name for spec in l1.BASE_FEATURES}


def _gauge_attr(gauge, *candidates):
    for name in candidates:
        if hasattr(gauge, name):
            return getattr(gauge, name)
    raise AttributeError(candidates)


def test_gauge_metric_name_is_prefixed_and_dot_free():
    for spec in l1.BASE_FEATURES:
        gauge = pe.gauge_for(spec.name)
        name = _gauge_attr(gauge, "name", "_name")
        assert name == pe._gauge_metric_name(spec.name)
        assert name.startswith(pe.GAUGE_NAME_PREFIX + "_")
        assert "." not in name


def test_gauge_carries_writer_id_label():
    for spec in l1.BASE_FEATURES:
        gauge = pe.gauge_for(spec.name)
        labelnames = _gauge_attr(gauge, "labelnames", "_labelnames")
        assert tuple(labelnames) == (pe.GAUGE_LABEL,) == ("writer_id",)


def test_install_hook_is_idempotent():
    pe.uninstall_hook()
    assert pe.install_hook() is True
    assert pe.install_hook() is False
    assert l1.OBSERVE_HOOKS.count(pe._hook) == 1


def test_uninstall_then_reinstall_round_trips():
    assert pe.uninstall_hook() is True
    assert pe.uninstall_hook() is False
    assert pe.install_hook() is True


# ---------------------------------------------------------------------------
# Flag gating
# ---------------------------------------------------------------------------

def test_default_flag_off_means_inactive():
    os.environ.pop(l1.FLAG_NAME, None)
    assert pe.is_active() is False


def test_flag_off_hook_is_a_noop(flag_off):
    l1.observe_frame("w0", {"type": "step", "k": 1}, b"{}")
    # Step 1 itself is inactive when the flag is off, so the hook is
    # never even invoked. The exporter's gauge values must still be
    # untouched.
    text = pe.render_text()
    for spec in l1.BASE_FEATURES:
        assert pe._gauge_metric_name(spec.name) in text
        assert 'writer_id="w0"' not in text


def test_flag_on_emits_one_value_per_feature(flag_on):
    record = l1.observe_frame("alpha", {"type": "step", "x": 1}, b"abcde")
    assert record is not None
    text = pe.render_text()
    for spec in l1.BASE_FEATURES:
        line_prefix = f'{pe._gauge_metric_name(spec.name)}{{writer_id="alpha"}}'
        assert any(line.startswith(line_prefix) for line in text.splitlines()), (
            f"missing exposition line for {spec.name}: {text!r}"
        )


# ---------------------------------------------------------------------------
# Per-frame semantics
# ---------------------------------------------------------------------------

def test_gauge_value_matches_emitter_value(flag_on):
    body = {"type": "step", "x": 1, "y": 2}
    body_bytes = b'{"type":"step","x":1,"y":2}'
    record = l1.observe_frame("beta", body, body_bytes)
    assert record is not None
    for spec in l1.BASE_FEATURES:
        gauge = pe.gauge_for(spec.name)
        if hasattr(gauge, "_get"):
            observed = gauge._get("beta")
        else:
            observed = gauge.labels(writer_id="beta")._value.get()
        assert observed == record.values[spec.name]


def test_gauge_tracks_latest_value_only(flag_on):
    body_a = {"type": "step", "x": 1}
    body_b = {"type": "blob", "x": 1, "y": 2, "z": 3}
    l1.observe_frame("g", body_a, b"a")
    l1.observe_frame("g", body_b, b"bbbb")
    body_key_gauge = pe.gauge_for("trace_writer_sb_v1.body_key_count")
    if hasattr(body_key_gauge, "_get"):
        latest = body_key_gauge._get("g")
    else:
        latest = body_key_gauge.labels(writer_id="g")._value.get()
    assert latest == float(len(body_b))  # 3 keys


def test_writer_id_is_a_distinct_label(flag_on):
    l1.observe_frame("w_one", {"type": "step"}, b"x")
    l1.observe_frame("w_two", {"type": "blob"}, b"yy")
    text = pe.render_text()
    assert 'writer_id="w_one"' in text
    assert 'writer_id="w_two"' in text


# ---------------------------------------------------------------------------
# Receipts / provenance
# ---------------------------------------------------------------------------

def test_artifact_is_an_auditable_artifact(flag_on):
    art = pe.artifact()
    assert art is not None
    assert art.kind == "feature"
    assert art.receipt_schema["id"] == pe.RECEIPT_SCHEMA_ID
    assert "writer_id" in art.receipt_schema["fields"]
    assert "frame_index" in art.receipt_schema["fields"]
    # First entry of the provenance chain is the creation entry.
    assert art.provenance.entries[0].layer == "L1"
    assert art.provenance.entries[0].module == pe.MODULE_LABEL


def test_one_receipt_per_feature_per_frame(flag_on):
    n = 5
    for i in range(n):
        l1.observe_frame("rcpt", {"type": "step", "i": i}, b"x")
    assert len(pe.receipts()) == n * len(l1.BASE_FEATURES)


def test_receipts_replay_against_their_artifact(flag_on):
    pytest.importorskip("comet_sigma.audit")
    from comet_sigma.audit import replay_receipt

    l1.observe_frame("rep", {"type": "step", "x": 1}, b"x")
    art = pe.artifact()
    rcpts = pe.receipts()
    assert rcpts, "expected at least one gauge receipt"
    for r in rcpts:
        assert replay_receipt(art, r) is True


def test_receipt_payload_pins_writer_and_frame(flag_on):
    l1.observe_frame("p", {"type": "step", "x": 1}, b"abc")
    l1.observe_frame("p", {"type": "blob", "x": 1, "y": 2}, b"abcdef")
    by_frame = {}
    for r in pe.receipts():
        key = (r.payload["writer_id"], r.payload["frame_index"], r.payload["feature"])
        by_frame[key] = r.payload["value"]
    assert ("p", 0, "trace_writer_sb_v1.body_key_count") in by_frame
    assert ("p", 1, "trace_writer_sb_v1.body_key_count") in by_frame
    assert by_frame[("p", 0, "trace_writer_sb_v1.body_key_count")] == 2.0
    assert by_frame[("p", 1, "trace_writer_sb_v1.body_key_count")] == 3.0


# ---------------------------------------------------------------------------
# Text exposition
# ---------------------------------------------------------------------------

def test_render_text_contains_help_and_type_for_every_gauge(flag_on):
    l1.observe_frame("xpos", {"type": "step"}, b"x")
    text = pe.render_text()
    for spec in l1.BASE_FEATURES:
        metric = pe._gauge_metric_name(spec.name)
        assert f"# HELP {metric}" in text
        assert f"# TYPE {metric} gauge" in text


def test_render_text_is_valid_prometheus_format(flag_on):
    pytest.importorskip("prometheus_client")
    from prometheus_client.parser import text_string_to_metric_families

    l1.observe_frame("rt", {"type": "step", "k": 1}, b"abcd")
    families = list(text_string_to_metric_families(pe.render_text()))
    names = {f.name for f in families}
    expected = {pe._gauge_metric_name(spec.name) for spec in l1.BASE_FEATURES}
    assert expected.issubset(names)


# ---------------------------------------------------------------------------
# Reset semantics
# ---------------------------------------------------------------------------

def test_reset_clears_gauge_values_and_receipts(flag_on):
    l1.observe_frame("z", {"type": "step"}, b"abc")
    assert pe.receipts(), "expected receipts before reset"
    pe.reset()
    assert pe.receipts() == []
    text = pe.render_text()
    assert 'writer_id="z"' not in text


def test_hook_is_resilient_to_bad_values(flag_on):
    # Inject a frame whose values dict references an unknown feature
    # name; the gauge update for that key should be skipped silently
    # without aborting the rest.
    rec = l1.FrameRecord(
        wallclock_ns=1,
        frame_index=0,
        frame_type="step",
        values={"trace_writer_sb_v1.frame_bytes": 7.0, "unknown.bogus": 99.0},
    )
    pe._hook("y", rec)
    gauge = pe.gauge_for("trace_writer_sb_v1.frame_bytes")
    if hasattr(gauge, "_get"):
        v = gauge._get("y")
    else:
        v = gauge.labels(writer_id="y")._value.get()
    assert v == 7.0
