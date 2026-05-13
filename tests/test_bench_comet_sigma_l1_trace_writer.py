"""Tests for the L1 emission-latency micro-benchmark (Step 6).

Covers :mod:`stepback.bench.comet_sigma_l1_trace_writer` — the
benchmark that measures per-frame latency of the Comet-Σ L1
``temporal_basis`` emitter wired in
:mod:`stepback.comet_sigma.l1_trace_writer` for ``stepback.trace_writer``
+ ``.sb`` format v1.
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

from stepback import comet_sigma as cs
from stepback.bench import comet_sigma_l1_trace_writer as bench
from stepback.comet_sigma import l1_trace_writer as l1


pytestmark = pytest.mark.skipif(
    not cs.comet_sigma_available(),
    reason="comet_sigma package not importable on this host",
)


@pytest.fixture(autouse=True)
def _quiet_global_state(monkeypatch):
    # Snapshot + restore the L1 flag and observer-hooks list so this
    # bench cannot bleed flag state into neighbouring tests.
    monkeypatch.delenv(l1.FLAG_NAME, raising=False)
    saved = list(l1.OBSERVE_HOOKS)
    l1.OBSERVE_HOOKS.clear()
    l1.reset()
    yield
    l1.OBSERVE_HOOKS.clear()
    l1.OBSERVE_HOOKS.extend(saved)
    l1.reset()


def test_run_returns_three_regimes_with_correct_schema():
    r = bench.run(n_frames=200, warmup_frames=20)
    assert r.schema_version == bench.SCHEMA_VERSION
    assert r.n_frames == 200
    assert r.warmup_frames == 20
    assert r.comet_sigma_available is True
    for regime in (r.off, r.on, r.on_temporal):
        assert regime.n_samples == 200 - 20
        # Percentiles obey monotonicity.
        assert regime.min_us <= regime.p50_us <= regime.p95_us <= regime.p99_us <= regime.max_us
        # Mean is non-negative and finite.
        assert regime.mean_us >= 0.0


def test_flag_off_is_strictly_cheaper_than_flag_on():
    """The whole point of the flag is to make the off-path nearly free."""
    r = bench.run(n_frames=300, warmup_frames=30)
    # Use mean to dampen scheduler noise; on_p50 is typically >>off_p50
    # by orders of magnitude when the flag is on.
    assert r.on.mean_us > r.off.mean_us, (
        f"flag-on regime ({r.on.mean_us:.2f}us) should be more "
        f"expensive than flag-off ({r.off.mean_us:.2f}us)"
    )
    assert r.on_temporal.mean_us >= r.on.mean_us * 0.5, (
        "temporal-projector hook should not be ~free; it must add "
        "measurable per-frame work"
    )


def test_run_does_not_leak_flag_or_hooks(monkeypatch):
    # Explicitly set the flag OFF and install a sentinel hook before
    # running — the bench must restore exactly this state.
    monkeypatch.setenv(l1.FLAG_NAME, "0")

    sentinel_called = []

    def sentinel(writer_id, record):
        sentinel_called.append(writer_id)

    l1.OBSERVE_HOOKS.append(sentinel)
    bench.run(n_frames=100, warmup_frames=10)
    assert sentinel in l1.OBSERVE_HOOKS
    assert os.environ.get(l1.FLAG_NAME) == "0"
    assert l1.is_active() is False


def test_run_and_save_writes_valid_json(tmp_path: pathlib.Path):
    out = tmp_path / "out.json"
    r = bench.run_and_save(str(out), n_frames=120, warmup_frames=20)
    blob = json.loads(out.read_text())
    assert blob["schema_version"] == bench.SCHEMA_VERSION
    assert blob["n_frames"] == 120
    assert blob["warmup_frames"] == 20
    assert set(blob["regimes"]) == {"off", "on", "on_temporal"}
    # Round-trips with the in-memory result.
    assert blob["regimes"]["off"]["n_samples"] == r.off.n_samples


def test_summary_line_is_a_single_oneliner():
    r = bench.run(n_frames=80, warmup_frames=10)
    line = r.summary_line()
    assert "\n" not in line
    assert "l1-trace-writer-latency" in line
    assert "off_p50=" in line
    assert "on_p50=" in line
    assert "on_p99=" in line


@pytest.mark.parametrize("bad", [
    {"n_frames": 0},
    {"n_frames": 1},
    {"n_frames": 50, "warmup_frames": -1},
    {"n_frames": 50, "warmup_frames": 50},
    {"n_frames": 50, "warmup_frames": 100},
])
def test_run_rejects_invalid_arguments(bad):
    with pytest.raises(ValueError):
        bench.run(**bad)


def test_persisted_reference_result_matches_schema():
    """The reference JSON committed under bench-results/ stays parseable."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    p = repo_root / "bench-results" / "comet-sigma-l1-trace-writer-n5000.json"
    if not p.exists():
        pytest.skip("reference bench result not present in this checkout")
    blob = json.loads(p.read_text())
    assert blob["schema_version"] == bench.SCHEMA_VERSION
    for r in ("off", "on", "on_temporal"):
        for k in ("p50_us", "p95_us", "p99_us", "mean_us"):
            assert isinstance(blob["regimes"][r][k], (int, float))
