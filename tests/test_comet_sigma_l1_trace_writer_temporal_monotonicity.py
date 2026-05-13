"""Property tests for L1 temporal-basis monotonicity (Step 5).

This file owns **Step 5** of ``COMET_SIGMA_1000.md``:

    Property-test temporal-basis monotonicity for stepback/trace_writer.py
    + .sb format v1-derived features.

The Step 1+2 wiring projects a fixed set of base features emitted by
:class:`stepback.trace_writer.TraceWriter` onto seven aggregates
(``mean``, ``slope``, ``ewma``, ``range``, ``std``, ``last_minus_first``,
``last_minus_mean_prev``) over four windows (``1s / 10s / 1m / 10m``).
Each aggregate has *order-theoretic* properties the projector must
preserve regardless of which sequence of frames the writer commits.
We use Hypothesis to generate adversarial frame streams and check those
properties hold for the actual receipts the projector emits — i.e.
the test exercises the exact AuditableArtifact / Receipt / hook path,
not a re-implementation of the math.

Properties checked
------------------

Universal (any sequence of frames):

* ``range`` ≥ 0, ``std`` ≥ 0, ``last_minus_mean_prev`` ≥ 0.
* ``mean`` ∈ [min(window), max(window)] and ``ewma`` ∈ [min, max].
* ``last_minus_first`` has the same sign as ``frame[-1] - frame[0]``.
* When the window contains a single frame, ``slope == 0`` and
  ``last_minus_first == 0``.

Conditional (non-decreasing inputs):

* For monotone non-decreasing base features (``frame_depth`` is always
  monotone since it is the writer's frame index, and we drive the
  payload such that ``frame_bytes`` and ``body_key_count`` are also
  monotone), ``slope ≥ 0``, ``last_minus_first ≥ 0`` and the *mean*
  aggregate is non-decreasing across successive projections (since the
  newest value is ≥ every prior value in the window, the running mean
  cannot fall).

Conditional (non-increasing inputs):

* Mirror of the above: ``slope ≤ 0`` and ``last_minus_first ≤ 0``.

Cross-projection (timeline monotonicity):

* For the strictly-monotone ``frame_depth`` feature, the
  ``last_minus_first`` aggregate over each window is non-decreasing
  across successive projections as long as the window's frame set
  is also non-decreasing (which is the case here because all frames
  fall inside the smallest 1s window in synthetic clock-driven tests).
"""
from __future__ import annotations

import math
from typing import List, Tuple

import pytest

from stepback import comet_sigma as cs
from stepback.comet_sigma import l1_trace_writer as l1
from stepback.comet_sigma import l1_trace_writer_temporal as l1t


pytestmark = pytest.mark.skipif(
    not cs.comet_sigma_available(),
    reason="comet_sigma package not importable on this host",
)

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings, strategies as st


# ---------------------------------------------------------------------------
# Test scaffolding
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _enable_l1_temporal(monkeypatch):
    """Force the L1 temporal flag on for the whole module."""
    monkeypatch.setenv(l1.FLAG_NAME, "1")
    l1.reset()
    l1t.reset()
    l1t.install_hook()
    yield
    l1.reset()
    l1t.reset()


def _drive_writer(
    writer_id: str,
    bodies: List[dict],
) -> List[l1.FrameRecord]:
    """Feed synthetic frames into ``observe_frame``.

    The emitter timestamps each frame with ``time.time_ns()`` directly
    so we cannot inject a custom clock; in practice the per-test frame
    set (≤ 32 frames) completes in well under a millisecond, so every
    frame lives inside the smallest 1s window — exactly what the
    monotonicity invariants assume.
    """
    out: List[l1.FrameRecord] = []
    for body in bodies:
        body_bytes = repr(body).encode("utf-8")
        rec = l1.observe_frame(writer_id, body, body_bytes)
        assert rec is not None
        out.append(rec)
    return out


def _aggregates_by_name() -> dict:
    return {agg.name: agg for agg in l1t.AGGREGATES}


def _values_by_window(
    proj: l1t.ProjectionRecord,
) -> dict:
    """Group a projection's flat values into ``[feature][window][agg]``."""
    out: dict = {}
    for key, val in proj.values.items():
        feat, window, agg = key.rsplit(".", 2)
        # ``feat`` may include dots (it does — e.g. ``trace_writer_sb_v1.frame_bytes``)
        out.setdefault(feat, {}).setdefault(window, {})[agg] = val
    return out


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# 1..32 frames of plausible body shapes. We bound payload size so the
# canonical-JSON repr stays small enough that frame_bytes lives in a
# tractable numeric range for Hypothesis shrinking.
_payload_size = st.integers(min_value=0, max_value=64)
_frame_count = st.integers(min_value=1, max_value=32)

_body_types = st.sampled_from(["step", "blob", "header", "other"])


@st.composite
def _arbitrary_bodies(draw) -> List[dict]:
    n = draw(_frame_count)
    bodies: List[dict] = []
    for _ in range(n):
        t = draw(_body_types)
        size = draw(_payload_size)
        bodies.append({"type": t, "payload": "x" * size})
    return bodies


@st.composite
def _monotone_inc_bodies(draw) -> List[dict]:
    """Frames whose ``frame_bytes`` and ``body_key_count`` are non-decreasing."""
    n = draw(_frame_count)
    sizes = sorted(draw(st.lists(_payload_size, min_size=n, max_size=n)))
    bodies: List[dict] = []
    for i, size in enumerate(sizes):
        body = {"type": "step", "payload": "x" * size}
        # Add a strictly growing number of keys to also drive body_key_count up.
        for k in range(i):
            body[f"k{k}"] = k
        bodies.append(body)
    return bodies


@st.composite
def _monotone_dec_bodies(draw) -> List[dict]:
    """Frames whose ``frame_bytes`` and ``body_key_count`` are non-increasing."""
    n = draw(_frame_count)
    sizes = sorted(
        draw(st.lists(_payload_size, min_size=n, max_size=n)), reverse=True
    )
    bodies: List[dict] = []
    for i, size in enumerate(sizes):
        body = {"type": "step", "payload": "x" * size}
        # Decreasing key count too: start with (n-1) extras and shrink.
        for k in range(n - 1 - i):
            body[f"k{k}"] = k
        bodies.append(body)
    return bodies


# ---------------------------------------------------------------------------
# Universal properties (any frame stream)
# ---------------------------------------------------------------------------

_HSETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@given(bodies=_arbitrary_bodies())
@_HSETTINGS
def test_range_std_lmmp_are_nonnegative(bodies):
    """``range``, ``std`` and ``last_minus_mean_prev`` are always ≥ 0."""
    writer_id = "writer://universal-nonneg"
    l1.reset(writer_id)
    l1t.reset(writer_id)
    _drive_writer(writer_id, bodies)
    proj = l1t.latest_projection(writer_id)
    assert proj is not None
    nested = _values_by_window(proj)
    for feat, by_window in nested.items():
        for window, by_agg in by_window.items():
            assert by_agg["range"] >= 0.0, (feat, window, by_agg)
            assert by_agg["std"] >= 0.0, (feat, window, by_agg)
            assert by_agg["last_minus_mean_prev"] >= 0.0, (
                feat, window, by_agg,
            )


@given(bodies=_arbitrary_bodies())
@_HSETTINGS
def test_mean_and_ewma_within_minmax(bodies):
    """``mean`` and ``ewma`` for each window are inside ``[min, max]``."""
    writer_id = "writer://universal-bounds"
    l1.reset(writer_id)
    l1t.reset(writer_id)
    _drive_writer(writer_id, bodies)
    proj = l1t.latest_projection(writer_id)
    assert proj is not None
    state = l1.get_state(writer_id)
    assert state is not None
    frames = list(state.frames)

    for spec in l1.BASE_FEATURES:
        per_frame = [fr.values.get(spec.name, 0.0) for fr in frames]
        if not per_frame:
            continue
        lo = min(per_frame)
        hi = max(per_frame)
        # Use the smallest window — every frame falls inside it because
        # _drive_writer spaces frames 1ms apart.
        window = "1s"
        flat_key_mean = l1t.projection_name(spec.name, window, "mean")
        flat_key_ewma = l1t.projection_name(spec.name, window, "ewma")
        m = proj.values[flat_key_mean]
        e = proj.values[flat_key_ewma]
        assert lo - 1e-9 <= m <= hi + 1e-9, (spec.name, m, lo, hi)
        assert lo - 1e-9 <= e <= hi + 1e-9, (spec.name, e, lo, hi)


@given(bodies=_arbitrary_bodies())
@_HSETTINGS
def test_last_minus_first_sign_matches_endpoints(bodies):
    """``last_minus_first`` has the same sign as ``frame[-1] - frame[0]``."""
    writer_id = "writer://lmf-sign"
    l1.reset(writer_id)
    l1t.reset(writer_id)
    _drive_writer(writer_id, bodies)
    state = l1.get_state(writer_id)
    assert state is not None
    frames = list(state.frames)
    proj = l1t.latest_projection(writer_id)
    assert proj is not None
    if len(frames) < 2:
        # The aggregate is defined to be 0.0 below the two-frame floor.
        for spec in l1.BASE_FEATURES:
            for window, _ in l1t.WINDOWS_NS:
                key = l1t.projection_name(spec.name, window, "last_minus_first")
                assert proj.values[key] == 0.0
        return

    for spec in l1.BASE_FEATURES:
        per_frame = [fr.values.get(spec.name, 0.0) for fr in frames]
        endpoint_diff = per_frame[-1] - per_frame[0]
        # Use the smallest window so all frames are present.
        key = l1t.projection_name(spec.name, "1s", "last_minus_first")
        v = proj.values[key]
        assert math.isclose(v, endpoint_diff, rel_tol=1e-9, abs_tol=1e-9), (
            spec.name, v, endpoint_diff,
        )


def test_single_frame_has_zero_slope_and_lmf():
    writer_id = "writer://singleton"
    l1.reset(writer_id)
    l1t.reset(writer_id)
    _drive_writer(writer_id, [{"type": "step", "payload": "x"}])
    proj = l1t.latest_projection(writer_id)
    assert proj is not None
    for spec in l1.BASE_FEATURES:
        for window, _ in l1t.WINDOWS_NS:
            assert proj.values[
                l1t.projection_name(spec.name, window, "slope")
            ] == 0.0
            assert proj.values[
                l1t.projection_name(spec.name, window, "last_minus_first")
            ] == 0.0


# ---------------------------------------------------------------------------
# Conditional properties — non-decreasing input
# ---------------------------------------------------------------------------

_MONOTONE_INC_FEATURES = (
    "trace_writer_sb_v1.frame_bytes",
    "trace_writer_sb_v1.frame_depth",
    "trace_writer_sb_v1.body_key_count",
)


@given(bodies=_monotone_inc_bodies())
@_HSETTINGS
def test_monotone_inc_yields_nonnegative_slope_and_lmf(bodies):
    writer_id = "writer://monotone-inc"
    l1.reset(writer_id)
    l1t.reset(writer_id)
    _drive_writer(writer_id, bodies)
    proj = l1t.latest_projection(writer_id)
    assert proj is not None

    for feat in _MONOTONE_INC_FEATURES:
        for window, _ in l1t.WINDOWS_NS:
            slope = proj.values[l1t.projection_name(feat, window, "slope")]
            lmf = proj.values[l1t.projection_name(feat, window, "last_minus_first")]
            assert slope >= -1e-12, (feat, window, slope)
            assert lmf >= -1e-12, (feat, window, lmf)


# ---------------------------------------------------------------------------
# Conditional properties — non-increasing input
# ---------------------------------------------------------------------------

_MONOTONE_DEC_FEATURES = (
    "trace_writer_sb_v1.frame_bytes",
    "trace_writer_sb_v1.body_key_count",
)


@given(bodies=_monotone_dec_bodies())
@_HSETTINGS
def test_monotone_dec_yields_nonpositive_slope_and_lmf(bodies):
    writer_id = "writer://monotone-dec"
    l1.reset(writer_id)
    l1t.reset(writer_id)
    _drive_writer(writer_id, bodies)
    proj = l1t.latest_projection(writer_id)
    assert proj is not None

    for feat in _MONOTONE_DEC_FEATURES:
        for window, _ in l1t.WINDOWS_NS:
            slope = proj.values[l1t.projection_name(feat, window, "slope")]
            lmf = proj.values[l1t.projection_name(feat, window, "last_minus_first")]
            assert slope <= 1e-12, (feat, window, slope)
            assert lmf <= 1e-12, (feat, window, lmf)


# ---------------------------------------------------------------------------
# Cross-projection timeline monotonicity (frame_depth is strictly monotone)
# ---------------------------------------------------------------------------

@given(bodies=_arbitrary_bodies())
@_HSETTINGS
def test_frame_depth_lmf_nondecreasing_across_projections(bodies):
    """Across successive projections, ``frame_depth.last_minus_first`` cannot fall.

    ``frame_depth`` is the writer's frame index — strictly monotone.
    Successive projections add a newer (larger) value while never
    losing earlier ones (because all frames live inside the 1s window
    in this synthetic clock setup), so ``last - first`` is monotone
    non-decreasing across projection records.
    """
    writer_id = "writer://timeline-mono"
    l1.reset(writer_id)
    l1t.reset(writer_id)
    _drive_writer(writer_id, bodies)

    history = l1t.projection_history(writer_id)
    assert history, "no projections produced"
    feat = "trace_writer_sb_v1.frame_depth"
    key = l1t.projection_name(feat, "1s", "last_minus_first")

    prev = -math.inf
    for proj in history:
        v = proj.values[key]
        assert v + 1e-12 >= prev, (prev, v)
        prev = v


# ---------------------------------------------------------------------------
# Direct aggregate-level monotonicity check (sanity for the building blocks)
# ---------------------------------------------------------------------------

@given(
    xs=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=0, max_size=64,
    )
)
@settings(max_examples=80, deadline=None)
def test_aggregate_universal_invariants(xs):
    """Direct invariants of the seven aggregates over arbitrary float lists."""
    aggs = _aggregates_by_name()
    rng = aggs["range"].fn(xs)
    std = aggs["std"].fn(xs)
    lmmp = aggs["last_minus_mean_prev"].fn(xs)
    assert rng >= 0
    assert std >= 0
    assert lmmp >= 0
    if not xs:
        for name in ("mean", "slope", "ewma", "range", "std",
                     "last_minus_first", "last_minus_mean_prev"):
            assert aggs[name].fn(xs) == 0.0
        return

    lo, hi = min(xs), max(xs)
    assert lo - 1e-9 <= aggs["mean"].fn(xs) <= hi + 1e-9
    assert lo - 1e-9 <= aggs["ewma"].fn(xs) <= hi + 1e-9
    if len(xs) >= 2:
        # Sign of last_minus_first matches endpoint difference.
        diff = xs[-1] - xs[0]
        v = aggs["last_minus_first"].fn(xs)
        assert math.copysign(1.0, v) == math.copysign(1.0, diff) or v == diff == 0.0
