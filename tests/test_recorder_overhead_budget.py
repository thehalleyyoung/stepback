"""Enforce recorder-overhead budget — Steps 99, 121, and 136 of 100_STEPS.md.

Budget policy
-------------
The Python **fast path** — :py:meth:`stepback.recorder.Recorder.llm_call`
with a deterministic in-process executor — must keep its p50 overhead
(``delta_p50_us = recorded_p50_us − baseline_p50_us``) under **50 µs** on
developer hardware.

**Unsigned fast path** (Step 136)
    Use ``record(path, signing=False)`` or ``arecord(path, signing=False)`` to
    skip Ed25519 signing.  In the default ``compression=True`` mode, Ed25519 is
    already deferred to ``close()``, so the per-step saving is ~2 µs; in
    ``compression=False`` (streaming) mode, signing is on the hot path and
    removing it saves ~33 µs.  The unsigned path must also stay under 50 µs.

Shim-specific exceptions
~~~~~~~~~~~~~~~~~~~~~~~~~
Shims wrap the recorder and add canonicalization work on top.  Their budgets
are explicitly relaxed:

``wrap_openai`` / ``wrap_anthropic`` / ``wrap_bedrock`` / ``wrap_gemini``
    Response canonicalization (SDK struct → canonical dict) adds roughly
    **5–20 µs** per call above the base recorder.  Budget ceiling: **100 µs**.

``wrap_azure_openai``
    Identical to ``wrap_openai`` with the addition of a regional endpoint
    header parse; no material extra overhead.  Same ceiling: **100 µs**.

``wrap_vertex_ai`` / ``wrap_cohere`` / ``wrap_mistral``
    Similar to the OpenAI/Anthropic shims; provider-specific struct parsing
    adds **5–25 µs**.  Budget ceiling: **150 µs**.

Streaming shims (``StreamingRecorder``, ``arecord`` streaming paths)
    Chunk reassembly and incremental SHA-256 hashing are inherently more
    expensive.  No hard CI budget applies; overhead is typically **>100 µs**.

Tool-call shims (``wrap_langchain_tool``, ``wrap_mcp_session``)
    Tool outputs are typically small blobs.  Overhead is close to the base
    recorder path; ceiling: **100 µs**.

CI guardrail vs. developer target
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:data:`stepback.bench.record_overhead.OVERHEAD_BUDGET_TARGET_US` = 50 µs
    Enforced on developer hardware (strict target).

:data:`stepback.bench.record_overhead.OVERHEAD_BUDGET_CI_US` = 500 µs
    Enforced on GitHub Actions ``ubuntu-latest`` shared VMs.  Shared runners
    are typically 2–10× slower than developer laptops; the looser guardrail
    catches catastrophic regressions (e.g. a CPU-heavy hash being added to
    the hot path) without generating noise from VM jitter.

Override either value by setting ``STEPBACK_OVERHEAD_BUDGET_US`` in the
environment before running the test.

Excluding from regular CI passes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This test is marked ``@pytest.mark.overhead_budget`` so the normal test
matrix can skip it with ``-m "not overhead_budget"``::

    pytest -q --tb=short -m "not overhead_budget"

The dedicated ``overhead-budget`` CI job (see ``.github/workflows/ci.yml``)
runs *only* ``-m overhead_budget`` and passes ``STEPBACK_OVERHEAD_BUDGET_US``
appropriate for its runner.
"""
from __future__ import annotations

import pytest

from stepback.bench import record_overhead as ro
from stepback.bench.record_overhead import (
    overhead_budget_unsigned_us,
    overhead_budget_us,
    shim_overhead_budget_us,
)


@pytest.mark.overhead_budget
def test_recorder_p50_overhead_budget() -> None:
    """p50 overhead of the base recorder fast path must be under budget.

    Uses :func:`stepback.bench.record_overhead.overhead_budget_us` to read
    the active budget from ``STEPBACK_OVERHEAD_BUDGET_US`` or from the
    environment defaults (50 µs locally, 500 µs on GitHub Actions).
    """
    budget = overhead_budget_us()
    result = ro.run(n_steps=500, n_warmup=50)

    assert result.delta_p50_us < budget, (
        f"Recorder fast-path overhead p50 {result.delta_p50_us:.1f} µs "
        f"exceeds budget {budget:.0f} µs. "
        f"Full result: {result.summary_line()}"
    )


@pytest.mark.overhead_budget
def test_recorder_overhead_result_has_delta_fields() -> None:
    """delta_p50_us, delta_p95_us, and delta_p99_us must be present, positive, and consistent."""
    result = ro.run(n_steps=100, n_warmup=20)

    assert result.delta_p50_us >= 0, (
        f"delta_p50_us must be non-negative; got {result.delta_p50_us:.2f}"
    )
    assert result.delta_p95_us >= 0, (
        f"delta_p95_us must be non-negative; got {result.delta_p95_us:.2f}"
    )
    assert result.delta_p99_us >= 0, (
        f"delta_p99_us must be non-negative; got {result.delta_p99_us:.2f}"
    )
    # p99 overhead must be at least as large as p50 overhead (monotone).
    assert result.delta_p99_us >= result.delta_p50_us, (
        f"delta_p99_us ({result.delta_p99_us:.2f}) < delta_p50_us "
        f"({result.delta_p50_us:.2f}) — percentile monotonicity violated"
    )
    # All fields appear in the JSON serialisation.
    body = result.to_json()
    assert "delta_p50_us" in body
    assert "delta_p95_us" in body
    assert "delta_p99_us" in body
    # All appear in the summary line.
    line = result.summary_line()
    assert "delta_p50_us=" in line
    assert "delta_p95_us=" in line
    assert "delta_p99_us=" in line


@pytest.mark.overhead_budget
def test_overhead_budget_us_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``STEPBACK_OVERHEAD_BUDGET_US`` env var overrides the default budget."""
    monkeypatch.setenv("STEPBACK_OVERHEAD_BUDGET_US", "123.4")
    assert overhead_budget_us() == pytest.approx(123.4)


@pytest.mark.overhead_budget
def test_overhead_budget_us_ci_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """When GITHUB_ACTIONS=true and no override, returns the CI guardrail."""
    monkeypatch.delenv("STEPBACK_OVERHEAD_BUDGET_US", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert overhead_budget_us() == ro.OVERHEAD_BUDGET_CI_US


@pytest.mark.overhead_budget
def test_overhead_budget_us_local_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """When neither env var is set, returns the developer target."""
    monkeypatch.delenv("STEPBACK_OVERHEAD_BUDGET_US", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert overhead_budget_us() == ro.OVERHEAD_BUDGET_TARGET_US


# ---------------------------------------------------------------------------
# Step 121 — per-shim overhead tests
# ---------------------------------------------------------------------------


def test_shim_overhead_result_fields() -> None:
    """ShimOverheadResult has all required fields and within_p50_budget property."""
    r = ro.ShimOverheadResult(
        shim_name="openai",
        n_steps=10,
        baseline_p50_us=5.0,
        baseline_p95_us=8.0,
        baseline_p99_us=12.0,
        recorded_p50_us=55.0,
        recorded_p95_us=80.0,
        recorded_p99_us=100.0,
        delta_p50_us=50.0,
        delta_p95_us=72.0,
        delta_p99_us=88.0,
        budget_p50_us=100.0,
        trace_bytes=1024,
    )
    assert r.within_p50_budget is True
    assert r.delta_p50_us == pytest.approx(50.0)
    assert r.delta_p95_us == pytest.approx(72.0)
    assert r.delta_p99_us == pytest.approx(88.0)
    body = r.to_json()
    for key in ("shim_name", "delta_p50_us", "delta_p95_us", "delta_p99_us",
                "budget_p50_us", "within_p50_budget", "trace_bytes"):
        assert key in body
    line = r.summary_line()
    assert "delta_p50_us=" in line
    assert "delta_p95_us=" in line


def test_shim_overhead_budget_us_known_shims(monkeypatch: pytest.MonkeyPatch) -> None:
    """All SHIM_NAMES (minus 'base') have non-zero dev and CI budgets."""
    monkeypatch.delenv("STEPBACK_OVERHEAD_BUDGET_US", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    for name in ro.SHIM_NAMES:
        if name == "base":
            continue
        budget = shim_overhead_budget_us(name)
        assert budget > 0, f"Budget for {name!r} should be positive, got {budget}"
        assert budget >= ro.OVERHEAD_BUDGET_TARGET_US, (
            f"Budget for shim {name!r} ({budget} µs) must be >= base target "
            f"({ro.OVERHEAD_BUDGET_TARGET_US} µs)"
        )


def test_shim_overhead_budget_us_unknown_raises() -> None:
    """shim_overhead_budget_us raises KeyError for unknown shim names."""
    with pytest.raises(KeyError, match="unknown_shim"):
        shim_overhead_budget_us("unknown_shim")


def test_shim_overhead_budget_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-shim env var overrides the default budget."""
    monkeypatch.setenv("STEPBACK_SHIM_OVERHEAD_BUDGET_OPENAI_US", "42.0")
    monkeypatch.delenv("STEPBACK_OVERHEAD_BUDGET_US", raising=False)
    assert shim_overhead_budget_us("openai") == pytest.approx(42.0)


def test_shim_overhead_budget_global_env_scales_proportionally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Global STEPBACK_OVERHEAD_BUDGET_US scales shim budgets proportionally."""
    monkeypatch.setenv("STEPBACK_OVERHEAD_BUDGET_US", "100.0")
    monkeypatch.delenv("STEPBACK_SHIM_OVERHEAD_BUDGET_OPENAI_US", raising=False)
    # openai dev budget = 100 µs, base dev budget = 50 µs → ratio 2×
    assert shim_overhead_budget_us("openai") == pytest.approx(200.0)


def test_shim_overhead_budget_ci_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """When GITHUB_ACTIONS=true, CI budgets are used."""
    monkeypatch.delenv("STEPBACK_OVERHEAD_BUDGET_US", raising=False)
    monkeypatch.delenv("STEPBACK_SHIM_OVERHEAD_BUDGET_OPENAI_US", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert shim_overhead_budget_us("openai") == pytest.approx(ro._SHIM_BUDGETS_CI_US["openai"])


@pytest.mark.overhead_budget
@pytest.mark.parametrize("shim_name", [
    "openai", "anthropic", "bedrock", "gemini",
    "langchain_tool", "mcp_tool", "openai_compat",
    "cohere", "mistral",
    "llamaindex", "dspy", "haystack", "autogen", "crewai",
    "semantic_kernel", "strands", "pydantic_ai", "inspect_ai",
])
def test_shim_p50_overhead_budget(shim_name: str) -> None:
    """p50 overhead for each shim must be under its budget."""
    budget = shim_overhead_budget_us(shim_name)
    result = ro.run_shim_overhead(shim_name, n_steps=200, n_warmup=30)

    assert result.delta_p50_us < budget, (
        f"Shim {shim_name!r} p50 overhead {result.delta_p50_us:.1f} µs "
        f"exceeds budget {budget:.0f} µs. "
        f"{result.summary_line()}"
    )
    assert result.delta_p50_us >= 0, (
        f"delta_p50_us must be non-negative for {shim_name!r}; "
        f"got {result.delta_p50_us:.2f}"
    )


# ---------------------------------------------------------------------------
# Step 136 — unsigned fast path tests
# ---------------------------------------------------------------------------

@pytest.mark.overhead_budget
def test_recorder_unsigned_p50_overhead_budget() -> None:
    """p50 overhead of the unsigned fast path must be under budget (Step 136).

    The unsigned path (``signing=False``) skips Ed25519 signing.  In the
    default ``compression=True`` mode Ed25519 is already deferred to
    ``close()`` so the per-step saving vs. signed is ~2 µs; the budget
    target is :data:`~stepback.bench.record_overhead.OVERHEAD_BUDGET_UNSIGNED_TARGET_US`
    (50 µs locally, 500 µs on GitHub Actions).
    """
    budget = overhead_budget_unsigned_us()
    result = ro.run_unsigned(n_steps=500, n_warmup=50)

    assert result.delta_p50_us < budget, (
        f"Unsigned fast-path p50 overhead {result.delta_p50_us:.1f} µs "
        f"exceeds budget {budget:.0f} µs. "
        f"Full result: {result.summary_line()}"
    )


@pytest.mark.overhead_budget
def test_recorder_unsigned_faster_than_signed() -> None:
    """Unsigned fast path must have a lower p50 overhead than the signed path.

    This is a regression guard: if Ed25519 signing is accidentally re-added
    to the unsigned per-step path, overhead will measurably increase.
    Marked ``overhead_budget`` because repeated micro-benchmark calls are
    required for statistical confidence.
    """
    signed = ro.run(n_steps=500, n_warmup=50)
    unsigned = ro.run_unsigned(n_steps=500, n_warmup=50)

    assert unsigned.delta_p50_us < signed.delta_p50_us + 5.0, (
        f"Unsigned p50 overhead ({unsigned.delta_p50_us:.1f} µs) is not "
        f"meaningfully lower than signed p50 overhead ({signed.delta_p50_us:.1f} µs). "
        f"Check that signing=False actually skips Ed25519 (in no-compression mode "
        f"the saving is ~33 µs; in compression=True mode the saving is ~2 µs)."
    )


def test_overhead_budget_unsigned_us_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """overhead_budget_unsigned_us() returns dev target when no env vars are set."""
    monkeypatch.delenv("STEPBACK_OVERHEAD_BUDGET_UNSIGNED_US", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert overhead_budget_unsigned_us() == pytest.approx(ro.OVERHEAD_BUDGET_UNSIGNED_TARGET_US)


def test_overhead_budget_unsigned_us_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    """overhead_budget_unsigned_us() returns CI guardrail when GITHUB_ACTIONS=true."""
    monkeypatch.delenv("STEPBACK_OVERHEAD_BUDGET_UNSIGNED_US", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert overhead_budget_unsigned_us() == pytest.approx(ro.OVERHEAD_BUDGET_UNSIGNED_CI_US)


def test_overhead_budget_unsigned_us_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """STEPBACK_OVERHEAD_BUDGET_UNSIGNED_US env var overrides the budget."""
    monkeypatch.setenv("STEPBACK_OVERHEAD_BUDGET_UNSIGNED_US", "99.5")
    assert overhead_budget_unsigned_us() == pytest.approx(99.5)


def test_run_unsigned_result_shape() -> None:
    """run_unsigned() returns a RecordOverheadResult with non-negative delta fields."""
    result = ro.run_unsigned(n_steps=50, n_warmup=10)

    assert result.delta_p50_us >= 0
    assert result.delta_p95_us >= 0
    assert result.delta_p99_us >= 0
    assert result.delta_p99_us >= result.delta_p50_us, (
        "p99 overhead must be >= p50 (monotone)"
    )
    body = result.to_json()
    for key in ("delta_p50_us", "delta_p95_us", "delta_p99_us", "trace_bytes"):
        assert key in body
    line = result.summary_line()
    assert "delta_p50_us=" in line


def test_unsigned_trace_is_readable() -> None:
    """A trace written with signing=False can be read back via verify_trace."""
    import os
    import tempfile

    from stepback.recorder import RecorderKey, record
    from stepback.trace_reader import verify_trace

    key = RecorderKey.fresh()
    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        path = f.name
    try:
        with record(path, key=key, signing=False) as rec:
            rec.tool_call("echo", {"x": 1}, lambda n, a: {"result": "ok"})
        trace = verify_trace(path, key.hmac_key)
        assert len(trace.steps) == 1
        assert trace.public_key_hex == ""
    finally:
        os.unlink(path)


def test_unsigned_trace_header_has_empty_public_key() -> None:
    """Unsigned traces must have public_key='' in the header."""
    import os
    import tempfile

    from stepback.recorder import RecorderKey, record
    from stepback.trace_reader import read_frames

    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        path = f.name
    try:
        with record(path, key=RecorderKey.fresh(), signing=False) as rec:
            rec.tool_call("ping", {}, lambda n, a: "pong")
        frames = read_frames(path)
        header_body = next(
            w["body"] for w in frames if isinstance(w, dict) and w.get("body", {}).get("type") == "header"
        )
        assert header_body["public_key"] == ""
        # All sig fields should be "none".
        for wrapper in frames:
            if isinstance(wrapper, dict) and "sig" in wrapper:
                assert wrapper["sig"] == "none", (
                    f"Expected sig='none' in unsigned trace, got {wrapper['sig']!r}"
                )
    finally:
        os.unlink(path)
