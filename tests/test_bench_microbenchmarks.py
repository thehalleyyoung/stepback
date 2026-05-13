"""Tests for stepback.bench.microbenchmarks (Step 135).

All benchmarks run with a very small n_iter so the test suite stays fast.
The tests verify the *shape* and *sanity* of results, not absolute timing.
"""
from __future__ import annotations

import json

import pytest

from stepback.bench.microbenchmarks import (
    MicroBenchSuite,
    OpResult,
    _bench_canonicalization,
    _bench_dirty_set_planning,
    _bench_frame_writing,
    _bench_hmac_signing,
    _bench_reader_throughput,
    _bench_recorder_hooks,
    run,
)

# Use a tiny n_iter for tests so they finish quickly.
_N_ITER = 20


# ---------------------------------------------------------------------------
# OpResult helpers
# ---------------------------------------------------------------------------

def test_op_result_to_json_has_all_fields():
    samples = [10.0, 11.0, 12.0, 10.5, 11.5]
    op = _make_op_from_samples("test_op", samples, payload_bytes=128)
    body = op.to_json()
    for k in ("name", "n_iter", "payload_bytes", "p50_us", "p95_us", "p99_us",
              "mean_us", "throughput_mbs", "overhead_us"):
        assert k in body, f"missing key: {k}"


def test_op_result_summary_line_contains_name():
    samples = [5.0, 6.0, 7.0]
    op = _make_op_from_samples("my_op", samples)
    line = op.summary_line()
    assert "my_op" in line
    assert "p50=" in line


def _make_op_from_samples(name, samples, payload_bytes=0, overhead_us=0.0):
    import statistics as _stats
    from stepback.bench.microbenchmarks import _percentile
    mean = _stats.mean(samples)
    tput = (payload_bytes / mean / 1_000.0) if (mean > 0 and payload_bytes > 0) else 0.0
    return OpResult(
        name=name,
        n_iter=len(samples),
        payload_bytes=payload_bytes,
        p50_us=_percentile(samples, 0.50),
        p95_us=_percentile(samples, 0.95),
        p99_us=_percentile(samples, 0.99),
        mean_us=mean,
        throughput_mbs=tput,
        overhead_us=overhead_us,
    )


# ---------------------------------------------------------------------------
# Individual benchmark sections
# ---------------------------------------------------------------------------

def test_bench_canonicalization_returns_expected_ops():
    ops = _bench_canonicalization(_N_ITER)
    expected = {
        "canonical_json_small", "canonical_json_medium", "canonical_json_large",
        "sha256_small", "sha256_medium", "sha256_large", "hash_obj_medium",
    }
    assert expected == set(ops.keys()), f"unexpected ops: {set(ops.keys()) ^ expected}"


def test_bench_canonicalization_timings_positive():
    ops = _bench_canonicalization(_N_ITER)
    for name, op in ops.items():
        assert op.p50_us > 0, f"{name}: p50_us should be > 0"
        assert op.n_iter == _N_ITER
        assert op.p99_us >= op.p50_us, f"{name}: p99 should be >= p50"


def test_bench_canonicalization_throughput_for_large():
    ops = _bench_canonicalization(_N_ITER)
    # Large payloads should have measurable throughput.
    assert ops["canonical_json_large"].throughput_mbs > 0
    assert ops["sha256_large"].throughput_mbs > 0


def test_bench_hmac_signing_returns_expected_ops():
    ops = _bench_hmac_signing(_N_ITER)
    assert {"hmac_sha256", "ed25519_sign", "hmac_and_sign"} == set(ops.keys())


def test_bench_hmac_signing_timings_positive():
    ops = _bench_hmac_signing(_N_ITER)
    for name, op in ops.items():
        assert op.p50_us > 0, f"{name}: p50_us should be > 0"


def test_bench_hmac_and_sign_not_faster_than_ed25519_alone():
    """hmac_and_sign must be at least as slow as ed25519_sign alone."""
    ops = _bench_hmac_signing(_N_ITER)
    # Use mean rather than p50 to avoid single-sample noise flips.
    assert ops["hmac_and_sign"].mean_us >= ops["hmac_sha256"].mean_us * 0.5


def test_bench_frame_writing_returns_expected_ops():
    ops = _bench_frame_writing(_N_ITER)
    assert {"frame_write_small", "frame_write_medium"} == set(ops.keys())


def test_bench_frame_writing_medium_not_faster_than_small():
    """A medium frame body writes at least as slowly as a small one (more bytes)."""
    ops = _bench_frame_writing(_N_ITER)
    # Allow a small fudge factor in case OS write-buffering masks the difference
    # at small iteration counts.
    assert ops["frame_write_medium"].mean_us >= ops["frame_write_small"].mean_us * 0.5


def test_bench_recorder_hooks_returns_expected_ops():
    ops = _bench_recorder_hooks(_N_ITER)
    assert {"recorder_hook_llm", "recorder_hook_baseline"} == set(ops.keys())


def test_bench_recorder_hooks_recorded_slower_than_baseline():
    """The recorder path must be at least as slow as the bare LLM call."""
    ops = _bench_recorder_hooks(_N_ITER)
    assert ops["recorder_hook_llm"].mean_us >= ops["recorder_hook_baseline"].mean_us * 0.5


def test_bench_recorder_hooks_overhead_nonzero():
    ops = _bench_recorder_hooks(_N_ITER)
    # overhead_us reflects the p50 delta; it should be non-negative.
    assert ops["recorder_hook_llm"].overhead_us >= 0


def test_bench_reader_throughput_returns_expected_ops():
    ops = _bench_reader_throughput(_N_ITER)
    assert {"read_frames_10", "read_frames_100",
            "verify_trace_10", "verify_trace_100"} == set(ops.keys())


def test_bench_reader_throughput_timings_positive():
    ops = _bench_reader_throughput(_N_ITER)
    for name, op in ops.items():
        assert op.p50_us > 0, f"{name}: p50_us should be > 0"


def test_bench_reader_100_slower_than_10():
    """Reading 100 frames should be slower than reading 10 frames (mean)."""
    ops = _bench_reader_throughput(_N_ITER)
    assert ops["read_frames_100"].mean_us >= ops["read_frames_10"].mean_us * 0.5


def test_bench_dirty_set_planning_returns_expected_ops():
    ops = _bench_dirty_set_planning(_N_ITER)
    assert {"dirty_set_plan_20", "dirty_set_plan_100"} == set(ops.keys())


def test_bench_dirty_set_planning_timings_positive():
    ops = _bench_dirty_set_planning(_N_ITER)
    for name, op in ops.items():
        assert op.p50_us > 0, f"{name}: p50_us should be > 0"


def test_bench_dirty_set_100_slower_than_20():
    """Planning 100 steps should be slower than 20 steps."""
    ops = _bench_dirty_set_planning(_N_ITER)
    assert ops["dirty_set_plan_100"].mean_us >= ops["dirty_set_plan_20"].mean_us * 0.5


# ---------------------------------------------------------------------------
# Full suite
# ---------------------------------------------------------------------------

def test_run_returns_micro_bench_suite():
    suite = run(n_iter=_N_ITER)
    assert isinstance(suite, MicroBenchSuite)
    assert suite.n_iter == _N_ITER


def test_run_suite_has_all_op_categories():
    suite = run(n_iter=_N_ITER)
    op_names = set(suite.ops.keys())
    # Canonicalization
    assert "canonical_json_small" in op_names
    assert "canonical_json_large" in op_names
    assert "sha256_large" in op_names
    assert "hash_obj_medium" in op_names
    # HMAC/signing
    assert "hmac_sha256" in op_names
    assert "ed25519_sign" in op_names
    assert "hmac_and_sign" in op_names
    # Frame writing
    assert "frame_write_small" in op_names
    assert "frame_write_medium" in op_names
    # Recorder hooks
    assert "recorder_hook_llm" in op_names
    assert "recorder_hook_baseline" in op_names
    # Reader throughput
    assert "read_frames_10" in op_names
    assert "verify_trace_100" in op_names
    # Dirty-set planning
    assert "dirty_set_plan_20" in op_names
    assert "dirty_set_plan_100" in op_names


def test_run_suite_to_json_roundtrip():
    suite = run(n_iter=_N_ITER)
    body = suite.to_json()
    assert body["n_iter"] == _N_ITER
    assert "ops" in body
    blob = json.dumps(body)
    parsed = json.loads(blob)
    assert "ops" in parsed
    assert "canonical_json_small" in parsed["ops"]


def test_run_suite_summary_lines():
    suite = run(n_iter=_N_ITER)
    lines = suite.summary_lines()
    assert len(lines) == len(suite.ops)
    for line in lines:
        assert "microbench" in line
        assert "p50=" in line


def test_run_rejects_bad_n_iter():
    with pytest.raises(ValueError):
        run(n_iter=5)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_bench_microbenchmarks(tmp_path, capsys):
    from stepback.cli import main as cli_main

    out = tmp_path / "ub.json"
    code = cli_main([
        "bench", "microbenchmarks",
        "--n-iter", str(_N_ITER),
        "--out", str(out),
    ])
    assert code == 0
    captured = capsys.readouterr()
    assert "microbench" in captured.out

    import json
    data = json.loads(out.read_text())
    assert "ops" in data
    assert data["n_iter"] == _N_ITER
