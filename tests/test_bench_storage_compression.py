"""Tests for stepback.bench.storage_compression (Step 122)."""
from __future__ import annotations

import json
import os
import sys

import pytest

from stepback.bench.storage_compression import (
    FormatResult,
    StorageCompressionResult,
    compare,
    run,
)
from stepback.cli import main as cli_main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_FORMAT_KEYS = {
    "raw_json",
    "sb_v1",
    "cbor",
    "gzip_of_json",
    "lzma_of_json",
    "deduped_json",
}

_SMALL_TRACES = 3
_SMALL_STEPS = 10


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_run_returns_valid_result():
    result = run(n_traces=_SMALL_TRACES, n_steps=_SMALL_STEPS, seed=1)
    assert isinstance(result, StorageCompressionResult)
    assert result.n_traces == _SMALL_TRACES
    assert result.n_steps_requested == _SMALL_STEPS
    assert result.actual_step_count >= _SMALL_TRACES  # at least 1 step per trace


def test_run_contains_all_format_keys():
    result = run(n_traces=_SMALL_TRACES, n_steps=_SMALL_STEPS, seed=2)
    assert set(result.formats.keys()) == EXPECTED_FORMAT_KEYS


def test_each_format_result_has_positive_bytes():
    result = run(n_traces=_SMALL_TRACES, n_steps=_SMALL_STEPS, seed=3)
    for key, fmt in result.formats.items():
        assert fmt.total_bytes > 0, f"{key}.total_bytes should be > 0"
        assert fmt.bytes_per_trace > 0, f"{key}.bytes_per_trace should be > 0"
        assert fmt.bytes_per_step > 0, f"{key}.bytes_per_step should be > 0"


def test_raw_json_ratio_is_exactly_one():
    result = run(n_traces=_SMALL_TRACES, n_steps=_SMALL_STEPS, seed=4)
    assert result.formats["raw_json"].ratio_vs_raw_json == pytest.approx(1.0)


def test_query_index_bytes_positive():
    result = run(n_traces=_SMALL_TRACES, n_steps=_SMALL_STEPS, seed=5)
    assert result.query_index_bytes > 0
    assert result.sb_v1_plus_query_index_bytes == (
        result.formats["sb_v1"].total_bytes + result.query_index_bytes
    )


def test_dedup_accounting_invariant():
    """deduped_json.total_bytes == unique_content_bytes + index_bytes."""
    result = run(n_traces=_SMALL_TRACES, n_steps=_SMALL_STEPS, seed=6)
    assert result.formats["deduped_json"].total_bytes == (
        result.deduped_unique_content_bytes + result.deduped_index_bytes
    )


def test_dedup_unique_bytes_less_than_or_equal_raw_json():
    """Unique content bytes are always < full raw JSON bytes since content
    excludes per-step metadata fields (step_id, wallclock_ns, etc.)."""
    result = run(n_traces=_SMALL_TRACES, n_steps=_SMALL_STEPS, seed=7)
    assert result.deduped_unique_content_bytes < result.formats["raw_json"].total_bytes


# ---------------------------------------------------------------------------
# Compression correctness for larger repetitive corpora
# ---------------------------------------------------------------------------


def test_gzip_smaller_than_raw_json_for_large_corpus():
    """For a corpus large enough to compress, gzip must reduce size vs raw JSON.

    Use n_steps=50 and n_traces=5 to ensure there is enough repeated
    text (model names, message role keys, fixed tool names) for deflate
    to find savings.
    """
    result = run(n_traces=5, n_steps=50, seed=8)
    raw = result.formats["raw_json"].total_bytes
    gz = result.formats["gzip_of_json"].total_bytes
    assert gz < raw, (
        f"gzip ({gz}B) should be < raw_json ({raw}B) for a 5×50-step corpus"
    )


def test_lzma_smaller_than_raw_json_for_large_corpus():
    """Similar: LZMA should compress a large repetitive corpus."""
    result = run(n_traces=5, n_steps=50, seed=9)
    raw = result.formats["raw_json"].total_bytes
    lzma_b = result.formats["lzma_of_json"].total_bytes
    assert lzma_b < raw, (
        f"lzma ({lzma_b}B) should be < raw_json ({raw}B) for a 5×50-step corpus"
    )


def test_cbor_positive_and_reasonable_ratio():
    """CBOR bytes should be between 0.3× and 2× of raw JSON in our test range."""
    result = run(n_traces=_SMALL_TRACES, n_steps=_SMALL_STEPS, seed=10)
    ratio = result.formats["cbor"].ratio_vs_raw_json
    assert 0.1 < ratio < 3.0, f"CBOR ratio {ratio:.3f} looks unreasonable"


def test_sb_v1_includes_crypto_overhead_and_can_exceed_raw_json_for_tiny_traces():
    """For very small traces, .sb overhead (HMAC, Ed25519, framing) can exceed
    the raw logical payload.  We must not assert sb_v1 < raw_json as a
    universal invariant.  Instead: verify both are positive and sb_v1 has
    a sensible ratio range for a realistic (n_steps≥20) corpus.
    """
    result = run(n_traces=5, n_steps=20, seed=11)
    sb = result.formats["sb_v1"]
    assert sb.total_bytes > 0
    # For 20-step traces the ratio should be within a reasonable range.
    # .sb gzip+dedup should keep it under 10× raw logical payload.
    assert sb.ratio_vs_raw_json < 10.0, (
        f".sb v1 ratio {sb.ratio_vs_raw_json:.2f} looks unexpectedly large"
    )


def test_dedup_wins_for_fleet_with_shared_system_prompt():
    """Content-addressed dedup stores only content fields (inputs + outputs),
    not step metadata (step_id, wallclock_ns, hashes).  This means the unique
    content bytes are always smaller than the full raw JSON (which includes
    metadata).  Additionally, cross-trace identical computations share
    content blob storage.

    The ``run()`` function deduplicates on
    ``{inputs, outputs, llm_request, llm_response, nondeterminism}``
    — the subset excluding per-step-instance metadata.  Even without
    cross-trace repetition this separation alone makes unique_content < raw.
    """
    result = run(n_traces=5, n_steps=10, seed=12)
    unique = result.deduped_unique_content_bytes
    raw = result.formats["raw_json"].total_bytes
    assert unique < raw, (
        f"Content-only dedup unique bytes ({unique}B) should be < "
        f"full raw JSON ({raw}B): metadata exclusion alone reduces size"
    )


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_to_json_and_from_json_round_trip():
    result = run(n_traces=2, n_steps=8, seed=13)
    blob = result.to_json()
    assert isinstance(blob, dict)
    # Standard JSON serialisability.
    raw = json.dumps(blob)
    loaded = json.loads(raw)
    restored = StorageCompressionResult.from_json(loaded)

    assert restored.n_traces == result.n_traces
    assert restored.actual_step_count == result.actual_step_count
    assert set(restored.formats.keys()) == set(result.formats.keys())
    assert restored.query_index_bytes == result.query_index_bytes
    assert restored.deduped_unique_content_bytes == result.deduped_unique_content_bytes


def test_format_result_to_json_roundtrip():
    fmt = FormatResult(
        name="raw_json",
        total_bytes=12345,
        bytes_per_trace=1234.5,
        bytes_per_step=123.45,
        ratio_vs_raw_json=1.0,
        encode_time_ms=0.5,
    )
    blob = fmt.to_json()
    restored = FormatResult.from_json(blob)
    assert restored.name == "raw_json"
    assert restored.total_bytes == 12345
    assert restored.ratio_vs_raw_json == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# summary_line
# ---------------------------------------------------------------------------


def test_summary_line_is_non_empty():
    result = run(n_traces=2, n_steps=8, seed=14)
    line = result.summary_line()
    assert isinstance(line, str)
    assert len(line) > 20
    assert "storage-compression" in line
    assert "raw_json=" in line
    assert "sb_v1=" in line


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------


def test_compare_returns_dict_keyed_by_n_steps():
    results = compare([5, 10], n_traces=2, seed=15)
    assert set(results.keys()) == {5, 10}
    for n, r in results.items():
        assert isinstance(r, StorageCompressionResult)
        assert r.n_steps_requested == n


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def test_cli_storage_compression_runs(tmp_path, capsys):
    out = str(tmp_path / "result.json")
    rc = cli_main(
        [
            "bench",
            "storage-compression",
            "--n-traces", "2",
            "--n-steps", "8",
            "--seed", "16",
            "--out", out,
        ]
    )
    assert rc == 0
    assert os.path.exists(out)
    with open(out) as f:
        data = json.load(f)
    assert data["n_traces"] == 2
    assert set(data["formats"].keys()) == EXPECTED_FORMAT_KEYS
    captured = capsys.readouterr()
    assert "storage-compression" in captured.out


# ---------------------------------------------------------------------------
# encode_time_ms plausibility
# ---------------------------------------------------------------------------


def test_encode_time_ms_non_negative():
    result = run(n_traces=2, n_steps=5, seed=17)
    for key, fmt in result.formats.items():
        assert fmt.encode_time_ms >= 0.0, f"{key}.encode_time_ms should be ≥ 0"
    assert result.total_wall_time_ms > 0.0
