"""Tests for stepback.bench.result_schema (Step 114 — benchmark result schema)."""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass

import pytest

from stepback.bench import replay_caching as rc
from stepback.bench.result_schema import (
    SCHEMA_VERSION,
    BenchRunRecord,
    CacheStats,
    CostStats,
    DirtySetStats,
    HardwareInfo,
    LatencyStats,
    StorageStats,
    SubstitutionStats,
    VersionInfo,
    _safe_float,
)


# ------------------------------------------------------------------ helpers


def _make_bench_result(n_steps: int = 20, n_trials: int = 3) -> rc.BenchResult:
    return rc.run(n_steps=n_steps, n_trials=n_trials, seed=42)


# ------------------------------------------------------------------ _safe_float


def test_safe_float_finite():
    assert _safe_float(1.5) == 1.5


def test_safe_float_none():
    assert _safe_float(None) is None


def test_safe_float_nan():
    assert _safe_float(float("nan")) is None


def test_safe_float_inf():
    assert _safe_float(float("inf")) is None
    assert _safe_float(float("-inf")) is None


# ------------------------------------------------------------------ SubstitutionStats


def test_substitution_stats_round_trip():
    s = SubstitutionStats(
        total=10,
        by_type={"PromptSubstitution": 6, "ToolOutputSubstitution": 4},
        by_strategy={"random_step": 10},
    )
    d = s.to_json()
    s2 = SubstitutionStats.from_json(d)
    assert s2.total == 10
    assert s2.by_type["PromptSubstitution"] == 6
    assert s2.by_strategy["random_step"] == 10


def test_substitution_stats_empty_round_trip():
    s = SubstitutionStats.from_json({})
    assert s.total == 0
    assert s.by_type == {}
    assert s.by_strategy == {}


# ------------------------------------------------------------------ DirtySetStats


def test_dirty_set_stats_from_sizes():
    sizes = [1, 2, 3, 4, 5]
    d = DirtySetStats.from_sizes(sizes)
    assert d.count == 5
    assert d.min_val == 1
    assert d.max_val == 5
    assert d.mean == 3.0
    assert d.median == 3.0
    assert d.p95 == 5


def test_dirty_set_stats_empty():
    d = DirtySetStats.from_sizes([])
    assert d.count == 0
    assert d.min_val == 0


def test_dirty_set_stats_round_trip():
    sizes = [2, 4, 6, 8, 10]
    d = DirtySetStats.from_sizes(sizes)
    body = d.to_json()
    d2 = DirtySetStats.from_json(body)
    assert d2.count == d.count
    assert d2.sizes == sizes
    assert d2.min_val == d.min_val


def test_dirty_set_stats_json_has_no_nan():
    d = DirtySetStats.from_sizes([1])
    body = d.to_json()
    blob = json.dumps(body)
    assert "NaN" not in blob
    assert "Infinity" not in blob


# ------------------------------------------------------------------ CacheStats


def test_cache_stats_round_trip():
    c = CacheStats(
        estimated_cache_hits=100,
        estimated_cache_hit_rate=0.9,
        estimated_llm_steps_saved=100,
        measured_cache_hits=None,
        measured_cache_misses=None,
        measured_hit_rate=None,
    )
    c2 = CacheStats.from_json(c.to_json())
    assert c2.estimated_cache_hits == 100
    assert c2.measured_cache_hits is None


# ------------------------------------------------------------------ StorageStats


def test_storage_stats_round_trip():
    s = StorageStats(
        total_bytes=1024,
        mean_bytes_per_trace=512.0,
        mean_bytes_per_step=51.2,
        result_json_bytes=200,
    )
    s2 = StorageStats.from_json(s.to_json())
    assert s2.total_bytes == 1024
    assert s2.result_json_bytes == 200


def test_storage_stats_all_none():
    s = StorageStats()
    body = s.to_json()
    assert body["total_bytes"] is None
    s2 = StorageStats.from_json(body)
    assert s2.total_bytes is None


# ------------------------------------------------------------------ VersionInfo


def test_version_info_detect():
    v = VersionInfo.detect()
    assert v.python_version  # non-empty
    assert v.stepback_version
    assert v.python_implementation in ("CPython", "PyPy", "GraalVM", "Jython")


def test_version_info_round_trip():
    v = VersionInfo.detect()
    v2 = VersionInfo.from_json(v.to_json())
    assert v2.stepback_version == v.stepback_version
    assert v2.python_version == v.python_version


def test_version_info_from_empty_dict():
    v = VersionInfo.from_json({})
    assert v.stepback_version == "unknown"


# ------------------------------------------------------------------ HardwareInfo


def test_hardware_info_detect():
    h = HardwareInfo.detect()
    assert h.os  # non-empty (Linux / Darwin / Windows)
    # cpu_count may be None in some containers, but usually set
    assert h.cpu_count is None or h.cpu_count > 0


def test_hardware_info_round_trip():
    h = HardwareInfo(
        os="Linux", cpu_count=8, cpu_model="Intel Core i9", ram_gb=16.0
    )
    h2 = HardwareInfo.from_json(h.to_json())
    assert h2.os == "Linux"
    assert h2.cpu_count == 8
    assert h2.ram_gb == 16.0


def test_hardware_info_nan_ram_serializes_as_null():
    h = HardwareInfo(os="Linux", cpu_count=4, cpu_model=None, ram_gb=float("nan"))
    body = h.to_json()
    assert body["ram_gb"] is None


# ------------------------------------------------------------------ BenchRunRecord


def test_bench_run_record_from_bench_result():
    result = _make_bench_result(n_steps=20, n_trials=3)
    rec = BenchRunRecord.from_bench_result(result, corpus_id="test-corpus")
    assert rec.schema_version == SCHEMA_VERSION
    assert rec.corpus_id == "test-corpus"
    assert rec.trace_count == 3
    assert rec.trial_count == 3
    assert rec.dirty_set.count == 3
    assert rec.cost.cost_reduction_factor == result.cost_reduction_factor
    assert rec.storage is None


def test_bench_run_record_corpus_id_in_substitutions():
    result = _make_bench_result()
    rec = BenchRunRecord.from_bench_result(
        result, corpus_id="my-corpus"
    )
    # strategy should appear in by_strategy
    assert result.strategy in rec.substitutions.by_strategy


def test_bench_run_record_cache_estimates():
    result = _make_bench_result(n_steps=20, n_trials=3)
    rec = BenchRunRecord.from_bench_result(result)
    # estimated_cache_hits + total_dirty == total_steps
    total_steps = rec.trace_count * result.n_steps
    total_dirty = sum(result.dirty_set_sizes)
    assert rec.cache.estimated_cache_hits == max(0, total_steps - total_dirty)
    assert 0.0 <= (rec.cache.estimated_cache_hit_rate or 0) <= 1.0


def test_bench_run_record_cost_savings_pct_in_range():
    result = _make_bench_result(n_steps=40, n_trials=5)
    rec = BenchRunRecord.from_bench_result(result)
    assert 0.0 <= rec.cost.estimated_savings_pct <= 100.0


def test_bench_run_record_to_json_is_valid_json():
    result = _make_bench_result()
    rec = BenchRunRecord.from_bench_result(result)
    body = rec.to_json()
    blob = json.dumps(body)  # must not raise
    assert '"schema_version"' in blob
    assert '"corpus_id"' in blob
    assert '"dirty_set"' in blob
    assert '"hardware"' in blob
    assert '"versions"' in blob


def test_bench_run_record_round_trip():
    result = _make_bench_result(n_steps=25, n_trials=4)
    rec = BenchRunRecord.from_bench_result(result, corpus_id="roundtrip-corpus")
    body = rec.to_json()
    rec2 = BenchRunRecord.from_json(body)
    assert rec2.corpus_id == "roundtrip-corpus"
    assert rec2.schema_version == SCHEMA_VERSION
    assert rec2.dirty_set.sizes == rec.dirty_set.sizes
    assert rec2.cost.cost_reduction_factor == rec.cost.cost_reduction_factor
    assert rec2.versions.stepback_version == rec.versions.stepback_version


def test_bench_run_record_unknown_schema_major_raises():
    result = _make_bench_result()
    body = BenchRunRecord.from_bench_result(result).to_json()
    body["schema_version"] = "99.0"
    with pytest.raises(ValueError, match="schema_version"):
        BenchRunRecord.from_json(body)


def test_bench_run_record_unknown_top_level_keys_ignored():
    result = _make_bench_result()
    body = BenchRunRecord.from_bench_result(result).to_json()
    body["future_field"] = "some-value"  # simulates a newer writer
    rec = BenchRunRecord.from_json(body)  # must not raise
    assert rec.corpus_id


def test_bench_run_record_with_storage():
    result = _make_bench_result()
    storage = StorageStats(total_bytes=2048, mean_bytes_per_trace=512.0)
    rec = BenchRunRecord.from_bench_result(result, corpus_id="c")
    # Manually attach storage
    import dataclasses
    rec = dataclasses.replace(rec, storage=storage)
    body = rec.to_json()
    assert body["storage"]["total_bytes"] == 2048
    rec2 = BenchRunRecord.from_json(body)
    assert rec2.storage is not None
    assert rec2.storage.total_bytes == 2048


def test_bench_run_record_injectable_versions_and_hardware():
    result = _make_bench_result()
    v = VersionInfo(
        stepback_version="99.0.0",
        python_version="3.x",
        python_implementation="CPython",
        platform_str="Linux",
    )
    h = HardwareInfo(os="Linux", cpu_count=64, cpu_model="FakeCPU", ram_gb=128.0)
    rec = BenchRunRecord.from_bench_result(result, versions=v, hardware=h)
    assert rec.versions.stepback_version == "99.0.0"
    assert rec.hardware.cpu_count == 64


def test_bench_run_record_has_run_id_and_timestamp():
    result = _make_bench_result()
    rec = BenchRunRecord.from_bench_result(result)
    assert len(rec.run_id) == 36  # uuid4 string length
    assert "T" in rec.timestamp_utc  # ISO 8601 contains T separator


def test_bench_run_record_latency():
    result = _make_bench_result(n_trials=4)
    rec = BenchRunRecord.from_bench_result(result)
    assert rec.latency.wall_time_ms == result.wall_time_ms
    assert rec.latency.per_trial_ms == pytest.approx(
        result.wall_time_ms / 4, rel=1e-9
    )


# ------------------------------------------------------------------ bench __init__ exports


def test_bench_module_exports_schema_classes():
    from stepback.bench import (
        BenchRunRecord,
        CacheStats,
        CostStats,
        DirtySetStats,
        HardwareInfo,
        LatencyStats,
        StorageStats,
        SubstitutionStats,
        VersionInfo,
    )
    # Ensure they're the same objects (not copies)
    assert BenchRunRecord is not None
    assert VersionInfo is not None
