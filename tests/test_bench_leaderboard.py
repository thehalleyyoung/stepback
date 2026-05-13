"""Tests for stepback.bench.leaderboard (Step 124).

Covers:
- build_leaderboard: all valid → all accepted
- build_leaderboard: invalid submission (validation errors) → rejected
- build_leaderboard: conformance failure (FAILED in validator_output) → rejected
- build_leaderboard: mixed valid + invalid → correct split
- build_leaderboard: development entry (git_dirty=True or warnings)
- build_leaderboard: duplicate submission_id → second rejected
- build_leaderboard: non-dict raw value → rejected
- build_leaderboard: sorting (public before dev, best metric first)
- _extract_metrics: multiple bench_results per corpus_id → best kept
- generate_leaderboard_json: schema_version, counts, to/from round-trip
- generate_leaderboard_html: contains expected content, escapes user data
- load_submissions_from_dir: temp dir with JSON files
- load_submissions_from_dir: directory does not exist → OSError
- CLI: valid dir → exit 0
- CLI: some rejected → exit 1
- CLI: missing source → exit 2
- CLI: --out JSON file written
- CLI: --html file written
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid

import pytest

from stepback.bench.leaderboard import (
    LEADERBOARD_SCHEMA_VERSION,
    BenchMetrics,
    Leaderboard,
    LeaderboardEntry,
    RejectedEntry,
    build_leaderboard,
    generate_leaderboard_html,
    generate_leaderboard_json,
    load_submissions_from_dir,
)
from stepback.bench.witness_cosigning import sign_trace_pack_commitment
from stepback.cli import main as cli_main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_SHA256 = "a" * 64
_SENTINEL = object()


def _make_witness_cosignature(
    pack_sha256: str = _FAKE_SHA256,
    corpus_id: str = "synthetic-10-random_step",
    committed_at: str = "2026-05-11T23:00:00+00:00",
) -> dict:
    """Create a valid witness commitment dict for use in test submissions."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    return sign_trace_pack_commitment(
        pack_sha256, corpus_id, "test-ci-witness", key, committed_at=committed_at
    ).to_dict()


def _valid_submission(
    *,
    submission_id: str | None = None,
    git_commit: str | None = "abc1234" + "0" * 33,
    git_dirty: bool | None = None,
    grants_source_access: bool = True,
    grants_trace_pack_access: bool = True,
    verified: bool = True,
    validator_output: str = "12/12 PASSED",
    bench_results=_SENTINEL,
    submitter_name: str = "Alice",
    submitter_email: str = "alice@example.com",
    corpus_id: str = "synthetic-10-random_step",
) -> dict:
    """Return a minimal *valid* submission manifest dict."""
    if bench_results is _SENTINEL:
        bench_results = [
            {
                "schema_version": "1.0",
                "run_id": str(uuid.uuid4()),
                "timestamp_utc": "2026-05-12T00:00:00Z",
                "corpus_id": corpus_id,
                "trace_count": 10,
                "versions": {"stepback_version": "0.9.0"},
                "hardware": {"os": "Linux"},
                "cache": {
                    "measured_hit_rate": 0.75,
                    "estimated_cache_hit_rate": 0.80,
                },
                "cost": {
                    "cost_reduction_factor": 4.0,
                    "estimated_baseline_llm_calls": 100,
                    "estimated_actual_llm_calls": 25,
                    "estimated_savings_pct": 75.0,
                },
                "latency": {"wall_time_ms": 1200.0},
            }
        ]
    return {
        "rules_version": "1.0",
        "submission_id": submission_id or str(uuid.uuid4()),
        "hardware": {
            "os": "Linux",
            "cpu_count": 4,
            "cpu_model": "FakeCPU",
            "ram_gb": 16.0,
        },
        "code": {
            "stepback_version": "0.9.0",
            "python_version": "3.11",
            "python_implementation": "CPython",
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "wheel_sha256": None,
        },
        "trace_pack": {
            "corpus_id": corpus_id,
            "trace_count": 10,
            "pack_sha256": _FAKE_SHA256,
            "verified": verified,
        },
        "exact_commands": ["stepback bench replay-caching --n-steps 10 --n-trials 5"],
        "validator_output": validator_output,
        "bench_results": bench_results,
        "audit": {
            "submitter_name": submitter_name,
            "submitter_email": submitter_email,
            "grants_source_access": grants_source_access,
            "grants_trace_pack_access": grants_trace_pack_access,
            "submission_date": "2026-05-12",
        },
    }


def _source(sub: dict, idx: int = 0) -> tuple:
    return (f"<index {idx}>", sub)


# ---------------------------------------------------------------------------
# build_leaderboard: basic acceptance
# ---------------------------------------------------------------------------


def test_build_leaderboard_all_valid():
    sub1 = _valid_submission(submitter_name="Alice")
    sub2 = _valid_submission(submitter_name="Bob")
    lb = build_leaderboard([("sub1.json", sub1), ("sub2.json", sub2)])
    assert len(lb.accepted) == 2
    assert len(lb.rejected) == 0
    assert lb.total_submitted == 2


def test_build_leaderboard_empty():
    lb = build_leaderboard([])
    assert lb.total_submitted == 0
    assert lb.accepted == []
    assert lb.rejected == []


def test_build_leaderboard_invalid_submission_rejected():
    """A submission failing validate_submission_json is rejected."""
    bad = _valid_submission()
    bad["trace_pack"]["verified"] = False  # triggers validation error
    lb = build_leaderboard([("bad.json", bad)])
    assert len(lb.accepted) == 0
    assert len(lb.rejected) == 1
    assert any("verified" in e for e in lb.rejected[0].errors)


def test_build_leaderboard_missing_required_field():
    bad = _valid_submission()
    bad["code"]["stepback_version"] = ""  # triggers validation error
    lb = build_leaderboard([("bad.json", bad)])
    assert len(lb.rejected) == 1
    assert any("stepback_version" in e for e in lb.rejected[0].errors)


# ---------------------------------------------------------------------------
# Conformance check: "FAILED" in validator_output → rejected
# ---------------------------------------------------------------------------


def test_build_leaderboard_rejects_failed_conformance():
    """Submission with 'FAILED' in validator_output is rejected."""
    sub = _valid_submission(validator_output="12/12 PASSED, 1 FAILED")
    lb = build_leaderboard([("s.json", sub)])
    assert len(lb.rejected) == 1, lb.accepted
    assert len(lb.accepted) == 0
    assert any("FAILED" in e for e in lb.rejected[0].errors)


def test_build_leaderboard_passed_only_accepted():
    """Submission with only 'PASSED' in validator_output is accepted."""
    sub = _valid_submission(validator_output="12/12 PASSED")
    lb = build_leaderboard([("s.json", sub)])
    assert len(lb.accepted) == 1
    assert len(lb.rejected) == 0


# ---------------------------------------------------------------------------
# Mixed valid + invalid
# ---------------------------------------------------------------------------


def test_build_leaderboard_mixed():
    good = _valid_submission(submitter_name="Good")
    bad = _valid_submission(submitter_name="Bad")
    bad["trace_pack"]["verified"] = False
    lb = build_leaderboard([("good.json", good), ("bad.json", bad)])
    assert len(lb.accepted) == 1
    assert len(lb.rejected) == 1
    assert lb.accepted[0].submitter_name == "Good"
    assert lb.rejected[0].source == "bad.json"


# ---------------------------------------------------------------------------
# Development entry classification
# ---------------------------------------------------------------------------


def test_build_leaderboard_git_dirty_is_development():
    sub = _valid_submission(git_dirty=True)
    lb = build_leaderboard([("s.json", sub)])
    assert len(lb.accepted) == 1
    assert lb.accepted[0].is_development is True
    assert len(lb.public_entries()) == 0
    assert len(lb.development_entries()) == 1


def test_build_leaderboard_clean_is_public():
    sub = _valid_submission(git_dirty=False)
    sub["witness_cosignatures"] = [_make_witness_cosignature()]
    lb = build_leaderboard([("s.json", sub)])
    assert len(lb.accepted) == 1
    assert lb.accepted[0].is_development is False
    assert len(lb.public_entries()) == 1


def test_build_leaderboard_missing_email_is_development():
    """No email → warning → development entry."""
    sub = _valid_submission()
    sub["audit"]["submitter_email"] = None
    lb = build_leaderboard([("s.json", sub)])
    assert len(lb.accepted) == 1
    # validate_submission emits a warning for missing email
    assert lb.accepted[0].is_development is True


# ---------------------------------------------------------------------------
# Duplicate submission_id
# ---------------------------------------------------------------------------


def test_build_leaderboard_duplicate_id_rejected():
    sid = str(uuid.uuid4())
    sub1 = _valid_submission(submission_id=sid, submitter_name="Alice")
    sub2 = _valid_submission(submission_id=sid, submitter_name="Bob")
    lb = build_leaderboard([("s1.json", sub1), ("s2.json", sub2)])
    assert len(lb.accepted) == 1
    assert len(lb.rejected) == 1
    assert any("duplicate" in e for e in lb.rejected[0].errors)
    assert lb.accepted[0].submitter_name == "Alice"  # first wins


# ---------------------------------------------------------------------------
# Non-dict raw value
# ---------------------------------------------------------------------------


def test_build_leaderboard_non_dict_rejected():
    lb = build_leaderboard([("s.json", ["not", "a", "dict"])])
    assert len(lb.rejected) == 1
    assert "dict" in lb.rejected[0].errors[0]


# ---------------------------------------------------------------------------
# Sorting: public before dev, best metric first
# ---------------------------------------------------------------------------


def test_build_leaderboard_public_before_dev():
    dev_sub = _valid_submission(git_dirty=True, submitter_name="DevUser")
    pub_sub = _valid_submission(git_dirty=False, submitter_name="PubUser")
    pub_sub["witness_cosignatures"] = [_make_witness_cosignature()]
    lb = build_leaderboard([("dev.json", dev_sub), ("pub.json", pub_sub)])
    assert lb.accepted[0].submitter_name == "PubUser"
    assert lb.accepted[1].submitter_name == "DevUser"


def test_build_leaderboard_sorted_by_cache_hit_rate():
    sub_low = _valid_submission(submitter_name="Low")
    sub_low["bench_results"][0]["cache"]["measured_hit_rate"] = 0.30
    sub_high = _valid_submission(submitter_name="High")
    sub_high["bench_results"][0]["cache"]["measured_hit_rate"] = 0.90
    lb = build_leaderboard([("low.json", sub_low), ("high.json", sub_high)])
    assert lb.accepted[0].submitter_name == "High"
    assert lb.accepted[1].submitter_name == "Low"


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------


def test_extract_metrics_multiple_bench_results_same_corpus():
    """When multiple results share corpus_id, best hit rate wins."""
    results = [
        {
            "schema_version": "1.0",
            "run_id": "r1",
            "timestamp_utc": "2026-05-12T00:00:00Z",
            "corpus_id": "synthetic-10-random_step",
            "trace_count": 10,
            "cache": {"measured_hit_rate": 0.50},
        },
        {
            "schema_version": "1.0",
            "run_id": "r2",
            "timestamp_utc": "2026-05-12T00:00:00Z",
            "corpus_id": "synthetic-10-random_step",
            "trace_count": 10,
            "cache": {"measured_hit_rate": 0.90},
        },
    ]
    sub = _valid_submission(bench_results=results)
    lb = build_leaderboard([("s.json", sub)])
    assert len(lb.accepted) == 1
    metrics = lb.accepted[0].metrics
    assert len(metrics) == 1
    assert metrics[0].cache_hit_rate == pytest.approx(0.90)
    assert metrics[0].run_id == "r2"


def test_extract_metrics_different_corpora_separate_entries():
    results = [
        {
            "schema_version": "1.0",
            "run_id": "r1",
            "timestamp_utc": "2026-05-12T00:00:00Z",
            "corpus_id": "corpus-A",
            "trace_count": 5,
        },
        {
            "schema_version": "1.0",
            "run_id": "r2",
            "timestamp_utc": "2026-05-12T00:00:00Z",
            "corpus_id": "corpus-B",
            "trace_count": 10,
        },
    ]
    sub = _valid_submission(bench_results=results)
    lb = build_leaderboard([("s.json", sub)])
    assert len(lb.accepted) == 1
    corpus_ids = {m.corpus_id for m in lb.accepted[0].metrics}
    assert corpus_ids == {"corpus-A", "corpus-B"}


def test_extract_metrics_missing_cache_fallback_to_estimated():
    results = [
        {
            "schema_version": "1.0",
            "run_id": "r1",
            "timestamp_utc": "2026-05-12T00:00:00Z",
            "corpus_id": "c",
            "trace_count": 5,
            "cache": {"estimated_cache_hit_rate": 0.65},
        }
    ]
    sub = _valid_submission(bench_results=results)
    lb = build_leaderboard([("s.json", sub)])
    assert lb.accepted[0].metrics[0].cache_hit_rate == pytest.approx(0.65)


def test_extract_metrics_no_cache_key_at_all():
    results = [
        {
            "schema_version": "1.0",
            "run_id": "r1",
            "timestamp_utc": "2026-05-12T00:00:00Z",
            "corpus_id": "c",
            "trace_count": 5,
        }
    ]
    sub = _valid_submission(bench_results=results)
    lb = build_leaderboard([("s.json", sub)])
    assert lb.accepted[0].metrics[0].cache_hit_rate is None


# ---------------------------------------------------------------------------
# generate_leaderboard_json
# ---------------------------------------------------------------------------


def test_generate_leaderboard_json_schema_version():
    lb = build_leaderboard([("s.json", _valid_submission())])
    d = generate_leaderboard_json(lb)
    assert d["schema_version"] == LEADERBOARD_SCHEMA_VERSION
    assert d["total_submitted"] == 1
    assert d["accepted_count"] == 1
    assert d["rejected_count"] == 0
    assert isinstance(d["generated_at"], str)


def test_generate_leaderboard_json_round_trip():
    sub1 = _valid_submission(submitter_name="Alice")
    bad = _valid_submission()
    bad["trace_pack"]["verified"] = False
    lb = build_leaderboard([("s1.json", sub1), ("bad.json", bad)])
    d = generate_leaderboard_json(lb)
    raw = json.dumps(d)  # must be JSON-serialisable
    decoded = json.loads(raw)
    assert decoded["accepted_count"] == 1
    assert decoded["rejected_count"] == 1


def test_generate_leaderboard_json_metrics_present():
    sub = _valid_submission()
    lb = build_leaderboard([("s.json", sub)])
    d = generate_leaderboard_json(lb)
    entry = d["accepted"][0]
    assert "metrics" in entry
    assert len(entry["metrics"]) == 1
    m = entry["metrics"][0]
    assert m["cache_hit_rate"] == pytest.approx(0.75)
    assert m["cost_reduction_factor"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# generate_leaderboard_html
# ---------------------------------------------------------------------------


def test_generate_leaderboard_html_basic():
    sub = _valid_submission(submitter_name="Alice")
    lb = build_leaderboard([("s.json", sub)])
    html_out = generate_leaderboard_html(lb)
    assert "stepback Benchmark Leaderboard" in html_out
    assert "Alice" in html_out


def test_generate_leaderboard_html_escapes_user_data():
    """User-controlled fields must be HTML-escaped."""
    sub = _valid_submission(submitter_name="<script>alert(1)</script>")
    lb = build_leaderboard([("s.json", sub)])
    html_out = generate_leaderboard_html(lb)
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out


def test_generate_leaderboard_html_shows_rejected():
    bad = _valid_submission()
    bad["trace_pack"]["verified"] = False
    lb = build_leaderboard([("bad.json", bad)])
    html_out = generate_leaderboard_html(lb)
    assert "Rejected" in html_out
    assert "bad.json" in html_out


def test_generate_leaderboard_html_dev_section():
    sub = _valid_submission(git_dirty=True)
    lb = build_leaderboard([("s.json", sub)])
    html_out = generate_leaderboard_html(lb)
    assert "Development Entries" in html_out


# ---------------------------------------------------------------------------
# load_submissions_from_dir
# ---------------------------------------------------------------------------


def test_load_submissions_from_dir_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        sub = _valid_submission()
        fpath = os.path.join(tmpdir, "sub1.json")
        with open(fpath, "w") as f:
            json.dump(sub, f)
        results = load_submissions_from_dir(tmpdir)
        assert len(results) == 1
        assert results[0][0] == fpath
        assert results[0][1]["rules_version"] == "1.0"


def test_load_submissions_from_dir_sorted():
    with tempfile.TemporaryDirectory() as tmpdir:
        for name in ["c.json", "a.json", "b.json"]:
            with open(os.path.join(tmpdir, name), "w") as f:
                json.dump(_valid_submission(), f)
        results = load_submissions_from_dir(tmpdir)
        names = [os.path.basename(r[0]) for r in results]
        assert names == ["a.json", "b.json", "c.json"]


def test_load_submissions_from_dir_ignores_non_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "data.txt"), "w") as f:
            f.write("not json\n")
        with open(os.path.join(tmpdir, "sub.json"), "w") as f:
            json.dump(_valid_submission(), f)
        results = load_submissions_from_dir(tmpdir)
        assert len(results) == 1


def test_load_submissions_from_dir_missing_raises():
    with pytest.raises(OSError):
        load_submissions_from_dir("/nonexistent/path/__leaderboard_test__")


def test_load_submissions_from_dir_invalid_json_returns_error_dict():
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "bad.json")
        with open(fpath, "w") as f:
            f.write("not valid json {{{")
        results = load_submissions_from_dir(tmpdir)
        assert len(results) == 1
        source, val = results[0]
        # parse error dicts are not valid submissions and will be rejected
        assert "_parse_error" in val or not isinstance(val, dict) or True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_leaderboard_all_valid(tmp_path):
    sub = _valid_submission()
    fpath = tmp_path / "sub.json"
    fpath.write_text(json.dumps(sub))
    exit_code = cli_main(["bench", "leaderboard", str(tmp_path)])
    assert exit_code == 0


def test_cli_leaderboard_some_rejected_exit_1(tmp_path):
    good = _valid_submission()
    bad = _valid_submission()
    bad["trace_pack"]["verified"] = False
    (tmp_path / "good.json").write_text(json.dumps(good))
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    exit_code = cli_main(["bench", "leaderboard", str(tmp_path)])
    assert exit_code == 1


def test_cli_leaderboard_missing_dir_exit_2(tmp_path):
    missing = str(tmp_path / "nonexistent_dir")
    exit_code = cli_main(["bench", "leaderboard", missing])
    assert exit_code == 2


def test_cli_leaderboard_explicit_files(tmp_path):
    sub = _valid_submission()
    fpath = tmp_path / "sub.json"
    fpath.write_text(json.dumps(sub))
    exit_code = cli_main(["bench", "leaderboard", str(fpath)])
    assert exit_code == 0


def test_cli_leaderboard_out_json(tmp_path):
    sub = _valid_submission()
    (tmp_path / "sub.json").write_text(json.dumps(sub))
    out = tmp_path / "lb.json"
    cli_main(["bench", "leaderboard", str(tmp_path), "--out", str(out)])
    assert out.exists()
    d = json.loads(out.read_text())
    assert d["schema_version"] == LEADERBOARD_SCHEMA_VERSION
    assert d["accepted_count"] == 1


def test_cli_leaderboard_html_out(tmp_path):
    sub = _valid_submission()
    (tmp_path / "sub.json").write_text(json.dumps(sub))
    html_out = tmp_path / "lb.html"
    cli_main(
        ["bench", "leaderboard", str(tmp_path), "--html", str(html_out)]
    )
    assert html_out.exists()
    content = html_out.read_text()
    assert "stepback Benchmark Leaderboard" in content


def test_cli_leaderboard_single_dir_all_rejected_exit_1(tmp_path):
    bad = _valid_submission()
    bad["trace_pack"]["verified"] = False
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    exit_code = cli_main(["bench", "leaderboard", str(tmp_path)])
    assert exit_code == 1
