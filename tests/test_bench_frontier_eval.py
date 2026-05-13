"""Tests for the frontier-model re-evaluation scheduler (Step 125).

Coverage:
* ModelRecord: to_json / from_json round-trip; is_superseded; release_date_obj
* FrontierModelCatalog: find_by_id, find_by_family, latest_in_family,
  current_frontier, is_superseded, superseder, frontier_families, len,
  to_json / from_json round-trip, empty catalog
* default_catalog: non-empty, contains expected families
* load_catalog_from_json / save_catalog_json: round-trip via temp file
* ReEvaluationPolicy: to_json, default values
* compute_schedule — age-based staleness:
    - fresh submission → not flagged
    - stale submission (age > max_age_days) → REASON_TOO_OLD
    - submission exactly on the boundary (age == max_age_days) → not flagged
    - submission one day past boundary → flagged
    - max_age_days=0 disables age check
* compute_schedule — model supersession:
    - superseded model → REASON_MODEL_SUPERSEDED + suggested newer model
    - current model → not flagged on supersession
    - check_superseded=False disables that check
* compute_schedule — new frontier in family:
    - older model in family (not the latest) → REASON_NEW_FRONTIER
    - latest model → not flagged on new-frontier
    - check_new_frontier=False disables that check
* compute_schedule — unknown model IDs:
    - unknown model_id → REASON_UNKNOWN_MODEL, model in unknown_models
* compute_schedule — no submission_models:
    - only age check applied when submission_models absent
* compute_schedule — mixed dict and object submissions
* compute_schedule — empty submission list → empty schedule
* compute_schedule — no reasons → item not in schedule
* compute_schedule — multiple reasons on one submission
* compute_schedule — items are sorted by submission_id
* compute_schedule — reference_date override
* ReEvaluationSchedule: needs_reeval_count, summary, to_json
* ReEvaluationItem: to_json fields including sorted lists
* save_schedule_json / load_schedule_json: round-trip via temp file
* CLI smoke test: --submissions dir --out path
* CLI exit-0 when nothing needs re-eval; exit-1 when items present
* CLI --max-age-days override
* CLI --catalog override with saved catalog
* CLI --no-superseded flag
* CLI --no-new-frontier flag
* CLI --reference-date override
* CLI --models-file injection
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import date

import pytest

from stepback.bench.frontier_eval import (
    CATALOG_GENERATED_AT,
    FRONTIER_MODELS,
    REASON_MODEL_SUPERSEDED,
    REASON_NEW_FRONTIER,
    REASON_TOO_OLD,
    REASON_UNKNOWN_MODEL,
    FrontierModelCatalog,
    ModelRecord,
    ReEvaluationItem,
    ReEvaluationPolicy,
    ReEvaluationSchedule,
    _extract_submission_id_and_date,
    _iso_to_date_str,
    compute_schedule,
    default_catalog,
    load_catalog_from_json,
    load_schedule_json,
    save_catalog_json,
    save_schedule_json,
)


# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------

# A minimal catalog with two families for isolated tests.
_REC_A1 = ModelRecord(
    model_id="model-a-v1",
    provider="provider-a",
    family="family-a",
    release_date="2023-01-01",
    superseded_by="model-a-v2",
)
_REC_A2 = ModelRecord(
    model_id="model-a-v2",
    provider="provider-a",
    family="family-a",
    release_date="2024-06-01",
    superseded_by=None,
)
_REC_B1 = ModelRecord(
    model_id="model-b-v1",
    provider="provider-b",
    family="family-b",
    release_date="2024-01-01",
    superseded_by=None,
)
_MINI_CATALOG = FrontierModelCatalog(
    [_REC_A1, _REC_A2, _REC_B1], catalog_generated_at="2025-01-01T00:00:00Z"
)

_REFERENCE_DATE = date(2025, 6, 15)  # fixed "today" for deterministic tests

# A stale submission dict (submitted 2 years ago relative to reference_date)
def _sub(
    sub_id: str = "sub-001",
    submission_date: str = "2023-06-01",
    bench_results: list | None = None,
) -> dict:
    return {
        "submission_id": sub_id,
        "audit": {"submission_date": submission_date},
        "bench_results": bench_results or [],
    }


def _fresh_sub(sub_id: str = "sub-fresh") -> dict:
    """Submission dated 10 days before reference_date."""
    return _sub(sub_id, submission_date="2025-06-05")


def _stale_sub(sub_id: str = "sub-stale") -> dict:
    """Submission dated 800 days before reference_date (well over 180)."""
    return _sub(sub_id, submission_date="2023-04-21")


# ---------------------------------------------------------------------------
# ModelRecord tests
# ---------------------------------------------------------------------------


def test_model_record_to_from_json_roundtrip():
    rec = _REC_A1
    d = rec.to_json()
    assert d["model_id"] == "model-a-v1"
    assert d["superseded_by"] == "model-a-v2"
    restored = ModelRecord.from_json(d)
    assert restored.model_id == rec.model_id
    assert restored.superseded_by == rec.superseded_by
    assert restored.release_date == rec.release_date


def test_model_record_from_json_no_superseded():
    d = {"model_id": "x", "provider": "p", "family": "f", "release_date": "2024-01-01"}
    rec = ModelRecord.from_json(d)
    assert rec.superseded_by is None


def test_model_record_is_superseded():
    assert _REC_A1.is_superseded() is True
    assert _REC_A2.is_superseded() is False


def test_model_record_release_date_obj():
    assert _REC_A1.release_date_obj == date(2023, 1, 1)


# ---------------------------------------------------------------------------
# FrontierModelCatalog tests
# ---------------------------------------------------------------------------


def test_catalog_len():
    assert len(_MINI_CATALOG) == 3


def test_catalog_find_by_id_found():
    rec = _MINI_CATALOG.find_by_id("model-a-v1")
    assert rec is not None
    assert rec.model_id == "model-a-v1"


def test_catalog_find_by_id_missing():
    assert _MINI_CATALOG.find_by_id("nonexistent") is None


def test_catalog_find_by_family_sorted():
    members = _MINI_CATALOG.find_by_family("family-a")
    assert len(members) == 2
    # sorted oldest first
    assert members[0].release_date <= members[1].release_date


def test_catalog_find_by_family_missing():
    assert _MINI_CATALOG.find_by_family("no-such-family") == []


def test_catalog_latest_in_family():
    latest = _MINI_CATALOG.latest_in_family("family-a")
    assert latest is not None
    assert latest.model_id == "model-a-v2"


def test_catalog_latest_in_family_missing():
    assert _MINI_CATALOG.latest_in_family("no-such-family") is None


def test_catalog_current_frontier_no_supersession():
    # model-b-v1 has no superseded_by and is the only member of family-b
    curr = _MINI_CATALOG.current_frontier("family-b")
    assert curr is not None
    assert curr.model_id == "model-b-v1"


def test_catalog_current_frontier_with_supersession():
    # model-a-v1 is superseded; model-a-v2 is active
    curr = _MINI_CATALOG.current_frontier("family-a")
    assert curr is not None
    assert curr.model_id == "model-a-v2"


def test_catalog_current_frontier_missing():
    assert _MINI_CATALOG.current_frontier("no-such-family") is None


def test_catalog_is_superseded_true():
    assert _MINI_CATALOG.is_superseded("model-a-v1") is True


def test_catalog_is_superseded_false():
    assert _MINI_CATALOG.is_superseded("model-a-v2") is False


def test_catalog_is_superseded_unknown():
    assert _MINI_CATALOG.is_superseded("nonexistent") is False


def test_catalog_superseder_present():
    succ = _MINI_CATALOG.superseder("model-a-v1")
    assert succ is not None
    assert succ.model_id == "model-a-v2"


def test_catalog_superseder_absent():
    assert _MINI_CATALOG.superseder("model-a-v2") is None


def test_catalog_superseder_unknown_id():
    assert _MINI_CATALOG.superseder("nonexistent") is None


def test_catalog_frontier_families():
    fams = _MINI_CATALOG.frontier_families()
    assert sorted(fams) == fams  # sorted
    assert "family-a" in fams
    assert "family-b" in fams


def test_catalog_models_property():
    models = _MINI_CATALOG.models
    assert len(models) == 3


def test_catalog_to_from_json_roundtrip():
    d = _MINI_CATALOG.to_json()
    assert d["catalog_generated_at"] == "2025-01-01T00:00:00Z"
    assert len(d["models"]) == 3
    restored = FrontierModelCatalog.from_json(d)
    assert len(restored) == 3
    assert restored.catalog_generated_at == "2025-01-01T00:00:00Z"
    assert restored.find_by_id("model-a-v1") is not None


def test_catalog_from_json_missing_models_key():
    restored = FrontierModelCatalog.from_json({})
    assert len(restored) == 0


def test_empty_catalog():
    cat = FrontierModelCatalog([])
    assert len(cat) == 0
    assert cat.find_by_id("x") is None
    assert cat.find_by_family("f") == []
    assert cat.latest_in_family("f") is None
    assert cat.frontier_families() == []


# ---------------------------------------------------------------------------
# default_catalog and FRONTIER_MODELS
# ---------------------------------------------------------------------------


def test_default_catalog_non_empty():
    cat = default_catalog()
    assert len(cat) > 0


def test_default_catalog_has_expected_families():
    cat = default_catalog()
    fams = cat.frontier_families()
    assert "gpt-4o" in fams
    assert "claude-3-5-sonnet" in fams


def test_default_catalog_generated_at():
    cat = default_catalog()
    assert cat.catalog_generated_at == CATALOG_GENERATED_AT


def test_frontier_models_list_is_list():
    assert isinstance(FRONTIER_MODELS, list)
    assert all(isinstance(m, ModelRecord) for m in FRONTIER_MODELS)


# ---------------------------------------------------------------------------
# save / load catalog
# ---------------------------------------------------------------------------


def test_save_load_catalog_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "catalog.json")
        save_catalog_json(_MINI_CATALOG, path)
        loaded = load_catalog_from_json(path)
    assert len(loaded) == len(_MINI_CATALOG)
    assert loaded.catalog_generated_at == _MINI_CATALOG.catalog_generated_at
    assert loaded.find_by_id("model-b-v1") is not None


def test_load_catalog_missing_file():
    with pytest.raises(FileNotFoundError):
        load_catalog_from_json("/nonexistent/path/catalog.json")


def test_load_catalog_bad_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('["not", "an", "object"]')
    with pytest.raises(ValueError, match="Expected a JSON object"):
        load_catalog_from_json(str(bad))


# ---------------------------------------------------------------------------
# ReEvaluationPolicy
# ---------------------------------------------------------------------------


def test_policy_defaults():
    p = ReEvaluationPolicy()
    assert p.max_age_days == 180
    assert p.check_superseded is True
    assert p.check_new_frontier is True


def test_policy_to_json():
    p = ReEvaluationPolicy(max_age_days=90, check_superseded=False, check_new_frontier=True)
    d = p.to_json()
    assert d == {"max_age_days": 90, "check_superseded": False, "check_new_frontier": True}


# ---------------------------------------------------------------------------
# compute_schedule — age-based tests
# ---------------------------------------------------------------------------


def test_fresh_submission_not_flagged():
    sched = compute_schedule(
        [_fresh_sub()],
        catalog=_MINI_CATALOG,
        policy=ReEvaluationPolicy(max_age_days=180),
        reference_date=_REFERENCE_DATE,
    )
    assert sched.needs_reeval_count == 0


def test_stale_submission_flagged_too_old():
    sched = compute_schedule(
        [_stale_sub()],
        catalog=_MINI_CATALOG,
        policy=ReEvaluationPolicy(max_age_days=180),
        reference_date=_REFERENCE_DATE,
    )
    assert sched.needs_reeval_count == 1
    item = sched.items[0]
    assert REASON_TOO_OLD in item.reasons


def test_submission_exactly_on_boundary_not_flagged():
    # age == max_age_days → NOT stale (> not >=)
    boundary_date = date(2024, 12, 17)  # exactly 180 days before 2025-06-15
    sched = compute_schedule(
        [_sub(submission_date=boundary_date.isoformat())],
        catalog=_MINI_CATALOG,
        policy=ReEvaluationPolicy(max_age_days=180),
        reference_date=_REFERENCE_DATE,
    )
    assert sched.needs_reeval_count == 0


def test_submission_one_day_past_boundary_flagged():
    # 181 days before reference_date
    past_date = date(2024, 12, 16)
    sched = compute_schedule(
        [_sub(submission_date=past_date.isoformat())],
        catalog=_MINI_CATALOG,
        policy=ReEvaluationPolicy(max_age_days=180),
        reference_date=_REFERENCE_DATE,
    )
    assert sched.needs_reeval_count == 1


def test_max_age_days_zero_disables_age_check():
    sched = compute_schedule(
        [_stale_sub()],
        catalog=_MINI_CATALOG,
        policy=ReEvaluationPolicy(max_age_days=0),
        reference_date=_REFERENCE_DATE,
    )
    assert sched.needs_reeval_count == 0


def test_age_days_field_populated():
    sched = compute_schedule(
        [_sub(submission_date="2024-12-16")],
        catalog=_MINI_CATALOG,
        policy=ReEvaluationPolicy(max_age_days=180),
        reference_date=_REFERENCE_DATE,
    )
    item = sched.items[0]
    assert item.age_days == 181


# ---------------------------------------------------------------------------
# compute_schedule — model supersession tests
# ---------------------------------------------------------------------------


def test_superseded_model_flagged():
    sched = compute_schedule(
        [_fresh_sub("sub-sup")],
        catalog=_MINI_CATALOG,
        policy=ReEvaluationPolicy(max_age_days=0, check_superseded=True),
        reference_date=_REFERENCE_DATE,
        submission_models={"sub-sup": ["model-a-v1"]},
    )
    assert sched.needs_reeval_count == 1
    item = sched.items[0]
    assert REASON_MODEL_SUPERSEDED in item.reasons
    assert "model-a-v1" in item.stale_models
    assert "model-a-v2" in item.suggested_models


def test_current_model_not_flagged_for_supersession():
    sched = compute_schedule(
        [_fresh_sub("sub-curr")],
        catalog=_MINI_CATALOG,
        policy=ReEvaluationPolicy(max_age_days=0, check_superseded=True, check_new_frontier=False),
        reference_date=_REFERENCE_DATE,
        submission_models={"sub-curr": ["model-a-v2"]},
    )
    assert sched.needs_reeval_count == 0


def test_check_superseded_false_skips_check():
    sched = compute_schedule(
        [_fresh_sub("sub-no-sup")],
        catalog=_MINI_CATALOG,
        policy=ReEvaluationPolicy(max_age_days=0, check_superseded=False, check_new_frontier=False),
        reference_date=_REFERENCE_DATE,
        submission_models={"sub-no-sup": ["model-a-v1"]},
    )
    assert sched.needs_reeval_count == 0


# ---------------------------------------------------------------------------
# compute_schedule — new frontier tests
# ---------------------------------------------------------------------------


def test_older_model_in_family_flagged_new_frontier():
    sched = compute_schedule(
        [_fresh_sub("sub-nf")],
        catalog=_MINI_CATALOG,
        policy=ReEvaluationPolicy(max_age_days=0, check_superseded=False, check_new_frontier=True),
        reference_date=_REFERENCE_DATE,
        submission_models={"sub-nf": ["model-a-v1"]},
    )
    assert sched.needs_reeval_count == 1
    item = sched.items[0]
    assert REASON_NEW_FRONTIER in item.reasons
    assert "model-a-v2" in item.suggested_models


def test_latest_model_not_flagged_new_frontier():
    sched = compute_schedule(
        [_fresh_sub("sub-latest")],
        catalog=_MINI_CATALOG,
        policy=ReEvaluationPolicy(max_age_days=0, check_superseded=False, check_new_frontier=True),
        reference_date=_REFERENCE_DATE,
        submission_models={"sub-latest": ["model-a-v2"]},
    )
    assert sched.needs_reeval_count == 0


def test_check_new_frontier_false_skips_check():
    sched = compute_schedule(
        [_fresh_sub("sub-no-nf")],
        catalog=_MINI_CATALOG,
        policy=ReEvaluationPolicy(max_age_days=0, check_new_frontier=False),
        reference_date=_REFERENCE_DATE,
        submission_models={"sub-no-nf": ["model-a-v1"]},
    )
    # model-a-v1 is superseded; but let's also disable that
    sched2 = compute_schedule(
        [_fresh_sub("sub-no-nf")],
        catalog=_MINI_CATALOG,
        policy=ReEvaluationPolicy(max_age_days=0, check_superseded=False, check_new_frontier=False),
        reference_date=_REFERENCE_DATE,
        submission_models={"sub-no-nf": ["model-a-v1"]},
    )
    assert sched2.needs_reeval_count == 0


# ---------------------------------------------------------------------------
# compute_schedule — unknown model
# ---------------------------------------------------------------------------


def test_unknown_model_flagged():
    sched = compute_schedule(
        [_fresh_sub("sub-unk")],
        catalog=_MINI_CATALOG,
        policy=ReEvaluationPolicy(max_age_days=0),
        reference_date=_REFERENCE_DATE,
        submission_models={"sub-unk": ["no-such-model-id"]},
    )
    assert sched.needs_reeval_count == 1
    item = sched.items[0]
    assert REASON_UNKNOWN_MODEL in item.reasons
    assert "no-such-model-id" in item.unknown_models


# ---------------------------------------------------------------------------
# compute_schedule — no submission_models
# ---------------------------------------------------------------------------


def test_no_submission_models_only_age_check():
    # stale submission with no model mapping → only age-based check
    sched = compute_schedule(
        [_stale_sub("sub-nomap")],
        catalog=_MINI_CATALOG,
        policy=ReEvaluationPolicy(max_age_days=180),
        reference_date=_REFERENCE_DATE,
        submission_models=None,
    )
    assert sched.needs_reeval_count == 1
    assert sched.items[0].reasons == [REASON_TOO_OLD]


# ---------------------------------------------------------------------------
# compute_schedule — empty list
# ---------------------------------------------------------------------------


def test_empty_submissions():
    sched = compute_schedule(
        [],
        catalog=_MINI_CATALOG,
        reference_date=_REFERENCE_DATE,
    )
    assert sched.total_submissions == 0
    assert sched.needs_reeval_count == 0
    assert sched.items == []


# ---------------------------------------------------------------------------
# compute_schedule — multiple reasons on one submission
# ---------------------------------------------------------------------------


def test_multiple_reasons_on_one_submission():
    sched = compute_schedule(
        [_stale_sub("sub-multi")],
        catalog=_MINI_CATALOG,
        policy=ReEvaluationPolicy(max_age_days=180, check_superseded=True),
        reference_date=_REFERENCE_DATE,
        submission_models={"sub-multi": ["model-a-v1"]},
    )
    item = sched.items[0]
    # both TOO_OLD and MODEL_SUPERSEDED (and maybe NEW_FRONTIER)
    assert REASON_TOO_OLD in item.reasons
    assert REASON_MODEL_SUPERSEDED in item.reasons


# ---------------------------------------------------------------------------
# compute_schedule — items sorted by submission_id
# ---------------------------------------------------------------------------


def test_items_sorted_by_submission_id():
    subs = [
        _sub("zzz-sub", submission_date="2022-01-01"),
        _sub("aaa-sub", submission_date="2022-01-01"),
        _sub("mmm-sub", submission_date="2022-01-01"),
    ]
    sched = compute_schedule(
        subs,
        catalog=_MINI_CATALOG,
        policy=ReEvaluationPolicy(max_age_days=30),
        reference_date=_REFERENCE_DATE,
    )
    ids = [item.submission_id for item in sched.items]
    assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# compute_schedule — reference_date override
# ---------------------------------------------------------------------------


def test_reference_date_override():
    # With a reference_date only 100 days after the submission, should NOT be stale.
    ref = date(2023, 9, 9)  # 100 days after 2023-06-01
    sched = compute_schedule(
        [_sub(submission_date="2023-06-01")],
        catalog=_MINI_CATALOG,
        policy=ReEvaluationPolicy(max_age_days=180),
        reference_date=ref,
    )
    assert sched.needs_reeval_count == 0


# ---------------------------------------------------------------------------
# compute_schedule — no reasons → item not in schedule
# ---------------------------------------------------------------------------


def test_no_reasons_not_in_schedule():
    sched = compute_schedule(
        [_fresh_sub("sub-good")],
        catalog=_MINI_CATALOG,
        policy=ReEvaluationPolicy(max_age_days=180, check_superseded=False, check_new_frontier=False),
        reference_date=_REFERENCE_DATE,
    )
    ids = [item.submission_id for item in sched.items]
    assert "sub-good" not in ids


# ---------------------------------------------------------------------------
# compute_schedule — bench_results timestamp preferred over audit date
# ---------------------------------------------------------------------------


def test_bench_results_timestamp_used_as_date():
    sub = {
        "submission_id": "sub-ts",
        "audit": {"submission_date": "2022-01-01"},  # stale
        "bench_results": [{"timestamp_utc": "2025-06-10T12:00:00Z"}],  # fresh
    }
    sched = compute_schedule(
        [sub],
        catalog=_MINI_CATALOG,
        policy=ReEvaluationPolicy(max_age_days=180),
        reference_date=_REFERENCE_DATE,
    )
    # 5 days ago → fresh
    assert sched.needs_reeval_count == 0


# ---------------------------------------------------------------------------
# ReEvaluationSchedule helpers
# ---------------------------------------------------------------------------


def _make_schedule(n_stale: int = 1) -> ReEvaluationSchedule:
    subs = [_stale_sub(f"sub-{i:03d}") for i in range(n_stale)]
    return compute_schedule(
        subs,
        catalog=_MINI_CATALOG,
        policy=ReEvaluationPolicy(max_age_days=180),
        reference_date=_REFERENCE_DATE,
    )


def test_schedule_needs_reeval_count():
    sched = _make_schedule(3)
    assert sched.needs_reeval_count == 3


def test_schedule_total_submissions():
    sched = compute_schedule(
        [_fresh_sub(), _stale_sub()],
        catalog=_MINI_CATALOG,
        policy=ReEvaluationPolicy(max_age_days=180),
        reference_date=_REFERENCE_DATE,
    )
    assert sched.total_submissions == 2


def test_schedule_summary_contains_key_tokens():
    sched = _make_schedule()
    s = sched.summary()
    assert "1/" in s or "1 /" in s or "1/1" in s
    assert "180" in s  # max_age_days
    assert _REFERENCE_DATE.isoformat() in s


def test_schedule_to_json_structure():
    sched = _make_schedule(2)
    d = sched.to_json()
    assert d["total_submissions"] == 2
    assert d["needs_reeval_count"] == 2
    assert "generated_at" in d
    assert "reference_date" in d
    assert "policy" in d
    assert isinstance(d["items"], list)


# ---------------------------------------------------------------------------
# ReEvaluationItem.to_json
# ---------------------------------------------------------------------------


def test_reeval_item_to_json_sorted_lists():
    item = ReEvaluationItem(
        submission_id="sub-001",
        submission_date="2023-01-01",
        age_days=365,
        reasons=[REASON_TOO_OLD, REASON_MODEL_SUPERSEDED],
        stale_models=["model-z", "model-a"],
        suggested_models=["model-b"],
        unknown_models=[],
    )
    d = item.to_json()
    assert d["reasons"] == sorted(d["reasons"])
    assert d["stale_models"] == sorted(d["stale_models"])
    assert d["submission_id"] == "sub-001"
    assert d["age_days"] == 365


# ---------------------------------------------------------------------------
# save / load schedule round-trip
# ---------------------------------------------------------------------------


def test_save_load_schedule_roundtrip():
    sched = _make_schedule(2)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "schedule.json")
        save_schedule_json(sched, path)
        loaded = load_schedule_json(path)
    assert loaded["total_submissions"] == 2
    assert loaded["needs_reeval_count"] == 2
    assert len(loaded["items"]) == 2


def test_save_schedule_creates_parent_dirs(tmp_path):
    sched = _make_schedule(1)
    deep = str(tmp_path / "a" / "b" / "c" / "schedule.json")
    save_schedule_json(sched, deep)
    assert os.path.exists(deep)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def test_iso_to_date_str_datetime():
    assert _iso_to_date_str("2025-03-01T12:00:00Z") == "2025-03-01"


def test_iso_to_date_str_date_only():
    assert _iso_to_date_str("2025-03-01") == "2025-03-01"


def test_iso_to_date_str_empty():
    assert _iso_to_date_str("") == "unknown"


def test_extract_submission_id_and_date_from_dict():
    d = {"submission_id": "s1", "audit": {"submission_date": "2024-01-01"}}
    sid, sdate = _extract_submission_id_and_date(d)
    assert sid == "s1"
    assert sdate == "2024-01-01"


def test_extract_submission_id_and_date_from_obj():
    class FakeManifest:
        submission_id = "s2"
        bench_results = []
        audit = type("A", (), {"submission_date": "2024-02-01"})()

    sid, sdate = _extract_submission_id_and_date(FakeManifest())
    assert sid == "s2"
    assert sdate == "2024-02-01"


def test_extract_submission_id_missing():
    sid, _ = _extract_submission_id_and_date({})
    assert sid == "unknown"


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def _run_cli(*args: str) -> tuple[int, str]:
    """Run the CLI and return (exit_code, stdout_captured)."""
    import io
    import sys

    from stepback.cli import main

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        exit_code = main(list(args))
    except SystemExit as e:
        exit_code = int(e.code) if e.code is not None else 0
    finally:
        sys.stdout = old_stdout
    return exit_code, captured.getvalue()


def test_cli_no_submissions_empty_schedule():
    """Running with no --submissions should produce an empty schedule (exit 0)."""
    code, out = _run_cli("bench", "frontier-reeval-schedule")
    assert code == 0
    assert "0/0" in out or "needs_reeval_count" in out


def test_cli_with_stale_submission_file(tmp_path):
    sub = _stale_sub("cli-stale")
    sub_path = tmp_path / "sub.json"
    sub_path.write_text(json.dumps(sub))

    out_path = tmp_path / "schedule.json"
    code, _ = _run_cli(
        "bench", "frontier-reeval-schedule",
        "--submissions", str(sub_path),
        "--reference-date", "2025-06-15",
        "--out", str(out_path),
    )
    assert code == 1  # items present → exit 1
    sched = json.loads(out_path.read_text())
    assert sched["needs_reeval_count"] == 1


def test_cli_fresh_submission_exits_zero(tmp_path):
    sub = _fresh_sub("cli-fresh")
    sub_path = tmp_path / "sub_fresh.json"
    sub_path.write_text(json.dumps(sub))

    code, out = _run_cli(
        "bench", "frontier-reeval-schedule",
        "--submissions", str(sub_path),
        "--reference-date", "2025-06-15",
    )
    assert code == 0


def test_cli_max_age_days_override(tmp_path):
    # submission 5 days old; with --max-age-days 3 it should be stale
    sub = _sub("cli-custom-age", submission_date="2025-06-10")
    sub_path = tmp_path / "sub_age.json"
    sub_path.write_text(json.dumps(sub))

    code, out = _run_cli(
        "bench", "frontier-reeval-schedule",
        "--submissions", str(sub_path),
        "--max-age-days", "3",
        "--reference-date", "2025-06-15",
    )
    assert code == 1


def test_cli_no_superseded_disables_model_check(tmp_path):
    """With --no-superseded, a submission with a superseded model should not be flagged
    for supersession (but may still be flagged for new-frontier unless also disabled)."""
    sub = _fresh_sub("cli-nosup")
    sub_path = tmp_path / "sub_nosup.json"
    sub_path.write_text(json.dumps(sub))

    models_file = tmp_path / "models.json"
    models_file.write_text(json.dumps({"cli-nosup": ["gpt-4o-2024-05-13"]}))

    cat = default_catalog()
    cat_path = tmp_path / "catalog.json"
    save_catalog_json(cat, str(cat_path))

    code, out = _run_cli(
        "bench", "frontier-reeval-schedule",
        "--submissions", str(sub_path),
        "--catalog", str(cat_path),
        "--models-file", str(models_file),
        "--reference-date", "2025-06-15",
        "--no-superseded",
        "--no-new-frontier",
    )
    # no-superseded + no-new-frontier + fresh submission → exit 0
    assert code == 0


def test_cli_custom_catalog(tmp_path):
    # use a mini catalog with no frontier models → unknown model
    sub = _fresh_sub("cli-cattest")
    sub_path = tmp_path / "sub_cat.json"
    sub_path.write_text(json.dumps(sub))

    mini_cat = FrontierModelCatalog([], catalog_generated_at="2025-01-01T00:00:00Z")
    cat_path = tmp_path / "mini_catalog.json"
    save_catalog_json(mini_cat, str(cat_path))

    models_file = tmp_path / "models.json"
    models_file.write_text(json.dumps({"cli-cattest": ["some-model-id"]}))

    code, out = _run_cli(
        "bench", "frontier-reeval-schedule",
        "--submissions", str(sub_path),
        "--catalog", str(cat_path),
        "--models-file", str(models_file),
        "--reference-date", "2025-06-15",
        "--max-age-days", "0",
    )
    # unknown model → REASON_UNKNOWN_MODEL → exit 1
    assert code == 1
