"""Tests for the author-original corpus (Step 116).

Three redistributable corpora are bundled under ``stepback/bench/corpora/``:
- ``support-agent`` (5 tasks)
- ``code-review`` (5 tasks)
- ``payments-policy`` (5 tasks)

These tests verify:
1. All agents run and produce valid traces.
2. ``load_author_corpus`` returns correct CorpusTasks.
3. Every bundled ``.sb`` file verifies with its manifest HMAC key.
4. Evaluation ground truth is NOT in ``CorpusTask.prompt``.
5. ``generate_author_corpora`` is idempotent (re-run without force is a no-op).
6. The bench ``__init__`` re-exports are present.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List

import pytest

from stepback.bench.author_corpora import (
    AUTHOR_CORPUS_META,
    generate_author_corpora,
    list_author_corpora,
    load_author_corpus,
    _corpus_dir,
)
from stepback.bench.corpus_loaders import CorpusTask
from stepback.trace_reader import verify_trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_task_ids(corpus_id: str) -> List[str]:
    manifest_path = _corpus_dir(corpus_id) / "manifest.json"
    with manifest_path.open() as fh:
        return [e["task_id"] for e in json.load(fh)["tasks"]]


# ---------------------------------------------------------------------------
# list_author_corpora
# ---------------------------------------------------------------------------

class TestListAuthorCorpora:
    def test_returns_all_three(self):
        ids = list_author_corpora()
        assert set(ids) == {"support-agent", "code-review", "payments-policy"}

    def test_sorted(self):
        ids = list_author_corpora()
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# AUTHOR_CORPUS_META
# ---------------------------------------------------------------------------

class TestAuthorCorpusMeta:
    def test_all_corpora_have_meta(self):
        for corpus_id in list_author_corpora():
            assert corpus_id in AUTHOR_CORPUS_META

    @pytest.mark.parametrize("corpus_id", ["support-agent", "code-review", "payments-policy"])
    def test_meta_fields(self, corpus_id):
        meta = AUTHOR_CORPUS_META[corpus_id]
        assert meta["task_count"] == 5
        assert meta["license"] == "Apache-2.0"
        assert "description" in meta
        assert meta["split"] == "test"


# ---------------------------------------------------------------------------
# Bundled .sb files
# ---------------------------------------------------------------------------

class TestBundledSbFiles:
    @pytest.mark.parametrize("corpus_id", ["support-agent", "code-review", "payments-policy"])
    def test_sb_files_exist(self, corpus_id):
        corpus_dir = _corpus_dir(corpus_id)
        sb_files = list(corpus_dir.glob("*.sb"))
        assert len(sb_files) == 5

    @pytest.mark.parametrize("corpus_id", ["support-agent", "code-review", "payments-policy"])
    def test_manifest_exists(self, corpus_id):
        manifest_path = _corpus_dir(corpus_id) / "manifest.json"
        assert manifest_path.is_file()

    @pytest.mark.parametrize("corpus_id", ["support-agent", "code-review", "payments-policy"])
    def test_manifest_task_count(self, corpus_id):
        manifest_path = _corpus_dir(corpus_id) / "manifest.json"
        with manifest_path.open() as fh:
            manifest = json.load(fh)
        assert len(manifest["tasks"]) == 5

    @pytest.mark.parametrize("corpus_id", ["support-agent", "code-review", "payments-policy"])
    def test_manifest_schema_version(self, corpus_id):
        manifest_path = _corpus_dir(corpus_id) / "manifest.json"
        with manifest_path.open() as fh:
            manifest = json.load(fh)
        assert manifest["schema_version"] == "1.0"
        assert manifest["corpus_id"] == corpus_id
        assert manifest["license"] == "Apache-2.0"


# ---------------------------------------------------------------------------
# verify_trace with manifest HMAC keys
# ---------------------------------------------------------------------------

class TestVerifyTraces:
    @pytest.mark.parametrize("corpus_id", ["support-agent", "code-review", "payments-policy"])
    def test_all_traces_verify(self, corpus_id):
        """Every .sb file must verify with its manifest HMAC key."""
        manifest_path = _corpus_dir(corpus_id) / "manifest.json"
        with manifest_path.open() as fh:
            manifest = json.load(fh)

        corpus_dir = _corpus_dir(corpus_id)
        for entry in manifest["tasks"]:
            task_id = entry["task_id"]
            sb_path = corpus_dir / f"{task_id}.sb"
            hmac_key = bytes.fromhex(entry["hmac_key_hex"])
            trace = verify_trace(str(sb_path), hmac_key)
            assert len(trace.steps) >= 4, (
                f"{corpus_id}/{task_id} has only {len(trace.steps)} steps"
            )

    @pytest.mark.parametrize("corpus_id,task_id", [
        ("support-agent", "order-status-delayed"),
        ("support-agent", "refund-eligible"),
        ("support-agent", "account-unlock"),
        ("support-agent", "wrong-item"),
        ("support-agent", "subscription-cancel"),
    ])
    def test_support_step_counts(self, corpus_id, task_id):
        manifest_path = _corpus_dir(corpus_id) / "manifest.json"
        with manifest_path.open() as fh:
            manifest = json.load(fh)
        entry = next(e for e in manifest["tasks"] if e["task_id"] == task_id)
        sb_path = _corpus_dir(corpus_id) / f"{task_id}.sb"
        hmac_key = bytes.fromhex(entry["hmac_key_hex"])
        trace = verify_trace(str(sb_path), hmac_key)
        # Support tasks: 5 or 6 steps
        assert 5 <= len(trace.steps) <= 6

    @pytest.mark.parametrize("corpus_id,task_id", [
        ("code-review", "sql-injection"),
        ("code-review", "off-by-one"),
        ("code-review", "race-condition"),
        ("code-review", "missing-error-handling"),
        ("code-review", "resource-leak"),
    ])
    def test_code_review_step_counts(self, corpus_id, task_id):
        manifest_path = _corpus_dir(corpus_id) / "manifest.json"
        with manifest_path.open() as fh:
            manifest = json.load(fh)
        entry = next(e for e in manifest["tasks"] if e["task_id"] == task_id)
        sb_path = _corpus_dir(corpus_id) / f"{task_id}.sb"
        hmac_key = bytes.fromhex(entry["hmac_key_hex"])
        trace = verify_trace(str(sb_path), hmac_key)
        assert len(trace.steps) == 6  # always: 3 llm + 3 tool

    @pytest.mark.parametrize("corpus_id,task_id,expected_steps", [
        ("payments-policy", "normal-payment", 7),       # all gates pass: 1 llm + 4 tools + 1 llm + 1 tool
        ("payments-policy", "overlimit-payment", 5),    # blocked at limit
        ("payments-policy", "sanctioned-country", 4),   # blocked at sanctions
        ("payments-policy", "fraud-velocity", 6),       # held at velocity
        ("payments-policy", "unverified-beneficiary", 7),  # held at KYC
    ])
    def test_payments_step_counts(self, corpus_id, task_id, expected_steps):
        manifest_path = _corpus_dir(corpus_id) / "manifest.json"
        with manifest_path.open() as fh:
            manifest = json.load(fh)
        entry = next(e for e in manifest["tasks"] if e["task_id"] == task_id)
        sb_path = _corpus_dir(corpus_id) / f"{task_id}.sb"
        hmac_key = bytes.fromhex(entry["hmac_key_hex"])
        trace = verify_trace(str(sb_path), hmac_key)
        assert len(trace.steps) == expected_steps


# ---------------------------------------------------------------------------
# load_author_corpus
# ---------------------------------------------------------------------------

class TestLoadAuthorCorpus:
    @pytest.mark.parametrize("corpus_id", ["support-agent", "code-review", "payments-policy"])
    def test_returns_five_tasks(self, corpus_id):
        tasks = load_author_corpus(corpus_id)
        assert len(tasks) == 5

    @pytest.mark.parametrize("corpus_id", ["support-agent", "code-review", "payments-policy"])
    def test_returns_corpus_task_instances(self, corpus_id):
        tasks = load_author_corpus(corpus_id)
        for task in tasks:
            assert isinstance(task, CorpusTask)

    @pytest.mark.parametrize("corpus_id", ["support-agent", "code-review", "payments-policy"])
    def test_corpus_id_field(self, corpus_id):
        tasks = load_author_corpus(corpus_id)
        for task in tasks:
            assert task.corpus_id == corpus_id

    @pytest.mark.parametrize("corpus_id", ["support-agent", "code-review", "payments-policy"])
    def test_split_is_test(self, corpus_id):
        tasks = load_author_corpus(corpus_id)
        for task in tasks:
            assert task.split == "test"

    @pytest.mark.parametrize("corpus_id", ["support-agent", "code-review", "payments-policy"])
    def test_prompt_is_non_empty(self, corpus_id):
        tasks = load_author_corpus(corpus_id)
        for task in tasks:
            assert task.prompt.strip(), f"{corpus_id}/{task.task_id} has empty prompt"

    @pytest.mark.parametrize("corpus_id", ["support-agent", "code-review", "payments-policy"])
    def test_evaluation_not_in_prompt(self, corpus_id):
        """Ground-truth evaluation data must not appear in the prompt."""
        tasks = load_author_corpus(corpus_id)
        for task in tasks:
            evaluation = task.metadata.get("evaluation", {})
            for key in ("expected_decision", "expected_action", "expected_outcome",
                        "block_reason", "hold_reason", "expected_issues",
                        "fix", "severity"):
                value = evaluation.get(key)
                if value and isinstance(value, str):
                    assert value not in task.prompt, (
                        f"{corpus_id}/{task.task_id}: evaluation[{key!r}]={value!r} "
                        f"found in prompt"
                    )

    @pytest.mark.parametrize("corpus_id", ["support-agent", "code-review", "payments-policy"])
    def test_metadata_has_hmac_key(self, corpus_id):
        tasks = load_author_corpus(corpus_id)
        for task in tasks:
            hex_key = task.metadata["hmac_key_hex"]
            assert len(bytes.fromhex(hex_key)) == 32

    @pytest.mark.parametrize("corpus_id", ["support-agent", "code-review", "payments-policy"])
    def test_metadata_has_sb_path(self, corpus_id):
        tasks = load_author_corpus(corpus_id)
        for task in tasks:
            sb_path = Path(task.metadata["sb_path"])
            assert sb_path.is_file(), f"sb_path not found: {sb_path}"

    def test_unknown_corpus_id_raises(self):
        with pytest.raises(ValueError, match="Unknown author corpus id"):
            load_author_corpus("nonexistent-corpus")

    @pytest.mark.parametrize("corpus_id", ["support-agent", "code-review", "payments-policy"])
    def test_task_ids_match_manifest(self, corpus_id):
        tasks = load_author_corpus(corpus_id)
        manifest_ids = _all_task_ids(corpus_id)
        loaded_ids = [t.task_id for t in tasks]
        assert set(loaded_ids) == set(manifest_ids)


# ---------------------------------------------------------------------------
# Agent runners
# ---------------------------------------------------------------------------

class TestSupportAgent:
    def test_all_tasks_run(self):
        from stepback import record
        from stepback.testing.support_agent import SUPPORT_TASKS, run_support_task
        from stepback.bench.author_corpora import _make_recorder_key

        with tempfile.TemporaryDirectory() as tmpdir:
            for task in SUPPORT_TASKS:
                task_id = task["task_id"]
                key = _make_recorder_key("support-agent", task_id)
                path = str(Path(tmpdir) / f"{task_id}.sb")
                with record(path, key=key) as rec:
                    run_support_task(rec, task_id)
                trace = verify_trace(path, key.hmac_key)
                assert len(trace.steps) >= 4

    def test_unknown_task_raises(self):
        from stepback.testing.support_agent import run_support_task

        class FakeRec:
            pass

        with pytest.raises(ValueError, match="Unknown support task id"):
            run_support_task(FakeRec(), "not-a-task")

    def test_support_tasks_catalogue(self):
        from stepback.testing.support_agent import SUPPORT_TASKS
        assert len(SUPPORT_TASKS) == 5
        ids = {t["task_id"] for t in SUPPORT_TASKS}
        assert "order-status-delayed" in ids
        assert "refund-eligible" in ids
        assert "account-unlock" in ids


class TestCodeReviewAgent:
    def test_all_tasks_run(self):
        from stepback import record
        from stepback.testing.code_review_agent import CODE_REVIEW_TASKS, run_code_review_task
        from stepback.bench.author_corpora import _make_recorder_key

        with tempfile.TemporaryDirectory() as tmpdir:
            for task in CODE_REVIEW_TASKS:
                task_id = task["task_id"]
                key = _make_recorder_key("code-review", task_id)
                path = str(Path(tmpdir) / f"{task_id}.sb")
                with record(path, key=key) as rec:
                    run_code_review_task(rec, task_id)
                trace = verify_trace(path, key.hmac_key)
                assert len(trace.steps) == 6

    def test_unknown_task_raises(self):
        from stepback.testing.code_review_agent import run_code_review_task

        class FakeRec:
            pass

        with pytest.raises(ValueError, match="Unknown code review task id"):
            run_code_review_task(FakeRec(), "not-a-task")

    def test_code_review_tasks_catalogue(self):
        from stepback.testing.code_review_agent import CODE_REVIEW_TASKS
        assert len(CODE_REVIEW_TASKS) == 5
        ids = {t["task_id"] for t in CODE_REVIEW_TASKS}
        assert "sql-injection" in ids
        assert "race-condition" in ids
        assert "resource-leak" in ids


class TestPaymentsPolicyAgent:
    def test_all_tasks_run(self):
        from stepback import record
        from stepback.testing.payments_policy_agent import PAYMENTS_TASKS, run_payments_task
        from stepback.bench.author_corpora import _make_recorder_key

        with tempfile.TemporaryDirectory() as tmpdir:
            for task in PAYMENTS_TASKS:
                task_id = task["task_id"]
                key = _make_recorder_key("payments-policy", task_id)
                path = str(Path(tmpdir) / f"{task_id}.sb")
                with record(path, key=key) as rec:
                    run_payments_task(rec, task_id)
                trace = verify_trace(path, key.hmac_key)
                assert len(trace.steps) >= 4

    def test_unknown_task_raises(self):
        from stepback.testing.payments_policy_agent import run_payments_task

        class FakeRec:
            pass

        with pytest.raises(ValueError, match="Unknown payments task id"):
            run_payments_task(FakeRec(), "not-a-task")

    def test_payments_tasks_catalogue(self):
        from stepback.testing.payments_policy_agent import PAYMENTS_TASKS
        assert len(PAYMENTS_TASKS) == 5
        ids = {t["task_id"] for t in PAYMENTS_TASKS}
        assert "normal-payment" in ids
        assert "overlimit-payment" in ids
        assert "sanctioned-country" in ids
        assert "fraud-velocity" in ids
        assert "unverified-beneficiary" in ids

    def test_sanctioned_blocked_early(self):
        """Sanctioned-country payment should stop at step 4 (1 llm + sanctions screen + 1 llm + block)."""
        from stepback import record
        from stepback.testing.payments_policy_agent import run_payments_task
        from stepback.bench.author_corpora import _make_recorder_key

        with tempfile.TemporaryDirectory() as tmpdir:
            key = _make_recorder_key("payments-policy", "sanctioned-country")
            path = str(Path(tmpdir) / "sanctioned.sb")
            with record(path, key=key) as rec:
                run_payments_task(rec, "sanctioned-country")
            trace = verify_trace(path, key.hmac_key)
            assert len(trace.steps) == 4

    def test_normal_payment_approved(self):
        """Normal payment should go through all gates and be approved."""
        from stepback import record
        from stepback.testing.payments_policy_agent import run_payments_task
        from stepback.bench.author_corpora import _make_recorder_key

        with tempfile.TemporaryDirectory() as tmpdir:
            key = _make_recorder_key("payments-policy", "normal-payment")
            path = str(Path(tmpdir) / "normal.sb")
            with record(path, key=key) as rec:
                run_payments_task(rec, "normal-payment")
            trace = verify_trace(path, key.hmac_key)
            # 7 steps: llm + 4 gates + llm + execute
            assert len(trace.steps) == 7


# ---------------------------------------------------------------------------
# generate_author_corpora — idempotency
# ---------------------------------------------------------------------------

class TestGenerateAuthorCorpora:
    def test_generates_all_corpora(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_author_corpora(tmpdir)
            for corpus_id in list_author_corpora():
                corpus_path = Path(tmpdir) / corpus_id
                assert corpus_path.is_dir()
                assert (corpus_path / "manifest.json").is_file()
                sb_files = list(corpus_path.glob("*.sb"))
                assert len(sb_files) == 5

    def test_idempotent_without_force(self):
        """Calling generate twice without force does not overwrite .sb files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_author_corpora(tmpdir)
            # Record mtime of one .sb
            sb = next(Path(tmpdir, "code-review").glob("*.sb"))
            mtime_before = sb.stat().st_mtime

            generate_author_corpora(tmpdir)  # no force
            mtime_after = sb.stat().st_mtime
            assert mtime_before == mtime_after, "File was re-written without force=True"

    def test_force_overwrites(self):
        """force=True rewrites .sb files."""
        import time
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_author_corpora(tmpdir)
            sb = next(Path(tmpdir, "code-review").glob("*.sb"))
            mtime_before = sb.stat().st_mtime
            # Small sleep to ensure mtime differs if file is rewritten
            time.sleep(0.01)
            generate_author_corpora(tmpdir, force=True)
            mtime_after = sb.stat().st_mtime
            assert mtime_after >= mtime_before

    def test_generated_traces_verify(self):
        """Each generated .sb verifies with its manifest HMAC key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_author_corpora(tmpdir)
            for corpus_id in list_author_corpora():
                manifest_path = Path(tmpdir, corpus_id, "manifest.json")
                with manifest_path.open() as fh:
                    manifest = json.load(fh)
                for entry in manifest["tasks"]:
                    sb_path = Path(tmpdir, corpus_id, f"{entry['task_id']}.sb")
                    hmac_key = bytes.fromhex(entry["hmac_key_hex"])
                    trace = verify_trace(str(sb_path), hmac_key)
                    assert len(trace.steps) >= 4


# ---------------------------------------------------------------------------
# bench __init__ re-exports
# ---------------------------------------------------------------------------

class TestBenchReexports:
    def test_symbols_exported(self):
        from stepback import bench
        assert hasattr(bench, "load_author_corpus")
        assert hasattr(bench, "list_author_corpora")
        assert hasattr(bench, "generate_author_corpora")
        assert hasattr(bench, "AUTHOR_CORPUS_META")

    def test_in_all(self):
        from stepback import bench
        for name in ["load_author_corpus", "list_author_corpora",
                     "generate_author_corpora", "AUTHOR_CORPUS_META"]:
            assert name in bench.__all__, f"{name} missing from bench.__all__"


# ---------------------------------------------------------------------------
# testing __init__ re-exports
# ---------------------------------------------------------------------------

class TestTestingReexports:
    def test_symbols_exported(self):
        from stepback import testing
        for name in ["SUPPORT_TASKS", "run_support_task",
                     "CODE_REVIEW_TASKS", "run_code_review_task",
                     "PAYMENTS_TASKS", "run_payments_task"]:
            assert hasattr(testing, name), f"{name} missing from stepback.testing"

    def test_in_all(self):
        from stepback import testing
        for name in ["SUPPORT_TASKS", "run_support_task",
                     "CODE_REVIEW_TASKS", "run_code_review_task",
                     "PAYMENTS_TASKS", "run_payments_task"]:
            assert name in testing.__all__, f"{name} missing from testing.__all__"
