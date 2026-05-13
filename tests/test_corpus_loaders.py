"""Tests for stepback.bench.corpus_loaders (Step 115).

All fixtures are synthetic in-memory data written to tmp_path — no
network calls, no real benchmark datasets required.

Coverage targets:
* All six loaders: SWEBenchLoader, GAIALoader, TauBenchLoader,
  AgentBenchLoader, OSWorldLoader, WebArenaLoader
* load_all() returns the right count and fields
* sample() is deterministic and handles n >= len(tasks) and n == 0
* load_corpus() dispatch works for all six corpus ids
* list_loaders() returns all six ids
* Ground-truth isolation: prompt never contains patch/answer/solution
* LoadWarning emitted for missing-required-field rows (not raised)
* CorpusLoadError raised on whole-file parse failure (JSON-list formats)
* Directory-based loading for GAIA and OSWorld
* metadata["evaluation"] contains ground-truth fields
* metadata["raw"], metadata["source_path"], metadata["line_number"] present
* CorpusTask repr and __repr__ work
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import List

import pytest

from stepback.bench.corpus_loaders import (
    AgentBenchLoader,
    CorpusLoadError,
    CorpusLoader,
    CorpusTask,
    GAIALoader,
    LoadWarning,
    OSWorldLoader,
    SWEBenchLoader,
    TauBenchLoader,
    WebArenaLoader,
    list_loaders,
    load_corpus,
)


# ------------------------------------------------------------------ fixtures helpers


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _swe_bench_rows(n: int = 4) -> str:
    rows = []
    for i in range(n):
        rows.append(json.dumps({
            "instance_id": f"django__django-{1000 + i}",
            "problem_statement": f"Fix bug {i}: the widget crashes on empty input.",
            "repo": "django/django",
            "version": f"3.{i}",
            "hints_text": f"Hint {i}: check the validator.",
            "base_commit": f"abc{i:04x}",
            "patch": f"diff --git a/widget.py b/widget.py\n+fix_{i}",
            "test_patch": f"diff --git a/tests/test_widget.py b/tests/test_widget.py\n+assert_{i}",
        }))
    return "\n".join(rows)


def _gaia_rows(n: int = 3) -> list:
    rows = []
    for i in range(n):
        rows.append({
            "task_id": str(i),
            "question": f"What is the capital of Country {i}?",
            "level": (i % 3) + 1,
            "final_answer": f"Capital{i}",
            "annotator_metadata": {"steps": i + 1},
            "file_name": f"attachment_{i}.pdf" if i == 1 else "",
        })
    return rows


def _tau_bench_rows(n: int = 4) -> str:
    rows = []
    for i in range(n):
        rows.append(json.dumps({
            "task_id": str(i),
            "domain": "retail" if i % 2 == 0 else "airline",
            "instructions": f"Help the customer with order #{i * 100}.",
            "tools": [{"name": "lookup_order", "description": "Lookup order by id"}],
            "solution": [{"action": "lookup_order", "args": {"id": i * 100}}],
            "reward_threshold": 0.9,
        }))
    return "\n".join(rows)


def _agentbench_rows(n: int = 4) -> str:
    rows = []
    for i in range(n):
        rows.append(json.dumps({
            "id": str(i),
            "content": f"List the files in directory /tmp/{i}.",
            "type": ["os", "db", "kg", "alfworld"][i % 4],
            "answer": f"file_{i}.txt",
        }))
    return "\n".join(rows)


def _osworld_rows(n: int = 3) -> list:
    rows = []
    for i in range(n):
        rows.append({
            "id": f"task_{i}",
            "instruction": f"Open the file manager and navigate to /home/user/Documents/Task{i}.",
            "snapshot": f"snapshot_{i}",
            "domain": "file_manager",
            "result": {"type": "vm_command_line", "command": f"ls /home/user/Documents/Task{i}"},
        })
    return rows


def _webarena_rows(n: int = 4) -> str:
    rows = []
    for i in range(n):
        rows.append(json.dumps({
            "task_id": i,
            "intent": f"Find the cheapest {['laptop', 'phone', 'tablet', 'monitor'][i % 4]} under $500.",
            "sites": ["shopping"],
            "start_url": f"http://localhost:7770/task/{i}",
            "require_login": True,
            "eval": {
                "eval_types": ["string_match"],
                "reference_answers": {"exact_match": f"product_{i}"},
            },
        }))
    return "\n".join(rows)


# ====================================================================
# SWEBenchLoader
# ====================================================================


class TestSWEBenchLoader:
    def test_load_all_returns_all_tasks(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _swe_bench_rows(4))
        tasks = SWEBenchLoader(f).load_all()
        assert len(tasks) == 4

    def test_task_fields_populated(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _swe_bench_rows(1))
        task = SWEBenchLoader(f).load_all()[0]
        assert task.corpus_id == "swe-bench-verified"
        assert task.task_id == "django__django-1000"
        assert "bug 0" in task.prompt
        assert task.inputs["repo"] == "django/django"
        assert task.inputs["version"] == "3.0"

    def test_prompt_does_not_contain_patch(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _swe_bench_rows(4))
        for task in SWEBenchLoader(f).load_all():
            assert "diff --git" not in task.prompt
            assert "patch" not in task.prompt.lower() or "patch" in task.task_id

    def test_evaluation_contains_patch(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _swe_bench_rows(1))
        task = SWEBenchLoader(f).load_all()[0]
        assert "patch" in task.metadata["evaluation"]
        assert "diff --git" in task.metadata["evaluation"]["patch"]

    def test_metadata_provenance(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _swe_bench_rows(1))
        task = SWEBenchLoader(f).load_all()[0]
        assert "raw" in task.metadata
        assert "source_path" in task.metadata
        assert "line_number" in task.metadata
        assert task.metadata["line_number"] == 1

    def test_missing_required_field_emits_warning(self, tmp_path):
        bad_row = json.dumps({"problem_statement": "Fix me"})  # no instance_id
        f = _write(tmp_path, "tasks.jsonl", bad_row)
        with pytest.warns(LoadWarning):
            tasks = SWEBenchLoader(f).load_all()
        assert tasks == []

    def test_malformed_json_row_emits_warning(self, tmp_path):
        rows = _swe_bench_rows(1) + "\nnot-json\n" + _swe_bench_rows(1).replace("1000", "1001")
        f = _write(tmp_path, "tasks.jsonl", rows)
        with pytest.warns(LoadWarning):
            tasks = SWEBenchLoader(f).load_all()
        assert len(tasks) == 2  # two valid rows survive

    def test_directory_loading(self, tmp_path):
        corpus_dir = tmp_path / "swe_corpus"
        corpus_dir.mkdir()
        _write(corpus_dir, "test.jsonl", _swe_bench_rows(3))
        tasks = SWEBenchLoader(corpus_dir).load_all()
        assert len(tasks) == 3

    def test_split_from_file_name(self, tmp_path):
        f = _write(tmp_path, "train.jsonl", _swe_bench_rows(2))
        tasks = SWEBenchLoader(f).load_all()
        assert all(t.split == "train" for t in tasks)

    def test_split_override(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _swe_bench_rows(2))
        tasks = SWEBenchLoader(f, split="test").load_all()
        assert all(t.split == "test" for t in tasks)


# ====================================================================
# GAIALoader
# ====================================================================


class TestGAIALoader:
    def _file(self, tmp_path: Path, rows=None) -> Path:
        data = rows or _gaia_rows(3)
        f = _write(tmp_path, "metadata.json", json.dumps({"tasks": data}))
        return f

    def test_load_all_from_tasks_dict(self, tmp_path):
        tasks = GAIALoader(self._file(tmp_path)).load_all()
        assert len(tasks) == 3

    def test_load_all_from_bare_list(self, tmp_path):
        f = _write(tmp_path, "tasks.json", json.dumps(_gaia_rows(3)))
        tasks = GAIALoader(f).load_all()
        assert len(tasks) == 3

    def test_task_fields_populated(self, tmp_path):
        task = GAIALoader(self._file(tmp_path)).load_all()[0]
        assert task.corpus_id == "gaia"
        assert task.task_id == "0"
        assert "capital" in task.prompt.lower()
        assert task.inputs.get("level") == 1

    def test_prompt_does_not_contain_answer(self, tmp_path):
        for task in GAIALoader(self._file(tmp_path)).load_all():
            assert "Capital" not in task.prompt

    def test_evaluation_contains_answer(self, tmp_path):
        task = GAIALoader(self._file(tmp_path)).load_all()[0]
        assert task.metadata["evaluation"]["final_answer"] == "Capital0"

    def test_metadata_provenance(self, tmp_path):
        task = GAIALoader(self._file(tmp_path)).load_all()[0]
        assert "raw" in task.metadata
        assert "source_path" in task.metadata
        assert "line_number" in task.metadata

    def test_directory_loading(self, tmp_path):
        corpus_dir = tmp_path / "gaia_corpus"
        corpus_dir.mkdir()
        _write(corpus_dir, "metadata.json", json.dumps({"tasks": _gaia_rows(4)}))
        tasks = GAIALoader(corpus_dir).load_all()
        assert len(tasks) == 4

    def test_directory_with_files_subdir(self, tmp_path):
        corpus_dir = tmp_path / "gaia_corpus"
        corpus_dir.mkdir()
        files_dir = corpus_dir / "files"
        files_dir.mkdir()
        # Create a real attachment file
        (files_dir / "attachment_1.pdf").write_bytes(b"%PDF-1.4 fake")
        _write(corpus_dir, "metadata.json", json.dumps({"tasks": _gaia_rows(2)}))
        tasks = GAIALoader(corpus_dir).load_all()
        # Task index 1 has file_name set; file_path should resolve
        task1 = tasks[1]
        assert task1.inputs.get("file_name") == "attachment_1.pdf"
        assert task1.inputs.get("file_path") is not None

    def test_missing_required_field_emits_warning(self, tmp_path):
        bad = [{"question": "What?"}]  # no task_id
        f = _write(tmp_path, "tasks.json", json.dumps(bad))
        with pytest.warns(LoadWarning):
            tasks = GAIALoader(f).load_all()
        assert tasks == []

    def test_invalid_json_raises(self, tmp_path):
        f = _write(tmp_path, "tasks.json", "NOT JSON")
        with pytest.raises(CorpusLoadError):
            GAIALoader(f).load_all()


# ====================================================================
# TauBenchLoader
# ====================================================================


class TestTauBenchLoader:
    def test_load_all(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _tau_bench_rows(4))
        tasks = TauBenchLoader(f).load_all()
        assert len(tasks) == 4

    def test_task_fields(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _tau_bench_rows(1))
        task = TauBenchLoader(f).load_all()[0]
        assert task.corpus_id == "tau-bench"
        assert task.task_id == "0"
        assert "customer" in task.prompt.lower()
        assert task.inputs["domain"] == "retail"
        assert "tools" in task.inputs

    def test_prompt_does_not_contain_solution(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _tau_bench_rows(4))
        for task in TauBenchLoader(f).load_all():
            assert "lookup_order" not in task.prompt

    def test_evaluation_contains_solution(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _tau_bench_rows(1))
        task = TauBenchLoader(f).load_all()[0]
        assert "solution" in task.metadata["evaluation"]
        assert task.metadata["evaluation"]["reward_threshold"] == 0.9

    def test_missing_task_id_emits_warning(self, tmp_path):
        bad = json.dumps({"instructions": "Help me.", "domain": "retail"})
        f = _write(tmp_path, "tasks.jsonl", bad)
        with pytest.warns(LoadWarning):
            tasks = TauBenchLoader(f).load_all()
        assert tasks == []

    def test_no_instructions_emits_warning(self, tmp_path):
        bad = json.dumps({"task_id": "0", "domain": "retail"})
        f = _write(tmp_path, "tasks.jsonl", bad)
        with pytest.warns(LoadWarning):
            tasks = TauBenchLoader(f).load_all()
        assert tasks == []

    def test_alternate_instruction_key(self, tmp_path):
        row = json.dumps({"task_id": "0", "instruction": "Book a flight.", "domain": "airline"})
        f = _write(tmp_path, "tasks.jsonl", row)
        tasks = TauBenchLoader(f).load_all()
        assert len(tasks) == 1
        assert "flight" in tasks[0].prompt


# ====================================================================
# AgentBenchLoader
# ====================================================================


class TestAgentBenchLoader:
    def test_load_all_jsonl(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _agentbench_rows(4))
        tasks = AgentBenchLoader(f).load_all()
        assert len(tasks) == 4

    def test_load_all_json_list(self, tmp_path):
        rows = [json.loads(r) for r in _agentbench_rows(4).splitlines()]
        f = _write(tmp_path, "tasks.json", json.dumps(rows))
        tasks = AgentBenchLoader(f).load_all()
        assert len(tasks) == 4

    def test_task_fields(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _agentbench_rows(1))
        task = AgentBenchLoader(f).load_all()[0]
        assert task.corpus_id == "agentbench"
        assert task.task_id == "0"
        assert "/tmp/0" in task.prompt
        assert task.inputs["type"] == "os"

    def test_prompt_does_not_contain_answer(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _agentbench_rows(4))
        for task in AgentBenchLoader(f).load_all():
            # 'answer' field should be in evaluation, not prompt
            assert task.metadata["evaluation"].get("answer") is not None
            assert task.metadata["evaluation"]["answer"] not in task.prompt

    def test_missing_id_emits_warning(self, tmp_path):
        bad = json.dumps({"content": "Do something."})
        f = _write(tmp_path, "tasks.jsonl", bad)
        with pytest.warns(LoadWarning):
            tasks = AgentBenchLoader(f).load_all()
        assert tasks == []

    def test_alternate_instruction_key(self, tmp_path):
        row = json.dumps({"id": "0", "instruction": "Run ls.", "type": "os"})
        f = _write(tmp_path, "tasks.jsonl", row)
        tasks = AgentBenchLoader(f).load_all()
        assert len(tasks) == 1
        assert "ls" in tasks[0].prompt


# ====================================================================
# OSWorldLoader
# ====================================================================


class TestOSWorldLoader:
    def test_load_all_json_list(self, tmp_path):
        f = _write(tmp_path, "tasks.json", json.dumps(_osworld_rows(3)))
        tasks = OSWorldLoader(f).load_all()
        assert len(tasks) == 3

    def test_load_single_task_json(self, tmp_path):
        row = _osworld_rows(1)[0]
        f = _write(tmp_path, "task_0.json", json.dumps(row))
        tasks = OSWorldLoader(f).load_all()
        assert len(tasks) == 1
        assert tasks[0].task_id == "task_0"

    def test_directory_loading(self, tmp_path):
        corpus_dir = tmp_path / "osworld"
        corpus_dir.mkdir()
        for i, row in enumerate(_osworld_rows(3)):
            _write(corpus_dir, f"task_{i}.json", json.dumps(row))
        tasks = OSWorldLoader(corpus_dir).load_all()
        assert len(tasks) == 3

    def test_task_fields(self, tmp_path):
        f = _write(tmp_path, "tasks.json", json.dumps(_osworld_rows(1)))
        task = OSWorldLoader(f).load_all()[0]
        assert task.corpus_id == "osworld"
        assert task.task_id == "task_0"
        assert "file manager" in task.prompt.lower()
        assert task.inputs["snapshot"] == "snapshot_0"

    def test_prompt_does_not_contain_eval_command(self, tmp_path):
        f = _write(tmp_path, "tasks.json", json.dumps(_osworld_rows(3)))
        for task in OSWorldLoader(f).load_all():
            assert "vm_command_line" not in task.prompt

    def test_evaluation_contains_result(self, tmp_path):
        f = _write(tmp_path, "tasks.json", json.dumps(_osworld_rows(1)))
        task = OSWorldLoader(f).load_all()[0]
        assert "result" in task.metadata["evaluation"]
        assert task.metadata["evaluation"]["result"]["type"] == "vm_command_line"

    def test_missing_required_field_emits_warning(self, tmp_path):
        bad = [{"id": "0"}]  # no instruction
        f = _write(tmp_path, "tasks.json", json.dumps(bad))
        with pytest.warns(LoadWarning):
            tasks = OSWorldLoader(f).load_all()
        assert tasks == []

    def test_invalid_json_raises(self, tmp_path):
        f = _write(tmp_path, "tasks.json", "NOT JSON")
        with pytest.raises(CorpusLoadError):
            OSWorldLoader(f).load_all()


# ====================================================================
# WebArenaLoader
# ====================================================================


class TestWebArenaLoader:
    def test_load_all_jsonl(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _webarena_rows(4))
        tasks = WebArenaLoader(f).load_all()
        assert len(tasks) == 4

    def test_load_all_json_list(self, tmp_path):
        rows = [json.loads(r) for r in _webarena_rows(4).splitlines()]
        f = _write(tmp_path, "tasks.json", json.dumps(rows))
        tasks = WebArenaLoader(f).load_all()
        assert len(tasks) == 4

    def test_task_fields(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _webarena_rows(1))
        task = WebArenaLoader(f).load_all()[0]
        assert task.corpus_id == "webarena"
        assert task.task_id == "0"
        assert "laptop" in task.prompt.lower()
        assert task.inputs["sites"] == ["shopping"]

    def test_prompt_does_not_contain_eval(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _webarena_rows(4))
        for task in WebArenaLoader(f).load_all():
            assert "string_match" not in task.prompt
            assert "reference_answers" not in task.prompt

    def test_evaluation_contains_eval(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _webarena_rows(1))
        task = WebArenaLoader(f).load_all()[0]
        assert "eval" in task.metadata["evaluation"]
        assert "reference_answers" in task.metadata["evaluation"]

    def test_missing_required_field_emits_warning(self, tmp_path):
        bad = json.dumps({"task_id": 0, "sites": ["shopping"]})  # no intent
        f = _write(tmp_path, "tasks.jsonl", bad)
        with pytest.warns(LoadWarning):
            tasks = WebArenaLoader(f).load_all()
        assert tasks == []


# ====================================================================
# Sample tests
# ====================================================================


class TestSampleMethod:
    def test_sample_returns_n_tasks(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _swe_bench_rows(10))
        tasks = SWEBenchLoader(f).sample(5, seed=0)
        assert len(tasks) == 5

    def test_sample_deterministic(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _swe_bench_rows(10))
        loader = SWEBenchLoader(f)
        s1 = [t.task_id for t in loader.sample(5, seed=42)]
        s2 = [t.task_id for t in loader.sample(5, seed=42)]
        assert s1 == s2

    def test_different_seed_different_order(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _swe_bench_rows(10))
        loader = SWEBenchLoader(f)
        s1 = [t.task_id for t in loader.sample(6, seed=0)]
        s2 = [t.task_id for t in loader.sample(6, seed=1)]
        # Different seeds should produce different orderings (with overwhelming probability)
        assert s1 != s2

    def test_sample_n_equals_len(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _swe_bench_rows(5))
        tasks = SWEBenchLoader(f).sample(5, seed=0)
        assert len(tasks) == 5

    def test_sample_n_exceeds_len_returns_all(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _swe_bench_rows(3))
        tasks = SWEBenchLoader(f).sample(100, seed=0)
        assert len(tasks) == 3

    def test_sample_n_zero_returns_empty(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _swe_bench_rows(5))
        tasks = SWEBenchLoader(f).sample(0, seed=0)
        assert tasks == []

    def test_sample_n_negative_raises(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _swe_bench_rows(5))
        with pytest.raises(ValueError, match="n must be"):
            SWEBenchLoader(f).sample(-1)


# ====================================================================
# load_corpus dispatch
# ====================================================================


class TestLoadCorpus:
    def test_dispatch_swe_bench(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _swe_bench_rows(2))
        tasks = load_corpus("swe-bench-verified", f)
        assert len(tasks) == 2
        assert all(t.corpus_id == "swe-bench-verified" for t in tasks)

    def test_dispatch_gaia(self, tmp_path):
        f = _write(tmp_path, "tasks.json", json.dumps({"tasks": _gaia_rows(2)}))
        tasks = load_corpus("gaia", f)
        assert len(tasks) == 2

    def test_dispatch_tau_bench(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _tau_bench_rows(2))
        tasks = load_corpus("tau-bench", f)
        assert len(tasks) == 2

    def test_dispatch_agentbench(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _agentbench_rows(2))
        tasks = load_corpus("agentbench", f)
        assert len(tasks) == 2

    def test_dispatch_osworld(self, tmp_path):
        f = _write(tmp_path, "tasks.json", json.dumps(_osworld_rows(2)))
        tasks = load_corpus("osworld", f)
        assert len(tasks) == 2

    def test_dispatch_webarena(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", _webarena_rows(2))
        tasks = load_corpus("webarena", f)
        assert len(tasks) == 2

    def test_unknown_corpus_raises_key_error(self, tmp_path):
        f = _write(tmp_path, "tasks.jsonl", "")
        with pytest.raises(KeyError, match="Unknown corpus_id"):
            load_corpus("imaginary-benchmark", f)


# ====================================================================
# list_loaders
# ====================================================================


class TestListLoaders:
    def test_returns_all_six(self):
        ids = list_loaders()
        assert set(ids) == {
            "swe-bench-verified",
            "gaia",
            "tau-bench",
            "agentbench",
            "osworld",
            "webarena",
        }

    def test_sorted(self):
        ids = list_loaders()
        assert ids == sorted(ids)


# ====================================================================
# CorpusTask repr and basic properties
# ====================================================================


class TestCorpusTask:
    def test_repr_short(self):
        task = CorpusTask(
            corpus_id="swe-bench-verified",
            task_id="django__django-1234",
            split="test",
            prompt="Fix the bug in the widget module.",
        )
        r = repr(task)
        assert "swe-bench-verified" in r
        assert "django__django-1234" in r

    def test_repr_long_prompt_truncated(self):
        task = CorpusTask(
            corpus_id="gaia",
            task_id="0",
            split="test",
            prompt="x" * 200,
        )
        r = repr(task)
        assert "…" in r

    def test_default_dicts(self):
        task = CorpusTask(corpus_id="gaia", task_id="0", split="test", prompt="Hi")
        assert task.inputs == {}
        assert task.metadata == {}


# ====================================================================
# bench __init__ re-exports
# ====================================================================


def test_bench_init_exports():
    """All corpus loader symbols are accessible from stepback.bench."""
    from stepback.bench import (  # noqa: F401
        AgentBenchLoader,
        CorpusLoadError,
        CorpusLoader,
        CorpusTask,
        GAIALoader,
        LoadWarning,
        OSWorldLoader,
        SWEBenchLoader,
        TauBenchLoader,
        WebArenaLoader,
        list_loaders,
        load_corpus,
    )


def test_load_corpus_callable_via_bench(tmp_path):
    """load_corpus works when imported from stepback.bench."""
    from stepback.bench import load_corpus as lc

    f = _write(tmp_path, "tasks.jsonl", _swe_bench_rows(2))
    tasks = lc("swe-bench-verified", f)
    assert len(tasks) == 2
