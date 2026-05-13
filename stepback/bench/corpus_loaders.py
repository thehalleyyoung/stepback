"""Corpus loaders for external agent benchmarks (Step 115).

These loaders convert locally-downloaded benchmark datasets into stepback's
common :class:`CorpusTask` representation, enabling uniform replay-caching
and minimization benchmarks across standard agent evaluation suites.

No network calls are ever made; users must download the benchmark corpora
themselves and point the loaders at local paths.

Supported corpora
-----------------
``swe-bench-verified``
    `SWE-bench Verified <https://swebench.com>`_ — GitHub issues for software
    engineering agents.  Each task contains a problem statement derived from a
    real GitHub issue; ground-truth patch lives in ``metadata["evaluation"]``.
    Native format: JSONL, one task per line.

``gaia``
    `GAIA <https://gaia-benchmark.com>`_ — general AI assistant tasks at three
    difficulty levels.  Native format: JSON file with a ``"tasks"`` list, or a
    directory containing ``metadata.json`` with an optional ``"files"``
    subdirectory for task attachments.

``tau-bench``
    `tau-bench <https://github.com/sierra-research/tau-bench>`_ — tool-use and
    agent interaction tasks across retail and airline domains.
    Native format: JSONL, one task per line.

``agentbench``
    `AgentBench <https://github.com/THUDM/AgentBench>`_ — multi-domain agent
    evaluation (OS, DB, KG, AlfWorld, etc.).
    Native format: JSONL or JSON list, one task per entry.

``osworld``
    `OSWorld <https://os-world.github.io>`_ — desktop computer control tasks.
    Native format: JSON file with a list of task dicts, or a directory where
    each file is a task JSON.

``webarena``
    `WebArena <https://webarena.dev>`_ — web navigation tasks across multiple
    websites.
    Native format: JSONL or JSON list, one task per entry.

Usage example
-------------
.. code-block:: python

    from stepback.bench.corpus_loaders import SWEBenchLoader, load_corpus

    # Load all tasks
    loader = SWEBenchLoader("swe-bench-verified.jsonl")
    tasks = loader.load_all()

    # Sample 10 deterministically
    sample = loader.sample(10, seed=42)

    # Via dispatch
    tasks = load_corpus("swe-bench-verified", "swe-bench-verified.jsonl")

Ground-truth isolation
----------------------
``CorpusTask.prompt`` contains **only** the human-facing task description —
no answer, patch, solution, or evaluation config.  Ground-truth and
evaluation configuration live in ``metadata["evaluation"]`` so agent
implementations cannot accidentally consume them.

Malformed input handling
------------------------
* JSONL loaders skip malformed or missing-required-field rows and emit a
  :class:`LoadWarning` for each.
* JSON-list loaders raise a :class:`CorpusLoadError` on a whole-file parse
  failure but skip individual items with missing required fields (with a
  :class:`LoadWarning`).
"""
from __future__ import annotations

import json
import random
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Type


# ------------------------------------------------------------------ warnings / errors


class LoadWarning(UserWarning):
    """Emitted when a corpus row is skipped due to a missing field or
    parse error; the loader continues with the remaining rows.

    Catch with ``warnings.catch_warnings()`` or
    ``pytest.warns(LoadWarning)`` in tests.
    """


class CorpusLoadError(ValueError):
    """Raised when the corpus file cannot be parsed at all (e.g. a
    JSON-list file with invalid JSON at the top level).
    """


# ------------------------------------------------------------------ CorpusTask


@dataclass
class CorpusTask:
    """A single task drawn from an external agent benchmark corpus.

    Attributes
    ----------
    corpus_id:
        Stable corpus identifier, e.g. ``"swe-bench-verified"``.
    task_id:
        Identifier unique within the corpus, e.g. ``"django__django-12345"``.
    split:
        Dataset split: ``"train"``, ``"dev"``, ``"test"``, or ``"unknown"``
        when the source file does not embed split information.
    prompt:
        Human-readable task description suitable for use as an LLM prompt.
        **Never contains ground-truth answers, patches, or evaluation labels.**
    inputs:
        Runtime-relevant inputs that a graded agent implementation would need
        beyond the prompt — e.g. available tools, website URLs, file
        attachments.  Keys vary by corpus; absent keys default to empty/None.
    metadata:
        Corpus provenance and **evaluation-only** data.  Always contains:

        * ``"raw"``: the original parsed row dict.
        * ``"source_path"``: absolute path of the file this row came from.
        * ``"line_number"``: 1-based line number in JSONL files, or 0 for
          JSON-list files where the position is an array index.
        * ``"evaluation"``: ground-truth answer / patch / eval config.
          **Do not expose this to the agent under evaluation.**
    """

    corpus_id: str
    task_id: str
    split: str
    prompt: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        prompt_excerpt = self.prompt[:60].replace("\n", " ")
        if len(self.prompt) > 60:
            prompt_excerpt += "…"
        return (
            f"CorpusTask(corpus_id={self.corpus_id!r}, task_id={self.task_id!r},"
            f" split={self.split!r}, prompt={prompt_excerpt!r})"
        )


# ------------------------------------------------------------------ base loader


class CorpusLoader(ABC):
    """Abstract base class for all corpus loaders.

    Subclass and implement :meth:`_iter_tasks` to add a new corpus.
    The constructor takes the local path; iteration is lazy so callers can
    stream large corpora without loading the whole dataset into memory.

    Parameters
    ----------
    path:
        Local path to the corpus file or directory.  The loader never makes
        network calls; the corpus must already be present on disk.
    split:
        Override the split label used for every task.  When ``None`` (the
        default) the loader derives the split from the file name or embedded
        field where possible, falling back to ``"unknown"``.
    """

    #: Stable corpus identifier.  Set by each concrete subclass.
    corpus_id: str = ""

    def __init__(self, path: str | Path, *, split: Optional[str] = None) -> None:
        self.path = Path(path)
        self._split_override = split

    # ------------------------------------------------------------------
    # public API

    def load(self) -> Iterator[CorpusTask]:
        """Yield tasks lazily from the corpus path.

        Skips malformed rows and emits a :class:`LoadWarning` for each.
        """
        yield from self._iter_tasks()

    def load_all(self) -> List[CorpusTask]:
        """Return all tasks as a list."""
        return list(self._iter_tasks())

    def sample(self, n: int, seed: int = 0) -> List[CorpusTask]:
        """Return a deterministic sample of *n* tasks.

        Parameters
        ----------
        n:
            Number of tasks to return.  If *n* ≥ len(tasks) the full list
            is returned without repetition (no over-sampling).
        seed:
            Random seed; the same seed always produces the same sample order
            from the same corpus file.
        """
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n!r}")
        all_tasks = self.load_all()
        if n >= len(all_tasks):
            return all_tasks
        rng = random.Random(seed)
        return rng.sample(all_tasks, n)

    # ------------------------------------------------------------------
    # helpers for subclasses

    def _resolve_split(self, embedded: Optional[str] = None) -> str:
        """Return the effective split label."""
        if self._split_override is not None:
            return self._split_override
        if embedded:
            return embedded
        # Try to derive from file name stem
        stem = self.path.stem.lower()
        for candidate in ("train", "dev", "val", "validation", "test"):
            if candidate in stem:
                return candidate if candidate not in ("val", "validation") else "dev"
        return "unknown"

    @staticmethod
    def _warn(msg: str) -> None:
        warnings.warn(msg, LoadWarning, stacklevel=3)

    # ------------------------------------------------------------------
    # abstract

    @abstractmethod
    def _iter_tasks(self) -> Iterator[CorpusTask]:
        """Yield :class:`CorpusTask` objects from the corpus."""


# ------------------------------------------------------------------ JSONL helper


def _iter_jsonl(path: Path) -> Iterator[Tuple[int, dict]]:
    """Yield ``(line_number, parsed_row)`` pairs from a JSONL file."""
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.warn(
                    f"{path}:{lineno}: JSON parse error — {exc}; row skipped.",
                    LoadWarning,
                    stacklevel=5,
                )
                continue
            if not isinstance(row, dict):
                warnings.warn(
                    f"{path}:{lineno}: expected JSON object, got {type(row).__name__}; "
                    "row skipped.",
                    LoadWarning,
                    stacklevel=5,
                )
                continue
            yield lineno, row


def _load_json_list(path: Path) -> List[dict]:
    """Load a JSON file that must be a list of objects."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise CorpusLoadError(f"Cannot parse JSON list from {path}: {exc}") from exc
    if not isinstance(data, list):
        raise CorpusLoadError(
            f"Expected a JSON list in {path}, got {type(data).__name__}"
        )
    return data


# ------------------------------------------------------------------ SWE-bench


class SWEBenchLoader(CorpusLoader):
    """Loader for the SWE-bench-Verified corpus.

    Native format: JSONL where each row is a task dict.

    Required fields per row:
    * ``instance_id`` (str)
    * ``problem_statement`` (str)

    Optional fields placed in ``inputs``:
    * ``repo`` (str)
    * ``version`` (str)
    * ``hints_text`` (str)
    * ``base_commit`` (str)
    * ``environment_setup_commit`` (str)
    * ``PASS_TO_PASS``, ``FAIL_TO_PASS`` (list[str])

    Ground truth (in ``metadata["evaluation"]``):
    * ``patch`` (str)
    * ``test_patch`` (str)
    """

    corpus_id = "swe-bench-verified"

    def _iter_tasks(self) -> Iterator[CorpusTask]:
        path = self.path
        if path.is_dir():
            # Accept a directory — try common file name patterns
            candidates = list(path.glob("*.jsonl")) + list(path.glob("*.json"))
            if not candidates:
                raise CorpusLoadError(f"No JSONL/JSON files found in {path}")
            path = candidates[0]

        split = self._resolve_split()

        for lineno, row in _iter_jsonl(path):
            # Validate required fields
            missing = [f for f in ("instance_id", "problem_statement") if f not in row]
            if missing:
                self._warn(
                    f"{path}:{lineno}: SWE-bench row missing required fields "
                    f"{missing}; row skipped."
                )
                continue

            task_id = str(row["instance_id"])
            prompt = str(row["problem_statement"])

            inputs: Dict[str, Any] = {}
            for key in (
                "repo",
                "version",
                "hints_text",
                "base_commit",
                "environment_setup_commit",
                "PASS_TO_PASS",
                "FAIL_TO_PASS",
            ):
                if key in row:
                    inputs[key] = row[key]

            evaluation: Dict[str, Any] = {}
            for key in ("patch", "test_patch"):
                if key in row:
                    evaluation[key] = row[key]

            yield CorpusTask(
                corpus_id=self.corpus_id,
                task_id=task_id,
                split=split,
                prompt=prompt,
                inputs=inputs,
                metadata={
                    "raw": row,
                    "source_path": str(path.resolve()),
                    "line_number": lineno,
                    "evaluation": evaluation,
                },
            )


# ------------------------------------------------------------------ GAIA


class GAIALoader(CorpusLoader):
    """Loader for the GAIA benchmark corpus.

    Accepts either:

    * A single JSON file with a ``"tasks"`` list (or a bare JSON list).
    * A directory containing ``metadata.json`` (with a ``"tasks"`` key) and
      an optional ``files/`` subdirectory for task attachments.

    Required fields per task:
    * ``task_id`` (str)
    * ``question`` (str)

    Optional fields placed in ``inputs``:
    * ``level`` (int)
    * ``file_name`` (str) — attachment file name; resolved relative to the
      corpus ``files/`` directory if it exists.

    Ground truth (in ``metadata["evaluation"]``):
    * ``final_answer`` (str)
    * ``annotator_metadata`` (dict)
    """

    corpus_id = "gaia"

    def _iter_tasks(self) -> Iterator[CorpusTask]:
        path = self.path
        files_dir: Optional[Path] = None

        if path.is_dir():
            # Look for metadata.json inside the directory
            metadata_path = path / "metadata.json"
            if not metadata_path.exists():
                # Try any .json file
                candidates = list(path.glob("*.json"))
                if not candidates:
                    raise CorpusLoadError(f"No JSON file found in GAIA directory {path}")
                metadata_path = candidates[0]
            files_dir = path / "files"
            if not files_dir.is_dir():
                files_dir = None
            path = metadata_path

        split = self._resolve_split()

        # Load the JSON file — may be {"tasks": [...]} or a bare list
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            raise CorpusLoadError(f"Cannot parse GAIA JSON from {path}: {exc}") from exc

        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            rows = data.get("tasks") or data.get("data") or []
            if not isinstance(rows, list):
                raise CorpusLoadError(
                    f"Expected 'tasks' list in GAIA JSON {path}, "
                    f"got {type(rows).__name__}"
                )
        else:
            raise CorpusLoadError(
                f"Unexpected top-level type in GAIA JSON {path}: {type(data).__name__}"
            )

        source_path = str(path.resolve())
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                self._warn(
                    f"{path}[{idx}]: expected dict, got {type(row).__name__}; skipped."
                )
                continue
            missing = [f for f in ("task_id", "question") if f not in row]
            if missing:
                self._warn(
                    f"{path}[{idx}]: GAIA task missing required fields {missing}; skipped."
                )
                continue

            task_id = str(row["task_id"])
            prompt = str(row["question"])

            inputs: Dict[str, Any] = {}
            if "level" in row:
                inputs["level"] = row["level"]
            file_name = row.get("file_name") or ""
            if file_name:
                inputs["file_name"] = file_name
                if files_dir is not None:
                    candidate = files_dir / file_name
                    inputs["file_path"] = str(candidate) if candidate.exists() else None

            evaluation: Dict[str, Any] = {}
            for key in ("final_answer", "annotator_metadata"):
                if key in row:
                    evaluation[key] = row[key]

            yield CorpusTask(
                corpus_id=self.corpus_id,
                task_id=task_id,
                split=split,
                prompt=prompt,
                inputs=inputs,
                metadata={
                    "raw": row,
                    "source_path": source_path,
                    "line_number": idx,
                    "evaluation": evaluation,
                },
            )


# ------------------------------------------------------------------ tau-bench


class TauBenchLoader(CorpusLoader):
    """Loader for the tau-bench corpus.

    Native format: JSONL where each row is a task dict.

    Required fields per row:
    * ``task_id`` (str or int, coerced to str)

    At least one of:
    * ``instructions`` (str) — agent instructions
    * ``instruction`` (str) — alternate single-instruction key

    Optional fields placed in ``inputs``:
    * ``domain`` (str) — e.g. ``"retail"``, ``"airline"``
    * ``tools`` (list) — tool specs available to the agent
    * ``user_instructions`` (str)

    Ground truth (in ``metadata["evaluation"]``):
    * ``solution`` (list or dict)
    * ``expected_result`` (any)
    * ``reward_threshold`` (float)
    """

    corpus_id = "tau-bench"

    def _iter_tasks(self) -> Iterator[CorpusTask]:
        path = self.path
        if path.is_dir():
            candidates = list(path.glob("*.jsonl")) + list(path.glob("*.json"))
            if not candidates:
                raise CorpusLoadError(f"No JSONL/JSON files found in {path}")
            path = candidates[0]

        split = self._resolve_split()

        for lineno, row in _iter_jsonl(path):
            if "task_id" not in row:
                self._warn(
                    f"{path}:{lineno}: tau-bench row missing 'task_id'; skipped."
                )
                continue

            task_id = str(row["task_id"])
            prompt_text = (
                row.get("instructions")
                or row.get("instruction")
                or row.get("user_instructions")
                or ""
            )
            if not prompt_text:
                self._warn(
                    f"{path}:{lineno}: tau-bench row {task_id!r} has no instructions; skipped."
                )
                continue

            inputs: Dict[str, Any] = {}
            for key in ("domain", "tools", "user_instructions"):
                if key in row:
                    inputs[key] = row[key]

            evaluation: Dict[str, Any] = {}
            for key in ("solution", "expected_result", "reward_threshold", "actions"):
                if key in row:
                    evaluation[key] = row[key]

            yield CorpusTask(
                corpus_id=self.corpus_id,
                task_id=task_id,
                split=split,
                prompt=str(prompt_text),
                inputs=inputs,
                metadata={
                    "raw": row,
                    "source_path": str(path.resolve()),
                    "line_number": lineno,
                    "evaluation": evaluation,
                },
            )


# ------------------------------------------------------------------ AgentBench


class AgentBenchLoader(CorpusLoader):
    """Loader for the AgentBench corpus.

    Accepts either:

    * JSONL file — one task object per line.
    * JSON file — either a list or a dict mapping type → task list.

    Required fields per task:
    * ``id`` (str or int, coerced to str)

    At least one of:
    * ``content`` (str) — task description / instruction
    * ``instruction`` (str) — alternate instruction key
    * ``description`` (str) — alternate description key

    Optional fields placed in ``inputs``:
    * ``type`` (str) — task domain: ``"os"``, ``"db"``, ``"kg"``, ``"alfworld"``, etc.
    * ``initial_config`` (dict)
    * ``start_url`` (str)

    Ground truth (in ``metadata["evaluation"]``):
    * ``answer`` (any)
    * ``evaluation`` (dict)
    * ``result`` (any)
    """

    corpus_id = "agentbench"

    def _iter_tasks(self) -> Iterator[CorpusTask]:
        path = self.path
        if path.is_dir():
            candidates = list(path.glob("*.jsonl")) + list(path.glob("*.json"))
            if not candidates:
                raise CorpusLoadError(f"No JSONL/JSON files found in {path}")
            path = candidates[0]

        split = self._resolve_split()

        # Try JSONL first; fall back to JSON list/dict
        rows: List[Tuple[int, dict]] = []
        if path.suffix.lower() == ".jsonl" or _sniff_jsonl(path):
            rows = list(_iter_jsonl(path))
        else:
            data = _load_json_list_or_dict(path)
            rows = [(i, item) for i, item in enumerate(data, start=0)
                    if isinstance(item, dict)]
            non_dict = sum(1 for item in data if not isinstance(item, dict))
            if non_dict:
                warnings.warn(
                    f"{path}: {non_dict} non-object items skipped.",
                    LoadWarning,
                    stacklevel=2,
                )

        source_path = str(path.resolve())
        for lineno, row in rows:
            if "id" not in row:
                self._warn(
                    f"{path}:{lineno}: AgentBench row missing 'id'; skipped."
                )
                continue

            task_id = str(row["id"])
            content = (
                row.get("content")
                or row.get("instruction")
                or row.get("description")
                or ""
            )
            if not content:
                self._warn(
                    f"{path}:{lineno}: AgentBench task {task_id!r} has no content; skipped."
                )
                continue

            inputs: Dict[str, Any] = {}
            for key in ("type", "initial_config", "start_url"):
                if key in row:
                    inputs[key] = row[key]

            evaluation: Dict[str, Any] = {}
            for key in ("answer", "evaluation", "result"):
                if key in row:
                    evaluation[key] = row[key]

            yield CorpusTask(
                corpus_id=self.corpus_id,
                task_id=task_id,
                split=split,
                prompt=str(content),
                inputs=inputs,
                metadata={
                    "raw": row,
                    "source_path": source_path,
                    "line_number": lineno,
                    "evaluation": evaluation,
                },
            )


# ------------------------------------------------------------------ OSWorld


class OSWorldLoader(CorpusLoader):
    """Loader for the OSWorld corpus.

    Accepts either:

    * A single JSON file — a list of task dicts.
    * A directory — each ``*.json`` file is one task.

    Required fields per task:
    * ``id`` (str or int, coerced to str)
    * ``instruction`` (str) — desktop task description

    Optional fields placed in ``inputs``:
    * ``snapshot`` (str) — VM snapshot name
    * ``config`` (list[dict]) — setup action sequence
    * ``domain`` (str) — task domain

    Ground truth (in ``metadata["evaluation"]``):
    * ``result`` (dict) — evaluation configuration (commands, expected values, etc.)
    * ``evaluator`` (dict)
    """

    corpus_id = "osworld"

    def _iter_tasks(self) -> Iterator[CorpusTask]:
        path = self.path
        split = self._resolve_split()

        if path.is_dir():
            json_files = sorted(path.glob("*.json"))
            if not json_files:
                raise CorpusLoadError(f"No JSON task files found in OSWorld directory {path}")
            for task_file in json_files:
                yield from self._load_file(task_file, split)
        else:
            yield from self._load_file(path, split)

    def _load_file(self, path: Path, split: str) -> Iterator[CorpusTask]:
        """Load tasks from a single JSON file (list or single object)."""
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            raise CorpusLoadError(f"Cannot parse OSWorld JSON from {path}: {exc}") from exc

        if isinstance(data, dict):
            # Single task file
            rows = [data]
        elif isinstance(data, list):
            rows = data
        else:
            raise CorpusLoadError(
                f"Unexpected top-level type in OSWorld JSON {path}: {type(data).__name__}"
            )

        source_path = str(path.resolve())
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                self._warn(
                    f"{path}[{idx}]: expected dict, got {type(row).__name__}; skipped."
                )
                continue
            missing = [f for f in ("id", "instruction") if f not in row]
            if missing:
                self._warn(
                    f"{path}[{idx}]: OSWorld task missing required fields {missing}; skipped."
                )
                continue

            task_id = str(row["id"])
            prompt = str(row["instruction"])

            inputs: Dict[str, Any] = {}
            for key in ("snapshot", "config", "domain"):
                if key in row:
                    inputs[key] = row[key]

            evaluation: Dict[str, Any] = {}
            for key in ("result", "evaluator"):
                if key in row:
                    evaluation[key] = row[key]

            yield CorpusTask(
                corpus_id=self.corpus_id,
                task_id=task_id,
                split=split,
                prompt=prompt,
                inputs=inputs,
                metadata={
                    "raw": row,
                    "source_path": source_path,
                    "line_number": idx,
                    "evaluation": evaluation,
                },
            )


# ------------------------------------------------------------------ WebArena


class WebArenaLoader(CorpusLoader):
    """Loader for the WebArena corpus.

    Accepts either JSONL or a JSON list file.

    Required fields per task:
    * ``task_id`` (str or int, coerced to str)
    * ``intent`` (str) — task instruction for the web-navigation agent

    Optional fields placed in ``inputs``:
    * ``sites`` (list[str]) — websites the task involves
    * ``start_url`` (str)
    * ``require_login`` (bool)
    * ``storage_state`` (str)

    Ground truth (in ``metadata["evaluation"]``):
    * ``eval`` (dict) — evaluation specification
    * ``reference_answers`` (dict)
    """

    corpus_id = "webarena"

    def _iter_tasks(self) -> Iterator[CorpusTask]:
        path = self.path
        if path.is_dir():
            candidates = list(path.glob("*.jsonl")) + list(path.glob("*.json"))
            if not candidates:
                raise CorpusLoadError(f"No JSONL/JSON files found in {path}")
            path = candidates[0]

        split = self._resolve_split()
        source_path = str(path.resolve())

        if path.suffix.lower() == ".jsonl" or _sniff_jsonl(path):
            row_iter = _iter_jsonl(path)
        else:
            data = _load_json_list(path)
            row_iter = (
                (i, item)
                for i, item in enumerate(data, start=0)
                if isinstance(item, dict)
            )

        for lineno, row in row_iter:
            missing = [f for f in ("task_id", "intent") if f not in row]
            if missing:
                self._warn(
                    f"{path}:{lineno}: WebArena row missing required fields {missing}; skipped."
                )
                continue

            task_id = str(row["task_id"])
            prompt = str(row["intent"])

            inputs: Dict[str, Any] = {}
            for key in ("sites", "start_url", "require_login", "storage_state"):
                if key in row:
                    inputs[key] = row[key]

            eval_block = row.get("eval") or {}
            evaluation: Dict[str, Any] = {"eval": eval_block}
            ref = eval_block.get("reference_answers") or row.get("reference_answers")
            if ref is not None:
                evaluation["reference_answers"] = ref

            yield CorpusTask(
                corpus_id=self.corpus_id,
                task_id=task_id,
                split=split,
                prompt=prompt,
                inputs=inputs,
                metadata={
                    "raw": row,
                    "source_path": source_path,
                    "line_number": lineno,
                    "evaluation": evaluation,
                },
            )


# ------------------------------------------------------------------ helpers


def _sniff_jsonl(path: Path) -> bool:
    """Return True if *path* looks like JSONL (first non-empty line is a JSON object)."""
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    return line.startswith("{")
    except OSError:
        pass
    return False


def _load_json_list_or_dict(path: Path) -> List[dict]:
    """Load a JSON file that may be a list or a dict (returns items)."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise CorpusLoadError(f"Cannot parse JSON from {path}: {exc}") from exc
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Maybe it's a type→task dict
        result: List[dict] = []
        for v in data.values():
            if isinstance(v, list):
                result.extend(item for item in v if isinstance(item, dict))
            elif isinstance(v, dict):
                result.append(v)
        return result
    raise CorpusLoadError(
        f"Expected JSON list or dict in {path}, got {type(data).__name__}"
    )


# ------------------------------------------------------------------ registry


#: Registry mapping stable corpus ID → loader class.
_LOADERS: Dict[str, Type[CorpusLoader]] = {
    "swe-bench-verified": SWEBenchLoader,
    "gaia": GAIALoader,
    "tau-bench": TauBenchLoader,
    "agentbench": AgentBenchLoader,
    "osworld": OSWorldLoader,
    "webarena": WebArenaLoader,
}


def list_loaders() -> List[str]:
    """Return the list of stable corpus IDs supported by this module."""
    return sorted(_LOADERS.keys())


def load_corpus(corpus_id: str, path: str | Path, **kwargs: Any) -> List[CorpusTask]:
    """Load all tasks for *corpus_id* from a local *path*.

    Parameters
    ----------
    corpus_id:
        One of the IDs returned by :func:`list_loaders`.
    path:
        Local file or directory.
    **kwargs:
        Forwarded to the loader constructor (e.g. ``split="test"``).

    Raises
    ------
    KeyError
        If *corpus_id* is not known.
    CorpusLoadError
        If the file cannot be parsed.
    """
    try:
        loader_cls = _LOADERS[corpus_id]
    except KeyError:
        raise KeyError(
            f"Unknown corpus_id {corpus_id!r}. Known: {list_loaders()}"
        ) from None
    return loader_cls(path, **kwargs).load_all()


__all__ = [
    "AgentBenchLoader",
    "CorpusLoadError",
    "CorpusLoader",
    "CorpusTask",
    "GAIALoader",
    "LoadWarning",
    "OSWorldLoader",
    "SWEBenchLoader",
    "TauBenchLoader",
    "WebArenaLoader",
    "list_loaders",
    "load_corpus",
]
