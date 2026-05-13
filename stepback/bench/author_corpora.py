"""Author-original benchmark corpora for stepback (Step 116).

Three deterministic, redistributable corpora are bundled with the package
under ``stepback/bench/corpora/``:

``support-agent``
    A customer-support bot handling order inquiries, refunds, account unlocks,
    wrong-item replacements, and subscription cancellations (5 tasks).

``code-review``
    A code-review bot identifying SQL injection, off-by-one errors, race
    conditions, missing error handling, and resource leaks (5 tasks).

``payments-policy``
    A compliance-aware payment processor screening through sanctions, daily
    limits, velocity fraud detection, and KYC (5 tasks, varied outcomes:
    approve, block, hold).

Each corpus ships as a directory containing:

* ``manifest.json`` — task metadata, HMAC keys, and evaluation ground truth.
* ``<task_id>.sb`` — the frozen recorded trace for each task.

Because the keys used to generate the traces are deterministic (derived from a
fixed seed + corpus id + task id), ``verify_trace`` can be called with the
``hmac_key`` stored in ``manifest.json`` to prove trace integrity.

Public API
----------
::

    from stepback.bench.author_corpora import (
        list_author_corpora,
        load_author_corpus,
        generate_author_corpora,
        AUTHOR_CORPUS_META,
    )

    # enumerate corpora
    ids = list_author_corpora()   # ['code-review', 'payments-policy', 'support-agent']

    # load CorpusTasks from bundled .sb files
    tasks = load_author_corpus("support-agent")

    # re-generate .sb files into an output directory (useful for regeneration)
    generate_author_corpora("/tmp/corpora")
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Dict, List, Optional

from stepback.bench.corpus_loaders import CorpusTask

# ---------------------------------------------------------------------------
# Deterministic key derivation
# ---------------------------------------------------------------------------

# A fixed, public, non-secret seed. The derived keys are for test/corpus
# material only; they carry no confidentiality guarantee.
_CORPUS_SEED = b"stepback-author-corpora-v1"


def _derive_hmac_key(corpus_id: str, task_id: str) -> bytes:
    """Return the 32-byte HMAC key for a given corpus task.

    The derivation is deterministic: same inputs always give the same key.
    The keys are non-secret — they are published in ``manifest.json`` so
    that any consumer can verify the bundled ``.sb`` files with
    ``stepback.trace_reader.verify_trace``.
    """
    h = hashlib.sha256(
        _CORPUS_SEED + corpus_id.encode() + b":" + task_id.encode()
    ).digest()
    return hashlib.sha256(b"hmac:" + h).digest()


def _derive_signing_seed(corpus_id: str, task_id: str) -> bytes:
    """Return the 32-byte Ed25519 private key seed for a corpus task."""
    h = hashlib.sha256(
        _CORPUS_SEED + corpus_id.encode() + b":" + task_id.encode()
    ).digest()
    return hashlib.sha256(b"sign:" + h).digest()


# ---------------------------------------------------------------------------
# Corpus metadata
# ---------------------------------------------------------------------------

AUTHOR_CORPUS_META: Dict[str, Dict[str, Any]] = {
    "support-agent": {
        "corpus_id": "support-agent",
        "description": (
            "Author-original corpus: a customer-support bot handling five canonical "
            "service scenarios. All traces are author-original and redistributable "
            "under the Apache-2.0 licence."
        ),
        "task_count": 5,
        "license": "Apache-2.0",
        "split": "test",
    },
    "code-review": {
        "corpus_id": "code-review",
        "description": (
            "Author-original corpus: a code-review bot identifying five canonical "
            "defect categories in Python snippets. All traces are author-original "
            "and redistributable under the Apache-2.0 licence."
        ),
        "task_count": 5,
        "license": "Apache-2.0",
        "split": "test",
    },
    "payments-policy": {
        "corpus_id": "payments-policy",
        "description": (
            "Author-original corpus: a compliance-aware payment processor applying "
            "four sequential policy gates. Tasks cover approve, block, and hold "
            "outcomes. All traces are author-original and redistributable under the "
            "Apache-2.0 licence."
        ),
        "task_count": 5,
        "license": "Apache-2.0",
        "split": "test",
    },
}


def list_author_corpora() -> List[str]:
    """Return sorted list of bundled author-original corpus ids."""
    return sorted(AUTHOR_CORPUS_META.keys())


# ---------------------------------------------------------------------------
# Bundled resource helpers
# ---------------------------------------------------------------------------

def _corpora_root() -> Path:
    """Return the on-disk path of the bundled ``corpora/`` directory."""
    resource = files("stepback.bench").joinpath("corpora")
    with as_file(resource) as path:
        return Path(path)


def _corpus_dir(corpus_id: str) -> Path:
    return _corpora_root() / corpus_id


def _load_manifest(corpus_id: str) -> Dict[str, Any]:
    manifest_path = _corpus_dir(corpus_id) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No manifest.json for corpus {corpus_id!r}. "
            "Run scripts/generate_author_corpora.py to regenerate."
        )
    with manifest_path.open(encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_author_corpus(corpus_id: str) -> List[CorpusTask]:
    """Load bundled author-original corpus tasks as :class:`CorpusTask` objects.

    Parameters
    ----------
    corpus_id:
        One of the ids returned by :func:`list_author_corpora`.

    Returns
    -------
    list[CorpusTask]
        One task per ``.sb`` file in the corpus directory.  Each task carries:

        * ``corpus_id``, ``task_id``, ``split`` — corpus identity
        * ``prompt`` — the natural-language task description (no ground truth)
        * ``inputs`` — runtime-relevant inputs for an agent implementation
        * ``metadata["evaluation"]`` — ground-truth outcomes (do not pass
          to the agent under evaluation)
        * ``metadata["hmac_key_hex"]`` — hex HMAC key for ``verify_trace``
        * ``metadata["sb_path"]`` — path to the frozen ``.sb`` trace file
        * ``metadata["raw"]`` — the raw manifest entry

    Raises
    ------
    ValueError
        If ``corpus_id`` is not a known author-original corpus.
    FileNotFoundError
        If the bundled manifest or a ``.sb`` file is missing.
    """
    if corpus_id not in AUTHOR_CORPUS_META:
        raise ValueError(
            f"Unknown author corpus id {corpus_id!r}. "
            f"Known: {list_author_corpora()}"
        )

    manifest = _load_manifest(corpus_id)
    tasks: List[CorpusTask] = []
    corpus_dir = _corpus_dir(corpus_id)
    meta = AUTHOR_CORPUS_META[corpus_id]

    for entry in manifest.get("tasks", []):
        task_id = entry["task_id"]
        sb_path = corpus_dir / f"{task_id}.sb"
        if not sb_path.is_file():
            raise FileNotFoundError(
                f"Missing .sb file for {corpus_id}/{task_id}: {sb_path}"
            )
        tasks.append(
            CorpusTask(
                corpus_id=corpus_id,
                task_id=task_id,
                split=meta["split"],
                prompt=entry["prompt"],
                inputs=entry.get("inputs", {}),
                metadata={
                    "evaluation": entry.get("evaluation", {}),
                    "hmac_key_hex": entry["hmac_key_hex"],
                    "sb_path": str(sb_path),
                    "raw": entry,
                    "source_path": str(sb_path),
                    "line_number": 0,
                },
            )
        )

    return tasks


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _make_recorder_key(corpus_id: str, task_id: str):
    """Return a :class:`~stepback.recorder.RecorderKey` for a corpus task."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from stepback.recorder import RecorderKey

    hmac_key = _derive_hmac_key(corpus_id, task_id)
    signing_key = Ed25519PrivateKey.from_private_bytes(
        _derive_signing_seed(corpus_id, task_id)
    )
    return RecorderKey(hmac_key=hmac_key, signing_key=signing_key)


def generate_author_corpora(output_dir: str | Path, *, force: bool = False) -> None:
    """Generate (or regenerate) all author-original corpus ``.sb`` files.

    Writes ``<output_dir>/<corpus_id>/<task_id>.sb`` and
    ``<output_dir>/<corpus_id>/manifest.json`` for each corpus.

    The HMAC keys are deterministic — re-running this function always produces
    the same keys (though not byte-identical ``.sb`` files due to wallclock
    timestamps).

    Parameters
    ----------
    output_dir:
        Directory to write corpora into.  Created if it does not exist.
    force:
        If True, overwrite existing ``.sb`` files.  If False (default),
        skip tasks whose ``.sb`` file already exists.
    """
    from stepback.recorder import record as sb_record
    from stepback.testing.support_agent import SUPPORT_TASKS, run_support_task
    from stepback.testing.code_review_agent import CODE_REVIEW_TASKS, run_code_review_task
    from stepback.testing.payments_policy_agent import PAYMENTS_TASKS, run_payments_task

    out = Path(output_dir)

    _run_corpus(
        out,
        corpus_id="support-agent",
        tasks=SUPPORT_TASKS,
        runner=run_support_task,
        force=force,
        sb_record=sb_record,
    )
    _run_corpus(
        out,
        corpus_id="code-review",
        tasks=CODE_REVIEW_TASKS,
        runner=run_code_review_task,
        force=force,
        sb_record=sb_record,
    )
    _run_corpus(
        out,
        corpus_id="payments-policy",
        tasks=PAYMENTS_TASKS,
        runner=run_payments_task,
        force=force,
        sb_record=sb_record,
    )


def _run_corpus(
    out: Path,
    corpus_id: str,
    tasks: List[Dict[str, Any]],
    runner,
    force: bool,
    sb_record,
) -> None:
    corpus_dir = out / corpus_id
    corpus_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: List[Dict[str, Any]] = []

    for task in tasks:
        task_id = task["task_id"]
        sb_path = corpus_dir / f"{task_id}.sb"
        key = _make_recorder_key(corpus_id, task_id)

        if sb_path.exists() and not force:
            # Reuse existing file; just collect its key for the manifest
            pass
        else:
            with sb_record(str(sb_path), key=key) as rec:
                runner(rec, task_id)

        manifest_entries.append({
            "task_id": task_id,
            "prompt": task["prompt"],
            "inputs": task.get("inputs", {}),
            "evaluation": task.get("evaluation", {}),
            "hmac_key_hex": key.hmac_key.hex(),
        })

    manifest = {
        "corpus_id": corpus_id,
        "schema_version": "1.0",
        "description": AUTHOR_CORPUS_META[corpus_id]["description"],
        "license": AUTHOR_CORPUS_META[corpus_id]["license"],
        "tasks": manifest_entries,
    }
    manifest_path = corpus_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
