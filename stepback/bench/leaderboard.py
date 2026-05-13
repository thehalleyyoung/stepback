"""Hosted leaderboard generation from signed JSON submissions (Step 124).

This module builds a benchmark leaderboard from a collection of submission
manifest JSON files (as produced by :mod:`stepback.bench.submission`).
Invalid or unattested submissions are **rejected** before they appear in
any output.

Acceptance policy
-----------------
A submission is *accepted* when ALL of:

1. :func:`~stepback.bench.submission.validate_submission_json` returns
   ``valid=True`` (no :class:`~stepback.bench.submission.ValidationError`).
2. The ``validator_output`` field does not contain the text ``"FAILED"``
   — conformance failures are blocking regardless of pass/fail context.
3. Its ``submission_id`` has not appeared in a previously-accepted entry
   within the same :func:`build_leaderboard` call (first-seen wins;
   duplicates are rejected with a ``"duplicate submission_id"`` error).

An accepted submission is classified as:

* **public** — accepted, no warnings, ``code.git_dirty`` is not ``True``,
  **and** at least one valid witness cosignature is present.
* **development** — accepted but has one or more validation warnings
  (e.g. ``git_dirty=True``, missing e-mail, non-UUID submission ID) **or**
  does not carry any valid witness cosignatures.

.. note::
   Step 124 scopes *signing* to the existing ``trace_pack.verified=True``
   attestation flag (which the submission rules require to be ``True``).
   Full cryptographic Ed25519 signing of submission JSON payloads is a
   planned future enhancement; see Step 128 (key rotation) for the
   relevant attestation infrastructure.

Usage example
-------------
::

    from stepback.bench.leaderboard import build_leaderboard, generate_leaderboard_json

    submissions = load_submissions_from_dir("/path/to/submissions/")
    lb = build_leaderboard(submissions)
    print(f"Accepted: {len(lb.accepted)}, Rejected: {len(lb.rejected)}")
    with open("leaderboard.json", "w") as f:
        import json; json.dump(generate_leaderboard_json(lb), f, indent=2)
"""
from __future__ import annotations

import html as _html
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .submission import validate_submission_json
from .witness_cosigning import (
    WitnessCommitment,
    WitnessCosigningError,
    verify_witness_commitments,
)

# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

LEADERBOARD_SCHEMA_VERSION = "1.0"
"""JSON schema version for :func:`generate_leaderboard_json` output."""


@dataclass
class BenchMetrics:
    """Key benchmark metrics extracted from a :class:`SubmissionManifest`'s
    ``bench_results`` list.

    All fields are ``None`` when the relevant result record is absent or
    the metric was not measured.
    """

    corpus_id: str
    """Corpus the results were produced on."""

    cache_hit_rate: Optional[float]
    """Best *measured* cache hit rate; falls back to estimated if absent."""

    cost_reduction_factor: Optional[float]
    """Cost-reduction factor from replay-caching benchmark."""

    estimated_savings_pct: Optional[float]
    """Estimated LLM-call savings percentage."""

    wall_time_ms: Optional[float]
    """Wall-clock time of the primary benchmark run."""

    trace_count: Optional[int]
    """Number of traces in the primary run."""

    run_id: Optional[str]
    """``run_id`` of the primary result record."""

    def to_json(self) -> dict:
        return {
            "corpus_id": self.corpus_id,
            "cache_hit_rate": self.cache_hit_rate,
            "cost_reduction_factor": self.cost_reduction_factor,
            "estimated_savings_pct": self.estimated_savings_pct,
            "wall_time_ms": self.wall_time_ms,
            "trace_count": self.trace_count,
            "run_id": self.run_id,
        }


@dataclass
class LeaderboardEntry:
    """A single accepted entry on the leaderboard."""

    submission_id: str
    """UUID from the submission manifest."""

    submitter_name: str
    """Human-readable name."""

    submitter_organization: Optional[str]
    """Institutional affiliation, if provided."""

    submission_date: str
    """ISO 8601 date string (``"YYYY-MM-DD"``)."""

    stepback_version: str
    """``stepback.__version__`` used to produce the results."""

    is_development: bool
    """``True`` when the entry has validation warnings (e.g. git_dirty)."""

    warnings: List[str]
    """Validation warnings (non-empty iff ``is_development``). """

    metrics: List[BenchMetrics]
    """Per-corpus-id metrics extracted from the submission."""

    source: str
    """Source identifier (file path or dict index) for traceability."""

    has_witness_cosignatures: bool = False
    """``True`` when the entry carries at least one valid witness cosignature.

    Entries without witness cosignatures are classified as *development*
    regardless of other flags, because they cannot prove the trace pack
    existed before evaluation.
    """

    witness_count: int = 0
    """Number of valid witness cosignatures verified during leaderboard build."""

    def to_json(self) -> dict:
        return {
            "submission_id": self.submission_id,
            "submitter_name": self.submitter_name,
            "submitter_organization": self.submitter_organization,
            "submission_date": self.submission_date,
            "stepback_version": self.stepback_version,
            "is_development": self.is_development,
            "warnings": list(self.warnings),
            "metrics": [m.to_json() for m in self.metrics],
            "source": self.source,
            "has_witness_cosignatures": self.has_witness_cosignatures,
            "witness_count": self.witness_count,
        }


@dataclass
class RejectedEntry:
    """A submission that failed conformance or validation checks."""

    source: str
    """Source identifier (file path or dict index) for diagnostics."""

    submission_id: Optional[str]
    """Submission ID from the manifest, if parseable; otherwise ``None``."""

    errors: List[str]
    """Human-readable rejection reasons (field + message pairs)."""

    def to_json(self) -> dict:
        return {
            "source": self.source,
            "submission_id": self.submission_id,
            "errors": list(self.errors),
        }


@dataclass
class Leaderboard:
    """Complete leaderboard built from a set of submissions.

    Entries in :attr:`accepted` are ordered per corpus by descending
    ``cache_hit_rate`` (or ``cost_reduction_factor`` as secondary key),
    then ascending ``wall_time_ms``, then ascending ``submission_date``.
    """

    accepted: List[LeaderboardEntry]
    """Submissions that passed all validation and conformance checks."""

    rejected: List[RejectedEntry]
    """Submissions that failed at least one check."""

    generated_at: str
    """UTC ISO 8601 timestamp when this leaderboard was generated."""

    total_submitted: int
    """Total number of source submissions processed."""

    def public_entries(self) -> List[LeaderboardEntry]:
        """Return accepted entries that are *not* flagged as development."""
        return [e for e in self.accepted if not e.is_development]

    def development_entries(self) -> List[LeaderboardEntry]:
        """Return accepted entries flagged as development."""
        return [e for e in self.accepted if e.is_development]

    def to_json(self) -> dict:
        return {
            "schema_version": LEADERBOARD_SCHEMA_VERSION,
            "generated_at": self.generated_at,
            "total_submitted": self.total_submitted,
            "accepted_count": len(self.accepted),
            "rejected_count": len(self.rejected),
            "public_count": len(self.public_entries()),
            "development_count": len(self.development_entries()),
            "accepted": [e.to_json() for e in self.accepted],
            "rejected": [r.to_json() for r in self.rejected],
        }


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_leaderboard(
    submissions: List[Tuple[str, Any]],
) -> "Leaderboard":
    """Build a :class:`Leaderboard` from *submissions*.

    Parameters
    ----------
    submissions:
        List of ``(source_name, raw_value)`` pairs.  *source_name* is a
        human-readable identifier (file path, ``"<index N>"``, etc.).
        *raw_value* should be a ``dict``; non-dict values are rejected.

    Returns
    -------
    Leaderboard
        Populated leaderboard with accepted and rejected entries sorted
        as described in :class:`Leaderboard`.
    """
    accepted: List[LeaderboardEntry] = []
    rejected: List[RejectedEntry] = []
    seen_ids: set = set()

    for source, raw in submissions:
        if not isinstance(raw, dict):
            rejected.append(
                RejectedEntry(
                    source=source,
                    submission_id=None,
                    errors=[f"expected a JSON object (dict), got {type(raw).__name__}"],
                )
            )
            continue

        # --- run standard validation -----------------------------------------
        result = validate_submission_json(raw)
        submission_id: Optional[str] = raw.get("submission_id") or None

        errors: List[str] = []

        if not result.valid:
            for e in result.errors:
                errors.append(f"[{e.field}] {e.message}")

        # --- stricter conformance check: "FAILED" in validator_output is
        # blocking for leaderboard acceptance, even if validate_submission()
        # only issued a warning.
        vout = raw.get("validator_output") or ""
        if "FAILED" in vout and result.valid:
            errors.append(
                "[validator_output] contains 'FAILED'; "
                "all conformance fixtures must pass before submission"
            )

        # --- duplicate submission_id check -----------------------------------
        if submission_id and submission_id in seen_ids:
            errors.append(
                f"[submission_id] duplicate: '{submission_id}' already accepted"
            )

        if errors:
            rejected.append(
                RejectedEntry(
                    source=source,
                    submission_id=submission_id,
                    errors=errors,
                )
            )
            continue

        # --- accepted: extract metrics ---------------------------------------
        if submission_id:
            seen_ids.add(submission_id)

        code = raw.get("code") or {}
        audit = raw.get("audit") or {}
        bench_results = raw.get("bench_results") or []

        metrics = _extract_metrics(bench_results)

        is_development = bool(result.warnings) or bool(
            (code.get("git_dirty"))
        )

        # --- witness cosigning -----------------------------------------------
        # Attempt to verify cosignatures from the submission.  Malformed or
        # missing cosignatures demote the entry to development status; they
        # do not cause rejection.
        has_witness_cosignatures = False
        witness_count = 0
        raw_cosigs = raw.get("witness_cosignatures") or []
        if raw_cosigs:
            trace_pack = raw.get("trace_pack") or {}
            pack_sha256 = trace_pack.get("pack_sha256") or ""
            corpus_id = trace_pack.get("corpus_id") or ""

            # Find the earliest evaluation timestamp from bench_results.
            eval_ts: Optional[str] = _earliest_bench_timestamp(bench_results)

            try:
                commitments = [
                    WitnessCommitment.from_dict(c)
                    for c in raw_cosigs
                    if isinstance(c, dict)
                ]
                if commitments and pack_sha256:
                    witness_count = verify_witness_commitments(
                        pack_sha256,
                        corpus_id,
                        commitments,
                        min_witnesses=1,
                        evaluation_timestamp=eval_ts,
                    )
                    has_witness_cosignatures = True
            except WitnessCosigningError:
                # Invalid cosignatures demote to development; do not reject.
                pass

        if not has_witness_cosignatures:
            is_development = True

        entry = LeaderboardEntry(
            submission_id=submission_id or "",
            submitter_name=audit.get("submitter_name") or "",
            submitter_organization=audit.get("submitter_organization"),
            submission_date=audit.get("submission_date") or "",
            stepback_version=code.get("stepback_version") or "unknown",
            is_development=is_development,
            warnings=list(result.warnings),
            metrics=metrics,
            source=source,
            has_witness_cosignatures=has_witness_cosignatures,
            witness_count=witness_count,
        )
        accepted.append(entry)

    # Sort accepted entries: public first, then dev; within each group by
    # best cache_hit_rate descending, then submission_date ascending.
    accepted.sort(key=_entry_sort_key)

    return Leaderboard(
        accepted=accepted,
        rejected=rejected,
        generated_at=datetime.now(timezone.utc).isoformat(),
        total_submitted=len(submissions),
    )


def _earliest_bench_timestamp(bench_results: List[Any]) -> Optional[str]:
    """Return the earliest ``timestamp_utc`` from *bench_results*, or ``None``."""
    timestamps = []
    for rec in bench_results:
        if isinstance(rec, dict):
            ts = rec.get("timestamp_utc")
            if ts and isinstance(ts, str):
                timestamps.append(ts)
    if not timestamps:
        return None
    return min(timestamps)


def _entry_sort_key(entry: LeaderboardEntry) -> tuple:
    """Sort key: public before dev, then best metrics descending."""
    dev_flag = 1 if entry.is_development else 0
    best_hit = _best_metric(entry.metrics, "cache_hit_rate")
    best_cost = _best_metric(entry.metrics, "cost_reduction_factor")
    # Negate for descending order; None sorts last.
    hit_key = (-best_hit) if best_hit is not None else float("inf")
    cost_key = (-best_cost) if best_cost is not None else float("inf")
    return (dev_flag, hit_key, cost_key, entry.submission_date)


def _best_metric(metrics: List[BenchMetrics], attr: str) -> Optional[float]:
    """Return the best (highest) value of *attr* across all metrics entries."""
    values = [getattr(m, attr) for m in metrics if getattr(m, attr) is not None]
    return max(values) if values else None


def _extract_metrics(bench_results: List[Any]) -> List[BenchMetrics]:
    """Extract a :class:`BenchMetrics` per corpus_id from raw bench result dicts.

    When multiple records share the same ``corpus_id``, the one with the
    highest ``measured_hit_rate`` (or ``estimated_cache_hit_rate``) is
    selected as the representative entry for that corpus.
    """
    by_corpus: Dict[str, BenchMetrics] = {}

    for rec in bench_results:
        if not isinstance(rec, dict):
            continue
        corpus_id = str(rec.get("corpus_id") or "unknown")
        cache_d = rec.get("cache") or {}
        cost_d = rec.get("cost") or {}
        latency_d = rec.get("latency") or {}

        hit_rate: Optional[float] = None
        mhr = cache_d.get("measured_hit_rate")
        ehr = cache_d.get("estimated_cache_hit_rate")
        if mhr is not None:
            try:
                hit_rate = float(mhr)
            except (TypeError, ValueError):
                pass
        if hit_rate is None and ehr is not None:
            try:
                hit_rate = float(ehr)
            except (TypeError, ValueError):
                pass

        cost_reduction: Optional[float] = None
        crf = cost_d.get("cost_reduction_factor")
        if crf is not None:
            try:
                cost_reduction = float(crf)
            except (TypeError, ValueError):
                pass

        savings_pct: Optional[float] = None
        sp = cost_d.get("estimated_savings_pct")
        if sp is not None:
            try:
                savings_pct = float(sp)
            except (TypeError, ValueError):
                pass

        wall_time_ms: Optional[float] = None
        wt = latency_d.get("wall_time_ms")
        if wt is not None:
            try:
                wall_time_ms = float(wt)
            except (TypeError, ValueError):
                pass

        trace_count: Optional[int] = None
        tc = rec.get("trace_count")
        if tc is not None:
            try:
                trace_count = int(tc)
            except (TypeError, ValueError):
                pass

        candidate = BenchMetrics(
            corpus_id=corpus_id,
            cache_hit_rate=hit_rate,
            cost_reduction_factor=cost_reduction,
            estimated_savings_pct=savings_pct,
            wall_time_ms=wall_time_ms,
            trace_count=trace_count,
            run_id=rec.get("run_id"),
        )

        existing = by_corpus.get(corpus_id)
        if existing is None:
            by_corpus[corpus_id] = candidate
        else:
            # Keep the entry with the higher cache hit rate.
            new_hr = hit_rate if hit_rate is not None else -1.0
            old_hr = (
                existing.cache_hit_rate
                if existing.cache_hit_rate is not None
                else -1.0
            )
            if new_hr > old_hr:
                by_corpus[corpus_id] = candidate

    return list(by_corpus.values())


# ---------------------------------------------------------------------------
# Output generators
# ---------------------------------------------------------------------------


def generate_leaderboard_json(leaderboard: Leaderboard) -> dict:
    """Return a JSON-serialisable dict representing *leaderboard*.

    The output schema is versioned at :data:`LEADERBOARD_SCHEMA_VERSION`.
    """
    return leaderboard.to_json()


def generate_leaderboard_html(leaderboard: Leaderboard) -> str:
    """Return a self-contained HTML page rendering *leaderboard*.

    All user-supplied strings are HTML-escaped to prevent XSS.
    """
    esc = _html.escape

    def _fmt_pct(v: Optional[float]) -> str:
        if v is None:
            return "–"
        return f"{v:.1f}%"

    def _fmt_f(v: Optional[float], prec: int = 2) -> str:
        if v is None:
            return "–"
        return f"{v:.{prec}f}"

    def _entry_rows(entries: List[LeaderboardEntry], rank_offset: int = 0) -> str:
        rows = []
        rank = rank_offset + 1
        for entry in entries:
            best_hr = _best_metric(entry.metrics, "cache_hit_rate")
            best_cr = _best_metric(entry.metrics, "cost_reduction_factor")
            best_sp = _best_metric(entry.metrics, "estimated_savings_pct")
            rows.append(
                f"<tr>"
                f"<td>{rank}</td>"
                f"<td>{esc(entry.submitter_name)}"
                + (
                    f" <small>({esc(entry.submitter_organization)})</small>"
                    if entry.submitter_organization
                    else ""
                )
                + f"</td>"
                f"<td>{esc(entry.stepback_version)}</td>"
                f"<td>{esc(entry.submission_date)}</td>"
                f"<td>{_fmt_pct(best_hr)}</td>"
                f"<td>{_fmt_f(best_cr)}</td>"
                f"<td>{_fmt_pct(best_sp)}</td>"
                f"<td>{'⚠ dev' if entry.is_development else '✓'}</td>"
                f"<td>{'✓ ' + str(entry.witness_count) if entry.has_witness_cosignatures else '–'}</td>"
                f"</tr>"
            )
            rank += 1
        return "\n".join(rows)

    def _rejected_rows(entries: List[RejectedEntry]) -> str:
        rows = []
        for rej in entries:
            errs = "; ".join(esc(e) for e in rej.errors)
            rows.append(
                f"<tr>"
                f"<td>{esc(rej.source)}</td>"
                f"<td>{esc(rej.submission_id or '–')}</td>"
                f"<td>{errs}</td>"
                f"</tr>"
            )
        return "\n".join(rows)

    public_rows = _entry_rows(leaderboard.public_entries())
    dev_rows = _entry_rows(leaderboard.development_entries())
    rej_rows = _rejected_rows(leaderboard.rejected)

    th = (
        "<tr><th>#</th><th>Submitter</th><th>Version</th>"
        "<th>Date</th><th>Cache Hit Rate</th><th>Cost Reduction</th>"
        "<th>Savings %</th><th>Status</th><th>Witnesses</th></tr>"
    )

    dev_section = ""
    if leaderboard.development_entries():
        dev_section = f"""
<h2>Development Entries</h2>
<p>Accepted but flagged (e.g. git dirty working tree, missing contact email, no witness cosignatures).</p>
<table border="1" cellpadding="4" cellspacing="0">
  <thead>{th}</thead>
  <tbody>{dev_rows}</tbody>
</table>
"""

    rej_section = ""
    if leaderboard.rejected:
        rej_section = f"""
<h2>Rejected Submissions ({len(leaderboard.rejected)})</h2>
<table border="1" cellpadding="4" cellspacing="0">
  <thead><tr><th>Source</th><th>Submission ID</th><th>Errors</th></tr></thead>
  <tbody>{rej_rows}</tbody>
</table>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>stepback Benchmark Leaderboard</title>
<style>
  body {{ font-family: sans-serif; margin: 2em; }}
  table {{ border-collapse: collapse; margin-bottom: 1em; }}
  th {{ background: #eee; }}
  td, th {{ padding: 4px 8px; }}
  small {{ color: #666; }}
</style>
</head>
<body>
<h1>stepback Benchmark Leaderboard</h1>
<p>Generated: {esc(leaderboard.generated_at)} &nbsp;|&nbsp;
   Total submitted: {leaderboard.total_submitted} &nbsp;|&nbsp;
   Accepted: {len(leaderboard.accepted)} (public: {len(leaderboard.public_entries())},
   dev: {len(leaderboard.development_entries())}) &nbsp;|&nbsp;
   Rejected: {len(leaderboard.rejected)}</p>

<h2>Public Leaderboard</h2>
<table border="1" cellpadding="4" cellspacing="0">
  <thead>{th}</thead>
  <tbody>{public_rows}</tbody>
</table>
{dev_section}
{rej_section}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Directory loader
# ---------------------------------------------------------------------------


def load_submissions_from_dir(directory: str) -> List[Tuple[str, Any]]:
    """Load all ``*.json`` files from *directory* as submission dicts.

    Returns a list of ``(file_path, parsed_value)`` tuples sorted by
    file path for deterministic ordering.  Files that cannot be parsed
    as JSON return ``(file_path, None)`` so the caller can surface a
    ``RejectedEntry`` for them.
    """
    results: List[Tuple[str, Any]] = []
    try:
        names = sorted(
            n for n in os.listdir(directory) if n.endswith(".json")
        )
    except OSError as exc:
        raise OSError(f"cannot list directory '{directory}': {exc}") from exc

    for name in names:
        fpath = os.path.join(directory, name)
        try:
            with open(fpath, encoding="utf-8") as f:
                value = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            results.append((fpath, {"_parse_error": str(exc)}))
            continue
        results.append((fpath, value))
    return results
