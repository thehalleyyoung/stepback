"""Scheduled frontier-model re-evaluation (Step 125).

Prevents benchmark results from fossilising around one provider generation by:

1. Maintaining a **catalog** of known frontier models with release dates,
   family groupings, and explicit supersession links.
2. Defining **re-evaluation policies** — age-based and supersession-based.
3. **Computing a schedule**: given a list of benchmark submissions and
   (optionally) the frontier models they exercised, return the set of
   submissions that need re-running and why.
4. Persisting the schedule as JSON so CI (e.g. a GitHub Actions cron job)
   can consume it and open re-evaluation issues automatically.

Scheduling workflow
-------------------
The typical usage pattern in a nightly / weekly CI job is::

    from stepback.bench.frontier_eval import (
        default_catalog,
        compute_schedule,
        save_schedule_json,
        ReEvaluationPolicy,
    )
    from stepback.bench.leaderboard import load_submissions_from_dir

    submissions = load_submissions_from_dir("bench-results/submissions/")
    policy = ReEvaluationPolicy(max_age_days=180, check_superseded=True)
    schedule = compute_schedule(submissions, policy=policy)
    save_schedule_json(schedule, "bench-results/reeval-schedule.json")
    print(schedule.summary())

GitHub Actions cron example (.github/workflows/frontier-reeval.yml)::

    on:
      schedule:
        - cron: '0 6 * * 1'   # every Monday 06:00 UTC
    jobs:
      schedule:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - run: pip install -e ".[dev]"
          - run: stepback bench frontier-reeval-schedule
                   --submissions bench-results/submissions/
                   --out bench-results/reeval-schedule.json
          - run: cat bench-results/reeval-schedule.json

Catalog staleness
-----------------
The built-in :data:`FRONTIER_MODELS` seed list has a
:data:`CATALOG_GENERATED_AT` timestamp.  Maintainers can override it by
passing a custom :class:`FrontierModelCatalog` or by loading one from a
JSON file with :func:`load_catalog_from_json`.  The catalog version is
included in every schedule's JSON so consumers know whether it is fresh.

Model provenance in submissions
--------------------------------
The existing :class:`~stepback.bench.submission.SubmissionManifest` does not
store which LLM models were exercised during a benchmark run (it records
*stepback* software versions, not provider-model versions).  Until
``SubmissionManifest`` gains an official ``evaluation_models`` field, callers
can pass ``submission_models: dict[submission_id, list[model_id]]`` to
:func:`compute_schedule`.  When this mapping is absent for a submission, only
age-based staleness is checked.

Re-evaluation reasons
---------------------
:data:`REASON_TOO_OLD`
    The submission date is older than ``policy.max_age_days``.
:data:`REASON_MODEL_SUPERSEDED`
    A model in ``submission_models`` has an explicit ``superseded_by``
    successor in the catalog.
:data:`REASON_NEW_FRONTIER`
    The same model family contains a newer release that was not the
    submitted model, even if no explicit supersession link exists.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Re-evaluation reason constants
# ---------------------------------------------------------------------------

REASON_TOO_OLD = "results_too_old"
"""The submission is older than the policy's ``max_age_days`` threshold."""

REASON_MODEL_SUPERSEDED = "model_superseded"
"""A model used in the submission has an explicit ``superseded_by`` link in
the catalog."""

REASON_NEW_FRONTIER = "new_frontier_available"
"""A newer release in the same model family exists in the catalog, even
without an explicit supersession link."""

REASON_UNKNOWN_MODEL = "unknown_model"
"""A model ID in ``submission_models`` is not present in the catalog; the
schedule includes it as a warning so maintainers can update the catalog."""

# ---------------------------------------------------------------------------
# Catalog data structures
# ---------------------------------------------------------------------------


@dataclass
class ModelRecord:
    """A single entry in the frontier model catalog.

    Parameters
    ----------
    model_id:
        Canonical model identifier, e.g. ``"gpt-4o-2024-11-20"``.
    provider:
        Provider name, e.g. ``"openai"``, ``"anthropic"``, ``"google"``.
    family:
        Model family string, e.g. ``"gpt-4o"`` or ``"claude-3-5-sonnet"``.
        Multiple :class:`ModelRecord` items share a family; the newest
        non-superseded member is the current frontier representative.
    release_date:
        ISO 8601 date string, e.g. ``"2024-11-20"``.
    superseded_by:
        ``model_id`` of the explicit successor, or ``None`` if this model is
        still current.  A superseded model still appears in the catalog so
        old submissions can be mapped to the chain.
    """

    model_id: str
    provider: str
    family: str
    release_date: str
    superseded_by: Optional[str] = None

    # ------------------------------------------------------------------
    # helpers

    @property
    def release_date_obj(self) -> date:
        """Parse :attr:`release_date` to a :class:`datetime.date`."""
        return date.fromisoformat(self.release_date)

    def is_superseded(self) -> bool:
        """Return ``True`` if another model explicitly supersedes this one."""
        return self.superseded_by is not None

    def to_json(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "family": self.family,
            "release_date": self.release_date,
            "superseded_by": self.superseded_by,
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "ModelRecord":
        """Deserialise from a JSON dict."""
        return cls(
            model_id=str(d["model_id"]),
            provider=str(d["provider"]),
            family=str(d["family"]),
            release_date=str(d["release_date"]),
            superseded_by=d.get("superseded_by"),
        )


#: ISO 8601 datetime of when the built-in catalog was last curated.
CATALOG_GENERATED_AT: str = "2026-05-12T00:00:00Z"

#: Seed catalog of representative frontier models from major providers.
#: Maintainers should refresh this list and bump :data:`CATALOG_GENERATED_AT`
#: whenever new flagship models are released.
FRONTIER_MODELS: List[ModelRecord] = [
    # ---- OpenAI GPT-4o family ----------------------------------------
    ModelRecord(
        model_id="gpt-4o-2024-05-13",
        provider="openai",
        family="gpt-4o",
        release_date="2024-05-13",
        superseded_by="gpt-4o-2024-08-06",
    ),
    ModelRecord(
        model_id="gpt-4o-2024-08-06",
        provider="openai",
        family="gpt-4o",
        release_date="2024-08-06",
        superseded_by="gpt-4o-2024-11-20",
    ),
    ModelRecord(
        model_id="gpt-4o-2024-11-20",
        provider="openai",
        family="gpt-4o",
        release_date="2024-11-20",
        superseded_by=None,
    ),
    # ---- OpenAI o1 family --------------------------------------------
    ModelRecord(
        model_id="o1-2024-12-17",
        provider="openai",
        family="o1",
        release_date="2024-12-17",
        superseded_by=None,
    ),
    # ---- Anthropic Claude-3.5-Sonnet family --------------------------
    ModelRecord(
        model_id="claude-3-5-sonnet-20240620",
        provider="anthropic",
        family="claude-3-5-sonnet",
        release_date="2024-06-20",
        superseded_by="claude-3-5-sonnet-20241022",
    ),
    ModelRecord(
        model_id="claude-3-5-sonnet-20241022",
        provider="anthropic",
        family="claude-3-5-sonnet",
        release_date="2024-10-22",
        superseded_by=None,
    ),
    # ---- Anthropic Claude-3.5-Haiku family ---------------------------
    ModelRecord(
        model_id="claude-3-5-haiku-20241022",
        provider="anthropic",
        family="claude-3-5-haiku",
        release_date="2024-10-22",
        superseded_by=None,
    ),
    # ---- Google Gemini-1.5-Pro family --------------------------------
    ModelRecord(
        model_id="gemini-1.5-pro-001",
        provider="google",
        family="gemini-1.5-pro",
        release_date="2024-05-14",
        superseded_by="gemini-1.5-pro-002",
    ),
    ModelRecord(
        model_id="gemini-1.5-pro-002",
        provider="google",
        family="gemini-1.5-pro",
        release_date="2024-09-24",
        superseded_by=None,
    ),
    # ---- Google Gemini-2.0-Flash family ------------------------------
    ModelRecord(
        model_id="gemini-2.0-flash-001",
        provider="google",
        family="gemini-2.0-flash",
        release_date="2025-02-05",
        superseded_by=None,
    ),
    # ---- Meta Llama-3.1 family ---------------------------------------
    ModelRecord(
        model_id="meta-llama/Meta-Llama-3.1-405B-Instruct",
        provider="meta",
        family="llama-3.1",
        release_date="2024-07-23",
        superseded_by="meta-llama/Llama-3.3-70B-Instruct",
    ),
    ModelRecord(
        model_id="meta-llama/Llama-3.3-70B-Instruct",
        provider="meta",
        family="llama-3.3",
        release_date="2024-12-06",
        superseded_by=None,
    ),
    # ---- Mistral Large family ----------------------------------------
    ModelRecord(
        model_id="mistral-large-2407",
        provider="mistral",
        family="mistral-large",
        release_date="2024-07-24",
        superseded_by="mistral-large-2411",
    ),
    ModelRecord(
        model_id="mistral-large-2411",
        provider="mistral",
        family="mistral-large",
        release_date="2024-11-18",
        superseded_by=None,
    ),
]


class FrontierModelCatalog:
    """A searchable collection of :class:`ModelRecord` entries.

    Parameters
    ----------
    models:
        Sequence of :class:`ModelRecord` entries.  The catalog is indexed on
        first access for O(1) lookups; pass ``models=[]`` for an empty catalog.
    catalog_generated_at:
        ISO 8601 datetime string indicating when the catalog was curated.
        Defaults to :data:`CATALOG_GENERATED_AT` when the default seed list is
        used, or ``None`` when loaded from external JSON without the field.
    """

    def __init__(
        self,
        models: List[ModelRecord],
        catalog_generated_at: Optional[str] = None,
    ) -> None:
        self._models: List[ModelRecord] = list(models)
        self.catalog_generated_at: Optional[str] = catalog_generated_at
        self._by_id: Dict[str, ModelRecord] = {}
        self._by_family: Dict[str, List[ModelRecord]] = {}
        self._build_index()

    # ------------------------------------------------------------------
    # index construction

    def _build_index(self) -> None:
        self._by_id = {}
        self._by_family = {}
        for m in self._models:
            self._by_id[m.model_id] = m
            self._by_family.setdefault(m.family, []).append(m)
        # Sort each family list by release_date ascending.
        for fam in self._by_family:
            self._by_family[fam].sort(key=lambda r: r.release_date)

    # ------------------------------------------------------------------
    # lookup helpers

    def find_by_id(self, model_id: str) -> Optional[ModelRecord]:
        """Return the :class:`ModelRecord` for *model_id*, or ``None``."""
        return self._by_id.get(model_id)

    def find_by_family(self, family: str) -> List[ModelRecord]:
        """Return all models in *family*, sorted oldest-first.

        Returns an empty list if no models match.
        """
        return list(self._by_family.get(family, []))

    def latest_in_family(self, family: str) -> Optional[ModelRecord]:
        """Return the most recently released model in *family*.

        If multiple models share the same latest release date, the one with
        the lexicographically greatest ``model_id`` is returned for stability.
        """
        members = self._by_family.get(family, [])
        if not members:
            return None
        latest_date = max(m.release_date for m in members)
        candidates = [m for m in members if m.release_date == latest_date]
        return max(candidates, key=lambda m: m.model_id)

    def current_frontier(self, family: str) -> Optional[ModelRecord]:
        """Return the non-superseded model in *family*, or the latest if all
        are superseded.

        A model is *current* if ``superseded_by is None``.  If multiple
        non-superseded members exist (e.g. parallel branches), the
        most-recently-released one is returned.
        """
        members = self._by_family.get(family, [])
        active = [m for m in members if m.superseded_by is None]
        if active:
            latest_date = max(m.release_date for m in active)
            candidates = [m for m in active if m.release_date == latest_date]
            return max(candidates, key=lambda m: m.model_id)
        # Fall back to the newest member even if superseded.
        return self.latest_in_family(family)

    def is_superseded(self, model_id: str) -> bool:
        """Return ``True`` if *model_id* has an explicit successor."""
        rec = self._by_id.get(model_id)
        return rec is not None and rec.superseded_by is not None

    def superseder(self, model_id: str) -> Optional[ModelRecord]:
        """Return the :class:`ModelRecord` that supersedes *model_id*."""
        rec = self._by_id.get(model_id)
        if rec is None or rec.superseded_by is None:
            return None
        return self._by_id.get(rec.superseded_by)

    def frontier_families(self) -> List[str]:
        """Return sorted list of distinct family strings in the catalog."""
        return sorted(self._by_family.keys())

    @property
    def models(self) -> List[ModelRecord]:
        """All models in the catalog (insertion order)."""
        return list(self._models)

    def __len__(self) -> int:
        return len(self._models)

    # ------------------------------------------------------------------
    # serialisation

    def to_json(self) -> Dict[str, Any]:
        """Serialise the catalog to a JSON-safe dict."""
        return {
            "catalog_generated_at": self.catalog_generated_at,
            "models": [m.to_json() for m in self._models],
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "FrontierModelCatalog":
        """Load a catalog from a JSON dict (as produced by :meth:`to_json`)."""
        models = [ModelRecord.from_json(r) for r in d.get("models", [])]
        return cls(
            models=models,
            catalog_generated_at=d.get("catalog_generated_at"),
        )


def default_catalog() -> FrontierModelCatalog:
    """Return the built-in seed :class:`FrontierModelCatalog`.

    This is a convenience wrapper around :data:`FRONTIER_MODELS` and
    :data:`CATALOG_GENERATED_AT`.  Maintainers should call
    :func:`load_catalog_from_json` to override it with a fresh catalog file.
    """
    return FrontierModelCatalog(FRONTIER_MODELS, catalog_generated_at=CATALOG_GENERATED_AT)


def load_catalog_from_json(path: str) -> FrontierModelCatalog:
    """Load a :class:`FrontierModelCatalog` from *path*.

    The file must contain a JSON object in the format produced by
    :meth:`FrontierModelCatalog.to_json`.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the JSON is malformed or missing required fields.
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path!r}, got {type(raw).__name__}")
    return FrontierModelCatalog.from_json(raw)


def save_catalog_json(catalog: FrontierModelCatalog, path: str) -> None:
    """Write *catalog* to *path* as JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(catalog.to_json(), fh, indent=2)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass
class ReEvaluationPolicy:
    """Parameters governing when a benchmark submission needs re-evaluation.

    Parameters
    ----------
    max_age_days:
        Submissions whose ``submission_date`` is older than this many days
        from ``reference_date`` are flagged with :data:`REASON_TOO_OLD`.
        Set to ``0`` or a negative value to disable age-based staleness.
    check_superseded:
        When ``True`` (default), a submission that used a model which has
        an explicit ``superseded_by`` link is flagged with
        :data:`REASON_MODEL_SUPERSEDED`.
    check_new_frontier:
        When ``True`` (default), a submission that used a model which is
        not the latest release in its family is flagged with
        :data:`REASON_NEW_FRONTIER`, even without a supersession link.
    """

    max_age_days: int = 180
    check_superseded: bool = True
    check_new_frontier: bool = True

    def to_json(self) -> Dict[str, Any]:
        return {
            "max_age_days": self.max_age_days,
            "check_superseded": self.check_superseded,
            "check_new_frontier": self.check_new_frontier,
        }


# ---------------------------------------------------------------------------
# Schedule output structures
# ---------------------------------------------------------------------------


@dataclass
class ReEvaluationItem:
    """One submission that needs re-evaluation.

    Parameters
    ----------
    submission_id:
        The submission's unique identifier.
    submission_date:
        ISO 8601 date string taken from the submission (audit declaration or
        bench_results timestamp).  May be ``"unknown"`` if unavailable.
    age_days:
        Number of calendar days between *submission_date* and the reference
        date used in :func:`compute_schedule`.  ``None`` if the date could
        not be parsed.
    reasons:
        Sorted list of reason constants explaining why re-evaluation is
        required (e.g. :data:`REASON_TOO_OLD`, :data:`REASON_MODEL_SUPERSEDED`).
    stale_models:
        Model IDs that triggered re-evaluation (may be empty when the only
        reason is :data:`REASON_TOO_OLD`).
    suggested_models:
        Recommended newer model IDs from the catalog.
    unknown_models:
        Model IDs in the submission that were not found in the catalog.
    """

    submission_id: str
    submission_date: str
    age_days: Optional[int]
    reasons: List[str]
    stale_models: List[str]
    suggested_models: List[str]
    unknown_models: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "submission_date": self.submission_date,
            "age_days": self.age_days,
            "reasons": sorted(self.reasons),
            "stale_models": sorted(self.stale_models),
            "suggested_models": sorted(self.suggested_models),
            "unknown_models": sorted(self.unknown_models),
        }


@dataclass
class ReEvaluationSchedule:
    """The computed re-evaluation schedule.

    Parameters
    ----------
    generated_at:
        ISO 8601 datetime of when :func:`compute_schedule` was called.
    reference_date:
        ISO 8601 date used as "today" for age calculations.
    catalog_generated_at:
        Timestamp from the catalog that was used.
    policy:
        The :class:`ReEvaluationPolicy` used.
    items:
        Submissions that need re-evaluation, sorted by ``submission_id``.
    total_submissions:
        Number of submissions evaluated (including those that do not need
        re-evaluation).
    """

    generated_at: str
    reference_date: str
    catalog_generated_at: Optional[str]
    policy: ReEvaluationPolicy
    items: List[ReEvaluationItem]
    total_submissions: int

    @property
    def needs_reeval_count(self) -> int:
        """Number of submissions that need re-evaluation."""
        return len(self.items)

    def to_json(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "reference_date": self.reference_date,
            "catalog_generated_at": self.catalog_generated_at,
            "policy": self.policy.to_json(),
            "total_submissions": self.total_submissions,
            "needs_reeval_count": self.needs_reeval_count,
            "items": [item.to_json() for item in self.items],
        }

    def summary(self) -> str:
        """One-line human-readable summary of the schedule."""
        return (
            f"Re-evaluation schedule: {self.needs_reeval_count}"
            f"/{self.total_submissions} submission(s) need re-evaluation"
            f" (reference date {self.reference_date},"
            f" policy max_age_days={self.policy.max_age_days})"
        )


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def compute_schedule(
    submissions: List[Any],
    catalog: Optional[FrontierModelCatalog] = None,
    policy: Optional[ReEvaluationPolicy] = None,
    reference_date: Optional[date] = None,
    submission_models: Optional[Dict[str, List[str]]] = None,
) -> ReEvaluationSchedule:
    """Compute the re-evaluation schedule for *submissions*.

    Parameters
    ----------
    submissions:
        A list of :class:`~stepback.bench.submission.SubmissionManifest`
        objects **or** raw JSON dicts in the same format.  Only the fields
        ``submission_id``, ``audit.submission_date``, and ``bench_results``
        are read.
    catalog:
        The :class:`FrontierModelCatalog` to check against.  Defaults to
        :func:`default_catalog`.
    policy:
        The :class:`ReEvaluationPolicy` to apply.  Defaults to
        ``ReEvaluationPolicy()``.
    reference_date:
        The date treated as "today" for age calculations.  Defaults to
        ``datetime.now(timezone.utc).date()``.  Pass a fixed date in tests.
    submission_models:
        Optional ``{submission_id: [model_id, ...]}`` mapping.  When present,
        model-based staleness checks are applied to the listed submissions.
        Submissions absent from this mapping only receive age-based checks.

    Returns
    -------
    ReEvaluationSchedule
        The computed schedule.  Items are sorted by ``submission_id``.
    """
    if catalog is None:
        catalog = default_catalog()
    if policy is None:
        policy = ReEvaluationPolicy()
    if reference_date is None:
        reference_date = datetime.now(timezone.utc).date()
    if submission_models is None:
        submission_models = {}

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    items: List[ReEvaluationItem] = []

    for raw in submissions:
        sub_id, sub_date_str = _extract_submission_id_and_date(raw)
        models_used: List[str] = list(submission_models.get(sub_id, []))

        reasons: List[str] = []
        stale_models: List[str] = []
        suggested: List[str] = []
        unknown: List[str] = []

        # ---- age-based check ----------------------------------------
        age_days: Optional[int] = None
        if sub_date_str != "unknown":
            try:
                sub_date = date.fromisoformat(sub_date_str)
                age_days = (reference_date - sub_date).days
                if policy.max_age_days > 0 and age_days > policy.max_age_days:
                    reasons.append(REASON_TOO_OLD)
            except ValueError:
                pass  # unparseable date; only model checks apply

        # ---- model-based checks -------------------------------------
        for mid in models_used:
            rec = catalog.find_by_id(mid)
            if rec is None:
                unknown.append(mid)
                continue
            # superseded check
            if policy.check_superseded and rec.is_superseded():
                if REASON_MODEL_SUPERSEDED not in reasons:
                    reasons.append(REASON_MODEL_SUPERSEDED)
                if mid not in stale_models:
                    stale_models.append(mid)
                # suggest the immediate successor
                succ = catalog.superseder(mid)
                if succ and succ.model_id not in suggested:
                    suggested.append(succ.model_id)
            # new-frontier check
            if policy.check_new_frontier:
                latest = catalog.latest_in_family(rec.family)
                if latest and latest.model_id != mid:
                    if REASON_NEW_FRONTIER not in reasons:
                        reasons.append(REASON_NEW_FRONTIER)
                    if mid not in stale_models:
                        stale_models.append(mid)
                    if latest.model_id not in suggested:
                        suggested.append(latest.model_id)

        if unknown:
            reasons.append(REASON_UNKNOWN_MODEL)

        if reasons:
            items.append(
                ReEvaluationItem(
                    submission_id=sub_id,
                    submission_date=sub_date_str,
                    age_days=age_days,
                    reasons=sorted(set(reasons)),
                    stale_models=sorted(set(stale_models)),
                    suggested_models=sorted(set(suggested)),
                    unknown_models=sorted(set(unknown)),
                )
            )

    # stable sort by submission_id
    items.sort(key=lambda x: x.submission_id)

    return ReEvaluationSchedule(
        generated_at=generated_at,
        reference_date=reference_date.isoformat(),
        catalog_generated_at=catalog.catalog_generated_at,
        policy=policy,
        items=items,
        total_submissions=len(submissions),
    )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_schedule_json(schedule: ReEvaluationSchedule, path: str) -> None:
    """Write *schedule* to *path* as indented JSON.

    Parent directories are created if they do not exist.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(schedule.to_json(), fh, indent=2)


def load_schedule_json(path: str) -> Dict[str, Any]:
    """Load a schedule JSON file produced by :func:`save_schedule_json`.

    Returns the raw dict; no schema validation is performed.
    """
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_submission_id_and_date(raw: Any) -> Tuple[str, str]:
    """Extract (submission_id, submission_date) from a raw submission.

    Accepts either a dataclass / object with attributes or a dict.
    Returns ``("unknown", "unknown")`` on failure.
    """
    if isinstance(raw, dict):
        sub_id = str(raw.get("submission_id", "unknown"))
        date_str = _find_date_in_dict(raw)
    else:
        # SubmissionManifest or similar object
        sub_id = str(getattr(raw, "submission_id", "unknown"))
        date_str = _find_date_in_obj(raw)
    return sub_id, date_str


def _find_date_in_dict(d: Dict[str, Any]) -> str:
    """Extract the best available date from a submission dict.

    Preference order:
    1. ``bench_results[0]["timestamp_utc"]`` (most precise: actual run time)
    2. ``audit["submission_date"]``
    3. ``"unknown"``
    """
    bench_results = d.get("bench_results", [])
    if bench_results and isinstance(bench_results, list):
        ts = bench_results[0].get("timestamp_utc", "") if isinstance(bench_results[0], dict) else ""
        if ts:
            return _iso_to_date_str(ts)
    audit = d.get("audit", {})
    if isinstance(audit, dict):
        sub_date = audit.get("submission_date", "")
        if sub_date:
            return str(sub_date)
    return "unknown"


def _find_date_in_obj(obj: Any) -> str:
    """Extract the best available date from a SubmissionManifest object."""
    bench_results = getattr(obj, "bench_results", None)
    if bench_results and isinstance(bench_results, list):
        first = bench_results[0]
        ts = (
            first.get("timestamp_utc", "") if isinstance(first, dict)
            else getattr(first, "timestamp_utc", "")
        )
        if ts:
            return _iso_to_date_str(str(ts))
    audit = getattr(obj, "audit", None)
    if audit is not None:
        sub_date = (
            audit.get("submission_date", "") if isinstance(audit, dict)
            else getattr(audit, "submission_date", "")
        )
        if sub_date:
            return str(sub_date)
    return "unknown"


def _iso_to_date_str(ts: str) -> str:
    """Convert an ISO 8601 datetime string to a ``YYYY-MM-DD`` date string.

    Returns the original string unchanged if parsing fails (the caller
    will then attempt ``date.fromisoformat()`` and catch any error).
    """
    if not ts:
        return "unknown"
    # Handle both "2025-03-01" and "2025-03-01T12:00:00Z" forms.
    return ts[:10]
