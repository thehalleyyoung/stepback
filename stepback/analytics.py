"""ClickHouse analytics: schema DDL and record helpers for stepback traces.

This module provides two ClickHouse tables and their Python dataclass
counterparts:

**Dirty-set summary table** (Step 65) — one row per
``compute_dirty_set`` call:

* :py:class:`DirtySetRecord` — one analytics row per
  ``compute_dirty_set`` call.
* :py:data:`DIRTY_SET_TABLE_DDL` — ``CREATE TABLE`` DDL (MergeTree).
* :py:data:`DIRTY_SET_REPLICATED_TABLE_DDL` — DDL using
  ``ReplicatedMergeTree``.
* :py:func:`record_from_summary` — build a :py:class:`DirtySetRecord`
  from a :py:class:`~stepback.divergence.DirtySetSummary`.
* :py:class:`ClickHouseAnalytics` — thin writer for
  :py:class:`DirtySetRecord` rows.

**Trace-step table** (Step 139) — one row per step, enabling queries
by model, step kind, dirty reason, cost, policy decision, canonical
hash, and incident id:

* :py:class:`TraceStepRecord` — one analytics row per recorded step.
* :py:data:`TRACE_STEP_TABLE_DDL` — ``CREATE TABLE`` DDL (MergeTree).
* :py:data:`TRACE_STEP_REPLICATED_TABLE_DDL` — DDL using
  ``ReplicatedMergeTree``.
* :py:func:`step_record_from_dirty_entry` — build a
  :py:class:`TraceStepRecord` from a
  :py:class:`~stepback.divergence.DirtySetEntry` and the corresponding
  recorded step dict.
* :py:class:`TraceStepAnalytics` — thin writer for
  :py:class:`TraceStepRecord` rows.

Both modules are **optional** — nothing in the core stepback runtime
imports them.  The ClickHouse driver (``clickhouse-connect`` or
``clickhouse-driver``) is not listed in the core dependencies; import
errors from those packages are caught and re-raised with an
installation hint.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Mapping, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .divergence import DirtySetEntry, DirtySetSummary
    from .substitutions import Substitution

__all__ = [
    # --- Step 65: dirty-set summary ----------------------------------
    "DirtySetRecord",
    "DIRTY_SET_TABLE_DDL",
    "DIRTY_SET_REPLICATED_TABLE_DDL",
    "DIRTY_SET_TABLE_NAME",
    "record_from_summary",
    "ClickHouseAnalytics",
    # --- Step 139: per-step trace queries ----------------------------
    "TraceStepRecord",
    "TRACE_STEP_TABLE_DDL",
    "TRACE_STEP_REPLICATED_TABLE_DDL",
    "TRACE_STEP_TABLE_NAME",
    "step_record_from_dirty_entry",
    "TraceStepAnalytics",
]

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

#: Name of the summary table in ClickHouse.
DIRTY_SET_TABLE_NAME = "stepback_dirty_set_summary"

#: ``CREATE TABLE`` DDL for the summary table using a local ``MergeTree``.
#: Suitable for single-node and test environments.
DIRTY_SET_TABLE_DDL = f"""\
CREATE TABLE IF NOT EXISTS {DIRTY_SET_TABLE_NAME}
(
    -- Partition / identity columns
    recorded_at         DateTime64(3, 'UTC')  COMMENT 'Wall-clock time of compute_dirty_set call',
    trace_id            String                COMMENT 'Trace identifier from the SB-Trace header',
    substitution_kind   LowCardinality(String) COMMENT 'Canonical name of the dominant substitution type, e.g. PromptSubstitution',

    -- Dirty-set metrics
    step_count          UInt32                COMMENT 'Total number of steps in the trace',
    dirty_count         UInt32                COMMENT 'Number of dirty (must-re-execute) steps',
    clean_count         UInt32                COMMENT 'Number of clean (cache-hit) steps',
    branch_count        UInt32                COMMENT 'Number of parallel_branch_open events in the trace',
    calls_saved         UInt32                COMMENT 'Equivalent to clean_count: LLM/tool calls avoided by caching',

    -- Optional provenance
    agent_id            String                DEFAULT '' COMMENT 'Agent or application identifier; empty if not provided',
    schema_version      UInt8                 DEFAULT 1  COMMENT 'Analytics schema version for forward compatibility'
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(recorded_at)
ORDER BY (trace_id, recorded_at)
SETTINGS index_granularity = 8192;
"""

#: ``CREATE TABLE`` DDL using ``ReplicatedMergeTree`` for HA cluster
#: deployments.  Replace ``{{shard}}`` and ``{{replica}}`` with your
#: ClickHouse macro values, e.g. via Ansible or the ClickHouse Keeper
#: ``{shard}``/``{replica}`` built-in macros.
DIRTY_SET_REPLICATED_TABLE_DDL = f"""\
CREATE TABLE IF NOT EXISTS {DIRTY_SET_TABLE_NAME}
(
    recorded_at         DateTime64(3, 'UTC'),
    trace_id            String,
    substitution_kind   LowCardinality(String),
    step_count          UInt32,
    dirty_count         UInt32,
    clean_count         UInt32,
    branch_count        UInt32,
    calls_saved         UInt32,
    agent_id            String                DEFAULT '',
    schema_version      UInt8                 DEFAULT 1
)
ENGINE = ReplicatedMergeTree('/clickhouse/tables/{{shard}}/{DIRTY_SET_TABLE_NAME}', '{{replica}}')
PARTITION BY toYYYYMM(recorded_at)
ORDER BY (trace_id, recorded_at)
SETTINGS index_granularity = 8192;
"""


# ---------------------------------------------------------------------------
# DirtySetRecord — one analytics row
# ---------------------------------------------------------------------------

@dataclass
class DirtySetRecord:
    """One row in the ``stepback_dirty_set_summary`` ClickHouse table.

    Fields
    ------
    recorded_at : datetime.datetime
        UTC timestamp of the :py:func:`~stepback.divergence.compute_dirty_set`
        call.  Should be timezone-aware (``tzinfo=datetime.timezone.utc``).
    trace_id : str
        Identifier of the trace (from the SB-Trace ``header.trace_id`` field).
    substitution_kind : str
        Canonical class name of the *dominant* substitution applied, e.g.
        ``"PromptSubstitution"``.  If multiple substitution types were
        applied, this field records the first one in topological order.
        Use ``""`` if *no* substitutions were applied (empty sigma).
    step_count : int
        Total number of steps in the trace.
    dirty_count : int
        Number of steps in the dirty set D.
    clean_count : int
        Number of clean (cache-hit) steps (``step_count - dirty_count``).
    branch_count : int
        Number of ``parallel_branch_open`` events detected in the trace —
        a proxy for the DAG width.
    calls_saved : int
        LLM/tool calls avoided by caching (equivalent to ``clean_count``).
    agent_id : str
        Optional application or agent identifier for multi-tenant analytics.
        Defaults to ``""``.
    schema_version : int
        Analytics schema version; currently ``1``.
    """

    recorded_at: datetime.datetime
    trace_id: str
    substitution_kind: str
    step_count: int
    dirty_count: int
    clean_count: int
    branch_count: int
    calls_saved: int
    agent_id: str = ""
    schema_version: int = 1

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict suitable for passing to a ClickHouse insert."""
        return {
            "recorded_at": self.recorded_at,
            "trace_id": self.trace_id,
            "substitution_kind": self.substitution_kind,
            "step_count": self.step_count,
            "dirty_count": self.dirty_count,
            "clean_count": self.clean_count,
            "branch_count": self.branch_count,
            "calls_saved": self.calls_saved,
            "agent_id": self.agent_id,
            "schema_version": self.schema_version,
        }

    @classmethod
    def column_names(cls) -> List[str]:
        """Return the ordered list of column names (matches DDL order)."""
        return [f.name for f in fields(cls)]


# ---------------------------------------------------------------------------
# record_from_summary — factory
# ---------------------------------------------------------------------------

def record_from_summary(
    summary: "DirtySetSummary",
    trace_id: str,
    *,
    substitutions: Optional[Sequence["Substitution"]] = None,
    agent_id: str = "",
    recorded_at: Optional[datetime.datetime] = None,
) -> DirtySetRecord:
    """Build a :py:class:`DirtySetRecord` from a
    :py:class:`~stepback.divergence.DirtySetSummary`.

    Parameters
    ----------
    summary:
        Result returned by :py:func:`~stepback.divergence.compute_dirty_set`.
    trace_id:
        Identifier of the trace (``trace.header["trace_id"]`` or equivalent).
    substitutions:
        The substitutions that were passed to ``compute_dirty_set``.  Used to
        populate :py:attr:`DirtySetRecord.substitution_kind`.  Pass ``None``
        or an empty sequence when no substitutions were applied.
    agent_id:
        Optional agent / application tag for multi-tenant warehouses.
    recorded_at:
        UTC timestamp of the call.  Defaults to ``datetime.datetime.now(UTC)``.

    Returns
    -------
    DirtySetRecord
        A fully populated row ready for insertion.
    """
    if recorded_at is None:
        recorded_at = datetime.datetime.now(datetime.timezone.utc)

    # Determine dominant substitution kind.
    substitution_kind = _dominant_substitution_kind(substitutions)

    # Count parallel_branch_open events from the per-step entries.
    branch_count = sum(
        1
        for e in summary.entries
        if e.step_kind == "parallel_branch_open"
    )

    return DirtySetRecord(
        recorded_at=recorded_at,
        trace_id=trace_id,
        substitution_kind=substitution_kind,
        step_count=summary.step_count,
        dirty_count=summary.dirty_count,
        clean_count=summary.clean_count,
        branch_count=branch_count,
        calls_saved=summary.calls_saved,
        agent_id=agent_id,
    )


def _dominant_substitution_kind(
    substitutions: Optional[Sequence["Substitution"]],
) -> str:
    """Return the class name of the first substitution, or ``""``."""
    if not substitutions:
        return ""
    return type(substitutions[0]).__name__


# ---------------------------------------------------------------------------
# ClickHouseAnalytics — optional writer
# ---------------------------------------------------------------------------

class ClickHouseAnalytics:
    """Thin writer that inserts :py:class:`DirtySetRecord` rows into
    ClickHouse via the ``clickhouse-connect`` package.

    The writer is intentionally minimal: it delegates all connection
    management to the caller-supplied ``client`` object so that
    connection pooling, retry logic, and auth are handled at the
    application layer.

    Parameters
    ----------
    client:
        A ``clickhouse_connect`` client (or any object with an
        ``insert(table, data, column_names=...)`` method).  Typically
        obtained via::

            import clickhouse_connect
            client = clickhouse_connect.get_client(host='localhost')

    table_name:
        Override the default table name (useful for integration tests).

    Raises
    ------
    ImportError
        If ``clickhouse-connect`` is not installed and
        :py:meth:`ensure_table` is called (which requires the package).
        Inserting records does *not* require the package — any object
        with an ``insert`` method is accepted.
    """

    def __init__(
        self,
        client: Any,
        table_name: str = DIRTY_SET_TABLE_NAME,
    ) -> None:
        self._client = client
        self._table_name = table_name

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def ensure_table(self, *, replicated: bool = False) -> None:
        """Create the summary table if it does not already exist.

        Parameters
        ----------
        replicated:
            If ``True``, use the ``ReplicatedMergeTree`` DDL; otherwise
            use the local ``MergeTree`` DDL.
        """
        ddl = (
            DIRTY_SET_REPLICATED_TABLE_DDL
            if replicated
            else DIRTY_SET_TABLE_DDL
        )
        # When a non-default table name is in use, substitute it in the DDL.
        if self._table_name != DIRTY_SET_TABLE_NAME:
            ddl = ddl.replace(DIRTY_SET_TABLE_NAME, self._table_name, 1)
        self._client.command(ddl)

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def insert_record(self, record: DirtySetRecord) -> None:
        """Insert a single :py:class:`DirtySetRecord` row.

        Parameters
        ----------
        record:
            The record to insert.
        """
        self.insert_records([record])

    def insert_records(self, records: Sequence[DirtySetRecord]) -> None:
        """Batch-insert a sequence of :py:class:`DirtySetRecord` rows.

        Parameters
        ----------
        records:
            One or more records to insert.  An empty sequence is a no-op.
        """
        if not records:
            return
        column_names = DirtySetRecord.column_names()
        data = [
            [row.to_dict()[col] for col in column_names]
            for row in records
        ]
        self._client.insert(
            self._table_name,
            data,
            column_names=column_names,
        )


# ===========================================================================
# Step 139 — per-step trace query schema
# ===========================================================================

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

#: Name of the per-step trace-query table in ClickHouse.
TRACE_STEP_TABLE_NAME = "stepback_trace_steps"

#: ``CREATE TABLE`` DDL for the per-step table using a local ``MergeTree``.
#: Enables queries by model, step kind, dirty reason, cost, policy decision,
#: canonical hash, and incident id.
TRACE_STEP_TABLE_DDL = f"""\
CREATE TABLE IF NOT EXISTS {TRACE_STEP_TABLE_NAME}
(
    -- Partition / identity columns
    recorded_at         DateTime64(3, 'UTC')       COMMENT 'Wall-clock time the analytics row was written',
    trace_id            String                     COMMENT 'Trace identifier from the SB-Trace header',
    step_id             String                     COMMENT 'Step identifier (e.g. step:1) from the recorded trace',
    step_index          UInt32                     COMMENT 'Zero-based position of this step in topological trace order; use for range queries',
    incident_id         String      DEFAULT ''     COMMENT 'Incident identifier for cross-trace correlation; empty string if not set',

    -- Step metadata
    step_kind           LowCardinality(String)     COMMENT 'step_kind from the recorded trace: llm_call, tool_call, router, etc.',
    model               LowCardinality(String) DEFAULT '' COMMENT 'Model identifier (e.g. gpt-4o); empty for non-LLM steps',

    -- Dirty-set classification
    dirty               UInt8                      COMMENT '1 if this step is in the dirty set (must re-execute); 0 for cache hit',
    dirty_reason        LowCardinality(String) DEFAULT '' COMMENT 'substituted | input_drift | parent_dirty | nondeterminism; empty for clean steps',

    -- Canonical inputs hash
    canonical_hash      String      DEFAULT ''     COMMENT 'SHA-256 hex of the canonical inputs (inputs_hash from the recorded step); empty if not recorded',

    -- Cost analytics
    cost_usd            Float64     DEFAULT 0.0    COMMENT 'Estimated USD cost for this step (0 for cache hits and non-LLM steps)',

    -- Policy decision (populated when a policy audit is run; empty otherwise)
    policy_decision     LowCardinality(String) DEFAULT '' COMMENT 'unchanged | divergent | newly_blocked | newly_allowed; empty if no policy audit run',

    -- Schema version
    schema_version      UInt8       DEFAULT 1      COMMENT 'Analytics schema version for forward compatibility'
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(recorded_at)
ORDER BY (trace_id, step_index)
SETTINGS index_granularity = 8192;
"""

#: ``CREATE TABLE`` DDL using ``ReplicatedMergeTree`` for HA cluster
#: deployments.  Replace ``{{shard}}`` and ``{{replica}}`` with your
#: ClickHouse macro values.
TRACE_STEP_REPLICATED_TABLE_DDL = f"""\
CREATE TABLE IF NOT EXISTS {TRACE_STEP_TABLE_NAME}
(
    recorded_at         DateTime64(3, 'UTC'),
    trace_id            String,
    step_id             String,
    step_index          UInt32,
    incident_id         String      DEFAULT '',
    step_kind           LowCardinality(String),
    model               LowCardinality(String) DEFAULT '',
    dirty               UInt8,
    dirty_reason        LowCardinality(String) DEFAULT '',
    canonical_hash      String      DEFAULT '',
    cost_usd            Float64     DEFAULT 0.0,
    policy_decision     LowCardinality(String) DEFAULT '',
    schema_version      UInt8       DEFAULT 1
)
ENGINE = ReplicatedMergeTree('/clickhouse/tables/{{shard}}/{TRACE_STEP_TABLE_NAME}', '{{replica}}')
PARTITION BY toYYYYMM(recorded_at)
ORDER BY (trace_id, step_index)
SETTINGS index_granularity = 8192;
"""


# ---------------------------------------------------------------------------
# TraceStepRecord — one analytics row per step
# ---------------------------------------------------------------------------

@dataclass
class TraceStepRecord:
    """One row in the ``stepback_trace_steps`` ClickHouse table.

    Each row represents a single recorded step and carries enough context
    to support warehouse queries by model, step kind, dirty reason, cost,
    policy decision, canonical hash, and incident id.

    Fields
    ------
    recorded_at : datetime.datetime
        UTC timestamp at which the analytics row was written.  Should be
        timezone-aware (``tzinfo=datetime.timezone.utc``).
    trace_id : str
        Identifier of the trace (from the SB-Trace ``header.trace_id``).
    step_id : str
        Step identifier from the recorded trace (e.g. ``"step:3"``).
    step_index : int
        Zero-based position of the step in topological trace order.  Used
        as the ``ORDER BY`` key so range queries are physically co-located.
    incident_id : str
        Optional incident identifier linking this step to an incident
        record.  Defaults to ``""``.
    step_kind : str
        Kind of step: ``"llm_call"``, ``"tool_call"``, ``"router"``, etc.
    model : str
        Model identifier for LLM steps (e.g. ``"gpt-4o"``).  Empty string
        for non-LLM steps.
    dirty : int
        ``1`` if this step is in the dirty set (must re-execute on replay);
        ``0`` for cache hits.  Stored as ``UInt8`` following ClickHouse
        convention.
    dirty_reason : str
        Human-readable reason code for dirty steps:
        ``"substituted"`` | ``"input_drift"`` | ``"parent_dirty"`` |
        ``"nondeterminism"``.  Empty string for clean steps.
    canonical_hash : str
        SHA-256 hex digest of the canonical inputs
        (``inputs_hash`` from the recorded step dict).  Empty string if
        the step was not hashed or the information is unavailable.
    cost_usd : float
        Estimated USD cost for this step.  ``0.0`` for cache hits and
        non-LLM steps.
    policy_decision : str
        Policy classification from a policy audit:
        ``"unchanged"`` | ``"divergent"`` | ``"newly_blocked"`` |
        ``"newly_allowed"``.  Empty string when no policy audit was run.
    schema_version : int
        Analytics schema version; currently ``1``.
    """

    recorded_at: datetime.datetime
    trace_id: str
    step_id: str
    step_index: int
    incident_id: str
    step_kind: str
    model: str
    dirty: int
    dirty_reason: str
    canonical_hash: str
    cost_usd: float
    policy_decision: str
    schema_version: int = 1

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict suitable for passing to a ClickHouse insert."""
        return {
            "recorded_at": self.recorded_at,
            "trace_id": self.trace_id,
            "step_id": self.step_id,
            "step_index": self.step_index,
            "incident_id": self.incident_id,
            "step_kind": self.step_kind,
            "model": self.model,
            "dirty": self.dirty,
            "dirty_reason": self.dirty_reason,
            "canonical_hash": self.canonical_hash,
            "cost_usd": self.cost_usd,
            "policy_decision": self.policy_decision,
            "schema_version": self.schema_version,
        }

    @classmethod
    def column_names(cls) -> List[str]:
        """Return the ordered list of column names (matches DDL order)."""
        return [f.name for f in fields(cls)]


# ---------------------------------------------------------------------------
# step_record_from_dirty_entry — factory
# ---------------------------------------------------------------------------

def step_record_from_dirty_entry(
    entry: "DirtySetEntry",
    step: Mapping[str, Any],
    *,
    trace_id: str,
    step_index: int = 0,
    incident_id: str = "",
    policy_decision: str = "",
    recorded_at: Optional[datetime.datetime] = None,
) -> TraceStepRecord:
    """Build a :py:class:`TraceStepRecord` from a dirty-set entry and step dict.

    Parameters
    ----------
    entry:
        A :py:class:`~stepback.divergence.DirtySetEntry` returned by
        :py:func:`~stepback.divergence.compute_dirty_set`.
    step:
        The corresponding recorded step dict from
        ``Trace.recorded_steps``.  Used to derive ``model``,
        ``cost_usd``, and ``canonical_hash`` (``inputs_hash``).
    trace_id:
        Trace identifier from the SB-Trace header.
    step_index:
        Zero-based position of the step in topological trace order.
        Defaults to ``0`` (callers should pass the actual index when
        iterating over ``enumerate(summary.entries)``).
    incident_id:
        Optional incident identifier.  Defaults to ``""``.
    policy_decision:
        Policy-audit classification for this step.  Pass the
        ``classification`` field from a per-step
        :class:`~stepback.policy_audit.StepComparison` when available.
        Defaults to ``""``.
    recorded_at:
        UTC timestamp.  Defaults to ``datetime.datetime.now(UTC)``.

    Returns
    -------
    TraceStepRecord
        A fully-populated row ready for insertion.
    """
    if recorded_at is None:
        recorded_at = datetime.datetime.now(datetime.timezone.utc)

    model: str = step.get("model") or step.get("model_id") or ""
    cost_usd: float = float(step.get("cost_usd") or 0.0)
    canonical_hash: str = step.get("inputs_hash") or ""
    dirty_reason: str = entry.dirty_reason or ""

    return TraceStepRecord(
        recorded_at=recorded_at,
        trace_id=trace_id,
        step_id=entry.step_id,
        step_index=step_index,
        incident_id=incident_id,
        step_kind=entry.step_kind,
        model=model,
        dirty=1 if entry.dirty else 0,
        dirty_reason=dirty_reason,
        canonical_hash=canonical_hash,
        cost_usd=cost_usd,
        policy_decision=policy_decision,
    )


# ---------------------------------------------------------------------------
# TraceStepAnalytics — optional writer
# ---------------------------------------------------------------------------

class TraceStepAnalytics:
    """Thin writer that inserts :py:class:`TraceStepRecord` rows into
    ClickHouse via the ``clickhouse-connect`` package.

    Mirrors the :py:class:`ClickHouseAnalytics` pattern for the per-step
    trace-query table.

    Parameters
    ----------
    client:
        A ``clickhouse_connect`` client (or any object with an
        ``insert(table, data, column_names=...)`` method and a
        ``command(ddl)`` method).

    table_name:
        Override the default table name (useful for integration tests).
    """

    def __init__(
        self,
        client: Any,
        table_name: str = TRACE_STEP_TABLE_NAME,
    ) -> None:
        self._client = client
        self._table_name = table_name

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def ensure_table(self, *, replicated: bool = False) -> None:
        """Create the trace-step table if it does not already exist.

        Parameters
        ----------
        replicated:
            If ``True``, use the ``ReplicatedMergeTree`` DDL.
        """
        if replicated:
            ddl = _build_trace_step_ddl_replicated(self._table_name)
        else:
            ddl = _build_trace_step_ddl(self._table_name)
        self._client.command(ddl)

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def insert_record(self, record: TraceStepRecord) -> None:
        """Insert a single :py:class:`TraceStepRecord` row."""
        self.insert_records([record])

    def insert_records(self, records: Sequence[TraceStepRecord]) -> None:
        """Batch-insert a sequence of :py:class:`TraceStepRecord` rows.

        An empty sequence is a no-op.
        """
        if not records:
            return
        column_names = TraceStepRecord.column_names()
        data = [
            [row.to_dict()[col] for col in column_names]
            for row in records
        ]
        self._client.insert(
            self._table_name,
            data,
            column_names=column_names,
        )


# ---------------------------------------------------------------------------
# Internal DDL builders — avoid string-replace hazards with table names
# ---------------------------------------------------------------------------

def _build_trace_step_ddl(table_name: str) -> str:
    """Return local MergeTree DDL for ``table_name``."""
    return TRACE_STEP_TABLE_DDL.replace(TRACE_STEP_TABLE_NAME, table_name, 1)


def _build_trace_step_ddl_replicated(table_name: str) -> str:
    """Return ReplicatedMergeTree DDL for ``table_name``.

    The ZooKeeper path uses the *default* table name so that custom
    ``table_name`` values don't accidentally share a replication path.
    The CREATE TABLE statement itself is updated to ``table_name``.
    """
    return TRACE_STEP_REPLICATED_TABLE_DDL.replace(
        TRACE_STEP_TABLE_NAME, table_name, 1
    )
