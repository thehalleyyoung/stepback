"""Tests for ClickHouse dirty-set analytics (Step 65).

Validates that:

1. :py:class:`DirtySetRecord` is a correct dataclass with all required fields.
2. :py:func:`record_from_summary` builds records correctly from
   :py:class:`~stepback.divergence.DirtySetSummary` instances.
3. ``DIRTY_SET_TABLE_DDL`` and ``DIRTY_SET_REPLICATED_TABLE_DDL`` are
   syntactically plausible (contain required column names and engine directives).
4. :py:class:`ClickHouseAnalytics` delegates inserts to the supplied client
   without making any real network calls.
5. New public symbols are exported from ``stepback``.
"""
from __future__ import annotations

import datetime
import os
import tempfile
from typing import List
from unittest.mock import MagicMock, call

import pytest

import stepback
from stepback.analytics import (
    ClickHouseAnalytics,
    DIRTY_SET_REPLICATED_TABLE_DDL,
    DIRTY_SET_TABLE_DDL,
    DIRTY_SET_TABLE_NAME,
    DirtySetRecord,
    record_from_summary,
)
from stepback.divergence import compute_dirty_set
from stepback.substitutions import PromptSubstitution, ToolOutputSubstitution
from stepback import RecorderKey, record


# ------------------------------------------------------------------ helpers


def _fake_llm(model: str, messages: list) -> dict:
    blob = "|".join(f"{m['role']}={m['content']}" for m in messages)
    text = f"reply:{len(blob)}:{hash(blob) & 0xFFFF:04x}"
    return {
        "id": f"chatcmpl-{len(blob)}",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
        "usage": {
            "prompt_tokens": len(blob),
            "completion_tokens": len(text),
            "total_tokens": len(blob) + len(text),
        },
    }


def _make_trace(tmp_path):
    """Record a simple 3-step linear trace and return it."""
    path = str(tmp_path / "test.sb")
    key = RecorderKey.fresh()
    with record(path, key=key) as r:
        convo = [{"role": "user", "content": "hello"}]
        r.llm_call("gpt-4", convo, executor=_fake_llm)
        convo2 = [{"role": "user", "content": "world"}]
        r.llm_call("gpt-4", convo2, executor=_fake_llm)
        r.tool_call("search", {"q": "foo"}, executor=lambda n, a: {"result": n, "args": a})

    from stepback import replay
    return replay(path, hmac_key=key.hmac_key)


# ================================================================== DirtySetRecord


class TestDirtySetRecord:
    def test_fields_present(self):
        """All required fields exist on DirtySetRecord."""
        rec = DirtySetRecord(
            recorded_at=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
            trace_id="trace-abc",
            substitution_kind="PromptSubstitution",
            step_count=12,
            dirty_count=3,
            clean_count=9,
            branch_count=2,
            calls_saved=9,
        )
        assert rec.trace_id == "trace-abc"
        assert rec.substitution_kind == "PromptSubstitution"
        assert rec.step_count == 12
        assert rec.dirty_count == 3
        assert rec.clean_count == 9
        assert rec.branch_count == 2
        assert rec.calls_saved == 9
        assert rec.agent_id == ""
        assert rec.schema_version == 1

    def test_to_dict_keys(self):
        """to_dict returns all required column keys."""
        rec = DirtySetRecord(
            recorded_at=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
            trace_id="t",
            substitution_kind="",
            step_count=1,
            dirty_count=0,
            clean_count=1,
            branch_count=0,
            calls_saved=1,
        )
        d = rec.to_dict()
        for key in (
            "recorded_at",
            "trace_id",
            "substitution_kind",
            "step_count",
            "dirty_count",
            "clean_count",
            "branch_count",
            "calls_saved",
            "agent_id",
            "schema_version",
        ):
            assert key in d, f"Missing key: {key}"

    def test_column_names_order(self):
        """column_names() returns a list matching to_dict() keys."""
        rec = DirtySetRecord(
            recorded_at=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
            trace_id="t",
            substitution_kind="",
            step_count=0,
            dirty_count=0,
            clean_count=0,
            branch_count=0,
            calls_saved=0,
        )
        cols = DirtySetRecord.column_names()
        d = rec.to_dict()
        assert set(cols) == set(d.keys())

    def test_dirty_plus_clean_equals_step_count(self):
        """dirty_count + clean_count must equal step_count."""
        rec = DirtySetRecord(
            recorded_at=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
            trace_id="t",
            substitution_kind="",
            step_count=10,
            dirty_count=3,
            clean_count=7,
            branch_count=0,
            calls_saved=7,
        )
        assert rec.dirty_count + rec.clean_count == rec.step_count

    def test_agent_id_custom(self):
        """Custom agent_id is preserved."""
        rec = DirtySetRecord(
            recorded_at=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
            trace_id="t",
            substitution_kind="",
            step_count=0,
            dirty_count=0,
            clean_count=0,
            branch_count=0,
            calls_saved=0,
            agent_id="my-agent",
        )
        assert rec.agent_id == "my-agent"


# ================================================================== DDL strings


class TestDDL:
    def test_table_ddl_contains_required_columns(self):
        """MergeTree DDL contains all required column declarations."""
        required_columns = [
            "trace_id",
            "substitution_kind",
            "dirty_count",
            "clean_count",
            "branch_count",
            "calls_saved",
            "step_count",
            "recorded_at",
        ]
        for col in required_columns:
            assert col in DIRTY_SET_TABLE_DDL, (
                f"Column '{col}' missing from DIRTY_SET_TABLE_DDL"
            )

    def test_table_ddl_mergetree_engine(self):
        """MergeTree DDL uses MergeTree engine."""
        assert "MergeTree" in DIRTY_SET_TABLE_DDL
        assert "ReplicatedMergeTree" not in DIRTY_SET_TABLE_DDL

    def test_replicated_ddl_contains_required_columns(self):
        """ReplicatedMergeTree DDL contains all required column declarations."""
        required_columns = [
            "trace_id",
            "substitution_kind",
            "dirty_count",
            "clean_count",
            "branch_count",
            "calls_saved",
            "step_count",
            "recorded_at",
        ]
        for col in required_columns:
            assert col in DIRTY_SET_REPLICATED_TABLE_DDL, (
                f"Column '{col}' missing from DIRTY_SET_REPLICATED_TABLE_DDL"
            )

    def test_replicated_ddl_uses_replicated_mergetree(self):
        """ReplicatedMergeTree DDL uses the replicated engine."""
        assert "ReplicatedMergeTree" in DIRTY_SET_REPLICATED_TABLE_DDL

    def test_table_name_constant_present_in_ddl(self):
        """DIRTY_SET_TABLE_NAME appears in both DDL strings."""
        assert DIRTY_SET_TABLE_NAME in DIRTY_SET_TABLE_DDL
        assert DIRTY_SET_TABLE_NAME in DIRTY_SET_REPLICATED_TABLE_DDL

    def test_ddl_has_partition_by(self):
        """DDL contains a PARTITION BY clause."""
        assert "PARTITION BY" in DIRTY_SET_TABLE_DDL
        assert "PARTITION BY" in DIRTY_SET_REPLICATED_TABLE_DDL

    def test_ddl_has_order_by(self):
        """DDL contains an ORDER BY clause."""
        assert "ORDER BY" in DIRTY_SET_TABLE_DDL
        assert "ORDER BY" in DIRTY_SET_REPLICATED_TABLE_DDL

    def test_ddl_create_if_not_exists(self):
        """DDL uses CREATE TABLE IF NOT EXISTS."""
        assert "CREATE TABLE IF NOT EXISTS" in DIRTY_SET_TABLE_DDL
        assert "CREATE TABLE IF NOT EXISTS" in DIRTY_SET_REPLICATED_TABLE_DDL


# ================================================================== record_from_summary


class TestRecordFromSummary:
    def test_basic_linear_no_substitution(self, tmp_path):
        """record_from_summary on empty sigma produces substitution_kind=''."""
        trace = _make_trace(tmp_path)
        summary = compute_dirty_set(trace, [])
        ts = datetime.datetime(2026, 5, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        rec = record_from_summary(
            summary,
            trace_id="trace-xyz",
            substitutions=[],
            recorded_at=ts,
        )
        assert rec.trace_id == "trace-xyz"
        assert rec.substitution_kind == ""
        assert rec.step_count == summary.step_count
        assert rec.dirty_count == summary.dirty_count
        assert rec.clean_count == summary.clean_count
        assert rec.calls_saved == summary.calls_saved
        assert rec.recorded_at == ts

    def test_prompt_substitution_kind(self, tmp_path):
        """record_from_summary sets substitution_kind from first substitution."""
        trace = _make_trace(tmp_path)
        first_llm = next(s for s in trace.recorded_steps if s.get("step_kind") == "llm_call")
        subs = [PromptSubstitution(
            at_step=first_llm["step_id"],
            new_messages=[{"role": "user", "content": "changed"}],
        )]
        summary = compute_dirty_set(trace, subs)
        rec = record_from_summary(summary, trace_id="t", substitutions=subs)
        assert rec.substitution_kind == "PromptSubstitution"

    def test_tool_output_substitution_kind(self, tmp_path):
        """record_from_summary uses first substitution's class name."""
        trace = _make_trace(tmp_path)
        first_tool = next(s for s in trace.recorded_steps if s.get("step_kind") == "tool_call")
        subs = [ToolOutputSubstitution(
            at_step=first_tool["step_id"],
            fake_response={"result": "bar"},
        )]
        summary = compute_dirty_set(trace, subs)
        rec = record_from_summary(summary, trace_id="t", substitutions=subs)
        assert rec.substitution_kind == "ToolOutputSubstitution"

    def test_dirty_clean_invariant(self, tmp_path):
        """dirty_count + clean_count == step_count from summary."""
        trace = _make_trace(tmp_path)
        first_llm = next(s for s in trace.recorded_steps if s.get("step_kind") == "llm_call")
        subs = [PromptSubstitution(
            at_step=first_llm["step_id"],
            new_messages=[{"role": "user", "content": "x"}],
        )]
        summary = compute_dirty_set(trace, subs)
        rec = record_from_summary(summary, trace_id="t", substitutions=subs)
        assert rec.dirty_count + rec.clean_count == rec.step_count

    def test_calls_saved_equals_clean_count(self, tmp_path):
        """calls_saved equals clean_count."""
        trace = _make_trace(tmp_path)
        summary = compute_dirty_set(trace, [])
        rec = record_from_summary(summary, trace_id="t")
        assert rec.calls_saved == rec.clean_count

    def test_agent_id_propagated(self, tmp_path):
        """agent_id kwarg is propagated to the record."""
        trace = _make_trace(tmp_path)
        summary = compute_dirty_set(trace, [])
        rec = record_from_summary(summary, trace_id="t", agent_id="prod-agent-1")
        assert rec.agent_id == "prod-agent-1"

    def test_default_recorded_at_is_utc(self, tmp_path):
        """Default recorded_at is timezone-aware UTC."""
        trace = _make_trace(tmp_path)
        summary = compute_dirty_set(trace, [])
        rec = record_from_summary(summary, trace_id="t")
        assert rec.recorded_at.tzinfo is not None

    def test_branch_count_linear_trace(self, tmp_path):
        """branch_count is zero for a linear trace (no parallel_branch_open)."""
        trace = _make_trace(tmp_path)
        summary = compute_dirty_set(trace, [])
        rec = record_from_summary(summary, trace_id="t")
        assert rec.branch_count == 0

    def test_substitutions_none_gives_empty_kind(self, tmp_path):
        """Passing substitutions=None gives substitution_kind=''."""
        trace = _make_trace(tmp_path)
        summary = compute_dirty_set(trace, [])
        rec = record_from_summary(summary, trace_id="t", substitutions=None)
        assert rec.substitution_kind == ""

    def test_schema_version_is_one(self, tmp_path):
        """schema_version defaults to 1."""
        trace = _make_trace(tmp_path)
        summary = compute_dirty_set(trace, [])
        rec = record_from_summary(summary, trace_id="t")
        assert rec.schema_version == 1


# ================================================================== ClickHouseAnalytics


class TestClickHouseAnalytics:
    """Tests for ClickHouseAnalytics that use a mock client — no real ClickHouse."""

    def _make_record(self) -> DirtySetRecord:
        return DirtySetRecord(
            recorded_at=datetime.datetime(2026, 5, 1, tzinfo=datetime.timezone.utc),
            trace_id="trace-001",
            substitution_kind="PromptSubstitution",
            step_count=12,
            dirty_count=4,
            clean_count=8,
            branch_count=1,
            calls_saved=8,
        )

    def test_insert_record_calls_client_insert(self):
        """insert_record delegates to client.insert with correct args."""
        client = MagicMock()
        analytics = ClickHouseAnalytics(client)
        rec = self._make_record()
        analytics.insert_record(rec)

        assert client.insert.call_count == 1
        args, kwargs = client.insert.call_args
        assert args[0] == DIRTY_SET_TABLE_NAME
        # data is a list of one row
        assert isinstance(args[1], list)
        assert len(args[1]) == 1
        assert "column_names" in kwargs

    def test_insert_records_batch(self):
        """insert_records sends all rows in one client.insert call."""
        client = MagicMock()
        analytics = ClickHouseAnalytics(client)
        recs = [self._make_record(), self._make_record()]
        analytics.insert_records(recs)

        assert client.insert.call_count == 1
        args, _ = client.insert.call_args
        assert len(args[1]) == 2  # two data rows

    def test_insert_records_empty_is_noop(self):
        """insert_records with an empty list does not call client.insert."""
        client = MagicMock()
        analytics = ClickHouseAnalytics(client)
        analytics.insert_records([])
        client.insert.assert_not_called()

    def test_ensure_table_calls_command(self):
        """ensure_table calls client.command with the DDL."""
        client = MagicMock()
        analytics = ClickHouseAnalytics(client)
        analytics.ensure_table()
        assert client.command.call_count == 1
        ddl_arg = client.command.call_args[0][0]
        assert DIRTY_SET_TABLE_NAME in ddl_arg

    def test_ensure_table_replicated(self):
        """ensure_table(replicated=True) passes ReplicatedMergeTree DDL."""
        client = MagicMock()
        analytics = ClickHouseAnalytics(client)
        analytics.ensure_table(replicated=True)
        ddl_arg = client.command.call_args[0][0]
        assert "ReplicatedMergeTree" in ddl_arg

    def test_custom_table_name(self):
        """ClickHouseAnalytics respects a custom table_name."""
        client = MagicMock()
        analytics = ClickHouseAnalytics(client, table_name="my_custom_table")
        rec = self._make_record()
        analytics.insert_record(rec)
        args, _ = client.insert.call_args
        assert args[0] == "my_custom_table"

    def test_ensure_table_custom_name_in_ddl(self):
        """ensure_table substitutes custom table_name into the DDL."""
        client = MagicMock()
        analytics = ClickHouseAnalytics(client, table_name="test_summary")
        analytics.ensure_table()
        ddl_arg = client.command.call_args[0][0]
        assert "test_summary" in ddl_arg

    def test_column_names_match_ddl_columns(self):
        """DirtySetRecord.column_names() matches columns in DIRTY_SET_TABLE_DDL."""
        cols = DirtySetRecord.column_names()
        for col in cols:
            assert col in DIRTY_SET_TABLE_DDL, (
                f"Column '{col}' in DirtySetRecord not found in DDL"
            )

    def test_insert_record_data_values_correct(self):
        """Inserted row data matches record field values."""
        client = MagicMock()
        analytics = ClickHouseAnalytics(client)
        rec = self._make_record()
        analytics.insert_record(rec)
        _, kwargs = client.insert.call_args
        data = client.insert.call_args[0][1]
        column_names = kwargs["column_names"]
        row = dict(zip(column_names, data[0]))

        assert row["trace_id"] == rec.trace_id
        assert row["dirty_count"] == rec.dirty_count
        assert row["clean_count"] == rec.clean_count
        assert row["branch_count"] == rec.branch_count
        assert row["calls_saved"] == rec.calls_saved
        assert row["step_count"] == rec.step_count
        assert row["substitution_kind"] == rec.substitution_kind


# ================================================================== public API


class TestPublicExports:
    def test_analytics_symbols_exported_from_stepback(self):
        """All analytics symbols are exported from the top-level stepback package."""
        assert hasattr(stepback, "DirtySetRecord")
        assert hasattr(stepback, "ClickHouseAnalytics")
        assert hasattr(stepback, "DIRTY_SET_TABLE_DDL")
        assert hasattr(stepback, "DIRTY_SET_REPLICATED_TABLE_DDL")
        assert hasattr(stepback, "DIRTY_SET_TABLE_NAME")
        assert hasattr(stepback, "record_from_summary")

    def test_analytics_symbols_in_all(self):
        """All analytics symbols appear in stepback.__all__."""
        for name in (
            "DirtySetRecord",
            "ClickHouseAnalytics",
            "DIRTY_SET_TABLE_DDL",
            "DIRTY_SET_REPLICATED_TABLE_DDL",
            "DIRTY_SET_TABLE_NAME",
            "record_from_summary",
        ):
            assert name in stepback.__all__, f"{name} missing from __all__"


# ===========================================================================
# Step 139 — per-step trace query schema tests
# ===========================================================================

from stepback.analytics import (  # noqa: E402 (import after class definitions)
    TraceStepRecord,
    TRACE_STEP_TABLE_DDL,
    TRACE_STEP_REPLICATED_TABLE_DDL,
    TRACE_STEP_TABLE_NAME,
    step_record_from_dirty_entry,
    TraceStepAnalytics,
)
from stepback.divergence import DirtySetEntry


# ================================================================== TraceStepRecord


class TestTraceStepRecord:
    def _make_record(self, **overrides) -> TraceStepRecord:
        defaults = dict(
            recorded_at=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
            trace_id="trace-abc",
            step_id="step:1",
            step_index=0,
            incident_id="",
            step_kind="llm_call",
            model="gpt-4o",
            dirty=1,
            dirty_reason="input_drift",
            canonical_hash="a" * 64,
            cost_usd=0.012,
            policy_decision="",
        )
        defaults.update(overrides)
        return TraceStepRecord(**defaults)

    def test_all_fields_present(self):
        """TraceStepRecord has all required fields."""
        rec = self._make_record()
        assert rec.trace_id == "trace-abc"
        assert rec.step_id == "step:1"
        assert rec.step_index == 0
        assert rec.incident_id == ""
        assert rec.step_kind == "llm_call"
        assert rec.model == "gpt-4o"
        assert rec.dirty == 1
        assert rec.dirty_reason == "input_drift"
        assert rec.canonical_hash == "a" * 64
        assert rec.cost_usd == pytest.approx(0.012)
        assert rec.policy_decision == ""
        assert rec.schema_version == 1

    def test_to_dict_has_all_keys(self):
        """to_dict() includes all expected column keys."""
        rec = self._make_record()
        d = rec.to_dict()
        required_keys = [
            "recorded_at", "trace_id", "step_id", "step_index",
            "incident_id", "step_kind", "model", "dirty", "dirty_reason",
            "canonical_hash", "cost_usd", "policy_decision", "schema_version",
        ]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"

    def test_column_names_match_to_dict(self):
        """column_names() set matches to_dict() key set."""
        rec = self._make_record()
        assert set(TraceStepRecord.column_names()) == set(rec.to_dict().keys())

    def test_column_names_order_matches_ddl_declaration_order(self):
        """column_names() from dataclass fields matches to_dict() insertion order."""
        rec = self._make_record()
        cols = TraceStepRecord.column_names()
        d = rec.to_dict()
        # Verify all column_names appear in to_dict()
        assert set(cols) == set(d.keys())

    def test_dirty_zero_for_clean_step(self):
        """dirty=0 for a clean (cache-hit) step."""
        rec = self._make_record(dirty=0, dirty_reason="")
        assert rec.dirty == 0
        assert rec.dirty_reason == ""

    def test_incident_id_preserved(self):
        """incident_id is preserved in the record."""
        rec = self._make_record(incident_id="INC-2026-001")
        assert rec.incident_id == "INC-2026-001"
        assert rec.to_dict()["incident_id"] == "INC-2026-001"

    def test_policy_decision_values(self):
        """policy_decision can hold all expected policy audit outcomes."""
        for decision in ("unchanged", "divergent", "newly_blocked", "newly_allowed", ""):
            rec = self._make_record(policy_decision=decision)
            assert rec.policy_decision == decision

    def test_schema_version_defaults_to_one(self):
        """schema_version defaults to 1."""
        rec = self._make_record()
        assert rec.schema_version == 1

    def test_step_index_ordering_safe(self):
        """step_index is numeric, so step:10 orders correctly after step:9."""
        rec9 = self._make_record(step_id="step:9", step_index=9)
        rec10 = self._make_record(step_id="step:10", step_index=10)
        assert rec9.step_index < rec10.step_index

    def test_empty_canonical_hash_allowed(self):
        """canonical_hash can be empty string (field unavailable)."""
        rec = self._make_record(canonical_hash="")
        assert rec.canonical_hash == ""

    def test_tool_call_has_empty_model(self):
        """Tool-call steps have empty model string."""
        rec = self._make_record(step_kind="tool_call", model="")
        assert rec.model == ""
        assert rec.step_kind == "tool_call"


# ================================================================== TRACE_STEP DDL


class TestTraceStepDDL:
    _REQUIRED_COLS = [
        "trace_id", "step_id", "step_index", "incident_id",
        "step_kind", "model", "dirty", "dirty_reason",
        "canonical_hash", "cost_usd", "policy_decision",
        "recorded_at", "schema_version",
    ]

    def test_local_ddl_contains_required_columns(self):
        """MergeTree DDL contains all required column declarations."""
        for col in self._REQUIRED_COLS:
            assert col in TRACE_STEP_TABLE_DDL, (
                f"Column '{col}' missing from TRACE_STEP_TABLE_DDL"
            )

    def test_replicated_ddl_contains_required_columns(self):
        """ReplicatedMergeTree DDL contains all required column declarations."""
        for col in self._REQUIRED_COLS:
            assert col in TRACE_STEP_REPLICATED_TABLE_DDL, (
                f"Column '{col}' missing from TRACE_STEP_REPLICATED_TABLE_DDL"
            )

    def test_local_ddl_uses_mergetree(self):
        """Local DDL uses MergeTree engine (not replicated)."""
        assert "MergeTree" in TRACE_STEP_TABLE_DDL
        assert "ReplicatedMergeTree" not in TRACE_STEP_TABLE_DDL

    def test_replicated_ddl_uses_replicated_mergetree(self):
        """Replicated DDL uses ReplicatedMergeTree engine."""
        assert "ReplicatedMergeTree" in TRACE_STEP_REPLICATED_TABLE_DDL

    def test_table_name_in_both_ddls(self):
        """TRACE_STEP_TABLE_NAME appears in both DDL strings."""
        assert TRACE_STEP_TABLE_NAME in TRACE_STEP_TABLE_DDL
        assert TRACE_STEP_TABLE_NAME in TRACE_STEP_REPLICATED_TABLE_DDL

    def test_ddl_has_partition_by(self):
        """DDL contains PARTITION BY clause."""
        assert "PARTITION BY" in TRACE_STEP_TABLE_DDL
        assert "PARTITION BY" in TRACE_STEP_REPLICATED_TABLE_DDL

    def test_ddl_has_order_by_with_step_index(self):
        """ORDER BY uses step_index for correct numeric ordering."""
        assert "step_index" in TRACE_STEP_TABLE_DDL
        assert "ORDER BY" in TRACE_STEP_TABLE_DDL

    def test_ddl_create_if_not_exists(self):
        """DDL uses CREATE TABLE IF NOT EXISTS."""
        assert "CREATE TABLE IF NOT EXISTS" in TRACE_STEP_TABLE_DDL
        assert "CREATE TABLE IF NOT EXISTS" in TRACE_STEP_REPLICATED_TABLE_DDL

    def test_column_names_match_ddl(self):
        """Every TraceStepRecord column appears in the DDL."""
        for col in TraceStepRecord.column_names():
            assert col in TRACE_STEP_TABLE_DDL, (
                f"Column '{col}' in TraceStepRecord not found in TRACE_STEP_TABLE_DDL"
            )


# ================================================================== step_record_from_dirty_entry


class TestStepRecordFromDirtyEntry:
    def _make_entry(self, **kwargs) -> DirtySetEntry:
        defaults = dict(
            step_id="step:1",
            kind="llm_call",
            dirty=True,
            cache_hit=False,
            dirty_reason="input_drift",
            parent_step_id=None,
            parent_step_ids=frozenset(),
        )
        defaults.update(kwargs)
        return DirtySetEntry(**defaults)

    def _make_step(self, **kwargs) -> dict:
        defaults = dict(
            step_id="step:1",
            step_kind="llm_call",
            model="gpt-4o",
            inputs_hash="b" * 64,
            cost_usd=0.005,
        )
        defaults.update(kwargs)
        return defaults

    def test_basic_dirty_entry(self):
        """Factory builds a correct TraceStepRecord from a dirty entry."""
        ts = datetime.datetime(2026, 5, 1, tzinfo=datetime.timezone.utc)
        entry = self._make_entry()
        step = self._make_step()
        rec = step_record_from_dirty_entry(
            entry, step, trace_id="trace-x", step_index=1, recorded_at=ts
        )
        assert rec.trace_id == "trace-x"
        assert rec.step_id == "step:1"
        assert rec.step_index == 1
        assert rec.step_kind == "llm_call"
        assert rec.model == "gpt-4o"
        assert rec.dirty == 1
        assert rec.dirty_reason == "input_drift"
        assert rec.canonical_hash == "b" * 64
        assert rec.cost_usd == pytest.approx(0.005)
        assert rec.policy_decision == ""
        assert rec.incident_id == ""
        assert rec.recorded_at == ts

    def test_clean_entry_gives_dirty_zero(self):
        """A clean DirtySetEntry produces dirty=0 and empty dirty_reason."""
        entry = self._make_entry(dirty=False, cache_hit=True, dirty_reason=None)
        step = self._make_step()
        rec = step_record_from_dirty_entry(entry, step, trace_id="t", step_index=0)
        assert rec.dirty == 0
        assert rec.dirty_reason == ""

    def test_tool_call_no_model(self):
        """Tool-call step with no model field yields empty model."""
        entry = self._make_entry(kind="tool_call")
        step = self._make_step(step_kind="tool_call", model=None)
        rec = step_record_from_dirty_entry(entry, step, trace_id="t", step_index=2)
        assert rec.step_kind == "tool_call"
        assert rec.model == ""

    def test_incident_id_propagated(self):
        """incident_id kwarg is written to the record."""
        entry = self._make_entry()
        step = self._make_step()
        rec = step_record_from_dirty_entry(
            entry, step, trace_id="t", step_index=0, incident_id="INC-99"
        )
        assert rec.incident_id == "INC-99"

    def test_policy_decision_propagated(self):
        """policy_decision kwarg is written to the record."""
        entry = self._make_entry()
        step = self._make_step()
        rec = step_record_from_dirty_entry(
            entry, step, trace_id="t", step_index=0, policy_decision="newly_blocked"
        )
        assert rec.policy_decision == "newly_blocked"

    def test_missing_inputs_hash_yields_empty_canonical_hash(self):
        """Step dict without inputs_hash yields canonical_hash=''."""
        entry = self._make_entry()
        step = {k: v for k, v in self._make_step().items() if k != "inputs_hash"}
        rec = step_record_from_dirty_entry(entry, step, trace_id="t", step_index=0)
        assert rec.canonical_hash == ""

    def test_missing_cost_usd_yields_zero(self):
        """Step dict without cost_usd yields cost_usd=0.0."""
        entry = self._make_entry()
        step = {k: v for k, v in self._make_step().items() if k != "cost_usd"}
        rec = step_record_from_dirty_entry(entry, step, trace_id="t", step_index=0)
        assert rec.cost_usd == 0.0

    def test_default_recorded_at_is_utc_aware(self):
        """Default recorded_at is timezone-aware UTC."""
        entry = self._make_entry()
        step = self._make_step()
        rec = step_record_from_dirty_entry(entry, step, trace_id="t", step_index=0)
        assert rec.recorded_at.tzinfo is not None

    def test_substituted_dirty_reason(self):
        """substituted dirty_reason is preserved."""
        entry = self._make_entry(dirty_reason="substituted")
        step = self._make_step()
        rec = step_record_from_dirty_entry(entry, step, trace_id="t", step_index=0)
        assert rec.dirty_reason == "substituted"

    def test_parent_dirty_reason(self):
        """parent_dirty dirty_reason is preserved."""
        entry = self._make_entry(dirty_reason="parent_dirty", parent_step_id="step:0")
        step = self._make_step()
        rec = step_record_from_dirty_entry(entry, step, trace_id="t", step_index=1)
        assert rec.dirty_reason == "parent_dirty"

    def test_full_trace_batch(self, tmp_path):
        """step_record_from_dirty_entry works for every step in a recorded trace."""
        trace = _make_trace(tmp_path)
        from stepback.divergence import compute_dirty_set
        summary = compute_dirty_set(trace, [])
        records = [
            step_record_from_dirty_entry(
                entry,
                trace.recorded_steps[i],
                trace_id="batch-trace",
                step_index=i,
            )
            for i, entry in enumerate(summary.entries)
        ]
        assert len(records) == len(summary.entries)
        for i, rec in enumerate(records):
            assert rec.step_index == i
            assert rec.trace_id == "batch-trace"


# ================================================================== TraceStepAnalytics


class TestTraceStepAnalytics:
    def _make_record(self) -> TraceStepRecord:
        return TraceStepRecord(
            recorded_at=datetime.datetime(2026, 5, 1, tzinfo=datetime.timezone.utc),
            trace_id="trace-001",
            step_id="step:3",
            step_index=3,
            incident_id="",
            step_kind="llm_call",
            model="claude-3-opus",
            dirty=1,
            dirty_reason="input_drift",
            canonical_hash="c" * 64,
            cost_usd=0.023,
            policy_decision="",
        )

    def test_insert_record_calls_client_insert(self):
        """insert_record delegates to client.insert."""
        client = MagicMock()
        analytics = TraceStepAnalytics(client)
        analytics.insert_record(self._make_record())
        assert client.insert.call_count == 1
        args, kwargs = client.insert.call_args
        assert args[0] == TRACE_STEP_TABLE_NAME
        assert len(args[1]) == 1  # one row
        assert "column_names" in kwargs

    def test_insert_records_batch(self):
        """insert_records sends all rows in a single client.insert call."""
        client = MagicMock()
        analytics = TraceStepAnalytics(client)
        analytics.insert_records([self._make_record(), self._make_record()])
        assert client.insert.call_count == 1
        args, _ = client.insert.call_args
        assert len(args[1]) == 2

    def test_insert_records_empty_noop(self):
        """insert_records([]) does not call client.insert."""
        client = MagicMock()
        analytics = TraceStepAnalytics(client)
        analytics.insert_records([])
        client.insert.assert_not_called()

    def test_ensure_table_calls_command(self):
        """ensure_table() calls client.command with the DDL."""
        client = MagicMock()
        analytics = TraceStepAnalytics(client)
        analytics.ensure_table()
        assert client.command.call_count == 1
        ddl_arg = client.command.call_args[0][0]
        assert "CREATE TABLE" in ddl_arg

    def test_ensure_table_replicated(self):
        """ensure_table(replicated=True) passes ReplicatedMergeTree DDL."""
        client = MagicMock()
        analytics = TraceStepAnalytics(client)
        analytics.ensure_table(replicated=True)
        ddl_arg = client.command.call_args[0][0]
        assert "ReplicatedMergeTree" in ddl_arg

    def test_custom_table_name_used_in_insert(self):
        """Custom table_name is passed to client.insert."""
        client = MagicMock()
        analytics = TraceStepAnalytics(client, table_name="my_step_table")
        analytics.insert_record(self._make_record())
        args, _ = client.insert.call_args
        assert args[0] == "my_step_table"

    def test_custom_table_name_in_ensure_table_ddl(self):
        """ensure_table substitutes custom table_name into the DDL."""
        client = MagicMock()
        analytics = TraceStepAnalytics(client, table_name="custom_steps")
        analytics.ensure_table()
        ddl_arg = client.command.call_args[0][0]
        assert "custom_steps" in ddl_arg

    def test_insert_data_values_correct(self):
        """Inserted row data values match the record fields."""
        client = MagicMock()
        analytics = TraceStepAnalytics(client)
        rec = self._make_record()
        analytics.insert_record(rec)
        _, kwargs = client.insert.call_args
        data = client.insert.call_args[0][1]
        column_names = kwargs["column_names"]
        row = dict(zip(column_names, data[0]))

        assert row["trace_id"] == rec.trace_id
        assert row["step_id"] == rec.step_id
        assert row["step_index"] == rec.step_index
        assert row["step_kind"] == rec.step_kind
        assert row["model"] == rec.model
        assert row["dirty"] == rec.dirty
        assert row["dirty_reason"] == rec.dirty_reason
        assert row["canonical_hash"] == rec.canonical_hash
        assert row["cost_usd"] == pytest.approx(rec.cost_usd)
        assert row["policy_decision"] == rec.policy_decision
        assert row["incident_id"] == rec.incident_id


# ================================================================== Public exports (Step 139)


class TestStep139PublicExports:
    def test_trace_step_symbols_exported_from_stepback(self):
        """All Step 139 symbols are accessible from stepback package."""
        assert hasattr(stepback, "TraceStepRecord")
        assert hasattr(stepback, "TRACE_STEP_TABLE_DDL")
        assert hasattr(stepback, "TRACE_STEP_REPLICATED_TABLE_DDL")
        assert hasattr(stepback, "TRACE_STEP_TABLE_NAME")
        assert hasattr(stepback, "step_record_from_dirty_entry")
        assert hasattr(stepback, "TraceStepAnalytics")

    def test_trace_step_symbols_in_all(self):
        """All Step 139 symbols appear in stepback.__all__."""
        for name in (
            "TraceStepRecord",
            "TRACE_STEP_TABLE_DDL",
            "TRACE_STEP_REPLICATED_TABLE_DDL",
            "TRACE_STEP_TABLE_NAME",
            "step_record_from_dirty_entry",
            "TraceStepAnalytics",
        ):
            assert name in stepback.__all__, f"{name} missing from __all__"
