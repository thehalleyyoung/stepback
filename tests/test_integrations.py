"""Tests for stepback ecosystem integrations.

Covers three integration modules:
* stepback.integrations.toolwarden  — Warden shim + AuditLog importer
* stepback.integrations.flowwarden  — RunLog importer + recorder attachment
* stepback.integrations.ragdoctor   — RagDoctorShim + Trace importer

All tests use duck-typed fakes shaped like the real companion-library APIs.
No network calls. No actual ragdoctor / flowwarden / toolwarden execution.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

from stepback import record, replay
from stepback.integrations.toolwarden import (
    WardenRecorderShim,
    import_toolwarden_audit,
)
from stepback.integrations.flowwarden import (
    import_flowwarden_runlog,
    FlowWardenRecorderAttachment,
)
from stepback.integrations.ragdoctor import (
    RagDoctorShim,
    import_ragdoctor_trace,
    record_rag_query,
)


# ======================================================================
# Shared helpers
# ======================================================================

def _read_steps(path: str) -> List[dict]:
    """Return all recorded steps from a .sb trace file."""
    t = replay(path)
    return list(t.recorded_steps)


def _sb_path(tmp_path: Path, name: str) -> str:
    return str(tmp_path / name)


# ======================================================================
# § toolwarden fakes
# ======================================================================

class _FakeOutcome:
    def __init__(self, value: str):
        self.value = value

    def __str__(self) -> str:
        return self.value


@dataclass
class _FakeDecision:
    outcome: _FakeOutcome
    tool: str
    args: Dict[str, Any]
    reasons: List[str] = field(default_factory=list)
    matched_rules: List[str] = field(default_factory=list)
    rewritten_args: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.outcome.value in ("allow", "redact")

    @property
    def effective_args(self) -> Dict[str, Any]:
        return self.rewritten_args if self.rewritten_args is not None else self.args


class _FakeWardenError(Exception):
    def __init__(self, decision: _FakeDecision):
        self.decision = decision
        super().__init__(f"DENY {decision.tool}: {'; '.join(decision.reasons)}")


class _FakePolicy:
    def __init__(self, outcome: str = "allow", reasons: Optional[List[str]] = None):
        self._outcome = outcome
        self._reasons = reasons or []

    def evaluate(self, tool: str, args: Dict[str, Any], ctx: Any = None) -> _FakeDecision:
        return _FakeDecision(
            outcome=_FakeOutcome(self._outcome),
            tool=tool,
            args=args,
            reasons=list(self._reasons),
            metadata={"decision_id": f"fake-{tool}-{self._outcome}"},
        )


class _FakeWarden:
    """Duck-typed Warden backed by a _FakePolicy."""

    def __init__(self, policy: _FakePolicy):
        self._policy = policy

    def check(self, tool: str, args: Dict[str, Any], ctx: Any = None) -> _FakeDecision:
        return self._policy.evaluate(tool, args, ctx)

    def invoke(
        self,
        tool: str,
        args: Dict[str, Any],
        executor: Callable,
        ctx: Any = None,
        **kwargs: Any,
    ) -> Any:
        decision = self.check(tool, args, ctx)
        if not decision.allowed:
            raise _FakeWardenError(decision)
        return executor(tool, decision.effective_args)


@dataclass
class _FakeAuditEntry:
    ts: float
    principal: str
    tool: str
    outcome: str
    reasons: List[str]
    matched_rules: List[str]
    args: Dict[str, Any]
    rewritten_args: Optional[Dict[str, Any]] = None
    channel: str = "request"
    policy_version_id: Optional[str] = None
    decision_id: Optional[str] = None
    parent_decision_id: Optional[str] = None
    agent_chain: List[str] = field(default_factory=list)


@dataclass
class _FakeAuditLog:
    entries: List[_FakeAuditEntry] = field(default_factory=list)


# ======================================================================
# § toolwarden — WardenRecorderShim tests
# ======================================================================


class TestWardenRecorderShim:
    def test_allowed_call_records_step(self, tmp_path):
        """ALLOW: a step is recorded and the executor result is returned."""
        warden = _FakeWarden(_FakePolicy("allow"))
        with record(_sb_path(tmp_path, "t.sb")) as rec:
            shim = WardenRecorderShim(warden, rec, policy_version="pol@v1")
            result = shim.invoke("web_search", {"q": "hello"}, executor=lambda n, a: {"results": ["r1"]})
        assert result == {"results": ["r1"]}
        steps = _read_steps(_sb_path(tmp_path, "t.sb"))
        assert len(steps) == 1
        assert steps[0]["name"] == "web_search"
        assert steps[0]["step_kind"] == "tool_call"
        # recorder.tool_call wraps outputs inside {"result": ...}
        out = steps[0]["outputs"]["result"]
        assert out["outcome"] == "allow"
        assert out["result"] == {"results": ["r1"]}

    def test_policy_version_in_inputs(self, tmp_path):
        """Policy version appears in step inputs (in arguments) for cache-identity purposes."""
        warden = _FakeWarden(_FakePolicy("allow"))
        with record(_sb_path(tmp_path, "t.sb")) as rec:
            shim = WardenRecorderShim(warden, rec, policy_version="pol@v42")
            shim.invoke("lookup", {"id": "1"}, executor=lambda n, a: {"val": "x"})
        steps = _read_steps(_sb_path(tmp_path, "t.sb"))
        # policy_version is in inputs["arguments"] which is part of inputs_hash
        assert steps[0]["inputs"]["arguments"]["policy_version"] == "pol@v42"

    def test_denied_call_records_step_then_raises(self, tmp_path):
        """DENY: step is recorded with outcome='deny' and an exception is raised."""
        warden = _FakeWarden(_FakePolicy("deny", reasons=["blocked by rule"]))
        with record(_sb_path(tmp_path, "t.sb")) as rec:
            shim = WardenRecorderShim(warden, rec, policy_version="pol@v1")
            with pytest.raises(Exception) as exc_info:
                shim.invoke("send_email", {"to": "x@y.com"}, executor=lambda n, a: {})
        assert "deny" in str(exc_info.value).lower() or "DENY" in str(exc_info.value)
        steps = _read_steps(_sb_path(tmp_path, "t.sb"))
        assert len(steps) == 1
        # recorder.tool_call wraps outputs inside {"result": ...}
        out = steps[0]["outputs"]["result"]
        assert out["outcome"] == "deny"
        assert "blocked by rule" in out["reasons"]

    def test_denied_call_executor_not_invoked(self, tmp_path):
        """Executor must NOT be called when the call is denied."""
        executed = []
        warden = _FakeWarden(_FakePolicy("deny"))
        with record(_sb_path(tmp_path, "t.sb")) as rec:
            shim = WardenRecorderShim(warden, rec)
            try:
                shim.invoke("rm_rf", {}, executor=lambda n, a: executed.append(1) or {})
            except Exception:
                pass
        assert executed == [], "executor was called despite DENY"

    def test_redact_outcome_recorded(self, tmp_path):
        """REDACT: step recorded with outcome='redact', rewritten args used."""
        policy = _FakePolicy("redact")

        class _RedactWarden(_FakeWarden):
            def check(self, tool, args, ctx=None):
                d = super().check(tool, args, ctx)
                d.rewritten_args = {"q": "[REDACTED]"}
                return d

        warden = _RedactWarden(policy)
        received_args = []
        def _exec(n, a):
            received_args.append(dict(a))
            return {"ok": True}

        with record(_sb_path(tmp_path, "t.sb")) as rec:
            shim = WardenRecorderShim(warden, rec)
            shim.invoke("web_search", {"q": "secret"}, executor=_exec)
        steps = _read_steps(_sb_path(tmp_path, "t.sb"))
        # tw_args_rewritten is in the arguments dict
        assert steps[0]["inputs"]["arguments"]["tw_args_rewritten"] is True
        out = steps[0]["outputs"]["result"]
        assert out["outcome"] == "redact"
        # Executor received rewritten args.
        assert received_args[0].get("q") == "[REDACTED]"

    def test_default_policy_version_when_none(self, tmp_path):
        """When policy_version=None, a default sentinel is used."""
        warden = _FakeWarden(_FakePolicy("allow"))
        with record(_sb_path(tmp_path, "t.sb")) as rec:
            shim = WardenRecorderShim(warden, rec)
            shim.invoke("lookup", {}, executor=lambda n, a: {})
        steps = _read_steps(_sb_path(tmp_path, "t.sb"))
        pv = steps[0]["inputs"]["arguments"]["policy_version"]
        assert pv and pv != ""  # sentinel must be non-empty

    def test_check_records_router_step(self, tmp_path):
        """check() records a router step (no side effects)."""
        warden = _FakeWarden(_FakePolicy("allow"))
        with record(_sb_path(tmp_path, "t.sb")) as rec:
            shim = WardenRecorderShim(warden, rec)
            shim.check("read_file", {"path": "/etc/passwd"})
        steps = _read_steps(_sb_path(tmp_path, "t.sb"))
        assert len(steps) == 1
        assert steps[0]["step_kind"] == "router"
        assert "toolwarden.check" in steps[0]["name"]

    def test_multiple_calls_ordered(self, tmp_path):
        """Multiple invoke() calls produce steps in order."""
        warden = _FakeWarden(_FakePolicy("allow"))
        with record(_sb_path(tmp_path, "t.sb")) as rec:
            shim = WardenRecorderShim(warden, rec)
            shim.invoke("tool_a", {}, executor=lambda n, a: {"v": 1})
            shim.invoke("tool_b", {}, executor=lambda n, a: {"v": 2})
        steps = _read_steps(_sb_path(tmp_path, "t.sb"))
        assert [s["name"] for s in steps] == ["tool_a", "tool_b"]


# ======================================================================
# § toolwarden — import_toolwarden_audit tests
# ======================================================================


class TestImportToolwardenAudit:
    def _make_audit_log(self, entries: List[_FakeAuditEntry]) -> _FakeAuditLog:
        return _FakeAuditLog(entries=entries)

    def test_empty_audit_log(self, tmp_path):
        audit = self._make_audit_log([])
        report = import_toolwarden_audit(audit, _sb_path(tmp_path, "a.sb"))
        assert report.step_count == 0
        assert report.source_format == "toolwarden_audit"
        assert os.path.exists(_sb_path(tmp_path, "a.sb"))

    def test_single_allow_entry(self, tmp_path):
        entry = _FakeAuditEntry(
            ts=1000.0,
            principal="agent",
            tool="web_search",
            outcome="allow",
            reasons=[],
            matched_rules=["default_allow"],
            args={"q": "hello"},
            policy_version_id="pol@v1",
            decision_id="dec-001",
        )
        audit = self._make_audit_log([entry])
        report = import_toolwarden_audit(audit, _sb_path(tmp_path, "a.sb"))
        assert report.step_count == 1
        assert report.kind_counts.get("tool_call", 0) == 1

        steps = _read_steps(_sb_path(tmp_path, "a.sb"))
        assert len(steps) == 1
        s = steps[0]
        assert s["name"] == "web_search"
        assert s["step_kind"] == "tool_call"
        assert s["inputs"]["policy_version"] == "pol@v1"
        assert s["inputs"]["tw_outcome"] == "allow"
        assert s["outputs"]["outcome"] == "allow"
        assert s["outputs"]["decision_id"] == "dec-001"

    def test_single_deny_entry(self, tmp_path):
        entry = _FakeAuditEntry(
            ts=2000.0,
            principal="agent",
            tool="delete_all",
            outcome="deny",
            reasons=["high_risk"],
            matched_rules=["deny_destructive"],
            args={},
            policy_version_id="pol@v2",
            decision_id="dec-002",
        )
        audit = self._make_audit_log([entry])
        report = import_toolwarden_audit(audit, _sb_path(tmp_path, "a.sb"))
        assert report.step_count == 1

        steps = _read_steps(_sb_path(tmp_path, "a.sb"))
        s = steps[0]
        assert s["outputs"]["outcome"] == "deny"
        assert "high_risk" in s["outputs"]["reasons"]

    def test_policy_version_in_inputs_for_cache_identity(self, tmp_path):
        """Two entries with different policy versions produce different inputs_hash."""
        e1 = _FakeAuditEntry(1000.0, "a", "t", "allow", [], [], {}, policy_version_id="v1", decision_id="d1")
        e2 = _FakeAuditEntry(1001.0, "a", "t", "allow", [], [], {}, policy_version_id="v2", decision_id="d2")
        audit = self._make_audit_log([e1, e2])
        report = import_toolwarden_audit(audit, _sb_path(tmp_path, "a.sb"))
        steps = _read_steps(_sb_path(tmp_path, "a.sb"))
        assert steps[0]["inputs_hash"] != steps[1]["inputs_hash"]

    def test_multiple_entries_in_order(self, tmp_path):
        entries = [
            _FakeAuditEntry(1000.0 + i, "a", f"tool_{i}", "allow", [], [], {}, decision_id=f"d{i}")
            for i in range(5)
        ]
        audit = self._make_audit_log(entries)
        report = import_toolwarden_audit(audit, _sb_path(tmp_path, "a.sb"))
        assert report.step_count == 5
        steps = _read_steps(_sb_path(tmp_path, "a.sb"))
        assert [s["name"] for s in steps] == [f"tool_{i}" for i in range(5)]

    def test_parent_decision_id_linkage(self, tmp_path):
        """parent_decision_id is resolved to parent_step_id."""
        e_root = _FakeAuditEntry(1000.0, "a", "root_tool", "allow", [], [], {}, decision_id="root")
        e_child = _FakeAuditEntry(1001.0, "a", "child_tool", "allow", [], [], {}, decision_id="child",
                                   parent_decision_id="root")
        audit = self._make_audit_log([e_root, e_child])
        report = import_toolwarden_audit(audit, _sb_path(tmp_path, "a.sb"))
        steps = _read_steps(_sb_path(tmp_path, "a.sb"))
        assert steps[1]["parent_step_id"] == steps[0]["step_id"]

    def test_iterable_of_entries_accepted(self, tmp_path):
        """import_toolwarden_audit accepts a plain iterable, not just AuditLog."""
        entries = [_FakeAuditEntry(1000.0, "a", "t", "allow", [], [], {}, decision_id="d1")]
        report = import_toolwarden_audit(entries, _sb_path(tmp_path, "a.sb"))
        assert report.step_count == 1

    def test_args_are_sanitized_to_json_safe(self, tmp_path):
        """Non-JSON-native args (bytes, nested sets) are converted to strings."""
        entry = _FakeAuditEntry(
            ts=1000.0,
            principal="a",
            tool="t",
            outcome="allow",
            reasons=[],
            matched_rules=[],
            args={"data": b"\x00\x01"},  # bytes not JSON-safe
            decision_id="d1",
        )
        audit = self._make_audit_log([entry])
        # Should not raise
        report = import_toolwarden_audit(audit, _sb_path(tmp_path, "a.sb"))
        assert report.step_count == 1

    def test_channel_in_inputs(self, tmp_path):
        """response-channel entries are recorded with tw_channel='response'."""
        entry = _FakeAuditEntry(1000.0, "a", "t", "allow", [], [], {}, channel="response", decision_id="d1")
        audit = self._make_audit_log([entry])
        report = import_toolwarden_audit(audit, _sb_path(tmp_path, "a.sb"))
        steps = _read_steps(_sb_path(tmp_path, "a.sb"))
        assert steps[0]["inputs"]["tw_channel"] == "response"


# ======================================================================
# § flowwarden fakes
# ======================================================================


@dataclass
class _FakeRunLog:
    events: List[Dict[str, Any]] = field(default_factory=list)

    def append(self, **event: Any) -> None:
        import time as _time
        event.setdefault("ts", _time.time())
        self.events.append(event)


def _make_runlog(*events: Dict[str, Any]) -> _FakeRunLog:
    rl = _FakeRunLog()
    for ev in events:
        rl.events.append(ev)
    return rl


# ======================================================================
# § flowwarden — import_flowwarden_runlog tests
# ======================================================================


class TestImportFlowwardenRunlog:
    def test_empty_runlog(self, tmp_path):
        rl = _make_runlog()
        report = import_flowwarden_runlog(rl, _sb_path(tmp_path, "fw.sb"))
        assert report.step_count == 0
        assert os.path.exists(_sb_path(tmp_path, "fw.sb"))

    def test_run_start_only(self, tmp_path):
        """run_start is skipped; no steps emitted."""
        rl = _make_runlog({"event": "run_start", "run_id": "abc", "policy_version": "fw@v1", "ts": 1.0})
        report = import_flowwarden_runlog(rl, _sb_path(tmp_path, "fw.sb"))
        assert report.step_count == 0

    def test_tool_call_step(self, tmp_path):
        rl = _make_runlog(
            {"event": "run_start", "run_id": "r1", "policy_version": "fw@v1", "ts": 1.0},
            {
                "event": "tool_call",
                "step": 1,
                "node_id": "t1",
                "name": "lookup_customer",
                "labels_in": [],
                "labels_out": ["pii"],
                "policy_decision": "ALLOW",
                "result_preview": "Customer c-42",
                "ts": 2.0,
            },
        )
        report = import_flowwarden_runlog(rl, _sb_path(tmp_path, "fw.sb"))
        assert report.step_count == 1
        assert report.kind_counts.get("tool_call", 0) == 1

        steps = _read_steps(_sb_path(tmp_path, "fw.sb"))
        s = steps[0]
        assert s["name"] == "lookup_customer"
        assert s["step_kind"] == "tool_call"
        assert s["inputs"]["fw_labels_in"] == []
        assert s["outputs"]["fw_labels_out"] == ["pii"]
        assert s["outputs"]["fw_policy_decision"] == "ALLOW"

    def test_labels_in_inputs_for_cache_identity(self, tmp_path):
        """Different labels_in produce different inputs_hash."""
        rl1 = _make_runlog(
            {"event": "tool_call", "name": "t", "labels_in": ["pii"], "labels_out": [], "ts": 1.0}
        )
        rl2 = _make_runlog(
            {"event": "tool_call", "name": "t", "labels_in": [], "labels_out": [], "ts": 1.0}
        )
        r1 = import_flowwarden_runlog(rl1, _sb_path(tmp_path, "fw1.sb"))
        r2 = import_flowwarden_runlog(rl2, _sb_path(tmp_path, "fw2.sb"))
        s1 = _read_steps(_sb_path(tmp_path, "fw1.sb"))[0]
        s2 = _read_steps(_sb_path(tmp_path, "fw2.sb"))[0]
        assert s1["inputs_hash"] != s2["inputs_hash"]

    def test_policy_version_propagated_to_steps(self, tmp_path):
        """policy_version from run_start appears in every subsequent step."""
        rl = _make_runlog(
            {"event": "run_start", "policy_version": "fw@v99", "ts": 1.0},
            {"event": "tool_call", "name": "t", "labels_in": [], "labels_out": [], "ts": 2.0},
            {"event": "llm_call", "model": "gpt-4o", "labels_in": ["pii"], "labels_out": ["pii"], "ts": 3.0},
        )
        report = import_flowwarden_runlog(rl, _sb_path(tmp_path, "fw.sb"))
        steps = _read_steps(_sb_path(tmp_path, "fw.sb"))
        for s in steps:
            assert s["inputs"].get("fw_policy_version") == "fw@v99"

    def test_llm_call_step(self, tmp_path):
        rl = _make_runlog({
            "event": "llm_call",
            "step": 2,
            "node_id": "l1",
            "model": "gpt-4o-2026-03",
            "labels_in": ["pii"],
            "labels_out": ["pii"],
            "markers_in": ["1"],
            "markers_out": ["1"],
            "response_preview": "Summary: ...",
            "ts": 3.0,
        })
        report = import_flowwarden_runlog(rl, _sb_path(tmp_path, "fw.sb"))
        assert report.kind_counts.get("llm_call", 0) == 1
        steps = _read_steps(_sb_path(tmp_path, "fw.sb"))
        s = steps[0]
        assert s["step_kind"] == "llm_call"
        assert s["inputs"]["model"] == "gpt-4o-2026-03"
        assert s["inputs"]["fw_labels_in"] == ["pii"]
        assert s["inputs"]["fw_markers_in"] == ["1"]

    def test_violation_step(self, tmp_path):
        rl = _make_runlog({
            "event": "violation",
            "step": 3,
            "rule_line": 1,
            "sink": "send_email",
            "labels": ["pii"],
            "ts": 4.0,
        })
        report = import_flowwarden_runlog(rl, _sb_path(tmp_path, "fw.sb"))
        assert report.kind_counts.get("router", 0) == 1
        steps = _read_steps(_sb_path(tmp_path, "fw.sb"))
        s = steps[0]
        assert s["step_kind"] == "router"
        assert s["name"] == "fw_violation"
        assert s["outputs"]["fw_sink"] == "send_email"
        assert "pii" in s["outputs"]["fw_labels"]

    def test_decision_step(self, tmp_path):
        rl = _make_runlog({
            "event": "decision",
            "step": 3,
            "rule": "forbid PII -> email",
            "rule_line": 1,
            "fired": True,
            "decision": "FORBID",
            "ts": 4.0,
        })
        report = import_flowwarden_runlog(rl, _sb_path(tmp_path, "fw.sb"))
        steps = _read_steps(_sb_path(tmp_path, "fw.sb"))
        assert steps[0]["step_kind"] == "router"
        assert steps[0]["outputs"]["choice"] == "FORBID"
        assert steps[0]["outputs"]["fw_fired"] is True

    def test_run_end_step(self, tmp_path):
        rl = _make_runlog({
            "event": "run_end",
            "run_id": "r1",
            "graph_hash": "sha256:abc123",
            "response_text": "Final answer.",
            "ts": 5.0,
        })
        report = import_flowwarden_runlog(rl, _sb_path(tmp_path, "fw.sb"))
        steps = _read_steps(_sb_path(tmp_path, "fw.sb"))
        s = steps[0]
        assert s["name"] == "fw_run_end"
        assert s["outputs"]["fw_graph_hash"] == "sha256:abc123"

    def test_full_run_sequence(self, tmp_path):
        """Full run: run_start, tool_call, llm_call, violation, run_end."""
        rl = _make_runlog(
            {"event": "run_start", "run_id": "r1", "policy_version": "fw@v1", "ts": 1.0},
            {"event": "tool_call", "name": "fetch_customer", "labels_in": [], "labels_out": ["pii"], "ts": 2.0},
            {"event": "llm_call", "model": "gpt-4o", "labels_in": ["pii"], "labels_out": ["pii"], "ts": 3.0},
            {"event": "violation", "sink": "send_email", "labels": ["pii"], "rule_line": 1, "ts": 4.0},
            {"event": "run_end", "run_id": "r1", "graph_hash": "h1", "ts": 5.0},
        )
        report = import_flowwarden_runlog(rl, _sb_path(tmp_path, "fw.sb"))
        assert report.step_count == 4  # run_start is skipped
        steps = _read_steps(_sb_path(tmp_path, "fw.sb"))
        kinds = [s["step_kind"] for s in steps]
        assert kinds == ["tool_call", "llm_call", "router", "router"]

    def test_file_path_input(self, tmp_path):
        """RunLog can be provided as a JSONL file path."""
        jsonl = tmp_path / "runlog.jsonl"
        events = [
            {"event": "run_start", "run_id": "r1", "policy_version": "fw@v1", "ts": 1.0},
            {"event": "tool_call", "name": "t", "labels_in": [], "labels_out": [], "ts": 2.0},
        ]
        with jsonl.open("w") as fh:
            for ev in events:
                fh.write(json.dumps(ev) + "\n")
        report = import_flowwarden_runlog(str(jsonl), _sb_path(tmp_path, "fw.sb"))
        assert report.step_count == 1


# ======================================================================
# § flowwarden — FlowWardenRecorderAttachment tests
# ======================================================================


class TestFlowWardenRecorderAttachment:
    def test_flush_records_events(self, tmp_path):
        """flush() records all tool_call / llm_call events from a run-log."""

        class _FakeRun:
            runlog = _make_runlog(
                {"event": "run_start", "policy_version": "fw@v1", "ts": 1.0},
                {"event": "tool_call", "name": "search", "labels_in": [], "labels_out": ["pii"], "ts": 2.0},
                {"event": "llm_call", "model": "gpt-4o", "labels_in": ["pii"], "labels_out": ["pii"], "ts": 3.0},
            )

        with record(_sb_path(tmp_path, "t.sb")) as rec:
            attachment = FlowWardenRecorderAttachment(_FakeRun(), rec)
            n = attachment.flush()
        assert n == 2
        steps = _read_steps(_sb_path(tmp_path, "t.sb"))
        assert len(steps) == 2
        assert steps[0]["step_kind"] == "tool_call"
        assert steps[1]["step_kind"] == "llm_call"

    def test_flush_violation_becomes_router(self, tmp_path):
        class _FakeRun:
            runlog = _make_runlog(
                {"event": "run_start", "policy_version": "fw@v1", "ts": 1.0},
                {"event": "violation", "sink": "email", "labels": ["pii"], "ts": 2.0},
            )

        with record(_sb_path(tmp_path, "t.sb")) as rec:
            attachment = FlowWardenRecorderAttachment(_FakeRun(), rec)
            n = attachment.flush()
        assert n == 1
        steps = _read_steps(_sb_path(tmp_path, "t.sb"))
        assert steps[0]["step_kind"] == "router"


# ======================================================================
# § ragdoctor fakes
# ======================================================================


@dataclass
class _FakeChunk:
    text: str
    doc_id: str = "doc-1"


@dataclass
class _FakeHit:
    chunk: _FakeChunk
    score: float


@dataclass
class _FakeFinding:
    code: str
    severity: str
    message: str


@dataclass
class _FakeDiagnosticReport:
    findings: List[_FakeFinding] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    queries: List[str] = field(default_factory=list)


class _FakeDoctor:
    def __init__(self, pipeline: Any):
        self._pipeline = pipeline

    def diagnose(self, queries=None, **kwargs) -> _FakeDiagnosticReport:
        return _FakeDiagnosticReport(
            findings=[_FakeFinding("low_coverage", "warn", "some queries have low coverage")],
            metrics={"mean_top1_score": 0.72, "n_chunks": 10},
            queries=list(queries or []),
        )


class _FakePipeline:
    """Minimal duck-typed RagPipeline."""

    def query(self, q: str, *, k: int = 5) -> List[_FakeHit]:
        return [
            _FakeHit(chunk=_FakeChunk(text=f"chunk {i} for {q}", doc_id=f"doc-{i}"), score=0.9 - 0.1 * i)
            for i in range(min(k, 3))
        ]

    def answer(self, q: str, *, k: int = 5) -> str:
        return f"Answer to: {q}"


# ======================================================================
# § ragdoctor — RagDoctorShim tests
# ======================================================================


class TestRagDoctorShim:
    def test_query_records_retrieval_step(self, tmp_path):
        pipe = _FakePipeline()
        with record(_sb_path(tmp_path, "rag.sb")) as rec:
            shim = RagDoctorShim(pipe, rec, doctor=_FakeDoctor(pipe))
            hits = shim.query("what is dirty-set propagation?", k=3)
        assert len(hits) == 3
        steps = _read_steps(_sb_path(tmp_path, "rag.sb"))
        assert len(steps) == 1
        s = steps[0]
        assert s["name"] == "rag_retrieval"
        assert s["step_kind"] == "tool_call"
        # recorder.tool_call puts arguments in inputs["arguments"]
        assert s["inputs"]["arguments"]["query"] == "what is dirty-set propagation?"
        assert s["inputs"]["arguments"]["k"] == 3
        # recorder.tool_call wraps result in outputs["result"]
        assert s["outputs"]["result"]["n_hits"] == 3

    def test_query_outputs_are_json_safe(self, tmp_path):
        """Hit objects are normalized to JSON-safe dicts."""
        pipe = _FakePipeline()
        with record(_sb_path(tmp_path, "rag.sb")) as rec:
            shim = RagDoctorShim(pipe, rec, doctor=_FakeDoctor(pipe))
            shim.query("test query")
        steps = _read_steps(_sb_path(tmp_path, "rag.sb"))
        # hits are inside outputs["result"]["hits"]
        hits = steps[0]["outputs"]["result"]["hits"]
        assert all(isinstance(h, dict) for h in hits)
        assert all(isinstance(h["score"], float) for h in hits)

    def test_diagnose_records_diagnostic_step(self, tmp_path):
        pipe = _FakePipeline()
        with record(_sb_path(tmp_path, "rag.sb")) as rec:
            shim = RagDoctorShim(pipe, rec, doctor=_FakeDoctor(pipe))
            report = shim.diagnose(["q1", "q2"])
        steps = _read_steps(_sb_path(tmp_path, "rag.sb"))
        assert len(steps) == 1
        s = steps[0]
        assert s["name"] == "rag_diagnose"
        # outputs wrapped in {"result": ...} by recorder.tool_call
        result = s["outputs"]["result"]
        assert result["n_findings"] == 1
        assert result["n_warnings"] == 1
        assert result["n_errors"] == 0

    def test_diagnose_metrics_captured(self, tmp_path):
        pipe = _FakePipeline()
        with record(_sb_path(tmp_path, "rag.sb")) as rec:
            shim = RagDoctorShim(pipe, rec, doctor=_FakeDoctor(pipe))
            shim.diagnose()
        steps = _read_steps(_sb_path(tmp_path, "rag.sb"))
        metrics = steps[0]["outputs"]["result"]["metrics"]
        assert "mean_top1_score" in metrics
        assert "n_chunks" in metrics

    def test_answer_records_retrieval_and_generation(self, tmp_path):
        pipe = _FakePipeline()
        with record(_sb_path(tmp_path, "rag.sb")) as rec:
            shim = RagDoctorShim(pipe, rec, doctor=_FakeDoctor(pipe))
            answer = shim.answer("what is replay?", k=2)
        assert "replay" in answer.lower()
        steps = _read_steps(_sb_path(tmp_path, "rag.sb"))
        assert len(steps) == 2
        kinds = {s["step_kind"] for s in steps}
        assert "tool_call" in kinds
        assert "llm_call" in kinds

    def test_query_then_diagnose_two_steps(self, tmp_path):
        pipe = _FakePipeline()
        with record(_sb_path(tmp_path, "rag.sb")) as rec:
            shim = RagDoctorShim(pipe, rec, doctor=_FakeDoctor(pipe))
            shim.query("question 1")
            shim.diagnose(["question 1"])
        steps = _read_steps(_sb_path(tmp_path, "rag.sb"))
        assert len(steps) == 2
        assert steps[0]["name"] == "rag_retrieval"
        assert steps[1]["name"] == "rag_diagnose"

    def test_record_rag_query_functional_form(self, tmp_path):
        pipe = _FakePipeline()
        with record(_sb_path(tmp_path, "rag.sb")) as rec:
            hits = record_rag_query(pipe, "hello world", rec, k=2)
        assert len(hits) == 2
        steps = _read_steps(_sb_path(tmp_path, "rag.sb"))
        assert steps[0]["name"] == "rag_retrieval"


# ======================================================================
# § ragdoctor — import_ragdoctor_trace tests
# ======================================================================


@dataclass
class _FakeSpan:
    name: str
    span_id: str
    parent_id: Optional[str] = None
    start_ts: float = 0.0
    end_ts: float = 0.1
    status: str = "ok"
    error: Optional[str] = None
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeTrace:
    trace_id: str
    spans: List[_FakeSpan] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)


class TestImportRagdoctorTrace:
    def test_empty_trace(self, tmp_path):
        trace = _FakeTrace(trace_id="t1", spans=[])
        report = import_ragdoctor_trace(trace, _sb_path(tmp_path, "rd.sb"))
        assert report.step_count == 0
        assert os.path.exists(_sb_path(tmp_path, "rd.sb"))

    def test_retrieve_span_becomes_tool_call(self, tmp_path):
        trace = _FakeTrace(
            trace_id="t1",
            spans=[_FakeSpan("retrieve", "s1", attrs={"k": 5})],
        )
        report = import_ragdoctor_trace(trace, _sb_path(tmp_path, "rd.sb"))
        assert report.kind_counts.get("tool_call", 0) == 1
        steps = _read_steps(_sb_path(tmp_path, "rd.sb"))
        assert steps[0]["step_kind"] == "tool_call"
        assert steps[0]["name"] == "retrieve"

    def test_rerank_span_becomes_tool_call(self, tmp_path):
        trace = _FakeTrace(
            trace_id="t1",
            spans=[_FakeSpan("rerank", "s1")],
        )
        report = import_ragdoctor_trace(trace, _sb_path(tmp_path, "rd.sb"))
        steps = _read_steps(_sb_path(tmp_path, "rd.sb"))
        assert steps[0]["step_kind"] == "tool_call"

    def test_generate_span_becomes_llm_call(self, tmp_path):
        trace = _FakeTrace(
            trace_id="t1",
            spans=[_FakeSpan("generate", "s1", attrs={"response": "Generated answer."})],
        )
        report = import_ragdoctor_trace(trace, _sb_path(tmp_path, "rd.sb"))
        assert report.kind_counts.get("llm_call", 0) == 1
        steps = _read_steps(_sb_path(tmp_path, "rd.sb"))
        assert steps[0]["step_kind"] == "llm_call"

    def test_answer_span_becomes_llm_call(self, tmp_path):
        trace = _FakeTrace(
            trace_id="t1",
            spans=[_FakeSpan("answer", "s1")],
        )
        report = import_ragdoctor_trace(trace, _sb_path(tmp_path, "rd.sb"))
        steps = _read_steps(_sb_path(tmp_path, "rd.sb"))
        assert steps[0]["step_kind"] == "llm_call"

    def test_unknown_span_becomes_router(self, tmp_path):
        trace = _FakeTrace(
            trace_id="t1",
            spans=[_FakeSpan("pipeline", "s1")],
        )
        report = import_ragdoctor_trace(trace, _sb_path(tmp_path, "rd.sb"))
        steps = _read_steps(_sb_path(tmp_path, "rd.sb"))
        assert steps[0]["step_kind"] == "router"

    def test_parent_span_id_linkage(self, tmp_path):
        """Child span's parent_id is resolved to parent_step_id."""
        trace = _FakeTrace(
            trace_id="t1",
            spans=[
                _FakeSpan("pipeline", "root", parent_id=None),
                _FakeSpan("retrieve", "child", parent_id="root"),
            ],
        )
        report = import_ragdoctor_trace(trace, _sb_path(tmp_path, "rd.sb"))
        steps = _read_steps(_sb_path(tmp_path, "rd.sb"))
        assert steps[1]["parent_step_id"] == steps[0]["step_id"]

    def test_error_status_in_outputs(self, tmp_path):
        trace = _FakeTrace(
            trace_id="t1",
            spans=[_FakeSpan("retrieve", "s1", status="error", error="timeout")],
        )
        report = import_ragdoctor_trace(trace, _sb_path(tmp_path, "rd.sb"))
        steps = _read_steps(_sb_path(tmp_path, "rd.sb"))
        assert steps[0]["outputs"].get("rd_ok") is False or \
               steps[0]["outputs"].get("rd_error") == "timeout" or \
               steps[0]["outputs"].get("error") == "timeout"

    def test_multiple_spans_in_order(self, tmp_path):
        spans = [
            _FakeSpan("retrieve", f"s{i}", start_ts=float(i), end_ts=float(i) + 0.1)
            for i in range(4)
        ]
        trace = _FakeTrace(trace_id="t1", spans=spans)
        report = import_ragdoctor_trace(trace, _sb_path(tmp_path, "rd.sb"))
        assert report.step_count == 4
        steps = _read_steps(_sb_path(tmp_path, "rd.sb"))
        assert all(s["step_kind"] == "tool_call" for s in steps)

    def test_attrs_propagated_to_inputs(self, tmp_path):
        trace = _FakeTrace(
            trace_id="t1",
            spans=[_FakeSpan("retrieve", "s1", attrs={"k": 10, "query": "hello"})],
        )
        report = import_ragdoctor_trace(trace, _sb_path(tmp_path, "rd.sb"))
        steps = _read_steps(_sb_path(tmp_path, "rd.sb"))
        # Attrs are stored with rd_ prefix.
        assert steps[0]["inputs"].get("rd_k") == 10 or \
               steps[0]["inputs"].get("rd_query") == "hello"


# ======================================================================
# § Cross-integration: round-trip consistency
# ======================================================================


class TestRoundTrips:
    def test_toolwarden_import_readable(self, tmp_path):
        """Imported audit log is a valid .sb trace that TraceReader can parse."""
        entries = [
            _FakeAuditEntry(1000.0 + i, "a", f"t{i}", "allow", [], [], {}, decision_id=f"d{i}")
            for i in range(3)
        ]
        import_toolwarden_audit(entries, _sb_path(tmp_path, "tw.sb"))
        steps = _read_steps(_sb_path(tmp_path, "tw.sb"))
        assert len(steps) == 3

    def test_flowwarden_import_readable(self, tmp_path):
        """Imported run-log is a valid .sb trace that TraceReader can parse."""
        rl = _make_runlog(
            {"event": "run_start", "policy_version": "v1", "ts": 1.0},
            {"event": "tool_call", "name": "t", "labels_in": [], "labels_out": [], "ts": 2.0},
        )
        import_flowwarden_runlog(rl, _sb_path(tmp_path, "fw.sb"))
        steps = _read_steps(_sb_path(tmp_path, "fw.sb"))
        assert len(steps) == 1

    def test_ragdoctor_import_readable(self, tmp_path):
        """Imported trace is a valid .sb trace that TraceReader can parse."""
        trace = _FakeTrace(
            trace_id="t1",
            spans=[
                _FakeSpan("retrieve", "s1"),
                _FakeSpan("generate", "s2", parent_id="s1"),
            ],
        )
        import_ragdoctor_trace(trace, _sb_path(tmp_path, "rd.sb"))
        steps = _read_steps(_sb_path(tmp_path, "rd.sb"))
        assert len(steps) == 2

    def test_toolwarden_shim_replay_cache_hit(self, tmp_path):
        """A recorded trace is a valid .sb trace that can be replayed."""
        warden = _FakeWarden(_FakePolicy("allow"))
        sb_path = _sb_path(tmp_path, "tw.sb")
        with record(sb_path) as rec:
            shim = WardenRecorderShim(warden, rec, policy_version="v1")
            shim.invoke("search", {"q": "hi"}, executor=lambda n, a: {"hits": ["x"]})

        # Verify trace is readable and has expected structure
        steps = _read_steps(sb_path)
        assert len(steps) == 1
        args = steps[0]["inputs"]["arguments"]
        assert args["q"] == "hi"
        assert args["policy_version"] == "v1"
