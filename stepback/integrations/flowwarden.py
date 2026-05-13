"""stepback ↔ flowwarden integration.

Converts a flowwarden run-log (JSONL events emitted by ``AgentRun``) into a
``.sb`` trace so that the full IFC/provenance trail — including per-step
taint labels, policy decisions, and violation events — is preserved for
stepback's replay, bisect, and substitution machinery.

Two entry points:

* :func:`import_flowwarden_runlog` — convert a ``RunLog`` object (or JSONL
  file path) into a ``.sb`` trace.

* :class:`FlowWardenRecorderAttachment` — record a completed ``AgentRun``'s
  run-log into an already-open stepback ``Recorder``.  Useful when you want
  the flowwarden events interleaved with other stepback steps recorded in the
  same trace.

IFC label mapping
-----------------
flowwarden ``labels_in`` / ``labels_out`` are stored in the step ``inputs``
and ``outputs`` dicts under the keys ``fw_labels_in`` / ``fw_labels_out``
(lists of strings), making them part of the step's cache identity.  A change
in taint labels across the same tool invocation therefore marks downstream
steps dirty — exactly the replay-time audit behaviour the integration aims to
support.

Policy version
--------------
The ``policy_version`` field from the ``run_start`` event is propagated to
every step's ``inputs`` dict as ``fw_policy_version``.  Swapping the policy
version in a substitution will mark all policy-governed steps dirty.

Violation events
----------------
``violation`` and ``decision`` events become ``router`` steps (no side
effects) with the rule, decision, and label information in ``outputs``.
``run_end`` events become a ``router`` step capturing the ``graph_hash``.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional, Union

from ..canonical import hash_obj, sha256_hex, canonical_json
from ..importers import ImportReport, _emit_step, _open_writer, RecorderKey
from ..recorder import Recorder


__all__ = [
    "import_flowwarden_runlog",
    "FlowWardenRecorderAttachment",
]

_ZERO_NONDET = sha256_hex(canonical_json({}))


# ======================================================================
# import_flowwarden_runlog
# ======================================================================


def import_flowwarden_runlog(
    run_log: Any,
    output_path: str,
    *,
    key: Optional[RecorderKey] = None,
    compression: bool = False,
) -> ImportReport:
    """Convert a flowwarden ``RunLog`` (or list of event dicts) to ``.sb``.

    Each flowwarden event becomes one stepback step:

    +-----------------+-------------------+----------------------------------+
    | flowwarden event| stepback step_kind| Notes                            |
    +=================+===================+==================================+
    | ``run_start``   | *skipped*         | ``policy_version`` captured for  |
    |                 |                   | all subsequent steps.            |
    +-----------------+-------------------+----------------------------------+
    | ``tool_call``   | ``tool_call``     | labels in ``fw_labels_in`` /     |
    |                 |                   | ``fw_labels_out``; policy_version|
    |                 |                   | in inputs.                       |
    +-----------------+-------------------+----------------------------------+
    | ``llm_call``    | ``llm_call``      | markers / model in inputs;       |
    |                 |                   | labels preserved.                |
    +-----------------+-------------------+----------------------------------+
    | ``decision``    | ``router``        | rule + decision in outputs.      |
    +-----------------+-------------------+----------------------------------+
    | ``violation``   | ``router``        | rule_line + sink + labels.       |
    +-----------------+-------------------+----------------------------------+
    | ``agent_event`` | ``router``        | custom agent lifecycle events.   |
    +-----------------+-------------------+----------------------------------+
    | ``run_end``     | ``router``        | graph_hash captured.             |
    +-----------------+-------------------+----------------------------------+

    Args:
        run_log: A ``flowwarden.RunLog`` instance (has ``.events`` attribute)
            or a list of event dicts.  Alternatively a file-path string
            pointing to a JSONL run-log file.
        output_path: Destination ``.sb`` file path.
        key: Optional :class:`~stepback.importers.RecorderKey`.
        compression: Write compressed frames (default ``False``).

    Returns:
        An :class:`~stepback.importers.ImportReport`.
    """
    events = _extract_events(run_log)
    writer, key = _open_writer(output_path, key=key, compression=compression)
    report = ImportReport(output_path=output_path, source_format="flowwarden_runlog")

    policy_version = "flowwarden@unknown"
    run_id = ""
    step_counter = 0

    # Map flowwarden ``node_id`` / ``step`` → stepback step_id for parentage.
    # flowwarden's step numbering is sequential; the immediately preceding
    # step is used as the parent for decision/violation events that lack a
    # node_id.
    last_step_id: Optional[str] = None

    try:
        for ev in events:
            event_name = str(ev.get("event", ""))

            if event_name == "run_start":
                policy_version = str(ev.get("policy_version") or "flowwarden@unknown")
                run_id = str(ev.get("run_id") or "")
                continue  # not emitted as its own step

            step_counter += 1
            sid = f"step:{step_counter}"
            ts_float = float(ev.get("ts") or 0.0)
            wallclock_ns = int(ts_float * 1_000_000_000) if ts_float else time.time_ns()

            if event_name == "tool_call":
                labels_in = [str(lb) for lb in (ev.get("labels_in") or [])]
                labels_out = [str(lb) for lb in (ev.get("labels_out") or [])]
                tool_name = str(ev.get("name") or "tool")
                node_id = str(ev.get("node_id") or sid)

                inputs: Dict[str, Any] = {
                    "kind": "tool_call",
                    "name": tool_name,
                    "arguments": {},  # flowwarden run-logs don't expose raw args
                    "fw_policy_version": policy_version,
                    "fw_labels_in": labels_in,
                    "fw_node_id": node_id,
                }
                policy_decision = str(ev.get("policy_decision") or "ALLOW")
                result_preview = str(ev.get("result_preview") or "")
                outputs: Dict[str, Any] = {
                    "result": result_preview,
                    "fw_labels_out": labels_out,
                    "fw_policy_decision": policy_decision,
                }
                _emit_step(
                    writer,
                    step_id=sid,
                    step_kind="tool_call",
                    name=tool_name,
                    parent_step_id=last_step_id,
                    inputs=inputs,
                    outputs=outputs,
                    wallclock_ns=wallclock_ns,
                )
                report.kind_counts["tool_call"] = report.kind_counts.get("tool_call", 0) + 1
                report.step_count += 1
                last_step_id = sid

            elif event_name == "llm_call":
                labels_in = [str(lb) for lb in (ev.get("labels_in") or [])]
                labels_out = [str(lb) for lb in (ev.get("labels_out") or [])]
                model = str(ev.get("model") or "unknown")
                node_id = str(ev.get("node_id") or sid)
                markers_in = list(ev.get("markers_in") or [])
                markers_out = list(ev.get("markers_out") or [])
                response_preview = str(ev.get("response_preview") or "")

                inputs = {
                    "kind": "llm_call",
                    "model": model,
                    "messages": [],  # previews only
                    "temperature": 0.0,
                    "seed": None,
                    "fw_policy_version": policy_version,
                    "fw_labels_in": labels_in,
                    "fw_markers_in": markers_in,
                    "fw_node_id": node_id,
                }
                outputs = {
                    "choices": [{"message": {"content": response_preview}, "finish_reason": "stop"}],
                    "usage": {},
                    "fw_labels_out": labels_out,
                    "fw_markers_out": markers_out,
                }
                _emit_step(
                    writer,
                    step_id=sid,
                    step_kind="llm_call",
                    name=model,
                    parent_step_id=last_step_id,
                    inputs=inputs,
                    outputs=outputs,
                    wallclock_ns=wallclock_ns,
                )
                report.kind_counts["llm_call"] = report.kind_counts.get("llm_call", 0) + 1
                report.step_count += 1
                last_step_id = sid

            elif event_name == "decision":
                rule = str(ev.get("rule") or "")
                fired = bool(ev.get("fired", False))
                decision_str = str(ev.get("decision") or "UNKNOWN")

                inputs = {
                    "kind": "router",
                    "name": "fw_policy_decision",
                    "options": ["ALLOW", "FORBID"],
                    "fw_policy_version": policy_version,
                    "fw_rule": rule,
                }
                outputs = {
                    "choice": decision_str,
                    "fw_fired": fired,
                    "fw_rule_line": ev.get("rule_line"),
                }
                _emit_step(
                    writer,
                    step_id=sid,
                    step_kind="router",
                    name="fw_policy_decision",
                    parent_step_id=last_step_id,
                    inputs=inputs,
                    outputs=outputs,
                    wallclock_ns=wallclock_ns,
                )
                report.kind_counts["router"] = report.kind_counts.get("router", 0) + 1
                report.step_count += 1
                last_step_id = sid

            elif event_name == "violation":
                labels = [str(lb) for lb in (ev.get("labels") or [])]
                sink = str(ev.get("sink") or "unknown")
                rule_line = ev.get("rule_line")

                inputs = {
                    "kind": "router",
                    "name": "fw_violation",
                    "options": ["VIOLATION"],
                    "fw_policy_version": policy_version,
                    "fw_labels": labels,
                    "fw_sink": sink,
                }
                outputs = {
                    "choice": "VIOLATION",
                    "fw_rule_line": rule_line,
                    "fw_sink": sink,
                    "fw_labels": labels,
                }
                _emit_step(
                    writer,
                    step_id=sid,
                    step_kind="router",
                    name="fw_violation",
                    parent_step_id=last_step_id,
                    inputs=inputs,
                    outputs=outputs,
                    wallclock_ns=wallclock_ns,
                )
                report.kind_counts["router"] = report.kind_counts.get("router", 0) + 1
                report.step_count += 1
                last_step_id = sid

            elif event_name == "agent_event":
                ev_name = str(ev.get("name") or "event")
                metadata = dict(ev.get("metadata") or {})

                inputs = {
                    "kind": "router",
                    "name": f"fw_agent_event:{ev_name}",
                    "options": [ev_name],
                    "fw_policy_version": policy_version,
                }
                outputs = {"choice": ev_name, "fw_metadata": metadata}
                _emit_step(
                    writer,
                    step_id=sid,
                    step_kind="router",
                    name=f"fw_agent_event:{ev_name}",
                    parent_step_id=last_step_id,
                    inputs=inputs,
                    outputs=outputs,
                    wallclock_ns=wallclock_ns,
                )
                report.kind_counts["router"] = report.kind_counts.get("router", 0) + 1
                report.step_count += 1
                last_step_id = sid

            elif event_name == "run_end":
                graph_hash = str(ev.get("graph_hash") or "")
                response_preview = str(ev.get("response_text") or "")

                inputs = {
                    "kind": "router",
                    "name": "fw_run_end",
                    "options": ["complete"],
                    "fw_policy_version": policy_version,
                    "fw_run_id": run_id,
                }
                outputs = {
                    "choice": "complete",
                    "fw_graph_hash": graph_hash,
                    "fw_response_preview": response_preview,
                }
                _emit_step(
                    writer,
                    step_id=sid,
                    step_kind="router",
                    name="fw_run_end",
                    parent_step_id=last_step_id,
                    inputs=inputs,
                    outputs=outputs,
                    wallclock_ns=wallclock_ns,
                )
                report.kind_counts["router"] = report.kind_counts.get("router", 0) + 1
                report.step_count += 1
                last_step_id = sid

            else:
                # Unknown event type — emit as a generic router step.
                inputs = {
                    "kind": "router",
                    "name": f"fw_{event_name}",
                    "options": [event_name],
                    "fw_policy_version": policy_version,
                }
                outputs = {"choice": event_name, "fw_raw": dict(ev)}
                _emit_step(
                    writer,
                    step_id=sid,
                    step_kind="router",
                    name=f"fw_{event_name}",
                    parent_step_id=last_step_id,
                    inputs=inputs,
                    outputs=outputs,
                    wallclock_ns=wallclock_ns,
                )
                report.kind_counts["router"] = report.kind_counts.get("router", 0) + 1
                report.step_count += 1
                last_step_id = sid

    finally:
        writer.close()

    return report


# ======================================================================
# FlowWardenRecorderAttachment
# ======================================================================


class FlowWardenRecorderAttachment:
    """Record a completed flowwarden ``AgentRun`` into a stepback ``Recorder``.

    After the agent run completes, call :meth:`flush` to replay the run-log
    events into the recorder.  The events are recorded as stepback steps using
    the same mapping as :func:`import_flowwarden_runlog`, but through the live
    recorder rather than a ``TraceWriter`` — so the steps are hash-chained
    into the same ``.sb`` file as any other steps recorded in the same
    ``with record(...) as rec:`` block.

    Usage::

        from flowwarden import AgentRun, install
        from stepback import record
        from stepback.integrations.flowwarden import FlowWardenRecorderAttachment

        run = AgentRun(policy=my_policy)
        with record("trace.sb") as rec:
            attachment = FlowWardenRecorderAttachment(run, rec)
            with run:
                result = my_agent(run)
            attachment.flush()

    Args:
        agent_run: A ``flowwarden.AgentRun`` (or any object with a ``.runlog``
            attribute that has an ``.events`` list).
        recorder: An active :class:`~stepback.recorder.Recorder`.
    """

    def __init__(self, agent_run: Any, recorder: Recorder) -> None:
        self._run = agent_run
        self._recorder = recorder

    def flush(self) -> int:
        """Record all run-log events into the attached ``Recorder``.

        Returns:
            The number of events that were recorded as steps.
        """
        events = _extract_events(self._run.runlog)
        policy_version = "flowwarden@unknown"
        recorded = 0

        for ev in events:
            event_name = str(ev.get("event", ""))
            if event_name == "run_start":
                policy_version = str(ev.get("policy_version") or "flowwarden@unknown")
                continue

            if event_name == "tool_call":
                labels_in = [str(lb) for lb in (ev.get("labels_in") or [])]
                labels_out = [str(lb) for lb in (ev.get("labels_out") or [])]
                tool_name = str(ev.get("name") or "tool")
                policy_decision = str(ev.get("policy_decision") or "ALLOW")
                result_preview = str(ev.get("result_preview") or "")

                def _tool_exec(n: str, a: Dict[str, Any]) -> Dict[str, Any]:
                    return {
                        "result": result_preview,
                        "fw_labels_out": labels_out,
                        "fw_policy_decision": policy_decision,
                    }

                self._recorder.tool_call(
                    tool_name,
                    {
                        "fw_policy_version": policy_version,
                        "fw_labels_in": labels_in,
                    },
                    executor=_tool_exec,
                )
                recorded += 1

            elif event_name == "llm_call":
                labels_in = [str(lb) for lb in (ev.get("labels_in") or [])]
                labels_out = [str(lb) for lb in (ev.get("labels_out") or [])]
                model = str(ev.get("model") or "unknown")
                response_preview = str(ev.get("response_preview") or "")

                def _llm_exec(m: str, msgs: List[Dict]) -> Dict[str, Any]:
                    return {
                        "choices": [{"message": {"content": response_preview}, "finish_reason": "stop"}],
                        "usage": {},
                        "fw_labels_out": labels_out,
                    }

                self._recorder.llm_call(
                    model,
                    [{"role": "user", "content": ""}],
                    executor=_llm_exec,
                    temperature=0.0,
                    seed=None,
                )
                recorded += 1

            elif event_name in ("decision", "violation", "run_end", "agent_event"):
                name_map = {
                    "decision": "fw_policy_decision",
                    "violation": "fw_violation",
                    "run_end": "fw_run_end",
                    "agent_event": f"fw_agent_event:{ev.get('name', '')}",
                }
                step_name = name_map.get(event_name, f"fw_{event_name}")
                choice = str(
                    ev.get("decision") or ev.get("name") or "complete"
                    if event_name != "violation" else "VIOLATION"
                )
                self._recorder.router(
                    name=step_name,
                    choice=choice,
                    options=[choice],
                )
                recorded += 1

        return recorded


# ======================================================================
# Internal helpers
# ======================================================================


def _extract_events(run_log: Any) -> List[Dict[str, Any]]:
    """Return a list of event dicts from a RunLog, list, or JSONL file path."""
    if isinstance(run_log, str):
        # Treat as a file path.
        import json
        events = []
        with open(run_log, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events
    if hasattr(run_log, "events"):
        return list(run_log.events)
    return list(run_log)
