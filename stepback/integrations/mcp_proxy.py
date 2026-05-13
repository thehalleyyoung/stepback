"""stepback MCP recorder/proxy mode.

This module provides an **MCP-protocol-aware recording proxy** that sits
between an MCP client and a real upstream MCP server, transparently recording
every ``tools/call`` invocation as a stepback ``.sb`` step.

Two entry points
----------------

* :class:`MCPRecorderProxy` — wraps an upstream MCP session object and exposes
  the same ``call_tool`` / ``list_tools`` / ``initialize`` surface so existing
  client code can swap in the proxy with a one-line change.  Every
  ``call_tool`` is recorded as a ``tool_call`` step in the attached
  :class:`~stepback.recorder.Recorder`.

* :func:`import_mcp_log` — converts an MCP server-side JSON event log
  (structured as ``{"events": [...]}`` where each event has ``type`` and for
  ``"call_tool"`` events has ``"name"`` / ``"arguments"`` / ``"result"``) into
  a standalone ``.sb`` file for offline replay and substitution experiments.

Usage — live proxy
~~~~~~~~~~~~~~~~~~
.. code-block:: python

    from stepback import record
    from stepback.integrations.mcp_proxy import MCPRecorderProxy

    real_session = MyMCPSession.connect("http://localhost:8090")

    with record("mcp_run.sb") as rec:
        session = MCPRecorderProxy(real_session, rec, server_name="my-server")

        tools = session.list_tools()
        result = session.call_tool("search", {"q": "stepback replay caching"})

Usage — import a server event log
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    from stepback.integrations.mcp_proxy import import_mcp_log

    report = import_mcp_log("events.json", "mcp_run.sb")
    print(report.step_count, "MCP calls imported")

Compatibility
~~~~~~~~~~~~~
The proxy duck-types the MCP ``ClientSession`` protocol (MCP spec 2024-11-05
and later): any object with ``call_tool(name, arguments) -> result``,
``list_tools() -> [{"name": ..., ...}]``, and ``initialize()`` methods is
supported.  The recorder wraps the underlying ``call_tool`` in an executor so
dirty-step replay through the proxy re-invokes the real server for changed
steps while serving cache hits without any network call for unchanged ones.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from ..canonical import hash_obj, canonical_json, sha256_hex
from ..importers import ImportReport, _emit_step, _open_writer, _bump_kind, _ZERO_NONDET
from ..recorder import Recorder, RecorderKey


__all__ = [
    "MCPRecorderProxy",
    "import_mcp_log",
    "MCPProxyError",
]


class MCPProxyError(ValueError):
    """Raised when the upstream MCP session is missing required methods."""


# ======================================================================
# MCPRecorderProxy
# ======================================================================

class MCPRecorderProxy:
    """An MCP-protocol-aware recording proxy.

    Acts as a drop-in replacement for an MCP ``ClientSession`` object.  Every
    ``call_tool`` invocation is:

    1. Forwarded to the real upstream session.
    2. Recorded as a ``tool_call`` step in the attached recorder.

    The proxy also records ``initialize`` and exposes ``list_tools``
    pass-through.  All other attributes are transparently forwarded to the
    real session via ``__getattr__``.

    Args:
        session: The real upstream MCP ``ClientSession`` (or any object with a
            ``call_tool(name, arguments)`` method).
        recorder: An open :class:`~stepback.recorder.Recorder` context that
            should receive the step frames.
        server_name: Human-readable server identifier prefixed to tool names
            on the timeline (e.g. ``"filesystem"``, ``"github"``).  Defaults
            to ``"mcp"``.
        record_list_tools: When *True* (default *False*), ``list_tools``
            calls are also recorded as ``tool_call`` steps with
            ``name="__list_tools__"`` so the full session negotiation is
            visible in the trace.
    """

    def __init__(
        self,
        session: Any,
        recorder: Recorder,
        *,
        server_name: str = "mcp",
        record_list_tools: bool = False,
    ) -> None:
        if not hasattr(session, "call_tool"):
            raise MCPProxyError(
                "MCPRecorderProxy: session lacks call_tool(); "
                "expected an MCP ClientSession-shaped object"
            )
        self._session = session
        self._recorder = recorder
        self._server_name = server_name
        self._record_list_tools = record_list_tools

    # ------------------------------------------------------------------
    # MCP protocol surface
    # ------------------------------------------------------------------

    def call_tool(self, name: str, arguments: Optional[dict] = None) -> Any:
        """Record and forward a ``tools/call`` request.

        The tool name on the stepback timeline is
        ``{server_name}:{name}`` so multi-server traces stay legible.
        """
        args = dict(arguments or {})
        qualified = f"{self._server_name}:{name}"

        def _executor(_qname: str, _args: dict) -> Any:
            return self._session.call_tool(name, _args)

        step = self._recorder.tool_call(qualified, args, executor=_executor)
        return step["outputs"]["result"]

    def list_tools(self) -> List[dict]:
        """Forward ``tools/list`` to the upstream session.

        When *record_list_tools* is *True*, the call is also recorded as a
        ``tool_call`` step with ``name="{server_name}:__list_tools__"``.
        """
        tools = self._session.list_tools()
        if self._record_list_tools:
            qualified = f"{self._server_name}:__list_tools__"

            def _exec(_n: str, _a: dict) -> Any:
                return self._session.list_tools()

            self._recorder.tool_call(qualified, {}, executor=_exec)
        return tools

    def initialize(self) -> Any:
        """Forward the ``initialize`` handshake to the upstream session."""
        return self._session.initialize()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._session, item)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def server_name(self) -> str:
        """The server name prefix used for tool steps."""
        return self._server_name

    def __repr__(self) -> str:
        return (
            f"MCPRecorderProxy(server_name={self._server_name!r}, "
            f"session={self._session!r})"
        )


# ======================================================================
# import_mcp_log  — server event log → .sb
# ======================================================================

def import_mcp_log(
    input_path: str,
    output_path: str,
    *,
    server_name: str = "mcp",
    key: Optional[RecorderKey] = None,
    compression: bool = True,
) -> ImportReport:
    """Convert an MCP server event log to a ``.sb`` trace.

    The input file must be a JSON document in one of these shapes:

    * ``{"events": [...]}`` — a list of event objects (preferred).
    * ``[...]`` — a bare list of event objects.

    Each event object must have a ``"type"`` field.  Events with
    ``type == "call_tool"`` (or ``"tools/call"`` per the MCP 2024-11-05
    wire format) are converted to ``tool_call`` steps.  All other
    event types are converted to ``router`` steps so the full
    session lifecycle is visible in the trace.

    Each ``call_tool`` event should carry:

    * ``"name"`` — tool name (string).
    * ``"arguments"`` — call arguments (dict, optional).
    * ``"result"`` — raw server response (any JSON value, optional).
    * ``"error"`` — error message string (optional; mutually exclusive
      with ``result``).
    * ``"duration_ms"`` — wallclock duration in milliseconds (optional).

    Args:
        input_path: Path to the MCP event log JSON file.
        output_path: Path to write the ``.sb`` file.
        server_name: Server name prefix for tool names on the timeline.
        key: Recorder key for HMAC + signing.  A fresh key is generated
            when *None*.
        compression: Whether to compress step bodies.

    Returns:
        :class:`~stepback.importers.ImportReport` with per-kind step counts.

    Raises:
        :class:`~stepback.importers.ImportError`: If the input file cannot
            be parsed as a valid MCP event log.
    """
    from ..importers import ImportError as StepbackImportError

    try:
        with open(input_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        raise StepbackImportError(
            f"import_mcp_log: cannot read {input_path!r}: {exc}"
        ) from exc

    if isinstance(payload, list):
        events: List[dict] = payload
    elif isinstance(payload, dict) and isinstance(payload.get("events"), list):
        events = payload["events"]
    else:
        raise StepbackImportError(
            f"import_mcp_log: expected a list or {{\"events\": [...]}} mapping; "
            f"got {type(payload).__name__}"
        )

    writer, _key = _open_writer(output_path, key=key, compression=compression)
    report = ImportReport(output_path=output_path, source_format="mcp_log")
    prev_id: Optional[str] = None

    try:
        for idx, event in enumerate(events):
            if not isinstance(event, dict):
                report.skipped_count += 1
                continue

            ev_type = str(event.get("type") or "unknown").lower()
            step_id = str(event.get("id") or uuid.uuid4())
            duration_ms = event.get("duration_ms")
            wallclock_ns = (
                int(duration_ms * 1_000_000) if isinstance(duration_ms, (int, float))
                else None
            )

            # --- tool call events ---
            if ev_type in ("call_tool", "tools/call", "tool_call"):
                name = str(event.get("name") or "unknown_tool")
                qualified = f"{server_name}:{name}"
                arguments = dict(event.get("arguments") or {})
                if "result" in event:
                    outputs: dict = {"result": event["result"]}
                elif "error" in event:
                    outputs = {"error": str(event["error"]), "result": None}
                else:
                    outputs = {"result": None}

                _emit_step(
                    writer,
                    step_id=step_id,
                    step_kind="tool_call",
                    name=qualified,
                    parent_step_id=prev_id,
                    inputs={"name": qualified, "arguments": arguments},
                    outputs=outputs,
                    wallclock_ns=wallclock_ns,
                )
                _bump_kind(report, "tool_call")

            # --- initialize / ping / other lifecycle events ---
            else:
                decision = ev_type
                extras_payload = {
                    k: v for k, v in event.items()
                    if k not in ("type", "id", "duration_ms")
                }
                _emit_step(
                    writer,
                    step_id=step_id,
                    step_kind="router",
                    name=f"mcp.{ev_type}",
                    parent_step_id=prev_id,
                    inputs=extras_payload,
                    outputs={"decision": decision},
                    wallclock_ns=wallclock_ns,
                )
                _bump_kind(report, "router")

            prev_id = step_id

    finally:
        writer.close()

    return report
