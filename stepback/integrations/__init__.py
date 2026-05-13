"""Ecosystem integrations for stepback.

This package provides thin adapter layers between stepback's trace recorder /
importer machinery and sibling projects in the same ecosystem:

* :mod:`stepback.integrations.ragdoctor` — record and import ragdoctor
  diagnostic RAG runs as ``.sb`` traces.
* :mod:`stepback.integrations.flowwarden` — import flowwarden run-logs with
  IFC/provenance labels as ``.sb`` traces; attach a live ``AgentRun`` to a
  stepback ``Recorder``.
* :mod:`stepback.integrations.toolwarden` — wrap a toolwarden ``Warden`` to
  record every enforcement decision (ALLOW / DENY / REDACT) as a stepback
  step; import an ``AuditLog`` as a ``.sb`` trace for replay-time audits.
* :mod:`stepback.integrations.mcp_proxy` — MCP-protocol-aware recording proxy
  that sits between an MCP client and a real upstream MCP server, recording
  every ``tools/call`` as a stepback step.  Also provides
  :func:`~stepback.integrations.mcp_proxy.import_mcp_log` to convert an MCP
  server event log into a ``.sb`` trace.

None of the submodules import their companion libraries at module load time —
all companion-library symbols are imported inside the functions/classes that
use them.  This means ``import stepback.integrations.toolwarden`` is safe even
when toolwarden is not installed; a helpful ``ImportError`` is only raised at
call time.
"""
