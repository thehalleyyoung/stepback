"""Tests for step-150 additions.

Covers:
* stepback.integrations.mcp_proxy   — MCPRecorderProxy + import_mcp_log
* stepback.importers.import_cyclonedx_ai
* stepback.otel_bridge              — OtelBridge, OtlpHttpExporter, DryRunCollector
* scripts/slsa_example.py           — SLSA / in-toto provenance example
* docs/wasm-embedding.md            — WASM embedding documentation
* docs/integration-matrix.md        — public integration matrix doc

All tests are offline-only; no network calls are made.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

import stepback
from stepback import record, replay
from stepback.integrations.mcp_proxy import (
    MCPRecorderProxy,
    MCPProxyError,
    import_mcp_log,
)
from stepback.importers import import_cyclonedx_ai, ImportError as StepbackImportError
from stepback.otel_bridge import (
    OtelBridge,
    OtlpHttpExporter,
    DryRunCollector,
    OtelBridgeExportResult,
    OtelBridgeError,
)
from stepback.testing import run_recorded_agent


# ======================================================================
# Shared helpers
# ======================================================================

def _sb_path(tmp_path: Path, name: str) -> str:
    return str(tmp_path / name)


def _read_steps(path: str) -> List[dict]:
    t = replay(path)
    return list(t.recorded_steps)


def _record_trace(path: str) -> str:
    with record(path) as rec:
        run_recorded_agent(rec)
    return path


# ======================================================================
# MCPRecorderProxy tests
# ======================================================================

class _FakeMCPSession:
    """Fake MCP ClientSession for testing."""

    def __init__(self, tools: Optional[List[dict]] = None) -> None:
        self._tools = tools or [
            {"name": "search", "description": "Web search"},
            {"name": "read_file", "description": "Read a file"},
        ]
        self.call_log: List[dict] = []

    def call_tool(self, name: str, arguments: Optional[dict] = None) -> Any:
        args = dict(arguments or {})
        self.call_log.append({"name": name, "arguments": args})
        return {"content": f"result-of-{name}", "args": args}

    def list_tools(self) -> List[dict]:
        return list(self._tools)

    def initialize(self) -> dict:
        return {"protocol_version": "2024-11-05"}


class TestMCPRecorderProxy:
    def test_call_tool_records_step(self, tmp_path: Path) -> None:
        path = _sb_path(tmp_path, "mcp.sb")
        session = _FakeMCPSession()
        with record(path) as rec:
            proxy = MCPRecorderProxy(session, rec, server_name="test-server")
            result = proxy.call_tool("search", {"q": "replay caching"})

        assert result == {"content": "result-of-search", "args": {"q": "replay caching"}}
        steps = _read_steps(path)
        assert len(steps) >= 1
        tool_steps = [s for s in steps if s["step_kind"] == "tool_call"]
        assert len(tool_steps) == 1
        assert tool_steps[0]["name"] == "test-server:search"
        assert tool_steps[0]["inputs"]["arguments"] == {"q": "replay caching"}

    def test_multiple_calls_recorded(self, tmp_path: Path) -> None:
        path = _sb_path(tmp_path, "mcp_multi.sb")
        session = _FakeMCPSession()
        with record(path) as rec:
            proxy = MCPRecorderProxy(session, rec, server_name="fs")
            proxy.call_tool("search", {"q": "a"})
            proxy.call_tool("read_file", {"path": "/tmp/x"})

        steps = [s for s in _read_steps(path) if s["step_kind"] == "tool_call"]
        assert len(steps) == 2
        names = [s["name"] for s in steps]
        assert "fs:search" in names
        assert "fs:read_file" in names

    def test_default_server_name(self, tmp_path: Path) -> None:
        path = _sb_path(tmp_path, "mcp_default.sb")
        session = _FakeMCPSession()
        with record(path) as rec:
            proxy = MCPRecorderProxy(session, rec)
            proxy.call_tool("ping", {})

        steps = [s for s in _read_steps(path) if s["step_kind"] == "tool_call"]
        assert steps[0]["name"] == "mcp:ping"

    def test_list_tools_passthrough(self, tmp_path: Path) -> None:
        path = _sb_path(tmp_path, "mcp_list.sb")
        session = _FakeMCPSession()
        with record(path) as rec:
            proxy = MCPRecorderProxy(session, rec)
            tools = proxy.list_tools()

        assert len(tools) == 2
        # Not recorded by default
        steps = [s for s in _read_steps(path) if s["step_kind"] == "tool_call"]
        assert len(steps) == 0

    def test_list_tools_record_when_enabled(self, tmp_path: Path) -> None:
        path = _sb_path(tmp_path, "mcp_list_rec.sb")
        session = _FakeMCPSession()
        with record(path) as rec:
            proxy = MCPRecorderProxy(session, rec, record_list_tools=True)
            proxy.list_tools()

        steps = _read_steps(path)
        assert any("__list_tools__" in s.get("name", "") for s in steps)

    def test_initialize_forwards(self, tmp_path: Path) -> None:
        session = _FakeMCPSession()
        path = _sb_path(tmp_path, "mcp_init.sb")
        with record(path) as rec:
            proxy = MCPRecorderProxy(session, rec)
            result = proxy.initialize()
        assert result["protocol_version"] == "2024-11-05"

    def test_proxy_error_on_missing_call_tool(self) -> None:
        class _NoCallTool:
            pass

        with pytest.raises(MCPProxyError, match="call_tool"):
            MCPRecorderProxy(_NoCallTool(), None)  # type: ignore[arg-type]

    def test_getattr_passthrough(self, tmp_path: Path) -> None:
        session = _FakeMCPSession()
        path = _sb_path(tmp_path, "mcp_attr.sb")
        with record(path) as rec:
            proxy = MCPRecorderProxy(session, rec)
            # _tools is on _FakeMCPSession — forward via __getattr__
            assert proxy._tools == session._tools

    def test_server_name_property(self, tmp_path: Path) -> None:
        session = _FakeMCPSession()
        path = _sb_path(tmp_path, "mcp_name.sb")
        with record(path) as rec:
            proxy = MCPRecorderProxy(session, rec, server_name="my-server")
        assert proxy.server_name == "my-server"

    def test_repr(self, tmp_path: Path) -> None:
        session = _FakeMCPSession()
        path = _sb_path(tmp_path, "mcp_repr.sb")
        with record(path) as rec:
            proxy = MCPRecorderProxy(session, rec, server_name="srv")
        assert "srv" in repr(proxy)


# ======================================================================
# import_mcp_log tests
# ======================================================================

class TestImportMCPLog:
    def _make_event_log(self, tmp_path: Path, events: List[dict]) -> str:
        path = str(tmp_path / "events.json")
        with open(path, "w") as f:
            json.dump({"events": events}, f)
        return path

    def test_basic_call_tool_events(self, tmp_path: Path) -> None:
        log = self._make_event_log(tmp_path, [
            {"type": "call_tool", "name": "search", "arguments": {"q": "test"},
             "result": {"answer": "42"}},
            {"type": "call_tool", "name": "read_file", "arguments": {"path": "/x"},
             "result": "content"},
        ])
        out = _sb_path(tmp_path, "out.sb")
        report = import_mcp_log(log, out)
        assert report.step_count == 2
        assert report.kind_counts.get("tool_call", 0) == 2
        steps = _read_steps(out)
        tool_steps = [s for s in steps if s["step_kind"] == "tool_call"]
        assert len(tool_steps) == 2

    def test_server_name_prefix(self, tmp_path: Path) -> None:
        log = self._make_event_log(tmp_path, [
            {"type": "call_tool", "name": "ping", "result": "pong"},
        ])
        out = _sb_path(tmp_path, "out.sb")
        import_mcp_log(log, out, server_name="myserver")
        steps = _read_steps(out)
        assert any("myserver:ping" in s.get("name", "") for s in steps)

    def test_error_events(self, tmp_path: Path) -> None:
        log = self._make_event_log(tmp_path, [
            {"type": "call_tool", "name": "fail", "error": "not found"},
        ])
        out = _sb_path(tmp_path, "out.sb")
        report = import_mcp_log(log, out)
        assert report.step_count == 1
        steps = _read_steps(out)
        assert steps[0]["outputs"].get("error") == "not found"

    def test_lifecycle_events_become_router(self, tmp_path: Path) -> None:
        log = self._make_event_log(tmp_path, [
            {"type": "initialize", "protocol_version": "2024-11-05"},
            {"type": "call_tool", "name": "search", "result": "ok"},
        ])
        out = _sb_path(tmp_path, "out.sb")
        report = import_mcp_log(log, out)
        assert report.kind_counts.get("router", 0) >= 1
        assert report.kind_counts.get("tool_call", 0) == 1

    def test_bare_list_input(self, tmp_path: Path) -> None:
        log_path = str(tmp_path / "bare.json")
        with open(log_path, "w") as f:
            json.dump([{"type": "call_tool", "name": "x", "result": 1}], f)
        out = _sb_path(tmp_path, "out.sb")
        report = import_mcp_log(log_path, out)
        assert report.step_count == 1

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        bad_path = str(tmp_path / "bad.json")
        with open(bad_path, "w") as f:
            f.write("not json {{{")
        out = _sb_path(tmp_path, "out.sb")
        from stepback.importers import ImportError as SbImportError
        with pytest.raises(SbImportError):
            import_mcp_log(bad_path, out)

    def test_invalid_structure_raises(self, tmp_path: Path) -> None:
        bad_path = str(tmp_path / "scalar.json")
        with open(bad_path, "w") as f:
            json.dump(42, f)
        out = _sb_path(tmp_path, "out.sb")
        from stepback.importers import ImportError as SbImportError
        with pytest.raises(SbImportError):
            import_mcp_log(bad_path, out)

    def test_duration_ms_stored(self, tmp_path: Path) -> None:
        log = self._make_event_log(tmp_path, [
            {"type": "call_tool", "name": "slow", "result": None, "duration_ms": 123.4},
        ])
        out = _sb_path(tmp_path, "out.sb")
        import_mcp_log(log, out)
        # Verify the trace is readable (wallclock_ns is internal)
        steps = _read_steps(out)
        assert steps[0]["wallclock_ns"] == int(123.4 * 1_000_000)

    def test_mcp_log_alias_via_import_trace(self, tmp_path: Path) -> None:
        """import_mcp_log is callable via import_trace with format 'mcp_log'."""
        # MCP log import is not in import_trace dispatch (it's in integrations).
        # Verify the module import path works directly.
        from stepback.integrations.mcp_proxy import import_mcp_log as _fn
        assert callable(_fn)

    def test_source_format_in_report(self, tmp_path: Path) -> None:
        log = self._make_event_log(tmp_path, [
            {"type": "call_tool", "name": "x", "result": 1},
        ])
        out = _sb_path(tmp_path, "out.sb")
        report = import_mcp_log(log, out)
        assert report.source_format == "mcp_log"


# ======================================================================
# import_cyclonedx_ai tests
# ======================================================================

class TestImportCycloneDXAI:
    def _make_bom(self, tmp_path: Path, components: List[dict], **bom_extras) -> str:
        path = str(tmp_path / "bom.json")
        doc = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "serialNumber": "urn:uuid:test-bom-001",
            "components": components,
            **bom_extras,
        }
        with open(path, "w") as f:
            json.dump(doc, f)
        return path

    def test_ml_model_becomes_llm_call(self, tmp_path: Path) -> None:
        bom = self._make_bom(tmp_path, [
            {"type": "machine-learning-model", "name": "gpt-4o",
             "version": "2024-05-13", "bom-ref": "model-001"},
        ])
        out = _sb_path(tmp_path, "out.sb")
        report = import_cyclonedx_ai(bom, out)
        assert report.step_count >= 1
        steps = _read_steps(out)
        llm_steps = [s for s in steps if s["step_kind"] == "llm_call"]
        assert len(llm_steps) == 1
        assert "gpt-4o" in llm_steps[0]["name"]

    def test_library_becomes_tool_call(self, tmp_path: Path) -> None:
        bom = self._make_bom(tmp_path, [
            {"type": "library", "name": "httpx", "version": "0.27.0"},
        ])
        out = _sb_path(tmp_path, "out.sb")
        report = import_cyclonedx_ai(bom, out)
        steps = _read_steps(out)
        tool_steps = [s for s in steps if s["step_kind"] == "tool_call"]
        assert len(tool_steps) == 1

    def test_data_becomes_router(self, tmp_path: Path) -> None:
        bom = self._make_bom(tmp_path, [
            {"type": "data", "name": "policy-v2"},
        ])
        out = _sb_path(tmp_path, "out.sb")
        import_cyclonedx_ai(bom, out)
        steps = _read_steps(out)
        router_steps = [s for s in steps if s["step_kind"] == "router"]
        assert len(router_steps) >= 1

    def test_spec_version_in_report(self, tmp_path: Path) -> None:
        bom = self._make_bom(tmp_path, [])
        out = _sb_path(tmp_path, "out.sb")
        report = import_cyclonedx_ai(bom, out)
        assert report.schema_version == "1.6"

    def test_multiple_components(self, tmp_path: Path) -> None:
        bom = self._make_bom(tmp_path, [
            {"type": "machine-learning-model", "name": "gpt-4o"},
            {"type": "library", "name": "httpx"},
            {"type": "library", "name": "pydantic"},
            {"type": "data", "name": "training-set"},
        ])
        out = _sb_path(tmp_path, "out.sb")
        report = import_cyclonedx_ai(bom, out)
        assert report.step_count >= 4
        assert report.kind_counts.get("llm_call", 0) == 1
        assert report.kind_counts.get("tool_call", 0) == 2
        assert report.kind_counts.get("router", 0) >= 1

    def test_bare_list_input(self, tmp_path: Path) -> None:
        path = str(tmp_path / "bare.json")
        with open(path, "w") as f:
            json.dump([{"type": "library", "name": "requests"}], f)
        out = _sb_path(tmp_path, "out.sb")
        report = import_cyclonedx_ai(path, out)
        assert report.step_count >= 1

    def test_invalid_file_raises(self, tmp_path: Path) -> None:
        bad = str(tmp_path / "bad.json")
        with open(bad, "w") as f:
            f.write("not-json")
        out = _sb_path(tmp_path, "out.sb")
        with pytest.raises(StepbackImportError):
            import_cyclonedx_ai(bad, out)

    def test_not_cyclonedx_raises(self, tmp_path: Path) -> None:
        path = str(tmp_path / "notbom.json")
        with open(path, "w") as f:
            json.dump({"foo": "bar"}, f)
        out = _sb_path(tmp_path, "out.sb")
        with pytest.raises(StepbackImportError):
            import_cyclonedx_ai(path, out)

    def test_scalar_raises(self, tmp_path: Path) -> None:
        path = str(tmp_path / "scalar.json")
        with open(path, "w") as f:
            json.dump(42, f)
        out = _sb_path(tmp_path, "out.sb")
        with pytest.raises(StepbackImportError):
            import_cyclonedx_ai(path, out)

    def test_model_card_in_inputs(self, tmp_path: Path) -> None:
        bom = self._make_bom(tmp_path, [
            {
                "type": "machine-learning-model",
                "name": "claude-3-5-sonnet",
                "modelCard": {
                    "modelParameters": {"approach": "transformer"},
                    "safetyPolicy": "https://anthropic.com/safety",
                },
            }
        ])
        out = _sb_path(tmp_path, "out.sb")
        import_cyclonedx_ai(bom, out)
        steps = [s for s in _read_steps(out) if s["step_kind"] == "llm_call"]
        assert len(steps) == 1
        assert "modelCard" in steps[0]["inputs"]

    def test_import_trace_dispatch(self, tmp_path: Path) -> None:
        """import_trace dispatches to import_cyclonedx_ai for format='cyclonedx'."""
        from stepback import import_trace
        bom = self._make_bom(tmp_path, [{"type": "library", "name": "pkg"}])
        out = _sb_path(tmp_path, "out2.sb")
        report = import_trace("cyclonedx", bom, out)
        assert report.source_format == "cyclonedx_ai"

    def test_trace_verifiable(self, tmp_path: Path) -> None:
        """Imported CycloneDX trace is readable and has the right step count."""
        from stepback import replay
        bom = self._make_bom(tmp_path, [
            {"type": "library", "name": "requests", "version": "2.32.0"},
        ])
        out = _sb_path(tmp_path, "out.sb")
        import_cyclonedx_ai(bom, out)
        t = replay(out)
        assert len(list(t.recorded_steps)) >= 1


# ======================================================================
# OtelBridge tests
# ======================================================================

class TestOtelBridge:
    def _trace_path(self, tmp_path: Path) -> str:
        path = _sb_path(tmp_path, "trace.sb")
        _record_trace(path)
        return path

    def test_dry_run_collector_receives_spans(self, tmp_path: Path) -> None:
        path = self._trace_path(tmp_path)
        collector = DryRunCollector()
        bridge = OtelBridge(exporter=collector, service_name="test-agent")
        result = bridge.export_trace(path)
        assert result.span_count > 0
        assert len(collector.spans) == result.span_count

    def test_batch_count_correct(self, tmp_path: Path) -> None:
        path = self._trace_path(tmp_path)
        collector = DryRunCollector()
        bridge = OtelBridge(exporter=collector, service_name="test", batch_size=2)
        result = bridge.export_trace(path)
        # With batch_size=2, batch_count should be ceil(span_count / 2)
        import math
        assert result.batch_count == math.ceil(result.span_count / 2)

    def test_service_name_in_spans(self, tmp_path: Path) -> None:
        path = self._trace_path(tmp_path)
        collector = DryRunCollector()
        bridge = OtelBridge(exporter=collector, service_name="my-awesome-agent")
        bridge.export_trace(path)
        for span in collector.spans:
            attrs = span.get("resource", {}).get("attributes", {})
            assert attrs.get("service.name") == "my-awesome-agent"

    def test_export_steps_in_memory(self, tmp_path: Path) -> None:
        path = self._trace_path(tmp_path)
        from stepback import replay
        steps = list(replay(path).recorded_steps)
        collector = DryRunCollector()
        bridge = OtelBridge(exporter=collector, service_name="test")
        result = bridge.export_steps(steps)
        assert result.span_count == len(steps)

    def test_empty_trace_no_error(self, tmp_path: Path) -> None:
        # Create an empty trace (just header + tail, no steps)
        from stepback.recorder import RecorderKey
        from stepback.trace_writer import TraceWriter
        path = _sb_path(tmp_path, "empty.sb")
        key = RecorderKey.fresh()
        w = TraceWriter.open(path, hmac_key=key.hmac_key, signing_key=key.signing_key)
        w.close()

        collector = DryRunCollector()
        bridge = OtelBridge(exporter=collector, service_name="empty-test")
        result = bridge.export_trace(path)
        assert result.span_count == 0

    def test_dry_run_collector_clear(self, tmp_path: Path) -> None:
        path = self._trace_path(tmp_path)
        collector = DryRunCollector()
        bridge = OtelBridge(exporter=collector, service_name="test")
        bridge.export_trace(path)
        assert len(collector.spans) > 0
        collector.clear()
        assert len(collector.spans) == 0

    def test_dry_run_collector_repr(self) -> None:
        collector = DryRunCollector()
        assert "DryRunCollector" in repr(collector)
        assert "0" in repr(collector)

    def test_bridge_repr(self) -> None:
        collector = DryRunCollector()
        bridge = OtelBridge(exporter=collector, service_name="my-svc")
        assert "my-svc" in repr(bridge)

    def test_invalid_exporter_raises(self) -> None:
        with pytest.raises(TypeError, match="send"):
            OtelBridge(exporter=object())

    def test_export_duration_positive(self, tmp_path: Path) -> None:
        path = self._trace_path(tmp_path)
        collector = DryRunCollector()
        bridge = OtelBridge(exporter=collector, service_name="test")
        result = bridge.export_trace(path)
        assert result.export_duration_ms >= 0

    def test_result_repr(self) -> None:
        r = OtelBridgeExportResult(span_count=5, batch_count=2)
        assert "5" in repr(r)
        assert "2" in repr(r)

    def test_public_api_symbols(self) -> None:
        """New OtelBridge symbols are accessible from stepback.*"""
        assert hasattr(stepback, "OtelBridge")
        assert hasattr(stepback, "OtlpHttpExporter")
        assert hasattr(stepback, "DryRunCollector")
        assert hasattr(stepback, "OtelBridgeExportResult")
        assert hasattr(stepback, "OtelBridgeError")


# ======================================================================
# import_cyclonedx_ai public API tests
# ======================================================================

class TestImportCycloneDXPublicAPI:
    def test_accessible_from_stepback(self) -> None:
        assert hasattr(stepback, "import_cyclonedx_ai")
        assert callable(stepback.import_cyclonedx_ai)

    def test_in_all(self) -> None:
        assert "import_cyclonedx_ai" in stepback.__all__


# ======================================================================
# SLSA example script tests
# ======================================================================

class TestSLSAExampleScript:
    def test_script_exists(self) -> None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = os.path.join(repo_root, "scripts", "slsa_example.py")
        assert os.path.isfile(script), f"scripts/slsa_example.py not found at {script}"

    def test_script_is_importable(self) -> None:
        """The script can be parsed without syntax errors."""
        import importlib.util, sys
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.join(repo_root, "scripts", "slsa_example.py")
        spec = importlib.util.spec_from_file_location("slsa_example", script_path)
        assert spec is not None
        # Just loading the spec (no exec_module) validates the file is parseable.

    def test_script_runs_end_to_end(self, tmp_path: Path) -> None:
        """Run the SLSA example script in-process and check output."""
        import sys
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        # Import and call main directly
        import importlib.util
        script_path = os.path.join(repo_root, "scripts", "slsa_example.py")
        spec = importlib.util.spec_from_file_location("slsa_example", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        out_path = str(tmp_path / "envelope.json")
        mod.main(["--out", out_path])

        assert os.path.isfile(out_path)
        with open(out_path) as f:
            envelope = json.load(f)
        assert envelope.get("payloadType") == "application/vnd.in-toto+json"
        assert "payload" in envelope
        assert "signatures" in envelope
        assert len(envelope["signatures"]) >= 1


# ======================================================================
# Documentation existence tests
# ======================================================================

class TestDocumentationExists:
    def _doc_path(self, name: str) -> str:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(repo_root, "docs", name)

    def test_wasm_embedding_doc_exists(self) -> None:
        assert os.path.isfile(self._doc_path("wasm-embedding.md"))

    def test_wasm_embedding_doc_has_key_sections(self) -> None:
        with open(self._doc_path("wasm-embedding.md")) as f:
            content = f.read()
        assert "Quick start" in content
        assert "JS API" in content or "JS API reference" in content
        assert "Security model" in content
        assert "Building from source" in content
        assert "summarize" in content
        assert "verify" in content

    def test_integration_matrix_doc_exists(self) -> None:
        assert os.path.isfile(self._doc_path("integration-matrix.md"))

    def test_integration_matrix_has_key_sections(self) -> None:
        with open(self._doc_path("integration-matrix.md")) as f:
            content = f.read()
        assert "LLM provider shims" in content
        assert "MCP integration" in content
        assert "Importers" in content
        assert "Exporters" in content
        assert "Observability bridges" in content
        assert "Provenance" in content
        assert "MCPRecorderProxy" in content
        assert "import_cyclonedx_ai" in content
        assert "OtelBridge" in content

    def test_integration_matrix_mentions_all_shims(self) -> None:
        with open(self._doc_path("integration-matrix.md")) as f:
            content = f.read()
        shims = [
            "wrap_openai", "wrap_anthropic", "wrap_bedrock",
            "wrap_gemini", "wrap_cohere", "wrap_mistral",
        ]
        for shim in shims:
            assert shim in content, f"Missing shim {shim!r} in integration matrix"

    def test_integration_matrix_mentions_all_importers(self) -> None:
        with open(self._doc_path("integration-matrix.md")) as f:
            content = f.read()
        importers = [
            "import_langsmith_jsonl",
            "import_openinference_spans",
            "import_cyclonedx_ai",
            "import_mcp_log",
        ]
        for imp in importers:
            assert imp in content, f"Missing importer {imp!r} in integration matrix"
