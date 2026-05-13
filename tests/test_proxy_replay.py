"""Tests for the ``POST /v1/replay`` endpoint added in step 71.

All tests run against a real loopback HTTP server so there are no mocks
and no network calls outside localhost.  The replay endpoint streams NDJSON,
so we read the full response body and split by newline to get the events.
"""
from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from typing import List

import pytest

from stepback.canonical import canonical_json, hash_obj, sha256_hex
from stepback.proxy import ProxyHTTPServer, ProxyState
from stepback.recorder import RecorderKey
from stepback.replay import replay_events, Executor
from stepback.substitutions import SubstitutionSet, ToolOutputSubstitution
from stepback.trace_writer import TraceWriter


# ------------------------------------------------------------------ helpers


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def server(tmp_path):
    state = ProxyState(write_dir=str(tmp_path))
    port = _free_port()
    srv = ProxyHTTPServer(("127.0.0.1", port), state)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    deadline = time.time() + 2.0
    while time.time() < deadline:
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=0.2)
            c.request("GET", "/healthz")
            r = c.getresponse()
            r.read()
            c.close()
            if r.status == 200:
                break
        except OSError:
            time.sleep(0.02)
    yield ("127.0.0.1", port, state, str(tmp_path))
    srv.shutdown()
    srv.server_close()
    state.close_all()


def _request(host, port, method, path, body=None):
    """Send a JSON request and parse the response as JSON."""
    c = http.client.HTTPConnection(host, port, timeout=5.0)
    headers = {}
    raw = b""
    if body is not None:
        raw = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(raw))
    c.request(method, path, body=raw, headers=headers)
    r = c.getresponse()
    data = r.read()
    c.close()
    payload = json.loads(data.decode("utf-8")) if data else {}
    return r.status, payload


def _request_ndjson(host, port, method, path, body=None):
    """Send a JSON request and parse the response as NDJSON (list of dicts)."""
    c = http.client.HTTPConnection(host, port, timeout=5.0)
    headers = {}
    raw = b""
    if body is not None:
        raw = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(raw))
    c.request(method, path, body=raw, headers=headers)
    r = c.getresponse()
    status = r.status
    data = r.read()
    c.close()
    text = data.decode("utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    try:
        events = [json.loads(ln) for ln in lines]
    except json.JSONDecodeError:
        events = []
    return status, events


def _build_llm_step(n: int) -> dict:
    """Minimal pre-hashed llm_call step dict."""
    inputs = {
        "kind": "llm_call",
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "seed": 42,
        "messages": [{"role": "user", "content": f"hello {n}"}],
        "tools": None,
        "response_format": None,
    }
    outputs = {
        "choices": [{"message": {"role": "assistant", "content": f"hi {n}"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5},
    }
    return {
        "step_id": f"step:{n}",
        "step_kind": "llm_call",
        "name": "gpt-4o-mini",
        "parent_step_id": None,
        "inputs": inputs,
        "outputs": outputs,
        "inputs_hash": hash_obj(inputs),
        "outputs_hash": hash_obj(outputs),
        "nondeterminism_hash": sha256_hex(canonical_json({})),
        "wallclock_ns": 1_000_000_000,
        "cost_usd": 0.001,
    }


def _build_tool_step(n: int, tool_result: str = "ok") -> dict:
    """Minimal pre-hashed tool_call step dict."""
    inputs = {
        "name": "my_tool",
        "arguments": {"x": n},
    }
    outputs = {"result": tool_result}
    return {
        "step_id": f"tool:{n}",
        "step_kind": "tool_call",
        "name": "my_tool",
        "parent_step_id": None,
        "inputs": inputs,
        "outputs": outputs,
        "inputs_hash": hash_obj(inputs),
        "outputs_hash": hash_obj(outputs),
        "nondeterminism_hash": sha256_hex(canonical_json({})),
        "wallclock_ns": 100_000,
        "cost_usd": 0.0,
    }


def _write_trace(path: str, steps: list) -> RecorderKey:
    """Write pre-built step dicts to a .sb file; return the RecorderKey."""
    key = RecorderKey.fresh()
    writer = TraceWriter.open(path, hmac_key=key.hmac_key, signing_key=key.signing_key)
    for step in steps:
        writer.write_step(step)
    writer.close()
    return key


def _record_and_close(host, port, steps: list) -> dict:
    """Create a trace via proxy, write steps, close it; return StartTrace body."""
    _, start = _request(host, port, "POST", "/v1/traces", {"filename": "test.sb"})
    tid = start["trace_id"]
    for s in steps:
        _request(host, port, "POST", f"/v1/traces/{tid}/steps", {"step": s})
    _request(host, port, "POST", f"/v1/traces/{tid}/end", {})
    return start


# ------------------------------------------------------------------ unit tests for replay_events()


def test_replay_events_all_cache_hits(tmp_path):
    """All steps are cache hits when no substitutions are applied."""
    from stepback.replay import replay as load_trace

    path = str(tmp_path / "t.sb")
    key = _write_trace(path, [_build_llm_step(1)])
    trace = load_trace(path, hmac_key=key.hmac_key)
    subs = SubstitutionSet()
    executor = Executor(fallback_recorded=True)

    events = list(replay_events(trace.recorded_steps, subs, executor))
    assert len(events) == 2  # 1 step_complete + 1 replay_done
    step_evt = events[0]
    assert step_evt["event"] == "step_complete"
    assert step_evt["cache_hit"] is True
    assert step_evt["dirty"] is False
    done_evt = events[1]
    assert done_evt["event"] == "replay_done"
    assert done_evt["dirty_count"] == 0
    assert done_evt["cache_hit_count"] == 1
    assert done_evt["step_count"] == 1
    assert done_evt["real_executions"] == 0


def test_replay_events_tool_output_substitution_marks_dirty(tmp_path):
    """Applying a ToolOutputSubstitution makes the step dirty."""
    from stepback.replay import replay as load_trace

    path = str(tmp_path / "t.sb")
    key = _write_trace(path, [_build_tool_step(1, tool_result="original")])
    trace = load_trace(path, hmac_key=key.hmac_key)
    subs = SubstitutionSet()
    subs.add(ToolOutputSubstitution(at_step="tool:1", fake_response="replaced"))
    executor = Executor(fallback_recorded=True)

    events = list(replay_events(trace.recorded_steps, subs, executor))
    step_evt = events[0]
    assert step_evt["event"] == "step_complete"
    assert step_evt["dirty"] is True
    assert step_evt["cache_hit"] is False
    assert step_evt["output_changed"] is True
    done_evt = events[1]
    assert done_evt["dirty_count"] == 1
    assert done_evt["cache_hit_count"] == 0


def test_replay_events_empty_trace(tmp_path):
    """An empty trace yields only the replay_done event."""
    from stepback.replay import replay as load_trace

    path = str(tmp_path / "empty.sb")
    key = _write_trace(path, [])
    trace = load_trace(path, hmac_key=key.hmac_key)
    events = list(replay_events(
        trace.recorded_steps, SubstitutionSet(), Executor(fallback_recorded=True)
    ))
    assert len(events) == 1
    assert events[0]["event"] == "replay_done"
    assert events[0]["step_count"] == 0


def test_replay_events_multiple_steps(tmp_path):
    """Events are yielded in step order; summary counts are consistent."""
    from stepback.replay import replay as load_trace

    path = str(tmp_path / "multi.sb")
    key = _write_trace(path, [_build_llm_step(i) for i in range(1, 6)])
    trace = load_trace(path, hmac_key=key.hmac_key)
    events = list(replay_events(
        trace.recorded_steps, SubstitutionSet(), Executor(fallback_recorded=True)
    ))
    step_events = [e for e in events if e["event"] == "step_complete"]
    assert len(step_events) == 5
    assert [e["step_id"] for e in step_events] == [f"step:{i}" for i in range(1, 6)]
    done = events[-1]
    assert done["event"] == "replay_done"
    assert done["step_count"] == 5
    assert done["dirty_count"] + done["cache_hit_count"] == 5


def test_replay_events_step_complete_fields(tmp_path):
    """step_complete events contain all required fields."""
    from stepback.replay import replay as load_trace

    path = str(tmp_path / "fields.sb")
    key = _write_trace(path, [_build_llm_step(1)])
    trace = load_trace(path, hmac_key=key.hmac_key)
    events = list(replay_events(
        trace.recorded_steps, SubstitutionSet(), Executor(fallback_recorded=True)
    ))
    evt = events[0]
    for field in ("event", "step_id", "step_kind", "dirty", "cache_hit",
                  "cost_usd", "current_inputs_hash", "recorded_inputs_hash",
                  "output_changed"):
        assert field in evt, f"missing field: {field}"


def test_replay_events_replay_done_fields(tmp_path):
    """replay_done event contains all required fields."""
    from stepback.replay import replay as load_trace

    path = str(tmp_path / "done_fields.sb")
    key = _write_trace(path, [_build_llm_step(1)])
    trace = load_trace(path, hmac_key=key.hmac_key)
    events = list(replay_events(
        trace.recorded_steps, SubstitutionSet(), Executor(fallback_recorded=True)
    ))
    done = events[-1]
    for field in ("event", "step_count", "dirty_count", "cache_hit_count",
                  "total_cost_usd", "real_executions"):
        assert field in done, f"missing field in replay_done: {field}"


# ------------------------------------------------------------------ HTTP endpoint tests


def test_replay_endpoint_no_path(server):
    host, port, _, _ = server
    status, events = _request_ndjson(host, port, "POST", "/v1/replay", {})
    assert status == 400


def test_replay_endpoint_missing_file(server):
    host, port, _, _ = server
    status, events = _request_ndjson(
        host, port, "POST", "/v1/replay",
        {"path": "/nonexistent/file.sb"},
    )
    assert status == 404


def test_replay_endpoint_invalid_hmac_hex(server):
    host, port, _, _ = server
    status, _ = _request_ndjson(
        host, port, "POST", "/v1/replay",
        {"path": "/some/file.sb", "hmac_key_hex": "gg"},
    )
    assert status == 400


def test_replay_endpoint_all_cache_hits(server, tmp_path):
    """End-to-end: record a trace via proxy, then replay it without substitutions."""
    host, port, _, write_dir = server
    steps = [_build_llm_step(i) for i in range(1, 4)]
    start = _record_and_close(host, port, steps)
    path = start["path"]
    hmac_key_hex = start["hmac_key_hex"]

    status, events = _request_ndjson(
        host, port, "POST", "/v1/replay",
        {"path": path, "hmac_key_hex": hmac_key_hex},
    )
    assert status == 200
    step_events = [e for e in events if e["event"] == "step_complete"]
    assert len(step_events) == 3
    assert all(e["cache_hit"] for e in step_events)
    done = next(e for e in events if e["event"] == "replay_done")
    assert done["dirty_count"] == 0
    assert done["cache_hit_count"] == 3
    assert done["step_count"] == 3
    assert done["real_executions"] == 0


def test_replay_endpoint_step_events_have_required_fields(server):
    host, port, _, _ = server
    steps = [_build_llm_step(1)]
    start = _record_and_close(host, port, steps)

    _, events = _request_ndjson(
        host, port, "POST", "/v1/replay",
        {"path": start["path"], "hmac_key_hex": start["hmac_key_hex"]},
    )
    step_evt = next(e for e in events if e["event"] == "step_complete")
    for field in ("step_id", "step_kind", "dirty", "cache_hit", "cost_usd",
                  "current_inputs_hash", "recorded_inputs_hash", "output_changed"):
        assert field in step_evt, f"missing field: {field}"


def test_replay_endpoint_replay_done_has_required_fields(server):
    host, port, _, _ = server
    start = _record_and_close(host, port, [_build_llm_step(1)])

    _, events = _request_ndjson(
        host, port, "POST", "/v1/replay",
        {"path": start["path"], "hmac_key_hex": start["hmac_key_hex"]},
    )
    done = next(e for e in events if e["event"] == "replay_done")
    for field in ("step_count", "dirty_count", "cache_hit_count",
                  "total_cost_usd", "real_executions"):
        assert field in done, f"missing field in replay_done: {field}"


def test_replay_endpoint_tool_output_substitution(server):
    """Substituting a tool output via the HTTP endpoint marks the step dirty."""
    host, port, _, _ = server
    tool_step = _build_tool_step(1, tool_result="original")
    start = _record_and_close(host, port, [tool_step])

    _, events = _request_ndjson(
        host, port, "POST", "/v1/replay",
        {
            "path": start["path"],
            "hmac_key_hex": start["hmac_key_hex"],
            "substitutions": [
                {"step_id": "tool:1", "kind": "tool_output", "value": "replaced"}
            ],
        },
    )
    step_evt = next(e for e in events if e["event"] == "step_complete")
    assert step_evt["dirty"] is True
    assert step_evt["output_changed"] is True
    done = next(e for e in events if e["event"] == "replay_done")
    assert done["dirty_count"] == 1


def test_replay_endpoint_without_hmac_verification(server):
    """Replay without hmac_key_hex loads the trace without verification."""
    host, port, _, _ = server
    start = _record_and_close(host, port, [_build_llm_step(1)])

    status, events = _request_ndjson(
        host, port, "POST", "/v1/replay",
        {"path": start["path"]},  # no hmac_key_hex
    )
    assert status == 200
    done = next(e for e in events if e["event"] == "replay_done")
    assert done["step_count"] == 1


def test_replay_endpoint_bad_substitution_kind(server):
    host, port, _, _ = server
    start = _record_and_close(host, port, [_build_llm_step(1)])

    status, _ = _request_ndjson(
        host, port, "POST", "/v1/replay",
        {
            "path": start["path"],
            "substitutions": [{"step_id": "step:1", "kind": "unknown_kind"}],
        },
    )
    assert status == 400


def test_replay_endpoint_substitution_missing_step_id(server):
    host, port, _, _ = server
    start = _record_and_close(host, port, [_build_llm_step(1)])

    status, _ = _request_ndjson(
        host, port, "POST", "/v1/replay",
        {
            "path": start["path"],
            "substitutions": [{"kind": "tool_output", "value": "x"}],  # no step_id
        },
    )
    assert status == 400


def test_replay_endpoint_content_type_ndjson(server):
    """Response Content-Type is application/x-ndjson."""
    host, port, _, _ = server
    start = _record_and_close(host, port, [_build_llm_step(1)])

    c = http.client.HTTPConnection(host, port, timeout=5.0)
    raw = json.dumps({"path": start["path"]}).encode("utf-8")
    c.request(
        "POST", "/v1/replay",
        body=raw,
        headers={"Content-Type": "application/json", "Content-Length": str(len(raw))},
    )
    r = c.getresponse()
    r.read()
    c.close()
    assert "ndjson" in r.getheader("Content-Type", "").lower()


def test_replay_endpoint_multiple_substitutions(server):
    """Multiple tool_output substitutions all applied in a single replay."""
    host, port, _, _ = server
    steps = [_build_tool_step(1, "a"), _build_tool_step(2, "b")]
    start = _record_and_close(host, port, steps)

    _, events = _request_ndjson(
        host, port, "POST", "/v1/replay",
        {
            "path": start["path"],
            "substitutions": [
                {"step_id": "tool:1", "kind": "tool_output", "value": "A"},
                {"step_id": "tool:2", "kind": "tool_output", "value": "B"},
            ],
        },
    )
    step_events = [e for e in events if e["event"] == "step_complete"]
    assert all(e["dirty"] for e in step_events)
    done = next(e for e in events if e["event"] == "replay_done")
    assert done["dirty_count"] == 2


def test_replay_endpoint_event_ordering(server):
    """step_complete events appear before replay_done."""
    host, port, _, _ = server
    steps = [_build_llm_step(i) for i in range(1, 4)]
    start = _record_and_close(host, port, steps)

    _, events = _request_ndjson(
        host, port, "POST", "/v1/replay",
        {"path": start["path"], "hmac_key_hex": start["hmac_key_hex"]},
    )
    assert events[-1]["event"] == "replay_done"
    step_events = [e for e in events[:-1] if e["event"] == "step_complete"]
    assert len(step_events) == 3


# ------------------------------------------------------------------ grpc unit tests (no network)


def test_grpc_replay_trace_stream_no_path(tmp_path):
    """_replay_trace_stream yields an error event when 'path' is absent."""
    from stepback.proxy.grpc_server import _replay_trace_stream
    from stepback.proxy.storage import ProxyState

    state = ProxyState(write_dir=str(tmp_path))
    events = list(_replay_trace_stream(state, {}))
    assert len(events) == 1
    assert events[0]["event"] == "error"


def test_grpc_replay_trace_stream_missing_file(tmp_path):
    """_replay_trace_stream yields an error event for a nonexistent file."""
    from stepback.proxy.grpc_server import _replay_trace_stream
    from stepback.proxy.storage import ProxyState

    state = ProxyState(write_dir=str(tmp_path))
    events = list(_replay_trace_stream(state, {"path": str(tmp_path / "nope.sb")}))
    assert events[0]["event"] == "error"


def test_grpc_replay_trace_stream_all_cache_hits(tmp_path):
    """_replay_trace_stream yields step_complete + replay_done for a clean trace."""
    from stepback.proxy.grpc_server import _replay_trace_stream
    from stepback.proxy.storage import ProxyState

    path = str(tmp_path / "g.sb")
    key = _write_trace(path, [_build_llm_step(1)])

    state = ProxyState(write_dir=str(tmp_path))
    events = list(_replay_trace_stream(state, {
        "path": path,
        "hmac_key": key.hmac_key,
    }))
    step_evts = [e for e in events if e["event"] == "step_complete"]
    assert len(step_evts) == 1
    assert step_evts[0]["cache_hit"] is True
    done = next(e for e in events if e["event"] == "replay_done")
    assert done["dirty_count"] == 0


def test_grpc_replay_trace_stream_with_substitution(tmp_path):
    """_replay_trace_stream applies substitutions correctly."""
    from stepback.proxy.grpc_server import _replay_trace_stream
    from stepback.proxy.storage import ProxyState

    path = str(tmp_path / "sub.sb")
    key = _write_trace(path, [_build_tool_step(1, "original")])

    state = ProxyState(write_dir=str(tmp_path))
    events = list(_replay_trace_stream(state, {
        "path": path,
        "hmac_key": key.hmac_key,
        "substitutions": [
            {"step_id": "tool:1", "kind": "tool_output", "value": "new"}
        ],
    }))
    step_evt = next(e for e in events if e["event"] == "step_complete")
    assert step_evt["dirty"] is True


def test_grpc_build_substitution_set_tool_output():
    """_build_substitution_set correctly parses a tool_output spec."""
    from stepback.proxy.grpc_server import _build_substitution_set
    import json as _json

    subs = _build_substitution_set([
        {
            "step_id": "step:1",
            "kind": "tool_output",
            "value_json": _json.dumps({"result": "x"}).encode("utf-8"),
        }
    ])
    assert len(subs.items) == 1
    s = subs.items[0]
    assert s.at_step == "step:1"
    assert s.is_output_forcing()


def test_grpc_build_substitution_set_plain_value():
    """_build_substitution_set also works with a plain Python value (no value_json)."""
    from stepback.proxy.grpc_server import _build_substitution_set

    subs = _build_substitution_set([
        {"step_id": "step:2", "kind": "tool_output", "value": 42}
    ])
    assert subs.items[0].fake_response == 42


def test_grpc_build_substitution_set_unknown_kind_raises():
    from stepback.proxy.grpc_server import _build_substitution_set
    from stepback.proxy.storage import ProxyError

    with pytest.raises(ProxyError, match="unsupported"):
        _build_substitution_set([
            {"step_id": "step:1", "kind": "model_swap"}
        ])


def test_grpc_build_substitution_set_missing_step_id_raises():
    from stepback.proxy.grpc_server import _build_substitution_set
    from stepback.proxy.storage import ProxyError

    with pytest.raises(ProxyError, match="step_id"):
        _build_substitution_set([
            {"kind": "tool_output", "value": "x"}
        ])



# ------------------------------------------------------------------ helpers


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def server(tmp_path):
    state = ProxyState(write_dir=str(tmp_path))
    port = _free_port()
    srv = ProxyHTTPServer(("127.0.0.1", port), state)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    deadline = time.time() + 2.0
    while time.time() < deadline:
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=0.2)
            c.request("GET", "/healthz")
            r = c.getresponse()
            r.read()
            c.close()
            if r.status == 200:
                break
        except OSError:
            time.sleep(0.02)
    yield ("127.0.0.1", port, state, str(tmp_path))
    srv.shutdown()
    srv.server_close()
    state.close_all()


def _request(host, port, method, path, body=None):
    """Send a JSON request and parse the response as JSON."""
    c = http.client.HTTPConnection(host, port, timeout=5.0)
    headers = {}
    raw = b""
    if body is not None:
        raw = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(raw))
    c.request(method, path, body=raw, headers=headers)
    r = c.getresponse()
    data = r.read()
    c.close()
    payload = json.loads(data.decode("utf-8")) if data else {}
    return r.status, payload


def _request_ndjson(host, port, method, path, body=None) -> tuple[int, list[dict]]:
    """Send a JSON request and parse the response as NDJSON (list of dicts)."""
    c = http.client.HTTPConnection(host, port, timeout=5.0)
    headers = {}
    raw = b""
    if body is not None:
        raw = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(raw))
    c.request(method, path, body=raw, headers=headers)
    r = c.getresponse()
    data = r.read()
    c.close()
    lines = [ln for ln in data.decode("utf-8").splitlines() if ln.strip()]
    events = [json.loads(ln) for ln in lines]
    return r.status, events


def _build_step(n: int, parent_n: int | None = None) -> dict:
    """Minimal llm_call step suitable for writing via the proxy."""
    inputs = {
        "kind": "llm_call",
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "seed": 42,
        "messages": [{"role": "user", "content": f"hello {n}"}],
        "tools": None,
        "response_format": None,
    }
    if parent_n is not None:
        # Simulate parent context binding used by the replay engine.
        parent_outputs = {
            "choices": [{"message": {"role": "assistant", "content": f"hi {parent_n}"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        }
        inputs["context"] = hash_obj(parent_outputs)
    outputs = {
        "choices": [{"message": {"role": "assistant", "content": f"hi {n}"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5},
    }
    return {
        "step_id": f"step:{n}",
        "step_kind": "llm_call",
        "name": "gpt-4o-mini",
        "parent_step_id": f"step:{parent_n}" if parent_n is not None else None,
        "inputs": inputs,
        "outputs": outputs,
        "inputs_hash": hash_obj(inputs),
        "outputs_hash": hash_obj(outputs),
        "nondeterminism_hash": sha256_hex(canonical_json({})),
        "wallclock_ns": 1_000_000_000,
        "cost_usd": 0.001,
    }


def _build_tool_step(n: int, tool_result: str = "ok") -> dict:
    """Minimal tool_call step."""
    inputs = {
        "name": "my_tool",
        "arguments": {"x": n},
    }
    outputs = {"result": tool_result}
    return {
        "step_id": f"tool:{n}",
        "step_kind": "tool_call",
        "name": "my_tool",
        "parent_step_id": None,
        "inputs": inputs,
        "outputs": outputs,
        "inputs_hash": hash_obj(inputs),
        "outputs_hash": hash_obj(outputs),
        "nondeterminism_hash": sha256_hex(canonical_json({})),
        "wallclock_ns": 100_000,
        "cost_usd": 0.0,
    }
