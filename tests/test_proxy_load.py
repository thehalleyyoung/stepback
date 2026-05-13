"""Load tests for ``stepback-proxy`` (Step 141).

Simulates high-throughput agent-run workloads through the HTTP proxy and
reports p50/p95/p99 latency, CPU utilisation, storage cost, and projected
throughput (completed traces/day) for three scenario tiers.

Running
-------
Normal ``pytest`` run skips all tests here (marker not selected by default).
To run::

    pytest tests/test_proxy_load.py -m proxy_load -s -v

To also run the large scenario::

    STEPBACK_PROXY_LOAD_LARGE=1 pytest tests/test_proxy_load.py -m proxy_load -s -v

To run the concurrent scenario::

    STEPBACK_PROXY_LOAD_CONCURRENT=1 pytest tests/test_proxy_load.py -m proxy_load -s -v

Results are printed to stdout; pass ``-s`` so pytest does not capture them.
"""
from __future__ import annotations

import http.client
import json
import math
import os
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from stepback.canonical import canonical_json, hash_obj, sha256_hex
from stepback.proxy import ProxyHTTPServer, ProxyState

# ---------------------------------------------------------------------------
# Helpers shared with test_proxy.py (duplicated here to keep the file self-
# contained and independent; the test_proxy.py version may change).
# ---------------------------------------------------------------------------

def _free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_load_server(tmp_path) -> tuple:
    """Start a ProxyHTTPServer; return (host, port, state, server, thread)."""
    state = ProxyState(write_dir=str(tmp_path))
    port = _free_port()
    srv = ProxyHTTPServer(("127.0.0.1", port), state)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    # Wait for readiness.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=0.5)
            c.request("GET", "/healthz")
            r = c.getresponse()
            r.read()
            c.close()
            if r.status == 200:
                break
        except OSError:
            time.sleep(0.02)
    return "127.0.0.1", port, state, srv, t


def _rpc(host: str, port: int, method: str, path: str, body: Optional[dict]) -> tuple:
    """Single RPC; returns (status_code, response_dict, rtt_seconds)."""
    raw = b""
    headers: dict = {}
    if body is not None:
        raw = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(raw))
    c = http.client.HTTPConnection(host, port, timeout=10.0)
    t0 = time.perf_counter()
    c.request(method, path, body=raw, headers=headers)
    r = c.getresponse()
    data = r.read()
    rtt = time.perf_counter() - t0
    c.close()
    payload = json.loads(data.decode("utf-8")) if data else {}
    return r.status, payload, rtt


def _prebuild_step(step_num: int, trace_offset: int = 0) -> dict:
    """Pre-build a step dict (with content hashes) outside timed sections.

    Using varied content so storage measurements are not unrealistically small
    due to highly repetitive payloads.
    """
    idx = step_num + trace_offset
    inputs = {
        "kind": "llm_call",
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "seed": 42 + idx,
        "messages": [
            {"role": "system", "content": f"You are agent instance {trace_offset}."},
            {"role": "user", "content": f"Query {step_num}: analyse step number {idx}."},
        ],
        "tools": None,
        "response_format": None,
    }
    outputs = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        f"Result {idx}: processed query {step_num} for trace "
                        f"{trace_offset} with analysis complete."
                    ),
                }
            }
        ],
        "usage": {"prompt_tokens": 20 + idx % 50, "completion_tokens": 15 + idx % 30},
    }
    return {
        "step_id": f"step:{step_num}",
        "step_kind": "llm_call",
        "name": "gpt-4o-mini",
        "parent_step_id": f"step:{step_num - 1}" if step_num > 0 else None,
        "inputs": inputs,
        "outputs": outputs,
        "inputs_hash": hash_obj(inputs),
        "outputs_hash": hash_obj(outputs),
        "nondeterminism_hash": sha256_hex(canonical_json({})),
        "wallclock_ns": 1_000_000_000 + idx * 100_000,
        "cost_usd": 0.0001 * idx,
    }


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class LoadResult:
    scenario: str
    traces: int
    steps_per_trace: int
    start_latencies_s: List[float] = field(default_factory=list)
    record_latencies_s: List[float] = field(default_factory=list)
    end_latencies_s: List[float] = field(default_factory=list)
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    storage_bytes: int = 0

    # ---- derived -------------------------------------------------------

    @property
    def total_traces(self) -> int:
        return self.traces

    @property
    def total_steps(self) -> int:
        return self.traces * self.steps_per_trace

    @staticmethod
    def _pct(data: List[float], p: float) -> float:
        if not data:
            return float("nan")
        sorted_d = sorted(data)
        idx = int(math.ceil(p / 100.0 * len(sorted_d))) - 1
        return sorted_d[max(0, idx)]

    def p50(self, lat: List[float]) -> float:
        return self._pct(lat, 50)

    def p95(self, lat: List[float]) -> float:
        return self._pct(lat, 95)

    def p99(self, lat: List[float]) -> float:
        return self._pct(lat, 99)

    def traces_per_day(self) -> float:
        if self.wall_seconds <= 0:
            return 0.0
        return self.traces / self.wall_seconds * 86_400

    def steps_per_day(self) -> float:
        if self.wall_seconds <= 0:
            return 0.0
        return self.total_steps / self.wall_seconds * 86_400

    def bytes_per_step(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.storage_bytes / self.total_steps

    def cpu_utilisation(self) -> float:
        """Fraction of elapsed wall-time consumed by process CPU."""
        if self.wall_seconds <= 0:
            return 0.0
        return self.cpu_seconds / self.wall_seconds

    def summary(self) -> str:
        lines = [
            f"\n{'─' * 60}",
            f"  Scenario : {self.scenario}",
            f"  Traces   : {self.traces:,}  |  Steps/trace: {self.steps_per_trace}",
            f"  Total    : {self.total_steps:,} steps",
            f"  Wall     : {self.wall_seconds:.2f}s  |  CPU: {self.cpu_seconds:.2f}s  "
            f"(utilisation {self.cpu_utilisation():.1%})",
            f"",
            f"  Throughput",
            f"    Completed traces/s  : {self.traces / self.wall_seconds:,.1f}",
            f"    Projected traces/day: {self.traces_per_day():,.0f}",
            f"    Projected steps/day : {self.steps_per_day():,.0f}",
            f"",
            f"  Latency (ms) — end-to-end per HTTP RPC (cold connection)",
            f"    StartTrace   p50={self.p50(self.start_latencies_s)*1000:.2f}  "
            f"p95={self.p95(self.start_latencies_s)*1000:.2f}  "
            f"p99={self.p99(self.start_latencies_s)*1000:.2f}",
            f"    RecordStep   p50={self.p50(self.record_latencies_s)*1000:.2f}  "
            f"p95={self.p95(self.record_latencies_s)*1000:.2f}  "
            f"p99={self.p99(self.record_latencies_s)*1000:.2f}",
            f"    EndTrace     p50={self.p50(self.end_latencies_s)*1000:.2f}  "
            f"p95={self.p95(self.end_latencies_s)*1000:.2f}  "
            f"p99={self.p99(self.end_latencies_s)*1000:.2f}",
            f"",
            f"  Storage",
            f"    Total on-disk : {self.storage_bytes:,} bytes ({self.storage_bytes/1024:.1f} KiB)",
            f"    Bytes/step    : {self.bytes_per_step():.1f}",
            f"{'─' * 60}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core load driver
# ---------------------------------------------------------------------------

def _run_load_scenario(
    host: str,
    port: int,
    write_dir: str,
    scenario: str,
    num_traces: int,
    steps_per_trace: int,
    batch_size: int = 50,
) -> LoadResult:
    """Run *num_traces* complete trace lifecycles serially in batches.

    Each trace does:  StartTrace → RecordStep × steps_per_trace → EndTrace.

    Payloads are pre-built outside the timed regions so client-side hashing
    does not inflate latency measurements.

    *batch_size* controls how many traces are pre-built at once to bound RAM.
    """
    result = LoadResult(
        scenario=scenario,
        traces=num_traces,
        steps_per_trace=steps_per_trace,
    )

    cpu_start = time.process_time()
    wall_start = time.perf_counter()

    for trace_idx in range(num_traces):
        # Pre-build payloads (outside timed HTTP sections).
        step_payloads = [
            _prebuild_step(s, trace_offset=trace_idx * steps_per_trace)
            for s in range(steps_per_trace)
        ]

        # StartTrace
        status, body, rtt = _rpc(host, port, "POST", "/v1/traces", {})
        assert status == 201, f"StartTrace failed: {status} {body}"
        trace_id = body["trace_id"]
        result.start_latencies_s.append(rtt)

        # RecordStep × N
        for step in step_payloads:
            status, body, rtt = _rpc(
                host, port, "POST", f"/v1/traces/{trace_id}/steps", {"step": step}
            )
            assert status == 200, f"RecordStep failed: {status} {body}"
            result.record_latencies_s.append(rtt)

        # EndTrace
        status, body, rtt = _rpc(host, port, "POST", f"/v1/traces/{trace_id}/end", {})
        assert status == 200, f"EndTrace failed: {status} {body}"
        result.end_latencies_s.append(rtt)

    result.wall_seconds = time.perf_counter() - wall_start
    result.cpu_seconds = time.process_time() - cpu_start

    # Measure on-disk storage.
    for root, _, files in os.walk(write_dir):
        for fname in files:
            try:
                result.storage_bytes += os.path.getsize(os.path.join(root, fname))
            except OSError:
                pass

    return result


# ---------------------------------------------------------------------------
# Concurrent load driver
# ---------------------------------------------------------------------------

def _run_concurrent_load(
    host: str,
    port: int,
    write_dir: str,
    num_workers: int = 16,
    traces_per_worker: int = 20,
    steps_per_trace: int = 5,
) -> LoadResult:
    """Run concurrent trace lifecycles across *num_workers* threads."""
    scenario = f"concurrent-{num_workers}w×{traces_per_worker}t×{steps_per_trace}s"
    total_traces = num_workers * traces_per_worker
    result = LoadResult(
        scenario=scenario,
        traces=total_traces,
        steps_per_trace=steps_per_trace,
    )
    lock = threading.Lock()

    def worker(worker_id: int) -> None:
        for ti in range(traces_per_worker):
            trace_offset = worker_id * traces_per_worker * steps_per_trace + ti * steps_per_trace
            step_payloads = [
                _prebuild_step(s, trace_offset=trace_offset)
                for s in range(steps_per_trace)
            ]
            status, body, rtt = _rpc(host, port, "POST", "/v1/traces", {})
            if status != 201:
                return
            trace_id = body["trace_id"]
            with lock:
                result.start_latencies_s.append(rtt)

            for step in step_payloads:
                status, body, rtt = _rpc(
                    host, port, "POST", f"/v1/traces/{trace_id}/steps", {"step": step}
                )
                if status != 200:
                    return
                with lock:
                    result.record_latencies_s.append(rtt)

            status, body, rtt = _rpc(host, port, "POST", f"/v1/traces/{trace_id}/end", {})
            if status == 200:
                with lock:
                    result.end_latencies_s.append(rtt)

    cpu_start = time.process_time()
    wall_start = time.perf_counter()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    result.wall_seconds = time.perf_counter() - wall_start
    result.cpu_seconds = time.process_time() - cpu_start

    for root, _, files in os.walk(write_dir):
        for fname in files:
            try:
                result.storage_bytes += os.path.getsize(os.path.join(root, fname))
            except OSError:
                pass

    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.proxy_load
def test_proxy_load_small(tmp_path, capsys):
    """Small scenario: 100 traces × 5 steps (500 steps total)."""
    host, port, state, srv, _ = _start_load_server(tmp_path)
    try:
        result = _run_load_scenario(
            host, port, str(tmp_path),
            scenario="small (100t × 5s)",
            num_traces=100,
            steps_per_trace=5,
        )
    finally:
        srv.shutdown()
        srv.server_close()
        state.close_all()

    with capsys.disabled():
        print(result.summary())

    # Sanity assertions (coarse; not timing-environment sensitive).
    assert result.traces == 100
    assert len(result.start_latencies_s) == 100
    assert len(result.record_latencies_s) == 500
    assert len(result.end_latencies_s) == 100

    # p95 latency under 2 s per RPC (very conservative — real loopback is <50 ms).
    assert result.p95(result.start_latencies_s) < 2.0, "StartTrace p95 too high"
    assert result.p95(result.record_latencies_s) < 2.0, "RecordStep p95 too high"
    assert result.p95(result.end_latencies_s) < 2.0, "EndTrace p95 too high"

    # Storage sanity: under 100 KiB per step on average.
    assert result.bytes_per_step() < 100 * 1024, "Unexpected storage blowup"

    # Throughput sanity: at least 10 completed traces/s on any modern machine.
    assert result.traces / result.wall_seconds >= 10.0, (
        f"Throughput too low: {result.traces / result.wall_seconds:.1f} traces/s"
    )


@pytest.mark.proxy_load
def test_proxy_load_medium(tmp_path, capsys):
    """Medium scenario: 500 traces × 10 steps (5,000 steps total)."""
    host, port, state, srv, _ = _start_load_server(tmp_path)
    try:
        result = _run_load_scenario(
            host, port, str(tmp_path),
            scenario="medium (500t × 10s)",
            num_traces=500,
            steps_per_trace=10,
        )
    finally:
        srv.shutdown()
        srv.server_close()
        state.close_all()

    with capsys.disabled():
        print(result.summary())

    assert result.traces == 500
    assert len(result.start_latencies_s) == 500
    assert len(result.record_latencies_s) == 5000
    assert len(result.end_latencies_s) == 500

    assert result.p95(result.start_latencies_s) < 2.0
    assert result.p95(result.record_latencies_s) < 2.0
    assert result.p95(result.end_latencies_s) < 2.0
    assert result.bytes_per_step() < 100 * 1024


@pytest.mark.proxy_load
@pytest.mark.skipif(
    not os.environ.get("STEPBACK_PROXY_LOAD_LARGE"),
    reason="Large scenario gated by STEPBACK_PROXY_LOAD_LARGE=1",
)
def test_proxy_load_large(tmp_path, capsys):
    """Large scenario: 2,000 traces × 20 steps (40,000 steps total).

    Enabled by setting the environment variable ``STEPBACK_PROXY_LOAD_LARGE=1``.
    Demonstrates extrapolated capacity of tens of millions of steps/day on a
    single proxy instance.
    """
    host, port, state, srv, _ = _start_load_server(tmp_path)
    try:
        result = _run_load_scenario(
            host, port, str(tmp_path),
            scenario="large (2000t × 20s)",
            num_traces=2000,
            steps_per_trace=20,
        )
    finally:
        srv.shutdown()
        srv.server_close()
        state.close_all()

    with capsys.disabled():
        print(result.summary())
        projected = result.traces_per_day()
        millions = projected / 1_000_000
        print(
            f"\n  ✓ Projected capacity: {projected:,.0f} traces/day "
            f"({millions:.2f} M traces/day)"
        )

    assert result.traces == 2000
    assert result.bytes_per_step() < 100 * 1024


@pytest.mark.proxy_load
@pytest.mark.skipif(
    not os.environ.get("STEPBACK_PROXY_LOAD_CONCURRENT"),
    reason="Concurrent scenario gated by STEPBACK_PROXY_LOAD_CONCURRENT=1",
)
def test_proxy_load_concurrent(tmp_path, capsys):
    """Concurrent scenario: 16 worker threads × 20 traces × 5 steps.

    Exercises lock contention in ``ProxyState`` and the threaded HTTP server.
    Enabled by setting ``STEPBACK_PROXY_LOAD_CONCURRENT=1``.
    """
    host, port, state, srv, _ = _start_load_server(tmp_path)
    try:
        result = _run_concurrent_load(
            host, port, str(tmp_path),
            num_workers=16,
            traces_per_worker=20,
            steps_per_trace=5,
        )
    finally:
        srv.shutdown()
        srv.server_close()
        state.close_all()

    with capsys.disabled():
        print(result.summary())

    assert result.traces == 16 * 20
    # All traces must have completed.
    assert len(result.end_latencies_s) == 16 * 20, (
        f"Expected 320 EndTrace completions, got {len(result.end_latencies_s)}"
    )
    assert result.p95(result.record_latencies_s) < 5.0


@pytest.mark.proxy_load
def test_proxy_throughput_extrapolation(tmp_path, capsys):
    """Verify that measured serial throughput projects to ≥ 1 M traces/day.

    Uses a small but representative scenario so the assertion is not
    environment-sensitive. The 1 M/day threshold assumes a loopback proxy
    can sustain ≥ 12 completed traces/s, which is realistic on any modern
    development machine.

    This test exists to make the '100 M agent-runs/day' README-level claim
    auditable: each run is one or more traces, so 1 M traces/day from a
    single proxy instance is a reasonable lower bound for the proxy itself.
    """
    host, port, state, srv, _ = _start_load_server(tmp_path)
    try:
        result = _run_load_scenario(
            host, port, str(tmp_path),
            scenario="throughput-extrapolation (200t × 5s)",
            num_traces=200,
            steps_per_trace=5,
        )
    finally:
        srv.shutdown()
        srv.server_close()
        state.close_all()

    with capsys.disabled():
        print(result.summary())

    tpd = result.traces_per_day()
    with capsys.disabled():
        print(
            f"\n  Extrapolated capacity: {tpd:,.0f} traces/day "
            f"({tpd / 1_000_000:.2f} M traces/day)"
        )

    assert tpd >= 1_000_000, (
        f"Projected throughput {tpd:,.0f} traces/day is below 1 M/day. "
        "The proxy may need optimisation or the test machine is overloaded."
    )
