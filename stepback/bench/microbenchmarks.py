"""Low-level microbenchmarks for core stepback operations (Step 135).

Measures wall-clock latency (p50/p95/p99) and, where relevant, throughput
(MB/s) for six operation categories:

* **Canonicalization** — :func:`~stepback.canonical.canonical_json`,
  :func:`~stepback.canonical.sha256_hex`, and
  :func:`~stepback.canonical.hash_obj` on small (~128 B), medium (~4 KB),
  and large (~64 KB) payloads.

* **HMAC / signing** — isolated :func:`hmac.new` + ``digest()`` (HMAC-SHA256)
  and :meth:`Ed25519PrivateKey.sign` on a 32-byte digest.

* **Frame writing** — :meth:`~stepback.trace_writer.TraceWriter._write_frame`
  for a small step body (~256 B) and a medium step body (~4 KB), measuring
  the full HMAC-chain + Ed25519 sign + length-prefix + disk write path.

* **Recorder hooks** — :func:`~stepback.recorder.record` context-manager
  overhead for a single ``llm_call`` step (the diff against the bare LLM
  call is :attr:`OpResult.overhead_us`).

* **Reader throughput** — :func:`~stepback.trace_reader.read_frames` and
  :func:`~stepback.trace_reader.verify_trace` on pre-built traces with 10
  and 100 steps.

* **Dirty-set planning** — :func:`~stepback.divergence.compute_dirty_set`
  on pre-recorded linear traces with 20 and 100 steps and a single
  :class:`~stepback.substitutions.PromptSubstitution` at position 1
  (worst-case propagation: all downstream steps are dirty).

All benchmarks are self-contained (no network calls, no SDK dependencies).

Run from the CLI::

    stepback bench microbenchmarks [--n-iter N] [--out result.json]
"""
from __future__ import annotations

import hashlib
import hmac as _hmac
import os
import statistics
import struct
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ..canonical import canonical_json, hash_obj, sha256_hex
from ..recorder import RecorderKey, record
from ..trace_reader import read_frames, verify_trace
from ..trace_writer import TraceWriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(samples: List[float], q: float) -> float:
    """Return the q-th percentile of *samples* (0 ≤ q ≤ 1)."""
    if not samples:
        return 0.0
    s = sorted(samples)
    idx = min(len(s) - 1, max(0, int(q * len(s))))
    return s[idx]


def _time_fn(fn, n_iter: int, n_warmup: int = 20) -> List[float]:
    """Return a list of *n_iter* per-call timings in microseconds."""
    for _ in range(n_warmup):
        fn()
    samples: List[float] = []
    for _ in range(n_iter):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1_000.0)
    return samples


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OpResult:
    """Timing result for a single micro-operation.

    Attributes:
        name:            Human-readable identifier.
        n_iter:          Number of timed iterations.
        payload_bytes:   Input or output payload size in bytes (0 if N/A).
        p50_us:          Median latency in microseconds.
        p95_us:          95th-percentile latency in microseconds.
        p99_us:          99th-percentile latency in microseconds.
        mean_us:         Mean latency in microseconds.
        throughput_mbs:  Throughput in MB/s (payload_bytes / mean_us * 1e6 / 1e6),
                         or 0.0 when payload_bytes is 0.
        overhead_us:     Optional overhead relative to a baseline (e.g. recorder
                         delta). 0.0 when not applicable.
    """
    name: str
    n_iter: int
    payload_bytes: int
    p50_us: float
    p95_us: float
    p99_us: float
    mean_us: float
    throughput_mbs: float
    overhead_us: float = 0.0

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "n_iter": self.n_iter,
            "payload_bytes": self.payload_bytes,
            "p50_us": self.p50_us,
            "p95_us": self.p95_us,
            "p99_us": self.p99_us,
            "mean_us": self.mean_us,
            "throughput_mbs": self.throughput_mbs,
            "overhead_us": self.overhead_us,
        }

    def summary_line(self) -> str:
        tput = f" tput={self.throughput_mbs:.1f}MB/s" if self.throughput_mbs > 0 else ""
        overhead = f" overhead_us={self.overhead_us:.1f}" if self.overhead_us != 0.0 else ""
        return (
            f"microbench op={self.name} n={self.n_iter}"
            f" p50={self.p50_us:.2f}µs p95={self.p95_us:.2f}µs"
            f" p99={self.p99_us:.2f}µs mean={self.mean_us:.2f}µs"
            f"{tput}{overhead}"
        )


def _make_op(name: str, samples: List[float], payload_bytes: int = 0, overhead_us: float = 0.0) -> OpResult:
    mean = statistics.mean(samples) if samples else 0.0
    tput = (payload_bytes / mean / 1_000.0) if (mean > 0 and payload_bytes > 0) else 0.0
    return OpResult(
        name=name,
        n_iter=len(samples),
        payload_bytes=payload_bytes,
        p50_us=_percentile(samples, 0.50),
        p95_us=_percentile(samples, 0.95),
        p99_us=_percentile(samples, 0.99),
        mean_us=mean,
        throughput_mbs=tput,
        overhead_us=overhead_us,
    )


@dataclass
class MicroBenchSuite:
    """Full microbenchmark suite result.

    Attributes:
        n_iter:  Number of timed iterations per operation.
        ops:     Dict mapping operation name → :class:`OpResult`.
    """
    n_iter: int
    ops: Dict[str, OpResult] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "n_iter": self.n_iter,
            "ops": {k: v.to_json() for k, v in self.ops.items()},
        }

    def summary_lines(self) -> List[str]:
        return [op.summary_line() for op in self.ops.values()]


# ---------------------------------------------------------------------------
# Payload generators (deterministic, no network)
# ---------------------------------------------------------------------------

def _make_dict_small() -> dict:
    return {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.7,
    }


def _make_dict_medium() -> dict:
    return {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. " * 20},
            {"role": "user", "content": "Please summarize the following. " * 30},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
        "extra": {str(i): i * 1.5 for i in range(50)},
    }


def _make_dict_large() -> dict:
    base = _make_dict_medium()
    base["large_context"] = [
        {"index": i, "text": f"sentence number {i} with some content. " * 5}
        for i in range(200)
    ]
    return base


def _make_step_body_small() -> dict:
    return {
        "type": "step",
        "step_id": "step-0001",
        "step_kind": "llm_call",
        "name": "gpt-4o",
        "inputs": {"messages": [{"role": "user", "content": "hello"}]},
        "outputs": {"choices": [{"message": {"role": "assistant", "content": "Hi!"}}]},
        "inputs_hash": "sha256:abcdef1234567890",
        "outputs_hash": "sha256:fedcba0987654321",
        "wallclock_ns": 1_700_000_000_000_000_000,
        "cost_usd": 0.0001,
    }


def _make_step_body_medium() -> dict:
    body = _make_step_body_small()
    body["inputs"] = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. " * 20},
            {"role": "user", "content": "Explain this concept. " * 30},
        ]
    }
    body["outputs"] = {
        "choices": [
            {"message": {"role": "assistant", "content": "Sure, here is the explanation. " * 40}}
        ]
    }
    return body


# ---------------------------------------------------------------------------
# Individual benchmark sections
# ---------------------------------------------------------------------------

def _bench_canonicalization(n_iter: int) -> Dict[str, OpResult]:
    ops: Dict[str, OpResult] = {}

    small = _make_dict_small()
    medium = _make_dict_medium()
    large = _make_dict_large()

    small_bytes = len(canonical_json(small))
    medium_bytes = len(canonical_json(medium))
    large_bytes = len(canonical_json(large))

    ops["canonical_json_small"] = _make_op(
        "canonical_json_small",
        _time_fn(lambda: canonical_json(small), n_iter),
        payload_bytes=small_bytes,
    )
    ops["canonical_json_medium"] = _make_op(
        "canonical_json_medium",
        _time_fn(lambda: canonical_json(medium), n_iter),
        payload_bytes=medium_bytes,
    )
    ops["canonical_json_large"] = _make_op(
        "canonical_json_large",
        _time_fn(lambda: canonical_json(large), n_iter),
        payload_bytes=large_bytes,
    )

    raw_small = os.urandom(128)
    raw_medium = os.urandom(4096)
    raw_large = os.urandom(65536)

    ops["sha256_small"] = _make_op(
        "sha256_small",
        _time_fn(lambda: sha256_hex(raw_small), n_iter),
        payload_bytes=len(raw_small),
    )
    ops["sha256_medium"] = _make_op(
        "sha256_medium",
        _time_fn(lambda: sha256_hex(raw_medium), n_iter),
        payload_bytes=len(raw_medium),
    )
    ops["sha256_large"] = _make_op(
        "sha256_large",
        _time_fn(lambda: sha256_hex(raw_large), n_iter),
        payload_bytes=len(raw_large),
    )

    ops["hash_obj_medium"] = _make_op(
        "hash_obj_medium",
        _time_fn(lambda: hash_obj(medium), n_iter),
        payload_bytes=medium_bytes,
    )

    return ops


def _bench_hmac_signing(n_iter: int) -> Dict[str, OpResult]:
    ops: Dict[str, OpResult] = {}

    hmac_key = os.urandom(32)
    prev_hmac = os.urandom(32)
    body_bytes = canonical_json(_make_step_body_small())
    signing_key = Ed25519PrivateKey.generate()

    ops["hmac_sha256"] = _make_op(
        "hmac_sha256",
        _time_fn(
            lambda: _hmac.new(hmac_key, prev_hmac + body_bytes, hashlib.sha256).digest(),
            n_iter,
        ),
    )

    digest = _hmac.new(hmac_key, prev_hmac + body_bytes, hashlib.sha256).digest()
    ops["ed25519_sign"] = _make_op(
        "ed25519_sign",
        _time_fn(lambda: signing_key.sign(digest), n_iter),
    )

    def _hmac_and_sign() -> None:
        h = _hmac.new(hmac_key, prev_hmac + body_bytes, hashlib.sha256).digest()
        signing_key.sign(h)

    ops["hmac_and_sign"] = _make_op(
        "hmac_and_sign",
        _time_fn(_hmac_and_sign, n_iter),
    )

    return ops


def _bench_frame_writing(n_iter: int) -> Dict[str, OpResult]:
    ops: Dict[str, OpResult] = {}

    small_body = _make_step_body_small()
    medium_body = _make_step_body_medium()

    for label, body in [("frame_write_small", small_body), ("frame_write_medium", medium_body)]:
        payload_bytes = len(canonical_json(body))
        with tempfile.TemporaryDirectory(prefix="stepback-ub-fw-") as d:
            path = os.path.join(d, "bench.sb")
            w = TraceWriter.open(path, emit_merkle_summary=False)

            def _write(w=w, body=body):
                w._write_frame(body)  # type: ignore[attr-defined]

            samples = _time_fn(_write, n_iter, n_warmup=10)
            w.f.close()  # type: ignore[union-attr]

        ops[label] = _make_op(label, samples, payload_bytes=payload_bytes)

    return ops


def _bench_recorder_hooks(n_iter: int) -> Dict[str, OpResult]:
    """Benchmark the record() context-manager overhead for one llm_call step."""
    from .replay_caching import _bench_llm

    convo = [
        {"role": "system", "content": "you are a bench agent"},
        {"role": "user", "content": "please respond"},
    ]

    # Baseline: bare LLM call.
    for _ in range(20):
        _bench_llm("gpt-4o-2024-11-20", convo)
    baseline_us: List[float] = []
    for _ in range(n_iter):
        t0 = time.perf_counter_ns()
        _bench_llm("gpt-4o-2024-11-20", convo)
        baseline_us.append((time.perf_counter_ns() - t0) / 1_000.0)

    # Recorder path.
    with tempfile.TemporaryDirectory(prefix="stepback-ub-rh-warmup-") as wd:
        wp = os.path.join(wd, "warmup.sb")
        with record(wp, key=RecorderKey.fresh()) as warm_rec:
            for _ in range(20):
                warm_rec.llm_call("gpt-4o-2024-11-20", convo, executor=_bench_llm)

    recorded_us: List[float] = []
    with tempfile.TemporaryDirectory(prefix="stepback-ub-rh-") as d:
        path = os.path.join(d, "bench.sb")
        with record(path, key=RecorderKey.fresh()) as rec:
            for _ in range(n_iter):
                t0 = time.perf_counter_ns()
                rec.llm_call("gpt-4o-2024-11-20", convo, executor=_bench_llm)
                recorded_us.append((time.perf_counter_ns() - t0) / 1_000.0)

    baseline_p50 = _percentile(baseline_us, 0.50)
    recorded_p50 = _percentile(recorded_us, 0.50)
    overhead = recorded_p50 - baseline_p50

    ops: Dict[str, OpResult] = {}
    ops["recorder_hook_llm"] = _make_op(
        "recorder_hook_llm",
        recorded_us,
        overhead_us=overhead,
    )
    ops["recorder_hook_baseline"] = _make_op(
        "recorder_hook_baseline",
        baseline_us,
    )
    return ops


def _build_trace_file(n_steps: int, tmp_dir: str) -> tuple:
    """Record a synthetic n-step trace; return ``(path, hmac_key)``."""
    from .replay_caching import _bench_llm, _bench_tool

    path = os.path.join(tmp_dir, f"trace_{n_steps}.sb")
    convo_base = [{"role": "user", "content": "step input"}]
    key = RecorderKey.fresh()
    with record(path, key=key) as rec:
        for i in range(n_steps):
            convo = convo_base + [{"role": "assistant", "content": f"prev-{i}"}]
            if i % 3 == 0:
                rec.tool_call(f"tool_{i % 4}", {"arg": i}, executor=lambda n, a: _bench_tool(n, a))
            else:
                rec.llm_call("gpt-4o-2024-11-20", convo, executor=_bench_llm)
    return path, key.hmac_key


def _bench_reader_throughput(n_iter: int) -> Dict[str, OpResult]:
    ops: Dict[str, OpResult] = {}
    with tempfile.TemporaryDirectory(prefix="stepback-ub-rd-") as d:
        for n_steps in (10, 100):
            path, hmac_key = _build_trace_file(n_steps, d)
            trace_bytes = os.path.getsize(path)

            ops[f"read_frames_{n_steps}"] = _make_op(
                f"read_frames_{n_steps}",
                _time_fn(lambda p=path: read_frames(p), n_iter),
                payload_bytes=trace_bytes,
            )
            ops[f"verify_trace_{n_steps}"] = _make_op(
                f"verify_trace_{n_steps}",
                _time_fn(lambda p=path, k=hmac_key: verify_trace(p, k), n_iter),
                payload_bytes=trace_bytes,
            )
    return ops


def _bench_dirty_set_planning(n_iter: int) -> Dict[str, OpResult]:
    from ..divergence import compute_dirty_set
    from ..replay import replay
    from ..substitutions import PromptSubstitution

    ops: Dict[str, OpResult] = {}

    with tempfile.TemporaryDirectory(prefix="stepback-ub-ds-") as d:
        for n_steps in (20, 100):
            path, _hmac_key = _build_trace_file(n_steps, d)
            trace = replay(path)
            # Substitute at step 1 → all downstream steps dirty (worst case).
            step_id = trace.recorded_steps[1]["step_id"]
            subs = [PromptSubstitution(
                at_step=step_id,
                new_messages=[{"role": "user", "content": "bench-changed"}],
            )]

            ops[f"dirty_set_plan_{n_steps}"] = _make_op(
                f"dirty_set_plan_{n_steps}",
                _time_fn(lambda t=trace, s=subs: compute_dirty_set(t, s), n_iter),
            )

    return ops


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(n_iter: int = 500) -> MicroBenchSuite:
    """Run the full microbenchmark suite and return a :class:`MicroBenchSuite`.

    Args:
        n_iter: Number of timed iterations per operation (after warm-up).
            Lower values run faster; higher values reduce measurement noise.
            Default of 500 gives stable p50/p95/p99 estimates in < 30 s on
            typical developer hardware.
    """
    if n_iter < 10:
        raise ValueError("n_iter must be >= 10")

    suite = MicroBenchSuite(n_iter=n_iter)

    suite.ops.update(_bench_canonicalization(n_iter))
    suite.ops.update(_bench_hmac_signing(n_iter))
    suite.ops.update(_bench_frame_writing(n_iter))
    suite.ops.update(_bench_recorder_hooks(n_iter))
    suite.ops.update(_bench_reader_throughput(n_iter))
    suite.ops.update(_bench_dirty_set_planning(n_iter))

    return suite
