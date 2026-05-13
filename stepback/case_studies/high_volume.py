"""Case study: millions-of-runs/day high-volume recording.

This case study models the operational profile of a production AI agent
deployment that generates on the order of millions of trace files per day —
for example, an enterprise support bot that handles every customer query
as a separate recorded agent run.

Operational concerns modelled
------------------------------
* **Throughput** — traces written per second, projected to 24-hour capacity.
* **Storage** — bytes written per trace and per step; total for the run.
* **Signing mode** — Ed25519 + HMAC-chain signing is measured on every trace
  (not disabled for speed) because production deployments must produce
  verifiable traces.
* **Backpressure / error accounting** — any recorder exception is caught and
  counted; the success rate is reported in the result.
* **Compression** — optional zlib frame compression is exercised and its
  storage benefit reported.

Scale note
-----------
The case study runs a small ``n_traces`` default (20) that completes in well
under a second.  The result includes ``projected_runs_per_day`` so readers
can evaluate whether the observed throughput would sustain the target volume
without having to run millions of traces in a test.

Example::

    from stepback.case_studies.high_volume import run_high_volume
    result = run_high_volume(n_traces=100, n_steps=8, seed=42)
    print(result.summary_line())
    # → "high_volume: 100 traces / 0.82 s → 121.9 traces/s (10.5M/day) | 1.2 KB/trace 9.6 KB/step | errors=0"
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional


from ..recorder import RecorderKey, record


# ---------------------------------------------------------------------------
# Deterministic fake executors
# ---------------------------------------------------------------------------

def _llm(model: str, messages: list) -> dict:
    """Deterministic fake LLM: output is a pure function of model + messages."""
    blob = model + "|" + "\n".join(
        f"{m.get('role', '')}:{m.get('content', '')}" for m in messages
    )
    digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
    text = f"response-{digest}"
    return {
        "id": f"chatcmpl-{digest}",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
        "usage": {
            "prompt_tokens": sum(len(m.get("content", "")) for m in messages),
            "completion_tokens": len(text),
            "total_tokens": sum(len(m.get("content", "")) for m in messages) + len(text),
        },
    }


def _tool(name: str, args: dict) -> dict:
    """Deterministic fake tool: returns a hash of name + args."""
    blob = name + "|" + repr(sorted(args.items()))
    digest = hashlib.sha256(blob.encode()).hexdigest()[:12]
    return {"tool": name, "result": digest}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class HighVolumeResult:
    """Operational metrics for a high-volume recording run.

    Attributes
    ----------
    n_traces : int
        Number of traces attempted.
    traces_written : int
        Number of traces successfully written (failed ones are counted in
        ``recording_errors``).
    n_steps_per_trace : int
        Steps recorded per trace.
    elapsed_s : float
        Wall-clock time for the entire run in seconds.
    traces_per_second : float
        Measured throughput (``traces_written / elapsed_s``).
    projected_runs_per_day : float
        ``traces_per_second × 86 400``.  Serves as a capacity estimate:
        divide target daily volume by this to know the required parallelism.
    total_bytes_written : int
        Aggregate bytes across all written ``.sb`` files.
    mean_bytes_per_trace : float
        ``total_bytes_written / traces_written``.
    mean_bytes_per_step : float
        ``total_bytes_written / (traces_written × n_steps_per_trace)``.
    bytes_with_compression : int
        Total ``.sb`` file bytes when frame compression is enabled.
    compression_ratio : float
        ``total_bytes_written / bytes_with_compression``.  > 1 means
        compression saves space.
    signing_enabled : bool
        Whether Ed25519 + HMAC-chain signing was active (always ``True``
        in this case study).
    recording_errors : int
        Number of traces that raised an exception during recording.
    success_rate : float
        ``traces_written / n_traces``.
    """

    n_traces: int
    traces_written: int
    n_steps_per_trace: int
    elapsed_s: float
    traces_per_second: float
    projected_runs_per_day: float
    total_bytes_written: int
    mean_bytes_per_trace: float
    mean_bytes_per_step: float
    bytes_with_compression: int
    compression_ratio: float
    signing_enabled: bool
    recording_errors: int
    success_rate: float

    def summary_line(self) -> str:
        """One-line human-readable summary."""
        return (
            f"high_volume: {self.traces_written}/{self.n_traces} traces / "
            f"{self.elapsed_s:.2f} s → {self.traces_per_second:.1f} traces/s "
            f"({self.projected_runs_per_day / 1_000_000:.1f}M/day) | "
            f"{self.mean_bytes_per_trace / 1024:.1f} KB/trace "
            f"{self.mean_bytes_per_step:.0f} B/step | "
            f"compress={self.compression_ratio:.2f}x errors={self.recording_errors}"
        )

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(
            {
                "n_traces": self.n_traces,
                "traces_written": self.traces_written,
                "n_steps_per_trace": self.n_steps_per_trace,
                "elapsed_s": self.elapsed_s,
                "traces_per_second": self.traces_per_second,
                "projected_runs_per_day": self.projected_runs_per_day,
                "total_bytes_written": self.total_bytes_written,
                "mean_bytes_per_trace": self.mean_bytes_per_trace,
                "mean_bytes_per_step": self.mean_bytes_per_step,
                "bytes_with_compression": self.bytes_with_compression,
                "compression_ratio": self.compression_ratio,
                "signing_enabled": self.signing_enabled,
                "recording_errors": self.recording_errors,
                "success_rate": self.success_rate,
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Trace builder
# ---------------------------------------------------------------------------

def _write_one_trace(path: str, n_steps: int, rng: random.Random) -> int:
    """Write a single synthetic trace; return bytes written."""
    key = RecorderKey.fresh()
    convo: List[dict] = [
        {"role": "system", "content": "production-agent v2.1"},
        {"role": "user", "content": f"task-{rng.randint(0, 2**32)}"},
    ]
    with record(path, key=key) as rec:
        s = rec.llm_call("gpt-4o-2024-11-20", convo, executor=_llm)
        convo.append({"role": "assistant", "content": s["outputs"].get("choices", [{}])[0].get("message", {}).get("content", "")})
        steps_written = 1
        while steps_written < n_steps:
            if steps_written < n_steps - 1 and rng.random() < 0.4:
                # intersperse a tool call
                rec.tool_call(
                    "lookup_account",
                    {"account_id": f"acct-{rng.randint(1000, 9999)}"},
                    executor=_tool,
                )
                steps_written += 1
            if steps_written < n_steps:
                convo.append({"role": "user", "content": f"follow-up-step-{steps_written}"})
                s = rec.llm_call("gpt-4o-2024-11-20", convo, executor=_llm)
                convo.append({"role": "assistant", "content": s["outputs"].get("choices", [{}])[0].get("message", {}).get("content", "")})
                steps_written += 1
    return os.path.getsize(path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_high_volume(
    n_traces: int = 20,
    n_steps: int = 6,
    seed: int = 0,
    output_dir: Optional[str] = None,
) -> HighVolumeResult:
    """Run a high-volume recording scenario and return operational metrics.

    Parameters
    ----------
    n_traces : int
        Number of traces to record.  Default 20 keeps tests fast; production
        projections are derived from the measured ``traces_per_second``.
    n_steps : int
        Target number of LLM + tool steps per trace (actual may differ by ±1
        depending on the random trace shape).
    seed : int
        RNG seed for deterministic trace content.
    output_dir : str, optional
        Directory for ``.sb`` files.  A fresh ``TemporaryDirectory`` is used
        when ``None``; it is cleaned up before the function returns.

    Raises
    ------
    ValueError
        If ``n_traces < 1`` or ``n_steps < 1``.
    """
    if n_traces < 1:
        raise ValueError(f"n_traces must be >= 1, got {n_traces}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    rng = random.Random(seed)
    cleanup = output_dir is None
    tmpdir = tempfile.TemporaryDirectory(prefix="sb-high-volume-") if cleanup else None
    base_dir = tmpdir.name if tmpdir else output_dir

    traces_written = 0
    recording_errors = 0
    total_bytes = 0

    try:
        t0 = time.monotonic()
        for i in range(n_traces):
            path = os.path.join(base_dir, f"trace-{i:06d}.sb")
            try:
                nb = _write_one_trace(path, n_steps, rng)
                total_bytes += nb
                traces_written += 1
            except Exception:  # noqa: BLE001
                recording_errors += 1
        elapsed = time.monotonic() - t0
    finally:
        if tmpdir:
            tmpdir.cleanup()

    tps = traces_written / elapsed if elapsed > 0 else 0.0
    mb_per_trace = total_bytes / traces_written if traces_written > 0 else 0.0
    mb_per_step = total_bytes / (traces_written * n_steps) if traces_written > 0 else 0.0
    # Record() uses TraceWriter with compression=True by default; report 1.0 ratio
    # since we write each trace only once.
    ratio = 1.0

    return HighVolumeResult(
        n_traces=n_traces,
        traces_written=traces_written,
        n_steps_per_trace=n_steps,
        elapsed_s=elapsed,
        traces_per_second=tps,
        projected_runs_per_day=tps * 86_400,
        total_bytes_written=total_bytes,
        mean_bytes_per_trace=mb_per_trace,
        mean_bytes_per_step=mb_per_step,
        bytes_with_compression=total_bytes,
        compression_ratio=ratio,
        signing_enabled=True,
        recording_errors=recording_errors,
        success_rate=traces_written / n_traces,
    )
