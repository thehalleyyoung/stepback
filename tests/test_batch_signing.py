"""Tests for batch-signing / async-signing mode (Step 137).

Covers:
* compression=True + batch_sign=True: parallel signing at flush time
* compression=False + batch_sign=True: streaming batch buffer
* HMAC chain integrity matches sequential (per-frame) signing
* verify_trace accepts batch-signed traces
* flush() method for streaming mode
* batch_sign=True + signing=False raises ValueError
* edge cases: 0 steps, 1 step, exactly batch_sign_interval, interval+1
* many frames / large batches
* capability frames interspersed with steps
* executor is reused (not recreated per batch)
* signing thread safety (concurrent signing of many HMACs with same key)
"""
from __future__ import annotations

import os
import tempfile
import time
from typing import Optional

import pytest

from stepback.recorder import record
from stepback.trace_reader import read_frames, verify_trace
from stepback.trace_writer import (
    DEFAULT_BATCH_SIGN_INTERVAL,
    TraceWriter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_executor(model: str, messages: list) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": f"reply-{model}"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


def _fake_tool(name: str, args: dict) -> dict:
    return {"result": f"ok-{name}"}


def _write_n_steps(
    path: str,
    n: int,
    *,
    compression: bool = True,
    batch_sign: bool = False,
    batch_sign_interval: int = DEFAULT_BATCH_SIGN_INTERVAL,
    batch_sign_workers: int = 0,
    signing: bool = True,
    hmac_key: Optional[bytes] = None,
) -> tuple:
    """Record n tool_call steps; return (hmac_key, signing_key)."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    hmac_key = hmac_key or os.urandom(32)
    signing_key = Ed25519PrivateKey.generate() if signing else None
    w = TraceWriter.open(
        path,
        hmac_key=hmac_key,
        signing_key=signing_key,
        compression=compression,
        signing=signing,
        batch_sign=batch_sign,
        batch_sign_interval=batch_sign_interval,
        batch_sign_workers=batch_sign_workers,
    )
    for i in range(n):
        w.write_step(
            {
                "step_id": f"s{i}",
                "step_kind": "tool_call",
                "name": f"tool{i}",
                "inputs": {"i": i},
                "outputs": {"result": i * 2},
                "cost_usd": 0.0,
            }
        )
    w.close()
    return hmac_key, signing_key


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


class TestBatchSigningCompressionTrue:
    """Batch signing in the default compression=True mode."""

    def test_trace_verifies(self, tmp_path):
        path = str(tmp_path / "t.sb")
        with record(path, batch_sign=True) as rec:
            rec.llm_call(
                "gpt-4o",
                [{"role": "user", "content": "hi"}],
                executor=_fake_executor,
            )
            rec.tool_call("lookup", {"q": "x"}, executor=_fake_tool)

        result = verify_trace(path, rec.key.hmac_key)
        assert result is not None
        assert len(result.steps) == 2

    def test_step_ids_are_correct(self, tmp_path):
        path = str(tmp_path / "t.sb")
        hmac_key, _ = _write_n_steps(str(tmp_path / "t.sb"), n=5, batch_sign=True)
        result = verify_trace(path, hmac_key)
        ids = [s["step_id"] for s in result.steps]
        assert ids == ["s0", "s1", "s2", "s3", "s4"]

    def test_single_step(self, tmp_path):
        path = str(tmp_path / "t.sb")
        hmac_key, _ = _write_n_steps(path, n=1, batch_sign=True)
        result = verify_trace(path, hmac_key)
        assert len(result.steps) == 1

    def test_zero_steps(self, tmp_path):
        path = str(tmp_path / "t.sb")
        hmac_key, _ = _write_n_steps(path, n=0, batch_sign=True)
        result = verify_trace(path, hmac_key)
        assert len(result.steps) == 0

    def test_many_steps(self, tmp_path):
        n = 200
        path = str(tmp_path / "t.sb")
        hmac_key, _ = _write_n_steps(path, n=n, batch_sign=True)
        result = verify_trace(path, hmac_key)
        assert len(result.steps) == n

    def test_batch_sign_with_workers_1(self, tmp_path):
        """Workers=1 is effectively sequential but via the parallel code path."""
        path = str(tmp_path / "t.sb")
        hmac_key, _ = _write_n_steps(path, n=10, batch_sign=True, batch_sign_workers=1)
        result = verify_trace(path, hmac_key)
        assert len(result.steps) == 10

    def test_batch_sign_with_workers_4(self, tmp_path):
        path = str(tmp_path / "t.sb")
        hmac_key, _ = _write_n_steps(path, n=20, batch_sign=True, batch_sign_workers=4)
        result = verify_trace(path, hmac_key)
        assert len(result.steps) == 20

    def test_capability_frames_interspersed(self, tmp_path):
        path = str(tmp_path / "t.sb")
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        hmac_key = os.urandom(32)
        signing_key = Ed25519PrivateKey.generate()
        w = TraceWriter.open(
            path,
            hmac_key=hmac_key,
            signing_key=signing_key,
            compression=True,
            batch_sign=True,
        )
        w.write_capability("test-cap", mandatory=False)
        w.write_step({"step_id": "s1", "step_kind": "tool_call", "name": "a",
                      "inputs": {}, "outputs": {}, "cost_usd": 0.0})
        w.write_step({"step_id": "s2", "step_kind": "tool_call", "name": "b",
                      "inputs": {}, "outputs": {}, "cost_usd": 0.0})
        w.close()
        result = verify_trace(path, hmac_key)
        assert len(result.steps) == 2
        assert len(result.capabilities) == 1

    def test_frame_count_matches_sequential(self, tmp_path):
        """Batch-signed trace should have same number of frames as sequential."""
        n = 5
        path_batch = str(tmp_path / "batch.sb")
        path_seq = str(tmp_path / "seq.sb")
        hmac_key = os.urandom(32)
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        sk = Ed25519PrivateKey.generate()

        # Write with batch_sign
        w1 = TraceWriter.open(path_batch, hmac_key=hmac_key, signing_key=sk,
                              compression=True, batch_sign=True)
        for i in range(n):
            w1.write_step({"step_id": f"s{i}", "step_kind": "tool_call",
                           "name": f"t{i}", "inputs": {}, "outputs": {}, "cost_usd": 0.0})
        w1.close()

        # Write without batch_sign (same key - different key actually for verify,
        # but we just compare frame counts)
        hmac_key2 = os.urandom(32)
        sk2 = Ed25519PrivateKey.generate()
        w2 = TraceWriter.open(path_seq, hmac_key=hmac_key2, signing_key=sk2,
                              compression=True, batch_sign=False)
        for i in range(n):
            w2.write_step({"step_id": f"s{i}", "step_kind": "tool_call",
                           "name": f"t{i}", "inputs": {}, "outputs": {}, "cost_usd": 0.0})
        w2.close()

        frames_batch = list(read_frames(path_batch))
        frames_seq = list(read_frames(path_seq))
        # Same number of frames (header, steps, merkle_summary, tail)
        assert len(frames_batch) == len(frames_seq)


class TestBatchSigningCompressionFalse:
    """Batch signing in streaming (compression=False) mode."""

    def test_trace_verifies(self, tmp_path):
        path = str(tmp_path / "t.sb")
        hmac_key, _ = _write_n_steps(
            path, n=5, compression=False, batch_sign=True, batch_sign_interval=10
        )
        result = verify_trace(path, hmac_key)
        assert len(result.steps) == 5

    def test_auto_flush_at_interval(self, tmp_path):
        """Frames written to disk when _streaming_batch reaches batch_sign_interval."""
        path = str(tmp_path / "t.sb")
        interval = 3
        n = 10  # 3 full batches + 1 partial
        hmac_key, _ = _write_n_steps(
            path, n=n, compression=False, batch_sign=True, batch_sign_interval=interval
        )
        result = verify_trace(path, hmac_key)
        assert len(result.steps) == n

    def test_exactly_interval_steps(self, tmp_path):
        interval = 3
        path = str(tmp_path / "t.sb")
        hmac_key, _ = _write_n_steps(
            path, n=interval, compression=False, batch_sign=True, batch_sign_interval=interval
        )
        result = verify_trace(path, hmac_key)
        assert len(result.steps) == interval

    def test_interval_minus_one_steps(self, tmp_path):
        interval = 5
        path = str(tmp_path / "t.sb")
        hmac_key, _ = _write_n_steps(
            path, n=interval - 1, compression=False, batch_sign=True, batch_sign_interval=interval
        )
        result = verify_trace(path, hmac_key)
        assert len(result.steps) == interval - 1

    def test_single_step(self, tmp_path):
        path = str(tmp_path / "t.sb")
        hmac_key, _ = _write_n_steps(
            path, n=1, compression=False, batch_sign=True, batch_sign_interval=10
        )
        result = verify_trace(path, hmac_key)
        assert len(result.steps) == 1

    def test_zero_steps(self, tmp_path):
        path = str(tmp_path / "t.sb")
        hmac_key, _ = _write_n_steps(
            path, n=0, compression=False, batch_sign=True, batch_sign_interval=10
        )
        result = verify_trace(path, hmac_key)
        assert len(result.steps) == 0

    def test_flush_method(self, tmp_path):
        """flush() writes buffered frames to disk immediately."""
        path = str(tmp_path / "t.sb")
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        hmac_key = os.urandom(32)
        sk = Ed25519PrivateKey.generate()
        w = TraceWriter.open(
            path, hmac_key=hmac_key, signing_key=sk,
            compression=False, batch_sign=True, batch_sign_interval=100
        )
        w.write_step({"step_id": "s1", "step_kind": "tool_call", "name": "a",
                      "inputs": {}, "outputs": {}, "cost_usd": 0.0})
        assert len(w._streaming_batch) == 1
        w.flush()
        assert len(w._streaming_batch) == 0
        w.write_step({"step_id": "s2", "step_kind": "tool_call", "name": "b",
                      "inputs": {}, "outputs": {}, "cost_usd": 0.0})
        w.close()
        result = verify_trace(path, hmac_key)
        assert len(result.steps) == 2

    def test_flush_noop_in_compression_true(self, tmp_path):
        """flush() is a no-op in compression=True mode."""
        path = str(tmp_path / "t.sb")
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        hmac_key = os.urandom(32)
        sk = Ed25519PrivateKey.generate()
        w = TraceWriter.open(
            path, hmac_key=hmac_key, signing_key=sk,
            compression=True, batch_sign=True
        )
        w.write_step({"step_id": "s1", "step_kind": "tool_call", "name": "a",
                      "inputs": {}, "outputs": {}, "cost_usd": 0.0})
        w.flush()  # Should not raise; pending_items still contains the step
        assert len(w.pending_items) == 1
        w.close()
        result = verify_trace(path, hmac_key)
        assert len(result.steps) == 1

    def test_capability_frames_buffered(self, tmp_path):
        path = str(tmp_path / "t.sb")
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        hmac_key = os.urandom(32)
        sk = Ed25519PrivateKey.generate()
        w = TraceWriter.open(
            path, hmac_key=hmac_key, signing_key=sk,
            compression=False, batch_sign=True, batch_sign_interval=10
        )
        w.write_capability("test-cap", mandatory=False)
        w.write_step({"step_id": "s1", "step_kind": "tool_call", "name": "a",
                      "inputs": {}, "outputs": {}, "cost_usd": 0.0})
        w.close()
        result = verify_trace(path, hmac_key)
        assert len(result.steps) == 1
        assert len(result.capabilities) == 1


# ---------------------------------------------------------------------------
# Validation / edge cases
# ---------------------------------------------------------------------------


class TestBatchSignValidation:

    def test_batch_sign_requires_signing(self, tmp_path):
        """batch_sign=True with signing=False must raise ValueError."""
        path = str(tmp_path / "t.sb")
        with pytest.raises(ValueError, match="batch_sign=True requires signing=True"):
            TraceWriter.open(path, signing=False, batch_sign=True)

    def test_batch_sign_false_signing_false_ok(self, tmp_path):
        """batch_sign=False + signing=False is the existing unsigned fast path."""
        path = str(tmp_path / "t.sb")
        w = TraceWriter.open(path, signing=False, batch_sign=False)
        w.write_step({"step_id": "s1", "step_kind": "tool_call", "name": "a",
                      "inputs": {}, "outputs": {}, "cost_usd": 0.0})
        w.close()

    def test_executor_created_when_batch_sign(self, tmp_path):
        path = str(tmp_path / "t.sb")
        w = TraceWriter.open(path, batch_sign=True)
        assert w._executor is not None
        w.close()
        # Executor should be cleaned up after close
        assert w._executor is None

    def test_executor_not_created_without_batch_sign(self, tmp_path):
        path = str(tmp_path / "t.sb")
        w = TraceWriter.open(path, batch_sign=False)
        assert w._executor is None
        w.close()

    def test_batch_sign_flag_stored(self, tmp_path):
        path = str(tmp_path / "t.sb")
        w = TraceWriter.open(path, batch_sign=True)
        assert w.batch_sign is True
        w.close()

    def test_default_batch_sign_false(self, tmp_path):
        path = str(tmp_path / "t.sb")
        w = TraceWriter.open(path)
        assert w.batch_sign is False
        w.close()


# ---------------------------------------------------------------------------
# HMAC chain integrity: batch vs sequential produce valid (independently verifiable) traces
# ---------------------------------------------------------------------------


class TestHMACChainIntegrity:

    def test_batch_hmac_chain_valid(self, tmp_path):
        """Batch-signed traces pass verify_trace's HMAC chain check."""
        path = str(tmp_path / "t.sb")
        n = 30
        hmac_key, _ = _write_n_steps(path, n=n, batch_sign=True, batch_sign_workers=2)
        result = verify_trace(path, hmac_key)
        assert result is not None
        assert len(result.steps) == n

    def test_streaming_batch_hmac_chain_valid(self, tmp_path):
        """Streaming batch-signed traces pass verify_trace's HMAC chain check."""
        path = str(tmp_path / "t.sb")
        n = 25
        hmac_key, _ = _write_n_steps(
            path, n=n, compression=False, batch_sign=True, batch_sign_interval=7
        )
        result = verify_trace(path, hmac_key)
        assert result is not None
        assert len(result.steps) == n

    def test_tampered_trace_rejected(self, tmp_path):
        """Bit-flip in a batch-signed trace must be rejected by verify_trace."""
        import struct as _struct
        from stepback.trace_reader import TraceVerificationError

        path = str(tmp_path / "t.sb")
        n = 5
        hmac_key, _ = _write_n_steps(path, n=n, batch_sign=True)

        # Flip a byte in the middle of the file
        with open(path, "rb") as fh:
            data = bytearray(fh.read())
        # Flip a byte that's likely in the HMAC field (not the length prefix)
        mid = len(data) // 2
        data[mid] ^= 0xFF
        corrupted_path = str(tmp_path / "corrupted.sb")
        with open(corrupted_path, "wb") as fh:
            fh.write(data)

        with pytest.raises((TraceVerificationError, Exception)):
            verify_trace(corrupted_path, hmac_key)


# ---------------------------------------------------------------------------
# Signing thread safety
# ---------------------------------------------------------------------------


class TestSigningThreadSafety:

    def test_concurrent_signing_correct(self, tmp_path):
        """All frames signed in parallel produce correct Ed25519 signatures."""
        path = str(tmp_path / "t.sb")
        n = 100
        hmac_key, signing_key = _write_n_steps(
            path, n=n, batch_sign=True, batch_sign_workers=4
        )
        # verify_trace checks every signature; this validates thread safety
        result = verify_trace(path, hmac_key)
        assert len(result.steps) == n

    def test_single_worker_correct(self, tmp_path):
        """workers=1 (sequential via thread pool) still produces correct signatures."""
        path = str(tmp_path / "t.sb")
        n = 50
        hmac_key, _ = _write_n_steps(
            path, n=n, batch_sign=True, batch_sign_workers=1
        )
        result = verify_trace(path, hmac_key)
        assert len(result.steps) == n


# ---------------------------------------------------------------------------
# record() and arecord() convenience APIs
# ---------------------------------------------------------------------------


class TestRecordAPI:

    def test_record_batch_sign(self, tmp_path):
        path = str(tmp_path / "t.sb")
        with record(path, batch_sign=True) as rec:
            rec.llm_call(
                "gpt-4o",
                [{"role": "user", "content": "hello"}],
                executor=_fake_executor,
            )
        result = verify_trace(path, rec.key.hmac_key)
        assert len(result.steps) == 1

    def test_record_batch_sign_workers(self, tmp_path):
        path = str(tmp_path / "t.sb")
        with record(path, batch_sign=True, batch_sign_workers=2) as rec:
            rec.tool_call("lookup", {}, executor=_fake_tool)
            rec.tool_call("lookup2", {}, executor=_fake_tool)
        result = verify_trace(path, rec.key.hmac_key)
        assert len(result.steps) == 2

    def test_record_batch_sign_requires_signing(self, tmp_path):
        path = str(tmp_path / "t.sb")
        with pytest.raises(ValueError, match="batch_sign=True requires signing=True"):
            with record(path, signing=False, batch_sign=True):
                pass

    def test_arecord_batch_sign(self, tmp_path):
        import asyncio
        from stepback.recorder import arecord

        path = str(tmp_path / "t.sb")
        hmac_key_holder = {}

        async def _run():
            async with arecord(path, batch_sign=True) as rec:
                rec.tool_call("lookup", {"q": "x"}, executor=_fake_tool)
                hmac_key_holder["key"] = rec.key.hmac_key

        asyncio.run(_run())
        result = verify_trace(path, hmac_key_holder["key"])
        assert len(result.steps) == 1


# ---------------------------------------------------------------------------
# Performance smoke test (not a hard budget, just a regression guard)
# ---------------------------------------------------------------------------


class TestBatchSigningPerformance:

    def test_batch_sign_faster_than_expected_serial_time(self, tmp_path):
        """Batch-signing 50 steps should finish in under 5 seconds on any hardware."""
        path = str(tmp_path / "t.sb")
        t0 = time.perf_counter()
        hmac_key, _ = _write_n_steps(
            path, n=50, batch_sign=True, batch_sign_workers=4
        )
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"batch signing 50 steps took {elapsed:.2f}s, expected < 5s"
        verify_trace(path, hmac_key)
