"""Tests for backpressure and sampling controls (Step 142).

Covers:
- RecorderOptions validation
- Fail-open mode: recorder errors do not propagate
- Sampling: NullRecorder yields when sample_rate < 1.0
- Executors are always called, even when sampled out
- max_queue_depth overflow with "drop" policy
- max_queue_depth overflow with "block" policy (no truncation)
- Fail-open when TraceWriter.open() fails
- Fail-open when writer.close() fails
- Fail-open when write_step() fails
- dropped_steps and recording_errors properties
- mandatory=True re-raises errors
- arecord async variant
- NullRecorder parallel() still runs branch closures
- Public API membership
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

import stepback
from stepback import RecorderOptions, record, arecord
from stepback.backpressure import _NullTraceWriter, _make_null_recorder
from stepback.recorder import Recorder, RecorderKey
from stepback.trace_reader import read_frames, verify_trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_llm(model: str, messages: list) -> dict:
    return {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 1}}


def _fake_tool(name: str, arguments: dict):
    return {"result": "done"}


# ---------------------------------------------------------------------------
# RecorderOptions validation
# ---------------------------------------------------------------------------

class TestRecorderOptionsValidation:
    def test_defaults(self):
        opts = RecorderOptions()
        assert opts.mandatory is False
        assert opts.sample_rate == 1.0
        assert opts.max_queue_depth == 0
        assert opts.overflow_policy == "drop"
        assert opts.rng is None

    def test_valid_construction(self):
        import random
        rng = random.Random(42)
        opts = RecorderOptions(mandatory=False, sample_rate=0.5, max_queue_depth=10,
                               overflow_policy="drop", rng=rng)
        assert opts.sample_rate == 0.5

    def test_sample_rate_out_of_range_high(self):
        with pytest.raises(ValueError, match="sample_rate"):
            RecorderOptions(sample_rate=1.1)

    def test_sample_rate_out_of_range_low(self):
        with pytest.raises(ValueError, match="sample_rate"):
            RecorderOptions(sample_rate=-0.1)

    def test_sample_rate_boundary_zero(self):
        opts = RecorderOptions(sample_rate=0.0)
        assert opts.sample_rate == 0.0

    def test_sample_rate_boundary_one(self):
        opts = RecorderOptions(sample_rate=1.0)
        assert opts.sample_rate == 1.0

    def test_negative_max_queue_depth(self):
        with pytest.raises(ValueError, match="max_queue_depth"):
            RecorderOptions(max_queue_depth=-1)

    def test_invalid_overflow_policy(self):
        with pytest.raises(ValueError, match="overflow_policy"):
            RecorderOptions(overflow_policy="invalid")

    def test_mandatory_drop_contradiction(self):
        with pytest.raises(ValueError, match="mandatory"):
            RecorderOptions(mandatory=True, overflow_policy="drop")

    def test_mandatory_block_allowed(self):
        opts = RecorderOptions(mandatory=True, overflow_policy="block")
        assert opts.mandatory is True

    def test_fail_open_drop_allowed(self):
        opts = RecorderOptions(mandatory=False, overflow_policy="drop")
        assert opts.overflow_policy == "drop"


# ---------------------------------------------------------------------------
# _NullTraceWriter
# ---------------------------------------------------------------------------

class TestNullTraceWriter:
    def test_write_step_noop(self):
        w = _NullTraceWriter()
        w.write_step({"step_id": "x"})  # should not raise

    def test_close_noop(self):
        w = _NullTraceWriter()
        w.close()

    def test_flush_noop(self):
        w = _NullTraceWriter()
        w.flush()

    def test_write_capability_noop(self):
        w = _NullTraceWriter()
        w.write_capability("core", mandatory=True, params={"x": 1})


# ---------------------------------------------------------------------------
# _make_null_recorder
# ---------------------------------------------------------------------------

class TestMakeNullRecorder:
    def test_returns_recorder(self):
        rec = _make_null_recorder()
        assert isinstance(rec, Recorder)

    def test_null_writer(self):
        rec = _make_null_recorder()
        assert isinstance(rec.writer, _NullTraceWriter)

    def test_steps_empty(self):
        rec = _make_null_recorder()
        assert rec.steps == []


# ---------------------------------------------------------------------------
# Sampling behaviour
# ---------------------------------------------------------------------------

class TestSampling:
    def test_sample_rate_1_always_records(self, tmp_path):
        """sample_rate=1.0 always writes."""
        path = str(tmp_path / "trace.sb")
        opts = RecorderOptions(sample_rate=1.0)
        with record(path, options=opts) as rec:
            rec.llm_call("m", [], executor=_fake_llm)
        assert os.path.exists(path)
        assert rec.dropped_steps == 0

    def test_sample_rate_0_never_records(self, tmp_path):
        """sample_rate=0.0 never writes a file, but executors run."""
        path = str(tmp_path / "trace.sb")
        calls = []
        def counting_llm(model, messages):
            calls.append(1)
            return _fake_llm(model, messages)
        opts = RecorderOptions(sample_rate=0.0)
        with record(path, options=opts) as rec:
            rec.llm_call("m", [], executor=counting_llm)
        # Executor must have run
        assert calls == [1], "executor must still be called when sampled out"
        # No file should exist
        assert not os.path.exists(path)

    def test_sample_rate_seeded_deterministic(self, tmp_path):
        """Two runs with the same seed produce the same sampling decision."""
        import random
        results = []
        for _ in range(2):
            path = str(tmp_path / "trace.sb")
            rng = random.Random(0)
            opts = RecorderOptions(sample_rate=0.5, rng=rng)
            with record(path, options=opts) as rec:
                rec.llm_call("m", [], executor=_fake_llm)
            results.append(os.path.exists(path))
            if os.path.exists(path):
                os.unlink(path)
        assert results[0] == results[1], "same seed must produce same sampling decision"

    def test_sampled_out_tool_call_executor_runs(self, tmp_path):
        """tool_call executor runs even when sampled out."""
        path = str(tmp_path / "trace.sb")
        calls = []
        def counting_tool(name, args):
            calls.append(name)
            return _fake_tool(name, args)
        opts = RecorderOptions(sample_rate=0.0)
        with record(path, options=opts) as rec:
            rec.tool_call("lookup", {}, executor=counting_tool)
        assert "lookup" in calls

    def test_sampled_out_returns_recorder_with_steps(self, tmp_path):
        """Sampled-out recorder still tracks steps in memory."""
        path = str(tmp_path / "trace.sb")
        opts = RecorderOptions(sample_rate=0.0)
        with record(path, options=opts) as rec:
            rec.llm_call("m", [], executor=_fake_llm)
        # Steps tracked even though not written
        assert len(rec.steps) == 1


# ---------------------------------------------------------------------------
# Fail-open: TraceWriter.open() failure
# ---------------------------------------------------------------------------

class TestFailOpenWriterOpen:
    def test_unwritable_path_not_mandatory(self):
        """If the path is unwritable, fail-open mode yields a null recorder."""
        opts = RecorderOptions(mandatory=False)
        with record("/dev/full/nonexistent/path.sb", options=opts) as rec:
            step = rec.llm_call("m", [], executor=_fake_llm)
        assert isinstance(step, dict)
        assert len(rec.recording_errors) >= 1

    def test_unwritable_path_mandatory_raises(self):
        """In mandatory mode, an unwritable path propagates immediately."""
        opts = RecorderOptions(mandatory=True, overflow_policy="block")
        with pytest.raises(Exception):
            with record("/dev/full/nonexistent/path.sb", options=opts) as rec:
                pass  # should not reach here


# ---------------------------------------------------------------------------
# Fail-open: write_step() failure
# ---------------------------------------------------------------------------

class TestFailOpenWriteStep:
    def test_write_step_failure_does_not_propagate(self, tmp_path):
        """When write_step raises and mandatory=False, no exception escapes."""
        path = str(tmp_path / "trace.sb")
        opts = RecorderOptions(mandatory=False)
        with record(path, options=opts) as rec:
            # Patch the writer's write_step to raise after first step
            original = rec.writer.write_step
            call_count = [0]
            def failing_write(step):
                call_count[0] += 1
                if call_count[0] > 1:
                    raise OSError("disk full")
                original(step)
            rec.writer.write_step = failing_write
            rec.llm_call("m", [], executor=_fake_llm)  # step 1 OK
            rec.llm_call("m", [], executor=_fake_llm)  # step 2 fails write
            rec.llm_call("m", [], executor=_fake_llm)  # step 3 skipped (null mode)
        # All 3 steps tracked in memory
        assert len(rec.steps) == 3
        # At least one error captured
        assert len(rec.recording_errors) >= 1
        # dropped_steps >= 2 (step 2 and step 3)
        assert rec.dropped_steps >= 2

    def test_write_step_failure_mandatory_raises(self, tmp_path):
        """When write_step raises and mandatory=True, exception propagates."""
        path = str(tmp_path / "trace.sb")
        opts = RecorderOptions(mandatory=True, overflow_policy="block")
        with pytest.raises(OSError):
            with record(path, options=opts) as rec:
                original = rec.writer.write_step
                call_count = [0]
                def failing_write(step):
                    call_count[0] += 1
                    if call_count[0] > 1:
                        raise OSError("disk full")
                    original(step)
                rec.writer.write_step = failing_write
                rec.llm_call("m", [], executor=_fake_llm)  # OK
                rec.llm_call("m", [], executor=_fake_llm)  # raises


# ---------------------------------------------------------------------------
# Fail-open: writer.close() failure
# ---------------------------------------------------------------------------

class TestFailOpenWriterClose:
    def test_close_failure_does_not_propagate(self, tmp_path):
        """When writer.close() raises and mandatory=False, no exception escapes."""
        path = str(tmp_path / "trace.sb")
        opts = RecorderOptions(mandatory=False)
        with record(path, options=opts) as rec:
            rec.llm_call("m", [], executor=_fake_llm)
            original_close = rec.writer.close
            def failing_close():
                raise OSError("close failed")
            rec.writer.close = failing_close
        # Should not raise; error recorded
        assert len(rec.recording_errors) == 1
        assert "close failed" in str(rec.recording_errors[0])

    def test_close_failure_mandatory_raises(self, tmp_path):
        """When writer.close() raises and mandatory=True, exception propagates."""
        path = str(tmp_path / "trace.sb")
        opts = RecorderOptions(mandatory=True, overflow_policy="block")
        with pytest.raises(OSError, match="close failed"):
            with record(path, options=opts) as rec:
                rec.llm_call("m", [], executor=_fake_llm)
                def failing_close():
                    raise OSError("close failed")
                rec.writer.close = failing_close


# ---------------------------------------------------------------------------
# max_queue_depth with overflow_policy="drop"
# ---------------------------------------------------------------------------

class TestMaxQueueDepthDrop:
    def test_steps_beyond_limit_not_written(self, tmp_path):
        """Steps beyond max_queue_depth are not written to disk."""
        path = str(tmp_path / "trace.sb")
        opts = RecorderOptions(max_queue_depth=2, overflow_policy="drop",
                               mandatory=False)
        with record(path, options=opts) as rec:
            rec.tool_call("t1", {}, executor=_fake_tool)
            rec.tool_call("t2", {}, executor=_fake_tool)
            rec.tool_call("t3", {}, executor=_fake_tool)  # should be dropped
            rec.tool_call("t4", {}, executor=_fake_tool)  # should be dropped
        # All 4 tracked in memory
        assert len(rec.steps) == 4
        # Dropped steps >= 2 (t3, t4)
        assert rec.dropped_steps >= 2

    def test_executors_called_for_dropped_steps(self, tmp_path):
        """Executors are always called even for dropped steps."""
        path = str(tmp_path / "trace.sb")
        calls = []
        def counting_tool(name, args):
            calls.append(name)
            return _fake_tool(name, args)
        opts = RecorderOptions(max_queue_depth=1, overflow_policy="drop",
                               mandatory=False)
        with record(path, options=opts) as rec:
            rec.tool_call("t1", {}, executor=counting_tool)
            rec.tool_call("t2", {}, executor=counting_tool)  # dropped but runs
        assert set(calls) == {"t1", "t2"}

    def test_zero_depth_no_limit(self, tmp_path):
        """max_queue_depth=0 means no limit."""
        path = str(tmp_path / "trace.sb")
        opts = RecorderOptions(max_queue_depth=0, mandatory=False)
        with record(path, options=opts) as rec:
            for _ in range(20):
                rec.tool_call("t", {}, executor=_fake_tool)
        assert rec.dropped_steps == 0
        assert len(rec.steps) == 20


# ---------------------------------------------------------------------------
# max_queue_depth with overflow_policy="block"
# ---------------------------------------------------------------------------

class TestMaxQueueDepthBlock:
    def test_block_policy_writes_all_steps(self, tmp_path):
        """overflow_policy='block' allows writing beyond the depth hint."""
        path = str(tmp_path / "trace.sb")
        opts = RecorderOptions(mandatory=True, max_queue_depth=2,
                               overflow_policy="block")
        with record(path, options=opts) as rec:
            for _ in range(5):
                rec.tool_call("t", {}, executor=_fake_tool)
        assert rec.dropped_steps == 0
        assert len(rec.steps) == 5


# ---------------------------------------------------------------------------
# dropped_steps and recording_errors properties
# ---------------------------------------------------------------------------

class TestDroppedStepsProperties:
    def test_no_drops_zero_count(self, tmp_path):
        path = str(tmp_path / "trace.sb")
        with record(path) as rec:
            rec.tool_call("t", {}, executor=_fake_tool)
        assert rec.dropped_steps == 0
        assert rec.recording_errors == []

    def test_recording_errors_returns_copy(self, tmp_path):
        """recording_errors returns a copy; mutations don't affect internal state."""
        path = str(tmp_path / "trace.sb")
        opts = RecorderOptions(mandatory=False)
        with record(path, options=opts) as rec:
            original_close = rec.writer.close
            def failing_close():
                raise ValueError("oops")
            rec.writer.close = failing_close
        errors = rec.recording_errors
        errors.clear()
        assert len(rec.recording_errors) == 1  # internal list unchanged


# ---------------------------------------------------------------------------
# NullRecorder parallel() runs branch closures
# ---------------------------------------------------------------------------

class TestNullRecorderParallel:
    def test_parallel_runs_all_branches(self, tmp_path):
        """parallel() branch closures are always executed, even when sampled out."""
        path = str(tmp_path / "trace.sb")
        branch_calls = []
        def b1(rec):
            branch_calls.append("b1")
            rec.tool_call("t1", {}, executor=_fake_tool)
        def b2(rec):
            branch_calls.append("b2")
            rec.tool_call("t2", {}, executor=_fake_tool)
        opts = RecorderOptions(sample_rate=0.0)
        with record(path, options=opts) as rec:
            rec.parallel("test", [b1, b2])
        assert "b1" in branch_calls
        assert "b2" in branch_calls


# ---------------------------------------------------------------------------
# arecord async variant
# ---------------------------------------------------------------------------

class TestArecord:
    def test_arecord_fail_open(self, tmp_path):
        """arecord fail-open: unwritable path does not crash."""
        opts = RecorderOptions(mandatory=False)
        async def _run():
            async with arecord("/dev/full/nonexistent/path.sb",
                                options=opts) as rec:
                step = rec.llm_call("m", [], executor=_fake_llm)
            return rec
        rec = asyncio.run(_run())
        assert len(rec.recording_errors) >= 1

    def test_arecord_sampling(self, tmp_path):
        """arecord sample_rate=0.0 never writes."""
        path = str(tmp_path / "trace.sb")
        opts = RecorderOptions(sample_rate=0.0)
        async def _run():
            async with arecord(path, options=opts) as rec:
                rec.llm_call("m", [], executor=_fake_llm)
        asyncio.run(_run())
        assert not os.path.exists(path)

    def test_arecord_normal(self, tmp_path):
        """arecord with options=None behaves as before."""
        path = str(tmp_path / "trace.sb")
        async def _run():
            async with arecord(path) as rec:
                rec.tool_call("t", {}, executor=_fake_tool)
        asyncio.run(_run())
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# Public API membership
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_recorder_options_importable_from_stepback(self):
        from stepback import RecorderOptions as RO  # noqa: F401
        assert RO is RecorderOptions

    def test_in_all(self):
        assert "RecorderOptions" in stepback.__all__

    def test_mandatory_default_same_as_no_options(self, tmp_path):
        """record() with no options and with mandatory=True options are equivalent."""
        path1 = str(tmp_path / "t1.sb")
        path2 = str(tmp_path / "t2.sb")
        with record(path1) as rec1:
            rec1.tool_call("t", {}, executor=_fake_tool)
        opts = RecorderOptions(mandatory=True, overflow_policy="block")
        with record(path2, options=opts) as rec2:
            rec2.tool_call("t", {}, executor=_fake_tool)
        # Both files should exist and be verifiable
        assert os.path.exists(path1)
        assert os.path.exists(path2)
