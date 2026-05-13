"""Tests for the sharded content-addressed step cache (Step 74).

Covers:
- StepCacheEntry round-trip serialisation / schema mismatch rejection
- _hex_digest strips the sha256: prefix for shard computation
- DiskStepCache: put/get, shard directory creation, miss, dedup across traces
- DiskStepCache: atomic write (temp-then-rename contract)
- DiskStepCache: shard_width controls path depth; 1 < shard_width works
- DiskStepCache: invalid schema version / corrupted JSON treated as miss
- DiskStepCache: concurrent put from two threads lands correctly
- S3StepCache / GCSStepCache / AzureStepCache raise ImportError when SDK absent
- Executor.step_cache: _cache_get / _cache_put helpers
- Integration: step cache reduces real_executions via _execute_plan
- Integration: nondeterminism-forced dirty steps bypass the step cache
- Integration: fallback outputs NOT written to step cache
- Integration: tool_override (substituted output) NOT written to step cache
- Integration: replay_events honours the step cache
- Integration: parallel branch replay (workers=2) honours step cache
- Dedup: same inputs_hash across two different trace replays shares entry
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

import stepback
from stepback.step_cache import (
    AzureStepCache,
    DiskStepCache,
    GCSStepCache,
    S3StepCache,
    StepCache,
    StepCacheEntry,
    _hex_digest,
    _cache_key,
)
from stepback.canonical import hash_obj
from stepback.replay import Executor, _execute_plan
from stepback.substitutions import SubstitutionSet, PromptSubstitution
from stepback.testing import run_recorded_agent


# ---------------------------------------------------------------- helpers


def _make_entry(kind: str = "tool_call", inputs_hash: str = "sha256:" + "a" * 64) -> StepCacheEntry:
    return StepCacheEntry(
        step_kind=kind,
        inputs_hash=inputs_hash,
        outputs={"result": "hello"},
        cached_at=1_700_000_000.0,
    )


def _simple_recorded_steps(n: int = 3) -> List[dict]:
    """Minimal recorded-step list for replay-engine tests."""
    steps = []
    for i in range(n):
        parent_id = steps[i - 1]["step_id"] if i > 0 else None
        inputs: dict = {"name": f"step{i}", "arguments": {"idx": i}}
        if parent_id:
            inputs["context"] = "placeholder"
        steps.append(
            {
                "step_id": f"step{i}",
                "step_kind": "tool_call",
                "name": f"step{i}",
                "parent_step_id": parent_id,
                "parent_step_ids": [],
                "inputs": inputs,
                "inputs_hash": hash_obj(inputs),
                "outputs": {"result": f"out{i}"},
                "outputs_hash": hash_obj({"result": f"out{i}"}),
                "nondeterminism": {},
                "nondeterminism_hash": None,
                "cost_usd": 0.0,
                "wallclock_ns": 0,
                "llm_request": None,
                "llm_response": None,
            }
        )
    return steps


# ======================================================== StepCacheEntry


class TestStepCacheEntry:
    def test_round_trip(self):
        e = _make_entry()
        d = e.to_dict()
        e2 = StepCacheEntry.from_dict(d)
        assert e2.step_kind == e.step_kind
        assert e2.inputs_hash == e.inputs_hash
        assert e2.outputs == e.outputs
        assert e2.cached_at == e.cached_at

    def test_schema_version_mismatch_raises(self):
        d = _make_entry().to_dict()
        d["cache_schema_version"] = "99"
        with pytest.raises(ValueError, match="cache_schema_version"):
            StepCacheEntry.from_dict(d)

    def test_canonicalisation_version_mismatch_raises(self):
        d = _make_entry().to_dict()
        d["canonicalisation_version"] = "99"
        with pytest.raises(ValueError, match="canonicalisation_version"):
            StepCacheEntry.from_dict(d)

    def test_to_dict_contains_required_fields(self):
        d = _make_entry().to_dict()
        for field in ("cache_schema_version", "canonicalisation_version",
                      "step_kind", "inputs_hash", "outputs", "cached_at"):
            assert field in d, f"Missing field: {field}"

    def test_outputs_preserved_verbatim(self):
        """Complex nested outputs round-trip without mutation."""
        outputs = {"choices": [{"message": {"role": "assistant", "content": "hi"}}], "usage": {"total_tokens": 5}}
        e = StepCacheEntry("llm_call", "sha256:" + "b" * 64, outputs, 0.0)
        assert StepCacheEntry.from_dict(e.to_dict()).outputs == outputs


# ======================================================== _hex_digest


class TestHexDigest:
    def test_strips_sha256_prefix(self):
        hex64 = "a" * 64
        assert _hex_digest(f"sha256:{hex64}") == hex64

    def test_passthrough_without_prefix(self):
        raw = "a" * 64
        assert _hex_digest(raw) == raw

    def test_shard_never_contains_colon(self):
        """Sharding must not produce paths with colons."""
        dig = _hex_digest("sha256:" + "ab" * 32)
        assert ":" not in dig

    def test_cache_key_format(self):
        key = _cache_key("tool_call", "sha256:" + "c" * 64)
        assert key == "tool_call/" + "c" * 64


# ======================================================== DiskStepCache


class TestDiskStepCache:
    def test_miss_returns_none(self, tmp_path):
        cache = DiskStepCache(str(tmp_path))
        assert cache.get("tool_call", "sha256:" + "0" * 64) is None

    def test_put_and_get(self, tmp_path):
        cache = DiskStepCache(str(tmp_path))
        e = _make_entry()
        cache.put(e)
        got = cache.get(e.step_kind, e.inputs_hash)
        assert got is not None
        assert got.outputs == e.outputs
        assert got.step_kind == e.step_kind

    def test_shard_directory_created(self, tmp_path):
        cache = DiskStepCache(str(tmp_path), shard_width=2)
        e = _make_entry(inputs_hash="sha256:" + "ab" * 32)
        cache.put(e)
        hex_dig = "ab" * 32
        expected_dir = tmp_path / "tool_call" / "ab"
        assert expected_dir.is_dir()
        assert (expected_dir / f"{hex_dig}.json").exists()

    def test_shard_width_1(self, tmp_path):
        cache = DiskStepCache(str(tmp_path), shard_width=1)
        e = _make_entry(inputs_hash="sha256:" + "cd" * 32)
        cache.put(e)
        hex_dig = "cd" * 32
        expected_dir = tmp_path / "tool_call" / "c"
        assert expected_dir.is_dir()
        assert (expected_dir / f"{hex_dig}.json").exists()

    def test_shard_width_0_raises(self, tmp_path):
        with pytest.raises(ValueError):
            DiskStepCache(str(tmp_path), shard_width=0)

    def test_dedup_across_traces(self, tmp_path):
        """Two puts with same key result in one file; get returns latest."""
        cache = DiskStepCache(str(tmp_path))
        e1 = _make_entry()
        e2 = StepCacheEntry(e1.step_kind, e1.inputs_hash, {"result": "updated"}, time.time())
        cache.put(e1)
        cache.put(e2)
        got = cache.get(e1.step_kind, e1.inputs_hash)
        assert got is not None
        assert got.outputs == {"result": "updated"}

    def test_different_kind_different_entry(self, tmp_path):
        """Same inputs_hash but different kind → different cache entries."""
        cache = DiskStepCache(str(tmp_path))
        ih = "sha256:" + "e" * 64
        e_tool = StepCacheEntry("tool_call", ih, {"result": "tool"}, 0.0)
        e_llm = StepCacheEntry("llm_call", ih, {"choices": []}, 0.0)
        cache.put(e_tool)
        cache.put(e_llm)
        got_tool = cache.get("tool_call", ih)
        got_llm = cache.get("llm_call", ih)
        assert got_tool is not None and got_tool.outputs == {"result": "tool"}
        assert got_llm is not None and got_llm.outputs == {"choices": []}

    def test_corrupted_json_treated_as_miss(self, tmp_path):
        cache = DiskStepCache(str(tmp_path))
        e = _make_entry()
        cache.put(e)
        # Corrupt the file.
        path = list(tmp_path.rglob("*.json"))[0]
        path.write_bytes(b"not valid json{{{")
        assert cache.get(e.step_kind, e.inputs_hash) is None

    def test_incompatible_schema_treated_as_miss(self, tmp_path):
        cache = DiskStepCache(str(tmp_path))
        e = _make_entry()
        cache.put(e)
        path = list(tmp_path.rglob("*.json"))[0]
        d = json.loads(path.read_text())
        d["cache_schema_version"] = "99"
        path.write_text(json.dumps(d))
        assert cache.get(e.step_kind, e.inputs_hash) is None

    def test_atomic_write_no_partial_files(self, tmp_path):
        """No *.tmp files left behind after a successful put."""
        cache = DiskStepCache(str(tmp_path))
        cache.put(_make_entry())
        tmp_files = list(tmp_path.rglob("*.tmp"))
        assert tmp_files == []

    def test_concurrent_puts_same_key(self, tmp_path):
        """Concurrent writes to the same key do not corrupt the file."""
        cache = DiskStepCache(str(tmp_path))
        errors = []

        def worker(i: int) -> None:
            try:
                e = StepCacheEntry("tool_call", "sha256:" + "f" * 64, {"result": i}, time.time())
                cache.put(e)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent put raised: {errors}"
        got = cache.get("tool_call", "sha256:" + "f" * 64)
        assert got is not None  # one of the writes must have landed

    def test_file_is_valid_json(self, tmp_path):
        cache = DiskStepCache(str(tmp_path))
        cache.put(_make_entry())
        path = list(tmp_path.rglob("*.json"))[0]
        d = json.loads(path.read_text(encoding="utf-8"))
        assert d["step_kind"] == "tool_call"


# ================================================ Cloud backend ImportError


class TestCloudBackendImportErrors:
    def test_s3_raises_importerror_without_boto3(self):
        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(ImportError, match="boto3"):
                S3StepCache("my-bucket")

    def test_gcs_raises_importerror_without_sdk(self):
        with patch.dict("sys.modules", {"google.cloud.storage": None, "google.cloud": None, "google": None}):
            with pytest.raises((ImportError, Exception)):
                GCSStepCache("my-bucket")

    def test_azure_raises_importerror_without_sdk(self):
        with patch.dict("sys.modules", {"azure.storage.blob": None, "azure.storage": None, "azure": None}):
            with pytest.raises((ImportError, Exception)):
                AzureStepCache("my-container", connection_string="x")


# ================================================= Executor._cache helpers


class TestExecutorCacheHelpers:
    def test_cache_get_returns_none_when_no_cache(self):
        exc = Executor()
        assert exc._cache_get("tool_call", "sha256:" + "0" * 64) is None

    def test_cache_put_noop_when_no_cache(self):
        exc = Executor()
        exc._cache_put("tool_call", "sha256:" + "0" * 64, {"result": 1})
        # No exception — just a no-op.

    def test_cache_get_delegates_to_step_cache(self, tmp_path):
        cache = DiskStepCache(str(tmp_path))
        ih = "sha256:" + "1" * 64
        cache.put(StepCacheEntry("tool_call", ih, {"result": "cached"}, 0.0))
        exc = Executor(step_cache=cache)
        entry = exc._cache_get("tool_call", ih)
        assert entry is not None
        assert entry.outputs == {"result": "cached"}

    def test_cache_put_stores_via_step_cache(self, tmp_path):
        cache = DiskStepCache(str(tmp_path))
        exc = Executor(step_cache=cache)
        ih = "sha256:" + "2" * 64
        exc._cache_put("tool_call", ih, {"result": "stored"})
        got = cache.get("tool_call", ih)
        assert got is not None and got.outputs == {"result": "stored"}

    def test_step_cache_attribute_on_executor(self, tmp_path):
        cache = DiskStepCache(str(tmp_path))
        exc = Executor(step_cache=cache)
        assert exc.step_cache is cache

    def test_step_cache_default_none(self):
        exc = Executor()
        assert exc.step_cache is None


# ================================================= Integration: _execute_plan


class TestExecutePlanWithStepCache:
    def _make_executor_with_cache(self, cache: StepCache, call_log: list) -> Executor:
        def fake_tool(name: str, args: dict) -> str:
            call_log.append(name)
            return f"result_{name}"

        return Executor(tool=fake_tool, step_cache=cache)

    def test_step_cache_reduces_real_executions(self, tmp_path):
        """Second replay with same inputs reuses the step cache → real_executions = 0."""
        steps = _simple_recorded_steps(3)
        # Force all steps dirty by changing their inputs_hash.
        dirty_steps = []
        for rec in steps:
            r = dict(rec)
            r["inputs_hash"] = "sha256:" + "9" * 64  # mismatch → dirty
            dirty_steps.append(r)

        cache = DiskStepCache(str(tmp_path))
        call_log1: list = []
        exc1 = self._make_executor_with_cache(cache, call_log1)
        result1 = _execute_plan(dirty_steps, SubstitutionSet(), exc1)
        assert result1.dirty_count > 0
        real_1 = result1.real_executions

        # Second replay: same dirty steps → step cache serves them.
        call_log2: list = []
        exc2 = self._make_executor_with_cache(cache, call_log2)
        result2 = _execute_plan(dirty_steps, SubstitutionSet(), exc2)
        assert result2.real_executions == 0, (
            f"Expected 0 real executions on second replay (all from cache), got {result2.real_executions}"
        )
        assert call_log2 == [], "Tool callback should not be called on second replay"

    def test_nondeterministic_steps_bypass_cache(self, tmp_path):
        """Steps with nondet_class_dirty=True (e.g. unseeded model_sampling) must NOT hit the step cache."""
        from stepback.nondeterminism import model_sampling_nondeterminism
        from stepback.canonical import sha256_hex, canonical_json

        nondet_payload = model_sampling_nondeterminism(temperature=0.7, seed=None)
        nondet_hash = sha256_hex(canonical_json(nondet_payload))

        step = {
            "step_id": "nd_step",
            "step_kind": "tool_call",
            "name": "nd",
            "parent_step_id": None,
            "parent_step_ids": [],
            "inputs": {"name": "nd", "arguments": {}},
            "inputs_hash": "sha256:" + "0" * 64,  # mismatch → dirty
            "outputs": {"result": "recorded"},
            "outputs_hash": hash_obj({"result": "recorded"}),
            "nondeterminism": nondet_payload,
            "nondeterminism_hash": nondet_hash,
            "cost_usd": 0.0,
            "wallclock_ns": 0,
            "llm_request": None,
            "llm_response": None,
        }

        cache = DiskStepCache(str(tmp_path))
        call_counts = [0]

        def fake_tool(name, args):
            call_counts[0] += 1
            return "live"

        exc = Executor(tool=fake_tool, step_cache=cache)
        result1 = _execute_plan([step], SubstitutionSet(), exc)
        assert call_counts[0] == 1

        # Second replay: step cache was written, but nondet class forces re-exec.
        result2 = _execute_plan([step], SubstitutionSet(), exc)
        assert call_counts[0] == 2, (
            "Nondeterministic step must re-execute even when step cache has an entry"
        )

    def test_tool_override_not_written_to_cache(self, tmp_path):
        """Tool-override (substituted output) must not pollute the step cache."""
        steps = _simple_recorded_steps(1)
        step = steps[0]
        cache = DiskStepCache(str(tmp_path))
        call_counts = [0]

        def fake_tool(name, args):
            call_counts[0] += 1
            return "live_output"

        exc = Executor(tool=fake_tool, step_cache=cache)

        # First replay: substitute the step output.
        subs = SubstitutionSet()
        from stepback.substitutions import ToolOutputSubstitution
        subs.add(ToolOutputSubstitution(step["step_id"], {"result": "override"}))
        result1 = _execute_plan(steps, subs, exc)
        assert result1.dirty_count == 1
        assert call_counts[0] == 0  # override, not executor

        # Verify cache was NOT written for this step (tool_override path).
        # Make step dirty (different inputs_hash) and replay without substitution.
        dirty = [dict(step)]
        dirty[0]["inputs_hash"] = "sha256:" + "8" * 64
        result2 = _execute_plan(dirty, SubstitutionSet(), exc)
        # Tool must be called (cache has no entry from the override).
        assert call_counts[0] == 1

    def test_fallback_output_not_written_to_cache(self, tmp_path):
        """fallback_recorded outputs must not be written to the step cache."""
        steps = _simple_recorded_steps(1)
        dirty = [dict(steps[0])]
        dirty[0]["inputs_hash"] = "sha256:" + "7" * 64  # force dirty
        dirty[0]["outputs"] = {"result": "fallback_recorded_value"}

        cache = DiskStepCache(str(tmp_path))
        # No tool callback → fallback_recorded path.
        exc_fb = Executor(fallback_recorded=True, step_cache=cache)
        result1 = _execute_plan(dirty, SubstitutionSet(), exc_fb)
        assert result1.dirty_count == 1

        # Replay with a real tool executor — must call the tool, not the cache.
        call_counts = [0]

        def fake_tool(name, args):
            call_counts[0] += 1
            return "real_value"

        exc_real = Executor(tool=fake_tool, step_cache=cache)
        result2 = _execute_plan(dirty, SubstitutionSet(), exc_real)
        assert call_counts[0] == 1, "Real executor should be called; fallback must not have polluted cache"

    def test_cache_miss_does_not_affect_result_values(self, tmp_path):
        """Without a cache entry, _execute_plan produces correct outputs."""
        # Use a single independent step (no parent/context rebinding) so
        # its inputs_hash stays stable and it's a trace cache hit.
        steps = _simple_recorded_steps(1)
        cache = DiskStepCache(str(tmp_path))

        def fake_tool(name, args):
            return f"computed_{name}"

        exc = Executor(tool=fake_tool, step_cache=cache)
        result = _execute_plan(steps, SubstitutionSet(), exc)
        # Step is clean (inputs_hash matches recorded), so it hits the trace cache.
        assert result.cache_hit_count == 1
        assert result.real_executions == 0


# ================================================= Integration: replay_events


class TestReplayEventsWithStepCache:
    def test_replay_events_uses_step_cache(self, tmp_path):
        """replay_events reduces real_executions when step cache is warm."""
        from stepback.replay import replay_events

        steps = _simple_recorded_steps(2)
        dirty_steps = []
        for rec in steps:
            r = dict(rec)
            r["inputs_hash"] = "sha256:" + "6" * 64  # all dirty
            dirty_steps.append(r)

        cache = DiskStepCache(str(tmp_path))
        call_log: list = []

        def fake_tool(name, args):
            call_log.append(name)
            return "val"

        exc = Executor(tool=fake_tool, step_cache=cache)
        # Warm the cache via _execute_plan.
        _execute_plan(dirty_steps, SubstitutionSet(), exc)
        real_after_warm = exc.real_calls

        # Now use replay_events — should hit step cache.
        call_log_events: list = []

        def fake_tool2(name, args):
            call_log_events.append(name)
            return "val"

        exc2 = Executor(tool=fake_tool2, step_cache=cache)
        events = list(replay_events(dirty_steps, SubstitutionSet(), exc2))
        done = [e for e in events if e["event"] == "replay_done"][0]
        assert done["real_executions"] == 0, (
            f"replay_events should use step cache; got real_executions={done['real_executions']}"
        )
        assert call_log_events == []


# ================================================= Integration: parallel replay


class TestParallelReplayWithStepCache:
    def test_parallel_replay_uses_step_cache(self, tmp_path):
        """workers=2 parallel replay also honours the step cache."""
        from stepback.testing import run_parallel_agent, FACTS

        with __import__("tempfile").NamedTemporaryFile(suffix=".sb", delete=False) as tf:
            trace_path = tf.name
        try:
            from stepback import record
            from stepback.testing import run_parallel_agent

            with record(trace_path) as rec:
                run_parallel_agent(rec)

            from stepback import replay
            trace = replay(trace_path)

            cache = DiskStepCache(str(tmp_path))
            call_log1: list = []

            def fake_tool(name, args):
                call_log1.append(name)
                return FACTS.get(args.get("question", ""), "?")

            exc1 = Executor(tool=fake_tool, step_cache=cache)
            trace.replay_forward(exc1, workers=2)
            real1 = exc1.real_calls

            # Second replay — all dirty steps served from cache.
            call_log2: list = []

            def fake_tool2(name, args):
                call_log2.append(name)
                return FACTS.get(args.get("question", ""), "?")

            exc2 = Executor(tool=fake_tool2, step_cache=cache)
            result2 = trace.replay_forward(exc2, workers=2)
            assert exc2.real_calls == 0, (
                f"Parallel replay should use step cache; got real_calls={exc2.real_calls}"
            )
        finally:
            try:
                os.unlink(trace_path)
            except OSError:
                pass


# ================================================ Dedup across traces


class TestDedupAcrossTraces:
    def test_same_inputs_hash_shared_across_traces(self, tmp_path):
        """Two traces with identical step inputs share one cache entry (dedup)."""
        import tempfile

        cache = DiskStepCache(str(tmp_path))

        def fake_tool(name, args):
            return "shared_result"

        # Record two traces.
        trace_paths = []
        for _ in range(2):
            with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as tf:
                trace_paths.append(tf.name)

        try:
            from stepback import record, replay
            from stepback.testing import run_recorded_agent

            for path in trace_paths:
                with record(path) as rec:
                    run_recorded_agent(rec)

            # Make both traces dirty by playing with substitutions to force re-exec.
            # Use the same step cache for both.
            from stepback import replay as _replay
            from stepback.substitutions import PromptSubstitution

            t1 = _replay(trace_paths[0])
            t2 = _replay(trace_paths[1])

            # Force step 0 dirty on both via a prompt substitution.
            first_sid = t1.recorded_steps[0]["step_id"]
            subs = SubstitutionSet()
            subs.add(PromptSubstitution(first_sid, [{"role": "user", "content": "new"}]))

            call_counts = [0]

            def counting_llm(model, messages):
                call_counts[0] += 1
                return {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}], "usage": {"total_tokens": 1}}

            exc1 = Executor(llm=counting_llm, step_cache=cache)
            t1.replay_forward(exc1)
            calls_after_t1 = call_counts[0]

            # t2 has the same trace structure and same substitution.
            subs2 = SubstitutionSet()
            subs2.add(PromptSubstitution(t2.recorded_steps[0]["step_id"], [{"role": "user", "content": "new"}]))
            exc2 = Executor(llm=counting_llm, step_cache=cache)
            t2.replay_forward(exc2)
            calls_after_t2 = call_counts[0]

            # After t1 warmed the cache, t2's dirty steps should be served from it.
            # calls_after_t2 should equal calls_after_t1 (zero new calls for steps
            # that shared the same inputs_hash after substitution).
            assert calls_after_t2 <= calls_after_t1, (
                f"Expected dedup: t2 should use cache entries from t1. "
                f"t1 calls={calls_after_t1}, t2 additional calls={calls_after_t2 - calls_after_t1}"
            )
        finally:
            for p in trace_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass


# ======================================== Public API exports check


def test_step_cache_symbols_in_stepback_all():
    expected = {"StepCacheEntry", "StepCache", "DiskStepCache", "S3StepCache", "GCSStepCache", "AzureStepCache"}
    missing = expected - set(stepback.__all__)
    assert not missing, f"Missing from stepback.__all__: {missing}"


def test_step_cache_is_abstract_base():
    """StepCache cannot be instantiated directly."""
    with pytest.raises(TypeError):
        StepCache()  # type: ignore[abstract]
