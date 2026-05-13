"""Tests for async recorder support (Step 98).

Proves that:
1. ``arecord()`` works as an async context manager producing a Recorder.
2. ``get_current_recorder()`` is accessible after ``await`` (context survives
   across yield points within the same task).
3. ``get_current_recorder()`` is accessible in child tasks spawned by
   ``asyncio.create_task`` (ContextVar copies at task-creation time).
4. Two parallel tasks that each spawn their own ``arecord()`` maintain
   completely independent recorders and traces.
5. Task-local parent step ids mean concurrent tasks using the SAME recorder
   record the correct parent_step_id without colliding.
6. ``autorecord.aenable()`` propagates the recorder to async child tasks.
7. ``record()`` (sync) also sets the ContextVar so ``get_current_recorder()``
   works in sync contexts.
8. ``get_current_recorder()`` returns None outside any recording block.

All tests use ``asyncio.run(...)`` from normal pytest functions — no
pytest-asyncio dependency.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pytest

import stepback
from stepback import arecord, get_current_recorder, record
from stepback import autorecord
from stepback.recorder import _RECORDER_VAR, _PARENT_STEP_VAR


# ------------------------------------------------------------------ helpers

def _fake_llm(_model: str, _msgs: list) -> dict:
    return {
        "id": "fake-0",
        "object": "chat.completion",
        "model": _model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "ok",
                                 "tool_calls": None}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _fake_tool(_name: str, _args: dict) -> dict:
    return {"status": "done"}


# ------------------------------------------------------------------ basic

def test_arecord_basic():
    """arecord() opens a trace and yields a Recorder."""
    async def _run():
        with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
            path = f.name
        try:
            async with arecord(path) as rec:
                step = rec.tool_call("ping", {}, executor=_fake_tool)
                assert step["step_kind"] == "tool_call"
            # In-memory steps should include the tool_call step.
            assert any(s["step_kind"] == "tool_call" for s in rec.steps)
        finally:
            os.unlink(path)

    asyncio.run(_run())


def test_arecord_yields_recorder_matches_get_current():
    """The recorder yielded by arecord() is the same object as get_current_recorder()."""
    async def _run():
        with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
            path = f.name
        try:
            async with arecord(path) as rec:
                assert get_current_recorder() is rec
        finally:
            os.unlink(path)

    asyncio.run(_run())


def test_get_current_recorder_none_outside():
    """get_current_recorder() returns None when called outside any recording block."""
    assert get_current_recorder() is None


def test_record_sync_also_sets_contextvar():
    """The synchronous record() context manager also sets the ContextVar."""
    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        path = f.name
    try:
        with record(path) as rec:
            assert get_current_recorder() is rec
        assert get_current_recorder() is None
    finally:
        os.unlink(path)


def test_record_sync_contextvar_reset_after_exit():
    """The ContextVar is reset to None after the sync record() block exits."""
    with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
        path = f.name
    try:
        with record(path):
            pass
        assert get_current_recorder() is None
    finally:
        os.unlink(path)


def test_arecord_contextvar_reset_after_exit():
    """The ContextVar is reset to None after the arecord() block exits."""
    async def _run():
        with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
            path = f.name
        try:
            async with arecord(path):
                pass
            assert get_current_recorder() is None
        finally:
            os.unlink(path)

    asyncio.run(_run())


# ------------------------------------------------------------------ await propagation

def test_context_survives_await():
    """The recorder is accessible after an await inside the same task."""
    async def _check_after_await(expected_rec) -> bool:
        await asyncio.sleep(0)  # yield to event loop
        return get_current_recorder() is expected_rec

    async def _run():
        with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
            path = f.name
        try:
            async with arecord(path) as rec:
                ok = await _check_after_await(rec)
                assert ok, "get_current_recorder() returned wrong value after await"
        finally:
            os.unlink(path)

    asyncio.run(_run())


def test_context_survives_multiple_awaits():
    """Recorder stays bound through a chain of multiple awaits."""
    async def _inner(expected_rec) -> bool:
        await asyncio.sleep(0)
        return get_current_recorder() is expected_rec

    async def _middle(expected_rec) -> bool:
        await asyncio.sleep(0)
        return await _inner(expected_rec)

    async def _run():
        with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
            path = f.name
        try:
            async with arecord(path) as rec:
                assert await _middle(rec)
        finally:
            os.unlink(path)

    asyncio.run(_run())


# ------------------------------------------------------------------ child tasks

def test_context_in_create_task():
    """A child task spawned via asyncio.create_task() sees the recorder."""
    results: list = []

    async def _child(expected_rec) -> None:
        await asyncio.sleep(0)
        results.append(get_current_recorder() is expected_rec)

    async def _run():
        with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
            path = f.name
        try:
            async with arecord(path) as rec:
                task = asyncio.create_task(_child(rec))
                await task
            assert results == [True]
        finally:
            os.unlink(path)

    asyncio.run(_run())


def test_context_in_gather():
    """asyncio.gather() children all see the recorder."""
    async def _child(expected_rec, idx: int) -> bool:
        await asyncio.sleep(0)
        return get_current_recorder() is expected_rec

    async def _run():
        with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
            path = f.name
        try:
            async with arecord(path) as rec:
                results = await asyncio.gather(
                    _child(rec, 0), _child(rec, 1), _child(rec, 2)
                )
            assert all(results)
        finally:
            os.unlink(path)

    asyncio.run(_run())


@pytest.mark.skipif(sys.version_info < (3, 11), reason="asyncio.TaskGroup requires Python 3.11+")
def test_context_in_task_group():
    """asyncio.TaskGroup children see the recorder (Python 3.11+)."""
    results: list = []

    async def _child(expected_rec) -> None:
        await asyncio.sleep(0)
        results.append(get_current_recorder() is expected_rec)

    async def _run():
        with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
            path = f.name
        try:
            async with arecord(path) as rec:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(_child(rec))
                    tg.create_task(_child(rec))
            assert all(results) and len(results) == 2
        finally:
            os.unlink(path)

    asyncio.run(_run())


# ------------------------------------------------------------------ isolation

def test_two_recorders_isolated():
    """Two parallel arecord() contexts produce independent recorders."""
    async def _agent(path: str, model_name: str) -> object:
        async with arecord(path) as rec:
            await asyncio.sleep(0)
            return get_current_recorder()

    async def _run():
        with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f2:
            p1, p2 = f1.name, f2.name
        try:
            r1, r2 = await asyncio.gather(_agent(p1, "m1"), _agent(p2, "m2"))
            assert r1 is not r2
        finally:
            os.unlink(p1)
            os.unlink(p2)

    asyncio.run(_run())


def test_parallel_tasks_independent_parent_chains():
    """Concurrent tasks sharing a recorder maintain task-local parent step ids.

    Task A records tool-a1 then (after yielding) tool-a2.
    Task B records tool-b after yielding once.

    The key invariant:
    - tool-a2's parent_step_id must be tool-a1's step_id, NOT tool-b's step_id.
    - tool-b's parent_step_id must be root's step_id (inherited at creation).
    """
    recorded: list = []

    async def _task_a(rec, root_id: str) -> None:
        step_a = rec.tool_call("tool-a1", {}, executor=_fake_tool)
        recorded.append(("A", "a1", step_a["step_id"], step_a["parent_step_id"]))
        await asyncio.sleep(0)  # yield — task B may run here
        step_a2 = rec.tool_call("tool-a2", {}, executor=_fake_tool)
        recorded.append(("A", "a2", step_a2["step_id"], step_a2["parent_step_id"]))

    async def _task_b(rec, root_id: str) -> None:
        await asyncio.sleep(0)  # yield first so task A records a1 before us
        step_b = rec.tool_call("tool-b", {}, executor=_fake_tool)
        recorded.append(("B", "b", step_b["step_id"], step_b["parent_step_id"]))

    async def _run():
        with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
            path = f.name
        try:
            async with arecord(path) as rec:
                root = rec.tool_call("root", {}, executor=_fake_tool)
                root_id = root["step_id"]
                # Both tasks inherit _PARENT_STEP_VAR = root_id at creation time.
                await asyncio.gather(_task_a(rec, root_id), _task_b(rec, root_id))
        finally:
            os.unlink(path)

    asyncio.run(_run())

    a1 = next((r for r in recorded if r[1] == "a1"), None)
    a2 = next((r for r in recorded if r[1] == "a2"), None)
    b  = next((r for r in recorded if r[1] == "b"),  None)
    assert a1 and a2 and b, f"Missing steps: recorded={recorded}"

    # a1's parent should be root.
    assert a1[3] == "step:1", (
        f"tool-a1's parent_step_id should be root('step:1'), got {a1[3]!r}"
    )
    # a2's parent must be a1 (task-local chain), NOT b.
    assert a2[3] == a1[2], (
        f"tool-a2's parent_step_id should be tool-a1({a1[2]!r}), got {a2[3]!r}. "
        "Parallel task contaminated the parent chain."
    )
    # b's parent should also be root (inherited at task creation, before a1/a2 ran).
    assert b[3] == "step:1", (
        f"tool-b's parent_step_id should be root('step:1'), got {b[3]!r}"
    )


# ------------------------------------------------------------------ recording correctness

def test_arecord_steps_written_to_trace():
    """Steps recorded inside arecord() appear in the written trace file."""
    async def _run():
        with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
            path = f.name
        try:
            async with arecord(path) as rec:
                rec.tool_call("step-one", {"x": 1}, executor=_fake_tool)
                rec.llm_call("gpt-4o", [{"role": "user", "content": "hi"}],
                             executor=_fake_llm)
            kinds = [s["step_kind"] for s in rec.steps]
            assert "tool_call" in kinds
            assert "llm_call" in kinds
        finally:
            os.unlink(path)

    asyncio.run(_run())


def test_arecord_parent_chain_sequential():
    """Sequential steps in arecord() form a correct parent chain."""
    async def _run():
        with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
            path = f.name
        try:
            async with arecord(path) as rec:
                s1 = rec.tool_call("s1", {}, executor=_fake_tool)
                await asyncio.sleep(0)
                s2 = rec.tool_call("s2", {}, executor=_fake_tool)
                assert s2["parent_step_id"] == s1["step_id"]
        finally:
            os.unlink(path)

    asyncio.run(_run())


# ------------------------------------------------------------------ autorecord.aenable

def test_aenable_basic():
    """aenable() installs the ambient recorder and get_current_recorder() works."""
    async def _run():
        with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
            path = f.name
        try:
            async with autorecord.aenable(path, autopatch=False) as rec:
                assert autorecord.active()
                assert autorecord.current_recorder() is rec
                assert get_current_recorder() is rec
            assert not autorecord.active()
        finally:
            os.unlink(path)

    asyncio.run(_run())


def test_aenable_context_in_task():
    """Recorder installed by aenable() is visible in child asyncio tasks."""
    seen: list = []

    async def _child() -> None:
        await asyncio.sleep(0)
        seen.append(get_current_recorder())

    async def _run():
        with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f:
            path = f.name
        try:
            async with autorecord.aenable(path, autopatch=False) as rec:
                task = asyncio.create_task(_child())
                await task
            assert seen and seen[0] is rec
        finally:
            os.unlink(path)

    asyncio.run(_run())


def test_aenable_reentrancy_raises():
    """Calling aenable() while it is already active raises RuntimeError."""
    async def _run():
        with tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix=".sb", delete=False) as f2:
            p1, p2 = f1.name, f2.name
        try:
            async with autorecord.aenable(p1, autopatch=False):
                with pytest.raises(RuntimeError, match="already active"):
                    async with autorecord.aenable(p2, autopatch=False):
                        pass
        finally:
            try:
                os.unlink(p1)
            except OSError:
                pass
            try:
                os.unlink(p2)
            except OSError:
                pass

    asyncio.run(_run())


# ------------------------------------------------------------------ public API

def test_arecord_exported_from_stepback():
    """stepback.arecord is exported from the top-level package."""
    assert hasattr(stepback, "arecord")
    assert stepback.arecord is arecord


def test_get_current_recorder_exported_from_stepback():
    """stepback.get_current_recorder is exported from the top-level package."""
    assert hasattr(stepback, "get_current_recorder")
    assert stepback.get_current_recorder is get_current_recorder
