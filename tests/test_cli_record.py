"""Tests for `stepback record` CLI + `stepback.autorecord` ambient recorder.

These cover the README §7 headline command::

    stepback record  --output trace.sb -- python my_agent.py

and the underlying :func:`stepback.autorecord.enable` /
:func:`stepback.autorecord.current_recorder` API.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap

import pytest

from stepback import autorecord, replay
from stepback.recorder import RecorderKey
from stepback.trace_reader import verify_trace


# --------------------------------------------------------- enable() / current

def test_current_recorder_raises_when_inactive():
    autorecord.disable()  # belt + braces
    with pytest.raises(RuntimeError, match="outside an active enable"):
        autorecord.current_recorder()
    assert not autorecord.active()


def test_enable_yields_recorder_and_clears_on_exit(tmp_path):
    out = str(tmp_path / "ambient.sb")
    assert not autorecord.active()
    with autorecord.enable(out, key=RecorderKey.fresh()) as rec:
        assert autorecord.active()
        assert autorecord.current_recorder() is rec
        rec.llm_call(
            "gpt-4o-2024-11-20",
            [{"role": "user", "content": "hi"}],
            executor=lambda model, msgs: {
                "id": "c1",
                "model": model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "ok"}}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    assert not autorecord.active()
    assert os.path.exists(out)
    # Trace is well-formed and contains the recorded step.
    t = replay(out)
    assert len(t.recorded_steps) == 1
    assert t.recorded_steps[0]["step_kind"] == "llm_call"


def test_enable_is_not_reentrant(tmp_path):
    out_a = str(tmp_path / "a.sb")
    out_b = str(tmp_path / "b.sb")
    with autorecord.enable(out_a):
        with pytest.raises(RuntimeError, match="already active"):
            with autorecord.enable(out_b):
                pass
    assert not autorecord.active()


def test_enable_sets_env_var_and_clears(tmp_path):
    out = str(tmp_path / "env.sb")
    assert "STEPBACK_RECORD_PATH" not in os.environ
    with autorecord.enable(out):
        assert os.environ["STEPBACK_RECORD_PATH"] == out
    assert "STEPBACK_RECORD_PATH" not in os.environ


def test_disable_is_idempotent_when_inactive():
    autorecord.disable()
    autorecord.disable()  # no exception


# ----------------------------------------------------------- autopatch openai

def test_openai_autopatch_records_chat_completions(tmp_path, monkeypatch):
    """If `openai` is in sys.modules, `OpenAI(...)` returns a wrapped client
    whose `.chat.completions.create(...)` produces a recorded `llm_call`."""
    import types

    fake_openai = types.ModuleType("openai")

    class _Completions:
        def create(self, *, model, messages, **kw):
            return {
                "id": "cmpl_fake_1",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "fake-response",
                            "tool_calls": None,
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            }

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _OpenAI:
        def __init__(self, *a, **kw):
            self.chat = _Chat()

    fake_openai.OpenAI = _OpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    out = str(tmp_path / "openai.sb")
    with autorecord.enable(out):
        # After enable(), `openai.OpenAI` should be the patched factory
        # returning a WrappedOpenAI.
        client = sys.modules["openai"].OpenAI(api_key="sk-test")
        from stepback.shims import WrappedOpenAI
        assert isinstance(client, WrappedOpenAI)
        resp = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": "ping"}],
        )
        assert resp.choices[0].message.content == "fake-response"

    # The patch must be reverted after enable() exits.
    assert sys.modules["openai"].OpenAI is _OpenAI

    t = replay(out)
    assert len(t.recorded_steps) == 1
    s = t.recorded_steps[0]
    assert s["step_kind"] == "llm_call"
    assert s["llm_request"]["model"] == "gpt-4o-2024-11-20"


# -------------------------------------------------------------- CLI: record

_FIXTURE_SCRIPT = textwrap.dedent(
    """
    import sys
    from stepback.autorecord import current_recorder

    rec = current_recorder()

    def _llm(model, msgs):
        return {
            "id": "c-fixture",
            "model": model,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"}}
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        }

    def _tool(name, args):
        return {"text": f"ran {name} with {args}"}

    rec.llm_call("gpt-4o-2024-11-20", [{"role": "user", "content": "hi"}], executor=_llm)
    rec.tool_call("lookup", {"q": "Acme"}, executor=_tool)
    rec.llm_call("gpt-4o-2024-11-20", [{"role": "user", "content": "bye"}], executor=_llm)
    print("FIXTURE_OK", *sys.argv[1:])
    """
)


def _write_fixture(tmp_path) -> str:
    p = tmp_path / "agent_under_test.py"
    p.write_text(_FIXTURE_SCRIPT, encoding="utf-8")
    return str(p)


def _run_cli(args, env=None) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    e["PYTHONPATH"] = (
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        + os.pathsep
        + e.get("PYTHONPATH", "")
    )
    if env:
        e.update(env)
    return subprocess.run(
        [sys.executable, "-m", "stepback.cli", *args],
        capture_output=True, text=True, env=e,
    )


def test_cli_record_runs_python_script(tmp_path):
    script = _write_fixture(tmp_path)
    out = str(tmp_path / "trace.sb")
    cp = _run_cli(["record", "--output", out, "--", "python", script])
    assert cp.returncode == 0, f"stdout={cp.stdout}\nstderr={cp.stderr}"
    assert "FIXTURE_OK" in cp.stdout
    assert os.path.exists(out)

    t = replay(out)
    assert len(t.recorded_steps) == 3
    kinds = [s["step_kind"] for s in t.recorded_steps]
    assert kinds == ["llm_call", "tool_call", "llm_call"]


def test_cli_record_without_python_token(tmp_path):
    """Bare `script.py` (no leading python) is also accepted."""
    script = _write_fixture(tmp_path)
    out = str(tmp_path / "t2.sb")
    cp = _run_cli(["record", "-o", out, "--", script])
    assert cp.returncode == 0, cp.stderr
    assert os.path.exists(out)
    assert len(replay(out).recorded_steps) == 3


def test_cli_record_passes_argv_to_script(tmp_path):
    script = _write_fixture(tmp_path)
    out = str(tmp_path / "argv.sb")
    cp = _run_cli(["record", "-o", out, "--", "python", script, "alpha", "beta"])
    assert cp.returncode == 0
    assert "FIXTURE_OK alpha beta" in cp.stdout


def test_cli_record_no_command_errors():
    cp = _run_cli(["record", "-o", "/tmp/nope.sb"])
    assert cp.returncode == 2
    assert "must follow `--`" in cp.stderr or "must follow" in cp.stderr


def test_cli_record_captures_script_exception(tmp_path):
    """Even if the script raises, the partial trace is flushed and the
    exception is logged as a step (README §"Use-cases" §1)."""
    bad = tmp_path / "boom.py"
    bad.write_text(
        textwrap.dedent(
            """
            from stepback.autorecord import current_recorder
            rec = current_recorder()
            rec.llm_call(
                "gpt-4o-2024-11-20",
                [{"role": "user", "content": "x"}],
                executor=lambda m, ms: {
                    "id":"c","model":m,
                    "choices":[{"index":0,"message":{"role":"assistant","content":"y"}}],
                    "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2},
                },
            )
            raise RuntimeError("kaboom")
            """
        ),
        encoding="utf-8",
    )
    out = str(tmp_path / "boom.sb")
    cp = _run_cli(["record", "-o", out, "--", "python", str(bad)])
    assert cp.returncode == 1
    assert os.path.exists(out)
    t = replay(out)
    kinds = [s["step_kind"] for s in t.recorded_steps]
    assert "exception" in kinds
    # The llm_call still appears before the exception step.
    assert kinds.index("llm_call") < kinds.index("exception")


def test_cli_record_propagates_systemexit_code(tmp_path):
    """`sys.exit(7)` from the agent should surface as exit code 7."""
    p = tmp_path / "exit7.py"
    p.write_text(
        textwrap.dedent(
            """
            import sys
            from stepback.autorecord import current_recorder
            current_recorder().tool_call(
                "noop", {}, executor=lambda n,a: {"ok":True},
            )
            sys.exit(7)
            """
        ),
        encoding="utf-8",
    )
    out = str(tmp_path / "exit7.sb")
    cp = _run_cli(["record", "-o", out, "--", "python", str(p)])
    assert cp.returncode == 7
    assert os.path.exists(out)
    assert len(replay(out).recorded_steps) == 1


def test_cli_record_trace_round_trips_through_replay_and_inspect(tmp_path):
    script = _write_fixture(tmp_path)
    out = str(tmp_path / "rt.sb")
    cp = _run_cli(["record", "-o", out, "--", "python", script])
    assert cp.returncode == 0

    # The recorded trace should also go through the existing CLI
    # `inspect --json` happily — proving end-to-end that
    # `record` + `inspect` compose.
    cp2 = _run_cli(["inspect", out, "--json"])
    assert cp2.returncode == 0, cp2.stderr
    body = json.loads(cp2.stdout)
    assert body["step_count"] == 3
    assert {s["kind"] for s in body["steps"]} == {"llm_call", "tool_call"}


def test_cli_record_help_documents_subcommand():
    cp = _run_cli(["record", "--help"])
    assert cp.returncode == 0
    assert "ambient" in cp.stdout.lower() or "recorder" in cp.stdout.lower()
    assert "--output" in cp.stdout


def test_cli_top_level_help_lists_record():
    cp = _run_cli(["--help"])
    assert cp.returncode == 0
    assert "record" in cp.stdout
