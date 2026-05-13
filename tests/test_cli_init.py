"""Tests for `stepback init` CLI command (step 151)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from stepback.cli import main as cli_main


# ------------------------------------------------------------------ helpers

def run_init(tmp_path: Path, extra_args=None) -> int:
    args = ["init", str(tmp_path)]
    if extra_args:
        args.extend(extra_args)
    return cli_main(args)


# ------------------------------------------------------------------ basic scaffold

class TestInit:
    def test_creates_stepback_toml(self, tmp_path):
        rc = run_init(tmp_path)
        assert rc == 0
        toml = tmp_path / "stepback.toml"
        assert toml.exists(), "stepback.toml not created"
        text = toml.read_text()
        assert "[stepback]" in text
        assert "trace_dir" in text

    def test_creates_traces_directory(self, tmp_path):
        run_init(tmp_path)
        assert (tmp_path / "traces").is_dir()

    def test_creates_quickstart_agent(self, tmp_path):
        run_init(tmp_path)
        agent = tmp_path / "examples" / "quickstart" / "agent.py"
        assert agent.exists(), "quickstart agent.py not created"
        text = agent.read_text()
        assert "run_recorded_agent" in text
        assert "stepback.record" in text

    def test_records_demo_trace(self, tmp_path):
        run_init(tmp_path)
        trace = tmp_path / "examples" / "quickstart" / "traces" / "quickstart.sb"
        assert trace.exists(), "demo trace not recorded"
        assert trace.stat().st_size > 0

    def test_demo_trace_is_replayable(self, tmp_path):
        run_init(tmp_path)
        trace = tmp_path / "examples" / "quickstart" / "traces" / "quickstart.sb"
        rc = cli_main(["replay", str(trace)])
        assert rc == 0

    def test_replay_has_all_cache_hits(self, tmp_path):
        run_init(tmp_path)
        trace = tmp_path / "examples" / "quickstart" / "traces" / "quickstart.sb"
        # Capture output
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli_main(["replay", str(trace)])
        output = buf.getvalue()
        # 12 steps, all should cache-hit
        assert "cache_hits=12" in output
        assert "dirty=0" in output

    def test_returns_zero_exit_code(self, tmp_path):
        assert run_init(tmp_path) == 0


# ------------------------------------------------------------------ idempotency & --force

class TestInitIdempotency:
    def test_skip_existing_toml_without_force(self, tmp_path, capsys):
        run_init(tmp_path)
        # Second call without --force should skip
        rc = run_init(tmp_path)
        assert rc == 0
        captured = capsys.readouterr()
        assert "skip" in captured.out

    def test_force_overwrites_toml(self, tmp_path):
        run_init(tmp_path)
        toml = tmp_path / "stepback.toml"
        toml.write_text("[stepback]\n# overwritten\n")
        run_init(tmp_path, ["--force"])
        text = toml.read_text()
        assert "trace_dir" in text  # template content restored

    def test_force_overwrites_agent(self, tmp_path):
        run_init(tmp_path)
        agent = tmp_path / "examples" / "quickstart" / "agent.py"
        agent.write_text("# sentinel\n")
        run_init(tmp_path, ["--force"])
        text = agent.read_text()
        assert "run_recorded_agent" in text


# ------------------------------------------------------------------ target directory

class TestInitDirectory:
    def test_default_directory_is_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rc = cli_main(["init"])
        assert rc == 0
        assert (tmp_path / "stepback.toml").exists()

    def test_explicit_nonexistent_dir_created(self, tmp_path):
        target = tmp_path / "newproject"
        rc = cli_main(["init", str(target)])
        assert rc == 0
        assert (target / "stepback.toml").exists()
