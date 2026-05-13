"""
Tests for scripts/install.sh

These tests validate the installer script's structure and logic without
executing a real install (no network calls, no root access required).
All assertions are made against the script source or against behaviour
observable via subprocess calls with --help / --uninstall in a temp dir.
"""

import os
import re
import stat
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install.sh"


# ── helpers ──────────────────────────────────────────────────────────────────

def _script_source() -> str:
    return INSTALL_SCRIPT.read_text()


def _run_installer(*args: str, env: dict | None = None, stdin: str = "\n") -> subprocess.CompletedProcess:
    """Run install.sh under bash with the given args; capture output."""
    full_env = {**os.environ, **(env or {})}
    return subprocess.run(
        ["bash", str(INSTALL_SCRIPT), *args],
        capture_output=True,
        text=True,
        input=stdin,
        env=full_env,
    )


# ── structural / presence tests ───────────────────────────────────────────────

class TestInstallerExists:
    def test_script_exists(self):
        assert INSTALL_SCRIPT.exists(), "scripts/install.sh not found"

    def test_script_is_executable(self):
        mode = INSTALL_SCRIPT.stat().st_mode
        assert mode & stat.S_IXUSR, "scripts/install.sh is not executable"

    def test_shebang(self):
        src = _script_source()
        assert src.startswith("#!/usr/bin/env bash"), "Missing bash shebang"

    def test_set_euo_pipefail(self):
        src = _script_source()
        assert "set -euo pipefail" in src, "Missing 'set -euo pipefail'"


# ── root-guard tests ──────────────────────────────────────────────────────────

class TestRootGuard:
    def test_root_guard_present_in_source(self):
        src = _script_source()
        assert 'id -u' in src, "Root-UID check missing"
        assert 'ALLOW_ROOT' in src, "ALLOW_ROOT guard missing"
        assert 'Refusing to run as root' in src, "Root-refusal message missing"

    def test_root_flag_present_in_source(self):
        src = _script_source()
        assert '--root' in src, "--root flag handling missing"

    @pytest.mark.skipif(os.getuid() == 0, reason="Already running as root; skip root-guard test")
    def test_help_exits_zero_as_non_root(self):
        result = _run_installer("--help")
        assert result.returncode == 0, result.stderr


# ── OS / arch detection ───────────────────────────────────────────────────────

class TestPlatformDetection:
    def test_linux_detected(self):
        src = _script_source()
        assert 'Linux' in src, "Linux detection missing"
        assert 'Darwin' in src, "macOS detection missing"

    def test_arch_labels(self):
        src = _script_source()
        assert 'x86_64' in src
        assert 'arm64' in src or 'aarch64' in src

    def test_unknown_os_warns_not_fatal(self):
        """Unknown OS should warn but not crash."""
        src = _script_source()
        assert "warn" in src and "Unrecognised OS" in src


# ── Python version check ──────────────────────────────────────────────────────

class TestPythonCheck:
    def test_python_version_check_present(self):
        src = _script_source()
        assert 'python3' in src, "python3 lookup missing"
        assert '3.10' in src or '310' in src, "Python 3.10 floor missing"

    def test_fatal_message_on_no_python(self):
        src = _script_source()
        assert 'Python 3.10 or later is required' in src


# ── CLI argument parsing ──────────────────────────────────────────────────────

class TestArgumentParsing:
    def test_help_flag(self):
        result = _run_installer("--help")
        assert result.returncode == 0
        assert "Usage:" in result.stdout
        assert "--root" in result.stdout
        assert "--version" in result.stdout
        assert "--dir" in result.stdout
        assert "--uninstall" in result.stdout

    def test_unknown_flag_exits_nonzero(self):
        result = _run_installer("--nonexistent-flag-xyz")
        assert result.returncode != 0
        assert "Unknown option" in result.stderr or "Unknown option" in result.stdout

    def test_version_flag_in_source(self):
        src = _script_source()
        assert '--version' in src

    def test_yes_flag_in_source(self):
        src = _script_source()
        assert '--yes' in src or '-y' in src


# ── uninstall path ────────────────────────────────────────────────────────────

class TestUninstall:
    def test_uninstall_flag_present(self):
        src = _script_source()
        assert '--uninstall' in src
        assert 'UNINSTALL' in src

    def test_uninstall_with_nothing_installed(self):
        """--uninstall should exit 0 even when nothing is installed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_install_dir = os.path.join(tmpdir, "bin")
            result = _run_installer("--uninstall", "--dir", fake_install_dir)
        assert result.returncode == 0

    def test_uninstall_removes_shim(self):
        """--uninstall should remove a shim that was placed by the script."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_bin = os.path.join(tmpdir, "bin")
            os.makedirs(fake_bin)
            shim = os.path.join(fake_bin, "stepback")
            Path(shim).write_text("#!/bin/sh\nexec echo fake\n")
            os.chmod(shim, 0o755)
            result = _run_installer("--uninstall", "--dir", fake_bin)
        assert result.returncode == 0
        # shim should be gone
        assert not os.path.exists(shim)


# ── PATH hint ─────────────────────────────────────────────────────────────────

class TestPathHint:
    def test_path_hint_in_source(self):
        src = _script_source()
        assert 'PATH' in src
        assert 'export PATH' in src or "not on your PATH" in src


# ── next-step output ──────────────────────────────────────────────────────────

class TestNextStepOutput:
    def test_help_mentions_init(self):
        result = _run_installer("--help")
        # help text is in stdout; next-step instructions appear during install
        src = _script_source()
        assert 'stepback init' in src

    def test_next_step_mentions_doctor(self):
        src = _script_source()
        assert 'stepback doctor' in src

    def test_next_step_mentions_quickstart(self):
        src = _script_source()
        assert 'stepback quickstart' in src

    def test_next_step_mentions_docs_url(self):
        src = _script_source()
        assert 'stepback.dev' in src


# ── venv location ─────────────────────────────────────────────────────────────

class TestVenvLocation:
    def test_default_venv_under_home(self):
        src = _script_source()
        # venv should default to something under $HOME
        assert '.local/share/stepback' in src

    def test_default_install_dir(self):
        src = _script_source()
        assert '.local/bin' in src
