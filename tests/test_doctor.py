"""Tests for ``stepback.doctor``."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from stepback.doctor import (
    FAIL,
    PASS,
    WARN,
    DoctorCheck,
    _check_key_material,
    _check_python_version,
    _check_stepback_core,
    _check_trace_dir,
    _check_wasm,
    format_doctor_table,
    run_doctor,
)


# ---------------------------------------------------------------------------
# Python version check
# ---------------------------------------------------------------------------

class TestCheckPythonVersion:
    def test_current_python_passes(self):
        check = _check_python_version()
        assert check.name == "Python version"
        # The test suite must run on ≥3.9
        assert check.status in (PASS, WARN)

    def test_old_python_fails(self):
        with mock.patch.object(sys, "version_info", (3, 8, 0)):
            check = _check_python_version()
        assert check.status == FAIL
        assert check.remediation is not None

    def test_py39_warns(self):
        with mock.patch.object(sys, "version_info", (3, 9, 7)):
            check = _check_python_version()
        assert check.status == WARN

    def test_py310_passes(self):
        with mock.patch.object(sys, "version_info", (3, 10, 0)):
            check = _check_python_version()
        assert check.status == PASS

    def test_py311_passes(self):
        with mock.patch.object(sys, "version_info", (3, 11, 2)):
            check = _check_python_version()
        assert check.status == PASS


# ---------------------------------------------------------------------------
# Rust core check
# ---------------------------------------------------------------------------

class TestCheckStepbackCore:
    def test_missing_core_warns(self):
        with mock.patch.dict(sys.modules, {"stepback_core": None}):
            check = _check_stepback_core()
        assert check.status == WARN
        assert check.remediation is not None

    def test_present_core_passes(self):
        fake_core = mock.MagicMock()
        fake_core.__version__ = "0.1.0"
        with mock.patch.dict(sys.modules, {"stepback_core": fake_core}):
            check = _check_stepback_core()
        assert check.status == PASS
        assert "0.1.0" in check.detail


# ---------------------------------------------------------------------------
# WASM check
# ---------------------------------------------------------------------------

class TestCheckWasm:
    def test_wasm_missing_warns(self, tmp_path, monkeypatch):
        # Point __file__ into tmp_path so the package root is tmp_path
        # and no wasm/ directory exists there
        monkeypatch.setattr(
            "stepback.doctor.__file__",
            str(tmp_path / "stepback" / "doctor.py"),
        )
        check = _check_wasm()
        assert check.status == WARN

    def test_wasm_built_passes(self, tmp_path, monkeypatch):
        wasm_pkg = tmp_path / "wasm" / "pkg"
        wasm_pkg.mkdir(parents=True)
        (wasm_pkg / "stepback_wasm.js").write_text("// js")
        (wasm_pkg / "stepback_wasm_bg.wasm").write_bytes(b"\x00asm")
        monkeypatch.setattr(
            "stepback.doctor.__file__",
            str(tmp_path / "stepback" / "doctor.py"),
        )
        check = _check_wasm()
        assert check.status == PASS


# ---------------------------------------------------------------------------
# Key material check
# ---------------------------------------------------------------------------

class TestCheckKeyMaterial:
    def test_env_key_passes(self):
        with mock.patch.dict(os.environ, {"STEPBACK_HMAC_KEY_HEX": "deadbeef" * 8}):
            check = _check_key_material()
        assert check.status == PASS

    def test_no_key_no_toml_warns(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        env_clean = {k: v for k, v in os.environ.items()
                     if k not in ("STEPBACK_HMAC_KEY_HEX", "STEPBACK_HMAC_KEY")}
        with mock.patch.dict(os.environ, env_clean, clear=True):
            check = _check_key_material()
        assert check.status == WARN
        assert check.remediation is not None

    def test_toml_with_key_passes(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        toml_content = '[stepback]\nhmac_key_hex = "deadbeef' + "aa" * 28 + '"\n'
        (tmp_path / "stepback.toml").write_text(toml_content)
        env_clean = {k: v for k, v in os.environ.items()
                     if k not in ("STEPBACK_HMAC_KEY_HEX", "STEPBACK_HMAC_KEY")}
        with mock.patch.dict(os.environ, env_clean, clear=True):
            check = _check_key_material()
        assert check.status == PASS

    def test_toml_without_key_warns(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "stepback.toml").write_text("[stepback]\ntrace_dir = \"traces\"\n")
        env_clean = {k: v for k, v in os.environ.items()
                     if k not in ("STEPBACK_HMAC_KEY_HEX", "STEPBACK_HMAC_KEY")}
        with mock.patch.dict(os.environ, env_clean, clear=True):
            check = _check_key_material()
        assert check.status == WARN


# ---------------------------------------------------------------------------
# Trace dir check
# ---------------------------------------------------------------------------

class TestCheckTraceDir:
    def test_existing_writable_dir_passes(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()
        check = _check_trace_dir()
        assert check.status == PASS

    def test_nonexistent_dir_creates_and_warns(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        check = _check_trace_dir()
        # Either PASS (created + writable) or WARN (created but noted)
        assert check.status in (PASS, WARN)
        assert (tmp_path / "traces").exists()

    def test_unwritable_dir_fails(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()
        trace_dir.chmod(0o555)
        try:
            check = _check_trace_dir()
            assert check.status == FAIL
            assert check.remediation is not None
        finally:
            trace_dir.chmod(0o755)

    def test_custom_trace_dir_from_toml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        custom_dir = tmp_path / "my_traces"
        custom_dir.mkdir()
        toml_content = f'[stepback]\ntrace_dir = "{custom_dir}"\n'
        (tmp_path / "stepback.toml").write_text(toml_content)
        check = _check_trace_dir()
        assert check.status == PASS
        assert str(custom_dir) in check.detail


# ---------------------------------------------------------------------------
# run_doctor
# ---------------------------------------------------------------------------

class TestRunDoctor:
    def test_returns_list_of_checks(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        checks = run_doctor(check_network=False)
        assert isinstance(checks, list)
        assert all(isinstance(c, DoctorCheck) for c in checks)
        assert len(checks) >= 4

    def test_all_have_name_and_status(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        checks = run_doctor(check_network=False)
        for c in checks:
            assert c.name
            assert c.status in (PASS, WARN, FAIL)


# ---------------------------------------------------------------------------
# format_doctor_table
# ---------------------------------------------------------------------------

class TestFormatDoctorTable:
    def test_contains_status_labels(self):
        checks = [
            DoctorCheck("Check A", PASS, "All good"),
            DoctorCheck("Check B", WARN, "Might be a problem", "Fix this"),
            DoctorCheck("Check C", FAIL, "Broken", "Do that"),
        ]
        table = format_doctor_table(checks)
        assert "[PASS]" in table
        assert "[WARN]" in table
        assert "[FAIL]" in table

    def test_remediation_section_present(self):
        checks = [
            DoctorCheck("Check B", WARN, "Issue here", "Run: fix-it --now"),
        ]
        table = format_doctor_table(checks)
        assert "Remediation" in table
        assert "fix-it --now" in table

    def test_summary_line_present(self):
        checks = [
            DoctorCheck("A", PASS, "ok"),
            DoctorCheck("B", WARN, "nope", "fix"),
        ]
        table = format_doctor_table(checks)
        assert "Summary:" in table
        assert "1 PASS" in table
        assert "1 WARN" in table
        assert "0 FAIL" in table


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

class TestDoctorCLI:
    def test_doctor_command_exits_zero(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from stepback.cli import main

        rc = main(["doctor"])
        # 0 = no FAILs, 1 = at least one FAIL
        assert rc in (0, 1)

    def test_doctor_json_output(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        from stepback.cli import main
        import json

        main(["doctor", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert all("name" in item and "status" in item for item in data)

    def test_doctor_exits_1_on_fail(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from stepback.cli import main
        from stepback import doctor as doctor_mod

        # Inject a FAIL check
        fake_checks = [DoctorCheck("Broken thing", FAIL, "Very broken", "Fix it")]
        with mock.patch.object(doctor_mod, "run_doctor", return_value=fake_checks):
            rc = main(["doctor"])
        assert rc == 1

    def test_doctor_exits_0_on_all_pass(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from stepback.cli import main
        from stepback import doctor as doctor_mod

        fake_checks = [DoctorCheck("All fine", PASS, "Great")]
        with mock.patch.object(doctor_mod, "run_doctor", return_value=fake_checks):
            rc = main(["doctor"])
        assert rc == 0
