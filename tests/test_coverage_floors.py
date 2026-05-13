"""Tests for ``scripts/check_coverage_floors.py``.

This locks down Step 26 of ``100_STEPS.md``: the per-module coverage floors
declared in ``pyproject.toml`` must be enforceable by an out-of-the-box dev
install. We test the loader, the evaluator, and the CLI exit codes against a
synthetic ``coverage.json`` so the test does not depend on the real test
suite's runtime coverage numbers (which would create circular dependencies
between the floors and the test that asserts they hold).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_coverage_floors.py"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "check_coverage_floors", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_coverage_json(path: Path, files: dict[str, float]) -> None:
    payload = {
        "files": {
            name: {
                "summary": {
                    "covered_lines": int(pct),
                    "num_statements": 100,
                    "percent_covered": pct,
                    "missing_lines": 100 - int(pct),
                    "excluded_lines": 0,
                }
            }
            for name, pct in files.items()
        },
        "totals": {"percent_covered": 0.0},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_script_exists_and_executable():
    assert SCRIPT_PATH.is_file()


def test_pyproject_declares_required_floors():
    mod = _load_module()
    floors = mod.load_floors(PYPROJECT)
    required = {
        "stepback/canonical.py",
        "stepback/trace_reader.py",
        "stepback/trace_writer.py",
        "stepback/attestation.py",
        "stepback/divergence.py",
        "stepback/replay.py",
    }
    missing = required - set(floors)
    assert not missing, f"Step 26 floors missing for: {sorted(missing)}"
    for path, value in floors.items():
        assert 0 <= value <= 100, f"floor for {path} out of range: {value}"


def test_evaluate_passes_when_all_above_floor(tmp_path: Path):
    mod = _load_module()
    floors = {"stepback/canonical.py": 50.0}
    coverages = {"stepback/canonical.py": 73.5}
    failing, missing = mod.evaluate(floors, coverages)
    assert failing == []
    assert missing == []


def test_evaluate_fails_when_below_floor():
    mod = _load_module()
    floors = {"stepback/canonical.py": 50.0, "stepback/replay.py": 90.0}
    coverages = {"stepback/canonical.py": 49.99, "stepback/replay.py": 95.0}
    failing, missing = mod.evaluate(floors, coverages)
    assert missing == []
    assert len(failing) == 1
    path, actual, floor = failing[0]
    assert path == "stepback/canonical.py"
    assert pytest.approx(actual, rel=1e-6) == 49.99
    assert floor == 50.0


def test_evaluate_reports_missing_files():
    mod = _load_module()
    floors = {"stepback/canonical.py": 50.0}
    coverages: dict[str, float] = {}
    failing, missing = mod.evaluate(floors, coverages)
    assert failing == []
    assert missing == ["stepback/canonical.py"]


def test_cli_exit_zero_when_floors_met(tmp_path: Path):
    cov = tmp_path / "coverage.json"
    py = tmp_path / "pyproject.toml"
    py.write_text(
        """
[tool.coverage.floors]
"stepback/canonical.py" = 50
""".strip(),
        encoding="utf-8",
    )
    _write_coverage_json(cov, {"stepback/canonical.py": 60.0})
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--coverage-json",
            str(cov),
            "--pyproject",
            str(py),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Coverage floors met" in result.stdout


def test_cli_exit_one_when_floor_violated(tmp_path: Path):
    cov = tmp_path / "coverage.json"
    py = tmp_path / "pyproject.toml"
    py.write_text(
        """
[tool.coverage.floors]
"stepback/canonical.py" = 80
""".strip(),
        encoding="utf-8",
    )
    _write_coverage_json(cov, {"stepback/canonical.py": 50.0})
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--coverage-json",
            str(cov),
            "--pyproject",
            str(py),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "NOT met" in result.stderr
    assert "stepback/canonical.py" in result.stderr


def test_cli_exit_two_when_floor_target_missing_from_report(tmp_path: Path):
    cov = tmp_path / "coverage.json"
    py = tmp_path / "pyproject.toml"
    py.write_text(
        """
[tool.coverage.floors]
"stepback/replay.py" = 80
""".strip(),
        encoding="utf-8",
    )
    _write_coverage_json(cov, {"stepback/canonical.py": 99.0})
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--coverage-json",
            str(cov),
            "--pyproject",
            str(py),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "not present in coverage report" in result.stderr


def test_cli_exit_two_when_coverage_json_missing(tmp_path: Path):
    py = tmp_path / "pyproject.toml"
    py.write_text(
        '[tool.coverage.floors]\n"stepback/canonical.py" = 50\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--coverage-json",
            str(tmp_path / "missing.json"),
            "--pyproject",
            str(py),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "not found" in result.stderr


def test_cli_exit_two_on_invalid_floor_value(tmp_path: Path):
    cov = tmp_path / "coverage.json"
    py = tmp_path / "pyproject.toml"
    py.write_text(
        '[tool.coverage.floors]\n"stepback/canonical.py" = 150\n',
        encoding="utf-8",
    )
    _write_coverage_json(cov, {"stepback/canonical.py": 50.0})
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--coverage-json",
            str(cov),
            "--pyproject",
            str(py),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "in [0, 100]" in result.stderr


def test_dev_extra_includes_coverage():
    """Dev extra must install ``coverage`` so floors can be measured."""
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]
    with PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    dev = data["project"]["optional-dependencies"]["dev"]
    assert any("coverage" in entry for entry in dev), (
        "Step 26 of 100_STEPS.md: `coverage` must be in the dev extra so "
        "`pip install -e .[dev]` is sufficient to enforce coverage floors."
    )


def test_coverage_run_section_targets_stepback():
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]
    with PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    run = data["tool"]["coverage"]["run"]
    assert "stepback" in run["source"]
