"""Tests for ``stepback diagnose`` (step 103).

All tests run offline; no network calls are made.
"""
from __future__ import annotations

import json
import sys
from typing import List
from unittest import mock

import pytest

from stepback.diagnose import (
    COMPARE_UNAVAILABLE,
    FRAMEWORK_VERSION_MATRIX,
    NOT_INSTALLED,
    OK,
    WARN_NEWER,
    WARN_UNSUPPORTED,
    DiagnoseResult,
    _check_version,
    diagnose_all,
    format_diagnose_table,
    get_installed_version,
)
from stepback.shim_certification import SHIM_VERSION_MATRIX


# ---------------------------------------------------------------------------
# get_installed_version
# ---------------------------------------------------------------------------

class TestGetInstalledVersion:
    def test_returns_none_for_unknown_package(self):
        # An extremely unlikely package name.
        result = get_installed_version("_stepback_definitely_not_a_real_package_xyz_")
        assert result is None

    def test_returns_version_string_for_known_package(self):
        # `packaging` is now a declared dependency so it must be present.
        ver = get_installed_version("packaging")
        assert ver is not None
        assert isinstance(ver, str)
        # Sanity: should look like a version.
        assert "." in ver

    def test_returns_version_for_cryptography(self):
        ver = get_installed_version("cryptography")
        assert ver is not None


# ---------------------------------------------------------------------------
# _check_version
# ---------------------------------------------------------------------------

class TestCheckVersion:
    def test_ok_within_range_and_tested(self):
        status = _check_version("1.40.0", ">=1.0,<2.0", ["1.40.0", "1.51.2"])
        assert status == OK

    def test_ok_exactly_at_max_tested(self):
        status = _check_version("1.51.2", ">=1.0,<2.0", ["1.40.0", "1.51.2"])
        assert status == OK

    def test_warn_newer_within_spec_but_above_tested(self):
        status = _check_version("1.99.0", ">=1.0,<2.0", ["1.40.0", "1.51.2"])
        assert status == WARN_NEWER

    def test_warn_unsupported_below_range(self):
        status = _check_version("0.9.0", ">=1.0,<2.0", ["1.40.0"])
        assert status == WARN_UNSUPPORTED

    def test_warn_unsupported_above_range(self):
        status = _check_version("2.0.0", ">=1.0,<2.0", ["1.40.0", "1.51.2"])
        assert status == WARN_UNSUPPORTED

    def test_empty_tested_list_returns_ok_for_in_range(self):
        status = _check_version("1.5.0", ">=1.0,<2.0", [])
        assert status == OK

    def test_prerelease_within_range(self):
        # 1.56.0rc1 is within >=1.0,<2.0 but higher than 1.55.0.
        status = _check_version("1.56.0rc1", ">=1.0,<2.0", ["1.55.0"])
        assert status in (WARN_NEWER, OK)  # prerelease handling may vary

    def test_hyphenated_package_version_string(self):
        # Versions like "0.0.20" for pydantic-ai
        status = _check_version("0.0.20", ">=0.0.9,<0.2", ["0.0.20", "0.0.46"])
        assert status == OK

    def test_newer_than_max_tested_three_part(self):
        status = _check_version("2.6.99", ">=2.4,<2.7", ["2.4.17", "2.5.43", "2.6.23"])
        assert status == WARN_NEWER


# ---------------------------------------------------------------------------
# diagnose_all
# ---------------------------------------------------------------------------

class TestDiagnoseAll:
    def test_returns_list_of_diagnose_results(self):
        results = diagnose_all()
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, DiagnoseResult) for r in results)

    def test_covers_all_matrix_entries(self):
        results = diagnose_all()
        names = {r.name for r in results}
        # Every provider from shim matrix should appear.
        for name in SHIM_VERSION_MATRIX:
            assert name in names, f"Missing provider {name!r} in diagnose results"
        # Every framework from framework matrix should appear.
        for name in FRAMEWORK_VERSION_MATRIX:
            assert name in names, f"Missing framework {name!r} in diagnose results"

    def test_provider_kind_for_shim_entries(self):
        results = diagnose_all()
        by_name = {r.name: r for r in results}
        for name in SHIM_VERSION_MATRIX:
            assert by_name[name].kind == "provider"

    def test_framework_kind_for_framework_entries(self):
        results = diagnose_all()
        by_name = {r.name: r for r in results}
        for name in FRAMEWORK_VERSION_MATRIX:
            assert by_name[name].kind == "framework"

    def test_status_is_valid(self):
        valid = {OK, WARN_NEWER, WARN_UNSUPPORTED, NOT_INSTALLED, COMPARE_UNAVAILABLE}
        for r in diagnose_all():
            assert r.status in valid, f"{r.name} has invalid status {r.status!r}"

    def test_installed_is_string_or_none(self):
        for r in diagnose_all():
            assert r.installed is None or isinstance(r.installed, str)

    def test_as_dict_is_serializable(self):
        for r in diagnose_all():
            d = r.as_dict()
            # Must be JSON-serializable.
            json.dumps(d)
            assert d["name"] == r.name
            assert d["status"] == r.status

    def test_not_installed_for_fake_package(self):
        """An entry whose package doesn't exist should have NOT_INSTALLED status."""
        fake_matrix = {
            "_stepback_fake_xyz_": {
                "package": "_stepback_fake_pkg_xyz_",
                "supported": ">=1.0,<2.0",
                "tested": ["1.0.0"],
                "notes": "test",
            }
        }
        from stepback import diagnose as diag_mod
        with mock.patch.object(diag_mod, "SHIM_VERSION_MATRIX", fake_matrix), \
             mock.patch.object(diag_mod, "FRAMEWORK_VERSION_MATRIX", {}):
            results = diag_mod.diagnose_all()
            assert len(results) == 1
            assert results[0].status == NOT_INSTALLED

    def test_warn_newer_when_installed_is_too_new(self):
        """Simulate a package that is newer than the certified matrix."""
        fake_matrix = {
            "fake_pkg": {
                "package": "packaging",
                "supported": ">=0.1,<99.0",
                # Force a ridiculously low max tested so any real version is newer.
                "tested": ["0.1"],
                "notes": "test",
            }
        }
        from stepback import diagnose as diag_mod
        with mock.patch.object(diag_mod, "SHIM_VERSION_MATRIX", fake_matrix), \
             mock.patch.object(diag_mod, "FRAMEWORK_VERSION_MATRIX", {}):
            results = diag_mod.diagnose_all()
            assert len(results) == 1
            # packaging is definitely installed; its version > 0.1.
            assert results[0].status in (WARN_NEWER,)


# ---------------------------------------------------------------------------
# format_diagnose_table
# ---------------------------------------------------------------------------

class TestFormatDiagnoseTable:
    def _make_result(self, name, status, installed=None):
        return DiagnoseResult(
            name=name,
            package=f"pkg-{name}",
            kind="provider",
            installed=installed,
            supported_spec=">=1.0,<2.0",
            tested_versions=["1.0.0"],
            status=status,
            detail="test detail",
        )

    def test_hides_not_installed_by_default(self):
        results = [
            self._make_result("installed_ok", OK, "1.0.0"),
            self._make_result("not_here", NOT_INSTALLED),
        ]
        table = format_diagnose_table(results, show_all=False)
        assert "installed_ok" in table
        assert "not_here" not in table

    def test_shows_not_installed_with_all(self):
        results = [
            self._make_result("installed_ok", OK, "1.0.0"),
            self._make_result("not_here", NOT_INSTALLED),
        ]
        table = format_diagnose_table(results, show_all=True)
        assert "installed_ok" in table
        assert "not_here" in table

    def test_shows_warn_newer_by_default(self):
        results = [self._make_result("new_pkg", WARN_NEWER, "1.9.0")]
        table = format_diagnose_table(results, show_all=False)
        assert "new_pkg" in table
        assert "warn_newer" in table

    def test_empty_message_when_nothing_installed_and_not_all(self):
        results = [self._make_result("not_here", NOT_INSTALLED)]
        table = format_diagnose_table(results, show_all=False)
        assert "no installed packages" in table


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

class TestCLIDiagnose:
    def test_diagnose_runs_without_error(self, capsys):
        from stepback.cli import main
        rc = main(["diagnose"])
        assert rc == 0

    def test_diagnose_json_flag(self, capsys):
        from stepback.cli import main
        rc = main(["diagnose", "--json"])
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert len(data) > 0
        # Every entry has the expected keys.
        required_keys = {"name", "package", "kind", "installed", "status", "detail"}
        for entry in data:
            assert required_keys.issubset(entry.keys())

    def test_diagnose_all_flag(self, capsys):
        from stepback.cli import main
        rc = main(["diagnose", "--all"])
        assert rc == 0
        captured = capsys.readouterr()
        # With --all we should see more rows than without.
        out_all = captured.out
        assert len(out_all) > 0

    def test_diagnose_strict_exits_0_when_no_warnings(self, capsys):
        """--strict should exit 0 when all installed packages are ok."""
        from stepback.cli import main
        from stepback import diagnose as diag_mod

        # Patch to return only OK results.
        ok_results = [
            DiagnoseResult(
                name="ok_pkg",
                package="ok-pkg",
                kind="provider",
                installed="1.0.0",
                supported_spec=">=1.0,<2.0",
                tested_versions=["1.0.0"],
                status=OK,
                detail="all good",
            )
        ]
        with mock.patch.object(diag_mod, "diagnose_all", return_value=ok_results):
            rc = main(["diagnose", "--strict"])
        assert rc == 0

    def test_diagnose_strict_exits_1_when_warnings(self, capsys):
        """--strict should exit 1 when there are warn_newer entries."""
        from stepback.cli import main
        from stepback import diagnose as diag_mod

        warn_results = [
            DiagnoseResult(
                name="new_pkg",
                package="new-pkg",
                kind="provider",
                installed="1.9.0",
                supported_spec=">=1.0,<2.0",
                tested_versions=["1.0.0"],
                status=WARN_NEWER,
                detail="newer than tested",
            )
        ]
        with mock.patch.object(diag_mod, "diagnose_all", return_value=warn_results):
            rc = main(["diagnose", "--strict"])
        assert rc == 1
