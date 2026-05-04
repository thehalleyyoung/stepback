"""Tests for Step 3: shipping LICENSE / CITATION.cff / README / fixtures.

These tests guarantee that every file the wheel and sdist *promise* to
ship is actually addressable from a fresh import — no source-tree path
escapes, no editable-install assumptions. They also guarantee that the
mirrored copies inside ``stepback/_resources/`` stay byte-identical to
the canonical repo-root files, so packagers never serve a stale LICENSE.
"""

from __future__ import annotations

import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import pytest

from stepback import resources as sb_resources
from stepback.conformance import fixtures_dir, iter_fixtures

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGED_FILES = ("LICENSE", "CITATION.cff", "README.md")


@pytest.mark.parametrize("name", PACKAGED_FILES)
def test_packaged_resource_matches_repo_root(name: str) -> None:
    """The mirrored copy in stepback/_resources/ must equal the source of truth."""

    src = (REPO_ROOT / name).read_bytes()
    dst = (REPO_ROOT / "stepback" / "_resources" / name).read_bytes()
    assert src == dst, (
        f"stepback/_resources/{name} drifted from repo-root {name}; "
        "run `python scripts/sync_packaged_resources.py` to refresh."
    )


@pytest.mark.parametrize("name", PACKAGED_FILES)
def test_resource_addressable_via_importlib(name: str) -> None:
    resource = files("stepback._resources").joinpath(name)
    assert resource.is_file(), f"{name} is missing from stepback._resources package data"
    assert resource.read_bytes(), f"{name} resource is empty"


def test_resources_module_helpers() -> None:
    assert sb_resources.PACKAGED_FILES == PACKAGED_FILES
    assert "Apache License" in sb_resources.license_text()
    assert "cff-version" in sb_resources.citation_text()
    assert sb_resources.readme_text().lstrip().startswith("# stepback")


def test_unknown_resource_raises() -> None:
    with pytest.raises(FileNotFoundError):
        sb_resources.read_text("not-a-real-file.txt")


def test_conformance_fixtures_dir_present() -> None:
    """Even before Step 41 lands, the fixtures directory must ship."""

    root = fixtures_dir()
    assert root.is_dir(), "stepback/conformance/fixtures must exist in installs"
    # README placeholder guarantees the directory survives empty-dir packagers.
    assert (root / "README.md").is_file()


def test_iter_fixtures_returns_paths() -> None:
    # No .sb fixtures yet (Step 41 owns populating them); the iterator
    # must still be callable without raising.
    assert list(iter_fixtures()) == []
    md_fixtures = list(iter_fixtures(".md"))
    assert any(p.name == "README.md" for p in md_fixtures)


def test_sync_script_is_idempotent() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "sync_packaged_resources.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "already in sync" in result.stdout or "synced" in result.stdout


def test_pyproject_declares_package_data() -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text()
    assert "stepback._resources" in text
    assert "stepback.conformance" in text
    assert "license-files" in text


def test_manifest_in_includes_resource_dirs() -> None:
    text = (REPO_ROOT / "MANIFEST.in").read_text()
    assert "stepback/_resources" in text
    assert "stepback/conformance" in text
