"""Tests for packaging/homebrew/bump_formulae.py.

These tests exercise the template rendering logic offline — no network calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add the packaging/homebrew directory to sys.path so we can import directly.
_HOMEBREW_DIR = Path(__file__).parent.parent / "packaging" / "homebrew"
sys.path.insert(0, str(_HOMEBREW_DIR))

import bump_formulae  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_ARGS = {
    "--version": "1.2.3",
    "--sdist-url": "https://example.com/stepback-1.2.3.tar.gz",
    "--sdist-sha256": "aabbcc" * 8 + "00",
    "--cryptography-url": "https://example.com/cryptography-42.0.0.tar.gz",
    "--cryptography-sha256": "ddeeff" * 8 + "11",
    "--macos-arm64-url": "https://example.com/sb-1.2.3-aarch64-apple-darwin.tar.gz",
    "--macos-arm64-sha256": "112233" * 8 + "22",
    "--macos-x86-64-url": "https://example.com/sb-1.2.3-x86_64-apple-darwin.tar.gz",
    "--macos-x86-64-sha256": "445566" * 8 + "33",
    "--linux-aarch64-url": "https://example.com/sb-1.2.3-aarch64-unknown-linux-gnu.tar.gz",
    "--linux-aarch64-sha256": "778899" * 8 + "44",
    "--linux-x86-64-url": "https://example.com/sb-1.2.3-x86_64-unknown-linux-gnu.tar.gz",
    "--linux-x86-64-sha256": "aabbcc" * 8 + "55",
    "--wasm-viewer-url": "https://example.com/stepback-viewer-1.2.3.html",
    "--wasm-viewer-sha256": "ddeeff" * 8 + "66",
}


def _argv(tmp_path: Path, overrides: dict[str, str] | None = None) -> list[str]:
    args = dict(_FAKE_ARGS)
    if overrides:
        args.update(overrides)
    flat: list[str] = []
    for k, v in args.items():
        flat.extend([k, v])
    flat.extend(["--output-dir", str(tmp_path)])
    return flat


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

class TestTemplateLoading:
    def test_all_templates_exist(self) -> None:
        for tpl_name in bump_formulae._TEMPLATES.values():
            path = _HOMEBREW_DIR / tpl_name
            assert path.exists(), f"Missing template: {path}"

    def test_templates_are_valid_utf8(self) -> None:
        for tpl_name in bump_formulae._TEMPLATES.values():
            text = bump_formulae._load_template(tpl_name)
            assert isinstance(text, str)
            assert len(text) > 0

    def test_missing_template_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            bump_formulae._load_template("nonexistent.rb.tpl")


# ---------------------------------------------------------------------------
# Render helper
# ---------------------------------------------------------------------------

class TestRender:
    def test_basic_replacement(self) -> None:
        result = bump_formulae._render("hello {{NAME}}", {"NAME": "world"})
        assert result == "hello world"

    def test_multiple_replacements(self) -> None:
        result = bump_formulae._render("{{A}} + {{B}}", {"A": "foo", "B": "bar"})
        assert result == "foo + bar"

    def test_missing_placeholder_raises(self) -> None:
        with pytest.raises(ValueError, match="MISSING"):
            bump_formulae._render("no placeholder here", {"MISSING": "value"})

    def test_unreplaced_placeholder_warns(self, capsys) -> None:
        # If the template still has placeholders after substitution, a warning
        # is printed to stderr.
        result = bump_formulae._render("{{A}} {{B}}", {"A": "ok"})
        captured = capsys.readouterr()
        assert "B" in captured.err
        assert result == "ok {{B}}"


# ---------------------------------------------------------------------------
# Full render (offline — no network)
# ---------------------------------------------------------------------------

class TestFullRender:
    def test_render_produces_three_files(self, tmp_path: Path) -> None:
        rc = bump_formulae.main(_argv(tmp_path))
        assert rc == 0
        assert (tmp_path / "stepback.rb").exists()
        assert (tmp_path / "stepback-core.rb").exists()
        assert (tmp_path / "stepback-wasm-viewer.rb").exists()

    def test_stepback_rb_contains_version(self, tmp_path: Path) -> None:
        bump_formulae.main(_argv(tmp_path))
        content = (tmp_path / "stepback.rb").read_text()
        assert "1.2.3" in content

    def test_stepback_rb_contains_sdist_url(self, tmp_path: Path) -> None:
        bump_formulae.main(_argv(tmp_path))
        content = (tmp_path / "stepback.rb").read_text()
        assert "stepback-1.2.3.tar.gz" in content

    def test_stepback_rb_contains_sdist_sha256(self, tmp_path: Path) -> None:
        bump_formulae.main(_argv(tmp_path))
        content = (tmp_path / "stepback.rb").read_text()
        sha = _FAKE_ARGS["--sdist-sha256"]
        assert sha in content

    def test_core_rb_contains_all_platform_urls(self, tmp_path: Path) -> None:
        bump_formulae.main(_argv(tmp_path))
        content = (tmp_path / "stepback-core.rb").read_text()
        assert "aarch64-apple-darwin" in content
        assert "x86_64-apple-darwin" in content
        assert "aarch64-unknown-linux-gnu" in content
        assert "x86_64-unknown-linux-gnu" in content

    def test_core_rb_contains_all_platform_shas(self, tmp_path: Path) -> None:
        bump_formulae.main(_argv(tmp_path))
        content = (tmp_path / "stepback-core.rb").read_text()
        for key in (
            "--macos-arm64-sha256",
            "--macos-x86-64-sha256",
            "--linux-aarch64-sha256",
            "--linux-x86-64-sha256",
        ):
            assert _FAKE_ARGS[key] in content, f"Missing sha for {key}"

    def test_wasm_rb_contains_viewer_url(self, tmp_path: Path) -> None:
        bump_formulae.main(_argv(tmp_path))
        content = (tmp_path / "stepback-wasm-viewer.rb").read_text()
        assert "stepback-viewer-1.2.3.html" in content

    def test_wasm_rb_contains_viewer_sha(self, tmp_path: Path) -> None:
        bump_formulae.main(_argv(tmp_path))
        content = (tmp_path / "stepback-wasm-viewer.rb").read_text()
        assert _FAKE_ARGS["--wasm-viewer-sha256"] in content

    def test_no_unreplaced_placeholders_in_stepback_rb(self, tmp_path: Path) -> None:
        import re
        bump_formulae.main(_argv(tmp_path))
        content = (tmp_path / "stepback.rb").read_text()
        remaining = re.findall(r"\{\{[A-Z0-9_]+\}\}", content)
        assert remaining == [], f"Unreplaced placeholders in stepback.rb: {remaining}"

    def test_no_unreplaced_placeholders_in_core_rb(self, tmp_path: Path) -> None:
        import re
        bump_formulae.main(_argv(tmp_path))
        content = (tmp_path / "stepback-core.rb").read_text()
        remaining = re.findall(r"\{\{[A-Z0-9_]+\}\}", content)
        assert remaining == [], f"Unreplaced placeholders in stepback-core.rb: {remaining}"

    def test_no_unreplaced_placeholders_in_wasm_rb(self, tmp_path: Path) -> None:
        import re
        bump_formulae.main(_argv(tmp_path))
        content = (tmp_path / "stepback-wasm-viewer.rb").read_text()
        remaining = re.findall(r"\{\{[A-Z0-9_]+\}\}", content)
        assert remaining == [], f"Unreplaced placeholders in stepback-wasm-viewer.rb: {remaining}"

    def test_output_dir_created_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "dir"
        assert not nested.exists()
        bump_formulae.main(_argv(nested))
        assert nested.exists()
        assert (nested / "stepback.rb").exists()

    def test_bottle_block_injected(self, tmp_path: Path) -> None:
        bump_formulae.main(_argv(tmp_path, {"--bottle-block": "sha256 cellar: :any, arm64_sequoia: \"deadbeef\""}))
        content = (tmp_path / "stepback.rb").read_text()
        assert "deadbeef" in content

    def test_missing_sha_without_download_flag_raises(self, tmp_path: Path) -> None:
        """Without --download-missing-shas, omitting a sha should raise ValueError."""
        argv = _argv(tmp_path)
        # Remove --sdist-sha256 value
        idx = argv.index("--sdist-sha256")
        argv[idx + 1] = ""  # empty string = "not provided"
        with pytest.raises(ValueError, match="download-missing-shas"):
            bump_formulae.main(argv)
