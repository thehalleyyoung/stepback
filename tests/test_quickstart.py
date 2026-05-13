"""Tests for ``stepback quickstart`` wizard and ``stepback quickstart`` CLI command."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stepback.quickstart import (
    PROVIDERS,
    QuickstartResult,
    _PROVIDER_ORDER,
    _export_html,
    _record_demo_trace,
    retrieve_api_key,
    run_wizard,
    store_api_key,
)


# ─── unit: provider registry ─────────────────────────────────────────────────


class TestProviderRegistry:
    def test_all_expected_providers_present(self):
        assert set(PROVIDERS) == {"openai", "anthropic", "bedrock", "gemini", "demo"}

    def test_provider_order_contains_all_providers(self):
        assert set(_PROVIDER_ORDER) == set(PROVIDERS)

    def test_demo_provider_has_no_env_var(self):
        assert PROVIDERS["demo"].env_var == ""

    def test_demo_provider_has_no_keyring_service(self):
        assert PROVIDERS["demo"].keyring_service == ""

    def test_each_non_demo_provider_has_env_var(self):
        for name, info in PROVIDERS.items():
            if name != "demo":
                assert info.env_var, f"{name} should have env_var set"

    def test_each_non_demo_provider_has_keyring_service(self):
        for name, info in PROVIDERS.items():
            if name != "demo":
                assert info.keyring_service, f"{name} should have keyring_service set"

    def test_provider_label_not_empty(self):
        for name, info in PROVIDERS.items():
            assert info.label, f"{name} label should not be empty"


# ─── unit: keyring helpers ────────────────────────────────────────────────────


class TestKeyringHelpers:
    def test_store_returns_false_when_keyring_unavailable(self):
        """store_api_key returns False when keyring module is missing."""
        provider = PROVIDERS["openai"]
        with patch("stepback.quickstart._HAS_KEYRING", False):
            result = store_api_key(provider, "sk-test")
        assert result is False

    def test_store_returns_false_for_demo_provider(self):
        provider = PROVIDERS["demo"]
        result = store_api_key(provider, "anything")
        assert result is False

    def test_store_returns_false_for_empty_key(self):
        provider = PROVIDERS["openai"]
        result = store_api_key(provider, "")
        assert result is False

    def test_retrieve_returns_none_when_keyring_unavailable(self):
        provider = PROVIDERS["openai"]
        with patch("stepback.quickstart._HAS_KEYRING", False):
            result = retrieve_api_key(provider)
        assert result is None

    def test_retrieve_returns_none_for_demo_provider(self):
        provider = PROVIDERS["demo"]
        result = retrieve_api_key(provider)
        assert result is None

    def test_store_and_retrieve_round_trip_when_keyring_available(self):
        """Round-trip via a mocked keyring."""
        fake_store: dict = {}

        def fake_set_password(service, username, password):
            fake_store[(service, username)] = password

        def fake_get_password(service, username):
            return fake_store.get((service, username))

        mock_kr = MagicMock()
        mock_kr.set_password.side_effect = fake_set_password
        mock_kr.get_password.side_effect = fake_get_password

        provider = PROVIDERS["openai"]
        with (
            patch("stepback.quickstart._HAS_KEYRING", True),
            patch("stepback.quickstart.keyring", mock_kr, create=True),
            patch.dict("sys.modules", {"keyring": mock_kr}),
        ):
            stored = store_api_key(provider, "sk-roundtrip")
            assert stored is True
            retrieved = retrieve_api_key(provider)
            assert retrieved == "sk-roundtrip"


# ─── unit: demo trace recording ──────────────────────────────────────────────


class TestRecordDemoTrace:
    def test_creates_sb_file(self, tmp_path):
        trace_path = tmp_path / "test.sb"
        _record_demo_trace(trace_path)
        assert trace_path.exists()
        assert trace_path.stat().st_size > 0

    def test_creates_parent_dirs(self, tmp_path):
        trace_path = tmp_path / "a" / "b" / "c" / "test.sb"
        _record_demo_trace(trace_path)
        assert trace_path.exists()


# ─── unit: HTML export ────────────────────────────────────────────────────────


class TestExportHtml:
    def test_creates_html_file(self, tmp_path):
        trace_path = tmp_path / "test.sb"
        _record_demo_trace(trace_path)
        html_path = tmp_path / "test.html"
        _export_html(trace_path, html_path)
        assert html_path.exists()
        content = html_path.read_text(encoding="utf-8")
        assert "<html" in content.lower() or "<!DOCTYPE" in content.lower() or "<div" in content.lower()


# ─── integration: run_wizard non-interactive ─────────────────────────────────


class TestRunWizardNonInteractive:
    def test_demo_provider_returns_result(self, tmp_path):
        result = run_wizard(
            non_interactive=True,
            provider_name="demo",
            output_dir=tmp_path,
            skip_open=True,
        )
        assert isinstance(result, QuickstartResult)
        assert result.provider == "demo"
        assert result.trace_path.exists()
        assert result.html_path.exists()
        assert not result.opened_browser

    def test_default_provider_is_demo_in_non_interactive(self, tmp_path):
        result = run_wizard(
            non_interactive=True,
            output_dir=tmp_path,
            skip_open=True,
        )
        assert result.provider == "demo"

    def test_trace_path_in_output_dir(self, tmp_path):
        result = run_wizard(
            non_interactive=True,
            output_dir=tmp_path,
            skip_open=True,
        )
        assert result.trace_path.parent == tmp_path

    def test_html_path_in_output_dir(self, tmp_path):
        result = run_wizard(
            non_interactive=True,
            output_dir=tmp_path,
            skip_open=True,
        )
        assert result.html_path.parent == tmp_path

    def test_uses_temp_dir_when_no_output_dir(self):
        result = run_wizard(
            non_interactive=True,
            skip_open=True,
        )
        assert result.trace_path.exists()
        assert result.html_path.exists()

    def test_unknown_provider_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown provider"):
            run_wizard(
                non_interactive=True,
                provider_name="bogus",
                output_dir=tmp_path,
                skip_open=True,
            )

    def test_api_key_stored_false_for_demo(self, tmp_path):
        result = run_wizard(
            non_interactive=True,
            provider_name="demo",
            output_dir=tmp_path,
            skip_open=True,
        )
        assert result.api_key_stored is False

    def test_no_errors_for_demo(self, tmp_path):
        result = run_wizard(
            non_interactive=True,
            provider_name="demo",
            output_dir=tmp_path,
            skip_open=True,
        )
        assert result.errors == []

    def test_env_var_api_key_picked_up(self, tmp_path):
        """If OPENAI_API_KEY is set but we're in non-interactive mode without
        a real SDK, the wizard should fall back to demo gracefully."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-fake-key-for-test"}):
            # Without the real openai SDK, _record_openai_trace will fail;
            # the wizard should catch the error and fall back to demo.
            result = run_wizard(
                non_interactive=True,
                provider_name="openai",
                output_dir=tmp_path,
                skip_open=True,
            )
        # Whether it succeeded with openai or fell back to demo, a trace must exist.
        assert result.trace_path.exists()


# ─── integration: CLI subcommand ─────────────────────────────────────────────


class TestQuickstartCLI:
    def test_cli_non_interactive_demo(self, tmp_path):
        from stepback.cli import main as cli_main

        exit_code = cli_main(
            [
                "quickstart",
                "--non-interactive",
                "--provider", "demo",
                "--output-dir", str(tmp_path),
                "--skip-open",
            ]
        )
        assert exit_code == 0
        assert (tmp_path / "quickstart.sb").exists()
        assert (tmp_path / "quickstart.html").exists()

    def test_cli_help_exits_zero(self):
        from stepback.cli import main as cli_main

        with pytest.raises(SystemExit) as exc_info:
            cli_main(["quickstart", "--help"])
        assert exc_info.value.code == 0

    def test_cli_unknown_provider_exits_nonzero(self, tmp_path):
        """argparse choice validation rejects unknown providers."""
        from stepback.cli import main as cli_main

        with pytest.raises(SystemExit) as exc_info:
            cli_main(
                [
                    "quickstart",
                    "--non-interactive",
                    "--provider", "nonexistent_provider",
                    "--output-dir", str(tmp_path),
                    "--skip-open",
                ]
            )
        assert exc_info.value.code != 0
