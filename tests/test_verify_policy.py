"""Tests for stepback.verify_policy — policy-gated trace verification.

Covers:
* VerifyPolicy.from_dict / strict_default
* load_policy from JSON file
* verify_with_policy: happy path
* verify_with_policy: crypto failure
* verify_with_policy: canonical bytes failure
* verify_with_policy: recorder identity (key allowlist, version allowlist)
* verify_with_policy: schema check
* strict mode promotes warnings to errors
* CLI: stepback verify --strict
* CLI: stepback verify --policy <file>
* CLI: legacy verify still works (no regressions)
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys

import pytest

from stepback import RecorderKey, record
from stepback.canonical import canonical_json
from stepback.testing import run_recorded_agent
from stepback.verify_policy import (
    PolicyCheckResult,
    PolicyViolation,
    VerifyPolicy,
    load_policy,
    verify_with_policy,
)


# ------------------------------------------------------------------ helpers


def _record_small(tmp_path) -> tuple[str, RecorderKey]:
    """Record a 12-step fixture trace; return (path, key)."""
    key = RecorderKey.fresh()
    path = str(tmp_path / "trace.sb")
    with record(path, key=key) as rec:
        run_recorded_agent(rec)
    return path, key


# ------------------------------------------------------------------ unit tests


class TestVerifyPolicyFromDict:
    def test_defaults(self):
        p = VerifyPolicy.from_dict({})
        assert p.strict is False
        assert p.crypto.enabled is True
        assert p.schema.enabled is True
        assert p.canonical_bytes.enabled is True
        assert p.recorder_identity.enabled is True
        assert p.recorder_identity.allowed_public_keys == []
        assert p.recorder_identity.allowed_recorder_versions == []

    def test_strict_flag(self):
        p = VerifyPolicy.from_dict({"strict": True})
        assert p.strict is True

    def test_disable_checks(self):
        p = VerifyPolicy.from_dict({
            "crypto": {"enabled": False},
            "schema": {"enabled": False},
            "canonical_bytes": {"enabled": False},
            "recorder_identity": {"enabled": False},
        })
        assert p.crypto.enabled is False
        assert p.schema.enabled is False
        assert p.canonical_bytes.enabled is False
        assert p.recorder_identity.enabled is False

    def test_allowed_keys_lowercased(self):
        p = VerifyPolicy.from_dict({
            "recorder_identity": {
                "allowed_public_keys": ["ABCD1234", "ef567890"],
            }
        })
        assert "abcd1234" in p.recorder_identity.allowed_public_keys
        assert "ef567890" in p.recorder_identity.allowed_public_keys

    def test_allowed_versions(self):
        p = VerifyPolicy.from_dict({
            "recorder_identity": {
                "allowed_recorder_versions": ["0.1.0", "0.2.0"],
            }
        })
        assert "0.1.0" in p.recorder_identity.allowed_recorder_versions

    def test_unknown_top_level_keys_ignored(self):
        p = VerifyPolicy.from_dict({"unknown_future_key": 42})
        assert p.strict is False  # no crash

    def test_strict_default_factory(self):
        p = VerifyPolicy.strict_default()
        assert p.strict is True
        assert p.crypto.enabled is True
        assert p.schema.enabled is True
        assert p.canonical_bytes.enabled is True


class TestLoadPolicy:
    def test_load_valid_json(self, tmp_path):
        data = {"strict": True, "crypto": {"enabled": True}}
        pol_path = tmp_path / "policy.json"
        pol_path.write_text(json.dumps(data))
        p = load_policy(str(pol_path))
        assert p.strict is True

    def test_load_invalid_json_raises(self, tmp_path):
        pol_path = tmp_path / "bad.json"
        pol_path.write_text("{not json}")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_policy(str(pol_path))

    def test_load_non_object_raises(self, tmp_path):
        pol_path = tmp_path / "array.json"
        pol_path.write_text("[1, 2, 3]")
        with pytest.raises(ValueError, match="must be a JSON object"):
            load_policy(str(pol_path))

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises((ValueError, OSError)):
            load_policy(str(tmp_path / "nonexistent.json"))


class TestVerifyWithPolicyHappyPath:
    def test_all_checks_pass(self, tmp_path):
        path, key = _record_small(tmp_path)
        policy = VerifyPolicy.strict_default()
        result = verify_with_policy(path, key.hmac_key, policy)
        assert result.ok, [str(v) for v in result.violations]
        assert result.trace is not None
        assert len(result.trace.steps) == 12

    def test_no_violations_means_empty_errors(self, tmp_path):
        path, key = _record_small(tmp_path)
        result = verify_with_policy(path, key.hmac_key, VerifyPolicy())
        assert result.errors == []


class TestVerifyWithPolicyCrypto:
    def test_wrong_key_fails_crypto_check(self, tmp_path):
        path, key = _record_small(tmp_path)
        bad_key = RecorderKey.fresh()
        result = verify_with_policy(path, bad_key.hmac_key, VerifyPolicy())
        assert not result.ok
        assert any(v.check == "crypto" for v in result.violations)
        assert result.trace is None

    def test_crypto_disabled_skips_check(self, tmp_path):
        path, key = _record_small(tmp_path)
        bad_key = RecorderKey.fresh()
        policy = VerifyPolicy.from_dict({"crypto": {"enabled": False}})
        result = verify_with_policy(path, bad_key.hmac_key, policy)
        # crypto not run; schema/canonical/identity may still fail but not crypto
        assert not any(v.check == "crypto" for v in result.violations)


class TestVerifyWithPolicyCanonicalBytes:
    def test_canonical_trace_passes(self, tmp_path):
        path, key = _record_small(tmp_path)
        policy = VerifyPolicy.from_dict({"canonical_bytes": {"enabled": True}})
        result = verify_with_policy(path, key.hmac_key, policy)
        canonical_violations = [v for v in result.violations if v.check == "canonical_bytes"]
        assert canonical_violations == [], canonical_violations

    def test_non_canonical_frame_detected(self, tmp_path):
        """Inject non-canonical whitespace into a frame and verify detection."""
        path, key = _record_small(tmp_path)
        # Read raw bytes
        raw = open(path, "rb").read()
        # Corrupt the first frame: inject whitespace into its JSON body
        offset = 4
        (n,) = struct.unpack(">I", raw[:4])
        payload = raw[offset : offset + n]
        original_decoded = json.loads(payload)
        # Re-encode with non-canonical pretty-print
        non_canonical = json.dumps(original_decoded, indent=2).encode("utf-8")
        if non_canonical == payload:
            pytest.skip("payload happens to be already non-canonical (unlikely)")
        new_len = struct.pack(">I", len(non_canonical))
        corrupt = new_len + non_canonical + raw[offset + n :]
        bad_path = str(tmp_path / "bad_canonical.sb")
        open(bad_path, "wb").write(corrupt)
        policy = VerifyPolicy.from_dict({
            "crypto": {"enabled": False},
            "schema": {"enabled": False},
            "recorder_identity": {"enabled": False},
            "canonical_bytes": {"enabled": True},
        })
        result = verify_with_policy(bad_path, key.hmac_key, policy)
        assert any(v.check == "canonical_bytes" for v in result.violations)


class TestVerifyWithPolicyRecorderIdentity:
    def test_correct_key_in_allowlist_passes(self, tmp_path):
        path, key = _record_small(tmp_path)
        from stepback.trace_reader import verify_trace
        t = verify_trace(path, key.hmac_key)
        pubkey = t.public_key_hex
        policy = VerifyPolicy.from_dict({
            "recorder_identity": {
                "enabled": True,
                "allowed_public_keys": [pubkey],
            }
        })
        result = verify_with_policy(path, key.hmac_key, policy)
        id_violations = [v for v in result.violations if v.check == "recorder_identity"]
        assert id_violations == []

    def test_wrong_key_in_allowlist_fails(self, tmp_path):
        path, key = _record_small(tmp_path)
        wrong_key = "aabbccdd" * 8  # 64-char hex, won't match
        policy = VerifyPolicy.from_dict({
            "recorder_identity": {
                "enabled": True,
                "allowed_public_keys": [wrong_key],
            }
        })
        result = verify_with_policy(path, key.hmac_key, policy)
        assert any(v.check == "recorder_identity" for v in result.violations)

    def test_empty_allowlist_means_no_pinning(self, tmp_path):
        path, key = _record_small(tmp_path)
        policy = VerifyPolicy.from_dict({
            "recorder_identity": {
                "enabled": True,
                "allowed_public_keys": [],
            }
        })
        result = verify_with_policy(path, key.hmac_key, policy)
        id_violations = [v for v in result.violations if v.check == "recorder_identity"]
        assert id_violations == []

    def test_correct_recorder_version_passes(self, tmp_path):
        from stepback.trace_writer import RECORDER_VERSION
        path, key = _record_small(tmp_path)
        policy = VerifyPolicy.from_dict({
            "recorder_identity": {
                "enabled": True,
                "allowed_recorder_versions": [RECORDER_VERSION],
            }
        })
        result = verify_with_policy(path, key.hmac_key, policy)
        id_violations = [v for v in result.violations if v.check == "recorder_identity"]
        assert id_violations == []

    def test_wrong_recorder_version_fails(self, tmp_path):
        path, key = _record_small(tmp_path)
        policy = VerifyPolicy.from_dict({
            "recorder_identity": {
                "enabled": True,
                "allowed_recorder_versions": ["99.99.99"],
            }
        })
        result = verify_with_policy(path, key.hmac_key, policy)
        assert any(v.check == "recorder_identity" for v in result.violations)


class TestVerifyWithPolicyStrict:
    def test_strict_promotes_schema_warnings_to_errors(self, tmp_path):
        """In strict mode, warnings from schema check become errors."""
        path, key = _record_small(tmp_path)
        # Run in non-strict mode and collect any schema warnings
        non_strict = VerifyPolicy.from_dict({"strict": False})
        non_strict_result = verify_with_policy(path, key.hmac_key, non_strict)
        warnings_before = [v for v in non_strict_result.violations if v.severity == "warning"]
        if not warnings_before:
            pytest.skip("no schema warnings on this trace; cannot test promotion")
        # In strict mode those same warnings become errors
        strict = VerifyPolicy.strict_default()
        strict_result = verify_with_policy(path, key.hmac_key, strict)
        assert len(strict_result.errors) >= len(warnings_before)

    def test_strict_mode_all_checks_on_clean_trace(self, tmp_path):
        path, key = _record_small(tmp_path)
        policy = VerifyPolicy.strict_default()
        result = verify_with_policy(path, key.hmac_key, policy)
        assert result.ok, [str(v) for v in result.violations]


class TestCLIVerify:
    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "stepback.cli", *args],
            capture_output=True, text=True, check=False,
        )

    def test_legacy_verify_still_works(self, tmp_path):
        path, key = _record_small(tmp_path)
        rc = self._run("verify", path, "--hmac-key-hex", key.hmac_key.hex())
        assert rc.returncode == 0, rc.stderr
        assert rc.stdout.startswith("OK")

    def test_strict_flag_passes_on_valid_trace(self, tmp_path):
        path, key = _record_small(tmp_path)
        rc = self._run(
            "verify", path, "--hmac-key-hex", key.hmac_key.hex(), "--strict"
        )
        assert rc.returncode == 0, rc.stderr
        assert rc.stdout.startswith("OK")

    def test_strict_flag_fails_on_bad_key(self, tmp_path):
        path, key = _record_small(tmp_path)
        bad_key = RecorderKey.fresh()
        rc = self._run(
            "verify", path, "--hmac-key-hex", bad_key.hmac_key.hex(), "--strict"
        )
        assert rc.returncode != 0

    def test_policy_file_passes_on_valid_trace(self, tmp_path):
        path, key = _record_small(tmp_path)
        pol = tmp_path / "policy.json"
        pol.write_text(json.dumps({"strict": True}))
        rc = self._run(
            "verify", path, "--hmac-key-hex", key.hmac_key.hex(),
            "--policy", str(pol),
        )
        assert rc.returncode == 0, rc.stderr
        assert rc.stdout.startswith("OK")

    def test_policy_file_with_key_pinning_passes(self, tmp_path):
        path, key = _record_small(tmp_path)
        from stepback.trace_reader import verify_trace
        t = verify_trace(path, key.hmac_key)
        pol = tmp_path / "policy.json"
        pol.write_text(json.dumps({
            "recorder_identity": {
                "allowed_public_keys": [t.public_key_hex],
            }
        }))
        rc = self._run(
            "verify", path, "--hmac-key-hex", key.hmac_key.hex(),
            "--policy", str(pol),
        )
        assert rc.returncode == 0, rc.stderr

    def test_policy_file_with_wrong_key_pinning_fails(self, tmp_path):
        path, key = _record_small(tmp_path)
        pol = tmp_path / "policy.json"
        pol.write_text(json.dumps({
            "recorder_identity": {
                "allowed_public_keys": ["aabbccdd" * 8],
            }
        }))
        rc = self._run(
            "verify", path, "--hmac-key-hex", key.hmac_key.hex(),
            "--policy", str(pol),
        )
        assert rc.returncode != 0

    def test_missing_policy_file_returns_nonzero(self, tmp_path):
        path, key = _record_small(tmp_path)
        rc = self._run(
            "verify", path, "--hmac-key-hex", key.hmac_key.hex(),
            "--policy", str(tmp_path / "nonexistent.json"),
        )
        assert rc.returncode != 0

    def test_invalid_hmac_key_hex_returns_nonzero(self, tmp_path):
        path, key = _record_small(tmp_path)
        rc = self._run(
            "verify", path, "--hmac-key-hex", "not-hex", "--strict"
        )
        assert rc.returncode != 0
