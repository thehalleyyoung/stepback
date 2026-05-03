"""End-to-end tests for `stepback attest` / regulator-replay attestation packs.

Drives the same deterministic 12-step payments-bot fixture used by
`test_e2e_replay.py` through the new attestation pipeline:

* records THREE traces under different recorder keys
* builds an attestation pack covering all three, with a
  ``ToolOutputSubstitution`` that fixes the bad-IBAN bug applied to
  every trace (a stand-in for the regulator-replay use-case where a
  newer policy version is pinned across a quarter of trace history)
* asserts the pack signature verifies
* asserts the per-trace verdicts make sense (verify_status == "ok",
  divergent_step_count > 0 because the substitution flips the wired
  IBAN)
* asserts a tampered pack body fails verification
* asserts an attestor-key-mismatch is detected
* exercises the CLI: ``stepback attest`` then ``stepback verify-pack``
* asserts a corrupt trace inside the corpus is reported as
  ``verify_status="fail"`` rather than crashing the whole pack build.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from stepback import RecorderKey, record
from stepback.attestation import (
    AttestationVerificationError,
    PACK_FORMAT_VERSION,
    PACK_MAGIC,
    build_attestation_pack,
    read_attestation_pack,
    verify_attestation_pack,
    write_attestation_pack,
)
from stepback.substitutions import ToolOutputSubstitution
from tests.fixtures.agent import LOOKUP_FIXED_ROW, run_recorded_agent


# ----------------------------------------------------------- helpers


def _record_trace(tmp_path, name: str) -> tuple[str, RecorderKey]:
    key = RecorderKey.fresh()
    path = str(tmp_path / f"{name}.sb")
    with record(path, key=key) as rec:
        run_recorded_agent(rec)
    return path, key


def _fix_iban_subs():
    # The fixture's recorded run has a `lookup_customer` tool call at
    # step:2 that returns the wrong customer row. Substituting the
    # FIXED row turns the downstream "wired payment ..." text into a
    # different stable string, so divergent_step_count must be > 0.
    return [
        ToolOutputSubstitution(at_step="step:2", fake_response=LOOKUP_FIXED_ROW),
    ]


# --------------------------------------------------- builder happy path


def test_build_attestation_pack_three_traces(tmp_path):
    paths_and_keys = [_record_trace(tmp_path, f"t{i}") for i in range(3)]
    traces_with_keys = [(p, k.hmac_key) for p, k in paths_and_keys]

    signing = Ed25519PrivateKey.generate()
    pack = build_attestation_pack(
        traces_with_keys,
        substitutions=_fix_iban_subs(),
        policy_version_pin="2026-04-15",
        attestor_signing_key=signing,
    )
    assert pack.policy_version_pin == "2026-04-15"
    assert pack.attestor_public_key.startswith("ed25519:")
    assert len(pack.entries) == 3
    for e in pack.entries:
        assert e.verify_status == "ok"
        assert e.replay_status == "ok"
        assert e.step_count > 0
        # The substitution fixes the bad IBAN — at least the
        # tool-output step itself is dirty + every step downstream
        # of it that depends on the lookup output is divergent.
        assert e.divergent_step_count >= 1
        assert e.recorder_public_key.startswith("ed25519:")
        assert e.trace_chain_hash.startswith("sha256:")
    s = pack.summary
    assert s["trace_count"] == 3
    assert s["verified_ok"] == 3
    assert s["verified_fail"] == 0
    assert s["divergent_traces"] == 3


# --------------------------------------------------- write / verify / tamper


def test_write_and_verify_pack_roundtrip(tmp_path):
    p, k = _record_trace(tmp_path, "trace")
    signing = Ed25519PrivateKey.generate()
    pack = build_attestation_pack(
        [(p, k.hmac_key)],
        substitutions=_fix_iban_subs(),
        attestor_signing_key=signing,
    )
    out = str(tmp_path / "regulator-Q1.pack")
    body_hash = write_attestation_pack(pack, out, signing_key=signing)
    assert body_hash.startswith("sha256:")
    assert os.path.exists(out)

    data = verify_attestation_pack(out)
    assert data["magic"] == PACK_MAGIC
    assert data["format_version"] == PACK_FORMAT_VERSION
    assert data["body_hash"] == body_hash
    assert data["attestor_public_key"] == pack.attestor_public_key
    assert len(data["entries"]) == 1


def test_verify_pack_pinned_attestor_key_mismatch(tmp_path):
    p, k = _record_trace(tmp_path, "trace")
    signing = Ed25519PrivateKey.generate()
    pack = build_attestation_pack(
        [(p, k.hmac_key)],
        substitutions=_fix_iban_subs(),
        attestor_signing_key=signing,
    )
    out = str(tmp_path / "x.pack")
    write_attestation_pack(pack, out, signing_key=signing)
    other = "ed25519:" + ("ab" * 32)
    with pytest.raises(AttestationVerificationError):
        verify_attestation_pack(out, expected_public_key=other)


def test_tampered_pack_body_fails_verification(tmp_path):
    p, k = _record_trace(tmp_path, "trace")
    signing = Ed25519PrivateKey.generate()
    pack = build_attestation_pack(
        [(p, k.hmac_key)],
        attestor_signing_key=signing,
    )
    out = str(tmp_path / "x.pack")
    write_attestation_pack(pack, out, signing_key=signing)

    with open(out) as f:
        raw = json.loads(f.read())
    # Flip the policy_version_pin to a different value (not in the
    # original body) without re-signing — must fail.
    raw["policy_version_pin"] = "tampered-2030-01-01"
    with open(out, "w") as f:
        json.dump(raw, f, sort_keys=True, indent=2)
    with pytest.raises(AttestationVerificationError):
        verify_attestation_pack(out)


def test_write_pack_rejects_signing_key_mismatch(tmp_path):
    p, k = _record_trace(tmp_path, "trace")
    pack = build_attestation_pack(
        [(p, k.hmac_key)],
        attestor_signing_key=Ed25519PrivateKey.generate(),
    )
    other = Ed25519PrivateKey.generate()
    with pytest.raises(ValueError):
        write_attestation_pack(pack, str(tmp_path / "x.pack"), signing_key=other)


# --------------------------------------------------- corrupt trace handling


def test_corrupt_trace_reported_as_failed_entry(tmp_path):
    good_p, good_k = _record_trace(tmp_path, "good")
    bad_p, bad_k = _record_trace(tmp_path, "bad")
    # Corrupt the bad trace by truncating its last 32 bytes (snaps
    # the HMAC chain on the tail frame).
    sz = os.path.getsize(bad_p)
    with open(bad_p, "rb+") as f:
        f.truncate(sz - 32)

    pack = build_attestation_pack(
        [(good_p, good_k.hmac_key), (bad_p, bad_k.hmac_key)],
        attestor_signing_key=Ed25519PrivateKey.generate(),
    )
    statuses = [e.verify_status for e in pack.entries]
    assert "ok" in statuses and "fail" in statuses
    fail_entry = next(e for e in pack.entries if e.verify_status == "fail")
    assert fail_entry.verify_error
    assert fail_entry.step_count == 0
    # Summary reflects the failure.
    assert pack.summary["verified_fail"] == 1
    assert pack.summary["verified_ok"] == 1


# ----------------------------------------------------- CLI integration


def test_cli_attest_then_verify_pack(tmp_path):
    p1, k1 = _record_trace(tmp_path, "t1")
    p2, k2 = _record_trace(tmp_path, "t2")
    # Use one shared HMAC key for the CLI (the production case where
    # all traces in a quarter were signed under the same KMS key).
    # Re-record under a single shared key so --hmac-key-hex works.
    shared = RecorderKey.fresh()
    p1 = str(tmp_path / "shared1.sb")
    p2 = str(tmp_path / "shared2.sb")
    for path in (p1, p2):
        with record(path, key=shared) as rec:
            run_recorded_agent(rec)

    out = str(tmp_path / "Q1.pack")
    signing = Ed25519PrivateKey.generate()
    signing_hex = signing.private_bytes_raw().hex()

    env = dict(os.environ)
    env["PYTHONPATH"] = str(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    proc = subprocess.run(
        [
            sys.executable, "-m", "stepback.cli", "attest",
            p1, p2,
            "--hmac-key-hex", shared.hmac_key.hex(),
            "--signing-key-hex", signing_hex,
            "--out", out,
            "--policy-version-pin", "2026-04-15",
            "--json",
        ],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    body = json.loads(proc.stdout)
    assert body["summary"]["trace_count"] == 2
    assert body["summary"]["verified_ok"] == 2

    # verify-pack subcommand.
    proc2 = subprocess.run(
        [sys.executable, "-m", "stepback.cli", "verify-pack", out],
        capture_output=True, text=True, env=env,
    )
    assert proc2.returncode == 0, proc2.stderr
    assert "OK" in proc2.stdout

    # Pinned-attestor mismatch path.
    proc3 = subprocess.run(
        [
            sys.executable, "-m", "stepback.cli", "verify-pack", out,
            "--expected-public-key", "ed25519:" + ("00" * 32),
        ],
        capture_output=True, text=True, env=env,
    )
    assert proc3.returncode == 2

    # Round-trip: pack must parse via the python API too.
    parsed = read_attestation_pack(out)
    assert parsed["policy_version_pin"] == "2026-04-15"
    assert parsed["summary"]["trace_count"] == 2
