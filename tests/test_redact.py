"""Tests for the trace redactor (`stepback/redact.py`)."""
from __future__ import annotations

import json
import os
import re
import tempfile

import pytest

from stepback import record
from stepback.canonical import hash_obj
from stepback.recorder import RecorderKey
from stepback.redact import (
    Detector,
    EMAIL_RE,
    IBAN_RE,
    PROTECTED_KEYS,
    RedactionManifest,
    RedactionPolicy,
    STANDARD_POLICY,
    STRICT_POLICY,
    redact_step,
    redact_steps,
    redact_string,
    redact_trace_file,
    redact_value,
    _luhn_ok,
)
from stepback.trace_reader import verify_trace
from tests.fixtures.agent import run_recorded_agent


# ---------------------------------------------------------------- detectors


def test_email_regex_matches_common_forms():
    s = "Contact alice@example.com or bob.smith+notif@sub.dom.uk for info."
    spans = [m.group(0) for m in EMAIL_RE.finditer(s)]
    assert "alice@example.com" in spans
    assert "bob.smith+notif@sub.dom.uk" in spans


def test_iban_regex_matches_grouped_iban():
    s = "wire to GB99-9999-9999 or DE89 3704 0044 0532 0130 00 today"
    found = [m.group(0) for m in IBAN_RE.finditer(s)]
    assert any("GB99" in f for f in found)
    assert any("DE89" in f for f in found)


def test_luhn_validator():
    assert _luhn_ok("4242424242424242")  # Stripe test
    assert _luhn_ok("5555555555554444")
    assert not _luhn_ok("1234567890123456")
    assert not _luhn_ok("12")  # too short


# ---------------------------------------------------------------- redact_string


def test_redact_string_emits_stable_token_for_same_input():
    out_a = redact_string("alice@example.com", STANDARD_POLICY)
    out_b = redact_string("alice@example.com", STANDARD_POLICY)
    assert out_a == out_b
    assert "alice@example.com" not in out_a
    assert out_a.startswith("<REDACTED:email:")


def test_redact_string_different_inputs_get_different_tokens():
    out_a = redact_string("alice@example.com", STANDARD_POLICY)
    out_b = redact_string("bob@example.com", STANDARD_POLICY)
    assert out_a != out_b


def test_redact_string_different_salt_different_token():
    p1 = RedactionPolicy(name="p1", detectors=list(STANDARD_POLICY.detectors), salt=b"AAA")
    p2 = RedactionPolicy(name="p2", detectors=list(STANDARD_POLICY.detectors), salt=b"BBB")
    assert redact_string("alice@example.com", p1) != redact_string("alice@example.com", p2)


def test_redact_string_handles_overlap_first_detector_wins():
    # An IBAN-looking string that also matches the IPv4 pattern? Not really.
    # Test overlap with two custom detectors.
    pol = RedactionPolicy(
        name="overlap",
        detectors=[
            Detector("first", re.compile(r"abc\d+"), strategy="mask"),
            Detector("second", re.compile(r"\d+def"), strategy="mask"),
        ],
        salt=b"x",
    )
    # "abc123def" matches "first" as "abc123"; "second" then can't take "123def"
    out = redact_string("abc123def", pol)
    assert "<REDACTED:first>" in out
    assert "<REDACTED:second>" not in out


def test_redact_string_records_manifest_hits():
    m = RedactionManifest(policy_name="standard", salt_id="x")
    redact_string("Email alice@example.com or call 555-123-4567.", STANDARD_POLICY, m)
    assert m.per_detector.get("email") == 1
    assert m.per_detector.get("phone") == 1
    assert m.n_redactions == 2


def test_redact_string_drop_strategy_removes_text():
    pol = RedactionPolicy(
        name="drop",
        detectors=[Detector("nuke", re.compile(r"SECRET"), strategy="drop")],
        salt=b"x",
    )
    assert redact_string("hello SECRET world", pol) == "hello  world"


def test_redact_string_credit_card_only_when_luhn_passes():
    # 4242 4242 4242 4242 is a valid Luhn test card.
    out = redact_string("Card 4242 4242 4242 4242 charged", STANDARD_POLICY)
    assert "<REDACTED:credit_card:" in out
    assert "4242 4242 4242 4242" not in out

    # 16 random non-Luhn digits should NOT be redacted as credit_card.
    bad = "Card 1234 5678 9012 3456 charged"
    out2 = redact_string(bad, STANDARD_POLICY)
    assert "<REDACTED:credit_card" not in out2


def test_redact_string_aws_key_masked():
    out = redact_string("aws AKIAIOSFODNN7EXAMPLE creds", STANDARD_POLICY)
    assert out == "aws <REDACTED:aws_key> creds"


def test_redact_string_jwt_masked():
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    out = redact_string(f"Auth: Bearer {jwt}", STANDARD_POLICY)
    assert jwt not in out
    assert "<REDACTED:jwt>" in out


# ---------------------------------------------------------------- redact_value / step


def test_redact_value_walks_nested_dict_and_list():
    val = {
        "messages": [
            {"role": "user", "content": "email me at alice@example.com"},
            {"role": "assistant", "content": "ok"},
        ]
    }
    out = redact_value(val, STANDARD_POLICY)
    assert "alice@example.com" not in json.dumps(out)
    # role "user" must NOT be redacted (it's a structural value but
    # it's a leaf string under a non-protected key so this test is
    # really about the email substitution).
    assert out["messages"][0]["role"] == "user"


def test_redact_value_skips_protected_keys():
    # `step_id`, `model`, `role` are in PROTECTED_KEYS.
    val = {
        "step_id": "step:42",
        "model": "gpt-4o-2024-11-20",
        "messages": [{"role": "user", "content": "alice@example.com"}],
    }
    out = redact_value(val, STANDARD_POLICY)
    assert out["step_id"] == "step:42"
    assert out["model"] == "gpt-4o-2024-11-20"
    assert "alice@example.com" not in out["messages"][0]["content"]


def test_protected_keys_includes_critical_replay_fields():
    for k in ("step_id", "step_kind", "parent_step_id", "model", "role"):
        assert k in PROTECTED_KEYS


def test_redact_step_recomputes_inputs_and_outputs_hash():
    step = {
        "step_id": "step:1",
        "step_kind": "tool_call",
        "parent_step_id": None,
        "inputs": {"kind": "tool_call", "name": "lookup", "args": {"q": "alice@example.com"}},
        "outputs": {"result": "ok"},
        "inputs_hash": "sha256:deadbeef",
        "outputs_hash": "sha256:cafef00d",
    }
    out = redact_step(step, STANDARD_POLICY)
    assert "alice@example.com" not in json.dumps(out)
    # The new hash field must equal the canonical hash of the new
    # inputs dict — i.e. self-consistent.
    assert out["inputs_hash"] == hash_obj(out["inputs"])
    assert out["outputs_hash"] == hash_obj(out["outputs"])
    # And it must NOT equal the old (pre-redaction) hash.
    assert out["inputs_hash"] != "sha256:deadbeef"


def test_redact_step_preserves_step_id_and_parent_link():
    step = {
        "step_id": "step:7",
        "parent_step_id": "step:6",
        "step_kind": "llm_call",
        "inputs": {"messages": [{"role": "user", "content": "alice@x.com"}]},
        "outputs": {"text": "ok"},
        "inputs_hash": "sha256:0",
        "outputs_hash": "sha256:0",
    }
    out = redact_step(step, STANDARD_POLICY)
    assert out["step_id"] == "step:7"
    assert out["parent_step_id"] == "step:6"
    assert out["step_kind"] == "llm_call"


def test_redact_steps_preserves_dedup_structure():
    # If the same email appears twice across two steps, the redacted
    # tokens must be byte-identical so cache structure is preserved.
    s1 = {
        "step_id": "step:1",
        "step_kind": "llm_call",
        "parent_step_id": None,
        "inputs": {"messages": [{"role": "user", "content": "alice@example.com"}]},
        "outputs": {"text": "1"},
    }
    s2 = {
        "step_id": "step:2",
        "step_kind": "llm_call",
        "parent_step_id": "step:1",
        "inputs": {"messages": [{"role": "user", "content": "alice@example.com"}]},
        "outputs": {"text": "2"},
    }
    out, manifest = redact_steps([s1, s2], STANDARD_POLICY)
    c1 = out[0]["inputs"]["messages"][0]["content"]
    c2 = out[1]["inputs"]["messages"][0]["content"]
    assert c1 == c2
    assert manifest.n_steps == 2
    assert manifest.per_detector["email"] == 2


def test_strict_policy_redacts_proper_names_and_currency():
    s = "Pay Acme Bolts $50000 today"
    out_std = redact_string(s, STANDARD_POLICY)
    out_strict = redact_string(s, STRICT_POLICY)
    # standard policy should NOT touch Acme Bolts / $50000
    assert "Acme Bolts" in out_std
    assert "$50000" in out_std
    # strict policy SHOULD redact both
    assert "Acme Bolts" not in out_strict
    assert "$50000" not in out_strict


# ---------------------------------------------------------------- e2e file


def test_redact_trace_file_end_to_end_round_trip(tmp_path):
    # 1. record a trace using the bundled fixture agent (which is
    #    full of PII-ish data: IBANs, customer names, amounts).
    in_path = str(tmp_path / "in.sb")
    in_key = RecorderKey.fresh()
    with record(in_path, key=in_key) as rec:
        run_recorded_agent(rec)

    # 2. redact it.
    out_path = str(tmp_path / "out.sb")
    manifest = redact_trace_file(
        in_path,
        out_path,
        in_hmac_key=in_key.hmac_key,
        policy=STANDARD_POLICY,
    )

    # 3. the output file exists and is verifiable as a fresh trace.
    assert os.path.exists(out_path)
    # The output trace was signed with a fresh key — we don't have
    # its hmac key here. But we can read frames + see it parses.
    from stepback.trace_reader import read_frames
    frames = read_frames(out_path)
    assert len(frames) >= 3  # header + at least one step + tail

    # 4. the manifest reports redactions.
    assert manifest.n_steps >= 1
    # The fixture agent has IBAN GB99-9999-9999 in the messages, so:
    assert manifest.per_detector.get("iban", 0) >= 1
    # Standard policy preserves the email-less fixture's other PII.

    # 5. the IBAN string itself does NOT appear in the redacted file.
    raw = open(out_path, "rb").read()
    assert b"GB99-9999-9999" not in raw


def test_redact_trace_file_writes_self_consistent_hashes(tmp_path):
    in_path = str(tmp_path / "in.sb")
    in_key = RecorderKey.fresh()
    with record(in_path, key=in_key) as rec:
        run_recorded_agent(rec)

    out_path = str(tmp_path / "out.sb")
    out_key = RecorderKey.fresh()
    redact_trace_file(
        in_path,
        out_path,
        in_hmac_key=in_key.hmac_key,
        policy=STANDARD_POLICY,
        out_key=out_key,
    )

    # Verify the redacted trace's HMAC chain end-to-end with the
    # fresh out_key, then check inputs_hash / outputs_hash are
    # self-consistent on every step.
    parsed = verify_trace(out_path, out_key.hmac_key)
    assert len(parsed.steps) > 0
    for s in parsed.steps:
        if "inputs" in s and "inputs_hash" in s:
            assert s["inputs_hash"] == hash_obj(s["inputs"]), (
                f"step {s.get('step_id')} inputs_hash not self-consistent"
            )
        if "outputs" in s and "outputs_hash" in s:
            assert s["outputs_hash"] == hash_obj(s["outputs"])


def test_redact_trace_file_refuses_tampered_input(tmp_path):
    in_path = str(tmp_path / "in.sb")
    in_key = RecorderKey.fresh()
    with record(in_path, key=in_key) as rec:
        run_recorded_agent(rec)

    # Corrupt one byte of the file in the middle.
    data = bytearray(open(in_path, "rb").read())
    # Skip the 4-byte length prefix and a few bytes into the header
    # to land in payload territory.
    target = len(data) // 2
    data[target] ^= 0xFF
    open(in_path, "wb").write(bytes(data))

    from stepback.trace_reader import TraceVerificationError

    with pytest.raises((TraceVerificationError, UnicodeDecodeError, ValueError)):
        redact_trace_file(
            in_path,
            str(tmp_path / "out.sb"),
            in_hmac_key=in_key.hmac_key,
            policy=STANDARD_POLICY,
        )


# ---------------------------------------------------------------- CLI


def test_cli_redact_subcommand(tmp_path, capsys):
    from stepback.cli import main as cli_main

    in_path = str(tmp_path / "in.sb")
    in_key = RecorderKey.fresh()
    with record(in_path, key=in_key) as rec:
        run_recorded_agent(rec)

    out_path = str(tmp_path / "out.sb")
    manifest_path = str(tmp_path / "manifest.json")

    rc = cli_main(
        [
            "redact",
            in_path,
            "-o",
            out_path,
            "--hmac-key-hex",
            in_key.hmac_key.hex(),
            "--policy",
            "standard",
            "--manifest",
            manifest_path,
            "--json",
        ]
    )
    assert rc == 0
    assert os.path.exists(out_path)
    assert os.path.exists(manifest_path)

    payload = json.load(open(manifest_path))
    assert payload["policy_name"] == "standard"
    assert payload["n_steps"] >= 1
    assert payload["n_redactions"] >= 1


def test_cli_redact_unknown_policy_returns_2(tmp_path):
    from stepback.cli import main as cli_main

    in_path = str(tmp_path / "in.sb")
    in_key = RecorderKey.fresh()
    with record(in_path, key=in_key) as rec:
        run_recorded_agent(rec)

    rc = cli_main(
        [
            "redact",
            in_path,
            "-o",
            str(tmp_path / "out.sb"),
            "--hmac-key-hex",
            in_key.hmac_key.hex(),
            "--policy",
            "no-such-policy",
        ]
    )
    assert rc == 2


# ============================================================
# Layer-3 MoA expansion: provider-token detectors, allowlist,
# KeyContextDetector, scan-only mode, streaming redactor.
# ============================================================

import io
import re as _re
from contextlib import redirect_stdout, redirect_stderr

from stepback.redact import (
    ANTHROPIC_KEY_RE,
    AZURE_SAS_RE,
    BTC_ADDR_RE,
    ETH_ADDR_RE,
    GITHUB_PAT_RE,
    GOOGLE_API_KEY_RE,
    KeyContextDetector,
    MAC_ADDR_RE,
    OPENAI_KEY_RE,
    STRIPE_KEY_RE,
    STRIPE_WEBHOOK_RE,
    SLACK_TOKEN_RE,
    TWILIO_SID_RE,
    ScanReport,
    redact_trace_file_streaming,
    scan_steps,
    scan_trace_file,
    scan_value,
)
from stepback.trace_writer import TraceWriter as _TraceWriterImport
TraceWriter = _TraceWriterImport


# ---------------------------------------------------------------- new detectors


def test_github_pat_regex_matches_and_misses():
    assert GITHUB_PAT_RE.search("ghp_" + "a" * 36)
    assert GITHUB_PAT_RE.search("ghs_" + "B" * 40)
    # Wrong prefix length / unknown letter
    assert not GITHUB_PAT_RE.search("ghx_" + "a" * 36)
    # Too short
    assert not GITHUB_PAT_RE.search("ghp_" + "a" * 5)


def test_openai_anthropic_google_keys_match():
    assert OPENAI_KEY_RE.search("sk-proj-" + "X" * 40)
    assert OPENAI_KEY_RE.search("sk-" + "abc" * 14)
    assert ANTHROPIC_KEY_RE.search("sk-ant-api03-" + "Y" * 40)
    assert GOOGLE_API_KEY_RE.search("AIza" + "Z" * 35)
    assert not GOOGLE_API_KEY_RE.search("AIza" + "Z" * 30)


def test_slack_stripe_twilio_keys_match():
    assert SLACK_TOKEN_RE.search("xoxb-1234567890-abcdefghij")
    assert STRIPE_KEY_RE.search("sk_live_" + "Q" * 24)
    assert STRIPE_WEBHOOK_RE.search("whsec_" + "W" * 30)
    assert TWILIO_SID_RE.search("AC" + "0123456789abcdef" * 2)


def test_azure_mac_btc_eth_match():
    assert AZURE_SAS_RE.search(
        "https://x.blob.core.windows.net/?sig=abc123%2Fdef&se=2026-12-31T00%3A00Z"
    )
    assert MAC_ADDR_RE.search("0a:1b:2c:3d:4e:5f")
    assert ETH_ADDR_RE.search("0x" + "a" * 40)
    # Real BTC P2PKH-style address shape
    assert BTC_ADDR_RE.search("1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2")
    # base58-shaped lowercase hex (no 0 / o / i / l) → matches regex
    # but predicate rejects pure-hex.
    pseudo = "1" + ("abcdef" * 5)[:30]  # all lower-hex
    pol = STANDARD_POLICY
    spans = []
    for det in pol.detectors:
        if det.name == "btc_address":
            spans = det.find(pseudo)
            break
    assert spans == []  # predicate dropped it



def test_pem_block_redacted_as_whole_block():
    pem = (
        "key=-----BEGIN PRIVATE KEY-----\n"
        "MIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQ\n"
        "RsAaaaaaaa.bbbbbbbbbb.cccccccccc\n"
        "-----END PRIVATE KEY-----"
    )
    out = redact_string(pem, STANDARD_POLICY)
    assert "BEGIN PRIVATE KEY" not in out
    assert "<REDACTED:pem_private_key>" in out
    # Internals (which would otherwise have been picked up by JWT_RE)
    # are gone.
    assert "RsAaaaaaaa" not in out


def test_provider_tokens_all_masked_in_one_step():
    blob = (
        "github=ghp_" + "a" * 36
        + " openai=sk-proj-" + "B" * 40
        + " anthropic=sk-ant-api03-" + "C" * 40
        + " google=AIza" + "D" * 35
        + " slack=xoxb-1-2-abcdefghijkl"
        + " stripe=sk_live_" + "E" * 24
        + " stripe_wh=whsec_" + "F" * 30
        + " twilio=AC" + "0123456789abcdef" * 2
        + " mac=0a:1b:2c:3d:4e:5f"
        + " eth=0x" + "9" * 40
    )
    out = redact_string(blob, STANDARD_POLICY)
    for kind in (
        "github_pat", "openai_key", "anthropic_key", "google_api_key",
        "slack_token", "stripe_key", "stripe_webhook", "twilio_sid",
        "mac_address", "eth_address",
    ):
        assert (
            f"<REDACTED:{kind}>" in out or f"<REDACTED:{kind}:" in out
        ), f"detector {kind} did not fire on {out!r}"
    # No raw secret survived
    assert "ghp_" + "a" * 36 not in out
    assert "sk-proj-" + "B" * 40 not in out


# ---------------------------------------------------------------- allowlist


def test_allowlist_exact_skips_redaction():
    policy = RedactionPolicy(
        name="al-test",
        detectors=[Detector("email", EMAIL_RE, strategy="hash")],
        salt=b"x",
        allowlist=frozenset({"alice@example.com"}),
    )
    s = "ping alice@example.com and bob@example.com"
    out = redact_string(s, policy)
    assert "alice@example.com" in out
    assert "bob@example.com" not in out
    assert "<REDACTED:email:" in out


def test_allowlist_pattern_skips_redaction():
    policy = RedactionPolicy(
        name="alp-test",
        detectors=[Detector("email", EMAIL_RE, strategy="hash")],
        salt=b"x",
        allowlist_patterns=(_re.compile(r".+@safe\.example\.com"),),
    )
    out = redact_string("a@safe.example.com b@unsafe.example.com", policy)
    assert "a@safe.example.com" in out
    assert "b@unsafe.example.com" not in out


# ---------------------------------------------------------------- key context


def test_key_context_detector_catches_bare_authorization_header():
    s = 'Authorization: Bearer abracadabra1234567890'
    out = redact_string(s, STANDARD_POLICY)
    # Header name preserved, value masked
    assert "Authorization" in out
    assert "abracadabra1234567890" not in out
    assert "<REDACTED:" in out


def test_key_context_detector_catches_apikey_assignment():
    s = "config: api_key=hunter2hunter2hunter2hunter2"
    out = redact_string(s, STANDARD_POLICY)
    assert "hunter2hunter2hunter2hunter2" not in out
    assert "api_key" in out


def test_key_context_detector_skips_short_numeric():
    # short numeric IDs after `id:` are not secrets
    s = "session_id: 42"
    out = redact_string(s, STANDARD_POLICY)
    assert "42" in out  # untouched


# ---------------------------------------------------------------- scan-only


def test_scan_value_reports_findings_without_mutation():
    payload = {
        "input": "email me at alice@example.com",
        "raw_request": "POST /v1\nAuthorization: Bearer xyz123abcxyz123abcxyz\n",
    }
    snapshot = json.dumps(payload, sort_keys=True)
    report = scan_value(payload, STANDARD_POLICY)
    # Original value is untouched
    assert json.dumps(payload, sort_keys=True) == snapshot
    assert report.n_findings >= 2
    assert "email" in report.per_detector


def test_scan_steps_findings_align_with_actual_redact_steps(tmp_path):
    in_key = RecorderKey.fresh()
    in_path = str(tmp_path / "scan.sb")
    with record(in_path, key=in_key) as rec:
        run_recorded_agent(rec)
    parsed = verify_trace(in_path, in_key.hmac_key)
    # Inject a known secret string into the first llm-call step's outputs
    # so we have something to scan / redact.
    found_idx = None
    for i, s in enumerate(parsed.steps):
        if s.get("step_kind") == "llm_call":
            s.setdefault("outputs", {})["leak"] = (
                "ghp_" + "z" * 36 + " ; alice@example.com"
            )
            found_idx = i
            break
    assert found_idx is not None
    report = scan_steps(parsed.steps, STANDARD_POLICY)
    _, manifest = redact_steps(parsed.steps, STANDARD_POLICY)
    # Same per-detector counts in scan vs. apply.
    assert report.per_detector == manifest.per_detector
    assert report.n_findings == manifest.n_redactions


def test_scan_trace_file_does_not_create_output(tmp_path):
    in_key = RecorderKey.fresh()
    in_path = str(tmp_path / "in.sb")
    with record(in_path, key=in_key) as rec:
        run_recorded_agent(rec)
    pre = set(os.listdir(tmp_path))
    report = scan_trace_file(in_path, in_hmac_key=in_key.hmac_key, policy=STANDARD_POLICY)
    post = set(os.listdir(tmp_path))
    assert pre == post  # zero side effects
    assert isinstance(report, ScanReport)
    assert report.n_steps > 0


# ---------------------------------------------------------------- streaming


def test_streaming_redact_equivalent_to_eager(tmp_path):
    in_key = RecorderKey.fresh()
    in_path = str(tmp_path / "src.sb")
    with record(in_path, key=in_key) as rec:
        run_recorded_agent(rec)
    # Inject some secrets into outputs
    parsed = verify_trace(in_path, in_key.hmac_key)
    for s in parsed.steps[:3]:
        s.setdefault("outputs", {})["leak"] = "x@y.com"

    # Re-write the trace with the injected secrets so both passes see them.
    inj_path = str(tmp_path / "src_inj.sb")
    inj_key = RecorderKey.fresh()
    w = TraceWriter.open(inj_path, hmac_key=inj_key.hmac_key, signing_key=inj_key.signing_key)
    try:
        for s in parsed.steps:
            w.write_step(s)
    finally:
        w.close()

    eager_out = str(tmp_path / "eager.sb")
    stream_out = str(tmp_path / "stream.sb")
    out_key = RecorderKey.fresh()
    m_e = redact_trace_file(
        inj_path, eager_out, in_hmac_key=inj_key.hmac_key,
        policy=STANDARD_POLICY, out_key=out_key,
    )
    m_s = redact_trace_file_streaming(
        inj_path, stream_out, in_hmac_key=inj_key.hmac_key,
        policy=STANDARD_POLICY, out_key=out_key,
    )
    assert m_e.n_steps == m_s.n_steps
    assert m_e.n_redactions == m_s.n_redactions
    assert m_e.per_detector == m_s.per_detector


# ---------------------------------------------------------------- CLI


def test_cli_redact_scan_emits_json(tmp_path):
    from stepback.cli import main as cli_main

    in_key = RecorderKey.fresh()
    in_path = str(tmp_path / "in.sb")
    with record(in_path, key=in_key) as rec:
        run_recorded_agent(rec)

    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        rc = cli_main([
            "redact-scan",
            in_path,
            "--hmac-key-hex",
            in_key.hmac_key.hex(),
            "--json",
        ])
    assert rc == 0
    parsed = json.loads(buf.getvalue())
    assert parsed["policy_name"] == "standard"
    assert "per_detector" in parsed
    assert parsed["n_steps"] > 0


def test_cli_redact_streaming_writes_output(tmp_path):
    from stepback.cli import main as cli_main

    in_key = RecorderKey.fresh()
    in_path = str(tmp_path / "in.sb")
    out_path = str(tmp_path / "out.sb")
    with record(in_path, key=in_key) as rec:
        run_recorded_agent(rec)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main([
            "redact",
            in_path,
            "-o", out_path,
            "--hmac-key-hex", in_key.hmac_key.hex(),
            "--streaming",
            "--json",
        ])
    assert rc == 0
    assert os.path.exists(out_path)
    summary = json.loads(buf.getvalue())
    assert summary["policy_name"] == "standard"
    assert summary["n_steps"] > 0
