"""Tests for stepback.provenance — SLSA / in-toto provenance attestations.

Covers:
* Statement structure for traces, benchmarks, and replay packs.
* DSSE sign / verify round-trip.
* Tamper detection (payload, signature, payloadType).
* Utility helpers (sha256_of_file, sha256_of_bytes).
* Integration: derive provenance from a real recorded trace and a
  real attestation pack.

All tests are offline — no network calls.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from stepback.provenance import (
    BUILD_TYPE_BENCHMARK,
    BUILD_TYPE_PACK,
    BUILD_TYPE_TRACE,
    BUILDER_ID,
    DSSE_PAYLOAD_TYPE,
    INTOTO_STATEMENT_TYPE,
    MEDIA_TYPE_BENCHMARK,
    MEDIA_TYPE_PACK,
    MEDIA_TYPE_TRACE,
    SLSA_PREDICATE_TYPE,
    ProvenanceVerificationError,
    benchmark_provenance,
    pack_provenance,
    sha256_of_bytes,
    sha256_of_file,
    sign_provenance,
    trace_provenance,
    verify_provenance_signature,
)


# ------------------------------------------------------------------ helpers

def _gen_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


def _fake_sha256() -> str:
    return "sha256:" + "ab" * 32


# ================================================================== statement shape


class TestTraceProvenance:
    def test_required_structure(self):
        stmt = trace_provenance("run.sb", _fake_sha256())
        assert stmt["_type"] == INTOTO_STATEMENT_TYPE
        assert stmt["predicateType"] == SLSA_PREDICATE_TYPE
        assert len(stmt["subject"]) == 1
        subj = stmt["subject"][0]
        assert subj["name"] == "run.sb"
        assert subj["digest"]["sha256"] == "ab" * 32
        assert subj["mediaType"] == MEDIA_TYPE_TRACE

    def test_predicate_shape(self):
        stmt = trace_provenance("t.sb", _fake_sha256())
        pred = stmt["predicate"]
        bd = pred["buildDefinition"]
        assert bd["buildType"] == BUILD_TYPE_TRACE
        assert "externalParameters" in bd
        assert "internalParameters" in bd
        assert "resolvedDependencies" in bd
        rd = pred["runDetails"]
        assert rd["builder"]["id"] == BUILDER_ID
        assert "metadata" in rd
        assert "byproducts" in rd

    def test_optional_fields_appear_as_byproducts(self):
        stmt = trace_provenance(
            "t.sb",
            "sha256:" + "cd" * 32,
            trace_chain_hash="sha256:" + "01" * 32,
            merkle_root="sha256:" + "02" * 32,
            recorder_public_key="ed25519:" + "03" * 32,
            step_count=12,
        )
        byproducts = stmt["predicate"]["runDetails"]["byproducts"]
        names = {b["name"]: b["value"] for b in byproducts}
        assert names["trace_chain_hash"] == "sha256:" + "01" * 32
        assert names["merkle_root"] == "sha256:" + "02" * 32
        assert names["recorder_public_key"] == "ed25519:" + "03" * 32
        assert names["step_count"] == 12

    def test_invocation_id_defaults_to_sha256(self):
        digest = "sha256:" + "ff" * 32
        stmt = trace_provenance("t.sb", digest)
        assert stmt["predicate"]["runDetails"]["metadata"]["invocationId"] == digest

    def test_invocation_id_prefers_chain_hash(self):
        chain = "sha256:" + "aa" * 32
        stmt = trace_provenance("t.sb", _fake_sha256(), trace_chain_hash=chain)
        assert stmt["predicate"]["runDetails"]["metadata"]["invocationId"] == chain

    def test_internal_params_recorder_version(self):
        stmt = trace_provenance(
            "t.sb",
            _fake_sha256(),
            recorder_version="0.1.0",
            canonicalisation_version="1",
        )
        ip = stmt["predicate"]["buildDefinition"]["internalParameters"]
        assert ip["recorder_version"] == "0.1.0"
        assert ip["canonicalisation_version"] == "1"


class TestBenchmarkProvenance:
    _content = b'{"corpus_id":"test","results":[]}'

    def test_required_structure(self):
        stmt = benchmark_provenance("bench.json", self._content)
        assert stmt["_type"] == INTOTO_STATEMENT_TYPE
        assert stmt["predicateType"] == SLSA_PREDICATE_TYPE
        subj = stmt["subject"][0]
        assert subj["name"] == "bench.json"
        expected_hex = hashlib.sha256(self._content).hexdigest()
        assert subj["digest"]["sha256"] == expected_hex
        assert subj["mediaType"] == MEDIA_TYPE_BENCHMARK

    def test_predicate_shape(self):
        stmt = benchmark_provenance("bench.json", self._content)
        bd = stmt["predicate"]["buildDefinition"]
        assert bd["buildType"] == BUILD_TYPE_BENCHMARK

    def test_corpus_id_in_external_params(self):
        stmt = benchmark_provenance(
            "bench.json", self._content, corpus_id="swe-bench-verified"
        )
        ep = stmt["predicate"]["buildDefinition"]["externalParameters"]
        assert ep["corpus_id"] == "swe-bench-verified"

    def test_trace_count_byproduct(self):
        stmt = benchmark_provenance("b.json", self._content, trace_count=100)
        bps = {b["name"]: b["value"] for b in stmt["predicate"]["runDetails"]["byproducts"]}
        assert bps["trace_count"] == 100

    def test_digest_matches_content_bytes(self):
        content = b'{"hello": "world"}'
        stmt = benchmark_provenance("x.json", content)
        expected = hashlib.sha256(content).hexdigest()
        assert stmt["subject"][0]["digest"]["sha256"] == expected


class TestPackProvenance:
    def test_required_structure(self):
        stmt = pack_provenance("Q1.pack", _fake_sha256())
        assert stmt["_type"] == INTOTO_STATEMENT_TYPE
        assert stmt["predicateType"] == SLSA_PREDICATE_TYPE
        subj = stmt["subject"][0]
        assert subj["name"] == "Q1.pack"
        assert subj["mediaType"] == MEDIA_TYPE_PACK

    def test_predicate_shape(self):
        stmt = pack_provenance("Q1.pack", _fake_sha256())
        bd = stmt["predicate"]["buildDefinition"]
        assert bd["buildType"] == BUILD_TYPE_PACK

    def test_trace_descriptors_as_resolved_dependencies(self):
        deps = [
            {"name": "t1.sb", "digest": {"sha256": "11" * 32}},
            {"name": "t2.sb", "digest": {"sha256": "22" * 32}},
        ]
        stmt = pack_provenance("Q1.pack", _fake_sha256(), trace_descriptors=deps)
        rd = stmt["predicate"]["buildDefinition"]["resolvedDependencies"]
        assert len(rd) == 2
        assert rd[0]["name"] == "t1.sb"

    def test_policy_pin_in_external_params(self):
        stmt = pack_provenance(
            "Q1.pack", _fake_sha256(), policy_version_pin="2026-04-15"
        )
        ep = stmt["predicate"]["buildDefinition"]["externalParameters"]
        assert ep["policy_version_pin"] == "2026-04-15"

    def test_attestor_key_in_internal_params(self):
        stmt = pack_provenance(
            "Q1.pack",
            _fake_sha256(),
            attestor_public_key="ed25519:" + "ab" * 32,
        )
        ip = stmt["predicate"]["buildDefinition"]["internalParameters"]
        assert ip["attestor_public_key"] == "ed25519:" + "ab" * 32

    def test_summary_byproducts(self):
        stmt = pack_provenance(
            "Q1.pack",
            _fake_sha256(),
            trace_count=50,
            verified_ok=49,
            divergent_traces=3,
        )
        bps = {b["name"]: b["value"] for b in stmt["predicate"]["runDetails"]["byproducts"]}
        assert bps["trace_count"] == 50
        assert bps["verified_ok"] == 49
        assert bps["divergent_traces"] == 3


# ================================================================== DSSE signing


class TestDSSESignVerify:
    def test_sign_envelope_shape(self):
        key = _gen_key()
        stmt = trace_provenance("t.sb", _fake_sha256())
        env = sign_provenance(stmt, key)
        assert env["payloadType"] == DSSE_PAYLOAD_TYPE
        assert "payload" in env
        assert len(env["signatures"]) == 1
        sig_entry = env["signatures"][0]
        assert "keyid" in sig_entry
        assert "sig" in sig_entry

    def test_keyid_defaults_to_pubhex(self):
        key = _gen_key()
        stmt = trace_provenance("t.sb", _fake_sha256())
        env = sign_provenance(stmt, key)
        pub_hex = key.public_key().public_bytes_raw().hex()
        assert env["signatures"][0]["keyid"] == f"ed25519:{pub_hex}"

    def test_keyid_override(self):
        key = _gen_key()
        stmt = trace_provenance("t.sb", _fake_sha256())
        env = sign_provenance(stmt, key, keyid="my-key-id")
        assert env["signatures"][0]["keyid"] == "my-key-id"

    def test_payload_is_standard_base64(self):
        key = _gen_key()
        stmt = trace_provenance("t.sb", _fake_sha256())
        env = sign_provenance(stmt, key)
        # Standard base64 must decode cleanly.
        decoded = base64.b64decode(env["payload"])
        assert json.loads(decoded) == stmt

    def test_sig_is_standard_base64(self):
        key = _gen_key()
        stmt = trace_provenance("t.sb", _fake_sha256())
        env = sign_provenance(stmt, key)
        raw_sig = base64.b64decode(env["signatures"][0]["sig"])
        # Ed25519 signatures are 64 bytes.
        assert len(raw_sig) == 64

    def test_verify_round_trip(self):
        key = _gen_key()
        stmt = trace_provenance("run.sb", _fake_sha256())
        env = sign_provenance(stmt, key)
        recovered = verify_provenance_signature(env, key.public_key())
        assert recovered == stmt

    def test_verify_benchmark_round_trip(self):
        key = _gen_key()
        stmt = benchmark_provenance("b.json", b'{"x":1}', corpus_id="test")
        env = sign_provenance(stmt, key)
        recovered = verify_provenance_signature(env, key.public_key())
        assert recovered["predicateType"] == SLSA_PREDICATE_TYPE

    def test_verify_pack_round_trip(self):
        key = _gen_key()
        stmt = pack_provenance("Q1.pack", _fake_sha256(), trace_count=10)
        env = sign_provenance(stmt, key)
        recovered = verify_provenance_signature(env, key.public_key())
        assert recovered["_type"] == INTOTO_STATEMENT_TYPE

    def test_wrong_key_raises(self):
        key1 = _gen_key()
        key2 = _gen_key()
        stmt = trace_provenance("t.sb", _fake_sha256())
        env = sign_provenance(stmt, key1)
        with pytest.raises(ProvenanceVerificationError):
            verify_provenance_signature(env, key2.public_key())

    def test_tampered_payload_raises(self):
        key = _gen_key()
        stmt = trace_provenance("t.sb", _fake_sha256())
        env = sign_provenance(stmt, key)
        # Tamper: flip a byte in the base64-decoded payload, re-encode.
        original = base64.b64decode(env["payload"])
        tampered = bytearray(original)
        tampered[0] ^= 0xFF
        env["payload"] = base64.b64encode(bytes(tampered)).decode("ascii")
        with pytest.raises(ProvenanceVerificationError):
            verify_provenance_signature(env, key.public_key())

    def test_tampered_signature_raises(self):
        key = _gen_key()
        stmt = trace_provenance("t.sb", _fake_sha256())
        env = sign_provenance(stmt, key)
        raw_sig = bytearray(base64.b64decode(env["signatures"][0]["sig"]))
        raw_sig[0] ^= 0xFF
        env["signatures"][0]["sig"] = base64.b64encode(bytes(raw_sig)).decode("ascii")
        with pytest.raises(ProvenanceVerificationError):
            verify_provenance_signature(env, key.public_key())

    def test_wrong_payload_type_raises(self):
        key = _gen_key()
        stmt = trace_provenance("t.sb", _fake_sha256())
        env = sign_provenance(stmt, key)
        env["payloadType"] = "application/json"
        with pytest.raises(ProvenanceVerificationError, match="payloadType"):
            verify_provenance_signature(env, key.public_key())

    def test_no_signatures_raises(self):
        key = _gen_key()
        stmt = trace_provenance("t.sb", _fake_sha256())
        env = sign_provenance(stmt, key)
        env["signatures"] = []
        with pytest.raises(ProvenanceVerificationError, match="no signatures"):
            verify_provenance_signature(env, key.public_key())

    def test_deterministic_payload_across_calls(self):
        """Signing the same statement twice should produce identical payloads."""
        key = _gen_key()
        stmt = trace_provenance("t.sb", _fake_sha256())
        env1 = sign_provenance(stmt, key)
        env2 = sign_provenance(stmt, key)
        # Payloads must be identical (deterministic JSON).
        assert env1["payload"] == env2["payload"]


# ================================================================== helpers


class TestSha256Helpers:
    def test_sha256_of_bytes(self):
        data = b"hello stepback"
        result = sha256_of_bytes(data)
        expected = "sha256:" + hashlib.sha256(data).hexdigest()
        assert result == expected

    def test_sha256_of_file(self, tmp_path):
        content = b"trace file content"
        f = tmp_path / "t.sb"
        f.write_bytes(content)
        result = sha256_of_file(str(f))
        expected = "sha256:" + hashlib.sha256(content).hexdigest()
        assert result == expected

    def test_sha256_of_file_large(self, tmp_path):
        # Ensure the 64 KiB chunked read works correctly.
        content = os.urandom(200_000)
        f = tmp_path / "big.sb"
        f.write_bytes(content)
        result = sha256_of_file(str(f))
        expected = "sha256:" + hashlib.sha256(content).hexdigest()
        assert result == expected


# ================================================================== integration


class TestIntegrationWithRealTrace:
    """End-to-end: record a trace, derive SLSA provenance, verify it."""

    def test_trace_file_provenance_roundtrip(self, tmp_path):
        """Record a real trace, generate provenance, sign, verify."""
        from stepback import RecorderKey, record
        from stepback.testing import run_recorded_agent

        key = RecorderKey.fresh()
        trace_path = str(tmp_path / "run.sb")
        with record(trace_path, key=key) as rec:
            run_recorded_agent(rec)

        digest = sha256_of_file(trace_path)
        pub_hex = key.signing_key.public_key().public_bytes_raw().hex()
        stmt = trace_provenance(
            "run.sb",
            digest,
            recorder_public_key="ed25519:" + pub_hex,
            step_count=12,
        )
        assert stmt["subject"][0]["digest"]["sha256"] == digest.removeprefix("sha256:")

        signing = Ed25519PrivateKey.generate()
        env = sign_provenance(stmt, signing)
        recovered = verify_provenance_signature(env, signing.public_key())
        assert recovered["_type"] == INTOTO_STATEMENT_TYPE
        assert recovered["predicateType"] == SLSA_PREDICATE_TYPE

    def test_pack_provenance_with_trace_deps(self, tmp_path):
        """Build an attestation pack, derive pack provenance with trace deps."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey as _Ed25519PK
        from stepback import RecorderKey, record
        from stepback.attestation import build_attestation_pack, write_attestation_pack
        from stepback.testing import run_recorded_agent

        k = RecorderKey.fresh()
        trace_path = str(tmp_path / "t.sb")
        with record(trace_path, key=k) as rec:
            run_recorded_agent(rec)

        attestor_key = _Ed25519PK.generate()
        pack = build_attestation_pack(
            [(trace_path, k.hmac_key)],
            attestor_signing_key=attestor_key,
        )
        pack_path = str(tmp_path / "Q1.pack")
        write_attestation_pack(pack, pack_path, signing_key=attestor_key)

        pack_digest = sha256_of_file(pack_path)
        trace_digest = sha256_of_file(trace_path)

        trace_deps = [
            {
                "name": os.path.basename(trace_path),
                "digest": {"sha256": trace_digest.removeprefix("sha256:")},
                "mediaType": MEDIA_TYPE_TRACE,
            }
        ]
        stmt = pack_provenance(
            "Q1.pack",
            pack_digest,
            trace_descriptors=trace_deps,
            attestor_public_key=pack.attestor_public_key,
            trace_count=pack.summary["trace_count"],
            verified_ok=pack.summary["verified_ok"],
        )
        rd = stmt["predicate"]["buildDefinition"]["resolvedDependencies"]
        assert len(rd) == 1
        assert rd[0]["digest"]["sha256"] == trace_digest.removeprefix("sha256:")

        signing = Ed25519PrivateKey.generate()
        env = sign_provenance(stmt, signing)
        recovered = verify_provenance_signature(env, signing.public_key())
        assert recovered == stmt
