"""Tests for stepback.transparency_log (Step 132).

Covers:
- RFC 6962 inclusion-proof correctness for tree sizes 1..8, every leaf index.
- verify_inclusion_proof: valid proofs return True.
- verify_inclusion_proof: wrong root_hash returns False.
- verify_inclusion_proof: tampered leaf_hash returns False.
- verify_inclusion_proof: tampered audit_path returns False.
- verify_inclusion_proof: malformed hex raises TransparencyLogError.
- verify_inclusion_proof: leaf_index out of range raises TransparencyLogError.
- verify_inclusion_proof: wrong audit_path length for single-leaf tree raises.
- TransparencyLog: empty log has no STH.
- TransparencyLog: append returns correct proofs; all verify True.
- TransparencyLog: multiple appends maintain consistent roots.
- TransparencyLog: canonical JSON canonicalization means key order doesn't affect proof.
- TransparencyLog: disk persistence (save/load round-trip).
- TransparencyLog: load from directory preserves leaf hashes.
- log_attestation_pack: embeds proof in pack; does not break verify_attestation_pack.
- log_attestation_pack: proof verifies offline.
- log_attestation_pack: pack without body_hash raises TransparencyLogError.
- log_incident_record: returns valid proof.
- log_benchmark_pack: returns valid proof; embeds in file when path given.
- InclusionProof: to_dict / from_dict round-trip.
- SignedTreeHead: to_dict / from_dict round-trip.
"""
from __future__ import annotations

import json
import os

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from stepback.transparency_log import (
    InclusionProof,
    SignedTreeHead,
    TransparencyLog,
    TransparencyLogError,
    log_attestation_pack,
    log_benchmark_pack,
    log_incident_record,
    verify_inclusion_proof,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_log(n: int) -> tuple[TransparencyLog, list[InclusionProof]]:
    """Build an in-memory log with ``n`` generic entries.  Returns (log, proofs)."""
    log = TransparencyLog(log_id="test-log")
    proofs: list[InclusionProof] = []
    for i in range(n):
        proof = log.append({"seq": i, "data": f"entry-{i}"})
        proofs.append(proof)
    return log, proofs


# ---------------------------------------------------------------------------
# RFC 6962 correctness: all proofs for sizes 1-8 verify
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", range(1, 9))
def test_all_proofs_verify_for_tree_size(n: int) -> None:
    """Every leaf in a tree of size n should produce a valid inclusion proof."""
    log = TransparencyLog(log_id="test")
    for i in range(n):
        log.append({"i": i})
    sth = log.get_signed_tree_head()
    assert sth is not None
    # Rebuild proofs for every leaf against the current root.
    from stepback.transparency_log import _inclusion_path
    from stepback.merkle import leaf_hash, merkle_root
    from stepback.canonical import canonical_json

    # Re-derive leaf hashes.
    leaf_hashes = [leaf_hash(canonical_json({"i": i})) for i in range(n)]
    root = merkle_root(leaf_hashes).hex()

    for m in range(n):
        path = _inclusion_path(m, leaf_hashes)
        proof = InclusionProof(
            leaf_index=m,
            tree_size=n,
            leaf_hash=leaf_hashes[m].hex(),
            audit_path=[p.hex() for p in path],
            root_hash=root,
            log_id="test",
            timestamp="",
        )
        assert verify_inclusion_proof(proof), f"proof failed for n={n} m={m}"


# ---------------------------------------------------------------------------
# verify_inclusion_proof: valid proof from append
# ---------------------------------------------------------------------------

def test_append_single_verify() -> None:
    log = TransparencyLog(log_id="test")
    proof = log.append({"hello": "world"})
    assert proof.tree_size == 1
    assert proof.leaf_index == 0
    assert len(proof.audit_path) == 0
    assert verify_inclusion_proof(proof)


def test_append_multiple_all_verify() -> None:
    """All previously returned proofs must still verify."""
    log, proofs = _make_log(7)
    for proof in proofs:
        # Each proof was issued against the tree state at that time.
        assert verify_inclusion_proof(proof)


# ---------------------------------------------------------------------------
# verify_inclusion_proof: tamper detection
# ---------------------------------------------------------------------------

def test_wrong_root_returns_false() -> None:
    log = TransparencyLog()
    proof = log.append({"x": 1})
    tampered = InclusionProof(
        leaf_index=proof.leaf_index,
        tree_size=proof.tree_size,
        leaf_hash=proof.leaf_hash,
        audit_path=proof.audit_path,
        root_hash="a" * 64,  # wrong root
        log_id=proof.log_id,
        timestamp=proof.timestamp,
    )
    assert not verify_inclusion_proof(tampered)


def test_wrong_leaf_hash_returns_false() -> None:
    log, proofs = _make_log(4)
    proof = proofs[2]
    tampered = InclusionProof(
        leaf_index=proof.leaf_index,
        tree_size=proof.tree_size,
        leaf_hash="b" * 64,  # wrong leaf
        audit_path=proof.audit_path,
        root_hash=proof.root_hash,
        log_id=proof.log_id,
        timestamp=proof.timestamp,
    )
    assert not verify_inclusion_proof(tampered)


def test_wrong_audit_path_returns_false() -> None:
    log, proofs = _make_log(4)
    proof = proofs[1]
    if not proof.audit_path:
        pytest.skip("no audit path to tamper")
    tampered_path = [("c" * 64)] + proof.audit_path[1:]
    tampered = InclusionProof(
        leaf_index=proof.leaf_index,
        tree_size=proof.tree_size,
        leaf_hash=proof.leaf_hash,
        audit_path=tampered_path,
        root_hash=proof.root_hash,
        log_id=proof.log_id,
        timestamp=proof.timestamp,
    )
    assert not verify_inclusion_proof(tampered)


# ---------------------------------------------------------------------------
# verify_inclusion_proof: error cases
# ---------------------------------------------------------------------------

def test_malformed_hex_leaf_raises() -> None:
    log = TransparencyLog()
    proof = log.append({"a": 1})
    bad = InclusionProof(
        leaf_index=0, tree_size=1,
        leaf_hash="not-hex",
        audit_path=[],
        root_hash=proof.root_hash,
    )
    with pytest.raises(TransparencyLogError, match="invalid hex"):
        verify_inclusion_proof(bad)


def test_malformed_hex_root_raises() -> None:
    log = TransparencyLog()
    proof = log.append({"a": 1})
    bad = InclusionProof(
        leaf_index=0, tree_size=1,
        leaf_hash=proof.leaf_hash,
        audit_path=[],
        root_hash="ZZZZ",
    )
    with pytest.raises(TransparencyLogError):
        verify_inclusion_proof(bad)


def test_leaf_index_out_of_range_raises() -> None:
    log = TransparencyLog()
    proof = log.append({"a": 1})
    bad = InclusionProof(
        leaf_index=5, tree_size=1,
        leaf_hash=proof.leaf_hash,
        audit_path=[],
        root_hash=proof.root_hash,
    )
    with pytest.raises(TransparencyLogError, match="out of range"):
        verify_inclusion_proof(bad)


def test_zero_tree_size_raises() -> None:
    bad = InclusionProof(
        leaf_index=0, tree_size=0,
        leaf_hash="a" * 64,
        audit_path=[],
        root_hash="b" * 64,
    )
    with pytest.raises(TransparencyLogError, match="positive"):
        verify_inclusion_proof(bad)


def test_single_leaf_nonempty_path_raises() -> None:
    log = TransparencyLog()
    proof = log.append({"a": 1})
    bad = InclusionProof(
        leaf_index=0, tree_size=1,
        leaf_hash=proof.leaf_hash,
        audit_path=["d" * 64],  # non-empty for single leaf tree
        root_hash=proof.root_hash,
    )
    with pytest.raises(TransparencyLogError, match="non-empty audit_path"):
        verify_inclusion_proof(bad)


# ---------------------------------------------------------------------------
# TransparencyLog state and metadata
# ---------------------------------------------------------------------------

def test_empty_log_no_sth() -> None:
    log = TransparencyLog()
    assert log.get_signed_tree_head() is None
    assert len(log) == 0


def test_log_len_grows() -> None:
    log, _ = _make_log(5)
    assert len(log) == 5


def test_sth_tree_size_grows() -> None:
    log = TransparencyLog()
    for i in range(1, 6):
        log.append({"i": i})
        assert log.get_signed_tree_head().tree_size == i


def test_canonical_json_key_order_independent() -> None:
    """dict with same content but different key order produces same leaf hash."""
    log = TransparencyLog()
    p1 = log.append({"a": 1, "b": 2})
    p2 = log.append({"b": 2, "a": 1})
    # Both entries were identical after canonicalization → same leaf hash.
    assert p1.leaf_hash == p2.leaf_hash


# ---------------------------------------------------------------------------
# Disk persistence
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path) -> None:
    log = TransparencyLog(log_id="disk-test")
    proofs = [log.append({"n": i}) for i in range(5)]
    save_dir = str(tmp_path / "log")
    log.save(save_dir)

    loaded = TransparencyLog.load(save_dir)
    assert len(loaded) == len(log)
    assert loaded.get_signed_tree_head().root_hash == log.get_signed_tree_head().root_hash
    # Verify proofs still check out against the same root.
    for proof in proofs:
        assert verify_inclusion_proof(proof)


def test_disk_backed_log_auto_persists(tmp_path) -> None:
    log_dir = str(tmp_path / "auto")
    log = TransparencyLog(log_id="auto-persist", directory=log_dir)
    log.append({"x": 1})
    log.append({"x": 2})

    assert os.path.isfile(os.path.join(log_dir, "entries.jsonl"))
    assert os.path.isfile(os.path.join(log_dir, "sth_history.jsonl"))

    # Load fresh from disk.
    log2 = TransparencyLog.load(log_dir)
    assert len(log2) == 2
    assert log2.get_signed_tree_head().root_hash == log.get_signed_tree_head().root_hash


def test_disk_backed_log_append_after_load(tmp_path) -> None:
    log_dir = str(tmp_path / "append-after")
    log1 = TransparencyLog(log_id="al", directory=log_dir)
    proof1 = log1.append({"seq": 0})
    log1.append({"seq": 1})

    log2 = TransparencyLog(log_id="al", directory=log_dir)
    assert len(log2) == 2
    # Append a new entry; proof should verify.
    proof3 = log2.append({"seq": 2})
    assert verify_inclusion_proof(proof3)


# ---------------------------------------------------------------------------
# log_attestation_pack
# ---------------------------------------------------------------------------

def _build_and_write_pack(tmp_path) -> tuple[str, str]:
    """Record a tiny trace and build+write an attestation pack; return (pack_path, body_hash)."""
    from stepback import RecorderKey, record
    from stepback.testing import run_recorded_agent
    from stepback.attestation import build_attestation_pack, write_attestation_pack

    key = RecorderKey.fresh()
    trace_path = str(tmp_path / "trace.sb")
    with record(trace_path, key=key) as rec:
        run_recorded_agent(rec)

    signing_key = Ed25519PrivateKey.generate()
    pub_hex = signing_key.public_key().public_bytes_raw().hex()
    pack = build_attestation_pack(
        [trace_path],
        hmac_key=key.hmac_key,
        attestor_signing_key=signing_key,
    )
    pack_path = str(tmp_path / "pack.json")
    body_hash = write_attestation_pack(pack, pack_path, signing_key=signing_key)
    return pack_path, body_hash


def test_log_attestation_pack_embeds_proof(tmp_path) -> None:
    pack_path, body_hash = _build_and_write_pack(tmp_path)
    log = TransparencyLog(log_id="attest-log")
    proof = log_attestation_pack(pack_path, log)

    assert proof.entry_type == "attestation_pack"
    assert verify_inclusion_proof(proof)

    with open(pack_path) as f:
        on_disk = json.load(f)
    assert "transparency_log_proof" in on_disk
    embedded = InclusionProof.from_dict(on_disk["transparency_log_proof"])
    assert verify_inclusion_proof(embedded)


def test_log_attestation_pack_does_not_break_verify(tmp_path) -> None:
    """Embedding the proof must not invalidate the pack signature."""
    from stepback.attestation import verify_attestation_pack
    from stepback import RecorderKey, record
    from stepback.testing import run_recorded_agent
    from stepback.attestation import build_attestation_pack, write_attestation_pack

    key = RecorderKey.fresh()
    trace_path = str(tmp_path / "trace.sb")
    with record(trace_path, key=key) as rec:
        run_recorded_agent(rec)
    signing_key = Ed25519PrivateKey.generate()
    pack = build_attestation_pack(
        [trace_path],
        hmac_key=key.hmac_key,
        attestor_signing_key=signing_key,
    )
    pack_path = str(tmp_path / "pack.json")
    write_attestation_pack(pack, pack_path, signing_key=signing_key)

    log = TransparencyLog()
    log_attestation_pack(pack_path, log)

    # Must still verify after proof embedding (raises on failure).
    pub_hex = signing_key.public_key().public_bytes_raw().hex()
    data = verify_attestation_pack(
        pack_path, expected_public_key=f"ed25519:{pub_hex}"
    )
    assert data.get("magic") == "stepback/.pack"


def test_log_attestation_pack_no_body_hash_raises(tmp_path) -> None:
    pack_path = str(tmp_path / "bad.json")
    with open(pack_path, "w") as f:
        json.dump({"magic": "stepback/.pack"}, f)
    log = TransparencyLog()
    with pytest.raises(TransparencyLogError, match="body_hash"):
        log_attestation_pack(pack_path, log)


# ---------------------------------------------------------------------------
# log_incident_record
# ---------------------------------------------------------------------------

def test_log_incident_record_valid_proof() -> None:
    log = TransparencyLog(log_id="incident-log")
    record = {
        "incident_id": "INC-2026-001",
        "severity": "high",
        "summary": "Replay mismatch detected in batch run",
        "affected_traces": ["abc123", "def456"],
    }
    proof = log_incident_record(record, log)
    assert proof.entry_type == "incident_record"
    assert proof.tree_size == 1
    assert verify_inclusion_proof(proof)


def test_log_incident_record_deterministic() -> None:
    """Same record logged twice produces same leaf hash."""
    log = TransparencyLog()
    record = {"id": "INC-001", "detail": "test"}
    p1 = log_incident_record(record, log)
    p2 = log_incident_record(record, log)
    assert p1.leaf_hash == p2.leaf_hash
    assert p1.leaf_index == 0
    assert p2.leaf_index == 1


# ---------------------------------------------------------------------------
# log_benchmark_pack
# ---------------------------------------------------------------------------

def test_log_benchmark_pack_no_file() -> None:
    log = TransparencyLog()
    artifact = {"corpus": "swe-bench-verified", "score": 0.83, "model": "gpt-4o"}
    proof = log_benchmark_pack(artifact, log)
    assert proof.entry_type == "benchmark_pack"
    assert verify_inclusion_proof(proof)


def test_log_benchmark_pack_with_file(tmp_path) -> None:
    artifact = {"corpus": "gaia", "score": 0.72, "model": "claude-3-opus"}
    pack_path = str(tmp_path / "bench.json")
    with open(pack_path, "w") as f:
        json.dump(artifact, f)

    log = TransparencyLog()
    proof = log_benchmark_pack(artifact, log, pack_path=pack_path)
    assert verify_inclusion_proof(proof)

    with open(pack_path) as f:
        on_disk = json.load(f)
    assert "transparency_log_proof" in on_disk
    embedded = InclusionProof.from_dict(on_disk["transparency_log_proof"])
    assert verify_inclusion_proof(embedded)


# ---------------------------------------------------------------------------
# InclusionProof / SignedTreeHead dataclass round-trips
# ---------------------------------------------------------------------------

def test_inclusion_proof_round_trip() -> None:
    log = TransparencyLog()
    proof = log.append({"round": "trip"})
    d = proof.to_dict()
    restored = InclusionProof.from_dict(d)
    assert restored.leaf_index == proof.leaf_index
    assert restored.tree_size == proof.tree_size
    assert restored.leaf_hash == proof.leaf_hash
    assert restored.root_hash == proof.root_hash
    assert verify_inclusion_proof(restored)


def test_signed_tree_head_round_trip() -> None:
    sth = SignedTreeHead(
        tree_size=42,
        root_hash="a" * 64,
        timestamp="2026-05-12T00:00:00Z",
        log_id="test",
        signature="b" * 128,
    )
    d = sth.to_dict()
    restored = SignedTreeHead.from_dict(d)
    assert restored.tree_size == 42
    assert restored.root_hash == "a" * 64
    assert restored.log_id == "test"


# ---------------------------------------------------------------------------
# Mixed entry types in the same log
# ---------------------------------------------------------------------------

def test_mixed_entry_types_all_verify() -> None:
    log = TransparencyLog(log_id="mixed")
    p1 = log_incident_record({"id": "INC-A"}, log)
    p2 = log_benchmark_pack({"corpus": "x", "score": 0.5}, log)
    p3 = log.append({"generic": True})
    p4 = log_incident_record({"id": "INC-B"}, log)

    for i, p in enumerate([p1, p2, p3, p4]):
        assert verify_inclusion_proof(p), f"proof {i} failed"
    assert log.get_signed_tree_head().tree_size == 4


# ---------------------------------------------------------------------------
# Repr smoke test
# ---------------------------------------------------------------------------

def test_log_repr_smoke() -> None:
    log = TransparencyLog(log_id="repr-test")
    assert "empty" in repr(log)
    log.append({"x": 1})
    assert "repr-test" in repr(log)
    assert "size=1" in repr(log)
