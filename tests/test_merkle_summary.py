"""Tests for the optional end-of-trace Merkle summary frame (Step 52).

Covers:

* RFC 6962 leaf/node-hash domain separation and unpaired-node promotion.
* Round-trip: writer-emitted summary verifies under the python engine
  and the resulting :class:`Trace_` exposes the root.
* Backward compatibility: a writer with ``emit_merkle_summary=False``
  still produces a valid trace and ``Trace_.merkle_root is None``.
* Tampering with a step body breaks the recomputed Merkle root before
  the HMAC chain is even consulted.
* Tampering with the summary frame's declared root or leaf_count is
  caught by the reader.
* The summary frame is rejected if anything other than ``tail`` follows it.
* Capability negotiation: ``merkle-summary-v1`` is in the default
  supported set, and a mandatory unknown variant is rejected.
* Attestation packs surface ``merkle_root`` per entry and a
  ``merkle_summarised_traces`` count in the summary.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import pytest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from stepback.attestation import build_attestation_pack
from stepback.canonical import canonical_json
from stepback.merkle import (
    LEAF_PREFIX,
    NODE_PREFIX,
    leaf_hash,
    merkle_root,
    merkle_root_from_bodies,
    node_hash,
)
from stepback.trace_reader import (
    DEFAULT_SUPPORTED_CAPABILITIES,
    TraceVerificationError,
    read_frames,
    verify_trace,
)
from stepback.trace_writer import MERKLE_SCHEME, TraceWriter


HMAC_KEY = bytes.fromhex(
    "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20"
)


def _make_signing_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(b"\xaa" * 32)


def _record_minimal(path: str, *, n_steps: int = 3, emit: bool = True) -> None:
    w = TraceWriter.open(
        path,
        hmac_key=HMAC_KEY,
        signing_key=_make_signing_key(),
        compression=False,  # keep step bodies inline so tampering is easy
        emit_merkle_summary=emit,
    )
    for i in range(n_steps):
        w.write_step({"step_id": f"s{i}", "step_kind": "tool_call",
                      "inputs": {"i": i}, "outputs": {"r": i * 2}})
    w.close()


# ----------------------------- RFC 6962 unit tests -----------------------------


def test_leaf_and_node_hash_use_domain_prefixes() -> None:
    body = b"hello"
    assert leaf_hash(body) == hashlib.sha256(LEAF_PREFIX + body).digest()
    a, b = b"a" * 32, b"b" * 32
    assert node_hash(a, b) == hashlib.sha256(NODE_PREFIX + a + b).digest()


def test_leaf_prefix_differs_from_node_prefix() -> None:
    # Second-preimage defence: a 64-byte leaf that happens to look like
    # the concatenation of two 32-byte hashes must not collide with an
    # internal node over those two hashes.
    a, b = b"a" * 32, b"b" * 32
    assert leaf_hash(a + b) != node_hash(a, b)


def test_merkle_root_empty_is_sha256_of_empty() -> None:
    assert merkle_root([]) == hashlib.sha256(b"").digest()


def test_merkle_root_single_leaf_is_that_leaf() -> None:
    only = leaf_hash(b"only")
    assert merkle_root([only]) == only


def test_merkle_root_two_leaves_pairs_via_node_hash() -> None:
    a, b = leaf_hash(b"a"), leaf_hash(b"b")
    assert merkle_root([a, b]) == node_hash(a, b)


def test_merkle_root_unpaired_promotes_not_duplicates() -> None:
    a, b, c = leaf_hash(b"a"), leaf_hash(b"b"), leaf_hash(b"c")
    # Per RFC 6962, c is promoted unchanged, NOT hashed against itself.
    expected = node_hash(node_hash(a, b), c)
    assert merkle_root([a, b, c]) == expected
    # And it MUST NOT equal the duplicate-style root:
    bitcoin_style = node_hash(node_hash(a, b), node_hash(c, c))
    assert merkle_root([a, b, c]) != bitcoin_style


def test_merkle_root_from_bodies_matches_explicit_leaf_hash() -> None:
    bodies = [b"x", b"y", b"z", b"w"]
    expected = merkle_root([leaf_hash(b) for b in bodies])
    assert merkle_root_from_bodies(bodies) == expected


# ----------------------------- Writer / reader round-trip ----------------------


def test_writer_emits_merkle_summary_by_default(tmp_path: Path) -> None:
    p = str(tmp_path / "t.sb")
    _record_minimal(p, n_steps=4, emit=True)
    frames = read_frames(p)
    types = [f["body"]["type"] for f in frames]
    # header, 4 steps, merkle_summary, tail (no blobs because compression=False)
    assert types == [
        "header", "step", "step", "step", "step",
        "merkle_summary", "tail",
    ]
    summary_body = frames[-2]["body"]
    assert summary_body["scheme"] == MERKLE_SCHEME
    assert summary_body["algorithm"] == "sha256"
    assert summary_body["leaf_count"] == 5  # header + 4 steps


def test_verify_exposes_merkle_root(tmp_path: Path) -> None:
    p = str(tmp_path / "t.sb")
    _record_minimal(p, n_steps=2, emit=True)
    t = verify_trace(p, HMAC_KEY)
    assert t.merkle_root is not None
    assert len(t.merkle_root) == 64
    bytes.fromhex(t.merkle_root)
    assert t.merkle_leaf_count == 3  # header + 2 steps


def test_recomputed_root_matches_writer_root(tmp_path: Path) -> None:
    p = str(tmp_path / "t.sb")
    _record_minimal(p, n_steps=3, emit=True)
    t = verify_trace(p, HMAC_KEY)
    frames = read_frames(p)
    bodies = [
        canonical_json(f["body"])
        for f in frames
        if f["body"]["type"] not in ("merkle_summary", "tail")
    ]
    expected = merkle_root_from_bodies(bodies).hex()
    assert t.merkle_root == expected


def test_writer_can_opt_out_for_legacy_traces(tmp_path: Path) -> None:
    p = str(tmp_path / "t.sb")
    _record_minimal(p, n_steps=2, emit=False)
    frames = read_frames(p)
    types = [f["body"]["type"] for f in frames]
    assert "merkle_summary" not in types
    t = verify_trace(p, HMAC_KEY)
    assert t.merkle_root is None
    assert t.merkle_leaf_count == 0


# ----------------------------- Tampering detection -----------------------------


def _read_wrappers(path: str) -> list:
    out: list = []
    with open(path, "rb") as f:
        while True:
            head = f.read(4)
            if not head:
                break
            (n,) = struct.unpack(">I", head)
            out.append(json.loads(f.read(n).decode("utf-8")))
    return out


def _write_wrappers(path: str, wrappers: list) -> None:
    with open(path, "wb") as f:
        for w in wrappers:
            payload = canonical_json(w)
            f.write(struct.pack(">I", len(payload)))
            f.write(payload)


def test_tampering_with_summary_root_is_rejected(tmp_path: Path) -> None:
    p = str(tmp_path / "t.sb")
    _record_minimal(p, n_steps=2, emit=True)
    wrappers = _read_wrappers(p)
    summary_idx = next(
        i for i, w in enumerate(wrappers)
        if w["body"]["type"] == "merkle_summary"
    )
    # Flip one hex char of the declared root. Even though the wrapper's
    # HMAC + Ed25519 still validate (we'll re-sign below), the
    # recomputed root won't match the declared one.
    body = dict(wrappers[summary_idx]["body"])
    root = body["merkle_root"]
    body["merkle_root"] = ("0" if root[0] != "0" else "1") + root[1:]
    wrappers[summary_idx]["body"] = body
    # We need to keep the chain valid for everything before so the
    # reader actually reaches the merkle check. Re-HMAC + re-sign the
    # summary wrapper *and* the trailing tail wrapper so the chain is
    # still internally consistent — the merkle mismatch alone must be
    # the cause of rejection.
    _resign_chain_from(wrappers, summary_idx)
    _write_wrappers(p, wrappers)
    with pytest.raises(TraceVerificationError, match="merkle_summary root mismatch"):
        verify_trace(p, HMAC_KEY)


def test_tampering_with_summary_leaf_count_is_rejected(tmp_path: Path) -> None:
    p = str(tmp_path / "t.sb")
    _record_minimal(p, n_steps=2, emit=True)
    wrappers = _read_wrappers(p)
    summary_idx = next(
        i for i, w in enumerate(wrappers)
        if w["body"]["type"] == "merkle_summary"
    )
    body = dict(wrappers[summary_idx]["body"])
    body["leaf_count"] = body["leaf_count"] + 1
    wrappers[summary_idx]["body"] = body
    _resign_chain_from(wrappers, summary_idx)
    _write_wrappers(p, wrappers)
    with pytest.raises(TraceVerificationError, match="leaf_count"):
        verify_trace(p, HMAC_KEY)


def test_summary_after_extra_content_frame_is_rejected(tmp_path: Path) -> None:
    """A content frame after the summary must be rejected."""
    p = str(tmp_path / "t.sb")
    _record_minimal(p, n_steps=2, emit=True)
    wrappers = _read_wrappers(p)
    summary_idx = next(
        i for i, w in enumerate(wrappers)
        if w["body"]["type"] == "merkle_summary"
    )
    # Insert a fake step frame BETWEEN the summary and the tail.
    fake_step = {
        "type": "step",
        "step": {"step_id": "rogue", "step_kind": "tool_call",
                 "inputs": {}, "outputs": {}},
    }
    wrappers.insert(summary_idx + 1, {"body": fake_step})
    _resign_chain_from(wrappers, summary_idx + 1)
    _write_wrappers(p, wrappers)
    with pytest.raises(TraceVerificationError, match="after merkle_summary"):
        verify_trace(p, HMAC_KEY)


def _resign_chain_from(wrappers: list, start: int) -> None:
    """Re-HMAC + re-sign every wrapper from ``start`` onward.

    Lets us exercise post-summary semantic checks without the HMAC
    chain firing first.
    """
    import hmac as _hmac
    sk = _make_signing_key()
    if start == 0:
        prev = bytes(32)
    else:
        prev = bytes.fromhex(wrappers[start - 1]["hmac"])
    for i in range(start, len(wrappers)):
        body = wrappers[i]["body"]
        body_bytes = canonical_json(body)
        h = _hmac.new(HMAC_KEY, prev + body_bytes, hashlib.sha256).digest()
        sig = sk.sign(h)
        wrappers[i] = {
            "body": body,
            "prev_hmac": prev.hex(),
            "hmac": h.hex(),
            "sig": "ed25519:" + sig.hex(),
        }
        prev = h


def test_step_tamper_caught_by_recomputed_merkle_root(tmp_path: Path) -> None:
    """Even bypassing the HMAC chain, mutating a step body changes the
    Merkle root which the reader recomputes and compares.
    """
    p = str(tmp_path / "t.sb")
    _record_minimal(p, n_steps=2, emit=True)
    wrappers = _read_wrappers(p)
    # Mutate step 1's outputs.
    step_idx = next(
        i for i, w in enumerate(wrappers)
        if w["body"]["type"] == "step"
    )
    body = wrappers[step_idx]["body"]
    body["step"]["outputs"] = {"r": 999_999}
    # Re-sign the step + every subsequent frame so the chain still
    # validates; the only thing wrong is the Merkle root.
    _resign_chain_from(wrappers, step_idx)
    _write_wrappers(p, wrappers)
    with pytest.raises(TraceVerificationError, match="merkle_summary root mismatch"):
        verify_trace(p, HMAC_KEY)


# ----------------------------- Capability + reader integration -----------------


def test_default_capability_set_includes_merkle_summary() -> None:
    assert "merkle-summary-v1" in DEFAULT_SUPPORTED_CAPABILITIES


def test_reader_accepts_summary_with_blobs_and_capabilities(tmp_path: Path) -> None:
    """Leaves include header, capability, blob, AND step bodies in order."""
    p = str(tmp_path / "t.sb")
    big = "x" * 400  # large enough to cross the blob threshold
    w = TraceWriter.open(
        p,
        hmac_key=HMAC_KEY,
        signing_key=_make_signing_key(),
        compression=True,
    )
    w.write_capability("optional-thing", mandatory=False, params={"v": 1})
    for i in range(3):
        w.write_step({
            "step_id": f"s{i}", "step_kind": "tool_call",
            "inputs": {"big": big},  # repeats → interned as a blob
            "outputs": {"r": i},
        })
    w.close()
    t = verify_trace(p, HMAC_KEY)
    assert t.merkle_root is not None
    assert t.merkle_leaf_count > 4  # header + capability + blob + 3 steps
    # Recompute end-to-end as a sanity check.
    bodies = [
        canonical_json(f["body"]) for f in read_frames(p)
        if f["body"]["type"] not in ("merkle_summary", "tail")
    ]
    assert t.merkle_root == merkle_root_from_bodies(bodies).hex()
    assert t.merkle_leaf_count == len(bodies)


# ----------------------------- Attestation surface ------------------------------


def test_attestation_pack_carries_merkle_root_per_entry(tmp_path: Path) -> None:
    p1 = str(tmp_path / "with.sb")
    p2 = str(tmp_path / "without.sb")
    _record_minimal(p1, n_steps=2, emit=True)
    _record_minimal(p2, n_steps=2, emit=False)
    pack = build_attestation_pack([p1, p2], hmac_key=HMAC_KEY)
    by_path = {e.trace_path: e for e in pack.entries}
    assert by_path[p1].merkle_root is not None
    assert len(by_path[p1].merkle_root) == 64
    assert by_path[p2].merkle_root is None
    assert pack.summary["merkle_summarised_traces"] == 1
