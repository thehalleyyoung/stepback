"""RFC 6962 / Rekor-style transparency log for attestation packs and incident records.

An entry is content-addressed by SHA-256 of its canonical JSON body.  The log
is append-only: each :meth:`TransparencyLog.append` returns an
:class:`InclusionProof` that proves the new leaf is at ``leaf_index`` in a
tree of size ``tree_size`` whose root is ``root_hash``.

The inclusion-proof algorithm and verification follow RFC 6962 §2.1.2 / §2.1.3
exactly — the same conventions used by :mod:`stepback.merkle` so roots can be
cross-verified.  Specifically:

* leaf hash:  ``SHA-256(0x00 || leaf_bytes)``
* node hash:  ``SHA-256(0x01 || left || right)``
* unpaired tail nodes at any level are *promoted* (not duplicated)

File-backed log
---------------
A :class:`TransparencyLog` can persist to a directory.  It writes:

* ``<dir>/entries.jsonl`` – one JSON line per entry:
  ``{"index": N, "leaf_hash_hex": "...", "entry": {...}}``
* ``<dir>/sth_history.jsonl`` – one JSON line per committed
  :class:`SignedTreeHead` appended after each :meth:`~TransparencyLog.append`.

Public API
----------
* :class:`InclusionProof`       – leaf-to-root audit path + tree metadata.
* :class:`SignedTreeHead`        – signed commitment to log state.
* :class:`TransparencyLog`       – in-memory log; optional disk persistence.
* :func:`verify_inclusion_proof` – offline verifier, no log instance needed.
* :func:`log_attestation_pack`   – submit a pack to a log; embed proof in the
                                   on-disk JSON as unsigned envelope field.
* :func:`log_incident_record`    – submit an incident-record dict.
* :func:`log_benchmark_pack`     – submit a benchmark-pack artifact dict.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .canonical import canonical_json, sha256_hex
from .merkle import LEAF_PREFIX, NODE_PREFIX, leaf_hash, merkle_root

__all__ = [
    "InclusionProof",
    "SignedTreeHead",
    "TransparencyLog",
    "TransparencyLogError",
    "verify_inclusion_proof",
    "log_attestation_pack",
    "log_incident_record",
    "log_benchmark_pack",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_FORMAT_VERSION = 1
_HASH_ALG = "sha256"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TransparencyLogError(Exception):
    """Raised on inclusion-proof verification failures or log corruption."""

    #: Canonical error code; see :mod:`stepback.errors` for details.
    code: str = "SB505"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class InclusionProof:
    """RFC 6962 inclusion proof for a single log entry.

    Attributes
    ----------
    leaf_index:
        Zero-based index of the entry in the log.
    tree_size:
        Size of the tree when this proof was issued (number of leaves).
    leaf_hash:
        Hex-encoded RFC 6962 leaf hash (``SHA-256(0x00 || leaf_bytes)``).
    audit_path:
        Ordered list of hex-encoded sibling hashes from the leaf to the
        tree root (RFC 6962 §2.1.3 order).
    root_hash:
        Hex-encoded Merkle root at the time of logging.  Callers can use
        this to skip fetching a SignedTreeHead for simple offline checks.
    log_id:
        Opaque identifier for the log that issued this proof.
    timestamp:
        UTC ISO 8601 timestamp of the ``SignedTreeHead`` that produced this
        proof.
    entry_type:
        Kind of artifact logged (``"attestation_pack"``, ``"incident_record"``,
        ``"benchmark_pack"``, or ``"generic"``).
    """

    leaf_index: int
    tree_size: int
    leaf_hash: str  # hex (no prefix)
    audit_path: List[str]  # list of hex hashes (no prefix)
    root_hash: str  # hex (no prefix)
    log_id: str = "stepback/local"
    timestamp: str = ""
    entry_type: str = "generic"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InclusionProof":
        return cls(**d)


@dataclass
class SignedTreeHead:
    """Signed commitment to a log state.

    In a production log this would carry an Ed25519 signature from the log
    operator; here the ``signature`` field is optional and left empty for
    software-simulator use.

    Attributes
    ----------
    tree_size:
        Number of leaves committed to.
    root_hash:
        Hex-encoded Merkle root.
    timestamp:
        UTC ISO 8601 timestamp when this STH was produced.
    log_id:
        Identifies the log.
    signature:
        Optional hex-encoded Ed25519 signature over the canonical JSON of
        ``{tree_size, root_hash, timestamp, log_id}``.  Empty string when
        running without a signing key.
    """

    tree_size: int
    root_hash: str
    timestamp: str
    log_id: str = "stepback/local"
    signature: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SignedTreeHead":
        return cls(**d)


# ---------------------------------------------------------------------------
# RFC 6962 proof helpers
# ---------------------------------------------------------------------------

def _largest_power_of_2_less_than(n: int) -> int:
    """Return the largest power of 2 strictly less than ``n`` (n >= 2)."""
    k = 1
    while k * 2 < n:
        k *= 2
    return k


def _inclusion_path(leaf_index: int, leaves: List[bytes]) -> List[bytes]:
    """Recursive RFC 6962 audit path for ``leaves[leaf_index]``.

    ``leaves`` are already-hashed leaf digests (32 bytes each).
    Returns a list of sibling hashes ordered from bottom (nearest to the
    leaf) to top (nearest to the root).
    """
    n = len(leaves)
    if n == 1:
        return []
    k = _largest_power_of_2_less_than(n)
    if leaf_index < k:
        left_path = _inclusion_path(leaf_index, leaves[:k])
        right_root = merkle_root(leaves[k:])
        return left_path + [right_root]
    else:
        left_root = merkle_root(leaves[:k])
        right_path = _inclusion_path(leaf_index - k, leaves[k:])
        return right_path + [left_root]


def _recompute_root(
    leaf_hash_bytes: bytes,
    leaf_index: int,
    tree_size: int,
    audit_path_bytes: List[bytes],
) -> bytes:
    """Recompute the Merkle root from a proof.  RFC 6962 §2.1.3 recursive form."""
    n = tree_size
    m = leaf_index
    r = leaf_hash_bytes
    path = list(audit_path_bytes)  # shallow copy; we consume from front

    def _inner(leaf_node: bytes, idx: int, sz: int, siblings: List[bytes]) -> bytes:
        if sz == 1:
            return leaf_node
        k = _largest_power_of_2_less_than(sz)
        if idx < k:
            left = _inner(leaf_node, idx, k, siblings[:-1])
            right = siblings[-1]
        else:
            left = siblings[-1]
            right = _inner(leaf_node, idx - k, sz - k, siblings[:-1])
        return hashlib.sha256(NODE_PREFIX + left + right).digest()

    return _inner(r, m, n, path)


# ---------------------------------------------------------------------------
# Public verifier
# ---------------------------------------------------------------------------

def verify_inclusion_proof(proof: InclusionProof) -> bool:
    """Verify ``proof`` offline without access to the log.

    Returns ``True`` if the proof is internally consistent (the audit path
    hashes back to ``proof.root_hash``).  Raises
    :class:`TransparencyLogError` if the proof is malformed.
    """
    if proof.tree_size <= 0:
        raise TransparencyLogError("tree_size must be positive")
    if not (0 <= proof.leaf_index < proof.tree_size):
        raise TransparencyLogError(
            f"leaf_index {proof.leaf_index} out of range [0, {proof.tree_size})"
        )

    def _decode_hex(h: str, label: str) -> bytes:
        try:
            b = bytes.fromhex(h)
        except ValueError:
            raise TransparencyLogError(f"{label}: invalid hex: {h!r}")
        if len(b) != 32:
            raise TransparencyLogError(f"{label}: expected 32 bytes, got {len(b)}")
        return b

    lh = _decode_hex(proof.leaf_hash, "leaf_hash")
    path_bytes = [_decode_hex(p, f"audit_path[{i}]") for i, p in enumerate(proof.audit_path)]
    expected_root = _decode_hex(proof.root_hash, "root_hash")

    if proof.tree_size == 1:
        if path_bytes:
            raise TransparencyLogError("non-empty audit_path for single-leaf tree")
        return lh == expected_root

    recomputed = _recompute_root(lh, proof.leaf_index, proof.tree_size, path_bytes)
    return recomputed == expected_root


# ---------------------------------------------------------------------------
# Transparency log
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


class TransparencyLog:
    """In-memory RFC 6962 transparency log with optional disk persistence.

    Parameters
    ----------
    log_id:
        Opaque identifier embedded in all proofs and STHs from this log.
    directory:
        Optional path to a directory where the log is persisted.  The
        directory is created if absent.  Pass ``None`` for an ephemeral
        in-memory-only log.
    """

    def __init__(self, log_id: str = "stepback/local", directory: Optional[str] = None) -> None:
        self._log_id = log_id
        self._directory = directory
        self._leaf_hashes: List[bytes] = []
        self._sth_history: List[SignedTreeHead] = []

        if directory is not None and os.path.isdir(directory):
            self._load_from_dir(directory)

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append(
        self,
        entry_body: Dict[str, Any],
        entry_type: str = "generic",
    ) -> InclusionProof:
        """Append ``entry_body`` to the log and return an inclusion proof.

        ``entry_body`` is serialised to canonical JSON before hashing so the
        proof is independent of Python dict ordering.
        """
        body_bytes = canonical_json(entry_body)
        lh = leaf_hash(body_bytes)
        leaf_index = len(self._leaf_hashes)
        self._leaf_hashes.append(lh)

        # Compute the proof against the *new* tree (includes the new leaf).
        tree_size = len(self._leaf_hashes)
        root_bytes = merkle_root(self._leaf_hashes)
        root_hex = root_bytes.hex()
        ts = _utc_now_iso()

        audit_path_bytes = _inclusion_path(leaf_index, self._leaf_hashes)
        proof = InclusionProof(
            leaf_index=leaf_index,
            tree_size=tree_size,
            leaf_hash=lh.hex(),
            audit_path=[p.hex() for p in audit_path_bytes],
            root_hash=root_hex,
            log_id=self._log_id,
            timestamp=ts,
            entry_type=entry_type,
        )

        sth = SignedTreeHead(
            tree_size=tree_size,
            root_hash=root_hex,
            timestamp=ts,
            log_id=self._log_id,
        )
        self._sth_history.append(sth)

        if self._directory is not None:
            self._persist_entry(leaf_index, lh, entry_body)
            self._persist_sth(sth)

        return proof

    # ------------------------------------------------------------------
    # Signed tree head
    # ------------------------------------------------------------------

    def get_signed_tree_head(self) -> Optional[SignedTreeHead]:
        """Return the latest SignedTreeHead, or ``None`` for an empty log."""
        if not self._sth_history:
            return None
        return self._sth_history[-1]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_entry(self, index: int, lh: bytes, entry: Dict[str, Any]) -> None:
        assert self._directory is not None
        os.makedirs(self._directory, exist_ok=True)
        line = json.dumps(
            {"index": index, "leaf_hash_hex": lh.hex(), "entry": entry},
            sort_keys=True,
            separators=(",", ":"),
        )
        with open(os.path.join(self._directory, "entries.jsonl"), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _persist_sth(self, sth: SignedTreeHead) -> None:
        assert self._directory is not None
        os.makedirs(self._directory, exist_ok=True)
        line = json.dumps(sth.to_dict(), sort_keys=True, separators=(",", ":"))
        with open(os.path.join(self._directory, "sth_history.jsonl"), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _load_from_dir(self, directory: str) -> None:
        """Reload state from a persisted directory."""
        entries_path = os.path.join(directory, "entries.jsonl")
        sth_path = os.path.join(directory, "sth_history.jsonl")
        if os.path.exists(entries_path):
            with open(entries_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    row = json.loads(raw_line)
                    lh_bytes = bytes.fromhex(row["leaf_hash_hex"])
                    if len(lh_bytes) != 32:
                        raise TransparencyLogError(
                            f"Corrupt entries.jsonl: bad leaf_hash at index {row.get('index')}"
                        )
                    self._leaf_hashes.append(lh_bytes)
        if os.path.exists(sth_path):
            with open(sth_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    self._sth_history.append(SignedTreeHead.from_dict(json.loads(raw_line)))

    def save(self, directory: str) -> None:
        """Write the full log to ``directory`` (overwrites existing files)."""
        os.makedirs(directory, exist_ok=True)
        entries_path = os.path.join(directory, "entries.jsonl")
        sth_path = os.path.join(directory, "sth_history.jsonl")
        # Entries file — we don't have the original entry bodies if loaded
        # from an existing directory (we only keep leaf hashes in memory).
        # To avoid data loss, only write the entries file if the log was
        # built entirely in-memory (i.e. _directory was None initially).
        if self._directory is None:
            # In-memory log: we lose original entry bodies here because we
            # don't cache them.  Write a placeholder per entry.
            with open(entries_path, "w", encoding="utf-8") as f:
                for i, lh in enumerate(self._leaf_hashes):
                    line = json.dumps(
                        {"index": i, "leaf_hash_hex": lh.hex()},
                        sort_keys=True, separators=(",", ":"),
                    )
                    f.write(line + "\n")
        if self._sth_history:
            with open(sth_path, "w", encoding="utf-8") as f:
                for sth in self._sth_history:
                    f.write(json.dumps(sth.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")
        self._directory = directory

    @classmethod
    def load(cls, directory: str, log_id: str = "stepback/local") -> "TransparencyLog":
        """Load a log from ``directory``.

        The ``log_id`` is taken from the most-recent STH if available;
        the parameter is used as fallback.
        """
        inst = cls(log_id=log_id, directory=None)
        inst._load_from_dir(directory)
        inst._directory = directory
        if inst._sth_history:
            log_id = inst._sth_history[-1].log_id
            inst._log_id = log_id
        return inst

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._leaf_hashes)

    def __repr__(self) -> str:
        sth = self.get_signed_tree_head()
        root = sth.root_hash[:16] + "..." if sth else "(empty)"
        return f"TransparencyLog(log_id={self._log_id!r}, size={len(self)}, root={root!r})"


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def _pack_log_entry(pack_path: str, body_hash: str, entry_type: str) -> Dict[str, Any]:
    """Build the canonical log-entry dict for an attestation or benchmark pack."""
    return {
        "type": entry_type,
        "pack_path": os.path.basename(pack_path),  # no absolute paths in logs
        "body_hash": body_hash,
        "format_version": LOG_FORMAT_VERSION,
    }


def log_attestation_pack(
    pack_path: str,
    log: TransparencyLog,
) -> InclusionProof:
    """Log an attestation pack to ``log`` and embed the proof back into the file.

    The logged entry is::

        {"type": "attestation_pack", "pack_path": "<basename>",
         "body_hash": "<sha256 hex from pack>", "format_version": 1}

    The ``body_hash`` is the one already computed by
    :func:`~stepback.attestation.write_attestation_pack` and present in the
    on-disk JSON — it is the SHA-256 of the canonical signed body, which is
    independent of the unsigned envelope fields (``signature``,
    ``transparency_log_proof``, etc.).

    The resulting :class:`InclusionProof` is written back into the pack file
    as a new top-level field ``transparency_log_proof`` (an unsigned envelope
    field, analogous to ``signature``).  This does **not** invalidate the pack
    signature.

    Returns the :class:`InclusionProof`.
    """
    with open(pack_path, "r", encoding="utf-8") as f:
        pack_data = json.load(f)

    body_hash: str = pack_data.get("body_hash", "")
    if not body_hash:
        raise TransparencyLogError(
            f"pack at {pack_path!r} has no 'body_hash' field; "
            "write it with write_attestation_pack first"
        )

    entry = _pack_log_entry(pack_path, body_hash, "attestation_pack")
    proof = log.append(entry, entry_type="attestation_pack")

    pack_data["transparency_log_proof"] = proof.to_dict()
    with open(pack_path, "w", encoding="utf-8") as f:
        json.dump(pack_data, f, sort_keys=True, indent=2, ensure_ascii=False)
        f.write("\n")

    return proof


def log_incident_record(
    record: Dict[str, Any],
    log: TransparencyLog,
) -> InclusionProof:
    """Append ``record`` to ``log`` as an incident-record entry.

    The logged entry is::

        {"type": "incident_record", "incident": <record dict>,
         "format_version": 1}

    Returns the :class:`InclusionProof`.
    """
    entry = {
        "type": "incident_record",
        "incident": record,
        "format_version": LOG_FORMAT_VERSION,
    }
    return log.append(entry, entry_type="incident_record")


def log_benchmark_pack(
    artifact: Dict[str, Any],
    log: TransparencyLog,
    *,
    pack_path: Optional[str] = None,
) -> InclusionProof:
    """Append a benchmark-pack artifact to ``log``.

    Parameters
    ----------
    artifact:
        The benchmark pack dict (e.g. a leaderboard submission JSON).
    log:
        The log to append to.
    pack_path:
        Optional file path; if given, the proof is written back into the
        file as ``transparency_log_proof`` (unsigned envelope field).

    Returns the :class:`InclusionProof`.
    """
    body_hash = sha256_hex(canonical_json(artifact))
    entry = {
        "type": "benchmark_pack",
        "body_hash": body_hash,
        "format_version": LOG_FORMAT_VERSION,
    }
    if pack_path is not None:
        entry["pack_path"] = os.path.basename(pack_path)
    proof = log.append(entry, entry_type="benchmark_pack")

    if pack_path is not None:
        data = dict(artifact)
        data["transparency_log_proof"] = proof.to_dict()
        with open(pack_path, "w", encoding="utf-8") as f:
            json.dump(data, f, sort_keys=True, indent=2, ensure_ascii=False)
            f.write("\n")

    return proof
