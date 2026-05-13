"""Storage-compression benchmark (Step 122).

Measures on-disk byte cost for a synthetic corpus of traces across six
encoding strategies and two additive overhead metrics.

Encoding formats
----------------
raw_json
    Canonical UTF-8 JSON (``stepback.canonical.canonical_json``) of each
    decoded step body, concatenated with no delimiters.  This is the
    *logical payload* baseline with zero cryptographic or framing overhead.

sb_v1
    Actual ``.sb`` files as written by the production recorder:
    length-prefixed frames, chained HMAC-SHA256, Ed25519 receipts,
    per-step gzip+base64 compression, and content-addressed blob
    deduplication.  This is the full on-wire format.

cbor
    Canonical CBOR (RFC 8949 §4.2, ``stepback.canonical_cbor.encode``)
    of each decoded step body, concatenated.  Candidate encoding for
    ``.sb`` v2 (Steps 43–44).

gzip_of_json
    ``gzip.compress`` applied to the raw-JSON corpus bytes for a given
    trace in isolation.  Represents the gain from simple deflate without
    the HMAC/framing overhead of ``.sb``.

lzma_of_json
    ``lzma.compress`` applied to per-trace raw-JSON bytes (LZMA2/XZ).
    Used as a stdlib substitute for zstd, which is not a Python standard
    library module and is not in the project's declared dependencies.
    Results are labelled ``lzma`` and should not be interpreted as zstd
    numbers.

deduped_json
    Content-addressed object-store simulation: unique step bodies (keyed
    by SHA-256 of their canonical JSON) are stored once, and a per-trace
    step-index maps each ``(trace_id, step_id)`` pair to its content
    hash.  Total size is ``unique_content_bytes + index_bytes``.  Captures
    the cross-trace saving that a blob-dedup store would achieve.

Additive overhead metrics
--------------------------
query_index_bytes
    Byte length of a compact JSON query-index mapping each trace to a
    list of ``{step_id, inputs_hash, outputs_hash, step_kind}`` records.
    This is the *additional* storage cost on top of any encoding format
    if a query-optimised secondary index is maintained.

sb_v1_plus_query_index_bytes
    ``sb_v1 total_bytes + query_index_bytes``.

Note on zstd
------------
The step description mentions "zstd" as one of the candidate encodings.
zstd (python-zstandard) is not in the Python standard library and is not
a declared dependency of this project.  This benchmark reports LZMA via
``lzma.compress`` as the closest available stdlib alternative.  A future
step can add an optional ``zstd`` format gated behind
``try: import zstd`` when the dependency is adopted.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import lzma
import os
import random
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..canonical import canonical_json
from ..canonical_cbor import canonical_cbor as cbor_encode
from ..recorder import RecorderKey, record
from ..trace_reader import (
    _decode_blob,
    _decode_gz_step,
    _has_blob_ref,
    _materialise,
    read_frames,
)
from .replay_caching import (
    SyntheticTrace,
    _bench_llm,
    _bench_tool,
    _bench_router,
)


# ---------------------------------------------------------------------------
# Per-format result
# ---------------------------------------------------------------------------


@dataclass
class FormatResult:
    """Size statistics for one encoding strategy across the whole corpus."""

    name: str
    """Short identifier used as the dict key (e.g. ``"sb_v1"``)."""

    total_bytes: int
    """Sum of encoded bytes across all traces in the corpus."""

    bytes_per_trace: float
    """Mean bytes per trace."""

    bytes_per_step: float
    """Mean bytes per *actual recorded step* (not requested n_steps)."""

    ratio_vs_raw_json: float
    """``total_bytes / raw_json_total_bytes``.  Values < 1 mean smaller than
    canonical JSON; values > 1 mean larger (e.g. due to HMAC/frame overhead
    for small traces).
    """

    encode_time_ms: float
    """Wall-clock time (ms) to produce all bytes for this format in ``run``."""

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "total_bytes": self.total_bytes,
            "bytes_per_trace": self.bytes_per_trace,
            "bytes_per_step": self.bytes_per_step,
            "ratio_vs_raw_json": self.ratio_vs_raw_json,
            "encode_time_ms": self.encode_time_ms,
        }

    @classmethod
    def from_json(cls, d: dict) -> "FormatResult":
        return cls(
            name=d["name"],
            total_bytes=d["total_bytes"],
            bytes_per_trace=d["bytes_per_trace"],
            bytes_per_step=d["bytes_per_step"],
            ratio_vs_raw_json=d["ratio_vs_raw_json"],
            encode_time_ms=d.get("encode_time_ms", 0.0),
        )


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------


@dataclass
class StorageCompressionResult:
    """Aggregated storage-compression benchmark result.

    ``formats`` contains one :class:`FormatResult` for each encoding
    strategy (``raw_json``, ``sb_v1``, ``cbor``, ``gzip_of_json``,
    ``lzma_of_json``, ``deduped_json``).

    ``query_index_bytes`` is the *additional* overhead of maintaining a
    secondary query index on top of any base format.
    """

    n_traces: int
    """Number of synthetic traces in the benchmark corpus."""

    n_steps_requested: int
    """``n_steps`` argument passed to :func:`run`."""

    actual_step_count: int
    """Total recorded steps across all traces (may differ from
    ``n_traces * n_steps_requested`` due to parallel-block emission)."""

    formats: Dict[str, FormatResult]
    """Per-format results keyed by format name."""

    # Additive overhead metrics.
    query_index_bytes: int
    """Byte length of a standalone JSON query-index for this corpus."""

    sb_v1_plus_query_index_bytes: int
    """``.sb v1`` total bytes plus query-index bytes."""

    query_index_overhead_ratio_vs_sb_v1: float
    """``query_index_bytes / sb_v1.total_bytes``."""

    # Dedup accounting breakdown.
    deduped_unique_content_bytes: int
    """Unique step-body bytes stored in the content-addressed object store."""

    deduped_index_bytes: int
    """Step-to-hash index bytes for the content-addressed layout."""

    total_wall_time_ms: float
    """Total elapsed time for the benchmark run (ms)."""

    def summary_line(self) -> str:
        """One-line human-readable summary."""
        raw = self.formats.get("raw_json")
        sb = self.formats.get("sb_v1")
        gz = self.formats.get("gzip_of_json")
        lz = self.formats.get("lzma_of_json")
        raw_b = raw.total_bytes if raw else 0
        sb_b = sb.total_bytes if sb else 0
        gz_b = gz.total_bytes if gz else 0
        lz_b = lz.total_bytes if lz else 0
        return (
            f"storage-compression: {self.n_traces} traces × "
            f"~{self.n_steps_requested} steps  "
            f"raw_json={raw_b:,}B  "
            f"sb_v1={sb_b:,}B({sb_b/max(raw_b,1):.2f}×)  "
            f"gzip={gz_b:,}B  "
            f"lzma={lz_b:,}B  "
            f"qi+{self.query_index_bytes:,}B"
        )

    def to_json(self) -> dict:
        return {
            "n_traces": self.n_traces,
            "n_steps_requested": self.n_steps_requested,
            "actual_step_count": self.actual_step_count,
            "formats": {k: v.to_json() for k, v in self.formats.items()},
            "query_index_bytes": self.query_index_bytes,
            "sb_v1_plus_query_index_bytes": self.sb_v1_plus_query_index_bytes,
            "query_index_overhead_ratio_vs_sb_v1": self.query_index_overhead_ratio_vs_sb_v1,
            "deduped_unique_content_bytes": self.deduped_unique_content_bytes,
            "deduped_index_bytes": self.deduped_index_bytes,
            "total_wall_time_ms": self.total_wall_time_ms,
        }

    @classmethod
    def from_json(cls, d: dict) -> "StorageCompressionResult":
        return cls(
            n_traces=d["n_traces"],
            n_steps_requested=d["n_steps_requested"],
            actual_step_count=d["actual_step_count"],
            formats={k: FormatResult.from_json(v) for k, v in d["formats"].items()},
            query_index_bytes=d["query_index_bytes"],
            sb_v1_plus_query_index_bytes=d["sb_v1_plus_query_index_bytes"],
            query_index_overhead_ratio_vs_sb_v1=d["query_index_overhead_ratio_vs_sb_v1"],
            deduped_unique_content_bytes=d["deduped_unique_content_bytes"],
            deduped_index_bytes=d["deduped_index_bytes"],
            total_wall_time_ms=d.get("total_wall_time_ms", 0.0),
        )


# ---------------------------------------------------------------------------
# Step extraction helpers
# ---------------------------------------------------------------------------


def _extract_steps(sb_path: str) -> List[dict]:
    """Return the list of decoded logical step dicts from a ``.sb`` file.

    Materialises blob references and decompresses gzip-encoded step bodies
    so the returned dicts are the full logical payloads, not on-disk frames.
    """
    frames = read_frames(sb_path)
    blobs: dict = {}
    steps: list = []
    for wrapper in frames:
        body = wrapper["body"]
        kind = body.get("type")
        if kind == "blob":
            try:
                blobs[body["id"]] = _decode_blob(body)
            except Exception:
                pass
        elif kind == "step":
            try:
                step = _decode_gz_step(body)
                if _has_blob_ref(step):
                    step = _materialise(step, blobs)
                steps.append(step)
            except Exception:
                pass
    return steps


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


def run(
    n_traces: int = 20,
    n_steps: int = 30,
    *,
    seed: int = 0,
) -> StorageCompressionResult:
    """Build a corpus of ``n_traces`` synthetic traces and measure storage cost.

    Parameters
    ----------
    n_traces:
        Number of independent synthetic traces to generate.
    n_steps:
        Target steps per trace (the actual count may vary slightly due to
        parallel-block emission).
    seed:
        Base random seed; each trace uses ``seed + i`` to remain
        reproducible.

    Returns
    -------
    StorageCompressionResult
    """
    t0 = time.monotonic()
    rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Phase 1: build all synthetic traces and collect logical steps.
    # ------------------------------------------------------------------
    # Per-trace data: (.sb path, [step dicts], file_size_bytes)
    trace_data: List[Tuple[str, List[dict], int]] = []
    tmpdirs = []

    for i in range(n_traces):
        st = SyntheticTrace(n_steps=n_steps, seed=seed + i)
        sb_path = st.build()
        tmpdirs.append(st._tmpdir)
        steps = _extract_steps(sb_path)
        file_size = os.path.getsize(sb_path)
        trace_data.append((sb_path, steps, file_size))

    total_actual_steps = sum(len(steps) for _, steps, _ in trace_data)

    # ------------------------------------------------------------------
    # Phase 2: measure raw_json (baseline).
    # raw_json = canonical_json(step) for every step, concatenated.
    # ------------------------------------------------------------------
    t_raw = time.monotonic()
    raw_json_per_trace: List[bytes] = []
    for _, steps, _ in trace_data:
        raw_json_per_trace.append(b"".join(canonical_json(s) for s in steps))
    raw_json_total = sum(len(b) for b in raw_json_per_trace)
    raw_json_time_ms = (time.monotonic() - t_raw) * 1000.0

    # ------------------------------------------------------------------
    # Phase 3: sb_v1 (already built — just sum file sizes).
    # ------------------------------------------------------------------
    t_sb = time.monotonic()
    sb_v1_total = sum(sz for _, _, sz in trace_data)
    sb_v1_time_ms = (time.monotonic() - t_sb) * 1000.0

    # ------------------------------------------------------------------
    # Phase 4: CBOR encoding.
    # ------------------------------------------------------------------
    t_cbor = time.monotonic()
    cbor_total = 0
    for _, steps, _ in trace_data:
        for step in steps:
            cbor_total += len(cbor_encode(step))
    cbor_time_ms = (time.monotonic() - t_cbor) * 1000.0

    # ------------------------------------------------------------------
    # Phase 5: gzip of raw JSON (per-trace, independent).
    # ------------------------------------------------------------------
    t_gz = time.monotonic()
    gzip_total = 0
    for raw_bytes in raw_json_per_trace:
        gzip_total += len(gzip.compress(raw_bytes, compresslevel=6, mtime=0))
    gzip_time_ms = (time.monotonic() - t_gz) * 1000.0

    # ------------------------------------------------------------------
    # Phase 6: LZMA of raw JSON (per-trace, independent).
    # stdlib substitute for zstd (which is not a declared dependency).
    # ------------------------------------------------------------------
    t_lzma = time.monotonic()
    lzma_total = 0
    for raw_bytes in raw_json_per_trace:
        lzma_total += len(lzma.compress(raw_bytes))
    lzma_time_ms = (time.monotonic() - t_lzma) * 1000.0

    # ------------------------------------------------------------------
    # Phase 7: content-addressed deduplication simulation.
    #
    # In a real object store, step *content* (inputs dict, outputs dict,
    # llm_request, llm_response) is stored once per unique content hash.
    # Per-step *metadata* (step_id, parent_step_id, cost_usd, wallclock_ns,
    # inputs_hash, outputs_hash, nondeterminism_hash, step_kind, name) lives
    # only in the lightweight index, not in the content blob.
    #
    # Content key: sha256(canonical_json({inputs, outputs, llm_request,
    #                                     llm_response, nondeterminism}))
    # which equals the pair (inputs_hash, outputs_hash) under the same
    # canonical scheme; we derive it directly to avoid re-hashing.
    #
    # Total = unique_content_bytes + canonical_json(step_index) bytes.
    # ------------------------------------------------------------------
    #: Fields that are step metadata, not content.
    _METADATA_FIELDS = frozenset(
        {
            "step_id",
            "parent_step_id",
            "step_kind",
            "cost_usd",
            "wallclock_ns",
            "inputs_hash",
            "outputs_hash",
            "nondeterminism_hash",
            "name",
        }
    )

    t_dedup = time.monotonic()
    unique_bodies: Dict[str, bytes] = {}  # content_key → canonical_json(content)
    step_index: Dict[str, List[dict]] = {}  # trace_id → [{step_id, content_key}]

    for trace_i, (_, steps, _) in enumerate(trace_data):
        trace_id = f"trace:{trace_i}"
        step_index[trace_id] = []
        for step in steps:
            content = {k: v for k, v in step.items() if k not in _METADATA_FIELDS}
            body_bytes = canonical_json(content)
            digest = "sha256:" + hashlib.sha256(body_bytes).hexdigest()
            unique_bodies[digest] = body_bytes
            step_index[trace_id].append(
                {"step_id": step.get("step_id", ""), "content_key": digest}
            )

    dedup_unique_bytes = sum(len(v) for v in unique_bodies.values())
    dedup_index_bytes = len(canonical_json(step_index))
    dedup_total = dedup_unique_bytes + dedup_index_bytes
    dedup_time_ms = (time.monotonic() - t_dedup) * 1000.0

    # ------------------------------------------------------------------
    # Phase 8: query index (additive overhead).
    #
    # A secondary index mapping each trace to a list of
    # {step_id, inputs_hash, outputs_hash, step_kind} records.
    # ------------------------------------------------------------------
    query_index: Dict[str, List[dict]] = {}
    for trace_i, (_, steps, _) in enumerate(trace_data):
        trace_id = f"trace:{trace_i}"
        query_index[trace_id] = [
            {
                "step_id": s.get("step_id", ""),
                "inputs_hash": s.get("inputs_hash", ""),
                "outputs_hash": s.get("outputs_hash", ""),
                "step_kind": s.get("step_kind", ""),
            }
            for s in steps
        ]
    query_index_bytes = len(canonical_json(query_index))

    # ------------------------------------------------------------------
    # Cleanup temp directories.
    # ------------------------------------------------------------------
    for tmpdir in tmpdirs:
        if tmpdir is not None:
            try:
                tmpdir.cleanup()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Assemble results.
    # ------------------------------------------------------------------
    n_t = max(n_traces, 1)
    n_s = max(total_actual_steps, 1)
    rj = max(raw_json_total, 1)

    formats: Dict[str, FormatResult] = {
        "raw_json": FormatResult(
            name="raw_json",
            total_bytes=raw_json_total,
            bytes_per_trace=raw_json_total / n_t,
            bytes_per_step=raw_json_total / n_s,
            ratio_vs_raw_json=1.0,
            encode_time_ms=raw_json_time_ms,
        ),
        "sb_v1": FormatResult(
            name="sb_v1",
            total_bytes=sb_v1_total,
            bytes_per_trace=sb_v1_total / n_t,
            bytes_per_step=sb_v1_total / n_s,
            ratio_vs_raw_json=sb_v1_total / rj,
            encode_time_ms=sb_v1_time_ms,
        ),
        "cbor": FormatResult(
            name="cbor",
            total_bytes=cbor_total,
            bytes_per_trace=cbor_total / n_t,
            bytes_per_step=cbor_total / n_s,
            ratio_vs_raw_json=cbor_total / rj,
            encode_time_ms=cbor_time_ms,
        ),
        "gzip_of_json": FormatResult(
            name="gzip_of_json",
            total_bytes=gzip_total,
            bytes_per_trace=gzip_total / n_t,
            bytes_per_step=gzip_total / n_s,
            ratio_vs_raw_json=gzip_total / rj,
            encode_time_ms=gzip_time_ms,
        ),
        "lzma_of_json": FormatResult(
            name="lzma_of_json",
            total_bytes=lzma_total,
            bytes_per_trace=lzma_total / n_t,
            bytes_per_step=lzma_total / n_s,
            ratio_vs_raw_json=lzma_total / rj,
            encode_time_ms=lzma_time_ms,
        ),
        "deduped_json": FormatResult(
            name="deduped_json",
            total_bytes=dedup_total,
            bytes_per_trace=dedup_total / n_t,
            bytes_per_step=dedup_total / n_s,
            ratio_vs_raw_json=dedup_total / rj,
            encode_time_ms=dedup_time_ms,
        ),
    }

    sb_v1_bytes = sb_v1_total
    total_wall_ms = (time.monotonic() - t0) * 1000.0

    return StorageCompressionResult(
        n_traces=n_traces,
        n_steps_requested=n_steps,
        actual_step_count=total_actual_steps,
        formats=formats,
        query_index_bytes=query_index_bytes,
        sb_v1_plus_query_index_bytes=sb_v1_bytes + query_index_bytes,
        query_index_overhead_ratio_vs_sb_v1=query_index_bytes / max(sb_v1_bytes, 1),
        deduped_unique_content_bytes=dedup_unique_bytes,
        deduped_index_bytes=dedup_index_bytes,
        total_wall_time_ms=total_wall_ms,
    )


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------


def compare(
    n_steps_list: List[int],
    n_traces: int = 5,
    *,
    seed: int = 0,
) -> Dict[int, StorageCompressionResult]:
    """Run the benchmark for each corpus size in ``n_steps_list``.

    Returns a dict mapping each ``n_steps`` value to its
    :class:`StorageCompressionResult`.
    """
    return {
        n: run(n_traces=n_traces, n_steps=n, seed=seed)
        for n in n_steps_list
    }
