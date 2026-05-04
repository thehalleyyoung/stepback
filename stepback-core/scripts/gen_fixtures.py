#!/usr/bin/env python3
"""Generate the frozen `.sb` conformance fixture corpus.

This script is the single source of truth for the fixtures the Rust
`sb-verify` crate (and any other independent reader) must accept or
reject. Running it produces *bit-identical* output across runs and
across machines: the HMAC key, the Ed25519 private key, and every
wallclock value are pinned to constants below; nothing depends on
`os.urandom` or the current time.

Layout written under ``stepback-core/fixtures/v1/``::

    manifest.json                # machine-readable fixture index
    good/header_only.sb          # header + tail
    good/multi_step.sb           # header + 3 step frames + tail
    good/with_blobs.sb           # header + blob + steps + tail
    corrupt/truncated_body.sb    # last frame body chopped short
    corrupt/flipped_hmac.sb      # one byte of frame[1].hmac mutated
    corrupt/flipped_sig.sb       # one byte of frame[1].sig mutated
    corrupt/broken_chain.sb      # frame[1].prev_hmac mutated
    corrupt/bad_format_version.sb  # header.format_version=99

The ``manifest.json`` carries the HMAC key (hex), public key (hex),
and the expected outcome for each fixture so any independent verifier
can do strict round-trip validation.

Re-run with::

    python3 stepback-core/scripts/gen_fixtures.py

Files are only rewritten if their bytes change; an unchanged run
produces an empty git diff.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from stepback.trace_writer import TraceWriter

# Pinned, well-known test material. NEVER use these in production —
# the private key is checked into the repo.
HMAC_KEY = bytes.fromhex(
    "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20"
)
SIGNING_KEY_BYTES = bytes.fromhex(
    "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
)
PRICE_LIST_VERSION = "test-2026-01-01"

FIXTURE_ROOT = REPO_ROOT / "stepback-core" / "fixtures" / "v1"


def _make_signing_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(SIGNING_KEY_BYTES)


def _frozen_clock_factory():
    """Return a fresh deterministic ``time.time_ns`` substitute.

    Each call inside a single fixture generation increments by 1ns so
    that tail timestamps differ from header timestamps but the entire
    fixture remains reproducible.
    """
    counter = {"n": 1_700_000_000_000_000_000}

    def time_ns() -> int:
        v = counter["n"]
        counter["n"] = v + 1
        return v

    return time_ns


def _write_trace(path: Path, builder, **writer_kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fake_clock = _frozen_clock_factory()
    with mock.patch("stepback.trace_writer.time.time_ns", fake_clock):
        w = TraceWriter.open(
            str(path),
            hmac_key=HMAC_KEY,
            signing_key=_make_signing_key(),
            price_list_version=PRICE_LIST_VERSION,
            **writer_kwargs,
        )
        try:
            builder(w)
        finally:
            w.close()


def _build_header_only(_w: TraceWriter) -> None:
    """No-op: TraceWriter.open already wrote the header; close() will
    write the tail. Result: 2 frames."""
    return None


def _build_multi_step(w: TraceWriter) -> None:
    for i, kind in enumerate(("llm_call", "tool_call", "router")):
        w.write_step(
            {
                "step_id": f"01HZZ{i:026d}",
                "step_kind": kind,
                "parent_step_id": None,
                "inputs": {"i": i, "k": kind, "msg": f"hello {i}"},
                "outputs": {"reply": f"ack {i}"},
                "inputs_hash": f"sha256:{'0' * 63}{i}",
                "wallclock_ns": 1_700_000_000_000_000_000 + i,
            }
        )


def _build_with_blobs(w: TraceWriter) -> None:
    # The same large-ish payload appears in two consecutive steps so
    # the writer's reuse-aware blob table actually emits a blob frame.
    big = {
        "shared": "x" * 400,
        "manifest": [{"id": j, "label": f"item-{j}"} for j in range(20)],
    }
    for i in range(2):
        w.write_step(
            {
                "step_id": f"01HZZBLOB{i:022d}",
                "step_kind": "llm_call",
                "parent_step_id": None,
                "inputs": {"i": i, "payload": big},
                "outputs": {"reply": f"ack {i}"},
                "inputs_hash": f"sha256:{'b' * 63}{i}",
            }
        )


def _flip_byte(buf: bytearray, off: int) -> None:
    buf[off] ^= 0x55


def _frame_offsets(buf: bytes) -> list[tuple[int, int]]:
    """Return ``(body_start, body_len)`` for every frame in ``buf``.

    The wrappers carry a 4-byte big-endian length prefix; the body
    is the canonical-JSON payload right after.
    """
    out: list[tuple[int, int]] = []
    pos = 0
    while pos < len(buf):
        if pos + 4 > len(buf):
            break
        (n,) = struct.unpack(">I", buf[pos : pos + 4])
        out.append((pos + 4, n))
        pos += 4 + n
    return out


def _find_in_frame(buf: bytes, frame_start: int, frame_len: int, needle: bytes) -> int:
    end = frame_start + frame_len
    return buf.index(needle, frame_start, end)


def _build_corrupt_variants(good_multi_step: bytes, dest: Path) -> dict[str, dict]:
    """Derive corrupt variants from the good multi-step trace."""
    out: dict[str, dict] = {}

    # 1. Truncate the last frame's body in half.
    frames = _frame_offsets(good_multi_step)
    last_start, last_len = frames[-1]
    truncated = good_multi_step[: last_start + last_len // 2]
    p = dest / "truncated_body.sb"
    p.write_bytes(truncated)
    out[p.name] = {
        "expected": "reject",
        "expected_error_kind": "Parse",
        "note": "last frame body chopped in half — readers must reject UnexpectedEof",
    }

    # 2. Flip a byte inside frame[1]'s hmac hex value.
    fb = bytearray(good_multi_step)
    f1_start, f1_len = frames[1]
    needle = b'"hmac":"'
    off = _find_in_frame(bytes(fb), f1_start, f1_len, needle) + len(needle)
    _flip_byte(fb, off)
    p = dest / "flipped_hmac.sb"
    p.write_bytes(bytes(fb))
    out[p.name] = {
        "expected": "reject",
        "expected_error_kind": "BadHexOrHmacMismatch",
        "note": "one nibble of frame[1].hmac flipped — HMAC or chain check must fail",
    }

    # 3. Flip a byte inside frame[1]'s ed25519:<hex> signature.
    fb = bytearray(good_multi_step)
    f1_start, f1_len = frames[1]
    needle = b'"sig":"ed25519:'
    off = _find_in_frame(bytes(fb), f1_start, f1_len, needle) + len(needle)
    _flip_byte(fb, off)
    p = dest / "flipped_sig.sb"
    p.write_bytes(bytes(fb))
    out[p.name] = {
        "expected": "reject",
        "expected_error_kind": "SignatureMismatch",
        "note": "one nibble of frame[1].sig flipped — Ed25519 verify must fail",
    }

    # 4. Mutate frame[1]'s prev_hmac so the chain to frame[0] breaks.
    fb = bytearray(good_multi_step)
    f1_start, f1_len = frames[1]
    needle = b'"prev_hmac":"'
    off = _find_in_frame(bytes(fb), f1_start, f1_len, needle) + len(needle)
    _flip_byte(fb, off)
    p = dest / "broken_chain.sb"
    p.write_bytes(bytes(fb))
    out[p.name] = {
        "expected": "reject",
        "expected_error_kind": "BrokenChainOrHmacMismatch",
        "note": (
            "frame[1].prev_hmac mutated — verifier must reject "
            "(the HMAC over body+prev no longer matches either)"
        ),
    }

    # 5. Forge a header with format_version=99. Length-preserving
    #    rewrite (1 char -> 1 char) so frame layout stays intact and
    #    the verifier reaches the version check on the first frame.
    fb = bytearray(good_multi_step)
    f0_start, f0_len = frames[0]
    needle = b'"format_version":1'
    off = _find_in_frame(bytes(fb), f0_start, f0_len, needle) + len(needle) - 1
    fb[off : off + 1] = b"9"
    p = dest / "bad_format_version.sb"
    p.write_bytes(bytes(fb))
    out[p.name] = {
        "expected": "reject",
        "expected_error_kind": "UnsupportedFormatVersionOrHmacMismatch",
        "note": (
            "header.format_version forged to 9 (length-preserving). "
            "Either the version check or the HMAC check must reject it."
        ),
    }

    return out


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_if_changed(path: Path, data: bytes) -> bool:
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def main() -> int:
    good_dir = FIXTURE_ROOT / "good"
    corrupt_dir = FIXTURE_ROOT / "corrupt"
    good_dir.mkdir(parents=True, exist_ok=True)
    corrupt_dir.mkdir(parents=True, exist_ok=True)

    builders = [
        ("header_only.sb", _build_header_only, 2, {"compression": False}),
        ("multi_step.sb", _build_multi_step, 5, {"compression": False}),
        ("with_blobs.sb", _build_with_blobs, 5, {"compression": True}),
    ]

    public_key_hex = _make_signing_key().public_key().public_bytes_raw().hex()

    good_entries: list[dict] = []
    for name, builder, frame_count, kwargs in builders:
        path = good_dir / name
        _write_trace(path, builder, **kwargs)
        data = path.read_bytes()
        good_entries.append(
            {
                "name": name,
                "expected": "ok",
                "expected_frame_count_min": frame_count,
                "size_bytes": len(data),
                "sha256": _sha256_hex(data),
            }
        )

    multi_step_bytes = (good_dir / "multi_step.sb").read_bytes()
    corrupt_meta = _build_corrupt_variants(multi_step_bytes, corrupt_dir)
    corrupt_entries: list[dict] = []
    for name, meta in corrupt_meta.items():
        data = (corrupt_dir / name).read_bytes()
        corrupt_entries.append(
            {
                "name": name,
                "expected": meta["expected"],
                "expected_error_kind": meta["expected_error_kind"],
                "note": meta["note"],
                "size_bytes": len(data),
                "sha256": _sha256_hex(data),
            }
        )

    manifest = {
        "format_version": 1,
        "canonicalisation_version": "1",
        "price_list_version": PRICE_LIST_VERSION,
        "hmac_key_hex": HMAC_KEY.hex(),
        "public_key_hex": public_key_hex,
        "good": good_entries,
        "corrupt": corrupt_entries,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    changed = _write_if_changed(FIXTURE_ROOT / "manifest.json", manifest_bytes)

    print(
        f"wrote {len(good_entries)} good + {len(corrupt_entries)} corrupt fixtures "
        f"under {FIXTURE_ROOT.relative_to(REPO_ROOT)} "
        f"(manifest {'updated' if changed else 'unchanged'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
