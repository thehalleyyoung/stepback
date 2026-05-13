"""Exhaustive single-byte / single-bit corruption tests for the ``.sb``
trace reader.

Implements step 31 of ``100_STEPS.md``:

    Add corruption tests that flip every byte in a small `.sb` file and
    assert the verifier rejects with typed errors.

Where the existing ``test_reader_fuzz.py`` covers *random* input
(prefix-blob fuzzing, huge length claims, structural mutations), this
module covers the deterministic, exhaustive complement: take a small,
genuinely-valid trace, mutate **one position at a time**, and assert the
verifier rejects the result with one of the typed error classes the
``.sb`` v1 spec promises. No mutation may be silently accepted as a
valid trace.

The mutations we apply at each byte offset are:

* **byte XOR 0xFF** — a maximal change of every bit at that position;
* **byte += 1 (mod 256)** — a minimal change of the low bit;
* **bit-flip in each of the 8 bit positions** for a small carved
  prefix region (the 4-byte length prefix of the first frame plus the
  first 64 bytes of the header body), where bit-precise coverage is
  most valuable for catching missed boundary cases.

For every mutated file the contract is: ``verify_trace`` raises a
typed :class:`TraceVerificationError`, OR the mutation tripped a
read-side typed error from the
:data:`_REASONABLE_REJECTIONS` family (length-prefix decode, JSON
decode, UTF-8 decode, ``OSError``). The reader must never crash with a
generic exception, hang, or silently return a ``Trace_``.

Two important invariants are explicitly checked:

1. **No silent-pass mutation.** If a mutated byte stream parses *and*
   verifies, the test fails: byte was unused, which would be a real
   audit-trail hole. The HMAC chain + Ed25519 signature should make
   every byte load-bearing.

2. **No untyped exception leaks.** Any rejection must come from
   :data:`_REASONABLE_REJECTIONS`. ``AttributeError``,
   ``KeyError``, ``IndexError``, ``MemoryError``, ``RecursionError``,
   etc. are unconditional failures.

To keep the suite fast, the trace is recorded once at module scope
(single ``tool_call`` step; ~1.8 KiB on disk) and shared across
mutations via byte-level rewrites in ``tmp_path``.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Tuple

import pytest

from stepback import RecorderKey, record
from stepback.trace_reader import (
    TraceVerificationError,
    read_frames,
    verify_trace,
)


# ---------------------------------------------------------------------------
# Typed-error allow-lists (mirrors test_reader_fuzz.py contract)
# ---------------------------------------------------------------------------

# Errors acceptable at the *layout* layer (length prefix decode,
# UTF-8 decode of bodies, JSON decode of bodies, file-system errors).
_READ_REJECTIONS: Tuple[type, ...] = (
    TraceVerificationError,
    json.JSONDecodeError,
    UnicodeDecodeError,
    struct.error,
    OSError,
)

# Errors acceptable at the *verify* layer (HMAC chain mismatch,
# Ed25519 signature mismatch, structural shape errors). The reader
# narrows everything past ``read_frames`` into ``TraceVerificationError``.
_VERIFY_REJECTIONS: Tuple[type, ...] = (TraceVerificationError,)

# Disallow-list: if the verifier raises one of these for a corrupted
# trace, the typed-error contract is broken.
_FORBIDDEN: Tuple[type, ...] = (
    AttributeError,
    KeyError,
    IndexError,
    MemoryError,
    RecursionError,
    AssertionError,
)


# ---------------------------------------------------------------------------
# Fixture: one small recorded trace, reused across thousands of mutations.
# ---------------------------------------------------------------------------


def _record_small_trace(path: str) -> Tuple[bytes, bytes]:
    """Record a minimal one-step trace and return (raw_bytes, hmac_key).

    The trace is intentionally tiny: a single ``tool_call`` with a
    deterministic executor. This keeps the byte count low enough to
    afford full coverage (every offset, multiple mutations).
    """
    key = RecorderKey.fresh()
    with record(path, key=key) as rec:
        rec.tool_call("echo", {"x": 1}, lambda name, args: {"y": 2})
    with open(path, "rb") as f:
        raw = f.read()
    return raw, key.hmac_key


@pytest.fixture(scope="module")
def small_trace(tmp_path_factory) -> Tuple[bytes, bytes, str]:
    """Module-scoped (raw_bytes, hmac_key, source_path) tuple.

    Recording is moderately expensive (Ed25519 keygen + signing); we
    only do it once and let each test re-write mutated copies into
    its own per-test ``tmp_path``.
    """
    base_dir = tmp_path_factory.mktemp("corruption_base")
    src = str(base_dir / "trace.sb")
    raw, hmac_key = _record_small_trace(src)
    # Sanity: the unmodified bytes verify cleanly. If this ever stops
    # being true the rest of the suite is meaningless.
    parsed = verify_trace(src, hmac_key)
    assert len(parsed.steps) >= 1
    return raw, hmac_key, src


def _write(tmp_path, name: str, data: bytes) -> str:
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def _check_rejected(path: str, hmac_key: bytes, label: str) -> None:
    """Assert that ``verify_trace(path)`` rejects with a typed error.

    Acceptable rejection paths:

    * ``read_frames`` (called inside ``verify_trace``) raises one of
      :data:`_READ_REJECTIONS`.
    * ``verify_trace`` itself raises :class:`TraceVerificationError`.

    Any other exception type is a typed-error contract violation and
    fails the test with a precise label naming the mutation.

    The verifier is looked up via the live module attribute so that
    the meta-test below can monkey-patch ``stepback.trace_reader``
    and observe the failure path.
    """
    import stepback.trace_reader as _tr

    try:
        result = _tr.verify_trace(path, hmac_key)
    except _READ_REJECTIONS:
        return
    except Exception as exc:  # pragma: no cover - defensive: caught by assert below
        raise AssertionError(
            f"corruption {label}: verifier raised untyped {type(exc).__name__}: {exc!r}"
        ) from exc
    raise AssertionError(
        f"corruption {label}: verifier silently accepted a mutated trace "
        f"(returned {len(result.steps)} steps, tail={result.tail!r})"
    )


# ---------------------------------------------------------------------------
# 1. Exhaustive byte XOR: mutate every offset with a maximal-change op.
# ---------------------------------------------------------------------------


def test_xor_every_byte_rejected(tmp_path, small_trace):
    """Flip every bit at every offset (XOR 0xFF). Reader must reject all."""
    raw, hmac_key, _ = small_trace
    n = len(raw)
    # Sanity: the test is only meaningful on a non-trivial trace.
    assert n > 100, f"expected a multi-byte trace, got {n} bytes"

    path = str(tmp_path / "xor.sb")
    buf = bytearray(raw)
    for offset in range(n):
        original = buf[offset]
        buf[offset] = original ^ 0xFF
        with open(path, "wb") as f:
            f.write(buf)
        _check_rejected(path, hmac_key, label=f"xor-0xFF@{offset}")
        buf[offset] = original  # restore for next mutation


# ---------------------------------------------------------------------------
# 2. Exhaustive byte +1 (mod 256): minimal-change complement to XOR 0xFF.
# ---------------------------------------------------------------------------


def test_increment_every_byte_rejected(tmp_path, small_trace):
    """Increment every byte by 1 (mod 256). Reader must reject all.

    XOR-0xFF and (+1 mod 256) cover the two extremes of single-byte
    perturbation: a maximal Hamming distance of 8 and a minimal one
    of 1 (or more, by carry into adjacent bits). Together they catch
    bugs that a single mutation type might miss — e.g. a parser that
    happens to accept the XORed value but rejects the incremented one,
    or vice versa.
    """
    raw, hmac_key, _ = small_trace
    n = len(raw)

    path = str(tmp_path / "inc.sb")
    buf = bytearray(raw)
    for offset in range(n):
        original = buf[offset]
        buf[offset] = (original + 1) & 0xFF
        with open(path, "wb") as f:
            f.write(buf)
        _check_rejected(path, hmac_key, label=f"+1@{offset}")
        buf[offset] = original


# ---------------------------------------------------------------------------
# 3. Bit-precise flip across the prefix region (length prefix + early header).
# ---------------------------------------------------------------------------


def test_bit_flip_prefix_region_rejected(tmp_path, small_trace):
    """Bit-precise flip across the file's first 68 bytes (every bit).

    Why limit to a prefix? A full per-bit sweep (raw bytes × 8) would
    be ~14 600 mutated traces for the small fixture, which is
    measurable in CI cost. Per-bit coverage is most valuable on the
    layout-critical region: the 4-byte length prefix of the header
    frame and the first ~64 bytes of the header body where canonical
    JSON keys live (``{"frame_kind":"header",...``). Mutations there
    exercise the length-prefix decoder, the UTF-8 decoder, and the
    JSON shape guards — exactly the read-side error paths step 30 set
    up. Byte-wise XOR/increment above already covers the rest.
    """
    raw, hmac_key, _ = small_trace
    region = min(len(raw), 4 + 64)

    path = str(tmp_path / "bit.sb")
    buf = bytearray(raw)
    for offset in range(region):
        for bit in range(8):
            original = buf[offset]
            buf[offset] = original ^ (1 << bit)
            with open(path, "wb") as f:
                f.write(buf)
            _check_rejected(path, hmac_key, label=f"bit{bit}@{offset}")
            buf[offset] = original


# ---------------------------------------------------------------------------
# 4. Read-frames invariants on the unmutated trace.
# ---------------------------------------------------------------------------


def test_unmutated_trace_still_verifies(small_trace):
    """Sanity check: the fixture trace is valid as long as we don't touch it.

    If this regresses, the rest of the corruption suite would be
    vacuous (every mutation would "reject" trivially), so we assert
    it explicitly here rather than relying on the fixture's setup
    assertion.
    """
    raw, hmac_key, src = small_trace
    parsed = verify_trace(src, hmac_key)
    assert parsed.header.get("type") == "header"
    assert any(s.get("step_kind") == "tool_call" for s in parsed.steps)
    assert read_frames(src), "read_frames returned no frames for valid trace"


# ---------------------------------------------------------------------------
# 5. Forbidden-exception guard: confirm the contract is observable.
# ---------------------------------------------------------------------------


def test_forbidden_exceptions_are_actually_caught(tmp_path, small_trace):
    """Meta-test: prove the typed-error contract above is enforceable.

    If we deliberately raise an :class:`AttributeError` from a faked
    verifier, ``_check_rejected`` must convert it to an
    :class:`AssertionError`. This guards against accidental refactors
    that loosen the contract (e.g. adding ``Exception`` to
    ``_READ_REJECTIONS``).
    """
    raw, hmac_key, _ = small_trace
    path = str(tmp_path / "meta.sb")
    with open(path, "wb") as f:
        f.write(raw)

    class _Boom:
        steps = ()
        tails = ()

    def _fake_verify(p, k, **kw):  # noqa: ARG001 - signature mirror
        raise AttributeError("simulated untyped failure")

    import stepback.trace_reader as tr

    real = tr.verify_trace
    tr.verify_trace = _fake_verify  # type: ignore[assignment]
    try:
        with pytest.raises(AssertionError, match="untyped AttributeError"):
            _check_rejected(path, hmac_key, label="meta")
    finally:
        tr.verify_trace = real  # type: ignore[assignment]
