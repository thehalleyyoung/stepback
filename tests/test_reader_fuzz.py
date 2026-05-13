"""Fuzz tests for the `.sb` trace reader.

Implements step 30 of ``100_STEPS.md``:

    Add fuzz tests for the trace reader: random prefixes, huge frame
    claims, truncation, duplicate frames, invalid UTF-8, and invalid
    JSON.

The reader is the trust boundary between untrusted trace bytes (which
may have been written by a foreign recorder, transferred over the wire,
or sat on disk for years) and the rest of the stepback runtime. It
must reject every malformed input with a typed
:class:`TraceVerificationError` rather than crashing with
``UnicodeDecodeError``, ``json.JSONDecodeError``, ``MemoryError``,
``struct.error``, or any other generic Python exception.

These tests cover the six fuzz axes called out in step 30:

1. **Random prefixes / random bytes** — purely random input must not
   crash or hang the reader; it must either reject as malformed or
   (if it happens to deserialise as JSON) reject during HMAC/signature
   verification.
2. **Huge frame claims** — a 4-byte length prefix declaring a body
   gigabytes long must not be honoured allocation-blindly; it must be
   rejected as truncated when the bytes don't exist on disk, and
   bounded by the reader's documented denial-of-service limits when
   they do.
3. **Truncation** — every prefix of a valid trace shorter than the full
   file must either parse (at exact frame boundaries with the chain
   ending mid-trace, which fails verification) or be flagged as a
   truncated length prefix or truncated body.
4. **Duplicate frames** — appending an exact copy of a real frame
   breaks the HMAC chain (because ``prev_hmac`` no longer matches) and
   must be rejected.
5. **Invalid UTF-8** — frame bodies containing bytes that are not valid
   UTF-8 must be rejected by ``read_frames`` (via JSON's UTF-8 decode)
   rather than crashing.
6. **Invalid JSON** — frame bodies that are valid UTF-8 but not valid
   JSON must be rejected by ``read_frames`` rather than crashing.

In every case the contract is: a known typed error
(:class:`TraceVerificationError` for the frame layer; for raw-byte
inputs that don't even reach JSON we accept any of
``TraceVerificationError``, :class:`json.JSONDecodeError`,
:class:`UnicodeDecodeError`, :class:`struct.error`, or the standard
``OSError`` family — never a SIGSEGV, hang, or runaway memory
allocation).
"""
from __future__ import annotations

import io
import json
import os
import struct
from typing import List

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from stepback import RecorderKey, record
from stepback.testing import run_recorded_agent
from stepback.trace_reader import (
    TraceVerificationError,
    read_frames,
    verify_trace,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Anything below this is a "frame too large" rejection class we accept;
# anything above is a real DoS vector we want documented behaviour for.
# The current reader does not enforce a hard cap at the layout layer,
# but it does refuse to read bytes that aren't on disk, which gives us
# cheap protection against a malicious 2 GiB length prefix.
_REASONABLE_REJECTIONS = (
    TraceVerificationError,
    json.JSONDecodeError,
    UnicodeDecodeError,
    struct.error,
    OSError,
)

# Once frames have been parsed by ``read_frames``, every downstream
# rejection from ``verify_trace`` should be a typed
# :class:`TraceVerificationError`. We allow ``KeyError`` / ``TypeError``
# *only* on the read side (e.g. JSON arrays-as-bodies caught before the
# new shape guards), and force-narrow on the verify side.
_VERIFY_REJECTIONS = (TraceVerificationError,)


def _record_real_trace(tmp_path) -> tuple[str, RecorderKey, bytes]:
    """Run the canonical fixture agent and return (path, key, raw_bytes)."""
    key = RecorderKey.fresh()
    path = str(tmp_path / "trace.sb")
    with record(path, key=key) as rec:
        run_recorded_agent(rec)
    with open(path, "rb") as f:
        raw = f.read()
    return path, key, raw


def _frame_offsets(raw: bytes) -> List[int]:
    """Return the byte offsets of every length prefix in ``raw``.

    Useful for truncation/duplication tests that need to operate on
    real frame boundaries rather than blind byte indices.
    """
    out = [0]
    pos = 0
    while pos < len(raw):
        if pos + 4 > len(raw):
            break
        (n,) = struct.unpack(">I", raw[pos : pos + 4])
        pos += 4 + n
        if pos <= len(raw):
            out.append(pos)
    return out


def _write(tmp_path, name: str, data: bytes) -> str:
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


# ---------------------------------------------------------------------------
# 1. Random prefixes / random bytes
# ---------------------------------------------------------------------------


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(blob=st.binary(min_size=0, max_size=4096))
def test_random_bytes_never_crash_reader(tmp_path_factory, blob):
    """Reader must reject pure-random byte blobs with a typed error.

    The reader may not raise ``MemoryError``, segfault, hang, or yield
    a partially valid :class:`Trace_`. The empty input is the boundary
    case: ``read_frames`` returns an empty list (no frames), and
    ``verify_trace`` then raises ``TraceVerificationError("empty
    trace")``.
    """
    tmp = tmp_path_factory.mktemp("rand")
    path = _write(tmp, "rand.sb", blob)

    try:
        frames = read_frames(path)
    except _REASONABLE_REJECTIONS:
        return  # Accepted rejection.
    # If parsing somehow succeeded, the verifier must still reject
    # (random bytes have no chance of producing a valid HMAC chain).
    key = RecorderKey.fresh()
    if not frames:
        with pytest.raises(TraceVerificationError):
            verify_trace(path, key.hmac_key)
        return
    with pytest.raises(_VERIFY_REJECTIONS):
        verify_trace(path, key.hmac_key)


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(prefix=st.binary(min_size=1, max_size=512))
def test_random_prefix_before_real_trace(tmp_path_factory, prefix):
    """Prepending random bytes to a real trace must be rejected.

    The leading bytes will be interpreted as a length prefix + body,
    which will either fail to UTF-8-decode, fail to JSON-parse, or
    parse but then break the HMAC chain (because the recorded
    ``prev_hmac`` won't match the noise's running HMAC). Either way,
    no path through ``verify_trace`` may succeed.
    """
    tmp = tmp_path_factory.mktemp("prefixed")
    real_path, key, real_bytes = _record_real_trace(tmp)
    poisoned = _write(tmp, "poisoned.sb", prefix + real_bytes)
    with pytest.raises(_REASONABLE_REJECTIONS):
        verify_trace(poisoned, key.hmac_key)


# ---------------------------------------------------------------------------
# 2. Huge frame claims
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "claimed_size",
    [
        2**16,         # 64 KiB
        2**20,         # 1 MiB
        2**28,         # 256 MiB
        2**31 - 1,     # ~2 GiB, max signed int32
        2**32 - 1,     # max uint32
    ],
)
def test_huge_length_prefix_with_no_body(tmp_path, claimed_size):
    """A 4-byte length prefix declaring a body that isn't on disk must
    be rejected — either by the per-reader ``MAX_FRAME_BYTES`` cap
    (Step 50) before any allocation, or, for claims under the cap, as
    a truncated body. Naively honouring the prefix and allocating
    multiple gigabytes is the failure mode this test forbids.
    """
    from stepback.trace_reader import MAX_FRAME_BYTES

    path = _write(tmp_path, "huge.sb", struct.pack(">I", claimed_size))
    if claimed_size > MAX_FRAME_BYTES:
        expected = (
            r"frame length \d+ exceeds MAX_FRAME_BYTES"
        )
    else:
        expected = "truncated frame body"
    with pytest.raises(TraceVerificationError, match=expected):
        read_frames(path)


def test_huge_length_prefix_with_partial_body(tmp_path):
    """Same as above but with *some* body bytes — still must reject."""
    body = b"x" * 100
    payload = struct.pack(">I", 2**24) + body  # claim 16 MiB, deliver 100 B
    path = _write(tmp_path, "partial.sb", payload)
    with pytest.raises(TraceVerificationError, match="truncated frame body"):
        read_frames(path)


# ---------------------------------------------------------------------------
# 3. Truncation
# ---------------------------------------------------------------------------


def test_truncated_length_prefix_rejected(tmp_path):
    """A trailing 1–3 byte length prefix is a truncation, not EOF."""
    for n in (1, 2, 3):
        path = _write(tmp_path, f"trunc_{n}.sb", b"\x00" * n)
        with pytest.raises(TraceVerificationError, match="truncated length prefix"):
            read_frames(path)


def test_every_byte_truncation_of_real_trace(tmp_path):
    """Every prefix of a real trace shorter than the full file must
    parse to fewer frames or fail verification.

    Only truncations that land *exactly* on a frame boundary can yield
    a clean :func:`read_frames`; in that case the chain ends prematurely
    and ``verify_trace`` rejects (no header on a 0-byte truncation;
    HMAC chain incomplete on a mid-trace boundary; signature mismatch
    on a partial body that happened to UTF-8-decode). The contract:
    *no truncation may be silently accepted as a valid trace.*
    """
    path, key, raw = _record_real_trace(tmp_path)
    boundaries = set(_frame_offsets(raw))

    # Sample 30 truncation points across the file; full N would be slow.
    step = max(1, len(raw) // 30)
    for cut in range(0, len(raw), step):
        if cut == len(raw):
            continue
        truncated = _write(tmp_path, f"t_{cut}.sb", raw[:cut])
        try:
            frames = read_frames(truncated)
        except _REASONABLE_REJECTIONS:
            continue  # Acceptable: rejected at the layout layer.
        # Layout-level parse succeeded. Two sub-cases:
        if cut not in boundaries:
            pytest.fail(
                f"truncation at offset {cut} parsed cleanly but is not "
                f"on a frame boundary; reader silently dropped data"
            )
        # On a clean boundary, verification must still reject because
        # the trace is now shorter than the recorder intended (no tail,
        # potentially no header for cut==0).
        with pytest.raises(TraceVerificationError):
            verify_trace(truncated, key.hmac_key)


# ---------------------------------------------------------------------------
# 4. Duplicate frames
# ---------------------------------------------------------------------------


def test_duplicate_frame_breaks_chain(tmp_path):
    """Appending a verbatim copy of a real frame breaks the HMAC chain.

    The duplicate's ``prev_hmac`` field won't match the running HMAC
    after the genuine tail, so verification rejects.
    """
    path, key, raw = _record_real_trace(tmp_path)
    offsets = _frame_offsets(raw)
    # Pick the last frame (the tail) and append it.
    last_off = offsets[-2]  # offsets always ends with len(raw)
    dup_frame = raw[last_off:]
    poisoned = _write(tmp_path, "dup.sb", raw + dup_frame)
    with pytest.raises(TraceVerificationError):
        verify_trace(poisoned, key.hmac_key)


def test_duplicated_header_in_middle(tmp_path):
    """Inserting a duplicated header frame in the middle must fail.

    Even ignoring HMAC chain breakage, a second header is a structural
    error: the reader pins ``pub`` from the first header and would
    silently switch keys mid-trace if it accepted the second. The
    chain check catches this first in practice, but the failure mode
    must be a typed reject either way.
    """
    path, key, raw = _record_real_trace(tmp_path)
    offsets = _frame_offsets(raw)
    header_frame = raw[offsets[0] : offsets[1]]
    # Insert a copy of the header just before the tail.
    insert_at = offsets[-2]
    poisoned = _write(
        tmp_path,
        "dup_hdr.sb",
        raw[:insert_at] + header_frame + raw[insert_at:],
    )
    with pytest.raises(TraceVerificationError):
        verify_trace(poisoned, key.hmac_key)


# ---------------------------------------------------------------------------
# 5. Invalid UTF-8
# ---------------------------------------------------------------------------


# Bytes guaranteed to be invalid UTF-8 (continuation byte without a
# start byte, lone 0x80, lone surrogate, overlong, etc.). We avoid
# code points that happen to encode to valid sequences.
_INVALID_UTF8_BODIES = [
    b"\x80",
    b"\xff",
    b"\xc3\x28",          # invalid 2-byte sequence
    b"\xa0\xa1",
    b"\xe2\x28\xa1",      # invalid 3-byte sequence
    b"\xf8\xa1\xa1\xa1",  # invalid 4-byte (out of range)
    b"\xed\xa0\x80",      # surrogate U+D800 encoded as utf-8 (rejected by Python)
]


@pytest.mark.parametrize("body", _INVALID_UTF8_BODIES)
def test_invalid_utf8_body_rejected(tmp_path, body):
    """Bodies that aren't valid UTF-8 must be rejected by the reader."""
    payload = struct.pack(">I", len(body)) + body
    path = _write(tmp_path, "bad_utf8.sb", payload)
    with pytest.raises((UnicodeDecodeError, TraceVerificationError, json.JSONDecodeError)):
        read_frames(path)


@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(body=st.binary(min_size=1, max_size=128))
def test_random_body_typed_rejection(tmp_path_factory, body):
    """A frame whose body is random bytes must produce a typed error,
    never a generic Python crash.

    If the body happens to be valid UTF-8 *and* valid JSON, the
    verifier will reject it (no real wrapper has the required
    ``hmac``/``prev_hmac``/``sig``/``body`` keys with the right shape).
    If it's invalid UTF-8 or invalid JSON, ``read_frames`` rejects.
    """
    tmp = tmp_path_factory.mktemp("rb")
    payload = struct.pack(">I", len(body)) + body
    path = _write(tmp, "rb.sb", payload)
    key = RecorderKey.fresh()
    try:
        frames = read_frames(path)
    except _REASONABLE_REJECTIONS:
        return
    # Parsed; verification must now reject with a typed error.
    with pytest.raises(_VERIFY_REJECTIONS):
        verify_trace(path, key.hmac_key)


# ---------------------------------------------------------------------------
# 6. Invalid JSON
# ---------------------------------------------------------------------------


_INVALID_JSON_BODIES = [
    b"",                         # empty body — not valid JSON
    b"not json",
    b"{",
    b"}",
    b"[1, 2,",
    b"{\"unterminated\": \"str",
    b"{trailing: comma,}",
    b"\xef\xbb\xbf{}",           # BOM + empty obj — strict JSON rejects BOM
]


@pytest.mark.parametrize("body", _INVALID_JSON_BODIES)
def test_invalid_json_body_rejected(tmp_path, body):
    """Bodies that decode as UTF-8 but are not valid JSON must be
    rejected by ``read_frames`` rather than producing a half-parsed
    dict or crashing.
    """
    payload = struct.pack(">I", len(body)) + body
    path = _write(tmp_path, "bad_json.sb", payload)
    with pytest.raises((json.JSONDecodeError, TraceVerificationError)):
        read_frames(path)


def test_valid_json_but_wrong_shape_rejected(tmp_path):
    """A frame whose body is valid JSON but not a wrapper object must
    be rejected by the verifier (``KeyError`` would be a bug; we
    require a typed exception class).
    """
    # JSON scalar where the verifier expects a dict.
    body = json.dumps(42).encode("utf-8")
    payload = struct.pack(">I", len(body)) + body
    path = _write(tmp_path, "wrong_shape.sb", payload)
    key = RecorderKey.fresh()
    with pytest.raises(_VERIFY_REJECTIONS):
        verify_trace(path, key.hmac_key)


def test_valid_json_object_missing_required_keys(tmp_path):
    """A frame that is a JSON object but lacks ``body``/``hmac``/``sig``
    must not pass verification.
    """
    body = json.dumps({"hello": "world"}).encode("utf-8")
    payload = struct.pack(">I", len(body)) + body
    path = _write(tmp_path, "no_keys.sb", payload)
    key = RecorderKey.fresh()
    with pytest.raises(_VERIFY_REJECTIONS):
        verify_trace(path, key.hmac_key)


# ---------------------------------------------------------------------------
# Cross-axis: structured Hypothesis fuzzer over framed payloads
# ---------------------------------------------------------------------------


_FRAME_BODIES = st.one_of(
    st.binary(min_size=0, max_size=64),
    st.text(min_size=0, max_size=64).map(lambda s: s.encode("utf-8", "replace")),
    st.builds(
        lambda d: json.dumps(d).encode("utf-8"),
        st.dictionaries(
            st.text(max_size=8),
            st.one_of(st.integers(), st.text(max_size=8), st.booleans(), st.none()),
            max_size=4,
        ),
    ),
)


@settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(
    frames=st.lists(_FRAME_BODIES, min_size=0, max_size=8),
    # Sometimes claim a different size than we actually write, to
    # exercise the truncation/over-claim paths.
    fudges=st.lists(st.integers(min_value=-3, max_value=8), max_size=8),
)
def test_structured_fuzz_framed_payload(tmp_path_factory, frames, fudges):
    """Build framed payloads with random bodies and possibly mismatched
    length prefixes, and assert the reader either parses cleanly or
    raises a typed rejection — never a generic crash.
    """
    tmp = tmp_path_factory.mktemp("fuzz")
    buf = io.BytesIO()
    for i, body in enumerate(frames):
        fudge = fudges[i] if i < len(fudges) else 0
        claimed = max(0, len(body) + fudge)
        buf.write(struct.pack(">I", claimed))
        buf.write(body)
    path = _write(tmp, "fuzz.sb", buf.getvalue())
    key = RecorderKey.fresh()
    try:
        out = read_frames(path)
    except _REASONABLE_REJECTIONS:
        return
    # Parsed; verifying random frames must reject.
    if not out:
        with pytest.raises(TraceVerificationError):
            verify_trace(path, key.hmac_key)
        return
    with pytest.raises(_VERIFY_REJECTIONS):
        verify_trace(path, key.hmac_key)


# ---------------------------------------------------------------------------
# 7. Per-reader denial-of-service bounds (Step 50)
# ---------------------------------------------------------------------------
#
# These tests pin the contract documented in ``docs/reader-limits.md``:
# every conformant reader rejects pathological frames with
# :class:`TraceVerificationError` rather than allocating unboundedly,
# blowing the stack, or otherwise enabling a DoS on the verifier.


def _wrap_body(body: dict) -> bytes:
    wrapper = {
        "body": body,
        "prev_hmac": "00" * 32,
        "hmac": "00" * 32,
        "sig": "ed25519:" + "00" * 64,
    }
    payload = json.dumps(wrapper, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload


def test_max_frame_bytes_rejects_oversize_prefix(tmp_path):
    """A length prefix above ``MAX_FRAME_BYTES`` must fail before any
    body bytes are read."""
    from stepback.trace_reader import MAX_FRAME_BYTES

    path = _write(
        tmp_path,
        "oversize.sb",
        struct.pack(">I", MAX_FRAME_BYTES + 1) + b"x" * 10,
    )
    with pytest.raises(
        TraceVerificationError,
        match=r"frame length \d+ exceeds MAX_FRAME_BYTES",
    ):
        read_frames(path)


def test_max_nesting_depth_rejects_deep_object(tmp_path):
    """A frame body deeper than ``MAX_NESTING_DEPTH`` must be rejected."""
    from stepback.trace_reader import MAX_NESTING_DEPTH

    body: dict | list = {}
    cursor = body
    for _ in range(MAX_NESTING_DEPTH + 5):
        cursor["x"] = {}
        cursor = cursor["x"]
    payload = _wrap_body({"type": "tail", "deep": body})
    path = _write(tmp_path, "deep.sb", payload)
    with pytest.raises(
        TraceVerificationError,
        match=r"nesting depth exceeds MAX_NESTING_DEPTH",
    ):
        read_frames(path)


def test_max_nesting_depth_rejects_deep_array(tmp_path):
    """Deep *arrays* are bounded the same way as deep objects."""
    from stepback.trace_reader import MAX_NESTING_DEPTH

    body: list = []
    cursor = body
    for _ in range(MAX_NESTING_DEPTH + 5):
        nxt: list = []
        cursor.append(nxt)
        cursor = nxt
    payload = _wrap_body({"type": "tail", "deep": body})
    path = _write(tmp_path, "deep_arr.sb", payload)
    with pytest.raises(
        TraceVerificationError,
        match=r"nesting depth exceeds MAX_NESTING_DEPTH",
    ):
        read_frames(path)


def test_max_string_bytes_rejects_huge_string(tmp_path):
    """A single inline string longer than ``MAX_STRING_BYTES`` must be
    rejected. Genuinely large payloads belong in blob frames."""
    from stepback.trace_reader import MAX_STRING_BYTES

    huge = "x" * (MAX_STRING_BYTES + 1)
    # Lift the frame-bytes cap for *this* test so the wrapper itself
    # fits, isolating the string-size check.
    body = {"type": "tail", "huge": huge}
    payload = _wrap_body(body)
    path = _write(tmp_path, "huge_str.sb", payload)
    with pytest.raises(
        TraceVerificationError,
        match=r"string value exceeding MAX_STRING_BYTES",
    ):
        read_frames(path, max_frame_bytes=len(payload) + 1024)


def test_max_string_bytes_rejects_huge_object_key(tmp_path):
    from stepback.trace_reader import MAX_STRING_BYTES

    huge_key = "k" * (MAX_STRING_BYTES + 1)
    body = {"type": "tail", huge_key: 1}
    payload = _wrap_body(body)
    path = _write(tmp_path, "huge_key.sb", payload)
    with pytest.raises(
        TraceVerificationError,
        match=r"object key exceeding MAX_STRING_BYTES",
    ):
        read_frames(path, max_frame_bytes=len(payload) + 1024)


def test_limits_are_tunable_upwards(tmp_path):
    """Callers that consciously process larger payloads can opt in by
    passing higher caps, but the *defaults* stay strict."""
    from stepback.trace_reader import MAX_NESTING_DEPTH

    body: dict | list = {}
    cursor = body
    # Build something just past the default depth so we know the
    # default would have rejected.
    for _ in range(MAX_NESTING_DEPTH + 2):
        cursor["x"] = {}
        cursor = cursor["x"]
    payload = _wrap_body({"type": "tail", "deep": body})
    path = _write(tmp_path, "tunable.sb", payload)
    with pytest.raises(TraceVerificationError):
        read_frames(path)
    # With a relaxed depth cap, the same bytes parse fine.
    out = read_frames(path, max_depth=MAX_NESTING_DEPTH + 1024)
    assert len(out) == 1
