"""Step 24 — capability negotiation frames fail closed.

The integrity-layer ``verify_trace`` (not just the schema-layer
``SBTraceSpec.validate_trace``) must reject traces whose ``capability``
frames declare a ``mandatory`` extension this build does not implement.
Optional capability frames must round-trip without rejection.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from stepback.trace_reader import (
    DEFAULT_SUPPORTED_CAPABILITIES,
    TraceVerificationError,
    verify_trace,
)
from stepback.trace_writer import TraceWriter


@pytest.fixture
def hmac_key() -> bytes:
    return b"k" * 32


def _new_writer(tmp_path: Path, hmac_key: bytes, *, compression: bool = False) -> TraceWriter:
    return TraceWriter.open(
        str(tmp_path / "t.sb"),
        hmac_key=hmac_key,
        compression=compression,
    )


def test_writer_rejects_empty_capability_name(tmp_path: Path, hmac_key: bytes) -> None:
    w = _new_writer(tmp_path, hmac_key)
    try:
        with pytest.raises(ValueError):
            w.write_capability("")
    finally:
        w.close()


def test_writer_rejects_non_dict_params(tmp_path: Path, hmac_key: bytes) -> None:
    w = _new_writer(tmp_path, hmac_key)
    try:
        with pytest.raises(TypeError):
            w.write_capability("blobs", params=["not", "a", "dict"])  # type: ignore[arg-type]
    finally:
        w.close()


def test_optional_unknown_capability_round_trips(
    tmp_path: Path, hmac_key: bytes
) -> None:
    w = _new_writer(tmp_path, hmac_key)
    w.write_capability("post-quantum-receipts", mandatory=False, params={"alg": "ML-DSA-65"})
    w.write_step({"step_index": 0, "kind": "llm_call"})
    w.close()

    trace = verify_trace(w.path, hmac_key)
    assert len(trace.capabilities) == 1
    cap = trace.capabilities[0]
    assert cap == {
        "name": "post-quantum-receipts",
        "mandatory": False,
        "params": {"alg": "ML-DSA-65"},
    }
    assert len(trace.steps) == 1


def test_mandatory_supported_capability_passes(
    tmp_path: Path, hmac_key: bytes
) -> None:
    w = _new_writer(tmp_path, hmac_key)
    w.write_capability("hmac-sha256-chain", mandatory=True)
    w.write_step({"step_index": 0, "kind": "llm_call"})
    w.close()

    trace = verify_trace(w.path, hmac_key)
    assert trace.capabilities[0]["mandatory"] is True
    assert "hmac-sha256-chain" in DEFAULT_SUPPORTED_CAPABILITIES


def test_mandatory_unknown_capability_fails_closed(
    tmp_path: Path, hmac_key: bytes
) -> None:
    w = _new_writer(tmp_path, hmac_key)
    w.write_capability("post-quantum-receipts", mandatory=True)
    w.write_step({"step_index": 0, "kind": "llm_call"})
    w.close()

    with pytest.raises(TraceVerificationError) as exc_info:
        verify_trace(w.path, hmac_key)
    assert "post-quantum-receipts" in str(exc_info.value)
    assert "mandatory" in str(exc_info.value)


def test_explicit_supported_capabilities_arg_overrides_default(
    tmp_path: Path, hmac_key: bytes
) -> None:
    w = _new_writer(tmp_path, hmac_key)
    w.write_capability("post-quantum-receipts", mandatory=True)
    w.write_step({"step_index": 0, "kind": "llm_call"})
    w.close()

    trace = verify_trace(
        w.path, hmac_key, supported_capabilities={"post-quantum-receipts"}
    )
    assert trace.capabilities[0]["name"] == "post-quantum-receipts"


def test_env_var_overrides_default_supported_capabilities(
    tmp_path: Path, hmac_key: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = _new_writer(tmp_path, hmac_key)
    w.write_capability("post-quantum-receipts", mandatory=True)
    w.write_step({"step_index": 0, "kind": "llm_call"})
    w.close()

    monkeypatch.setenv(
        "STEPBACK_SUPPORTED_CAPABILITIES", "post-quantum-receipts,foo bar"
    )
    trace = verify_trace(w.path, hmac_key)
    assert any(c["name"] == "post-quantum-receipts" for c in trace.capabilities)


def test_env_var_empty_string_means_only_core(
    tmp_path: Path, hmac_key: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = _new_writer(tmp_path, hmac_key)
    w.write_capability("blobs", mandatory=True)  # in default allow-list
    w.write_step({"step_index": 0, "kind": "llm_call"})
    w.close()

    monkeypatch.setenv("STEPBACK_SUPPORTED_CAPABILITIES", "")
    with pytest.raises(TraceVerificationError):
        verify_trace(w.path, hmac_key)


def test_explicit_arg_takes_precedence_over_env(
    tmp_path: Path, hmac_key: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = _new_writer(tmp_path, hmac_key)
    w.write_capability("post-quantum-receipts", mandatory=True)
    w.write_step({"step_index": 0, "kind": "llm_call"})
    w.close()

    monkeypatch.setenv("STEPBACK_SUPPORTED_CAPABILITIES", "")
    trace = verify_trace(
        w.path, hmac_key, supported_capabilities={"post-quantum-receipts"}
    )
    assert trace.capabilities[0]["name"] == "post-quantum-receipts"


def test_capability_frames_are_hmac_chained(
    tmp_path: Path, hmac_key: bytes
) -> None:
    """Capability frames participate in the same HMAC chain so tampering
    is detected by the standard verifier."""
    w = _new_writer(tmp_path, hmac_key)
    w.write_capability("blobs", mandatory=True)
    w.write_step({"step_index": 0, "kind": "llm_call"})
    w.close()

    raw = Path(w.path).read_bytes()
    # Flip a byte well inside the file (capability frame body).
    needle = b'"blobs"'
    idx = raw.find(needle)
    assert idx != -1
    tampered = bytearray(raw)
    tampered[idx + 2] = (tampered[idx + 2] + 1) % 256
    Path(w.path).write_bytes(bytes(tampered))

    with pytest.raises(TraceVerificationError):
        verify_trace(w.path, hmac_key)


def test_multiple_capability_frames_preserve_order(
    tmp_path: Path, hmac_key: bytes
) -> None:
    w = _new_writer(tmp_path, hmac_key)
    w.write_capability("blobs", mandatory=False)
    w.write_capability("gzip-step-bodies", mandatory=True)
    w.write_capability("hmac-sha256-chain", mandatory=True, params={"key_id": "abc"})
    w.write_step({"step_index": 0, "kind": "llm_call"})
    w.close()

    trace = verify_trace(w.path, hmac_key)
    names = [c["name"] for c in trace.capabilities]
    assert names == ["blobs", "gzip-step-bodies", "hmac-sha256-chain"]
    assert trace.capabilities[2]["params"] == {"key_id": "abc"}


def test_mandatory_unknown_among_known_still_fails(
    tmp_path: Path, hmac_key: bytes
) -> None:
    w = _new_writer(tmp_path, hmac_key)
    w.write_capability("blobs", mandatory=True)
    w.write_capability("post-quantum-receipts", mandatory=True)
    w.write_capability("hmac-sha256-chain", mandatory=True)
    w.write_step({"step_index": 0, "kind": "llm_call"})
    w.close()

    with pytest.raises(TraceVerificationError) as exc_info:
        verify_trace(w.path, hmac_key)
    msg = str(exc_info.value)
    assert "post-quantum-receipts" in msg
    # The two known names should NOT be in the rejection message.
    assert "'blobs'" not in msg
    assert "'hmac-sha256-chain'" not in msg


def test_capability_frame_works_with_compression(
    tmp_path: Path, hmac_key: bytes
) -> None:
    """When compression=True step bodies are deferred; capability frames
    must still land in the on-disk order callers wrote them."""
    w = TraceWriter.open(
        str(tmp_path / "t.sb"), hmac_key=hmac_key, compression=True
    )
    w.write_step({"step_index": 0, "kind": "llm_call", "payload": "x" * 500})
    w.write_capability("post-quantum-receipts", mandatory=False)
    w.write_step({"step_index": 1, "kind": "llm_call", "payload": "y" * 500})
    w.close()

    trace = verify_trace(w.path, hmac_key)
    assert [c["name"] for c in trace.capabilities] == ["post-quantum-receipts"]
    assert len(trace.steps) == 2
    assert trace.steps[0]["step_index"] == 0
    assert trace.steps[1]["step_index"] == 1


def test_malformed_capability_frame_is_corruption_not_negotiation(
    tmp_path: Path, hmac_key: bytes
) -> None:
    """A capability frame missing 'name' is a reader error, not a
    negotiation result. We construct one by surgically inserting a
    valid HMAC-chained frame via the writer's private API."""
    w = _new_writer(tmp_path, hmac_key)
    # Bypass write_capability's validation to forge a malformed body.
    w._write_frame({"type": "capability", "mandatory": True})
    w.write_step({"step_index": 0, "kind": "llm_call"})
    w.close()

    with pytest.raises(TraceVerificationError) as exc_info:
        verify_trace(w.path, hmac_key)
    assert "name" in str(exc_info.value)
