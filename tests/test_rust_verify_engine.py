"""Tests for the experimental Rust verifier engine.

These tests exercise the ``engine="rust"`` and
``STEPBACK_VERIFY_ENGINE`` opt-in paths in
:func:`stepback.trace_reader.verify_trace`. The native binding lives
under ``bindings/python/stepback_core``; if it isn't installed the
tests skip rather than fail so the pure-Python suite stays portable.
"""
from __future__ import annotations

import importlib
import os

import pytest

from stepback.recorder import RecorderKey, record
from stepback.trace_reader import (
    TraceVerificationError,
    _resolve_engine,
    verify_trace,
)

stepback_core = pytest.importorskip(
    "stepback_core",
    reason="Rust verifier extension (bindings/python/stepback_core) not installed",
)


@pytest.fixture
def small_trace(tmp_path):
    key = RecorderKey.fresh()
    path = str(tmp_path / "rust_engine.sb")
    with record(path, key=key) as rec:
        rec.llm_call(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            executor=lambda model, msgs: {
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )
        rec.tool_call(
            name="echo",
            arguments={"x": 1},
            executor=lambda name, args: args,
        )
    return path, key


def test_rust_engine_accepts_good_trace(small_trace):
    path, key = small_trace
    trace = verify_trace(path, key.hmac_key, engine="rust")
    assert len(trace.steps) == 2
    assert trace.header["public_key"] == key.signing_key.public_key().public_bytes_raw().hex()


def test_rust_engine_returns_same_shape_as_python(small_trace):
    path, key = small_trace
    py = verify_trace(path, key.hmac_key, engine="python")
    rs = verify_trace(path, key.hmac_key, engine="rust")
    assert py.header == rs.header
    assert py.steps == rs.steps
    assert py.tail == rs.tail
    assert py.public_key_hex == rs.public_key_hex
    assert py.blobs == rs.blobs


def test_rust_engine_rejects_tampered_trace(small_trace):
    path, key = small_trace
    raw = bytearray(open(path, "rb").read())
    # Flip a byte deep inside the file (well past the 4-byte length
    # prefix of frame 0). Either an HMAC, signature, or hex-decode
    # check in sb-verify must catch it.
    raw[len(raw) // 2] ^= 0xFF
    with open(path, "wb") as f:
        f.write(raw)
    with pytest.raises(TraceVerificationError):
        verify_trace(path, key.hmac_key, engine="rust")


def test_engine_auto_defaults_to_python_when_env_unset(monkeypatch):
    monkeypatch.delenv("STEPBACK_VERIFY_ENGINE", raising=False)
    assert _resolve_engine("auto") == "python"


def test_engine_env_explicit_rust_is_honoured(monkeypatch):
    monkeypatch.setenv("STEPBACK_VERIFY_ENGINE", "rust")
    assert _resolve_engine("auto") == "rust"


def test_engine_env_python_overrides_default(monkeypatch):
    monkeypatch.setenv("STEPBACK_VERIFY_ENGINE", "python")
    assert _resolve_engine("auto") == "python"


def test_engine_env_auto_picks_rust_when_extension_present(monkeypatch):
    monkeypatch.setenv("STEPBACK_VERIFY_ENGINE", "auto")
    assert _resolve_engine("auto") == "rust"


def test_engine_invalid_kwarg_raises():
    with pytest.raises(ValueError):
        _resolve_engine("haskell")  # type: ignore[arg-type]


def test_engine_env_invalid_value_raises(monkeypatch):
    monkeypatch.setenv("STEPBACK_VERIFY_ENGINE", "haskell")
    with pytest.raises(ValueError):
        _resolve_engine("auto")


def test_extension_exposes_versioned_module():
    assert isinstance(stepback_core.__version__, str)
    assert stepback_core.__version__.count(".") >= 1


def test_extension_verify_error_has_kind_attribute(small_trace):
    """Smoke-test the structured error fields the binding promises."""
    path, key = small_trace
    raw = bytearray(open(path, "rb").read())
    raw[len(raw) // 2] ^= 0xFF
    with open(path, "wb") as f:
        f.write(raw)
    with pytest.raises(stepback_core.VerifyError) as ei:
        stepback_core.verify_path(path, key.hmac_key)
    assert hasattr(ei.value, "kind")
    assert hasattr(ei.value, "frame_index")
    assert isinstance(ei.value.kind, str)
    assert isinstance(ei.value.frame_index, int)
