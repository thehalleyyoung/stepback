"""Experimental Rust acceleration for the stepback `.sb` verifier.

This package ships a PyO3 binding around the Rust `sb-verify` crate.
The pure-Python verifier in `stepback.trace_reader.verify_trace`
remains the default; this binding is opt-in behind the ``engine="rust"``
flag (or the ``STEPBACK_VERIFY_ENGINE=rust`` environment variable) so
that downstream users can flip the engine without taking a hard
dependency on the Rust toolchain.

Public surface (semver-tracked from 0.1):

* :func:`verify_bytes` — verify a `.sb` byte buffer.
* :func:`verify_path`  — verify a `.sb` on disk (uses the OS read).
* :class:`VerifiedTrace` — header summary returned on success.
* :class:`VerifyError`  — Python exception subclass raised on failure.
  Instances carry ``kind`` (e.g. ``"HmacMismatch"``) and
  ``frame_index`` attributes for programmatic dispatch.

Anything not listed here is internal and may change without notice.
"""
from __future__ import annotations

from . import _stepback_core as _native

verify_bytes = _native.verify_bytes
verify_path = _native.verify_path
VerifiedTrace = _native.VerifiedTrace
VerifyError = _native.VerifyError

__version__ = _native.__version__

__all__ = [
    "VerifiedTrace",
    "VerifyError",
    "__version__",
    "verify_bytes",
    "verify_path",
]
