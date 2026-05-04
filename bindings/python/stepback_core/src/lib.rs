//! PyO3 bindings exposing the Rust `sb-verify` verifier to Python.
//!
//! This module is intentionally minimal: it wraps `verify_bytes` and
//! the `VerifyError` taxonomy from `sb_verify`, plus a convenience
//! `verify_path` that mmaps/reads a file and runs the verifier. The
//! goal is the Step 7 deliverable from `100_STEPS.md` — a Python-
//! callable Rust verifier behind an experimental flag — *not* a full
//! Rust replacement of `stepback.trace_reader`. Step decoding,
//! blob materialisation, and the `Trace_` shape stay in Python until
//! later steps in the Rust workstream.
//!
//! Public surface:
//!
//! * `verify_bytes(buf: bytes, hmac_key: bytes) -> VerifiedTrace`
//! * `verify_path(path: str, hmac_key: bytes) -> VerifiedTrace`
//! * `VerifiedTrace` — frozen dataclass-like object exposing
//!   `format_version`, `recorder_version`, `canonicalisation_version`,
//!   `price_list_version`, `public_key_hex`, `hmac_key_id`, and
//!   `frame_count`.
//! * `VerifyError` — Python exception subclass; every Rust
//!   `VerifyError` variant maps to an instance carrying a
//!   machine-readable `kind` string and (when applicable) a
//!   `frame_index`.

use std::fs;

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyOSError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use sb_verify::{verify_bytes as rust_verify_bytes, VerifiedTrace as RustVerifiedTrace, VerifyError};

create_exception!(
    _stepback_core,
    SbVerifyError,
    PyException,
    "Raised when an `.sb` trace fails Rust-side verification.\n\n\
     Carries `kind` (machine-readable error class, e.g. \
     'HmacMismatch', 'SignatureMismatch', 'BrokenChain') and \
     `frame_index` (0-based index of the offending frame, or -1 \
     when the error is not frame-local)."
);

/// Result of a successful verification. Mirrors `sb_verify::VerifiedTrace`
/// but lifts the header fields to top-level attributes so Python callers
/// don't need a second roundtrip through Rust to read them.
#[pyclass(name = "VerifiedTrace", frozen, module = "stepback_core")]
struct PyVerifiedTrace {
    #[pyo3(get)]
    format_version: u32,
    #[pyo3(get)]
    recorder_version: String,
    #[pyo3(get)]
    canonicalisation_version: String,
    #[pyo3(get)]
    price_list_version: String,
    #[pyo3(get)]
    public_key_hex: String,
    #[pyo3(get)]
    hmac_key_id: String,
    #[pyo3(get)]
    frame_count: usize,
}

#[pymethods]
impl PyVerifiedTrace {
    fn __repr__(&self) -> String {
        format!(
            "VerifiedTrace(format_version={}, recorder_version={:?}, \
             canonicalisation_version={:?}, price_list_version={:?}, \
             public_key_hex={:?}, hmac_key_id={:?}, frame_count={})",
            self.format_version,
            self.recorder_version,
            self.canonicalisation_version,
            self.price_list_version,
            self.public_key_hex,
            self.hmac_key_id,
            self.frame_count,
        )
    }
}

impl From<RustVerifiedTrace> for PyVerifiedTrace {
    fn from(v: RustVerifiedTrace) -> Self {
        PyVerifiedTrace {
            format_version: v.header.format_version,
            recorder_version: v.header.recorder_version,
            canonicalisation_version: v.header.canonicalisation_version,
            price_list_version: v.header.price_list_version,
            public_key_hex: v.header.public_key,
            hmac_key_id: v.header.hmac_key_id,
            frame_count: v.frame_count,
        }
    }
}

/// Convert a Rust `VerifyError` into a `SbVerifyError` Python exception
/// carrying machine-readable `kind` and `frame_index` attributes.
fn make_sb_verify_error(py: Python<'_>, err: VerifyError) -> PyErr {
    let (kind, frame_index): (&'static str, i64) = match &err {
        VerifyError::Parse { index, .. } => ("Parse", *index as i64),
        VerifyError::MissingHeader { .. } => ("MissingHeader", -1),
        VerifyError::UnsupportedFormatVersion { .. } => ("UnsupportedFormatVersion", 0),
        VerifyError::BrokenChain { index } => ("BrokenChain", *index as i64),
        VerifyError::HmacMismatch { index } => ("HmacMismatch", *index as i64),
        VerifyError::BadHex { index, .. } => ("BadHex", *index as i64),
        VerifyError::UnsupportedSignature { index, .. } => ("UnsupportedSignature", *index as i64),
        VerifyError::BadSignatureLength { index, .. } => ("BadSignatureLength", *index as i64),
        VerifyError::SignatureMismatch { index } => ("SignatureMismatch", *index as i64),
        VerifyError::BadPublicKey => ("BadPublicKey", -1),
    };
    let exc = SbVerifyError::new_err(err.to_string());
    // Attach structured fields as instance attributes so callers can
    // dispatch on `e.kind` instead of substring-matching the message.
    let exc_value = exc.value_bound(py);
    let _ = exc_value.setattr("kind", kind);
    let _ = exc_value.setattr("frame_index", frame_index);
    exc
}

/// Verify the bytes of a `.sb` trace.
#[pyfunction]
#[pyo3(signature = (buf, hmac_key))]
fn verify_bytes<'py>(
    py: Python<'py>,
    buf: &Bound<'py, PyBytes>,
    hmac_key: &Bound<'py, PyBytes>,
) -> PyResult<PyVerifiedTrace> {
    let buf_bytes = buf.as_bytes();
    let key_bytes = hmac_key.as_bytes();
    match rust_verify_bytes(buf_bytes, key_bytes) {
        Ok(v) => Ok(v.into()),
        Err(e) => Err(make_sb_verify_error(py, e)),
    }
}

/// Convenience: read `path` from disk and verify.
#[pyfunction]
#[pyo3(signature = (path, hmac_key))]
fn verify_path<'py>(
    py: Python<'py>,
    path: &str,
    hmac_key: &Bound<'py, PyBytes>,
) -> PyResult<PyVerifiedTrace> {
    let buf = fs::read(path)
        .map_err(|e| PyOSError::new_err(format!("could not read {:?}: {}", path, e)))?;
    let key_bytes = hmac_key.as_bytes();
    match rust_verify_bytes(&buf, key_bytes) {
        Ok(v) => Ok(v.into()),
        Err(e) => Err(make_sb_verify_error(py, e)),
    }
}

/// Module entry point. Mirrors the Cargo `[lib] name`.
#[pymodule]
fn _stepback_core(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("VerifyError", py.get_type_bound::<SbVerifyError>())?;
    m.add_class::<PyVerifiedTrace>()?;
    m.add_function(wrap_pyfunction!(verify_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(verify_path, m)?)?;
    Ok(())
}
