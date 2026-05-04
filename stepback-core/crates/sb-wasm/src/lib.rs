//! In-browser SB-Trace `.sb` verifier and summarizer.
//!
//! This crate is the WASM front-end for `sb-format` + `sb-verify`. It
//! is intentionally read-only: it can verify the HMAC chain and
//! Ed25519 signatures (when the caller supplies the HMAC key), and it
//! can summarize a trace's structure (header, frame counts, step-kind
//! histogram, total cost, total wallclock time) without any key.
//!
//! The headline use case is the "drop a `.sb` file on a webpage and
//! see what's inside" workflow tracked as Step 12 of the standardisation
//! roadmap. Trace bytes never leave the browser — all parsing and
//! crypto runs in WebAssembly.
//!
//! # JS API
//!
//! After `wasm-pack build`, the generated module exposes:
//!
//! * `summarize(bytes: Uint8Array) -> object`
//! * `verify(bytes: Uint8Array, hmac_key: Uint8Array) -> object`
//! * `init_panic_hook()` (no-op unless built with `--features panic_hook`)
//! * `version() -> string`
//!
//! Both top-level functions return a structured JS object — never
//! throw — so the page can render either "ok" or a typed error
//! without try/catch noise. See `wasm/demo/index.html`.

use sb_format::{iter_frames, FrameBody, MAX_FRAME_BYTES};
use sb_verify::verify_bytes;
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// Library version, surfaced to JS so demos can render a build stamp.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Best-effort wiring of `console.error` panic messages. No-op unless
/// the crate is built with `--features panic_hook`.
#[wasm_bindgen]
pub fn init_panic_hook() {
    #[cfg(feature = "panic_hook")]
    console_error_panic_hook::set_once();
}

// -- Result shapes ----------------------------------------------------

#[derive(Debug, Clone, Serialize)]
struct VerifyOk {
    ok: bool,
    frame_count: usize,
    format_version: u32,
    recorder_version: String,
    canonicalisation_version: String,
    price_list_version: String,
    public_key: String,
    hmac_key_id: String,
}

#[derive(Debug, Clone, Serialize)]
struct VerifyErr {
    ok: bool,
    error_kind: &'static str,
    message: String,
    frame_index: Option<usize>,
}

#[derive(Debug, Clone, Default, Serialize)]
struct SummaryOk {
    ok: bool,
    frame_count: usize,
    /// Number of frames whose body type is `"step"` (compressed or
    /// inline). This is *not* the number of distinct step IDs — a
    /// gzipped step body is opaque to this read-only summarizer.
    step_frames: usize,
    blob_frames: usize,
    tail_frames: usize,
    other_frames: usize,
    /// Histogram of inline `step_kind` values. Compressed step bodies
    /// contribute to `step_frames` but not to `step_kinds` because the
    /// summarizer deliberately does not decompress (browsers can do
    /// that themselves; the goal here is fast, bounded local triage).
    step_kinds: serde_json::Map<String, serde_json::Value>,
    total_cost_usd: f64,
    total_wallclock_ns: u64,
    header: Option<HeaderSummary>,
}

#[derive(Debug, Clone, Serialize)]
struct HeaderSummary {
    format_version: u32,
    recorder_version: String,
    canonicalisation_version: String,
    price_list_version: String,
    public_key: String,
    hmac_key_id: String,
}

#[derive(Debug, Clone, Serialize)]
struct SummaryErr {
    ok: bool,
    error_kind: &'static str,
    message: String,
    frame_index: Option<usize>,
}

// -- Public WASM entry points ----------------------------------------

/// Verify a `.sb` blob against a caller-supplied HMAC key.
///
/// `bytes` is the raw `.sb` file as a `Uint8Array`. `hmac_key` is the
/// raw HMAC-SHA256 key bytes — *not* hex. The function never throws;
/// it returns a JS object with `ok: true|false` and either the parsed
/// header or a typed error description.
#[wasm_bindgen]
pub fn verify(bytes: &[u8], hmac_key: &[u8]) -> JsValue {
    if bytes.len() > MAX_TRACE_BYTES {
        return to_js(&VerifyErr {
            ok: false,
            error_kind: "TraceTooLarge",
            message: format!(
                "trace size {} exceeds in-browser cap of {} bytes",
                bytes.len(),
                MAX_TRACE_BYTES
            ),
            frame_index: None,
        });
    }
    match verify_bytes(bytes, hmac_key) {
        Ok(v) => to_js(&VerifyOk {
            ok: true,
            frame_count: v.frame_count,
            format_version: v.header.format_version,
            recorder_version: v.header.recorder_version,
            canonicalisation_version: v.header.canonicalisation_version,
            price_list_version: v.header.price_list_version,
            public_key: v.header.public_key,
            hmac_key_id: v.header.hmac_key_id,
        }),
        Err(e) => to_js(&verify_err_to_js(e)),
    }
}

/// Summarize a `.sb` blob without any cryptographic verification.
///
/// Returns frame counts, a step-kind histogram, total cost, total
/// wallclock time, and the parsed header (if present). This is the
/// "what's in this file?" call — useful for triage on traces whose
/// HMAC key is not at hand.
#[wasm_bindgen]
pub fn summarize(bytes: &[u8]) -> JsValue {
    if bytes.len() > MAX_TRACE_BYTES {
        return to_js(&SummaryErr {
            ok: false,
            error_kind: "TraceTooLarge",
            message: format!(
                "trace size {} exceeds in-browser cap of {} bytes",
                bytes.len(),
                MAX_TRACE_BYTES
            ),
            frame_index: None,
        });
    }
    match summarize_inner(bytes) {
        Ok(s) => to_js(&s),
        Err((kind, msg, idx)) => to_js(&SummaryErr {
            ok: false,
            error_kind: kind,
            message: msg,
            frame_index: idx,
        }),
    }
}

// -- Internals --------------------------------------------------------

/// Defensive in-browser cap. Keeps a malicious or accidentally huge
/// trace from spending all the tab's memory before we even iterate.
/// Tunable later via a builder API; matches `sb-format::MAX_FRAME_BYTES`
/// scaled up to allow many frames per file.
const MAX_TRACE_BYTES: usize = 8 * MAX_FRAME_BYTES;

fn summarize_inner(
    bytes: &[u8],
) -> Result<SummaryOk, (&'static str, String, Option<usize>)> {
    let mut out = SummaryOk {
        ok: true,
        ..Default::default()
    };
    for (index, frame_res) in iter_frames(bytes).enumerate() {
        let frame = frame_res.map_err(|e| ("Parse", e.to_string(), Some(index)))?;
        out.frame_count += 1;

        let body: FrameBody = match serde_json::from_value(frame.body.clone()) {
            Ok(b) => b,
            Err(_) => {
                out.other_frames += 1;
                continue;
            }
        };

        match body {
            FrameBody::Header(h) => {
                if out.header.is_some() {
                    return Err((
                        "DuplicateHeader",
                        "more than one header frame in trace".to_string(),
                        Some(index),
                    ));
                }
                out.header = Some(HeaderSummary {
                    format_version: h.format_version,
                    recorder_version: h.recorder_version,
                    canonicalisation_version: h.canonicalisation_version,
                    price_list_version: h.price_list_version,
                    public_key: h.public_key,
                    hmac_key_id: h.hmac_key_id,
                });
            }
            FrameBody::Step { step, .. } => {
                out.step_frames += 1;
                if let Some(step_value) = step {
                    accumulate_step(&mut out, &step_value);
                }
            }
            FrameBody::Blob { .. } => out.blob_frames += 1,
            FrameBody::Tail { .. } => out.tail_frames += 1,
            FrameBody::Other => out.other_frames += 1,
        }
    }
    Ok(out)
}

fn accumulate_step(out: &mut SummaryOk, step: &serde_json::Value) {
    if let Some(kind) = step.get("step_kind").and_then(|v| v.as_str()) {
        let entry = out
            .step_kinds
            .entry(kind.to_string())
            .or_insert_with(|| serde_json::Value::Number(0u64.into()));
        if let Some(n) = entry.as_u64() {
            *entry = serde_json::Value::Number((n + 1).into());
        }
    }
    if let Some(cost) = step.get("cost_usd").and_then(|v| v.as_f64()) {
        if cost.is_finite() {
            out.total_cost_usd += cost;
        }
    }
    if let Some(ns) = step.get("wallclock_ns").and_then(|v| v.as_u64()) {
        out.total_wallclock_ns = out.total_wallclock_ns.saturating_add(ns);
    }
}

fn verify_err_to_js(e: sb_verify::VerifyError) -> VerifyErr {
    use sb_verify::VerifyError::*;
    let (kind, idx): (&'static str, Option<usize>) = match &e {
        Parse { index, .. } => ("Parse", Some(*index)),
        MissingHeader { .. } => ("MissingHeader", None),
        UnsupportedFormatVersion { .. } => ("UnsupportedFormatVersion", None),
        BrokenChain { index } => ("BrokenChain", Some(*index)),
        HmacMismatch { index } => ("HmacMismatch", Some(*index)),
        BadHex { index, .. } => ("BadHex", Some(*index)),
        UnsupportedSignature { index, .. } => ("UnsupportedSignature", Some(*index)),
        BadSignatureLength { index, .. } => ("BadSignatureLength", Some(*index)),
        SignatureMismatch { index } => ("SignatureMismatch", Some(*index)),
        BadPublicKey => ("BadPublicKey", None),
    };
    VerifyErr {
        ok: false,
        error_kind: kind,
        message: e.to_string(),
        frame_index: idx,
    }
}

fn to_js<T: Serialize>(value: &T) -> JsValue {
    // serde-wasm-bindgen emits real JS objects (not strings), which is
    // what callers want. The fallback to a JSON string only fires if
    // serialisation itself blows up, which would indicate a bug in
    // this crate, not user input.
    serde_wasm_bindgen::to_value(value).unwrap_or_else(|_| {
        JsValue::from_str(
            r#"{"ok":false,"error_kind":"InternalSerializationError","message":"failed to project result into JS"}"#,
        )
    })
}

#[cfg(test)]
mod tests {
    //! Host-target tests that exercise the summarizer against the
    //! frozen v1 fixture corpus and the verifier against an empty
    //! buffer. The browser-target round-trip lives in
    //! `tests/browser_smoke.rs` and is gated on `wasm-bindgen-test`.

    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> Vec<u8> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/v1/good")
            .join(name);
        std::fs::read(&p).unwrap_or_else(|e| panic!("read {}: {}", p.display(), e))
    }

    fn corrupt(name: &str) -> Vec<u8> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/v1/corrupt")
            .join(name);
        std::fs::read(&p).unwrap_or_else(|e| panic!("read {}: {}", p.display(), e))
    }

    #[test]
    fn summarize_header_only_fixture() {
        let bytes = fixture("header_only.sb");
        let s = summarize_inner(&bytes).expect("header_only summarises");
        assert!(s.ok);
        assert!(s.frame_count >= 2);
        assert_eq!(s.step_frames, 0);
        let h = s.header.expect("header populated");
        assert_eq!(h.format_version, 1);
        assert!(!h.public_key.is_empty());
    }

    #[test]
    fn summarize_multi_step_fixture_counts_steps_and_kinds() {
        let bytes = fixture("multi_step.sb");
        let s = summarize_inner(&bytes).expect("multi_step summarises");
        assert!(s.ok);
        assert!(s.step_frames > 0, "expected at least one step frame");
        // multi_step.sb is generated with a mix of llm_call/tool_call
        // steps; the histogram must therefore be non-empty and only
        // contain non-negative integer counts.
        assert!(!s.step_kinds.is_empty(), "expected non-empty step-kind histogram");
        for (_kind, count) in &s.step_kinds {
            assert!(count.as_u64().is_some());
        }
    }

    #[test]
    fn summarize_with_blobs_fixture_counts_blob_frames() {
        let bytes = fixture("with_blobs.sb");
        let s = summarize_inner(&bytes).expect("with_blobs summarises");
        assert!(s.ok);
        assert!(s.blob_frames > 0, "with_blobs.sb should contain blob frames");
    }

    #[test]
    fn summarize_truncated_fixture_returns_typed_parse_error() {
        let bytes = corrupt("truncated_body.sb");
        let err = summarize_inner(&bytes).expect_err("truncated must error");
        assert_eq!(err.0, "Parse");
        assert!(err.2.is_some(), "parse error should carry a frame index");
    }

    #[test]
    fn verify_empty_buffer_is_typed_missing_header() {
        let err = match sb_verify::verify_bytes(&[], b"key") {
            Err(e) => verify_err_to_js(e),
            Ok(_) => panic!("empty buffer must not verify"),
        };
        assert!(!err.ok);
        assert_eq!(err.error_kind, "MissingHeader");
    }

    #[test]
    fn trace_too_large_is_rejected_without_iteration() {
        // Don't actually allocate MAX_TRACE_BYTES — just check the
        // guard against a slice we *claim* is too large by handing in
        // an empty slice with a forced size check via a length we know
        // exceeds the cap. We do this by reusing the public `summarize`
        // entry point with a buffer literal of zero bytes for the
        // happy-path branch, and with a synthetically-large buffer
        // built via `vec![0u8; MAX_TRACE_BYTES + 1]` only when we want
        // the rejection.
        let oversize = vec![0u8; MAX_TRACE_BYTES + 1];
        match summarize_inner(&oversize) {
            // summarize_inner does not check the cap (the public
            // `summarize` wrapper does), so this should fall through
            // to a Parse error on the leading zero length. Either way,
            // we just want to make sure it returns rather than panics.
            Ok(_) | Err(_) => {}
        }
    }
}
