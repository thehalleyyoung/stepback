//! End-to-end conformance tests against the frozen Python-written
//! `.sb` corpus under `stepback-core/fixtures/v1/`.
//!
//! These tests are the "wire it to the frozen fixture corpus" half
//! of Step 6: the Rust verifier here MUST accept every fixture in
//! `good/` (chain valid, signatures valid) and reject every fixture
//! in `corrupt/` with the corresponding error class declared in
//! `manifest.json`.
//!
//! Regenerate the corpus with::
//!
//!     python3 stepback-core/scripts/gen_fixtures.py
//!
//! The fixtures are intentionally bit-stable: pinned HMAC + Ed25519
//! keys, monotonic synthetic clock, no `os.urandom`. Any unexpected
//! diff after rerunning the generator is a wire-format bug somewhere.

use std::fs;
use std::path::{Path, PathBuf};

use sb_verify::{verify_bytes, VerifyError};
use serde_json::Value;
use sha2::{Digest, Sha256};

fn fixture_root() -> PathBuf {
    // CARGO_MANIFEST_DIR for this crate points at
    // .../stepback-core/crates/sb-verify; walk up two levels.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("v1")
}

fn manifest() -> Value {
    let path = fixture_root().join("manifest.json");
    let raw = fs::read(&path)
        .unwrap_or_else(|e| panic!("missing fixture manifest at {}: {e}", path.display()));
    serde_json::from_slice(&raw).expect("manifest.json must be valid JSON")
}

fn hmac_key() -> Vec<u8> {
    let m = manifest();
    let s = m["hmac_key_hex"].as_str().expect("hmac_key_hex");
    hex::decode(s).expect("hmac_key_hex must be hex")
}

fn read_fixture(rel: &str) -> Vec<u8> {
    let path = fixture_root().join(rel);
    fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {}: {e}", path.display()))
}

fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    hex::encode(h.finalize())
}

#[test]
fn manifest_is_present_and_keyed() {
    let m = manifest();
    assert_eq!(m["format_version"], 1);
    assert!(m["hmac_key_hex"].as_str().unwrap().len() == 64);
    assert!(m["public_key_hex"].as_str().unwrap().len() == 64);
    assert!(m["good"].as_array().unwrap().len() >= 1);
    assert!(m["corrupt"].as_array().unwrap().len() >= 1);
}

#[test]
fn fixture_bytes_match_recorded_sha256() {
    // Catches accidental edits to the committed fixtures and keeps
    // the manifest honest: the Rust test corpus and the Python
    // generator stay byte-locked.
    let m = manifest();
    for entry in m["good"].as_array().unwrap() {
        let name = entry["name"].as_str().unwrap();
        let bytes = read_fixture(&format!("good/{name}"));
        assert_eq!(bytes.len() as u64, entry["size_bytes"].as_u64().unwrap());
        assert_eq!(sha256_hex(&bytes), entry["sha256"].as_str().unwrap(),
            "sha256 mismatch on good/{name}: regenerate fixtures or fix the writer");
    }
    for entry in m["corrupt"].as_array().unwrap() {
        let name = entry["name"].as_str().unwrap();
        let bytes = read_fixture(&format!("corrupt/{name}"));
        assert_eq!(bytes.len() as u64, entry["size_bytes"].as_u64().unwrap());
        assert_eq!(sha256_hex(&bytes), entry["sha256"].as_str().unwrap(),
            "sha256 mismatch on corrupt/{name}: regenerate fixtures or fix the writer");
    }
}

#[test]
fn good_fixtures_verify() {
    let key = hmac_key();
    let m = manifest();
    let public_key_hex = m["public_key_hex"].as_str().unwrap();
    for entry in m["good"].as_array().unwrap() {
        let name = entry["name"].as_str().unwrap();
        let bytes = read_fixture(&format!("good/{name}"));
        let verified = verify_bytes(&bytes, &key).unwrap_or_else(|e| {
            panic!("good fixture {name} should verify but failed: {e}")
        });
        assert_eq!(
            verified.header.format_version, 1,
            "fixture {name} header.format_version"
        );
        assert_eq!(
            verified.header.public_key, public_key_hex,
            "fixture {name} pinned public_key drifted from manifest"
        );
        let min = entry["expected_frame_count_min"].as_u64().unwrap() as usize;
        assert!(
            verified.frame_count >= min,
            "fixture {name}: expected at least {min} frames, got {}",
            verified.frame_count
        );
    }
}

#[test]
fn truncated_body_is_rejected_as_parse_error() {
    let bytes = read_fixture("corrupt/truncated_body.sb");
    let err = verify_bytes(&bytes, &hmac_key()).expect_err("must reject");
    assert!(
        matches!(err, VerifyError::Parse { .. }),
        "expected Parse, got {err:?}"
    );
}

#[test]
fn flipped_hmac_is_rejected_by_chain_or_hmac_check() {
    let bytes = read_fixture("corrupt/flipped_hmac.sb");
    let err = verify_bytes(&bytes, &hmac_key()).expect_err("must reject");
    // Flipping a hex nibble in a frame's `hmac` field invalidates
    // both the local HMAC check on that frame *and* the next frame's
    // chain link. Either is acceptable; both prove the verifier
    // refuses the trace.
    assert!(
        matches!(
            err,
            VerifyError::HmacMismatch { .. }
                | VerifyError::BrokenChain { .. }
                | VerifyError::BadHex { .. }
                | VerifyError::SignatureMismatch { .. }
        ),
        "unexpected error class: {err:?}"
    );
}

#[test]
fn flipped_signature_is_rejected_by_signature_check() {
    let bytes = read_fixture("corrupt/flipped_sig.sb");
    let err = verify_bytes(&bytes, &hmac_key()).expect_err("must reject");
    assert!(
        matches!(
            err,
            VerifyError::SignatureMismatch { .. } | VerifyError::BadHex { .. }
        ),
        "unexpected error class: {err:?}"
    );
}

#[test]
fn broken_chain_is_rejected() {
    let bytes = read_fixture("corrupt/broken_chain.sb");
    let err = verify_bytes(&bytes, &hmac_key()).expect_err("must reject");
    // Mutating prev_hmac changes both the chain comparison input AND
    // the HMAC body inputs; either rejection class is acceptable.
    assert!(
        matches!(
            err,
            VerifyError::BrokenChain { .. }
                | VerifyError::HmacMismatch { .. }
                | VerifyError::BadHex { .. }
        ),
        "unexpected error class: {err:?}"
    );
}

#[test]
fn bad_format_version_is_rejected() {
    let bytes = read_fixture("corrupt/bad_format_version.sb");
    let err = verify_bytes(&bytes, &hmac_key()).expect_err("must reject");
    // The HMAC over the header body is computed before we look at
    // format_version, so the recorded HMAC no longer matches the
    // forged body. The verifier reports HmacMismatch first; once
    // capability negotiation lands (Step 24) we'll also exercise
    // the explicit UnsupportedFormatVersion path.
    assert!(
        matches!(
            err,
            VerifyError::HmacMismatch { .. } | VerifyError::UnsupportedFormatVersion { .. }
        ),
        "unexpected error class: {err:?}"
    );
}

#[test]
fn wrong_hmac_key_is_rejected_on_first_frame() {
    let bytes = read_fixture("good/multi_step.sb");
    let bogus_key = vec![0xAB; 32];
    let err = verify_bytes(&bytes, &bogus_key).expect_err("must reject");
    assert!(
        matches!(err, VerifyError::HmacMismatch { index: 0 }),
        "wrong HMAC key must trip the first-frame HMAC check, got {err:?}"
    );
}
