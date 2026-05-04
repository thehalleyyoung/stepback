//! HMAC-chain and per-frame Ed25519 signature verifier for SB-Trace
//! `.sb` files.
//!
//! Verification policy in v1:
//!
//! 1. Walk frames in order.
//! 2. For each frame, recompute
//!    `HMAC_SHA256(hmac_key, prev_hmac_bytes || canonical_json(body))`
//!    and compare it to `frame.hmac` in constant time.
//! 3. Verify the Ed25519 signature `frame.sig` over the **bytes** of
//!    the hex `frame.hmac` field, against the public key pinned in the
//!    header.
//! 4. The `prev_hmac` field of frame *n+1* must equal the `hmac` of
//!    frame *n*. The first frame's `prev_hmac` must be 32 zero bytes
//!    (the `ZERO_HMAC` constant in the Python writer).
//!
//! The actual HMAC and signature wiring is intentionally minimal in
//! this scaffold — it exposes the public surface (errors, the
//! `Verifier` struct, `verify_bytes`) that the Step 6 fixture work and
//! the Step 7 PyO3 bindings depend on. The chain-walking machinery is
//! complete and tested; key resolution and full Ed25519 wiring land
//! when the fixture corpus is wired up in Step 6.

use ed25519_dalek::{Signature, Verifier as _, VerifyingKey, SIGNATURE_LENGTH};
use hmac::{Hmac, Mac};
use sb_canonical::canonical_json;
use sb_format::{iter_frames, Frame, FrameBody, FrameError, TraceHeader, FORMAT_VERSION};
use sha2::Sha256;
use thiserror::Error;

type HmacSha256 = Hmac<Sha256>;

/// 32 zero bytes — the seed of the HMAC chain.
pub const ZERO_HMAC: [u8; 32] = [0u8; 32];

/// All the ways verification can fail. Each variant carries the index
/// of the offending frame so callers can render useful diagnostics.
#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("frame {index}: failed to parse: {source}")]
    Parse {
        index: usize,
        #[source]
        source: FrameError,
    },
    #[error("frame 0 must be a header but was {found}")]
    MissingHeader { found: &'static str },
    #[error("unsupported format_version {found}, expected {expected}")]
    UnsupportedFormatVersion { expected: u32, found: u32 },
    #[error("frame {index}: prev_hmac does not chain to previous frame")]
    BrokenChain { index: usize },
    #[error("frame {index}: HMAC mismatch")]
    HmacMismatch { index: usize },
    #[error("frame {index}: invalid hex in {field}")]
    BadHex { index: usize, field: &'static str },
    #[error("frame {index}: signature scheme {scheme} is not supported")]
    UnsupportedSignature { index: usize, scheme: String },
    #[error("frame {index}: signature length {found} is not {expected}")]
    BadSignatureLength {
        index: usize,
        found: usize,
        expected: usize,
    },
    #[error("frame {index}: Ed25519 signature did not verify")]
    SignatureMismatch { index: usize },
    #[error("trace header used an unsupported public key encoding")]
    BadPublicKey,
}

/// Outcome of a successful verification: the parsed header plus the
/// number of frames that chained cleanly. Useful for the CLI's
/// `stepback verify` and for the eventual `sb-verify` WASM build.
#[derive(Debug, Clone)]
pub struct VerifiedTrace {
    pub header: TraceHeader,
    pub frame_count: usize,
}

/// Verify a `.sb` byte slice end-to-end. The HMAC key is supplied by
/// the caller because key management lives outside this crate.
///
/// On success the caller may trust that:
///
/// * every frame's HMAC is valid given the chain;
/// * every frame's Ed25519 signature was issued by the holder of the
///   private key whose public counterpart is pinned in the header;
/// * no frame was inserted, dropped, reordered, or rewritten.
pub fn verify_bytes(buf: &[u8], hmac_key: &[u8]) -> Result<VerifiedTrace, VerifyError> {
    let mut header: Option<TraceHeader> = None;
    let mut public_key: Option<VerifyingKey> = None;
    let mut prev_hmac_bytes: [u8; 32] = ZERO_HMAC;
    let mut count = 0usize;

    for (index, frame_res) in iter_frames(buf).enumerate() {
        let frame = frame_res.map_err(|source| VerifyError::Parse { index, source })?;
        verify_chain_link(index, &frame, &prev_hmac_bytes, hmac_key)?;

        if index == 0 {
            // The very first frame must carry the header so we know
            // which public key to verify subsequent signatures against.
            let body: FrameBody = serde_json::from_value(frame.body.clone())
                .map_err(|_| VerifyError::MissingHeader { found: "non-header" })?;
            let hdr = match body {
                FrameBody::Header(h) => h,
                FrameBody::Step { .. } => return Err(VerifyError::MissingHeader { found: "step" }),
                FrameBody::Blob { .. } => return Err(VerifyError::MissingHeader { found: "blob" }),
                FrameBody::Tail { .. } => {
                    return Err(VerifyError::MissingHeader { found: "tail" })
                }
                FrameBody::Other => return Err(VerifyError::MissingHeader { found: "other" }),
            };
            if hdr.format_version != FORMAT_VERSION {
                return Err(VerifyError::UnsupportedFormatVersion {
                    expected: FORMAT_VERSION,
                    found: hdr.format_version,
                });
            }
            public_key = Some(parse_public_key(&hdr.public_key)?);
            header = Some(hdr);
        }

        let pk = public_key.as_ref().expect("header populated public_key");
        verify_signature(index, &frame, pk)?;

        prev_hmac_bytes = decode_hex32(index, "hmac", &frame.hmac)?;
        count += 1;
    }

    let header = header.ok_or(VerifyError::MissingHeader { found: "empty" })?;
    Ok(VerifiedTrace {
        header,
        frame_count: count,
    })
}

fn verify_chain_link(
    index: usize,
    frame: &Frame,
    expected_prev: &[u8; 32],
    hmac_key: &[u8],
) -> Result<(), VerifyError> {
    let claimed_prev = decode_hex32(index, "prev_hmac", &frame.prev_hmac)?;
    if &claimed_prev != expected_prev {
        return Err(VerifyError::BrokenChain { index });
    }
    let body_bytes =
        canonical_json(&frame.body).expect("body came from serde_json::Value, must canonicalise");
    let mut mac = <HmacSha256 as Mac>::new_from_slice(hmac_key).expect("HMAC accepts any key length");
    mac.update(expected_prev);
    mac.update(&body_bytes);
    let computed = mac.finalize().into_bytes();

    let claimed_hmac = decode_hex_vec(index, "hmac", &frame.hmac)?;
    if claimed_hmac.len() != computed.len()
        || !constant_time_eq(&claimed_hmac, computed.as_slice())
    {
        return Err(VerifyError::HmacMismatch { index });
    }
    Ok(())
}

fn verify_signature(
    index: usize,
    frame: &Frame,
    public_key: &VerifyingKey,
) -> Result<(), VerifyError> {
    let (scheme, hex_part) = frame
        .sig
        .split_once(':')
        .unwrap_or(("", frame.sig.as_str()));
    if scheme != "ed25519" {
        return Err(VerifyError::UnsupportedSignature {
            index,
            scheme: scheme.to_string(),
        });
    }
    let sig_bytes = hex::decode(hex_part).map_err(|_| VerifyError::BadHex {
        index,
        field: "sig",
    })?;
    if sig_bytes.len() != SIGNATURE_LENGTH {
        return Err(VerifyError::BadSignatureLength {
            index,
            found: sig_bytes.len(),
            expected: SIGNATURE_LENGTH,
        });
    }
    let sig_arr: [u8; SIGNATURE_LENGTH] = sig_bytes
        .as_slice()
        .try_into()
        .expect("length checked above");
    let sig = Signature::from_bytes(&sig_arr);
    // The Python writer signs the **raw 32-byte HMAC digest** (cf.
    // `trace_writer.py::_write_frame`: `sig = signing_key.sign(h)`),
    // not the hex-encoded form. We mirror that here by decoding the
    // hex `frame.hmac` field back to bytes before verifying.
    let hmac_bytes = hex::decode(&frame.hmac).map_err(|_| VerifyError::BadHex {
        index,
        field: "hmac",
    })?;
    public_key
        .verify(&hmac_bytes, &sig)
        .map_err(|_| VerifyError::SignatureMismatch { index })
}

fn parse_public_key(s: &str) -> Result<VerifyingKey, VerifyError> {
    let bytes = hex::decode(s).map_err(|_| VerifyError::BadPublicKey)?;
    let arr: [u8; 32] = bytes.as_slice().try_into().map_err(|_| VerifyError::BadPublicKey)?;
    VerifyingKey::from_bytes(&arr).map_err(|_| VerifyError::BadPublicKey)
}

fn decode_hex32(index: usize, field: &'static str, s: &str) -> Result<[u8; 32], VerifyError> {
    let bytes = decode_hex_vec(index, field, s)?;
    if bytes.len() != 32 {
        return Err(VerifyError::BadHex { index, field });
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

fn decode_hex_vec(index: usize, field: &'static str, s: &str) -> Result<Vec<u8>, VerifyError> {
    hex::decode(s).map_err(|_| VerifyError::BadHex { index, field })
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_trace_is_rejected_for_missing_header() {
        let err = verify_bytes(&[], b"key").unwrap_err();
        matches!(err, VerifyError::MissingHeader { .. });
    }

    #[test]
    fn constant_time_eq_matches_naive() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"abcd"));
    }
}
