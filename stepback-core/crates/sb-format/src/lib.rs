//! Data types and frame layout for the SB-Trace `.sb` v1 file format.
//!
//! On disk, every frame is:
//!
//! ```text
//!     | 4-byte big-endian length | canonical-JSON wrapper |
//! ```
//!
//! where the wrapper is:
//!
//! ```text
//!     {
//!       "body": <frame body>,
//!       "prev_hmac": "<hex>",
//!       "hmac":      "<hex>",
//!       "sig":       "ed25519:<hex>"
//!     }
//! ```
//!
//! This crate is intentionally I/O-policy-free: it owns the type
//! definitions, the constants pinned in the header, and a low-level
//! frame splitter. The HMAC chain check and the Ed25519 signature check
//! live in `sb-verify`. Replay live in `sb-replay`. Dirty-set lives in
//! `sb-dirty`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use sb_canonical::CANONICALISATION_VERSION;

/// `format_version` value pinned in every v1 header. Bumping this is a
/// hard wire-format break; v2 is reserved for the CBOR encoding the
/// roadmap discusses.
pub const FORMAT_VERSION: u32 = 1;

/// Width of the 4-byte big-endian length prefix that precedes every
/// frame on disk.
pub const FRAME_LENGTH_PREFIX: usize = 4;

/// Defensive cap on a single decoded frame, in bytes. Large enough for
/// any realistic agent step (LLM bodies, tool payloads), small enough
/// that a corrupted length prefix can't trigger a multi-gigabyte
/// allocation. Tunable behind a builder API later.
pub const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

/// All step kinds recognised in v1. Mirrors the Python `RecordedStep`
/// taxonomy. Unknown kinds are preserved as `Other(String)` so a v1
/// reader can round-trip frames it doesn't fully understand.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum StepKind {
    LlmCall,
    ToolCall,
    Router,
    PolicyCheck,
    McpCall,
    ParallelBranchOpen,
    ParallelBranchJoin,
    Exception,
    Other(String),
}

impl From<String> for StepKind {
    fn from(s: String) -> Self {
        match s.as_str() {
            "llm_call" => StepKind::LlmCall,
            "tool_call" => StepKind::ToolCall,
            "router" => StepKind::Router,
            "policy_check" => StepKind::PolicyCheck,
            "mcp_call" => StepKind::McpCall,
            "parallel_branch_open" => StepKind::ParallelBranchOpen,
            "parallel_branch_join" => StepKind::ParallelBranchJoin,
            "exception" => StepKind::Exception,
            _ => StepKind::Other(s),
        }
    }
}

impl From<StepKind> for String {
    fn from(k: StepKind) -> Self {
        match k {
            StepKind::LlmCall => "llm_call".into(),
            StepKind::ToolCall => "tool_call".into(),
            StepKind::Router => "router".into(),
            StepKind::PolicyCheck => "policy_check".into(),
            StepKind::McpCall => "mcp_call".into(),
            StepKind::ParallelBranchOpen => "parallel_branch_open".into(),
            StepKind::ParallelBranchJoin => "parallel_branch_join".into(),
            StepKind::Exception => "exception".into(),
            StepKind::Other(s) => s,
        }
    }
}

/// Header frame body — the very first frame of every `.sb`. Pins all
/// the versions a reader needs to decide whether it can interpret the
/// rest of the file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceHeader {
    pub format_version: u32,
    pub recorder_version: String,
    pub canonicalisation_version: String,
    pub price_list_version: String,
    /// Hex-encoded Ed25519 public key.
    pub public_key: String,
    /// Identifier of the HMAC key the writer used. Lookup is the
    /// caller's responsibility — this crate doesn't manage keys.
    pub hmac_key_id: String,
    /// Free-form metadata bag. Reserved keys are documented in the
    /// SB-Trace spec; unknown keys MUST round-trip.
    #[serde(default)]
    pub meta: serde_json::Map<String, serde_json::Value>,
}

/// One recorded step — what `stepback/recorder.py` emits per
/// LLM/tool/router/policy/MCP call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedStep {
    pub step_id: String,
    pub step_kind: StepKind,
    #[serde(default)]
    pub parent_step_id: Option<String>,
    pub inputs: serde_json::Value,
    pub outputs: serde_json::Value,
    /// `sha256:<hex>` of the canonical-JSON form of `inputs`.
    pub inputs_hash: String,
    /// `sha256:<hex>` of the canonical-JSON form of any
    /// non-deterministic inputs consumed by the step.
    #[serde(default)]
    pub nondeterminism_hash: Option<String>,
    #[serde(default)]
    pub wallclock_ns: Option<u64>,
    #[serde(default)]
    pub cpu_ns: Option<u64>,
    #[serde(default)]
    pub cost_usd: Option<f64>,
}

/// Discriminated union of every kind of frame body v1 may contain.
///
/// The discriminator is `"type"` to match the Python reference
/// recorder (`stepback/trace_writer.py`). The `Header` variant is
/// fully typed because verification needs the pinned `format_version`
/// and `public_key`; the other variants are intentionally permissive
/// — the verifier only needs to identify the header frame, and
/// downstream readers (sb-replay, importers) classify steps and
/// blobs themselves.
///
/// Frames with an unrecognised `type` discriminator deserialize as
/// [`FrameBody::Other`] so the verifier can still HMAC/sig-check
/// them and refuse to claim semantic understanding.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum FrameBody {
    /// First frame of every `.sb`. Pins the wire-format and signing
    /// key. Unknown sibling fields (the Python recorder writes
    /// `magic`, `compression`, `wallclock_ns`, `blob_threshold`,
    /// `blob_min_reuse`, ...) are ignored by serde and round-trip
    /// through the raw `serde_json::Value` the verifier canonicalises
    /// for HMAC recomputation.
    #[serde(rename = "header")]
    Header(TraceHeader),
    /// Either an inline step (`{"type":"step","step":{...}}`) or a
    /// gzip+base64-compressed step
    /// (`{"type":"step","encoding":"gzip+base64","data":"..."}`).
    #[serde(rename = "step")]
    Step {
        #[serde(default)]
        step: Option<serde_json::Value>,
        #[serde(default)]
        encoding: Option<String>,
        #[serde(default)]
        data: Option<String>,
    },
    /// Content-addressed blob frame written when the recorder
    /// detects a sub-tree that recurs across steps.
    #[serde(rename = "blob")]
    Blob {
        id: String,
        encoding: String,
        data: String,
    },
    /// Last frame written by `TraceWriter.close()`. Permissive
    /// because future versions may add fields.
    #[serde(rename = "tail")]
    Tail {
        #[serde(default)]
        wallclock_ns: Option<u64>,
    },
    /// Catch-all for forward-compatible frame types. The verifier
    /// still HMAC/sig-checks `Other` frames; only semantic
    /// interpretation is deferred to a newer reader.
    #[serde(other)]
    Other,
}

/// Wrapper as it appears on disk after the length prefix is consumed.
/// The HMAC chain and the per-frame Ed25519 signature are validated by
/// `sb-verify`; this struct merely owns the parsed shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    pub body: serde_json::Value,
    pub prev_hmac: String,
    pub hmac: String,
    pub sig: String,
}

/// Errors that can occur while splitting a byte stream into frames.
#[derive(Debug, Error)]
pub enum FrameError {
    #[error("unexpected end of input while reading frame {role}")]
    UnexpectedEof { role: &'static str },
    #[error("frame length {len} exceeds MAX_FRAME_BYTES={max}")]
    FrameTooLarge { len: usize, max: usize },
    #[error("frame wrapper was not valid JSON: {0}")]
    BadJson(#[from] serde_json::Error),
}

/// Split a contiguous byte slice into framed wrappers.
///
/// This does **no** crypto verification. Use `sb-verify` for that.
/// It exists here so that `sb-verify`, `sb-replay`, and the eventual
/// importers all share one frame splitter.
pub fn iter_frames(mut buf: &[u8]) -> impl Iterator<Item = Result<Frame, FrameError>> + '_ {
    std::iter::from_fn(move || {
        if buf.is_empty() {
            return None;
        }
        if buf.len() < FRAME_LENGTH_PREFIX {
            return Some(Err(FrameError::UnexpectedEof { role: "length-prefix" }));
        }
        let len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        buf = &buf[FRAME_LENGTH_PREFIX..];
        if len > MAX_FRAME_BYTES {
            return Some(Err(FrameError::FrameTooLarge {
                len,
                max: MAX_FRAME_BYTES,
            }));
        }
        if buf.len() < len {
            return Some(Err(FrameError::UnexpectedEof { role: "body" }));
        }
        let (frame_bytes, rest) = buf.split_at(len);
        buf = rest;
        Some(serde_json::from_slice::<Frame>(frame_bytes).map_err(FrameError::from))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn step_kind_round_trip_known() {
        let s: String = StepKind::LlmCall.into();
        assert_eq!(s, "llm_call");
        let k: StepKind = "tool_call".to_string().into();
        assert_eq!(k, StepKind::ToolCall);
    }

    #[test]
    fn step_kind_round_trip_unknown_preserved() {
        let k: StepKind = "future_kind".to_string().into();
        assert_eq!(k, StepKind::Other("future_kind".into()));
        let s: String = k.into();
        assert_eq!(s, "future_kind");
    }

    #[test]
    fn iter_frames_splits_two_frames() {
        let f1 = serde_json::to_vec(&json!({
            "body": {"type": "tail"},
            "prev_hmac": "00",
            "hmac": "11",
            "sig": "ed25519:22"
        }))
        .unwrap();
        let f2 = serde_json::to_vec(&json!({
            "body": {"type": "tail"},
            "prev_hmac": "11",
            "hmac": "33",
            "sig": "ed25519:44"
        }))
        .unwrap();
        let mut buf = Vec::new();
        for f in [&f1, &f2] {
            buf.extend_from_slice(&(f.len() as u32).to_be_bytes());
            buf.extend_from_slice(f);
        }
        let frames: Vec<_> = iter_frames(&buf).collect::<Result<_, _>>().unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].hmac, "11");
        assert_eq!(frames[1].prev_hmac, "11");
    }

    #[test]
    fn iter_frames_rejects_truncated_body() {
        let f1 = b"{\"body\":{}, \"prev_hmac\":\"\",\"hmac\":\"\",\"sig\":\"\"}";
        let mut buf = Vec::new();
        buf.extend_from_slice(&((f1.len() + 10) as u32).to_be_bytes());
        buf.extend_from_slice(f1);
        let err = iter_frames(&buf).next().unwrap().unwrap_err();
        matches!(err, FrameError::UnexpectedEof { .. });
    }

    #[test]
    fn iter_frames_rejects_oversized_length_prefix() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&u32::MAX.to_be_bytes());
        let err = iter_frames(&buf).next().unwrap().unwrap_err();
        matches!(err, FrameError::FrameTooLarge { .. });
    }
}
