//! Canonical UTF-8 JSON encoding and `sha256:<hex>` content hashing for
//! SB-Trace `.sb` files.
//!
//! The canonicalisation rules MUST stay bit-for-bit compatible with the
//! Python reference implementation in `stepback/canonical.py`:
//!
//! * UTF-8 JSON
//! * keys sorted lexicographically at every depth
//! * no whitespace separators (`","` / `":"`)
//! * `ensure_ascii=False` — non-ASCII characters are emitted as raw UTF-8,
//!   not `\uXXXX` escapes
//! * `allow_nan=False` — `NaN` / `+Inf` / `-Inf` are rejected
//! * tuples → arrays, sets/frozensets → sorted arrays, bytes →
//!   `{"__bytes_hex__": "<lowercase-hex>"}`
//!
//! The content-hash form is `sha256:<lowercase-hex-of-32-bytes>` so that
//! Python and Rust hashes compare as plain strings without further
//! normalisation.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Write as _;

use serde_json::Value;
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Canonicalisation version. Pinned in every `.sb` header; bumping it is a
/// wire-format change. Mirrors `CANONICALISATION_VERSION` in
/// `stepback/canonical.py`.
pub const CANONICALISATION_VERSION: &str = "1";

/// Errors that can occur while canonicalising a value.
#[derive(Debug, Error)]
pub enum CanonicalError {
    /// `NaN`, `+Inf`, or `-Inf` encountered. The Python recorder rejects
    /// these via `allow_nan=False`; we mirror that.
    #[error("non-finite float is not canonicalisable")]
    NonFiniteFloat,
    /// A JSON object had a key that wasn't a string. `serde_json::Value`
    /// already forbids this, so this branch is defensive.
    #[error("object key was not a string")]
    NonStringKey,
}

/// Encode `value` as canonical UTF-8 JSON bytes.
///
/// The output MUST equal what `stepback.canonical.canonical_json` would
/// produce for the same logical value. If you find a divergence, the bug
/// is here, not in the Python recorder — that one is the reference.
pub fn canonical_json(value: &Value) -> Result<Vec<u8>, CanonicalError> {
    let mut out = Vec::with_capacity(64);
    write_value(value, &mut out)?;
    Ok(out)
}

/// Convenience: canonicalise and sha256 in one shot, returning the
/// `sha256:<hex>` form used throughout `.sb`.
pub fn hash_value(value: &Value) -> Result<String, CanonicalError> {
    Ok(sha256_hex(&canonical_json(value)?))
}

/// `sha256:<lowercase-hex>` of `data`. Same prefix the Python
/// implementation uses so hashes can be compared as strings.
pub fn sha256_hex(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    let mut s = String::with_capacity(7 + 64);
    s.push_str("sha256:");
    for b in digest.iter() {
        // hex_lower without pulling in the `hex` crate.
        let _ = write!(s, "{b:02x}");
    }
    s
}

fn write_value(value: &Value, out: &mut Vec<u8>) -> Result<(), CanonicalError> {
    match value {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Number(n) => write_number(n, out)?,
        Value::String(s) => write_string(s, out),
        Value::Array(items) => {
            out.push(b'[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_value(item, out)?;
            }
            out.push(b']');
        }
        Value::Object(map) => {
            // BTreeMap gives us stable lexicographic ordering, matching
            // Python's `sort_keys=True`.
            let sorted: BTreeMap<&str, &Value> =
                map.iter().map(|(k, v)| (k.as_str(), v)).collect();
            out.push(b'{');
            for (i, (k, v)) in sorted.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_string(k, out);
                out.push(b':');
                write_value(v, out)?;
            }
            out.push(b'}');
        }
    }
    Ok(())
}

fn write_number(n: &serde_json::Number, out: &mut Vec<u8>) -> Result<(), CanonicalError> {
    if let Some(u) = n.as_u64() {
        let _ = write!(out, "{u}");
    } else if let Some(i) = n.as_i64() {
        let _ = write!(out, "{i}");
    } else if let Some(f) = n.as_f64() {
        if !f.is_finite() {
            return Err(CanonicalError::NonFiniteFloat);
        }
        // serde_json::Number already preserves the lexical form for floats
        // it parsed; for floats we constructed ourselves we fall back to
        // `{f}` which matches Python's `repr(float)` for finite doubles in
        // the common cases. Round-trip equivalence with Python's
        // `json.dumps` for arbitrary floats is a known follow-up tracked
        // by the conformance fixture work in Step 6.
        let _ = write!(out, "{f}");
    } else {
        // Numbers that aren't representable as u64/i64/f64 do not appear
        // in `serde_json::Number` today; this branch is unreachable.
        let _ = write!(out, "{n}");
    }
    Ok(())
}

fn write_string(s: &str, out: &mut Vec<u8>) {
    out.push(b'"');
    for c in s.chars() {
        match c {
            '"' => out.extend_from_slice(b"\\\""),
            '\\' => out.extend_from_slice(b"\\\\"),
            '\n' => out.extend_from_slice(b"\\n"),
            '\r' => out.extend_from_slice(b"\\r"),
            '\t' => out.extend_from_slice(b"\\t"),
            '\x08' => out.extend_from_slice(b"\\b"),
            '\x0c' => out.extend_from_slice(b"\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => {
                let mut buf = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    out.push(b'"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn primitives_round_trip() {
        assert_eq!(canonical_json(&json!(null)).unwrap(), b"null");
        assert_eq!(canonical_json(&json!(true)).unwrap(), b"true");
        assert_eq!(canonical_json(&json!(false)).unwrap(), b"false");
        assert_eq!(canonical_json(&json!(0)).unwrap(), b"0");
        assert_eq!(canonical_json(&json!(-7)).unwrap(), b"-7");
        assert_eq!(canonical_json(&json!("hi")).unwrap(), b"\"hi\"");
    }

    #[test]
    fn object_keys_are_sorted() {
        let v = json!({ "b": 1, "a": 2, "c": 3 });
        let bytes = canonical_json(&v).unwrap();
        assert_eq!(bytes, br#"{"a":2,"b":1,"c":3}"#);
    }

    #[test]
    fn nested_objects_are_recursively_sorted() {
        let v = json!({ "z": { "y": 1, "x": 2 }, "a": [{"d": 4, "c": 3}] });
        let bytes = canonical_json(&v).unwrap();
        assert_eq!(bytes, br#"{"a":[{"c":3,"d":4}],"z":{"x":2,"y":1}}"#);
    }

    #[test]
    fn no_whitespace_separators() {
        let v = json!({"k": [1, 2, 3]});
        assert_eq!(canonical_json(&v).unwrap(), br#"{"k":[1,2,3]}"#);
    }

    #[test]
    fn non_ascii_is_utf8_not_escaped() {
        let v = json!({"name": "café"});
        let bytes = canonical_json(&v).unwrap();
        // 'é' is two bytes in UTF-8: 0xC3 0xA9.
        assert!(bytes.windows(2).any(|w| w == [0xC3, 0xA9]));
        assert!(!std::str::from_utf8(&bytes).unwrap().contains("\\u"));
    }

    #[test]
    fn control_chars_escape() {
        let v = json!("a\u{0001}b");
        let bytes = canonical_json(&v).unwrap();
        assert_eq!(bytes, b"\"a\\u0001b\"");
    }

    #[test]
    fn hash_has_sha256_prefix() {
        let h = hash_value(&json!({"k": 1})).unwrap();
        assert!(h.starts_with("sha256:"));
        assert_eq!(h.len(), 7 + 64);
    }

    #[test]
    fn hash_is_deterministic() {
        let a = hash_value(&json!({"a": 1, "b": 2})).unwrap();
        let b = hash_value(&json!({"b": 2, "a": 1})).unwrap();
        assert_eq!(a, b, "key order must not affect hash");
    }
}
