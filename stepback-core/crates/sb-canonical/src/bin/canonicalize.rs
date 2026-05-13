//! `canonicalize` — small CLI that emits canonical-JSON bytes for a value
//! it parses from stdin. Used by the cross-language differential harness
//! in `spec/canonical/differential.py` (Step 48 of `100_STEPS.md`).
//!
//! Two modes:
//!
//! * **single** (default): reads one JSON document from stdin, writes the
//!   canonical bytes verbatim to stdout (no trailing newline).
//!
//! * **batch** (`--batch`): reads one JSON document per line from stdin and
//!   writes a stream of `<len>\n<bytes>\n` chunks to stdout. The leading
//!   length is decimal ASCII; the bytes that follow are exactly the
//!   canonical-JSON output for the corresponding input. The framing
//!   tolerates canonical-JSON bytes containing newlines (e.g. inside an
//!   escaped string) because the decoder uses the length, not a newline
//!   delimiter, to bound the chunk.

use std::io::Read as _;
use std::io::Write as _;

fn main() -> std::process::ExitCode {
    let mut argv = std::env::args().skip(1);
    let mode = argv.next();
    let batch = matches!(mode.as_deref(), Some("--batch"));

    let mut stdin = String::new();
    if let Err(err) = std::io::stdin().read_to_string(&mut stdin) {
        eprintln!("canonicalize: read stdin: {err}");
        return std::process::ExitCode::from(2);
    }

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    let process_one = |line: &str, out: &mut dyn std::io::Write, with_len: bool| -> Result<(), String> {
        let value: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| format!("parse: {e} (line={line:?})"))?;
        let bytes = sb_canonical::canonical_json(&value)
            .map_err(|e| format!("canonicalize: {e}"))?;
        if with_len {
            write!(out, "{}\n", bytes.len()).map_err(|e| e.to_string())?;
            out.write_all(&bytes).map_err(|e| e.to_string())?;
            out.write_all(b"\n").map_err(|e| e.to_string())?;
        } else {
            out.write_all(&bytes).map_err(|e| e.to_string())?;
        }
        Ok(())
    };

    if batch {
        for line in stdin.lines() {
            if line.is_empty() {
                continue;
            }
            if let Err(err) = process_one(line, &mut out, true) {
                eprintln!("canonicalize --batch: {err}");
                return std::process::ExitCode::from(1);
            }
        }
    } else {
        let trimmed = stdin.trim_end_matches(['\n', '\r']);
        if let Err(err) = process_one(trimmed, &mut out, false) {
            eprintln!("canonicalize: {err}");
            return std::process::ExitCode::from(1);
        }
    }

    std::process::ExitCode::SUCCESS
}
