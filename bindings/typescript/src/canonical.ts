// Canonical UTF-8 JSON encoding plus `sha256:<hex>` content hashing for
// SB-Trace `.sb` v1 traces. This module is a bit-for-bit port of the
// Python reference in `stepback/canonical.py` and the Rust reference in
// `stepback-core/crates/sb-canonical/src/lib.rs`. If the bytes this
// module emits ever differ from those of the Python reference for the
// same logical value, the bug is here, not there.
//
// Rules:
//   * UTF-8 JSON
//   * keys sorted lexicographically at every depth
//   * no whitespace separators (`","` / `":"`)
//   * non-ASCII characters emitted as raw UTF-8 (no `\uXXXX` escape)
//   * NaN / +Infinity / -Infinity rejected
//   * control characters U+0000..U+001F escaped as lowercase `\u00xx`
//
// This module deliberately operates on `unknown` JSON-shaped values
// rather than typed step shapes so it can canonicalise an arbitrary
// frame body the verifier reads back from disk.

import { createHash } from "node:crypto";

/** Canonicalisation version. Pinned in every `.sb` header.
 *  Mirrors `CANONICALISATION_VERSION` in `stepback/canonical.py`
 *  and `sb_canonical::CANONICALISATION_VERSION`. */
export const CANONICALISATION_VERSION = "1" as const;

/** Error thrown when a value cannot be canonicalised
 *  (currently only non-finite floats). */
export class CanonicalError extends Error {
  override readonly name = "CanonicalError";
  constructor(message: string) {
    super(message);
  }
}

/**
 * Encode `value` as canonical UTF-8 JSON bytes.
 *
 * Output is byte-equal to `stepback.canonical.canonical_json` for the
 * same logical value, modulo Python's `repr(float)` quirks for floats
 * that don't fit in i64/u64. Trace-cache safety relies on this.
 */
export function canonicalJson(value: unknown): Uint8Array {
  const chunks: number[] = [];
  writeValue(value, chunks);
  return new Uint8Array(chunks);
}

/**
 * Convenience: canonicalise + sha256 in one shot, returning the
 * `sha256:<lowercase-hex>` string used everywhere in `.sb`.
 */
export function hashValue(value: unknown): string {
  return sha256Hex(canonicalJson(value));
}

/**
 * `sha256:<lowercase-hex>` of `data`. Matches the Python and Rust
 * reference implementations so hashes can be compared as plain strings.
 */
export function sha256Hex(data: Uint8Array): string {
  const digest = createHash("sha256").update(data).digest("hex");
  return `sha256:${digest}`;
}

function writeValue(value: unknown, out: number[]): void {
  if (value === null) {
    pushAscii(out, "null");
    return;
  }
  if (value === true) {
    pushAscii(out, "true");
    return;
  }
  if (value === false) {
    pushAscii(out, "false");
    return;
  }
  switch (typeof value) {
    case "number":
      writeNumber(value, out);
      return;
    case "bigint":
      pushAscii(out, value.toString(10));
      return;
    case "string":
      writeString(value, out);
      return;
    case "object":
      if (Array.isArray(value)) {
        writeArray(value, out);
        return;
      }
      writeObject(value as Record<string, unknown>, out);
      return;
  }
  throw new CanonicalError(`unsupported value type: ${typeof value}`);
}

function writeArray(arr: readonly unknown[], out: number[]): void {
  out.push(0x5b); // '['
  for (let i = 0; i < arr.length; i++) {
    if (i > 0) out.push(0x2c); // ','
    writeValue(arr[i], out);
  }
  out.push(0x5d); // ']'
}

function writeObject(obj: Record<string, unknown>, out: number[]): void {
  // Lexicographic sort over UTF-16 code units matches Python's
  // `sorted(dict.keys())` for the ASCII-keyed dictionaries that appear
  // inside `.sb` frame bodies; non-ASCII keys would need a code-point
  // sort, which UTF-16 produces correctly except for surrogate-pair
  // edge cases that don't appear in `.sb` v1 frame schemas.
  const keys = Object.keys(obj).sort();
  out.push(0x7b); // '{'
  for (let i = 0; i < keys.length; i++) {
    if (i > 0) out.push(0x2c); // ','
    const k = keys[i]!;
    writeString(k, out);
    out.push(0x3a); // ':'
    writeValue(obj[k], out);
  }
  out.push(0x7d); // '}'
}

function writeNumber(n: number, out: number[]): void {
  if (!Number.isFinite(n)) {
    throw new CanonicalError("non-finite float is not canonicalisable");
  }
  if (Number.isInteger(n) && Object.is(n, Math.trunc(n))) {
    // Avoid the "1" vs "1.0" ambiguity by emitting integers without a
    // fractional part, matching Python's `json.dumps(1)` -> `"1"`.
    pushAscii(out, n.toString(10));
    return;
  }
  // For floats, JS `Number.prototype.toString` emits the shortest
  // round-trip representation, which agrees with Python's
  // `repr(float)` in the common cases. Float exact-equality is
  // already a known interoperability follow-up tracked by the
  // `sb-canonical` crate; we mirror the same trade-off.
  pushAscii(out, n.toString(10));
}

function writeString(s: string, out: number[]): void {
  out.push(0x22); // '"'
  for (let i = 0; i < s.length; i++) {
    const code = s.charCodeAt(i);
    if (code === 0x22) {
      pushAscii(out, '\\"');
    } else if (code === 0x5c) {
      pushAscii(out, "\\\\");
    } else if (code === 0x0a) {
      pushAscii(out, "\\n");
    } else if (code === 0x0d) {
      pushAscii(out, "\\r");
    } else if (code === 0x09) {
      pushAscii(out, "\\t");
    } else if (code === 0x08) {
      pushAscii(out, "\\b");
    } else if (code === 0x0c) {
      pushAscii(out, "\\f");
    } else if (code < 0x20) {
      pushAscii(out, "\\u" + code.toString(16).padStart(4, "0"));
    } else if (code < 0x80) {
      out.push(code);
    } else if (code < 0x800) {
      out.push(0xc0 | (code >> 6));
      out.push(0x80 | (code & 0x3f));
    } else if (code >= 0xd800 && code <= 0xdbff) {
      // High surrogate — combine with following low surrogate.
      const next = s.charCodeAt(i + 1);
      if (next >= 0xdc00 && next <= 0xdfff) {
        const cp = 0x10000 + (((code - 0xd800) << 10) | (next - 0xdc00));
        out.push(0xf0 | (cp >> 18));
        out.push(0x80 | ((cp >> 12) & 0x3f));
        out.push(0x80 | ((cp >> 6) & 0x3f));
        out.push(0x80 | (cp & 0x3f));
        i++;
      } else {
        // Lone high surrogate — emit as U+FFFD to stay valid UTF-8.
        pushReplacement(out);
      }
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      pushReplacement(out);
    } else {
      out.push(0xe0 | (code >> 12));
      out.push(0x80 | ((code >> 6) & 0x3f));
      out.push(0x80 | (code & 0x3f));
    }
  }
  out.push(0x22); // '"'
}

function pushAscii(out: number[], s: string): void {
  for (let i = 0; i < s.length; i++) {
    out.push(s.charCodeAt(i));
  }
}

function pushReplacement(out: number[]): void {
  // U+FFFD as UTF-8: 0xEF 0xBF 0xBD.
  out.push(0xef, 0xbf, 0xbd);
}
