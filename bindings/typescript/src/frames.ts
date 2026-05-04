// On-disk frame layout and frame splitter for `.sb` v1 traces.
//
// Each frame on disk is:
//
//   | 4-byte big-endian length | canonical-JSON wrapper |
//
// where the wrapper is:
//
//   { "body": <frame body>,
//     "prev_hmac": "<hex>",
//     "hmac":      "<hex>",
//     "sig":       "ed25519:<hex>" }
//
// This module is intentionally I/O-policy-free: it owns the data
// shapes, pinned constants, and a low-level frame splitter. The HMAC
// chain check and the Ed25519 signature check live in `verify.ts`.

/** `format_version` value pinned in every v1 header. Bumping this is
 *  a hard wire-format break. */
export const FORMAT_VERSION = 1 as const;

/** Width of the 4-byte big-endian length prefix that precedes every
 *  frame on disk. */
export const FRAME_LENGTH_PREFIX = 4 as const;

/** Defensive cap on a single decoded frame, in bytes. Large enough
 *  for any realistic agent step, small enough that a corrupted length
 *  prefix can't trigger a multi-gigabyte allocation. */
export const MAX_FRAME_BYTES = 64 * 1024 * 1024;

/** Step kinds recognised in v1. Mirrors the Python `RecordedStep`
 *  taxonomy and the Rust `StepKind` enum. Unknown kinds are preserved
 *  verbatim as plain strings so that v1 readers can round-trip frames
 *  they don't fully understand. */
export type StepKind =
  | "llm_call"
  | "tool_call"
  | "router"
  | "policy_check"
  | "mcp_call"
  | "parallel_branch_open"
  | "parallel_branch_join"
  | "exception"
  | (string & {});

/** Header frame body — the very first frame of every `.sb`. Pins all
 *  the versions a reader needs to decide whether it can interpret the
 *  rest of the file. Unknown sibling fields round-trip in `meta`. */
export interface TraceHeader {
  format_version: number;
  recorder_version: string;
  canonicalisation_version: string;
  price_list_version: string;
  /** Hex-encoded Ed25519 public key. */
  public_key: string;
  /** Identifier of the HMAC key the writer used. Lookup is the
   *  caller's responsibility — this package doesn't manage keys. */
  hmac_key_id: string;
  /** Free-form metadata bag. Reserved keys are documented in the
   *  SB-Trace spec; unknown keys MUST round-trip. */
  meta?: Record<string, unknown>;
  // Forward-compatible additional fields are tolerated.
  [key: string]: unknown;
}

/** One recorded step — what `stepback/recorder.py` emits per
 *  LLM/tool/router/policy/MCP call. */
export interface RecordedStep {
  step_id: string;
  step_kind: StepKind;
  parent_step_id?: string | null;
  inputs: unknown;
  outputs: unknown;
  /** `sha256:<hex>` of the canonical-JSON form of `inputs`. */
  inputs_hash: string;
  nondeterminism_hash?: string | null;
  wallclock_ns?: number | null;
  cpu_ns?: number | null;
  cost_usd?: number | null;
  [key: string]: unknown;
}

/** Discriminated union of every kind of frame body v1 may contain.
 *  The discriminator is `"type"` to match the Python reference. */
export type FrameBody =
  | (TraceHeader & { type: "header" })
  | { type: "step"; step?: RecordedStep; encoding?: string; data?: string }
  | { type: "blob"; id: string; encoding: string; data: string }
  | { type: "tail"; wallclock_ns?: number | null }
  | { type: string; [key: string]: unknown };

/** Wrapper as it appears on disk after the length prefix is consumed.
 *
 *  `bodyBytes` is the raw canonical-JSON byte slice of the body field
 *  as it appeared inside the wrapper. The verifier must HMAC-check
 *  these bytes directly rather than re-canonicalizing `body`, because
 *  JavaScript `Number` cannot losslessly round-trip the 64-bit
 *  integers (`wallclock_ns`, `cpu_ns`) that frame bodies routinely
 *  carry — re-canonicalizing those would emit a different lexical
 *  form and the HMAC would (correctly) mismatch. */
export interface Frame {
  body: unknown;
  bodyBytes: Uint8Array;
  prev_hmac: string;
  hmac: string;
  sig: string;
}

/** Errors that can occur while splitting a byte stream into frames.
 *  Carries a `kind` discriminator for machine inspection. */
export class FrameError extends Error {
  override readonly name = "FrameError";
  readonly kind:
    | "UnexpectedEof"
    | "FrameTooLarge"
    | "BadJson"
    | "BadWrapperShape";
  readonly role?: string;
  readonly length?: number;
  constructor(
    kind: FrameError["kind"],
    message: string,
    extra: { role?: string; length?: number } = {}
  ) {
    super(message);
    this.kind = kind;
    if (extra.role !== undefined) this.role = extra.role;
    if (extra.length !== undefined) this.length = extra.length;
  }
}

/**
 * Split a contiguous byte slice into framed wrappers.
 *
 * This does **no** crypto verification. Use `verify.ts` for that.
 * It exists here so that the verifier and any future readers share
 * one frame splitter.
 */
export function* iterFrames(buf: Uint8Array): Generator<Frame, void, void> {
  let offset = 0;
  // Use a single TextDecoder instance for performance; fail fast on
  // invalid UTF-8 inside frame wrappers.
  const decoder = new TextDecoder("utf-8", { fatal: true });
  while (offset < buf.length) {
    if (buf.length - offset < FRAME_LENGTH_PREFIX) {
      throw new FrameError(
        "UnexpectedEof",
        "unexpected end of input while reading frame length-prefix",
        { role: "length-prefix" }
      );
    }
    const len =
      (buf[offset]! << 24) |
      (buf[offset + 1]! << 16) |
      (buf[offset + 2]! << 8) |
      buf[offset + 3]!;
    // Bitwise ops yield signed 32-bit; mask back to unsigned.
    const length = len >>> 0;
    offset += FRAME_LENGTH_PREFIX;
    if (length > MAX_FRAME_BYTES) {
      throw new FrameError(
        "FrameTooLarge",
        `frame length ${length} exceeds MAX_FRAME_BYTES=${MAX_FRAME_BYTES}`,
        { length }
      );
    }
    if (buf.length - offset < length) {
      throw new FrameError(
        "UnexpectedEof",
        "unexpected end of input while reading frame body",
        { role: "body" }
      );
    }
    const slice = buf.subarray(offset, offset + length);
    offset += length;
    let text: string;
    try {
      text = decoder.decode(slice);
    } catch (e) {
      throw new FrameError("BadJson", `frame wrapper was not valid UTF-8: ${(e as Error).message}`);
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch (e) {
      throw new FrameError("BadJson", `frame wrapper was not valid JSON: ${(e as Error).message}`);
    }
    const bodyBytes = extractBodyBytes(slice);
    yield asFrame(parsed, bodyBytes);
  }
}

function asFrame(value: unknown, bodyBytes: Uint8Array): Frame {
  if (
    value === null ||
    typeof value !== "object" ||
    Array.isArray(value)
  ) {
    throw new FrameError("BadWrapperShape", "frame wrapper was not a JSON object");
  }
  const obj = value as Record<string, unknown>;
  if (
    !("body" in obj) ||
    typeof obj["prev_hmac"] !== "string" ||
    typeof obj["hmac"] !== "string" ||
    typeof obj["sig"] !== "string"
  ) {
    throw new FrameError(
      "BadWrapperShape",
      "frame wrapper missing one of body/prev_hmac/hmac/sig"
    );
  }
  return {
    body: obj["body"],
    bodyBytes,
    prev_hmac: obj["prev_hmac"] as string,
    hmac: obj["hmac"] as string,
    sig: obj["sig"] as string,
  };
}

// Constant-folded byte sequence for the wrapper prefix produced by the
// Python writer (`canonical_json({"body": ..., "hmac": ..., ...})`).
// Because keys are sorted, "body" is always the first key, so every
// well-formed wrapper begins with these 8 bytes: `{"body":`.
const WRAPPER_BODY_PREFIX = new Uint8Array([
  0x7b, 0x22, 0x62, 0x6f, 0x64, 0x79, 0x22, 0x3a,
]);

/**
 * Extract the raw bytes of the body JSON value from a canonical
 * wrapper. The wrapper byte form is exactly what the Python writer
 * fed into HMAC, so reusing this slice guarantees byte-for-byte
 * equivalence regardless of JavaScript Number precision.
 */
function extractBodyBytes(wrapper: Uint8Array): Uint8Array {
  if (wrapper.length < WRAPPER_BODY_PREFIX.length) {
    throw new FrameError("BadWrapperShape", "wrapper too short to contain body");
  }
  for (let i = 0; i < WRAPPER_BODY_PREFIX.length; i++) {
    if (wrapper[i] !== WRAPPER_BODY_PREFIX[i]) {
      throw new FrameError(
        "BadWrapperShape",
        "wrapper does not start with canonical {\"body\": prefix"
      );
    }
  }
  const start = WRAPPER_BODY_PREFIX.length;
  const end = scanJsonValueEnd(wrapper, start);
  return wrapper.subarray(start, end);
}

/**
 * Return the byte offset just past the end of the JSON value that
 * starts at `start` within `buf`. Handles objects, arrays, strings,
 * numbers, `true` / `false` / `null`. Assumes canonical JSON
 * (no whitespace), matching what the Python writer produces.
 */
function scanJsonValueEnd(buf: Uint8Array, start: number): number {
  if (start >= buf.length) {
    throw new FrameError("BadWrapperShape", "empty body value in wrapper");
  }
  const c0 = buf[start]!;
  if (c0 === 0x7b /* { */ || c0 === 0x5b /* [ */) {
    return scanContainerEnd(buf, start);
  }
  if (c0 === 0x22 /* " */) {
    return scanStringEnd(buf, start);
  }
  // number / true / false / null — scan until a structural char at
  // depth 0 (comma, close-brace, close-bracket).
  let i = start;
  while (i < buf.length) {
    const c = buf[i]!;
    if (c === 0x2c /* , */ || c === 0x7d /* } */ || c === 0x5d /* ] */) {
      return i;
    }
    i++;
  }
  return i;
}

function scanContainerEnd(buf: Uint8Array, start: number): number {
  let depth = 0;
  let inString = false;
  let escape = false;
  let i = start;
  while (i < buf.length) {
    const c = buf[i]!;
    if (escape) {
      escape = false;
      i++;
      continue;
    }
    if (inString) {
      if (c === 0x5c /* \ */) {
        escape = true;
      } else if (c === 0x22 /* " */) {
        inString = false;
      }
      i++;
      continue;
    }
    if (c === 0x22 /* " */) {
      inString = true;
      i++;
      continue;
    }
    if (c === 0x7b /* { */ || c === 0x5b /* [ */) {
      depth++;
    } else if (c === 0x7d /* } */ || c === 0x5d /* ] */) {
      depth--;
      if (depth === 0) {
        return i + 1;
      }
    }
    i++;
  }
  throw new FrameError("BadWrapperShape", "unterminated container in body");
}

function scanStringEnd(buf: Uint8Array, start: number): number {
  let i = start + 1;
  let escape = false;
  while (i < buf.length) {
    const c = buf[i]!;
    if (escape) {
      escape = false;
      i++;
      continue;
    }
    if (c === 0x5c /* \ */) {
      escape = true;
      i++;
      continue;
    }
    if (c === 0x22 /* " */) {
      return i + 1;
    }
    i++;
  }
  throw new FrameError("BadWrapperShape", "unterminated string in body");
}
