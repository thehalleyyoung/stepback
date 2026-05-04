// HMAC-chain and per-frame Ed25519 signature verifier for `.sb` v1
// traces. Mirrors `stepback-core/crates/sb-verify/src/lib.rs`.
//
// Verification policy in v1:
//
//   1. Walk frames in order.
//   2. For each frame, recompute
//        HMAC_SHA256(hmac_key, prev_hmac_bytes || canonical_json(body))
//      and compare it to `frame.hmac` in constant time.
//   3. Verify the Ed25519 signature `frame.sig` over the **raw**
//      32-byte HMAC digest, not the hex-encoded form (matching the
//      Python writer's `signing_key.sign(h)`).
//   4. The `prev_hmac` field of frame n+1 must equal the `hmac` of
//      frame n. The first frame's `prev_hmac` must be 32 zero bytes.

import {
  createHmac,
  createPublicKey,
  timingSafeEqual,
  verify as cryptoVerify,
  type KeyObject,
} from "node:crypto";
import { promises as fs } from "node:fs";

import {
  FORMAT_VERSION,
  FrameError,
  iterFrames,
  type Frame,
  type TraceHeader,
} from "./frames.js";

/** 32 zero bytes — the seed of the HMAC chain. */
export const ZERO_HMAC: Uint8Array = new Uint8Array(32);

/** All the ways verification can fail. The `kind` discriminator is
 *  stable and machine-readable; `frameIndex` is the 0-based index of
 *  the offending frame, or `-1` when the error is not frame-local. */
export type VerifyErrorKind =
  | "Parse"
  | "MissingHeader"
  | "UnsupportedFormatVersion"
  | "BrokenChain"
  | "HmacMismatch"
  | "BadHex"
  | "UnsupportedSignature"
  | "BadSignatureLength"
  | "SignatureMismatch"
  | "BadPublicKey";

export class VerifyError extends Error {
  override readonly name = "VerifyError";
  readonly kind: VerifyErrorKind;
  readonly frameIndex: number;
  readonly field?: string;
  constructor(
    kind: VerifyErrorKind,
    message: string,
    frameIndex: number,
    extra: { field?: string } = {}
  ) {
    super(message);
    this.kind = kind;
    this.frameIndex = frameIndex;
    if (extra.field !== undefined) this.field = extra.field;
  }
}

/** Outcome of a successful verification: the parsed header plus the
 *  number of frames that chained cleanly. */
export interface VerifiedTrace {
  header: TraceHeader;
  frameCount: number;
}

/** Ed25519 signature length in bytes. */
const SIGNATURE_LENGTH = 64;

/**
 * Verify a `.sb` byte slice end-to-end. The HMAC key is supplied by
 * the caller because key management lives outside this package.
 *
 * On success the caller may trust that:
 *   * every frame's HMAC is valid given the chain;
 *   * every frame's Ed25519 signature was issued by the holder of
 *     the private key whose public counterpart is pinned in the
 *     header;
 *   * no frame was inserted, dropped, reordered, or rewritten.
 */
export function verifyBytes(buf: Uint8Array, hmacKey: Uint8Array): VerifiedTrace {
  let header: TraceHeader | undefined;
  let publicKey: KeyObject | undefined;
  let prevHmac: Uint8Array = ZERO_HMAC;
  let count = 0;

  let frameIter: Generator<Frame, void, void>;
  try {
    frameIter = iterFrames(buf);
  } catch (e) {
    throw wrapFrameError(0, e);
  }

  while (true) {
    let next: IteratorResult<Frame, void>;
    try {
      next = frameIter.next();
    } catch (e) {
      throw wrapFrameError(count, e);
    }
    if (next.done) break;
    const frame = next.value;
    const index = count;

    verifyChainLink(index, frame, prevHmac, hmacKey);

    if (index === 0) {
      const hdr = expectHeader(frame.body);
      if (hdr.format_version !== FORMAT_VERSION) {
        throw new VerifyError(
          "UnsupportedFormatVersion",
          `unsupported format_version ${hdr.format_version}, expected ${FORMAT_VERSION}`,
          0
        );
      }
      try {
        publicKey = parseEd25519PublicKey(hdr.public_key);
      } catch {
        throw new VerifyError(
          "BadPublicKey",
          "trace header used an unsupported public key encoding",
          -1
        );
      }
      header = hdr;
    }

    if (!publicKey) {
      // Defensive: parseEd25519PublicKey only returns on success, so
      // this is unreachable in practice.
      throw new VerifyError("BadPublicKey", "no public key resolved from header", -1);
    }

    verifySignature(index, frame, publicKey);

    prevHmac = decodeHex(index, "hmac", frame.hmac, 32);
    count += 1;
  }

  if (!header) {
    throw new VerifyError("MissingHeader", "trace has no header frame (empty input)", -1);
  }
  return { header, frameCount: count };
}

/**
 * Convenience: read `path` from disk and verify.
 */
export async function verifyPath(
  path: string,
  hmacKey: Uint8Array
): Promise<VerifiedTrace> {
  const buf = await fs.readFile(path);
  return verifyBytes(new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength), hmacKey);
}

function verifyChainLink(
  index: number,
  frame: Frame,
  expectedPrev: Uint8Array,
  hmacKey: Uint8Array
): void {
  const claimedPrev = decodeHex(index, "prev_hmac", frame.prev_hmac, 32);
  if (!constantTimeEqual(claimedPrev, expectedPrev)) {
    throw new VerifyError("BrokenChain", `frame ${index}: prev_hmac does not chain to previous frame`, index);
  }
  const mac = createHmac("sha256", hmacKey);
  mac.update(expectedPrev);
  // Use the verbatim byte slice from the wrapper rather than
  // re-canonicalizing `frame.body`. JavaScript `Number` loses
  // precision on 64-bit integer fields like `wallclock_ns`, so a
  // re-canonicalized body would (correctly) HMAC-mismatch.
  mac.update(frame.bodyBytes);
  const computed = mac.digest();
  const claimedHmac = decodeHexVec(index, "hmac", frame.hmac);
  if (!constantTimeEqual(claimedHmac, new Uint8Array(computed))) {
    throw new VerifyError("HmacMismatch", `frame ${index}: HMAC mismatch`, index);
  }
}

function verifySignature(index: number, frame: Frame, publicKey: KeyObject): void {
  const colon = frame.sig.indexOf(":");
  const scheme = colon >= 0 ? frame.sig.slice(0, colon) : "";
  const hexPart = colon >= 0 ? frame.sig.slice(colon + 1) : frame.sig;
  if (scheme !== "ed25519") {
    throw new VerifyError(
      "UnsupportedSignature",
      `frame ${index}: signature scheme '${scheme}' is not supported`,
      index
    );
  }
  let sigBytes: Uint8Array;
  try {
    sigBytes = hexDecode(hexPart);
  } catch {
    throw new VerifyError("BadHex", `frame ${index}: invalid hex in sig`, index, { field: "sig" });
  }
  if (sigBytes.length !== SIGNATURE_LENGTH) {
    throw new VerifyError(
      "BadSignatureLength",
      `frame ${index}: signature length ${sigBytes.length} is not ${SIGNATURE_LENGTH}`,
      index
    );
  }
  // The Python writer signs the **raw 32-byte HMAC digest**, not its
  // hex form. Decode `frame.hmac` back to bytes before verifying.
  let hmacBytes: Uint8Array;
  try {
    hmacBytes = hexDecode(frame.hmac);
  } catch {
    throw new VerifyError("BadHex", `frame ${index}: invalid hex in hmac`, index, { field: "hmac" });
  }
  const ok = cryptoVerify(null, hmacBytes, publicKey, sigBytes);
  if (!ok) {
    throw new VerifyError(
      "SignatureMismatch",
      `frame ${index}: Ed25519 signature did not verify`,
      index
    );
  }
}

function expectHeader(body: unknown): TraceHeader {
  if (body === null || typeof body !== "object" || Array.isArray(body)) {
    throw new VerifyError("MissingHeader", "frame 0 must be a header object", -1);
  }
  const obj = body as Record<string, unknown>;
  if (obj["type"] !== "header") {
    throw new VerifyError(
      "MissingHeader",
      `frame 0 must be a header but was '${String(obj["type"])}'`,
      -1
    );
  }
  if (typeof obj["format_version"] !== "number") {
    throw new VerifyError("MissingHeader", "header missing numeric format_version", -1);
  }
  // Permissive cast: extra fields ride along on the index signature.
  return obj as unknown as TraceHeader;
}

function parseEd25519PublicKey(hex: string): KeyObject {
  const raw = hexDecode(hex);
  if (raw.length !== 32) {
    throw new Error("Ed25519 public key must be 32 bytes");
  }
  // Build an SPKI DER for an Ed25519 public key:
  //   SEQUENCE {
  //     SEQUENCE { OID 1.3.101.112 }
  //     BIT STRING { 0x00 || raw32 }
  //   }
  const oidPrefix = new Uint8Array([
    0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00,
  ]);
  const der = new Uint8Array(oidPrefix.length + raw.length);
  der.set(oidPrefix, 0);
  der.set(raw, oidPrefix.length);
  return createPublicKey({ key: Buffer.from(der), format: "der", type: "spki" });
}

function decodeHex(
  index: number,
  field: string,
  s: string,
  expectedLen: number
): Uint8Array {
  const bytes = decodeHexVec(index, field, s);
  if (bytes.length !== expectedLen) {
    throw new VerifyError(
      "BadHex",
      `frame ${index}: ${field} expected ${expectedLen} bytes, got ${bytes.length}`,
      index,
      { field }
    );
  }
  return bytes;
}

function decodeHexVec(index: number, field: string, s: string): Uint8Array {
  try {
    return hexDecode(s);
  } catch {
    throw new VerifyError("BadHex", `frame ${index}: invalid hex in ${field}`, index, { field });
  }
}

function hexDecode(s: string): Uint8Array {
  if (s.length % 2 !== 0) {
    throw new Error("hex string has odd length");
  }
  const out = new Uint8Array(s.length / 2);
  for (let i = 0; i < out.length; i++) {
    const hi = hexNibble(s.charCodeAt(i * 2));
    const lo = hexNibble(s.charCodeAt(i * 2 + 1));
    out[i] = (hi << 4) | lo;
  }
  return out;
}

function hexNibble(code: number): number {
  if (code >= 0x30 && code <= 0x39) return code - 0x30;
  if (code >= 0x61 && code <= 0x66) return code - 0x61 + 10;
  if (code >= 0x41 && code <= 0x46) return code - 0x41 + 10;
  throw new Error(`invalid hex character code ${code}`);
}

function constantTimeEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  return timingSafeEqual(a, b);
}

function wrapFrameError(index: number, e: unknown): VerifyError {
  if (e instanceof FrameError) {
    return new VerifyError("Parse", `frame ${index}: ${e.message}`, index);
  }
  if (e instanceof Error) {
    return new VerifyError("Parse", `frame ${index}: ${e.message}`, index);
  }
  return new VerifyError("Parse", `frame ${index}: ${String(e)}`, index);
}
