// Public entry point for `@stepback/core`.
//
// This module is the equivalent of the Rust `sb-verify` and Python
// `stepback_core` packages: a read-only verifier for `.sb` v1 traces
// that runs against Python-written fixtures with no native deps.
//
// The package version pinned here MUST stay in sync with
// `package.json`'s `version` field; the test suite asserts this.

import { canonicalJson, hashValue, sha256Hex, CANONICALISATION_VERSION } from "./canonical.js";
import {
  FORMAT_VERSION,
  FRAME_LENGTH_PREFIX,
  MAX_FRAME_BYTES,
  iterFrames,
  FrameError,
} from "./frames.js";
import {
  ZERO_HMAC,
  verifyBytes,
  verifyPath,
  VerifyError,
} from "./verify.js";

export type {
  Frame,
  FrameBody,
  RecordedStep,
  StepKind,
  TraceHeader,
} from "./frames.js";
export type { VerifiedTrace, VerifyErrorKind } from "./verify.js";

export {
  // canonical.ts
  CANONICALISATION_VERSION,
  canonicalJson,
  hashValue,
  sha256Hex,
  // frames.ts
  FORMAT_VERSION,
  FRAME_LENGTH_PREFIX,
  MAX_FRAME_BYTES,
  iterFrames,
  FrameError,
  // verify.ts
  ZERO_HMAC,
  verifyBytes,
  verifyPath,
  VerifyError,
};

/** Package version. Kept in sync with `package.json` by
 *  `test/version.test.mjs`. */
export const VERSION = "0.1.0" as const;
