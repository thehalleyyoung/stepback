// Step 50 — reader resource limits / DoS bounds.
//
// Mirrors the Python tests in tests/test_reader_fuzz.py (section 7)
// and the Rust unit tests in stepback-core/crates/sb-format/src/lib.rs.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  iterFrames,
  FrameError,
  MAX_FRAME_BYTES,
  MAX_NESTING_DEPTH,
  MAX_STRING_BYTES,
} from "../dist/esm/index.js";

function wrapBody(body) {
  const wrapper = {
    body,
    prev_hmac: "00".repeat(32),
    hmac: "00".repeat(32),
    sig: "ed25519:" + "00".repeat(64),
  };
  // canonical JSON: sorted keys, no whitespace.
  function sortedStringify(v) {
    if (v === null || typeof v !== "object") return JSON.stringify(v);
    if (Array.isArray(v))
      return "[" + v.map(sortedStringify).join(",") + "]";
    const keys = Object.keys(v).sort();
    return (
      "{" +
      keys
        .map((k) => JSON.stringify(k) + ":" + sortedStringify(v[k]))
        .join(",") +
      "}"
    );
  }
  const payload = new TextEncoder().encode(sortedStringify(wrapper));
  const out = new Uint8Array(4 + payload.length);
  new DataView(out.buffer).setUint32(0, payload.length, false);
  out.set(payload, 4);
  return out;
}

test("iterFrames rejects length prefix above MAX_FRAME_BYTES", () => {
  const buf = new Uint8Array(4);
  // MAX_FRAME_BYTES + 1 = 0x04000001
  new DataView(buf.buffer).setUint32(0, MAX_FRAME_BYTES + 1, false);
  assert.throws(
    () => Array.from(iterFrames(buf)),
    (err) =>
      err instanceof FrameError &&
      err.kind === "FrameTooLarge" &&
      err.message.includes("MAX_FRAME_BYTES")
  );
});

test("iterFrames rejects deeply nested object", () => {
  let deep = {};
  let cursor = deep;
  for (let i = 0; i < MAX_NESTING_DEPTH + 5; i++) {
    cursor.x = {};
    cursor = cursor.x;
  }
  const buf = wrapBody({ type: "tail", deep });
  assert.throws(
    () => Array.from(iterFrames(buf)),
    (err) => err instanceof FrameError && err.kind === "DepthExceeded"
  );
});

test("iterFrames rejects deeply nested array", () => {
  let deep = [];
  let cursor = deep;
  for (let i = 0; i < MAX_NESTING_DEPTH + 5; i++) {
    const nxt = [];
    cursor.push(nxt);
    cursor = nxt;
  }
  const buf = wrapBody({ type: "tail", deep });
  assert.throws(
    () => Array.from(iterFrames(buf)),
    (err) => err instanceof FrameError && err.kind === "DepthExceeded"
  );
});

test("iterFrames rejects huge inline string", () => {
  const huge = "x".repeat(MAX_STRING_BYTES + 1);
  const buf = wrapBody({ type: "tail", huge });
  // MAX_STRING_BYTES (16 MiB) is well below MAX_FRAME_BYTES (64 MiB)
  // so the wrapper itself fits — the string-length check must fire.
  assert.throws(
    () => Array.from(iterFrames(buf)),
    (err) => err instanceof FrameError && err.kind === "StringTooLarge"
  );
});

test("iterFrames limits are tunable upwards", () => {
  let deep = {};
  let cursor = deep;
  for (let i = 0; i < MAX_NESTING_DEPTH + 2; i++) {
    cursor.x = {};
    cursor = cursor.x;
  }
  const buf = wrapBody({ type: "tail", deep });
  // Default rejects.
  assert.throws(() => Array.from(iterFrames(buf)));
  // Relaxed accepts.
  const out = Array.from(
    iterFrames(buf, { maxDepth: MAX_NESTING_DEPTH + 1024 })
  );
  assert.equal(out.length, 1);
});
