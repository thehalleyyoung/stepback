// Fixture tests for `@stepback/core`. We read the same `.sb` corpus
// the Rust verifier uses (`stepback-core/fixtures/v1/`), so any
// divergence between the JS and Rust verifiers shows up here. Tests
// run against the *source* TypeScript via Node 22+'s built-in
// `--experimental-strip-types` (or equivalent) is unnecessary because
// we test the *built* JavaScript: the build script must run first.
//
// To keep the test runner dependency-free we invoke tsc from the
// build script if dist/ is missing.

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = resolve(__dirname, "..");
const fixturesRoot = resolve(root, "..", "..", "stepback-core", "fixtures", "v1");
const distEsm = join(root, "dist", "esm", "index.js");

function ensureBuilt() {
  if (existsSync(distEsm)) return;
  const res = spawnSync("node", ["scripts/build.mjs"], { cwd: root, stdio: "inherit" });
  if (res.status !== 0) {
    throw new Error(`build failed with status ${res.status}`);
  }
}

ensureBuilt();

const mod = await import("../dist/esm/index.js");
const { verifyBytes, VerifyError, FORMAT_VERSION, CANONICALISATION_VERSION, canonicalJson, hashValue, iterFrames, ZERO_HMAC } = mod;

const manifest = JSON.parse(readFileSync(join(fixturesRoot, "manifest.json"), "utf8"));

function hexToBytes(hex) {
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) {
    out[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return out;
}

const hmacKey = hexToBytes(manifest.hmac_key_hex);
const expectedPublicKeyHex = manifest.public_key_hex;

test("manifest pins format and canonicalisation versions we support", () => {
  assert.equal(manifest.format_version, FORMAT_VERSION);
  assert.equal(manifest.canonicalisation_version, CANONICALISATION_VERSION);
});

test("ZERO_HMAC is 32 zero bytes", () => {
  assert.equal(ZERO_HMAC.length, 32);
  assert.ok(ZERO_HMAC.every((b) => b === 0));
});

for (const fixture of manifest.good) {
  test(`good fixture ${fixture.name} verifies`, () => {
    const buf = readFileSync(join(fixturesRoot, "good", fixture.name));
    const v = verifyBytes(new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength), hmacKey);
    assert.equal(v.header.format_version, FORMAT_VERSION);
    assert.equal(v.header.canonicalisation_version, CANONICALISATION_VERSION);
    assert.equal(v.header.public_key, expectedPublicKeyHex);
    assert.equal(v.header.price_list_version, manifest.price_list_version);
    assert.ok(
      v.frameCount >= fixture.expected_frame_count_min,
      `frame_count ${v.frameCount} >= ${fixture.expected_frame_count_min}`
    );
  });
}

test("good multi_step fixture: iterFrames reports the same frame count", () => {
  const fixture = manifest.good.find((g) => g.name === "multi_step.sb");
  const buf = readFileSync(join(fixturesRoot, "good", fixture.name));
  const u8 = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  let n = 0;
  for (const _ of iterFrames(u8)) n++;
  const v = verifyBytes(u8, hmacKey);
  assert.equal(n, v.frameCount);
});

// Acceptance sets for corrupt fixtures. Mirrors the Rust verifier's
// fixture tests (`stepback-core/crates/sb-verify/tests/fixtures.rs`):
// for any given mutation, multiple error classes can be the legitimate
// "first failure", and any one of them proves the verifier rejected.
const CORRUPT_ACCEPTED = {
  "truncated_body.sb": ["Parse"],
  "flipped_hmac.sb": ["HmacMismatch", "BrokenChain", "BadHex", "SignatureMismatch"],
  "flipped_sig.sb": ["SignatureMismatch", "BadHex"],
  "broken_chain.sb": ["BrokenChain", "HmacMismatch", "BadHex"],
  "bad_format_version.sb": ["HmacMismatch", "UnsupportedFormatVersion"],
};

for (const fixture of manifest.corrupt) {
  test(`corrupt fixture ${fixture.name} is rejected`, () => {
    const buf = readFileSync(join(fixturesRoot, "corrupt", fixture.name));
    const u8 = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    let err;
    try {
      verifyBytes(u8, hmacKey);
    } catch (e) {
      err = e;
    }
    assert.ok(err, `expected ${fixture.name} to reject`);
    assert.ok(err instanceof VerifyError, `expected VerifyError, got ${err?.constructor?.name}`);
    const acceptable = CORRUPT_ACCEPTED[fixture.name] ?? fixture.expected_error_kind.split("Or");
    assert.ok(
      acceptable.includes(err.kind),
      `kind ${err.kind} not in accepted set [${acceptable.join(", ")}] for ${fixture.name}`
    );
  });
}

test("verifyBytes rejects empty input with MissingHeader", () => {
  let err;
  try {
    verifyBytes(new Uint8Array(0), hmacKey);
  } catch (e) {
    err = e;
  }
  assert.ok(err instanceof VerifyError);
  assert.equal(err.kind, "MissingHeader");
});

test("verifyBytes rejects wrong HMAC key", () => {
  const fixture = manifest.good.find((g) => g.name === "multi_step.sb");
  const buf = readFileSync(join(fixturesRoot, "good", fixture.name));
  const u8 = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  const wrongKey = new Uint8Array(hmacKey.length);
  let err;
  try {
    verifyBytes(u8, wrongKey);
  } catch (e) {
    err = e;
  }
  assert.ok(err instanceof VerifyError);
  assert.equal(err.kind, "HmacMismatch");
  assert.equal(err.frameIndex, 0);
});

test("canonicalJson sorts object keys lexicographically", () => {
  const a = canonicalJson({ b: 1, a: 2, c: 3 });
  assert.equal(new TextDecoder().decode(a), '{"a":2,"b":1,"c":3}');
});

test("canonicalJson recursively sorts nested objects", () => {
  const a = canonicalJson({ z: { y: 1, x: 2 }, a: [{ d: 4, c: 3 }] });
  assert.equal(new TextDecoder().decode(a), '{"a":[{"c":3,"d":4}],"z":{"x":2,"y":1}}');
});

test("canonicalJson emits non-ASCII as raw UTF-8 not \\uXXXX", () => {
  const a = canonicalJson({ name: "café" });
  // 'é' is 0xC3 0xA9 in UTF-8.
  assert.ok(Array.from(a).join(",").includes("195,169"));
  const text = new TextDecoder().decode(a);
  assert.ok(!text.includes("\\u"));
});

test("canonicalJson escapes control characters as lowercase \\u00xx", () => {
  const a = canonicalJson("a\u0001b");
  assert.equal(new TextDecoder().decode(a), '"a\\u0001b"');
});

test("canonicalJson rejects NaN and Infinity", () => {
  assert.throws(() => canonicalJson(NaN));
  assert.throws(() => canonicalJson(Infinity));
  assert.throws(() => canonicalJson(-Infinity));
});

test("hashValue is deterministic under key reordering", () => {
  const a = hashValue({ a: 1, b: 2 });
  const b = hashValue({ b: 2, a: 1 });
  assert.equal(a, b);
  assert.ok(a.startsWith("sha256:"));
  assert.equal(a.length, 7 + 64);
});

test("iterFrames rejects oversized length prefix", () => {
  const buf = new Uint8Array(4);
  // u32::MAX big-endian, way over MAX_FRAME_BYTES.
  buf[0] = 0xff; buf[1] = 0xff; buf[2] = 0xff; buf[3] = 0xff;
  let err;
  try {
    for (const _ of iterFrames(buf)) {/* unreachable */}
  } catch (e) {
    err = e;
  }
  assert.ok(err);
  assert.equal(err.kind, "FrameTooLarge");
});

test("iterFrames rejects truncated body", () => {
  const body = new TextEncoder().encode('{"body":{},"prev_hmac":"","hmac":"","sig":""}');
  // Lie about the length: claim 10 more bytes than we have.
  const buf = new Uint8Array(4 + body.length);
  const claim = body.length + 10;
  buf[0] = (claim >>> 24) & 0xff;
  buf[1] = (claim >>> 16) & 0xff;
  buf[2] = (claim >>> 8) & 0xff;
  buf[3] = claim & 0xff;
  buf.set(body, 4);
  let err;
  try {
    for (const _ of iterFrames(buf)) {/* unreachable */}
  } catch (e) {
    err = e;
  }
  assert.ok(err);
  assert.equal(err.kind, "UnexpectedEof");
});
