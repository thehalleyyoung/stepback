// Asserts that the package.json `version` field matches the
// `VERSION` constant exported from `src/index.ts`. This catches
// release-time drift where one is bumped without the other.

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = resolve(__dirname, "..");
const distEsm = join(root, "dist", "esm", "index.js");

if (!existsSync(distEsm)) {
  const res = spawnSync("node", ["scripts/build.mjs"], { cwd: root, stdio: "inherit" });
  if (res.status !== 0) throw new Error(`build failed: ${res.status}`);
}

const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
const mod = await import("../dist/esm/index.js");

test("package.json version matches exported VERSION", () => {
  assert.equal(mod.VERSION, pkg.version);
});

test("ESM and CJS entry points export the same surface", async () => {
  const esm = await import("../dist/esm/index.js");
  // Use `createRequire` to load the CJS entry from this ESM test.
  const { createRequire } = await import("node:module");
  const require_ = createRequire(import.meta.url);
  const cjs = require_("../dist/cjs/index.cjs");
  // Compare the set of named exports. Default export is allowed to
  // differ (CJS adds `default` for interop, ESM does not).
  const esmKeys = new Set(Object.keys(esm).filter((k) => k !== "default"));
  const cjsKeys = new Set(Object.keys(cjs).filter((k) => k !== "default"));
  for (const k of esmKeys) {
    assert.ok(cjsKeys.has(k), `CJS missing export: ${k}`);
  }
  for (const k of cjsKeys) {
    assert.ok(esmKeys.has(k), `ESM missing export: ${k}`);
  }
});
