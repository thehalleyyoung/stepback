// canonicalize.mjs — small CLI mirroring `stepback-core` Rust binary
// of the same name. Used by the cross-language differential harness in
// `spec/canonical/differential.py` (Step 48 of `100_STEPS.md`).
//
// Modes:
//   * single (default): one JSON document on stdin, canonical bytes on stdout.
//   * --batch:          one JSON document per line on stdin; emits
//                        `<len>\n<bytes>\n` chunks on stdout (decimal length).

import { canonicalJson } from "../dist/esm/canonical.js";

const argv = process.argv.slice(2);
const batch = argv.includes("--batch");

let buf = Buffer.alloc(0);
process.stdin.on("data", (chunk) => {
  buf = Buffer.concat([buf, chunk]);
});

process.stdin.on("end", () => {
  try {
    if (batch) {
      const text = buf.toString("utf8");
      // Use indexOf-based split so we don't break on \r\n on Windows.
      const lines = text.split(/\r?\n/);
      for (const line of lines) {
        if (!line) continue;
        const value = JSON.parse(line);
        const out = canonicalJson(value);
        process.stdout.write(`${out.length}\n`);
        process.stdout.write(Buffer.from(out));
        process.stdout.write("\n");
      }
    } else {
      const text = buf.toString("utf8").replace(/[\r\n]+$/, "");
      const value = JSON.parse(text);
      const out = canonicalJson(value);
      process.stdout.write(Buffer.from(out));
    }
  } catch (err) {
    process.stderr.write(`canonicalize: ${err && err.stack ? err.stack : String(err)}\n`);
    process.exit(1);
  }
});
