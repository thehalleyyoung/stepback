# `@stepback/core` — TypeScript reader/verifier for `.sb` v1 traces

This package is the TypeScript half of [`stepback`](../../README.md)'s
multi-language reader story. It reads, parses, and **cryptographically
verifies** SB-Trace `.sb` v1 files written by the Python recorder
(`stepback/trace_writer.py`) or the Rust core
(`stepback-core/crates/sb-format`). It uses Node's built-in `crypto`
module — **no native binary, no `node-gyp`, no compile step on
install** — and works under both ESM and CJS.

It is intentionally **read-only** in its first release. Writing `.sb`
traces from JavaScript is on the roadmap once the Rust writer in
`sb-format` ships.

## Status

Experimental. Tracks `format_version=1` and
`canonicalisation_version=1` of the SB-Trace specification. Verifies
the same fixture corpus the Rust verifier exercises
(`stepback-core/fixtures/v1/`), including the canonical "good" traces
and the canonical "corrupt" traces that must be rejected.

## Install

```bash
npm install @stepback/core
# or
pnpm add @stepback/core
yarn add @stepback/core
```

Requires Node ≥ 18.17 (for `crypto.verify` Ed25519 support and
`node:test`).

## Usage

```ts
import { readFile } from "node:fs/promises";
import { verifyBytes, VerifyError } from "@stepback/core";

const trace = await readFile("./incident.sb");
const hmacKey = Buffer.from(process.env.SB_HMAC_KEY_HEX!, "hex");

try {
  const v = verifyBytes(new Uint8Array(trace), hmacKey);
  console.log(`OK: ${v.frameCount} frames, recorder=${v.header.recorder_version}`);
} catch (e) {
  if (e instanceof VerifyError) {
    console.error(`reject ${e.kind} at frame ${e.frameIndex}: ${e.message}`);
  } else {
    throw e;
  }
}
```

CommonJS:

```js
const { verifyPath } = require("@stepback/core");
verifyPath("./incident.sb", Buffer.from(hex, "hex")).then(console.log);
```

## What it verifies

For each frame in the file, in order:

1. The 4-byte big-endian length prefix is well-formed and within the
   `MAX_FRAME_BYTES` bound.
2. The frame body parses as canonical JSON.
3. `HMAC_SHA256(hmac_key, prev_hmac || canonical_json(body))` matches
   `frame.hmac` in constant time.
4. `frame.prev_hmac` chains to the previous frame's `hmac` (or to 32
   zero bytes for frame 0).
5. `frame.sig` is `ed25519:<hex>`, decodes to 64 bytes, and verifies
   against the public key pinned in the header.
6. Frame 0 is the header and pins `format_version = 1`.

Tampering, reordering, truncation, and silent forgery of the format
version are all rejected with a typed `VerifyError` whose `kind`
discriminator matches the Rust `VerifyError` variants.

## Building from source

```bash
npm install
npm run build   # emits dist/esm, dist/cjs, dist/types
npm test        # runs node --test against built artifacts + fixtures
```

The build emits dual ESM + CJS bundles and ambient `.d.ts` types from
a single TypeScript source tree under `src/`. Node's `package.json`
`exports` map routes consumers to the right bundle.

## Layout

```
bindings/typescript/
├── src/
│   ├── canonical.ts   # canonical JSON + sha256 hashing
│   ├── frames.ts      # frame splitter + on-disk shapes
│   ├── verify.ts      # HMAC chain + Ed25519 signature verifier
│   └── index.ts       # public re-exports
├── test/
│   ├── fixtures.test.mjs   # runs against shared `.sb` fixture corpus
│   └── version.test.mjs    # version-skew + ESM/CJS surface parity
├── scripts/build.mjs       # dual ESM + CJS + types build (no rollup)
└── package.json
```

## Conformance

The fixture corpus under `../../stepback-core/fixtures/v1/` is the
shared compatibility surface. Any `.sb` reader claiming SB-Trace v1
support — Python, Rust, TypeScript, Go, JVM, .NET — must accept every
trace in `good/` and reject every trace in `corrupt/` with an error
whose kind matches the manifest's `expected_error_kind`.

## License

Apache-2.0 — same as the rest of `stepback`.
