# `wasm/` — In-browser SB-Trace verifier and summarizer

This directory holds the WebAssembly build of stepback's `.sb` reader.
The crate that produces it lives at
`stepback-core/crates/sb-wasm/`; this directory holds the build
artifacts, the static demo page, and a node smoke test.

The headline use case (Step 12 of the standardisation roadmap) is:

> Open a `.sb` trace in a browser, see what's in it, optionally verify
> the HMAC chain and Ed25519 signatures — without sending any trace
> bytes to a server.

## Layout

```text
wasm/
├── README.md            this file
├── demo/
│   └── index.html       single-file static demo (uses ../pkg/)
├── pkg/                 wasm-pack `--target web` bundle (gitignored)
├── pkg-bundler/         wasm-pack `--target bundler` bundle (gitignored)
└── pkg-nodejs/          wasm-pack `--target nodejs` bundle (gitignored)
```

Build outputs are gitignored — they are reproducible from the Rust
source under `stepback-core/crates/sb-wasm`.

## Building

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack            # only needed once
./scripts/build_wasm.sh            # produces wasm/pkg/, pkg-bundler/, pkg-nodejs/
```

The build script emits three bundles so the same Rust crate covers
browser ESM, Webpack/Vite/etc. bundlers, and Node.js. All three are
read-only — they verify and summarize but do not record.

## JS API

Every entry point returns a structured result object — never throws —
so callers can render either `ok: true` with the parsed header, or
`ok: false` with a typed `error_kind`. See
`stepback-core/crates/sb-wasm/src/lib.rs` for the exact shape.

```js
import init, { version, init_panic_hook, summarize, verify } from "stepback-wasm";

await init();
init_panic_hook(); // optional, prettier panics in DevTools

const bytes = new Uint8Array(await file.arrayBuffer());

// 1. Summarize without any key
const s = summarize(bytes);
// { ok: true, frame_count, step_frames, blob_frames, tail_frames,
//   step_kinds: { llm_call: 4, tool_call: 7, ... },
//   total_cost_usd, total_wallclock_ns, header: { ... } }

// 2. Verify HMAC chain + per-frame Ed25519 signatures
const key = new Uint8Array(/* raw 32 bytes; not hex */);
const v = verify(bytes, key);
// { ok: true,  frame_count, format_version, recorder_version, ... }
// or
// { ok: false, error_kind: "HmacMismatch", message: "...", frame_index: 3 }
```

## Trying the demo locally

```bash
./scripts/build_wasm.sh
python3 -m http.server 8000
# then open http://localhost:8000/wasm/demo/
```

A static file server is fine; there is no server-side code. The demo
loads the wasm module via dynamic ESM import and hands a `Uint8Array`
of the chosen file directly to the WASM exports.

## Trust model

* **Trace bytes never leave the browser.** Verification and
  summarization run entirely in WebAssembly; the page issues no
  network calls beyond the initial load of the static `.js`/`.wasm`
  bundle.
* **HMAC keys come from the user, not the page.** The demo treats the
  HMAC key as input — it does not fetch keys, it does not persist
  keys, and it does not log keys.
* **Signature key is read from the trace.** The Ed25519 public key is
  pinned in the trace header; verification fails closed if it is
  malformed.
* **Bounded memory.** A defensive cap (8 × `MAX_FRAME_BYTES`) rejects
  pathologically large blobs before iteration to keep a malicious file
  from blowing out the tab's heap.

## Where this fits

* `stepback-core/crates/sb-format` — frame layout
* `stepback-core/crates/sb-canonical` — canonical JSON
* `stepback-core/crates/sb-verify` — HMAC + Ed25519 verifier
* **`stepback-core/crates/sb-wasm` — WASM bindings (this directory's source)**
* `bindings/python|typescript|go|jvm|dotnet` — native bindings for
  servers and CLIs
