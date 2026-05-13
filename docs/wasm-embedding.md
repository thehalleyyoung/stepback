# Embedding the stepback WASM Viewer

This document explains how to embed the stepback in-browser `.sb` trace viewer
in your own web application or documentation site.

The WASM viewer lets end-users open, summarize, and verify stepback traces
entirely in their browser — **no server-side code, no trace data leaves the
user's machine**.

---

## Contents

1. [Quick start — CDN / pre-built bundle](#quick-start)
2. [Embedding in a plain HTML page](#plain-html)
3. [Embedding in a React / Vue / Svelte app](#bundler)
4. [Embedding in a Node.js script](#nodejs)
5. [JS API reference](#js-api)
6. [Customizing the UI](#customizing)
7. [Security model](#security)
8. [Building from source](#building-from-source)

---

## Quick start — CDN / pre-built bundle <a id="quick-start"></a>

The fastest path is to copy the self-contained demo page and point it at
your own `.sb` file (or let users drag-and-drop their own):

```bash
# After running scripts/build_wasm.sh the bundle lives under wasm/pkg/
ls wasm/pkg/   # stepback_wasm.js  stepback_wasm_bg.wasm  …
```

The `wasm/demo/index.html` is a single-file standalone viewer — copy it
next to the `pkg/` directory and open it in any modern browser.

---

## Embedding in a plain HTML page <a id="plain-html"></a>

### Step 1 — copy the WASM bundle

```bash
# Build the web (ESM) bundle once
./scripts/build_wasm.sh

# Copy the bundle to your web root
cp -r wasm/pkg/ /var/www/html/stepback-wasm/
```

### Step 2 — add the script tag

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>My Agent Dashboard</title>
</head>
<body>

  <!-- Drop zone for .sb files -->
  <input type="file" id="trace-file" accept=".sb" />
  <pre id="output"></pre>

  <script type="module">
    import init, { version, init_panic_hook, summarize, verify }
      from "/stepback-wasm/stepback_wasm.js";

    await init();
    init_panic_hook(); // optional — nicer DevTools errors

    console.log("stepback-wasm", version());

    document.getElementById("trace-file").addEventListener("change", async (ev) => {
      const file = ev.target.files[0];
      if (!file) return;

      const bytes = new Uint8Array(await file.arrayBuffer());

      // Summarize without any key — always safe, no secrets needed.
      const summary = summarize(bytes);
      document.getElementById("output").textContent = JSON.stringify(summary, null, 2);
    });
  </script>
</body>
</html>
```

### Step 3 — serve with COOP/COEP headers (required for SharedArrayBuffer)

The WASM binary requires `SharedArrayBuffer` on some platforms.  Add these
response headers to your web server:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

For Nginx:

```nginx
add_header Cross-Origin-Opener-Policy same-origin;
add_header Cross-Origin-Embedder-Policy require-corp;
```

For Caddy:

```caddy
header {
    Cross-Origin-Opener-Policy same-origin
    Cross-Origin-Embedder-Policy require-corp
}
```

---

## Embedding in a React / Vue / Svelte app <a id="bundler"></a>

Use the **bundler** build (`wasm/pkg-bundler/`) with Vite, Webpack, or
Rollup.

### Install (after building from source)

```bash
# Link the local package for development
cd wasm/pkg-bundler
npm link

cd /path/to/your-app
npm link @stepback/core
```

Or publish to npm and install normally:

```bash
npm install @stepback/core
```

### React example

```tsx
import { useEffect, useRef, useState } from "react";

let wasmInit: Promise<void> | null = null;

async function loadWasm() {
  if (!wasmInit) {
    const mod = await import("@stepback/core");
    wasmInit = mod.default();
    await wasmInit;
  }
}

export function TraceViewer({ file }: { file: File | null }) {
  const [summary, setSummary] = useState<object | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!file) return;

    (async () => {
      try {
        await loadWasm();
        const { summarize } = await import("@stepback/core");
        const bytes = new Uint8Array(await file.arrayBuffer());
        const result = summarize(bytes);
        setSummary(result);
        setError(null);
      } catch (err) {
        setError(String(err));
      }
    })();
  }, [file]);

  if (error) return <div className="error">{error}</div>;
  if (!summary) return <div>Drop a .sb file to inspect it.</div>;
  return <pre>{JSON.stringify(summary, null, 2)}</pre>;
}
```

### Vue 3 example

```vue
<script setup lang="ts">
import { ref, watch } from "vue";

const props = defineProps<{ file: File | null }>();
const summary = ref<object | null>(null);

let wasmReady = false;
let summarizeFn: ((bytes: Uint8Array) => object) | null = null;

async function ensureWasm() {
  if (!wasmReady) {
    const mod = await import("@stepback/core");
    await mod.default();
    summarizeFn = mod.summarize;
    wasmReady = true;
  }
}

watch(() => props.file, async (file) => {
  if (!file) return;
  await ensureWasm();
  const bytes = new Uint8Array(await file.arrayBuffer());
  summary.value = summarizeFn!(bytes);
});
</script>

<template>
  <pre v-if="summary">{{ JSON.stringify(summary, null, 2) }}</pre>
  <div v-else>Drop a .sb file to inspect it.</div>
</template>
```

---

## Embedding in a Node.js script <a id="nodejs"></a>

Use the **Node.js** build (`wasm/pkg-nodejs/`).

```js
const { summarize, verify, version } = require("./wasm/pkg-nodejs/stepback_wasm.js");
const fs = require("fs");

const bytes = new Uint8Array(fs.readFileSync("trace.sb"));

const summary = summarize(bytes);
console.log("frame_count:", summary.frame_count);
console.log("step_kinds:", summary.step_kinds);
console.log("total_cost_usd:", summary.total_cost_usd);

// Verify with HMAC key (hex string from trace header)
const result = verify(bytes, "aabbccdd...");
if (result.ok) {
  console.log("Trace is valid ✓");
} else {
  console.error("Verification failed:", result.error_kind, result.message);
}
```

---

## JS API reference <a id="js-api"></a>

All functions return a structured result object and **never throw** — check
`result.ok` before reading success fields.

### `version() → string`

Returns the crate version string (e.g. `"0.1.0"`).

### `summarize(bytes: Uint8Array) → SummarizeResult`

Reads the frame sequence and returns:

```ts
interface SummarizeResult {
  ok: boolean;
  // On ok=true:
  frame_count?: number;
  step_frames?: number;
  blob_frames?: number;
  tail_frames?: number;
  step_kinds?: Record<string, number>;  // e.g. { llm_call: 4, tool_call: 7 }
  total_cost_usd?: number;
  total_wallclock_ns?: number;
  header?: {
    format_version: number;
    recorder_version: string;
    trace_id: string;
    // … other header fields
  };
  // On ok=false:
  error_kind?: string;
  message?: string;
}
```

Does **not** verify HMAC or signatures — safe to call without any keys.

### `verify(bytes: Uint8Array, hmac_key_hex: string) → VerifyResult`

Walks the HMAC-SHA256 chain and per-frame Ed25519 signatures.

```ts
interface VerifyResult {
  ok: boolean;
  // On ok=true:
  frame_count?: number;
  // On ok=false:
  error_kind?: string;  // "HmacMismatch" | "SignatureMismatch" | "UnexpectedEof" | …
  message?: string;
  frame_index?: number; // Frame where verification failed, when applicable.
}
```

The `hmac_key_hex` is the **raw HMAC key** in lowercase hex — typically
stored securely and injected server-side or derived from a user-provided
passphrase.  If you only need to summarize structure (not cryptographic
integrity), call `summarize` instead.

---

## Customizing the UI <a id="customizing"></a>

The `wasm/demo/index.html` single-file demo is intentionally minimal.
Common customizations:

### Hide the HMAC key field

If your use case is read-only summary without key verification, remove the
"Verify" panel and only expose the "Summarize" button.

### Inject the HMAC key server-side

For managed dashboards where users should not supply the key themselves,
fetch the key from an authenticated endpoint and inject it before calling
`verify`:

```js
const keyHex = await (await fetch("/api/trace-key/" + traceId)).text();
const result = verify(bytes, keyHex);
```

### Dark mode

The demo uses `color-scheme: light dark` via CSS.  No JS changes needed.

### Iframe embedding

You can embed the demo page in an `<iframe>` with `allow="clipboard-write"`:

```html
<iframe
  src="/stepback-viewer/"
  width="100%"
  height="600"
  allow="clipboard-write"
  style="border: 1px solid #ccc; border-radius: 6px;"
></iframe>
```

---

## Security model <a id="security"></a>

| Property | Guarantee |
|---|---|
| **No server upload** | All parsing and crypto runs in WASM in the browser process. |
| **No eval / external fetch** | The WASM bundle is self-contained; it makes no network calls. |
| **Signature verification** | Ed25519 signature verification runs inside WASM (ring/dalek library). |
| **HMAC verification** | HMAC-SHA256 chain verification runs inside WASM. |
| **Key exposure** | The HMAC key is passed as a JS string to the WASM function; it is not persisted or logged by the viewer. Avoid storing it in `localStorage`. |
| **Content Security Policy** | Add `'wasm-unsafe-eval'` to your `script-src` CSP directive to allow WASM execution. |

---

## Building from source <a id="building-from-source"></a>

```bash
# Prerequisites
rustup target add wasm32-unknown-unknown
cargo install wasm-pack            # only needed once

# Build all three targets
./scripts/build_wasm.sh

# Outputs:
#   wasm/pkg/           — browser ESM (import via <script type="module">)
#   wasm/pkg-bundler/   — Webpack / Vite / Rollup
#   wasm/pkg-nodejs/    — Node.js (require / CommonJS)
```

The crate source is at `stepback-core/crates/sb-wasm/src/lib.rs`.

To run the included Node.js smoke test:

```bash
cd wasm
node smoke_test.js   # requires wasm/pkg-nodejs/ to exist
```
