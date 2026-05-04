#!/usr/bin/env bash
# Build the in-browser SB-Trace verifier/summarizer.
#
# Output layout (after a successful build):
#
#     wasm/pkg/                 wasm-bindgen --target=web bundle
#     wasm/pkg-bundler/         wasm-bindgen --target=bundler bundle
#     wasm/pkg-nodejs/          wasm-bindgen --target=nodejs bundle
#
# Each bundle is self-contained: the .wasm + the generated JS shim +
# TypeScript declarations + a package.json.
#
# Requirements:
#   * stable Rust toolchain with the wasm32-unknown-unknown target
#       rustup target add wasm32-unknown-unknown
#   * wasm-pack >= 0.12 (for the `web` target bundling and the
#     wasm-opt pass)
#       cargo install wasm-pack
#
# This script is invoked by the WASM CI job and by the
# `scripts/smoke_install.sh` flow when `STEPBACK_BUILD_WASM=1`.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CRATE_DIR="${ROOT}/stepback-core/crates/sb-wasm"
OUT_ROOT="${ROOT}/wasm"

if ! command -v wasm-pack >/dev/null 2>&1; then
    echo "wasm-pack not found on PATH. Install with: cargo install wasm-pack" >&2
    exit 1
fi

if ! rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown; then
    echo "wasm32-unknown-unknown target missing. Run: rustup target add wasm32-unknown-unknown" >&2
    exit 1
fi

echo "==> Building sb-wasm for target=web"
wasm-pack build "${CRATE_DIR}" \
    --release \
    --target web \
    --out-dir "${OUT_ROOT}/pkg" \
    --out-name stepback_wasm \
    -- --features panic_hook

echo "==> Building sb-wasm for target=bundler"
wasm-pack build "${CRATE_DIR}" \
    --release \
    --target bundler \
    --out-dir "${OUT_ROOT}/pkg-bundler" \
    --out-name stepback_wasm \
    -- --features panic_hook

echo "==> Building sb-wasm for target=nodejs"
wasm-pack build "${CRATE_DIR}" \
    --release \
    --target nodejs \
    --out-dir "${OUT_ROOT}/pkg-nodejs" \
    --out-name stepback_wasm \
    -- --features panic_hook

echo "==> wasm-pack build artifacts:"
find "${OUT_ROOT}" -maxdepth 2 -name '*.wasm' -exec ls -lh {} +
