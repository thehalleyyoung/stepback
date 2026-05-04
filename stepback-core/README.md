# stepback-core

Rust workspace for the [`stepback`](../README.md) reversible-debugger and
counterfactual-evaluation runtime.

This workspace is the foundation for **Step 5** of the standardisation plan:
a multi-crate Rust core that the eventual `stepback-core` crate, the language
bindings (Python, TypeScript, Go, JVM, .NET), the WASM verifier, and the
`sb proxy` sidecar all build on. The Python implementation in `../stepback/`
remains the reference recorder while this workspace grows.

## Crates

| Crate | Status | Purpose |
| --- | --- | --- |
| [`sb-format`](crates/sb-format) | scaffold | `.sb` v1 frame layout: 4-byte big-endian length prefix, canonical-JSON wrapper with `body` / `prev_hmac` / `hmac` / `sig`. Pure data types, no I/O policy. |
| [`sb-canonical`](crates/sb-canonical) | scaffold | Canonical UTF-8 JSON encoding (sorted keys, no whitespace, `allow_nan=false`) and `sha256:<hex>` content hashing. Bit-for-bit compatible with `stepback/canonical.py`. |
| [`sb-dirty`](crates/sb-dirty) | scaffold | Dirty-set propagation over a trace DAG: given a substitution at step *k*, computes the set of downstream steps whose canonical inputs would re-hash differently and therefore must be re-executed. |
| [`sb-replay`](crates/sb-replay) | scaffold | Replay planner / executor traits. Consumes a dirty-set, decides which steps come from cache and which must be re-run, and orders execution respecting the DAG. |
| [`sb-verify`](crates/sb-verify) | scaffold | Header validation, HMAC-SHA256 chain verification, and per-frame Ed25519 signature verification. The first end-to-end consumer of a Python-written `.sb`. |

The crates are deliberately split so that read-only consumers (verifiers, the
WASM build, importers) can depend on `sb-format` + `sb-canonical` + `sb-verify`
without pulling in the dirty-set / replay machinery.

## Status

This is the **workspace scaffold**. Each crate compiles and exposes a stable
public surface (types, traits, error enums, version constants), but the
heavy implementations land in subsequent steps:

- **Step 6** — ✅ `sb-verify` is wired to the frozen `.sb` fixture corpus
  under [`fixtures/v1/`](fixtures/v1/). The Python generator
  [`scripts/gen_fixtures.py`](scripts/gen_fixtures.py) emits bit-stable
  `.sb` files with pinned HMAC + Ed25519 keys; `cargo test -p sb-verify`
  walks every fixture (good + corrupt) and asserts the verifier accepts
  or rejects with the correct error class. The Python suite re-runs the
  generator and asserts no diff (`tests/test_sbtrace_fixtures.py`) so
  the corpus stays byte-locked between recorder and verifier.
- **Step 7** — PyO3 bindings under `bindings/python/stepback_core/`,
  routing `verify_trace` through Rust behind an experimental flag.
- **Steps 8–12** — TypeScript / Go / JVM / .NET / WASM bindings on top of
  the same crates.

### Conformance fixtures

```text
fixtures/v1/
├── manifest.json          # hmac key, public key, expected outcomes, sha256s
├── good/
│   ├── header_only.sb     # header + tail
│   ├── multi_step.sb      # header + 3 steps + tail (uncompressed)
│   └── with_blobs.sb      # header + blob + steps + tail (gzip+dedup)
└── corrupt/
    ├── truncated_body.sb       → Parse(UnexpectedEof)
    ├── flipped_hmac.sb         → HmacMismatch / BrokenChain / BadHex
    ├── flipped_sig.sb          → SignatureMismatch / BadHex
    ├── broken_chain.sb         → BrokenChain / HmacMismatch / BadHex
    └── bad_format_version.sb   → HmacMismatch (version forgery breaks the
                                    body HMAC) — once capability negotiation
                                    lands the explicit UnsupportedFormatVersion
                                    branch becomes reachable too.
```

Regenerate with:

```bash
python3 stepback-core/scripts/gen_fixtures.py
```

Files only change if their bytes change; an unchanged run produces an empty
`git diff`. Independent SB-Trace implementations should run their own
verifier against this corpus and report results back to the conformance
suite (Step 46).

## Build

```bash
cd stepback-core
cargo build --workspace
cargo test  --workspace
```

Minimum supported Rust version (MSRV) is pinned in the workspace `Cargo.toml`.

## Compatibility with the Python recorder

`sb-canonical` MUST produce byte-identical output to
`stepback/canonical.py::canonical_json` for the same input value. The
test suites under each crate use the same fixture corpus the Python tests
use. If you add a value-shape Python can write that this crate cannot
encode identically, that is a bug in this crate, not the recorder.

## License

Apache-2.0, matching the rest of the project.
