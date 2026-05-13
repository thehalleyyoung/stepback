# Installing stepback

> **Note (2026-05):** stepback is **not currently published to PyPI,
> crates.io, npm, or any other package registry.** Every install path
> below that names a registry (`pip install stepback`, `pipx install
> stepback`, `cargo install`, `npm install`, `pip install
> stepback-core`, etc.) describes the *intended* publish layout and is
> not yet wired up. Until publishing is configured, install from the
> GitHub source:
>
> ```bash
> # Python recorder + replayer + CLI:
> pip install git+https://github.com/thehalleyyoung/stepback.git
>
> # Or for development from a clone:
> git clone https://github.com/thehalleyyoung/stepback
> cd stepback && pip install -e '.[dev]'
> ```
>
> Rust crates and other bindings build from a checkout of the same
> repository. Anything past that point in this file is roadmap.

---

This document is the canonical install reference across every language
binding, container image, and tool stepback ships. Each section labels its
maturity using one of the following badges:

| Badge | Meaning |
| --- | --- |
| **Stable** | API is covered by SemVer. Breaking changes only on a major bump. Suitable for production use. |
| **Experimental** | Read-only / preview. APIs may change between minor releases until promoted to **Stable**. Pin exact versions. |
| **Preview** | Skeleton / scaffold. Builds and runs against the shared fixture corpus, but feature surface is intentionally narrow. Do not depend on it in production. |

The Python recorder is the reference writer; everything else is currently a
**reader/verifier**. The first **writer** outside Python will land via the
Rust core (`stepback-core/crates/sb-format`) and propagate to the bindings
from there.

> **What every binding agrees on.** SB-Trace `format_version=1`,
> `canonicalisation_version=1`, length-prefixed canonical JSON frames,
> per-frame HMAC-SHA256 chained via `prev_hmac`, per-frame Ed25519
> signatures over `prev_hmac || canonical_json(body)`. The shared
> conformance corpus lives at `stepback-core/fixtures/v1/` and every
> binding's test suite is wired to it.

---

## Quick reference

| Path | Language / Runtime | Capability | Maturity |
| --- | --- | --- | --- |
| `pipx install stepback` | Python ≥ 3.11 | Recorder + replay + CLI | **Stable** |
| `pip install stepback` (in venv) | Python ≥ 3.11 | Recorder + replay + library | **Stable** |
| `pip install -e ".[dev]"` | Python ≥ 3.11 | Source / contributor install | **Stable** |
| `pip install stepback[shims,bench]` | Python ≥ 3.11 | + provider SDK shims + bench harness | **Stable** |
| `cargo install --path stepback-core/crates/sb-verify` | Rust ≥ 1.74 | CLI verifier | **Experimental** |
| `cargo add sb-format sb-verify` | Rust ≥ 1.74 | Reader + verifier crates | **Experimental** |
| `pip install stepback-core` (PyO3 wheel) | Python ≥ 3.11 | Rust verifier behind `STEPBACK_USE_RUST_VERIFIER=1` | **Experimental** |
| `npm install @stepback/core` | Node ≥ 18.17 | Reader + verifier (ESM/CJS) | **Experimental** |
| `go get github.com/stepback-dev/stepback-go` | Go ≥ 1.21 | Reader + verifier | **Experimental** |
| `bindings/jvm` (Gradle) | JDK 17 LTS | Reader + verifier | **Preview** |
| `bindings/dotnet` (`dotnet add package Stepback.Sb`) | .NET 8 LTS | Reader + verifier | **Preview** |
| `wasm-pack build stepback-core/crates/sb-wasm` | WASM (browser / Node) | In-browser verify + summarize | **Experimental** |
| `docker pull stepback/proxy:distroless` | Container (any host) | HTTP/gRPC sidecar recorder | **Experimental** |
| `docker pull stepback/proxy:debug` | Container (any host) | Same, with shell + curl + jq for triage | **Experimental** |

Anything not listed here (Vertex / Azure shims, framework recorders for
LangGraph / DSPy / AutoGen, etc.) is **roadmap**, not installable.

---

## 1. Python — pipx (recommended for the CLI)

**Maturity: Stable.** This is the path most users want. `pipx` installs
`stepback` into its own isolated venv and exposes the `stepback` command on
`$PATH`, sidestepping PEP 668 on Homebrew / distro Pythons.

```bash
pipx install stepback
stepback --help
```

Optional extras that pull in provider SDKs and the benchmark harness:

```bash
pipx install "stepback[shims,bench]"
```

Upgrade and uninstall:

```bash
pipx upgrade stepback
pipx uninstall stepback
```

Requires Python ≥ 3.11. `pipx` itself can be installed with
`brew install pipx`, `apt install pipx`, or `python -m pip install --user pipx`.

### Shell completion

`stepback completion <shell>` prints a completion script for bash, zsh,
fish, or PowerShell (pwsh). Run once after installing, then restart your
shell (or source the file immediately).

**bash** — add to `~/.bashrc` or drop a file into `~/.bash_completion.d/`:

```bash
mkdir -p ~/.bash_completion.d
stepback completion bash > ~/.bash_completion.d/stepback
# Then reload: source ~/.bash_completion.d/stepback
```

Or for eval-based sourcing (no file required):

```bash
echo 'source <(stepback completion bash)' >> ~/.bashrc
```

**zsh** — requires `$fpath` to include `~/.zfunc` and `compinit` loaded:

```zsh
mkdir -p ~/.zfunc
stepback completion zsh > ~/.zfunc/_stepback
# Add to ~/.zshrc if not already present:
#   fpath=(~/.zfunc $fpath)
#   autoload -Uz compinit && compinit
```

**fish**:

```fish
stepback completion fish > ~/.config/fish/completions/stepback.fish
```

**PowerShell (pwsh)**:

```pwsh
stepback completion pwsh >> $PROFILE
```

---

## 2. Python — venv + pip (recommended for libraries)

**Maturity: Stable.** Use this when you are importing `stepback` from your
own Python project rather than running the CLI.

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install stepback                # runtime only
pip install "stepback[shims]"       # + OpenAI / Anthropic / Bedrock / Gemini SDKs
pip install "stepback[shims,bench,dev]"  # contributor / benchmarking install
```

PEP 668-managed Pythons (Homebrew, Debian/Ubuntu system Python) will refuse
`pip install` outside a venv. Either use a venv as above, use `pipx`, or
pass `--user` if you genuinely want a user-site install.

### Source / contributor install

```bash
git clone https://github.com/stepback-dev/stepback
cd stepback
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                              # 605+ tests, ~15s on a developer laptop
```

The `dev` extra pulls in `pytest`, `hypothesis`, `cryptography`, and the
proxy-side test deps. Build distributables with:

```bash
python -m build                     # wheel + sdist into dist/
./scripts/smoke_install.sh          # builds, installs into a fresh venv, smoke-tests
```

---

## 3. Rust — `stepback-core` workspace

**Maturity: Experimental.** The Rust workspace under `stepback-core/` holds
five crates — `sb-format`, `sb-canonical`, `sb-dirty`, `sb-replay`, and
`sb-verify` — plus a `sb-wasm` shim. None are on crates.io yet; the
intended publish path is on a per-crate basis once the public API has been
stable for a release cycle.

### From a git checkout

```bash
git clone https://github.com/stepback-dev/stepback
cd stepback/stepback-core
cargo build --workspace
cargo test  --workspace
```

### As a dependency in your own crate

While the crates are pre-publish, depend on a git revision:

```toml
# Cargo.toml
[dependencies]
sb-format = { git = "https://github.com/stepback-dev/stepback", rev = "<commit-sha>" }
sb-verify = { git = "https://github.com/stepback-dev/stepback", rev = "<commit-sha>" }
```

### As a CLI verifier

```bash
cargo install --path stepback-core/crates/sb-verify
sb-verify ./traces/incident.sb --hmac-key-hex "$SB_HMAC_KEY_HEX"
```

Toolchain: Rust ≥ 1.74. The workspace pins `rust-version = "1.74"` and is
tested against stable + the previous stable.

---

## 4. Python ↔ Rust — PyO3 bindings (`stepback-core` wheel)

**Maturity: Experimental.** `bindings/python/stepback_core/` is a PyO3
extension that re-exports the Rust verifier under stable Python names. The
Python package routes `verify_trace` through Rust when
`STEPBACK_USE_RUST_VERIFIER=1` is set; otherwise the pure-Python verifier
runs (and is the default).

Build a local wheel with [maturin](https://www.maturin.rs/):

```bash
pip install maturin
cd bindings/python/stepback_core
maturin develop --release            # installs into the active venv
```

Use:

```bash
STEPBACK_USE_RUST_VERIFIER=1 stepback verify ./traces/incident.sb
```

A pre-built wheel will be published to PyPI once the Rust ABI is frozen.

---

## 5. TypeScript / Node — `@stepback/core`

**Maturity: Experimental.** Pure TypeScript reader + verifier with no
native dependency. Uses Node's built-in `crypto` (Ed25519 via
`crypto.verify`, HMAC-SHA256 via `crypto.createHmac`).

```bash
npm install @stepback/core           # or: pnpm add / yarn add
```

ESM:

```ts
import { verifyBytes, VerifyError } from "@stepback/core";
import { readFile } from "node:fs/promises";

const trace = await readFile("./incident.sb");
const hmacKey = Buffer.from(process.env.SB_HMAC_KEY_HEX!, "hex");
const v = verifyBytes(new Uint8Array(trace), hmacKey);
console.log(`OK: ${v.frameCount} frames, recorder=${v.header.recorder_version}`);
```

CJS:

```js
const { verifyBytes } = require("@stepback/core");
```

Requires Node ≥ 18.17 (for stable `crypto.verify` Ed25519 support and
`node:test`). Browser builds should consume the WASM bundle (see §9), not
this package.

### From a git checkout

```bash
cd bindings/typescript
npm install
npm run build         # populates dist/{esm,cjs,types}
npm test
```

---

## 6. Go — `github.com/stepback-dev/stepback-go`

**Maturity: Experimental.** Pure-Go reader + verifier. Standard library
only (`crypto/ed25519`, `crypto/hmac`, `crypto/sha256`,
`encoding/binary`, `encoding/json`).

```bash
go get github.com/stepback-dev/stepback-go
```

```go
import (
    "log"
    sb "github.com/stepback-dev/stepback-go"
)

func main() {
    hmacKey := []byte{ /* 32 bytes from your KMS */ }
    v, err := sb.VerifyPath("trace.sb", hmacKey)
    if err != nil {
        log.Fatal(err)
    }
    log.Printf("verified %d frames; recorder=%s", v.FrameCount, v.Header.RecorderVersion)
}
```

Requires Go ≥ 1.21. The module path is
`github.com/stepback-dev/stepback-go` and the package name is `sb`.

### From a git checkout

```bash
cd bindings/go
go test ./...
```

---

## 7. JVM — `bindings/jvm`

**Maturity: Preview.** Pure JDK 17 reader + verifier; no third-party
runtime dependencies. JUnit 5 is used for tests only.

```bash
cd bindings/jvm
gradle build
gradle test
```

There is no Maven Central release yet. Consume from a local build via
`gradle publishToMavenLocal` and depend on
`dev.stepback:stepback-jvm:0.1.0-SNAPSHOT` from your project.

```java
import dev.stepback.sb.Verifier;
import dev.stepback.sb.VerifiedTrace;
import java.nio.file.Path;

byte[] hmacKey = ...; // 32 bytes from your KMS
VerifiedTrace v = Verifier.verifyPath(Path.of("trace.sb"), hmacKey);
System.out.printf("verified %d frames; recorder=%s%n",
    v.frameCount(), v.header().recorderVersion());
```

Toolchain target: Java 17 LTS. Kotlin consumers can use the same Java API
unchanged; first-class Kotlin idioms are roadmap.

---

## 8. .NET — `Stepback.Sb`

**Maturity: Preview.** .NET 8 reader + verifier. Uses the BCL
(`System.Security.Cryptography.HMACSHA256`, `System.Text.Json`) plus
[BouncyCastle.Cryptography](https://www.bouncycastle.org/csharp/) for
Ed25519 (the .NET 8 BCL still does not ship Ed25519 in any stable surface).

There is no NuGet release yet. Build locally:

```bash
cd bindings/dotnet
dotnet build -c Release
dotnet test
```

Then reference the project from your solution, or
`dotnet pack` to produce a local `.nupkg`. Future:

```bash
dotnet add package Stepback.Sb
```

```csharp
using Stepback.Sb;

byte[] hmacKey = ...; // 32 bytes from your KMS
var v = SbVerifier.VerifyPath("trace.sb", hmacKey);
Console.WriteLine(
    $"verified {v.FrameCount} frames; recorder={v.Header.RecorderVersion}");
```

Targets .NET 8 LTS only. Older targets (`netstandard2.0`,
`net6.0`) are not supported and have no roadmap entry.

---

## 9. WASM — in-browser verify and summarize

**Maturity: Experimental.** The WASM bundle wraps the Rust `sb-verify`
crate. Headline use case: open a `.sb` trace in a browser, see what is in
it, optionally verify the HMAC chain and Ed25519 signatures — without
sending any trace bytes to a server.

### Build

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack             # one-time
./scripts/build_wasm.sh
```

This produces three bundles under `wasm/`:

| Bundle | `wasm-pack` target | Consumer |
| --- | --- | --- |
| `wasm/pkg/`         | `--target web`      | Native browser ESM (`<script type="module">`) |
| `wasm/pkg-bundler/` | `--target bundler`  | Webpack / Vite / Rollup / esbuild |
| `wasm/pkg-nodejs/`  | `--target nodejs`   | Node.js (smoke test target) |

### Try the demo

```bash
# from the repo root, after build_wasm.sh has run
python3 -m http.server -d wasm 8000
# open http://localhost:8000/demo/
```

Drop a `.sb` file into the demo page; verification runs entirely in the
browser tab.

### Use from a bundler

```ts
import init, { verify_bytes } from "@stepback/core-wasm";
await init();
const v = verify_bytes(new Uint8Array(traceBytes), hmacKeyBytes);
```

The WASM bundle is **read-only**. There is no recorder in the browser and
there is no roadmap entry to add one.

---

## 10. `sb proxy` — sidecar recorder (HTTP + gRPC)

**Maturity: Experimental.** A language-agnostic sidecar so Go, Rust, JVM,
Node, and legacy stacks can record `.sb` traces without a Python
dependency. HTTP endpoints today (`StartTrace`, `RecordStep`, `EndTrace`,
`VerifyTrace`, `/healthz`, `/v1/info`); gRPC equivalents track behind the
HTTP surface.

### Run from source

```bash
pip install "stepback[grpc]"
sb proxy --listen :4319 --write ./traces
```

### Run from a container image (recommended)

Two flavours, both built from the same Python 3.11 source tree, both
listening on `0.0.0.0:4319`, both running as UID 65532:

| Image | Base | Use case |
| --- | --- | --- |
| `stepback/proxy:distroless` | `gcr.io/distroless/python3-debian12:nonroot` | Production / sidecar |
| `stepback/proxy:debug`      | `python:3.11-slim-bookworm` (+ bash, curl, jq, tini, gRPC extra) | Local triage, CI smoke |

```bash
mkdir -p ./traces
docker run --rm -p 4319:4319 \
  -v "$PWD/traces:/var/lib/stepback/traces" \
  stepback/proxy:distroless

# in another shell
curl -sf http://127.0.0.1:4319/healthz | jq
curl -sf http://127.0.0.1:4319/v1/info | jq
```

The trace directory inside the container is `/var/lib/stepback/traces`;
mount a host directory over it so `.sb` files survive container exits.

### Build the images yourself

```bash
docker build -f docker/Dockerfile.distroless -t stepback/proxy:distroless .
docker build -f docker/Dockerfile.debug      -t stepback/proxy:debug      .
./scripts/smoke_proxy_docker.sh                    # distroless smoke test
./scripts/smoke_proxy_docker.sh debug              # debug-image smoke test
```

The smoke test drives a full
`StartTrace → RecordStep × 3 → EndTrace → VerifyTrace` cycle over HTTP and
re-verifies the resulting `.sb` from the host using the canonical Python
reader. It is also wired into CI (`.github/workflows/docker.yml`).

The distroless image carries **no shell, package manager, or setuid
binary**; `docker exec` will fail. Pull `:debug` instead when you need a
shell. Neither image bakes in the `[shims]` extra — clients send
fully-formed step dicts.

---

## Cross-binding compatibility matrix

A `.sb` trace written by **any** writer below is verifiable by **every**
reader marked ✅:

| Writer ↓ \ Reader → | Python | Rust | PyO3 | TS / Node | Go | JVM | .NET | WASM |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Python (`stepback`)        | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Rust (`sb-format`, planned) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `sb proxy` (containerised) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

The shared conformance corpus that backs this matrix lives at
`stepback-core/fixtures/v1/`. Every reader's CI job runs against it.

---

## Choosing an install path

- **You want to record a Python agent.** §1 (pipx) for the CLI, §2 (pip in
  a venv) for `from stepback import record, replay`.
- **You want to record from Go / Rust / JVM / Node / .NET.** §10 (the
  proxy). Native recorders are roadmap, not shipping.
- **You want to verify or inspect a `.sb` file from $LANGUAGE.** §3
  (Rust), §5 (TS/Node), §6 (Go), §7 (JVM), §8 (.NET), §9 (WASM in a
  browser tab).
- **You are on a container-only host.** §10. The image carries `cryptography`
  and the canonical Python reader; nothing else is required.
- **You are contributing.** §2 (source install) and `pytest`. Then `cargo
  test --workspace` under `stepback-core/`, plus whichever binding you are
  touching.

---

## Versioning

Two SemVer tracks coexist:

1. **Package SemVer.** Each binding (`stepback`, `sb-format`,
   `@stepback/core`, `stepback-go`, `stepback-jvm`, `Stepback.Sb`) is
   versioned independently using the standard ecosystem rules.
2. **SB-Trace wire-format SemVer.** Versioned independently of any
   binding. `format_version=1` and `canonicalisation_version=1` are the
   current frozen surface. Old readers reject unknown mandatory fields;
   new readers must read v1 forever; v2 features are negotiated via
   capability frames rather than guessed from optional blobs.

A binding marked **Stable** above promises: no breaking source-level API
changes within a major version, and no breaking `.sb` interpretation
within a `format_version` / `canonicalisation_version` pair.

A binding marked **Experimental** or **Preview** does **not** promise
either yet — pin exact versions and read the changelog before upgrading.
