# stepback-go

Read-only Go reader and verifier for SB-Trace `.sb` v1 traces.

This package is the Go counterpart of the Rust reference
(`stepback-core/crates/sb-format`, `sb-verify`) and the TypeScript
reference (`@stepback/core`). It reads `.sb` traces written by the
Python recorder and verifies HMAC chaining + per-frame Ed25519
signatures. There is no writer here — recording stays in the Python
reference until the multi-language core lands.

## Install

```bash
go get github.com/stepback-dev/stepback-go
```

The module path is `github.com/stepback-dev/stepback-go`. The package
name is `sb`.

## Usage

```go
import (
    "fmt"
    "log"

    sb "github.com/stepback-dev/stepback-go"
)

func main() {
    hmacKey := []byte{ /* 32 bytes from your KMS */ }
    v, err := sb.VerifyPath("trace.sb", hmacKey)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("verified %d frames; recorder=%s, public_key=%s\n",
        v.FrameCount, v.Header.RecorderVersion, v.Header.PublicKey)
}
```

Frame-by-frame streaming:

```go
f, _ := os.Open("trace.sb")
defer f.Close()
r := sb.NewReader(f)
for {
    frame, err := r.Next()
    if errors.Is(err, io.EOF) { break }
    if err != nil { log.Fatal(err) }
    fmt.Printf("body bytes: %s\n", frame.BodyBytes)
}
```

## Public surface

| Symbol | Purpose |
| --- | --- |
| `Reader`, `NewReader`, `(*Reader).Next` | Streaming frame splitter; no crypto. |
| `Frame` | One wrapper: `Body`, `BodyBytes`, `PrevHMAC`, `HMAC`, `Sig`. |
| `IterFrames(buf)` | In-memory frame splitter. |
| `Verify(buf, hmacKey)` | End-to-end HMAC + signature verification. |
| `VerifyPath(path, hmacKey)` | Convenience wrapper around `os.ReadFile`. |
| `VerifyReader(r, hmacKey)` | Streaming verification. |
| `VerifiedTrace` | `Header`, `FrameCount`. |
| `TraceHeader` | Parsed header body with `Extra map[string]json.RawMessage`. |
| `VerifyError`, `VerifyErrorKind` | Stable, machine-readable error discriminator. |
| `FrameError`, `FrameErrorKind` | Frame-splitter errors. |
| `FormatVersion`, `CanonicalisationVersion`, `MaxFrameBytes`, `FrameLengthPrefix`, `ZeroHMAC`, `Version` | Pinned constants. |

The verifier:

1. Walks frames in order.
2. Recomputes `HMAC_SHA256(hmac_key, prev_hmac_bytes || canonical_json(body))`
   for each frame, comparing in constant time.
3. Verifies the per-frame Ed25519 signature against the **raw 32-byte
   HMAC digest**, not the hex-encoded form, using the public key
   pinned in the header.
4. Rejects trace tampering (chain breaks, frame insertion, frame
   reordering, frame truncation, or any single-bit flip in `body`,
   `hmac`, `prev_hmac`, or `sig`).

The frame splitter uses the verbatim wrapper byte slice for the
body, not a re-canonicalised form. Go's `encoding/json` decodes
numbers into `float64` by default and would not round-trip the int64
fields routinely carried in step bodies (`wallclock_ns`, `cpu_ns`);
re-canonicalising them would (correctly) HMAC-mismatch.

## Conformance

Tests run against the same `.sb` corpus the Rust verifier uses
(`stepback-core/fixtures/v1/`), so any divergence between the Go,
Rust, and TypeScript verifiers shows up immediately. Coverage
includes:

- The three "good" fixtures (`header_only.sb`, `multi_step.sb`,
  `with_blobs.sb`) — verify cleanly.
- The five "corrupt" fixtures (`truncated_body.sb`, `flipped_hmac.sb`,
  `flipped_sig.sb`, `broken_chain.sb`, `bad_format_version.sb`) —
  rejected with one of the documented `VerifyErrorKind` values.
- Empty input is rejected with `MissingHeader`.
- Wrong HMAC key is rejected with `HmacMismatch` at frame 0.
- Oversized length prefix is rejected with `FrameTooLarge`.
- Truncated body / truncated length prefix surface as `UnexpectedEof`.
- Fixture SHA-256s match the manifest.

```bash
cd bindings/go
go test ./...
```

## Compatibility

| | Read | Write |
| --- | --- | --- |
| `.sb` v1 | ✅ | ❌ |
| `.sb` v2 | n/a | n/a |

This is a v0.1 release. The package surface is stable for the v1
trace format; future minor versions will add helpers but not break
the read path.

## License

Apache-2.0. See `LICENSE` at the repo root.
