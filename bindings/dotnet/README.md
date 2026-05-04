# stepback-dotnet

Read-only .NET reader and verifier for SB-Trace `.sb` v1 traces.

This package is the .NET counterpart of the Rust reference
(`stepback-core/crates/sb-format`, `sb-verify`), the Go reference
(`bindings/go`), the JVM reference (`bindings/jvm`), and the
TypeScript reference (`@stepback/core`). It reads `.sb` traces written
by the Python recorder and verifies HMAC-SHA256 chaining + per-frame
Ed25519 signatures. There is no writer here — recording stays in the
Python reference until the multi-language core lands.

The implementation targets **.NET 8** and uses only the BCL
(`System.Security.Cryptography.HMACSHA256`, `System.Convert.FromHexString`,
`System.Buffers.Binary.BinaryPrimitives`, `System.Text.Json`) plus
**BouncyCastle.Cryptography** for Ed25519 verification. The .NET BCL
does not ship Ed25519 in any current stable release, so a managed
crypto provider is unavoidable for this binding; BouncyCastle is pure
managed and a single NuGet dependency.

xUnit is used for tests only.

## Build

```bash
cd bindings/dotnet
dotnet build -c Release
dotnet test
```

Targets .NET 8 LTS. The test project copies the shared corpus from
`stepback-core/fixtures/v1/` into the test bin output so the
conformance tests run against the exact same bytes the Rust, Go, JVM,
and TypeScript verifiers consume.

## Usage

```csharp
using Stepback.Sb;

byte[] hmacKey = ...; // 32 bytes from your KMS
var v = SbVerifier.VerifyPath("trace.sb", hmacKey);
Console.WriteLine(
    $"verified {v.FrameCount} frames; " +
    $"recorder={v.Header.RecorderVersion}, public_key={v.Header.PublicKeyHex}");
```

Frame-by-frame streaming:

```csharp
using var fs = File.OpenRead(path);
using var r = new SbTraceReader(fs);
while (r.Next() is { } f)
{
    Console.WriteLine(System.Text.Encoding.UTF8.GetString(f.BodyBytes));
}
```

## Public surface

| Symbol | Purpose |
| --- | --- |
| `SbTraceReader`, `SbTraceReader.Next()` | Streaming frame splitter; no crypto. |
| `SbTraceReader.IterFrames(byte[])` | In-memory frame splitter. |
| `Frame` | One wrapper: `BodyBytes`, `PrevHmac`, `Hmac`, `Sig`. |
| `SbVerifier.Verify(byte[], byte[])` | End-to-end HMAC + signature verification. |
| `SbVerifier.VerifyPath(string, byte[])` | Convenience wrapper around `File.ReadAllBytes`. |
| `SbVerifier.VerifyStream(Stream, byte[])` | Streaming verification. |
| `VerifiedTrace` | `Header`, `FrameCount`. |
| `TraceHeader` | Parsed header body — `FormatVersion`, `PublicKeyHex`, … |
| `VerifyException`, `VerifyErrorKind` | Stable, machine-readable error discriminator. |
| `FrameException`, `FrameErrorKind` | Frame-splitter errors. |
| `Constants` | `FormatVersion`, `CanonicalisationVersion`, `MaxFrameBytes`, `FrameLengthPrefix`, `Version`. |

The verifier:

1. Walks frames in order.
2. Recomputes `HMAC_SHA256(hmac_key, prev_hmac_bytes || canonical_json(body))`
   for each frame, comparing in constant time
   (`CryptographicOperations.FixedTimeEquals`).
3. Verifies the per-frame Ed25519 signature against the **raw 32-byte
   HMAC digest**, not its hex form, using the public key pinned in the
   header.
4. Rejects trace tampering (chain breaks, frame insertion, frame
   reordering, frame truncation, or any single-bit flip in `body`,
   `hmac`, `prev_hmac`, or `sig`).

The frame splitter uses the verbatim wrapper byte slice for the body,
not a re-canonicalised form. Re-canonicalising would lose precision on
int64 fields (`wallclock_ns`, `cpu_ns`) and trigger spurious HMAC
mismatches.

## Conformance

Tests run against the same corpus the Rust, Go, JVM, and TypeScript
verifiers use (`stepback-core/fixtures/v1/`), so any divergence shows
up immediately. Coverage includes:

- Three "good" fixtures (`header_only.sb`, `multi_step.sb`,
  `with_blobs.sb`) — verify cleanly.
- Five "corrupt" fixtures (`truncated_body.sb`, `flipped_hmac.sb`,
  `flipped_sig.sb`, `broken_chain.sb`, `bad_format_version.sb`) —
  rejected with one of the documented `VerifyErrorKind` values.
- Empty input rejected with `MissingHeader`.
- Wrong HMAC key rejected with `HmacMismatch` at frame 0.
- Oversized length prefix rejected with `FrameTooLarge`.
- Truncated body / truncated length prefix surface as `UnexpectedEof`.
- Fixture SHA-256s match the manifest.

## Compatibility

|              | Read | Write |
| ---          | ---  | ---   |
| `.sb` v1     | ✅   | ❌    |
| `.sb` v2     | n/a  | n/a   |

This is a v0.1 release. The package surface is stable for the v1
trace format; future minor versions will add helpers but not break
the read path.

## License

Apache-2.0. See `LICENSE` at the repo root.
