# stepback-jvm

Read-only Java reader and verifier for SB-Trace `.sb` v1 traces.

This package is the JVM counterpart of the Rust reference
(`stepback-core/crates/sb-format`, `sb-verify`), the Go reference
(`bindings/go`), and the TypeScript reference (`@stepback/core`). It
reads `.sb` traces written by the Python recorder and verifies
HMAC-SHA256 chaining + per-frame Ed25519 signatures. There is no
writer here — recording stays in the Python reference until the
multi-language core lands.

The implementation is **pure JDK 17** — no third-party JSON, crypto,
or codec dependencies are pulled into the runtime classpath. JUnit 5
is used for tests only.

## Build

```bash
cd bindings/jvm
gradle build       # or: ./gradlew build once a wrapper is added
gradle test
```

The `test` task pins `stepback.fixtures` to the shared corpus at
`stepback-core/fixtures/v1/` so the conformance tests run against the
exact same bytes the Rust, Go, and TypeScript verifiers consume.
Toolchain target: Java 17 LTS.

## Usage

```java
import dev.stepback.sb.Verifier;
import dev.stepback.sb.VerifiedTrace;
import java.nio.file.Path;

byte[] hmacKey = ... ; // 32 bytes from your KMS
VerifiedTrace v = Verifier.verifyPath(Path.of("trace.sb"), hmacKey);
System.out.printf("verified %d frames; recorder=%s, public_key=%s%n",
    v.frameCount(), v.header().recorderVersion(), v.header().publicKeyHex());
```

Frame-by-frame streaming:

```java
import dev.stepback.sb.Frame;
import dev.stepback.sb.FrameReader;

try (FrameReader r = new FrameReader(Files.newInputStream(path))) {
    Frame f;
    while ((f = r.next()) != null) {
        System.out.println(new String(f.bodyBytes()));
    }
}
```

## Public surface

| Symbol | Purpose |
| --- | --- |
| `FrameReader`, `FrameReader#next` | Streaming frame splitter; no crypto. |
| `FrameReader.iterFrames(byte[])` | In-memory frame splitter. |
| `Frame` | One wrapper: `bodyBytes`, `prevHmac`, `hmac`, `sig`. |
| `Verifier.verify(byte[], byte[])` | End-to-end HMAC + signature verification. |
| `Verifier.verifyPath(Path, byte[])` | Convenience wrapper around `Files.readAllBytes`. |
| `Verifier.verifyStream(InputStream, byte[])` | Streaming verification. |
| `VerifiedTrace` | `header`, `frameCount`. |
| `TraceHeader` | Parsed header body — `formatVersion`, `publicKeyHex`, … |
| `VerifyError`, `VerifyError.Kind` | Stable, machine-readable error discriminator. |
| `FrameError`, `FrameError.Kind` | Frame-splitter errors. |
| `Constants` | `FORMAT_VERSION`, `CANONICALISATION_VERSION`, `MAX_FRAME_BYTES`, `FRAME_LENGTH_PREFIX`, `VERSION`. |

The verifier:

1. Walks frames in order.
2. Recomputes `HMAC_SHA256(hmac_key, prev_hmac_bytes || canonical_json(body))`
   for each frame, comparing in constant time.
3. Verifies the per-frame Ed25519 signature against the **raw 32-byte
   HMAC digest**, not its hex form, using the public key pinned in
   the header.
4. Rejects trace tampering (chain breaks, frame insertion, frame
   reordering, frame truncation, or any single-bit flip in `body`,
   `hmac`, `prev_hmac`, or `sig`).

The frame splitter uses the verbatim wrapper byte slice for the body,
not a re-canonicalised form. Re-canonicalising would lose precision
on int64 fields (`wallclock_ns`, `cpu_ns`) and trigger spurious HMAC
mismatches.

The Ed25519 public key carried in the header as 32 raw hex bytes is
wrapped in a synthesised X.509 SubjectPublicKeyInfo envelope so the
JDK's stock `KeyFactory.getInstance("Ed25519")` can ingest it without
BouncyCastle or any other third-party provider.

## Conformance

Tests run against the same corpus the Rust, Go, and TypeScript
verifiers use (`stepback-core/fixtures/v1/`), so any divergence shows
up immediately. Coverage includes:

- Three "good" fixtures (`header_only.sb`, `multi_step.sb`,
  `with_blobs.sb`) — verify cleanly.
- Five "corrupt" fixtures (`truncated_body.sb`, `flipped_hmac.sb`,
  `flipped_sig.sb`, `broken_chain.sb`, `bad_format_version.sb`) —
  rejected with one of the documented `VerifyError.Kind` values.
- Empty input rejected with `MissingHeader`.
- Wrong HMAC key rejected with `HmacMismatch` at frame 0.
- Oversized length prefix rejected with `FrameTooLarge`.
- Truncated body / truncated length prefix surface as
  `UnexpectedEof`.
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
