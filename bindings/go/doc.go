// Package sb is a read-only reader and verifier for SB-Trace `.sb` v1
// traces written by the Python reference recorder (`stepback`).
//
// The on-disk wire format is the same one implemented by the Rust
// reference (`stepback-core/crates/sb-format`, `sb-verify`) and the
// TypeScript reference (`@stepback/core`):
//
//	| 4-byte big-endian length | canonical-JSON wrapper |
//
// where the wrapper is a canonical-JSON object with keys `body`,
// `prev_hmac`, `hmac`, and `sig`. The HMAC chain seed is 32 zero
// bytes; each frame's HMAC is computed as
//
//	HMAC_SHA256(hmac_key, prev_hmac_bytes || canonical_json(body))
//
// and the per-frame Ed25519 signature is over the **raw 32-byte HMAC
// digest**, not its hex-encoded form, against the public key pinned
// in the header frame.
//
// This package is intentionally minimal: it owns the Reader, the
// Frame struct, and the Verify entry points. It does not write
// traces, manage HMAC keys, or interpret step semantics — that lives
// in the Python recorder and (eventually) in `stepback-core`.
//
// Example:
//
//	f, err := os.Open("trace.sb")
//	if err != nil { log.Fatal(err) }
//	defer f.Close()
//	r := sb.NewReader(f)
//	for {
//	    frame, err := r.Next()
//	    if errors.Is(err, io.EOF) { break }
//	    if err != nil { log.Fatal(err) }
//	    fmt.Printf("frame body: %s\n", frame.BodyBytes)
//	}
//
// Or end-to-end verification against a known HMAC key:
//
//	v, err := sb.Verify(buf, hmacKey)
//	if err != nil { log.Fatal(err) }
//	fmt.Printf("verified %d frames; recorder=%s\n",
//	    v.FrameCount, v.Header.RecorderVersion)
package sb
