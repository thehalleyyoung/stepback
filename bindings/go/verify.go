package sb

import (
	"crypto/ed25519"
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
)

// VerifyErrorKind is a stable, machine-readable discriminator for
// the ways verification can fail. The same set is used by the Rust
// (`sb-verify::VerifyError`) and TypeScript (`@stepback/core`)
// references.
type VerifyErrorKind string

const (
	VerifyErrorParse                    VerifyErrorKind = "Parse"
	VerifyErrorMissingHeader            VerifyErrorKind = "MissingHeader"
	VerifyErrorUnsupportedFormatVersion VerifyErrorKind = "UnsupportedFormatVersion"
	VerifyErrorBrokenChain              VerifyErrorKind = "BrokenChain"
	VerifyErrorHmacMismatch             VerifyErrorKind = "HmacMismatch"
	VerifyErrorBadHex                   VerifyErrorKind = "BadHex"
	VerifyErrorUnsupportedSignature     VerifyErrorKind = "UnsupportedSignature"
	VerifyErrorBadSignatureLength       VerifyErrorKind = "BadSignatureLength"
	VerifyErrorSignatureMismatch        VerifyErrorKind = "SignatureMismatch"
	VerifyErrorBadPublicKey             VerifyErrorKind = "BadPublicKey"
)

// VerifyError is the error type returned by Verify, VerifyPath, and
// VerifyReader.
type VerifyError struct {
	Kind       VerifyErrorKind
	Message    string
	FrameIndex int    // 0-based; -1 if the error is not frame-local
	Field      string // optional: "hmac", "sig", "prev_hmac", ...
	Cause      error
}

func (e *VerifyError) Error() string {
	if e.FrameIndex >= 0 {
		return fmt.Sprintf("sb: %s (frame %d): %s", e.Kind, e.FrameIndex, e.Message)
	}
	return fmt.Sprintf("sb: %s: %s", e.Kind, e.Message)
}

func (e *VerifyError) Unwrap() error { return e.Cause }

// VerifiedTrace is the outcome of a successful verification: the
// parsed header plus the number of frames that chained cleanly.
type VerifiedTrace struct {
	Header     *TraceHeader
	FrameCount int
}

// ZeroHMAC is 32 zero bytes — the seed of the HMAC chain.
var ZeroHMAC = make([]byte, 32)

// signatureLength is the byte length of an Ed25519 signature.
const signatureLength = ed25519.SignatureSize

// Verify reads a `.sb` byte slice end-to-end. The HMAC key is supplied
// by the caller because key management lives outside this package.
//
// On success the caller may trust that:
//
//   - every frame's HMAC is valid given the chain;
//   - every frame's Ed25519 signature was issued by the holder of the
//     private key whose public counterpart is pinned in the header;
//   - no frame was inserted, dropped, reordered, or rewritten.
func Verify(buf []byte, hmacKey []byte) (*VerifiedTrace, error) {
	frames, ferr := IterFrames(buf)
	if ferr != nil {
		// Even if some frames parsed before the failure, surface the
		// frame-level error at the index where parsing failed.
		return nil, &VerifyError{
			Kind:       VerifyErrorParse,
			Message:    fmt.Sprintf("frame %d: %s", len(frames), ferr.Error()),
			FrameIndex: len(frames),
			Cause:      ferr,
		}
	}
	return verifyFrames(frames, hmacKey)
}

// VerifyPath reads `path` from disk and verifies it.
func VerifyPath(path string, hmacKey []byte) (*VerifiedTrace, error) {
	buf, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Verify(buf, hmacKey)
}

// VerifyReader streams a `.sb` from r and verifies it. Useful for
// large traces that should not be held in memory.
func VerifyReader(r io.Reader, hmacKey []byte) (*VerifiedTrace, error) {
	rdr := NewReader(r)
	frames := make([]*Frame, 0, 8)
	for {
		f, err := rdr.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			var fe *FrameError
			if errors.As(err, &fe) {
				return nil, &VerifyError{
					Kind:       VerifyErrorParse,
					Message:    fmt.Sprintf("frame %d: %s", len(frames), err.Error()),
					FrameIndex: len(frames),
					Cause:      err,
				}
			}
			return nil, err
		}
		frames = append(frames, f)
	}
	return verifyFrames(frames, hmacKey)
}

func verifyFrames(frames []*Frame, hmacKey []byte) (*VerifiedTrace, error) {
	if len(frames) == 0 {
		return nil, &VerifyError{
			Kind:       VerifyErrorMissingHeader,
			Message:    "trace has no header frame (empty input)",
			FrameIndex: -1,
		}
	}
	var header *TraceHeader
	var publicKey ed25519.PublicKey
	prevHmacBytes := append([]byte(nil), ZeroHMAC...)

	for index, frame := range frames {
		if err := verifyChainLink(index, frame, prevHmacBytes, hmacKey); err != nil {
			return nil, err
		}
		if index == 0 {
			hdr, err := parseHeader(frame.BodyBytes)
			if err != nil || hdr.Type != "header" {
				return nil, &VerifyError{
					Kind:       VerifyErrorMissingHeader,
					Message:    "frame 0 must be a header object",
					FrameIndex: -1,
				}
			}
			if hdr.FormatVersion != FormatVersion {
				return nil, &VerifyError{
					Kind: VerifyErrorUnsupportedFormatVersion,
					Message: fmt.Sprintf(
						"unsupported format_version %d, expected %d",
						hdr.FormatVersion, FormatVersion),
					FrameIndex: -1,
				}
			}
			pk, perr := parseEd25519PublicKey(hdr.PublicKey)
			if perr != nil {
				return nil, &VerifyError{
					Kind:       VerifyErrorBadPublicKey,
					Message:    "trace header used an unsupported public key encoding",
					FrameIndex: -1,
					Cause:      perr,
				}
			}
			publicKey = pk
			header = hdr
		}
		if publicKey == nil {
			return nil, &VerifyError{
				Kind:       VerifyErrorBadPublicKey,
				Message:    "no public key resolved from header",
				FrameIndex: -1,
			}
		}
		if err := verifySignature(index, frame, publicKey); err != nil {
			return nil, err
		}
		nextPrev, err := decodeHexFixed(index, "hmac", frame.HMAC, 32)
		if err != nil {
			return nil, err
		}
		prevHmacBytes = nextPrev
	}
	return &VerifiedTrace{Header: header, FrameCount: len(frames)}, nil
}

func verifyChainLink(index int, frame *Frame, expectedPrev, hmacKey []byte) error {
	claimedPrev, err := decodeHexFixed(index, "prev_hmac", frame.PrevHMAC, 32)
	if err != nil {
		return err
	}
	if subtle.ConstantTimeCompare(claimedPrev, expectedPrev) != 1 {
		return &VerifyError{
			Kind:       VerifyErrorBrokenChain,
			Message:    "prev_hmac does not chain to previous frame",
			FrameIndex: index,
		}
	}
	mac := hmac.New(sha256.New, hmacKey)
	mac.Write(expectedPrev)
	// Use the verbatim wrapper byte slice rather than re-canonicalising
	// frame.Body. Go's encoding/json decodes JSON numbers into
	// float64 by default and would not round-trip int64 fields like
	// wallclock_ns or cpu_ns; re-canonicalising would produce a
	// different lexical form and (correctly) HMAC-mismatch.
	mac.Write(frame.BodyBytes)
	computed := mac.Sum(nil)
	claimedHmac, err := decodeHexAny(index, "hmac", frame.HMAC)
	if err != nil {
		return err
	}
	if subtle.ConstantTimeCompare(claimedHmac, computed) != 1 {
		return &VerifyError{
			Kind:       VerifyErrorHmacMismatch,
			Message:    "HMAC mismatch",
			FrameIndex: index,
		}
	}
	return nil
}

func verifySignature(index int, frame *Frame, publicKey ed25519.PublicKey) error {
	colon := strings.IndexByte(frame.Sig, ':')
	scheme := ""
	hexPart := frame.Sig
	if colon >= 0 {
		scheme = frame.Sig[:colon]
		hexPart = frame.Sig[colon+1:]
	}
	if scheme != "ed25519" {
		return &VerifyError{
			Kind:       VerifyErrorUnsupportedSignature,
			Message:    fmt.Sprintf("signature scheme %q is not supported", scheme),
			FrameIndex: index,
		}
	}
	sigBytes, err := hex.DecodeString(hexPart)
	if err != nil {
		return &VerifyError{
			Kind:       VerifyErrorBadHex,
			Message:    "invalid hex in sig",
			FrameIndex: index,
			Field:      "sig",
			Cause:      err,
		}
	}
	if len(sigBytes) != signatureLength {
		return &VerifyError{
			Kind: VerifyErrorBadSignatureLength,
			Message: fmt.Sprintf(
				"signature length %d is not %d", len(sigBytes), signatureLength),
			FrameIndex: index,
		}
	}
	// The Python writer signs the **raw 32-byte HMAC digest**, not
	// its hex form. Decode frame.HMAC back to bytes before verifying.
	hmacBytes, err := hex.DecodeString(frame.HMAC)
	if err != nil {
		return &VerifyError{
			Kind:       VerifyErrorBadHex,
			Message:    "invalid hex in hmac",
			FrameIndex: index,
			Field:      "hmac",
			Cause:      err,
		}
	}
	if !ed25519.Verify(publicKey, hmacBytes, sigBytes) {
		return &VerifyError{
			Kind:       VerifyErrorSignatureMismatch,
			Message:    "Ed25519 signature did not verify",
			FrameIndex: index,
		}
	}
	return nil
}

func parseEd25519PublicKey(hexStr string) (ed25519.PublicKey, error) {
	raw, err := hex.DecodeString(hexStr)
	if err != nil {
		return nil, err
	}
	if len(raw) != ed25519.PublicKeySize {
		return nil, fmt.Errorf("ed25519 public key must be %d bytes, got %d",
			ed25519.PublicKeySize, len(raw))
	}
	return ed25519.PublicKey(raw), nil
}

func decodeHexFixed(index int, field, s string, expected int) ([]byte, error) {
	bytes, err := decodeHexAny(index, field, s)
	if err != nil {
		return nil, err
	}
	if len(bytes) != expected {
		return nil, &VerifyError{
			Kind: VerifyErrorBadHex,
			Message: fmt.Sprintf(
				"%s expected %d bytes, got %d", field, expected, len(bytes)),
			FrameIndex: index,
			Field:      field,
		}
	}
	return bytes, nil
}

func decodeHexAny(index int, field, s string) ([]byte, error) {
	bytes, err := hex.DecodeString(s)
	if err != nil {
		return nil, &VerifyError{
			Kind:       VerifyErrorBadHex,
			Message:    "invalid hex in " + field,
			FrameIndex: index,
			Field:      field,
			Cause:      err,
		}
	}
	return bytes, nil
}
