package sb

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// FormatVersion is the value pinned in every v1 header. Bumping this
// is a hard wire-format break; this package only reads v1.
const FormatVersion = 1

// FrameLengthPrefix is the width of the big-endian length prefix that
// precedes every frame on disk.
const FrameLengthPrefix = 4

// MaxFrameBytes is a defensive cap on a single decoded frame, in
// bytes. Large enough for any realistic agent step, small enough that
// a corrupted length prefix can't trigger a multi-gigabyte
// allocation.
const MaxFrameBytes = 64 * 1024 * 1024

// CanonicalisationVersion is the canonicalisation version this
// reader understands. Mirrors `stepback/canonical.py` and the Rust
// `sb-canonical` crate.
const CanonicalisationVersion = "1"

// FrameErrorKind is a stable, machine-readable discriminator for
// the ways frame parsing can fail.
type FrameErrorKind string

const (
	FrameErrorUnexpectedEof    FrameErrorKind = "UnexpectedEof"
	FrameErrorFrameTooLarge    FrameErrorKind = "FrameTooLarge"
	FrameErrorBadJson          FrameErrorKind = "BadJson"
	FrameErrorBadWrapperShape  FrameErrorKind = "BadWrapperShape"
)

// FrameError is returned by the Reader when a frame cannot be parsed.
// The Kind field carries a stable discriminator suitable for
// programmatic inspection; Length and Role are populated when
// relevant. FrameError implements `error` and is the cause wrapped
// inside a Parse-flavoured VerifyError when verification fails on a
// malformed frame.
type FrameError struct {
	Kind    FrameErrorKind
	Message string
	// Role is "length-prefix" or "body" when Kind == UnexpectedEof.
	Role string
	// Length carries the offending byte length when
	// Kind == FrameTooLarge.
	Length uint32
}

func (e *FrameError) Error() string {
	return fmt.Sprintf("sb: %s: %s", e.Kind, e.Message)
}

func newFrameError(kind FrameErrorKind, msg string) *FrameError {
	return &FrameError{Kind: kind, Message: msg}
}

// TraceHeader is the parsed body of frame 0. Unknown sibling fields
// in the header round-trip via Extra so v1 readers can tolerate
// forward-compatible additions.
type TraceHeader struct {
	Type                    string                 `json:"type"`
	FormatVersion           int                    `json:"format_version"`
	RecorderVersion         string                 `json:"recorder_version"`
	CanonicalisationVersion string                 `json:"canonicalisation_version"`
	PriceListVersion        string                 `json:"price_list_version"`
	PublicKey               string                 `json:"public_key"`
	HmacKeyID               string                 `json:"hmac_key_id"`
	Meta                    map[string]interface{} `json:"meta,omitempty"`
	// Extra captures any additional top-level header fields not
	// covered by the named ones above. Populated by parseHeader.
	Extra map[string]json.RawMessage `json:"-"`
}

// Frame is one wrapper as it appears on disk after the 4-byte length
// prefix is consumed.
//
// BodyBytes is the verbatim canonical-JSON byte slice of the body
// field as it appeared inside the wrapper. The verifier hashes these
// bytes directly rather than re-canonicalising Body, because a Go
// `json.Unmarshal` round-trip would lose precision on the int64
// fields routinely carried in step bodies (`wallclock_ns`, `cpu_ns`).
type Frame struct {
	// Body is the parsed JSON value of the wrapper's `body` field
	// (object, array, or scalar). Useful for high-level inspection.
	Body interface{}
	// BodyBytes is the raw canonical-JSON byte slice of the body,
	// suitable for HMAC verification.
	BodyBytes []byte
	// PrevHMAC is the hex-encoded 32-byte digest the writer claimed
	// chained from the previous frame.
	PrevHMAC string
	// HMAC is the hex-encoded 32-byte digest the writer claimed for
	// this frame's body.
	HMAC string
	// Sig is the per-frame Ed25519 signature in `ed25519:<hex>` form.
	Sig string
}

// Reader streams v1 frames from an io.Reader. It does no crypto
// verification. Use Verify or VerifyPath for end-to-end checking.
type Reader struct {
	r        io.Reader
	finished bool
}

// NewReader returns a Reader that reads framed wrappers from r.
func NewReader(r io.Reader) *Reader {
	return &Reader{r: r}
}

// Next returns the next frame. When the underlying stream is
// exhausted at a frame boundary it returns (nil, io.EOF). A truncated
// frame (EOF mid-frame) returns a *FrameError with Kind UnexpectedEof.
func (r *Reader) Next() (*Frame, error) {
	if r.finished {
		return nil, io.EOF
	}
	var lenBuf [FrameLengthPrefix]byte
	n, err := io.ReadFull(r.r, lenBuf[:])
	if err == io.EOF && n == 0 {
		r.finished = true
		return nil, io.EOF
	}
	if err == io.ErrUnexpectedEOF || (err == io.EOF && n != 0) {
		r.finished = true
		fe := newFrameError(FrameErrorUnexpectedEof,
			"unexpected end of input while reading frame length-prefix")
		fe.Role = "length-prefix"
		return nil, fe
	}
	if err != nil {
		return nil, err
	}
	length := uint32(lenBuf[0])<<24 | uint32(lenBuf[1])<<16 |
		uint32(lenBuf[2])<<8 | uint32(lenBuf[3])
	if length > MaxFrameBytes {
		fe := newFrameError(FrameErrorFrameTooLarge,
			fmt.Sprintf("frame length %d exceeds MaxFrameBytes=%d", length, MaxFrameBytes))
		fe.Length = length
		return nil, fe
	}
	body := make([]byte, int(length))
	_, err = io.ReadFull(r.r, body)
	if err == io.ErrUnexpectedEOF || err == io.EOF {
		r.finished = true
		fe := newFrameError(FrameErrorUnexpectedEof,
			"unexpected end of input while reading frame body")
		fe.Role = "body"
		return nil, fe
	}
	if err != nil {
		return nil, err
	}
	return parseWrapper(body)
}

// IterFrames parses an in-memory `.sb` byte slice, returning each
// frame in order. It is a convenience wrapper around Reader for
// callers that already have the full trace in memory.
func IterFrames(buf []byte) ([]*Frame, error) {
	frames := make([]*Frame, 0, 8)
	offset := 0
	for offset < len(buf) {
		if len(buf)-offset < FrameLengthPrefix {
			fe := newFrameError(FrameErrorUnexpectedEof,
				"unexpected end of input while reading frame length-prefix")
			fe.Role = "length-prefix"
			return frames, fe
		}
		length := uint32(buf[offset])<<24 | uint32(buf[offset+1])<<16 |
			uint32(buf[offset+2])<<8 | uint32(buf[offset+3])
		offset += FrameLengthPrefix
		if length > MaxFrameBytes {
			fe := newFrameError(FrameErrorFrameTooLarge,
				fmt.Sprintf("frame length %d exceeds MaxFrameBytes=%d", length, MaxFrameBytes))
			fe.Length = length
			return frames, fe
		}
		if uint32(len(buf)-offset) < length {
			fe := newFrameError(FrameErrorUnexpectedEof,
				"unexpected end of input while reading frame body")
			fe.Role = "body"
			return frames, fe
		}
		// Slice without copy is fine because the caller owns buf for
		// the duration of the returned frames.
		body := buf[offset : offset+int(length)]
		offset += int(length)
		f, err := parseWrapper(body)
		if err != nil {
			return frames, err
		}
		frames = append(frames, f)
	}
	return frames, nil
}

// parseWrapper consumes a single canonical-JSON wrapper byte slice
// (the bytes between two length prefixes) and returns the parsed
// Frame, including the verbatim body byte slice.
func parseWrapper(wrapper []byte) (*Frame, error) {
	// Parse the wrapper as a generic map first so we can validate
	// shape and extract the four required fields. We will then
	// re-scan the bytes to recover the verbatim body slice.
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(wrapper, &raw); err != nil {
		return nil, newFrameError(FrameErrorBadJson,
			"frame wrapper was not valid JSON: "+err.Error())
	}
	bodyRaw, ok := raw["body"]
	if !ok {
		return nil, newFrameError(FrameErrorBadWrapperShape,
			"frame wrapper missing required field 'body'")
	}
	prevRaw, ok := raw["prev_hmac"]
	if !ok {
		return nil, newFrameError(FrameErrorBadWrapperShape,
			"frame wrapper missing required field 'prev_hmac'")
	}
	hmacRaw, ok := raw["hmac"]
	if !ok {
		return nil, newFrameError(FrameErrorBadWrapperShape,
			"frame wrapper missing required field 'hmac'")
	}
	sigRaw, ok := raw["sig"]
	if !ok {
		return nil, newFrameError(FrameErrorBadWrapperShape,
			"frame wrapper missing required field 'sig'")
	}
	var prevHmac, hmacStr, sig string
	if err := json.Unmarshal(prevRaw, &prevHmac); err != nil {
		return nil, newFrameError(FrameErrorBadWrapperShape,
			"frame wrapper field 'prev_hmac' is not a string")
	}
	if err := json.Unmarshal(hmacRaw, &hmacStr); err != nil {
		return nil, newFrameError(FrameErrorBadWrapperShape,
			"frame wrapper field 'hmac' is not a string")
	}
	if err := json.Unmarshal(sigRaw, &sig); err != nil {
		return nil, newFrameError(FrameErrorBadWrapperShape,
			"frame wrapper field 'sig' is not a string")
	}
	bodyBytes, err := extractBodyBytes(wrapper)
	if err != nil {
		return nil, err
	}
	// Sanity check: the byte-scan body slice must be parseable JSON
	// and equal to the value json.Unmarshal already produced for
	// raw["body"].
	if !bytesEqual(bodyBytes, []byte(bodyRaw)) {
		return nil, newFrameError(FrameErrorBadWrapperShape,
			"recovered body byte slice does not match wrapper body field")
	}
	var bodyVal interface{}
	if err := json.Unmarshal(bodyBytes, &bodyVal); err != nil {
		return nil, newFrameError(FrameErrorBadJson,
			"frame body was not valid JSON: "+err.Error())
	}
	return &Frame{
		Body:      bodyVal,
		BodyBytes: bodyBytes,
		PrevHMAC:  prevHmac,
		HMAC:      hmacStr,
		Sig:       sig,
	}, nil
}

// wrapperBodyPrefix is the constant-folded byte sequence that every
// canonical wrapper begins with: `{"body":`. Because canonical JSON
// sorts keys lexicographically, "body" is always the first key.
var wrapperBodyPrefix = []byte{
	0x7b, 0x22, 0x62, 0x6f, 0x64, 0x79, 0x22, 0x3a,
}

// extractBodyBytes returns the raw bytes of the body JSON value from
// a canonical wrapper. The wrapper byte form is exactly what the
// Python writer fed into HMAC, so reusing this slice guarantees
// byte-for-byte equivalence regardless of language-level number
// precision.
func extractBodyBytes(wrapper []byte) ([]byte, error) {
	if len(wrapper) < len(wrapperBodyPrefix) {
		return nil, newFrameError(FrameErrorBadWrapperShape,
			"wrapper too short to contain body")
	}
	for i := range wrapperBodyPrefix {
		if wrapper[i] != wrapperBodyPrefix[i] {
			return nil, newFrameError(FrameErrorBadWrapperShape,
				`wrapper does not start with canonical {"body": prefix`)
		}
	}
	start := len(wrapperBodyPrefix)
	end, err := scanJSONValueEnd(wrapper, start)
	if err != nil {
		return nil, err
	}
	return wrapper[start:end], nil
}

// scanJSONValueEnd returns the byte offset just past the end of the
// JSON value that starts at start within buf. Handles objects,
// arrays, strings, numbers, true, false, and null. Assumes canonical
// JSON (no whitespace), matching what the Python writer produces.
func scanJSONValueEnd(buf []byte, start int) (int, error) {
	if start >= len(buf) {
		return 0, newFrameError(FrameErrorBadWrapperShape, "empty body value in wrapper")
	}
	c0 := buf[start]
	switch c0 {
	case '{', '[':
		return scanContainerEnd(buf, start)
	case '"':
		return scanStringEnd(buf, start)
	}
	// number / true / false / null — scan until a structural char.
	for i := start; i < len(buf); i++ {
		c := buf[i]
		if c == ',' || c == '}' || c == ']' {
			return i, nil
		}
	}
	return len(buf), nil
}

func scanContainerEnd(buf []byte, start int) (int, error) {
	depth := 0
	inString := false
	escape := false
	for i := start; i < len(buf); i++ {
		c := buf[i]
		if escape {
			escape = false
			continue
		}
		if inString {
			switch c {
			case '\\':
				escape = true
			case '"':
				inString = false
			}
			continue
		}
		switch c {
		case '"':
			inString = true
		case '{', '[':
			depth++
		case '}', ']':
			depth--
			if depth == 0 {
				return i + 1, nil
			}
		}
	}
	return 0, newFrameError(FrameErrorBadWrapperShape, "unterminated container in body")
}

func scanStringEnd(buf []byte, start int) (int, error) {
	escape := false
	for i := start + 1; i < len(buf); i++ {
		c := buf[i]
		if escape {
			escape = false
			continue
		}
		if c == '\\' {
			escape = true
			continue
		}
		if c == '"' {
			return i + 1, nil
		}
	}
	return 0, newFrameError(FrameErrorBadWrapperShape, "unterminated string in body")
}

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// errBodyNotObject signals that the body field of the header frame
// did not decode to a JSON object.
var errBodyNotObject = errors.New("body is not a JSON object")

// parseHeader decodes the body of frame 0 into a TraceHeader.
// Unknown top-level keys round-trip via TraceHeader.Extra.
func parseHeader(bodyBytes []byte) (*TraceHeader, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(bodyBytes, &raw); err != nil {
		return nil, errBodyNotObject
	}
	var hdr TraceHeader
	if err := json.Unmarshal(bodyBytes, &hdr); err != nil {
		return nil, err
	}
	known := map[string]struct{}{
		"type":                     {},
		"format_version":           {},
		"recorder_version":         {},
		"canonicalisation_version": {},
		"price_list_version":       {},
		"public_key":               {},
		"hmac_key_id":              {},
		"meta":                     {},
	}
	for k, v := range raw {
		if _, ok := known[k]; ok {
			continue
		}
		if hdr.Extra == nil {
			hdr.Extra = make(map[string]json.RawMessage, len(raw))
		}
		hdr.Extra[k] = v
	}
	return &hdr, nil
}
