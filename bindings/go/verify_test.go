package sb

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
)

// fixturesRoot returns the absolute path of the shared fixture
// corpus shipped with stepback-core. Tests in every binding read
// from the same place so divergence shows up immediately.
func fixturesRoot(t *testing.T) string {
	t.Helper()
	// bindings/go -> ../.. -> stepback root -> stepback-core/fixtures/v1
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	return filepath.Join(wd, "..", "..", "stepback-core", "fixtures", "v1")
}

type manifestEntry struct {
	Name                    string `json:"name"`
	Expected                string `json:"expected"`
	ExpectedFrameCountMin   int    `json:"expected_frame_count_min,omitempty"`
	ExpectedErrorKind       string `json:"expected_error_kind,omitempty"`
	SHA256                  string `json:"sha256"`
	SizeBytes               int    `json:"size_bytes"`
	Note                    string `json:"note,omitempty"`
}

type manifest struct {
	FormatVersion           int             `json:"format_version"`
	CanonicalisationVersion string          `json:"canonicalisation_version"`
	PriceListVersion        string          `json:"price_list_version"`
	HmacKeyHex              string          `json:"hmac_key_hex"`
	PublicKeyHex            string          `json:"public_key_hex"`
	Good                    []manifestEntry `json:"good"`
	Corrupt                 []manifestEntry `json:"corrupt"`
}

func loadManifest(t *testing.T) (*manifest, []byte) {
	t.Helper()
	root := fixturesRoot(t)
	raw, err := os.ReadFile(filepath.Join(root, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var m manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	key, err := hex.DecodeString(m.HmacKeyHex)
	if err != nil {
		t.Fatal(err)
	}
	return &m, key
}

func TestManifestVersionsMatch(t *testing.T) {
	m, _ := loadManifest(t)
	if m.FormatVersion != FormatVersion {
		t.Fatalf("manifest format_version = %d, want %d",
			m.FormatVersion, FormatVersion)
	}
	if m.CanonicalisationVersion != CanonicalisationVersion {
		t.Fatalf("manifest canonicalisation_version = %q, want %q",
			m.CanonicalisationVersion, CanonicalisationVersion)
	}
}

func TestZeroHMACIs32ZeroBytes(t *testing.T) {
	if len(ZeroHMAC) != 32 {
		t.Fatalf("len(ZeroHMAC) = %d, want 32", len(ZeroHMAC))
	}
	for i, b := range ZeroHMAC {
		if b != 0 {
			t.Fatalf("ZeroHMAC[%d] = %d, want 0", i, b)
		}
	}
}

func TestGoodFixturesVerify(t *testing.T) {
	m, key := loadManifest(t)
	root := fixturesRoot(t)
	for _, fx := range m.Good {
		fx := fx
		t.Run(fx.Name, func(t *testing.T) {
			path := filepath.Join(root, "good", fx.Name)
			v, err := VerifyPath(path, key)
			if err != nil {
				t.Fatalf("VerifyPath(%s) error = %v", fx.Name, err)
			}
			if v.Header.FormatVersion != FormatVersion {
				t.Fatalf("header format_version = %d, want %d",
					v.Header.FormatVersion, FormatVersion)
			}
			if v.Header.CanonicalisationVersion != CanonicalisationVersion {
				t.Fatalf("header canonicalisation_version = %q, want %q",
					v.Header.CanonicalisationVersion, CanonicalisationVersion)
			}
			if v.Header.PublicKey != m.PublicKeyHex {
				t.Fatalf("header public_key = %q, want %q",
					v.Header.PublicKey, m.PublicKeyHex)
			}
			if v.Header.PriceListVersion != m.PriceListVersion {
				t.Fatalf("header price_list_version = %q, want %q",
					v.Header.PriceListVersion, m.PriceListVersion)
			}
			if v.FrameCount < fx.ExpectedFrameCountMin {
				t.Fatalf("frame_count %d < expected_min %d",
					v.FrameCount, fx.ExpectedFrameCountMin)
			}
		})
	}
}

func TestGoodMultiStepIterFramesAgreesWithVerify(t *testing.T) {
	m, key := loadManifest(t)
	root := fixturesRoot(t)
	path := filepath.Join(root, "good", "multi_step.sb")
	buf, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	frames, err := IterFrames(buf)
	if err != nil {
		t.Fatalf("IterFrames error = %v", err)
	}
	v, err := Verify(buf, key)
	if err != nil {
		t.Fatalf("Verify error = %v", err)
	}
	if len(frames) != v.FrameCount {
		t.Fatalf("IterFrames count %d != Verify count %d", len(frames), v.FrameCount)
	}
	// Reader must agree too.
	rdr := NewReader(bytes.NewReader(buf))
	n := 0
	for {
		_, err := rdr.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Reader.Next error = %v", err)
		}
		n++
	}
	if n != v.FrameCount {
		t.Fatalf("Reader frame count %d != Verify count %d", n, v.FrameCount)
	}
	_ = m
}

// corruptAccepted defines the acceptance set per corrupt fixture.
// A given mutation can legitimately surface as several different
// VerifyError kinds depending on which check trips first; matching
// any one of them proves the verifier rejected. Mirrors the Rust and
// TypeScript fixture tests.
var corruptAccepted = map[string][]VerifyErrorKind{
	"truncated_body.sb": {VerifyErrorParse},
	"flipped_hmac.sb": {
		VerifyErrorHmacMismatch,
		VerifyErrorBrokenChain,
		VerifyErrorBadHex,
		VerifyErrorSignatureMismatch,
	},
	"flipped_sig.sb": {
		VerifyErrorSignatureMismatch,
		VerifyErrorBadHex,
	},
	"broken_chain.sb": {
		VerifyErrorBrokenChain,
		VerifyErrorHmacMismatch,
		VerifyErrorBadHex,
	},
	"bad_format_version.sb": {
		VerifyErrorHmacMismatch,
		VerifyErrorUnsupportedFormatVersion,
	},
}

func TestCorruptFixturesAreRejected(t *testing.T) {
	m, key := loadManifest(t)
	root := fixturesRoot(t)
	for _, fx := range m.Corrupt {
		fx := fx
		t.Run(fx.Name, func(t *testing.T) {
			path := filepath.Join(root, "corrupt", fx.Name)
			_, err := VerifyPath(path, key)
			if err == nil {
				t.Fatalf("expected %s to reject, got nil error", fx.Name)
			}
			var ve *VerifyError
			if !errors.As(err, &ve) {
				t.Fatalf("expected *VerifyError, got %T: %v", err, err)
			}
			acceptable, ok := corruptAccepted[fx.Name]
			if !ok {
				t.Fatalf("no acceptance set defined for %s", fx.Name)
			}
			for _, k := range acceptable {
				if ve.Kind == k {
					return
				}
			}
			t.Fatalf("kind %q not in accepted set %v for %s",
				ve.Kind, acceptable, fx.Name)
		})
	}
}

func TestVerifyRejectsEmptyInput(t *testing.T) {
	_, key := loadManifest(t)
	_, err := Verify(nil, key)
	if err == nil {
		t.Fatal("expected error on empty input")
	}
	var ve *VerifyError
	if !errors.As(err, &ve) {
		t.Fatalf("expected *VerifyError, got %T", err)
	}
	if ve.Kind != VerifyErrorMissingHeader {
		t.Fatalf("kind = %q, want %q", ve.Kind, VerifyErrorMissingHeader)
	}
}

func TestVerifyRejectsWrongHmacKey(t *testing.T) {
	m, key := loadManifest(t)
	root := fixturesRoot(t)
	buf, err := os.ReadFile(filepath.Join(root, "good", "multi_step.sb"))
	if err != nil {
		t.Fatal(err)
	}
	wrong := make([]byte, len(key))
	_, err = Verify(buf, wrong)
	if err == nil {
		t.Fatal("expected error with wrong HMAC key")
	}
	var ve *VerifyError
	if !errors.As(err, &ve) {
		t.Fatalf("expected *VerifyError, got %T", err)
	}
	if ve.Kind != VerifyErrorHmacMismatch {
		t.Fatalf("kind = %q, want %q", ve.Kind, VerifyErrorHmacMismatch)
	}
	if ve.FrameIndex != 0 {
		t.Fatalf("frame_index = %d, want 0", ve.FrameIndex)
	}
	_ = m
}

func TestIterFramesRejectsOversizedLengthPrefix(t *testing.T) {
	buf := []byte{0xff, 0xff, 0xff, 0xff}
	_, err := IterFrames(buf)
	if err == nil {
		t.Fatal("expected error")
	}
	var fe *FrameError
	if !errors.As(err, &fe) {
		t.Fatalf("expected *FrameError, got %T", err)
	}
	if fe.Kind != FrameErrorFrameTooLarge {
		t.Fatalf("kind = %q, want %q", fe.Kind, FrameErrorFrameTooLarge)
	}
}

func TestIterFramesRejectsTruncatedBody(t *testing.T) {
	body := []byte(`{"body":{},"prev_hmac":"","hmac":"","sig":""}`)
	// Lie about the length: claim 10 more bytes than we have.
	claim := uint32(len(body) + 10)
	buf := make([]byte, 4+len(body))
	binary.BigEndian.PutUint32(buf[:4], claim)
	copy(buf[4:], body)
	_, err := IterFrames(buf)
	if err == nil {
		t.Fatal("expected error")
	}
	var fe *FrameError
	if !errors.As(err, &fe) {
		t.Fatalf("expected *FrameError, got %T", err)
	}
	if fe.Kind != FrameErrorUnexpectedEof {
		t.Fatalf("kind = %q, want %q", fe.Kind, FrameErrorUnexpectedEof)
	}
}

func TestReaderTruncatedLengthPrefix(t *testing.T) {
	// Only 2 bytes — not enough for a length prefix.
	rdr := NewReader(bytes.NewReader([]byte{0x00, 0x01}))
	_, err := rdr.Next()
	if err == nil {
		t.Fatal("expected error")
	}
	var fe *FrameError
	if !errors.As(err, &fe) {
		t.Fatalf("expected *FrameError, got %T", err)
	}
	if fe.Kind != FrameErrorUnexpectedEof {
		t.Fatalf("kind = %q, want %q", fe.Kind, FrameErrorUnexpectedEof)
	}
}

func TestReaderEmptyStreamReturnsEOF(t *testing.T) {
	rdr := NewReader(bytes.NewReader(nil))
	_, err := rdr.Next()
	if !errors.Is(err, io.EOF) {
		t.Fatalf("expected io.EOF, got %v", err)
	}
}

func TestVerifyReaderMatchesVerify(t *testing.T) {
	_, key := loadManifest(t)
	root := fixturesRoot(t)
	buf, err := os.ReadFile(filepath.Join(root, "good", "multi_step.sb"))
	if err != nil {
		t.Fatal(err)
	}
	a, err := Verify(buf, key)
	if err != nil {
		t.Fatal(err)
	}
	b, err := VerifyReader(bytes.NewReader(buf), key)
	if err != nil {
		t.Fatal(err)
	}
	if a.FrameCount != b.FrameCount {
		t.Fatalf("Verify=%d VerifyReader=%d", a.FrameCount, b.FrameCount)
	}
}

func TestFixtureSHA256Matches(t *testing.T) {
	// Belt-and-braces: confirm the fixtures we read are the ones the
	// manifest pinned. Mirrors the Rust and TypeScript checks.
	m, _ := loadManifest(t)
	root := fixturesRoot(t)
	check := func(sub string, entries []manifestEntry) {
		for _, fx := range entries {
			path := filepath.Join(root, sub, fx.Name)
			buf, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("%s: %v", path, err)
			}
			if len(buf) != fx.SizeBytes {
				t.Fatalf("%s size = %d, want %d", fx.Name, len(buf), fx.SizeBytes)
			}
			got := sha256Hex(buf)
			if got != fx.SHA256 {
				t.Fatalf("%s sha256 = %s, want %s", fx.Name, got, fx.SHA256)
			}
		}
	}
	check("good", m.Good)
	check("corrupt", m.Corrupt)
}
