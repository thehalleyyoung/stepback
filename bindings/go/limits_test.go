package sb

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// wrapBody builds a length-prefixed wrapper carrying the supplied
// body, suitable for IterFrames. Mirrors the Python and TS test
// helpers.
func wrapBody(t *testing.T, body interface{}) []byte {
	t.Helper()
	wrapper := map[string]interface{}{
		"body":      body,
		"prev_hmac": strings.Repeat("00", 32),
		"hmac":      strings.Repeat("00", 32),
		"sig":       "ed25519:" + strings.Repeat("00", 64),
	}
	// Use the canonical (sorted-keys, no-whitespace) variant from
	// canonical.go so the wrapperBodyPrefix scan succeeds.
	payload, err := canonicalJSON(wrapper)
	if err != nil {
		t.Fatalf("canonicalJSON: %v", err)
	}
	out := make([]byte, 4+len(payload))
	binary.BigEndian.PutUint32(out[:4], uint32(len(payload)))
	copy(out[4:], payload)
	return out
}

// canonicalJSON is a minimal local helper for the test (production
// code uses the wrapper that came off disk verbatim, so it doesn't
// otherwise need a canonicalizer).
func canonicalJSON(v interface{}) ([]byte, error) {
	switch n := v.(type) {
	case map[string]interface{}:
		keys := make([]string, 0, len(n))
		for k := range n {
			keys = append(keys, k)
		}
		// stable sort
		for i := 1; i < len(keys); i++ {
			for j := i; j > 0 && keys[j-1] > keys[j]; j-- {
				keys[j-1], keys[j] = keys[j], keys[j-1]
			}
		}
		out := []byte{'{'}
		for i, k := range keys {
			if i > 0 {
				out = append(out, ',')
			}
			kb, err := json.Marshal(k)
			if err != nil {
				return nil, err
			}
			out = append(out, kb...)
			out = append(out, ':')
			vb, err := canonicalJSON(n[k])
			if err != nil {
				return nil, err
			}
			out = append(out, vb...)
		}
		return append(out, '}'), nil
	case []interface{}:
		out := []byte{'['}
		for i, el := range n {
			if i > 0 {
				out = append(out, ',')
			}
			eb, err := canonicalJSON(el)
			if err != nil {
				return nil, err
			}
			out = append(out, eb...)
		}
		return append(out, ']'), nil
	default:
		return json.Marshal(v)
	}
}

func TestIterFramesRejectsLengthPrefixAboveMaxFrameBytes(t *testing.T) {
	buf := make([]byte, 4)
	binary.BigEndian.PutUint32(buf, uint32(MaxFrameBytes+1))
	_, err := IterFrames(buf)
	var fe *FrameError
	if !errors.As(err, &fe) || fe.Kind != FrameErrorFrameTooLarge {
		t.Fatalf("want FrameTooLarge, got %v", err)
	}
}

func TestIterFramesRejectsDeepObject(t *testing.T) {
	deep := map[string]interface{}{}
	cursor := deep
	for i := 0; i < MaxNestingDepth+5; i++ {
		next := map[string]interface{}{}
		cursor["x"] = next
		cursor = next
	}
	body := map[string]interface{}{"type": "tail", "deep": deep}
	buf := wrapBody(t, body)
	_, err := IterFrames(buf)
	var fe *FrameError
	if !errors.As(err, &fe) || fe.Kind != FrameErrorDepthExceeded {
		t.Fatalf("want DepthExceeded, got %v", err)
	}
}

func TestIterFramesRejectsHugeString(t *testing.T) {
	// 16 MiB + 1 byte string. Wrapper is well under MaxFrameBytes.
	huge := strings.Repeat("x", MaxStringBytes+1)
	body := map[string]interface{}{"type": "tail", "huge": huge}
	buf := wrapBody(t, body)
	_, err := IterFrames(buf)
	var fe *FrameError
	if !errors.As(err, &fe) || fe.Kind != FrameErrorStringTooLarge {
		t.Fatalf("want StringTooLarge, got %v", err)
	}
}

func TestIterFramesLimitsTunable(t *testing.T) {
	deep := map[string]interface{}{}
	cursor := deep
	for i := 0; i < MaxNestingDepth+2; i++ {
		next := map[string]interface{}{}
		cursor["x"] = next
		cursor = next
	}
	body := map[string]interface{}{"type": "tail", "deep": deep}
	buf := wrapBody(t, body)
	if _, err := IterFrames(buf); err == nil {
		t.Fatal("default limits should reject")
	}
	relaxed := DefaultLimits()
	relaxed.MaxDepth = MaxNestingDepth + 1024
	frames, err := IterFramesWithLimits(buf, relaxed)
	if err != nil {
		t.Fatalf("relaxed limits should accept, got %v", err)
	}
	if len(frames) != 1 {
		t.Fatalf("want 1 frame, got %d", len(frames))
	}
}
