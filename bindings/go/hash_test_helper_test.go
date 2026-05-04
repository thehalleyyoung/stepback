package sb

import (
	"crypto/sha256"
	"encoding/hex"
)

// sha256Hex returns the lowercase hex digest of buf. Test helper —
// the verifier itself does not expose a hashing API because hashing
// is the writer's job.
func sha256Hex(buf []byte) string {
	sum := sha256.Sum256(buf)
	return hex.EncodeToString(sum[:])
}
