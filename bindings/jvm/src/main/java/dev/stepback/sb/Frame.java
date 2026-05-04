package dev.stepback.sb;

/**
 * One {@code .sb} v1 wrapper as it appears on disk after the 4-byte
 * length prefix is consumed.
 *
 * <p>{@link #bodyBytes()} is the verbatim canonical-JSON byte slice
 * of the wrapper's {@code body} field. The verifier hashes these
 * bytes directly rather than re-canonicalising, because a JSON
 * round-trip would lose precision on the int64 fields routinely
 * carried in step bodies ({@code wallclock_ns}, {@code cpu_ns}).
 */
public final class Frame {
    private final byte[] bodyBytes;
    private final String prevHmac;
    private final String hmac;
    private final String sig;

    public Frame(byte[] bodyBytes, String prevHmac, String hmac, String sig) {
        this.bodyBytes = bodyBytes;
        this.prevHmac = prevHmac;
        this.hmac = hmac;
        this.sig = sig;
    }

    /** Verbatim canonical-JSON bytes of the wrapper's {@code body} field. */
    public byte[] bodyBytes() { return bodyBytes; }

    /** Hex-encoded 32-byte HMAC the writer claimed for the previous frame. */
    public String prevHmac() { return prevHmac; }

    /** Hex-encoded 32-byte HMAC the writer claimed for this frame's body. */
    public String hmac() { return hmac; }

    /** Per-frame signature in {@code "<scheme>:<hex>"} form, e.g. {@code "ed25519:..."}. */
    public String sig() { return sig; }
}
