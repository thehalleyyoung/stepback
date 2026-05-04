package dev.stepback.sb;

/**
 * Thrown by {@link Verifier} when end-to-end verification fails.
 * The {@link #kind()} discriminator is part of the cross-language
 * conformance surface and matches the Rust / Go / TypeScript
 * verifier kinds 1:1.
 */
public final class VerifyError extends RuntimeException {
    private static final long serialVersionUID = 1L;

    public enum Kind {
        Parse,
        MissingHeader,
        UnsupportedFormatVersion,
        BrokenChain,
        HmacMismatch,
        BadHex,
        UnsupportedSignature,
        BadSignatureLength,
        SignatureMismatch,
        BadPublicKey
    }

    private final Kind kind;
    private final int frameIndex;
    private final String field;

    public VerifyError(Kind kind, String message, int frameIndex) {
        this(kind, message, frameIndex, null, null);
    }

    public VerifyError(Kind kind, String message, int frameIndex,
                       String field, Throwable cause) {
        super(formatMessage(kind, message, frameIndex), cause);
        this.kind = kind;
        this.frameIndex = frameIndex;
        this.field = field;
    }

    private static String formatMessage(Kind kind, String message, int frameIndex) {
        if (frameIndex >= 0) {
            return "sb: " + kind + " (frame " + frameIndex + "): " + message;
        }
        return "sb: " + kind + ": " + message;
    }

    public Kind kind() { return kind; }
    /** 0-based frame index, or -1 if the error is not frame-local. */
    public int frameIndex() { return frameIndex; }
    /** Optional: "hmac" / "sig" / "prev_hmac" / etc. */
    public String field() { return field; }
}
