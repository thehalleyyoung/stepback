package dev.stepback.sb;

/**
 * Thrown by {@link FrameReader} when a frame cannot be parsed.
 * The {@link #kind()} discriminator is part of the cross-language
 * conformance surface; callers may match on it programmatically.
 */
public final class FrameError extends RuntimeException {
    private static final long serialVersionUID = 1L;

    public enum Kind {
        UnexpectedEof,
        FrameTooLarge,
        BadJson,
        BadWrapperShape
    }

    private final Kind kind;
    private final String role;
    private final long length;

    public FrameError(Kind kind, String message) {
        this(kind, message, null, -1);
    }

    public FrameError(Kind kind, String message, String role, long length) {
        super("sb: " + kind + ": " + message);
        this.kind = kind;
        this.role = role;
        this.length = length;
    }

    public Kind kind() { return kind; }
    /** "length-prefix" or "body" when {@link #kind()} == UnexpectedEof. */
    public String role() { return role; }
    /** Offending byte length when {@link #kind()} == FrameTooLarge; -1 otherwise. */
    public long length() { return length; }
}
