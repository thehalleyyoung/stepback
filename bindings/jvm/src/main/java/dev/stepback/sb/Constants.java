package dev.stepback.sb;

/**
 * Pinned constants that mirror the Python writer, the Rust
 * {@code sb-format} crate, the Go {@code sb} package, and the
 * TypeScript {@code @stepback/core} reader.
 */
public final class Constants {
    private Constants() {}

    /** Wire-format major version this reader understands. */
    public static final int FORMAT_VERSION = 1;

    /** Width in bytes of the big-endian length prefix preceding every frame. */
    public static final int FRAME_LENGTH_PREFIX = 4;

    /**
     * Defensive cap on a single decoded frame, in bytes. Large enough
     * for any realistic agent step, small enough that a corrupted
     * length prefix cannot trigger a multi-gigabyte allocation.
     */
    public static final int MAX_FRAME_BYTES = 64 * 1024 * 1024;

    /** Canonicalisation version this reader understands. */
    public static final String CANONICALISATION_VERSION = "1";

    /** This binding's own release version. Independent of the wire format. */
    public static final String VERSION = "0.1.0";
}
