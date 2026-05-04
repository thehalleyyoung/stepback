package dev.stepback.sb;

/**
 * Outcome of a successful verification: the parsed header plus the
 * number of frames that chained cleanly.
 */
public final class VerifiedTrace {
    private final TraceHeader header;
    private final int frameCount;

    public VerifiedTrace(TraceHeader header, int frameCount) {
        this.header = header;
        this.frameCount = frameCount;
    }

    public TraceHeader header() { return header; }
    public int frameCount() { return frameCount; }
}
