package dev.stepback.sb;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Streaming v1 frame splitter. Performs no crypto; use
 * {@link Verifier} for end-to-end verification.
 *
 * <p>Each call to {@link #next()} reads one length-prefixed wrapper
 * from the underlying stream and extracts the four required string
 * fields ({@code body}, {@code prev_hmac}, {@code hmac}, {@code sig}).
 * The body is returned as the verbatim byte slice that appeared in
 * the wrapper, which is what the writer fed into HMAC.
 */
public final class FrameReader implements AutoCloseable {
    private final InputStream in;
    private boolean finished;

    public FrameReader(InputStream in) {
        this.in = in;
    }

    /**
     * Read and return the next frame, or {@code null} on clean EOF.
     * @throws FrameError on truncation or malformed wrapper.
     * @throws IOException on underlying stream errors.
     */
    public Frame next() throws IOException {
        if (finished) return null;
        byte[] lenBuf = new byte[Constants.FRAME_LENGTH_PREFIX];
        int read = readFully(in, lenBuf, 0, lenBuf.length);
        if (read == 0) {
            finished = true;
            return null;
        }
        if (read != lenBuf.length) {
            finished = true;
            throw new FrameError(
                FrameError.Kind.UnexpectedEof,
                "unexpected end of input while reading frame length-prefix",
                "length-prefix", -1);
        }
        long length = ((lenBuf[0] & 0xffL) << 24)
                    | ((lenBuf[1] & 0xffL) << 16)
                    | ((lenBuf[2] & 0xffL) << 8)
                    |  (lenBuf[3] & 0xffL);
        if (length > Constants.MAX_FRAME_BYTES) {
            throw new FrameError(
                FrameError.Kind.FrameTooLarge,
                "frame length " + length + " exceeds MAX_FRAME_BYTES="
                    + Constants.MAX_FRAME_BYTES,
                null, length);
        }
        byte[] wrapper = new byte[(int) length];
        int got = readFully(in, wrapper, 0, wrapper.length);
        if (got != wrapper.length) {
            finished = true;
            throw new FrameError(
                FrameError.Kind.UnexpectedEof,
                "unexpected end of input while reading frame body",
                "body", -1);
        }
        return parseWrapper(wrapper);
    }

    @Override
    public void close() throws IOException {
        in.close();
    }

    /** Slurp a full {@code .sb} byte array into a list of frames. */
    public static List<Frame> iterFrames(byte[] buf) {
        List<Frame> out = new ArrayList<>(8);
        int offset = 0;
        while (offset < buf.length) {
            if (buf.length - offset < Constants.FRAME_LENGTH_PREFIX) {
                throw new FrameError(
                    FrameError.Kind.UnexpectedEof,
                    "unexpected end of input while reading frame length-prefix",
                    "length-prefix", -1);
            }
            long length = ((buf[offset] & 0xffL) << 24)
                        | ((buf[offset + 1] & 0xffL) << 16)
                        | ((buf[offset + 2] & 0xffL) << 8)
                        |  (buf[offset + 3] & 0xffL);
            offset += Constants.FRAME_LENGTH_PREFIX;
            if (length > Constants.MAX_FRAME_BYTES) {
                throw new FrameError(
                    FrameError.Kind.FrameTooLarge,
                    "frame length " + length + " exceeds MAX_FRAME_BYTES="
                        + Constants.MAX_FRAME_BYTES,
                    null, length);
            }
            if (buf.length - offset < length) {
                throw new FrameError(
                    FrameError.Kind.UnexpectedEof,
                    "unexpected end of input while reading frame body",
                    "body", -1);
            }
            byte[] wrapper = new byte[(int) length];
            System.arraycopy(buf, offset, wrapper, 0, wrapper.length);
            offset += wrapper.length;
            out.add(parseWrapper(wrapper));
        }
        return out;
    }

    // --- wrapper parsing ---------------------------------------------------

    /** Constant prefix every canonical wrapper begins with: {@code {"body":}. */
    private static final byte[] BODY_PREFIX = {
        '{', '"', 'b', 'o', 'd', 'y', '"', ':'
    };

    static Frame parseWrapper(byte[] wrapper) {
        if (wrapper.length < BODY_PREFIX.length) {
            throw new FrameError(FrameError.Kind.BadWrapperShape,
                "wrapper too short to contain body");
        }
        for (int i = 0; i < BODY_PREFIX.length; i++) {
            if (wrapper[i] != BODY_PREFIX[i]) {
                throw new FrameError(FrameError.Kind.BadWrapperShape,
                    "wrapper does not start with canonical {\"body\": prefix");
            }
        }
        int bodyStart = BODY_PREFIX.length;
        int bodyEnd = scanJsonValueEnd(wrapper, bodyStart);
        byte[] bodyBytes = new byte[bodyEnd - bodyStart];
        System.arraycopy(wrapper, bodyStart, bodyBytes, 0, bodyBytes.length);
        // After body, the canonical wrapper contains:
        //   ,"hmac":"...","prev_hmac":"...","sig":"..."}
        // We extract each by name lookup; canonical-JSON sorting puts
        // keys in alphabetical order (body < hmac < prev_hmac < sig)
        // but we don't rely on positional parsing here.
        String hmac = readStringField(wrapper, "hmac", bodyEnd);
        String prevHmac = readStringField(wrapper, "prev_hmac", bodyEnd);
        String sig = readStringField(wrapper, "sig", bodyEnd);
        return new Frame(bodyBytes, prevHmac, hmac, sig);
    }

    /**
     * Returns the offset just past the end of the JSON value that
     * begins at {@code start}. Assumes canonical JSON (no whitespace).
     */
    private static int scanJsonValueEnd(byte[] buf, int start) {
        if (start >= buf.length) {
            throw new FrameError(FrameError.Kind.BadWrapperShape,
                "empty body value in wrapper");
        }
        byte c = buf[start];
        if (c == '{' || c == '[') return scanContainerEnd(buf, start);
        if (c == '"') return scanStringEnd(buf, start);
        for (int i = start; i < buf.length; i++) {
            byte b = buf[i];
            if (b == ',' || b == '}' || b == ']') return i;
        }
        return buf.length;
    }

    private static int scanContainerEnd(byte[] buf, int start) {
        int depth = 0;
        boolean inString = false;
        boolean escape = false;
        for (int i = start; i < buf.length; i++) {
            byte c = buf[i];
            if (escape) { escape = false; continue; }
            if (inString) {
                if (c == '\\') escape = true;
                else if (c == '"') inString = false;
                continue;
            }
            if (c == '"') inString = true;
            else if (c == '{' || c == '[') depth++;
            else if (c == '}' || c == ']') {
                depth--;
                if (depth == 0) return i + 1;
            }
        }
        throw new FrameError(FrameError.Kind.BadWrapperShape,
            "unterminated container in body");
    }

    private static int scanStringEnd(byte[] buf, int start) {
        boolean escape = false;
        for (int i = start + 1; i < buf.length; i++) {
            byte c = buf[i];
            if (escape) { escape = false; continue; }
            if (c == '\\') { escape = true; continue; }
            if (c == '"') return i + 1;
        }
        throw new FrameError(FrameError.Kind.BadWrapperShape,
            "unterminated string in body");
    }

    /**
     * Locate {@code "name":"<value>"} starting at or after {@code from}
     * and return the value with JSON escape sequences resolved.
     * Hex string values used in wrappers contain no escapes.
     */
    private static String readStringField(byte[] buf, String name, int from) {
        byte[] needle = ("\"" + name + "\":\"").getBytes();
        outer:
        for (int i = from; i + needle.length <= buf.length; i++) {
            for (int j = 0; j < needle.length; j++) {
                if (buf[i + j] != needle[j]) continue outer;
            }
            // Make sure preceding char is `,` or `{` so we don't match
            // a substring inside the body (e.g. "shmac":"..."). The
            // wrapper fields always sit at the top level.
            byte prev = buf[i - 1];
            if (prev != ',' && prev != '{') continue;
            int valStart = i + needle.length;
            int end = scanStringEnd(buf, valStart - 1); // re-use string scanner
            // valStart-1 points at the opening '"'. scanStringEnd returns
            // index just past the closing '"'.
            return new String(buf, valStart, end - valStart - 1);
        }
        throw new FrameError(FrameError.Kind.BadWrapperShape,
            "frame wrapper missing required field '" + name + "'");
    }

    private static int readFully(InputStream in, byte[] buf, int off, int len)
            throws IOException {
        int total = 0;
        while (total < len) {
            int n = in.read(buf, off + total, len - total);
            if (n < 0) break;
            total += n;
        }
        return total;
    }
}
