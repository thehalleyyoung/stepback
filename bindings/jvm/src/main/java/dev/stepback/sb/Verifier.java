package dev.stepback.sb;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.GeneralSecurityException;
import java.security.KeyFactory;
import java.security.MessageDigest;
import java.security.PublicKey;
import java.security.Signature;
import java.security.spec.X509EncodedKeySpec;
import java.util.List;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;

/**
 * End-to-end verifier for SB-Trace v1 {@code .sb} streams.
 *
 * <p>On success the caller may trust that:
 * <ul>
 *   <li>every frame's HMAC is valid given the chain;</li>
 *   <li>every frame's Ed25519 signature was issued by the holder of
 *       the private key whose public counterpart is pinned in the
 *       header;</li>
 *   <li>no frame was inserted, dropped, reordered, or rewritten.</li>
 * </ul>
 */
public final class Verifier {
    /** 32 zero bytes — the seed of the HMAC chain. */
    public static final byte[] ZERO_HMAC = new byte[32];

    /** Byte length of an Ed25519 signature. */
    public static final int SIGNATURE_LENGTH = 64;

    private Verifier() {}

    /** Verify an in-memory {@code .sb} trace. */
    public static VerifiedTrace verify(byte[] buf, byte[] hmacKey) {
        List<Frame> frames;
        try {
            frames = FrameReader.iterFrames(buf);
        } catch (FrameError fe) {
            throw new VerifyError(VerifyError.Kind.Parse,
                fe.getMessage(), -1, null, fe);
        }
        return verifyFrames(frames, hmacKey);
    }

    /** Verify a {@code .sb} trace from disk. */
    public static VerifiedTrace verifyPath(Path path, byte[] hmacKey)
            throws IOException {
        return verify(Files.readAllBytes(path), hmacKey);
    }

    /** Verify a {@code .sb} trace from a stream. */
    public static VerifiedTrace verifyStream(InputStream in, byte[] hmacKey)
            throws IOException {
        try (FrameReader r = new FrameReader(in)) {
            java.util.ArrayList<Frame> frames = new java.util.ArrayList<>(8);
            while (true) {
                Frame f;
                try {
                    f = r.next();
                } catch (FrameError fe) {
                    throw new VerifyError(VerifyError.Kind.Parse,
                        fe.getMessage(), frames.size(), null, fe);
                }
                if (f == null) break;
                frames.add(f);
            }
            return verifyFrames(frames, hmacKey);
        }
    }

    private static VerifiedTrace verifyFrames(List<Frame> frames, byte[] hmacKey) {
        if (frames.isEmpty()) {
            throw new VerifyError(VerifyError.Kind.MissingHeader,
                "trace has no header frame (empty input)", -1);
        }
        TraceHeader header = null;
        PublicKey publicKey = null;
        byte[] expectedPrev = ZERO_HMAC.clone();

        for (int index = 0; index < frames.size(); index++) {
            Frame frame = frames.get(index);
            byte[] computedHmac = verifyChainLink(index, frame, expectedPrev, hmacKey);
            if (index == 0) {
                header = parseHeader(frame.bodyBytes());
                if (header.formatVersion() != Constants.FORMAT_VERSION) {
                    throw new VerifyError(VerifyError.Kind.UnsupportedFormatVersion,
                        "unsupported format_version " + header.formatVersion()
                            + ", expected " + Constants.FORMAT_VERSION, -1);
                }
                publicKey = parseEd25519PublicKey(header.publicKeyHex());
            }
            if (publicKey == null) {
                throw new VerifyError(VerifyError.Kind.BadPublicKey,
                    "no public key resolved from header", -1);
            }
            verifySignature(index, frame, publicKey);
            expectedPrev = computedHmac;
        }
        return new VerifiedTrace(header, frames.size());
    }

    private static byte[] verifyChainLink(int index, Frame frame,
                                          byte[] expectedPrev, byte[] hmacKey) {
        byte[] claimedPrev = decodeHexFixed(index, "prev_hmac",
                                            frame.prevHmac(), 32);
        if (!constantTimeEquals(claimedPrev, expectedPrev)) {
            throw new VerifyError(VerifyError.Kind.BrokenChain,
                "prev_hmac does not chain to previous frame", index);
        }
        byte[] computed;
        try {
            Mac mac = Mac.getInstance("HmacSHA256");
            mac.init(new SecretKeySpec(hmacKey, "HmacSHA256"));
            mac.update(expectedPrev);
            mac.update(frame.bodyBytes());
            computed = mac.doFinal();
        } catch (GeneralSecurityException gse) {
            throw new VerifyError(VerifyError.Kind.HmacMismatch,
                "HMAC primitive failed: " + gse.getMessage(), index, null, gse);
        }
        byte[] claimedHmac = decodeHexFixed(index, "hmac", frame.hmac(), 32);
        if (!constantTimeEquals(claimedHmac, computed)) {
            throw new VerifyError(VerifyError.Kind.HmacMismatch,
                "HMAC mismatch", index);
        }
        return computed;
    }

    private static void verifySignature(int index, Frame frame, PublicKey publicKey) {
        String sigStr = frame.sig();
        int colon = sigStr.indexOf(':');
        String scheme = colon >= 0 ? sigStr.substring(0, colon) : "";
        String hexPart = colon >= 0 ? sigStr.substring(colon + 1) : sigStr;
        if (!"ed25519".equals(scheme)) {
            throw new VerifyError(VerifyError.Kind.UnsupportedSignature,
                "signature scheme \"" + scheme + "\" is not supported", index);
        }
        byte[] sigBytes;
        try {
            sigBytes = hexDecode(hexPart);
        } catch (IllegalArgumentException iae) {
            throw new VerifyError(VerifyError.Kind.BadHex,
                "invalid hex in sig", index, "sig", iae);
        }
        if (sigBytes.length != SIGNATURE_LENGTH) {
            throw new VerifyError(VerifyError.Kind.BadSignatureLength,
                "signature length " + sigBytes.length + " is not "
                    + SIGNATURE_LENGTH, index);
        }
        // The Python writer signs the raw 32-byte HMAC digest, not the
        // hex form. Decode frame.hmac() back to bytes before verifying.
        byte[] hmacBytes;
        try {
            hmacBytes = hexDecode(frame.hmac());
        } catch (IllegalArgumentException iae) {
            throw new VerifyError(VerifyError.Kind.BadHex,
                "invalid hex in hmac", index, "hmac", iae);
        }
        try {
            Signature sig = Signature.getInstance("Ed25519");
            sig.initVerify(publicKey);
            sig.update(hmacBytes);
            if (!sig.verify(sigBytes)) {
                throw new VerifyError(VerifyError.Kind.SignatureMismatch,
                    "Ed25519 signature did not verify", index);
            }
        } catch (GeneralSecurityException gse) {
            throw new VerifyError(VerifyError.Kind.SignatureMismatch,
                "Ed25519 verify failed: " + gse.getMessage(), index, null, gse);
        }
    }

    /**
     * Wrap a raw 32-byte Ed25519 public key in a SubjectPublicKeyInfo
     * envelope so the JDK's {@code KeyFactory} can ingest it without
     * a third-party crypto provider.
     */
    private static PublicKey parseEd25519PublicKey(String hexStr) {
        byte[] raw;
        try {
            raw = hexDecode(hexStr);
        } catch (IllegalArgumentException iae) {
            throw new VerifyError(VerifyError.Kind.BadPublicKey,
                "trace header public_key is not valid hex", -1, null, iae);
        }
        if (raw.length != 32) {
            throw new VerifyError(VerifyError.Kind.BadPublicKey,
                "ed25519 public key must be 32 bytes, got " + raw.length, -1);
        }
        // X.509 SPKI prefix for Ed25519: AlgorithmIdentifier { 1.3.101.112 }
        // followed by 32-byte BIT STRING.
        byte[] spki = new byte[12 + 32];
        byte[] prefix = {
            0x30, 0x2a,                            // SEQUENCE (42 bytes)
            0x30, 0x05,                            // SEQUENCE (5 bytes) – algorithm
            0x06, 0x03, 0x2b, 0x65, 0x70,          // OID 1.3.101.112 (Ed25519)
            0x03, 0x21, 0x00                        // BIT STRING (33 bytes; 0 unused bits)
        };
        System.arraycopy(prefix, 0, spki, 0, prefix.length);
        System.arraycopy(raw, 0, spki, prefix.length, raw.length);
        try {
            KeyFactory kf = KeyFactory.getInstance("Ed25519");
            return kf.generatePublic(new X509EncodedKeySpec(spki));
        } catch (GeneralSecurityException gse) {
            throw new VerifyError(VerifyError.Kind.BadPublicKey,
                "JDK cannot decode Ed25519 public key: " + gse.getMessage(),
                -1, null, gse);
        }
    }

    // --- header parsing (canonical-JSON name lookup) -----------------------

    static TraceHeader parseHeader(byte[] body) {
        if (body.length == 0 || body[0] != '{') {
            throw new VerifyError(VerifyError.Kind.MissingHeader,
                "frame 0 must be a header object", -1);
        }
        String type = readString(body, "type");
        if (!"header".equals(type)) {
            throw new VerifyError(VerifyError.Kind.MissingHeader,
                "frame 0 must be a header object (type=\"" + type + "\")", -1);
        }
        return new TraceHeader(
            type,
            readInt(body, "format_version"),
            readString(body, "recorder_version"),
            readString(body, "canonicalisation_version"),
            readString(body, "price_list_version"),
            readString(body, "public_key"),
            readString(body, "hmac_key_id")
        );
    }

    private static String readString(byte[] buf, String name) {
        byte[] needle = ("\"" + name + "\":\"").getBytes();
        outer:
        for (int i = 0; i + needle.length <= buf.length; i++) {
            for (int j = 0; j < needle.length; j++) {
                if (buf[i + j] != needle[j]) continue outer;
            }
            byte prev = i > 0 ? buf[i - 1] : (byte) '{';
            if (prev != ',' && prev != '{') continue;
            int valStart = i + needle.length;
            for (int k = valStart; k < buf.length; k++) {
                if (buf[k] == '\\') { k++; continue; }
                if (buf[k] == '"') {
                    return new String(buf, valStart, k - valStart);
                }
            }
        }
        throw new VerifyError(VerifyError.Kind.MissingHeader,
            "header missing string field '" + name + "'", -1);
    }

    private static int readInt(byte[] buf, String name) {
        byte[] needle = ("\"" + name + "\":").getBytes();
        outer:
        for (int i = 0; i + needle.length <= buf.length; i++) {
            for (int j = 0; j < needle.length; j++) {
                if (buf[i + j] != needle[j]) continue outer;
            }
            byte prev = i > 0 ? buf[i - 1] : (byte) '{';
            if (prev != ',' && prev != '{') continue;
            int valStart = i + needle.length;
            int end = valStart;
            while (end < buf.length) {
                byte c = buf[end];
                if (c == ',' || c == '}') break;
                end++;
            }
            try {
                return Integer.parseInt(new String(buf, valStart, end - valStart));
            } catch (NumberFormatException nfe) {
                throw new VerifyError(VerifyError.Kind.UnsupportedFormatVersion,
                    "header field '" + name + "' is not an integer", -1, null, nfe);
            }
        }
        throw new VerifyError(VerifyError.Kind.MissingHeader,
            "header missing integer field '" + name + "'", -1);
    }

    // --- helpers -----------------------------------------------------------

    private static byte[] decodeHexFixed(int index, String field, String s, int expected) {
        byte[] bytes;
        try {
            bytes = hexDecode(s);
        } catch (IllegalArgumentException iae) {
            throw new VerifyError(VerifyError.Kind.BadHex,
                "invalid hex in " + field, index, field, iae);
        }
        if (bytes.length != expected) {
            throw new VerifyError(VerifyError.Kind.BadHex,
                field + " expected " + expected + " bytes, got " + bytes.length,
                index, field, null);
        }
        return bytes;
    }

    static byte[] hexDecode(String s) {
        int len = s.length();
        if ((len & 1) != 0) throw new IllegalArgumentException("odd-length hex string");
        byte[] out = new byte[len / 2];
        for (int i = 0; i < len; i += 2) {
            int hi = Character.digit(s.charAt(i), 16);
            int lo = Character.digit(s.charAt(i + 1), 16);
            if (hi < 0 || lo < 0) throw new IllegalArgumentException("bad hex char");
            out[i / 2] = (byte) ((hi << 4) | lo);
        }
        return out;
    }

    /** Constant-time equality on equal-length byte arrays. */
    static boolean constantTimeEquals(byte[] a, byte[] b) {
        if (a.length != b.length) return false;
        int diff = 0;
        for (int i = 0; i < a.length; i++) diff |= (a[i] ^ b[i]) & 0xff;
        return diff == 0;
    }

    /** Compute SHA-256 digest. Used by tests that want to assert fixture identity. */
    public static byte[] sha256(byte[] data) {
        try {
            return MessageDigest.getInstance("SHA-256").digest(data);
        } catch (GeneralSecurityException gse) {
            throw new IllegalStateException(gse);
        }
    }

    /** Hex-encode a byte array (lowercase, no separators). */
    public static String hexEncode(byte[] data) {
        char[] out = new char[data.length * 2];
        for (int i = 0; i < data.length; i++) {
            int v = data[i] & 0xff;
            out[i * 2]     = "0123456789abcdef".charAt(v >>> 4);
            out[i * 2 + 1] = "0123456789abcdef".charAt(v & 0x0f);
        }
        return new String(out);
    }

}
