package dev.stepback.sb;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.io.ByteArrayInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Conformance tests against the shared SB-Trace v1 fixture corpus
 * shipped under {@code stepback-core/fixtures/v1/}. These tests run
 * iff the {@code stepback.fixtures} system property is set (the
 * Gradle build wires it up automatically).
 */
@EnabledIfSystemProperty(named = "stepback.fixtures", matches = ".+")
final class VerifierTest {

    private static Path fixturesRoot() {
        return Paths.get(System.getProperty("stepback.fixtures"));
    }

    private static byte[] hmacKey() throws Exception {
        Map<String, String> m = parseManifest();
        return Verifier.hexDecode(m.get("hmac_key_hex"));
    }

    @Test
    void manifestVersionsMatch() throws Exception {
        Map<String, String> m = parseManifest();
        assertEquals(String.valueOf(Constants.FORMAT_VERSION),
                     m.get("format_version"));
        assertEquals(Constants.CANONICALISATION_VERSION,
                     m.get("canonicalisation_version"));
    }

    @Test
    void zeroHmacIs32ZeroBytes() {
        assertEquals(32, Verifier.ZERO_HMAC.length);
        for (byte b : Verifier.ZERO_HMAC) assertEquals(0, b);
    }

    // --- good fixtures: must verify cleanly --------------------------------

    @Test
    void verifyHeaderOnly() throws Exception {
        verifyGood("header_only.sb", 2);
    }

    @Test
    void verifyMultiStep() throws Exception {
        verifyGood("multi_step.sb", 5);
    }

    @Test
    void verifyWithBlobs() throws Exception {
        verifyGood("with_blobs.sb", 5);
    }

    private void verifyGood(String name, int minFrames) throws Exception {
        Path p = fixturesRoot().resolve("good").resolve(name);
        byte[] data = Files.readAllBytes(p);
        VerifiedTrace v = Verifier.verify(data, hmacKey());
        assertNotNull(v.header());
        assertEquals(Constants.FORMAT_VERSION, v.header().formatVersion());
        assertTrue(v.frameCount() >= minFrames,
            name + " expected at least " + minFrames + " frames, got "
                + v.frameCount());
        // Streaming path must agree with the in-memory path.
        VerifiedTrace stream = Verifier.verifyStream(
            new ByteArrayInputStream(data), hmacKey());
        assertEquals(v.frameCount(), stream.frameCount());
    }

    // --- corrupt fixtures: must reject -------------------------------------

    @Test
    void rejectTruncatedBody() throws Exception {
        Path p = fixturesRoot().resolve("corrupt").resolve("truncated_body.sb");
        VerifyError ve = assertThrows(VerifyError.class,
            () -> Verifier.verify(Files.readAllBytes(p), hmacKey()));
        assertEquals(VerifyError.Kind.Parse, ve.kind());
    }

    @Test
    void rejectFlippedHmac() throws Exception {
        Path p = fixturesRoot().resolve("corrupt").resolve("flipped_hmac.sb");
        VerifyError ve = assertThrows(VerifyError.class,
            () -> Verifier.verify(Files.readAllBytes(p), hmacKey()));
        // Manifest says "BadHexOrHmacMismatch" — the HMAC body covers
        // both prev_hmac and current hmac, so a flipped nibble shows
        // up as a chain or HMAC mismatch depending on which half was
        // mutated.
        assertTrue(ve.kind() == VerifyError.Kind.HmacMismatch
                || ve.kind() == VerifyError.Kind.BrokenChain
                || ve.kind() == VerifyError.Kind.BadHex,
            "unexpected kind: " + ve.kind());
    }

    @Test
    void rejectFlippedSig() throws Exception {
        Path p = fixturesRoot().resolve("corrupt").resolve("flipped_sig.sb");
        VerifyError ve = assertThrows(VerifyError.class,
            () -> Verifier.verify(Files.readAllBytes(p), hmacKey()));
        assertEquals(VerifyError.Kind.SignatureMismatch, ve.kind());
    }

    @Test
    void rejectBrokenChain() throws Exception {
        Path p = fixturesRoot().resolve("corrupt").resolve("broken_chain.sb");
        VerifyError ve = assertThrows(VerifyError.class,
            () -> Verifier.verify(Files.readAllBytes(p), hmacKey()));
        assertTrue(ve.kind() == VerifyError.Kind.BrokenChain
                || ve.kind() == VerifyError.Kind.HmacMismatch,
            "unexpected kind: " + ve.kind());
    }

    @Test
    void rejectBadFormatVersion() throws Exception {
        Path p = fixturesRoot().resolve("corrupt").resolve("bad_format_version.sb");
        VerifyError ve = assertThrows(VerifyError.class,
            () -> Verifier.verify(Files.readAllBytes(p), hmacKey()));
        // Forging format_version mutates the body. Either the version
        // check (if HMAC happens to still align) or HMAC fails first.
        assertTrue(ve.kind() == VerifyError.Kind.UnsupportedFormatVersion
                || ve.kind() == VerifyError.Kind.HmacMismatch,
            "unexpected kind: " + ve.kind());
    }

    // --- empty / malformed inputs ------------------------------------------

    @Test
    void rejectEmpty() throws Exception {
        VerifyError ve = assertThrows(VerifyError.class,
            () -> Verifier.verify(new byte[0], hmacKey()));
        assertEquals(VerifyError.Kind.MissingHeader, ve.kind());
    }

    @Test
    void rejectWrongHmacKey() throws Exception {
        Path p = fixturesRoot().resolve("good").resolve("multi_step.sb");
        byte[] wrong = new byte[32];
        for (int i = 0; i < 32; i++) wrong[i] = (byte) 0xaa;
        VerifyError ve = assertThrows(VerifyError.class,
            () -> Verifier.verify(Files.readAllBytes(p), wrong));
        assertEquals(VerifyError.Kind.HmacMismatch, ve.kind());
        assertEquals(0, ve.frameIndex());
    }

    @Test
    void rejectOversizedLengthPrefix() {
        byte[] buf = new byte[]{ 0x7f, (byte) 0xff, (byte) 0xff, (byte) 0xff };
        FrameError fe = assertThrows(FrameError.class,
            () -> FrameReader.iterFrames(buf));
        assertEquals(FrameError.Kind.FrameTooLarge, fe.kind());
    }

    @Test
    void rejectTruncatedLengthPrefix() {
        FrameError fe = assertThrows(FrameError.class,
            () -> FrameReader.iterFrames(new byte[]{ 0x00, 0x00 }));
        assertEquals(FrameError.Kind.UnexpectedEof, fe.kind());
        assertEquals("length-prefix", fe.role());
    }

    @Test
    void fixtureSha256sMatchManifest() throws Exception {
        Map<String, String> good = parseFixtureChecksums("good");
        Map<String, String> corrupt = parseFixtureChecksums("corrupt");
        for (Map.Entry<String, String> e : good.entrySet()) {
            byte[] data = Files.readAllBytes(
                fixturesRoot().resolve("good").resolve(e.getKey()));
            assertEquals(e.getValue(), Verifier.hexEncode(Verifier.sha256(data)),
                "sha256 drift on good/" + e.getKey());
        }
        for (Map.Entry<String, String> e : corrupt.entrySet()) {
            byte[] data = Files.readAllBytes(
                fixturesRoot().resolve("corrupt").resolve(e.getKey()));
            assertEquals(e.getValue(), Verifier.hexEncode(Verifier.sha256(data)),
                "sha256 drift on corrupt/" + e.getKey());
        }
    }

    // --- minimal manifest.json reader (no JSON dep) ------------------------
    // The manifest is tiny and has a stable shape; we extract the few
    // top-level scalars we need with regex rather than depending on a
    // JSON library.

    private static Map<String, String> parseManifest() throws Exception {
        String src = Files.readString(fixturesRoot().resolve("manifest.json"));
        Map<String, String> out = new LinkedHashMap<>();
        for (String key : new String[]{
            "format_version", "canonicalisation_version", "price_list_version",
            "hmac_key_hex", "public_key_hex"
        }) {
            Pattern strPat = Pattern.compile(
                "\"" + Pattern.quote(key) + "\"\\s*:\\s*\"([^\"]+)\"");
            Matcher m = strPat.matcher(src);
            if (m.find()) { out.put(key, m.group(1)); continue; }
            Pattern intPat = Pattern.compile(
                "\"" + Pattern.quote(key) + "\"\\s*:\\s*([0-9]+)");
            m = intPat.matcher(src);
            if (m.find()) { out.put(key, m.group(1)); continue; }
            throw new AssertionError("manifest missing key: " + key);
        }
        return out;
    }

    /** Returns name -> sha256 for entries under the given top-level array key. */
    private static Map<String, String> parseFixtureChecksums(String section)
            throws Exception {
        String src = Files.readString(fixturesRoot().resolve("manifest.json"));
        // Find the "section": [ ... ] block.
        Pattern arrayPat = Pattern.compile(
            "\"" + Pattern.quote(section) + "\"\\s*:\\s*\\[(.*?)\\](?=\\s*[,}])",
            Pattern.DOTALL);
        Matcher am = arrayPat.matcher(src);
        if (!am.find()) throw new AssertionError("no array: " + section);
        String body = am.group(1);
        Map<String, String> out = new LinkedHashMap<>();
        Pattern entryPat = Pattern.compile(
            "\\{[^{}]*?\"name\"\\s*:\\s*\"([^\"]+)\"[^{}]*?\"sha256\"\\s*:\\s*\"([0-9a-f]+)\"[^{}]*?\\}",
            Pattern.DOTALL);
        Matcher em = entryPat.matcher(body);
        while (em.find()) {
            out.put(em.group(1), em.group(2));
        }
        // Manifest also allows sha256 to appear before name; try the
        // reverse order if we missed any.
        Pattern entryRev = Pattern.compile(
            "\\{[^{}]*?\"sha256\"\\s*:\\s*\"([0-9a-f]+)\"[^{}]*?\"name\"\\s*:\\s*\"([^\"]+)\"[^{}]*?\\}",
            Pattern.DOTALL);
        Matcher er = entryRev.matcher(body);
        while (er.find()) {
            out.putIfAbsent(er.group(2), er.group(1));
        }
        return out;
    }
}
