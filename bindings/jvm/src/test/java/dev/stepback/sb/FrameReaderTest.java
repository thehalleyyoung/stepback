package dev.stepback.sb;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

final class FrameReaderTest {

    @Test
    void parseSimpleWrapper() {
        // Build a canonical wrapper by hand:
        //   {"body":1,"hmac":"AA","prev_hmac":"BB","sig":"ed25519:CC"}
        String wrapper = "{\"body\":1,\"hmac\":\"AA\",\"prev_hmac\":\"BB\",\"sig\":\"ed25519:CC\"}";
        byte[] body = wrapper.getBytes();
        Frame f = FrameReader.parseWrapper(body);
        assertEquals("AA", f.hmac());
        assertEquals("BB", f.prevHmac());
        assertEquals("ed25519:CC", f.sig());
        assertEquals("1", new String(f.bodyBytes()));
    }

    @Test
    void parseObjectWrapper() {
        String wrapper = "{\"body\":{\"x\":1,\"y\":\"z\"},\"hmac\":\"a\",\"prev_hmac\":\"b\",\"sig\":\"c\"}";
        Frame f = FrameReader.parseWrapper(wrapper.getBytes());
        assertEquals("{\"x\":1,\"y\":\"z\"}", new String(f.bodyBytes()));
    }

    @Test
    void parseStringBodyWithEscapedQuote() {
        String wrapper = "{\"body\":\"hi \\\"quoted\\\"\",\"hmac\":\"a\",\"prev_hmac\":\"b\",\"sig\":\"c\"}";
        Frame f = FrameReader.parseWrapper(wrapper.getBytes());
        assertEquals("\"hi \\\"quoted\\\"\"", new String(f.bodyBytes()));
    }

    @Test
    void rejectMissingBodyPrefix() {
        FrameError fe = assertThrows(FrameError.class,
            () -> FrameReader.parseWrapper("{\"hmac\":\"a\"}".getBytes()));
        assertEquals(FrameError.Kind.BadWrapperShape, fe.kind());
    }

    @Test
    void rejectMissingRequiredField() {
        // Wrapper with body and hmac but no sig.
        String wrapper = "{\"body\":1,\"hmac\":\"a\",\"prev_hmac\":\"b\"}";
        FrameError fe = assertThrows(FrameError.class,
            () -> FrameReader.parseWrapper(wrapper.getBytes()));
        assertEquals(FrameError.Kind.BadWrapperShape, fe.kind());
    }
}
