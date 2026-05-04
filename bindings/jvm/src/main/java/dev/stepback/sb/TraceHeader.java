package dev.stepback.sb;

/**
 * Parsed body of frame 0. Unknown sibling fields are ignored; this
 * is a v1 reader and v1 only.
 */
public final class TraceHeader {
    private final String type;
    private final int formatVersion;
    private final String recorderVersion;
    private final String canonicalisationVersion;
    private final String priceListVersion;
    private final String publicKeyHex;
    private final String hmacKeyId;

    public TraceHeader(String type, int formatVersion, String recorderVersion,
                       String canonicalisationVersion, String priceListVersion,
                       String publicKeyHex, String hmacKeyId) {
        this.type = type;
        this.formatVersion = formatVersion;
        this.recorderVersion = recorderVersion;
        this.canonicalisationVersion = canonicalisationVersion;
        this.priceListVersion = priceListVersion;
        this.publicKeyHex = publicKeyHex;
        this.hmacKeyId = hmacKeyId;
    }

    public String type() { return type; }
    public int formatVersion() { return formatVersion; }
    public String recorderVersion() { return recorderVersion; }
    public String canonicalisationVersion() { return canonicalisationVersion; }
    public String priceListVersion() { return priceListVersion; }
    public String publicKeyHex() { return publicKeyHex; }
    public String hmacKeyId() { return hmacKeyId; }
}
