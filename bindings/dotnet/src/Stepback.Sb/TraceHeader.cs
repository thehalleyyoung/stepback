namespace Stepback.Sb;

/// <summary>
/// Parsed body of frame 0. Unknown sibling fields are ignored; this
/// is a v1 reader and v1 only.
/// </summary>
public sealed class TraceHeader
{
    public TraceHeader(
        string type,
        int formatVersion,
        string recorderVersion,
        string canonicalisationVersion,
        string priceListVersion,
        string publicKeyHex,
        string hmacKeyId)
    {
        Type = type;
        FormatVersion = formatVersion;
        RecorderVersion = recorderVersion;
        CanonicalisationVersion = canonicalisationVersion;
        PriceListVersion = priceListVersion;
        PublicKeyHex = publicKeyHex;
        HmacKeyId = hmacKeyId;
    }

    public string Type { get; }
    public int FormatVersion { get; }
    public string RecorderVersion { get; }
    public string CanonicalisationVersion { get; }
    public string PriceListVersion { get; }
    public string PublicKeyHex { get; }
    public string HmacKeyId { get; }
}
