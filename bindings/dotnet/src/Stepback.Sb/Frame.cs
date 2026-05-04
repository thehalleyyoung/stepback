namespace Stepback.Sb;

/// <summary>
/// One <c>.sb</c> v1 wrapper as it appears on disk after the 4-byte
/// length prefix is consumed.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="BodyBytes"/> is the verbatim canonical-JSON byte slice
/// of the wrapper's <c>body</c> field. The verifier hashes these
/// bytes directly rather than re-canonicalising, because a JSON
/// round-trip would lose precision on the int64 fields routinely
/// carried in step bodies (<c>wallclock_ns</c>, <c>cpu_ns</c>).
/// </para>
/// </remarks>
public sealed class Frame
{
    public Frame(byte[] bodyBytes, string prevHmac, string hmac, string sig)
    {
        BodyBytes = bodyBytes;
        PrevHmac = prevHmac;
        Hmac = hmac;
        Sig = sig;
    }

    /// <summary>Verbatim canonical-JSON bytes of the wrapper's <c>body</c> field.</summary>
    public byte[] BodyBytes { get; }

    /// <summary>Hex-encoded 32-byte HMAC the writer claimed for the previous frame.</summary>
    public string PrevHmac { get; }

    /// <summary>Hex-encoded 32-byte HMAC the writer claimed for this frame's body.</summary>
    public string Hmac { get; }

    /// <summary>Per-frame signature in <c>"&lt;scheme&gt;:&lt;hex&gt;"</c> form, e.g. <c>ed25519:...</c>.</summary>
    public string Sig { get; }
}
