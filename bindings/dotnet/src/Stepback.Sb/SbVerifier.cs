using System.IO;
using System.Security.Cryptography;
using System.Text;
using Org.BouncyCastle.Crypto.Parameters;
using Org.BouncyCastle.Crypto.Signers;

namespace Stepback.Sb;

/// <summary>
/// End-to-end verifier for SB-Trace v1 <c>.sb</c> streams.
/// </summary>
/// <remarks>
/// <para>On success the caller may trust that:</para>
/// <list type="bullet">
///   <item>every frame's HMAC is valid given the chain;</item>
///   <item>every frame's Ed25519 signature was issued by the holder of
///         the private key whose public counterpart is pinned in the
///         header;</item>
///   <item>no frame was inserted, dropped, reordered, or rewritten.</item>
/// </list>
/// </remarks>
public static class SbVerifier
{
    /// <summary>32 zero bytes — the seed of the HMAC chain.</summary>
    public static readonly byte[] ZeroHmac = new byte[32];

    /// <summary>Byte length of an Ed25519 signature.</summary>
    public const int SignatureLength = 64;

    /// <summary>Verify an in-memory <c>.sb</c> trace.</summary>
    public static VerifiedTrace Verify(byte[] buf, byte[] hmacKey)
    {
        if (buf is null) throw new ArgumentNullException(nameof(buf));
        if (hmacKey is null) throw new ArgumentNullException(nameof(hmacKey));
        List<Frame> frames;
        try
        {
            frames = SbTraceReader.IterFrames(buf);
        }
        catch (FrameException fe)
        {
            throw new VerifyException(VerifyErrorKind.Parse, fe.Message,
                frameIndex: -1, innerException: fe);
        }
        return VerifyFrames(frames, hmacKey);
    }

    /// <summary>Verify a <c>.sb</c> trace from disk.</summary>
    public static VerifiedTrace VerifyPath(string path, byte[] hmacKey)
    {
        if (path is null) throw new ArgumentNullException(nameof(path));
        return Verify(File.ReadAllBytes(path), hmacKey);
    }

    /// <summary>Verify a <c>.sb</c> trace from a stream.</summary>
    public static VerifiedTrace VerifyStream(Stream input, byte[] hmacKey)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (hmacKey is null) throw new ArgumentNullException(nameof(hmacKey));
        var frames = new List<Frame>(8);
        using var r = new SbTraceReader(input, leaveOpen: true);
        while (true)
        {
            Frame? f;
            try { f = r.Next(); }
            catch (FrameException fe)
            {
                throw new VerifyException(VerifyErrorKind.Parse, fe.Message,
                    frameIndex: frames.Count, innerException: fe);
            }
            if (f is null) break;
            frames.Add(f);
        }
        return VerifyFrames(frames, hmacKey);
    }

    private static VerifiedTrace VerifyFrames(List<Frame> frames, byte[] hmacKey)
    {
        if (frames.Count == 0)
        {
            throw new VerifyException(VerifyErrorKind.MissingHeader,
                "trace has no header frame (empty input)");
        }
        TraceHeader? header = null;
        Ed25519PublicKeyParameters? publicKey = null;
        byte[] expectedPrev = (byte[])ZeroHmac.Clone();

        for (int index = 0; index < frames.Count; index++)
        {
            Frame frame = frames[index];
            byte[] computedHmac = VerifyChainLink(index, frame, expectedPrev, hmacKey);
            if (index == 0)
            {
                header = ParseHeader(frame.BodyBytes);
                if (header.FormatVersion != Constants.FormatVersion)
                {
                    throw new VerifyException(
                        VerifyErrorKind.UnsupportedFormatVersion,
                        $"unsupported format_version {header.FormatVersion}, expected {Constants.FormatVersion}");
                }
                publicKey = ParseEd25519PublicKey(header.PublicKeyHex);
            }
            if (publicKey is null)
            {
                throw new VerifyException(VerifyErrorKind.BadPublicKey,
                    "no public key resolved from header");
            }
            VerifySignature(index, frame, publicKey);
            expectedPrev = computedHmac;
        }
        return new VerifiedTrace(header!, frames.Count);
    }

    private static byte[] VerifyChainLink(int index, Frame frame,
                                          byte[] expectedPrev, byte[] hmacKey)
    {
        byte[] claimedPrev = DecodeHexFixed(index, "prev_hmac", frame.PrevHmac, 32);
        if (!CryptographicOperations.FixedTimeEquals(claimedPrev, expectedPrev))
        {
            throw new VerifyException(VerifyErrorKind.BrokenChain,
                "prev_hmac does not chain to previous frame", frameIndex: index);
        }
        // Use the verbatim wrapper byte slice for the body. Re-canonicalising
        // would lose precision on int64 fields (wallclock_ns, cpu_ns) and
        // produce a different lexical form.
        byte[] computed;
        using (var mac = new HMACSHA256(hmacKey))
        {
            mac.TransformBlock(expectedPrev, 0, expectedPrev.Length, null, 0);
            mac.TransformFinalBlock(frame.BodyBytes, 0, frame.BodyBytes.Length);
            computed = mac.Hash!;
        }
        byte[] claimedHmac = DecodeHexFixed(index, "hmac", frame.Hmac, 32);
        if (!CryptographicOperations.FixedTimeEquals(claimedHmac, computed))
        {
            throw new VerifyException(VerifyErrorKind.HmacMismatch,
                "HMAC mismatch", frameIndex: index);
        }
        return computed;
    }

    private static void VerifySignature(int index, Frame frame,
                                        Ed25519PublicKeyParameters publicKey)
    {
        string sigStr = frame.Sig;
        int colon = sigStr.IndexOf(':');
        string scheme = colon >= 0 ? sigStr.Substring(0, colon) : "";
        string hexPart = colon >= 0 ? sigStr.Substring(colon + 1) : sigStr;
        if (scheme != "ed25519")
        {
            throw new VerifyException(VerifyErrorKind.UnsupportedSignature,
                $"signature scheme \"{scheme}\" is not supported",
                frameIndex: index);
        }
        byte[] sigBytes;
        try { sigBytes = Convert.FromHexString(hexPart); }
        catch (FormatException fe)
        {
            throw new VerifyException(VerifyErrorKind.BadHex,
                "invalid hex in sig", frameIndex: index, field: "sig",
                innerException: fe);
        }
        if (sigBytes.Length != SignatureLength)
        {
            throw new VerifyException(VerifyErrorKind.BadSignatureLength,
                $"signature length {sigBytes.Length} is not {SignatureLength}",
                frameIndex: index);
        }
        // The Python writer signs the raw 32-byte HMAC digest, not its hex form.
        byte[] hmacBytes;
        try { hmacBytes = Convert.FromHexString(frame.Hmac); }
        catch (FormatException fe)
        {
            throw new VerifyException(VerifyErrorKind.BadHex,
                "invalid hex in hmac", frameIndex: index, field: "hmac",
                innerException: fe);
        }
        var verifier = new Ed25519Signer();
        verifier.Init(false, publicKey);
        verifier.BlockUpdate(hmacBytes, 0, hmacBytes.Length);
        if (!verifier.VerifySignature(sigBytes))
        {
            throw new VerifyException(VerifyErrorKind.SignatureMismatch,
                "Ed25519 signature did not verify", frameIndex: index);
        }
    }

    private static Ed25519PublicKeyParameters ParseEd25519PublicKey(string hexStr)
    {
        byte[] raw;
        try { raw = Convert.FromHexString(hexStr); }
        catch (FormatException fe)
        {
            throw new VerifyException(VerifyErrorKind.BadPublicKey,
                "trace header public_key is not valid hex",
                innerException: fe);
        }
        if (raw.Length != 32)
        {
            throw new VerifyException(VerifyErrorKind.BadPublicKey,
                $"ed25519 public key must be 32 bytes, got {raw.Length}");
        }
        return new Ed25519PublicKeyParameters(raw, 0);
    }

    // --- header parsing (canonical-JSON name lookup) -----------------------

    internal static TraceHeader ParseHeader(byte[] body)
    {
        if (body.Length == 0 || body[0] != (byte)'{')
        {
            throw new VerifyException(VerifyErrorKind.MissingHeader,
                "frame 0 must be a header object");
        }
        string type = ReadString(body, "type");
        if (type != "header")
        {
            throw new VerifyException(VerifyErrorKind.MissingHeader,
                $"frame 0 must be a header object (type=\"{type}\")");
        }
        return new TraceHeader(
            type,
            ReadInt(body, "format_version"),
            ReadString(body, "recorder_version"),
            ReadString(body, "canonicalisation_version"),
            ReadString(body, "price_list_version"),
            ReadString(body, "public_key"),
            ReadString(body, "hmac_key_id"));
    }

    private static string ReadString(byte[] buf, string name)
    {
        byte[] needle = Encoding.ASCII.GetBytes("\"" + name + "\":\"");
        for (int i = 0; i + needle.Length <= buf.Length; i++)
        {
            bool match = true;
            for (int j = 0; j < needle.Length; j++)
            {
                if (buf[i + j] != needle[j]) { match = false; break; }
            }
            if (!match) continue;
            byte prev = i > 0 ? buf[i - 1] : (byte)'{';
            if (prev != (byte)',' && prev != (byte)'{') continue;
            int valStart = i + needle.Length;
            for (int k = valStart; k < buf.Length; k++)
            {
                if (buf[k] == (byte)'\\') { k++; continue; }
                if (buf[k] == (byte)'"')
                {
                    return Encoding.UTF8.GetString(buf, valStart, k - valStart);
                }
            }
        }
        throw new VerifyException(VerifyErrorKind.MissingHeader,
            $"header missing string field '{name}'");
    }

    private static int ReadInt(byte[] buf, string name)
    {
        byte[] needle = Encoding.ASCII.GetBytes("\"" + name + "\":");
        for (int i = 0; i + needle.Length <= buf.Length; i++)
        {
            bool match = true;
            for (int j = 0; j < needle.Length; j++)
            {
                if (buf[i + j] != needle[j]) { match = false; break; }
            }
            if (!match) continue;
            byte prev = i > 0 ? buf[i - 1] : (byte)'{';
            if (prev != (byte)',' && prev != (byte)'{') continue;
            int valStart = i + needle.Length;
            int end = valStart;
            while (end < buf.Length)
            {
                byte c = buf[end];
                if (c == (byte)',' || c == (byte)'}') break;
                end++;
            }
            string raw = Encoding.ASCII.GetString(buf, valStart, end - valStart);
            if (!int.TryParse(raw, out int parsed))
            {
                throw new VerifyException(
                    VerifyErrorKind.UnsupportedFormatVersion,
                    $"header field '{name}' is not an integer");
            }
            return parsed;
        }
        throw new VerifyException(VerifyErrorKind.MissingHeader,
            $"header missing integer field '{name}'");
    }

    // --- helpers -----------------------------------------------------------

    private static byte[] DecodeHexFixed(int index, string field, string s, int expected)
    {
        byte[] bytes;
        try { bytes = Convert.FromHexString(s); }
        catch (FormatException fe)
        {
            throw new VerifyException(VerifyErrorKind.BadHex,
                "invalid hex in " + field, frameIndex: index, field: field,
                innerException: fe);
        }
        if (bytes.Length != expected)
        {
            throw new VerifyException(VerifyErrorKind.BadHex,
                $"{field} expected {expected} bytes, got {bytes.Length}",
                frameIndex: index, field: field);
        }
        return bytes;
    }

    /// <summary>Compute SHA-256 digest. Used by tests that want to assert fixture identity.</summary>
    public static byte[] Sha256(byte[] data) => SHA256.HashData(data);

    /// <summary>Hex-encode a byte array (lowercase, no separators).</summary>
    public static string HexEncode(byte[] data) => Convert.ToHexString(data).ToLowerInvariant();

    /// <summary>Hex-decode a string (case-insensitive). Mirrors <c>Verifier.hexDecode</c> on JVM.</summary>
    public static byte[] HexDecode(string s) => Convert.FromHexString(s);
}
