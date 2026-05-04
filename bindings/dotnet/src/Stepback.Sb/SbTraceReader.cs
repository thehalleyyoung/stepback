using System.Buffers.Binary;
using System.IO;

namespace Stepback.Sb;

/// <summary>
/// Streaming v1 frame splitter. Performs no crypto; use
/// <see cref="SbVerifier"/> for end-to-end verification.
/// </summary>
/// <remarks>
/// <para>
/// Each call to <see cref="Next"/> reads one length-prefixed wrapper
/// from the underlying stream and extracts the four required string
/// fields (<c>body</c>, <c>prev_hmac</c>, <c>hmac</c>, <c>sig</c>).
/// The body is returned as the verbatim byte slice that appeared in
/// the wrapper, which is what the writer fed into HMAC.
/// </para>
/// </remarks>
public sealed class SbTraceReader : IDisposable
{
    private readonly Stream _stream;
    private readonly bool _leaveOpen;
    private bool _finished;
    private bool _disposed;

    public SbTraceReader(Stream stream, bool leaveOpen = false)
    {
        _stream = stream ?? throw new ArgumentNullException(nameof(stream));
        _leaveOpen = leaveOpen;
    }

    /// <summary>
    /// Read and return the next frame, or <c>null</c> on clean EOF.
    /// </summary>
    /// <exception cref="FrameException">Thrown on truncation or malformed wrapper.</exception>
    /// <exception cref="IOException">Thrown on underlying stream errors.</exception>
    public Frame? Next()
    {
        if (_finished) return null;
        var lenBuf = new byte[Constants.FrameLengthPrefix];
        int read = ReadFully(_stream, lenBuf, 0, lenBuf.Length);
        if (read == 0)
        {
            _finished = true;
            return null;
        }
        if (read != lenBuf.Length)
        {
            _finished = true;
            throw new FrameException(
                FrameErrorKind.UnexpectedEof,
                "unexpected end of input while reading frame length-prefix",
                role: "length-prefix");
        }
        long length = BinaryPrimitives.ReadUInt32BigEndian(lenBuf);
        if (length > Constants.MaxFrameBytes)
        {
            throw new FrameException(
                FrameErrorKind.FrameTooLarge,
                $"frame length {length} exceeds MaxFrameBytes={Constants.MaxFrameBytes}",
                length: length);
        }
        var wrapper = new byte[(int)length];
        int got = ReadFully(_stream, wrapper, 0, wrapper.Length);
        if (got != wrapper.Length)
        {
            _finished = true;
            throw new FrameException(
                FrameErrorKind.UnexpectedEof,
                "unexpected end of input while reading frame body",
                role: "body");
        }
        return ParseWrapper(wrapper);
    }

    /// <summary>
    /// Slurp a full <c>.sb</c> byte array into a list of frames.
    /// </summary>
    public static List<Frame> IterFrames(byte[] buf)
    {
        if (buf is null) throw new ArgumentNullException(nameof(buf));
        var frames = new List<Frame>(8);
        int offset = 0;
        while (offset < buf.Length)
        {
            if (buf.Length - offset < Constants.FrameLengthPrefix)
            {
                throw new FrameException(
                    FrameErrorKind.UnexpectedEof,
                    "unexpected end of input while reading frame length-prefix",
                    role: "length-prefix");
            }
            long length = BinaryPrimitives.ReadUInt32BigEndian(
                new ReadOnlySpan<byte>(buf, offset, Constants.FrameLengthPrefix));
            offset += Constants.FrameLengthPrefix;
            if (length > Constants.MaxFrameBytes)
            {
                throw new FrameException(
                    FrameErrorKind.FrameTooLarge,
                    $"frame length {length} exceeds MaxFrameBytes={Constants.MaxFrameBytes}",
                    length: length);
            }
            if (buf.Length - offset < length)
            {
                throw new FrameException(
                    FrameErrorKind.UnexpectedEof,
                    "unexpected end of input while reading frame body",
                    role: "body");
            }
            var wrapper = new byte[(int)length];
            Buffer.BlockCopy(buf, offset, wrapper, 0, wrapper.Length);
            offset += wrapper.Length;
            frames.Add(ParseWrapper(wrapper));
        }
        return frames;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (!_leaveOpen) _stream.Dispose();
    }

    // --- wrapper parsing ---------------------------------------------------

    /// <summary>Constant prefix every canonical wrapper begins with: <c>{"body":</c>.</summary>
    private static readonly byte[] BodyPrefix = "{\"body\":"u8.ToArray();

    internal static Frame ParseWrapper(byte[] wrapper)
    {
        if (wrapper.Length < BodyPrefix.Length)
        {
            throw new FrameException(FrameErrorKind.BadWrapperShape,
                "wrapper too short to contain body");
        }
        for (int i = 0; i < BodyPrefix.Length; i++)
        {
            if (wrapper[i] != BodyPrefix[i])
            {
                throw new FrameException(FrameErrorKind.BadWrapperShape,
                    "wrapper does not start with canonical {\"body\": prefix");
            }
        }
        int bodyStart = BodyPrefix.Length;
        int bodyEnd = ScanJsonValueEnd(wrapper, bodyStart);
        var bodyBytes = new byte[bodyEnd - bodyStart];
        Buffer.BlockCopy(wrapper, bodyStart, bodyBytes, 0, bodyBytes.Length);
        // After body, the canonical wrapper contains:
        //   ,"hmac":"...","prev_hmac":"...","sig":"..."}
        // We extract each by name lookup; canonical-JSON sorting puts
        // keys in alphabetical order (body < hmac < prev_hmac < sig)
        // but we don't rely on positional parsing here.
        string hmac = ReadStringField(wrapper, "hmac", bodyEnd);
        string prevHmac = ReadStringField(wrapper, "prev_hmac", bodyEnd);
        string sig = ReadStringField(wrapper, "sig", bodyEnd);
        return new Frame(bodyBytes, prevHmac, hmac, sig);
    }

    /// <summary>
    /// Returns the offset just past the end of the JSON value that
    /// begins at <paramref name="start"/>. Assumes canonical JSON
    /// (no whitespace).
    /// </summary>
    private static int ScanJsonValueEnd(byte[] buf, int start)
    {
        if (start >= buf.Length)
        {
            throw new FrameException(FrameErrorKind.BadWrapperShape,
                "empty body value in wrapper");
        }
        byte c = buf[start];
        if (c == (byte)'{' || c == (byte)'[') return ScanContainerEnd(buf, start);
        if (c == (byte)'"') return ScanStringEnd(buf, start);
        for (int i = start; i < buf.Length; i++)
        {
            byte b = buf[i];
            if (b == (byte)',' || b == (byte)'}' || b == (byte)']') return i;
        }
        return buf.Length;
    }

    private static int ScanContainerEnd(byte[] buf, int start)
    {
        int depth = 0;
        bool inString = false;
        bool escape = false;
        for (int i = start; i < buf.Length; i++)
        {
            byte c = buf[i];
            if (escape) { escape = false; continue; }
            if (inString)
            {
                if (c == (byte)'\\') escape = true;
                else if (c == (byte)'"') inString = false;
                continue;
            }
            if (c == (byte)'"') inString = true;
            else if (c == (byte)'{' || c == (byte)'[') depth++;
            else if (c == (byte)'}' || c == (byte)']')
            {
                depth--;
                if (depth == 0) return i + 1;
            }
        }
        throw new FrameException(FrameErrorKind.BadWrapperShape,
            "unterminated container in body");
    }

    private static int ScanStringEnd(byte[] buf, int start)
    {
        bool escape = false;
        for (int i = start + 1; i < buf.Length; i++)
        {
            byte c = buf[i];
            if (escape) { escape = false; continue; }
            if (c == (byte)'\\') { escape = true; continue; }
            if (c == (byte)'"') return i + 1;
        }
        throw new FrameException(FrameErrorKind.BadWrapperShape,
            "unterminated string in body");
    }

    /// <summary>
    /// Locate <c>"name":"&lt;value&gt;"</c> starting at or after
    /// <paramref name="from"/> and return the value as-is. The
    /// wrapper field values are hex strings without escapes.
    /// </summary>
    private static string ReadStringField(byte[] buf, string name, int from)
    {
        byte[] needle = System.Text.Encoding.ASCII.GetBytes("\"" + name + "\":\"");
        for (int i = from; i + needle.Length <= buf.Length; i++)
        {
            bool match = true;
            for (int j = 0; j < needle.Length; j++)
            {
                if (buf[i + j] != needle[j]) { match = false; break; }
            }
            if (!match) continue;
            // Make sure preceding char is `,` or `{` so we don't match
            // a substring inside the body. The wrapper fields always
            // sit at the top level.
            byte prev = i > 0 ? buf[i - 1] : (byte)'{';
            if (prev != (byte)',' && prev != (byte)'{') continue;
            int valStart = i + needle.Length;
            // valStart-1 points at the opening '"'. ScanStringEnd
            // returns the index just past the closing '"'.
            int end = ScanStringEnd(buf, valStart - 1);
            return System.Text.Encoding.UTF8.GetString(buf, valStart, end - valStart - 1);
        }
        throw new FrameException(FrameErrorKind.BadWrapperShape,
            $"frame wrapper missing required field '{name}'");
    }

    private static int ReadFully(Stream s, byte[] buf, int off, int len)
    {
        int total = 0;
        while (total < len)
        {
            int n = s.Read(buf, off + total, len - total);
            if (n <= 0) break;
            total += n;
        }
        return total;
    }
}
