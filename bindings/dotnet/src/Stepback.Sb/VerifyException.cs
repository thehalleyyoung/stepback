namespace Stepback.Sb;

/// <summary>
/// Stable, machine-readable discriminator for the ways verification
/// can fail. The same set is used by the Rust, Go, JVM, and
/// TypeScript references.
/// </summary>
public enum VerifyErrorKind
{
    Parse,
    MissingHeader,
    UnsupportedFormatVersion,
    BrokenChain,
    HmacMismatch,
    BadHex,
    UnsupportedSignature,
    BadSignatureLength,
    SignatureMismatch,
    BadPublicKey
}

/// <summary>
/// Thrown by <see cref="SbVerifier"/> when end-to-end verification
/// fails. The <see cref="Kind"/> discriminator matches the Rust /
/// Go / JVM / TypeScript verifier kinds 1:1.
/// </summary>
public sealed class VerifyException : Exception
{
    public VerifyException(VerifyErrorKind kind, string message,
                           int frameIndex = -1, string? field = null,
                           Exception? innerException = null)
        : base(FormatMessage(kind, message, frameIndex), innerException)
    {
        Kind = kind;
        FrameIndex = frameIndex;
        Field = field;
    }

    public VerifyErrorKind Kind { get; }

    /// <summary>0-based frame index, or -1 if the error is not frame-local.</summary>
    public int FrameIndex { get; }

    /// <summary>Optional: <c>"hmac"</c> / <c>"sig"</c> / <c>"prev_hmac"</c> / etc.</summary>
    public string? Field { get; }

    private static string FormatMessage(VerifyErrorKind kind, string message, int frameIndex)
    {
        return frameIndex >= 0
            ? $"sb: {kind} (frame {frameIndex}): {message}"
            : $"sb: {kind}: {message}";
    }
}
