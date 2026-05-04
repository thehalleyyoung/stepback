namespace Stepback.Sb;

/// <summary>
/// Stable, machine-readable discriminator for the ways frame parsing
/// can fail. The same set is used by the Rust, Go, JVM, and
/// TypeScript references.
/// </summary>
public enum FrameErrorKind
{
    UnexpectedEof,
    FrameTooLarge,
    BadJson,
    BadWrapperShape
}

/// <summary>
/// Thrown by <see cref="SbTraceReader"/> when a frame cannot be
/// parsed. The <see cref="Kind"/> discriminator is part of the
/// cross-language conformance surface; callers may match on it
/// programmatically.
/// </summary>
public sealed class FrameException : Exception
{
    public FrameException(FrameErrorKind kind, string message,
                          string? role = null, long length = -1,
                          Exception? innerException = null)
        : base($"sb: {kind}: {message}", innerException)
    {
        Kind = kind;
        Role = role;
        Length = length;
    }

    public FrameErrorKind Kind { get; }

    /// <summary><c>"length-prefix"</c> or <c>"body"</c> when <see cref="Kind"/> == UnexpectedEof.</summary>
    public string? Role { get; }

    /// <summary>Offending byte length when <see cref="Kind"/> == FrameTooLarge; -1 otherwise.</summary>
    public long Length { get; }
}
