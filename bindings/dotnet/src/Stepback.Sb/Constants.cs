namespace Stepback.Sb;

/// <summary>
/// Pinned constants that mirror the Python writer, the Rust
/// <c>sb-format</c> crate, the Go <c>sb</c> package, the JVM
/// <c>dev.stepback.sb</c> package, and the TypeScript
/// <c>@stepback/core</c> reader.
/// </summary>
public static class Constants
{
    /// <summary>Wire-format major version this reader understands.</summary>
    public const int FormatVersion = 1;

    /// <summary>Width in bytes of the big-endian length prefix preceding every frame.</summary>
    public const int FrameLengthPrefix = 4;

    /// <summary>
    /// Defensive cap on a single decoded frame, in bytes. Large enough
    /// for any realistic agent step, small enough that a corrupted
    /// length prefix cannot trigger a multi-gigabyte allocation.
    /// </summary>
    public const int MaxFrameBytes = 64 * 1024 * 1024;

    /// <summary>Canonicalisation version this reader understands.</summary>
    public const string CanonicalisationVersion = "1";

    /// <summary>This binding's own release version. Independent of the wire format.</summary>
    public const string Version = "0.1.0";
}
