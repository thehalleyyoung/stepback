namespace Stepback.Sb;

/// <summary>
/// Outcome of a successful verification: the parsed header plus the
/// number of frames that chained cleanly.
/// </summary>
public sealed class VerifiedTrace
{
    public VerifiedTrace(TraceHeader header, int frameCount)
    {
        Header = header;
        FrameCount = frameCount;
    }

    public TraceHeader Header { get; }
    public int FrameCount { get; }
}
