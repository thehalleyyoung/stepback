using System.Text;
using Xunit;

namespace Stepback.Sb.Tests;

public sealed class FrameReaderTests
{
    [Fact]
    public void ParseSimpleWrapper()
    {
        // Build a canonical wrapper by hand:
        //   {"body":1,"hmac":"AA","prev_hmac":"BB","sig":"ed25519:CC"}
        const string wrapper = "{\"body\":1,\"hmac\":\"AA\",\"prev_hmac\":\"BB\",\"sig\":\"ed25519:CC\"}";
        var f = SbTraceReader.ParseWrapper(Encoding.UTF8.GetBytes(wrapper));
        Assert.Equal("AA", f.Hmac);
        Assert.Equal("BB", f.PrevHmac);
        Assert.Equal("ed25519:CC", f.Sig);
        Assert.Equal("1", Encoding.UTF8.GetString(f.BodyBytes));
    }

    [Fact]
    public void ParseObjectWrapper()
    {
        const string wrapper = "{\"body\":{\"x\":1,\"y\":\"z\"},\"hmac\":\"a\",\"prev_hmac\":\"b\",\"sig\":\"c\"}";
        var f = SbTraceReader.ParseWrapper(Encoding.UTF8.GetBytes(wrapper));
        Assert.Equal("{\"x\":1,\"y\":\"z\"}", Encoding.UTF8.GetString(f.BodyBytes));
    }

    [Fact]
    public void ParseStringBodyWithEscapedQuote()
    {
        const string wrapper = "{\"body\":\"hi \\\"quoted\\\"\",\"hmac\":\"a\",\"prev_hmac\":\"b\",\"sig\":\"c\"}";
        var f = SbTraceReader.ParseWrapper(Encoding.UTF8.GetBytes(wrapper));
        Assert.Equal("\"hi \\\"quoted\\\"\"", Encoding.UTF8.GetString(f.BodyBytes));
    }

    [Fact]
    public void RejectMissingBodyPrefix()
    {
        var ex = Assert.Throws<FrameException>(() =>
            SbTraceReader.ParseWrapper(Encoding.UTF8.GetBytes("{\"hmac\":\"a\"}")));
        Assert.Equal(FrameErrorKind.BadWrapperShape, ex.Kind);
    }

    [Fact]
    public void RejectMissingRequiredField()
    {
        const string wrapper = "{\"body\":1,\"hmac\":\"a\",\"prev_hmac\":\"b\"}";
        var ex = Assert.Throws<FrameException>(() =>
            SbTraceReader.ParseWrapper(Encoding.UTF8.GetBytes(wrapper)));
        Assert.Equal(FrameErrorKind.BadWrapperShape, ex.Kind);
    }

    [Fact]
    public void RejectOversizedLengthPrefix()
    {
        var buf = new byte[] { 0x7f, 0xff, 0xff, 0xff };
        var ex = Assert.Throws<FrameException>(() => SbTraceReader.IterFrames(buf));
        Assert.Equal(FrameErrorKind.FrameTooLarge, ex.Kind);
    }

    [Fact]
    public void RejectTruncatedLengthPrefix()
    {
        var ex = Assert.Throws<FrameException>(() =>
            SbTraceReader.IterFrames(new byte[] { 0x00, 0x00 }));
        Assert.Equal(FrameErrorKind.UnexpectedEof, ex.Kind);
        Assert.Equal("length-prefix", ex.Role);
    }
}
