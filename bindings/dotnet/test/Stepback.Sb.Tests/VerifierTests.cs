using System.IO;
using System.Text.Json;
using Xunit;

namespace Stepback.Sb.Tests;

/// <summary>
/// Conformance tests against the shared SB-Trace v1 fixture corpus
/// shipped under <c>stepback-core/fixtures/v1/</c>. The .csproj
/// copies that corpus into the test bin directory.
/// </summary>
public sealed class VerifierTests
{
    private static string FixturesRoot()
        => Path.Combine(AppContext.BaseDirectory, "fixtures");

    private static byte[] HmacKey()
    {
        var manifest = LoadManifest();
        return Convert.FromHexString(manifest.GetProperty("hmac_key_hex").GetString()!);
    }

    private static JsonElement LoadManifest()
    {
        var text = File.ReadAllText(Path.Combine(FixturesRoot(), "manifest.json"));
        var doc = JsonDocument.Parse(text);
        return doc.RootElement.Clone();
    }

    [Fact]
    public void ManifestVersionsMatch()
    {
        var m = LoadManifest();
        Assert.Equal(Constants.FormatVersion, m.GetProperty("format_version").GetInt32());
        Assert.Equal(Constants.CanonicalisationVersion,
                     m.GetProperty("canonicalisation_version").GetString());
    }

    [Fact]
    public void ZeroHmacIs32ZeroBytes()
    {
        Assert.Equal(32, SbVerifier.ZeroHmac.Length);
        foreach (var b in SbVerifier.ZeroHmac) Assert.Equal(0, b);
    }

    // --- good fixtures: must verify cleanly --------------------------------

    [Theory]
    [InlineData("header_only.sb", 2)]
    [InlineData("multi_step.sb", 5)]
    [InlineData("with_blobs.sb", 5)]
    public void VerifyGood(string name, int minFrames)
    {
        var p = Path.Combine(FixturesRoot(), "good", name);
        var data = File.ReadAllBytes(p);
        var v = SbVerifier.Verify(data, HmacKey());
        Assert.NotNull(v.Header);
        Assert.Equal(Constants.FormatVersion, v.Header.FormatVersion);
        Assert.True(v.FrameCount >= minFrames,
            $"{name} expected at least {minFrames} frames, got {v.FrameCount}");
        // Streaming path must agree with the in-memory path.
        using var ms = new MemoryStream(data);
        var streamed = SbVerifier.VerifyStream(ms, HmacKey());
        Assert.Equal(v.FrameCount, streamed.FrameCount);
    }

    // --- corrupt fixtures: must reject -------------------------------------

    [Fact]
    public void RejectTruncatedBody()
    {
        var p = Path.Combine(FixturesRoot(), "corrupt", "truncated_body.sb");
        var ex = Assert.Throws<VerifyException>(() =>
            SbVerifier.Verify(File.ReadAllBytes(p), HmacKey()));
        Assert.Equal(VerifyErrorKind.Parse, ex.Kind);
    }

    [Fact]
    public void RejectFlippedHmac()
    {
        var p = Path.Combine(FixturesRoot(), "corrupt", "flipped_hmac.sb");
        var ex = Assert.Throws<VerifyException>(() =>
            SbVerifier.Verify(File.ReadAllBytes(p), HmacKey()));
        Assert.True(ex.Kind == VerifyErrorKind.HmacMismatch
                 || ex.Kind == VerifyErrorKind.BrokenChain
                 || ex.Kind == VerifyErrorKind.BadHex,
            $"unexpected kind: {ex.Kind}");
    }

    [Fact]
    public void RejectFlippedSig()
    {
        var p = Path.Combine(FixturesRoot(), "corrupt", "flipped_sig.sb");
        var ex = Assert.Throws<VerifyException>(() =>
            SbVerifier.Verify(File.ReadAllBytes(p), HmacKey()));
        Assert.Equal(VerifyErrorKind.SignatureMismatch, ex.Kind);
    }

    [Fact]
    public void RejectBrokenChain()
    {
        var p = Path.Combine(FixturesRoot(), "corrupt", "broken_chain.sb");
        var ex = Assert.Throws<VerifyException>(() =>
            SbVerifier.Verify(File.ReadAllBytes(p), HmacKey()));
        Assert.True(ex.Kind == VerifyErrorKind.BrokenChain
                 || ex.Kind == VerifyErrorKind.HmacMismatch,
            $"unexpected kind: {ex.Kind}");
    }

    [Fact]
    public void RejectBadFormatVersion()
    {
        var p = Path.Combine(FixturesRoot(), "corrupt", "bad_format_version.sb");
        var ex = Assert.Throws<VerifyException>(() =>
            SbVerifier.Verify(File.ReadAllBytes(p), HmacKey()));
        Assert.True(ex.Kind == VerifyErrorKind.UnsupportedFormatVersion
                 || ex.Kind == VerifyErrorKind.HmacMismatch,
            $"unexpected kind: {ex.Kind}");
    }

    // --- empty / malformed inputs ------------------------------------------

    [Fact]
    public void RejectEmpty()
    {
        var ex = Assert.Throws<VerifyException>(() =>
            SbVerifier.Verify(Array.Empty<byte>(), HmacKey()));
        Assert.Equal(VerifyErrorKind.MissingHeader, ex.Kind);
    }

    [Fact]
    public void RejectWrongHmacKey()
    {
        var p = Path.Combine(FixturesRoot(), "good", "multi_step.sb");
        var wrong = new byte[32];
        for (int i = 0; i < 32; i++) wrong[i] = 0xaa;
        var ex = Assert.Throws<VerifyException>(() =>
            SbVerifier.Verify(File.ReadAllBytes(p), wrong));
        Assert.Equal(VerifyErrorKind.HmacMismatch, ex.Kind);
        Assert.Equal(0, ex.FrameIndex);
    }

    [Fact]
    public void FixtureSha256sMatchManifest()
    {
        var m = LoadManifest();
        AssertGroupChecksums(m, "good");
        AssertGroupChecksums(m, "corrupt");
    }

    private static void AssertGroupChecksums(JsonElement manifest, string section)
    {
        foreach (var entry in manifest.GetProperty(section).EnumerateArray())
        {
            string name = entry.GetProperty("name").GetString()!;
            string sha = entry.GetProperty("sha256").GetString()!;
            var data = File.ReadAllBytes(Path.Combine(FixturesRoot(), section, name));
            Assert.Equal(sha, SbVerifier.HexEncode(SbVerifier.Sha256(data)));
        }
    }
}
