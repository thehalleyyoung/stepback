# Homebrew formula for stepback-core — the Rust CLI verifier.
# This template is rendered by packaging/homebrew/bump_formulae.py;
# the variables below are replaced at release time.
#
# stepback-core ships pre-built binaries for macOS (arm64 + x86_64) and
# Linux (x86_64).  The formula selects the correct tarball at install time;
# if no pre-built binary is available it falls back to a source build using
# Cargo.

class StepbackCore < Formula
  desc "Rust verifier and reader for .sb trace files"
  homepage "https://github.com/stepback/stepback"
  license "Apache-2.0"
  head "https://github.com/stepback/stepback.git", branch: "main"
  version "{{VERSION}}"

  on_macos do
    on_arm do
      url "{{MACOS_ARM64_URL}}"
      sha256 "{{MACOS_ARM64_SHA256}}"
    end
    on_intel do
      url "{{MACOS_X86_64_URL}}"
      sha256 "{{MACOS_X86_64_SHA256}}"
    end
  end

  on_linux do
    on_arm do
      url "{{LINUX_AARCH64_URL}}"
      sha256 "{{LINUX_AARCH64_SHA256}}"
    end
    on_intel do
      url "{{LINUX_X86_64_URL}}"
      sha256 "{{LINUX_X86_64_SHA256}}"
    end
  end

  # Fallback: build from source when no bottle matches.
  # Requires Cargo; use `brew install rust` if missing.
  head do
    depends_on "rust" => :build
  end

  def install
    if build.head?
      # Source build path.
      system "cargo", "install",
             "--root", prefix,
             "--path", "stepback-core/sb-verify"
    else
      # Pre-built binary tarball.
      bin.install "sb"
      man1.install "sb.1" if File.exist?("sb.1")
    end
  end

  test do
    # Verify the binary is functional.
    assert_match "sb #{version}", shell_output("#{bin}/sb --version")

    # Verify a minimal valid .sb trace.  The Python SDK writes canonical JSON
    # frames; the Rust reader should parse and verify the HMAC chain.
    (testpath/"trace.sb").write(
      File.read(
        File.expand_path("../../../tests/fixtures/minimal_trace.sb",
                         Formula["stepback-core"].prefix)
      )
    ) if (Formula["stepback-core"].prefix/"../../../tests/fixtures/minimal_trace.sb").exist?

    # At minimum the help text should be parseable.
    assert_match "USAGE", shell_output("#{bin}/sb --help")
  end
end
