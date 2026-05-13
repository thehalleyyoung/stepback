# Homebrew formula for the stepback WASM viewer — an offline, single-file PWA
# that loads and visualises .sb trace files in the browser without ever
# sending trace contents to a server.
#
# The viewer is built by `wasm-pack` from the Rust crate under `wasm/` and
# bundled into a single self-contained HTML file at release time.  This
# formula installs that HTML file plus a tiny launch helper script.

class StepbackWasmViewer < Formula
  desc "Offline single-file WASM viewer for .sb AI-agent trace files"
  homepage "https://github.com/stepback/stepback"
  url "{{WASM_VIEWER_URL}}"
  sha256 "{{WASM_VIEWER_SHA256}}"
  license "Apache-2.0"
  version "{{VERSION}}"

  def install
    # Install the self-contained viewer HTML.
    pkgshare.install "stepback-viewer.html"

    # Install a launch helper that opens the viewer on a free local port or
    # directly via a file:// URL (browsers allow WASM from file:// on macOS).
    (bin/"stepback-viewer").write <<~SHELL
      #!/usr/bin/env bash
      set -euo pipefail
      HTML="#{pkgshare}/stepback-viewer.html"
      if [[ $# -gt 0 ]]; then
          # If a .sb file is passed, pass it to the viewer via a query param.
          TRACE_PATH="$(realpath "$1")"
          URL="file://${HTML}?trace=${TRACE_PATH}"
      else
          URL="file://${HTML}"
      fi
      echo "Opening: ${URL}"
      open "${URL}" 2>/dev/null || xdg-open "${URL}" 2>/dev/null || echo "Could not open browser; open manually: ${URL}"
    SHELL
    chmod "+x", bin/"stepback-viewer"
  end

  test do
    assert_predicate pkgshare/"stepback-viewer.html", :exist?
    assert_match "#!/usr/bin/env bash", File.read(bin/"stepback-viewer")
    assert_match "stepback-viewer.html", File.read(bin/"stepback-viewer")
  end
end
