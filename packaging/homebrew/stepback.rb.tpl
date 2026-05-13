# Homebrew formula for the stepback Python CLI.
# This template is rendered by packaging/homebrew/bump_formulae.py;
# the variables below are replaced at release time.
#
# Manual install from source:
#   brew install --build-from-source ./packaging/homebrew/stepback.rb
#
# After a release is tagged the CI workflow renders this into the tap repo at
# https://github.com/stepback/homebrew-tap

class Stepback < Formula
  include Language::Python::Virtualenv

  desc "Record, replay, and debug multi-step AI agent traces"
  homepage "https://github.com/stepback/stepback"
  url "{{SDIST_URL}}"
  sha256 "{{SDIST_SHA256}}"
  license "Apache-2.0"
  head "https://github.com/stepback/stepback.git", branch: "main"

  bottle do
    # Bottles are built by GitHub Actions and uploaded to the tap.
    # Regenerated on each tagged release via brew/publish.
    root_url "https://github.com/stepback/homebrew-tap/releases/download/{{VERSION}}"
    {{BOTTLE_BLOCK}}
  end

  depends_on "python@3.12"

  resource "cryptography" do
    url "{{CRYPTOGRAPHY_URL}}"
    sha256 "{{CRYPTOGRAPHY_SHA256}}"
  end

  def install
    virtualenv_install_with_resources
  end

  test do
    system bin/"stepback", "--version"
    (testpath/"agent.py").write(<<~PYTHON)
      import json, pathlib
      from stepback import record

      with record(pathlib.Path("test.sb")) as ctx:
          ctx.step(
              kind="tool",
              inputs={"query": "hello"},
              outputs={"answer": "world"},
          )

      print(json.dumps({"ok": True}))
    PYTHON
    output = shell_output("#{python3} agent.py")
    assert_match '"ok": true', output
    assert_predicate testpath/"test.sb", :exist?
  end
end
