#!/usr/bin/env python3
"""Render Homebrew formula templates for a given stepback release.

Usage (called by the brew-release CI workflow):
    python3 packaging/homebrew/bump_formulae.py \\
        --version 0.2.0 \\
        --sdist-url  https://files.pythonhosted.org/packages/.../stepback-0.2.0.tar.gz \\
        --sdist-sha256 abc123... \\
        --cryptography-url  https://files.pythonhosted.org/.../cryptography-42.0.0.tar.gz \\
        --cryptography-sha256 def456... \\
        --macos-arm64-url   https://github.com/stepback/stepback/releases/download/v0.2.0/sb-0.2.0-aarch64-apple-darwin.tar.gz \\
        --macos-arm64-sha256 ... \\
        --macos-x86-64-url  ... \\
        --macos-x86-64-sha256 ... \\
        --linux-aarch64-url ... \\
        --linux-aarch64-sha256 ... \\
        --linux-x86-64-url  ... \\
        --linux-x86-64-sha256 ... \\
        --wasm-viewer-url   https://github.com/stepback/stepback/releases/download/v0.2.0/stepback-viewer-0.2.0.html \\
        --wasm-viewer-sha256 ... \\
        --output-dir /path/to/homebrew-tap/Formula

The rendered .rb files are written to --output-dir (default: current directory).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import urllib.request
from pathlib import Path


_TEMPLATES = {
    "stepback.rb": "stepback.rb.tpl",
    "stepback-core.rb": "stepback-core.rb.tpl",
    "stepback-wasm-viewer.rb": "stepback-wasm-viewer.rb.tpl",
}

_HERE = Path(__file__).parent


def _sha256_url(url: str) -> str:
    """Download *url* and return its hex-encoded SHA-256 digest."""
    print(f"  sha256({url}) ...", file=sys.stderr)
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (scheme is https)
        data = resp.read()
    return hashlib.sha256(data).hexdigest()


def _load_template(name: str) -> str:
    path = _HERE / name
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_text(encoding="utf-8")


def _render(template: str, replacements: dict[str, str]) -> str:
    """Replace all {{KEY}} placeholders in *template*."""
    for key, value in replacements.items():
        placeholder = "{{" + key + "}}"
        if placeholder not in template:
            raise ValueError(f"Placeholder {placeholder!r} not found in template")
        template = template.replace(placeholder, value)
    # Warn about any remaining unreplaced placeholders.
    remaining = re.findall(r"\{\{[A-Z0-9_]+\}\}", template)
    if remaining:
        print(
            f"WARNING: unreplaced placeholders remain: {remaining}",
            file=sys.stderr,
        )
    return template


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="Release version, e.g. 0.2.0")
    # Python sdist
    parser.add_argument("--sdist-url", required=True)
    parser.add_argument("--sdist-sha256", default="", help="Leave blank to download and compute")
    # cryptography dependency
    parser.add_argument("--cryptography-url", required=True)
    parser.add_argument("--cryptography-sha256", default="")
    # stepback-core binaries
    parser.add_argument("--macos-arm64-url", required=True)
    parser.add_argument("--macos-arm64-sha256", default="")
    parser.add_argument("--macos-x86-64-url", required=True)
    parser.add_argument("--macos-x86-64-sha256", default="")
    parser.add_argument("--linux-aarch64-url", required=True)
    parser.add_argument("--linux-aarch64-sha256", default="")
    parser.add_argument("--linux-x86-64-url", required=True)
    parser.add_argument("--linux-x86-64-sha256", default="")
    # WASM viewer
    parser.add_argument("--wasm-viewer-url", required=True)
    parser.add_argument("--wasm-viewer-sha256", default="")
    # Bottles block (optional, injected as-is)
    parser.add_argument("--bottle-block", default="# No bottles yet; build from source")
    # Output
    parser.add_argument("--output-dir", default=".", type=Path)
    parser.add_argument(
        "--download-missing-shas",
        action="store_true",
        help="Download any asset whose --*-sha256 flag was not provided and compute it",
    )

    args = parser.parse_args(argv)

    def maybe_sha(url: str, provided: str) -> str:
        if provided:
            return provided
        if args.download_missing_shas:
            return _sha256_url(url)
        raise ValueError(
            f"--*-sha256 not provided for {url!r} and --download-missing-shas not set"
        )

    sdist_sha = maybe_sha(args.sdist_url, args.sdist_sha256)
    crypto_sha = maybe_sha(args.cryptography_url, args.cryptography_sha256)
    macos_arm64_sha = maybe_sha(args.macos_arm64_url, args.macos_arm64_sha256)
    macos_x86_sha = maybe_sha(args.macos_x86_64_url, args.macos_x86_64_sha256)
    linux_aarch64_sha = maybe_sha(args.linux_aarch64_url, args.linux_aarch64_sha256)
    linux_x86_sha = maybe_sha(args.linux_x86_64_url, args.linux_x86_64_sha256)
    wasm_sha = maybe_sha(args.wasm_viewer_url, args.wasm_viewer_sha256)

    # ---- stepback.rb ----
    stepback_repl = {
        "SDIST_URL": args.sdist_url,
        "SDIST_SHA256": sdist_sha,
        "CRYPTOGRAPHY_URL": args.cryptography_url,
        "CRYPTOGRAPHY_SHA256": crypto_sha,
        "VERSION": args.version,
        "BOTTLE_BLOCK": args.bottle_block,
    }
    stepback_rb = _render(_load_template("stepback.rb.tpl"), stepback_repl)

    # ---- stepback-core.rb ----
    core_repl = {
        "VERSION": args.version,
        "MACOS_ARM64_URL": args.macos_arm64_url,
        "MACOS_ARM64_SHA256": macos_arm64_sha,
        "MACOS_X86_64_URL": args.macos_x86_64_url,
        "MACOS_X86_64_SHA256": macos_x86_sha,
        "LINUX_AARCH64_URL": args.linux_aarch64_url,
        "LINUX_AARCH64_SHA256": linux_aarch64_sha,
        "LINUX_X86_64_URL": args.linux_x86_64_url,
        "LINUX_X86_64_SHA256": linux_x86_sha,
    }
    core_rb = _render(_load_template("stepback-core.rb.tpl"), core_repl)

    # ---- stepback-wasm-viewer.rb ----
    wasm_repl = {
        "VERSION": args.version,
        "WASM_VIEWER_URL": args.wasm_viewer_url,
        "WASM_VIEWER_SHA256": wasm_sha,
    }
    wasm_rb = _render(_load_template("stepback-wasm-viewer.rb.tpl"), wasm_repl)

    # Write outputs.
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for fname, content in [
        ("stepback.rb", stepback_rb),
        ("stepback-core.rb", core_rb),
        ("stepback-wasm-viewer.rb", wasm_rb),
    ]:
        dest = output_dir / fname
        dest.write_text(content, encoding="utf-8")
        print(f"Wrote {dest}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
