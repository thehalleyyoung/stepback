# Homebrew tap formulae

This directory contains the Ruby formula templates for the
[`stepback/homebrew-tap`](https://github.com/stepback/homebrew-tap) Homebrew
tap.  The templates are rendered at release time by `bump_formulae.py`.

## Formulae

| File | Package | Description |
|------|---------|-------------|
| `stepback.rb.tpl` | `stepback` | Python CLI — `pip`-style virtualenv formula |
| `stepback-core.rb.tpl` | `stepback-core` | Rust CLI verifier (`sb`) — pre-built binary |
| `stepback-wasm-viewer.rb.tpl` | `stepback-wasm-viewer` | Offline WASM HTML viewer |

## Install (end-user, once tap is published)

```sh
brew tap stepback/tap
brew install stepback              # Python CLI
brew install stepback-core         # Rust sb verifier
brew install stepback-wasm-viewer  # offline HTML viewer
```

## Bump script (maintainer / CI)

`bump_formulae.py` takes `--version`, URLs and SHA-256 digests for every
release asset and writes rendered `.rb` files to `--output-dir`.

The GitHub Actions workflow `.github/workflows/brew-release.yml` calls this
script automatically on every tagged release and opens a PR against the tap
repo.  See that file for the required `TAP_BUMP_TOKEN` secret.

### Manual render (dry-run)

```sh
python3 packaging/homebrew/bump_formulae.py \
    --version 0.1.0 \
    --sdist-url            "https://example.com/stepback-0.1.0.tar.gz" \
    --sdist-sha256         "aaaa..." \
    --cryptography-url     "https://example.com/cryptography-42.0.0.tar.gz" \
    --cryptography-sha256  "bbbb..." \
    --macos-arm64-url      "https://example.com/sb-0.1.0-aarch64-apple-darwin.tar.gz" \
    --macos-arm64-sha256   "cccc..." \
    --macos-x86-64-url     "https://example.com/sb-0.1.0-x86_64-apple-darwin.tar.gz" \
    --macos-x86-64-sha256  "dddd..." \
    --linux-aarch64-url    "https://example.com/sb-0.1.0-aarch64-unknown-linux-gnu.tar.gz" \
    --linux-aarch64-sha256 "eeee..." \
    --linux-x86-64-url     "https://example.com/sb-0.1.0-x86_64-unknown-linux-gnu.tar.gz" \
    --linux-x86-64-sha256  "ffff..." \
    --wasm-viewer-url      "https://example.com/stepback-viewer-0.1.0.html" \
    --wasm-viewer-sha256   "gggg..." \
    --output-dir /tmp/formulae
```
