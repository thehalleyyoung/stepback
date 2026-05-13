#!/usr/bin/env bash
# install.sh — install the stepback CLI into ~/.local/bin
#
# Intended to be served at https://get.stepback.dev and run as:
#   curl -fsSL https://get.stepback.dev | sh
#
# Can also be run directly from a local clone:
#   sh scripts/install.sh
#
# Flags
#   --root          Allow running as root (sets INSTALL_DIR=/usr/local/bin)
#   --dir <path>    Override the install directory (default ~/.local/bin)
#   --version <v>   Install a specific version (default: latest)
#   --yes           Non-interactive; skip confirmation prompt
#   --uninstall     Remove a previous install made by this script

set -euo pipefail

# ── constants ────────────────────────────────────────────────────────────────

PACKAGE_NAME="stepback"
VENV_DIR="${HOME}/.local/share/stepback/venv"
DEFAULT_INSTALL_DIR="${HOME}/.local/bin"
GITHUB_REPO="stepback-dev/stepback"
PYPI_NAME="stepback"

# ── colour helpers ───────────────────────────────────────────────────────────

if [ -t 1 ] && command -v tput >/dev/null 2>&1; then
    BOLD=$(tput bold)
    GREEN=$(tput setaf 2)
    YELLOW=$(tput setaf 3)
    RED=$(tput setaf 1)
    RESET=$(tput sgr0)
else
    BOLD="" GREEN="" YELLOW="" RED="" RESET=""
fi

info()    { printf '%s[stepback]%s %s\n'    "$GREEN"  "$RESET" "$*"; }
warn()    { printf '%s[stepback]%s %s\n'    "$YELLOW" "$RESET" "$*" >&2; }
fatal()   { printf '%s[stepback] ERROR:%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }
section() { printf '\n%s==> %s%s\n'         "$BOLD"   "$*" "$RESET"; }

# ── argument parsing ─────────────────────────────────────────────────────────

ALLOW_ROOT=0
INSTALL_DIR="$DEFAULT_INSTALL_DIR"
VERSION="latest"
YES=0
UNINSTALL=0

while [ $# -gt 0 ]; do
    case "$1" in
        --root)      ALLOW_ROOT=1 ; INSTALL_DIR="/usr/local/bin" ;;
        --dir)       shift; INSTALL_DIR="$1" ;;
        --version)   shift; VERSION="$1" ;;
        --yes|-y)    YES=1 ;;
        --uninstall) UNINSTALL=1 ;;
        -h|--help)
            cat <<'USAGE'
Usage: sh install.sh [OPTIONS]

Options:
  --root            Allow running as root; installs to /usr/local/bin
  --dir <path>      Override install directory (default: ~/.local/bin)
  --version <v>     Install specific version tag (default: latest)
  --yes             Non-interactive; skip confirmation prompt
  --uninstall       Remove stepback CLI and its virtual environment
  -h, --help        Show this help and exit

Examples:
  # Quick install (recommended)
  curl -fsSL https://get.stepback.dev | sh

  # Pin a version
  curl -fsSL https://get.stepback.dev | sh -s -- --version v0.4.1

  # Uninstall
  sh scripts/install.sh --uninstall
USAGE
            exit 0
            ;;
        *) fatal "Unknown option: $1. Run with --help for usage." ;;
    esac
    shift
done

# ── root guard ───────────────────────────────────────────────────────────────

if [ "$(id -u)" = "0" ] && [ "$ALLOW_ROOT" = "0" ]; then
    fatal "Refusing to run as root.
  Installing as root places stepback in a system-wide location and can
  create permission issues.  If you truly need a system-wide install, re-run
  with the --root flag:
      sudo sh install.sh --root
  For personal use, run without sudo — this script installs to ~/.local/bin."
fi

# ── uninstall path ───────────────────────────────────────────────────────────

if [ "$UNINSTALL" = "1" ]; then
    section "Uninstalling stepback"
    removed=0
    if [ -f "$INSTALL_DIR/stepback" ]; then
        rm -f "$INSTALL_DIR/stepback"
        info "Removed $INSTALL_DIR/stepback"
        removed=$((removed + 1))
    fi
    if [ -d "$VENV_DIR" ]; then
        rm -rf "$(dirname "$VENV_DIR")"   # remove ~/.local/share/stepback/
        info "Removed virtual environment at $(dirname "$VENV_DIR")"
        removed=$((removed + 1))
    fi
    if [ "$removed" = "0" ]; then
        warn "Nothing to remove (no stepback install found in $INSTALL_DIR)."
    else
        info "stepback uninstalled."
    fi
    exit 0
fi

# ── OS / arch detection ──────────────────────────────────────────────────────

section "Detecting platform"

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux*)   OS_LABEL="linux" ;;
    Darwin*)  OS_LABEL="macos" ;;
    MINGW*|MSYS*|CYGWIN*) OS_LABEL="windows" ;;
    *)        warn "Unrecognised OS '$OS'; treating as linux." ; OS_LABEL="linux" ;;
esac

case "$ARCH" in
    x86_64|amd64) ARCH_LABEL="x86_64" ;;
    aarch64|arm64) ARCH_LABEL="arm64" ;;
    armv7l)        ARCH_LABEL="armv7" ;;
    *)             warn "Unrecognised arch '$ARCH'; treating as x86_64." ; ARCH_LABEL="x86_64" ;;
esac

info "Platform: $OS_LABEL / $ARCH_LABEL"

# ── Python requirement check ─────────────────────────────────────────────────

section "Checking Python"

PYTHON=""
for py in python3 python3.14 python3.13 python3.12 python3.11 python3.10; do
    if command -v "$py" >/dev/null 2>&1; then
        ver=$("$py" -c 'import sys; print("%d%d" % sys.version_info[:2])')
        if [ "$ver" -ge 310 ] 2>/dev/null; then
            PYTHON="$py"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    fatal "Python 3.10 or later is required but was not found.
  Install Python 3.10+ and re-run this script, or use:
      pipx install stepback
  See https://python.org/downloads for platform-specific instructions."
fi

PYTHON_VER=$("$PYTHON" -c 'import sys; print(".".join(str(x) for x in sys.version_info[:3]))')
info "Using $PYTHON ($PYTHON_VER)"

# ── confirmation prompt ──────────────────────────────────────────────────────

section "Install plan"
printf '  Package   : %s\n'   "$PACKAGE_NAME"
printf '  Version   : %s\n'   "$VERSION"
printf '  Virtualenv: %s\n'   "$VENV_DIR"
printf '  CLI shim  : %s\n'   "$INSTALL_DIR/stepback"
printf '  Python    : %s\n'   "$PYTHON ($PYTHON_VER)"
printf '\n'

if [ "$YES" = "0" ] && [ -t 0 ]; then
    printf '%sContinue? [Y/n] %s' "$BOLD" "$RESET"
    read -r answer </dev/tty
    case "$answer" in
        n|N|no|NO) info "Aborted." ; exit 0 ;;
    esac
fi

# ── create virtual environment ───────────────────────────────────────────────

section "Creating virtual environment"

mkdir -p "$(dirname "$VENV_DIR")"

if [ -d "$VENV_DIR" ]; then
    info "Existing virtualenv found at $VENV_DIR — reusing."
else
    "$PYTHON" -m venv "$VENV_DIR"
    info "Created virtualenv at $VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

"$VENV_PIP" install --quiet --upgrade pip

# ── install stepback ─────────────────────────────────────────────────────────

section "Installing stepback"

if [ "$VERSION" = "latest" ]; then
    VERSION_SPEC="$PYPI_NAME"
else
    VERSION_SPEC="${PYPI_NAME}==${VERSION#v}"   # strip leading 'v'
fi

# Try PyPI first; fall back to GitHub releases if needed.
if "$VENV_PIP" install --quiet "$VERSION_SPEC" 2>/dev/null; then
    info "Installed via PyPI."
else
    warn "PyPI install failed; attempting GitHub release download."
    _download_from_github() {
        _tag="${VERSION:-latest}"
        if [ "$_tag" = "latest" ]; then
            _api_url="https://api.github.com/repos/${GITHUB_REPO}/releases/latest"
        else
            _api_url="https://api.github.com/repos/${GITHUB_REPO}/releases/tags/${_tag}"
        fi
        if command -v curl >/dev/null 2>&1; then
            _json=$(curl -fsSL "$_api_url")
        elif command -v wget >/dev/null 2>&1; then
            _json=$(wget -qO- "$_api_url")
        else
            return 1
        fi
        # Parse the wheel URL from the JSON assets list (no jq dependency)
        _wheel_url=$(printf '%s' "$_json" | grep -o '"browser_download_url": *"[^"]*\.whl"' \
                     | head -n1 | sed 's/.*"\(https[^"]*\)"/\1/')
        if [ -z "$_wheel_url" ]; then
            return 1
        fi
        _tmpdir=$(mktemp -d)
        _wheel_file="$_tmpdir/stepback.whl"
        if command -v curl >/dev/null 2>&1; then
            curl -fsSL -o "$_wheel_file" "$_wheel_url"
        else
            wget -qO "$_wheel_file" "$_wheel_url"
        fi
        "$VENV_PIP" install --quiet "$_wheel_file"
        rm -rf "$_tmpdir"
    }
    _download_from_github || fatal "Could not install stepback from PyPI or GitHub.
  Check your internet connection or install manually with:
      pip install stepback"
fi

INSTALLED_VERSION=$("$VENV_PYTHON" -c 'import stepback; print(stepback.__version__)')
info "Installed stepback $INSTALLED_VERSION"

# ── create CLI shim ───────────────────────────────────────────────────────────

section "Installing CLI shim"

mkdir -p "$INSTALL_DIR"

SHIM_PATH="$INSTALL_DIR/stepback"

cat > "$SHIM_PATH" <<SHIM
#!/usr/bin/env sh
# stepback CLI shim — managed by scripts/install.sh
# Do not edit; re-run install.sh to update.
exec "$VENV_DIR/bin/stepback" "\$@"
SHIM

chmod +x "$SHIM_PATH"
info "Wrote shim to $SHIM_PATH"

# ── PATH check ────────────────────────────────────────────────────────────────

case ":$PATH:" in
    *":$INSTALL_DIR:"*) : ;;   # already on PATH
    *)
        warn "$INSTALL_DIR is not on your PATH."
        _shell_rc=""
        case "${SHELL:-}" in
            */zsh)  _shell_rc="~/.zshrc" ;;
            */bash) _shell_rc="~/.bashrc" ;;
            */fish) _shell_rc="~/.config/fish/config.fish" ;;
        esac
        if [ -n "$_shell_rc" ]; then
            printf '  Add this line to %s:\n' "$_shell_rc"
        fi
        printf '      export PATH="%s:$PATH"\n'  "$INSTALL_DIR"
        printf '  Then restart your shell or run:\n'
        printf '      source %s\n'  "${_shell_rc:-~/.profile}"
        ;;
esac

# ── next-step instructions ────────────────────────────────────────────────────

section "stepback $INSTALLED_VERSION installed!"

cat <<NEXT

  ${GREEN}Try it now:${RESET}

    ${BOLD}stepback --help${RESET}
      Show all available commands.

    ${BOLD}stepback init${RESET}
      Scaffold a starter project with an example agent and a recorded trace.

    ${BOLD}stepback doctor${RESET}
      Check your environment (Python, Rust/WASM components, API keys, …).

    ${BOLD}stepback quickstart${RESET}
      Interactive wizard: pick a provider, record your first trace, open the viewer.

  ${GREEN}Documentation:${RESET}  https://stepback.dev/docs
  ${GREEN}Source:${RESET}         https://github.com/${GITHUB_REPO}
  ${GREEN}Issues:${RESET}         https://github.com/${GITHUB_REPO}/issues

  To update later:      sh -c "\$(curl -fsSL https://get.stepback.dev)"
  To uninstall:         stepback_install --uninstall
                        (or: sh scripts/install.sh --uninstall)

NEXT
