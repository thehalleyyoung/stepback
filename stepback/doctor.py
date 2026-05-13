"""``stepback doctor`` — environment health checks.

Checks Python version, optional Rust/WASM components, key material,
writable trace directory, and (optionally) network reachability of
configured LLM providers.  Prints a PASS/WARN/FAIL table; exits 1 if
any row is FAIL.

Public API::

    from stepback.doctor import run_doctor, DoctorCheck, PASS, WARN, FAIL

    checks = run_doctor()
    for c in checks:
        print(c.status, c.name, c.detail)
"""
from __future__ import annotations

import dataclasses
import importlib
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


@dataclasses.dataclass
class DoctorCheck:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str
    remediation: Optional[str] = None


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_python_version() -> DoctorCheck:
    vi = sys.version_info
    version_str = f"{vi[0]}.{vi[1]}.{vi[2]}"
    if vi >= (3, 10):
        return DoctorCheck(
            name="Python version",
            status=PASS,
            detail=f"Python {version_str}",
        )
    if vi >= (3, 9):
        return DoctorCheck(
            name="Python version",
            status=WARN,
            detail=f"Python {version_str} (3.10+ recommended)",
            remediation="Upgrade to Python 3.10 or later: https://www.python.org/downloads/",
        )
    return DoctorCheck(
        name="Python version",
        status=FAIL,
        detail=f"Python {version_str} (unsupported; stepback requires ≥3.9)",
        remediation="Upgrade to Python 3.10 or later: https://www.python.org/downloads/",
    )


def _check_stepback_core() -> DoctorCheck:
    """Check whether the optional Rust extension (PyO3 bindings) is importable."""
    try:
        import stepback_core  # type: ignore[import]
        version = getattr(stepback_core, "__version__", "unknown")
        return DoctorCheck(
            name="Rust core (stepback_core)",
            status=PASS,
            detail=f"stepback_core {version} available",
        )
    except ImportError:
        return DoctorCheck(
            name="Rust core (stepback_core)",
            status=WARN,
            detail="stepback_core not installed (pure-Python fallback active)",
            remediation=(
                "Install the Rust extension for faster verification:\n"
                "  pip install stepback[rust]\n"
                "  # or build from source: cd stepback-core && maturin develop"
            ),
        )


def _check_wasm() -> DoctorCheck:
    """Check whether the WASM build artefacts exist relative to the package."""
    # Locate the package root
    pkg_root = Path(__file__).parent.parent
    wasm_dir = pkg_root / "wasm"
    wasm_pkg = wasm_dir / "pkg"
    wasm_js = wasm_pkg / "stepback_wasm.js"
    wasm_bg = wasm_pkg / "stepback_wasm_bg.wasm"

    if wasm_js.exists() and wasm_bg.exists():
        return DoctorCheck(
            name="WASM build",
            status=PASS,
            detail=f"WASM artefacts found at {wasm_pkg}",
        )
    if wasm_dir.exists():
        return DoctorCheck(
            name="WASM build",
            status=WARN,
            detail=f"wasm/ source present but not yet built ({wasm_pkg} missing)",
            remediation=(
                "Build the WASM module:\n"
                "  cd wasm && wasm-pack build --target web --out-dir pkg"
            ),
        )
    return DoctorCheck(
        name="WASM build",
        status=WARN,
        detail="wasm/ directory not found (WASM viewer unavailable)",
        remediation=(
            "The WASM viewer is an optional component.  To build it:\n"
            "  cd wasm && wasm-pack build --target web --out-dir pkg"
        ),
    )


def _find_stepback_toml(start: Path) -> Optional[Path]:
    """Walk up from *start* looking for stepback.toml (mirrors the standard search)."""
    for parent in [start, *start.parents]:
        candidate = parent / "stepback.toml"
        if candidate.exists():
            return candidate
    return None


def _load_toml_config(path: Path) -> dict:
    """Load a TOML file; returns empty dict on failure."""
    try:
        if sys.version_info >= (3, 11):
            import tomllib
            return tomllib.loads(path.read_text())
        else:
            # Fallback: try tomli (third-party) then a naive key=value parser
            try:
                import tomllib  # type: ignore[import]
                return tomllib.loads(path.read_text())
            except ImportError:
                pass
            try:
                import tomli  # type: ignore[import]
                return tomli.loads(path.read_text())
            except ImportError:
                pass
            # Very naive parser — enough to extract simple string values
            config: dict = {}
            section: dict = config
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("["):
                    # section header — ignore nesting, just collect into root
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    section[key] = val
            return config
    except Exception:
        return {}


def _check_key_material() -> DoctorCheck:
    """Check whether HMAC key material is configured."""
    env_key = os.environ.get("STEPBACK_HMAC_KEY_HEX") or os.environ.get("STEPBACK_HMAC_KEY")
    if env_key:
        return DoctorCheck(
            name="HMAC key",
            status=PASS,
            detail="Key loaded from environment variable",
        )

    toml_path = _find_stepback_toml(Path.cwd())
    if toml_path:
        cfg = _load_toml_config(toml_path)
        stepback_section = cfg.get("stepback", cfg)
        if stepback_section.get("hmac_key_hex"):
            return DoctorCheck(
                name="HMAC key",
                status=PASS,
                detail=f"Key configured in {toml_path}",
            )
        return DoctorCheck(
            name="HMAC key",
            status=WARN,
            detail=f"stepback.toml found at {toml_path} but hmac_key_hex is not set",
            remediation=(
                "Add an HMAC key to stepback.toml:\n"
                "  python -c \"import secrets; print(secrets.token_hex(32))\"\n"
                "  # then set: hmac_key_hex = \"<output above>\""
            ),
        )

    return DoctorCheck(
        name="HMAC key",
        status=WARN,
        detail="No HMAC key configured (traces will be unsigned)",
        remediation=(
            "Generate and configure an HMAC key:\n"
            "  stepback init  # creates stepback.toml with a commented hmac_key_hex\n"
            "  # or set env: export STEPBACK_HMAC_KEY_HEX=$(python -c "
            "\"import secrets; print(secrets.token_hex(32))\")"
        ),
    )


def _check_trace_dir() -> DoctorCheck:
    """Check that the configured (or default) trace directory is writable."""
    # Determine trace_dir from config or default
    trace_dir_str = "traces"
    toml_path = _find_stepback_toml(Path.cwd())
    if toml_path:
        cfg = _load_toml_config(toml_path)
        stepback_section = cfg.get("stepback", cfg)
        trace_dir_str = stepback_section.get("trace_dir", "traces")

    trace_dir = Path(trace_dir_str)
    if not trace_dir.is_absolute():
        trace_dir = Path.cwd() / trace_dir

    if trace_dir.exists():
        # Check writable with a temp file
        try:
            with tempfile.NamedTemporaryFile(dir=trace_dir, delete=True):
                pass
            return DoctorCheck(
                name="Trace directory",
                status=PASS,
                detail=f"{trace_dir} exists and is writable",
            )
        except OSError as exc:
            return DoctorCheck(
                name="Trace directory",
                status=FAIL,
                detail=f"{trace_dir} is not writable: {exc}",
                remediation=f"Fix permissions: chmod u+w {trace_dir}",
            )
    else:
        # Directory doesn't exist yet — try creating it
        try:
            trace_dir.mkdir(parents=True, exist_ok=True)
            return DoctorCheck(
                name="Trace directory",
                status=WARN,
                detail=f"{trace_dir} did not exist; created successfully",
                remediation=None,
            )
        except OSError as exc:
            return DoctorCheck(
                name="Trace directory",
                status=FAIL,
                detail=f"Cannot create trace directory {trace_dir}: {exc}",
                remediation=(
                    f"Create the directory manually:\n  mkdir -p {trace_dir}\n"
                    f"  # or update trace_dir in stepback.toml"
                ),
            )


_PROVIDER_HOSTS = {
    "OpenAI": "api.openai.com",
    "Anthropic": "api.anthropic.com",
    "Google Generative AI": "generativelanguage.googleapis.com",
    "Cohere": "api.cohere.ai",
}


def _check_network(timeout: float = 3.0) -> List[DoctorCheck]:
    """Check TCP reachability of known LLM provider endpoints (port 443)."""
    import socket

    checks: List[DoctorCheck] = []
    for name, host in _PROVIDER_HOSTS.items():
        try:
            conn = socket.create_connection((host, 443), timeout=timeout)
            conn.close()
            checks.append(DoctorCheck(
                name=f"Network → {name}",
                status=PASS,
                detail=f"{host}:443 reachable",
            ))
        except OSError as exc:
            checks.append(DoctorCheck(
                name=f"Network → {name}",
                status=WARN,
                detail=f"{host}:443 unreachable: {exc}",
                remediation=(
                    f"Check your internet connection or proxy settings.\n"
                    f"If you do not use {name}, this warning is safe to ignore."
                ),
            ))
    return checks


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_doctor(check_network: bool = False) -> List[DoctorCheck]:
    """Run all environment checks and return a list of :class:`DoctorCheck` results."""
    results: List[DoctorCheck] = [
        _check_python_version(),
        _check_stepback_core(),
        _check_wasm(),
        _check_key_material(),
        _check_trace_dir(),
    ]
    if check_network:
        results.extend(_check_network())
    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_STATUS_ICON = {PASS: "✓", WARN: "!", FAIL: "✗"}
_COL_WIDTHS = {"name": 32, "status": 4, "detail": 52}


def format_doctor_table(checks: List[DoctorCheck]) -> str:
    """Render a human-readable PASS/WARN/FAIL table from *checks*."""
    lines: List[str] = []
    # Header
    lines.append(f"{'CHECK':<{_COL_WIDTHS['name']}}  {'':4}  DETAIL")
    lines.append("-" * (_COL_WIDTHS["name"] + 2 + 4 + 2 + _COL_WIDTHS["detail"]))

    for c in checks:
        icon = _STATUS_ICON.get(c.status, "?")
        status_col = f"[{c.status}]"
        first_detail_line, *rest_detail = (c.detail or "").splitlines() or [""]
        lines.append(
            f"{c.name:<{_COL_WIDTHS['name']}}  {status_col:<6}  {first_detail_line}"
        )
        for extra in rest_detail:
            lines.append(f"{'':>{_COL_WIDTHS['name']}}          {extra}")

    # Remediation block for WARNs and FAILs
    remediations = [(c.name, c.remediation) for c in checks if c.remediation]
    if remediations:
        lines.append("")
        lines.append("Remediation:")
        for name, rem in remediations:
            lines.append(f"  [{name}]")
            for rem_line in (rem or "").splitlines():
                lines.append(f"    {rem_line}")
            lines.append("")

    lines.append("")
    n_pass = sum(1 for c in checks if c.status == PASS)
    n_warn = sum(1 for c in checks if c.status == WARN)
    n_fail = sum(1 for c in checks if c.status == FAIL)
    lines.append(f"Summary: {n_pass} PASS  {n_warn} WARN  {n_fail} FAIL")
    lines.append("")
    return "\n".join(lines)
