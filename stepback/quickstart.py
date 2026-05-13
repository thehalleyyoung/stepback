"""``stepback quickstart`` — interactive first-run wizard.

Guides a new user through:

1. Picking a provider shim (OpenAI / Anthropic / Bedrock / Gemini / demo).
2. Entering their API key; storing it in the OS keyring (never on disk or in
   the repo) if ``keyring`` is available, otherwise exporting it as an env-var
   for the session.
3. Recording one demo trace (using the built-in fixture agent for the demo
   provider, or a real one-step call for live providers).
4. Opening the trace in the HTML viewer.

The wizard uses ``prompt_toolkit`` for the interactive UI if it is installed;
if not, it falls back to plain ``input()`` so the command always works.

Non-interactive mode (``--non-interactive``) is supported for scripting and
tests: all prompts are skipped and the demo provider + a synthetic demo trace
are used.
"""
from __future__ import annotations

import os
import sys
import tempfile
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# ─── optional dependency sentinels ───────────────────────────────────────────

def _try_import_prompt_toolkit():
    try:
        import prompt_toolkit  # noqa: F401
        return True
    except ImportError:
        return False


def _try_import_keyring():
    try:
        import keyring  # noqa: F401
        return True
    except ImportError:
        return False


_HAS_PROMPT_TOOLKIT = _try_import_prompt_toolkit()
_HAS_KEYRING = _try_import_keyring()

# ─── provider registry ───────────────────────────────────────────────────────

@dataclass
class ProviderInfo:
    """Metadata about a supported LLM provider shim."""

    name: str
    label: str
    env_var: str
    keyring_service: str
    default_model: str
    sdk_package: str
    api_key_hint: str


PROVIDERS: Dict[str, ProviderInfo] = {
    "openai": ProviderInfo(
        name="openai",
        label="OpenAI (gpt-4o-mini-2024-07-18)",
        env_var="OPENAI_API_KEY",
        keyring_service="stepback-openai",
        default_model="gpt-4o-mini-2024-07-18",
        sdk_package="openai",
        api_key_hint="sk-...",
    ),
    "anthropic": ProviderInfo(
        name="anthropic",
        label="Anthropic (claude-3-5-haiku-20241022)",
        env_var="ANTHROPIC_API_KEY",
        keyring_service="stepback-anthropic",
        default_model="claude-3-5-haiku-20241022",
        sdk_package="anthropic",
        api_key_hint="sk-ant-...",
    ),
    "bedrock": ProviderInfo(
        name="bedrock",
        label="AWS Bedrock (us.anthropic.claude-3-5-haiku)",
        env_var="AWS_ACCESS_KEY_ID",
        keyring_service="stepback-bedrock",
        default_model="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        sdk_package="boto3",
        api_key_hint="(uses AWS credentials from environment or ~/.aws)",
    ),
    "gemini": ProviderInfo(
        name="gemini",
        label="Google Gemini (gemini-1.5-flash)",
        env_var="GEMINI_API_KEY",
        keyring_service="stepback-gemini",
        default_model="gemini-1.5-flash",
        sdk_package="google-genai",
        api_key_hint="AIza...",
    ),
    "demo": ProviderInfo(
        name="demo",
        label="Demo (no API key required — uses built-in fixture)",
        env_var="",
        keyring_service="",
        default_model="demo",
        sdk_package="",
        api_key_hint="",
    ),
}

_PROVIDER_ORDER: List[str] = ["openai", "anthropic", "bedrock", "gemini", "demo"]

# ─── keyring helpers ─────────────────────────────────────────────────────────

_KEYRING_USERNAME = "api_key"


def store_api_key(provider: ProviderInfo, api_key: str) -> bool:
    """Store *api_key* in the OS keyring.

    Returns ``True`` on success, ``False`` if the keyring is unavailable.
    Does nothing if *provider.keyring_service* is empty.
    """
    if not provider.keyring_service or not api_key:
        return False
    if not _HAS_KEYRING:
        return False
    import keyring as _kr
    _kr.set_password(provider.keyring_service, _KEYRING_USERNAME, api_key)
    return True


def retrieve_api_key(provider: ProviderInfo) -> Optional[str]:
    """Retrieve the stored API key from the OS keyring.

    Returns ``None`` if the keyring is unavailable or no key is stored.
    """
    if not provider.keyring_service:
        return None
    if not _HAS_KEYRING:
        return None
    import keyring as _kr
    return _kr.get_password(provider.keyring_service, _KEYRING_USERNAME)


# ─── I/O helpers ─────────────────────────────────────────────────────────────

def _print_banner() -> None:
    print()
    print("┌───────────────────────────────────────────────────┐")
    print("│   stepback quickstart — first-run setup wizard    │")
    print("└───────────────────────────────────────────────────┘")
    print()
    print("This wizard will:")
    print("  1. Help you pick a provider shim")
    print("  2. Store your API key safely in the OS keyring")
    print("  3. Record a demo trace")
    print("  4. Open the trace in the HTML viewer")
    print()


def _prompt_choice(
    prompt: str,
    choices: Sequence[str],
    labels: Sequence[str],
    default: int = 0,
) -> str:
    """Prompt the user to pick one of *choices*.

    Uses ``prompt_toolkit`` radio buttons if available, otherwise ``input()``.
    Returns the chosen value from *choices*.
    """
    if _HAS_PROMPT_TOOLKIT:
        return _prompt_choice_pt(prompt, choices, labels, default)
    return _prompt_choice_input(prompt, choices, labels, default)


def _prompt_choice_pt(
    prompt: str,
    choices: Sequence[str],
    labels: Sequence[str],
    default: int,
) -> str:
    from prompt_toolkit import prompt as pt_prompt
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.shortcuts import radiolist_dialog

    result = radiolist_dialog(
        title="stepback quickstart",
        text=HTML(f"<b>{prompt}</b>"),
        values=list(zip(choices, labels)),
        default=choices[default],
    ).run()
    if result is None:
        raise KeyboardInterrupt
    return result


def _prompt_choice_input(
    prompt: str,
    choices: Sequence[str],
    labels: Sequence[str],
    default: int,
) -> str:
    print(prompt)
    for i, (_, label) in enumerate(zip(choices, labels)):
        marker = " (default)" if i == default else ""
        print(f"  [{i + 1}] {label}{marker}")
    raw = input(f"\nEnter number [1–{len(choices)}] (default {default + 1}): ").strip()
    if not raw:
        return choices[default]
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(choices):
            return choices[idx]
    except ValueError:
        pass
    print(f"  Invalid choice; using default ({labels[default]})")
    return choices[default]


def _prompt_password(prompt: str, hint: str = "") -> str:
    """Prompt for a secret value, hiding input if possible."""
    if hint:
        print(f"  (format: {hint})")
    if _HAS_PROMPT_TOOLKIT:
        from prompt_toolkit import prompt as pt_prompt
        return pt_prompt(f"{prompt}: ", is_password=True).strip()
    import getpass
    return getpass.getpass(f"{prompt}: ").strip()


def _prompt_confirm(prompt: str, default: bool = True) -> bool:
    """Ask a yes/no question; returns a bool."""
    suffix = " [Y/n]" if default else " [y/N]"
    raw = input(f"{prompt}{suffix}: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


# ─── trace recording helpers ─────────────────────────────────────────────────

def _record_demo_trace(trace_path: Path) -> None:
    """Record the built-in 12-step fixture trace."""
    from .recorder import record as _record
    from .testing.agent import run_recorded_agent

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with _record(str(trace_path), signing=False) as rec:
        run_recorded_agent(rec)


def _record_openai_trace(trace_path: Path, api_key: str, model: str) -> None:
    """Record a one-step OpenAI trace to show the shim working."""
    import importlib
    openai_mod = importlib.import_module("openai")
    from .recorder import record as _record
    from .shims import wrap_openai

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    client = openai_mod.OpenAI(api_key=api_key)
    with _record(str(trace_path), signing=False) as rec:
        wrapped = wrap_openai(client, rec, default_model=model)
        wrapped.chat.completions.create(
            messages=[{"role": "user", "content": "Say 'stepback recording successful' and nothing else."}],
            max_tokens=20,
        )


def _record_anthropic_trace(trace_path: Path, api_key: str, model: str) -> None:
    """Record a one-step Anthropic trace."""
    import importlib
    anthropic_mod = importlib.import_module("anthropic")
    from .recorder import record as _record
    from .shims import wrap_anthropic

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    client = anthropic_mod.Anthropic(api_key=api_key)
    with _record(str(trace_path), signing=False) as rec:
        wrapped = wrap_anthropic(client, rec)
        wrapped.messages.create(
            model=model,
            max_tokens=20,
            messages=[{"role": "user", "content": "Say 'stepback recording successful' and nothing else."}],
        )


def _record_gemini_trace(trace_path: Path, api_key: str, model: str) -> None:
    """Record a one-step Gemini trace."""
    import importlib
    genai_mod = importlib.import_module("google.genai")
    from .recorder import record as _record
    from .shims import wrap_gemini

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    client = genai_mod.Client(api_key=api_key)
    with _record(str(trace_path), signing=False) as rec:
        wrapped = wrap_gemini(client, rec, default_model=model)
        wrapped.models.generate_content(
            model=model,
            contents="Say 'stepback recording successful' and nothing else.",
        )


def _record_trace(provider: ProviderInfo, api_key: Optional[str], trace_path: Path) -> None:
    """Dispatch to the right recording helper for *provider*."""
    if provider.name == "demo":
        _record_demo_trace(trace_path)
    elif provider.name == "openai":
        assert api_key, "API key required for OpenAI"
        _record_openai_trace(trace_path, api_key, provider.default_model)
    elif provider.name == "anthropic":
        assert api_key, "API key required for Anthropic"
        _record_anthropic_trace(trace_path, api_key, provider.default_model)
    elif provider.name == "gemini":
        assert api_key, "API key required for Gemini"
        _record_gemini_trace(trace_path, api_key, provider.default_model)
    elif provider.name == "bedrock":
        # Bedrock uses AWS credentials from env; fall back to demo trace
        _record_demo_trace(trace_path)
    else:
        _record_demo_trace(trace_path)


# ─── main wizard entry point ─────────────────────────────────────────────────

@dataclass
class QuickstartResult:
    """Result returned by :func:`run_wizard`."""

    provider: str
    trace_path: Path
    api_key_stored: bool
    html_path: Path
    opened_browser: bool
    errors: List[str] = field(default_factory=list)


def run_wizard(
    *,
    non_interactive: bool = False,
    provider_name: Optional[str] = None,
    api_key: Optional[str] = None,
    output_dir: Optional[Path] = None,
    open_browser: bool = True,
    skip_open: bool = False,
) -> QuickstartResult:
    """Run the quickstart wizard.

    Parameters
    ----------
    non_interactive:
        Skip all interactive prompts; use *provider_name* (default: ``"demo"``)
        and *api_key* as given.
    provider_name:
        Provider to use.  One of the keys in :data:`PROVIDERS`.  Defaults to
        ``"demo"`` in non-interactive mode and to an interactive choice in
        interactive mode.
    api_key:
        API key for the chosen provider.  Looked up from the OS keyring if
        ``None`` and the keyring is available.
    output_dir:
        Where to place the recorded trace and HTML file.  Defaults to a
        temporary directory.
    open_browser:
        Whether to open the HTML viewer in the default browser.
    skip_open:
        Skip opening the browser even if *open_browser* is ``True``.  Useful
        in non-interactive / test mode.
    """
    errors: List[str] = []
    api_key_stored = False

    if not non_interactive:
        _print_banner()

    # ── 1. pick provider ─────────────────────────────────────────────────────
    if provider_name is None:
        if non_interactive:
            provider_name = "demo"
        else:
            keys = _PROVIDER_ORDER
            labels = [PROVIDERS[k].label for k in keys]
            provider_name = _prompt_choice(
                "Which LLM provider would you like to use?",
                keys,
                labels,
                default=0,
            )

    if provider_name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider_name!r}.  "
            f"Choose from: {', '.join(PROVIDERS)}"
        )
    provider = PROVIDERS[provider_name]

    # ── 2. API key ───────────────────────────────────────────────────────────
    if provider.env_var and api_key is None:
        # Try OS keyring first
        api_key = retrieve_api_key(provider)
        if api_key:
            if not non_interactive:
                print(f"✓ API key retrieved from OS keyring ({provider.keyring_service})")
        else:
            # Try environment variable
            api_key = os.environ.get(provider.env_var)
            if api_key:
                if not non_interactive:
                    print(f"✓ API key found in environment variable {provider.env_var}")
            elif not non_interactive:
                # Prompt interactively
                print()
                print(f"Enter your {provider.label} API key.")
                print("  It will be stored in your OS keyring — never written to disk or the repo.")
                api_key = _prompt_password("API key", hint=provider.api_key_hint)
                if api_key:
                    stored = store_api_key(provider, api_key)
                    api_key_stored = stored
                    if stored:
                        print(f"✓ API key stored in OS keyring ({provider.keyring_service})")
                    else:
                        print(
                            "  keyring not available; exporting as env var for this session only."
                        )
                        os.environ[provider.env_var] = api_key
                else:
                    errors.append("No API key provided; falling back to demo mode.")
                    provider = PROVIDERS["demo"]
                    provider_name = "demo"

    # ── 3. record trace ──────────────────────────────────────────────────────
    if output_dir is None:
        _tmpdir = tempfile.mkdtemp(prefix="stepback-quickstart-")
        output_dir = Path(_tmpdir)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    trace_path = output_dir / "quickstart.sb"

    if not non_interactive:
        print()
        print(f"Recording trace → {trace_path}")

    try:
        _record_trace(provider, api_key, trace_path)
        if not non_interactive:
            print("✓ Trace recorded.")
    except Exception as exc:
        errors.append(f"Recording failed: {exc}")
        if not non_interactive:
            print(f"  Recording failed: {exc}")
            print("  Falling back to demo mode.")
        provider = PROVIDERS["demo"]
        provider_name = "demo"
        _record_demo_trace(trace_path)
        if not non_interactive:
            print("✓ Demo trace recorded.")

    # ── 4. export HTML and open viewer ────────────────────────────────────────
    html_path = output_dir / "quickstart.html"
    _export_html(trace_path, html_path)

    opened = False
    if open_browser and not skip_open:
        try:
            webbrowser.open(html_path.as_uri())
            opened = True
            if not non_interactive:
                print(f"✓ Opened HTML viewer: {html_path}")
        except Exception as exc:
            errors.append(f"Could not open browser: {exc}")

    # ── 5. next-steps banner ─────────────────────────────────────────────────
    if not non_interactive:
        print()
        print("━" * 54)
        print("  Quickstart complete!  Next steps:")
        print()
        print(f"  Inspect the trace:")
        print(f"    stepback inspect {trace_path}")
        print()
        print(f"  Replay (all steps cache-hit):")
        print(f"    stepback replay {trace_path}")
        print()
        if not opened:
            print(f"  Open the HTML viewer:")
            print(f"    stepback view {trace_path}")
            print()

    return QuickstartResult(
        provider=provider_name,
        trace_path=trace_path,
        api_key_stored=api_key_stored,
        html_path=html_path,
        opened_browser=opened,
        errors=errors,
    )


def _export_html(trace_path: Path, html_path: Path) -> None:
    """Write a self-contained HTML viewer for *trace_path*."""
    from .html_view import write_trace_html

    write_trace_html(str(trace_path), str(html_path))
