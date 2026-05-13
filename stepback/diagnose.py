"""``stepback diagnose`` — inspect installed SDK/framework versions against the
certified compatibility matrix and warn when they fall outside certified ranges.

Public API
----------
::

    from stepback.diagnose import diagnose_all, DiagnoseResult, DiagnoseStatus

    results = diagnose_all()
    for r in results:
        print(r.name, r.status, r.installed)

The certified ranges are taken from two sources:

* :data:`~stepback.shim_certification.SHIM_VERSION_MATRIX` — LLM provider
  SDKs (openai, anthropic, boto3, google-genai, cohere, mistral, groq, …).
* :data:`FRAMEWORK_VERSION_MATRIX` — orchestration-framework packages
  (langchain-core, llama-index-core, dspy-ai, haystack-ai, pyautogen, crewai,
  semantic-kernel, strands-agents, pydantic-ai).

Each entry is checked at the **integration** level (one row per provider /
framework name) rather than the package level so that multiple integrations
sharing a common package (e.g. ``openai`` used by ``openai``, ``azure_openai``,
``nvidia_nim`` …) each appear in context.

Status values
~~~~~~~~~~~~~

``ok``
    Installed version is within the supported specifier and does not exceed
    the highest tested version.

``warn_newer``
    Installed version is within the supported specifier but higher than the
    highest explicitly tested version.  Things may work; please report issues.

``warn_unsupported``
    Installed version is outside the supported specifier.  Behaviour is
    undefined and you should update or downgrade.

``not_installed``
    The package is not found in the current environment.  This is normal for
    optional integrations you have not installed.

``compare_unavailable``
    The ``packaging`` library is absent *and* the installed version string
    could not be compared via simple heuristics.  Install ``packaging>=23``
    to get a definitive result.
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import json
from typing import Any, Dict, List, Optional

from .shim_certification import SHIM_VERSION_MATRIX

# ---------------------------------------------------------------------------
# Framework version matrix (orchestration frameworks)
# ---------------------------------------------------------------------------

#: Certified version ranges for orchestration-framework packages that
#: stepback recorders / shims target.  Same schema as
#: :data:`~stepback.shim_certification.SHIM_VERSION_MATRIX`.
FRAMEWORK_VERSION_MATRIX: Dict[str, Dict[str, Any]] = {
    "langchain": {
        "package": "langchain-core",
        "supported": ">=0.1,<0.4",
        "tested": ["0.1.52", "0.2.43", "0.3.29"],
        "notes": (
            "stepback.shims.langchain_callback_handler / wrap_langchain_tool; "
            "duck-typed against langchain_core.callbacks.BaseCallbackHandler."
        ),
    },
    "llama_index": {
        "package": "llama-index-core",
        "supported": ">=0.10,<0.13",
        "tested": ["0.10.68", "0.12.4"],
        "notes": (
            "stepback.shims.LlamaIndexCallbackHandler; "
            "duck-typed against llama_index.core.callbacks.BaseCallbackHandler."
        ),
    },
    "dspy": {
        "package": "dspy-ai",
        "supported": ">=2.4,<2.7",
        "tested": ["2.4.17", "2.5.43", "2.6.23"],
        "notes": (
            "stepback.shims.DSPyCallbackHandler; "
            "duck-typed against dspy.Callback."
        ),
    },
    "haystack": {
        "package": "haystack-ai",
        "supported": ">=2.3,<2.10",
        "tested": ["2.3.1", "2.7.1"],
        "notes": (
            "stepback.shims.HaystackTracer; "
            "duck-typed against haystack.tracing.Tracer."
        ),
    },
    "autogen": {
        "package": "pyautogen",
        "supported": ">=0.2,<0.5",
        "tested": ["0.2.36", "0.3.2"],
        "notes": (
            "stepback.shims.AutoGenEventHandler; "
            "duck-typed against autogen.runtime_logging event signatures."
        ),
    },
    "crewai": {
        "package": "crewai",
        "supported": ">=0.55,<0.90",
        "tested": ["0.55.2", "0.80.0"],
        "notes": (
            "stepback.shims.crewai_step_recorder; "
            "duck-typed against crewai.Agent/Task execute hooks."
        ),
    },
    "semantic_kernel": {
        "package": "semantic-kernel",
        "supported": ">=1.0,<2.0",
        "tested": ["1.3.0", "1.16.0"],
        "notes": (
            "stepback.shims.SemanticKernelFilter; "
            "duck-typed against semantic_kernel.filters function-invocation API."
        ),
    },
    "strands_agents": {
        "package": "strands-agents",
        "supported": ">=0.1,<0.2",
        "tested": ["0.1.4"],
        "notes": (
            "stepback.shims.StrandsAgentRecorder; "
            "duck-typed against strands.Agent callback hooks."
        ),
    },
    "pydantic_ai": {
        "package": "pydantic-ai",
        "supported": ">=0.0.9,<0.2",
        "tested": ["0.0.20", "0.0.46"],
        "notes": (
            "stepback.shims.PydanticAIInstrumentor; "
            "duck-typed against pydantic_ai.Agent run hooks."
        ),
    },
}

# ---------------------------------------------------------------------------
# Status enum (plain strings to keep JSON output clean)
# ---------------------------------------------------------------------------

DiagnoseStatus = str  # one of: ok | warn_newer | warn_unsupported | not_installed | compare_unavailable

OK: DiagnoseStatus = "ok"
WARN_NEWER: DiagnoseStatus = "warn_newer"
WARN_UNSUPPORTED: DiagnoseStatus = "warn_unsupported"
NOT_INSTALLED: DiagnoseStatus = "not_installed"
COMPARE_UNAVAILABLE: DiagnoseStatus = "compare_unavailable"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DiagnoseResult:
    """Version-check result for a single integration."""

    #: Integration/provider name (key in the matrix).
    name: str
    #: PyPI package name.
    package: str
    #: ``"provider"`` or ``"framework"``.
    kind: str
    #: Installed version string, or ``None`` if not installed.
    installed: Optional[str]
    #: PEP 440 specifier string from the matrix (e.g. ``">=1.0,<2.0"``).
    supported_spec: str
    #: Explicitly tested versions from the matrix.
    tested_versions: List[str]
    #: Diagnosis status.
    status: DiagnoseStatus
    #: Human-readable detail message.
    detail: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "package": self.package,
            "kind": self.kind,
            "installed": self.installed,
            "supported_spec": self.supported_spec,
            "tested_versions": self.tested_versions,
            "status": self.status,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------

def get_installed_version(package_name: str) -> Optional[str]:
    """Return the installed version of *package_name* or ``None``."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _check_version(
    installed_str: str,
    supported_spec_str: str,
    tested: List[str],
) -> DiagnoseStatus:
    """Classify *installed_str* against *supported_spec_str* and *tested*.

    Uses ``packaging`` if available; falls back to a simple numeric heuristic
    that is safe for the common ``X.Y.Z`` case but returns
    :data:`COMPARE_UNAVAILABLE` if the strings are not in that form.
    """
    try:
        from packaging.version import Version
        from packaging.specifiers import SpecifierSet

        installed = Version(installed_str)
        spec = SpecifierSet(supported_spec_str)
        if not spec.contains(installed, prereleases=True):
            return WARN_UNSUPPORTED
        if tested:
            max_tested = max(Version(v) for v in tested)
            if installed > max_tested:
                return WARN_NEWER
        return OK
    except Exception:
        pass

    # Fallback: plain X.Y.Z tuple comparison — sufficient for most cases.
    def _parse(s: str):
        try:
            return tuple(int(x) for x in s.split(".")[:3])
        except ValueError:
            return None

    installed_t = _parse(installed_str)
    if tested:
        tested_ts = [_parse(v) for v in tested]
        if None not in tested_ts and installed_t is not None:
            max_tested_t = max(tested_ts)  # type: ignore[type-var]
            if installed_t > max_tested_t:
                return WARN_NEWER
            return OK
    return COMPARE_UNAVAILABLE


def _check_entry(name: str, entry: Dict[str, Any], kind: str) -> DiagnoseResult:
    """Build a :class:`DiagnoseResult` for one matrix entry."""
    package = entry["package"]
    supported_spec = entry.get("supported", "")
    tested = list(entry.get("tested", []))

    installed = get_installed_version(package)
    if installed is None:
        return DiagnoseResult(
            name=name,
            package=package,
            kind=kind,
            installed=None,
            supported_spec=supported_spec,
            tested_versions=tested,
            status=NOT_INSTALLED,
            detail=f"{package} is not installed",
        )

    status = _check_version(installed, supported_spec, tested)

    if status == OK:
        detail = f"{package}=={installed} is within certified range"
    elif status == WARN_NEWER:
        max_tested = tested[-1] if tested else "?"
        detail = (
            f"{package}=={installed} is newer than the highest tested version "
            f"({max_tested}); things may work but please report issues"
        )
    elif status == WARN_UNSUPPORTED:
        detail = (
            f"{package}=={installed} is outside supported specifier "
            f"({supported_spec}); behaviour is undefined"
        )
    else:
        detail = (
            f"{package}=={installed} could not be compared against the matrix "
            "(install packaging>=23 for a definitive result)"
        )

    return DiagnoseResult(
        name=name,
        package=package,
        kind=kind,
        installed=installed,
        supported_spec=supported_spec,
        tested_versions=tested,
        status=status,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def diagnose_all() -> List[DiagnoseResult]:
    """Return a :class:`DiagnoseResult` for every entry in the combined matrix.

    The combined matrix is :data:`~stepback.shim_certification.SHIM_VERSION_MATRIX`
    (kind ``"provider"``) plus :data:`FRAMEWORK_VERSION_MATRIX` (kind
    ``"framework"``), iterated in insertion order.
    """
    results: List[DiagnoseResult] = []
    for name, entry in SHIM_VERSION_MATRIX.items():
        results.append(_check_entry(name, entry, kind="provider"))
    for name, entry in FRAMEWORK_VERSION_MATRIX.items():
        results.append(_check_entry(name, entry, kind="framework"))
    return results


def format_diagnose_table(
    results: List[DiagnoseResult],
    show_all: bool = False,
) -> str:
    """Format *results* as a human-readable table.

    Parameters
    ----------
    results:
        List returned by :func:`diagnose_all`.
    show_all:
        When ``False`` (default) rows with status ``not_installed`` are
        omitted unless there are no installed packages at all.
    """
    visible = [
        r for r in results
        if show_all or r.status != NOT_INSTALLED
    ]
    if not visible:
        return "(no installed packages found; run with --all to see the full matrix)\n"

    # Column widths.
    col_name = max(len(r.name) for r in visible)
    col_pkg = max(len(r.package) for r in visible)
    col_inst = max(len(r.installed or "—") for r in visible)
    col_status = max(len(r.status) for r in visible)

    header = (
        f"{'name':<{col_name}}  {'package':<{col_pkg}}  "
        f"{'installed':<{col_inst}}  {'status':<{col_status}}  detail"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in visible:
        inst_str = r.installed or "—"
        lines.append(
            f"{r.name:<{col_name}}  {r.package:<{col_pkg}}  "
            f"{inst_str:<{col_inst}}  {r.status:<{col_status}}  {r.detail}"
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "FRAMEWORK_VERSION_MATRIX",
    "DiagnoseResult",
    "DiagnoseStatus",
    "OK",
    "WARN_NEWER",
    "WARN_UNSUPPORTED",
    "NOT_INSTALLED",
    "COMPARE_UNAVAILABLE",
    "get_installed_version",
    "diagnose_all",
    "format_diagnose_table",
]
