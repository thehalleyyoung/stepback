"""Comet-Σ integration package for stepback.

This package wires the Comet-Σ five-layer auditable controller stack
(see ``kitchensink/comet_sigma/``) into stepback's primary surfaces.
Every sublayer is gated by the matching ``COMET_SIGMA_*`` flag in
``comet_sigma.flags``; when a flag is OFF the integration is a no-op
and contributes zero overhead to the hot path.

The first wired surface (Step 1 of ``COMET_SIGMA_1000.md``) is
:mod:`stepback.trace_writer` — see :mod:`stepback.comet_sigma.l1_trace_writer`.
"""
from __future__ import annotations

# Re-export the upstream Comet-Σ symbols so callers inside stepback can
# write ``from stepback.comet_sigma import AuditableArtifact`` regardless
# of where the upstream package lives on the import path. We try a couple
# of well-known import paths so this works both when ``comet_sigma`` is
# installed as a top-level package and when it ships inside the
# ``kitchensink`` namespace package.

try:  # pragma: no cover - exercised indirectly by import paths
    from comet_sigma.audit import (  # type: ignore[import-not-found]
        AuditableArtifact,
        ProvenanceChain,
        ProvenanceEntry,
        Receipt,
        new_artifact,
        extend_provenance,
    )
    from comet_sigma.flags import is_enabled, all_flags  # type: ignore[import-not-found]
    _COMET_SIGMA_AVAILABLE = True
    _COMET_SIGMA_SOURCE = "comet_sigma"
except Exception:  # pragma: no cover
    try:
        from kitchensink.comet_sigma.audit import (  # type: ignore[import-not-found]
            AuditableArtifact,
            ProvenanceChain,
            ProvenanceEntry,
            Receipt,
            new_artifact,
            extend_provenance,
        )
        from kitchensink.comet_sigma.flags import is_enabled, all_flags  # type: ignore[import-not-found]
        _COMET_SIGMA_AVAILABLE = True
        _COMET_SIGMA_SOURCE = "kitchensink.comet_sigma"
    except Exception:
        _COMET_SIGMA_AVAILABLE = False
        _COMET_SIGMA_SOURCE = ""
        AuditableArtifact = None  # type: ignore[assignment,misc]
        ProvenanceChain = None  # type: ignore[assignment,misc]
        ProvenanceEntry = None  # type: ignore[assignment,misc]
        Receipt = None  # type: ignore[assignment,misc]

        def new_artifact(*_args, **_kwargs):  # type: ignore[no-redef]
            raise RuntimeError(
                "comet_sigma is not installed; install the kitchensink "
                "package or add comet_sigma to the import path"
            )

        def extend_provenance(*_args, **_kwargs):  # type: ignore[no-redef]
            raise RuntimeError("comet_sigma is not installed")

        def is_enabled(name: str) -> bool:  # type: ignore[no-redef]
            # Be defensive: when comet_sigma is missing every flag is OFF.
            return False

        def all_flags() -> dict[str, bool]:  # type: ignore[no-redef]
            return {}


def comet_sigma_available() -> bool:
    """Return True iff the upstream ``comet_sigma`` package is importable."""
    return _COMET_SIGMA_AVAILABLE


def comet_sigma_source() -> str:
    """Return the dotted path under which ``comet_sigma`` was imported.

    Returns the empty string when :func:`comet_sigma_available` is False.
    """
    return _COMET_SIGMA_SOURCE


__all__ = [
    "AuditableArtifact",
    "ProvenanceChain",
    "ProvenanceEntry",
    "Receipt",
    "new_artifact",
    "extend_provenance",
    "is_enabled",
    "all_flags",
    "comet_sigma_available",
    "comet_sigma_source",
]
