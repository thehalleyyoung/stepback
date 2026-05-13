"""Policy-gated trace verification.

``stepback verify --strict --policy policy.json`` runs four independent
checks in one pass and exits non-zero if any enabled check fails:

* **crypto** — HMAC chain + Ed25519 signature verification (delegates to
  :func:`stepback.trace_reader.verify_trace`).
* **schema** — SB-Trace v1 schema conformance (delegates to
  :meth:`stepback.spec.SBTraceSpec.validate_trace`).
* **canonical_bytes** — every on-disk frame is exactly the canonical-JSON
  re-encoding of its parsed body (catches whitespace, key-ordering, or
  alternate-escape variants written by non-reference encoders).
* **recorder_identity** — the trace's embedded public key and recorder
  version are constrained against an allowlist.

Policy files are **JSON** objects.  The ``--strict`` CLI flag is equivalent
to loading a policy with ``{"strict": true}`` and all checks enabled.

Example policy file::

    {
        "strict": true,
        "crypto":            {"enabled": true},
        "schema":            {"enabled": true},
        "canonical_bytes":   {"enabled": true},
        "recorder_identity": {
            "enabled": true,
            "allowed_public_keys":        [],
            "allowed_recorder_versions":  []
        }
    }

All ``enabled`` fields default to ``true``.  An empty
``allowed_public_keys`` / ``allowed_recorder_versions`` list means *no
pinning* — any value is accepted.  A non-empty list means *the trace must
match one entry*.

Public API
----------
* :class:`VerifyPolicy` — parsed policy.
* :class:`PolicyViolation` — a single check finding.
* :class:`PolicyCheckResult` — aggregated result of all enabled checks.
* :func:`verify_with_policy` — main entry point.
* :func:`load_policy` — load a :class:`VerifyPolicy` from a JSON file.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from typing import List, Optional

from .canonical import canonical_json
from .trace_reader import (
    TraceVerificationError,
    Trace_,
    verify_trace,
)


# ------------------------------------------------------------------ types


@dataclass
class PolicyViolation:
    """A single finding produced during policy-gated verification."""

    check: str
    """Which check produced this finding: ``"crypto"``, ``"schema"``,
    ``"canonical_bytes"``, or ``"recorder_identity"``."""

    message: str
    """Human-readable description of the violation."""

    severity: str = "error"
    """``"error"`` (fails :attr:`PolicyCheckResult.ok`) or ``"warning"``
    (demoted to warning when ``strict=False``).  In strict mode all
    warnings are promoted to errors."""

    def __str__(self) -> str:
        return f"[{self.severity}] {self.check}: {self.message}"


@dataclass
class PolicyCheckResult:
    """Aggregated result of all enabled policy checks."""

    ok: bool
    """``True`` iff no *error*-severity violation was produced."""

    violations: List[PolicyViolation] = field(default_factory=list)
    """All violations, both errors and warnings."""

    trace: Optional[Trace_] = None
    """The parsed :class:`~stepback.trace_reader.Trace_` if crypto check
    succeeded (``None`` on crypto failure)."""

    @property
    def errors(self) -> List[PolicyViolation]:
        """Only error-severity violations."""
        return [v for v in self.violations if v.severity == "error"]

    @property
    def warnings(self) -> List[PolicyViolation]:
        """Only warning-severity violations."""
        return [v for v in self.violations if v.severity == "warning"]


# ---------------------------------------------------------- policy dataclass


@dataclass
class _CheckConfig:
    enabled: bool = True


@dataclass
class _RecorderIdentityConfig:
    enabled: bool = True
    allowed_public_keys: List[str] = field(default_factory=list)
    allowed_recorder_versions: List[str] = field(default_factory=list)


@dataclass
class VerifyPolicy:
    """Declarative policy for :func:`verify_with_policy`.

    All boolean flags default to enabled-or-unconstrained so that the
    default (no policy file, no ``--strict``) reproduces existing behaviour.
    """

    strict: bool = False
    """If ``True`` warnings are promoted to errors."""

    crypto: _CheckConfig = field(default_factory=_CheckConfig)
    schema: _CheckConfig = field(default_factory=_CheckConfig)
    canonical_bytes: _CheckConfig = field(default_factory=_CheckConfig)
    recorder_identity: _RecorderIdentityConfig = field(
        default_factory=_RecorderIdentityConfig
    )

    @classmethod
    def from_dict(cls, data: dict) -> "VerifyPolicy":
        """Build a :class:`VerifyPolicy` from a plain mapping.

        Unknown top-level keys are silently ignored for forward-compatibility.
        """
        strict = bool(data.get("strict", False))

        def _check(key: str) -> _CheckConfig:
            sub = data.get(key, {})
            if not isinstance(sub, dict):
                sub = {}
            return _CheckConfig(enabled=bool(sub.get("enabled", True)))

        ri_sub = data.get("recorder_identity", {})
        if not isinstance(ri_sub, dict):
            ri_sub = {}
        ri = _RecorderIdentityConfig(
            enabled=bool(ri_sub.get("enabled", True)),
            allowed_public_keys=[
                str(k).lower()
                for k in ri_sub.get("allowed_public_keys", [])
            ],
            allowed_recorder_versions=list(
                ri_sub.get("allowed_recorder_versions", [])
            ),
        )

        return cls(
            strict=strict,
            crypto=_check("crypto"),
            schema=_check("schema"),
            canonical_bytes=_check("canonical_bytes"),
            recorder_identity=ri,
        )

    @classmethod
    def strict_default(cls) -> "VerifyPolicy":
        """All checks enabled, strict mode on."""
        return cls(strict=True)


# ---------------------------------------------------------------- loader


def load_policy(path: str) -> VerifyPolicy:
    """Load a :class:`VerifyPolicy` from a JSON file at *path*.

    Raises :exc:`ValueError` on malformed JSON or missing required fields.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"policy file {path!r} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"policy file {path!r} must be a JSON object, got "
            f"{type(data).__name__}"
        )
    return VerifyPolicy.from_dict(data)


# ----------------------------------------------------- canonical bytes check


def _check_canonical_bytes(path: str) -> List[PolicyViolation]:
    """Verify every frame on disk is exactly the canonical re-encoding.

    Reads raw length-prefixed frames and compares each payload to
    ``canonical_json(json.loads(payload))``.  Returns a list of
    :class:`PolicyViolation` (empty on success).
    """
    violations: List[PolicyViolation] = []
    try:
        with open(path, "rb") as fh:
            idx = 0
            while True:
                ln = fh.read(4)
                if not ln:
                    break
                if len(ln) < 4:
                    violations.append(
                        PolicyViolation(
                            check="canonical_bytes",
                            message=f"frame {idx}: truncated length prefix",
                        )
                    )
                    break
                (n,) = struct.unpack(">I", ln)
                payload = fh.read(n)
                if len(payload) < n:
                    violations.append(
                        PolicyViolation(
                            check="canonical_bytes",
                            message=f"frame {idx}: truncated body (expected {n} bytes)",
                        )
                    )
                    break
                try:
                    decoded = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    # Already caught by verify_trace; skip here.
                    violations.append(
                        PolicyViolation(
                            check="canonical_bytes",
                            message=f"frame {idx}: cannot decode: {exc}",
                        )
                    )
                    idx += 1
                    continue
                expected = canonical_json(decoded)
                if payload != expected:
                    violations.append(
                        PolicyViolation(
                            check="canonical_bytes",
                            message=(
                                f"frame {idx}: on-disk bytes differ from canonical "
                                f"re-encoding (on-disk={len(payload)}B, "
                                f"canonical={len(expected)}B)"
                            ),
                        )
                    )
                idx += 1
    except OSError as exc:
        violations.append(
            PolicyViolation(
                check="canonical_bytes",
                message=f"cannot read trace file: {exc}",
            )
        )
    return violations


# ------------------------------------------------ recorder identity check


def _check_recorder_identity(
    trace_: Trace_,
    cfg: _RecorderIdentityConfig,
) -> List[PolicyViolation]:
    """Verify recorder public-key and version constraints.

    An empty allowlist means no pinning — any value passes.
    """
    violations: List[PolicyViolation] = []
    header = trace_.header

    if cfg.allowed_public_keys:
        actual_key = trace_.public_key_hex.lower()
        if not any(
            actual_key == allowed.lower() for allowed in cfg.allowed_public_keys
        ):
            violations.append(
                PolicyViolation(
                    check="recorder_identity",
                    message=(
                        f"public key {actual_key[:16]}… not in allowed_public_keys"
                    ),
                )
            )

    if cfg.allowed_recorder_versions:
        actual_ver = str(header.get("recorder_version", ""))
        if actual_ver not in cfg.allowed_recorder_versions:
            violations.append(
                PolicyViolation(
                    check="recorder_identity",
                    message=(
                        f"recorder_version {actual_ver!r} not in "
                        f"allowed_recorder_versions "
                        f"{cfg.allowed_recorder_versions!r}"
                    ),
                )
            )

    return violations


# ---------------------------------------------------------- schema check


def _check_schema(path: str) -> List[PolicyViolation]:
    """Run SBTraceSpec schema validation and convert to PolicyViolation list."""
    from .spec import SBTraceSpec, SBTraceVersionError

    violations: List[PolicyViolation] = []
    try:
        spec = SBTraceSpec.for_wire_version("1.0.0")
        report = spec.validate_trace(path)
        for issue in report.errors:
            violations.append(
                PolicyViolation(
                    check="schema",
                    message=issue.message,
                    severity="error",
                )
            )
        for issue in report.warnings:
            violations.append(
                PolicyViolation(
                    check="schema",
                    message=issue.message,
                    severity="warning",
                )
            )
    except SBTraceVersionError as exc:
        violations.append(
            PolicyViolation(check="schema", message=f"spec error: {exc}")
        )
    except Exception as exc:  # noqa: BLE001
        violations.append(
            PolicyViolation(check="schema", message=f"unexpected error: {exc}")
        )
    return violations


# -------------------------------------------------------- main entry point


def verify_with_policy(
    path: str,
    hmac_key: bytes,
    policy: VerifyPolicy,
) -> PolicyCheckResult:
    """Run all enabled checks for *path* under *policy*.

    Returns a :class:`PolicyCheckResult`; never raises unless *path* is not
    a string or *hmac_key* is not bytes — those are programming errors.

    Parameters
    ----------
    path:
        Filesystem path to the ``.sb`` trace file.
    hmac_key:
        HMAC key bytes (as written by the recorder).
    policy:
        Parsed :class:`VerifyPolicy` controlling which checks run and in
        what mode.
    """
    violations: List[PolicyViolation] = []
    trace_: Optional[Trace_] = None

    # 1. Cryptographic verification ----------------------------------------
    if policy.crypto.enabled:
        try:
            trace_ = verify_trace(path, hmac_key)
        except TraceVerificationError as exc:
            violations.append(
                PolicyViolation(check="crypto", message=str(exc))
            )
        except Exception as exc:  # noqa: BLE001
            violations.append(
                PolicyViolation(check="crypto", message=f"unexpected error: {exc}")
            )

    # 2. Schema validation -------------------------------------------------
    if policy.schema.enabled:
        violations.extend(_check_schema(path))

    # 3. Canonical bytes check ---------------------------------------------
    if policy.canonical_bytes.enabled:
        violations.extend(_check_canonical_bytes(path))

    # 4. Recorder identity -------------------------------------------------
    if policy.recorder_identity.enabled and trace_ is not None:
        violations.extend(
            _check_recorder_identity(trace_, policy.recorder_identity)
        )

    # Promote warnings to errors in strict mode.
    if policy.strict:
        for v in violations:
            v.severity = "error"

    ok = not any(v.severity == "error" for v in violations)
    return PolicyCheckResult(ok=ok, violations=violations, trace=trace_)
