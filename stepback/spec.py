"""SB-Trace wire-format SemVer (Step 22 of ``100_STEPS.md``).

The ``stepback`` Python package and the ``.sb`` wire format evolve on
**separate** SemVer tracks. The package version
(:data:`stepback.__version__`) covers the Python API surface; the wire
version defined here covers the bytes inside a ``.sb`` file.

This separation is required because:

* Python releases happen on every bug fix, refactor, or new shim. The
  on-disk format must not be invalidated by any of those.
* Independent implementations (Rust, TypeScript, Go, JVM, .NET, WASM,
  the ``sb proxy``) ship on their own cadences; they all need a single,
  shared version to negotiate against.
* Production incident archives, attestation packs, and the benchmark
  consortium need a SemVer that talks about the *trace*, not about the
  recorder that produced it.

The compatibility contract for the ``sbtrace`` track is:

``sbtrace 1.x``
    Append-only stream of length-prefixed canonical UTF-8 JSON frames
    (sorted keys, no whitespace) with HMAC-SHA256 chaining and per-frame
    Ed25519 signatures. Within ``1.x`` the wire layout is stable; only
    additive, optional fields and capability frames may be introduced
    on minor or patch bumps. Unknown mandatory capabilities still fail
    closed (see Step 24); unknown optional fields are ignored.

``sbtrace 2.0``
    Reserved for the **first breaking encoding** change. The current
    candidate is deterministic CBOR per RFC 8949 §4.2 (see Step 43).
    A 2.0 reader is permitted to read 1.x via a documented translation
    layer, but a 1.x reader is not required to read any 2.x trace.

The integer ``format_version`` field that appears in every ``.sb``
header (see :mod:`stepback.trace_writer`) is the **on-disk**
representation. It maps to a wire SemVer via
:func:`wire_version_for_format_version`. Today only ``format_version=1``
exists; it corresponds to ``sbtrace 1.0.0``.

Public API
----------

The following names are part of the wire-format compatibility contract
and are re-exported from :mod:`stepback`:

* :data:`SBTRACE_WIRE_VERSION` — string, e.g. ``"1.0.0"``.
* :data:`SBTRACE_WIRE_VERSION_INFO` — ``(major, minor, patch)`` tuple.
* :data:`SBTRACE_WIRE_ENCODING` — encoding label, e.g. ``"canonical-json"``.
* :data:`SBTRACE_WIRE_ENCODINGS` — mapping ``major -> encoding label``.
* :func:`parse_wire_version` — parse ``"X.Y.Z"`` into a tuple.
* :func:`is_compatible_reader` — does this build accept a given trace?
* :func:`wire_version_for_format_version` — header int → SemVer string.
* :func:`format_version_for_wire_version` — SemVer string → header int.
* :exc:`SBTraceVersionError` — raised for malformed or unsupported
  wire versions.

These names are stable on the wire-format SemVer track described above,
**not** on the Python package SemVer track. Renaming or removing any of
them is governed by :doc:`/docs/deprecation` plus a wire-format major
bump.
"""
from __future__ import annotations

import glob as _glob
import json
import os
from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

__all__ = [
    # --- Step 22: wire-format SemVer track --------------------------------
    "SBTRACE_WIRE_VERSION",
    "SBTRACE_WIRE_VERSION_INFO",
    "SBTRACE_WIRE_MAJOR",
    "SBTRACE_WIRE_MINOR",
    "SBTRACE_WIRE_PATCH",
    "SBTRACE_WIRE_ENCODING",
    "SBTRACE_WIRE_ENCODINGS",
    "SBTRACE_FORMAT_VERSION_TO_WIRE",
    "SBTraceVersionError",
    "parse_wire_version",
    "is_compatible_reader",
    "wire_version_for_format_version",
    "format_version_for_wire_version",
    # --- Step 23: SBTraceSpec versioned schema validator ------------------
    "SBTRACE_FORMAT_VERSION_TO_ENCODINGS",
    "SBTraceSpec",
    "SBTraceConformanceError",
    "ConformanceIssue",
    "ConformanceReport",
    "current_spec",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Wire-format major version. Bumping this is a breaking encoding change.
SBTRACE_WIRE_MAJOR: int = 1

#: Wire-format minor version. Bumping this signals additive, optional fields
#: or new optional capability frames; old readers must keep working.
SBTRACE_WIRE_MINOR: int = 0

#: Wire-format patch version. Bumping this signals an editorial spec
#: clarification or a recorder bug fix that does not change byte layout.
SBTRACE_WIRE_PATCH: int = 0

#: ``(major, minor, patch)`` tuple form of :data:`SBTRACE_WIRE_VERSION`.
SBTRACE_WIRE_VERSION_INFO: Tuple[int, int, int] = (
    SBTRACE_WIRE_MAJOR,
    SBTRACE_WIRE_MINOR,
    SBTRACE_WIRE_PATCH,
)

#: Canonical ``"X.Y.Z"`` SemVer string for the wire format this build
#: produces. Independent of :data:`stepback.__version__`.
SBTRACE_WIRE_VERSION: str = (
    f"{SBTRACE_WIRE_MAJOR}.{SBTRACE_WIRE_MINOR}.{SBTRACE_WIRE_PATCH}"
)

#: Encoding label for the wire-format major this build produces.
SBTRACE_WIRE_ENCODING: str = "canonical-json"

#: Mapping from wire-format major version to its encoding label.
#:
#: * ``1`` → ``"canonical-json"`` (UTF-8, sorted keys, no whitespace).
#: * ``2`` → ``"deterministic-cbor"`` (RFC 8949 §4.2, candidate; not
#:   yet implemented — reserved entry so readers can produce a clear
#:   error message instead of "unknown major").
SBTRACE_WIRE_ENCODINGS: Mapping[int, str] = {
    1: "canonical-json",
    2: "deterministic-cbor",
}

#: Mapping from the integer ``format_version`` field that appears in
#: every ``.sb`` header to the wire-format SemVer string it represents.
#:
#: Today only ``1`` is defined and it points at ``"1.0.0"``. Future
#: minor/patch bumps within wire-major 1 do **not** allocate new
#: ``format_version`` integers — they are negotiated via capability
#: frames (Step 24). A wire-major bump (``2.0``) **does** allocate a
#: new integer (``format_version=2``) so that old readers detect it
#: at the first byte of the header instead of trying to parse a CBOR
#: payload as JSON.
SBTRACE_FORMAT_VERSION_TO_WIRE: Mapping[int, str] = {
    1: "1.0.0",
    2: "2.0.0",
}

#: Wire encodings allocated per ``format_version``. v1 is canonical
#: UTF-8 JSON. v2 is dual-encoded — a v2 trace MAY carry frames in
#: canonical UTF-8 JSON or in deterministic CBOR (RFC 8949 §4.2),
#: and ``stepback.semantic_hash.semantic_hash`` produces the same
#: cache key for both. See ``spec/sbtrace-v2.md`` for the full
#: definition.
SBTRACE_FORMAT_VERSION_TO_ENCODINGS: Mapping[int, tuple[str, ...]] = {
    1: ("canonical-json",),
    2: ("canonical-json", "deterministic-cbor"),
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SBTraceVersionError(ValueError):
    """Raised when an SB-Trace wire version is malformed or unsupported.

    Subclass of :class:`ValueError` so existing ``except ValueError``
    handlers in importer / verifier code keep working. Always carries a
    human-readable message naming both the offending version and the
    set of versions this build accepts.
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_wire_version(version: str) -> Tuple[int, int, int]:
    """Parse ``"X.Y.Z"`` into a ``(major, minor, patch)`` tuple.

    Only strict, three-component numeric SemVer is accepted. Pre-release
    or build-metadata suffixes (``"1.0.0-rc1"``, ``"1.0.0+gabc"``) are
    rejected — the wire format must not encode unstable variants on the
    public track. Raises :class:`SBTraceVersionError` on any other
    input.
    """

    if not isinstance(version, str):
        raise SBTraceVersionError(
            f"sbtrace wire version must be a str, got {type(version).__name__}"
        )
    parts = version.split(".")
    if len(parts) != 3:
        raise SBTraceVersionError(
            f"sbtrace wire version must be 'MAJOR.MINOR.PATCH', got {version!r}"
        )
    try:
        major, minor, patch = (int(p) for p in parts)
    except ValueError as exc:
        raise SBTraceVersionError(
            f"sbtrace wire version components must be integers, got {version!r}"
        ) from exc
    if major < 0 or minor < 0 or patch < 0:
        raise SBTraceVersionError(
            f"sbtrace wire version components must be non-negative, got {version!r}"
        )
    return (major, minor, patch)


def is_compatible_reader(version: str) -> bool:
    """Return ``True`` iff this build can read traces tagged ``version``.

    The compatibility rule mirrors SemVer: a reader at
    ``MAJOR.MINOR.PATCH`` accepts any trace whose wire major equals
    ``MAJOR`` and whose ``(minor, patch)`` is less than or equal to
    ``(MINOR, PATCH)``. Future minor/patch bumps within the same major
    add only additive, optional capabilities so an older reader handles
    them gracefully — but this *function* is conservative on purpose: it
    only claims compatibility for versions this build already knows
    about. A newer-minor trace from a future recorder must be checked by
    a newer reader (or by the capability-negotiation frame defined in
    Step 24), not by this build.

    Raises :class:`SBTraceVersionError` if ``version`` is malformed.
    """

    major, minor, patch = parse_wire_version(version)
    if major != SBTRACE_WIRE_MAJOR:
        return False
    if (minor, patch) > (SBTRACE_WIRE_MINOR, SBTRACE_WIRE_PATCH):
        return False
    return True


def wire_version_for_format_version(format_version: int) -> str:
    """Translate the on-disk ``format_version`` int into a wire SemVer.

    The header of every ``.sb`` file carries an integer
    ``format_version`` (currently ``1``). That integer is the on-disk
    representation; the SemVer string returned here is the human-facing
    form used in spec text, attestation receipts, error messages, and
    the SB-Trace conformance suite.

    Raises :class:`SBTraceVersionError` for unknown integers so callers
    fail closed instead of silently treating an unknown trace as
    compatible.
    """

    if not isinstance(format_version, int) or isinstance(format_version, bool):
        raise SBTraceVersionError(
            "format_version must be an int, got "
            f"{type(format_version).__name__}"
        )
    try:
        return SBTRACE_FORMAT_VERSION_TO_WIRE[format_version]
    except KeyError:
        known = sorted(SBTRACE_FORMAT_VERSION_TO_WIRE)
        raise SBTraceVersionError(
            f"unknown sbtrace format_version={format_version!r}; "
            f"this build knows {known}"
        ) from None


def format_version_for_wire_version(version: str) -> int:
    """Translate a wire SemVer string into the on-disk ``format_version``.

    Inverse of :func:`wire_version_for_format_version`. The on-disk int
    is allocated per **wire major**, so any ``1.y.z`` maps to
    ``format_version=1``. Raises :class:`SBTraceVersionError` for
    unknown majors.
    """

    major, _minor, _patch = parse_wire_version(version)
    for fv, wire in SBTRACE_FORMAT_VERSION_TO_WIRE.items():
        if parse_wire_version(wire)[0] == major:
            return fv
    raise SBTraceVersionError(
        f"no on-disk format_version is allocated for sbtrace major {major}; "
        f"known majors: {sorted({parse_wire_version(v)[0] for v in SBTRACE_FORMAT_VERSION_TO_WIRE.values()})}"
    )


# ---------------------------------------------------------------------------
# Step 23: SBTraceSpec — versioned schema loader + conformance validator
# ---------------------------------------------------------------------------
#
# ``SBTraceSpec`` is the in-memory representation of the SB-Trace wire-format
# *schema* at a given wire-version. ``parse_wire_version`` and
# :data:`SBTRACE_FORMAT_VERSION_TO_WIRE` answer "which version is this trace?";
# ``SBTraceSpec`` answers "given that version, what fields are required, what
# frame kinds and step kinds are legal, and which capabilities does this build
# support?". It is the loader hook the README advertises::
#
#     SBTraceSpec.load("sbtrace-v1.0.rfc.yaml").assert_conformant("./traces/*.sb")
#
# Today only one version (``"1.0.0"``) is shipped; the loader accepts JSON or
# YAML files that override individual fields so external implementations can
# pin or extend a spec without forking the package.
#
# The validator is intentionally *schema-level only*: it does not re-verify
# the HMAC chain or Ed25519 signatures (that is :func:`stepback.trace_reader.
# verify_trace`'s job). Conformance and crypto verification are independent
# axes — a trace can be cryptographically intact while declaring an unknown
# wire major (fail closed on schema), and a trace can match every field
# constraint while being tampered with at the byte level (fail closed on
# crypto). Production audit pipelines should call both.

# Wrapper-frame (envelope) keys are the same for every wire-1.x frame.
_WRAPPER_REQUIRED: FrozenSet[str] = frozenset(
    {"body", "prev_hmac", "hmac", "sig"}
)

# Header keys the v1.0.0 recorder writes (see stepback/trace_writer.py).
_HEADER_REQUIRED_V1: FrozenSet[str] = frozenset(
    {
        "type",
        "magic",
        "format_version",
        "recorder_version",
        "canonicalisation_version",
        "public_key",
        "hmac_key_id",
        "price_list_version",
        "wallclock_ns",
    }
)
_HEADER_OPTIONAL_V1: FrozenSet[str] = frozenset(
    {"compression", "blob_threshold", "blob_min_reuse"}
)

# Step body keys. ``step_id`` and ``step_kind`` are the only universally
# required fields — the rest depend on the ``step_kind``.
_STEP_REQUIRED_V1: FrozenSet[str] = frozenset({"step_id", "step_kind"})
_STEP_OPTIONAL_V1: FrozenSet[str] = frozenset(
    {
        "name",
        "parent_step_id",
        "parent_step_ids",
        "inputs",
        "outputs",
        "inputs_hash",
        "outputs_hash",
        "nondeterminism",
        "nondeterminism_hash",
        "wallclock_ns",
        "cpu_ns",
        "cost_usd",
        "llm_request",
        "llm_response",
        "tool_name",
        "policy_decision",
        "router_decision",
        "branch_id",
        "branch_ids",
        "exception",
        "metadata",
        "tags",
    }
)

# Closed set of step kinds the v1.0.0 recorder emits. Mirrors
# :class:`stepback.step_types.StepKind` — keep in sync.
_STEP_KINDS_V1: FrozenSet[str] = frozenset(
    {
        "llm_call",
        "tool_call",
        "router",
        "policy_check",
        "mcp_call",
        "parallel_branch_open",
        "parallel_branch_join",
        "exception",
    }
)

# Frame kinds (the ``body.type`` discriminator).
_FRAME_KINDS_V1: FrozenSet[str] = frozenset(
    {"header", "step", "blob", "tail", "capability", "merkle_summary"}
)

# Capabilities this build understands by name. Capability frames are
# defined in Step 24; the v1.0.0 spec already reserves the names below so
# that fail-closed checks have something to compare declared mandatory
# capabilities against. ``"core"`` is the implicit minimum.
_SUPPORTED_CAPABILITIES_V1: FrozenSet[str] = frozenset(
    {
        "core",
        "blobs",
        "gzip-step-bodies",
        "ed25519-receipts",
        "hmac-sha256-chain",
        # Step 52: optional end-of-trace Merkle summary frame.
        "merkle-summary-v1",
    }
)

_BUILTIN_SPECS: Dict[str, "SBTraceSpec"] = {}


class SBTraceConformanceError(Exception):
    """Raised by :meth:`SBTraceSpec.assert_conformant` on any error.

    The exception carries the full :class:`ConformanceReport` on
    :attr:`report` so callers can inspect every individual issue
    after catching::

        try:
            spec.assert_conformant("traces/*.sb")
        except SBTraceConformanceError as exc:
            for issue in exc.report.errors:
                log.error("%s frame %s: %s", issue.path, issue.frame_index, issue.message)
    """

    def __init__(self, message: str, report: "ConformanceReport") -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class ConformanceIssue:
    """One schema-level finding produced by :class:`SBTraceSpec` validation.

    Issues carry a stable :attr:`code` for programmatic dispatch and a
    human-readable :attr:`message`. ``severity`` is ``"error"`` for any
    violation that makes the trace non-conformant and ``"warning"`` for
    forward-compatible deviations (an unknown optional field, an unknown
    step kind, a recorder-version drift) that newer recorders may
    legitimately emit.
    """

    code: str
    message: str
    severity: str = "error"
    path: Optional[str] = None
    frame_index: Optional[int] = None
    field_name: Optional[str] = None

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        loc_parts = []
        if self.path is not None:
            loc_parts.append(self.path)
        if self.frame_index is not None:
            loc_parts.append(f"frame[{self.frame_index}]")
        if self.field_name is not None:
            loc_parts.append(self.field_name)
        loc = ":".join(loc_parts)
        return f"[{self.severity}] {self.code}: {self.message}" + (
            f" ({loc})" if loc else ""
        )


@dataclass
class ConformanceReport:
    """Result of validating one or more ``.sb`` files against a spec.

    A report is *conformant* iff :attr:`errors` is empty. Warnings do not
    invalidate the trace — they exist so future capability frames or
    recorder versions can be surfaced without forcing a fail-closed.
    """

    spec_version: str
    paths: List[str] = field(default_factory=list)
    issues: List[ConformanceIssue] = field(default_factory=list)
    unsupported_capabilities: List[str] = field(default_factory=list)

    @property
    def errors(self) -> List[ConformanceIssue]:
        """All issues with severity ``"error"``."""
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ConformanceIssue]:
        """All issues with severity ``"warning"``."""
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def is_conformant(self) -> bool:
        """``True`` iff there are no error-severity issues."""
        return not self.errors

    def add(self, issue: ConformanceIssue) -> None:
        """Append a single :class:`ConformanceIssue` to the report."""
        self.issues.append(issue)

    def extend(self, issues: Iterable[ConformanceIssue]) -> None:
        """Append every issue in ``issues`` to the report."""
        self.issues.extend(issues)

    def summary(self) -> str:
        """One-line human-readable summary, useful for CI output."""
        return (
            f"sbtrace {self.spec_version}: "
            f"{len(self.paths)} file(s), "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s), "
            f"{len(self.unsupported_capabilities)} unsupported capability(ies)"
        )


@dataclass(frozen=True)
class SBTraceSpec:
    """In-memory schema for one SB-Trace wire-format version.

    The shipped spec for ``"1.0.0"`` is :func:`current_spec`. Custom
    specs come from :meth:`load` (JSON/YAML override file) or
    :meth:`from_mapping` (programmatic). All collection fields are
    frozensets so spec instances are safe to share across threads and
    use as dict keys.

    Validation methods (:meth:`validate_wrapper`, :meth:`validate_header`,
    :meth:`validate_step`, :meth:`validate_blob`, :meth:`validate_frame`)
    return lists of :class:`ConformanceIssue`; :meth:`validate_trace`
    walks an entire ``.sb`` file and produces a :class:`ConformanceReport`;
    :meth:`assert_conformant` raises :exc:`SBTraceConformanceError` when
    any error is found.
    """

    wire_version: str
    format_version: int
    encoding: str
    wrapper_required: FrozenSet[str]
    header_required: FrozenSet[str]
    header_optional: FrozenSet[str]
    step_required: FrozenSet[str]
    step_optional: FrozenSet[str]
    step_kinds: FrozenSet[str]
    frame_kinds: FrozenSet[str]
    supported_capabilities: FrozenSet[str]
    magic: str = "stepback/.sb"
    strict_unknown_step_kinds: bool = False
    strict_unknown_optional_fields: bool = False

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def for_wire_version(cls, version: str) -> "SBTraceSpec":
        """Return the built-in spec for ``version`` (e.g. ``"1.0.0"``).

        Raises :exc:`SBTraceVersionError` if the version is malformed
        or no built-in spec is registered for it.
        """
        parse_wire_version(version)  # validate shape
        try:
            return _BUILTIN_SPECS[version]
        except KeyError:
            known = sorted(_BUILTIN_SPECS)
            raise SBTraceVersionError(
                f"no built-in SBTraceSpec for wire version {version!r}; "
                f"this build ships {known}"
            ) from None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SBTraceSpec":
        """Build an :class:`SBTraceSpec` from a plain mapping.

        The mapping must carry a ``wire_version`` key; every other field
        is optional and falls back to the built-in spec for that wire
        version. Frozenset fields accept any iterable in the mapping.
        """
        if "wire_version" not in data:
            raise SBTraceVersionError(
                "spec mapping must contain 'wire_version'"
            )
        wire_version = str(data["wire_version"])
        try:
            base = cls.for_wire_version(wire_version)
        except SBTraceVersionError:
            base = cls._minimal_for_unknown_wire(wire_version)

        def _frozen(key: str, fallback: FrozenSet[str]) -> FrozenSet[str]:
            if key in data and data[key] is not None:
                value = data[key]
                if isinstance(value, str) or not hasattr(value, "__iter__"):
                    raise SBTraceVersionError(
                        f"spec field {key!r} must be an iterable of strings"
                    )
                return frozenset(str(v) for v in value)
            return fallback

        format_version = int(data.get("format_version", base.format_version))
        encoding = str(data.get("encoding", base.encoding))
        return cls(
            wire_version=wire_version,
            format_version=format_version,
            encoding=encoding,
            wrapper_required=_frozen("wrapper_required", base.wrapper_required),
            header_required=_frozen("header_required", base.header_required),
            header_optional=_frozen("header_optional", base.header_optional),
            step_required=_frozen("step_required", base.step_required),
            step_optional=_frozen("step_optional", base.step_optional),
            step_kinds=_frozen("step_kinds", base.step_kinds),
            frame_kinds=_frozen("frame_kinds", base.frame_kinds),
            supported_capabilities=_frozen(
                "supported_capabilities", base.supported_capabilities
            ),
            magic=str(data.get("magic", base.magic)),
            strict_unknown_step_kinds=bool(
                data.get("strict_unknown_step_kinds", base.strict_unknown_step_kinds)
            ),
            strict_unknown_optional_fields=bool(
                data.get(
                    "strict_unknown_optional_fields",
                    base.strict_unknown_optional_fields,
                )
            ),
        )

    @classmethod
    def _minimal_for_unknown_wire(cls, wire_version: str) -> "SBTraceSpec":
        """Fallback for ``from_mapping`` when wire_version isn't built-in.

        Used so an external spec file can introduce a new wire version
        (e.g. ``"1.1.0"``) without first patching this module. The
        defaults are deliberately conservative — they require every
        v1-shaped field and reject every step kind unless the spec file
        explicitly relaxes things.
        """
        major = parse_wire_version(wire_version)[0]
        return cls(
            wire_version=wire_version,
            format_version=major,
            encoding=SBTRACE_WIRE_ENCODINGS.get(major, "canonical-json"),
            wrapper_required=_WRAPPER_REQUIRED,
            header_required=_HEADER_REQUIRED_V1,
            header_optional=_HEADER_OPTIONAL_V1,
            step_required=_STEP_REQUIRED_V1,
            step_optional=_STEP_OPTIONAL_V1,
            step_kinds=frozenset(),
            frame_kinds=_FRAME_KINDS_V1,
            supported_capabilities=frozenset({"core"}),
        )

    @classmethod
    def load(cls, path: Union[str, "os.PathLike[str]"]) -> "SBTraceSpec":
        """Load a spec override file (JSON or YAML) from disk.

        File extensions ``.json`` is read with the stdlib; ``.yaml`` /
        ``.yml`` requires the optional ``pyyaml`` dependency (declared
        in the ``bench`` extra). Unknown extensions are tried as JSON
        first, then YAML.

        The file's top-level mapping is passed verbatim to
        :meth:`from_mapping`, so it must include ``wire_version`` and
        may override any other field. A fully empty file is rejected
        — load it with :meth:`for_wire_version` instead.
        """
        path_str = os.fspath(path)
        with open(path_str, "rb") as f:
            raw = f.read()
        if not raw.strip():
            raise SBTraceVersionError(
                f"spec file {path_str!r} is empty; "
                "use SBTraceSpec.for_wire_version(...) for the built-in spec"
            )
        ext = os.path.splitext(path_str)[1].lower()
        data: Any
        if ext == ".json":
            data = json.loads(raw.decode("utf-8"))
        elif ext in (".yaml", ".yml"):
            data = _load_yaml(raw, path_str)
        else:
            try:
                data = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                data = _load_yaml(raw, path_str)
        if not isinstance(data, Mapping):
            raise SBTraceVersionError(
                f"spec file {path_str!r} must contain a mapping at the top level, "
                f"got {type(data).__name__}"
            )
        return cls.from_mapping(data)

    # ------------------------------------------------------------------
    # Capability negotiation
    # ------------------------------------------------------------------

    def unsupported_capabilities(
        self, declared: Iterable[str]
    ) -> List[str]:
        """Return the subset of ``declared`` this build does not support.

        Used to implement Step 24's fail-closed rule: any *mandatory*
        capability declared by a trace that isn't in
        :attr:`supported_capabilities` makes the trace non-conformant
        for this reader.
        """
        return [c for c in declared if c not in self.supported_capabilities]

    def supports_capability(self, capability: str) -> bool:
        """``True`` iff ``capability`` is in :attr:`supported_capabilities`."""
        return capability in self.supported_capabilities

    # ------------------------------------------------------------------
    # Frame validators
    # ------------------------------------------------------------------

    def validate_wrapper(
        self,
        wrapper: Mapping[str, Any],
        *,
        path: Optional[str] = None,
        frame_index: Optional[int] = None,
    ) -> List[ConformanceIssue]:
        """Validate the on-disk envelope (``body`` + ``prev_hmac`` + …).

        Schema-level only: no HMAC recomputation, no signature check.
        """
        issues: List[ConformanceIssue] = []
        if not isinstance(wrapper, Mapping):
            issues.append(
                ConformanceIssue(
                    code="wrapper.not_mapping",
                    message=(
                        "frame wrapper must be a JSON object, got "
                        f"{type(wrapper).__name__}"
                    ),
                    path=path,
                    frame_index=frame_index,
                )
            )
            return issues
        missing = self.wrapper_required - wrapper.keys()
        for name in sorted(missing):
            issues.append(
                ConformanceIssue(
                    code="wrapper.missing_field",
                    message=f"wrapper is missing required field {name!r}",
                    path=path,
                    frame_index=frame_index,
                    field_name=name,
                )
            )
        sig = wrapper.get("sig")
        if sig is not None and not (
            isinstance(sig, str) and sig.startswith("ed25519:")
        ):
            issues.append(
                ConformanceIssue(
                    code="wrapper.bad_signature_scheme",
                    message=(
                        f"sig must start with 'ed25519:' in wire {self.wire_version!r}, "
                        f"got {sig!r}"
                    ),
                    path=path,
                    frame_index=frame_index,
                    field_name="sig",
                )
            )
        return issues

    def validate_header(
        self,
        header: Mapping[str, Any],
        *,
        path: Optional[str] = None,
        frame_index: Optional[int] = None,
    ) -> List[ConformanceIssue]:
        """Validate the header frame body."""
        issues: List[ConformanceIssue] = []
        if not isinstance(header, Mapping):
            issues.append(
                ConformanceIssue(
                    code="header.not_mapping",
                    message="header body must be a JSON object",
                    path=path,
                    frame_index=frame_index,
                )
            )
            return issues
        if header.get("type") != "header":
            issues.append(
                ConformanceIssue(
                    code="header.wrong_type",
                    message=(
                        "header body type must be 'header', got "
                        f"{header.get('type')!r}"
                    ),
                    path=path,
                    frame_index=frame_index,
                    field_name="type",
                )
            )
        if header.get("magic") != self.magic:
            issues.append(
                ConformanceIssue(
                    code="header.bad_magic",
                    message=(
                        f"magic must be {self.magic!r}, got {header.get('magic')!r}"
                    ),
                    path=path,
                    frame_index=frame_index,
                    field_name="magic",
                )
            )
        fv = header.get("format_version")
        if fv != self.format_version:
            issues.append(
                ConformanceIssue(
                    code="header.format_version_mismatch",
                    message=(
                        f"format_version must be {self.format_version}, got {fv!r}"
                    ),
                    path=path,
                    frame_index=frame_index,
                    field_name="format_version",
                )
            )
        for name in sorted(self.header_required - header.keys()):
            issues.append(
                ConformanceIssue(
                    code="header.missing_field",
                    message=f"header is missing required field {name!r}",
                    path=path,
                    frame_index=frame_index,
                    field_name=name,
                )
            )
        known = self.header_required | self.header_optional
        for name in sorted(set(header.keys()) - known):
            severity = "error" if self.strict_unknown_optional_fields else "warning"
            issues.append(
                ConformanceIssue(
                    code="header.unknown_field",
                    message=(
                        f"header carries unknown field {name!r}; "
                        "newer recorders may add fields, treat as warning unless "
                        "strict_unknown_optional_fields=True"
                    ),
                    severity=severity,
                    path=path,
                    frame_index=frame_index,
                    field_name=name,
                )
            )
        return issues

    def validate_step(
        self,
        step: Mapping[str, Any],
        *,
        path: Optional[str] = None,
        frame_index: Optional[int] = None,
    ) -> List[ConformanceIssue]:
        """Validate a step body (already de-blobbed and decompressed)."""
        issues: List[ConformanceIssue] = []
        if not isinstance(step, Mapping):
            issues.append(
                ConformanceIssue(
                    code="step.not_mapping",
                    message="step body must be a JSON object",
                    path=path,
                    frame_index=frame_index,
                )
            )
            return issues
        for name in sorted(self.step_required - step.keys()):
            issues.append(
                ConformanceIssue(
                    code="step.missing_field",
                    message=f"step is missing required field {name!r}",
                    path=path,
                    frame_index=frame_index,
                    field_name=name,
                )
            )
        kind = step.get("step_kind")
        if kind is not None and kind not in self.step_kinds:
            severity = "error" if self.strict_unknown_step_kinds else "warning"
            issues.append(
                ConformanceIssue(
                    code="step.unknown_kind",
                    message=(
                        f"step_kind {kind!r} is not in the wire {self.wire_version} "
                        "closed set; treat as forward-compatible warning unless "
                        "strict_unknown_step_kinds=True"
                    ),
                    severity=severity,
                    path=path,
                    frame_index=frame_index,
                    field_name="step_kind",
                )
            )
        known = self.step_required | self.step_optional
        for name in sorted(set(step.keys()) - known):
            severity = "error" if self.strict_unknown_optional_fields else "warning"
            issues.append(
                ConformanceIssue(
                    code="step.unknown_field",
                    message=(
                        f"step carries unknown field {name!r}; "
                        "newer recorders may add fields, treat as warning unless "
                        "strict_unknown_optional_fields=True"
                    ),
                    severity=severity,
                    path=path,
                    frame_index=frame_index,
                    field_name=name,
                )
            )
        return issues

    def validate_blob(
        self,
        blob: Mapping[str, Any],
        *,
        path: Optional[str] = None,
        frame_index: Optional[int] = None,
    ) -> List[ConformanceIssue]:
        """Validate a content-addressed blob frame body."""
        issues: List[ConformanceIssue] = []
        for name in ("type", "id", "encoding", "data"):
            if name not in blob:
                issues.append(
                    ConformanceIssue(
                        code="blob.missing_field",
                        message=f"blob is missing required field {name!r}",
                        path=path,
                        frame_index=frame_index,
                        field_name=name,
                    )
                )
        enc = blob.get("encoding")
        if enc is not None and enc not in ("json", "gzip+base64"):
            issues.append(
                ConformanceIssue(
                    code="blob.unknown_encoding",
                    message=(
                        f"blob encoding {enc!r} not in wire {self.wire_version} "
                        "set {'json', 'gzip+base64'}"
                    ),
                    severity="warning",
                    path=path,
                    frame_index=frame_index,
                    field_name="encoding",
                )
            )
        return issues

    def validate_frame(
        self,
        wrapper: Mapping[str, Any],
        *,
        path: Optional[str] = None,
        frame_index: Optional[int] = None,
    ) -> List[ConformanceIssue]:
        """Validate one wrapped frame: wrapper + body of the relevant kind."""
        issues = list(
            self.validate_wrapper(wrapper, path=path, frame_index=frame_index)
        )
        body = wrapper.get("body") if isinstance(wrapper, Mapping) else None
        if not isinstance(body, Mapping):
            return issues
        kind = body.get("type")
        if kind not in self.frame_kinds:
            issues.append(
                ConformanceIssue(
                    code="frame.unknown_kind",
                    message=(
                        f"frame body type {kind!r} not in wire {self.wire_version} "
                        f"set {sorted(self.frame_kinds)}"
                    ),
                    path=path,
                    frame_index=frame_index,
                    field_name="type",
                )
            )
            return issues
        if kind == "header":
            issues.extend(
                self.validate_header(body, path=path, frame_index=frame_index)
            )
        elif kind == "step":
            # Step bodies may be wrapped in a gzip envelope; validating
            # the *encoded* shape is enough at the spec layer because the
            # full decode lives in trace_reader. We still surface a
            # warning if neither shape matches.
            if "step" in body:
                issues.extend(
                    self.validate_step(
                        body["step"], path=path, frame_index=frame_index
                    )
                )
            elif body.get("encoding") == "gzip+base64" and "data" in body:
                pass  # opaque at this layer
            else:
                issues.append(
                    ConformanceIssue(
                        code="step.bad_envelope",
                        message=(
                            "step frame must contain either an inline 'step' "
                            "object or 'encoding'+'data'"
                        ),
                        path=path,
                        frame_index=frame_index,
                    )
                )
        elif kind == "blob":
            issues.extend(
                self.validate_blob(body, path=path, frame_index=frame_index)
            )
        elif kind == "tail":
            if "wallclock_ns" not in body:
                issues.append(
                    ConformanceIssue(
                        code="tail.missing_field",
                        message="tail is missing required field 'wallclock_ns'",
                        path=path,
                        frame_index=frame_index,
                        field_name="wallclock_ns",
                    )
                )
        elif kind == "capability":
            # Capability frames (Step 24) carry a "name" and an optional
            # "mandatory" flag; the spec layer enforces the shape and
            # records unsupported mandatory entries on the report later.
            if "name" not in body:
                issues.append(
                    ConformanceIssue(
                        code="capability.missing_field",
                        message="capability frame is missing required field 'name'",
                        path=path,
                        frame_index=frame_index,
                        field_name="name",
                    )
                )
        return issues

    # ------------------------------------------------------------------
    # Trace-level validators
    # ------------------------------------------------------------------

    def validate_trace(
        self,
        path: Union[str, "os.PathLike[str]"],
    ) -> ConformanceReport:
        """Read ``path`` and produce a :class:`ConformanceReport`.

        This does *not* call :func:`stepback.trace_reader.verify_trace`
        — schema and crypto checks are independent. Pair them in audit
        pipelines.
        """
        # Local import to avoid a circular dependency at module load time.
        from .trace_reader import read_frames, TraceVerificationError

        path_str = os.fspath(path)
        report = ConformanceReport(spec_version=self.wire_version, paths=[path_str])
        try:
            frames = read_frames(path_str)
        except TraceVerificationError as exc:
            report.add(
                ConformanceIssue(
                    code="trace.unreadable",
                    message=f"cannot read .sb frames: {exc}",
                    path=path_str,
                )
            )
            return report
        except OSError as exc:
            report.add(
                ConformanceIssue(
                    code="trace.io_error",
                    message=f"OS error reading {path_str!r}: {exc}",
                    path=path_str,
                )
            )
            return report
        if not frames:
            report.add(
                ConformanceIssue(
                    code="trace.empty",
                    message="trace contains no frames",
                    path=path_str,
                )
            )
            return report

        first = frames[0]
        first_body = first.get("body") if isinstance(first, Mapping) else None
        if not isinstance(first_body, Mapping) or first_body.get("type") != "header":
            report.add(
                ConformanceIssue(
                    code="trace.missing_header",
                    message="first frame must be a header frame",
                    path=path_str,
                    frame_index=0,
                )
            )
        else:
            declared_fv = first_body.get("format_version")
            if declared_fv != self.format_version:
                report.add(
                    ConformanceIssue(
                        code="trace.format_version_mismatch",
                        message=(
                            f"trace declares format_version={declared_fv!r}; "
                            f"this spec is for format_version={self.format_version}"
                        ),
                        path=path_str,
                        frame_index=0,
                        field_name="format_version",
                    )
                )

        last = frames[-1]
        last_body = last.get("body") if isinstance(last, Mapping) else None
        if not (isinstance(last_body, Mapping) and last_body.get("type") == "tail"):
            report.add(
                ConformanceIssue(
                    code="trace.missing_tail",
                    message="last frame must be a tail frame",
                    path=path_str,
                    frame_index=len(frames) - 1,
                )
            )

        for idx, wrapper in enumerate(frames):
            report.extend(
                self.validate_frame(
                    wrapper, path=path_str, frame_index=idx
                )
            )
            body = wrapper.get("body") if isinstance(wrapper, Mapping) else None
            if (
                isinstance(body, Mapping)
                and body.get("type") == "capability"
                and bool(body.get("mandatory", False))
            ):
                name = str(body.get("name", ""))
                if name and not self.supports_capability(name):
                    if name not in report.unsupported_capabilities:
                        report.unsupported_capabilities.append(name)
                    report.add(
                        ConformanceIssue(
                            code="capability.unsupported_mandatory",
                            message=(
                                f"trace declares mandatory capability {name!r} "
                                "which this reader does not support; "
                                "treat trace as non-conformant for this build"
                            ),
                            path=path_str,
                            frame_index=idx,
                            field_name="name",
                        )
                    )
        return report

    def assert_conformant(
        self,
        target: Union[str, "os.PathLike[str]", Sequence[Union[str, "os.PathLike[str]"]]],
    ) -> ConformanceReport:
        """Validate one path, a glob, or a sequence of paths.

        Returns the merged :class:`ConformanceReport` on success and
        raises :exc:`SBTraceConformanceError` (carrying the same report
        on its ``.report`` attribute) if any error-severity issue is
        found.

        ``target`` may be:

        * a single ``str`` / path-like — validated directly, with
          :mod:`glob` expansion applied if it contains wildcard chars
          (``*``, ``?``, or ``[``);
        * a non-string sequence of path-likes — each entry is treated
          like the single-string case above.
        """
        paths: List[str] = []
        if isinstance(target, (str, os.PathLike)):
            paths = self._expand_target(target)
        else:
            for entry in target:
                paths.extend(self._expand_target(entry))
        if not paths:
            raise SBTraceConformanceError(
                f"no .sb files matched {target!r}",
                ConformanceReport(spec_version=self.wire_version),
            )
        merged = ConformanceReport(spec_version=self.wire_version, paths=list(paths))
        for p in paths:
            sub = self.validate_trace(p)
            merged.extend(sub.issues)
            for cap in sub.unsupported_capabilities:
                if cap not in merged.unsupported_capabilities:
                    merged.unsupported_capabilities.append(cap)
        if not merged.is_conformant:
            raise SBTraceConformanceError(
                f"{len(merged.errors)} conformance error(s) across "
                f"{len(merged.paths)} file(s) under sbtrace {self.wire_version}",
                merged,
            )
        return merged

    @staticmethod
    def _expand_target(
        target: Union[str, "os.PathLike[str]"],
    ) -> List[str]:
        path_str = os.fspath(target)
        if any(ch in path_str for ch in "*?["):
            return sorted(_glob.glob(path_str))
        return [path_str]


def _load_yaml(raw: bytes, path_str: str) -> Any:
    """Lazy YAML loader; raises a helpful error if PyYAML is missing."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SBTraceVersionError(
            f"YAML spec file {path_str!r} requires PyYAML; "
            "install with `pip install stepback[bench]` "
            "or `pip install pyyaml`."
        ) from exc
    return yaml.safe_load(raw)


# Register the v1.0.0 built-in spec.
_BUILTIN_SPECS["1.0.0"] = SBTraceSpec(
    wire_version="1.0.0",
    format_version=1,
    encoding="canonical-json",
    wrapper_required=_WRAPPER_REQUIRED,
    header_required=_HEADER_REQUIRED_V1,
    header_optional=_HEADER_OPTIONAL_V1,
    step_required=_STEP_REQUIRED_V1,
    step_optional=_STEP_OPTIONAL_V1,
    step_kinds=_STEP_KINDS_V1,
    frame_kinds=_FRAME_KINDS_V1,
    supported_capabilities=_SUPPORTED_CAPABILITIES_V1,
)


def current_spec() -> SBTraceSpec:
    """Return the :class:`SBTraceSpec` for this build's wire version."""
    return SBTraceSpec.for_wire_version(SBTRACE_WIRE_VERSION)
