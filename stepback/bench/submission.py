"""MLPerf-style benchmark submission rules (Step 123).

This module defines the submission rules for the *stepback* benchmark
leaderboard, the :class:`SubmissionManifest` structure that every entrant
must produce, and the :func:`validate_submission` validator that checks a
manifest against those rules.

Submission rules (version 1.0)
-------------------------------
A valid submission must carry:

1. **Frozen code** (:class:`CodeProvenance`) — exact ``stepback`` version,
   an optional git commit SHA, and an optional wheel SHA-256.  At least one
   of ``git_commit`` or ``wheel_sha256`` must be present so reviewers can
   reproduce the exact binary.

2. **Signed trace pack** (:class:`TracePackManifest`) — SHA-256 digest of
   the corpus trace pack (tarball or directory listing), plus a flag
   confirming that ``stepback verify`` passed on every trace in the pack.

3. **Hardware manifest** (:class:`HardwareManifest`) — OS, CPU count, and
   optional CPU model / RAM.  Container image digest if results were produced
   inside a container.

4. **Exact commands** — a non-empty list of verbatim CLI strings used to
   reproduce the benchmark run.  Each entry must be a non-empty string.

5. **Validator output** — the captured stdout/stderr of ``stepback spec test``
   (or equivalent) demonstrating that all conformance fixtures pass.  The
   field must contain the literal text ``"PASSED"`` and must not contain
   ``"FAILED"`` with a non-zero failure count pattern.

6. **Benchmark results** — one or more :class:`~stepback.bench.BenchRunRecord`
   dicts (``schema_version == "1.0"``) covering at least the ``replay-caching``
   or ``storage-compression`` benchmark.

7. **Audit declaration** (:class:`AuditDeclaration`) — submitter name, date,
   and explicit grants of source-access and trace-pack-access rights.
   ``grants_source_access`` and ``grants_trace_pack_access`` must both be
   ``True`` for a submission to be accepted on the public leaderboard.

Quick-start
-----------
::

    from stepback.bench.submission import (
        SubmissionManifest,
        CodeProvenance,
        HardwareManifest,
        TracePackManifest,
        AuditDeclaration,
        validate_submission,
    )

    manifest = SubmissionManifest(
        submission_id="<uuid4>",
        hardware=HardwareManifest.detect(),
        code=CodeProvenance.detect(),
        trace_pack=TracePackManifest(
            corpus_id="synthetic-200-random_step",
            trace_count=200,
            pack_sha256="<sha256 of pack tarball>",
            hmac_key_id=None,
            verified=True,
        ),
        exact_commands=[
            "stepback bench replay-caching --n-steps 200 --n-trials 10 --out result.json"
        ],
        validator_output="..conformance output..  12/12 PASSED",
        bench_results=[record.to_json()],
        audit=AuditDeclaration(
            submitter_name="Alice",
            submitter_email="alice@example.com",
            submitter_organization="Acme Inc.",
            grants_source_access=True,
            grants_trace_pack_access=True,
            submission_date="2026-05-12",
        ),
    )

    result = validate_submission(manifest)
    if result.valid:
        print("Submission is valid!")
    else:
        for err in result.errors:
            print(f"[{err.field}] {err.message}")
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

RULES_VERSION = "1.0"
"""Current submission-rules version.  Bump when breaking changes are made."""

# ---------------------------------------------------------------------------
# Sub-records
# ---------------------------------------------------------------------------


@dataclass
class HardwareManifest:
    """Host hardware used to produce a submission.

    Extended from :class:`~stepback.bench.result_schema.HardwareInfo` with
    optional container information required for reproducible submissions.
    """

    os: str
    """``platform.system()``, e.g. ``'Linux'`` or ``'Darwin'``."""

    cpu_count: Optional[int]
    """Logical core count; ``None`` if unavailable."""

    cpu_model: Optional[str]
    """Best-effort CPU model string; ``None`` if not detectable."""

    ram_gb: Optional[float]
    """Total RAM in gigabytes; ``None`` if not detectable."""

    docker_image_digest: Optional[str] = None
    """``sha256:…`` digest of the container image, if results were produced
    inside a container.  ``None`` for bare-metal / VM runs."""

    container_runtime: Optional[str] = None
    """E.g. ``"docker 24.0.5"`` or ``"podman 4.6.0"``; ``None`` if not
    containerised."""

    def to_json(self) -> dict:
        return {
            "os": self.os,
            "cpu_count": self.cpu_count,
            "cpu_model": self.cpu_model,
            "ram_gb": self.ram_gb,
            "docker_image_digest": self.docker_image_digest,
            "container_runtime": self.container_runtime,
        }

    @classmethod
    def from_json(cls, d: dict) -> "HardwareManifest":
        return cls(
            os=str(d.get("os", "")),
            cpu_count=d.get("cpu_count"),
            cpu_model=d.get("cpu_model"),
            ram_gb=d.get("ram_gb"),
            docker_image_digest=d.get("docker_image_digest"),
            container_runtime=d.get("container_runtime"),
        )

    @classmethod
    def detect(cls) -> "HardwareManifest":
        """Capture hardware information from the running host.

        Container fields are left ``None`` unless ``DOCKER_IMAGE_DIGEST`` /
        ``CONTAINER_RUNTIME`` environment variables are set by the caller's
        runner script.
        """
        return cls(
            os=platform.system(),
            cpu_count=os.cpu_count(),
            cpu_model=_detect_cpu_model(),
            ram_gb=_detect_ram_gb(),
            docker_image_digest=os.environ.get("DOCKER_IMAGE_DIGEST"),
            container_runtime=os.environ.get("CONTAINER_RUNTIME"),
        )


@dataclass
class CodeProvenance:
    """Exact code version used to produce a submission.

    At least one of ``git_commit`` or ``wheel_sha256`` must be non-``None``
    so a reviewer can reproduce the exact binary.
    """

    stepback_version: str
    """``stepback.__version__``."""

    python_version: str
    """``sys.version``."""

    python_implementation: str
    """``platform.python_implementation()``."""

    git_commit: Optional[str] = None
    """Full 40-character hex SHA-1 of the HEAD commit, if available."""

    git_dirty: Optional[bool] = None
    """``True`` if the working tree had uncommitted changes.  Submissions
    with ``git_dirty=True`` are accepted as *development* submissions only
    and will be flagged in the leaderboard."""

    wheel_sha256: Optional[str] = None
    """SHA-256 of the installed ``stepback-*.whl`` wheel file; ``None`` if
    installed in editable mode or from source."""

    source_uri: Optional[str] = None
    """Git remote URL or wheel download URL; informational only."""

    def to_json(self) -> dict:
        return {
            "stepback_version": self.stepback_version,
            "python_version": self.python_version,
            "python_implementation": self.python_implementation,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "wheel_sha256": self.wheel_sha256,
            "source_uri": self.source_uri,
        }

    @classmethod
    def from_json(cls, d: dict) -> "CodeProvenance":
        return cls(
            stepback_version=str(d.get("stepback_version", "unknown")),
            python_version=str(d.get("python_version", "")),
            python_implementation=str(d.get("python_implementation", "")),
            git_commit=d.get("git_commit"),
            git_dirty=d.get("git_dirty"),
            wheel_sha256=d.get("wheel_sha256"),
            source_uri=d.get("source_uri"),
        )

    @classmethod
    def detect(cls) -> "CodeProvenance":
        """Capture code provenance from the running interpreter."""
        try:
            from stepback import __version__ as sv  # type: ignore[import]
        except Exception:  # pragma: no cover
            sv = "unknown"

        git_commit: Optional[str] = None
        git_dirty: Optional[bool] = None
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                git_commit = result.stdout.strip() or None
        except Exception:  # pragma: no cover
            pass

        if git_commit:
            try:
                dirt = subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                git_dirty = bool(dirt.stdout.strip()) if dirt.returncode == 0 else None
            except Exception:  # pragma: no cover
                pass

        return cls(
            stepback_version=sv,
            python_version=sys.version,
            python_implementation=platform.python_implementation(),
            git_commit=git_commit,
            git_dirty=git_dirty,
        )


@dataclass
class TracePackManifest:
    """Descriptor for the corpus trace pack used in the submission.

    The ``pack_sha256`` is the hex SHA-256 digest of the trace pack artifact
    (gzip-compressed tar archive, zip archive, or a JSON listing of every
    trace ID + its SHA-256).  Reviewers use this digest to confirm they are
    evaluating the same traces.
    """

    corpus_id: str
    """Corpus identifier, e.g. ``"synthetic-200-random_step"`` or
    ``"swe-bench-verified-v1"``."""

    trace_count: int
    """Number of traces in the pack."""

    pack_sha256: str
    """Hex SHA-256 digest of the pack artifact (see class docstring)."""

    hmac_key_id: Optional[str] = None
    """Identifier of the HMAC signing key, if traces were produced with
    ``--key-id``; ``None`` for unsigned traces."""

    verified: bool = True
    """``True`` if ``stepback verify`` (or the Rust verifier) confirmed
    every trace in the pack before submission.  Must be ``True`` for a
    submission to pass validation."""

    def to_json(self) -> dict:
        return {
            "corpus_id": self.corpus_id,
            "trace_count": self.trace_count,
            "pack_sha256": self.pack_sha256,
            "hmac_key_id": self.hmac_key_id,
            "verified": self.verified,
        }

    @classmethod
    def from_json(cls, d: dict) -> "TracePackManifest":
        return cls(
            corpus_id=str(d.get("corpus_id", "")),
            trace_count=int(d.get("trace_count", 0)),
            pack_sha256=str(d.get("pack_sha256", "")),
            hmac_key_id=d.get("hmac_key_id"),
            verified=bool(d.get("verified", False)),
        )

    @classmethod
    def from_directory(
        cls,
        directory: str,
        corpus_id: str,
        hmac_key_id: Optional[str] = None,
        verified: bool = True,
    ) -> "TracePackManifest":
        """Build a manifest by scanning *directory* for ``.sb`` files.

        The ``pack_sha256`` is the SHA-256 of a deterministic JSON listing
        ``[{"file": "<relpath>", "sha256": "<hex>"}]`` sorted by path, so
        the digest changes if any trace changes or any trace is added/removed.
        """
        entries: List[Dict[str, str]] = []
        for root, _dirs, files in os.walk(directory):
            for fname in sorted(files):
                if not fname.endswith(".sb"):
                    continue
                fpath = os.path.join(root, fname)
                relpath = os.path.relpath(fpath, directory)
                sha = hashlib.sha256()
                with open(fpath, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        sha.update(chunk)
                entries.append({"file": relpath, "sha256": sha.hexdigest()})

        listing_bytes = json.dumps(
            sorted(entries, key=lambda e: e["file"]),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        pack_sha256 = hashlib.sha256(listing_bytes).hexdigest()

        return cls(
            corpus_id=corpus_id,
            trace_count=len(entries),
            pack_sha256=pack_sha256,
            hmac_key_id=hmac_key_id,
            verified=verified,
        )


@dataclass
class AuditDeclaration:
    """Submitter's declaration of audit rights.

    Both ``grants_source_access`` and ``grants_trace_pack_access`` must be
    ``True`` for a submission to appear on the *public* leaderboard.
    Development submissions (``git_dirty=True`` or either grant ``False``)
    are accepted but shown in a separate section.
    """

    submitter_name: str
    """Full name or pseudonym of the submitter / organisation."""

    grants_source_access: bool
    """Submitter will provide full source code on reviewer request."""

    grants_trace_pack_access: bool
    """Submitter will provide the trace pack on reviewer request."""

    submission_date: str
    """ISO 8601 date string (``"YYYY-MM-DD"``), set to today by
    :meth:`today`."""

    submitter_email: Optional[str] = None
    """Contact e-mail; strongly recommended but optional."""

    submitter_organization: Optional[str] = None
    """Institutional or company affiliation, if any."""

    def to_json(self) -> dict:
        return {
            "submitter_name": self.submitter_name,
            "submitter_email": self.submitter_email,
            "submitter_organization": self.submitter_organization,
            "grants_source_access": self.grants_source_access,
            "grants_trace_pack_access": self.grants_trace_pack_access,
            "submission_date": self.submission_date,
        }

    @classmethod
    def from_json(cls, d: dict) -> "AuditDeclaration":
        return cls(
            submitter_name=str(d.get("submitter_name", "")),
            submitter_email=d.get("submitter_email"),
            submitter_organization=d.get("submitter_organization"),
            grants_source_access=bool(d.get("grants_source_access", False)),
            grants_trace_pack_access=bool(d.get("grants_trace_pack_access", False)),
            submission_date=str(d.get("submission_date", "")),
        )

    @classmethod
    def today(
        cls,
        submitter_name: str,
        grants_source_access: bool = True,
        grants_trace_pack_access: bool = True,
        submitter_email: Optional[str] = None,
        submitter_organization: Optional[str] = None,
    ) -> "AuditDeclaration":
        """Convenience constructor with today's UTC date."""
        return cls(
            submitter_name=submitter_name,
            grants_source_access=grants_source_access,
            grants_trace_pack_access=grants_trace_pack_access,
            submission_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            submitter_email=submitter_email,
            submitter_organization=submitter_organization,
        )


# ---------------------------------------------------------------------------
# Top-level manifest
# ---------------------------------------------------------------------------


@dataclass
class SubmissionManifest:
    """Complete MLPerf-style submission manifest.

    Combine all sub-records and produce a self-contained JSON document that
    a leaderboard reviewer can validate offline with
    :func:`validate_submission`.
    """

    rules_version: str
    """Must equal :data:`RULES_VERSION` (``"1.0"``)."""

    submission_id: str
    """UUID4 assigned by the submitter; unique per submission attempt."""

    hardware: HardwareManifest
    code: CodeProvenance
    trace_pack: TracePackManifest

    exact_commands: List[str]
    """Verbatim CLI command(s) used to reproduce the results, in order.
    Each entry must be a non-empty string."""

    validator_output: str
    """Captured stdout/stderr of ``stepback spec test`` (or equivalent)
    demonstrating all conformance fixtures pass."""

    bench_results: List[dict]
    """One or more :class:`~stepback.bench.result_schema.BenchRunRecord`
    dicts (``schema_version == "1.0"``)."""

    audit: AuditDeclaration

    witness_cosignatures: List[dict] = field(default_factory=list)
    """Optional list of witness commitments produced by
    :func:`~stepback.bench.witness_cosigning.sign_trace_pack_commitment`
    and serialised with :meth:`~stepback.bench.witness_cosigning.WitnessCommitment.to_dict`.

    At least one valid commitment from a trusted witness is required for an
    entry to be classified as *public* (rather than *development*) on the
    leaderboard.  Absence of this field is allowed for development
    submissions.
    """

    def to_json(self) -> dict:
        d: dict = {
            "rules_version": self.rules_version,
            "submission_id": self.submission_id,
            "hardware": self.hardware.to_json(),
            "code": self.code.to_json(),
            "trace_pack": self.trace_pack.to_json(),
            "exact_commands": list(self.exact_commands),
            "validator_output": self.validator_output,
            "bench_results": [dict(r) for r in self.bench_results],
            "audit": self.audit.to_json(),
        }
        if self.witness_cosignatures:
            d["witness_cosignatures"] = [dict(c) for c in self.witness_cosignatures]
        return d

    @classmethod
    def from_json(cls, d: dict) -> "SubmissionManifest":
        return cls(
            rules_version=str(d.get("rules_version", "")),
            submission_id=str(d.get("submission_id", "")),
            hardware=HardwareManifest.from_json(d.get("hardware") or {}),
            code=CodeProvenance.from_json(d.get("code") or {}),
            trace_pack=TracePackManifest.from_json(d.get("trace_pack") or {}),
            exact_commands=list(d.get("exact_commands") or []),
            validator_output=str(d.get("validator_output") or ""),
            bench_results=list(d.get("bench_results") or []),
            audit=AuditDeclaration.from_json(d.get("audit") or {}),
            witness_cosignatures=list(d.get("witness_cosignatures") or []),
        )

    @classmethod
    def create(
        cls,
        trace_pack: TracePackManifest,
        exact_commands: List[str],
        validator_output: str,
        bench_results: List[dict],
        audit: AuditDeclaration,
        hardware: Optional[HardwareManifest] = None,
        code: Optional[CodeProvenance] = None,
    ) -> "SubmissionManifest":
        """Build a :class:`SubmissionManifest` with auto-detected provenance.

        *hardware* and *code* default to :meth:`HardwareManifest.detect`
        and :meth:`CodeProvenance.detect` respectively.
        """
        return cls(
            rules_version=RULES_VERSION,
            submission_id=str(uuid.uuid4()),
            hardware=hardware or HardwareManifest.detect(),
            code=code or CodeProvenance.detect(),
            trace_pack=trace_pack,
            exact_commands=exact_commands,
            validator_output=validator_output,
            bench_results=bench_results,
            audit=audit,
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class ValidationError:
    """A single rule violation found by :func:`validate_submission`."""

    field: str
    """Dot-path to the offending field, e.g. ``"audit.grants_source_access"``."""

    message: str
    """Human-readable description of the violation."""

    def to_json(self) -> dict:
        return {"field": self.field, "message": self.message}


@dataclass
class SubmissionValidationResult:
    """Outcome of :func:`validate_submission`.

    A submission is *valid for the public leaderboard* when ``valid`` is
    ``True`` and ``warnings`` is empty.  Submissions with warnings are
    accepted but flagged as *development* entries.
    """

    valid: bool
    """``True`` iff there are zero :class:`ValidationError` items."""

    errors: List[ValidationError]
    """Rule violations that disqualify the submission."""

    warnings: List[str]
    """Non-blocking issues (e.g. ``git_dirty=True``, missing e-mail)."""

    rules_version: str
    """The rules version used during validation."""

    submission_id: Optional[str]
    """Forwarded from :attr:`SubmissionManifest.submission_id`."""

    def to_json(self) -> dict:
        return {
            "valid": self.valid,
            "errors": [e.to_json() for e in self.errors],
            "warnings": list(self.warnings),
            "rules_version": self.rules_version,
            "submission_id": self.submission_id,
        }

    @classmethod
    def from_json(cls, d: dict) -> "SubmissionValidationResult":
        return cls(
            valid=bool(d.get("valid", False)),
            errors=[
                ValidationError(**e) for e in (d.get("errors") or [])
            ],
            warnings=list(d.get("warnings") or []),
            rules_version=str(d.get("rules_version", RULES_VERSION)),
            submission_id=d.get("submission_id"),
        )

    def summary_line(self) -> str:
        """One-line human-readable summary."""
        if self.valid:
            wpart = f" ({len(self.warnings)} warning(s))" if self.warnings else ""
            return f"VALID{wpart}  submission_id={self.submission_id}"
        return (
            f"INVALID  {len(self.errors)} error(s)  "
            f"{len(self.warnings)} warning(s)  "
            f"submission_id={self.submission_id}"
        )


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

_REQUIRED_BENCH_SCHEMA_VERSION = "1.0"
_KNOWN_BENCH_CORPUS_PREFIXES = (
    "synthetic-",
    "swe-bench",
    "gaia",
    "tau-bench",
    "agentbench",
    "osworld",
    "webarena",
    "author-",
)


def validate_submission(
    manifest: SubmissionManifest,
) -> SubmissionValidationResult:
    """Validate *manifest* against the submission rules (version 1.0).

    Returns a :class:`SubmissionValidationResult`.  The caller should check
    ``result.valid`` and inspect ``result.errors`` for details.
    """
    errors: List[ValidationError] = []
    warnings: List[str] = []

    def err(field: str, msg: str) -> None:
        errors.append(ValidationError(field=field, message=msg))

    def warn(msg: str) -> None:
        warnings.append(msg)

    # ---- rules_version ---------------------------------------------------
    if manifest.rules_version != RULES_VERSION:
        err(
            "rules_version",
            f"expected '{RULES_VERSION}', got '{manifest.rules_version}'",
        )

    # ---- submission_id ---------------------------------------------------
    if not manifest.submission_id:
        err("submission_id", "must be a non-empty string (UUID4 recommended)")
    else:
        try:
            uuid.UUID(manifest.submission_id)
        except ValueError:
            warn(
                f"submission_id '{manifest.submission_id}' is not a valid UUID4 "
                "(accepted, but UUID4 is strongly recommended)"
            )

    # ---- code provenance -------------------------------------------------
    code = manifest.code
    if not code.stepback_version or code.stepback_version == "unknown":
        err("code.stepback_version", "must be a known version string")
    if code.git_commit is None and code.wheel_sha256 is None:
        err(
            "code",
            "at least one of git_commit or wheel_sha256 must be present "
            "so reviewers can reproduce the exact binary",
        )
    if code.git_commit is not None:
        if len(code.git_commit) not in (40, 7):
            warn(
                f"code.git_commit '{code.git_commit}' should be a full "
                "40-char SHA-1 or a 7-char short SHA"
            )
    if code.git_dirty:
        warn(
            "code.git_dirty=True: working tree had uncommitted changes; "
            "submission will be listed as a development entry"
        )

    # ---- hardware manifest -----------------------------------------------
    hw = manifest.hardware
    if not hw.os:
        err("hardware.os", "OS must be a non-empty string")
    if hw.cpu_count is not None and hw.cpu_count < 1:
        err("hardware.cpu_count", "must be >= 1 or null")

    # ---- trace pack ------------------------------------------------------
    tp = manifest.trace_pack
    if not tp.corpus_id:
        err("trace_pack.corpus_id", "corpus_id must be non-empty")
    if tp.trace_count < 1:
        err("trace_pack.trace_count", "trace_count must be >= 1")
    if not tp.pack_sha256:
        err("trace_pack.pack_sha256", "pack_sha256 must be non-empty")
    elif len(tp.pack_sha256) != 64:
        err(
            "trace_pack.pack_sha256",
            f"expected 64-char hex SHA-256, got {len(tp.pack_sha256)} chars",
        )
    if not tp.verified:
        err(
            "trace_pack.verified",
            "verified must be True: run 'stepback verify' on every trace "
            "in the pack before submitting",
        )

    # ---- exact commands --------------------------------------------------
    if not manifest.exact_commands:
        err(
            "exact_commands",
            "must be a non-empty list of verbatim CLI command strings",
        )
    else:
        for i, cmd in enumerate(manifest.exact_commands):
            if not isinstance(cmd, str) or not cmd.strip():
                err(
                    f"exact_commands[{i}]",
                    "each command must be a non-empty string",
                )

    # ---- validator output ------------------------------------------------
    vout = manifest.validator_output
    if not vout:
        err("validator_output", "must be non-empty: capture stdout of 'stepback spec test'")
    else:
        if "PASSED" not in vout:
            err(
                "validator_output",
                "must contain the literal text 'PASSED'; "
                "run 'stepback spec test <impl>' and include the full output",
            )
        if "FAILED" in vout:
            warn(
                "validator_output contains 'FAILED'; "
                "ensure all conformance fixtures pass before final submission"
            )

    # ---- bench results ---------------------------------------------------
    if not manifest.bench_results:
        err("bench_results", "at least one BenchRunRecord dict is required")
    else:
        for i, rec in enumerate(manifest.bench_results):
            if not isinstance(rec, dict):
                err(f"bench_results[{i}]", "must be a dict (BenchRunRecord.to_json())")
                continue
            sv = rec.get("schema_version")
            if sv != _REQUIRED_BENCH_SCHEMA_VERSION:
                err(
                    f"bench_results[{i}].schema_version",
                    f"expected '{_REQUIRED_BENCH_SCHEMA_VERSION}', got '{sv}'",
                )
            for req_key in ("run_id", "timestamp_utc", "corpus_id", "trace_count"):
                if req_key not in rec:
                    err(
                        f"bench_results[{i}].{req_key}",
                        f"required key '{req_key}' is missing",
                    )
            if "versions" not in rec:
                warn(f"bench_results[{i}] is missing 'versions'; include VersionInfo")
            if "hardware" not in rec:
                warn(f"bench_results[{i}] is missing 'hardware'; include HardwareInfo")

    # ---- audit declaration -----------------------------------------------
    audit = manifest.audit
    if not audit.submitter_name:
        err("audit.submitter_name", "submitter_name must be non-empty")
    if not audit.submission_date:
        err("audit.submission_date", "submission_date must be non-empty")
    else:
        try:
            datetime.strptime(audit.submission_date, "%Y-%m-%d")
        except ValueError:
            err(
                "audit.submission_date",
                f"must be ISO 8601 'YYYY-MM-DD', got '{audit.submission_date}'",
            )
    if not audit.grants_source_access:
        err(
            "audit.grants_source_access",
            "must be True for public-leaderboard submissions",
        )
    if not audit.grants_trace_pack_access:
        err(
            "audit.grants_trace_pack_access",
            "must be True for public-leaderboard submissions",
        )
    if not audit.submitter_email:
        warn("audit.submitter_email is missing; strongly recommended for contact")

    return SubmissionValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        rules_version=RULES_VERSION,
        submission_id=manifest.submission_id or None,
    )


def validate_submission_json(
    d: dict,
) -> SubmissionValidationResult:
    """Validate a submission expressed as a plain dict (JSON-decoded).

    Convenience wrapper around :func:`validate_submission` that handles
    :class:`KeyError` / type errors by converting them to
    :class:`ValidationError` items.
    """
    try:
        manifest = SubmissionManifest.from_json(d)
    except Exception as exc:  # pragma: no cover
        return SubmissionValidationResult(
            valid=False,
            errors=[ValidationError(field="<manifest>", message=f"Parse error: {exc}")],
            warnings=[],
            rules_version=RULES_VERSION,
            submission_id=None,
        )
    return validate_submission(manifest)


# ---------------------------------------------------------------------------
# Hardware detection helpers (private)
# ---------------------------------------------------------------------------


def _detect_cpu_model() -> Optional[str]:
    sys_name = platform.system()
    if sys_name == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="ascii", errors="replace") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif sys_name == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip() or None
        except Exception:
            pass
    return None


def _detect_ram_gb() -> Optional[float]:
    sys_name = platform.system()
    if sys_name == "Linux":
        try:
            with open("/proc/meminfo", encoding="ascii", errors="replace") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / 1_048_576, 2)
        except (OSError, ValueError):
            pass
    elif sys_name == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return round(int(result.stdout.strip()) / 1_073_741_824, 2)
        except Exception:
            pass
    return None
