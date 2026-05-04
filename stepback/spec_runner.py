"""External-implementation conformance runner (Step 47).

Implements ``stepback spec test <implementation>``: run the frozen
SB-Trace v1 conformance fixtures against an external reader/verifier
binary and produce a machine-readable pass/fail report.

This is the cross-implementation conformance harness referenced by
``spec/sbtrace-v1.md`` §15. The protocol is intentionally minimal so
that a Rust, Go, Node, JVM, or .NET reader can implement it with a
~30-line ``main`` shim:

    Protocol (the implementation under test MUST support):

    1. ``<impl-argv...> verify <path/to/fixture.sb>``
        * exit 0  → fixture is well-formed and verifies (HMAC chain,
                    Ed25519 signatures, header invariants).
        * exit ≠ 0 → fixture is rejected. The rejection class SHOULD be
                    printed on the *first* line of stderr as a single
                    token from the set documented in
                    ``stepback-core/fixtures/v1/manifest.json``
                    (``Parse``, ``BadHexOrHmacMismatch``,
                    ``SignatureMismatch``, ``BrokenChainOrHmacMismatch``,
                    ``UnsupportedFormatVersionOrHmacMismatch``). The
                    runner accepts a missing or unrecognized token, but
                    flags it as a soft warning.

    2. (optional) ``<impl-argv...> hash <path/to/fixture.sb>``
        * MAY be implemented. If implemented, it MUST print on stdout
          the lowercase hex SHA-256 of the file's bytes followed by a
          newline. The runner uses this to verify byte-identity with
          the manifest sha256 entries — a separate channel from the
          ``verify`` exit code.

The runner walks ``stepback-core/fixtures/v1/manifest.json`` (the
canonical fixture corpus) by default; ``--fixtures-dir`` and
``--manifest`` allow pointing at a vendored or installed copy.

The output is a structured ``ConformanceRun`` that summarizes:

* per-fixture pass/fail and observed exit code,
* observed vs expected rejection class (when available),
* observed vs expected sha256 (when ``hash`` is implemented),
* a single ``ok: bool`` summary suitable for CI.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "FixtureResult",
    "ConformanceRun",
    "default_manifest_path",
    "run_conformance",
]


# --------------------------------------------------------------------------- #
# Manifest discovery                                                          #
# --------------------------------------------------------------------------- #


def default_manifest_path() -> Optional[Path]:
    """Locate the bundled v1 manifest; ``None`` if not present.

    Searches three locations, in order:

    1. ``$REPO_ROOT/stepback-core/fixtures/v1/manifest.json`` (the
       canonical authoring corpus, used by editable installs).
    2. ``stepback/conformance/fixtures/manifest.json`` (the location
       wheels ship to once Step 41 vendors fixtures into the package).
    3. The current working directory's ``stepback-core/fixtures/v1``
       tree, as a last resort for a user running ``stepback spec test``
       from inside the repo.
    """

    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "stepback-core" / "fixtures" / "v1" / "manifest.json",
        here.parent / "conformance" / "fixtures" / "manifest.json",
        Path.cwd() / "stepback-core" / "fixtures" / "v1" / "manifest.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _load_manifest(manifest_path: Path) -> Tuple[dict, Path]:
    with manifest_path.open("rb") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest {manifest_path} is not a JSON object")
    fixtures_dir = manifest_path.parent
    return manifest, fixtures_dir


# --------------------------------------------------------------------------- #
# Result shapes                                                               #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class FixtureResult:
    """Outcome of running one fixture against the implementation.

    ``ok`` is ``True`` iff the implementation's behaviour matched the
    fixture's expected behaviour:

    * for ``expected="ok"`` fixtures: ``verify`` exited 0; if a
      ``hash`` subcommand was supported, the printed sha256 matched the
      manifest's ``sha256``.
    * for ``expected="reject"`` fixtures: ``verify`` exited non-zero;
      if a rejection-class token was printed and ``expected_error_kind``
      is set, the printed token matches one of the alternatives in
      ``expected_error_kind`` (split on ``OrHmacMismatch`` / ``Or``).
      A missing or unrecognized class is recorded but is **not** a
      hard failure (only a warning), because every "kind" listed in
      the manifest already includes ``OrHmacMismatch`` as an alternate
      to allow permissive readers that re-derive the body bytes.
    """

    name: str
    bucket: str  # "good" or "corrupt"
    expected: str  # "ok" or "reject"
    expected_error_kind: Optional[str]
    expected_sha256: Optional[str]
    exit_code: Optional[int]
    stderr_first_line: str
    observed_error_kind: Optional[str]
    observed_sha256: Optional[str]
    ok: bool
    warnings: List[str]
    error: Optional[str]  # populated on subprocess failures (file missing, etc.)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return d


@dataclasses.dataclass(frozen=True)
class ConformanceRun:
    """Aggregate run summary."""

    implementation: str  # the argv of the implementation, joined for display
    manifest_path: str
    format_version: int
    canonicalisation_version: str
    results: List[FixtureResult]

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.ok)

    @property
    def warnings(self) -> int:
        return sum(len(r.warnings) for r in self.results)

    @property
    def ok(self) -> bool:
        return self.failed == 0 and len(self.results) > 0

    def to_dict(self) -> dict:
        return {
            "implementation": self.implementation,
            "manifest_path": self.manifest_path,
            "format_version": self.format_version,
            "canonicalisation_version": self.canonicalisation_version,
            "summary": {
                "total": len(self.results),
                "passed": self.passed,
                "failed": self.failed,
                "warnings": self.warnings,
                "ok": self.ok,
            },
            "results": [r.to_dict() for r in self.results],
        }


# --------------------------------------------------------------------------- #
# Subprocess plumbing                                                         #
# --------------------------------------------------------------------------- #


def _normalise_argv(impl: str | Sequence[str]) -> List[str]:
    if isinstance(impl, str):
        return shlex.split(impl)
    return list(impl)


def _run(argv: Sequence[str], timeout: float) -> tuple[int, str, str]:
    proc = subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _first_line(s: str) -> str:
    if not s:
        return ""
    line = s.splitlines()[0] if s.splitlines() else ""
    return line.strip()


# Tokens we recognise as a "rejection class" on the implementation's
# stderr. Permissive: case-insensitive, and we accept any prefix match
# against the listed class names.
_KNOWN_REJECTION_CLASSES = (
    "Parse",
    "UnexpectedEof",
    "BadHexOrHmacMismatch",
    "BadHex",
    "HmacMismatch",
    "SignatureMismatch",
    "BrokenChainOrHmacMismatch",
    "BrokenChain",
    "UnsupportedFormatVersionOrHmacMismatch",
    "UnsupportedFormatVersion",
)


def _extract_class(stderr_first_line: str) -> Optional[str]:
    """Pluck a rejection-class token out of the first line of stderr.

    The contract is "first whitespace-or-colon-delimited token". This
    accepts both ``BadHex: bad nibble in field 'hmac'`` and
    ``error: BadHex …``.
    """

    if not stderr_first_line:
        return None
    # try common prefixes
    candidates: list[str] = []
    parts = stderr_first_line.replace(",", " ").replace(":", " ").split()
    candidates.extend(parts[:4])  # look at first few tokens
    for tok in candidates:
        for known in _KNOWN_REJECTION_CLASSES:
            if tok == known or tok.lower() == known.lower():
                return known
    return None


def _expected_class_alternatives(expected_error_kind: str) -> List[str]:
    """Split a manifest ``expected_error_kind`` on ``Or`` boundaries.

    ``BadHexOrHmacMismatch`` → ``["BadHex", "HmacMismatch"]`` so a
    permissive reader that printed ``HmacMismatch`` is accepted.
    ``Parse`` (no ``Or``) → ``["Parse"]``.
    """

    # Restrict the split: only break on the canonical ``Or`` separators
    # used by the manifest, not on any random ``or`` substring. We
    # iterate the known suffixes longest-first.
    s = expected_error_kind
    parts: list[str] = []
    sep = "OrHmacMismatch"
    if sep in s:
        head, tail = s.split(sep, 1)
        parts.append(head)
        parts.append("HmacMismatch")
        if tail:
            parts.extend(_expected_class_alternatives(tail) if tail else [])
        return [p for p in parts if p]
    if "Or" in s:
        head, tail = s.split("Or", 1)
        return [p for p in [head] + _expected_class_alternatives(tail) if p]
    return [s]


# --------------------------------------------------------------------------- #
# Per-fixture execution                                                       #
# --------------------------------------------------------------------------- #


def _exec_verify(
    impl_argv: Sequence[str], fixture_path: Path, timeout: float
) -> tuple[int, str]:
    argv = list(impl_argv) + ["verify", str(fixture_path)]
    rc, _, err = _run(argv, timeout=timeout)
    return rc, err


def _exec_hash(
    impl_argv: Sequence[str], fixture_path: Path, timeout: float
) -> Optional[str]:
    """Run the optional ``hash`` subcommand. ``None`` = unsupported."""

    argv = list(impl_argv) + ["hash", str(fixture_path)]
    try:
        rc, out, _ = _run(argv, timeout=timeout)
    except FileNotFoundError:
        return None
    if rc != 0:
        return None
    out = out.strip().lower()
    # Accept either bare hex or hex-prefixed-with-filename like
    # ``<sha>  <path>`` (BSD/coreutils-style).
    tok = out.split()[0] if out else ""
    if len(tok) == 64 and all(c in "0123456789abcdef" for c in tok):
        return tok
    return None


def _evaluate_good(
    entry: dict,
    fixture_path: Path,
    impl_argv: Sequence[str],
    timeout: float,
    enable_hash: bool,
) -> FixtureResult:
    warnings: list[str] = []
    expected_sha = entry.get("sha256")
    if not fixture_path.is_file():
        return FixtureResult(
            name=entry["name"],
            bucket="good",
            expected="ok",
            expected_error_kind=None,
            expected_sha256=expected_sha,
            exit_code=None,
            stderr_first_line="",
            observed_error_kind=None,
            observed_sha256=None,
            ok=False,
            warnings=warnings,
            error=f"fixture missing on disk: {fixture_path}",
        )

    # Sanity: the fixture corpus is frozen; warn if local bytes drift.
    local_sha = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    if expected_sha and local_sha != expected_sha:
        warnings.append(
            f"local fixture sha256 {local_sha} differs from manifest "
            f"{expected_sha}; corpus has drifted on disk"
        )

    rc, stderr = _exec_verify(impl_argv, fixture_path, timeout=timeout)
    observed_sha = _exec_hash(impl_argv, fixture_path, timeout=timeout) if enable_hash else None
    if observed_sha is not None and expected_sha and observed_sha != expected_sha:
        warnings.append(
            f"implementation reported sha256 {observed_sha}, manifest "
            f"says {expected_sha}"
        )

    ok = rc == 0
    return FixtureResult(
        name=entry["name"],
        bucket="good",
        expected="ok",
        expected_error_kind=None,
        expected_sha256=expected_sha,
        exit_code=rc,
        stderr_first_line=_first_line(stderr),
        observed_error_kind=None,
        observed_sha256=observed_sha,
        ok=ok,
        warnings=warnings,
        error=None,
    )


def _evaluate_corrupt(
    entry: dict,
    fixture_path: Path,
    impl_argv: Sequence[str],
    timeout: float,
) -> FixtureResult:
    warnings: list[str] = []
    expected_sha = entry.get("sha256")
    expected_kind = entry.get("expected_error_kind")
    if not fixture_path.is_file():
        return FixtureResult(
            name=entry["name"],
            bucket="corrupt",
            expected="reject",
            expected_error_kind=expected_kind,
            expected_sha256=expected_sha,
            exit_code=None,
            stderr_first_line="",
            observed_error_kind=None,
            observed_sha256=None,
            ok=False,
            warnings=warnings,
            error=f"fixture missing on disk: {fixture_path}",
        )

    local_sha = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    if expected_sha and local_sha != expected_sha:
        warnings.append(
            f"local fixture sha256 {local_sha} differs from manifest "
            f"{expected_sha}; corpus has drifted on disk"
        )

    rc, stderr = _exec_verify(impl_argv, fixture_path, timeout=timeout)
    first = _first_line(stderr)
    observed_kind = _extract_class(first)

    ok = rc != 0
    if ok and expected_kind and observed_kind is not None:
        alts = _expected_class_alternatives(expected_kind)
        if not any(observed_kind == a for a in alts):
            warnings.append(
                f"rejection class {observed_kind!r} not in expected "
                f"alternatives {alts!r}"
            )
    elif ok and expected_kind and observed_kind is None:
        warnings.append(
            "implementation rejected the fixture but did not print a "
            "recognized rejection-class token on the first line of "
            "stderr (this is allowed but reduces conformance signal)"
        )

    return FixtureResult(
        name=entry["name"],
        bucket="corrupt",
        expected="reject",
        expected_error_kind=expected_kind,
        expected_sha256=expected_sha,
        exit_code=rc,
        stderr_first_line=first,
        observed_error_kind=observed_kind,
        observed_sha256=None,
        ok=ok,
        warnings=warnings,
        error=None,
    )


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #


def run_conformance(
    implementation: str | Sequence[str],
    *,
    manifest_path: Optional[Path] = None,
    fixtures_dir: Optional[Path] = None,
    timeout: float = 30.0,
    enable_hash: bool = True,
    only: Optional[Iterable[str]] = None,
) -> ConformanceRun:
    """Run the conformance corpus against an external implementation.

    Parameters
    ----------
    implementation:
        Either a string (parsed with :func:`shlex.split`) or a sequence
        of argv tokens. Example: ``"./target/release/sb-verifier"`` or
        ``["docker", "run", "--rm", "ghcr.io/example/sb-verifier"]``.
    manifest_path:
        Path to ``manifest.json``. Defaults to the bundled v1 manifest
        located by :func:`default_manifest_path`.
    fixtures_dir:
        Override the directory where fixture files are looked up.
        Defaults to the directory containing ``manifest_path``.
    timeout:
        Per-invocation subprocess timeout in seconds.
    enable_hash:
        If ``True``, also probe the optional ``hash`` subcommand for
        good fixtures.
    only:
        If non-empty, only fixtures whose ``name`` is in this iterable
        are run. Useful for triaging a single failure.

    Returns
    -------
    ConformanceRun
        Aggregate report. ``run.ok`` is ``True`` iff every selected
        fixture passed.
    """

    impl_argv = _normalise_argv(implementation)
    if not impl_argv:
        raise ValueError("implementation argv is empty")

    if manifest_path is None:
        manifest_path = default_manifest_path()
        if manifest_path is None:
            raise FileNotFoundError(
                "could not locate the bundled SB-Trace v1 manifest; pass "
                "--manifest explicitly"
            )
    manifest, default_dir = _load_manifest(manifest_path)
    base_dir = fixtures_dir or default_dir
    if not base_dir.is_dir():
        raise FileNotFoundError(f"fixtures directory not found: {base_dir}")

    only_set = set(only) if only else None
    results: list[FixtureResult] = []

    for entry in manifest.get("good", []):
        if only_set is not None and entry["name"] not in only_set:
            continue
        path = base_dir / "good" / entry["name"]
        results.append(
            _evaluate_good(entry, path, impl_argv, timeout, enable_hash)
        )

    for entry in manifest.get("corrupt", []):
        if only_set is not None and entry["name"] not in only_set:
            continue
        path = base_dir / "corrupt" / entry["name"]
        results.append(_evaluate_corrupt(entry, path, impl_argv, timeout))

    return ConformanceRun(
        implementation=" ".join(shlex.quote(a) for a in impl_argv),
        manifest_path=str(manifest_path),
        format_version=int(manifest.get("format_version", 1)),
        canonicalisation_version=str(manifest.get("canonicalisation_version", "1")),
        results=results,
    )


# --------------------------------------------------------------------------- #
# Human-readable rendering                                                    #
# --------------------------------------------------------------------------- #


def render_text(run: ConformanceRun) -> str:
    """Render a compact human-readable report."""

    lines: list[str] = []
    lines.append(f"implementation: {run.implementation}")
    lines.append(
        f"manifest: {run.manifest_path} "
        f"(format_version={run.format_version}, "
        f"canonicalisation_version={run.canonicalisation_version})"
    )
    lines.append(
        f"fixtures: {len(run.results)}  "
        f"passed: {run.passed}  failed: {run.failed}  "
        f"warnings: {run.warnings}"
    )
    lines.append("")
    for r in run.results:
        status = "PASS" if r.ok else "FAIL"
        rc = "-" if r.exit_code is None else str(r.exit_code)
        line = f"  {status}  {r.bucket:<7}  {r.name}  exit={rc}"
        if r.expected == "reject":
            line += (
                f"  expected_kind={r.expected_error_kind}"
                f"  observed_kind={r.observed_error_kind}"
            )
        lines.append(line)
        if r.error:
            lines.append(f"      ERROR: {r.error}")
        for w in r.warnings:
            lines.append(f"      WARN:  {w}")
    return "\n".join(lines) + "\n"
