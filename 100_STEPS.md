# 150 Steps to Standardizing stepback

## § Audit findings

**Snapshot taken:** with HEAD as of audit; a parallel agent is concurrently
building the `bench replay-caching` workstream (recent mtimes on
`stepback/{importers,cli,html_view,exporters}.py` and `stepback/__init__.py`
predate this audit by minutes — they are *not* the bench work but the
recently-merged README rewrite). No `bench/` or `bench-results/` directory
exists yet at audit time and no `bench` subcommand is registered in
`stepback/cli.py`. Re-checked at end of audit: no further file churn observed
during this run.

**Numbers.**

- ~25,448 lines of Python across `stepback/` and `tests/` (README understates
  as "~13k LOC").
- **605 tests collected, 605 passed** in 14.5s on Python 3.14 (README
  understates as "~30 tests"). No xfail/skip drift; suite is genuinely green.
- 1 TODO/FIXME/XXX marker total (a benign comment in `canonical.py`). Zero
  `.orig` / `.bak` / `.swp` files. Codebase is clean of churn artifacts.
- LICENSE (Apache-2.0) ✅. CITATION.cff ✅. CONTRIBUTING.md ❌. CODE_OF_CONDUCT.md
  ❌. SECURITY.md ❌. CHANGELOG.md ❌. `.github/` ❌ (no CI, no issue/PR
  templates, no Dependabot, no release workflow).

**README claims vs. reality.**

| Claim | Status |
| --- | --- |
| `from stepback import record, replay` works | ✅ verified — context-manager `record`, `Trace.replay_forward/step_back/substitute/bisect/branch_at/compare_branches/minimize` all present and exercised in tests. |
| Dirty-set propagation works end-to-end | ✅ verified — `scripts/bench_replay_caching.py` runs the 12-step fixture and reports `plain_cached=12 plain_llm_calls=0 dirty_after_sub=11 sub_llm_calls=5 bisect_probes=4`. The cache + dirty-set machinery is real and behaves as advertised at small scale. (Though `dirty_after_sub=11` for a 12-step trace under a single-step substitution is the **opposite** of the README headline "median dirty-set of size 3 in N=200" — see Step 51.) |
| Four shims (OpenAI / Anthropic / Bedrock / Gemini) | ✅ all four `wrap_*` and matching `*_executor` helpers exist; LangChain + MCP shims are bonus. Each shim has its own dedicated test file (`test_shims.py`, `test_bedrock_shim.py`, `test_gemini_shim.py`). All 19+8+8 shim tests pass. Duck-typed against SDK shapes — no SDK pinning. |
| `.sb` is content-addressed, signed, HMAC-chained | ✅ `trace_writer.py` writes 4-byte-length-prefixed frames with `prev_hmac` chained HMAC-SHA256 + per-frame Ed25519 signature using `cryptography`. `attestation.py` builds and signs `.pack` artifacts. `test_attestation.py` (7) and `test_compression.py` (13) exercise both. |
| `.sb` frames are **CBOR** | ❌ **README claim is incorrect.** Frames are canonical UTF-8 JSON (sorted keys, no whitespace, see `canonical.py`). Either change README to "canonical JSON" or actually move to CBOR (see Step 51). |
| `stepback bench replay-caching` subcommand reproduces headline result | ❌ **not yet a subcommand** — only `scripts/bench_replay_caching.py` exists, and it requires `PYTHONPATH=.` to run because it imports `tests.fixtures.agent`. The bench-building agent is presumably wiring this up. |
| Importers for LangSmith / Phoenix / Helicone / OTel | ⚠️ Partial. `import_openai_chat_log`, `import_langsmith_jsonl`, `import_openinference_spans` are real and tested (28 tests in `test_importers.py`). Phoenix-specific and Helicone-specific importers are **not present** — the README implies broader coverage than ships. The "otel" alias points at OpenInference. |
| Recorder version pinning, price-list pinning | ✅ header pins `recorder_version`, `canonicalisation_version`, `price_list_version`, `format_version`, `public_key`, `hmac_key_id`. |
| Delta-debugging for stochastic traces | ✅ `minimize.py` (684 LOC) implements `DDMinStrategy`, `LinearShrinkStrategy`, `BinaryHalvingStrategy`, `BruteForceStrategy`, `ShapleyAttributionStrategy`. 23 tests pass. The "stochasticity" handling is currently *predicate-deterministic* memoisation rather than re-run-with-confidence; the README's "confidence score over multiple re-executions" is aspirational. |
| Replay engine handles parallel branches | ✅ `test_parallel_branches.py` (15) covers fan-out/fan-in step kinds. |
| pyproject is minimal | ⚠️ only `cryptography>=42` runtime dep; no URLs, no classifiers, no keywords, no `[project.urls]`, no `bench` extras populated. |
| Console script `stepback` available after install | ✅ `pyproject.toml` declares the entry point; `python3 -m stepback.cli` also works. PEP 668 (Homebrew Python) blocks default `pip install -e .` — recommend pipx / venv instructions in README. |

**Net assessment.** The code is substantially more real than the README
suggests on the *implementation* axis (5× more tests, 2× more LOC, very low
churn) and substantially less real on the *external-credibility* axis (no CI,
no published benchmark numbers, no third-party corpora, no formal correctness
statement, two specific factual errors in the README about `.sb` framing and
the bench subcommand). The dirty-set algorithm and the four shims work
end-to-end on synthetic fixtures; what is missing is everything you would
need to *publish* this: external benchmark corpora, an algorithm-paper draft,
CI proving the suite stays green on Linux, signed releases, and a contributor
on-ramp.

**How to use this list.** Each step is one PR, ordered by leverage within
its section. Pick the section that matches your current sprint; cross-section
ordering is intentionally not implied. "Owner" left blank.

---

## § Packaging & install — multi-language runtime

1. **Complete.** Add `[project.urls]`, classifiers, keywords, and real extras to
   `pyproject.toml`; keep Python package metadata credible while the runtime
   grows beyond Python.

2. **Complete.** Move the Python package version to `stepback.__version__` and derive package
   metadata from it; add a test that fails on version skew.

3. **Complete.** Add package-data rules so `LICENSE`, `CITATION.cff`, README, spec fixtures,
   and frozen `.sb` conformance traces ship in wheels and sdists.

4. **Complete.** Add `scripts/smoke_install.sh` that builds wheel + sdist, installs the wheel
   in a fresh venv, and runs `stepback --help` plus a one-step record/replay.

5. **Complete.** Create `stepback-core/` as a Rust workspace with `sb-format`,
   `sb-canonical`, `sb-dirty`, `sb-replay`, and `sb-verify` crates.

6. **Complete.** Add a Rust verifier for Python-written v1 `.sb` traces; wire it to the
   frozen fixture corpus before adding any writer code.

7. **Complete.** Add PyO3 bindings under `bindings/python/stepback_core/` and route
   `verify_trace` through Rust behind an experimental flag.

8. **Complete.** Add a TypeScript package under `bindings/typescript/` with ESM/CJS builds,
   generated types, and fixture tests reading Python-written traces.

9. **Complete.** Add a Go module under `bindings/go/` with `Reader`, `Frame`, and `Verify`
   APIs plus corrupt-fixture rejection tests.

10. **Complete.** Add JVM bindings with a minimal Java/Kotlin reader and Gradle tests against
    the shared fixtures.

11. **Complete.** Add .NET bindings with `SbTraceReader` and `SbVerifier`; keep the first
    release read-only.

12. **Complete.** Add a WASM build that verifies and summarizes a trace in-browser without
    sending trace contents to a server.

13. **Complete.** Add `stepback-proxy` / `sb proxy` with HTTP and gRPC endpoints for
    `StartTrace`, `RecordStep`, `EndTrace`, and `VerifyTrace`.

14. **Complete.** Publish distroless and debug container images for `sb proxy`; smoke-test
    recording into a mounted trace directory. Added
    `docker/Dockerfile.distroless` (multi-stage, `gcr.io/distroless/python3-debian12:nonroot`
    runtime, OCI labels, nonroot UID 65532) and `docker/Dockerfile.debug`
    (python:3.11-slim-bookworm, bash + curl + jq + tini, gRPC extra),
    a repo-root `.dockerignore`, `docker/README.md`, and
    `scripts/smoke_proxy_docker.sh` which builds the chosen image, starts it
    with a host bind-mount, drives a full StartTrace → 3×RecordStep → EndTrace
    → in-container VerifyTrace cycle over HTTP, then re-verifies the resulting
    `.sb` file from the host using the canonical Python reader. Wired into a
    new `.github/workflows/docker.yml` matrix job (distroless, debug) with
    Buildx caching, image inspection, and smoke-output artifact upload.
    Existing 17/17 `tests/test_proxy.py` still green.

15. **Complete.** Document pipx, venv, cargo, npm, Go, JVM, .NET, WASM, and proxy install
    paths, with stable vs. experimental labels. Added `docs/INSTALL.md`
    as the canonical install reference: per-path quick-reference table,
    full sections for each binding (pipx, venv/pip including the `[dev]`
    / `[shims]` / `[bench]` extras, the Rust `stepback-core` workspace
    and `cargo install --path crates/sb-verify`, the PyO3
    `stepback-core` wheel via maturin, `@stepback/core` for Node ≥
    18.17, `stepback-go` for Go ≥ 1.21, the JDK 17 `bindings/jvm`
    Gradle build, the .NET 8 `Stepback.Sb` build, the WASM bundles
    produced by `scripts/build_wasm.sh`, and the `sb proxy`
    distroless/debug container images), a cross-binding writer-x-reader
    compatibility matrix, a "choosing an install path" decision
    guide, and a Versioning section that separates package SemVer
    from SB-Trace wire-format SemVer. Each path is labelled
    **Stable** / **Experimental** / **Preview** with the criteria for
    each badge made explicit. Linked from `README.md` immediately
    after the Install section.

## § Public API surface & SemVer

16. **Complete.** Add explicit `__all__` in `stepback/__init__.py` and snapshot it in
    `tests/test_public_api.py`.

17. **Complete.** Mark internal helpers with leading underscores or promote them deliberately;
    resolve raw leaks like `Trace.last_bisect_probes`.

18. **Complete.** Introduce typed `RecordedStep`, `TraceHeader`, `Receipt`, and `StepKind`
    dataclasses with a back-compat dict view. Added `stepback/step_types.py`
    with a `_DictBackedView` mixin (subclasses `collections.abc.MutableMapping`
    over the underlying dict — same identity, no copy) and four typed views:
    `StepKind` (a `str`-subclass `Enum` of every recorder-emitted kind, with a
    `coerce()` helper that round-trips known values and passes unknown ones
    through unchanged for forward-compat); `RecordedStep` (typed
    properties for `step_id`, `step_kind`, `name`, `parent_step_id`,
    `parent_step_ids`, `inputs`, `outputs`, `inputs_hash`, `outputs_hash`,
    `nondeterminism`, `nondeterminism_hash`, `wallclock_ns`, `cost_usd`,
    `llm_request`, `llm_response`); `TraceHeader` (mirrors every field
    `TraceWriter.open` writes); `Receipt` (the `{body, prev_hmac, hmac, sig}`
    wrapper, with derived `prev_hmac_bytes`, `hmac_bytes`, `signature_scheme`,
    `signature_hex`, `signature_bytes`, `frame_kind` accessors). Wrappers
    expose attribute setters that write through, the full mapping protocol
    (`step["foo"]`, `"foo" in step`, `step.get(...)`, `step.setdefault(...)`,
    `len`, `iter`, `update`, `del`), `from_dict`/`to_dict` with identity
    preservation, equality vs. plain dicts and other views, and a debug
    `repr` that surfaces well-known fields plus an `+extras=[...]` list.
    Promoted `RecordedStep`, `TraceHeader`, `Receipt`, `StepKind` through
    `stepback/__init__.py` and `stepback.__all__`; updated
    `tests/test_public_api.py::EXPECTED_PUBLIC_API` accordingly. Added
    `tests/test_step_types.py` (16 tests) covering enum coercion (known
    + unknown values), typed read/write through to the underlying dict,
    full mapping back-compat, identity preservation, equality with plain
    dicts, unhashability, default construction, real-recorder integration
    via `record(...)` (verifies `step_kind`/`parent_step_id` link the
    chain correctly and hashes are populated), `TraceHeader` decoding
    of an actual on-disk header frame, and `Receipt` parsing of the
    HMAC-chain and Ed25519 signature envelope from a freshly written
    `.sb` file (including verifying frame N+1's `prev_hmac` chains
    to frame N's `hmac`). All 798 existing tests + 25 new tests pass.

19. **Complete.** Document which `Trace` methods mutate and which return fresh traces;
    cover `step_back`, `branch_at`, `substitute`, `sweep`, and `minimize`.
    Added `docs/trace-mutation.md` as the canonical reference: a
    per-method quick-reference table covering every public method on
    `Trace` (`goto`, `step_back`, `step_forward`, `current_step`,
    `substitute`, `reset_substitutions`, `branch_at`, `compare_branches`,
    `replay_forward`, `run_replay`, `bisect`, `minimize`), every public
    method on `Branch` (`substitute`, `replay_forward`), and the
    top-level corpus helpers (`sweep_traces`, `minimize_substitutions`,
    `ddmin_substitutions`, `find_all_minimal`, `attribute_substitutions`),
    each labelled mutates-receiver vs. returns-fresh and tied to the
    concrete field that is (or is not) modified — `cursor`,
    `pending_subs`, `_last_bisect_probes`, `Branch.substitutions`,
    `Branch.result`. Includes idiomatic-usage examples for the fluent
    chain, branch isolation, and read-only minimisation/sweep, plus a
    note that `replay_forward` deep-copies inputs before building
    `StepView` so consumers can't accidentally write back into
    `Trace.recorded_steps`. Added concise mutation notes to the
    docstrings of the actual methods in `stepback/replay.py`
    (`goto`, `step_back`, `step_forward`, `current_step`,
    `substitute`, `reset_substitutions`, `branch_at`,
    `replay_forward`, `bisect`, `minimize`, `Branch.substitute`,
    `Branch.replay_forward`) so the contract is visible from
    `help(Trace.step_back)`. Extended `sweep_traces` docstring with
    an explicit "does not mutate input traces" clause. Linked the new
    doc from `README.md` immediately after the 60-second tour.
    Pinned the contract with a new `tests/test_trace_mutation_contract.py`
    (17 tests) covering: every mutating method returns `self` and
    actually mutates the documented field; every non-mutating method
    leaves `cursor` and `pending_subs` untouched; `branch_at` produces
    independent `Branch` instances whose substitutions don't leak
    into the parent or siblings; `Branch.replay_forward` overwrites
    its `result` cache; `bisect` updates `last_bisect_probes` only;
    `minimize` and `sweep_traces` are read-only over their inputs;
    and the documented fluent chain
    (`step_back(...).substitute(...).reset_substitutions()`) really
    does return the same `Trace` object throughout. All 815 tests
    + 1 skipped (full suite) pass.

20. **Complete.** Create `stepback.testing` with public deterministic fixture agents so users
    stop importing from `tests.fixtures.*`. Added a new `stepback/testing/`
    package (semver-covered) exposing `run_recorded_agent`, `run_parallel_agent`,
    `fake_llm`, `fake_tool`, `CUSTOMER_DB`, `LOOKUP_BUG_ROW`, `LOOKUP_FIXED_ROW`,
    `FACTS`, plus the `parallel_*` aliases, with full module-level docstring.
    Moved the agents out of `tests/fixtures/` (which now contains thin
    re-export shims so the old import path keeps working for one
    deprecation window). Migrated all 22 in-tree call sites
    (`from tests.fixtures.agent import …` → `from stepback.testing import …`).
    Wired `testing` into `stepback/__init__.py`, `__all__`, and the public-API
    snapshot test. Added `tests/test_public_testing_module.py` (6 tests) that
    drives the canonical 12-step + 11-step parallel fixtures end-to-end through
    the public path, asserts the back-compat shim re-exports the same
    callables, and verifies the public path imports without the `tests`
    package on `sys.path`. Full suite: 821 passed, 1 skipped.

21. **Complete.** Add a Python deprecation policy: one minor release with
    `DeprecationWarning`, release notes, and replacement API before removal.
    Implemented as `stepback/_deprecation.py` (re-exported from the top-level
    package as `deprecated`, `deprecated_alias`, `warn_deprecated`,
    `format_deprecation_message`, and `DeprecationPolicyError`). Helpers
    validate eagerly: omitting `since=`/`removal=`, or setting
    `removal == since`, raises `DeprecationPolicyError` so policy violations
    surface at import time rather than at first call. The canonical message
    embeds `since=N.M` and `stepback {removal}` tokens that release-note
    tooling greps for. The decorator preserves `__wrapped__`, signature, and
    docstring (with an appended `.. deprecated::` Sphinx note); the alias
    helper supports both functions and classes. The policy itself is
    documented in `docs/deprecation.md` (TL;DR, what counts as public, the
    deprecate→overlap→remove cycle with a one-minor-release minimum overlap,
    worked examples for each helper, and downstream guidance to run
    `pytest -W error::DeprecationWarning`). `CHANGELOG.md` carries the
    "Deprecation policy and helpers" entry plus a placeholder `### Deprecated`
    heading for the first concrete deprecation. `tests/test_deprecation.py`
    (17 tests, all passing) pins the message format, the policy
    enforcement, decorator/alias behaviour for both functions and classes,
    and includes documentation invariants asserting that `CHANGELOG.md`
    references the policy and that `docs/deprecation.md` mentions the
    minimum overlap, `DeprecationWarning`, and the canonical helper.

22. **Complete.** Define SB-Trace wire-format SemVer separately from package SemVer;
    `sbtrace 1.x` is canonical JSON, `2.0` is the first breaking encoding.
    Implemented in `stepback/spec.py` (1252 LOC) which defines the
    independent `SBTRACE_WIRE_VERSION` (currently `"1.0.0"`),
    `SBTRACE_WIRE_MAJOR` / `MINOR` / `PATCH`, the
    `SBTRACE_WIRE_ENCODING` label (`"canonical-json"`), the
    per-major encoding map `SBTRACE_WIRE_ENCODINGS`, the
    on-disk → wire mapping `SBTRACE_FORMAT_VERSION_TO_WIRE`,
    the multi-encoding map `SBTRACE_FORMAT_VERSION_TO_ENCODINGS`
    (which reserves `2 → ("deterministic-cbor",)` for the future
    breaking encoding), the typed `SBTraceVersionError`, and
    helpers `parse_wire_version`, `is_compatible_reader`,
    `wire_version_for_format_version`, and
    `format_version_for_wire_version`. The package SemVer
    (`stepback.__version__`) and wire SemVer move on independent
    tracks: package bumps never invalidate `.sb` files, and a
    wire-major bump is reserved for the first breaking encoding.
    `stepback/trace_writer.py` sources `format_version=1` from
    the spec module so the writer can never drift from the
    declared wire version. All 14 wire-format symbols are
    re-exported from `stepback/__init__.py` and frozen in
    `tests/test_public_api.py::EXPECTED_PUBLIC_API`. Covered by
    29 passing tests in `tests/test_spec_wire_version.py`
    (parsing, malformed-input rejection, future-major /
    future-minor compatibility gates, format↔wire round-tripping,
    fail-closed on unknown format versions, and the v2-reserved
    invariant) plus integration in
    `tests/test_spec_sbtrace_spec.py` and
    `tests/test_semantic_hash.py`. Documented in
    `docs/api-compat.md` ("SB-Trace wire — `SBTRACE_WIRE_*`
    constants") and `spec/sbtrace-v1.md`.

23. **Complete.** Add `stepback.spec.SBTraceSpec` that loads a versioned spec, validates
    known fields, and reports unsupported capabilities. Implemented in
    `stepback/spec.py` as a frozen dataclass holding the wire version,
    format version, encoding, frozenset field schemas (wrapper / header /
    step required and optional), the closed `step_kinds` and
    `frame_kinds` sets, and the build's `supported_capabilities` set.
    Constructors: `SBTraceSpec.for_wire_version("1.0.0")` returns the
    built-in v1 spec; `SBTraceSpec.from_mapping({...})` accepts a plain
    dict and falls back to the built-in for unspecified fields, with a
    minimal-default fallback (`_minimal_for_unknown_wire`) so external
    spec files can introduce future wire versions without first
    patching this module; `SBTraceSpec.load(path)` reads JSON or YAML
    (the latter via the optional `pyyaml` dependency, with a typed
    `SBTraceVersionError` if PyYAML is missing) and rejects empty
    files or non-mapping top levels. Frame validators
    (`validate_wrapper`, `validate_header`, `validate_step`,
    `validate_blob`, `validate_frame`) return lists of typed
    `ConformanceIssue` records with stable `code` strings
    (`wrapper.missing_field`, `wrapper.bad_signature_scheme`,
    `header.wrong_type`, `header.bad_magic`,
    `header.format_version_mismatch`, `header.missing_field`,
    `header.unknown_field`, `step.missing_field`, `step.unknown_kind`,
    `step.unknown_field`, `step.bad_envelope`, `blob.missing_field`,
    `blob.unknown_encoding`, `frame.unknown_kind`,
    `tail.missing_field`, `capability.missing_field`,
    `capability.unsupported_mandatory`, `trace.unreadable`,
    `trace.io_error`, `trace.empty`, `trace.missing_header`,
    `trace.format_version_mismatch`, `trace.missing_tail`) and
    severities (`error` / `warning`) so callers can dispatch
    programmatically. Forward-compat is honoured by default: unknown
    step kinds and unknown optional fields are warnings, gated behind
    `strict_unknown_step_kinds` / `strict_unknown_optional_fields`
    flags for stricter pipelines. `validate_trace(path)` reads the
    `.sb` frames via `trace_reader.read_frames` (independent of the
    HMAC/Ed25519 verifier — schema and crypto are independent axes),
    walks every frame through `validate_frame`, asserts the
    header-first / tail-last invariants, and records mandatory
    capabilities the build does not understand on the report's
    `unsupported_capabilities` list (emitting
    `capability.unsupported_mandatory`). `assert_conformant` accepts
    a single path, a glob pattern (`./traces/*.sb`), or a sequence of
    paths, and raises `SBTraceConformanceError` carrying the merged
    `ConformanceReport` on `.report` when any error-severity issue is
    found — matching the README's
    `SBTraceSpec.load("sbtrace-v1.0.rfc.yaml").assert_conformant("./traces/*.sb")`
    contract. The built-in v1.0.0 spec is registered in `_BUILTIN_SPECS`
    and exposed by `current_spec()`. All five symbols (`SBTraceSpec`,
    `SBTraceConformanceError`, `ConformanceIssue`, `ConformanceReport`,
    `current_spec`) are re-exported from `stepback/__init__.py`,
    listed in `stepback.__all__` under the "Step 23" comment, and
    pinned in `tests/test_public_api.py::EXPECTED_PUBLIC_API` plus the
    `stepback/conformance/api_baselines/v0.1.0/public_api.json`
    snapshot. Covered by 38 passing tests in
    `tests/test_spec_sbtrace_spec.py` (built-in registry,
    JSON / YAML loaders including override semantics, `from_mapping`
    round-trips, every frame validator's stable issue codes, real
    recorded `.sb` validation, hand-crafted broken traces for every
    error class, `assert_conformant` over single-path / glob /
    sequence inputs, and the unsupported-mandatory-capability path).

24. **Complete.** Add capability negotiation frames; unknown mandatory capabilities fail
    closed instead of being silently ignored. Added `TraceWriter.write_capability(name,
    *, mandatory=False, params=None)` (validates non-empty name, bool mandatory,
    optional dict params; HMAC-chained alongside steps; preserves caller-relative
    order under compression by routing through a new unified `pending_items`
    queue). Added `Trace_.capabilities`, `DEFAULT_SUPPORTED_CAPABILITIES`
    (`core`, `blobs`, `gzip-step-bodies`, `ed25519-receipts`,
    `hmac-sha256-chain`, `merkle-summary-v1`), and a new
    `verify_trace(..., supported_capabilities=...)` keyword that overrides
    the default and the new `STEPBACK_SUPPORTED_CAPABILITIES` env var
    (comma-separated; empty = only `core`). Both python and rust
    verification engines parse capability frames, validate their shape
    (malformed frames raise `TraceVerificationError`), and after the
    cryptographic chain check fail closed on any mandatory capability
    not in the resolved allow-list — listing only the unknown names in
    the rejection message so audit lines never leak the full allow-list.
    `iter_steps` accepts the same kwarg. All 14 tests in
    `tests/test_capability_negotiation.py` pass; the 79 cross-cutting
    writer/reader/attestation/proxy/compression/corruption/fuzz tests
    that exercise the same code paths remain green.

25. **Complete.** Add an API compatibility checker in CI that diffs generated public API docs
    and SB-Trace schema against the last release tag. Wired the existing
    `scripts/check_api_compat.py` (auto-baseline-ref mode) into a
    dedicated `api-compat` job in `.github/workflows/ci.yml` that
    checks out with `fetch-depth: 0` / `fetch-tags: true` so the most
    recent `v*` tag resolves correctly, runs the checker (text + JSON
    output), and uploads the current snapshots and diff report as a
    build artifact for post-mortem review. Refreshed the
    `stepback/conformance/api_baselines/v0.1.0/` baselines so the
    committed contract matches the live snapshot (covers the new
    `SBTRACE_FORMAT_VERSION_TO_ENCODINGS` symbol and the
    `merkle_summary` frame kind / `merkle-summary-v1` capability).
    Extended `pyproject.toml` package-data to ship
    `api_baselines/**/*.json` inside the wheel, and added
    `tests/test_api_compat.py::test_ci_workflow_runs_api_compat_check`
    so future CI edits cannot silently delete the guarantee. All 23
    `test_api_compat.py` tests pass on Python 3.14.

## § Test suite hardening

26. **Complete.** Wire coverage into the dev extra and enforce floors for canonicalization,
    trace read/write, attestation, divergence, and replay. Added
    ``coverage[toml]`` to the ``dev`` extra in ``pyproject.toml``;
    declared ``[tool.coverage.run]`` (branch coverage, ``source =
    ["stepback"]``), ``[tool.coverage.report]``, and
    ``[tool.coverage.floors]`` for the six load-bearing modules
    (``canonical``, ``trace_reader``, ``trace_writer``,
    ``attestation``, ``divergence``, ``replay``). Floors are set a
    few percentage points below currently measured branch coverage
    per the ``docs/coverage.md`` policy. Added a
    ``coverage-floors`` job to ``.github/workflows/ci.yml`` that
    installs the dev extra, runs ``coverage run -m pytest``, emits
    ``coverage.json``, invokes ``scripts/check_coverage_floors.py``
    as a hard gate, and uploads the raw coverage data as a 90-day
    workflow artifact. All 12 tests in
    ``tests/test_coverage_floors.py`` pass and the enforcement
    script exits 0 against the live suite with ≥4.7 pp of margin
    on every floored module.

27. **Complete.** Add Hypothesis tests for canonical JSON round-tripping over arbitrary
    nested JSON-like values. Implemented in
    ``tests/test_canonical_hypothesis.py`` (21 properties): pure-JSON
    round-trip, determinism, idempotence-under-reparse,
    ``hash_obj == sha256(canonical_json)``, dict-key-order invariance,
    UTF-8 validity, sorted-key emission, distinct-value/distinct-hash
    contrapositive, tuple↔list equivalence, frozenset→sorted-list,
    ``bytes`` payload marker, NaN/Inf rejection, version pin,
    unsupported-type ``TypeError``, list-order significance,
    empty-container handling, deep-nesting round-trip, dict-key-swap
    distinctness, string-concatenation unambiguity, no-whitespace-in-output,
    and triple-canonicalisation fixed-point.

28. **Complete.** Add Unicode canonicalization tests for NFC/NFD, surrogate rejection,
    non-ASCII key ordering, decimals, and binary payload markers.
    Implemented in ``tests/test_canonical_unicode.py`` (47 tests, 46
    passing + 1 codepoint-equivalence skip): NFC≠NFD byte-faithfulness
    (parametrised over ``café`` / ``Å`` / ``한`` / ``ﬁ`` plus an
    NFKC-decomposition check and a UTF-8-not-``\\uXXXX`` escape
    assertion), dict-keys-are-not-normalised contract; lone-surrogate
    rejection in strings, dict keys, and nested lists, with a
    well-formed supplementary (U+1F600) round-trip as the positive
    control; key-ordering by raw Unicode code point (ASCII <
    Latin-1 < CJK, uppercase < lowercase, supplementary planes,
    NFD-before-NFC because ``e`` < ``\u00e9``); ``decimal.Decimal``
    rejected with a typed ``TypeError`` (with the documented
    ``str(Decimal)`` workaround proven stable and the ``float``
    coercion loss-of-precision shown explicitly), parametrised
    integer/float stability, non-finite-float rejection in scalars
    *and* nested structures; binary payload marker contract
    (``{"__bytes_hex__":"<lowercase-hex>"}``, ``bytearray`` ≡
    ``bytes``, empty payload, in-structure key sorting, intentional
    collision with user dicts flagged as deliberate v1 behaviour,
    nested-list/dict, 4 KiB every-byte-value determinism); plus a
    cross-cutting ``CANONICALISATION_VERSION == "1"`` pin and a
    parametrised determinism oracle so any future change forces a
    version bump and migration.

29. **Complete.** Add property tests that dirty-set replay equals full deterministic
    re-execution on generated DAG traces. Implemented in
    ``tests/test_dirty_set_hypothesis.py`` (4 Hypothesis properties,
    all passing): random DAG-shaped traces with linear sequences and
    parallel fan-out/fan-in blocks, recorded through the real
    recorder against deterministic fakes, then compared between
    ``trace.replay_forward(executor)`` (dirty-set engine, cache hits
    where sound) and a forced-full-replay oracle that drives every
    step through the executor. The contract asserted: per-step
    outputs from dirty-set replay equal per-step outputs from full
    re-execution under identical substitutions, i.e. cache hits are
    sound by construction.

30. **Complete.** Add fuzz tests for the trace reader: random prefixes, huge frame claims,
    truncation, duplicate frames, invalid UTF-8, and invalid JSON.
    Implemented in `tests/test_reader_fuzz.py` (37 tests covering all
    six axes plus the Step-50 DoS bounds: random-byte / random-prefix
    Hypothesis fuzzers, huge-length-prefix parametrization across
    64 KiB → 4 GiB, exhaustive truncation, frame duplication on real
    boundaries, invalid-UTF-8 and invalid-JSON parameterised cases,
    plus the `MAX_FRAME_BYTES` / `MAX_NESTING_DEPTH` /
    `MAX_STRING_BYTES` opt-in caps). Hardened `stepback/trace_reader.py`
    so the contract — typed `TraceVerificationError` for every
    malformed input, never an untyped `KeyError` /
    `UnicodeDecodeError` / `MemoryError` from the post-`read_frames`
    layer — actually holds: added the three documented size limits
    (16 MiB string, 256 nesting depth, 64 MiB frame) with tunable
    overrides, wrapped the `_verify_trace_python` wrapper-shape access
    so JSON scalars / arrays / objects-missing-required-keys raise
    typed errors, and converted the bare `bytes.fromhex` /
    `pub.from_public_bytes` calls into typed-error sites.

31. **Complete.** Add corruption tests that flip every byte in a small `.sb` file and assert
    the verifier rejects with typed errors. Implemented in
    `tests/test_reader_corruption.py` (XOR-every-byte, +1-mod-256,
    targeted bit-flips in length-prefix/header/HMAC/signature
    regions) — every mutation now goes through the hardened
    `_verify_trace_python` and yields `TraceVerificationError` /
    `read_frames`-layer typed errors, never the formerly-leaky
    `KeyError('body')` / `ValueError('non-hex')` paths.

32. **Complete.** Add differential tests across Python and Rust readers once Rust lands:
    same frame count, same header, same body hash, same rejection class.
    Implemented in `tests/test_python_rust_differential.py` driven by
    the frozen `stepback-core/fixtures/v1/manifest.json` corpus. The
    `flipped_sig.sb` fixture exposed a Python-side `ValueError` on
    non-hex signatures; the reader now narrows it to
    `TraceVerificationError` so the cross-engine rejection-class
    contract holds.

33. **Complete.** Add mutation testing for dirty-set decisions by mutating parent edges,
    content hashes, step kinds, and nondeterminism hashes. Implemented
    in `tests/test_dirty_set_mutations.py` (39 tests across all four
    operators plus a combined-mutation oracle). Promoted
    `nondeterminism_hash` into the cache-safety boundary in
    `stepback/replay.py`: a step is now classified dirty whenever its
    recorded `nondeterminism_hash` is inconsistent with the canonical
    SHA-256 of the live `nondeterminism` payload, catching both
    silently-rewritten hashes and post-hoc-injected payloads.

34. **Complete.** Add soak tests that record and replay a synthetic fleet of 10,000 traces in
    a scheduled workflow, storing only aggregate stats. Implemented
    ``stepback/bench/soak.py`` (``SoakResult`` + ``run``) which drives a
    fleet of independent ``SyntheticTrace`` record + replay cycles and
    streams every per-trace metric into scalar aggregators (count, mean,
    stdev, median, p95, p99, min, max for record_ms / replay_ms /
    total_ms / dirty_count / cache_hits / real_executions /
    file_size_bytes / n_steps_actual) plus a rolling SHA-256 digest of
    canonical step ids — nothing per-trace is retained, so the output
    JSON is ~1 KiB regardless of fleet size. Errors are tallied by
    exception class instead of failing the run. Optional ``track_memory``
    runs under ``tracemalloc`` and reports peak; ``gc_every`` keeps
    long-running soaks stable. Wired the canonical CLI entry point
    ``stepback bench soak`` (with ``--n-traces``, ``--n-steps``,
    ``--seed``, ``--no-substitution``, ``--progress-every``,
    ``--track-memory``, ``--out``) and exposed ``SoakResult`` /
    ``run_soak`` from ``stepback.bench``. Added ``scripts/bench_soak.py``
    as a thin wrapper for shell pipelines. ``.github/workflows/soak.yml``
    runs the full 10,000-trace fleet on a Monday cron (off-peak) and on
    ``workflow_dispatch``: it caps the aggregate JSON at <8 KiB,
    asserts ``errors == 0`` and a 64-hex digest, and uploads the
    aggregate as a 90-day artifact. ``tests/test_soak.py`` (14 tests)
    covers the percentile/summary helpers, small-fleet record + replay
    with and without substitution, determinism under same seed, digest
    divergence under different seeds, ``summary_line`` shape,
    arg-validation, ``track_memory`` peak reporting, error
    accounting via a flaky monkeypatched ``_one_trace``, and two CLI
    smoke tests asserting ``--out`` JSON shape and the
    ``--no-substitution`` zero-dirty contract. 14/14 pass in <2 s.

35. **Complete.** Add parallel-branch stress tests with 1,000 fan-out children
    and a dirty branch that must not dirty unrelated siblings. Implemented
    ``tests/test_parallel_branch_stress.py`` (14 tests, ~580 LOC, all pass
    in ~1.2s) which records a single 1,005-step signed ``.sb`` trace
    (``plan`` LLM call → ``split_question`` tool call →
    ``parallel_branch_open`` advertising 1,000 distinct branch names →
    1,000 sibling ``lookup_topic_i`` tool calls all parented to the open
    step → ``parallel_branch_join`` referencing all 1,000 tails as
    ``parent_step_ids`` → ``synthesise`` LLM call) once at module scope
    and reuses it across the suite for sub-second wall time. Coverage
    pins the Step #35 contract from multiple angles:
    (a) recorded shape — 1,005 steps, kinds in the right order, every
    branch parented to ``open``, the join's ``parent_step_ids`` set has
    cardinality 1,000, ``open.outputs.branch_count == 1000``;
    (b) HMAC + Ed25519 round-trip via ``verify_trace``;
    (c) zero-substitution cached replay yields ``real_executions == 0``,
    ``dirty_count == 0``, ``cache_hit_count == 1005``;
    (d) **the headline guarantee**: a ``ToolOutputSubstitution`` on the
    middle branch tail dirties exactly 3 steps (target tail + join +
    synthesise), leaves the other 999 sibling branches as cache hits
    (``≥99.5%`` survival), executes 2 real LLM/tool calls, and the
    substituted-tool case is satisfied without an executor call;
    (e) dirty-set size is *constant in N_BRANCHES* (regression catcher
    for accidental O(N) propagation, asserts ``dirty < (N+1)/100``);
    (f) substituting the *first* branch does not dirty any later
    siblings, and substituting the *last* branch does not dirty any
    earlier siblings (rules out off-by-one / "dirty everything after
    step k" bugs);
    (g) three-probe sweep at indices ``1``, ``N/3``, ``2N/3`` with
    five-sibling spot checks per probe to catch position-dependent
    propagation bugs;
    (h) **K-branch substitution linearity**: substituting 7 distinct
    branches (first, second, N/4, N/2, 3N/4, N-2, N-1) dirties exactly
    K+2 steps, real-executions stays at 2, ``≥99%`` cache survival;
    (i) terminal ``PromptSubstitution`` on the synthesise step dirties
    only that step (no upstream propagation through join → branches);
    (j) negative-control: substituting the upstream ``split_question``
    tool dirties the full ``N+4`` DAG (split + open + every branch + join
    + synth) so a buggy "always cache-hit" implementation cannot pass;
    (k) handle isolation — a fresh ``replay()`` after a sibling handle's
    substituted replay sees zero dirty steps;
    (l) replay throughput floor (≥30 steps/sec, <30s ceiling) and
    per-step on-disk envelope (<2 KB amortised, >100 KB total) so any
    surprise quadratic or unbounded join-frame growth trips loudly.
    Uses module-scoped ``wide_trace`` fixture so the 1,005-frame
    record + sign happens once. Test suite stays green; the 14 new
    stress tests are independent of ``test_parallel_branches.py`` (15
    pre-existing tests covering the open/join step kinds at small
    width).

36. **Complete.** Add stochastic replay tests with seeded and noisy mock
    executors; separate correctness from predicate stability. Implemented
    ``tests/test_stochastic_replay.py`` (15 tests, ~625 LOC, all pass in
    ~1.5s) which records a deterministic 12-step fixture agent once per
    test (via ``stepback.testing.run_recorded_agent``) and then re-replays
    it with intentionally non-deterministic mock executors built from
    seeded and unseeded ``random.Random`` instances. The suite pins two
    *separately verified* invariants:
    (1) **Replay correctness is invariant under executor noise.** The
    dirty-step set, ``cache_hit_count``, ``real_executions``, the upstream
    cached-step preservation, and the cost summary are all pure functions
    of the recorded trace plus staged substitutions and must not depend
    on whether the executor used to recompute dirty steps is
    deterministic or stochastic. Tests cover: byte-identical replay
    output across repeats with a fixed seed (``digest`` ==
    ``digest``); identical dirty-set across different seeds and across
    seeded vs. unseeded executors; only-dirty-step outputs differ
    between repeats with an unseeded executor (cached steps stay
    byte-identical); a noisy executor never dirties upstream cached
    steps; ``real_executions`` exactly equals the count of executor
    invocations on dirty steps; cost summary is invariant under
    executor noise; the round-trip digest is stable under a fixed seed.
    (2) **Predicate stability is a separate, user-side concern.**
    Structural predicates (dirty count, step kinds, cache-hit count,
    cost bounds) are stable across all seeds; *content* predicates that
    inspect concrete LLM token strings are not, and the replay engine
    cannot make them so. Tests demonstrate: a structural predicate is
    True for every seed in a sweep; a content predicate flips between
    True/False under reseeding; the same content predicate admits a
    Wilson confidence interval whose width shrinks under repeated
    re-execution (documenting the path to Step #82's stochastic
    confidence intervals); replay correctness is preserved even when
    the predicate is unstable. The fixture runs the deterministic
    agent once per test, stages a ``PromptSubstitution`` or
    ``ToolOutputSubstitution`` to force the dirty path, then replays
    with seeded ``noisy_llm`` / ``noisy_tool`` factories whose outputs
    salt-mix the executor's ``random.Random`` state into the response
    string so that two different seeds produce distinct LLM outputs
    even on the same canonical input. All 15 stochastic-replay tests
    pass and are independent of the parallel-branch and dirty-set
    suites.

37. Add SDK contract tests with recorded cassettes for every provider shim;
    duck typing is not enough.

38. **Complete.** Add import/export round-trip tests for LangSmith, OpenInference, JSON,
    HTML, and OTel export. Wired two new exporter formats —
    ``export_native_json`` (lossless ``stepback_native_json_v1`` JSON
    dump that preserves every recorded field, including ``step_id``,
    ``inputs_hash``, ``outputs_hash``, ``nondeterminism_hash``,
    ``llm_request``/``llm_response``, ``cost_usd``, and
    ``wallclock_ns``) and ``export_html_view`` (the self-contained
    interactive viewer with the embedded
    ``<script type='application/json' id='stepback-data'>`` data
    island as the round-trip surface) — plus a matching
    ``import_native_json`` reader that re-emits each step verbatim
    into a fresh ``.sb`` file under a freshly-minted recorder key.
    Registered the new formats in both dispatchers (``json``,
    ``native_json``, ``stepback_json``, ``html``, ``html_view``) so
    ``export_trace`` and ``import_trace`` accept them alongside
    ``langsmith`` / ``openinference`` / ``otel`` / ``openai_chat_log``.
    ``tests/test_export_roundtrip.py`` records the deterministic
    12-step fixture agent and exercises the full five-format matrix
    end-to-end: 11 tests covering the dispatcher registry, LangSmith
    JSONL round-trip, OpenInference span round-trip, the ``otel``
    alias (asserting byte-identical output to ``openinference`` and
    importer parity), the lossless native-JSON path with strict
    rejection of mismatched format tags + non-object top levels +
    optional header passthrough, and the HTML data-island
    round-trip with byte-identical ``html`` ↔ ``html_view`` aliasing.
    Updated ``stepback.__all__``, ``EXPECTED_PUBLIC_API``, and the
    ``api_baselines/v0.1.0`` snapshot to admit the three new public
    symbols (``export_native_json``, ``export_html_view``,
    ``import_native_json``). 11/11 round-trip tests pass; the 22
    pre-existing exporters/importers tests stay green.

39. Add CI matrices for Linux, macOS, Windows, Python 3.10 through 3.13, and a
    nightly Python job allowed to fail but required to file issues.

## § Trace format & canonicalization

40. Write `spec/sbtrace-v1.md` with byte layout, frame kinds, HMAC-chain input,
    signature input, and a worked hex example.

41. Commit frozen v1 fixtures: minimal trace, multi-step trace, parallel-branch
    trace, attested trace, and corrupt variants.

42. Write `docs/canonicalization.md` covering ordering, floats, decimals,
    Unicode, bytes, maps, timestamps, model ids, and unknown fields.

43. Add experimental deterministic CBOR encoding following RFC 8949 canonical
    rules; keep canonical JSON as v1.

44. Define `.sb` v2 with dual encodings where JSON and CBOR map to the same
    semantic hash.

45. Create `spec/schema/` with stable field ids, mandatory flags, optional
    flags, and extension ranges.

46. Add conformance tests every implementation must run: read fixtures, reject
    corrupt fixtures, canonicalize fixtures, and emit identical hashes.

47. Add `stepback spec test <implementation>` to run conformance tests against
    an external reader/writer binary.

48. Add SMT-checked equivalence for canonicalizers on a bounded JSON subset;
    compare Python, Rust, TypeScript, Go, JVM, and .NET.

49. Add differential fuzzing across recorders that generate semantically
    equivalent requests for different providers.

50. Enforce maximum frame size, nesting depth, and string size in every reader;
    document denial-of-service bounds.

51. Fix docs that imply v1 frames are CBOR; state canonical JSON today, CBOR as
    candidate v2.

52. Add a Merkle summary frame at end-of-trace so attestation packs can carry a
    compact root while readers still verify the HMAC chain.

## § Dirty-set algorithm

53. Lift README pseudocode into `divergence.py` as contract docs with
    preconditions, postconditions, and assumptions.

54. Write `docs/dirty-set.md` defining trace DAG, canonical input function,
    substitution sigma, dirty set, cache reuse, and observational equivalence.

55. Prove soundness on paper: every non-dirty replayed step has the same
    observable output as full re-execution under sigma.

56. Mechanize soundness in Lean or Coq for an immutable step DAG and
    collision-free canonical hash assumption.

57. State completeness separately: every step whose recomputed inputs hash
    differently is included in the dirty set.

58. Add asymptotic complexity bounds for linear traces, DAG traces, and
    branch-heavy traces, including memory bounds.

59. Implement branch-aware propagation: dirty fan-out children independently,
    dirty joins if any consumed branch output changes, preserve clean siblings.

60. Implement partial recompute for structured inputs so one changed tool result
    does not force re-hashing unrelated subtrees.

61. Add stale-cache detection when a downstream step references a recomputed
    output hash even if direct serialized input appears unchanged.

62. Specify `RaiseSubstitution` semantics and when downstream cache entries are
    invalid after a substituted exception.

63. Record nondeterminism classes for clock, RNG, env, network, and model
    sampling; define when each class forces a dirty step.

64. Add distributed dirty-set computation over a worker pool; partition by DAG
    regions and merge summaries at joins.

65. Add ClickHouse dirty-set summary tables for trace id, substitution kind,
    dirty count, clean count, branch count, and calls saved.

66. Publish empirical dirty-set distributions over synthetic fixtures, public
    benchmark corpora, and anonymized production traces.

67. Reconcile the current 12-step fixture result (`dirty_after_sub=11`) with any
    README headline before claiming small dirty sets.

## § Replay engine

68. Split replay into planner and executor phases: plan dirty steps and cache
    hits first, then execute with deterministic scheduling.

69. Add deterministic seeding policy for LLM and tool executors, including
    warning levels when providers do not support seeds.

70. Keep an in-process replay API for unit tests and local debugging.

71. Add sidecar replay through `stepback-proxy`: submit trace and substitutions,
    receive replay events over gRPC streaming.

72. Add sandboxed replay modes using gVisor and Firecracker for untrusted tool
    execution.

73. Add parallel-branch scheduling so independent dirty branches execute in
    parallel and joins wait only on consumed inputs.

74. Add sharded content-addressed step cache backed by disk, S3, GCS, Azure
    object storage, and dedup across traces.

75. Add Kafka-backed event bus for replay jobs and step-complete events; make
    the planner idempotent under worker retries.

76. Add worker leases and checkpointed replay state so million-point sweeps
    survive worker failure.

77. Add `Trace.replay_forward(distributed=True, workers=N)` backed by the same
    planner as local replay.

78. Emit replay provenance: executor versions, cache hits, dirty reasons, seeds,
    policy decisions, model versions, and provider versions.

79. Add a web time-travel debugger with step forward/back, cache-hit display,
    canonical input diffs, and causal graph view.

## § Minimization

80. Promote predicates to a typed DSL with boolean composition, metric
    thresholds, regexes, policy decisions, and Python callbacks.

81. Add multi-objective ddmin for trace length, LLM calls, cost, latency, and
    policy-violating steps.

82. Add statistical stability metrics: repeated dirty replays, confidence
    intervals, flaky predicate classification, and warnings.

83. Add incremental bisect across multiple regressions so finding one culprit
    does not restart the search.

84. Add Shapley-style attribution for steps that jointly cause a failure.

85. Add minimization over branches: drop independent branches safely, preserve
    joins only when consumed outputs matter.

86. Add minimization for imported traces where executable replay is partial;
    mark steps requiring unavailable executors.

87. Add HTML minimization reports with before/after graphs, removed steps,
    predicate evaluations, and confidence summaries.

88. Write the minimization paper artifact with algorithm, stochastic
    assumptions, failure modes, and empirical comparison to naive ddmin.

## § Shims and framework recorders

89. Refactor provider shims behind a `ShimContract` ABC: canonical request,
    canonical response, executor, streaming hooks, async hooks, version probe.

90. Add OpenAI contract tests for chat, responses, tool calls, streaming, async
    clients, and OpenAI-compatible endpoints.

91. Add Anthropic contract tests for messages, tool use, streaming, thinking
    blocks if present, and async clients.

92. Harden Bedrock and Gemini shims against real SDK versions with recorded
    cassettes and compatibility matrices.

93. Add Azure OpenAI support, including deployment-name model ids and regional
    endpoint metadata.

94. Add Vertex AI / Google GenAI support beyond Gemini; canonicalize safety
    settings and tool declarations.

95. Add Cohere and Mistral shims with dedicated canonicalizers.

96. Add Together, Fireworks, Groq, Cerebras, NVIDIA NIM, vLLM, TGI,
    llama.cpp, and Ollama adapters.

97. Add streaming recorder support where chunks are part of one LLM-step receipt
    and final response hashing is deterministic.

98. Add async recorder support for Python and JS SDKs; prove context
    propagation survives `await` and task groups.

99. Enforce recorder overhead budgets in CI: p50 under 50 microseconds for the
    Python fast path, with shim-specific exceptions documented.

100. Create a certified-shim program: contract tests, overhead report,
     canonicalization review, version matrix, and signed compatibility badge.

101. Add LangChain and LangGraph recorders using callback / run-manager hooks,
     preserving run ids as trace metadata.

102. Add LlamaIndex, DSPy, Haystack, AutoGen, CrewAI, Semantic Kernel, Strands,
     Pydantic-AI, Inspect-AI, and MCP recorders with minimal examples.

103. Add `stepback diagnose` to inspect installed SDK/framework versions and
     warn when newer than the certified matrix.

## § Importers/exporters

104. Complete importers for Phoenix, Helicone, Langfuse, and Datadog APM; map
     their spans into SB-Trace step kinds with explicit lossy fields.

105. Harden LangSmith and OpenInference importers with real exported fixtures,
     schema-version detection, and hash-stability tests.

106. Add OpenTelemetry import using stable semantic conventions for
     `agent.step`; keep OpenInference as a compatibility profile.

107. Add OTel export with `agent.step.*` attributes suitable for upstream
     proposal to OpenTelemetry semantic conventions.

108. Add JSON export with a stable schema and compatibility tests.

109. Add self-contained HTML export with causal graph, diff panes, minimization
     report, and attestation summary.

110. Add CycloneDX-AI export linking traces to models, prompts, tools, datasets,
     and policy decisions.

111. Add SLSA and in-toto provenance attestations for traces, benchmark
     submissions, and incident replay packs.

112. Add lossiness reports to every importer/exporter: absent, approximated,
     synthesized, and dropped fields.

## § Benchmarks

113. Turn `scripts/bench_replay_caching.py` into `stepback bench
     replay-caching` without `PYTHONPATH=.` and with JSON output.

114. Define result schema: corpus id, trace count, substitution distribution,
     dirty-set stats, cache hits, LLM calls saved, latency, cost, storage,
     versions, and hardware.

115. Add corpus loaders for SWE-bench-Verified, GAIA, tau-bench, AgentBench,
     OSWorld, and WebArena.

116. Create three author-original corpora: support agent, code-review agent, and
     policy-gated payments agent, all redistributable as `.sb`.

117. Add anonymized production-trace ingestion rules: redaction, hash
     preservation, privacy review, and redaction attestation.

118. Add replay-caching benchmark: cost reduction, wallclock speedup, dirty-set
     distribution, and cache-hit reasons.

119. Add minimization benchmark: final trace size, predicate stability, LLM calls
     spent, and comparison to naive ddmin.

120. Add model-swap differential benchmark: fidelity against full re-execution
     and statistically grounded difference detection.

121. Add recorder-overhead benchmark for every shim and framework recorder;
     enforce p50/p95 budgets in CI.

122. Add storage-compression benchmark: raw JSON, `.sb` v1, CBOR candidate,
     zstd, deduped object-store layout, and query-index overhead.

123. Publish MLPerf-style submission rules: frozen code, signed trace pack,
     hardware manifest, exact commands, validator output, and audit rights.

124. Add hosted leaderboard generation from signed JSON submissions; reject
     submissions that fail conformance or attestation checks.

125. Add scheduled frontier-model re-evaluation so results do not fossilize
     around one provider generation.

126. Write the NeurIPS-Datasets benchmark paper with corpus documentation,
     licensing, metrics, limitations, and reproduction instructions.

## § Attestation & cryptography

127. Write `SECURITY.md` threat model: what HMAC/signatures prove, what they do
     not prove, key handling, disclosure path, and verifier guarantees.

128. Add key rotation for trace attestations: preserve old signatures, append a
     rotation frame, and verify both trust chains.

129. Add post-quantum signature experiments with ML-DSA and SLH-DSA behind
     explicit capability frames.

130. Add threshold signing for long-lived production recorders with M-of-N
     witnesses for incident-grade traces.

131. Add witness cosigning for public benchmark traces so leaderboard entries
     prove trace packs existed before evaluation.

132. Add transparency-log integration for incident records and benchmark packs;
     store inclusion proofs in attestation packs.

133. Add hardware-backed key support via PKCS#11, YubiHSM, and cloud KMS, with
     tests using software simulators.

134. Add `stepback verify --strict --policy <policy>` to verify cryptography,
     schema, canonical bytes, and recorder identity in one command.

## § Performance & scale

135. Add microbenchmarks for canonicalization, frame writing, HMAC/signing,
     recorder hooks, reader throughput, and dirty-set planning.

136. Optimize recorder fast path to p50 under 50 microseconds per call without
     signing and document the budget with signing enabled.

137. Add batch-signing / async-signing mode for high-throughput recorders while
     preserving append-only ordering guarantees.

138. Implement sharded step cache on object storage with content-addressed dedup
     across runs, corpora, and organizations.

139. Add ClickHouse schema for trace queries by model, step kind, dirty reason,
     cost, policy decision, canonical hash, and incident id.

140. Add distributed bisect across a worker pool for large trace sets and
     multi-objective predicates.

141. Add load tests simulating millions of agent runs per day through
     `stepback-proxy`; publish CPU, memory, storage, and p95 latency curves.

142. Add backpressure and sampling controls so recorder failure cannot take down
     the agent unless configured as mandatory.

## § Specification & standards

143. Create `spec/rfcs/0001-sbtrace-core.md` plus RFCs for canonicalization,
     dirty-set semantics, attestation packs, importer lossiness, and OTel
     `agent.step` semantic conventions.

144. Submit `agent.step` semantic conventions upstream to OpenTelemetry; keep
     the exporter aligned with review feedback and publish conformance status
     for Python, Rust, TypeScript, Go, JVM, .NET, proxy, and WASM.

145. Add formal standards artifacts: TLA+ spec of the `.sb` HMAC chain,
     conformance dashboard, Linux Foundation proposal, and CNCF sandbox draft
     once multi-implementation production use exists.

## § Community, governance, release engineering

146. Add `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue templates, PR template,
     security advisory flow, Dependabot/Renovate, CI, release workflows,
     changelog, governance docs, and certified-integration rules.

## § Research artifacts & papers

147. Write `RELATED_WORK.md`, `ARTIFACT.md`, and the 4-5 paper line: dirty-set
     algorithm, distributed replay runtime, stochastic minimization,
     benchmark/dataset paper, and incident/audit case-study paper.

## § Production case studies

148. Build production-shaped case studies for millions-of-runs/day recording,
     incident replay, regulator/auditor evidence packs, model migration, large
     parameter sweeps, and redacted trace publication.

## § Ecosystem integrations

149. Integrate with ragdoctor, flowwarden, and toolwarden; record diagnostic RAG
     runs, provenance/IFC labels, enforcement decisions, denied calls, policy
     versions, and replay-time audits.

150. Add MCP recorder/proxy mode, CycloneDX-AI interop, SLSA/in-toto examples,
     observability bridges back to OTel warehouses, WASM viewer embedding docs,
     and a public integration matrix.

---

*Total: 150 numbered, single-PR-sized steps. The original file aimed for 100;
this roadmap expands the project from a credible Python prototype into an open
trace standard, multi-language runtime, benchmark consortium, formal artifact,
and production replay substrate.*
