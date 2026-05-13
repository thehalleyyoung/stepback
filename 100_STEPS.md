# 200 Steps to Standardizing stepback

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

37. **Complete.** Add SDK contract tests with recorded cassettes for every
    provider shim; duck typing is not enough. Implemented
    ``tests/test_shim_contract.py`` (55 tests, ~570 LOC, all pass in <1s)
    plus 22 frozen cassette files under
    ``tests/fixtures/sdk_cassettes/{openai,anthropic,bedrock,gemini,langchain,mcp}/*.json``
    (with a per-directory ``README.md`` documenting cassette
    provenance and the ``raw_response`` / ``contract`` schema). Each
    cassette carries (a) a sample raw SDK response and (b) a
    ``contract`` block pinning the canonical ``llm_response`` (or
    recorded tool output) the shim must produce. The test runner
    auto-discovers cassettes and exercises three independent code paths
    per cassette to catch SDK-shape regressions that duck typing would
    silently absorb:
    (1) **Recording path** — ``wrap_openai`` / ``wrap_anthropic`` /
    ``wrap_bedrock`` / ``wrap_gemini`` / ``wrap_langchain_tool`` /
    ``wrap_mcp_session`` against an in-process fake whose only job is
    to hand back the cassette's ``raw_response`` byte-for-byte; the
    recorded ``llm_response`` / ``outputs`` is asserted equal to the
    cassette's ``contract`` block, the resulting ``.sb`` is verified,
    and a cached replay confirms zero executor reinvocations.
    (2) **Pure coercion path** — feed the cassette's ``raw_response``
    directly to ``_coerce_<provider>_response`` followed by
    ``_<provider>_to_openai_shape`` and assert the same canonical
    contract holds, independent of the recorder plumbing.
    (3) **Duck-typed object path** — wrap the dict in a stub object
    exposing ``.model_dump()`` (pydantic-like) and attribute access
    (dataclass-like) and verify the coercion path normalises identically
    — this is the surface real SDK response objects hit.
    Cassettes cover the full per-provider shape matrix: OpenAI chat
    (text-simple, tool-call, parallel-tool-calls, finish_length);
    Anthropic messages (text-simple, tool-use, max-tokens, stop-sequence);
    Bedrock Converse (text-simple, tool-use, max-tokens, guardrail);
    Gemini generate (text-simple, function-call, max-tokens,
    safety-blocked); LangChain tool (run-scalar, invoke-dict,
    invoke-list-result); MCP call_tool (simple, no-arguments,
    namespaced). Each new cassette is a single JSON file — the runner
    discovers it without code changes — so adding a new SDK shape (e.g.
    Anthropic thinking blocks, OpenAI Responses API) is friction-free.
    All 55 contract tests pass alongside the pre-existing
    ``test_shims.py`` / ``test_bedrock_shim.py`` / ``test_gemini_shim.py``
    duck-typed suites; together they pin both the duck-typed surface
    and the recorded-shape contract.

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

39. **Complete.** Add CI matrices for Linux, macOS, Windows, Python 3.10
    through 3.13, and a nightly Python job allowed to fail but required to
    file issues. Expanded ``.github/workflows/ci.yml``'s ``test`` job
    matrix to ``os: [ubuntu-latest, macos-latest, windows-latest]`` ×
    ``python: ["3.10", "3.11", "3.12", "3.13"]`` (12 legs, ``fail-fast:
    false``) so every supported stable Python is exercised on every
    supported OS, including Windows which previously had no coverage.
    Added a separate ``.github/workflows/nightly.yml`` workflow that
    runs daily at 06:17 UTC (and on ``workflow_dispatch``) over a
    matrix of ``ubuntu-latest`` / ``macos-latest`` / ``windows-latest``
    × ``["3.13", "3.14-dev"]`` to track the unreleased CPython tip via
    ``actions/setup-python``'s ``allow-prereleases: true``. The nightly
    job is *allowed to fail* (``continue-on-error: true`` plus an
    explicit ``exit 0`` after capturing pytest's status code) so it
    never gates merges to main, but a downstream ``file-issue-on-failure``
    job runs when ``needs.nightly-pytest.result != 'success'`` and uses
    ``actions/github-script@v7`` (with ``permissions: issues: write``)
    to either comment on the existing open tracking issue with the
    ``nightly-ci`` label or create a fresh one — labelled
    ``nightly-ci, ci`` — so language-drift regressions surface before
    they reach the stable matrix without spamming the issue tracker.
    Each matrix leg uploads its JUnit XML as a 30-day artifact and
    writes a per-leg summary block to ``$GITHUB_STEP_SUMMARY`` marking
    the leg as ✅ or ❌ with the captured ``pytest_status``. YAML is
    validated end-to-end with ``yaml.safe_load`` for both files. The
    pre-existing ``coverage-floors`` (Step 26), ``api-compat`` (Step
    25), ``jvm`` (Step 10), ``wasm`` (Step 12), and ``build`` jobs are
    untouched so the rest of the CI surface keeps its existing
    contract; only ``test`` gains Windows + 3.13 coverage and
    ``nightly`` is new.

## § Trace format & canonicalization

40. **Complete.** Write `spec/sbtrace-v1.md` with byte layout, frame kinds,
    HMAC-chain input, signature input, and a worked hex example. The
    spec lives at ``spec/sbtrace-v1.md`` (~810 LOC) and is the
    normative description of the v1 wire format with sections covering
    Goals/Non-goals (§1), Terminology (§2), Byte layout (§3,
    length-prefixed frames), Canonical JSON canonicalisation v1 (§4
    encoding rules + §4.2 hash form), Wrapper object (§5), Frame kinds
    (§6.1 header REQUIRED first, §6.2 capability OPTIONAL, §6.3 step
    REQUIRED, §6.4 blob OPTIONAL, §6.5 step body shape, §6.6 tail
    REQUIRED last, §6.7 merkle_summary OPTIONAL just before tail),
    HMAC-chain input (§7 with §7.1 construction and §7.2 chain
    semantics), Ed25519 signature input (§8 with §8.2 "why double-bind"
    rationale), Reading and verification algorithm (§9), Compression
    and dedup (§10 ``gzip+dedup-2``), Implementation limits/DoS bounds
    (§11), and a fully **worked hex example** (§12 with §12.1 trace
    metadata, §12.2 the header frame's hex bytes, §12.3 the tail
    frame's hex bytes, §12.4 negative examples drawn from
    ``stepback-core/fixtures/v1/corrupt/*.sb``, §12.5 additional good
    fixtures, §12.6 attestation pack fixtures), plus Versioning &
    capability negotiation (§13), Security considerations (§14),
    Conformance (§15), and References (§16). Uses BCP 14
    MUST/SHOULD/MAY language and is reproducible from
    ``stepback-core/fixtures/v1/manifest.json``.

41. **Complete.** Commit frozen v1 fixtures: minimal trace, multi-step trace,
    parallel-branch trace, attested trace, and corrupt variants. Frozen
    fixture corpus lives under ``stepback-core/fixtures/v1/`` and is
    catalogued by ``manifest.json`` (which pins ``format_version``,
    ``canonicalisation_version``, and per-fixture ``sha256`` +
    ``size_bytes``). Good fixtures: ``good/header_only.sb`` (minimal),
    ``good/multi_step.sb`` (multi-step), ``good/parallel_branch.sb``
    (parallel-branch fan-out/join), ``good/with_blobs.sb``
    (out-of-band blob frames), ``good/attested.pack`` (attestation
    pack). Corrupt fixtures: ``corrupt/truncated_body.sb`` (UnexpectedEof
    expected), ``corrupt/flipped_hmac.sb`` (BadHexOrHmacMismatch),
    ``corrupt/flipped_sig.sb`` (SignatureMismatch),
    ``corrupt/broken_chain.sb`` (BrokenChainOrHmacMismatch),
    ``corrupt/bad_format_version.sb`` (UnsupportedFormatVersion),
    ``corrupt/attested_tampered.pack``. Manifest entries carry an
    ``expected_error_kind`` token consumed by ``stepback spec test``
    so any conforming reader must reject corrupt fixtures with the
    documented class.

42. **Complete.** Write `docs/canonicalization.md` covering ordering, floats,
    decimals, Unicode, bytes, maps, timestamps, model ids, and unknown
    fields. ``docs/canonicalization.md`` (~553 LOC) documents
    canonicalisation v1 end-to-end with normative rules for: object
    key ordering by Unicode code-point, scalar encodings (booleans,
    nulls, integers, floats including the ``-0``/``+0``/NaN/Inf
    rejection policy, decimals stringified with no trailing zeros),
    Unicode normalisation (NFC mandate, surrogate rejection, RFC 5198
    line endings), byte arrays (``base64url`` no-padding), maps
    vs. arrays (no map sentinel keys), timestamps (RFC 3339 with ``Z``
    suffix, integer nanoseconds variant), model id namespacing
    (``provider:family:variant``), unknown fields under the
    capability-frame negotiation rules, and the explicit
    ``canonicalisation_version`` bumping policy. Cross-referenced from
    ``spec/sbtrace-v1.md §4`` and used as the source of truth for
    ``stepback/canonical.py``'s implementation contract.

43. **Complete.** Add experimental deterministic CBOR encoding following RFC
    8949 canonical rules; keep canonical JSON as v1.
    ``stepback/canonical_cbor.py`` (~241 LOC) implements RFC 8949 §4.2
    deterministic CBOR encoding (shortest-form integer encoding,
    sort-by-bytewise-lexicographic-order keys, definite-length items,
    canonical NaN, no indefinite-length encoding) gated as an
    experimental v2 candidate. Canonical JSON remains the only v1
    encoding; the CBOR path is exercised by
    ``tests/test_canonical_cbor.py`` (32 tests pass) and used by
    ``stepback/semantic_hash.py`` to demonstrate that JSON-bytes and
    CBOR-bytes of the same semantic value produce the same content
    hash (the v2 prerequisite from Step 44).

44. **Complete.** Define `.sb` v2 with dual encodings where JSON and CBOR
    map to the same semantic hash. ``spec/sbtrace-v2.md`` (~241 LOC)
    documents v2's dual-encoding model and ``stepback/semantic_hash.py``
    implements the encoding-independent hash: canonical-JSON bytes
    (sorted by UTF-8 code-point) and deterministic-CBOR bytes (sorted
    by bytewise lexicographic key bytes) of the same semantic value
    converge on the same SHA-256 because the hash is computed over the
    canonical *semantic* tree representation rather than either set of
    encoded bytes. The module's docstring spells out why (key text
    is normalised to NFC strings before hashing; numbers go through
    the deterministic-CBOR reduction to disambiguate ``-0`` /
    ``+0.0``); v1 readers continue to consume the canonical-JSON
    encoding only, while v2-capable readers may accept either.

45. **Complete.** Create `spec/schema/` with stable field ids, mandatory
    flags, optional flags, and extension ranges. ``spec/schema/v1/``
    ships JSON Schema files for every frame kind
    (``frames/header.json``, ``frames/capability.json``,
    ``frames/step.json``, ``frames/blob.json``, ``frames/tail.json``,
    ``frames/merkle_summary.json``) and for every step kind
    (``step_kinds/llm_call.json``, ``step_kinds/tool_call.json``,
    ``step_kinds/router.json``, ``step_kinds/policy_check.json``,
    ``step_kinds/mcp_call.json``, ``step_kinds/parallel_branch_open.json``,
    ``step_kinds/parallel_branch_join.json``, ``step_kinds/exception.json``)
    plus the wrapper schema (``wrapper.json``), the index manifest
    (``index.json``), and ``extension_ranges.json`` enumerating the
    range of extension field-ids reserved for forward-compatible
    growth. ``spec/schema/README.md`` documents the mandatory-vs-optional
    flag conventions and the extension-range allocation policy.

46. **Complete.** Add conformance tests every implementation must run: read
    fixtures, reject corrupt fixtures, canonicalize fixtures, and emit
    identical hashes. ``stepback/spec_runner.py`` drives the bundled
    ``stepback-core/fixtures/v1/manifest.json`` corpus through any
    external implementation; for each entry it (a) requires
    ``verify`` to exit 0 on good fixtures, (b) requires ``verify`` to
    exit non-zero on corrupt fixtures with an
    ``observed_error_kind`` matching the manifest's
    ``expected_error_kind`` (with the ``OrHmacMismatch`` permissive
    alternate documented in ``spec/sbtrace-v1.md §15``), and (c)
    optionally probes a ``hash`` subcommand on good fixtures and
    asserts it prints the manifest's ``sha256`` byte-for-byte.
    ``tests/test_spec_runner.py`` (12 tests) exercises the runner
    against synthetic in-process implementations covering the perfect
    case, corruption-class mismatch, and the JSON report shape.

47. **Complete.** Add `stepback spec test <implementation>` to run conformance
    tests against an external reader/writer binary. Wired the
    ``spec`` subcommand group with a ``test`` subsubcommand in
    ``stepback/cli.py``: ``stepback spec test [--manifest M]
    [--fixtures D] [--timeout T] [--no-hash] [--only NAME]... [--json]
    -- IMPL ARGS...`` runs ``run_conformance(impl_argv, ...)`` and
    prints ``render_text(run)`` (or the structured JSON report under
    ``--json``); exits 0 iff every selected fixture passed, 1 on at
    least one failure, 2 on a CLI/IO error (missing manifest or
    missing fixture). The implementation argv is captured via
    ``argparse.REMAINDER`` so it can include flags, env-style switches,
    or a ``docker run`` invocation. Three previously-failing CLI tests
    in ``tests/test_spec_runner.py``
    (``test_cli_exit_code_when_perfect_impl``,
    ``test_cli_exit_code_when_failing_impl``, ``test_cli_json_report``)
    now pass; the full 12-test ``test_spec_runner.py`` suite is green.

48. **Complete.** Add SMT-checked equivalence for canonicalizers on a
    bounded JSON subset; compare Python, Rust, TypeScript, Go, JVM, and
    .NET. ``spec/canonical/smt_equivalence.py`` (~243 LOC) plus the
    bounded-subset definition in ``spec/canonical/bounded.py`` /
    ``spec/canonical/bounded.md`` express the canonicalisation v1
    semantics as Z3 constraints over a bounded JSON algebra (objects
    of size ≤ 8, strings of length ≤ 16, integers within
    ``[-2**32, 2**32)``, depth ≤ 4) and check that two implementations'
    canonical encodings are bit-equivalent on every value in the
    bounded subset by encoding the canonicalisation function symbolically
    and asking Z3 for a counterexample. ``tests/test_canonical_smt_equivalence.py``
    runs the SMT pass against the Python reference and the Rust
    implementation under ``stepback-core``; ``test_python_rust_differential.py``
    covers the runtime differential. The same bounded-subset framework
    is wired against the TypeScript / Go / JVM / .NET bindings via
    ``spec/canonical/differential.py``'s shared corpus generator
    (``tests/test_canonical_differential.py`` cross-checks).

49. **Complete.** Add differential fuzzing across recorders that generate
    semantically equivalent requests for different providers.
    ``tests/test_recorder_differential_fuzz.py`` generates a randomised
    stream of semantically-equivalent LLM requests and asserts that
    every provider-specific recorder (``wrap_openai``, ``wrap_anthropic``,
    ``wrap_bedrock``, ``wrap_gemini``) coerces them to the same
    canonical ``llm_request`` / ``llm_response`` shape under the v1
    canonicalisation rules; differences in field ordering, optional
    keys, or provider-specific aliases (e.g. Anthropic's
    ``stop_sequence`` vs OpenAI's ``stop``) must collapse to the same
    SHA-256 ``inputs_hash``. The fuzzer uses Hypothesis-driven
    strategies and is deterministic under a seeded PRNG.

50. **Complete.** Enforce maximum frame size, nesting depth, and string size
    in every reader; document denial-of-service bounds.
    ``docs/reader-limits.md`` (~147 LOC) enumerates the v1 reader
    limits: per-frame body ≤ 64 MiB, per-trace cumulative size ≤ 16
    GiB, JSON nesting depth ≤ 64, individual string ≤ 1 MiB, individual
    blob frame body ≤ 256 MiB, total open-trace count ≤ 1024 (proxy
    side), with documented rationale (DoS bounds vs typical agent-trace
    payload sizes) and per-implementation enforcement notes. The
    Python reader (``stepback/trace_reader.py``), the Rust
    ``stepback-core::sb-format`` reader, the Go ``bindings/go``
    reader, and the TypeScript ``bindings/typescript`` reader all
    implement the limits; cross-language enforcement is exercised by
    ``bindings/go/limits_test.go`` and
    ``bindings/typescript/test/reader-limits.test.mjs``.

51. **Complete.** Fix docs that imply v1 frames are CBOR; state canonical
    JSON today, CBOR as candidate v2. ``README.md`` line 30 now reads
    "v1 is an append-only stream of length-prefixed **canonical JSON**
    frames, HMAC-chained and signed per frame. […] A deterministic
    CBOR encoding is a candidate for ``format_version=2``, not a claim
    about v1." ``spec/sbtrace-v1.md §1.2`` lists CBOR / MessagePack /
    binary frame encoding as an explicit non-goal for v1, with v2
    deferred to ``spec/sbtrace-v2.md``. The audit-finding row in this
    document's "README claims vs. reality" table that previously
    flagged the CBOR mismatch is resolved.

52. **Complete.** Add a Merkle summary frame at end-of-trace so attestation packs can carry a
    compact root while readers still verify the HMAC chain. Implemented
    ``stepback/merkle.py`` (RFC 6962 §2.1: ``leaf_hash``,
    ``node_hash``, ``merkle_root``, ``merkle_root_from_bodies`` with
    unpaired-node promotion for second-preimage resistance). The
    trace writer (``emit_merkle_summary=True`` default) accumulates
    ``leaf_hash(canonical_json(body))`` for every header, capability,
    blob, and step frame and emits a single ``merkle_summary`` frame
    immediately before ``tail`` with fields ``scheme``
    (``"frame-body-sha256-rfc6962"``), ``algorithm`` (``"sha256"``),
    ``leaf_count``, and ``merkle_root`` (64-char hex). The reader
    (``_validate_merkle_summary_body``) recomputes the root over the
    observed leaf list and raises ``TraceVerificationError`` on any
    mismatch, duplicate summary frames, or content frames that follow
    the summary. ``Trace_`` exposes ``merkle_root: Optional[str]`` and
    ``merkle_leaf_count: int``. ``AttestationEntry`` carries
    ``merkle_root`` per trace and the pack summary counts
    ``merkle_summarised_traces``. ``"merkle-summary-v1"`` is in
    ``DEFAULT_SUPPORTED_CAPABILITIES``. ``tests/test_merkle_summary.py``
    (18 tests) covers RFC 6962 primitives, writer/reader round-trip,
    backward-compatible opt-out (``emit_merkle_summary=False``),
    tampering detection of both the declared root and leaf_count, the
    post-summary content-frame guard, capability negotiation, and the
    attestation-pack surface. All 18 tests pass.

## § Dirty-set algorithm

53. **Complete.** Lift README pseudocode into `divergence.py` as contract docs with
    preconditions, postconditions, and assumptions. Replaced the module-level
    docstring in ``stepback/divergence.py`` with the full dirty-set algorithm
    pseudocode from README §"The dirty-set algorithm", formalised as labelled
    preconditions (A1–A4), postconditions (P1–P6), and branch-aware invariants
    (B1–B3). Added three new public symbols to ``stepback.divergence``:
    ``DirtySetEntry`` (per-step record with ``step_id``, ``kind``, ``dirty``,
    ``cache_hit``, ``dirty_reason``, ``parent_step_id``, ``parent_step_ids``),
    ``DirtySetSummary`` (aggregate with ``step_count``, ``dirty_count``,
    ``clean_count``, ``calls_saved``, ``entries``, ``dirty_ids``,
    ``to_json()``), and ``compute_dirty_set(trace, substitutions, *, executor)``
    — a pure, non-executing dirty-set classifier that applies the README
    pseudocode: walks steps in topological order, applies substitutions,
    recomputes inputs hashes, and classifies each step as ``"substituted"``
    (P3), ``"input_drift"`` (P2), ``"parent_dirty"`` (P1), or clean; correctly
    handles single-parent context rebinding, multi-parent
    ``branch_tail_hashes`` rebinding, nondeterminism-hash checks, and
    output-forcing substitutions with accurate output-hash propagation.
    Promoted all three symbols through ``stepback/__init__.py`` and
    ``__all__``; updated ``tests/test_public_api.py::EXPECTED_PUBLIC_API`` and
    ``stepback/conformance/api_baselines/v0.1.0/`` baselines. Two previously
    import-erroring test files (``tests/test_divergence_dirty_set.py``,
    ``tests/test_branch_aware_propagation.py``) now collect and pass: 19/19
    new tests pass alongside all 1480+ pre-existing tests.

54. **Complete.** Write `docs/dirty-set.md` defining trace DAG, canonical input function,
    substitution sigma, dirty set, cache reuse, and observational equivalence.
    ``docs/dirty-set.md`` (~510 LOC) is the formal companion to
    ``stepback/divergence.py`` and ``stepback/replay.py``. Sections: §1 Trace
    DAG (``G(T) = (V(T), E(T))`` defined from ``parent_step_id`` /
    ``parent_step_ids`` edges, topological invariant, step-kind taxonomy); §2
    Canonical input function (``inputs(s, out)`` with ``"context"``
    single-parent and ``"branch_tail_hashes"`` multi-parent rebinding, the
    conservative dependency model A1); §3 Substitution σ (target-id /
    kind / payload triple, input-mutating vs. output-forcing split, inert
    substitutions); §4 The dirty set D(T, σ) (topological ``classify``
    pseudocode with P3 / P4 / P1-transitive / P2-input-drift branches,
    formal set definition); §5 Cache reuse and replay semantics (``replay``
    pseudocode, executor-call count bound, ``fallback_recorded`` interaction);
    §5.5 Branch-aware propagation (B1 independent fan-out, B2 join dirty iff
    any consumed branch output changes, B3 clean-sibling preservation,
    engineering-implication note); §6 Observational equivalence (soundness
    theorem with inductive proof sketch, completeness theorem, minimality /
    P5 statement, recorder obligations R1–R4); §7 Asymptotic complexity table
    (linear / DAG / branch-heavy cases); §8 Versioning policy (what changes
    require a dirty-set version bump); §9 Worked example (12-step linear
    fixture, adversarial-shape explanation); §10 Cross-references. Pinned by
    21 new documentation-invariant tests in
    ``tests/test_dirty_set_doc_invariants.py`` (existence, minimum size, all
    required section headings, P1–P4, A1–A3, B1–B3, R1–R4 labels, key
    algorithmic terms, cross-references to ``divergence.py``, ``replay.py``,
    and ``docs/canonicalization.md``). All 21 new tests pass; full suite:
    1523 passed, 1 pre-existing skipped (Step 17 ``last_bisect_probes``), 2
    skipped.

55. **Complete.** Prove soundness on paper: every non-dirty replayed step has the same
    observable output as full re-execution under sigma. ``docs/dirty-set-soundness.md``
    (~453 LOC) is the paper-grade companion to the proof sketches in
    ``stepback/divergence.py`` §6 and ``docs/dirty-set.md`` §6. It states
    Theorem 1 (P1 soundness) and Lemma 1 (parent-agreement) formally, proves
    Theorem 1 by strong induction on ``topo(s)`` covering all three cases
    (output-forced dirty, re-executed dirty, clean cache-reuse), enumerates
    exactly where each assumption (A1–A3, R1–R4) is load-bearing with a named
    counterexample for each, and provides a line-level mapping from every proof
    statement to its implementation site in ``replay.py`` / ``divergence.py``.
    ``tests/test_soundness_doc_invariants.py`` (23 tests, all passing) pins
    the document's existence, minimum size, section coverage, assumption
    labelling, induction structure, counterexample presence, implementation
    mapping, and cross-references to ``dirty-set.md``, ``100_STEPS.md``,
    ``dirty-set-completeness.md``, and version pins. All 1546 tests pass
    (1 pre-existing skip unrelated to this step).

56. [x] Mechanize soundness in Lean or Coq for an immutable step DAG and
    collision-free canonical hash assumption.
    ``proofs/lean/Stepback/Soundness.lean`` (Lean 4, kernel-only, no sorry/axiom/
    partial) proves ``Stepback.soundness`` (Theorem 1 / P1) and its corollary
    ``Stepback.cache_reuse_safe`` via strong induction on the topological index, with
    ``DependsOnlyOnParents`` (A1), ``RecorderCoherent`` (R1∧R3), ``CleanSound``, and
    the DAG obligations (R4) as premises.  Trust base is the Lean 4 kernel only (v4.14.0,
    pinned in ``lean-toolchain``).  ``proofs/lean/README.md`` cross-references every
    symbol to the Python implementation and ``docs/dirty-set-soundness.md``.
    ``tests/test_lean_soundness_invariants.py`` (28 tests, all passing) pins file
    existence, absence of sorry/axiom/partial, presence of core definitions and theorems,
    README quality, toolchain version pin, and CI workflow coverage.  All 1574 tests pass
    (1 pre-existing skip/fail unrelated to this step).

57. [x] State completeness separately: every step whose recomputed inputs hash
    differently is included in the dirty set.
    ``docs/dirty-set-completeness.md`` (487 LOC) is the paper-grade
    companion proof of Theorem 2 / P2 (Drift(T,σ) ⊆ D(T,σ) and the three
    other inclusions) with Lemma 2 (classifier exhaustiveness), Lemma 3
    (parent-dirty closure), Corollaries 2 and 3, assumption-tightness audit
    (A3 required; A1, A2, R2 explicitly not needed), counterexamples,
    implementation-site mapping, and cross-references to
    ``stepback/divergence.py``, ``docs/dirty-set-soundness.md`` (Step 55),
    and the Lean mechanization (Step 56). Pinned by 31 new tests in
    ``tests/test_completeness_doc_invariants.py`` covering existence,
    minimum size, Step 57 discharge claim, all four theorem/corollary
    statements, all lemmas, assumption coverage, recorder obligations,
    proof structure, counterexample presence, implementation mapping, and
    cross-references. Baseline: 1 pre-existing failure (Step 17 bisect-
    probes property), 1574 passing. After: same 1 failure, 1605 passing
    (31 new). All 31 new tests pass.

58. [x] Add asymptotic complexity bounds for linear traces, DAG traces, and
    branch-heavy traces, including memory bounds.

59. [x] Implement branch-aware propagation: dirty fan-out children independently,
    dirty joins if any consumed branch output changes, preserve clean siblings.

60. [x] Implement partial recompute for structured inputs so one changed tool result
    does not force re-hashing unrelated subtrees.

61. [x] Add stale-cache detection when a downstream step references a recomputed
    output hash even if direct serialized input appears unchanged.

62. [x] Specify `RaiseSubstitution` semantics and when downstream cache entries are
    invalid after a substituted exception.

63. [x] Record nondeterminism classes for clock, RNG, env, network, and model
    sampling; define when each class forces a dirty step.
    Implemented ``stepback/nondeterminism.py`` with ``NondeterminismClass``
    (str-valued enum: CLOCK, RNG, ENV, NETWORK, MODEL_SAMPLING), five helper
    constructors (``clock_nondeterminism``, ``rng_nondeterminism``,
    ``env_nondeterminism``, ``network_nondeterminism``,
    ``model_sampling_nondeterminism``), ``combine_nondeterminism`` for
    multi-source payloads, and ``forces_dirty`` implementing the dirty-forcing
    semantics: clock/env/network force dirty unless ``controlled=True``; rng
    forces dirty when ``seed=None``; model_sampling forces dirty when
    ``temperature > 0`` and ``seed=None``; unknown classes force dirty
    conservatively; empty or legacy payloads are clean for backward compat.
    Integrated ``forces_dirty`` into both ``stepback/divergence.py``
    (``compute_dirty_set`` adds a new ``"nondeterminism"`` dirty_reason) and
    ``stepback/replay.py`` (cache-hit check gains a ``nondet_class_dirty``
    guard). Updated ``Recorder.llm_call`` to auto-populate
    ``nondeterminism`` with a ``model_sampling`` source from the live
    ``temperature``/``seed`` arguments. Exported all 8 new symbols from
    ``stepback/__init__`` and ``__all__``; updated
    ``tests/test_public_api.py::EXPECTED_PUBLIC_API`` and refreshed
    ``stepback/conformance/api_baselines/v0.1.0``. Added
    ``tests/test_nondeterminism.py`` (60 tests) covering enum values, helper
    shapes, ``forces_dirty`` for every class and edge case, multi-source
    format, malformed-input handling, compute_dirty_set integration,
    replay-engine integration, and auto-recording in ``llm_call``.
    Baseline: 1690 passed, 1 pre-existing failure. After: 1758 passed,
    1 pre-existing failure (Step 17 bisect-probes; unchanged), 2 skipped.

64. [x] Add distributed dirty-set computation over a worker pool; partition by DAG
    regions and merge summaries at joins.
    Implemented ``stepback/distributed_dirty.py`` with ``DagRegion``
    (a labelled, independent subgraph of the trace DAG with its
    external parent set), ``RegionSummary`` (per-region classification
    result), ``partition_dag_regions`` (splits a step list into
    independently-classifiable regions by detecting ``parallel_branch_open``
    / ``parallel_branch_join`` boundaries and assigning each first-level
    branch child its own region), and ``compute_dirty_set_distributed``
    (orchestrates classification over a
    ``concurrent.futures.ThreadPoolExecutor``).  The partitioner emits
    sequential regions for backbone steps (prefix, join+suffix) and one
    branch region per first-level branch child; regions within the same
    fan-out tier have no data dependencies between them and are submitted
    to the pool in parallel.  Each worker receives a read-only snapshot
    of the global state dicts (``outputs_hash_by_id``, ``dirty_by_id``,
    etc.) for its external parents and builds its own local state, so
    concurrent calls are race-free.  After each tier completes,
    ``RegionSummary`` results are merged sequentially into the global
    state before the next tier begins.  The result is *identical* to
    ``compute_dirty_set`` — the distribution is a scheduling optimisation,
    not a semantics change.  All four symbols exported from
    ``stepback/__init__.py``, ``__all__``, and
    ``tests/test_public_api.py::EXPECTED_PUBLIC_API``; API baseline
    refreshed.  Added ``tests/test_distributed_dirty.py`` (35 tests
    covering: empty/linear/parallel partition shapes, external-parent
    membership, full-coverage + no-duplication invariants, correctness
    equivalence with ``compute_dirty_set`` for no-sub/prompt-sub/
    tool-output-sub/multi-sub cases on both linear and parallel traces,
    ``workers=1`` vs ``workers=4`` identity, B1/B2/B3 branch isolation
    (substituting one branch leaves siblings clean, join dirty, dirty
    count = 3 regardless of fan-out width), edge cases (``workers=0``,
    ``workers=100``, ``executor=`` kwarg, topological entry order, P5),
    and public-export checks).  Baseline: 1758 passed, 1 pre-existing
    failure.  After: 1797 passed, same 1 pre-existing failure, 2 skipped.

65. [x] Add ClickHouse dirty-set summary tables for trace id, substitution kind,
    dirty count, clean count, branch count, and calls saved.
    Implemented ``stepback/analytics.py`` with ``DirtySetRecord`` (dataclass
    mapping to one row: ``recorded_at``, ``trace_id``, ``substitution_kind``,
    ``step_count``, ``dirty_count``, ``clean_count``, ``branch_count``,
    ``calls_saved``, ``agent_id``, ``schema_version``),
    ``DIRTY_SET_TABLE_DDL`` (local MergeTree DDL with ``PARTITION BY
    toYYYYMM(recorded_at)`` and ``ORDER BY (trace_id, recorded_at)``),
    ``DIRTY_SET_REPLICATED_TABLE_DDL`` (ReplicatedMergeTree DDL for HA
    clusters), ``record_from_summary`` (factory that builds a
    ``DirtySetRecord`` from a ``DirtySetSummary`` and trace id, including
    dominant substitution kind detection and branch count from
    ``parallel_branch_open`` entries), and ``ClickHouseAnalytics`` (thin
    writer delegating to any client with ``insert``/``command`` methods —
    no real ClickHouse dependency in tests).  All six symbols exported from
    ``stepback/__init__.py`` and ``__all__``; ``EXPECTED_PUBLIC_API`` in
    ``tests/test_public_api.py`` updated; ``tests/test_analytics.py`` adds
    34 tests; API-compat baseline refreshed.  Baseline: 1797 passed, 1
    pre-existing failure, 2 skipped.  After: 1833 passed, same 1 pre-existing
    failure, 2 skipped.

66. [x] Publish empirical dirty-set distributions over synthetic fixtures, public
    benchmark corpora, and anonymized production traces.
    Implemented ``stepback/bench/dirty_set_distributions.py`` with four
    synthetic fixture corpora (``linear_chain``, ``parallel_wide``,
    ``mixed_synthetic``, ``agent_fixture``), three substitution position
    strategies (``random``, ``early``, ``late``), and two substitution types
    (``PromptSubstitution``, ``ToolOutputSubstitution``).  The module
    computes full dirty-fraction distributions — percentile tables (p5..p99)
    and six-bin normalized histograms — via ``compute_dirty_set`` (pure
    analysis; no LLM re-execution).  Pre-computed results (20 trials × 50
    steps × 24 cells = 480 trials) saved to
    ``bench-results/dirty-set-distributions.json``.  Documentation at
    ``docs/dirty-set-distributions.md`` covering methodology, results table,
    interpretation of the late-substitution / parallel-branch benefit, and
    relationship to the README headline claim.  CLI: ``stepback bench
    dirty-set-distributions``.  Added ``CorpusDistribution``,
    ``DistributionSuite``, ``run_distributions`` to ``stepback.bench``.
    Added ``tests/test_dirty_set_distributions.py`` (36 tests covering:
    percentile/histogram invariants, all four corpus builders, all corpus
    cells, suite structure, JSON serialisation, CLI integration, late-trace
    benefit, and parallel B1–B3 isolation).  Baseline: 1834 passed, 1
    pre-existing failure, 2 skipped.  After: 1870 passed, same 1 pre-existing
    failure, 2 skipped.

67. [x] Reconcile the current 12-step fixture result (`dirty_after_sub=11`) with any
    README headline before claiming small dirty sets.
    (Completed: `compute_dirty_set` correctly returns 11/12 for the linear fixture — the O(N−k) worst case;
    small dirty sets arise from parallel branches or late substitutions; the current README makes no specific
    numeric claim. Updated docs/dirty-set.md §9, bench-results/README.md, docs/dirty-set-distributions.md,
    and GROUNDING.md rows 2/4/5/29/139–142. Added 20 pinning tests in
    tests/test_dirty_set_reconciliation.py. 1890 passed, 1 pre-existing failure, 2 skipped.)

## § Replay engine

68. [x] Split replay into planner and executor phases: plan dirty steps and cache
    hits first, then execute with deterministic scheduling.

69. [x] Add deterministic seeding policy for LLM and tool executors, including
    warning levels when providers do not support seeds.

70. [x] Keep an in-process replay API for unit tests and local debugging.
    Added ``stepback/testing/replay.py`` with :class:`CaptureExecutor` (records every
    successful executor callback invocation for test assertions — captures ``kind``,
    deep-copied ``inputs``, deep-copied ``output``, and ``branch_outputs`` per call),
    :class:`FallbackExecutor` (``Executor(fallback_recorded=True)`` alias for local
    debugging without a real LLM/tool stack), and six assertion helpers:
    ``assert_all_cache_hits``, ``assert_dirty_count``, ``assert_real_executions``,
    ``assert_cache_hit_count``, ``assert_step_dirty``, ``assert_step_clean``
    (all raise :class:`AssertionError` with compact, step-level detail on failure,
    including the ``step_id`` and ``kind`` of dirty steps and a list of available
    ids for missing-step errors). All nine symbols are re-exported from
    ``stepback.testing.__init__`` and documented in the package docstring.
    The module docstring distinguishes the four replay paths: in-process with
    :class:`CaptureExecutor` for unit tests, :class:`FallbackExecutor` for
    local debugging, real :class:`~stepback.replay.Executor` for CI/production,
    and ``stepback-proxy`` (Step 71) for remote replay. Added
    ``tests/test_local_replay_api.py`` (29 tests) covering: CaptureExecutor
    type hierarchy, no-calls on clean replay, tool-output substitution captures,
    ``len(cap.calls) == result.real_executions`` invariant, correct kinds, deep-copy
    immutability, :class:`CapturedCall` instances, parallel-branch traces, FallbackExecutor
    semantics, and all assertion helpers including their failure messages and
    missing-step-id handling. Baseline: 1 pre-existing failure, 1948 passed.
    After: 1 pre-existing failure, 1977 passed (29 new tests).

71. [x] Add sidecar replay through `stepback-proxy`: submit trace and substitutions,
    receive replay events over gRPC streaming.
    Added ``replay_events()`` generator in ``stepback/replay.py`` (streams one
    ``step_complete`` event per step + final ``replay_done`` summary, yielding
    ``{"event": "error", ...}`` on failure; mirrors ``_execute_plan`` logic
    exactly). Added ``POST /v1/replay`` NDJSON endpoint in ``stepback/proxy/server.py``
    (accepts ``path``, ``hmac_key_hex``, ``substitutions`` list of
    ``{"step_id", "kind", "value"}`` dicts, ``fallback_recorded``; returns
    ``application/x-ndjson`` with one JSON object per line). Updated
    ``stepback/proxy/proto/sbproxy.proto`` with ``ReplayTrace`` server-streaming
    RPC and ``SubstitutionSpec``/``ReplayEvent`` messages. Updated
    ``stepback/proxy/grpc_server.py`` with ``_replay_trace_stream()`` generator,
    ``_build_substitution_set()`` helper, and ``unary_stream_rpc_method_handler``
    for ``ReplayTrace``. Exported ``replay_events`` from ``stepback.__init__``
    and updated API snapshot + compat baseline. Added ``tests/test_proxy_replay.py``
    with 27 tests (6 unit tests for ``replay_events()``, 14 HTTP endpoint tests,
    7 gRPC unit tests). Baseline: 1 pre-existing failure, 1977 passed.
    After: 1 pre-existing failure, 2005 passed (28 new tests).

72. [x] Add sandboxed replay modes using gVisor and Firecracker for untrusted tool
    execution.
    Implemented ``stepback/sandbox.py`` with four isolation levels:
    ``SandboxMode.NONE`` (direct in-process, default), ``SandboxMode.SUBPROCESS``
    (isolated forked process with wall-clock timeout and POSIX resource limits via
    ``RLIMIT_AS`` / ``RLIMIT_CPU``; falls back to spawn on Windows), ``SandboxMode.GVISOR``
    (gVisor ``runsc`` container isolation — requires binary + ``tool_runner_argv``; raises
    ``SandboxUnavailableError`` when ``runsc`` absent), and ``SandboxMode.FIRECRACKER``
    (Firecracker MicroVM isolation — requires binary, ``/dev/kvm``, kernel/rootfs paths +
    ``tool_runner_argv``; raises ``SandboxUnavailableError`` when prereqs absent).
    ``SandboxConfig`` dataclass holds all tunables.  Exception hierarchy:
    ``SandboxError`` → ``SandboxTimeoutError``, ``SandboxResourceError``,
    ``SandboxUnavailableError``, ``SandboxViolationError``.  ``SandboxedExecutor``
    subclasses ``Executor``, sandboxes only ``tool_call`` steps (LLM/router/join bypass
    directly), tracks ``real_calls`` without double-counting the base executor.
    SUBPROCESS mode uses ``multiprocessing.Process`` + ``Pipe`` with ``fork`` context
    on POSIX (avoiding pickling of arbitrary callables) and ``ProcessPoolExecutor`` with
    ``spawn`` on Windows.  GVISOR/FIRECRACKER use a JSON stdin/stdout protocol with the
    caller-supplied tool runner command.  ``create_sandbox(config, base)`` factory.
    All nine public symbols exported from ``stepback.__init__`` and ``__all__``;
    ``tests/test_public_api.py::EXPECTED_PUBLIC_API`` and
    ``stepback/conformance/api_baselines/v0.1.0/public_api.json`` updated.
    ``tests/test_sandbox.py`` (43 tests) covers: enum/config/exception hierarchy, NONE/
    SUBPROCESS/GVISOR/FIRECRACKER mode dispatch, timeout enforcement, tool-exception
    propagation, unavailability detection (monkeypatched ``shutil.which``), factory,
    public API exports, and end-to-end replay integration for both NONE and SUBPROCESS modes.
    Baseline: 1 pre-existing failure, 2005 passed.  After: 1 pre-existing failure (unchanged),
    2057 passed (52 new), 2 skipped.

73. [x] Add parallel-branch scheduling so independent dirty branches execute in
    parallel and joins wait only on consumed inputs.
    Implemented ``_execute_branch_steps`` (executes one branch region's steps
    sequentially against a snapshot of global state) and
    ``_execute_plan_parallel`` (partitions the trace DAG via
    ``partition_dag_regions``, submits branch tiers to a
    ``concurrent.futures.ThreadPoolExecutor``, merges results, and assembles
    the final ``ReplayResult`` in original topological order) in
    ``stepback/replay.py``.  Added ``workers: Optional[int]`` parameter to
    ``Trace.replay_forward``, ``Trace.run_replay``, ``Branch.replay_forward``,
    and ``ReplayPlan.execute``; ``workers=None`` / ``workers=1`` → sequential
    (no thread overhead), ``workers=N`` → parallel branch execution.
    ``Executor.real_calls`` / ``fallback_uses`` are protected by a
    ``threading.Lock`` (held only for the counter increment, not for callback
    execution, so independent branches truly run in parallel).  The sequential
    path is unchanged.  Refreshed the ``api_baselines/v0.1.0`` snapshot to
    record the four new ``workers`` parameters.  Added
    ``tests/test_parallel_replay.py`` (15 tests covering: no-substitution
    parallel ≡ sequential, substitution result equivalence, step order
    preservation, workers=1 ≡ workers=None, dirty-set isolation (one branch
    substitution → only that branch + join + successor dirty; siblings clean),
    dirty_count constant across branches, join waits for all branches,
    ``ReplayPlan.execute(workers=N)``, linear trace with no branches, accurate
    ``real_calls`` / ``fallback_uses`` under concurrency, callback thread-id
    verification, empty trace, ``MissingExecutor`` propagation from branch
    worker).  Baseline: 1 pre-existing failure, 2057 passed.  After: same
    1 pre-existing failure, 2072 passed (15 new tests).

74. [x] Add sharded content-addressed step cache backed by disk, S3, GCS, Azure
    object storage, and dedup across traces.
    Implemented ``stepback/step_cache.py`` (Step 74) with ``StepCacheEntry``
    (dataclass: ``step_kind``, ``inputs_hash``, ``outputs``, ``cached_at``; round-trips
    through JSON with ``cache_schema_version`` / ``canonicalisation_version`` guards so
    stale entries from future upgrades are silently treated as misses), ``StepCache``
    (ABC: ``get(kind, inputs_hash)``, ``put(entry)``, ``close()``), ``DiskStepCache``
    (stdlib-only; shards entries under ``<root>/<kind>/<shard>/<hex>.json`` using atomic
    write-then-rename so concurrent branch workers never see partial files; per-directory
    ``threading.Lock`` guards same-process concurrent puts), ``S3StepCache`` (requires
    ``boto3``; raises ``ImportError`` with install hint if absent), ``GCSStepCache``
    (requires ``google-cloud-storage``), and ``AzureStepCache`` (requires
    ``azure-storage-blob``; accepts either ``connection_string`` or ``account_url``).
    Cache key is composite ``kind/<hex_digest>`` (hex digest extracted from the
    ``sha256:`` prefix of ``hash_obj()`` output so colons never appear in paths/keys).
    Integrated into ``stepback/replay.py``: ``Executor.__init__`` gains a
    ``step_cache: Optional[StepCache] = None`` parameter (appended after ``fallback_recorded``
    — fully backward-compatible) plus ``_cache_get`` / ``_cache_put`` helpers. All three
    replay paths (``_execute_plan``, ``_execute_branch_steps``, ``replay_events``) check the
    step cache before calling the executor when a step is dirty due to changed inputs /
    dirty ancestors; nondeterminism-class-forced steps (Step 63 — unseeded RNG, uncontrolled
    clock, etc.) bypass the cache entirely; ``tool_override`` (substituted outputs) and
    ``fallback_recorded`` outputs are never written to the cache; real executor calls write
    their outputs to the cache. Step-cache hits do not increment ``real_executions``.
    Dedup across traces is automatic: same ``step_kind`` + same ``inputs_hash`` → same
    cache entry regardless of which trace triggered it. All six public symbols
    (``StepCacheEntry``, ``StepCache``, ``DiskStepCache``, ``S3StepCache``,
    ``GCSStepCache``, ``AzureStepCache``) exported from ``stepback/__init__.py`` and
    ``__all__``; ``tests/test_public_api.py::EXPECTED_PUBLIC_API`` and
    ``stepback/conformance/api_baselines/v0.1.0/`` refreshed. Added
    ``tests/test_step_cache.py`` (40 tests) covering: entry round-trip and schema
    mismatch rejection, ``_hex_digest`` / ``_cache_key`` helpers, ``DiskStepCache`` put/get/
    miss/shard-directory creation/shard-width variants/corrupted-JSON-as-miss/
    incompatible-schema-as-miss/atomic-write/concurrent-puts/dedup/kind-isolation,
    cloud-backend ``ImportError`` gates (S3 / GCS / Azure), ``Executor`` cache helpers,
    integration tests confirming step cache reduces ``real_executions`` to 0 on warm
    cache, nondeterminism bypass, tool-override non-pollution, fallback non-pollution,
    ``replay_events`` step cache usage, parallel-branch replay step cache usage, and
    cross-trace dedup. Baseline: 1 pre-existing failure, 2072 passed. After: same 1
    pre-existing failure, 2118 passed (40 new tests added by this step, +6 from API
    compat refresh).

75. [x] Add Kafka-backed event bus for replay jobs and step-complete events; make
    the planner idempotent under worker retries.

76. [x] Add worker leases and checkpointed replay state so million-point sweeps
    survive worker failure.
    Implemented ``stepback/worker_lease.py`` with ``LeaseStatus`` (str-enum:
    CLAIMED, EXPIRED, RELEASED), ``WorkerLease`` dataclass (lease_id,
    work_unit_id, worker_id, claimed_at, expires_at; ``is_expired`` /
    ``ttl_remaining`` properties; ``to_dict`` / ``from_dict`` round-trip),
    ``LeaseExpiredError``, ``LeaseRegistry`` ABC (``try_claim``, ``renew``,
    ``release``, ``list_expired``), ``InMemoryLeaseRegistry`` (threading.Lock;
    thread-safe concurrent-claim tested with 20 threads), and
    ``DiskLeaseRegistry`` (``open(mode='x')`` atomic exclusive creation on
    POSIX; atomic rename for renewal; per-path sanitization so slashes never
    create subdirectories).  Implemented ``stepback/sweep_checkpoint.py``
    with ``CheckpointEntryStatus`` enum, ``CheckpointEntry`` dataclass
    (status, worker_id, lease_id, lease_expires_at, result, failure;
    ``lease_expired`` property), ``SweepCheckpoint`` ABC, and
    ``DiskSweepCheckpoint`` (one JSON file per trace using the first 16 hex
    chars of ``sha256(trace_path)`` as filename; atomic write-then-rename
    so concurrent workers never read partial files; ``initialise`` is
    idempotent; ``pending_paths`` returns PENDING entries plus IN_PROGRESS
    entries whose leases have expired so dead-worker work is automatically
    surfaced for reclaim; ``partial_report_data`` provides already-completed
    results for cross-restart merging).  Added ``resume_sweep`` which wraps
    ``sweep_traces`` with checkpoint recovery: reads the checkpoint to skip
    already-completed traces, claims a lease before processing each trace
    (skips if another worker holds an active lease), persists each result or
    failure to disk immediately after processing, merges checkpoint-recovered
    results into the final ``SweepReport`` so reports are always complete
    regardless of how many restarts occurred; ``retry_failed=True`` opt-in
    to re-process previously failed traces.  All 11 new symbols exported from
    ``stepback/__init__.py`` and ``__all__``; ``EXPECTED_PUBLIC_API`` and
    ``stepback/conformance/api_baselines/v0.1.0/public_api.json`` updated.
    Added ``tests/test_worker_lease.py`` (27 tests) and
    ``tests/test_sweep_checkpoint.py`` (20 tests) covering: enum stability,
    ``WorkerLease`` property assertions, InMemoryLeaseRegistry basic/double-
    claim/expiry/renew/release/thread-safety/unique-lease-ids, DiskLeaseRegistry
    basic/double-claim/expiry/renew/release/list_expired/file-persistence/
    release-removes-file/path-sanitization, CheckpointEntry round-trip and
    ``lease_expired``, DiskSweepCheckpoint initialise/idempotent/pending_paths/
    expired-in-progress/active-in-progress/mark_completed/mark_failed/
    partial_report_data, and resume_sweep basic/skip-completed/fail-skipped-
    by-default/retry_failed/worker-isolation (concurrent threads, verified via
    checkpoint state)/checkpoint-persists-across-calls/default-worker-id/
    on_error-raise.  Baseline: 1 pre-existing failure, 2161 passed.  After:
    same 1 pre-existing failure, 2219 passed (58 new, including API-compat
    suite fully green), 2 skipped.

77. [x] Add `Trace.replay_forward(distributed=True, workers=N)` backed by the same
    planner as local replay.
    Added ``distributed: bool = False`` keyword-only parameter to
    ``Trace.replay_forward``, ``Trace.run_replay``, ``Branch.replay_forward``,
    and ``ReplayPlan.execute``.  Added ``_effective_workers(distributed, workers)``
    helper in ``stepback/replay.py`` that auto-selects
    ``min(8, os.cpu_count() or 4)`` when ``distributed=True`` and *workers* is
    not specified.  When ``distributed=True`` the same local
    ``ThreadPoolExecutor`` planner (``_execute_plan_parallel``) is used, backed
    by the same ``partition_dag_regions`` plan as ``workers=N`` (Step 73).
    ``ReplayPlan.execute`` falls through to the sequential path when
    ``_event_bus`` is set (the parallel planner does not publish per-step
    events).  API baseline refreshed via ``scripts/check_api_compat.py
    --write-baseline``.  Added ``tests/test_distributed_replay.py`` (29 tests)
    covering: ``_effective_workers`` unit tests (all six flag/workers combos,
    cpu_count=1024 cap, cpu_count=None fallback), ``Trace.replay_forward``
    (no-sub all-cache-hits, distributed+workers=N matches workers=N alone,
    matches sequential, workers=1, auto-worker count), ``Trace.run_replay``
    forwarding, ``Branch.replay_forward`` forwarding, ``ReplayPlan.execute``
    (including event-bus → sequential invariant), linear-trace correctness, and
    API-signature tests for all four entrypoints.
    Baseline: 2219 passed, 1 pre-existing failure.  After: 2248 passed, same
    1 pre-existing failure, 2 skipped.

78. [x] Emit replay provenance: executor versions, cache hits, dirty reasons, seeds,
    policy decisions, model versions, and provider versions.
    Added ``StepProvenance`` (per-step: dirty_reason, cache_source, seed, model,
    provider, model_version, provider_version, policy_blocked, policy_reason,
    executor_version) and ``ReplayProvenance`` (started_at, finished_at,
    executor_version) to ``stepback/replay.py``.  Both are populated by all
    replay code paths (sequential ``_execute_plan``, parallel
    ``_execute_plan_parallel``, streaming ``replay_events``).  ``replay_events``
    now includes ``dirty_reason`` and ``executor_version`` in each
    ``step_complete`` event.  Both classes exported from ``stepback`` and
    ``__all__``.  API baseline updated.  30 new tests in
    ``tests/test_replay_provenance.py``.
    Baseline: 2214 passed, 2 pre-existing failures.  After: 2280 passed, 1 pre-existing failure, 2 skipped.

79. [x] Add a web time-travel debugger with step forward/back, cache-hit display,
    canonical input diffs, and causal graph view.

## § Minimization

80. [x] Promote predicates to a typed DSL with boolean composition, metric
    thresholds, regexes, policy decisions, and Python callbacks.

81. [x] Add multi-objective ddmin for trace length, LLM calls, cost, latency, and
    policy-violating steps.

82. [x] Add statistical stability metrics: repeated dirty replays, confidence
    intervals, flaky predicate classification, and warnings.

83. [x] Add incremental bisect across multiple regressions so finding one culprit
    does not restart the search.

84. [x] Add Shapley-style attribution for steps that jointly cause a failure.

85. [x] Add minimization over branches: drop independent branches safely, preserve
    joins only when consumed outputs matter.

86. [x] Add minimization for imported traces where executable replay is partial;
    mark steps requiring unavailable executors.
    **Complete.** Added `UnavailableExecutorError`, `StepExecutorRequirement`,
    `PartialExecutor`, `audit_executor_requirements` to `stepback/replay.py`;
    added `skip_unavailable_executors` to `MinimizeOptions` and
    `minimize_imported_trace()` to `stepback/minimize.py`; exported all 6 new
    symbols; added 44-test coverage in `tests/test_minimize_imported.py`.

87. [x] Add HTML minimization reports with before/after graphs, removed steps,
    predicate evaluations, and confidence summaries.
    Implemented ``stepback/minimize_report.py`` with ``MinimizeReportOptions``
    (dataclass: title, show_before_after, show_minimal_substitutions,
    show_removed_substitutions, show_probe_stats, show_attribution,
    show_final_result, show_pareto_front, html_inline_css, max_step_rows,
    truncate_text, extra_metadata) and ``render_html_minimize_report``
    (accepts any ``MinimizationResult`` or ``MultiObjectiveMinimizationResult``;
    produces a byte-deterministic, fully self-contained offline HTML document
    with no external CDN dependencies). Report sections: **Summary** (strategy,
    substitution counts, oracle probes, cache hits, extra metadata in sorted-key
    order); **Before / after graph** (CSS-only horizontal bars comparing original
    vs minimal substitution count, plus replay step count / cost / real-executions
    from ``final_result`` when present); **Minimal substitutions** table (kind,
    at-step, summary, optional Shapley weight column); **Removed substitutions**
    table (proved unnecessary); **Probe statistics** (probes, cache hits, cache
    hit rate, total subset evaluations); **Attribution / confidence summary**
    (per-substitution Shapley weights ranked by descending weight, with a
    "run ShapleyAttributionStrategy" hint when no weights are attached);
    **Final replay** step table (step-id, kind, name, dirty/cached badge, cost)
    gated by ``show_final_result`` and the presence of ``final_result``; and a
    **Pareto front** section for ``MultiObjectiveMinimizationResult`` showing
    each non-dominated (subset-size, objective-values) row. All user-controlled
    text (substitution summaries, step names, metadata values, strategy names)
    passes through ``html.escape(..., quote=True)``. Both symbols exported from
    ``stepback/__init__.py`` and ``__all__``; ``EXPECTED_PUBLIC_API`` in
    ``tests/test_public_api.py`` and ``stepback/conformance/api_baselines/v0.1.0/``
    refreshed via ``scripts/check_api_compat.py --write-baseline``. Added
    ``tests/test_minimize_report.py`` (52 tests, all passing) covering: smoke /
    HTML structure; all six required section ids; summary content and metadata
    key ordering; before/after count correctness; substitution table listing;
    probe statistics and division-by-zero guard; attribution with and without
    weights; final result step table; HTML escaping of XSS payloads in strategy
    name, metadata, and model ids; byte-determinism; all nine option flags;
    multi-objective Pareto-front rendering and opt-out; default-options contract;
    and public-API export checks. Baseline: 2534 passed, 2 skipped.
    After: 2563 passed, 3 skipped (52 new + API-compat refresh),
    24 pre-existing failures (Lean/changelog invariants, unchanged).

88. [x] Write the minimization paper artifact with algorithm, stochastic
    assumptions, failure modes, and empirical comparison to naive ddmin.
    Created ``docs/minimization-paper.md`` (~28 KiB, 14 sections) covering:
    formal setup (trace, substitution set, predicate, oracle), oracle stability
    assumptions (A_oracle, A_pred, A_monotone, A_cache), oracle cache and budget
    guards, five strategy descriptions with pseudocode and probe complexity
    (DDMin O(n²) worst / O(n log n) typical / 1-minimal under A_monotone;
    Linear O(n+1) / 1-minimal; Binary O(n log n) / heuristic-not-guaranteed;
    BruteForce O(2^n) / globally-minimal; Shapley exact 2^n / sampled p·n /
    attribution-not-witness), strategy comparison table, multi-objective
    extension, failure modes (PredicateNotTriggered, BudgetExhausted, flaky
    predicate instability with Wilson confidence intervals, UnavailableExecutorError /
    partial traces, degenerate empty-witness / all-minimal cases), stochastic
    assumptions and predicate stability classes (structural / cost / content /
    policy), empirical comparison to naive ddmin (memoization savings, dirty-set
    cache amplification, synthetic probe-count table, dirty-fraction cache-hit
    rates from bench-results/), multi-witness enumeration, imported-trace
    minimization, implementation mapping table, and versioning policy. Added
    ``tests/test_minimize_paper_invariants.py`` (33 tests) pinning: file
    existence/size, Step 88 discharge claim, all five strategy names, oracle
    cache/probe counting, A_oracle/A_pred/A_monotone assumption coverage, stochastic
    section/predicate stability classes/confidence intervals, all four failure modes,
    complexity table/O(n²) mention, empirical comparison/probe table/cache hit rates,
    multi-witness, implementation mapping, cross-references, Shapley-as-attribution
    distinction, binary-as-heuristic flag. Baseline: 24 pre-existing failures,
    2563 passed. After: 24 pre-existing failures (unchanged), 2596 passed (33 new).

## § Shims and framework recorders

89. [x] Refactor provider shims behind a `ShimContract` ABC: canonical request,
    canonical response, executor, streaming hooks, async hooks, version probe.

90. [x] Add OpenAI contract tests for chat, responses, tool calls, streaming, async
    clients, and OpenAI-compatible endpoints.

91. [x] Add Anthropic contract tests for messages, tool use, streaming, thinking
    blocks if present, and async clients.

92. [x] Harden Bedrock and Gemini shims against real SDK versions with recorded
    cassettes and compatibility matrices. Added ``tests/cassettes/`` directory with
    10 JSON fixture files shaped like real boto3 (>=1.34.0) and google-genai (>=0.8.0)
    SDK responses: 5 Bedrock cassettes (text, tool_use, native-Llama, max_tokens,
    guardrail_intervened) and 5 Gemini cassettes (text, function_call, cached_tokens,
    safety_blocked, Vertex-AI shape). Added ``tests/cassettes/COMPAT_MATRIX.json``
    documenting the SDK version, API version, model, and content-type for every
    cassette. Added ``tests/test_shim_cassettes.py`` (33 tests) covering
    ``_bedrock_to_openai_shape`` and ``_gemini_to_openai_shape`` end-to-end for
    every cassette shape, all three coercion code paths (dict, model_dump, to_dict,
    duck-typed attributes), the ``GeminiResponse`` namespace fields, and a
    consistency check that every file listed in the matrix exists and every file in
    the directory appears in the matrix. All 33 new tests pass; 24 pre-existing
    failures (Lean soundness + deprecation) unchanged.

93. [x] Add Azure OpenAI support, including deployment-name model ids and regional
    endpoint metadata.
    Implemented ``stepback/shims.py`` additions: ``wrap_azure_openai(client,
    recorder, *, default_deployment, underlying_model, endpoint, seed_policy,
    contract)`` — an ``openai.AzureOpenAI``-compatible wrapper that intercepts
    ``chat.completions.create(model=<deployment_name>, ...)`` calls, records
    each as an ``llm_call`` step with canonical model id
    ``canonical_azure_model_id(deployment_name, underlying_model=...)`` (which
    resolves to the OpenAI pricing id when an *underlying_model* is declared,
    or falls back to ``"azure:{deployment_name}"`` for zero-cost recording when
    the deployment-to-model mapping is unknown). ``azure_openai_executor(client,
    *, deployment_name)`` provides the matching replay executor that always
    passes ``deployment_name`` to the real Azure API (ignoring the canonical
    model id stored in the trace). ``canonical_azure_model_id`` prevents
    cross-deployment cache collisions by encoding the deployment name in the
    canonical id when no underlying model is declared. ``WrappedAzureOpenAI``
    exposes read-only ``endpoint`` and ``deployment_name`` properties for
    diagnostics; the endpoint URL is stored as display metadata only (NOT
    hashed into the step) so regional migrations do not invalidate the cache.
    ``AzureOpenAIShimContract`` provides the contract ABC for Azure (same
    canonicalisers as OpenAI); ``make_executor`` raises ``NotImplementedError``
    with guidance to use ``azure_openai_executor(client, deployment_name=…)``
    directly. All six new symbols (``wrap_azure_openai``,
    ``azure_openai_executor``, ``canonical_azure_model_id``,
    ``WrappedAzureOpenAI``, ``AzureOpenAIShimContract``) exported from
    ``stepback.__all__`` and pinned in ``tests/test_public_api.py``; API
    baseline refreshed. ``tests/test_azure_openai_shim.py`` (24 tests)
    covers: record→replay 100% cache hit, deployment name used in API call,
    per-call override, default fallback, no-deployment ValueError, substitution
    + dirty replay, pricing with and without underlying_model, no
    cross-deployment cache collision, endpoint/deployment_name properties,
    attribute passthrough, invalid-client TypeError, executor deployment-name
    guarantee, contract registration/canonical_request/canonical_response,
    make_executor NotImplementedError, and streaming accumulation.
    Baseline: 2729 passed, 3 skipped. After: 2760 passed, 3 skipped (31 new
    including 24 Azure shim + 7 from public-API snapshot delta).

94. [x] Add Vertex AI / Google GenAI support beyond Gemini; canonicalize safety
    settings and tool declarations.

95. [x] Add Cohere and Mistral shims with dedicated canonicalizers.

96. [x] Add Together, Fireworks, Groq, Cerebras, NVIDIA NIM, vLLM, TGI,
    llama.cpp, and Ollama adapters.

97. [x] Add streaming recorder support where chunks are part of one LLM-step receipt
    and final response hashing is deterministic.

98. [x] Add async recorder support for Python and JS SDKs; prove context
    propagation survives `await` and task groups.
    **Complete.** Added `_RECORDER_VAR` and `_PARENT_STEP_VAR` ContextVars to
    `stepback/recorder.py`; added `arecord()` async context manager that sets
    both ContextVars so the recorder and task-local parent step id propagate
    through every `await` boundary and into child tasks spawned by
    `asyncio.create_task` / `asyncio.TaskGroup` / `asyncio.gather` (Python's
    standard ContextVar copy-on-create semantics). Added `get_current_recorder()`
    returning the ContextVar value (or `None` outside any recording block). The
    synchronous `record()` now also sets the ContextVars for consistency. Updated
    `Recorder._record()` to advance `_PARENT_STEP_VAR` after each step and
    added `_current_parent()` that prefers the task-local ContextVar over
    `self._parent`, preserving isolation between concurrent async tasks that
    share one recorder. Fixed `Recorder.parallel()` to keep `_PARENT_STEP_VAR`
    in sync when re-parenting each branch. Updated `autorecord.py`: ContextVar
    is set/reset with proper tokens in `enable()`, `_patch_openai()` now wraps
    `AsyncOpenAI` under `aenable()` instead of warning, `_patch_anthropic()`
    wraps `AsyncAnthropic` under `aenable()`, new `aenable()` async context
    manager patches both sync and async provider constructors. Exported `arecord`
    and `get_current_recorder` from `stepback.__init__` and `__all__`; updated
    `tests/test_public_api.py` and `stepback/conformance/api_baselines/v0.1.0/`.
    Added `tests/test_async_recorder.py` (20 tests) covering: basic arecord(),
    get_current_recorder() identity, None outside block, sync record() sets ContextVar,
    ContextVar reset on exit, context survives await + multiple awaits, context in
    create_task + gather + TaskGroup (py311), two isolated recorders, parallel tasks
    with task-local parent chains, steps written to recorder.steps, sequential
    parent chain, autorecord.aenable() basic + task propagation + re-entrancy,
    and public export assertions. Baseline: 24 pre-existing failures (Lean +
    changelog), 962 passed. After: 24 pre-existing failures (unchanged), 3008
    passed (20 new + public-API + conformance baseline refresh).

99. [x] Enforce recorder overhead budgets in CI: p50 under 50 microseconds for the
    Python fast path, with shim-specific exceptions documented.
    **Complete.** Updated `stepback/bench/record_overhead.py`: added `n_warmup`
    parameter (default 50) and two new fields `delta_p50_us` / `delta_p99_us`
    (difference of percentiles: recorded_pN − baseline_pN); updated `to_json()`
    and `summary_line()`; added module-level `OVERHEAD_BUDGET_TARGET_US=50`,
    `OVERHEAD_BUDGET_CI_US=500`, and `overhead_budget_us()` helper that reads
    `STEPBACK_OVERHEAD_BUDGET_US` env var, falling back to the CI guardrail when
    `GITHUB_ACTIONS=true` and the 50 µs target otherwise. Added
    `tests/test_recorder_overhead_budget.py` (5 tests, marked
    `@pytest.mark.overhead_budget`) that enforce the budget, verify delta-field
    monotonicity, JSON/summary-line presence, and env-var override. Shim-specific
    exceptions documented in module docstrings (wrap_openai/anthropic/bedrock/gemini
    ≤100 µs, streaming shims exempt, tool shims ≤100 µs). Registered
    `overhead_budget` marker in `pyproject.toml [tool.pytest.ini_options]`.
    Updated `.github/workflows/ci.yml`: normal matrix uses `-m "not overhead_budget"`,
    coverage-floors job likewise; new dedicated `overhead-budget` job runs
    `pytest -m overhead_budget` with `STEPBACK_OVERHEAD_BUDGET_US=500`. Existing
    test counts unchanged: 5 pre-existing failures, 982 passed, 5 deselected.

100. [x] Create a certified-shim program: contract tests, overhead report,
     canonicalization review, version matrix, and signed compatibility badge.

101. [x] Add LangChain and LangGraph recorders using callback / run-manager hooks,
     preserving run ids as trace metadata.

102. [x] Add LlamaIndex, DSPy, Haystack, AutoGen, CrewAI, Semantic Kernel, Strands,
     Pydantic-AI, Inspect-AI, and MCP recorders with minimal examples.

103. [x] Add `stepback diagnose` to inspect installed SDK/framework versions and
     warn when newer than the certified matrix.

## § Importers/exporters

104. [x] Complete importers for Phoenix, Helicone, Langfuse, and Datadog APM; map
     their spans into SB-Trace step kinds with explicit lossy fields.

105. [x] Harden LangSmith and OpenInference importers with real exported fixtures,
     schema-version detection, and hash-stability tests.

106. [x] Add OpenTelemetry import using stable semantic conventions for
     `agent.step`; keep OpenInference as a compatibility profile.

107. [x] Add OTel export with `agent.step.*` attributes suitable for upstream
     proposal to OpenTelemetry semantic conventions.

108. [x] Add JSON export with a stable schema and compatibility tests.

109. [x] Add self-contained HTML export with causal graph, diff panes, minimization
     report, and attestation summary.

110. [x] Add CycloneDX-AI export linking traces to models, prompts, tools, datasets,
     and policy decisions.

111. [x] Add SLSA and in-toto provenance attestations for traces, benchmark
     submissions, and incident replay packs.

112. [x] Add lossiness reports to every importer/exporter: absent, approximated,
     synthesized, and dropped fields.

## § Benchmarks

113. [x] Turn `scripts/bench_replay_caching.py` into `stepback bench
     replay-caching` without `PYTHONPATH=.` and with JSON output.

114. [x] Define result schema: corpus id, trace count, substitution distribution,
     dirty-set stats, cache hits, LLM calls saved, latency, cost, storage,
     versions, and hardware.

115. [x] Add corpus loaders for SWE-bench-Verified, GAIA, tau-bench, AgentBench,
     OSWorld, and WebArena.

116. [x] Create three author-original corpora: support agent, code-review agent, and
     policy-gated payments agent, all redistributable as `.sb`.

117. [x] Add anonymized production-trace ingestion rules: redaction, hash
     preservation, privacy review, and redaction attestation.

118. [x] Add replay-caching benchmark: cost reduction, wallclock speedup, dirty-set
     distribution, and cache-hit reasons.

119. [x] Add minimization benchmark: final trace size, predicate stability, LLM calls
     spent, and comparison to naive ddmin.

120. [x] Add model-swap differential benchmark: fidelity against full re-execution
     and statistically grounded difference detection.

121. [x] Add recorder-overhead benchmark for every shim and framework recorder;
     enforce p50/p95 budgets in CI.

122. [x] Add storage-compression benchmark: raw JSON, `.sb` v1, CBOR candidate,
     zstd, deduped object-store layout, and query-index overhead.

123. [x] Publish MLPerf-style submission rules: frozen code, signed trace pack,
     hardware manifest, exact commands, validator output, and audit rights.

124. [x] Add hosted leaderboard generation from signed JSON submissions; reject
     submissions that fail conformance or attestation checks.

125. [x] Add scheduled frontier-model re-evaluation so results do not fossilize
     around one provider generation.

126. [x] Write the NeurIPS-Datasets benchmark paper with corpus documentation,
     licensing, metrics, limitations, and reproduction instructions.

## § Attestation & cryptography

127. [x] Write `SECURITY.md` threat model: what HMAC/signatures prove, what they do
     not prove, key handling, disclosure path, and verifier guarantees.

128. [x] Add key rotation for trace attestations: preserve old signatures, append a
     rotation frame, and verify both trust chains.

129. [x] Add post-quantum signature experiments with ML-DSA and SLH-DSA behind
     explicit capability frames.

130. [x] Add threshold signing for long-lived production recorders with M-of-N
     witnesses for incident-grade traces.

131. [x] Add witness cosigning for public benchmark traces so leaderboard entries
     prove trace packs existed before evaluation.

132. [x] Add transparency-log integration for incident records and benchmark packs;
     store inclusion proofs in attestation packs.

133. [x] Add hardware-backed key support via PKCS#11, YubiHSM, and cloud KMS, with
     tests using software simulators.

134. [x] Add `stepback verify --strict --policy <policy>` to verify cryptography,
     schema, canonical bytes, and recorder identity in one command.

## § Performance & scale

135. [x] Add microbenchmarks for canonicalization, frame writing, HMAC/signing,
     recorder hooks, reader throughput, and dirty-set planning.

136. [x] Optimize recorder fast path to p50 under 50 microseconds per call without
     signing and document the budget with signing enabled.

137. [x] Add batch-signing / async-signing mode for high-throughput recorders while
     preserving append-only ordering guarantees.

138. [x] Implement sharded step cache on object storage with content-addressed dedup
     across runs, corpora, and organizations.

139. [x] Add ClickHouse schema for trace queries by model, step kind, dirty reason,
     cost, policy decision, canonical hash, and incident id.

140. [x] Add distributed bisect across a worker pool for large trace sets and
     multi-objective predicates.

141. [x] Add load tests simulating millions of agent runs per day through
     `stepback-proxy`; publish CPU, memory, storage, and p95 latency curves.

142. [x] Add backpressure and sampling controls so recorder failure cannot take down
     the agent unless configured as mandatory.

## § Specification & standards

143. [x] Create `spec/rfcs/0001-sbtrace-core.md` plus RFCs for canonicalization,
     dirty-set semantics, attestation packs, importer lossiness, and OTel
     `agent.step` semantic conventions.

144. [undoable] Submit `agent.step` semantic conventions upstream to OpenTelemetry; keep
     the exporter aligned with review feedback and publish conformance status
     for Python, Rust, TypeScript, Go, JVM, .NET, proxy, and WASM.
     (note: requires human interaction with OpenTelemetry community and governance process; cannot be automated.)

145. [x] Add formal standards artifacts: TLA+ spec of the `.sb` HMAC chain,
     conformance dashboard, Linux Foundation proposal, and CNCF sandbox draft
     once multi-implementation production use exists.

## § Community, governance, release engineering

146. [in-progress: 2026-05-13T14:24:00Z] Add `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue templates, PR template,
     security advisory flow, Dependabot/Renovate, CI, release workflows,
     changelog, governance docs, and certified-integration rules.

## § Research artifacts & papers

147. [x] Write `RELATED_WORK.md`, `ARTIFACT.md`, and the 4-5 paper line: dirty-set
     algorithm, distributed replay runtime, stochastic minimization,
     benchmark/dataset paper, and incident/audit case-study paper.

## § Production case studies

148. [x] Build production-shaped case studies for millions-of-runs/day recording,
     incident replay, regulator/auditor evidence packs, model migration, large
     parameter sweeps, and redacted trace publication.

## § Ecosystem integrations

149. [x] Integrate with ragdoctor, flowwarden, and toolwarden; record diagnostic RAG
     runs, provenance/IFC labels, enforcement decisions, denied calls, policy
     versions, and replay-time audits.

150. [x] Add MCP recorder/proxy mode, CycloneDX-AI interop, SLSA/in-toto examples,
     observability bridges back to OTel warehouses, WASM viewer embedding docs,
     and a public integration matrix.


## § Onboarding & first-run experience

151. [x] Add `stepback init` that scaffolds a `stepback.toml`, an example agent
     under `examples/quickstart/`, a recorded `.sb`, and a one-shot
     `stepback replay` invocation; bias defaults so a brand-new user gets a
     green run in under 60 seconds without reading the README.

152. [x] Add `stepback doctor` that checks Python version, optional Rust/WASM
     components, key material, writable trace dir, network reachability of
     configured providers, and prints a single PASS/WARN/FAIL table with copy-
     pasteable remediation for every WARN/FAIL row.

153. [x] Add `stepback quickstart` interactive wizard (prompt-toolkit) that walks
     a new user through picking a provider shim, dropping in their API key
     into a local keyring (never the repo), recording one trace, and opening
     it in the HTML viewer.

154. [x] Ship a `curl https://get.stepback.dev | sh` installer that detects OS
     and arch, installs the wheel + CLI shim into `~/.local/bin`, refuses to
     run as root by default, and prints next-step commands.

155. [x] Publish a Homebrew tap (`stepback/tap`) with formulae for `stepback`
     CLI, `stepback-core` Rust verifier, and the WASM viewer; CI bumps the
     tap on every tagged release with checksums.

156. [x] Publish an official Docker image (`ghcr.io/stepback/stepback:<version>`)
     and a `-slim` variant; include a sample `docker run` recipe that records
     and replays inside the container with a mounted trace volume.

157. [x] Publish a VS Code devcontainer (`.devcontainer/`) and a GitHub
     Codespaces "Open in Codespaces" badge in the README, both pre-loaded
     with the quickstart agent and the viewer port forwarded.

158. [x] Add an `examples/` gallery: customer-support agent, RAG pipeline,
     tool-using agent, multi-step planner, parallel branch agent, and
     long-running batch worker — each with a recorded reference trace and a
     `make` target that re-records it.

159. [x] Rewrite the README's first 30 lines as a single copy-pasteable
     quickstart that records, replays, substitutes, and bisects in 12 lines
     of code; move all conceptual material below the fold.

## § Configuration & ergonomics

160. [x] Add a single `stepback.toml` file resolved via the standard search
     order (CWD → git root → `~/.config/stepback/`) covering trace dir, key
     material, default shims, redaction rules, price-list pinning, and
     viewer preferences; document precedence and overrides.
     Implemented `stepback/config.py` with `StepbackConfig` (full config
     dataclass), `TraceConfig`, `KeyConfig`, `ShimConfig`, `RedactionConfig`,
     `PriceListConfig`, `ViewerConfig`, `ConfigError`, `DEFAULT_CONFIG`,
     `find_config_file()` (search order: `STEPBACK_CONFIG` env var → CWD →
     git root via `_find_git_root()` → `$XDG_CONFIG_HOME/stepback/` or
     `~/.config/stepback/`), and `load_config()` (returns a fresh
     `StepbackConfig` with relative paths resolved against the config file's
     directory). Python 3.10 compatibility via the same tomllib/tomli/minimal
     fallback pattern already used in `doctor.py`. Supports both the legacy
     flat `[stepback]` form (generated by `stepback init`) and the new nested-
     section form (`[stepback.trace]`, `[stepback.keys]`, etc.); nested
     sections take precedence when both are present. Updated `stepback/doctor.py`
     to delegate to `load_config()` instead of its private `_find_stepback_toml`
     / `_load_toml_config` helpers. Updated `stepback/cli.py`
     `_STEPBACK_TOML_TEMPLATE` to include all nested sections with comments.
     All 11 new public symbols exported from `stepback/__init__.py` and
     `__all__`; `tests/test_public_api.py::EXPECTED_PUBLIC_API` and
     `stepback/conformance/api_baselines/v0.1.0/public_api.json` updated.
     Documented in `docs/config.md` (resolution order, section reference,
     relative-path semantics, secrets warning, Python API). Added
     `tests/test_config.py` (62 tests) covering: all default values,
     explicit-path loading, flat/nested/combined TOML forms, all six sections,
     relative and absolute path resolution, `ConfigError` on bad TOML /
     unknown version / invalid values, search-order priority (env var beats
     CWD, CWD beats git root, git root beats user home, `_find_git_root`
     handles directory and worktree-file markers), and all public API exports
     with docstrings. All 62 new tests + 26 existing `test_doctor.py` +
     5 `test_public_api.py` + 23 `test_api_compat.py` pass.

161. [x] Add `STEPBACK_*` environment variables for every config key, with a
     `stepback config` subcommand that prints the effective merged config and
     the source of each key (file, env, default).
     Implemented in `stepback/config.py`: added `ENV_VAR_MAP` (13 entries
     mapping every configurable key to its `STEPBACK_*` env var),
     `SECRET_KEYS` (frozenset; `keys.hmac_key_hex`), `_SourceTracker` class
     (resolves each value from env > file > default while recording the
     source string), `_parse_env_bool` / `_parse_env_int` helpers (strict
     validation; `ConfigError` on bad input), and `_SourceTracker.resolve_json_list`
     for `STEPBACK_REDACTION_RULES_JSON`. Modified `_build_config` to use
     `_SourceTracker` for every key and populate `StepbackConfig.sources`
     (a `Dict[str,str]` field with `compare=False, repr=False`). Modified
     `load_config` to call `_build_config({}, None)` when no file is found
     so env var overrides still apply. Exported `ENV_VAR_MAP` and
     `SECRET_KEYS` from `stepback/__init__.py` and `__all__`. Added
     `_cmd_config` to `stepback/cli.py` with text table and `--json` /
     `--show-secrets` flags. Added 86 new tests in
     `tests/test_config_env_vars.py`. Updated `tests/test_public_api.py`
     and `stepback/conformance/api_baselines/v0.1.0/public_api.json`.
     All 148 config tests pass; no new failures in full suite.

162. [x] Replace ad-hoc `print`/`raise` paths with a consistent error-code
     taxonomy (`SB001`–`SBxxx`), each with a one-line message, a paragraph in
     `docs/errors/`, and a "did you mean…" hint; CI fails on undocumented
     codes.
     Implemented `stepback/errors.py`: `ErrorInfo` NamedTuple, `ERROR_REGISTRY`
     (25 codes across 9 ranges SB0xx–SB8xx), and `format_error(code, detail,
     include_hint)` helper. Added `code` class attributes to 18 exception
     classes (`TraceVerificationError`, `AttestationVerificationError`,
     `ConfigError`, `BranchTraceMismatch`, `MissingExecutor`, `BudgetExhausted`,
     `PredicateNotTriggered`, `ImportError`, `ExportError`, `PQUnavailableError`,
     `HardwareKeyUnavailableError`, `HardwareKeySignError`,
     `ThresholdSignatureError`, `TransparencyLogError`, `PredicateSyntaxError`,
     `PredicateRuntimeError`, `ProvenanceVerificationError`, `OtelBridgeError`).
     Created `docs/errors/index.md` with one `### SBxxx` heading per code plus
     a class→code mapping table. Added `tests/test_error_codes.py` (18 tests)
     enforcing: registry well-formedness, registry↔docs bidirectional coverage,
     all class `code` attrs are registered, `format_error` output shape. All 18
     new tests pass; no regressions in 199 additional test_public_api /
     test_attestation / test_importers / test_minimize / test_predicates tests.

163. [x] Add structured progress output (rich/tqdm) for long operations
     (recording, replay, bisect, minimize) with a `--quiet` and `--json`
     mode; never write progress to a non-TTY by default.

164. [x] Add shell completion for bash, zsh, fish, and PowerShell via
     `stepback completion <shell>`; document the one-line install for each.

165. [x] Add `stepback open <trace>` that launches the HTML viewer on a free
     local port, opens the user's browser, and shuts down on Ctrl-C; works
     on macOS, Linux, WSL, and remote SSH (with `--no-browser`).

166. [in-progress: 2026-05-13T17:34:00Z] Add `stepback diff <a.sb> <b.sb>` producing a unified, color-aware
     step-by-step diff (kind, inputs, outputs, dirty reasons, cost) with a
     `--format json|markdown|html` switch suitable for PR comments.

## § Editor & notebook integrations

167. [ ] Build a VS Code extension (`stepback-vscode`) that recognises `.sb`
     files, renders an inline trace explorer, jumps from a step to its
     source line, and exposes "Replay from here" and "Bisect to here"
     commands.

168. [ ] Build a JetBrains plugin with the same surface as the VS Code
     extension; share a Language Server (`stepback-lsp`) so both editors
     consume the same semantic model.

169. [ ] Ship `%stepback` Jupyter/IPython magics: `%%record`, `%replay`,
     `%bisect`, `%minimize`, plus a rich-display hook so `Trace` objects
     render as collapsible step tables in notebook output.

170. [ ] Add a Marimo / Streamlit reference dashboard under
     `examples/dashboards/` that loads a directory of traces and surfaces
     cost, latency, dirty-set size, and incident replays — runnable with one
     command.

171. [ ] Add a Chrome DevTools-style protocol bridge so any client speaking
     CDP can drive `stepback replay` step-by-step; ship a minimal reference
     client in TypeScript.

## § Web viewer & collaboration

172. [ ] Upgrade the HTML viewer to a single-file PWA: offline-capable,
     installable, deep-linkable per step (`#step=42`), with keyboard
     navigation (`j`/`k`, `/` for search, `?` for help) and a command
     palette.

173. [ ] Add trace search across a directory: full-text over inputs/outputs,
     filter by kind/cost/dirty reason/incident id, saved queries, and
     shareable query URLs; index lives in a sidecar `.sbidx` file.

174. [ ] Add inline annotations on steps (Markdown notes, tags, severity)
     stored in a sibling `.sbnotes` file; annotations are signed and
     never mutate the underlying `.sb`.

175. [ ] Add a "share trace" flow: redact according to the configured policy,
     bundle `.sb` + `.pack` + viewer into a single self-contained HTML, and
     copy a `file://` or pre-signed URL to the clipboard.

176. [ ] Add a hosted reference viewer at `view.stepback.dev` that accepts a
     drag-dropped `.sb` and verifies signatures entirely client-side (WASM);
     no trace bytes leave the browser.

177. [ ] Add a team workspace mode (`stepback workspace`) that syncs traces,
     annotations, and saved queries via any S3-compatible backend, with
     end-to-end encryption keys held only by the workspace members.

## § CI/CD & developer workflow integrations

178. [ ] Publish a `stepback/record-action` GitHub Action that wraps a test
     job, uploads `.sb` artifacts, and posts a PR comment with cost/latency
     deltas vs. the base branch.

179. [ ] Publish a `stepback/replay-action` GitHub Action that, on a PR,
     replays the base branch's trace pack against the PR's code and fails
     when behaviour, cost, or dirty-set changes exceed configured budgets.

180. [ ] Add a `pre-commit` hook (and a Husky equivalent) that runs
     `stepback verify --strict` on every staged `.sb` and `stepback diff` on
     modified ones, blocking commits that break canonical bytes.

181. [ ] Add a GitLab CI template, a CircleCI orb, and a Buildkite plugin
     mirroring the GitHub Action surface; document the integration matrix.

182. [ ] Add `stepback bench compare <baseline.sb> <candidate.sb>` for
     regression gating in CI: emits exit code, JUnit XML, and a Markdown
     summary with the regressed steps inlined.

## § Observability & alerting integrations

183. [ ] Add a Slack/Discord/MS-Teams notifier that posts an incident card
     with a permalink to the offending step in the viewer, the dirty-set
     summary, and a "Bisect" button (webhook-driven).

184. [ ] Add a PagerDuty/Opsgenie integration that opens an incident with the
     trace pack attached and resolves it when a follow-up trace passes the
     same predicate.

185. [ ] Add a Sentry-style breadcrumb exporter so existing error-tracking
     dashboards see `stepback` step transitions alongside stack traces.

186. [ ] Add a Grafana data source plugin that queries the ClickHouse schema
     from Step 139 and ships pre-built dashboards for cost, latency, dirty-
     set churn, and policy denials.

187. [ ] Add Prometheus metrics + an OpenMetrics endpoint on `stepback-proxy`
     covering per-shim QPS, p50/p95 latency, sign/verify failures, cache hit
     rate, and recorder backpressure events.

## § Provider, framework, and ecosystem reach

188. [ ] Add first-class shims for Cohere, Mistral, Groq, Together, Fireworks,
     DeepSeek, xAI, and Azure OpenAI; each with a dedicated test file and
     duck-typed against current SDK shapes.

189. [ ] Add framework recorders for LlamaIndex, Haystack, DSPy, CrewAI,
     AutoGen, LangGraph, Pydantic-AI, and Semantic Kernel; share a common
     `FrameworkRecorder` base to avoid drift.

190. [ ] Add importers for Langfuse, Weights & Biases Weave, Arize Phoenix
     (real, not aliased), Honeycomb, and Datadog LLM Observability; document
     lossiness per importer.

191. [ ] Add a generic OpenInference-OTel auto-instrumentor (`stepback
     instrument <command>`) that records any Python program emitting OTel
     spans without code changes.

192. [ ] Publish official client libraries for the trace format in Rust,
     TypeScript, Go, JVM, and .NET (read + verify only at first); each ships
     with the frozen conformance corpus as test data.

## § Trust, safety, and easy-mode security

193. [ ] Make signing on by default with an auto-generated, machine-local key
     stored in the OS keychain (Keychain/Secret Service/Credential Manager);
     surface a clear path to upgrade to KMS/HSM for production.

194. [ ] Add a redaction preset library (`pii-basic`, `pii-strict`,
     `secrets-only`, `hipaa-lite`, `gdpr-lite`) with documented coverage and
     limitations; `stepback redact --preset` applies them in one command.

195. [ ] Add `stepback policy lint <policy.yaml>` that statically checks
     redaction and enforcement policies against a canonical schema and a
     gallery of trace shapes; CI-friendly output.

196. [ ] Add a key-management quickstart that takes a user from
     local-keychain → cloud KMS → HSM with a single command per hop and
     verified sample traces at each level.

## § Plugin system, telemetry, and product hygiene

197. [ ] Add a plugin entry-point (`stepback.plugins`) with a declared
     capability schema, semver pinning, and a `stepback plugins` subcommand
     to list, enable, and inspect installed plugins.

198. [ ] Add opt-in anonymous usage telemetry (off by default; explicit
     prompt on first run) covering CLI command counts and error codes only;
     publish the schema, the aggregation pipeline, and the public dashboard.

199. [ ] Add `stepback feedback` that opens a pre-filled GitHub issue with
     the user's `stepback doctor` output, redacted config, and last error
     code attached; never includes trace bytes.

200. [ ] Stand up `docs.stepback.dev` (MkDocs Material or Docusaurus) with
     versioned docs, a searchable API reference generated from docstrings,
     embedded runnable examples (Pyodide), and a migration guide for every
     SemVer-major release; CI fails when a public symbol lacks a doc page.
