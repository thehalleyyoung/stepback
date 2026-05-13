# Submission Rules — stepback Benchmark Leaderboard

**Rules version:** 1.0  
**Effective:** 2026-05-12

This document describes the requirements every entrant must satisfy to have
a result appear on the *stepback* benchmark leaderboard.  The rules follow
the spirit of MLPerf's submission process: reproducibility is first-class,
cryptographic traceability is required, and audit rights are a prerequisite
for public listing.

---

## 1. Overview

A **submission** is a signed, self-contained JSON document called a
*submission manifest*.  The manifest bundles:

| Section | Purpose |
|---|---|
| `code` | Frozen code provenance (commit SHA or wheel hash) |
| `trace_pack` | Digest of the corpus trace pack + verification status |
| `hardware` | Hardware manifest for the machine that ran the benchmark |
| `exact_commands` | Verbatim CLI commands to reproduce the run |
| `validator_output` | Captured output of `stepback spec test` |
| `bench_results` | One or more `BenchRunRecord` dicts (schema v1.0) |
| `audit` | Submitter identity and audit-rights grants |

Validate a manifest offline with:

```
stepback bench validate-submission path/to/manifest.json
```

Exit code `0` = accepted; `1` = rejected; `2` = I/O error.

---

## 2. Rule 1 — Frozen Code

A submission **must** include exact code provenance so reviewers can
reproduce the binary:

- `code.stepback_version` — the output of `stepback --version`; must not
  be `"unknown"`.
- At least **one** of:
  - `code.git_commit` — full 40-character SHA-1 of `HEAD`; or
  - `code.wheel_sha256` — hex SHA-256 of the installed `.whl` file.

### Optional fields

| Field | Meaning |
|---|---|
| `code.git_dirty` | `true` if the working tree had uncommitted changes; submission is flagged as *development* |
| `code.source_uri` | Git remote URL or wheel download URL |

### Capturing provenance automatically

```python
from stepback.bench.submission import CodeProvenance
cp = CodeProvenance.detect()   # auto-detects version, commit, dirty flag
print(cp.to_json())
```

---

## 3. Rule 2 — Signed Trace Pack

Every submission **must** include a descriptor of the corpus trace pack:

- `trace_pack.corpus_id` — a non-empty identifier, e.g.
  `"synthetic-200-random_step"` or `"swe-bench-verified-v1"`.
- `trace_pack.trace_count` — integer ≥ 1.
- `trace_pack.pack_sha256` — 64-character hex SHA-256 of the pack artifact.
  The artifact may be:
  - a `.tar.gz` or `.zip` containing all `.sb` trace files; or
  - a JSON listing `[{"file": "<relpath>", "sha256": "<hex>"}]` sorted by
    path (produced by `TracePackManifest.from_directory()`).
- `trace_pack.verified` — **must be `true`**.  Run `stepback verify` on
  every trace in the pack and confirm all pass before submitting.

### Generating the pack digest from a directory

```python
from stepback.bench.submission import TracePackManifest
tp = TracePackManifest.from_directory(
    directory="traces/",
    corpus_id="my-corpus-v1",
    hmac_key_id="key-2026-a",  # optional; None for unsigned traces
    verified=True,
)
print(tp.to_json())
```

---

## 4. Rule 3 — Hardware Manifest

The manifest **must** identify the hardware used to produce the results:

- `hardware.os` — `platform.system()`, e.g. `"Linux"` or `"Darwin"`.
- `hardware.cpu_count` — logical core count (recommended; may be `null`).

### Optional fields

| Field | Meaning |
|---|---|
| `hardware.cpu_model` | CPU model string |
| `hardware.ram_gb` | Total RAM in gigabytes |
| `hardware.docker_image_digest` | `sha256:…` digest of the container image (if containerised) |
| `hardware.container_runtime` | e.g. `"docker 24.0.5"` |

For container-based runs, set the `DOCKER_IMAGE_DIGEST` and
`CONTAINER_RUNTIME` environment variables before calling
`HardwareManifest.detect()`.

### Auto-detecting hardware

```python
from stepback.bench.submission import HardwareManifest
hw = HardwareManifest.detect()
print(hw.to_json())
```

---

## 5. Rule 4 — Exact Commands

The manifest **must** include a non-empty list of the verbatim CLI commands
used to produce the benchmark results, in execution order:

```json
"exact_commands": [
  "stepback bench replay-caching --n-steps 200 --n-trials 10 --seed 42 --out rc_result.json",
  "stepback bench storage-compression --n-traces 50 --n-steps 30 --out sc_result.json"
]
```

Each entry must be a non-empty string.  The commands must be reproducible
on the claimed hardware with the claimed code version.

---

## 6. Rule 5 — Validator Output

The manifest **must** include the captured stdout/stderr of a successful run
of the SB-Trace conformance suite:

```bash
stepback spec test <impl-argv> 2>&1 | tee validator_output.txt
```

- The captured text **must** contain the literal string `"PASSED"`.
- If the text also contains `"FAILED"`, the submission is accepted but
  flagged with a warning.

Include the complete output, not just the summary line.

---

## 7. Rule 6 — Benchmark Results

The manifest **must** include at least one
`BenchRunRecord` dict (from `BenchRunRecord.to_json()`).  Each record must:

- Have `schema_version == "1.0"`.
- Include `run_id`, `timestamp_utc`, `corpus_id`, and `trace_count`.
- Include `versions` and `hardware` sub-records (warnings are issued if
  absent, but the submission is not rejected).

Accepted benchmark types:

| CLI command | Description |
|---|---|
| `stepback bench replay-caching` | Dirty-set size vs. trace length |
| `stepback bench storage-compression` | Storage byte cost across encodings |
| `stepback bench model-swap` | Model-swap fidelity and statistical difference detection |
| `stepback bench record-overhead` | Per-call recorder overhead |
| `stepback bench soak` | Soak throughput |

---

## 8. Rule 7 — Audit Rights

The submission **must** include an `audit` section signed by the submitter:

- `audit.submitter_name` — non-empty name or pseudonym.
- `audit.submission_date` — ISO 8601 date (`"YYYY-MM-DD"`).
- `audit.grants_source_access` — **must be `true`** for public leaderboard.
  The submitter agrees to provide full source code on reviewer request.
- `audit.grants_trace_pack_access` — **must be `true`** for public
  leaderboard.  The submitter agrees to provide the trace pack on request.

Submissions with either grant set to `false` are accepted as *development*
entries and are shown in a separate section of the leaderboard.

| Field | Required | Notes |
|---|---|---|
| `submitter_name` | Yes | |
| `submission_date` | Yes | `YYYY-MM-DD` |
| `grants_source_access` | Yes (= `true`) | public leaderboard |
| `grants_trace_pack_access` | Yes (= `true`) | public leaderboard |
| `submitter_email` | Recommended | contact address |
| `submitter_organization` | Optional | institutional affiliation |

---

## 9. Creating and Validating a Manifest

### End-to-end example

```python
import uuid
from stepback.bench.submission import (
    AuditDeclaration,
    CodeProvenance,
    HardwareManifest,
    SubmissionManifest,
    TracePackManifest,
    validate_submission,
)
from stepback.bench.result_schema import BenchRunRecord

# 1. Run your benchmark and capture the result
from stepback.bench.replay_caching import run as run_rc
bench_result = run_rc(n_steps=200, n_trials=10, seed=42)

# 2. Convert bench result to BenchRunRecord schema
from stepback.bench.result_schema import BenchRunRecord, VersionInfo, HardwareInfo
record = BenchRunRecord.from_bench_result(bench_result)

# 3. Build manifest
manifest = SubmissionManifest.create(
    trace_pack=TracePackManifest.from_directory(
        directory="my_traces/",
        corpus_id="synthetic-200-random_step",
        verified=True,
    ),
    exact_commands=[
        "stepback bench replay-caching --n-steps 200 --n-trials 10 --seed 42 --out rc.json",
    ],
    validator_output=open("validator_output.txt").read(),
    bench_results=[record.to_json()],
    audit=AuditDeclaration.today(
        submitter_name="Alice",
        submitter_email="alice@example.com",
        submitter_organization="Acme Inc.",
        grants_source_access=True,
        grants_trace_pack_access=True,
    ),
)

# 4. Validate
result = validate_submission(manifest)
if result.valid:
    import json
    with open("submission.json", "w") as f:
        json.dump(manifest.to_json(), f, indent=2)
    print("Submission saved to submission.json")
else:
    for err in result.errors:
        print(f"ERROR [{err.field}] {err.message}")
```

### CLI validation

```bash
stepback bench validate-submission submission.json
stepback bench validate-submission submission.json --out validation_result.json
```

---

## 10. Submission Process

1. **Prepare** — run the benchmark, run `stepback spec test`, collect results.
2. **Build manifest** — use `SubmissionManifest.create()` or the JSON schema.
3. **Validate locally** — `stepback bench validate-submission submission.json`
   must exit `0`.
4. **Submit** — open a pull request against the `leaderboard/` directory of
   the `stepback-results` repository, attaching `submission.json`.
5. **Review** — an automated CI job re-runs `validate-submission`; a human
   reviewer may request source or trace pack access under the audit rights
   you granted.
6. **Publish** — once approved, the result is added to the leaderboard
   and the submission manifest is archived.

---

## 11. Development vs. Public Submissions

| Condition | Classification |
|---|---|
| All rules pass, no warnings | Public leaderboard |
| `code.git_dirty == true` | Development entry (separate section) |
| `grants_source_access == false` OR `grants_trace_pack_access == false` | Development entry |
| Any `ValidationError` | Rejected |

---

## 12. Audit Procedure

Any reviewer may invoke their audit rights by contacting the submitter at
the address in `audit.submitter_email`.  The submitter must respond within
30 days with:

- Full source code of any non-public components used to produce the results.
- The trace pack identified by `trace_pack.pack_sha256`.

Failure to respond results in the submission being removed from the
leaderboard.

---

## 13. Python API Reference

```
stepback.bench.submission
├── RULES_VERSION                  # "1.0"
├── HardwareManifest               # hardware info + container fields
│   ├── .detect()                  # auto-detect from running host
│   ├── .to_json() / .from_json()
├── CodeProvenance                 # frozen code version
│   ├── .detect()
│   ├── .to_json() / .from_json()
├── TracePackManifest              # corpus trace pack descriptor
│   ├── .from_directory(dir, ...)  # scan .sb files, compute digest
│   ├── .to_json() / .from_json()
├── AuditDeclaration               # submitter identity + grants
│   ├── .today(name, ...)          # convenience with today's date
│   ├── .to_json() / .from_json()
├── SubmissionManifest             # top-level manifest
│   ├── .create(...)               # auto-detect hw + code provenance
│   ├── .to_json() / .from_json()
├── ValidationError                # single rule violation
├── SubmissionValidationResult     # outcome of validate_submission
│   ├── .valid                     # True iff no errors
│   ├── .errors                    # list[ValidationError]
│   ├── .warnings                  # list[str]
│   ├── .summary_line()
│   ├── .to_json() / .from_json()
├── validate_submission(manifest)  # validate SubmissionManifest
└── validate_submission_json(dict) # validate a plain dict
```
