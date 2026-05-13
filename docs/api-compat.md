# API + SB-Trace schema compatibility checking

Step 25 of [`100_STEPS.md`](../100_STEPS.md) calls for a CI gate that
diffs the **public Python API** of `stepback` and the **on-disk
SB-Trace schema** against the last release tag, so that silent
breaking changes never reach a tagged release. This document is the
operator manual for that gate.

## Components

| Path | Role |
| --- | --- |
| `stepback/api_snapshot.py` | Pure Python module that produces JSON-shaped snapshots of `stepback.__all__` and the current `SBTraceSpec`, plus diff functions that classify changes as **breaking** or **compatible**. |
| `scripts/check_api_compat.py` | CLI that resolves a baseline (committed file, explicit dir, or git ref), runs the diff, and exits non-zero on any breaking change. |
| `stepback/conformance/api_baselines/v<version>/{public_api,sbtrace_schema}.json` | Committed baselines, shipped inside the wheel via `pyproject.toml` `package-data`. The baseline equals the snapshot of the most recent tagged release of that `<version>`. |
| `tests/test_api_compat.py` | Unit + CLI tests; **the in-tree baseline must equal the live snapshot** or the suite fails locally too. |
| `.github/workflows/ci.yml` (`api-compat` job) | Runs `python scripts/check_api_compat.py --auto-baseline-ref` against the latest `v*` tag. |

## How CI uses it

```yaml
- name: Check public API + SB-Trace schema vs last release tag
  run: python scripts/check_api_compat.py --auto-baseline-ref
```

The job checks out with `fetch-depth: 0` so all tags are available.
With no tag yet, the CLI falls back to the committed baseline directory
under `stepback/conformance/api_baselines/v<__version__>/`. That file
is the single source of truth for "what shipped in v0.1.0".

## When you intentionally change the API

1. Decide whether the change is **compatible** (new symbol, new
   optional parameter, new optional field) or **breaking** (removed
   symbol, removed/required parameter change, schema-level removal,
   wire-major bump).
2. For breaking changes, bump the relevant SemVer track:
   * Python package — `stepback.__version__` (see
     [`docs/deprecation.md`](deprecation.md)).
   * SB-Trace wire — `SBTRACE_WIRE_*` constants in `stepback/spec.py`
     (see [`docs/trace-mutation.md`](trace-mutation.md) for trace-level
     compatibility considerations).
3. Refresh the committed baseline:
   ```bash
   python scripts/check_api_compat.py --write-baseline
   ```
   This writes `public_api.json` and `sbtrace_schema.json` into
   `stepback/conformance/api_baselines/v<new-version>/`.
4. Commit the baseline together with the change.

## What gets diffed

### Public API

For every name in `stepback.__all__` the snapshot records:

* kind (`function` / `class` / `module` / `value`),
* qualified name,
* function signatures (parameter names, kinds, annotations, defaults,
  return annotations),
* class base classes, dataclass fields, enum members, public methods,
* whether the symbol is marked `@deprecated`.

Removing a symbol that is **not** flagged as deprecated, removing a
parameter default, adding a required parameter, dropping a base class,
adding a required dataclass field, removing a public method, removing
or renaming an enum member, or changing the type of a value constant
all surface as **breaking**. Adding a new symbol, a new optional
parameter, a new optional dataclass field, or a new method is
**compatible**.

### SB-Trace schema

Captured from `stepback.spec.current_spec()`:

* `wire_version` / `wire_version_info`, `format_version`, `encoding`,
  `magic`,
* required and optional field sets for the wrapper, header, and step
  frames,
* recognized step kinds, frame kinds, and supported capabilities,
* the strictness flags.

A wire-major bump, magic / encoding / `format_version` change, removal
of a required field, removal of a known step or frame kind, and
removal of a supported capability are **breaking**. Adding optional
fields, capabilities, or new step/frame kinds is **compatible** within
the same wire major.

## Local verification

```bash
# diff against the committed baseline (what CI does without tags)
python scripts/check_api_compat.py

# diff against an explicit tag or branch
python scripts/check_api_compat.py --baseline-ref v0.1.0

# emit a JSON report (no human-readable text)
python scripts/check_api_compat.py --json
```

A non-zero exit status with `breaking` entries means refresh the
baseline only after the SemVer bump is in the same commit.
