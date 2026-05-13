# Coverage policy

Step 26 of `100_STEPS.md` requires that the load-bearing modules of stepback
carry an enforced coverage floor so that a regression on any of them fails
CI rather than silently shipping. The floors are declared in
`pyproject.toml` under `[tool.coverage.floors]` and checked by
`scripts/check_coverage_floors.py`.

## Modules under floor

| Module | Why it's load-bearing |
| --- | --- |
| `stepback/canonical.py` | Canonical-JSON encoding underwrites every content hash; an off-by-one here corrupts the dirty-set algorithm and invalidates signed traces. |
| `stepback/trace_writer.py` | Writes the on-disk `.sb` framing with HMAC chaining and Ed25519 signatures. A bug here breaks downstream verification across every binding (Rust, Go, JVM, .NET, WASM). |
| `stepback/trace_reader.py` | Counterpart of the writer; rejects malformed and adversarial inputs. Coverage drops here usually mean a fuzz/corruption test was deleted. |
| `stepback/attestation.py` | Builds the signed `.pack` artifact that downstream auditors rely on. |
| `stepback/divergence.py` | Implements dirty-set propagation — the headline algorithmic claim of the project. |
| `stepback/replay.py` | Re-executes traces under substitutions; used in every counterfactual test and benchmark. |

## Running locally

```bash
pip install -e '.[dev]'
coverage run -m pytest
coverage json -o coverage.json
python scripts/check_coverage_floors.py
```

The check script exits with:

| Exit code | Meaning |
| --- | --- |
| `0` | Every floored module met its floor. The CLI prints the per-module margin so you know how much headroom you have. |
| `1` | At least one floored module is below its threshold. The script prints a `file / actual / floor` table to stderr. |
| `2` | Configuration error: `pyproject.toml` is missing, `coverage.json` is missing, a floor is out of range, or a floored file is absent from the coverage report. |

## Updating floors

Floors are intentionally set a few percentage points below the most recently
measured branch-coverage rate so incidental refactors don't break CI. Two
rules:

1. **Bumping a floor up is encouraged** when measured coverage permanently
   improves; do this in the same PR that adds the new tests.
2. **Lowering a floor must be flagged in the PR description**, with a link
   to the test(s) being deliberately removed and the reviewer it was
   discussed with. Silent floor relaxation defeats the purpose of Step 26.

## CI integration

The `coverage-floors` job in `.github/workflows/ci.yml` runs on every push
and pull request to `main`. It installs the dev extra, runs `pytest` under
`coverage`, emits `coverage.json`, and invokes
`scripts/check_coverage_floors.py` as a hard gate. The raw coverage data is
uploaded as a workflow artifact for 90 days so trends can be inspected
without re-running the suite.
