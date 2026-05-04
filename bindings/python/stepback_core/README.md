# stepback-core (Python binding)

**Experimental.** PyO3 bindings that route stepback's `verify_trace` through
the Rust [`sb-verify`](../../../stepback-core/crates/sb-verify) crate. This
is the Step 7 deliverable from `100_STEPS.md`: a working PyO3 surface that
the Python `stepback` package can opt into behind an explicit flag.

## Status

* Wraps `sb_verify::verify_bytes` only.
* Read-only verification — no writer, no replay, no dirty-set yet.
* Wheel name is `stepback-core`; the importable Python package is
  `stepback_core`.

## Build

```bash
cd bindings/python/stepback_core
maturin develop --release        # installs into the active venv
```

Or build a wheel without installing:

```bash
maturin build --release -o ../../../dist
```

## Use directly

```python
from stepback_core import verify_bytes, VerifyError

with open("trace.sb", "rb") as f:
    info = verify_bytes(f.read(), bytes.fromhex("0102..."))

print(info.frame_count, info.format_version)
```

## Use through the Python `stepback` package

```python
from stepback.trace_reader import verify_trace

# The default engine is "python" — pure-Python verifier, always available.
trace = verify_trace("trace.sb", hmac_key)

# Opt into the Rust verifier explicitly. Raises ImportError /
# RuntimeError if the `stepback_core` extension is not installed.
trace = verify_trace("trace.sb", hmac_key, engine="rust")

# Or set the env var once and let "auto" pick rust when available.
#   export STEPBACK_VERIFY_ENGINE=rust
```

## Stability

`stepback-core` predates the SB-Trace v1 RFC. The Python surface tracks
the v1 wire format; signatures may change before 1.0. Pin to an exact
version until the RFC freezes.
