# Reader resource limits and DoS bounds

Step 50 of the SB-Trace standardization roadmap requires every
conformant reader to enforce explicit, documented bounds on the size
and shape of decoded frames. Without those bounds an adversarial trace
— a corrupted length prefix, a deeply-nested JSON body, or a single
multi-gigabyte string — can trick a naive reader into

* allocating gigabytes of memory before realising the input is
  malformed (the canonical denial-of-service vector for
  length-prefixed binary formats);
* recursing into the host runtime's call stack until it overflows; or
* spending unbounded CPU on a JSON value the writer never could have
  legitimately emitted.

This document is the *single source of truth* for the limits every
implementation pins. The numbers here are the contract; any reader
that decodes `.sb` v1 frames MUST reject inputs that violate them and
MAY refuse to read them at all.

## The three limits

### 1. `MAX_FRAME_BYTES` — on-wire frame size

* **Value:** `64 * 1024 * 1024` (64 MiB).
* **What it caps:** the value of the 4-byte big-endian length prefix
  that precedes every frame on disk. The 4-byte prefix itself is *not*
  counted against the limit.
* **Where it's enforced:** before the reader allocates a buffer for or
  reads any of the body bytes. A length prefix exceeding the cap MUST
  be rejected without further I/O.
* **Why this number:** large enough for the largest realistic LLM /
  tool payload after gzip+base64 expansion (we routinely see
  multi-megabyte tool outputs in production traces), but small enough
  that a corrupted `\xff\xff\xff\xff` prefix cannot trigger a 4 GiB
  allocation. Recorders that expect to emit larger payloads should
  either use blob frames (content-addressed deduplication) or set the
  capability `large-frame-bytes` in the header (capability negotiation
  is fail-closed in v1; see `docs/canonicalization.md`).

### 2. `MAX_NESTING_DEPTH` — JSON nesting depth

* **Value:** `256`.
* **What it caps:** the maximum depth any JSON value inside a frame
  body or wrapper may reach. The wrapper itself contributes 1; an
  object inside the body's `step` field at the typical recorder depth
  bottoms out around 6–8.
* **Where it's enforced:** as the reader walks the parsed value, or
  during the parse itself if the host JSON parser exposes a depth
  hook. Implementations MUST track depth iteratively to avoid making
  the host language's stack-overflow behaviour the effective cap.
* **Why this number:** chosen so that a malicious `{"x":{"x":{"x":...`
  ladder cannot blow the stack of any conformant reader (Python's
  default `sys.setrecursionlimit` is 1000; Rust release builds run on
  ~8 MiB stacks; Go ships on growable stacks but pays per frame), and
  still leaves >30× headroom over the deepest legitimately recorded
  step body we have observed.

### 3. `MAX_STRING_BYTES` — single inline string size

* **Value:** `16 * 1024 * 1024` (16 MiB).
* **What it caps:** the UTF-8 byte length of any single JSON string
  *value* or *object key* inside a frame body. It is independent of,
  and stricter than, `MAX_FRAME_BYTES`: a 64 MiB frame may legally
  contain four 16 MiB strings, but not one 17 MiB string.
* **Where it's enforced:** during the same depth-bounded walk that
  enforces `MAX_NESTING_DEPTH`. Some implementations short-circuit
  during parse for a constant-factor speedup; both are conformant.
* **Why this number:** any payload genuinely above 16 MiB belongs in a
  blob frame, where it is content-addressed, deduplicated across
  steps, and gzip-compressed by the writer. Capping inline strings
  prevents pathological traces from blowing past per-frame budgeting
  expectations even when the wrapper itself is below `MAX_FRAME_BYTES`.

## Conformance checklist

A reader is `step-50-conformant` if and only if all of the following
hold:

1. It rejects a frame whose length prefix exceeds `MAX_FRAME_BYTES`
   *before* allocating or reading the body bytes.
2. It rejects a frame whose body or wrapper contains a JSON value
   nested deeper than `MAX_NESTING_DEPTH`.
3. It rejects a frame whose body or wrapper contains a JSON string
   value or object key whose UTF-8 encoding exceeds
   `MAX_STRING_BYTES`.
4. The error path is the same typed error class the reader uses for
   other malformed-frame rejections (`TraceVerificationError` in
   Python, `FrameError`/`VerifyError` in Rust, `FrameError` in
   TypeScript / Go / JVM / .NET, `VerifyError` in WASM).
5. Each cap is exposed as a public constant under a stable name so
   downstream tooling can introspect what its reader will accept:

| Implementation | Frame size | Nesting depth | String size |
| --- | --- | --- | --- |
| Python (`stepback.trace_reader`) | `MAX_FRAME_BYTES` | `MAX_NESTING_DEPTH` | `MAX_STRING_BYTES` |
| Rust (`sb_format`) | `MAX_FRAME_BYTES` | `MAX_NESTING_DEPTH` | `MAX_STRING_BYTES` |
| TypeScript (`@stepback/sb`) | `MAX_FRAME_BYTES` | `MAX_NESTING_DEPTH` | `MAX_STRING_BYTES` |
| Go (`bindings/go`) | `MaxFrameBytes` | `MaxNestingDepth` | `MaxStringBytes` |
| JVM (`dev.stepback.sb`) | `Constants.MAX_FRAME_BYTES` | `Constants.MAX_NESTING_DEPTH` | `Constants.MAX_STRING_BYTES` |
| .NET (`Stepback.Sb`) | `Constants.MaxFrameBytes` | `Constants.MaxNestingDepth` | `Constants.MaxStringBytes` |
| WASM (`sb-wasm`) | inherited from `sb_format` | inherited from `sb_format` | inherited from `sb_format` |

## Tunability

The defaults are the contract. Implementations MAY accept *larger*
limits via a builder API or per-call keyword for tooling that
consciously processes oversized payloads (recovery scripts, archival
inspection of legacy traces). Implementations MUST NOT accept
*smaller* limits silently — a stricter cap is fine for testing but
should be visible at the call site.

The Python reader exposes the three caps as keyword arguments on
`stepback.trace_reader.read_frames`:

```python
from stepback.trace_reader import read_frames

# Default — strictest, the contract.
read_frames("trace.sb")

# Recovery use only: opt in to a larger frame budget. Still rejects
# anything above the new caps.
read_frames("legacy.sb", max_frame_bytes=256 * 1024 * 1024)
```

Equivalent escape hatches in the other bindings:

* **Rust**: `iter_frames_with_limits` (see `sb_format::FrameLimits`).
* **TypeScript**: `iterFrames(buf, { maxFrameBytes, maxDepth, maxStringBytes })`.
* **Go**: `IterFramesWithLimits(buf, FrameLimits{...})`.
* **JVM / .NET**: corresponding builder constructor on the reader.

## Threat model alignment

The limits above complement, rather than replace, the
cryptographic protections in `SECURITY.md` (HMAC chain + per-frame
Ed25519 signature). HMAC and signatures bind authentic content, but
the verifier still has to *parse* a hostile-but-signed-with-a-leaked-key
trace before it can decide whether to trust it; the limits above
ensure that adversarial-but-syntactically-valid frames cost a bounded
amount of CPU and memory to reject.

In particular, an attacker who has compromised a recorder's HMAC and
signing key still cannot DoS the verifier by producing
syntactically-monstrous frames: every conformant reader will refuse
the bytes long before exhaustive crypto verification kicks in.
