# Comet-Σ L1 emitters for stepback

This document is the canonical schema reference for stepback's
[Comet-Σ](https://github.com/halleyyoung/kitchensink) **L1**
(`temporal_basis` / `comet_delta` / `move_synth`) feature emitters.
Each section pins the base-feature names, types, and provenance shape
emitted by one stepback surface.

The emitters are **opt-in**: they are gated behind the per-sublayer
flags from `comet_sigma.flags` and contribute zero overhead on the
hot path when the flag is off.

| Surface                         | Flag                            | Layer | Module label                       | Receipt schema id          |
|---------------------------------|---------------------------------|-------|------------------------------------|----------------------------|
| `stepback/trace_writer.py` (.sb v1) | `COMET_SIGMA_L1_TEMPORAL`   | L1    | `stepback.trace_writer.sb_v1`      | `trace_writer_l1_v1`       |

> Steps 11–90 of `COMET_SIGMA_1000.md` will extend this table with
> rows for `canonical.py`, `divergence.py`, `replay.py`,
> `minimize.py`, `sweep.py`, `trace_diff.py`, `attestation.py`, and
> the `stepback-core` Rust crate.

## `trace_writer.py` + `.sb` format v1 — `trace_writer_l1_v1`

**Implementation.** [`stepback/comet_sigma/l1_trace_writer.py`](../../stepback/comet_sigma/l1_trace_writer.py).

**Wiring.** `TraceWriter._write_frame` calls `observe_frame(self.path, body, body_bytes)` after computing `body_bytes` but before writing the wrapper. The hook is read-only with respect to the on-disk frame.

**Activation.**

```bash
COMET_SIGMA_L1_TEMPORAL=1 python -m stepback.cli ...
```

When the flag is off (the upstream default in `comet_sigma/flags.py`)
the emitter is a strict no-op: a single boolean check on the hot
path, no allocations, no provenance entries, no receipts.

**Base-feature schema.** Every emitted `Receipt` shares the same
payload shape:

```json
{
  "value": 1234.0,        // float — the per-frame feature value
  "frame_index": 42,       // int   — zero-based position in the HMAC chain
  "frame_type": "step"     // str   — "header" | "step" | "blob" | "merkle_summary" | "tail" | "capability" | ...
}
```

The seven base features declared by `BASE_FEATURES` are:

| Name                                    | Domain                | Definition                                                                        |
|-----------------------------------------|-----------------------|-----------------------------------------------------------------------------------|
| `trace_writer_sb_v1.frame_bytes`        | `[0, +∞)` (float)     | `len(canonical_json(body))` for the frame about to be wrapped.                    |
| `trace_writer_sb_v1.prev_dt_ns`         | `[0, +∞)` (float ns)  | Wall-clock nanoseconds since the previous observed frame on this writer (0 first). |
| `trace_writer_sb_v1.frame_depth`        | `{0, 1, 2, …}` (float)| Zero-based index of this frame in the writer's HMAC chain.                        |
| `trace_writer_sb_v1.is_blob_frame`      | `{0.0, 1.0}`          | `1.0` iff `body["type"] == "blob"`.                                              |
| `trace_writer_sb_v1.is_step_frame`      | `{0.0, 1.0}`          | `1.0` iff `body["type"] == "step"`.                                              |
| `trace_writer_sb_v1.is_header_frame`    | `{0.0, 1.0}`          | `1.0` iff `body["type"] == "header"`.                                            |
| `trace_writer_sb_v1.body_key_count`     | `{0, 1, 2, …}` (float)| Number of top-level keys in the frame body dict.                                  |

Each name is also a registered `AuditableArtifact` with
`kind="feature"`, `layer="L1"`, `module="stepback.trace_writer.sb_v1"`,
and a single-entry `ProvenanceChain`. The `src_sha256` field of the
provenance head pins the literal Python source of the extractor that
produced the value.

**Per-writer state.** A `WriterState` is created lazily on the first
observed frame for each `path`. It owns:

- a tuple of seven artifacts (one per base feature),
- a FIFO ring of `FrameRecord`s capped at `MAX_RECENT_FRAMES = 10_000`,
- the cumulative list of `Receipt`s emitted on this writer,
- the `last_wallclock_ns` and `frame_index` cursors.

State is process-local. Tests can call
`stepback.comet_sigma.l1_trace_writer.reset()` to drop it.

**Failure mode.** `observe_frame` is wrapped in a top-level
`try/except` that swallows every exception: a malformed frame, a
schema mismatch, or a missing upstream `comet_sigma` package must
never propagate into the writer's hot path. When `comet_sigma` is
not importable, `is_active()` returns `False` and the emitter is
inert.

## `trace_writer.py` + `.sb` format v1 — temporal-basis projection (`trace_writer_l1_projection_v1`)

**Implementation.** [`stepback/comet_sigma/l1_trace_writer_temporal.py`](../../stepback/comet_sigma/l1_trace_writer_temporal.py).

**Wiring.** Importing the module installs an idempotent hook in
`stepback.comet_sigma.l1_trace_writer.OBSERVE_HOOKS`. Every observed
frame triggers a single `project(writer_id, now_ns=record.wallclock_ns)`
call so the latest projection is always one observation behind the
newest frame. The projector reads its inputs from the Step 1 ring
buffer; it never re-reads the on-disk `.sb` file.

**Activation.** Shares the `COMET_SIGMA_L1_TEMPORAL` flag with the
Step 1 emitter — there is no separate kill-switch.

**Windows.** Four wall-clock windows, named exactly as in
`COMET_SIGMA_1000.md` step 2:

| Name  | Width (ns)        |
|-------|-------------------|
| `1s`  | 1 000 000 000     |
| `10s` | 10 000 000 000    |
| `1m`  | 60 000 000 000    |
| `10m` | 600 000 000 000   |

`(now − window_ns, now]` is the half-open inclusion rule: a frame
exactly `window_ns` old is **excluded**.

**Aggregates.** Seven scalar functions over a `Sequence[float]`,
mirroring the shipped library in `comet_sigma.l1.temporal_basis`:
`mean`, `slope`, `ewma` (α = 0.5), `range`, `std`,
`last_minus_first`, `last_minus_mean_prev`.

**Projection schema.** For every base feature `F`, every window `W`,
and every aggregate `A` the projector exposes one feature
`F.W.A` (e.g. `trace_writer_sb_v1.frame_bytes.10s.mean`). The full
canonical list is `l1_trace_writer_temporal.projection_names()` — 7
features × 4 windows × 7 aggregates = **196** projections per writer.

**Receipt schema id:** `trace_writer_l1_projection_v1`. Payload
fields:

| Field                  | Type    |
|------------------------|---------|
| `value`                | float   |
| `window`               | str     |
| `window_ns`            | int     |
| `n_frames_in_window`   | int     |
| `frame_index`          | int     |

**Provenance.** Each of the 196 artifacts has a single
`ProvenanceEntry` with `layer="L1"`,
`module="stepback.trace_writer.sb_v1.temporal_projection"` and a
`src_sha256` pinning the literal aggregate body (so all 28 artifacts
sharing aggregate `mean` share the same `src_sha256`, regardless of
which feature/window they cover).

**Failure mode.** Identical to Step 1's emitter: the per-frame hook
catches every exception so a misconfigured projection can never
propagate into the writer.

## `trace_writer.py` + `.sb` format v1 — L1 emission persistence (Step 3)

**Module:** `stepback.comet_sigma.l1_trace_writer_store`.

Step 3 persists the in-memory artifacts and receipts emitted by Steps
1 and 2 to a content-addressed on-disk **store** so an external
auditor can reconstruct the artifact tree, replay every receipt and
verify each artifact's `ProvenanceChain` against the sha256 of the
literal extractor source.

**Activation.** Persistence shares the `COMET_SIGMA_L1_TEMPORAL`
flag with Steps 1 and 2 (Step 3 is a strict refinement of Step 1, not
an independent surface). It additionally requires a store directory,
configured via either the `STEPBACK_COMET_SIGMA_L1_STORE`
environment variable or the programmatic
`l1_trace_writer_store.set_store_dir()` setter. With the flag off, or
no store directory configured, persistence is a strict no-op and
makes zero filesystem writes.

**On-disk layout** (rooted at `<store_dir>`):

```
<store_dir>/writers/<writer_id_hash>/
    writer_meta.json
    artifacts/<feature_name_hash>.json   # one per AuditableArtifact
    receipts/base.jsonl                  # Step 1 receipts (one per line)
    receipts/projection.jsonl            # Step 2 receipts (one per line)
```

`<writer_id_hash>` and `<feature_name_hash>` are the first 12 hex
characters of the sha256 of the writer path / artifact name. Each
artifact JSON is the canonical `AuditableArtifact.to_dict()` form,
ready for `comet_sigma.audit.from_dict`.

**Provenance.** The first time an artifact is persisted the store
appends one new `ProvenanceEntry` to its chain via
`comet_sigma.audit.extend_provenance`, recording `layer="L1"`,
`module="stepback.trace_writer.sb_v1.persistence"` and a payload
sha256 over `{writer_id_hash, artifact_name}`. The auditor can
therefore tell, by looking at the persisted artifact alone, both
*who* declared the feature (Step 1's creation entry) and *that* the
artifact was committed to the store (Step 3's persistence entry).

**Receipts JSONL.** Each line of `base.jsonl` /
`projection.jsonl` is the canonical `Receipt.to_dict()` form
including the receipt's `payload_sha256`. The watermark in
`l1_trace_writer_store._WATERMARKS` ensures repeated `persist_writer()`
calls only ever append *new* lines, so an external auditor can
`tail -f` the file safely.

**Failure mode.** Every filesystem operation in
`l1_trace_writer_store` is wrapped in `try/except`; a failure to
persist returns `None` from `persist_writer` and *never* propagates
into `TraceWriter._write_frame`. The trace itself is therefore
guaranteed to be byte-identical regardless of whether persistence
succeeds.


## Step 4 — historical-trace backfill

`stepback/comet_sigma/l1_trace_writer_backfill.py` ingests **existing**
`.sb` v1 traces produced by past runs of `stepback.trace_writer` and
replays each frame through the Step 1 emitter so the temporal-basis
store gains a faithful historical record.

**Flag-gating.** Backfill shares `COMET_SIGMA_L1_TEMPORAL` with Steps
1 / 2 / 3; when the flag is OFF the entire surface is a strict no-op
and `backfill_trace()` returns `None`.

**Writer-id namespace.** Each backfilled trace is registered under
`backfill:<abs_path>` so it can never collide with a live writer that
is currently appending to the same file. The on-disk store dirs the
persister produces are therefore distinct from any concurrent
production writer.

**Per-trace provenance.** Before any frame is replayed, every
`AuditableArtifact` in the writer's state gains a single new
`ProvenanceEntry` with `module="stepback.trace_writer.sb_v1.backfill"`
whose `payload_sha256` covers `{source_path, frames_backfilled,
fingerprint_sha256_64k}`. The fingerprint is the sha256 of the first
64 KiB of the source file — small enough to hash cheaply, distinctive
enough to disambiguate identically-named traces. Each persisted
artifact on disk therefore carries the chain *creation → backfill →
persistence*, and an auditor can reconstruct exactly which historical
trace produced each emission.

**APIs.** `backfill_trace(path, *, store_dir=None, reset=False,
max_frame_bytes=None)` handles a single file; `backfill_paths(...)`
and `backfill_directory(root, *, pattern="**/*.sb", ...)` walk
multiple files. Each returns one `BackfillResult` (or a list thereof)
with the source path, derived writer id, observed / skipped frame
counts, base+projection receipt counts, the persistence counters and
the file fingerprint.

**Failure mode.** The same defensive contract as Steps 1 / 2 / 3:
read errors, decode errors and per-frame extractor failures are
caught and surfaced as a structured `BackfillResult.error` field;
nothing here ever raises into the caller's hot path.


## Step 7 — canonical schema reference for `trace_writer.py` + `.sb` v1

This section is the **single normative reference** an external Comet-Σ
auditor needs to validate, replay or version-pin every L1 emission
produced by stepback's `trace_writer.py` surface. It supersedes any
schema fragment scattered above when those disagree; the source of
truth is the constants in
[`stepback/comet_sigma/l1_trace_writer.py`](../../stepback/comet_sigma/l1_trace_writer.py)
and
[`stepback/comet_sigma/l1_trace_writer_temporal.py`](../../stepback/comet_sigma/l1_trace_writer_temporal.py).
The doc is regression-tested implicitly by the property and benchmark
suites that pin every constant cited below.

### 7.1 Identity matrix

| Concept                       | Stable identifier                                              | Source-of-truth constant                                  |
|-------------------------------|----------------------------------------------------------------|-----------------------------------------------------------|
| Comet-Σ layer                 | `L1`                                                           | `MODULE_LABEL` / artifact `layer`                         |
| Sublayer flag                 | `COMET_SIGMA_L1_TEMPORAL`                                      | `l1_trace_writer.FLAG_NAME`                               |
| Module label (provenance)     | `stepback.trace_writer.sb_v1`                                  | `l1_trace_writer.MODULE_LABEL`                            |
| Module label (projection)     | `stepback.trace_writer.sb_v1.temporal_projection`              | `l1_trace_writer_temporal.MODULE_LABEL`                   |
| Module label (persistence)    | `stepback.trace_writer.sb_v1.persistence`                      | `l1_trace_writer_store.MODULE_LABEL`                      |
| Module label (backfill)       | `stepback.trace_writer.sb_v1.backfill`                         | `l1_trace_writer_backfill.MODULE_LABEL`                   |
| Base receipt schema id        | `trace_writer_l1_v1`                                           | `l1_trace_writer.RECEIPT_SCHEMA_ID`                       |
| Projection receipt schema id  | `trace_writer_l1_projection_v1`                                | `l1_trace_writer_temporal.RECEIPT_SCHEMA_ID`              |
| Artifact `kind`               | `feature`                                                      | `make_artifact(kind="feature", …)`                        |
| Artifact name prefix          | `trace_writer_sb_v1.`                                          | every entry of `BASE_FEATURES`                            |
| Frame-buffer cap (per writer) | `MAX_RECENT_FRAMES = 10_000`                                   | `l1_trace_writer.MAX_RECENT_FRAMES`                       |

`writer_id` is the absolute filesystem path of the `.sb` file the
writer is appending to. Backfilled writers prepend `backfill:` so the
two namespaces never collide (see Step 4).

### 7.2 Base-feature artifact registry

Seven artifacts are declared. For each, the registry below pins:

* the **canonical name** (also the `AuditableArtifact.name`),
* the **value domain** (closed under the runtime extractor),
* the **`src` body** that is sha256'd into the provenance head, and
* the **frame-type subset** for which the value is mechanically interesting
  (the extractor still runs for every frame; the column simply names
  where the value carries information versus a degenerate constant).

| `name`                                  | Domain                | `src` (Python body)                                                                                  | Informative on frame types          |
|-----------------------------------------|-----------------------|-------------------------------------------------------------------------------------------------------|--------------------------------------|
| `trace_writer_sb_v1.frame_bytes`        | `[0, +∞)` float       | `return float(len(body_bytes))`                                                                       | every frame                          |
| `trace_writer_sb_v1.prev_dt_ns`         | `[0, +∞)` float ns    | `last=ctx.get('last_wallclock_ns'); now=ctx['wallclock_ns']; return float(now-last) if last else 0.0` | every frame after the first          |
| `trace_writer_sb_v1.frame_depth`        | `{0,1,2,…}` float     | `return float(ctx['frame_index'])`                                                                    | every frame                          |
| `trace_writer_sb_v1.is_blob_frame`      | `{0.0, 1.0}`          | `return 1.0 if body.get('type')=='blob' else 0.0`                                                     | `blob`                               |
| `trace_writer_sb_v1.is_step_frame`      | `{0.0, 1.0}`          | `return 1.0 if body.get('type')=='step' else 0.0`                                                     | `step`                               |
| `trace_writer_sb_v1.is_header_frame`    | `{0.0, 1.0}`          | `return 1.0 if body.get('type')=='header' else 0.0`                                                   | `header`                             |
| `trace_writer_sb_v1.body_key_count`     | `{0,1,2,…}` float     | `return float(len(body))`                                                                             | every frame                          |

Every artifact is registered with:

```python
new_artifact(
    kind="feature",
    name="trace_writer_sb_v1.<feature>",
    src=<exact Python body above>,
    doc=<short prose>,
    receipt_schema={"id": "trace_writer_l1_v1",
                    "fields": {"value": "float",
                               "frame_index": "int",
                               "frame_type": "str"}},
    layer="L1",
    module="stepback.trace_writer.sb_v1",
    note="trace_writer.sb v1 base feature",
)
```

The `src_sha256` field of the provenance head is `sha256(src.encode())`,
where `src` is the **literal string** above (newlines included). Two
processes that ship the same stepback wheel are required to produce
the same `src_sha256` for a given feature; conversely, any extractor
edit forces the hash to roll, and the auditor will see the schema id
unchanged but the `src_sha256` change — exactly the desired
"version-pinned extractor body" semantic.

### 7.3 Base-receipt payload — `trace_writer_l1_v1`

```jsonc
{
  "value": 1234.0,                 // float — the per-frame feature value
  "frame_index": 42,               // int   — zero-based position in the writer's HMAC chain
  "frame_type": "step"             // str   — body["type"]; one of:
                                    //   "header" | "step" | "blob" | "merkle_summary"
                                    //   | "tail" | "capability" | "redaction"
                                    //   | any future stepback frame body type
}
```

Invariants (all enforced by the test suite):

* `payload.frame_index >= 0` and is monotonically non-decreasing on
  successive receipts of the same artifact within a single writer.
* `payload.frame_type` is exactly `body.get("type")` after coercion to
  `str`; the emitter does **not** synthesise a default.
* `payload.value` is always `float`, never `int` or `bool`, even for
  the `is_*_frame` indicator features.
* For every observed frame the emitter produces **exactly seven**
  receipts (one per `BASE_FEATURES` entry), in the declaration order
  of `BASE_FEATURES`.

### 7.4 Projection-receipt payload — `trace_writer_l1_projection_v1`

```jsonc
{
  "value": 0.5,                    // float — aggregate value
  "window": "1s",                  // str   — exactly one of "1s" | "10s" | "1m" | "10m"
  "window_ns": 1000000000,         // int   — must equal the canonical width below
  "n_frames_in_window": 17,        // int   — frames the projector saw inside (now - W, now]
  "frame_index": 42                // int   — frame_index of the triggering frame
}
```

Canonical window widths (`window` → `window_ns`) — the auditor MUST
treat any deviation as a versioning error:

| `window` | `window_ns`       |
|----------|-------------------|
| `1s`     | `1_000_000_000`   |
| `10s`    | `10_000_000_000`  |
| `1m`     | `60_000_000_000`  |
| `10m`    | `600_000_000_000` |

The projector exposes **7 base features × 4 windows × 7 aggregates =
196 projection artifacts** per writer. Naming is positional and
deterministic: `<base>.<window>.<aggregate>`, e.g.
`trace_writer_sb_v1.frame_bytes.10s.mean`. The full canonical list
is regenerable as `l1_trace_writer_temporal.projection_names()`.

The seven aggregates (over a `Sequence[float]` of length `n`):

| Aggregate              | Definition                                                                |
|------------------------|---------------------------------------------------------------------------|
| `mean`                 | `sum(xs)/n`                                                               |
| `slope`                | OLS slope of `xs` against `range(n)` (0 when `n < 2`)                     |
| `ewma`                 | EWMA with α = 0.5, seeded with `xs[0]`                                    |
| `range`                | `max(xs) - min(xs)`                                                       |
| `std`                  | population standard deviation                                             |
| `last_minus_first`     | `xs[-1] - xs[0]`                                                          |
| `last_minus_mean_prev` | `xs[-1] - mean(xs[:-1])`                                                  |

For `n == 0` every aggregate returns `0.0` and `n_frames_in_window` is
`0`. Any aggregate edit forces the per-aggregate `src_sha256` to roll
and is detectable by hashing the aggregate's literal `src` body.

### 7.5 Provenance chain for every Step-7 emission

For a base-feature receipt persisted by the Step 3 store after a Step
4 backfill, the chain `head → tail` is:

1. **Creation entry** — written by `make_artifact()`,
   `module="stepback.trace_writer.sb_v1"`,
   `payload_sha256` over `{name, doc, src, receipt_schema}`.
2. **Backfill entry** *(only for backfilled writers)* — written by
   `l1_trace_writer_backfill.backfill_trace`,
   `module="stepback.trace_writer.sb_v1.backfill"`,
   `payload_sha256` over `{source_path, frames_backfilled,
   fingerprint_sha256_64k}`.
3. **Persistence entry** — written by
   `l1_trace_writer_store.persist_writer`,
   `module="stepback.trace_writer.sb_v1.persistence"`,
   `payload_sha256` over `{writer_id_hash, artifact_name}`.

For projection artifacts the creation entry instead carries
`module="stepback.trace_writer.sb_v1.temporal_projection"`. Live
(non-backfilled) writers omit step 2.

### 7.6 On-disk artifact / receipt layout (Step 3)

```
<store_dir>/writers/<writer_id_hash>/
    writer_meta.json                              # {writer_id, created_ns, …}
    artifacts/<feature_name_hash>.json            # canonical AuditableArtifact.to_dict()
    receipts/base.jsonl                           # one Receipt.to_dict() per line
    receipts/projection.jsonl                     # ditto, projection schema
```

Hashes use the first 12 hex chars of the sha256 of the underlying
identifier. Receipts are appended monotonically; every line is
self-contained JSON terminated by `\n`. Append-only watermarking
guarantees `tail -f`-style consumers see each receipt exactly once.

### 7.7 Versioning discipline

The schema id (`trace_writer_l1_v1` / `trace_writer_l1_projection_v1`)
is the **structural** version: any change to receipt field names,
types or count rolls the trailing integer (`…_v2`). The artifact
`src_sha256` is the **semantic** version: any change to the extractor
or aggregate body rolls the hash but keeps the schema id. An external
auditor that observes:

* the same schema id and the same `src_sha256` MUST be able to replay
  any receipt of any older Comet-Σ stepback build;
* the same schema id and a different `src_sha256` MUST treat the new
  receipts as semantically distinct (no cross-build equality);
* a new schema id MUST be treated as a fresh feature space.

### 7.8 Cross-references

* Step 1 implementation — `stepback/comet_sigma/l1_trace_writer.py`
* Step 2 implementation — `stepback/comet_sigma/l1_trace_writer_temporal.py`
* Step 3 implementation — `stepback/comet_sigma/l1_trace_writer_store.py`
* Step 4 implementation — `stepback/comet_sigma/l1_trace_writer_backfill.py`
* Step 5 monotonicity property tests — `tests/test_comet_sigma_l1_trace_writer_temporal_monotonicity.py`
* Step 6 emission-latency benchmark — `tests/test_bench_comet_sigma_l1_trace_writer.py` (results at `bench-results/comet_sigma_l1_trace_writer/`).
* Step 9 Prometheus exporter — `stepback/comet_sigma/l1_trace_writer_prometheus.py`

### 7.9 Prometheus exposition (Step 9)

Each entry of `BASE_FEATURES` is also exposed as a Prometheus
**Gauge** with the single label `writer_id`. Metric names follow
`comet_sigma_l1_trace_writer_sb_v1_<short>`, where `<short>` is the
trailing component of the base-feature name (the dot is dropped to
keep the metric identifier valid). The exporter:

* shares the `COMET_SIGMA_L1_TEMPORAL` flag with Steps 1 and 2 — when
  the flag is OFF the auto-installed observe hook is a no-op and no
  gauge is ever updated;
* installs a single hook into
  `stepback.comet_sigma.l1_trace_writer.OBSERVE_HOOKS` (idempotent;
  removed by `uninstall_hook`) that, on every observed frame, calls
  `Gauge.labels(writer_id=...).set(value)` for each base feature;
* records the wiring as a single `AuditableArtifact`
  (`trace_writer_sb_v1.prometheus_export`, schema id
  `trace_writer_l1_prometheus_v1`) and emits one `Receipt` per
  (frame × base feature) pair pinning `writer_id`, `frame_index`,
  `feature` and `value`;
* renders a Prometheus text-format snapshot via `render_text()` —
  using `prometheus_client.generate_latest` when available and a
  deterministic in-process fallback (same metric names, same label
  set) otherwise.

The fallback exists so the exporter is testable on hosts that do not
ship `prometheus_client`; install the soft dependency to expose the
gauges through a real Prometheus pipeline.

Subsequent rows of the L1 surface table will be filled by Steps 11–90
of `COMET_SIGMA_1000.md`.
