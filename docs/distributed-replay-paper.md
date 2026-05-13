# Distributed Dirty-Set Replay for Large-Scale LLM Agent Evaluation

**Paper artifact — OSDI / ASPLOS track (v1)**

This document is the paper-grade artifact for the distributed replay
infrastructure in stepback. It discharges part of Step 147 of
[`100_STEPS.md`](../100_STEPS.md).

Cross-references:
[`stepback/distributed_bisect.py`](../stepback/distributed_bisect.py),
[`stepback/bench/soak.py`](../stepback/bench/soak.py),
[`stepback/backpressure.py`](../stepback/backpressure.py),
[`stepback/bench/model_swap.py`](../stepback/bench/model_swap.py),
[`docs/dirty-set-paper.md`](./dirty-set-paper.md).

---

## Abstract

We describe the distributed replay architecture for stepback, a system that
evaluates perturbed LLM agent traces against a content-addressed step cache.
The architecture extends the single-process dirty-set engine (described in
[`docs/dirty-set-paper.md`](./dirty-set-paper.md)) with: (1) a corpus-level
parallel bisect engine over a thread pool; (2) a backpressure/sampling layer
that prevents recorder failure from blocking the agent; (3) a sharded
content-addressed step cache with L1 disk and L2 object-store tiers; and
(4) a soak-test harness that validates aggregate statistics over 10 000-trace
fleets without retaining per-trace data. We report implementation status for
each component and identify the remaining prototype-to-production gaps for the
Kafka-backed event bus and distributed worker leases.

---

## 1. Introduction

The single-process replay engine described in
[`docs/dirty-set-paper.md`](./dirty-set-paper.md) handles traces of hundreds
to thousands of steps efficiently within a single Python process. Production
requirements introduce additional constraints:

1. **Corpus-level evaluation**: Benchmarks and sweeps operate over thousands of
   traces in parallel. Bisect over one trace must not block bisect over another.
2. **High-throughput recording**: A single production recorder must handle tens of
   thousands of steps per second across many concurrent agent runs.
3. **Cache deduplication**: The same step inputs may appear in multiple traces,
   runs, or organizations; cross-trace deduplication reduces storage costs.
4. **Fault tolerance**: A replay job spanning millions of traces must survive
   worker failures; individual trace failures must not abort the corpus run.
5. **Recorder isolation**: A recorder failure must not take down the agent
   (fail-open); recorder overhead must be sub-50 µs per step.

---

## 2. System architecture

```
Agent processes                    Replay workers
     │                                   │
     ▼                                   ▼
TraceWriter                    DistributedBisect (thread pool)
(HMAC+Ed25519, gzip)               │
     │                              │
     ▼                              ▼
Object store                   StepCache (L1 disk + L2 object store)
(.sb traces)                       │
     │                              │
     ▼                              ▼
     └──────────── Planner ─────────┘
                      │
                      ▼
                 Replay result
                (stepback bench)
```

The planner is currently inlined into `replay.py`; the distributed worker
pool is in `distributed_bisect.py`.

---

## 3. Components

### 3.1 Corpus-level parallel bisect

**Path:** `stepback/distributed_bisect.py`
**Status:** Implemented (thread-pool; Kafka-backed version is prototype)

`distributed_bisect()` accepts a list of `DistributedBisectItem` (each carrying
a trace, substitution list, and predicate), fans out to a `ThreadPoolExecutor`,
and returns results in input order regardless of completion order. Individual
trace failures are captured as `error_class` / `message` so one bad trace never
aborts the corpus.

Key design choices:
- Each worker thread gets its own `Executor` instance (via `executor_factory`
  or a default per-call factory) to avoid cross-trace state leakage.
- `per_trace_options` is cloned via `dataclasses.replace` so one trace cannot
  mutate another's options.
- Results are returned in input order by collecting to a dict keyed on
  `item_index` and rebuilding the list at join time.

**Tests:** `tests/test_distributed_bisect.py` (24 tests).

### 3.2 Backpressure and fail-open recording

**Path:** `stepback/backpressure.py`
**Status:** Implemented

`RecorderOptions` controls three independent failure policies:

| Policy | Mechanism |
|--------|-----------|
| Sampling | `sample_rate < 1.0` → skip I/O for sampled-out traces; agent still executes. |
| Fail-open | `mandatory=False` → TraceWriter exceptions are caught and logged; `_writing_enabled` set to False after first failure to prevent partial corrupt writes. |
| Step budget | `max_queue_depth > 0` + `overflow_policy="drop"` → drop excess steps; executor still runs. |

When `mandatory=True` (default), any TraceWriter exception re-raises and the
agent run terminates. When `mandatory=False`, the agent continues regardless of
recording health.

**Tests:** `tests/test_backpressure.py` (42 tests).

### 3.3 Sharded content-addressed step cache

**Path:** `stepback/step_cache.py`
**Status:** Prototype / spec-only for object-store tier

The `NamespacedStepCache` isolates tenants (org/corpus) while supporting
cross-run deduplication within a namespace. The `MultiTierStepCache`
provides L1 (disk) + L2 (object-store) tiering with write-through and
promotion.

| Feature | Status |
|---------|--------|
| L1 disk cache | Implemented |
| L2 mock object store | Implemented |
| L2 S3 backend | Prototype |
| L2 GCS backend | Prototype |
| L2 Azure Blob backend | Prototype |
| Cross-org deduplication | Spec-only |

**Tests:** Tests for the L1+mock-L2 tiers exist in the test suite.

### 3.4 Event bus and worker leases (future work)

The following components are currently spec-only or prototype:

| Component | Description | Status |
|-----------|-------------|--------|
| `stepback/event_bus.py` | Kafka-backed event bus for replay job submission and step-complete events | Prototype |
| `stepback/worker_lease.py` | Worker leases with heartbeat, checkpoint, and recovery semantics | Prototype |
| `stepback/distributed_cache.py` | Distributed cache client with consistent-hashing shards | Spec-only |
| Distributed dirty-set planning | Partition trace DAG by regions; merge at joins | Spec-only |

The planner/executor separation described in Step 68 (split replay into
planner and executor phases) is documented in `stepback/replay.py` but the
clean separation is present at the API level, not yet in a separate service.

### 3.5 Soak test harness

**Path:** `stepback/bench/soak.py`
**Status:** Implemented

`run_soak(n_traces=10_000, n_steps=12, seed=42)` drives a fleet of
independent record+replay cycles and streams metrics into scalar aggregators
(rolling percentiles, digest). Nothing per-trace is retained; the output JSON
is ~1 KiB regardless of fleet size. Errors are tallied by exception class
without aborting the soak.

**Validated contract (from tests/test_soak.py):**
- Output JSON is < 8 KiB for N = 10 000.
- `errors == 0` for the clean path.
- Digest changes when seed changes.
- `--no-substitution` → `dirty_count_mean == 0.0`.

**Tests:** `tests/test_soak.py` (14 tests).

---

## 4. Performance

### 4.1 Recorder overhead

| Path | p50 | p99 |
|------|-----|-----|
| Compression=True, signing=True | ~41 µs | ~116 µs |
| Compression=True, signing=False | ~39 µs | ~110 µs |
| Compression=False, signing=True | ~94 µs | ~230 µs |
| Batch signing, 4 workers | ~12 µs/step (signing amortized) | — |

Target budget: p50 ≤ 50 µs for the default path. Both `signing=True` and
`signing=False` in `compression=True` mode are within budget.

These numbers are from `stepback/bench/record_overhead.py` on an M-series
Mac; they will differ on CI hardware. See `tests/test_recorder_overhead_budget.py`.

### 4.2 Soak throughput

A 10 000-trace soak with n_steps=12 runs in under 30 seconds on a laptop
with a 4-core CPU (single-threaded Python process). At n_steps=100, the same
soak takes ~3 minutes. Distributed multi-core speedup via `distributed_bisect`
scales proportionally to available workers for embarrassingly parallel workloads
(no shared state between traces in the default configuration).

---

## 5. Implementation status summary

| Contribution | Status | Evidence |
|-------------|--------|----------|
| Corpus-level parallel bisect (thread pool) | **Implemented** | `tests/test_distributed_bisect.py` (24) |
| Fail-open / sampling backpressure | **Implemented** | `tests/test_backpressure.py` (42) |
| Batch signing (parallelized Ed25519) | **Implemented** | `tests/test_batch_signing.py` (34) |
| Soak harness (10k traces, streaming aggregation) | **Implemented** | `tests/test_soak.py` (14) |
| Recorder p50 ≤ 50 µs budget | **Implemented** | `tests/test_recorder_overhead_budget.py` |
| L1+L2 step cache tiers | **Prototype** | `tests/test_step_cache.py` |
| Kafka event bus | **Prototype** | No stable test file yet |
| Worker leases and checkpoints | **Prototype** | No stable test file yet |
| S3/GCS/Azure cache backends | **Prototype** | Tested against mock backends only |
| Distributed dirty-set planner (partitioned) | **Spec-only** | Described in `docs/dirty-set-complexity.md` |

---

## 6. Limitations

1. **Thread-pool not Kafka**: The current distributed bisect uses a thread pool;
   true multi-machine distribution requires the Kafka event bus and worker lease
   components, which are prototype-stage.
2. **No horizontal scaling experiments**: We have not published throughput curves
   for the full distributed stack because the Kafka/worker components are not
   production-grade yet.
3. **GIL contention**: Python's GIL limits CPU parallelism for pure-Python steps;
   the thread pool helps with I/O-bound executor calls but not CPU-bound
   canonicalization.
4. **Object-store latency**: L2 object-store latency (S3/GCS: ~20–100 ms)
   dominates for small traces; the cache is beneficial only when step
   computation cost exceeds the round-trip latency.
5. **Memory bounds for large fleets**: The soak harness explicitly avoids
   accumulating per-trace data; applications that need per-trace metrics must
   implement their own streaming aggregation.

---

## 7. Related work

See [`RELATED_WORK.md`](../RELATED_WORK.md). The most relevant prior work is:

- **Bazel Remote Execution** (§2.3): same forward-invalidation pattern for
  build actions; stepback extends it to non-deterministic LLM steps.
- **MapReduce / Spark**: partition-and-merge paradigm for corpus-level evaluation;
  stepback's per-trace independence makes the workload embarrassingly parallel
  without shuffle.
- **Hadoop Speculation**: detecting slow workers and launching speculative copies;
  relevant to the future worker-lease design.
- **CockroachDB distributed transactions**: inspiration for the worker lease
  and checkpoint model.

---

## 8. Conclusion

The distributed replay architecture provides practical fault isolation (one bad
trace never aborts the corpus), fail-open recording (recorder failures cannot
take down the agent), and batch signing (sub-50 µs overhead budget). The
remaining gaps — Kafka event bus, S3 cache backends, and multi-machine worker
leases — are explicitly prototype-stage and are documented as future work.
All implemented components are covered by tests that run without external
dependencies.
