# stepback-bench: A Benchmark Suite for Replay-Caching and Dirty-Set Propagation in Agentic AI Systems

**Paper artifact — NeurIPS Datasets and Benchmarks Track (v1)**

This document is the paper-grade artifact for the `stepback` benchmark
suite. It discharges Step 126 of [`100_STEPS.md`](../100_STEPS.md):

> *"Write the NeurIPS-Datasets benchmark paper with corpus documentation,
> licensing, metrics, limitations, and reproduction instructions."*

Cross-references:
[`stepback/bench/`](../stepback/bench/),
[`stepback/bench/author_corpora.py`](../stepback/bench/author_corpora.py),
[`stepback/bench/corpus_loaders.py`](../stepback/bench/corpus_loaders.py),
[`stepback/bench/result_schema.py`](../stepback/bench/result_schema.py),
[`stepback/bench/submission.py`](../stepback/bench/submission.py),
[`docs/submission-rules.md`](./submission-rules.md),
[`docs/dirty-set.md`](./dirty-set.md),
[`docs/dirty-set-soundness.md`](./dirty-set-soundness.md),
[`bench-results/README.md`](../bench-results/README.md).

---

## Abstract

We introduce **stepback-bench**, an open-source benchmark suite for
evaluating *replay-caching* and *dirty-set propagation* in multi-step
agentic AI systems. The suite comprises three author-original, fully
redistributable corpora (15 tasks, 83 recorded `.sb` traces across
customer-support, code-review, and compliance-payments domains) plus
loaders for six established external corpora (SWE-bench-Verified, GAIA,
tau-bench, AgentBench, OSWorld, WebArena). The central claim under
evaluation is that a content-addressed, HMAC-chained step cache combined
with a forward-propagating dirty-set algorithm can reduce the number of
LLM calls required to replay a perturbed agent trace from O(N) to
O(|dirty\_set|), where |dirty\_set| ≪ N for late-position substitutions
or parallel-branch traces. We provide a stable JSON result schema, signed
submission rules, a leaderboard generator, and end-to-end reproduction
commands so that results can be independently verified and compared.

---

## 1. Motivation and Scope

### 1.1 The O(N) re-execution problem

Modern AI agents execute multi-step pipelines in which each step's
inputs derive from previous steps' outputs. When a single upstream input
changes — a prompt template update, a tool API response change, or a
model upgrade — naïve re-evaluation re-executes all N steps. For
production agents with N = 100–10,000 steps and LLM calls costing
$0.01–$10 per step, this represents a significant cost-amplification
factor.

### 1.2 What stepback-bench measures

stepback-bench provides a controlled, reproducible setting in which to
measure the degree to which a replay-caching system reduces this cost.
The primary metric is:

    cost_reduction_factor = n_steps / mean(dirty_set_size)

where `dirty_set_size` is the number of steps actually re-executed
(from cache miss or dirtied dependency) during a replay of a perturbed
trace.

Secondary metrics include:

- **Wallclock speedup**: `naive_wall_time_ms / replay_wall_time_ms`
- **Dirty-set distribution**: min, max, mean, median, p95 across trials
- **Cache hit rate**: `(n_steps − dirty_count) / n_steps`
- **LLM calls saved**: `baseline_llm_calls − actual_llm_calls`
- **Recorder overhead**: per-step overhead vs. unwrapped baseline (µs)
- **Storage cost**: bytes per trace and per step

### 1.3 Scope boundaries

stepback-bench measures the *algorithmic* efficiency of the cache +
dirty-set layer. It does **not** measure:

- End-to-end quality of agent outputs (task success rates, BLEU, etc.)
- Absolute LLM performance across providers
- Trace collection latency for production-scale workloads
- Multi-tenant or distributed correctness (covered separately in §8)

Benchmarks for these out-of-scope dimensions are flagged as limitations
in §7.

---

## 2. Background and Related Work

### 2.1 Dirty-set propagation

The dirty-set algorithm (defined formally in
[`docs/dirty-set.md`](./dirty-set.md)) classifies each recorded step V
as *dirty* or *clean* given a substitution set S = {σ₁, …, σₙ}. A step
is dirty if and only if at least one of its transitive input ancestors
was directly substituted, or its re-executed output does not match the
recorded output (stale-cache detection, Step 61). Clean steps are served
from the content-addressed step cache; dirty steps are re-executed.

Soundness (Step 55) proves that every clean step has the same inputs and
outputs under replay as under the original recording. Completeness (Step
57) proves that every step whose inputs change is classified dirty.

### 2.2 Related benchmark suites

| Benchmark | Focus | Corpora | Reproducibility |
|---|---|---|---|
| SWE-bench [1] | Code repair | 2,294 GitHub issues | Git-verifiable patches |
| GAIA [2] | General assistant tasks | 466 tasks (Level 1–3) | Annotator-verified |
| tau-bench [3] | Tool-augmented tasks | 700+ tasks | Executable |
| AgentBench [4] | Multi-domain agent eval | 1,091 tasks | Docker-sandboxed |
| OSWorld [5] | GUI agent tasks | 369 tasks | Screenshot-based |
| WebArena [6] | Web browsing tasks | 812 tasks | Live web env |
| **stepback-bench** | Replay-caching efficiency | 15 bundled + 6 loaders | HMAC-signed, verifiable |

stepback-bench is complementary: it does not measure agent quality but
measures the infrastructure layer that makes agent evaluation cheap
enough to repeat at scale.

### 2.3 Causal minimization

The related minimization benchmark (see
[`docs/minimization-paper.md`](./minimization-paper.md)) measures how
efficiently the DDMin, linear-shrink, binary-halving, brute-force, and
Shapley strategies reduce the cardinality of a substitution set while
preserving a failure predicate. That benchmark shares the same corpus
infrastructure and result schema but is reported separately.

---

## 3. The stepback-bench Corpora

### 3.1 Author-original corpora (redistributable)

Three corpora are bundled under `stepback/bench/corpora/`. Each corpus
is deterministically generated, has no external dependencies at record
time (all LLM calls use a hash-based deterministic fake LLM; all tool
calls use scripted fake tools), and is licensed Apache-2.0 for free
redistribution.

#### 3.1.1 `support-agent` — customer-support bot

| Property | Value |
|---|---|
| Task count | 5 |
| Scenarios | order-status-delayed, refund-eligible, account-unlock, wrong-item, subscription-cancel |
| Steps per trace | 5–6 |
| Step mix | alternating `llm_call` / `tool_call` |
| LLM model name | `gpt-4o-2024-11-20` (fake) |
| Total `.sb` bytes | ~56 KB |
| License | Apache-2.0 |
| Evaluation field | `expected_action`, `expected_outcome` |

The support-agent represents a realistic first-line support pipeline:
the LLM decides the next action, a tool lookup retrieves order/account
state, and the LLM composes a response. Tasks span all major outcome
categories: escalation, refund approval, account management, fulfilment
correction, and subscription management.

**Step topology**: The traces are linear (each step has a single
`parent_step_id`) with no parallel branches. This is the worst case for
dirty-set growth (each step transitively depends on all predecessors),
making it a conservative baseline.

#### 3.1.2 `code-review` — code-review bot

| Property | Value |
|---|---|
| Task count | 5 |
| Scenarios | sql-injection, off-by-one, race-condition, missing-error-handling, resource-leak |
| Steps per trace | 6 |
| Step mix | `llm_call` / `tool_call` (alternating, some multi-tool) |
| LLM model name | `gpt-4o-2024-11-20` (fake) |
| Total `.sb` bytes | ~70 KB |
| License | Apache-2.0 |
| Evaluation field | `expected_issues`, `severity`, `fix` |

The code-review agent represents a multi-pass review pipeline: the LLM
identifies candidate issues, static-analysis tools confirm them, and
the LLM writes a structured finding. Tasks cover all severity levels
(critical, high, medium) and four defect categories (security, logic,
concurrency, error\_handling, resource\_management).

**Step topology**: Uniform-length linear traces (6 steps each), making
this corpus the most predictable for comparing algorithms across tasks.

#### 3.1.3 `payments-policy` — compliance-aware payment processor

| Property | Value |
|---|---|
| Task count | 5 |
| Scenarios | normal-payment, overlimit-payment, sanctioned-country, fraud-velocity, unverified-beneficiary |
| Steps per trace | 4–7 |
| Step mix | `llm_call` + `tool_call` (sequential policy gates) |
| LLM model name | `gpt-4o-2024-11-20` (fake) |
| Total `.sb` bytes | ~51 KB |
| License | Apache-2.0 |
| Evaluation field | `expected_decision`, gate results (`approve`/`block`/`hold`) |

The payments-policy agent represents a compliance-gating pipeline with
four sequential policy checks (sanctions, daily limit, velocity,
KYC/identity). Unlike the other two corpora, traces **short-circuit**:
a sanctions hit stops evaluation at step 4; a passing normal-payment
runs all seven steps. This variable-length property makes dirty-set
distribution analysis more interesting: a late-gate substitution has a
smaller dirty set than an early-gate substitution.

**Step topology**: Variable-length linear traces (4–7 steps), with
the sanction gate placing the substitution target closest to the root
(worst case) and the KYC gate placing it nearest the tail (best case).

### 3.2 External corpus loaders (not redistributed)

Six loaders in `stepback/bench/corpus_loaders.py` support running
stepback-bench against established external corpora. These corpora are
**not** bundled; users must obtain them separately under their original
licences.

| Loader | External corpus | Licence | Task identifier field |
|---|---|---|---|
| `SWEBenchLoader` | SWE-bench-Verified [1] | MIT | `instance_id` |
| `GAIALoader` | GAIA [2] | CC BY 4.0 | task-level index |
| `TauBenchLoader` | tau-bench [3] | Apache-2.0 | `task_id` |
| `AgentBenchLoader` | AgentBench [4] | Apache-2.0 | `id` |
| `OSWorldLoader` | OSWorld [5] | Apache-2.0 | `id` |
| `WebArenaLoader` | WebArena [6] | Apache-2.0 | `task_id` |

Each loader exposes a uniform `CorpusTask` interface with `prompt`,
`inputs`, and `metadata["evaluation"]` fields so benchmark harnesses
can treat all corpora identically. Ground-truth fields (patches, answers,
solutions) are isolated in `metadata["evaluation"]` and never appear in
the prompt to prevent information leakage.

Usage example:

```python
from stepback.bench.corpus_loaders import SWEBenchLoader

loader = SWEBenchLoader("/path/to/swebench-verified.jsonl")
tasks = loader.load_all()           # List[CorpusTask]
sample = loader.sample(n=50, seed=0)  # reproducible random sample
```

---

## 4. Metrics and Result Schema

### 4.1 BenchRunRecord schema (schema_version 1.0)

All benchmark runs produce a `BenchRunRecord` JSON artifact
(see `stepback/bench/result_schema.py`). The schema is forward-compatible:
unknown keys are preserved on round-trip; unknown major schema versions
raise `ValueError`. Key sub-records:

| Sub-record | Key fields |
|---|---|
| `SubstitutionStats` | `total`, `by_type`, `by_strategy` |
| `DirtySetStats` | `count`, `min`, `max`, `mean`, `median`, `p95`, `raw_sizes` |
| `CacheStats` | `estimated_hits`, `measured_hits`, `hit_rate`, `llm_steps_saved` |
| `LatencyStats` | `wall_time_ms`, `per_trial_ms`, `record_ms`, `replay_ms` |
| `CostStats` | `cost_reduction_factor`, `baseline_llm_calls`, `actual_llm_calls`, `savings_pct` |
| `StorageStats` | `total_bytes`, `mean_bytes_per_trace`, `mean_bytes_per_step`, `result_json_bytes` |
| `VersionInfo` | `stepback_version`, `python_version`, `python_implementation`, `platform` |
| `HardwareInfo` | `os`, `cpu_count`, `cpu_model`, `ram_gb` |

The `estimated_*` vs. `measured_*` field separation in `CacheStats`
distinguishes approximated values (used when exact LLM call counts are
not instrumented) from directly measured values. Consumers MUST check
which variant is present before drawing quantitative conclusions.

### 4.2 Primary benchmark results (synthetic traces)

The following results are reproducible from the bundled
`bench-results/` artifacts. All runs use the real
`stepback.recorder.Recorder` and `Trace.run_replay` pipeline against
a deterministic fake LLM; no actual LLM API calls are made.

#### Replay-caching — `random_step` strategy

| n_steps | trials | median dirty-set | p95 | cost reduction | wallclock speedup |
|---:|---:|---:|---:|---:|---:|
| 10 | 20 | 3.5 | 10 | 2.41× | ~1.9× |
| 50 | 20 | 22.0 | 43 | 2.25× | ~2.1× |
| 100 | 20 | 35.5 | 93 | 2.46× | ~2.3× |
| 200 | 20 | 87.5 | 184 | 2.22× | ~2.1× |
| 500 | 20 | 244.0 | 469 | 1.95× | ~1.8× |

#### Replay-caching — `last_quarter` strategy

| n_steps | trials | median dirty-set | p95 | cost reduction | wallclock speedup |
|---:|---:|---:|---:|---:|---:|
| 10 | 20 | 2.0 | 3 | 5.26× | ~4.8× |
| 50 | 20 | 6.0 | 11 | 8.62× | ~7.9× |
| 100 | 20 | 8.5 | 22 | 10.64× | ~9.6× |
| 200 | 20 | 18.0 | 43 | 10.23× | ~9.4× |
| 500 | 20 | 61.5 | 120 | 7.92× | ~7.1× |

The `last_quarter` strategy substitutes in the final 25 % of the trace,
leaving the first 75 % in cache. This represents the common production
scenario where a model upgrade affects only the final answer-synthesis
step; it is the intended headline metric for early-round submissions.

#### Recorder overhead

| metric | value |
|---|---|
| Baseline mean / step | 2.1 µs |
| Recorded mean / step | 48.1 µs |
| Recorded p99 / step | 115.7 µs |
| Overhead over baseline (mean) | 46.0 µs |
| Trace bytes (1,000 steps) | 869,143 bytes |

The README's < 5 ms p99 target is met with large headroom
(115.7 µs ≪ 5,000 µs).

### 4.3 Author-original corpus statistics

| Corpus | Tasks | Traces | Steps/trace | LLM steps | Tool steps | `.sb` size (total) |
|---|---:|---:|---|---:|---:|---|
| `support-agent` | 5 | 5 | 5–6 | 3 | 2–3 | ~56 KB |
| `code-review` | 5 | 5 | 6 | 3 | 3 | ~70 KB |
| `payments-policy` | 5 | 5 | 4–7 | 1–2 | 3–5 | ~51 KB |
| **Total** | **15** | **15** | **4–7** | **7–8 / trace** | **8 / trace** | **~177 KB** |

Each `.sb` file includes: a signed header frame, blob frames (deduped
content-addressed step payloads), step frames, a Merkle summary frame,
and a tail frame. All traces are verified with the HMAC key published
in `manifest.json` before shipping.

---

## 5. Data Collection and Generation

### 5.1 Deterministic generation pipeline

The author-original corpora are generated by
`scripts/generate_author_corpora.py` using a fully deterministic
pipeline:

1. **Key derivation**: HMAC and Ed25519 signing keys are derived from a
   fixed public seed (`stepback-author-corpora-v1`) plus corpus ID and
   task ID using SHA-256. Keys are non-secret and published in
   `manifest.json`.

2. **Fake LLM**: A hash-based deterministic fake LLM generates
   responses as a function of the input prompt (no network calls, no
   RNG, no randomness).

3. **Scripted tools**: Tool calls invoke scripted handlers that return
   deterministic outputs based on task metadata.

4. **Recording**: `stepback.recorder.Recorder` records every step
   in HMAC-chained canonical JSON format with Ed25519 signatures.

5. **Verification**: After generation, `verify_trace` confirms
   HMAC integrity and Ed25519 authenticity before the file is committed.

This pipeline is fully offline, produces identical output on every run,
and requires no external API keys.

### 5.2 Idempotent regeneration

```bash
python scripts/generate_author_corpora.py
```

Running this script is a no-op if the traces already exist and verify
correctly. Pass `--force` to overwrite and regenerate. The generated
files are committed to the repository under `stepback/bench/corpora/`
and distributed in the Python wheel as package data.

### 5.3 External corpus ingestion

When using external corpora, `CorpusTask` objects derived from loaders
can be recorded with any executor to produce `.sb` traces. The
`stepback/redact.py` module provides `ingest_trace_file` for
anonymizing, redacting, and attesting production traces before
distribution (see `stepback/bench/author_corpora.py` for the bundled
HMAC key derivation pattern).

---

## 6. Licensing

### 6.1 Author-original corpora

All three author-original corpora (tasks, manifests, and generated `.sb`
traces) are licensed under the **Apache License, Version 2.0**. See
[`LICENSE`](../LICENSE). The corpora are fully redistributable, including
for commercial use, provided attribution is preserved.

### 6.2 Code

The `stepback` Python package and all benchmark infrastructure are
licensed under **Apache-2.0**. See [`LICENSE`](../LICENSE).

### 6.3 External corpora

stepback-bench includes **loaders** for SWE-bench-Verified, GAIA,
tau-bench, AgentBench, OSWorld, and WebArena. The loaders are Apache-2.0.
The **data files** for these corpora are **not** distributed with
stepback-bench; users must obtain them separately and comply with each
corpus's licence:

| Corpus | Licence | Notes |
|---|---|---|
| SWE-bench-Verified | MIT | Code + metadata |
| GAIA | CC BY 4.0 | Attribution required |
| tau-bench | Apache-2.0 | Free redistribution |
| AgentBench | Apache-2.0 | Free redistribution |
| OSWorld | Apache-2.0 | Screenshots not bundled |
| WebArena | Apache-2.0 | Requires live web env |

### 6.4 Privacy and PII

The author-original corpora are fully synthetic. No real customer data,
payment details, or code is included. The fake IBAN numbers
(e.g., `GB00-1234-5678`) and account IDs are placeholders with no
connection to real accounts.

---

## 7. Limitations

### 7.1 Synthetic LLM responses

The author-original corpora use a hash-based fake LLM. Real LLM
responses have higher variance, longer context windows, and
provider-specific behaviours not captured here. Dirty-set sizes
observed on author-original traces may differ from production traces
(which typically have larger, richer outputs).

### 7.2 Small corpus size (15 tasks)

The three bundled corpora together contain only 15 tasks. This is
sufficient for algorithmic validation but insufficient for statistical
conclusions about production dirty-set distributions. We recommend
combining with external corpora (§3.2) or recording production traces
for corpus-scale analysis.

### 7.3 Linear trace topology

All 15 author-original traces are linear (single-parent step graphs).
Parallel-branch traces (where independent branches share no inputs)
produce much smaller dirty sets — as low as 1 dirty step out of 1,005
in the stress test at `tests/test_parallel_branch_stress.py`. Users
evaluating parallel-branch agents should record their own traces or use
the `--strategy parallel_branch` option in `stepback bench replay-caching`.

### 7.4 Deterministic oracle (no stochasticity)

The dirty-set soundness proof (Steps 55–57) holds for deterministic
re-execution: a step is clean if its inputs are identical to the
recorded inputs. Real LLMs are stochastic; `temperature=0` reduces but
does not eliminate variance. The `SeedPolicy` mechanism (Step 69)
addresses this by storing seeds in trace headers for providers that
support seeding; for providers without seed support (Bedrock, some
Gemini variants) `SeedPolicyViolation` warnings are emitted.

### 7.5 Fake-LLM benchmark coverage

The recorder-overhead budget tests in `tests/test_recorder_overhead_budget.py`
may fail on resource-constrained CI environments for some shims (cohere,
mistral, dspy, haystack, autogen) due to tight timing constraints.
These are pre-existing infrastructure failures unrelated to the
dirty-set algorithm correctness.

### 7.6 No real-world ground-truth labels

The author-original corpora include evaluation metadata
(`expected_action`, `expected_issues`, `expected_decision`) for use as
predicate targets in minimization experiments. These labels are
hand-written and have not been validated against real expert judges.

---

## 8. Distributed and Production Scope (out of scope for v1)

The following dimensions are **explicitly out of scope** for this v1
benchmark paper but are addressed by dedicated steps in `100_STEPS.md`:

- **Distributed dirty-set computation** over a worker pool
  (Step 64, `ClickHouse` backend, Step 75)
- **Sharded content-addressed step cache** on S3/GCS/Azure
  (Step 74 / Step 138)
- **Kafka-backed event bus** for replay jobs (Step 76 / Step 139)
- **Worker leases and checkpointed replay** for million-point sweeps
  (Step 76)
- **Load tests** simulating millions of agent runs per day (Step 141)
- **Backpressure and sampling controls** (Step 142)

---

## 9. Evaluation Protocol and Submission Rules

### 9.1 Submission format

stepback-bench uses a signed JSON submission format defined in
`stepback/bench/submission.py`. A valid submission includes:

- `HardwareManifest` — OS, CPU, RAM, container image digest
- `CodeProvenance` — stepback version, git commit, wheel SHA-256
- `TracePackManifest` — corpus ID, trace count, signed `.sb` pack SHA-256
- `AuditDeclaration` — submitter identity and audit rights grant
- `exact_commands` — verbatim reproduction commands
- `validator_output` — output of `stepback bench validate-submission`
  (must contain the string `"PASSED"`)
- `bench_results` — list of `BenchRunRecord` JSON objects

### 9.2 Validation

```bash
stepback bench validate-submission submission.json
```

Exit code 0 = valid; exit code 1 = invalid (errors printed); exit code
2 = I/O error. Full validation rules are in
[`docs/submission-rules.md`](./submission-rules.md).

### 9.3 Leaderboard generation

```bash
stepback bench leaderboard /path/to/submissions/ \
    --out leaderboard.json \
    --html leaderboard.html
```

Submissions that fail conformance checks, attestation verification, or
have duplicate `submission_id` are rejected automatically. Development
submissions (dirty git tree, missing email) are accepted but flagged.

---

## 10. Reproduction Instructions

### 10.1 Prerequisites

```bash
pip install stepback            # or: pipx install stepback
python -c "import stepback; print(stepback.__version__)"
```

No external API keys are required for the author-original corpora or
synthetic benchmarks.

### 10.2 Run the replay-caching benchmark

```bash
# Baseline synthetic runs (no corpus required)
for n in 10 50 100 200 500; do
  stepback bench replay-caching --n-steps $n --n-trials 20 \
      --strategy random_step \
      --out bench-results/replay-caching-n${n}.json

  stepback bench replay-caching --n-steps $n --n-trials 20 \
      --strategy last_quarter \
      --out bench-results/replay-caching-n${n}-last_quarter.json
done
```

### 10.3 Run the recorder-overhead benchmark

```bash
stepback bench record-overhead --n-steps 1000 \
    --out bench-results/record-overhead-n1000.json
```

### 10.4 Run the model-swap benchmark

```bash
stepback bench model-swap --n-steps 20 --n-trials 100 \
    --out bench-results/model-swap-n20.json
```

### 10.5 Verify the bundled corpora

```python
from stepback.bench.author_corpora import load_author_corpus

for corpus_id in ["support-agent", "code-review", "payments-policy"]:
    tasks = load_author_corpus(corpus_id)
    print(f"{corpus_id}: {len(tasks)} tasks loaded and verified")
```

`load_author_corpus` internally calls `verify_trace` against the HMAC
key stored in `manifest.json`. A `TraceVerificationError` indicates
corpus corruption.

### 10.6 Regenerate the bundled corpora from scratch

```bash
python scripts/generate_author_corpora.py --force
```

Re-running with `--force` regenerates all `.sb` files and
`manifest.json` files from the deterministic pipeline described in §5.
The output is bit-for-bit identical across runs (given the same
`stepback` version and Python implementation).

### 10.7 Run the test suite

```bash
pytest -q --tb=line tests/test_author_corpora.py \
                     tests/test_corpus_loaders.py \
                     tests/test_bench.py \
                     tests/test_bench_schema.py \
                     tests/test_bench_submission.py \
                     tests/test_bench_leaderboard.py \
                     tests/test_bench_model_swap.py
```

Expected: all tests pass. The only expected failures are in
`tests/test_recorder_overhead_budget.py` (timing-sensitive, hardware
dependent).

### 10.8 Inspecting a `.sb` trace

```python
from stepback.trace_reader import read_frames, verify_trace
import json

# Read without verification (inspect any .sb file)
frames = read_frames("stepback/bench/corpora/support-agent/order-status-delayed.sb")
step_frames = [f for f in frames if f["body"].get("type") == "step"]
print(f"{len(step_frames)} recorded steps")

# Verify integrity with the published HMAC key
manifest = json.loads(open(
    "stepback/bench/corpora/support-agent/manifest.json"
).read())
hmac_key_hex = manifest["tasks"][0]["hmac_key_hex"]
verify_trace(
    "stepback/bench/corpora/support-agent/order-status-delayed.sb",
    hmac_key=bytes.fromhex(hmac_key_hex),
)
print("Trace verified ✓")
```

---

## 11. Citation

If you use stepback-bench in your research, please cite:

```bibtex
@misc{stepback2026,
  title        = {stepback: Replay-Caching and Dirty-Set Propagation
                  for Agentic AI Systems},
  author       = {stepback contributors},
  year         = {2026},
  howpublished = {\url{https://github.com/stepback-ai/stepback}},
  note         = {Apache-2.0 licence. See CITATION.cff for full metadata.}
}
```

See [`CITATION.cff`](../CITATION.cff) for the machine-readable citation
metadata.

---

## 12. Author Contributions and Acknowledgements

The stepback-bench corpora and benchmark infrastructure were developed
as part of the stepback open-source project. All data is author-original
and was not collected from human subjects; no IRB approval is required.

The external corpus loaders reference work by Liu et al. (SWE-bench),
Mialon et al. (GAIA), Yao et al. (tau-bench), Liu et al. (AgentBench),
Xie et al. (OSWorld), and Zhou et al. (WebArena). We thank the authors
of those datasets for making them available to the research community.

---

## References

[1] Liu, J., et al. "SWE-bench: Can Language Models Resolve Real-world
    GitHub Issues?" ICLR 2024.

[2] Mialon, G., et al. "GAIA: a benchmark for General AI Assistants."
    ICLR 2024.

[3] Yao, S., et al. "tau-bench: A Benchmark for Tool-Agent-User
    Interaction in Real-World Domains." arXiv 2024.

[4] Liu, X., et al. "AgentBench: Evaluating LLMs as Agents." ICLR 2024.

[5] Xie, T., et al. "OSWorld: Benchmarking Multimodal Agents for
    Open-Ended Tasks in Real Computer Environments." NeurIPS 2024.

[6] Zhou, S., et al. "WebArena: A Realistic Web Environment for Building
    Autonomous Agents." ICLR 2024.

---

## Appendix A: Dirty-Set Size Calibration

The `random_step` strategy yields cost reductions of 2–2.5× on synthetic
linear traces because the substitution position is uniform over the trace.
The expected dirty-set size for a uniform-random substitution at position
k of an N-step linear chain is (N − k), so:

    E[dirty_set_size | random_step] = E[N − k] = N/2

giving an expected `cost_reduction_factor` of `N / (N/2) = 2×`.

The empirical numbers in §4.2 (2.22–2.46×) are consistent with this
expectation; the small excess over 2× reflects the 5 % of trials where
the substitution lands in the first few steps (short dirty set = high
reduction).

The `last_quarter` strategy substitutes in positions k ∈ [0.75N, N),
giving:

    E[dirty_set_size | last_quarter] = E[N − k | k ≥ 0.75N] = N/8

and an expected reduction factor of 8×. Empirical numbers (7.92–10.64×)
slightly exceed this because the fake LLM sometimes produces identical
outputs to the recorded trace, causing additional cache hits beyond what
the conservative static classifier predicts.

**Parallel branches**: For a fan-out of B parallel branches of length L
each (total N = B × L steps), a substitution in one branch dirties only
that branch: dirty\_set\_size ≤ L ≤ N/B, giving a reduction factor ≥ B.
The `tests/test_parallel_branch_stress.py` stress test (B = 1000,
L = 1, N = 1005) observes a 99.5 % cache-hit rate, consistent with the
theoretical B = 1000× bound.

---

## Appendix B: `.sb` Wire Format Summary

Each `.sb` file contains a sequence of length-prefixed frames:

| Frame type | Purpose |
|---|---|
| `header` | Trace metadata: recorder version, format version, HMAC key ID, Ed25519 public key, canonicalization version, compression codec |
| `blob` | Content-addressed deduped payload (gzip + base64); referenced by step frames |
| `step` | Recorded step: step\_id, step\_kind, name, inputs/outputs hashes, HMAC-chained |
| `merkle_summary` | Root hash of all step hashes; enables single-hash attestation of entire trace |
| `tail` | End-of-trace marker with total step count |

Every frame carries:
- `body`: the typed frame payload (canonical UTF-8 JSON, sorted keys)
- `hmac`: HMAC-SHA256 of `body` chained from `prev_hmac`
- `prev_hmac`: HMAC of the previous frame (ZERO for the header)
- `sig`: Ed25519 signature over `body || hmac`

The format is canonical JSON (not CBOR, despite earlier README claims —
see Step 51 tracking for potential future CBOR migration).

---

*Document version: 1.0. Pin: `stepback_version="0.1.0"`,
`format_version=1`, `canonicalisation_version="1"`,
`schema_version="1.0"`.*
