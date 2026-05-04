# Reference benchmark results

Reproducible numbers for the two micro-benchmarks shipped under
`stepback/bench/`. The JSON files in this directory are the raw
output of `stepback bench` invocations on a single laptop; the
regenerate command is below.

## Replay-caching / dirty-set propagation

The benchmark builds a deterministic synthetic agent trace using the
real `stepback.recorder.Recorder` (mixed `llm_call` / `tool_call` /
`router` / `parallel_branch_open` / `parallel_branch_join` steps),
applies a single typed substitution (`PromptSubstitution` for
`llm_call` targets, `ToolOutputSubstitution` for `tool_call`
targets), and replays the trace through `Trace.run_replay` against
the real executor. The headline metric is

    cost_reduction_factor = n_steps / mean(dirty_set_size)

i.e. how many real LLM calls per substitution the cache + dirty-set
propagation avoids vs. naive O(N) re-execution.

### `random_step` strategy (uniform-random substitution position)

| n_steps | trials | median dirty-set | p95 | cost reduction |
| ------: | -----: | ---------------: | --: | -------------: |
|      10 |     20 |              3.5 |  10 |          2.41× |
|      50 |     20 |             22.0 |  43 |          2.25× |
|     100 |     20 |             35.5 |  93 |          2.46× |
|     200 |     20 |             87.5 | 184 |          2.22× |
|     500 |     20 |            244.0 | 469 |          1.95× |

### `last_quarter` strategy (substitute in the final 25 % of the trace)

| n_steps | trials | median dirty-set | p95 | cost reduction |
| ------: | -----: | ---------------: | --: | -------------: |
|      10 |     20 |              2.0 |   3 |          5.26× |
|      50 |     20 |              6.0 |  11 |          8.62× |
|     100 |     20 |              8.5 |  22 |         10.64× |
|     200 |     20 |             18.0 |  43 |         10.23× |
|     500 |     20 |             61.5 | 120 |          7.92× |

### Calibration vs. the README headline

The top-level `README.md` advertises a **>60× cost reduction on a
representative 200-step trace** with median dirty-set of 3. The
synthetic traces shipped here do **not** reproduce that exact number
— our measured 200-step reductions are ~2× under uniform-random
single-step substitution and ~10× under late-trace substitution, with
median dirty-set sizes of 88 and 18 respectively.

Two reasons that gap is expected:

1. **Conservative dirty propagation.** `stepback.replay.run_replay`
   uses the parent-dirty fast path: any descendant of a dirty step
   is marked dirty as well, regardless of whether the executor's
   re-computed output happens to match the recorded one. This keeps
   the cache strictly safe but means the dirty set on a synthetic
   linear chain grows with the suffix length.
2. **Synthetic vs. real-agent topology.** The README's >60× claim is
   anchored to a specific recorded customer-payments fixture (see
   `tests/fixtures/agent.py`) where most steps are independent
   look-ups whose outputs the LLM never reads back into a downstream
   prompt. The synthetic builder here is intentionally *worse* for
   the cache: every step has a `context_from_parent` edge to its
   parent, which is the most pessimistic case for dirty-set growth.

The reproducible-headline takeaway is therefore: **on purely-synthetic
traces the dirty-set algorithm delivers a 2–10× cost reduction over
naive O(N) re-execution depending on substitution position, with
late-trace substitutions getting the largest benefit.** Larger
reductions require either (a) traces with more parallel-branch
fan-out (where sibling branches stay clean across substitutions) or
(b) dirty-equality propagation (an open extension that compares
re-executed outputs to recorded outputs to short-circuit propagation
when they happen to match — see `divergence.py` for the comparator).

## Record overhead

`stepback bench record-overhead --n-steps 1000` measures per-step
recorder overhead vs. an unwrapped fake-LLM baseline:

| metric                  | value (µs) |
| ----------------------- | ---------: |
| baseline mean / step    |        2.1 |
| recorded mean / step    |       48.1 |
| recorded p99 / step     |      115.7 |
| trace bytes (1000 step) |    869 143 |

The recorder is therefore ~46 µs / step over the bare LLM call on
this hardware; the README's "< 5 ms p99" target is comfortably met
(115.7 µs ≪ 5 ms).

## Regenerate

```sh
for n in 10 50 100 200 500; do
  stepback bench replay-caching --n-steps $n --n-trials 20 \
      --strategy random_step \
      --out bench-results/replay-caching-n${n}.json
  stepback bench replay-caching --n-steps $n --n-trials 20 \
      --strategy last_quarter \
      --out bench-results/replay-caching-n${n}-last_quarter.json
done
stepback bench record-overhead --n-steps 1000 \
    --out bench-results/record-overhead-n1000.json
```

All inputs are deterministic (RNG seeded from `--seed`, default 0)
so re-runs on the same machine should produce the same dirty-set
distribution; wall-clock numbers will of course vary.
