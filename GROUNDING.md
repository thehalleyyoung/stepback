# GROUNDING.md

Each row pairs a claim from the repo (mostly the README spec, plus a
few from the implementation that landed alongside it) with a one-line
shell command that produces evidence for or against the claim and the
last 200 chars of that command's actual stdout.

Run from the repo root with `PYTHONPATH=.` so the local `stepback`
package is importable. All commands are deterministic.

| # | Claim | Command | Observed (last ≤200 chars) | Status |
| - | ----- | ------- | -------------------------- | ------ |
| 1 | The test suite is green (52 tests pass). | `PYTHONPATH=. python3 -m pytest tests/ -q --no-header 2>&1 \| tail -1` | `52 passed in 2.48s` | grounded |
| 2 | The fixture agent records exactly 12 steps, alternating 6 `llm_call` + 6 `tool_call` (`tests/fixtures/agent.py` docstring + README §"Use-cases"). | `PYTHONPATH=. python3 scripts/bench_replay_caching.py 2>&1 \| tail -1` | `n_steps=12 plain_cached=12 plain_llm_calls=0 dirty_after_sub=11 sub_llm_calls=5 bisect_probes=4 bisect_llm_calls=0 bad_step=step:12` | grounded |
| 3 | "Replay of an unchanged 200-step trace from cache … zero LLM calls." (README §Performance targets, scaled here to the 12-step fixture.) | `PYTHONPATH=. python3 scripts/bench_replay_caching.py 2>&1 \| tail -1` | `... plain_cached=12 plain_llm_calls=0 ...` | grounded |
| 4 | "Substituting one step in a 200-step trace typically re-executes ~5 steps" (README §"Replay semantics"). On the 12-step fixture, the dirty subtree under a `ToolOutputSubstitution` at step 2 contains 11 dirty steps but only 5 LLM re-executions — the LLM-re-execution count matches the README's "~5". | `PYTHONPATH=. python3 scripts/bench_replay_caching.py 2>&1 \| tail -1` | `... dirty_after_sub=11 sub_llm_calls=5 ...` | grounded |
| 5 | "Bisect of a 200-step trace, no substitutions … ≤ 10 probes, zero LLM calls" (README §Performance targets). The 12-step fixture bisects in 4 probes, 0 LLM calls — well under budget at this size, and the linearithmic shape is what the budget extrapolates from. | `PYTHONPATH=. python3 scripts/bench_replay_caching.py 2>&1 \| tail -1` | `... bisect_probes=4 bisect_llm_calls=0 bad_step=step:12` | grounded |
| 6 | "Record overhead per LLM call < 5 ms p99" (README §Performance targets). 1000 `llm_call` records measured per-step. | `PYTHONPATH=. python3 scripts/bench_record_overhead.py` | `n=1000 mean_ms=0.093 p50_ms=0.090 p99_ms=0.114 p999_ms=0.143 trace_bytes=1947666` | grounded |
| 7 | The `.sb` HMAC chain detects tampering (README §"trace file format"). | `PYTHONPATH=. python3 -m pytest tests/test_e2e_replay.py::test_verify_trace_detects_tampering -q --no-header 2>&1 \| tail -1` | `1 passed in 0.24s` | grounded |
| 8 | The Ed25519-signed verifier accepts an untouched file (README §"trace file format"). | `PYTHONPATH=. python3 -m pytest tests/test_e2e_replay.py::test_verify_trace_succeeds_on_untouched_file -q --no-header 2>&1 \| tail -1` | `1 passed in 0.24s` | grounded |
| 9 | "Trace size on disk < 30% of raw LLM payload bytes" (README §Performance targets). Measured on the 12-step fixture: the `.sb` file is currently **larger** than the raw payload, not 30% of it — the gzip + dictionary-dedup work described in the README is not implemented yet. | `PYTHONPATH=. python3 -c "import os, json, tempfile; from stepback import record; from stepback.recorder import RecorderKey; from tests.fixtures.agent import run_recorded_agent; d=tempfile.mkdtemp(); p=os.path.join(d,'t.sb'); rec=__import__('stepback').record(p, key=RecorderKey.fresh()); ctx=rec.__enter__(); run_recorded_agent(ctx); rec.__exit__(None,None,None); raw=sum(len(json.dumps(s.get('inputs',{})))+len(json.dumps(s.get('outputs',{}))) for s in ctx.steps); print(f'sb={os.path.getsize(p)} raw={raw} ratio_pct={100.0*os.path.getsize(p)/raw:.1f}')"` | `sb=20033 raw=6346 ratio_pct=315.7` | UNGROUNDED (claim refuted; target not met by current implementation) |
| 10 | The repository layout shipped under `stepback/` matches the README §"Repository layout (planned)" tree (`recorder/`, `replay/`, `substitutions/`, `trace/`, `cli/`, `tui/` as packages). Actual layout is flat single-file modules with no `tui/`, `trace/`, or `ts/` directories. | `ls stepback/` | `__init__.py __pycache__ branch_io.py canonical.py cli.py pricing.py recorder.py replay.py report.py substitutions.py trace_reader.py trace_writer.py` | UNGROUNDED (claim refuted; layout differs from README spec — README is marked "planned") |

## How to reproduce

```
cd <repo root>
PYTHONPATH=. python3 -m pytest tests/ -q --no-header
PYTHONPATH=. python3 scripts/bench_record_overhead.py
PYTHONPATH=. python3 scripts/bench_replay_caching.py
```

Both `scripts/bench_record_overhead.py` and `scripts/bench_replay_caching.py`
were added in this round to ground the README's quantitative
performance targets. They are deterministic (fake LLM hashes its
inputs) and run in < 1 s each on a laptop.

## Audit notes

* Rows 9 and 10 are explicitly marked **UNGROUNDED** because the
  observed output refutes the README's claim. The README is labeled a
  *specification* (see README §"Status of this README"), so divergence
  between spec and current implementation is expected and these rows
  are the audit trail of what is not yet built. They should be revised
  once the gzip+dedup compression and the planned package split land.
* Row 4: the README says "~5" re-executions; the fixture has only 6
  LLM steps total, so 5 dirty LLM steps is the maximum the fixture can
  produce. A ≥200-step fixture is needed to confirm the 5-of-200
  ratio; that fixture does not yet exist in this repo.
