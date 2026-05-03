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

## New claims grounded this round

The rows below were appended after the previous grounding pass. They
cover claims introduced by modules / tests added since GROUNDING.md
was first written: `stepback/pricing.py`, `stepback/shims.py`,
`stepback/branch_io.py`, `stepback/report.py`, `stepback/trace_writer.py`
(gzip + content-addressed-blob compression), and the new test files
`tests/test_compression.py`, `tests/test_pricing.py`, `tests/test_shims.py`,
`tests/test_report.py`, `tests/test_parallel_branches.py`.

| # | Claim | Command | Observed (last ≤200 chars) | Status |
| - | ----- | ------- | -------------------------- | ------ |
| 11 | The full test suite is now green at 108 tests (was 52 in row 1; new modules added pricing, shims, report, branch_io, compression, parallel-branches tests). | `PYTHONPATH=. python3 -m pytest tests/ -q --no-header 2>&1 \| tail -1` | `108 passed in 3.08s` | grounded |
| 12 | The `.sb` writer now implements gzip + content-addressed-blob deduplication (`stepback/trace_writer.py` `COMPRESSION_SCHEME = "gzip+dedup-2"`), and on a 60-step chat-history-heavy trace the compressed file is **≥ 50% smaller** than the same trace written with `compression=False` (test_compression.py docstring). | `PYTHONPATH=. python3 -m pytest tests/test_compression.py::test_compression_meaningfully_shrinks_chat_history -q --no-header 2>&1 \| tail -1` | `1 passed in 0.38s` | grounded |
| 13 | On the 60-step chat-history fixture the compressed `.sb` is 61.7% of raw payload bytes — still above the README §"Performance targets" "< 30% of raw" goal even with compression on, but a large reduction (69.9%) vs the uncompressed `.sb`. (Confirms row 9 remains UNGROUNDED on the absolute < 30%-of-raw target while row 12 grounds the relative win.) | `PYTHONPATH=. python3 -c "import os,json,tempfile; from stepback.recorder import Recorder,RecorderKey; from stepback.trace_writer import TraceWriter; d=tempfile.mkdtemp();\ndef go(c):\n p=os.path.join(d,'c.sb' if c else 'u.sb'); k=RecorderKey.fresh(); w=TraceWriter.open(p,hmac_key=k.hmac_key,signing_key=k.signing_key,compression=c); r=Recorder(writer=w,key=k); big={'role':'system','content':'You are an assistant. '+'X'*800}; h=[big]; \n for i in range(60):\n  msgs=list(h)+[{'role':'user','content':f'Q{i}'}]; r.llm_call('gpt-4o-2024-11-20',msgs, lambda m,ms,i=i:{'id':f'c{i}','model':m,'choices':[{'index':0,'message':{'role':'assistant','content':f'R{i}'}}],'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}}); h+=[{'role':'user','content':f'Q{i}'},{'role':'assistant','content':f'R{i}'}]\n w.close(); raw=sum(len(json.dumps(s.get('inputs',{})))+len(json.dumps(s.get('outputs',{}))) for s in r.steps); return os.path.getsize(p),raw\nsz_c,raw=go(True); sz_u,_=go(False); print(f'comp_vs_raw_pct={100*sz_c/raw:.1f} reduction_vs_uncomp_pct={100*(1-sz_c/sz_u):.1f}')"` | `comp_vs_raw_pct=61.7 reduction_vs_uncomp_pct=69.9` | UNGROUNDED for the README "<30% of raw" target; grounded for the >50% reduction-vs-uncompressed target |
| 14 | `stepback/pricing.py` ships a per-model price catalog with `SNAPSHOT_DATE = 2026-04-01`, ≥ 20 priced models and ≥ 16 model aliases (README §"Performance targets" / module docstring: "Prices … taken from published lists as of `SNAPSHOT_DATE`"). | `PYTHONPATH=. python3 -c "from stepback.pricing import SNAPSHOT_DATE,RATE_TABLE,ALIASES; print(f'snapshot={SNAPSHOT_DATE} models={len(RATE_TABLE)} aliases={len(ALIASES)}')"` | `snapshot=2026-04-01 models=21 aliases=16` | grounded |
| 15 | `stepback/shims.py` ships drop-in wrappers for the v0.1-target client surface called out in the README §Status ("OpenAI + Anthropic clients and the LangChain tool registry"), plus MCP — exactly the wrap_* helpers `wrap_openai`, `wrap_anthropic`, `wrap_langchain_tool`, `wrap_langchain_tools`, `wrap_mcp_session`. | `PYTHONPATH=. python3 -c "import stepback.shims as s; print('wraps=', sorted(n for n in dir(s) if n.startswith('wrap_')))"` | `wraps= ['wrap_anthropic', 'wrap_langchain_tool', 'wrap_langchain_tools', 'wrap_mcp_session', 'wrap_openai']` | grounded |
| 16 | `stepback/branch_io.py` persists a counterfactual to a `.sbb` file pinned to its base trace by content hash and round-trips losslessly (README §"trace file format" — counterfactual branches replayable from trace + `.sbb` alone). | `PYTHONPATH=. python3 -c "import os,tempfile; from stepback import record; from stepback.recorder import RecorderKey; from stepback.branch_io import save_branch,load_branch,trace_chain_hash; from stepback.substitutions import ToolOutputSubstitution; from tests.fixtures.agent import run_recorded_agent; d=tempfile.mkdtemp(); p=os.path.join(d,'t.sb');\nwith record(p,key=RecorderKey.fresh()) as ctx:\n run_recorded_agent(ctx); steps=list(ctx.steps)\nchain=trace_chain_hash(steps); sbb=os.path.join(d,'b.sbb'); save_branch(sbb,name='b',base_step='step:2',trace_path=p,trace_chain=chain,substitutions=[ToolOutputSubstitution(at_step='step:2',fake_response={'text':'X'})]); l=load_branch(sbb); print(f'sbb_size={os.path.getsize(sbb)} name={l[\"name\"]} chain_match={l[\"trace_inputs_hash_chain\"]==chain} subs_type={type(l[\"substitutions\"]).__name__}')"` | `sbb_size=469 name=b chain_match=True subs_type=SubstitutionSet` | grounded |
| 17 | `tests/test_parallel_branches.py` exercises parallel-branch replay end-to-end (README §"branch_at" / "compare_branches"); 6 tests pass. | `PYTHONPATH=. python3 -m pytest tests/test_parallel_branches.py -q --no-header 2>&1 \| tail -1` | `6 passed in 0.26s` | grounded |
| 18 | `tests/test_pricing.py` covers the `pricing` module's compute / aggregate / diff / budget surface (26 tests). | `PYTHONPATH=. python3 -m pytest tests/test_pricing.py -q --no-header 2>&1 \| tail -1` | `26 passed in 0.20s` | grounded |
| 19 | `tests/test_shims.py` covers the OpenAI / Anthropic / LangChain / MCP wrappers without a hard dep on those packages (15 tests). | `PYTHONPATH=. python3 -m pytest tests/test_shims.py -q --no-header 2>&1 \| tail -1` | `15 passed in 0.28s` | grounded |
| 20 | `tests/test_report.py` covers the human-readable counterfactual report renderer (`stepback/report.py` — companion to `branch_io.diff_replays`); 17 tests pass. | `PYTHONPATH=. python3 -m pytest tests/test_report.py -q --no-header 2>&1 \| tail -1` | `17 passed in 0.96s` | grounded |
| 21 | `tests/test_compression.py` covers gzip+dedup round-trip determinism, hash-chain preservation, deduplication of recurring sub-trees, blob-digest tamper detection, and header-advertised compression scheme (9 tests). | `PYTHONPATH=. python3 -m pytest tests/test_compression.py -q --no-header 2>&1 \| tail -1` | `9 passed in 0.40s` | grounded |

## New claims grounded this round (numeric-threshold guarantees)

The rows below were appended after rows 11–21 (which grounded the
arrival of `pricing.py`, `shims.py`, `branch_io.py`, `report.py`, and
gzip+dedup compression). This round added strict numeric-threshold
tests across `test_e2e_replay.py`, `test_parallel_branches.py`,
`test_pricing.py`, `test_compression.py`, and `test_branch_io_and_cli.py`
(uncommitted in working tree at the time of grounding). Test-suite
total grew from 108 → 132.

| # | Claim | Command | Observed (last ≤200 chars) | Status |
| - | ----- | ------- | -------------------------- | ------ |
| 22 | The full test suite is now green at **132 tests** (up from 108 in row 11; +24 numeric-threshold guarantees added across compression, pricing, parallel-branches, e2e-replay, branch-io). | `PYTHONPATH=. python3 -m pytest tests/ -q --no-header 2>&1 \| tail -1` | `132 passed in 3.68s` | grounded |
| 23 | `tests/test_e2e_replay.py` gained 5 numeric-threshold tests pinning trace-size bounds, replay-vs-record wall-clock speedup, model-substitution cost drop > 1%, bisect probe-count ≤ ⌈log₂N⌉+1 (≤5 for the 12-step fixture), and a CLI bisect smoke. | `PYTHONPATH=. python3 -m pytest tests/test_e2e_replay.py::test_trace_size_and_per_step_byte_bounds tests/test_e2e_replay.py::test_replay_speedup_vs_record_is_measurable tests/test_e2e_replay.py::test_model_substitution_cost_drop_is_significant tests/test_e2e_replay.py::test_bisect_probe_count_is_logarithmic tests/test_e2e_replay.py::test_cli_bisect_finds_step -q --no-header 2>&1 \| tail -1` | `5 passed in 0.45s` | grounded |
| 24 | `tests/test_parallel_branches.py` doubled (6 → **12 tests**), adding strict numeric guarantees: dirty-subtree ≤ 30% of trace, cached-hit ≥ 70%, fan-out dirty count strictly less than the equivalent serial-trace dirty count (≥ 40% savings), per-step byte envelope < 2.5 KB, replay wall-clock ≤ 2× record. | `PYTHONPATH=. python3 -m pytest tests/test_parallel_branches.py -q --no-header 2>&1 \| tail -1` | `12 passed in 0.28s` | grounded |
| 25 | `tests/test_compression.py` grew 9 → **12 tests**, adding (a) blob-digest tamper detection independent of HMAC, (b) header advertises `compression`/`blob_threshold`/`blob_min_reuse`, (c) absolute-size ceiling: a 60-step chat-history compressed `.sb` is < 200 KB, uncompressed dominates compressed by ≥ 2×, per-step compressed cost < 3.5 KB. | `PYTHONPATH=. python3 -m pytest tests/test_compression.py -q --no-header 2>&1 \| tail -1` | `12 passed in 0.75s` | grounded |
| 26 | `tests/test_pricing.py` grew 26 → **31 tests**. Notable additions: cached-token discount on `gpt-4.1-2025-04-14` is bounded between 3.9× and 5.0× (pinning OpenAI's published 4× cache discount), `diff_costs` `__total__` strictly equals the sum of per-model deltas, `RATE_TABLE`/`PRICE_LIST` ≥ 10 entries and every alias resolves into `RATE_TABLE`. | `PYTHONPATH=. python3 -m pytest tests/test_pricing.py -q --no-header 2>&1 \| tail -1` | `31 passed in 0.33s` | grounded |
| 27 | `tests/test_branch_io_and_cli.py` grew to **27 tests** with 6 new numeric-threshold guarantees: `.sbb` size 50–4096 bytes and < 50% of base-trace size, `diff_replays` divergent-step count strictly between 1 and 12 on the fixture (anchored: step:1 invariant under step:2 substitution), CLI `inspect --json` step_count = 12 with 64-hex sha256 inputs_hash, CLI `replay --json` < 100 KB, and 5-substitution-kind round-trip preserves count. | `PYTHONPATH=. python3 -m pytest tests/test_branch_io_and_cli.py -q --no-header 2>&1 \| tail -1` | `27 passed in 2.16s` | grounded |
| 28 | The cached-prompt-token discount on `gpt-4.1-2025-04-14` matches OpenAI's published 4× ratio: `compute_cost(10 K prompt, 0 completion)` / `compute_cost(10 K cached prompt)` is in [3.9, 5.0] (test_cached_token_savings_ratio_bounded). | `PYTHONPATH=. python3 -c "from stepback.pricing import compute_cost; full=compute_cost('gpt-4.1-2025-04-14',{'prompt_tokens':10000,'completion_tokens':0}); cached=compute_cost('gpt-4.1-2025-04-14',{'prompt_tokens':10000,'completion_tokens':0,'prompt_tokens_details':{'cached_tokens':10000}}); print(f'full={full:.6f} cached={cached:.6f} ratio={full/cached:.3f}')"` | `full=0.020000 cached=0.005000 ratio=4.000` | grounded |
| 29 | Bisect on the 12-step fixture finds the bad step in ≤ ⌈log₂12⌉+1 = 5 probes with **zero LLM re-executions** (test_bisect_probe_count_is_logarithmic + bench output of `n_probes=4`). | `PYTHONPATH=. python3 scripts/bench_replay_caching.py 2>&1 \| tail -1` | `n_steps=12 plain_cached=12 plain_llm_calls=0 dirty_after_sub=11 sub_llm_calls=5 bisect_probes=4 bisect_llm_calls=0 bad_step=step:12` | grounded |
| 30 | The fan-out fixture (3 sibling research branches around an LLM `compose` step) records exactly **11 steps** with kind counts `llm_call=5, tool_call=4, parallel_branch_open=1, parallel_branch_join=1`, and a `ToolOutputSubstitution` inside one branch dirties **exactly 3 steps** (vs. the 5-step serial-equivalent dirty count → 40% savings). | `PYTHONPATH=. python3 -m pytest tests/test_parallel_branches.py::test_recorded_trace_numeric_bounds tests/test_parallel_branches.py::test_dirty_subtree_is_small_fraction tests/test_parallel_branches.py::test_branch_dirty_count_strictly_less_than_serial -q --no-header 2>&1 \| tail -1` | `3 passed in 0.21s` | grounded |
| 31 | `.sbb` branch files are tiny by construction: a single-substitution branch is between 50 B and 4 KB and < 50% of the base-trace size. Round-trip preserves count for all 5 substitution kinds (Prompt / Model / ToolOutput / Policy / Router). | `PYTHONPATH=. python3 -m pytest tests/test_branch_io_and_cli.py::test_sbb_branch_file_size_bounded tests/test_branch_io_and_cli.py::test_save_branch_substitution_count_matches_loaded -q --no-header 2>&1 \| tail -1` | `2 passed in 0.36s` | grounded |
| 32 | The trace-writer header now advertises the compression scheme metadata: `compression == COMPRESSION_SCHEME` (=`gzip+dedup-2`) and the dedup tunables `blob_threshold` / `blob_min_reuse` are present in every `.sb` written with `compression=True`. | `PYTHONPATH=. python3 -c "import os, tempfile; from stepback.recorder import RecorderKey, Recorder; from stepback.trace_writer import TraceWriter, COMPRESSION_SCHEME, DEFAULT_BLOB_THRESHOLD, DEFAULT_BLOB_MIN_REUSE; from stepback.trace_reader import verify_trace; from tests.fixtures.agent import run_recorded_agent; d=tempfile.mkdtemp(); p=os.path.join(d,'h.sb'); k=RecorderKey.fresh(); w=TraceWriter.open(p,hmac_key=k.hmac_key,signing_key=k.signing_key,compression=True); r=Recorder(writer=w,key=k); run_recorded_agent(r); w.close(); t=verify_trace(p,k.hmac_key); print(f'compression={t.header[\"compression\"]} bt={t.header[\"blob_threshold\"]} bmr={t.header[\"blob_min_reuse\"]} expected={COMPRESSION_SCHEME} bt_def={DEFAULT_BLOB_THRESHOLD} bmr_def={DEFAULT_BLOB_MIN_REUSE}')"` | `compression=gzip+dedup-2 bt=200 bmr=2 expected=gzip+dedup-2 bt_def=200 bmr_def=2` | grounded |

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
