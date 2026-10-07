# Reproducing the evaluation

All numbers in `README.md` and `paper/tool_paper.tex` come from the files under
`results/`, produced by the commands below. Python 3.12, macOS 14 (arm64).

## Environment

```bash
python3.12 -m venv venv && . venv/bin/activate
pip install -e ".[dev,bench]" pytest-xdist pytest-timeout pyarrow pandas tiktoken
# live experiment only
git clone https://github.com/sierra-research/tau-bench.git tau-bench-src
git -C tau-bench-src checkout 59a200c6d575d595120f1cb70fea53cef0632f6b
pip install -e tau-bench-src vcrpy==8.3.0 langgraph==1.2.14 openai==2.54.0
```

## Test suite

```bash
python -m pytest -q -n 2 --timeout 300
```

Timing-budget tests (`-m overhead_budget`) are machine dependent and excluded by
default (`[tool.pytest.ini_options]` in `pyproject.toml`); run them with
`python -m pytest -m overhead_budget`.

## Data (RQ1)

| File | Source | Version | SHA-256 |
| --- | --- | --- | --- |
| `tau_gpt-4o-retail.json` | github.com/sierra-research/tau-bench `historical_trajectories/gpt-4o-retail.json` | commit `59a200c6` | `df01707894836168ff0ec9616b0bf08f66c7e5afcf313e5fe4f7a2f5c2ec938b` |
| `tau_gpt-4o-airline.json` | same repo, `gpt-4o-airline.json` | commit `59a200c6` | `e9e6c0297660c537f83d4fd9c476ce7a9a86ecd2784874b7bfc13be598e37bfa` |
| `tau_sonnet-35-new-retail.json` | same repo, `sonnet-35-new-retail.json` | commit `59a200c6` | `0df526398e9d2720c32d340815cffb04fe8c4f8a61b1f4f84bf3bb558f760131` |
| `tau_sonnet-35-new-airline.json` | same repo, `sonnet-35-new-airline.json` | commit `59a200c6` | `fe62fcd514b855b36f156dd4c3c7748597b392b006aff739b53337a9f3ba94d1` |
| `swea_00000.parquet` | huggingface.co/datasets/nebius/SWE-agent-trajectories `data/train-00000-of-00012.parquet` | revision `68195a14` | `5a395e8c7bb8ddc4b8f4d268506b3a0e2cf9b5ec3922600117322fe788067a13` |
| `oh_swegym.parquet` | huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories `data/train.success.oss-00000-of-00001.parquet` | revision `4aaa5a4a` | `ea4bf37de020e165c5210bedddeef523d8834a89a35a8c65fec24f76f0eae4f1` |

```bash
mkdir -p ../data && cd ../data
for f in gpt-4o-retail gpt-4o-airline sonnet-35-new-retail sonnet-35-new-airline; do
  curl -sL -o tau_$f.json https://raw.githubusercontent.com/sierra-research/tau-bench/59a200c6d575d595120f1cb70fea53cef0632f6b/historical_trajectories/$f.json
done
curl -sL -o swea_00000.parquet https://huggingface.co/datasets/nebius/SWE-agent-trajectories/resolve/68195a1450865274106246d0d0296a1d6807b88e/data/train-00000-of-00012.parquet
curl -sL -o oh_swegym.parquet https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories/resolve/4aaa5a4a4b5861f4799d2336908760c190ac3b17/data/train.success.oss-00000-of-00001.parquet
shasum -a 256 *
cd -
```

## RQ1 and RQ3: static dirty sets on public trajectories

```bash
python scripts/eval_real_traces.py --data ../data --out results/real_traces
```

Samples 150 episodes from each tau-bench file (the first 150), 300 SWE-agent
episodes and 300 OpenHands episodes (seeded shuffle, seed 0), applies five
substitution types per episode and writes `results/real_traces/rows.csv`
(one row per episode and substitution) and `results/real_traces/summary.json`.
Runs offline in about 15 minutes.

## RQ2: live counterfactual replay on tau-bench retail

Needs an OpenRouter key in `OPENROUTER_API_KEY`. Model `openai/gpt-4o-mini`
for agent and simulated user, temperature 0, seed 0.

```bash
python scripts/eval_tau_live.py --tasks 0-49 --out results/tau_live --budget-usd 3.0
python scripts/analyze_tau_live.py results/tau_live/rows.jsonl
```

`rows.jsonl` has one row per (task, method); `summary.json` aggregates over the
tasks that have at least one read-only tool call. The recorded `.sb` traces
with their HMAC keys are in `results/tau_live/traces.tar.gz`, the vcrpy
cassettes (authorization header filtered) in `results/tau_live/cassettes.tar.gz`.
Every live API call is logged with the provider-reported cost in
`results/api_costs.jsonl` (total for the run and a two-task pilot: 1.70 USD).
LLM output is not bit-reproducible across runs (see the paper), so a rerun
gives different suffix lengths and rewards; the reuse counts depend only on
the recorded prefix.

## Bounded local audit

The audit uses Python 3.12.10 and the existing dependency environment, with one
pytest worker. It makes no provider calls. Historical cost logs are recounted
from disk, rather than reconciled against a fresh provider invoice.

```bash
python3 study_audit/recount.py
PYTHONPATH=. python3 study_audit/semantic_controls.py results/audit/semantic-control
PYTHONPATH=. python3 study_audit/structural_controls.py results/audit/semantic-before
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q -n 1 --timeout 300
cd paper
latexmk -pdf -interaction=nonstopmode tool_paper.tex
```

Signed local traces and their local test keys are retained with the controls.
The generic replay control deliberately forwards recorded messages; live
content reconstruction is supplied by an agent adapter. Shared step caches
require a stable `cache_namespace` version and trusted entries. Default
Executor caches are isolated between instances.

## Fresh local boundary control and site

Run from repository root: `python3 study_audit/mutable_state_boundary.py /tmp/stepback-fresh-boundary` (output directory must not already exist). This synthetic experiment makes no provider calls. The original output-wrapper harness failure and corrected12-input result remain under results/audit/. Rebuild the site with `python3 scripts/stage_research_site.py`; all staged local HTML links are checked. GitHub Pages can deploy the staged artifact using the manual Research artifact Pages workflow after repository Pages settings are configured. The sole publication PDF is root `tool_paper.pdf`; its TeX source is paper/tool_paper.tex. No site has been deployed by this audit.
