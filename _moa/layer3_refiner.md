# Layer 3 refiner — final implementation spec (this is what gets applied)

Inherits the synthesis from `layer2_refiner.md`; this layer pins the
*exact* shape of code, JSON schema, CLI flags and tests.

## Citations
* **Proposer 1** — `_ReportModel` intermediate dataclass; `render_report_json`
  + `dump_report_json` public surface; CLI `--format {md,json}`. (Layer 2
  adopted this as the backbone.)
* **Proposer 2** — `_attribute_dirty_steps` helper, `_first_divergence` helper,
  Markdown `## Causal attribution` section, JSON fields
  `first_divergence_step_id` + `causal_attribution`.
* **Proposer 3** — executive-summary banner under the H1 (verdict line +
  cost delta + first-divergence step). The predicate-assertion DSL from
  Proposer 3 is **deferred** as Layer 2 decided.

## Module changes (`stepback/report.py`)

1. Add `SCHEMA_VERSION = 1` constant.
2. Add `_ReportModel` dataclass (private) with fields:
   ```
   schema_version: int
   title: str
   trace_path: str
   recorder_version: Optional[str]
   canonicalisation_version: Optional[str]
   step_count: int
   extra_metadata: dict
   substitutions: list[dict]   # {index, summary, kind, at_step}
   cost_summary: dict          # baseline / counterfactual / delta
   dirty_subtree: list[dict]   # step_id, kind, name, cost_usd
   decision_diffs: list[dict]  # step_id, kind, a, b, cost_delta_usd
   step_table: list[dict]      # per-row dict
   first_divergence_step_id: Optional[str]
   causal_attribution: dict    # step_id -> [substitution_index]
   verdict: str                # "unchanged" | "diverged" | "no-counterfactual"
   ```
3. Add `_build_report_model(trace, baseline, counterfactual, subs, options)`.
4. Add `_attribute_dirty_steps(steps, subs) -> dict[str, list[int]]`:
   * Build `parents: dict[step_id, parent_step_id]` from `steps`.
   * For each substitution index `i` with `at_step = sid`, every step
     whose parent-chain passes through `sid` (inclusive) is attributed
     to `i` *if* it is dirty in `steps`. Use BFS forward over
     `children = inverse(parents)`.
5. Add `_first_divergence(a_steps, b_steps) -> Optional[str]`:
   * Walk steps sorted by integer suffix; first id where `hash_obj(a.outputs)
     != hash_obj(b.outputs)` is the answer. Returns `None` if none diverge
     or `b_steps` is None.
6. Make `render_replay_report` and `render_counterfactual_report`
   build the model first, then format from it. Add new sections
   in this order: H1 → `## Headline` (banner; rendered iff
   `counterfactual` provided OR `verdict != "no-counterfactual"`)
   → trace metadata bullets → existing sections … → new
   `## Causal attribution` section *just before* `## Step timeline`
   (only when there are attributed steps).
7. Add public `render_report_json(...)` returning a `dict`, and
   `dump_report_json(...)` returning sorted-keys JSON string with
   `indent=2`. Float costs rounded to 8dp via the model itself.
8. Export `render_report_json`, `dump_report_json` from `stepback.report`
   `__all__` and re-export from `stepback/__init__.py`.

## CLI changes (`stepback/cli.py`)
* Add `--format {md,json}` (default `md`) to the `report` subparser.
* In `_cmd_report`, when `--format json`, build a counterfactual or
  single-replay model and emit `dump_report_json(...)`.

## Tests (`tests/test_report.py`)
Add the following (all using the existing `run_recorded_agent` fixture):

* `test_render_report_json_round_trip`: produce JSON, `json.loads` it,
  assert required keys (`schema_version`, `verdict`, `cost_summary`,
  `decision_diffs`, `causal_attribution`).
* `test_first_divergence_pinpoints_substituted_step`: with a
  `ToolOutputSubstitution` at `step:2`, the first-divergence id is
  exactly `"step:2"`.
* `test_causal_attribution_covers_all_dirty_steps`: every dirty step
  appears in `causal_attribution` and maps to substitution index 0
  (the only sub).
* `test_render_report_json_is_byte_stable`: same inputs ⇒ identical
  string twice.
* `test_headline_banner_in_markdown`: counterfactual Markdown contains
  `Headline` heading and the first-divergence step id.
* `test_cli_report_json_format`: `python -m stepback report TRACE -s
  ... --format json -o out.json` writes parseable JSON with
  `verdict == "diverged"`.

## Compatibility
All existing tests must still pass — keep:
* `render_replay_report` / `render_counterfactual_report` signatures
  unchanged.
* All existing Markdown headings (`## Substitutions`, `## Cost summary`,
  `## Dirty subtree`, `## Decision diffs`, `## Step timeline`)
  preserved exactly. Only *additional* sections are inserted.
* `_summarise_substitution`, `_render_dirty_subtree`,
  `_render_step_table` remain (model-builder calls them).

## Verification
* `pytest -x -q` must pass.
