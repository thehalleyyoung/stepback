# Proposer 1 — Machine-readable JSON output mode for `stepback/report.py`

## Framing
Today `report.py` only emits Markdown. Real users (CI gates, regulator
pipelines, dashboards) want a structured representation they can diff,
query with `jq`, or feed into a Slack bot. Make the same renderer
produce *both* Markdown and JSON from a single intermediate
representation.

## Concrete plan
1. Introduce a private `_ReportModel` dataclass holding the structured
   shape of a report: trace metadata, substitution summaries (already
   produced by `_summarise_substitution`), cost summary (from
   `_render_cost_summary` data), dirty subtree, decision diffs, and
   the step table rows. All scalars are JSON-safe (str/int/float/bool/None).
2. Add `_build_report_model(trace, baseline, counterfactual_or_none, subs, options)`
   that the existing Markdown renderers `render_replay_report` and
   `render_counterfactual_report` will *also* call internally.
3. Add public `render_report_json(trace, baseline, counterfactual, subs, options)`
   returning a `dict` (or sorted-key JSON string via `dump_report_json`).
   Cost numbers rounded to 8 decimals so the file is byte-stable.
4. CLI: extend `_cmd_report` with `--format {md,json}` (default `md`).
   When `--format json`, emit `dump_report_json(...)` instead of
   Markdown. Adjust subcommand help text.
5. Tests:
   * Round-trip: build model from a fixture, dump JSON, reload, assert
     `decision_diffs` non-empty when there's a tool-output substitution.
   * Determinism: same trace + same subs ⇒ byte-identical JSON.
   * CLI smoke: `stepback report TRACE -s ... --format json -o out.json`
     yields a parseable JSON file with `schema_version` and required keys.

## Why this matters
A counterfactual report is the artifact a regulator or PR reviewer
consumes. Markdown is great for humans; JSON is essential for
programmatic gates ("fail CI if cost_delta > $0.50" on the report,
not on the raw replay). The intermediate model also de-duplicates
the rendering logic — the Markdown becomes a thin formatter over the
same data the JSON exposes.
