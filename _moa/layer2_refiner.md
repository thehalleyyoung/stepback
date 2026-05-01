# Layer 2 refiner — synthesis

## Chosen backbone
Take **Proposer 1**'s structural backbone (an intermediate
`_ReportModel` dataclass that both Markdown and JSON renderers feed
off). This is the right architectural decision: every other feature
is just another field on the model. Without it, Proposer 2 and
Proposer 3 each end up duplicating string-formatting logic.

## Integrated ideas
* From **Proposer 2**: the *content* of two new model fields:
  * `first_divergence_step_id` (Optional[str])
  * `causal_attribution: dict[str, list[int]]` mapping a dirty step ID
    to the indices of substitutions that explain it (walk parent
    chains from each substitution's `at_step`).
  Both render as a "Causal attribution" Markdown section AND surface
  in the JSON. The first-divergence finding gets promoted into a
  banner under the H1.
* From **Proposer 3**: the **executive summary** banner concept —
  but trimmed. Drop the full assertions/predicate framework for this
  round (too much surface area for one thematic improvement; it can
  be a follow-up). Keep the verdict-style banner that surfaces the
  headline numbers (cost delta, dirty count, first-divergence step)
  immediately under the H1, plus a derived `verdict` field on the
  model: `"unchanged"` if no diverging steps, `"diverged"` otherwise.
  This is enough to make CI gating possible (`jq -r .verdict`) without
  building the predicate DSL yet.

## Concrete deliverables for layer 3 to finalise
1. `_ReportModel` dataclass with: `schema_version`, `title`,
   `trace_path`, `recorder_version`, `canonicalisation_version`,
   `step_count`, `extra_metadata`, `substitutions`, `cost_summary`
   (with both `baseline` and optional `counterfactual` sub-objects
   plus `delta_total_cost_usd`), `dirty_subtree`, `decision_diffs`,
   `step_table`, `first_divergence_step_id`, `causal_attribution`,
   `verdict`.
2. `_build_report_model(...)` constructs it; both Markdown renderers
   call this and format from the model. New `render_report_json`
   / `dump_report_json` use the model directly.
3. New Markdown sections rendered iff data is non-empty:
   `## Headline` (banner), `## Causal attribution`.
4. CLI: `stepback report ... --format {md,json}`.
5. Tests covering JSON shape, determinism, attribution correctness,
   first-divergence pinpoint, and the new banner.

## Explicit citations
* Backbone: **Proposer 1** (model + JSON output + CLI `--format`).
* Causal attribution + first divergence: **Proposer 2**.
* Executive-summary banner + `verdict` field: **Proposer 3** (the
  predicate DSL part is intentionally deferred).
