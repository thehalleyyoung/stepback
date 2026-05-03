# Layer 2 Refiner — chosen backbone + integrated ideas

## Decision

Adopt **Proposer 1's HTML renderer** as the structural backbone of
this round (smallest blast radius, immediately useful for the
README-stated regulator-replay use-case). Layer in:

* From **Proposer 2**: the deterministic `severity_score()` helper +
  `SeverityScore` dataclass — but expose it as a *first-class symbol*
  in `report.py` and surface its components as a new section in the
  Markdown report ("Severity") and a coloured banner in the HTML
  report. SARIF export is *deferred*: it's a 200+ LOC commitment
  with a JSON-Schema dependency that pushes the round past one
  coherent expansion (Constitution rule 2).
* From **Proposer 3**: the `available_formats()` lookup function,
  but *without* a full pluggable Protocol. Ship a single
  `render_report(format=...)` dispatcher that delegates to the
  three concrete renderers (`markdown`, `json`, `html`) we actually
  have. Keeps surface area honest; a future round can promote it to
  a registry once we have a 4th concrete consumer.

## Concrete file plan (for L3 to deepen)

### `stepback/report.py`

1. New dataclass:
   ```python
   @dataclass(frozen=True)
   class SeverityScore:
       score: int
       level: str
       components: Dict[str, float]
       reasons: List[str]
       def to_dict(self) -> dict: ...
   ```
2. New function `severity_score(baseline, counterfactual, subs) ->
   SeverityScore`. Implements P2's rubric:
   * `cost_delta = min(1.0, abs(b.total - a.total) / max(a.total,
     0.01))`
   * `dirty_fraction = b.dirty_count / max(1, len(b.steps))`
   * `decision_flips`: count llm_call steps where `finish_reason`
     OR first `tool_calls[0].name` differs / max(1, llm_step_count)
   * `subtree_depth`: dirty-subtree max depth / total step depth
     (use `parent_step_id` chain)
   * `nondeterminism`: fraction of dirty steps with changed
     `nondeterminism_hash`
   * Weights: 0.25 / 0.25 / 0.30 / 0.10 / 0.10. Score = round(sum
     × 100). Bands: <10 info, <25 low, <50 medium, <75 high, ≥75
     critical.
   * Each non-zero axis appends one `reason` string.
   * For single-replay (no counterfactual), `severity_score` returns
     a degenerate `SeverityScore(score=0, level="info",
     components={}, reasons=[])` so call sites don't need to branch.
3. New function `render_html_report(trace, baseline, counterfactual,
   subs, *, options=None) -> str` (P1).
   * Reuse `_build_report_model` for the data.
   * Sections: headline, severity banner, substitutions, cost,
     dirty subtree, decision diffs, step timeline.
   * Inline `<style>`. No JS, no remote resources.
   * `html.escape(..., quote=True)` everywhere user content lands.
4. New `render_report(trace, baseline, counterfactual, subs, *,
   format="markdown", options=None) -> str` dispatcher.
5. New `available_formats() -> list[str]` returning
   `["markdown", "json", "html"]`.
6. Extend `ReportOptions` with:
   * `show_severity: bool = True`
   * `html_inline_css: bool = True`
   * `html_collapsed_step_table: bool = True`
7. Wire `severity_score` into `_build_report_model` so the JSON
   model exposes a `severity` key. Markdown renderer emits a
   "Severity" subsection between "Headline" and "Substitutions"
   when `show_severity` is True and a counterfactual exists.

### `stepback/cli.py`

* Extend the `report` subcommand with `--format
  {markdown,json,html}` (default `markdown`). When `-o` is given
  and `--format` is omitted, infer from the file extension.

### `tests/test_report.py`

(Adopt P1's + P2's test list; deepen in L3.)

* HTML renders for replay-only and counterfactual cases.
* HTML is byte-deterministic across two calls.
* HTML escapes `<script>` payloads in substitution text.
* Severity score is 0/info when counterfactual is None.
* Severity score increases monotonically with cost delta.
* Severity components sum (× weights × 100, rounded) equals score.
* `available_formats()` returns the expected list.
* `render_report(format="json")` matches `dump_report_json` byte-
  for-byte.

## What L3 must add (~30% deeper, per advisory)

1. Pin the Markdown report's new "Severity" section format so the
   existing `test_report.py` golden assertions don't break — show
   the table that L3 will produce.
2. Spell out the HTML template literal precisely (header,
   `<style>`, section anchors, table column order) so the
   determinism test isn't a moving target.
3. Decide what happens when `b.total_cost_usd == 0` and
   `a.total_cost_usd == 0` — degenerate denominator. L3 must
   pin: cost component is 0 in that case (no inflation).
4. Pin the depth calculation rule for `subtree_depth` when the
   dirty set is empty — must be 0, not divide-by-zero.
5. CLI `--format` extension semantics + how it interacts with the
   existing `--branch` / `--baseline-branch` flags.

Deferred to a future round (so this stays one coherent expansion):
SARIF, JUnit, CSV-steps, the renderer Protocol, Sphinx integration.
