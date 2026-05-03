# Layer 3 Refiner — final design + concrete file plan

## What's locked in from Layer 2

L2 picked **Proposer 1's HTML renderer** as the backbone, layered
**P2's deterministic `severity_score()`** (without SARIF) and
**P3's small `available_formats()` / `render_report(format=...)`
dispatcher** (without the full Protocol registry). L3 keeps all of
that and tightens the contract along the five axes L2 flagged for
deepening.

## L3 deepenings (~30% more depth)

### A. Severity rubric — pinned, with `nondeterminism` axis dropped

`StepView` (`stepback/replay.py:106`) does **not** expose
`nondeterminism_hash` (only `recorded_inputs_hash` /
`current_inputs_hash`). Rather than thread a new field through
replay this round (would balloon scope past one coherent expansion
per Constitution rule 2), drop the nondeterminism axis and
redistribute weights:

| Axis | Signal | Weight |
| --- | --- | --- |
| `cost_delta` | `min(1.0, abs(b.total - a.total) / max(a.total, 0.01))` | 0.30 |
| `dirty_fraction` | `b.dirty_count / max(1, len(b.steps))` | 0.30 |
| `decision_flips` | flips / max(1, llm_step_count) | 0.30 |
| `subtree_depth` | dirty depth / max(1, total depth) | 0.10 |

Edge cases pinned:
* If `a.total_cost_usd == 0` and `b.total_cost_usd == 0`, the
  cost component is **0** (no inflation from the `max(a.total,
  0.01)` floor — that floor only matters when there's a real delta).
* `subtree_depth` = 0 when dirty set is empty (no divide-by-zero
  on `max(1, total_depth)`).
* `decision_flips` only counts steps with `kind == "llm_call"`.
  A "flip" is: `outputs.choices[0].finish_reason` differs OR the
  first tool_call name differs (use the same OpenAI-shape access
  path as `_short_output` in `report.py:241`, swallowing
  `KeyError/IndexError/TypeError`).
* For single-replay (no counterfactual), `severity_score` returns
  `SeverityScore(0, "info", {}, [])` — **callers don't branch**.

Score = `round(weighted_sum * 100)`. Bands:
* `<10` → `info`
* `<25` → `low`
* `<50` → `medium`
* `<75` → `high`
* `≥75` → `critical`

`reasons` (deterministic order: cost, dirty, flips, depth) appends
one short string per non-zero axis, e.g. `"cost +$1.234500"`,
`"dirty 12/40 steps"`, `"3 decision flips"`, `"dirty subtree depth
4/9"`.

### B. HTML template — pinned literal

The HTML body is a single f-string. Top-level structure:

```html
<!doctype html>
<html lang="en"><head>
  <meta charset="utf-8">
  <title>{escaped_title}</title>
  <style>{INLINE_CSS}</style>
</head><body>
  <h1>{escaped_title}</h1>
  <section id="meta"><dl>...trace path / recorder / step_count / extra metadata...</dl></section>
  <section id="severity" data-level="{level}">
    <h2>Severity</h2>
    <p class="badge badge-{level}">{score}/100 — {level}</p>
    <ul>{<li> per reason}</ul>
  </section>
  <section id="headline"><h2>Headline</h2><pre>{escaped_headline_text}</pre></section>
  <section id="substitutions">...</section>
  <section id="cost">...</section>
  <section id="dirty"><h2>Dirty subtree</h2><ul>...</ul></section>
  <section id="decision-diffs">...</section>
  <section id="step-table">
    <h2>Step timeline</h2>
    <table>
      <thead><tr><th>step_id</th><th>kind</th><th>name</th><th>dirty</th><th>cache</th><th>cost_usd</th></tr></thead>
      <tbody>{rows...}</tbody>
    </table>
  </section>
</body></html>
```

`INLINE_CSS` is a module-level constant (not regenerated per call —
keeps determinism + lets a future round override). Rows are emitted
in `step_id` numeric order (same key as `_decision_diff_rows`).
Dirty rows get `class="dirty"`, cache hits `class="cached"`. All
text passes through `html.escape(s, quote=True)` — including
substitution payloads, headline, and metadata values. Costs
formatted with `f"{x:.8f}"` (matches `_round8`).

For single-replay, the HTML omits the `<section id="decision-
diffs">` and the cost section drops the Δ row; severity section
still renders (with `info / 0`).

### C. Markdown "Severity" section

When `options.show_severity` is True and a counterfactual is
present, `render_counterfactual_report` emits between "Headline"
and "Substitutions":

```
## Severity

- score: **{score}/100** ({level})
- cost_delta: {components.cost_delta:.3f}
- dirty_fraction: {components.dirty_fraction:.3f}
- decision_flips: {components.decision_flips:.3f}
- subtree_depth: {components.subtree_depth:.3f}

Reasons: {", ".join(reasons) or "_(none)_"}

```

The single-replay report does **not** emit the Severity section
(degenerate). This keeps `tests/test_report.py`'s existing golden
assertions on single-replay reports byte-stable.

For counterfactual reports, the existing test
`test_render_counterfactual_report_*` may assert specific section
*headers* — if it does, we add the Severity header to the expected
list. (See section F for test plan.)

### D. JSON model extension

`_build_report_model` gains a `"severity"` key at the top level
whenever `counterfactual is not None`, with value
`severity_score(...).to_dict()`. Schema is additive — existing JSON
consumers ignore unknown keys. **Do not** bump
`SCHEMA_VERSION` (additive optional field, per the comment on
`_build_report_model`).

### E. Dispatcher + CLI

```python
def render_report(trace, baseline, counterfactual, subs, *,
                  format="markdown", options=None) -> str:
    options = options or ReportOptions()
    if format in ("markdown", "md"):
        if counterfactual is None:
            return render_replay_report(trace, baseline, subs, options=options)
        return render_counterfactual_report(trace, baseline, counterfactual, subs, options=options)
    if format == "json":
        return dump_report_json(trace, baseline, counterfactual, subs, options=options)
    if format == "html":
        return render_html_report(trace, baseline, counterfactual, subs, options=options)
    raise ValueError(f"unknown format: {format!r}; available: {available_formats()}")

def available_formats() -> List[str]:
    return ["markdown", "json", "html"]
```

CLI (`stepback/cli.py:401`): widen `--format` choices to
`["md", "markdown", "json", "html"]`. When `--output OUT` is given
and `--format` is omitted, infer from the suffix (`.html → html`,
`.json → json`, `.md → markdown`). HTML output writes with
`encoding="utf-8"` and `newline="\n"`.

### F. Test plan (additive — don't break the 132 passing)

In `tests/test_report.py`:

1. `test_severity_score_zero_when_replays_identical` — replay the
   same trace twice with empty subs; severity == 0, level "info".
2. `test_severity_score_increases_with_cost_delta` — patch a
   `ReplayResult` with higher cost; verify monotone increase.
3. `test_severity_components_within_bounds` — every component in
   `[0, 1]`; sum × 100 (rounded) == score.
4. `test_severity_decision_flip_detection` — synthetic two
   `ReplayResult`s with one llm_call whose finish_reason differs;
   `components["decision_flips"] > 0`.
5. `test_severity_single_replay_returns_zero` — counterfactual=None
   path returns degenerate score.
6. `test_html_renders_for_replay_only` — assert `<!doctype html>`,
   `<table`, `Severity` header absent (single-replay) or `data-
   level="info"` if rendered.
7. `test_html_renders_for_counterfactual` — assert `<!doctype
   html>`, `<table`, `class="dirty"` row count == dirty_count.
8. `test_html_is_byte_deterministic` — render twice, assert ==.
9. `test_html_escapes_xss` — substitution payload contains
   `<script>alert(1)</script>`; assert literal string absent and
   `&lt;script&gt;` present.
10. `test_available_formats_lists_three` —
    `["markdown", "json", "html"]`.
11. `test_render_report_dispatch_json_matches_dump_report_json` —
    same bytes as `dump_report_json`.
12. `test_render_report_unknown_format_raises_valueerror`.
13. `test_json_model_includes_severity_when_counterfactual` —
    counterfactual case has `"severity"` key with `score`, `level`,
    `components`, `reasons`.
14. `test_json_model_omits_severity_when_single_replay`.

In `tests/test_branch_io_and_cli.py` (or a new
`test_cli_report_html.py`):

15. `test_cli_report_html_to_file` — invoke
    `stepback report ... --format html -o out.html`; assert file
    contains `<!doctype html>` and at least one `<table`.
16. `test_cli_report_format_inferred_from_extension` — omit
    `--format`; pass `-o out.html`; assert HTML emitted.

### G. Files touched (final)

* `stepback/report.py` — new `SeverityScore` dataclass +
  `severity_score`, `render_html_report`, `render_report`,
  `available_formats`, INLINE_CSS constant, `ReportOptions` extra
  fields, JSON-model `severity` key, Markdown "Severity" section,
  `__all__` extended.
* `stepback/cli.py` — widen `--format` choices, add extension
  inference.
* `tests/test_report.py` — 14 new tests (items 1–14 above).
* `tests/test_branch_io_and_cli.py` — 2 new CLI tests (items
  15–16).

### H. Citations to prior layers

* P1 → HTML renderer surface (`render_html_report`), inline
  CSS-only (no JS) discipline, byte-determinism + XSS escape tests.
  ([_moa/layer1_proposer1.md](_moa/layer1_proposer1.md))
* P2 → `SeverityScore` dataclass shape, the four-axis weighted
  rubric, severity bands, decision-flip detection on
  `finish_reason` / first tool_call.name. ([_moa/layer1_proposer2.md
  ](_moa/layer1_proposer2.md))
* P3 → minimal `render_report(format=...)` dispatcher + the
  `available_formats()` lookup (without the full Renderer Protocol
  registry — deferred). ([_moa/layer1_proposer3.md
  ](_moa/layer1_proposer3.md))
* L2 → backbone choice + scope discipline (defer SARIF and full
  registry). ([_moa/layer2_refiner.md](_moa/layer2_refiner.md))

### I. Out of scope this round

SARIF (P2), JUnit/CSV renderers + Renderer Protocol (P3),
`nondeterminism` axis (needs `StepView` change), Sphinx HTML theme
integration, severity-based CI exit codes for `stepback replay`.
