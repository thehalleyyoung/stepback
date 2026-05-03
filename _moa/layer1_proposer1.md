# Proposer 1 — Standalone HTML renderer for counterfactual reports

## Framing

`stepback/report.py` ships a rich Markdown renderer
(`render_counterfactual_report`) and a JSON dump
(`dump_report_json`). What's missing for the regulator-replay /
incident-write-up workflow described in `README.md` is a
**self-contained HTML** artifact that opens in a browser without a
server, embeds inline CSS, and visually highlights the dirty subtree
+ decision diffs + cost delta.

## Public surface to add

```python
def render_html_report(
    trace: Trace,
    baseline: ReplayResult,
    counterfactual: Optional[ReplayResult],
    subs: SubstitutionSet,
    *,
    options: Optional[ReportOptions] = None,
) -> str: ...

def dump_html_report(*, ..., out_path: str | None = None) -> str: ...
```

Plus an extra ReportOptions field:

```python
html_inline_css: bool = True
html_collapsed_step_table: bool = True
```

## Architecture

1. Build the JSON model via the existing `_build_report_model` (do
   NOT duplicate logic — reuse).
2. Walk that model into `<section>` blocks: headline, substitutions,
   cost, dirty subtree, decision diffs, step timeline.
3. Emit semantic HTML5: `<table>` for the step timeline (with
   `<thead>`, `<tbody>`, `data-step-id` attributes), `<details>`
   blocks for collapsible sections.
4. Inline CSS: a tiny `<style>` block with `.dirty { background:
   #ffe9e9 }`, `.cached { background: #e9f7e9 }`, `.delta-pos {
   color:#a30 }`, `.delta-neg { color:#063 }`.
5. Escape all text with `html.escape(..., quote=True)`. No external
   resources, no JS — air-gapped reviewer can open the file.

## Determinism

Byte-identical for the same `(trace, baseline, counterfactual, subs,
options)` triple. No timestamps in the HTML body. Costs rendered
through the same `_round8`. Hash the rendered string in tests for
golden-file comparison.

## CLI

Extend `stepback report ... --format {markdown,json,html}` (default
`markdown` for back-compat). `cli.py` already has the report
subcommand — bolt on a `--format html` branch that calls
`render_html_report` and writes to `-o OUT.html`.

## Tests

`tests/test_report.py` adds:
* `test_html_renders_for_replay_only`
* `test_html_renders_for_counterfactual` (assert `<table`,
  `class="dirty"` present iff dirty steps exist, cost delta sign)
* `test_html_is_deterministic` — render twice, assert byte-equal
* `test_html_escapes_xss` — substitute a prompt containing
  `</script><img src=x onerror=alert(1)>` and assert the literal
  bytes appear escaped (`&lt;/script&gt;...`).

## Why this framing

Lowest-risk additive expansion. Reuses the existing JSON model so
renderers can never disagree. Air-gapped HTML matches the regulator
use-case in the README ("regulator-replay use-case"). XSS test is
not paranoia — substitution payloads are user-controlled.
