# Proposer 3 — Multi-format pluggable renderer registry

## Framing

Today `report.py` has hand-coded `render_replay_report` and
`render_counterfactual_report` (Markdown), plus
`dump_report_json`. Adding HTML, SARIF, JUnit-XML, etc., one ad-hoc
function at a time will sprawl. Build a **renderer registry**: a
dict of `format_id -> Renderer` plugins, each consuming the same
report-model dict produced by the existing `_build_report_model`.

## Public surface

```python
class Renderer(Protocol):
    format_id: str
    media_type: str
    file_ext: str
    def render(self, model: dict, options: ReportOptions) -> str | bytes: ...

def register_renderer(r: Renderer) -> None: ...
def get_renderer(format_id: str) -> Renderer: ...
def available_formats() -> list[str]: ...
def render_report(
    trace, baseline, counterfactual, subs, *,
    format: str = "markdown", options=None,
) -> str | bytes: ...
```

Built-ins shipped at import time:

| format_id | media_type | file_ext | producer |
| --- | --- | --- | --- |
| `markdown` | `text/markdown` | `.md` | wraps existing `render_*_report` |
| `json`     | `application/json` | `.json` | wraps `dump_report_json` |
| `html`     | `text/html` | `.html` | new (proposer 1's design) |
| `csv-steps` | `text/csv` | `.csv` | one row per step, columns: id, kind, dirty, cost, hash_a, hash_b |
| `junit`    | `application/xml` | `.xml` | one `<testcase>` per substitution; failure if dirty subtree non-empty |

## Why a registry

* Third parties can register formats from outside the package
  (`stepback.report.register_renderer(MySarifRenderer())` in user
  code).
* Single source of truth — the Markdown, HTML, CSV all consume the
  same JSON model. They cannot disagree on what dirty/cached/cost
  means.
* CLI's `--format` autocompletes from `available_formats()`.

## Determinism + safety

* All renderers must be pure functions of the model dict.
* HTML/CSV escape inputs (csv via `csv.writer`, html via
  `html.escape`).
* `dispatch` writes through `Path(out).write_bytes` when renderer
  returns bytes, else `write_text(..., encoding="utf-8",
  newline="\n")`.

## CLI

`stepback report TRACE [--substitute SPEC...] --format FMT [-o OUT]`.
Auto-pick file extension if `OUT` omitted.

## Tests

* `test_registry_has_builtins` — markdown, json, html, csv-steps,
  junit all listed.
* `test_register_custom_renderer_roundtrip` — register a fake
  renderer, dispatch via `render_report(format="x")`.
* `test_csv_steps_has_one_row_per_step`
* `test_junit_marks_failure_when_dirty_subtree_nonempty`
* `test_unknown_format_raises`

## Why this framing

Bets on extensibility. Solves the *future* HTML/SARIF/X requests
in one stroke. Highest leverage but largest surface area: the
registry contract has to be right or downstream code that depends
on the protocol shape will break.
