# Proposer 2 — Severity scoring + SARIF export

## Framing

The Markdown report tells a human "here's what happened". For
incident triage, CI gating, and bisect heuristics we want a single
scalar **severity score** (0..100) plus a structured **SARIF v2.1.0**
artifact so existing security/code-review tooling (GitHub code
scanning, SonarQube, custom dashboards) can ingest stepback
counterfactuals as findings.

## Public surface

```python
@dataclass
class SeverityScore:
    score: int                # 0..100, higher = worse
    level: str                # "info" | "low" | "medium" | "high" | "critical"
    components: dict[str, float]  # per-axis breakdown
    reasons: list[str]        # human strings: "cost +$1.23"

def severity_score(
    baseline: ReplayResult,
    counterfactual: ReplayResult,
    subs: SubstitutionSet,
) -> SeverityScore: ...

def render_sarif(
    trace: Trace,
    baseline: ReplayResult,
    counterfactual: ReplayResult,
    subs: SubstitutionSet,
    *,
    options: Optional[ReportOptions] = None,
) -> dict: ...

def dump_sarif(...) -> str: ...
```

## Scoring rubric (deterministic, no LLM)

Components, each clamped 0..1, then weighted sum × 100:

| Axis | Signal | Weight |
| --- | --- | --- |
| `cost_delta` | `min(1.0, abs(b.total - a.total) / max(a.total, 0.01))` | 0.25 |
| `dirty_fraction` | `b.dirty_count / len(b.steps)` | 0.25 |
| `decision_flips` | count of llm_call steps where finish_reason or first tool_call.name differs / len | 0.30 |
| `subtree_depth` | depth of dirty subtree / total step depth | 0.10 |
| `nondeterminism` | fraction of dirty steps whose nondeterminism_hash changed | 0.10 |

Level bands: `<10 info`, `<25 low`, `<50 medium`, `<75 high`, `>=75
critical`. Stable + reproducible → CI-gateable.

## SARIF mapping

* `runs[0].tool.driver.name = "stepback"`,
  `version = stepback.__version__`.
* Each substitution → one `result` with `ruleId =
  "stepback/" + sub.kind`.
* `result.level` from severity bands; `result.message.text` is the
  headline string.
* `result.properties` carries `cost_delta_usd`, `dirty_count`,
  `step_id_first_divergence`, `severity_score`.
* `runs[0].artifacts[0]` references the trace path.
* SARIF schema 2.1.0 URL pinned. Output validated against the
  published JSON Schema in tests (skip if `jsonschema` not
  installed).

## CLI

`stepback report ... --format sarif` and
`stepback severity TRACE --branch B.json` printing JSON.

## Tests

* `test_severity_score_zero_when_replays_identical`
* `test_severity_score_increases_with_cost_delta`
* `test_severity_components_sum_to_score`
* `test_sarif_has_required_fields`
* `test_sarif_round_trip_through_json_dumps_loads`
* `test_severity_is_deterministic`

## Why this framing

Pushes stepback into the CI lane. A regression test in CI can fail
if `severity_score(...).level >= "high"` after a prompt edit. SARIF
unlocks zero-effort dashboards. The scoring is GOFAI on purpose —
deterministic, byte-identical for the same trace; no LLM cost on
the hot path. (The runbook's GOFAI guidance still applies for
*open-ended judgment* — we deliberately stay rule-based for
reproducibility of the score in regulator replay.)
