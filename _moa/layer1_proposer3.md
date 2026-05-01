# Proposer 3 — Pass/fail verdict + executive summary banner in `stepback/report.py`

## Framing
Reports today are a wall of tables. A reviewer scrolling on a phone
needs the headline first: did the counterfactual *fix* the issue, or
did it make things worse? Add a verdict layer driven by user-supplied
predicates, with an executive summary at the top.

## Concrete plan
1. Extend `ReportOptions` with:
   * `assertions: list[ReportAssertion]` — each has `name: str`,
     `predicate: Callable[[ReplayResult], bool]`, `severity: str`
     (`"error"`/`"warn"`/`"info"`).
   * `show_executive_summary: bool = True`.
2. New `## Executive summary` section rendered immediately after the
   H1 header containing:
   * **Verdict:** PASS / FAIL / WARN, derived from worst-severity
     failing assertion.
   * total cost A vs B and Δ
   * dirty count, real-execution count for B
   * one-line of each substitution's summary
3. New `## Assertions` section listing each assertion with ✓/✗ and
   its severity.
4. CLI: `--assert "name:expression"` repeatable, where expression is a
   Python expression over `result` (e.g.
   `"no_pii_leak:not result.any_step(lambda s: 'ssn' in str(s.outputs))"`).
   Exit code 1 if any error-severity assertion fails — wires the
   report into CI gates.
5. Tests:
   * Assertion that passes → verdict PASS, exit 0.
   * Assertion that fails (severity error) → verdict FAIL, exit 1.
   * Executive-summary banner has the right cost numbers.

## Why this matters
Incident reports are useless if a human has to read the whole thing
to know whether action is needed. A PASS/FAIL banner + assertion
section turns the report into something a CI pipeline can gate on
and a manager can read in 5 seconds. It also formalises the
"counterfactual debugging" workflow: state your hypothesis as an
assertion, run the report, see if the substitution made it pass.
