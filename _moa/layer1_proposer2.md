# Proposer 2 — Causal attribution + first-divergence pinpoint in `stepback/report.py`

## Framing
The current report tells you *which* steps are dirty and *which*
diverged, but not *why*. In a 200-step trace with 4 substitutions an
incident author has to manually correlate the dirty subtree against
the substitution list. Bake that correlation into the report.

## Concrete plan
1. New helper `_attribute_dirty_steps(steps, subs) -> dict[step_id, list[sub_index]]`:
   * For each substitution in `subs.items`, find its `at_step` and
     mark every descendant in the trace topology as caused by it.
     (Use `parent_step_id` chains from `StepView`; the engine already
     propagates dirtiness, we just need to walk the parent edges to
     attribute *which* substitution(s) reach a given step.)
2. New helper `_first_divergence(baseline_steps, counterfactual_steps) -> Optional[StepView]`:
   * The first step (in topological order) whose `outputs` hash
     differs between the two replays. This is the "regression
     introduction site" — the headline finding of any incident
     write-up.
3. New section `## Causal attribution` listing each substitution and
   the dirty step IDs it explains, with an "unattributed dirty steps"
   bucket for anything not reachable (which usually indicates a bug
   in the dependency model — also valuable to surface).
4. New banner immediately under the H1: `**First divergence:**
   step:N (kind=…) — outputs changed by substitution(s) [#0, #2]`.
5. Tests: with 1 ToolOutputSubstitution at step:2 in the payments
   fixture, assert that all dirty steps are attributed to substitution
   index 0 and the first-divergence step is `step:2`.

## Why this matters
Counterfactual debugging is a *causal* task. "Why did total cost go
up?" should be answerable from the report alone, not by re-reading
the trace. Pinning the first divergence step + attributing each dirty
descendant to its causing substitution converts the report from
"diff" to "explanation".
