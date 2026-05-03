# Proposer 3 — Multi-witness search and budget controls

## Target module
`stepback/minimize.py`.

## Theme
Two pragmatic limits in today's API:

1. **Single witness.** ddmin returns one minimal subset. In practice
   multiple disjoint minimal subsets often exist (e.g. either fixing
   the lookup tool OR overriding the policy makes the predicate fire).
   Returning only the first-found witness hides alternatives the user
   should choose between.
2. **Unbounded probes.** The current loop has no probe budget, no
   timeout, no early-stop. A pathological predicate (slow LLM, dirty
   subtree) can hang.

## What to add

1. `find_all_minimal(trace, subs, predicate, *, max_witnesses=8,
   probe_budget=200, executor=None) -> list[MinimizationResult]`.
   - Runs ddmin, records the witness, then "blocks" that witness by
     pinning at least one of its members to `forbidden=True` and
     re-runs ddmin on the remaining set. Repeat until no more minimal
     subsets fit the budget or `max_witnesses` is reached.
   - Each returned `MinimizationResult` carries `witness_index: int`
     and references the same probe cache.
2. Budget controls in `MinimizeOptions` dataclass passed to ddmin:
   - `probe_budget: int | None` — abort with `BudgetExhausted` when
     exceeded; return best-effort partial result.
   - `time_budget_s: float | None` — wall-clock bound.
   - `progress: Callable[[int, int], None] | None` — called after each
     probe with `(probes_so_far, current_subset_size)` for live UI.
3. `BudgetExhausted(MinimizationResult)` exception subclass carrying
   the partial result so callers can salvage it.
4. **Predicate combinators** in a new sub-namespace
   `stepback.minimize.predicates`:
   - `all_of(*preds)`, `any_of(*preds)`, `not_(p)`.
   - `step_output_contains(step_id, key, needle)`.
   - `total_cost_exceeds(usd: float)`.
   - `step_called_tool(step_id, tool_name)`.
   - `policy_allowed(step_id) / policy_denied(step_id)`.
   These let the CLI accept high-level predicate DSL strings instead
   of raw Python `eval`.

## CLI
- `stepback minimize TRACE --all-witnesses --max-witnesses 4 …` emits
  a JSON list of witnesses.
- `--probe-budget 50` and `--time-budget 30s` flags.
- `--predicate-dsl 'step_output_contains(step-7, country, US)'` as a
  safer alternative to `--predicate '<python expr>'`.

## Tests
- `test_find_all_minimal_returns_two_disjoint_witnesses`: build a
  fixture where either ToolOutput@1 or PolicyOverride@9 alone makes
  the predicate fire; assert exactly two single-element witnesses.
- `test_probe_budget_raises_with_partial_result`: predicate that
  always returns False; budget=5 → `BudgetExhausted` with
  `partial.probes == 5`.
- `test_time_budget_terminates`: predicate sleeps 0.4s; budget 1s →
  terminates within 2s real time.
- `test_predicate_dsl_step_output_contains`: ensures the DSL
  evaluator parses correctly without `eval()` of arbitrary Python.
- `test_progress_callback_invoked`: list collects progress events;
  asserts called >= probes count.

## Why this framing
The two missing pieces today are *operational*: alternatives and
safety. Multi-witness reframes minimisation as enumerating the
equivalence class of root causes, which matches how engineers
actually triage. Budgets prevent the tool from being unusable on
slow / hanging predicates. The predicate DSL replaces an `eval()`
security hole in the CLI.

## Risk
The "block witness then re-run" loop is heuristic; not guaranteed to
enumerate ALL minimal subsets in pathological cases. Document this:
the function returns "up to k disjoint minimal witnesses found
within budget", not "all minimal subsets".

## LOC estimate
~300 LOC core, ~100 LOC predicate DSL, ~200 LOC tests, ~60 LOC CLI.
