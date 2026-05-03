# Proposer 3 — Orchestration: composites, validators, scenarios

## Theme
What's missing is not raw "more substitution kinds" but the
ability to *compose* substitutions into a named, reusable scenario
and to attach *invariants* the replay must satisfy at substitution
points. A debugger of agent runs is most useful when you can say:
*"here is the 'paranoid-vendor' scenario — 4 substitutions across
3 steps + 2 invariants — replay every nightly trace under it"*.

This proposer wraps a smaller set of new substitutions in a
*scenario* + *invariant* layer.

## New substitutions

1. **`SystemPromptSubstitution(at_step, system_text)`** — patch only
   the system message (most common ergonomic miss today).
2. **`SamplingSubstitution(at_step, temperature=None, top_p=None,
   max_tokens=None)`** — set sampling knobs in inputs.
3. **`RaiseSubstitution(at_step, exception_type, message)`** —
   force the step to raise. Stored as
   `{"__error__": {"type":..., "message":...}}` output.
4. **`CompositeSubstitution(at_step, children)`** — bundle multiple
   subs that target the same step into one named unit. Apply order
   = list order. Round-trips by recursing in `substitution_to_dict`.
5. **`AssertSubstitution(at_step, predicate)`** — `predicate` is a
   small JSON-DSL expression (`{"op":"==","path":"/model","value":
   "gpt-4o-mini-2024-07-18"}`) evaluated against `inputs` *after*
   substitutions and *before* the step runs. On failure raises
   `InvariantViolation` so the replay loudly stops. Lets you write:
   *"if I'm replaying with `policy=paranoid`, assert step:7's model
   is the cheap one"*.

## Scenario object

* New module `stepback/scenarios.py` (~80 LoC):

  ```python
  @dataclass
  class Scenario:
      name: str
      description: str = ""
      substitutions: List[Substitution] = field(default_factory=list)
      invariants: List[Substitution] = field(default_factory=list)

      def applied_to(self, trace: Trace) -> SubstitutionSet:
          subs = SubstitutionSet()
          for s in self.substitutions: subs.add(s)
          for s in self.invariants:    subs.add(s)
          return subs

      def to_json(self) -> dict: ...
      @classmethod
      def from_json(cls, d: dict) -> "Scenario": ...
  ```

* `.sbs` ("stepback scenario") files: JSON, list of scenarios.
  Authored once, applied to many traces (not pinned to a trace
  hash, unlike `.sbb`).

## CLI

* `stepback replay TRACE --scenario scenarios/paranoid.sbs`
* `stepback diff TRACE --scenario A.sbs --scenario B.sbs`
* New parser verbs:
  - `system@step:N=...`
  - `sampling@step:N=temperature=0.0,max_tokens=256`
  - `raise@step:N=TimeoutError:timed out`
  - `assert@step:N=:inline:{"op":"==","path":"/model","value":"..."}`

## Tests

* Unit: scenarios round-trip, composite apply order, predicate DSL.
* E2E: load a `.sbs` with 4 subs + 2 asserts, replay against a
  fixture trace, assert the dirty-set / cost / one InvariantViolation.

## Risks

* The scenario layer adds a second persistence format on top of
  `.sbb`; users may be confused about when to use which (`.sbb` =
  pinned to trace; `.sbs` = portable across traces).
* The mini-DSL for predicates is yet another grammar — could just
  use Python `eval` over the dict, but that's a security cliff.
* `CompositeSubstitution` introduces nesting; `substitution_to_dict`
  has to recurse. Worth it for batching.
* Doesn't add a *generic* input mutator (no JSON Patch) — relies on
  having the right typed substitution for the question.
