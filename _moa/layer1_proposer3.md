# Proposer 3 — JSONLogic-style structured predicates + parser

## Target module
`stepback/predicates.py`. Same gap (no string DSL).

## Framing
Rather than a textual DSL, expose predicates as **JSON-serialisable
trees** so they can be saved into `.sb` traces, attached to bisect
runs, diffed across users, and round-tripped through a CLI.

```python
from stepback.predicates import from_json, to_json, compile_predicate

p_json = {
    "and": [
        {">":  [{"var": "total_cost_usd"}, 0.10]},
        {"any_step": {
            "and": [
                {"==": [{"var": "step.kind"}, "tool_call"]},
                {"in": ["GB99", {"call": ["str", {"var": "step.outputs"}]}]},
            ],
        }},
    ],
}
p = from_json(p_json)              # → Callable
src = "total_cost_usd > 0.10 and any(step.kind == 'tool_call' " \
      "and 'GB99' in str(step.outputs))"
p2 = compile_predicate(src)        # textual surface → same JSON tree
assert to_json(p2) == p_json       # canonical form
```

## Why JSON-first

stepback's whole pitch is "every artefact in the debugger is
inspectable, signed, and replayable." A predicate used to bisect a
production trace is itself an artefact. JSON-first means:

1. The predicate that found a regression goes into the
   `report.md` verbatim.
2. CI can ship a library of predicates as YAML files in repo.
3. Two engineers comparing branches can diff predicates structurally,
   not via string-equal.
4. The textual DSL is just sugar — it lowers to JSON.

## Architecture

### Two layers
- `from_json(node) -> Callable`: walks a JSON dict, returns a
  Python callable. Each operator is a registered handler in a
  small dispatch table.
- `compile_predicate(src) -> Callable`: tokenizes + Pratt-parses
  the textual surface into the **same JSON tree**, then calls
  `from_json`. So all sandboxing logic lives in one place
  (`from_json`) and the textual surface gains nothing beyond
  ergonomic.

### Operator table
```
arith:    + - * / %                  (binary, numeric)
compare:  == != < <= > >= in
boolean:  and or not                 (variadic and/or, unary not)
access:   var (dotted path resolved against ctx),
          item (subscript)
quant:    any_step, all_step         (body evaluated per step)
calls:    call (name + args, name in {len,str,abs,min,max,sum})
literals: bare JSON scalars (int/float/str/bool/null)
```

### `var` resolution
`{"var": "step.outputs.error_class"}` walks dotted path against
the active context. Names bound by default:
- `total_cost_usd`, `dirty_count`, `cache_hit_count`,
  `real_executions`, `steps`
- `result` (full ReplayResult)
- `step` (only inside `any_step`/`all_step` bodies)

Dotted access uses `getattr` for objects, `[]` for dicts, and is
**limited to attribute names not starting with `_`**.

### Pratt parser for the textual surface
Pratt parsing handles the precedence ladder
(`or` < `and` < `not` < compare < `in` < `+/-` < `*/%`) in ~80
LOC with a single token loop and a `bp` (binding-power) table.
It's strictly more compact than recursive-descent for an
expression-only grammar.

## Public API
```python
def from_json(spec: dict | list | str | int | float | bool | None) -> Callable
def to_json(predicate: Callable) -> dict        # only round-trips DSL-built predicates
def compile_predicate(src: str) -> Callable     # sugar: parse → from_json
def register_op(name: str, arity: int, fn: Callable) -> None  # extension hook
class PredicateError(ValueError): ...
```

`register_op` lets advanced users wire in a custom op — but only
*before* `compile_predicate` is called for that op name; defensive
double-registration raises.

## Persistence integration
Add a tiny piece in `stepback/branch_io.py`: when a `Branch` is
serialised it includes the JSON predicate that was used to discover
it (if any), in a new `discovery_predicate` field. This is the
artefact-as-evidence story the README talks about.

## Tests
`tests/test_predicate_dsl.py`:
- JSON round-trip for ~10 expressions
- text → JSON canonicalisation (textual `1 + 2 * 3` → JSON tree
  with correct precedence)
- denylist: `{"var": "step.__class__"}` raises
- extension: `register_op("regex_match", 2, ...)` works once and
  errors on re-registration
- end-to-end: predicate used by `Trace.bisect`, then serialised
  into a `Branch` and re-loaded

## Strengths / weaknesses
- + Predicates are *data*: storable, diffable, signable, shareable.
- + Single sandboxing chokepoint (`from_json`); textual parser is
  pure sugar that can't reach the runtime directly.
- + Pratt parser is the smallest correct expression parser known.
- − JSON form is verbose for humans; expect everyone to prefer the
  textual surface in REPLs.
- − Two surfaces means two mental models for newcomers.
