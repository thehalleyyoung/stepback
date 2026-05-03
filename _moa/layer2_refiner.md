# Layer 2 Refiner — pick a backbone, integrate the best of the rest

## Choice of backbone: **Proposer 2 (sandboxed `ast` walker)**

Rationale:
- Smallest code surface (~150 LOC); leverages CPython's parser.
- Users get familiar Python operator semantics for free
  (precedence, short-circuit, `in`, comprehensions).
- The whole sandboxing story collapses to *one* allowlist visitor
  whose correctness is locally verifiable.

Proposer 1's hand-rolled grammar duplicates work CPython already
does and would expand the diff to ~400 LOC for no semantic win.
Proposer 3's JSON-first design is *architecturally* nice but
ships two surfaces in one round, which violates "one coherent
thematic improvement."

## What we take from Proposer 1

- **Tokenizer-style structured error reporting**: P2's plan
  surfaces `ast.SyntaxError` directly, which is fine for parse
  errors but unhelpful for *visitor* rejections (e.g.
  `step.__class__`). Adopt P1's `PredicateSyntaxError(pos, src)`
  shape so the message points exactly at the offending node using
  `node.lineno` / `node.col_offset`, with a 1-line caret excerpt
  (like Python ≥ 3.11's traceback hints).
- **The four built-in helpers**: P1 names `len`, `str`, `any`,
  `all` as the minimum useful set. Adopt that exact set as the
  *initial* `BUILTINS` in the visitor. P2's wider list (`min`,
  `max`, `sum`, `abs`, …) can come in a follow-up; smaller is
  safer for round-1.
- **`compile_predicate` returning a `Callable[[Any], bool]`** that
  always coerces with `bool(...)` — predicate semantics, not
  truthy-Python semantics. Borrowed from P1.

## What we take from Proposer 3

- **`steps` quantifiers as first-class concepts.** P2's plan uses
  Python comprehensions (`any(s.kind == "llm_call" for s in steps)`),
  which works but commits us to allowing `GeneratorExp` and
  `comprehension` AST nodes — a meaningful expansion of the
  attack surface (custom iterators, scoping subtleties).

  Replace generators with two **sugar functions** `any_step(expr)`
  / `all_step(expr)` resolved at *parse time* by a small
  AST-rewrite step before the `_SafeVisitor` runs:

  ```
  any_step(<EXPR>)   -->   any(<EXPR'> for step in steps)
  ```

  …except the rewrite produces an internal `_StepQuantifier`
  custom node (an `ast.Call` we synthesise) that the evaluator
  handles by iterating `result.steps` itself, with `step` bound
  in a per-iteration locals frame. **No `GeneratorExp` ever
  reaches the visitor's allowlist.** This is the single biggest
  hardening win across the three proposals.

- **Allow lifting predicates back to a structured form** via a
  `predicate.source` attribute on the returned callable. Round-1
  doesn't need full JSON round-trip, but stash the original src
  string + parsed AST dump so `report.md` can quote it. (Defer
  full JSON to a future round, per Rule 2.)

- **`extra_names` injection** (P2's idea, kept) plus P3's
  defensive **double-registration check** to prevent silent
  shadowing of built-in names.

## What we drop

- P1's full hand-rolled grammar (replaced by `ast`-based parsing).
- P3's JSON IR + `from_json/to_json` API — defer.
- P3's `branch_io` integration — defer.
- P2's `min/max/sum/abs/int/float/bool` builtins — defer; only
  `len`, `str`, `any_step`, `all_step` ship round-1.
- P2's allowance of `Pow` — kept dropped (DoS via `9**9**9`).
- P2's `Set`/`Dict` literals — kept dropped (no use case).

## Final API (frozen for L3)

```python
def compile_predicate(
    src: str,
    *,
    extra_names: dict[str, object] | None = None,
) -> Callable[[Any], bool]      # the callable has .source : str

def parse_predicate(src: str) -> ast.Expression   # exposed for tooling

class PredicateSyntaxError(ValueError):
    src: str
    pos: tuple[int, int]   # (lineno, col_offset)

class PredicateRuntimeError(RuntimeError): ...
```

`__all__` adds `compile_predicate`, `parse_predicate`,
`PredicateSyntaxError`, `PredicateRuntimeError`.

## Allowlist (frozen)

Nodes: `Expression`, `BoolOp(And/Or)`, `UnaryOp(Not/USub)`,
`Compare(Eq/NotEq/Lt/LtE/Gt/GtE/In/NotIn)`,
`BinOp(Add/Sub/Mult/Div/Mod)`, `Constant`, `Name`, `Load`,
`Attribute` (no leading `_`), `Subscript`, `Index`/`Slice`,
`Call` (only allowlisted bare-name callees),
`List`, `Tuple` (literal length ≤ 1024, enforced by visitor).

Built-ins: `len`, `str`, `any_step`, `all_step` (the last two
synthesised by the AST rewrite, not present in user globals).

Names auto-bound to evaluator locals: `result`, `total_cost_usd`,
`dirty_count`, `cache_hit_count`, `real_executions`, `steps`,
plus any keys from `extra_names`. `step` is bound only inside
`any_step`/`all_step` bodies.

Globals at exec time: `{"__builtins__": {}}`.

## Tests promised for L3

`tests/test_predicate_dsl.py` covering:
- happy paths (cost, kind, string `in`, nested `and`/`or`,
  `len(steps) > 5`)
- `any_step` / `all_step` quantifiers
- denylist: `step.__class__`, `__import__('os')`,
  `lambda x: x`, `9**9**9`, `().__class__.__mro__`,
  `step.outputs.get('x')` (method call rejected)
- syntax error: `pos` points at offending token
- `extra_names` round-trip
- end-to-end: `Trace.bisect(predicate=compile_predicate(...))`
  on `tests/fixtures/...` finds the right step
- `predicate.source == src`

L3 will lock the file plan + LOC budget.
