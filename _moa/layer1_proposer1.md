# Proposer 1 — Hand-rolled recursive-descent parser

## Target module
`stepback/predicates.py` — currently exposes only 4 combinators
(`all_of`, `any_of`, `not_`, `xor_`). Module docstring already
admits "A future round may add a safe predicate DSL string-parser."
That's the gap to fill.

## Framing
Implement a small **string-DSL** so users can write predicates as
plain text and pass the compiled `Callable[[ReplayResult], bool]`
to `Trace.bisect`, `find_minimal`, etc.

```python
from stepback.predicates import compile_predicate

p = compile_predicate(
    'total_cost_usd > 0.10 and any(step.kind == "tool_call" '
    'and "GB99" in str(step.outputs))'
)
trace.bisect(predicate=p)
```

## Architecture: hand-written recursive descent

Three layers — tokenizer, parser, evaluator — no third-party deps,
no `eval`, no `compile`. ~200 LOC.

### Tokenizer
Regex-based scanner producing `Token(kind, value, pos)`. Tokens:
- literals: `NUMBER`, `STRING` (single/double quotes, `\` escapes),
  `TRUE`, `FALSE`, `NONE`
- identifiers: `IDENT`
- operators: `==`, `!=`, `<`, `<=`, `>`, `>=`, `+`, `-`, `*`, `/`,
  `%`, `(`, `)`, `[`, `]`, `,`, `.`
- keywords: `and`, `or`, `not`, `in`, `any`, `all`, `len`,
  `step` (the per-step iterator var)

### Grammar (pseudo-EBNF)
```
expr     := or_expr
or_expr  := and_expr ('or' and_expr)*
and_expr := not_expr ('and' not_expr)*
not_expr := 'not' not_expr | comp
comp     := add (('==' | '!=' | '<' | '<=' | '>' | '>=' | 'in') add)?
add      := mul (('+' | '-') mul)*
mul      := unary (('*' | '/' | '%') unary)*
unary    := '-' unary | postfix
postfix  := primary ('.' IDENT | '[' expr ']' | '(' args? ')')*
primary  := NUMBER | STRING | TRUE | FALSE | NONE
          | 'any' '(' expr ')'   # iterates over result.steps as 'step'
          | 'all' '(' expr ')'
          | 'len' '(' expr ')'
          | 'str' '(' expr ')'
          | IDENT                # resolved against context
          | '(' expr ')'
```

### AST nodes
Plain dataclasses: `BinOp(op,l,r)`, `UnaryOp(op,x)`, `Compare(op,l,r)`,
`MemberAccess(obj,attr)`, `Index(obj,key)`, `Call(name,args)`,
`Literal(value)`, `Identifier(name)`, `Quantifier(kind,body)`.

### Evaluator
`eval_node(node, ctx)` where `ctx` is a chain-map dict containing
the top-level binding `result -> ReplayResult` plus convenience
short-cuts: `total_cost_usd`, `dirty_count`, `cache_hit_count`,
`real_executions`, `steps`. Inside `any(...)`/`all(...)` the
evaluator pushes a frame `{step: StepView}` per iteration over
`result.steps`.

`MemberAccess` allowlists attributes: only public dataclass fields
of `StepView` / `ReplayResult` plus the documented `cost` and
`error_class` properties (already present). Anything else raises
`PredicateError("attribute 'X' not allowed in DSL")`.

`Call` only resolves to the four whitelisted built-ins (`len`,
`str`, `any`, `all`). No arbitrary callables.

## Public API additions
```python
def compile_predicate(src: str) -> Callable[[Any], bool]
class PredicateSyntaxError(ValueError): pos: int; src: str
class PredicateRuntimeError(RuntimeError): ...
```

`__all__` adds `compile_predicate`, `PredicateSyntaxError`,
`PredicateRuntimeError`.

## Why hand-rolled (not Python `ast`)
Python's `ast` module is appealing but its grammar surface is huge
— anyone who later swaps `ast.parse` for the underlying compiler
opens an injection vector. A hand-written grammar is auditable in
one screen.

## Tests
`tests/test_predicate_dsl.py`:
- happy paths: cost compare, `any(step.kind == "llm_call")`,
  `len(steps) > 5`, string `in`, nested booleans
- syntax errors with `pos` pointing at offending token
- attribute denylist: `compile_predicate("step.__class__")` raises
- builtin denylist: `compile_predicate("open('/etc/passwd')")` raises
- end-to-end: `Trace.bisect(predicate=compile_predicate(...))` finds
  the right step on the existing fixture trace

## Strengths / weaknesses
- + Zero deps, ~200 LOC, fully sandboxed by construction.
- + Easy to extend (add a node type, add a parse rule).
- − Re-implements the wheel; no operator precedence climbing helper.
- − Error messages are only as good as we make them.
