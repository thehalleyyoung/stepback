# Proposer 2 — Sandboxed `ast` walker

## Target module
`stepback/predicates.py`. Same opportunity: ship a string DSL.

## Framing
Re-use Python's own parser (`ast.parse(src, mode="eval")`), then
walk the resulting tree through a strict **allowlist visitor**
that rejects everything not on the safe list. Compile the surviving
tree with `compile(tree, "<predicate>", "eval")` and execute against
a frozen `{"__builtins__": {}}` globals dict. ~150 LOC.

This is the same pattern Django's template-tag `Variable` resolver,
asteval, and `simpleeval` use; but we keep it tiny and tailored.

## Allowlisted AST node types
```
Module, Expression,
BoolOp(And/Or), UnaryOp(Not/USub),
Compare(Eq/NotEq/Lt/LtE/Gt/GtE/In/NotIn),
BinOp(Add/Sub/Mult/Div/Mod),
Constant, Name, Load,
Attribute, Subscript, Index, Slice,
Call (function name only — see below),
GeneratorExp, comprehension, ListComp,
List, Tuple
```
Forbidden (raises `PredicateSyntaxError` from the visitor):
`Lambda`, `FunctionDef`, `Import`, `ImportFrom`, `Assign`,
`AugAssign`, `AnnAssign`, `Global`, `Nonlocal`, `Yield`,
`Await`, `Try`, `For`, `While`, `If`, `With`, `Raise`,
`Delete`, dunder attribute access (`Attribute.attr.startswith("_")`),
star-args, `f-string` formatting (`JoinedStr`), `Set`, `Dict`.

## Allowed names
- `result` → the `ReplayResult`
- shorthand fields: `total_cost_usd`, `dirty_count`,
  `cache_hit_count`, `real_executions`, `steps`
- `step` (only inside a generator/list comp iterating `steps`)
- builtins: `len`, `str`, `int`, `float`, `bool`, `any`, `all`,
  `min`, `max`, `sum`, `abs`

`Call` requires the callee to be a `Name` whose id is in the
builtin allowlist — no method calls, no chained `getattr`. (The
strictness here matters: allowing `Attribute` callees lets users
reach `''.__class__.__mro__[1].__subclasses__()` style escapes.)

## Public API
```python
def compile_predicate(src: str, *, extra_names: dict | None = None) -> Callable[[Any], bool]
def parse_predicate(src: str) -> ast.Expression   # exposed for IDEs/REPLs
class PredicateSyntaxError(ValueError): ...
```

`extra_names` lets advanced callers inject named references (e.g.
a regex they precompiled) without weakening the global denylist.

## Walker structure
```
class _SafeVisitor(ast.NodeVisitor):
    ALLOWED = { ... node-class set ... }
    BUILTINS = { "len": len, ... }

    def visit(self, node):
        cls = type(node)
        if cls not in self.ALLOWED:
            raise PredicateSyntaxError(f"{cls.__name__} not allowed",
                                       lineno=getattr(node,"lineno",1),
                                       col=getattr(node,"col_offset",0))
        if cls is ast.Attribute and node.attr.startswith("_"):
            raise PredicateSyntaxError("dunder attribute access blocked")
        if cls is ast.Call and not isinstance(node.func, ast.Name):
            raise PredicateSyntaxError("only bare-name calls allowed")
        if cls is ast.Call and node.func.id not in self.BUILTINS:
            raise PredicateSyntaxError(f"function '{node.func.id}' not allowed")
        self.generic_visit(node)
```

After `_SafeVisitor` validates, we still defend at runtime by
running with `{"__builtins__": {}}` as `globals` and
`{**BUILTINS, "result": result, ...shorthands}` as `locals`.

## Why this beats hand-rolled
- Re-uses CPython's battle-tested parser → exact Python operator
  precedence semantics, exact f-string-style error positions
  (`SyntaxError.lineno`/`offset`).
- ~150 LOC vs ~400 LOC. Smaller attack surface in our code, even
  though we depend on a bigger underlying parser.
- Users already know Python; no DSL learning curve.

## Risks & mitigations
- The classic escape vector is `Attribute` access on string/dict
  literals reaching dunder method tables. Mitigated by:
  (a) blocking any attr starting with `_`, and (b) blocking
  attribute-call (only `Name` callees allowed).
- `**` (`Pow`) is excluded — a 1-char input can DoS via
  `9**9**9`. Excluded from BinOp allowlist.
- `[i]` is allowed but step-bounds are validated by Python; we
  limit list/tuple literal length to 1024 in the visitor.
- Compiled bytecode is cached via `functools.lru_cache` on `src`.

## Tests (mirror Proposer 1 plus)
- explicit denylist tests for `__class__`, `().__class__`,
  `lambda x: x`, `__import__("os")`, `(1).bit_length`
- `9**9**9` rejected
- comprehension scoping: `step` only resolves inside the comp
- `extra_names` round-trips a precompiled regex

## Strengths / weaknesses
- + Tiny + leverages CPython semantics; users get full Python
  expressivity in the safe subset.
- + Familiar grammar (no DSL manual to write).
- − Allowlist drift is a real risk: a future Python version may
  add a node type that bypasses the visitor. CI must pin
  `sys.version_info` and re-validate.
- − Less control over error UX than a hand-rolled parser.
