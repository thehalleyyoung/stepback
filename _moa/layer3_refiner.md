# Layer 3 Refiner — final pinned design + concrete file plan

## What's locked in from Layer 2

L2 picked **Proposer 2's sandboxed `ast` walker** as the backbone,
adopted **Proposer 1's `PredicateSyntaxError(pos, src)` shape** and
its tight 4-builtin set (`len`, `str`, `any_step`, `all_step`),
and integrated **Proposer 3's quantifier-as-sugar idea** (so we
never let `GeneratorExp` reach the visitor). JSON IR, branch_io
persistence, and the wider builtin set are deferred per Rule 2.

This refiner pins implementation details and tightens scope so the
whole change fits one round + tests + e2e fixture coverage.

## One additional ground-truth fact discovered before pinning

Inspection of `stepback/replay.py:440` shows
`Trace.bisect(predicate=...)` calls the predicate with a
**`StepView`**, while `stepback/minimize.py:205` calls it with a
**`ReplayResult`**. The L2 plan treated only the latter. The
compiled predicate must work in both contexts.

**Resolution**: at call-time the compiled callable inspects the
argument:
- if it has `.steps` attribute → bind `result = arg` and unpack
  shorthands (`total_cost_usd`, `dirty_count`, `cache_hit_count`,
  `real_executions`, `steps`).
- elif it has `.step_id` and `.kind` → bind `step = arg` and unpack
  StepView field shorthands (`kind`, `name`, `outputs`, `inputs`,
  `cost_usd`, `cost`, `dirty`, `cache_hit`, `error_class`,
  `step_id`, `parent_step_id`).
- else: bind `value = arg` only.

Names not in the active context just resolve to `None` so a single
predicate can target both surfaces, e.g.
`compile_predicate("kind == 'tool_call' and 'GB99' in str(outputs)")`
works as a `bisect` predicate, and
`compile_predicate("total_cost_usd > 0.10")` works as a minimize
predicate, with no API split.

## Concrete file changes

### 1. `stepback/predicates.py` — extended (~340 LOC up from 76)

Existing combinators (`all_of`, `any_of`, `not_`, `xor_`) stay
unchanged. New additions, in this order in the file:

#### Public exception classes
```python
class PredicateSyntaxError(ValueError):
    def __init__(self, msg: str, *, src: str, pos: tuple[int, int] = (1, 0)):
        self.src, self.pos = src, pos
        line, col = pos
        excerpt = src.splitlines()[line - 1] if src.splitlines() else src
        caret = " " * col + "^"
        super().__init__(f"{msg}\n  at line {line}, col {col}:\n    {excerpt}\n    {caret}")

class PredicateRuntimeError(RuntimeError): ...
```

#### `parse_predicate(src) -> ast.Expression`
Wraps `ast.parse(src, mode="eval")`. Catches `SyntaxError` and
re-raises `PredicateSyntaxError` carrying the original
`(lineno, offset-1)`.

#### `_QuantifierRewriter(ast.NodeTransformer)`
Walks the parsed tree and rewrites top-level
`Call(func=Name(id="any_step"|"all_step"), args=[BODY])` into a
synthesised marker node `_Quantifier(kind, body)`. We use a
dataclass-based node held inside a `ast.Constant`-wrapped
`(kind, body)` payload — but cleaner: subclass `ast.expr` and
register the node type so the visitor can recognise it. **This
ensures `GeneratorExp` never appears in the validated tree.**

Rejects `any_step`/`all_step` with arity ≠ 1 with
`PredicateSyntaxError`.

#### `_SafeVisitor(ast.NodeVisitor)`
Allowlist (frozen):

| AST node | Allowed? |
| -------- | -------- |
| `Expression` | ✅ |
| `BoolOp` (`And`, `Or`) | ✅ |
| `UnaryOp` (`Not`, `USub`) | ✅ |
| `Compare` (`Eq` `NotEq` `Lt` `LtE` `Gt` `GtE` `In` `NotIn`) | ✅ |
| `BinOp` (`Add` `Sub` `Mult` `Div` `Mod`) | ✅ |
| `Constant` (str, int, float, bool, None only) | ✅ |
| `Name` / `Load` | ✅ |
| `Attribute` (attr name not starting `_`) | ✅ |
| `Subscript`, `Index`, `Slice` | ✅ |
| `Call` — only `Name` callees in `{len, str}`; `_Quantifier` exempt | ✅ |
| `List`, `Tuple` (literal length ≤ 1024) | ✅ |
| any other node (incl. `Lambda`, `GeneratorExp`, `Pow`, `Set`, `Dict`, `JoinedStr`, …) | ❌ |

`Compare` left/right operands restricted: `Pow` rejected by
not allowing `ast.Pow` operator.

`Call` callees that are `Attribute` (i.e. method calls like
`x.get(...)`) are explicitly rejected with a clear message:
"method calls are not allowed in predicate DSL".

#### `_Evaluator`
Compiles the validated tree by walking it and producing a Python
closure. Quantifier nodes become a Python `for step in steps:`
loop with `step` pushed onto a thread-local context stack. Name
lookups consult the stack top first, then the bound shorthands,
then `extra_names`, then return `None`. (No `NameError` — predicate
DSL is forgiving so cross-context predicates just don't fire.)

#### `compile_predicate(src, *, extra_names=None) -> Callable`
1. `tree = parse_predicate(src)`
2. `tree = _QuantifierRewriter().visit(tree); ast.fix_missing_locations(tree)`
3. `_SafeVisitor().visit(tree)` — raises on disallowed nodes
4. Build the evaluator closure
5. Wrap in `lambda v: bool(_run(v))`; attach `.source = src`,
   `.parsed = tree` for tooling

The wrapper coerces with `bool(...)` so predicate semantics are
strict booleans.

`functools.lru_cache(maxsize=128)` on `(src, frozenset(extra_names or {}))`.

#### `__all__` extended
```python
__all__ = [
    "all_of", "any_of", "not_", "xor_",
    "compile_predicate", "parse_predicate",
    "PredicateSyntaxError", "PredicateRuntimeError",
]
```

### 2. `tests/test_predicate_dsl.py` — NEW (~280 LOC)

Real coverage in 5 sections:

1. **Smoke / happy paths** — constants, arithmetic precedence,
   string `in`, `and/or/not`, `len(steps)`, attribute access,
   subscript on dicts, comparison chains.
2. **Quantifiers** — `any_step(kind == "llm_call")` and
   `all_step(cost_usd >= 0)` against a synthetic `ReplayResult`.
3. **Sandboxing denylist** — every escape vector listed in L2:
   `step.__class__`, `__import__('os')`, `lambda x: x`,
   `9**9**9`, `().__class__.__mro__`, `step.outputs.get('x')`,
   bare `import os`, `{1,2,3}`, `f"{x}"`, generators
   (`any(s for s in steps)`).
4. **Error UX** — `PredicateSyntaxError.pos` is `(line, col)`
   pointing at the offending token; `PredicateSyntaxError.src`
   round-trips; message contains a caret.
5. **End-to-end with real fixtures** (Rule 3a):
   - Build a real fixture trace using
     `tests.fixtures.agent.simulate_agent` (already used by other
     e2e tests in the suite).
   - Replay and apply
     `compile_predicate("any_step(kind == 'tool_call')")` via
     `result.any_step(...)`.
   - Use `Trace.bisect(good=..., bad=..., predicate=compile_predicate("kind == 'tool_call'"))`
     and assert it locates a real tool-call step (no mocks).
   - Assert the predicate's `.source` round-trips into `report.md`
     unchanged.

### 3. Update `stepback/predicates.py` module docstring
Replace the "A future round may add a safe predicate DSL string-
parser; for now…" sentence with a usage block pointing to the new
`compile_predicate` and listing the safe subset.

## LOC budget
- `stepback/predicates.py`: 76 → ~340 (+264)
- `tests/test_predicate_dsl.py`: 0 → ~280 (+280)
- Total: ~544 LOC, one coherent feature.

## Verification gates (must pass before exit)
```
pytest -x -q                                # all 449 + new tests green
pytest -x -q tests/test_predicate_dsl.py    # specifically green
```

## Citations to prior layers
- Backbone (visitor architecture, allowlist node set, sandboxed
  globals): **Proposer 2** §"Allowlisted AST node types",
  §"Walker structure".
- Quantifier-as-sugar to avoid GeneratorExp + extra_names hook:
  **Proposer 3** §"Operator table" (any_step/all_step) + L2's
  hardening rationale.
- Error class shape + tight builtin set + `bool(...)` coercion:
  **Proposer 1** §"Public API additions" + §"Tests".
- Cross-context (StepView vs ReplayResult) call-time binding: **L3
  refinement** (not in any L1 proposer; surfaced from
  `stepback/replay.py:440` reading).
