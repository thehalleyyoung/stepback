"""Composable predicate helpers + a sandboxed string-DSL for
:mod:`stepback.minimize` and :mod:`stepback.replay`.

A *predicate* is any ``Callable[[X], bool]`` accepted by
``Trace.bisect`` (where ``X = StepView``) or ``find_minimal``
(where ``X = ReplayResult``).

Two ways to build one:

1. The original combinator helpers (``all_of``, ``any_of``, ``not_``,
   ``xor_``) which compose plain Python callables.

2. The string DSL (``compile_predicate``) which compiles a small
   sandboxed subset of Python expression syntax to a callable.

DSL usage::

    from stepback.predicates import compile_predicate

    p = compile_predicate(
        "total_cost_usd > 0.10 and "
        "any_step(kind == 'tool_call' and 'GB99' in str(outputs))"
    )
    trace.bisect(good="step:1", bad="step:12",
                 predicate=compile_predicate(
                     "kind == 'tool_call' and 'GB99' in str(outputs)"))

DSL safe subset (frozen):

* Booleans:     ``and``, ``or``, ``not``
* Comparisons:  ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``,
                ``in``, ``not in``
* Arithmetic:   ``+``, ``-``, ``*``, ``/``, ``%`` (NO ``**``)
* Literals:     ints, floats, strings, ``True``, ``False``, ``None``
* Containers:   list / tuple literals (≤ 1024 items)
* Names:        ``result``, ``step``, ``steps``, ``total_cost_usd``,
                ``dirty_count``, ``cache_hit_count``,
                ``real_executions``, plus per-StepView shorthands
                (``kind``, ``name``, ``outputs``, ``inputs``,
                ``cost_usd``, ``cost``, ``dirty``, ``cache_hit``,
                ``error_class``, ``step_id``, ``parent_step_id``)
                and any keys in ``extra_names``.
* Attribute:    ``x.y`` (attribute names starting with ``_`` are
                rejected)
* Subscript:    ``x[k]`` (no slice steps)
* Calls:        ``len(...)``, ``str(...)``, ``any_step(...)``,
                ``all_step(...)``. Method calls (``x.y(...)``) are
                rejected.

``any_step`` / ``all_step`` are sugar — they bind ``step`` over
``result.steps`` and evaluate the body. The textual surface never
admits ``GeneratorExp`` or ``Lambda``.

Names that don't resolve in the active context evaluate to ``None``
rather than raising — so a single predicate string may target both
``StepView`` (via ``bisect``) and ``ReplayResult`` (via
``find_minimal``).
"""
from __future__ import annotations

import ast
import functools
from typing import Any, Callable, Dict, Optional

__all__ = [
    "all_of",
    "any_of",
    "not_",
    "xor_",
    "compile_predicate",
    "parse_predicate",
    "PredicateSyntaxError",
    "PredicateRuntimeError",
]


# ----------------------------------------------------------- combinators


def all_of(*predicates: Callable[[Any], bool]) -> Callable[[Any], bool]:
    """Return a predicate that fires iff *all* ``predicates`` fire.

    Short-circuits on the first ``False``. With zero predicates,
    returns a predicate that is always ``True`` (the empty conjunction).
    """
    preds = list(predicates)

    def _combined(value: Any) -> bool:
        for p in preds:
            if not p(value):
                return False
        return True

    return _combined


def any_of(*predicates: Callable[[Any], bool]) -> Callable[[Any], bool]:
    """Return a predicate that fires iff *any* ``predicates`` fire.

    Short-circuits on the first ``True``. With zero predicates,
    returns a predicate that is always ``False`` (the empty disjunction).
    """
    preds = list(predicates)

    def _combined(value: Any) -> bool:
        for p in preds:
            if p(value):
                return True
        return False

    return _combined


def not_(predicate: Callable[[Any], bool]) -> Callable[[Any], bool]:
    """Return the negation of ``predicate``."""

    def _negated(value: Any) -> bool:
        return not predicate(value)

    return _negated


def xor_(a: Callable[[Any], bool], b: Callable[[Any], bool]) -> Callable[[Any], bool]:
    """Return a predicate that fires iff exactly one of ``a`` or ``b`` fires."""

    def _xor(value: Any) -> bool:
        return bool(a(value)) != bool(b(value))

    return _xor


# ------------------------------------------------------------- DSL: errors


class PredicateSyntaxError(ValueError):
    """Raised when DSL source fails the parser or the safe-subset visitor.

    Attributes:
        src: the original DSL source string.
        pos: a ``(lineno, col_offset)`` tuple pointing at the offending
            token. Lines are 1-indexed; columns 0-indexed (matching
            ``ast``).
    """

    def __init__(self, msg: str, *, src: str, pos: tuple = (1, 0)) -> None:
        self.src = src
        self.pos = pos
        line, col = pos
        lines = src.splitlines() or [src]
        line_idx = max(0, min(len(lines) - 1, line - 1))
        excerpt = lines[line_idx]
        caret = " " * max(0, col) + "^"
        super().__init__(
            f"{msg}\n  at line {line}, col {col}:\n    {excerpt}\n    {caret}"
        )


class PredicateRuntimeError(RuntimeError):
    """Raised when an otherwise-valid DSL predicate explodes at eval time."""


# ------------------------------------------------------------- DSL: parse


_QUANT_NAMES = ("any_step", "all_step")
_BUILTIN_NAMES = ("len", "str")  # plus the two quantifiers, handled specially
_MAX_LITERAL_LEN = 1024
_STEPVIEW_SHORTCUTS = (
    "kind",
    "name",
    "outputs",
    "inputs",
    "cost_usd",
    "cost",
    "dirty",
    "cache_hit",
    "error_class",
    "step_id",
    "parent_step_id",
)
_RESULT_SHORTCUTS = (
    "total_cost_usd",
    "dirty_count",
    "cache_hit_count",
    "real_executions",
    "steps",
)


def parse_predicate(src: str) -> ast.Expression:
    """Parse ``src`` into an ``ast.Expression`` tree.

    Raises ``PredicateSyntaxError`` on parse failure.
    """
    try:
        return ast.parse(src, mode="eval")
    except SyntaxError as e:  # pragma: no cover - exercised in tests
        line = e.lineno or 1
        # ast.SyntaxError offset is 1-indexed; our pos is 0-indexed col.
        col = max(0, (e.offset or 1) - 1)
        raise PredicateSyntaxError(
            f"could not parse predicate: {e.msg}", src=src, pos=(line, col)
        ) from None


# ------------------------------------------------------ DSL: AST rewrite


class _Quantifier(ast.expr):
    """Synthesised AST node representing ``any_step(BODY)`` / ``all_step(BODY)``.

    Holds the unevaluated body. Created by :class:`_QuantifierRewriter`
    so that ``GeneratorExp`` never enters the validated tree.
    """

    _fields = ("kind", "body")
    _attributes = ("lineno", "col_offset", "end_lineno", "end_col_offset")

    def __init__(self, kind: str, body: ast.expr, **kw: Any) -> None:
        super().__init__(**kw)
        self.kind = kind
        self.body = body


class _QuantifierRewriter(ast.NodeTransformer):
    def __init__(self, src: str) -> None:
        self.src = src

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in _QUANT_NAMES
        ):
            if len(node.args) != 1 or node.keywords:
                raise PredicateSyntaxError(
                    f"{node.func.id}(...) takes exactly one positional argument",
                    src=self.src,
                    pos=(node.lineno, node.col_offset),
                )
            body = self.generic_visit(node.args[0]) if isinstance(
                node.args[0], (ast.expr,)
            ) else node.args[0]
            q = _Quantifier(
                kind=node.func.id,
                body=body if isinstance(body, ast.expr) else node.args[0],
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=getattr(node, "end_lineno", node.lineno),
                end_col_offset=getattr(node, "end_col_offset", node.col_offset),
            )
            return q
        self.generic_visit(node)
        return node


# --------------------------------------------------- DSL: safe-subset visitor


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod)
_ALLOWED_UNARY = (ast.Not, ast.USub)
_ALLOWED_BOOL = (ast.And, ast.Or)
_ALLOWED_CMP = (
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn,
)
_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp, ast.UnaryOp, ast.Compare, ast.BinOp,
    ast.Constant, ast.Name, ast.Load,
    ast.Attribute, ast.Subscript,
    ast.List, ast.Tuple,
    ast.Call,
    _Quantifier,
)
# `ast.Index` was removed in Python 3.9 — subscript values are now
# expressions directly. Slices use ast.Slice, which we forbid (no
# slicing in predicate DSL).


class _SafeVisitor(ast.NodeVisitor):
    def __init__(self, src: str) -> None:
        self.src = src

    def _err(self, msg: str, node: ast.AST) -> "PredicateSyntaxError":
        return PredicateSyntaxError(
            msg,
            src=self.src,
            pos=(getattr(node, "lineno", 1), getattr(node, "col_offset", 0)),
        )

    def visit(self, node: ast.AST) -> None:
        cls = type(node)
        if cls is _Quantifier:
            self.visit(node.body)  # type: ignore[attr-defined]
            return
        # Operator marker nodes (ast.Add, ast.Eq, ast.And, ...) are
        # validated at their parent's level; skip them here.
        if isinstance(
            node, (ast.operator, ast.boolop, ast.unaryop, ast.cmpop, ast.expr_context)
        ):
            return
        if not isinstance(node, _ALLOWED_NODES):
            raise self._err(
                f"{cls.__name__} not allowed in predicate DSL", node
            )
        # Per-node tightening.
        if isinstance(node, ast.BinOp) and not isinstance(node.op, _ALLOWED_BINOPS):
            raise self._err(
                f"binary operator {type(node.op).__name__} not allowed", node
            )
        if isinstance(node, ast.UnaryOp) and not isinstance(node.op, _ALLOWED_UNARY):
            raise self._err(
                f"unary operator {type(node.op).__name__} not allowed", node
            )
        if isinstance(node, ast.BoolOp) and not isinstance(node.op, _ALLOWED_BOOL):
            raise self._err(
                f"boolean operator {type(node.op).__name__} not allowed", node
            )
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if not isinstance(op, _ALLOWED_CMP):
                    raise self._err(
                        f"comparison operator {type(op).__name__} not allowed",
                        node,
                    )
        if isinstance(node, ast.Constant) and not isinstance(
            node.value, (str, int, float, bool, type(None))
        ):
            raise self._err(
                f"literal of type {type(node.value).__name__} not allowed",
                node,
            )
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise self._err(
                    f"attribute '{node.attr}' starts with '_' (denied)",
                    node,
                )
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Slice):
                raise self._err("slice indexing not allowed", node)
        if isinstance(node, (ast.List, ast.Tuple)):
            if len(node.elts) > _MAX_LITERAL_LEN:
                raise self._err(
                    f"literal with {len(node.elts)} elements exceeds "
                    f"limit {_MAX_LITERAL_LEN}",
                    node,
                )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise self._err(
                    "method calls are not allowed in predicate DSL "
                    "(only bare-name builtins)",
                    node,
                )
            if node.func.id not in _BUILTIN_NAMES + _QUANT_NAMES:
                raise self._err(
                    f"function '{node.func.id}' not allowed",
                    node,
                )
            if node.keywords:
                raise self._err("keyword arguments not allowed", node)
            for a in node.args:
                if isinstance(a, ast.Starred):
                    raise self._err("star-args not allowed", node)
        # Recurse.
        for child in ast.iter_child_nodes(node):
            self.visit(child)


# ----------------------------------------------------------- DSL: eval


_BUILTINS_DISPATCH = {"len": len, "str": str}


def _resolve_name(name: str, scope_stack: list, default_ctx: dict) -> Any:
    for frame in reversed(scope_stack):
        if name in frame:
            return frame[name]
    if name in default_ctx:
        return default_ctx[name]
    return None


def _eval(node: ast.AST, scope_stack: list, default_ctx: dict) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body, scope_stack, default_ctx)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return _resolve_name(node.id, scope_stack, default_ctx)
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, scope_stack, default_ctx)
        if isinstance(node.op, ast.Not):
            return not v
        if isinstance(node.op, ast.USub):
            return -v
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            for v in node.values:
                r = _eval(v, scope_stack, default_ctx)
                if not r:
                    return r
            return r
        if isinstance(node.op, ast.Or):
            for v in node.values:
                r = _eval(v, scope_stack, default_ctx)
                if r:
                    return r
            return r
    if isinstance(node, ast.BinOp):
        l = _eval(node.left, scope_stack, default_ctx)
        r = _eval(node.right, scope_stack, default_ctx)
        op = node.op
        if isinstance(op, ast.Add):
            return l + r
        if isinstance(op, ast.Sub):
            return l - r
        if isinstance(op, ast.Mult):
            return l * r
        if isinstance(op, ast.Div):
            return l / r
        if isinstance(op, ast.Mod):
            return l % r
    if isinstance(node, ast.Compare):
        left = _eval(node.left, scope_stack, default_ctx)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp, scope_stack, default_ctx)
            ok = _cmp(op, left, right)
            if not ok:
                return False
            left = right
        return True
    if isinstance(node, ast.Attribute):
        obj = _eval(node.value, scope_stack, default_ctx)
        if obj is None:
            return None
        try:
            return getattr(obj, node.attr)
        except AttributeError:
            if isinstance(obj, dict):
                return obj.get(node.attr)
            return None
    if isinstance(node, ast.Subscript):
        obj = _eval(node.value, scope_stack, default_ctx)
        key = _eval(node.slice, scope_stack, default_ctx)
        if obj is None:
            return None
        try:
            return obj[key]
        except (KeyError, IndexError, TypeError):
            return None
    if isinstance(node, ast.List):
        return [_eval(e, scope_stack, default_ctx) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval(e, scope_stack, default_ctx) for e in node.elts)
    if isinstance(node, ast.Call):
        # Visitor guarantees node.func is ast.Name and id is allowed.
        fn_name = node.func.id  # type: ignore[union-attr]
        args = [_eval(a, scope_stack, default_ctx) for a in node.args]
        try:
            return _BUILTINS_DISPATCH[fn_name](*args)
        except Exception as e:
            raise PredicateRuntimeError(
                f"builtin {fn_name}(...) failed: {e}"
            ) from e
    if isinstance(node, _Quantifier):
        steps = _resolve_name("steps", scope_stack, default_ctx) or []
        kind = node.kind
        if kind == "any_step":
            for s in steps:
                scope_stack.append(_step_frame(s))
                try:
                    if _eval(node.body, scope_stack, default_ctx):
                        return True
                finally:
                    scope_stack.pop()
            return False
        if kind == "all_step":
            for s in steps:
                scope_stack.append(_step_frame(s))
                try:
                    if not _eval(node.body, scope_stack, default_ctx):
                        return False
                finally:
                    scope_stack.pop()
            return True
    raise PredicateRuntimeError(f"unhandled node: {type(node).__name__}")


def _cmp(op: ast.cmpop, l: Any, r: Any) -> bool:
    if isinstance(op, ast.Eq):
        return l == r
    if isinstance(op, ast.NotEq):
        return l != r
    if isinstance(op, ast.In):
        try:
            return l in r
        except TypeError:
            return False
    if isinstance(op, ast.NotIn):
        try:
            return l not in r
        except TypeError:
            return False
    # Ordering ops: None propagates as False rather than TypeError.
    if l is None or r is None:
        return False
    try:
        if isinstance(op, ast.Lt):
            return l < r
        if isinstance(op, ast.LtE):
            return l <= r
        if isinstance(op, ast.Gt):
            return l > r
        if isinstance(op, ast.GtE):
            return l >= r
    except TypeError:
        return False
    return False


def _step_frame(step: Any) -> Dict[str, Any]:
    """Build a scope frame exposing StepView shorthands as bare names."""
    frame: Dict[str, Any] = {"step": step}
    for attr in _STEPVIEW_SHORTCUTS:
        frame[attr] = getattr(step, attr, None)
    return frame


def _value_frame(value: Any) -> Dict[str, Any]:
    """Build the top-level scope frame given an arbitrary predicate input.

    Predicates may be invoked with either a ``ReplayResult`` (minimize)
    or a ``StepView`` (bisect). Bind both styles of shortcuts so a single
    DSL string can target either.
    """
    frame: Dict[str, Any] = {"value": value}
    if hasattr(value, "steps") and hasattr(value, "total_cost_usd"):
        frame["result"] = value
        for attr in _RESULT_SHORTCUTS:
            frame[attr] = getattr(value, attr, None)
    if hasattr(value, "step_id") and hasattr(value, "kind"):
        frame.update(_step_frame(value))
    return frame


# ----------------------------------------------------------- DSL: compile


@functools.lru_cache(maxsize=128)
def _compile_cached(src: str, extra_keys: tuple) -> tuple:
    tree = parse_predicate(src)
    tree = _QuantifierRewriter(src).visit(tree)
    ast.fix_missing_locations(tree)
    _SafeVisitor(src).visit(tree)
    return (tree,)


def compile_predicate(
    src: str,
    *,
    extra_names: Optional[Dict[str, Any]] = None,
) -> Callable[[Any], bool]:
    """Compile DSL ``src`` into a sandboxed boolean predicate.

    The returned callable accepts either a ``ReplayResult`` or a
    ``StepView`` (or any object); names that don't resolve in the
    active context evaluate to ``None`` rather than raising.

    The returned callable carries:

    * ``.source`` — the original ``src`` string.
    * ``.parsed`` — the validated ``ast.Expression`` tree (for tooling).

    Compiled trees are LRU-cached on ``src`` (extra_names participates
    in the cache key by name only).
    """
    if not isinstance(src, str):
        raise TypeError("predicate source must be str")
    extras = dict(extra_names or {})
    cache_key = tuple(sorted(extras.keys()))
    (tree,) = _compile_cached(src, cache_key)

    def _predicate(value: Any) -> bool:
        scope_stack: list = [_value_frame(value)]
        default_ctx: Dict[str, Any] = dict(extras)
        try:
            r = _eval(tree, scope_stack, default_ctx)
        except PredicateRuntimeError:
            raise
        except Exception as e:
            raise PredicateRuntimeError(
                f"predicate {src!r} raised at eval: {e}"
            ) from e
        return bool(r)

    _predicate.source = src  # type: ignore[attr-defined]
    _predicate.parsed = tree  # type: ignore[attr-defined]
    return _predicate
