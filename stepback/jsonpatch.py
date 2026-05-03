"""RFC 6902 JSON Patch (subset) + RFC 6901 JSON Pointer.

Implemented in-tree to avoid pulling a dependency into the wheel.
We support the ops actually useful for `stepback` substitutions —
add, replace, remove, test, copy, move — with strict semantics.

Errors are subclasses of :class:`PatchError` which is itself a
:class:`ValueError`, so callers can catch broadly.
"""
from __future__ import annotations

import copy
from typing import Any, List, Tuple


class PatchError(ValueError):
    """Base for all JSON Patch / Pointer errors."""


class PatchPathNotFound(PatchError):
    """The pointer doesn't resolve in the document."""


class PatchTestFailed(PatchError):
    """A `test` op asserted a value that didn't match."""


class PatchInvalidOp(PatchError):
    """The patch op is malformed or unsupported."""


_SUPPORTED_OPS = {"add", "replace", "remove", "test", "copy", "move"}


def _decode_token(tok: str) -> str:
    # RFC 6901: ~1 -> /, ~0 -> ~. Decode ~1 first so ~01 -> ~1, not /.
    return tok.replace("~1", "/").replace("~0", "~")


def _split_pointer(path: str) -> List[str]:
    if path == "":
        return []
    if not path.startswith("/"):
        raise PatchInvalidOp(f"path must start with '/' or be empty, got {path!r}")
    return [_decode_token(t) for t in path.split("/")[1:]]


def _is_array_index(tok: str) -> bool:
    if tok == "":
        return False
    if tok == "0":
        return True
    if tok[0] == "0":  # no leading zeros
        return False
    return tok.isdigit()


def _resolve_parent(doc: Any, tokens: List[str]) -> Tuple[Any, str]:
    """Walk all but the last token; return (parent, last_token)."""
    if not tokens:
        raise PatchInvalidOp("cannot resolve parent of root")
    cur = doc
    for tok in tokens[:-1]:
        if isinstance(cur, list):
            if not _is_array_index(tok):
                raise PatchPathNotFound(f"expected array index, got {tok!r}")
            i = int(tok)
            if i >= len(cur):
                raise PatchPathNotFound(f"array index {i} out of range")
            cur = cur[i]
        elif isinstance(cur, dict):
            if tok not in cur:
                raise PatchPathNotFound(f"missing key {tok!r}")
            cur = cur[tok]
        else:
            raise PatchPathNotFound(
                f"cannot descend into {type(cur).__name__} at {tok!r}"
            )
    return cur, tokens[-1]


def _resolve_get(doc: Any, tokens: List[str]) -> Any:
    if not tokens:
        return doc
    parent, last = _resolve_parent(doc, tokens)
    if isinstance(parent, list):
        if not _is_array_index(last):
            raise PatchPathNotFound(f"expected array index, got {last!r}")
        i = int(last)
        if i >= len(parent):
            raise PatchPathNotFound(f"array index {i} out of range")
        return parent[i]
    if isinstance(parent, dict):
        if last not in parent:
            raise PatchPathNotFound(f"missing key {last!r}")
        return parent[last]
    raise PatchPathNotFound(f"cannot index {type(parent).__name__}")


def _op_add(doc: Any, tokens: List[str], value: Any) -> Any:
    if not tokens:
        return value
    parent, last = _resolve_parent(doc, tokens)
    if isinstance(parent, list):
        if last == "-":
            parent.append(value)
        elif _is_array_index(last):
            i = int(last)
            if i > len(parent):
                raise PatchPathNotFound(f"array index {i} out of range for add")
            parent.insert(i, value)
        else:
            raise PatchPathNotFound(f"expected array index, got {last!r}")
    elif isinstance(parent, dict):
        parent[last] = value
    else:
        raise PatchPathNotFound(f"cannot add into {type(parent).__name__}")
    return doc


def _op_replace(doc: Any, tokens: List[str], value: Any) -> Any:
    if not tokens:
        return value
    parent, last = _resolve_parent(doc, tokens)
    if isinstance(parent, list):
        if not _is_array_index(last):
            raise PatchPathNotFound(f"expected array index, got {last!r}")
        i = int(last)
        if i >= len(parent):
            raise PatchPathNotFound(f"array index {i} out of range for replace")
        parent[i] = value
    elif isinstance(parent, dict):
        if last not in parent:
            raise PatchPathNotFound(f"missing key {last!r} for replace")
        parent[last] = value
    else:
        raise PatchPathNotFound(f"cannot replace into {type(parent).__name__}")
    return doc


def _op_remove(doc: Any, tokens: List[str]) -> Any:
    if not tokens:
        raise PatchInvalidOp("cannot remove root")
    parent, last = _resolve_parent(doc, tokens)
    if isinstance(parent, list):
        if not _is_array_index(last):
            raise PatchPathNotFound(f"expected array index, got {last!r}")
        i = int(last)
        if i >= len(parent):
            raise PatchPathNotFound(f"array index {i} out of range for remove")
        del parent[i]
    elif isinstance(parent, dict):
        if last not in parent:
            raise PatchPathNotFound(f"missing key {last!r} for remove")
        del parent[last]
    else:
        raise PatchPathNotFound(f"cannot remove from {type(parent).__name__}")
    return doc


def _op_test(doc: Any, tokens: List[str], value: Any) -> Any:
    have = _resolve_get(doc, tokens)
    # Strict equality: 1 != 1.0 (matches RFC 6902 §4.6 — JSON-equality).
    if type(have) is not type(value) or have != value:
        raise PatchTestFailed(f"test failed at {'/'.join(tokens) or '/'}: {have!r} != {value!r}")
    return doc


def apply_patch(doc: Any, ops: List[dict]) -> Any:
    """Apply a JSON Patch list to ``doc`` (deep-copied first).

    Returns the new document. Raises subclasses of :class:`PatchError`
    on any failure; a `test` failure aborts the rest of the ops
    (transactional within one call).
    """
    if not isinstance(ops, list):
        raise PatchInvalidOp(f"ops must be a list, got {type(ops).__name__}")
    out = copy.deepcopy(doc)
    for i, op in enumerate(ops):
        if not isinstance(op, dict):
            raise PatchInvalidOp(f"op #{i} must be a dict, got {type(op).__name__}")
        name = op.get("op")
        if name not in _SUPPORTED_OPS:
            raise PatchInvalidOp(f"op #{i}: unsupported op {name!r}")
        path = op.get("path")
        if not isinstance(path, str):
            raise PatchInvalidOp(f"op #{i}: missing/non-string 'path'")
        tokens = _split_pointer(path)
        if name == "add":
            if "value" not in op:
                raise PatchInvalidOp(f"op #{i}: 'add' requires 'value'")
            out = _op_add(out, tokens, op["value"])
        elif name == "replace":
            if "value" not in op:
                raise PatchInvalidOp(f"op #{i}: 'replace' requires 'value'")
            out = _op_replace(out, tokens, op["value"])
        elif name == "remove":
            out = _op_remove(out, tokens)
        elif name == "test":
            if "value" not in op:
                raise PatchInvalidOp(f"op #{i}: 'test' requires 'value'")
            _op_test(out, tokens, op["value"])
        elif name in ("copy", "move"):
            from_path = op.get("from")
            if not isinstance(from_path, str):
                raise PatchInvalidOp(f"op #{i}: {name!r} requires 'from'")
            from_tokens = _split_pointer(from_path)
            if name == "move" and from_tokens == tokens:
                continue  # no-op per RFC 6902
            value = copy.deepcopy(_resolve_get(out, from_tokens))
            if name == "move":
                out = _op_remove(out, from_tokens)
            out = _op_add(out, tokens, value)
    return out
