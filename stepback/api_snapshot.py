"""Generate machine-readable snapshots of stepback's public surface.

Two snapshots are produced:

* :func:`public_api_snapshot` enumerates every name in
  :data:`stepback.__all__` together with a stable, version-comparable
  description of the symbol (kind, signature, base classes, fields,
  enum members, deprecated flag).

* :func:`sbtrace_schema_snapshot` enumerates the on-disk SB-Trace
  schema published by :mod:`stepback.spec`: wire/format versions,
  encoding, magic, required/optional fields per frame kind, recognized
  step kinds, frame kinds, and supported capabilities.

Both snapshots are pure JSON-compatible Python and are designed to
diff cleanly against earlier releases. They are the input to
``scripts/check_api_compat.py``, which is invoked by CI to detect
silent breaking changes against the last released tag.

The shape of the snapshot itself is part of the contract: if you
reorder, rename, or add fields here you must also bump
:data:`SNAPSHOT_FORMAT_VERSION` and update the baselines under
``stepback/conformance/api_baselines/``.
"""
from __future__ import annotations

import enum
import inspect
import json
from dataclasses import fields, is_dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import stepback
from stepback import spec as _spec_module


SNAPSHOT_FORMAT_VERSION: int = 1
"""Schema version for the JSON produced by this module.

Bump this whenever a non-backwards-compatible field is added, removed,
or renamed in the snapshot output. Comparators check this first and
refuse to compare snapshots across major versions.
"""


# ---------------------------------------------------------------------------
# Public API snapshot
# ---------------------------------------------------------------------------


def _format_annotation(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return ""
    if isinstance(annotation, str):
        return annotation
    mod = getattr(annotation, "__module__", "")
    name = getattr(annotation, "__qualname__", None) or getattr(
        annotation, "__name__", None
    )
    if name and mod and mod not in ("builtins", "typing"):
        return f"{mod}.{name}"
    if name:
        return name
    return repr(annotation)


def _format_default(default: Any) -> str:
    if default is inspect.Parameter.empty:
        return ""
    try:
        return repr(default)
    except Exception:  # pragma: no cover - defensive
        return "<unrepr>"


def _signature_record(obj: Any) -> Dict[str, Any]:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return {"signature": None}
    params: List[Dict[str, Any]] = []
    for p in sig.parameters.values():
        params.append(
            {
                "name": p.name,
                "kind": p.kind.name,
                "annotation": _format_annotation(p.annotation),
                "default": _format_default(p.default),
            }
        )
    return {
        "signature": {
            "parameters": params,
            "return_annotation": _format_annotation(sig.return_annotation),
        }
    }


def _class_record(obj: type) -> Dict[str, Any]:
    bases = [
        f"{b.__module__}.{b.__qualname__}"
        for b in obj.__mro__[1:-1]  # skip self and object
    ]
    record: Dict[str, Any] = {
        "kind": "class",
        "qualname": f"{obj.__module__}.{obj.__qualname__}",
        "bases": bases,
        "is_exception": isinstance(obj, type) and issubclass(obj, BaseException),
        "is_dataclass": is_dataclass(obj),
        "is_enum": isinstance(obj, type) and issubclass(obj, enum.Enum),
    }
    if record["is_dataclass"]:
        record["fields"] = [
            {
                "name": f.name,
                "type": _format_annotation(f.type),
                "has_default": (
                    f.default is not inspect.Parameter.empty
                    and repr(f.default) != "<factory>"
                )
                or f.default_factory is not inspect.Parameter.empty,  # type: ignore[attr-defined]
            }
            for f in fields(obj)
        ]
    if record["is_enum"]:
        record["enum_members"] = sorted(
            (m.name, _format_default(m.value)) for m in obj  # type: ignore[arg-type]
        )
    # Public methods on the class itself (no inherited methods, no dunders
    # except __init__ / __call__ which actually matter for API users).
    methods: Dict[str, Any] = {}
    for name, value in inspect.getmembers(obj):
        if name.startswith("_") and name not in ("__init__", "__call__"):
            continue
        if name not in obj.__dict__:
            continue
        if not callable(value):
            continue
        try:
            methods[name] = _signature_record(value)["signature"]
        except Exception:  # pragma: no cover - defensive
            methods[name] = None
    if methods:
        record["methods"] = methods
    return record


def _function_record(obj: Any) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "kind": "function",
        "qualname": f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', '?')}",
    }
    record.update(_signature_record(obj))
    return record


def _module_record(obj: Any) -> Dict[str, Any]:
    return {
        "kind": "module",
        "qualname": getattr(obj, "__name__", repr(obj)),
    }


def _value_record(obj: Any) -> Dict[str, Any]:
    type_name = type(obj).__name__
    summary: Dict[str, Any] = {"kind": "value", "type": type_name}
    if isinstance(obj, (str, int, float, bool, type(None))):
        summary["value"] = obj
    elif isinstance(obj, (list, tuple)):
        summary["value"] = [
            x if isinstance(x, (str, int, float, bool, type(None))) else repr(x)
            for x in obj
        ]
        summary["type"] = "list" if isinstance(obj, list) else "tuple"
    elif isinstance(obj, frozenset):
        summary["value"] = sorted(repr(x) for x in obj)
        summary["type"] = "frozenset"
    elif isinstance(obj, Mapping):
        summary["value"] = {
            str(k): (
                v if isinstance(v, (str, int, float, bool, type(None))) else repr(v)
            )
            for k, v in obj.items()
        }
        summary["type"] = "mapping"
    return summary


def _is_deprecated(obj: Any) -> bool:
    return bool(getattr(obj, "__stepback_deprecated__", False))


def _describe(name: str, obj: Any) -> Dict[str, Any]:
    if inspect.ismodule(obj):
        record = _module_record(obj)
    elif inspect.isclass(obj):
        record = _class_record(obj)
    elif inspect.isfunction(obj) or inspect.isbuiltin(obj) or callable(obj) and not isinstance(
        obj, type
    ) and hasattr(obj, "__call__") and inspect.isroutine(obj):
        record = _function_record(obj)
    else:
        record = _value_record(obj)
    record["name"] = name
    if _is_deprecated(obj):
        record["deprecated"] = True
    return record


def public_api_snapshot() -> Dict[str, Any]:
    """Return a JSON-serializable snapshot of ``stepback.__all__``.

    Each entry contains the symbol's kind (``"function"``, ``"class"``,
    ``"module"``, ``"value"``), its qualified name, and enough structural
    information to detect breaking changes: function signatures,
    dataclass fields, base classes, and whether the symbol is marked
    deprecated via :func:`stepback._deprecation.deprecated`.
    """
    symbols: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    for name in sorted(stepback.__all__):
        try:
            obj = getattr(stepback, name)
        except AttributeError:
            missing.append(name)
            continue
        symbols[name] = _describe(name, obj)
    return {
        "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
        "package": "stepback",
        "package_version": stepback.__version__,
        "symbols": symbols,
        "missing": missing,
    }


# ---------------------------------------------------------------------------
# SB-Trace schema snapshot
# ---------------------------------------------------------------------------


def sbtrace_schema_snapshot() -> Dict[str, Any]:
    """Return a JSON-serializable snapshot of the current SB-Trace schema.

    Captures the wire-format SemVer pins, encoding labels, magic string,
    required/optional fields per frame kind, recognized step kinds,
    frame kinds, and supported capabilities. A change to any of these
    is, by definition, a wire-format change and must trigger a wire
    SemVer bump (see ``stepback/spec.py``).
    """
    spec = _spec_module.current_spec()

    def _sorted(values: Iterable[str]) -> List[str]:
        return sorted(values)

    return {
        "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
        "wire_version": spec.wire_version,
        "wire_version_info": list(_spec_module.SBTRACE_WIRE_VERSION_INFO),
        "format_version": spec.format_version,
        "encoding": spec.encoding,
        "magic": spec.magic,
        "wire_encodings": {str(k): v for k, v in _spec_module.SBTRACE_WIRE_ENCODINGS.items()},
        "format_version_to_wire": {
            str(k): v for k, v in _spec_module.SBTRACE_FORMAT_VERSION_TO_WIRE.items()
        },
        "wrapper_required": _sorted(spec.wrapper_required),
        "header_required": _sorted(spec.header_required),
        "header_optional": _sorted(spec.header_optional),
        "step_required": _sorted(spec.step_required),
        "step_optional": _sorted(spec.step_optional),
        "step_kinds": _sorted(spec.step_kinds),
        "frame_kinds": _sorted(spec.frame_kinds),
        "supported_capabilities": _sorted(spec.supported_capabilities),
        "strict_unknown_step_kinds": spec.strict_unknown_step_kinds,
        "strict_unknown_optional_fields": spec.strict_unknown_optional_fields,
    }


# ---------------------------------------------------------------------------
# Diffing
# ---------------------------------------------------------------------------


def _diff_signatures(
    old: Any, new: Any, qualified: str
) -> List[Tuple[str, str]]:
    """Return ``(severity, message)`` pairs describing signature changes.

    Severity is ``"breaking"`` for changes that may break callers and
    ``"compatible"`` for backward-compatible additions.
    """
    if old is None and new is None:
        return []
    if old is None and new is not None:
        return [("compatible", f"{qualified}: signature now introspectable")]
    if new is None and old is not None:
        return [("breaking", f"{qualified}: signature no longer introspectable")]
    issues: List[Tuple[str, str]] = []
    old_params = old.get("parameters") or []
    new_params = new.get("parameters") or []
    old_by_name = {p["name"]: p for p in old_params}
    new_by_name = {p["name"]: p for p in new_params}
    for name, p in old_by_name.items():
        if name not in new_by_name:
            issues.append(("breaking", f"{qualified}: removed parameter {name!r}"))
            continue
        np = new_by_name[name]
        if p["kind"] != np["kind"]:
            issues.append(
                (
                    "breaking",
                    f"{qualified}: parameter {name!r} kind changed "
                    f"{p['kind']} -> {np['kind']}",
                )
            )
        old_default = p.get("default", "")
        new_default = np.get("default", "")
        if old_default != "" and new_default == "":
            issues.append(
                (
                    "breaking",
                    f"{qualified}: parameter {name!r} no longer optional "
                    f"(default removed)",
                )
            )
    for name, p in new_by_name.items():
        if name not in old_by_name:
            severity = "compatible" if p.get("default", "") != "" else "breaking"
            issues.append(
                (
                    severity,
                    f"{qualified}: new parameter {name!r} "
                    f"({'has default' if severity == 'compatible' else 'required'})",
                )
            )
    if old.get("return_annotation") != new.get("return_annotation"):
        # Treat return-annotation changes as informational; runtime behavior
        # may still be compatible. We surface as compatible so reviewers see
        # it without failing CI.
        issues.append(
            (
                "compatible",
                f"{qualified}: return annotation changed "
                f"{old.get('return_annotation')!r} -> "
                f"{new.get('return_annotation')!r}",
            )
        )
    return issues


def diff_public_api(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> Dict[str, List[str]]:
    """Compare two :func:`public_api_snapshot` outputs.

    Returns a dict with two keys, ``"breaking"`` and ``"compatible"``,
    each a list of human-readable change descriptions. An empty
    ``"breaking"`` list means CI should pass.
    """
    breaking: List[str] = []
    compatible: List[str] = []
    if old.get("snapshot_format_version") != new.get("snapshot_format_version"):
        breaking.append(
            f"snapshot_format_version changed "
            f"{old.get('snapshot_format_version')!r} -> "
            f"{new.get('snapshot_format_version')!r}; "
            f"refusing to compare across snapshot major versions"
        )
        return {"breaking": breaking, "compatible": compatible}
    old_syms: Dict[str, Dict[str, Any]] = dict(old.get("symbols") or {})
    new_syms: Dict[str, Dict[str, Any]] = dict(new.get("symbols") or {})
    for name, rec in old_syms.items():
        if name not in new_syms:
            if rec.get("deprecated"):
                compatible.append(f"removed deprecated symbol {name!r}")
            else:
                breaking.append(f"removed public symbol {name!r}")
            continue
        nrec = new_syms[name]
        if rec.get("kind") != nrec.get("kind"):
            breaking.append(
                f"{name}: kind changed {rec.get('kind')!r} -> {nrec.get('kind')!r}"
            )
            continue
        if rec.get("kind") == "function":
            for severity, msg in _diff_signatures(
                rec.get("signature"), nrec.get("signature"), name
            ):
                (breaking if severity == "breaking" else compatible).append(msg)
        elif rec.get("kind") == "class":
            old_bases = list(rec.get("bases") or [])
            new_bases = list(nrec.get("bases") or [])
            removed_bases = [b for b in old_bases if b not in new_bases]
            if removed_bases:
                breaking.append(
                    f"{name}: removed base class(es) {removed_bases!r}"
                )
            old_fields = {f["name"]: f for f in rec.get("fields") or []}
            new_fields = {f["name"]: f for f in nrec.get("fields") or []}
            for fname in old_fields:
                if fname not in new_fields:
                    breaking.append(
                        f"{name}.{fname}: removed dataclass field"
                    )
            for fname, f in new_fields.items():
                if fname not in old_fields and not f.get("has_default"):
                    breaking.append(
                        f"{name}.{fname}: new required dataclass field"
                    )
                elif fname not in old_fields:
                    compatible.append(
                        f"{name}.{fname}: new optional dataclass field"
                    )
            old_methods = dict(rec.get("methods") or {})
            new_methods = dict(nrec.get("methods") or {})
            for mname in old_methods:
                if mname not in new_methods:
                    breaking.append(f"{name}.{mname}: removed public method")
                else:
                    for severity, msg in _diff_signatures(
                        old_methods.get(mname),
                        new_methods.get(mname),
                        f"{name}.{mname}",
                    ):
                        (breaking if severity == "breaking" else compatible).append(
                            msg
                        )
            old_enum = dict(rec.get("enum_members") or [])
            new_enum = dict(nrec.get("enum_members") or [])
            for ename in old_enum:
                if ename not in new_enum:
                    breaking.append(f"{name}.{ename}: removed enum member")
                elif old_enum[ename] != new_enum[ename]:
                    breaking.append(
                        f"{name}.{ename}: enum value changed "
                        f"{old_enum[ename]!r} -> {new_enum[ename]!r}"
                    )
        elif rec.get("kind") == "value":
            if rec.get("type") != nrec.get("type"):
                breaking.append(
                    f"{name}: value type changed "
                    f"{rec.get('type')!r} -> {nrec.get('type')!r}"
                )
            elif rec.get("value") != nrec.get("value"):
                # Value identity changed; for primitives this is breaking
                # (constants are part of the API contract).
                breaking.append(
                    f"{name}: value changed "
                    f"{rec.get('value')!r} -> {nrec.get('value')!r}"
                )
    for name in new_syms:
        if name not in old_syms:
            compatible.append(f"new public symbol {name!r}")
    return {"breaking": breaking, "compatible": compatible}


_SCHEMA_BREAKING_REMOVAL_FIELDS = (
    "wrapper_required",
    "header_required",
    "step_required",
    "step_kinds",
    "frame_kinds",
    "supported_capabilities",
)
_SCHEMA_BREAKING_ADDITION_FIELDS = (
    "wrapper_required",
    "header_required",
    "step_required",
)


def diff_sbtrace_schema(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> Dict[str, List[str]]:
    """Compare two :func:`sbtrace_schema_snapshot` outputs.

    A change to required fields, magic, or the wire major bumps the
    breaking list. Additive changes to optional fields and capabilities
    are compatible.
    """
    breaking: List[str] = []
    compatible: List[str] = []
    if old.get("snapshot_format_version") != new.get("snapshot_format_version"):
        breaking.append(
            f"snapshot_format_version changed "
            f"{old.get('snapshot_format_version')!r} -> "
            f"{new.get('snapshot_format_version')!r}"
        )
        return {"breaking": breaking, "compatible": compatible}
    old_major = (old.get("wire_version_info") or [None])[0]
    new_major = (new.get("wire_version_info") or [None])[0]
    if old_major != new_major:
        breaking.append(
            f"wire major version changed {old_major!r} -> {new_major!r}"
        )
    if old.get("magic") != new.get("magic"):
        breaking.append(
            f"magic changed {old.get('magic')!r} -> {new.get('magic')!r}"
        )
    if old.get("encoding") != new.get("encoding"):
        breaking.append(
            f"encoding changed {old.get('encoding')!r} -> {new.get('encoding')!r}"
        )
    if old.get("format_version") != new.get("format_version"):
        breaking.append(
            f"format_version changed {old.get('format_version')!r} -> "
            f"{new.get('format_version')!r}"
        )
    for fname in _SCHEMA_BREAKING_REMOVAL_FIELDS:
        old_set = set(old.get(fname) or [])
        new_set = set(new.get(fname) or [])
        removed = sorted(old_set - new_set)
        if removed:
            breaking.append(f"{fname}: removed entries {removed!r}")
        added = sorted(new_set - old_set)
        if added:
            if fname in _SCHEMA_BREAKING_ADDITION_FIELDS:
                breaking.append(
                    f"{fname}: new required entries {added!r}"
                )
            else:
                compatible.append(f"{fname}: added entries {added!r}")
    for fname in ("header_optional", "step_optional"):
        old_set = set(old.get(fname) or [])
        new_set = set(new.get(fname) or [])
        removed = sorted(old_set - new_set)
        added = sorted(new_set - old_set)
        if removed:
            # Removing an optional field is *technically* compatible for
            # writers but breaking for readers that expected to see it.
            # Conservatively treat as breaking; minor bumps are cheap.
            breaking.append(f"{fname}: removed optional entries {removed!r}")
        if added:
            compatible.append(f"{fname}: added optional entries {added!r}")
    if old.get("strict_unknown_step_kinds") != new.get("strict_unknown_step_kinds"):
        breaking.append(
            f"strict_unknown_step_kinds changed "
            f"{old.get('strict_unknown_step_kinds')!r} -> "
            f"{new.get('strict_unknown_step_kinds')!r}"
        )
    if old.get("strict_unknown_optional_fields") != new.get(
        "strict_unknown_optional_fields"
    ):
        breaking.append(
            f"strict_unknown_optional_fields changed "
            f"{old.get('strict_unknown_optional_fields')!r} -> "
            f"{new.get('strict_unknown_optional_fields')!r}"
        )
    return {"breaking": breaking, "compatible": compatible}


def dumps(snapshot: Mapping[str, Any]) -> str:
    """Serialize a snapshot to canonical, diff-friendly JSON."""
    return json.dumps(snapshot, sort_keys=True, indent=2) + "\n"
