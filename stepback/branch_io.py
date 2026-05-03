"""Counterfactual branch persistence (`.sbb` files).

A `.sbb` ("stepback branch") is a JSON document that pins a counterfactual
to a base trace by content hash and records the typed substitution list
that produced it. Replaying a branch is reproducible from the trace +
the `.sbb` alone — no Python source required — which is what the
README's `stepback replay` and `stepback diff` CLIs need.

The on-disk shape is intentionally minimal::

    {
      "magic": "stepback/.sbb",
      "format_version": 1,
      "name": "counterfactual-paranoid",
      "base_step": "step:7",
      "trace_path": "./traces/incident-2026-04-12.sb",
      "trace_inputs_hash_chain": "sha256:...",   # binds branch to trace
      "substitutions": [
        {"type": "PromptSubstitution", "at_step": "step:7",
         "new_messages": [...]},
        {"type": "ToolOutputSubstitution", "at_step": "step:12",
         "fake_response": {...}, "tool_call_id": "call_a1"},
        ...
      ]
    }

If the trace is later mutated, ``trace_inputs_hash_chain`` no longer
matches and load_branch raises :py:class:`BranchTraceMismatch` so a
counterfactual can never silently bind to a different recorded run.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

from .canonical import hash_obj
from .substitutions import (
    InputsPatchSubstitution,
    MessagePatchSubstitution,
    ModelSubstitution,
    OutputsPatchSubstitution,
    PolicySubstitution,
    PromptSubstitution,
    RaiseSubstitution,
    RouterSubstitution,
    SamplingSubstitution,
    Substitution,
    SubstitutionSet,
    SystemPromptSubstitution,
    ToolArgumentsSubstitution,
    ToolOutputSubstitution,
)

SBB_MAGIC = "stepback/.sbb"
SBB_FORMAT_VERSION = 1


_TYPE_MAP = {
    "PromptSubstitution": PromptSubstitution,
    "ModelSubstitution": ModelSubstitution,
    "ToolOutputSubstitution": ToolOutputSubstitution,
    "PolicySubstitution": PolicySubstitution,
    "RouterSubstitution": RouterSubstitution,
    "SystemPromptSubstitution": SystemPromptSubstitution,
    "MessagePatchSubstitution": MessagePatchSubstitution,
    "SamplingSubstitution": SamplingSubstitution,
    "ToolArgumentsSubstitution": ToolArgumentsSubstitution,
    "InputsPatchSubstitution": InputsPatchSubstitution,
    "OutputsPatchSubstitution": OutputsPatchSubstitution,
    "RaiseSubstitution": RaiseSubstitution,
}


class BranchTraceMismatch(ValueError):
    """A `.sbb` was loaded against a trace whose inputs-hash chain differs.

    This means the trace was edited, re-recorded, or simply the wrong
    file. The counterfactual is no longer reproducible against this
    trace, so we refuse to load it rather than produce a misleading
    diff.
    """


def trace_chain_hash(recorded_steps: List[dict]) -> str:
    """Stable digest of a trace's identity.

    We hash the ordered list of ``(step_id, inputs_hash, outputs_hash)``
    triples, which is independent of recorder timestamps but uniquely
    pins the recorded computation a branch was authored against.
    """
    triples = [
        (s["step_id"], s.get("inputs_hash", ""), s.get("outputs_hash", ""))
        for s in recorded_steps
    ]
    return hash_obj(triples)


def substitution_to_dict(sub: Substitution) -> Dict[str, Any]:
    if not is_dataclass(sub):
        raise TypeError(f"substitution {sub!r} is not a dataclass")
    body = {k: _jsonify(v) for k, v in asdict(sub).items()}
    body["type"] = type(sub).__name__
    return body


def substitution_from_dict(d: Dict[str, Any]) -> Substitution:
    type_name = d.get("type")
    cls = _TYPE_MAP.get(type_name)
    if cls is None:
        raise ValueError(f"unknown substitution type {type_name!r}")
    kwargs = {k: v for k, v in d.items() if k != "type"}
    return cls(**kwargs)


def save_branch(
    path: str,
    *,
    name: str,
    base_step: str,
    trace_path: str,
    trace_chain: str,
    substitutions: List[Substitution],
) -> None:
    """Write a `.sbb` file describing a counterfactual branch."""
    body = {
        "magic": SBB_MAGIC,
        "format_version": SBB_FORMAT_VERSION,
        "name": name,
        "base_step": base_step,
        "trace_path": trace_path,
        "trace_inputs_hash_chain": trace_chain,
        "substitutions": [substitution_to_dict(s) for s in substitutions],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2, sort_keys=True)
        f.write("\n")


def load_branch(
    path: str, *, expected_chain: Optional[str] = None
) -> Dict[str, Any]:
    """Load a `.sbb` file. If ``expected_chain`` is supplied and does not
    match the recorded chain, raise :py:class:`BranchTraceMismatch`.

    Returns a dict with keys ``name``, ``base_step``, ``trace_path``,
    ``substitutions`` (a fresh :py:class:`SubstitutionSet`) and the raw
    ``trace_inputs_hash_chain``.
    """
    with open(path, "r", encoding="utf-8") as f:
        body = json.load(f)
    if body.get("magic") != SBB_MAGIC:
        raise ValueError(f"{path}: not a stepback branch file")
    if body.get("format_version") != SBB_FORMAT_VERSION:
        raise ValueError(
            f"{path}: unsupported format_version {body.get('format_version')!r}"
        )
    chain = body.get("trace_inputs_hash_chain", "")
    if expected_chain is not None and chain != expected_chain:
        raise BranchTraceMismatch(
            f"{path} was authored against a different trace: "
            f"recorded={chain!r} actual={expected_chain!r}"
        )
    subset = SubstitutionSet()
    for d in body.get("substitutions", []):
        subset.add(substitution_from_dict(d))
    return {
        "name": body["name"],
        "base_step": body["base_step"],
        "trace_path": body["trace_path"],
        "trace_inputs_hash_chain": chain,
        "substitutions": subset,
    }


# --------------------------------------------------------- substitution
# spec parser used by the CLI for `--substitute KIND@step:N=BODY` flags.


def parse_substitution_spec(spec: str) -> Substitution:
    """Parse a CLI-friendly substitution spec.

    Grammar (deliberately small, one substitution per spec)::

        prompt@step:N=path/to/messages.json
        prompt@step:N=:inline:<json-array>
        model@step:N=gpt-4o-mini-2024-07-18
        tool_output@step:N=path/to/response.json
        tool_output@step:N=:inline:<json-object>
        policy@step:N=path/to/policy.tw
        router@step:N=choice_name

    The ``path`` form reads JSON from disk; the ``:inline:`` form
    accepts a JSON literal directly on the command line (handy for
    one-off shell invocations).
    """
    if "@" not in spec or "=" not in spec:
        raise ValueError(
            f"bad substitution spec {spec!r}: expected KIND@STEP=BODY"
        )
    kind, rest = spec.split("@", 1)
    step_id, body = rest.split("=", 1)
    kind = kind.strip().lower()
    step_id = step_id.strip()
    body = body.strip()

    def _read_json(b: str) -> Any:
        if b.startswith(":inline:"):
            return json.loads(b[len(":inline:") :])
        with open(b, "r", encoding="utf-8") as f:
            return json.load(f)

    if kind == "prompt":
        msgs = _read_json(body)
        if not isinstance(msgs, list):
            raise ValueError("prompt body must be a JSON array of messages")
        return PromptSubstitution(at_step=step_id, new_messages=msgs)
    if kind == "model":
        return ModelSubstitution(at_step=step_id, new_model_id=body)
    if kind == "tool_output":
        return ToolOutputSubstitution(
            at_step=step_id, fake_response=_read_json(body)
        )
    if kind == "policy":
        return PolicySubstitution(at_step=step_id, policy_path=body)
    if kind == "router":
        return RouterSubstitution(at_step=step_id, choice=body)
    if kind in ("system", "system_replace", "system_prepend", "system_append"):
        text = _read_json(body) if body.startswith((":inline:", "{", "[", '"')) else body
        if body.startswith(":inline:"):
            text = json.loads(body[len(":inline:") :])
        if not isinstance(text, str):
            raise ValueError("system body must be a JSON string")
        mode = "replace" if kind in ("system", "system_replace") else (
            "prepend" if kind == "system_prepend" else "append"
        )
        return SystemPromptSubstitution(at_step=step_id, system_text=text, mode=mode)
    if kind == "message":
        idx, msg = _parse_indexed(body)
        return MessagePatchSubstitution(at_step=step_id, index=idx, new_message=msg)
    if kind == "sampling":
        params = _parse_kv_or_json(body)
        return SamplingSubstitution(at_step=step_id, **params)
    if kind == "tool_args":
        return ToolArgumentsSubstitution(at_step=step_id, new_arguments=_read_json(body))
    if kind == "inputs_patch":
        ops = _read_json(body)
        if not isinstance(ops, list):
            raise ValueError("inputs_patch body must be a JSON array of ops")
        return InputsPatchSubstitution(at_step=step_id, ops=ops)
    if kind == "outputs_patch":
        ops = _read_json(body)
        if not isinstance(ops, list):
            raise ValueError("outputs_patch body must be a JSON array of ops")
        return OutputsPatchSubstitution(at_step=step_id, ops=ops)
    if kind == "raise":
        if ":" in body:
            etype, msg = body.split(":", 1)
        else:
            etype, msg = body, ""
        return RaiseSubstitution(
            at_step=step_id, exception_type=etype.strip(), message=msg.strip()
        )
    raise ValueError(
        f"unknown substitution kind {kind!r}; expected one of "
        f"prompt/model/tool_output/policy/router/system/system_prepend/"
        f"system_append/message/sampling/tool_args/inputs_patch/"
        f"outputs_patch/raise"
    )


def _parse_indexed(body: str):
    """Parse `:idx=N,inline:JSON` or `:idx=N,path=FILE` for message subs."""
    if not body.startswith(":idx="):
        raise ValueError(f"message body must start with ':idx=', got {body!r}")
    rest = body[len(":idx=") :]
    if "," not in rest:
        raise ValueError(f"message body must be ':idx=N,inline:..' or ':idx=N,path=..', got {body!r}")
    idx_str, payload = rest.split(",", 1)
    try:
        idx = int(idx_str)
    except ValueError as e:
        raise ValueError(f"message index must be int, got {idx_str!r}") from e
    if payload.startswith("inline:"):
        msg = json.loads(payload[len("inline:") :])
    elif payload.startswith("path="):
        with open(payload[len("path=") :], "r", encoding="utf-8") as f:
            msg = json.load(f)
    else:
        raise ValueError(f"message payload must be 'inline:..' or 'path=..', got {payload!r}")
    if not isinstance(msg, dict):
        raise ValueError("message new_message must be a JSON object")
    return idx, msg


def _parse_kv_or_json(body: str) -> Dict[str, Any]:
    """Parse `:kv:k=v,k=v` or `:inline:JSON` or `path` to a dict."""
    if body.startswith(":inline:"):
        d = json.loads(body[len(":inline:") :])
        if not isinstance(d, dict):
            raise ValueError("sampling :inline: body must be a JSON object")
        return d
    if body.startswith(":kv:"):
        out: Dict[str, Any] = {}
        for pair in body[len(":kv:") :].split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(f"sampling kv pair missing '=': {pair!r}")
            k, v = pair.split("=", 1)
            k, v = k.strip(), v.strip()
            if v == "null":
                out[k] = None
            else:
                try:
                    if "." in v or "e" in v or "E" in v:
                        out[k] = float(v)
                    else:
                        out[k] = int(v)
                except ValueError:
                    out[k] = v
        return out
    # path fallback
    with open(body, "r", encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise ValueError("sampling path body must be a JSON object")
    return d


# ------------------------------------------------------------- diffing


def diff_replays(a, b) -> Dict[str, Any]:
    """Step-by-step diff of two :py:class:`ReplayResult` objects.

    Returns a JSON-serialisable dict shaped like::

        {
          "a_total_cost_usd": 0.012,
          "b_total_cost_usd": 0.009,
          "total_cost_delta_usd": -0.003,
          "divergent_step_count": 3,
          "step_diffs": [
            {"step_id": "step:7", "kind": "llm_call",
             "diverged": true, "cost_delta_usd": -0.0011,
             "a_outputs_hash": "sha256:...", "b_outputs_hash": "sha256:..."},
            ...
          ]
        }

    Used by :func:`stepback.cli._cmd_diff` and any caller that wants a
    machine-readable comparison without paying the cost of formatting.
    """
    a_by = {s.step_id: s for s in a.steps}
    b_by = {s.step_id: s for s in b.steps}
    ids = sorted(set(a_by) | set(b_by), key=lambda x: int(x.split(":")[-1]))
    step_diffs: List[Dict[str, Any]] = []
    diverged = 0
    for sid in ids:
        sa = a_by.get(sid)
        sb = b_by.get(sid)
        kind = (sa or sb).kind
        ah = hash_obj(sa.outputs) if sa else None
        bh = hash_obj(sb.outputs) if sb else None
        same = ah == bh
        if not same:
            diverged += 1
        ca = sa.cost_usd if sa else 0.0
        cb = sb.cost_usd if sb else 0.0
        step_diffs.append(
            {
                "step_id": sid,
                "kind": kind,
                "diverged": not same,
                "cost_delta_usd": round(cb - ca, 8),
                "a_outputs_hash": ah,
                "b_outputs_hash": bh,
            }
        )
    return {
        "a_total_cost_usd": round(a.total_cost_usd, 8),
        "b_total_cost_usd": round(b.total_cost_usd, 8),
        "total_cost_delta_usd": round(b.total_cost_usd - a.total_cost_usd, 8),
        "divergent_step_count": diverged,
        "step_diffs": step_diffs,
    }


def _jsonify(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, list):
        return [_jsonify(x) for x in v]
    if isinstance(v, tuple):
        return [_jsonify(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonify(x) for k, x in v.items()}
    if is_dataclass(v):
        return _jsonify(asdict(v))
    return v
