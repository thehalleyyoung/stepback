"""Typed substitutions.

Substitutions are the *only* way to alter a replay; arbitrary
monkey-patching is rejected. Every substitution carries an
``at_step`` selector (a `step_id` like ``"step:7"``) and an
``apply(inputs, recorded_step)`` method that mutates the per-step
inputs dict in place. The replay engine recomputes the inputs hash
afterwards and decides cache-hit vs. dirty.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .jsonpatch import apply_patch


class Substitution:
    """Base class. Subclasses implement :py:meth:`apply`.

    A substitution is one of two shapes:

    * *Input-mutating* (the default) — overrides :py:meth:`apply` to
      mutate the per-step ``inputs`` dict in place. The replay engine
      recomputes the inputs hash and decides cache-hit vs. dirty.
    * *Output-forcing* — overrides :py:meth:`is_output_forcing` to
      return ``True`` and :py:meth:`force_output` to return the
      replacement output. The step is marked dirty and the underlying
      tool / LLM is **not** invoked.
    """

    at_step: str

    def apply(self, inputs: dict, recorded_step: dict) -> None:  # pragma: no cover
        raise NotImplementedError

    def is_output_forcing(self) -> bool:
        """If True, replay calls :py:meth:`force_output` instead of
        running the underlying step. Default: False."""
        return False

    def force_output(self, recorded_step: dict) -> Any:  # pragma: no cover
        raise NotImplementedError

    def kind(self) -> str:
        return type(self).__name__


@dataclass
class PromptSubstitution(Substitution):
    at_step: str
    new_messages: List[dict]

    def apply(self, inputs: dict, recorded_step: dict) -> None:
        inputs["messages"] = list(self.new_messages)


@dataclass
class ModelSubstitution(Substitution):
    at_step: str
    new_model_id: str

    def apply(self, inputs: dict, recorded_step: dict) -> None:
        inputs["model"] = self.new_model_id


@dataclass
class ToolOutputSubstitution(Substitution):
    """Pin a tool call's output to ``fake_response``.

    Output-forcing: the step is marked dirty (its output differs from
    the recorded one), but the tool is *not* invoked — the fake output
    is returned directly. Any descendant whose inputs depend on this
    step's output will then itself become dirty.
    """

    at_step: str
    fake_response: Any
    tool_call_id: Optional[str] = None

    def apply(self, inputs: dict, recorded_step: dict) -> None:
        return  # no-op on inputs

    def is_output_forcing(self) -> bool:
        return True

    def force_output(self, recorded_step: dict) -> Any:
        return {"result": self.fake_response}


@dataclass
class PolicySubstitution(Substitution):
    at_step: str
    policy_path: str

    def apply(self, inputs: dict, recorded_step: dict) -> None:
        inputs["policy_path"] = self.policy_path


@dataclass
class RouterSubstitution(Substitution):
    at_step: str
    choice: str

    def apply(self, inputs: dict, recorded_step: dict) -> None:
        inputs["choice"] = self.choice


@dataclass
class SystemPromptSubstitution(Substitution):
    """Patch only the system message instead of the whole ``messages``.

    Modes:
        * ``replace`` (default) — set the first system message's
          content to ``system_text``; if no system message exists,
          insert one at index 0.
        * ``prepend`` — prepend ``system_text + "\\n\\n"`` to the
          first system message's content; insert if missing.
        * ``append`` — append ``"\\n\\n" + system_text``; insert if
          missing.
    """

    at_step: str
    system_text: str
    mode: str = "replace"

    def __post_init__(self) -> None:
        if self.mode not in ("replace", "prepend", "append"):
            raise ValueError(
                f"SystemPromptSubstitution.mode must be replace/prepend/append, "
                f"got {self.mode!r}"
            )

    def apply(self, inputs: dict, recorded_step: dict) -> None:
        msgs = list(inputs.get("messages") or [])
        msgs = [copy.deepcopy(m) for m in msgs]
        sys_idx = next(
            (i for i, m in enumerate(msgs) if isinstance(m, dict) and m.get("role") == "system"),
            None,
        )
        if sys_idx is None:
            msgs.insert(0, {"role": "system", "content": self.system_text})
        else:
            cur = msgs[sys_idx].get("content", "")
            if not isinstance(cur, str):
                cur = str(cur)
            if self.mode == "replace":
                msgs[sys_idx]["content"] = self.system_text
            elif self.mode == "prepend":
                msgs[sys_idx]["content"] = self.system_text + "\n\n" + cur
            else:  # append
                msgs[sys_idx]["content"] = cur + "\n\n" + self.system_text
        inputs["messages"] = msgs


@dataclass
class MessagePatchSubstitution(Substitution):
    """Replace one message at ``index`` with ``new_message``.

    ``index`` may be negative (Python semantics, ``-1`` = last).
    ``index == len(messages)`` appends. Out-of-range otherwise raises
    :class:`IndexError`.
    """

    at_step: str
    index: int
    new_message: dict

    def __post_init__(self) -> None:
        if not isinstance(self.new_message, dict):
            raise ValueError("MessagePatchSubstitution.new_message must be a dict")
        if "role" not in self.new_message or "content" not in self.new_message:
            raise ValueError(
                "MessagePatchSubstitution.new_message must include 'role' and 'content'"
            )
        if not isinstance(self.index, int):
            raise ValueError("MessagePatchSubstitution.index must be int")

    def apply(self, inputs: dict, recorded_step: dict) -> None:
        msgs = [copy.deepcopy(m) for m in (inputs.get("messages") or [])]
        n = len(msgs)
        if self.index == n:
            msgs.append(copy.deepcopy(self.new_message))
        else:
            i = self.index
            if i < 0:
                i = n + i
            if i < 0 or i >= n:
                raise IndexError(
                    f"MessagePatchSubstitution index {self.index} out of range "
                    f"for messages of length {n}"
                )
            msgs[i] = copy.deepcopy(self.new_message)
        inputs["messages"] = msgs


@dataclass
class SamplingSubstitution(Substitution):
    """Set sampling knobs on an LLM call: temperature / top_p /
    max_tokens / seed. Only non-None fields are written; pre-existing
    keys with other names are left untouched.
    """

    at_step: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.temperature is not None:
            if not isinstance(self.temperature, (int, float)) or isinstance(self.temperature, bool):
                raise ValueError("SamplingSubstitution.temperature must be a number")
            if not (0.0 <= float(self.temperature) <= 2.0):
                raise ValueError(
                    f"SamplingSubstitution.temperature out of range [0,2]: {self.temperature}"
                )
        if self.top_p is not None:
            if not isinstance(self.top_p, (int, float)) or isinstance(self.top_p, bool):
                raise ValueError("SamplingSubstitution.top_p must be a number")
            if not (0.0 <= float(self.top_p) <= 1.0):
                raise ValueError(
                    f"SamplingSubstitution.top_p out of range [0,1]: {self.top_p}"
                )
        if self.max_tokens is not None:
            if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool):
                raise ValueError("SamplingSubstitution.max_tokens must be int")
            if self.max_tokens <= 0:
                raise ValueError(
                    f"SamplingSubstitution.max_tokens must be positive: {self.max_tokens}"
                )
        if self.seed is not None:
            if not isinstance(self.seed, int) or isinstance(self.seed, bool):
                raise ValueError("SamplingSubstitution.seed must be int")

    def apply(self, inputs: dict, recorded_step: dict) -> None:
        if self.temperature is not None:
            inputs["temperature"] = float(self.temperature)
        if self.top_p is not None:
            inputs["top_p"] = float(self.top_p)
        if self.max_tokens is not None:
            inputs["max_tokens"] = int(self.max_tokens)
        if self.seed is not None:
            inputs["seed"] = int(self.seed)


@dataclass
class ToolArgumentsSubstitution(Substitution):
    """For ``tool_call`` steps: replace ``inputs["arguments"]``.

    The tool actually runs again because the inputs hash changes.
    """

    at_step: str
    new_arguments: Any

    def apply(self, inputs: dict, recorded_step: dict) -> None:
        inputs["arguments"] = copy.deepcopy(self.new_arguments)


@dataclass
class InputsPatchSubstitution(Substitution):
    """Apply a JSON Patch (RFC 6902 subset) to the per-step inputs.

    See :mod:`stepback.jsonpatch` for the supported ops.
    """

    at_step: str
    ops: List[dict] = field(default_factory=list)

    def apply(self, inputs: dict, recorded_step: dict) -> None:
        patched = apply_patch(inputs, list(self.ops))
        if not isinstance(patched, dict):
            raise ValueError(
                "InputsPatchSubstitution must yield a dict (root of inputs); "
                f"got {type(patched).__name__}"
            )
        inputs.clear()
        inputs.update(patched)


@dataclass
class OutputsPatchSubstitution(Substitution):
    """Output-forcing: apply a JSON Patch to the recorded outputs."""

    at_step: str
    ops: List[dict] = field(default_factory=list)

    def apply(self, inputs: dict, recorded_step: dict) -> None:
        return  # no-op on inputs

    def is_output_forcing(self) -> bool:
        return True

    def force_output(self, recorded_step: dict) -> Any:
        outputs = recorded_step.get("outputs")
        return apply_patch({} if outputs is None else outputs, list(self.ops))


@dataclass
class FieldOutputSubstitution(Substitution):
    """Output-forcing: update specific top-level fields in a step's recorded output.

    All top-level fields in the recorded output NOT listed in ``field_updates``
    are preserved unchanged.  This is the companion to the ``context_fields``
    recorder parameter: by substituting only the fields a particular downstream
    step depends on, you can make that downstream step dirty while keeping
    other downstream steps — whose ``context_fields`` declarations do not
    reference the changed fields — as cache hits.

    Example::

        # Record step A producing {"fast_result": 1, "slow_result": 2}.
        # Step B was recorded with context_fields=["fast_result"].
        # Step C was recorded with context_fields=["slow_result"].
        #
        # Substitute only slow_result → B stays clean, C goes dirty.
        trace.substitute(FieldOutputSubstitution("step:1", {"slow_result": 99}))
        result = trace.replay_forward(executor)
        # result: B is cache_hit=True, C is dirty=True

    Parameters
    ----------
    at_step : str
        The ``step_id`` of the step whose output fields are replaced.
    field_updates : dict
        Mapping of top-level output field name → new value.  Must be
        non-empty.  Keys that do not exist in the recorded output are
        added (forward-compatible extension).
    """

    at_step: str
    field_updates: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.field_updates, dict):
            raise ValueError("FieldOutputSubstitution.field_updates must be a dict")
        if not self.field_updates:
            raise ValueError("FieldOutputSubstitution.field_updates must be non-empty")

    def apply(self, inputs: dict, recorded_step: dict) -> None:
        return  # no-op on inputs

    def is_output_forcing(self) -> bool:
        return True

    def force_output(self, recorded_step: dict) -> Any:
        recorded_outputs = recorded_step.get("outputs") or {}
        return {**recorded_outputs, **self.field_updates}


@dataclass
class RaiseSubstitution(Substitution):
    """Output-forcing: pretend the step raised ``exception_type``.

    The step's output is replaced with::

        {"__error__": {"type": exception_type, "message": message}}

    so descendants can branch on the error sentinel without the
    replay engine itself blowing up. Cost is treated as 0.0 (no
    ``usage`` field on the synthetic output).
    """

    at_step: str
    exception_type: str
    message: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.exception_type, str) or not self.exception_type:
            raise ValueError("RaiseSubstitution.exception_type must be a non-empty str")
        if not isinstance(self.message, str):
            raise ValueError("RaiseSubstitution.message must be a str")

    def apply(self, inputs: dict, recorded_step: dict) -> None:
        return

    def is_output_forcing(self) -> bool:
        return True

    def force_output(self, recorded_step: dict) -> Any:
        return {"__error__": {"type": self.exception_type, "message": self.message}}


@dataclass
class SubstitutionSet:
    """A bundle of substitutions, indexed by ``at_step`` for fast lookup."""

    items: List[Substitution] = field(default_factory=list)
    _by_step: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_step = {}
        for s in self.items:
            self._by_step.setdefault(s.at_step, []).append(s)

    def at(self, step_id: str) -> List[Substitution]:
        return list(self._by_step.get(step_id, ()))

    def add(self, sub: Substitution) -> "SubstitutionSet":
        self.items.append(sub)
        self._by_step.setdefault(sub.at_step, []).append(sub)
        return self

    def __bool__(self) -> bool:
        return bool(self.items)
