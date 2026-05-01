"""Typed substitutions.

Substitutions are the *only* way to alter a replay; arbitrary
monkey-patching is rejected. Every substitution carries an
``at_step`` selector (a `step_id` like ``"step:7"``) and an
``apply(inputs, recorded_step)`` method that mutates the per-step
inputs dict in place. The replay engine recomputes the inputs hash
afterwards and decides cache-hit vs. dirty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


class Substitution:
    """Base class. Subclasses implement :py:meth:`apply`."""

    at_step: str

    def apply(self, inputs: dict, recorded_step: dict) -> None:  # pragma: no cover
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

    Unique among substitutions in that it forces the *output* of a
    step rather than mutating its input. The replay engine handles
    this specially: the step is marked dirty (its output differs from
    the recorded one), but the tool is *not* invoked — the fake output
    is returned directly. Any descendant whose inputs depend on this
    step's output will then itself become dirty.
    """

    at_step: str
    fake_response: Any
    tool_call_id: Optional[str] = None

    def apply(self, inputs: dict, recorded_step: dict) -> None:
        return  # no-op on inputs


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
