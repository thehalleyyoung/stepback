"""In-process recorder shim.

`record(path)` is a context manager returning a `Recorder`. The
recorder exposes three primitives — `llm_call(...)`, `tool_call(...)`,
and `router(...)` — which any agent can call directly. Recorders also
have an `exception(...)` primitive for failed steps.

For OpenAI users a thin `wrap_openai(client)` helper is planned for
m1; v0.1 keeps the surface small and explicit so the substitution +
replay semantics are testable in isolation.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .canonical import canonical_json, hash_obj, sha256_hex
from .pricing import compute_cost
from .trace_writer import TraceWriter


@dataclass
class RecorderKey:
    """Bundle of per-trace HMAC key + Ed25519 signing key.

    Tests need to read the keys back to verify; production code
    typically reads them from a KMS-backed secret.
    """

    hmac_key: bytes
    signing_key: Ed25519PrivateKey

    @classmethod
    def fresh(cls) -> "RecorderKey":
        return cls(hmac_key=os.urandom(32), signing_key=Ed25519PrivateKey.generate())


@dataclass
class Recorder:
    writer: TraceWriter
    key: RecorderKey
    steps: List[dict] = field(default_factory=list)
    _counter: int = 0
    _parent: Optional[str] = None

    # ------------------------------------------------------------ ids
    def _new_id(self) -> str:
        self._counter += 1
        return f"step:{self._counter}"

    # --------------------------------------------------------- helpers
    def _record(self, step: dict) -> dict:
        step["inputs_hash"] = hash_obj(step["inputs"])
        step["outputs_hash"] = hash_obj(step["outputs"])
        step.setdefault(
            "nondeterminism_hash",
            sha256_hex(canonical_json(step.get("nondeterminism", {}))),
        )
        step.setdefault("wallclock_ns", time.time_ns())
        self.writer.write_step(step)
        self.steps.append(step)
        self._parent = step["step_id"]
        return step

    # --------------------------------------------------------- llm_call
    def llm_call(
        self,
        model: str,
        messages: List[dict],
        executor: Callable[[str, List[dict]], dict],
        *,
        temperature: float = 0.0,
        seed: Optional[int] = 42,
        tools: Optional[list] = None,
        response_format: Optional[dict] = None,
        context_from_parent: bool = True,
    ) -> dict:
        """Execute ``executor(model, messages)`` and record an ``llm_call`` step.

        The executor must return a dict shaped like an OpenAI chat
        completion (``{"choices": [...], "usage": {...}}``); a
        synthesised fake LLM is fine for tests.

        When ``context_from_parent`` is true (the default), the
        recorded inputs include a ``"context"`` field bound to the
        previous step's output hash. The replay engine uses this as
        the data-dependency edge: if the parent's output changes, this
        step's inputs hash changes, and the step becomes dirty.
        """
        sid = self._new_id()
        request = {
            "model": model,
            "temperature": temperature,
            "seed": seed,
            "messages": messages,
            "tools": tools,
            "response_format": response_format,
        }
        response = executor(model, messages)
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        cost = compute_cost(model, usage)
        inputs = {"kind": "llm_call", **request}
        if context_from_parent and self._parent is not None:
            inputs["context"] = self.steps[-1]["outputs_hash"]
        step = {
            "step_id": sid,
            "step_kind": "llm_call",
            "name": model,
            "parent_step_id": self._parent,
            "inputs": inputs,
            "outputs": response,
            "llm_request": request,
            "llm_response": response,
            "cost_usd": cost,
        }
        return self._record(step)

    # --------------------------------------------------------- tool_call
    def tool_call(
        self,
        name: str,
        arguments: dict,
        executor: Callable[[str, dict], Any],
        *,
        context_from_parent: bool = True,
    ) -> dict:
        sid = self._new_id()
        result = executor(name, arguments)
        inputs = {"kind": "tool_call", "name": name, "arguments": arguments}
        if context_from_parent and self._parent is not None:
            inputs["context"] = self.steps[-1]["outputs_hash"]
        step = {
            "step_id": sid,
            "step_kind": "tool_call",
            "name": name,
            "parent_step_id": self._parent,
            "inputs": inputs,
            "outputs": {"result": result},
            "cost_usd": 0.0,
        }
        return self._record(step)

    # ----------------------------------------------------------- router
    def router(self, name: str, choice: str, options: List[str]) -> dict:
        sid = self._new_id()
        inputs = {"kind": "router", "name": name, "options": options}
        if self._parent is not None:
            inputs["context"] = self.steps[-1]["outputs_hash"]
        step = {
            "step_id": sid,
            "step_kind": "router",
            "name": name,
            "parent_step_id": self._parent,
            "inputs": inputs,
            "outputs": {"choice": choice},
            "cost_usd": 0.0,
        }
        return self._record(step)

    # ------------------------------------------------------- exception
    def exception(self, error_class: str, message: str) -> dict:
        sid = self._new_id()
        step = {
            "step_id": sid,
            "step_kind": "exception",
            "name": error_class,
            "parent_step_id": self._parent,
            "inputs": {"kind": "exception"},
            "outputs": {"error_class": error_class, "message": message},
            "cost_usd": 0.0,
        }
        return self._record(step)


@contextmanager
def record(path: str, *, key: Optional[RecorderKey] = None):
    """Open ``path`` for writing as a `.sb` trace and yield a `Recorder`.

    Usage::

        with record("./trace.sb") as rec:
            rec.llm_call("gpt-4o", [...], executor=my_llm)
            rec.tool_call("lookup", {...}, executor=my_tool)
    """
    key = key or RecorderKey.fresh()
    writer = TraceWriter.open(path, hmac_key=key.hmac_key, signing_key=key.signing_key)
    rec = Recorder(writer=writer, key=key)
    try:
        yield rec
    finally:
        writer.close()
