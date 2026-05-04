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
from typing import Any, Callable, Iterator, List, Optional

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

    def _parent_outputs_hash(self) -> Optional[str]:
        """Return the recorded outputs_hash of the current ``_parent`` step,
        or None if no parent. Looks up by id rather than relying on
        ``self.steps[-1]``, because parallel-branch recording temporarily
        re-parents to the open frame even though the most recently
        appended step belongs to a sibling branch.
        """
        if self._parent is None:
            return None
        for s in reversed(self.steps):
            if s["step_id"] == self._parent:
                return s["outputs_hash"]
        return None

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
            inputs["context"] = self._parent_outputs_hash()
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
        inputs: dict[str, Any] = {"kind": "tool_call", "name": name, "arguments": arguments}
        if context_from_parent and self._parent is not None:
            inputs["context"] = self._parent_outputs_hash()
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
        inputs: dict[str, Any] = {"kind": "router", "name": name, "options": options}
        if self._parent is not None:
            inputs["context"] = self._parent_outputs_hash()
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

    # ----------------------------------------------------- parallel
    def parallel(
        self,
        name: str,
        branches: List[Callable[["Recorder"], Any]],
        *,
        join: Optional[Callable[[List[Any]], Any]] = None,
        branch_names: Optional[List[str]] = None,
    ) -> dict:
        """Record a fan-out / fan-in across ``branches``.

        Emits one ``parallel_branch_open`` frame, then sequentially
        drives each branch closure (each closure receives this same
        recorder, with ``_parent`` rebracketed to the open frame so
        every step it appends becomes a child of the open). Then
        emits one ``parallel_branch_join`` frame whose ``parent_step_id``
        is the open frame and whose ``parent_step_ids`` lists each
        branch's tail step.

        The join's ``inputs`` carry a ``branch_tail_hashes`` field bound
        to each tail's recorded ``outputs_hash``. The replay engine
        rebinds those to current tail output hashes, so a substitution
        inside any branch propagates dirtiness through the join (and
        only through the join — sibling branches stay cached).

        ``join`` is an optional callable receiving the list of branch
        tail outputs and returning the join's ``outputs`` payload.
        Defaults to ``{"branches": [...tail_outputs...]}``.
        Sequential execution today; the on-disk frames are explicitly
        marked parallel so a future scheduler can re-execute branches
        concurrently without changing the trace shape.
        """
        if branch_names is not None and len(branch_names) != len(branches):
            raise ValueError("branch_names length must match branches length")
        bnames = list(branch_names) if branch_names else [
            f"{name}/branch_{i}" for i in range(len(branches))
        ]
        open_parent = self._parent
        sid_open = self._new_id()
        open_inputs = {
            "kind": "parallel_branch_open",
            "name": name,
            "branch_names": bnames,
            "branch_count": len(branches),
        }
        if open_parent is not None:
            open_inputs["context"] = self._parent_outputs_hash()
        open_step = {
            "step_id": sid_open,
            "step_kind": "parallel_branch_open",
            "name": name,
            "parent_step_id": open_parent,
            "inputs": open_inputs,
            "outputs": {"branch_names": bnames, "branch_count": len(branches)},
            "cost_usd": 0.0,
        }
        self._record(open_step)

        branch_tails: List[str] = []
        branch_tail_outputs: List[Any] = []
        for fn in branches:
            self._parent = sid_open
            fn(self)
            tail_id = self._parent
            if tail_id == sid_open:
                raise RuntimeError(
                    "parallel branch produced no steps; "
                    "every branch must record at least one step"
                )
            branch_tails.append(tail_id)
            branch_tail_outputs.append(self.steps[-1]["outputs"])

        join_outputs = (
            join(branch_tail_outputs)
            if join is not None
            else {"branches": branch_tail_outputs}
        )

        sid_join = self._new_id()
        # Build a position-stable mapping of tail_id -> recorded outputs_hash.
        tail_hashes = {}
        for tid in branch_tails:
            for s in self.steps:
                if s["step_id"] == tid:
                    tail_hashes[tid] = s["outputs_hash"]
                    break
        join_inputs = {
            "kind": "parallel_branch_join",
            "name": name,
            "open_step_id": sid_open,
            "branch_tails": list(branch_tails),
            "branch_tail_hashes": [tail_hashes[t] for t in branch_tails],
        }
        join_step = {
            "step_id": sid_join,
            "step_kind": "parallel_branch_join",
            "name": name,
            "parent_step_id": sid_open,
            "parent_step_ids": list(branch_tails),
            "inputs": join_inputs,
            "outputs": join_outputs,
            "cost_usd": 0.0,
        }
        self._record(join_step)
        return join_step

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
def record(path: str, *, key: Optional[RecorderKey] = None) -> Iterator["Recorder"]:
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
