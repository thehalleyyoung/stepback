"""In-process recorder shim.

`record(path)` is a context manager returning a `Recorder`. The
recorder exposes three primitives — `llm_call(...)`, `tool_call(...)`,
and `router(...)` — which any agent can call directly. Recorders also
have an `exception(...)` primitive for failed steps.

For OpenAI users a thin `wrap_openai(client)` helper is planned for
m1; v0.1 keeps the surface small and explicit so the substitution +
replay semantics are testable in isolation.

**Async support.** Use :func:`arecord` as an async context manager so
``await``-chains and ``asyncio.TaskGroup`` children can access the
recorder via :func:`get_current_recorder`.  Each spawned task inherits
a snapshot of the ambient recorder and task-local parent step id at
creation time; subsequent ``await``-yields do not lose those bindings.
"""
from __future__ import annotations

import contextvars
import os
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Iterator, List, Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .canonical import canonical_json, hash_obj, sha256_hex
from .nondeterminism import combine_nondeterminism, model_sampling_nondeterminism
from .pricing import compute_cost
from .trace_writer import TraceWriter


# ---------------------------------------------------------------------------
# Context-variable storage for async propagation (Step 98).
#
# _RECORDER_VAR  — the active Recorder for the current asyncio Task (or
#                  sync call chain). Set by record() and arecord().
# _PARENT_STEP_VAR — the current parent step id, task-local so concurrent
#                  asyncio tasks each maintain an independent parent chain
#                  even when they share a single Recorder object.
# ---------------------------------------------------------------------------

_RECORDER_VAR: contextvars.ContextVar[Optional["Recorder"]] = contextvars.ContextVar(
    "stepback_recorder", default=None
)
_PARENT_STEP_VAR: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "stepback_parent_step", default=None
)


def get_current_recorder() -> Optional["Recorder"]:
    """Return the ambient :class:`Recorder` for the current task/context.

    Returns ``None`` when called outside an active :func:`record` or
    :func:`arecord` block (or :func:`stepback.autorecord.enable` /
    :func:`stepback.autorecord.aenable`).

    In async code the recorder is propagated via Python's
    :class:`contextvars.ContextVar`, so it is always available after
    ``await`` calls and in child tasks spawned by
    ``asyncio.create_task`` / ``asyncio.TaskGroup``.  Each task also
    carries a *task-local* parent step id, so parallel branches record
    the correct parent without races.
    """
    return _RECORDER_VAR.get()


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
    _options: Optional["RecorderOptions"] = field(default=None, repr=False)  # type: ignore[name-defined]
    _dropped_steps: int = field(default=0, repr=False)
    _recording_errors: List[Exception] = field(default_factory=list, repr=False)
    _null_mode: bool = field(default=False, repr=False)

    @property
    def dropped_steps(self) -> int:
        """Number of steps not written to disk due to overflow or null-mode."""
        return self._dropped_steps

    @property
    def recording_errors(self) -> List[Exception]:
        """Copy of errors silently absorbed when ``mandatory=False``."""
        return list(self._recording_errors)

    # ------------------------------------------------------------ ids
    def _new_id(self) -> str:
        self._counter += 1
        return f"step:{self._counter}"

    # --------------------------------------------------------- async parent
    def _current_parent(self) -> Optional[str]:
        """Return the effective parent step id.

        In async code each task maintains a task-local copy of the parent
        via :data:`_PARENT_STEP_VAR`, so concurrent tasks do not corrupt
        each other's parent chains even when sharing a :class:`Recorder`.
        Falls back to the synchronous ``_parent`` attribute when no
        task-local value has been set (sync path or the first step in a
        task that was created before any step was recorded).
        """
        cv = _PARENT_STEP_VAR.get()
        if cv is not None:
            return cv
        return self._parent

    # --------------------------------------------------------- helpers
    def _record(self, step: dict) -> dict:
        step["inputs_hash"] = hash_obj(step["inputs"])
        step["outputs_hash"] = hash_obj(step["outputs"])
        step.setdefault(
            "nondeterminism_hash",
            sha256_hex(canonical_json(step.get("nondeterminism", {}))),
        )
        step.setdefault("wallclock_ns", time.time_ns())

        # Null mode: just track steps in memory, don't write to disk
        if self._null_mode:
            self._dropped_steps += 1
            self.steps.append(step)
            sid = step["step_id"]
            self._parent = sid
            _PARENT_STEP_VAR.set(sid)
            return step

        # max_queue_depth overflow check (options-gated)
        opts = self._options
        if (
            opts is not None
            and opts.max_queue_depth > 0
            and len(self.steps) >= opts.max_queue_depth
        ):
            if opts.overflow_policy == "drop":
                # Switch to null mode for remaining steps
                self._null_mode = True
                self._dropped_steps += 1
                self.steps.append(step)
                sid = step["step_id"]
                self._parent = sid
                _PARENT_STEP_VAR.set(sid)
                return step
            # overflow_policy == "block": fall through to normal write

        # Normal write path with optional fail-open wrapping
        mandatory = opts.mandatory if opts is not None else True
        try:
            self.writer.write_step(step)
        except Exception as exc:
            if mandatory:
                raise
            self._recording_errors.append(exc)
            self._null_mode = True
            self._dropped_steps += 1
            self.steps.append(step)
            sid = step["step_id"]
            self._parent = sid
            _PARENT_STEP_VAR.set(sid)
            return step

        self.steps.append(step)
        sid = step["step_id"]
        # Keep self._parent in sync for the synchronous parallel() helper
        # and backward-compatible sync code that reads it directly.
        self._parent = sid
        # Also advance the task-local ContextVar so async tasks maintain
        # their own independent parent chain.
        _PARENT_STEP_VAR.set(sid)
        return step

    def _parent_outputs_hash(self) -> Optional[str]:
        """Return the recorded outputs_hash of the current ``_parent`` step,
        or None if no parent. Looks up by id rather than relying on
        ``self.steps[-1]``, because parallel-branch recording temporarily
        re-parents to the open frame even though the most recently
        appended step belongs to a sibling branch.
        """
        parent = self._current_parent()
        if parent is None:
            return None
        for s in reversed(self.steps):
            if s["step_id"] == parent:
                return s["outputs_hash"]
        return None

    def _parent_outputs(self) -> Optional[dict]:
        """Return the recorded outputs of the current ``_parent`` step,
        or None if no parent.  Used to compute field-level context hashes
        when ``context_fields`` is supplied to a recorder primitive.
        """
        parent = self._current_parent()
        if parent is None:
            return None
        for s in reversed(self.steps):
            if s["step_id"] == parent:
                return s.get("outputs")
        return None

    def _build_partial_context(
        self, context_fields: List[str]
    ) -> "tuple[str, List[str]]":
        """Compute a partial context hash from specific output fields of the parent.

        Returns ``(partial_context_hash, context_fields)`` for embedding in
        step inputs.  Raises :class:`ValueError` if any declared field is
        absent from the parent's outputs (early validation so mismatches
        don't produce silent false cache hits at replay time).
        """
        parent_output = self._parent_outputs()
        if parent_output is None:
            raise ValueError(
                "context_fields requires a parent step but no parent is set"
            )
        missing = [k for k in context_fields if k not in parent_output]
        if missing:
            raise ValueError(
                f"context_fields declares fields not present in parent output: "
                f"{missing!r}.  Available top-level keys: "
                f"{sorted(parent_output.keys())!r}"
            )
        partial = {k: parent_output[k] for k in context_fields}
        return hash_obj(partial), list(context_fields)

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
        safety_settings: Optional[list] = None,
        context_from_parent: bool = True,
        context_fields: Optional[List[str]] = None,
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

        ``context_fields`` enables partial recompute (Step 60): instead
        of hashing the parent's *entire* output, only the listed top-level
        field names are included in the context hash.  The replay engine
        then re-hashes only those fields, so a downstream step stays clean
        when the parent's output changes but the declared fields do not.
        Requires ``context_from_parent=True`` and a parent step to be
        active; raises :class:`ValueError` if any declared field is missing
        from the parent's output at record time.

        The optional *safety_settings* parameter records provider-level
        content-safety configuration (e.g. Gemini ``HarmCategory`` thresholds).
        When non-``None`` it is included in the inputs hash so that calls
        with different safety policies produce distinct cache keys.
        """
        if context_fields and self._current_parent() is None:
            raise ValueError(
                "context_fields requires a parent step but no parent is set"
            )
        sid = self._new_id()
        request: dict = {
            "model": model,
            "temperature": temperature,
            "seed": seed,
            "messages": messages,
            "tools": tools,
            "response_format": response_format,
        }
        if safety_settings is not None:
            request["safety_settings"] = safety_settings
        response = executor(model, messages)
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        cost = compute_cost(model, usage)
        inputs = {"kind": "llm_call", **request}
        if context_from_parent and self._current_parent() is not None:
            if context_fields:
                ctx_hash, cf = self._build_partial_context(context_fields)
                inputs["context"] = ctx_hash
                inputs["_stepback_context_fields"] = cf
            else:
                inputs["context"] = self._parent_outputs_hash()
        step = {
            "step_id": sid,
            "step_kind": "llm_call",
            "name": model,
            "parent_step_id": self._current_parent(),
            "inputs": inputs,
            "outputs": response,
            "llm_request": request,
            "llm_response": response,
            "cost_usd": cost,
            # Auto-record model-sampling nondeterminism (Step 63).
            "nondeterminism": combine_nondeterminism(
                model_sampling_nondeterminism(temperature=temperature, seed=seed)
            ),
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
        context_fields: Optional[List[str]] = None,
    ) -> dict:
        if context_fields and self._current_parent() is None:
            raise ValueError(
                "context_fields requires a parent step but no parent is set"
            )
        sid = self._new_id()
        result = executor(name, arguments)
        inputs: dict[str, Any] = {"kind": "tool_call", "name": name, "arguments": arguments}
        if context_from_parent and self._current_parent() is not None:
            if context_fields:
                ctx_hash, cf = self._build_partial_context(context_fields)
                inputs["context"] = ctx_hash
                inputs["_stepback_context_fields"] = cf
            else:
                inputs["context"] = self._parent_outputs_hash()
        step = {
            "step_id": sid,
            "step_kind": "tool_call",
            "name": name,
            "parent_step_id": self._current_parent(),
            "inputs": inputs,
            "outputs": {"result": result},
            "cost_usd": 0.0,
        }
        return self._record(step)

    # ----------------------------------------------------------- router
    def router(
        self,
        name: str,
        choice: str,
        options: List[str],
        *,
        context_fields: Optional[List[str]] = None,
    ) -> dict:
        sid = self._new_id()
        inputs: dict[str, Any] = {"kind": "router", "name": name, "options": options}
        if self._current_parent() is not None:
            if context_fields:
                ctx_hash, cf = self._build_partial_context(context_fields)
                inputs["context"] = ctx_hash
                inputs["_stepback_context_fields"] = cf
            else:
                inputs["context"] = self._parent_outputs_hash()
        step = {
            "step_id": sid,
            "step_kind": "router",
            "name": name,
            "parent_step_id": self._current_parent(),
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
        open_parent = self._current_parent()
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
            # Keep _PARENT_STEP_VAR in sync so _current_parent() sees the
            # re-parented value inside each branch's synchronous call.
            _PARENT_STEP_VAR.set(sid_open)
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
            "parent_step_id": self._current_parent(),
            "inputs": {"kind": "exception"},
            "outputs": {"error_class": error_class, "message": message},
            "cost_usd": 0.0,
        }
        return self._record(step)


@contextmanager
def record(path: str, *, key: Optional[RecorderKey] = None, options: Optional["RecorderOptions"] = None, signing: bool = True, batch_sign: bool = False, batch_sign_workers: int = 0, batch_sign_interval: Optional[int] = None) -> Iterator["Recorder"]:  # type: ignore[name-defined]
    """Open ``path`` for writing as a `.sb` trace and yield a `Recorder`.

    Also sets the ambient :data:`_RECORDER_VAR` ContextVar so that
    :func:`get_current_recorder` returns this recorder for the duration
    of the ``with`` block.

    Usage::

        with record("./trace.sb") as rec:
            rec.llm_call("gpt-4o", [...], executor=my_llm)
            rec.tool_call("lookup", {...}, executor=my_tool)
    """
    from .backpressure import _NullTraceWriter  # noqa: PLC0415

    key = key or RecorderKey.fresh()
    mandatory = options.mandatory if options is not None else True

    # Sampling decision
    if options is not None and not options._should_sample():
        null_writer = _NullTraceWriter()
        rec = Recorder(writer=null_writer, key=key, _options=options, _null_mode=True)
        rec_token = _RECORDER_VAR.set(rec)
        parent_token = _PARENT_STEP_VAR.set(None)
        try:
            yield rec
        finally:
            _RECORDER_VAR.reset(rec_token)
            _PARENT_STEP_VAR.reset(parent_token)
        return

    # Normal path: open the writer (with optional fail-open)
    from .trace_writer import DEFAULT_BATCH_SIGN_INTERVAL as _DBS  # noqa: PLC0415
    writer_kwargs: dict = dict(
        hmac_key=key.hmac_key,
        signing_key=key.signing_key,
        signing=signing,
        batch_sign=batch_sign,
        batch_sign_workers=batch_sign_workers,
    )
    if batch_sign_interval is not None:
        writer_kwargs["batch_sign_interval"] = batch_sign_interval
    try:
        writer = TraceWriter.open(path, **writer_kwargs)
    except Exception as exc:
        if mandatory:
            raise
        null_writer = _NullTraceWriter()
        rec = Recorder(writer=null_writer, key=key, _options=options, _null_mode=True)
        rec._recording_errors.append(exc)
        rec_token = _RECORDER_VAR.set(rec)
        parent_token = _PARENT_STEP_VAR.set(None)
        try:
            yield rec
        finally:
            _RECORDER_VAR.reset(rec_token)
            _PARENT_STEP_VAR.reset(parent_token)
        return

    rec = Recorder(writer=writer, key=key, _options=options)
    rec_token = _RECORDER_VAR.set(rec)
    parent_token = _PARENT_STEP_VAR.set(None)
    try:
        yield rec
    finally:
        _RECORDER_VAR.reset(rec_token)
        _PARENT_STEP_VAR.reset(parent_token)
        try:
            writer.close()
        except Exception as exc:
            if mandatory:
                raise
            rec._recording_errors.append(exc)


@asynccontextmanager
async def arecord(path: str, *, key: Optional[RecorderKey] = None, options: Optional["RecorderOptions"] = None, signing: bool = True, batch_sign: bool = False, batch_sign_workers: int = 0, batch_sign_interval: Optional[int] = None) -> AsyncIterator["Recorder"]:  # type: ignore[name-defined]
    """Async context manager: open ``path`` and yield a :class:`Recorder`.

    Unlike the synchronous :func:`record`, this variant sets the ambient
    :data:`_RECORDER_VAR` and :data:`_PARENT_STEP_VAR` ContextVars so
    that the recorder and task-local parent step id survive across
    ``await`` boundaries and are copied into child tasks spawned with
    ``asyncio.create_task`` or ``asyncio.TaskGroup``.

    Usage::

        async def my_agent():
            async with arecord("./trace.sb") as rec:
                client = wrap_openai_async(AsyncOpenAI(), rec,
                                           default_model="gpt-4o-mini-2024-07-18")
                resp = await client.chat.completions.create(messages=[...])

    **Parallel tasks.** Each task spawned inside the ``async with`` block
    inherits a snapshot of the recorder and parent at creation time.
    Because :data:`_PARENT_STEP_VAR` is task-local, concurrent branches
    maintain independent parent chains even when they share the same
    :class:`Recorder` object::

        async with arecord("./trace.sb") as rec:
            step1 = rec.tool_call("fetch", {}, executor=...)
            async with asyncio.TaskGroup() as tg:
                tg.create_task(branch_a(rec))  # parent_step_id = step1
                tg.create_task(branch_b(rec))  # parent_step_id = step1  (independent)

    **Lifetime.** Do not allow tasks to outlive the ``async with`` block;
    a task that holds a reference to the closed :class:`Recorder` and
    attempts to record a step will receive an error from the underlying
    :class:`~stepback.trace_writer.TraceWriter`.
    """
    from .backpressure import _NullTraceWriter  # noqa: PLC0415

    key = key or RecorderKey.fresh()
    mandatory = options.mandatory if options is not None else True

    # Sampling decision
    if options is not None and not options._should_sample():
        null_writer = _NullTraceWriter()
        rec = Recorder(writer=null_writer, key=key, _options=options, _null_mode=True)
        rec_token = _RECORDER_VAR.set(rec)
        parent_token = _PARENT_STEP_VAR.set(None)
        try:
            yield rec
        finally:
            _RECORDER_VAR.reset(rec_token)
            _PARENT_STEP_VAR.reset(parent_token)
        return

    # Normal path with optional fail-open
    writer_kwargs_a: dict = dict(
        hmac_key=key.hmac_key,
        signing_key=key.signing_key,
        signing=signing,
        batch_sign=batch_sign,
        batch_sign_workers=batch_sign_workers,
    )
    if batch_sign_interval is not None:
        writer_kwargs_a["batch_sign_interval"] = batch_sign_interval
    try:
        writer = TraceWriter.open(path, **writer_kwargs_a)
    except Exception as exc:
        if mandatory:
            raise
        null_writer = _NullTraceWriter()
        rec = Recorder(writer=null_writer, key=key, _options=options, _null_mode=True)
        rec._recording_errors.append(exc)
        rec_token = _RECORDER_VAR.set(rec)
        parent_token = _PARENT_STEP_VAR.set(None)
        try:
            yield rec
        finally:
            _RECORDER_VAR.reset(rec_token)
            _PARENT_STEP_VAR.reset(parent_token)
        return

    rec = Recorder(writer=writer, key=key, _options=options)
    rec_token = _RECORDER_VAR.set(rec)
    parent_token = _PARENT_STEP_VAR.set(None)
    try:
        yield rec
    finally:
        _RECORDER_VAR.reset(rec_token)
        _PARENT_STEP_VAR.reset(parent_token)
        try:
            writer.close()
        except Exception as exc:
            if mandatory:
                raise
            rec._recording_errors.append(exc)
