"""Replay engine.

The whole point: walk the recorded trace in topological order and, for
each step, decide cache-hit vs. dirty. A step is a *cache hit* iff:

* its current inputs (post-substitution and post-parent-rebinding)
  hash to the recorded ``inputs_hash``, and
* no ancestor it depends on was dirty, and
* its recorded ``nondeterminism_hash`` is unchanged, and
* its :class:`~stepback.nondeterminism.NondeterminismClass` does not
  force re-execution (Step 63).

Otherwise it is dirty: the engine invokes the user-supplied
:py:class:`Executor` to recompute it, and propagates dirtiness to
descendants whose inputs depend on this step's output.

Dependency model for v0.1 is intentionally conservative: each step's
``"context"`` field is rebound from the *current* parent output hash
before its hash is computed. This is enough to model the common chat
agent loop where each step reads the previous step's result. (m4+
will introduce explicit input-pointer recording for non-tree
dependencies.)
"""
from __future__ import annotations

import copy
import datetime
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple

from .canonical import canonical_json, hash_obj, sha256_hex
from .nondeterminism import forces_dirty as _nondet_forces_dirty
from .pricing import compute_cost
from .step_cache import StepCache, StepCacheEntry  # Step 74: persistent step cache
from .substitutions import (
    Substitution,
    SubstitutionSet,
    ToolOutputSubstitution,
)
from .trace_reader import TraceVerificationError, verify_trace  # noqa: F401 (reexport)

# ---- stepback version (importlib.metadata avoids circular import via __init__) ----
try:
    from importlib.metadata import version as _importlib_version
    _STEPBACK_VERSION: str = _importlib_version("stepback")
except Exception:  # pragma: no cover
    _STEPBACK_VERSION = "0.1.0"

# ---- provider inference helpers ----
# Maps model-string prefixes (longest-first) to canonical provider names.
_PROVIDER_PREFIX_MAP: List[Tuple[str, str]] = [
    ("anthropic.", "anthropic"),
    ("amazon.", "bedrock"),
    ("meta.", "bedrock"),
    ("mistral.", "bedrock"),
    ("claude-", "anthropic"),
    ("gemini-", "google"),
    ("gpt-", "openai"),
    ("o1-", "openai"),
    ("o3-", "openai"),
    ("o4-", "openai"),
    ("text-", "openai"),
    ("command", "cohere"),
    ("mistral-", "mistral"),
]

# Maps canonical provider names to installable Python SDK package names.
_PROVIDER_SDK_PACKAGES: Dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google-generativeai",
    "bedrock": "boto3",
    "cohere": "cohere",
    "mistral": "mistralai",
}


def _infer_provider(model: Optional[str]) -> Optional[str]:
    """Infer a canonical provider name from a model identifier string."""
    if not model:
        return None
    for prefix, provider in _PROVIDER_PREFIX_MAP:
        if model.startswith(prefix):
            return provider
    return None


def _probe_provider_version(provider: Optional[str]) -> Optional[str]:
    """Return the installed version of ``provider``'s Python SDK, or ``None``."""
    if not provider:
        return None
    pkg = _PROVIDER_SDK_PACKAGES.get(provider)
    if not pkg:
        return None
    try:
        from importlib.metadata import version as _v
        return _v(pkg)
    except Exception:
        return None


def _structural_dirty_reason(
    current_inputs_hash: str,
    recorded_inputs_hash: str,
    nondet_dirty: bool,
    nondet_class_dirty: bool,
    use_parent_dirty: bool,
) -> str:
    """Return the structural reason a step is dirty (excluding forced-output and fallback).

    Returns one of: ``"inputs_changed"``, ``"nondeterminism_hash_changed"``,
    ``"nondeterminism_forced"``, ``"ancestor_dirty"``.
    """
    if current_inputs_hash != recorded_inputs_hash:
        return "inputs_changed"
    if nondet_dirty:
        return "nondeterminism_hash_changed"
    if nondet_class_dirty:
        return "nondeterminism_forced"
    return "ancestor_dirty"


def _utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string ending in ``Z``."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

#: Default worker count used when ``distributed=True`` and ``workers`` is not
#: specified.  Capped at 8 so tests and small machines are not overwhelmed.
_DISTRIBUTED_DEFAULT_WORKERS: int = 8


def _effective_workers(
    distributed: bool,
    workers: Optional[int],
) -> Optional[int]:
    """Resolve the effective worker count for a replay call.

    When *distributed* is ``True`` and *workers* is not given, returns
    ``min(_DISTRIBUTED_DEFAULT_WORKERS, os.cpu_count() or 4)`` so that
    ``Trace.replay_forward(distributed=True)`` automatically uses as many
    threads as the machine has CPUs (up to the cap).

    When *distributed* is ``False``, *workers* is returned unchanged —
    preserving the existing ``workers=N`` API (Step 73).
    """
    if distributed and workers is None:
        return min(_DISTRIBUTED_DEFAULT_WORKERS, os.cpu_count() or 4)
    return workers


# ------------------------------------------------------------ Executor


class MissingExecutor(RuntimeError):
    """Raised when a dirty step has no executor callback registered."""

    #: Canonical error code; see :mod:`stepback.errors` for details.
    code: str = "SB300"


class UnavailableExecutorError(MissingExecutor):
    """Raised by :class:`PartialExecutor` when a dirty step requires an executor
    that was explicitly declared unavailable in the current environment.

    Unlike :class:`MissingExecutor` (which means "no callback registered"),
    this exception means the caller explicitly declared that the executor for
    a particular tool name or model is not accessible — e.g. because an
    imported trace references a production database tool that does not exist
    locally.

    Attributes
    ----------
    step_id :
        ``step_id`` of the step that could not be executed, or ``None`` if
        the error was raised outside a replay loop.
    kind :
        Step kind (``"llm_call"``, ``"tool_call"``, etc.), or ``None``.
    name :
        Tool name or model name that was unavailable, or ``None``.
    executor_type :
        One of ``"llm"``, ``"tool"``, ``"router"``, ``"join"``, or ``None``.
    """

    def __init__(
        self,
        message: str,
        *,
        step_id: Optional[str] = None,
        kind: Optional[str] = None,
        name: Optional[str] = None,
        executor_type: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.step_id = step_id
        self.kind = kind
        self.name = name
        self.executor_type = executor_type


@dataclass
class StepExecutorRequirement:
    """Describes what executor a step needs when it becomes dirty.

    Returned by :func:`audit_executor_requirements` for every step in a
    trace that requires an external executor (``llm_call``, ``tool_call``,
    ``router``).  Built-in step kinds (``parallel_branch_open``,
    ``exception``) are omitted because they never need a user-supplied
    callback.

    Attributes
    ----------
    step_id :
        Identifier of the recorded step.
    kind :
        Step kind (``"llm_call"``, ``"tool_call"``, ``"router"``).
    name :
        Model name for ``llm_call``; tool name for ``tool_call`` and
        ``router``; ``None`` if the recorded inputs did not include a name.
    executor_type :
        Canonical executor type: ``"llm"``, ``"tool"``, or ``"router"``.
    executor_available :
        ``True`` if the supplied executor can handle this step; ``False`` if
        the step would raise :class:`UnavailableExecutorError` (or
        :class:`MissingExecutor` for a plain :class:`Executor` with no
        callback) when dirty.
    """

    step_id: str
    kind: str
    name: Optional[str]
    executor_type: str
    executor_available: bool


class Executor:
    """User-supplied callbacks for re-executing dirty steps.

    Any of the three methods may be unset; if the engine needs to
    re-execute a step kind for which no callback was provided, it
    raises :py:class:`MissingExecutor` — unless ``fallback_recorded``
    is true, in which case the engine reuses the recorded output for
    the step (best-effort replay; useful for the CLI's `diff` /
    `replay` commands which have no LLM/tool wired up but still want
    to show *what would diverge* at substitution points).
    """

    def __init__(
        self,
        llm: Optional[Callable[[str, List[dict]], dict]] = None,
        tool: Optional[Callable[[str, dict], Any]] = None,
        router: Optional[Callable[[str, List[str]], str]] = None,
        join: Optional[Callable[[str, List[Any]], Any]] = None,
        fallback_recorded: bool = False,
        step_cache: Optional[StepCache] = None,
    ) -> None:
        self.llm = llm
        self.tool = tool
        self.router = router
        self.join = join
        self.fallback_recorded = fallback_recorded
        self.step_cache = step_cache
        self.real_calls: int = 0
        self.fallback_uses: int = 0
        # Lock protects only the counter increments so parallel branch workers
        # do not race on real_calls / fallback_uses.  Actual callback execution
        # happens outside the lock so independent branches truly run in parallel.
        self._lock = threading.Lock()

    def _inc_real(self) -> None:
        """Increment ``real_calls`` in a thread-safe way."""
        with self._lock:
            self.real_calls += 1

    def _inc_fallback(self) -> None:
        """Increment ``fallback_uses`` in a thread-safe way."""
        with self._lock:
            self.fallback_uses += 1

    def _cache_get(self, kind: str, inputs_hash: str) -> Optional[StepCacheEntry]:
        """Check persistent step cache; returns entry or ``None`` on miss."""
        if self.step_cache is None:
            return None
        return self.step_cache.get(kind, inputs_hash)

    def _cache_put(self, kind: str, inputs_hash: str, outputs: Any) -> None:
        """Store outputs in persistent step cache."""
        if self.step_cache is None:
            return
        self.step_cache.put(
            StepCacheEntry(
                step_kind=kind,
                inputs_hash=inputs_hash,
                outputs=outputs,
                cached_at=time.time(),
            )
        )

    def execute(self, kind: str, inputs: dict, *, branch_outputs: Optional[List[Any]] = None) -> Any:
        self._inc_real()
        if kind == "llm_call":
            if self.llm is None:
                raise MissingExecutor("no llm executor for dirty llm_call")
            return self.llm(inputs["model"], inputs["messages"])
        if kind == "tool_call":
            if self.tool is None:
                raise MissingExecutor("no tool executor for dirty tool_call")
            return {"result": self.tool(inputs["name"], inputs["arguments"])}
        if kind == "router":
            if self.router is None:
                raise MissingExecutor("no router executor for dirty router")
            return {"choice": self.router(inputs["name"], inputs["options"])}
        if kind == "parallel_branch_open":
            return {
                "branch_names": list(inputs.get("branch_names", [])),
                "branch_count": int(inputs.get("branch_count", 0)),
            }
        if kind == "parallel_branch_join":
            outs = list(branch_outputs or [])
            if self.join is not None:
                return self.join(inputs.get("name", ""), outs)
            return {"branches": outs}
        if kind == "exception":
            return {"error_class": "Replayed", "message": ""}
        raise MissingExecutor(f"unknown step kind: {kind}")


class PartialExecutor(Executor):
    """Executor that declares which tool names and model names it can handle.

    For steps whose tool/model is **not** in the declared available sets,
    :meth:`execute` raises :class:`UnavailableExecutorError` instead of
    :class:`MissingExecutor`.  This lets the replay engine (and the
    :func:`minimize_imported_trace` oracle) distinguish between
    "executor not configured at all" and "executor explicitly unavailable
    for this step".

    When used together with ``fallback_recorded=True``, the replay engine
    catches :class:`UnavailableExecutorError` and falls back to the
    recorded output, enabling **partial replay** on imported traces where
    only some tools or models are available locally.

    Parameters
    ----------
    llm, tool, router, join, fallback_recorded, step_cache :
        Same as :class:`Executor`.
    available_tools :
        Set of tool names this executor can handle.  ``None`` (the default)
        means *all* tool names are available (no name-based restriction).
        Pass an empty set ``set()`` to mark every tool as unavailable.
    available_models :
        Set of model names this executor can handle.  ``None`` means *all*
        models are available.  Pass an empty set to mark every model as
        unavailable.

    Examples
    --------
    ::

        # Only 'search' tool is available; all LLM models are available.
        exec_ = PartialExecutor(
            llm=my_llm, tool=my_tool, available_tools={"search"}
        )
        # 'lookup_database' will raise UnavailableExecutorError.

        # Nothing is available; fall back to recorded outputs for all dirty steps.
        exec_ = PartialExecutor(fallback_recorded=True, available_tools=set(), available_models=set())
    """

    def __init__(
        self,
        llm: Optional[Callable] = None,
        tool: Optional[Callable] = None,
        router: Optional[Callable] = None,
        join: Optional[Callable] = None,
        fallback_recorded: bool = False,
        step_cache: Optional["StepCache"] = None,
        available_tools: Optional["Set[str]"] = None,
        available_models: Optional["Set[str]"] = None,
    ) -> None:
        super().__init__(
            llm=llm,
            tool=tool,
            router=router,
            join=join,
            fallback_recorded=fallback_recorded,
            step_cache=step_cache,
        )
        # None means "all available"; a set means only those names are available.
        self._available_tools: Optional["Set[str]"] = available_tools
        self._available_models: Optional["Set[str]"] = available_models

    def is_available(self, kind: str, inputs: dict) -> bool:
        """Return ``True`` if this executor can handle the step.

        Parameters
        ----------
        kind :
            Step kind (``"llm_call"``, ``"tool_call"``, etc.).
        inputs :
            Recorded or current inputs dict for the step.
        """
        if kind == "llm_call":
            if self.llm is None:
                return False
            if self._available_models is not None:
                model = inputs.get("model", "")
                return model in self._available_models
            return True
        if kind == "tool_call":
            if self.tool is None:
                return False
            if self._available_tools is not None:
                name = inputs.get("name", "")
                return name in self._available_tools
            return True
        if kind == "router":
            return self.router is not None
        # parallel_branch_open, exception, parallel_branch_join use built-in logic.
        return True

    def execute(
        self,
        kind: str,
        inputs: dict,
        *,
        branch_outputs: Optional[List[Any]] = None,
    ) -> Any:
        """Execute a step, raising :class:`UnavailableExecutorError` for unavailable steps."""
        if kind == "llm_call" and self._available_models is not None:
            model = inputs.get("model", "")
            if model not in self._available_models:
                raise UnavailableExecutorError(
                    f"model '{model}' is not in available_models",
                    kind=kind,
                    name=model,
                    executor_type="llm",
                )
        if kind == "tool_call" and self._available_tools is not None:
            name = inputs.get("name", "")
            if name not in self._available_tools:
                raise UnavailableExecutorError(
                    f"tool '{name}' is not in available_tools",
                    kind=kind,
                    name=name,
                    executor_type="tool",
                )
        return super().execute(kind, inputs, branch_outputs=branch_outputs)


def _executor_type_for_kind(kind: str) -> Optional[str]:
    """Return the executor-type name for a step kind, or ``None`` for built-in steps.

    Built-in step kinds (``parallel_branch_open``, ``parallel_branch_join``,
    ``exception``) do not require a user-supplied executor callback and are
    excluded from :func:`audit_executor_requirements`.
    """
    if kind == "llm_call":
        return "llm"
    if kind == "tool_call":
        return "tool"
    if kind == "router":
        return "router"
    return None


def audit_executor_requirements(
    trace: "Trace",
    executor: Optional[Executor] = None,
) -> List[StepExecutorRequirement]:
    """Scan a trace and report what executor each step requires when dirty.

    Returns a :class:`StepExecutorRequirement` for every step whose kind
    needs an external executor callback (``llm_call``, ``tool_call``,
    ``router``).  Steps served by the engine's built-in logic
    (``parallel_branch_open``, ``parallel_branch_join``, ``exception``)
    are omitted.

    Parameters
    ----------
    trace :
        The :class:`Trace` to audit.
    executor :
        Optional executor to check availability against.

        * :class:`PartialExecutor` — uses :meth:`PartialExecutor.is_available`
          to populate :attr:`StepExecutorRequirement.executor_available`.
        * Plain :class:`Executor` — a step is available iff its matching
          callback (``llm``, ``tool``, ``router``) is non-``None``.
        * ``None`` — all steps are marked unavailable
          (``executor_available=False``).

    Returns
    -------
    List[StepExecutorRequirement]
        One entry per executor-requiring step, in recorded order.

    Example
    -------
    ::

        trace = replay("run.sb")
        exec_ = PartialExecutor(tool=my_tool, available_tools={"search"})
        reqs = audit_executor_requirements(trace, exec_)
        unavailable = [r for r in reqs if not r.executor_available]
    """
    requirements: List[StepExecutorRequirement] = []
    for rec in trace.recorded_steps:
        kind = rec.get("step_kind", "")
        executor_type = _executor_type_for_kind(kind)
        if executor_type is None:
            continue

        inputs = rec.get("inputs", {})
        if executor_type == "llm":
            name: Optional[str] = inputs.get("model")
        elif executor_type in ("tool", "router"):
            name = inputs.get("name")
        else:
            name = None

        # Determine availability.
        if executor is None:
            available = False
        elif isinstance(executor, PartialExecutor):
            available = executor.is_available(kind, inputs)
        else:
            # Plain Executor: available iff the relevant callback is set.
            if executor_type == "llm":
                available = executor.llm is not None
            elif executor_type == "tool":
                available = executor.tool is not None
            elif executor_type == "router":
                available = executor.router is not None
            else:
                available = True

        requirements.append(
            StepExecutorRequirement(
                step_id=rec["step_id"],
                kind=kind,
                name=name,
                executor_type=executor_type,
                executor_available=available,
            )
        )
    return requirements


# --------------------------------------------------------------- provenance


@dataclass
class StepProvenance:
    """Per-step provenance captured automatically during every replay pass.

    Always present on :attr:`StepView.provenance`.  Callers may inspect
    these fields to audit why a step was re-executed and which executor /
    provider version produced the result.
    """

    #: Structural reason the step was dirty, or ``None`` for a clean cache hit.
    #: Possible values: ``"output_forced"``, ``"inputs_changed"``,
    #: ``"nondeterminism_hash_changed"``, ``"nondeterminism_forced"``,
    #: ``"ancestor_dirty"``.  ``None`` means the step was served from the
    #: recorded trace or the persistent step cache without re-execution.
    dirty_reason: Optional[str]

    #: Where the step's output was sourced during this replay.
    #: ``"recorded_trace"`` — reused verbatim from the recorded SB-Trace
    #: (clean cache hit, ``dirty=False``).
    #: ``"persistent_cache"`` — found in the on-disk :class:`~stepback.step_cache.StepCache`
    #: (still dirty in trace terms but no executor call was made).
    #: ``"executor"`` — re-executed via the user-supplied :class:`Executor`.
    #: ``"fallback"`` — ``executor.fallback_recorded`` path: recorded output
    #: reused because no executor was provided for this step kind.
    #: ``"output_forced"`` — an output-forcing substitution supplied the output
    #: directly (no executor call).
    cache_source: str

    #: Seed submitted to the LLM for this step (from ``inputs["seed"]``),
    #: or ``None`` if the step is not an LLM call or no seed was recorded.
    seed: Optional[int]

    #: Model identifier from ``inputs["model"]`` (LLM steps only), or ``None``.
    model: Optional[str]

    #: Canonical provider name inferred from :attr:`model`, or ``None`` if
    #: the model string is absent or unrecognised.
    provider: Optional[str]

    #: Model version string returned in the step's outputs by providers that
    #: expose it (e.g. Gemini's ``model_version`` field), or ``None``.
    model_version: Optional[str]

    #: Installed version of the provider's Python SDK, or ``None`` if the
    #: provider is not recognised or its SDK is not installed.
    provider_version: Optional[str]

    #: ``True`` if the step's outputs look like a policy-denial response,
    #: detected via :func:`~stepback.policy_audit.is_policy_blocked`.
    policy_blocked: bool

    #: Human-readable policy denial reason extracted from the step outputs
    #: (the ``"reason"`` field of a policy-denial dict), or ``None``.
    policy_reason: Optional[str]

    #: Version of the stepback library that executed this step.
    executor_version: str


@dataclass
class ReplayProvenance:
    """Provenance for an entire replay run, attached to :attr:`ReplayResult.provenance`.

    Captures timing and executor metadata that spans all steps.
    """

    #: Version of the stepback library that ran this replay.
    executor_version: str

    #: ISO-8601 UTC timestamp when the replay loop started (``…Z`` suffix).
    started_at: str

    #: ISO-8601 UTC timestamp when the replay loop finished (``…Z`` suffix).
    finished_at: str


def _build_step_provenance(
    cur_inputs: dict,
    cur_outputs: Any,
    is_dirty: bool,
    dirty_reason_str: Optional[str],
    cache_source_str: str,
) -> StepProvenance:
    """Construct a :class:`StepProvenance` from per-step execution state."""
    model: Optional[str] = cur_inputs.get("model") if isinstance(cur_inputs, dict) else None
    provider: Optional[str] = _infer_provider(model)
    model_version: Optional[str] = None
    if isinstance(cur_outputs, dict):
        model_version = cur_outputs.get("model_version")

    seed_raw = cur_inputs.get("seed") if isinstance(cur_inputs, dict) else None
    seed: Optional[int] = int(seed_raw) if seed_raw is not None else None

    # Lazy import to avoid circular dependency; policy_audit imports nothing from replay.
    from .policy_audit import is_policy_blocked as _is_policy_blocked  # noqa: PLC0415
    policy_blocked = _is_policy_blocked(cur_outputs)
    policy_reason: Optional[str] = None
    if policy_blocked and isinstance(cur_outputs, dict):
        policy_reason = cur_outputs.get("reason") or cur_outputs.get("policy_reason")

    return StepProvenance(
        dirty_reason=dirty_reason_str,
        cache_source=cache_source_str,
        seed=seed,
        model=model,
        provider=provider,
        model_version=model_version,
        provider_version=_probe_provider_version(provider),
        policy_blocked=policy_blocked,
        policy_reason=policy_reason,
        executor_version=_STEPBACK_VERSION,
    )


# --------------------------------------------------------------- core


@dataclass
class StepView:
    """A single step in a replay result.

    Attribute access is the public API used in `bisect` predicates and
    in user code that walks ``result.steps``.
    """

    step_id: str
    kind: str
    name: Optional[str]
    parent_step_id: Optional[str]
    inputs: dict
    outputs: Any
    cost_usd: float
    dirty: bool
    cache_hit: bool
    recorded_inputs_hash: str
    current_inputs_hash: str
    # True iff this step was dirty AND its recomputed output hash differs from
    # the recorded output hash.  Used by Step 61 stale-cache propagation so
    # that downstream steps are not invalidated when a parent is re-executed
    # but produces an identical output.
    output_changed: bool = False
    #: Provenance for this step; always populated by all replay code paths.
    provenance: Optional[StepProvenance] = None

    # Dot-notation aliases used in the README's example predicates.
    @property
    def cost(self) -> float:
        return self.cost_usd

    @property
    def error_class(self) -> Optional[str]:
        if isinstance(self.outputs, dict):
            return self.outputs.get("error_class")
        return None


@dataclass
class ReplayResult:
    steps: List[StepView]
    total_cost_usd: float
    dirty_count: int
    cache_hit_count: int
    real_executions: int
    #: Overall replay provenance; always populated by all replay code paths.
    provenance: Optional[ReplayProvenance] = None

    def any_step(self, predicate: Callable[[StepView], bool]) -> bool:
        return any(predicate(s) for s in self.steps)

    def find(self, predicate: Callable[[StepView], bool]) -> Optional[StepView]:
        for s in self.steps:
            if predicate(s):
                return s
        return None

    def __iter__(self):
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, i):
        return self.steps[i]


# ------------------------------------------------------------- Branch


@dataclass
class Branch:
    name: str
    base_step: str
    substitutions: SubstitutionSet = field(default_factory=SubstitutionSet)
    result: Optional[ReplayResult] = None
    _owner: Optional["Trace"] = None

    def substitute(self, *subs: Substitution) -> "Branch":
        for s in subs:
            self.substitutions.add(s)
        return self

    def replay_forward(
        self,
        executor: Optional[Executor] = None,
        *,
        workers: Optional[int] = None,
        distributed: bool = False,
    ) -> ReplayResult:
        assert self._owner is not None, "Branch must be created via Trace.branch_at"
        self.result = self._owner.run_replay(
            self.substitutions, executor or Executor(), workers=workers,
            distributed=distributed,
        )
        return self.result


@dataclass
class StepDiff:
    step_id: str
    kind: str
    output_diff: dict
    cost_delta_usd: float
    diverged_from_cache: bool


@dataclass
class BranchDiff:
    a: str
    b: str
    step_diffs: List[StepDiff]
    total_cost_delta_usd: float
    divergent_step_count: int


# -------------------------------------------------------------- Planner


@dataclass
class PlannedStep:
    """A single step as decided by the planner phase.

    ``planned_action`` is an **estimate** derived by walking the trace
    using recorded outputs as placeholders for dirty parents.  Downstream
    cascade effects (a dirty parent producing a different output) cannot be
    known until the executor phase runs the step.

    Use :attr:`requires_runtime_validation` to detect steps whose planned
    action may be revised during execution.
    """

    step_id: str
    kind: str
    name: Optional[str]
    parent_step_id: Optional[str]
    recorded_inputs_hash: str
    #: Inputs after substitutions and parent rebinding (using recorded
    #: outputs as placeholders for dirty-parent outputs).
    planned_inputs: dict
    planned_inputs_hash: str
    #: ``"cache_hit"`` or ``"execute"``.  This is an optimistic estimate
    #: when :attr:`requires_runtime_validation` is True.
    planned_action: str
    #: Human-readable explanation when ``planned_action == "execute"``.
    dirty_reason: Optional[str]
    #: True for any step whose planned action assumed a dirty parent
    #: produces the same output as recorded.  The executor re-evaluates
    #: these steps against the actual parent output at runtime.
    requires_runtime_validation: bool


@dataclass
class ReplayPlan:
    """An ordered list of planned step decisions produced by
    :meth:`Trace.plan_replay`.

    The plan is a scheduling artifact, not an authoritative result.
    Call :meth:`execute` to obtain a :class:`ReplayResult` with final,
    correct dirty/cache-hit decisions.
    """

    planned_steps: List[PlannedStep]
    #: Lower-bound estimate of dirty steps (direct substitution / nondet /
    #: inputs-hash mismatches, not including cascade from dirty parents).
    estimated_dirty_count: int
    #: Upper-bound estimate of cache hits (may drop during execution when
    #: cascade dirtiness propagates from dirty-parent actual outputs).
    estimated_cache_hit_count: int
    #: Reference to the recorded steps so the executor can access full
    #: step metadata (outputs, nondeterminism, etc.) without holding a
    #: separate Trace reference.
    _recorded_steps: List[dict] = field(default_factory=list, repr=False)
    #: The substitution set that was used to build this plan.
    _subs: SubstitutionSet = field(default_factory=SubstitutionSet, repr=False)

    def execute(
        self,
        executor: Optional[Executor] = None,
        *,
        workers: Optional[int] = None,
        distributed: bool = False,
        _event_bus: Optional["Any"] = None,
        _job_id: Optional[str] = None,
    ) -> ReplayResult:
        """Execute the plan and return the authoritative :class:`ReplayResult`.

        The executor phase **fully re-evaluates** every step using actual
        parent outputs, so cascading dirtiness is handled correctly even
        when the planner optimistically marked descendants as cache hits.

        ``executor`` defaults to a no-op :class:`Executor`.  Dirty steps
        with no matching executor callback raise :class:`MissingExecutor`
        unless ``executor.fallback_recorded`` is True.

        Parameters
        ----------
        workers :
            See :meth:`Trace.replay_forward`.  ``None`` or ``1`` →
            sequential.  ``N > 1`` → parallel branch execution.
        distributed :
            See :meth:`Trace.replay_forward`.  When ``True`` and *workers*
            is not specified, an appropriate worker count is chosen
            automatically.  Delegates to the same local
            :class:`~concurrent.futures.ThreadPoolExecutor` planner.
            Note: when *_event_bus* is also set, the sequential path is
            used regardless of *distributed* / *workers*, because the
            parallel planner does not yet support per-step event
            publishing.
        _event_bus :
            Optional :class:`~stepback.event_bus.EventBus` for publishing
            ``REPLAY_STEP_COMPLETE`` events after each step.  Intended for
            use via :func:`~stepback.event_bus.execute_with_idempotency`;
            the leading underscore signals this is an implementation detail.
        _job_id :
            Job identifier included in every emitted event.  If *_event_bus*
            is set and *_job_id* is omitted, a placeholder ``"<unnamed>"`` is
            used.
        """
        # Delegate to the full evaluation loop, which is the same logic as
        # Trace.run_replay but operates on the pre-validated plan context.
        eff_executor = executor or Executor()
        eff_workers = _effective_workers(distributed, workers)
        # The parallel planner does not publish per-step events; fall back to
        # sequential when an event bus is attached.
        if eff_workers is not None and eff_workers > 1 and _event_bus is None:
            return _execute_plan_parallel(self._recorded_steps, self._subs, eff_executor, eff_workers)
        return _execute_plan(self._recorded_steps, self._subs, eff_executor,
                             event_bus=_event_bus, job_id=_job_id)


def _execute_plan(
    recorded_steps: List[dict],
    subs: SubstitutionSet,
    executor: Executor,
    *,
    event_bus: Optional[Any] = None,
    job_id: Optional[str] = None,
    on_step: Optional[Callable[[int, int], None]] = None,
) -> ReplayResult:
    """Full re-evaluation loop used by both ``ReplayPlan.execute`` and
    ``Trace.run_replay``.  Kept as a module-level helper so both callers
    share identical semantics.

    Parameters
    ----------
    event_bus :
        Optional :class:`~stepback.event_bus.EventBus`.  When provided,
        a ``REPLAY_STEP_COMPLETE`` event is published after each step.
    job_id :
        Job identifier forwarded to every emitted event.
    on_step :
        Optional progress callback invoked after every step with
        ``(steps_done, total_steps)``.  Useful for CLI progress bars.
        Exceptions raised by the callback are silently ignored.
    """
    started_at = _utc_now_iso()
    outputs_by_id: Dict[str, Any] = {}
    outputs_hash_by_id: Dict[str, str] = {}
    dirty_by_id: Dict[str, bool] = {}
    output_changed_by_id: Dict[str, bool] = {}
    steps_view: List[StepView] = []
    total_cost = 0.0
    dirty_n = 0
    hit_n = 0
    real_n = 0
    sentinel = object()

    for rec in recorded_steps:
        sid = rec["step_id"]
        kind = rec["step_kind"]
        recorded_inputs_hash = rec["inputs_hash"]
        cur_inputs = copy.deepcopy(rec["inputs"])

        # Rebind "context" from the parent's current output.
        parent_id = rec.get("parent_step_id")
        context_fields: Optional[List[str]] = cur_inputs.get("_stepback_context_fields")
        if parent_id and "context" in cur_inputs and parent_id in outputs_by_id:
            if context_fields:
                parent_out = outputs_by_id[parent_id]
                if isinstance(parent_out, dict):
                    partial = {k: parent_out[k] for k in context_fields if k in parent_out}
                    cur_inputs["context"] = hash_obj(partial)
                else:
                    cur_inputs["context"] = outputs_hash_by_id.get(parent_id, cur_inputs["context"])
            else:
                cur_inputs["context"] = outputs_hash_by_id[parent_id]
        elif parent_id and "context" in cur_inputs and parent_id in outputs_hash_by_id and not context_fields:
            cur_inputs["context"] = outputs_hash_by_id[parent_id]

        # Multi-parent rebinding for parallel_branch_join.
        parent_ids: List[str] = list(rec.get("parent_step_ids") or [])
        if "branch_tails" in cur_inputs and "branch_tail_hashes" in cur_inputs:
            tails = cur_inputs["branch_tails"]
            cur_inputs["branch_tail_hashes"] = [
                outputs_hash_by_id.get(t, cur_inputs["branch_tail_hashes"][i])
                for i, t in enumerate(tails)
            ]

        # Apply substitutions targeting this step.
        tool_override: Any = sentinel
        for sub in subs.at(sid):
            if sub.is_output_forcing():
                tool_override = sub.force_output(rec)
            else:
                sub.apply(cur_inputs, rec)

        current_inputs_hash = hash_obj(cur_inputs)
        parent_dirty = dirty_by_id.get(parent_id, False) if parent_id else False
        if parent_ids:
            parent_dirty = parent_dirty or any(
                dirty_by_id.get(pid, False) for pid in parent_ids
            )

        parent_output_changed = (
            output_changed_by_id.get(parent_id, False) if parent_id else False
        )
        if parent_ids:
            parent_output_changed = parent_output_changed or any(
                output_changed_by_id.get(pid, False) for pid in parent_ids
            )

        use_parent_dirty = parent_output_changed and not context_fields

        # Nondeterminism-hash check.
        recorded_nondet_hash = rec.get("nondeterminism_hash")
        if recorded_nondet_hash is not None:
            live_nondet_hash = sha256_hex(
                canonical_json(rec.get("nondeterminism", {}))
            )
            nondet_dirty = recorded_nondet_hash != live_nondet_hash
        else:
            nondet_dirty = False

        nondet_class_dirty = _nondet_forces_dirty(rec.get("nondeterminism", {}))

        _is_fallback = False
        _is_persistent_cache = False

        if tool_override is not sentinel:
            cur_outputs = tool_override
            is_dirty = True
            cache_hit = False
        elif (
            current_inputs_hash == recorded_inputs_hash
            and not use_parent_dirty
            and not nondet_dirty
            and not nondet_class_dirty
        ):
            cur_outputs = rec["outputs"]
            is_dirty = False
            cache_hit = True
            hit_n += 1
        else:
            if executor.fallback_recorded and (
                (kind == "llm_call" and executor.llm is None)
                or (kind == "tool_call" and executor.tool is None)
                or (kind == "router" and executor.router is None)
                or (kind == "parallel_branch_join" and executor.join is None
                    and not parent_ids)
            ):
                cur_outputs = rec["outputs"]
                executor._inc_fallback()
                _is_fallback = True
            else:
                # Check persistent step cache before calling the executor.
                # Only eligible when not forced-dirty by nondeterminism class
                # (clock, unseeded RNG, etc.) — those must always re-execute.
                _cached_entry = (
                    None
                    if (nondet_dirty or nondet_class_dirty)
                    else executor._cache_get(kind, current_inputs_hash)
                )
                if _cached_entry is not None:
                    cur_outputs = _cached_entry.outputs
                    # Cache hit via persistent cache: dirty in trace terms but
                    # no executor call, so real_n is NOT incremented.
                    _is_persistent_cache = True
                elif kind == "parallel_branch_join" and parent_ids:
                    b_outs = [outputs_by_id[pid] for pid in parent_ids]
                    try:
                        cur_outputs = executor.execute(
                            kind, cur_inputs, branch_outputs=b_outs
                        )
                        real_n += 1
                        executor._cache_put(kind, current_inputs_hash, cur_outputs)
                    except UnavailableExecutorError:
                        if executor.fallback_recorded:
                            cur_outputs = rec["outputs"]
                            executor._inc_fallback()
                            _is_fallback = True
                        else:
                            raise
                else:
                    try:
                        cur_outputs = executor.execute(kind, cur_inputs)
                        real_n += 1
                        executor._cache_put(kind, current_inputs_hash, cur_outputs)
                    except UnavailableExecutorError:
                        if executor.fallback_recorded:
                            cur_outputs = rec["outputs"]
                            executor._inc_fallback()
                            _is_fallback = True
                        else:
                            raise
            is_dirty = True
            cache_hit = False

        if cache_hit:
            cost = float(rec.get("cost_usd", 0.0))
        else:
            if kind == "llm_call" and isinstance(cur_outputs, dict):
                cost = compute_cost(
                    cur_inputs.get("model", ""), cur_outputs.get("usage", {})
                )
            else:
                cost = float(rec.get("cost_usd", 0.0))

        outputs_by_id[sid] = cur_outputs
        cur_hash = hash_obj(cur_outputs)
        outputs_hash_by_id[sid] = cur_hash
        dirty_by_id[sid] = is_dirty
        output_changed_by_id[sid] = is_dirty and cur_hash != hash_obj(rec["outputs"])
        total_cost += cost
        if is_dirty:
            dirty_n += 1

        # --- provenance ---
        if not is_dirty:
            _dirty_reason: Optional[str] = None
            _cache_source = "recorded_trace"
        elif tool_override is not sentinel:
            _dirty_reason = "output_forced"
            _cache_source = "output_forced"
        elif _is_fallback:
            _dirty_reason = _structural_dirty_reason(
                current_inputs_hash, recorded_inputs_hash,
                nondet_dirty, nondet_class_dirty, use_parent_dirty,
            )
            _cache_source = "fallback"
        elif _is_persistent_cache:
            _dirty_reason = _structural_dirty_reason(
                current_inputs_hash, recorded_inputs_hash,
                nondet_dirty, nondet_class_dirty, use_parent_dirty,
            )
            _cache_source = "persistent_cache"
        else:
            _dirty_reason = _structural_dirty_reason(
                current_inputs_hash, recorded_inputs_hash,
                nondet_dirty, nondet_class_dirty, use_parent_dirty,
            )
            _cache_source = "executor"

        step_prov = _build_step_provenance(cur_inputs, cur_outputs, is_dirty, _dirty_reason, _cache_source)

        steps_view.append(
            StepView(
                step_id=sid,
                kind=kind,
                name=rec.get("name"),
                parent_step_id=parent_id,
                inputs=cur_inputs,
                outputs=cur_outputs,
                cost_usd=round(cost, 8),
                dirty=is_dirty,
                cache_hit=cache_hit,
                recorded_inputs_hash=recorded_inputs_hash,
                current_inputs_hash=current_inputs_hash,
                output_changed=output_changed_by_id[sid],
                provenance=step_prov,
            )
        )

        # Emit step event if an event bus is configured.
        if event_bus is not None:
            from .event_bus import EventKind, ReplayEvent  # lazy import avoids circular dep
            event_bus.publish(ReplayEvent(
                kind=EventKind.REPLAY_STEP_COMPLETE,
                job_id=job_id or "<unnamed>",
                step_id=sid,
                step_index=len(steps_view) - 1,
                dirty=is_dirty,
                payload={"kind": kind, "cache_hit": cache_hit},
            ))

        # Progress callback — failures must not abort the replay.
        if on_step is not None:
            try:
                on_step(len(steps_view), len(recorded_steps))
            except Exception:  # pragma: no cover
                pass

    finished_at = _utc_now_iso()
    return ReplayResult(
        steps=steps_view,
        total_cost_usd=round(total_cost, 8),
        dirty_count=dirty_n,
        cache_hit_count=hit_n,
        real_executions=real_n,
        provenance=ReplayProvenance(
            executor_version=_STEPBACK_VERSION,
            started_at=started_at,
            finished_at=finished_at,
        ),
    )


# ------------------------------------------------------- parallel execution


def _execute_branch_steps(
    branch_steps: List[dict],
    subs: SubstitutionSet,
    executor: Executor,
    snap_outputs_by_id: Dict[str, Any],
    snap_outputs_hash_by_id: Dict[str, str],
    snap_dirty_by_id: Dict[str, bool],
    snap_output_changed_by_id: Dict[str, bool],
) -> Tuple[List[StepView], float, int, int, Dict[str, Any], Dict[str, str], Dict[str, bool], Dict[str, bool]]:
    """Execute one branch region's steps sequentially.

    Reads external-parent state from the immutable *snap_* dicts (produced by
    snapshotting global state before the parallel tier) and builds its own
    local state dicts so concurrent branch workers never race on shared state.

    The only shared object that branch workers write concurrently is
    ``executor.real_calls`` / ``executor.fallback_uses``; those writes go
    through :py:meth:`Executor._inc_real` / :py:meth:`Executor._inc_fallback`
    which hold a ``threading.Lock`` for the duration of the increment only —
    user callbacks are invoked outside the lock so branches truly run in
    parallel.

    Returns
    -------
    (steps_view, total_cost, dirty_n, hit_n,
     local_outputs_by_id, local_outputs_hash_by_id,
     local_dirty_by_id, local_output_changed_by_id)
    """
    # Local state for this branch — never shared with sibling branches.
    local_outputs_by_id: Dict[str, Any] = {}
    local_outputs_hash_by_id: Dict[str, str] = {}
    local_dirty_by_id: Dict[str, bool] = {}
    local_output_changed_by_id: Dict[str, bool] = {}

    steps_view: List[StepView] = []
    total_cost = 0.0
    dirty_n = 0
    hit_n = 0
    sentinel = object()

    def _get_output(sid: str) -> Any:
        if sid in local_outputs_by_id:
            return local_outputs_by_id[sid]
        return snap_outputs_by_id.get(sid)

    def _get_hash(sid: str) -> Optional[str]:
        if sid in local_outputs_hash_by_id:
            return local_outputs_hash_by_id[sid]
        return snap_outputs_hash_by_id.get(sid)

    def _is_dirty(sid: str) -> bool:
        if sid in local_dirty_by_id:
            return local_dirty_by_id[sid]
        return snap_dirty_by_id.get(sid, False)

    def _is_output_changed(sid: str) -> bool:
        if sid in local_output_changed_by_id:
            return local_output_changed_by_id[sid]
        return snap_output_changed_by_id.get(sid, False)

    for rec in branch_steps:
        sid = rec["step_id"]
        kind = rec["step_kind"]
        recorded_inputs_hash = rec["inputs_hash"]
        cur_inputs = copy.deepcopy(rec["inputs"])

        parent_id = rec.get("parent_step_id")
        context_fields: Optional[List[str]] = cur_inputs.get("_stepback_context_fields")
        parent_hash = _get_hash(parent_id) if parent_id else None

        if parent_id and "context" in cur_inputs and parent_hash is not None:
            if context_fields:
                parent_out = _get_output(parent_id)
                if isinstance(parent_out, dict):
                    partial = {k: parent_out[k] for k in context_fields if k in parent_out}
                    cur_inputs["context"] = hash_obj(partial)
                else:
                    cur_inputs["context"] = parent_hash
            else:
                cur_inputs["context"] = parent_hash
        elif parent_id and "context" in cur_inputs and parent_hash is not None and not context_fields:
            cur_inputs["context"] = parent_hash

        parent_ids: List[str] = list(rec.get("parent_step_ids") or [])
        if "branch_tails" in cur_inputs and "branch_tail_hashes" in cur_inputs:
            tails = cur_inputs["branch_tails"]
            cur_inputs["branch_tail_hashes"] = [
                (_get_hash(t) or cur_inputs["branch_tail_hashes"][i])
                for i, t in enumerate(tails)
            ]

        tool_override: Any = sentinel
        for sub in subs.at(sid):
            if sub.is_output_forcing():
                tool_override = sub.force_output(rec)
            else:
                sub.apply(cur_inputs, rec)

        current_inputs_hash = hash_obj(cur_inputs)

        parent_dirty = _is_dirty(parent_id) if parent_id else False
        if parent_ids:
            parent_dirty = parent_dirty or any(_is_dirty(pid) for pid in parent_ids)

        parent_output_changed = _is_output_changed(parent_id) if parent_id else False
        if parent_ids:
            parent_output_changed = parent_output_changed or any(
                _is_output_changed(pid) for pid in parent_ids
            )

        use_parent_dirty = parent_output_changed and not context_fields

        recorded_nondet_hash = rec.get("nondeterminism_hash")
        if recorded_nondet_hash is not None:
            live_nondet_hash = sha256_hex(canonical_json(rec.get("nondeterminism", {})))
            nondet_dirty = recorded_nondet_hash != live_nondet_hash
        else:
            nondet_dirty = False

        nondet_class_dirty = _nondet_forces_dirty(rec.get("nondeterminism", {}))

        _is_fallback_b = False
        _is_persistent_cache_b = False

        if tool_override is not sentinel:
            cur_outputs = tool_override
            is_dirty = True
            cache_hit = False
        elif (
            current_inputs_hash == recorded_inputs_hash
            and not use_parent_dirty
            and not nondet_dirty
            and not nondet_class_dirty
        ):
            cur_outputs = rec["outputs"]
            is_dirty = False
            cache_hit = True
            hit_n += 1
        else:
            if executor.fallback_recorded and (
                (kind == "llm_call" and executor.llm is None)
                or (kind == "tool_call" and executor.tool is None)
                or (kind == "router" and executor.router is None)
                or (kind == "parallel_branch_join" and executor.join is None
                    and not parent_ids)
            ):
                cur_outputs = rec["outputs"]
                executor._inc_fallback()
                _is_fallback_b = True
            else:
                # Check persistent step cache before calling the executor.
                # Bypass for nondeterminism-class-forced steps (Step 63).
                _cached_entry = (
                    None
                    if (nondet_dirty or nondet_class_dirty)
                    else executor._cache_get(kind, current_inputs_hash)
                )
                if _cached_entry is not None:
                    cur_outputs = _cached_entry.outputs
                    _is_persistent_cache_b = True
                elif kind == "parallel_branch_join" and parent_ids:
                    b_outs = [_get_output(pid) for pid in parent_ids]
                    try:
                        cur_outputs = executor.execute(kind, cur_inputs, branch_outputs=b_outs)
                        executor._cache_put(kind, current_inputs_hash, cur_outputs)
                    except UnavailableExecutorError:
                        if executor.fallback_recorded:
                            cur_outputs = rec["outputs"]
                            executor._inc_fallback()
                            _is_fallback_b = True
                        else:
                            raise
                else:
                    try:
                        cur_outputs = executor.execute(kind, cur_inputs)
                        executor._cache_put(kind, current_inputs_hash, cur_outputs)
                    except UnavailableExecutorError:
                        if executor.fallback_recorded:
                            cur_outputs = rec["outputs"]
                            executor._inc_fallback()
                            _is_fallback_b = True
                        else:
                            raise
            is_dirty = True
            cache_hit = False

        if cache_hit:
            cost = float(rec.get("cost_usd", 0.0))
        else:
            if kind == "llm_call" and isinstance(cur_outputs, dict):
                cost = compute_cost(cur_inputs.get("model", ""), cur_outputs.get("usage", {}))
            else:
                cost = float(rec.get("cost_usd", 0.0))

        local_outputs_by_id[sid] = cur_outputs
        cur_hash = hash_obj(cur_outputs)
        local_outputs_hash_by_id[sid] = cur_hash
        local_dirty_by_id[sid] = is_dirty
        local_output_changed_by_id[sid] = is_dirty and cur_hash != hash_obj(rec["outputs"])
        total_cost += cost
        if is_dirty:
            dirty_n += 1

        # --- provenance ---
        if not is_dirty:
            _dirty_reason_b: Optional[str] = None
            _cache_source_b = "recorded_trace"
        elif tool_override is not sentinel:
            _dirty_reason_b = "output_forced"
            _cache_source_b = "output_forced"
        elif _is_fallback_b:
            _dirty_reason_b = _structural_dirty_reason(
                current_inputs_hash, recorded_inputs_hash,
                nondet_dirty, nondet_class_dirty, use_parent_dirty,
            )
            _cache_source_b = "fallback"
        elif _is_persistent_cache_b:
            _dirty_reason_b = _structural_dirty_reason(
                current_inputs_hash, recorded_inputs_hash,
                nondet_dirty, nondet_class_dirty, use_parent_dirty,
            )
            _cache_source_b = "persistent_cache"
        else:
            _dirty_reason_b = _structural_dirty_reason(
                current_inputs_hash, recorded_inputs_hash,
                nondet_dirty, nondet_class_dirty, use_parent_dirty,
            )
            _cache_source_b = "executor"

        step_prov = _build_step_provenance(cur_inputs, cur_outputs, is_dirty, _dirty_reason_b, _cache_source_b)

        steps_view.append(
            StepView(
                step_id=sid,
                kind=kind,
                name=rec.get("name"),
                parent_step_id=parent_id,
                inputs=cur_inputs,
                outputs=cur_outputs,
                cost_usd=round(cost, 8),
                dirty=is_dirty,
                cache_hit=cache_hit,
                recorded_inputs_hash=recorded_inputs_hash,
                current_inputs_hash=current_inputs_hash,
                output_changed=local_output_changed_by_id[sid],
                provenance=step_prov,
            )
        )

    return (
        steps_view,
        total_cost,
        dirty_n,
        hit_n,
        local_outputs_by_id,
        local_outputs_hash_by_id,
        local_dirty_by_id,
        local_output_changed_by_id,
    )


def _execute_plan_parallel(
    recorded_steps: List[dict],
    subs: SubstitutionSet,
    executor: Executor,
    workers: int,
) -> ReplayResult:
    """Parallel variant of :func:`_execute_plan`.

    Partitions the trace DAG into regions using
    :py:func:`~stepback.distributed_dirty.partition_dag_regions`.
    Sequential "backbone" regions are processed inline.  Branch regions
    within the same fan-out tier are submitted to a
    :py:class:`~concurrent.futures.ThreadPoolExecutor` and execute
    concurrently; each branch's dirty steps call ``executor`` callbacks in
    parallel.  The join step (and any suffix steps) waits for all branch
    futures before executing, so "joins wait only on consumed inputs"
    is preserved by construction: every branch tail output is available in
    the global state before the join runs.

    **Thread safety**
    Each branch worker receives a read-only snapshot of the global state
    at the start of its tier.  Workers build their own local state dicts
    and never mutate the snapshot.  ``executor.real_calls`` /
    ``executor.fallback_uses`` are updated via locked helpers so the
    counters are accurate even under concurrent execution.

    **Exception semantics**
    If a branch worker raises (e.g. :class:`MissingExecutor` or a
    user-callback error), the exception propagates from
    :py:meth:`concurrent.futures.Future.result` after all futures in the
    tier complete (cancelled futures are not waited for).  Sibling branches
    that have *already started* will run to completion; branches that have
    not yet started may be cancelled by the pool.  This matches the
    contract of :class:`~concurrent.futures.ThreadPoolExecutor`.

    The result is **semantically identical** to :func:`_execute_plan` for
    any well-formed trace.  Use ``workers=1`` to disable parallelism while
    keeping the same code path (useful for debugging).

    Parameters
    ----------
    workers : int
        Maximum number of concurrent branch workers per fan-out tier.
        Must be ≥ 1.
    """
    from .distributed_dirty import partition_dag_regions

    started_at = _utc_now_iso()
    regions = partition_dag_regions(recorded_steps)

    # Fast path: single sequential region — no parallel work.
    if len(regions) <= 1:
        return _execute_plan(recorded_steps, subs, executor)

    # ---- build tier ordering (same BFS as compute_dirty_set_distributed) ----
    step_to_region: Dict[str, str] = {}
    for reg in regions:
        for s in reg.steps:
            step_to_region[s["step_id"]] = reg.region_id

    region_by_id = {r.region_id: r for r in regions}
    all_step_ids_in_region: Dict[str, set] = {
        r.region_id: {s["step_id"] for s in r.steps} for r in regions
    }

    region_deps: Dict[str, set] = {}
    for reg in regions:
        deps: set = set()
        for sid in reg.external_parent_ids:
            dep_rid = step_to_region.get(sid)
            if dep_rid:
                deps.add(dep_rid)
        region_deps[reg.region_id] = deps

    in_degree = {r.region_id: len(region_deps[r.region_id]) for r in regions}
    dependents: Dict[str, List[str]] = {r.region_id: [] for r in regions}
    for reg in regions:
        for dep_rid in region_deps[reg.region_id]:
            dependents[dep_rid].append(reg.region_id)

    tier: Dict[str, int] = {}
    queue = [r.region_id for r in regions if in_degree[r.region_id] == 0]
    for rid in queue:
        tier[rid] = 0

    processing = list(queue)
    while processing:
        next_wave: List[str] = []
        for rid in processing:
            for dep_rid in dependents[rid]:
                in_degree[dep_rid] -= 1
                tier[dep_rid] = max(tier.get(dep_rid, 0), tier[rid] + 1)
                if in_degree[dep_rid] == 0:
                    next_wave.append(dep_rid)
        processing = next_wave

    max_tier = max(tier.values()) if tier else 0
    tier_groups: List[List[str]] = [[] for _ in range(max_tier + 1)]
    for rid, t in tier.items():
        tier_groups[t].append(rid)

    # ---- process tier by tier ----
    global_outputs_by_id: Dict[str, Any] = {}
    global_outputs_hash_by_id: Dict[str, str] = {}
    global_dirty_by_id: Dict[str, bool] = {}
    global_output_changed_by_id: Dict[str, bool] = {}

    # step_id → StepView, populated as tiers complete; assembled in order at end.
    step_view_by_id: Dict[str, StepView] = {}
    total_cost = 0.0
    dirty_n = 0
    hit_n = 0

    actual_workers = max(1, workers)

    for tier_rids in tier_groups:
        if len(tier_rids) == 1 or actual_workers == 1:
            # Sequential: process each region in the tier without thread overhead.
            for rid in tier_rids:
                region_steps = region_by_id[rid].steps
                (
                    sv_list, cost, d_n, h_n,
                    loc_out, loc_hash, loc_dirty, loc_changed,
                ) = _execute_branch_steps(
                    region_steps, subs, executor,
                    global_outputs_by_id, global_outputs_hash_by_id,
                    global_dirty_by_id, global_output_changed_by_id,
                )
                for sv in sv_list:
                    step_view_by_id[sv.step_id] = sv
                total_cost += cost
                dirty_n += d_n
                hit_n += h_n
                global_outputs_by_id.update(loc_out)
                global_outputs_hash_by_id.update(loc_hash)
                global_dirty_by_id.update(loc_dirty)
                global_output_changed_by_id.update(loc_changed)
        else:
            # Parallel: snapshot global state so all workers in this tier see
            # the same ancestor outputs.  Workers never mutate the snapshot.
            snap_out = dict(global_outputs_by_id)
            snap_hash = dict(global_outputs_hash_by_id)
            snap_dirty = dict(global_dirty_by_id)
            snap_changed = dict(global_output_changed_by_id)

            n_workers = min(actual_workers, len(tier_rids))
            tier_results: Dict[str, tuple] = {}

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(
                        _execute_branch_steps,
                        region_by_id[rid].steps, subs, executor,
                        snap_out, snap_hash, snap_dirty, snap_changed,
                    ): rid
                    for rid in tier_rids
                }
                for fut in as_completed(futures):
                    rid = futures[fut]
                    tier_results[rid] = fut.result()  # propagates exceptions

            # Merge all branch results into global state.  Branch regions
            # are disjoint so merge order does not affect correctness.
            for rid in tier_rids:
                sv_list, cost, d_n, h_n, loc_out, loc_hash, loc_dirty, loc_changed = tier_results[rid]
                for sv in sv_list:
                    step_view_by_id[sv.step_id] = sv
                total_cost += cost
                dirty_n += d_n
                hit_n += h_n
                global_outputs_by_id.update(loc_out)
                global_outputs_hash_by_id.update(loc_hash)
                global_dirty_by_id.update(loc_dirty)
                global_output_changed_by_id.update(loc_changed)

    # Assemble final ReplayResult in original topological step order.
    steps_view = [step_view_by_id[rec["step_id"]] for rec in recorded_steps]
    real_n = executor.real_calls  # accurate because _inc_real is locked
    finished_at = _utc_now_iso()
    return ReplayResult(
        steps=steps_view,
        total_cost_usd=round(total_cost, 8),
        dirty_count=dirty_n,
        cache_hit_count=hit_n,
        real_executions=real_n,
        provenance=ReplayProvenance(
            executor_version=_STEPBACK_VERSION,
            started_at=started_at,
            finished_at=finished_at,
        ),
    )



def replay_events(
    recorded_steps: List[dict],
    subs: SubstitutionSet,
    executor: Executor,
) -> Generator[dict, None, None]:
    """Streaming variant of :func:`_execute_plan`.

    Yields one ``"step_complete"`` event dict per step, then one
    ``"replay_done"`` summary event at the end.  On exception yields a
    single ``"error"`` event.

    Callers (HTTP NDJSON, gRPC server-streaming) consume this generator and
    forward each event to the client as it is produced, enabling incremental
    progress reporting for long traces.

    Event shapes::

        # emitted once per step
        {"event": "step_complete", "step_id": "...", "step_kind": "...",
         "dirty": bool, "cache_hit": bool, "cost_usd": float,
         "current_inputs_hash": "...", "recorded_inputs_hash": "...",
         "output_changed": bool}

        # emitted once at the end
        {"event": "replay_done", "step_count": N, "dirty_count": N,
         "cache_hit_count": N, "total_cost_usd": float, "real_executions": N}

        # emitted on error (generator then stops)
        {"event": "error", "error": "<message>"}
    """
    outputs_by_id: Dict[str, Any] = {}
    outputs_hash_by_id: Dict[str, str] = {}
    dirty_by_id: Dict[str, bool] = {}
    output_changed_by_id: Dict[str, bool] = {}
    total_cost = 0.0
    dirty_n = 0
    hit_n = 0
    real_n = 0
    sentinel = object()

    try:
        for rec in recorded_steps:
            sid = rec["step_id"]
            kind = rec["step_kind"]
            recorded_inputs_hash = rec["inputs_hash"]
            cur_inputs = copy.deepcopy(rec["inputs"])

            parent_id = rec.get("parent_step_id")
            context_fields: Optional[List[str]] = cur_inputs.get("_stepback_context_fields")
            if parent_id and "context" in cur_inputs and parent_id in outputs_by_id:
                if context_fields:
                    parent_out = outputs_by_id[parent_id]
                    if isinstance(parent_out, dict):
                        partial = {k: parent_out[k] for k in context_fields if k in parent_out}
                        cur_inputs["context"] = hash_obj(partial)
                    else:
                        cur_inputs["context"] = outputs_hash_by_id.get(parent_id, cur_inputs["context"])
                else:
                    cur_inputs["context"] = outputs_hash_by_id[parent_id]
            elif parent_id and "context" in cur_inputs and parent_id in outputs_hash_by_id and not context_fields:
                cur_inputs["context"] = outputs_hash_by_id[parent_id]

            parent_ids: List[str] = list(rec.get("parent_step_ids") or [])
            if "branch_tails" in cur_inputs and "branch_tail_hashes" in cur_inputs:
                tails = cur_inputs["branch_tails"]
                cur_inputs["branch_tail_hashes"] = [
                    outputs_hash_by_id.get(t, cur_inputs["branch_tail_hashes"][i])
                    for i, t in enumerate(tails)
                ]

            tool_override: Any = sentinel
            for sub in subs.at(sid):
                if sub.is_output_forcing():
                    tool_override = sub.force_output(rec)
                else:
                    sub.apply(cur_inputs, rec)

            current_inputs_hash = hash_obj(cur_inputs)
            parent_dirty = dirty_by_id.get(parent_id, False) if parent_id else False
            if parent_ids:
                parent_dirty = parent_dirty or any(
                    dirty_by_id.get(pid, False) for pid in parent_ids
                )

            parent_output_changed = (
                output_changed_by_id.get(parent_id, False) if parent_id else False
            )
            if parent_ids:
                parent_output_changed = parent_output_changed or any(
                    output_changed_by_id.get(pid, False) for pid in parent_ids
                )

            use_parent_dirty = parent_output_changed and not context_fields

            recorded_nondet_hash = rec.get("nondeterminism_hash")
            if recorded_nondet_hash is not None:
                live_nondet_hash = sha256_hex(
                    canonical_json(rec.get("nondeterminism", {}))
                )
                nondet_dirty = recorded_nondet_hash != live_nondet_hash
            else:
                nondet_dirty = False

            nondet_class_dirty = _nondet_forces_dirty(rec.get("nondeterminism", {}))

            _ev_is_fallback = False
            _ev_is_persistent_cache = False

            if tool_override is not sentinel:
                cur_outputs = tool_override
                is_dirty = True
                cache_hit = False
            elif (
                current_inputs_hash == recorded_inputs_hash
                and not use_parent_dirty
                and not nondet_dirty
                and not nondet_class_dirty
            ):
                cur_outputs = rec["outputs"]
                is_dirty = False
                cache_hit = True
                hit_n += 1
            else:
                if executor.fallback_recorded and (
                    (kind == "llm_call" and executor.llm is None)
                    or (kind == "tool_call" and executor.tool is None)
                    or (kind == "router" and executor.router is None)
                    or (kind == "parallel_branch_join" and executor.join is None
                        and not parent_ids)
                ):
                    cur_outputs = rec["outputs"]
                    executor._inc_fallback()
                    _ev_is_fallback = True
                else:
                    # Check persistent step cache before calling the executor.
                    # Bypass for nondeterminism-class-forced steps (Step 63).
                    _cached_entry = (
                        None
                        if (nondet_dirty or nondet_class_dirty)
                        else executor._cache_get(kind, current_inputs_hash)
                    )
                    if _cached_entry is not None:
                        cur_outputs = _cached_entry.outputs
                        _ev_is_persistent_cache = True
                    elif kind == "parallel_branch_join" and parent_ids:
                        b_outs = [outputs_by_id[pid] for pid in parent_ids]
                        try:
                            cur_outputs = executor.execute(
                                kind, cur_inputs, branch_outputs=b_outs
                            )
                            real_n += 1
                            executor._cache_put(kind, current_inputs_hash, cur_outputs)
                        except UnavailableExecutorError:
                            if executor.fallback_recorded:
                                cur_outputs = rec["outputs"]
                                executor._inc_fallback()
                                _ev_is_fallback = True
                            else:
                                raise
                    else:
                        try:
                            cur_outputs = executor.execute(kind, cur_inputs)
                            real_n += 1
                            executor._cache_put(kind, current_inputs_hash, cur_outputs)
                        except UnavailableExecutorError:
                            if executor.fallback_recorded:
                                cur_outputs = rec["outputs"]
                                executor._inc_fallback()
                                _ev_is_fallback = True
                            else:
                                raise
                is_dirty = True
                cache_hit = False

            if cache_hit:
                cost = float(rec.get("cost_usd", 0.0))
            else:
                if kind == "llm_call" and isinstance(cur_outputs, dict):
                    cost = compute_cost(
                        cur_inputs.get("model", ""), cur_outputs.get("usage", {})
                    )
                else:
                    cost = float(rec.get("cost_usd", 0.0))

            outputs_by_id[sid] = cur_outputs
            cur_hash = hash_obj(cur_outputs)
            outputs_hash_by_id[sid] = cur_hash
            dirty_by_id[sid] = is_dirty
            output_changed_by_id[sid] = is_dirty and cur_hash != hash_obj(rec["outputs"])
            total_cost += cost
            if is_dirty:
                dirty_n += 1

            # Compute dirty_reason for the streaming event.
            if not is_dirty:
                _ev_dirty_reason: Optional[str] = None
            elif tool_override is not sentinel:
                _ev_dirty_reason = "output_forced"
            elif _ev_is_fallback:
                _ev_dirty_reason = _structural_dirty_reason(
                    current_inputs_hash, recorded_inputs_hash,
                    nondet_dirty, nondet_class_dirty, use_parent_dirty,
                )
            else:
                _ev_dirty_reason = _structural_dirty_reason(
                    current_inputs_hash, recorded_inputs_hash,
                    nondet_dirty, nondet_class_dirty, use_parent_dirty,
                )

            yield {
                "event": "step_complete",
                "step_id": sid,
                "step_kind": kind,
                "dirty": is_dirty,
                "cache_hit": cache_hit,
                "cost_usd": round(cost, 8),
                "current_inputs_hash": current_inputs_hash,
                "recorded_inputs_hash": recorded_inputs_hash,
                "output_changed": output_changed_by_id[sid],
                "dirty_reason": _ev_dirty_reason,
                "executor_version": _STEPBACK_VERSION,
            }

    except Exception as exc:  # noqa: BLE001
        yield {"event": "error", "error": str(exc)}
        return

    yield {
        "event": "replay_done",
        "step_count": len(recorded_steps),
        "dirty_count": dirty_n,
        "cache_hit_count": hit_n,
        "total_cost_usd": round(total_cost, 8),
        "real_executions": real_n,
    }


# -------------------------------------------------------------- Trace


@dataclass
class BisectTarget:
    """One regression to locate via :py:meth:`Trace.bisect_multi`.

    Attributes
    ----------
    good:
        Step id at which ``predicate`` is known to be ``False``
        (i.e. the regression has *not* yet appeared).
    bad:
        Step id at which ``predicate`` is known to be ``True``
        (i.e. the regression *has* appeared).
    predicate:
        A callable that accepts a :class:`StepView` and returns ``True``
        once the regression is visible.
    """

    good: str
    bad: str
    predicate: Callable[["StepView"], bool]


@dataclass
class BisectMultiResult:
    """Results of an incremental multi-regression bisect.

    Attributes
    ----------
    culprits:
        One entry per :class:`BisectTarget` (in the same order).
        Each entry is the earliest :class:`StepView` where the
        corresponding predicate became ``True``, or ``None`` if no
        step in the target range matched.
    total_probes:
        Sum of binary-search array accesses across all targets.
        Shared replay means this is *not* ``N × probes_per_target``.
    """

    culprits: List[Optional["StepView"]]
    total_probes: int


@dataclass
class Trace:
    """A loaded `.sb` trace ready for navigation, substitution, replay."""

    path: str
    header: dict
    recorded_steps: List[dict]
    cursor: int = 0
    pending_subs: SubstitutionSet = field(default_factory=SubstitutionSet)
    _last_bisect_probes: int = field(default=0, repr=False)
    _id_index: Dict[str, int] = field(default_factory=dict, repr=False)

    @property
    def last_bisect_probes(self) -> int:
        """Number of step probes taken during the most recent bisect call.

        This counter is updated by both :py:meth:`bisect` (single-target)
        and :py:meth:`bisect_multi` (multi-target).  For ``bisect_multi``
        the value is the *sum* of probe counts across all targets, reflecting
        the total search work — not the per-target count.
        """
        return self._last_bisect_probes

    # ------------------------------------------------- navigation
    def goto(self, step_id: str) -> "Trace":
        self.cursor = self._idx(step_id)
        return self

    def step_back(self, *, to: Optional[str] = None) -> "Trace":
        if to is not None:
            return self.goto(to)
        if self.cursor > 0:
            self.cursor -= 1
        return self

    def step_forward(self) -> "Trace":
        if self.cursor < len(self.recorded_steps) - 1:
            self.cursor += 1
        return self

    def current_step(self) -> dict:
        return self.recorded_steps[self.cursor]

    # ------------------------------------------------ substitution
    def substitute(self, *subs: Substitution) -> "Trace":
        for s in subs:
            self.pending_subs.add(s)
        return self

    def reset_substitutions(self) -> "Trace":
        self.pending_subs = SubstitutionSet()
        return self

    # --------------------------------------------------- branches
    def branch_at(self, step_id: str, name: str) -> Branch:
        self._idx(step_id)  # validate
        return Branch(name=name, base_step=step_id, _owner=self)

    def compare_branches(self, a: Branch, b: Branch) -> BranchDiff:
        ra = a.result or a.replay_forward()
        rb = b.result or b.replay_forward()
        diffs: List[StepDiff] = []
        cost_delta = 0.0
        diverged = 0
        ids = {s.step_id for s in ra.steps} | {s.step_id for s in rb.steps}
        a_by = {s.step_id: s for s in ra.steps}
        b_by = {s.step_id: s for s in rb.steps}
        for sid in sorted(ids, key=lambda x: int(x.split(":")[-1])):
            sa = a_by.get(sid)
            sb = b_by.get(sid)
            present = sa or sb
            assert present is not None
            kind = present.kind
            ao = sa.outputs if sa else None
            bo = sb.outputs if sb else None
            same = ao == bo
            d = {} if same else {"a": ao, "b": bo}
            ca = sa.cost_usd if sa else 0.0
            cb = sb.cost_usd if sb else 0.0
            cost_delta += cb - ca
            if not same:
                diverged += 1
            diffs.append(
                StepDiff(
                    step_id=sid,
                    kind=kind,
                    output_diff=d,
                    cost_delta_usd=round(cb - ca, 8),
                    diverged_from_cache=bool(sb and sb.dirty) or bool(sa and sa.dirty),
                )
            )
        return BranchDiff(
            a=a.name,
            b=b.name,
            step_diffs=diffs,
            total_cost_delta_usd=round(cost_delta, 8),
            divergent_step_count=diverged,
        )

    # ----------------------------------------------------- replay
    def replay_forward(
        self,
        executor: Optional[Executor] = None,
        *,
        workers: Optional[int] = None,
        distributed: bool = False,
    ) -> ReplayResult:
        """Replay the trace from the beginning, applying any pending substitutions.

        Parameters
        ----------
        executor :
            Callbacks for re-executing dirty steps.  Defaults to a no-op
            :class:`Executor` (cache-hit-only replay).
        workers :
            Number of worker threads for parallel branch execution.
            ``None`` (default) or ``1`` → sequential execution (no thread
            overhead).  ``N > 1`` → submits independent branch regions to a
            :class:`~concurrent.futures.ThreadPoolExecutor` with *N* threads.
            The result is semantically identical to sequential replay.
        distributed :
            When ``True``, enables distributed scheduling mode backed by the
            same local :class:`~concurrent.futures.ThreadPoolExecutor` planner
            (Step 73).  If *workers* is not specified, an appropriate count is
            chosen automatically via ``min(8, os.cpu_count() or 4)``.

            ``distributed=True, workers=N`` is equivalent to ``workers=N``
            when N > 1.  ``distributed=False`` (the default) leaves all
            existing behaviour unchanged.

            .. note::
               "Distributed" here refers to distributing independent branch
               regions across local worker threads, not cross-machine
               distribution.  Cross-machine distribution requires the
               ``stepback-proxy`` event bus (Steps 71, 75).
        """
        return self.run_replay(self.pending_subs, executor or Executor(),
                               workers=workers, distributed=distributed)

    def plan_replay(self, subs: Optional[SubstitutionSet] = None) -> "ReplayPlan":
        """Produce a :class:`ReplayPlan` for the pending (or given) substitutions.

        The plan phase walks the recorded trace in topological order and
        classifies each step as ``"cache_hit"`` or ``"execute"`` **without
        calling any executor**.  Dirty-parent outputs are **not** available
        during planning; recorded outputs are used as placeholders, so the
        plan is an *optimistic estimate* — it may under-count dirty steps
        when a dirty parent produces a new output that cascades dirtiness to
        its children.

        Steps that have a dirty parent in the plan are marked with
        ``requires_runtime_validation=True`` to signal that the executor
        phase must re-evaluate their decision against the actual parent
        output.

        The plan is intentionally cheap: no executor callbacks, no cost
        computation.  Call :meth:`ReplayPlan.execute` (or pass the plan to
        the executor phase) to obtain the final :class:`ReplayResult`.
        """
        effective_subs = subs if subs is not None else self.pending_subs
        planned: List[PlannedStep] = []
        dirty_n = 0
        hit_n = 0

        # Planner-local state: use recorded outputs as placeholders.
        planner_outputs_hash_by_id: Dict[str, str] = {}
        planner_dirty_by_id: Dict[str, bool] = {}

        sentinel = object()

        for rec in self.recorded_steps:
            sid = rec["step_id"]
            kind = rec["step_kind"]
            recorded_inputs_hash = rec["inputs_hash"]
            cur_inputs = copy.deepcopy(rec["inputs"])

            parent_id = rec.get("parent_step_id")
            context_fields: Optional[List[str]] = cur_inputs.get("_stepback_context_fields")

            # Rebind using recorded outputs (or already-planned hash if the
            # parent was planned dirty — still uses recorded as placeholder).
            if parent_id and "context" in cur_inputs and parent_id in planner_outputs_hash_by_id:
                if context_fields:
                    # Use the recorded parent output dict for partial hashing.
                    parent_rec_out = None
                    for r in self.recorded_steps:
                        if r["step_id"] == parent_id:
                            parent_rec_out = r.get("outputs")
                            break
                    if isinstance(parent_rec_out, dict):
                        partial = {k: parent_rec_out[k] for k in context_fields if k in parent_rec_out}
                        cur_inputs["context"] = hash_obj(partial)
                    else:
                        cur_inputs["context"] = planner_outputs_hash_by_id[parent_id]
                else:
                    cur_inputs["context"] = planner_outputs_hash_by_id[parent_id]
            elif parent_id and "context" in cur_inputs and parent_id in planner_outputs_hash_by_id and not context_fields:
                cur_inputs["context"] = planner_outputs_hash_by_id[parent_id]

            parent_ids: List[str] = list(rec.get("parent_step_ids") or [])
            if "branch_tails" in cur_inputs and "branch_tail_hashes" in cur_inputs:
                tails = cur_inputs["branch_tails"]
                cur_inputs["branch_tail_hashes"] = [
                    planner_outputs_hash_by_id.get(t, cur_inputs["branch_tail_hashes"][i])
                    for i, t in enumerate(tails)
                ]

            # Apply substitutions.
            tool_override: Any = sentinel
            for sub in effective_subs.at(sid):
                if sub.is_output_forcing():
                    tool_override = sub.force_output(rec)
                else:
                    sub.apply(cur_inputs, rec)

            current_inputs_hash = hash_obj(cur_inputs)

            # Nondeterminism checks (same logic as run_replay).
            recorded_nondet_hash = rec.get("nondeterminism_hash")
            if recorded_nondet_hash is not None:
                live_nondet_hash = sha256_hex(canonical_json(rec.get("nondeterminism", {})))
                nondet_dirty = recorded_nondet_hash != live_nondet_hash
            else:
                nondet_dirty = False
            nondet_class_dirty = _nondet_forces_dirty(rec.get("nondeterminism", {}))

            parent_dirty = planner_dirty_by_id.get(parent_id, False) if parent_id else False
            if parent_ids:
                parent_dirty = parent_dirty or any(
                    planner_dirty_by_id.get(pid, False) for pid in parent_ids
                )

            # Determine planned action.
            dirty_reason: Optional[str] = None
            if tool_override is not sentinel:
                planned_action = "execute"
                dirty_reason = "output-forcing substitution"
                is_planned_dirty = True
            elif (
                current_inputs_hash == recorded_inputs_hash
                and not nondet_dirty
                and not nondet_class_dirty
                and not parent_dirty
            ):
                planned_action = "cache_hit"
                is_planned_dirty = False
            else:
                planned_action = "execute"
                is_planned_dirty = True
                if tool_override is not sentinel:
                    dirty_reason = "output-forcing substitution"
                elif current_inputs_hash != recorded_inputs_hash:
                    dirty_reason = "inputs hash changed"
                elif nondet_dirty:
                    dirty_reason = "nondeterminism_hash mismatch"
                elif nondet_class_dirty:
                    dirty_reason = "nondeterminism class forces re-execution"
                else:
                    dirty_reason = "ancestor dirty"

            # A step is requires_runtime_validation if a dirty parent exists
            # (the planner used recorded output as placeholder — real output
            # may cascade dirtiness differently).
            requires_validation = parent_dirty and not context_fields

            # Store recorded output hash for downstream steps.
            planner_outputs_hash_by_id[sid] = hash_obj(rec.get("outputs"))
            planner_dirty_by_id[sid] = is_planned_dirty

            if is_planned_dirty:
                dirty_n += 1
            else:
                hit_n += 1

            planned.append(
                PlannedStep(
                    step_id=sid,
                    kind=kind,
                    name=rec.get("name"),
                    parent_step_id=parent_id,
                    recorded_inputs_hash=recorded_inputs_hash,
                    planned_inputs=cur_inputs,
                    planned_inputs_hash=current_inputs_hash,
                    planned_action=planned_action,
                    dirty_reason=dirty_reason,
                    requires_runtime_validation=requires_validation,
                )
            )

        return ReplayPlan(
            planned_steps=planned,
            estimated_dirty_count=dirty_n,
            estimated_cache_hit_count=hit_n,
            _recorded_steps=self.recorded_steps,
            _subs=effective_subs,
        )

    def run_replay(
        self,
        subs: SubstitutionSet,
        executor: Executor,
        *,
        workers: Optional[int] = None,
        distributed: bool = False,
        on_step: Optional[Callable[[int, int], None]] = None,
    ) -> ReplayResult:
        """Full planning + execution in a single pass.

        Delegates to :func:`_execute_plan` (sequential) or
        :func:`_execute_plan_parallel` (parallel) depending on *workers*
        and *distributed*.

        Parameters
        ----------
        workers :
            See :meth:`replay_forward`.  ``None`` or ``1`` → sequential.
        distributed :
            See :meth:`replay_forward`.  When ``True`` and *workers* is not
            given, auto-selects a worker count.
        on_step :
            Optional progress callback ``(steps_done, total_steps)`` invoked
            after every step.  Only used in sequential mode; ignored when
            *workers* > 1.
        """
        eff_workers = _effective_workers(distributed, workers)
        if eff_workers is not None and eff_workers > 1:
            return _execute_plan_parallel(self.recorded_steps, subs, executor, eff_workers)
        return _execute_plan(self.recorded_steps, subs, executor, on_step=on_step)

    # --------------------------------------------------- minimize
    def minimize(
        self,
        substitutions: "SubstitutionSet",
        predicate: Callable[["ReplayResult"], bool],
        *,
        executor: Optional[Executor] = None,
    ):
        """Delta-debug ``substitutions`` to a 1-minimal triggering subset.

        Convenience wrapper around
        :func:`stepback.minimize.ddmin_substitutions`.
        """
        from .minimize import ddmin_substitutions
        return ddmin_substitutions(
            self, substitutions, predicate, executor=executor,
        )

    def attribute_steps(
        self,
        predicate: Callable[["ReplayResult"], bool],
        *,
        executor: Optional[Executor] = None,
        permutations: Optional[int] = None,
        rng_seed: int = 0xC0DE,
    ):
        """Shapley-value attribution for steps that jointly cause a failure.

        Convenience wrapper around :func:`stepback.minimize.attribute_steps`.

        Runs the trace once, then computes the Shapley value of each step
        under the coalition game
        ``v(S) = predicate(result masked to steps in S)``.

        Parameters
        ----------
        predicate :
            A callable ``(ReplayResult) -> bool`` that must return ``True``
            for the full replay; raises
            :class:`~stepback.minimize.PredicateNotTriggered` otherwise.
        executor :
            Optional executor; defaults to no-op stub.
        permutations :
            Number of random orderings for the sampled estimator (used when
            the trace has more than 8 steps).
        rng_seed :
            Seed for the permutation estimator.

        Returns
        -------
        StepAttributionResult
        """
        from .minimize import attribute_steps as _attribute_steps
        return _attribute_steps(
            self, predicate, executor=executor,
            permutations=permutations, rng_seed=rng_seed,
        )


    def bisect(
        self,
        good: str,
        bad: str,
        predicate: Callable[[StepView], bool],
        executor: Optional[Executor] = None,
    ) -> Optional[StepView]:
        """Binary-search the linearised step range ``[good, bad]`` for the
        earliest step where ``predicate(step)`` becomes true.

        With no substitutions in flight, every probe is a cache hit so
        the entire bisection is zero-LLM-call.
        """
        executor = executor or Executor()
        result = self.run_replay(self.pending_subs, executor)
        gi = self._idx(good)
        bi = self._idx(bad)
        if gi > bi:
            gi, bi = bi, gi
        lo, hi = gi, bi
        candidate: Optional[StepView] = None
        probes = 0
        while lo <= hi:
            probes += 1
            mid = (lo + hi) // 2
            sv = result.steps[mid]
            if predicate(sv):
                candidate = sv
                hi = mid - 1
            else:
                lo = mid + 1
        self._last_bisect_probes = probes
        return candidate

    def bisect_multi(
        self,
        targets: List[BisectTarget],
        executor: Optional[Executor] = None,
    ) -> BisectMultiResult:
        """Locate culprit steps for multiple regressions in a single replay.

        Runs the trace exactly **once** (a cache-hit replay when no
        substitutions are pending) and applies each target's predicate
        independently via binary search.  Finding one culprit does **not**
        restart the search for the others; all targets share the same replay
        result, so the total LLM-call cost is that of a single
        :py:meth:`replay_forward` regardless of how many targets are given.

        Parameters
        ----------
        targets:
            A list of :class:`BisectTarget` descriptors.  Each specifies a
            *good* step id (predicate is ``False``), a *bad* step id
            (predicate is ``True``), and the predicate itself.
        executor:
            Optional :class:`Executor`; defaults to the stub that always
            returns cache hits.

        Returns
        -------
        BisectMultiResult
            ``culprits[i]`` is the earliest :class:`StepView` in
            ``targets[i]``'s range where the predicate is ``True``, or
            ``None`` if no step matched.  ``total_probes`` is the sum of
            per-target binary-search array accesses.
        """
        executor = executor or Executor()
        result = self.run_replay(self.pending_subs, executor)
        culprits: List[Optional[StepView]] = []
        total_probes = 0
        for target in targets:
            gi = self._idx(target.good)
            bi = self._idx(target.bad)
            if gi > bi:
                gi, bi = bi, gi
            lo, hi = gi, bi
            candidate: Optional[StepView] = None
            probes = 0
            while lo <= hi:
                probes += 1
                mid = (lo + hi) // 2
                sv = result.steps[mid]
                if target.predicate(sv):
                    candidate = sv
                    hi = mid - 1
                else:
                    lo = mid + 1
            culprits.append(candidate)
            total_probes += probes
        self._last_bisect_probes = total_probes
        return BisectMultiResult(culprits=culprits, total_probes=total_probes)

    # --------------------------------------------------- internals
    def _idx(self, step_id: str) -> int:
        if not self._id_index:
            self._id_index = {s["step_id"]: i for i, s in enumerate(self.recorded_steps)}
        if step_id not in self._id_index:
            raise KeyError(f"unknown step_id {step_id!r}")
        return self._id_index[step_id]


# --------------------------------------------------------- entry point


def replay(path: str, *, hmac_key: Optional[bytes] = None) -> Trace:
    """Load ``path`` and return a :py:class:`Trace`.

    If ``hmac_key`` is given, the entire chain + every Ed25519
    signature is verified before the trace is returned. Otherwise the
    file is parsed without verification (useful for inspection of
    third-party traces whose key you don't possess).
    """
    if hmac_key is not None:
        verified = verify_trace(path, hmac_key)
        header = verified.header
        steps = verified.steps
    else:
        from .trace_reader import read_frames, _decode_blob, _decode_gz_step, _materialise
        frames = read_frames(path)
        header = next(f["body"] for f in frames if f["body"].get("type") == "header")
        blobs: dict = {}
        steps = []
        for f in frames:
            body = f["body"]
            t = body.get("type")
            if t == "blob":
                blobs[body["id"]] = _decode_blob(body)
            elif t == "step":
                step = _decode_gz_step(body)
                if blobs:
                    step = _materialise(step, blobs)
                steps.append(step)
    return Trace(path=path, header=header, recorded_steps=steps)
