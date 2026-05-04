"""Replay engine.

The whole point: walk the recorded trace in topological order and, for
each step, decide cache-hit vs. dirty. A step is a *cache hit* iff:

* its current inputs (post-substitution and post-parent-rebinding)
  hash to the recorded ``inputs_hash``, and
* no ancestor it depends on was dirty, and
* its recorded ``nondeterminism_hash`` is unchanged.

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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .canonical import hash_obj
from .pricing import compute_cost
from .substitutions import (
    Substitution,
    SubstitutionSet,
    ToolOutputSubstitution,
)
from .trace_reader import TraceVerificationError, verify_trace  # noqa: F401 (reexport)


# ------------------------------------------------------------ Executor


class MissingExecutor(RuntimeError):
    """Raised when a dirty step has no executor callback registered."""


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
    ) -> None:
        self.llm = llm
        self.tool = tool
        self.router = router
        self.join = join
        self.fallback_recorded = fallback_recorded
        self.real_calls: int = 0
        self.fallback_uses: int = 0

    def execute(self, kind: str, inputs: dict, *, branch_outputs: Optional[List[Any]] = None) -> Any:
        self.real_calls += 1
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

    def replay_forward(self, executor: Optional[Executor] = None) -> ReplayResult:
        assert self._owner is not None, "Branch must be created via Trace.branch_at"
        self.result = self._owner.run_replay(self.substitutions, executor or Executor())
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


# -------------------------------------------------------------- Trace


@dataclass
class Trace:
    """A loaded `.sb` trace ready for navigation, substitution, replay."""

    path: str
    header: dict
    recorded_steps: List[dict]
    cursor: int = 0
    pending_subs: SubstitutionSet = field(default_factory=SubstitutionSet)
    last_bisect_probes: int = 0
    _id_index: Dict[str, int] = field(default_factory=dict, repr=False)

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
    def replay_forward(self, executor: Optional[Executor] = None) -> ReplayResult:
        return self.run_replay(self.pending_subs, executor or Executor())

    def run_replay(self, subs: SubstitutionSet, executor: Executor) -> ReplayResult:
        outputs_by_id: Dict[str, Any] = {}
        outputs_hash_by_id: Dict[str, str] = {}
        dirty_by_id: Dict[str, bool] = {}
        steps_view: List[StepView] = []
        total_cost = 0.0
        dirty_n = 0
        hit_n = 0
        real_n = 0
        sentinel = object()

        for rec in self.recorded_steps:
            sid = rec["step_id"]
            kind = rec["step_kind"]
            recorded_inputs_hash = rec["inputs_hash"]
            cur_inputs = copy.deepcopy(rec["inputs"])

            # Rebind any "context" from the parent's *current* outputs hash.
            parent_id = rec.get("parent_step_id")
            if parent_id and "context" in cur_inputs and parent_id in outputs_hash_by_id:
                cur_inputs["context"] = outputs_hash_by_id[parent_id]

            # Multi-parent rebinding: parallel_branch_join carries
            # `branch_tail_hashes` (one per branch tail). Rebind each
            # to the current output hash so a substitution inside any
            # branch flows through the join.
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
            # Multi-parent dirtiness for parallel joins.
            if parent_ids:
                parent_dirty = parent_dirty or any(
                    dirty_by_id.get(pid, False) for pid in parent_ids
                )

            if tool_override is not sentinel:
                cur_outputs = tool_override
                is_dirty = True
                cache_hit = False
            elif current_inputs_hash == recorded_inputs_hash and not parent_dirty:
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
                    executor.fallback_uses += 1
                else:
                    if kind == "parallel_branch_join" and parent_ids:
                        b_outs = [outputs_by_id[pid] for pid in parent_ids]
                        cur_outputs = executor.execute(
                            kind, cur_inputs, branch_outputs=b_outs
                        )
                    else:
                        cur_outputs = executor.execute(kind, cur_inputs)
                    real_n += 1
                is_dirty = True
                cache_hit = False

            # Cost: prefer recorded on cache hit; otherwise recompute from
            # usage if available; otherwise fall back to the recorded cost.
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
            outputs_hash_by_id[sid] = hash_obj(cur_outputs)
            dirty_by_id[sid] = is_dirty
            total_cost += cost
            if is_dirty:
                dirty_n += 1

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
                )
            )

        return ReplayResult(
            steps=steps_view,
            total_cost_usd=round(total_cost, 8),
            dirty_count=dirty_n,
            cache_hit_count=hit_n,
            real_executions=real_n,
        )

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

    # ----------------------------------------------------- bisect
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
        self.last_bisect_probes = probes
        return candidate

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
