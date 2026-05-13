"""Distributed dirty-set computation over a thread-pool worker pool.

Partitions the trace DAG into independent regions and classifies them
in parallel using :py:class:`concurrent.futures.ThreadPoolExecutor`.
Each region is a contiguous subgraph whose dirty-set classification
depends only on the outputs of steps in earlier (already-classified)
regions.

The result is *identical* to :py:func:`~stepback.divergence.compute_dirty_set`
— the distribution is an implementation optimisation for traces with
large parallel-branch fan-outs where each branch is an independent
unit of classification work.

**Region structure**

Given a trace with parallel branches::

    s1 → s2 (parallel_branch_open)
         ↓           ↓
    s3 (branch A)  s4 (branch B)
         ↓           ↓
    s5 (parallel_branch_join ← s3, s4)
         ↓
    s6 (suffix)

The partitioner produces four regions:

* ``seq:0``          — [s1, s2]          (prefix, sequential)
* ``branch:s3``      — [s3]              (branch A, parallel)
* ``branch:s4``      — [s4]              (branch B, parallel)
* ``seq:1``          — [s5, s6]          (suffix, sequential)

Tiers 0 and 2 are processed sequentially; tier 1 is processed with
up to *workers* threads in parallel.

**Thread safety**

Each worker receives a read-only snapshot of the global state dicts.
Workers never mutate the snapshot; they produce their own local state
dicts.  Summaries are merged into the global state sequentially after
each tier completes.  The only shared object read concurrently is
:py:class:`~stepback.substitutions.SubstitutionSet`, whose
:py:meth:`~stepback.substitutions.SubstitutionSet.at` method is
read-only (returns a new list).

Public surface
--------------
* :py:func:`compute_dirty_set_distributed` — distributed variant of
  :py:func:`~stepback.divergence.compute_dirty_set`.
* :py:class:`DagRegion` — a labelled, independent subgraph of the
  trace DAG with its external parent set.
* :py:class:`RegionSummary` — per-region classification result.
* :py:func:`partition_dag_regions` — split a trace step list into
  regions for inspection and testing.

Usage::

    from stepback import replay
    from stepback.distributed_dirty import compute_dirty_set_distributed
    from stepback.substitutions import PromptSubstitution

    trace = replay("incident.sb")
    summary = compute_dirty_set_distributed(
        trace,
        [PromptSubstitution(at_step="step:llm_call:1", new_messages=[...])],
        workers=4,
    )
    print(f"dirty: {summary.dirty_count}/{summary.step_count}")
    print(f"calls saved: {summary.calls_saved}")
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set

from .canonical import canonical_json, hash_obj, sha256_hex
from .divergence import DirtySetEntry, DirtySetSummary
from .nondeterminism import forces_dirty as _nondet_forces_dirty

# ------------------------------------------------------------------ types


@dataclass
class DagRegion:
    """A labelled, contiguous subgraph of the trace DAG.

    Each region holds a list of steps (in topological order within the
    region) that can be classified as a unit once the outputs and dirty
    status of their *external parents* (steps in earlier regions) are
    known.

    Fields
    ------
    region_id : str
        Human-readable identifier, e.g. ``"seq:0"``, ``"branch:step:3"``,
        ``"seq:1"``.
    steps : List[dict]
        Recorded step dicts in topological order within the region.
    external_parent_ids : FrozenSet[str]
        Step IDs that are parents of steps in this region but belong to
        earlier regions.  The classifier uses their output hashes and
        dirty flags from the global state snapshot.
    """

    region_id: str
    steps: List[dict]
    external_parent_ids: FrozenSet[str]


@dataclass
class RegionSummary:
    """Per-region classification result produced by the distributed classifier.

    Fields
    ------
    region_id : str
        Matches the ``DagRegion.region_id``.
    entries : List[DirtySetEntry]
        Per-step dirty-set entries for every step in this region, in
        topological (recorded) order.
    outputs_hash_by_id : Dict[str, str]
        Maps each step_id *in this region* to its effective output hash.
        Successor regions read this when rebinding ``"context"`` fields.
    outputs_by_id : Dict[str, Optional[Any]]
        Maps each step_id in this region to its actual output (or
        ``None`` for dirty input-drift steps whose output is unknown
        without an executor).
    dirty_by_id : Dict[str, bool]
        Maps each step_id in this region to its dirty flag.
    output_changed_by_id : Dict[str, bool]
        ``True`` iff the step was dirty AND its output hash changed from
        the recorded value (Step 61 stale-cache signal for successors).
    """

    region_id: str
    entries: List[DirtySetEntry]
    outputs_hash_by_id: Dict[str, str]
    outputs_by_id: Dict[str, Optional[Any]]
    dirty_by_id: Dict[str, bool]
    output_changed_by_id: Dict[str, bool]


# ------------------------------------------------------------------ helpers


def _make_ext_parents(region_steps: List[dict]) -> FrozenSet[str]:
    """Compute external parent step_ids for a region.

    A parent is external if it is referenced by a step in the region
    via ``parent_step_id`` or ``parent_step_ids`` but is not itself a
    step in the region.
    """
    step_ids: Set[str] = {s["step_id"] for s in region_steps}
    ext: Set[str] = set()
    for s in region_steps:
        pid = s.get("parent_step_id")
        if pid and pid not in step_ids:
            ext.add(pid)
        for pid2 in (s.get("parent_step_ids") or []):
            if pid2 not in step_ids:
                ext.add(pid2)
    return frozenset(ext)


def _assign_branch_roots(steps: List[dict]) -> Dict[str, Optional[str]]:
    """Assign each step to its branch root, if any.

    The *branch root* for a step is the step_id of the first-level
    child of a ``parallel_branch_open`` that is an ancestor of this
    step (or this step itself, if it is a first-level child).

    Returns ``None`` for backbone steps (steps not inside any branch
    opened by a ``parallel_branch_open``).

    The assignment propagates topologically: each step inherits its
    branch root from its ``parent_step_id``.  For nested branches (a
    ``parallel_branch_open`` inside a branch), inner branch steps are
    assigned to the *outer* branch root — they form one region for
    correctness, sacrificing inner-level parallelism.

    ``parallel_branch_join`` is always assigned ``None`` (backbone)
    because it closes the branch scope and its outputs feed the
    subsequent sequential region.
    """
    by_id: Dict[str, dict] = {s["step_id"]: s for s in steps}
    result: Dict[str, Optional[str]] = {}

    for s in steps:
        sid = s["step_id"]
        kind = s["step_kind"]
        parent_id = s.get("parent_step_id")

        if kind == "parallel_branch_join":
            # Always backbone — closes the branch scope.
            result[sid] = None
        elif (
            parent_id
            and by_id.get(parent_id, {}).get("step_kind") == "parallel_branch_open"
            and result.get(parent_id) is None
        ):
            # First-level child of an open step that is in the backbone:
            # this step IS a branch root.
            result[sid] = sid
        elif parent_id and result.get(parent_id) is not None:
            # Inside a branch: inherit parent's branch root.
            result[sid] = result[parent_id]
        else:
            # Backbone (includes parallel_branch_open itself, and steps
            # with no parent, or whose parent is not yet in a branch).
            result[sid] = None

    return result


# ------------------------------------------------------------------ partitioner


def partition_dag_regions(steps: List[dict]) -> List[DagRegion]:
    """Partition a list of recorded steps into independent DAG regions.

    The algorithm identifies ``parallel_branch_open`` / ``join`` section
    boundaries and assigns each branch a separate region.  Steps outside
    any branch scope form sequential regions.  Sequential regions are
    ordered before the branches they open and after the branches they
    follow; branch regions within the same fan-out share no data
    dependencies and are therefore *independent*.

    For traces without any parallel branches a single sequential region
    is returned (degenerate case).

    Parameters
    ----------
    steps : List[dict]
        Recorded step dicts from ``Trace.recorded_steps``, in
        topological order (assumption A4).

    Returns
    -------
    List[DagRegion]
        Regions in dependency order.  Regions sharing no
        ``external_parent_ids`` within the same "tier" can be processed
        in parallel by :py:func:`compute_dirty_set_distributed`.
    """
    if not steps:
        return []

    branch_root_by_id = _assign_branch_roots(steps)

    regions: List[DagRegion] = []
    current_seq: List[dict] = []
    pending_branches: Dict[str, List[dict]] = {}  # branch_root_id → steps
    seq_counter = 0

    def _flush_seq() -> None:
        nonlocal seq_counter
        if not current_seq:
            return
        regions.append(DagRegion(
            region_id=f"seq:{seq_counter}",
            steps=list(current_seq),
            external_parent_ids=_make_ext_parents(current_seq),
        ))
        seq_counter += 1
        current_seq.clear()

    def _flush_branches() -> None:
        for br_id in list(pending_branches.keys()):
            br_steps = pending_branches.pop(br_id)
            regions.append(DagRegion(
                region_id=f"branch:{br_id}",
                steps=br_steps,
                external_parent_ids=_make_ext_parents(br_steps),
            ))

    for s in steps:
        sid = s["step_id"]
        kind = s["step_kind"]
        br_root = branch_root_by_id.get(sid)

        if br_root is None:
            # Backbone step.
            if kind == "parallel_branch_open":
                # Include the open step in the current sequential region
                # (it belongs to the backbone), then flush to start branches.
                current_seq.append(s)
                _flush_seq()
            elif kind == "parallel_branch_join" and pending_branches:
                # All branches are done; emit them, then start a new
                # sequential region with the join step.
                _flush_branches()
                current_seq.append(s)
            else:
                current_seq.append(s)
        else:
            # Branch step: accumulate under its branch root.
            if br_root not in pending_branches:
                pending_branches[br_root] = []
            pending_branches[br_root].append(s)

    # Flush any remaining state (well-formed traces should not have
    # unmatched opens, but be defensive).
    _flush_branches()
    _flush_seq()

    return regions


# ------------------------------------------------------------------ classifier


def _classify_region(
    region_steps: List[dict],
    subs: Any,  # SubstitutionSet — read-only concurrent access is safe
    initial_outputs_hash: Dict[str, str],
    initial_dirty: Dict[str, bool],
    initial_outputs: Dict[str, Optional[Any]],
    initial_output_changed: Dict[str, bool],
) -> RegionSummary:
    """Classify dirty-set entries for a single region.

    Implements the same logic as :py:func:`~stepback.divergence.compute_dirty_set`
    but operates on a subset of steps and reads parent state from the
    supplied *initial_* snapshots instead of an inline global dict.

    Workers must treat all four *initial_* dicts as **read-only**.
    This function builds its own local state dicts and never mutates
    the snapshots, making concurrent calls on disjoint step sets safe.
    """
    sentinel_pfx = "__dirty_sentinel__"

    # Local state for steps within this region.
    outputs_hash_by_id: Dict[str, str] = {}
    outputs_by_id: Dict[str, Optional[Any]] = {}
    dirty_by_id: Dict[str, bool] = {}
    output_changed_by_id: Dict[str, bool] = {}
    entries: List[DirtySetEntry] = []

    def _out_hash(sid: str) -> Optional[str]:
        if sid in outputs_hash_by_id:
            return outputs_hash_by_id[sid]
        return initial_outputs_hash.get(sid)

    def _dirty(sid: str) -> bool:
        if sid in dirty_by_id:
            return dirty_by_id[sid]
        return initial_dirty.get(sid, False)

    def _outputs(sid: str) -> Optional[Any]:
        if sid in outputs_by_id:
            return outputs_by_id[sid]
        return initial_outputs.get(sid)

    def _out_changed(sid: str) -> bool:
        if sid in output_changed_by_id:
            return output_changed_by_id[sid]
        return initial_output_changed.get(sid, False)

    for rec in region_steps:
        sid: str = rec["step_id"]
        kind: str = rec["step_kind"]
        recorded_inputs_hash: str = rec["inputs_hash"]
        cur_inputs: dict = copy.deepcopy(rec["inputs"])

        parent_id: Optional[str] = rec.get("parent_step_id")
        context_fields: Optional[list] = cur_inputs.get("_stepback_context_fields")

        # Rebind "context" from the parent's current output hash.
        if parent_id and "context" in cur_inputs:
            if context_fields:
                parent_out = _outputs(parent_id)
                if parent_out is None:
                    cur_inputs["context"] = f"{sentinel_pfx}_field_{parent_id}"
                elif isinstance(parent_out, dict):
                    partial = {k: parent_out[k] for k in context_fields if k in parent_out}
                    if len(partial) < len(context_fields):
                        cur_inputs["context"] = f"{sentinel_pfx}_missing_{parent_id}"
                    else:
                        cur_inputs["context"] = hash_obj(partial)
                else:
                    ph = _out_hash(parent_id)
                    if ph:
                        cur_inputs["context"] = ph
            else:
                ph = _out_hash(parent_id)
                if ph:
                    cur_inputs["context"] = ph

        # Multi-parent rebinding for parallel_branch_join.
        parent_ids: List[str] = list(rec.get("parent_step_ids") or [])
        if "branch_tails" in cur_inputs and "branch_tail_hashes" in cur_inputs:
            tails = cur_inputs["branch_tails"]
            cur_inputs["branch_tail_hashes"] = [
                _out_hash(t) or cur_inputs["branch_tail_hashes"][i]
                for i, t in enumerate(tails)
            ]

        # Apply substitutions targeting this step.
        _sentinel = object()
        tool_override: Any = _sentinel
        for sub in subs.at(sid):
            if sub.is_output_forcing():
                tool_override = sub.force_output(rec)
            else:
                sub.apply(cur_inputs, rec)

        current_inputs_hash: str = hash_obj(cur_inputs)

        # Parent-dirty closure (P1).
        parent_dirty: bool = _dirty(parent_id) if parent_id else False
        if parent_ids:
            parent_dirty = parent_dirty or any(_dirty(pid) for pid in parent_ids)

        # Step 61: propagate dirtiness only when parent output hash changed.
        parent_output_changed: bool = _out_changed(parent_id) if parent_id else False
        if parent_ids:
            parent_output_changed = parent_output_changed or any(
                _out_changed(pid) for pid in parent_ids
            )
        use_parent_dirty = parent_output_changed and not context_fields

        # Nondeterminism-hash check (Step 33).
        recorded_nondet_hash: Optional[str] = rec.get("nondeterminism_hash")
        nondet_dirty: bool = False
        if recorded_nondet_hash is not None:
            live_nondet_hash = sha256_hex(canonical_json(rec.get("nondeterminism", {})))
            nondet_dirty = recorded_nondet_hash != live_nondet_hash

        # Nondeterminism-class check (Step 63).
        nondet_class_dirty: bool = _nondet_forces_dirty(rec.get("nondeterminism", {}))

        # Classify.
        if tool_override is not _sentinel:
            is_dirty = True
            dirty_reason: Optional[str] = "substituted"
            cur_output_hash = hash_obj(tool_override)
            cur_actual_output: Optional[Any] = tool_override
        elif nondet_class_dirty:
            is_dirty = True
            dirty_reason = "nondeterminism"
            cur_output_hash = f"{sentinel_pfx}{sid}"
            cur_actual_output = None
        elif current_inputs_hash != recorded_inputs_hash or nondet_dirty or subs.at(sid):
            is_dirty = True
            dirty_reason = "input_drift"
            cur_output_hash = f"{sentinel_pfx}{sid}"
            cur_actual_output = None
        elif use_parent_dirty:
            is_dirty = True
            dirty_reason = "parent_dirty"
            cur_output_hash = f"{sentinel_pfx}{sid}"
            cur_actual_output = None
        else:
            is_dirty = False
            dirty_reason = None
            cur_output_hash = rec.get("outputs_hash") or hash_obj(rec.get("outputs", {}))
            cur_actual_output = rec.get("outputs")

        outputs_hash_by_id[sid] = cur_output_hash
        outputs_by_id[sid] = cur_actual_output
        dirty_by_id[sid] = is_dirty
        recorded_out_hash = rec.get("outputs_hash") or hash_obj(rec.get("outputs", {}))
        output_changed_by_id[sid] = is_dirty and cur_output_hash != recorded_out_hash

        all_parent_ids: FrozenSet[str] = frozenset(
            ([parent_id] if parent_id else []) + parent_ids
        )
        entries.append(DirtySetEntry(
            step_id=sid,
            kind=kind,
            name=rec.get("name"),
            dirty=is_dirty,
            cache_hit=not is_dirty,
            dirty_reason=dirty_reason,
            parent_step_ids=list(all_parent_ids),
            parent_step_id=parent_id,
            recorded_inputs_hash=recorded_inputs_hash,
            current_inputs_hash=current_inputs_hash,
        ))

    return RegionSummary(
        region_id="",  # Set by the orchestrator after submission.
        entries=entries,
        outputs_hash_by_id=outputs_hash_by_id,
        outputs_by_id=outputs_by_id,
        dirty_by_id=dirty_by_id,
        output_changed_by_id=output_changed_by_id,
    )


# ------------------------------------------------------------------ public API


def compute_dirty_set_distributed(
    trace: "Trace_",  # type: ignore[name-defined]
    substitutions: Optional[Sequence["Substitution"]],  # type: ignore[name-defined]
    *,
    workers: int = 4,
    executor: Optional[Any] = None,  # accepted for API symmetry; not used
) -> DirtySetSummary:
    """Compute the dirty-set D for ``trace`` under ``substitutions``, using a
    thread-pool worker pool to classify independent DAG regions in parallel.

    The trace DAG is partitioned into regions by
    :py:func:`partition_dag_regions`.  Regions with no data dependencies
    between them (within the same "tier") are submitted to a
    :py:class:`~concurrent.futures.ThreadPoolExecutor` and classified
    concurrently.  Tier boundaries are processed sequentially: each tier
    waits for all previous tiers to complete before starting.

    The returned :py:class:`~stepback.divergence.DirtySetSummary` is
    **identical** to the one produced by
    :py:func:`~stepback.divergence.compute_dirty_set` for the same inputs.
    Both functions implement the same dirty-set algorithm; this one adds
    parallel scheduling across branch regions.

    Parameters
    ----------
    trace : Trace
        A loaded ``Trace`` object.  Must satisfy A4: ``recorded_steps``
        are in topological order.
    substitutions : Sequence[Substitution] | None
        Substitutions to apply.  ``None`` or ``[]`` both yield an
        all-clean result (P5).
    workers : int
        Maximum number of worker threads used per tier.  ``workers=1``
        falls back to sequential execution (no thread overhead).
        Default: 4.
    executor : Executor | None
        Accepted for API symmetry with
        :py:func:`~stepback.divergence.compute_dirty_set`; not used —
        dirty-set classification is pure and non-executing.

    Returns
    -------
    DirtySetSummary
        Same structure as :py:func:`~stepback.divergence.compute_dirty_set`
        output.  Entries are in the original topological (recorded)
        step order.
    """
    from .substitutions import SubstitutionSet

    subs: SubstitutionSet = SubstitutionSet()
    for s in (substitutions or []):
        subs.add(s)

    steps = list(trace.recorded_steps)
    if not steps:
        return DirtySetSummary(
            step_count=0, dirty_count=0, clean_count=0, calls_saved=0, entries=[]
        )

    regions = partition_dag_regions(steps)

    # Map step_id → region_id for dependency and final-assembly lookups.
    step_to_region: Dict[str, str] = {}
    for reg in regions:
        for s in reg.steps:
            step_to_region[s["step_id"]] = reg.region_id

    region_by_id: Dict[str, DagRegion] = {r.region_id: r for r in regions}

    # Build region dependency graph (region_id → set of predecessor region_ids).
    region_deps: Dict[str, Set[str]] = {}
    for reg in regions:
        deps: Set[str] = set()
        for sid in reg.external_parent_ids:
            dep_rid = step_to_region.get(sid)
            if dep_rid:
                deps.add(dep_rid)
        region_deps[reg.region_id] = deps

    # Compute tier (level) for each region via BFS from roots.
    in_degree: Dict[str, int] = {r.region_id: len(region_deps[r.region_id]) for r in regions}
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

    # Process tier by tier; parallelize within each tier.
    global_outputs_hash: Dict[str, str] = {}
    global_dirty: Dict[str, bool] = {}
    global_outputs: Dict[str, Optional[Any]] = {}
    global_output_changed: Dict[str, bool] = {}
    region_summaries: Dict[str, RegionSummary] = {}

    actual_workers = max(1, workers)

    for tier_rids in tier_groups:
        if len(tier_rids) == 1 or actual_workers == 1:
            # Sequential: no thread overhead.
            for rid in tier_rids:
                summary = _classify_region(
                    region_by_id[rid].steps,
                    subs,
                    global_outputs_hash,
                    global_dirty,
                    global_outputs,
                    global_output_changed,
                )
                summary.region_id = rid
                region_summaries[rid] = summary
                global_outputs_hash.update(summary.outputs_hash_by_id)
                global_dirty.update(summary.dirty_by_id)
                global_outputs.update(summary.outputs_by_id)
                global_output_changed.update(summary.output_changed_by_id)
        else:
            # Parallel: snapshot global state once so all workers in this
            # tier see the same ancestor outputs.  Workers never mutate
            # the snapshot; each builds its own local state dicts.
            snap_out_hash = dict(global_outputs_hash)
            snap_dirty = dict(global_dirty)
            snap_outputs = dict(global_outputs)
            snap_out_changed = dict(global_output_changed)

            n_workers = min(actual_workers, len(tier_rids))
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(
                        _classify_region,
                        region_by_id[rid].steps,
                        subs,
                        snap_out_hash,
                        snap_dirty,
                        snap_outputs,
                        snap_out_changed,
                    ): rid
                    for rid in tier_rids
                }
                for fut in as_completed(futures):
                    rid = futures[fut]
                    summary = fut.result()
                    summary.region_id = rid
                    region_summaries[rid] = summary

            # Merge all tier summaries into global state after all workers
            # complete.  Branch regions are disjoint, so merge order does
            # not matter.
            for rid in tier_rids:
                s = region_summaries[rid]
                global_outputs_hash.update(s.outputs_hash_by_id)
                global_dirty.update(s.dirty_by_id)
                global_outputs.update(s.outputs_by_id)
                global_output_changed.update(s.output_changed_by_id)

    # Assemble the final DirtySetSummary in original topological step order.
    all_entries: List[DirtySetEntry] = []
    # Build fast lookup: step_id → DirtySetEntry from its region.
    entry_by_step: Dict[str, DirtySetEntry] = {}
    for rs in region_summaries.values():
        for entry in rs.entries:
            entry_by_step[entry.step_id] = entry

    for step in steps:
        entry = entry_by_step.get(step["step_id"])
        if entry is not None:
            all_entries.append(entry)

    dirty_count = sum(1 for e in all_entries if e.dirty)
    clean_count = len(all_entries) - dirty_count
    return DirtySetSummary(
        step_count=len(all_entries),
        dirty_count=dirty_count,
        clean_count=clean_count,
        calls_saved=clean_count,
        entries=all_entries,
    )


# Forward reference for type checkers only.
try:
    from .replay import Trace as Trace_  # noqa: F401
except ImportError:  # pragma: no cover
    pass
