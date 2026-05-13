"""Dirty-set classifier and replay-divergence detector.

stepback's whole value proposition rests on **deterministic replay**:
"same canonical inputs → same hash → cache hit → no LLM re-execution".
But real LLM endpoints (OpenAI, Anthropic, Bedrock, vLLM, …) are not
fully deterministic. ``temperature=0`` reduces variance but doesn't
eliminate it; provider-side speculative decoding, model fleet
heterogeneity, and silent model upgrades all break the bytewise
guarantee that the per-step cache relies on.

This module provides two related but distinct surfaces:

1. **Dirty-set classifier** (:py:func:`compute_dirty_set`) — a
   *pure, non-executing* analysis that determines which steps would be
   dirty under a given set of substitutions, *without* invoking any
   LLM or tool executor. This is the formal implementation of the
   README's dirty-set pseudocode.

2. **Replay-divergence detector** (:py:func:`detect_divergences`) —
   re-executes every step in a recorded trace against a user-supplied
   executor and classifies each output against the recorded output
   through a **structural classifier** (not a heuristic).

---

**Dirty-set algorithm** (from README §"The dirty-set algorithm")::

    Given:
      trace T = [s0, s1, ..., sN]          # recorded steps, topological order
      substitution sigma at step sk         # one or more Substitution objects

    Compute dirty-set D:
      D := {}
      for i in 0..N:
        inputs_prime_i := apply sigma and propagate D to recompute si inputs
        if sigma targets si directly (output-forcing):
          D := D union {si}               # P3: directly-substituted step is dirty
        elif hash(inputs_prime_i) != hash(inputs_i):
          D := D union {si}               # P2: input hash drifted
        elif any parent of si is in D:
          D := D union {si}               # P1: parent-dirty closure
        else:
          reuse cached outputs(si)        # cache hit: outputs unchanged

    Replay cost: |D| ≤ N LLM/tool calls instead of N.
    Correctness target: every step NOT in D is observationally equivalent
    to full re-execution under sigma (soundness; proved in docs/dirty-set.md).

**Postconditions** (P1–P6, as labelled in contract tests):

* **P1** *Parent-dirty closure* — if step s_i is in D and s_j directly
  or transitively depends on s_i, then s_j ∈ D.
* **P2** *Input-drift inclusion* — if ``hash(inputs_prime_i) ≠ hash(inputs_i)``
  (the canonical inputs hash changed after applying sigma), s_i ∈ D.
* **P3** *Direct-substitution inclusion* — if any substitution in sigma
  targets s_i directly (output-forcing or not), s_i ∈ D.
* **P4** *Clean-step soundness* — if s_i ∉ D, then
  ``hash(inputs_prime_i) == hash(inputs_i)`` and every parent of s_i
  is not in D (given collision-free canonical hashing).
* **P5** *Minimality under empty sigma* — if sigma is empty, D = {}.
* **P6** *Monotonicity* — adding more substitutions to sigma can only
  grow D; it cannot remove a step from D.

**Assumptions**:

* A1. Canonical hashing is collision-free on the set of actual step
  inputs that arise in the recorded trace (standard cryptographic
  assumption for SHA-256).
* A2. The dependency structure of the trace is correctly described by
  ``parent_step_id`` / ``parent_step_ids`` and the ``"context"`` +
  ``"branch_tail_hashes"`` rebinding rules — no hidden data-flow between
  steps.
* A3. The recorded ``inputs_hash`` field in each step was computed by
  the same :py:func:`~stepback.canonical.hash_obj` function used here.
* A4. Steps are stored and processed in topological order (every parent
  appears before its children in ``recorded_steps``).

**Branch-aware propagation** (B1–B3):

* **B1** *Independent fan-out* — for parallel branches, a substitution
  inside branch B_i is dirty only for steps inside B_i; it never
  propagates into sibling branches B_k (k ≠ i).
* **B2** *Exact join dirtying* — a ``parallel_branch_join`` step is
  dirty iff at least one consumed branch tail was dirty (multi-parent
  OR over branch dirtiness via ``branch_tail_hashes`` rebinding).
* **B3** *Clean-sibling preservation* — the dirty count under a
  single-branch substitution is independent of fan-out width W.

---

**Divergence-detection severity levels** (for :py:func:`detect_divergences`):

* :py:data:`IDENTICAL` — bytewise equal after canonical JSON.
* :py:data:`EQUIVALENT` — equal after stripping volatile fields
  (``id``, ``created``, ``request_id``, ``response_id``,
  ``system_fingerprint``, ``timestamp`` …) and zeroing token-usage
  counts.  This is the "noise floor" — replay is still cache-safe in
  spirit; only provider bookkeeping differs.
* :py:data:`MINOR` — same top-level structure and same final
  assistant message text, but other fields differ.
* :py:data:`SEMANTIC` — same structure but the final assistant message
  text or the tool result body changed (the case users care about).
* :py:data:`STRUCTURAL` — different keys / types at the top level,
  or one side is non-dict (schema drift or model upgrade).

---

Public surface:

Dirty-set classifier:

* :py:func:`compute_dirty_set` — pure dirty-set analysis with no execution.
* :py:class:`DirtySetEntry` — per-step dirty-set record.
* :py:class:`DirtySetSummary` — aggregate dirty-set result.

Divergence detector:

* :py:data:`IDENTICAL`, :py:data:`EQUIVALENT`, :py:data:`MINOR`,
  :py:data:`SEMANTIC`, :py:data:`STRUCTURAL` — severity class constants.
* :py:data:`SEVERITY` — ordered list (lowest → highest).
* :py:data:`SEVERITY_WEIGHT` — int weights for scoring.
* :py:data:`VOLATILE_KEYS` — provider-noise key set.
* :py:func:`compare_outputs` — classify a single (recorded, replayed) pair.
* :py:func:`detect_divergences` — re-execute a whole trace and return a
  :py:class:`DivergenceReport`.
* :py:class:`Divergence`, :py:class:`DivergenceReport` —
  serialisable result types with ``to_json`` / Markdown rendering.

Usage — dirty-set classifier::

    from stepback import replay
    from stepback.divergence import compute_dirty_set
    from stepback.substitutions import PromptSubstitution

    trace = replay("incident.sb")
    summary = compute_dirty_set(
        trace,
        [PromptSubstitution(at_step="step:llm_call:1", new_messages=[...])],
    )
    print(f"dirty: {summary.dirty_count}/{summary.step_count}")
    print(f"calls saved: {summary.calls_saved}")

Usage — divergence detector::

    from stepback.divergence import detect_divergences
    from stepback.replay import Executor

    report = detect_divergences(
        "trace.sb",
        hmac_key=key.hmac_key,
        executor=Executor(llm=my_real_llm, tool=my_real_tool),
    )
    print(report.render_markdown())
    if report.severity_score > 5:
        sys.exit(2)

---

**§Complexity** (full proofs in ``docs/dirty-set-complexity.md``, Step 58):

Let ``N = |V(T)|``, ``E = |E(T)|``, ``I = Σ_s |J(s.inputs)|``,
``O = Σ_s |J(s.outputs)|``, ``m = |σ|``, ``D_real ≤ |D|`` = real
executor calls, and ``S_max = max_s (I_s + O_s)``.

**Time** — ``compute_dirty_set`` runs in
``Θ(N + E + I + O + m)`` for any trace shape (linear, general DAG,
or branch-heavy).  The loop visits every step exactly once, performs
``O(w(s))`` parent-edge work per step (``Σ_s w(s) = E``), and one
canonicalize-and-hash pass of cost ``Θ(I_s + O_s)`` per step.
There is no inner loop over ``N`` or ``E``.  Adding dirty re-execution:
``T_total = Θ(N + E + I + O + m + D_real · X)`` for per-call cost
``X``.

**Memory** — ``Θ(N + O + S_max + m)`` peak heap.  The two persistent
maps ``outputs_by_id`` and ``outputs_hash_by_id`` hold ``N`` keys and
``O`` value bytes; per-iteration scratch (``cur_inputs``,
``cur_outputs``) is bounded by ``S_max`` and released each iteration.

**Trace shapes** (summary table; proved in ``docs/dirty-set-complexity.md``):

+-------------------------------------------+--------------------------+----------------+--------------------+
| Trace shape                               | Classifier time          | Executor calls | Peak memory        |
+===========================================+==========================+================+====================+
| Linear (``E = N − 1``)                   | ``Θ(N + I + O + m)``    | ``O(D_real)``  | ``Θ(N + O + m)``   |
+-------------------------------------------+--------------------------+----------------+--------------------+
| General DAG                               | ``Θ(N + E + I + O + m)``| ``O(D_real)``  | ``Θ(N+O+S_max+m)`` |
+-------------------------------------------+--------------------------+----------------+--------------------+
| Branch-heavy (max join width ``W``)       | ``Θ(N + E + I + O + m)``| ``O(D_real)``  | ``Θ(N+O+S_max+m)`` |
+-------------------------------------------+--------------------------+----------------+--------------------+

These bounds are tight (``Ω(N + E + I + O)`` lower bound from the
need to read every recorded step, edge, input and output at least
once).  On branch-rich DAGs a single-branch substitution typically
produces ``|D|/N ≈ 1/B`` where ``B`` is the fan-out width, giving
the ``O(|D|)`` LLM-calls headline.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from .canonical import canonical_json, hash_obj, sha256_hex
from .nondeterminism import forces_dirty as _nondet_forces_dirty
from .replay import Executor, MissingExecutor
from .trace_reader import verify_trace

# ====================================================================
# § Dirty-set classifier
# ====================================================================


@dataclass
class DirtySetEntry:
    """Per-step record produced by :py:func:`compute_dirty_set`.

    Fields
    ------
    step_id : str
        The canonical step ULID for this step.
    kind : str
        The ``step_kind`` of this step (``llm_call``, ``tool_call``, …).
    dirty : bool
        ``True`` iff this step is in the dirty set D under the supplied
        substitutions (must be re-executed on replay).
    cache_hit : bool
        Complement of ``dirty`` — ``True`` iff this step's recorded
        output can be reused as-is.
    dirty_reason : Optional[str]
        Human-readable reason code; ``None`` for clean steps.

        * ``"substituted"`` — an output-forcing substitution targets
          this step directly (P3, highest priority).
        * ``"input_drift"`` — the canonical inputs hash drifted after
          applying sigma (P2).
        * ``"parent_dirty"`` — a parent step is in D and this step's
          inputs were rebound from the parent's (possibly changed)
          output hash (P1).
        * ``"nondeterminism"`` — the step's :class:`~stepback.nondeterminism.NondeterminismClass`
          forces re-execution regardless of input-hash equality (Step 63).
          Examples: ``clock`` source with ``controlled=False``, ``rng``
          source with ``seed=None``, ``model_sampling`` with
          ``temperature > 0`` and no seed.
    parent_step_id : Optional[str]
        The ``parent_step_id`` from the recorded step (single-parent
        edge, or ``None`` for root steps).
    parent_step_ids : FrozenSet[str]
        All parent step ids for this step — includes ``parent_step_id``
        when set, plus every id in ``parent_step_ids`` (parallel joins).
        Always a frozenset; empty for root steps.
    """

    step_id: str
    kind: str
    dirty: bool
    cache_hit: bool
    dirty_reason: Optional[str]
    parent_step_id: Optional[str]
    parent_step_ids: FrozenSet[str]


@dataclass
class DirtySetSummary:
    """Aggregate dirty-set result produced by :py:func:`compute_dirty_set`.

    Fields
    ------
    step_count : int
        Total number of steps in the trace (== len(entries)).
    dirty_count : int
        Number of steps in the dirty set D (must be re-executed).
    clean_count : int
        Number of steps that are cache hits (step_count - dirty_count).
    calls_saved : int
        Equivalent to ``clean_count`` — the number of LLM/tool calls
        that would be avoided by replay under the supplied substitutions.
    entries : List[DirtySetEntry]
        Per-step records in topological (recorded) order.
    dirty_ids : List[str]
        ``step_id`` values of dirty steps, in topological order.
    """

    step_count: int
    dirty_count: int
    clean_count: int
    calls_saved: int
    entries: List[DirtySetEntry]

    @property
    def dirty_ids(self) -> List[str]:
        return [e.step_id for e in self.entries if e.dirty]

    def to_json(self) -> dict:
        return {
            "step_count": self.step_count,
            "dirty_count": self.dirty_count,
            "clean_count": self.clean_count,
            "calls_saved": self.calls_saved,
            "entries": [
                {
                    "step_id": e.step_id,
                    "kind": e.kind,
                    "dirty": e.dirty,
                    "cache_hit": e.cache_hit,
                    "dirty_reason": e.dirty_reason,
                    "parent_step_id": e.parent_step_id,
                    "parent_step_ids": sorted(e.parent_step_ids),
                }
                for e in self.entries
            ],
        }


def compute_dirty_set(
    trace: "Trace_",  # type: ignore[name-defined]
    substitutions: Optional[Sequence["Substitution"]],  # type: ignore[name-defined]
    *,
    executor: Optional[Executor] = None,  # accepted for API symmetry; not used
) -> DirtySetSummary:
    """Compute the dirty-set D for ``trace`` under ``substitutions``.

    This is a **pure, non-executing** analysis: no LLM or tool callback
    is ever invoked.  The function applies the substitutions to each
    step's inputs (in topological order) and determines whether the
    canonical inputs hash drifts from the recorded value, exactly as
    the replay engine would — but without actually running the
    re-executed steps.

    For output-forcing substitutions (:py:class:`~stepback.substitutions.ToolOutputSubstitution`,
    :py:class:`~stepback.substitutions.RaiseSubstitution`) the function
    records the *hash of the forced output* so that downstream steps can
    use the correct rebound hash when checking their own inputs.  For
    input-only substitutions where the step becomes dirty, a sentinel
    hash is used (the downstream inputs will not match the recorded hash,
    so their dirty classification is still correct).

    Parameters
    ----------
    trace : Trace
        A loaded ``Trace`` object (from :py:func:`stepback.replay`).
        Must satisfy **A4**: ``recorded_steps`` are in topological order.
    substitutions : Sequence[Substitution] | None
        The substitutions to apply.  ``None`` or ``[]`` both produce an
        all-clean result (postcondition P5).
        Substitutions targeting unknown step IDs are silently ignored
        (precondition note: they cannot dirty anything they can't reach).
    executor : Executor | None
        Accepted for API symmetry with the replay engine; not used.
        Pass ``Executor(fallback_recorded=True)`` to signal intent in
        tests without triggering any real execution.

    Returns
    -------
    DirtySetSummary
        Aggregate result with per-step :py:class:`DirtySetEntry` records.

    Preconditions
    -------------
    1. ``trace.recorded_steps`` is non-empty and in topological order
       (A4 — every ``parent_step_id`` refers to a step that appeared
       earlier in the list).
    2. Every step has an ``inputs_hash`` field that was computed by
       :py:func:`~stepback.canonical.hash_obj` on the step's ``inputs``
       at record time (A3).
    3. Substitutions targeting unknown step IDs are tolerated and treated
       as no-ops; callers should not rely on them dirtying anything.

    Postconditions
    --------------
    * P1 — Parent-dirty closure: every transitive descendant of a dirty
      step is also in the returned dirty set.
    * P2 — Input-drift inclusion: every step whose canonical inputs hash
      changes under sigma is in the dirty set.
    * P3 — Direct-substitution inclusion: every step directly targeted
      by a substitution in sigma is in the dirty set.
    * P4 — Clean-step soundness: every step NOT in the dirty set has the
      same inputs hash as the recorded value and no dirty ancestor.
    * P5 — Minimality under empty sigma: if substitutions is empty (or
      None), dirty_count == 0.
    * P6 — Monotonicity: adding more substitutions can only grow the set.
    """
    from .substitutions import SubstitutionSet

    subs: SubstitutionSet = SubstitutionSet()
    for s in (substitutions or []):
        subs.add(s)

    outputs_hash_by_id: Dict[str, str] = {}
    # Actual step outputs, used for field-level partial recompute (Step 60).
    # Value is None when the output is unknown (dirty input-only step with
    # no executor), which triggers conservative sentinel propagation.
    outputs_by_id: Dict[str, Optional[Any]] = {}
    dirty_by_id: Dict[str, bool] = {}
    # Step 61 – stale-cache detection: True iff a step was dirty AND its
    # output hash actually changed from the recorded value.  Only real
    # output-change propagates dirtiness to downstream steps.
    output_changed_by_id: Dict[str, bool] = {}
    sentinel_pfx = "__dirty_sentinel__"
    entries: List[DirtySetEntry] = []

    for rec in trace.recorded_steps:
        sid: str = rec["step_id"]
        kind: str = rec["step_kind"]
        recorded_inputs_hash: str = rec["inputs_hash"]
        cur_inputs: dict = copy.deepcopy(rec["inputs"])

        # Rebind "context" from the parent's current output.
        # For steps with _stepback_context_fields (Step 60 partial recompute),
        # hash only the declared fields of the parent output so that unrelated
        # field changes don't propagate dirtiness here.
        parent_id: Optional[str] = rec.get("parent_step_id")
        context_fields: Optional[list] = cur_inputs.get("_stepback_context_fields")
        if parent_id and "context" in cur_inputs:
            if context_fields:
                parent_out = outputs_by_id.get(parent_id)
                if parent_out is None:
                    # Parent output unknown (dirty without executor): conservative.
                    cur_inputs["context"] = f"{sentinel_pfx}_field_{parent_id}"
                elif isinstance(parent_out, dict):
                    partial = {k: parent_out[k] for k in context_fields if k in parent_out}
                    if len(partial) < len(context_fields):
                        # Missing declared fields: conservative sentinel.
                        cur_inputs["context"] = f"{sentinel_pfx}_missing_{parent_id}"
                    else:
                        cur_inputs["context"] = hash_obj(partial)
                else:
                    # Non-dict parent output: fall back to full hash.
                    if parent_id in outputs_hash_by_id:
                        cur_inputs["context"] = outputs_hash_by_id[parent_id]
            elif parent_id in outputs_hash_by_id:
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
        _sentinel = object()
        tool_override: Any = _sentinel
        for sub in subs.at(sid):
            if sub.is_output_forcing():
                tool_override = sub.force_output(rec)
            else:
                sub.apply(cur_inputs, rec)

        current_inputs_hash: str = hash_obj(cur_inputs)

        # Parent dirtiness (P1 closure).
        # For field-dep steps (context_fields set), the closure is relaxed:
        # inputs_hash drift is the sole dirtiness signal, because we already
        # rebound "context" from only the declared fields.
        parent_dirty: bool = dirty_by_id.get(parent_id, False) if parent_id else False
        if parent_ids:
            parent_dirty = parent_dirty or any(
                dirty_by_id.get(pid, False) for pid in parent_ids
            )
        # Step 61 – stale-cache detection: propagate dirtiness based on
        # whether the parent's output hash *actually changed*, not merely
        # whether the parent was dirty.  For output-forcing substitutions
        # we know the exact output hash; for input-drift steps the sentinel
        # hash already diverges, so downstream steps pick that up via
        # current_inputs_hash drift without needing use_parent_dirty.
        parent_output_changed: bool = (
            output_changed_by_id.get(parent_id, False) if parent_id else False
        )
        if parent_ids:
            parent_output_changed = parent_output_changed or any(
                output_changed_by_id.get(pid, False) for pid in parent_ids
            )
        use_parent_dirty = parent_output_changed and not context_fields

        # Nondeterminism-hash check (consistent with replay engine Step 33).
        recorded_nondet_hash: Optional[str] = rec.get("nondeterminism_hash")
        nondet_dirty: bool = False
        if recorded_nondet_hash is not None:
            live_nondet_hash = sha256_hex(
                canonical_json(rec.get("nondeterminism", {}))
            )
            nondet_dirty = recorded_nondet_hash != live_nondet_hash

        # Nondeterminism-class check (Step 63): even when inputs hash and
        # nondeterminism_hash match, a step whose recorded class inherently
        # forces re-execution must be treated as dirty.
        nondet_class_dirty: bool = _nondet_forces_dirty(rec.get("nondeterminism", {}))

        # Classify (P2, P3).
        if tool_override is not _sentinel:
            # P3 — direct output-forcing substitution: always dirty.
            is_dirty = True
            dirty_reason: Optional[str] = "substituted"
            # Use hash of the forced output for accurate downstream rebinding.
            cur_output_hash = hash_obj(tool_override)
            cur_actual_output: Optional[Any] = tool_override
        elif current_inputs_hash != recorded_inputs_hash or nondet_dirty:
            # P2 — inputs drifted (input-only sub or upstream hash changed).
            is_dirty = True
            dirty_reason = "input_drift"
            # We have no executor to compute the real new output; use a
            # sentinel that won't match any recorded hash, so all
            # descendants that consume this step's output will also see
            # a drift.
            cur_output_hash = f"{sentinel_pfx}{sid}"
            cur_actual_output = None  # unknown — conservative for field-dep children
        elif use_parent_dirty:
            # P1 — parent is dirty; this step's rebound inputs may differ.
            is_dirty = True
            dirty_reason = "parent_dirty"
            cur_output_hash = f"{sentinel_pfx}{sid}"
            cur_actual_output = None
        elif nondet_class_dirty:
            # Nondeterminism class forces re-execution (Step 63).
            is_dirty = True
            dirty_reason = "nondeterminism"
            cur_output_hash = f"{sentinel_pfx}{sid}"
            cur_actual_output = None
        else:
            # Clean cache hit.
            is_dirty = False
            dirty_reason = None
            # Use the recorded outputs_hash if available, else recompute.
            cur_output_hash = rec.get("outputs_hash") or hash_obj(rec.get("outputs", {}))
            cur_actual_output = rec.get("outputs")

        outputs_hash_by_id[sid] = cur_output_hash
        outputs_by_id[sid] = cur_actual_output
        dirty_by_id[sid] = is_dirty
        # Step 61: output_changed is True iff dirty AND current hash ≠ recorded hash.
        # For sentinel hashes (input-drift steps without known output), we treat the
        # output as changed so that downstream steps remain conservatively dirty.
        recorded_out_hash = rec.get("outputs_hash") or hash_obj(rec.get("outputs", {}))
        output_changed_by_id[sid] = is_dirty and cur_output_hash != recorded_out_hash

        all_parent_ids: FrozenSet[str] = frozenset(
            ([parent_id] if parent_id else []) + parent_ids
        )
        entries.append(DirtySetEntry(
            step_id=sid,
            kind=kind,
            dirty=is_dirty,
            cache_hit=not is_dirty,
            dirty_reason=dirty_reason,
            parent_step_id=parent_id,
            parent_step_ids=all_parent_ids,
        ))

    dirty_count = sum(1 for e in entries if e.dirty)
    clean_count = len(entries) - dirty_count
    return DirtySetSummary(
        step_count=len(entries),
        dirty_count=dirty_count,
        clean_count=clean_count,
        calls_saved=clean_count,
        entries=entries,
    )


# Avoid circular-import loop: Trace lives in replay.py which imports us
# for MissingExecutor; use a forward-reference string annotation only.
try:
    from .replay import Trace as Trace_  # noqa: F401
except ImportError:  # pragma: no cover
    pass


# ----------------------------------------------------------------- consts

IDENTICAL = "identical"
EQUIVALENT = "equivalent"
MINOR = "minor"
SEMANTIC = "semantic"
STRUCTURAL = "structural"

SEVERITY: Tuple[str, ...] = (IDENTICAL, EQUIVALENT, MINOR, SEMANTIC, STRUCTURAL)
SEVERITY_WEIGHT: Dict[str, int] = {
    IDENTICAL: 0,
    EQUIVALENT: 1,
    MINOR: 2,
    SEMANTIC: 4,
    STRUCTURAL: 5,
}

# Provider-side bookkeeping that is allowed to vary between runs without
# being treated as a real divergence. Keys are matched case-insensitively
# at any depth in the JSON tree.
VOLATILE_KEYS: Tuple[str, ...] = (
    "id",
    "created",
    "created_at",
    "request_id",
    "response_id",
    "x_request_id",
    "system_fingerprint",
    "fingerprint",
    "timestamp",
    "trace_id",
    "span_id",
)

# Token counts get zeroed (not removed) before structural compare so a
# different completion length doesn't cascade through every nested
# `usage` block.
USAGE_KEYS: Tuple[str, ...] = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_tokens",
    "input_tokens",
    "output_tokens",
)


# --------------------------------------------------------------- helpers


def _strip_volatile(obj: Any) -> Any:
    """Recursively drop volatile bookkeeping keys; zero token counters.

    Returns a NEW structure; the input is not mutated.
    """
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in VOLATILE_KEYS:
                continue
            if kl in USAGE_KEYS and isinstance(v, (int, float)):
                out[k] = 0
                continue
            out[k] = _strip_volatile(v)
        return out
    if isinstance(obj, list):
        return [_strip_volatile(x) for x in obj]
    return obj


def _assistant_text(outputs: Any) -> Optional[str]:
    """Extract the final assistant message text from an OpenAI/Anthropic-shaped output."""
    if not isinstance(outputs, dict):
        return None
    # OpenAI chat/completion shape.
    choices = outputs.get("choices")
    if isinstance(choices, list) and choices:
        last = choices[-1]
        if isinstance(last, dict):
            msg = last.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(last.get("text"), str):
                return last["text"]
    # Anthropic messages shape.
    content = outputs.get("content")
    if isinstance(content, list) and content:
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        if parts:
            return "".join(parts)
    if isinstance(content, str):
        return content
    # Tool-call envelope shape.
    if "result" in outputs:
        r = outputs["result"]
        if isinstance(r, str):
            return r
    return None


def _structural_skeleton(obj: Any, _depth: int = 0) -> Any:
    """A type/shape-only skeleton: dict→sorted key list, list→length, leaf→type name."""
    if _depth > 6:
        return "..."
    if isinstance(obj, dict):
        return {k: _structural_skeleton(obj[k], _depth + 1) for k in sorted(obj)}
    if isinstance(obj, list):
        if not obj:
            return []
        # Use the first element's skeleton as a representative.
        return [_structural_skeleton(obj[0], _depth + 1)]
    return type(obj).__name__


# -------------------------------------------------------------- dataclass


@dataclass
class Divergence:
    """One step's divergence classification."""

    step_id: str
    kind: str
    name: Optional[str]
    classification: str
    recorded_hash: str
    replayed_hash: str
    summary: str
    error: Optional[str] = None  # set if executor raised on this step

    @property
    def severity(self) -> int:
        return SEVERITY_WEIGHT.get(self.classification, 0)

    @property
    def is_divergent(self) -> bool:
        return self.classification != IDENTICAL

    def to_json(self) -> dict:
        d = {
            "step_id": self.step_id,
            "kind": self.kind,
            "name": self.name,
            "classification": self.classification,
            "severity": self.severity,
            "recorded_hash": self.recorded_hash,
            "replayed_hash": self.replayed_hash,
            "summary": self.summary,
        }
        if self.error is not None:
            d["error"] = self.error
        return d


@dataclass
class DivergenceReport:
    """Aggregate divergence over a whole trace re-execution."""

    trace_path: str
    step_count: int
    divergences: List[Divergence] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)  # step_ids without executor support

    @property
    def class_counts(self) -> Dict[str, int]:
        out = {c: 0 for c in SEVERITY}
        for d in self.divergences:
            out[d.classification] = out.get(d.classification, 0) + 1
        return out

    @property
    def divergent_count(self) -> int:
        return sum(1 for d in self.divergences if d.is_divergent)

    @property
    def severity_score(self) -> int:
        """Sum of per-step severity weights — 0 means perfect replay."""
        return sum(d.severity for d in self.divergences)

    @property
    def reproducibility_pct(self) -> float:
        """Percent of compared steps classified as IDENTICAL or EQUIVALENT."""
        if not self.divergences:
            return 100.0
        good = sum(1 for d in self.divergences if d.classification in (IDENTICAL, EQUIVALENT))
        return round(100.0 * good / len(self.divergences), 2)

    def to_json(self) -> dict:
        return {
            "trace_path": self.trace_path,
            "step_count": self.step_count,
            "compared_count": len(self.divergences),
            "divergent_count": self.divergent_count,
            "severity_score": self.severity_score,
            "reproducibility_pct": self.reproducibility_pct,
            "class_counts": self.class_counts,
            "skipped_step_ids": list(self.skipped),
            "divergences": [d.to_json() for d in self.divergences],
        }

    def render_markdown(self, max_rows: int = 50) -> str:
        lines: List[str] = []
        lines.append("# stepback divergence report")
        lines.append("")
        lines.append(f"trace: `{self.trace_path}`")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("| - | - |")
        lines.append(f"| step_count | {self.step_count} |")
        lines.append(f"| compared | {len(self.divergences)} |")
        lines.append(f"| divergent | {self.divergent_count} |")
        lines.append(f"| severity_score | {self.severity_score} |")
        lines.append(f"| reproducibility_pct | {self.reproducibility_pct} |")
        for c in SEVERITY:
            lines.append(f"| {c} | {self.class_counts.get(c, 0)} |")
        lines.append("")
        if self.divergences:
            lines.append("## per-step")
            lines.append("")
            lines.append("| step | kind | name | class | severity | summary |")
            lines.append("| - | - | - | - | - | - |")
            for d in self.divergences[:max_rows]:
                summary = d.summary.replace("|", "\\|")
                lines.append(
                    f"| {d.step_id} | {d.kind} | {d.name or ''} | {d.classification} "
                    f"| {d.severity} | {summary} |"
                )
            if len(self.divergences) > max_rows:
                lines.append("")
                lines.append(f"_... {len(self.divergences) - max_rows} more rows truncated_")
        if self.skipped:
            lines.append("")
            lines.append(f"_Skipped {len(self.skipped)} step(s) (no executor): "
                         f"{', '.join(self.skipped[:10])}"
                         f"{'…' if len(self.skipped) > 10 else ''}_")
        return "\n".join(lines) + "\n"


# ------------------------------------------------------------ classifier


def compare_outputs(recorded: Any, replayed: Any) -> Tuple[str, str]:
    """Classify ``(recorded, replayed)`` into one of the SEVERITY classes.

    Returns ``(classification, summary)`` where ``summary`` is a
    short human-readable explanation suitable for a report row.
    """
    if hash_obj(recorded) == hash_obj(replayed):
        return IDENTICAL, "bytewise-equal"

    sr = _strip_volatile(recorded)
    sp = _strip_volatile(replayed)
    if hash_obj(sr) == hash_obj(sp):
        return EQUIVALENT, "equal after stripping volatile fields"

    # From here on we know there's a real content difference.
    skel_r = _structural_skeleton(sr)
    skel_p = _structural_skeleton(sp)
    if hash_obj(skel_r) != hash_obj(skel_p):
        return STRUCTURAL, _structural_summary(sr, sp)

    text_r = _assistant_text(recorded)
    text_p = _assistant_text(replayed)
    if text_r is not None and text_p is not None and text_r == text_p:
        return MINOR, "same assistant text; metadata differs"
    if text_r is None and text_p is None:
        # No extractable assistant text — fall back to "semantic" since
        # structures match but content differs.
        return SEMANTIC, "no extractable assistant text; bodies differ"
    return SEMANTIC, _semantic_summary(text_r, text_p)


def _structural_summary(rec: Any, rep: Any) -> str:
    if not isinstance(rec, dict) or not isinstance(rep, dict):
        return f"top-level types differ: {type(rec).__name__} vs {type(rep).__name__}"
    rk = set(rec.keys())
    pk = set(rep.keys())
    only_r = sorted(rk - pk)
    only_p = sorted(pk - rk)
    parts = []
    if only_r:
        parts.append(f"recorded-only keys: {only_r[:5]}")
    if only_p:
        parts.append(f"replayed-only keys: {only_p[:5]}")
    if not parts:
        parts.append("nested type/shape mismatch")
    return "; ".join(parts)


def _semantic_summary(text_r: Optional[str], text_p: Optional[str]) -> str:
    def _trim(s: Optional[str]) -> str:
        if s is None:
            return "<none>"
        s = s.strip()
        return s if len(s) <= 60 else s[:57] + "..."
    return f"recorded={_trim(text_r)!r} → replayed={_trim(text_p)!r}"


# --------------------------------------------------------- whole-trace API


def detect_divergences(
    trace_path: str,
    hmac_key: bytes,
    executor: Executor,
    *,
    step_kinds: Iterable[str] = ("llm_call", "tool_call", "router"),
    max_steps: Optional[int] = None,
) -> DivergenceReport:
    """Re-execute every supported step in ``trace_path`` and classify divergences.

    Steps whose kind is not in ``step_kinds``, or for which the
    executor has no callback, are recorded in ``report.skipped``.

    The executor is called with the **recorded inputs**, so this is a
    pure replay-determinism probe — it does not propagate
    counterfactuals (use the substitution / replay-forward API for
    that).  This is exactly the question "if I re-ran the same
    prompts today, would I get the same outputs?".
    """
    trace = verify_trace(trace_path, hmac_key)
    steps = trace.steps
    if max_steps is not None:
        steps = steps[:max_steps]

    divergences: List[Divergence] = []
    skipped: List[str] = []
    for rec in steps:
        sid = rec["step_id"]
        kind = rec.get("step_kind") or rec.get("kind")
        name = rec.get("name")
        if kind not in step_kinds:
            skipped.append(sid)
            continue
        # Skip kinds the executor can't handle without aborting the sweep.
        if kind == "llm_call" and executor.llm is None:
            skipped.append(sid)
            continue
        if kind == "tool_call" and executor.tool is None:
            skipped.append(sid)
            continue
        if kind == "router" and executor.router is None:
            skipped.append(sid)
            continue
        recorded_outputs = rec.get("outputs")
        try:
            replayed = executor.execute(kind, rec.get("inputs", {}))
        except Exception as exc:  # executor failure is a real divergence signal
            divergences.append(Divergence(
                step_id=sid,
                kind=kind,
                name=name,
                classification=STRUCTURAL,
                recorded_hash=hash_obj(recorded_outputs),
                replayed_hash="<error>",
                summary=f"executor raised {type(exc).__name__}: {exc}",
                error=f"{type(exc).__name__}: {exc}",
            ))
            continue
        cls, summary = compare_outputs(recorded_outputs, replayed)
        divergences.append(Divergence(
            step_id=sid,
            kind=kind,
            name=name,
            classification=cls,
            recorded_hash=hash_obj(recorded_outputs),
            replayed_hash=hash_obj(replayed),
            summary=summary,
        ))
    return DivergenceReport(
        trace_path=trace_path,
        step_count=len(trace.steps),
        divergences=divergences,
        skipped=skipped,
    )


__all__ = [
    # Dirty-set classifier
    "DirtySetEntry",
    "DirtySetSummary",
    "compute_dirty_set",
    # Divergence detector
    "IDENTICAL",
    "EQUIVALENT",
    "MINOR",
    "SEMANTIC",
    "STRUCTURAL",
    "SEVERITY",
    "SEVERITY_WEIGHT",
    "VOLATILE_KEYS",
    "USAGE_KEYS",
    "Divergence",
    "DivergenceReport",
    "compare_outputs",
    "detect_divergences",
]
