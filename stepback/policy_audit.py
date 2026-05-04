"""Policy-impact audit: regulator-replay reporting over many traces.

This module implements the "what would happen if our *current* policy
had been in force when these traces were recorded?" question described
in README §Use-cases #2 ("counterfactual policy") and #5 (regulator
replay), and the CLI hook the README §CLI block calls
``stepback verify --policy-changed-since DATE`` (audited as ungrounded
in `_audit/findings.jsonl`).

The audit re-replays each trace twice through the existing
:class:`stepback.replay.Trace` engine:

* a **baseline** replay that applies *no* counterfactual substitutions
  — it just walks the recorded steps from cache, so every per-step
  output is the recorded one and total cost is the recorded cost;
* a **replayed** pass that applies a (caller-supplied) substitution
  set — typically a :class:`stepback.substitutions.PolicySubstitution`
  pointing at the new policy file plus, optionally, a set of
  :class:`stepback.substitutions.RaiseSubstitution` /
  :class:`stepback.substitutions.OutputsPatchSubstitution` entries
  that simulate per-step policy denials.

For each trace we then walk the two step lists in lockstep and
classify each step into one of:

* ``unchanged`` — outputs equal between baseline and replayed;
* ``divergent`` — outputs differ (the substitution propagated here);
* ``newly_blocked`` — divergent **and** the replayed output looks like
  a policy denial (``error_class`` or ``__error__.type`` contains
  ``"Policy"``) **and** the baseline output did not;
* ``newly_allowed`` — the inverse: baseline blocked, replayed allowed.

The aggregate :class:`PolicyImpactReport` carries per-trace
:class:`TraceImpact` rows plus roll-up counters used for the regulator
attestation summary line ("12,418 traces re-executed; 41 produced
different decisions" — README §Use-cases #5).

Public API
----------

* :func:`audit_policy_change` — main entry point; takes a list of
  trace paths + an HMAC key and returns a :class:`PolicyImpactReport`.
* :func:`is_policy_blocked` — public predicate exposed so callers can
  reuse the same "is this step a policy denial?" check that the
  report uses internally.
* :class:`PolicyImpactReport` / :class:`TraceImpact` — dataclasses
  with ``to_json`` and (on the report) ``to_markdown`` renderers.

The renderers are intentionally **pure functions of the report data**
(no clock reads, no random tie-breaking, no environment lookups) so
two engineers running the same audit get byte-identical Markdown —
the same byte-stability guarantee that ``stepback report`` makes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from .replay import Executor, ReplayResult, StepView, Trace, replay as load_trace
from .substitutions import PolicySubstitution, SubstitutionSet, Substitution


# ---------------------------------------------------------- predicates


_POLICY_TOKEN = "policy"


def is_policy_blocked(outputs: Any) -> bool:
    """True if ``outputs`` looks like a policy-denial step output.

    Recognises three shapes the rest of the codebase already produces:

    * The recorder's ``exception`` step kind: ``{"error_class":
      "PolicyDenied", "message": ...}``;
    * The substitution-engine's :class:`RaiseSubstitution` output
      sentinel: ``{"__error__": {"type": "PolicyDenied", "message":
      ...}}``;
    * A free-form dict with a ``"blocked": True`` flag and either a
      ``"reason"`` or a ``"policy"`` field — used by example
      pipelines that wrap toolwarden's reject path.

    The match is case-insensitive substring on the literal token
    ``"policy"`` so ``"PolicyDenied"``, ``"PolicyViolation"``,
    ``"policy.deny"``, etc. all count. This avoids hard-coding a
    single toolwarden version's exact class name.
    """
    if not isinstance(outputs, dict):
        return False
    ec = outputs.get("error_class")
    if isinstance(ec, str) and _POLICY_TOKEN in ec.lower():
        return True
    err = outputs.get("__error__")
    if isinstance(err, dict):
        t = err.get("type")
        if isinstance(t, str) and _POLICY_TOKEN in t.lower():
            return True
    if outputs.get("blocked") is True:
        # Any explicit blocked:true with a reason field counts.
        for k in ("reason", "policy", "policy_id", "policy_path"):
            if k in outputs:
                return True
    return False


# --------------------------------------------------------- dataclasses


@dataclass
class StepImpact:
    """One step's classification under the policy substitution."""

    step_id: str
    kind: str
    name: Optional[str]
    classification: str  # "unchanged" | "divergent" | "newly_blocked" | "newly_allowed"
    baseline_blocked: bool
    replayed_blocked: bool
    cost_delta_usd: float
    diverged_from_cache: bool

    def to_json(self) -> dict:
        return {
            "step_id": self.step_id,
            "kind": self.kind,
            "name": self.name,
            "classification": self.classification,
            "baseline_blocked": self.baseline_blocked,
            "replayed_blocked": self.replayed_blocked,
            "cost_delta_usd": round(self.cost_delta_usd, 8),
            "diverged_from_cache": self.diverged_from_cache,
        }


@dataclass
class TraceImpact:
    """Per-trace audit result."""

    trace_path: str
    step_count: int
    baseline_cost_usd: float
    replayed_cost_usd: float
    cost_delta_usd: float
    divergent_step_count: int
    newly_blocked_step_ids: List[str] = field(default_factory=list)
    newly_allowed_step_ids: List[str] = field(default_factory=list)
    step_impacts: List[StepImpact] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def has_divergence(self) -> bool:
        return self.divergent_step_count > 0

    def to_json(self) -> dict:
        return {
            "trace_path": self.trace_path,
            "step_count": self.step_count,
            "baseline_cost_usd": round(self.baseline_cost_usd, 8),
            "replayed_cost_usd": round(self.replayed_cost_usd, 8),
            "cost_delta_usd": round(self.cost_delta_usd, 8),
            "divergent_step_count": self.divergent_step_count,
            "newly_blocked_step_ids": list(self.newly_blocked_step_ids),
            "newly_allowed_step_ids": list(self.newly_allowed_step_ids),
            "step_impacts": [s.to_json() for s in self.step_impacts],
            "error": self.error,
        }


@dataclass
class PolicyImpactReport:
    """Aggregate result of :func:`audit_policy_change`."""

    policy_path: Optional[str]
    policy_version_pin: Optional[str]
    trace_impacts: List[TraceImpact] = field(default_factory=list)
    substitution_kinds: List[str] = field(default_factory=list)

    # ----------------------------------- aggregate stats (computed)

    @property
    def trace_count(self) -> int:
        return len(self.trace_impacts)

    @property
    def traces_with_divergence(self) -> int:
        return sum(1 for t in self.trace_impacts if t.has_divergence)

    @property
    def traces_with_error(self) -> int:
        return sum(1 for t in self.trace_impacts if t.error is not None)

    @property
    def total_divergent_steps(self) -> int:
        return sum(t.divergent_step_count for t in self.trace_impacts)

    @property
    def total_newly_blocked_steps(self) -> int:
        return sum(len(t.newly_blocked_step_ids) for t in self.trace_impacts)

    @property
    def total_newly_allowed_steps(self) -> int:
        return sum(len(t.newly_allowed_step_ids) for t in self.trace_impacts)

    @property
    def total_baseline_cost_usd(self) -> float:
        return round(sum(t.baseline_cost_usd for t in self.trace_impacts), 8)

    @property
    def total_replayed_cost_usd(self) -> float:
        return round(sum(t.replayed_cost_usd for t in self.trace_impacts), 8)

    @property
    def total_cost_delta_usd(self) -> float:
        return round(sum(t.cost_delta_usd for t in self.trace_impacts), 8)

    # ----------------------------------------------- serialization

    def to_json(self) -> dict:
        return {
            "policy_path": self.policy_path,
            "policy_version_pin": self.policy_version_pin,
            "substitution_kinds": list(self.substitution_kinds),
            "trace_count": self.trace_count,
            "traces_with_divergence": self.traces_with_divergence,
            "traces_with_error": self.traces_with_error,
            "total_divergent_steps": self.total_divergent_steps,
            "total_newly_blocked_steps": self.total_newly_blocked_steps,
            "total_newly_allowed_steps": self.total_newly_allowed_steps,
            "total_baseline_cost_usd": self.total_baseline_cost_usd,
            "total_replayed_cost_usd": self.total_replayed_cost_usd,
            "total_cost_delta_usd": self.total_cost_delta_usd,
            "trace_impacts": [t.to_json() for t in self.trace_impacts],
        }

    def to_json_str(self) -> str:
        return json.dumps(self.to_json(), indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        """Render a deterministic Markdown report.

        Sections:
            1. Header (policy path / version pin / substitution kinds)
            2. Aggregate roll-up table
            3. Per-trace summary table
            4. Per-trace step-level breakdowns (only for divergent
               traces — clean traces are covered by the roll-up)
        """
        lines: List[str] = []
        lines.append("# Policy Impact Report")
        lines.append("")
        lines.append("## Inputs")
        lines.append("")
        lines.append(f"- policy_path: `{self.policy_path or '(none)'}`")
        lines.append(f"- policy_version_pin: `{self.policy_version_pin or '(none)'}`")
        lines.append(
            f"- substitution_kinds: `{', '.join(self.substitution_kinds) or '(none)'}`"
        )
        lines.append("")
        lines.append("## Aggregate")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("| --- | ---: |")
        lines.append(f"| traces audited | {self.trace_count} |")
        lines.append(f"| traces with divergence | {self.traces_with_divergence} |")
        lines.append(f"| traces with error | {self.traces_with_error} |")
        lines.append(f"| divergent steps (total) | {self.total_divergent_steps} |")
        lines.append(f"| newly blocked steps | {self.total_newly_blocked_steps} |")
        lines.append(f"| newly allowed steps | {self.total_newly_allowed_steps} |")
        lines.append(
            f"| baseline cost (USD) | {self.total_baseline_cost_usd:.6f} |"
        )
        lines.append(
            f"| replayed cost (USD) | {self.total_replayed_cost_usd:.6f} |"
        )
        lines.append(f"| cost delta (USD) | {self.total_cost_delta_usd:+.6f} |")
        lines.append("")
        lines.append("## Per-trace summary")
        lines.append("")
        lines.append(
            "| trace | steps | divergent | newly_blocked | newly_allowed | "
            "baseline $ | replayed $ | delta $ | error |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for t in self.trace_impacts:
            lines.append(
                "| `{p}` | {n} | {d} | {nb} | {na} | {bc:.6f} | {rc:.6f} | "
                "{cd:+.6f} | {err} |".format(
                    p=t.trace_path,
                    n=t.step_count,
                    d=t.divergent_step_count,
                    nb=len(t.newly_blocked_step_ids),
                    na=len(t.newly_allowed_step_ids),
                    bc=t.baseline_cost_usd,
                    rc=t.replayed_cost_usd,
                    cd=t.cost_delta_usd,
                    err=t.error or "",
                )
            )
        lines.append("")
        any_div = any(t.has_divergence or t.error for t in self.trace_impacts)
        if any_div:
            lines.append("## Divergent step detail")
            lines.append("")
            for t in self.trace_impacts:
                if not (t.has_divergence or t.error):
                    continue
                lines.append(f"### `{t.trace_path}`")
                lines.append("")
                if t.error:
                    lines.append(f"**ERROR:** {t.error}")
                    lines.append("")
                    continue
                lines.append("| step | kind | name | classification | delta $ |")
                lines.append("| --- | --- | --- | --- | ---: |")
                for s in t.step_impacts:
                    if s.classification == "unchanged":
                        continue
                    lines.append(
                        "| `{sid}` | {k} | {n} | {c} | {cd:+.6f} |".format(
                            sid=s.step_id,
                            k=s.kind,
                            n=s.name or "",
                            c=s.classification,
                            cd=s.cost_delta_usd,
                        )
                    )
                lines.append("")
        # Trailing newline so Markdown viewers render the last row.
        return "\n".join(lines) + "\n"


# ----------------------------------------------------------- main API


def _build_subs(
    extra: Optional[Sequence[Substitution]],
    policy_path: Optional[str],
    policy_step_id: str,
) -> SubstitutionSet:
    subs = SubstitutionSet()
    if policy_path is not None:
        subs.add(PolicySubstitution(at_step=policy_step_id, policy_path=policy_path))
    for s in extra or ():
        subs.add(s)
    return subs


def _classify_pair(
    base: Optional[StepView], rep: Optional[StepView]
) -> Tuple[str, bool, bool, float, bool]:
    base_block = is_policy_blocked(base.outputs) if base is not None else False
    rep_block = is_policy_blocked(rep.outputs) if rep is not None else False
    base_out = base.outputs if base is not None else None
    rep_out = rep.outputs if rep is not None else None
    diverged_cache = bool(rep is not None and rep.dirty)
    cd = (rep.cost_usd if rep else 0.0) - (base.cost_usd if base else 0.0)
    if base_out == rep_out:
        return ("unchanged", base_block, rep_block, cd, diverged_cache)
    if rep_block and not base_block:
        return ("newly_blocked", base_block, rep_block, cd, diverged_cache)
    if base_block and not rep_block:
        return ("newly_allowed", base_block, rep_block, cd, diverged_cache)
    return ("divergent", base_block, rep_block, cd, diverged_cache)


def _audit_one(
    trace_path: str,
    *,
    hmac_key: Optional[bytes],
    subs: SubstitutionSet,
    executor: Executor,
) -> TraceImpact:
    try:
        t: Trace = load_trace(trace_path, hmac_key=hmac_key)
    except Exception as e:  # noqa: BLE001
        return TraceImpact(
            trace_path=trace_path,
            step_count=0,
            baseline_cost_usd=0.0,
            replayed_cost_usd=0.0,
            cost_delta_usd=0.0,
            divergent_step_count=0,
            error=f"{type(e).__name__}: {e}",
        )

    baseline_exec = Executor(fallback_recorded=True)
    baseline = t.run_replay(SubstitutionSet(), baseline_exec)
    replayed = t.run_replay(subs, executor)

    base_by = {s.step_id: s for s in baseline.steps}
    rep_by = {s.step_id: s for s in replayed.steps}
    all_ids = list(base_by.keys()) + [
        sid for sid in rep_by.keys() if sid not in base_by
    ]

    impacts: List[StepImpact] = []
    newly_blocked: List[str] = []
    newly_allowed: List[str] = []
    divergent = 0
    for sid in all_ids:
        b = base_by.get(sid)
        r = rep_by.get(sid)
        present = r or b
        assert present is not None
        kind = present.kind
        name = present.name
        cls, bb, rb, cd, divc = _classify_pair(b, r)
        impacts.append(
            StepImpact(
                step_id=sid,
                kind=kind,
                name=name,
                classification=cls,
                baseline_blocked=bb,
                replayed_blocked=rb,
                cost_delta_usd=round(cd, 8),
                diverged_from_cache=divc,
            )
        )
        if cls == "newly_blocked":
            newly_blocked.append(sid)
        if cls == "newly_allowed":
            newly_allowed.append(sid)
        if cls != "unchanged":
            divergent += 1

    return TraceImpact(
        trace_path=trace_path,
        step_count=len(replayed.steps),
        baseline_cost_usd=round(baseline.total_cost_usd, 8),
        replayed_cost_usd=round(replayed.total_cost_usd, 8),
        cost_delta_usd=round(replayed.total_cost_usd - baseline.total_cost_usd, 8),
        divergent_step_count=divergent,
        newly_blocked_step_ids=newly_blocked,
        newly_allowed_step_ids=newly_allowed,
        step_impacts=impacts,
    )


def audit_policy_change(
    trace_paths: Iterable[str],
    *,
    hmac_key: Optional[bytes] = None,
    new_policy_path: Optional[str] = None,
    policy_version_pin: Optional[str] = None,
    extra_substitutions: Optional[Sequence[Substitution]] = None,
    policy_step_id: str = "step:0",
    executor: Optional[Executor] = None,
) -> PolicyImpactReport:
    """Re-execute every trace under a (counterfactual) policy.

    Parameters
    ----------
    trace_paths
        Iterable of `.sb` paths to audit. Order is preserved in the
        output report.
    hmac_key
        Optional HMAC key. If supplied, every trace is verified
        before audit; if a trace fails verification, its
        :class:`TraceImpact` row carries ``error`` and ``divergent_step_count
        = 0`` and the audit continues with the rest.
    new_policy_path
        Path to the new policy file. Wired in as a single
        :class:`PolicySubstitution` at ``policy_step_id`` (default
        ``step:0``); the substitution writes ``policy_path`` into
        the step inputs which (for traces whose recorded inputs
        carried no policy) marks the entire downstream subtree
        dirty.
    policy_version_pin
        Free-form metadata recorded in the report header (e.g.
        ``"2026-04-15"``). Used by the regulator-replay
        attestation summary line.
    extra_substitutions
        Additional substitutions to apply alongside the policy
        substitution. Typically per-step
        :class:`RaiseSubstitution` /
        :class:`OutputsPatchSubstitution` entries that simulate the
        new policy's denials on specific tool calls (for traces
        that would not naturally re-execute under the new policy
        in the test executor).
    policy_step_id
        Step ID where the :class:`PolicySubstitution` is anchored
        when ``new_policy_path`` is set. Default ``step:0``.
    executor
        Optional :class:`Executor` for the replayed pass. Default
        is an executor with ``fallback_recorded=True``, which
        means dirty steps fall back to the recorded output —
        appropriate for offline policy audits where the user does
        not want to spend real LLM/tool calls.

    Returns
    -------
    PolicyImpactReport
    """
    subs = _build_subs(extra_substitutions, new_policy_path, policy_step_id)
    sub_kinds = sorted({type(s).__name__ for s in subs.items})
    paths = list(trace_paths)
    if executor is None:
        executor = Executor(fallback_recorded=True)
    impacts = [
        _audit_one(p, hmac_key=hmac_key, subs=subs, executor=executor)
        for p in paths
    ]
    return PolicyImpactReport(
        policy_path=new_policy_path,
        policy_version_pin=policy_version_pin,
        trace_impacts=impacts,
        substitution_kinds=sub_kinds,
    )


__all__ = [
    "PolicyImpactReport",
    "TraceImpact",
    "StepImpact",
    "audit_policy_change",
    "is_policy_blocked",
]
