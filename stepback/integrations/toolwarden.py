"""stepback ↔ toolwarden integration.

Records toolwarden enforcement decisions (ALLOW / DENY / REDACT / DRY_RUN /
SANDBOX / REQUIRE_APPROVAL) as ``tool_call`` steps in a stepback ``.sb``
trace so that the full policy audit trail is reproducible via stepback's
replay engine.

Two entry points:

* :class:`WardenRecorderShim` — wrap an existing ``Warden`` and a stepback
  ``Recorder`` together so that every ``invoke()`` call is both enforced by
  the policy and recorded as a trace step.  Denied calls are recorded *before*
  ``WardenError`` is re-raised, so the trace captures the full call history
  including denials.

* :func:`import_toolwarden_audit` — convert a toolwarden ``AuditLog`` (or an
  iterable of ``AuditEntry`` objects) into a standalone ``.sb`` file for
  offline replay-time auditing, bisect, and substitution experiments.

Public API
----------
.. code-block:: python

    from stepback import record
    from stepback.integrations.toolwarden import WardenRecorderShim

    with record("audit.sb") as rec:
        shim = WardenRecorderShim(warden, rec, policy_version="pol@v1")
        try:
            result = shim.invoke("web_search", {"q": "..."}, executor=my_fn)
        except Exception:
            pass  # DENY recorded before the raise

.. code-block:: python

    from stepback.integrations.toolwarden import import_toolwarden_audit

    report = import_toolwarden_audit(warden.audit_log, "audit.sb")
    print(report.step_count, "decisions imported")
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from ..canonical import hash_obj, sha256_hex, canonical_json
from ..importers import ImportReport, _emit_step, _open_writer, RecorderKey
from ..recorder import Recorder


__all__ = [
    "WardenRecorderShim",
    "import_toolwarden_audit",
]

# Nondeterminism hash for a deterministic policy decision (no model sampling).
_ZERO_NONDET = sha256_hex(canonical_json({}))


# ======================================================================
# WardenRecorderShim
# ======================================================================


class WardenRecorderShim:
    """Wrap a toolwarden ``Warden`` and a stepback ``Recorder`` together.

    Every ``invoke()`` call is:

    1. Checked by the toolwarden policy via ``warden.check()``.
    2. Recorded as a ``tool_call`` step in the stepback recorder.
    3. Executed if ALLOW/REDACT, or an exception is raised if DENY/etc.

    The recording step always happens *before* any exception is raised so
    denied calls appear in the trace.  The step's ``inputs`` dict includes
    the ``policy_version`` so that a replay with a different policy version
    produces a dirty step (enabling bisect over policy changes).

    Args:
        warden: A ``toolwarden.Warden`` instance (or any duck-typed object
            with a ``check(tool, args, ctx)`` method returning an object
            with ``.outcome``, ``.effective_args``, ``.reasons``, and
            ``.metadata``).
        recorder: An active :class:`~stepback.recorder.Recorder`.
        policy_version: Optional opaque string pinned into every recorded
            step's ``inputs``.  Defaults to the string
            ``"toolwarden@unknown"`` when ``None``.
    """

    def __init__(
        self,
        warden: Any,
        recorder: Recorder,
        *,
        policy_version: Optional[str] = None,
    ) -> None:
        self._warden = warden
        self._recorder = recorder
        self._policy_version = policy_version or "toolwarden@unknown"

    def check(
        self,
        tool: str,
        args: Dict[str, Any],
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Delegate to ``warden.check()`` and return the Decision unchanged.

        This records a ``router`` step (a policy decision without side
        effects) so bisect can isolate which step's policy outcome changed.
        """
        decision = self._warden.check(tool, args, ctx)
        outcome_str = _outcome_str(decision)
        captured = {"outcome": outcome_str, "reasons": list(decision.reasons)}
        self._recorder.router(
            name=f"toolwarden.check:{tool}",
            choice=outcome_str,
            options=["allow", "deny", "redact", "require_approval", "dry_run", "sandbox"],
        )
        return decision

    def invoke(
        self,
        tool: str,
        args: Dict[str, Any],
        executor: Callable[[str, Dict[str, Any]], Any],
        ctx: Optional[Dict[str, Any]] = None,
        **warden_kwargs: Any,
    ) -> Any:
        """Guard an executor via toolwarden and record the outcome as a step.

        The flow is:

        1. ``warden.check(tool, args, ctx)`` — policy evaluation (no side effects).
        2. Record a ``tool_call`` step immediately (before execution) using a
           synthetic executor so that denied calls appear in the trace.
        3. If allowed/redacted: execute ``executor(tool, effective_args)``; the
           step outputs are updated with the real result.
        4. If denied: re-raise ``WardenError`` after recording.

        The step ``inputs`` dict always includes:

        * ``policy_version`` — enables dirty-set propagation across policy upgrades.
        * ``tw_outcome`` — the policy decision string.
        * ``tw_args_rewritten`` — True when the policy rewrote the args (REDACT).

        Args:
            tool: Tool name.
            args: Tool arguments as a plain dict.
            executor: Callable ``(tool_name, args_dict) -> result`` that performs
                the actual tool call.  Only invoked when the policy ALLOWS or
                REDACTs the call.
            ctx: Optional context dict forwarded to ``warden.check()``.
            **warden_kwargs: Extra keyword arguments forwarded to ``warden.invoke()``
                when the call is allowed (e.g. ``approval_id``).

        Returns:
            The return value of ``executor(tool, effective_args)`` on success.

        Raises:
            WardenError: When the policy denies the call (after recording the step).
        """
        decision = self._warden.check(tool, args, ctx)
        outcome_str = _outcome_str(decision)
        effective_args = decision.rewritten_args if decision.rewritten_args is not None else args
        is_allowed = outcome_str in ("allow", "redact")

        # Build deterministic step inputs *including* policy_version so that
        # a policy bump marks downstream steps dirty.
        extra_inputs: Dict[str, Any] = {
            "policy_version": self._policy_version,
            "tw_outcome": outcome_str,
            "tw_args_rewritten": decision.rewritten_args is not None,
        }

        if is_allowed:
            # Execute the tool, then record with the real result.
            result = executor(tool, effective_args)

            def _precomputed(_n: str, _a: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "result": _safe_json(result),
                    "outcome": outcome_str,
                    "decision_id": decision.metadata.get("decision_id"),
                    "reasons": list(decision.reasons),
                }

            self._recorder.tool_call(
                tool,
                {**args, **extra_inputs},
                executor=_precomputed,
            )
            return result
        else:
            # Record the denied/pending/dry-run call before raising.
            def _denied(_n: str, _a: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "result": None,
                    "outcome": outcome_str,
                    "decision_id": decision.metadata.get("decision_id"),
                    "reasons": list(decision.reasons),
                }

            self._recorder.tool_call(
                tool,
                {**args, **extra_inputs},
                executor=_denied,
            )
            # Re-raise using toolwarden's own exception class when available.
            try:
                from toolwarden import WardenError
                raise WardenError(decision)
            except ImportError:
                raise RuntimeError(
                    f"DENY[toolwarden] {tool}: {'; '.join(decision.reasons)}"
                ) from None


# ======================================================================
# import_toolwarden_audit
# ======================================================================


def import_toolwarden_audit(
    audit_source: Any,
    output_path: str,
    *,
    principal: str = "unknown",
    key: Optional[RecorderKey] = None,
    compression: bool = False,
) -> ImportReport:
    """Import a toolwarden ``AuditLog`` (or iterable of ``AuditEntry`` objects)
    into a ``.sb`` trace file.

    Each ``AuditEntry`` becomes a ``tool_call`` step:

    * ``inputs`` — ``{kind, name, arguments, policy_version, tw_outcome,
      tw_channel, tw_principal}`` providing full replay identity.
    * ``outputs`` — ``{result, outcome, decision_id, reasons, matched_rules}``.
    * ``nondeterminism_hash`` — the zero hash (policy decisions are
      deterministic given the same inputs + policy version).

    Steps appear in the order they were recorded in the audit log.  Parent/child
    relationships from the ``parent_decision_id`` field are preserved via
    stepback's ``parent_step_id`` linkage.

    Args:
        audit_source: A ``toolwarden.AuditLog`` instance, or any iterable of
            ``AuditEntry``-compatible objects (must have ``tool``, ``args``,
            ``outcome``, ``reasons``, ``matched_rules``, ``channel``,
            ``principal``, ``policy_version_id``, ``decision_id``,
            ``parent_decision_id`` attributes/keys).
        output_path: Path for the output ``.sb`` file.
        principal: Default principal name used as the trace author.
        key: Optional :class:`~stepback.importers.RecorderKey`.  A fresh key
            is generated when ``None``.
        compression: Write compressed frames (default ``False``).

    Returns:
        An :class:`~stepback.importers.ImportReport` with step counts.
    """
    entries: Iterable[Any]
    if hasattr(audit_source, "entries"):
        entries = audit_source.entries
    else:
        entries = list(audit_source)

    writer, key = _open_writer(output_path, key=key, compression=compression)
    report = ImportReport(
        output_path=output_path,
        source_format="toolwarden_audit",
    )

    # Map decision_id → step_id for parent chain reconstruction.
    decision_id_to_step_id: Dict[str, str] = {}
    step_counter = 0

    try:
        for entry in entries:
            step_counter += 1
            sid = f"step:{step_counter}"

            # Extract fields tolerantly (support both attribute and dict access).
            e = _AttrOrDictAccessor(entry)
            tool_name = str(e["tool"])
            args = dict(e.get("args") or {})
            outcome = str(e.get("outcome") or "unknown")
            reasons = list(e.get("reasons") or [])
            matched_rules = list(e.get("matched_rules") or [])
            channel = str(e.get("channel") or "request")
            entry_principal = str(e.get("principal") or principal)
            policy_version_id = e.get("policy_version_id") or "unknown"
            decision_id = e.get("decision_id") or sid
            parent_decision_id = e.get("parent_decision_id")

            # Resolve parent_step_id from prior decision_id mapping.
            parent_step_id: Optional[str] = None
            if parent_decision_id and parent_decision_id in decision_id_to_step_id:
                parent_step_id = decision_id_to_step_id[parent_decision_id]

            decision_id_to_step_id[decision_id] = sid

            # Sanitize args for canonical JSON.
            safe_args = _safe_json(args)

            inputs: Dict[str, Any] = {
                "kind": "tool_call",
                "name": tool_name,
                "arguments": safe_args,
                "policy_version": str(policy_version_id),
                "tw_outcome": outcome,
                "tw_channel": channel,
                "tw_principal": entry_principal,
            }
            outputs: Dict[str, Any] = {
                "result": None,
                "outcome": outcome,
                "decision_id": str(decision_id),
                "reasons": reasons,
                "matched_rules": matched_rules,
            }

            ts_float = e.get("ts") or 0.0
            wallclock_ns = int(float(ts_float) * 1_000_000_000) if ts_float else time.time_ns()

            _emit_step(
                writer,
                step_id=sid,
                step_kind="tool_call",
                name=tool_name,
                parent_step_id=parent_step_id,
                inputs=inputs,
                outputs=outputs,
                cost_usd=0.0,
                wallclock_ns=wallclock_ns,
            )
            report.kind_counts["tool_call"] = report.kind_counts.get("tool_call", 0) + 1
            report.step_count += 1

    finally:
        writer.close()

    return report


# ======================================================================
# Internal helpers
# ======================================================================


def _outcome_str(decision: Any) -> str:
    """Return the string outcome from a toolwarden Decision (or duck-type)."""
    outcome = decision.outcome
    if hasattr(outcome, "value"):
        return str(outcome.value)
    return str(outcome)


def _safe_json(obj: Any) -> Any:
    """Recursively coerce an object to a JSON-serializable primitive tree.

    Handles dicts, lists, str/int/float/bool/None.  Anything else is
    converted to its ``str()`` representation so canonical_json never raises.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    return str(obj)


class _AttrOrDictAccessor:
    """Uniform ``[]`` / ``.get()`` access over dict-like or attribute-based objects."""

    def __init__(self, obj: Any) -> None:
        self._obj = obj

    def __getitem__(self, key: str) -> Any:
        if isinstance(self._obj, dict):
            return self._obj[key]
        return getattr(self._obj, key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, AttributeError):
            return default
