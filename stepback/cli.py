"""`stepback` CLI.

Subcommands:

* ``stepback inspect TRACE [--json]`` — print step timeline.
* ``stepback verify  TRACE --hmac-key-hex HEX`` — verify chain + sigs.
* ``stepback bisect  TRACE --good ID --bad ID --predicate PYEXPR``
* ``stepback replay  TRACE [--substitute SPEC ...] [--branch-out FILE]
                     [--json]`` — apply typed substitutions and print a
  cache/dirty/cost report. ``--branch-out`` saves the counterfactual to
  a `.sbb` file so it can be diffed later without re-typing the specs.
* ``stepback diff TRACE [BRANCH_A.sbb] [--branch BRANCH_B.sbb]
                  [--substitute SPEC ...]`` — compare two replays of
  the same trace. With no branch arguments, compares
  ``no-substitution`` vs ``--substitute`` flags. Output is JSON.

Substitution spec grammar (see :func:`stepback.branch_io.parse_substitution_spec`):

    prompt@step:N=path/to/messages.json
    prompt@step:N=:inline:[{"role":"system","content":"..."}]
    model@step:N=gpt-4o-mini-2024-07-18
    tool_output@step:N=path/to/response.json
    tool_output@step:N=:inline:{"customer_id":null}
    policy@step:N=path/to/policy.tw
    router@step:N=branchA
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import autorecord
from .recorder import RecorderKey
from .branch_io import (
    BranchTraceMismatch,
    diff_replays,
    load_branch,
    parse_substitution_spec,
    save_branch,
    trace_chain_hash,
)
from .replay import Trace, replay
from .report import (
    ReportOptions,
    available_formats,
    dump_report_json,
    render_counterfactual_report,
    render_html_report,
    render_replay_report,
    render_report,
)
from .substitutions import SubstitutionSet
from .trace_diff import diff_traces, render_trace_diff
from .policy_audit import audit_policy_change
from .trace_reader import verify_trace
from .attestation import (
    AttestationVerificationError,
    build_attestation_pack,
    read_attestation_pack,
    verify_attestation_pack,
    write_attestation_pack,
)


def _cmd_record(args: argparse.Namespace) -> int:
    """Run a Python script under an ambient stepback recorder.

    Usage::

        stepback record --output trace.sb -- python my_agent.py [SCRIPT_ARGS...]

    The leading ``python`` / ``python3`` token is optional — if the first
    token after ``--`` ends in ``.py`` we run it directly via
    :func:`runpy.run_path`. The script runs *in this process* so that
    autorecord patches against ``openai`` / ``anthropic`` / LangChain
    take effect transparently. The script can also call
    :func:`stepback.autorecord.current_recorder` to record explicitly.

    Exit code is the script's: a non-zero exit propagates, but the trace
    file is still flushed to disk so the partial run can be replayed and
    bisected (the README §"Use-cases" §1 incident-investigation flow
    depends on this).
    """
    import runpy

    output = args.output
    cmd = list(args.command or [])
    # argparse.REMAINDER preserves a leading `--`; strip it.
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print(
            "stepback record: a python script must follow `--`. "
            "Example: stepback record --output trace.sb -- python my_agent.py",
            file=sys.stderr,
        )
        return 2

    # Strip a leading python/python3/sys.executable token if present.
    head = cmd[0]
    if head in ("python", "python3", sys.executable) or head.endswith(
        ("/python", "/python3")
    ):
        cmd = cmd[1:]
    if not cmd:
        print(
            "stepback record: no script path after `python` token.",
            file=sys.stderr,
        )
        return 2

    script = cmd[0]
    if not script.endswith(".py") and "/" not in script and "\\" not in script:
        # Allow `-m module.name` form.
        if script == "-m" and len(cmd) >= 2:
            module_name = cmd[1]
            script_argv = cmd[1:]
        else:
            print(
                f"stepback record: don't know how to run {script!r}; "
                "expected a .py path or `-m module`.",
                file=sys.stderr,
            )
            return 2
        run_kind = "module"
    elif script == "-m" and len(cmd) >= 2:
        module_name = cmd[1]
        script_argv = cmd[1:]
        run_kind = "module"
    else:
        run_kind = "script"
        script_argv = cmd

    saved_argv = sys.argv[:]
    sys.argv = list(script_argv)

    rc = 0
    try:
        with autorecord.enable(output, key=RecorderKey.fresh()):
            try:
                if run_kind == "script":
                    runpy.run_path(script, run_name="__main__")
                else:
                    runpy.run_module(module_name, run_name="__main__", alter_sys=True)
            except SystemExit as se:
                rc = int(se.code) if isinstance(se.code, int) else (0 if se.code is None else 1)
            except BaseException as exc:  # noqa: BLE001
                # Log via the recorder so the trace captures the failure.
                try:
                    autorecord.current_recorder().exception(
                        type(exc).__name__, str(exc)
                    )
                except Exception:
                    pass
                print(
                    f"stepback record: script raised {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                rc = 1
    finally:
        sys.argv = saved_argv

    if rc == 0:
        print(f"wrote {output}")
    else:
        print(f"wrote {output} (script exited with {rc})", file=sys.stderr)
    return rc


def _cmd_inspect(args: argparse.Namespace) -> int:
    t = replay(args.trace)
    if args.json:
        body = {
            "trace": args.trace,
            "header": {
                "recorder_version": t.header.get("recorder_version"),
                "canonicalisation_version": t.header.get("canonicalisation_version"),
                "price_list_version": t.header.get("price_list_version"),
            },
            "step_count": len(t.recorded_steps),
            "steps": [
                {
                    "step_id": s["step_id"],
                    "kind": s["step_kind"],
                    "name": s.get("name"),
                    "parent_step_id": s.get("parent_step_id"),
                    "cost_usd": s.get("cost_usd", 0.0),
                    "inputs_hash": s.get("inputs_hash"),
                    "outputs_hash": s.get("outputs_hash"),
                }
                for s in t.recorded_steps
            ],
        }
        json.dump(body, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    print(f"trace: {args.trace}")
    print(f"  recorder_version          = {t.header.get('recorder_version')}")
    print(f"  canonicalisation_version  = {t.header.get('canonicalisation_version')}")
    print(f"  steps                     = {len(t.recorded_steps)}")
    for s in t.recorded_steps:
        print(
            f"  {s['step_id']:>10}  {s['step_kind']:<14} "
            f"name={(s.get('name') or '')!r:<30} cost=${s.get('cost_usd', 0):.5f}"
        )
    return 0


def _apply_subs(t: Trace, specs: List[str]) -> SubstitutionSet:
    subs = SubstitutionSet()
    for spec in specs:
        subs.add(parse_substitution_spec(spec))
    return subs


def _cmd_replay(args: argparse.Namespace) -> int:
    t = replay(args.trace)
    if args.from_branch:
        loaded = load_branch(
            args.from_branch, expected_chain=trace_chain_hash(t.recorded_steps)
        )
        subs = loaded["substitutions"]
        for s in (args.substitute or []):
            subs.add(parse_substitution_spec(s))
        base_step = loaded["base_step"]
        name = loaded["name"]
    else:
        subs = _apply_subs(t, args.substitute or [])
        base_step = args.base_step or (
            t.recorded_steps[0]["step_id"] if t.recorded_steps else "step:1"
        )
        name = args.name or "counterfactual"

    if args.branch_out:
        save_branch(
            args.branch_out,
            name=name,
            base_step=base_step,
            trace_path=args.trace,
            trace_chain=trace_chain_hash(t.recorded_steps),
            substitutions=list(subs.items),
        )

    from .replay import Executor

    executor = Executor(fallback_recorded=True)
    try:
        result = t.run_replay(subs, executor)
    except Exception as e:  # MissingExecutor or similar
        print(
            f"replay would require real LLM/tool execution; "
            f"the CLI cannot supply executors. error: {e}",
            file=sys.stderr,
        )
        return 3

    if args.json:
        body = {
            "trace": args.trace,
            "branch_name": name,
            "base_step": base_step,
            "substitution_count": len(subs.items),
            "real_executions": result.real_executions,
            "cache_hit_count": result.cache_hit_count,
            "dirty_count": result.dirty_count,
            "total_cost_usd": round(result.total_cost_usd, 8),
            "branch_out": args.branch_out,
            "steps": [
                {
                    "step_id": s.step_id,
                    "kind": s.kind,
                    "dirty": s.dirty,
                    "cache_hit": s.cache_hit,
                    "cost_usd": s.cost_usd,
                }
                for s in result.steps
            ],
        }
        json.dump(body, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(
            f"replay {args.trace}  subs={len(subs.items)}  "
            f"cache_hits={result.cache_hit_count}  dirty={result.dirty_count}  "
            f"real_exec={result.real_executions}  "
            f"total_cost=${result.total_cost_usd:.5f}"
        )
        if args.branch_out:
            print(f"  → wrote branch  {args.branch_out}")
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    t = replay(args.trace)
    chain = trace_chain_hash(t.recorded_steps)

    def _resolve(branch_path: Optional[str], specs: List[str]) -> SubstitutionSet:
        if branch_path:
            try:
                loaded = load_branch(branch_path, expected_chain=chain)
            except BranchTraceMismatch as e:
                print(f"FAIL: {e}", file=sys.stderr)
                sys.exit(4)
            subs = loaded["substitutions"]
        else:
            subs = SubstitutionSet()
        for s in specs or []:
            subs.add(parse_substitution_spec(s))
        return subs

    subs_a = _resolve(args.a_branch, args.a_substitute)
    subs_b = _resolve(args.b_branch, args.b_substitute)

    from .replay import Executor

    ra = t.run_replay(subs_a, Executor(fallback_recorded=True))
    rb = t.run_replay(subs_b, Executor(fallback_recorded=True))
    body = diff_replays(ra, rb)
    body["a_substitution_count"] = len(subs_a.items)
    body["b_substitution_count"] = len(subs_b.items)
    json.dump(body, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    key = bytes.fromhex(args.hmac_key_hex)
    try:
        v = verify_trace(args.trace, key)
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 2
    print(f"OK  steps={len(v.steps)}  pubkey={v.public_key_hex[:16]}…")
    return 0


def _load_subs_from_args(
    t: Trace, branch_path: Optional[str], specs: List[str]
) -> tuple[SubstitutionSet, Optional[str]]:
    """Resolve substitutions from a `.sbb` branch file and/or `--substitute` flags.

    Returns ``(subs, branch_name)``. If neither source contributes
    anything, returns ``(empty_set, None)``.
    """
    name: Optional[str] = None
    if branch_path:
        loaded = load_branch(
            branch_path, expected_chain=trace_chain_hash(t.recorded_steps)
        )
        subs = loaded["substitutions"]
        name = loaded.get("name")
    else:
        subs = SubstitutionSet()
    for s in specs or []:
        subs.add(parse_substitution_spec(s))
    return subs, name


def _infer_format_from_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = path.lower()
    if p.endswith(".html") or p.endswith(".htm"):
        return "html"
    if p.endswith(".json"):
        return "json"
    if p.endswith(".md") or p.endswith(".markdown"):
        return "markdown"
    return None


def _cmd_report(args: argparse.Namespace) -> int:
    from .replay import Executor

    t = replay(args.trace)
    executor = Executor(fallback_recorded=True)

    subs_b, branch_name = _load_subs_from_args(
        t, args.branch, args.substitute or []
    )
    if args.baseline_branch:
        subs_a, _ = _load_subs_from_args(t, args.baseline_branch, [])
    else:
        subs_a = SubstitutionSet()

    title = args.title or (
        f"counterfactual: {branch_name}"
        if branch_name
        else "stepback counterfactual report"
    )
    options = ReportOptions(
        title=title,
        max_step_rows=args.max_rows,
        truncate_text=args.truncate,
    )

    fmt = (args.format or "").lower()
    if not args.format or args.format == "auto":
        fmt = _infer_format_from_path(args.output) or "markdown"
    if fmt == "md":
        fmt = "markdown"

    if not subs_b.items and not args.branch:
        # Single-replay report: just the recorded run.
        result = t.run_replay(subs_a, executor)
        out_text = render_report(t, result, None, subs_a, format=fmt, options=options)
        if fmt == "json" and not out_text.endswith("\n"):
            out_text += "\n"
    else:
        baseline = t.run_replay(subs_a, executor)
        counterfactual = t.run_replay(subs_b, executor)
        out_text = render_report(
            t, baseline, counterfactual, subs_b, format=fmt, options=options
        )
        if fmt == "json" and not out_text.endswith("\n"):
            out_text += "\n"

    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as f:
            f.write(out_text)
        print(f"wrote {args.output} ({len(out_text)} bytes)")
    else:
        sys.stdout.write(out_text)
    return 0


def _cmd_bisect(args: argparse.Namespace) -> int:
    t = replay(args.trace)
    pred = compile(args.predicate, "<predicate>", "eval")
    found = t.bisect(
        good=args.good,
        bad=args.bad,
        predicate=lambda step: bool(eval(pred, {"step": step})),
    )
    if found is None:
        print("no step matched the predicate")
        return 1
    print(json.dumps({"first_bad": found.step_id, "kind": found.kind, "name": found.name}))
    return 0


def _cmd_minimize(args: argparse.Namespace) -> int:
    """Delta-debug a substitution set down to a 1-minimal triggering subset."""
    from .minimize import (
        BinaryHalvingStrategy,
        BruteForceStrategy,
        BudgetExhausted,
        DDMinStrategy,
        LinearShrinkStrategy,
        MinimizeOptions,
        PredicateNotTriggered,
        ShapleyAttributionStrategy,
        find_all_minimal,
        minimize_substitutions,
    )

    t = replay(args.trace)
    subs = SubstitutionSet()
    if args.from_branch:
        loaded = load_branch(args.from_branch)
        for s in loaded.substitutions.items:
            subs.add(s)
    for spec in args.substitute or []:
        subs.add(parse_substitution_spec(spec))
    if not subs.items:
        print("error: no substitutions provided", file=sys.stderr)
        return 2

    pred_code = compile(args.predicate, "<predicate>", "eval")

    def predicate(result):
        return bool(eval(pred_code, {"result": result, "any": any, "all": all}))

    strategy_name = getattr(args, "strategy", "ddmin") or "ddmin"
    strategy_map = {
        "ddmin": DDMinStrategy(),
        "linear": LinearShrinkStrategy(),
        "binary": BinaryHalvingStrategy(),
        "brute": BruteForceStrategy(),
        "shapley": ShapleyAttributionStrategy(),
    }
    if strategy_name not in strategy_map:
        print(f"error: unknown strategy '{strategy_name}'", file=sys.stderr)
        return 2
    strategy = strategy_map[strategy_name]
    options = MinimizeOptions(
        strategy=strategy,
        probe_budget=getattr(args, "probe_budget", None),
    )

    from .replay import Executor

    def _serialise_result(outcome) -> dict:
        d = {
            "strategy": outcome.strategy_name,
            "probes": outcome.probes,
            "cache_hits": outcome.cache_hits,
            "minimal_count": len(outcome.minimal),
            "removed_count": len(outcome.removed),
            "minimal": [
                {"kind": type(s).__name__, "at_step": s.at_step}
                for s in outcome.minimal
            ],
            "removed": [
                {"kind": type(s).__name__, "at_step": s.at_step}
                for s in outcome.removed
            ],
            "final_cost_usd": (
                outcome.final_result.total_cost_usd if outcome.final_result else 0.0
            ),
            "final_dirty_count": (
                outcome.final_result.dirty_count if outcome.final_result else 0
            ),
        }
        if outcome.weights is not None:
            d["weights"] = [
                {
                    "kind": type(s).__name__,
                    "at_step": s.at_step,
                    "weight": round(outcome.weight_for(s), 6),
                }
                for s in list(outcome.minimal) + list(outcome.removed)
            ]
        return d

    try:
        if getattr(args, "all_witnesses", False):
            witnesses = find_all_minimal(
                t, subs, predicate,
                max_witnesses=getattr(args, "max_witnesses", 4),
                options=options,
                executor=Executor(fallback_recorded=True),
            )
            if not witnesses:
                print(
                    "error: predicate does not fire under the full substitution set",
                    file=sys.stderr,
                )
                return 4
            payload = {
                "strategy": strategy.name,
                "witness_count": len(witnesses),
                "witnesses": [_serialise_result(w) for w in witnesses],
            }
        else:
            outcome = minimize_substitutions(
                t, subs, predicate,
                options=options,
                executor=Executor(fallback_recorded=True),
            )
            payload = _serialise_result(outcome)
    except PredicateNotTriggered as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    except BudgetExhausted as exc:
        print(f"error: {exc}", file=sys.stderr)
        partial = _serialise_result(exc.partial)
        partial["budget_exhausted"] = True
        json.dump(partial, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 5

    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _cmd_attest(args: argparse.Namespace) -> int:
    """Build a regulator-replay attestation pack over one or more traces."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    hmac_key = bytes.fromhex(args.hmac_key_hex)
    if args.signing_key_hex:
        signing_key = Ed25519PrivateKey.from_private_bytes(
            bytes.fromhex(args.signing_key_hex)
        )
    else:
        signing_key = Ed25519PrivateKey.generate()

    # Build a single shared substitution list (applied to every trace).
    subs = SubstitutionSet()
    for spec in args.substitute or []:
        subs.add(parse_substitution_spec(spec))
    if args.policy_file:
        # Convenience: --policy-file FILE expands to a PolicySubstitution
        # at step:0 that pins every trace to this policy file.
        subs.add(parse_substitution_spec(f"policy@step:0={args.policy_file}"))

    pack = build_attestation_pack(
        list(args.traces),
        hmac_key=hmac_key,
        substitutions=subs if subs else None,
        policy_version_pin=args.policy_version_pin,
        attestor_signing_key=signing_key,
    )
    write_attestation_pack(pack, args.out, signing_key=signing_key)
    summary = pack.summary
    if args.json:
        json.dump(
            {
                "out": args.out,
                "attestor_public_key": pack.attestor_public_key,
                "summary": summary,
            },
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
    else:
        print(f"wrote attestation pack: {args.out}")
        print(f"  attestor: {pack.attestor_public_key}")
        print(f"  traces:        {summary['trace_count']}")
        print(
            f"  verified ok:   {summary['verified_ok']}"
            f"  fail: {summary['verified_fail']}"
        )
        print(
            f"  divergent:     {summary['divergent_traces']}"
            f"  total $\u0394: {summary['total_cost_delta_usd']:.6f}"
        )
    # Non-zero exit if any trace failed verification, so this composes
    # in shell pipelines.
    return 0 if summary["verified_fail"] == 0 else 3


def _cmd_verify_pack(args: argparse.Namespace) -> int:
    try:
        data = verify_attestation_pack(
            args.pack, expected_public_key=args.expected_public_key
        )
    except AttestationVerificationError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 2
    summary = data.get("summary", {})
    print(
        f"OK  pack={args.pack}  attestor={data.get('attestor_public_key', '')[:32]}…"
    )
    print(
        f"    traces={summary.get('trace_count', 0)}"
        f"  verified_ok={summary.get('verified_ok', 0)}"
        f"  divergent={summary.get('divergent_traces', 0)}"
    )
    return 0


def _cmd_trace_diff(args: argparse.Namespace) -> int:
    """Cross-trace structural diff (regression analysis)."""
    hmac_a = bytes.fromhex(args.a_hmac_key_hex) if args.a_hmac_key_hex else None
    hmac_b = bytes.fromhex(args.b_hmac_key_hex) if args.b_hmac_key_hex else None
    d = diff_traces(args.a_trace, args.b_trace, hmac_key_a=hmac_a, hmac_key_b=hmac_b)

    fmt = (args.format or "auto").lower()
    if fmt == "auto":
        fmt = _infer_format_from_path(args.output) or "markdown"
    if fmt == "md":
        fmt = "markdown"
    if args.summary_only:
        out_text = json.dumps(d.summary(), indent=2, sort_keys=True) + "\n"
    else:
        out_text = render_trace_diff(
            d, format=fmt, max_rows=args.max_rows, truncate=args.truncate
        )
        if fmt == "json" and not out_text.endswith("\n"):
            out_text += "\n"

    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as f:
            f.write(out_text)
        print(f"wrote {args.output} ({len(out_text)} bytes)")
    else:
        sys.stdout.write(out_text)

    if args.exit_nonzero_on_divergence and not d.is_identical:
        return 3
    return 0


def _cmd_policy_audit(args: argparse.Namespace) -> int:
    """Re-run a set of traces under a new policy; emit a structured report."""
    hmac_key = bytes.fromhex(args.hmac_key_hex) if args.hmac_key_hex else None
    extra: List = []
    for spec in args.substitute or []:
        extra.append(parse_substitution_spec(spec))

    report = audit_policy_change(
        list(args.traces),
        hmac_key=hmac_key,
        new_policy_path=args.policy_path,
        policy_version_pin=args.policy_version_pin,
        extra_substitutions=extra,
        policy_step_id=args.policy_step_id,
    )

    fmt = (args.format or "auto").lower()
    if fmt == "auto":
        fmt = _infer_format_from_path(args.output) or "markdown"
    if fmt == "md":
        fmt = "markdown"
    if fmt == "json":
        out_text = report.to_json_str() + "\n"
    else:
        out_text = report.to_markdown()

    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as f:
            f.write(out_text)
        print(f"wrote {args.output} ({len(out_text)} bytes)")
    else:
        sys.stdout.write(out_text)

    if args.exit_nonzero_on_divergence and report.traces_with_divergence > 0:
        return 3
    return 0


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(prog="stepback", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_record = sub.add_parser(
        "record",
        help="run a Python agent script under an ambient stepback recorder",
        description=(
            "Run a Python agent script with stepback recording enabled "
            "process-globally. openai/anthropic/langchain clients already "
            "imported (or imported by the script) are auto-wrapped. The "
            "script can also call stepback.autorecord.current_recorder() "
            "to record explicit llm_call/tool_call/router steps."
        ),
    )
    p_record.add_argument(
        "--output", "-o", required=True,
        help="output .sb trace path",
    )
    p_record.add_argument(
        "command", nargs=argparse.REMAINDER,
        help="`-- python my_agent.py [ARGS...]` (the leading `python` is "
             "optional; bare `script.py [ARGS...]` also works)",
    )
    p_record.set_defaults(func=_cmd_record)

    p_inspect = sub.add_parser("inspect", help="print step timeline")
    p_inspect.add_argument("trace")
    p_inspect.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    p_inspect.set_defaults(func=_cmd_inspect)

    p_replay = sub.add_parser(
        "replay",
        help="apply substitutions and report dirty/cache/cost stats",
    )
    p_replay.add_argument("trace")
    p_replay.add_argument(
        "--substitute", "-s", action="append",
        help="KIND@step:N=BODY (repeatable). See module docstring for grammar.",
    )
    p_replay.add_argument(
        "--from-branch", help="load substitutions from a saved .sbb file"
    )
    p_replay.add_argument(
        "--branch-out", help="save the resulting counterfactual to a .sbb file"
    )
    p_replay.add_argument("--name", help="branch name (default: 'counterfactual')")
    p_replay.add_argument("--base-step", help="base step id for the saved branch")
    p_replay.add_argument("--json", action="store_true")
    p_replay.set_defaults(func=_cmd_replay)

    p_diff = sub.add_parser(
        "diff", help="compare two replays of a trace; emits JSON",
    )
    p_diff.add_argument("trace")
    p_diff.add_argument("--a-branch", help=".sbb file describing branch A")
    p_diff.add_argument("--b-branch", help=".sbb file describing branch B")
    p_diff.add_argument(
        "--a-substitute", action="append", default=[],
        help="extra substitution spec for branch A (repeatable)",
    )
    p_diff.add_argument(
        "--b-substitute", action="append", default=[],
        help="extra substitution spec for branch B (repeatable)",
    )
    p_diff.set_defaults(func=_cmd_diff)

    p_verify = sub.add_parser("verify", help="verify HMAC chain + signatures")
    p_verify.add_argument("trace")
    p_verify.add_argument("--hmac-key-hex", required=True)
    p_verify.set_defaults(func=_cmd_verify)

    p_bisect = sub.add_parser(
        "bisect",
        help="binary-search for the earliest step matching a predicate",
    )
    p_bisect.add_argument("trace")
    p_bisect.add_argument("--good", required=True)
    p_bisect.add_argument("--bad", required=True)
    p_bisect.add_argument(
        "--predicate", required=True, help="Python expression over `step`"
    )
    p_bisect.set_defaults(func=_cmd_bisect)

    p_min = sub.add_parser(
        "minimize",
        help="delta-debug a substitution set to a 1-minimal triggering subset",
        description=(
            "Given a set of substitutions and a Python predicate over the "
            "ReplayResult (`result`), shrink the set to a 1-minimal subset "
            "that still flips the predicate. Each probe is a cached replay "
            "(zero LLM calls when no executor is configured)."
        ),
    )
    p_min.add_argument("trace")
    p_min.add_argument(
        "--substitute", "-s", action="append",
        help="substitution spec (repeatable). Same grammar as `replay`.",
    )
    p_min.add_argument(
        "--from-branch", help="load substitutions from a saved .sbb file"
    )
    p_min.add_argument(
        "--predicate", required=True,
        help='Python expression over `result` (a ReplayResult), e.g. '
             '"result.any_step(lambda s: s.cost_usd > 0.5)" or '
             '"any(\'GB99\' in str(s.outputs) for s in result.steps)"',
    )
    p_min.add_argument(
        "--strategy",
        choices=["ddmin", "linear", "binary", "brute", "shapley"],
        default="ddmin",
        help="reduction strategy (default: ddmin)",
    )
    p_min.add_argument(
        "--probe-budget", type=int, default=None,
        help="abort search after N replays; emits partial result with exit code 5",
    )
    p_min.add_argument(
        "--all-witnesses", action="store_true",
        help="enumerate up to --max-witnesses disjoint minimal subsets",
    )
    p_min.add_argument(
        "--max-witnesses", type=int, default=4,
        help="max number of witnesses for --all-witnesses (default 4)",
    )
    p_min.set_defaults(func=_cmd_minimize)

    p_report = sub.add_parser(
        "report",
        help="render a Markdown counterfactual report (incident write-up)",
    )
    p_report.add_argument("trace")
    p_report.add_argument(
        "--substitute", "-s", action="append",
        help="substitution spec for branch B (repeatable)",
    )
    p_report.add_argument(
        "--branch", help="load branch B substitutions from a .sbb file"
    )
    p_report.add_argument(
        "--baseline-branch",
        help="load branch A (baseline) substitutions from a .sbb file "
             "(default: empty / recorded run)",
    )
    p_report.add_argument(
        "-o", "--output", help="write report to this file instead of stdout"
    )
    p_report.add_argument("--title", help="report title (markdown H1)")
    p_report.add_argument(
        "--max-rows", type=int, default=200,
        help="max rows in the step timeline table (default: 200)",
    )
    p_report.add_argument(
        "--truncate", type=int, default=120,
        help="character cap for inline output snippets (default: 120)",
    )
    p_report.add_argument(
        "--format", choices=["md", "markdown", "json", "html", "auto"],
        default="auto",
        help="output format. 'auto' (default) infers from --output extension "
             "(.html→html, .json→json, .md→markdown), else 'markdown'.",
    )
    p_report.set_defaults(func=_cmd_report)

    p_attest = sub.add_parser(
        "attest",
        help="build a regulator-replay attestation pack over one or more traces",
    )
    p_attest.add_argument("traces", nargs="+", help=".sb trace paths")
    p_attest.add_argument("--hmac-key-hex", required=True)
    p_attest.add_argument(
        "--out", required=True, help="output .pack path"
    )
    p_attest.add_argument(
        "--signing-key-hex",
        help="32-byte Ed25519 attestor private key (hex). If omitted, "
             "a fresh ephemeral key is generated and the public half is "
             "written into the pack.",
    )
    p_attest.add_argument(
        "--policy-version-pin",
        help="opaque label recorded in the pack body (e.g. '2026-04-15')",
    )
    p_attest.add_argument(
        "--policy-file",
        help="convenience: applies policy@step:0=PATH to every trace",
    )
    p_attest.add_argument(
        "--substitute", "-s", action="append",
        help="extra substitution spec applied to every trace (repeatable)",
    )
    p_attest.add_argument("--json", action="store_true")
    p_attest.set_defaults(func=_cmd_attest)

    p_verify_pack = sub.add_parser(
        "verify-pack",
        help="verify the signature on an attestation pack",
    )
    p_verify_pack.add_argument("pack")
    p_verify_pack.add_argument(
        "--expected-public-key",
        help="pin the attestor key (e.g. 'ed25519:abcd...'); fails "
             "verification on mismatch",
    )
    p_verify_pack.set_defaults(func=_cmd_verify_pack)

    p_trace_diff = sub.add_parser(
        "trace-diff",
        help="structurally diff two recorded .sb traces (regression analysis)",
    )
    p_trace_diff.add_argument("a_trace", help="baseline .sb trace (A)")
    p_trace_diff.add_argument("b_trace", help="comparison .sb trace (B)")
    p_trace_diff.add_argument(
        "--a-hmac-key-hex",
        help="optional HMAC key for verifying trace A before diffing",
    )
    p_trace_diff.add_argument(
        "--b-hmac-key-hex",
        help="optional HMAC key for verifying trace B before diffing",
    )
    p_trace_diff.add_argument(
        "--format", choices=["md", "markdown", "json", "auto"], default="auto"
    )
    p_trace_diff.add_argument("-o", "--output")
    p_trace_diff.add_argument("--max-rows", type=int, default=200)
    p_trace_diff.add_argument("--truncate", type=int, default=80)
    p_trace_diff.add_argument(
        "--summary-only", action="store_true",
        help="emit just the JSON summary (no per-step detail)",
    )
    p_trace_diff.add_argument(
        "--exit-nonzero-on-divergence", action="store_true",
        help="exit code 3 if the two traces are not structurally identical "
             "(useful in CI: detect a regression vs. a golden trace)",
    )
    p_trace_diff.set_defaults(func=_cmd_trace_diff)

    p_policy_audit = sub.add_parser(
        "policy-audit",
        help="re-execute traces under a new policy; emit a structured impact report",
        description=(
            "Regulator-replay: re-run one or many .sb traces under a "
            "(counterfactual) policy and emit a structured report of "
            "every step whose decision would now differ. Implements "
            "README §Use-cases #2 (counterfactual policy) and #5 "
            "(regulator replay) and the README §CLI line "
            "`stepback verify --policy-changed-since DATE`."
        ),
    )
    p_policy_audit.add_argument(
        "traces", nargs="+", help="one or more .sb trace paths to audit"
    )
    p_policy_audit.add_argument(
        "--hmac-key-hex",
        help="optional HMAC key (hex) for verifying every trace before audit",
    )
    p_policy_audit.add_argument(
        "--policy-path",
        help="path to the new policy file (wired in via PolicySubstitution at --policy-step-id)",
    )
    p_policy_audit.add_argument(
        "--policy-step-id", default="step:0",
        help="step ID to anchor the PolicySubstitution at (default step:0)",
    )
    p_policy_audit.add_argument(
        "--policy-version-pin",
        help="free-form policy version tag recorded in the report header",
    )
    p_policy_audit.add_argument(
        "--substitute", "-s", action="append", default=[],
        help="extra substitution spec (same grammar as `stepback replay`); may be repeated",
    )
    p_policy_audit.add_argument(
        "--format", choices=["md", "markdown", "json", "auto"], default="auto",
        help="output format (default: infer from --output extension, else markdown)",
    )
    p_policy_audit.add_argument("-o", "--output", help="write report to FILE instead of stdout")
    p_policy_audit.add_argument(
        "--exit-nonzero-on-divergence", action="store_true",
        help="exit code 3 if any trace diverged under the new policy",
    )
    p_policy_audit.set_defaults(func=_cmd_policy_audit)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
