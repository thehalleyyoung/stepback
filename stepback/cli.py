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
from pathlib import Path
from typing import Any, List, Optional

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
from .html_view import write_trace_html, write_time_travel_html
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


def _cmd_view(args: argparse.Namespace) -> int:
    hmac_key: Optional[bytes] = None
    if args.hmac_key_hex:
        try:
            hmac_key = bytes.fromhex(args.hmac_key_hex)
        except ValueError as e:
            print(f"invalid --hmac-key-hex: {e}", file=sys.stderr)
            return 2
    summary = write_trace_html(
        args.trace,
        args.output,
        hmac_key=hmac_key,
        title=args.title,
    )
    if args.json:
        json.dump(
            {
                "output_path": summary.output_path,
                "step_count": summary.step_count,
                "total_cost_usd": summary.total_cost_usd,
                "by_kind": summary.by_kind,
                "bytes_written": summary.bytes_written,
            },
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
    else:
        print(
            f"wrote {summary.output_path}  "
            f"({summary.step_count} steps, {summary.bytes_written} bytes)"
        )
    return 0


def _cmd_debug(args: argparse.Namespace) -> int:
    """Render a time-travel debugger HTML page for a trace + optional substitutions."""
    hmac_key: Optional[bytes] = None
    if args.hmac_key_hex:
        try:
            hmac_key = bytes.fromhex(args.hmac_key_hex)
        except ValueError as e:
            print(f"invalid --hmac-key-hex: {e}", file=sys.stderr)
            return 2
    subs = [parse_substitution_spec(s) for s in (args.sub or [])]
    summary = write_time_travel_html(
        args.trace,
        args.output,
        hmac_key=hmac_key,
        substitutions=subs or None,
        title=args.title,
    )
    if args.json:
        json.dump(
            {
                "output_path": summary.output_path,
                "step_count": summary.step_count,
                "dirty_count": summary.dirty_count,
                "cache_hit_count": summary.cache_hit_count,
                "total_cost_usd": summary.total_cost_usd,
                "by_kind": summary.by_kind,
                "bytes_written": summary.bytes_written,
            },
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
    else:
        print(
            f"wrote {summary.output_path}  "
            f"({summary.step_count} steps, {summary.dirty_count} dirty, "
            f"{summary.cache_hit_count} cache hits, {summary.bytes_written} bytes)"
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
    strict = getattr(args, "strict", False)
    policy_path = getattr(args, "policy", None)

    if strict or policy_path:
        from .verify_policy import VerifyPolicy, load_policy, verify_with_policy
        if policy_path:
            try:
                policy = load_policy(policy_path)
            except ValueError as e:
                print(f"FAIL: {e}", file=sys.stderr)
                return 2
            if strict:
                policy.strict = True
        else:
            policy = VerifyPolicy.strict_default()
        result = verify_with_policy(args.trace, key, policy)
        if result.ok:
            v = result.trace
            pubkey_prefix = v.public_key_hex[:16] if v else "unknown"
            steps = len(v.steps) if v else "?"
            print(f"OK  steps={steps}  pubkey={pubkey_prefix}…")
            return 0
        else:
            for violation in result.violations:
                print(f"FAIL [{violation.check}]: {violation.message}", file=sys.stderr)
            return 1
    else:
        try:
            v = verify_trace(args.trace, key)
        except Exception as e:
            print(f"FAIL: {e}", file=sys.stderr)
            return 2
        print(f"OK  steps={len(v.steps)}  pubkey={v.public_key_hex[:16]}…")
        return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from .exporters import export_trace_file, TraceExportError
    try:
        key = bytes.fromhex(args.hmac_key_hex)
    except ValueError as e:
        print(f"FAIL: --hmac-key-hex not valid hex: {e}", file=sys.stderr)
        return 2
    try:
        report = export_trace_file(
            args.format, args.input, args.output, hmac_key=key,
        )
    except TraceExportError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 4
    if args.json:
        print(json.dumps(report.as_dict(), sort_keys=True))
    else:
        print(
            f"OK  format={report.target_format}  steps={report.step_count}  "
            f"output={report.output_path}"
        )
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    from .importers import import_trace, TraceImportError
    key: Optional[RecorderKey]
    if args.hmac_key_hex:
        try:
            raw = bytes.fromhex(args.hmac_key_hex)
        except ValueError as e:
            print(f"FAIL: --hmac-key-hex not valid hex: {e}", file=sys.stderr)
            return 2
        key = RecorderKey.from_bytes(raw) if hasattr(RecorderKey, "from_bytes") \
            else RecorderKey.fresh()
        # Best-effort: if a from_bytes is unavailable, mint fresh and
        # warn so the caller knows the key was not honoured.
        if not hasattr(RecorderKey, "from_bytes"):
            print("WARN: RecorderKey.from_bytes unavailable; minted fresh key",
                  file=sys.stderr)
    else:
        key = RecorderKey.fresh()
    try:
        report = import_trace(
            args.format, args.input, args.output,
            key=key, compression=not args.no_compression,
        )
    except TraceImportError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 4
    if args.json:
        print(json.dumps(report.as_dict(), sort_keys=True))
    else:
        print(
            f"OK  format={report.source_format}  steps={report.step_count}  "
            f"output={report.output_path}"
        )
    return 0


def _cmd_redact(args: argparse.Namespace) -> int:
    from .redact import (
        STANDARD_POLICY, STRICT_POLICY,
        redact_trace_file, redact_trace_file_streaming,
    )
    policies = {"standard": STANDARD_POLICY, "strict": STRICT_POLICY}
    pol = policies.get(args.policy)
    if pol is None:
        print(
            f"FAIL: unknown --policy {args.policy!r}; "
            f"known: {sorted(policies)}",
            file=sys.stderr,
        )
        return 2
    try:
        in_key = bytes.fromhex(args.hmac_key_hex)
    except ValueError as e:
        print(f"FAIL: --hmac-key-hex not valid hex: {e}", file=sys.stderr)
        return 2
    fn = redact_trace_file_streaming if getattr(args, "streaming", False) \
        else redact_trace_file
    try:
        manifest = fn(
            args.trace, args.output,
            in_hmac_key=in_key,
            policy=pol,
            compression=not args.no_compression,
        )
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 3
    payload = manifest.to_dict() if hasattr(manifest, "to_dict") else dict(
        policy_name=manifest.policy_name,
        n_steps=getattr(manifest, "n_steps", 0),
        n_redactions=getattr(manifest, "n_redactions", 0),
    )
    if args.manifest:
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True, indent=2)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(
            f"OK  policy={payload.get('policy_name')}  "
            f"steps={payload.get('n_steps')}  "
            f"redactions={payload.get('n_redactions')}"
        )
    return 0


def _cmd_redact_scan(args: argparse.Namespace) -> int:
    from .redact import STANDARD_POLICY, STRICT_POLICY, scan_trace_file
    policies = {"standard": STANDARD_POLICY, "strict": STRICT_POLICY}
    pol = policies.get(args.policy)
    if pol is None:
        print(
            f"FAIL: unknown --policy {args.policy!r}; "
            f"known: {sorted(policies)}",
            file=sys.stderr,
        )
        return 2
    try:
        in_key = bytes.fromhex(args.hmac_key_hex)
    except ValueError as e:
        print(f"FAIL: --hmac-key-hex not valid hex: {e}", file=sys.stderr)
        return 2
    try:
        report = scan_trace_file(args.trace, in_hmac_key=in_key, policy=pol)
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 3
    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(
            f"OK  policy={payload.get('policy_name')}  "
            f"steps={payload.get('n_steps')}  "
            f"findings={payload.get('n_findings')}"
        )
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
        loaded_subs: SubstitutionSet = loaded["substitutions"]
        for s in loaded_subs.items:
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


def _cmd_sweep(args: argparse.Namespace) -> int:
    from .sweep import (
        render_sweep_report,
        render_sweep_report_json,
        sweep_traces,
    )

    paths: List[str] = list(args.traces)
    candidate_subs = [parse_substitution_spec(s) for s in (args.substitute or [])]
    baseline_subs = [parse_substitution_spec(s) for s in (args.baseline_substitute or [])]

    report = sweep_traces(
        paths,
        candidate_subs,
        baseline_substitutions=baseline_subs,
        base_step=args.base_step,
        on_error="record",
    )

    fmt = args.format
    if fmt == "auto":
        if args.output and args.output.endswith(".json"):
            fmt = "json"
        else:
            fmt = "markdown"
    if fmt == "md":
        fmt = "markdown"

    if fmt == "json":
        out_text = json.dumps(
            render_sweep_report_json(report, include_diffs=args.include_diffs),
            indent=2, sort_keys=True,
        ) + "\n"
    else:
        out_text = render_sweep_report(
            report, title=args.title, max_rows=args.max_rows
        )

    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as f:
            f.write(out_text)
        print(f"wrote {args.output} ({len(out_text)} bytes)")
    else:
        sys.stdout.write(out_text)

    if args.exit_nonzero_on_failure and report.n_traces_failed > 0:
        return 4
    if args.exit_nonzero_on_divergence and report.n_traces_diverged > 0:
        return 3
    return 0


def _cmd_divergence(args: argparse.Namespace) -> int:
    """Re-execute every step in TRACE against an Executor and report divergences.

    The executor is built by importing a Python callable per kind via
    ``--llm-callable pkg.mod:fn`` / ``--tool-callable pkg.mod:fn``.
    Steps without a wired-up executor for their kind are listed under
    ``skipped_step_ids`` in the report.

    With no executor flags, the only useful classification is the
    self-replay sanity check: pass ``--executor-recorded`` and every
    step is "executed" by returning the recorded output (so every
    classification is IDENTICAL).  Mostly useful for smoke tests and
    for asserting that the comparator itself is sound.
    """
    from .divergence import detect_divergences

    hmac_key = bytes.fromhex(args.hmac_key_hex)
    llm = _import_callable(args.llm_callable) if args.llm_callable else None
    tool = _import_callable(args.tool_callable) if args.tool_callable else None
    router = _import_callable(args.router_callable) if args.router_callable else None

    if args.executor_recorded and any([llm, tool, router]):
        print(
            "--executor-recorded is mutually exclusive with --llm-callable / "
            "--tool-callable / --router-callable",
            file=sys.stderr,
        )
        return 2

    if args.executor_recorded:
        # Synthesise an Executor that re-emits the recorded outputs for
        # every kind by reading them back from the trace.
        from .replay import Executor as _Exec
        from .trace_reader import verify_trace as _vt
        recorded_by_step: dict = {}
        for s in _vt(args.trace, hmac_key).steps:
            recorded_by_step[s["step_id"]] = s.get("outputs")
        # We need to know which step is being executed. The Executor
        # interface only provides (kind, inputs); we key by the canonical
        # hash of inputs to look up the matching recorded output.
        from .canonical import hash_obj as _h
        by_inp_hash: dict = {}
        for s in _vt(args.trace, hmac_key).steps:
            sk = s.get("step_kind") or s.get("kind")
            by_inp_hash[(sk, _h(s.get("inputs", {})))] = s.get("outputs")
        def _replay_llm(model, messages):
            return by_inp_hash[("llm_call", _h({"model": model, "messages": messages}))]
        def _replay_tool(name, arguments):
            return by_inp_hash[("tool_call", _h({"name": name, "arguments": arguments}))]
        def _replay_router(name, options):
            return by_inp_hash[("router", _h({"name": name, "options": options}))]
        executor = _Exec(llm=_replay_llm, tool=_replay_tool, router=_replay_router)
        # Strip the {"result": ...} wrapper executor.execute adds for tools.
        # Easier: use the raw recorded output via a custom executor subclass.
        class _RecordedExecutor(_Exec):
            def execute(self, kind, inputs, *, branch_outputs=None):
                self.real_calls += 1
                return by_inp_hash.get((kind, _h(inputs)))
        # Set every callback to a sentinel so detect_divergences() does
        # not classify the kind as 'no executor available'.
        _noop: Any = lambda *a, **k: None
        executor = _RecordedExecutor(llm=_noop, tool=_noop, router=_noop)
    else:
        from .replay import Executor
        executor = Executor(llm=llm, tool=tool, router=router)

    report = detect_divergences(
        args.trace,
        hmac_key=hmac_key,
        executor=executor,
        max_steps=args.max_steps,
    )

    fmt = args.format
    if fmt == "auto":
        if args.output and args.output.endswith(".json"):
            fmt = "json"
        else:
            fmt = "markdown"
    if fmt == "md":
        fmt = "markdown"

    if fmt == "json":
        out_text = json.dumps(report.to_json(), indent=2, sort_keys=True) + "\n"
    else:
        out_text = report.render_markdown(max_rows=args.max_rows)

    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as f:
            f.write(out_text)
        print(f"wrote {args.output} ({len(out_text)} bytes)")
    else:
        sys.stdout.write(out_text)

    if args.exit_nonzero_on_divergence and report.divergent_count > 0:
        return 3
    if args.severity_threshold is not None and report.severity_score > args.severity_threshold:
        return 5
    return 0


def _import_callable(spec: str):
    """Resolve ``pkg.mod:attr`` into a Python callable."""
    if ":" not in spec:
        raise ValueError(
            f"callable spec must be 'pkg.mod:attr', got {spec!r}"
        )
    mod_name, attr = spec.split(":", 1)
    import importlib
    mod = importlib.import_module(mod_name)
    obj = mod
    for part in attr.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"{spec!r} resolved to non-callable {type(obj).__name__}")
    return obj



def _cmd_bench_replay_caching(args: argparse.Namespace) -> int:
    from .bench.replay_caching import run as _run_rc

    try:
        result = _run_rc(
            n_steps=args.n_steps,
            n_trials=args.n_trials,
            strategy=args.strategy,
            seed=args.seed,
        )
    except RuntimeError as e:
        print(f"bench failed: {e}", file=sys.stderr)
        return 4
    print(result.summary_line())
    if args.out:
        import os as _os
        d = _os.path.dirname(_os.path.abspath(args.out))
        if d:
            _os.makedirs(d, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result.to_json(), f, indent=2, sort_keys=True)
            f.write("\n")
    return 0


def _cmd_bench_record_overhead(args: argparse.Namespace) -> int:
    from .bench.record_overhead import run as _run_ro

    try:
        result = _run_ro(n_steps=args.n_steps)
    except RuntimeError as e:
        print(f"bench failed: {e}", file=sys.stderr)
        return 4
    print(result.summary_line())
    if args.out:
        import os as _os
        d = _os.path.dirname(_os.path.abspath(args.out))
        if d:
            _os.makedirs(d, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result.to_json(), f, indent=2, sort_keys=True)
            f.write("\n")
    return 0


def _cmd_bench_soak(args: argparse.Namespace) -> int:
    from .bench.soak import run as _run_soak

    try:
        result = _run_soak(
            n_traces=args.n_traces,
            n_steps=args.n_steps,
            seed=args.seed,
            do_substitution=not args.no_substitution,
            progress_every=args.progress_every,
            track_memory=args.track_memory,
        )
    except (RuntimeError, ValueError) as e:
        print(f"bench failed: {e}", file=sys.stderr)
        return 4
    print(result.summary_line())
    if args.out:
        import os as _os
        d = _os.path.dirname(_os.path.abspath(args.out))
        if d:
            _os.makedirs(d, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result.to_json(), f, indent=2, sort_keys=True)
            f.write("\n")
    return 0


def _cmd_bench_dirty_set_distributions(args: argparse.Namespace) -> int:
    from .bench.dirty_set_distributions import run as _run_dsd

    try:
        corpora = args.corpora.split(",") if args.corpora else None
        result = _run_dsd(
            n_trials=args.n_trials,
            n_steps=args.n_steps,
            seed=args.seed,
            corpora=corpora,
        )
    except (RuntimeError, ValueError) as e:
        print(f"bench failed: {e}", file=sys.stderr)
        return 4

    # Print summary table
    print(
        f"dirty-set distributions: {result.total_trials} trials across "
        f"{len(result.distributions)} cells ({result.wall_time_ms:.0f} ms)\n"
    )
    print(
        f"{'corpus':<20} {'position':<8} {'sub_kind':<24} "
        f"{'p50':>6} {'p95':>6} {'mean':>6}"
    )
    print("-" * 72)
    for d in result.distributions:
        sk = d.sub_kind.replace("Substitution", "Sub")
        print(
            f"{d.corpus:<20} {d.position:<8} {sk:<24} "
            f"{d.p50:>6.2f} {d.p95:>6.2f} {d.mean:>6.2f}"
        )

    if args.out:
        import os as _os
        _dir = _os.path.dirname(_os.path.abspath(args.out))
        if _dir:
            _os.makedirs(_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result.to_json(), f, indent=2, sort_keys=True)
            f.write("\n")
    return 0


def _cmd_bench_model_swap(args: argparse.Namespace) -> int:
    from .bench.model_swap import run as _run_ms

    try:
        result = _run_ms(
            n_steps=args.n_steps,
            n_trials=args.n_trials,
            seed=args.seed,
            model_a_name=args.model_a,
            model_b_name=args.model_b,
        )
    except (RuntimeError, ValueError) as e:
        print(f"bench failed: {e}", file=sys.stderr)
        return 4
    print(result.summary_line())
    if args.out:
        import os as _os
        d = _os.path.dirname(_os.path.abspath(args.out))
        if d:
            _os.makedirs(d, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result.to_json(), f, indent=2, sort_keys=True)
            f.write("\n")
    return 0


def _cmd_bench_storage_compression(args: argparse.Namespace) -> int:
    from .bench.storage_compression import run as _run_sc

    try:
        result = _run_sc(
            n_traces=args.n_traces,
            n_steps=args.n_steps,
            seed=args.seed,
        )
    except (RuntimeError, ValueError) as e:
        print(f"bench failed: {e}", file=sys.stderr)
        return 4
    print(result.summary_line())
    if args.out:
        import os as _os
        d = _os.path.dirname(_os.path.abspath(args.out))
        if d:
            _os.makedirs(d, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result.to_json(), f, indent=2, sort_keys=True)
            f.write("\n")
    return 0


def _cmd_bench_microbenchmarks(args: argparse.Namespace) -> int:
    from .bench.microbenchmarks import run as _run_mb

    try:
        result = _run_mb(n_iter=args.n_iter)
    except (RuntimeError, ValueError) as e:
        print(f"bench failed: {e}", file=sys.stderr)
        return 4
    for line in result.summary_lines():
        print(line)
    if args.out:
        import os as _os
        d = _os.path.dirname(_os.path.abspath(args.out))
        if d:
            _os.makedirs(d, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result.to_json(), f, indent=2, sort_keys=True)
            f.write("\n")
    return 0


def _cmd_bench_leaderboard(args: argparse.Namespace) -> int:
    import os as _os
    from .bench.leaderboard import (
        build_leaderboard,
        generate_leaderboard_html,
        generate_leaderboard_json,
        load_submissions_from_dir,
    )

    # Accept a directory or individual files
    inputs = args.submissions
    if not inputs:
        print("bench leaderboard: no inputs provided", file=sys.stderr)
        return 2

    raw_submissions = []
    for path in inputs:
        if _os.path.isdir(path):
            raw_submissions.extend(load_submissions_from_dir(path))
        elif _os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                raw_submissions.append((path, data))
            except Exception as e:
                raw_submissions.append((path, {"_parse_error": str(e)}))
        else:
            print(f"bench leaderboard: path not found: {path}", file=sys.stderr)
            return 2

    lb = build_leaderboard(raw_submissions)
    accepted = len(lb.accepted)
    rejected = len(lb.rejected)
    print(
        f"leaderboard: {accepted} accepted, {rejected} rejected "
        f"({lb.total_submitted} total)"
    )

    if args.out:
        d = _os.path.dirname(_os.path.abspath(args.out))
        if d:
            _os.makedirs(d, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(generate_leaderboard_json(lb), f, indent=2, sort_keys=True)
            f.write("\n")

    if args.html:
        d = _os.path.dirname(_os.path.abspath(args.html))
        if d:
            _os.makedirs(d, exist_ok=True)
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(generate_leaderboard_html(lb))

    return 0 if rejected == 0 else 1


def _cmd_bench_frontier_reeval_schedule(args: argparse.Namespace) -> int:
    import os as _os
    from .bench.frontier_eval import (
        ReEvaluationPolicy,
        compute_schedule,
        default_catalog,
        load_catalog_from_json,
        save_schedule_json,
    )
    from .bench.leaderboard import load_submissions_from_dir as _lb_load
    from datetime import date as _date

    raw_submissions: list = []
    if args.submissions:
        for path in args.submissions:
            if _os.path.isdir(path):
                loaded = _lb_load(path)
                raw_submissions.extend(raw for _, raw in loaded)
            elif _os.path.isfile(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        raw_submissions.append(json.load(f))
                except Exception as e:
                    raw_submissions.append({"_parse_error": str(e)})
            else:
                print(
                    f"bench frontier-reeval-schedule: path not found: {path}",
                    file=sys.stderr,
                )
                return 2

    catalog = (
        load_catalog_from_json(args.catalog) if args.catalog else default_catalog()
    )

    policy = ReEvaluationPolicy(
        max_age_days=args.max_age_days,
        check_superseded=not args.no_superseded,
        check_new_frontier=not args.no_new_frontier,
    )

    # Build a submission_id → models mapping if --models-file was given
    models_map: dict = {}
    if args.models_file:
        with open(args.models_file, encoding="utf-8") as f:
            models_map = json.load(f)

    schedule = compute_schedule(
        raw_submissions,
        catalog=catalog,
        policy=policy,
        reference_date=_date.fromisoformat(args.reference_date) if args.reference_date else None,
        submission_models=models_map if models_map else None,
    )

    total = schedule.total_submissions
    needs = schedule.needs_reeval_count
    print(
        f"frontier-reeval-schedule: {needs}/{total} submissions need re-evaluation"
    )

    if args.out:
        d = _os.path.dirname(_os.path.abspath(args.out))
        if d:
            _os.makedirs(d, exist_ok=True)
        save_schedule_json(schedule, args.out)

    return 0 if needs == 0 else 1


def _cmd_bench_minimization(args: argparse.Namespace) -> int:
    from .bench.minimization import run as _run_min

    try:
        result = _run_min(
            n_steps=args.n_steps,
            n_noisy=args.n_noisy,
            n_trials=args.n_trials,
            seed=args.seed,
        )
    except (RuntimeError, ValueError) as e:
        print(f"bench failed: {e}", file=sys.stderr)
        return 4
    print(result.summary_line())
    if args.out:
        import os as _os
        d = _os.path.dirname(_os.path.abspath(args.out))
        if d:
            _os.makedirs(d, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result.to_json(), f, indent=2, sort_keys=True)
            f.write("\n")
    return 0


def _cmd_bench_validate_submission(args: argparse.Namespace) -> int:
    import os as _os
    from .bench.submission import validate_submission_json

    path = args.manifest
    if not _os.path.isfile(path):
        print(
            f"bench validate-submission: file not found: {path}", file=sys.stderr
        )
        return 2

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"bench validate-submission: JSON parse error: {e}", file=sys.stderr)
        return 2

    result = validate_submission_json(data)
    print(result.summary_line())

    if args.out:
        d = _os.path.dirname(_os.path.abspath(args.out))
        if d:
            _os.makedirs(d, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result.to_json(), f, indent=2, sort_keys=True)
            f.write("\n")

    return 0 if result.valid else 1


def _cmd_spec_test(args: argparse.Namespace) -> int:
    """Step 47: ``stepback spec test <impl-argv>...`` runs the conformance
    corpus against an external implementation and exits 0 iff every
    fixture passes."""
    from .spec_runner import run_conformance, render_text

    manifest_path = Path(args.manifest) if args.manifest else None
    fixtures_dir = Path(args.fixtures) if args.fixtures else None
    only = args.only or None

    if not args.impl_argv:
        print(
            "stepback spec test: at least one IMPL token is required",
            file=sys.stderr,
        )
        return 2
    try:
        run = run_conformance(
            list(args.impl_argv),
            manifest_path=manifest_path,
            fixtures_dir=fixtures_dir,
            timeout=float(args.timeout),
            enable_hash=not args.no_hash,
            only=only,
        )
    except FileNotFoundError as e:
        print(f"stepback spec test: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"stepback spec test: {e}", file=sys.stderr)
        return 2

    if args.json:
        json.dump(run.to_dict(), sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_text(run))
    return 0 if run.ok else 1


def _parse_listen(spec: str) -> tuple:
    """Parse HOST:PORT or :PORT, returning (host, port)."""
    if ":" not in spec:
        raise ValueError(f"--listen must be HOST:PORT, got {spec!r}")
    host, _, port_str = spec.rpartition(":")
    if not host:
        host = "127.0.0.1"
    try:
        port = int(port_str)
    except ValueError as e:
        raise ValueError(f"invalid port in {spec!r}: {e}") from e
    if port < 0 or port > 65535:
        raise ValueError(f"port out of range in {spec!r}")
    return host, port


def _cmd_diagnose(args: argparse.Namespace) -> int:
    """stepback diagnose — check installed SDK/framework versions."""
    import json as _json
    from .diagnose import diagnose_all, format_diagnose_table, WARN_NEWER, WARN_UNSUPPORTED

    results = diagnose_all()

    if args.json_output:
        data = [
            {
                "name": r.name,
                "package": r.package,
                "kind": r.kind,
                "installed": r.installed,
                "status": r.status,
                "detail": r.detail,
            }
            for r in results
        ]
        print(_json.dumps(data, indent=2))
    else:
        print(format_diagnose_table(results, show_all=args.show_all), end="")

    if args.strict:
        has_warnings = any(
            r.status in (WARN_NEWER, WARN_UNSUPPORTED) for r in results
        )
        return 1 if has_warnings else 0
    return 0


def _cmd_proxy(args: argparse.Namespace) -> int:
    import logging
    import signal
    import threading

    from .proxy import ProxyState, ProxyHTTPServer

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("stepback.cli.proxy")

    try:
        http_host, http_port = _parse_listen(args.listen)
    except ValueError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 2

    state = ProxyState(
        write_dir=args.write,
        max_open_traces=args.max_open_traces,
    )
    http_server = ProxyHTTPServer(
        (http_host, http_port), state, max_body_bytes=args.max_body_bytes,
    )
    log.info(
        "HTTP listening on http://%s:%s (write_dir=%s)",
        http_host, http_port, state.write_dir,
    )

    grpc_server = None
    if args.grpc_listen:
        try:
            grpc_host, grpc_port = _parse_listen(args.grpc_listen)
        except ValueError as e:
            print(f"FAIL: {e}", file=sys.stderr)
            return 2
        try:
            from .proxy.grpc_server import serve_grpc
            grpc_server = serve_grpc(
                state, host=grpc_host, port=grpc_port,
                max_workers=args.grpc_max_workers,
            )
            log.info("gRPC listening on %s:%s", grpc_host, grpc_port)
        except ImportError as e:
            print(f"FAIL: gRPC not available: {e}", file=sys.stderr)
            return 2

    stop = threading.Event()

    def _on_signal(signum, _frame):
        log.info("received signal %s, shutting down", signum)
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            # Not on the main thread (e.g. pytest)
            pass

    http_thread = threading.Thread(
        target=http_server.serve_forever, name="proxy-http", daemon=True
    )
    http_thread.start()
    try:
        stop.wait()
    finally:
        log.info("stopping HTTP server")
        http_server.shutdown()
        http_server.server_close()
        if grpc_server is not None:
            log.info("stopping gRPC server")
            grpc_server.stop(grace=2.0).wait(timeout=5.0)
        log.info("flushing %d open trace(s)", len(state.list_traces()))
        state.close_all()
    return 0


# ─────────────────────────────────────────────────────── stepback doctor ─────


def _cmd_doctor(args: argparse.Namespace) -> int:
    """stepback doctor — run environment health checks."""
    import json as _json
    from .doctor import run_doctor, format_doctor_table, FAIL, WARN

    checks = run_doctor(check_network=getattr(args, "network", False))

    if getattr(args, "json_output", False):
        data = [
            {
                "name": c.name,
                "status": c.status,
                "detail": c.detail,
                "remediation": c.remediation,
            }
            for c in checks
        ]
        print(_json.dumps(data, indent=2))
    else:
        print(format_doctor_table(checks), end="")

    if any(c.status == FAIL for c in checks):
        return 1
    return 0


# ─────────────────────────────────────────────────────── stepback quickstart ─

def _cmd_quickstart(args: argparse.Namespace) -> int:
    """stepback quickstart — interactive first-run setup wizard."""
    from pathlib import Path as _Path
    from .quickstart import run_wizard

    output_dir = _Path(args.output_dir).resolve() if getattr(args, "output_dir", None) else None
    provider = getattr(args, "provider", None)
    api_key = getattr(args, "api_key", None)
    non_interactive = getattr(args, "non_interactive", False)
    skip_open = getattr(args, "skip_open", False)

    try:
        result = run_wizard(
            non_interactive=non_interactive,
            provider_name=provider or None,
            api_key=api_key or None,
            output_dir=output_dir,
            open_browser=True,
            skip_open=skip_open,
        )
    except KeyboardInterrupt:
        print("\nAborted.")
        return 1

    if result.errors and not non_interactive:
        print("\nWarnings during quickstart:")
        for e in result.errors:
            print(f"  • {e}")

    return 0


# ─────────────────────────────────────────────────────── stepback init ───────

_STEPBACK_TOML_TEMPLATE = """\
# stepback configuration — https://github.com/stepback-dev/stepback
[stepback]
version = "1"

# Directory where .sb trace files are written by default.
trace_dir = "traces"

# Set to a hex HMAC key to sign every frame.  Generate with:
#   python -c "import secrets; print(secrets.token_hex(32))"
# hmac_key_hex = ""
"""

_QUICKSTART_AGENT_TEMPLATE = '''\
"""Quickstart example agent for stepback.

This script records a deterministic 12-step customer-payments agent
using stepback\'s built-in fixture.  It demonstrates record / replay /
substitute without requiring an LLM API key.

Run:
    python agent.py

Then replay:
    stepback replay traces/quickstart.sb

Substitute the buggy lookup result and see the dirty-set shrink:
    stepback replay traces/quickstart.sb \\
        --substitute "tool_output@step:2=:inline:{{\\"id\\": \\"acme-us\\", \\"name\\": \\"Acme Bolts Inc\\", \\"country\\": \\"US\\", \\"iban\\": \\"US12-3456-7890\\"}}"
"""
from __future__ import annotations

from pathlib import Path

# Allow running as a standalone script; tries the installed package first
# then falls back to the source tree two levels up.
try:
    import stepback  # noqa: F401 (installed package)
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import stepback
from stepback.testing.agent import run_recorded_agent

TRACE_PATH = Path(__file__).parent / "traces" / "quickstart.sb"


def main() -> None:
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Recording agent → {TRACE_PATH}")
    with stepback.record(str(TRACE_PATH)) as rec:
        run_recorded_agent(rec)
    print("Done.  Replay with:")
    print(f"  stepback replay {TRACE_PATH}")


if __name__ == "__main__":
    main()
'''


def _cmd_init(args: argparse.Namespace) -> int:
    """stepback init — scaffold a new stepback project in the current directory.

    Creates:
      * ``stepback.toml`` — project config (skipped if it already exists).
      * ``examples/quickstart/agent.py`` — runnable demo agent.
      * ``traces/`` directory referenced by the config.
      * Records a deterministic demo trace at
        ``examples/quickstart/traces/quickstart.sb``.

    After this command a brand-new user can immediately run::

        stepback replay examples/quickstart/traces/quickstart.sb
    """
    import textwrap

    target = Path(args.directory).resolve()
    target.mkdir(parents=True, exist_ok=True)

    # ── stepback.toml ────────────────────────────────────────────────────────
    toml_path = target / "stepback.toml"
    if toml_path.exists() and not args.force:
        print(f"  skip  {toml_path}  (already exists; use --force to overwrite)")
    else:
        toml_path.write_text(_STEPBACK_TOML_TEMPLATE)
        print(f"  write {toml_path}")

    # ── traces/ dir ──────────────────────────────────────────────────────────
    traces_dir = target / "traces"
    traces_dir.mkdir(exist_ok=True)
    print(f"  mkdir {traces_dir}")

    # ── examples/quickstart/agent.py ─────────────────────────────────────────
    qs_dir = target / "examples" / "quickstart"
    qs_dir.mkdir(parents=True, exist_ok=True)
    agent_path = qs_dir / "agent.py"
    if agent_path.exists() and not args.force:
        print(f"  skip  {agent_path}  (already exists; use --force to overwrite)")
    else:
        agent_path.write_text(_QUICKSTART_AGENT_TEMPLATE)
        print(f"  write {agent_path}")

    # ── record demo trace ────────────────────────────────────────────────────
    from .recorder import record as _record
    from .testing.agent import run_recorded_agent

    qs_traces_dir = qs_dir / "traces"
    qs_traces_dir.mkdir(exist_ok=True)
    trace_path = qs_traces_dir / "quickstart.sb"

    if trace_path.exists() and not args.force:
        print(f"  skip  {trace_path}  (already exists; use --force to overwrite)")
    else:
        with _record(str(trace_path), signing=False) as rec:
            run_recorded_agent(rec)
        print(f"  write {trace_path}")

    # ── success banner ───────────────────────────────────────────────────────
    rel_trace = trace_path.relative_to(target) if trace_path.is_relative_to(target) else trace_path
    print()
    print("✓ stepback project initialised.  Next steps:")
    print()
    print(f"  1. Inspect the demo trace:")
    print(f"       stepback inspect {rel_trace}")
    print()
    print(f"  2. Replay with no substitutions (all 12 steps cache-hit):")
    print(f"       stepback replay {rel_trace}")
    print()
    print(f"  3. Substitute the buggy tool output and see the dirty-set:")
    print(
        f"       stepback replay {rel_trace} \\\n"
        f'           --substitute "tool_output@step:2=:inline:'
        f'{{\\"id\\": \\"acme-us\\", \\"name\\": \\"Acme Bolts Inc\\", '
        f'\\"country\\": \\"US\\", \\"iban\\": \\"US12-3456-7890\\"}}"'
    )
    print()
    print(f"  4. Run the example agent yourself:")
    print(f"       python {agent_path.relative_to(target) if agent_path.is_relative_to(target) else agent_path}")
    print()
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
    p_verify.add_argument(
        "--strict", action="store_true",
        help="enable strict mode: warnings become errors",
    )
    p_verify.add_argument(
        "--policy",
        metavar="POLICY_FILE",
        help="path to a JSON verify-policy file (VerifyPolicy schema)",
    )
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

    p_sweep = sub.add_parser(
        "sweep",
        help="apply substitutions across a corpus of .sb traces; aggregate the deltas",
        description=(
            "README §Use-cases #3 — 'test a new system prompt on 1000 "
            "production traces'. Loads every trace, branches at "
            "--base-step (default step:0), applies the candidate "
            "substitutions, replays forward (cached unless --executor-* "
            "is wired), and prints a markdown or JSON report of "
            "Δcost / divergent-step counts / decisions changed."
        ),
    )
    p_sweep.add_argument("traces", nargs="+", help=".sb trace paths (or shell glob)")
    p_sweep.add_argument(
        "--substitute", "-s", action="append", default=[],
        help="candidate substitution spec (repeatable). Same grammar as `replay`.",
    )
    p_sweep.add_argument(
        "--baseline-substitute", "-S", action="append", default=[],
        help="baseline substitution spec applied to branch A (repeatable, default empty)",
    )
    p_sweep.add_argument(
        "--base-step", default="step:1",
        help="branch base step id (default: step:1 — first recorded step)",
    )
    p_sweep.add_argument(
        "--format", choices=["md", "markdown", "json", "auto"], default="auto",
        help="output format (default: infer from --output extension, else markdown)",
    )
    p_sweep.add_argument("-o", "--output", help="write report to FILE instead of stdout")
    p_sweep.add_argument("--max-rows", type=int, default=50)
    p_sweep.add_argument("--title", help="report title (markdown H1)")
    p_sweep.add_argument(
        "--include-diffs", action="store_true",
        help="(JSON only) include the full per-step diff per trace",
    )
    p_sweep.add_argument(
        "--exit-nonzero-on-divergence", action="store_true",
        help="exit code 3 if at least one trace diverged",
    )
    p_sweep.add_argument(
        "--exit-nonzero-on-failure", action="store_true",
        help="exit code 4 if at least one trace failed to load/replay",
    )
    p_sweep.set_defaults(func=_cmd_sweep)

    p_div = sub.add_parser(
        "divergence",
        help="re-execute a trace and classify per-step output divergence",
        description=(
            "README §Architecture 'replay/nondet.py' — re-execute every "
            "step in TRACE against a real LLM/tool callable and classify "
            "each step as identical / equivalent / minor / semantic / "
            "structural.  Answers 'how reproducible is this trace right "
            "now against my provider?'."
        ),
    )
    p_div.add_argument("trace", help=".sb trace path")
    p_div.add_argument("--hmac-key-hex", required=True)
    p_div.add_argument(
        "--llm-callable",
        help="pkg.mod:fn — Python callable matching Executor.llm signature",
    )
    p_div.add_argument(
        "--tool-callable",
        help="pkg.mod:fn — Python callable matching Executor.tool signature",
    )
    p_div.add_argument(
        "--router-callable",
        help="pkg.mod:fn — Python callable matching Executor.router signature",
    )
    p_div.add_argument(
        "--executor-recorded", action="store_true",
        help="self-replay: re-emit recorded outputs (smoke-tests the comparator)",
    )
    p_div.add_argument(
        "--format", choices=["md", "markdown", "json", "auto"], default="auto",
    )
    p_div.add_argument("-o", "--output")
    p_div.add_argument("--max-rows", type=int, default=50)
    p_div.add_argument("--max-steps", type=int, default=None)
    p_div.add_argument(
        "--exit-nonzero-on-divergence", action="store_true",
        help="exit code 3 if any step is not IDENTICAL",
    )
    p_div.add_argument(
        "--severity-threshold", type=int, default=None,
        help="exit code 5 if total severity_score > THRESHOLD",
    )
    p_div.set_defaults(func=_cmd_divergence)

    p_export = sub.add_parser(
        "export",
        help="export a verified .sb trace to openai/langsmith/openinference",
        description=(
            "Verify the .sb file's HMAC chain, then render the step list "
            "into one of the foreign trace formats supported by stepback's "
            "exporters module (openai_chat_log, langsmith[_jsonl], "
            "openinference[_spans], otel)."
        ),
    )
    p_export.add_argument("--format", required=True,
                          help="target export format")
    p_export.add_argument("--input", required=True, help=".sb trace path")
    p_export.add_argument("--output", required=True, help="output file path")
    p_export.add_argument("--hmac-key-hex", required=True,
                          help="hex-encoded HMAC key the trace was signed with")
    p_export.add_argument("--json", action="store_true",
                          help="print the ExportReport as JSON on stdout")
    p_export.set_defaults(func=_cmd_export)

    p_import = sub.add_parser(
        "import",
        help="import a foreign trace file into a verified .sb trace",
        description=(
            "Parse an openai_chat_log JSON, langsmith JSONL, or "
            "openinference spans JSON file and emit a freshly-signed .sb "
            "trace. The resulting trace is HMAC-chain-verified end-to-end."
        ),
    )
    p_import.add_argument(
        "--format", required=True,
        choices=sorted({
            "openai", "openai_chat_log",
            "langsmith", "langsmith_jsonl",
            "openinference", "openinference_spans", "otel",
        }),
        help="source import format",
    )
    p_import.add_argument("-i", "--input", required=True,
                          help="path to the foreign trace file")
    p_import.add_argument("-o", "--output", required=True,
                          help="path to the .sb file to write")
    p_import.add_argument("--hmac-key-hex",
                          help="hex-encoded HMAC key (else a fresh key is minted)")
    p_import.add_argument("--no-compression", action="store_true",
                          help="disable per-frame zlib compression")
    p_import.add_argument("--json", action="store_true",
                          help="print the ImportReport as JSON on stdout")
    p_import.set_defaults(func=_cmd_import)

    p_redact = sub.add_parser(
        "redact",
        help="redact PII / secrets from a verified .sb trace",
        description=(
            "Verify the input .sb file, run the configured redaction "
            "policy over every step, and write a re-signed .sb trace "
            "to --output. Optionally writes a JSON manifest summarising "
            "what was redacted (counts only — no raw substrings)."
        ),
    )
    p_redact.add_argument("trace", help="input .sb trace path")
    p_redact.add_argument("-o", "--output", required=True,
                          help="output .sb trace path")
    p_redact.add_argument("--hmac-key-hex", required=True,
                          help="hex-encoded HMAC key the input was signed with")
    p_redact.add_argument("--policy", default="standard",
                          help="redaction policy name: standard | strict")
    p_redact.add_argument("--manifest",
                          help="path to write the JSON RedactionManifest")
    p_redact.add_argument("--no-compression", action="store_true",
                          help="disable per-frame zlib compression on output")
    p_redact.add_argument("--streaming", action="store_true",
                          help="use the streaming redactor for large traces")
    p_redact.add_argument("--json", action="store_true",
                          help="print the manifest as JSON on stdout")
    p_redact.set_defaults(func=_cmd_redact)

    p_scan = sub.add_parser(
        "redact-scan",
        help="dry-run redaction: scan a verified .sb and report findings",
        description=(
            "Run the redaction policy in scan-only mode: no output trace "
            "is written, but a JSON ScanReport is emitted summarising "
            "what *would* be redacted (counts + per-step findings)."
        ),
    )
    p_scan.add_argument("trace", help="input .sb trace path")
    p_scan.add_argument("--hmac-key-hex", required=True)
    p_scan.add_argument("--policy", default="standard",
                        help="redaction policy name: standard | strict")
    p_scan.add_argument("--json", action="store_true",
                        help="print the ScanReport as JSON on stdout")
    p_scan.set_defaults(func=_cmd_redact_scan)

    p_view = sub.add_parser(
        "view",
        help="render a self-contained interactive HTML viewer for a .sb trace",
        description=(
            "Read a recorded .sb trace and emit a single HTML file with "
            "an interactive timeline (filter by step kind, search, expand "
            "step details). The output has no external dependencies and "
            "can be opened directly in a browser."
        ),
    )
    p_view.add_argument("trace", help="input .sb trace path")
    p_view.add_argument(
        "--output", "-o", required=True,
        help="output HTML file path",
    )
    p_view.add_argument(
        "--title", default=None,
        help="title shown in the viewer header (default: derived from the trace filename)",
    )
    p_view.add_argument(
        "--hmac-key-hex", default=None,
        help="optional HMAC key (hex) used to verify the chain while loading",
    )
    p_view.add_argument(
        "--json", action="store_true",
        help="print a TraceViewSummary as JSON to stdout instead of a one-liner",
    )
    p_view.set_defaults(func=_cmd_view)

    # ------------------------------------------------------ debug (time-travel)
    p_debug = sub.add_parser(
        "debug",
        help="render a time-travel debugger HTML page for a trace",
        description=(
            "Read a recorded .sb trace, replay it (with optional substitutions), "
            "and produce a self-contained HTML page with step forward/back navigation, "
            "cache-hit display, canonical input diffs, and a causal graph view."
        ),
    )
    p_debug.add_argument("trace", help="input .sb trace path")
    p_debug.add_argument(
        "--output", "-o", required=True,
        help="output HTML file path",
    )
    p_debug.add_argument(
        "--title", default=None,
        help="title shown in the viewer header (default: derived from the trace filename)",
    )
    p_debug.add_argument(
        "--hmac-key-hex", default=None,
        help="optional HMAC key (hex) used to verify the chain while loading",
    )
    p_debug.add_argument(
        "--sub", action="append", default=[],
        metavar="SPEC",
        help=(
            "substitution spec applied before replay "
            "(can be repeated; same syntax as 'stepback replay --sub')"
        ),
    )
    p_debug.add_argument(
        "--json", action="store_true",
        help="print a TimeTravelSummary as JSON to stdout instead of a one-liner",
    )
    p_debug.set_defaults(func=_cmd_debug)

    # ------------------------------------------------------ bench
    p_bench = sub.add_parser(
        "bench",
        help="run reproducible micro-benchmarks (dirty-set, recorder overhead)",
        description=(
            "stepback bench REPLAY-CACHING|RECORD-OVERHEAD — reproduce the "
            "headline numbers from the README. Each subcommand prints a "
            "one-line summary and, with --out, writes a structured JSON "
            "BenchResult."
        ),
    )
    bench_sub = p_bench.add_subparsers(dest="bench_cmd", required=True)

    p_bench_rc = bench_sub.add_parser(
        "replay-caching",
        help="benchmark dirty-set size vs. trace length on synthetic traces",
    )
    p_bench_rc.add_argument("--n-steps", type=int, default=200,
                            help="approximate number of steps in the synthetic trace")
    p_bench_rc.add_argument("--n-trials", type=int, default=10,
                            help="independent trials to aggregate")
    p_bench_rc.add_argument(
        "--strategy",
        choices=["random_step", "first_quarter", "last_quarter",
                 "prompt_only", "tool_only"],
        default="random_step",
        help="how to pick the substitution target within each trial",
    )
    p_bench_rc.add_argument("--seed", type=int, default=0,
                            help="base RNG seed (default 0)")
    p_bench_rc.add_argument("--out", help="write structured JSON BenchResult to this path")
    p_bench_rc.set_defaults(func=_cmd_bench_replay_caching)

    p_bench_ro = bench_sub.add_parser(
        "record-overhead",
        help="benchmark per-LLM-call recorder overhead in microseconds",
    )
    p_bench_ro.add_argument("--n-steps", type=int, default=1000,
                            help="number of llm_call steps to time")
    p_bench_ro.add_argument("--out", help="write structured JSON RecordOverheadResult to this path")
    p_bench_ro.set_defaults(func=_cmd_bench_record_overhead)

    p_bench_soak = bench_sub.add_parser(
        "soak",
        help="record + replay a synthetic fleet (aggregates only)",
        description=(
            "Drive a fleet of N synthetic record/replay cycles and emit "
            "only aggregate statistics (counts, percentiles, rolling "
            "digest, error tallies). Designed to be safe to run on a "
            "schedule indefinitely without unbounded artifact growth."
        ),
    )
    p_bench_soak.add_argument("--n-traces", type=int, default=10_000,
                              help="number of synthetic traces in the fleet (default 10000)")
    p_bench_soak.add_argument("--n-steps", type=int, default=10,
                              help="approximate number of steps per trace (default 10)")
    p_bench_soak.add_argument("--seed", type=int, default=0,
                              help="base RNG seed (default 0)")
    p_bench_soak.add_argument("--no-substitution", action="store_true",
                              help="skip the per-trace substitution; replay is pure cache-hit")
    p_bench_soak.add_argument("--progress-every", type=int, default=0,
                              help="print progress every N traces (0 disables)")
    p_bench_soak.add_argument("--track-memory", action="store_true",
                              help="run under tracemalloc and report peak memory")
    p_bench_soak.add_argument("--out", help="write aggregate JSON SoakResult to this path")
    p_bench_soak.set_defaults(func=_cmd_bench_soak)

    p_bench_dsd = bench_sub.add_parser(
        "dirty-set-distributions",
        help="empirical dirty-set fraction distributions over multiple fixture corpora",
        description=(
            "Compute full dirty-set distributions — percentile tables and "
            "normalized histograms — across multiple synthetic trace corpora "
            "(linear_chain, parallel_wide, mixed_synthetic, agent_fixture) "
            "and substitution strategies.  Uses compute_dirty_set (pure "
            "analysis; no LLM/tool re-execution).  Step 66 of 100_STEPS.md."
        ),
    )
    p_bench_dsd.add_argument("--n-trials", type=int, default=20,
                             help="trials per (corpus, position, sub_kind) cell (default 20)")
    p_bench_dsd.add_argument("--n-steps", type=int, default=50,
                             help="target steps for linear/mixed corpora (default 50)")
    p_bench_dsd.add_argument("--seed", type=int, default=0,
                             help="base RNG seed (default 0)")
    p_bench_dsd.add_argument(
        "--corpora",
        default="",
        help=(
            "comma-separated subset of corpora to run "
            "(linear_chain,parallel_wide,mixed_synthetic,agent_fixture); "
            "default: all"
        ),
    )
    p_bench_dsd.add_argument("--out", help="write DistributionSuite JSON to this path")
    p_bench_dsd.set_defaults(func=_cmd_bench_dirty_set_distributions)

    p_bench_ms = bench_sub.add_parser(
        "model-swap",
        help="differential benchmark: replay with model A vs model B",
    )
    p_bench_ms.add_argument("--n-steps", type=int, default=20,
                            help="approximate number of steps per synthetic trace (default 20)")
    p_bench_ms.add_argument("--n-trials", type=int, default=5,
                            help="independent trials to aggregate (default 5)")
    p_bench_ms.add_argument("--seed", type=int, default=0,
                            help="base RNG seed (default 0)")
    p_bench_ms.add_argument("--model-a", default="bench-model-a-v1",
                            help="name tag for model A (default bench-model-a-v1)")
    p_bench_ms.add_argument("--model-b", default="bench-model-b-v1",
                            help="name tag for model B (default bench-model-b-v1)")
    p_bench_ms.add_argument("--out", help="write structured JSON ModelSwapResult to this path")
    p_bench_ms.set_defaults(func=_cmd_bench_model_swap)

    p_bench_sc = bench_sub.add_parser(
        "storage-compression",
        help="benchmark raw JSON vs .sb v1 vs CBOR vs zstd storage costs",
    )
    p_bench_sc.add_argument("--n-traces", type=int, default=20,
                            help="number of synthetic traces (default 20)")
    p_bench_sc.add_argument("--n-steps", type=int, default=30,
                            help="approximate number of steps per trace (default 30)")
    p_bench_sc.add_argument("--seed", type=int, default=0,
                            help="base RNG seed (default 0)")
    p_bench_sc.add_argument("--out", help="write structured JSON StorageCompressionResult to this path")
    p_bench_sc.set_defaults(func=_cmd_bench_storage_compression)

    p_bench_mb = bench_sub.add_parser(
        "microbenchmarks",
        help="canonicalization, frame write, HMAC/signing, and hash microbenchmarks",
    )
    p_bench_mb.add_argument("--n-iter", type=int, default=500,
                            help="timed iterations per operation (default 500)")
    p_bench_mb.add_argument("--out", help="write structured JSON MicroBenchSuite to this path")
    p_bench_mb.set_defaults(func=_cmd_bench_microbenchmarks)

    p_bench_lb = bench_sub.add_parser(
        "leaderboard",
        help="build a ranked leaderboard from signed JSON submissions",
    )
    p_bench_lb.add_argument(
        "submissions", nargs="+",
        help="paths to submission JSON files or directories containing them",
    )
    p_bench_lb.add_argument("--out", help="write leaderboard JSON to this path")
    p_bench_lb.add_argument("--html", help="write leaderboard HTML to this path")
    p_bench_lb.set_defaults(func=_cmd_bench_leaderboard)

    p_bench_fr = bench_sub.add_parser(
        "frontier-reeval-schedule",
        help="compute which submissions need re-evaluation against frontier models",
    )
    p_bench_fr.add_argument(
        "--submissions", nargs="*", default=[],
        help="paths to submission JSON files or directories (default: none → empty schedule)",
    )
    p_bench_fr.add_argument("--catalog", default=None,
                            help="path to frontier model catalog JSON (default: bundled)")
    p_bench_fr.add_argument("--models-file", default=None,
                            help="JSON mapping submission_id → [model_id, ...] used by each")
    p_bench_fr.add_argument("--max-age-days", type=int, default=180,
                            help="submissions older than this need re-evaluation (default 180)")
    p_bench_fr.add_argument("--reference-date", default=None,
                            help="YYYY-MM-DD reference date for age calculation (default: today)")
    p_bench_fr.add_argument("--no-superseded", action="store_true",
                            help="disable superseded-model check")
    p_bench_fr.add_argument("--no-new-frontier", action="store_true",
                            help="disable new-frontier-model check")
    p_bench_fr.add_argument("--out", help="write schedule JSON to this path")
    p_bench_fr.set_defaults(func=_cmd_bench_frontier_reeval_schedule)

    p_bench_vs = bench_sub.add_parser(
        "validate-submission",
        help="validate a submission JSON manifest against MLPerf-style submission rules",
    )
    p_bench_vs.add_argument(
        "manifest",
        help="path to the submission JSON file to validate",
    )
    p_bench_vs.add_argument("--out", help="write SubmissionValidationResult JSON to this path")
    p_bench_vs.set_defaults(func=_cmd_bench_validate_submission)

    p_bench_mn = bench_sub.add_parser(
        "minimization",
        help="minimization benchmark: trace size, predicate stability, LLM calls vs naive ddmin",
    )
    p_bench_mn.add_argument("--n-steps", type=int, default=20,
                            help="approximate number of steps per synthetic trace (default 20)")
    p_bench_mn.add_argument("--n-noisy", type=int, default=5,
                            help="number of noisy substitutions per trial (default 5)")
    p_bench_mn.add_argument("--n-trials", type=int, default=5,
                            help="independent trials to aggregate (default 5)")
    p_bench_mn.add_argument("--seed", type=int, default=0,
                            help="base RNG seed (default 0)")
    p_bench_mn.add_argument("--out", help="write structured JSON MinimizationBenchResult to this path")
    p_bench_mn.set_defaults(func=_cmd_bench_minimization)

    # --- spec ---------------------------------------------------------- #
    p_spec = sub.add_parser(
        "spec",
        help="SB-Trace specification tooling (conformance runner)",
        description=(
            "Group of subcommands that operate on the SB-Trace v1 "
            "specification — currently the conformance runner that "
            "drives an external implementation through the bundled "
            "fixture corpus."
        ),
    )
    spec_sub = p_spec.add_subparsers(dest="spec_cmd", required=True)
    p_spec_test = spec_sub.add_parser(
        "test",
        help="run the SB-Trace conformance corpus against IMPL",
        description=(
            "Drive an external SB-Trace implementation through the "
            "bundled v1 conformance corpus (good fixtures must verify, "
            "corrupt fixtures must reject) and print a pass/fail "
            "summary. Exit code is 0 iff every selected fixture passed."
        ),
    )
    p_spec_test.add_argument(
        "--manifest", default=None,
        help="override path to manifest.json (defaults to bundled v1 manifest)",
    )
    p_spec_test.add_argument(
        "--fixtures", default=None,
        help="override fixtures directory (defaults to manifest's parent dir)",
    )
    p_spec_test.add_argument(
        "--timeout", type=float, default=30.0,
        help="per-invocation subprocess timeout in seconds (default 30)",
    )
    p_spec_test.add_argument(
        "--no-hash", action="store_true",
        help="skip the optional 'hash' subcommand probe on good fixtures",
    )
    p_spec_test.add_argument(
        "--only", action="append", default=None,
        help="restrict to fixtures with this name (repeatable)",
    )
    p_spec_test.add_argument(
        "--json", action="store_true",
        help="emit the structured ConformanceRun JSON instead of text",
    )
    p_spec_test.add_argument(
        "impl_argv", nargs=argparse.REMAINDER,
        help="argv of the implementation to drive (e.g. ./sb-verifier or 'docker run ghcr.io/...')",
    )
    p_spec_test.set_defaults(func=_cmd_spec_test)

    p_proxy = sub.add_parser(
        "proxy",
        help="run the stepback HTTP (and optional gRPC) proxy",
        description=(
            "Run a sidecar that exposes StartTrace/RecordStep/EndTrace/"
            "VerifyTrace over JSON-over-HTTP. Optional gRPC is enabled "
            "with --grpc-listen if 'grpcio' is installed. See "
            "stepback/proxy/server.py for the HTTP wire schema."
        ),
    )
    p_proxy.add_argument(
        "--listen", default="127.0.0.1:4319",
        help="HOST:PORT for the HTTP listener (default 127.0.0.1:4319)",
    )
    p_proxy.add_argument(
        "--write", required=True,
        help="directory under which new .sb traces are created",
    )
    p_proxy.add_argument(
        "--max-body-bytes", type=int, default=64 * 1024 * 1024,
        help="hard cap on request body size in bytes (default 64 MiB)",
    )
    p_proxy.add_argument(
        "--max-open-traces", type=int, default=1024,
        help="refuse StartTrace once this many traces are simultaneously open",
    )
    p_proxy.add_argument(
        "--grpc-listen", default=None,
        help="optional HOST:PORT for the gRPC listener (requires grpcio)",
    )
    p_proxy.add_argument(
        "--grpc-max-workers", type=int, default=8,
        help="grpc thread-pool size (default 8)",
    )
    p_proxy.set_defaults(func=_cmd_proxy)

    # ------------------------------------------------------------------ diagnose
    p_diag = sub.add_parser(
        "diagnose",
        help="Inspect installed SDK/framework versions against the certified matrix.",
    )
    p_diag.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        default=False,
        help="Emit results as a JSON array instead of a table.",
    )
    p_diag.add_argument(
        "--all",
        action="store_true",
        dest="show_all",
        default=False,
        help="Include packages that are not installed in the output.",
    )
    p_diag.add_argument(
        "--strict",
        action="store_true",
        dest="strict",
        default=False,
        help="Exit 1 if any package has a warning status (warn_newer or warn_unsupported).",
    )
    p_diag.set_defaults(func=_cmd_diagnose)

    # ------------------------------------------------------------------ doctor
    p_doctor = sub.add_parser(
        "doctor",
        help="run environment health checks and print a PASS/WARN/FAIL table",
        description=(
            "Check Python version, optional Rust/WASM components, HMAC key "
            "material, writable trace directory, and (with --network) TCP "
            "reachability of configured LLM providers.  Exits 1 if any check "
            "is FAIL."
        ),
    )
    p_doctor.add_argument(
        "--network",
        action="store_true",
        default=False,
        help="Also check TCP reachability of known LLM provider hosts (makes outbound connections).",
    )
    p_doctor.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        default=False,
        help="Emit results as a JSON array instead of a table.",
    )
    p_doctor.set_defaults(func=_cmd_doctor)

    # ------------------------------------------------------------------ init
    p_init = sub.add_parser(
        "init",
        help="scaffold a stepback.toml, quickstart example, and demo trace",
        description=(
            "Initialise a new stepback project in DIRECTORY (default: current "
            "working directory). Creates stepback.toml, examples/quickstart/agent.py, "
            "and records a deterministic demo trace so you can immediately run "
            "`stepback replay examples/quickstart/traces/quickstart.sb`."
        ),
    )
    p_init.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Target directory (default: current working directory).",
    )
    p_init.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing stepback.toml and agent.py if present.",
    )
    p_init.set_defaults(func=_cmd_init)

    # -------------------------------------------------------------- quickstart
    p_qs = sub.add_parser(
        "quickstart",
        help="interactive first-run wizard: pick a provider, store API key, record a trace",
        description=(
            "Walk through picking a provider shim, storing your API key safely "
            "in the OS keyring (never on disk or in the repo), recording one "
            "demo trace, and opening it in the HTML viewer.  "
            "Use --non-interactive for scripted / CI use."
        ),
    )
    p_qs.add_argument(
        "--provider",
        choices=["openai", "anthropic", "bedrock", "gemini", "demo"],
        default=None,
        help="Provider to use (skips the interactive picker).",
    )
    p_qs.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="API key for the chosen provider (skips the interactive key prompt).",
    )
    p_qs.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Directory for the recorded trace and HTML file (default: a temp dir).",
    )
    p_qs.add_argument(
        "--non-interactive",
        dest="non_interactive",
        action="store_true",
        default=False,
        help="Skip all prompts; use --provider (default: demo) and --api-key as given.",
    )
    p_qs.add_argument(
        "--skip-open",
        dest="skip_open",
        action="store_true",
        default=False,
        help="Do not open the HTML viewer in the browser.",
    )
    p_qs.set_defaults(func=_cmd_quickstart)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
