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
    dump_report_json,
    render_counterfactual_report,
    render_replay_report,
)
from .substitutions import SubstitutionSet
from .trace_reader import verify_trace


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

    if not subs_b.items and not args.branch:
        # Single-replay report: just the recorded run.
        result = t.run_replay(subs_a, executor)
        if args.format == "json":
            out_text = dump_report_json(t, result, None, subs_a, options=options) + "\n"
        else:
            out_text = render_replay_report(t, result, subs_a, options=options)
    else:
        baseline = t.run_replay(subs_a, executor)
        counterfactual = t.run_replay(subs_b, executor)
        if args.format == "json":
            out_text = dump_report_json(
                t, baseline, counterfactual, subs_b, options=options
            ) + "\n"
        else:
            out_text = render_counterfactual_report(
                t, baseline, counterfactual, subs_b, options=options
            )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
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


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(prog="stepback", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

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
        "--format", choices=["md", "json"], default="md",
        help="output format: 'md' (default) renders Markdown, 'json' "
             "emits the structured report model (schema_version, verdict, "
             "first_divergence_step_id, causal_attribution, ...)",
    )
    p_report.set_defaults(func=_cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
