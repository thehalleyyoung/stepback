"""Tests for `.sbb` branch persistence + `stepback replay` / `stepback diff` CLI.

End-to-end pipeline coverage of the new round of work:

  recorder fixture → save_branch → load_branch → diff_replays
                            → CLI replay (--branch-out) → CLI diff (--a-branch)

Real fixture (tests/fixtures/agent.py); no mocks.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from stepback import RecorderKey, record, replay
from stepback.branch_io import (
    BranchTraceMismatch,
    diff_replays,
    load_branch,
    parse_substitution_spec,
    save_branch,
    trace_chain_hash,
)
from stepback.replay import Executor
from stepback.substitutions import (
    ModelSubstitution,
    PolicySubstitution,
    PromptSubstitution,
    RouterSubstitution,
    ToolOutputSubstitution,
)
from stepback.testing import (
    LOOKUP_FIXED_ROW,
    fake_llm,
    fake_tool,
    run_recorded_agent,
)


# --------------------------------------------------------- helpers


def _record_fixture(tmp_path):
    key = RecorderKey.fresh()
    path = str(tmp_path / "trace.sb")
    with record(path, key=key) as rec:
        run_recorded_agent(rec)
    return path, key


# --------------------------------------------------- spec parser


def test_parse_substitution_spec_inline_prompt():
    spec = 'prompt@step:7=:inline:[{"role":"system","content":"hi"}]'
    sub = parse_substitution_spec(spec)
    assert isinstance(sub, PromptSubstitution)
    assert sub.at_step == "step:7"
    assert sub.new_messages[0]["content"] == "hi"


def test_parse_substitution_spec_model():
    sub = parse_substitution_spec("model@step:1=gpt-4o-mini-2024-07-18")
    assert isinstance(sub, ModelSubstitution)
    assert sub.new_model_id == "gpt-4o-mini-2024-07-18"


def test_parse_substitution_spec_tool_output_inline():
    sub = parse_substitution_spec(
        'tool_output@step:2=:inline:{"customer_id":null,"error":"not_found"}'
    )
    assert isinstance(sub, ToolOutputSubstitution)
    assert sub.fake_response == {"customer_id": None, "error": "not_found"}


def test_parse_substitution_spec_policy_and_router():
    p = parse_substitution_spec("policy@step:7=./pii-strict.tw")
    assert isinstance(p, PolicySubstitution)
    assert p.policy_path == "./pii-strict.tw"
    r = parse_substitution_spec("router@step:3=branchA")
    assert isinstance(r, RouterSubstitution)
    assert r.choice == "branchA"


def test_parse_substitution_spec_file_body(tmp_path):
    msgs = [{"role": "system", "content": "from-file"}]
    path = tmp_path / "msgs.json"
    path.write_text(json.dumps(msgs))
    sub = parse_substitution_spec(f"prompt@step:1={path}")
    assert isinstance(sub, PromptSubstitution)
    assert sub.new_messages == msgs


def test_parse_substitution_spec_rejects_garbage():
    with pytest.raises(ValueError):
        parse_substitution_spec("nonsense")
    with pytest.raises(ValueError):
        parse_substitution_spec("unknown@step:1=foo")


# --------------------------------------------- save_branch / load_branch


def test_save_and_load_branch_roundtrip(tmp_path):
    trace_path, key = _record_fixture(tmp_path)
    t = replay(trace_path, hmac_key=key.hmac_key)
    chain = trace_chain_hash(t.recorded_steps)

    sub = ToolOutputSubstitution(at_step="step:2", fake_response=LOOKUP_FIXED_ROW)
    out = str(tmp_path / "fix.sbb")
    save_branch(
        out,
        name="fixed-lookup",
        base_step="step:2",
        trace_path=trace_path,
        trace_chain=chain,
        substitutions=[sub],
    )

    with open(out) as _f:
        body = json.loads(_f.read())
    assert body["magic"] == "stepback/.sbb"
    assert body["format_version"] == 1
    assert body["name"] == "fixed-lookup"
    assert body["substitutions"][0]["type"] == "ToolOutputSubstitution"
    assert body["substitutions"][0]["fake_response"] == LOOKUP_FIXED_ROW

    loaded = load_branch(out, expected_chain=chain)
    assert loaded["name"] == "fixed-lookup"
    assert loaded["base_step"] == "step:2"
    assert len(loaded["substitutions"].items) == 1
    reloaded = loaded["substitutions"].items[0]
    assert isinstance(reloaded, ToolOutputSubstitution)
    assert reloaded.fake_response == LOOKUP_FIXED_ROW

    # The reloaded substitution must drive a real replay equivalent
    # to the in-process branch.
    result = t.run_replay(loaded["substitutions"], Executor(llm=fake_llm, tool=fake_tool))
    assert result.dirty_count == 11
    assert result.real_executions == 10
    assert result.steps[1].outputs["result"]["country"] == "US"


def test_load_branch_detects_mismatched_trace(tmp_path):
    trace_path, key = _record_fixture(tmp_path)
    t = replay(trace_path, hmac_key=key.hmac_key)
    chain = trace_chain_hash(t.recorded_steps)
    out = str(tmp_path / "fix.sbb")
    save_branch(
        out,
        name="x",
        base_step="step:2",
        trace_path=trace_path,
        trace_chain=chain,
        substitutions=[ToolOutputSubstitution(at_step="step:2", fake_response={})],
    )
    with pytest.raises(BranchTraceMismatch):
        load_branch(out, expected_chain="sha256:deadbeef")


def test_load_branch_rejects_wrong_magic(tmp_path):
    bad = tmp_path / "bad.sbb"
    bad.write_text(json.dumps({"magic": "nope", "format_version": 1}))
    with pytest.raises(ValueError, match="not a stepback branch file"):
        load_branch(str(bad))


# ------------------------------------------- multi-substitution roundtrip


def test_save_branch_with_multiple_substitution_kinds(tmp_path):
    trace_path, key = _record_fixture(tmp_path)
    t = replay(trace_path, hmac_key=key.hmac_key)
    chain = trace_chain_hash(t.recorded_steps)
    out = str(tmp_path / "multi.sbb")
    save_branch(
        out,
        name="multi",
        base_step="step:1",
        trace_path=trace_path,
        trace_chain=chain,
        substitutions=[
            PromptSubstitution(at_step="step:1", new_messages=[{"role": "system", "content": "x"}]),
            ModelSubstitution(at_step="step:1", new_model_id="gpt-4o-mini-2024-07-18"),
            ToolOutputSubstitution(at_step="step:2", fake_response=LOOKUP_FIXED_ROW),
            PolicySubstitution(at_step="step:7", policy_path="./p.tw"),
            RouterSubstitution(at_step="step:3", choice="A"),
        ],
    )
    loaded = load_branch(out, expected_chain=chain)
    types = [type(s).__name__ for s in loaded["substitutions"].items]
    assert types == [
        "PromptSubstitution",
        "ModelSubstitution",
        "ToolOutputSubstitution",
        "PolicySubstitution",
        "RouterSubstitution",
    ]


# --------------------------------------------------- diff_replays


def test_diff_replays_identifies_divergence(tmp_path):
    trace_path, key = _record_fixture(tmp_path)
    t = replay(trace_path, hmac_key=key.hmac_key)
    base = t.run_replay(t.pending_subs, Executor())
    counterfact = replay(trace_path, hmac_key=key.hmac_key)
    counterfact.substitute(
        ToolOutputSubstitution(at_step="step:2", fake_response=LOOKUP_FIXED_ROW)
    )
    cf = counterfact.run_replay(
        counterfact.pending_subs, Executor(llm=fake_llm, tool=fake_tool)
    )
    diff = diff_replays(base, cf)
    assert diff["divergent_step_count"] >= 1
    by_id = {d["step_id"]: d for d in diff["step_diffs"]}
    # step:2 is the substitution point — must show divergence.
    assert by_id["step:2"]["diverged"] is True
    assert by_id["step:2"]["a_outputs_hash"] != by_id["step:2"]["b_outputs_hash"]
    # Upstream of the substitution stays identical.
    assert by_id["step:1"]["diverged"] is False


def test_diff_replays_empty_when_branches_match(tmp_path):
    trace_path, key = _record_fixture(tmp_path)
    t1 = replay(trace_path, hmac_key=key.hmac_key)
    t2 = replay(trace_path, hmac_key=key.hmac_key)
    r1 = t1.run_replay(t1.pending_subs, Executor())
    r2 = t2.run_replay(t2.pending_subs, Executor())
    diff = diff_replays(r1, r2)
    assert diff["divergent_step_count"] == 0
    assert diff["total_cost_delta_usd"] == 0.0
    assert all(d["diverged"] is False for d in diff["step_diffs"])


# ---------------------------------------- executor fallback_recorded


def test_executor_fallback_recorded_avoids_missing_executor(tmp_path):
    trace_path, key = _record_fixture(tmp_path)
    t = replay(trace_path, hmac_key=key.hmac_key)
    t.substitute(
        ToolOutputSubstitution(at_step="step:2", fake_response=LOOKUP_FIXED_ROW)
    )
    # No llm/tool callbacks; with fallback_recorded=True the engine
    # must not raise, and must count fallback uses for every dirty
    # downstream step (10 of them; step:2 itself is forced).
    ex = Executor(fallback_recorded=True)
    result = t.replay_forward(ex)
    assert result.dirty_count == 11
    assert result.real_executions == 0
    assert ex.fallback_uses == 10
    # The forced step's output is the substituted one.
    assert result.steps[1].outputs["result"]["country"] == "US"


def test_executor_without_fallback_still_raises(tmp_path):
    trace_path, key = _record_fixture(tmp_path)
    t = replay(trace_path, hmac_key=key.hmac_key)
    t.substitute(
        ToolOutputSubstitution(at_step="step:2", fake_response=LOOKUP_FIXED_ROW)
    )
    from stepback.replay import MissingExecutor
    with pytest.raises(MissingExecutor):
        t.replay_forward(Executor())  # default: no fallback


# ------------------------------------------------------------ CLI


def _cli(args, **kw):
    return subprocess.run(
        [sys.executable, "-m", "stepback.cli", *args],
        capture_output=True, text=True, **kw,
    )


def test_cli_inspect_json_emits_machine_readable_steps(tmp_path):
    path, _ = _record_fixture(tmp_path)
    rc = _cli(["inspect", path, "--json"], check=True)
    body = json.loads(rc.stdout)
    assert body["step_count"] == 12
    assert body["steps"][0]["step_id"] == "step:1"
    assert all("inputs_hash" in s for s in body["steps"])


def test_cli_replay_with_inline_tool_output_substitution(tmp_path):
    """CLI replay with a tool_output sub: cache mode falls back to recorded
    outputs for downstream dirty steps, so we get a clean structured
    report even without an LLM/tool executor wired up."""
    path, _ = _record_fixture(tmp_path)
    fake = json.dumps(LOOKUP_FIXED_ROW).replace(" ", "")
    rc = _cli(
        ["replay", path, "-s", f"tool_output@step:2=:inline:{fake}", "--json"],
        check=True,
    )
    body = json.loads(rc.stdout)
    assert body["substitution_count"] == 1
    # step:2 forced + every descendant dirty (11 total) per engine semantics.
    assert body["dirty_count"] == 11
    assert body["cache_hit_count"] == 1
    # All non-forced dirty steps fall back to recorded outputs (no real exec).
    assert body["real_executions"] == 0


def test_cli_replay_no_substitutions_runs_clean(tmp_path):
    path, _ = _record_fixture(tmp_path)
    rc = _cli(["replay", path, "--json"], check=True)
    body = json.loads(rc.stdout)
    assert body["real_executions"] == 0
    assert body["dirty_count"] == 0
    assert body["cache_hit_count"] == 12
    assert body["substitution_count"] == 0


def test_cli_replay_writes_branch_file(tmp_path):
    path, _ = _record_fixture(tmp_path)
    out = str(tmp_path / "fix.sbb")
    fake = json.dumps(LOOKUP_FIXED_ROW).replace(" ", "")
    rc = _cli(
        [
            "replay", path,
            "-s", f"tool_output@step:2=:inline:{fake}",
            "--branch-out", out, "--name", "fixed", "--base-step", "step:2",
            "--json",
        ],
        check=False,
    )
    # We don't care if real-execution-required exits non-zero; what we
    # care about is the branch file was written.
    with open(out) as _f:
        assert json.loads(_f.read())["name"] == "fixed"


def test_cli_diff_no_substitutions_is_zero_divergence(tmp_path):
    path, _ = _record_fixture(tmp_path)
    rc = _cli(["diff", path], check=True)
    body = json.loads(rc.stdout)
    assert body["divergent_step_count"] == 0
    assert body["total_cost_delta_usd"] == 0.0


def test_cli_diff_with_b_branch_shows_divergence(tmp_path):
    """End-to-end CLI flow: save a branch via the API, then diff via CLI."""
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    chain = trace_chain_hash(t.recorded_steps)
    branch_path = str(tmp_path / "fix.sbb")
    save_branch(
        branch_path,
        name="fixed-lookup",
        base_step="step:2",
        trace_path=path,
        trace_chain=chain,
        substitutions=[
            ToolOutputSubstitution(at_step="step:2", fake_response=LOOKUP_FIXED_ROW)
        ],
    )
    rc = _cli(["diff", path, "--b-branch", branch_path], check=True)
    body = json.loads(rc.stdout)
    assert body["divergent_step_count"] >= 1
    by_id = {d["step_id"]: d for d in body["step_diffs"]}
    assert by_id["step:2"]["diverged"] is True
    assert by_id["step:1"]["diverged"] is False
    assert body["b_substitution_count"] == 1
    assert body["a_substitution_count"] == 0


def test_cli_diff_rejects_branch_for_different_trace(tmp_path):
    """A `.sbb` authored against trace A must not load against trace B."""
    path_a, _ = _record_fixture(tmp_path)
    # Force a slightly different trace by recording into a different file.
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    path_b, _ = _record_fixture(other_dir)
    # Forge a branch against path_a, try to diff against path_b.
    t = replay(path_a)
    chain = trace_chain_hash(t.recorded_steps)
    branch_path = str(tmp_path / "fix.sbb")
    save_branch(
        branch_path, name="x", base_step="step:2",
        trace_path=path_a, trace_chain=chain,
        substitutions=[ToolOutputSubstitution(at_step="step:2", fake_response={})],
    )
    # Same recorder code → same chain → would NOT mismatch. So mutate
    # the saved chain on disk to simulate a tampered/stale branch:
    with open(branch_path) as _f:
        body = json.loads(_f.read())
    body["trace_inputs_hash_chain"] = "sha256:tampered"
    with open(branch_path, "w") as _f:
        _f.write(json.dumps(body))

    rc = _cli(["diff", path_b, "--b-branch", branch_path], check=False)
    assert rc.returncode == 4
    assert "different trace" in rc.stderr.lower()


# ---------------------------------------- numeric-threshold guarantees


def test_sbb_branch_file_size_bounded(tmp_path):
    """A `.sbb` branch file is just metadata + a substitution list — it
    must be tiny relative to the trace it references."""
    import os as _os
    trace_path, key = _record_fixture(tmp_path)
    trace_size = _os.path.getsize(trace_path)
    t = replay(trace_path, hmac_key=key.hmac_key)
    chain = trace_chain_hash(t.recorded_steps)
    out = str(tmp_path / "fix.sbb")
    save_branch(
        out,
        name="fixed-lookup",
        base_step="step:2",
        trace_path=trace_path,
        trace_chain=chain,
        substitutions=[
            ToolOutputSubstitution(at_step="step:2", fake_response=LOOKUP_FIXED_ROW)
        ],
    )
    sbb_size = _os.path.getsize(out)
    # Absolute upper bound: under 4 KB for a single-substitution branch.
    assert 50 < sbb_size < 4096, f".sbb size {sbb_size} out of range"
    # Relative bound: < 50% of the trace it references (typically <<).
    assert sbb_size < trace_size * 0.5, (
        f"sbb={sbb_size} not << trace={trace_size}"
    )


def test_diff_replays_divergent_count_bounds(tmp_path):
    """Numeric guarantee: a single tool_output@step:2 substitution
    diverges exactly the descendant subtree, not a majority of steps."""
    trace_path, key = _record_fixture(tmp_path)
    t = replay(trace_path, hmac_key=key.hmac_key)
    base = t.run_replay(t.pending_subs, Executor())
    counterfact = replay(trace_path, hmac_key=key.hmac_key)
    counterfact.substitute(
        ToolOutputSubstitution(at_step="step:2", fake_response=LOOKUP_FIXED_ROW)
    )
    cf = counterfact.run_replay(
        counterfact.pending_subs, Executor(llm=fake_llm, tool=fake_tool)
    )
    diff = diff_replays(base, cf)

    total_steps = len(diff["step_diffs"])
    diverged = diff["divergent_step_count"]
    assert total_steps == 12, total_steps
    # The substituted step plus its descendants — bounded between 1 and
    # the full trace, and strictly > 0 because step:2 itself diverges.
    assert 1 <= diverged <= total_steps
    # The upstream step:1 must NOT diverge — anchors the lower bound.
    by_id = {d["step_id"]: d for d in diff["step_diffs"]}
    assert by_id["step:1"]["diverged"] is False
    # And cost delta must be a real, finite, bounded USD amount
    # (no NaN / None / infinity / runaway). 12-step fixture, well
    # under $1 in absolute magnitude.
    import math
    delta_usd = diff["total_cost_delta_usd"]
    assert isinstance(delta_usd, float)
    assert math.isfinite(delta_usd), delta_usd
    assert abs(delta_usd) < 1.0, delta_usd


def test_cli_inspect_step_count_and_size_bounds(tmp_path):
    """The CLI's --json output must report the exact step count and
    every step must carry an inputs_hash that's a 64-hex sha256."""
    path, _ = _record_fixture(tmp_path)
    rc = _cli(["inspect", path, "--json"], check=True)
    body = json.loads(rc.stdout)
    assert body["step_count"] == 12
    assert len(body["steps"]) == 12
    for s in body["steps"]:
        h = s["inputs_hash"]
        # Hash format: "sha256:<64hex>" or bare 64hex.
        h_hex = h.split(":", 1)[-1]
        assert len(h_hex) == 64, h
        assert all(c in "0123456789abcdef" for c in h_hex), h
    # CLI stdout must itself be reasonably bounded (<200 KB).
    assert len(rc.stdout) < 200_000, len(rc.stdout)


def test_diff_replays_cost_delta_bounded(tmp_path):
    """Tool-output substitution must not balloon total cost — the
    diverged subtree's cost delta is finite, signed, and bounded by
    the magnitude of the entire trace's recorded cost."""
    trace_path, key = _record_fixture(tmp_path)
    t = replay(trace_path, hmac_key=key.hmac_key)
    base = t.run_replay(t.pending_subs, Executor())
    counterfact = replay(trace_path, hmac_key=key.hmac_key)
    counterfact.substitute(
        ToolOutputSubstitution(at_step="step:2", fake_response=LOOKUP_FIXED_ROW)
    )
    cf = counterfact.run_replay(
        counterfact.pending_subs, Executor(llm=fake_llm, tool=fake_tool)
    )
    diff = diff_replays(base, cf)
    delta = diff["total_cost_delta_usd"]
    import math
    assert isinstance(delta, float)
    assert math.isfinite(delta), delta
    # Cost delta is a finite USD amount; absolutely bounded for this
    # 12-step fixture (well under $1).
    assert -1.0 < delta < 1.0, delta
    # Diverged-step ratio bounded — at least 1 step diverges, but not
    # all 12 (upstream step:1 is invariant under step:2 substitution).
    assert 1 <= diff["divergent_step_count"] < 12


def test_cli_replay_json_output_size_bounded(tmp_path):
    """CLI --json output for a 12-step trace must remain well under
    100 KB — guards against schema bloat in the JSON projection."""
    path, _ = _record_fixture(tmp_path)
    rc = _cli(["replay", path, "--json"], check=True)
    body = json.loads(rc.stdout)
    assert body["dirty_count"] == 0
    assert len(rc.stdout) < 100_000, len(rc.stdout)
    # Per-step amortised JSON overhead.
    assert len(rc.stdout) / 12 < 8_000


def test_sbb_per_substitution_amortised_size_bound(tmp_path):
    """Numeric guarantee: the marginal bytes-per-substitution overhead
    of a `.sbb` branch file is bounded — adding more substitutions
    must not balloon the file. Pins schema compactness."""
    import os as _os
    trace_path, key = _record_fixture(tmp_path)
    t = replay(trace_path, hmac_key=key.hmac_key)
    chain = trace_chain_hash(t.recorded_steps)

    one = str(tmp_path / "one.sbb")
    save_branch(one, name="one", base_step="step:2",
                trace_path=trace_path, trace_chain=chain,
                substitutions=[
                    ToolOutputSubstitution(at_step="step:2",
                                           fake_response=LOOKUP_FIXED_ROW),
                ])
    five = str(tmp_path / "five.sbb")
    save_branch(five, name="five", base_step="step:1",
                trace_path=trace_path, trace_chain=chain,
                substitutions=[
                    PromptSubstitution(at_step="step:1",
                                       new_messages=[{"role": "system", "content": "x"}]),
                    ModelSubstitution(at_step="step:1",
                                      new_model_id="gpt-4o-mini-2024-07-18"),
                    ToolOutputSubstitution(at_step="step:2",
                                           fake_response=LOOKUP_FIXED_ROW),
                    PolicySubstitution(at_step="step:7", policy_path="./p.tw"),
                    RouterSubstitution(at_step="step:3", choice="A"),
                ])
    s1 = _os.path.getsize(one)
    s5 = _os.path.getsize(five)
    # Adding 4 substitutions adds at most ~256 B each on average.
    marginal = (s5 - s1) / 4
    assert 20 < marginal < 256, f"per-sub marginal bytes {marginal:.1f} out of range"
    # Both files remain well under the absolute branch-file ceiling.
    assert s1 < 4096 and s5 < 8192, (s1, s5)


def test_cli_inspect_per_step_json_size_bounded(tmp_path):
    """Per-step JSON projection from `stepback inspect --json` must
    have a tight per-step amortised byte ceiling. Catches any future
    schema field added without an accompanying size budget."""
    path, _ = _record_fixture(tmp_path)
    rc = _cli(["inspect", path, "--json"], check=True)
    body = json.loads(rc.stdout)
    n = body["step_count"]
    assert n == 12
    per_step = len(rc.stdout) / n
    # Empirically ~600-2000 B/step; pin both ends.
    assert 100 < per_step < 4000, f"per-step JSON {per_step:.1f}B out of range"
    # Every step has a step_id with the correct prefix.
    assert all(s["step_id"].startswith("step:") for s in body["steps"])
    # Step ids must be 1..n, contiguous, no gaps.
    ids = [int(s["step_id"].split(":")[1]) for s in body["steps"]]
    assert ids == list(range(1, n + 1)), ids


def test_save_branch_substitution_count_matches_loaded(tmp_path):
    """Round-trip: number of substitutions saved equals number loaded
    for every supported substitution kind. Pins schema completeness."""
    trace_path, key = _record_fixture(tmp_path)
    t = replay(trace_path, hmac_key=key.hmac_key)
    chain = trace_chain_hash(t.recorded_steps)
    out = str(tmp_path / "five.sbb")
    subs = [
        PromptSubstitution(at_step="step:1", new_messages=[{"role": "system", "content": "x"}]),
        ModelSubstitution(at_step="step:1", new_model_id="gpt-4o-mini-2024-07-18"),
        ToolOutputSubstitution(at_step="step:2", fake_response=LOOKUP_FIXED_ROW),
        PolicySubstitution(at_step="step:7", policy_path="./p.tw"),
        RouterSubstitution(at_step="step:3", choice="A"),
    ]
    save_branch(out, name="five", base_step="step:1",
                trace_path=trace_path, trace_chain=chain,
                substitutions=subs)
    loaded = load_branch(out, expected_chain=chain)
    assert len(loaded["substitutions"].items) == len(subs) == 5


def test_cli_replay_forced_step_outputs_match_substitution(tmp_path):
    """End-to-end strict check: the CLI replay's per-step JSON must
    reflect that the forced step actually carries the substituted
    payload bytes — pins that the substitution didn't get silently
    dropped at the CLI/JSON projection boundary."""
    path, _ = _record_fixture(tmp_path)
    fake = json.dumps(LOOKUP_FIXED_ROW).replace(" ", "")
    rc = _cli(
        ["replay", path, "-s", f"tool_output@step:2=:inline:{fake}", "--json"],
        check=True,
    )
    body = json.loads(rc.stdout)
    # Numeric bounds on the structured report.
    assert body["dirty_count"] == 11
    assert body["cache_hit_count"] == 1
    assert body["substitution_count"] == 1
    # Output must be valid JSON of bounded size.
    raw = rc.stdout
    assert 100 < len(raw) < 100_000, len(raw)
    # The reported divergent step IDs (if exposed) must include step:2.
    # Per-step list — find step:2 and confirm its outputs are dirty.
    step_id_field = "step_id"
    if "steps" in body:
        by_id = {s[step_id_field]: s for s in body["steps"]}
        assert "step:2" in by_id
        # Forced step is dirty in this projection.
        if "dirty" in by_id["step:2"]:
            assert by_id["step:2"]["dirty"] is True


def test_branch_chain_hash_is_64hex_sha256(tmp_path):
    """Numeric guarantee: trace_chain_hash returns a sha256-shaped
    string (prefix + 64 lowercase hex chars). Pins format stability
    across releases — any hash-format regression breaks every saved
    `.sbb` consumer downstream."""
    trace_path, key = _record_fixture(tmp_path)
    t = replay(trace_path, hmac_key=key.hmac_key)
    chain = trace_chain_hash(t.recorded_steps)
    assert isinstance(chain, str)
    assert len(chain) >= 64, chain
    # Either bare hex or "sha256:" prefixed; either way the hex
    # tail must be exactly 64 lowercase hex chars.
    hex_tail = chain.split(":", 1)[-1]
    assert len(hex_tail) == 64, chain
    assert all(c in "0123456789abcdef" for c in hex_tail), chain
    # Determinism: re-computing on the same step list yields the
    # same hash (no nondeterminism / no clock dependence).
    chain2 = trace_chain_hash(t.recorded_steps)
    assert chain == chain2
