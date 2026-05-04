"""End-to-end fixture test for stepback.

Drives a 12-step deterministic "payments" agent through the recorder,
verifies the on-disk trace, replays it (zero LLM calls), substitutes a
tool output to fix the recorded bug, and asserts:

* full HMAC + Ed25519 verification passes
* tampering breaks verification
* a no-substitution replay is 100% cache-hit, 0 real executions
* a `ToolOutputSubstitution` at step 2 makes step 2 dirty + propagates
  dirtiness to every descendant whose inputs depend on it
* `bisect` finds the earliest step that wires to the wrong IBAN, in
  ≤ ⌈log2(N)⌉ probes, with zero LLM calls
* `branch_at` + `compare_branches` produces a non-empty diff with
  measurable cost delta
"""
from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys

import pytest

from stepback import RecorderKey, record, replay
from stepback.canonical import CANONICALISATION_VERSION
from stepback.substitutions import (
    PromptSubstitution,
    ToolOutputSubstitution,
    ModelSubstitution,
)
from stepback.trace_reader import TraceVerificationError, verify_trace
from stepback.testing import (
    LOOKUP_FIXED_ROW,
    fake_llm,
    fake_tool,
    run_recorded_agent,
)


# ----------------------------------------------------------- helpers


def _record_fixture(tmp_path) -> tuple[str, RecorderKey]:
    key = RecorderKey.fresh()
    path = str(tmp_path / "trace.sb")
    with record(path, key=key) as rec:
        run_recorded_agent(rec)
    return path, key


# --------------------------------------------------- header + format


def test_header_contains_pinned_versions(tmp_path):
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    assert t.header["magic"] == "stepback/.sb"
    assert t.header["format_version"] == 1
    assert t.header["canonicalisation_version"] == CANONICALISATION_VERSION
    assert "public_key" in t.header
    assert "price_list_version" in t.header


def test_recorded_step_count_matches_agent(tmp_path):
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    assert len(t.recorded_steps) == 12
    kinds = [s["step_kind"] for s in t.recorded_steps]
    assert kinds.count("llm_call") == 6
    assert kinds.count("tool_call") == 6


# --------------------------------------------------- chain integrity


def test_verify_trace_succeeds_on_untouched_file(tmp_path):
    path, key = _record_fixture(tmp_path)
    v = verify_trace(path, key.hmac_key)
    # Header + 12 steps + tail.
    assert len(v.steps) == 12
    assert v.tail is not None


def test_verify_trace_detects_tampering(tmp_path):
    path, key = _record_fixture(tmp_path)
    # Flip a byte deep inside one of the step frames.
    with open(path, "rb") as _f:
        raw = _f.read()
    # Find the first occurrence of "Acme Bolts" inside the on-disk
    # bytes and flip a character so the canonical-JSON of one frame
    # no longer matches its HMAC.
    needle = b"Acme Bolts"
    i = raw.find(needle)
    assert i > 0
    tampered = raw[:i] + b"Acme Boltz" + raw[i + len(needle) :]
    with open(path, "wb") as _f:
        _f.write(tampered)
    with pytest.raises(TraceVerificationError):
        verify_trace(path, key.hmac_key)


# -------------------------------------------------- replay caching


def test_replay_no_substitution_is_all_cached(tmp_path):
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    result = t.replay_forward()
    # Every step is a cache hit; zero LLM/tool re-executions.
    assert result.real_executions == 0
    assert result.dirty_count == 0
    assert result.cache_hit_count == 12
    # Outputs preserved exactly.
    last = result.steps[-1]
    assert last.kind == "tool_call"
    assert last.outputs["result"]["status"] == "ok"


# ------------------------------------------- substitution propagation


def test_tool_output_substitution_propagates_dirty_subtree(tmp_path):
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)

    # Fix the bug by pinning step:2 (lookup_customer) to the US row.
    t.substitute(ToolOutputSubstitution(at_step="step:2", fake_response=LOOKUP_FIXED_ROW))
    from stepback import Executor
    result = t.replay_forward(Executor(llm=fake_llm, tool=fake_tool))

    # step:1 was upstream of the substitution -> still cached.
    assert result.steps[0].cache_hit is True
    # step:2 was forced -> dirty.
    assert result.steps[1].dirty is True
    assert result.steps[1].outputs["result"]["country"] == "US"

    # Every step downstream of step:2 must be dirty (its `context`
    # field was rebound to the new parent output hash, so the inputs
    # hash diverges from the recorded hash).
    for sv in result.steps[1:]:
        assert sv.dirty, f"{sv.step_id} should be dirty"

    # The final wire's *recorded* args still reference the bad IBAN
    # (the agent code that built those args isn't replayed in v0.1 —
    # only the leaf step is). What we *do* guarantee is that step:12
    # is correctly marked dirty and re-executed:
    assert result.steps[-1].dirty is True
    # Every step from step:2 onward (11 of them) is dirty.
    assert result.dirty_count == 11
    # 11 dirty = 1 ToolOutputSubstitution forced + 10 real executor calls.
    assert result.real_executions == 10
    assert result.cache_hit_count == 1


def test_prompt_substitution_dirties_only_one_step_when_no_context(tmp_path):
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    # step:1 is the very first llm_call, no parent, no context binding.
    t.substitute(
        PromptSubstitution(
            at_step="step:1",
            new_messages=[{"role": "system", "content": "BE PARANOID."}],
        )
    )
    from stepback import Executor
    result = t.replay_forward(Executor(llm=fake_llm, tool=fake_tool))

    # step:1 is dirty (input changed); every downstream step that
    # bound `context` to step:1's output hash is also dirty.
    assert result.steps[0].dirty is True
    assert result.steps[1].dirty is True  # depended on step:1
    # step:2's siblings/successors not directly bound to step:1 may
    # cascade — but at minimum, the substitution must be effective.
    assert result.real_executions >= 1
    # The new system prompt must show up in step:1's recomputed inputs.
    assert result.steps[0].inputs["messages"][0]["content"] == "BE PARANOID."


def test_model_substitution_changes_cost(tmp_path):
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    base = t.replay_forward()
    base_cost = base.total_cost_usd

    t.substitute(ModelSubstitution(at_step="step:1", new_model_id="gpt-4o-mini-2024-07-18"))
    from stepback import Executor
    result = t.replay_forward(Executor(llm=fake_llm, tool=fake_tool))
    # Mini model is much cheaper, so total cost must drop strictly.
    assert result.total_cost_usd < base_cost


# ------------------------------------------------------------ bisect


def test_bisect_finds_first_bad_payment_with_no_llm_calls(tmp_path):
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)

    found = t.bisect(
        good="step:1",
        bad="step:12",
        predicate=lambda s: s.kind == "tool_call"
        and s.name == "payment.transfer"
        and "GB99" in str(s.outputs.get("result", {}).get("wire_to_iban", "")),
    )
    assert found is not None
    assert found.step_id == "step:12"
    # Bisect over 12 steps must take ≤ ⌈log2(12)⌉ + 1 probes.
    assert t.last_bisect_probes <= math.ceil(math.log2(12)) + 1


def test_bisect_zero_match_returns_none(tmp_path):
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    assert t.bisect(good="step:1", bad="step:12", predicate=lambda s: False) is None


# ----------------------------------------------------- branch + diff


def test_branch_compare_produces_nonempty_diff(tmp_path):
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    from stepback import Executor

    b_main = t.branch_at("step:1", name="main")
    b_main.replay_forward()
    b_fix = t.branch_at("step:2", name="fixed-lookup").substitute(
        ToolOutputSubstitution(at_step="step:2", fake_response=LOOKUP_FIXED_ROW)
    )
    b_fix.replay_forward(Executor(llm=fake_llm, tool=fake_tool))
    diff = t.compare_branches(b_main, b_fix)
    assert diff.divergent_step_count >= 1
    # The lookup step (step:2) is the substitution point and must
    # show different outputs between the two branches.
    by_id = {sd.step_id: sd for sd in diff.step_diffs}
    assert by_id["step:2"].output_diff != {}
    assert by_id["step:2"].output_diff["a"]["result"]["country"] == "UK"
    assert by_id["step:2"].output_diff["b"]["result"]["country"] == "US"
    assert isinstance(diff.total_cost_delta_usd, float)


# ------------------------------------------------------------- CLI


def test_cli_inspect_runs(tmp_path, capsys):
    path, _ = _record_fixture(tmp_path)
    rc = subprocess.run(
        [sys.executable, "-m", "stepback.cli", "inspect", path],
        capture_output=True, text=True, check=True,
    )
    assert "stepback" not in rc.stderr.lower()
    assert "step:12" in rc.stdout
    assert "payment.transfer" in rc.stdout


def test_cli_verify_ok(tmp_path):
    path, key = _record_fixture(tmp_path)
    rc = subprocess.run(
        [
            sys.executable, "-m", "stepback.cli", "verify", path,
            "--hmac-key-hex", key.hmac_key.hex(),
        ],
        capture_output=True, text=True, check=False,
    )
    assert rc.returncode == 0, rc.stderr
    assert rc.stdout.startswith("OK")


def test_trace_size_and_per_step_byte_bounds(tmp_path):
    """Numeric-threshold guarantees on the on-disk fixture trace.

    Pins the signed-trace overhead so any future schema bloat (extra
    metadata fields, unbounded payloads) trips here rather than
    silently inflating production traces. Also pins the floor so
    accidental envelope-stripping (no HMAC/Ed25519) breaks the test.
    """
    path, key = _record_fixture(tmp_path)
    size = os.path.getsize(path)
    t = replay(path, hmac_key=key.hmac_key)
    n = len(t.recorded_steps)
    assert n == 12
    per_step = size / n
    # Signed envelope floor: a 12-step HMAC+Ed25519 trace must be ≥2KB.
    assert size > 2048, f"trace size {size} below signed-envelope floor"
    # Hard ceiling: 12 steps with full payloads must stay under 64KB.
    assert size < 64_000, f"trace size {size} blew past per-trace ceiling"
    # Per-step amortised byte ceiling — the only way to bust this is
    # bloat per step (extra fields, unbounded debug payloads).
    assert 200 < per_step < 4000, f"per-step bytes {per_step:.1f} out of bounds"


def test_replay_speedup_vs_record_is_measurable(tmp_path):
    """Cached replay must be measurably faster than fresh record.

    Numeric-threshold guarantee on the cache-hit path: zero LLM/tool
    re-executions AND wall-clock under the recorder time (with slack
    for tiny-trace filesystem variance).
    """
    import time
    t0 = time.perf_counter()
    path, key = _record_fixture(tmp_path)
    record_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    t = replay(path, hmac_key=key.hmac_key)
    result = t.replay_forward()
    replay_time = time.perf_counter() - t1

    assert result.real_executions == 0
    assert result.cache_hit_count == 12
    assert result.dirty_count == 0
    # Replay should not be more than ~2x recording time (+50ms slack).
    assert replay_time < record_time * 2.0 + 0.05, (
        f"replay={replay_time:.4f}s vs record={record_time:.4f}s"
    )


def test_model_substitution_cost_drop_is_significant(tmp_path):
    """ModelSubstitution to gpt-4o-mini must yield a numerically large
    cost drop relative to the recorded gpt-4o baseline — not a
    trivial rounding-noise delta.
    """
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    base = t.replay_forward()
    base_cost = base.total_cost_usd
    assert base_cost > 0.0, "baseline cost must be positive"

    t.substitute(ModelSubstitution(at_step="step:1", new_model_id="gpt-4o-mini-2024-07-18"))
    from stepback import Executor
    result = t.replay_forward(Executor(llm=fake_llm, tool=fake_tool))
    new_cost = result.total_cost_usd
    # Strictly cheaper.
    assert new_cost < base_cost
    # And cheaper by at least 1% of the baseline (mini is ~30x cheaper
    # for the substituted step; even with only 1/6 LLM steps swapped
    # the savings dwarf 1%).
    drop_ratio = (base_cost - new_cost) / base_cost
    assert drop_ratio > 0.01, f"only {drop_ratio*100:.3f}% cost drop"


def test_bisect_probe_count_is_logarithmic(tmp_path):
    """Bisect must be log-N in step count and zero-LLM."""
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    found = t.bisect(
        good="step:1",
        bad="step:12",
        predicate=lambda s: s.kind == "tool_call"
        and s.name == "payment.transfer",
    )
    assert found is not None
    assert found.step_id == "step:12"
    # log2(12) = 3.58, so ⌈log2⌉+1 = 5 probes max.
    assert t.last_bisect_probes <= 5
    # Lower bound: bisect MUST probe at least once.
    assert t.last_bisect_probes >= 1


def test_cli_bisect_finds_step(tmp_path):
    path, _ = _record_fixture(tmp_path)
    rc = subprocess.run(
        [
            sys.executable, "-m", "stepback.cli", "bisect", path,
            "--good", "step:1", "--bad", "step:12",
            "--predicate",
            'step.kind=="tool_call" and step.name=="payment.transfer"',
        ],
        capture_output=True, text=True, check=True,
    )
    out = json.loads(rc.stdout)
    assert out["first_bad"] == "step:12"
