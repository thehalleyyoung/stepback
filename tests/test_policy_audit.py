"""Tests for stepback.policy_audit (regulator-replay impact reports).

Covers the public surface added by `stepback/policy_audit.py`:

* `is_policy_blocked` — predicate that classifies a step's outputs
  as a policy denial across the three shapes the codebase emits
  (recorder `exception` step, `RaiseSubstitution.__error__` sentinel,
  free-form `{"blocked": True, ...}` dict).
* `audit_policy_change` — the main entry point that re-replays each
  trace under a (counterfactual) policy and returns a
  `PolicyImpactReport`.
* `PolicyImpactReport.to_json` / `to_markdown` — deterministic
  serializers.
* The `stepback policy-audit` CLI subcommand (smoke test).

Every audit test runs against the real on-disk 12-step `.sb` fixture
produced by `tests.fixtures.agent.run_recorded_agent` (no mocks),
satisfying the e2e-fixture-testing rule.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from stepback import (
    RecorderKey,
    audit_policy_change,
    is_policy_blocked,
    record,
)
from stepback.policy_audit import PolicyImpactReport, StepImpact, TraceImpact
from stepback.substitutions import (
    OutputsPatchSubstitution,
    PolicySubstitution,
    RaiseSubstitution,
)
from stepback.testing import run_recorded_agent


# ----------------------------------------------------------- helpers


def _record_trace(tmp_path, name: str = "t.sb"):
    key = RecorderKey.fresh()
    p = str(tmp_path / name)
    with record(p, key=key) as rec:
        run_recorded_agent(rec)
    return p, key.hmac_key


def _all_step_ids(report: PolicyImpactReport, trace_index: int = 0):
    return [s.step_id for s in report.trace_impacts[trace_index].step_impacts]


# ------------------------------------------------------ is_policy_blocked


class TestIsPolicyBlocked:
    def test_recognises_recorder_exception_shape(self):
        assert is_policy_blocked({"error_class": "PolicyDenied", "message": "no PII"})

    def test_recognises_raise_substitution_sentinel(self):
        assert is_policy_blocked(
            {"__error__": {"type": "PolicyViolation", "message": "blocked"}}
        )

    def test_recognises_free_form_blocked_dict(self):
        assert is_policy_blocked({"blocked": True, "reason": "PII detected"})
        assert is_policy_blocked({"blocked": True, "policy_path": "p.tw"})

    def test_blocked_true_without_reason_is_not_enough(self):
        # A bare `blocked: True` with no reason / policy field is too
        # generic — many tools have a `blocked` field meaning something else.
        assert not is_policy_blocked({"blocked": True})

    def test_case_insensitive_substring_on_token(self):
        assert is_policy_blocked({"error_class": "policy.deny"})
        assert is_policy_blocked({"error_class": "TOOLWARDEN_POLICY_REJECT"})

    def test_non_policy_error_does_not_match(self):
        assert not is_policy_blocked({"error_class": "TimeoutError"})
        assert not is_policy_blocked({"__error__": {"type": "ValueError"}})

    def test_non_dict_returns_false(self):
        assert not is_policy_blocked(None)
        assert not is_policy_blocked("PolicyDenied")
        assert not is_policy_blocked(["PolicyDenied"])
        assert not is_policy_blocked(42)

    def test_normal_llm_output_returns_false(self):
        assert not is_policy_blocked(
            {"choices": [{"message": {"content": "hi"}}], "usage": {}}
        )


# ----------------------------------------------- audit_policy_change basics


class TestAuditNoSubstitution:
    """A null audit (no policy, no extras) must report zero divergence."""

    def test_no_substitutions_zero_divergence(self, tmp_path):
        p, k = _record_trace(tmp_path)
        report = audit_policy_change([p], hmac_key=k)
        assert report.trace_count == 1
        assert report.traces_with_divergence == 0
        assert report.total_divergent_steps == 0
        assert report.total_newly_blocked_steps == 0
        assert report.total_newly_allowed_steps == 0
        assert report.total_cost_delta_usd == 0.0
        ti = report.trace_impacts[0]
        assert ti.error is None
        assert ti.step_count == 12
        assert ti.divergent_step_count == 0
        assert all(
            s.classification == "unchanged" for s in ti.step_impacts
        )

    def test_substitution_kinds_empty_when_no_subs(self, tmp_path):
        p, k = _record_trace(tmp_path)
        report = audit_policy_change([p], hmac_key=k)
        assert report.substitution_kinds == []


class TestAuditNewlyBlocked:
    """Forcing one step to a PolicyDenied output must classify it
    `newly_blocked` and propagate dirtiness through descendants."""

    def test_raise_substitution_marks_step_newly_blocked(self, tmp_path):
        p, k = _record_trace(tmp_path)
        # Block the final tool_call (step:12 is `payment.transfer`).
        sub = RaiseSubstitution(
            at_step="step:12",
            exception_type="PolicyDenied",
            message="payment over $10k requires approval",
        )
        report = audit_policy_change(
            [p], hmac_key=k, extra_substitutions=[sub]
        )
        ti = report.trace_impacts[0]
        assert ti.divergent_step_count >= 1
        assert "step:12" in ti.newly_blocked_step_ids
        # No step was previously blocked, so newly_allowed must be empty.
        assert ti.newly_allowed_step_ids == []
        assert report.total_newly_blocked_steps >= 1
        # Aggregate divergence count matches the per-trace count.
        assert report.total_divergent_steps == ti.divergent_step_count

    def test_outputs_patch_classified_as_divergent(self, tmp_path):
        # Patch step:8 (a tool_call) to a non-block divergence — it
        # must be classified `divergent`, not `newly_blocked`.
        p, k = _record_trace(tmp_path)
        sub = OutputsPatchSubstitution(
            at_step="step:8",
            ops=[{"op": "add", "path": "/audited", "value": True}],
        )
        report = audit_policy_change(
            [p], hmac_key=k, extra_substitutions=[sub]
        )
        ti = report.trace_impacts[0]
        s8 = next(s for s in ti.step_impacts if s.step_id == "step:8")
        assert s8.classification == "divergent"
        assert "step:8" not in ti.newly_blocked_step_ids

    def test_substitution_kinds_recorded(self, tmp_path):
        p, k = _record_trace(tmp_path)
        sub = RaiseSubstitution(at_step="step:12", exception_type="PolicyDenied")
        report = audit_policy_change(
            [p], hmac_key=k, new_policy_path="/tmp/p.tw",
            extra_substitutions=[sub],
        )
        # Sorted alphabetically.
        assert report.substitution_kinds == [
            "PolicySubstitution",
            "RaiseSubstitution",
        ]


class TestAuditNewlyAllowed:
    """If the BASELINE has a blocked step (recorded `error_class`
    containing "Policy") and the substitution flips it to allowed,
    that must be classified `newly_allowed`."""

    def test_outputs_patch_unblock_is_newly_allowed(self, tmp_path):
        # Step 1: record the trace normally (baseline has no block).
        # Step 2: build a fresh trace where one step's recorded
        # outputs already look blocked, by patching the baseline
        # via OutputsPatch in REVERSE: actually easier — make the
        # baseline blocked by adding a custom "blocked" exception
        # step via the recorder API.
        key = RecorderKey.fresh()
        p = str(tmp_path / "blocked.sb")
        with record(p, key=key) as rec:
            run_recorded_agent(rec)
            # Append an extra blocked step at the end.
            rec.exception("PolicyDenied", "test-block")

        # Audit with an OutputsPatch that REPLACES the blocked
        # output with an allowed one.
        # Find the last step's id.
        from stepback import replay as load_trace
        t = load_trace(p, hmac_key=key.hmac_key)
        last_id = t.recorded_steps[-1]["step_id"]
        assert is_policy_blocked(t.recorded_steps[-1]["outputs"])

        unblock = OutputsPatchSubstitution(
            at_step=last_id,
            ops=[
                {"op": "remove", "path": "/error_class"},
                {"op": "add", "path": "/status", "value": "ok"},
            ],
        )
        report = audit_policy_change(
            [p], hmac_key=key.hmac_key, extra_substitutions=[unblock]
        )
        ti = report.trace_impacts[0]
        assert last_id in ti.newly_allowed_step_ids
        assert last_id not in ti.newly_blocked_step_ids
        assert report.total_newly_allowed_steps == 1


class TestAuditMultipleTraces:
    def test_aggregates_across_traces(self, tmp_path):
        p1, k1 = _record_trace(tmp_path, "a.sb")
        p2, k2 = _record_trace(tmp_path, "b.sb")
        # Audit both with the SAME blocking substitution.
        sub = RaiseSubstitution(
            at_step="step:12", exception_type="PolicyDenied"
        )
        # Both fixtures share the same key shape (fresh per trace);
        # we can't pass an hmac_key here because they differ. Run
        # without verification — that exercises the optional path.
        report = audit_policy_change([p1, p2], extra_substitutions=[sub])
        assert report.trace_count == 2
        assert report.traces_with_divergence == 2
        assert report.total_newly_blocked_steps == 2
        # Trace order is preserved.
        assert report.trace_impacts[0].trace_path == p1
        assert report.trace_impacts[1].trace_path == p2

    def test_bad_hmac_key_recorded_as_per_trace_error(self, tmp_path):
        p, _real_key = _record_trace(tmp_path)
        # Use a DIFFERENT key — verification fails for this trace.
        bad = bytes(32)  # all-zero, almost certainly wrong
        report = audit_policy_change([p], hmac_key=bad)
        ti = report.trace_impacts[0]
        assert ti.error is not None
        assert ti.divergent_step_count == 0
        assert ti.step_count == 0
        # Aggregate counter reflects the failure.
        assert report.traces_with_error == 1


# -------------------------------------------------------- serialization


class TestReportSerialization:
    def test_to_json_round_trip(self, tmp_path):
        p, k = _record_trace(tmp_path)
        sub = RaiseSubstitution(at_step="step:12", exception_type="PolicyDenied")
        report = audit_policy_change(
            [p], hmac_key=k, new_policy_path="policies/v7.tw",
            policy_version_pin="2026-04-15",
            extra_substitutions=[sub],
        )
        body = report.to_json()
        # Required top-level fields.
        for f in [
            "policy_path", "policy_version_pin", "substitution_kinds",
            "trace_count", "traces_with_divergence", "traces_with_error",
            "total_divergent_steps", "total_newly_blocked_steps",
            "total_newly_allowed_steps", "total_baseline_cost_usd",
            "total_replayed_cost_usd", "total_cost_delta_usd",
            "trace_impacts",
        ]:
            assert f in body
        assert body["policy_path"] == "policies/v7.tw"
        assert body["policy_version_pin"] == "2026-04-15"
        assert body["trace_count"] == 1
        assert body["substitution_kinds"] == [
            "PolicySubstitution", "RaiseSubstitution",
        ]
        # JSON-roundtrippable.
        s = report.to_json_str()
        parsed = json.loads(s)
        assert parsed == body

    def test_step_impact_to_json_shape(self, tmp_path):
        p, k = _record_trace(tmp_path)
        report = audit_policy_change([p], hmac_key=k)
        si = report.trace_impacts[0].step_impacts[0]
        body = si.to_json()
        assert set(body.keys()) == {
            "step_id", "kind", "name", "classification",
            "baseline_blocked", "replayed_blocked",
            "cost_delta_usd", "diverged_from_cache",
        }

    def test_to_markdown_contains_required_sections(self, tmp_path):
        p, k = _record_trace(tmp_path)
        sub = RaiseSubstitution(at_step="step:12", exception_type="PolicyDenied")
        report = audit_policy_change(
            [p], hmac_key=k, new_policy_path="p.tw",
            policy_version_pin="v1", extra_substitutions=[sub],
        )
        md = report.to_markdown()
        assert md.startswith("# Policy Impact Report")
        assert "## Inputs" in md
        assert "## Aggregate" in md
        assert "## Per-trace summary" in md
        assert "## Divergent step detail" in md
        assert "p.tw" in md
        assert "v1" in md
        assert "newly_blocked" in md
        assert "PolicySubstitution" in md
        assert "RaiseSubstitution" in md
        # The trace path appears in the per-trace table.
        assert p in md
        # Aggregate row line present.
        assert "traces audited | 1" in md
        # Markdown ends with a newline.
        assert md.endswith("\n")
        # Length is non-trivial.
        assert len(md) > 500

    def test_to_markdown_omits_detail_when_no_divergence(self, tmp_path):
        p, k = _record_trace(tmp_path)
        report = audit_policy_change([p], hmac_key=k)
        md = report.to_markdown()
        assert "## Divergent step detail" not in md
        # But the summary section is always present.
        assert "## Per-trace summary" in md

    def test_to_markdown_is_deterministic(self, tmp_path):
        p, k = _record_trace(tmp_path)
        sub = RaiseSubstitution(at_step="step:12", exception_type="PolicyDenied")
        r1 = audit_policy_change([p], hmac_key=k, extra_substitutions=[sub])
        r2 = audit_policy_change([p], hmac_key=k, extra_substitutions=[sub])
        assert r1.to_markdown() == r2.to_markdown()
        assert r1.to_json_str() == r2.to_json_str()


# ------------------------------------------------------------- CLI


class TestCLI:
    def test_cli_policy_audit_json_smoke(self, tmp_path):
        p, k = _record_trace(tmp_path)
        out = tmp_path / "report.json"
        rc = subprocess.call(
            [
                sys.executable, "-m", "stepback.cli", "policy-audit",
                p, "--hmac-key-hex", k.hex(),
                "--substitute", "raise@step:12=PolicyDenied:blocked",
                "--format", "json",
                "--output", str(out),
            ],
            cwd="/Users/halleyyoung/projects/kitchensink/repos/stepback",
        )
        assert rc == 0
        body = json.loads(out.read_text())
        assert body["trace_count"] == 1
        assert body["traces_with_divergence"] == 1
        assert body["total_newly_blocked_steps"] == 1
        # substitution_kinds should include RaiseSubstitution.
        assert "RaiseSubstitution" in body["substitution_kinds"]

    def test_cli_policy_audit_markdown_default(self, tmp_path):
        p, k = _record_trace(tmp_path)
        out = tmp_path / "report.md"
        rc = subprocess.call(
            [
                sys.executable, "-m", "stepback.cli", "policy-audit",
                p, "--hmac-key-hex", k.hex(),
                "--output", str(out),
            ],
            cwd="/Users/halleyyoung/projects/kitchensink/repos/stepback",
        )
        assert rc == 0
        text = out.read_text()
        assert text.startswith("# Policy Impact Report")
        assert "traces audited | 1" in text

    def test_cli_exit_nonzero_on_divergence(self, tmp_path):
        p, k = _record_trace(tmp_path)
        rc = subprocess.call(
            [
                sys.executable, "-m", "stepback.cli", "policy-audit",
                p, "--hmac-key-hex", k.hex(),
                "--substitute", "raise@step:12=PolicyDenied:blocked",
                "--format", "json",
                "--exit-nonzero-on-divergence",
            ],
            cwd="/Users/halleyyoung/projects/kitchensink/repos/stepback",
            stdout=subprocess.DEVNULL,
        )
        assert rc == 3

    def test_cli_exit_zero_when_no_divergence(self, tmp_path):
        p, k = _record_trace(tmp_path)
        rc = subprocess.call(
            [
                sys.executable, "-m", "stepback.cli", "policy-audit",
                p, "--hmac-key-hex", k.hex(),
                "--exit-nonzero-on-divergence",
                "--format", "json",
            ],
            cwd="/Users/halleyyoung/projects/kitchensink/repos/stepback",
            stdout=subprocess.DEVNULL,
        )
        assert rc == 0

    def test_cli_help_lists_subcommand(self, tmp_path):
        r = subprocess.run(
            [sys.executable, "-m", "stepback.cli", "--help"],
            cwd="/Users/halleyyoung/projects/kitchensink/repos/stepback",
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "policy-audit" in r.stdout


# --------------------------------------------- numeric-threshold tightening


class TestNumericThresholds:
    def test_step_count_matches_recorded(self, tmp_path):
        p, k = _record_trace(tmp_path)
        report = audit_policy_change([p], hmac_key=k)
        # The fixture is exactly 12 steps.
        assert report.trace_impacts[0].step_count == 12
        # And so the step_impacts list also has exactly 12 entries.
        assert len(report.trace_impacts[0].step_impacts) == 12

    def test_unchanged_count_equals_step_count_when_no_subs(self, tmp_path):
        p, k = _record_trace(tmp_path)
        report = audit_policy_change([p], hmac_key=k)
        unchanged = sum(
            1 for s in report.trace_impacts[0].step_impacts
            if s.classification == "unchanged"
        )
        assert unchanged == 12

    def test_divergent_count_after_blocking_step12_matches_dirty_subtree(self, tmp_path):
        # After blocking step:12, only step:12 should be classified
        # divergent — it is the LAST step in the trace, so no
        # downstream cascade is possible.
        p, k = _record_trace(tmp_path)
        sub = RaiseSubstitution(at_step="step:12", exception_type="PolicyDenied")
        report = audit_policy_change(
            [p], hmac_key=k, extra_substitutions=[sub]
        )
        ti = report.trace_impacts[0]
        assert ti.divergent_step_count == 1
        assert ti.newly_blocked_step_ids == ["step:12"]
        unchanged = sum(
            1 for s in ti.step_impacts if s.classification == "unchanged"
        )
        assert unchanged == 11

    def test_baseline_cost_matches_recorded_cost(self, tmp_path):
        p, k = _record_trace(tmp_path)
        report = audit_policy_change([p], hmac_key=k)
        # baseline replay walks every step from cache, so the
        # baseline cost equals the per-step recorded cost sum.
        from stepback import replay as load_trace
        t = load_trace(p, hmac_key=k)
        recorded_total = sum(s.get("cost_usd", 0.0) for s in t.recorded_steps)
        # Round to compare floats.
        assert (
            abs(report.trace_impacts[0].baseline_cost_usd - recorded_total)
            < 1e-9
        )

    def test_aggregate_cost_delta_is_sum_of_per_trace(self, tmp_path):
        p1, _ = _record_trace(tmp_path, "a.sb")
        p2, _ = _record_trace(tmp_path, "b.sb")
        sub = RaiseSubstitution(at_step="step:12", exception_type="PolicyDenied")
        report = audit_policy_change([p1, p2], extra_substitutions=[sub])
        s = sum(t.cost_delta_usd for t in report.trace_impacts)
        assert abs(report.total_cost_delta_usd - round(s, 8)) < 1e-9
