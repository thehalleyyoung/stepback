"""Tests for the production case studies (step 148).

All tests are:
* Fully offline — no network calls.
* Deterministic — seeded RNG where applicable.
* Testing invariants, not exact values — elapsed times, throughput, and
  randomly generated keys are not compared directly.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from stepback.case_studies import (
    EvidencePackResult,
    HighVolumeResult,
    IncidentReplayResult,
    ModelMigrationResult,
    ParameterSweepResult,
    RedactedPublicationResult,
    run_evidence_pack,
    run_high_volume,
    run_incident_replay,
    run_model_migration,
    run_parameter_sweep,
    run_redacted_publication,
)


# ---------------------------------------------------------------------------
# High-volume recording
# ---------------------------------------------------------------------------

class TestHighVolume:
    def test_returns_result_type(self):
        result = run_high_volume(n_traces=5, n_steps=4, seed=0)
        assert isinstance(result, HighVolumeResult)

    def test_traces_written_count(self):
        result = run_high_volume(n_traces=8, n_steps=3, seed=1)
        assert result.traces_written == 8
        assert result.n_traces == 8

    def test_elapsed_positive(self):
        result = run_high_volume(n_traces=3, n_steps=2, seed=0)
        assert result.elapsed_s > 0

    def test_throughput_positive(self):
        result = run_high_volume(n_traces=5, n_steps=3, seed=0)
        assert result.traces_per_second > 0

    def test_projected_runs_per_day_positive(self):
        result = run_high_volume(n_traces=5, n_steps=3, seed=0)
        assert result.projected_runs_per_day > 0

    def test_bytes_written_positive(self):
        result = run_high_volume(n_traces=5, n_steps=3, seed=0)
        assert result.total_bytes_written > 0

    def test_compression_ratio_gte_one(self):
        # compression_ratio is reported as 1.0 (traces written with default compression)
        result = run_high_volume(n_traces=5, n_steps=3, seed=0)
        assert result.compression_ratio >= 1.0

    def test_signing_enabled_true(self):
        result = run_high_volume(n_traces=5, n_steps=3, seed=0)
        assert result.signing_enabled is True

    def test_no_recording_errors(self):
        result = run_high_volume(n_traces=5, n_steps=3, seed=0)
        assert result.recording_errors == 0

    def test_success_rate_one(self):
        result = run_high_volume(n_traces=5, n_steps=3, seed=0)
        assert result.success_rate == 1.0

    def test_mean_bytes_per_trace_positive(self):
        result = run_high_volume(n_traces=5, n_steps=4, seed=0)
        assert result.mean_bytes_per_trace > 0

    def test_mean_bytes_per_step_positive(self):
        result = run_high_volume(n_traces=5, n_steps=4, seed=0)
        assert result.mean_bytes_per_step > 0

    def test_summary_line_contains_key_tokens(self):
        result = run_high_volume(n_traces=5, n_steps=4, seed=0)
        line = result.summary_line()
        assert "high_volume" in line
        assert "traces/s" in line
        assert "compress=" in line

    def test_to_json_round_trips(self):
        result = run_high_volume(n_traces=3, n_steps=2, seed=0)
        payload = json.loads(result.to_json())
        assert payload["n_traces"] == 3
        assert payload["traces_written"] == 3
        assert payload["signing_enabled"] is True

    def test_invalid_n_traces_raises(self):
        with pytest.raises(ValueError, match="n_traces"):
            run_high_volume(n_traces=0)

    def test_invalid_n_steps_raises(self):
        with pytest.raises(ValueError, match="n_steps"):
            run_high_volume(n_steps=0)

    def test_output_dir_respected(self):
        with tempfile.TemporaryDirectory() as d:
            result = run_high_volume(n_traces=3, n_steps=2, seed=0, output_dir=d)
            assert result.traces_written == 3

    def test_bytes_per_trace_consistent_with_n_steps(self):
        # More steps → more bytes
        r_few = run_high_volume(n_traces=3, n_steps=2, seed=0)
        r_many = run_high_volume(n_traces=3, n_steps=6, seed=0)
        assert r_many.mean_bytes_per_trace > r_few.mean_bytes_per_trace


# ---------------------------------------------------------------------------
# Incident replay
# ---------------------------------------------------------------------------

class TestIncidentReplay:
    def test_returns_result_type(self):
        result = run_incident_replay()
        assert isinstance(result, IncidentReplayResult)

    def test_six_steps_recorded(self):
        result = run_incident_replay()
        assert result.n_steps == 6

    def test_culprit_found(self):
        result = run_incident_replay()
        assert result.culprit_found is True

    def test_culprit_step_matches_expected(self):
        result = run_incident_replay()
        assert result.culprit_step_id == result.expected_culprit_step

    def test_bisect_probes_positive(self):
        result = run_incident_replay()
        assert result.bisect_probes >= 1

    def test_bisect_probes_logarithmic(self):
        # For 6 steps, binary search should take at most ceil(log2(6)) = 3 probes
        result = run_incident_replay()
        assert result.bisect_probes <= 4  # generous bound

    def test_dirty_after_fix_at_least_one(self):
        result = run_incident_replay()
        assert result.dirty_after_fix >= 1

    def test_dirty_after_fix_at_most_n_steps(self):
        result = run_incident_replay()
        assert result.dirty_after_fix <= result.n_steps

    def test_original_policy_blocked_true(self):
        result = run_incident_replay()
        assert result.original_policy_output.get("policy_blocked") is True

    def test_fixed_policy_blocked_false(self):
        result = run_incident_replay()
        assert result.fixed_policy_output.get("policy_blocked") is False

    def test_summary_line_contains_tokens(self):
        result = run_incident_replay()
        line = result.summary_line()
        assert "incident_replay" in line
        assert "culprit=" in line
        assert "found=" in line

    def test_to_json_round_trips(self):
        result = run_incident_replay()
        payload = json.loads(result.to_json())
        assert payload["n_steps"] == 6
        assert payload["culprit_found"] is True


# ---------------------------------------------------------------------------
# Evidence packs
# ---------------------------------------------------------------------------

class TestEvidencePacks:
    def test_returns_result_type(self):
        result = run_evidence_pack()
        assert isinstance(result, EvidencePackResult)

    def test_pack_verified(self):
        result = run_evidence_pack()
        assert result.pack_verified is True

    def test_pack_signature_valid(self):
        result = run_evidence_pack()
        assert result.pack_signature_valid is True

    def test_n_entries_one(self):
        result = run_evidence_pack()
        assert result.n_entries == 1

    def test_n_steps_15(self):
        # 5 applications × 3 steps each = 15
        result = run_evidence_pack()
        assert result.n_steps == 15

    def test_policy_version_pin_default(self):
        result = run_evidence_pack()
        assert result.policy_version_pin == "v2.3.1"

    def test_policy_version_pin_custom(self):
        result = run_evidence_pack(policy_version="v3.0.0")
        assert result.policy_version_pin == "v3.0.0"

    def test_attestor_public_key_present(self):
        result = run_evidence_pack()
        assert isinstance(result.attestor_public_key, str)
        assert len(result.attestor_public_key) > 8

    def test_produced_at_present(self):
        result = run_evidence_pack()
        assert isinstance(result.produced_at, str)
        # ISO timestamp contains 'T'
        assert "T" in result.produced_at

    def test_summary_line_contains_tokens(self):
        result = run_evidence_pack()
        line = result.summary_line()
        assert "evidence_pack" in line
        assert "pack_verified=True" in line

    def test_to_json_round_trips(self):
        result = run_evidence_pack()
        payload = json.loads(result.to_json())
        assert payload["pack_verified"] is True
        assert payload["n_entries"] == 1


# ---------------------------------------------------------------------------
# Model migration
# ---------------------------------------------------------------------------

class TestModelMigration:
    def test_returns_result_type(self):
        result = run_model_migration(n_traces=3, n_steps=4, seed=0)
        assert isinstance(result, ModelMigrationResult)

    def test_correct_model_ids(self):
        result = run_model_migration(n_traces=3, n_steps=4, seed=0)
        assert result.model_a == "gpt-4o-2024-11-20"
        assert result.model_b == "gpt-4o-mini-2024-07-18"

    def test_dirty_count_mean_positive(self):
        result = run_model_migration(n_traces=3, n_steps=4, seed=0)
        assert result.dirty_count_mean > 0

    def test_dirty_fraction_between_zero_and_one(self):
        result = run_model_migration(n_traces=3, n_steps=4, seed=0)
        assert 0 < result.dirty_fraction_mean <= 1.0

    def test_all_llm_steps_dirty(self):
        # Every llm_call step should be dirty after a full model swap
        result = run_model_migration(n_traces=3, n_steps=4, seed=0)
        # dirty_count_mean >= llm_steps_per_trace
        assert result.dirty_count_mean >= result.llm_steps_per_trace

    def test_cost_reduction_factor_present(self):
        result = run_model_migration(n_traces=3, n_steps=4, seed=0)
        # gpt-4o and gpt-4o-mini are both in the pricing catalog
        assert result.cost_reduction_factor is not None
        assert result.cost_reduction_factor > 1.0  # mini is cheaper

    def test_n_traces_correct(self):
        result = run_model_migration(n_traces=4, n_steps=4, seed=0)
        assert result.n_traces == 4

    def test_invalid_n_traces_raises(self):
        with pytest.raises(ValueError, match="n_traces"):
            run_model_migration(n_traces=0)

    def test_invalid_n_steps_raises(self):
        with pytest.raises(ValueError, match="n_steps"):
            run_model_migration(n_steps=1)

    def test_summary_line_contains_tokens(self):
        result = run_model_migration(n_traces=3, n_steps=4, seed=0)
        line = result.summary_line()
        assert "model_migration" in line
        assert "cost_red=" in line

    def test_to_json_round_trips(self):
        result = run_model_migration(n_traces=3, n_steps=4, seed=0)
        payload = json.loads(result.to_json())
        assert payload["n_traces"] == 3
        assert payload["model_a"] == "gpt-4o-2024-11-20"


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

class TestParameterSweep:
    def test_returns_result_type(self):
        result = run_parameter_sweep(n_traces=3, n_steps=4, seed=0)
        assert isinstance(result, ParameterSweepResult)

    def test_n_candidates_two(self):
        result = run_parameter_sweep(n_traces=3, n_steps=4, seed=0)
        assert result.n_candidates == 2

    def test_candidate_labels_correct(self):
        result = run_parameter_sweep(n_traces=3, n_steps=4, seed=0)
        assert "strict_system_prompt" in result.candidate_labels
        assert "critical_finding_injection" in result.candidate_labels

    def test_n_reports_equals_n_candidates(self):
        result = run_parameter_sweep(n_traces=3, n_steps=4, seed=0)
        assert len(result.reports) == result.n_candidates

    def test_reports_attempted_correct(self):
        result = run_parameter_sweep(n_traces=3, n_steps=4, seed=0)
        for report in result.reports:
            assert report.n_traces_attempted == 3

    def test_diverged_fraction_between_zero_and_one(self):
        result = run_parameter_sweep(n_traces=3, n_steps=4, seed=0)
        assert 0.0 <= result.diverged_fraction <= 1.0

    def test_mean_dirty_count_non_negative(self):
        result = run_parameter_sweep(n_traces=3, n_steps=4, seed=0)
        assert result.mean_dirty_count >= 0

    def test_dirty_per_candidate_keys(self):
        result = run_parameter_sweep(n_traces=3, n_steps=4, seed=0)
        for label in result.candidate_labels:
            assert label in result.dirty_count_per_candidate

    def test_invalid_n_traces_raises(self):
        with pytest.raises(ValueError, match="n_traces"):
            run_parameter_sweep(n_traces=0)

    def test_invalid_n_steps_raises(self):
        with pytest.raises(ValueError, match="n_steps"):
            run_parameter_sweep(n_steps=1)

    def test_summary_line_contains_tokens(self):
        result = run_parameter_sweep(n_traces=3, n_steps=4, seed=0)
        line = result.summary_line()
        assert "parameter_sweep" in line
        assert "candidates" in line
        assert "mean_dirty=" in line

    def test_to_json_round_trips(self):
        result = run_parameter_sweep(n_traces=3, n_steps=4, seed=0)
        payload = json.loads(result.to_json())
        assert payload["n_candidates"] == 2
        assert len(payload["candidate_labels"]) == 2


# ---------------------------------------------------------------------------
# Redacted publication
# ---------------------------------------------------------------------------

class TestRedactedPublication:
    def test_returns_result_type(self):
        result = run_redacted_publication()
        assert isinstance(result, RedactedPublicationResult)

    def test_n_steps_8(self):
        # 4 customers × 2 steps each = 8
        result = run_redacted_publication()
        assert result.n_steps == 8

    def test_scan_matches_positive(self):
        # The STANDARD_POLICY should flag at least the 4 email addresses
        result = run_redacted_publication()
        assert result.scan_matches > 0

    def test_redaction_matches_positive(self):
        result = run_redacted_publication()
        assert result.redaction_matches > 0

    def test_redacted_trace_verified(self):
        result = run_redacted_publication()
        assert result.redacted_trace_verified is True

    def test_attestation_verified(self):
        result = run_redacted_publication()
        assert result.attestation_verified is True

    def test_no_raw_pii_in_redacted(self):
        result = run_redacted_publication()
        assert result.no_raw_pii_in_redacted is True

    def test_known_emails_redacted_positive(self):
        result = run_redacted_publication()
        assert result.known_emails_redacted > 0

    def test_scan_matches_gte_redaction_matches(self):
        # Scan (comprehensive) should find at least as many as the actual redaction
        result = run_redacted_publication()
        assert result.scan_matches >= result.redaction_matches

    def test_summary_line_contains_tokens(self):
        result = run_redacted_publication()
        line = result.summary_line()
        assert "redacted_publication" in line
        assert "verified=" in line
        assert "no_raw_pii=" in line

    def test_to_json_round_trips(self):
        result = run_redacted_publication()
        payload = json.loads(result.to_json())
        assert payload["n_steps"] == 8
        assert payload["redacted_trace_verified"] is True
        assert payload["attestation_verified"] is True
        assert payload["no_raw_pii_in_redacted"] is True

    def test_output_dir_respected(self):
        with tempfile.TemporaryDirectory() as d:
            result = run_redacted_publication(output_dir=d)
            assert result.redacted_trace_verified is True


# ---------------------------------------------------------------------------
# Module-level imports
# ---------------------------------------------------------------------------

def test_case_studies_package_imports():
    """All six public symbols are importable from the package."""
    import stepback.case_studies as cs
    assert hasattr(cs, "run_high_volume")
    assert hasattr(cs, "run_incident_replay")
    assert hasattr(cs, "run_evidence_pack")
    assert hasattr(cs, "run_model_migration")
    assert hasattr(cs, "run_parameter_sweep")
    assert hasattr(cs, "run_redacted_publication")
