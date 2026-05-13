"""Production-shaped case studies demonstrating stepback at scale.

Six scenarios covering the full operational spectrum of AI-agent observability:

* :mod:`~stepback.case_studies.high_volume` — recording throughput at
  millions-of-runs/day scale with backpressure, signing, and storage metrics.
* :mod:`~stepback.case_studies.incident_replay` — isolating the root cause
  of a production incident via bisection and substitution.
* :mod:`~stepback.case_studies.evidence_packs` — building signed
  regulator/auditor evidence packs with HMAC-chained attestation.
* :mod:`~stepback.case_studies.model_migration` — quantifying dirty-set
  impact and cost reduction when migrating between LLM models.
* :mod:`~stepback.case_studies.parameter_sweep` — sweeping a grid of
  substitution candidates over a trace corpus.
* :mod:`~stepback.case_studies.redacted_publication` — redacting PII from
  production traces before external publication.

All case studies are fully offline (no network calls) and deterministic
(seeded RNG) so they run in CI without API keys.  They are designed to be
imported as library code and also serve as narrative documentation.

Typical use::

    from stepback.case_studies.high_volume import run_high_volume, HighVolumeResult
    result = run_high_volume(n_traces=50)
    print(result.projected_runs_per_day)
"""

from .high_volume import HighVolumeResult, run_high_volume
from .incident_replay import IncidentReplayResult, run_incident_replay
from .evidence_packs import EvidencePackResult, run_evidence_pack
from .model_migration import ModelMigrationResult, run_model_migration
from .parameter_sweep import ParameterSweepResult, run_parameter_sweep
from .redacted_publication import RedactedPublicationResult, run_redacted_publication

__all__ = [
    "HighVolumeResult",
    "run_high_volume",
    "IncidentReplayResult",
    "run_incident_replay",
    "EvidencePackResult",
    "run_evidence_pack",
    "ModelMigrationResult",
    "run_model_migration",
    "ParameterSweepResult",
    "run_parameter_sweep",
    "RedactedPublicationResult",
    "run_redacted_publication",
]
