"""Snapshot tests for the public API surface re-exported by ``stepback``.

These tests are the semver tripwire promised by the README: any silent
rename, removal, or unintentional addition to ``stepback.__all__`` will
cause this suite to fail and force the change to be acknowledged in a
PR review.

If you intentionally add or remove a public symbol, update
``EXPECTED_PUBLIC_API`` below in the same commit, and consider whether a
``DeprecationWarning`` was needed (see ``docs/deprecation.md`` once it
lands).
"""
from __future__ import annotations

import inspect

import stepback


# -----------------------------------------------------------------------
# Snapshot of the public surface as of v0.1.x.
# -----------------------------------------------------------------------
EXPECTED_PUBLIC_API = frozenset({
    # --- core record/replay ------------------------------------------
    "record",
    "autorecord",
    "replay",
    "Recorder",
    "RecorderKey",
    "Trace",
    "Branch",
    "BranchDiff",
    "ReplayResult",
    "Executor",
    # --- branch I/O ---------------------------------------------------
    "BranchTraceMismatch",
    "diff_replays",
    "load_branch",
    "parse_substitution_spec",
    "save_branch",
    "trace_chain_hash",
    # --- substitutions ------------------------------------------------
    "Substitution",
    "SubstitutionSet",
    "PromptSubstitution",
    "ModelSubstitution",
    "ToolOutputSubstitution",
    "PolicySubstitution",
    "RouterSubstitution",
    "SystemPromptSubstitution",
    "MessagePatchSubstitution",
    "SamplingSubstitution",
    "ToolArgumentsSubstitution",
    "InputsPatchSubstitution",
    "OutputsPatchSubstitution",
    "RaiseSubstitution",
    # --- jsonpatch ----------------------------------------------------
    "apply_patch",
    "PatchError",
    "PatchInvalidOp",
    "PatchPathNotFound",
    "PatchTestFailed",
    # --- reports ------------------------------------------------------
    "ReportOptions",
    "render_counterfactual_report",
    "render_replay_report",
    "render_report_json",
    "dump_report_json",
    # --- diffs --------------------------------------------------------
    "CrossTraceDiff",
    "StepPair",
    "diff_traces",
    "render_trace_diff",
    # --- policy audit -------------------------------------------------
    "PolicyImpactReport",
    "TraceImpact",
    "StepImpact",
    "audit_policy_change",
    "is_policy_blocked",
    # --- shims --------------------------------------------------------
    "OpenAIChatCompletion",
    "AnthropicMessage",
    "GeminiResponse",
    "GeminiCandidate",
    "GeminiUsageMetadata",
    "WrappedOpenAI",
    "WrappedAnthropic",
    "WrappedBedrock",
    "WrappedGemini",
    "WrappedLangchainTool",
    "WrappedMCPSession",
    "wrap_openai",
    "wrap_anthropic",
    "wrap_bedrock",
    "wrap_gemini",
    "wrap_vertex_model",
    "wrap_langchain_tool",
    "wrap_langchain_tools",
    "wrap_mcp_session",
    "openai_executor",
    "anthropic_executor",
    "bedrock_executor",
    "gemini_executor",
    "langchain_tool_executor",
    "mcp_tool_executor",
    "canonical_bedrock_model_id",
    "canonical_gemini_model_id",
    # --- attestation --------------------------------------------------
    "AttestationEntry",
    "AttestationPack",
    "AttestationVerificationError",
    "build_attestation_pack",
    "read_attestation_pack",
    "verify_attestation_pack",
    "write_attestation_pack",
    # --- minimize -----------------------------------------------------
    "MinimizationResult",
    "PredicateNotTriggered",
    "ddmin_substitutions",
    # --- sweep --------------------------------------------------------
    "DistStats",
    "SweepFailure",
    "SweepResult",
    "SweepReport",
    "sweep_traces",
    "render_sweep_report",
    "render_sweep_report_json",
    # --- divergence ---------------------------------------------------
    "Divergence",
    "DivergenceReport",
    "DIVERGENCE_SEVERITY",
    "DIVERGENCE_SEVERITY_WEIGHT",
    "DIVERGENCE_VOLATILE_KEYS",
    "compare_outputs",
    "detect_divergences",
    # --- exporters ----------------------------------------------------
    "ExportError",
    "ExportReport",
    "TraceExportError",
    "available_export_formats",
    "export_langsmith_jsonl",
    "export_openai_chat_log",
    "export_openinference_spans",
    "export_trace",
    "export_trace_file",
    # --- importers ----------------------------------------------------
    "ImportReport",
    "TraceImportError",
    "import_langsmith_jsonl",
    "import_openai_chat_log",
    "import_openinference_spans",
    "import_trace",
    # --- html view ----------------------------------------------------
    "TraceViewSummary",
    "render_trace_html",
    "write_trace_html",
})


def test_public_api_snapshot_matches() -> None:
    """``stepback.__all__`` must match the snapshot exactly.

    Any drift forces an explicit acknowledgement in this file.
    """
    actual = frozenset(stepback.__all__)
    added = sorted(actual - EXPECTED_PUBLIC_API)
    removed = sorted(EXPECTED_PUBLIC_API - actual)
    assert not added and not removed, (
        f"Public API drift detected.\n"
        f"  Added (newly public):   {added}\n"
        f"  Removed (was public):   {removed}\n"
        f"Update tests/test_public_api.py::EXPECTED_PUBLIC_API in the same "
        f"commit and decide whether a DeprecationWarning is required."
    )


def test_all_is_sorted_or_at_least_unique() -> None:
    """``__all__`` must contain no duplicates."""
    assert len(stepback.__all__) == len(set(stepback.__all__)), (
        "stepback.__all__ contains duplicate entries"
    )


def test_every_public_symbol_is_importable() -> None:
    """Every name in ``__all__`` must resolve via attribute access."""
    missing = [n for n in stepback.__all__ if not hasattr(stepback, n)]
    assert not missing, f"__all__ lists names not exported: {missing}"


def test_every_public_symbol_has_a_docstring() -> None:
    """Every public symbol must carry a docstring (Step 11).

    This is enforced for the v0.1 public surface so downstream users get
    useful ``help(stepback.X)`` output and IDE hovers without having to
    read the source.
    """
    undocumented: list[str] = []
    for name in stepback.__all__:
        obj = getattr(stepback, name)
        doc = inspect.getdoc(obj)
        if not doc or len(doc.strip()) < 3:
            undocumented.append(name)
    assert not undocumented, (
        f"Public symbols missing docstrings: {undocumented}. "
        "Add a one-line docstring to the underlying definition."
    )
