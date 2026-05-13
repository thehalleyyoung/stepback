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
    "arecord",
    "autorecord",
    "testing",
    "RecorderOptions",
    "replay",
    "replay_events",
    "Recorder",
    "RecorderKey",
    "RecordedStep",
    "Receipt",
    "StepKind",
    "TraceHeader",
    "Trace",
    "Branch",
    "BranchDiff",
    "ReplayResult",
    "Executor",
    "ReplayPlan",
    "PlannedStep",
    "StepProvenance",
    "ReplayProvenance",
    "BisectTarget",
    "BisectMultiResult",
    "MissingExecutor",
    "UnavailableExecutorError",
    "StepExecutorRequirement",
    "PartialExecutor",
    "audit_executor_requirements",
    "get_current_recorder",
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
    "FieldOutputSubstitution",
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
    # --- minimize reports (Step 87) ----------------------------------
    "MinimizeReportOptions",
    "render_html_minimize_report",
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
    "OpenAIResponsesOutput",
    "AnthropicMessage",
    "AsyncWrappedOpenAI",
    "AsyncWrappedAnthropic",
    "AzureOpenAIShimContract",
    "CohereMessage",
    "CohereShimContract",
    "GeminiResponse",
    "GeminiCandidate",
    "GeminiUsageMetadata",
    "MistralChatResponse",
    "MistralShimContract",
    "StreamedLLMResponse",
    "WrappedOpenAI",
    "WrappedAnthropic",
    "WrappedAzureOpenAI",
    "WrappedBedrock",
    "WrappedCohere",
    "WrappedGemini",
    "WrappedLangchainTool",
    "WrappedMCPSession",
    "WrappedMistral",
    "WrappedOpenAICompat",
    "wrap_openai",
    "wrap_openai_async",
    "wrap_openai_responses",
    "wrap_anthropic",
    "wrap_anthropic_async",
    "wrap_azure_openai",
    "wrap_bedrock",
    "wrap_cohere",
    "wrap_gemini",
    "wrap_mistral",
    "wrap_vertex_model",
    "wrap_langchain_tool",
    "wrap_langchain_tools",
    "wrap_mcp_session",
    "openai_executor",
    "anthropic_executor",
    "azure_openai_executor",
    "bedrock_executor",
    "cohere_executor",
    "gemini_executor",
    "langchain_tool_executor",
    "mcp_tool_executor",
    "mistral_executor",
    "canonical_azure_model_id",
    "canonical_bedrock_model_id",
    "canonical_cohere_model_id",
    "canonical_gemini_model_id",
    "canonical_mistral_model_id",
    # --- OpenAI-compatible providers (Step 96) ------------------------
    "OpenAICompatShimContract",
    "GroqShimContract",
    "TogetherShimContract",
    "FireworksShimContract",
    "CerebrasShimContract",
    "NvidiaNIMShimContract",
    "VLLMShimContract",
    "TGIShimContract",
    "LlamaCppShimContract",
    "OllamaShimContract",
    "openai_compat_executor",
    "groq_executor",
    "together_executor",
    "fireworks_executor",
    "cerebras_executor",
    "nvidia_nim_executor",
    "vllm_executor",
    "tgi_executor",
    "llamacpp_executor",
    "ollama_executor",
    "wrap_groq",
    "wrap_together",
    "wrap_fireworks",
    "wrap_cerebras",
    "wrap_nvidia_nim",
    "wrap_vllm",
    "wrap_tgi",
    "wrap_llamacpp",
    "wrap_ollama",
    # --- framework recorders (Steps 101-102) --------------------------
    "StepbackCallbackHandler",
    "langchain_callback_handler",
    "LlamaIndexCallbackHandler",
    "llamaindex_callback_handler",
    "DSPyCallbackHandler",
    "dspy_callback_handler",
    "HaystackTracer",
    "haystack_tracer",
    "AutoGenEventHandler",
    "autogen_event_handler",
    "CrewAIStepRecorder",
    "crewai_step_recorder",
    "SemanticKernelFilter",
    "semantic_kernel_filter",
    "StrandsCallbackHandler",
    "strands_callback_handler",
    "PydanticAIInstrument",
    "pydantic_ai_instrument",
    "InspectAIRecorder",
    "inspect_ai_recorder",
    # --- attestation --------------------------------------------------
    "AttestationEntry",
    "AttestationPack",
    "AttestationVerificationError",
    "build_attestation_pack",
    "read_attestation_pack",
    "verify_attestation_pack",
    "write_attestation_pack",
    # --- SLSA / in-toto provenance (Step 111) ------------------------
    "ProvenanceVerificationError",
    "INTOTO_STATEMENT_TYPE",
    "SLSA_PREDICATE_TYPE",
    "DSSE_PAYLOAD_TYPE",
    "BUILD_TYPE_TRACE",
    "BUILD_TYPE_BENCHMARK",
    "BUILD_TYPE_PACK",
    "BUILDER_ID",
    "MEDIA_TYPE_TRACE",
    "MEDIA_TYPE_BENCHMARK",
    "MEDIA_TYPE_PACK",
    "trace_provenance",
    "benchmark_provenance",
    "pack_provenance",
    "sign_provenance",
    "verify_provenance_signature",
    "sha256_of_file",
    "sha256_of_bytes",
    # --- minimize -----------------------------------------------------
    "MinimizationResult",
    "PredicateNotTriggered",
    "ddmin_substitutions",
    "BranchGroup",
    "BranchMinimizationResult",
    "identify_branch_groups",
    "minimize_branches",
    "StepAttributionResult",
    "attribute_steps",
    "TraceObjectives",
    "ParetoEntry",
    "MultiObjectiveMinimizationResult",
    "MultiObjectiveDDMinStrategy",
    "extract_objectives",
    "multi_objective_minimize",
    "minimize_imported_trace",
    # --- typed predicate DSL (Step 80) --------------------------------
    "TypedPredicate",
    "threshold",
    "regex_match",
    "regex_search",
    "policy_check",
    "callback",
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
    "DirtySetEntry",
    "DirtySetSummary",
    "compute_dirty_set",
    "DIVERGENCE_SEVERITY",
    "DIVERGENCE_SEVERITY_WEIGHT",
    "DIVERGENCE_VOLATILE_KEYS",
    "compare_outputs",
    "detect_divergences",
    # --- distributed dirty-set (Step 64) -----------------------------
    "DagRegion",
    "RegionSummary",
    "compute_dirty_set_distributed",
    "partition_dag_regions",
    # --- distributed bisect (Step 140) --------------------------------
    "DistributedBisectItem",
    "DistributedBisectOptions",
    "DistributedBisectResult",
    "DistributedBisectSummary",
    "DistributedMultiObjectiveResult",
    "DistributedMultiObjectiveSummary",
    "distributed_bisect",
    "distributed_multi_objective_bisect",
    # --- ClickHouse analytics (Step 65) ------------------------------
    "ClickHouseAnalytics",
    "DIRTY_SET_TABLE_DDL",
    "DIRTY_SET_REPLICATED_TABLE_DDL",
    "DIRTY_SET_TABLE_NAME",
    "DirtySetRecord",
    "record_from_summary",
    # --- step 139: ClickHouse step schema ----------------------------
    "TraceStepRecord",
    "TRACE_STEP_TABLE_DDL",
    "TRACE_STEP_REPLICATED_TABLE_DDL",
    "TRACE_STEP_TABLE_NAME",
    "step_record_from_dirty_entry",
    "TraceStepAnalytics",
    # --- nondeterminism (Step 63) ------------------------------------
    "NondeterminismClass",
    "nondeterminism_forces_dirty",
    "clock_nondeterminism",
    "rng_nondeterminism",
    "env_nondeterminism",
    "network_nondeterminism",
    "model_sampling_nondeterminism",
    "combine_nondeterminism",
    # --- deterministic seeding (Step 69) -----------------------------
    "SeedSupport",
    "SeedWarnLevel",
    "SeedPolicyViolation",
    "SeedPolicyError",
    "SeedPolicy",
    "PROVIDER_SEED_SUPPORT",
    "DEFAULT_SEED_POLICY",
    "get_seed_policy",
    "set_seed_policy",
    # --- exporters ----------------------------------------------------
    "ExportError",
    "ExportReport",
    "TraceExportError",
    "available_export_formats",
    "export_cyclonedx_ai",
    "export_langsmith_jsonl",
    "export_native_json",
    "validate_native_json_doc",
    "export_html_view",
    "export_openai_chat_log",
    "export_openinference_spans",
    "export_otel_spans",
    "export_trace",
    "export_trace_file",
    # --- importers ----------------------------------------------------
    "ImportReport",
    "LossReport",
    "TraceImportError",
    "import_datadog_apm",
    "import_helicone_log",
    "import_langfuse_export",
    "import_langsmith_jsonl",
    "import_native_json",
    "import_openai_chat_log",
    "import_openinference_spans",
    "import_otel_spans",
    "import_phoenix_spans",
    "import_trace",
    # --- CycloneDX-AI importer (Step 150) ----------------------------
    "import_cyclonedx_ai",
    # --- OtelBridge (Step 150) ----------------------------------------
    "OtelBridge",
    "OtelBridgeError",
    "OtelBridgeExportResult",
    "OtlpHttpExporter",
    "DryRunCollector",
    # --- html view ----------------------------------------------------
    "TraceViewSummary",
    "render_trace_html",
    "write_trace_html",
    "TimeTravelSummary",
    "render_time_travel_html",
    "write_time_travel_html",
    "FullReportSummary",
    "render_full_html_report",
    "write_full_html_report",
    # --- SB-Trace wire-format SemVer (Step 22) ----------------------
    "SBTRACE_WIRE_VERSION",
    "SBTRACE_WIRE_VERSION_INFO",
    "SBTRACE_WIRE_MAJOR",
    "SBTRACE_WIRE_MINOR",
    "SBTRACE_WIRE_PATCH",
    "SBTRACE_WIRE_ENCODING",
    "SBTRACE_WIRE_ENCODINGS",
    "SBTRACE_FORMAT_VERSION_TO_WIRE",
    "SBTRACE_FORMAT_VERSION_TO_ENCODINGS",
    "SBTraceVersionError",
    "parse_wire_version",
    "is_compatible_reader",
    "wire_version_for_format_version",
    "format_version_for_wire_version",
    # --- SBTraceSpec versioned schema validator (Step 23) -----------
    "SBTraceSpec",
    "SBTraceConformanceError",
    "ConformanceIssue",
    "ConformanceReport",
    "current_spec",
    # --- deprecation policy (Step 21) --------------------------------
    "DeprecationPolicyError",
    "deprecated",
    "deprecated_alias",
    "format_deprecation_message",
    "warn_deprecated",
    # --- sandboxed replay (Step 72) -----------------------------------
    "SandboxMode",
    "SandboxConfig",
    "SandboxError",
    "SandboxTimeoutError",
    "SandboxResourceError",
    "SandboxUnavailableError",
    "SandboxViolationError",
    "SandboxedExecutor",
    "create_sandbox",
    # --- sharded step cache (Step 74) --------------------------------
    "StepCacheEntry",
    "StepCache",
    "DiskStepCache",
    "S3StepCache",
    "GCSStepCache",
    "AzureStepCache",
    "NamespacedStepCache",
    "MultiTierStepCache",
    # --- event bus + idempotency (Step 75) ----------------------------
    "EventKind",
    "ReplayEvent",
    "EventBus",
    "NullEventBus",
    "InMemoryEventBus",
    "KafkaEventBus",
    "IdempotencyStatus",
    "IdempotencyRegistry",
    "InMemoryIdempotencyRegistry",
    "compute_replay_idempotency_key",
    "execute_with_idempotency",
    # --- worker leases + checkpointed sweep (Step 76) -----------------
    "LeaseStatus",
    "WorkerLease",
    "LeaseExpiredError",
    "LeaseRegistry",
    "InMemoryLeaseRegistry",
    "DiskLeaseRegistry",
    "CheckpointEntryStatus",
    "CheckpointEntry",
    "SweepCheckpoint",
    "DiskSweepCheckpoint",
    "resume_sweep",
    # --- statistical stability metrics (Step 82) ----------------------
    "FlakyClass",
    "FlakyPredicateWarning",
    "StabilityConfig",
    "StabilityResult",
    "measure_predicate_stability",
    # --- stepback.toml config loader (Step 160) ----------------------
    "ConfigError",
    "TraceConfig",
    "KeyConfig",
    "ShimConfig",
    "RedactionConfig",
    "PriceListConfig",
    "ViewerConfig",
    "StepbackConfig",
    "DEFAULT_CONFIG",
    "ENV_VAR_MAP",
    "SECRET_KEYS",
    "find_config_file",
    "load_config",
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


def test_trace_last_bisect_probes_is_public_read_only() -> None:
    """``Trace.last_bisect_probes`` is a deliberately-public read-only metric.

    Step 17 promoted this from a raw dataclass field to a documented
    property backed by ``_last_bisect_probes``. The contract:

    * read access via attribute syntax keeps working
    * the property has a non-empty docstring
    * the storage attribute is underscore-prefixed (i.e. internal)
    * fresh traces start at zero
    """
    prop = inspect.getattr_static(stepback.Trace, "last_bisect_probes")
    assert isinstance(prop, property), (
        "Trace.last_bisect_probes must be a property, not a raw dataclass "
        "field; see Step 17 in docs/100_STEPS.md."
    )
    assert prop.fget is not None and prop.fget.__doc__, (
        "Trace.last_bisect_probes property must carry a docstring."
    )
    # The internal storage slot is the underscore-prefixed twin.
    assert "_last_bisect_probes" in stepback.Trace.__dataclass_fields__, (
        "Trace must store bisect probe count in _last_bisect_probes."
    )
    assert "last_bisect_probes" not in stepback.Trace.__dataclass_fields__, (
        "Trace.last_bisect_probes must not be a raw dataclass field anymore."
    )
