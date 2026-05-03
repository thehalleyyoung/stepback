"""stepback — time-travel debugger for AI agents.

Public surface for v0.1::

    from stepback import record, replay
    from stepback.substitutions import (
        PromptSubstitution, ToolOutputSubstitution,
        ModelSubstitution, PolicySubstitution, RouterSubstitution,
    )

`record(path)` is a context manager returning a `Recorder` whose
`llm_call(...)` and `tool_call(...)` methods append signed,
HMAC-chained frames to an append-only `.sb` file.

`replay(path)` returns a `Trace` with reversible step navigation
(`step_back`, `goto`), typed `substitute(...)`, `replay_forward(...)`,
`bisect(...)`, `branch_at(...)`, and `compare_branches(...)`.
"""
from .recorder import record, Recorder, RecorderKey
from . import autorecord
from .replay import replay, Trace, Branch, BranchDiff, ReplayResult, Executor
from .minimize import (
    BinaryHalvingStrategy,
    BruteForceStrategy,
    BudgetExhausted,
    DDMinStrategy,
    LinearShrinkStrategy,
    MinimizationResult,
    MinimizeOptions,
    PredicateNotTriggered,
    ShapleyAttributionStrategy,
    Strategy,
    attribute_substitutions,
    ddmin_substitutions,
    find_all_minimal,
    minimize_substitutions,
)
from . import predicates
from .attestation import (
    AttestationEntry,
    AttestationPack,
    AttestationVerificationError,
    build_attestation_pack,
    read_attestation_pack,
    verify_attestation_pack,
    write_attestation_pack,
)
from .branch_io import (
    BranchTraceMismatch,
    diff_replays,
    load_branch,
    parse_substitution_spec,
    save_branch,
    trace_chain_hash,
)
from .substitutions import (
    InputsPatchSubstitution,
    MessagePatchSubstitution,
    ModelSubstitution,
    OutputsPatchSubstitution,
    PolicySubstitution,
    PromptSubstitution,
    RaiseSubstitution,
    RouterSubstitution,
    SamplingSubstitution,
    Substitution,
    SubstitutionSet,
    SystemPromptSubstitution,
    ToolArgumentsSubstitution,
    ToolOutputSubstitution,
)
from .jsonpatch import (
    PatchError,
    PatchInvalidOp,
    PatchPathNotFound,
    PatchTestFailed,
    apply_patch,
)
from .trace_diff import (
    CrossTraceDiff,
    StepPair,
    diff_traces,
    render_trace_diff,
)
from .policy_audit import (
    PolicyImpactReport,
    StepImpact,
    TraceImpact,
    audit_policy_change,
    is_policy_blocked,
)
from .report import (
    ReportOptions,
    render_counterfactual_report,
    render_replay_report,
    render_report_json,
    dump_report_json,
)
from .shims import (
    AnthropicMessage,
    OpenAIChatCompletion,
    WrappedAnthropic,
    WrappedLangchainTool,
    WrappedMCPSession,
    WrappedOpenAI,
    anthropic_executor,
    langchain_tool_executor,
    mcp_tool_executor,
    openai_executor,
    wrap_anthropic,
    wrap_langchain_tool,
    wrap_langchain_tools,
    wrap_mcp_session,
    wrap_openai,
)

__all__ = [
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
    "BranchTraceMismatch",
    "diff_replays",
    "load_branch",
    "parse_substitution_spec",
    "save_branch",
    "trace_chain_hash",
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
    "apply_patch",
    "PatchError",
    "PatchInvalidOp",
    "PatchPathNotFound",
    "PatchTestFailed",
    "ReportOptions",
    "render_counterfactual_report",
    "render_replay_report",
    "render_report_json",
    "dump_report_json",
    "CrossTraceDiff",
    "StepPair",
    "diff_traces",
    "render_trace_diff",
    "PolicyImpactReport",
    "TraceImpact",
    "StepImpact",
    "audit_policy_change",
    "is_policy_blocked",
    "OpenAIChatCompletion",
    "AnthropicMessage",
    "WrappedOpenAI",
    "WrappedAnthropic",
    "WrappedLangchainTool",
    "WrappedMCPSession",
    "wrap_openai",
    "wrap_anthropic",
    "wrap_langchain_tool",
    "wrap_langchain_tools",
    "wrap_mcp_session",
    "openai_executor",
    "anthropic_executor",
    "langchain_tool_executor",
    "mcp_tool_executor",
    "AttestationEntry",
    "AttestationPack",
    "AttestationVerificationError",
    "build_attestation_pack",
    "read_attestation_pack",
    "verify_attestation_pack",
    "write_attestation_pack",
    "MinimizationResult",
    "PredicateNotTriggered",
    "ddmin_substitutions",
]

__version__ = "0.1.0"
