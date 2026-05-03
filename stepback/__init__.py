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
from .replay import replay, Trace, Branch, BranchDiff, ReplayResult, Executor
from .branch_io import (
    BranchTraceMismatch,
    diff_replays,
    load_branch,
    parse_substitution_spec,
    save_branch,
    trace_chain_hash,
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
    "ReportOptions",
    "render_counterfactual_report",
    "render_replay_report",
    "render_report_json",
    "dump_report_json",
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
]

__version__ = "0.1.0"
