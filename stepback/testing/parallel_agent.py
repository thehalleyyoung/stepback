"""Fan-out / fan-in fixture: a research agent that runs three parallel
sub-investigations and merges the answers.

Recorded shape (sequential step ids; the parallel frames mark the
fan-out so the replay engine knows the branches are siblings)::

    step:1  llm_call            "plan"
    step:2  tool_call            "split_question"
    step:3  parallel_branch_open "research_fanout"  parent=step:2
    step:4  llm_call             parent=step:3   (branch A: history)
    step:5  tool_call            parent=step:4   (branch A: lookup_history)
    step:6  llm_call             parent=step:3   (branch B: economics)
    step:7  tool_call            parent=step:6   (branch B: lookup_econ)
    step:8  llm_call             parent=step:3   (branch C: politics)
    step:9  tool_call            parent=step:8   (branch C: lookup_polit)
    step:10 parallel_branch_join parent=step:3
            parent_step_ids=[step:5, step:7, step:9]
    step:11 llm_call            "synthesise"     parent=step:10

This module previously lived under ``tests.fixtures.parallel_agent`` and
is now promoted to ``stepback.testing.parallel_agent`` so downstream
users can drive a realistic fan-out/fan-in pipeline through the
recorder without depending on the test tree.
"""
from __future__ import annotations

import hashlib
from typing import Any, List


def _digest(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:8]


def fake_llm(model: str, messages: List[dict]) -> dict:
    """Deterministic LLM stand-in for the parallel research fixture."""
    blob = "\n".join(f"{m['role']}:{m['content']}" for m in messages)
    digest = _digest(blob)
    last = messages[-1]["content"] if messages else ""
    text = f"reply-{digest} echo:{last[:40]}"
    return {
        "id": f"chatcmpl-{digest}",
        "model": model,
        "choices": [
            {"index": 0, "finish_reason": "stop",
             "message": {"role": "assistant", "content": text}}
        ],
        "usage": {
            "prompt_tokens": sum(len(m["content"]) for m in messages),
            "completion_tokens": len(text),
            "total_tokens": sum(len(m["content"]) for m in messages) + len(text),
        },
    }


# Recorded "facts": each lookup tool returns a stable answer per topic.
FACTS = {
    "history":   {"answer": "founded 1789", "confidence": 0.9},
    "economics": {"answer": "GDP 21T USD",  "confidence": 0.7},
    "politics":  {"answer": "two-party",    "confidence": 0.8},
}


def fake_tool(name: str, args: dict) -> Any:
    """Tool stand-ins for the parallel research fixture."""
    if name == "split_question":
        return {"topics": list(FACTS.keys())}
    if name.startswith("lookup_"):
        topic = name[len("lookup_"):]
        return FACTS[topic]
    raise KeyError(f"unknown fake tool: {name!r}")


def _make_branch(topic: str):
    """Closure that records one llm_call + one tool_call for ``topic``."""
    def branch(rec):
        rec.llm_call(
            "gpt-4o-2024-11-20",
            [
                {"role": "system", "content": f"You research {topic}."},
                {"role": "user",   "content": f"What is the latest on {topic}?"},
            ],
            executor=fake_llm,
        )
        rec.tool_call(f"lookup_{topic}", {"topic": topic}, executor=fake_tool)
    return branch


def run_parallel_agent(rec) -> dict:
    """Drive an 11-step research agent through ``rec``.

    Returns the final synthesise step.
    """
    convo = [
        {"role": "system", "content": "You are a research agent."},
        {"role": "user",   "content": "Summarise the United States."},
    ]
    rec.llm_call("gpt-4o-2024-11-20", convo, executor=fake_llm)
    rec.tool_call("split_question", {"q": convo[-1]["content"]}, executor=fake_tool)

    join_step = rec.parallel(
        "research_fanout",
        [_make_branch(t) for t in FACTS.keys()],
        join=lambda outs: {
            "merged": [o.get("result", o) for o in outs],
            "n": len(outs),
        },
        branch_names=[f"research/{t}" for t in FACTS.keys()],
    )

    # Synthesise: messages incorporate the merged join output so the
    # synthesise call genuinely depends on the join's output (and
    # therefore on every branch tail). This is what makes the
    # dirty-set propagate through the join into the downstream call.
    merged_summary = "; ".join(
        f"{m.get('answer', '?')}" for m in join_step["outputs"]["merged"]
    )
    return rec.llm_call(
        "gpt-4o-2024-11-20",
        convo + [
            {"role": "assistant", "content": "synthesising"},
            {"role": "user",      "content": f"merged: {merged_summary}"},
        ],
        executor=fake_llm,
    )
