"""RAG pipeline agent.

Simulates a retrieve → re-rank → synthesise RAG workflow:

  1. LLM decomposes the question into sub-queries.
  2. ``retrieve`` tool fetches candidate passages for each sub-query.
  3. ``rerank`` tool scores the candidates and returns the top-k.
  4. LLM synthesises a final answer from the top-k passages.
  5. ``grade_answer`` tool applies an automated quality check.
  6. LLM refines the answer if the grade is below threshold.

The fake LLM and tools are fully deterministic so no API keys are needed.

Demonstrates:
- Recording a retrieval-augmented pipeline.
- Substituting the retrieval result to inject a different passage set.
- Observing that the re-rank, synthesis, and grading steps all become dirty.

Run directly::

    python examples/rag_pipeline/agent.py

Or via Make::

    make -C examples rag_pipeline
"""
from __future__ import annotations

import hashlib
import os
import sys
from typing import Any, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from stepback import Executor, record, replay
from stepback.substitutions import ToolOutputSubstitution

TRACE_PATH = os.path.join(os.path.dirname(__file__), "trace.sb")

# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

_PASSAGES = {
    "history": [
        {"id": "h1", "text": "The company was founded in 1982 by three engineers.", "score": 0.92},
        {"id": "h2", "text": "It went public on the NYSE in 2001.", "score": 0.87},
        {"id": "h3", "text": "Revenue reached $1B in 2010.", "score": 0.81},
    ],
    "products": [
        {"id": "p1", "text": "The flagship product is the Model-X widget.", "score": 0.95},
        {"id": "p2", "text": "Model-Y launched in 2020 and outsold Model-X by 2022.", "score": 0.89},
        {"id": "p3", "text": "The accessories line accounts for 12 % of revenue.", "score": 0.75},
    ],
    "leadership": [
        {"id": "l1", "text": "Alice Chen has served as CEO since 2015.", "score": 0.93},
        {"id": "l2", "text": "The board has seven members, four of whom are independent.", "score": 0.80},
    ],
}

# Alternative passages injected by the substitution demo.
_ALT_PASSAGES = {
    "history": [
        {"id": "h1-alt", "text": "The company was founded in 1995 as a spin-off.", "score": 0.88},
    ],
    "products": _PASSAGES["products"],
    "leadership": _PASSAGES["leadership"],
}


def _digest(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Fake LLM
# ---------------------------------------------------------------------------

def fake_llm(model: str, messages: List[dict]) -> dict:
    """Deterministic LLM stand-in."""
    blob = "\n".join(f"{m['role']}:{m.get('content','')}" for m in messages)
    digest = _digest(blob)
    last = messages[-1].get("content", "") if messages else ""

    if "sub-quer" in last.lower() or "decompose" in last.lower():
        text = f"Sub-queries: history, products, leadership ({digest})"
    elif "synthesise" in last.lower() or "synthesize" in last.lower() or "answer" in last.lower():
        text = f"Based on the passages, the company was founded in 1982. ({digest})"
    elif "refine" in last.lower() or "improve" in last.lower():
        text = f"Refined answer: The company was founded in 1982 by three engineers. ({digest})"
    else:
        text = f"reply-{digest}"

    return {
        "id": f"chatcmpl-{digest}",
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": text}}],
        "usage": {
            "prompt_tokens": sum(len(m.get("content", "")) for m in messages),
            "completion_tokens": len(text),
            "total_tokens": sum(len(m.get("content", "")) for m in messages) + len(text),
        },
    }


# ---------------------------------------------------------------------------
# Fake tools
# ---------------------------------------------------------------------------

def fake_tool(name: str, args: dict) -> Any:
    """Tool stand-ins for the RAG pipeline."""
    if name == "retrieve":
        topic = args.get("topic", "history")
        return {"passages": _PASSAGES.get(topic, []), "topic": topic}

    if name == "rerank":
        passages = args.get("passages", [])
        top_k = args.get("top_k", 2)
        ranked = sorted(passages, key=lambda p: p.get("score", 0), reverse=True)
        return {"ranked": ranked[:top_k]}

    if name == "grade_answer":
        answer = args.get("answer", "")
        score = min(1.0, len(answer) / 120)
        return {"grade": round(score, 2), "needs_refinement": score < 0.7}

    raise KeyError(f"unknown RAG tool: {name!r}")


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

def run_rag_agent(rec: Any) -> None:
    """Drive a 6-step RAG pipeline through *rec*."""
    model = "gpt-4o-2024-11-20"
    question = "Summarise the company's history, products, and leadership."
    convo = [
        {"role": "system", "content": "You are a research assistant."},
        {"role": "user", "content": question},
    ]

    # 1) LLM decomposes question into sub-queries.
    rec.llm_call(model, convo + [{"role": "user", "content": "decompose into sub-queries"}],
                 executor=fake_llm)

    # 2) Retrieve passages for each topic (one tool call per topic).
    all_passages = []
    for topic in ("history", "products", "leadership"):
        result = rec.tool_call("retrieve", {"topic": topic, "k": 3}, executor=fake_tool)
        all_passages.extend(result["outputs"]["result"]["passages"])

    # 3) Re-rank all candidates to top-4.
    ranked = rec.tool_call("rerank", {"passages": all_passages, "top_k": 4},
                           executor=fake_tool)

    top_passages_text = " | ".join(
        p["text"] for p in ranked["outputs"]["result"]["ranked"]
    )
    convo.append({"role": "assistant",
                  "content": f"Top passages: {top_passages_text}"})

    # 4) LLM synthesises answer.
    synth = rec.llm_call(
        model,
        convo + [{"role": "user",
                  "content": f"synthesise an answer from: {top_passages_text}"}],
        executor=fake_llm,
    )
    answer = synth["outputs"]["choices"][0]["message"]["content"]

    # 5) Grade the answer.
    grade_result = rec.tool_call("grade_answer", {"answer": answer, "question": question},
                                 executor=fake_tool)
    grade = grade_result["outputs"]["result"]

    # 6) LLM refines if needed.
    rec.llm_call(
        model,
        convo + [
            {"role": "assistant", "content": answer},
            {"role": "user",
             "content": (
                 f"Grade: {grade['grade']:.2f}. "
                 + ("Please refine." if grade["needs_refinement"] else "Answer accepted.")
             )},
        ],
        executor=fake_llm,
    )


# ---------------------------------------------------------------------------
# Record & replay demo
# ---------------------------------------------------------------------------

def record_trace(path: str = TRACE_PATH) -> None:
    """Write the reference RAG trace to *path*."""
    with record(path) as rec:
        run_rag_agent(rec)
    print(f"  recorded → {path}")


def replay_demo(path: str = TRACE_PATH) -> None:
    """Substitute the 'history' retrieval result and observe dirty propagation."""
    trace = replay(path)
    steps = trace.recorded_steps

    # Find the first retrieve step (topic=history).
    retrieve_step = next(
        s for s in steps
        if s.get("name") == "retrieve"
        and s.get("inputs", {}).get("arguments", {}).get("topic") == "history"
    )

    alt_output = {"passages": _ALT_PASSAGES["history"], "topic": "history"}
    trace.substitute(
        ToolOutputSubstitution(
            at_step=retrieve_step["step_id"],
            fake_response=alt_output,
        )
    )

    result = trace.replay_forward(
        Executor(llm=fake_llm, tool=fake_tool)
    )
    dirty = [sv for sv in result.steps if sv.dirty]
    print(f"  dirty steps after retrieval substitution: {len(dirty)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== rag_pipeline example ===")
    print("Recording…")
    record_trace()
    print("Replaying with substitution…")
    replay_demo()
    print("Done.")
