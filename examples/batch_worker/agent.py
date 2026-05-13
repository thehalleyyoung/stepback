"""Long-running batch worker agent.

Simulates a content-moderation pipeline that processes a queue of five user
posts.  For each post the agent:

  1. Calls the LLM to generate a moderation decision.
  2. Calls ``classify_post`` to apply rule-based category labels.
  3. Calls ``record_decision`` to persist the outcome.

The full trace therefore has 15 steps (3 per post × 5 posts).

Demonstrates:
- Recording a batch/loop trace (multiple identical-shape step triples).
- Substituting the LLM decision for one post (e.g. flip "approve" to "reject")
  and observing that only the two downstream steps for that post become dirty.
- Sweeping a PromptSubstitution across all LLM calls in one pass.

Run directly::

    python examples/batch_worker/agent.py

Or via Make::

    make -C examples batch_worker
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
# Job queue
# ---------------------------------------------------------------------------

_POSTS = [
    {"post_id": "p1", "text": "Great product! Highly recommend."},
    {"post_id": "p2", "text": "This is spam — buy cheap watches now!"},
    {"post_id": "p3", "text": "I hate everything about this."},
    {"post_id": "p4", "text": "Neutral review: works as described."},
    {"post_id": "p5", "text": "Amazing! Five stars!"},
]

_SCRIPTED_DECISIONS = {
    "p1": "approve",
    "p2": "reject",
    "p3": "review",
    "p4": "approve",
    "p5": "approve",
}

_CATEGORY_LABELS = {
    "p1": ["positive_sentiment"],
    "p2": ["spam", "commercial"],
    "p3": ["negative_sentiment", "potential_violation"],
    "p4": ["neutral"],
    "p5": ["positive_sentiment"],
}


def _digest(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Fake LLM
# ---------------------------------------------------------------------------

def fake_llm(model: str, messages: List[dict]) -> dict:
    """Deterministic LLM: returns a scripted decision based on post_id."""
    blob = "\n".join(f"{m['role']}:{m.get('content','')}" for m in messages)
    digest = _digest(blob)

    # Extract post_id from the last user message if present.
    last = messages[-1].get("content", "") if messages else ""
    decision = "approve"
    for post_id, d in _SCRIPTED_DECISIONS.items():
        if post_id in last:
            decision = d
            break

    text = f"Decision: {decision}. Ref: {digest}"
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
    """Tool stand-ins for the batch worker."""
    if name == "classify_post":
        post_id = args.get("post_id", "")
        return {
            "post_id": post_id,
            "labels": _CATEGORY_LABELS.get(post_id, ["unknown"]),
            "confidence": 0.95,
        }

    if name == "record_decision":
        return {
            "post_id": args.get("post_id", ""),
            "decision": args.get("decision", ""),
            "stored": True,
            "record_id": f"rec-{_digest(str(args))}",
        }

    raise KeyError(f"unknown batch tool: {name!r}")


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

def run_batch_agent(rec: Any) -> None:
    """Drive a 15-step batch moderation agent through *rec*."""
    model = "gpt-4o-2024-11-20"
    system_msg = {"role": "system",
                  "content": "You are a content moderation agent. Reply with approve/reject/review."}

    for post in _POSTS:
        post_id = post["post_id"]
        user_msg = {"role": "user",
                    "content": f"Moderate post {post_id}: {post['text']}"}

        # 1) LLM decision.
        llm_result = rec.llm_call(model, [system_msg, user_msg], executor=fake_llm)
        decision_text = llm_result["outputs"]["choices"][0]["message"]["content"]
        decision = (
            "reject" if "reject" in decision_text.lower()
            else "review" if "review" in decision_text.lower()
            else "approve"
        )

        # 2) Classify.
        rec.tool_call("classify_post", {"post_id": post_id, "text": post["text"]},
                      executor=fake_tool)

        # 3) Record decision.
        rec.tool_call("record_decision",
                      {"post_id": post_id, "decision": decision},
                      executor=fake_tool)


# ---------------------------------------------------------------------------
# Record & replay demo
# ---------------------------------------------------------------------------

def record_trace(path: str = TRACE_PATH) -> None:
    with record(path) as rec:
        run_batch_agent(rec)
    print(f"  recorded → {path}")


def replay_demo(path: str = TRACE_PATH) -> None:
    """Substitute the LLM decision for p2 (spam → approve) and check dirty set."""
    trace = replay(path)
    steps = trace.recorded_steps

    # Find the LLM step for post p2 (messages contain "p2").
    p2_llm_step = next(
        s for s in steps
        if s.get("step_kind") == "llm_call"
        and "p2" in str(s.get("inputs", {}).get("messages", ""))
    )

    # Flip the decision: override the LLM output for p2 to say "approve".
    approve_output = {
        "id": "chatcmpl-override",
        "model": "gpt-4o-2024-11-20",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant",
                                 "content": "Decision: approve. Ref: override"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    trace.substitute(
        ToolOutputSubstitution(
            at_step=p2_llm_step["step_id"],
            fake_response=approve_output,
        )
    )

    result = trace.replay_forward(
        Executor(llm=fake_llm, tool=fake_tool)
    )
    dirty = [sv for sv in result.steps if sv.dirty]
    print(f"  dirty steps after p2 override: {len(dirty)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== batch_worker example ===")
    print("Recording…")
    record_trace()
    print("Replaying with substitution…")
    replay_demo()
    print("Done.")
