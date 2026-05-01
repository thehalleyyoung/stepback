"""Deterministic fake LLM + tools used as the e2e fixture.

This is the "real" pipeline `tests/test_e2e_replay.py` records, then
counterfactually re-executes. The agent is a 6-LLM-step + 6-tool-step
"customer payments bot" that, in its recorded form, makes a bad wire
to the wrong customer because `lookup_customer` returned the UK row
for "Acme Bolts" instead of the US one. Tests then bisect to find
that step, swap a `ToolOutputSubstitution` in to fix it, and assert
the dirty subtree shrinks back to a clean payment.
"""
from __future__ import annotations

import hashlib
from typing import Any, List

# A tiny "customer database". Note both rows match a fuzzy "Acme Bolts" query.
CUSTOMER_DB = {
    "acme-us": {"id": "acme-us", "name": "Acme Bolts Inc",     "country": "US",
                "iban": "US12-3456-7890"},
    "acme-uk": {"id": "acme-uk", "name": "Acme Bolts Ltd UK", "country": "UK",
                "iban": "GB99-9999-9999"},
}

# Recorded run: the lookup tool returns the UK row (the bug).
LOOKUP_BUG_ROW = CUSTOMER_DB["acme-uk"]
# Fixed (counterfactual) row.
LOOKUP_FIXED_ROW = CUSTOMER_DB["acme-us"]


def fake_llm(model: str, messages: List[dict]) -> dict:
    """Deterministic 'LLM': hashes the messages and emits a stable reply.

    The reply text encodes the last user/assistant message + a
    monotone counter pulled from message length so different message
    lists produce different outputs (so substitutions actually
    propagate visibly through the trace).
    """
    blob = "\n".join(f"{m['role']}:{m['content']}" for m in messages)
    digest = hashlib.sha256(blob.encode()).hexdigest()[:12]
    last = messages[-1]["content"] if messages else ""
    if "wire to" in last.lower():
        text = f"OK wired payment {digest}"
    elif "lookup" in last.lower():
        text = f"calling lookup_customer {digest}"
    else:
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


def fake_tool(name: str, args: dict) -> Any:
    """Tool stand-ins. The bug is in the recorded `lookup_customer`."""
    if name == "lookup_customer":
        # Recorded behaviour: return the UK row (wrong). The
        # counterfactual ToolOutputSubstitution will pin this to the
        # US row.
        return LOOKUP_BUG_ROW
    if name == "payment.transfer":
        return {"status": "ok", "wire_to_iban": args["iban"], "amount_usd": args["amount"]}
    if name == "echo":
        return {"echo": args.get("text", "")}
    raise KeyError(f"unknown fake tool: {name!r}")


def run_recorded_agent(rec) -> None:
    """Drive a 12-step deterministic 'customer payments' agent through ``rec``.

    Order: 6 alternations of ``llm_call``, ``tool_call``. The bad wire
    appears at step 12 (the final `payment.transfer`).
    """
    convo: List[dict] = [
        {"role": "system", "content": "You are a payments agent."},
        {"role": "user", "content": "Pay invoice INV-118 to vendor Acme Bolts, $50000."},
    ]

    # 1) LLM plans
    rec.llm_call("gpt-4o-2024-11-20", convo, executor=fake_llm)
    # 2) tool: lookup_customer
    cust = rec.tool_call("lookup_customer", {"name": "Acme Bolts"}, executor=fake_tool)
    convo.append({"role": "assistant", "content": f"Found customer: {cust['outputs']['result']}"})

    # 3) LLM verifies
    rec.llm_call("gpt-4o-2024-11-20", convo, executor=fake_llm)
    # 4) tool echo: amount confirm
    rec.tool_call("echo", {"text": "amount=50000"}, executor=fake_tool)

    # 5) LLM continues
    rec.llm_call("gpt-4o-2024-11-20",
                 convo + [{"role": "assistant", "content": "lookup ok"}],
                 executor=fake_llm)
    # 6) tool echo: country confirm
    rec.tool_call("echo", {"text": f"country={cust['outputs']['result']['country']}"},
                  executor=fake_tool)

    # 7) LLM thinks
    rec.llm_call("gpt-4o-2024-11-20",
                 convo + [{"role": "assistant", "content": "checking"}],
                 executor=fake_llm)
    # 8) tool echo
    rec.tool_call("echo", {"text": "ready"}, executor=fake_tool)

    # 9) LLM thinks again
    rec.llm_call("gpt-4o-2024-11-20",
                 convo + [{"role": "assistant", "content": "ready"}],
                 executor=fake_llm)
    # 10) tool echo: dest IBAN preview
    rec.tool_call("echo", {"text": f"iban={cust['outputs']['result']['iban']}"},
                  executor=fake_tool)

    # 11) LLM emits the wire instruction
    convo.append({"role": "assistant",
                  "content": f"wire to {cust['outputs']['result']['iban']} amount 50000"})
    rec.llm_call("gpt-4o-2024-11-20", convo, executor=fake_llm)

    # 12) tool: payment.transfer (this is the visible bad outcome)
    rec.tool_call(
        "payment.transfer",
        {"iban": cust["outputs"]["result"]["iban"], "amount": 50000},
        executor=fake_tool,
    )
