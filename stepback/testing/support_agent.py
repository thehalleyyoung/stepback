"""Deterministic support-agent fixture for the author-original corpus.

Simulates a customer-support bot that handles five canonical task types:
order status inquiry, refund request, account unlock, wrong-item exchange,
and subscription cancellation.  The agent is network-free and fully
deterministic: the fake LLM hashes its messages to produce a stable reply,
and the fake tools return scripted outcomes based on task id.

Each task is identified by a ``task_id`` string.  Pass it to
:func:`run_support_task` together with an open :class:`~stepback.Recorder`::

    from stepback import record
    from stepback.testing.support_agent import SUPPORT_TASKS, run_support_task

    for task in SUPPORT_TASKS:
        with record(f"/tmp/{task['task_id']}.sb", key=task_key(task['task_id'])) as rec:
            run_support_task(rec, task['task_id'])
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Task catalogue
# ---------------------------------------------------------------------------

SUPPORT_TASKS: List[Dict[str, Any]] = [
    {
        "task_id": "order-status-delayed",
        "prompt": (
            "Customer email: 'My order #1234 was supposed to arrive 3 days ago but "
            "I haven't received it yet. Can you help?'"
        ),
        "inputs": {"order_id": "1234", "customer_id": "cust-001"},
        "evaluation": {
            "expected_action": "escalate_to_shipping",
            "expected_outcome": "escalated",
        },
    },
    {
        "task_id": "refund-eligible",
        "prompt": (
            "Customer email: 'I would like to return order #5678 and get a refund. "
            "The product was defective.'"
        ),
        "inputs": {"order_id": "5678", "customer_id": "cust-002"},
        "evaluation": {
            "expected_action": "approve_refund",
            "expected_outcome": "refund_approved",
        },
    },
    {
        "task_id": "account-unlock",
        "prompt": (
            "Customer email: 'I cannot log in to my account. It says my account is "
            "locked. Please help me unlock it.'"
        ),
        "inputs": {"customer_id": "cust-003", "email": "user@example.com"},
        "evaluation": {
            "expected_action": "unlock_account",
            "expected_outcome": "account_unlocked",
        },
    },
    {
        "task_id": "wrong-item",
        "prompt": (
            "Customer email: 'I received the wrong item in my order #9012. I ordered "
            "a blue widget but received a red one. Please send the correct item.'"
        ),
        "inputs": {"order_id": "9012", "customer_id": "cust-004"},
        "evaluation": {
            "expected_action": "initiate_replacement",
            "expected_outcome": "replacement_initiated",
        },
    },
    {
        "task_id": "subscription-cancel",
        "prompt": (
            "Customer email: 'I would like to cancel my monthly subscription. "
            "Please process the cancellation effective immediately.'"
        ),
        "inputs": {"customer_id": "cust-005", "subscription_id": "sub-999"},
        "evaluation": {
            "expected_action": "cancel_subscription",
            "expected_outcome": "subscription_cancelled",
        },
    },
]

# ---------------------------------------------------------------------------
# Scripted database
# ---------------------------------------------------------------------------

_ORDERS: Dict[str, Dict[str, Any]] = {
    "1234": {
        "order_id": "1234",
        "status": "shipped",
        "shipping_days_overdue": 3,
        "item": "Blue Widget Pro",
        "amount_usd": 49.99,
        "refund_eligible": False,
    },
    "5678": {
        "order_id": "5678",
        "status": "delivered",
        "shipping_days_overdue": 0,
        "item": "Red Gadget X",
        "amount_usd": 129.99,
        "refund_eligible": True,
        "defective": True,
    },
    "9012": {
        "order_id": "9012",
        "status": "delivered",
        "shipping_days_overdue": 0,
        "item_ordered": "Blue Widget",
        "item_received": "Red Widget",
        "wrong_item": True,
        "amount_usd": 39.99,
        "refund_eligible": True,
    },
}

_ACCOUNTS: Dict[str, Dict[str, Any]] = {
    "cust-003": {
        "customer_id": "cust-003",
        "locked": True,
        "lock_reason": "too_many_failed_logins",
        "email": "user@example.com",
    },
    "cust-005": {
        "customer_id": "cust-005",
        "locked": False,
        "subscription_id": "sub-999",
    },
}

_SUBSCRIPTIONS: Dict[str, Dict[str, Any]] = {
    "sub-999": {
        "subscription_id": "sub-999",
        "plan": "monthly",
        "status": "active",
        "customer_id": "cust-005",
        "amount_usd": 19.99,
    },
}


def _digest(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Fake LLM + tools
# ---------------------------------------------------------------------------

def fake_llm(model: str, messages: List[dict]) -> dict:
    """Deterministic LLM stand-in: hashes messages to produce a stable reply."""
    blob = "\n".join(f"{m['role']}:{m.get('content','')}" for m in messages)
    digest = _digest(blob)
    last = messages[-1].get("content", "") if messages else ""

    if "escalate" in last.lower():
        text = f"Escalating to shipping department. Reference: {digest}"
    elif "refund" in last.lower():
        text = f"Processing refund. Authorization: {digest}"
    elif "unlock" in last.lower():
        text = f"Account unlocked. Confirmation: {digest}"
    elif "replacement" in last.lower() or "wrong" in last.lower():
        text = f"Initiating replacement shipment. Ticket: {digest}"
    elif "cancel" in last.lower() or "subscription" in last.lower():
        text = f"Subscription cancellation processed. Ref: {digest}"
    else:
        text = f"I will help you with that. {digest}"

    return {
        "id": f"chatcmpl-{digest}",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
        "usage": {
            "prompt_tokens": sum(len(m.get("content", "")) for m in messages),
            "completion_tokens": len(text),
            "total_tokens": sum(len(m.get("content", "")) for m in messages) + len(text),
        },
    }


def fake_tool(name: str, args: dict) -> Any:
    """Tool stand-ins for the support agent corpus."""
    if name == "lookup_order":
        order_id = str(args.get("order_id", ""))
        return _ORDERS.get(order_id, {"error": f"order {order_id} not found"})

    if name == "check_shipping":
        order = args.get("order", {})
        days_overdue = order.get("shipping_days_overdue", 0)
        return {
            "status": "delayed" if days_overdue > 0 else "on_time",
            "days_overdue": days_overdue,
            "carrier": "FedEx",
            "tracking": f"TRK-{_digest(str(order))}",
        }

    if name == "escalate_to_shipping":
        return {
            "ticket_id": f"ESC-{_digest(str(args))}",
            "status": "escalated",
            "eta_hours": 24,
        }

    if name == "validate_refund":
        order = args.get("order", {})
        return {
            "eligible": order.get("refund_eligible", False),
            "reason": "defective_product" if order.get("defective") else "policy",
            "amount_usd": order.get("amount_usd", 0),
        }

    if name == "approve_refund":
        return {
            "refund_id": f"REF-{_digest(str(args))}",
            "status": "approved",
            "amount_usd": args.get("amount_usd", 0),
        }

    if name == "lookup_account":
        customer_id = args.get("customer_id", "")
        return _ACCOUNTS.get(customer_id, {"error": f"account {customer_id} not found"})

    if name == "unlock_account":
        customer_id = args.get("customer_id", "")
        return {
            "customer_id": customer_id,
            "status": "unlocked",
            "confirmation": f"UNLOCK-{_digest(customer_id)}",
        }

    if name == "initiate_replacement":
        return {
            "replacement_id": f"REPL-{_digest(str(args))}",
            "status": "initiated",
            "shipping_days": 3,
        }

    if name == "lookup_subscription":
        sub_id = args.get("subscription_id", "")
        return _SUBSCRIPTIONS.get(sub_id, {"error": f"subscription {sub_id} not found"})

    if name == "cancel_subscription":
        sub_id = args.get("subscription_id", "")
        return {
            "subscription_id": sub_id,
            "status": "cancelled",
            "confirmation": f"CANCEL-{_digest(sub_id)}",
            "effective": "immediate",
        }

    raise KeyError(f"unknown fake support tool: {name!r}")


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

def run_support_task(rec: Any, task_id: str) -> None:
    """Drive the support agent through ``rec`` for the given ``task_id``.

    The agent is deterministic: the fake LLM and tools always return the
    same outputs for the same inputs so replay will produce bit-identical
    step hashes.

    Parameters
    ----------
    rec:
        An open :class:`~stepback.Recorder` (from ``with record(...) as rec``).
    task_id:
        One of the ids in :data:`SUPPORT_TASKS`.
    """
    task = next((t for t in SUPPORT_TASKS if t["task_id"] == task_id), None)
    if task is None:
        raise ValueError(f"Unknown support task id: {task_id!r}")

    model = "gpt-4o-2024-11-20"
    system_msg = {"role": "system", "content": "You are a customer support agent."}
    user_msg = {"role": "user", "content": task["prompt"]}
    convo = [system_msg, user_msg]

    if task_id == "order-status-delayed":
        # 1) LLM decides to look up the order
        rec.llm_call(model, convo, executor=fake_llm)
        # 2) look up the order
        order = rec.tool_call("lookup_order",
                              {"order_id": task["inputs"]["order_id"]},
                              executor=fake_tool)
        # 3) check shipping status
        shipping = rec.tool_call("check_shipping",
                                 {"order": order["outputs"]["result"]},
                                 executor=fake_tool)
        convo.append({"role": "assistant",
                      "content": f"Shipping: {shipping['outputs']['result']}"})
        # 4) LLM decides to escalate
        rec.llm_call(model, convo + [{"role": "user", "content": "escalate if delayed"}],
                     executor=fake_llm)
        # 5) escalate
        rec.tool_call("escalate_to_shipping",
                      {"order_id": task["inputs"]["order_id"],
                       "reason": "delayed_delivery"},
                      executor=fake_tool)
        # 6) LLM drafts response
        rec.llm_call(model, convo + [{"role": "assistant", "content": "escalation done"}],
                     executor=fake_llm)

    elif task_id == "refund-eligible":
        # 1) LLM
        rec.llm_call(model, convo, executor=fake_llm)
        # 2) lookup order
        order = rec.tool_call("lookup_order",
                              {"order_id": task["inputs"]["order_id"]},
                              executor=fake_tool)
        # 3) validate refund eligibility
        refund_check = rec.tool_call("validate_refund",
                                     {"order": order["outputs"]["result"]},
                                     executor=fake_tool)
        convo.append({"role": "assistant",
                      "content": f"Refund check: {refund_check['outputs']['result']}"})
        # 4) LLM decides to approve
        rec.llm_call(model, convo + [{"role": "user", "content": "approve the refund"}],
                     executor=fake_llm)
        # 5) approve refund
        amount = refund_check["outputs"]["result"].get("amount_usd", 0)
        rec.tool_call("approve_refund",
                      {"order_id": task["inputs"]["order_id"],
                       "amount_usd": amount},
                      executor=fake_tool)
        # 6) LLM confirms
        rec.llm_call(model,
                     convo + [{"role": "assistant",
                                "content": f"refund of ${amount} approved"}],
                     executor=fake_llm)

    elif task_id == "account-unlock":
        # 1) LLM
        rec.llm_call(model, convo, executor=fake_llm)
        # 2) lookup account
        account = rec.tool_call("lookup_account",
                                {"customer_id": task["inputs"]["customer_id"]},
                                executor=fake_tool)
        convo.append({"role": "assistant",
                      "content": f"Account: {account['outputs']['result']}"})
        # 3) LLM decides to unlock
        rec.llm_call(model, convo + [{"role": "user", "content": "unlock the account"}],
                     executor=fake_llm)
        # 4) unlock
        rec.tool_call("unlock_account",
                      {"customer_id": task["inputs"]["customer_id"]},
                      executor=fake_tool)
        # 5) LLM confirms
        rec.llm_call(model,
                     convo + [{"role": "assistant", "content": "account is now unlocked"}],
                     executor=fake_llm)

    elif task_id == "wrong-item":
        # 1) LLM
        rec.llm_call(model, convo, executor=fake_llm)
        # 2) lookup order
        order = rec.tool_call("lookup_order",
                              {"order_id": task["inputs"]["order_id"]},
                              executor=fake_tool)
        convo.append({"role": "assistant",
                      "content": f"Order: {order['outputs']['result']}"})
        # 3) LLM decides to initiate replacement
        rec.llm_call(model,
                     convo + [{"role": "user",
                                "content": "initiate replacement for wrong item"}],
                     executor=fake_llm)
        # 4) initiate replacement
        rec.tool_call("initiate_replacement",
                      {"order_id": task["inputs"]["order_id"],
                       "item_to_send": order["outputs"]["result"].get("item_ordered"),
                       "reason": "wrong_item_delivered"},
                      executor=fake_tool)
        # 5) LLM confirms
        rec.llm_call(model,
                     convo + [{"role": "assistant",
                                "content": "replacement initiated"}],
                     executor=fake_llm)

    elif task_id == "subscription-cancel":
        # 1) LLM
        rec.llm_call(model, convo, executor=fake_llm)
        # 2) lookup subscription
        sub = rec.tool_call("lookup_subscription",
                            {"subscription_id": task["inputs"]["subscription_id"]},
                            executor=fake_tool)
        convo.append({"role": "assistant",
                      "content": f"Subscription: {sub['outputs']['result']}"})
        # 3) LLM drafts cancellation message
        rec.llm_call(model,
                     convo + [{"role": "user",
                                "content": "cancel the subscription immediately"}],
                     executor=fake_llm)
        # 4) cancel
        rec.tool_call("cancel_subscription",
                      {"subscription_id": task["inputs"]["subscription_id"],
                       "reason": "customer_request"},
                      executor=fake_tool)
        # 5) LLM confirms
        rec.llm_call(model,
                     convo + [{"role": "assistant",
                                "content": "subscription cancelled"}],
                     executor=fake_llm)
