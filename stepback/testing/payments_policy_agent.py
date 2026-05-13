"""Deterministic policy-gated payments agent for the author-original corpus.

Simulates a compliance-aware payment processor that screens each payment
request through four policy gates before approving, blocking, or holding:

  1. Sanctions screening — checks payee against an OFAC-like blocklist.
  2. Daily-limit check — rejects payments that would breach per-account caps.
  3. Velocity check — flags accounts that exceed the transaction rate limit.
  4. KYC (Know-Your-Customer) — holds payments to unverified beneficiaries.

The agent flow for each task:

  1. LLM parses the payment request.
  2. ``screen_sanctions`` tool checks payee against the blocklist.
  3. ``check_daily_limit`` tool checks remaining daily headroom.
  4. ``check_velocity`` tool checks recent transaction rate.
  5. ``check_kyc`` tool verifies beneficiary identity.
  6. LLM produces a policy decision (approve / block / hold).
  7. ``execute_decision`` tool executes the decision.

Usage::

    from stepback import record
    from stepback.testing.payments_policy_agent import (
        PAYMENTS_TASKS, run_payments_task
    )

    for task in PAYMENTS_TASKS:
        with record(f"/tmp/{task['task_id']}.sb") as rec:
            run_payments_task(rec, task['task_id'])
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Task catalogue
# ---------------------------------------------------------------------------

PAYMENTS_TASKS: List[Dict[str, Any]] = [
    {
        "task_id": "normal-payment",
        "prompt": (
            "Process payment: send $100 to beneficiary IBAN GB00-1234-5678, "
            "reference: Invoice INV-001, originator account ACC-111."
        ),
        "inputs": {
            "amount_usd": 100,
            "beneficiary_iban": "GB00-1234-5678",
            "beneficiary_name": "ACME Corp",
            "originator_account": "ACC-111",
            "reference": "INV-001",
        },
        "evaluation": {
            "expected_decision": "approve",
            "gate_results": {
                "sanctions": "clear",
                "daily_limit": "within_limit",
                "velocity": "normal",
                "kyc": "verified",
            },
        },
    },
    {
        "task_id": "overlimit-payment",
        "prompt": (
            "Process payment: send $50,000 to beneficiary IBAN US98-9999-0000, "
            "reference: Invoice INV-002, originator account ACC-222."
        ),
        "inputs": {
            "amount_usd": 50_000,
            "beneficiary_iban": "US98-9999-0000",
            "beneficiary_name": "Big Vendor LLC",
            "originator_account": "ACC-222",
            "reference": "INV-002",
        },
        "evaluation": {
            "expected_decision": "block",
            "block_reason": "daily_limit_exceeded",
            "gate_results": {
                "sanctions": "clear",
                "daily_limit": "exceeded",
                "velocity": "normal",
                "kyc": "verified",
            },
        },
    },
    {
        "task_id": "sanctioned-country",
        "prompt": (
            "Process payment: send $500 to beneficiary IBAN SY01-0001-0002, "
            "reference: Invoice INV-003, originator account ACC-333."
        ),
        "inputs": {
            "amount_usd": 500,
            "beneficiary_iban": "SY01-0001-0002",
            "beneficiary_name": "Damascus Trading Co",
            "originator_account": "ACC-333",
            "reference": "INV-003",
        },
        "evaluation": {
            "expected_decision": "block",
            "block_reason": "sanctions_hit",
            "gate_results": {
                "sanctions": "hit",
                "daily_limit": "not_checked",
                "velocity": "not_checked",
                "kyc": "not_checked",
            },
        },
    },
    {
        "task_id": "fraud-velocity",
        "prompt": (
            "Process payment: send $200 to beneficiary IBAN FR76-3000-6000, "
            "reference: Invoice INV-004, originator account ACC-444."
        ),
        "inputs": {
            "amount_usd": 200,
            "beneficiary_iban": "FR76-3000-6000",
            "beneficiary_name": "Legit Supplies SARL",
            "originator_account": "ACC-444",
            "reference": "INV-004",
        },
        "evaluation": {
            "expected_decision": "hold",
            "hold_reason": "velocity_fraud_flag",
            "gate_results": {
                "sanctions": "clear",
                "daily_limit": "within_limit",
                "velocity": "flagged",
                "kyc": "not_checked",
            },
        },
    },
    {
        "task_id": "unverified-beneficiary",
        "prompt": (
            "Process payment: send $750 to beneficiary IBAN DE89-3704-0044, "
            "reference: Invoice INV-005, originator account ACC-555."
        ),
        "inputs": {
            "amount_usd": 750,
            "beneficiary_iban": "DE89-3704-0044",
            "beneficiary_name": "New Startup GmbH",
            "originator_account": "ACC-555",
            "reference": "INV-005",
        },
        "evaluation": {
            "expected_decision": "hold",
            "hold_reason": "kyc_required",
            "gate_results": {
                "sanctions": "clear",
                "daily_limit": "within_limit",
                "velocity": "normal",
                "kyc": "unverified",
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Scripted policy engine responses
# ---------------------------------------------------------------------------

_SANCTIONS: Dict[str, Dict[str, Any]] = {
    "GB00-1234-5678": {"result": "clear", "match_score": 0.0, "list": "OFAC"},
    "US98-9999-0000": {"result": "clear", "match_score": 0.0, "list": "OFAC"},
    "SY01-0001-0002": {"result": "hit", "match_score": 0.99,
                       "matched_entity": "Damascus Trading Co",
                       "list": "OFAC", "reason": "Syria SDN"},
    "FR76-3000-6000": {"result": "clear", "match_score": 0.0, "list": "OFAC"},
    "DE89-3704-0044": {"result": "clear", "match_score": 0.0, "list": "OFAC"},
}

_LIMITS: Dict[str, Dict[str, Any]] = {
    "ACC-111": {"daily_limit_usd": 10_000, "used_today_usd": 500, "remaining_usd": 9_500},
    "ACC-222": {"daily_limit_usd": 25_000, "used_today_usd": 20_000, "remaining_usd": 5_000},
    "ACC-444": {"daily_limit_usd": 10_000, "used_today_usd": 1_000, "remaining_usd": 9_000},
    "ACC-555": {"daily_limit_usd": 10_000, "used_today_usd": 200, "remaining_usd": 9_800},
}

_VELOCITY: Dict[str, Dict[str, Any]] = {
    "ACC-111": {"tx_last_hour": 2, "tx_limit_per_hour": 20, "flagged": False},
    "ACC-222": {"tx_last_hour": 5, "tx_limit_per_hour": 20, "flagged": False},
    "ACC-444": {"tx_last_hour": 18, "tx_limit_per_hour": 10, "flagged": True,
                "flag_reason": "exceeded_hourly_limit"},
    "ACC-555": {"tx_last_hour": 3, "tx_limit_per_hour": 20, "flagged": False},
}

_KYC: Dict[str, Dict[str, Any]] = {
    "GB00-1234-5678": {"verified": True, "verification_date": "2024-01-15", "tier": "full"},
    "US98-9999-0000": {"verified": True, "verification_date": "2023-11-20", "tier": "full"},
    "FR76-3000-6000": {"verified": True, "verification_date": "2024-03-01", "tier": "full"},
    "DE89-3704-0044": {"verified": False, "verification_date": None, "tier": "none",
                       "required_docs": ["company_registration", "beneficial_owner"]},
}


def _digest(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Fake LLM + tools
# ---------------------------------------------------------------------------

def fake_llm(model: str, messages: List[dict]) -> dict:
    """Deterministic LLM stand-in for the payments-policy corpus."""
    blob = "\n".join(f"{m['role']}:{m.get('content', '')}" for m in messages)
    digest = _digest(blob)
    last = messages[-1].get("content", "") if messages else ""

    if "hit" in last.lower() and "sanction" in last.lower():
        text = f"BLOCK: Payment halted — sanctions hit detected. Ref: {digest}"
    elif "exceeded" in last.lower() or "limit" in last.lower():
        text = f"BLOCK: Daily limit exceeded. Ref: {digest}"
    elif "velocity" in last.lower() or "fraud" in last.lower():
        text = f"HOLD: Velocity fraud flag. Manual review required. Ref: {digest}"
    elif "unverified" in last.lower() or "kyc" in last.lower():
        text = f"HOLD: Pending KYC verification. Ref: {digest}"
    elif "approve" in last.lower() or "clear" in last.lower():
        text = f"APPROVED: All policy gates passed. Ref: {digest}"
    else:
        text = f"Processing policy checks... Ref: {digest}"

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
    """Policy tool stand-ins for the payments-policy corpus."""
    if name == "screen_sanctions":
        iban = args.get("beneficiary_iban", "")
        return _SANCTIONS.get(iban, {"result": "clear", "match_score": 0.0, "list": "OFAC"})

    if name == "check_daily_limit":
        account = args.get("originator_account", "")
        amount = args.get("amount_usd", 0)
        limit_data = _LIMITS.get(account, {
            "daily_limit_usd": 10_000,
            "used_today_usd": 0,
            "remaining_usd": 10_000,
        })
        remaining = limit_data["remaining_usd"]
        return {
            **limit_data,
            "requested_usd": amount,
            "decision": "exceeded" if amount > remaining else "within_limit",
        }

    if name == "check_velocity":
        account = args.get("originator_account", "")
        return _VELOCITY.get(account, {
            "tx_last_hour": 1,
            "tx_limit_per_hour": 20,
            "flagged": False,
        })

    if name == "check_kyc":
        iban = args.get("beneficiary_iban", "")
        return _KYC.get(iban, {"verified": False, "tier": "none",
                               "required_docs": ["identity"]})

    if name == "execute_decision":
        decision = args.get("decision", "hold")
        payment_ref = args.get("reference", "")
        if decision == "approve":
            return {
                "status": "approved",
                "payment_id": f"PAY-{_digest(payment_ref)}",
                "reference": payment_ref,
            }
        elif decision == "block":
            return {
                "status": "blocked",
                "reason": args.get("reason", "policy"),
                "reference": payment_ref,
                "case_id": f"BLOCK-{_digest(payment_ref)}",
            }
        else:  # hold
            return {
                "status": "on_hold",
                "reason": args.get("reason", "review_required"),
                "reference": payment_ref,
                "review_id": f"HOLD-{_digest(payment_ref)}",
                "review_queue": "compliance",
            }

    raise KeyError(f"unknown fake payments tool: {name!r}")


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

def run_payments_task(rec: Any, task_id: str) -> None:
    """Drive the payments-policy agent through ``rec`` for the given ``task_id``.

    Parameters
    ----------
    rec:
        An open :class:`~stepback.Recorder` (from ``with record(...) as rec``).
    task_id:
        One of the ids in :data:`PAYMENTS_TASKS`.
    """
    task = next((t for t in PAYMENTS_TASKS if t["task_id"] == task_id), None)
    if task is None:
        raise ValueError(f"Unknown payments task id: {task_id!r}")

    model = "gpt-4o-2024-11-20"
    inp = task["inputs"]
    convo = [
        {"role": "system",
         "content": "You are a compliance officer. Screen payments through all policy gates."},
        {"role": "user", "content": task["prompt"]},
    ]

    # 1) LLM parses request
    rec.llm_call(model, convo, executor=fake_llm)

    # 2) Sanctions screening
    sanctions = rec.tool_call("screen_sanctions",
                               {"beneficiary_iban": inp["beneficiary_iban"],
                                "beneficiary_name": inp["beneficiary_name"]},
                               executor=fake_tool)
    sanctions_result = sanctions["outputs"]["result"]
    convo.append({"role": "assistant",
                  "content": f"Sanctions: {sanctions_result['result']}"})

    # Fast-path block on sanctions hit
    if sanctions_result.get("result") == "hit":
        rec.llm_call(model,
                     convo + [{"role": "user",
                                "content": "sanctions hit detected, block payment"}],
                     executor=fake_llm)
        rec.tool_call("execute_decision",
                       {"decision": "block",
                        "reason": "sanctions_hit",
                        "reference": inp["reference"]},
                       executor=fake_tool)
        return

    # 3) Daily limit check
    limit = rec.tool_call("check_daily_limit",
                           {"originator_account": inp["originator_account"],
                            "amount_usd": inp["amount_usd"]},
                           executor=fake_tool)
    limit_result = limit["outputs"]["result"]
    convo.append({"role": "assistant",
                  "content": f"Limit: {limit_result['decision']}"})

    # Fast-path block on limit exceeded
    if limit_result.get("decision") == "exceeded":
        rec.llm_call(model,
                     convo + [{"role": "user",
                                "content": "limit exceeded, block payment"}],
                     executor=fake_llm)
        rec.tool_call("execute_decision",
                       {"decision": "block",
                        "reason": "daily_limit_exceeded",
                        "reference": inp["reference"]},
                       executor=fake_tool)
        return

    # 4) Velocity check
    velocity = rec.tool_call("check_velocity",
                              {"originator_account": inp["originator_account"]},
                              executor=fake_tool)
    velocity_result = velocity["outputs"]["result"]
    convo.append({"role": "assistant",
                  "content": f"Velocity: {'flagged' if velocity_result.get('flagged') else 'normal'}"})

    # Fast-path hold on velocity flag
    if velocity_result.get("flagged"):
        rec.llm_call(model,
                     convo + [{"role": "user",
                                "content": "velocity fraud flag, hold for review"}],
                     executor=fake_llm)
        rec.tool_call("execute_decision",
                       {"decision": "hold",
                        "reason": "velocity_fraud_flag",
                        "reference": inp["reference"]},
                       executor=fake_tool)
        return

    # 5) KYC check
    kyc = rec.tool_call("check_kyc",
                         {"beneficiary_iban": inp["beneficiary_iban"],
                          "beneficiary_name": inp["beneficiary_name"]},
                         executor=fake_tool)
    kyc_result = kyc["outputs"]["result"]
    convo.append({"role": "assistant",
                  "content": f"KYC: {'verified' if kyc_result.get('verified') else 'unverified'}"})

    # Hold on unverified KYC
    if not kyc_result.get("verified"):
        rec.llm_call(model,
                     convo + [{"role": "user",
                                "content": "unverified beneficiary, hold for KYC"}],
                     executor=fake_llm)
        rec.tool_call("execute_decision",
                       {"decision": "hold",
                        "reason": "kyc_required",
                        "reference": inp["reference"]},
                       executor=fake_tool)
        return

    # 6) All clear — approve
    rec.llm_call(model,
                 convo + [{"role": "user",
                            "content": "all gates clear, approve payment"}],
                 executor=fake_llm)
    rec.tool_call("execute_decision",
                   {"decision": "approve",
                    "reference": inp["reference"]},
                   executor=fake_tool)
