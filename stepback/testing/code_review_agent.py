"""Deterministic code-review agent fixture for the author-original corpus.

Simulates a code-review bot that analyses Python code snippets and identifies
five canonical defect types: SQL injection, off-by-one error, race condition,
missing error handling, and resource leak.  The agent flow is:

  1. LLM reads the code and forms an initial assessment.
  2. ``analyze_code`` tool extracts AST-level metrics (line count, complexity).
  3. ``identify_issues`` tool scans for known defect patterns and returns hits.
  4. LLM synthesises the findings into a review comment.
  5. ``suggest_fix`` tool generates a patch snippet.
  6. LLM writes the final review.

The fake LLM hashes its inputs to produce a deterministic reply; the fake
tools return scripted findings based on the task id.

Usage::

    from stepback import record
    from stepback.testing.code_review_agent import CODE_REVIEW_TASKS, run_code_review_task

    for task in CODE_REVIEW_TASKS:
        with record(f"/tmp/{task['task_id']}.sb") as rec:
            run_code_review_task(rec, task['task_id'])
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Task catalogue
# ---------------------------------------------------------------------------

CODE_REVIEW_TASKS: List[Dict[str, Any]] = [
    {
        "task_id": "sql-injection",
        "prompt": (
            "Review the following Python function for security issues:\n\n"
            "def get_user(username):\n"
            "    query = f\"SELECT * FROM users WHERE name = '{username}'\"\n"
            "    return db.execute(query).fetchall()\n"
        ),
        "inputs": {
            "language": "python",
            "file": "db_helpers.py",
            "defect_category": "security",
        },
        "evaluation": {
            "expected_issues": ["sql_injection"],
            "severity": "critical",
            "fix": "use parameterized queries",
        },
    },
    {
        "task_id": "off-by-one",
        "prompt": (
            "Review the following Python function for correctness:\n\n"
            "def slice_last_n(items, n):\n"
            "    return items[len(items) - n:len(items) + 1]\n"
        ),
        "inputs": {
            "language": "python",
            "file": "utils.py",
            "defect_category": "logic",
        },
        "evaluation": {
            "expected_issues": ["off_by_one"],
            "severity": "medium",
            "fix": "use items[-n:] instead",
        },
    },
    {
        "task_id": "race-condition",
        "prompt": (
            "Review the following Python function for thread safety:\n\n"
            "import threading\n"
            "_counter = 0\n\n"
            "def increment():\n"
            "    global _counter\n"
            "    tmp = _counter\n"
            "    _counter = tmp + 1\n"
        ),
        "inputs": {
            "language": "python",
            "file": "worker.py",
            "defect_category": "concurrency",
        },
        "evaluation": {
            "expected_issues": ["race_condition"],
            "severity": "high",
            "fix": "use threading.Lock or threading.local",
        },
    },
    {
        "task_id": "missing-error-handling",
        "prompt": (
            "Review the following Python function for robustness:\n\n"
            "import urllib.request\n\n"
            "def fetch_config(url):\n"
            "    with urllib.request.urlopen(url) as r:\n"
            "        return r.read().decode()\n"
        ),
        "inputs": {
            "language": "python",
            "file": "config_loader.py",
            "defect_category": "error_handling",
        },
        "evaluation": {
            "expected_issues": ["missing_error_handling"],
            "severity": "medium",
            "fix": "wrap in try/except for URLError and timeout",
        },
    },
    {
        "task_id": "resource-leak",
        "prompt": (
            "Review the following Python function for resource management:\n\n"
            "def read_lines(path):\n"
            "    f = open(path)\n"
            "    lines = f.readlines()\n"
            "    return lines\n"
        ),
        "inputs": {
            "language": "python",
            "file": "file_utils.py",
            "defect_category": "resource_management",
        },
        "evaluation": {
            "expected_issues": ["resource_leak"],
            "severity": "medium",
            "fix": "use 'with open(path) as f:' context manager",
        },
    },
]

# ---------------------------------------------------------------------------
# Scripted analysis results per task
# ---------------------------------------------------------------------------

_ANALYSIS: Dict[str, Dict[str, Any]] = {
    "sql-injection": {
        "line_count": 3,
        "cyclomatic_complexity": 1,
        "has_string_formatting": True,
        "has_db_call": True,
    },
    "off-by-one": {
        "line_count": 2,
        "cyclomatic_complexity": 1,
        "has_slice": True,
        "has_arithmetic": True,
    },
    "race-condition": {
        "line_count": 7,
        "cyclomatic_complexity": 1,
        "has_global": True,
        "has_shared_state": True,
    },
    "missing-error-handling": {
        "line_count": 4,
        "cyclomatic_complexity": 1,
        "has_network_call": True,
        "has_try_except": False,
    },
    "resource-leak": {
        "line_count": 4,
        "cyclomatic_complexity": 1,
        "has_file_open": True,
        "has_context_manager": False,
    },
}

_ISSUES: Dict[str, List[Dict[str, Any]]] = {
    "sql-injection": [
        {
            "issue_id": "SEC-001",
            "type": "sql_injection",
            "line": 2,
            "severity": "critical",
            "description": "f-string used to construct SQL query with unsanitized input",
        }
    ],
    "off-by-one": [
        {
            "issue_id": "BUG-001",
            "type": "off_by_one",
            "line": 2,
            "severity": "medium",
            "description": "Upper bound `len(items) + 1` exceeds list length by 1",
        }
    ],
    "race-condition": [
        {
            "issue_id": "CON-001",
            "type": "race_condition",
            "line": 5,
            "severity": "high",
            "description": "read-modify-write on shared global `_counter` without lock",
        }
    ],
    "missing-error-handling": [
        {
            "issue_id": "ROB-001",
            "type": "missing_error_handling",
            "line": 4,
            "severity": "medium",
            "description": "urlopen may raise URLError or socket.timeout; neither caught",
        }
    ],
    "resource-leak": [
        {
            "issue_id": "RES-001",
            "type": "resource_leak",
            "line": 2,
            "severity": "medium",
            "description": "file handle opened without context manager; not closed on exception",
        }
    ],
}

_FIXES: Dict[str, str] = {
    "sql-injection": (
        "def get_user(username):\n"
        "    query = 'SELECT * FROM users WHERE name = ?'\n"
        "    return db.execute(query, (username,)).fetchall()\n"
    ),
    "off-by-one": "def slice_last_n(items, n):\n    return items[-n:]\n",
    "race-condition": (
        "import threading\n"
        "_counter = 0\n"
        "_lock = threading.Lock()\n\n"
        "def increment():\n"
        "    global _counter\n"
        "    with _lock:\n"
        "        _counter += 1\n"
    ),
    "missing-error-handling": (
        "import urllib.request\n"
        "import urllib.error\n\n"
        "def fetch_config(url, timeout=10):\n"
        "    try:\n"
        "        with urllib.request.urlopen(url, timeout=timeout) as r:\n"
        "            return r.read().decode()\n"
        "    except (urllib.error.URLError, TimeoutError) as e:\n"
        "        raise RuntimeError(f'Failed to fetch config: {e}') from e\n"
    ),
    "resource-leak": (
        "def read_lines(path):\n"
        "    with open(path) as f:\n"
        "        return f.readlines()\n"
    ),
}


def _digest(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Fake LLM + tools
# ---------------------------------------------------------------------------

def fake_llm(model: str, messages: List[dict]) -> dict:
    """Deterministic LLM stand-in for the code-review corpus."""
    blob = "\n".join(f"{m['role']}:{m.get('content', '')}" for m in messages)
    digest = _digest(blob)
    last = messages[-1].get("content", "") if messages else ""

    if "critical" in last.lower() or "injection" in last.lower():
        text = f"CRITICAL: SQL injection vulnerability detected. Fix with parameterized queries. [{digest}]"
    elif "fix" in last.lower() or "patch" in last.lower():
        text = f"Suggested fix applied. Review the patch carefully. [{digest}]"
    elif "race" in last.lower() or "concurrency" in last.lower():
        text = f"HIGH: Race condition on shared state. Add a threading.Lock. [{digest}]"
    elif "leak" in last.lower() or "resource" in last.lower():
        text = f"MEDIUM: Resource leak — use a context manager. [{digest}]"
    elif "error" in last.lower() or "exception" in last.lower():
        text = f"MEDIUM: Missing error handling for network/IO operations. [{digest}]"
    else:
        text = f"Code review in progress. Analyzing... [{digest}]"

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
    """Tool stand-ins for the code-review agent corpus."""
    task_id = args.get("task_id", "")

    if name == "analyze_code":
        return _ANALYSIS.get(task_id, {
            "line_count": 1,
            "cyclomatic_complexity": 1,
        })

    if name == "identify_issues":
        return {
            "issues": _ISSUES.get(task_id, []),
            "issue_count": len(_ISSUES.get(task_id, [])),
        }

    if name == "suggest_fix":
        issue_type = args.get("issue_type", "")
        fix = _FIXES.get(task_id, "# No fix available")
        return {
            "issue_type": issue_type,
            "fix_patch": fix,
            "confidence": 0.95,
        }

    raise KeyError(f"unknown fake code-review tool: {name!r}")


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

def run_code_review_task(rec: Any, task_id: str) -> None:
    """Drive the code-review agent through ``rec`` for the given ``task_id``.

    Parameters
    ----------
    rec:
        An open :class:`~stepback.Recorder` (from ``with record(...) as rec``).
    task_id:
        One of the ids in :data:`CODE_REVIEW_TASKS`.
    """
    task = next((t for t in CODE_REVIEW_TASKS if t["task_id"] == task_id), None)
    if task is None:
        raise ValueError(f"Unknown code review task id: {task_id!r}")

    model = "gpt-4o-2024-11-20"
    convo = [
        {"role": "system", "content": "You are an expert code reviewer."},
        {"role": "user", "content": task["prompt"]},
    ]

    # 1) LLM initial assessment
    rec.llm_call(model, convo, executor=fake_llm)

    # 2) Analyze code structure
    analysis = rec.tool_call("analyze_code",
                             {"task_id": task_id, "code": task["prompt"]},
                             executor=fake_tool)

    # 3) Identify specific issues
    issues_result = rec.tool_call("identify_issues",
                                  {"task_id": task_id,
                                   "analysis": analysis["outputs"]["result"]},
                                  executor=fake_tool)
    issues = issues_result["outputs"]["result"]["issues"]

    convo.append({"role": "assistant",
                  "content": f"Found {len(issues)} issue(s): {issues}"})

    # 4) LLM synthesises findings
    rec.llm_call(model, convo, executor=fake_llm)

    # 5) Generate fix for the first issue (if any)
    issue_type = issues[0]["type"] if issues else "none"
    rec.tool_call("suggest_fix",
                  {"task_id": task_id, "issue_type": issue_type},
                  executor=fake_tool)

    # 6) LLM writes final review
    rec.llm_call(model,
                 convo + [{"role": "assistant",
                            "content": f"fix for {issue_type} generated"}],
                 executor=fake_llm)
