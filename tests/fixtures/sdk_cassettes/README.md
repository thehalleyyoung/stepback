# SDK contract cassettes

Each subdirectory holds JSON cassettes captured from the public SDK shapes for a
provider shim in `stepback.shims`. A cassette is **not** a recorded `.sb` trace —
it is a frozen sample of *what the upstream SDK returns* (or what the upstream
tool registry expects), so we can lock down the shim's coercion contract
independently of any live SDK install.

Cassette schema:

```json
{
  "name": "human readable id",
  "description": "what the cassette covers",
  "request": { "model": "...", "messages": [...], "kwargs": {...} },
  "raw_response": { /* bytes-equivalent SDK response shape */ },
  "contract": {
    "openai_shape": {
      "id": "...",
      "model": "...",
      "finish_reason": "...",
      "content": "...",
      "tool_calls": [...] | null,
      "usage": { "prompt_tokens": N, "completion_tokens": N, "total_tokens": N },
      "preserves_native_under": "_anthropic" | "_bedrock" | "_gemini" | null
    }
  }
}
```

The `tests/test_shim_contract.py` runner loads every cassette in this tree,
plays it through the matching `wrap_*` shim against an in-process fake whose
sole responsibility is to return `raw_response` byte-for-byte, and asserts that:

1. The recorded `llm_response` (or `tool_call.outputs`) matches the cassette's
   `contract` block.
2. Replaying the resulting `.sb` trace serves every step from cache (no
   executor calls).
3. The same `raw_response` fed through the duck-typed object path
   (`.model_dump()` / `.to_dict()` / attribute walk) produces an identical
   canonical shape.

Add a new cassette by dropping a JSON file in the appropriate provider folder;
the test runner picks it up automatically.
