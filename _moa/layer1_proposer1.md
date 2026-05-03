# Proposer 1 — Breadth-first: more typed substitution dataclasses

## Theme
The substitution vocabulary in `stepback/substitutions.py` only has 5
kinds (Prompt / Model / ToolOutput / Policy / Router). The README
markets `stepback` as a "time-travel debugger for AI agents" with
the ability to substitute *prompts / tool outputs / policies / models*
— but in practice debugging an LLM agent regression demands finer
levers. A 200-step trace has dozens of failure-shape questions:
*"what if the system prompt added a guardrail?"*, *"what if temperature
were 0?"*, *"what if this tool call had different arguments?"*,
*"what if this step raised TimeoutError?"*. None of these are
expressible today.

This proposer favours **breadth** — add the largest reasonable set of
new typed substitution dataclasses, each one small, dataclass-shaped,
and round-trip-serialisable through `branch_io.substitution_to_dict`
/ `substitution_from_dict`.

## New substitutions (each a dataclass with `at_step`)

1. **`SystemPromptSubstitution(at_step, system_text, mode={"prepend",
   "replace","append"})`** — patch only the system message instead of
   replacing the whole `messages` list. Touches `inputs["messages"]`.
2. **`MessagePatchSubstitution(at_step, index, new_message)`** — replace
   one message at index `i`; negative indices supported.
3. **`SamplingSubstitution(at_step, temperature=None, top_p=None,
   max_tokens=None, seed=None)`** — set any of the four standard
   sampling knobs in `inputs`.
4. **`ToolArgumentsSubstitution(at_step, new_arguments)`** — for
   `tool_call` steps; mutate `inputs["arguments"]` (changes the input
   hash, so the tool actually runs again with new args).
5. **`ToolSpecSubstitution(at_step, new_tools)`** — replace the
   `inputs["tools"]` JSON-schema list available to an LLM step; lets
   you ask "would the agent have called the right tool if it knew
   about `refund_invoice`?".
6. **`RouterOptionsSubstitution(at_step, new_options)`** — change the
   list of router branch names the recorded router was choosing
   among.
7. **`RaiseSubstitution(at_step, exception_type, message)`** — force
   the step to raise instead of returning. The replay engine catches
   it and stores `{"__error__": {"type": ..., "message": ...}}` as
   the output; descendants become dirty as usual. Lets you debug
   "what happens downstream if THIS tool call had failed?".
8. **`OutputPatchSubstitution(at_step, json_pointer, new_value)`** —
   like `ToolOutputSubstitution` but mutates one field of the
   recorded output (RFC 6901 pointer like `/result/customer/id`)
   rather than replacing the whole thing. Marks step dirty.

## Wiring

* All new subs registered in `branch_io._TYPE_MAP` and exported from
  `stepback/__init__.py`.
* `parse_substitution_spec` extended with new kinds:
  - `system@step:N=:inline:"You are..."` (mode=`replace`)
  - `system_prepend@step:N=...`, `system_append@step:N=...`
  - `message@step:N=:idx=2,inline:{...}`
  - `sampling@step:N=:inline:{"temperature":0.0,"max_tokens":256}`
  - `tool_args@step:N=:inline:{"vendor":"Acme"}`
  - `tools@step:N=:inline:[...]`
  - `router_options@step:N=:inline:["a","b","c"]`
  - `raise@step:N=TimeoutError:request timed out`
  - `output_patch@step:N=:ptr=/result/total,inline:0.0`
* `replay.py` only needs ONE change: special-case `RaiseSubstitution`
  the way `ToolOutputSubstitution` is special-cased, because it
  forces an output rather than mutating an input.

## Tests

For each new substitution kind:
1. Round-trip through `substitution_to_dict` /
   `substitution_from_dict`.
2. CLI spec parses to the same dataclass.
3. Replay against a 3-step toy trace shows the right
   `(cache_hit, dirty, output)` triple.
4. End-to-end test: run a fixture trace through
   `stepback replay --substitute <new spec>` and assert the
   diff-replay JSON has `divergent_step_count >= 1`.

## Risks / why this might not be the best framing

* 8 new dataclasses + 8 new spec verbs is a lot of surface; some pairs
  overlap (e.g. `MessagePatchSubstitution` ⊃ `SystemPromptSubstitution`
  if `index=0`).
* No abstraction — each new debugging question requires a new
  dataclass. Doesn't scale to "what if I want to flip a single
  boolean inside the recorded outputs?" without adding yet another.
