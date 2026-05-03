# Layer 3 Refiner — final pinned design + concrete file plan

## What's locked in from Layer 2

L2 picked **Proposer 2's `InputsPatchSubstitution` + `is_output_forcing`
predicate** as the backbone, layered **P1's `SystemPromptSubstitution`
(with `mode`), `MessagePatchSubstitution`, `ToolArgumentsSubstitution`,
`RaiseSubstitution`**, and **P2's `OutputsPatchSubstitution` and
`SamplingSubstitution`**. Drops P1's `ToolSpecSubstitution`,
`RouterOptionsSubstitution`, and `OutputPatchSubstitution`
(subsumed by patch ops). Drops P3's `Scenario` / `.sbs` /
`AssertSubstitution` (separate-round material).

L3 keeps all of that and tightens the contract along the five axes
L2 flagged for deepening, then pins the concrete file plan.

## L3 deepenings (~30% more depth)

### A. `_apply_patch` semantics — pinned

Subset of RFC 6902 + RFC 6901 we implement (`stepback/jsonpatch.py`,
new module, ~120 LoC):

* Ops: `add`, `replace`, `remove`, `test`, `copy`, `move`. Any
  other op → `PatchInvalidOp`.
* `path = ""` → whole-document target. `replace` on root replaces
  the doc; `remove` on root → `PatchInvalidOp("cannot remove root")`.
* Path tokens decoded per RFC 6901: `~1` → `/`, `~0` → `~`.
  Decoded in **that order** (test case: `~01` → `~1`, not `/`).
* Array `-` end-sentinel valid only for `add`. For `replace` /
  `remove` / `test` on arrays, `-` is `PatchPathNotFound`.
* Negative array indices (`/foo/-1`) → `PatchPathNotFound` (RFC
  6901 says indices must be non-negative integers without leading
  zeros; we enforce both).
* `add` on an existing object key replaces the value (RFC 6902
  §4.1). `add` on a missing object key adds it. `add` on an
  intermediate missing object key → `PatchPathNotFound`.
* `test` failure → `PatchTestFailed` (subclass of `PatchError`),
  remaining ops do NOT execute (transactional within one
  substitution).
* `copy` / `move`: read-then-add semantics, with `move` removing
  source after read; if source path equals target path, `move` is
  a no-op (per RFC 6902 example).
* Comparison for `test`: deep equality after `canonical_json` round
  trip (so `1` == `1.0` is **False**, matching JSON Patch spec).
* All errors are subclasses of `PatchError(ValueError)`.

### B. `SamplingSubstitution` validation

In `__post_init__`:
* `temperature`: None or `0.0 <= x <= 2.0`. Else `ValueError`.
* `top_p`: None or `0.0 <= x <= 1.0`. Else `ValueError`.
* `max_tokens`: None or positive `int`. `0` rejected.
* `seed`: None or `int` (any value, including negative).
* `apply()`: only sets the keys whose value is not None; does NOT
  delete pre-existing keys. So `SamplingSubstitution(temperature=0.0)`
  leaves `inputs["max_tokens"]` untouched even if it was set.
* If the recorded step's `inputs` dict has no `temperature` key and
  the substitution sets one, that's fine — the inputs hash will
  change and the step becomes dirty.

### C. `SystemPromptSubstitution` modes

`mode in {"replace", "prepend", "append"}`. Validated in
`__post_init__`.

* `replace`: find the first message with `role == "system"` and
  set its `content` to `system_text`. If none exists, **insert one
  at index 0**.
* `prepend`: prepend `system_text + "\n\n"` to the existing first
  system message's content. If none exists, insert one at index 0
  with just `system_text`.
* `append`: append `"\n\n" + system_text` to the existing first
  system message's content. If none exists, insert one at index 0.
* Operates on a deep copy of `inputs["messages"]` and assigns back
  (no in-place mutation of recorded structures).
* If `inputs` has no `messages` key, set it to a singleton list
  `[{"role":"system","content":system_text}]`.

### D. `MessagePatchSubstitution` index semantics

* `index` is an `int`. Negative indices supported per Python
  semantics (`-1` = last).
* `index == len(messages)` → append (extend by one).
* Out of range otherwise → `IndexError` (raised from `apply`,
  surfaces during replay; not silently dropped).
* `new_message` must be a dict with at least `role` and `content`
  keys; `__post_init__` enforces.

### E. CLI spec grammar — pinned

Existing `KIND@step:N=BODY` extended with these new verbs (parser
in `branch_io.parse_substitution_spec`):

* `system@step:N=:inline:"You are concise"`
  (mode defaults to `replace`)
* `system_prepend@step:N=:inline:"Be brief."`
* `system_append@step:N=:inline:"Cite sources."`
* `message@step:N=:idx=2,inline:{"role":"user","content":"hi"}`
  (or `:idx=2,path=msg.json`)
* `sampling@step:N=:kv:temperature=0.0,max_tokens=256` —
  parsed as kv pairs, numeric coercion: `int` if integer-shaped,
  else `float`. `seed=null` accepted as None.
* `sampling@step:N=:inline:{"temperature":0.0,"max_tokens":256}`
  — JSON form also accepted.
* `tool_args@step:N=:inline:{"vendor":"Acme"}` (or path).
* `inputs_patch@step:N=:inline:[{"op":"replace","path":"/model",
  "value":"gpt-4o"}]` (or path).
* `outputs_patch@step:N=:inline:[{"op":"replace","path":"/result",
  "value":42}]` (or path).
* `raise@step:N=TimeoutError:request timed out` —
  `TYPE:MSG`, message optional (`raise@step:N=TimeoutError`).

Unknown KIND → existing `ValueError` (with the new kinds added to
the error message).

## Output-forcing dispatch in `replay.py`

Single change at the substitution-application loop (currently
`stepback/replay.py:331-337`):

```python
tool_override: Any = sentinel
for sub in subs.at(sid):
    if sub.is_output_forcing():
        tool_override = sub.force_output(rec)
    else:
        sub.apply(cur_inputs, rec)
```

`Substitution.is_output_forcing()` returns `False` by default;
`ToolOutputSubstitution`, `OutputsPatchSubstitution`, and
`RaiseSubstitution` each override to return `True` and implement
`force_output(recorded_step) -> Any`.

Cost handling in the existing `if cache_hit / else` block already
defaults to `float(rec.get("cost_usd", 0.0))` for non-llm steps and
recomputes for llm-shaped outputs; for `RaiseSubstitution`'s
`{"__error__":...}` output, `compute_cost(...)` will see no `usage`
field and return `0.0`. No code change needed there.

## Concrete file plan

* **`stepback/jsonpatch.py`** (new, ~120 LoC) — `_apply_patch`,
  `PatchError` hierarchy.
* **`stepback/substitutions.py`** (~+170 LoC) — 7 new dataclasses
  (`SystemPromptSubstitution`, `MessagePatchSubstitution`,
  `SamplingSubstitution`, `ToolArgumentsSubstitution`,
  `InputsPatchSubstitution`, `OutputsPatchSubstitution`,
  `RaiseSubstitution`) + `is_output_forcing` / `force_output`
  on the base class.
* **`stepback/branch_io.py`** (~+60 LoC) — extend `_TYPE_MAP` with
  the 7 new kinds; extend `parse_substitution_spec` with the new
  verbs (factor out the `:idx=N,...`, `:kv:k=v,...`, `:inline:JSON`,
  and bare-`PATH` body parsers).
* **`stepback/replay.py`** (~+5/-3 LoC) — replace the
  `isinstance(sub, ToolOutputSubstitution)` branch with the
  predicate dispatch above.
* **`stepback/__init__.py`** (~+8 LoC) — re-export the new
  substitution classes.
* **`tests/test_jsonpatch.py`** (new, ~140 LoC) — table-driven
  tests for every op + every error path + RFC 6901 escapes.
* **`tests/test_substitutions_new.py`** (new, ~200 LoC) — per-sub
  unit tests: `apply` / `force_output` / round-trip through
  `substitution_to_dict`/`from_dict` / validation rejections.
* **`tests/test_cli_new_subs.py`** (new, ~150 LoC) — every new
  spec verb parses; one end-to-end `stepback replay --substitute
  system@step:0=...` against a fixture trace asserts the report
  contains `dirty=`.
* **`tests/test_e2e_substitutions.py`** (new, ~120 LoC) — build a
  small in-memory recorded trace, replay with one substitution of
  each new kind, assert `(cache_hits, dirty_count)` matches pinned
  expected values, plus one composite test with two new subs
  layered.

## Lessons-applied trailer

`Lessons-applied: L470161` — diversify proposers (P1 breadth, P2
power, P3 orchestration), refine by picking ONE backbone (P2) and
integrating only the strongest pieces from the others rather than
averaging.
