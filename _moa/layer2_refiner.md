# Layer 2 Refiner — pick a backbone, integrate the rest

## Decision

Take **Proposer 2's structure** as the backbone (one generic
`InputsPatchSubstitution` + a small set of typed conveniences +
the `is_output_forcing` predicate refactor in `replay.py`). It scales
better than P1's "one dataclass per question" and avoids P3's
second persistence format.

Then layer in the most useful pieces from the others:

* From **P1**: the high-frequency typed conveniences
  `SystemPromptSubstitution` (with mode prepend/replace/append),
  `MessagePatchSubstitution` (one message at index `i`), and
  `ToolArgumentsSubstitution`. Keep them — they're the things users
  would actually type. Drop `ToolSpecSubstitution`,
  `RouterOptionsSubstitution`, `OutputPatchSubstitution` — all are
  trivially expressible by `InputsPatchSubstitution` /
  `OutputsPatchSubstitution`. Adopt P1's `RaiseSubstitution`.
* From **P2**: `InputsPatchSubstitution`, `OutputsPatchSubstitution`,
  `SamplingSubstitution` (with strict validation), `RaiseSubstitution`,
  the `_apply_patch` mini-engine, the `is_output_forcing` predicate
  refactor, and the JSON Patch dialect (subset).
* From **P3**: `is_output_forcing` is essentially the same idea as
  P3's `CompositeSubstitution` for cleaner dispatch — keep that
  spirit. **Drop** scenarios + `.sbs` (second format = scope creep
  beyond one coherent theme). **Drop** `AssertSubstitution` for now
  (pre-step invariants are a different feature — better as a
  separate round). **Keep** the `mode={"prepend","replace","append"}`
  pattern for `SystemPromptSubstitution` (P1's idea, applied
  cleanly).

## Final substitution roster (8 new + 5 existing)

Existing: PromptSubstitution, ModelSubstitution, ToolOutputSubstitution,
PolicySubstitution, RouterSubstitution.

New, all dataclasses with `at_step`:

1. `SystemPromptSubstitution(at_step, system_text, mode="replace")`
2. `MessagePatchSubstitution(at_step, index, new_message)`
3. `SamplingSubstitution(at_step, temperature=None, top_p=None,
   max_tokens=None, seed=None)`
4. `ToolArgumentsSubstitution(at_step, new_arguments)`
5. `InputsPatchSubstitution(at_step, ops)`
6. `OutputsPatchSubstitution(at_step, ops)` — output-forcing
7. `RaiseSubstitution(at_step, exception_type, message="")` —
   output-forcing
8. (drop: keep at 7 — eight is a clean number for one round)

Output-forcing predicate: `Substitution.is_output_forcing`
(default False; True for `ToolOutputSubstitution`,
`OutputsPatchSubstitution`, `RaiseSubstitution`).

`replay.py` change: replace the `isinstance(sub, ToolOutputSubstitution)`
branch with `if sub.is_output_forcing()`. Each output-forcing sub
gets a `force_output(recorded_step) -> Any` method:

* `ToolOutputSubstitution.force_output` → `{"result": fake_response}`
* `OutputsPatchSubstitution.force_output` → apply ops to deep-copy
  of `recorded_step["outputs"]`, return result
* `RaiseSubstitution.force_output` → `{"__error__": {"type":...,
  "message":...}}`

## Five axes L3 should deepen

1. **`_apply_patch` semantics** — pin every op + every error path,
   especially RFC 6901 escapes (`~0`, `~1`), `-` end-of-array
   sentinel for `add`, missing key vs out-of-bounds index.
2. **`SamplingSubstitution` validation** — exact ranges, what
   counts as None vs explicit, key names in the LLM call inputs
   (`temperature` / `top_p` / `max_tokens` / `seed`), and the case
   where the recorded step didn't have those fields at all.
3. **`SystemPromptSubstitution` modes** — what if there's no
   system message in `messages`? `replace` adds one at index 0;
   `prepend` adds; `append` adds at end. Pin these.
4. **`MessagePatchSubstitution` negative indices** — `-1` =
   last; raise on out-of-bounds (don't silently expand). Allow
   `index=len(messages)` to mean append.
5. **CLI spec parsing** — pick a small, learnable grammar for
   the new verbs that matches the existing `KIND@step:N=BODY`
   shape. Multi-arg subs use `:kv:k=v,k=v` or `:inline:JSON`.

## Output-forcing dispatch sketch

```python
class Substitution:
    def is_output_forcing(self) -> bool: return False
    def force_output(self, recorded_step: dict) -> Any:
        raise NotImplementedError
```

In `replay.py`:
```python
tool_override: Any = sentinel
for sub in subs.at(sid):
    if sub.is_output_forcing():
        tool_override = sub.force_output(rec)
    else:
        sub.apply(cur_inputs, rec)
```

This is the only `replay.py` change required (modulo cost handling
for the `__error__` path, which is `0.0`).

## Tests plan

* `tests/test_substitutions_patch.py` — `_apply_patch` table:
  add/replace/remove/test/copy/move × array+object × good+bad
  paths; `~` escaping; `-` sentinel.
* `tests/test_substitutions_new.py` — round-trip, `force_output`,
  `apply` per new sub; sampling validation rejects
  `temperature=-0.1`, `top_p=1.5`, `max_tokens=0`.
* `tests/test_e2e_substitutions.py` — fixture trace; for each new
  sub kind, run `replay()` and assert
  `(cache_hits, dirty_count, total_cost_delta)` triple matches a
  pinned expected value.
* `tests/test_cli_new_subs.py` — every new spec verb parses to the
  expected dataclass; `stepback replay --substitute system@step:0=
  :inline:"You are concise"` runs end-to-end against a fixture trace.
