# Proposer 2 — Power: a generic JSON-Patch substitution + 3 typed ones

## Theme
The current 5 substitutions are all "shaped" — each carries a
custom field (`new_messages`, `new_model_id`, ...) that maps to a
specific input key. Adding a new substitution per debugging
question doesn't scale. This proposer favours **expressive power**:
introduce one generic `InputsPatchSubstitution` that can mutate
*any* path inside `inputs`, plus three high-frequency typed
substitutions for ergonomics, plus one substitution that targets
*outputs* instead of inputs.

## New substitutions

1. **`InputsPatchSubstitution(at_step, ops)`** — `ops` is a list of
   RFC 6902 JSON Patch operations (`{"op":"replace","path":"/messages
   /0/content","value":"..."}`, `{"op":"add"}`, `{"op":"remove"}`,
   `{"op":"test"}`). Implementation: a tiny ~50-LoC pure-Python
   `_apply_patch(doc, ops)` (no `jsonpatch` dependency — keeps the
   wheel lean). `test` ops short-circuit-raise `PatchTestFailed` so
   you can pin assumptions: *"this step's temperature was 0.7"*.
2. **`OutputsPatchSubstitution(at_step, ops)`** — same JSON Patch
   shape, but applied to a deep-copy of the recorded outputs. Marks
   step dirty (output diverges) without re-invoking the underlying
   tool / LLM. Generalises `ToolOutputSubstitution`.
3. **`SamplingSubstitution(at_step, temperature=None, top_p=None,
   max_tokens=None, seed=None)`** — special-case typed sub for the
   most common LLM debugging knob; equivalent to a 1-op
   `InputsPatchSubstitution` but with strict field validation
   (rejects `temperature=1.5`, `top_p=2.0`, etc.).
4. **`RaiseSubstitution(at_step, exception_type, message)`** —
   force an exception. Like `ToolOutputSubstitution`, this is an
   *output-forcing* substitution; the replay engine catches and
   stores `{"__error__": {...}}`.

## Wiring

* `branch_io._TYPE_MAP` updated; `parse_substitution_spec` gains:
  - `inputs_patch@step:N=:inline:[{"op":"replace",...}]`
  - `inputs_patch@step:N=path/to/ops.json`
  - `outputs_patch@step:N=...`
  - `sampling@step:N=temperature=0.0,max_tokens=256` (kv-pair
    grammar — easier to type than JSON for sampling)
  - `raise@step:N=TimeoutError:request timed out`
* `replay.py` gets ONE new branch: `OutputsPatchSubstitution` and
  `RaiseSubstitution` join `ToolOutputSubstitution` as the only
  sub kinds that override `cur_outputs` rather than mutating inputs.
  Refactor: introduce `Substitution.is_output_forcing` predicate
  (default False; True for those three classes) to make the dispatch
  clean instead of `isinstance` chains.

## JSON Patch dialect (subset, tested)

* `op`: `add` | `replace` | `remove` | `test` | `copy` | `move`
* `path`: `/foo/0/bar` (RFC 6901; `~1` for `/`, `~0` for `~`)
* `value`: any JSON value (required for `add`/`replace`/`test`)
* `from`: source path for `copy`/`move`
* Errors: `PatchPathNotFound`, `PatchTestFailed`,
  `PatchInvalidOp` — all subclasses of `PatchError(ValueError)`.

## Tests

* `tests/test_substitutions_patch.py`:
  - 12 unit cases of `_apply_patch` covering every op + edge cases
    (negative array index → reject; `path:""` = whole doc; `~`
    escaping).
  - Round-trip through `substitution_to_dict`/`from_dict`.
  - CLI spec parsing for the new kinds.
* `tests/test_e2e_inputs_patch.py`: replay a 5-step trace,
  patch `inputs.messages.[0].content` at step:2, assert step:2
  becomes dirty and the LLM is re-invoked.
* `tests/test_e2e_raise_substitution.py`: force step:3 to raise
  `TimeoutError`, assert downstream step:4 sees the error in its
  context and is itself dirty.

## Risks

* JSON Patch is more powerful than typical users want; CLI ergonomics
  hurt (`:inline:[{"op":"replace","path":"/messages/0/content",
  "value":"hi"}]` is a mouthful).
* Output-forcing substitutions multiply the special-case dispatch in
  `replay.py` — needs the `is_output_forcing` predicate refactor to
  avoid `isinstance` chains.
* No coverage of "what if this step's *step kind* itself were
  different?" — out of scope.
