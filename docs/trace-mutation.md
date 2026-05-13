# `Trace` mutation semantics

This document is the canonical reference for **which methods on `Trace` (and
its companions `Branch` / module-level helpers) mutate state in place and
which return fresh objects.** It exists because the navigation methods
(`step_back`, `goto`) read like functional combinators in the README example
but are in fact *mutating*: they return `self` only as a convenience for
fluent chaining, not as a hint that they produced a new `Trace`.

stepback follows a deliberate split:

* **Navigation, substitution staging, and bisect bookkeeping mutate the
  loaded `Trace` in place.** A `Trace` is a thin handle around an immutable
  on-disk `.sb` file plus a small bag of *session* state — cursor,
  pending substitutions, last-bisect probe count. Mutating that session
  state is what the reversible-debugger UX is built on.
* **Replay never mutates the `Trace`.** `replay_forward`, `run_replay`,
  `compare_branches`, and `bisect` each compute a fresh
  `ReplayResult` (or `BranchDiff` / `StepView`) without touching
  `recorded_steps` or `header`. The recorded bytes on disk are likewise
  never written back — a `.sb` file is append-only by construction.
* **`branch_at` produces a fresh `Branch`.** Branch substitutions live on
  the `Branch`, not on the parent `Trace`, so two branches built from the
  same `Trace` are independent.
* **The top-level corpus helpers `sweep_traces`, `minimize_substitutions`,
  `ddmin_substitutions`, `find_all_minimal`, and `attribute_substitutions`
  do not mutate the traces they are given.** They open each trace, build
  branches, and replay; the `Trace` objects you pass in (or that the
  helpers open internally) are discarded after the call.

Each `Trace` corresponds to exactly one `.sb` file. Loading the same path
twice with `stepback.replay(path)` returns two **independent** `Trace`
handles with their own cursors and pending-substitution sets.

## Quick reference

| Method / function                     | Receiver               | Mutates? | Returns                |
| ------------------------------------- | ---------------------- | -------- | ---------------------- |
| `replay(path, *, hmac_key=None)`      | module-level           | n/a      | new `Trace`            |
| `Trace.goto(step_id)`                 | `Trace` cursor         | **yes**  | `self`                 |
| `Trace.step_back(*, to=None)`         | `Trace` cursor         | **yes**  | `self`                 |
| `Trace.step_forward()`                | `Trace` cursor         | **yes**  | `self`                 |
| `Trace.current_step()`                | —                      | no       | recorded step `dict`   |
| `Trace.substitute(*subs)`             | `Trace.pending_subs`   | **yes**  | `self`                 |
| `Trace.reset_substitutions()`         | `Trace.pending_subs`   | **yes**  | `self`                 |
| `Trace.branch_at(step_id, name)`      | —                      | no       | new `Branch`           |
| `Trace.compare_branches(a, b)`        | —                      | no¹      | new `BranchDiff`       |
| `Trace.replay_forward(executor=None)` | —                      | no       | new `ReplayResult`     |
| `Trace.run_replay(subs, executor)`    | —                      | no       | new `ReplayResult`     |
| `Trace.bisect(good, bad, predicate)`  | `_last_bisect_probes`² | **yes**² | `Optional[StepView]`   |
| `Trace.minimize(subs, predicate, …)`  | —                      | no³      | `MinimizationResult`   |
| `Branch.substitute(*subs)`            | `Branch.substitutions` | **yes**  | `self`                 |
| `Branch.replay_forward(executor=None)`| `Branch.result`        | **yes**⁴ | `ReplayResult`         |
| `sweep_traces(paths, subs, …)`        | —                      | no       | new `SweepReport`      |
| `minimize_substitutions(trace, …)`    | —                      | no       | new `MinimizationResult` |
| `ddmin_substitutions(trace, …)`       | —                      | no       | new `MinimizationResult` |
| `find_all_minimal(trace, …)`          | —                      | no       | `list[MinimizationResult]` |
| `attribute_substitutions(trace, …)`   | —                      | no       | attribution dict       |

¹ `compare_branches` calls `Branch.replay_forward` on each branch if its
  `result` is `None`, which mutates that branch's `result` cache (see
  note ⁴). It does not mutate the `Trace`.

² `bisect` updates the read-only `Trace.last_bisect_probes` property to the
  number of probes performed by the binary search. This is part of the
  documented v0.1 API and is the cost metric users assert on. The
  `Trace`'s cursor and `pending_subs` are not touched.

³ `Trace.minimize` is a thin shim around `ddmin_substitutions(trace, …)`;
  neither builds nor stores any per-`Trace` state.

⁴ `Branch.replay_forward` memoises its `ReplayResult` on `Branch.result`.
  Calling it again will rerun replay and overwrite that cache. It never
  mutates the owning `Trace`.

## Idiomatic usage

### Mutating cursor + pending subs is the intended UX

```python
from stepback import replay, ToolOutputSubstitution

t = replay("incident.sb")
t.step_back(to="step:7")            # mutates cursor
t.substitute(ToolOutputSubstitution("step:7", outputs={"k": "v"}))  # stages
result = t.replay_forward()         # returns fresh ReplayResult; no mutation
t.reset_substitutions()             # clears pending_subs
```

Because every navigation/substitution method returns `self`, the same code
chains:

```python
result = (
    replay("incident.sb")
    .step_back(to="step:7")
    .substitute(ToolOutputSubstitution("step:7", outputs={"k": "v"}))
    .replay_forward()
)
```

The chain still mutates the intermediate `Trace`; it is just that the
intermediate `Trace` is unnamed.

### Branches isolate substitutions

```python
t = replay("incident.sb")
a = t.branch_at("step:5", name="baseline")           # new Branch
b = t.branch_at("step:5", name="gpt-5-tool-redirect") # new Branch
b.substitute(ToolOutputSubstitution("step:7", outputs={"k": "v2"}))
diff = t.compare_branches(a, b)   # replays each branch as needed
```

`a.substitutions` and `b.substitutions` are independent `SubstitutionSet`
instances. `t.pending_subs` is unaffected. Calling `t.replay_forward()`
after the branch construction will *not* see any of `a`'s or `b`'s
substitutions — only what was staged on `t.pending_subs`.

### Minimisation and sweeps are read-only over their inputs

`Trace.minimize`, `minimize_substitutions`, and `ddmin_substitutions` build
their own `Branch` / `SubstitutionSet` objects internally and call
`run_replay` repeatedly. None of them mutates the receiver `Trace`. The
returned `MinimizationResult` carries the 1-minimal triggering subset and
the oracle call counts.

`sweep_traces` opens each path with `stepback.replay`, builds two branches,
and discards the `Trace` after computing the per-trace
`SweepResult`. The list of paths you pass in is not modified.

## How "fresh" is "fresh"?

`replay_forward`, `compare_branches`, and `bisect` all build their
`StepView` / `BranchDiff` / `ReplayResult` from a deep copy of the
recorded inputs (see `copy.deepcopy(rec["inputs"])` in
`stepback.replay.Trace.run_replay`), so mutating the `inputs` dict on a
returned `StepView` will not propagate back to `Trace.recorded_steps`.
The `outputs` field on a cache-hit step is the recorded dict itself
(by reference) — treat it as read-only.

## See also

* `stepback/replay.py` — implementation of `Trace`, `Branch`,
  `ReplayResult`, `StepView`.
* `stepback/sweep.py` — `sweep_traces` and `SweepReport`.
* `stepback/minimize.py` — `MinimizationResult`, all delta-debugging
  strategies.
* `stepback/__init__.py` — the public surface that re-exports the names
  above.
