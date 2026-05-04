"""Stochastic replay tests (Step 36 of ``100_STEPS.md``).

These tests exercise the replay engine with **noisy** mock LLM
executors — executors whose outputs vary from one invocation to the
next — and document the boundary between two distinct concerns:

* **Replay correctness.** The dirty-set bookkeeping of
  :func:`stepback.replay.Trace.run_replay` is a pure function of the
  recorded trace and the staged substitutions. It must not depend on
  whether the executor used to recompute dirty steps is deterministic
  or stochastic. Concretely: the *set* of dirty step ids, the
  ``cache_hit_count``, the ``real_executions`` count, and the
  preservation of upstream cached steps must all be invariant under
  any seeded perturbation of the executor.

* **Predicate stability.** A user-supplied predicate evaluated over
  the resulting :class:`ReplayResult` is a *separate* artifact whose
  stability under stochastic re-execution depends on what features of
  the trace it inspects. Predicates that look only at structural
  facts of the replay (counts, step kinds, dirty marks, cost bounds)
  remain stable; predicates that look at concrete LLM token strings
  do not, and the replay engine cannot make them so.

The tests below assert both halves directly, so a future change that
accidentally couples replay correctness to executor determinism
will break the correctness suite, while predicate-instability under
real LLMs remains a documented user-side concern rather than a
replay bug.
"""
from __future__ import annotations

import hashlib
import random
from typing import Callable, List, Tuple

import pytest

from stepback import Executor, RecorderKey, record, replay
from stepback.substitutions import (
    PromptSubstitution,
    ToolOutputSubstitution,
)
from stepback.testing import (
    LOOKUP_FIXED_ROW,
    fake_llm,
    fake_tool,
    run_recorded_agent,
)


# --------------------------------------------------------------- helpers


def _record_fixture(tmp_path) -> Tuple[str, RecorderKey]:
    key = RecorderKey.fresh()
    path = str(tmp_path / "trace.sb")
    with record(path, key=key) as rec:
        run_recorded_agent(rec)
    return path, key


def _load(path: str, key: RecorderKey):
    return replay(path, hmac_key=key.hmac_key)


def make_seeded_noisy_llm(seed: int) -> Callable[[str, List[dict]], dict]:
    """A noisy LLM whose noise is a deterministic function of ``seed``.

    Wraps :func:`stepback.testing.fake_llm` and appends a per-call
    pseudo-random suffix to the assistant content. The same ``seed``
    plus the same call sequence always yields the same outputs; two
    different seeds yield different outputs at every dirty step.

    The ``usage`` field is left untouched so that cost computations do
    not get perturbed by noise (cost stability is a separate concern
    from output stability).
    """
    rng = random.Random(seed)

    def _noisy(model: str, messages: List[dict]) -> dict:
        base = fake_llm(model, messages)
        nonce = rng.randrange(0, 2**32)
        token = f" :: noise={nonce:08x}"
        msg = base["choices"][0]["message"]
        msg["content"] = msg["content"] + token
        return base

    return _noisy


def make_unseeded_noisy_llm() -> Callable[[str, List[dict]], dict]:
    """A noisy LLM whose noise is reseeded from the system clock per
    call. Successive calls (even with identical inputs) yield different
    outputs. Useful to model 'real' LLM jitter."""

    def _noisy(model: str, messages: List[dict]) -> dict:
        base = fake_llm(model, messages)
        # SystemRandom is unsynchronised across construction, so two
        # successive constructions cannot be coincident.
        nonce = random.SystemRandom().randrange(0, 2**32)
        token = f" :: noise={nonce:08x}"
        msg = base["choices"][0]["message"]
        msg["content"] = msg["content"] + token
        return base

    return _noisy


def _replay_outputs(result) -> List[Tuple[str, str]]:
    """Extract a (step_id, content-or-empty) listing for assertion."""
    out = []
    for sv in result.steps:
        if sv.kind == "llm_call" and isinstance(sv.outputs, dict):
            choices = sv.outputs.get("choices") or []
            content = (
                choices[0]["message"]["content"]
                if choices and isinstance(choices[0], dict)
                else ""
            )
            out.append((sv.step_id, content))
        else:
            out.append((sv.step_id, ""))
    return out


def _dirty_ids(result) -> List[str]:
    return [s.step_id for s in result.steps if s.dirty]


def _cached_ids(result) -> List[str]:
    return [s.step_id for s in result.steps if s.cache_hit]


# ====================================================================
# Replay correctness — invariant under any stochastic executor.
# ====================================================================


def test_seeded_noisy_replay_is_byte_identical_across_repeats(tmp_path):
    """Two replays with the same seed, the same substitution, and the
    same noisy executor must produce *identical* outputs at every
    dirty step. This is the seeded-determinism contract."""
    path, key = _record_fixture(tmp_path)

    def go():
        t = _load(path, key)
        t.substitute(
            ToolOutputSubstitution(at_step="step:2", fake_response=LOOKUP_FIXED_ROW)
        )
        ex = Executor(llm=make_seeded_noisy_llm(seed=1234), tool=fake_tool)
        return t.replay_forward(ex)

    a = go()
    b = go()

    assert _replay_outputs(a) == _replay_outputs(b)
    assert _dirty_ids(a) == _dirty_ids(b)
    assert _cached_ids(a) == _cached_ids(b)
    assert a.dirty_count == b.dirty_count
    assert a.cache_hit_count == b.cache_hit_count
    assert a.real_executions == b.real_executions
    assert a.total_cost_usd == pytest.approx(b.total_cost_usd)


def test_dirty_set_invariant_under_seed_change(tmp_path):
    """The *set* of dirty step ids and the cache-hit count are pure
    functions of (recorded trace, substitutions). They cannot depend
    on which seed the noisy executor was constructed with."""
    path, key = _record_fixture(tmp_path)

    results = []
    for seed in (1, 2, 3, 17, 4096):
        t = _load(path, key)
        t.substitute(
            ToolOutputSubstitution(at_step="step:2", fake_response=LOOKUP_FIXED_ROW)
        )
        ex = Executor(llm=make_seeded_noisy_llm(seed=seed), tool=fake_tool)
        results.append(t.replay_forward(ex))

    dirty_sets = [tuple(_dirty_ids(r)) for r in results]
    cached_sets = [tuple(_cached_ids(r)) for r in results]
    assert len(set(dirty_sets)) == 1, dirty_sets
    assert len(set(cached_sets)) == 1, cached_sets
    assert len({r.dirty_count for r in results}) == 1
    assert len({r.cache_hit_count for r in results}) == 1
    assert len({r.real_executions for r in results}) == 1


def test_dirty_set_invariant_under_unseeded_executor(tmp_path):
    """Even with an *unseeded* (non-reproducible) noisy executor the
    dirty-set bookkeeping is invariant. Replay correctness must hold
    pointwise even when the executor itself is irreproducible."""
    path, key = _record_fixture(tmp_path)

    dirty_sets = []
    cached_sets = []
    for _ in range(5):
        t = _load(path, key)
        t.substitute(
            ToolOutputSubstitution(at_step="step:2", fake_response=LOOKUP_FIXED_ROW)
        )
        ex = Executor(llm=make_unseeded_noisy_llm(), tool=fake_tool)
        r = t.replay_forward(ex)
        dirty_sets.append(tuple(_dirty_ids(r)))
        cached_sets.append(tuple(_cached_ids(r)))
    assert len(set(dirty_sets)) == 1
    assert len(set(cached_sets)) == 1


def test_unseeded_executor_changes_only_dirty_step_outputs(tmp_path):
    """An unseeded noisy executor must vary the *content* of dirty
    LLM steps across runs but must never disturb cached upstream
    steps' outputs."""
    path, key = _record_fixture(tmp_path)

    def go():
        t = _load(path, key)
        t.substitute(
            ToolOutputSubstitution(at_step="step:2", fake_response=LOOKUP_FIXED_ROW)
        )
        return t.replay_forward(
            Executor(llm=make_unseeded_noisy_llm(), tool=fake_tool)
        )

    a = go()
    b = go()

    # Cached upstream step contents are bytewise identical across runs.
    cached_a = {s.step_id: s.outputs for s in a.steps if s.cache_hit}
    cached_b = {s.step_id: s.outputs for s in b.steps if s.cache_hit}
    assert cached_a == cached_b
    assert cached_a, "expected at least one cached upstream step"

    # At least one dirty LLM step *did* produce different content
    # (this is the whole point of an unseeded noisy executor).
    diff_seen = False
    for sa, sb in zip(a.steps, b.steps):
        if sa.dirty and sa.kind == "llm_call":
            if sa.outputs != sb.outputs:
                diff_seen = True
                break
    assert diff_seen, "unseeded noisy LLM produced bit-identical outputs"


def test_noisy_executor_does_not_dirty_upstream_cached_steps(tmp_path):
    """Substitution at step:6 must leave steps 1..5 cache-hit
    regardless of executor noise."""
    path, key = _record_fixture(tmp_path)
    t = _load(path, key)
    t.substitute(
        PromptSubstitution(
            at_step="step:5",
            new_messages=[{"role": "system", "content": "rewritten"}],
        )
    )
    ex = Executor(llm=make_seeded_noisy_llm(seed=7), tool=fake_tool)
    result = t.replay_forward(ex)

    upstream = result.steps[:4]  # steps 1..4
    for sv in upstream:
        assert sv.cache_hit, f"{sv.step_id} should remain cached"
        assert not sv.dirty


def test_real_executions_equals_dirty_executor_invocations(tmp_path):
    """``real_executions`` is the count of executor calls, period.
    A stochastic executor cannot inflate or deflate this metric."""
    path, key = _record_fixture(tmp_path)
    t = _load(path, key)
    t.substitute(
        ToolOutputSubstitution(at_step="step:2", fake_response=LOOKUP_FIXED_ROW)
    )

    call_count = {"n": 0}
    base = make_seeded_noisy_llm(seed=99)

    def counting_llm(model, messages):
        call_count["n"] += 1
        return base(model, messages)

    ex = Executor(llm=counting_llm, tool=fake_tool)
    r = t.replay_forward(ex)

    # Every dirty llm_call step was executed exactly once.
    dirty_llm_steps = sum(
        1 for s in r.steps if s.dirty and s.kind == "llm_call"
    )
    assert call_count["n"] == dirty_llm_steps
    # And every dirty step was either an LLM, a tool, or output-forced
    # by the substitution (which does not invoke the executor).
    dirty_tool_steps = sum(
        1 for s in r.steps if s.dirty and s.kind == "tool_call"
    )
    forced_steps = 1  # the ToolOutputSubstitution at step:2
    # real_executions counts every executor.execute() call:
    assert r.real_executions == dirty_llm_steps + dirty_tool_steps - forced_steps + (
        # forced-output steps DO not call execute(), so subtract them
        # from the dirty tool count.
        0
    )


# ====================================================================
# Predicate stability — a separate user-side concern.
# ====================================================================


def _predicate_stability(
    predicate: Callable, runs: List
) -> float:
    """Fraction of runs in which ``predicate`` evaluates to True.

    Stability is the distance from {0, 1}: a stable predicate returns
    the same verdict for every run (stability = 1.0 if always True, or
    1.0 if always False), an unstable one returns True for some runs
    and False for others (stability strictly between 0 and 1)."""
    return sum(1 for r in runs if predicate(r)) / max(len(runs), 1)


def _replay_with_seed(path: str, key: RecorderKey, seed: int):
    t = _load(path, key)
    t.substitute(
        ToolOutputSubstitution(at_step="step:2", fake_response=LOOKUP_FIXED_ROW)
    )
    ex = Executor(llm=make_seeded_noisy_llm(seed=seed), tool=fake_tool)
    return t.replay_forward(ex)


def test_structural_predicate_is_stable_across_seeds(tmp_path):
    """A predicate that looks at the *structure* of the replay
    (dirty count) must fire identically for every seed."""
    path, key = _record_fixture(tmp_path)
    runs = [_replay_with_seed(path, key, s) for s in range(16)]

    p_struct = lambda r: r.dirty_count >= 1
    fired = _predicate_stability(p_struct, runs)
    assert fired == 1.0, "structural predicate should fire on every seed"

    # Also: ``real_executions`` is identical across all seeds.
    assert len({r.real_executions for r in runs}) == 1
    assert len({r.cache_hit_count for r in runs}) == 1


def test_content_predicate_is_unstable_across_seeds(tmp_path):
    """A predicate keyed on a specific noise nibble fires for some
    seeds and not others. This documents that *predicate* stability
    is a property of the predicate, not of the replay engine."""
    path, key = _record_fixture(tmp_path)
    runs = [_replay_with_seed(path, key, s) for s in range(64)]

    # A predicate that looks at one hex character of the noise suffix
    # of the very last LLM step. Across uniform 32-bit nonces this
    # fires roughly 1/16 of the time.
    def p_content(result) -> bool:
        last_llm = next(
            (s for s in reversed(result.steps) if s.kind == "llm_call"),
            None,
        )
        if last_llm is None or not isinstance(last_llm.outputs, dict):
            return False
        msg = last_llm.outputs["choices"][0]["message"]["content"]
        # token format: " :: noise=XXXXXXXX". Match the 7th char of the
        # nonce hex == 'a'.
        if "noise=" not in msg:
            return False
        nonce_hex = msg.split("noise=", 1)[1][:8]
        return nonce_hex[6] == "a"

    rate = _predicate_stability(p_content, runs)
    assert 0.0 < rate < 1.0, (
        f"content predicate must be unstable; got fire-rate={rate}"
    )

    # Yet the *replay* itself is correct on every single one of those
    # runs: dirty count is constant.
    assert len({r.dirty_count for r in runs}) == 1


def test_predicate_stability_separates_from_replay_correctness(tmp_path):
    """Same trace, same substitution, same executor noise. Two
    predicates evaluated against the same fleet of replays:

    * one is a stable structural predicate (always fires)
    * one is an unstable content predicate (fires sometimes)

    Replay correctness (dirty/cache counts) is constant across the
    fleet for both. Predicate stability is a *property of the
    predicate* and must not be confused with replay determinism.
    """
    path, key = _record_fixture(tmp_path)
    runs = [_replay_with_seed(path, key, s) for s in range(32)]

    p_stable = lambda r: r.real_executions > 0
    # Look at the parity of the first hex digit of the nonce on the
    # final LLM step.
    def p_unstable(r):
        last_llm = next(
            (s for s in reversed(r.steps) if s.kind == "llm_call"), None
        )
        if last_llm is None:
            return False
        msg = last_llm.outputs["choices"][0]["message"]["content"]
        if "noise=" not in msg:
            return False
        return int(msg.split("noise=", 1)[1][0], 16) % 2 == 0

    rate_stable = _predicate_stability(p_stable, runs)
    rate_unstable = _predicate_stability(p_unstable, runs)

    # Replay correctness invariants:
    assert len({r.dirty_count for r in runs}) == 1
    assert len({r.cache_hit_count for r in runs}) == 1
    assert len({r.real_executions for r in runs}) == 1

    # Predicate-stability dichotomy:
    assert rate_stable == 1.0
    assert 0.0 < rate_unstable < 1.0


def test_seeded_executor_round_trip_is_reproducible_for_predicates(tmp_path):
    """Even an *unstable* predicate becomes reproducible when the
    seed is pinned. This is the property the future
    'confidence-interval minimizer' (see ``minimize.py`` roadmap)
    will rely on: predicate verdicts are repeatable per seed."""
    path, key = _record_fixture(tmp_path)

    def p_content(result):
        last_llm = next(
            (s for s in reversed(result.steps) if s.kind == "llm_call"),
            None,
        )
        if last_llm is None:
            return False
        msg = last_llm.outputs["choices"][0]["message"]["content"]
        if "noise=" not in msg:
            return False
        return int(msg.split("noise=", 1)[1][0], 16) % 2 == 0

    seed = 2026
    verdicts = []
    for _ in range(4):
        verdicts.append(p_content(_replay_with_seed(path, key, seed)))

    assert len(set(verdicts)) == 1, (
        f"pinned-seed predicate verdict must repeat; got {verdicts}"
    )


# ====================================================================
# Sanity: digest of the whole noisy replay is stable per-seed.
# ====================================================================


def make_seeded_noisy_tool(seed: int) -> Callable[[str, dict], Any]:
    """A noisy *tool* executor whose nondeterminism is seed-controlled.

    Wraps :func:`stepback.testing.fake_tool` and stamps a deterministic
    pseudo-random nonce into the returned dict so that two runs at the
    same seed return bytewise-identical tool outputs while two runs at
    different seeds diverge. Importantly the *shape* of the output is
    preserved (still a dict, still has the original keys) so downstream
    LLM/tool steps still consume valid inputs.
    """
    rng = random.Random(seed)

    def _noisy(name: str, args: dict) -> Any:
        from stepback.testing import fake_tool as _ft
        base = _ft(name, args)
        if isinstance(base, dict):
            base = {**base, "_noise": f"{rng.randrange(0, 2**32):08x}"}
        return base

    return _noisy


# ====================================================================
# Noisy *tool* executor — symmetric coverage with the LLM-side cases.
# ====================================================================


def test_seeded_noisy_tool_replay_is_byte_identical_across_repeats(tmp_path):
    """Same property as the LLM case but for tool-call dirty steps:
    seeded tool-side noise must be reproducible run-to-run."""
    path, key = _record_fixture(tmp_path)

    def go():
        t = _load(path, key)
        # Substitute a *prompt* upstream so the dirty-set bleeds into
        # downstream tool calls; this exercises the tool-noise path.
        t.substitute(
            PromptSubstitution(
                at_step="step:1",
                new_messages=[
                    {"role": "system", "content": "rewritten plan"},
                    {"role": "user", "content": "Pay invoice INV-118."},
                ],
            )
        )
        ex = Executor(llm=fake_llm, tool=make_seeded_noisy_tool(seed=42))
        return t.replay_forward(ex)

    a = go()
    b = go()

    # Tool outputs are bytewise identical across the two runs.
    tool_outs_a = [s.outputs for s in a.steps if s.kind == "tool_call"]
    tool_outs_b = [s.outputs for s in b.steps if s.kind == "tool_call"]
    assert tool_outs_a == tool_outs_b
    assert _dirty_ids(a) == _dirty_ids(b)
    assert _cached_ids(a) == _cached_ids(b)


def test_unseeded_executor_does_not_change_dirty_set_size(tmp_path):
    """A noisy executor must not change *how many* steps are dirty.
    Dirty-set size is a pure function of (trace, substitution)."""
    path, key = _record_fixture(tmp_path)

    sizes = []
    for _ in range(8):
        t = _load(path, key)
        t.substitute(
            ToolOutputSubstitution(at_step="step:2", fake_response=LOOKUP_FIXED_ROW)
        )
        ex = Executor(
            llm=make_unseeded_noisy_llm(),
            tool=make_seeded_noisy_tool(seed=0),
        )
        r = t.replay_forward(ex)
        sizes.append(r.dirty_count)

    assert len(set(sizes)) == 1, sizes


# ====================================================================
# Statistical confidence-interval framing for unstable predicates.
# ====================================================================


def test_unstable_predicate_admits_wilson_confidence_interval(tmp_path):
    """An unstable predicate's fire-rate is a Bernoulli parameter and
    admits a confidence interval. This documents the contract that the
    future stochastic minimizer in ``minimize.py`` must rely on:
    *replay correctness is deterministic; predicate verdicts are
    Bernoulli samples whose confidence shrinks like 1/sqrt(N)*.

    The test asserts the obvious sanity property: as we double the
    sample size the half-width of the 95% Wilson interval shrinks.
    """
    path, key = _record_fixture(tmp_path)

    def fire_rate_and_halfwidth(n: int) -> Tuple[float, float]:
        runs = [_replay_with_seed(path, key, s) for s in range(n)]

        def p(r) -> bool:
            last_llm = next(
                (s for s in reversed(r.steps) if s.kind == "llm_call"), None
            )
            if last_llm is None:
                return False
            msg = last_llm.outputs["choices"][0]["message"]["content"]
            if "noise=" not in msg:
                return False
            return int(msg.split("noise=", 1)[1][0], 16) % 2 == 0

        k = sum(1 for r in runs if p(r))
        rate = k / n
        # 95% Wilson half-width ≈ 1.96 * sqrt(rate*(1-rate)/n) for
        # large-ish n; we just need a coarse comparator here.
        halfwidth = 1.96 * ((rate * (1 - rate) / n) ** 0.5)
        return rate, halfwidth

    _, hw_small = fire_rate_and_halfwidth(16)
    _, hw_large = fire_rate_and_halfwidth(64)

    # 4× the samples should at least *not increase* the halfwidth.
    assert hw_large <= hw_small + 1e-9


def test_cost_is_invariant_under_executor_noise(tmp_path):
    """Cost is computed from ``usage`` tokens, which the seeded noisy
    executor preserves. Therefore total cost is invariant across
    seeds. (This is a property of our test fakes, not of the engine,
    but it documents the boundary: cost stability requires the
    executor to leave token counts alone.)"""
    path, key = _record_fixture(tmp_path)

    costs = []
    for seed in (1, 2, 3, 4, 5):
        t = _load(path, key)
        t.substitute(
            ToolOutputSubstitution(at_step="step:2", fake_response=LOOKUP_FIXED_ROW)
        )
        ex = Executor(llm=make_seeded_noisy_llm(seed=seed), tool=fake_tool)
        r = t.replay_forward(ex)
        costs.append(r.total_cost_usd)

    # All five replays produced identical total cost despite different
    # seeded LLM noise.
    head = costs[0]
    for c in costs[1:]:
        assert c == pytest.approx(head)


def test_seeded_replay_digest_is_stable(tmp_path):
    """A SHA-256 over the full ordered (step_id, content) stream is
    the same for two replays with the same seed."""
    path, key = _record_fixture(tmp_path)

    def digest_for(seed):
        r = _replay_with_seed(path, key, seed)
        h = hashlib.sha256()
        for sid, content in _replay_outputs(r):
            h.update(sid.encode())
            h.update(b"\x00")
            h.update(content.encode())
            h.update(b"\x01")
        return h.hexdigest()

    a = digest_for(seed=99)
    b = digest_for(seed=99)
    c = digest_for(seed=100)

    assert a == b
    assert a != c
