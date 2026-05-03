"""Tests for the predicate DSL added in :mod:`stepback.predicates`.

Covers five sections per L3 plan:
1. Smoke / happy paths.
2. Quantifiers (``any_step`` / ``all_step``).
3. Sandboxing denylist (every documented escape).
4. Error UX (``PredicateSyntaxError.pos`` + caret).
5. End-to-end against a real recorded fixture trace.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List

import pytest

from stepback import RecorderKey, record, replay
from stepback.predicates import (
    PredicateRuntimeError,
    PredicateSyntaxError,
    compile_predicate,
    parse_predicate,
)
from tests.fixtures.agent import fake_llm, fake_tool, run_recorded_agent  # noqa: F401


# ----------------------------------------------------------- shim types


@dataclass
class _Step:
    step_id: str = "s1"
    kind: str = "llm_call"
    name: str = None
    outputs: Any = None
    inputs: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    dirty: bool = False
    cache_hit: bool = True
    error_class: str = None
    parent_step_id: str = None

    @property
    def cost(self) -> float:
        return self.cost_usd


@dataclass
class _Result:
    steps: List[_Step] = field(default_factory=list)
    total_cost_usd: float = 0.0
    dirty_count: int = 0
    cache_hit_count: int = 0
    real_executions: int = 0


# ============================================================== 1. Smoke


class TestSmoke:
    def test_int_arith_precedence(self):
        assert compile_predicate("1 + 2 * 3 == 7")(None) is True
        assert compile_predicate("(1 + 2) * 3 == 9")(None) is True

    def test_float_div_mod(self):
        assert compile_predicate("10 / 4 == 2.5")(None) is True
        assert compile_predicate("10 % 3 == 1")(None) is True

    def test_string_in(self):
        r = _Result(steps=[_Step(outputs={"iban": "GB99-XXX"})])
        assert compile_predicate("'GB99' in str(steps[0].outputs)")(r) is True
        assert compile_predicate("'US' not in str(steps[0].outputs)")(r) is True

    def test_chained_comparison(self):
        assert compile_predicate("1 < 2 < 3")(None) is True
        assert compile_predicate("3 < 2 < 1")(None) is False

    def test_bool_short_circuit(self):
        assert compile_predicate("True and False")(None) is False
        assert compile_predicate("False or True")(None) is True
        assert compile_predicate("not False")(None) is True

    def test_attribute_and_subscript(self):
        s = _Step(outputs={"k": "v"})
        assert compile_predicate("step.outputs['k'] == 'v'")(s) is True
        assert compile_predicate("outputs['k'] == 'v'")(s) is True  # shorthand

    def test_len_builtin(self):
        r = _Result(steps=[_Step(), _Step(), _Step()])
        assert compile_predicate("len(steps) == 3")(r) is True

    def test_predicate_returns_strict_bool(self):
        # Falsy non-bool is coerced to False, not returned as 0.
        out = compile_predicate("0")(None)
        assert out is False and isinstance(out, bool)
        out = compile_predicate("'x'")(None)
        assert out is True and isinstance(out, bool)

    def test_source_attribute_round_trips(self):
        src = "total_cost_usd > 0.05"
        p = compile_predicate(src)
        assert p.source == src
        assert hasattr(p, "parsed")

    def test_extra_names_injection(self):
        thresh = 0.42
        p = compile_predicate("total_cost_usd > T", extra_names={"T": thresh})
        assert p(_Result(total_cost_usd=0.5)) is True
        assert p(_Result(total_cost_usd=0.1)) is False

    def test_unknown_name_resolves_to_none(self):
        # No NameError — names just go to None so cross-context predicates
        # don't blow up.
        assert compile_predicate("nosuch == None")(None) is True

    def test_lru_cache_returns_consistent_predicates(self):
        p1 = compile_predicate("1 == 1")
        p2 = compile_predicate("1 == 1")
        assert p1.parsed is p2.parsed  # underlying tree cached


# ======================================================== 2. Quantifiers


class TestQuantifiers:
    def test_any_step_fires_when_one_matches(self):
        r = _Result(
            steps=[_Step(kind="llm_call"), _Step(kind="tool_call")]
        )
        assert compile_predicate("any_step(kind == 'tool_call')")(r) is True

    def test_any_step_false_when_none_match(self):
        r = _Result(steps=[_Step(kind="llm_call"), _Step(kind="llm_call")])
        assert compile_predicate("any_step(kind == 'tool_call')")(r) is False

    def test_all_step_true_when_all_match(self):
        r = _Result(steps=[_Step(cost_usd=0.1), _Step(cost_usd=0.2)])
        assert compile_predicate("all_step(cost_usd >= 0)")(r) is True

    def test_all_step_false_on_one_violator(self):
        r = _Result(steps=[_Step(cost_usd=0.1), _Step(cost_usd=-0.1)])
        assert compile_predicate("all_step(cost_usd >= 0)")(r) is False

    def test_quantifier_step_binding_does_not_leak(self):
        r = _Result(steps=[_Step(kind="llm_call")])
        # After the any_step body finishes, `kind` outside it should
        # resolve via the top-level value frame (here: _Result has no
        # `.kind`), so it's None.
        p = compile_predicate("any_step(kind == 'llm_call') and kind == None")
        assert p(r) is True

    def test_nested_boolean_with_quantifier(self):
        r = _Result(
            total_cost_usd=0.5,
            steps=[_Step(kind="tool_call", outputs={"iban": "GB99-XX"})],
        )
        src = (
            "total_cost_usd > 0.10 and "
            "any_step(kind == 'tool_call' and 'GB99' in str(outputs))"
        )
        assert compile_predicate(src)(r) is True


# ======================================================= 3. Sandboxing


@pytest.mark.parametrize(
    "src,reason_substring",
    [
        ("step.__class__", "starts with '_'"),
        ("().__class__.__mro__", "starts with '_'"),
        ("__import__('os')", "function '__import__' not allowed"),
        ("lambda x: x", "Lambda"),
        ("9**9**9", "Pow"),
        ("x.get('y')", "method calls"),
        ("any(s for s in steps)", "function 'any' not allowed"),
        ("[s for s in steps]", "ListComp"),
        ("{1, 2, 3}", "Set"),
        ('{"a": 1}', "Dict"),
        ('f"{x}"', "JoinedStr"),
        ("steps[1:2]", "slice indexing"),
        ("min(1, 2)", "function 'min' not allowed"),
        ("any_step(kind == 'x', 'extra')", "exactly one positional argument"),
    ],
)
def test_denylist(src, reason_substring):
    with pytest.raises(PredicateSyntaxError) as exc:
        compile_predicate(src)
    assert reason_substring in str(exc.value)


def test_literal_length_cap_enforced():
    too_big = "[" + ",".join(["0"] * 1500) + "]"
    with pytest.raises(PredicateSyntaxError) as exc:
        compile_predicate(too_big + " == []")
    assert "exceeds limit" in str(exc.value)


def test_keyword_args_rejected():
    with pytest.raises(PredicateSyntaxError) as exc:
        compile_predicate("len(x=1)")
    assert "keyword arguments" in str(exc.value)


def test_runtime_error_wraps_eval_failure():
    # Comparing two None ordering returns False (no raise), so force a
    # builtin failure instead.
    with pytest.raises(PredicateRuntimeError):
        compile_predicate("len(123)")(None)


# ============================================================ 4. Error UX


def test_syntax_error_carries_pos_and_src():
    src = "1 +"
    with pytest.raises(PredicateSyntaxError) as exc:
        compile_predicate(src)
    e = exc.value
    assert e.src == src
    assert isinstance(e.pos, tuple) and len(e.pos) == 2
    assert "^" in str(e)


def test_visitor_error_pos_points_at_offending_node():
    src = "1 + lambda x: x"
    with pytest.raises(PredicateSyntaxError) as exc:
        compile_predicate(src)
    e = exc.value
    line, col = e.pos
    assert line == 1
    # The lambda starts at col 4 in "1 + lambda x: x".
    assert col == 4


def test_parse_predicate_returns_ast_expression():
    tree = parse_predicate("1 == 1")
    import ast as _ast
    assert isinstance(tree, _ast.Expression)


# ============================================================ 5. End-to-end


def _record_payments_fixture(tmp_path) -> tuple:
    key = RecorderKey.fresh()
    path = str(tmp_path / "trace.sb")
    with record(path, key=key) as rec:
        run_recorded_agent(rec)
    return path, key


def test_e2e_compile_predicate_used_with_bisect(tmp_path):
    """Compile a DSL string and feed it to ``Trace.bisect`` against the
    real recorded payments-agent fixture (the same one
    ``test_e2e_replay`` uses). No mocks; bisect must locate a real
    tool_call step and stay within the log2 probe budget."""
    path, key = _record_payments_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)

    # Identical predicate to test_e2e_replay.py's hand-rolled lambda,
    # but expressed in the DSL.
    p = compile_predicate(
        "kind == 'tool_call' and 'GB99' in str(outputs)"
    )
    found = t.bisect(good="step:1", bad="step:12", predicate=p)
    assert found is not None
    assert found.kind == "tool_call"
    assert "GB99" in str(found.outputs)
    # Probe budget: ⌈log2(12)⌉ + 1 = 5
    assert t.last_bisect_probes <= math.ceil(math.log2(12)) + 1


def test_e2e_compile_predicate_used_with_replayresult(tmp_path):
    """Same DSL surface, this time invoked over a ``ReplayResult`` via
    ``ReplayResult.any_step`` / ``find``."""
    path, key = _record_payments_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    from stepback.replay import Executor
    result = t.run_replay(t.pending_subs, Executor())

    # any_step shorthand
    p_any = compile_predicate(
        "any_step(kind == 'tool_call' and 'GB99' in str(outputs))"
    )
    assert p_any(result) is True

    # cost gate (fixture's recorded total cost is positive)
    assert compile_predicate("total_cost_usd >= 0")(result) is True
    assert compile_predicate("dirty_count == 0")(result) is True

    # The predicate's source is preserved verbatim — the artefact-as-
    # evidence story.
    assert "any_step" in p_any.source


def test_e2e_predicate_source_round_trips_for_reporting(tmp_path):
    """A predicate used to bisect can be embedded into a report
    verbatim via ``.source`` — usable by ``stepback/report.py``
    consumers."""
    src = "kind == 'tool_call' and 'GB99' in str(outputs)"
    p = compile_predicate(src)
    assert p.source == src
    # And the same string compiles to an equivalent predicate.
    p2 = compile_predicate(p.source)
    s = _Step(kind="tool_call", outputs={"iban": "GB99-XYZ"})
    assert p(s) == p2(s) is True
