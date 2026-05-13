"""Tests for the typed predicate DSL extensions added in step 80.

Covers:
1. TypedPredicate class
2. threshold() factory
3. regex_match() factory
4. regex_search() factory
5. policy_check() factory
6. callback() factory
7. Composition of typed predicates with combinators (all_of, any_of, not_, xor_)
8. DSL string extensions: re_match(), re_search(), policy_blocked()
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List

import pytest

from stepback.predicates import (
    TypedPredicate,
    all_of,
    any_of,
    callback,
    compile_predicate,
    not_,
    policy_check,
    regex_match,
    regex_search,
    threshold,
    xor_,
    PredicateSyntaxError,
)


# --------------------------------------------------------- test shim types


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


# ======================================================= 1. TypedPredicate


class TestTypedPredicate:
    def test_callable(self):
        p = TypedPredicate(lambda x: x > 0, name="positive")
        assert p(1) is True
        assert p(-1) is False

    def test_returns_strict_bool(self):
        p = TypedPredicate(lambda x: x, name="truthy")
        out = p(42)
        assert out is True and isinstance(out, bool)
        out = p(0)
        assert out is False and isinstance(out, bool)

    def test_name_and_description(self):
        p = TypedPredicate(lambda x: True, name="always", description="fires always")
        assert p.name == "always"
        assert p.description == "fires always"

    def test_source_optional(self):
        p = TypedPredicate(lambda x: True, name="t", source="x > 0")
        assert p.source == "x > 0"

    def test_source_defaults_to_none(self):
        p = TypedPredicate(lambda x: True, name="t")
        assert p.source is None

    def test_repr_includes_name(self):
        p = TypedPredicate(lambda x: True, name="my_pred")
        assert "my_pred" in repr(p)

    def test_non_callable_raises(self):
        with pytest.raises(TypeError):
            TypedPredicate("not_callable", name="bad")


# ========================================================= 2. threshold()


class TestThreshold:
    def test_greater_than_attribute(self):
        p = threshold("total_cost_usd", ">", 0.10)
        assert p(_Result(total_cost_usd=0.20)) is True
        assert p(_Result(total_cost_usd=0.05)) is False

    def test_equal(self):
        p = threshold("dirty_count", "==", 0)
        assert p(_Result(dirty_count=0)) is True
        assert p(_Result(dirty_count=1)) is False

    def test_less_than_equal(self):
        p = threshold("real_executions", "<=", 5)
        assert p(_Result(real_executions=5)) is True
        assert p(_Result(real_executions=6)) is False

    def test_not_equal(self):
        p = threshold("dirty_count", "!=", 0)
        assert p(_Result(dirty_count=1)) is True
        assert p(_Result(dirty_count=0)) is False

    def test_missing_attribute_returns_false(self):
        p = threshold("nonexistent_metric", ">", 0)
        assert p(_Result()) is False

    def test_none_attribute_returns_false(self):
        p = threshold("name", ">", 0)
        assert p(_Step(name=None)) is False

    def test_invalid_op_raises(self):
        with pytest.raises(ValueError, match="not recognised"):
            threshold("total_cost_usd", "~=", 0.1)

    def test_name_auto_generated(self):
        p = threshold("total_cost_usd", ">", 0.10)
        assert "total_cost_usd" in p.name
        assert ">" in p.name

    def test_custom_name(self):
        p = threshold("total_cost_usd", ">", 0.10, name="expensive")
        assert p.name == "expensive"

    def test_source_attribute_set(self):
        p = threshold("total_cost_usd", ">", 0.10)
        assert p.source is not None
        assert "threshold" in p.source

    def test_dict_lookup_fallback(self):
        p = threshold("cost", ">", 5)
        assert p({"cost": 10}) is True
        assert p({"cost": 3}) is False

    def test_type_error_in_comparison_returns_false(self):
        p = threshold("kind", ">", 0)
        assert p(_Step(kind="llm_call")) is False


# ====================================================== 3. regex_match()


class TestRegexMatch:
    def test_matches_at_start(self):
        p = regex_match(r"GB\d{2}", field="outputs")
        assert p(_Step(outputs="GB99-XXX")) is True
        assert p(_Step(outputs="XXX-GB99")) is False  # not at start

    def test_no_field_uses_str_of_value(self):
        p = regex_match(r"\d+")
        assert p("123abc") is True
        assert p("abc123") is False

    def test_none_field_value_treated_as_empty(self):
        p = regex_match(r"\w+", field="outputs")
        assert p(_Step(outputs=None)) is False

    def test_flags_respected(self):
        p = regex_match(r"hello", field="outputs", flags=re.IGNORECASE)
        assert p(_Step(outputs="HELLO world")) is True

    def test_custom_name(self):
        p = regex_match(r"GB\d{2}", name="iban_check")
        assert p.name == "iban_check"

    def test_auto_name_contains_pattern(self):
        p = regex_match(r"GB\d{2}")
        assert "GB" in p.name and "d{2}" in p.name

    def test_source_set(self):
        p = regex_match(r"GB\d{2}", field="outputs")
        assert p.source is not None

    def test_dict_field_resolution(self):
        p = regex_match(r"hello", field="message")
        assert p({"message": "hello world"}) is True

    def test_complex_dict_outputs(self):
        step = _Step(outputs={"iban": "GB99-XYZ"})
        p = regex_match(r".*GB99.*", field="outputs")
        assert p(step) is True


# ====================================================== 4. regex_search()


class TestRegexSearch:
    def test_finds_anywhere(self):
        p = regex_search(r"GB\d{2}", field="outputs")
        assert p(_Step(outputs="PREFIX-GB99-SUFFIX")) is True
        assert p(_Step(outputs="no match here")) is False

    def test_no_field_stringifies_value(self):
        p = regex_search(r"cat")
        assert p("my cat sat") is True
        assert p("my dog sat") is False

    def test_none_field_returns_false(self):
        p = regex_search(r"\w+", field="outputs")
        assert p(_Step(outputs=None)) is False

    def test_flags(self):
        p = regex_search(r"ERROR", field="outputs", flags=re.IGNORECASE)
        assert p(_Step(outputs="there was an error here")) is True

    def test_auto_name_contains_pattern(self):
        p = regex_search(r"ERROR")
        assert "ERROR" in p.name


# ====================================================== 5. policy_check()


class TestPolicyCheck:
    def test_fires_on_error_class(self):
        p = policy_check()
        s = _Step(outputs={"error_class": "PolicyDenied", "message": "blocked"})
        assert p(s) is True

    def test_fires_on_error_type(self):
        p = policy_check()
        s = _Step(outputs={"__error__": {"type": "PolicyViolation", "message": "no"}})
        assert p(s) is True

    def test_fires_on_blocked_true(self):
        p = policy_check()
        s = _Step(outputs={"blocked": True, "reason": "rate limit"})
        assert p(s) is True

    def test_does_not_fire_on_normal_outputs(self):
        p = policy_check()
        s = _Step(outputs={"result": "success"})
        assert p(s) is False

    def test_does_not_fire_on_non_dict_outputs(self):
        p = policy_check()
        assert p(_Step(outputs="some string")) is False
        assert p(_Step(outputs=None)) is False

    def test_case_insensitive_policy_match(self):
        p = policy_check()
        s = _Step(outputs={"error_class": "POLICYDENIED"})
        assert p(s) is True

    def test_default_name(self):
        p = policy_check()
        assert p.name == "policy_blocked"

    def test_custom_name(self):
        p = policy_check(name="content_policy")
        assert p.name == "content_policy"

    def test_blocked_false_does_not_fire(self):
        p = policy_check()
        s = _Step(outputs={"blocked": False, "reason": "ok"})
        assert p(s) is False


# ========================================================= 6. callback()


class TestCallback:
    def test_wraps_lambda(self):
        p = callback(lambda s: s.kind == "tool_call", name="is_tool")
        assert p(_Step(kind="tool_call")) is True
        assert p(_Step(kind="llm_call")) is False

    def test_non_callable_raises(self):
        with pytest.raises(TypeError):
            callback("not_a_callable", name="bad")

    def test_name_auto_from_function(self):
        def my_check(s):
            return True

        p = callback(my_check)
        assert p.name == "my_check"

    def test_custom_name_and_description(self):
        p = callback(lambda x: True, name="always_true", description="fires always")
        assert p.name == "always_true"
        assert p.description == "fires always"

    def test_returns_strict_bool(self):
        p = callback(lambda x: 42, name="truthy")
        out = p(None)
        assert out is True and isinstance(out, bool)


# =============================== 7. Composition with combinators


class TestTypedPredicateComposition:
    def test_all_of_typed(self):
        p1 = threshold("total_cost_usd", ">", 0.0)
        p2 = threshold("dirty_count", "==", 0)
        combined = all_of(p1, p2)
        assert combined(_Result(total_cost_usd=0.5, dirty_count=0)) is True
        assert combined(_Result(total_cost_usd=0.0, dirty_count=0)) is False
        assert combined(_Result(total_cost_usd=0.5, dirty_count=1)) is False

    def test_any_of_typed(self):
        p1 = policy_check()
        p2 = threshold("cost_usd", ">", 1.0)
        combined = any_of(p1, p2)
        assert combined(_Step(outputs={"blocked": True})) is True
        assert combined(_Step(cost_usd=2.0)) is True
        assert combined(_Step()) is False

    def test_not_typed(self):
        p = not_(policy_check())
        assert p(_Step(outputs={"result": "ok"})) is True
        assert p(_Step(outputs={"blocked": True})) is False

    def test_xor_typed(self):
        cheap = threshold("cost_usd", "<=", 1.0)
        policy = policy_check()
        p = xor_(cheap, policy)
        assert p(_Step(cost_usd=0.5)) is True   # cheap but not blocked
        assert p(_Step(outputs={"blocked": True}, cost_usd=2.0)) is True  # blocked but expensive
        # Both fire: cost_usd=0.5 is cheap AND outputs blocked
        assert p(_Step(cost_usd=0.5, outputs={"blocked": True})) is False

    def test_regex_and_policy_compound(self):
        has_iban = regex_search(r"GB\d{2}", field="outputs")
        is_blocked = policy_check()
        flagged = all_of(has_iban, not_(is_blocked))
        good = _Step(outputs={"iban": "GB99-XXXX", "result": "ok"})
        blocked = _Step(outputs={"iban": "GB99-XXXX", "blocked": True})
        assert flagged(good) is True
        assert flagged(blocked) is False


# ========================== 8. DSL string extensions: re_match / re_search


class TestDSLRegexBuiltins:
    def test_re_match_in_dsl(self):
        p = compile_predicate("re_match('GB\\\\d{2}', str(outputs))")
        assert p(_Step(outputs="GB99-XXX")) is True
        assert p(_Step(outputs="XXX-GB99")) is False

    def test_re_search_in_dsl(self):
        p = compile_predicate("re_search('GB\\\\d{2}', str(outputs))")
        assert p(_Step(outputs="PREFIX-GB99-SUFFIX")) is True
        assert p(_Step(outputs="no match")) is False

    def test_re_match_inside_any_step(self):
        r = _Result(steps=[_Step(outputs="GB99-XXX"), _Step(outputs="no match")])
        p = compile_predicate("any_step(re_match('GB', str(outputs)))")
        assert p(r) is True

    def test_re_search_no_match_returns_falsy(self):
        p = compile_predicate("re_search('NOTFOUND', str(outputs))")
        assert p(_Step(outputs="hello world")) is False

    def test_re_match_wrong_arity_rejected(self):
        with pytest.raises(PredicateSyntaxError, match="two positional"):
            compile_predicate("re_match('pat')")

    def test_re_search_wrong_arity_rejected(self):
        with pytest.raises(PredicateSyntaxError, match="two positional"):
            compile_predicate("re_search('pat', 'text', 'extra')")


# ======================== 9. DSL policy_blocked()


class TestDSLPolicyBlocked:
    def test_fires_on_policy_denied(self):
        p = compile_predicate("policy_blocked()")
        s = _Step(outputs={"error_class": "PolicyDenied"})
        assert p(s) is True

    def test_does_not_fire_on_normal(self):
        p = compile_predicate("policy_blocked()")
        s = _Step(outputs={"result": "ok"})
        assert p(s) is False

    def test_inside_any_step_quantifier(self):
        r = _Result(steps=[
            _Step(outputs={"result": "ok"}),
            _Step(outputs={"blocked": True}),
        ])
        p = compile_predicate("any_step(policy_blocked())")
        assert p(r) is True

    def test_all_step_none_blocked(self):
        r = _Result(steps=[
            _Step(outputs={"result": "ok"}),
            _Step(outputs={"result": "also ok"}),
        ])
        p = compile_predicate("all_step(not policy_blocked())")
        assert p(r) is True

    def test_policy_blocked_wrong_arity_rejected(self):
        with pytest.raises(PredicateSyntaxError, match="no arguments"):
            compile_predicate("policy_blocked('extra')")

    def test_combined_cost_and_policy(self):
        r = _Result(
            total_cost_usd=0.5,
            steps=[_Step(outputs={"blocked": True})],
        )
        p = compile_predicate(
            "total_cost_usd > 0 and any_step(policy_blocked())"
        )
        assert p(r) is True
