"""Tests for stepback.predicates combinators."""
from __future__ import annotations

from stepback.predicates import all_of, any_of, not_


def test_all_of_short_circuits_false():
    calls = []

    def p1(_):
        calls.append("p1")
        return True

    def p2(_):
        calls.append("p2")
        return False

    def p3(_):
        calls.append("p3")
        return True

    assert all_of(p1, p2, p3)(None) is False
    assert calls == ["p1", "p2"]


def test_any_of_short_circuits_true():
    calls = []

    def p1(_):
        calls.append("p1")
        return False

    def p2(_):
        calls.append("p2")
        return True

    def p3(_):
        calls.append("p3")
        return True

    assert any_of(p1, p2, p3)(None) is True
    assert calls == ["p1", "p2"]


def test_not_inverts():
    assert not_(lambda x: True)(None) is False
    assert not_(lambda x: False)(None) is True


def test_combinators_compose():
    p = all_of(
        lambda r: r > 0,
        any_of(
            lambda r: r > 100,
            not_(lambda r: r % 2 == 0),
        ),
    )
    assert p(101) is True   # > 0 and > 100
    assert p(3) is True     # > 0 and odd
    assert p(4) is False    # > 0 but even and <= 100
    assert p(-1) is False   # not > 0


def test_empty_all_of_is_true():
    assert all_of()(object()) is True


def test_empty_any_of_is_false():
    assert any_of()(object()) is False


# ---------------- numeric-threshold metrics ----------------


def test_all_of_short_circuit_call_count_threshold():
    """all_of must stop calling predicates the moment one returns False."""
    calls = []

    def make(name, ret):
        def p(_):
            calls.append(name)
            return ret
        return p

    preds = [make(f"p{i}", True) for i in range(10)]
    preds[3] = make("p3", False)  # short-circuit at index 3
    preds.extend(make(f"q{i}", True) for i in range(10))

    assert all_of(*preds)(None) is False
    # Exactly 4 predicates evaluated (indices 0..3 inclusive)
    assert len(calls) == 4
    assert calls == ["p0", "p1", "p2", "p3"]
    # Saved 16 of 20 evaluations: ratio bound
    assert len(calls) / 20 <= 0.25


def test_any_of_short_circuit_call_count_threshold():
    """any_of must stop calling predicates the moment one returns True."""
    calls = []

    def make(name, ret):
        def p(_):
            calls.append(name)
            return ret
        return p

    preds = [make(f"p{i}", False) for i in range(7)]
    preds[2] = make("p2", True)  # short-circuit at index 2
    preds.extend(make(f"q{i}", False) for i in range(7))

    assert any_of(*preds)(None) is True
    assert len(calls) == 3
    assert calls == ["p0", "p1", "p2"]
    assert len(calls) / 14 <= 0.25


def test_compose_truth_table_count():
    """Across 200 integer inputs, predicate must accept exactly the
    integers that are (>0 and (>100 or odd)). Compare to ground truth."""
    p = all_of(
        lambda r: r > 0,
        any_of(
            lambda r: r > 100,
            not_(lambda r: r % 2 == 0),
        ),
    )
    inputs = list(range(-50, 150))
    expected = [r for r in inputs if r > 0 and (r > 100 or r % 2 == 1)]
    actual = [r for r in inputs if p(r)]
    assert actual == expected
    # Numeric threshold: exact accepted count
    assert len(actual) == 99
    assert len(expected) == 99
    # No false positives, no false negatives
    assert sum(1 for r in inputs if p(r) and r not in expected) == 0
    assert sum(1 for r in inputs if (r in expected) and not p(r)) == 0


def test_not_double_negation_identity_count():
    base = lambda r: r % 3 == 0
    double = not_(not_(base))
    inputs = list(range(100))
    agree = sum(1 for r in inputs if base(r) == double(r))
    assert agree == 100
    # Exactly 34 multiples of 3 in [0,100)
    assert sum(1 for r in inputs if double(r)) == 34


def test_all_of_zero_evaluations_when_first_false():
    """Strict short-circuit: only the FIRST predicate runs when it returns False."""
    calls = []

    def make(name, ret):
        def p(_):
            calls.append(name)
            return ret
        return p

    preds = [make("p0", False)] + [make(f"p{i}", True) for i in range(1, 50)]
    assert all_of(*preds)(None) is False
    # Exactly 1 of 50 predicates evaluated
    assert len(calls) == 1
    assert calls == ["p0"]
    # Saved 49/50 evaluations: ratio bound
    assert len(calls) / 50 <= 0.02


def test_any_of_zero_evaluations_when_first_true():
    """Strict short-circuit: only the FIRST predicate runs when it returns True."""
    calls = []

    def make(name, ret):
        def p(_):
            calls.append(name)
            return ret
        return p

    preds = [make("p0", True)] + [make(f"p{i}", False) for i in range(1, 30)]
    assert any_of(*preds)(None) is True
    assert len(calls) == 1
    assert len(calls) / 30 <= 0.04


def test_not_idempotent_under_quadruple_negation():
    """not_(not_(not_(not_(p)))) must equal p on every input."""
    base = lambda r: (r * 7 + 3) % 5 == 0
    quad = not_(not_(not_(not_(base))))
    inputs = list(range(500))
    disagreements = sum(1 for r in inputs if base(r) != quad(r))
    assert disagreements == 0
    # Exactly 100 of 500 satisfy the base predicate (every 5th)
    matches = sum(1 for r in inputs if quad(r))
    assert matches == 100
    assert matches / 500 == 0.20
