"""Tests for the in-tree JSON Patch / Pointer subset (RFC 6902/6901)."""
from __future__ import annotations

import pytest

from stepback.jsonpatch import (
    PatchError,
    PatchInvalidOp,
    PatchPathNotFound,
    PatchTestFailed,
    apply_patch,
)


# ---------------- add ----------------


def test_add_object_key():
    assert apply_patch({"a": 1}, [{"op": "add", "path": "/b", "value": 2}]) == {"a": 1, "b": 2}


def test_add_replaces_existing_key():
    # RFC 6902 §4.1: add on existing key behaves like replace
    assert apply_patch({"a": 1}, [{"op": "add", "path": "/a", "value": 9}]) == {"a": 9}


def test_add_array_insert():
    assert apply_patch([1, 2, 3], [{"op": "add", "path": "/1", "value": 9}]) == [1, 9, 2, 3]


def test_add_array_dash_appends():
    assert apply_patch([1, 2], [{"op": "add", "path": "/-", "value": 3}]) == [1, 2, 3]


def test_add_index_one_past_end_ok():
    assert apply_patch([1, 2], [{"op": "add", "path": "/2", "value": 3}]) == [1, 2, 3]


def test_add_index_too_large_fails():
    with pytest.raises(PatchPathNotFound):
        apply_patch([1, 2], [{"op": "add", "path": "/5", "value": 3}])


def test_add_into_missing_intermediate_fails():
    with pytest.raises(PatchPathNotFound):
        apply_patch({"a": 1}, [{"op": "add", "path": "/missing/x", "value": 1}])


def test_add_root_replaces_doc():
    assert apply_patch({"a": 1}, [{"op": "add", "path": "", "value": [1, 2]}]) == [1, 2]


# ---------------- replace ----------------


def test_replace_missing_key_fails():
    with pytest.raises(PatchPathNotFound):
        apply_patch({"a": 1}, [{"op": "replace", "path": "/b", "value": 2}])


def test_replace_array_dash_fails():
    with pytest.raises(PatchPathNotFound):
        apply_patch([1, 2], [{"op": "replace", "path": "/-", "value": 9}])


def test_replace_root():
    assert apply_patch({"a": 1}, [{"op": "replace", "path": "", "value": 42}]) == 42


# ---------------- remove ----------------


def test_remove_object_key():
    assert apply_patch({"a": 1, "b": 2}, [{"op": "remove", "path": "/a"}]) == {"b": 2}


def test_remove_array_element():
    assert apply_patch([1, 2, 3], [{"op": "remove", "path": "/1"}]) == [1, 3]


def test_remove_root_rejected():
    with pytest.raises(PatchInvalidOp):
        apply_patch({"a": 1}, [{"op": "remove", "path": ""}])


def test_remove_missing_key_fails():
    with pytest.raises(PatchPathNotFound):
        apply_patch({"a": 1}, [{"op": "remove", "path": "/nope"}])


# ---------------- test ----------------


def test_test_passes():
    assert apply_patch({"a": 1}, [{"op": "test", "path": "/a", "value": 1}]) == {"a": 1}


def test_test_fails():
    with pytest.raises(PatchTestFailed):
        apply_patch({"a": 1}, [{"op": "test", "path": "/a", "value": 2}])


def test_test_int_vs_float_strict():
    with pytest.raises(PatchTestFailed):
        apply_patch({"a": 1}, [{"op": "test", "path": "/a", "value": 1.0}])


def test_test_failure_aborts_remaining_ops():
    with pytest.raises(PatchTestFailed):
        apply_patch(
            {"a": 1},
            [
                {"op": "test", "path": "/a", "value": 99},
                {"op": "add", "path": "/b", "value": 2},
            ],
        )


# ---------------- copy / move ----------------


def test_copy():
    out = apply_patch({"a": {"x": 1}}, [{"op": "copy", "from": "/a", "path": "/b"}])
    assert out == {"a": {"x": 1}, "b": {"x": 1}}
    # Deep copy: mutating one doesn't change the other.
    out["b"]["x"] = 9
    assert out == {"a": {"x": 1}, "b": {"x": 9}}


def test_move():
    assert apply_patch(
        {"a": 1, "b": 2}, [{"op": "move", "from": "/a", "path": "/c"}]
    ) == {"b": 2, "c": 1}


def test_move_to_self_is_noop():
    assert apply_patch(
        {"a": 1}, [{"op": "move", "from": "/a", "path": "/a"}]
    ) == {"a": 1}


# ---------------- pointer escapes / negatives ----------------


def test_tilde_escapes():
    # ~1 -> /, ~0 -> ~
    assert apply_patch(
        {"a/b": 1}, [{"op": "replace", "path": "/a~1b", "value": 9}]
    ) == {"a/b": 9}
    assert apply_patch(
        {"a~b": 1}, [{"op": "replace", "path": "/a~0b", "value": 9}]
    ) == {"a~b": 9}


def test_tilde_decoded_in_correct_order():
    # ~01 must decode to ~1 (NOT to /)
    out = apply_patch({"~1": 1}, [{"op": "replace", "path": "/~01", "value": 9}])
    assert out == {"~1": 9}


def test_negative_array_index_rejected():
    with pytest.raises(PatchPathNotFound):
        apply_patch([1, 2, 3], [{"op": "remove", "path": "/-1"}])


def test_leading_zero_index_rejected():
    with pytest.raises(PatchPathNotFound):
        apply_patch([1, 2, 3], [{"op": "remove", "path": "/01"}])


# ---------------- malformed ops ----------------


def test_unknown_op_rejected():
    with pytest.raises(PatchInvalidOp):
        apply_patch({}, [{"op": "frob", "path": "/x", "value": 1}])


def test_missing_value_rejected():
    with pytest.raises(PatchInvalidOp):
        apply_patch({}, [{"op": "add", "path": "/x"}])


def test_path_must_start_with_slash():
    with pytest.raises(PatchInvalidOp):
        apply_patch({}, [{"op": "add", "path": "x", "value": 1}])


def test_ops_must_be_list():
    with pytest.raises(PatchInvalidOp):
        apply_patch({}, {"op": "add", "path": "/x", "value": 1})  # type: ignore[arg-type]


def test_op_must_be_dict():
    with pytest.raises(PatchInvalidOp):
        apply_patch({}, ["nope"])  # type: ignore[list-item]


def test_error_hierarchy():
    assert issubclass(PatchPathNotFound, PatchError)
    assert issubclass(PatchTestFailed, PatchError)
    assert issubclass(PatchInvalidOp, PatchError)
    assert issubclass(PatchError, ValueError)


def test_apply_patch_does_not_mutate_input():
    src = {"a": [1, 2, 3]}
    snapshot = {"a": [1, 2, 3]}
    out = apply_patch(src, [{"op": "remove", "path": "/a/0"}])
    # Strict: source unchanged byte-for-byte; result distinct object with size-1
    assert src == snapshot
    assert len(src["a"]) == 3
    assert sum(src["a"]) == 6
    assert id(out) != id(src)
    assert len(out["a"]) == 2
    assert sum(out["a"]) == 5


# ---------------- numeric-threshold metrics ----------------


def test_remove_decreases_array_length_by_one():
    src = list(range(20))
    out = apply_patch(src, [{"op": "remove", "path": "/5"}])
    assert len(out) == len(src) - 1
    assert len(out) == 19
    # The removed element is gone, surrounding elements preserved
    assert sum(1 for x in out if x == 5) == 0
    assert out[:5] == [0, 1, 2, 3, 4]
    assert out[5:] == [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


def test_add_increases_array_length_by_one():
    src = list(range(50))
    out = apply_patch(src, [{"op": "add", "path": "/-", "value": 999}])
    assert len(out) == len(src) + 1
    assert len(out) == 51
    assert out[-1] == 999


def test_bulk_patch_op_count_threshold():
    src = {"k0": 0, "k1": 1, "k2": 2, "k3": 3, "k4": 4}
    ops = [
        {"op": "replace", "path": f"/k{i}", "value": i * 10} for i in range(5)
    ]
    out = apply_patch(src, ops)
    assert len(out) == 5
    # All five values were rewritten
    rewritten = sum(1 for i in range(5) if out[f"k{i}"] == i * 10)
    assert rewritten == 5


def test_copy_does_not_share_reference():
    src = {"a": {"x": [1, 2, 3]}}
    out = apply_patch(src, [{"op": "copy", "from": "/a", "path": "/b"}])
    # Mutating one side must leave the other untouched (deep copy)
    out["b"]["x"].append(99)
    assert len(out["a"]["x"]) == 3
    assert len(out["b"]["x"]) == 4
    assert out["b"]["x"][-1] == 99


def test_large_sequential_add_grows_to_exact_size():
    """100 sequential add-at-end ops must yield length exactly 100."""
    src: list[int] = []
    ops = [{"op": "add", "path": "/-", "value": i} for i in range(100)]
    out = apply_patch(src, ops)
    assert len(out) == 100
    assert out[0] == 0
    assert out[-1] == 99
    # Sum of 0..99 is 4950 (Gauss formula): exact arithmetic invariant
    assert sum(out) == 4950
    # Source untouched
    assert len(src) == 0


def test_move_preserves_object_size_exactly():
    src = {f"k{i}": i for i in range(50)}
    ops = [{"op": "move", "from": "/k0", "path": "/moved"}]
    out = apply_patch(src, ops)
    # Move = remove + add: net size unchanged
    assert len(out) == len(src)
    assert len(out) == 50
    assert "k0" not in out
    assert out["moved"] == 0


def test_test_op_failure_atomicity_byte_exact():
    """When test fails mid-batch, NO partial mutation must reach the result."""
    src = {"a": 1, "counter": 0}
    ops = [
        {"op": "replace", "path": "/counter", "value": 1},
        {"op": "replace", "path": "/counter", "value": 2},
        {"op": "test", "path": "/a", "value": 999},  # fails
        {"op": "replace", "path": "/counter", "value": 3},
    ]
    with pytest.raises(PatchTestFailed):
        apply_patch(src, ops)
    # Source must be byte-identical to original
    assert src == {"a": 1, "counter": 0}
    assert len(src) == 2
    assert src["counter"] == 0


def test_pointer_escape_roundtrip_byte_count():
    """Path with both ~0 and ~1 escapes resolves to exactly the right key."""
    weird_key = "a/b~c"  # contains both / and ~
    # Escape: / -> ~1, ~ -> ~0; ordering matters (~ first)
    escaped = "/a~1b~0c"
    src = {weird_key: "x" * 64}
    out = apply_patch(src, [{"op": "replace", "path": escaped, "value": "y" * 32}])
    assert out[weird_key] == "y" * 32
    assert len(out[weird_key]) == 32
    assert len(out) == 1


def test_chained_remove_then_add_net_size_invariant():
    """remove + add at end must yield same length as the source."""
    src = list(range(64))
    ops = [
        {"op": "remove", "path": "/0"},
        {"op": "add", "path": "/-", "value": 999},
    ]
    out = apply_patch(src, ops)
    assert len(out) == len(src)
    assert len(out) == 64
    assert out[0] == 1
    assert out[-1] == 999
    # Sum invariant: original sum 0..63 = 2016; remove 0 keeps 2016; +999 = 3015
    assert sum(out) == 2016 + 999
    assert sum(out) == 3015


def test_deep_nested_add_path_resolves_exactly():
    """Adding into a 5-level-deep object reaches the right leaf, others unchanged."""
    src = {"a": {"b": {"c": {"d": {"e": 1}}}}}
    out = apply_patch(
        src,
        [{"op": "add", "path": "/a/b/c/d/f", "value": 2}],
    )
    assert out["a"]["b"]["c"]["d"]["e"] == 1
    assert out["a"]["b"]["c"]["d"]["f"] == 2
    assert len(out["a"]["b"]["c"]["d"]) == 2
    # Source must be untouched
    assert "f" not in src["a"]["b"]["c"]["d"]
    assert len(src["a"]["b"]["c"]["d"]) == 1


def test_nested_array_replace_index_count_invariant():
    """Replacing an element inside a nested array preserves all lengths."""
    src = {"rows": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}
    out = apply_patch(
        src,
        [{"op": "replace", "path": "/rows/1/1", "value": 99}],
    )
    assert len(out["rows"]) == 3
    assert all(len(row) == 3 for row in out["rows"])
    assert out["rows"][1][1] == 99
    # Only one cell changed: exact diff count
    diffs = sum(
        1
        for i in range(3)
        for j in range(3)
        if src["rows"][i][j] != out["rows"][i][j]
    )
    assert diffs == 1


def test_thousand_op_batch_exact_arithmetic():
    """1000 sequential add-then-remove pairs: net length unchanged, sum bounded."""
    src = list(range(1000))
    ops = []
    for i in range(500):
        ops.append({"op": "add", "path": "/-", "value": 10_000 + i})
        ops.append({"op": "remove", "path": "/0"})
    out = apply_patch(src, ops)
    assert len(out) == 1000
    assert len(out) == len(src)
    # Original sum 0..999 = 499500. Removed 0..499 (sum 124750), added 10000..10499 (sum 5124750).
    expected = sum(range(1000)) - sum(range(500)) + sum(range(10_000, 10_500))
    assert sum(out) == expected
    # Source must be byte-identical
    assert src == list(range(1000))


def test_error_subclass_chain_depth_exact():
    """Strict: every patch error inherits from PatchError AND ValueError, depth >= 2."""
    for cls in (PatchPathNotFound, PatchTestFailed, PatchInvalidOp):
        mro_names = [c.__name__ for c in cls.__mro__]
        assert "PatchError" in mro_names
        assert "ValueError" in mro_names
        # Distance from concrete -> ValueError must be at least 2 hops
        assert mro_names.index("ValueError") >= 2
