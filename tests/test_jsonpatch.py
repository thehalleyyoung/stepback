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
    apply_patch(src, [{"op": "remove", "path": "/a/0"}])
    assert src == {"a": [1, 2, 3]}


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
