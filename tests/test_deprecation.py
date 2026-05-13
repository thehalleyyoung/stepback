"""Tests for the deprecation policy helpers documented in
``docs/deprecation.md``.

These tests are the behavioural contract for ``stepback.deprecated``,
``stepback.deprecated_alias``, ``stepback.warn_deprecated``,
``stepback.format_deprecation_message``, and
``stepback.DeprecationPolicyError``. They are intentionally strict
about the message format because release-note tooling greps for the
``since=`` and ``removal=`` tokens.
"""

from __future__ import annotations

import inspect
import warnings

import pytest

import stepback
from stepback import (
    DeprecationPolicyError,
    deprecated,
    deprecated_alias,
    format_deprecation_message,
    warn_deprecated,
)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_helpers_are_public() -> None:
    for name in (
        "deprecated",
        "deprecated_alias",
        "warn_deprecated",
        "format_deprecation_message",
        "DeprecationPolicyError",
    ):
        assert name in stepback.__all__, f"{name!r} missing from stepback.__all__"
        assert getattr(stepback, name) is not None


# ---------------------------------------------------------------------------
# format_deprecation_message
# ---------------------------------------------------------------------------


def test_format_message_includes_required_tokens() -> None:
    msg = format_deprecation_message(
        "stepback.old_thing",
        replacement="stepback.new_thing",
        since="0.2",
        removal="0.4",
    )
    assert "stepback.old_thing" in msg
    assert "since=0.2" in msg
    assert "stepback 0.4" in msg
    assert "stepback.new_thing" in msg
    assert "docs/deprecation.md" in msg


def test_format_message_handles_no_replacement() -> None:
    msg = format_deprecation_message(
        "stepback.old_thing",
        replacement=None,
        since="0.2",
        removal="0.4",
    )
    assert "no direct replacement" in msg


@pytest.mark.parametrize(
    "kwargs",
    [
        {"replacement": "x", "since": "", "removal": "0.4"},
        {"replacement": "x", "since": "0.2", "removal": ""},
        {"replacement": "x", "since": "0.2", "removal": "0.2"},
    ],
)
def test_policy_violations_raise(kwargs: dict) -> None:
    with pytest.raises(DeprecationPolicyError):
        format_deprecation_message("stepback.foo", **kwargs)


def test_empty_name_is_rejected() -> None:
    with pytest.raises(DeprecationPolicyError):
        format_deprecation_message("", replacement="x", since="0.2", removal="0.4")


# ---------------------------------------------------------------------------
# warn_deprecated
# ---------------------------------------------------------------------------


def test_warn_deprecated_emits_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_deprecated(
            "stepback.foo",
            replacement="stepback.bar",
            since="0.2",
            removal="0.4",
        )
    assert len(caught) == 1
    assert issubclass(caught[0].category, DeprecationWarning)
    assert "stepback.foo" in str(caught[0].message)
    assert "since=0.2" in str(caught[0].message)


# ---------------------------------------------------------------------------
# @deprecated decorator
# ---------------------------------------------------------------------------


def test_deprecated_function_still_runs_and_warns() -> None:
    @deprecated(replacement="new_add", since="0.2", removal="0.4")
    def old_add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = old_add(2, 3)

    assert result == 5
    assert len(caught) == 1
    assert issubclass(caught[0].category, DeprecationWarning)
    assert "old_add" in str(caught[0].message)


def test_deprecated_function_preserves_signature_and_doc() -> None:
    @deprecated(replacement="new_add", since="0.2", removal="0.4")
    def old_add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    sig = inspect.signature(old_add)
    assert list(sig.parameters) == ["a", "b"]
    assert "Add two numbers." in (old_add.__doc__ or "")
    assert ".. deprecated:: 0.2" in (old_add.__doc__ or "")


def test_deprecated_class_warns_on_construction() -> None:
    @deprecated(replacement="NewBox", since="0.2", removal="0.4")
    class OldBox:
        def __init__(self, value: int) -> None:
            self.value = value

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        instance = OldBox(7)

    assert instance.value == 7
    assert len(caught) == 1
    assert issubclass(caught[0].category, DeprecationWarning)
    assert "OldBox" in str(caught[0].message)


def test_deprecated_rejects_missing_versions() -> None:
    with pytest.raises(DeprecationPolicyError):

        @deprecated(replacement="x", since="0.2", removal="")
        def f() -> None:
            return None


# ---------------------------------------------------------------------------
# deprecated_alias
# ---------------------------------------------------------------------------


def test_deprecated_alias_function_forwards_and_warns() -> None:
    def real(x: int) -> int:
        return x * 2

    old_real = deprecated_alias(real, name="old_real", since="0.2", removal="0.4")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = old_real(5)

    assert result == 10
    assert old_real.__name__ == "old_real"
    assert len(caught) == 1
    assert "old_real" in str(caught[0].message)
    # Default replacement points at the target's qualname.
    assert "real" in str(caught[0].message)


def test_deprecated_alias_class_forwards_and_warns() -> None:
    class Real:
        def __init__(self, value: int) -> None:
            self.value = value

    OldReal = deprecated_alias(Real, name="OldReal", since="0.2", removal="0.4")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        instance = OldReal(11)

    assert isinstance(instance, Real)
    assert instance.value == 11
    assert OldReal.__name__ == "OldReal"
    assert len(caught) == 1
    assert "OldReal" in str(caught[0].message)


def test_deprecated_alias_rejects_self_aliasing_versions() -> None:
    def real() -> None:
        return None

    with pytest.raises(DeprecationPolicyError):
        deprecated_alias(real, name="old_real", since="0.2", removal="0.2")


# ---------------------------------------------------------------------------
# Documentation invariant
# ---------------------------------------------------------------------------


def test_changelog_mentions_deprecation_policy() -> None:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    changelog = (repo_root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "Deprecation policy" in changelog
    assert "docs/deprecation.md" in changelog


def test_policy_doc_exists_and_states_minimum_overlap() -> None:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    doc = (repo_root / "docs" / "deprecation.md").read_text(encoding="utf-8")
    assert "minor release" in doc
    assert "DeprecationWarning" in doc
    assert "stepback.format_deprecation_message" in doc
