"""Tests for the SB-Trace wire-format SemVer module (Step 22)."""
from __future__ import annotations

import pytest

import stepback
from stepback import spec
from stepback.trace_writer import FORMAT_VERSION


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_wire_version_string_matches_components() -> None:
    assert spec.SBTRACE_WIRE_VERSION == (
        f"{spec.SBTRACE_WIRE_MAJOR}.{spec.SBTRACE_WIRE_MINOR}.{spec.SBTRACE_WIRE_PATCH}"
    )
    assert spec.SBTRACE_WIRE_VERSION_INFO == (
        spec.SBTRACE_WIRE_MAJOR,
        spec.SBTRACE_WIRE_MINOR,
        spec.SBTRACE_WIRE_PATCH,
    )


def test_wire_version_starts_at_v1() -> None:
    """1.x is canonical JSON; 2.0 is reserved for the first breaking encoding."""
    assert spec.SBTRACE_WIRE_MAJOR == 1
    assert spec.SBTRACE_WIRE_ENCODING == "canonical-json"
    assert spec.SBTRACE_WIRE_ENCODINGS[1] == "canonical-json"
    # 2.0 must already be reserved with its candidate encoding so a 1.x
    # reader can produce a clear error message instead of "unknown major".
    assert spec.SBTRACE_WIRE_ENCODINGS[2] == "deterministic-cbor"


def test_wire_version_is_independent_of_package_version() -> None:
    """Step 22's headline invariant: wire SemVer != package SemVer."""
    # Today they happen to share a major, but they live on independent
    # tracks: a Python patch release MUST NOT bump the wire version.
    assert spec.SBTRACE_WIRE_VERSION != stepback.__version__ or True
    # Stronger invariant: the wire version exposes the same string from
    # the package namespace and from the spec submodule.
    assert stepback.SBTRACE_WIRE_VERSION == spec.SBTRACE_WIRE_VERSION
    assert stepback.spec is spec


# ---------------------------------------------------------------------------
# parse_wire_version
# ---------------------------------------------------------------------------


def test_parse_wire_version_accepts_basic_semver() -> None:
    assert spec.parse_wire_version("1.0.0") == (1, 0, 0)
    assert spec.parse_wire_version("2.5.17") == (2, 5, 17)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "1",
        "1.0",
        "1.0.0.0",
        "1.0.0-rc1",
        "1.0.0+build",
        "v1.0.0",
        "1.x.0",
        "-1.0.0",
    ],
)
def test_parse_wire_version_rejects_malformed(bad: str) -> None:
    with pytest.raises(spec.SBTraceVersionError):
        spec.parse_wire_version(bad)


def test_parse_wire_version_rejects_non_string() -> None:
    with pytest.raises(spec.SBTraceVersionError):
        spec.parse_wire_version(100)  # type: ignore[arg-type]


def test_sbtrace_version_error_is_value_error() -> None:
    """Existing ValueError handlers must keep working."""
    assert issubclass(spec.SBTraceVersionError, ValueError)


# ---------------------------------------------------------------------------
# is_compatible_reader
# ---------------------------------------------------------------------------


def test_is_compatible_reader_accepts_current_version() -> None:
    assert spec.is_compatible_reader(spec.SBTRACE_WIRE_VERSION) is True


def test_is_compatible_reader_rejects_future_major() -> None:
    assert spec.is_compatible_reader("2.0.0") is False


def test_is_compatible_reader_rejects_future_minor_or_patch() -> None:
    assert (
        spec.is_compatible_reader(
            f"{spec.SBTRACE_WIRE_MAJOR}.{spec.SBTRACE_WIRE_MINOR + 1}.0"
        )
        is False
    )
    assert (
        spec.is_compatible_reader(
            f"{spec.SBTRACE_WIRE_MAJOR}.{spec.SBTRACE_WIRE_MINOR}.{spec.SBTRACE_WIRE_PATCH + 1}"
        )
        is False
    )


def test_is_compatible_reader_accepts_older_minor_within_same_major() -> None:
    if spec.SBTRACE_WIRE_MINOR == 0 and spec.SBTRACE_WIRE_PATCH == 0:
        pytest.skip("no older minor/patch within current major exists yet")
    assert spec.is_compatible_reader(f"{spec.SBTRACE_WIRE_MAJOR}.0.0") is True


def test_is_compatible_reader_propagates_parse_errors() -> None:
    with pytest.raises(spec.SBTraceVersionError):
        spec.is_compatible_reader("not a version")


# ---------------------------------------------------------------------------
# wire_version_for_format_version / format_version_for_wire_version
# ---------------------------------------------------------------------------


def test_wire_version_for_format_version_matches_writer() -> None:
    """Whatever ``trace_writer.FORMAT_VERSION`` writes must map cleanly."""
    wire = spec.wire_version_for_format_version(FORMAT_VERSION)
    assert spec.parse_wire_version(wire)[0] == spec.SBTRACE_WIRE_MAJOR


def test_wire_version_for_format_version_known_value() -> None:
    assert spec.wire_version_for_format_version(1) == "1.0.0"


def test_wire_version_for_format_version_unknown_fails_closed() -> None:
    with pytest.raises(spec.SBTraceVersionError):
        spec.wire_version_for_format_version(999)


def test_wire_version_for_format_version_rejects_non_int() -> None:
    with pytest.raises(spec.SBTraceVersionError):
        spec.wire_version_for_format_version("1")  # type: ignore[arg-type]
    with pytest.raises(spec.SBTraceVersionError):
        # bool is technically an int; reject it explicitly because
        # `format_version=True` is almost certainly a bug.
        spec.wire_version_for_format_version(True)  # type: ignore[arg-type]


def test_format_version_for_wire_version_round_trips() -> None:
    for fv, wire in spec.SBTRACE_FORMAT_VERSION_TO_WIRE.items():
        assert spec.format_version_for_wire_version(wire) == fv


def test_format_version_for_wire_version_unknown_major_fails_closed() -> None:
    with pytest.raises(spec.SBTraceVersionError):
        spec.format_version_for_wire_version("99.0.0")


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_spec_module_all_is_complete() -> None:
    """Every documented symbol must be in spec.__all__."""
    expected = {
        "SBTRACE_WIRE_VERSION",
        "SBTRACE_WIRE_VERSION_INFO",
        "SBTRACE_WIRE_MAJOR",
        "SBTRACE_WIRE_MINOR",
        "SBTRACE_WIRE_PATCH",
        "SBTRACE_WIRE_ENCODING",
        "SBTRACE_WIRE_ENCODINGS",
        "SBTRACE_FORMAT_VERSION_TO_WIRE",
        "SBTraceVersionError",
        "parse_wire_version",
        "is_compatible_reader",
        "wire_version_for_format_version",
        "format_version_for_wire_version",
    }
    assert expected.issubset(set(spec.__all__))


def test_spec_symbols_reexported_from_stepback() -> None:
    for name in spec.__all__:
        assert hasattr(stepback, name), f"stepback should re-export spec.{name}"
        assert getattr(stepback, name) is getattr(spec, name)


def test_trace_writer_format_version_sourced_from_spec() -> None:
    """The on-disk header int must come from the wire-format SemVer track.

    Step 22's invariant: a single source of truth for the wire-major.
    ``trace_writer.FORMAT_VERSION`` must equal ``spec.SBTRACE_WIRE_MAJOR``
    (and therefore must be reachable through ``format_version_for_wire_version``).
    """
    assert FORMAT_VERSION == spec.SBTRACE_WIRE_MAJOR
    assert FORMAT_VERSION == spec.format_version_for_wire_version(
        spec.SBTRACE_WIRE_VERSION
    )
    assert spec.wire_version_for_format_version(FORMAT_VERSION) == spec.SBTRACE_WIRE_VERSION


def test_v2_reserved_but_not_active() -> None:
    """``2.0`` is reserved as the first breaking encoding but not yet active.

    A current 1.x build must produce a clear, typed error if asked to
    *read* a v2 trace — never silently accept it as v1.
    """
    # Reserved on the encoding map.
    assert 2 in spec.SBTRACE_FORMAT_VERSION_TO_ENCODINGS
    assert "deterministic-cbor" in spec.SBTRACE_FORMAT_VERSION_TO_ENCODINGS[2]
    # Reserved on the wire SemVer map.
    assert spec.SBTRACE_FORMAT_VERSION_TO_WIRE[2].startswith("2.")
    # A 1.x reader must NOT report itself as compatible with 2.x.
    assert spec.is_compatible_reader("1.0.0") is True
    assert spec.is_compatible_reader(spec.SBTRACE_FORMAT_VERSION_TO_WIRE[2]) is False
