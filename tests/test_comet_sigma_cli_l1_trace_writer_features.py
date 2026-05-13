"""Tests for ``stepback comet-sigma-l1-trace-writer-features`` (Step 8).

Covers the CLI subcommand wired in
:mod:`stepback.comet_sigma.cli_l1_trace_writer_features`:

* text + JSON output paths produce the expected base feature set,
* ``--include-temporal`` adds the Step-2 projection block and per-window
  frame counts,
* the command restores the ``COMET_SIGMA_L1_TEMPORAL`` env var and
  ``OBSERVE_HOOKS`` list to whatever they were before the invocation
  (i.e. it never leaks flag/hook state),
* error paths return the documented exit codes (missing trace, empty
  trace, malformed file).
"""
from __future__ import annotations

import io
import json
import os
import pathlib
from contextlib import redirect_stderr, redirect_stdout

import pytest

from stepback.cli import main as cli_main
from stepback.comet_sigma import (
    comet_sigma_available,
    l1_trace_writer as l1,
    l1_trace_writer_temporal as l1t,
)
from stepback.comet_sigma.cli_l1_trace_writer_features import SCHEMA_VERSION
from stepback.trace_writer import TraceWriter


pytestmark = pytest.mark.skipif(
    not comet_sigma_available(),
    reason="upstream comet_sigma not importable on this interpreter",
)


def _write_trace(tmp_path: pathlib.Path, n: int = 5, name: str = "t.sb") -> str:
    path = str(tmp_path / name)
    w = TraceWriter.open(path)
    for i in range(n):
        w.write_step({"type": "step", "i": i, "payload": "x" * 8})
    w.close()
    return path


def _capture_cli(argv: list) -> tuple:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = cli_main(argv)
    return rc, out.getvalue(), err.getvalue()


@pytest.fixture(autouse=True)
def _isolate_cli(monkeypatch):
    """Snapshot + restore env + registries around every test so cross-test
    leakage is impossible regardless of how the CLI behaves."""
    prev_flag = os.environ.get(l1.FLAG_NAME)
    prev_hooks = list(l1.OBSERVE_HOOKS)
    yield
    os.environ.pop(l1.FLAG_NAME, None)
    if prev_flag is not None:
        os.environ[l1.FLAG_NAME] = prev_flag
    l1.OBSERVE_HOOKS[:] = prev_hooks
    l1.reset()
    l1t.reset()


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

def test_text_output_lists_every_base_feature(tmp_path):
    trace = _write_trace(tmp_path)
    rc, out, err = _capture_cli(
        ["comet-sigma-l1-trace-writer-features", trace]
    )
    assert rc == 0, err
    # One line per base feature, plus header + writer_id + frames_observed.
    for spec in l1.BASE_FEATURES:
        assert f"{spec.name}\t" in out, f"{spec.name} missing from text output"
    assert "frames_observed\t" in out


def test_json_output_is_well_formed(tmp_path):
    trace = _write_trace(tmp_path, n=4)
    rc, out, _ = _capture_cli(
        ["comet-sigma-l1-trace-writer-features", trace, "--json"]
    )
    assert rc == 0
    obj = json.loads(out)
    assert obj["schema_version"] == SCHEMA_VERSION
    assert obj["frames_observed"] >= 1
    assert obj["writer_id"].startswith("cli:")
    assert obj["trace_path"] == os.path.abspath(trace)
    # Every base feature present and float-typed.
    for spec in l1.BASE_FEATURES:
        assert spec.name in obj["latest"]
        assert isinstance(obj["latest"][spec.name], float)
    # No temporal block unless asked for.
    assert "temporal" not in obj


def test_include_temporal_adds_projection_block(tmp_path):
    trace = _write_trace(tmp_path, n=6)
    rc, out, _ = _capture_cli(
        [
            "comet-sigma-l1-trace-writer-features", trace,
            "--include-temporal", "--json",
        ]
    )
    assert rc == 0
    obj = json.loads(out)
    assert obj["temporal"] is not None
    proj = obj["temporal"]
    assert set(proj["window_counts"].keys()) >= {"1s", "10s", "1m", "10m"}
    # Every window had at least one frame in it (we just wrote them).
    assert all(c >= 1 for c in proj["window_counts"].values())
    # Some projection values exist; pick a couple of canonical names.
    keys = list(proj["values"].keys())
    assert any(k.endswith(".1s.mean") for k in keys)
    assert any(k.endswith(".10m.last_minus_first") for k in keys)


def test_writer_id_override(tmp_path):
    trace = _write_trace(tmp_path)
    rc, out, _ = _capture_cli(
        [
            "comet-sigma-l1-trace-writer-features", trace,
            "--writer-id", "explicit-id", "--json",
        ]
    )
    assert rc == 0
    obj = json.loads(out)
    assert obj["writer_id"] == "explicit-id"


def test_max_frames_caps_observation(tmp_path):
    trace = _write_trace(tmp_path, n=20)
    rc, out, _ = _capture_cli(
        [
            "comet-sigma-l1-trace-writer-features", trace,
            "--max-frames", "3", "--json",
        ]
    )
    assert rc == 0
    obj = json.loads(out)
    # frames_observed counts only frames the L1 emitter accepted, but
    # cannot exceed the requested cap.
    assert 1 <= obj["frames_observed"] <= 3


# ---------------------------------------------------------------------------
# State hygiene
# ---------------------------------------------------------------------------

def test_cli_restores_flag_and_hooks(tmp_path, monkeypatch):
    monkeypatch.delenv(l1.FLAG_NAME, raising=False)
    pre_hooks = list(l1.OBSERVE_HOOKS)
    trace = _write_trace(tmp_path)
    rc, _, _ = _capture_cli(
        [
            "comet-sigma-l1-trace-writer-features", trace,
            "--include-temporal",
        ]
    )
    assert rc == 0
    # Flag must not have been left enabled.
    assert os.environ.get(l1.FLAG_NAME) is None
    # Hook list must be back exactly to its pre-call state.
    assert list(l1.OBSERVE_HOOKS) == pre_hooks


def test_cli_restores_preexisting_flag_value(tmp_path, monkeypatch):
    monkeypatch.setenv(l1.FLAG_NAME, "0")
    trace = _write_trace(tmp_path)
    rc, _, _ = _capture_cli(
        ["comet-sigma-l1-trace-writer-features", trace]
    )
    assert rc == 0
    assert os.environ.get(l1.FLAG_NAME) == "0"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_missing_trace_returns_2(tmp_path):
    bogus = str(tmp_path / "no-such.sb")
    rc, _, err = _capture_cli(
        ["comet-sigma-l1-trace-writer-features", bogus]
    )
    assert rc == 2
    assert "trace not found" in err


def test_malformed_trace_returns_2(tmp_path):
    bad = tmp_path / "bad.sb"
    bad.write_bytes(b"\x00\x00\x00\x05short")  # length prefix > body
    rc, _, err = _capture_cli(
        ["comet-sigma-l1-trace-writer-features", str(bad)]
    )
    assert rc == 2
    assert "failed to read frames" in err


def test_empty_trace_returns_4(tmp_path):
    empty = tmp_path / "empty.sb"
    empty.write_bytes(b"")
    rc, _, err = _capture_cli(
        ["comet-sigma-l1-trace-writer-features", str(empty)]
    )
    assert rc == 4
    assert "no frames observed" in err
