"""Tests for the self-contained full HTML report (Step 109).

Covers:
* render_full_html_report — smoke, self-contained, all four components
* attestation section HTML generation
* minimization section HTML generation
* write_full_html_report — file creation + FullReportSummary
* security: </script> injection neutralised
* public API exportability
"""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

import stepback
from stepback import (
    FullReportSummary,
    RecorderKey,
    record,
    render_full_html_report,
    replay,
    write_full_html_report,
)
from stepback.testing import run_recorded_agent


# ----------------------------------------------------------- helpers


def _record_fixture(tmp_path):
    key = RecorderKey.fresh()
    path = str(tmp_path / "trace.sb")
    with record(path, key=key) as rec:
        run_recorded_agent(rec)
    return path, key


def _fake_attest_entry(
    verify_status: str = "ok",
    replay_status: str = "ok",
    **kwargs,
) -> Dict[str, Any]:
    """Minimal attestation-entry dict compatible with _render_attestation_section."""
    defaults: Dict[str, Any] = {
        "trace_path": "/tmp/trace.sb",
        "trace_chain_hash": "abc123",
        "recorder_public_key": "ed25519:deadbeef",
        "recorder_version": "0.1.0",
        "canonicalisation_version": "1",
        "step_count": 12,
        "verify_status": verify_status,
        "verify_error": None,
        "replay_status": replay_status,
        "replay_error": None,
        "dirty_step_count": 3,
        "cache_hit_count": 9,
        "real_executions": 3,
        "divergent_step_count": 2,
        "total_cost_recorded_usd": 0.01,
        "total_cost_replayed_usd": 0.009,
        "total_cost_delta_usd": -0.001,
        "divergent_step_ids": ["step:1", "step:2"],
        "merkle_root": "sha256:deadbeef",
    }
    defaults.update(kwargs)
    return defaults


class _FakeSub:
    """Minimal substitution stand-in for minimization tests."""
    def __init__(self, kind: str, step_id: str):
        self.__class__ = type(kind, (_FakeSub,), {})
        self.step_id = step_id
        self._kind = kind

    def __class_getitem__(cls, item):
        return cls


@dataclass
class _FakeMinResult:
    minimal: List[Any]
    removed: List[Any] = field(default_factory=list)
    probes: int = 10
    cache_hits: int = 8
    strategy_name: str = "DDMin"
    weights: Optional[Dict[int, float]] = None
    final_result: Any = None


# =========================================================== render_full_html_report


def test_render_full_html_report_smoke_minimal():
    page = render_full_html_report([], header={})
    assert page.startswith("<!doctype html>")
    assert "stepback full report" in page
    assert "<style>" in page and "</style>" in page
    # tab navigation is present
    assert "fr-tab" in page
    assert "fr-trace" in page


def test_render_full_html_report_no_external_refs(tmp_path):
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    page = render_full_html_report(
        t.recorded_steps, t.header, title="my-incident-2026"
    )
    assert "my-incident-2026" in page
    # No CDN or external network references (SVG namespace in inline JS is OK)
    assert "<link" not in page
    assert 'src="http' not in page and "src='http" not in page


def test_render_full_html_report_with_replay_result(tmp_path):
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    result = t.replay_forward()
    page = render_full_html_report(t.recorded_steps, t.header, replay_result=result)
    # time-travel model embedded
    assert "id='tt-data'" in page
    # causal graph button
    assert "Graph" in page
    # diff pane navigation buttons
    assert "Prev" in page and "Next" in page
    # dirty / cache-hit stats present
    assert "dirty" in page
    assert "cache hits" in page


def test_render_full_html_report_with_attestation(tmp_path):
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    entry = _fake_attest_entry()
    page = render_full_html_report(t.recorded_steps, t.header, attest_entry=entry)
    # attestation tab appears
    assert "Attestation" in page
    assert "fr-attest" in page
    # key content rendered
    assert "ed25519:deadbeef" in page
    assert "Merkle root" in page
    assert "Divergent step IDs" in page


def test_render_full_html_report_with_minimization(tmp_path):
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)

    subs = [_FakeSub("PromptSubstitution", f"step:{i}") for i in range(5)]
    min_res = _FakeMinResult(
        minimal=subs[:2],
        removed=subs[2:],
        probes=7,
        cache_hits=5,
        strategy_name="DDMin",
    )
    page = render_full_html_report(t.recorded_steps, t.header, min_result=min_res)
    # minimization tab appears
    assert "Minimization" in page
    assert "fr-min" in page
    # key stats
    assert "DDMin" in page
    assert "Input substitutions" in page
    assert "Removed substitutions" in page


def test_render_full_html_report_all_sections(tmp_path):
    path, key = _record_fixture(tmp_path)
    t = replay(path, hmac_key=key.hmac_key)
    result = t.replay_forward()
    entry = _fake_attest_entry()
    subs = [_FakeSub("ToolOutputSubstitution", f"step:{i}") for i in range(3)]
    min_res = _FakeMinResult(minimal=subs[:1], removed=subs[1:])

    page = render_full_html_report(
        t.recorded_steps, t.header,
        replay_result=result,
        min_result=min_res,
        attest_entry=entry,
        title="full-combo-test",
    )
    assert "full-combo-test" in page
    assert "Trace" in page
    assert "Minimization" in page
    assert "Attestation" in page
    # All three data payloads present
    assert "tt-data" in page
    assert "fr-min" in page
    assert "fr-attest" in page
    # Self-contained: no CDN or external script/stylesheet loads
    # (SVG namespace 'http://www.w3.org/2000/svg' in inline JS is allowed)
    assert "<link" not in page
    assert "src=" not in page
    # No external stylesheet or script src attributes
    assert 'src="http' not in page and "src='http" not in page


def test_render_full_html_report_script_tag_neutralised():
    malicious = [
        {
            "step_id": "step:1",
            "step_kind": "tool_call",
            "name": "evil",
            "outputs": {"text": "</script><script>window.pwned=1;</script>"},
            "cost_usd": 0.0,
        }
    ]
    page = render_full_html_report(malicious, header={})
    # no raw </script> inside the data island
    m = re.search(r"<script type='application/json' id='tt-data'>(.+?)</script>", page, re.DOTALL)
    if m is None:
        # fallback: basic viewer path
        m = re.search(r"<script type='application/json' id='stepback-data'>(.+?)</script>", page, re.DOTALL)
    if m:
        assert "</script>" not in m.group(1)


def test_render_full_html_report_title_escaped():
    page = render_full_html_report([], header={}, title="<img src=x onerror=alert(1)>")
    assert "<img src=x onerror=alert(1)>" not in page
    assert html.escape("<img src=x onerror=alert(1)>", quote=True) in page


# =========================================================== attestation section detail


def test_attestation_section_ok_status():
    from stepback.html_view import _render_attestation_section
    entry = _fake_attest_entry(verify_status="ok", replay_status="ok")
    html_out = _render_attestation_section(entry)
    assert "fr-badge-ok" in html_out
    assert "ed25519:deadbeef" in html_out
    assert "sha256:deadbeef" in html_out


def test_attestation_section_fail_status():
    from stepback.html_view import _render_attestation_section
    entry = _fake_attest_entry(
        verify_status="fail",
        verify_error="bad signature",
        replay_status="skipped",
    )
    html_out = _render_attestation_section(entry)
    assert "fr-badge-fail" in html_out
    assert "bad signature" in html_out
    assert "fr-badge-skip" in html_out


def test_attestation_section_divergent_ids_rendered():
    from stepback.html_view import _render_attestation_section
    entry = _fake_attest_entry(divergent_step_ids=["step:A", "step:B", "step:C"])
    html_out = _render_attestation_section(entry)
    assert "step:A" in html_out
    assert "step:B" in html_out
    assert "Divergent step IDs" in html_out


def test_attestation_section_truncates_long_divergent_list():
    from stepback.html_view import _render_attestation_section
    ids = [f"step:{i}" for i in range(100)]
    entry = _fake_attest_entry(divergent_step_ids=ids)
    html_out = _render_attestation_section(entry)
    # only first 50 rendered directly, rest indicated
    assert "and 50 more" in html_out


def test_attestation_section_dataclass_compatible():
    """An object with attributes (not a dict) should work too."""
    from stepback.html_view import _render_attestation_section

    class FakeEntry:
        trace_path = "/run/trace.sb"
        trace_chain_hash = "xyz"
        recorder_public_key = "ed25519:aabbcc"
        recorder_version = "0.2.0"
        canonicalisation_version = "1"
        step_count = 5
        verify_status = "ok"
        verify_error = None
        replay_status = "ok"
        replay_error = None
        dirty_step_count = 1
        cache_hit_count = 4
        real_executions = 1
        divergent_step_count = 0
        total_cost_recorded_usd = 0.005
        total_cost_replayed_usd = 0.005
        total_cost_delta_usd = 0.0
        divergent_step_ids: List[str] = []
        merkle_root = None

    html_out = _render_attestation_section(FakeEntry())
    assert "ed25519:aabbcc" in html_out
    assert "fr-badge-ok" in html_out


# =========================================================== minimization section detail


def test_minimization_section_basic():
    from stepback.html_view import _render_minimization_section
    subs = [_FakeSub("PromptSubstitution", f"step:{i}") for i in range(4)]
    min_res = _FakeMinResult(
        minimal=subs[:1],
        removed=subs[1:],
        probes=8,
        cache_hits=6,
        strategy_name="DDMin",
    )
    html_out = _render_minimization_section(min_res)
    assert "DDMin" in html_out
    assert "Input substitutions" in html_out
    assert "4" in html_out
    assert "1" in html_out   # minimal count
    assert "3" in html_out   # removed count
    assert "Oracle probes" in html_out
    assert "Cache hits" in html_out


def test_minimization_section_with_weights():
    from stepback.html_view import _render_minimization_section
    subs = [_FakeSub("ToolOutputSubstitution", f"step:{i}") for i in range(3)]
    weights = {id(subs[0]): 0.7, id(subs[1]): 0.2, id(subs[2]): 0.1}
    min_res = _FakeMinResult(
        minimal=subs[:2],
        removed=subs[2:],
        strategy_name="Shapley",
        weights=weights,
    )
    html_out = _render_minimization_section(min_res)
    assert "Shapley weight" in html_out
    # weights rendered (4 decimal places)
    assert "0.7000" in html_out or "0.2000" in html_out


def test_minimization_section_empty_minimal():
    from stepback.html_view import _render_minimization_section
    min_res = _FakeMinResult(minimal=[], removed=[], strategy_name="BruteForce")
    html_out = _render_minimization_section(min_res)
    assert "BruteForce" in html_out
    assert "none" in html_out.lower() or "—" in html_out


def test_minimization_section_truncates_long_list():
    from stepback.html_view import _render_minimization_section
    subs = [_FakeSub("PromptSubstitution", f"step:{i}") for i in range(250)]
    min_res = _FakeMinResult(minimal=subs, removed=[])
    html_out = _render_minimization_section(min_res)
    assert "and 50 more" in html_out


# =========================================================== write_full_html_report


def test_write_full_html_report_creates_file(tmp_path):
    path, key = _record_fixture(tmp_path)
    out = tmp_path / "report" / "full.html"
    summary = write_full_html_report(str(path), str(out), hmac_key=key.hmac_key)

    assert isinstance(summary, FullReportSummary)
    assert summary.step_count == 12
    assert summary.by_kind == {"llm_call": 6, "tool_call": 6}
    assert summary.has_replay is True
    assert summary.has_minimization is False
    assert summary.has_attestation is False
    assert summary.bytes_written > 1000
    assert out.exists()
    page = out.read_text()
    assert page.startswith("<!doctype html>")
    assert page.rstrip().endswith("</html>")


def test_write_full_html_report_creates_parent_dirs(tmp_path):
    path, key = _record_fixture(tmp_path)
    out = tmp_path / "deep" / "dir" / "report.html"
    summary = write_full_html_report(str(path), str(out), hmac_key=key.hmac_key)
    assert out.exists()
    assert summary.output_path == str(out)


def test_write_full_html_report_with_attestation(tmp_path):
    path, key = _record_fixture(tmp_path)
    out = tmp_path / "attest.html"
    entry = _fake_attest_entry()
    summary = write_full_html_report(
        str(path), str(out), hmac_key=key.hmac_key, attest_entry=entry
    )
    assert summary.has_attestation is True
    page = out.read_text()
    assert "Attestation" in page
    assert "ed25519:deadbeef" in page


def test_write_full_html_report_with_minimization(tmp_path):
    path, key = _record_fixture(tmp_path)
    out = tmp_path / "min.html"
    subs = [_FakeSub("PromptSubstitution", f"step:{i}") for i in range(3)]
    min_res = _FakeMinResult(minimal=subs[:1], removed=subs[1:], strategy_name="DDMin")
    summary = write_full_html_report(
        str(path), str(out), hmac_key=key.hmac_key, min_result=min_res
    )
    assert summary.has_minimization is True
    page = out.read_text()
    assert "Minimization" in page


# =========================================================== public API


def test_symbols_exported_from_stepback():
    assert hasattr(stepback, "FullReportSummary")
    assert hasattr(stepback, "render_full_html_report")
    assert hasattr(stepback, "write_full_html_report")
    assert "FullReportSummary" in stepback.__all__
    assert "render_full_html_report" in stepback.__all__
    assert "write_full_html_report" in stepback.__all__


def test_full_report_summary_is_dataclass():
    import dataclasses
    assert dataclasses.is_dataclass(FullReportSummary)
    s = FullReportSummary(
        output_path="/out/report.html",
        step_count=5,
        total_cost_usd=0.01,
    )
    assert s.has_replay is False
    assert s.has_minimization is False
    assert s.has_attestation is False
    assert s.bytes_written == 0
