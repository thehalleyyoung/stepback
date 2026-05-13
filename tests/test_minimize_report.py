"""Tests for Step 87: HTML minimization reports.

Covers:
1. Basic smoke test: render_html_minimize_report returns non-empty HTML string.
2. All required section IDs appear in the output.
3. Summary section shows correct strategy/probe/cache-hit info.
4. Before/after section shows correct counts.
5. Minimal substitutions table is present.
6. Removed substitutions table is present.
7. Probe statistics section with cache-hit rate.
8. Attribution section present (with and without weights).
9. Final replay step table when final_result is set.
10. HTML escaping of adversarial substitution content.
11. Byte-determinism: same inputs produce same output.
12. Options flags: toggling sections on/off.
13. extra_metadata inserted in sorted key order.
14. Shapley weight values appear in the minimal-subs table.
15. MinimizeReportOptions dataclass defaults.
16. Result with no removed substitutions (1-of-1 minimal).
17. Result with no minimal substitutions (empty).
18. MultiObjectiveMinimizationResult renders Pareto-front section.
19. Public API exports.
20. report is well-formed HTML (doctype, head, body).
"""
from __future__ import annotations

import html
import re

import pytest

import stepback
from stepback import (
    MinimizeReportOptions,
    MinimizationResult,
    render_html_minimize_report,
)
from stepback.minimize import MultiObjectiveMinimizationResult
from stepback.replay import ReplayResult, StepView
from stepback.substitutions import (
    ModelSubstitution,
    PromptSubstitution,
    SubstitutionSet,
    ToolOutputSubstitution,
)


# ------------------------------------------------------------------- fixtures


def _make_prompt_sub(step_id: str = "step:1") -> PromptSubstitution:
    return PromptSubstitution(at_step=step_id, new_messages=[{"role": "user", "content": "hi"}])


def _make_tool_sub(step_id: str = "step:2") -> ToolOutputSubstitution:
    return ToolOutputSubstitution(at_step=step_id, fake_response={"result": "ok"})


def _make_model_sub(step_id: str = "step:3") -> ModelSubstitution:
    return ModelSubstitution(at_step=step_id, new_model_id="gpt-4o-mini")


def _make_step_view(
    step_id: str,
    step_kind: str = "tool_call",
    dirty: bool = False,
    cache_hit: bool = True,
    cost_usd: float = 0.0,
) -> StepView:
    """Build a minimal StepView for testing."""
    return StepView(
        step_id=step_id,
        kind=step_kind,
        name=f"step_{step_id}",
        parent_step_id=None,
        inputs={},
        outputs={},
        cost_usd=cost_usd,
        dirty=dirty,
        cache_hit=cache_hit,
        recorded_inputs_hash="sha256:abc",
        current_inputs_hash="sha256:abc",
    )


def _make_replay_result(n_steps: int = 3, dirty_indices: tuple = ()) -> ReplayResult:
    steps = []
    for i in range(n_steps):
        dirty = i in dirty_indices
        steps.append(
            _make_step_view(
                f"step:{i}",
                dirty=dirty,
                cache_hit=not dirty,
                cost_usd=0.01 * i,
            )
        )
    return ReplayResult(
        steps=steps,
        total_cost_usd=sum(s.cost_usd for s in steps),
        dirty_count=sum(1 for s in steps if s.dirty),
        cache_hit_count=sum(1 for s in steps if s.cache_hit),
        real_executions=sum(1 for s in steps if s.dirty),
        provenance=None,
    )


def _simple_result(
    minimal_count: int = 2,
    removed_count: int = 2,
    *,
    with_final: bool = False,
    with_weights: bool = False,
) -> MinimizationResult:
    """Build a synthetic MinimizationResult for testing."""
    minimal = [_make_prompt_sub(f"step:{i}") for i in range(minimal_count)]
    removed = [_make_tool_sub(f"step:{100+i}") for i in range(removed_count)]
    weights = {id(s): float(i + 1) / minimal_count for i, s in enumerate(minimal)} if with_weights else None
    return MinimizationResult(
        minimal=minimal,
        removed=removed,
        probes=10,
        cache_hits=5,
        strategy_name="DDMinStrategy",
        weights=weights,
        final_result=_make_replay_result(3, (1,)) if with_final else None,
    )


# ================================================================ basic tests


class TestSmoke:
    def test_returns_string(self):
        result = _simple_result()
        html_str = render_html_minimize_report(result)
        assert isinstance(html_str, str)
        assert len(html_str) > 100

    def test_is_html(self):
        result = _simple_result()
        html_str = render_html_minimize_report(result)
        assert "<!doctype html>" in html_str.lower()
        assert "<html" in html_str
        assert "</html>" in html_str

    def test_well_formed_head_body(self):
        result = _simple_result()
        html_str = render_html_minimize_report(result)
        assert "<head>" in html_str
        assert "</head>" in html_str
        assert "<body>" in html_str
        assert "</body>" in html_str


class TestRequiredSections:
    def test_summary_section(self):
        html_str = render_html_minimize_report(_simple_result())
        assert 'id="summary"' in html_str

    def test_before_after_section(self):
        html_str = render_html_minimize_report(_simple_result())
        assert 'id="before-after"' in html_str

    def test_minimal_subs_section(self):
        html_str = render_html_minimize_report(_simple_result())
        assert 'id="minimal-subs"' in html_str

    def test_removed_subs_section(self):
        html_str = render_html_minimize_report(_simple_result())
        assert 'id="removed-subs"' in html_str

    def test_probe_stats_section(self):
        html_str = render_html_minimize_report(_simple_result())
        assert 'id="probe-stats"' in html_str

    def test_attribution_section(self):
        html_str = render_html_minimize_report(_simple_result())
        assert 'id="attribution"' in html_str


class TestSummaryContent:
    def test_strategy_name_present(self):
        result = _simple_result()
        html_str = render_html_minimize_report(result)
        assert "DDMinStrategy" in html_str

    def test_probe_count_present(self):
        result = _simple_result()
        html_str = render_html_minimize_report(result)
        assert "10" in html_str  # probes=10

    def test_cache_hit_count_present(self):
        result = _simple_result()
        html_str = render_html_minimize_report(result)
        assert "5" in html_str  # cache_hits=5

    def test_correct_minimal_count_in_summary(self):
        result = _simple_result(minimal_count=3, removed_count=2)
        html_str = render_html_minimize_report(result)
        assert "3" in html_str  # minimal_n
        assert "2" in html_str  # removed_n

    def test_custom_title(self):
        opts = MinimizeReportOptions(title="My Custom Report")
        html_str = render_html_minimize_report(_simple_result(), options=opts)
        assert "My Custom Report" in html_str

    def test_extra_metadata_sorted(self):
        opts = MinimizeReportOptions(extra_metadata={"z_key": "val_z", "a_key": "val_a"})
        html_str = render_html_minimize_report(_simple_result(), options=opts)
        pos_a = html_str.index("a_key")
        pos_z = html_str.index("z_key")
        assert pos_a < pos_z, "extra_metadata keys must be rendered in sorted order"


class TestBeforeAfterSection:
    def test_original_count_shown(self):
        # minimal_count=2, removed_count=3 => original=5
        result = _simple_result(minimal_count=2, removed_count=3)
        html_str = render_html_minimize_report(result)
        assert "5" in html_str  # original total

    def test_final_result_stats_when_present(self):
        result = _simple_result(with_final=True)
        html_str = render_html_minimize_report(result)
        # should mention real executions from final_result
        assert "real executions" in html_str.lower() or "Real executions" in html_str


class TestSubsTables:
    def test_minimal_subs_listed(self):
        result = _simple_result(minimal_count=2)
        html_str = render_html_minimize_report(result)
        # PromptSubstitution kinds present in minimal table section
        assert "PromptSubstitution" in html_str

    def test_removed_subs_listed(self):
        result = _simple_result(removed_count=2)
        html_str = render_html_minimize_report(result)
        assert "ToolOutputSubstitution" in html_str

    def test_at_step_referenced(self):
        result = _simple_result(minimal_count=1)
        html_str = render_html_minimize_report(result)
        assert "step:0" in html_str

    def test_no_removed_subs_shows_none_message(self):
        result = _simple_result(minimal_count=2, removed_count=0)
        html_str = render_html_minimize_report(result)
        assert "(none)" in html_str


class TestProbeStats:
    def test_hit_rate_shown(self):
        # probes=10, cache_hits=5 => rate=33.3%
        result = _simple_result()
        html_str = render_html_minimize_report(result)
        assert "33.3%" in html_str

    def test_zero_probes_safe(self):
        result = MinimizationResult(
            minimal=[_make_prompt_sub()],
            removed=[],
            probes=0,
            cache_hits=0,
            strategy_name="test",
        )
        html_str = render_html_minimize_report(result)
        assert "n/a" in html_str  # division-by-zero guard shows "n/a"

    def test_total_evaluations(self):
        # probes=10 + cache_hits=5 = 15
        result = _simple_result()
        html_str = render_html_minimize_report(result)
        assert "15" in html_str


class TestAttribution:
    def test_no_weights_shows_placeholder(self):
        result = _simple_result(with_weights=False)
        html_str = render_html_minimize_report(result)
        assert "ShapleyAttributionStrategy" in html_str  # hint shown

    def test_weights_shown_when_present(self):
        result = _simple_result(minimal_count=2, with_weights=True)
        html_str = render_html_minimize_report(result)
        # Shapley weights are floats with 4 decimal places
        assert re.search(r"\d\.\d{4}", html_str), "Shapley weight values should appear"

    def test_attribution_ranking_present(self):
        result = _simple_result(minimal_count=2, with_weights=True)
        html_str = render_html_minimize_report(result)
        assert "Shapley weight" in html_str


class TestFinalResult:
    def test_final_result_section_when_present(self):
        result = _simple_result(with_final=True)
        html_str = render_html_minimize_report(result)
        assert 'id="final-result"' in html_str

    def test_no_final_result_section_when_absent(self):
        result = _simple_result(with_final=False)
        html_str = render_html_minimize_report(result)
        assert 'id="final-result"' not in html_str

    def test_step_ids_in_final_result(self):
        result = _simple_result(with_final=True)
        html_str = render_html_minimize_report(result)
        # _make_replay_result creates steps step:0, step:1, step:2
        assert "step:0" in html_str

    def test_dirty_badge_in_final_result(self):
        result = _simple_result(with_final=True)
        html_str = render_html_minimize_report(result)
        assert "badge-dirty" in html_str or "dirty" in html_str


class TestHTMLEscaping:
    def test_xss_in_strategy_name(self):
        result = MinimizationResult(
            minimal=[],
            removed=[],
            probes=0,
            cache_hits=0,
            strategy_name="<script>alert('xss')</script>",
        )
        html_str = render_html_minimize_report(result)
        assert "<script>" not in html_str
        assert "&lt;script&gt;" in html_str

    def test_xss_in_extra_metadata(self):
        opts = MinimizeReportOptions(extra_metadata={"<b>key</b>": "<script>evil</script>"})
        html_str = render_html_minimize_report(_simple_result(), options=opts)
        assert "<script>" not in html_str
        assert "&lt;script&gt;" in html_str

    def test_xss_in_model_sub_id(self):
        sub = ModelSubstitution(
            at_step="step:1",
            new_model_id="</td><script>evil()</script>",
        )
        result = MinimizationResult(minimal=[sub], removed=[], probes=1, cache_hits=0)
        html_str = render_html_minimize_report(result)
        assert "<script>" not in html_str


class TestDeterminism:
    def test_same_result_same_html(self):
        result = _simple_result()
        h1 = render_html_minimize_report(result)
        h2 = render_html_minimize_report(result)
        assert h1 == h2

    def test_different_results_different_html(self):
        r1 = _simple_result(minimal_count=2)
        r2 = _simple_result(minimal_count=3)
        h1 = render_html_minimize_report(r1)
        h2 = render_html_minimize_report(r2)
        assert h1 != h2


class TestOptionFlags:
    def test_hide_before_after(self):
        opts = MinimizeReportOptions(show_before_after=False)
        html_str = render_html_minimize_report(_simple_result(), options=opts)
        assert 'id="before-after"' not in html_str

    def test_hide_minimal_subs(self):
        opts = MinimizeReportOptions(show_minimal_substitutions=False)
        html_str = render_html_minimize_report(_simple_result(), options=opts)
        assert 'id="minimal-subs"' not in html_str

    def test_hide_removed_subs(self):
        opts = MinimizeReportOptions(show_removed_substitutions=False)
        html_str = render_html_minimize_report(_simple_result(), options=opts)
        assert 'id="removed-subs"' not in html_str

    def test_hide_probe_stats(self):
        opts = MinimizeReportOptions(show_probe_stats=False)
        html_str = render_html_minimize_report(_simple_result(), options=opts)
        assert 'id="probe-stats"' not in html_str

    def test_hide_attribution(self):
        opts = MinimizeReportOptions(show_attribution=False)
        html_str = render_html_minimize_report(_simple_result(), options=opts)
        assert 'id="attribution"' not in html_str

    def test_hide_final_result(self):
        opts = MinimizeReportOptions(show_final_result=False)
        result = _simple_result(with_final=True)
        html_str = render_html_minimize_report(result, options=opts)
        assert 'id="final-result"' not in html_str

    def test_no_inline_css(self):
        opts = MinimizeReportOptions(html_inline_css=False)
        html_str = render_html_minimize_report(_simple_result(), options=opts)
        assert "<style>" not in html_str


class TestMultiObjective:
    def test_pareto_section_present(self):
        """MultiObjectiveMinimizationResult renders a Pareto-front section."""
        from stepback.minimize import ParetoEntry, TraceObjectives

        minimal = [_make_prompt_sub()]
        removed = [_make_tool_sub()]
        obj = TraceObjectives(
            step_count=5, llm_call_count=2, total_cost_usd=0.01,
            policy_violation_count=0, latency_s=0.5,
        )
        pareto = [
            ParetoEntry(minimal=minimal, objectives=obj, final_result=None),
        ]
        result = MultiObjectiveMinimizationResult(
            minimal=minimal,
            removed=removed,
            probes=5,
            cache_hits=3,
            strategy_name="multi_objective_ddmin",
            pareto_front=pareto,
        )
        html_str = render_html_minimize_report(result)
        assert 'id="pareto-front"' in html_str

    def test_pareto_section_absent_for_plain_result(self):
        result = _simple_result()
        html_str = render_html_minimize_report(result)
        assert 'id="pareto-front"' not in html_str

    def test_hide_pareto_via_option(self):
        from stepback.minimize import ParetoEntry, TraceObjectives

        obj = TraceObjectives(
            step_count=3, llm_call_count=1, total_cost_usd=0.01,
            policy_violation_count=0, latency_s=0.1,
        )
        result = MultiObjectiveMinimizationResult(
            minimal=[_make_prompt_sub()],
            removed=[],
            probes=2,
            cache_hits=1,
            strategy_name="multi_objective_ddmin",
            pareto_front=[ParetoEntry(minimal=[], objectives=obj, final_result=None)],
        )
        opts = MinimizeReportOptions(show_pareto_front=False)
        html_str = render_html_minimize_report(result, options=opts)
        assert 'id="pareto-front"' not in html_str


class TestDefaultOptions:
    def test_default_title(self):
        opts = MinimizeReportOptions()
        assert opts.title == "stepback minimization report"

    def test_all_sections_on_by_default(self):
        opts = MinimizeReportOptions()
        assert opts.show_before_after is True
        assert opts.show_minimal_substitutions is True
        assert opts.show_removed_substitutions is True
        assert opts.show_probe_stats is True
        assert opts.show_attribution is True
        assert opts.show_final_result is True
        assert opts.show_pareto_front is True
        assert opts.html_inline_css is True

    def test_extra_metadata_default_empty(self):
        opts = MinimizeReportOptions()
        assert opts.extra_metadata == {}


class TestPublicAPI:
    def test_minimize_report_options_exported(self):
        assert hasattr(stepback, "MinimizeReportOptions")

    def test_render_html_minimize_report_exported(self):
        assert hasattr(stepback, "render_html_minimize_report")
        assert callable(stepback.render_html_minimize_report)

    def test_in_all(self):
        assert "MinimizeReportOptions" in stepback.__all__
        assert "render_html_minimize_report" in stepback.__all__
