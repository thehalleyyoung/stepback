"""Documentation invariants for ``docs/neurips-datasets-paper.md`` — Step 126.

Asserts that the NeurIPS Datasets & Benchmarks paper artifact:

* exists and is non-trivial in size
* references Step 126
* covers all required sections (abstract, motivation, corpora docs,
  licensing, metrics, limitations, reproduction instructions)
* documents all three author-original corpora by name
* documents all six external corpus loaders by name
* states that author-original corpora are Apache-2.0 licensed
* cites the external corpora (SWE-bench, GAIA, tau-bench, AgentBench,
  OSWorld, WebArena)
* includes reproduction commands for the main benchmarks
* includes the result schema description
* acknowledges limitations (synthetic LLM, linear topology, etc.)
* cross-references the relevant source modules and docs
* includes the .sb wire-format appendix
* is consistent with the actual bundled corpora (task counts, step counts)
"""
from __future__ import annotations

import json
import pathlib

DOCS_DIR = pathlib.Path(__file__).parent.parent / "docs"
PAPER_DOC = DOCS_DIR / "neurips-datasets-paper.md"
CORPORA_ROOT = pathlib.Path(__file__).parent.parent / "stepback" / "bench" / "corpora"


def _text() -> str:
    return PAPER_DOC.read_text(encoding="utf-8")


# -------------------------------------------------------------------------
# 1. File existence and minimum size
# -------------------------------------------------------------------------


def test_neurips_paper_exists():
    """``docs/neurips-datasets-paper.md`` must exist."""
    assert PAPER_DOC.exists(), f"missing: {PAPER_DOC}"


def test_neurips_paper_is_substantial():
    """The paper artifact must have substantial content (> 15 KiB)."""
    size = PAPER_DOC.stat().st_size
    assert size > 15_000, (
        f"docs/neurips-datasets-paper.md is suspiciously small ({size} bytes); "
        f"expected > 15 KiB"
    )


# -------------------------------------------------------------------------
# 2. Step 126 discharge claim
# -------------------------------------------------------------------------


def test_neurips_paper_references_step_126():
    """The document must claim to discharge Step 126."""
    text = _text()
    assert "Step 126" in text, (
        "docs/neurips-datasets-paper.md must reference Step 126 of 100_STEPS.md"
    )


# -------------------------------------------------------------------------
# 3. Required top-level sections
# -------------------------------------------------------------------------


def test_neurips_paper_has_abstract():
    text = _text()
    assert "## Abstract" in text or "## 0. Abstract" in text or "Abstract" in text[:500]


def test_neurips_paper_has_motivation_section():
    text = _text()
    assert "Motivation" in text or "motivation" in text


def test_neurips_paper_has_corpora_section():
    text = _text()
    assert "Corpora" in text or "corpora" in text


def test_neurips_paper_has_licensing_section():
    text = _text()
    assert "Licens" in text  # "License" or "Licensing"


def test_neurips_paper_has_metrics_section():
    text = _text()
    assert "Metrics" in text or "metrics" in text


def test_neurips_paper_has_limitations_section():
    text = _text()
    assert "Limitation" in text  # "Limitations"


def test_neurips_paper_has_reproduction_section():
    text = _text()
    assert "Reproduction" in text or "reproduction" in text


def test_neurips_paper_has_references():
    text = _text()
    assert "## References" in text


# -------------------------------------------------------------------------
# 4. Author-original corpus documentation
# -------------------------------------------------------------------------


def test_neurips_paper_mentions_support_agent():
    assert "support-agent" in _text()


def test_neurips_paper_mentions_code_review():
    assert "code-review" in _text()


def test_neurips_paper_mentions_payments_policy():
    assert "payments-policy" in _text()


def test_neurips_paper_states_15_tasks():
    """Paper must document that there are 15 bundled tasks total."""
    text = _text()
    assert "15" in text and "task" in text.lower()


def test_neurips_paper_mentions_apache_license():
    """Author-original corpora are Apache-2.0."""
    assert "Apache-2.0" in _text()


# -------------------------------------------------------------------------
# 5. External corpus loader documentation
# -------------------------------------------------------------------------


def test_neurips_paper_mentions_swe_bench():
    assert "SWE-bench" in _text() or "SWEBench" in _text()


def test_neurips_paper_mentions_gaia():
    assert "GAIA" in _text()


def test_neurips_paper_mentions_tau_bench():
    assert "tau-bench" in _text() or "TauBench" in _text()


def test_neurips_paper_mentions_agentbench():
    assert "AgentBench" in _text()


def test_neurips_paper_mentions_osworld():
    assert "OSWorld" in _text()


def test_neurips_paper_mentions_webarena():
    assert "WebArena" in _text()


def test_neurips_paper_mentions_six_external_corpora():
    """All six external loaders must be mentioned."""
    text = _text()
    missing = [
        name for name in ["SWE-bench", "GAIA", "tau-bench", "AgentBench", "OSWorld", "WebArena"]
        if name not in text
    ]
    assert not missing, f"Missing external corpus references: {missing}"


# -------------------------------------------------------------------------
# 6. Metrics content
# -------------------------------------------------------------------------


def test_neurips_paper_mentions_cost_reduction_factor():
    assert "cost_reduction_factor" in _text() or "cost reduction factor" in _text().lower()


def test_neurips_paper_mentions_dirty_set():
    assert "dirty_set" in _text() or "dirty-set" in _text() or "dirty set" in _text()


def test_neurips_paper_mentions_cache_hit_rate():
    text = _text()
    assert "cache hit" in text.lower() or "cache_hit" in text or "hit rate" in text.lower()


def test_neurips_paper_mentions_recorder_overhead():
    text = _text()
    assert "overhead" in text.lower() and ("µs" in text or "microsecond" in text.lower() or "46" in text)


def test_neurips_paper_includes_benchmark_tables():
    """Paper must include numeric benchmark results (markdown tables)."""
    text = _text()
    # Look for markdown table rows with numbers
    assert "|" in text and "×" in text, "Paper should include benchmark result tables with × speedup notation"


# -------------------------------------------------------------------------
# 7. Limitations documented
# -------------------------------------------------------------------------


def test_neurips_paper_mentions_synthetic_llm_limitation():
    text = _text()
    assert ("synthetic" in text.lower() and "llm" in text.lower()) or "fake LLM" in text


def test_neurips_paper_mentions_linear_topology_limitation():
    text = _text()
    assert "linear" in text.lower() and ("topolog" in text.lower() or "branch" in text.lower())


def test_neurips_paper_mentions_small_corpus_limitation():
    text = _text()
    assert "15 task" in text or "only 15" in text or "small corpus" in text.lower()


# -------------------------------------------------------------------------
# 8. Reproduction commands
# -------------------------------------------------------------------------


def test_neurips_paper_has_replay_caching_command():
    text = _text()
    assert "stepback bench replay-caching" in text


def test_neurips_paper_has_record_overhead_command():
    text = _text()
    assert "stepback bench record-overhead" in text


def test_neurips_paper_has_pytest_command():
    text = _text()
    assert "pytest" in text


def test_neurips_paper_mentions_generate_author_corpora_script():
    text = _text()
    assert "generate_author_corpora" in text


# -------------------------------------------------------------------------
# 9. Cross-references to codebase
# -------------------------------------------------------------------------


def test_neurips_paper_references_author_corpora_module():
    text = _text()
    assert "author_corpora" in text


def test_neurips_paper_references_corpus_loaders_module():
    text = _text()
    assert "corpus_loaders" in text


def test_neurips_paper_references_result_schema():
    text = _text()
    assert "result_schema" in text or "BenchRunRecord" in text


def test_neurips_paper_references_submission_rules():
    text = _text()
    assert "submission" in text.lower()


def test_neurips_paper_references_dirty_set_doc():
    text = _text()
    assert "dirty-set.md" in text


# -------------------------------------------------------------------------
# 10. Wire-format appendix
# -------------------------------------------------------------------------


def test_neurips_paper_has_wire_format_section():
    text = _text()
    assert "wire" in text.lower() or "Wire Format" in text or ".sb" in text


def test_neurips_paper_mentions_frame_types():
    text = _text()
    for frame_type in ["header", "blob", "step", "merkle_summary", "tail"]:
        assert frame_type in text, f"Wire format appendix should mention frame type '{frame_type}'"


def test_neurips_paper_mentions_hmac_chaining():
    text = _text()
    assert "HMAC" in text or "hmac" in text


def test_neurips_paper_mentions_ed25519():
    text = _text()
    assert "Ed25519" in text


# -------------------------------------------------------------------------
# 11. Consistency with actual bundled corpora
# -------------------------------------------------------------------------


def test_neurips_paper_corpus_task_counts_consistent():
    """Paper's stated task counts (5 per corpus) must match manifest.json files."""
    for corpus_name in ["support-agent", "code-review", "payments-policy"]:
        manifest_path = CORPORA_ROOT / corpus_name / "manifest.json"
        if not manifest_path.exists():
            continue  # Skip if corpus not generated yet
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_count = len(manifest["tasks"])
        assert actual_count == 5, (
            f"Corpus {corpus_name} has {actual_count} tasks in manifest, "
            f"but paper expects 5"
        )


def test_neurips_paper_corpus_license_consistent():
    """All corpora must be Apache-2.0 as stated in the paper."""
    for corpus_name in ["support-agent", "code-review", "payments-policy"]:
        manifest_path = CORPORA_ROOT / corpus_name / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest.get("license") == "Apache-2.0", (
            f"Corpus {corpus_name} manifest has license={manifest.get('license')!r}, "
            f"expected 'Apache-2.0'"
        )


def test_neurips_paper_bundled_sb_files_exist():
    """All 15 .sb trace files referenced in the paper must exist."""
    expected = {
        "support-agent": ["order-status-delayed", "refund-eligible", "account-unlock",
                          "wrong-item", "subscription-cancel"],
        "code-review": ["sql-injection", "off-by-one", "race-condition",
                        "missing-error-handling", "resource-leak"],
        "payments-policy": ["normal-payment", "overlimit-payment", "sanctioned-country",
                            "fraud-velocity", "unverified-beneficiary"],
    }
    for corpus_name, task_ids in expected.items():
        for task_id in task_ids:
            sb_path = CORPORA_ROOT / corpus_name / f"{task_id}.sb"
            assert sb_path.exists(), f"Missing bundled trace: {sb_path}"


def test_neurips_paper_sb_files_nonempty():
    """All bundled .sb files must be non-empty."""
    for corpus_name in ["support-agent", "code-review", "payments-policy"]:
        corpus_dir = CORPORA_ROOT / corpus_name
        if not corpus_dir.exists():
            continue
        for sb_file in corpus_dir.glob("*.sb"):
            size = sb_file.stat().st_size
            assert size > 0, f"Empty .sb file: {sb_file}"
            assert size > 1_000, (
                f".sb file suspiciously small ({size} bytes): {sb_file}"
            )


# -------------------------------------------------------------------------
# 12. Citation section
# -------------------------------------------------------------------------


def test_neurips_paper_has_citation_bibtex():
    text = _text()
    assert "@misc" in text or "@article" in text or "@inproceedings" in text, (
        "Paper should include a BibTeX citation entry"
    )


def test_neurips_paper_mentions_citation_cff():
    text = _text()
    assert "CITATION.cff" in text
