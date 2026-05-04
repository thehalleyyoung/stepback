"""Cross-language differential canonical-JSON tests (Step 48).

Each language arm is independently skippable so this test file stays
green on a fresh checkout where the Rust workspace hasn't been built or
the TypeScript package hasn't been installed. CI builds both; running
locally with `pytest` skips the missing arms with a visible reason.
"""
from __future__ import annotations

import pytest

from spec.canonical.bounded import enumerate_bounded, reference_canonical_json
from spec.canonical.differential import (
    canonicalize_batch_with_rust,
    canonicalize_batch_with_ts,
    detect_rust_binary,
    detect_ts_binary,
    run_differential,
)
from stepback.canonical import canonical_json


def test_python_arm_matches_reference_in_batch_form() -> None:
    """Python is always in the report and must always pass."""
    report = run_differential(rust=False, typescript=False)
    assert "python" in report.languages
    assert not report.failures, report.failures


@pytest.mark.skipif(
    detect_rust_binary() is None,
    reason="Rust `canonicalize` binary not built (see "
    "stepback-core/crates/sb-canonical/src/bin/canonicalize.rs)",
)
def test_rust_canonicaliser_matches_reference() -> None:
    binary = detect_rust_binary()
    assert binary is not None
    corpus = list(enumerate_bounded())
    outputs = canonicalize_batch_with_rust(corpus, binary=binary)
    assert len(outputs) == len(corpus)
    for v, got in zip(corpus, outputs):
        expected = reference_canonical_json(v)
        assert got == expected, (
            f"Rust canonicalise diverges from reference\n"
            f"input={v!r}\nexpected={expected!r}\nactual  ={got!r}"
        )


@pytest.mark.skipif(
    detect_ts_binary() is None,
    reason="TypeScript canonicalize script unavailable; build with "
    "`npm --prefix bindings/typescript install && npm --prefix "
    "bindings/typescript run build`",
)
def test_typescript_canonicaliser_matches_reference() -> None:
    script = detect_ts_binary()
    assert script is not None
    corpus = list(enumerate_bounded())
    outputs = canonicalize_batch_with_ts(corpus, script=script)
    assert len(outputs) == len(corpus)
    for v, got in zip(corpus, outputs):
        expected = reference_canonical_json(v)
        assert got == expected, (
            f"TypeScript canonicalise diverges from reference\n"
            f"input={v!r}\nexpected={expected!r}\nactual  ={got!r}"
        )


def test_diff_report_documents_reader_only_languages() -> None:
    """Go, JVM, .NET ship as read-only verifiers in v0.1; the report
    should self-document this rather than silently dropping them."""
    report = run_differential(rust=False, typescript=False)
    for lang in ("go", "jvm", "dotnet"):
        assert lang in report.skipped, report.skipped
        assert "reader-only" in report.skipped[lang]
