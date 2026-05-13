"""Regression test for Step 51 in ``100_STEPS.md``.

The audit found the old README claimed ``.sb`` v1 frames were CBOR. They
are not — v1 frames are canonical UTF-8 JSON. CBOR is a candidate
encoding for ``format_version=2``. This test pins the corrected claim
into every user-facing document so the regression cannot recur silently.

The test is deliberately defensive: it checks both (a) that the docs
*positively* describe v1 as canonical JSON / v2 as the CBOR candidate
and (b) that no document contains a substring asserting the inverse.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Forbidden phrasings — anything matching these regexes (case-insensitive)
# would re-introduce the audit-flagged claim that v1 is CBOR.
# ---------------------------------------------------------------------------
FORBIDDEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"v1\s+frames?\s+are\s+CBOR", re.IGNORECASE),
    re.compile(r"v1\s+is\s+CBOR", re.IGNORECASE),
    re.compile(r"v1\s+uses\s+CBOR", re.IGNORECASE),
    re.compile(r"format[_\s]version\s*=?\s*1[^\n]{0,40}\bCBOR\b", re.IGNORECASE),
    re.compile(r"\bCBOR[-\s]framed\s+v1\b", re.IGNORECASE),
)


# ---------------------------------------------------------------------------
# Documents that must explicitly describe v1 = canonical JSON and CBOR =
# candidate v2. Each entry maps a relative path to a list of regex patterns
# that MUST appear in the document.
# ---------------------------------------------------------------------------
REQUIRED_CLAIMS: dict[str, tuple[re.Pattern[str], ...]] = {
    "README.md": (
        re.compile(r"v1[^\n]{0,200}canonical\s+JSON", re.IGNORECASE | re.DOTALL),
        re.compile(
            r"CBOR[^\n]{0,200}(candidate|v2|format_version\s*=\s*2)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    "stepback/_resources/README.md": (
        re.compile(r"v1[^\n]{0,200}canonical\s+JSON", re.IGNORECASE | re.DOTALL),
    ),
    "spec/sbtrace-v1.md": (
        re.compile(r"v1\s+frames?\s+are\s+\*?\*?canonical\s+UTF-8\s+JSON", re.IGNORECASE),
        re.compile(r"CBOR\s+is\s+a\s+candidate\s+v2\s+encoding", re.IGNORECASE),
    ),
    "spec/sbtrace-v2.md": (
        re.compile(r"deterministic\s+CBOR", re.IGNORECASE),
    ),
    "docs/canonicalization.md": (
        re.compile(r"v2\s+candidate[^\n]{0,80}CBOR|CBOR[^\n]{0,80}v2\s+candidate|v2\s+\(CBOR\)", re.IGNORECASE),
    ),
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "rel_path,patterns",
    sorted(REQUIRED_CLAIMS.items()),
    ids=sorted(REQUIRED_CLAIMS.keys()),
)
def test_doc_states_v1_is_canonical_json(rel_path: str, patterns: tuple[re.Pattern[str], ...]) -> None:
    """Each tracked doc must positively assert v1 = canonical JSON."""
    path = REPO_ROOT / rel_path
    assert path.is_file(), f"missing tracked doc: {rel_path}"
    text = _read(path)
    for pat in patterns:
        assert pat.search(text), (
            f"{rel_path} does not contain required claim "
            f"{pat.pattern!r}; if you changed the wording you also need to "
            "update tests/test_doc_encoding_claims.py."
        )


@pytest.mark.parametrize(
    "rel_path",
    [
        "README.md",
        "stepback/_resources/README.md",
        "spec/sbtrace-v1.md",
        "spec/sbtrace-v2.md",
        "docs/canonicalization.md",
        "spec/schema/README.md",
        "CHANGELOG.md",
        "GROUNDING.md",
    ],
)
def test_doc_does_not_claim_v1_is_cbor(rel_path: str) -> None:
    """Forbidden phrasings (regression guard for Step 51)."""
    path = REPO_ROOT / rel_path
    if not path.is_file():
        pytest.skip(f"{rel_path} not present in this checkout")
    text = _read(path)
    for pat in FORBIDDEN_PATTERNS:
        match = pat.search(text)
        assert match is None, (
            f"{rel_path} contains forbidden phrasing matching "
            f"{pat.pattern!r}: {match.group(0)!r}. v1 frames are canonical "
            "UTF-8 JSON; CBOR is a candidate v2 encoding (see Step 51 in "
            "100_STEPS.md and §1.2 of spec/sbtrace-v1.md)."
        )


def test_v1_spec_lists_cbor_in_non_goals() -> None:
    """v1 spec §1.2 must explicitly call out CBOR as a non-goal for v1."""
    text = _read(REPO_ROOT / "spec/sbtrace-v1.md")
    non_goals_idx = text.lower().find("non-goals")
    assert non_goals_idx >= 0, "spec/sbtrace-v1.md is missing the Non-goals section"
    # CBOR mention must appear after the Non-goals heading and within a
    # reasonable window so we know it's part of that section.
    window = text[non_goals_idx : non_goals_idx + 2000]
    assert "CBOR" in window, (
        "spec/sbtrace-v1.md §1.2 must list CBOR (and binary frame encodings) "
        "as a v1 non-goal so independent implementers cannot mistake v1 for a "
        "CBOR format."
    )


def test_format_version_table_assigns_cbor_to_two() -> None:
    """``stepback.spec`` MUST advertise CBOR only at format_version >= 2."""
    from stepback import spec as spec_mod

    encodings = getattr(spec_mod, "SBTRACE_WIRE_ENCODINGS", None)
    assert encodings is not None, "stepback.spec.SBTRACE_WIRE_ENCODINGS missing"
    # v1 must be canonical JSON only.
    assert encodings.get(1) == ("canonical-json",) or encodings.get(1) == "canonical-json", (
        "format_version=1 must advertise canonical-json as its sole wire "
        f"encoding; got {encodings.get(1)!r}"
    )
    # v2 (if listed) must include deterministic CBOR among its encodings.
    v2 = encodings.get(2)
    if v2 is not None:
        if isinstance(v2, str):
            v2_tuple: tuple[str, ...] = (v2,)
        else:
            v2_tuple = tuple(v2)
        assert "deterministic-cbor" in v2_tuple, (
            "format_version=2 must list 'deterministic-cbor' among its wire "
            f"encodings; got {v2_tuple!r}"
        )
