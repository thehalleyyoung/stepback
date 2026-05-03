"""Trace redaction for safe sharing.

A recorded `.sb` file is a verbatim capture of an agent's production
input/output — it routinely contains PII (customer names, emails,
phone numbers, addresses), credentials (AWS access keys, JWT tokens,
API tokens), and financial identifiers (credit-card numbers, IBANs,
SSNs). Sharing such a trace with a vendor, regulator, or external
reviewer requires scrubbing those identifiers without destroying the
trace's structural usefulness for replay and bisection.

This module provides:

* :class:`Detector` — a regex (or callable) + replacement-strategy
  bundle that finds one class of sensitive token.
* :class:`RedactionPolicy` — an ordered collection of detectors with
  a name and a salt. Two builtin policies ship: :data:`STANDARD_POLICY`
  (emails, phones, IBANs, SSNs, credit cards, AWS keys, JWT, bearer
  tokens, IPs) and :data:`STRICT_POLICY` (everything in standard plus
  capitalised proper-name heuristics + numeric currency amounts).
* :func:`redact_value` / :func:`redact_step` / :func:`redact_steps` —
  pure functions that walk an arbitrary JSON-shaped value (or a
  recorded step's ``inputs`` / ``outputs`` / ``llm_request`` /
  ``llm_response``) and return a redacted copy + a manifest of what
  was matched.
* :func:`redact_trace_file` — read a `.sb` file via
  :func:`stepback.trace_reader.verify_trace`, redact every step,
  write a fresh `.sb` file with a new HMAC + Ed25519 chain, and
  return a :class:`RedactionManifest` summarising the work.

Replacement strategies:

* ``"hash"`` — replace the matched substring with a stable
  ``<REDACTED:<kind>:<8 hex chars>>`` token derived from
  ``HMAC_SHA256(policy.salt, kind || matched_text)``. Stable means:
  the same input substring under the same policy always yields the
  same token, so the cache structure (whether two LLM calls receive
  the same chat history) is preserved across redaction.
* ``"mask"`` — replace with a fixed sentinel like ``"<REDACTED:email>"``.
* ``"drop"`` — replace with the empty string.

The module is intentionally pure-Python and dependency-free beyond
``stepback`` itself; the goal is that a redacted trace can be
produced inside an air-gapped CI step before any data leaves the
production network.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Pattern, Tuple, Union

from .canonical import canonical_json, hash_obj, sha256_hex
from .recorder import RecorderKey
from .trace_reader import verify_trace
from .trace_writer import TraceWriter


REPLACEMENT_STRATEGIES = ("hash", "mask", "drop")
PROTECTED_KEYS = frozenset(
    {
        # structural fields the replay engine needs verbatim
        "step_id",
        "step_kind",
        "parent_step_id",
        "kind",
        "name",
        "id",
        "role",
        "tool_call_id",
        "finish_reason",
        "index",
        "type",
        "model",
        "wallclock_ns",
        "cpu_ns",
        # hash fields (will be recomputed)
        "inputs_hash",
        "outputs_hash",
        "nondeterminism_hash",
        # numeric usage / cost
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost_usd",
        "temperature",
        "top_p",
        "seed",
    }
)


# ---------------------------------------------------------------- detectors


# regex sources are deliberately conservative: false positives on a
# trace shared with a regulator are tolerable, false negatives (real
# PII leaking out) are not.

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
# IBAN: 2 country letters + 2 check digits + up to 30 alnum, possibly
# split by spaces or hyphens.
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:[ -]?[A-Z0-9]{1,4}){3,8}\b")
# US-style SSN
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Credit card numbers (Luhn-validated below; this pattern just narrows)
CC_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
# US 10-digit phone with optional country code
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})[ -]?\d{3}[ -]?\d{4}(?!\d)"
)
# IPv4
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# AWS access key
AWS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")
# JWT: three base64url chunks separated by dots
JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
# Bearer / API token style: "sk-" or "Bearer ABC..."
BEARER_RE = re.compile(r"\b(?:sk-|pk-|rk-)[A-Za-z0-9]{20,}\b")
# Capitalised proper-name heuristic: two consecutive Title-case words.
# Used only by STRICT_POLICY because it has many false positives.
PROPER_NAME_RE = re.compile(r"\b[A-Z][a-z]{1,20}(?: [A-Z][a-z]{1,20}){1,3}\b")
# Currency amounts like $50000 or €1,234.56. Strict only.
CURRENCY_RE = re.compile(r"[$€£¥]\s?\d[\d,]*(?:\.\d{2})?")

# ---- 2026 provider-token regexes (Layer-3 plan, sourced from Proposer 1) ----
# All conservative; positive samples come from each vendor's docs.
GITHUB_PAT_RE = re.compile(r"\bgh[psoru]_[A-Za-z0-9]{36,251}\b")
OPENAI_KEY_RE = re.compile(
    r"\bsk-(?:proj-|svcacct-|admin-|None-|)[A-Za-z0-9_\-]{20,}\b"
)
ANTHROPIC_KEY_RE = re.compile(
    r"\bsk-ant-(?:api|admin)\d{2}-[A-Za-z0-9_\-]{20,}\b"
)
GOOGLE_API_KEY_RE = re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")
SLACK_TOKEN_RE = re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")
STRIPE_KEY_RE = re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{20,}\b")
STRIPE_WEBHOOK_RE = re.compile(r"\bwhsec_[A-Za-z0-9]{20,}\b")
TWILIO_SID_RE = re.compile(r"\bAC[0-9a-f]{32}\b")
AZURE_SAS_RE = re.compile(r"sig=[A-Za-z0-9%]+&se=[^&\s\"']+")
MAC_ADDR_RE = re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b")
BTC_ADDR_RE = re.compile(r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b")
ETH_ADDR_RE = re.compile(r"\b0x[a-fA-F0-9]{40}\b")

# PEM blocks span multiple lines so a callable spans-finder is used
# instead of a regex.
_PEM_BEGIN_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED |)PRIVATE KEY(?: BLOCK)?-----"
)
_PEM_END_RE = re.compile(
    r"-----END (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED |)PRIVATE KEY(?: BLOCK)?-----"
)


def _pem_block_finder(s: str) -> List[Tuple[int, int]]:
    """Return spans of full ``-----BEGIN ... PRIVATE KEY ... -----END ...-----`` blocks."""
    out: List[Tuple[int, int]] = []
    cursor = 0
    while True:
        m_begin = _PEM_BEGIN_RE.search(s, cursor)
        if not m_begin:
            return out
        m_end = _PEM_END_RE.search(s, m_begin.end())
        if not m_end:
            return out
        out.append((m_begin.start(), m_end.end()))
        cursor = m_end.end()


def _btc_predicate(matched: str) -> bool:
    # Reject obvious commit-SHA-shaped (lowercase hex only) strings
    # that the BTC regex over-greedily catches.
    if matched.islower() and all(c in "0123456789abcdef" for c in matched):
        return False
    return True



def _luhn_ok(digits: str) -> bool:
    nums = [int(c) for c in digits if c.isdigit()]
    if not (13 <= len(nums) <= 19):
        return False
    total = 0
    for i, d in enumerate(reversed(nums)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _cc_match(s: str) -> bool:
    return _luhn_ok(s)


@dataclass
class Detector:
    """One named detector: pattern + replacement strategy.

    ``pattern`` is either a compiled regex (whose match's full ``group(0)``
    is the redacted span) or a callable ``str -> Iterable[(start, end)]``
    returning explicit spans.

    ``predicate`` is an optional ``str -> bool`` post-filter applied to
    the matched substring; useful for Luhn-validating credit cards.

    ``strategy`` is one of ``"hash"``, ``"mask"``, ``"drop"``.
    """

    name: str
    pattern: Union[Pattern[str], Callable[[str], List[Tuple[int, int]]]]
    strategy: str = "hash"
    predicate: Optional[Callable[[str], bool]] = None

    def __post_init__(self) -> None:
        if self.strategy not in REPLACEMENT_STRATEGIES:
            raise ValueError(
                f"unknown strategy {self.strategy!r}; must be one of {REPLACEMENT_STRATEGIES}"
            )

    def find(self, s: str) -> List[Tuple[int, int, str]]:
        """Return ``[(start, end, matched_text)]`` for every hit in ``s``."""
        spans: List[Tuple[int, int, str]] = []
        if hasattr(self.pattern, "finditer"):
            for m in self.pattern.finditer(s):
                txt = m.group(0)
                if self.predicate is not None and not self.predicate(txt):
                    continue
                spans.append((m.start(), m.end(), txt))
        else:
            for start, end in self.pattern(s):  # type: ignore[operator]
                txt = s[start:end]
                if self.predicate is not None and not self.predicate(txt):
                    continue
                spans.append((start, end, txt))
        return spans


@dataclass
class RedactionPolicy:
    """Ordered collection of detectors + a salt for stable hashing.

    Detectors are applied in declaration order. Spans claimed by an
    earlier detector are not re-examined by later ones, so the
    earliest-declared detector wins on overlap.

    The ``salt`` is mixed into every ``"hash"`` strategy replacement
    via HMAC-SHA256, so two parties can share traces redacted under
    different salts without any leakage of the underlying values.
    A fresh salt is generated by :meth:`fresh` if none is supplied.

    ``allowlist`` is a frozenset of exact substrings that, when matched
    by *any* detector, bypass redaction (useful for service emails,
    public IPs, the test user "alice@example.com", etc.).
    ``allowlist_patterns`` is the same idea keyed by regex.
    """

    name: str
    detectors: List[Detector] = field(default_factory=list)
    salt: bytes = b""
    mask_template: str = "<REDACTED:{kind}>"
    hash_template: str = "<REDACTED:{kind}:{token}>"
    allowlist: FrozenSet[str] = field(default_factory=frozenset)
    allowlist_patterns: Tuple[Pattern[str], ...] = field(default_factory=tuple)

    @classmethod
    def fresh(cls, name: str, detectors: List[Detector]) -> "RedactionPolicy":
        return cls(name=name, detectors=list(detectors), salt=os.urandom(16))

    def stable_token(self, kind: str, matched: str) -> str:
        h = hmac.new(self.salt or b"stepback-default-salt", (kind + ":" + matched).encode("utf-8"), hashlib.sha256)
        return h.hexdigest()[:8]

    def render_replacement(self, detector: Detector, matched: str) -> str:
        if detector.strategy == "drop":
            return ""
        if detector.strategy == "mask":
            return self.mask_template.format(kind=detector.name)
        token = self.stable_token(detector.name, matched)
        return self.hash_template.format(kind=detector.name, token=token)

    def is_allowlisted(self, matched: str) -> bool:
        if matched in self.allowlist:
            return True
        for pat in self.allowlist_patterns:
            if pat.fullmatch(matched):
                return True
        return False


# Sensitive-key names (case-insensitive) used by KeyContextDetector.
_SENSITIVE_KEY_RE = re.compile(
    r"\b(authorization|api[_-]?key|password|passwd|secret|token|"
    r"access[_-]?token|refresh[_-]?token|private[_-]?key|"
    r"client[_-]?secret|session[_-]?id|cookie)\b",
    re.IGNORECASE,
)
# After the key name, allow ``:``, ``=``, ``: Bearer``, optional
# whitespace and quotes, then capture until a delimiter.
_KEY_VALUE_PAIR_RE = re.compile(
    r"\b(authorization|api[_-]?key|password|passwd|secret|token|"
    r"access[_-]?token|refresh[_-]?token|private[_-]?key|"
    r"client[_-]?secret|session[_-]?id|cookie)\b"
    r"\s*[:=]\s*"
    r"(?:Bearer\s+|Basic\s+|Token\s+)?"
    r"['\"]?"
    r"(?P<val>[^\s,'\"\n&}\)]{4,})",
    re.IGNORECASE,
)


class KeyContextDetector(Detector):
    """Detector that extracts the *value* after a sensitive key name.

    Catches secrets that no prefix-based regex would match, e.g.::

        Authorization: Bearer abc123def456
        api_key=hunter2hunter2hunter2

    Only the value substring is redacted; the key name and delimiter
    are preserved so the trace remains parseable.
    """

    def __init__(self, name: str = "key_context", strategy: str = "mask") -> None:
        super().__init__(name=name, pattern=_SENSITIVE_KEY_RE, strategy=strategy)

    def find(self, s: str) -> List[Tuple[int, int, str]]:
        spans: List[Tuple[int, int, str]] = []
        for m in _KEY_VALUE_PAIR_RE.finditer(s):
            val = m.group("val")
            # Skip obvious non-secrets: short numeric-only IDs.
            if val.isdigit() and len(val) < 12:
                continue
            start = m.start("val")
            end = m.end("val")
            spans.append((start, end, val))
        return spans



def _build_standard_detectors() -> List[Detector]:
    return [
        # PEM blocks first: their base64 body would otherwise be picked
        # apart by JWT_RE / BEARER_RE / etc.
        Detector("pem_private_key", _pem_block_finder, strategy="mask"),
        # Provider-prefixed secrets (Layer-3 expansion). All ``mask``
        # because deterministic-hashing a leaf secret has no debugging
        # value.
        Detector("github_pat", GITHUB_PAT_RE, strategy="mask"),
        # Anthropic keys must come before openai_key — the OpenAI
        # regex is intentionally permissive and would otherwise eat
        # `sk-ant-…` strings as legacy `sk-…` tokens.
        Detector("anthropic_key", ANTHROPIC_KEY_RE, strategy="mask"),
        Detector("openai_key", OPENAI_KEY_RE, strategy="mask"),
        Detector("google_api_key", GOOGLE_API_KEY_RE, strategy="mask"),
        Detector("slack_token", SLACK_TOKEN_RE, strategy="mask"),
        Detector("stripe_key", STRIPE_KEY_RE, strategy="mask"),
        Detector("stripe_webhook", STRIPE_WEBHOOK_RE, strategy="mask"),
        Detector("twilio_sid", TWILIO_SID_RE, strategy="mask"),
        Detector("azure_sas", AZURE_SAS_RE, strategy="mask"),
        Detector("eth_address", ETH_ADDR_RE, strategy="hash"),
        Detector("btc_address", BTC_ADDR_RE, strategy="hash", predicate=_btc_predicate),
        Detector("mac_address", MAC_ADDR_RE, strategy="hash"),
        # Original v0.1 detectors retained in original order.
        Detector("email", EMAIL_RE, strategy="hash"),
        Detector("aws_key", AWS_KEY_RE, strategy="mask"),
        Detector("jwt", JWT_RE, strategy="mask"),
        Detector("bearer_token", BEARER_RE, strategy="mask"),
        Detector("ssn", SSN_RE, strategy="hash"),
        Detector("credit_card", CC_RE, strategy="hash", predicate=_cc_match),
        Detector("iban", IBAN_RE, strategy="hash"),
        Detector("phone", PHONE_RE, strategy="hash"),
        Detector("ipv4", IPV4_RE, strategy="hash"),
        # Final fallback: catch values after sensitive key names that
        # no upstream detector recognised.
        KeyContextDetector("key_context", strategy="mask"),
    ]



def _build_strict_detectors() -> List[Detector]:
    base = _build_standard_detectors()
    base.extend(
        [
            Detector("proper_name", PROPER_NAME_RE, strategy="hash"),
            Detector("currency", CURRENCY_RE, strategy="hash"),
        ]
    )
    return base


STANDARD_POLICY = RedactionPolicy(
    name="standard",
    detectors=_build_standard_detectors(),
    salt=b"stepback-standard-default-salt!!",
)

STRICT_POLICY = RedactionPolicy(
    name="strict",
    detectors=_build_strict_detectors(),
    salt=b"stepback-strict-default-salt!!!",
)


# ---------------------------------------------------------------- core


@dataclass
class RedactionManifest:
    """What a redaction pass did, in aggregate.

    ``per_detector`` is ``{detector_name: hit_count}`` summed across
    every string in every step. ``samples`` is up to ``sample_limit``
    pairs of ``(detector_name, redacted_token)`` — the redacted
    tokens, not the raw substrings, so the manifest itself is safe
    to share alongside the redacted trace.

    ``policy_name``, ``salt_id`` (sha256 of the salt, first 16 hex
    chars), and ``n_steps`` round out the audit trail.
    """

    policy_name: str
    salt_id: str
    n_steps: int = 0
    n_redactions: int = 0
    per_detector: Dict[str, int] = field(default_factory=dict)
    samples: List[Tuple[str, str]] = field(default_factory=list)
    sample_limit: int = 32

    def record(self, detector_name: str, replacement: str) -> None:
        self.n_redactions += 1
        self.per_detector[detector_name] = self.per_detector.get(detector_name, 0) + 1
        if len(self.samples) < self.sample_limit:
            self.samples.append((detector_name, replacement))

    def to_dict(self) -> dict:
        return {
            "policy_name": self.policy_name,
            "salt_id": self.salt_id,
            "n_steps": self.n_steps,
            "n_redactions": self.n_redactions,
            "per_detector": dict(self.per_detector),
            "samples": [list(s) for s in self.samples],
        }


def _collect_spans(
    s: str, policy: RedactionPolicy
) -> List[Tuple[int, int, Detector, str]]:
    """Run every detector in declaration order, return claimed spans.

    Honours ``policy.allowlist`` and ``policy.allowlist_patterns``:
    spans whose matched substring is allowlisted are not returned.
    Earlier detectors win on overlap.

    Shared by :func:`redact_string` and the scan-only path so dry-run
    findings always match what an apply-pass would do.
    """
    claimed: List[Tuple[int, int, Detector, str]] = []
    occupied: List[Tuple[int, int]] = []

    def _overlaps(a: int, b: int) -> bool:
        for x, y in occupied:
            if a < y and x < b:
                return True
        return False

    for det in policy.detectors:
        for start, end, txt in det.find(s):
            if _overlaps(start, end):
                continue
            if policy.is_allowlisted(txt):
                continue
            claimed.append((start, end, det, txt))
            occupied.append((start, end))
    claimed.sort(key=lambda t: t[0])
    return claimed


def redact_string(
    s: str, policy: RedactionPolicy, manifest: Optional[RedactionManifest] = None
) -> str:
    """Apply every detector in ``policy`` to ``s`` and return the redacted copy.

    Detectors are applied in declaration order; once a span is
    consumed, later detectors do not see it.
    """
    if not s:
        return s
    claimed = _collect_spans(s, policy)
    if not claimed:
        return s
    out: List[str] = []
    cursor = 0
    for start, end, det, txt in claimed:
        out.append(s[cursor:start])
        repl = policy.render_replacement(det, txt)
        out.append(repl)
        if manifest is not None:
            manifest.record(det.name, repl)
        cursor = end
    out.append(s[cursor:])
    return "".join(out)



def redact_value(
    value: Any,
    policy: RedactionPolicy,
    manifest: Optional[RedactionManifest] = None,
    *,
    in_protected_key: bool = False,
) -> Any:
    """Recursively walk ``value`` and redact every string leaf.

    ``in_protected_key`` skips redaction inside structural fields
    listed in :data:`PROTECTED_KEYS` (step ids, role names, model
    ids — fields the replay engine compares verbatim).
    """
    if isinstance(value, dict):
        return {
            k: redact_value(
                v,
                policy,
                manifest,
                in_protected_key=in_protected_key or k in PROTECTED_KEYS,
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            redact_value(v, policy, manifest, in_protected_key=in_protected_key)
            for v in value
        ]
    if isinstance(value, str):
        if in_protected_key:
            return value
        return redact_string(value, policy, manifest)
    return value


# Hash-bearing fields recomputed after redaction.
_HASH_FIELDS = {
    "inputs": "inputs_hash",
    "outputs": "outputs_hash",
}


def redact_step(
    step: dict, policy: RedactionPolicy, manifest: Optional[RedactionManifest] = None
) -> dict:
    """Redact ``step`` in place-of-a-copy and recompute content hashes.

    The returned step preserves ``step_id`` / ``parent_step_id`` /
    ``step_kind`` exactly so the replay engine's call-tree
    reconstruction still works. ``inputs_hash`` and ``outputs_hash``
    are recomputed from the redacted ``inputs`` / ``outputs`` so the
    redacted trace is self-consistent and verifiable.

    The hashes will *not* match the original trace's hashes (that is
    the point) — they re-establish content-addressing within the
    redacted trace.
    """
    redacted = redact_value(step, policy, manifest)
    # Recompute the canonical content hashes that were derived from
    # the now-mutated inputs/outputs.
    for src_key, hash_key in _HASH_FIELDS.items():
        if src_key in redacted and hash_key in redacted:
            redacted[hash_key] = hash_obj(redacted[src_key])
    return redacted


def redact_steps(
    steps: List[dict], policy: RedactionPolicy
) -> Tuple[List[dict], RedactionManifest]:
    """Bulk variant. Returns ``(redacted_steps, manifest)``."""
    salt_id = hashlib.sha256(policy.salt or b"").hexdigest()[:16]
    manifest = RedactionManifest(policy_name=policy.name, salt_id=salt_id)
    out = [redact_step(s, policy, manifest) for s in steps]
    manifest.n_steps = len(out)
    return out, manifest


# ---------------------------------------------------------------- file I/O


def redact_trace_file(
    in_path: str,
    out_path: str,
    *,
    in_hmac_key: bytes,
    policy: RedactionPolicy,
    out_key: Optional[RecorderKey] = None,
    compression: bool = True,
) -> RedactionManifest:
    """End-to-end: read + verify ``in_path``, redact, write ``out_path``.

    A fresh :class:`RecorderKey` is generated for the output trace if
    ``out_key`` is not supplied (the input HMAC + signing keys must
    not be reused, since the redacted trace is structurally a different
    artifact and has its own chain).

    Returns the :class:`RedactionManifest`. Raises
    :class:`stepback.trace_reader.TraceVerificationError` if ``in_path``
    fails verification — refusing to redact an already-tampered trace
    is the safest default.
    """
    parsed = verify_trace(in_path, in_hmac_key)
    redacted_steps, manifest = redact_steps(parsed.steps, policy)
    out_key = out_key or RecorderKey.fresh()
    writer = TraceWriter.open(
        out_path,
        hmac_key=out_key.hmac_key,
        signing_key=out_key.signing_key,
        compression=compression,
    )
    try:
        for s in redacted_steps:
            writer.write_step(s)
    finally:
        writer.close()
    return manifest


__all__ = [
    "Detector",
    "KeyContextDetector",
    "RedactionPolicy",
    "RedactionManifest",
    "ScanReport",
    "STANDARD_POLICY",
    "STRICT_POLICY",
    "PROTECTED_KEYS",
    "REPLACEMENT_STRATEGIES",
    "redact_string",
    "redact_value",
    "redact_step",
    "redact_steps",
    "redact_trace_file",
    "redact_trace_file_streaming",
    "scan_value",
    "scan_steps",
    "scan_trace_file",
    "EMAIL_RE",
    "IBAN_RE",
    "SSN_RE",
    "CC_RE",
    "PHONE_RE",
    "IPV4_RE",
    "AWS_KEY_RE",
    "JWT_RE",
    "BEARER_RE",
    "GITHUB_PAT_RE",
    "OPENAI_KEY_RE",
    "ANTHROPIC_KEY_RE",
    "GOOGLE_API_KEY_RE",
    "SLACK_TOKEN_RE",
    "STRIPE_KEY_RE",
    "STRIPE_WEBHOOK_RE",
    "TWILIO_SID_RE",
    "AZURE_SAS_RE",
    "MAC_ADDR_RE",
    "BTC_ADDR_RE",
    "ETH_ADDR_RE",
]


# ---------------------------------------------------------------- scan-only


@dataclass
class ScanReport:
    """Dry-run sibling of :class:`RedactionManifest`.

    A scan does NOT mutate steps or write any output; it walks every
    string leaf and records *what would be redacted, where*. Used by
    security review before a trace is shared.

    ``findings_by_step`` maps ``step_id -> [(detector_name, sample_replacement)]``;
    the sample replacement is the same token a real redaction would
    have produced, so reviewers can preview the redacted output
    without any I/O.
    """

    policy_name: str
    salt_id: str
    n_steps: int = 0
    n_findings: int = 0
    per_detector: Dict[str, int] = field(default_factory=dict)
    findings_by_step: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    sample_redactions: List[Tuple[str, str]] = field(default_factory=list)
    sample_limit: int = 32

    def _record(
        self, step_id: str, detector_name: str, replacement: str
    ) -> None:
        self.n_findings += 1
        self.per_detector[detector_name] = self.per_detector.get(detector_name, 0) + 1
        self.findings_by_step.setdefault(step_id, []).append(
            (detector_name, replacement)
        )
        if len(self.sample_redactions) < self.sample_limit:
            self.sample_redactions.append((detector_name, replacement))

    def to_dict(self) -> dict:
        return {
            "policy_name": self.policy_name,
            "salt_id": self.salt_id,
            "n_steps": self.n_steps,
            "n_findings": self.n_findings,
            "per_detector": dict(self.per_detector),
            "findings_by_step": {
                k: [list(t) for t in v] for k, v in self.findings_by_step.items()
            },
            "sample_redactions": [list(s) for s in self.sample_redactions],
        }


def _scan_value(
    value: Any,
    policy: RedactionPolicy,
    step_id: str,
    report: ScanReport,
    *,
    in_protected_key: bool = False,
) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _scan_value(
                v,
                policy,
                step_id,
                report,
                in_protected_key=in_protected_key or k in PROTECTED_KEYS,
            )
        return
    if isinstance(value, list):
        for v in value:
            _scan_value(v, policy, step_id, report, in_protected_key=in_protected_key)
        return
    if isinstance(value, str) and not in_protected_key and value:
        for start, end, det, txt in _collect_spans(value, policy):
            repl = policy.render_replacement(det, txt)
            report._record(step_id, det.name, repl)


def scan_value(value: Any, policy: RedactionPolicy) -> ScanReport:
    """Scan an arbitrary JSON-shaped value; report every prospective redaction."""
    salt_id = hashlib.sha256(policy.salt or b"").hexdigest()[:16]
    report = ScanReport(policy_name=policy.name, salt_id=salt_id)
    _scan_value(value, policy, step_id="(value)", report=report)
    return report


def scan_steps(
    steps: Iterable[dict], policy: RedactionPolicy
) -> ScanReport:
    """Dry-run: report what :func:`redact_steps` would change, without mutating."""
    salt_id = hashlib.sha256(policy.salt or b"").hexdigest()[:16]
    report = ScanReport(policy_name=policy.name, salt_id=salt_id)
    n = 0
    for step in steps:
        sid = str(step.get("step_id", f"<unknown:{n}>"))
        _scan_value(step, policy, step_id=sid, report=report)
        n += 1
    report.n_steps = n
    return report


def scan_trace_file(
    in_path: str,
    *,
    in_hmac_key: bytes,
    policy: RedactionPolicy,
) -> ScanReport:
    """End-to-end dry run: verify ``in_path``, scan every step, return report.

    Does not write anything to disk and does not require an output
    :class:`RecorderKey`. Useful as a CI pre-flight before
    :func:`redact_trace_file`.
    """
    parsed = verify_trace(in_path, in_hmac_key)
    return scan_steps(parsed.steps, policy)


# ---------------------------------------------------------------- streaming


def redact_trace_file_streaming(
    in_path: str,
    out_path: str,
    *,
    in_hmac_key: bytes,
    policy: RedactionPolicy,
    out_key: Optional[RecorderKey] = None,
    compression: bool = True,
) -> RedactionManifest:
    """Memory-light variant of :func:`redact_trace_file`.

    Reads + verifies ``in_path`` (the verifier still loads steps once),
    but immediately serialises each redacted step to the output writer
    and releases the input reference, so peak memory is O(one step).
    Output is byte-equivalent to :func:`redact_trace_file` when given
    the same ``out_key``; only the salt-derived HMAC chain differs
    when ``out_key`` is freshly generated.
    """
    parsed = verify_trace(in_path, in_hmac_key)
    salt_id = hashlib.sha256(policy.salt or b"").hexdigest()[:16]
    manifest = RedactionManifest(policy_name=policy.name, salt_id=salt_id)
    out_key = out_key or RecorderKey.fresh()
    writer = TraceWriter.open(
        out_path,
        hmac_key=out_key.hmac_key,
        signing_key=out_key.signing_key,
        compression=compression,
    )
    n = 0
    try:
        steps = parsed.steps
        for i in range(len(steps)):
            step = steps[i]
            redacted = redact_step(step, policy, manifest)
            writer.write_step(redacted)
            steps[i] = None  # release reference; help GC
            n += 1
    finally:
        writer.close()
    manifest.n_steps = n
    return manifest

