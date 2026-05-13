# RFC 0005 — Importer Lossiness Reporting

| Field | Value |
|---|---|
| RFC number | 0005 |
| Title | Importer / Exporter Lossiness Reporting |
| Status | Draft |
| Supersedes | — |
| Created | 2026-05-12 |
| Authors | stepback maintainers |
| Reference impl | `stepback.exporters.LossReport`, `stepback.importers` |

---

## Abstract

This RFC specifies the *lossiness report* interface that every SB-Trace
importer and exporter MUST produce.  A lossiness report is a machine-readable
summary of which fields were perfectly round-tripped, which were
approximated, which had to be synthesised from thin air, and which were
silently dropped.  It lets tool authors and end users reason explicitly
about what information is preserved when traces cross format boundaries.

The reference implementation is `stepback.exporters.LossReport` in
[`stepback/exporters.py`](../../stepback/exporters.py) and is used by
every importer in [`stepback/importers.py`](../../stepback/importers.py).

---

## 1. Motivation

Every import/export boundary involves information loss.  LangSmith does
not record `inputs_hash`; OpenInference does not record `cost_usd`;
OpenAI chat logs do not record parent edges.  Without an explicit lossiness
contract, users silently get traces where dirty-set decisions are
compromised (missing `inputs_hash`), cost estimates are wrong (missing
`cost_usd`), or the causal graph is flat (missing parent edges).

A lossiness report makes these gaps explicit and machine-checkable:
CI can assert that a given importer's synthesised fields have not grown
unexpectedly.

---

## 2. `LossReport` structure

```json
{
  "absent": ["<field_description>", ...],
  "approximated": ["<field_description>", ...],
  "synthesized": ["<field_description>", ...],
  "dropped": ["<field_description>", ...]
}
```

| Category | Meaning | Action required |
|---|---|---|
| `absent` | A field required by SB-Trace has no equivalent in the foreign format and could not be mapped or synthesised. | The importer MUST document this; callers MUST NOT rely on the field for correctness. |
| `approximated` | A field was mapped from the foreign format but may not be numerically or semantically identical to a native recorder's value (e.g. `step_kind` mapped from `run_type`). | The importer MUST document the mapping rule. |
| `synthesized` | A field's value was invented by the importer (e.g. `nondeterminism_hash` synthesised as `canonical_hash({})` for all imported steps). | The importer MUST document what value is used and why. |
| `dropped` | A field present in the foreign-format source was not preserved in the output (e.g. LangSmith `feedback_stats`). | The importer SHOULD document the dropped fields in its docstring. |

---

## 3. Per-field descriptions

Each entry in an `absent` / `approximated` / `synthesized` / `dropped`
list is a human-readable string of the form:

```
"<stepback_field_or_concept>: <reason>"
```

Examples:

```
"inputs_hash: foreign format does not record canonical input hashes; synthesised as blake2b(canonical_json(inputs))"
"step_kind: approximated from LangSmith run_type (llm→llm_call, tool→tool_call, chain→router)"
"nondeterminism_hash: synthesised as blake2b(canonical_json({}))"
"feedback_stats: no stepback equivalent; dropped"
```

---

## 4. Import report

Every importer function returns an `ImportReport` (or equivalent) that
includes a `lossiness: LossReport` field.  The caller can inspect it and
decide whether the import is acceptable for their use-case.

```python
report = import_langsmith_jsonl(path, output_path)
print(report.lossiness.synthesized)
# ["nondeterminism_hash: synthesised as blake2b(canonical_json({}))", ...]
```

---

## 5. Export report

Every exporter function returns an `ExportReport` with a `lossiness:
LossReport` field.  The exporter MAY populate `dropped` fields for
SB-Trace fields that have no foreign equivalent.

---

## 6. Conformance requirements

Implementations MUST:

1. Return a `LossReport` from every import and export function.
2. Never return an empty `LossReport` for a format that is known to be
   lossy.  Silence implies perfect fidelity.
3. Document in the function docstring which fields are in each category.

Implementations SHOULD:

4. Include at minimum `nondeterminism_hash` in `synthesized` for all
   importers, since no foreign format captures it.
5. Include `inputs_hash` in `synthesized` if the importer recomputes it
   from the imported inputs (acceptable) or leaves it absent (MUST be in
   `absent` instead).

---

## 7. Versioning

`LossReport` is a stable public API.  New categories MUST NOT be added
without a minor-version bump to the stepback package.  The four categories
(`absent`, `approximated`, `synthesized`, `dropped`) are exhaustive for v1.

---

## 8. Relationship to other RFCs

- RFC 0001 defines the SB-Trace fields that importers are expected to
  populate.
- RFC 0002 defines `canonical_hash`, used by importers to synthesise
  `inputs_hash` values.

---

## Appendix A — Per-format lossiness summary

| Format | `absent` | `approximated` | `synthesized` | `dropped` |
|---|---|---|---|---|
| LangSmith JSONL | — | `step_kind` (from `run_type`) | `nondeterminism_hash`, `inputs_hash` (recomputed), `step_id` (if missing) | `feedback_stats` |
| OpenInference spans | — | `step_kind` (from span kind) | `nondeterminism_hash`, `cost_usd` (if absent) | `metadata` spans without equivalents |
| OpenAI chat log | `parent_step_id` (flat log has no tree) | `step_kind` (all `llm_call`) | `nondeterminism_hash`, `step_id` | — |
| OTel spans (RFC 0006) | — | `step_kind` (from `agent.step.kind`) | `nondeterminism_hash` | — |

This table is informational; the normative per-importer lossiness is in
the function docstrings and verified by `tests/test_importers.py`.

---

## Appendix B — Changelog

| Date | Author | Change |
|---|---|---|
| 2026-05-12 | stepback maintainers | Initial draft |
