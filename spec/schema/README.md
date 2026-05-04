# SB-Trace Schema (`spec/schema/`)

This directory is the **machine-readable schema** for the SB-Trace
wire format. It is the companion of the human-readable specs in
`spec/sbtrace-v1.md` and `spec/sbtrace-v2.md` and the authority for:

* the **stable numeric field id** assigned to every well-known field
  in every frame kind,
* the **mandatory / optional** flag for each field,
* the **extension ranges** that govern how new fields may be added by
  future spec revisions, by registered capabilities, or by private /
  experimental implementations,
* the **closed set of frame kinds and step kinds** the spec recognises
  in each wire-format major version.

It is consumed by:

* `stepback.spec.SBTraceSpec` (Python reference) — the runtime
  validator. The frozenset constants in `stepback/spec.py` are
  cross-checked against the schema files in
  `tests/test_schema_field_ids.py`.
* `stepback-core` (Rust core) — the canonical decoder and verifier
  imports the same JSON files at build time.
* Independent implementations in any language — they can vendor or
  fetch the JSON directly.
* The conformance suite (`stepback/conformance/`) — test generators
  use the schema to enumerate fields they must round-trip and
  fields they must reject if mandatory and unknown.

## Why stable field ids?

`.sb` v1 is canonical-JSON on the wire, so on-disk fields are keyed
by string. v2 is dual-encoded (canonical-JSON + deterministic CBOR),
and a future v3 may use a compact tagged encoding. To make those
transitions cheap, **every well-known field is assigned a stable
integer id** the day the field is introduced. The integer id is
*never* reused, never renumbered, and never recycled across
versions, even after a field is deprecated or removed from a future
major version. The string name and the integer id are therefore both
authoritative, and either can be used as the canonical key in a
future encoding without ambiguity.

In v1 only the string names appear on the wire; the integer ids are
spec-internal but are exposed by the schema so independent encoders
can pre-compute their tag tables.

## Extension ranges

Every object schema (wrapper, frame body, step body) declares an
`extension_ranges` array whose entries partition the unsigned 16-bit
field-id space into four classes:

| Range | Class | Meaning |
| --- | --- | --- |
| `1` – `99` | **core** | Reserved for the spec itself. A reader MUST recognise every core id; new core ids may only be added on a wire-format minor or major bump. |
| `100` – `999` | **registered-extension** | Allocated by a capability frame (see `sbtrace-v1.md` §6.2). The capability name in the trace's capability frame enumerates which ids in this range it consumes. A reader MUST recognise every id whose owning capability is mandatory and supported. |
| `1000` – `1999` | **experimental** | Advisory / vendor-private use. A reader MUST ignore unknown ids in this range unless a mandatory capability claims them. Two implementations using overlapping experimental ids is a private agreement, not a wire-format conflict. |
| `2000` – `65535` | **reserved** | Reserved for future spec major versions. A v1 writer MUST NOT emit any field whose id is in this range; a v1 reader MUST reject any wire-format frame that carries one (after the unknown-string-name pass-through described below has been disabled by a future major). |

Field ids `0` and any id `>= 65536` are forbidden in every range.

For the v1 canonical-JSON encoding, *unknown string-keyed fields*
inside a step body are tolerated for forward compatibility (a future
recorder may emit an additional optional field at a string name no
existing reader knows about). This is independent of the integer
field-id range above and is stricter for wrapper, header, blob, and
tail frames, where unknown wrapper keys cause hard rejection.

## Layout

```text
spec/schema/
├── README.md                        # this file
└── v1/
    ├── index.json                   # manifest of every schema in v1
    ├── extension_ranges.json        # the extension-range table above
    ├── wrapper.json                 # the four wrapper keys
    ├── frames/
    │   ├── header.json
    │   ├── capability.json
    │   ├── step.json
    │   ├── blob.json
    │   └── tail.json
    └── step_kinds/
        ├── llm_call.json
        ├── tool_call.json
        ├── router.json
        ├── policy_check.json
        ├── mcp_call.json
        ├── parallel_branch_open.json
        ├── parallel_branch_join.json
        └── exception.json
```

## Schema entry format

Each `*.json` file (other than `index.json` and `extension_ranges.json`)
is a single object with these keys:

| Key | Type | Notes |
| --- | --- | --- |
| `name` | string | Human-readable identifier (e.g. `"step"`, `"llm_call"`). |
| `description` | string | Free-form one-line summary. |
| `wire_version` | string | The SB-Trace wire SemVer this schema entry is part of (`"1.0.0"` for everything in `v1/`). |
| `since` | string | Wire SemVer the entry first appeared in. |
| `fields` | array | Ordered list of field entries (see below). |
| `extension_ranges` | array | Range descriptors. Each entry has `lo`, `hi`, `class`, and `description`. |
| `forward_compatibility` | string | `"strict"` (unknown string keys → hard error) or `"tolerant"` (unknown string keys ignored). Per-frame setting. |

A field entry has these keys:

| Key | Type | Required | Notes |
| --- | --- | --- | --- |
| `id` | integer | yes | Stable field id (`1`..`65535`). Never reused. |
| `name` | string | yes | On-the-wire string key in canonical-JSON encodings. |
| `type` | string | yes | One of `"string"`, `"integer"`, `"boolean"`, `"number"`, `"object"`, `"array"`, `"any"`, `"null"`, `"hash"`, `"hex"`, `"enum"`. `"hash"` is `sha256:<hex>`; `"hex"` is lowercase hex. |
| `required` | boolean | yes | `true` ⇒ mandatory; `false` ⇒ optional. |
| `since` | string | yes | Wire SemVer this field first appeared in. |
| `deprecated_in` | string | no | Wire SemVer that deprecates the field (still readable, no longer emitted). |
| `removed_in` | string | no | Wire SemVer that removes the field (readers from this version onward MUST reject it). |
| `enum` | array | no | When `type == "enum"`, the closed set of allowed string values. |
| `description` | string | no | Human-readable note. |
| `applies_to_step_kind` | string | no | Restrict an optional step field to one step kind (e.g. `"llm_call"` for `llm_request`). |

A range descriptor:

| Key | Type | Notes |
| --- | --- | --- |
| `lo` | integer | inclusive lower bound |
| `hi` | integer | inclusive upper bound |
| `class` | string | `"core"`, `"registered-extension"`, `"experimental"`, or `"reserved"`. |
| `description` | string | free-form explanation |

## Versioning

A field's `id` is permanent. A field whose definition changes (type
narrowed, semantics changed) MUST be allocated a *new* id; the old
id remains in the schema with `removed_in` set so old traces still
parse against the schema they were written under.

Adding a field is a wire-format minor bump (per `stepback.spec`'s
SemVer rules). Removing one is a major bump. Renaming the string
form keeps the id but changes `name` and is also a major bump. None
of these have happened in v1.

## Cross-checks

`tests/test_schema_field_ids.py` enforces:

1. Field ids are unique inside each schema entry.
2. Field ids never collide with their schema's reserved ranges.
3. Every field marked `required: true` in `frames/header.json` /
   `wrapper.json` is in the corresponding `_REQUIRED_*` constant in
   `stepback/spec.py`.
4. Every step kind in `step_kinds/` is in
   `stepback.step_types.StepKind` and in `_STEP_KINDS_V1`.
5. Every range in `extension_ranges.json` matches the per-object
   ranges declared in each schema file.
