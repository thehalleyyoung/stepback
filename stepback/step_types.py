"""Typed views over recorded steps, trace headers, and frame receipts.

Step 18 of ``docs/100_STEPS.md`` introduces typed dataclass-like wrappers
for the four core on-the-wire dict shapes that the rest of the codebase
manipulates loosely:

* :class:`StepKind` — string-valued enum of every legal ``step_kind``
  value the recorder emits or the replay engine recognises.
* :class:`RecordedStep` — a typed view over a single recorded step
  dictionary (``step_id``, ``step_kind``, ``inputs``, ``outputs``, …).
* :class:`TraceHeader` — a typed view over the first frame of a
  ``.sb`` trace (``magic``, ``format_version``, ``public_key``, …).
* :class:`Receipt` — a typed view over the per-frame wrapper
  (``body``, ``prev_hmac``, ``hmac``, ``sig``).

Back-compat is the headline constraint: every existing call site that
indexes a recorded step like a dict (``step["step_id"]``,
``"foo" in step``, ``step.get("inputs", {})``, ``for k in step``,
``step["inputs"]["context"] = h``) keeps working unchanged. All four
typed wrappers implement :class:`collections.abc.MutableMapping` over
the *same* underlying dict the typed view was constructed from. There is
no copy: ``RecordedStep.from_dict(d).to_dict() is d`` holds, and a
mutation through either the attribute API or the mapping API is visible
through both views.

This is therefore a strictly additive layer. Nothing else in
``stepback`` is required to migrate. New code that wants a typed surface
can do so:

.. code-block:: python

    from stepback.step_types import RecordedStep, StepKind

    step = RecordedStep.from_dict(rec.steps[-1])
    if step.step_kind is StepKind.LLM_CALL:
        print(step.cost_usd, step.llm_request["model"])
        # also still works:
        print(step["outputs"]["choices"][0]["message"]["content"])
"""
from __future__ import annotations

from collections.abc import MutableMapping
from enum import Enum
from typing import Any, ClassVar, Iterator, Mapping, Optional


# ---------------------------------------------------------------------------
# StepKind
# ---------------------------------------------------------------------------


class StepKind(str, Enum):
    """The closed set of ``step_kind`` values emitted by the recorder.

    ``StepKind`` is a ``str`` subclass so equality with the raw on-disk
    string value is reflexive: ``StepKind.LLM_CALL == "llm_call"`` and a
    recorded step's ``step_kind`` field can be compared to a member
    without coercion. Use :meth:`coerce` to normalise an arbitrary
    string (or already-coerced enum member) without raising on unknown
    kinds; use the constructor directly when you want strict validation.
    """

    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    ROUTER = "router"
    POLICY_CHECK = "policy_check"
    MCP_CALL = "mcp_call"
    PARALLEL_BRANCH_OPEN = "parallel_branch_open"
    PARALLEL_BRANCH_JOIN = "parallel_branch_join"
    EXCEPTION = "exception"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value

    @classmethod
    def coerce(cls, value: Any) -> "StepKind | str":
        """Return ``value`` as a :class:`StepKind` if recognised, else as-is.

        Useful for tolerating forward-compatible kinds emitted by a
        newer recorder while still letting current code branch on the
        well-known members::

            kind = StepKind.coerce(step["step_kind"])
            if kind is StepKind.LLM_CALL:
                ...
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError:
            return value

    @classmethod
    def known_values(cls) -> frozenset[str]:
        """The frozen set of every well-known string value, for validation."""
        return frozenset(m.value for m in cls)


# ---------------------------------------------------------------------------
# Internal: dict-backed mapping view base class
# ---------------------------------------------------------------------------


class _DictBackedView(MutableMapping):
    """Mixin: a typed wrapper around an underlying dict.

    Subclasses declare a ``_FIELD_KEYS`` class attribute (the well-known
    keys they expose typed properties for) and implement those
    properties via :meth:`_get` / :meth:`_set`. The underlying dict is
    stored at :attr:`_data`. Identity is preserved: two views built
    over the same dict share state, and ``view.to_dict() is the_dict``.

    The mapping protocol delegates to ``_data`` so unknown keys, custom
    fields, and forward-compatible additions all flow through
    transparently.
    """

    __slots__ = ("_data",)

    _FIELD_KEYS: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, data: Optional[dict] = None) -> None:
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise TypeError(
                f"{type(self).__name__} requires a dict, got {type(data).__name__}"
            )
        object.__setattr__(self, "_data", data)

    # ---- mapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    # ---- equality / repr --------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if isinstance(other, _DictBackedView):
            return self._data == other._data
        if isinstance(other, Mapping):
            return dict(self._data) == dict(other)
        return NotImplemented

    def __hash__(self) -> int:  # pragma: no cover - intentionally unhashable
        raise TypeError(f"{type(self).__name__} is unhashable (mutable view)")

    def __repr__(self) -> str:
        present = {k: self._data.get(k) for k in self._FIELD_KEYS if k in self._data}
        extras = sorted(set(self._data) - self._FIELD_KEYS)
        extra_part = f", +extras={extras}" if extras else ""
        body = ", ".join(f"{k}={v!r}" for k, v in present.items())
        return f"{type(self).__name__}({body}{extra_part})"

    # ---- helpers ----------------------------------------------------------
    def _get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def _set(self, key: str, value: Any) -> None:
        self._data[key] = value

    @classmethod
    def from_dict(cls, data: dict):
        """Wrap an existing dict (no copy) as a typed view."""
        return cls(data)

    def to_dict(self) -> dict:
        """Return the underlying dict (same identity, no copy)."""
        return self._data


# ---------------------------------------------------------------------------
# RecordedStep
# ---------------------------------------------------------------------------


class RecordedStep(_DictBackedView):
    """A typed view over a single recorded step dict.

    Every field maps to the same key in the underlying dict; reads and
    writes go through the dict directly so a step view stays in sync
    with whatever ``Recorder.steps`` (or a freshly read trace) holds.

    The well-known fields are:

    * ``step_id``         — ULID-like string id, monotonic per trace
    * ``step_kind``       — a :class:`StepKind` value (or the raw string)
    * ``name``            — human-readable label, e.g. model id or tool name
    * ``parent_step_id``  — parent edge (None for the root)
    * ``parent_step_ids`` — multi-parent edges (parallel_branch_join only)
    * ``inputs`` / ``outputs``         — canonical-JSON-friendly payloads
    * ``inputs_hash`` / ``outputs_hash`` — sha256(canonical_json(payload))
    * ``nondeterminism`` / ``nondeterminism_hash``
    * ``wallclock_ns``    — recorder wallclock at write time
    * ``cost_usd``        — pinned-price-list cost
    * ``llm_request`` / ``llm_response`` — bytes-for-bytes LLM exchange
      (only present on ``llm_call`` steps)
    """

    _FIELD_KEYS = frozenset({
        "step_id",
        "step_kind",
        "name",
        "parent_step_id",
        "parent_step_ids",
        "inputs",
        "outputs",
        "inputs_hash",
        "outputs_hash",
        "nondeterminism",
        "nondeterminism_hash",
        "wallclock_ns",
        "cost_usd",
        "llm_request",
        "llm_response",
    })

    # ---- typed accessors --------------------------------------------------
    @property
    def step_id(self) -> str:
        """The step's stable id (e.g. ``"step:7"``)."""
        return self._data["step_id"]

    @step_id.setter
    def step_id(self, value: str) -> None:
        self._set("step_id", value)

    @property
    def step_kind(self) -> "StepKind | str":
        """The step kind, coerced to :class:`StepKind` when recognised."""
        raw = self._data.get("step_kind")
        return StepKind.coerce(raw) if raw is not None else raw

    @step_kind.setter
    def step_kind(self, value: "StepKind | str") -> None:
        self._set(
            "step_kind", value.value if isinstance(value, StepKind) else value
        )

    @property
    def name(self) -> Optional[str]:
        """Human-readable label (model id, tool name, etc.)."""
        return self._get("name")

    @name.setter
    def name(self, value: Optional[str]) -> None:
        self._set("name", value)

    @property
    def parent_step_id(self) -> Optional[str]:
        """Parent step id along the call tree (None for the root)."""
        return self._get("parent_step_id")

    @parent_step_id.setter
    def parent_step_id(self, value: Optional[str]) -> None:
        self._set("parent_step_id", value)

    @property
    def parent_step_ids(self) -> Optional[list[str]]:
        """Multi-parent edges (set on ``parallel_branch_join``)."""
        return self._get("parent_step_ids")

    @parent_step_ids.setter
    def parent_step_ids(self, value: Optional[list[str]]) -> None:
        self._set("parent_step_ids", value)

    @property
    def inputs(self) -> dict:
        """Canonical-JSON-friendly inputs payload."""
        return self._data.get("inputs", {})

    @inputs.setter
    def inputs(self, value: dict) -> None:
        self._set("inputs", value)

    @property
    def outputs(self) -> Any:
        """Canonical-JSON-friendly outputs payload."""
        return self._data.get("outputs")

    @outputs.setter
    def outputs(self, value: Any) -> None:
        self._set("outputs", value)

    @property
    def inputs_hash(self) -> Optional[str]:
        """sha256(canonical_json(inputs)), set by the recorder."""
        return self._get("inputs_hash")

    @inputs_hash.setter
    def inputs_hash(self, value: Optional[str]) -> None:
        self._set("inputs_hash", value)

    @property
    def outputs_hash(self) -> Optional[str]:
        """sha256(canonical_json(outputs)), set by the recorder."""
        return self._get("outputs_hash")

    @outputs_hash.setter
    def outputs_hash(self, value: Optional[str]) -> None:
        self._set("outputs_hash", value)

    @property
    def nondeterminism(self) -> dict:
        """Optional nondeterminism inputs (rng seeds, wallclock, …)."""
        return self._data.get("nondeterminism", {})

    @nondeterminism.setter
    def nondeterminism(self, value: dict) -> None:
        self._set("nondeterminism", value)

    @property
    def nondeterminism_hash(self) -> Optional[str]:
        """Hash of the nondeterminism payload."""
        return self._get("nondeterminism_hash")

    @nondeterminism_hash.setter
    def nondeterminism_hash(self, value: Optional[str]) -> None:
        self._set("nondeterminism_hash", value)

    @property
    def wallclock_ns(self) -> Optional[int]:
        """Recorder wallclock in nanoseconds at write time."""
        return self._get("wallclock_ns")

    @wallclock_ns.setter
    def wallclock_ns(self, value: Optional[int]) -> None:
        self._set("wallclock_ns", value)

    @property
    def cost_usd(self) -> float:
        """Cost of this step in USD per the pinned price list."""
        return float(self._data.get("cost_usd", 0.0) or 0.0)

    @cost_usd.setter
    def cost_usd(self, value: float) -> None:
        self._set("cost_usd", float(value))

    @property
    def llm_request(self) -> Optional[dict]:
        """Exact LLM request dict (llm_call steps only)."""
        return self._get("llm_request")

    @llm_request.setter
    def llm_request(self, value: Optional[dict]) -> None:
        self._set("llm_request", value)

    @property
    def llm_response(self) -> Optional[dict]:
        """Exact LLM response dict (llm_call steps only)."""
        return self._get("llm_response")

    @llm_response.setter
    def llm_response(self, value: Optional[dict]) -> None:
        self._set("llm_response", value)


# ---------------------------------------------------------------------------
# TraceHeader
# ---------------------------------------------------------------------------


class TraceHeader(_DictBackedView):
    """A typed view over a `.sb` trace header frame body.

    Header frames pin recorder identity, format/canonicalisation/
    price-list versions, and the cryptographic key material the rest
    of the trace authenticates against. See
    ``stepback/trace_writer.py::TraceWriter.open`` for the canonical
    field set this view mirrors.
    """

    _FIELD_KEYS = frozenset({
        "type",
        "magic",
        "format_version",
        "recorder_version",
        "canonicalisation_version",
        "public_key",
        "hmac_key_id",
        "price_list_version",
        "wallclock_ns",
        "compression",
        "blob_threshold",
        "blob_min_reuse",
    })

    @property
    def type(self) -> str:
        """Frame type tag — always ``"header"`` for a valid header."""
        return self._data.get("type", "")

    @property
    def magic(self) -> str:
        """The format magic string, ``"stepback/.sb"`` for v1."""
        return self._data.get("magic", "")

    @property
    def format_version(self) -> int:
        """Wire-format version (``1`` for canonical-JSON v1)."""
        return int(self._data.get("format_version", 0))

    @property
    def recorder_version(self) -> str:
        """Recorder library version string at write time."""
        return self._data.get("recorder_version", "")

    @property
    def canonicalisation_version(self) -> str:
        """Canonicalisation rules version (``stepback.canonical``)."""
        return self._data.get("canonicalisation_version", "")

    @property
    def public_key(self) -> str:
        """Hex-encoded Ed25519 public key the receipts sign with."""
        return self._data.get("public_key", "")

    @property
    def hmac_key_id(self) -> str:
        """Short hex digest identifying the HMAC key (not the key itself)."""
        return self._data.get("hmac_key_id", "")

    @property
    def price_list_version(self) -> str:
        """Pinned price-list version used to compute ``cost_usd``."""
        return self._data.get("price_list_version", "")

    @property
    def wallclock_ns(self) -> int:
        """Recorder wallclock when the header was written."""
        return int(self._data.get("wallclock_ns", 0))

    @property
    def compression(self) -> str:
        """Compression scheme tag (``"none"`` or e.g. ``"gzip+dedup-2"``)."""
        return self._data.get("compression", "none")

    @property
    def blob_threshold(self) -> int:
        """Min canonical-JSON byte size considered for blob interning."""
        return int(self._data.get("blob_threshold", 0))

    @property
    def blob_min_reuse(self) -> int:
        """Min reference count required for a sub-tree to be interned."""
        return int(self._data.get("blob_min_reuse", 0))


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


class Receipt(_DictBackedView):
    """A typed view over the per-frame wrapper written by ``TraceWriter``.

    The wrapper is the cryptographic envelope around every body frame::

        {
            "body":      <frame body>,
            "prev_hmac": "<hex>",
            "hmac":      "<hex>",
            "sig":       "ed25519:<hex>",
        }

    The receipt view exposes the four fields, plus convenience parsers
    for the cryptographic material:

    * :attr:`prev_hmac_bytes` / :attr:`hmac_bytes` decode the hex
      digests once.
    * :attr:`signature_scheme` and :attr:`signature_hex` split the
      ``"<scheme>:<hex>"`` envelope of the ``sig`` field.
    """

    _FIELD_KEYS = frozenset({"body", "prev_hmac", "hmac", "sig"})

    @property
    def body(self) -> dict:
        """The wrapped frame body (header / step / blob / tail)."""
        return self._data.get("body", {})

    @body.setter
    def body(self, value: dict) -> None:
        self._set("body", value)

    @property
    def prev_hmac(self) -> str:
        """Hex-encoded HMAC of the previous frame (or 64 zeros for the root)."""
        return self._data.get("prev_hmac", "")

    @prev_hmac.setter
    def prev_hmac(self, value: str) -> None:
        self._set("prev_hmac", value)

    @property
    def hmac(self) -> str:
        """Hex-encoded HMAC over (prev_hmac || canonical_json(body))."""
        return self._data.get("hmac", "")

    @hmac.setter
    def hmac(self, value: str) -> None:
        self._set("hmac", value)

    @property
    def sig(self) -> str:
        """Signature envelope, formatted ``"<scheme>:<hex>"``."""
        return self._data.get("sig", "")

    @sig.setter
    def sig(self, value: str) -> None:
        self._set("sig", value)

    # ---- derived ----------------------------------------------------------
    @property
    def prev_hmac_bytes(self) -> bytes:
        """Decoded :attr:`prev_hmac`. Empty bytes when not set."""
        v = self._data.get("prev_hmac", "")
        return bytes.fromhex(v) if v else b""

    @property
    def hmac_bytes(self) -> bytes:
        """Decoded :attr:`hmac`. Empty bytes when not set."""
        v = self._data.get("hmac", "")
        return bytes.fromhex(v) if v else b""

    @property
    def signature_scheme(self) -> str:
        """Signature scheme name parsed from :attr:`sig` (e.g. ``"ed25519"``)."""
        sig = self._data.get("sig", "")
        return sig.split(":", 1)[0] if ":" in sig else ""

    @property
    def signature_hex(self) -> str:
        """Hex-encoded signature payload parsed from :attr:`sig`."""
        sig = self._data.get("sig", "")
        return sig.split(":", 1)[1] if ":" in sig else ""

    @property
    def signature_bytes(self) -> bytes:
        """Decoded signature payload."""
        h = self.signature_hex
        return bytes.fromhex(h) if h else b""

    @property
    def frame_kind(self) -> Optional[str]:
        """Convenience: ``body.get("type")`` (header / step / blob / tail)."""
        body = self.body
        return body.get("type") if isinstance(body, Mapping) else None


__all__ = [
    "StepKind",
    "RecordedStep",
    "TraceHeader",
    "Receipt",
]
