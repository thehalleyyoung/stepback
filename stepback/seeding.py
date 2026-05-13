"""Deterministic seeding policy for LLM and tool executors.

Step 69 of ``docs/100_STEPS.md``:

  *Add deterministic seeding policy for LLM and tool executors, including
  warning levels when providers do not support seeds.*

Background
----------
Replay-caching works best when LLM calls are deterministic: same inputs →
same output. Most providers support a ``seed`` parameter for this purpose,
but support varies:

* **OpenAI** — ``seed`` is honoured by the inference backend; output is
  *system_fingerprint*-stable when the seed and fingerprint are both fixed.
* **Gemini** — ``seed`` is accepted and *best-effort* deterministic.
* **Anthropic** — no seed parameter; temperature=0 is deterministic in
  practice but not guaranteed.
* **Bedrock** — Bedrock's ``inferenceConfig`` does not expose a ``seed``
  field; determinism depends on the underlying model hosted via Bedrock.
* **Vertex AI** — similar to Gemini (best-effort via ``generation_config``).
* **Tool executors** — fully deterministic: no seed needed.

The :class:`SeedPolicy` class encodes:

1. A **default seed** applied to every LLM call when the caller does not
   supply one explicitly (``None`` disables the default, leaving the call
   seeded only when the caller passes a seed).
2. A **warn level** controlling what happens when a seed is requested but
   the provider cannot honour it:

   * :attr:`SeedWarnLevel.SILENT` — record the seed in the trace metadata
     and carry on silently.
   * :attr:`SeedWarnLevel.WARN` — issue a :class:`SeedPolicyViolation`
     :class:`UserWarning` via ``warnings.warn``.
   * :attr:`SeedWarnLevel.ERROR` — raise :class:`SeedPolicyError` so the
     caller must either remove the seed or choose a supporting provider.

3. Per-provider **overrides** so a custom deployment (e.g. a vLLM instance
   that *does* honour seeds for a model normally in the NONE bucket) can
   be registered without patching the global table.

The module-level :data:`DEFAULT_SEED_POLICY` is the singleton used by all
shim wrappers unless overridden per-call.  Swap it at import time or use
:func:`set_seed_policy` / :func:`get_seed_policy` for structured access.

Public surface
--------------
* :class:`SeedSupport` — enum of provider seed-support levels.
* :class:`SeedWarnLevel` — enum of violation-handling levels.
* :class:`SeedPolicyViolation` — ``UserWarning`` emitted at WARN level.
* :class:`SeedPolicyError` — ``ValueError`` raised at ERROR level.
* :data:`PROVIDER_SEED_SUPPORT` — canonical registry (provider → support).
* :class:`SeedPolicy` — configurable policy instance.
* :data:`DEFAULT_SEED_POLICY` — module-level singleton.
* :func:`get_seed_policy` — return the current default policy.
* :func:`set_seed_policy` — replace the current default policy.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SeedSupport(str, Enum):
    """How well a provider honours a caller-supplied ``seed`` parameter.

    * ``FULL`` — the provider backend reproducibly samples the same tokens
      for identical (model, messages, seed) triples.
    * ``BEST_EFFORT`` — the provider accepts the seed and tries to be
      deterministic, but does not guarantee bit-identical outputs.
    * ``NONE`` — the provider's API does not accept a seed parameter; any
      seed stored in the trace is metadata only and has no effect on the
      actual inference output.
    """

    FULL = "full"
    """Provider fully honours the seed; output is bit-stable."""

    BEST_EFFORT = "best_effort"
    """Provider accepts seed but determinism is not guaranteed."""

    NONE = "none"
    """Provider ignores or rejects seed; inference output is always stochastic."""


class SeedWarnLevel(str, Enum):
    """How aggressively the :class:`SeedPolicy` reacts when a seed is set
    but the target provider cannot honour it.

    * ``SILENT`` — store the seed in the trace and continue.
    * ``WARN`` — emit a :class:`SeedPolicyViolation` ``UserWarning``.
    * ``ERROR`` — raise :class:`SeedPolicyError` immediately.
    """

    SILENT = "silent"
    """No diagnostic emitted; the seed is stored in the trace silently."""

    WARN = "warn"
    """Issue a :class:`SeedPolicyViolation` warning via ``warnings.warn``."""

    ERROR = "error"
    """Raise :class:`SeedPolicyError`; the call does not proceed."""


# ---------------------------------------------------------------------------
# Exceptions / Warnings
# ---------------------------------------------------------------------------


class SeedPolicyViolation(UserWarning):
    """Warning emitted when a seed is set for a provider that cannot honour it.

    Raised as a :class:`UserWarning` (via ``warnings.warn``) when
    :attr:`SeedWarnLevel.WARN` is in effect.  Upgrade to
    :class:`SeedPolicyError` by setting ``warn_level=SeedWarnLevel.ERROR``.
    """


class SeedPolicyError(ValueError):
    """Exception raised when a seed is set for a provider that cannot honour it.

    Raised when :attr:`SeedWarnLevel.ERROR` is in effect. Downgrade to a
    warning via ``warn_level=SeedWarnLevel.WARN``, or suppress entirely
    with ``warn_level=SeedWarnLevel.SILENT``.
    """


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------


PROVIDER_SEED_SUPPORT: Dict[str, SeedSupport] = {
    "openai": SeedSupport.FULL,
    "anthropic": SeedSupport.NONE,
    "bedrock": SeedSupport.NONE,
    "gemini": SeedSupport.BEST_EFFORT,
    "vertex": SeedSupport.BEST_EFFORT,
    "tool": SeedSupport.FULL,   # deterministic; seed not applicable
    "mcp": SeedSupport.FULL,    # deterministic; seed not applicable
    "langchain": SeedSupport.FULL,  # deterministic tools; seed not applicable
    # OpenAI-compatible providers (Step 96) — conservative defaults.
    # Use SeedPolicy.provider_overrides to upgrade for specific deployments.
    "groq": SeedSupport.BEST_EFFORT,       # seed param accepted; backend varies
    "together": SeedSupport.BEST_EFFORT,   # seed param accepted; backend varies
    "fireworks": SeedSupport.BEST_EFFORT,  # seed param accepted; backend varies
    "cerebras": SeedSupport.NONE,          # no seed param in API
    "nvidia_nim": SeedSupport.BEST_EFFORT, # seed param accepted; NIM backend varies
    "vllm": SeedSupport.BEST_EFFORT,       # seed accepted; engine/version dependent
    "tgi": SeedSupport.BEST_EFFORT,        # seed accepted; version/config dependent
    "llamacpp": SeedSupport.BEST_EFFORT,   # seed accepted; runtime-dependent
    "ollama": SeedSupport.BEST_EFFORT,     # seed accepted via /v1 endpoint; varies
}
"""Canonical registry mapping provider names to seed-support levels.

Keys are lower-case provider identifiers matching the ``provider`` argument
accepted by :meth:`SeedPolicy.check`.  Entries in this dict reflect the
*default* level; use :attr:`SeedPolicy.provider_overrides` to adjust for
custom deployments.

.. note::

    ``tool``, ``mcp``, and ``langchain`` are listed as ``FULL`` because
    tool executors are deterministic by nature.  The seed is not passed to
    them, but a recorded seed in the trace causes no violation.
"""


# ---------------------------------------------------------------------------
# SeedPolicy
# ---------------------------------------------------------------------------


@dataclass
class SeedPolicy:
    """Configurable deterministic seeding policy for LLM executors.

    Attributes
    ----------
    default_seed:
        The seed injected when the caller does not supply one explicitly.
        Defaults to ``42``.  Set to ``None`` to leave calls un-seeded by
        default (relying on the caller to pass a seed when needed).
    warn_level:
        How to react when a seed is requested for a provider that cannot
        honour it.  Defaults to :attr:`SeedWarnLevel.WARN`.
    provider_overrides:
        Per-provider overrides for :data:`PROVIDER_SEED_SUPPORT`.  Useful
        when a custom deployment honours seeds for a normally-unsupported
        provider (e.g. a vLLM endpoint with ``--seed`` fixed).

    Examples
    --------
    Strict mode — fail fast when seeds cannot be honoured::

        from stepback.seeding import SeedPolicy, SeedWarnLevel, set_seed_policy
        set_seed_policy(SeedPolicy(warn_level=SeedWarnLevel.ERROR))

    Silent mode — record seeds in traces without any diagnostics::

        from stepback.seeding import SeedPolicy, SeedWarnLevel, set_seed_policy
        set_seed_policy(SeedPolicy(warn_level=SeedWarnLevel.SILENT))

    Custom Anthropic deployment that honours seeds::

        from stepback.seeding import SeedPolicy, SeedSupport, set_seed_policy
        set_seed_policy(SeedPolicy(
            provider_overrides={"anthropic": SeedSupport.BEST_EFFORT}
        ))
    """

    default_seed: Optional[int] = 42
    warn_level: SeedWarnLevel = SeedWarnLevel.WARN
    provider_overrides: Dict[str, SeedSupport] = field(default_factory=dict)

    # ------------------------------------------------------------------

    def seed_support(self, provider: str) -> SeedSupport:
        """Return the effective :class:`SeedSupport` level for *provider*.

        Checks :attr:`provider_overrides` first, then falls back to
        :data:`PROVIDER_SEED_SUPPORT`, and finally defaults to
        :attr:`SeedSupport.NONE` for unknown providers (conservative).
        """
        key = provider.lower()
        if key in self.provider_overrides:
            return self.provider_overrides[key]
        return PROVIDER_SEED_SUPPORT.get(key, SeedSupport.NONE)

    def effective_seed(self, caller_seed: Optional[int]) -> Optional[int]:
        """Return the seed to record in the trace and pass to the provider.

        If *caller_seed* is not ``None``, it is returned as-is.
        Otherwise :attr:`default_seed` is returned (which may itself be
        ``None`` when the policy has no default).
        """
        if caller_seed is not None:
            return caller_seed
        return self.default_seed

    def check(
        self,
        provider: str,
        seed: Optional[int],
        temperature: float = 0.0,
        *,
        model: Optional[str] = None,
    ) -> Optional[int]:
        """Apply the policy and return the seed to record in the trace.

        If the provider cannot honour the seed, a diagnostic is emitted
        according to :attr:`warn_level`.

        Parameters
        ----------
        provider:
            Lower-case provider name (e.g. ``"openai"``, ``"anthropic"``).
        seed:
            The seed the caller wants to use (may be ``None``).
        temperature:
            The sampling temperature.  Used in warning messages to note
            when nondeterminism is unavoidable.
        model:
            Optional model name, included in diagnostic messages.

        Returns
        -------
        Optional[int]
            The seed to record in the trace metadata.  This is the
            effective seed after applying :attr:`default_seed`.  Note that
            the returned value may not be honoured by the provider; the
            shim is responsible for deciding whether to pass it to the API.

        Raises
        ------
        SeedPolicyError
            When :attr:`warn_level` is :attr:`SeedWarnLevel.ERROR` and the
            provider cannot honour the seed.
        """
        eff_seed = self.effective_seed(seed)
        support = self.seed_support(provider)

        if eff_seed is not None and support is SeedSupport.NONE:
            model_info = f" (model={model!r})" if model else ""
            msg = (
                f"Provider {provider!r}{model_info} does not support a seed parameter. "
                f"seed={eff_seed!r} will be stored in the trace for replay metadata "
                f"but will NOT be sent to the provider API. "
                f"{'Inference will be nondeterministic' if temperature > 0 else 'temperature=0 may reduce variance'}. "
                f"Set warn_level=SeedWarnLevel.SILENT to suppress this message, "
                f"or choose a provider with SeedSupport.FULL."
            )
            if self.warn_level is SeedWarnLevel.ERROR:
                raise SeedPolicyError(msg)
            if self.warn_level is SeedWarnLevel.WARN:
                warnings.warn(msg, SeedPolicyViolation, stacklevel=4)

        if eff_seed is not None and support is SeedSupport.BEST_EFFORT and temperature > 0:
            model_info = f" (model={model!r})" if model else ""
            msg = (
                f"Provider {provider!r}{model_info} offers best-effort seed support. "
                f"Output may vary across runs even with seed={eff_seed!r} and "
                f"temperature={temperature!r}."
            )
            if self.warn_level is SeedWarnLevel.WARN:
                warnings.warn(msg, SeedPolicyViolation, stacklevel=4)

        return eff_seed


# ---------------------------------------------------------------------------
# Module-level default policy
# ---------------------------------------------------------------------------


_default_policy: SeedPolicy = SeedPolicy()

DEFAULT_SEED_POLICY: SeedPolicy = _default_policy
"""The module-level default :class:`SeedPolicy` used by all shim wrappers.

Replace via :func:`set_seed_policy` to apply a project-wide policy without
threading the policy through every ``wrap_*`` call.
"""


def get_seed_policy() -> SeedPolicy:
    """Return the current module-level default :class:`SeedPolicy`.

    Returns the singleton modified by :func:`set_seed_policy`, or the
    out-of-the-box :data:`DEFAULT_SEED_POLICY` if it has not been changed.
    """
    return _default_policy


def set_seed_policy(policy: SeedPolicy) -> None:
    """Replace the module-level default :class:`SeedPolicy`.

    All subsequent ``wrap_openai`` / ``wrap_anthropic`` / ``wrap_bedrock`` /
    ``wrap_gemini`` shim calls that do not receive an explicit
    ``seed_policy`` argument will use *policy*.

    Parameters
    ----------
    policy:
        The new policy to install as the module default.

    Example
    -------
    ::

        from stepback.seeding import SeedPolicy, SeedWarnLevel, set_seed_policy
        set_seed_policy(SeedPolicy(default_seed=7, warn_level=SeedWarnLevel.ERROR))
    """
    global _default_policy, DEFAULT_SEED_POLICY
    _default_policy = policy
    DEFAULT_SEED_POLICY = policy
