"""Nondeterminism class taxonomy for recorded steps.

Step 63 of ``docs/100_STEPS.md``:

  *Record nondeterminism classes for clock, RNG, env, network, and model
  sampling; define when each class forces a dirty step.*

Background
----------
Stepback's replay engine achieves cache hits by verifying that a step's
canonical inputs hash matches the value recorded at trace-write time.  For
fully deterministic steps this is sufficient: same inputs → same outputs.
But real agent traces routinely contain steps that are *inherently
nondeterministic*, i.e. the step's output may differ across executions even
when the canonical inputs are bit-identical:

* ``clock`` — the step observes a wall-clock timestamp or process uptime.
* ``rng`` — the step consumes a random-number generator without a fixed seed.
* ``env`` — the step reads environment variables that may change between
  recording and replay environments.
* ``network`` — the step makes an outbound network call whose response is
  not mocked/controlled.
* ``model_sampling`` — the step calls an LLM with ``temperature > 0`` and
  no seed, so the model's sampled tokens may differ.

For each class, the dirty-forcing semantics state whether the replay engine
**must** treat the step as dirty (i.e. re-execute it) even when the inputs
hash and ``nondeterminism_hash`` both match.  The semantics are:

+------------------+------------------------------------------+
| Class            | Forces dirty when …                      |
+==================+==========================================+
| ``clock``        | Always (unless ``controlled=True``).     |
+------------------+------------------------------------------+
| ``rng``          | Seed is ``None`` (truly random).         |
+------------------+------------------------------------------+
| ``env``          | Always (unless ``controlled=True``).     |
+------------------+------------------------------------------+
| ``network``      | ``controlled=False`` (uncontrolled call).|
+------------------+------------------------------------------+
| ``model_sampling`` | ``temperature > 0`` and ``seed`` is    |
|                  | ``None``.                                |
+------------------+------------------------------------------+

Wire schema
-----------
The ``nondeterminism`` field of a recorded step is a dict.  Two layouts are
supported:

**Single-source** (legacy, for steps with exactly one nondeterminism source)::

    {
        "class": "model_sampling",
        "temperature": 0.7,
        "seed": null
    }

**Multi-source** (preferred; emitted by helper functions in this module)::

    {
        "sources": [
            {"class": "clock", "observed_ns": 1715000000000000000, "controlled": false},
            {"class": "model_sampling", "temperature": 0.0, "seed": 42}
        ]
    }

An empty dict ``{}`` means no nondeterminism was recorded, which is backward
compatible with traces written before this schema existed.  An unknown class
name inside a source record is treated conservatively as *forces dirty*.

Public surface
--------------
* :class:`NondeterminismClass` — string-valued enum of the five source classes.
* :func:`forces_dirty` — determine whether a recorded ``nondeterminism`` payload
  forces the step dirty on replay.
* :func:`clock_nondeterminism` — build a clock-class source record.
* :func:`rng_nondeterminism` — build an RNG-class source record.
* :func:`env_nondeterminism` — build an env-class source record.
* :func:`network_nondeterminism` — build a network-class source record.
* :func:`model_sampling_nondeterminism` — build a model-sampling-class source
  record.
* :func:`combine_nondeterminism` — combine multiple source records into a
  multi-source payload.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# NondeterminismClass
# ---------------------------------------------------------------------------


class NondeterminismClass(str, Enum):
    """The closed set of nondeterminism source classes the recorder recognises.

    ``NondeterminismClass`` is a ``str`` subclass so equality with the raw
    on-disk string is reflexive: ``NondeterminismClass.CLOCK == "clock"``.

    Each value corresponds to a distinct source of nondeterminism and
    carries its own dirty-forcing semantics (see :func:`forces_dirty`).
    """

    CLOCK = "clock"
    """Wall-clock or process-uptime observation."""

    RNG = "rng"
    """Random-number-generator consumption."""

    ENV = "env"
    """Environment-variable observation."""

    NETWORK = "network"
    """Outbound network call whose response is not controlled."""

    MODEL_SAMPLING = "model_sampling"
    """LLM call with non-zero temperature or no fixed seed."""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


# ---------------------------------------------------------------------------
# Source-record helpers
# ---------------------------------------------------------------------------


def clock_nondeterminism(
    observed_ns: int,
    *,
    controlled: bool = False,
) -> dict:
    """Return a clock-class nondeterminism source record.

    Parameters
    ----------
    observed_ns:
        The wall-clock time observed by the step, in nanoseconds since the
        Unix epoch.  Pass ``time.time_ns()`` at the moment the step executes.
    controlled:
        ``True`` when the clock is injected / frozen for testing so that the
        same value will be seen on replay.  Defaults to ``False`` (real clock).

    Returns
    -------
    dict
        A source record suitable for :func:`combine_nondeterminism` or for
        direct use as the ``nondeterminism`` field of a step.
    """
    return {
        "class": NondeterminismClass.CLOCK.value,
        "observed_ns": int(observed_ns),
        "controlled": bool(controlled),
    }


def rng_nondeterminism(
    seed: Optional[int] = None,
    algorithm: str = "default",
) -> dict:
    """Return an RNG-class nondeterminism source record.

    Parameters
    ----------
    seed:
        The seed used to initialise the RNG, or ``None`` if no seed was set
        (fully random).  A fixed seed makes the step reproducible provided
        the RNG algorithm is stable.
    algorithm:
        The RNG algorithm identifier, e.g. ``"default"`` (Python's
        ``random.random()``), ``"numpy.default_rng"``, ``"secrets"``.

    Returns
    -------
    dict
        A nondeterminism source record.
    """
    return {
        "class": NondeterminismClass.RNG.value,
        "seed": seed,
        "algorithm": str(algorithm),
    }


def env_nondeterminism(
    observed: dict,
    *,
    controlled: bool = False,
) -> dict:
    """Return an env-class nondeterminism source record.

    .. warning::

        Do **not** include raw secret values (API keys, passwords) in
        ``observed``.  Either include only non-sensitive keys, or record
        a redacted/hashed representation.

    Parameters
    ----------
    observed:
        A mapping of environment-variable names to their observed values.
        Only include the variables the step actually depends on.
    controlled:
        ``True`` when the environment is fully isolated so the same values
        will be seen on replay (e.g. a hermetic test environment).
        Defaults to ``False`` (real host environment).

    Returns
    -------
    dict
        A nondeterminism source record.
    """
    return {
        "class": NondeterminismClass.ENV.value,
        "observed": dict(observed),
        "controlled": bool(controlled),
    }


def network_nondeterminism(
    endpoint: str,
    *,
    controlled: bool = False,
) -> dict:
    """Return a network-class nondeterminism source record.

    .. warning::

        Do **not** include credentials, tokens, or other secrets in
        ``endpoint``.  Use a sanitised URL without query parameters or
        authentication details.

    Parameters
    ----------
    endpoint:
        A sanitised identifier for the remote service, e.g.
        ``"https://api.example.com"`` (without query parameters or auth).
    controlled:
        ``True`` when the network call is mocked/cassette-backed so the
        response is reproducible.  Defaults to ``False`` (live network).

    Returns
    -------
    dict
        A nondeterminism source record.
    """
    return {
        "class": NondeterminismClass.NETWORK.value,
        "endpoint": str(endpoint),
        "controlled": bool(controlled),
    }


def model_sampling_nondeterminism(
    temperature: float,
    seed: Optional[int] = None,
) -> dict:
    """Return a model-sampling-class nondeterminism source record.

    Parameters
    ----------
    temperature:
        The sampling temperature used for the LLM call.  ``0.0`` is
        typically (but not universally) deterministic.
    seed:
        The provider-side seed, or ``None`` if none was requested.  Note
        that not all providers honour a seed even when supplied.

    Returns
    -------
    dict
        A nondeterminism source record.
    """
    return {
        "class": NondeterminismClass.MODEL_SAMPLING.value,
        "temperature": float(temperature),
        "seed": seed,
    }


def combine_nondeterminism(*sources: dict) -> dict:
    """Combine multiple nondeterminism source records into one multi-source payload.

    Parameters
    ----------
    *sources:
        Nondeterminism source records, each returned by one of the helper
        functions in this module (e.g. :func:`clock_nondeterminism`).

    Returns
    -------
    dict
        A multi-source ``nondeterminism`` payload that can be stored directly
        in the ``nondeterminism`` field of a recorded step.

    Example
    -------
    ::

        from stepback.nondeterminism import (
            clock_nondeterminism, model_sampling_nondeterminism,
            combine_nondeterminism,
        )
        import time

        nondet = combine_nondeterminism(
            clock_nondeterminism(observed_ns=time.time_ns()),
            model_sampling_nondeterminism(temperature=0.7, seed=None),
        )
    """
    return {"sources": [dict(s) for s in sources]}


# ---------------------------------------------------------------------------
# Dirty-forcing semantics
# ---------------------------------------------------------------------------


def _source_forces_dirty(source: Any) -> bool:
    """Return ``True`` if *source* (a single nondeterminism source dict) forces
    the enclosing step to be treated as dirty on replay.

    An unknown or malformed source is treated conservatively as forcing dirty.
    """
    if not isinstance(source, dict):
        return True
    try:
        cls = source.get("class")
        if cls == NondeterminismClass.CLOCK or cls == "clock":
            return not bool(source.get("controlled", False))
        if cls == NondeterminismClass.RNG or cls == "rng":
            # Dirty iff no fixed seed was recorded.
            return source.get("seed") is None
        if cls == NondeterminismClass.ENV or cls == "env":
            return not bool(source.get("controlled", False))
        if cls == NondeterminismClass.NETWORK or cls == "network":
            return not bool(source.get("controlled", False))
        if cls == NondeterminismClass.MODEL_SAMPLING or cls == "model_sampling":
            temp = source.get("temperature", 0.0)
            seed = source.get("seed")
            try:
                temp_f = float(temp)
            except (TypeError, ValueError):
                return True  # malformed temperature → conservative
            return temp_f > 0.0 and seed is None
        # Unknown class → conservative: forces dirty.
        return True
    except Exception:  # noqa: BLE001 - defensive
        return True


def forces_dirty(nondeterminism: Any) -> bool:
    """Determine whether a recorded ``nondeterminism`` payload forces a dirty step.

    This function implements the dirty-forcing semantics for each
    :class:`NondeterminismClass`.  It is called by the dirty-set classifier
    (:py:func:`~stepback.divergence.compute_dirty_set`) and the replay engine
    (:py:meth:`~stepback.replay.Trace.run_replay`) as an additional dirty
    condition beyond input-hash drift.

    Parameters
    ----------
    nondeterminism:
        The ``nondeterminism`` field of a recorded step dict, or any value
        that was stored there.  ``None`` and ``{}`` both return ``False``
        (no nondeterminism recorded — backward compatible with old traces).

    Returns
    -------
    bool
        ``True`` iff the step **must** be re-executed on replay regardless
        of whether the inputs hash matches.

    Examples
    --------
    ::

        >>> forces_dirty({})                                              # no-op
        False
        >>> forces_dirty(model_sampling_nondeterminism(0.7, seed=None))  # always dirty
        True
        >>> forces_dirty(model_sampling_nondeterminism(0.7, seed=42))    # seeded
        False
        >>> forces_dirty(model_sampling_nondeterminism(0.0, seed=None))  # temp=0
        False
        >>> forces_dirty(clock_nondeterminism(12345))                    # live clock
        True
        >>> forces_dirty(clock_nondeterminism(12345, controlled=True))   # frozen
        False
    """
    if not nondeterminism:
        return False
    if not isinstance(nondeterminism, dict):
        return True  # malformed → conservative

    # Multi-source format: {"sources": [...]}
    sources = nondeterminism.get("sources")
    if sources is not None:
        if not isinstance(sources, list):
            return True  # malformed
        return any(_source_forces_dirty(s) for s in sources)

    # Single-source format: {"class": "...", ...}
    if "class" in nondeterminism:
        return _source_forces_dirty(nondeterminism)

    # Unknown format (e.g. legacy ad-hoc payload with no "class") → clean,
    # for backward compatibility with pre-Step-63 traces.
    return False
