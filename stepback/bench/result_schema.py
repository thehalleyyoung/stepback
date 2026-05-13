"""Comprehensive benchmark result schema (Step 114).

Every benchmark result that stepback produces — whether from
``replay-caching``, ``record-overhead``, ``soak``, or a future
corpus-level run — can be expressed as a :class:`BenchRunRecord`.
The schema is stable JSON with a ``schema_version`` field to support
forward-compatible reads and future migrations.

Design goals
------------
* **Forward-compatible JSON** — unknown keys are preserved on round-trip
  so old readers do not explode on newer records.
* **Nullable / estimated fields** — fields that cannot be measured in
  every benchmark context (e.g. per-trace storage bytes, exact LLM-call
  counts) carry an explicit ``None`` or an ``*_estimated`` sibling so
  callers can distinguish measured from approximated values.
* **Injectable detection** — :class:`HardwareInfo` and
  :class:`VersionInfo` expose ``.detect()`` class-methods but can also
  be constructed directly for reproducible test assertions.
* **Strict JSON safety** — ``to_json()`` converts ``None`` → ``null``
  and clips non-finite floats to ``None`` so the output is always
  standard JSON.

Stable schema key layout (``schema_version == "1.0"``)::

    {
      "schema_version": "1.0",
      "run_id": "<uuid4>",
      "timestamp_utc": "<ISO8601>",
      "corpus_id": "synthetic-200-random_step",
      "trace_count": 10,
      "trial_count": 10,
      "substitutions": { ... },
      "dirty_set": { ... },
      "cache": { ... },
      "latency": { ... },
      "cost": { ... },
      "storage": { ... } | null,
      "versions": { ... },
      "hardware": { ... }
    }
"""
from __future__ import annotations

import math
import os
import platform
import statistics
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------ helpers

def _safe_float(v: Any) -> Optional[float]:
    """Return *v* as a finite float, or None if non-finite / None."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


# ------------------------------------------------------------------ sub-records


@dataclass
class SubstitutionStats:
    """Statistics about the substitutions applied during the benchmark run.

    ``by_type`` maps Python class names (``PromptSubstitution``,
    ``ToolOutputSubstitution``, …) to counts.  If the benchmark runner
    did not record per-type breakdown the dict is empty.
    ``by_strategy`` captures the strategy label from the CLI, e.g.
    ``{"random_step": 10}``.
    """

    total: int
    """Total number of substitutions applied across all trials."""

    by_type: Dict[str, int]
    """Counts per substitution class name (may be empty)."""

    by_strategy: Dict[str, int]
    """Counts per strategy label (e.g. "random_step")."""

    def to_json(self) -> dict:
        return {
            "total": self.total,
            "by_type": dict(self.by_type),
            "by_strategy": dict(self.by_strategy),
        }

    @classmethod
    def from_json(cls, d: dict) -> "SubstitutionStats":
        return cls(
            total=int(d.get("total", 0)),
            by_type=dict(d.get("by_type") or {}),
            by_strategy=dict(d.get("by_strategy") or {}),
        )


@dataclass
class DirtySetStats:
    """Dirty-set size statistics across all trials.

    All size values are in units of *steps* (not bytes).  ``sizes`` is
    the raw list of per-trial dirty counts; the scalar summaries are
    derived from it.
    """

    count: int
    """Number of observations (== trial count)."""

    min_val: int
    max_val: int
    mean: float
    median: float
    p95: float

    sizes: List[int]
    """Raw per-trial dirty-set sizes for downstream analysis."""

    def to_json(self) -> dict:
        return {
            "count": self.count,
            "min": self.min_val,
            "max": self.max_val,
            "mean": _safe_float(self.mean),
            "median": _safe_float(self.median),
            "p95": _safe_float(self.p95),
            "sizes": list(self.sizes),
        }

    @classmethod
    def from_json(cls, d: dict) -> "DirtySetStats":
        sizes = list(d.get("sizes") or [])
        return cls(
            count=int(d.get("count", len(sizes))),
            min_val=int(d.get("min", min(sizes, default=0))),
            max_val=int(d.get("max", max(sizes, default=0))),
            mean=float(d.get("mean") or 0),
            median=float(d.get("median") or 0),
            p95=float(d.get("p95") or 0),
            sizes=sizes,
        )

    @classmethod
    def from_sizes(cls, sizes: List[int]) -> "DirtySetStats":
        """Build from a raw list of per-trial dirty-step counts."""
        if not sizes:
            return cls(count=0, min_val=0, max_val=0, mean=0.0,
                       median=0.0, p95=0.0, sizes=[])
        n = len(sizes)
        sorted_s = sorted(sizes)
        p95_idx = min(n - 1, int(0.95 * n))
        return cls(
            count=n,
            min_val=sorted_s[0],
            max_val=sorted_s[-1],
            mean=statistics.mean(sizes),
            median=float(statistics.median(sizes)),
            p95=float(sorted_s[p95_idx]),
            sizes=list(sizes),
        )


@dataclass
class CacheStats:
    """Step-cache utilisation during replay.

    Fields marked ``_estimated`` are derived from ``n_steps`` and
    ``dirty_count``; fields without the suffix are directly measured by
    the replay engine.  When the benchmark runner captures the replay
    engine's hit/miss counters it should populate the ``measured_*``
    fields and leave ``estimated_*`` as ``None``.
    """

    estimated_cache_hits: Optional[int]
    """steps_saved = total_steps - total_dirty.  May be None."""

    estimated_cache_hit_rate: Optional[float]
    """0–1, derived from estimated hits.  May be None."""

    estimated_llm_steps_saved: Optional[int]
    """Dirty-set savings expressed in total steps (not just LLM steps)."""

    measured_cache_hits: Optional[int] = None
    measured_cache_misses: Optional[int] = None
    measured_hit_rate: Optional[float] = None

    def to_json(self) -> dict:
        return {
            "estimated_cache_hits": self.estimated_cache_hits,
            "estimated_cache_hit_rate": _safe_float(
                self.estimated_cache_hit_rate
            ),
            "estimated_llm_steps_saved": self.estimated_llm_steps_saved,
            "measured_cache_hits": self.measured_cache_hits,
            "measured_cache_misses": self.measured_cache_misses,
            "measured_hit_rate": _safe_float(self.measured_hit_rate),
        }

    @classmethod
    def from_json(cls, d: dict) -> "CacheStats":
        return cls(
            estimated_cache_hits=d.get("estimated_cache_hits"),
            estimated_cache_hit_rate=d.get("estimated_cache_hit_rate"),
            estimated_llm_steps_saved=d.get("estimated_llm_steps_saved"),
            measured_cache_hits=d.get("measured_cache_hits"),
            measured_cache_misses=d.get("measured_cache_misses"),
            measured_hit_rate=d.get("measured_hit_rate"),
        )


@dataclass
class LatencyStats:
    """Wall-clock timing for the benchmark run."""

    wall_time_ms: float
    """Total wall-clock time for the whole run (all trials)."""

    per_trial_ms: Optional[float] = None
    """Mean time per trial (wall_time_ms / trial_count)."""

    record_ms: Optional[float] = None
    """Time spent in the recorder path (if measured separately)."""

    replay_ms: Optional[float] = None
    """Time spent in the replay engine (if measured separately)."""

    def to_json(self) -> dict:
        return {
            "wall_time_ms": _safe_float(self.wall_time_ms),
            "per_trial_ms": _safe_float(self.per_trial_ms),
            "record_ms": _safe_float(self.record_ms),
            "replay_ms": _safe_float(self.replay_ms),
        }

    @classmethod
    def from_json(cls, d: dict) -> "LatencyStats":
        return cls(
            wall_time_ms=float(d.get("wall_time_ms") or 0),
            per_trial_ms=d.get("per_trial_ms"),
            record_ms=d.get("record_ms"),
            replay_ms=d.get("replay_ms"),
        )


@dataclass
class CostStats:
    """Cost-reduction statistics from the replay-caching benchmark."""

    cost_reduction_factor: float
    """n_steps / mean(dirty_set_sizes): how many times cheaper replay is."""

    estimated_baseline_llm_calls: int
    """Steps if every step were re-executed (n_steps * n_trials)."""

    estimated_actual_llm_calls: int
    """Estimated actual executions (sum of dirty_set_sizes)."""

    estimated_savings_pct: float
    """100 * (1 - actual/baseline)."""

    def to_json(self) -> dict:
        return {
            "cost_reduction_factor": _safe_float(self.cost_reduction_factor),
            "estimated_baseline_llm_calls": self.estimated_baseline_llm_calls,
            "estimated_actual_llm_calls": self.estimated_actual_llm_calls,
            "estimated_savings_pct": _safe_float(self.estimated_savings_pct),
        }

    @classmethod
    def from_json(cls, d: dict) -> "CostStats":
        return cls(
            cost_reduction_factor=float(d.get("cost_reduction_factor") or 0),
            estimated_baseline_llm_calls=int(
                d.get("estimated_baseline_llm_calls", 0)
            ),
            estimated_actual_llm_calls=int(
                d.get("estimated_actual_llm_calls", 0)
            ),
            estimated_savings_pct=float(d.get("estimated_savings_pct") or 0),
        )


@dataclass
class StorageStats:
    """On-disk `.sb` trace storage statistics.

    Fields are ``None`` when the benchmark runner does not capture trace
    sizes (e.g. when traces are cleaned up immediately after each trial).
    """

    total_bytes: Optional[int] = None
    """Sum of `.sb` file sizes for all traces in the run."""

    mean_bytes_per_trace: Optional[float] = None
    mean_bytes_per_step: Optional[float] = None

    result_json_bytes: Optional[int] = None
    """Byte length of ``BenchRunRecord.to_json()`` serialised to compact JSON."""

    def to_json(self) -> dict:
        return {
            "total_bytes": self.total_bytes,
            "mean_bytes_per_trace": _safe_float(self.mean_bytes_per_trace),
            "mean_bytes_per_step": _safe_float(self.mean_bytes_per_step),
            "result_json_bytes": self.result_json_bytes,
        }

    @classmethod
    def from_json(cls, d: dict) -> "StorageStats":
        return cls(
            total_bytes=d.get("total_bytes"),
            mean_bytes_per_trace=d.get("mean_bytes_per_trace"),
            mean_bytes_per_step=d.get("mean_bytes_per_step"),
            result_json_bytes=d.get("result_json_bytes"),
        )


@dataclass
class VersionInfo:
    """Software versions recorded at benchmark run time."""

    stepback_version: str
    python_version: str
    """``sys.version``, e.g. ``'3.11.8 (main, …)'``."""

    python_implementation: str
    """``platform.python_implementation()``, e.g. ``'CPython'``."""

    platform_str: str
    """``platform.platform()`` — OS + release string."""

    def to_json(self) -> dict:
        return {
            "stepback_version": self.stepback_version,
            "python_version": self.python_version,
            "python_implementation": self.python_implementation,
            "platform": self.platform_str,
        }

    @classmethod
    def from_json(cls, d: dict) -> "VersionInfo":
        return cls(
            stepback_version=str(d.get("stepback_version", "unknown")),
            python_version=str(d.get("python_version", "")),
            python_implementation=str(d.get("python_implementation", "")),
            platform_str=str(d.get("platform", "")),
        )

    @classmethod
    def detect(cls) -> "VersionInfo":
        """Capture version information from the running interpreter."""
        try:
            from stepback import __version__ as sv
        except Exception:  # pragma: no cover
            sv = "unknown"
        return cls(
            stepback_version=sv,
            python_version=sys.version,
            python_implementation=platform.python_implementation(),
            platform_str=platform.platform(),
        )


@dataclass
class HardwareInfo:
    """Host hardware captured at benchmark run time.

    All fields are best-effort; ``None`` means not available on this
    platform.  We deliberately avoid adding ``psutil`` as a hard
    dependency — RAM is detected only on Linux via ``/proc/meminfo``
    and macOS via ``sysctl``.
    """

    os: str
    """``platform.system()``, e.g. ``'Linux'`` or ``'Darwin'``."""

    cpu_count: Optional[int]
    """``os.cpu_count()`` (logical cores)."""

    cpu_model: Optional[str]
    """Best-effort CPU model string; ``None`` if not detectable."""

    ram_gb: Optional[float]
    """Total RAM in gigabytes; ``None`` if not detectable."""

    def to_json(self) -> dict:
        return {
            "os": self.os,
            "cpu_count": self.cpu_count,
            "cpu_model": self.cpu_model,
            "ram_gb": _safe_float(self.ram_gb),
        }

    @classmethod
    def from_json(cls, d: dict) -> "HardwareInfo":
        return cls(
            os=str(d.get("os", "")),
            cpu_count=d.get("cpu_count"),
            cpu_model=d.get("cpu_model"),
            ram_gb=d.get("ram_gb"),
        )

    @classmethod
    def detect(cls) -> "HardwareInfo":
        """Capture hardware information from the current host."""
        return cls(
            os=platform.system(),
            cpu_count=os.cpu_count(),
            cpu_model=_detect_cpu_model(),
            ram_gb=_detect_ram_gb(),
        )


def _detect_cpu_model() -> Optional[str]:
    """Return a best-effort CPU model string, or None."""
    sys_name = platform.system()
    if sys_name == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="ascii", errors="replace") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif sys_name == "Darwin":
        try:
            import subprocess
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL, timeout=2,
            )
            return out.decode("utf-8", errors="replace").strip()
        except Exception:
            pass
    return None


def _detect_ram_gb() -> Optional[float]:
    """Return total physical RAM in GiB, or None if not detectable."""
    sys_name = platform.system()
    if sys_name == "Linux":
        try:
            with open("/proc/meminfo", encoding="ascii", errors="replace") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024 * 1024), 2)
        except OSError:
            pass
    elif sys_name == "Darwin":
        try:
            import subprocess
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                stderr=subprocess.DEVNULL, timeout=2,
            )
            return round(int(out.strip()) / (1024 ** 3), 2)
        except Exception:
            pass
    return None


# ------------------------------------------------------------------ top-level

#: Stable schema version; bump the major component on breaking changes.
SCHEMA_VERSION = "1.0"


@dataclass
class BenchRunRecord:
    """Comprehensive benchmark run record with a stable JSON schema.

    Every field in the step-114 requirement is represented:

    * ``corpus_id`` — identifies the trace corpus or benchmark preset
    * ``trace_count`` — number of distinct traces in the corpus/run
    * ``trial_count`` — number of independent benchmark repetitions
    * ``substitutions`` — :class:`SubstitutionStats` (distribution)
    * ``dirty_set`` — :class:`DirtySetStats` (p50/p95/mean/max)
    * ``cache`` — :class:`CacheStats` (hits, LLM calls saved)
    * ``latency`` — :class:`LatencyStats` (wall-clock, per-trial)
    * ``cost`` — :class:`CostStats` (reduction factor, savings %)
    * ``storage`` — :class:`StorageStats` or None
    * ``versions`` — :class:`VersionInfo`
    * ``hardware`` — :class:`HardwareInfo`

    Fields that the current benchmark runner cannot measure directly are
    present but nullable / estimated (documented in their sub-records).
    """

    schema_version: str
    run_id: str
    timestamp_utc: str
    corpus_id: str
    trace_count: int
    trial_count: int
    substitutions: SubstitutionStats
    dirty_set: DirtySetStats
    cache: CacheStats
    latency: LatencyStats
    cost: CostStats
    storage: Optional[StorageStats]
    versions: VersionInfo
    hardware: HardwareInfo

    def to_json(self) -> dict:
        """Return a JSON-serialisable plain dict (no NaN / Infinity)."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "timestamp_utc": self.timestamp_utc,
            "corpus_id": self.corpus_id,
            "trace_count": self.trace_count,
            "trial_count": self.trial_count,
            "substitutions": self.substitutions.to_json(),
            "dirty_set": self.dirty_set.to_json(),
            "cache": self.cache.to_json(),
            "latency": self.latency.to_json(),
            "cost": self.cost.to_json(),
            "storage": self.storage.to_json() if self.storage else None,
            "versions": self.versions.to_json(),
            "hardware": self.hardware.to_json(),
        }

    @classmethod
    def from_json(cls, d: dict) -> "BenchRunRecord":
        """Re-hydrate a record from a plain dict.

        Unknown top-level keys are silently ignored to support forward
        compatibility.  Raises :class:`ValueError` on unsupported
        ``schema_version`` majors.
        """
        sv = str(d.get("schema_version", SCHEMA_VERSION))
        major = sv.split(".")[0]
        if major != SCHEMA_VERSION.split(".")[0]:
            raise ValueError(
                f"Unsupported BenchRunRecord schema_version {sv!r}; "
                f"expected major {SCHEMA_VERSION.split('.')[0]!r}"
            )
        storage_raw = d.get("storage")
        return cls(
            schema_version=sv,
            run_id=str(d.get("run_id", "")),
            timestamp_utc=str(d.get("timestamp_utc", "")),
            corpus_id=str(d.get("corpus_id", "")),
            trace_count=int(d.get("trace_count", 0)),
            trial_count=int(d.get("trial_count", 0)),
            substitutions=SubstitutionStats.from_json(
                d.get("substitutions") or {}
            ),
            dirty_set=DirtySetStats.from_json(d.get("dirty_set") or {}),
            cache=CacheStats.from_json(d.get("cache") or {}),
            latency=LatencyStats.from_json(d.get("latency") or {}),
            cost=CostStats.from_json(d.get("cost") or {}),
            storage=(
                StorageStats.from_json(storage_raw)
                if storage_raw is not None
                else None
            ),
            versions=VersionInfo.from_json(d.get("versions") or {}),
            hardware=HardwareInfo.from_json(d.get("hardware") or {}),
        )

    @classmethod
    def from_bench_result(
        cls,
        result: Any,
        *,
        corpus_id: str = "synthetic",
        versions: Optional[VersionInfo] = None,
        hardware: Optional[HardwareInfo] = None,
        run_id: Optional[str] = None,
        timestamp_utc: Optional[str] = None,
    ) -> "BenchRunRecord":
        """Build a :class:`BenchRunRecord` from a :class:`~stepback.bench.replay_caching.BenchResult`.

        Fields that the current ``BenchResult`` cannot provide are set to
        ``None`` (storage) or estimated (cache hits, LLM calls saved).

        Parameters
        ----------
        result:
            A :class:`~stepback.bench.replay_caching.BenchResult` instance.
        corpus_id:
            Human-readable identifier for the corpus, e.g.
            ``"synthetic-200-random_step"`` or ``"swe-bench-verified"``.
        versions:
            If ``None``, :meth:`VersionInfo.detect` is called.
        hardware:
            If ``None``, :meth:`HardwareInfo.detect` is called.
        run_id:
            UUID string; a fresh ``uuid4`` is generated if not supplied.
        timestamp_utc:
            ISO 8601 UTC timestamp; ``datetime.now(UTC)`` is used if not supplied.
        """
        sizes = list(result.dirty_set_sizes)
        n_steps = int(result.n_steps)
        n_trials = int(result.n_trials)
        n_subs = int(result.n_substitutions)
        strategy = str(result.strategy)

        # Cache estimates: (total_steps - total_dirty) / total_steps
        total_steps = n_steps * n_trials
        total_dirty = sum(sizes)
        est_hits: Optional[int] = max(0, total_steps - total_dirty)
        est_rate: Optional[float] = (
            est_hits / total_steps if total_steps > 0 else None
        )
        est_saved: Optional[int] = est_hits

        # Cost savings %
        baseline = total_steps
        actual = total_dirty
        savings_pct = (
            100.0 * (1.0 - actual / baseline) if baseline > 0 else 0.0
        )

        return cls(
            schema_version=SCHEMA_VERSION,
            run_id=run_id or str(uuid.uuid4()),
            timestamp_utc=(
                timestamp_utc
                or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            ),
            corpus_id=corpus_id,
            trace_count=n_trials,
            trial_count=n_trials,
            substitutions=SubstitutionStats(
                total=n_subs * n_trials,
                by_type={},
                by_strategy={strategy: n_subs * n_trials},
            ),
            dirty_set=DirtySetStats.from_sizes(sizes),
            cache=CacheStats(
                estimated_cache_hits=est_hits,
                estimated_cache_hit_rate=est_rate,
                estimated_llm_steps_saved=est_saved,
            ),
            latency=LatencyStats(
                wall_time_ms=float(result.wall_time_ms),
                per_trial_ms=(
                    result.wall_time_ms / n_trials if n_trials > 0 else None
                ),
            ),
            cost=CostStats(
                cost_reduction_factor=float(result.cost_reduction_factor),
                estimated_baseline_llm_calls=baseline,
                estimated_actual_llm_calls=actual,
                estimated_savings_pct=savings_pct,
            ),
            storage=None,
            versions=versions or VersionInfo.detect(),
            hardware=hardware or HardwareInfo.detect(),
        )
