"""stepback.bench — reproducible micro-benchmarks for the dirty-set
propagation algorithm and the recorder overhead.

Public surface::

    from stepback.bench.replay_caching import run, compare, BenchResult
    from stepback.bench.record_overhead import run as run_overhead, RecordOverheadResult

The benchmarks exercise the *real* recorder, replay engine, and
canonical-hash modules — no mocks. The CLI entry point is
``stepback bench replay-caching`` / ``stepback bench record-overhead``.
"""
from .replay_caching import (
    BenchResult,
    SyntheticTrace,
    compare,
    run,
)
from .record_overhead import RecordOverheadResult, run as run_record_overhead

__all__ = [
    "BenchResult",
    "RecordOverheadResult",
    "SyntheticTrace",
    "compare",
    "run",
    "run_record_overhead",
]
