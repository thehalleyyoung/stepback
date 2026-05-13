"""stepback.bench — reproducible micro-benchmarks for the dirty-set
propagation algorithm and the recorder overhead.

Public surface::

    from stepback.bench.replay_caching import run, compare, BenchResult
    from stepback.bench.record_overhead import run as run_overhead, RecordOverheadResult
    from stepback.bench.dirty_set_distributions import run as run_distributions, DistributionSuite

The benchmarks exercise the *real* recorder, replay engine, and
canonical-hash modules — no mocks. The CLI entry point is
``stepback bench replay-caching`` / ``stepback bench record-overhead`` /
``stepback bench dirty-set-distributions``.
"""
from .replay_caching import (
    BenchResult,
    SyntheticTrace,
    compare,
    run,
)
from .record_overhead import RecordOverheadResult, run as run_record_overhead
from .soak import SoakResult, run as run_soak
from .dirty_set_distributions import (
    CorpusDistribution,
    DistributionSuite,
    run as run_distributions,
)
from .result_schema import (
    BenchRunRecord,
    CacheStats,
    CostStats,
    DirtySetStats,
    HardwareInfo,
    LatencyStats,
    StorageStats,
    SubstitutionStats,
    VersionInfo,
)
from .corpus_loaders import (
    AgentBenchLoader,
    CorpusLoadError,
    CorpusLoader,
    CorpusTask,
    GAIALoader,
    LoadWarning,
    OSWorldLoader,
    SWEBenchLoader,
    TauBenchLoader,
    WebArenaLoader,
    list_loaders,
    load_corpus,
)
from .author_corpora import (
    AUTHOR_CORPUS_META,
    generate_author_corpora,
    list_author_corpora,
    load_author_corpus,
)
from .minimization import (
    MinimizationBenchResult,
    StrategyResult,
    run as run_minimization,
)

__all__ = [
    # replay-caching benchmark
    "BenchResult",
    "SyntheticTrace",
    "compare",
    "run",
    # record-overhead benchmark
    "RecordOverheadResult",
    "run_record_overhead",
    # soak benchmark
    "SoakResult",
    "run_soak",
    # dirty-set distributions
    "CorpusDistribution",
    "DistributionSuite",
    "run_distributions",
    # result schema
    "BenchRunRecord",
    "CacheStats",
    "CostStats",
    "DirtySetStats",
    "HardwareInfo",
    "LatencyStats",
    "StorageStats",
    "SubstitutionStats",
    "VersionInfo",
    # corpus loaders
    "AgentBenchLoader",
    "CorpusLoadError",
    "CorpusLoader",
    "CorpusTask",
    "GAIALoader",
    "LoadWarning",
    "OSWorldLoader",
    "SWEBenchLoader",
    "TauBenchLoader",
    "WebArenaLoader",
    "list_loaders",
    "load_corpus",
    # author corpora
    "AUTHOR_CORPUS_META",
    "generate_author_corpora",
    "list_author_corpora",
    "load_author_corpus",
    # minimization benchmark
    "MinimizationBenchResult",
    "StrategyResult",
    "run_minimization",
]
