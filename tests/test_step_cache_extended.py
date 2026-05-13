"""Tests for NamespacedStepCache and MultiTierStepCache (Step 138).

Covers:
- NamespacedStepCache: namespace isolation between orgs/corpora
- NamespacedStepCache: dedup within same namespace across runs
- NamespacedStepCache: transparent delegation; returned entry has logical kind
- NamespacedStepCache: unsafe namespace components rejected (path traversal, slashes)
- NamespacedStepCache: empty org_id / corpus_id combos (all 4 variants)
- NamespacedStepCache: close delegates to backend
- MultiTierStepCache: L1 hit skips L2
- MultiTierStepCache: L2 fallback when L1 misses
- MultiTierStepCache: L2 hit promotes to L1 (promote=True)
- MultiTierStepCache: no promotion when promote=False
- MultiTierStepCache: L2 authoritative (L2 failure raises from put)
- MultiTierStepCache: L1 best-effort on put (L1 failure silently ignored)
- MultiTierStepCache: write_l1=False skips L1 write but still promotes on get
- MultiTierStepCache: close delegates to both tiers
- Composition: NamespacedStepCache(MultiTierStepCache(...)) deduplicates within namespace
"""
from __future__ import annotations

import time
from typing import Optional
from unittest.mock import MagicMock

import pytest

import stepback
from stepback.step_cache import (
    DiskStepCache,
    MultiTierStepCache,
    NamespacedStepCache,
    StepCache,
    StepCacheEntry,
    _build_namespace_prefix,
    _validate_namespace_component,
)


# ---------------------------------------------------------------------- helpers


def _make_entry(
    kind: str = "tool_call",
    inputs_hash: str = "sha256:" + "a" * 64,
    outputs: object = None,
) -> StepCacheEntry:
    return StepCacheEntry(
        step_kind=kind,
        inputs_hash=inputs_hash,
        outputs=outputs if outputs is not None else {"result": "hello"},
        cached_at=1_700_000_000.0,
    )


class _DictCache(StepCache):
    """In-memory stub cache for testing."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], StepCacheEntry] = {}
        self.get_calls: list[tuple[str, str]] = []
        self.put_calls: list[StepCacheEntry] = []
        self.closed = False

    def get(self, kind: str, inputs_hash: str) -> Optional[StepCacheEntry]:
        self.get_calls.append((kind, inputs_hash))
        return self._store.get((kind, inputs_hash))

    def put(self, entry: StepCacheEntry) -> None:
        self.put_calls.append(entry)
        self._store[(entry.step_kind, entry.inputs_hash)] = entry

    def close(self) -> None:
        self.closed = True


class _RaisingCache(StepCache):
    """Always raises on put, always misses on get."""

    def get(self, kind: str, inputs_hash: str) -> Optional[StepCacheEntry]:
        return None

    def put(self, entry: StepCacheEntry) -> None:
        raise OSError("storage unavailable")


# ====================================================== _validate_namespace_component


class TestValidateNamespaceComponent:
    @pytest.mark.parametrize(
        "value",
        [
            "acme",
            "Acme-Corp",
            "my.corpus_v2",
            "a",
            "A" * 256,
        ],
    )
    def test_valid_components_accepted(self, value: str) -> None:
        _validate_namespace_component(value, "org_id")  # must not raise

    @pytest.mark.parametrize(
        "value",
        [
            "",          # empty
            "a/b",       # slash
            "../x",      # path traversal
            "/abs",      # leading slash
            "a\\b",      # backslash
            "a b",       # space
            "A" * 257,   # too long
            "a:b",       # colon (unsafe in some filesystems)
        ],
    )
    def test_invalid_components_rejected(self, value: str) -> None:
        with pytest.raises(ValueError):
            _validate_namespace_component(value, "org_id")


# ====================================================== _build_namespace_prefix


class TestBuildNamespacePrefix:
    def test_both_ids(self) -> None:
        assert _build_namespace_prefix("acme", "ci") == "acme/ci"

    def test_org_only(self) -> None:
        assert _build_namespace_prefix("acme", "") == "acme"

    def test_corpus_only(self) -> None:
        prefix = _build_namespace_prefix("", "ci")
        # Must not start with a slash or be ambiguous.
        assert not prefix.startswith("/")
        assert "ci" in prefix

    def test_neither(self) -> None:
        assert _build_namespace_prefix("", "") == ""


# ====================================================== NamespacedStepCache


class TestNamespacedStepCacheValidation:
    def test_invalid_org_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="org_id"):
            NamespacedStepCache(_DictCache(), org_id="bad/org")

    def test_invalid_corpus_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="corpus_id"):
            NamespacedStepCache(_DictCache(), corpus_id="bad/corpus")

    def test_slash_in_org_rejected(self) -> None:
        with pytest.raises(ValueError):
            NamespacedStepCache(_DictCache(), org_id="a/b")

    def test_dotdot_rejected(self) -> None:
        with pytest.raises(ValueError):
            NamespacedStepCache(_DictCache(), org_id="../evil")

    def test_empty_ids_are_allowed(self) -> None:
        # Both empty → no namespace → same as bare backend.
        ns = NamespacedStepCache(_DictCache())
        assert ns.org_id == ""
        assert ns.corpus_id == ""

    def test_valid_ids_accepted(self) -> None:
        ns = NamespacedStepCache(_DictCache(), org_id="acme", corpus_id="ci-2024")
        assert ns.org_id == "acme"
        assert ns.corpus_id == "ci-2024"


class TestNamespacedStepCacheDelegation:
    def test_get_delegates_with_namespaced_kind(self) -> None:
        backend = _DictCache()
        ns = NamespacedStepCache(backend, org_id="org1")
        ns.get("tool_call", "sha256:" + "0" * 64)
        assert len(backend.get_calls) == 1
        delegated_kind, _ = backend.get_calls[0]
        assert delegated_kind == "org1/tool_call"

    def test_put_delegates_with_namespaced_kind(self) -> None:
        backend = _DictCache()
        ns = NamespacedStepCache(backend, org_id="org1", corpus_id="corp1")
        ns.put(_make_entry("tool_call"))
        assert len(backend.put_calls) == 1
        stored = backend.put_calls[0]
        assert stored.step_kind == "org1/corp1/tool_call"

    def test_returned_entry_has_logical_kind(self) -> None:
        backend = _DictCache()
        ns = NamespacedStepCache(backend, org_id="org1")
        entry = _make_entry("tool_call")
        ns.put(entry)
        got = ns.get("tool_call", entry.inputs_hash)
        assert got is not None
        assert got.step_kind == "tool_call", (
            f"Expected logical kind 'tool_call', got {got.step_kind!r}"
        )

    def test_outputs_preserved_after_namespace_round_trip(self) -> None:
        backend = _DictCache()
        ns = NamespacedStepCache(backend, org_id="org1")
        entry = _make_entry(outputs={"answer": 42})
        ns.put(entry)
        got = ns.get("tool_call", entry.inputs_hash)
        assert got is not None
        assert got.outputs == {"answer": 42}

    def test_miss_returns_none(self) -> None:
        ns = NamespacedStepCache(_DictCache(), org_id="org1")
        assert ns.get("tool_call", "sha256:" + "9" * 64) is None

    def test_close_delegates_to_backend(self) -> None:
        backend = _DictCache()
        ns = NamespacedStepCache(backend, org_id="org1")
        ns.close()
        assert backend.closed


class TestNamespacedStepCacheIsolation:
    def test_org_isolation(self) -> None:
        """Entries written by org_A are invisible to org_B."""
        backend = _DictCache()
        ns_a = NamespacedStepCache(backend, org_id="org_a")
        ns_b = NamespacedStepCache(backend, org_id="org_b")
        ih = "sha256:" + "1" * 64
        ns_a.put(_make_entry(inputs_hash=ih, outputs={"owner": "a"}))
        got = ns_b.get("tool_call", ih)
        assert got is None, "org_b must not read org_a's entries"

    def test_corpus_isolation(self) -> None:
        """Same org but different corpus → entries are isolated."""
        backend = _DictCache()
        ns_c1 = NamespacedStepCache(backend, org_id="org", corpus_id="corpus1")
        ns_c2 = NamespacedStepCache(backend, org_id="org", corpus_id="corpus2")
        ih = "sha256:" + "2" * 64
        ns_c1.put(_make_entry(inputs_hash=ih, outputs={"corpus": "1"}))
        assert ns_c2.get("tool_call", ih) is None

    def test_same_namespace_deduplicates_across_runs(self) -> None:
        """Two 'runs' sharing the same namespace see the same cache entry."""
        backend = _DictCache()
        run1 = NamespacedStepCache(backend, org_id="org", corpus_id="ci")
        run2 = NamespacedStepCache(backend, org_id="org", corpus_id="ci")
        ih = "sha256:" + "3" * 64
        run1.put(_make_entry(inputs_hash=ih, outputs={"computed_by": "run1"}))
        got = run2.get("tool_call", ih)
        assert got is not None, "run2 must see run1's entry in same namespace"
        assert got.outputs == {"computed_by": "run1"}

    def test_no_namespace_no_isolation(self) -> None:
        """With no namespace, two caches sharing the backend share entries."""
        backend = _DictCache()
        ns1 = NamespacedStepCache(backend)
        ns2 = NamespacedStepCache(backend)
        ih = "sha256:" + "4" * 64
        ns1.put(_make_entry(inputs_hash=ih, outputs={"k": "v"}))
        got = ns2.get("tool_call", ih)
        assert got is not None and got.outputs == {"k": "v"}

    def test_namespace_key_uniqueness(self) -> None:
        """'a'+'b/c' must not collide with 'a/b'+'c'."""
        backend = _DictCache()
        # org_id="a", corpus_id="bc" → prefix "a/bc"
        ns1 = NamespacedStepCache(backend, org_id="a", corpus_id="bc")
        # org_id="ab", corpus_id="c" → prefix "ab/c"
        ns2 = NamespacedStepCache(backend, org_id="ab", corpus_id="c")
        ih = "sha256:" + "5" * 64
        ns1.put(_make_entry(inputs_hash=ih, outputs={"who": "ns1"}))
        # ns2 must NOT see ns1's entry.
        got = ns2.get("tool_call", ih)
        assert got is None or got.outputs != {"who": "ns1"}, (
            "Namespace prefix collision: 'a/bc' and 'ab/c' resolve to the same key"
        )


class TestNamespacedStepCacheDiskIntegration:
    def test_put_get_with_disk_backend(self, tmp_path) -> None:
        cache = NamespacedStepCache(
            DiskStepCache(str(tmp_path)), org_id="testorg", corpus_id="testcorpus"
        )
        ih = "sha256:" + "f" * 64
        cache.put(_make_entry(inputs_hash=ih, outputs={"x": 1}))
        got = cache.get("tool_call", ih)
        assert got is not None
        assert got.outputs == {"x": 1}
        assert got.step_kind == "tool_call"

    def test_directory_layout_includes_namespace(self, tmp_path) -> None:
        cache = NamespacedStepCache(
            DiskStepCache(str(tmp_path)), org_id="myorg", corpus_id="mycorpus"
        )
        ih = "sha256:" + "ab" * 32
        cache.put(_make_entry(inputs_hash=ih))
        # The path should contain the namespace prefix.
        all_files = list(tmp_path.rglob("*.json"))
        assert any("myorg" in str(p) for p in all_files), (
            f"Expected 'myorg' in path, got: {[str(p) for p in all_files]}"
        )


# ====================================================== MultiTierStepCache


class TestMultiTierStepCacheGet:
    def test_l1_hit_returned_directly(self) -> None:
        l1 = _DictCache()
        l2 = _DictCache()
        ih = "sha256:" + "0" * 64
        l1._store[("tool_call", ih)] = _make_entry(inputs_hash=ih, outputs={"tier": "l1"})
        tier = MultiTierStepCache(l1, l2)
        got = tier.get("tool_call", ih)
        assert got is not None and got.outputs == {"tier": "l1"}
        assert len(l2.get_calls) == 0, "L2 must not be consulted on L1 hit"

    def test_l1_miss_falls_back_to_l2(self) -> None:
        l1 = _DictCache()
        l2 = _DictCache()
        ih = "sha256:" + "1" * 64
        l2._store[("tool_call", ih)] = _make_entry(inputs_hash=ih, outputs={"tier": "l2"})
        tier = MultiTierStepCache(l1, l2)
        got = tier.get("tool_call", ih)
        assert got is not None and got.outputs == {"tier": "l2"}

    def test_both_miss_returns_none(self) -> None:
        tier = MultiTierStepCache(_DictCache(), _DictCache())
        assert tier.get("tool_call", "sha256:" + "2" * 64) is None

    def test_l2_hit_promotes_to_l1_when_promote_true(self) -> None:
        l1 = _DictCache()
        l2 = _DictCache()
        ih = "sha256:" + "3" * 64
        l2._store[("tool_call", ih)] = _make_entry(inputs_hash=ih)
        tier = MultiTierStepCache(l1, l2, promote=True)
        tier.get("tool_call", ih)
        # L1 should now have the entry.
        assert len(l1.put_calls) == 1

    def test_l2_hit_does_not_promote_when_promote_false(self) -> None:
        l1 = _DictCache()
        l2 = _DictCache()
        ih = "sha256:" + "4" * 64
        l2._store[("tool_call", ih)] = _make_entry(inputs_hash=ih)
        tier = MultiTierStepCache(l1, l2, promote=False)
        tier.get("tool_call", ih)
        assert len(l1.put_calls) == 0

    def test_promotion_error_does_not_lose_l2_hit(self) -> None:
        """A failing L1 promotion must not turn a cache hit into a miss."""
        l2 = _DictCache()
        ih = "sha256:" + "5" * 64
        l2._store[("tool_call", ih)] = _make_entry(inputs_hash=ih, outputs={"z": 9})
        tier = MultiTierStepCache(_RaisingCache(), l2, promote=True)
        got = tier.get("tool_call", ih)
        assert got is not None and got.outputs == {"z": 9}


class TestMultiTierStepCachePut:
    def test_put_writes_l2_and_l1(self) -> None:
        l1 = _DictCache()
        l2 = _DictCache()
        tier = MultiTierStepCache(l1, l2)
        tier.put(_make_entry())
        assert len(l2.put_calls) == 1
        assert len(l1.put_calls) == 1

    def test_l2_failure_propagates(self) -> None:
        """L2 write failure must raise — L2 is authoritative."""
        l1 = _DictCache()
        tier = MultiTierStepCache(l1, _RaisingCache())
        with pytest.raises(OSError):
            tier.put(_make_entry())

    def test_l1_failure_silently_ignored(self) -> None:
        """L1 write failure must not propagate when L2 succeeds."""
        l2 = _DictCache()
        tier = MultiTierStepCache(_RaisingCache(), l2, write_l1=True)
        # Must not raise even though L1 raises.
        tier.put(_make_entry())
        assert len(l2.put_calls) == 1

    def test_write_l1_false_skips_l1_write(self) -> None:
        l1 = _DictCache()
        l2 = _DictCache()
        tier = MultiTierStepCache(l1, l2, write_l1=False)
        tier.put(_make_entry())
        assert len(l1.put_calls) == 0
        assert len(l2.put_calls) == 1

    def test_write_l1_false_still_promotes_on_get(self) -> None:
        """write_l1=False only blocks direct puts; L2 reads still promote."""
        l1 = _DictCache()
        l2 = _DictCache()
        ih = "sha256:" + "6" * 64
        tier = MultiTierStepCache(l1, l2, promote=True, write_l1=False)
        l2._store[("tool_call", ih)] = _make_entry(inputs_hash=ih)
        tier.get("tool_call", ih)
        # Promotion should still happen.
        assert len(l1.put_calls) == 1


class TestMultiTierStepCacheClose:
    def test_close_calls_both_tiers(self) -> None:
        l1 = _DictCache()
        l2 = _DictCache()
        tier = MultiTierStepCache(l1, l2)
        tier.close()
        assert l1.closed
        assert l2.closed


class TestMultiTierWithDisk:
    def test_disk_l1_disk_l2_full_round_trip(self, tmp_path) -> None:
        l1_dir = tmp_path / "l1"
        l2_dir = tmp_path / "l2"
        l1_dir.mkdir()
        l2_dir.mkdir()
        l1 = DiskStepCache(str(l1_dir))
        l2 = DiskStepCache(str(l2_dir))
        tier = MultiTierStepCache(l1, l2)
        ih = "sha256:" + "c" * 64
        tier.put(_make_entry(inputs_hash=ih, outputs={"data": "ok"}))
        # Read from a fresh cache that shares the same dirs.
        tier2 = MultiTierStepCache(
            DiskStepCache(str(l1_dir)), DiskStepCache(str(l2_dir))
        )
        got = tier2.get("tool_call", ih)
        assert got is not None and got.outputs == {"data": "ok"}


# ====================================================== Composition


class TestComposedNamespacedMultiTier:
    def test_namespaced_wrapping_multi_tier(self, tmp_path) -> None:
        """NamespacedStepCache(MultiTierStepCache(Disk, Disk)) works end-to-end."""
        l1 = DiskStepCache(str(tmp_path / "l1"))
        l2 = DiskStepCache(str(tmp_path / "l2"))
        (tmp_path / "l1").mkdir()
        (tmp_path / "l2").mkdir()
        cache = NamespacedStepCache(
            MultiTierStepCache(l1, l2),
            org_id="myorg",
            corpus_id="mydata",
        )
        ih = "sha256:" + "e" * 64
        cache.put(_make_entry(inputs_hash=ih, outputs={"composed": True}))
        got = cache.get("tool_call", ih)
        assert got is not None
        assert got.outputs == {"composed": True}
        assert got.step_kind == "tool_call"  # logical kind, not namespaced

    def test_two_orgs_via_same_multi_tier_isolated(self, tmp_path) -> None:
        """Two NamespacedStepCaches wrapping the same MultiTierStepCache are isolated."""
        l1 = DiskStepCache(str(tmp_path / "l1"))
        l2 = DiskStepCache(str(tmp_path / "l2"))
        (tmp_path / "l1").mkdir()
        (tmp_path / "l2").mkdir()
        tier = MultiTierStepCache(l1, l2)
        ns_a = NamespacedStepCache(tier, org_id="orgA")
        ns_b = NamespacedStepCache(tier, org_id="orgB")
        ih = "sha256:" + "d" * 64
        ns_a.put(_make_entry(inputs_hash=ih, outputs={"owner": "A"}))
        assert ns_b.get("tool_call", ih) is None


# ====================================================== Public API export


class TestPublicApiExports:
    def test_namespaced_step_cache_exported(self) -> None:
        assert hasattr(stepback, "NamespacedStepCache")
        assert stepback.NamespacedStepCache is NamespacedStepCache

    def test_multi_tier_step_cache_exported(self) -> None:
        assert hasattr(stepback, "MultiTierStepCache")
        assert stepback.MultiTierStepCache is MultiTierStepCache

    def test_both_in_all(self) -> None:
        assert "NamespacedStepCache" in stepback.__all__
        assert "MultiTierStepCache" in stepback.__all__
