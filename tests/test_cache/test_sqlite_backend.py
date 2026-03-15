"""Tests for SqliteCacheBackend."""
import time
import pytest
from anchor.cache.sqlite_backend import SqliteCacheBackend
from anchor.protocols.cache import CacheBackend
from anchor.storage.sqlite import SqliteConnectionManager


@pytest.fixture
def cache():
    conn_mgr = SqliteConnectionManager(":memory:")
    return SqliteCacheBackend(conn_mgr, default_ttl=1.0)


class TestSqliteCacheProtocol:
    def test_satisfies_protocol(self, cache):
        assert isinstance(cache, CacheBackend)


class TestSqliteCacheOperations:
    def test_set_and_get(self, cache):
        cache.set("key1", {"data": "value"})
        assert cache.get("key1") == {"data": "value"}

    def test_get_missing_key(self, cache):
        assert cache.get("nonexistent") is None

    def test_invalidate(self, cache):
        cache.set("key1", "value")
        cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_clear(self, cache):
        cache.set("key1", "v1")
        cache.set("key2", "v2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_overwrite_existing_key(self, cache):
        cache.set("key1", "old")
        cache.set("key1", "new")
        assert cache.get("key1") == "new"

    def test_ttl_expiration(self, cache):
        cache.set("key1", "value", ttl=0.01)
        time.sleep(0.02)
        assert cache.get("key1") is None

    def test_no_ttl(self, cache):
        conn_mgr = SqliteConnectionManager(":memory:")
        no_ttl_cache = SqliteCacheBackend(conn_mgr, default_ttl=None)
        no_ttl_cache.set("key1", "value")
        assert no_ttl_cache.get("key1") == "value"

    def test_stores_various_types(self, cache):
        cache.set("str", "hello")
        cache.set("int", 42)
        cache.set("list", [1, 2, 3])
        cache.set("dict", {"a": 1})
        assert cache.get("str") == "hello"
        assert cache.get("int") == 42
        assert cache.get("list") == [1, 2, 3]
        assert cache.get("dict") == {"a": 1}
