"""Tests for RedisCacheBackend with mocked Redis client."""
from unittest.mock import MagicMock
import pytest
from anchor.cache.redis_backend import RedisCacheBackend
from anchor.protocols.cache import CacheBackend


@pytest.fixture
def mock_conn_manager():
    mgr = MagicMock()
    mgr.prefix = "anchor:"
    client = MagicMock()
    client.get.return_value = None
    client.scan_iter.return_value = iter([])
    mgr.get_client.return_value = client
    return mgr


@pytest.fixture
def cache(mock_conn_manager):
    return RedisCacheBackend(mock_conn_manager, default_ttl=300.0)


class TestRedisCacheProtocol:
    def test_satisfies_protocol(self, cache):
        assert isinstance(cache, CacheBackend)


class TestRedisCacheOperations:
    def test_set_calls_redis_setex(self, cache, mock_conn_manager):
        cache.set("mykey", {"data": 1}, ttl=60.0)
        client = mock_conn_manager.get_client()
        client.setex.assert_called_once()
        args = client.setex.call_args
        assert "anchor:cache:mykey" in str(args)

    def test_set_without_ttl_uses_default(self, cache, mock_conn_manager):
        cache.set("mykey", "value")
        client = mock_conn_manager.get_client()
        client.setex.assert_called_once()

    def test_set_with_no_ttl_uses_plain_set(self, mock_conn_manager):
        cache = RedisCacheBackend(mock_conn_manager, default_ttl=None)
        cache.set("mykey", "value")
        client = mock_conn_manager.get_client()
        client.set.assert_called_once()

    def test_get_returns_deserialized(self, cache, mock_conn_manager):
        import json
        client = mock_conn_manager.get_client()
        client.get.return_value = json.dumps({"data": 1}).encode()
        result = cache.get("mykey")
        assert result == {"data": 1}

    def test_get_returns_none_for_missing(self, cache, mock_conn_manager):
        client = mock_conn_manager.get_client()
        client.get.return_value = None
        assert cache.get("mykey") is None

    def test_invalidate_calls_delete(self, cache, mock_conn_manager):
        cache.invalidate("mykey")
        client = mock_conn_manager.get_client()
        client.delete.assert_called_once_with("anchor:cache:mykey")

    def test_clear_scans_and_deletes(self, cache, mock_conn_manager):
        client = mock_conn_manager.get_client()
        client.scan_iter.return_value = iter([b"anchor:cache:k1", b"anchor:cache:k2"])
        cache.clear()
        client.scan_iter.assert_called_once()
        # Keys are batched into a single delete call
        client.delete.assert_called_once_with(b"anchor:cache:k1", b"anchor:cache:k2")
