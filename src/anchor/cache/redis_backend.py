"""Redis-backed cache backend implementation."""
from __future__ import annotations
import json
import logging
import math
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from anchor.storage.redis._connection import RedisConnectionManager

logger = logging.getLogger(__name__)


class RedisCacheBackend:
    """Redis-backed cache. Implements CacheBackend protocol.

    Uses JSON serialization. Leverages native Redis SETEX for TTL.
    """

    __slots__ = ("_conn_manager", "_default_ttl", "_key_prefix")

    def __init__(
        self,
        connection_manager: RedisConnectionManager,
        default_ttl: float | None = 300.0,
        key_prefix: str = "cache:",
    ) -> None:
        self._conn_manager = connection_manager
        self._default_ttl = default_ttl
        self._key_prefix = key_prefix

    def _full_key(self, key: str) -> str:
        return f"{self._conn_manager.prefix}{self._key_prefix}{key}"

    def get(self, key: str) -> Any | None:
        client = self._conn_manager.get_client()
        raw = client.get(self._full_key(key))
        if raw is None:
            return None
        return json.loads(raw)

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        client = self._conn_manager.get_client()
        serialized = json.dumps(value, default=str)
        full_key = self._full_key(key)
        if effective_ttl is not None:
            client.setex(full_key, max(1, math.ceil(effective_ttl)), serialized)
        else:
            client.set(full_key, serialized)

    def invalidate(self, key: str) -> None:
        client = self._conn_manager.get_client()
        client.delete(self._full_key(key))

    def clear(self) -> None:
        client = self._conn_manager.get_client()
        pattern = f"{self._conn_manager.prefix}{self._key_prefix}*"
        batch: list[str] = []
        for redis_key in client.scan_iter(match=pattern):
            batch.append(redis_key)
            if len(batch) >= 1000:
                client.delete(*batch)
                batch.clear()
        if batch:
            client.delete(*batch)

    def __repr__(self) -> str:
        return f"RedisCacheBackend(default_ttl={self._default_ttl}, key_prefix={self._key_prefix!r})"
