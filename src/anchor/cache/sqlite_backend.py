"""SQLite-backed cache backend implementation."""
from __future__ import annotations
import json
import logging
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from anchor.storage.sqlite._connection import SqliteConnectionManager

logger = logging.getLogger(__name__)


class SqliteCacheBackend:
    """SQLite-backed cache with optional TTL expiration.

    Implements the CacheBackend protocol. Uses time.time() for persistence across restarts.
    """

    __slots__ = ("_conn_manager", "_default_ttl")

    def __init__(self, connection_manager: SqliteConnectionManager, default_ttl: float | None = 300.0) -> None:
        self._conn_manager = connection_manager
        self._default_ttl = default_ttl
        from anchor.storage.sqlite._schema import ensure_tables
        ensure_tables(self._conn_manager.get_connection())

    def get(self, key: str) -> Any | None:
        conn = self._conn_manager.get_connection()
        row = conn.execute(
            "SELECT value_json, expires_at FROM cache_entries WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        value_json, expires_at = row["value_json"], row["expires_at"]
        if expires_at is not None and time.time() >= expires_at:
            conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            conn.commit()
            return None
        return json.loads(value_json)

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        now = time.time()
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = (now + effective_ttl) if effective_ttl is not None else None
        conn = self._conn_manager.get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO cache_entries (key, value_json, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (key, json.dumps(value, default=str), now, expires_at),
        )
        conn.commit()

    def invalidate(self, key: str) -> None:
        conn = self._conn_manager.get_connection()
        conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
        conn.commit()

    def clear(self) -> None:
        conn = self._conn_manager.get_connection()
        conn.execute("DELETE FROM cache_entries")
        conn.commit()

    def __repr__(self) -> str:
        return f"SqliteCacheBackend(default_ttl={self._default_ttl})"
