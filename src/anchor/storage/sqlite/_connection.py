"""SQLite connection manager with WAL mode and thread-local connections."""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)


class SqliteConnectionManager:
    """Manages SQLite connections with WAL mode and thread-local storage.

    Each thread gets its own connection to allow concurrent reads.
    WAL mode enables readers to proceed without blocking writers.

    Example::

        mgr = SqliteConnectionManager("data.db")
        conn = mgr.get_connection()
        conn.execute("SELECT 1")
        mgr.close()
    """

    __slots__ = ("_db_path", "_local", "_wal_mode")

    def __init__(self, db_path: str | Path, *, wal_mode: bool = True) -> None:
        self._db_path = Path(db_path).resolve()
        self._wal_mode = wal_mode
        self._local = threading.local()

    @property
    def db_path(self) -> Path:
        """Return the resolved database file path."""
        return self._db_path

    def get_connection(self) -> sqlite3.Connection:
        """Return a thread-local connection, creating one if needed."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            if self._wal_mode:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
            logger.debug(
                "Opened SQLite connection to %s (thread=%s)",
                self._db_path,
                threading.current_thread().name,
            )
        return conn

    async def get_async_connection(self) -> aiosqlite.Connection:
        """Return an aiosqlite connection.

        Raises ``ImportError`` if ``aiosqlite`` is not installed.
        """
        try:
            import aiosqlite as _aiosqlite
        except ImportError as e:
            msg = (
                "aiosqlite is required for async SQLite operations. "
                "Install it with: pip install astro-anchor[sqlite]"
            )
            raise ImportError(msg) from e

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await _aiosqlite.connect(str(self._db_path))
        conn.row_factory = _aiosqlite.Row
        await conn.execute("PRAGMA busy_timeout=5000")
        await conn.execute("PRAGMA foreign_keys=ON")
        if self._wal_mode:
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def close(self) -> None:
        """Close the thread-local connection if it exists."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(db_path={self._db_path!s})"
