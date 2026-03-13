"""SQLite-backed ContextStore implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anchor.models.context import ContextItem
from anchor.storage._serialization import context_item_to_row, row_to_context_item

if TYPE_CHECKING:
    from anchor.storage.sqlite._connection import SqliteConnectionManager


class SqliteContextStore:
    """SQLite-backed context store. Implements the ContextStore protocol."""

    __slots__ = ("_conn_manager",)

    def __init__(self, conn_manager: SqliteConnectionManager) -> None:
        self._conn_manager = conn_manager

    def add(self, item: ContextItem) -> None:
        row = context_item_to_row(item)
        conn = self._conn_manager.get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO context_items "
            "(id, content, source, score, priority, token_count, "
            "metadata_json, created_at) "
            "VALUES (:id, :content, :source, :score, :priority, "
            ":token_count, :metadata_json, :created_at)",
            row,
        )
        conn.commit()

    def get(self, item_id: str) -> ContextItem | None:
        conn = self._conn_manager.get_connection()
        row = conn.execute(
            "SELECT * FROM context_items WHERE id = ?", (item_id,)
        ).fetchone()
        if row is None:
            return None
        return row_to_context_item(row)

    def get_all(self) -> list[ContextItem]:
        conn = self._conn_manager.get_connection()
        rows = conn.execute("SELECT * FROM context_items").fetchall()
        return [row_to_context_item(r) for r in rows]

    def delete(self, item_id: str) -> bool:
        conn = self._conn_manager.get_connection()
        cursor = conn.execute(
            "DELETE FROM context_items WHERE id = ?", (item_id,)
        )
        conn.commit()
        return cursor.rowcount > 0

    def clear(self) -> None:
        conn = self._conn_manager.get_connection()
        conn.execute("DELETE FROM context_items")
        conn.commit()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(db={self._conn_manager.db_path!s})"


class AsyncSqliteContextStore:
    """Async SQLite-backed context store.

    Implements the AsyncContextStore protocol.
    """

    __slots__ = ("_conn_manager",)

    def __init__(self, conn_manager: SqliteConnectionManager) -> None:
        self._conn_manager = conn_manager

    async def add(self, item: ContextItem) -> None:
        row = context_item_to_row(item)
        async with await self._conn_manager.get_async_connection() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO context_items "
                "(id, content, source, score, priority, token_count, "
                "metadata_json, created_at) "
                "VALUES (:id, :content, :source, :score, :priority, "
                ":token_count, :metadata_json, :created_at)",
                row,
            )
            await conn.commit()

    async def get(self, item_id: str) -> ContextItem | None:
        async with await self._conn_manager.get_async_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM context_items WHERE id = ?", (item_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return row_to_context_item(row)

    async def get_all(self) -> list[ContextItem]:
        async with await self._conn_manager.get_async_connection() as conn:
            cursor = await conn.execute("SELECT * FROM context_items")
            rows = await cursor.fetchall()
            return [row_to_context_item(r) for r in rows]

    async def delete(self, item_id: str) -> bool:
        async with await self._conn_manager.get_async_connection() as conn:
            cursor = await conn.execute(
                "DELETE FROM context_items WHERE id = ?", (item_id,)
            )
            await conn.commit()
            return cursor.rowcount > 0

    async def clear(self) -> None:
        async with await self._conn_manager.get_async_connection() as conn:
            await conn.execute("DELETE FROM context_items")
            await conn.commit()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(db={self._conn_manager.db_path!s})"
