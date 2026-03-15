"""SQLite-backed graph store implementation."""

from __future__ import annotations

import json
import logging
from collections import deque
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from anchor.storage.sqlite._connection import SqliteConnectionManager

from anchor.storage.sqlite._schema import ensure_tables, ensure_tables_async

logger = logging.getLogger(__name__)


class SqliteGraphStore:
    """SQLite-backed graph store. Implements GraphStore protocol."""

    __slots__ = ("_conn_manager",)

    def __init__(self, connection_manager: SqliteConnectionManager) -> None:
        self._conn_manager = connection_manager
        ensure_tables(self._conn_manager.get_connection())

    def add_node(self, node_id: str, metadata: dict[str, Any] | None = None) -> None:
        conn = self._conn_manager.get_connection()
        existing = conn.execute(
            "SELECT metadata_json FROM graph_nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        if existing is not None:
            if metadata:
                merged = json.loads(existing["metadata_json"])
                merged.update(metadata)
                conn.execute(
                    "UPDATE graph_nodes SET metadata_json = ? WHERE node_id = ?",
                    (json.dumps(merged), node_id),
                )
        else:
            conn.execute(
                "INSERT INTO graph_nodes (node_id, metadata_json) VALUES (?, ?)",
                (node_id, json.dumps(metadata or {})),
            )
        conn.commit()

    def add_edge(
        self, source: str, relation: str, target: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        conn = self._conn_manager.get_connection()
        for nid in (source, target):
            conn.execute(
                "INSERT OR IGNORE INTO graph_nodes (node_id, metadata_json) VALUES (?, '{}')",
                (nid,),
            )
        conn.execute(
            "INSERT OR IGNORE INTO graph_edges (source, relation, target, metadata_json) "
            "VALUES (?, ?, ?, ?)",
            (source, relation, target, json.dumps(metadata or {})),
        )
        conn.commit()

    def get_neighbors(
        self, node_id: str, max_depth: int = 1,
        relation_filter: str | list[str] | None = None,
    ) -> list[str]:
        conn = self._conn_manager.get_connection()
        if conn.execute("SELECT 1 FROM graph_nodes WHERE node_id = ?", (node_id,)).fetchone() is None:
            return []
        visited: set[str] = {node_id}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        result: list[str] = []
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            if relation_filter is not None:
                rels = [relation_filter] if isinstance(relation_filter, str) else list(relation_filter)
                placeholders = ",".join("?" * len(rels))
                rows = conn.execute(
                    f"SELECT target FROM graph_edges WHERE source = ? AND relation IN ({placeholders}) "
                    f"UNION SELECT source FROM graph_edges WHERE target = ? AND relation IN ({placeholders})",
                    (current, *rels, current, *rels),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT target FROM graph_edges WHERE source = ? "
                    "UNION SELECT source FROM graph_edges WHERE target = ?",
                    (current, current),
                ).fetchall()
            for row in rows:
                neighbor = row[0]
                if neighbor not in visited:
                    visited.add(neighbor)
                    result.append(neighbor)
                    queue.append((neighbor, depth + 1))
        return result

    def get_edges(self, node_id: str) -> list[tuple[str, str, str]]:
        conn = self._conn_manager.get_connection()
        rows = conn.execute(
            "SELECT source, relation, target FROM graph_edges WHERE source = ? OR target = ?",
            (node_id, node_id),
        ).fetchall()
        return [(r["source"], r["relation"], r["target"]) for r in rows]

    def get_node_metadata(self, node_id: str) -> dict[str, Any] | None:
        conn = self._conn_manager.get_connection()
        row = conn.execute(
            "SELECT metadata_json FROM graph_nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["metadata_json"])

    def link_memory(self, node_id: str, memory_id: str) -> None:
        conn = self._conn_manager.get_connection()
        if conn.execute("SELECT 1 FROM graph_nodes WHERE node_id = ?", (node_id,)).fetchone() is None:
            msg = f"Entity '{node_id}' does not exist in the graph"
            raise KeyError(msg)
        conn.execute(
            "INSERT OR IGNORE INTO graph_memory_links (node_id, memory_id) VALUES (?, ?)",
            (node_id, memory_id),
        )
        conn.commit()

    def get_memory_ids(self, node_id: str, max_depth: int = 1) -> list[str]:
        all_nodes = [node_id, *self.get_neighbors(node_id, max_depth=max_depth)]
        conn = self._conn_manager.get_connection()
        result: list[str] = []
        seen: set[str] = set()
        for nid in all_nodes:
            rows = conn.execute(
                "SELECT memory_id FROM graph_memory_links WHERE node_id = ?", (nid,)
            ).fetchall()
            for row in rows:
                mid = row["memory_id"]
                if mid not in seen:
                    seen.add(mid)
                    result.append(mid)
        return result

    def remove_node(self, node_id: str) -> None:
        conn = self._conn_manager.get_connection()
        conn.execute("DELETE FROM graph_edges WHERE source = ? OR target = ?", (node_id, node_id))
        conn.execute("DELETE FROM graph_memory_links WHERE node_id = ?", (node_id,))
        conn.execute("DELETE FROM graph_nodes WHERE node_id = ?", (node_id,))
        conn.commit()

    def remove_edge(self, source: str, relation: str, target: str) -> bool:
        conn = self._conn_manager.get_connection()
        cursor = conn.execute(
            "DELETE FROM graph_edges WHERE source = ? AND relation = ? AND target = ?",
            (source, relation, target),
        )
        conn.commit()
        return cursor.rowcount > 0

    def list_nodes(self) -> list[str]:
        conn = self._conn_manager.get_connection()
        rows = conn.execute("SELECT node_id FROM graph_nodes").fetchall()
        return [r["node_id"] for r in rows]

    def list_edges(self) -> list[tuple[str, str, str]]:
        conn = self._conn_manager.get_connection()
        rows = conn.execute("SELECT source, relation, target FROM graph_edges").fetchall()
        return [(r["source"], r["relation"], r["target"]) for r in rows]

    def clear(self) -> None:
        conn = self._conn_manager.get_connection()
        conn.execute("DELETE FROM graph_memory_links")
        conn.execute("DELETE FROM graph_edges")
        conn.execute("DELETE FROM graph_nodes")
        conn.commit()

    def __repr__(self) -> str:
        return "SqliteGraphStore()"


class AsyncSqliteGraphStore:
    """Async SQLite-backed graph store. Implements AsyncGraphStore protocol."""

    __slots__ = ("_conn_manager",)

    def __init__(self, connection_manager: SqliteConnectionManager) -> None:
        self._conn_manager = connection_manager

    async def _ensure_tables(self) -> None:
        conn = await self._conn_manager.get_async_connection()
        await ensure_tables_async(conn)

    async def add_node(self, node_id: str, metadata: dict[str, Any] | None = None) -> None:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute(
            "SELECT metadata_json FROM graph_nodes WHERE node_id = ?", (node_id,)
        )
        existing = await cursor.fetchone()
        if existing is not None:
            if metadata:
                merged = json.loads(existing["metadata_json"])
                merged.update(metadata)
                await conn.execute(
                    "UPDATE graph_nodes SET metadata_json = ? WHERE node_id = ?",
                    (json.dumps(merged), node_id),
                )
        else:
            await conn.execute(
                "INSERT INTO graph_nodes (node_id, metadata_json) VALUES (?, ?)",
                (node_id, json.dumps(metadata or {})),
            )
        await conn.commit()

    async def add_edge(
        self, source: str, relation: str, target: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        conn = await self._conn_manager.get_async_connection()
        for nid in (source, target):
            await conn.execute(
                "INSERT OR IGNORE INTO graph_nodes (node_id, metadata_json) VALUES (?, '{}')",
                (nid,),
            )
        await conn.execute(
            "INSERT OR IGNORE INTO graph_edges (source, relation, target, metadata_json) "
            "VALUES (?, ?, ?, ?)",
            (source, relation, target, json.dumps(metadata or {})),
        )
        await conn.commit()

    async def get_neighbors(
        self, node_id: str, max_depth: int = 1,
        relation_filter: str | list[str] | None = None,
    ) -> list[str]:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute("SELECT 1 FROM graph_nodes WHERE node_id = ?", (node_id,))
        if await cursor.fetchone() is None:
            return []
        visited: set[str] = {node_id}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        result: list[str] = []
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            if relation_filter is not None:
                rels = [relation_filter] if isinstance(relation_filter, str) else list(relation_filter)
                placeholders = ",".join("?" * len(rels))
                cursor = await conn.execute(
                    f"SELECT target FROM graph_edges WHERE source = ? AND relation IN ({placeholders}) "
                    f"UNION SELECT source FROM graph_edges WHERE target = ? AND relation IN ({placeholders})",
                    (current, *rels, current, *rels),
                )
            else:
                cursor = await conn.execute(
                    "SELECT target FROM graph_edges WHERE source = ? "
                    "UNION SELECT source FROM graph_edges WHERE target = ?",
                    (current, current),
                )
            rows = await cursor.fetchall()
            for row in rows:
                neighbor = row[0]
                if neighbor not in visited:
                    visited.add(neighbor)
                    result.append(neighbor)
                    queue.append((neighbor, depth + 1))
        return result

    async def get_edges(self, node_id: str) -> list[tuple[str, str, str]]:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute(
            "SELECT source, relation, target FROM graph_edges WHERE source = ? OR target = ?",
            (node_id, node_id),
        )
        rows = await cursor.fetchall()
        return [(r["source"], r["relation"], r["target"]) for r in rows]

    async def get_node_metadata(self, node_id: str) -> dict[str, Any] | None:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute(
            "SELECT metadata_json FROM graph_nodes WHERE node_id = ?", (node_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return json.loads(row["metadata_json"])

    async def link_memory(self, node_id: str, memory_id: str) -> None:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute("SELECT 1 FROM graph_nodes WHERE node_id = ?", (node_id,))
        if await cursor.fetchone() is None:
            msg = f"Entity '{node_id}' does not exist in the graph"
            raise KeyError(msg)
        await conn.execute(
            "INSERT OR IGNORE INTO graph_memory_links (node_id, memory_id) VALUES (?, ?)",
            (node_id, memory_id),
        )
        await conn.commit()

    async def get_memory_ids(self, node_id: str, max_depth: int = 1) -> list[str]:
        all_nodes = [node_id, *(await self.get_neighbors(node_id, max_depth=max_depth))]
        conn = await self._conn_manager.get_async_connection()
        result: list[str] = []
        seen: set[str] = set()
        for nid in all_nodes:
            cursor = await conn.execute(
                "SELECT memory_id FROM graph_memory_links WHERE node_id = ?", (nid,)
            )
            rows = await cursor.fetchall()
            for row in rows:
                mid = row["memory_id"]
                if mid not in seen:
                    seen.add(mid)
                    result.append(mid)
        return result

    async def remove_node(self, node_id: str) -> None:
        conn = await self._conn_manager.get_async_connection()
        await conn.execute("DELETE FROM graph_edges WHERE source = ? OR target = ?", (node_id, node_id))
        await conn.execute("DELETE FROM graph_memory_links WHERE node_id = ?", (node_id,))
        await conn.execute("DELETE FROM graph_nodes WHERE node_id = ?", (node_id,))
        await conn.commit()

    async def remove_edge(self, source: str, relation: str, target: str) -> bool:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute(
            "DELETE FROM graph_edges WHERE source = ? AND relation = ? AND target = ?",
            (source, relation, target),
        )
        await conn.commit()
        return cursor.rowcount > 0

    async def list_nodes(self) -> list[str]:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute("SELECT node_id FROM graph_nodes")
        rows = await cursor.fetchall()
        return [r["node_id"] for r in rows]

    async def list_edges(self) -> list[tuple[str, str, str]]:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute("SELECT source, relation, target FROM graph_edges")
        rows = await cursor.fetchall()
        return [(r["source"], r["relation"], r["target"]) for r in rows]

    async def clear(self) -> None:
        conn = await self._conn_manager.get_async_connection()
        await conn.execute("DELETE FROM graph_memory_links")
        await conn.execute("DELETE FROM graph_edges")
        await conn.execute("DELETE FROM graph_nodes")
        await conn.commit()

    def __repr__(self) -> str:
        return "AsyncSqliteGraphStore()"
