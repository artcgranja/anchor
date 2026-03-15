"""PostgreSQL-backed GraphStore implementation."""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from anchor.storage.postgres._connection import PostgresConnectionManager


class PostgresGraphStore:
    """Async PostgreSQL-backed graph store. Implements AsyncGraphStore protocol.

    Uses WITH RECURSIVE CTE for get_neighbors with cycle prevention via path array.
    """

    __slots__ = ("_conn_manager",)

    def __init__(self, conn_manager: PostgresConnectionManager) -> None:
        self._conn_manager = conn_manager

    async def add_node(self, node_id: str, metadata: dict[str, Any] | None = None) -> None:
        async with self._conn_manager.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT metadata FROM graph_nodes WHERE node_id = $1", node_id
            )
            if existing is not None:
                if metadata:
                    merged = json.loads(existing["metadata"]) if isinstance(existing["metadata"], str) else dict(existing["metadata"])
                    merged.update(metadata)
                    await conn.execute(
                        "UPDATE graph_nodes SET metadata = $1 WHERE node_id = $2",
                        json.dumps(merged), node_id,
                    )
            else:
                await conn.execute(
                    "INSERT INTO graph_nodes (node_id, metadata) VALUES ($1, $2)",
                    node_id, json.dumps(metadata or {}),
                )

    async def add_edge(
        self, source: str, relation: str, target: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        async with self._conn_manager.acquire() as conn:
            for nid in (source, target):
                await conn.execute(
                    "INSERT INTO graph_nodes (node_id, metadata) VALUES ($1, '{}') "
                    "ON CONFLICT (node_id) DO NOTHING",
                    nid,
                )
            await conn.execute(
                "INSERT INTO graph_edges (source, relation, target, metadata) "
                "VALUES ($1, $2, $3, $4) "
                "ON CONFLICT (source, relation, target) DO NOTHING",
                source, relation, target, json.dumps(metadata or {}),
            )

    async def get_neighbors(
        self, node_id: str, max_depth: int = 1,
        relation_filter: str | list[str] | None = None,
    ) -> list[str]:
        async with self._conn_manager.acquire() as conn:
            if await conn.fetchrow("SELECT 1 FROM graph_nodes WHERE node_id = $1", node_id) is None:
                return []
            if relation_filter is not None:
                rels = [relation_filter] if isinstance(relation_filter, str) else list(relation_filter)
                query = """
                    WITH RECURSIVE reachable(node, depth, path) AS (
                        SELECT $1::text, 0, ARRAY[$1::text]
                      UNION ALL
                        SELECT neighbor, r.depth + 1, r.path || neighbor
                        FROM reachable r
                        CROSS JOIN LATERAL (
                            SELECT e.target AS neighbor FROM graph_edges e
                            WHERE e.source = r.node AND e.relation = ANY($3)
                            UNION
                            SELECT e.source AS neighbor FROM graph_edges e
                            WHERE e.target = r.node AND e.relation = ANY($3)
                        ) neighbors
                        WHERE r.depth < $2
                          AND NOT neighbor = ANY(r.path)
                    )
                    SELECT DISTINCT node FROM reachable WHERE node != $1
                """
                rows = await conn.fetch(query, node_id, max_depth, rels)
            else:
                query = """
                    WITH RECURSIVE reachable(node, depth, path) AS (
                        SELECT $1::text, 0, ARRAY[$1::text]
                      UNION ALL
                        SELECT neighbor, r.depth + 1, r.path || neighbor
                        FROM reachable r
                        CROSS JOIN LATERAL (
                            SELECT e.target AS neighbor FROM graph_edges e
                            WHERE e.source = r.node
                            UNION
                            SELECT e.source AS neighbor FROM graph_edges e
                            WHERE e.target = r.node
                        ) neighbors
                        WHERE r.depth < $2
                          AND NOT neighbor = ANY(r.path)
                    )
                    SELECT DISTINCT node FROM reachable WHERE node != $1
                """
                rows = await conn.fetch(query, node_id, max_depth)
            return [r["node"] for r in rows]

    async def get_edges(self, node_id: str) -> list[tuple[str, str, str]]:
        async with self._conn_manager.acquire() as conn:
            rows = await conn.fetch(
                "SELECT source, relation, target FROM graph_edges WHERE source = $1 OR target = $1",
                node_id,
            )
            return [(r["source"], r["relation"], r["target"]) for r in rows]

    async def get_node_metadata(self, node_id: str) -> dict[str, Any] | None:
        async with self._conn_manager.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT metadata FROM graph_nodes WHERE node_id = $1", node_id
            )
            if row is None:
                return None
            meta = row["metadata"]
            return json.loads(meta) if isinstance(meta, str) else dict(meta)

    async def link_memory(self, node_id: str, memory_id: str) -> None:
        async with self._conn_manager.acquire() as conn:
            if await conn.fetchrow("SELECT 1 FROM graph_nodes WHERE node_id = $1", node_id) is None:
                msg = f"Entity '{node_id}' does not exist in the graph"
                raise KeyError(msg)
            await conn.execute(
                "INSERT INTO graph_memory_links (node_id, memory_id) VALUES ($1, $2) "
                "ON CONFLICT DO NOTHING",
                node_id, memory_id,
            )

    async def get_memory_ids(self, node_id: str, max_depth: int = 1) -> list[str]:
        all_nodes = [node_id, *(await self.get_neighbors(node_id, max_depth=max_depth))]
        async with self._conn_manager.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT memory_id FROM graph_memory_links WHERE node_id = ANY($1::text[])",
                all_nodes,
            )
            return [row["memory_id"] for row in rows]

    async def remove_node(self, node_id: str) -> None:
        async with self._conn_manager.acquire() as conn:
            await conn.execute("DELETE FROM graph_edges WHERE source = $1 OR target = $1", node_id)
            await conn.execute("DELETE FROM graph_memory_links WHERE node_id = $1", node_id)
            await conn.execute("DELETE FROM graph_nodes WHERE node_id = $1", node_id)

    async def remove_edge(self, source: str, relation: str, target: str) -> bool:
        async with self._conn_manager.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM graph_edges WHERE source = $1 AND relation = $2 AND target = $3",
                source, relation, target,
            )
            # asyncpg returns status like "DELETE N"
            try:
                return int(result.split()[-1]) > 0
            except (ValueError, IndexError):
                return False

    async def list_nodes(self) -> list[str]:
        async with self._conn_manager.acquire() as conn:
            rows = await conn.fetch("SELECT node_id FROM graph_nodes")
            return [r["node_id"] for r in rows]

    async def list_edges(self) -> list[tuple[str, str, str]]:
        async with self._conn_manager.acquire() as conn:
            rows = await conn.fetch("SELECT source, relation, target FROM graph_edges")
            return [(r["source"], r["relation"], r["target"]) for r in rows]

    async def clear(self) -> None:
        async with self._conn_manager.acquire() as conn:
            await conn.execute("DELETE FROM graph_memory_links")
            await conn.execute("DELETE FROM graph_edges")
            await conn.execute("DELETE FROM graph_nodes")

    def __repr__(self) -> str:
        return "PostgresGraphStore()"
