"""PostgreSQL-backed ConversationStore implementation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from anchor.models.memory import ConversationTurn, SummaryTier

if TYPE_CHECKING:
    from anchor.storage.postgres._connection import PostgresConnectionManager


class PostgresConversationStore:
    """Async PostgreSQL-backed conversation store. Implements AsyncConversationStore protocol."""

    __slots__ = ("_conn_manager",)

    def __init__(self, conn_manager: PostgresConnectionManager) -> None:
        self._conn_manager = conn_manager

    async def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        async with self._conn_manager.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COALESCE(MAX(turn_index), -1) + 1 AS next_idx "
                "FROM conversation_turns WHERE session_id = $1",
                session_id,
            )
            turn_index = row["next_idx"]
            await conn.execute(
                "INSERT INTO conversation_turns "
                "(session_id, turn_index, role, content, token_count, metadata, created_at) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7)",
                session_id, turn_index, str(turn.role), turn.content,
                turn.token_count, json.dumps(turn.metadata, default=str),
                turn.timestamp,
            )

    async def load_turns(self, session_id: str, limit: int | None = None) -> list[ConversationTurn]:
        async with self._conn_manager.acquire() as conn:
            if limit is not None:
                rows = await conn.fetch(
                    "SELECT * FROM ("
                    "  SELECT * FROM conversation_turns WHERE session_id = $1 "
                    "  ORDER BY turn_index DESC LIMIT $2"
                    ") sub ORDER BY turn_index ASC",
                    session_id, limit,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM conversation_turns WHERE session_id = $1 "
                    "ORDER BY turn_index ASC",
                    session_id,
                )
            return [
                ConversationTurn(
                    role=r["role"],
                    content=r["content"],
                    token_count=r["token_count"],
                    metadata=json.loads(r["metadata"]) if isinstance(r["metadata"], str) else dict(r["metadata"]),
                    timestamp=r["created_at"],
                )
                for r in rows
            ]

    async def save_summary_tiers(self, session_id: str, tiers: dict[int, SummaryTier | None]) -> None:
        async with self._conn_manager.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM summary_tiers WHERE session_id = $1", session_id)
                for level, tier in tiers.items():
                    if tier is not None:
                        await conn.execute(
                            "INSERT INTO summary_tiers "
                            "(session_id, tier_level, content, token_count, source_turn_count, created_at, updated_at) "
                            "VALUES ($1, $2, $3, $4, $5, $6, $7)",
                            session_id, tier.level, tier.content, tier.token_count,
                            tier.source_turn_count, tier.created_at, tier.updated_at,
                        )

    async def load_summary_tiers(self, session_id: str) -> dict[int, SummaryTier | None]:
        async with self._conn_manager.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM summary_tiers WHERE session_id = $1", session_id
            )
            result: dict[int, SummaryTier | None] = {1: None, 2: None, 3: None}
            for r in rows:
                result[r["tier_level"]] = SummaryTier(
                    level=r["tier_level"],
                    content=r["content"],
                    token_count=r["token_count"],
                    source_turn_count=r["source_turn_count"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                )
            return result

    async def truncate_turns(self, session_id: str, keep_last: int) -> None:
        async with self._conn_manager.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT MAX(turn_index) AS max_idx FROM conversation_turns WHERE session_id = $1",
                session_id,
            )
            if row["max_idx"] is None:
                return
            cutoff = row["max_idx"] - keep_last + 1
            await conn.execute(
                "DELETE FROM conversation_turns WHERE session_id = $1 AND turn_index < $2",
                session_id, cutoff,
            )

    async def delete_session(self, session_id: str) -> bool:
        async with self._conn_manager.acquire() as conn:
            r1 = await conn.execute("DELETE FROM conversation_turns WHERE session_id = $1", session_id)
            r2 = await conn.execute("DELETE FROM summary_tiers WHERE session_id = $1", session_id)
            # asyncpg returns status like "DELETE N"
            try:
                return (int(r1.split()[-1]) + int(r2.split()[-1])) > 0
            except (ValueError, IndexError):
                return False

    async def list_sessions(self) -> list[str]:
        async with self._conn_manager.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT session_id FROM conversation_turns "
                "UNION SELECT DISTINCT session_id FROM summary_tiers"
            )
            return [r["session_id"] for r in rows]

    async def clear(self) -> None:
        async with self._conn_manager.acquire() as conn:
            await conn.execute("DELETE FROM conversation_turns")
            await conn.execute("DELETE FROM summary_tiers")

    def __repr__(self) -> str:
        return "PostgresConversationStore()"
