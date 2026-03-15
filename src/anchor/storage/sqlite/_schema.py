"""SQLite schema definitions and table creation."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

_TABLES: list[str] = [
    """CREATE TABLE IF NOT EXISTS context_items (
        id          TEXT PRIMARY KEY,
        content     TEXT NOT NULL,
        source      TEXT NOT NULL,
        score       REAL NOT NULL DEFAULT 0.0,
        priority    INTEGER NOT NULL DEFAULT 5,
        token_count INTEGER NOT NULL DEFAULT 0,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at  TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS embeddings (
        item_id       TEXT PRIMARY KEY,
        embedding_blob BLOB NOT NULL,
        metadata_json  TEXT NOT NULL DEFAULT '{}'
    )""",
    """CREATE TABLE IF NOT EXISTS documents (
        doc_id        TEXT PRIMARY KEY,
        content       TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}'
    )""",
    """CREATE TABLE IF NOT EXISTS memory_entries (
        id              TEXT PRIMARY KEY,
        content         TEXT NOT NULL,
        relevance_score REAL NOT NULL DEFAULT 0.5,
        access_count    INTEGER NOT NULL DEFAULT 0,
        last_accessed   TEXT NOT NULL,
        created_at      TEXT NOT NULL,
        updated_at      TEXT NOT NULL,
        tags_json       TEXT NOT NULL DEFAULT '[]',
        metadata_json   TEXT NOT NULL DEFAULT '{}',
        memory_type     TEXT NOT NULL DEFAULT 'semantic',
        user_id         TEXT,
        session_id      TEXT,
        expires_at      TEXT,
        content_hash    TEXT NOT NULL DEFAULT '',
        source_turns_json TEXT NOT NULL DEFAULT '[]',
        links_json      TEXT NOT NULL DEFAULT '[]'
    )""",
    """CREATE TABLE IF NOT EXISTS cache_entries (
        key         TEXT PRIMARY KEY,
        value_json  TEXT NOT NULL,
        created_at  REAL NOT NULL,
        expires_at  REAL
    )""",
    """CREATE TABLE IF NOT EXISTS graph_nodes (
        node_id       TEXT PRIMARY KEY,
        metadata_json TEXT NOT NULL DEFAULT '{}'
    )""",
    """CREATE TABLE IF NOT EXISTS graph_edges (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        source        TEXT NOT NULL,
        relation      TEXT NOT NULL,
        target        TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        UNIQUE(source, relation, target)
    )""",
    """CREATE TABLE IF NOT EXISTS graph_memory_links (
        node_id   TEXT NOT NULL,
        memory_id TEXT NOT NULL,
        PRIMARY KEY (node_id, memory_id)
    )""",
    """CREATE TABLE IF NOT EXISTS conversation_turns (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id    TEXT NOT NULL,
        turn_index    INTEGER NOT NULL,
        role          TEXT NOT NULL,
        content       TEXT NOT NULL,
        token_count   INTEGER NOT NULL DEFAULT 0,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at    TEXT NOT NULL,
        UNIQUE (session_id, turn_index)
    )""",
    """CREATE TABLE IF NOT EXISTS summary_tiers (
        session_id       TEXT NOT NULL,
        tier_level       INTEGER NOT NULL,
        content          TEXT NOT NULL,
        token_count      INTEGER NOT NULL,
        source_turn_count INTEGER NOT NULL,
        created_at       TEXT NOT NULL,
        updated_at       TEXT NOT NULL,
        PRIMARY KEY (session_id, tier_level)
    )""",
]

_INDEXES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_memory_entries_user_id ON memory_entries(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_memory_entries_session_id ON memory_entries(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_memory_entries_memory_type ON memory_entries(memory_type)",
    "CREATE INDEX IF NOT EXISTS idx_memory_entries_created_at ON memory_entries(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_memory_entries_expires_at ON memory_entries(expires_at)",
    "CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at)",
    "CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source)",
    "CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target)",
    "CREATE INDEX IF NOT EXISTS idx_edges_relation ON graph_edges(relation)",
    "CREATE INDEX IF NOT EXISTS idx_memory_links_node ON graph_memory_links(node_id)",
    "CREATE INDEX IF NOT EXISTS idx_turns_session ON conversation_turns(session_id, turn_index)",
]


def ensure_tables(conn: sqlite3.Connection) -> None:
    """Create all storage tables and indexes if they do not exist."""
    for ddl in _TABLES:
        conn.execute(ddl)
    for ddl in _INDEXES:
        conn.execute(ddl)
    conn.commit()


async def ensure_tables_async(conn: aiosqlite.Connection) -> None:
    """Async variant of :func:`ensure_tables`."""
    for ddl in _TABLES:
        await conn.execute(ddl)
    for ddl in _INDEXES:
        await conn.execute(ddl)
    await conn.commit()
