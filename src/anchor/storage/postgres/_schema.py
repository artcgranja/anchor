"""PostgreSQL schema definitions and table creation with pgvector support."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg


async def ensure_tables(
    conn: asyncpg.Connection,  # type: ignore[type-arg]
    *,
    embedding_dim: int = 1536,
) -> None:
    """Create all storage tables and indexes if they do not exist.

    Parameters:
        conn: An asyncpg connection.
        embedding_dim: Dimension of embedding vectors for pgvector column.
    """
    # Enable pgvector extension (requires superuser or CREATE privilege)
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS context_items (
            id          TEXT PRIMARY KEY,
            content     TEXT NOT NULL,
            source      TEXT NOT NULL,
            score       DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            priority    INTEGER NOT NULL DEFAULT 5,
            token_count INTEGER NOT NULL DEFAULT 0,
            metadata    JSONB NOT NULL DEFAULT '{}',
            created_at  TIMESTAMPTZ NOT NULL
        )
    """)

    await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS embeddings (
            item_id   TEXT PRIMARY KEY,
            embedding vector({embedding_dim}),
            metadata  JSONB NOT NULL DEFAULT '{{}}'
        )
    """)

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id   TEXT PRIMARY KEY,
            content  TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}'
        )
    """)

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_entries (
            id              TEXT PRIMARY KEY,
            content         TEXT NOT NULL,
            relevance_score DOUBLE PRECISION NOT NULL DEFAULT 0.5,
            access_count    INTEGER NOT NULL DEFAULT 0,
            last_accessed   TIMESTAMPTZ NOT NULL,
            created_at      TIMESTAMPTZ NOT NULL,
            updated_at      TIMESTAMPTZ NOT NULL,
            tags            JSONB NOT NULL DEFAULT '[]',
            metadata        JSONB NOT NULL DEFAULT '{}',
            memory_type     TEXT NOT NULL DEFAULT 'semantic',
            user_id         TEXT,
            session_id      TEXT,
            expires_at      TIMESTAMPTZ,
            content_hash    TEXT NOT NULL DEFAULT '',
            source_turns    JSONB NOT NULL DEFAULT '[]',
            links           JSONB NOT NULL DEFAULT '[]'
        )
    """)

    # Indexes
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_me_user_id ON memory_entries(user_id)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_me_session_id ON memory_entries(session_id)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_me_memory_type ON memory_entries(memory_type)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_me_created_at ON memory_entries(created_at)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_me_expires_at ON memory_entries(expires_at)"
    )

    # pgvector IVFFlat index (requires data to exist for training)
    # Users should create this after initial data load:
    # CREATE INDEX ON embeddings
    #   USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
