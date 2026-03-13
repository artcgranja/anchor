"""Shared fixtures for storage tests.

Provides a parametrized ``entry_store`` fixture that yields both
:class:`InMemoryEntryStore`, :class:`JsonFileMemoryStore`, and
:class:`SqliteEntryStore` so that behavioral tests covering search,
filtering, and extra methods run against all implementations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from anchor.storage.json_file_store import JsonFileMemoryStore
from anchor.storage.json_memory_store import InMemoryEntryStore
from anchor.storage.sqlite import (
    SqliteConnectionManager,
    SqliteContextStore,
    SqliteDocumentStore,
    SqliteEntryStore,
    SqliteVectorStore,
    ensure_tables,
)


@pytest.fixture(params=["memory", "json_file", "sqlite"])
def entry_store(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    """Return an entry store instance -- parametrized across all implementations.

    Each test using this fixture runs once with :class:`InMemoryEntryStore`,
    once with :class:`JsonFileMemoryStore`, and once with
    :class:`SqliteEntryStore`.
    """
    kind: str = request.param
    if kind == "memory":
        return InMemoryEntryStore()
    if kind == "json_file":
        store_path = tmp_path / "test_memories.json"
        return JsonFileMemoryStore(store_path)
    # sqlite
    mgr = SqliteConnectionManager(tmp_path / "test.db")
    ensure_tables(mgr.get_connection())
    return SqliteEntryStore(mgr)


@pytest.fixture
def sqlite_conn_manager(tmp_path: Path) -> SqliteConnectionManager:
    """Return a SqliteConnectionManager with tables created."""
    mgr = SqliteConnectionManager(tmp_path / "test.db")
    ensure_tables(mgr.get_connection())
    return mgr


@pytest.fixture
def sqlite_context_store(
    sqlite_conn_manager: SqliteConnectionManager,
) -> SqliteContextStore:
    """Return a SqliteContextStore backed by the shared connection manager."""
    return SqliteContextStore(sqlite_conn_manager)


@pytest.fixture
def sqlite_vector_store(
    sqlite_conn_manager: SqliteConnectionManager,
) -> SqliteVectorStore:
    """Return a SqliteVectorStore backed by the shared connection manager."""
    return SqliteVectorStore(sqlite_conn_manager)


@pytest.fixture
def sqlite_document_store(
    sqlite_conn_manager: SqliteConnectionManager,
) -> SqliteDocumentStore:
    """Return a SqliteDocumentStore backed by the shared connection manager."""
    return SqliteDocumentStore(sqlite_conn_manager)


