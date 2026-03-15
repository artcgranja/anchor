"""Storage protocol definitions.

All storage backends implement these protocols using structural subtyping (PEP 544).
Users can provide any object that matches the interface -- no inheritance required.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from anchor.models.context import ContextItem
from anchor.models.memory import ConversationTurn, MemoryEntry, SummaryTier


@runtime_checkable
class ContextStore(Protocol):
    """Protocol for storing and retrieving context items."""

    def add(self, item: ContextItem) -> None:
        """Persist a single context item to the store.

        Parameters:
            item: The context item to store. If an item with the same
                ``id`` already exists, the implementation should overwrite it.

        Side Effects:
            The item is durably persisted (or held in memory, depending on
            the backend) and will be visible to subsequent ``get`` /
            ``get_all`` calls.
        """
        ...

    def get(self, item_id: str) -> ContextItem | None:
        """Retrieve a single context item by its unique identifier.

        Parameters:
            item_id: The unique string identifier of the item.

        Returns:
            The matching ``ContextItem``, or ``None`` if no item with the
            given id exists.
        """
        ...

    def get_all(self) -> list[ContextItem]:
        """Return every context item currently held in the store.

        Returns:
            A list of all stored ``ContextItem`` objects.  The order is
            implementation-defined.  Returns an empty list when the store
            is empty.
        """
        ...

    def delete(self, item_id: str) -> bool:
        """Remove a context item by id.

        Parameters:
            item_id: The unique identifier of the item to delete.

        Returns:
            ``True`` if an item was found and deleted, ``False`` if no
            item with the given id existed.

        Side Effects:
            The item is permanently removed from the store.
        """
        ...

    def clear(self) -> None:
        """Remove all items from the store.

        Side Effects:
            The store is left empty.  This operation is irreversible.
        """
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector similarity search backends.

    Implementations might wrap FAISS, Chroma, Qdrant, Pinecone, etc.
    """

    def add_embedding(
        self, item_id: str, embedding: list[float], metadata: dict[str, Any] | None = None
    ) -> None:
        """Store an embedding vector associated with an item id.

        Parameters:
            item_id: A unique identifier linking the embedding back to its
                source context item or document.
            embedding: The dense vector representation of the item.
            metadata: Optional key-value metadata attached to the vector
                entry (e.g., source filename, chunk index).

        Side Effects:
            The embedding is persisted in the vector index.  If an entry
            for ``item_id`` already exists, the implementation should
            overwrite it.
        """
        ...

    def search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Find the most similar embeddings to a query vector.

        Parameters:
            query_embedding: The dense vector to compare against stored
                embeddings.
            top_k: Maximum number of results to return.

        Returns:
            A list of ``(item_id, score)`` tuples ordered by descending
            similarity.  The score semantics (cosine, dot-product, etc.)
            are implementation-defined.
        """
        ...

    def delete(self, item_id: str) -> bool:
        """Remove the embedding associated with an item id.

        Parameters:
            item_id: The identifier of the embedding to remove.

        Returns:
            ``True`` if the embedding was found and removed, ``False``
            otherwise.

        Side Effects:
            The embedding is permanently removed from the vector index.
        """
        ...


@runtime_checkable
class DocumentStore(Protocol):
    """Protocol for document storage (raw text before chunking/indexing)."""

    def add_document(
        self, doc_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Store a raw document (pre-chunking / pre-indexing).

        Parameters:
            doc_id: A unique identifier for the document.
            content: The full text content of the document.
            metadata: Optional key-value metadata (e.g., source URL,
                author, ingestion timestamp).

        Side Effects:
            The document is persisted and becomes visible to subsequent
            ``get_document`` and ``list_documents`` calls.
        """
        ...

    def get_document(self, doc_id: str) -> str | None:
        """Retrieve a document's text content by its unique identifier.

        Parameters:
            doc_id: The identifier assigned when the document was added.

        Returns:
            The full text content of the document, or ``None`` if no
            document with the given id exists.
        """
        ...

    def list_documents(self) -> list[str]:
        """Return the ids of all stored documents.

        Returns:
            A list of document id strings.  The order is
            implementation-defined.  Returns an empty list when the store
            contains no documents.
        """
        ...

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document by its id.

        Parameters:
            doc_id: The identifier of the document to delete.

        Returns:
            ``True`` if a document was found and deleted, ``False`` if no
            document with the given id existed.

        Side Effects:
            The document is permanently removed from the store.
        """
        ...


@runtime_checkable
class MemoryEntryStore(Protocol):
    """Protocol for persistent memory entry storage.

    Implementations hold ``MemoryEntry`` objects and provide basic
    CRUD + search operations.  The store is used by ``MemoryManager``,
    ``ScoredMemoryRetriever``, and the eviction-promotion pipeline steps.
    """

    def add(self, entry: MemoryEntry) -> None:
        """Persist a single memory entry to the store.

        Parameters:
            entry: The memory entry to store.  If an entry with the same
                ``id`` already exists, the implementation should overwrite it.

        Side Effects:
            The entry is durably persisted (or held in memory, depending on
            the backend) and will be visible to subsequent ``search`` /
            ``list_all`` calls.
        """
        ...

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Search for memory entries relevant to a query string.

        Parameters:
            query: Free-text search query.  The matching semantics
                (keyword, embedding, hybrid) are implementation-defined.
            top_k: Maximum number of results to return.

        Returns:
            A list of up to ``top_k`` matching ``MemoryEntry`` objects,
            ordered by relevance (most relevant first).  Returns an empty
            list when no entries match.
        """
        ...

    def list_all(self) -> list[MemoryEntry]:
        """Return every non-expired memory entry in the store.

        Returns:
            A list of all stored ``MemoryEntry`` objects.  The order is
            implementation-defined.  Returns an empty list when the store
            is empty.
        """
        ...

    def delete(self, entry_id: str) -> bool:
        """Remove a memory entry by its unique identifier.

        Parameters:
            entry_id: The ``id`` of the entry to delete.

        Returns:
            ``True`` if an entry was found and deleted, ``False`` if no
            entry with the given id existed.

        Side Effects:
            The entry is permanently removed from the store.
        """
        ...

    def clear(self) -> None:
        """Remove all entries from the store.

        Side Effects:
            The store is left empty.  This operation is irreversible.
        """
        ...


@runtime_checkable
class GarbageCollectableStore(Protocol):
    """A memory entry store that supports garbage collection.

    Extends the base ``MemoryEntryStore`` contract with
    ``list_all_unfiltered`` so that expired entries can be discovered
    and deleted by the ``MemoryGarbageCollector``.
    """

    def list_all_unfiltered(self) -> list[MemoryEntry]:
        """Return *all* entries including expired ones.

        Unlike ``MemoryEntryStore.list_all`` which may filter out expired
        entries, this method returns every entry so that the garbage
        collector can identify and prune them.

        Returns:
            A list of all ``MemoryEntry`` objects in the store,
            regardless of expiration status.
        """
        ...

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by id.

        Parameters:
            entry_id: The ``id`` of the entry to delete.

        Returns:
            ``True`` if found and deleted, ``False`` otherwise.
        """
        ...


# ---------------------------------------------------------------------------
# Async protocol variants
# ---------------------------------------------------------------------------


@runtime_checkable
class AsyncContextStore(Protocol):
    """Async variant of :class:`ContextStore`.

    All methods mirror their sync counterparts.  See :class:`ContextStore`
    for full parameter documentation.
    """

    async def add(self, item: ContextItem) -> None: ...
    async def get(self, item_id: str) -> ContextItem | None: ...
    async def get_all(self) -> list[ContextItem]: ...
    async def delete(self, item_id: str) -> bool: ...
    async def clear(self) -> None: ...


@runtime_checkable
class AsyncVectorStore(Protocol):
    """Async variant of :class:`VectorStore`.

    All methods mirror their sync counterparts.  See :class:`VectorStore`
    for full parameter documentation.
    """

    async def add_embedding(
        self, item_id: str, embedding: list[float], metadata: dict[str, Any] | None = None
    ) -> None: ...
    async def search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[tuple[str, float]]: ...
    async def delete(self, item_id: str) -> bool: ...


@runtime_checkable
class AsyncDocumentStore(Protocol):
    """Async variant of :class:`DocumentStore`.

    All methods mirror their sync counterparts.  See :class:`DocumentStore`
    for full parameter documentation.
    """

    async def add_document(
        self, doc_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None: ...
    async def get_document(self, doc_id: str) -> str | None: ...
    async def list_documents(self) -> list[str]: ...
    async def delete_document(self, doc_id: str) -> bool: ...


@runtime_checkable
class AsyncMemoryEntryStore(Protocol):
    """Async variant of :class:`MemoryEntryStore`.

    All methods mirror their sync counterparts.  See :class:`MemoryEntryStore`
    for full parameter documentation.
    """

    async def add(self, entry: MemoryEntry) -> None: ...
    async def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]: ...
    async def list_all(self) -> list[MemoryEntry]: ...
    async def delete(self, entry_id: str) -> bool: ...
    async def clear(self) -> None: ...


@runtime_checkable
class AsyncGarbageCollectableStore(Protocol):
    """Async variant of :class:`GarbageCollectableStore`.

    All methods mirror their sync counterparts.  See
    :class:`GarbageCollectableStore` for full parameter documentation.
    """

    async def list_all_unfiltered(self) -> list[MemoryEntry]: ...
    async def delete(self, entry_id: str) -> bool: ...


@runtime_checkable
class GraphStore(Protocol):
    """Protocol for persistent graph storage (entities + relationships).

    Stores a directed labelled graph where nodes represent entities and
    edges represent typed relationships between them.  Each node and edge
    may carry arbitrary JSON-serialisable metadata.  Nodes can be linked
    to ``MemoryEntry`` IDs for knowledge-graph-backed retrieval.

    Behavioural contract
    --------------------
    * ``add_edge`` auto-creates source/target nodes if they do not exist.
    * ``remove_node`` cascades to remove associated edges and memory links.
    * ``link_memory`` raises ``KeyError`` if the node does not exist.
    * ``get_neighbors`` performs cycle-safe BFS up to *max_depth*.
    """

    def add_node(self, node_id: str, metadata: dict[str, Any] | None = None) -> None:
        """Create or update a node.  Merges *metadata* if the node exists."""
        ...

    def add_edge(
        self, source: str, relation: str, target: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a directed edge.  Implicitly creates source/target nodes."""
        ...

    def get_neighbors(
        self, node_id: str, max_depth: int = 1,
        relation_filter: str | list[str] | None = None,
    ) -> list[str]:
        """Return node IDs reachable within *max_depth* hops (BFS).

        Parameters:
            node_id: Starting node.
            max_depth: Maximum traversal depth (default 1).
            relation_filter: If provided, only traverse edges whose
                relation matches one of the given values.

        Returns:
            List of reachable node IDs (excluding *node_id* itself).
        """
        ...

    def get_edges(self, node_id: str) -> list[tuple[str, str, str]]:
        """Return all edges where *node_id* is source or target.

        Each edge is a ``(source, relation, target)`` tuple.
        """
        ...

    def get_node_metadata(self, node_id: str) -> dict[str, Any] | None:
        """Return metadata for *node_id*, or ``None`` if the node does not exist."""
        ...

    def link_memory(self, node_id: str, memory_id: str) -> None:
        """Associate a ``MemoryEntry`` ID with a node.

        Raises:
            KeyError: If *node_id* does not exist.
        """
        ...

    def get_memory_ids(self, node_id: str, max_depth: int = 1) -> list[str]:
        """Return distinct memory IDs linked to *node_id* and its neighbours."""
        ...

    def remove_node(self, node_id: str) -> None:
        """Delete a node, its edges, and its memory links."""
        ...

    def remove_edge(self, source: str, relation: str, target: str) -> bool:
        """Delete a specific edge.  Returns ``True`` if the edge existed."""
        ...

    def list_nodes(self) -> list[str]:
        """Return all node IDs in the graph."""
        ...

    def list_edges(self) -> list[tuple[str, str, str]]:
        """Return all edges as ``(source, relation, target)`` tuples."""
        ...

    def clear(self) -> None:
        """Remove all nodes, edges, and memory links."""
        ...


@runtime_checkable
class AsyncGraphStore(Protocol):
    """Async variant of :class:`GraphStore`.  Same behavioural contract."""

    async def add_node(self, node_id: str, metadata: dict[str, Any] | None = None) -> None:
        """Create or update a node.  Merges *metadata* if the node exists."""
        ...

    async def add_edge(
        self, source: str, relation: str, target: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a directed edge.  Implicitly creates source/target nodes."""
        ...

    async def get_neighbors(
        self, node_id: str, max_depth: int = 1,
        relation_filter: str | list[str] | None = None,
    ) -> list[str]:
        """Return node IDs reachable within *max_depth* hops (BFS)."""
        ...

    async def get_edges(self, node_id: str) -> list[tuple[str, str, str]]:
        """Return all edges where *node_id* is source or target."""
        ...

    async def get_node_metadata(self, node_id: str) -> dict[str, Any] | None:
        """Return metadata for *node_id*, or ``None`` if not found."""
        ...

    async def link_memory(self, node_id: str, memory_id: str) -> None:
        """Associate a ``MemoryEntry`` ID with a node.  Raises ``KeyError`` if missing."""
        ...

    async def get_memory_ids(self, node_id: str, max_depth: int = 1) -> list[str]:
        """Return distinct memory IDs linked to *node_id* and its neighbours."""
        ...

    async def remove_node(self, node_id: str) -> None:
        """Delete a node, its edges, and its memory links."""
        ...

    async def remove_edge(self, source: str, relation: str, target: str) -> bool:
        """Delete a specific edge.  Returns ``True`` if the edge existed."""
        ...

    async def list_nodes(self) -> list[str]:
        """Return all node IDs in the graph."""
        ...

    async def list_edges(self) -> list[tuple[str, str, str]]:
        """Return all edges as ``(source, relation, target)`` tuples."""
        ...

    async def clear(self) -> None:
        """Remove all nodes, edges, and memory links."""
        ...


@runtime_checkable
class ConversationStore(Protocol):
    """Protocol for persistent conversation history storage.

    Stores an append-only log of conversation turns per session, plus
    optional summary tiers produced by progressive summarisation.

    Behavioural contract
    --------------------
    * Turns are ordered by ``turn_index`` within each session.
    * ``load_turns`` returns turns in chronological order.
    * ``truncate_turns`` preserves only the last *keep_last* turns.
    * ``delete_session`` removes both turns and summary tiers atomically.
    * ``list_sessions`` returns sessions that have turns *or* tiers.
    """

    def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        """Append a conversation turn to the session log."""
        ...

    def load_turns(self, session_id: str, limit: int | None = None) -> list[ConversationTurn]:
        """Load turns for a session, optionally limited to the last *limit*.

        Returns:
            Turns in chronological order.
        """
        ...

    def save_summary_tiers(self, session_id: str, tiers: dict[int, SummaryTier | None]) -> None:
        """Atomically replace all summary tiers for a session."""
        ...

    def load_summary_tiers(self, session_id: str) -> dict[int, SummaryTier | None]:
        """Load summary tiers, returning ``{1: None, 2: None, 3: None}`` for missing tiers."""
        ...

    def truncate_turns(self, session_id: str, keep_last: int) -> None:
        """Delete all but the last *keep_last* turns in a session."""
        ...

    def delete_session(self, session_id: str) -> bool:
        """Delete all turns and summary tiers for a session.

        Returns:
            ``True`` if any data was deleted.
        """
        ...

    def list_sessions(self) -> list[str]:
        """Return all session IDs that have turns or tiers."""
        ...

    def clear(self) -> None:
        """Remove all sessions, turns, and tiers."""
        ...


@runtime_checkable
class AsyncConversationStore(Protocol):
    """Async variant of :class:`ConversationStore`.  Same behavioural contract."""

    async def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        """Append a conversation turn to the session log."""
        ...

    async def load_turns(self, session_id: str, limit: int | None = None) -> list[ConversationTurn]:
        """Load turns for a session in chronological order."""
        ...

    async def save_summary_tiers(self, session_id: str, tiers: dict[int, SummaryTier | None]) -> None:
        """Atomically replace all summary tiers for a session."""
        ...

    async def load_summary_tiers(self, session_id: str) -> dict[int, SummaryTier | None]:
        """Load summary tiers, returning defaults for missing tiers."""
        ...

    async def truncate_turns(self, session_id: str, keep_last: int) -> None:
        """Delete all but the last *keep_last* turns in a session."""
        ...

    async def delete_session(self, session_id: str) -> bool:
        """Delete all turns and summary tiers for a session."""
        ...

    async def list_sessions(self) -> list[str]:
        """Return all session IDs that have turns or tiers."""
        ...

    async def clear(self) -> None:
        """Remove all sessions, turns, and tiers."""
        ...
