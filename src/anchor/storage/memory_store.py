"""In-memory storage implementations for development and testing.

These are the default backends -- no external dependencies needed.
Production users provide their own implementations (Redis, Postgres, etc.)
that satisfy the storage protocols.
"""

from __future__ import annotations

import heapq
import logging
import threading
from typing import Any

from anchor._math import cosine_similarity
from anchor.models.context import ContextItem
from anchor.models.memory import ConversationTurn, SummaryTier

logger = logging.getLogger(__name__)


class InMemoryContextStore:
    """Dict-backed context store. Implements ContextStore protocol."""

    __slots__ = ("_items", "_lock")

    def __init__(self) -> None:
        self._items: dict[str, ContextItem] = {}
        self._lock = threading.Lock()

    def add(self, item: ContextItem) -> None:
        with self._lock:
            self._items[item.id] = item

    def get(self, item_id: str) -> ContextItem | None:
        with self._lock:
            return self._items.get(item_id)

    def get_all(self) -> list[ContextItem]:
        with self._lock:
            return list(self._items.values())

    def delete(self, item_id: str) -> bool:
        with self._lock:
            return self._items.pop(item_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(items={len(self._items)})"


class InMemoryVectorStore:
    """Brute-force cosine similarity vector store.

    For development/testing only. Production use should provide
    FAISS, Chroma, Qdrant, etc. via the VectorStore protocol.
    """

    __slots__ = ("_embeddings", "_large_store_warned", "_lock", "_metadata")

    _LARGE_STORE_THRESHOLD: int = 5000

    def __init__(self) -> None:
        self._embeddings: dict[str, list[float]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._large_store_warned: bool = False
        self._lock = threading.Lock()

    def add_embedding(
        self, item_id: str, embedding: list[float], metadata: dict[str, Any] | None = None
    ) -> None:
        with self._lock:
            self._embeddings[item_id] = embedding
            if metadata:
                self._metadata[item_id] = metadata

    def search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[tuple[str, float]]:
        with self._lock:
            if not self._embeddings:
                return []
            n = len(self._embeddings)
            if n > self._LARGE_STORE_THRESHOLD and not self._large_store_warned:
                logger.warning(
                    "InMemoryVectorStore has %d embeddings. Consider using a dedicated "
                    "vector database (FAISS, Chroma) for better performance.",
                    n,
                )
                self._large_store_warned = True
            results: list[tuple[str, float]] = []
            for item_id, emb in self._embeddings.items():
                score = cosine_similarity(query_embedding, emb)
                results.append((item_id, score))
            return heapq.nlargest(top_k, results, key=lambda x: x[1])

    def delete(self, item_id: str) -> bool:
        with self._lock:
            removed = self._embeddings.pop(item_id, None) is not None
            self._metadata.pop(item_id, None)
            return removed

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors without numpy.

        Delegates to :func:`anchor._math.cosine_similarity`.
        Kept for backwards compatibility with code that calls this static method.
        """
        return cosine_similarity(a, b)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(embeddings={len(self._embeddings)})"


class InMemoryDocumentStore:
    """Dict-backed document store. Implements DocumentStore protocol."""

    __slots__ = ("_documents", "_lock", "_metadata")

    def __init__(self) -> None:
        self._documents: dict[str, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def add_document(
        self, doc_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        with self._lock:
            self._documents[doc_id] = content
            if metadata:
                self._metadata[doc_id] = metadata

    def get_document(self, doc_id: str) -> str | None:
        with self._lock:
            return self._documents.get(doc_id)

    def list_documents(self) -> list[str]:
        with self._lock:
            return list(self._documents.keys())

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            removed = self._documents.pop(doc_id, None) is not None
            self._metadata.pop(doc_id, None)
            return removed

    def __repr__(self) -> str:
        return f"{type(self).__name__}(documents={len(self._documents)})"


class InMemoryGraphStore:
    """Dict-backed graph store. Implements GraphStore protocol."""

    __slots__ = ("_edge_metadata", "_edges",
                 "_entity_to_memories", "_lock", "_nodes")

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: list[tuple[str, str, str]] = []
        self._edge_metadata: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._entity_to_memories: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    def add_node(self, node_id: str, metadata: dict[str, Any] | None = None) -> None:
        with self._lock:
            if node_id in self._nodes:
                if metadata:
                    self._nodes[node_id].update(metadata)
            else:
                self._nodes[node_id] = dict(metadata) if metadata else {}

    def add_edge(
        self, source: str, relation: str, target: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            if source not in self._nodes:
                self._nodes[source] = {}
            if target not in self._nodes:
                self._nodes[target] = {}
            edge_key = (source, relation, target)
            if edge_key not in self._edge_metadata:
                self._edges.append(edge_key)
            if metadata:
                self._edge_metadata[edge_key] = metadata
            elif edge_key not in self._edge_metadata:
                self._edge_metadata[edge_key] = {}

    def _rebuild_adjacency(self, relation_filter: str | list[str] | None = None) -> dict[str, set[str]]:
        """Build adjacency index, optionally filtered by relation types."""
        if relation_filter is not None:
            allowed = {relation_filter} if isinstance(relation_filter, str) else set(relation_filter)
        else:
            allowed = None
        adj: dict[str, set[str]] = {}
        for src, rel, tgt in self._edges:
            if allowed is not None and rel not in allowed:
                continue
            adj.setdefault(src, set()).add(tgt)
            adj.setdefault(tgt, set()).add(src)
        return adj

    def get_neighbors(
        self, node_id: str, max_depth: int = 1,
        relation_filter: str | list[str] | None = None,
    ) -> list[str]:
        with self._lock:
            if node_id not in self._nodes:
                return []
            adj = self._rebuild_adjacency(relation_filter)
            from collections import deque
            visited: set[str] = {node_id}
            queue: deque[tuple[str, int]] = deque([(node_id, 0)])
            result: list[str] = []
            while queue:
                current, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                for neighbor in adj.get(current, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        result.append(neighbor)
                        queue.append((neighbor, depth + 1))
            return result

    def get_edges(self, node_id: str) -> list[tuple[str, str, str]]:
        with self._lock:
            return [
                (s, r, t) for s, r, t in self._edges
                if s == node_id or t == node_id
            ]

    def get_node_metadata(self, node_id: str) -> dict[str, Any] | None:
        with self._lock:
            if node_id not in self._nodes:
                return None
            return dict(self._nodes[node_id])

    def link_memory(self, node_id: str, memory_id: str) -> None:
        with self._lock:
            if node_id not in self._nodes:
                msg = f"Entity '{node_id}' does not exist in the graph"
                raise KeyError(msg)
            self._entity_to_memories.setdefault(node_id, []).append(memory_id)

    def get_memory_ids(self, node_id: str, max_depth: int = 1) -> list[str]:
        with self._lock:
            all_entities = [node_id, *self._get_neighbors_unlocked(node_id, max_depth)]
            seen: set[str] = set()
            result: list[str] = []
            for eid in all_entities:
                for mid in self._entity_to_memories.get(eid, []):
                    if mid not in seen:
                        seen.add(mid)
                        result.append(mid)
            return result

    def _get_neighbors_unlocked(self, node_id: str, max_depth: int = 1) -> list[str]:
        """Internal BFS without lock (caller must hold lock)."""
        if node_id not in self._nodes:
            return []
        adj = self._rebuild_adjacency()
        from collections import deque
        visited: set[str] = {node_id}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        result: list[str] = []
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor in adj.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    result.append(neighbor)
                    queue.append((neighbor, depth + 1))
        return result

    def remove_node(self, node_id: str) -> None:
        with self._lock:
            self._nodes.pop(node_id, None)
            old_edges = self._edges
            self._edges = [
                (s, r, t) for s, r, t in old_edges
                if s != node_id and t != node_id
            ]
            # Clean edge metadata
            for key in list(self._edge_metadata):
                if key[0] == node_id or key[2] == node_id:
                    del self._edge_metadata[key]
            self._entity_to_memories.pop(node_id, None)

    def remove_edge(self, source: str, relation: str, target: str) -> bool:
        with self._lock:
            edge_key = (source, relation, target)
            if edge_key in self._edge_metadata:
                self._edges.remove(edge_key)
                del self._edge_metadata[edge_key]
                return True
            return False

    def list_nodes(self) -> list[str]:
        with self._lock:
            return list(self._nodes)

    def list_edges(self) -> list[tuple[str, str, str]]:
        with self._lock:
            return list(self._edges)

    def clear(self) -> None:
        with self._lock:
            self._nodes.clear()
            self._edges.clear()
            self._edge_metadata.clear()
            self._entity_to_memories.clear()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nodes={len(self._nodes)}, edges={len(self._edges)})"


class InMemoryConversationStore:
    """Dict-backed conversation store. Implements ConversationStore protocol."""

    __slots__ = ("_lock", "_tiers", "_turns")

    def __init__(self) -> None:
        self._turns: dict[str, list[ConversationTurn]] = {}
        self._tiers: dict[str, dict[int, SummaryTier | None]] = {}
        self._lock = threading.Lock()

    def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        with self._lock:
            self._turns.setdefault(session_id, []).append(turn)

    def load_turns(self, session_id: str, limit: int | None = None) -> list[ConversationTurn]:
        with self._lock:
            turns = self._turns.get(session_id, [])
            if limit is not None:
                return list(turns[-limit:])
            return list(turns)

    def save_summary_tiers(self, session_id: str, tiers: dict[int, SummaryTier | None]) -> None:
        with self._lock:
            self._tiers[session_id] = dict(tiers)

    def load_summary_tiers(self, session_id: str) -> dict[int, SummaryTier | None]:
        with self._lock:
            return dict(self._tiers.get(session_id, {1: None, 2: None, 3: None}))

    def truncate_turns(self, session_id: str, keep_last: int) -> None:
        with self._lock:
            turns = self._turns.get(session_id)
            if turns is not None:
                self._turns[session_id] = turns[-keep_last:]

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            found = session_id in self._turns or session_id in self._tiers
            self._turns.pop(session_id, None)
            self._tiers.pop(session_id, None)
            return found

    def list_sessions(self) -> list[str]:
        with self._lock:
            return list(set(self._turns) | set(self._tiers))

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()
            self._tiers.clear()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(sessions={len(set(self._turns) | set(self._tiers))})"
