"""Tests for GraphStore protocol and InMemory implementation."""

from anchor.protocols.storage import GraphStore
from anchor.storage.memory_store import InMemoryGraphStore


class TestInMemoryGraphStoreProtocol:
    def test_satisfies_protocol(self):
        store = InMemoryGraphStore()
        assert isinstance(store, GraphStore)


class TestInMemoryGraphStoreNodes:
    def test_add_and_list_nodes(self):
        store = InMemoryGraphStore()
        store.add_node("alice", {"type": "person"})
        store.add_node("bob")
        assert sorted(store.list_nodes()) == ["alice", "bob"]

    def test_add_node_merges_metadata(self):
        store = InMemoryGraphStore()
        store.add_node("alice", {"type": "person"})
        store.add_node("alice", {"role": "engineer"})
        meta = store.get_node_metadata("alice")
        assert meta == {"type": "person", "role": "engineer"}

    def test_get_node_metadata_returns_none_for_missing(self):
        store = InMemoryGraphStore()
        assert store.get_node_metadata("nonexistent") is None

    def test_remove_node(self):
        store = InMemoryGraphStore()
        store.add_node("alice")
        store.add_edge("alice", "knows", "bob")
        store.remove_node("alice")
        assert "alice" not in store.list_nodes()
        assert store.list_edges() == []


class TestInMemoryGraphStoreEdges:
    def test_add_and_list_edges(self):
        store = InMemoryGraphStore()
        store.add_edge("alice", "knows", "bob")
        edges = store.list_edges()
        assert ("alice", "knows", "bob") in edges

    def test_add_edge_auto_creates_nodes(self):
        store = InMemoryGraphStore()
        store.add_edge("alice", "knows", "bob")
        assert sorted(store.list_nodes()) == ["alice", "bob"]

    def test_get_edges_for_node(self):
        store = InMemoryGraphStore()
        store.add_edge("alice", "knows", "bob")
        store.add_edge("alice", "works_with", "carol")
        edges = store.get_edges("alice")
        assert len(edges) == 2

    def test_remove_edge(self):
        store = InMemoryGraphStore()
        store.add_edge("alice", "knows", "bob")
        assert store.remove_edge("alice", "knows", "bob") is True
        assert store.list_edges() == []
        assert store.remove_edge("alice", "knows", "bob") is False

    def test_add_edge_with_metadata(self):
        store = InMemoryGraphStore()
        store.add_edge("alice", "knows", "bob", metadata={"weight": 0.9})
        # Edge metadata is stored but not exposed in list_edges tuples
        edges = store.list_edges()
        assert ("alice", "knows", "bob") in edges


class TestInMemoryGraphStoreTraversal:
    def test_get_neighbors_depth_1(self):
        store = InMemoryGraphStore()
        store.add_edge("alice", "knows", "bob")
        store.add_edge("bob", "knows", "carol")
        neighbors = store.get_neighbors("alice", max_depth=1)
        assert neighbors == ["bob"]

    def test_get_neighbors_depth_2(self):
        store = InMemoryGraphStore()
        store.add_edge("alice", "knows", "bob")
        store.add_edge("bob", "knows", "carol")
        neighbors = store.get_neighbors("alice", max_depth=2)
        assert sorted(neighbors) == ["bob", "carol"]

    def test_get_neighbors_with_relation_filter(self):
        store = InMemoryGraphStore()
        store.add_edge("alice", "knows", "bob")
        store.add_edge("alice", "works_with", "carol")
        neighbors = store.get_neighbors("alice", relation_filter="knows")
        assert neighbors == ["bob"]

    def test_get_neighbors_with_relation_filter_list(self):
        store = InMemoryGraphStore()
        store.add_edge("alice", "knows", "bob")
        store.add_edge("alice", "works_with", "carol")
        store.add_edge("alice", "manages", "dave")
        neighbors = store.get_neighbors("alice", relation_filter=["knows", "works_with"])
        assert sorted(neighbors) == ["bob", "carol"]

    def test_get_neighbors_missing_node(self):
        store = InMemoryGraphStore()
        assert store.get_neighbors("nonexistent") == []

    def test_get_neighbors_handles_cycles(self):
        store = InMemoryGraphStore()
        store.add_edge("a", "r", "b")
        store.add_edge("b", "r", "c")
        store.add_edge("c", "r", "a")
        neighbors = store.get_neighbors("a", max_depth=5)
        assert sorted(neighbors) == ["b", "c"]


class TestInMemoryGraphStoreMemoryLinks:
    def test_link_and_get_memory_ids(self):
        store = InMemoryGraphStore()
        store.add_node("alice")
        store.link_memory("alice", "mem-001")
        store.link_memory("alice", "mem-002")
        ids = store.get_memory_ids("alice")
        assert sorted(ids) == ["mem-001", "mem-002"]

    def test_link_memory_raises_for_missing_node(self):
        store = InMemoryGraphStore()
        import pytest
        with pytest.raises(KeyError):
            store.link_memory("nonexistent", "mem-001")

    def test_get_memory_ids_with_depth(self):
        store = InMemoryGraphStore()
        store.add_node("alice")
        store.add_node("bob")
        store.add_edge("alice", "knows", "bob")
        store.link_memory("alice", "mem-001")
        store.link_memory("bob", "mem-002")
        ids = store.get_memory_ids("alice", max_depth=1)
        assert sorted(ids) == ["mem-001", "mem-002"]

    def test_remove_node_removes_memory_links(self):
        store = InMemoryGraphStore()
        store.add_node("alice")
        store.link_memory("alice", "mem-001")
        store.remove_node("alice")
        assert store.get_memory_ids("alice") == []


class TestInMemoryGraphStoreClear:
    def test_clear(self):
        store = InMemoryGraphStore()
        store.add_edge("alice", "knows", "bob")
        store.add_node("carol")
        store.link_memory("alice", "mem-001")
        store.clear()
        assert store.list_nodes() == []
        assert store.list_edges() == []
