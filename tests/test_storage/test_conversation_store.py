"""Tests for ConversationStore protocol and InMemory implementation."""

from anchor.models.memory import ConversationTurn, SummaryTier
from anchor.protocols.storage import ConversationStore
from anchor.storage._serialization import (
    conversation_turn_to_row,
    row_to_conversation_turn,
    summary_tier_to_row,
    row_to_summary_tier,
)
from anchor.storage.memory_store import InMemoryConversationStore


def _turn(role: str, content: str) -> ConversationTurn:
    return ConversationTurn(role=role, content=content)


def _tier(level: int, content: str, turn_count: int = 1) -> SummaryTier:
    return SummaryTier(level=level, content=content, token_count=10, source_turn_count=turn_count)


class TestConversationSerialization:
    def test_turn_round_trip(self):
        turn = _turn("user", "hello world")
        row = conversation_turn_to_row(turn)
        restored = row_to_conversation_turn(row)
        assert restored.role == turn.role
        assert restored.content == turn.content

    def test_tier_round_trip(self):
        tier = _tier(1, "summary text", turn_count=5)
        row = summary_tier_to_row(tier)
        restored = row_to_summary_tier(row)
        assert restored.level == tier.level
        assert restored.content == tier.content
        assert restored.source_turn_count == 5


class TestInMemoryConversationStoreProtocol:
    def test_satisfies_protocol(self):
        store = InMemoryConversationStore()
        assert isinstance(store, ConversationStore)


class TestInMemoryConversationStoreTurns:
    def test_append_and_load(self):
        store = InMemoryConversationStore()
        store.append_turn("sess-1", _turn("user", "hello"))
        store.append_turn("sess-1", _turn("assistant", "hi"))
        turns = store.load_turns("sess-1")
        assert len(turns) == 2
        assert turns[0].content == "hello"
        assert turns[1].content == "hi"

    def test_load_with_limit(self):
        store = InMemoryConversationStore()
        for i in range(10):
            store.append_turn("sess-1", _turn("user", f"msg-{i}"))
        turns = store.load_turns("sess-1", limit=3)
        assert len(turns) == 3
        assert turns[0].content == "msg-7"

    def test_load_empty_session(self):
        store = InMemoryConversationStore()
        assert store.load_turns("nonexistent") == []

    def test_truncate_turns(self):
        store = InMemoryConversationStore()
        for i in range(10):
            store.append_turn("sess-1", _turn("user", f"msg-{i}"))
        store.truncate_turns("sess-1", keep_last=3)
        turns = store.load_turns("sess-1")
        assert len(turns) == 3
        assert turns[0].content == "msg-7"

    def test_sessions_are_isolated(self):
        store = InMemoryConversationStore()
        store.append_turn("sess-1", _turn("user", "hello"))
        store.append_turn("sess-2", _turn("user", "world"))
        assert len(store.load_turns("sess-1")) == 1
        assert len(store.load_turns("sess-2")) == 1


class TestInMemoryConversationStoreTiers:
    def test_save_and_load_tiers(self):
        store = InMemoryConversationStore()
        tiers = {1: _tier(1, "summary"), 2: None, 3: None}
        store.save_summary_tiers("sess-1", tiers)
        loaded = store.load_summary_tiers("sess-1")
        assert loaded[1] is not None
        assert loaded[1].content == "summary"
        assert loaded[2] is None

    def test_load_tiers_empty_session(self):
        store = InMemoryConversationStore()
        loaded = store.load_summary_tiers("nonexistent")
        assert loaded == {1: None, 2: None, 3: None}

    def test_save_tiers_overwrites(self):
        store = InMemoryConversationStore()
        store.save_summary_tiers("sess-1", {1: _tier(1, "old"), 2: None, 3: None})
        store.save_summary_tiers("sess-1", {1: _tier(1, "new"), 2: None, 3: None})
        loaded = store.load_summary_tiers("sess-1")
        assert loaded[1].content == "new"


class TestInMemoryConversationStoreSession:
    def test_list_sessions(self):
        store = InMemoryConversationStore()
        store.append_turn("sess-1", _turn("user", "hello"))
        store.append_turn("sess-2", _turn("user", "world"))
        assert sorted(store.list_sessions()) == ["sess-1", "sess-2"]

    def test_delete_session(self):
        store = InMemoryConversationStore()
        store.append_turn("sess-1", _turn("user", "hello"))
        store.save_summary_tiers("sess-1", {1: _tier(1, "summary"), 2: None, 3: None})
        assert store.delete_session("sess-1") is True
        assert store.load_turns("sess-1") == []
        assert store.delete_session("sess-1") is False

    def test_clear(self):
        store = InMemoryConversationStore()
        store.append_turn("sess-1", _turn("user", "hello"))
        store.append_turn("sess-2", _turn("user", "world"))
        store.clear()
        assert store.list_sessions() == []
