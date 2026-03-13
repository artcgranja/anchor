"""Tests for progressive summarization data models and memory."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from anchor.models.memory import FactType, KeyFact, SummaryTier, TierConfig


class TestFactType:
    def test_all_values(self) -> None:
        assert FactType.DECISION == "decision"
        assert FactType.ENTITY == "entity"
        assert FactType.NUMBER == "number"
        assert FactType.DATE == "date"
        assert FactType.PREFERENCE == "preference"
        assert FactType.CONSTRAINT == "constraint"


class TestKeyFact:
    def test_create_minimal(self) -> None:
        fact = KeyFact(fact_type=FactType.DECISION, content="Use FastAPI", source_tier=0)
        assert fact.fact_type == FactType.DECISION
        assert fact.content == "Use FastAPI"
        assert fact.source_tier == 0
        assert fact.id  # auto-generated UUID
        assert fact.token_count == 0

    def test_token_count_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            KeyFact(fact_type=FactType.NUMBER, content="x", source_tier=0, token_count=-1)


class TestSummaryTier:
    def test_create(self) -> None:
        tier = SummaryTier(level=1, content="Summary text", source_turn_count=5)
        assert tier.level == 1
        assert tier.content == "Summary text"
        assert tier.source_turn_count == 5
        assert tier.token_count == 0

    def test_token_count_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            SummaryTier(level=1, content="x", source_turn_count=1, token_count=-1)


class TestTierConfig:
    def test_create_default(self) -> None:
        config = TierConfig(level=0, max_tokens=4096)
        assert config.level == 0
        assert config.max_tokens == 4096
        assert config.target_tokens == 0
        assert config.priority == 7

    def test_frozen(self) -> None:
        config = TierConfig(level=0, max_tokens=4096)
        with pytest.raises(AttributeError):
            config.level = 1  # type: ignore[misc]


from anchor.memory.callbacks import ProgressiveSummarizationCallback


class TestProgressiveSummarizationCallback:
    def test_protocol_exists(self) -> None:
        assert hasattr(ProgressiveSummarizationCallback, 'on_tier_cascade')
        assert hasattr(ProgressiveSummarizationCallback, 'on_facts_extracted')
        assert hasattr(ProgressiveSummarizationCallback, 'on_compaction_error')

    def test_satisfies_protocol(self) -> None:
        class MyCallback:
            def on_tier_cascade(self, from_tier, to_tier, tokens_in, tokens_out):
                pass
            def on_facts_extracted(self, facts, source_tier):
                pass
            def on_compaction_error(self, tier, error):
                pass

        assert isinstance(MyCallback(), ProgressiveSummarizationCallback)
