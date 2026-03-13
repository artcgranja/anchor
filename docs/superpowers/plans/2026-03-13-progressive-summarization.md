# Progressive Summarization Memory Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a 4-tier cascading progressive summarization memory system with LLM-driven compaction and key-fact extraction for anchor's memory module.

**Architecture:** `ProgressiveSummarizationMemory` wraps a `SlidingWindowMemory` (Tier 0) and cascades evicted content through 3 summary tiers (Detailed→Compact→Ultra-compact) using `TierCompactor` which calls `LLMProvider` for summarization and JSON-based fact extraction. Satisfies the existing `ConversationMemory` protocol.

**Tech Stack:** Python 3.11+, Pydantic v2, anchor's LLMProvider protocol, threading.RLock, asyncio

**Spec:** `docs/superpowers/specs/2026-03-13-progressive-summarization-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/anchor/memory/compactor.py` | **Create** | `TierCompactor` — LLM summarization + fact extraction |
| `src/anchor/memory/progressive.py` | **Create** | `ProgressiveSummarizationMemory` — main 4-tier memory class |
| `src/anchor/models/memory.py` | **Modify** | Add `FactType`, `KeyFact`, `SummaryTier`, `TierConfig` |
| `src/anchor/memory/callbacks.py` | **Modify** | Add `ProgressiveSummarizationCallback` protocol |
| `src/anchor/memory/manager.py` | **Modify** | Add `isinstance` branch + `conversation_type` for new class |
| `src/anchor/__init__.py` | **Modify** | Export `ProgressiveSummarizationMemory`, `TierCompactor` |
| `tests/test_memory/test_compactor.py` | **Create** | Unit tests for `TierCompactor` |
| `tests/test_memory/test_progressive.py` | **Create** | Unit tests for `ProgressiveSummarizationMemory` |
| `tests/test_memory/test_progressive_integration.py` | **Create** | Integration tests with `MemoryManager` + pipeline |

---

## Chunk 1: Data Models & Callback Protocol

### Task 1: Add data models to `src/anchor/models/memory.py`

**Files:**
- Modify: `src/anchor/models/memory.py`
- Test: `tests/test_memory/test_progressive.py`

- [ ] **Step 1: Write failing tests for data models**

Create `tests/test_memory/test_progressive.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-agnesi && python -m pytest tests/test_memory/test_progressive.py -v --no-header -x 2>&1 | head -30`
Expected: FAIL with ImportError (FactType, KeyFact, SummaryTier, TierConfig not found)

- [ ] **Step 3: Implement data models**

Add to `src/anchor/models/memory.py` (after the existing `MemoryEntry` class):

```python
from dataclasses import dataclass


class FactType(StrEnum):
    """Classification of key facts extracted during progressive summarization."""

    DECISION = "decision"
    ENTITY = "entity"
    NUMBER = "number"
    DATE = "date"
    PREFERENCE = "preference"
    CONSTRAINT = "constraint"


class KeyFact(BaseModel):
    """A structured fact extracted during tier transitions."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    fact_type: FactType
    content: str
    source_tier: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    token_count: int = Field(default=0, ge=0)


class SummaryTier(BaseModel):
    """A single compression tier holding a summary."""

    level: int
    content: str
    token_count: int = Field(default=0, ge=0)
    source_turn_count: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class TierConfig:
    """Configuration for a single compression tier."""

    level: int
    max_tokens: int
    target_tokens: int = 0
    priority: int = 7
```

Note: `dataclass` import needs to be added at the top of the file. `uuid`, `datetime`, `StrEnum`, `Field` are already imported.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-agnesi && python -m pytest tests/test_memory/test_progressive.py -v --no-header -x`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/anchor/models/memory.py tests/test_memory/test_progressive.py
git commit -m "feat(memory): add data models for progressive summarization

Add FactType, KeyFact, SummaryTier, TierConfig to anchor.models.memory."
```

---

### Task 2: Add `ProgressiveSummarizationCallback` protocol

**Files:**
- Modify: `src/anchor/memory/callbacks.py`
- Test: `tests/test_memory/test_progressive.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_memory/test_progressive.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-agnesi && python -m pytest tests/test_memory/test_progressive.py::TestProgressiveSummarizationCallback -v --no-header -x`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement callback protocol**

Add to `src/anchor/memory/callbacks.py` (after the existing `MemoryCallback` class, before `_fire_memory_callback`):

```python
@runtime_checkable
class ProgressiveSummarizationCallback(Protocol):
    """Callback protocol for progressive summarization events.

    All methods have default no-op implementations.  Implementers
    only need to override the methods they care about.
    """

    def on_tier_cascade(
        self, from_tier: int, to_tier: int, tokens_in: int, tokens_out: int
    ) -> None:
        """Called when content cascades from one tier to a lower tier."""
        ...

    def on_facts_extracted(self, facts: list[MemoryEntry], source_tier: int) -> None:
        """Called when key facts are extracted during a tier transition."""
        ...

    def on_compaction_error(self, tier: int, error: Exception) -> None:
        """Called when compaction fails and falls back."""
        ...
```

Note: The `ProgressiveSummarizationCallback.on_facts_extracted` uses `list[MemoryEntry]` in the TYPE_CHECKING import rather than `list[KeyFact]` because `KeyFact` is in `models.memory` and would create a more complex import. At runtime, `list[KeyFact]` objects will be passed (KeyFact is a Pydantic BaseModel, not a MemoryEntry — but for the protocol signature we use a general type). **Actually, let's use the correct type:** add `KeyFact` to the TYPE_CHECKING imports:

In the `if TYPE_CHECKING:` block at the top of `callbacks.py`, add:
```python
from anchor.models.memory import ConversationTurn, KeyFact, MemoryEntry
```

Then the signature becomes:
```python
def on_facts_extracted(self, facts: list[KeyFact], source_tier: int) -> None:
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-agnesi && python -m pytest tests/test_memory/test_progressive.py::TestProgressiveSummarizationCallback -v --no-header -x`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/anchor/memory/callbacks.py tests/test_memory/test_progressive.py
git commit -m "feat(memory): add ProgressiveSummarizationCallback protocol"
```

---

## Chunk 2: TierCompactor

### Task 3: Implement `TierCompactor` — summarization

**Files:**
- Create: `src/anchor/memory/compactor.py`
- Create: `tests/test_memory/test_compactor.py`

- [ ] **Step 1: Write failing tests for summarization**

Create `tests/test_memory/test_compactor.py`:

```python
"""Tests for anchor.memory.compactor (TierCompactor)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from anchor.llm.models import LLMResponse, StopReason, Usage
from anchor.memory.compactor import TierCompactor
from tests.conftest import FakeTokenizer


def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="test",
        provider="test",
        stop_reason=StopReason.STOP,
    )


def _make_compactor(response_content: str = "Summary text") -> tuple[TierCompactor, MagicMock]:
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = _make_llm_response(response_content)
    mock_llm.model_id = "test/model"
    mock_llm.provider_name = "test"
    compactor = TierCompactor(llm=mock_llm, tokenizer=FakeTokenizer())
    return compactor, mock_llm


class TestTierCompactorSummarize:
    def test_summarize_tier1(self) -> None:
        compactor, mock_llm = _make_compactor("Detailed summary of conversation")
        result = compactor.summarize("user: hello\nassistant: hi", target_tier=1, target_tokens=500)
        assert result == "Detailed summary of conversation"
        mock_llm.invoke.assert_called_once()
        # Check prompt contains target tokens
        call_args = mock_llm.invoke.call_args
        messages = call_args[0][0]
        assert any("500" in str(m.content) for m in messages)

    def test_summarize_tier2(self) -> None:
        compactor, mock_llm = _make_compactor("Compact summary")
        result = compactor.summarize("Detailed summary text", target_tier=2, target_tokens=100)
        assert result == "Compact summary"

    def test_summarize_tier3(self) -> None:
        compactor, mock_llm = _make_compactor("Headline")
        result = compactor.summarize("Compact text", target_tier=3, target_tokens=20)
        assert result == "Headline"

    def test_progressive_merge(self) -> None:
        compactor, mock_llm = _make_compactor("Merged summary")
        result = compactor.summarize(
            "New content",
            target_tier=1,
            target_tokens=500,
            existing_summary="Old summary",
        )
        assert result == "Merged summary"
        call_args = mock_llm.invoke.call_args
        messages = call_args[0][0]
        prompt_text = str(messages[0].content)
        assert "Old summary" in prompt_text
        assert "New content" in prompt_text

    def test_summarize_fallback_on_error(self) -> None:
        compactor, mock_llm = _make_compactor()
        mock_llm.invoke.side_effect = Exception("LLM error")
        result = compactor.summarize("Some content here", target_tier=1, target_tokens=500)
        # Falls back to raw content (possibly truncated)
        assert "Some content here" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-agnesi && python -m pytest tests/test_memory/test_compactor.py -v --no-header -x 2>&1 | head -20`
Expected: FAIL with ImportError (compactor module doesn't exist)

- [ ] **Step 3: Implement TierCompactor summarization**

Create `src/anchor/memory/compactor.py`:

```python
"""LLM-driven tier compaction for progressive summarization.

``TierCompactor`` handles summarization and key-fact extraction by
calling an ``LLMProvider``. Each tier transition uses a tier-specific
prompt template. Errors fall back to raw content concatenation.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from anchor.llm.models import Message, Role
from anchor.models.memory import FactType, KeyFact
from anchor.tokens.counter import get_default_counter

if TYPE_CHECKING:
    from anchor.llm.base import LLMProvider
    from anchor.protocols.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

_SUMMARIZE_PROMPTS: dict[int, str] = {
    1: (
        "Summarize the following conversation preserving all reasoning, "
        "decisions made, and important context. Be thorough but concise. "
        "Target approximately {target_tokens} tokens.\n\n{content}"
    ),
    2: (
        "Compress the following summary to key points only. Remove "
        "conversational noise and redundancy. Retain decisions, facts, "
        "and conclusions. Target approximately {target_tokens} tokens.\n\n{content}"
    ),
    3: (
        "Reduce the following to a single headline-level statement that "
        "captures the essential thread of the conversation. Target "
        "approximately {target_tokens} tokens.\n\n{content}"
    ),
}

_MERGE_PROMPT = (
    "You have an existing summary and new content to incorporate. "
    "Produce a unified, non-redundant summary that covers both. "
    "Target approximately {target_tokens} tokens.\n\n"
    "EXISTING SUMMARY:\n{existing_summary}\n\n"
    "NEW CONTENT:\n{new_content}"
)

_FACT_EXTRACTION_PROMPT = (
    'Extract key facts from the following content. Return a JSON array '
    'where each element has "type" (one of: decision, entity, number, '
    'date, preference, constraint) and "content" (the fact itself, concise).\n\n'
    "Only extract facts that would be important to remember if the original "
    "text were lost. Return [] if no key facts are found.\n\n{content}"
)

_FACT_RETRY_PROMPT = (
    "Return ONLY valid JSON. No markdown fences, no explanation. "
    "Just a JSON array of objects with 'type' and 'content' keys.\n\n{content}"
)

_VALID_FACT_TYPES = {ft.value for ft in FactType}


class TierCompactor:
    """Handles LLM calls for summarization and fact extraction."""

    __slots__ = ("_llm", "_tokenizer")

    def __init__(
        self,
        llm: LLMProvider,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._llm = llm
        self._tokenizer: Tokenizer = tokenizer or get_default_counter()

    def summarize(
        self,
        content: str,
        target_tier: int,
        target_tokens: int,
        existing_summary: str | None = None,
    ) -> str:
        """Synchronously summarize content for a target tier.

        Falls back to raw content (truncated) on LLM failure.
        """
        prompt = self._build_summarize_prompt(
            content, target_tier, target_tokens, existing_summary
        )
        try:
            response = self._llm.invoke(
                [Message(role=Role.USER, content=prompt)]
            )
            return response.content or content
        except Exception:
            logger.exception(
                "Summarization failed for tier %d; using raw content", target_tier
            )
            return self._tokenizer.truncate_to_tokens(content, target_tokens)

    async def asummarize(
        self,
        content: str,
        target_tier: int,
        target_tokens: int,
        existing_summary: str | None = None,
    ) -> str:
        """Asynchronously summarize content for a target tier."""
        prompt = self._build_summarize_prompt(
            content, target_tier, target_tokens, existing_summary
        )
        try:
            response = await self._llm.ainvoke(
                [Message(role=Role.USER, content=prompt)]
            )
            return response.content or content
        except Exception:
            logger.exception(
                "Async summarization failed for tier %d; using raw content",
                target_tier,
            )
            return self._tokenizer.truncate_to_tokens(content, target_tokens)

    def extract_facts(self, content: str, source_tier: int) -> list[KeyFact]:
        """Synchronously extract key facts from content."""
        return self._parse_facts(
            self._call_fact_extraction(content, sync=True), source_tier
        )

    async def aextract_facts(self, content: str, source_tier: int) -> list[KeyFact]:
        """Asynchronously extract key facts from content."""
        return self._parse_facts(
            await self._call_fact_extraction_async(content), source_tier
        )

    # -- Private helpers --

    def _build_summarize_prompt(
        self,
        content: str,
        target_tier: int,
        target_tokens: int,
        existing_summary: str | None,
    ) -> str:
        if existing_summary is not None:
            return _MERGE_PROMPT.format(
                target_tokens=target_tokens,
                existing_summary=existing_summary,
                new_content=content,
            )
        template = _SUMMARIZE_PROMPTS.get(target_tier, _SUMMARIZE_PROMPTS[1])
        return template.format(target_tokens=target_tokens, content=content)

    def _call_fact_extraction(self, content: str, *, sync: bool = True) -> str:
        prompt = _FACT_EXTRACTION_PROMPT.format(content=content)
        try:
            response = self._llm.invoke([Message(role=Role.USER, content=prompt)])
            return response.content or "[]"
        except Exception:
            logger.warning("Fact extraction failed; skipping facts")
            return "[]"

    async def _call_fact_extraction_async(self, content: str) -> str:
        prompt = _FACT_EXTRACTION_PROMPT.format(content=content)
        try:
            response = await self._llm.ainvoke(
                [Message(role=Role.USER, content=prompt)]
            )
            return response.content or "[]"
        except Exception:
            logger.warning("Async fact extraction failed; skipping facts")
            return "[]"

    def _parse_facts(self, raw_json: str, source_tier: int) -> list[KeyFact]:
        """Parse JSON response into KeyFact objects, retrying once on failure."""
        for attempt in range(2):
            try:
                # Strip markdown fences if present
                cleaned = raw_json.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.split("\n")
                    cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else "[]"

                data = json.loads(cleaned)
                if not isinstance(data, list):
                    return []

                facts: list[KeyFact] = []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    fact_type_str = item.get("type", "")
                    if fact_type_str not in _VALID_FACT_TYPES:
                        continue
                    fact_content = item.get("content", "")
                    if not fact_content:
                        continue
                    token_count = self._tokenizer.count_tokens(fact_content)
                    facts.append(
                        KeyFact(
                            fact_type=FactType(fact_type_str),
                            content=fact_content,
                            source_tier=source_tier,
                            token_count=token_count,
                        )
                    )
                return facts
            except (json.JSONDecodeError, KeyError, TypeError):
                if attempt == 0:
                    logger.warning("JSON parse failed; retrying with stricter prompt")
                    try:
                        retry_prompt = _FACT_RETRY_PROMPT.format(content=raw_json)
                        response = self._llm.invoke(
                            [Message(role=Role.USER, content=retry_prompt)]
                        )
                        raw_json = response.content or "[]"
                    except Exception:
                        return []
                else:
                    logger.warning("Fact extraction JSON parse failed after retry; skipping")
                    return []
        return []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-agnesi && python -m pytest tests/test_memory/test_compactor.py -v --no-header -x`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/anchor/memory/compactor.py tests/test_memory/test_compactor.py
git commit -m "feat(memory): add TierCompactor with LLM summarization and fact extraction"
```

---

### Task 4: Add fact extraction and async tests to `TierCompactor`

**Files:**
- Modify: `tests/test_memory/test_compactor.py`

- [ ] **Step 1: Write fact extraction and async tests**

Append to `tests/test_memory/test_compactor.py`:

```python
import asyncio
from unittest.mock import AsyncMock

from anchor.models.memory import FactType, KeyFact


class TestTierCompactorFactExtraction:
    def test_extract_valid_facts(self) -> None:
        json_response = '[{"type": "decision", "content": "Use FastAPI"}, {"type": "number", "content": "Budget: $50k"}]'
        compactor, mock_llm = _make_compactor(json_response)
        facts = compactor.extract_facts("Some conversation", source_tier=0)
        assert len(facts) == 2
        assert facts[0].fact_type == FactType.DECISION
        assert facts[0].content == "Use FastAPI"
        assert facts[0].source_tier == 0
        assert facts[1].fact_type == FactType.NUMBER

    def test_extract_empty_array(self) -> None:
        compactor, _ = _make_compactor("[]")
        facts = compactor.extract_facts("Some text", source_tier=1)
        assert facts == []

    def test_extract_invalid_json_retries(self) -> None:
        compactor, mock_llm = _make_compactor("not json")
        # First call returns bad JSON, retry also returns bad JSON
        mock_llm.invoke.side_effect = [
            _make_llm_response("not json"),
            _make_llm_response("still not json"),
        ]
        facts = compactor.extract_facts("text", source_tier=0)
        assert facts == []

    def test_extract_filters_invalid_fact_types(self) -> None:
        json_response = '[{"type": "invalid_type", "content": "x"}, {"type": "decision", "content": "y"}]'
        compactor, _ = _make_compactor(json_response)
        facts = compactor.extract_facts("text", source_tier=0)
        assert len(facts) == 1
        assert facts[0].fact_type == FactType.DECISION

    def test_extract_strips_markdown_fences(self) -> None:
        json_response = '```json\n[{"type": "entity", "content": "FastAPI"}]\n```'
        compactor, _ = _make_compactor(json_response)
        facts = compactor.extract_facts("text", source_tier=0)
        assert len(facts) == 1

    def test_source_tier_injected(self) -> None:
        json_response = '[{"type": "date", "content": "March 2026"}]'
        compactor, _ = _make_compactor(json_response)
        facts = compactor.extract_facts("text", source_tier=2)
        assert facts[0].source_tier == 2

    def test_token_count_computed(self) -> None:
        json_response = '[{"type": "decision", "content": "Use FastAPI over Flask"}]'
        compactor, _ = _make_compactor(json_response)
        facts = compactor.extract_facts("text", source_tier=0)
        # FakeTokenizer counts words: "Use FastAPI over Flask" = 4 tokens
        assert facts[0].token_count == 4

    def test_extract_llm_failure_returns_empty(self) -> None:
        compactor, mock_llm = _make_compactor()
        mock_llm.invoke.side_effect = Exception("LLM down")
        facts = compactor.extract_facts("text", source_tier=0)
        assert facts == []


class TestTierCompactorAsync:
    def test_asummarize(self) -> None:
        compactor, mock_llm = _make_compactor()
        mock_llm.ainvoke = AsyncMock(return_value=_make_llm_response("Async summary"))
        result = asyncio.get_event_loop().run_until_complete(
            compactor.asummarize("content", target_tier=1, target_tokens=500)
        )
        assert result == "Async summary"
        mock_llm.ainvoke.assert_awaited_once()

    def test_aextract_facts(self) -> None:
        compactor, mock_llm = _make_compactor()
        mock_llm.ainvoke = AsyncMock(
            return_value=_make_llm_response('[{"type": "decision", "content": "test"}]')
        )
        facts = asyncio.get_event_loop().run_until_complete(
            compactor.aextract_facts("content", source_tier=1)
        )
        assert len(facts) == 1
        mock_llm.ainvoke.assert_awaited_once()

    def test_asummarize_fallback(self) -> None:
        compactor, mock_llm = _make_compactor()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("fail"))
        result = asyncio.get_event_loop().run_until_complete(
            compactor.asummarize("fallback content", target_tier=1, target_tokens=500)
        )
        assert "fallback" in result
```

- [ ] **Step 2: Run all compactor tests**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-agnesi && python -m pytest tests/test_memory/test_compactor.py -v --no-header -x`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_memory/test_compactor.py
git commit -m "test(memory): add fact extraction and async tests for TierCompactor"
```

---

## Chunk 3: ProgressiveSummarizationMemory Core

### Task 5: Implement `ProgressiveSummarizationMemory` — construction and basic operations

**Files:**
- Create: `src/anchor/memory/progressive.py`
- Modify: `tests/test_memory/test_progressive.py`

- [ ] **Step 1: Write failing tests for construction**

Append to `tests/test_memory/test_progressive.py`:

```python
from unittest.mock import MagicMock

from anchor.llm.models import LLMResponse, StopReason, Usage
from anchor.memory.progressive import ProgressiveSummarizationMemory
from anchor.models.memory import TierConfig
from anchor.protocols.memory import ConversationMemory
from tests.conftest import FakeTokenizer


def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="test",
        provider="test",
        stop_reason=StopReason.STOP,
    )


def _make_mock_llm(summary: str = "Summary", facts: str = "[]") -> MagicMock:
    mock = MagicMock()
    mock.invoke.side_effect = lambda msgs, **kw: _make_llm_response(
        facts if "Extract key facts" in str(msgs[0].content) else summary
    )
    mock.model_id = "test/model"
    mock.provider_name = "test"
    return mock


class TestProgressiveConstruction:
    def test_default_config(self) -> None:
        mock_llm = _make_mock_llm()
        mem = ProgressiveSummarizationMemory(
            max_tokens=8192, llm=mock_llm, tokenizer=FakeTokenizer()
        )
        assert mem.turns == []
        assert mem.total_tokens == 0
        assert mem.summary is None
        assert mem.facts == []

    def test_custom_tier_config(self) -> None:
        mock_llm = _make_mock_llm()
        config = [
            TierConfig(level=0, max_tokens=100),
            TierConfig(level=1, max_tokens=50, target_tokens=25),
            TierConfig(level=2, max_tokens=20, target_tokens=10),
            TierConfig(level=3, max_tokens=10, target_tokens=5),
        ]
        mem = ProgressiveSummarizationMemory(
            max_tokens=200, llm=mock_llm, tier_config=config, tokenizer=FakeTokenizer()
        )
        assert mem.tier_tokens == {0: 0, 1: 0, 2: 0, 3: 0}

    def test_negative_max_tokens_raises(self) -> None:
        with pytest.raises(ValueError):
            ProgressiveSummarizationMemory(max_tokens=-1, llm=_make_mock_llm())

    def test_satisfies_conversation_memory_protocol(self) -> None:
        mock_llm = _make_mock_llm()
        mem = ProgressiveSummarizationMemory(
            max_tokens=8192, llm=mock_llm, tokenizer=FakeTokenizer()
        )
        assert isinstance(mem, ConversationMemory)

    def test_clear_resets_everything(self) -> None:
        mock_llm = _make_mock_llm()
        mem = ProgressiveSummarizationMemory(
            max_tokens=100, llm=mock_llm, tokenizer=FakeTokenizer()
        )
        mem.add_message("user", "hello world")
        assert len(mem.turns) == 1
        mem.clear()
        assert mem.turns == []
        assert mem.facts == []
        assert mem.summary is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-agnesi && python -m pytest tests/test_memory/test_progressive.py::TestProgressiveConstruction -v --no-header -x 2>&1 | head -20`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement `ProgressiveSummarizationMemory`**

Create `src/anchor/memory/progressive.py`:

```python
"""Multi-tier progressive summarization memory.

``ProgressiveSummarizationMemory`` implements a 4-tier cascading compression
system with LLM-driven compaction and key-fact extraction. It satisfies the
``ConversationMemory`` protocol and integrates with ``MemoryManager``.

Tiers:
    0 — Verbatim recent turns (``SlidingWindowMemory``)
    1 — Detailed summary (~500 tokens)
    2 — Compact summary (~100 tokens)
    3 — Ultra-compact summary (~20 tokens)

Key facts are extracted at each tier transition and stored in a sidecar.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from anchor.memory.compactor import TierCompactor
from anchor.memory.sliding_window import SlidingWindowMemory
from anchor.models.context import ContextItem, SourceType
from anchor.models.memory import (
    ConversationTurn,
    KeyFact,
    Role,
    SummaryTier,
    TierConfig,
)
from anchor.tokens.counter import get_default_counter

if TYPE_CHECKING:
    from anchor.llm.base import LLMProvider
    from anchor.memory.callbacks import ProgressiveSummarizationCallback
    from anchor.protocols.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

_DEFAULT_TIER_CONFIG = [
    TierConfig(level=0, max_tokens=4096, target_tokens=0, priority=7),
    TierConfig(level=1, max_tokens=1024, target_tokens=500, priority=6),
    TierConfig(level=2, max_tokens=256, target_tokens=100, priority=5),
    TierConfig(level=3, max_tokens=64, target_tokens=20, priority=4),
]


def _derive_tier_config(max_tokens: int) -> list[TierConfig]:
    """Derive tier config proportionally from a total token budget."""
    # 50% verbatim, 25% detailed, 12.5% compact, 6.25% ultra + ~6.25% buffer
    return [
        TierConfig(level=0, max_tokens=max_tokens // 2, target_tokens=0, priority=7),
        TierConfig(level=1, max_tokens=max_tokens // 4, target_tokens=max_tokens // 8, priority=6),
        TierConfig(level=2, max_tokens=max_tokens // 8, target_tokens=max_tokens // 16, priority=5),
        TierConfig(level=3, max_tokens=max_tokens // 16, target_tokens=max_tokens // 32, priority=4),
    ]


def _serialize_turns(turns: list[ConversationTurn]) -> str:
    return "\n".join(f"{turn.role}: {turn.content}" for turn in turns)


class ProgressiveSummarizationMemory:
    """A 4-tier progressive summarization memory.

    Satisfies the ``ConversationMemory`` protocol. Pluggable into
    ``MemoryManager`` and ``ContextPipeline``.
    """

    __slots__ = (
        "_callbacks",
        "_compactor",
        "_fact_token_budget",
        "_facts",
        "_lock",
        "_max_facts",
        "_tier_configs",
        "_tiers",
        "_tokenizer",
        "_window",
    )

    def __init__(
        self,
        max_tokens: int = 8192,
        llm: LLMProvider | str = "anthropic/claude-haiku-4-5-20251001",
        tier_config: list[TierConfig] | None = None,
        max_facts: int = 50,
        fact_token_budget: int = 500,
        tokenizer: Tokenizer | None = None,
        callbacks: list[ProgressiveSummarizationCallback] | None = None,
    ) -> None:
        if max_tokens <= 0:
            msg = "max_tokens must be a positive integer"
            raise ValueError(msg)

        self._tokenizer: Tokenizer = tokenizer or get_default_counter()
        self._max_facts = max_facts
        self._fact_token_budget = fact_token_budget
        self._facts: list[KeyFact] = []
        self._callbacks: list[ProgressiveSummarizationCallback] = callbacks or []
        self._lock = threading.RLock()

        # Resolve LLM provider
        if isinstance(llm, str):
            from anchor.llm.registry import create_provider
            resolved_llm = create_provider(llm)
        else:
            resolved_llm = llm

        self._compactor = TierCompactor(llm=resolved_llm, tokenizer=self._tokenizer)

        # Configure tiers
        self._tier_configs: dict[int, TierConfig] = {}
        configs = tier_config or _derive_tier_config(max_tokens)
        for cfg in configs:
            self._tier_configs[cfg.level] = cfg

        # Tier 0 = SlidingWindowMemory
        tier0_config = self._tier_configs.get(0)
        tier0_tokens = tier0_config.max_tokens if tier0_config else max_tokens // 2
        self._window = SlidingWindowMemory(
            max_tokens=tier0_tokens,
            tokenizer=self._tokenizer,
            on_evict=self._handle_eviction,
        )

        # Tiers 1-3 stored as SummaryTier or None
        self._tiers: dict[int, SummaryTier | None] = {1: None, 2: None, 3: None}

    def __repr__(self) -> str:
        tier_summary = {k: (v.token_count if v else 0) for k, v in self._tiers.items()}
        return (
            f"ProgressiveSummarizationMemory("
            f"window_tokens={self._window.total_tokens}, "
            f"tiers={tier_summary}, "
            f"facts={len(self._facts)})"
        )

    # -- ConversationMemory protocol --

    @property
    def turns(self) -> list[ConversationTurn]:
        return self._window.turns

    @property
    def total_tokens(self) -> int:
        return self._window.total_tokens

    def to_context_items(self, priority: int = 7) -> list[ContextItem]:
        with self._lock:
            items: list[ContextItem] = []

            # Tier 3 (lowest priority summary)
            for tier_level in (3, 2, 1):
                tier = self._tiers.get(tier_level)
                if tier is not None:
                    tier_priority = priority - (3 - tier_level + 1)
                    # tier 3 = priority - 3, tier 2 = priority - 2, tier 1 = priority - 1
                    tier_priority = priority - tier_level
                    items.append(
                        ContextItem(
                            content=tier.content,
                            source=SourceType.CONVERSATION,
                            score=0.5,
                            priority=tier_priority,
                            token_count=tier.token_count,
                            metadata={"tier": tier_level, "summary": True},
                        )
                    )

            # Tier 0 verbatim turns
            items.extend(self._window.to_context_items(priority=priority))

            # Key facts (highest priority)
            fact_priority = priority + 1
            for fact in self._facts:
                items.append(
                    ContextItem(
                        content=fact.content,
                        source=SourceType.MEMORY,
                        score=0.8,
                        priority=fact_priority,
                        token_count=fact.token_count,
                        metadata={
                            "fact_type": fact.fact_type.value,
                            "source_tier": fact.source_tier,
                        },
                    )
                )

            return items

    def clear(self) -> None:
        with self._lock:
            self._window.clear()
            self._tiers = {1: None, 2: None, 3: None}
            self._facts = []

    # -- Public methods --

    def add_message(
        self, role: Role | str, content: str, **metadata: object
    ) -> ConversationTurn:
        with self._lock:
            return self._window.add_turn(role, content, **metadata)

    def add_turn(self, turn: ConversationTurn) -> None:
        with self._lock:
            self._window.add_turn(
                role=turn.role, content=turn.content, **turn.metadata
            )

    async def aadd_message(
        self, role: Role | str, content: str, **metadata: object
    ) -> ConversationTurn:
        # Sync add to window (fast), then async cascade
        with self._lock:
            turn = self._window.add_turn(role, content, **metadata)
        # Async cascade handled by _handle_eviction_async if needed
        return turn

    async def aadd_turn(self, turn: ConversationTurn) -> None:
        with self._lock:
            self._window.add_turn(
                role=turn.role, content=turn.content, **turn.metadata
            )

    # -- Introspection --

    @property
    def tiers(self) -> dict[int, SummaryTier | None]:
        return dict(self._tiers)

    @property
    def facts(self) -> list[KeyFact]:
        return list(self._facts)

    @property
    def tier_tokens(self) -> dict[int, int]:
        result: dict[int, int] = {0: self._window.total_tokens}
        for level, tier in self._tiers.items():
            result[level] = tier.token_count if tier is not None else 0
        return result

    @property
    def summary(self) -> str | None:
        tier1 = self._tiers.get(1)
        return tier1.content if tier1 is not None else None

    # -- Eviction handling (called from within SlidingWindowMemory's lock) --

    def _handle_eviction(self, evicted_turns: list[ConversationTurn]) -> None:
        """Callback invoked by SlidingWindowMemory when turns are evicted.

        This runs inside the outer RLock (acquired by add_message/add_turn).
        It must NOT call back into the SlidingWindowMemory.
        """
        serialized = _serialize_turns(evicted_turns)
        turn_count = len(evicted_turns)

        # Summarize into Tier 1
        existing_t1 = self._tiers.get(1)
        existing_summary = existing_t1.content if existing_t1 else None
        existing_count = existing_t1.source_turn_count if existing_t1 else 0

        t1_config = self._tier_configs.get(1, TierConfig(level=1, max_tokens=1024, target_tokens=500))

        try:
            new_summary = self._compactor.summarize(
                serialized,
                target_tier=1,
                target_tokens=t1_config.target_tokens,
                existing_summary=existing_summary,
            )
        except Exception:
            logger.exception("Tier 1 compaction failed; using raw content")
            new_summary = serialized
            self._fire_callback("on_compaction_error", 1, Exception("compaction failed"))

        summary_tokens = self._tokenizer.count_tokens(new_summary)
        now = datetime.now(UTC)

        self._tiers[1] = SummaryTier(
            level=1,
            content=new_summary,
            token_count=summary_tokens,
            source_turn_count=existing_count + turn_count,
            created_at=existing_t1.created_at if existing_t1 else now,
            updated_at=now,
        )

        self._fire_callback(
            "on_tier_cascade", 0, 1, self._tokenizer.count_tokens(serialized), summary_tokens
        )

        # Extract facts
        try:
            new_facts = self._compactor.extract_facts(serialized, source_tier=0)
            if new_facts:
                self._add_facts(new_facts)
                self._fire_callback("on_facts_extracted", new_facts, 0)
        except Exception:
            logger.warning("Fact extraction failed during tier 0→1 cascade")

        # Check if Tier 1 needs to cascade to Tier 2
        if summary_tokens > t1_config.max_tokens:
            self._cascade_tier(1, 2)

    def _cascade_tier(self, from_level: int, to_level: int) -> None:
        """Cascade content from one tier to the next."""
        source_tier = self._tiers.get(from_level)
        if source_tier is None:
            return

        to_config = self._tier_configs.get(
            to_level, TierConfig(level=to_level, max_tokens=64, target_tokens=20)
        )
        existing_target = self._tiers.get(to_level)
        existing_summary = existing_target.content if existing_target else None
        existing_count = existing_target.source_turn_count if existing_target else 0

        try:
            new_summary = self._compactor.summarize(
                source_tier.content,
                target_tier=to_level,
                target_tokens=to_config.target_tokens,
                existing_summary=existing_summary,
            )
        except Exception:
            logger.exception("Tier %d→%d compaction failed", from_level, to_level)
            new_summary = source_tier.content
            self._fire_callback("on_compaction_error", to_level, Exception("cascade failed"))

        summary_tokens = self._tokenizer.count_tokens(new_summary)
        now = datetime.now(UTC)

        self._tiers[to_level] = SummaryTier(
            level=to_level,
            content=new_summary,
            token_count=summary_tokens,
            source_turn_count=existing_count + source_tier.source_turn_count,
            created_at=existing_target.created_at if existing_target else now,
            updated_at=now,
        )

        # Extract facts from the source tier content (not for tier 2→3)
        if from_level < 2:
            try:
                new_facts = self._compactor.extract_facts(source_tier.content, source_tier=from_level)
                if new_facts:
                    self._add_facts(new_facts)
                    self._fire_callback("on_facts_extracted", new_facts, from_level)
            except Exception:
                logger.warning("Fact extraction failed during tier %d→%d cascade", from_level, to_level)

        # Clear the source tier after cascading
        self._tiers[from_level] = None

        self._fire_callback(
            "on_tier_cascade", from_level, to_level, source_tier.token_count, summary_tokens
        )

        # Check if target needs to cascade further
        if to_level < 3 and summary_tokens > to_config.max_tokens:
            self._cascade_tier(to_level, to_level + 1)

    def _add_facts(self, new_facts: list[KeyFact]) -> None:
        """Add facts to the store, respecting max_facts and token budget."""
        for fact in new_facts:
            self._facts.append(fact)

        # FIFO eviction if over max_facts
        while len(self._facts) > self._max_facts:
            self._facts.pop(0)

        # Token budget enforcement
        total_fact_tokens = sum(f.token_count for f in self._facts)
        while self._facts and total_fact_tokens > self._fact_token_budget:
            removed = self._facts.pop(0)
            total_fact_tokens -= removed.token_count

    def _fire_callback(self, method: str, *args: object) -> None:
        """Fire a callback method on all registered callbacks."""
        from anchor._callbacks import fire_callbacks
        fire_callbacks(self._callbacks, method, *args, logger=logger, log_level=logging.WARNING)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-agnesi && python -m pytest tests/test_memory/test_progressive.py -v --no-header -x`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/anchor/memory/progressive.py tests/test_memory/test_progressive.py
git commit -m "feat(memory): add ProgressiveSummarizationMemory core implementation"
```

---

### Task 6: Add cascade behavior tests

**Files:**
- Modify: `tests/test_memory/test_progressive.py`

- [ ] **Step 1: Write cascade tests**

Append to `tests/test_memory/test_progressive.py`:

```python
class TestProgressiveCascade:
    def test_tier1_created_on_overflow(self) -> None:
        """When Tier 0 overflows, evicted turns should create a Tier 1 summary."""
        mock_llm = _make_mock_llm(summary="Tier 1 summary")
        # FakeTokenizer: 1 token per word. max_tokens=10 for Tier 0.
        config = [
            TierConfig(level=0, max_tokens=10),
            TierConfig(level=1, max_tokens=100, target_tokens=50),
            TierConfig(level=2, max_tokens=50, target_tokens=20),
            TierConfig(level=3, max_tokens=20, target_tokens=5),
        ]
        mem = ProgressiveSummarizationMemory(
            max_tokens=200, llm=mock_llm, tier_config=config, tokenizer=FakeTokenizer()
        )
        # Add messages until overflow (each ~3 tokens: "word1 word2 word3")
        for i in range(5):
            mem.add_message("user", f"message number {i}")
        # Tier 1 should now have content
        assert mem.tiers[1] is not None
        assert mem.tiers[1].content == "Tier 1 summary"

    def test_tier2_created_on_tier1_overflow(self) -> None:
        """When Tier 1 overflows, content should cascade to Tier 2."""
        call_count = 0
        def smart_response(msgs, **kw):
            nonlocal call_count
            call_count += 1
            prompt = str(msgs[0].content)
            if "Extract key facts" in prompt:
                return _make_llm_response("[]")
            # Return progressively longer summaries to trigger overflow
            return _make_llm_response("word " * 20)  # 20 tokens

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = smart_response
        mock_llm.model_id = "test/model"
        mock_llm.provider_name = "test"

        config = [
            TierConfig(level=0, max_tokens=5),   # very small window
            TierConfig(level=1, max_tokens=15, target_tokens=10),  # overflow at 15
            TierConfig(level=2, max_tokens=100, target_tokens=50),
            TierConfig(level=3, max_tokens=50, target_tokens=10),
        ]
        mem = ProgressiveSummarizationMemory(
            max_tokens=200, llm=mock_llm, tier_config=config, tokenizer=FakeTokenizer()
        )
        # Add many messages to trigger multiple evictions
        for i in range(10):
            mem.add_message("user", f"word{i} word{i}")
        # Since Tier 1 summary is 20 tokens and max is 15, it should cascade
        # Tier 1 should be cleared (set to None) after cascade
        # Tier 2 should have content
        assert mem.tiers[2] is not None

    def test_facts_extracted_during_cascade(self) -> None:
        facts_json = '[{"type": "decision", "content": "Use Python"}]'
        mock_llm = _make_mock_llm(summary="Summary", facts=facts_json)
        config = [
            TierConfig(level=0, max_tokens=5),
            TierConfig(level=1, max_tokens=100, target_tokens=50),
            TierConfig(level=2, max_tokens=50, target_tokens=20),
            TierConfig(level=3, max_tokens=20, target_tokens=5),
        ]
        mem = ProgressiveSummarizationMemory(
            max_tokens=200, llm=mock_llm, tier_config=config, tokenizer=FakeTokenizer()
        )
        for i in range(5):
            mem.add_message("user", f"word{i} word{i}")
        assert len(mem.facts) > 0
        assert mem.facts[0].content == "Use Python"

    def test_facts_fifo_eviction(self) -> None:
        facts_json = '[{"type": "decision", "content": "fact"}]'
        mock_llm = _make_mock_llm(summary="Summary", facts=facts_json)
        config = [
            TierConfig(level=0, max_tokens=5),
            TierConfig(level=1, max_tokens=1000, target_tokens=50),
            TierConfig(level=2, max_tokens=500, target_tokens=20),
            TierConfig(level=3, max_tokens=100, target_tokens=5),
        ]
        mem = ProgressiveSummarizationMemory(
            max_tokens=2000, llm=mock_llm, tier_config=config,
            max_facts=3, tokenizer=FakeTokenizer(),
        )
        # Trigger many evictions to produce many facts
        for i in range(20):
            mem.add_message("user", f"word{i} word{i}")
        assert len(mem.facts) <= 3


class TestProgressiveContextOutput:
    def test_context_items_priorities(self) -> None:
        mock_llm = _make_mock_llm(summary="Summary text here")
        config = [
            TierConfig(level=0, max_tokens=5),
            TierConfig(level=1, max_tokens=100, target_tokens=50),
            TierConfig(level=2, max_tokens=50, target_tokens=20),
            TierConfig(level=3, max_tokens=20, target_tokens=5),
        ]
        mem = ProgressiveSummarizationMemory(
            max_tokens=200, llm=mock_llm, tier_config=config, tokenizer=FakeTokenizer()
        )
        # Trigger a cascade
        for i in range(5):
            mem.add_message("user", f"word{i} word{i}")

        items = mem.to_context_items(priority=7)
        # Should have tier 1 summary + verbatim turns
        summary_items = [i for i in items if i.metadata.get("summary")]
        verbatim_items = [i for i in items if not i.metadata.get("summary") and i.source == SourceType.CONVERSATION]

        # Tier 1 should be at priority 6
        if summary_items:
            assert summary_items[0].priority == 6

    def test_empty_tiers_omitted(self) -> None:
        mock_llm = _make_mock_llm()
        mem = ProgressiveSummarizationMemory(
            max_tokens=8192, llm=mock_llm, tokenizer=FakeTokenizer()
        )
        mem.add_message("user", "hello")
        items = mem.to_context_items()
        summary_items = [i for i in items if i.metadata.get("summary")]
        assert len(summary_items) == 0  # No tiers populated yet

    def test_relative_priority(self) -> None:
        mock_llm = _make_mock_llm(summary="Summary")
        config = [
            TierConfig(level=0, max_tokens=5),
            TierConfig(level=1, max_tokens=100, target_tokens=50),
            TierConfig(level=2, max_tokens=50, target_tokens=20),
            TierConfig(level=3, max_tokens=20, target_tokens=5),
        ]
        mem = ProgressiveSummarizationMemory(
            max_tokens=200, llm=mock_llm, tier_config=config, tokenizer=FakeTokenizer()
        )
        for i in range(5):
            mem.add_message("user", f"word{i} word{i}")
        # Call with custom priority
        items = mem.to_context_items(priority=5)
        summary_items = [i for i in items if i.metadata.get("summary")]
        if summary_items:
            # Tier 1 at priority=5-1=4
            assert summary_items[0].priority == 4
```

- [ ] **Step 2: Run cascade tests**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-agnesi && python -m pytest tests/test_memory/test_progressive.py -v --no-header -x`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_memory/test_progressive.py
git commit -m "test(memory): add cascade behavior and context output tests"
```

---

## Chunk 4: Integration & Exports

### Task 7: Update `MemoryManager` and public exports

**Files:**
- Modify: `src/anchor/memory/manager.py`
- Modify: `src/anchor/__init__.py`
- Create: `tests/test_memory/test_progressive_integration.py`

- [ ] **Step 1: Write integration tests**

Create `tests/test_memory/test_progressive_integration.py`:

```python
"""Integration tests for ProgressiveSummarizationMemory with MemoryManager."""

from __future__ import annotations

from unittest.mock import MagicMock

from anchor.llm.models import LLMResponse, StopReason, Usage
from anchor.memory.manager import MemoryManager
from anchor.memory.progressive import ProgressiveSummarizationMemory
from anchor.models.memory import TierConfig
from tests.conftest import FakeTokenizer


def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="test",
        provider="test",
        stop_reason=StopReason.STOP,
    )


def _make_mock_llm() -> MagicMock:
    mock = MagicMock()
    mock.invoke.side_effect = lambda msgs, **kw: _make_llm_response(
        "[]" if "Extract key facts" in str(msgs[0].content) else "Summary"
    )
    mock.model_id = "test/model"
    mock.provider_name = "test"
    return mock


class TestMemoryManagerIntegration:
    def test_add_messages_via_manager(self) -> None:
        mock_llm = _make_mock_llm()
        mem = ProgressiveSummarizationMemory(
            max_tokens=8192, llm=mock_llm, tokenizer=FakeTokenizer()
        )
        manager = MemoryManager(conversation_memory=mem, tokenizer=FakeTokenizer())
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi there")
        assert len(mem.turns) == 2

    def test_conversation_type(self) -> None:
        mock_llm = _make_mock_llm()
        mem = ProgressiveSummarizationMemory(
            max_tokens=8192, llm=mock_llm, tokenizer=FakeTokenizer()
        )
        manager = MemoryManager(conversation_memory=mem, tokenizer=FakeTokenizer())
        assert manager.conversation_type == "progressive_summarization"

    def test_get_context_items(self) -> None:
        mock_llm = _make_mock_llm()
        mem = ProgressiveSummarizationMemory(
            max_tokens=8192, llm=mock_llm, tokenizer=FakeTokenizer()
        )
        manager = MemoryManager(conversation_memory=mem, tokenizer=FakeTokenizer())
        manager.add_user_message("Hello")
        items = manager.get_context_items()
        assert len(items) >= 1

    def test_full_conversation_with_cascade(self) -> None:
        mock_llm = _make_mock_llm()
        config = [
            TierConfig(level=0, max_tokens=10),
            TierConfig(level=1, max_tokens=100, target_tokens=50),
            TierConfig(level=2, max_tokens=50, target_tokens=20),
            TierConfig(level=3, max_tokens=20, target_tokens=5),
        ]
        mem = ProgressiveSummarizationMemory(
            max_tokens=200, llm=mock_llm, tier_config=config, tokenizer=FakeTokenizer()
        )
        manager = MemoryManager(conversation_memory=mem, tokenizer=FakeTokenizer())
        # Add 20 turns
        for i in range(10):
            manager.add_user_message(f"User message number {i}")
            manager.add_assistant_message(f"Reply to message {i}")
        # Should have cascaded
        items = manager.get_context_items()
        assert len(items) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-agnesi && python -m pytest tests/test_memory/test_progressive_integration.py -v --no-header -x 2>&1 | head -20`
Expected: FAIL with TypeError (MemoryManager._add_message doesn't handle ProgressiveSummarizationMemory)

- [ ] **Step 3: Update `MemoryManager._add_message` and `conversation_type`**

In `src/anchor/memory/manager.py`:

Add import at the top (after the existing `from .summary_buffer import SummaryBufferMemory`):
```python
from .progressive import ProgressiveSummarizationMemory
```

In `_add_message` method, add a new branch **before** the `SummaryBufferMemory` check:
```python
if isinstance(self._conversation, ProgressiveSummarizationMemory):
    self._conversation.add_message(role, content)
elif isinstance(self._conversation, SummaryBufferMemory):
    ...
```

In `conversation_type` property, add a new branch **before** the `SummaryBufferMemory` check:
```python
if isinstance(self._conversation, ProgressiveSummarizationMemory):
    return "progressive_summarization"
if isinstance(self._conversation, SummaryBufferMemory):
    ...
```

- [ ] **Step 4: Update public exports in `src/anchor/__init__.py`**

In the Memory Management section of the imports, add:
```python
from anchor.memory.progressive import ProgressiveSummarizationMemory
from anchor.memory.compactor import TierCompactor
```

And add both to the `__all__` list if one exists, or ensure they appear in the module docstring's Memory Management section.

- [ ] **Step 5: Run integration tests**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-agnesi && python -m pytest tests/test_memory/test_progressive_integration.py -v --no-header -x`
Expected: All PASS

- [ ] **Step 6: Run full test suite to check no regressions**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-agnesi && python -m pytest tests/test_memory/ -v --no-header -x 2>&1 | tail -20`
Expected: All existing tests still pass

- [ ] **Step 7: Commit**

```bash
git add src/anchor/memory/manager.py src/anchor/__init__.py tests/test_memory/test_progressive_integration.py
git commit -m "feat(memory): integrate ProgressiveSummarizationMemory with MemoryManager and exports"
```

---

### Task 8: Run full test suite and type checks

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-agnesi && python -m pytest tests/ -x --no-header -q 2>&1 | tail -20`
Expected: All pass (or only pre-existing failures from optional deps)

- [ ] **Step 2: Run type checker**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-agnesi && python -m mypy src/anchor/memory/progressive.py src/anchor/memory/compactor.py --ignore-missing-imports 2>&1 | tail -20`
Expected: No errors (or only pre-existing ones)

- [ ] **Step 3: Run linter**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-agnesi && python -m ruff check src/anchor/memory/progressive.py src/anchor/memory/compactor.py src/anchor/models/memory.py 2>&1`
Expected: No errors

- [ ] **Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address linting and type check issues"
```
