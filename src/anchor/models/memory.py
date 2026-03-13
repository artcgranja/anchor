"""Memory models for conversation history and persistent memory."""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from anchor.llm.models import Role


class MemoryType(StrEnum):
    """Classification of memory entries by cognitive type."""

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    CONVERSATION = "conversation"


class ConversationTurn(BaseModel):
    """A single turn in a conversation (user or assistant message)."""

    role: Role
    content: str
    token_count: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


def _compute_content_hash(content: str) -> str:
    """Compute MD5 hash of content for deduplication."""
    return hashlib.md5(content.encode()).hexdigest()  # noqa: S324


class MemoryEntry(BaseModel):
    """A persistent memory entry with relevance tracking.

    Inspired by Mem0's memory entries with priority scoring.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    access_count: int = Field(default=0, ge=0)
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(UTC))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Phase 1 additions
    memory_type: MemoryType = MemoryType.SEMANTIC
    user_id: str | None = None
    session_id: str | None = None
    expires_at: datetime | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    content_hash: str = Field(default="")
    source_turns: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _set_content_hash(self) -> MemoryEntry:
        """Compute content_hash from content if not explicitly provided."""
        if not self.content_hash:
            object.__setattr__(self, "content_hash", _compute_content_hash(self.content))
        return self

    @property
    def is_expired(self) -> bool:
        """Check whether this memory entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(UTC) >= self.expires_at

    def touch(self) -> MemoryEntry:
        """Return a copy with incremented access_count and refreshed last_accessed."""
        return self.model_copy(
            update={
                "access_count": self.access_count + 1,
                "last_accessed": datetime.now(UTC),
            }
        )
