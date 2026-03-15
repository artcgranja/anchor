# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `GraphStore` and `AsyncGraphStore` protocols for persistent graph storage (nodes, edges, BFS traversal, relation filtering, memory linking)
- `ConversationStore` and `AsyncConversationStore` protocols for persistent conversation history (append-only turns, session scoping, summary tier persistence)
- `InMemoryGraphStore` — thread-safe, dict-backed graph store with cycle-safe BFS
- `InMemoryConversationStore` — thread-safe, session-scoped conversation store
- `SqliteGraphStore` and `AsyncSqliteGraphStore` — SQLite-backed graph persistence with parameterized queries
- `SqliteConversationStore` and `AsyncSqliteConversationStore` — SQLite-backed conversation persistence with UNIQUE(session_id, turn_index) constraint
- `PostgresGraphStore` — asyncpg-backed graph store with recursive CTE traversal
- `PostgresConversationStore` — asyncpg-backed conversation store with transaction-safe summary tier persistence
- `SqliteCacheBackend` — SQLite-backed cache with lazy TTL expiration via `time.time()`
- `RedisCacheBackend` — Redis-backed cache using SETEX for TTL, batched `clear()`, `math.ceil` for sub-second TTL safety
- Conversation turn and summary tier serialization helpers in `anchor.storage._serialization`
- `MemoryManager` integration: `conversation_store`, `session_id`, `auto_persist`, `save()`, `load()`, `_is_loading` guard
- `MemoryManager.clear()` now also clears the conversation store when configured
- `ProgressiveSummarizationMemory` tier persistence through `ConversationStore`
- SQLite schema tables: `graph_nodes`, `graph_edges`, `graph_memory_links`, `conversation_turns`, `summary_tiers`, `cache_entries`
- Full protocol method docstrings for `GraphStore`, `AsyncGraphStore`, `ConversationStore`, `AsyncConversationStore` with behavioural contracts
- Comprehensive test suites: graph store (25+ tests), conversation store (14+ tests), persistent memory integration (4 tests), cache backends (16 tests)
- Multi-provider LLM interface (`anchor.llm`) with support for Anthropic, OpenAI, Gemini, Grok, Ollama, OpenRouter, and LiteLLM
- `LLMProvider` protocol and `BaseLLMProvider` ABC with built-in retry and timeout logic
- `create_provider()` factory with `"provider/model"` string format and automatic lazy loading
- `FallbackProvider` for automatic provider failover (fallback only before first stream chunk)
- Provider error hierarchy: `ProviderError`, `RateLimitError`, `ServerError`, `TimeoutError`, `AuthenticationError`, `ModelNotFoundError`, `ContentFilterError`
- Thread-safe provider registry with `threading.Lock`
- Shared `_openai_compat` module for OpenAI/LiteLLM code deduplication
- Anthropic streaming usage tracking (`input_tokens` + `output_tokens`)
- LLM Providers API reference and guide documentation
- Unit tests for `_math.py` (cosine_similarity and clamp functions)
- `MemoryRetrieverAdapter` tests verifying Retriever protocol compliance
- `PipelineExecutionError` wrapping test with diagnostics verification
- Golden path integration test mirroring README usage pattern
- Example: `examples/hybrid_rag.py` -- hybrid RAG pipeline with dense retrieval
- Example: `examples/custom_retriever.py` -- custom Retriever protocol implementation
- Example: `examples/budget_management.py` -- token budget management and overflow handling
- README sections for Priority System (1--10 scale) and Token Budgets

### Changed
- `Agent` constructor: `client` parameter replaced with `llm: LLMProvider` and `fallbacks: list[str]`
- `Role` and `StopReason` enums changed from `(str, Enum)` to `StrEnum` for correct string formatting
- `AgentTool`: removed `to_anthropic_schema()`, `to_openai_schema()`, `to_generic_schema()`; replaced with unified `to_tool_schema() -> ToolSchema`

### Fixed
- 34 pre-existing test failures caused by missing optional dependencies (tiktoken, rank-bm25)
- `FallbackProvider.astream` mid-stream fallback semantics (yields now outside try/except)
- `test_consolidator.py`: eliminated shared mutable state (`_orthogonal_index` dict) by converting to factory function pattern (`make_orthogonal_embed()`)
- `test_graph_memory.py`: updated `link_memory` unknown entity test to expect `KeyError` instead of `ValueError`
- README: fixed retrieval example with runnable `embed_fn` and `ContextItem` creation
- README: updated test count from 961 to 1088

## [0.1.0] - 2026-02-20

### Added
- Core context pipeline with sync/async support (`ContextPipeline`)
- Token-aware sliding window memory (`SlidingWindowMemory`)
- Summary buffer memory with progressive compaction (`SummaryBufferMemory`)
- Memory manager facade unifying conversation and persistent memory (`MemoryManager`)
- Hybrid RAG retrieval: dense, sparse (BM25), and hybrid (RRF) retrievers
- Multi-signal memory retrieval with recency/relevance/importance scoring (`ScoredMemoryRetriever`)
- Provider-agnostic formatting: Anthropic, OpenAI, and generic text formatters
- Anthropic multi-block system formatting with prompt caching support
- Protocol-based extensibility (PEP 544) for all extension points
- Token budget management with per-source allocations and overflow tracking
- Pluggable eviction policies: FIFO, importance-based, and paired (user+assistant)
- Memory decay: Ebbinghaus forgetting curve and linear decay
- Recency scoring: exponential and linear strategies
- Memory consolidation with content-hash dedup and cosine-similarity merging
- Simple graph memory with BFS traversal for entity-relationship tracking
- Memory garbage collection with two-phase expired+decayed pruning
- Memory callback protocol for lifecycle observability
- Pipeline query enrichment with memory context
- Auto-promotion of evicted turns to long-term memory
- In-memory reference implementations for all storage protocols
- JSON file-backed persistent memory store
- CLI with index and query commands (via typer+rich)
- 961 tests with 94% coverage
